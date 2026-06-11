//! One-shot poster-frame extraction for library thumbnails.
//!
//! Deliberately separate from [`Decoder`](super::Decoder): thumbnails are a
//! cold path that needs exactly one frame, so this skips the keyframe index
//! (a full-file demux scan), hardware acceleration, and frame threading.
//! swscale does the colorspace conversion *and* downscale in one pass, which
//! also lifts the YUV420P/NV12/RGBA restriction of the preview pipeline.

use std::path::Path;

use ffmpeg_next::error::EAGAIN;
use ffmpeg_next::format::{self, Pixel};
use ffmpeg_next::media::Type;
use ffmpeg_next::software::scaling;
use ffmpeg_next::util::frame::video::Video;
use ffmpeg_next::{Error as FfmpegError, codec, packet::Packet};
use tracing::debug;

use crate::error::DecodeError;
use crate::video::decoder::ensure_ffmpeg_init;

/// Tightly packed RGBA8 image (no row padding).
#[derive(Debug, Clone)]
pub struct ThumbnailImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Seek a bit into the source and decode one frame, scaled to fit within
/// `max_w` × `max_h` (aspect preserved, never upscaled).
///
/// The target is ~10% into the file (capped at 3s) to skip black/fade-in
/// openings; for speed the first frame after the seek's keyframe is used
/// rather than decoding forward to the exact target.
pub fn video_thumbnail(path: &Path, max_w: u32, max_h: u32) -> Result<ThumbnailImage, DecodeError> {
    ensure_ffmpeg_init()?;
    if max_w == 0 || max_h == 0 {
        return Err(DecodeError::unsupported("zero thumbnail dimensions"));
    }

    let path_str = path
        .to_str()
        .ok_or_else(|| DecodeError::unsupported("path is not valid UTF-8"))?;
    let mut input = format::input(path_str).map_err(DecodeError::Open)?;

    let stream = input
        .streams()
        .best(Type::Video)
        .ok_or_else(|| DecodeError::unsupported("no video stream found"))?;
    let stream_index = stream.index();

    let mut decoder = codec::Context::from_parameters(stream.parameters())
        .map_err(DecodeError::Open)?
        .decoder()
        .video()
        .map_err(DecodeError::Open)?;

    // Best-effort seek in AV_TIME_BASE (microsecond) units; a failure (e.g.
    // tiny or poorly indexed files) just means we decode from the start.
    let duration_us = input.duration().max(0);
    let target_us = (duration_us / 10).min(3_000_000);
    if target_us > 0 && input.seek(target_us, ..target_us).is_err() {
        debug!(path = %path.display(), target_us, "thumbnail seek failed; decoding from start");
    }

    let frame = decode_first_frame(&mut input, &mut decoder, stream_index)?;
    scale_to_rgba(&frame, max_w, max_h)
}

fn decode_first_frame(
    input: &mut format::context::Input,
    decoder: &mut codec::decoder::Video,
    stream_index: usize,
) -> Result<Video, DecodeError> {
    let mut frame = Video::empty();
    let mut demuxer_done = false;
    loop {
        match decoder.receive_frame(&mut frame) {
            Ok(()) => return Ok(frame),
            Err(FfmpegError::Eof) => {
                return Err(DecodeError::unsupported("no decodable video frames"));
            }
            Err(FfmpegError::Other { errno }) if errno == EAGAIN => {
                if demuxer_done {
                    return Err(DecodeError::unsupported("no decodable video frames"));
                }
                let mut packet = Packet::empty();
                loop {
                    match packet.read(input) {
                        Ok(()) if packet.stream() == stream_index => {
                            decoder.send_packet(&packet).map_err(DecodeError::Decode)?;
                            break;
                        }
                        Ok(()) => continue,
                        Err(FfmpegError::Eof) => {
                            demuxer_done = true;
                            decoder.send_eof().map_err(DecodeError::Decode)?;
                            break;
                        }
                        Err(e) => return Err(DecodeError::Io(e)),
                    }
                }
            }
            Err(e) => return Err(DecodeError::Decode(e)),
        }
    }
}

/// Convert + downscale a decoded frame to fit within `max_w` × `max_h`
/// (aspect preserved, never upscaled). Shared with the filmstrip sampler.
pub(crate) fn scale_to_rgba(
    frame: &Video,
    max_w: u32,
    max_h: u32,
) -> Result<ThumbnailImage, DecodeError> {
    let (src_w, src_h) = (frame.width(), frame.height());
    if src_w == 0 || src_h == 0 {
        return Err(DecodeError::unsupported("zero video dimensions"));
    }

    // Fit inside the box without upscaling.
    let scale = f64::from(max_w) / f64::from(src_w);
    let scale = scale.min(f64::from(max_h) / f64::from(src_h)).min(1.0);
    let dst_w = ((f64::from(src_w) * scale).round() as u32).max(1);
    let dst_h = ((f64::from(src_h) * scale).round() as u32).max(1);

    let mut scaler = scaling::Context::get(
        frame.format(),
        src_w,
        src_h,
        Pixel::RGBA,
        dst_w,
        dst_h,
        scaling::Flags::AREA,
    )
    .map_err(DecodeError::Decode)?;

    let mut scaled = Video::empty();
    scaler.run(frame, &mut scaled).map_err(DecodeError::Decode)?;

    // Drop swscale row padding.
    let stride = scaled.stride(0);
    let row_bytes = dst_w as usize * 4;
    let data = scaled.data(0);
    let mut rgba = Vec::with_capacity(row_bytes * dst_h as usize);
    for row in 0..dst_h as usize {
        let start = row * stride;
        rgba.extend_from_slice(&data[start..start + row_bytes]);
    }

    Ok(ThumbnailImage {
        width: dst_w,
        height: dst_h,
        rgba,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn any_video_asset() -> Option<PathBuf> {
        std::fs::read_dir(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets"))
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|e| e == "mp4"))
    }

    #[test]
    fn thumbnail_fits_within_bounds() {
        let Some(path) = any_video_asset() else {
            return;
        };
        let thumb = video_thumbnail(&path, 256, 256).expect("thumbnail");
        assert!(thumb.width <= 256 && thumb.height <= 256);
        assert!(thumb.width > 0 && thumb.height > 0);
        assert_eq!(thumb.rgba.len(), (thumb.width * thumb.height * 4) as usize);
        // Not fully transparent: every pixel should have alpha 255.
        assert!(thumb.rgba.chunks_exact(4).all(|px| px[3] == 255));
    }

    #[test]
    fn thumbnail_preserves_aspect() {
        let Some(path) = any_video_asset() else {
            return;
        };
        let thumb = video_thumbnail(&path, 200, 200).expect("thumbnail");
        // All workspace assets are wider than tall.
        assert!(thumb.width >= thumb.height);
        assert_eq!(thumb.width, 200);
    }

    #[test]
    fn zero_box_is_rejected() {
        let Some(path) = any_video_asset() else {
            return;
        };
        assert!(matches!(
            video_thumbnail(&path, 0, 100),
            Err(DecodeError::Unsupported { .. })
        ));
    }
}
