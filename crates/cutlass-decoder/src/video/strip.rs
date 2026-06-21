//! Multi-frame sampling for timeline filmstrips.
//!
//! Like [`video_thumbnail`](super::thumbnail::video_thumbnail) this is a cold
//! path kept separate from the preview [`Decoder`](super::Decoder) (no
//! keyframe index, no hwaccel, no frame threading), but it amortizes one
//! demuxer open across many sample times: targets are visited in ascending
//! order, re-seeking only when the next target is behind the current decode
//! position or too far ahead to roll forward to. Unlike the poster-frame
//! path it decodes *forward to the requested timestamp*, so neighbouring
//! tiles inside one GOP show distinct frames instead of the shared keyframe.

use std::path::Path;

use ffmpeg_next::error::EAGAIN;
use ffmpeg_next::format;
use ffmpeg_next::media::Type;
use ffmpeg_next::util::frame::video::Video;
use ffmpeg_next::{Error as FfmpegError, codec, packet::Packet};
use tracing::debug;

use crate::error::DecodeError;
use crate::video::decoder::ensure_ffmpeg_init;
use crate::video::thumbnail::{ThumbnailImage, scale_to_rgba};

/// Targets further ahead than this roll a fresh seek instead of decoding
/// forward through the gap (≈ one long GOP).
const SEEK_AHEAD_S: f64 = 3.0;

/// Cap on frames decoded toward a single target, so a pathological GOP can't
/// stall the strip worker (≈ 2s of 60fps video).
const MAX_DECODE_STEPS: usize = 120;

/// Decode one frame per entry of `times_s` (seconds into the source), scaled
/// to fit within `max_w` × `max_h` (aspect preserved, never upscaled).
///
/// `on_frame(index, image)` fires as each frame lands, with `index` referring
/// to the caller's `times_s` order — callers can deliver tiles incrementally.
/// Times past the end of the stream resolve to the last decodable frame;
/// targets that produce no frame at all (e.g. corrupt tail) are skipped.
pub fn video_strip(
    path: &Path,
    times_s: &[f64],
    max_w: u32,
    max_h: u32,
    on_frame: &mut dyn FnMut(usize, ThumbnailImage),
) -> Result<(), DecodeError> {
    ensure_ffmpeg_init()?;
    if max_w == 0 || max_h == 0 {
        return Err(DecodeError::unsupported("zero strip dimensions"));
    }
    if times_s.is_empty() {
        return Ok(());
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
    let time_base = stream.time_base();
    let tb = f64::from(time_base.numerator()) / f64::from(time_base.denominator()).max(1.0);

    let mut decoder = codec::Context::from_parameters(stream.parameters())
        .map_err(DecodeError::Open)?
        .decoder()
        .video()
        .map_err(DecodeError::Open)?;

    // Don't chase targets past the end of the stream; the decode loop would
    // walk the whole tail GOP just to land on the last frame anyway.
    let duration_s = input.duration().max(0) as f64 / 1e6;
    let max_target = if duration_s > 0.0 {
        (duration_s - 0.05).max(0.0)
    } else {
        f64::MAX
    };

    // Visit ascending so forward decode rolls from target to target; deliver
    // under the caller's original index.
    let mut order: Vec<(usize, f64)> = times_s.iter().copied().enumerate().collect();
    order.sort_by(|a, b| a.1.total_cmp(&b.1));

    let mut frame = Video::empty();
    let mut have_frame = false;
    // Whether `frame` ever held a decoded picture. A seek resets `have_frame`
    // but not the buffer, so this gates the stale-frame fallback below.
    let mut ever_decoded = false;
    let mut demuxer_done = false;

    for &(index, target) in &order {
        let target = target.clamp(0.0, max_target);

        let current = have_frame.then(|| frame_time_s(&frame, tb)).flatten();
        let need_seek = match current {
            Some(t) => target < t || target - t > SEEK_AHEAD_S,
            None => true,
        };
        if need_seek {
            let target_us = (target * 1e6) as i64;
            if input.seek(target_us, ..target_us).is_err() {
                debug!(path = %path.display(), target_us, "strip seek failed; decoding from current position");
            }
            decoder.flush();
            demuxer_done = false;
            have_frame = false;
        }

        let mut steps = 0;
        loop {
            // A frame without a timestamp can't be compared to the target;
            // accept it as-is rather than decoding blind.
            match have_frame.then(|| frame_time_s(&frame, tb)).flatten() {
                Some(t) if t >= target => break,
                None if have_frame => break,
                _ => {}
            }
            if steps >= MAX_DECODE_STEPS {
                break;
            }
            if next_frame(
                &mut input,
                &mut decoder,
                stream_index,
                &mut demuxer_done,
                &mut frame,
            )? {
                have_frame = true;
                ever_decoded = true;
                steps += 1;
            } else {
                break; // end of stream: keep the last decoded frame
            }
        }

        if have_frame {
            on_frame(index, scale_to_rgba(&frame, max_w, max_h)?);
        } else if ever_decoded {
            // A seek reset the decoder and the stream had nothing left —
            // single-picture sources (PNG/JPEG/WebP stills) and corrupt
            // tails land here. Deliver the last decodable frame, keeping
            // the documented "targets past the end resolve to the last
            // frame" promise.
            debug!(path = %path.display(), target, "strip target reuses last decodable frame");
            on_frame(index, scale_to_rgba(&frame, max_w, max_h)?);
        } else {
            debug!(path = %path.display(), target, "strip target produced no frame");
        }
    }

    Ok(())
}

fn frame_time_s(frame: &Video, tb: f64) -> Option<f64> {
    frame.timestamp().or(frame.pts()).map(|ts| ts as f64 * tb)
}

/// Pump packets until the decoder yields one frame into `frame`.
/// `Ok(false)` means the stream is exhausted — and leaves `frame` untouched:
/// `avcodec_receive_frame` unrefs its output before failing, so decoding
/// into `frame` directly would wipe the last good picture exactly when the
/// caller needs it (targets past the end, single-picture stills).
fn next_frame(
    input: &mut format::context::Input,
    decoder: &mut codec::decoder::Video,
    stream_index: usize,
    demuxer_done: &mut bool,
    frame: &mut Video,
) -> Result<bool, DecodeError> {
    let mut fresh = Video::empty();
    loop {
        match decoder.receive_frame(&mut fresh) {
            Ok(()) => {
                *frame = fresh;
                return Ok(true);
            }
            Err(FfmpegError::Eof) => return Ok(false),
            Err(FfmpegError::Other { errno }) if errno == EAGAIN => {
                if *demuxer_done {
                    return Ok(false);
                }
                loop {
                    // Fresh packet per read so skipped (non-video) packets unref
                    // on drop; `Packet::read` never unrefs and reusing one packet
                    // would leak every packet's payload across the demux.
                    let mut packet = Packet::empty();
                    match packet.read(input) {
                        Ok(()) if packet.stream() == stream_index => {
                            decoder.send_packet(&packet).map_err(DecodeError::Decode)?;
                            break;
                        }
                        Ok(()) => continue,
                        Err(FfmpegError::Eof) => {
                            *demuxer_done = true;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn any_video_asset() -> Option<PathBuf> {
        std::fs::read_dir(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../local-assets/assets"),
        )
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|e| e == "mp4"))
    }

    #[test]
    fn strip_on_still_image_repeats_the_picture_for_every_target() {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../local-assets/assets/texture.png");
        if !path.exists() {
            return;
        }
        // Spans the default 5s still placement, including targets past the
        // forward-roll window that force (failing) seeks on a one-frame source.
        let times = [0.0, 1.0, 2.5, 4.0, 5.0];
        let mut seen: Vec<usize> = Vec::new();
        video_strip(&path, &times, 128, 128, &mut |i, img| {
            assert!(img.width > 0 && img.height > 0);
            assert_eq!(img.rgba.len(), (img.width * img.height * 4) as usize);
            seen.push(i);
        })
        .expect("strip on still");

        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2, 3, 4], "every tile shows the still");
    }

    #[test]
    fn strip_delivers_every_target_in_caller_order() {
        let Some(path) = any_video_asset() else {
            return;
        };
        // Deliberately unsorted: delivery must use the caller's indices.
        let times = [2.0, 0.0, 1.0];
        let mut seen: Vec<(usize, u32, u32)> = Vec::new();
        video_strip(&path, &times, 256, 128, &mut |i, img| {
            assert_eq!(img.rgba.len(), (img.width * img.height * 4) as usize);
            seen.push((i, img.width, img.height));
        })
        .expect("strip");

        let mut indices: Vec<usize> = seen.iter().map(|(i, ..)| *i).collect();
        indices.sort_unstable();
        assert_eq!(indices, vec![0, 1, 2]);
        assert!(
            seen.iter()
                .all(|&(_, w, h)| w <= 256 && h <= 128 && w > 0 && h > 0)
        );
    }

    #[test]
    fn strip_clamps_targets_past_the_end() {
        let Some(path) = any_video_asset() else {
            return;
        };
        let mut count = 0;
        video_strip(&path, &[1e9], 128, 128, &mut |_, _| count += 1).expect("strip");
        assert_eq!(
            count, 1,
            "an out-of-range target resolves to the last frame"
        );
    }

    #[test]
    fn zero_box_is_rejected() {
        let Some(path) = any_video_asset() else {
            return;
        };
        assert!(matches!(
            video_strip(&path, &[0.0], 0, 64, &mut |_, _| {}),
            Err(DecodeError::Unsupported { .. })
        ));
    }

    #[test]
    fn empty_times_is_a_noop() {
        // Returns before the file is even opened, so no asset is needed.
        video_strip(Path::new("/nonexistent.mp4"), &[], 64, 64, &mut |_, _| {
            panic!("no frames expected")
        })
        .expect("empty request is a no-op");
    }
}
