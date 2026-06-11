use std::sync::OnceLock;

use ffmpeg_next::error::EAGAIN;
use ffmpeg_next::format::context::Output;
use ffmpeg_next::util::frame::video::Video as VideoFrame;
use ffmpeg_next::{Codec, Error as FfmpegError, Packet, Rational, codec, encoder};

use crate::error::EncodeError;

static FFMPEG_INIT: OnceLock<Result<(), FfmpegError>> = OnceLock::new();

pub(crate) fn ensure_ffmpeg_init() -> Result<(), EncodeError> {
    match FFMPEG_INIT.get_or_init(ffmpeg_next::init) {
        Ok(()) => Ok(()),
        Err(e) => Err(EncodeError::Open(*e)),
    }
}

/// Write out every packet the encoder can currently produce. Works for any
/// stream kind — video and audio encoders both deref to the raw encoder.
pub(crate) fn drain_encoder(
    encoder: &mut encoder::Encoder,
    octx: &mut Output,
    ost_index: usize,
    enc_tb: Rational,
    ost_tb: Rational,
) -> Result<(), EncodeError> {
    let mut packet = Packet::empty();
    loop {
        match encoder.receive_packet(&mut packet) {
            Ok(()) => {
                packet.set_stream(ost_index);
                packet.rescale_ts(enc_tb, ost_tb);
                packet.write_interleaved(octx).map_err(EncodeError::Io)?;
            }
            Err(FfmpegError::Eof) => return Ok(()),
            Err(e) if is_eagain(&e) => return Ok(()),
            Err(e) => return Err(EncodeError::Encode(e)),
        }
    }
}

/// Pick an H.264 encoder, preferring a hardware one that accepts software input
/// frames; always falls back to libx264 / the generic H.264 encoder.
pub(crate) fn find_h264_encoder(hardware: bool) -> Option<Codec> {
    if hardware {
        #[cfg(target_os = "macos")]
        if let Some(c) = encoder::find_by_name("h264_videotoolbox") {
            return Some(c);
        }
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        if let Some(c) = encoder::find_by_name("h264_nvenc") {
            return Some(c);
        }
    }
    encoder::find_by_name("libx264").or_else(|| encoder::find(codec::Id::H264))
}

pub(crate) fn is_eagain(e: &FfmpegError) -> bool {
    matches!(e, FfmpegError::Other { errno } if *errno == EAGAIN)
}

/// Target dimensions: preserve aspect ratio, never upscale, round to even.
pub fn scaled_dims(src_w: u32, src_h: u32, target_h: u32) -> (u32, u32) {
    if src_h == 0 {
        return (src_w.max(2) & !1, src_h);
    }
    let h = target_h.clamp(2, src_h);
    let w = ((src_w as u64 * h as u64) / src_h as u64) as u32;
    (w.max(2) & !1, h.max(2) & !1)
}

/// Fill a YUV420P frame from tight plane buffers (GPU readback layout).
pub(crate) fn fill_yuv420p_frame(
    frame: &mut VideoFrame,
    width: u32,
    height: u32,
    y: &[u8],
    u: &[u8],
    v: &[u8],
) -> Result<(), EncodeError> {
    let w = width as usize;
    let h = height as usize;
    let y_need = w * h;
    let uv_need = (w / 2) * (h / 2);
    if y.len() != y_need || u.len() != uv_need || v.len() != uv_need {
        return Err(EncodeError::unsupported(format!(
            "yuv plane sizes mismatch for {width}x{height}"
        )));
    }

    let y_stride = frame.stride(0);
    let u_stride = frame.stride(1);
    let v_stride = frame.stride(2);
    for row in 0..h {
        let dst = &mut frame.data_mut(0)[row * y_stride..row * y_stride + w];
        let src = &y[row * w..row * w + w];
        dst.copy_from_slice(src);
    }
    let uv_h = h / 2;
    let uv_w = w / 2;
    for row in 0..uv_h {
        let dst = &mut frame.data_mut(1)[row * u_stride..row * u_stride + uv_w];
        let src = &u[row * uv_w..row * uv_w + uv_w];
        dst.copy_from_slice(src);
    }
    for row in 0..uv_h {
        let dst = &mut frame.data_mut(2)[row * v_stride..row * v_stride + uv_w];
        let src = &v[row * uv_w..row * uv_w + uv_w];
        dst.copy_from_slice(src);
    }
    Ok(())
}

/// Fill a reusable RGBA frame from a row-major RGBA8 buffer.
pub(crate) fn fill_rgba_frame(
    frame: &mut VideoFrame,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> Result<(), EncodeError> {
    let expected = (width as usize) * (height as usize) * 4;
    if rgba.len() != expected {
        return Err(EncodeError::unsupported(format!(
            "rgba buffer is {} bytes, expected {expected}",
            rgba.len()
        )));
    }

    let row_bytes = (width as usize) * 4;
    let stride = frame.stride(0);
    let data = frame.data_mut(0);
    for y in 0..height as usize {
        let dst = &mut data[y * stride..y * stride + row_bytes];
        let src = &rgba[y * row_bytes..y * row_bytes + row_bytes];
        dst.copy_from_slice(src);
    }
    Ok(())
}
