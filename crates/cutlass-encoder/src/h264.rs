use std::collections::HashSet;
use std::sync::OnceLock;

use ffmpeg_next::error::EAGAIN;
use ffmpeg_next::format::Pixel;
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

/// A chosen H.264 encoder together with the **software** pixel format its frames
/// must be fed in. Always a CPU layout we can produce with swscale — never a
/// hardware-surface format — so the caller can blindly upload composited frames.
#[derive(Clone, Copy)]
pub(crate) struct H264Encoder {
    pub codec: Codec,
    pub pixel: Pixel,
}

/// Software input layouts we know how to fill/convert to, most-preferred first.
/// `YUV420P` is the planar layout the compositor reads back; `NV12` is what
/// several OS encoders (notably Windows Media Foundation) want instead.
const SOFTWARE_INPUT_FORMATS: [Pixel; 2] = [Pixel::YUV420P, Pixel::NV12];

/// Hardware H.264 encoders to try when the caller opts into hardware encode,
/// best quality / throughput first.
#[cfg(target_os = "macos")]
const HARDWARE_H264: &[&str] = &["h264_videotoolbox"];
#[cfg(target_os = "windows")]
const HARDWARE_H264: &[&str] = &["h264_nvenc", "h264_amf", "h264_qsv", "h264_mf"];
#[cfg(target_os = "linux")]
const HARDWARE_H264: &[&str] = &["h264_nvenc", "h264_vaapi", "h264_qsv"];
#[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
const HARDWARE_H264: &[&str] = &[];

/// Encoders to fall back to when FFmpeg has **no** software H.264 (common on
/// LGPL Windows builds that drop GPL libx264). Ordered for *openability* on an
/// arbitrary machine rather than raw quality: on Windows, Media Foundation
/// opens on any GPU — or on the CPU — whereas vendor encoders are present in
/// the build yet fail to open without their specific hardware.
#[cfg(target_os = "windows")]
const FALLBACK_H264: &[&str] = &["h264_mf", "h264_nvenc", "h264_amf", "h264_qsv"];
#[cfg(not(target_os = "windows"))]
const FALLBACK_H264: &[&str] = HARDWARE_H264;

/// The most-preferred software input format `codec` advertises, or `None` if it
/// only takes hardware surfaces (e.g. `h264_d3d12va` accepts just `d3d12`,
/// `h264_qsv` just `qsv`). Encoders that advertise no pixel-format list at all
/// are assumed to take the classic planar default.
fn software_input_format(codec: Codec) -> Option<Pixel> {
    let video = codec.video().ok()?;
    match video.formats() {
        Some(formats) => {
            let supported: Vec<Pixel> = formats.collect();
            SOFTWARE_INPUT_FORMATS
                .into_iter()
                .find(|fmt| supported.contains(fmt))
        }
        None => Some(Pixel::YUV420P),
    }
}

fn as_software_encoder(codec: Codec) -> Option<H264Encoder> {
    software_input_format(codec).map(|pixel| H264Encoder { codec, pixel })
}

fn named_software_encoder(name: &str) -> Option<H264Encoder> {
    encoder::find_by_name(name).and_then(as_software_encoder)
}

/// Ordered H.264 encoders to try. Hardware-first when requested, then software,
/// then CPU-frame-capable hardware fallbacks, then the build default.
///
/// A hardware-surface-only encoder is never included — those reject the
/// pipeline's software frames at open time. Encoders that advertise CPU layouts
/// yet fail to open (NVENC without CUDA, VAAPI without a render node, …) are
/// skipped at open time via [`open_first_h264_encoder`].
pub(crate) fn h264_encoder_candidates(hardware: bool) -> Vec<H264Encoder> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut push = |enc: H264Encoder| {
        if seen.insert(enc.codec.name().to_string()) {
            out.push(enc);
        }
    };

    if hardware {
        for &name in HARDWARE_H264 {
            if let Some(enc) = named_software_encoder(name) {
                push(enc);
            }
        }
    }

    for &name in &["libx264", "libopenh264"] {
        if let Some(enc) = named_software_encoder(name) {
            push(enc);
        }
    }

    for &name in FALLBACK_H264 {
        if let Some(enc) = named_software_encoder(name) {
            push(enc);
        }
    }

    if let Some(enc) = encoder::find(codec::Id::H264).and_then(as_software_encoder) {
        push(enc);
    }

    out
}

/// Try `try_open` on each candidate in order; the first successful open wins.
pub(crate) fn open_first_h264_encoder<F, T>(
    hardware: bool,
    mut try_open: F,
) -> Result<(H264Encoder, T), EncodeError>
where
    F: FnMut(H264Encoder) -> Result<T, EncodeError>,
{
    let candidates = h264_encoder_candidates(hardware);
    if candidates.is_empty() {
        return Err(EncodeError::unsupported("no H.264 encoder available"));
    }

    let mut last = None;
    for chosen in candidates {
        match try_open(chosen) {
            Ok(result) => return Ok((chosen, result)),
            Err(e) => last = Some(e),
        }
    }
    Err(last.unwrap_or_else(|| EncodeError::unsupported("no H.264 encoder available")))
}

pub(crate) fn is_eagain(e: &FfmpegError) -> bool {
    matches!(e, FfmpegError::Other { errno } if *errno == EAGAIN)
}

/// Target dimensions: preserve aspect ratio, never upscale, round to even.
/// Proxy-only policy — export sizing honors the user's pick exactly
/// (`export_dims` in cutlass-engine).
pub(crate) fn scaled_dims(src_w: u32, src_h: u32, target_h: u32) -> (u32, u32) {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The core regression guard: on a build whose only H.264 encoders take
    /// hardware surfaces (e.g. Windows LGPL FFmpeg with just `h264_d3d12va`),
    /// we must never hand the software-frame pipeline such an encoder — it
    /// rejects `yuv420p` at open time and surfaces as "failed to open media".
    #[test]
    fn selected_encoder_always_takes_software_frames() {
        ensure_ffmpeg_init().expect("ffmpeg init");
        for hardware in [false, true] {
            for enc in h264_encoder_candidates(hardware) {
                assert!(
                    SOFTWARE_INPUT_FORMATS.contains(&enc.pixel),
                    "{} selected non-software pixel {:?}",
                    enc.codec.name(),
                    enc.pixel,
                );
                assert_eq!(
                    software_input_format(enc.codec),
                    Some(enc.pixel),
                    "{} pixel choice disagrees with advertised formats",
                    enc.codec.name(),
                );
            }
        }
    }

    /// When the GPL software encoder is present (CI/dev), the software path must
    /// prefer it at constant quality — never silently degrade to a hardware one.
    #[test]
    fn software_path_prefers_libx264_when_present() {
        ensure_ffmpeg_init().expect("ffmpeg init");
        if encoder::find_by_name("libx264").is_some() {
            let enc = h264_encoder_candidates(false)
                .into_iter()
                .next()
                .expect("an H.264 encoder");
            assert_eq!(enc.codec.name(), "libx264");
            assert_eq!(enc.pixel, Pixel::YUV420P);
        }
    }

    #[test]
    fn hardware_candidates_include_software_fallback() {
        ensure_ffmpeg_init().expect("ffmpeg init");
        let candidates = h264_encoder_candidates(true);
        assert!(!candidates.is_empty());
        if encoder::find_by_name("libx264").is_some() {
            assert!(
                candidates.iter().any(|c| c.codec.name() == "libx264"),
                "hardware path must fall back to libx264 when HW encoders fail to open"
            );
        }
    }
}
