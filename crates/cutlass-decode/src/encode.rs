//! Proxy builder: transcode a source into a 1080p **all-intra H.264** file.
//!
//! Why all-intra: every frame becomes its own keyframe, so a later seek lands
//! exactly on the target and decodes a single frame — turning the long-GOP cold
//! seek (0.4–1.6 s on 4K) into a flat ~9 ms (software) read. See
//! `docs/proxy-cache/research.md` for the measurements behind this.
//!
//! The build pass is deliberately **software frame-threaded decode → hardware
//! encode**: a single SW decoder already saturates all cores (≈7.7× realtime),
//! and HW encode is then almost free, so this is the fastest single pipeline
//! while leaving the HW *decoder* free for live preview.

use std::path::Path;

use ffmpeg_next::error::EAGAIN;
use ffmpeg_next::format::{self, context::Output};
use ffmpeg_next::media::Type as MediaType;
use ffmpeg_next::software::scaling;
use ffmpeg_next::util::format::Pixel;
use ffmpeg_next::util::frame::video::Video as VideoFrame;
use ffmpeg_next::{
    Codec, Dictionary, Error as FfmpegError, Packet, Rational, codec,
    decoder::Video as VideoDecoder, encoder, encoder::Video as VideoEncoder,
};
use tracing::debug;

use crate::decoder::ensure_ffmpeg_init;
use crate::error::DecodeError;
use crate::hwaccel::{self, HwAccel};

/// How to build a proxy file.
#[derive(Debug, Clone, Copy)]
pub struct ProxyConfig {
    /// Target height in pixels; width is derived from the source aspect ratio
    /// (rounded down to even). The source height caps it — proxies never upscale.
    pub target_height: u32,
    /// Constant-quality level for the software encoder (libx264 CRF, 0–51,
    /// lower = better quality / bigger file). ~18 is visually near-transparent.
    /// Used when a CRF-capable encoder is selected (the default path).
    pub quality: u8,
    /// Fallback target bitrate (bits/sec), used only for encoders without a CRF
    /// mode (e.g. hardware VideoToolbox when `hardware` is set).
    pub bitrate: usize,
    /// Prefer a hardware encoder (VideoToolbox / NVENC). Faster, but lower
    /// quality per bit — VideoToolbox's H.264 grains noticeably even at high
    /// bitrate. Off by default: software libx264 at constant quality (`quality`)
    /// gives clean all-intra proxies, which matters more than build speed here.
    pub hardware: bool,
}

impl Default for ProxyConfig {
    fn default() -> Self {
        // 1080p preview tier, constant-quality software encode. All-intra H.264
        // makes every frame a keyframe (great for seeking, poor for bit
        // efficiency), so quality-targeted libx264 keeps it clean without
        // guessing a bitrate. The 50 GiB disk budget evicts the overflow.
        Self {
            target_height: 1080,
            quality: 18,
            bitrate: 60_000_000,
            hardware: true,
        }
    }
}

/// Per-lane build knobs that the scheduler varies across concurrent builds.
///
/// The two interesting axes (see `docs/proxy-cache/research.md`):
/// - `decode`: a **software** lane is the fastest single pipeline (one SW decoder
///   frame-threads across all cores), while a **hardware** lane decodes on the GPU
///   block — slower per frame but it frees the CPU so a second concurrent import
///   doesn't fight the SW lane for cores.
/// - `decode_threads`: cap SW frame-threading so two lanes don't oversubscribe.
#[derive(Debug, Clone, Copy)]
pub struct ProxyBuildOptions {
    /// Decode acceleration for reading the *source* during the build.
    pub decode: HwAccel,
    /// Decoder frame-thread count (`0` = all cores). Ignored for HW decode.
    pub decode_threads: u32,
}

impl Default for ProxyBuildOptions {
    fn default() -> Self {
        // The fast single pipeline: SW frame-threaded decode on all cores.
        Self {
            decode: HwAccel::None,
            decode_threads: 0,
        }
    }
}

/// Result of a completed proxy build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProxyStats {
    pub frames: u64,
    pub width: u32,
    pub height: u32,
}

/// Transcode `source` into an all-intra H.264 proxy at `output`.
///
/// Decodes in software (frame-threaded) and encodes with a hardware H.264
/// encoder when available. The output is a constant-frame-rate, GOP-1 file whose
/// every frame is a keyframe. Equivalent to [`build_proxy_with`] with the default
/// (software, all-cores) options and no progress reporting.
pub fn build_proxy(
    source: &Path,
    output: &Path,
    config: ProxyConfig,
) -> Result<ProxyStats, DecodeError> {
    build_proxy_with(source, output, config, ProxyBuildOptions::default(), None)
}

/// Like [`build_proxy`], but with per-lane decode options and an optional
/// progress callback.
///
/// `progress`, when given, is invoked periodically with the number of frames
/// encoded so far — the scheduler maps that to a percentage for the UI. It runs
/// on the calling (worker) thread, between packets, so it must not block.
pub fn build_proxy_with(
    source: &Path,
    output: &Path,
    config: ProxyConfig,
    opts: ProxyBuildOptions,
    mut progress: Option<&mut dyn FnMut(u64)>,
) -> Result<ProxyStats, DecodeError> {
    ensure_ffmpeg_init()?;

    let src = source
        .to_str()
        .ok_or_else(|| DecodeError::unsupported("source path not valid UTF-8"))?;
    let mut ictx = format::input(src).map_err(DecodeError::Open)?;

    let (stream_index, src_fps) = {
        let stream = ictx
            .streams()
            .best(MediaType::Video)
            .ok_or_else(|| DecodeError::unsupported("no video stream found"))?;
        (stream.index(), stream.avg_frame_rate())
    };

    let mut dec_ctx = {
        let params = ictx.stream(stream_index).unwrap().parameters();
        codec::context::Context::from_parameters(params).map_err(DecodeError::Open)?
    };
    // Frame-threaded decode. `count: 0` lets ffmpeg use all cores; a lane may cap
    // it so two concurrent SW builds don't oversubscribe.
    dec_ctx.set_threading(codec::threading::Config {
        kind: codec::threading::Type::Frame,
        count: opts.decode_threads as usize,
    });
    // Optional hardware decode (offloads CPU for a concurrent lane). For `None`
    // this is a no-op and decode stays on the CPU.
    let _active = hwaccel::attach(&mut dec_ctx, opts.decode)?;
    let mut decoder = dec_ctx.decoder().video().map_err(DecodeError::Open)?;

    let src_w = decoder.width();
    let src_h = decoder.height();
    if src_w == 0 || src_h == 0 {
        return Err(DecodeError::unsupported("source has zero dimensions"));
    }
    let (dst_w, dst_h) = scaled_dims(src_w, src_h, config.target_height);

    let fps = if src_fps.numerator() > 0 && src_fps.denominator() > 0 {
        src_fps
    } else {
        Rational::new(30, 1)
    };
    // All-intra proxy: re-time on a clean 1/fps timeline, monotonic per frame.
    let enc_tb = fps.invert();

    let codec = find_h264_encoder(config.hardware)
        .ok_or_else(|| DecodeError::unsupported("no H.264 encoder available"))?;
    // libx264 supports true constant-quality (CRF); hardware encoders don't, so
    // they fall back to the target bitrate.
    let use_crf = codec.name() == "libx264";

    let mut octx = format::output(&output).map_err(DecodeError::Open)?;
    let global_header = octx.format().flags().contains(format::Flags::GLOBAL_HEADER);

    // Build + open the encoder first (it doesn't borrow octx), then register the
    // output stream and copy its parameters.
    let mut enc = codec::context::Context::new_with_codec(codec)
        .encoder()
        .video()
        .map_err(DecodeError::Open)?;
    enc.set_width(dst_w);
    enc.set_height(dst_h);
    enc.set_format(Pixel::YUV420P);
    // Be explicit so encoders don't warn / guess about limited-vs-full range.
    enc.set_color_range(ffmpeg_next::color::Range::MPEG);
    enc.set_frame_rate(Some(fps));
    enc.set_time_base(enc_tb);
    enc.set_gop(1);
    enc.set_max_b_frames(0);
    // In CRF mode the encoder picks the bitrate; setting one would force ABR and
    // defeat constant quality. Only bitrate-driven (hardware) encoders need it.
    if !use_crf {
        enc.set_bit_rate(config.bitrate);
    }
    if global_header {
        enc.set_flags(codec::Flags::GLOBAL_HEADER);
    }

    // Per-encoder open options: constant quality + a balanced speed preset for
    // libx264; nothing extra for hardware encoders.
    let mut enc_opts = Dictionary::new();
    if use_crf {
        let crf = config.quality.to_string();
        enc_opts.set("crf", &crf);
        enc_opts.set("preset", "faster");
    }
    let mut encoder = enc.open_with(enc_opts).map_err(DecodeError::Open)?;

    let ost_index = {
        let mut ost = octx.add_stream(codec).map_err(DecodeError::Open)?;
        ost.set_parameters(&encoder);
        ost.index()
    };

    octx.write_header().map_err(DecodeError::Io)?;
    let ost_tb = octx.stream(ost_index).unwrap().time_base();

    let mut scaler: Option<scaling::Context> = None;
    let mut frame_index: i64 = 0;

    for (stream, packet) in ictx.packets() {
        if stream.index() != stream_index {
            continue;
        }
        decoder.send_packet(&packet).map_err(DecodeError::Decode)?;
        drain_decoder(
            &mut decoder,
            &mut scaler,
            (dst_w, dst_h),
            &mut encoder,
            &mut octx,
            ost_index,
            enc_tb,
            ost_tb,
            &mut frame_index,
        )?;
        if let Some(cb) = progress.as_deref_mut() {
            cb(frame_index as u64);
        }
    }

    // Flush decoder, then encoder.
    decoder.send_eof().map_err(DecodeError::Decode)?;
    drain_decoder(
        &mut decoder,
        &mut scaler,
        (dst_w, dst_h),
        &mut encoder,
        &mut octx,
        ost_index,
        enc_tb,
        ost_tb,
        &mut frame_index,
    )?;
    encoder.send_eof().map_err(DecodeError::Decode)?;
    drain_encoder(&mut encoder, &mut octx, ost_index, enc_tb, ost_tb)?;

    octx.write_trailer().map_err(DecodeError::Io)?;

    debug!(frames = frame_index, dst_w, dst_h, "built proxy");
    Ok(ProxyStats {
        frames: frame_index as u64,
        width: dst_w,
        height: dst_h,
    })
}

/// Drain all frames the decoder can currently produce, scaling and encoding each.
#[allow(clippy::too_many_arguments)]
fn drain_decoder(
    decoder: &mut VideoDecoder,
    scaler: &mut Option<scaling::Context>,
    dst: (u32, u32),
    encoder: &mut VideoEncoder,
    octx: &mut Output,
    ost_index: usize,
    enc_tb: Rational,
    ost_tb: Rational,
    frame_index: &mut i64,
) -> Result<(), DecodeError> {
    let mut decoded = VideoFrame::empty();
    loop {
        match decoder.receive_frame(&mut decoded) {
            Ok(()) => {
                // A hardware lane yields GPU surfaces; copy them to CPU memory
                // (NV12) so the software scaler can read them. SW lanes skip this.
                let mut sw = VideoFrame::empty();
                let src: &VideoFrame = if hwaccel::is_hardware_pixel_format(decoded.format()) {
                    hwaccel::transfer_to_cpu(&decoded, &mut sw)?;
                    &sw
                } else {
                    &decoded
                };

                // Lazily build the scaler from the first frame's real format/dims.
                if scaler.is_none() {
                    *scaler = Some(
                        scaling::Context::get(
                            src.format(),
                            src.width(),
                            src.height(),
                            Pixel::YUV420P,
                            dst.0,
                            dst.1,
                            scaling::Flags::BILINEAR,
                        )
                        .map_err(DecodeError::Decode)?,
                    );
                }
                let sc = scaler.as_mut().unwrap();

                // Fresh output buffer per frame: the HW encoder may still hold a
                // ref to the previous one, so we must not scale in place over it.
                let mut scaled = VideoFrame::empty();
                sc.run(src, &mut scaled).map_err(DecodeError::Decode)?;
                scaled.set_pts(Some(*frame_index));
                encoder.send_frame(&scaled).map_err(DecodeError::Decode)?;
                *frame_index += 1;

                drain_encoder(encoder, octx, ost_index, enc_tb, ost_tb)?;
            }
            Err(FfmpegError::Eof) => return Ok(()),
            Err(e) if is_eagain(&e) => return Ok(()),
            Err(e) => return Err(DecodeError::Decode(e)),
        }
    }
}

/// Write out every packet the encoder can currently produce.
pub(crate) fn drain_encoder(
    encoder: &mut VideoEncoder,
    octx: &mut Output,
    ost_index: usize,
    enc_tb: Rational,
    ost_tb: Rational,
) -> Result<(), DecodeError> {
    let mut packet = Packet::empty();
    loop {
        match encoder.receive_packet(&mut packet) {
            Ok(()) => {
                packet.set_stream(ost_index);
                packet.rescale_ts(enc_tb, ost_tb);
                packet.write_interleaved(octx).map_err(DecodeError::Io)?;
            }
            Err(FfmpegError::Eof) => return Ok(()),
            Err(e) if is_eagain(&e) => return Ok(()),
            Err(e) => return Err(DecodeError::Decode(e)),
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

/// Target dimensions: preserve aspect ratio, never upscale, round to even.
fn scaled_dims(src_w: u32, src_h: u32, target_h: u32) -> (u32, u32) {
    if src_h == 0 {
        return (src_w.max(2) & !1, src_h);
    }
    let h = target_h.clamp(2, src_h);
    let w = ((src_w as u64 * h as u64) / src_h as u64) as u32;
    (w.max(2) & !1, h.max(2) & !1)
}

pub(crate) fn is_eagain(e: &FfmpegError) -> bool {
    matches!(e, FfmpegError::Other { errno } if *errno == EAGAIN)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::Duration;

    use crate::{DecodeOptions, Decoder, HwAccel};

    fn sibling_main_asset(name: &str) -> Option<PathBuf> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../cutlass-main/crates/decoder/tests/assets")
            .join(name);
        path.exists().then_some(path)
    }

    #[test]
    fn proxy_builds_all_intra_and_is_seekable() {
        let Some(src) = sibling_main_asset("testsrc_h264.mp4") else {
            return;
        };
        let out = std::env::temp_dir().join("cutlass_proxy_smoke.mp4");
        let _ = std::fs::remove_file(&out);

        let stats = build_proxy(
            &src,
            &out,
            ProxyConfig {
                target_height: 180,
                quality: 23,
                bitrate: 2_000_000,
                hardware: true,
            },
        )
        .expect("build proxy");
        assert!(stats.frames > 0, "no frames encoded");
        assert!(out.exists(), "proxy file missing");

        // The proxy must decode and seek: open it (software) and land a frame.
        let mut dec = Decoder::open_with(&out, DecodeOptions::default().hw_accel(HwAccel::None))
            .expect("open proxy");
        let frame = dec
            .seek_to_frame(Duration::from_millis(500))
            .expect("seek proxy")
            .expect("frame after seek");
        assert!(frame.width > 0 && !frame.planes.is_empty());

        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn proxy_builds_via_hardware_decode_lane() {
        let Some(src) = sibling_main_asset("testsrc_h264.mp4") else {
            return;
        };
        let out = std::env::temp_dir().join("cutlass_proxy_hwdec.mp4");
        let _ = std::fs::remove_file(&out);

        // Hardware-decode lane: source decodes on the GPU block, frames transfer
        // to CPU, then scale + HW encode. Auto falls back to SW where no HW exists,
        // so this stays correct on CI without a GPU.
        let mut progressed: u64 = 0;
        let mut on_progress = |done: u64| progressed = done;
        let stats = build_proxy_with(
            &src,
            &out,
            ProxyConfig {
                target_height: 180,
                quality: 23,
                bitrate: 2_000_000,
                hardware: true,
            },
            ProxyBuildOptions {
                decode: HwAccel::Auto,
                decode_threads: 0,
            },
            Some(&mut on_progress),
        )
        .expect("build proxy via hw decode");
        assert!(stats.frames > 0, "no frames encoded");
        assert!(progressed > 0, "progress callback never fired");
        assert!(out.exists(), "proxy file missing");

        let mut dec = Decoder::open_with(&out, DecodeOptions::default().hw_accel(HwAccel::None))
            .expect("open proxy");
        let frame = dec
            .seek_to_frame(Duration::from_millis(300))
            .expect("seek proxy")
            .expect("frame after seek");
        assert!(frame.width > 0 && !frame.planes.is_empty());

        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn scaled_dims_preserve_aspect_and_evenness() {
        // 3840x2160 -> 1080 tall, 1920 wide.
        assert_eq!(scaled_dims(3840, 2160, 1080), (1920, 1080));
        // Never upscale past the source height.
        assert_eq!(scaled_dims(1280, 720, 1080), (1280, 720));
        // Odd results round down to even.
        let (w, h) = scaled_dims(1921, 1081, 1080);
        assert_eq!(w % 2, 0);
        assert_eq!(h % 2, 0);
    }
}
