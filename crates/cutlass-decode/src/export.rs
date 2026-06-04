//! Timeline exporter sink: encode a stream of composited **RGBA8** frames into
//! an H.264 MP4.
//!
//! This is the encode half of "composite every frame `0..duration` -> mux to
//! mp4". It is deliberately source-agnostic: it knows nothing about the
//! timeline, only how to take fully-rendered RGBA canvases (one per output
//! frame, in order) and turn them into a constant-frame-rate file. The engine
//! owns the loop that produces those canvases (see `cutlass-engines`), so this
//! same sink backs both the Rust export path and the future Python `export`.
//!
//! Unlike [`build_proxy`](crate::build_proxy) — which is all-intra (GOP-1) for
//! fast seeking — the export uses the encoder's default GOP so the deliverable
//! is bit-efficient rather than scrub-optimized.

use std::path::Path;

use ffmpeg_next::format::{self, context::Output};
use ffmpeg_next::software::scaling;
use ffmpeg_next::util::format::Pixel;
use ffmpeg_next::util::frame::video::Video as VideoFrame;
use ffmpeg_next::{Dictionary, Rational, codec, encoder::Video as VideoEncoder};
use tracing::debug;

use crate::decoder::ensure_ffmpeg_init;
use crate::encode::{drain_encoder, find_h264_encoder};
use crate::error::DecodeError;

/// How to encode an exported timeline.
#[derive(Debug, Clone, Copy)]
pub struct ExportConfig {
    /// Output width in pixels. Must be non-zero and even (H.264 4:2:0).
    pub width: u32,
    /// Output height in pixels. Must be non-zero and even (H.264 4:2:0).
    pub height: u32,
    /// Output frame rate numerator (frames per `frame_rate_den` seconds).
    pub frame_rate_num: i32,
    /// Output frame rate denominator.
    pub frame_rate_den: i32,
    /// Constant-quality level (libx264 CRF, 0–51, lower = better). Used on the
    /// software path; ~18 is visually near-transparent.
    pub quality: u8,
    /// Target bitrate (bits/sec) for hardware encoders without a CRF mode.
    pub bitrate: usize,
    /// Prefer a hardware H.264 encoder. Off by default for export: software
    /// libx264 at constant quality gives the cleanest deliverable.
    pub hardware: bool,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            frame_rate_num: 30,
            frame_rate_den: 1,
            quality: 18,
            bitrate: 12_000_000,
            hardware: false,
        }
    }
}

/// Result of a completed export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExportStats {
    pub frames: u64,
    pub width: u32,
    pub height: u32,
}

/// A push-based H.264/MP4 encoder fed one RGBA8 canvas per output frame.
///
/// Create it with [`VideoExport::create`], call [`push_rgba`](Self::push_rgba)
/// once per timeline frame in order, then [`finish`](Self::finish) to flush the
/// encoder and write the container trailer. Frames are presented on a clean
/// `1/fps` timeline (pts = frame index), so the file is constant-frame-rate.
pub struct VideoExport {
    octx: Output,
    encoder: VideoEncoder,
    scaler: scaling::Context,
    /// Reused RGBA input frame: the scaler only reads it, so it's safe to refill
    /// each push and avoid a per-frame allocation.
    rgba: VideoFrame,
    ost_index: usize,
    enc_tb: Rational,
    ost_tb: Rational,
    frame_index: i64,
    width: u32,
    height: u32,
    finished: bool,
}

impl VideoExport {
    /// Open `output` and prepare the encoder for `config.width`×`config.height`
    /// RGBA frames at the given frame rate. Writes the container header.
    pub fn create(output: &Path, config: ExportConfig) -> Result<Self, DecodeError> {
        ensure_ffmpeg_init()?;

        let (width, height) = (config.width, config.height);
        if width == 0 || height == 0 || width % 2 != 0 || height % 2 != 0 {
            return Err(DecodeError::unsupported(
                "export dimensions must be non-zero and even",
            ));
        }
        if config.frame_rate_num <= 0 || config.frame_rate_den <= 0 {
            return Err(DecodeError::unsupported("export frame rate must be positive"));
        }

        let fps = Rational::new(config.frame_rate_num, config.frame_rate_den);
        let enc_tb = fps.invert();

        let codec = find_h264_encoder(config.hardware)
            .ok_or_else(|| DecodeError::unsupported("no H.264 encoder available"))?;
        let use_crf = codec.name() == "libx264";

        let mut octx = format::output(&output).map_err(DecodeError::Open)?;
        let global_header = octx.format().flags().contains(format::Flags::GLOBAL_HEADER);

        let mut enc = codec::context::Context::new_with_codec(codec)
            .encoder()
            .video()
            .map_err(DecodeError::Open)?;
        enc.set_width(width);
        enc.set_height(height);
        enc.set_format(Pixel::YUV420P);
        enc.set_color_range(ffmpeg_next::color::Range::MPEG);
        enc.set_frame_rate(Some(fps));
        enc.set_time_base(enc_tb);
        // Unlike the proxy builder we leave the GOP at the encoder default: the
        // export is a deliverable (bit-efficient), not a scrub-optimized cache.
        if !use_crf {
            enc.set_bit_rate(config.bitrate);
        }
        if global_header {
            enc.set_flags(codec::Flags::GLOBAL_HEADER);
        }

        let mut enc_opts = Dictionary::new();
        if use_crf {
            let crf = config.quality.to_string();
            enc_opts.set("crf", &crf);
            enc_opts.set("preset", "medium");
        }
        let encoder = enc.open_with(enc_opts).map_err(DecodeError::Open)?;

        let ost_index = {
            let mut ost = octx.add_stream(codec).map_err(DecodeError::Open)?;
            ost.set_parameters(&encoder);
            ost.index()
        };

        octx.write_header().map_err(DecodeError::Io)?;
        let ost_tb = octx.stream(ost_index).unwrap().time_base();

        // Canvas size == output size, so this only converts RGBA -> YUV420P.
        let scaler = scaling::Context::get(
            Pixel::RGBA,
            width,
            height,
            Pixel::YUV420P,
            width,
            height,
            scaling::Flags::BILINEAR,
        )
        .map_err(DecodeError::Decode)?;

        let rgba = VideoFrame::new(Pixel::RGBA, width, height);

        Ok(Self {
            octx,
            encoder,
            scaler,
            rgba,
            ost_index,
            enc_tb,
            ost_tb,
            frame_index: 0,
            width,
            height,
            finished: false,
        })
    }

    /// Encode one frame from a row-major RGBA8 buffer (`width*height*4` bytes).
    ///
    /// Frames are consumed in presentation order; the Nth call becomes output
    /// frame N. The buffer is copied into the encoder's pipeline, so the caller
    /// may reuse or drop it immediately after this returns.
    pub fn push_rgba(&mut self, rgba: &[u8]) -> Result<(), DecodeError> {
        let expected = (self.width as usize) * (self.height as usize) * 4;
        if rgba.len() != expected {
            return Err(DecodeError::unsupported(format!(
                "rgba buffer is {} bytes, expected {expected}",
                rgba.len()
            )));
        }

        // Refill the reusable RGBA frame row by row, honoring its stride (which
        // ffmpeg may pad above width*4 for alignment).
        let row_bytes = (self.width as usize) * 4;
        let stride = self.rgba.stride(0);
        let data = self.rgba.data_mut(0);
        for y in 0..self.height as usize {
            let dst = &mut data[y * stride..y * stride + row_bytes];
            let src = &rgba[y * row_bytes..y * row_bytes + row_bytes];
            dst.copy_from_slice(src);
        }

        // Fresh YUV output buffer per frame: a hardware encoder may still hold a
        // ref to the previous one, so we must not reuse it in place.
        let mut yuv = VideoFrame::empty();
        self.scaler
            .run(&self.rgba, &mut yuv)
            .map_err(DecodeError::Decode)?;
        yuv.set_pts(Some(self.frame_index));
        self.encoder.send_frame(&yuv).map_err(DecodeError::Decode)?;
        self.frame_index += 1;

        drain_encoder(
            &mut self.encoder,
            &mut self.octx,
            self.ost_index,
            self.enc_tb,
            self.ost_tb,
        )
    }

    /// Flush the encoder, write the container trailer, and return frame stats.
    pub fn finish(mut self) -> Result<ExportStats, DecodeError> {
        self.finish_inner()?;
        Ok(ExportStats {
            frames: self.frame_index as u64,
            width: self.width,
            height: self.height,
        })
    }

    fn finish_inner(&mut self) -> Result<(), DecodeError> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        self.encoder.send_eof().map_err(DecodeError::Decode)?;
        drain_encoder(
            &mut self.encoder,
            &mut self.octx,
            self.ost_index,
            self.enc_tb,
            self.ost_tb,
        )?;
        self.octx.write_trailer().map_err(DecodeError::Io)?;
        debug!(frames = self.frame_index, self.width, self.height, "exported timeline");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    use crate::{DecodeOptions, Decoder, HwAccel};

    #[test]
    fn exports_rgba_frames_to_seekable_mp4() {
        let out = std::env::temp_dir().join("cutlass_export_smoke.mp4");
        let _ = std::fs::remove_file(&out);

        let (w, h) = (64u32, 48u32);
        let mut sink = VideoExport::create(
            &out,
            ExportConfig {
                width: w,
                height: h,
                frame_rate_num: 30,
                frame_rate_den: 1,
                quality: 23,
                bitrate: 2_000_000,
                // Software libx264 keeps the test deterministic on CI (no GPU).
                hardware: false,
            },
        )
        .expect("create export");

        // 15 opaque red frames.
        let frame: Vec<u8> = std::iter::repeat_n([220u8, 30, 30, 255], (w * h) as usize)
            .flatten()
            .collect();
        for _ in 0..15 {
            sink.push_rgba(&frame).expect("push frame");
        }
        let stats = sink.finish().expect("finish export");

        assert_eq!(stats.frames, 15);
        assert_eq!((stats.width, stats.height), (w, h));
        assert!(out.exists(), "export file missing");

        // The output must decode and seek.
        let mut dec = Decoder::open_with(&out, DecodeOptions::default().hw_accel(HwAccel::None))
            .expect("open export");
        let decoded = dec
            .seek_to_frame(Duration::from_millis(200))
            .expect("seek export")
            .expect("frame after seek");
        assert!(decoded.width > 0 && !decoded.planes.is_empty());

        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn rejects_odd_or_zero_dimensions() {
        let out = std::env::temp_dir().join("cutlass_export_baddims.mp4");
        let err = VideoExport::create(
            &out,
            ExportConfig {
                width: 65,
                height: 48,
                ..Default::default()
            },
        );
        assert!(err.is_err(), "odd width should be rejected");
    }

    #[test]
    fn rejects_wrong_buffer_size() {
        let out = std::env::temp_dir().join("cutlass_export_badbuf.mp4");
        let _ = std::fs::remove_file(&out);
        let mut sink = VideoExport::create(
            &out,
            ExportConfig {
                width: 8,
                height: 8,
                hardware: false,
                ..Default::default()
            },
        )
        .expect("create export");
        assert!(sink.push_rgba(&[0u8; 10]).is_err());
        let _ = std::fs::remove_file(&out);
    }
}
