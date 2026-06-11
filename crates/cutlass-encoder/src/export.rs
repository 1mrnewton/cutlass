//! Timeline exporter sink: encode a stream of composited **RGBA8** frames into
//! an H.264 MP4, optionally muxed with an AAC stereo audio track.
//!
//! This is the encode half of "composite every frame `0..duration` -> mux to
//! mp4". It is deliberately source-agnostic: it knows nothing about the
//! timeline, only how to take fully-rendered RGBA canvases (one per output
//! frame, in order) plus interleaved stereo samples and turn them into a
//! constant-frame-rate file. The engine owns the loop that produces those
//! canvases, so this same sink backs both the Rust export path and the future
//! Python `export`.
//!
//! Unlike [`build_proxy`](crate::proxy::build_proxy) — which is all-intra
//! (GOP-1) for fast seeking — the export uses the encoder's default GOP so the
//! deliverable is bit-efficient rather than scrub-optimized.

use std::path::Path;

use ffmpeg_next::format::{self, context::Output};
use ffmpeg_next::software::scaling;
use ffmpeg_next::util::channel_layout::ChannelLayout;
use ffmpeg_next::util::format::Pixel;
use ffmpeg_next::util::format::sample::{Sample, Type as SampleType};
use ffmpeg_next::util::frame::audio::Audio as AudioFrame;
use ffmpeg_next::util::frame::video::Video as VideoFrame;
use ffmpeg_next::{
    Dictionary, Rational, codec, encoder,
    encoder::{Audio as AudioEncoder, Video as VideoEncoder},
};
use tracing::debug;

use crate::error::EncodeError;
use crate::h264::{
    drain_encoder, ensure_ffmpeg_init, fill_rgba_frame, fill_yuv420p_frame, find_h264_encoder,
};

/// Audio is always interleaved stereo at the sink boundary.
pub const AUDIO_CHANNELS: usize = 2;

/// AAC bitrate for the deliverable (192 kbps stereo: transparent for speech
/// and most music, the CapCut/YouTube default ballpark).
const AUDIO_BITRATE: usize = 192_000;

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
    /// Mux an AAC stereo track at this sample rate (Hz). `None` ⇒ video-only
    /// file (timelines with no audible clips).
    pub audio_rate: Option<u32>,
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
            audio_rate: None,
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
    /// YUV→YUV resize, created on the first pushed frame whose dimensions
    /// differ from the output (resolution presets smaller than the canvas).
    /// Keyed by input size so a mid-stream change rebuilds it.
    yuv_scaler: Option<(u32, u32, scaling::Context)>,
    rgba: VideoFrame,
    ost_index: usize,
    enc_tb: Rational,
    ost_tb: Rational,
    frame_index: i64,
    width: u32,
    height: u32,
    audio: Option<AudioTrack>,
    finished: bool,
}

/// The AAC side of the mux: buffers interleaved stereo f32 until a full
/// encoder frame is available, then encodes with sample-accurate PTS.
struct AudioTrack {
    encoder: AudioEncoder,
    ost_index: usize,
    enc_tb: Rational,
    ost_tb: Rational,
    /// Samples (per channel) the AAC encoder consumes per frame.
    frame_size: usize,
    /// Interleaved stereo samples not yet handed to the encoder.
    fifo: Vec<f32>,
    /// PTS of the next encoded frame, in samples (1/rate time base).
    samples_written: i64,
}

impl AudioTrack {
    /// Build the AAC encoder and add its stream to `octx`. Must run before
    /// the container header is written; `ost_tb` is finalized afterwards.
    fn new(octx: &mut Output, rate: u32, global_header: bool) -> Result<Self, EncodeError> {
        if rate == 0 {
            return Err(EncodeError::unsupported("zero audio sample rate"));
        }
        let codec = encoder::find_by_name("aac")
            .or_else(|| encoder::find(codec::Id::AAC))
            .ok_or_else(|| EncodeError::unsupported("no AAC encoder available"))?;

        let mut enc = codec::context::Context::new_with_codec(codec)
            .encoder()
            .audio()
            .map_err(EncodeError::Open)?;
        let enc_tb = Rational::new(1, rate as i32);
        enc.set_rate(rate as i32);
        enc.set_format(Sample::F32(SampleType::Planar));
        enc.set_channel_layout(ChannelLayout::STEREO);
        enc.set_time_base(enc_tb);
        enc.set_bit_rate(AUDIO_BITRATE);
        if global_header {
            enc.set_flags(codec::Flags::GLOBAL_HEADER);
        }
        let encoder = enc.open().map_err(EncodeError::Open)?;

        let ost_index = {
            let mut ost = octx.add_stream(codec).map_err(EncodeError::Open)?;
            ost.set_parameters(&encoder);
            ost.index()
        };
        let frame_size = match encoder.frame_size() {
            0 => 1024,
            n => n as usize,
        };

        Ok(Self {
            encoder,
            ost_index,
            enc_tb,
            ost_tb: enc_tb, // finalized after write_header
            frame_size,
            fifo: Vec::new(),
            samples_written: 0,
        })
    }

    /// Queue interleaved stereo samples and encode every full frame.
    fn push(&mut self, octx: &mut Output, samples: &[f32]) -> Result<(), EncodeError> {
        if !samples.len().is_multiple_of(AUDIO_CHANNELS) {
            return Err(EncodeError::unsupported(
                "audio buffer is not whole interleaved stereo frames",
            ));
        }
        self.fifo.extend_from_slice(samples);

        let chunk = self.frame_size * AUDIO_CHANNELS;
        if self.fifo.len() < chunk {
            return Ok(());
        }
        let buf = std::mem::take(&mut self.fifo);
        let mut frames = buf.chunks_exact(chunk);
        for frame in &mut frames {
            self.encode_samples(octx, frame)?;
        }
        self.fifo.extend_from_slice(frames.remainder());
        Ok(())
    }

    /// Encode the trailing partial frame (legal as the final frame), flush
    /// the encoder, and drain remaining packets.
    fn finish(&mut self, octx: &mut Output) -> Result<(), EncodeError> {
        if !self.fifo.is_empty() {
            let tail = std::mem::take(&mut self.fifo);
            self.encode_samples(octx, &tail)?;
        }
        self.encoder.send_eof().map_err(EncodeError::Encode)?;
        drain_encoder(
            &mut self.encoder,
            octx,
            self.ost_index,
            self.enc_tb,
            self.ost_tb,
        )
    }

    /// Encode one frame of interleaved stereo as planar f32.
    fn encode_samples(&mut self, octx: &mut Output, samples: &[f32]) -> Result<(), EncodeError> {
        let count = samples.len() / AUDIO_CHANNELS;
        let mut frame = AudioFrame::new(
            Sample::F32(SampleType::Planar),
            count,
            ChannelLayout::STEREO,
        );
        {
            // Planar fill: plane 0 = left, plane 1 = right.
            let left = frame.plane_mut::<f32>(0);
            for (dst, src) in left.iter_mut().zip(samples.chunks_exact(AUDIO_CHANNELS)) {
                *dst = src[0];
            }
            let right = frame.plane_mut::<f32>(1);
            for (dst, src) in right.iter_mut().zip(samples.chunks_exact(AUDIO_CHANNELS)) {
                *dst = src[1];
            }
        }
        frame.set_pts(Some(self.samples_written));
        self.samples_written += count as i64;
        self.encoder
            .send_frame(&frame)
            .map_err(EncodeError::Encode)?;
        drain_encoder(
            &mut self.encoder,
            octx,
            self.ost_index,
            self.enc_tb,
            self.ost_tb,
        )
    }
}

impl VideoExport {
    /// Open `output` and prepare the encoder for `config.width`×`config.height`
    /// RGBA frames at the given frame rate. Writes the container header.
    pub fn create(output: &Path, config: ExportConfig) -> Result<Self, EncodeError> {
        ensure_ffmpeg_init()?;

        let (width, height) = (config.width, config.height);
        if width == 0 || height == 0 || width % 2 != 0 || height % 2 != 0 {
            return Err(EncodeError::unsupported(
                "export dimensions must be non-zero and even",
            ));
        }
        if config.frame_rate_num <= 0 || config.frame_rate_den <= 0 {
            return Err(EncodeError::unsupported(
                "export frame rate must be positive",
            ));
        }

        let fps = Rational::new(config.frame_rate_num, config.frame_rate_den);
        let enc_tb = fps.invert();

        let codec = find_h264_encoder(config.hardware)
            .ok_or_else(|| EncodeError::unsupported("no H.264 encoder available"))?;
        let use_crf = codec.name() == "libx264";

        let mut octx = format::output(output).map_err(EncodeError::Open)?;
        let global_header = octx.format().flags().contains(format::Flags::GLOBAL_HEADER);

        let mut enc = codec::context::Context::new_with_codec(codec)
            .encoder()
            .video()
            .map_err(EncodeError::Open)?;
        enc.set_width(width);
        enc.set_height(height);
        enc.set_format(Pixel::YUV420P);
        enc.set_color_range(ffmpeg_next::color::Range::MPEG);
        enc.set_frame_rate(Some(fps));
        enc.set_time_base(enc_tb);
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
        let encoder = enc.open_with(enc_opts).map_err(EncodeError::Open)?;

        let ost_index = {
            let mut ost = octx.add_stream(codec).map_err(EncodeError::Open)?;
            ost.set_parameters(&encoder);
            ost.index()
        };

        // The audio stream must exist before the header is written.
        let mut audio = match config.audio_rate {
            Some(rate) => Some(AudioTrack::new(&mut octx, rate, global_header)?),
            None => None,
        };

        octx.write_header().map_err(EncodeError::Io)?;
        let ost_tb = octx.stream(ost_index).unwrap().time_base();
        if let Some(track) = &mut audio {
            track.ost_tb = octx.stream(track.ost_index).unwrap().time_base();
        }

        let scaler = scaling::Context::get(
            Pixel::RGBA,
            width,
            height,
            Pixel::YUV420P,
            width,
            height,
            scaling::Flags::BILINEAR,
        )
        .map_err(EncodeError::Encode)?;

        let rgba = VideoFrame::new(Pixel::RGBA, width, height);

        Ok(Self {
            octx,
            encoder,
            scaler,
            yuv_scaler: None,
            rgba,
            ost_index,
            enc_tb,
            ost_tb,
            frame_index: 0,
            width,
            height,
            audio,
            finished: false,
        })
    }

    /// Queue interleaved stereo f32 samples (any length of whole frames) for
    /// the AAC track. Call in rough lockstep with video pushes so the muxer
    /// can interleave; exact alignment is not required.
    ///
    /// Errors if the export was created without `audio_rate`.
    pub fn push_audio(&mut self, samples: &[f32]) -> Result<(), EncodeError> {
        let Some(track) = &mut self.audio else {
            return Err(EncodeError::unsupported(
                "export was created without an audio track",
            ));
        };
        track.push(&mut self.octx, samples)
    }

    /// Encode one frame from a row-major RGBA8 buffer (`width*height*4` bytes).
    ///
    /// Frames are consumed in presentation order; the Nth call becomes output
    /// frame N. The buffer is copied into the encoder's pipeline, so the caller
    /// may reuse or drop it immediately after this returns.
    /// Encode one frame from tight YUV420P planes (typical GPU readback layout).
    ///
    /// Frames whose dimensions differ from the configured output are resized
    /// (swscale bilinear, same as the RGBA path) — this is how resolution
    /// presets below the composite canvas are produced.
    pub fn push_yuv420p(
        &mut self,
        width: u32,
        height: u32,
        y: &[u8],
        u: &[u8],
        v: &[u8],
    ) -> Result<(), EncodeError> {
        let mut yuv = VideoFrame::new(Pixel::YUV420P, width, height);
        fill_yuv420p_frame(&mut yuv, width, height, y, u, v)?;

        let mut frame = if width == self.width && height == self.height {
            yuv
        } else {
            let scaler = self.yuv_scaler_for(width, height)?;
            let mut scaled = VideoFrame::empty();
            scaler.run(&yuv, &mut scaled).map_err(EncodeError::Encode)?;
            scaled
        };
        frame.set_pts(Some(self.frame_index));
        self.encoder.send_frame(&frame).map_err(EncodeError::Encode)?;
        self.frame_index += 1;

        drain_encoder(
            &mut self.encoder,
            &mut self.octx,
            self.ost_index,
            self.enc_tb,
            self.ost_tb,
        )
    }

    /// The cached YUV resize context for `src` → output dimensions. Input size
    /// is constant for a whole export, so this builds once in practice.
    fn yuv_scaler_for(
        &mut self,
        src_w: u32,
        src_h: u32,
    ) -> Result<&mut scaling::Context, EncodeError> {
        let stale = !matches!(self.yuv_scaler, Some((w, h, _)) if w == src_w && h == src_h);
        if stale {
            let ctx = scaling::Context::get(
                Pixel::YUV420P,
                src_w,
                src_h,
                Pixel::YUV420P,
                self.width,
                self.height,
                scaling::Flags::BILINEAR,
            )
            .map_err(EncodeError::Encode)?;
            self.yuv_scaler = Some((src_w, src_h, ctx));
        }
        match &mut self.yuv_scaler {
            Some((_, _, ctx)) => Ok(ctx),
            None => unreachable!("yuv scaler was just installed"),
        }
    }

    /// Legacy: encode from RGBA8 via CPU FFmpeg scaler.
    pub fn push_rgba(&mut self, rgba: &[u8]) -> Result<(), EncodeError> {
        fill_rgba_frame(&mut self.rgba, self.width, self.height, rgba)?;

        let mut yuv = VideoFrame::empty();
        self.scaler
            .run(&self.rgba, &mut yuv)
            .map_err(EncodeError::Encode)?;
        yuv.set_pts(Some(self.frame_index));
        self.encoder.send_frame(&yuv).map_err(EncodeError::Encode)?;
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
    pub fn finish(mut self) -> Result<ExportStats, EncodeError> {
        self.finish_inner()?;
        Ok(ExportStats {
            frames: self.frame_index as u64,
            width: self.width,
            height: self.height,
        })
    }

    fn finish_inner(&mut self) -> Result<(), EncodeError> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        self.encoder.send_eof().map_err(EncodeError::Encode)?;
        drain_encoder(
            &mut self.encoder,
            &mut self.octx,
            self.ost_index,
            self.enc_tb,
            self.ost_tb,
        )?;
        if let Some(track) = &mut self.audio {
            track.finish(&mut self.octx)?;
        }
        self.octx.write_trailer().map_err(EncodeError::Io)?;
        debug!(
            frames = self.frame_index,
            self.width, self.height, "exported timeline"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use cutlass_decoder::{DecodeOptions, Decoder, HwAccel};

    use super::*;

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
                hardware: false,
                audio_rate: None,
            },
        )
        .expect("create export");

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
    fn scales_yuv_input_to_smaller_output() {
        let out = std::env::temp_dir().join("cutlass_export_scaled.mp4");
        let _ = std::fs::remove_file(&out);

        // Input canvas 128x96, output 64x48: push_yuv420p must resize.
        let (in_w, in_h) = (128u32, 96u32);
        let mut sink = VideoExport::create(
            &out,
            ExportConfig {
                width: 64,
                height: 48,
                frame_rate_num: 24,
                frame_rate_den: 1,
                quality: 30,
                bitrate: 1_000_000,
                hardware: false,
                audio_rate: None,
            },
        )
        .expect("create export");

        let y = vec![120u8; (in_w * in_h) as usize];
        let u = vec![128u8; ((in_w / 2) * (in_h / 2)) as usize];
        let v = vec![128u8; ((in_w / 2) * (in_h / 2)) as usize];
        for _ in 0..8 {
            sink.push_yuv420p(in_w, in_h, &y, &u, &v).expect("push scaled frame");
        }
        let stats = sink.finish().expect("finish");
        assert_eq!(stats.frames, 8);
        assert_eq!((stats.width, stats.height), (64, 48));

        let mut dec = Decoder::open_with(&out, DecodeOptions::default().hw_accel(HwAccel::None))
            .expect("open scaled export");
        let frame = dec
            .seek_to_frame(Duration::ZERO)
            .expect("seek")
            .expect("frame");
        assert_eq!((frame.width, frame.height), (64, 48));

        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn muxes_audio_track_and_roundtrips_samples() {
        let out = std::env::temp_dir().join("cutlass_export_audio.mp4");
        let _ = std::fs::remove_file(&out);

        const RATE: u32 = 48_000;
        let (w, h) = (64u32, 48u32);
        let fps = 24;
        let seconds = 2;
        let mut sink = VideoExport::create(
            &out,
            ExportConfig {
                width: w,
                height: h,
                frame_rate_num: fps,
                frame_rate_den: 1,
                quality: 30,
                bitrate: 1_000_000,
                hardware: false,
                audio_rate: Some(RATE),
            },
        )
        .expect("create export with audio");

        // 2s of video, with a 440 Hz tone pushed per-frame (2000 samples each)
        // — deliberately not a multiple of the AAC frame size (1024).
        let frame: Vec<u8> = std::iter::repeat_n([40u8, 90, 160, 255], (w * h) as usize)
            .flatten()
            .collect();
        let samples_per_frame = (RATE / fps as u32) as usize;
        let mut t = 0usize;
        for _ in 0..(seconds * fps) {
            sink.push_rgba(&frame).expect("push video");
            let mut audio = Vec::with_capacity(samples_per_frame * AUDIO_CHANNELS);
            for _ in 0..samples_per_frame {
                let s = (t as f32 * 440.0 * std::f32::consts::TAU / RATE as f32).sin() * 0.5;
                audio.push(s);
                audio.push(s);
                t += 1;
            }
            sink.push_audio(&audio).expect("push audio");
        }
        sink.finish().expect("finish");

        // The exported file must contain a decodable audio stream of ~2s.
        let mut reader =
            cutlass_decoder::AudioReader::open(&out, RATE).expect("exported file has audio");
        let mut buf = vec![0f32; 4096 * AUDIO_CHANNELS];
        let mut frames = 0u64;
        let mut heard = false;
        loop {
            let n = reader.read(&mut buf).expect("read exported audio");
            if n == 0 {
                break;
            }
            heard |= buf[..n * AUDIO_CHANNELS].iter().any(|&s| s.abs() > 0.05);
            frames += n as u64;
        }
        assert!(heard, "exported audio is not silence");
        let got_s = frames as f64 / f64::from(RATE);
        assert!(
            (got_s - seconds as f64).abs() < 0.2,
            "exported {got_s:.3}s of audio, wanted ~{seconds}s"
        );

        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn push_audio_without_track_is_rejected() {
        let out = std::env::temp_dir().join("cutlass_export_noaudio.mp4");
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
        assert!(sink.push_audio(&[0.0, 0.0]).is_err());
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
