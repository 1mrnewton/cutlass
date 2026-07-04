//! Timeline → video export: composite every frame `0..duration` and push it to
//! a [`VideoEncoder`].
//!
//! [`export`] is the encoder-agnostic core (the "composite every frame → mux"
//! path the roadmap calls for): it renders each tick of the timeline, wraps the
//! composited [`RgbaImage`] as a packed-RGBA [`VideoFrame`], and feeds the
//! [`cutlass_core::VideoEncoder`] trait. The codec/container is whatever encoder
//! you pass. [`export_to_file`] is the batteries-included path: it opens the
//! platform-native encoder (H.264→mp4 on Apple) for you. For a portable,
//! codec-free output there is [`PngSequenceEncoder`] below, which writes one PNG
//! per frame (handy for tests, previews, and CI).

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use cutlass_core::{
    AudioEncoderConfig, ColorSpace, CpuImage, EncodeError, EncoderConfig, FrameData, PixelFormat,
    Plane, Rational, RationalTime, Rect, RgbaImage, Rotation, VideoEncoder, VideoFrame, resample,
};
use cutlass_models::Project;

use crate::error::RenderError;
use crate::export_audio::{
    EXPORT_AUDIO_CHANNELS, EXPORT_AUDIO_RATE, ExportAudioMixer, sample_boundary,
};
use crate::render::Renderer;
use crate::resolve::canvas_size;

/// Output shape of an export: pixel size and frame sampling rate. The
/// defaults ([`ExportSettings::for_project`]) reproduce the project exactly;
/// overrides drive the resolution / fps pickers in the shells' export sheets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExportSettings {
    /// Output size in pixels. When it differs from the project canvas, each
    /// frame is uniformly scaled and centered (letterboxed on an aspect
    /// mismatch) — see [`Renderer::render_frame_sized`].
    pub size: (u32, u32),
    /// Frame sampling rate over `0..duration`.
    pub frame_rate: Rational,
}

impl ExportSettings {
    /// The project's native output: resolved canvas size at the timeline rate.
    pub fn for_project(project: &Project) -> Self {
        Self {
            size: canvas_size(project),
            frame_rate: project.timeline().frame_rate,
        }
    }

    /// The same settings with both dimensions rounded down to even (min 2) —
    /// H.264 encoders reject odd sizes.
    pub fn evened(self) -> Self {
        let even = |v: u32| (v & !1).max(2);
        Self {
            size: (even(self.size.0), even(self.size.1)),
            frame_rate: self.frame_rate,
        }
    }
}

/// Per-frame export observer: called with `(frames_done, frames_total)` before
/// each frame is rendered and once more after the encoder is finalized.
/// Returning `false` cancels the export ([`RenderError::Cancelled`]); the
/// output is left unfinalized and should be discarded by the caller.
pub type ExportObserver<'a> = &'a mut dyn FnMut(u64, u64) -> bool;

/// Composite the whole timeline and encode it to a video file at `path`, using
/// the platform-native encoder (H.264/mp4 on Apple). Returns the frame count.
///
/// Convenience over [`export`]: builds the right [`EncoderConfig`] from the
/// project and opens the native encoder for you. On platforms without a native
/// encoder this returns [`RenderError::Encode`].
pub fn export_to_file(
    renderer: &mut Renderer,
    project: &Project,
    path: &Path,
) -> Result<u64, RenderError> {
    export_to_file_observed(
        renderer,
        project,
        path,
        ExportSettings::for_project(project),
        &mut |_, _| true,
    )
}

/// [`export_to_file`] with explicit output settings and a progress/cancel
/// observer — the export-job path behind the shells' export sheets.
pub fn export_to_file_observed(
    renderer: &mut Renderer,
    project: &Project,
    path: &Path,
    settings: ExportSettings,
    observer: ExportObserver<'_>,
) -> Result<u64, RenderError> {
    let settings = settings.evened();
    let mut encoder = cutlass_encoder::open_encoder(path, export_config_with(project, settings))?;
    export_observed(renderer, project, encoder.as_mut(), settings, observer)
}

/// Composite every frame of `project` and push it to `encoder`, returning the
/// number of frames written.
///
/// Frames are rendered at the timeline frame rate over `0..duration`. When the
/// project has audible audio *and* the encoder has an audio track, the mixed
/// stereo audio for each frame's wall-clock span is pushed right after the
/// video frame (running sample cursor, so non-integer NTSC rates stay in sync).
/// Encoders without an audio track (e.g. [`PngSequenceEncoder`]) report
/// [`EncodeError::Unsupported`] from the first `push_audio`, after which audio
/// is skipped for the rest of the export. The encoder is finalized before
/// returning.
pub fn export(
    renderer: &mut Renderer,
    project: &Project,
    encoder: &mut dyn VideoEncoder,
) -> Result<u64, RenderError> {
    export_observed(
        renderer,
        project,
        encoder,
        ExportSettings::for_project(project),
        &mut |_, _| true,
    )
}

/// [`export`] with explicit output settings and a progress/cancel observer.
///
/// The observer is called with `(frames_done, frames_total)` before each frame
/// (so `(0, total)` announces the length up front) and once more with
/// `(total, total)` after the encoder is finalized. Returning `false` stops
/// the export with [`RenderError::Cancelled`] *without* finalizing the
/// encoder: a cancelled output file is garbage and the caller should delete it.
pub fn export_observed(
    renderer: &mut Renderer,
    project: &Project,
    encoder: &mut dyn VideoEncoder,
    settings: ExportSettings,
    observer: ExportObserver<'_>,
) -> Result<u64, RenderError> {
    let rate = settings.frame_rate;
    let duration = resample(project.timeline().duration(), rate);
    let frames = duration.value.max(0) as u64;
    let native = settings.size == canvas_size(project);

    let mut mixer = ExportAudioMixer::for_project(project);
    let audio_rate = Rational::new(EXPORT_AUDIO_RATE as i32, 1);
    let mut block = Vec::new();

    for i in 0..frames {
        if !observer(i, frames) {
            return Err(RenderError::Cancelled);
        }
        let t = RationalTime::new(i as i64, rate);
        let image = if native {
            renderer.render_frame(project, t)?
        } else {
            renderer.render_frame_sized(project, t, settings.size.0, settings.size.1)?
        };
        let frame = rgba_to_frame(image, t);
        encoder.push(&frame)?;

        if let Some(mix) = mixer.as_mut() {
            let from = sample_boundary(i as i64, rate.num, rate.den);
            let to = sample_boundary(i as i64 + 1, rate.num, rate.den);
            let count = (to - from).max(0) as usize;
            block.resize(count * EXPORT_AUDIO_CHANNELS as usize, 0.0);
            mix.mix_into(from, &mut block)?;
            let pts = RationalTime::new(from, audio_rate);
            match encoder.push_audio(&block, pts) {
                Ok(()) => {}
                // The encoder has no audio track (PNG sink, video-only config):
                // stop attempting audio for the rest of the export.
                Err(EncodeError::Unsupported(_)) => mixer = None,
                Err(e) => return Err(e.into()),
            }
        }
    }
    encoder.finish()?;
    observer(frames, frames);
    Ok(frames)
}

/// The [`EncoderConfig`] an exporter should use for `project`: the resolved
/// canvas size, the timeline frame rate, and BT.709 colorimetry. When the
/// project carries audible audio, a stereo 48 kHz AAC track is configured too.
/// Construct a native encoder with this, then drive it via [`export`].
pub fn export_config(project: &Project) -> EncoderConfig {
    export_config_with(project, ExportSettings::for_project(project))
}

/// [`export_config`] for explicit output settings (resolution / fps
/// overrides). Pass the same settings to [`export_observed`] so rendered
/// frames match the encoder.
pub fn export_config_with(project: &Project, settings: ExportSettings) -> EncoderConfig {
    let mut config = EncoderConfig::new(settings.size, settings.frame_rate, ColorSpace::BT709);
    if ExportAudioMixer::for_project(project).is_some() {
        config = config.with_audio(AudioEncoderConfig::new(
            EXPORT_AUDIO_RATE,
            EXPORT_AUDIO_CHANNELS,
        ));
    }
    config
}

/// Wrap a composited [`RgbaImage`] as a single-plane packed-RGBA CPU frame.
fn rgba_to_frame(image: RgbaImage, pts: RationalTime) -> VideoFrame {
    let (width, height) = (image.width, image.height);
    let plane = Plane::new(image.pixels, width as usize * 4, height as usize);
    VideoFrame::new(
        pts,
        PixelFormat::Rgba8,
        ColorSpace::BT709,
        (width, height),
        Rect::from_size(width, height),
        Rotation::None,
        FrameData::Cpu(CpuImage::new(vec![plane])),
    )
}

/// A portable [`VideoEncoder`] that writes each pushed frame as a numbered PNG
/// (`frame_00000.png`, `frame_00001.png`, …) into a directory.
///
/// Pure-Rust and platform-independent: it proves the whole render→export loop
/// without a native codec, and gives an eyeballable result. It accepts only the
/// packed-RGBA frames [`export`] produces.
pub struct PngSequenceEncoder {
    dir: PathBuf,
    index: u64,
}

impl PngSequenceEncoder {
    /// Create the output directory (if needed) and start at frame 0.
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self, EncodeError> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir).map_err(|e| EncodeError::Start(e.to_string()))?;
        Ok(Self { dir, index: 0 })
    }

    /// Number of frames written so far.
    pub fn frames_written(&self) -> u64 {
        self.index
    }
}

impl VideoEncoder for PngSequenceEncoder {
    fn push(&mut self, frame: &VideoFrame) -> Result<(), EncodeError> {
        if frame.format != PixelFormat::Rgba8 {
            return Err(EncodeError::unsupported(format!(
                "PNG sink expects Rgba8 frames, got {:?}",
                frame.format
            )));
        }
        let cpu = frame
            .cpu()
            .ok_or_else(|| EncodeError::unsupported("PNG sink needs CPU frames"))?;
        let plane = cpu
            .planes
            .first()
            .ok_or_else(|| EncodeError::Encode("frame has no plane".into()))?;

        let path = self.dir.join(format!("frame_{:05}.png", self.index));
        write_png(&path, frame.width(), frame.height(), plane)?;
        self.index += 1;
        Ok(())
    }

    fn finish(&mut self) -> Result<(), EncodeError> {
        Ok(())
    }
}

/// Write one RGBA8 plane to an 8-bit PNG, compacting away any row padding.
fn write_png(path: &Path, width: u32, height: u32, plane: &Plane) -> Result<(), EncodeError> {
    let row_bytes = width as usize * 4;
    let file = File::create(path).map_err(|e| EncodeError::Encode(e.to_string()))?;
    let mut encoder = png::Encoder::new(BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|e| EncodeError::Encode(e.to_string()))?;

    if plane.stride == row_bytes {
        writer
            .write_image_data(&plane.data[..row_bytes * height as usize])
            .map_err(|e| EncodeError::Encode(e.to_string()))?;
    } else {
        // Decoder/compositor readback may pad rows; PNG wants tight rows.
        let mut tight = Vec::with_capacity(row_bytes * height as usize);
        for row in 0..height as usize {
            let start = row * plane.stride;
            tight.extend_from_slice(&plane.data[start..start + row_bytes]);
        }
        writer
            .write_image_data(&tight)
            .map_err(|e| EncodeError::Encode(e.to_string()))?;
    }
    Ok(())
}
