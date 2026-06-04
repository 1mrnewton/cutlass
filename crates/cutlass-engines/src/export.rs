//! Timeline -> video file export: the loop that joins the engine's per-frame
//! resolve/composite to the encoder sink.
//!
//! [`Engine::export`] walks every timeline frame `0..duration`, resolves and
//! decodes its layer stack ([`frame_at`](crate::Engine::frame_at)), flattens it
//! to an RGBA canvas with the reference compositor, and pushes the result into a
//! [`VideoExport`] sink (`cutlass-decode`). This is the Rust export path; the
//! planned Python `export` wraps the very same method, so the heavy lifting
//! stays here in Rust.

use std::path::Path;

use cutlass_compositor::{CompositeLayer, composite};
use cutlass_decode::{ExportConfig, ExportStats, VideoExport};
use cutlass_models::Generator;

use crate::engine::{Engine, RenderedContent, RenderedLayer};
use crate::error::EngineError;

/// User-facing export options. Resolution is the rendered canvas size; the frame
/// rate is taken from the project timeline so output frames map 1:1 to timeline
/// frames.
#[derive(Debug, Clone, Copy)]
pub struct ExportSettings {
    /// Output width in pixels (rounded down to even).
    pub width: u32,
    /// Output height in pixels (rounded down to even).
    pub height: u32,
    /// Constant-quality level (libx264 CRF, lower = better). ~18 is near-transparent.
    pub quality: u8,
    /// Target bitrate (bits/sec) for hardware encoders without a CRF mode.
    pub bitrate: usize,
    /// Prefer a hardware H.264 encoder (faster, lower quality per bit).
    pub hardware: bool,
}

impl ExportSettings {
    /// Settings for a `width`×`height` export at the default quality (clean
    /// software libx264).
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            quality: 18,
            bitrate: 12_000_000,
            hardware: false,
        }
    }
}

impl Default for ExportSettings {
    fn default() -> Self {
        Self::new(1920, 1080)
    }
}

/// Map the engine's resolved layers onto the compositor's layer type.
///
/// Media frames become sampled `Frame` layers; solid generators become fills.
/// Text/shape/adjustment generators aren't drawable by the CPU compositor yet,
/// so they're skipped (rather than failing the render) until the compositor
/// grows those layer kinds.
pub fn to_composite_layers(layers: &[RenderedLayer]) -> Vec<CompositeLayer<'_>> {
    let mut out = Vec::with_capacity(layers.len());
    for layer in layers {
        match &layer.content {
            RenderedContent::Media(frame) => out.push(CompositeLayer::Frame(frame.as_ref())),
            RenderedContent::Generated(Generator::SolidColor { rgba }) => {
                out.push(CompositeLayer::Solid(*rgba));
            }
            RenderedContent::Generated(other) => {
                tracing::warn!(?other, "skipping generator the CPU compositor can't draw yet");
            }
        }
    }
    out
}

impl Engine {
    /// Render the whole timeline to an H.264 MP4 at `output`.
    ///
    /// Equivalent to [`export_with`](Engine::export_with) with no progress
    /// callback.
    pub fn export(&mut self, output: &Path, settings: ExportSettings) -> Result<ExportStats, EngineError> {
        self.export_with(output, settings, None)
    }

    /// Render the whole timeline to an H.264 MP4, reporting progress.
    ///
    /// `progress`, when given, is called after each frame with
    /// `(frames_done, total_frames)` so a UI can show a bar. It runs on the
    /// calling thread between frames and must not block.
    ///
    /// Frames with no content (timeline gaps) export as transparent-over-black.
    /// Decode is paused for background proxy builds during the export so the
    /// full-resolution reads don't compete with the transcoder.
    pub fn export_with(
        &mut self,
        output: &Path,
        settings: ExportSettings,
        mut progress: Option<&mut dyn FnMut(i64, i64)>,
    ) -> Result<ExportStats, EngineError> {
        let total = self.duration();
        if total <= 0 {
            return Err(EngineError::Export("timeline is empty".into()));
        }

        // H.264 4:2:0 needs even dimensions; round down so a caller passing an
        // odd preview size still exports.
        let width = settings.width & !1;
        let height = settings.height & !1;
        if width == 0 || height == 0 {
            return Err(EngineError::Export(
                "export resolution must be at least 2x2".into(),
            ));
        }

        let fps = self.project().timeline().frame_rate;
        let config = ExportConfig {
            width,
            height,
            frame_rate_num: fps.num,
            frame_rate_den: fps.den,
            quality: settings.quality,
            bitrate: settings.bitrate,
            hardware: settings.hardware,
        };

        // Give the export the CPU: pause background transcodes for its duration,
        // and always resume even if a frame fails.
        self.set_background_paused(true);
        let mut run = || -> Result<ExportStats, EngineError> {
            let mut sink = VideoExport::create(output, config)?;
            for frame in 0..total {
                let layers = self.frame_at(frame)?;
                let composite_layers = to_composite_layers(&layers);
                let image = composite(width, height, &composite_layers);
                sink.push_rgba(&image.pixels)?;
                if let Some(cb) = progress.as_deref_mut() {
                    cb(frame + 1, total);
                }
            }
            Ok(sink.finish()?)
        };
        let result = run();
        self.set_background_paused(false);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_decode::{DecodeOptions, Decoder, HwAccel};
    use cutlass_models::{Generator, Rational, TimeRange, TrackKind};

    #[test]
    fn empty_timeline_cannot_export() {
        let mut engine = Engine::new("empty", Rational::FPS_30);
        let out = std::env::temp_dir().join("cutlass_engine_export_empty.mp4");
        let err = engine.export(&out, ExportSettings::new(64, 48));
        assert!(matches!(err, Err(EngineError::Export(_))));
    }

    #[test]
    fn exports_solid_generator_timeline() {
        // A generator-only timeline needs no media/decoder, so this runs on CI
        // without test assets while still exercising the full export loop.
        let mut engine = Engine::new("gen", Rational::FPS_30);
        let track = engine.project_mut().add_track(TrackKind::Video, "V1");
        engine
            .project_mut()
            .add_generated(
                track,
                Generator::SolidColor { rgba: [0, 90, 200, 255] },
                TimeRange::new(0, 12),
            )
            .unwrap();

        let out = std::env::temp_dir().join("cutlass_engine_export_solid.mp4");
        let _ = std::fs::remove_file(&out);

        let mut done = 0i64;
        let mut on_progress = |n: i64, _total: i64| done = n;
        let stats = engine
            .export_with(&out, ExportSettings::new(64, 48), Some(&mut on_progress))
            .expect("export solid timeline");

        assert_eq!(stats.frames, 12);
        assert_eq!(done, 12);
        assert!(out.exists());

        let mut dec = Decoder::open_with(&out, DecodeOptions::default().hw_accel(HwAccel::None))
            .expect("open export");
        let frame = dec.next_frame().expect("decode").expect("a frame");
        assert!(frame.width > 0 && !frame.planes.is_empty());

        let _ = std::fs::remove_file(&out);
    }
}
