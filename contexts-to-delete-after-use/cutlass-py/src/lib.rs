//! Python bindings for the Cutlass video engine.
//!
//! A small, MoviePy-style API over the pure-Rust pipeline: build a [`Project`]
//! (canvas + generated/solid/text content), pull a frame as a NumPy array, or
//! export the whole timeline to an `.mp4` via the platform-native encoder.
//!
//! The heavy machinery stays in Rust — this layer only wraps
//! [`cutlass_models::Project`] (editing) and [`cutlass_render`] (render/export);
//! it deliberately exposes no command/enum internals.
//!
//! ```python
//! import cutlass
//! p = cutlass.Project("demo", fps=30)
//! p.set_canvas("16:9", background=(20, 20, 30))
//! p.add_solid((38, 42, 64, 255), start=0.0, duration=2.0)
//! p.add_text("Cutlass", start=0.0, duration=2.0, size=220.0)
//! frame = p.get_frame(0.5)        # numpy uint8 array, shape (H, W, 4)
//! n = p.export("out.mp4")         # native H.264/mp4 on Apple
//! ```

use std::path::Path;

use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyArray3};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use cutlass_models::{
    CanvasAspect, CanvasSettings, ClipId, Generator, Project as Model, Rational, RationalTime,
    TextStyle, TimeRange, TrackKind,
};
use cutlass_render::{Renderer, canvas_size, export_to_file};

/// Map any displayable error into a Python `RuntimeError`.
fn runtime_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Integer frame rate → exact [`Rational`] (`fps/1`), clamped to ≥ 1.
fn rate_from_fps(fps: u32) -> Rational {
    Rational::new(fps.max(1) as i32, 1)
}

/// Seconds → tick count at `rate`, rounded, clamped to ≥ 0.
fn ticks(seconds: f64, rate: Rational) -> i64 {
    if rate.den == 0 {
        return 0;
    }
    let t = seconds * f64::from(rate.num) / f64::from(rate.den);
    t.round().max(0.0) as i64
}

/// A `[start, start+duration)` timeline range from seconds (duration ≥ 1 tick,
/// since zero-length generated clips are rejected by the model).
fn span(start: f64, duration: f64, rate: Rational) -> TimeRange {
    let start = ticks(start, rate);
    let dur = ticks(duration, rate).max(1);
    TimeRange::at_rate(start, dur, rate)
}

/// A scriptable Cutlass project: editing state plus a lazily created renderer.
///
/// `unsendable`: the renderer owns GPU/codec handles bound to the creating
/// thread, so the object must be used from one thread (enforced by PyO3).
#[pyclass(unsendable)]
pub struct Project {
    model: Model,
    /// Built on first `get_frame`/`export` — bringing up a headless GPU is
    /// expensive, so we don't pay for it on construction or pure editing.
    renderer: Option<Renderer>,
}

#[pymethods]
impl Project {
    /// Create an empty project named `name` at `fps` frames per second.
    #[new]
    #[pyo3(signature = (name, fps = 30))]
    fn new(name: &str, fps: u32) -> Self {
        Self {
            model: Model::new(name, rate_from_fps(fps)),
            renderer: None,
        }
    }

    /// Load a project previously written with [`save`](Self::save).
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let model = Model::load_from_file(Path::new(path)).map_err(runtime_err)?;
        Ok(Self {
            model,
            renderer: None,
        })
    }

    /// Save the project to `path` (Cutlass JSON document).
    fn save(&self, path: &str) -> PyResult<()> {
        self.model
            .save_to_file(Path::new(path))
            .map_err(runtime_err)
    }

    /// Set the canvas aspect (`"auto"`, `"16:9"`, `"9:16"`, `"1:1"`, `"4:5"`,
    /// `"21:9"`) and optional opaque background `(r, g, b)`.
    #[pyo3(signature = (aspect, background = None))]
    fn set_canvas(&mut self, aspect: &str, background: Option<(u8, u8, u8)>) -> PyResult<()> {
        let aspect = CanvasAspect::from_name(aspect)
            .ok_or_else(|| PyValueError::new_err(format!("unknown aspect {aspect:?}")))?;
        let (r, g, b) = background.unwrap_or((0, 0, 0));
        self.model.timeline_mut().set_canvas(CanvasSettings {
            aspect,
            background: [r, g, b],
        });
        Ok(())
    }

    /// Add a new track of `kind` (`"video"`, `"audio"`, `"text"`, `"sticker"`,
    /// `"effect"`, `"filter"`, `"adjustment"`) and return its id.
    #[pyo3(signature = (kind, name = ""))]
    fn add_track(&mut self, kind: &str, name: &str) -> PyResult<u64> {
        Ok(self.model.add_track(parse_track_kind(kind)?, name).raw())
    }

    /// Add a solid-color clip `(r, g, b, a)` on its own graphics lane and return
    /// its clip id. Lanes stack in insertion order (later calls draw on top).
    #[pyo3(signature = (color, start = 0.0, duration = 1.0))]
    fn add_solid(
        &mut self,
        color: (u8, u8, u8, u8),
        start: f64,
        duration: f64,
    ) -> PyResult<u64> {
        let rate = self.rate();
        let track = self.model.add_track(TrackKind::Sticker, "solid");
        let (r, g, b, a) = color;
        let clip = self
            .model
            .add_generated(
                track,
                Generator::SolidColor {
                    rgba: [r, g, b, a],
                },
                span(start, duration, rate),
            )
            .map_err(runtime_err)?;
        Ok(clip.raw())
    }

    /// Add a text clip on its own lane and return its clip id.
    #[pyo3(signature = (text, start = 0.0, duration = 1.0, size = 96.0, color = (255, 255, 255, 255)))]
    fn add_text(
        &mut self,
        text: &str,
        start: f64,
        duration: f64,
        size: f32,
        color: (u8, u8, u8, u8),
    ) -> PyResult<u64> {
        let rate = self.rate();
        let track = self.model.add_track(TrackKind::Text, "text");
        let (r, g, b, a) = color;
        let style = TextStyle {
            size,
            fill: [r, g, b, a],
            ..TextStyle::default()
        };
        let clip = self
            .model
            .add_generated(
                track,
                Generator::Text {
                    content: text.to_string(),
                    style,
                },
                span(start, duration, rate),
            )
            .map_err(runtime_err)?;
        Ok(clip.raw())
    }

    /// Split the clip `clip_id` at timeline time `at` (seconds), returning the
    /// id of the new right-hand clip.
    fn split(&mut self, clip_id: u64, at: f64) -> PyResult<u64> {
        let rate = self.rate();
        let new_id = self
            .model
            .split_clip(ClipId::from_raw(clip_id), RationalTime::new(ticks(at, rate), rate))
            .map_err(runtime_err)?;
        Ok(new_id.raw())
    }

    /// Register a font face (TTF/OTF) for deterministic text rendering.
    fn load_font(&mut self, path: &str) -> PyResult<()> {
        let data = std::fs::read(path).map_err(runtime_err)?;
        self.ensure_renderer()?;
        self.renderer
            .as_mut()
            .expect("renderer initialized")
            .load_font(data);
        Ok(())
    }

    /// Composite the frame at timeline time `t` (seconds) and return it as a
    /// NumPy `uint8` array of shape `(height, width, 4)` in RGBA order.
    fn get_frame<'py>(&mut self, py: Python<'py>, t: f64) -> PyResult<Bound<'py, PyArray3<u8>>> {
        let rate = self.rate();
        self.ensure_renderer()?;
        let image = self
            .renderer
            .as_mut()
            .expect("renderer initialized")
            .render_frame(&self.model, RationalTime::new(ticks(t, rate), rate))
            .map_err(runtime_err)?;
        let shape = (image.height as usize, image.width as usize, 4);
        let array = Array3::from_shape_vec(shape, image.pixels)
            .map_err(|e| PyRuntimeError::new_err(format!("frame buffer/shape mismatch: {e}")))?;
        Ok(array.into_pyarray(py))
    }

    /// Export the whole timeline to a video file at `path`, returning the number
    /// of frames written. Uses the platform-native encoder (H.264/mp4 on Apple);
    /// raises on platforms without one.
    fn export(&mut self, path: &str) -> PyResult<u64> {
        self.ensure_renderer()?;
        let renderer = self.renderer.as_mut().expect("renderer initialized");
        export_to_file(renderer, &self.model, Path::new(path)).map_err(runtime_err)
    }

    /// Timeline duration in seconds.
    #[getter]
    fn duration(&self) -> f64 {
        self.model.timeline().duration().seconds()
    }

    /// Output canvas size as `(width, height)` in pixels.
    #[getter]
    fn size(&self) -> (u32, u32) {
        canvas_size(&self.model)
    }

    /// Frame rate in frames per second (float).
    #[getter]
    fn fps(&self) -> f64 {
        self.model.timeline().frame_rate.as_f64()
    }

    fn __repr__(&self) -> String {
        let (w, h) = canvas_size(&self.model);
        format!(
            "Project(size=({w}, {h}), fps={:.3}, duration={:.3}s)",
            self.fps(),
            self.duration()
        )
    }
}

impl Project {
    /// The timeline frame rate (all timeline-space conversions use it).
    fn rate(&self) -> Rational {
        self.model.timeline().frame_rate
    }

    /// Bring up the headless renderer on first use.
    fn ensure_renderer(&mut self) -> PyResult<()> {
        if self.renderer.is_none() {
            self.renderer = Some(Renderer::new_headless().map_err(runtime_err)?);
        }
        Ok(())
    }
}

/// Parse a track-kind string into a [`TrackKind`].
fn parse_track_kind(kind: &str) -> PyResult<TrackKind> {
    Ok(match kind.to_ascii_lowercase().as_str() {
        "video" => TrackKind::Video,
        "audio" => TrackKind::Audio,
        "text" => TrackKind::Text,
        "sticker" => TrackKind::Sticker,
        "effect" => TrackKind::Effect,
        "filter" => TrackKind::Filter,
        "adjustment" => TrackKind::Adjustment,
        other => return Err(PyValueError::new_err(format!("unknown track kind {other:?}"))),
    })
}

/// The `cutlass` Python module.
#[pymodule]
fn cutlass(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Project>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
