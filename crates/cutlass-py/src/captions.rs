//! Caption groups: a run of cue clips bound by one shared look.
//!
//! A cue *is* a text clip, so `group.cues` hands back ordinary [`Clip`]s that
//! trim, move, and animate like any other. What the group adds is one place to
//! restyle all of them, the segmentation rules an import or re-flow follows,
//! and the karaoke highlight.

use std::path::Path;

use cutlass_captions::{
    ImportOptions, Placement, parse_subtitles, place_subtitles, subtitles_from_clips, write_srt,
    write_vtt,
};
use cutlass_models::{
    CaptionCueSpec, CaptionFileFormat, CaptionGroupId, CaptionGroupSpec, CaptionHighlightMode,
    CaptionSource, CaptionStyleScope, Param, Rational, TextCase, TrackId, caption_template_spec,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

use crate::clip::Clip;
use crate::convert::{parse_color, span};
use crate::errors::{CutlassError, model_err};
use crate::project::Project;

/// A caption group handle: the shared style, rules, and highlight behind a set
/// of cue clips.
#[pyclass(unsendable)]
pub struct Captions {
    project: Py<Project>,
    id: CaptionGroupId,
}

impl Captions {
    pub(crate) fn new(project: Py<Project>, id: CaptionGroupId) -> Self {
        Self { project, id }
    }

    fn with_project<F, R>(&self, py: Python, f: F) -> PyResult<R>
    where
        F: FnOnce(&mut Project) -> PyResult<R>,
    {
        let mut project = self.project.bind(py).borrow_mut();
        if project.model().timeline().caption_group(self.id).is_none() {
            return Err(CutlassError::new_err("stale captions handle"));
        }
        f(&mut project)
    }

    /// Cue text and placement in timeline order — the shape both subtitle
    /// writers consume.
    fn subtitle_cues(&self, project: &Project) -> Vec<cutlass_captions::SubtitleCue> {
        let placement = Placement::at_rate(project.rate());
        subtitles_from_clips(
            project
                .model()
                .timeline()
                .caption_cues(self.id)
                .into_iter()
                .map(|clip| {
                    (
                        clip.timeline,
                        clip.text_content().unwrap_or_default().to_string(),
                    )
                }),
            placement,
        )
    }
}

#[pymethods]
impl Captions {
    #[getter]
    fn label(&self, py: Python) -> PyResult<String> {
        self.with_project(py, |project| Ok(group(project, self.id)?.label.clone()))
    }

    #[setter]
    fn set_label(&self, py: Python, label: String) -> PyResult<()> {
        self.with_project(py, |project| {
            project
                .model_mut()
                .set_caption_group_label(self.id, label)
                .map_err(model_err)
        })
    }

    /// Caption template id the look came from, or `None` once hand-styled.
    #[getter]
    fn template(&self, py: Python) -> PyResult<Option<String>> {
        self.with_project(py, |project| Ok(group(project, self.id)?.template.clone()))
    }

    #[setter]
    fn set_template(&self, py: Python, template: &str) -> PyResult<()> {
        self.with_project(py, |project| {
            project
                .model_mut()
                .set_caption_group_template(self.id, template)
                .map_err(model_err)
        })
    }

    /// The cue clips, in timeline order.
    #[getter]
    fn cues(&self, py: Python) -> PyResult<Vec<Clip>> {
        self.with_project(py, |project| {
            Ok(project
                .model()
                .timeline()
                .caption_cue_ids(self.id)
                .into_iter()
                .map(|id| Clip::new(self.project.clone_ref(py), id))
                .collect())
        })
    }

    /// Restyle every line. Omitted arguments keep their current value; pass
    /// `keep_overrides=True` to leave individually styled lines alone.
    #[pyo3(signature = (
        font = None, size = None, fill = None, bold = None, italic = None,
        uppercase = None, position_y = None, scale = None, keep_overrides = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn style(
        &self,
        py: Python,
        font: Option<String>,
        size: Option<f32>,
        fill: Option<&Bound<'_, PyAny>>,
        bold: Option<bool>,
        italic: Option<bool>,
        uppercase: Option<bool>,
        position_y: Option<f32>,
        scale: Option<f32>,
        keep_overrides: bool,
    ) -> PyResult<()> {
        let fill = fill.map(parse_color).transpose()?;
        self.with_project(py, |project| {
            let mut style = group(project, self.id)?.style.clone();
            if let Some(font) = font {
                style.text.font = font;
            }
            if let Some(size) = size {
                style.text.size = Param::Constant(size);
            }
            if let Some(fill) = fill {
                style.text.fill = Param::Constant(fill);
            }
            if let Some(bold) = bold {
                style.text.bold = bold;
            }
            if let Some(italic) = italic {
                style.text.italic = italic;
            }
            if let Some(uppercase) = uppercase {
                style.text.case = if uppercase {
                    TextCase::Upper
                } else {
                    TextCase::Normal
                };
            }
            if let Some(y) = position_y {
                style.position[1] = y;
            }
            if let Some(scale) = scale {
                style.scale = scale;
            }
            let scope = if keep_overrides {
                CaptionStyleScope::KeepOverrides
            } else {
                CaptionStyleScope::All
            };
            project
                .model_mut()
                .set_caption_group_style(self.id, style, scope)
                .map_err(model_err)
        })
    }

    /// Set the line-breaking rules and safe area. These govern segmentation and
    /// re-flow; existing lines move into the new safe area but are not re-split.
    #[pyo3(signature = (
        max_chars_per_line = None, max_lines = None, min_duration = None,
        max_duration = None, min_gap = None, safe_area_bottom = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn layout(
        &self,
        py: Python,
        max_chars_per_line: Option<u16>,
        max_lines: Option<u8>,
        min_duration: Option<f64>,
        max_duration: Option<f64>,
        min_gap: Option<f64>,
        safe_area_bottom: Option<f32>,
    ) -> PyResult<()> {
        self.with_project(py, |project| {
            let mut layout = group(project, self.id)?.layout;
            if let Some(chars) = max_chars_per_line {
                layout.max_chars_per_line = chars;
            }
            if let Some(lines) = max_lines {
                layout.max_lines = lines;
            }
            if let Some(seconds) = min_duration {
                layout.min_duration_ms = milliseconds(seconds, "min_duration")?;
            }
            if let Some(seconds) = max_duration {
                layout.max_duration_ms = milliseconds(seconds, "max_duration")?;
            }
            if let Some(seconds) = min_gap {
                layout.min_gap_ms = milliseconds(seconds, "min_gap")?;
            }
            if let Some(safe_area) = safe_area_bottom {
                layout.safe_area_bottom = safe_area;
            }
            project
                .model_mut()
                .set_caption_group_layout(self.id, layout)
                .map_err(model_err)
        })
    }

    /// Karaoke highlighting: `mode` is `"off"`, `"word"`, or `"line"`.
    ///
    /// Highlighting needs per-word timings, which auto-generated and imported
    /// captions carry; on hand-written lines the setting is stored and simply
    /// does not draw.
    #[pyo3(signature = (mode = "word", fill = None, plate = None, plate_radius = None, scale = None))]
    fn highlight(
        &self,
        py: Python,
        mode: &str,
        fill: Option<&Bound<'_, PyAny>>,
        plate: Option<&Bound<'_, PyAny>>,
        plate_radius: Option<f32>,
        scale: Option<f32>,
    ) -> PyResult<()> {
        let mode = match mode.to_ascii_lowercase().as_str() {
            "off" | "none" => CaptionHighlightMode::Off,
            "word" => CaptionHighlightMode::Word,
            "line" => CaptionHighlightMode::Line,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown highlight mode {other:?} (use 'off', 'word', or 'line')"
                )));
            }
        };
        let fill = fill.map(parse_color).transpose()?;
        let plate = plate.map(parse_color).transpose()?;
        self.with_project(py, |project| {
            if mode == CaptionHighlightMode::Off {
                return project
                    .model_mut()
                    .set_caption_highlight(self.id, None)
                    .map_err(model_err);
            }
            let mut highlight = group(project, self.id)?
                .highlight
                .clone()
                .unwrap_or_default();
            highlight.mode = mode;
            if let Some(fill) = fill {
                highlight.fill = fill;
            }
            if let Some(plate) = plate {
                // A fully transparent plate means "no card", not an invisible one.
                highlight.plate = (plate[3] > 0).then_some(plate);
            }
            if let Some(radius) = plate_radius {
                highlight.plate_radius = radius;
            }
            if let Some(scale) = scale {
                highlight.scale = scale;
            }
            project
                .model_mut()
                .set_caption_highlight(self.id, Some(highlight))
                .map_err(model_err)
        })
    }

    /// This group as SubRip text. With `path`, also writes the sidecar file.
    #[pyo3(signature = (path = None))]
    fn to_srt(&self, py: Python, path: Option<&str>) -> PyResult<String> {
        self.write_subtitles(py, CaptionFileFormat::Srt, path)
    }

    /// This group as WebVTT text. With `path`, also writes the sidecar file.
    #[pyo3(signature = (path = None))]
    fn to_vtt(&self, py: Python, path: Option<&str>) -> PyResult<String> {
        self.write_subtitles(py, CaptionFileFormat::Vtt, path)
    }

    /// Remove the group and every line it owns.
    fn remove(&self, py: Python) -> PyResult<()> {
        self.with_project(py, |project| {
            project
                .model_mut()
                .remove_caption_group(self.id)
                .map(|_| ())
                .map_err(model_err)
        })
    }

    fn __len__(&self, py: Python) -> PyResult<usize> {
        self.with_project(py, |project| {
            Ok(project.model().timeline().caption_cue_ids(self.id).len())
        })
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let cues = self.cues(py)?;
        cues.into_pyobject(py)?.call_method0("__iter__")
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        self.with_project(py, |project| {
            let cues = project.model().timeline().caption_cue_ids(self.id).len();
            let group = group(project, self.id)?;
            Ok(format!(
                "Captions({:?}, cues={cues}, template={})",
                group.label,
                group.template.as_deref().unwrap_or("None"),
            ))
        })
    }
}

impl Captions {
    fn write_subtitles(
        &self,
        py: Python,
        format: CaptionFileFormat,
        path: Option<&str>,
    ) -> PyResult<String> {
        let text = self.with_project(py, |project| {
            let cues = self.subtitle_cues(project);
            Ok(match format {
                CaptionFileFormat::Srt => write_srt(&cues),
                CaptionFileFormat::Vtt => write_vtt(&cues),
            })
        })?;
        if let Some(path) = path {
            std::fs::write(path, &text)
                .map_err(|e| CutlassError::new_err(format!("{path}: {e}")))?;
        }
        Ok(text)
    }
}

fn group(project: &Project, id: CaptionGroupId) -> PyResult<&cutlass_models::CaptionGroup> {
    project
        .model()
        .timeline()
        .caption_group(id)
        .ok_or_else(|| CutlassError::new_err("stale captions handle"))
}

/// Seconds → whole milliseconds, the unit caption layout rules use.
fn milliseconds(value: f64, what: &str) -> PyResult<u32> {
    let ms = value * 1000.0;
    if !ms.is_finite() || !(0.0..=f64::from(u32::MAX)).contains(&ms) {
        return Err(PyValueError::new_err(format!("{what} is out of range")));
    }
    Ok(ms.round() as u32)
}

/// Place hand-written cues on `track` as one group.
///
/// Each cue is `(text, start, duration)` or a dict with `text`, `start`, and
/// either `duration` or `end` — all in seconds.
pub(crate) fn add_captions(
    project: &mut Project,
    project_py: Py<Project>,
    track: TrackId,
    cues: &Bound<'_, PyAny>,
    template: Option<&str>,
    label: Option<&str>,
) -> PyResult<Captions> {
    let rate = project.rate();
    let mut specs = Vec::new();
    for item in cues.try_iter()? {
        specs.push(cue_spec(&item?, rate)?);
    }
    if specs.is_empty() {
        return Err(PyValueError::new_err("add_captions needs at least one cue"));
    }
    if let Some(template) = template
        && caption_template_spec(template).is_none()
    {
        return Err(PyValueError::new_err(format!(
            "unknown caption template {template:?}"
        )));
    }
    let spec = CaptionGroupSpec {
        track,
        label: label.unwrap_or("Captions").to_string(),
        source: CaptionSource::Manual,
        template: template.map(str::to_string),
        style: None,
        layout: None,
        highlight: None,
    };
    let (id, _) = project
        .model_mut()
        .add_caption_group(&spec, &specs)
        .map_err(model_err)?;
    Ok(Captions::new(project_py, id))
}

/// Import an `.srt` or `.vtt` file onto `track` as one group, starting at
/// `start` seconds on the timeline.
#[allow(clippy::too_many_arguments)]
pub(crate) fn import_subtitles(
    project: &mut Project,
    project_py: Py<Project>,
    track: TrackId,
    path: &str,
    start: f64,
    template: Option<&str>,
    label: Option<&str>,
    rewrap: bool,
) -> PyResult<Captions> {
    if start < 0.0 {
        return Err(PyValueError::new_err("start must be >= 0"));
    }
    let text =
        std::fs::read_to_string(path).map_err(|e| CutlassError::new_err(format!("{path}: {e}")))?;
    let (format, cues) =
        parse_subtitles(&text).map_err(|e| CutlassError::new_err(format!("{path}: {e}")))?;
    if cues.is_empty() {
        return Err(CutlassError::new_err(format!("{path} has no caption cues")));
    }

    let rate = project.rate();
    let template_spec = match template {
        Some(id) => Some(
            caption_template_spec(id)
                .ok_or_else(|| PyValueError::new_err(format!("unknown caption template {id:?}")))?,
        ),
        None => None,
    };
    let layout = template_spec.map(|spec| spec.layout()).unwrap_or_default();
    let placement = Placement::new(rate, crate::convert::ticks(start, rate));
    let mut options = ImportOptions::new(placement).with_layout(layout);
    options.rewrap = rewrap;
    // A subtitle file carries no word timings; estimating them lets an imported
    // group drive the karaoke highlight.
    options.estimate_words = true;
    let specs =
        place_subtitles(&cues, &options).map_err(|e| CutlassError::new_err(e.to_string()))?;

    let file = Path::new(path)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string());
    let spec = CaptionGroupSpec {
        track,
        label: label.unwrap_or(&file).to_string(),
        source: CaptionSource::Imported { format },
        template: template.map(str::to_string),
        style: None,
        layout: None,
        highlight: None,
    };
    let (id, _) = project
        .model_mut()
        .add_caption_group(&spec, &specs)
        .map_err(model_err)?;
    Ok(Captions::new(project_py, id))
}

/// One cue from a `(text, start, duration)` tuple or a dict.
fn cue_spec(item: &Bound<'_, PyAny>, rate: Rational) -> PyResult<CaptionCueSpec> {
    if let Ok(dict) = item.cast::<PyDict>() {
        let text: String = dict
            .get_item("text")?
            .ok_or_else(|| PyValueError::new_err("a caption cue dict needs 'text'"))?
            .extract()?;
        let start: f64 = dict
            .get_item("start")?
            .ok_or_else(|| PyValueError::new_err("a caption cue dict needs 'start'"))?
            .extract()?;
        let duration = match dict.get_item("duration")? {
            Some(duration) => duration.extract::<f64>()?,
            None => {
                let end: f64 = dict
                    .get_item("end")?
                    .ok_or_else(|| {
                        PyValueError::new_err("a caption cue dict needs 'duration' or 'end'")
                    })?
                    .extract()?;
                end - start
            }
        };
        return spec_from(text, start, duration, rate);
    }
    if let Ok((text, start, duration)) = item.extract::<(String, f64, f64)>() {
        return spec_from(text, start, duration, rate);
    }
    // A bare string would silently become a zero-length cue at 0 s; name the
    // shape instead.
    if item.is_instance_of::<PyString>() {
        return Err(PyValueError::new_err(
            "a caption cue needs a time: use (text, start, duration)",
        ));
    }
    Err(PyValueError::new_err(
        "a caption cue must be (text, start, duration) or \
         {'text': …, 'start': …, 'duration': …}",
    ))
}

fn spec_from(text: String, start: f64, duration: f64, rate: Rational) -> PyResult<CaptionCueSpec> {
    if start < 0.0 {
        return Err(PyValueError::new_err("cue start must be >= 0"));
    }
    if duration <= 0.0 || !duration.is_finite() {
        return Err(PyValueError::new_err("cue duration must be > 0"));
    }
    let spec = CaptionCueSpec::new(text, span(start, duration, rate));
    spec.validate().map_err(model_err)?;
    Ok(spec)
}

/// Every caption group in the project, in id order.
pub(crate) fn groups(project: &Project) -> Vec<CaptionGroupId> {
    project
        .model()
        .timeline()
        .caption_groups_ordered()
        .iter()
        .map(|group| group.id)
        .collect()
}

/// The embedded caption template catalog, for `cutlass.caption_templates()`.
pub(crate) fn template_dicts(py: Python) -> PyResult<Vec<Py<PyDict>>> {
    cutlass_models::caption_template_catalog()
        .iter()
        .map(|spec| {
            let dict = PyDict::new(py);
            dict.set_item("id", spec.id)?;
            dict.set_item("label", spec.label)?;
            dict.set_item("highlight", spec.highlight().map(|h| h.mode.id()))?;
            dict.set_item("max_chars_per_line", spec.layout().max_chars_per_line)?;
            dict.set_item("max_lines", spec.layout().max_lines)?;
            Ok(dict.into())
        })
        .collect()
}
