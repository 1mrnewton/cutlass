//! [`Media`] pool handles and [`MediaSlice`] content descriptors.

use cutlass_models::{MediaId, MediaKind, TimeRange};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PySlice;

use crate::convert::ticks;
use crate::errors::{CutlassError, model_err};
use crate::project::Project;

/// A probed source file in the project's media pool.
#[pyclass(unsendable)]
pub struct Media {
    project: Py<Project>,
    id: MediaId,
}

/// A trimmed window into a [`Media`] asset (content descriptor, not a copy).
#[pyclass(skip_from_py_object)]
#[derive(Clone, Copy)]
pub struct MediaSlice {
    pub(crate) media: MediaId,
    pub(crate) start_sec: f64,
    pub(crate) end_sec: Option<f64>,
}

impl Media {
    pub(crate) fn new(project: Py<Project>, id: MediaId) -> Self {
        Self { project, id }
    }

    pub(crate) fn id(&self) -> MediaId {
        self.id
    }

    pub(crate) fn project(&self) -> &Py<Project> {
        &self.project
    }

    pub(crate) fn with_project<F, R>(&self, py: Python, f: F) -> PyResult<R>
    where
        F: FnOnce(&mut Project) -> PyResult<R>,
    {
        let mut project = self.project.bind(py).borrow_mut();
        if project.model().media(self.id).is_none() {
            return Err(CutlassError::new_err("stale media handle"));
        }
        f(&mut project)
    }

    pub(crate) fn require(
        project: &Project,
        id: MediaId,
    ) -> PyResult<&cutlass_models::MediaSource> {
        project
            .model()
            .media(id)
            .ok_or_else(|| CutlassError::new_err("stale media handle"))
    }
}

#[pymethods]
impl Media {
    #[getter]
    fn path(&self, py: Python) -> PyResult<String> {
        self.with_project(py, |project| {
            let media = Self::require(project, self.id)?;
            Ok(media.path().display().to_string())
        })
    }

    #[getter]
    fn kind(&self, py: Python) -> PyResult<String> {
        self.with_project(py, |project| {
            let media = Self::require(project, self.id)?;
            Ok(match media.kind() {
                MediaKind::Video => "video",
                MediaKind::Audio => "audio",
                MediaKind::Image => "image",
            }
            .to_string())
        })
    }

    #[getter]
    fn duration(&self, py: Python) -> PyResult<f64> {
        self.with_project(py, |project| {
            let media = Self::require(project, self.id)?;
            Ok(media.duration.seconds())
        })
    }

    #[getter]
    fn size(&self, py: Python) -> PyResult<(u32, u32)> {
        self.with_project(py, |project| {
            let media = Self::require(project, self.id)?;
            Ok((media.width, media.height))
        })
    }

    #[getter]
    fn fps(&self, py: Python) -> PyResult<f64> {
        self.with_project(py, |project| {
            let media = Self::require(project, self.id)?;
            Ok(media.frame_rate.as_f64())
        })
    }

    #[getter]
    fn has_audio(&self, py: Python) -> PyResult<bool> {
        self.with_project(py, |project| {
            let media = Self::require(project, self.id)?;
            Ok(media.has_audio)
        })
    }

    #[pyo3(signature = (start, end = None))]
    fn subclip(&self, start: f64, end: Option<f64>) -> MediaSlice {
        MediaSlice {
            media: self.id,
            start_sec: start,
            end_sec: end,
        }
    }

    /// `m[3.0:8.0]` — slice sugar for [`Media::subclip`], in seconds.
    /// Negative bounds count back from the media's end.
    fn __getitem__(&self, py: Python, slice: &Bound<'_, PySlice>) -> PyResult<MediaSlice> {
        if !slice.getattr("step")?.is_none() {
            return Err(PyValueError::new_err("media slices do not support a step"));
        }
        let duration = self.duration(py)?;
        let resolve = |bound: Bound<'_, PyAny>, default: f64| -> PyResult<f64> {
            if bound.is_none() {
                return Ok(default);
            }
            let v: f64 = bound.extract().map_err(|_| {
                PyValueError::new_err("media slice bounds must be numbers of seconds")
            })?;
            Ok(if v < 0.0 { duration + v } else { v })
        };
        let start = resolve(slice.getattr("start")?, 0.0)?;
        let stop = resolve(slice.getattr("stop")?, duration)?;
        Ok(MediaSlice {
            media: self.id,
            start_sec: start,
            end_sec: Some(stop),
        })
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        self.with_project(py, |project| {
            let media = Self::require(project, self.id)?;
            let name = media
                .path()
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("?");
            let kind = match media.kind() {
                MediaKind::Video => "video",
                MediaKind::Audio => "audio",
                MediaKind::Image => "image",
            };
            Ok(format!(
                "Media({kind} {:.3}s {}x{} @{:.3} '{name}')",
                media.duration.seconds(),
                media.width,
                media.height,
                media.frame_rate.as_f64()
            ))
        })
    }
}

#[pymethods]
impl MediaSlice {
    fn __repr__(&self) -> String {
        match self.end_sec {
            Some(end) => format!("MediaSlice({:.3}, {:.3})", self.start_sec, end),
            None => format!("MediaSlice({:.3}, end)", self.start_sec),
        }
    }
}

/// Resolve a slice's source range at placement time.
pub(crate) fn slice_source_range(
    project: &Project,
    slice: &MediaSlice,
    duration_override: Option<f64>,
) -> PyResult<TimeRange> {
    let media = Media::require(project, slice.media)?;
    let rate = media.frame_rate;
    let start = ticks(slice.start_sec, rate);
    let end_tick = if let Some(dur) = duration_override {
        start + ticks(dur, rate).max(1)
    } else {
        match slice.end_sec {
            Some(end) => ticks(end, rate),
            None => media.duration.value,
        }
    };
    if start < 0 || end_tick <= start {
        return Err(PyValueError::new_err("invalid source window"));
    }
    if end_tick > media.duration.value && !media.is_image {
        return Err(model_err(cutlass_models::ModelError::SourceOutOfBounds));
    }
    Ok(TimeRange::at_rate(start, (end_tick - start).max(1), rate))
}

/// Full source range for a media asset.
pub(crate) fn full_source_range(project: &Project, id: MediaId) -> PyResult<TimeRange> {
    let media = Media::require(project, id)?;
    Ok(media.full_range())
}
