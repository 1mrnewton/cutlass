//! [`Track`] timeline lane handles.

use cutlass_models::{TimeRange, TrackId};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::clip::Clip;
use crate::content::generator_from;
use crate::convert::{seconds, span, ticks, time_at, track_kind_name};
use crate::errors::{CutlassError, model_err};
use crate::media::{Media, MediaSlice, full_source_range, slice_source_range};
use crate::project::Project;

/// A single timeline lane holding non-overlapping clips.
#[pyclass(unsendable)]
pub struct Track {
    project: Py<Project>,
    id: TrackId,
}

impl Track {
    pub(crate) fn new(project: Py<Project>, id: TrackId) -> Self {
        Self { project, id }
    }

    pub(crate) fn id(&self) -> TrackId {
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
        if project.model().timeline().track(self.id).is_none() {
            return Err(CutlassError::new_err("stale track handle"));
        }
        f(&mut project)
    }

    pub(crate) fn require(project: &Project, id: TrackId) -> PyResult<&cutlass_models::Track> {
        project
            .model()
            .timeline()
            .track(id)
            .ok_or_else(|| CutlassError::new_err("stale track handle"))
    }
}

#[pymethods]
impl Track {
    #[pyo3(signature = (content, start, duration = None))]
    fn add(
        &self,
        py: Python,
        content: &Bound<'_, PyAny>,
        start: f64,
        duration: Option<f64>,
    ) -> PyResult<Clip> {
        self.with_project(py, |project| {
            place_content(
                project,
                py,
                self.project.clone_ref(py),
                self.id,
                content,
                start,
                duration,
            )
        })
    }

    #[pyo3(signature = (content, duration = None))]
    fn append(
        &self,
        py: Python,
        content: &Bound<'_, PyAny>,
        duration: Option<f64>,
    ) -> PyResult<Clip> {
        self.with_project(py, |project| {
            let track = Self::require(project, self.id)?;
            let start = seconds(track.content_end(), project.rate());
            place_content(
                project,
                py,
                self.project.clone_ref(py),
                self.id,
                content,
                start,
                duration,
            )
        })
    }

    #[getter]
    fn clips(&self, py: Python) -> PyResult<Vec<Clip>> {
        self.with_project(py, |project| {
            let track = Self::require(project, self.id)?;
            track
                .clips_ordered()
                .iter()
                .map(|c| Ok(Clip::new(self.project.clone_ref(py), c.id)))
                .collect()
        })
    }

    fn clip_at(&self, py: Python, t: f64) -> PyResult<Option<Clip>> {
        self.with_project(py, |project| {
            let track = Self::require(project, self.id)?;
            let pos = time_at(t, project.rate());
            Ok(track
                .clip_at(pos)
                .map_err(model_err)?
                .map(|c| Clip::new(self.project.clone_ref(py), c.id)))
        })
    }

    #[getter]
    fn end(&self, py: Python) -> PyResult<f64> {
        self.with_project(py, |project| {
            let track = Self::require(project, self.id)?;
            Ok(seconds(track.content_end(), project.rate()))
        })
    }

    #[getter]
    fn name(&self, py: Python) -> PyResult<String> {
        self.with_project(py, |project| {
            Ok(Self::require(project, self.id)?.name.clone())
        })
    }

    #[setter]
    fn set_name(&self, py: Python, name: String) -> PyResult<()> {
        self.with_project(py, |project| {
            project
                .model_mut()
                .timeline_mut()
                .track_mut(self.id)
                .expect("track exists")
                .name = name;
            Ok(())
        })
    }

    #[getter]
    fn kind(&self, py: Python) -> PyResult<String> {
        self.with_project(py, |project| {
            Ok(track_kind_name(Self::require(project, self.id)?.kind).to_string())
        })
    }

    #[getter]
    fn enabled(&self, py: Python) -> PyResult<bool> {
        self.with_project(py, |project| Ok(Self::require(project, self.id)?.enabled))
    }

    #[setter]
    fn set_enabled(&self, py: Python, enabled: bool) -> PyResult<()> {
        self.with_project(py, |project| {
            project
                .model_mut()
                .timeline_mut()
                .track_mut(self.id)
                .expect("track exists")
                .enabled = enabled;
            Ok(())
        })
    }

    #[getter]
    fn muted(&self, py: Python) -> PyResult<bool> {
        self.with_project(py, |project| Ok(Self::require(project, self.id)?.muted))
    }

    #[setter]
    fn set_muted(&self, py: Python, muted: bool) -> PyResult<()> {
        self.with_project(py, |project| {
            project
                .model_mut()
                .timeline_mut()
                .track_mut(self.id)
                .expect("track exists")
                .muted = muted;
            Ok(())
        })
    }

    #[getter]
    fn locked(&self, py: Python) -> PyResult<bool> {
        self.with_project(py, |project| Ok(Self::require(project, self.id)?.locked))
    }

    #[setter]
    fn set_locked(&self, py: Python, locked: bool) -> PyResult<()> {
        self.with_project(py, |project| {
            project
                .model_mut()
                .timeline_mut()
                .track_mut(self.id)
                .expect("track exists")
                .locked = locked;
            Ok(())
        })
    }

    fn remove(&self, py: Python) -> PyResult<()> {
        self.with_project(py, |project| {
            project
                .model_mut()
                .timeline_mut()
                .remove_track(self.id)
                .ok_or_else(|| CutlassError::new_err("stale track handle"))?;
            Ok(())
        })
    }

    fn __len__(&self, py: Python) -> PyResult<usize> {
        self.with_project(py, |project| Ok(Self::require(project, self.id)?.len()))
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let clips = self.clips(py)?;
        clips.into_pyobject(py)?.call_method0("__iter__")
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        self.with_project(py, |project| {
            let track = Self::require(project, self.id)?;
            Ok(format!(
                "Track({:?}, kind={}, clips={})",
                track.name,
                track_kind_name(track.kind),
                track.len()
            ))
        })
    }
}

fn place_content(
    project: &mut Project,
    py: Python,
    project_py: Py<Project>,
    track_id: TrackId,
    content: &Bound<'_, PyAny>,
    start: f64,
    duration: Option<f64>,
) -> PyResult<Clip> {
    let rate = project.rate();
    if start < 0.0 {
        return Err(PyValueError::new_err("start must be >= 0"));
    }
    if let Some(d) = duration
        && d <= 0.0
    {
        return Err(PyValueError::new_err("duration must be > 0"));
    }

    if let Ok(media) = content.extract::<PyRef<Media>>() {
        if !media.project().is(project_py.bind(py)) {
            return Err(CutlassError::new_err(
                "media belongs to a different project",
            ));
        }
        // `duration=d` on whole media is shorthand for `subclip(0, d)`.
        let source = match duration {
            Some(d) => {
                let media_rate = Media::require(project, media.id())?.frame_rate;
                TimeRange::at_rate(0, ticks(d, media_rate).max(1), media_rate)
            }
            None => full_source_range(project, media.id())?,
        };
        let clip_id = project
            .model_mut()
            .add_clip(track_id, media.id(), source, time_at(start, rate))
            .map_err(model_err)?;
        return Ok(Clip::new(project_py, clip_id));
    }

    if let Ok(slice) = content.extract::<PyRef<MediaSlice>>() {
        let source = slice_source_range(project, &slice, duration)?;
        let clip_id = project
            .model_mut()
            .add_clip(track_id, slice.media, source, time_at(start, rate))
            .map_err(model_err)?;
        return Ok(Clip::new(project_py, clip_id));
    }

    let generator = generator_from(content)?;
    let dur = duration
        .ok_or_else(|| PyValueError::new_err("duration is required for generated content"))?;
    let timeline = span(start, dur, rate);
    let clip_id = project
        .model_mut()
        .add_generated(track_id, generator, timeline)
        .map_err(model_err)?;
    Ok(Clip::new(project_py, clip_id))
}
