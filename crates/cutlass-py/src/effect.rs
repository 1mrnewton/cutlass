//! [`Effect`] instances on a clip.

use cutlass_models::{ClipId, ClipParam, ParamValue};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::clip::Clip;
use crate::convert::{clip_key_time, parse_easing, parse_keyframe_pairs};
use crate::errors::{CutlassError, model_err};
use crate::project::Project;

/// An effect in a clip's effect chain.
#[pyclass(unsendable)]
pub struct Effect {
    project: Py<Project>,
    clip_id: ClipId,
    index: usize,
}

impl Effect {
    pub(crate) fn new(project: Py<Project>, clip_id: ClipId, index: usize) -> Self {
        Self {
            project,
            clip_id,
            index,
        }
    }

    pub(crate) fn index(&self) -> usize {
        self.index
    }

    pub(crate) fn check_clip(&self, clip: &Clip) -> PyResult<()> {
        if clip.id() != self.clip_id {
            return Err(CutlassError::new_err("effect does not belong to this clip"));
        }
        Ok(())
    }

    pub(crate) fn set_param(&self, project: &mut Project, name: &str, value: f32) -> PyResult<()> {
        let clip = project
            .model()
            .clip(self.clip_id)
            .ok_or_else(|| CutlassError::new_err("stale effect handle"))?;
        let effect = clip
            .effects
            .get(self.index)
            .ok_or_else(|| CutlassError::new_err("stale effect handle"))?;
        let spec = effect.spec().map_err(model_err)?;
        let param_idx = spec
            .params
            .iter()
            .position(|p| p.name == name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown effect parameter {name:?}")))?;
        project
            .model_mut()
            .set_effect_param(self.clip_id, self.index, param_idx, value)
            .map_err(model_err)
    }

    fn with_project<F, R>(&self, py: Python, f: F) -> PyResult<R>
    where
        F: FnOnce(&mut Project) -> PyResult<R>,
    {
        let mut project = self.project.bind(py).borrow_mut();
        let clip = project
            .model()
            .clip(self.clip_id)
            .ok_or_else(|| CutlassError::new_err("stale effect handle"))?;
        if self.index >= clip.effects.len() {
            return Err(CutlassError::new_err("stale effect handle"));
        }
        f(&mut project)
    }

    fn param_index(
        project: &Project,
        clip_id: ClipId,
        effect_index: usize,
        name: &str,
    ) -> PyResult<usize> {
        let clip = project
            .model()
            .clip(clip_id)
            .ok_or_else(|| CutlassError::new_err("stale effect handle"))?;
        let effect = clip
            .effects
            .get(effect_index)
            .ok_or_else(|| CutlassError::new_err("stale effect handle"))?;
        let spec = effect.spec().map_err(model_err)?;
        spec.params
            .iter()
            .position(|p| p.name == name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown effect parameter {name:?}")))
    }
}

#[pymethods]
impl Effect {
    fn __getitem__(&self, py: Python, name: &str) -> PyResult<f32> {
        self.with_project(py, |project| {
            let clip = project
                .model()
                .clip(self.clip_id)
                .ok_or_else(|| CutlassError::new_err("stale effect handle"))?;
            let effect = &clip.effects[self.index];
            let spec = effect.spec().map_err(model_err)?;
            let param = spec.param(name).ok_or_else(|| {
                PyValueError::new_err(format!("unknown effect parameter {name:?}"))
            })?;
            Ok(effect
                .sample_param(param.name, 0.0)
                .unwrap_or(param.default))
        })
    }

    fn __setitem__(&self, py: Python, name: &str, value: f32) -> PyResult<()> {
        self.with_project(py, |project| self.set_param(project, name, value))
    }

    #[pyo3(signature = (**kwargs))]
    fn animate(&self, py: Python, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let dict = kwargs
            .ok_or_else(|| PyValueError::new_err("animate() requires at least one parameter"))?;
        let default_easing = match dict.get_item("easing")? {
            Some(v) => parse_easing(&v)?,
            None => cutlass_models::Easing::Linear,
        };
        let at_override: Option<f64> = match dict.get_item("at")? {
            Some(v) => Some(
                v.extract()
                    .map_err(|_| PyValueError::new_err("at= must be a number of seconds"))?,
            ),
            None => None,
        };
        self.with_project(py, |project| {
            let (clip_start, clip_dur) = Clip::span_ticks(project, self.clip_id)?;
            let rate = project.rate();
            let mut applied = 0usize;
            for (key, value) in dict.iter() {
                let name: String = key.extract()?;
                if name == "easing" || name == "at" {
                    continue;
                }
                let param_idx = Self::param_index(project, self.clip_id, self.index, &name)?;
                let param = ClipParam::Effect {
                    effect: self.index as u32,
                    param: param_idx as u32,
                };
                if value.is_instance_of::<PyList>() {
                    let list = value.cast::<PyList>()?;
                    let pairs = parse_keyframe_pairs(py, list, default_easing, |v| {
                        Ok(ParamValue::Scalar(v.extract::<f32>()?))
                    })?;
                    for (rel, val, easing) in pairs {
                        let abs = clip_key_time(clip_start, clip_dur, rel, rate);
                        project
                            .model_mut()
                            .set_param_keyframe(self.clip_id, param, abs, val, easing, None)
                            .map_err(model_err)?;
                    }
                } else {
                    let at = at_override.ok_or_else(|| {
                        PyValueError::new_err(format!("single keyframe for {name} requires at="))
                    })?;
                    let val = ParamValue::Scalar(value.extract::<f32>()?);
                    let abs = clip_key_time(clip_start, clip_dur, at, rate);
                    project
                        .model_mut()
                        .set_param_keyframe(self.clip_id, param, abs, val, default_easing, None)
                        .map_err(model_err)?;
                }
                applied += 1;
            }
            if applied == 0 {
                return Err(PyValueError::new_err(
                    "animate() requires at least one parameter",
                ));
            }
            Ok(())
        })
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        self.with_project(py, |project| {
            let clip = project
                .model()
                .clip(self.clip_id)
                .ok_or_else(|| CutlassError::new_err("stale effect handle"))?;
            let effect = &clip.effects[self.index];
            Ok(format!(
                "Effect({:?}, clip={}, index={})",
                effect.effect_id,
                self.clip_id.raw(),
                self.index
            ))
        })
    }
}
