//! [`Clip`] timeline placement handles.

use cutlass_models::{
    AnimationRef, AnimationSlot, ClipId, ClipParam, ClipTransform, CropRect, Easing, Generator,
    ParamValue, RationalTime, Shape, ShapeParam,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::content::apply_text_style;
use crate::convert::{
    clip_key_time, parse_color, parse_easing, parse_keyframe_pairs, seconds, span, speed_from_f64,
    speed_to_f64, ticks, time_at,
};
use crate::effect::Effect;
use crate::errors::{CutlassError, model_err};
use crate::media::Media;
use crate::project::Project;
use crate::track::Track;

/// A clip placement on a track.
#[pyclass(unsendable)]
pub struct Clip {
    project: Py<Project>,
    id: ClipId,
}

impl Clip {
    pub(crate) fn new(project: Py<Project>, id: ClipId) -> Self {
        Self { project, id }
    }

    pub(crate) fn id(&self) -> ClipId {
        self.id
    }

    pub(crate) fn with_project<F, R>(&self, py: Python, f: F) -> PyResult<R>
    where
        F: FnOnce(&mut Project) -> PyResult<R>,
    {
        let mut project = self.project.bind(py).borrow_mut();
        if project.model().clip(self.id).is_none() {
            return Err(CutlassError::new_err("stale clip handle"));
        }
        f(&mut project)
    }

    pub(crate) fn require(project: &Project, id: ClipId) -> PyResult<&cutlass_models::Clip> {
        project
            .model()
            .clip(id)
            .ok_or_else(|| CutlassError::new_err("stale clip handle"))
    }

    /// `(start, duration)` of the clip in timeline ticks.
    pub(crate) fn span_ticks(project: &Project, id: ClipId) -> PyResult<(i64, i64)> {
        let tl = Self::require(project, id)?.timeline;
        Ok((tl.start.value, tl.duration.value))
    }

    fn set_transform_constant(
        project: &mut Project,
        id: ClipId,
        transform: ClipTransform,
    ) -> PyResult<()> {
        project
            .model_mut()
            .set_transform(id, transform, None)
            .map_err(model_err)
    }

    fn current_transform(project: &Project, id: ClipId) -> PyResult<ClipTransform> {
        let clip = Self::require(project, id)?;
        Ok(clip.transform.sample(0))
    }
}

#[pymethods]
impl Clip {
    #[getter]
    fn start(&self, py: Python) -> PyResult<f64> {
        self.with_project(py, |project| {
            let clip = Self::require(project, self.id)?;
            Ok(clip.timeline.start.seconds())
        })
    }

    #[getter]
    fn end(&self, py: Python) -> PyResult<f64> {
        self.with_project(py, |project| {
            let clip = Self::require(project, self.id)?;
            Ok(clip.end().map_err(model_err)?.seconds())
        })
    }

    #[getter]
    fn duration(&self, py: Python) -> PyResult<f64> {
        self.with_project(py, |project| {
            let clip = Self::require(project, self.id)?;
            Ok(clip.timeline.duration.seconds())
        })
    }

    #[getter]
    fn track(&self, py: Python) -> PyResult<Track> {
        self.with_project(py, |project| {
            let track_id = project
                .model()
                .timeline()
                .track_of(self.id)
                .ok_or_else(|| CutlassError::new_err("stale clip handle"))?;
            Ok(Track::new(self.project.clone_ref(py), track_id))
        })
    }

    #[getter]
    fn media(&self, py: Python) -> PyResult<Option<Media>> {
        self.with_project(py, |project| {
            let clip = Self::require(project, self.id)?;
            Ok(clip
                .media()
                .map(|id| Media::new(self.project.clone_ref(py), id)))
        })
    }

    #[getter]
    fn source_start(&self, py: Python) -> PyResult<Option<f64>> {
        self.with_project(py, |project| {
            let clip = Self::require(project, self.id)?;
            Ok(clip.source_range().map(|r| r.start.seconds()))
        })
    }

    #[getter]
    fn source_duration(&self, py: Python) -> PyResult<Option<f64>> {
        self.with_project(py, |project| {
            let clip = Self::require(project, self.id)?;
            Ok(clip.source_range().map(|r| r.duration.seconds()))
        })
    }

    #[getter]
    fn speed(&self, py: Python) -> PyResult<f64> {
        self.with_project(py, |project| {
            let clip = Self::require(project, self.id)?;
            Ok(speed_to_f64(clip.speed))
        })
    }

    #[getter]
    fn reversed(&self, py: Python) -> PyResult<bool> {
        self.with_project(py, |project| Ok(Self::require(project, self.id)?.reversed))
    }

    fn split(&self, py: Python, at: f64) -> PyResult<Clip> {
        self.with_project(py, |project| {
            let rate = project.rate();
            let new_id = project
                .model_mut()
                .split_clip(self.id, time_at(at, rate))
                .map_err(model_err)?;
            Ok(Clip::new(self.project.clone_ref(py), new_id))
        })
    }

    #[pyo3(signature = (start = None, end = None))]
    fn trim(&self, py: Python, start: Option<f64>, end: Option<f64>) -> PyResult<()> {
        self.with_project(py, |project| {
            let rate = project.rate();
            let clip = Self::require(project, self.id)?;
            let start = start.unwrap_or_else(|| clip.timeline.start.seconds());
            let end = match end {
                Some(e) => e,
                None => clip.end().map_err(model_err)?.seconds(),
            };
            if end <= start {
                return Err(PyValueError::new_err("end must be greater than start"));
            }
            let timeline = span(start, end - start, rate);
            project
                .model_mut()
                .trim_clip(self.id, timeline)
                .map_err(model_err)
        })
    }

    #[pyo3(name = "move", signature = (start, track = None))]
    fn move_(&self, py: Python, start: f64, track: Option<&Track>) -> PyResult<()> {
        self.with_project(py, |project| {
            let rate = project.rate();
            let to_track = match track {
                Some(t) => {
                    if !t.project().is(self.project.bind(py)) {
                        return Err(CutlassError::new_err(
                            "track belongs to a different project",
                        ));
                    }
                    t.id()
                }
                None => project
                    .model()
                    .timeline()
                    .track_of(self.id)
                    .ok_or_else(|| CutlassError::new_err("stale clip handle"))?,
            };
            project
                .model_mut()
                .move_clip(self.id, to_track, time_at(start, rate))
                .map_err(model_err)
        })
    }

    fn delete(&self, py: Python) -> PyResult<()> {
        self.with_project(py, |project| {
            project.model_mut().timeline_mut().remove_clip(self.id);
            Ok(())
        })
    }

    fn ripple_delete(&self, py: Python) -> PyResult<()> {
        self.with_project(py, |project| {
            project
                .model_mut()
                .ripple_delete(self.id)
                .map_err(model_err)?;
            Ok(())
        })
    }

    #[getter]
    fn position(&self, py: Python) -> PyResult<(f32, f32)> {
        self.with_project(py, |project| {
            let t = Self::current_transform(project, self.id)?;
            Ok((t.position[0], t.position[1]))
        })
    }

    #[setter]
    fn set_position(&self, py: Python, value: (f32, f32)) -> PyResult<()> {
        self.with_project(py, |project| {
            let mut t = Self::current_transform(project, self.id)?;
            t.position = [value.0, value.1];
            Self::set_transform_constant(project, self.id, t)
        })
    }

    #[getter]
    fn anchor(&self, py: Python) -> PyResult<(f32, f32)> {
        self.with_project(py, |project| {
            let t = Self::current_transform(project, self.id)?;
            Ok((t.anchor_point[0], t.anchor_point[1]))
        })
    }

    #[setter]
    fn set_anchor(&self, py: Python, value: (f32, f32)) -> PyResult<()> {
        self.with_project(py, |project| {
            let mut t = Self::current_transform(project, self.id)?;
            t.anchor_point = [value.0, value.1];
            Self::set_transform_constant(project, self.id, t)
        })
    }

    #[getter]
    fn scale(&self, py: Python) -> PyResult<f32> {
        self.with_project(py, |project| {
            Ok(Self::current_transform(project, self.id)?.scale)
        })
    }

    #[setter]
    fn set_scale(&self, py: Python, value: f32) -> PyResult<()> {
        self.with_project(py, |project| {
            let mut t = Self::current_transform(project, self.id)?;
            t.scale = value;
            Self::set_transform_constant(project, self.id, t)
        })
    }

    #[getter]
    fn rotation(&self, py: Python) -> PyResult<f32> {
        self.with_project(py, |project| {
            Ok(Self::current_transform(project, self.id)?.rotation)
        })
    }

    #[setter]
    fn set_rotation(&self, py: Python, value: f32) -> PyResult<()> {
        self.with_project(py, |project| {
            let mut t = Self::current_transform(project, self.id)?;
            t.rotation = value;
            Self::set_transform_constant(project, self.id, t)
        })
    }

    #[getter]
    fn opacity(&self, py: Python) -> PyResult<f32> {
        self.with_project(py, |project| {
            Ok(Self::current_transform(project, self.id)?.opacity)
        })
    }

    #[setter]
    fn set_opacity(&self, py: Python, value: f32) -> PyResult<()> {
        self.with_project(py, |project| {
            let mut t = Self::current_transform(project, self.id)?;
            t.opacity = value;
            Self::set_transform_constant(project, self.id, t)
        })
    }

    #[getter]
    fn animation_in(&self, py: Python) -> PyResult<Option<String>> {
        self.with_project(py, |project| {
            Ok(Self::require(project, self.id)?
                .animation_in
                .as_ref()
                .map(|a| a.id.clone()))
        })
    }

    #[getter]
    fn animation_out(&self, py: Python) -> PyResult<Option<String>> {
        self.with_project(py, |project| {
            Ok(Self::require(project, self.id)?
                .animation_out
                .as_ref()
                .map(|a| a.id.clone()))
        })
    }

    #[getter]
    fn animation_combo(&self, py: Python) -> PyResult<Option<String>> {
        self.with_project(py, |project| {
            Ok(Self::require(project, self.id)?
                .animation_combo
                .as_ref()
                .map(|a| a.id.clone()))
        })
    }

    #[pyo3(signature = (slot, animation_id=None))]
    fn set_animation(
        &self,
        py: Python,
        slot: &str,
        animation_id: Option<&str>,
    ) -> PyResult<()> {
        let animation_slot = parse_animation_slot(slot)?;
        let animation = animation_id.map(AnimationRef::new);
        self.with_project(py, |project| {
            project
                .model_mut()
                .set_clip_animation(self.id, animation_slot, animation)
                .map_err(model_err)
        })
    }

    fn transform_at(&self, py: Python, t: f64) -> PyResult<((f32, f32), f32, f32, f32)> {
        self.with_project(py, |project| {
            let clip = Self::require(project, self.id)?;
            let tick = ticks(t, project.rate());
            let xf = clip.transform.sample(tick);
            Ok((
                (xf.position[0], xf.position[1]),
                xf.scale,
                xf.rotation,
                xf.opacity,
            ))
        })
    }

    #[pyo3(signature = (**kwargs))]
    fn animate(&self, py: Python, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let dict = kwargs
            .ok_or_else(|| PyValueError::new_err("animate() requires at least one property"))?;
        let default_easing = match dict.get_item("easing")? {
            Some(v) => parse_easing(&v)?,
            None => Easing::Linear,
        };
        let at_override: Option<f64> = match dict.get_item("at")? {
            Some(v) => Some(
                v.extract()
                    .map_err(|_| PyValueError::new_err("at= must be a number of seconds"))?,
            ),
            None => None,
        };
        self.with_project(py, |project| {
            let clip_span = Self::span_ticks(project, self.id)?;
            let rate = project.rate();
            let mut applied = 0usize;
            for (key, value) in dict.iter() {
                let name: String = key.extract()?;
                if name == "at" || name == "easing" {
                    continue;
                }
                apply_clip_animation(
                    project,
                    self.id,
                    clip_span,
                    rate,
                    &name,
                    &value,
                    at_override,
                    default_easing,
                )?;
                applied += 1;
            }
            if applied == 0 {
                return Err(PyValueError::new_err(
                    "animate() requires at least one property",
                ));
            }
            Ok(())
        })
    }

    #[pyo3(signature = (name, *, at))]
    fn remove_keyframe(&self, py: Python, name: &str, at: f64) -> PyResult<()> {
        self.with_project(py, |project| {
            let (clip_start, clip_dur) = Self::span_ticks(project, self.id)?;
            let abs = clip_key_time(clip_start, clip_dur, at, project.rate());
            let param = clip_param(name)?;
            project
                .model_mut()
                .remove_param_keyframe(self.id, param, abs)
                .map_err(model_err)
        })
    }

    #[pyo3(signature = (*names))]
    fn clear_animation(&self, py: Python, names: &Bound<'_, PyTuple>) -> PyResult<()> {
        let list: Vec<String> = names.extract().map_err(|_| {
            PyValueError::new_err("clear_animation expects property names as strings")
        })?;
        if list.is_empty() {
            return Err(PyValueError::new_err(
                "clear_animation requires at least one property name",
            ));
        }
        self.with_project(py, |project| {
            for name in list {
                let param = clip_param(&name)?;
                let value = sample_param_at_clip_start(project, self.id, &param)?;
                project
                    .model_mut()
                    .set_param_constant(self.id, param, value)
                    .map_err(model_err)?;
            }
            Ok(())
        })
    }

    #[getter]
    fn volume(&self, py: Python) -> PyResult<f32> {
        self.with_project(py, |project| {
            let clip = Self::require(project, self.id)?;
            Ok(clip.volume.sample(0))
        })
    }

    #[setter]
    fn set_volume(&self, py: Python, value: f32) -> PyResult<()> {
        self.with_project(py, |project| {
            let rate = project.rate();
            project
                .model_mut()
                .set_clip_audio(
                    self.id,
                    Some(value),
                    RationalTime::zero(rate),
                    RationalTime::zero(rate),
                )
                .map_err(model_err)
        })
    }

    #[getter]
    fn fade_in(&self, py: Python) -> PyResult<f64> {
        self.with_project(py, |project| {
            let clip = Self::require(project, self.id)?;
            Ok(seconds(clip.fade_in, project.rate()))
        })
    }

    #[setter]
    fn set_fade_in(&self, py: Python, value: f64) -> PyResult<()> {
        self.with_project(py, |project| {
            let clip = Self::require(project, self.id)?;
            let rate = project.rate();
            let fade_out = RationalTime::new(clip.fade_out, rate);
            project
                .model_mut()
                .set_clip_audio(self.id, None, time_at(value, rate), fade_out)
                .map_err(model_err)
        })
    }

    #[getter]
    fn fade_out(&self, py: Python) -> PyResult<f64> {
        self.with_project(py, |project| {
            let clip = Self::require(project, self.id)?;
            Ok(seconds(clip.fade_out, project.rate()))
        })
    }

    #[setter]
    fn set_fade_out(&self, py: Python, value: f64) -> PyResult<()> {
        self.with_project(py, |project| {
            let clip = Self::require(project, self.id)?;
            let rate = project.rate();
            let fade_in = RationalTime::new(clip.fade_in, rate);
            project
                .model_mut()
                .set_clip_audio(self.id, None, fade_in, time_at(value, rate))
                .map_err(model_err)
        })
    }

    #[pyo3(signature = (factor, reverse = false))]
    fn set_speed(&self, py: Python, factor: f64, reverse: bool) -> PyResult<()> {
        self.with_project(py, |project| {
            let speed = speed_from_f64(factor)?;
            project
                .model_mut()
                .set_clip_speed(self.id, speed, reverse)
                .map_err(model_err)
        })
    }

    #[pyo3(signature = (x = 0.0, y = 0.0, w = 1.0, h = 1.0, flip_h = false, flip_v = false))]
    #[allow(clippy::too_many_arguments)]
    fn crop(
        &self,
        py: Python,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        flip_h: bool,
        flip_v: bool,
    ) -> PyResult<()> {
        self.with_project(py, |project| {
            project
                .model_mut()
                .set_clip_crop(self.id, CropRect { x, y, w, h }, flip_h, flip_v, None)
                .map_err(model_err)
        })
    }

    #[pyo3(signature = (effect_id, **params))]
    fn add_effect(
        &self,
        py: Python,
        effect_id: &str,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Effect> {
        self.with_project(py, |project| {
            let index = project
                .model_mut()
                .add_effect(self.id, effect_id)
                .map_err(model_err)?;
            if let Some(dict) = params {
                let fx = Effect::new(self.project.clone_ref(py), self.id, index);
                for (key, value) in dict.iter() {
                    let name: String = key.extract()?;
                    let v: f32 = value.extract()?;
                    fx.set_param(project, &name, v)?;
                }
            }
            Ok(Effect::new(self.project.clone_ref(py), self.id, index))
        })
    }

    #[getter]
    fn effects(&self, py: Python) -> PyResult<Vec<Effect>> {
        self.with_project(py, |project| {
            let clip = Self::require(project, self.id)?;
            Ok((0..clip.effects.len())
                .map(|i| Effect::new(self.project.clone_ref(py), self.id, i))
                .collect())
        })
    }

    fn remove_effect(&self, py: Python, effect: &Effect) -> PyResult<()> {
        effect.check_clip(self)?;
        self.with_project(py, |project| {
            project
                .model_mut()
                .remove_effect(self.id, effect.index())
                .map_err(model_err)
        })
    }

    #[pyo3(signature = (transition_id, duration = None))]
    fn transition(&self, py: Python, transition_id: &str, duration: Option<f64>) -> PyResult<()> {
        self.with_project(py, |project| {
            project
                .model_mut()
                .add_transition(self.id, transition_id)
                .map_err(model_err)?;
            if let Some(dur) = duration {
                let ticks = ticks(dur, project.rate()).max(1);
                project
                    .model_mut()
                    .set_transition_duration(self.id, ticks)
                    .map_err(model_err)?;
            }
            Ok(())
        })
    }

    fn remove_transition(&self, py: Python) -> PyResult<()> {
        self.with_project(py, |project| {
            project
                .model_mut()
                .remove_transition(self.id)
                .map_err(model_err)
        })
    }

    #[getter]
    fn text(&self, py: Python) -> PyResult<String> {
        self.with_project(py, |project| {
            let clip = Self::require(project, self.id)?;
            match &clip.content {
                cutlass_models::ClipSource::Generated(Generator::Text { content, .. }) => {
                    Ok(content.clone())
                }
                _ => Err(PyValueError::new_err("clip is not a text clip")),
            }
        })
    }

    #[setter]
    fn set_text(&self, py: Python, content: String) -> PyResult<()> {
        self.with_project(py, |project| {
            let clip = Self::require(project, self.id)?;
            let cutlass_models::ClipSource::Generated(Generator::Text { style, .. }) =
                &clip.content
            else {
                return Err(PyValueError::new_err("clip is not a text clip"));
            };
            let generator = Generator::Text {
                content,
                style: style.clone(),
            };
            project
                .model_mut()
                .set_generator(self.id, generator)
                .map_err(model_err)
        })
    }

    #[pyo3(signature = (**kwargs))]
    fn set_style(&self, py: Python, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let kwargs =
            kwargs.ok_or_else(|| PyValueError::new_err("set_style requires keyword arguments"))?;
        self.with_project(py, |project| {
            enum Target {
                Text,
                Shape,
            }
            let target = match &Self::require(project, self.id)?.content {
                cutlass_models::ClipSource::Generated(Generator::Text { .. }) => Target::Text,
                cutlass_models::ClipSource::Generated(Generator::Shape { .. }) => Target::Shape,
                _ => {
                    return Err(PyValueError::new_err(
                        "set_style applies to text or shape clips",
                    ));
                }
            };
            match target {
                Target::Text => {
                    let clip = Self::require(project, self.id)?;
                    let cutlass_models::ClipSource::Generated(Generator::Text { content, style }) =
                        &clip.content
                    else {
                        unreachable!("target checked above");
                    };
                    let text = content.clone();
                    let mut new_style = style.clone();
                    apply_text_style(&mut new_style, kwargs)?;
                    project
                        .model_mut()
                        .set_generator(
                            self.id,
                            Generator::Text {
                                content: text,
                                style: new_style,
                            },
                        )
                        .map_err(model_err)
                }
                Target::Shape => {
                    for (key, value) in kwargs.iter() {
                        let key: String = key.extract()?;
                        let (param, val) = shape_style_param(&key, &value)?;
                        project
                            .model_mut()
                            .set_param_constant(self.id, ClipParam::Shape { param }, val)
                            .map_err(model_err)?;
                    }
                    Ok(())
                }
            }
        })
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        self.with_project(py, |project| {
            let clip = Self::require(project, self.id)?;
            Ok(format!(
                "Clip(start={:.3}s, duration={:.3}s, id={})",
                clip.timeline.start.seconds(),
                clip.timeline.duration.seconds(),
                self.id.raw()
            ))
        })
    }
}

fn clip_param(name: &str) -> PyResult<ClipParam> {
    Ok(match name {
        "position" => ClipParam::Position,
        "anchor" | "anchor_point" => ClipParam::AnchorPoint,
        "scale" => ClipParam::Scale,
        "rotation" => ClipParam::Rotation,
        "opacity" => ClipParam::Opacity,
        "volume" => ClipParam::Volume,
        "width" => ClipParam::Shape {
            param: ShapeParam::Width,
        },
        "height" => ClipParam::Shape {
            param: ShapeParam::Height,
        },
        "corner_radius" => ClipParam::Shape {
            param: ShapeParam::CornerRadius,
        },
        "inner_ratio" => ClipParam::Shape {
            param: ShapeParam::InnerRatio,
        },
        "fill" | "color" => ClipParam::Shape {
            param: ShapeParam::Fill,
        },
        "stroke_color" => ClipParam::Shape {
            param: ShapeParam::StrokeColor,
        },
        "stroke_width" => ClipParam::Shape {
            param: ShapeParam::StrokeWidth,
        },
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown animatable property {other:?}"
            )));
        }
    })
}

/// A `set_style` keyword on a shape clip → the shape parameter + constant value.
fn shape_style_param(name: &str, value: &Bound<'_, PyAny>) -> PyResult<(ShapeParam, ParamValue)> {
    let param = match name {
        "color" | "fill" => ShapeParam::Fill,
        "width" => ShapeParam::Width,
        "height" => ShapeParam::Height,
        "corner_radius" => ShapeParam::CornerRadius,
        "inner_ratio" => ShapeParam::InnerRatio,
        "stroke_color" => ShapeParam::StrokeColor,
        "stroke_width" => ShapeParam::StrokeWidth,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown shape style parameter {other:?}"
            )));
        }
    };
    let value = match param {
        ShapeParam::Fill | ShapeParam::StrokeColor => ParamValue::Color(parse_color(value)?),
        _ => ParamValue::Scalar(value.extract::<f32>()?),
    };
    Ok((param, value))
}

fn param_value_for(param: ClipParam, value: &Bound<'_, PyAny>) -> PyResult<ParamValue> {
    match param {
        ClipParam::Position | ClipParam::AnchorPoint => {
            let (x, y) = value.extract::<(f32, f32)>()?;
            Ok(ParamValue::Vec2([x, y]))
        }
        ClipParam::Volume | ClipParam::Scale | ClipParam::Rotation | ClipParam::Opacity => {
            Ok(ParamValue::Scalar(value.extract::<f32>()?))
        }
        ClipParam::Shape { param: shape_param } => match shape_param {
            ShapeParam::Fill | ShapeParam::StrokeColor => {
                Ok(ParamValue::Color(parse_color(value)?))
            }
            _ => Ok(ParamValue::Scalar(value.extract::<f32>()?)),
        },
        ClipParam::Effect { .. } => Err(PyValueError::new_err(
            "use Effect.animate for effect parameters",
        )),
        ClipParam::Speed => Err(PyValueError::new_err("speed curves are not exposed yet")),
    }
}

fn sample_param_at_clip_start(
    project: &Project,
    id: ClipId,
    param: &ClipParam,
) -> PyResult<ParamValue> {
    let clip = Clip::require(project, id)?;
    match param {
        ClipParam::Position => {
            let t = clip.transform.sample(0);
            Ok(ParamValue::Vec2(t.position))
        }
        ClipParam::AnchorPoint => {
            let t = clip.transform.sample(0);
            Ok(ParamValue::Vec2(t.anchor_point))
        }
        ClipParam::Scale => Ok(ParamValue::Scalar(clip.transform.sample(0).scale)),
        ClipParam::Rotation => Ok(ParamValue::Scalar(clip.transform.sample(0).rotation)),
        ClipParam::Opacity => Ok(ParamValue::Scalar(clip.transform.sample(0).opacity)),
        ClipParam::Volume => Ok(ParamValue::Scalar(clip.volume.sample(0))),
        ClipParam::Shape { param: shape_param } => {
            let cutlass_models::ClipSource::Generated(Generator::Shape {
                shape,
                rgba,
                width,
                height,
                corner_radius,
                stroke,
            }) = &clip.content
            else {
                return Err(PyValueError::new_err("clip is not a shape"));
            };
            match shape_param {
                ShapeParam::Fill => Ok(ParamValue::Color(rgba.sample(0))),
                ShapeParam::Width => Ok(ParamValue::Scalar(width.sample(0))),
                ShapeParam::Height => Ok(ParamValue::Scalar(height.sample(0))),
                ShapeParam::CornerRadius => Ok(ParamValue::Scalar(corner_radius.sample(0))),
                ShapeParam::InnerRatio => match shape {
                    Shape::Star { inner_ratio, .. } => {
                        Ok(ParamValue::Scalar(inner_ratio.sample(0)))
                    }
                    _ => Err(PyValueError::new_err(
                        "inner_ratio applies only to star shapes",
                    )),
                },
                ShapeParam::StrokeWidth | ShapeParam::StrokeColor => match stroke {
                    Some(s) => Ok(match shape_param {
                        ShapeParam::StrokeWidth => ParamValue::Scalar(s.width.sample(0)),
                        _ => ParamValue::Color(s.rgba.sample(0)),
                    }),
                    None => Err(PyValueError::new_err("shape has no stroke")),
                },
            }
        }
        _ => Err(PyValueError::new_err("unsupported parameter")),
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_clip_animation(
    project: &mut Project,
    clip_id: ClipId,
    clip_span: (i64, i64),
    rate: cutlass_models::Rational,
    name: &str,
    value: &Bound<'_, PyAny>,
    at: Option<f64>,
    default_easing: Easing,
) -> PyResult<()> {
    let (clip_start, clip_dur) = clip_span;
    let param = clip_param(name)?;
    if value.is_instance_of::<PyList>() {
        let list = value.cast::<PyList>()?;
        let pairs = parse_keyframe_pairs(value.py(), list, default_easing, |v| {
            param_value_for(param, v)
        })?;
        for (rel, val, easing) in pairs {
            let abs = clip_key_time(clip_start, clip_dur, rel, rate);
            project
                .model_mut()
                .set_param_keyframe(clip_id, param, abs, val, easing)
                .map_err(model_err)?;
        }
        return Ok(());
    }
    let at = at
        .ok_or_else(|| PyValueError::new_err(format!("single keyframe for {name} requires at=")))?;
    let val = param_value_for(param, value)?;
    let abs = clip_key_time(clip_start, clip_dur, at, rate);
    project
        .model_mut()
        .set_param_keyframe(clip_id, param, abs, val, default_easing)
        .map_err(model_err)
}

fn parse_animation_slot(slot: &str) -> PyResult<AnimationSlot> {
    match slot {
        "in" => Ok(AnimationSlot::In),
        "out" => Ok(AnimationSlot::Out),
        "combo" => Ok(AnimationSlot::Combo),
        _ => Err(PyValueError::new_err(
            "slot must be one of: 'in', 'out', 'combo'",
        )),
    }
}
