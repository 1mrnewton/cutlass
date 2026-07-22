//! Effects as data (v1 roadmap M4): a clip carries a list of
//! [`EffectInstance`]s, each `{effect_id, params}`. The model never holds
//! shader code — the compositor owns the WGSL and maps ids to GPU passes.
//!
//! The [`effect_catalog`] here is the validation + UI source of truth
//! (display names, parameter defaults / ranges). It is drift-checked against
//! the compositor's renderable descriptors from `cutlass-engine`, so the two
//! crates can never disagree on which ids and parameter names exist.

use serde::{Deserialize, Serialize};

use crate::Map;
use crate::clip::ParamValue;
use crate::error::ModelError;
use crate::param::{Easing, Param};

/// Value type of an effect parameter. Scalars are the common case; colors
/// and 2-d vectors get their own typed maps on `EffectInstance` so serde
/// stays unambiguous.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectParamKind {
    Scalar,
    Vec2,
    Color,
}

/// One parameter of an effect: its stable name, a human label, value kind,
/// and the default + inclusive range commands validate against.
///
/// `min`/`max` bound scalar values and each component of a [`EffectParamKind::Vec2`].
/// They are ignored for [`EffectParamKind::Color`]. `default` is the scalar
/// default; `default_color` / `default_vec2` are used for those kinds and
/// ignored for scalars.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EffectParamSpec {
    pub name: &'static str,
    pub label: &'static str,
    pub kind: EffectParamKind,
    pub default: f32,
    pub min: f32,
    pub max: f32,
    pub default_color: [u8; 4],
    pub default_vec2: [f32; 2],
}

const fn scalar(
    name: &'static str,
    label: &'static str,
    default: f32,
    min: f32,
    max: f32,
) -> EffectParamSpec {
    EffectParamSpec {
        name,
        label,
        kind: EffectParamKind::Scalar,
        default,
        min,
        max,
        default_color: [0, 0, 0, 0],
        default_vec2: [0.0, 0.0],
    }
}

const fn color(name: &'static str, label: &'static str, default_color: [u8; 4]) -> EffectParamSpec {
    EffectParamSpec {
        name,
        label,
        kind: EffectParamKind::Color,
        default: 0.0,
        min: 0.0,
        max: 0.0,
        default_color,
        default_vec2: [0.0, 0.0],
    }
}

const fn vec2(
    name: &'static str,
    label: &'static str,
    default_vec2: [f32; 2],
    min: f32,
    max: f32,
) -> EffectParamSpec {
    EffectParamSpec {
        name,
        label,
        kind: EffectParamKind::Vec2,
        default: 0.0,
        min,
        max,
        default_color: [0, 0, 0, 0],
        default_vec2,
    }
}

/// A catalog entry: an effect id, its display label, and its ordered
/// parameters. The order matches the compositor's uniform slot order (flattened
/// at resolve time: scalar → 1 float, vec2 → 2, color → 4).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EffectSpec {
    pub id: &'static str,
    pub label: &'static str,
    pub params: &'static [EffectParamSpec],
}

impl EffectSpec {
    /// The spec for `name`, or `None`.
    pub fn param(&self, name: &str) -> Option<&'static EffectParamSpec> {
        self.params.iter().find(|p| p.name == name)
    }

    /// The spec at slot `index`, or `None`.
    pub fn param_at(&self, index: usize) -> Option<&'static EffectParamSpec> {
        self.params.get(index)
    }
}

/// The starter pack (M4). Phase 3 extends this list; ids and parameter names
/// must stay in lockstep with `cutlass_compositor::effect_descriptors`.
const CATALOG: &[EffectSpec] = &[
    EffectSpec {
        id: "gaussian_blur",
        label: "Gaussian Blur",
        params: &[scalar("radius", "Radius", 4.0, 0.0, 64.0)],
    },
    EffectSpec {
        id: "vignette",
        label: "Vignette",
        params: &[scalar("amount", "Amount", 0.6, 0.0, 1.0)],
    },
    EffectSpec {
        id: "sharpen",
        label: "Sharpen",
        params: &[scalar("amount", "Amount", 0.5, 0.0, 3.0)],
    },
    EffectSpec {
        id: "pixelate",
        label: "Pixelate",
        params: &[scalar("size", "Cell Size", 8.0, 1.0, 256.0)],
    },
    EffectSpec {
        id: "glitch",
        label: "Glitch",
        params: &[
            scalar("amount", "Amount", 0.5, 0.0, 1.0),
            scalar("seed", "Seed", 0.0, 0.0, 1000.0),
        ],
    },
    EffectSpec {
        id: "chromatic_aberration",
        label: "Chromatic Aberration",
        params: &[scalar("amount", "Amount", 0.4, 0.0, 1.0)],
    },
    EffectSpec {
        id: "grain",
        label: "Film Grain",
        params: &[
            scalar("amount", "Amount", 0.3, 0.0, 1.0),
            scalar("seed", "Seed", 0.0, 0.0, 1000.0),
        ],
    },
    EffectSpec {
        id: "glow",
        label: "Glow",
        params: &[
            scalar("threshold", "Threshold", 0.7, 0.0, 1.0),
            scalar("intensity", "Intensity", 0.8, 0.0, 4.0),
        ],
    },
    EffectSpec {
        id: "zoom_blur",
        label: "Zoom Blur",
        params: &[scalar("amount", "Amount", 0.5, 0.0, 1.0)],
    },
    EffectSpec {
        id: "mirror",
        label: "Mirror",
        params: &[scalar("mode", "Mode", 0.0, 0.0, 3.0)],
    },
    EffectSpec {
        id: "color_overlay",
        label: "Color Overlay",
        params: &[
            color("color", "Color", [255, 0, 0, 255]),
            vec2("offset", "Offset", [0.0, 0.0], -1.0, 1.0),
            scalar("amount", "Amount", 0.5, 0.0, 1.0),
        ],
    },
];

/// Every effect the model knows about (validation + UI browsing).
pub fn effect_catalog() -> &'static [EffectSpec] {
    CATALOG
}

/// The catalog entry for `id`, or `None`.
pub fn effect_spec(id: &str) -> Option<&'static EffectSpec> {
    CATALOG.iter().find(|s| s.id == id)
}

/// An effect placed on a clip. Only parameters that differ from their catalog
/// default are stored (others fall back to the default), so a freshly-added
/// effect serializes to just its id and old files that predate a new
/// parameter keep working.
///
/// Scalar values live in [`Self::params`]; colors and vec2s use dedicated maps
/// so serde never confuses a bare float with a color/vec2 payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectInstance {
    pub effect_id: String,
    /// Explicitly-set scalar parameters, keyed by name; constant or keyframed.
    /// Keyframe ticks are clip-relative, like the transform params.
    #[serde(
        default,
        with = "crate::serde_map",
        skip_serializing_if = "Map::is_empty"
    )]
    pub params: Map<String, Param<f32>>,
    /// Explicitly-set color parameters (`EffectParamKind::Color`).
    #[serde(
        default,
        with = "crate::serde_map",
        skip_serializing_if = "Map::is_empty"
    )]
    pub color_params: Map<String, Param<[u8; 4]>>,
    /// Explicitly-set 2-d vector parameters (`EffectParamKind::Vec2`).
    #[serde(
        default,
        with = "crate::serde_map",
        skip_serializing_if = "Map::is_empty"
    )]
    pub vec2_params: Map<String, Param<[f32; 2]>>,
}

impl EffectInstance {
    /// A new instance of `effect_id` with every parameter at its default.
    pub fn new(effect_id: impl Into<String>) -> Self {
        Self {
            effect_id: effect_id.into(),
            params: Map::default(),
            color_params: Map::default(),
            vec2_params: Map::default(),
        }
    }

    /// The catalog entry for this instance, or an error if the id is unknown.
    pub fn spec(&self) -> Result<&'static EffectSpec, ModelError> {
        effect_spec(&self.effect_id)
            .ok_or_else(|| ModelError::InvalidParam(format!("unknown effect '{}'", self.effect_id)))
    }

    fn param_spec(&self, index: usize) -> Result<&'static EffectParamSpec, ModelError> {
        self.spec()?.param_at(index).ok_or_else(|| {
            ModelError::InvalidParam(format!(
                "effect '{}' has no parameter at index {index}",
                self.effect_id
            ))
        })
    }

    fn expect_kind(
        &self,
        pspec: &EffectParamSpec,
        kind: EffectParamKind,
    ) -> Result<(), ModelError> {
        if pspec.kind == kind {
            Ok(())
        } else {
            Err(ModelError::InvalidParam(format!(
                "effect parameter '{}' is {:?}, expected {:?}",
                pspec.name, pspec.kind, kind
            )))
        }
    }

    /// Sampled scalar value of `param` at clip-relative fractional `tick`,
    /// falling back to the catalog default when the parameter was never set.
    /// `None` when the effect id or parameter name is unknown, or the slot is
    /// not a scalar.
    pub fn sample_param(&self, param: &str, tick: f64) -> Option<f32> {
        let pspec = self.spec().ok()?.param(param)?;
        if pspec.kind != EffectParamKind::Scalar {
            return None;
        }
        Some(match self.params.get(param) {
            Some(p) => p.sample_at(tick),
            None => pspec.default,
        })
    }

    /// Sampled color value of `param` at clip-relative fractional `tick`.
    pub fn sample_color_param(&self, param: &str, tick: f64) -> Option<[u8; 4]> {
        let pspec = self.spec().ok()?.param(param)?;
        if pspec.kind != EffectParamKind::Color {
            return None;
        }
        Some(match self.color_params.get(param) {
            Some(p) => p.sample_at(tick),
            None => pspec.default_color,
        })
    }

    /// Sampled vec2 value of `param` at clip-relative fractional `tick`.
    pub fn sample_vec2_param(&self, param: &str, tick: f64) -> Option<[f32; 2]> {
        let pspec = self.spec().ok()?.param(param)?;
        if pspec.kind != EffectParamKind::Vec2 {
            return None;
        }
        Some(match self.vec2_params.get(param) {
            Some(p) => p.sample_at(tick),
            None => pspec.default_vec2,
        })
    }

    /// Insert or replace a keyframe on a scalar parameter slot.
    pub fn set_param_keyframe(
        &mut self,
        index: usize,
        tick: i64,
        value: f32,
        easing: Easing,
    ) -> Result<(), ModelError> {
        let pspec = self.param_spec(index)?;
        self.expect_kind(pspec, EffectParamKind::Scalar)?;
        range_check(pspec, value)?;
        easing.validate()?;
        let (name, default) = (pspec.name, pspec.default);
        self.params
            .entry(name.to_string())
            .or_insert(Param::Constant(default))
            .set_keyframe(tick, value, easing);
        Ok(())
    }

    /// Insert or replace a keyframe on a color parameter slot.
    pub fn set_color_param_keyframe(
        &mut self,
        index: usize,
        tick: i64,
        value: [u8; 4],
        easing: Easing,
    ) -> Result<(), ModelError> {
        let pspec = self.param_spec(index)?;
        self.expect_kind(pspec, EffectParamKind::Color)?;
        easing.validate()?;
        let (name, default) = (pspec.name, pspec.default_color);
        self.color_params
            .entry(name.to_string())
            .or_insert(Param::Constant(default))
            .set_keyframe(tick, value, easing);
        Ok(())
    }

    /// Insert or replace a keyframe on a vec2 parameter slot.
    pub fn set_vec2_param_keyframe(
        &mut self,
        index: usize,
        tick: i64,
        value: [f32; 2],
        easing: Easing,
    ) -> Result<(), ModelError> {
        let pspec = self.param_spec(index)?;
        self.expect_kind(pspec, EffectParamKind::Vec2)?;
        vec2_range_check(pspec, value)?;
        easing.validate()?;
        let (name, default) = (pspec.name, pspec.default_vec2);
        self.vec2_params
            .entry(name.to_string())
            .or_insert(Param::Constant(default))
            .set_keyframe(tick, value, easing);
        Ok(())
    }

    /// Remove the keyframe at exactly `tick` on parameter slot `index`.
    pub fn remove_param_keyframe(&mut self, index: usize, tick: i64) -> Result<(), ModelError> {
        let pspec = self.param_spec(index)?;
        let name = pspec.name;
        let removed = match pspec.kind {
            EffectParamKind::Scalar => self
                .params
                .get_mut(name)
                .is_some_and(|p| p.remove_keyframe(tick)),
            EffectParamKind::Color => self
                .color_params
                .get_mut(name)
                .is_some_and(|p| p.remove_keyframe(tick)),
            EffectParamKind::Vec2 => self
                .vec2_params
                .get_mut(name)
                .is_some_and(|p| p.remove_keyframe(tick)),
        };
        if removed {
            Ok(())
        } else {
            Err(ModelError::InvalidParam(format!(
                "no keyframe at tick {tick} on {}.{name}",
                self.effect_id
            )))
        }
    }

    /// Replace a scalar parameter slot with a constant, dropping keyframes.
    pub fn set_param_constant(&mut self, index: usize, value: f32) -> Result<(), ModelError> {
        let pspec = self.param_spec(index)?;
        self.expect_kind(pspec, EffectParamKind::Scalar)?;
        range_check(pspec, value)?;
        self.params
            .insert(pspec.name.to_string(), Param::Constant(value));
        Ok(())
    }

    /// Replace a color parameter slot with a constant, dropping keyframes.
    pub fn set_color_param_constant(
        &mut self,
        index: usize,
        value: [u8; 4],
    ) -> Result<(), ModelError> {
        let pspec = self.param_spec(index)?;
        self.expect_kind(pspec, EffectParamKind::Color)?;
        self.color_params
            .insert(pspec.name.to_string(), Param::Constant(value));
        Ok(())
    }

    /// Replace a vec2 parameter slot with a constant, dropping keyframes.
    pub fn set_vec2_param_constant(
        &mut self,
        index: usize,
        value: [f32; 2],
    ) -> Result<(), ModelError> {
        let pspec = self.param_spec(index)?;
        self.expect_kind(pspec, EffectParamKind::Vec2)?;
        vec2_range_check(pspec, value)?;
        self.vec2_params
            .insert(pspec.name.to_string(), Param::Constant(value));
        Ok(())
    }

    /// Set parameter slot `index` from a [`ParamValue`], dispatching on the
    /// catalog kind.
    pub fn set_param_value_constant(
        &mut self,
        index: usize,
        value: ParamValue,
    ) -> Result<(), ModelError> {
        let pspec = self.param_spec(index)?;
        match pspec.kind {
            EffectParamKind::Scalar => self.set_param_constant(index, expect_scalar(value)?),
            EffectParamKind::Color => self.set_color_param_constant(index, expect_color(value)?),
            EffectParamKind::Vec2 => self.set_vec2_param_constant(index, expect_vec2(value)?),
        }
    }

    /// Insert or replace a keyframe from a [`ParamValue`], dispatching on kind.
    pub fn set_param_value_keyframe(
        &mut self,
        index: usize,
        tick: i64,
        value: ParamValue,
        easing: Easing,
    ) -> Result<(), ModelError> {
        let pspec = self.param_spec(index)?;
        match pspec.kind {
            EffectParamKind::Scalar => {
                self.set_param_keyframe(index, tick, expect_scalar(value)?, easing)
            }
            EffectParamKind::Color => {
                self.set_color_param_keyframe(index, tick, expect_color(value)?, easing)
            }
            EffectParamKind::Vec2 => {
                self.set_vec2_param_keyframe(index, tick, expect_vec2(value)?, easing)
            }
        }
    }

    /// Shift every clip-relative parameter keyframe by `delta` ticks.
    pub(crate) fn shift_param_ticks(&mut self, delta: i64) -> Result<(), ModelError> {
        for param in self.params.values_mut() {
            param.shift_ticks(delta)?;
        }
        for param in self.color_params.values_mut() {
            param.shift_ticks(delta)?;
        }
        for param in self.vec2_params.values_mut() {
            param.shift_ticks(delta)?;
        }
        Ok(())
    }

    /// `Ok` iff the id is known, every set parameter names a real slot of the
    /// matching kind, every curve is structurally sound, and every value lies
    /// in range.
    pub fn validate(&self) -> Result<(), ModelError> {
        let spec = self.spec()?;
        for (name, param) in &self.params {
            let pspec = spec.param(name).ok_or_else(|| {
                ModelError::InvalidParam(format!(
                    "effect '{}' has no parameter '{name}'",
                    self.effect_id
                ))
            })?;
            if pspec.kind != EffectParamKind::Scalar {
                return Err(ModelError::InvalidParam(format!(
                    "effect parameter '{name}' is {:?}, not stored in params",
                    pspec.kind
                )));
            }
            param.validate_shape()?;
            param.for_each_value(|v| range_check(pspec, *v))?;
        }
        for (name, param) in &self.color_params {
            let pspec = spec.param(name).ok_or_else(|| {
                ModelError::InvalidParam(format!(
                    "effect '{}' has no parameter '{name}'",
                    self.effect_id
                ))
            })?;
            if pspec.kind != EffectParamKind::Color {
                return Err(ModelError::InvalidParam(format!(
                    "effect parameter '{name}' is {:?}, not stored in color_params",
                    pspec.kind
                )));
            }
            param.validate_shape()?;
        }
        for (name, param) in &self.vec2_params {
            let pspec = spec.param(name).ok_or_else(|| {
                ModelError::InvalidParam(format!(
                    "effect '{}' has no parameter '{name}'",
                    self.effect_id
                ))
            })?;
            if pspec.kind != EffectParamKind::Vec2 {
                return Err(ModelError::InvalidParam(format!(
                    "effect parameter '{name}' is {:?}, not stored in vec2_params",
                    pspec.kind
                )));
            }
            param.validate_shape()?;
            param.for_each_value(|v| vec2_range_check(pspec, *v))?;
        }
        Ok(())
    }
}

fn range_check(pspec: &EffectParamSpec, value: f32) -> Result<(), ModelError> {
    if !value.is_finite() || value < pspec.min || value > pspec.max {
        return Err(ModelError::InvalidParam(format!(
            "{} = {value} out of range [{}, {}]",
            pspec.name, pspec.min, pspec.max
        )));
    }
    Ok(())
}

fn vec2_range_check(pspec: &EffectParamSpec, value: [f32; 2]) -> Result<(), ModelError> {
    for (i, &component) in value.iter().enumerate() {
        if !component.is_finite() || component < pspec.min || component > pspec.max {
            return Err(ModelError::InvalidParam(format!(
                "{}[{i}] = {component} out of range [{}, {}]",
                pspec.name, pspec.min, pspec.max
            )));
        }
    }
    Ok(())
}

fn expect_scalar(value: ParamValue) -> Result<f32, ModelError> {
    match value {
        ParamValue::Scalar(v) => Ok(v),
        ParamValue::Vec2(_) | ParamValue::Color(_) => {
            Err(ModelError::InvalidParam("expected a scalar value".into()))
        }
    }
}

fn expect_vec2(value: ParamValue) -> Result<[f32; 2], ModelError> {
    match value {
        ParamValue::Vec2(v) => Ok(v),
        ParamValue::Scalar(_) | ParamValue::Color(_) => {
            Err(ModelError::InvalidParam("expected a vec2 value".into()))
        }
    }
}

fn expect_color(value: ParamValue) -> Result<[u8; 4], ModelError> {
    match value {
        ParamValue::Color(v) => Ok(v),
        ParamValue::Scalar(_) | ParamValue::Vec2(_) => {
            Err(ModelError::InvalidParam("expected a color value".into()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_ids_are_unique() {
        let mut ids: Vec<&str> = effect_catalog().iter().map(|s| s.id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), effect_catalog().len());
    }

    #[test]
    fn unknown_effect_fails_validation() {
        assert!(EffectInstance::new("nope").validate().is_err());
        assert!(EffectInstance::new("gaussian_blur").validate().is_ok());
    }

    #[test]
    fn sampled_param_falls_back_to_default() {
        let fx = EffectInstance::new("gaussian_blur");
        assert_eq!(fx.sample_param("radius", 0.0), Some(4.0));
        assert_eq!(fx.sample_param("missing", 0.0), None);
    }

    #[test]
    fn out_of_range_constant_rejected() {
        let mut fx = EffectInstance::new("vignette");
        assert!(fx.set_param_constant(0, 2.0).is_err()); // amount max 1.0
        assert!(fx.set_param_constant(0, 0.5).is_ok());
        assert_eq!(fx.sample_param("amount", 0.0), Some(0.5));
    }

    #[test]
    fn keyframe_roundtrip_on_a_param() {
        let mut fx = EffectInstance::new("gaussian_blur");
        fx.set_param_keyframe(0, 0, 0.0, Easing::Linear).unwrap();
        fx.set_param_keyframe(0, 24, 8.0, Easing::Linear).unwrap();
        assert_eq!(fx.sample_param("radius", 12.0), Some(4.0));
        fx.validate().unwrap();
        fx.remove_param_keyframe(0, 24).unwrap();
        // Removing the last-but-one keyframe leaves a constant.
        assert!(fx.remove_param_keyframe(0, 999).is_err());
    }

    #[test]
    fn legacy_scalar_json_still_loads() {
        let fx: EffectInstance =
            serde_json::from_str(r#"{"effect_id":"gaussian_blur","params":[["radius",4.0]]}"#)
                .unwrap();
        assert_eq!(fx.effect_id, "gaussian_blur");
        assert_eq!(fx.sample_param("radius", 0.0), Some(4.0));
        assert!(fx.color_params.is_empty());
        assert!(fx.vec2_params.is_empty());
        fx.validate().unwrap();
    }

    #[test]
    fn color_params_roundtrip() {
        let mut fx = EffectInstance::new("color_overlay");
        fx.set_color_param_constant(0, [10, 20, 30, 40]).unwrap();
        fx.set_vec2_param_constant(1, [0.25, -0.5]).unwrap();
        let json = serde_json::to_string(&fx).unwrap();
        let loaded: EffectInstance = serde_json::from_str(&json).unwrap();
        assert_eq!(
            loaded.sample_color_param("color", 0.0),
            Some([10, 20, 30, 40])
        );
        assert_eq!(loaded.sample_vec2_param("offset", 0.0), Some([0.25, -0.5]));
        loaded.validate().unwrap();
    }

    #[test]
    fn wrong_kind_set_rejected() {
        let mut fx = EffectInstance::new("color_overlay");
        // color slot is Color — scalar setter rejected.
        assert!(fx.set_param_constant(0, 0.5).is_err());
        // offset slot is Vec2 — color setter rejected.
        assert!(fx.set_color_param_constant(1, [1, 2, 3, 4]).is_err());
        // amount slot is Scalar — vec2 setter rejected.
        assert!(fx.set_vec2_param_constant(2, [0.0, 0.0]).is_err());
        // Scalar name stored in color_params fails validate.
        fx.color_params
            .insert("amount".into(), Param::Constant([1, 2, 3, 4]));
        assert!(fx.validate().is_err());
    }

    #[test]
    fn vec2_component_out_of_range_rejected() {
        let mut fx = EffectInstance::new("color_overlay");
        assert!(fx.set_vec2_param_constant(1, [0.0, 1.5]).is_err());
        assert!(fx.set_vec2_param_constant(1, [-1.0, 1.0]).is_ok());
    }
}
