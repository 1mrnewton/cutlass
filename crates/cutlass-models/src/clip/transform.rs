use serde::{Deserialize, Serialize};

use crate::error::ModelError;
use crate::param::{Easing, Param};

/// Spatial placement of a clip's content on the canvas (CapCut "Basic"
/// transform: position, anchor, scale, rotation, opacity).
///
/// Coordinates are normalized to the canvas so projects survive canvas-size
/// changes: `position` is the offset of the [`anchor_point`] from the canvas
/// center as a fraction of canvas width/height (+x right, +y down — screen
/// convention). With the default center anchor this matches the legacy
/// content-center semantics. `anchor_point` is the pivot within the content
/// bounds (0,0 = top-left, 0.5,0.5 = center). `scale` is uniform with 1.0 =
/// aspect-fit inside the canvas (CapCut's 100%). `rotation` is degrees
/// clockwise about the anchor.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClipTransform {
    /// Anchor offset from canvas center, normalized to canvas dimensions.
    /// `[0.0, 0.0]` = anchor on the canvas center; `[0.5, 0.0]` = anchor on
    /// the right canvas edge.
    pub position: [f32; 2],
    /// Pivot within the content bounds, normalized to the placed size
    /// (+x right, +y down). `[0.5, 0.5]` = content center (default).
    #[serde(default = "default_anchor_point")]
    pub anchor_point: [f32; 2],
    /// Uniform scale; 1.0 aspect-fits the content inside the canvas.
    pub scale: f32,
    /// Clockwise rotation in degrees about the anchor.
    pub rotation: f32,
    /// Layer opacity, 0.0 (transparent) ..= 1.0 (opaque).
    pub opacity: f32,
}

fn default_anchor_point() -> [f32; 2] {
    [0.5, 0.5]
}

impl ClipTransform {
    pub const IDENTITY: Self = Self {
        position: [0.0, 0.0],
        anchor_point: [0.5, 0.5],
        scale: 1.0,
        rotation: 0.0,
        opacity: 1.0,
    };

    pub fn is_identity(&self) -> bool {
        *self == Self::IDENTITY
    }

    /// `Ok` iff every component is finite, scale is positive, and opacity is
    /// within `0..=1` — the invariant [`crate::Project::set_transform`]
    /// enforces before storing.
    pub fn validate(&self) -> Result<(), ModelError> {
        let finite = self.position.iter().all(|v| v.is_finite())
            && self.anchor_point.iter().all(|v| v.is_finite())
            && self.scale.is_finite()
            && self.rotation.is_finite()
            && self.opacity.is_finite();
        if !finite {
            return Err(ModelError::InvalidTransform("non-finite component".into()));
        }
        if self.scale <= 0.0 {
            return Err(ModelError::InvalidTransform(
                "scale must be positive".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.opacity) {
            return Err(ModelError::InvalidTransform(
                "opacity must be in 0..=1".into(),
            ));
        }
        Ok(())
    }
}

impl Default for ClipTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Which animatable clip property a parameter command addresses. Grows as
/// later milestones make more properties animatable (effect params, volume).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClipParam {
    Position,
    AnchorPoint,
    Scale,
    Rotation,
    Opacity,
    /// The clip's crop window (kept-region rect in content fractions).
    /// Routed to [`Clip::crop`] instead of the transform. Always carries a
    /// [`ParamValue::Rect`] `[x, y, w, h]`.
    Crop,
    /// The clip's playback-rate ramp (M2 speed curves). Animates the
    /// instantaneous speed *multiplier* over the clip's normalized span
    /// (`speed_curve`), not the clip transform — its keyframe ticks live in
    /// `0..=`[`SPEED_CURVE_SCALE`], and editing it re-derives the clip's
    /// timeline duration. Always carries a [`ParamValue::Scalar`].
    Speed,
    /// The clip's audio gain envelope (M8 volume envelopes). Routed to the
    /// clip's `volume: Param<f32>` instead of the transform, so the same
    /// keyframe commands draw volume automation and ducking writes ordinary
    /// volume keyframes. Media-backed clips only. Always carries a
    /// [`ParamValue::Scalar`] in `0..=`[`MAX_CLIP_VOLUME`].
    Volume,
    /// A parameter of one of the clip's effects (M4): `effect` is the index
    /// into [`Clip::effects`], `param` the catalog slot. Routed to the
    /// effect's typed param maps instead of the transform. Carries
    /// [`ParamValue::Scalar`], [`ParamValue::Color`], or [`ParamValue::Vec2`]
    /// matching the catalog slot's [`crate::EffectParamKind`].
    Effect {
        effect: u32,
        param: u32,
    },
    /// An animatable property of a [`Generator::Shape`] clip. Routed to the
    /// generator's own `Param`s instead of the transform, so the same
    /// keyframe commands animate shape geometry and colors. Scalar
    /// properties carry [`ParamValue::Scalar`]; `Fill`/`StrokeColor` carry
    /// [`ParamValue::Color`].
    Shape {
        param: ShapeParam,
    },
    /// An animatable property of a [`Generator::Text`] clip's visual style.
    /// Scalar properties carry [`ParamValue::Scalar`]; color properties carry
    /// [`ParamValue::Color`].
    Text {
        param: TextParam,
    },
    /// An animatable property of the clip's color look. Structural look
    /// selections (filter id, LUT path, mask shape, chroma color) remain
    /// ordinary clip edits.
    Look {
        param: LookParam,
    },
    /// An animatable property of the clip's layer styles. Enabling/removing a
    /// style block remains a structural edit (`SetClipLayerStyles`).
    Style {
        param: StyleParam,
    },
}

/// The animatable properties of a [`Generator::Shape`] (see
/// [`ClipParam::Shape`]). Structural knobs — the shape kind, polygon sides,
/// star points, path points — are not animatable; they change through
/// `SetGenerator`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShapeParam {
    /// Shape box width (reference px). Scalar.
    Width,
    /// Shape box height (reference px). Scalar.
    Height,
    /// Corner rounding (reference px). Scalar; rect/polygon/star only honor
    /// it visually but it may be set on any shape.
    CornerRadius,
    /// Star inner-vertex radius fraction. Scalar; star shapes only.
    InnerRatio,
    /// Fill color. Color.
    Fill,
    /// Stroke color. Color; requires the shape to have a stroke.
    StrokeColor,
    /// Stroke width (reference px). Scalar; requires the shape to have a
    /// stroke.
    StrokeWidth,
}

/// The animatable visual treatment properties of a [`Generator::Text`] clip.
/// Font selection and layout structure remain ordinary generator edits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextParam {
    Size,
    Fill,
    LetterSpacing,
    LineSpacing,
    StrokeWidth,
    StrokeColor,
    ShadowBlur,
    ShadowDistance,
    ShadowColor,
}

/// The animatable properties in a clip's color look / mask.
///
/// [`LookParam::MaskCenter`] / [`LookParam::MaskSize`] carry [`ParamValue::Vec2`];
/// the rest carry [`ParamValue::Scalar`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LookParam {
    FilterIntensity,
    LutIntensity,
    AdjustBrightness,
    AdjustContrast,
    AdjustSaturation,
    AdjustExposure,
    AdjustTemperature,
    AdjustTint,
    AdjustHue,
    AdjustHighlights,
    AdjustShadows,
    AdjustSharpness,
    AdjustVignette,
    /// Mask edge softness (`0` … `1`). Scalar.
    MaskFeather,
    /// Mask center offset as a fraction of layer size. Vec2.
    MaskCenter,
    /// Mask size as a fraction of layer size. Vec2.
    MaskSize,
    /// Mask rotation in degrees (clockwise). Scalar.
    MaskRotation,
    /// Rectangle corner rounding (`0` … `1`). Scalar.
    MaskRoundness,
    ChromaStrength,
    ChromaShadow,
}

/// An animatable property of the clip's [`crate::LayerStyles`].
///
/// `*Color` params carry [`ParamValue::Color`]; [`StyleParam::ShadowOffset`]
/// carries [`ParamValue::Vec2`]; the rest carry [`ParamValue::Scalar`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StyleParam {
    /// Shadow color. Color.
    ShadowColor,
    /// Shadow offset in reference pixels. Vec2.
    ShadowOffset,
    /// Shadow blur radius in reference pixels. Scalar.
    ShadowBlur,
    /// Glow color. Color.
    GlowColor,
    /// Glow radius in reference pixels. Scalar.
    GlowRadius,
    /// Glow intensity (`0` … `4`). Scalar.
    GlowIntensity,
    /// Outline color. Color.
    OutlineColor,
    /// Outline width in reference pixels. Scalar.
    OutlineWidth,
    /// Background plate color. Color.
    BackgroundColor,
    /// Background padding in reference pixels. Scalar.
    BackgroundPadding,
    /// Background corner radius in reference pixels. Scalar.
    BackgroundRadius,
}

/// A value for a [`ClipParam`]: scalar properties take `Scalar`, `position`
/// takes `Vec2`, color properties (shape fill/stroke) take `Color`, and
/// crop (and any future 4-float rect) take `Rect` as `[x, y, w, h]`.
/// Commands carry this so one command shape serves every param kind.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamValue {
    Scalar(f32),
    Vec2([f32; 2]),
    Color([u8; 4]),
    /// Axis-aligned rect as `[x, y, w, h]` (e.g. [`ClipParam::Crop`]).
    Rect([f32; 4]),
}

impl ParamValue {
    pub(super) fn scalar(self) -> Result<f32, ModelError> {
        match self {
            ParamValue::Scalar(v) => Ok(v),
            _ => Err(ModelError::InvalidParam("expected a scalar value".into())),
        }
    }

    pub(super) fn vec2(self) -> Result<[f32; 2], ModelError> {
        match self {
            ParamValue::Vec2(v) => Ok(v),
            _ => Err(ModelError::InvalidParam("expected a vec2 value".into())),
        }
    }

    pub(super) fn color(self) -> Result<[u8; 4], ModelError> {
        match self {
            ParamValue::Color(v) => Ok(v),
            _ => Err(ModelError::InvalidParam("expected a color value".into())),
        }
    }

    pub(crate) fn rect(self) -> Result<[f32; 4], ModelError> {
        match self {
            ParamValue::Rect(v) => Ok(v),
            _ => Err(ModelError::InvalidParam("expected a rect value".into())),
        }
    }
}

/// The animatable spatial placement stored on a clip: each [`ClipTransform`]
/// property as a [`Param`] (M2 keystone). Constant params serialize as bare
/// values, so a never-animated transform is byte-identical to the pre-M2
/// `ClipTransform` JSON and old projects load unchanged.
///
/// Keyframe ticks are clip-relative (offset from the clip's timeline start)
/// at the timeline rate — animation rides along when a clip moves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnimatedTransform {
    /// Anchor offset from canvas center (see [`ClipTransform::position`]).
    #[serde(default = "default_position_param")]
    pub position: Param<[f32; 2]>,
    /// Pivot within the content bounds (see [`ClipTransform::anchor_point`]).
    #[serde(
        default = "default_anchor_point_param",
        skip_serializing_if = "is_default_anchor_param"
    )]
    pub anchor_point: Param<[f32; 2]>,
    /// Uniform scale (see [`ClipTransform::scale`]).
    #[serde(default = "default_scale_param")]
    pub scale: Param<f32>,
    /// Clockwise rotation in degrees (see [`ClipTransform::rotation`]).
    #[serde(default = "default_rotation_param")]
    pub rotation: Param<f32>,
    /// Layer opacity 0..=1 (see [`ClipTransform::opacity`]).
    #[serde(default = "default_opacity_param")]
    pub opacity: Param<f32>,
}

fn default_position_param() -> Param<[f32; 2]> {
    Param::Constant([0.0, 0.0])
}
fn default_anchor_point_param() -> Param<[f32; 2]> {
    Param::Constant([0.5, 0.5])
}
fn is_default_anchor_param(p: &Param<[f32; 2]>) -> bool {
    p.constant() == Some([0.5, 0.5])
}
fn default_scale_param() -> Param<f32> {
    Param::Constant(1.0)
}
fn default_rotation_param() -> Param<f32> {
    Param::Constant(0.0)
}
fn default_opacity_param() -> Param<f32> {
    Param::Constant(1.0)
}

impl AnimatedTransform {
    /// All-constant identity (centered, aspect-fit, opaque).
    pub fn identity() -> Self {
        Self::from(ClipTransform::IDENTITY)
    }

    /// True iff no property is animated and every constant is the identity.
    pub fn is_identity(&self) -> bool {
        !self.is_animated() && self.sample(0).is_identity()
    }

    /// True iff any property has keyframes.
    pub fn is_animated(&self) -> bool {
        self.position.is_animated()
            || self.anchor_point.is_animated()
            || self.scale.is_animated()
            || self.rotation.is_animated()
            || self.opacity.is_animated()
    }

    /// The transform value at a clip-relative `tick` — the per-frame hot
    /// path (pure, allocation-free).
    pub fn sample(&self, tick: i64) -> ClipTransform {
        self.sample_at(tick as f64)
    }

    /// [`sample`](Self::sample) at a fractional clip-relative tick:
    /// sub-frame animation sampling for export at rates above the timeline
    /// rate (see [`Param::sample_at`]).
    pub fn sample_at(&self, tick: f64) -> ClipTransform {
        ClipTransform {
            position: self.position.sample_at(tick),
            anchor_point: self.anchor_point.sample_at(tick),
            scale: self.scale.sample_at(tick),
            rotation: self.rotation.sample_at(tick),
            opacity: self.opacity.sample_at(tick),
        }
    }

    /// Set every property to a constant, dropping any keyframes.
    pub fn set_constant(&mut self, transform: ClipTransform) {
        self.position.set_constant(transform.position);
        self.anchor_point.set_constant(transform.anchor_point);
        self.scale.set_constant(transform.scale);
        self.rotation.set_constant(transform.rotation);
        self.opacity.set_constant(transform.opacity);
    }

    /// Apply a full-transform edit composing with animation CapCut-style:
    /// animated properties get a keyframe at `tick` (linear easing),
    /// constant properties stay constant. A gesture on a never-animated
    /// clip behaves exactly like the pre-M2 `set_constant`.
    pub fn compose_at(&mut self, transform: ClipTransform, tick: i64) {
        if self.position.is_animated() {
            self.position
                .set_keyframe(tick, transform.position, Easing::Linear);
        } else {
            self.position.set_constant(transform.position);
        }
        if self.anchor_point.is_animated() {
            self.anchor_point
                .set_keyframe(tick, transform.anchor_point, Easing::Linear);
        } else {
            self.anchor_point.set_constant(transform.anchor_point);
        }
        if self.scale.is_animated() {
            self.scale
                .set_keyframe(tick, transform.scale, Easing::Linear);
        } else {
            self.scale.set_constant(transform.scale);
        }
        if self.rotation.is_animated() {
            self.rotation
                .set_keyframe(tick, transform.rotation, Easing::Linear);
        } else {
            self.rotation.set_constant(transform.rotation);
        }
        if self.opacity.is_animated() {
            self.opacity
                .set_keyframe(tick, transform.opacity, Easing::Linear);
        } else {
            self.opacity.set_constant(transform.opacity);
        }
    }

    /// Upsert a keyframe on one property. The value kind must match the
    /// property and pass the property's range validation.
    pub fn set_param_keyframe(
        &mut self,
        param: ClipParam,
        tick: i64,
        value: ParamValue,
        easing: Easing,
    ) -> Result<(), ModelError> {
        easing.validate()?;
        match param {
            ClipParam::Position => {
                let v = value.vec2()?;
                validate_position(&v)?;
                self.position.set_keyframe(tick, v, easing);
            }
            ClipParam::AnchorPoint => {
                let v = value.vec2()?;
                validate_anchor_point(&v)?;
                self.anchor_point.set_keyframe(tick, v, easing);
            }
            ClipParam::Scale => {
                let v = value.scalar()?;
                validate_scale(v)?;
                self.scale.set_keyframe(tick, v, easing);
            }
            ClipParam::Rotation => {
                let v = value.scalar()?;
                validate_rotation(v)?;
                self.rotation.set_keyframe(tick, v, easing);
            }
            ClipParam::Opacity => {
                let v = value.scalar()?;
                validate_opacity(v)?;
                self.opacity.set_keyframe(tick, v, easing);
            }
            ClipParam::Effect { .. }
            | ClipParam::Crop
            | ClipParam::Speed
            | ClipParam::Volume
            | ClipParam::Shape { .. }
            | ClipParam::Text { .. }
            | ClipParam::Look { .. }
            | ClipParam::Style { .. } => {
                return Err(not_a_transform_param());
            }
        }
        Ok(())
    }

    /// Remove the keyframe at exactly `tick` on one property. Errors when no
    /// keyframe sits there (so a no-op never lands in undo history).
    pub fn remove_param_keyframe(&mut self, param: ClipParam, tick: i64) -> Result<(), ModelError> {
        let removed = match param {
            ClipParam::Position => self.position.remove_keyframe(tick),
            ClipParam::AnchorPoint => self.anchor_point.remove_keyframe(tick),
            ClipParam::Scale => self.scale.remove_keyframe(tick),
            ClipParam::Rotation => self.rotation.remove_keyframe(tick),
            ClipParam::Opacity => self.opacity.remove_keyframe(tick),
            ClipParam::Effect { .. }
            | ClipParam::Crop
            | ClipParam::Speed
            | ClipParam::Volume
            | ClipParam::Shape { .. }
            | ClipParam::Text { .. }
            | ClipParam::Look { .. }
            | ClipParam::Style { .. } => {
                return Err(not_a_transform_param());
            }
        };
        if removed {
            Ok(())
        } else {
            Err(ModelError::InvalidParam(format!(
                "no {param:?} keyframe at tick {tick}"
            )))
        }
    }

    /// Replace one property with a constant, dropping its keyframes.
    pub fn set_param_constant(
        &mut self,
        param: ClipParam,
        value: ParamValue,
    ) -> Result<(), ModelError> {
        match param {
            ClipParam::Position => {
                let v = value.vec2()?;
                validate_position(&v)?;
                self.position.set_constant(v);
            }
            ClipParam::AnchorPoint => {
                let v = value.vec2()?;
                validate_anchor_point(&v)?;
                self.anchor_point.set_constant(v);
            }
            ClipParam::Scale => {
                let v = value.scalar()?;
                validate_scale(v)?;
                self.scale.set_constant(v);
            }
            ClipParam::Rotation => {
                let v = value.scalar()?;
                validate_rotation(v)?;
                self.rotation.set_constant(v);
            }
            ClipParam::Opacity => {
                let v = value.scalar()?;
                validate_opacity(v)?;
                self.opacity.set_constant(v);
            }
            ClipParam::Effect { .. }
            | ClipParam::Crop
            | ClipParam::Speed
            | ClipParam::Volume
            | ClipParam::Shape { .. }
            | ClipParam::Text { .. }
            | ClipParam::Look { .. }
            | ClipParam::Style { .. } => {
                return Err(not_a_transform_param());
            }
        }
        Ok(())
    }

    /// Shift every transform keyframe by `delta` clip-relative ticks.
    pub(super) fn shift_ticks(&mut self, delta: i64) -> Result<(), ModelError> {
        self.position.shift_ticks(delta)?;
        self.anchor_point.shift_ticks(delta)?;
        self.scale.shift_ticks(delta)?;
        self.rotation.shift_ticks(delta)?;
        self.opacity.shift_ticks(delta)?;
        Ok(())
    }

    /// `Ok` iff every stored value (constants and keyframes) passes the
    /// per-property rules [`ClipTransform::validate`] enforces, and every
    /// keyframed param is structurally sound (sorted, non-empty, valid
    /// easings). Used on load and by model mutators.
    pub fn validate(&self) -> Result<(), ModelError> {
        self.position.validate_shape()?;
        self.anchor_point.validate_shape()?;
        self.scale.validate_shape()?;
        self.rotation.validate_shape()?;
        self.opacity.validate_shape()?;
        self.position.for_each_value(validate_position)?;
        self.anchor_point.for_each_value(validate_anchor_point)?;
        self.scale.for_each_value(|v| validate_scale(*v))?;
        self.rotation.for_each_value(|v| validate_rotation(*v))?;
        self.opacity.for_each_value(|v| validate_opacity(*v))?;
        Ok(())
    }
}

/// Effect params and the speed ramp route through their own clip fields, not
/// the transform; the transform mutators reject them so a misrouted command
/// fails loudly.
fn not_a_transform_param() -> ModelError {
    ModelError::InvalidParam("parameter is not a clip transform property".into())
}

fn validate_position(v: &[f32; 2]) -> Result<(), ModelError> {
    if v.iter().all(|c| c.is_finite()) {
        Ok(())
    } else {
        Err(ModelError::InvalidTransform("non-finite component".into()))
    }
}

fn validate_anchor_point(v: &[f32; 2]) -> Result<(), ModelError> {
    if v.iter().all(|c| c.is_finite()) {
        Ok(())
    } else {
        Err(ModelError::InvalidTransform("non-finite anchor".into()))
    }
}

fn validate_scale(v: f32) -> Result<(), ModelError> {
    if !v.is_finite() {
        return Err(ModelError::InvalidTransform("non-finite component".into()));
    }
    if v <= 0.0 {
        return Err(ModelError::InvalidTransform(
            "scale must be positive".into(),
        ));
    }
    Ok(())
}

fn validate_rotation(v: f32) -> Result<(), ModelError> {
    if v.is_finite() {
        Ok(())
    } else {
        Err(ModelError::InvalidTransform("non-finite component".into()))
    }
}

fn validate_opacity(v: f32) -> Result<(), ModelError> {
    if !v.is_finite() {
        return Err(ModelError::InvalidTransform("non-finite component".into()));
    }
    if !(0.0..=1.0).contains(&v) {
        return Err(ModelError::InvalidTransform(
            "opacity must be in 0..=1".into(),
        ));
    }
    Ok(())
}

impl Default for AnimatedTransform {
    fn default() -> Self {
        Self::identity()
    }
}

impl From<ClipTransform> for AnimatedTransform {
    fn from(t: ClipTransform) -> Self {
        Self {
            position: Param::Constant(t.position),
            anchor_point: Param::Constant(t.anchor_point),
            scale: Param::Constant(t.scale),
            rotation: Param::Constant(t.rotation),
            opacity: Param::Constant(t.opacity),
        }
    }
}
