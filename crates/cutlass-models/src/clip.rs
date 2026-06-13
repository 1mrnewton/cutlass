use serde::{Deserialize, Serialize};

use crate::effects::EffectInstance;
use crate::error::ModelError;
use crate::ids::{ClipId, LinkId, MediaId};
use crate::param::{Easing, Keyframe, Param};
use crate::time::{Rational, RationalTime, TimeRange, resample, time_sub};

/// What a clip draws. Either a trimmed range of imported media, or synthetic
/// content rendered by the engine (text, shapes, solids, ...).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ClipSource {
    /// A trimmed portion of a [`MediaSource`](crate::MediaSource).
    ///
    /// `source` is the in/out within the media at the media's native rate.
    Media { media: MediaId, source: TimeRange },
    /// Engine-generated content with no backing file.
    Generated(Generator),
}

/// A synthetic clip with no source media. Parameters are intentionally minimal
/// for now; richer styling (fonts, transforms, gradients) can be added per
/// variant without touching the timeline model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Generator {
    /// A title / text layer.
    ///
    /// `style` carries the full visual treatment (font, size, color, stroke,
    /// background, shadow, …). It is `#[serde(default)]` so projects written
    /// before styling existed load with the default look.
    Text {
        content: String,
        #[serde(default)]
        style: TextStyle,
    },
    /// A solid fill (RGBA, 0-255).
    SolidColor { rgba: [u8; 4] },
    /// A centered vector shape (rectangle or ellipse) with a fill color.
    ///
    /// `width` and `height` are in *reference pixels* relative to a 1080px-tall
    /// canvas; the rasterizer scales them by `canvas_height / 1080` (same
    /// convention as [`TextStyle::size`]). Missing fields deserialize to the
    /// legacy centered-50%-of-canvas look; freshly dropped shapes use
    /// [`SHAPE_DROP_WIDTH`] × [`SHAPE_DROP_HEIGHT`].
    Shape {
        shape: Shape,
        /// Fill color. Old projects without this field default to white.
        #[serde(default = "default_shape_rgba")]
        rgba: [u8; 4],
        #[serde(default = "default_shape_width")]
        width: f32,
        #[serde(default = "default_shape_height")]
        height: f32,
    },
    /// Image or animated sticker (asset wiring TBD).
    Sticker,
    /// Motion / composited VFX layer (implementation TBD).
    Effect,
    /// Blur, mask, and similar pixel filters (implementation TBD).
    Filter,
    /// Color grade / pass-through layer affecting tracks beneath it.
    Adjustment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Shape {
    Rectangle,
    Ellipse,
}

/// Default fill color for a shape without one (opaque white).
fn default_shape_rgba() -> [u8; 4] {
    [255, 255, 255, 255]
}

/// Default width for shapes missing the field — reproduces the legacy
/// centered 50%-of-canvas geometry on a 1920×1080 project.
fn default_shape_width() -> f32 {
    960.0
}

/// Default height for shapes missing the field — same legacy geometry.
fn default_shape_height() -> f32 {
    540.0
}

/// Size of a freshly dropped shape (reference pixels @ 1080 canvas height).
pub const SHAPE_DROP_WIDTH: f32 = 200.0;
pub const SHAPE_DROP_HEIGHT: f32 = 200.0;

impl Generator {
    /// A text generator with the default style. Convenience for the common
    /// case of creating a freshly-dropped title.
    pub fn text(content: impl Into<String>) -> Self {
        Generator::Text {
            content: content.into(),
            style: TextStyle::default(),
        }
    }

    /// A shape generator with the default drop size and fill color.
    pub fn shape(shape: Shape, rgba: [u8; 4]) -> Self {
        Generator::Shape {
            shape,
            rgba,
            width: SHAPE_DROP_WIDTH,
            height: SHAPE_DROP_HEIGHT,
        }
    }
}

/// Letter-casing transform applied to a title before shaping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum TextCase {
    /// Render the text as authored.
    #[default]
    Normal,
    /// UPPERCASE.
    Upper,
    /// lowercase.
    Lower,
    /// Title Case (first letter of each word).
    Title,
}

impl TextCase {
    /// Apply the casing transform to `s`.
    pub fn apply(self, s: &str) -> String {
        match self {
            TextCase::Normal => s.to_owned(),
            TextCase::Upper => s.to_uppercase(),
            TextCase::Lower => s.to_lowercase(),
            TextCase::Title => title_case(s),
        }
    }
}

/// Capitalize the first letter of every whitespace-separated word.
fn title_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut at_word_start = true;
    for ch in s.chars() {
        if ch.is_whitespace() {
            at_word_start = true;
            out.push(ch);
        } else if at_word_start {
            at_word_start = false;
            out.extend(ch.to_uppercase());
        } else {
            out.extend(ch.to_lowercase());
        }
    }
    out
}

/// Horizontal alignment of the laid-out title within the canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum TextAlignH {
    Left,
    #[default]
    Center,
    Right,
}

/// Vertical alignment of the title block within the canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum TextAlignV {
    Top,
    #[default]
    Middle,
    Bottom,
}

/// Outline drawn around glyphs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TextStroke {
    /// Stroke color (RGBA, 0-255).
    pub rgba: [u8; 4],
    /// Stroke width in reference pixels (see [`TextStyle::size`]).
    pub width: f32,
}

impl Default for TextStroke {
    fn default() -> Self {
        Self {
            rgba: [0, 0, 0, 255],
            width: 6.0,
        }
    }
}

/// A filled card drawn behind the title block.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TextBackground {
    /// Card color (RGBA, 0-255); the alpha doubles as the opacity slider.
    pub rgba: [u8; 4],
    /// Corner rounding, `0.0` (square) ..= `1.0` (pill).
    pub radius: f32,
}

impl Default for TextBackground {
    fn default() -> Self {
        Self {
            rgba: [0, 0, 0, 255],
            radius: 0.0,
        }
    }
}

/// A soft drop shadow behind the title, offset down-right at 45°.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TextShadow {
    /// Shadow color (RGBA, 0-255); the alpha doubles as the opacity slider.
    pub rgba: [u8; 4],
    /// Blur radius as a fraction of the effective font size, `0.0`..=`1.0`.
    pub blur: f32,
    /// Offset distance in reference pixels (see [`TextStyle::size`]).
    pub distance: f32,
}

impl Default for TextShadow {
    fn default() -> Self {
        Self {
            rgba: [0, 0, 0, 230],
            blur: 0.15,
            distance: 5.0,
        }
    }
}

/// The full visual treatment of a [`Generator::Text`] layer.
///
/// Sizes (`size`, `letter_spacing`, stroke width, shadow distance) are in
/// *reference pixels* relative to a 1080px-tall canvas; the rasterizer scales
/// them by `canvas_height / 1080` so a project looks the same regardless of
/// output resolution. Every field is `#[serde(default)]` so older projects
/// (which only stored `content`) deserialize to the legacy default look.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextStyle {
    /// Font family name (`""` ⇒ the system default font).
    #[serde(default)]
    pub font: String,
    /// Font size in reference pixels (1080px-tall canvas).
    #[serde(default = "default_font_size")]
    pub size: f32,
    #[serde(default)]
    pub bold: bool,
    #[serde(default)]
    pub italic: bool,
    #[serde(default)]
    pub underline: bool,
    #[serde(default)]
    pub case: TextCase,
    /// Fill color (RGBA, 0-255).
    #[serde(default = "default_text_fill")]
    pub fill: [u8; 4],
    /// Extra space between glyphs, in reference pixels (can be negative).
    #[serde(default)]
    pub letter_spacing: f32,
    /// Line-height multiplier (`1.2` ⇒ 120% of the font size).
    #[serde(default = "default_line_spacing")]
    pub line_spacing: f32,
    #[serde(default)]
    pub align_h: TextAlignH,
    #[serde(default)]
    pub align_v: TextAlignV,
    /// Whether the title wraps onto multiple lines when it overflows the
    /// canvas width. `true` (default) keeps the legacy auto-wrap; `false` lays
    /// the text out on a single line — explicit newlines still break — so a
    /// long title overflows the frame edges instead of reflowing (CapCut).
    #[serde(default = "default_wrap")]
    pub wrap: bool,
    /// Optional glyph outline.
    #[serde(default)]
    pub stroke: Option<TextStroke>,
    /// Optional background card.
    #[serde(default)]
    pub background: Option<TextBackground>,
    /// Optional drop shadow.
    #[serde(default)]
    pub shadow: Option<TextShadow>,
}

/// Default font size in reference pixels — matches the legacy `height / 12`
/// look at a 1080px canvas.
fn default_font_size() -> f32 {
    90.0
}

/// Default fill color for a title (opaque white), matching the legacy raster.
fn default_text_fill() -> [u8; 4] {
    [255, 255, 255, 255]
}

/// Default line-height multiplier (matches the legacy `font_size * 1.2`).
fn default_line_spacing() -> f32 {
    1.2
}

/// Default wrap behavior (on) — matches the legacy always-wrap raster so older
/// projects, which had no toggle, deserialize to their original look.
fn default_wrap() -> bool {
    true
}

impl Default for TextStyle {
    fn default() -> Self {
        Self {
            font: String::new(),
            size: default_font_size(),
            bold: false,
            italic: false,
            underline: false,
            case: TextCase::Normal,
            fill: default_text_fill(),
            letter_spacing: 0.0,
            line_spacing: default_line_spacing(),
            align_h: TextAlignH::Center,
            align_v: TextAlignV::Middle,
            wrap: default_wrap(),
            stroke: None,
            background: None,
            shadow: None,
        }
    }
}

/// Normalized crop window into a clip's content (CapCut crop, M1).
///
/// Fractions of the uncropped frame: `(x, y)` is the kept region's top-left
/// corner, `(w, h)` its extent — `{0, 0, 1, 1}` keeps everything. Crop
/// happens in content space *before* placement: the kept region aspect-fits
/// the canvas and transforms exactly like the full frame did, so cropping
/// never moves the layer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CropRect {
    /// Left edge of the kept region, `0.0..1.0` of content width.
    pub x: f32,
    /// Top edge of the kept region, `0.0..1.0` of content height.
    pub y: f32,
    /// Kept width fraction, `(0.0..=1.0]`.
    pub w: f32,
    /// Kept height fraction, `(0.0..=1.0]`.
    pub h: f32,
}

/// Smallest croppable extent per axis (1% of the content) — keeps the kept
/// region non-degenerate so placement math and UV rects never collapse.
pub const MIN_CROP_FRACTION: f32 = 0.01;

impl CropRect {
    /// Keep the whole frame (the default; absent from saves).
    pub const FULL: Self = Self {
        x: 0.0,
        y: 0.0,
        w: 1.0,
        h: 1.0,
    };

    /// True iff the crop keeps the whole frame.
    pub fn is_full(&self) -> bool {
        *self == Self::FULL
    }

    /// `Ok` iff the kept region is non-degenerate and inside the frame:
    /// finite fields, `w`/`h` at least [`MIN_CROP_FRACTION`], edges within
    /// `0..=1`.
    pub fn validate(&self) -> Result<(), ModelError> {
        let finite = [self.x, self.y, self.w, self.h].iter().all(|v| v.is_finite());
        if !finite {
            return Err(ModelError::InvalidParam("crop: non-finite component".into()));
        }
        if self.w < MIN_CROP_FRACTION || self.h < MIN_CROP_FRACTION {
            return Err(ModelError::InvalidParam(format!(
                "crop: kept region must be at least {MIN_CROP_FRACTION} per axis"
            )));
        }
        if self.x < 0.0 || self.y < 0.0 || self.x + self.w > 1.0 || self.y + self.h > 1.0 {
            return Err(ModelError::InvalidParam(
                "crop: kept region must lie inside the frame".into(),
            ));
        }
        Ok(())
    }
}

impl Default for CropRect {
    fn default() -> Self {
        Self::FULL
    }
}

/// Spatial placement of a clip's content on the canvas (CapCut "Basic"
/// transform: position, scale, rotation, opacity).
///
/// Coordinates are normalized to the canvas so projects survive canvas-size
/// changes: `position` is the offset of the content center from the canvas
/// center as a fraction of canvas width/height (+x right, +y down — screen
/// convention). `scale` is uniform with 1.0 = aspect-fit inside the canvas
/// (CapCut's 100%). `rotation` is degrees clockwise about the content center.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ClipTransform {
    /// Content-center offset from canvas center, normalized to canvas
    /// dimensions. `[0.0, 0.0]` = centered; `[0.5, 0.0]` = center sits on
    /// the right canvas edge.
    pub position: [f32; 2],
    /// Uniform scale; 1.0 aspect-fits the content inside the canvas.
    pub scale: f32,
    /// Clockwise rotation in degrees about the content center.
    pub rotation: f32,
    /// Layer opacity, 0.0 (transparent) ..= 1.0 (opaque).
    pub opacity: f32,
}

impl ClipTransform {
    pub const IDENTITY: Self = Self {
        position: [0.0, 0.0],
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
            && self.scale.is_finite()
            && self.rotation.is_finite()
            && self.opacity.is_finite();
        if !finite {
            return Err(ModelError::InvalidTransform("non-finite component".into()));
        }
        if self.scale <= 0.0 {
            return Err(ModelError::InvalidTransform("scale must be positive".into()));
        }
        if !(0.0..=1.0).contains(&self.opacity) {
            return Err(ModelError::InvalidTransform("opacity must be in 0..=1".into()));
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
    Scale,
    Rotation,
    Opacity,
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
    /// A scalar parameter of one of the clip's effects (M4): `effect` is the
    /// index into [`Clip::effects`], `param` the catalog slot. Routed to the
    /// effect's `Param<f32>` instead of the transform, so the same keyframe
    /// commands drive effect curves. Always carries a [`ParamValue::Scalar`].
    Effect { effect: u32, param: u32 },
}

/// A value for a [`ClipParam`]: scalar properties take `Scalar`, `position`
/// takes `Vec2`. Commands carry this so one command shape serves every
/// param kind.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamValue {
    Scalar(f32),
    Vec2([f32; 2]),
}

impl ParamValue {
    fn scalar(self) -> Result<f32, ModelError> {
        match self {
            ParamValue::Scalar(v) => Ok(v),
            ParamValue::Vec2(_) => Err(ModelError::InvalidParam(
                "expected a scalar value, got a vec2".into(),
            )),
        }
    }

    fn vec2(self) -> Result<[f32; 2], ModelError> {
        match self {
            ParamValue::Vec2(v) => Ok(v),
            ParamValue::Scalar(_) => Err(ModelError::InvalidParam(
                "expected a vec2 value, got a scalar".into(),
            )),
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
    /// Content-center offset from canvas center (see [`ClipTransform::position`]).
    #[serde(default = "default_position_param")]
    pub position: Param<[f32; 2]>,
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
            scale: self.scale.sample_at(tick),
            rotation: self.rotation.sample_at(tick),
            opacity: self.opacity.sample_at(tick),
        }
    }

    /// Set every property to a constant, dropping any keyframes.
    pub fn set_constant(&mut self, transform: ClipTransform) {
        self.position.set_constant(transform.position);
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
            self.position.set_keyframe(tick, transform.position, Easing::Linear);
        } else {
            self.position.set_constant(transform.position);
        }
        if self.scale.is_animated() {
            self.scale.set_keyframe(tick, transform.scale, Easing::Linear);
        } else {
            self.scale.set_constant(transform.scale);
        }
        if self.rotation.is_animated() {
            self.rotation.set_keyframe(tick, transform.rotation, Easing::Linear);
        } else {
            self.rotation.set_constant(transform.rotation);
        }
        if self.opacity.is_animated() {
            self.opacity.set_keyframe(tick, transform.opacity, Easing::Linear);
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
            ClipParam::Effect { .. } | ClipParam::Speed | ClipParam::Volume => {
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
            ClipParam::Scale => self.scale.remove_keyframe(tick),
            ClipParam::Rotation => self.rotation.remove_keyframe(tick),
            ClipParam::Opacity => self.opacity.remove_keyframe(tick),
            ClipParam::Effect { .. } | ClipParam::Speed | ClipParam::Volume => {
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
    pub fn set_param_constant(&mut self, param: ClipParam, value: ParamValue) -> Result<(), ModelError> {
        match param {
            ClipParam::Position => {
                let v = value.vec2()?;
                validate_position(&v)?;
                self.position.set_constant(v);
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
            ClipParam::Effect { .. } | ClipParam::Speed | ClipParam::Volume => {
                return Err(not_a_transform_param());
            }
        }
        Ok(())
    }

    /// `Ok` iff every stored value (constants and keyframes) passes the
    /// per-property rules [`ClipTransform::validate`] enforces, and every
    /// keyframed param is structurally sound (sorted, non-empty, valid
    /// easings). Used on load and by model mutators.
    pub fn validate(&self) -> Result<(), ModelError> {
        self.position.validate_shape()?;
        self.scale.validate_shape()?;
        self.rotation.validate_shape()?;
        self.opacity.validate_shape()?;
        self.position.for_each_value(validate_position)?;
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

fn validate_scale(v: f32) -> Result<(), ModelError> {
    if !v.is_finite() {
        return Err(ModelError::InvalidTransform("non-finite component".into()));
    }
    if v <= 0.0 {
        return Err(ModelError::InvalidTransform("scale must be positive".into()));
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
        return Err(ModelError::InvalidTransform("opacity must be in 0..=1".into()));
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
            scale: Param::Constant(t.scale),
            rotation: Param::Constant(t.rotation),
            opacity: Param::Constant(t.opacity),
        }
    }
}

/// A placement of some [`ClipSource`] on a track.
///
/// `timeline` is where the clip sits on the sequence, at the timeline rate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Clip {
    pub id: ClipId,
    pub content: ClipSource,
    pub timeline: TimeRange,
    /// Link group (CapCut linkage): clips sharing a `LinkId` are selected,
    /// moved, and trimmed together — e.g. the video+audio pair created by
    /// dropping media with an audio stream. `None` ⇔ unlinked.
    #[serde(default)]
    pub link: Option<LinkId>,
    /// Spatial placement on the canvas, animatable per property. Identity
    /// (aspect-fit, centered) for clips created before transforms existed.
    /// Ignored on audio tracks. Sample at a clip-relative tick via
    /// [`AnimatedTransform::sample`]; never-animated transforms serialize
    /// exactly like the pre-M2 plain [`ClipTransform`].
    #[serde(default)]
    pub transform: AnimatedTransform,
    /// Playback rate (CapCut speed, M1): source time advances `speed`× per
    /// unit of timeline time — `2/1` plays double speed (the clip occupies
    /// half its source duration on the timeline), `1/2` is 50% slow motion.
    /// Always positive; direction is the separate `reversed` flag. Stored
    /// as an exact rational so source-tick math never drifts. Meaningful on
    /// media clips only; `1/1` (and absent from saves) when never retimed,
    /// so old files load unchanged and untouched projects keep their shape.
    #[serde(default = "unit_speed", skip_serializing_if = "is_unit_speed")]
    pub speed: Rational,
    /// Play the source window backwards (timeline forward ⇒ source
    /// backward). Media clips only; absent from saves while false.
    #[serde(default, skip_serializing_if = "is_false")]
    pub reversed: bool,
    /// Playback-rate ramp (CapCut speed curves, M2): the instantaneous speed
    /// *multiplier* over the clip's normalized span. Constant `1.0` (the
    /// default, omitted from saves) ⇔ a flat ramp, so `speed`/`reversed`
    /// alone govern retiming and old/never-rammed clips are byte-identical.
    ///
    /// Keyframe ticks are normalized to `0..=`[`SPEED_CURVE_SCALE`] (`0` =
    /// clip start, `SPEED_CURVE_SCALE` = clip end), so the ramp's *shape*
    /// rides along when the clip is trimmed or its base speed changes. Speed
    /// is a rate: the source position swept to a point in the clip is the
    /// integral of `speed × speed_curve`, and the clip's timeline duration
    /// re-derives from the curve's average (see [`Clip::source_time_at`] and
    /// [`crate::Project::set_clip_speed_curve`]). Meaningful on media clips
    /// only.
    #[serde(default = "default_speed_curve", skip_serializing_if = "is_unit_speed_curve")]
    pub speed_curve: Param<f32>,
    /// Preserve pitch while retiming (CapCut's "pitch" toggle, M8 Phase 3).
    /// `true` (the default) time-stretches the audio so a sped-up clip keeps
    /// its original pitch; `false` is "chipmunk" mode where pitch rides the
    /// speed. Meaningful on retimed media clips only; `true` (and absent from
    /// saves) otherwise, so old files load pitch-locked.
    #[serde(default = "default_preserve_pitch", skip_serializing_if = "is_true")]
    pub preserve_pitch: bool,
    /// Audio gain envelope (CapCut volume, M1 → M8): `0.0` mutes, `1.0` is
    /// unchanged, up to [`MAX_CLIP_VOLUME`]× boost. Read by both audio mixers
    /// for clips on audio lanes; meaningless elsewhere. A constant for the
    /// common case (byte-identical to the pre-M8 bare-`f32` shape, so old
    /// files load unchanged), or a keyframed [`Param`] envelope (M8): the
    /// mixers sample it per sample-frame, and ducking writes ordinary volume
    /// keyframes. Keyframe ticks are clip-relative timeline ticks, like every
    /// other [`Param`]. `1.0` (and absent from saves) when never touched.
    #[serde(default = "default_volume", skip_serializing_if = "is_unit_volume")]
    pub volume: Param<f32>,
    /// Fade-in duration in timeline ticks from the clip's start: a linear
    /// gain ramp 0 → `volume`. First-class field like CapCut, not keyframe
    /// sugar. Absent from saves while 0.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub fade_in: i64,
    /// Fade-out duration in timeline ticks ending at the clip's end: a
    /// linear gain ramp `volume` → 0. Absent from saves while 0.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub fade_out: i64,
    /// Normalized crop window into the content (CapCut crop, M1): only the
    /// kept region renders, aspect-fit and transformed like the full frame
    /// was. Meaningful on visual clips; full-frame (and absent from saves)
    /// when never cropped, so old files load unchanged.
    #[serde(default, skip_serializing_if = "CropRect::is_full")]
    pub crop: CropRect,
    /// Mirror the content left-right (after crop). Absent from saves while
    /// false.
    #[serde(default, skip_serializing_if = "is_false")]
    pub flip_h: bool,
    /// Mirror the content top-bottom (after crop). Absent from saves while
    /// false.
    #[serde(default, skip_serializing_if = "is_false")]
    pub flip_v: bool,
    /// GPU effect chain (CapCut effects, M4): applied in order to the placed
    /// layer before it composites. Each entry is `{effect_id, params}` with
    /// parameters animatable per the catalog. Meaningful on visual clips;
    /// empty (and absent from saves) when never touched, so old files load
    /// unchanged.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub effects: Vec<EffectInstance>,
}

/// Upper bound for [`Clip::volume`] (CapCut's 1000% ceiling).
pub const MAX_CLIP_VOLUME: f32 = 10.0;

/// Normalized tick span of a [`Clip::speed_curve`]: keyframe tick `0` is the
/// clip's start, [`SPEED_CURVE_SCALE`] its end. The ramp is stored over this
/// fixed domain (not absolute clip ticks) so its shape survives trims and
/// base-speed changes that re-derive the clip's timeline duration.
pub const SPEED_CURVE_SCALE: i64 = 1000;

/// Slowest instantaneous speed multiplier a ramp keyframe may hold (matches
/// the agent's `set_clip_speed` floor). A positive floor keeps the curve's
/// average — and thus the derived duration — finite.
pub const MIN_SPEED: f32 = 0.05;

/// Fastest instantaneous speed multiplier a ramp keyframe may hold.
pub const MAX_SPEED: f32 = 100.0;

fn unit_speed() -> Rational {
    Rational::new(1, 1)
}

fn is_unit_speed(speed: &Rational) -> bool {
    speed.num == speed.den
}

fn default_speed_curve() -> Param<f32> {
    Param::Constant(1.0)
}

/// A flat unit ramp — no retiming contribution. `&` form for serde's
/// `skip_serializing_if`.
fn is_unit_speed_curve(curve: &Param<f32>) -> bool {
    matches!(curve, Param::Constant(v) if *v == 1.0)
}

fn default_preserve_pitch() -> bool {
    true
}

// `&bool` is the signature `skip_serializing_if` requires.
#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_true(b: &bool) -> bool {
    *b
}

fn default_volume() -> Param<f32> {
    Param::Constant(1.0)
}

/// A flat unit-gain envelope — no audio edit. `&` form for serde's
/// `skip_serializing_if`.
fn is_unit_volume(volume: &Param<f32>) -> bool {
    matches!(volume, Param::Constant(v) if *v == 1.0)
}

/// Range check for one volume value: finite, within `0..=`[`MAX_CLIP_VOLUME`].
/// Shared by `set_clip_audio`, the envelope keyframe routing, and load-time
/// envelope validation.
pub fn validate_volume(v: f32) -> Result<(), ModelError> {
    if !v.is_finite() || !(0.0..=MAX_CLIP_VOLUME).contains(&v) {
        return Err(ModelError::InvalidParam(format!(
            "volume must be between 0 and {MAX_CLIP_VOLUME}"
        )));
    }
    Ok(())
}

/// Validate a volume envelope (M8) before it is stored: structurally sound
/// (sorted, non-empty when keyframed, valid easings) with every value in
/// gain range.
pub fn validate_volume_envelope(volume: &Param<f32>) -> Result<(), ModelError> {
    volume.validate_shape()?;
    volume.for_each_value(|v| validate_volume(*v))
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_zero(ticks: &i64) -> bool {
    *ticks == 0
}

/// Audio gain at `pos` within a span of `len` (clip-relative, any unit —
/// ticks or sample frames — as long as all arguments and the `volume`
/// envelope share it): the envelope sampled at `pos` shaped by the linear
/// fade ramps. Fades anchor at the span edges, so a fade longer than a
/// trimmed span just ramps part-way. Both audio mixers evaluate this per
/// sample frame; keep it branch-light. The mixers rebase the envelope into
/// the sample-frame domain once per span ([`Param::map_ticks`]) so this
/// stays an O(log k) lookup, not a tick conversion.
pub fn audio_gain_at(pos: i64, len: i64, volume: &Param<f32>, fade_in: i64, fade_out: i64) -> f32 {
    let mut gain = volume.sample(pos);
    if fade_in > 0 && pos < fade_in {
        gain *= pos.max(0) as f32 / fade_in as f32;
    }
    if fade_out > 0 {
        let remain = len - pos;
        if remain < fade_out {
            gain *= remain.max(0) as f32 / fade_out as f32;
        }
    }
    gain
}

// `&bool` is the signature `skip_serializing_if` requires.
#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_false(b: &bool) -> bool {
    !*b
}

impl Clip {
    /// A clip backed by a trimmed range of imported media.
    pub fn from_media(media: MediaId, source: TimeRange, timeline: TimeRange) -> Self {
        Self {
            id: ClipId::next(),
            content: ClipSource::Media { media, source },
            timeline,
            link: None,
            transform: AnimatedTransform::identity(),
            speed: unit_speed(),
            reversed: false,
            speed_curve: default_speed_curve(),
            preserve_pitch: default_preserve_pitch(),
            volume: default_volume(),
            fade_in: 0,
            fade_out: 0,
            crop: CropRect::FULL,
            flip_h: false,
            flip_v: false,
            effects: Vec::new(),
        }
    }

    /// A generated clip (text, shape, solid, ...).
    pub fn generated(generator: Generator, timeline: TimeRange) -> Self {
        Self {
            id: ClipId::next(),
            content: ClipSource::Generated(generator),
            timeline,
            link: None,
            transform: AnimatedTransform::identity(),
            speed: unit_speed(),
            reversed: false,
            speed_curve: default_speed_curve(),
            preserve_pitch: default_preserve_pitch(),
            volume: default_volume(),
            fade_in: 0,
            fade_out: 0,
            crop: CropRect::FULL,
            flip_h: false,
            flip_v: false,
            effects: Vec::new(),
        }
    }

    /// True iff the clip's framing differs from the default (full frame,
    /// no mirroring) — drives the inspector reset state.
    pub fn has_custom_crop(&self) -> bool {
        !self.crop.is_full() || self.flip_h || self.flip_v
    }

    /// True iff the clip's audio mix differs from the default (full volume,
    /// no fades) — drives the inspector reset state and timeline badges.
    pub fn has_custom_audio(&self) -> bool {
        !is_unit_volume(&self.volume) || self.fade_in > 0 || self.fade_out > 0
    }

    /// True iff the clip carries a keyframed volume envelope (M8), versus a
    /// flat constant gain. Drives the inspector envelope UI and the badge.
    pub fn has_volume_envelope(&self) -> bool {
        self.volume.is_animated()
    }

    /// True iff the clip is inaudible: a constant gain of `0` (or below). A
    /// keyframed envelope is never treated as silent — it may be non-zero
    /// elsewhere — so the mixers keep it and sample per sample-frame.
    pub fn is_silent(&self) -> bool {
        matches!(self.volume.constant(), Some(v) if v <= 0.0)
    }

    /// True iff the clip plays at anything but forward 1× — the audio mixers
    /// time-stretch retimed clips (M8 Phase 3) and the UI badges them. A
    /// non-flat speed ramp counts (M2 speed curves).
    pub fn is_retimed(&self) -> bool {
        !is_unit_speed(&self.speed) || self.reversed || self.has_speed_curve()
    }

    /// True iff the clip carries a non-flat playback-rate ramp (M2 speed
    /// curves) — the constant `1.0` default does not.
    pub fn has_speed_curve(&self) -> bool {
        !is_unit_speed_curve(&self.speed_curve)
    }

    /// Frequency multiplier the varispeed renderer (M8 Phase 3) applies to a
    /// retimed clip's audio: `1.0` when pitch is locked (the CapCut default,
    /// time-stretch preserves pitch), else the clip's overall playback-speed
    /// ratio (`base speed × ramp average`) so pitch rides the speed — the
    /// optional "chipmunk" mode. Reverse does not change pitch.
    pub fn audio_pitch_factor(&self) -> f32 {
        if self.preserve_pitch {
            1.0
        } else {
            let base = f64::from(self.speed.num) / f64::from(self.speed.den);
            (base * self.speed_curve_average()) as f32
        }
    }

    /// `∫₀ᵖ speed_curve(q) dq` over the normalized clip span, `p` in `0..=1`
    /// (`0` = clip start, `1` = clip end). The speed curve is a *rate*, so
    /// this cumulative integral — not the sampled value — is what maps a
    /// timeline position to a fraction of the source window. Pure and
    /// allocation-free; `O(keyframes)` (a handful of ramp points).
    pub fn speed_curve_integral(&self, p: f64) -> f64 {
        speed_curve_integral(&self.speed_curve, p)
    }

    /// Average instantaneous multiplier of the speed ramp over the whole clip
    /// (`speed_curve_integral(1.0)`). The clip's timeline duration derives
    /// from `source_duration ÷ (base_speed × this)`.
    pub fn speed_curve_average(&self) -> f64 {
        self.speed_curve_integral(1.0)
    }

    /// Source ticks consumed by `tl_ticks` timeline ticks at this clip's
    /// speed (both in the same rate; exact rational scale, truncating).
    pub fn scale_by_speed(&self, tl_ticks: i64) -> i64 {
        tl_ticks * i64::from(self.speed.num) / i64::from(self.speed.den)
    }

    /// Timeline ticks covered by `src_ticks` source ticks at this clip's
    /// speed (the inverse of [`Self::scale_by_speed`], truncating).
    pub fn unscale_by_speed(&self, src_ticks: i64) -> i64 {
        src_ticks * i64::from(self.speed.den) / i64::from(self.speed.num)
    }

    /// Clip-relative animation tick for an absolute timeline position: the
    /// offset from the clip's start. Positions outside the clip clamp into
    /// `[0, duration)` so callers sampling at a stale playhead still get the
    /// nearest in-range value.
    pub fn animation_tick(&self, timeline_tick: i64) -> i64 {
        let offset = timeline_tick - self.timeline.start.value;
        offset.clamp(0, (self.timeline.duration.value - 1).max(0))
    }

    /// [`animation_tick`](Self::animation_tick) for a fractional timeline
    /// position — sub-frame export sampling. Clamps into the same
    /// `[0, duration - 1]` range, so the last frame's value holds through
    /// any trailing output frames.
    pub fn animation_tick_f(&self, timeline_tick: f64) -> f64 {
        let offset = timeline_tick - self.timeline.start.value as f64;
        offset.clamp(0.0, (self.timeline.duration.value - 1).max(0) as f64)
    }

    /// Timeline start position.
    pub fn start(&self) -> RationalTime {
        self.timeline.start
    }

    /// Exclusive timeline end.
    pub fn end(&self) -> Result<RationalTime, ModelError> {
        self.timeline.end()
    }

    /// The media this clip references, or `None` for generated content.
    pub fn media(&self) -> Option<MediaId> {
        match &self.content {
            ClipSource::Media { media, .. } => Some(*media),
            ClipSource::Generated(_) => None,
        }
    }

    /// The source in/out range, or `None` for generated content.
    pub fn source_range(&self) -> Option<TimeRange> {
        match &self.content {
            ClipSource::Media { source, .. } => Some(*source),
            ClipSource::Generated(_) => None,
        }
    }

    pub fn is_generated(&self) -> bool {
        matches!(self.content, ClipSource::Generated(_))
    }

    /// Map a timeline position to the corresponding source time, for media
    /// clips. Honors the clip's retiming: without a ramp the timeline offset
    /// scales by `speed` (exact rational math); with a [`Self::speed_curve`]
    /// the source offset is the curve's cumulative integral (speed is a
    /// rate). `reversed` walks the source window backward from its end. The
    /// result clamps into the source window so duration rounding can never
    /// read past an edge.
    pub fn source_time_at(&self, timeline_pos: RationalTime) -> Result<Option<RationalTime>, ModelError> {
        if !self.timeline.contains(timeline_pos)? {
            return Ok(None);
        }
        match &self.content {
            ClipSource::Media { source, .. } => {
                let offset_tl = time_sub(&timeline_pos, &self.timeline.start)?;
                let first = source.start.value;
                let last = first + (source.duration.value - 1).max(0);
                let offset_src = if self.has_speed_curve() {
                    // Speed is a rate: the fraction of the source window swept
                    // by clip-relative position `p` is `∫₀ᵖ curve ÷ ∫₀¹ curve`.
                    // base_speed and the derived duration cancel in the ratio
                    // (the duration was derived to consume the window exactly),
                    // so the curve *shape* alone places the source frame.
                    let dur = self.timeline.duration.value.max(1) as f64;
                    let p = offset_tl.value as f64 / dur;
                    let total = self.speed_curve_average();
                    let ratio = if total > 0.0 {
                        self.speed_curve_integral(p) / total
                    } else {
                        p
                    };
                    (source.duration.value as f64 * ratio).round() as i64
                } else {
                    // Flat ramp: the exact rational fast path (zero f64 drift),
                    // identical to M1 constant speed.
                    let scaled = RationalTime::new(self.scale_by_speed(offset_tl.value), offset_tl.rate);
                    resample(scaled, source.start.rate).value
                };
                let tick = if self.reversed {
                    last - offset_src
                } else {
                    first + offset_src
                };
                Ok(Some(RationalTime::new(
                    tick.clamp(first, last),
                    source.start.rate,
                )))
            }
            ClipSource::Generated(_) => Ok(None),
        }
    }
}

/// `∫₀ᵖ curve(q) dq` over the normalized clip span, `p` in `0..=1` (`0` = clip
/// start, `1` = clip end). Free-function core of [`Clip::speed_curve_integral`]
/// so the audio mixers can evaluate a clip's ramp without a whole [`Clip`].
/// Pure and allocation-free; `O(keyframes)`.
pub fn speed_curve_integral(curve: &Param<f32>, p: f64) -> f64 {
    let p = p.clamp(0.0, 1.0);
    match curve {
        Param::Constant(v) => f64::from(*v) * p,
        Param::Keyframed { keyframes } => {
            let scale = SPEED_CURVE_SCALE as f64;
            let pos = |kf: &Keyframe<f32>| kf.tick as f64 / scale;
            let first = &keyframes[0];
            let q0 = pos(first);
            // Leading flat region holds the first value (CapCut clamp).
            let mut acc = f64::from(first.value) * p.min(q0);
            if p <= q0 {
                return acc;
            }
            for pair in keyframes.windows(2) {
                let (k0, k1) = (&pair[0], &pair[1]);
                let (qa, qb) = (pos(k0), pos(k1));
                if p <= qa {
                    return acc;
                }
                let seg = qb - qa;
                if seg > 0.0 {
                    let upper = p.min(qb);
                    let t_hi = ((upper - qa) / seg) as f32;
                    let (va, vb) = (f64::from(k0.value), f64::from(k1.value));
                    // ∫ over [qa, upper] of lerp(va, vb, e(t)) dq, dq = seg·dt
                    //   = seg·[va·t_hi + (vb − va)·∫₀^{t_hi} e].
                    let e_int = f64::from(k0.easing.integral_to(t_hi));
                    acc += seg * (va * f64::from(t_hi) + (vb - va) * e_int);
                }
                if p <= qb {
                    return acc;
                }
            }
            // Trailing flat region holds the last value.
            let last = &keyframes[keyframes.len() - 1];
            acc + f64::from(last.value) * (p - pos(last))
        }
    }
}

/// Fraction of a clip's source window swept by clip-relative output position
/// `p` in `0..=1`: the normalized cumulative integral `∫₀ᵖ curve ÷ ∫₀¹ curve`.
/// Monotonic with `0 ↦ 0` and `1 ↦ 1`, this is exactly the source placement
/// [`Clip::source_time_at`] uses for video, so the varispeed audio renderer
/// (M8 Phase 3) warps the sound in lockstep with the picture. A degenerate
/// (zero-area) curve falls back to a linear map.
pub fn speed_curve_source_fraction(curve: &Param<f32>, p: f64) -> f64 {
    let total = speed_curve_integral(curve, 1.0);
    if total > 0.0 {
        (speed_curve_integral(curve, p) / total).clamp(0.0, 1.0)
    } else {
        p.clamp(0.0, 1.0)
    }
}

/// Validate a speed ramp (M2 speed curves) before it is stored: a structurally
/// sound `Param` (sorted, non-empty, valid easings) whose every keyframe value
/// is finite and within `[`[`MIN_SPEED`]`, `[`MAX_SPEED`]`]`, with normalized
/// ticks inside `0..=`[`SPEED_CURVE_SCALE`].
pub fn validate_speed_curve(curve: &Param<f32>) -> Result<(), ModelError> {
    curve.validate_shape()?;
    for kf in curve.keyframes() {
        if kf.tick < 0 || kf.tick > SPEED_CURVE_SCALE {
            return Err(ModelError::InvalidParam(format!(
                "speed ramp keyframe tick {} is outside 0..={SPEED_CURVE_SCALE}",
                kf.tick
            )));
        }
    }
    curve.for_each_value(|v| {
        if !v.is_finite() || !(MIN_SPEED..=MAX_SPEED).contains(v) {
            return Err(ModelError::InvalidParam(format!(
                "speed ramp value {v} must be within {MIN_SPEED}..={MAX_SPEED}"
            )));
        }
        Ok(())
    })
}

/// Built-in speed-ramp presets (M2 speed curves, "presets as data"). Each is
/// a normalized [`Param`] over `0..=`[`SPEED_CURVE_SCALE`] of multipliers on
/// the clip's base speed. Shared by the inspector buttons and the agent's
/// `set_speed_curve` tool. Returns `None` for an unknown name.
pub fn speed_preset(name: &str) -> Option<Param<f32>> {
    let s = SPEED_CURVE_SCALE;
    let kf = |frac: f64, value: f32, easing: Easing| Keyframe {
        tick: (frac * s as f64).round() as i64,
        value,
        easing,
    };
    let keyframes = match name {
        // Accelerate from slow-mo into fast (CapCut "speed up").
        "ramp_up" => vec![kf(0.0, 0.4, Easing::EaseIn), kf(1.0, 2.5, Easing::Linear)],
        // Decelerate from fast into slow-mo (CapCut "slow down").
        "ramp_down" => vec![kf(0.0, 2.5, Easing::EaseOut), kf(1.0, 0.4, Easing::Linear)],
        // Fast / slow / fast cuts — montage energy.
        "montage" => vec![
            kf(0.0, 2.0, Easing::EaseInOut),
            kf(0.5, 0.5, Easing::EaseInOut),
            kf(1.0, 2.0, Easing::Linear),
        ],
        // Normal, dip to slow-mo on the action, back to normal — "hero moment".
        "hero" => vec![
            kf(0.0, 1.5, Easing::EaseInOut),
            kf(0.5, 0.3, Easing::EaseInOut),
            kf(1.0, 1.5, Easing::Linear),
        ],
        // Punchy fast / hard slow / fast — "bullet time".
        "bullet" => vec![
            kf(0.0, 3.0, Easing::EaseInOut),
            kf(0.4, 0.25, Easing::EaseInOut),
            kf(0.6, 0.25, Easing::EaseInOut),
            kf(1.0, 3.0, Easing::Linear),
        ],
        _ => return None,
    };
    Some(Param::Keyframed { keyframes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::Rational;

    const R24: Rational = Rational::FPS_24;
    const R30: Rational = Rational::FPS_30;

    fn rt(value: i64, rate: Rational) -> RationalTime {
        RationalTime::new(value, rate)
    }

    fn tr(start: i64, duration: i64, rate: Rational) -> TimeRange {
        TimeRange::at_rate(start, duration, rate)
    }

    fn media_clip(
        media: MediaId,
        source: TimeRange,
        timeline: TimeRange,
    ) -> Clip {
        Clip::from_media(media, source, timeline)
    }

    // --- constructors -----------------------------------------------------

    #[test]
    fn from_media_wires_content_and_timeline() {
        let media = MediaId::from_raw(42);
        let source = tr(100, 50, R30);
        let timeline = tr(10, 40, R24);
        let clip = media_clip(media, source, timeline);

        assert_eq!(
            clip.content,
            ClipSource::Media {
                media,
                source,
            }
        );
        assert_eq!(clip.timeline, timeline);
        assert!(!clip.is_generated());
    }

    #[test]
    fn from_media_assigns_distinct_ids() {
        let media = MediaId::from_raw(1);
        let source = tr(0, 10, R24);
        let timeline = tr(0, 10, R24);
        let a = media_clip(media, source, timeline);
        let b = media_clip(media, source, timeline);
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn generated_text_clip() {
        let timeline = tr(0, 48, R24);
        let clip = Clip::generated(Generator::text("Hello"), timeline);
        assert_eq!(
            clip.content,
            ClipSource::Generated(Generator::text("Hello"))
        );
        assert_eq!(clip.timeline, timeline);
        assert!(clip.is_generated());
    }

    #[test]
    fn generated_all_variants() {
        let timeline = tr(0, 10, R24);

        let solid = Clip::generated(
            Generator::SolidColor {
                rgba: [255, 0, 0, 255],
            },
            timeline,
        );
        assert!(matches!(
            solid.content,
            ClipSource::Generated(Generator::SolidColor { .. })
        ));

        let shape = Clip::generated(
            Generator::shape(Shape::Ellipse, [0, 128, 255, 255]),
            timeline,
        );
        assert!(matches!(
            shape.content,
            ClipSource::Generated(Generator::Shape {
                shape: Shape::Ellipse,
                ..
            })
        ));

        let adj = Clip::generated(Generator::Adjustment, timeline);
        assert!(matches!(
            adj.content,
            ClipSource::Generated(Generator::Adjustment)
        ));
    }

    #[test]
    fn generated_assigns_distinct_ids() {
        let timeline = tr(0, 10, R24);
        let a = Clip::generated(Generator::Adjustment, timeline);
        let b = Clip::generated(Generator::Adjustment, timeline);
        assert_ne!(a.id, b.id);
    }

    // --- accessors --------------------------------------------------------

    #[test]
    fn media_clip_accessors() {
        let media = MediaId::from_raw(7);
        let source = tr(50, 25, R24);
        let timeline = tr(100, 25, R24);
        let clip = media_clip(media, source, timeline);

        assert_eq!(clip.media(), Some(media));
        assert_eq!(clip.source_range(), Some(source));
        assert_eq!(clip.start(), rt(100, R24));
        assert_eq!(clip.end().unwrap(), rt(125, R24));
    }

    #[test]
    fn generated_clip_accessors_are_none() {
        let clip = Clip::generated(Generator::text("x"), tr(5, 10, R24));
        assert_eq!(clip.media(), None);
        assert_eq!(clip.source_range(), None);
        assert_eq!(clip.start().value, 5);
        assert_eq!(clip.end().unwrap().value, 15);
    }

    #[test]
    fn clip_clone_and_eq() {
        let media = MediaId::from_raw(1);
        let source = tr(0, 10, R24);
        let timeline = tr(0, 10, R24);
        let a = media_clip(media, source, timeline);
        let b = a.clone();
        assert_eq!(a, b);
        assert_eq!(a.id, b.id);
    }

    // --- source_time_at: same-rate media ----------------------------------

    #[test]
    fn source_time_at_same_rate_maps_one_to_one() {
        // source [100, 110) placed at timeline [10, 20) — 1:1 at 24fps.
        let clip = media_clip(
            MediaId::from_raw(1),
            tr(100, 10, R24),
            tr(10, 10, R24),
        );

        assert_eq!(
            clip.source_time_at(rt(15, R24)).unwrap(),
            Some(rt(105, R24))
        );
        assert_eq!(
            clip.source_time_at(rt(10, R24)).unwrap(),
            Some(rt(100, R24))
        );
        assert_eq!(
            clip.source_time_at(rt(19, R24)).unwrap(),
            Some(rt(109, R24))
        );
    }

    #[test]
    fn source_time_at_half_open_boundaries() {
        let clip = media_clip(
            MediaId::from_raw(1),
            tr(0, 10, R24),
            tr(10, 10, R24),
        );

        // Exclusive end is not contained.
        assert_eq!(clip.source_time_at(rt(20, R24)).unwrap(), None);
        // Before start.
        assert_eq!(clip.source_time_at(rt(9, R24)).unwrap(), None);
        // After end.
        assert_eq!(clip.source_time_at(rt(21, R24)).unwrap(), None);
    }

    #[test]
    fn source_time_at_generated_always_none() {
        let clip = Clip::generated(Generator::text("title"), tr(0, 100, R24));
        assert_eq!(clip.source_time_at(rt(50, R24)).unwrap(), None);
    }

    // --- source_time_at: mixed rates ------------------------------------

    #[test]
    fn source_time_at_resamples_across_rates() {
        // 120 source ticks @ 30fps -> 96 timeline ticks @ 24fps.
        let clip = media_clip(
            MediaId::from_raw(1),
            tr(0, 120, R30),
            tr(0, 96, R24),
        );

        // Timeline midpoint should land near source midpoint after resample.
        let src = clip.source_time_at(rt(48, R24)).unwrap().unwrap();
        assert_eq!(src.rate, R30);
        // 48 @ 24fps = 60 @ 30fps offset from source start 0.
        assert_eq!(src.value, 60);

        // Timeline start maps to source start regardless of rate.
        assert_eq!(
            clip.source_time_at(rt(0, R24)).unwrap(),
            Some(rt(0, R30))
        );
    }

    #[test]
    fn source_time_at_offset_from_nonzero_source_start() {
        // source [200, 300) @ 30fps at timeline [0, 80) @ 24fps.
        let clip = media_clip(
            MediaId::from_raw(1),
            tr(200, 100, R30),
            tr(0, 80, R24),
        );

        let at_start = clip.source_time_at(rt(0, R24)).unwrap().unwrap();
        assert_eq!(at_start, rt(200, R30));

        // 40 timeline ticks @ 24fps -> 50 source ticks @ 30fps from in-point.
        let mid = clip.source_time_at(rt(40, R24)).unwrap().unwrap();
        assert_eq!(mid, rt(250, R30));
    }

    // --- source_time_at: speed & reverse (M1) -----------------------------

    #[test]
    fn source_time_at_scales_by_speed() {
        // 2× speed: source [100, 200) occupies timeline [0, 50).
        let mut clip = media_clip(MediaId::from_raw(1), tr(100, 100, R24), tr(0, 50, R24));
        clip.speed = Rational::new(2, 1);
        assert!(clip.is_retimed());
        assert_eq!(clip.source_time_at(rt(0, R24)).unwrap(), Some(rt(100, R24)));
        assert_eq!(clip.source_time_at(rt(20, R24)).unwrap(), Some(rt(140, R24)));
        assert_eq!(clip.source_time_at(rt(49, R24)).unwrap(), Some(rt(198, R24)));
    }

    #[test]
    fn source_time_at_half_speed_holds_frames() {
        // ½ speed: source [0, 50) stretches over timeline [0, 100); each
        // source frame holds for two timeline ticks.
        let mut clip = media_clip(MediaId::from_raw(1), tr(0, 50, R24), tr(0, 100, R24));
        clip.speed = Rational::new(1, 2);
        assert_eq!(clip.source_time_at(rt(0, R24)).unwrap(), Some(rt(0, R24)));
        assert_eq!(clip.source_time_at(rt(50, R24)).unwrap(), Some(rt(25, R24)));
        assert_eq!(clip.source_time_at(rt(51, R24)).unwrap(), Some(rt(25, R24)));
        assert_eq!(clip.source_time_at(rt(99, R24)).unwrap(), Some(rt(49, R24)));
    }

    #[test]
    fn source_time_at_reversed_walks_backward() {
        let mut clip = media_clip(MediaId::from_raw(1), tr(100, 50, R24), tr(0, 50, R24));
        clip.reversed = true;
        assert!(clip.is_retimed());
        assert_eq!(clip.source_time_at(rt(0, R24)).unwrap(), Some(rt(149, R24)));
        assert_eq!(clip.source_time_at(rt(25, R24)).unwrap(), Some(rt(124, R24)));
        assert_eq!(clip.source_time_at(rt(49, R24)).unwrap(), Some(rt(100, R24)));
    }

    #[test]
    fn source_time_at_reversed_double_speed() {
        // 2× + reverse: timeline [0, 25) covers source [100, 150) backward.
        let mut clip = media_clip(MediaId::from_raw(1), tr(100, 50, R24), tr(0, 25, R24));
        clip.speed = Rational::new(2, 1);
        clip.reversed = true;
        assert_eq!(clip.source_time_at(rt(0, R24)).unwrap(), Some(rt(149, R24)));
        assert_eq!(clip.source_time_at(rt(10, R24)).unwrap(), Some(rt(129, R24)));
        assert_eq!(clip.source_time_at(rt(24, R24)).unwrap(), Some(rt(101, R24)));
    }

    #[test]
    fn source_time_at_clamps_rounding_into_the_window() {
        // src dur 3 ÷ (2/3 speed) = 4.5 → 4 timeline ticks (truncating);
        // every timeline tick must still land inside the source window.
        let mut clip = media_clip(MediaId::from_raw(1), tr(10, 3, R24), tr(0, 4, R24));
        clip.speed = Rational::new(2, 3);
        for t in 0..4 {
            let src = clip.source_time_at(rt(t, R24)).unwrap().unwrap().value;
            assert!((10..13).contains(&src), "tick {t} mapped to {src}");
        }
    }

    // --- speed serde shape --------------------------------------------------

    #[test]
    fn never_retimed_clips_serialize_without_speed_fields() {
        let clip = media_clip(MediaId::from_raw(1), tr(0, 10, R24), tr(0, 10, R24));
        let value = serde_json::to_value(&clip).expect("serialize");
        let map = value.as_object().expect("clip serializes to a map");
        assert!(!map.contains_key("speed"), "1× speed must stay absent");
        assert!(!map.contains_key("reversed"), "forward must stay absent");

        // And a pre-speed save (no fields) loads as forward 1×.
        let loaded: Clip = serde_json::from_value(value).expect("deserialize");
        assert_eq!(loaded.speed, Rational::new(1, 1));
        assert!(!loaded.reversed);
        assert!(!loaded.is_retimed());
    }

    #[test]
    fn retimed_clip_roundtrips_speed_through_serde() {
        let mut clip = media_clip(MediaId::from_raw(1), tr(0, 10, R24), tr(0, 5, R24));
        clip.speed = Rational::new(2, 1);
        clip.reversed = true;
        let json = serde_json::to_string(&clip).expect("serialize");
        let loaded: Clip = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(loaded.speed, Rational::new(2, 1));
        assert!(loaded.reversed);
    }

    // --- pitch lock (M8 Phase 3) --------------------------------------------

    #[test]
    fn pitch_lock_defaults_on_and_is_omitted_from_saves() {
        let clip = media_clip(MediaId::from_raw(1), tr(0, 10, R24), tr(0, 5, R24));
        assert!(clip.preserve_pitch, "pitch is locked by default");
        let map = serde_json::to_value(&clip).unwrap();
        assert!(
            !map.as_object().unwrap().contains_key("preserve_pitch"),
            "the locked default stays absent so old files are byte-identical"
        );
        // A pre-Phase-3 save (no field) loads pitch-locked.
        let loaded: Clip = serde_json::from_value(map).unwrap();
        assert!(loaded.preserve_pitch);
    }

    #[test]
    fn pitch_unlock_roundtrips_and_drives_the_transpose_factor() {
        let mut clip = media_clip(MediaId::from_raw(1), tr(0, 10, R24), tr(0, 5, R24));
        clip.speed = Rational::new(2, 1);
        // Locked: no pitch shift regardless of speed.
        assert_eq!(clip.audio_pitch_factor(), 1.0);
        // Unlocked (chipmunk): pitch rides the 2× speed.
        clip.preserve_pitch = false;
        assert!((clip.audio_pitch_factor() - 2.0).abs() < 1e-6);
        let json = serde_json::to_string(&clip).expect("serialize");
        assert!(json.contains("preserve_pitch"), "the off state is saved");
        let loaded: Clip = serde_json::from_str(&json).expect("deserialize");
        assert!(!loaded.preserve_pitch);
        assert!((loaded.audio_pitch_factor() - 2.0).abs() < 1e-6);
    }

    // --- speed curves (M2) ---------------------------------------------------

    fn linear_ramp(v0: f32, v1: f32) -> Param<f32> {
        Param::Keyframed {
            keyframes: vec![
                Keyframe { tick: 0, value: v0, easing: Easing::Linear },
                Keyframe { tick: SPEED_CURVE_SCALE, value: v1, easing: Easing::Linear },
            ],
        }
    }

    #[test]
    fn flat_curve_is_not_retimed_and_omitted_from_saves() {
        let clip = media_clip(MediaId::from_raw(1), tr(0, 10, R24), tr(0, 10, R24));
        assert!(!clip.has_speed_curve());
        assert_eq!(clip.speed_curve_average(), 1.0);
        let map = serde_json::to_value(&clip).unwrap();
        assert!(!map.as_object().unwrap().contains_key("speed_curve"));
    }

    #[test]
    fn curve_integral_matches_analytic_linear_ramp() {
        let mut clip = media_clip(MediaId::from_raw(1), tr(0, 100, R24), tr(0, 100, R24));
        // Rate ramps 1 → 3 linearly; average = 2, ∫₀ᵖ (1+2q) dq = p + p².
        clip.speed_curve = linear_ramp(1.0, 3.0);
        assert!(clip.has_speed_curve());
        assert!((clip.speed_curve_average() - 2.0).abs() < 1e-6);
        assert!((clip.speed_curve_integral(0.5) - (0.5 + 0.25)).abs() < 1e-6);
        assert!((clip.speed_curve_integral(1.0) - 2.0).abs() < 1e-6);
        // Outside the unit range clamps.
        assert_eq!(clip.speed_curve_integral(0.0), 0.0);
        assert!((clip.speed_curve_integral(2.0) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn source_fraction_is_normalized_and_matches_source_time_at() {
        // Linear 1 → 3 ramp: ∫₀ᵖ (1+2q) dq / 2 = (p + p²)/2.
        let curve = linear_ramp(1.0, 3.0);
        assert_eq!(speed_curve_source_fraction(&curve, 0.0), 0.0);
        assert!((speed_curve_source_fraction(&curve, 1.0) - 1.0).abs() < 1e-9, "ends at 1");
        assert!(
            (speed_curve_source_fraction(&curve, 0.5) - (0.5 + 0.25) / 2.0).abs() < 1e-9,
            "midpoint sweeps the analytic fraction"
        );
        // It is exactly the placement `source_time_at` uses for the picture, so
        // the audio render warps in lockstep: fraction · window == source offset.
        let clip = {
            let mut c = media_clip(MediaId::from_raw(1), tr(0, 100, R24), tr(0, 100, R24));
            c.speed_curve = curve.clone();
            c
        };
        let src_dur = clip.source_range().unwrap().duration.value as f64;
        let mid = clip.start().value + clip.timeline.duration.value / 2;
        let src = clip
            .source_time_at(RationalTime::new(mid, R24))
            .unwrap()
            .unwrap();
        let expected = (src_dur * speed_curve_source_fraction(&curve, 0.5)).round() as i64;
        assert_eq!(src.value, expected, "audio fraction matches the video mapping");
    }

    #[test]
    fn curve_integral_holds_flat_outside_keyframes() {
        let mut clip = media_clip(MediaId::from_raw(1), tr(0, 100, R24), tr(0, 100, R24));
        // One mid keyframe: constant 2.0 everywhere (flat extrapolation).
        clip.speed_curve = Param::Keyframed {
            keyframes: vec![Keyframe { tick: SPEED_CURVE_SCALE / 2, value: 2.0, easing: Easing::Linear }],
        };
        assert!((clip.speed_curve_integral(0.25) - 0.5).abs() < 1e-6);
        assert!((clip.speed_curve_average() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn source_time_at_curve_sweeps_full_window_symmetrically() {
        // A symmetric slow-fast-slow ramp must still consume the whole source
        // window across the clip, and the midpoint sweeps exactly half.
        let mut clip = media_clip(MediaId::from_raw(1), tr(0, 100, R24), tr(0, 100, R24));
        clip.speed_curve = Param::Keyframed {
            keyframes: vec![
                Keyframe { tick: 0, value: 0.5, easing: Easing::Linear },
                Keyframe { tick: SPEED_CURVE_SCALE / 2, value: 2.0, easing: Easing::Linear },
                Keyframe { tick: SPEED_CURVE_SCALE, value: 0.5, easing: Easing::Linear },
            ],
        };
        let start = clip.source_time_at(rt(0, R24)).unwrap().unwrap();
        let mid = clip.source_time_at(rt(50, R24)).unwrap().unwrap();
        let endish = clip.source_time_at(rt(99, R24)).unwrap().unwrap();
        assert_eq!(start.value, 0);
        // By symmetry the middle of the clip is the middle of the source.
        assert_eq!(mid.value, 50);
        // The last frame clamps to the final source frame (window fully swept).
        assert_eq!(endish.value, 99);
    }

    #[test]
    fn source_time_at_flat_curve_matches_constant_speed_exact_path() {
        // A clip with a flat curve and a constant 2× must map identically to
        // the exact rational path (no f64 drift).
        let mut curved = media_clip(MediaId::from_raw(1), tr(0, 100, R24), tr(0, 50, R24));
        curved.speed = Rational::new(2, 1);
        for tick in [0, 7, 23, 49] {
            let got = curved.source_time_at(rt(tick, R24)).unwrap().unwrap();
            assert_eq!(got.value, (tick * 2).min(99), "tick {tick}");
        }
    }

    #[test]
    fn validate_speed_curve_rejects_out_of_range_values_and_ticks() {
        // Value below the floor.
        assert!(validate_speed_curve(&linear_ramp(0.0, 1.0)).is_err());
        // Tick outside the normalized span.
        let bad_tick = Param::Keyframed {
            keyframes: vec![Keyframe { tick: SPEED_CURVE_SCALE + 1, value: 1.0, easing: Easing::Linear }],
        };
        assert!(validate_speed_curve(&bad_tick).is_err());
        // A sane ramp passes.
        assert!(validate_speed_curve(&linear_ramp(0.5, 2.0)).is_ok());
    }

    #[test]
    fn speed_presets_are_valid_curves() {
        for name in ["ramp_up", "ramp_down", "montage", "hero", "bullet"] {
            let curve = speed_preset(name).unwrap_or_else(|| panic!("missing preset {name}"));
            validate_speed_curve(&curve).unwrap_or_else(|e| panic!("{name} invalid: {e:?}"));
        }
        assert!(speed_preset("nope").is_none());
    }

    #[test]
    fn curve_roundtrips_through_serde_and_marks_retimed() {
        let mut clip = media_clip(MediaId::from_raw(1), tr(0, 100, R24), tr(0, 100, R24));
        clip.speed_curve = speed_preset("montage").unwrap();
        assert!(clip.is_retimed());
        let json = serde_json::to_string(&clip).unwrap();
        let loaded: Clip = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.speed_curve, clip.speed_curve);
        assert!(loaded.has_speed_curve());
    }

    // --- audio mix: volume + fades (M1) --------------------------------------

    #[test]
    fn default_audio_serializes_without_fields() {
        let clip = media_clip(MediaId::from_raw(1), tr(0, 10, R24), tr(0, 10, R24));
        assert!(!clip.has_custom_audio());
        let value = serde_json::to_value(&clip).expect("serialize");
        let map = value.as_object().expect("clip serializes to a map");
        assert!(!map.contains_key("volume"), "unit volume must stay absent");
        assert!(!map.contains_key("fade_in"), "zero fade must stay absent");
        assert!(!map.contains_key("fade_out"), "zero fade must stay absent");

        // And a pre-volume save loads with the defaults.
        let loaded: Clip = serde_json::from_value(value).expect("deserialize");
        assert_eq!(loaded.volume, Param::Constant(1.0));
        assert_eq!((loaded.fade_in, loaded.fade_out), (0, 0));
    }

    #[test]
    fn custom_audio_roundtrips_through_serde() {
        let mut clip = media_clip(MediaId::from_raw(1), tr(0, 48, R24), tr(0, 48, R24));
        clip.volume = Param::Constant(0.5);
        clip.fade_in = 12;
        clip.fade_out = 24;
        assert!(clip.has_custom_audio());
        let json = serde_json::to_string(&clip).expect("serialize");
        let loaded: Clip = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(loaded.volume, Param::Constant(0.5));
        assert_eq!((loaded.fade_in, loaded.fade_out), (12, 24));
    }

    #[test]
    fn constant_volume_serializes_as_a_bare_value() {
        // M8 migrated `volume` to a `Param`, but a constant gain must stay
        // byte-identical to the pre-M8 bare-`f32` shape so old files load
        // unchanged and constant-only saves never grow a `{"kf":..}` wrapper.
        let mut clip = media_clip(MediaId::from_raw(1), tr(0, 48, R24), tr(0, 48, R24));
        clip.volume = Param::Constant(0.5);
        let value = serde_json::to_value(&clip).expect("serialize");
        assert_eq!(value.get("volume"), Some(&serde_json::json!(0.5)));
        // A pre-M8 bare value still loads as a constant.
        let loaded: Clip = serde_json::from_value(value).expect("deserialize");
        assert_eq!(loaded.volume, Param::Constant(0.5));
    }

    #[test]
    fn volume_envelope_roundtrips_and_validates() {
        let mut clip = media_clip(MediaId::from_raw(1), tr(0, 48, R24), tr(0, 48, R24));
        clip.volume = Param::Keyframed {
            keyframes: vec![
                Keyframe { tick: 0, value: 0.0, easing: Easing::Linear },
                Keyframe { tick: 24, value: 1.0, easing: Easing::EaseOut },
            ],
        };
        assert!(clip.has_volume_envelope());
        assert!(!clip.is_silent(), "an envelope is non-zero somewhere");
        validate_volume_envelope(&clip.volume).expect("in-range envelope");
        let json = serde_json::to_string(&clip).expect("serialize");
        let loaded: Clip = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(loaded.volume, clip.volume);

        // Out-of-range gain is rejected.
        let hot = Param::Keyframed {
            keyframes: vec![Keyframe {
                tick: 0,
                value: MAX_CLIP_VOLUME + 1.0,
                easing: Easing::Linear,
            }],
        };
        assert!(validate_volume_envelope(&hot).is_err());
    }

    #[test]
    fn audio_gain_ramps_linearly_at_both_edges() {
        let vol = |v: f32| Param::Constant(v);
        // No fades: flat volume everywhere.
        assert_eq!(audio_gain_at(0, 100, &vol(0.8), 0, 0), 0.8);
        assert_eq!(audio_gain_at(99, 100, &vol(0.8), 0, 0), 0.8);

        // Fade-in over the first 10: silence at 0, half at 5, full at 10.
        assert_eq!(audio_gain_at(0, 100, &vol(1.0), 10, 0), 0.0);
        assert_eq!(audio_gain_at(5, 100, &vol(1.0), 10, 0), 0.5);
        assert_eq!(audio_gain_at(10, 100, &vol(1.0), 10, 0), 1.0);

        // Fade-out over the last 10: full until 90, half at 95, ~0 at the end.
        assert_eq!(audio_gain_at(90, 100, &vol(1.0), 0, 10), 1.0);
        assert_eq!(audio_gain_at(95, 100, &vol(1.0), 0, 10), 0.5);
        assert!(audio_gain_at(99, 100, &vol(1.0), 0, 10) <= 0.11);

        // Ramps scale by the volume and overlapping fades multiply.
        assert_eq!(audio_gain_at(5, 100, &vol(2.0), 10, 0), 1.0);
        assert_eq!(audio_gain_at(5, 10, &vol(1.0), 10, 10), 0.25);

        // Out-of-span positions never go negative.
        assert_eq!(audio_gain_at(-3, 100, &vol(1.0), 10, 0), 0.0);
        assert_eq!(audio_gain_at(105, 100, &vol(1.0), 0, 10), 0.0);
    }

    #[test]
    fn audio_gain_follows_a_keyframed_envelope() {
        // A 0→1 ramp envelope over the span: the gain tracks the curve.
        let env = Param::Keyframed {
            keyframes: vec![
                Keyframe { tick: 0, value: 0.0, easing: Easing::Linear },
                Keyframe { tick: 100, value: 1.0, easing: Easing::Linear },
            ],
        };
        assert_eq!(audio_gain_at(0, 100, &env, 0, 0), 0.0);
        assert_eq!(audio_gain_at(50, 100, &env, 0, 0), 0.5);
        assert_eq!(audio_gain_at(100, 100, &env, 0, 0), 1.0);
        // Fades still multiply on top of the sampled envelope value.
        assert_eq!(audio_gain_at(50, 100, &env, 0, 20), 0.5);
        assert_eq!(audio_gain_at(90, 100, &env, 0, 20), 0.9 * 0.5);
    }

    // --- crop & flip (M1) ----------------------------------------------------

    #[test]
    fn default_crop_serializes_without_fields() {
        let clip = media_clip(MediaId::from_raw(1), tr(0, 10, R24), tr(0, 10, R24));
        assert!(!clip.has_custom_crop());
        let value = serde_json::to_value(&clip).expect("serialize");
        let map = value.as_object().expect("clip serializes to a map");
        assert!(!map.contains_key("crop"), "full crop must stay absent");
        assert!(!map.contains_key("flip_h"), "no flip must stay absent");
        assert!(!map.contains_key("flip_v"), "no flip must stay absent");

        // And a pre-crop save loads with the defaults.
        let loaded: Clip = serde_json::from_value(value).expect("deserialize");
        assert_eq!(loaded.crop, CropRect::FULL);
        assert!(!loaded.flip_h && !loaded.flip_v);
    }

    #[test]
    fn custom_crop_roundtrips_through_serde() {
        let mut clip = media_clip(MediaId::from_raw(1), tr(0, 10, R24), tr(0, 10, R24));
        clip.crop = CropRect {
            x: 0.1,
            y: 0.2,
            w: 0.5,
            h: 0.25,
        };
        clip.flip_h = true;
        assert!(clip.has_custom_crop());
        let json = serde_json::to_string(&clip).expect("serialize");
        let loaded: Clip = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(loaded.crop, clip.crop);
        assert!(loaded.flip_h && !loaded.flip_v);
    }

    #[test]
    fn crop_rect_validation() {
        assert!(CropRect::FULL.validate().is_ok());
        assert!(
            CropRect {
                x: 0.25,
                y: 0.0,
                w: 0.5,
                h: 1.0,
            }
            .validate()
            .is_ok()
        );

        // Degenerate extents.
        for (w, h) in [(0.0, 1.0), (1.0, 0.0), (0.001, 1.0)] {
            assert!(
                CropRect { x: 0.0, y: 0.0, w, h }.validate().is_err(),
                "w={w} h={h} must be rejected"
            );
        }
        // Out of frame.
        assert!(CropRect { x: -0.1, y: 0.0, w: 0.5, h: 0.5 }.validate().is_err());
        assert!(CropRect { x: 0.6, y: 0.0, w: 0.5, h: 0.5 }.validate().is_err());
        assert!(CropRect { x: 0.0, y: 0.9, w: 0.5, h: 0.2 }.validate().is_err());
        // Non-finite.
        assert!(
            CropRect { x: f32::NAN, y: 0.0, w: 1.0, h: 1.0 }
                .validate()
                .is_err()
        );
    }

    // --- transform ----------------------------------------------------------

    #[test]
    fn new_clips_have_identity_transform() {
        let clip = Clip::generated(Generator::Adjustment, tr(0, 10, R24));
        assert!(clip.transform.is_identity());
        assert_eq!(clip.transform, AnimatedTransform::default());
    }

    #[test]
    fn clip_without_transform_field_deserializes_to_identity() {
        // A clip serialized before transforms existed: no `transform` key.
        let clip = Clip::generated(Generator::text("old"), tr(0, 10, R24));
        let mut value = serde_json::to_value(&clip).expect("serialize");
        value
            .as_object_mut()
            .expect("clip serializes to a map")
            .remove("transform")
            .expect("transform field present");

        let loaded: Clip = serde_json::from_value(value).expect("deserialize legacy clip");
        assert!(loaded.transform.is_identity());
        assert_eq!(loaded.content, clip.content);
    }

    #[test]
    fn transform_roundtrips_through_serde() {
        let mut clip = Clip::generated(Generator::Adjustment, tr(0, 10, R24));
        clip.transform = ClipTransform {
            position: [-0.25, 0.5],
            scale: 1.5,
            rotation: 90.0,
            opacity: 0.25,
        }
        .into();
        let json = serde_json::to_string(&clip).expect("serialize");
        let loaded: Clip = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(loaded.transform, clip.transform);
    }

    #[test]
    fn legacy_plain_transform_json_deserializes_as_constants() {
        // The exact shape every pre-M2 save wrote: bare values per property.
        let json = r#"{
            "id": 1,
            "content": { "Generated": { "Text": { "content": "t" } } },
            "timeline": { "start": { "value": 0, "rate": { "num": 24, "den": 1 } },
                          "duration": { "value": 24, "rate": { "num": 24, "den": 1 } } },
            "transform": { "position": [0.25, -0.1], "scale": 2.0,
                           "rotation": 45.0, "opacity": 0.5 }
        }"#;
        let clip: Clip = serde_json::from_str(json).expect("deserialize pre-M2 transform");
        assert!(!clip.transform.is_animated());
        assert_eq!(
            clip.transform.sample(0),
            ClipTransform {
                position: [0.25, -0.1],
                scale: 2.0,
                rotation: 45.0,
                opacity: 0.5,
            }
        );
    }

    #[test]
    fn constant_transform_serializes_in_pre_m2_shape() {
        let mut clip = Clip::generated(Generator::Adjustment, tr(0, 10, R24));
        clip.transform = ClipTransform {
            position: [0.25, 0.5],
            scale: 1.5,
            rotation: 0.0,
            opacity: 1.0,
        }
        .into();
        let value = serde_json::to_value(&clip).expect("serialize");
        // Bare values, not {"kf": ...} wrappers — byte-compatible with old readers.
        assert_eq!(value["transform"]["scale"], 1.5);
        assert_eq!(value["transform"]["position"][0], 0.25);
    }

    #[test]
    fn keyframed_transform_roundtrips() {
        let mut clip = Clip::generated(Generator::Adjustment, tr(0, 48, R24));
        clip.transform
            .set_param_keyframe(ClipParam::Opacity, 0, ParamValue::Scalar(0.0), Easing::Linear)
            .unwrap();
        clip.transform
            .set_param_keyframe(ClipParam::Opacity, 24, ParamValue::Scalar(1.0), Easing::EaseOut)
            .unwrap();
        let json = serde_json::to_string(&clip).expect("serialize");
        let loaded: Clip = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(loaded.transform, clip.transform);
        assert!(loaded.transform.is_animated());
        // Segment 0→24 leaves the linear keyframe at tick 0: halfway = 0.5.
        assert_eq!(loaded.transform.sample(12).opacity, 0.5);
    }

    #[test]
    fn animated_transform_samples_per_property() {
        let mut t = AnimatedTransform::identity();
        t.set_param_keyframe(ClipParam::Scale, 0, ParamValue::Scalar(1.0), Easing::Linear)
            .unwrap();
        t.set_param_keyframe(ClipParam::Scale, 10, ParamValue::Scalar(2.0), Easing::Linear)
            .unwrap();
        // Scale animates; everything else stays constant.
        let mid = t.sample(5);
        assert_eq!(mid.scale, 1.5);
        assert_eq!(mid.position, [0.0, 0.0]);
        assert_eq!(mid.opacity, 1.0);
    }

    #[test]
    fn compose_at_writes_keyframe_only_on_animated_properties() {
        let mut t = AnimatedTransform::identity();
        t.set_param_keyframe(ClipParam::Scale, 0, ParamValue::Scalar(1.0), Easing::Linear)
            .unwrap();
        t.set_param_keyframe(ClipParam::Scale, 20, ParamValue::Scalar(3.0), Easing::Linear)
            .unwrap();

        let edit = ClipTransform {
            position: [0.3, 0.0],
            scale: 2.0,
            rotation: 0.0,
            opacity: 1.0,
        };
        t.compose_at(edit, 10);

        // Scale gained a keyframe at tick 10; the curve still animates.
        assert_eq!(t.scale.keyframes().len(), 3);
        assert_eq!(t.sample(10).scale, 2.0);
        assert_eq!(t.sample(0).scale, 1.0);
        assert_eq!(t.sample(20).scale, 3.0);
        // Position was constant and stays constant.
        assert!(!t.position.is_animated());
        assert_eq!(t.sample(0).position, [0.3, 0.0]);
    }

    #[test]
    fn remove_param_keyframe_errors_when_absent() {
        let mut t = AnimatedTransform::identity();
        assert!(t.remove_param_keyframe(ClipParam::Scale, 5).is_err());
        t.set_param_keyframe(ClipParam::Scale, 5, ParamValue::Scalar(2.0), Easing::Linear)
            .unwrap();
        assert!(t.remove_param_keyframe(ClipParam::Scale, 5).is_ok());
        assert!(!t.scale.is_animated());
        assert_eq!(t.scale.constant(), Some(2.0));
    }

    #[test]
    fn param_kind_mismatch_rejected() {
        let mut t = AnimatedTransform::identity();
        assert!(matches!(
            t.set_param_keyframe(ClipParam::Scale, 0, ParamValue::Vec2([1.0, 1.0]), Easing::Linear),
            Err(ModelError::InvalidParam(_))
        ));
        assert!(matches!(
            t.set_param_constant(ClipParam::Position, ParamValue::Scalar(1.0)),
            Err(ModelError::InvalidParam(_))
        ));
    }

    #[test]
    fn param_values_validated_per_property() {
        let mut t = AnimatedTransform::identity();
        assert!(t
            .set_param_keyframe(ClipParam::Scale, 0, ParamValue::Scalar(-1.0), Easing::Linear)
            .is_err());
        assert!(t
            .set_param_keyframe(ClipParam::Opacity, 0, ParamValue::Scalar(1.5), Easing::Linear)
            .is_err());
        assert!(t
            .set_param_constant(ClipParam::Position, ParamValue::Vec2([f32::NAN, 0.0]))
            .is_err());
    }

    #[test]
    fn animation_tick_clamps_into_clip() {
        let clip = Clip::generated(Generator::Adjustment, tr(100, 50, R24));
        assert_eq!(clip.animation_tick(100), 0);
        assert_eq!(clip.animation_tick(125), 25);
        assert_eq!(clip.animation_tick(149), 49);
        assert_eq!(clip.animation_tick(90), 0);
        assert_eq!(clip.animation_tick(500), 49);
        // The fractional variant keeps sub-frame offsets and clamps the
        // same way.
        assert!((clip.animation_tick_f(125.4) - 25.4).abs() < 1e-9);
        assert_eq!(clip.animation_tick_f(99.5), 0.0);
        assert_eq!(clip.animation_tick_f(149.6), 49.0);
    }

    // --- text style ---------------------------------------------------------

    #[test]
    fn legacy_text_clip_without_style_loads_default() {
        // A title serialized before styling existed: the Text variant only had
        // a `content` field.
        let json = r#"{
            "id": 1,
            "content": { "Generated": { "Text": { "content": "old title" } } },
            "timeline": { "start": { "value": 0, "rate": { "num": 24, "den": 1 } },
                          "duration": { "value": 24, "rate": { "num": 24, "den": 1 } } }
        }"#;
        let clip: Clip = serde_json::from_str(json).expect("deserialize legacy text clip");
        match clip.content {
            ClipSource::Generated(Generator::Text { content, style }) => {
                assert_eq!(content, "old title");
                assert_eq!(style, TextStyle::default());
            }
            other => panic!("expected text generator, got {other:?}"),
        }
    }

    #[test]
    fn legacy_shape_without_dimensions_loads_legacy_defaults() {
        let json = r#"{
            "id": 1,
            "content": { "Generated": { "Shape": {
                "shape": "Rectangle",
                "rgba": [255, 0, 0, 255]
            } } },
            "timeline": { "start": { "value": 0, "rate": { "num": 24, "den": 1 } },
                          "duration": { "value": 24, "rate": { "num": 24, "den": 1 } } }
        }"#;
        let clip: Clip = serde_json::from_str(json).expect("deserialize legacy shape clip");
        match clip.content {
            ClipSource::Generated(Generator::Shape {
                width,
                height,
                ..
            }) => {
                assert_eq!(width, 960.0);
                assert_eq!(height, 540.0);
            }
            other => panic!("expected shape generator, got {other:?}"),
        }
    }

    #[test]
    fn text_style_roundtrips_through_serde() {
        let style = TextStyle {
            font: "Helvetica".into(),
            size: 120.0,
            bold: true,
            italic: true,
            underline: true,
            case: TextCase::Upper,
            fill: [10, 20, 30, 255],
            letter_spacing: 3.0,
            line_spacing: 1.5,
            align_h: TextAlignH::Right,
            align_v: TextAlignV::Bottom,
            wrap: false,
            stroke: Some(TextStroke {
                rgba: [0, 0, 0, 255],
                width: 8.0,
            }),
            background: Some(TextBackground {
                rgba: [255, 255, 0, 200],
                radius: 0.5,
            }),
            shadow: Some(TextShadow {
                rgba: [0, 0, 0, 230],
                blur: 0.25,
                distance: 12.0,
            }),
        };
        let clip = Clip::generated(
            Generator::Text {
                content: "Styled".into(),
                style: style.clone(),
            },
            tr(0, 24, R24),
        );
        let json = serde_json::to_string(&clip).expect("serialize");
        let loaded: Clip = serde_json::from_str(&json).expect("deserialize");
        match loaded.content {
            ClipSource::Generated(Generator::Text { content, style: got }) => {
                assert_eq!(content, "Styled");
                assert_eq!(got, style);
            }
            other => panic!("expected text generator, got {other:?}"),
        }
    }

    #[test]
    fn text_case_apply() {
        assert_eq!(TextCase::Normal.apply("Hello World"), "Hello World");
        assert_eq!(TextCase::Upper.apply("Hello World"), "HELLO WORLD");
        assert_eq!(TextCase::Lower.apply("Hello World"), "hello world");
        assert_eq!(TextCase::Title.apply("hello world"), "Hello World");
        assert_eq!(TextCase::Title.apply("hELLO  wORLD"), "Hello  World");
    }

    #[test]
    fn transform_validation() {
        assert!(ClipTransform::IDENTITY.validate().is_ok());
        assert!(
            ClipTransform {
                position: [0.4, -0.4],
                scale: 3.0,
                rotation: -720.0,
                opacity: 0.0,
            }
            .validate()
            .is_ok()
        );

        let bad_scale = ClipTransform {
            scale: -0.5,
            ..ClipTransform::IDENTITY
        };
        assert!(matches!(
            bad_scale.validate(),
            Err(ModelError::InvalidTransform(_))
        ));

        let bad_opacity = ClipTransform {
            opacity: -0.1,
            ..ClipTransform::IDENTITY
        };
        assert!(matches!(
            bad_opacity.validate(),
            Err(ModelError::InvalidTransform(_))
        ));

        let bad_position = ClipTransform {
            position: [0.0, f32::NAN],
            ..ClipTransform::IDENTITY
        };
        assert!(matches!(
            bad_position.validate(),
            Err(ModelError::InvalidTransform(_))
        ));
    }

    // --- source_time_at: errors -------------------------------------------

    #[test]
    fn source_time_at_rate_mismatch_errors() {
        let clip = media_clip(
            MediaId::from_raw(1),
            tr(0, 10, R24),
            tr(0, 10, R24),
        );
        let err = clip.source_time_at(rt(5, R30)).unwrap_err();
        assert_eq!(
            err,
            ModelError::RateMismatch {
                expected: R30,
                got: R24,
            }
        );
    }
}
