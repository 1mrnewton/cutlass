use serde::{Deserialize, Serialize};

use crate::error::ModelError;
use crate::ids::{ClipId, LinkId, MediaId};
use crate::param::{Easing, Param};
use crate::time::{RationalTime, TimeRange, resample, time_add, time_sub};

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
    /// A vector shape with a fill color (RGBA, 0-255). Geometry (a centered
    /// rect/ellipse) is fixed until per-layer transforms land.
    Shape {
        shape: Shape,
        /// Fill color. Old projects without this field default to white.
        #[serde(default = "default_shape_rgba")]
        rgba: [u8; 4],
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

impl Generator {
    /// A text generator with the default style. Convenience for the common
    /// case of creating a freshly-dropped title.
    pub fn text(content: impl Into<String>) -> Self {
        Generator::Text {
            content: content.into(),
            style: TextStyle::default(),
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
            stroke: None,
            background: None,
            shadow: None,
        }
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
        ClipTransform {
            position: self.position.sample(tick),
            scale: self.scale.sample(tick),
            rotation: self.rotation.sample(tick),
            opacity: self.opacity.sample(tick),
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
        }
    }

    /// Clip-relative animation tick for an absolute timeline position: the
    /// offset from the clip's start. Positions outside the clip clamp into
    /// `[0, duration)` so callers sampling at a stale playhead still get the
    /// nearest in-range value.
    pub fn animation_tick(&self, timeline_tick: i64) -> i64 {
        let offset = timeline_tick - self.timeline.start.value;
        offset.clamp(0, (self.timeline.duration.value - 1).max(0))
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

    /// Map a timeline position to the corresponding source time, for media clips.
    pub fn source_time_at(&self, timeline_pos: RationalTime) -> Result<Option<RationalTime>, ModelError> {
        if !self.timeline.contains(timeline_pos)? {
            return Ok(None);
        }
        match &self.content {
            ClipSource::Media { source, .. } => {
                let offset_tl = time_sub(&timeline_pos, &self.timeline.start)?;
                let offset_src = resample(offset_tl, source.start.rate);
                Ok(Some(time_add(&source.start, &offset_src)?))
            }
            ClipSource::Generated(_) => Ok(None),
        }
    }
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
            Generator::Shape {
                shape: Shape::Ellipse,
                rgba: [0, 128, 255, 255],
            },
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
