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
    /// A centered vector shape with a fill color and optional outline.
    ///
    /// `width` and `height` are in *reference pixels* relative to a 1080px-tall
    /// canvas; the renderer scales them by `canvas_height / 1080` (same
    /// convention as [`TextStyle::size`]). Missing fields deserialize to the
    /// legacy centered-50%-of-canvas look; freshly dropped shapes use
    /// [`SHAPE_DROP_WIDTH`] × [`SHAPE_DROP_HEIGHT`].
    ///
    /// The geometry and style fields are [`Param`]s (M2 pattern): constants
    /// serialize as bare values, byte-identical to the pre-shape-animation
    /// format, and keyframed curves as `{"kf":[...]}`. Keyframe ticks are
    /// clip-relative, like every other clip param. Never-touched
    /// `corner_radius`/`stroke` are elided from saves entirely.
    Shape {
        shape: Shape,
        /// Fill color. Old projects without this field default to white.
        #[serde(default = "default_shape_rgba")]
        rgba: Param<[u8; 4]>,
        #[serde(default = "default_shape_width")]
        width: Param<f32>,
        #[serde(default = "default_shape_height")]
        height: Param<f32>,
        /// Corner rounding in reference pixels. Shapes with corners
        /// (rectangle, polygon, star) honor it; curved shapes ignore it.
        #[serde(default = "zero_param", skip_serializing_if = "is_zero_param")]
        corner_radius: Param<f32>,
        /// Outline centered on the shape edge (half in, half out).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stroke: Option<ShapeStroke>,
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

/// The geometry of a [`Generator::Shape`]: parametric figures the compositor
/// evaluates as GPU signed-distance fields, or a pen-tool bezier
/// [`ShapePath`] rasterized on the CPU. The legacy `Rectangle`/`Ellipse`
/// unit variants keep their serialized names, so old projects load
/// unchanged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Shape {
    Rectangle,
    Ellipse,
    /// Regular polygon (`sides >= 3`; a triangle is `sides: 3`), fit to the
    /// generator's `width`×`height` box.
    Polygon { sides: u32 },
    /// Star with `points` spikes (`>= 3`); `inner_ratio` is the inner-vertex
    /// radius as a fraction of the outer (`0..=1` — small is spiky, large is
    /// blunt), animatable.
    Star {
        points: u32,
        #[serde(default = "default_star_inner")]
        inner_ratio: Param<f32>,
    },
    /// A horizontal capsule (round-capped line) spanning the box; the box
    /// height is the line thickness. Rotate via the clip transform.
    Line,
    /// A right-pointing arrow filling the box.
    Arrow,
    /// An upright heart fit to the box.
    Heart,
    /// A custom pen-tool outline (cubic beziers).
    Path(ShapePath),
}

/// A pen-tool outline: anchors with absolute cubic handles, in shape-local
/// reference pixels around the shape's center (the pen tool normalizes a
/// committed path so its bounds center the origin — that point lands on the
/// clip's placed center). Point geometry is edited structurally via
/// `SetGenerator` (not keyframed); the fill/stroke/transform animate like
/// any other shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShapePath {
    pub points: Vec<ShapePathPoint>,
    /// Closed paths fill; open paths draw stroke only.
    pub closed: bool,
}

/// One pen-tool anchor. A handle equal to its anchor makes that side of the
/// point a sharp corner (the "click without drag" pen gesture).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ShapePathPoint {
    pub anchor: [f32; 2],
    /// Control of the segment arriving at this anchor.
    pub handle_in: [f32; 2],
    /// Control of the segment leaving this anchor.
    pub handle_out: [f32; 2],
}

impl ShapePathPoint {
    /// A corner point: both handles collapsed onto the anchor.
    pub fn corner(anchor: [f32; 2]) -> Self {
        Self {
            anchor,
            handle_in: anchor,
            handle_out: anchor,
        }
    }
}

/// Outline drawn on a shape's edge, centered on it. Color and width are
/// animatable [`Param`]s (constants serialize bare).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShapeStroke {
    pub rgba: Param<[u8; 4]>,
    pub width: Param<f32>,
}

impl ShapeStroke {
    /// A constant stroke.
    pub fn new(rgba: [u8; 4], width: f32) -> Self {
        Self {
            rgba: Param::Constant(rgba),
            width: Param::Constant(width),
        }
    }
}

/// Default fill color for a shape without one (opaque white).
fn default_shape_rgba() -> Param<[u8; 4]> {
    Param::Constant([255, 255, 255, 255])
}

/// Default width for shapes missing the field — reproduces the legacy
/// centered 50%-of-canvas geometry on a 1920×1080 project.
fn default_shape_width() -> Param<f32> {
    Param::Constant(960.0)
}

/// Default height for shapes missing the field — same legacy geometry.
fn default_shape_height() -> Param<f32> {
    Param::Constant(540.0)
}

fn default_star_inner() -> Param<f32> {
    Param::Constant(0.5)
}

fn zero_param() -> Param<f32> {
    Param::Constant(0.0)
}

/// A never-touched zero constant — elided from saves. `&` form for serde.
fn is_zero_param(p: &Param<f32>) -> bool {
    matches!(p, Param::Constant(v) if *v == 0.0)
}

/// Size of a freshly dropped shape (reference pixels @ 1080 canvas height).
pub const SHAPE_DROP_WIDTH: f32 = 200.0;
pub const SHAPE_DROP_HEIGHT: f32 = 200.0;

/// Largest shape extent / path coordinate magnitude, in reference pixels.
/// Sanity bound only — SDF shapes are resolution-independent, but a runaway
/// value would still blow up path rasters and content-box math.
pub const MAX_SHAPE_DIM: f32 = 8192.0;

/// Most spikes a star (or sides a polygon) may have. Mirrors the evaluator
/// bound in `cutlass-shapes` (`MAX_STAR_POINTS`), which sizes the fixed
/// vertex buffers on the CPU and in WGSL; a render-side test pins the two
/// constants together.
pub const MAX_STAR_POINTS: u32 = 20;

/// Widest stroke, in reference pixels.
pub const MAX_STROKE_WIDTH: f32 = 512.0;

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
            rgba: Param::Constant(rgba),
            width: Param::Constant(SHAPE_DROP_WIDTH),
            height: Param::Constant(SHAPE_DROP_HEIGHT),
            corner_radius: zero_param(),
            stroke: None,
        }
    }

    /// Resolve catalog presets into concrete fields — currently the text
    /// style's `effect_preset`: validate the id and bake the catalog's
    /// stroke / shadow / background onto the style (preset-owned while a
    /// preset is selected; shells clear the id before manual treatment
    /// edits). No-op for other generators and preset-less styles. Called by
    /// [`crate::Project::add_generated`] / [`crate::Project::set_generator`]
    /// so every platform gets identical baked fields from one source of
    /// truth.
    pub fn resolve_presets(&mut self) -> Result<(), ModelError> {
        let Generator::Text { style, .. } = self else {
            return Ok(());
        };
        let Some(preset) = &style.effect_preset else {
            return Ok(());
        };
        let spec = crate::look::text_effect_spec(preset).ok_or_else(|| {
            ModelError::InvalidParam(format!("unknown text effect '{preset}'"))
        })?;
        style.stroke = spec.stroke;
        style.shadow = spec.shadow;
        style.background = spec.background;
        Ok(())
    }

    /// `Ok` iff the generator's content is structurally sound and every
    /// stored value (constants and keyframes) is in range. Enforced by
    /// [`crate::Project::add_generated`] / [`crate::Project::set_generator`]
    /// so a project never holds a shape the renderer would have to clamp.
    pub fn validate(&self) -> Result<(), ModelError> {
        let Generator::Shape {
            shape,
            rgba,
            width,
            height,
            corner_radius,
            stroke,
        } = self
        else {
            return Ok(());
        };

        match shape {
            Shape::Polygon { sides } => validate_star_points(*sides, "polygon sides")?,
            Shape::Star {
                points,
                inner_ratio,
            } => {
                validate_star_points(*points, "star points")?;
                inner_ratio.validate_shape()?;
                inner_ratio.for_each_value(|v| validate_unit_fraction(*v, "star inner_ratio"))?;
            }
            Shape::Path(path) => path.validate()?,
            Shape::Rectangle | Shape::Ellipse | Shape::Line | Shape::Arrow | Shape::Heart => {}
        }

        rgba.validate_shape()?;
        for p in [width, height] {
            p.validate_shape()?;
            p.for_each_value(|v| validate_shape_dim(*v))?;
        }
        corner_radius.validate_shape()?;
        corner_radius.for_each_value(|v| validate_corner_radius(*v))?;
        if let Some(s) = stroke {
            s.rgba.validate_shape()?;
            s.width.validate_shape()?;
            s.width.for_each_value(|v| validate_stroke_width(*v))?;
        }
        Ok(())
    }

    /// Insert or replace a keyframe on one animatable shape property.
    /// Errors when the generator is not a shape, the property does not
    /// apply to its kind (e.g. `InnerRatio` on a rectangle, stroke params
    /// with no stroke set), or the value is out of range.
    pub fn set_shape_param_keyframe(
        &mut self,
        param: ShapeParam,
        tick: i64,
        value: ParamValue,
        easing: Easing,
    ) -> Result<(), ModelError> {
        easing.validate()?;
        self.with_shape_param(param, |target, kind| match kind {
            ShapeParamKind::Scalar { validate } => {
                let v = value.scalar()?;
                validate(v)?;
                target.scalar()?.set_keyframe(tick, v, easing);
                Ok(())
            }
            ShapeParamKind::Color => {
                let v = value.color()?;
                target.color()?.set_keyframe(tick, v, easing);
                Ok(())
            }
        })
    }

    /// Remove the keyframe at exactly `tick` on one shape property. Errors
    /// when no keyframe sits there.
    pub fn remove_shape_param_keyframe(
        &mut self,
        param: ShapeParam,
        tick: i64,
    ) -> Result<(), ModelError> {
        self.with_shape_param(param, |target, kind| {
            let removed = match kind {
                ShapeParamKind::Scalar { .. } => target.scalar()?.remove_keyframe(tick),
                ShapeParamKind::Color => target.color()?.remove_keyframe(tick),
            };
            if removed {
                Ok(())
            } else {
                Err(ModelError::InvalidParam(format!(
                    "no {param:?} keyframe at tick {tick}"
                )))
            }
        })
    }

    /// Replace one shape property with a constant, dropping its keyframes.
    pub fn set_shape_param_constant(
        &mut self,
        param: ShapeParam,
        value: ParamValue,
    ) -> Result<(), ModelError> {
        self.with_shape_param(param, |target, kind| match kind {
            ShapeParamKind::Scalar { validate } => {
                let v = value.scalar()?;
                validate(v)?;
                target.scalar()?.set_constant(v);
                Ok(())
            }
            ShapeParamKind::Color => {
                target.color()?.set_constant(value.color()?);
                Ok(())
            }
        })
    }

    /// Resolve `param` to the [`Param`] it names on this generator and run
    /// `f` on it — the single routing point for the three mutators above.
    fn with_shape_param<R>(
        &mut self,
        param: ShapeParam,
        f: impl FnOnce(ShapeParamTarget<'_>, ShapeParamKind) -> Result<R, ModelError>,
    ) -> Result<R, ModelError> {
        let Generator::Shape {
            shape,
            rgba,
            width,
            height,
            corner_radius,
            stroke,
        } = self
        else {
            return Err(ModelError::InvalidParam(
                "shape parameters apply only to shape generator clips".into(),
            ));
        };
        let scalar = |validate: fn(f32) -> Result<(), ModelError>| ShapeParamKind::Scalar { validate };
        match param {
            ShapeParam::Width => f(ShapeParamTarget::Scalar(width), scalar(validate_shape_dim)),
            ShapeParam::Height => f(ShapeParamTarget::Scalar(height), scalar(validate_shape_dim)),
            ShapeParam::CornerRadius => f(
                ShapeParamTarget::Scalar(corner_radius),
                scalar(validate_corner_radius),
            ),
            ShapeParam::Fill => f(ShapeParamTarget::Color(rgba), ShapeParamKind::Color),
            ShapeParam::InnerRatio => match shape {
                Shape::Star { inner_ratio, .. } => f(
                    ShapeParamTarget::Scalar(inner_ratio),
                    scalar(|v| validate_unit_fraction(v, "star inner_ratio")),
                ),
                _ => Err(ModelError::InvalidParam(
                    "inner_ratio applies only to star shapes".into(),
                )),
            },
            ShapeParam::StrokeWidth | ShapeParam::StrokeColor => match stroke {
                Some(s) => match param {
                    ShapeParam::StrokeWidth => f(
                        ShapeParamTarget::Scalar(&mut s.width),
                        scalar(validate_stroke_width),
                    ),
                    _ => f(ShapeParamTarget::Color(&mut s.rgba), ShapeParamKind::Color),
                },
                None => Err(ModelError::InvalidParam(
                    "shape has no stroke — set one via SetGenerator first".into(),
                )),
            },
        }
    }
}

/// A mutable reference to one animatable shape property, typed by value kind.
enum ShapeParamTarget<'a> {
    Scalar(&'a mut Param<f32>),
    Color(&'a mut Param<[u8; 4]>),
}

impl<'a> ShapeParamTarget<'a> {
    fn scalar(self) -> Result<&'a mut Param<f32>, ModelError> {
        match self {
            ShapeParamTarget::Scalar(p) => Ok(p),
            ShapeParamTarget::Color(_) => Err(ModelError::InvalidParam(
                "expected a color value for this shape parameter".into(),
            )),
        }
    }

    fn color(self) -> Result<&'a mut Param<[u8; 4]>, ModelError> {
        match self {
            ShapeParamTarget::Color(p) => Ok(p),
            ShapeParamTarget::Scalar(_) => Err(ModelError::InvalidParam(
                "expected a scalar value for this shape parameter".into(),
            )),
        }
    }
}

/// Value kind (and range rule) of one [`ShapeParam`].
enum ShapeParamKind {
    Scalar {
        validate: fn(f32) -> Result<(), ModelError>,
    },
    Color,
}

impl ShapePath {
    /// `Ok` iff the path is drawable (>= 2 points) with finite, bounded
    /// coordinates.
    pub fn validate(&self) -> Result<(), ModelError> {
        if self.points.len() < 2 {
            return Err(ModelError::InvalidParam(
                "shape path needs at least 2 points".into(),
            ));
        }
        for p in &self.points {
            for v in [p.anchor, p.handle_in, p.handle_out] {
                if !v.iter().all(|c| c.is_finite() && c.abs() <= MAX_SHAPE_DIM) {
                    return Err(ModelError::InvalidParam(format!(
                        "shape path coordinate out of range (|v| <= {MAX_SHAPE_DIM})"
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Range check for a shape width/height value.
fn validate_shape_dim(v: f32) -> Result<(), ModelError> {
    if !v.is_finite() || v <= 0.0 || v > MAX_SHAPE_DIM {
        return Err(ModelError::InvalidParam(format!(
            "shape size must be positive and at most {MAX_SHAPE_DIM} reference px"
        )));
    }
    Ok(())
}

/// Range check for a corner-radius value.
fn validate_corner_radius(v: f32) -> Result<(), ModelError> {
    if !v.is_finite() || !(0.0..=MAX_SHAPE_DIM).contains(&v) {
        return Err(ModelError::InvalidParam(format!(
            "corner radius must be in 0..={MAX_SHAPE_DIM} reference px"
        )));
    }
    Ok(())
}

/// Range check for a stroke-width value (0 = invisible but legal, so a width
/// animation can ease from nothing).
fn validate_stroke_width(v: f32) -> Result<(), ModelError> {
    if !v.is_finite() || !(0.0..=MAX_STROKE_WIDTH).contains(&v) {
        return Err(ModelError::InvalidParam(format!(
            "stroke width must be in 0..={MAX_STROKE_WIDTH} reference px"
        )));
    }
    Ok(())
}

/// Range check for a `0..=1` fraction value.
fn validate_unit_fraction(v: f32, what: &str) -> Result<(), ModelError> {
    if !v.is_finite() || !(0.0..=1.0).contains(&v) {
        return Err(ModelError::InvalidParam(format!("{what} must be in 0..=1")));
    }
    Ok(())
}

/// Range check for star spike / polygon side counts.
fn validate_star_points(n: u32, what: &str) -> Result<(), ModelError> {
    if !(3..=MAX_STAR_POINTS).contains(&n) {
        return Err(ModelError::InvalidParam(format!(
            "{what} must be in 3..={MAX_STAR_POINTS}"
        )));
    }
    Ok(())
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
    /// Text effect preset id (see [`crate::look::text_effect_catalog`]), the
    /// UI's selected chip. Setting a style with a preset **bakes** the
    /// catalog's stroke / shadow / background onto these fields (see
    /// [`Generator::resolve_presets`]), so files stay self-describing;
    /// `None` leaves the manual treatments alone. Absent from saves while
    /// unset, so old files load unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effect_preset: Option<String>,
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
            effect_preset: None,
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
        let finite = [self.x, self.y, self.w, self.h]
            .iter()
            .all(|v| v.is_finite());
        if !finite {
            return Err(ModelError::InvalidParam(
                "crop: non-finite component".into(),
            ));
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

/// A value for a [`ClipParam`]: scalar properties take `Scalar`, `position`
/// takes `Vec2`, color properties (shape fill/stroke) take `Color`. Commands
/// carry this so one command shape serves every param kind.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamValue {
    Scalar(f32),
    Vec2([f32; 2]),
    Color([u8; 4]),
}

impl ParamValue {
    fn scalar(self) -> Result<f32, ModelError> {
        match self {
            ParamValue::Scalar(v) => Ok(v),
            _ => Err(ModelError::InvalidParam(
                "expected a scalar value".into(),
            )),
        }
    }

    fn vec2(self) -> Result<[f32; 2], ModelError> {
        match self {
            ParamValue::Vec2(v) => Ok(v),
            _ => Err(ModelError::InvalidParam(
                "expected a vec2 value".into(),
            )),
        }
    }

    fn color(self) -> Result<[u8; 4], ModelError> {
        match self {
            ParamValue::Color(v) => Ok(v),
            _ => Err(ModelError::InvalidParam(
                "expected a color value".into(),
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
            ClipParam::Effect { .. } | ClipParam::Speed | ClipParam::Volume | ClipParam::Shape { .. } => {
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
            ClipParam::Effect { .. } | ClipParam::Speed | ClipParam::Volume | ClipParam::Shape { .. } => {
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
            ClipParam::Effect { .. } | ClipParam::Speed | ClipParam::Volume | ClipParam::Shape { .. } => {
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

/// What kind of media a [`Replaceable`] template slot accepts, mirroring
/// CapCut's per-clip "video only" / "image only" restriction (plus an audio
/// variant for marking a swappable music/soundtrack clip).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotMedia {
    /// Any visual media — a video clip or a still image.
    #[default]
    Any,
    /// Video clips only.
    VideoOnly,
    /// Still images only.
    ImageOnly,
    /// Audio only — marks a swappable music/soundtrack clip.
    AudioOnly,
}

impl SlotMedia {
    /// Whether a source of `kind` may fill a slot with this restriction.
    pub fn accepts(self, kind: crate::media::MediaKind) -> bool {
        use crate::media::MediaKind;
        match self {
            SlotMedia::Any => matches!(kind, MediaKind::Video | MediaKind::Image),
            SlotMedia::VideoOnly => kind == MediaKind::Video,
            SlotMedia::ImageOnly => kind == MediaKind::Image,
            SlotMedia::AudioOnly => kind == MediaKind::Audio,
        }
    }
}

/// Marks a [`Clip`] as a user-replaceable template slot (CapCut's "set
/// replaceable material clips"). The clip keeps its sample media so the
/// template previews like the author's video; applying the template swaps the
/// media in slot `order` while the slot's locked timeline duration, transform,
/// effects, and transitions are preserved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Replaceable {
    /// Fill order: slots are filled in ascending `order`, matching the
    /// sequence the user/agent picks media in.
    pub order: u32,
    /// Media-type restriction for this slot.
    #[serde(default)]
    pub accepts: SlotMedia,
    /// Optional author hint shown on the placeholder ("Your clip here"); also
    /// surfaced to the AI agent when auto-filling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

impl Replaceable {
    /// A slot at `order` accepting any visual media.
    pub fn new(order: u32) -> Self {
        Self {
            order,
            accepts: SlotMedia::Any,
            label: None,
        }
    }

    /// Restrict the media type this slot accepts.
    pub fn with_accepts(mut self, accepts: SlotMedia) -> Self {
        self.accepts = accepts;
        self
    }

    /// Attach an author hint for the placeholder.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
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
    #[serde(
        default = "default_speed_curve",
        skip_serializing_if = "is_unit_speed_curve"
    )]
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
    /// Noise reduction (CapCut "Reduce noise", M8 Phase 5): run this clip's
    /// audio through RNNoise to suppress steady background noise (hiss, hum,
    /// room tone) while keeping speech. Both audio mixers render the cleaned
    /// signal and serve it 1:1; meaningful for clips on audio lanes, ignored
    /// elsewhere. `false` (and absent from saves) when off, so old files load
    /// unchanged.
    #[serde(default, skip_serializing_if = "is_false")]
    pub denoise: bool,
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
    /// Detected beat positions (CapCut "Beat" markers, M8 Phase 6): source
    /// ticks at the media frame rate, sorted ascending. `DetectBeats` fills
    /// them from onset analysis; the timeline magnet snaps clip edges to them.
    /// Stored in *source* time so they ride the content — trims and splits
    /// keep exactly the beats inside each half's window visible (see
    /// [`Clip::beat_timeline_ticks`]). Meaningful on media clips; empty (and
    /// absent from saves) until detected, so old files load unchanged.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub beats: Vec<i64>,
    /// Shaped alpha mask (CapCut mask, Phase I): persisted + validated now,
    /// rendered later (documented render gap, like stickers). Meaningful on
    /// media-backed visual clips; `None` (and absent from saves) otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask: Option<crate::look::Mask>,
    /// Chroma keying (CapCut chroma key, Phase I): render-neutral this
    /// milestone. Media-backed visual clips only; absent while `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chroma_key: Option<crate::look::ChromaKey>,
    /// Stabilization strength (CapCut stabilize, Phase I): render-neutral
    /// this milestone. Media-backed *video* clips only (stills have no
    /// motion); absent while `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stabilize: Option<crate::look::StabilizeLevel>,
    /// Color-grade filter preset (CapCut filters, Phase I): render-neutral
    /// this milestone. Visual clips — including `Generator::Filter` lane
    /// bars, whose picked preset lives here; absent while `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<crate::look::Filter>,
    /// Manual color grade (CapCut adjust, Phase I): render-neutral this
    /// milestone. Visual clips — including `Generator::Adjustment` lane
    /// bars; neutral (and absent from saves) when never touched.
    #[serde(default, skip_serializing_if = "crate::look::ColorAdjustments::is_neutral")]
    pub adjust: crate::look::ColorAdjustments,
    /// Entrance animation preset (CapCut animation In tab, Phase I):
    /// render-neutral this milestone. Visual clips; mutually exclusive with
    /// `animation_combo` (enforced by [`crate::Project::set_clip_animation`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub animation_in: Option<crate::look::AnimationRef>,
    /// Exit animation preset (CapCut animation Out tab, Phase I).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub animation_out: Option<crate::look::AnimationRef>,
    /// Looping presence animation (CapCut animation Combo tab, Phase I):
    /// replaces both entrance and exit while set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub animation_combo: Option<crate::look::AnimationRef>,
    /// What this audio-lane clip *is* (music / sound FX / voiceover /
    /// extracted, Phase I): badges and future mixing defaults. Audio-track
    /// clips only; absent while `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_role: Option<crate::look::AudioRole>,
    /// CapCut-style replaceable placeholder marker (templates). `None` for an
    /// ordinary clip; `Some` marks this clip as a user-fillable slot. Absent
    /// from saves while `None`, so non-template projects stay byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replaceable: Option<Replaceable>,
    /// Whether a viewer may re-word this clip's text when using the project as
    /// a template (the text keeps its style and animation). Meaningful on
    /// `Generator::Text` clips; `false` (and absent from saves) otherwise.
    #[serde(default, skip_serializing_if = "is_false")]
    pub text_editable: bool,
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
            denoise: false,
            crop: CropRect::FULL,
            flip_h: false,
            flip_v: false,
            effects: Vec::new(),
            beats: Vec::new(),
            mask: None,
            chroma_key: None,
            stabilize: None,
            filter: None,
            adjust: crate::look::ColorAdjustments::default(),
            animation_in: None,
            animation_out: None,
            animation_combo: None,
            audio_role: None,
            replaceable: None,
            text_editable: false,
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
            denoise: false,
            crop: CropRect::FULL,
            flip_h: false,
            flip_v: false,
            effects: Vec::new(),
            beats: Vec::new(),
            mask: None,
            chroma_key: None,
            stabilize: None,
            filter: None,
            adjust: crate::look::ColorAdjustments::default(),
            animation_in: None,
            animation_out: None,
            animation_combo: None,
            audio_role: None,
            replaceable: None,
            text_editable: false,
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

    /// True iff the clip carries detected beat markers (M8 Phase 6).
    pub fn has_beats(&self) -> bool {
        !self.beats.is_empty()
    }

    /// Absolute timeline ticks (at the clip's timeline rate) for every detected
    /// beat that falls within the clip's visible source window. Beats are stored
    /// in source time, so this maps each through the clip's geometry: the
    /// fraction of the source window a beat sits at becomes the same fraction of
    /// the timeline span (exact for constant speed — the source window maps
    /// linearly onto the span — and a close approximation for speed ramps).
    /// `reversed` mirrors the window. Out-of-window beats (left behind by a
    /// trim/split) are skipped. Pure; `O(beats)`.
    pub fn beat_timeline_ticks(&self) -> Vec<i64> {
        let Some(source) = self.source_range() else {
            return Vec::new();
        };
        let dur = source.duration.value;
        if dur <= 0 || self.beats.is_empty() {
            return Vec::new();
        }
        let first = source.start.value;
        let last = first + dur; // exclusive
        let tl_start = self.timeline.start.value;
        let tl_dur = self.timeline.duration.value;
        let mut out = Vec::with_capacity(self.beats.len());
        for &b in &self.beats {
            if b < first || b >= last {
                continue;
            }
            let mut frac = (b - first) as f64 / dur as f64;
            if self.reversed {
                frac = 1.0 - frac;
            }
            out.push(tl_start + (frac * tl_dur as f64).round() as i64);
        }
        out.sort_unstable();
        out
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
        self.timeline.end().map_err(Into::into)
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
    pub fn source_time_at(
        &self,
        timeline_pos: RationalTime,
    ) -> Result<Option<RationalTime>, ModelError> {
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
                    let scaled =
                        RationalTime::new(self.scale_by_speed(offset_tl.value), offset_tl.rate);
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

/// One speed-ramp preset catalog entry (id + display label; the curve comes
/// from [`speed_preset`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeedPresetSpec {
    pub id: &'static str,
    pub label: &'static str,
}

const SPEED_PRESETS: &[SpeedPresetSpec] = &[
    SpeedPresetSpec {
        id: "ramp_up",
        label: "Ramp up",
    },
    SpeedPresetSpec {
        id: "ramp_down",
        label: "Ramp down",
    },
    SpeedPresetSpec {
        id: "montage",
        label: "Montage",
    },
    SpeedPresetSpec {
        id: "hero",
        label: "Hero",
    },
    SpeedPresetSpec {
        id: "bullet",
        label: "Bullet",
    },
    SpeedPresetSpec {
        id: "jump_cut",
        label: "Jump cut",
    },
    SpeedPresetSpec {
        id: "flash_in",
        label: "Flash in",
    },
    SpeedPresetSpec {
        id: "flash_out",
        label: "Flash out",
    },
];

/// Every speed-ramp preset (UI browsing order). Each id resolves through
/// [`speed_preset`]; the drift is locked by a test.
pub fn speed_preset_catalog() -> &'static [SpeedPresetSpec] {
    SPEED_PRESETS
}

/// The catalog id whose curve equals `curve`, or `None` for a flat ramp or a
/// hand-edited curve. How the shells highlight the active preset tile: curves
/// are normalized over `0..=`[`SPEED_CURVE_SCALE`], so a preset's shape (and
/// this match) survives trims and base-speed changes.
pub fn speed_preset_id(curve: &Param<f32>) -> Option<&'static str> {
    SPEED_PRESETS
        .iter()
        .find(|spec| speed_preset(spec.id).as_ref() == Some(curve))
        .map(|spec| spec.id)
}

/// Built-in speed-ramp presets (M2 speed curves, "presets as data"). Each is
/// a normalized [`Param`] over `0..=`[`SPEED_CURVE_SCALE`] of multipliers on
/// the clip's base speed. Shared by the inspector buttons, the mobile speed
/// panel, and the agent's `set_speed_curve` tool. Returns `None` for an
/// unknown name; the ids are cataloged in [`speed_preset_catalog`].
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
        // Alternating normal / triple-speed bursts — jump-cut energy.
        "jump_cut" => vec![
            kf(0.0, 1.0, Easing::Linear),
            kf(0.25, 3.0, Easing::Linear),
            kf(0.5, 1.0, Easing::Linear),
            kf(0.75, 3.0, Easing::Linear),
            kf(1.0, 1.0, Easing::Linear),
        ],
        // Blast in fast, settle to normal.
        "flash_in" => vec![
            kf(0.0, 3.0, Easing::EaseOut),
            kf(0.3, 1.0, Easing::Linear),
            kf(1.0, 1.0, Easing::Linear),
        ],
        // Hold normal, accelerate out.
        "flash_out" => vec![
            kf(0.0, 1.0, Easing::Linear),
            kf(0.7, 1.0, Easing::EaseIn),
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

    fn media_clip(media: MediaId, source: TimeRange, timeline: TimeRange) -> Clip {
        Clip::from_media(media, source, timeline)
    }

    // --- constructors -----------------------------------------------------

    #[test]
    fn from_media_wires_content_and_timeline() {
        let media = MediaId::from_raw(42);
        let source = tr(100, 50, R30);
        let timeline = tr(10, 40, R24);
        let clip = media_clip(media, source, timeline);

        assert_eq!(clip.content, ClipSource::Media { media, source });
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

    // --- beats (M8 Phase 6) -----------------------------------------------

    #[test]
    fn beat_timeline_ticks_map_proportionally_within_window() {
        // Source window [0, 100) at 24fps maps onto a [10, 60) timeline span
        // (50 ticks): a beat at source 50 (half-way) lands at 10 + 25 = 35.
        let mut clip = media_clip(MediaId::from_raw(1), tr(0, 100, R24), tr(10, 50, R24));
        clip.beats = vec![0, 50, 99];
        assert_eq!(clip.beat_timeline_ticks(), vec![10, 35, 60]);
    }

    #[test]
    fn beat_timeline_ticks_skip_out_of_window_beats() {
        // Window [40, 80): beats before/after it (left by a trim/split) drop.
        let mut clip = media_clip(MediaId::from_raw(1), tr(40, 40, R24), tr(0, 40, R24));
        clip.beats = vec![10, 40, 60, 80, 100];
        assert_eq!(clip.beat_timeline_ticks(), vec![0, 20]);
    }

    #[test]
    fn beat_timeline_ticks_mirror_when_reversed() {
        // Reversed: a beat near the source window end appears near clip start.
        let mut clip = media_clip(MediaId::from_raw(1), tr(0, 100, R24), tr(0, 100, R24));
        clip.reversed = true;
        clip.beats = vec![25, 75];
        assert_eq!(clip.beat_timeline_ticks(), vec![25, 75]);
        clip.beats = vec![10];
        assert_eq!(clip.beat_timeline_ticks(), vec![90]);
    }

    #[test]
    fn beats_serialize_only_when_present() {
        let clip = media_clip(MediaId::from_raw(1), tr(0, 100, R24), tr(0, 100, R24));
        let json = serde_json::to_value(&clip).unwrap();
        assert!(
            json.get("beats").is_none(),
            "beat-free clips serialize without the field"
        );
        // Pre-beats files (no key) deserialize to an empty list.
        let back: Clip = serde_json::from_value(json).unwrap();
        assert!(!back.has_beats());

        let mut beaten = clip.clone();
        beaten.beats = vec![10, 20, 30];
        let json = serde_json::to_value(&beaten).unwrap();
        assert_eq!(json["beats"], serde_json::json!([10, 20, 30]));
        let round: Clip = serde_json::from_value(json).unwrap();
        assert_eq!(round.beats, vec![10, 20, 30]);
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
        let clip = media_clip(MediaId::from_raw(1), tr(100, 10, R24), tr(10, 10, R24));

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
        let clip = media_clip(MediaId::from_raw(1), tr(0, 10, R24), tr(10, 10, R24));

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
        let clip = media_clip(MediaId::from_raw(1), tr(0, 120, R30), tr(0, 96, R24));

        // Timeline midpoint should land near source midpoint after resample.
        let src = clip.source_time_at(rt(48, R24)).unwrap().unwrap();
        assert_eq!(src.rate, R30);
        // 48 @ 24fps = 60 @ 30fps offset from source start 0.
        assert_eq!(src.value, 60);

        // Timeline start maps to source start regardless of rate.
        assert_eq!(clip.source_time_at(rt(0, R24)).unwrap(), Some(rt(0, R30)));
    }

    #[test]
    fn source_time_at_offset_from_nonzero_source_start() {
        // source [200, 300) @ 30fps at timeline [0, 80) @ 24fps.
        let clip = media_clip(MediaId::from_raw(1), tr(200, 100, R30), tr(0, 80, R24));

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
        assert_eq!(
            clip.source_time_at(rt(20, R24)).unwrap(),
            Some(rt(140, R24))
        );
        assert_eq!(
            clip.source_time_at(rt(49, R24)).unwrap(),
            Some(rt(198, R24))
        );
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
        assert_eq!(
            clip.source_time_at(rt(25, R24)).unwrap(),
            Some(rt(124, R24))
        );
        assert_eq!(
            clip.source_time_at(rt(49, R24)).unwrap(),
            Some(rt(100, R24))
        );
    }

    #[test]
    fn source_time_at_reversed_double_speed() {
        // 2× + reverse: timeline [0, 25) covers source [100, 150) backward.
        let mut clip = media_clip(MediaId::from_raw(1), tr(100, 50, R24), tr(0, 25, R24));
        clip.speed = Rational::new(2, 1);
        clip.reversed = true;
        assert_eq!(clip.source_time_at(rt(0, R24)).unwrap(), Some(rt(149, R24)));
        assert_eq!(
            clip.source_time_at(rt(10, R24)).unwrap(),
            Some(rt(129, R24))
        );
        assert_eq!(
            clip.source_time_at(rt(24, R24)).unwrap(),
            Some(rt(101, R24))
        );
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
                Keyframe {
                    tick: 0,
                    value: v0,
                    easing: Easing::Linear,
                },
                Keyframe {
                    tick: SPEED_CURVE_SCALE,
                    value: v1,
                    easing: Easing::Linear,
                },
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
        assert!(
            (speed_curve_source_fraction(&curve, 1.0) - 1.0).abs() < 1e-9,
            "ends at 1"
        );
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
        assert_eq!(
            src.value, expected,
            "audio fraction matches the video mapping"
        );
    }

    #[test]
    fn curve_integral_holds_flat_outside_keyframes() {
        let mut clip = media_clip(MediaId::from_raw(1), tr(0, 100, R24), tr(0, 100, R24));
        // One mid keyframe: constant 2.0 everywhere (flat extrapolation).
        clip.speed_curve = Param::Keyframed {
            keyframes: vec![Keyframe {
                tick: SPEED_CURVE_SCALE / 2,
                value: 2.0,
                easing: Easing::Linear,
            }],
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
                Keyframe {
                    tick: 0,
                    value: 0.5,
                    easing: Easing::Linear,
                },
                Keyframe {
                    tick: SPEED_CURVE_SCALE / 2,
                    value: 2.0,
                    easing: Easing::Linear,
                },
                Keyframe {
                    tick: SPEED_CURVE_SCALE,
                    value: 0.5,
                    easing: Easing::Linear,
                },
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
            keyframes: vec![Keyframe {
                tick: SPEED_CURVE_SCALE + 1,
                value: 1.0,
                easing: Easing::Linear,
            }],
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
                Keyframe {
                    tick: 0,
                    value: 0.0,
                    easing: Easing::Linear,
                },
                Keyframe {
                    tick: 24,
                    value: 1.0,
                    easing: Easing::EaseOut,
                },
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
                Keyframe {
                    tick: 0,
                    value: 0.0,
                    easing: Easing::Linear,
                },
                Keyframe {
                    tick: 100,
                    value: 1.0,
                    easing: Easing::Linear,
                },
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
                CropRect {
                    x: 0.0,
                    y: 0.0,
                    w,
                    h
                }
                .validate()
                .is_err(),
                "w={w} h={h} must be rejected"
            );
        }
        // Out of frame.
        assert!(
            CropRect {
                x: -0.1,
                y: 0.0,
                w: 0.5,
                h: 0.5
            }
            .validate()
            .is_err()
        );
        assert!(
            CropRect {
                x: 0.6,
                y: 0.0,
                w: 0.5,
                h: 0.5
            }
            .validate()
            .is_err()
        );
        assert!(
            CropRect {
                x: 0.0,
                y: 0.9,
                w: 0.5,
                h: 0.2
            }
            .validate()
            .is_err()
        );
        // Non-finite.
        assert!(
            CropRect {
                x: f32::NAN,
                y: 0.0,
                w: 1.0,
                h: 1.0
            }
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
            ..ClipTransform::IDENTITY
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
                ..ClipTransform::IDENTITY
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
            ..ClipTransform::IDENTITY
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
            .set_param_keyframe(
                ClipParam::Opacity,
                0,
                ParamValue::Scalar(0.0),
                Easing::Linear,
            )
            .unwrap();
        clip.transform
            .set_param_keyframe(
                ClipParam::Opacity,
                24,
                ParamValue::Scalar(1.0),
                Easing::EaseOut,
            )
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
        t.set_param_keyframe(
            ClipParam::Scale,
            10,
            ParamValue::Scalar(2.0),
            Easing::Linear,
        )
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
        t.set_param_keyframe(
            ClipParam::Scale,
            20,
            ParamValue::Scalar(3.0),
            Easing::Linear,
        )
        .unwrap();

        let edit = ClipTransform {
            position: [0.3, 0.0],
            scale: 2.0,
            rotation: 0.0,
            opacity: 1.0,
            ..ClipTransform::IDENTITY
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
            t.set_param_keyframe(
                ClipParam::Scale,
                0,
                ParamValue::Vec2([1.0, 1.0]),
                Easing::Linear
            ),
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
        assert!(
            t.set_param_keyframe(
                ClipParam::Scale,
                0,
                ParamValue::Scalar(-1.0),
                Easing::Linear
            )
            .is_err()
        );
        assert!(
            t.set_param_keyframe(
                ClipParam::Opacity,
                0,
                ParamValue::Scalar(1.5),
                Easing::Linear
            )
            .is_err()
        );
        assert!(
            t.set_param_constant(ClipParam::Position, ParamValue::Vec2([f32::NAN, 0.0]))
                .is_err()
        );
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
                rgba,
                width,
                height,
                corner_radius,
                stroke,
                ..
            }) => {
                // Bare legacy values load as constants (the M2 Param trick).
                assert_eq!(rgba.constant(), Some([255, 0, 0, 255]));
                assert_eq!(width.constant(), Some(960.0));
                assert_eq!(height.constant(), Some(540.0));
                // Fields that postdate the file default to "not set".
                assert_eq!(corner_radius.constant(), Some(0.0));
                assert!(stroke.is_none());
            }
            other => panic!("expected shape generator, got {other:?}"),
        }
    }

    #[test]
    fn never_touched_shape_serializes_byte_identical_to_legacy() {
        // A constant shape must not leak the new fields into saves: `Param`
        // constants serialize bare, corner_radius/stroke are elided.
        let generator = Generator::shape(Shape::Rectangle, [10, 20, 30, 255]);
        let json = serde_json::to_value(&generator).unwrap();
        let obj = &json["Shape"];
        assert_eq!(obj["rgba"], serde_json::json!([10, 20, 30, 255]));
        assert_eq!(obj["width"], serde_json::json!(200.0));
        assert!(
            obj.get("corner_radius").is_none(),
            "zero corner radius must be elided: {obj}"
        );
        assert!(obj.get("stroke").is_none(), "absent stroke must be elided");
    }

    #[test]
    fn new_shape_kinds_roundtrip_with_keyframed_params() {
        let mut inner = Param::Constant(0.5);
        inner.set_keyframe(0, 0.2, Easing::Linear);
        inner.set_keyframe(24, 0.9, Easing::EaseInOut);
        let generator = Generator::Shape {
            shape: Shape::Star {
                points: 5,
                inner_ratio: inner,
            },
            rgba: Param::Constant([255, 0, 0, 255]),
            width: Param::Constant(300.0),
            height: Param::Constant(300.0),
            corner_radius: Param::Constant(4.0),
            stroke: Some(ShapeStroke::new([0, 0, 0, 255], 8.0)),
        };
        generator.validate().expect("valid star");
        let json = serde_json::to_string(&generator).unwrap();
        let back: Generator = serde_json::from_str(&json).unwrap();
        assert_eq!(back, generator);

        let path = Generator::Shape {
            shape: Shape::Path(ShapePath {
                points: vec![
                    ShapePathPoint::corner([-50.0, 0.0]),
                    ShapePathPoint {
                        anchor: [50.0, 0.0],
                        handle_in: [0.0, -60.0],
                        handle_out: [50.0, 0.0],
                    },
                ],
                closed: false,
            }),
            rgba: Param::Constant([255, 255, 255, 255]),
            width: Param::Constant(100.0),
            height: Param::Constant(100.0),
            corner_radius: Param::Constant(0.0),
            stroke: Some(ShapeStroke::new([255, 255, 255, 255], 4.0)),
        };
        path.validate().expect("valid path");
        let json = serde_json::to_string(&path).unwrap();
        let back: Generator = serde_json::from_str(&json).unwrap();
        assert_eq!(back, path);
    }

    #[test]
    fn generator_validate_rejects_bad_shapes() {
        let bad = |shape: Shape| Generator::shape(shape, [255, 255, 255, 255]).validate();
        assert!(bad(Shape::Polygon { sides: 2 }).is_err());
        assert!(bad(Shape::Star { points: 99, inner_ratio: Param::Constant(0.5) }).is_err());
        assert!(
            bad(Shape::Star { points: 5, inner_ratio: Param::Constant(1.5) }).is_err(),
            "inner_ratio above 1 must be rejected"
        );
        assert!(
            bad(Shape::Path(ShapePath { points: vec![ShapePathPoint::corner([0.0, 0.0])], closed: true })).is_err(),
            "single-point path is not drawable"
        );
        assert!(
            bad(Shape::Path(ShapePath {
                points: vec![
                    ShapePathPoint::corner([0.0, f32::NAN]),
                    ShapePathPoint::corner([10.0, 0.0]),
                ],
                closed: true,
            }))
            .is_err(),
            "non-finite path coordinate must be rejected"
        );

        let mut wide = Generator::shape(Shape::Rectangle, [255, 255, 255, 255]);
        if let Generator::Shape { width, .. } = &mut wide {
            *width = Param::Constant(MAX_SHAPE_DIM * 2.0);
        }
        assert!(wide.validate().is_err(), "oversized width must be rejected");

        let mut hot_stroke = Generator::shape(Shape::Ellipse, [255, 255, 255, 255]);
        if let Generator::Shape { stroke, .. } = &mut hot_stroke {
            *stroke = Some(ShapeStroke::new([0, 0, 0, 255], MAX_STROKE_WIDTH + 1.0));
        }
        assert!(hot_stroke.validate().is_err());
    }

    #[test]
    fn shape_param_routing_sets_and_samples() {
        let mut g = Generator::shape(Shape::Rectangle, [255, 255, 255, 255]);
        g.set_shape_param_keyframe(ShapeParam::Width, 0, ParamValue::Scalar(100.0), Easing::Linear)
            .unwrap();
        g.set_shape_param_keyframe(ShapeParam::Width, 10, ParamValue::Scalar(300.0), Easing::Linear)
            .unwrap();
        g.set_shape_param_keyframe(ShapeParam::Fill, 0, ParamValue::Color([0, 0, 0, 255]), Easing::Linear)
            .unwrap();
        g.set_shape_param_keyframe(
            ShapeParam::Fill,
            10,
            ParamValue::Color([255, 0, 0, 255]),
            Easing::Linear,
        )
        .unwrap();
        let Generator::Shape { width, rgba, .. } = &g else {
            panic!()
        };
        assert_eq!(width.sample(5), 200.0);
        assert_eq!(rgba.sample(5), [128, 0, 0, 255], "colors lerp per channel");

        // Removing down to zero keyframes collapses to a constant.
        g.remove_shape_param_keyframe(ShapeParam::Width, 0).unwrap();
        g.remove_shape_param_keyframe(ShapeParam::Width, 10).unwrap();
        let Generator::Shape { width, .. } = &g else {
            panic!()
        };
        assert!(!width.is_animated());
    }

    #[test]
    fn shape_param_routing_rejects_wrong_targets() {
        let mut rect = Generator::shape(Shape::Rectangle, [255, 255, 255, 255]);
        // Star-only param on a rectangle.
        assert!(
            rect.set_shape_param_constant(ShapeParam::InnerRatio, ParamValue::Scalar(0.5))
                .is_err()
        );
        // Stroke params with no stroke set.
        assert!(
            rect.set_shape_param_constant(ShapeParam::StrokeWidth, ParamValue::Scalar(4.0))
                .is_err()
        );
        // Kind mismatches both ways.
        assert!(
            rect.set_shape_param_constant(ShapeParam::Width, ParamValue::Color([0, 0, 0, 255]))
                .is_err()
        );
        assert!(
            rect.set_shape_param_constant(ShapeParam::Fill, ParamValue::Scalar(1.0))
                .is_err()
        );
        // Out-of-range values are rejected at the routing boundary.
        assert!(
            rect.set_shape_param_constant(ShapeParam::Width, ParamValue::Scalar(-5.0))
                .is_err()
        );
        // Non-shape generators reject everything.
        let mut text = Generator::text("hi");
        assert!(
            text.set_shape_param_constant(ShapeParam::Width, ParamValue::Scalar(10.0))
                .is_err()
        );
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
            effect_preset: None,
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
            ClipSource::Generated(Generator::Text {
                content,
                style: got,
            }) => {
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
                ..ClipTransform::IDENTITY
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
        let clip = media_clip(MediaId::from_raw(1), tr(0, 10, R24), tr(0, 10, R24));
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
