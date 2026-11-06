//! The Swift and Kotlin surface over `ffi`.
//!
//! One thread owns each open [`cutlass_ffi::PhoneHost`], because its audio
//! output cannot move between threads. That thread runs at the interface's
//! priority. A call sends it a job and waits for the answer, then turns it into plain records. Times are
//! microseconds. A failure is a message for the phone to show; the draft is
//! unchanged.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{self, Sender};
use std::time::UNIX_EPOCH;

use cutlass_ffi::{Drag, FfiError, Home, PhoneHost};
use cutlass_ffi::{Micros, SegmentId};

uniffi::setup_scaffolding!();

/// Why a call left the project as it was.
#[derive(Debug, uniffi::Error)]
#[uniffi(flat_error)]
pub enum EditorError {
    Failed { message: String },
}

impl std::fmt::Display for EditorError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Failed { message } => formatter.write_str(message),
        }
    }
}

impl std::error::Error for EditorError {}

impl From<FfiError> for EditorError {
    fn from(error: FfiError) -> Self {
        Self::Failed {
            message: error.to_string(),
        }
    }
}

/// A composited picture, eight-bit RGBA, rows top to bottom.
#[derive(Clone, Debug, uniffi::Record)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// One clip on a row.
#[derive(Clone, Debug, uniffi::Record)]
pub struct Clip {
    pub id: String,
    pub name: String,
    pub start_us: i64,
    pub duration_us: i64,
    /// Linear gain: 1 is unity and 0 is silence.
    pub volume: f64,
    /// Constant speed: 2 plays twice as fast.
    pub speed: f64,
}

/// What a row holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, uniffi::Enum)]
pub enum LaneKind {
    Video,
    Audio,
    Text,
    Sticker,
    Effect,
    Filter,
    Adjust,
    Other,
}

/// One timeline row, top first.
#[derive(Clone, Debug, uniffi::Record)]
pub struct Lane {
    pub kind: LaneKind,
    pub main: bool,
    pub locked: bool,
    pub clips: Vec<Clip>,
}

/// One ruler tick. A major tick carries its label.
#[derive(Clone, Debug, uniffi::Record)]
pub struct Tick {
    pub time_us: i64,
    pub major: bool,
    pub label: Option<String>,
}

/// One filmstrip tile, in points from the clip's left edge. `key` names the
/// picture for [`Editor::thumb`]; `ready` says it is stored.
#[derive(Clone, Debug, uniffi::Record)]
pub struct Tile {
    pub left: f64,
    pub width: f64,
    pub key: String,
    pub ready: bool,
}

/// One clip's sound: a level, 0 to 1, per point from `first`, points from
/// the clip's left edge. `ready` is false while the file is still read.
#[derive(Clone, Debug, uniffi::Record)]
pub struct Wave {
    pub clip_id: String,
    pub first: u32,
    pub levels: Vec<f32>,
    pub under_picture: bool,
    pub muted: bool,
    pub ready: bool,
}

/// The filmstrip tiles of one clip.
#[derive(Clone, Debug, uniffi::Record)]
pub struct Film {
    pub clip_id: String,
    pub tiles: Vec<Tile>,
}

/// A finished export.
#[derive(Clone, Debug, uniffi::Record)]
pub struct Exported {
    pub path: String,
    pub frames: u64,
    pub width: u32,
    pub height: u32,
}

/// Hears an export's progress: the pictures done and the pictures in all.
/// Returning false stops it.
#[uniffi::export(with_foreign)]
pub trait ExportProgress: Send + Sync {
    fn progress(&self, done: u64, total: u64) -> bool;
}

/// The short sides, in pixels, an export may be, smallest first.
#[uniffi::export]
#[must_use]
pub fn export_sides() -> Vec<u32> {
    cutlass_ffi::PHONE_EXPORT_SIDES.to_vec()
}

/// The transport clock at `time_us`: whole seconds as `MM:SS`, or
/// `HH:MM:SS` from an hour.
#[uniffi::export]
#[must_use]
pub fn clock(time_us: i64) -> String {
    cutlass_ffi::clock(Micros(time_us))
}

/// The effects that ship in the app, in picker order.
#[uniffi::export]
#[must_use]
pub fn effect_choices() -> Vec<EffectChoice> {
    cutlass_ffi::phone_effects()
        .into_iter()
        .map(|effect| EffectChoice {
            id: effect.id,
            name: effect.name,
            preview: effect.preview,
        })
        .collect()
}

/// One step of a timeline coast: points moved, and the speed after.
#[derive(Clone, Copy, Debug, uniffi::Record)]
pub struct Fling {
    pub distance: f64,
    pub velocity: f64,
}

/// The speed, in points a second, a timeline let go at `velocity` coasts
/// from: zero for a slow swipe.
#[uniffi::export]
#[must_use]
pub fn fling_start(velocity: f64) -> f64 {
    cutlass_ffi::fling_start(velocity)
}

/// A coast at `velocity` points a second moved on by `elapsed` seconds.
/// The speed is zero once at rest.
#[uniffi::export]
#[must_use]
pub fn fling_step(velocity: f64, elapsed: f64) -> Fling {
    let step = cutlass_ffi::fling_step(velocity, elapsed);
    Fling {
        distance: step.distance,
        velocity: step.velocity,
    }
}

/// The gain at volume slider `position`, 0 to 1. Unity is the middle and
/// the right end is ten times.
#[uniffi::export]
#[must_use]
pub fn volume_at(position: f64) -> f64 {
    cutlass_ffi::volume_at(position)
}

/// Where `volume` sits on the volume slider, 0 to 1.
#[uniffi::export]
#[must_use]
pub fn volume_position(volume: f64) -> f64 {
    cutlass_ffi::volume_position(volume)
}

/// The speed at scale `position`, 0 to 1. The marks 0.1, 1, 2, 5, 10 and
/// 100 sit the same distance apart.
#[uniffi::export]
#[must_use]
pub fn speed_at(position: f64) -> f64 {
    cutlass_ffi::speed_at(position)
}

/// The Styles tab's colors, as `#RRGGBB`.
#[uniffi::export]
#[must_use]
pub fn text_colors() -> Vec<String> {
    cutlass_ffi::TEXT_COLORS
        .iter()
        .map(|color| (*color).to_owned())
        .collect()
}

/// The Styles tab's slider ends and the values a part starts at. Sizes are
/// pixels, outlines ems, shadow distances hundredths of an em; the rest are
/// 0 to 1.
#[derive(Clone, Copy, Debug, uniffi::Record)]
pub struct TextStyleRanges {
    pub size_min: f64,
    pub size_max: f64,
    pub outline_max: f64,
    pub outline_start: f64,
    pub shadow_distance_max: f64,
    pub shadow_distance_start: f64,
    pub shadow_blur_start: f64,
    pub background_alpha_start: f64,
    pub background_round_start: f64,
    pub glow_strength_start: f64,
    pub glow_range_start: f64,
}

/// See [`TextStyleRanges`].
#[uniffi::export]
#[must_use]
pub fn text_style_ranges() -> TextStyleRanges {
    let ranges = cutlass_ffi::TEXT_STYLE_RANGES;
    TextStyleRanges {
        size_min: ranges.size.0,
        size_max: ranges.size.1,
        outline_max: ranges.outline_max,
        outline_start: ranges.outline_start,
        shadow_distance_max: ranges.shadow_distance_max,
        shadow_distance_start: ranges.shadow_distance_start,
        shadow_blur_start: ranges.shadow_blur_start,
        background_alpha_start: ranges.background_alpha_start,
        background_round_start: ranges.background_round_start,
        glow_strength_start: ranges.glow_strength_start,
        glow_range_start: ranges.glow_range_start,
    }
}

/// The background swatches, as `#RRGGBB`.
#[uniffi::export]
#[must_use]
pub fn background_colors() -> Vec<String> {
    cutlass_ffi::BACKGROUND_COLORS
        .iter()
        .map(|color| (*color).to_owned())
        .collect()
}

/// The background blur strengths, weakest first.
#[uniffi::export]
#[must_use]
pub fn background_blurs() -> Vec<f64> {
    cutlass_ffi::BACKGROUND_BLURS.to_vec()
}

/// What fills the canvas around the main pictures.
#[derive(Clone, Debug, PartialEq, uniffi::Enum)]
pub enum Background {
    /// Clear: black shows.
    None,
    /// The project color, `#RRGGBB`.
    Color { color: String },
    /// Every main picture over a blurred copy of itself.
    Blur { blur: f64 },
}

impl From<cutlass_ffi::PhoneBackground> for Background {
    fn from(background: cutlass_ffi::PhoneBackground) -> Self {
        match background {
            cutlass_ffi::PhoneBackground::None => Self::None,
            cutlass_ffi::PhoneBackground::Color(color) => Self::Color { color },
            cutlass_ffi::PhoneBackground::Blur(blur) => Self::Blur { blur },
        }
    }
}

impl From<Background> for cutlass_ffi::PhoneBackground {
    fn from(background: Background) -> Self {
        match background {
            Background::None => Self::None,
            Background::Color { color } => Self::Color(color),
            Background::Blur { blur } => Self::Blur(blur),
        }
    }
}

/// Where `speed` sits on the speed scale, 0 to 1.
#[uniffi::export]
#[must_use]
pub fn speed_position(speed: f64) -> f64 {
    cutlass_ffi::speed_position(speed)
}

/// A face text can be set in. `name` is what a clip stores; `label` is for
/// people.
#[derive(Clone, Debug, uniffi::Record)]
pub struct Font {
    pub name: String,
    pub label: String,
}

/// A canvas ratio the picker shows. `ratio` is what `set_ratio` takes;
/// `width` and `height` are the frame it gives this project.
#[derive(Clone, Debug, uniffi::Record)]
pub struct CanvasRatio {
    pub ratio: String,
    pub label: String,
    pub width: i32,
    pub height: i32,
    pub chosen: bool,
}

/// A picture clip's box on the canvas, in canvas pixels with y down. The
/// guides say the clip's centre sits on a canvas centre line.
#[derive(Clone, Debug, uniffi::Record)]
pub struct Stage {
    pub canvas_width: f64,
    pub canvas_height: f64,
    pub center_x: f64,
    pub center_y: f64,
    pub width: f64,
    pub height: f64,
    /// Degrees clockwise on screen.
    pub turn: f64,
    pub vertical_guide: bool,
    pub horizontal_guide: bool,
    /// The clip is a title, not a picture.
    pub text: bool,
}

/// A clip's keyframes as timeline times, earliest first, the one under the
/// playhead, and the span between two of them the playhead is in.
#[derive(Clone, Debug, uniffi::Record)]
pub struct Keyframes {
    pub times_us: Vec<i64>,
    /// A keyframe is within a frame of the playhead.
    pub here: bool,
    /// The playhead is in a span between two keyframes, which Graphs eases.
    pub in_span: bool,
    /// The Graphs curve easing that span, or empty for a straight one.
    pub graph: String,
}

impl From<cutlass_ffi::PhoneKeys> for Keyframes {
    fn from(keys: cutlass_ffi::PhoneKeys) -> Self {
        Self {
            times_us: keys.times.iter().map(|time| time.0).collect(),
            here: keys.here.is_some(),
            in_span: keys.span.is_some(),
            graph: keys.graph.unwrap_or_default(),
        }
    }
}

/// A clip's in and out animations, each a preset's resource id and length
/// in microseconds; an empty id is none.
#[derive(uniffi::Record)]
pub struct ClipAnimations {
    pub in_id: String,
    pub in_us: i64,
    pub out_id: String,
    pub out_us: i64,
    /// The clip's length, which the two share.
    pub length_us: i64,
}

impl From<cutlass_ffi::PhoneAnimations> for ClipAnimations {
    fn from(animations: cutlass_ffi::PhoneAnimations) -> Self {
        let side = |side: Option<cutlass_ffi::PhoneAnimation>| {
            side.map_or((String::new(), 0), |side| {
                (side.resource_id, side.duration.0)
            })
        };
        let (in_id, in_us) = side(animations.enter);
        let (out_id, out_us) = side(animations.leave);
        Self {
            in_id,
            in_us,
            out_id,
            out_us,
            length_us: animations.length.0,
        }
    }
}

/// One animation in the Animations panel.
#[derive(uniffi::Record)]
pub struct AnimationPreset {
    pub name: String,
    /// On the In tab; otherwise on Out.
    pub enter: bool,
    pub id: String,
    /// The length a new one starts with.
    pub duration_us: i64,
}

/// The Animations panel's presets, in its order after None.
#[uniffi::export]
#[must_use]
pub fn animation_presets() -> Vec<AnimationPreset> {
    cutlass_ffi::animation_presets()
        .into_iter()
        .map(|preset| AnimationPreset {
            name: preset.name,
            enter: preset.enter,
            id: preset.resource_id,
            duration_us: preset.duration.0,
        })
        .collect()
}

/// A Graphs curve: its name and its line through the unit square, x the
/// share of time and y the share of the change, y up.
#[derive(Clone, Debug, uniffi::Record)]
pub struct GraphCurve {
    pub name: String,
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
}

/// The Graphs panel's curves, in its order after None.
#[uniffi::export]
#[must_use]
pub fn graph_curves() -> Vec<GraphCurve> {
    cutlass_ffi::graphs()
        .into_iter()
        .map(|graph| GraphCurve {
            name: graph.name,
            xs: graph.points.iter().map(|point| point.0).collect(),
            ys: graph.points.iter().map(|point| point.1).collect(),
        })
        .collect()
}

impl From<cutlass_ffi::PhoneStage> for Stage {
    fn from(stage: cutlass_ffi::PhoneStage) -> Self {
        Self {
            canvas_width: stage.canvas_width,
            canvas_height: stage.canvas_height,
            center_x: stage.center_x,
            center_y: stage.center_y,
            width: stage.width,
            height: stage.height,
            turn: stage.turn,
            vertical_guide: stage.vertical_guide,
            horizontal_guide: stage.horizontal_guide,
            text: stage.text,
        }
    }
}

/// An effect the picker shows. `id` is what `add_effect` takes; `preview`
/// is the file name of its looping preview video in the app bundle.
#[derive(Clone, Debug, uniffi::Record)]
pub struct EffectChoice {
    pub id: String,
    pub name: String,
    pub preview: String,
}

/// What a text clip says, the face it says it in, and how it looks.
#[derive(Clone, Debug, uniffi::Record)]
pub struct TextInfo {
    pub content: String,
    pub face: String,
    pub style: TextStyle,
}

/// How a title looks. A part with an empty color is off. Widths and
/// distances are in ems and hundredths of an em; the rest are 0 to 1.
#[derive(Clone, Debug, uniffi::Record)]
pub struct TextStyle {
    pub size: f64,
    pub color: String,
    pub outline_color: String,
    pub outline_width: f64,
    pub shadow_color: String,
    pub shadow_distance: f64,
    pub shadow_blur: f64,
    pub background_color: String,
    pub background_alpha: f64,
    pub background_round: f64,
    pub glow_color: String,
    pub glow_strength: f64,
    pub glow_range: f64,
}

impl From<cutlass_ffi::PhoneTextStyle> for TextStyle {
    fn from(style: cutlass_ffi::PhoneTextStyle) -> Self {
        Self {
            size: style.size,
            color: style.color,
            outline_color: style.outline_color,
            outline_width: style.outline_width,
            shadow_color: style.shadow_color,
            shadow_distance: style.shadow_distance,
            shadow_blur: style.shadow_blur,
            background_color: style.background_color,
            background_alpha: style.background_alpha,
            background_round: style.background_round,
            glow_color: style.glow_color,
            glow_strength: style.glow_strength,
            glow_range: style.glow_range,
        }
    }
}

/// One change to a title's look. See [`TextStyle`] for the units.
#[derive(Clone, Debug, PartialEq, uniffi::Enum)]
pub enum TextStyleChange {
    Size {
        size: f64,
    },
    Color {
        color: String,
    },
    Outline {
        color: String,
        width: f64,
    },
    Shadow {
        color: String,
        distance: f64,
        blur: f64,
    },
    Background {
        color: String,
        alpha: f64,
        round: f64,
    },
    Glow {
        color: String,
        strength: f64,
        range: f64,
    },
}

impl From<TextStyleChange> for cutlass_ffi::TextStyleChange {
    fn from(change: TextStyleChange) -> Self {
        match change {
            TextStyleChange::Size { size } => Self::Size(size),
            TextStyleChange::Color { color } => Self::Color(color),
            TextStyleChange::Outline { color, width } => Self::Outline { color, width },
            TextStyleChange::Shadow {
                color,
                distance,
                blur,
            } => Self::Shadow {
                color,
                distance,
                blur,
            },
            TextStyleChange::Background {
                color,
                alpha,
                round,
            } => Self::Background {
                color,
                alpha,
                round,
            },
            TextStyleChange::Glow {
                color,
                strength,
                range,
            } => Self::Glow {
                color,
                strength,
                range,
            },
        }
    }
}

/// A phone tool bar. See [`tool_bar`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, uniffi::Enum)]
pub enum ToolBar {
    Root,
    MainClip,
    OverlayClip,
    AudioClip,
    TextClip,
    OtherClip,
    Audio,
    Text,
    Effects,
    Overlay,
}

/// One tool. Each shell maps it to its own icon.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, uniffi::Enum)]
pub enum ToolKind {
    Edit,
    Audio,
    Text,
    Effects,
    Overlay,
    Captions,
    Filters,
    Adjust,
    Stickers,
    AspectRatio,
    Background,
    Split,
    Volume,
    Speed,
    Animations,
    Delete,
    Duplicate,
    Replace,
    Crop,
    Mirror,
    Rotate,
    Reverse,
    Freeze,
    ExtractAudio,
    Fade,
    Beats,
    Unlink,
    Splice,
    AddOverlay,
    ImportSound,
    Record,
    AddText,
    EditText,
    Draw,
    VideoEffects,
}

/// A tool as a bar shows it. `ready` says it works on the phone today;
/// `opens` names the bar it opens instead of acting.
#[derive(Clone, Debug, uniffi::Record)]
pub struct ToolItem {
    pub kind: ToolKind,
    pub label: String,
    pub ready: bool,
    pub opens: Option<ToolBar>,
}

/// The tools on `bar`, in order.
#[uniffi::export]
#[must_use]
pub fn tool_bar(bar: ToolBar) -> Vec<ToolItem> {
    cutlass_ffi::tools(bar_in(bar))
        .iter()
        .map(|tool| ToolItem {
            kind: tool_kind(tool.id),
            label: tool.label.to_owned(),
            ready: tool.ready,
            opens: tool.opens.map(bar_out),
        })
        .collect()
}

fn bar_in(bar: ToolBar) -> cutlass_ffi::Bar {
    use cutlass_ffi::Bar;
    match bar {
        ToolBar::Root => Bar::Root,
        ToolBar::MainClip => Bar::MainClip,
        ToolBar::OverlayClip => Bar::OverlayClip,
        ToolBar::AudioClip => Bar::AudioClip,
        ToolBar::TextClip => Bar::TextClip,
        ToolBar::OtherClip => Bar::OtherClip,
        ToolBar::Audio => Bar::Audio,
        ToolBar::Text => Bar::Text,
        ToolBar::Effects => Bar::Effects,
        ToolBar::Overlay => Bar::Overlay,
    }
}

fn bar_out(bar: cutlass_ffi::Bar) -> ToolBar {
    use cutlass_ffi::Bar;
    match bar {
        Bar::Root => ToolBar::Root,
        Bar::MainClip => ToolBar::MainClip,
        Bar::OverlayClip => ToolBar::OverlayClip,
        Bar::AudioClip => ToolBar::AudioClip,
        Bar::TextClip => ToolBar::TextClip,
        Bar::OtherClip => ToolBar::OtherClip,
        Bar::Audio => ToolBar::Audio,
        Bar::Text => ToolBar::Text,
        Bar::Effects => ToolBar::Effects,
        Bar::Overlay => ToolBar::Overlay,
    }
}

fn tool_kind(id: cutlass_ffi::ToolId) -> ToolKind {
    use cutlass_ffi::ToolId as T;
    match id {
        T::Edit => ToolKind::Edit,
        T::Audio => ToolKind::Audio,
        T::Text => ToolKind::Text,
        T::Effects => ToolKind::Effects,
        T::Overlay => ToolKind::Overlay,
        T::Captions => ToolKind::Captions,
        T::Filters => ToolKind::Filters,
        T::Adjust => ToolKind::Adjust,
        T::Stickers => ToolKind::Stickers,
        T::AspectRatio => ToolKind::AspectRatio,
        T::Background => ToolKind::Background,
        T::Split => ToolKind::Split,
        T::Volume => ToolKind::Volume,
        T::Speed => ToolKind::Speed,
        T::Animations => ToolKind::Animations,
        T::Delete => ToolKind::Delete,
        T::Duplicate => ToolKind::Duplicate,
        T::Replace => ToolKind::Replace,
        T::Crop => ToolKind::Crop,
        T::Mirror => ToolKind::Mirror,
        T::Rotate => ToolKind::Rotate,
        T::Reverse => ToolKind::Reverse,
        T::Freeze => ToolKind::Freeze,
        T::ExtractAudio => ToolKind::ExtractAudio,
        T::Fade => ToolKind::Fade,
        T::Beats => ToolKind::Beats,
        T::Unlink => ToolKind::Unlink,
        T::Splice => ToolKind::Splice,
        T::AddOverlay => ToolKind::AddOverlay,
        T::ImportSound => ToolKind::ImportSound,
        T::Record => ToolKind::Record,
        T::AddText => ToolKind::AddText,
        T::EditText => ToolKind::EditText,
        T::Draw => ToolKind::Draw,
        T::VideoEffects => ToolKind::VideoEffects,
    }
}

/// One saved project in the launch list. `modified_ms` is milliseconds
/// since 1970.
#[derive(Clone, Debug, uniffi::Record)]
pub struct ProjectSummary {
    pub path: String,
    pub name: String,
    pub duration_us: i64,
    pub modified_ms: i64,
}

/// Lay out the Cutlass folder under `root` and open the process log in it.
///
/// `root` is the one folder everything Cutlass writes lives under: the
/// projects, the cache, the downloads, the proxies, the models, and the
/// logs. A phone passes a folder in its app container, once at launch.
/// Calling it again with the same root changes nothing.
#[uniffi::export]
pub fn start(root: String) -> Result<(), EditorError> {
    let home = open_home(&root)?;
    cutlass_ffi::start_log(&home.logs()).map_err(|error| EditorError::Failed {
        message: error.to_string(),
    })?;
    Ok(())
}

/// The projects in the Cutlass folder at `root`, newest first. A missing
/// root lists nothing.
#[uniffi::export]
#[must_use]
pub fn list_projects(root: String) -> Vec<ProjectSummary> {
    let Ok(home) = Home::at(root) else {
        return Vec::new();
    };
    cutlass_ffi::list_projects(&home.projects())
        .into_iter()
        .map(|project| ProjectSummary {
            path: project.path.display().to_string(),
            name: project.name,
            duration_us: project.duration.0,
            modified_ms: project
                .modified
                .duration_since(UNIX_EPOCH)
                .map_or(0, |since| {
                    i64::try_from(since.as_millis()).unwrap_or(i64::MAX)
                }),
        })
        .collect()
}

/// Write a new empty project in the Cutlass folder at `root`, named like the
/// desktop's, and return its folder.
#[uniffi::export]
pub fn create_project(root: String) -> Result<String, EditorError> {
    let home = open_home(&root)?;
    cutlass_ffi::create_project(&home.projects())
        .map(|path| path.display().to_string())
        .map_err(|error| EditorError::from(FfiError::from(error)))
}

/// The app hears this device. Tests keep the software buffer, so they need
/// no speaker.
fn device_audio() -> cutlass_ffi::AudioHost {
    if cfg!(test) {
        cutlass_ffi::AudioHost::Software
    } else {
        cutlass_ffi::AudioHost::current()
    }
}

fn open_home(root: &str) -> Result<Home, EditorError> {
    Home::open(root).map_err(|error| EditorError::from(FfiError::from(error)))
}

/// A project folder must sit directly in the root's projects folder.
fn project_in(home: &Home, project: &Path) -> Result<(), EditorError> {
    if project.parent() == Some(home.projects().as_path()) {
        Ok(())
    } else {
        Err(EditorError::Failed {
            message: format!(
                "{} is not a project in the Cutlass folder",
                project.display()
            ),
        })
    }
}

/// One open project.
#[derive(uniffi::Object)]
pub struct Editor {
    jobs: Sender<Job>,
}

type Job = Box<dyn FnOnce(&mut PhoneHost) + Send>;

#[uniffi::export]
impl Editor {
    /// Open a project folder in the Cutlass folder at `root`, or create an
    /// empty one when it has no draft. `bundle` is the app's read-only
    /// resources and must be a directory. The cache is the root's.
    #[uniffi::constructor]
    pub fn open(root: String, bundle: String, project: String) -> Result<Arc<Self>, EditorError> {
        let home = open_home(&root)?;
        let project = PathBuf::from(project);
        project_in(&home, &project)?;
        let cache = home.cache();
        let (jobs, inbox) = mpsc::channel::<Job>();
        let (opened, answer) = mpsc::sync_channel(1);
        std::thread::Builder::new()
            .name("cutlass-editor".to_owned())
            .spawn(move || {
                cutlass_ffi::serve_the_interface();
                let bundle = PathBuf::from(bundle);
                let host = if project.join("draft_info.json").is_file() {
                    PhoneHost::open(&project, &bundle, &cache, 0)
                } else {
                    PhoneHost::create(&project, "draft", &bundle, &cache, 0)
                };
                let mut host = match host {
                    Ok(mut host) => {
                        host.open_audio(device_audio());
                        let _ = opened.send(Ok(()));
                        host
                    }
                    Err(error) => {
                        let _ = opened.send(Err(EditorError::from(error)));
                        return;
                    }
                };
                // The loop ends when the editor is dropped.
                while let Ok(job) = inbox.recv() {
                    job(&mut host);
                }
            })
            .map_err(|error| EditorError::Failed {
                message: error.to_string(),
            })?;
        answer.recv().map_err(|_| closed())??;
        Ok(Arc::new(Self { jobs }))
    }

    /// Read a picked file, copy it into the project, and add it to the
    /// library. Rust probes it; the phone only hands over the path.
    /// Returns the material id.
    pub fn import(&self, path: String) -> Result<String, EditorError> {
        self.run(move |host| host.import(&PathBuf::from(path)))?
            .map_err(EditorError::from)
    }

    /// Add a library item at the end of its lane. Returns the clip id.
    pub fn append(&self, material_id: String) -> Result<String, EditorError> {
        self.run(move |host| host.append(&material_id))?
            .map(|id| id.as_str().to_owned())
            .map_err(EditorError::from)
    }

    /// Look for faces in `folders` and the folders below them, replacing
    /// any found before.
    pub fn set_font_folders(&self, folders: Vec<String>) -> Result<(), EditorError> {
        let folders: Vec<PathBuf> = folders.into_iter().map(Into::into).collect();
        self.run(move |host| host.set_font_folders(&folders))
    }

    /// The faces text can be set in, by file name.
    pub fn fonts(&self) -> Result<Vec<Font>, EditorError> {
        let fonts = self.run(|host| host.fonts())?;
        Ok(fonts
            .into_iter()
            .map(|font| Font {
                name: font.name,
                label: font.label,
            })
            .collect())
    }

    /// Add `content` at `at_us` in `face`, or in the first face when `face`
    /// is empty, as one undo step. Returns the new clip's id.
    pub fn add_text(
        &self,
        content: String,
        face: String,
        at_us: i64,
    ) -> Result<String, EditorError> {
        self.run(move |host| host.add_text(&content, &face, Micros(at_us)))?
            .map(|id| id.as_str().to_owned())
            .map_err(EditorError::from)
    }

    /// Type into a title as the words change. With no `clip_id`, the first
    /// words that are not blank add one at `at_us` in `face`. The whole
    /// typing is one undo step; blank words change nothing. Returns the
    /// clip, or nothing when none has been added yet.
    pub fn type_text(
        &self,
        clip_id: Option<String>,
        content: String,
        face: String,
        at_us: i64,
    ) -> Result<Option<String>, EditorError> {
        self.run(move |host| {
            let clip = clip_id.map(SegmentId::new);
            host.type_text(clip.as_ref(), &content, &face, Micros(at_us))
        })?
        .map(|id| id.map(|id| id.as_str().to_owned()))
        .map_err(EditorError::from)
    }

    /// Add the effect `effect_id` at `at_us` over everything below it, as
    /// one undo step. Returns the new clip's id.
    pub fn add_effect(&self, effect_id: String, at_us: i64) -> Result<String, EditorError> {
        self.run(move |host| host.add_effect(&effect_id, Micros(at_us)))?
            .map(|id| id.as_str().to_owned())
            .map_err(EditorError::from)
    }

    /// Bring in a subtitle file (SRT or LRC, by the extension of
    /// `file_name`) whose `text` the phone read, as captions on a new
    /// subtitle lane: one undo step.
    pub fn import_subtitles(&self, file_name: String, text: String) -> Result<(), EditorError> {
        self.run(move |host| host.import_subtitles(&file_name, &text))?
            .map_err(EditorError::from)
    }

    /// Change what a text clip says, as one undo step.
    pub fn set_text(&self, clip_id: String, content: String) -> Result<(), EditorError> {
        self.run(move |host| host.set_text(&SegmentId::new(clip_id), &content))?
            .map_err(EditorError::from)
    }

    /// Set a text clip in `face`, as one undo step.
    pub fn set_text_font(&self, clip_id: String, face: String) -> Result<(), EditorError> {
        self.run(move |host| host.set_text_font(&SegmentId::new(clip_id), &face))?
            .map_err(EditorError::from)
    }

    /// Change how a title looks. Changes made while `live` join one undo
    /// step, so a slider drag is one step; its last change passes `live`
    /// false, and so does a single tap.
    pub fn style_text(
        &self,
        clip_id: String,
        change: TextStyleChange,
        live: bool,
    ) -> Result<(), EditorError> {
        self.run(move |host| host.style_text(&SegmentId::new(clip_id), change.into(), live))?
            .map_err(EditorError::from)
    }

    /// What a text clip says, or `None` when it is not a text clip.
    pub fn text(&self, clip_id: String) -> Result<Option<TextInfo>, EditorError> {
        let text = self.run(move |host| host.text(&SegmentId::new(clip_id)))?;
        Ok(text.map(|text| TextInfo {
            content: text.content,
            face: text.face,
            style: text.style.into(),
        }))
    }

    /// The rows to draw, top first.
    pub fn lanes(&self) -> Result<Vec<Lane>, EditorError> {
        let lanes = self.run(|host| host.lanes())?;
        Ok(lanes
            .into_iter()
            .map(|lane| Lane {
                kind: lane_kind(lane.kind),
                main: lane.main,
                locked: lane.locked,
                clips: lane
                    .clips
                    .into_iter()
                    .map(|clip| Clip {
                        id: clip.id.as_str().to_owned(),
                        name: clip.name,
                        start_us: clip.start.0,
                        duration_us: clip.duration.0,
                        volume: clip.volume,
                        speed: clip.speed,
                    })
                    .collect(),
            })
            .collect())
    }

    /// The tool bar for `clip_id` while it is selected, or `None` for an id
    /// no row holds.
    pub fn clip_bar(&self, clip_id: String) -> Result<Option<ToolBar>, EditorError> {
        let bar = self.run(move |host| host.clip_bar(&SegmentId::new(clip_id)))?;
        Ok(bar.map(bar_out))
    }

    /// The end of the last clip, in microseconds.
    pub fn duration_us(&self) -> Result<i64, EditorError> {
        self.run(|host| host.draft().duration.0)
    }

    /// Play from the playhead with sound. An empty project does not play.
    pub fn play(&self) -> Result<(), EditorError> {
        self.run(PhoneHost::play)?.map_err(EditorError::from)
    }

    pub fn pause(&self) -> Result<(), EditorError> {
        self.run(PhoneHost::pause)
    }

    /// Move the playhead to `time_us`, held inside the project. Playing
    /// goes on from there.
    pub fn seek(&self, time_us: i64) -> Result<(), EditorError> {
        self.run(move |host| host.seek(Micros(time_us)))
    }

    /// Move a playing playhead by `elapsed_us` with the heard sound and
    /// return it. Call once per screen frame while playing.
    pub fn tick(&self, elapsed_us: i64) -> Result<i64, EditorError> {
        self.run(move |host| host.tick(elapsed_us).0)
    }

    /// Play stopped at the end, or was never started.
    pub fn is_playing(&self) -> Result<bool, EditorError> {
        self.run(|host| host.is_playing())
    }

    /// The picture at `time_us`.
    pub fn frame(&self, time_us: i64) -> Result<Frame, EditorError> {
        let picture = self
            .run(move |host| host.frame(Micros(time_us)))?
            .map_err(EditorError::from)?;
        Ok(Frame {
            width: picture.width,
            height: picture.height,
            rgba: picture.rgba,
        })
    }

    /// Cut a clip at `at_us`. Returns the right piece's id.
    pub fn split(&self, clip_id: String, at_us: i64) -> Result<String, EditorError> {
        self.run(move |host| host.split(&SegmentId::new(clip_id), Micros(at_us)))?
            .map(|id| id.as_str().to_owned())
            .map_err(EditorError::from)
    }

    /// Move a clip to `row` (top is 0) starting near `start_us`, as one undo
    /// step. It snaps to a cut or the playhead within `snap_us`. Returns the
    /// start it landed on.
    pub fn move_clip(
        &self,
        clip_id: String,
        row: u32,
        start_us: i64,
        snap_us: i64,
        playhead_us: i64,
    ) -> Result<i64, EditorError> {
        let drag = Drag {
            row,
            time: Micros(start_us),
            snap: Micros(snap_us),
            playhead: Micros(playhead_us),
        };
        self.run(move |host| host.move_clip(&SegmentId::new(clip_id), drag))?
            .map(|start| start.0)
            .map_err(EditorError::from)
    }

    /// Drag a clip's start edge, or its end edge, to `time_us`, as one undo
    /// step. Returns where the edge landed.
    pub fn trim(
        &self,
        clip_id: String,
        start_edge: bool,
        time_us: i64,
        snap_us: i64,
        playhead_us: i64,
    ) -> Result<i64, EditorError> {
        let drag = Drag {
            row: 0,
            time: Micros(time_us),
            snap: Micros(snap_us),
            playhead: Micros(playhead_us),
        };
        self.run(move |host| host.trim(&SegmentId::new(clip_id), start_edge, drag))?
            .map(|time| time.0)
            .map_err(EditorError::from)
    }

    /// Remove one clip.
    /// Set a clip's linear gain, as one undo step. Below zero is refused.
    pub fn set_volume(&self, clip_id: String, volume: f64) -> Result<(), EditorError> {
        self.run(move |host| host.set_volume(&SegmentId::new(clip_id), volume))?
            .map_err(EditorError::from)
    }

    /// The box of a picture clip on the canvas at `at_us`, where its
    /// keyframes put it. Other clips have none.
    pub fn stage(&self, clip_id: String, at_us: i64) -> Result<Option<Stage>, EditorError> {
        Ok(self
            .run(move |host| host.stage(&SegmentId::new(clip_id), Micros(at_us)))?
            .map(Stage::from))
    }

    /// The topmost picture clip under the canvas point `(x, y)` at `at_us`.
    pub fn stage_pick(&self, at_us: i64, x: f64, y: f64) -> Result<Option<String>, EditorError> {
        Ok(self
            .run(move |host| host.stage_pick(Micros(at_us), x, y))?
            .map(|id| id.as_str().to_owned()))
    }

    /// Start moving a picture clip on the preview at `at_us`. A clip with
    /// keyframes keys its new pose there.
    pub fn stage_begin(&self, clip_id: String, at_us: i64) -> Result<(), EditorError> {
        self.run(move |host| host.stage_begin(&SegmentId::new(clip_id), Micros(at_us)))?
            .map_err(EditorError::from)
    }

    /// A picture or title clip's keyframes, and whether one is under
    /// `at_us`. Other clips have none.
    pub fn keyframes(&self, clip_id: String, at_us: i64) -> Result<Option<Keyframes>, EditorError> {
        Ok(self
            .run(move |host| host.keyframes(&SegmentId::new(clip_id), Micros(at_us)))?
            .map(Keyframes::from))
    }

    /// Ease the span between keyframes that `at_us` is in along the Graphs
    /// curve `graph`, or make it straight with an empty name. One undo step.
    pub fn ease_keyframes(
        &self,
        clip_id: String,
        at_us: i64,
        graph: String,
    ) -> Result<(), EditorError> {
        let graph = (!graph.is_empty()).then_some(graph);
        self.run(move |host| host.ease_keyframes(&SegmentId::new(clip_id), Micros(at_us), graph))?
            .map_err(EditorError::from)
    }

    /// The picture or title clip's in and out animations, or nil for a clip
    /// that takes none.
    pub fn animations(&self, clip_id: String) -> Result<Option<ClipAnimations>, EditorError> {
        Ok(self
            .run(move |host| host.animations(&SegmentId::new(clip_id)))?
            .map(ClipAnimations::from))
    }

    /// Give the clip the in (`enter`) or out animation `id`, `duration_us`
    /// long, or clear that side with an empty id. One undo step.
    pub fn set_animation(
        &self,
        clip_id: String,
        enter: bool,
        id: String,
        duration_us: i64,
    ) -> Result<(), EditorError> {
        let id = (!id.is_empty()).then_some(id);
        self.run(move |host| {
            host.set_animation(&SegmentId::new(clip_id), enter, id, Micros(duration_us))
        })?
        .map_err(EditorError::from)
    }

    /// The keyframe button: remove the keyframe under `at_us`, else key the
    /// clip there. One undo step.
    pub fn toggle_keyframe(&self, clip_id: String, at_us: i64) -> Result<(), EditorError> {
        self.run(move |host| host.toggle_keyframe(&SegmentId::new(clip_id), Micros(at_us)))?
            .map_err(EditorError::from)
    }

    /// Place the clip `(shift_x, shift_y)` canvas pixels from where the
    /// gesture started, `pinch` times its size, turned `turn` degrees
    /// clockwise. The whole gesture is one undo step.
    pub fn stage_move(
        &self,
        shift_x: f64,
        shift_y: f64,
        pinch: f64,
        turn: f64,
    ) -> Result<Stage, EditorError> {
        self.run(move |host| host.stage_move(shift_x, shift_y, pinch, turn))?
            .map(Stage::from)
            .map_err(EditorError::from)
    }

    /// End the preview gesture.
    pub fn stage_end(&self) -> Result<(), EditorError> {
        self.run(PhoneHost::stage_end)
    }

    /// The canvas ratios, Original first.
    pub fn ratios(&self) -> Result<Vec<CanvasRatio>, EditorError> {
        let ratios = self.run(|host| host.ratios())?;
        Ok(ratios
            .into_iter()
            .map(|ratio| CanvasRatio {
                ratio: ratio.ratio,
                label: ratio.label,
                width: ratio.width,
                height: ratio.height,
                chosen: ratio.chosen,
            })
            .collect())
    }

    /// What fills the canvas around the main pictures.
    pub fn background(&self) -> Result<Background, EditorError> {
        Ok(self.run(|host| host.background())?.into())
    }

    /// Fill the canvas around the main pictures, as one undo step.
    pub fn set_background(&self, background: Background) -> Result<(), EditorError> {
        let background = cutlass_ffi::PhoneBackground::from(background);
        self.run(move |host| host.set_background(&background))?
            .map_err(EditorError::from)
    }

    /// Give the canvas a ratio from [`Self::ratios`], as one undo step.
    pub fn set_ratio(&self, ratio: String) -> Result<(), EditorError> {
        self.run(move |host| host.set_ratio(&ratio))?
            .map_err(EditorError::from)
    }

    /// Play a clip at a constant speed, as one undo step. Its length follows.
    pub fn set_speed(&self, clip_id: String, speed: f64) -> Result<(), EditorError> {
        self.run(move |host| host.set_speed(&SegmentId::new(clip_id), speed))?
            .map_err(EditorError::from)
    }

    /// Put a main clip at `place` in the main row's order, counted from zero
    /// without it, as one undo step.
    pub fn reorder(&self, clip_id: String, place: u32) -> Result<(), EditorError> {
        self.run(move |host| host.reorder(&SegmentId::new(clip_id), place))?
            .map_err(EditorError::from)
    }

    /// Ruler ticks from `from_us` through `to_us` at `pixel_size` points per
    /// microsecond.
    pub fn ruler(
        &self,
        pixel_size: f64,
        from_us: i64,
        to_us: i64,
    ) -> Result<Vec<Tick>, EditorError> {
        let ticks = self.run(move |host| host.ruler(pixel_size, Micros(from_us), Micros(to_us)))?;
        Ok(ticks
            .into_iter()
            .map(|tick| Tick {
                time_us: tick.time.0,
                major: tick.major,
                label: tick.label,
            })
            .collect())
    }

    /// `pixel_size` times `factor`, held inside what the cut and a row
    /// `lane_width` points wide allow.
    pub fn zoom(&self, pixel_size: f64, factor: f64, lane_width: f64) -> Result<f64, EditorError> {
        self.run(move |host| host.zoom(pixel_size, factor, lane_width))
    }

    /// Filmstrip tiles, `height` points tall, for the picture clips that
    /// meet `from_us` to `to_us`. Pictures not stored yet are asked for.
    pub fn film(
        &self,
        from_us: i64,
        to_us: i64,
        pixel_size: f64,
        height: f64,
    ) -> Result<Vec<Film>, EditorError> {
        let films =
            self.run(move |host| host.film(Micros(from_us), Micros(to_us), pixel_size, height))?;
        Ok(films
            .into_iter()
            .map(|film| Film {
                clip_id: film.id.as_str().to_owned(),
                tiles: film
                    .tiles
                    .into_iter()
                    .map(|tile| Tile {
                        left: tile.left,
                        width: tile.width,
                        key: tile.key,
                        ready: tile.ready,
                    })
                    .collect(),
            })
            .collect())
    }

    /// The waveforms of the clips with sound from `from_us` to `to_us`.
    pub fn waves(
        &self,
        from_us: i64,
        to_us: i64,
        pixel_size: f64,
    ) -> Result<Vec<Wave>, EditorError> {
        let waves = self.run(move |host| host.waves(Micros(from_us), Micros(to_us), pixel_size))?;
        Ok(waves
            .into_iter()
            .map(|wave| Wave {
                clip_id: wave.id.as_str().to_owned(),
                first: wave.first,
                levels: wave.levels,
                under_picture: wave.under_picture,
                muted: wave.muted,
                ready: wave.ready,
            })
            .collect())
    }

    /// The picture of a tile [`Self::film`] named, once stored.
    pub fn thumb(&self, key: String) -> Result<Option<Frame>, EditorError> {
        let picture = self.run(move |host| host.thumb(&key))?;
        Ok(picture.map(|picture| Frame {
            width: picture.width,
            height: picture.height,
            rgba: picture.rgba,
        }))
    }

    /// Write the whole cut to `path` as an MP4, `short_side` pixels on its
    /// short side: one of [`export_sides`]. Play stops first. `progress`
    /// hears each picture on the editor's own thread and stops the export
    /// by returning false; a stopped or failed export leaves no file. Every
    /// other call waits until it is done.
    pub fn export(
        &self,
        path: String,
        short_side: u32,
        progress: Arc<dyn ExportProgress>,
    ) -> Result<Exported, EditorError> {
        let exported = self.run(move |host| {
            host.export(&PathBuf::from(path), short_side, &mut |done, total| {
                progress.progress(done, total)
            })
        })??;
        Ok(Exported {
            path: exported.path.to_string_lossy().into_owned(),
            frames: exported.frames,
            width: exported.width,
            height: exported.height,
        })
    }

    pub fn remove(&self, clip_id: String) -> Result<(), EditorError> {
        self.run(move |host| host.remove(&SegmentId::new(clip_id)))?
            .map_err(EditorError::from)
    }

    pub fn undo(&self) -> Result<(), EditorError> {
        self.run(|host| host.undo().map(|_| ()))?
            .map_err(EditorError::from)
    }

    pub fn redo(&self) -> Result<(), EditorError> {
        self.run(|host| host.redo().map(|_| ()))?
            .map_err(EditorError::from)
    }

    /// Write the draft into the project folder.
    pub fn save(&self) -> Result<(), EditorError> {
        self.run(PhoneHost::save)?.map_err(EditorError::from)
    }
}

impl Editor {
    /// Run `job` on the host's thread and wait for its answer.
    fn run<T: Send + 'static>(
        &self,
        job: impl FnOnce(&mut PhoneHost) -> T + Send + 'static,
    ) -> Result<T, EditorError> {
        let (reply, answer) = mpsc::sync_channel(1);
        self.jobs
            .send(Box::new(move |host| {
                let _ = reply.send(job(host));
            }))
            .map_err(|_| closed())?;
        answer.recv().map_err(|_| closed())
    }
}

fn closed() -> EditorError {
    EditorError::Failed {
        message: "the project is closed".to_owned(),
    }
}

fn lane_kind(kind: cutlass_ffi::TrackType) -> LaneKind {
    use cutlass_ffi::TrackType;
    match kind {
        TrackType::Video => LaneKind::Video,
        TrackType::Audio => LaneKind::Audio,
        TrackType::Text => LaneKind::Text,
        TrackType::Sticker => LaneKind::Sticker,
        TrackType::Effect => LaneKind::Effect,
        TrackType::Filter => LaneKind::Filter,
        TrackType::Adjust => LaneKind::Adjust,
        _ => LaneKind::Other,
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::{Editor, LaneKind, create_project, list_projects, start};

    struct Temp(PathBuf);

    impl Temp {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("cutlass-bindings-{}-{n}", std::process::id()));
            std::fs::create_dir_all(path.join("bundle")).unwrap();
            Self(path)
        }

        fn root(&self) -> String {
            self.0.join("Cutlass").display().to_string()
        }

        fn bundle(&self) -> String {
            self.0.join("bundle").display().to_string()
        }

        fn project(&self) -> String {
            self.0
                .join("Cutlass/Projects/project")
                .display()
                .to_string()
        }

        fn open(&self) -> Arc<Editor> {
            Editor::open(self.root(), self.bundle(), self.project()).unwrap()
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn pattern() -> String {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../engine/fixtures/pattern.png")
            .display()
            .to_string()
    }

    #[test]
    fn a_clip_trims_and_moves_and_a_bad_drop_says_why() {
        let temp = Temp::new();
        let editor = temp.open();
        let first = editor.append(editor.import(pattern()).unwrap()).unwrap();
        let second = editor.append(editor.import(pattern()).unwrap()).unwrap();
        let length = editor.lanes().unwrap()[0].clips[0].duration_us;
        let edge = editor
            .trim(second.clone(), false, length + length / 2, 0, 0)
            .unwrap();
        assert_eq!(edge, length + length / 2);
        assert_eq!(editor.duration_us().unwrap(), length + length / 2);
        let start = editor
            .move_clip(first.clone(), 0, 10 * length, 0, 0)
            .unwrap();
        let clips = &editor.lanes().unwrap()[0].clips;
        assert_eq!(
            clips.iter().find(|clip| clip.id == first).unwrap().start_us,
            start
        );
        let super::EditorError::Failed { message } =
            editor.move_clip(first, 9, 0, 0, 0).unwrap_err();
        assert!(!message.is_empty());
    }

    #[test]
    fn volume_and_speed_show_on_the_clip_and_a_negative_volume_says_why() {
        let temp = Temp::new();
        let editor = temp.open();
        let clip = editor.append(editor.import(pattern()).unwrap()).unwrap();
        let length = editor.duration_us().unwrap();
        editor.set_volume(clip.clone(), 0.25).unwrap();
        editor.set_speed(clip.clone(), 2.0).unwrap();
        let shown = editor.lanes().unwrap()[0].clips[0].clone();
        assert_eq!((shown.volume, shown.speed), (0.25, 2.0));
        assert_eq!(shown.duration_us, length / 2);
        let super::EditorError::Failed { message } = editor.set_volume(clip, -1.0).unwrap_err();
        assert!(message.contains("volume"), "{message}");
        assert_eq!(editor.lanes().unwrap()[0].clips[0].volume, 0.25);
    }

    #[test]
    fn the_strip_reorders_draws_a_ruler_and_fills_a_filmstrip() {
        let temp = Temp::new();
        let editor = temp.open();
        let first = editor.append(editor.import(pattern()).unwrap()).unwrap();
        let second = editor.append(editor.import(pattern()).unwrap()).unwrap();
        editor.reorder(first.clone(), 1).unwrap();
        let ids: Vec<String> = editor.lanes().unwrap()[0]
            .clips
            .iter()
            .map(|clip| clip.id.clone())
            .collect();
        assert_eq!(ids, [second, first.clone()]);
        assert!(editor.reorder("nope".to_owned(), 0).is_err());
        assert_eq!(
            editor.clip_bar(first.clone()).unwrap(),
            Some(super::ToolBar::MainClip)
        );
        assert_eq!(editor.clip_bar("nope".to_owned()).unwrap(), None);

        let size = 0.000_05;
        let ticks = editor.ruler(size, 0, 1_000_000).unwrap();
        assert_eq!(ticks[0].label.as_deref(), Some("00:00"));
        assert_eq!(editor.zoom(size, 2.0, 400.0).unwrap(), size * 2.0);

        let end = editor.duration_us().unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let tile = loop {
            let films = editor.film(0, end, size, 40.0).unwrap();
            assert_eq!(films.len(), 2);
            let tile = films[0].tiles[0].clone();
            if tile.ready || std::time::Instant::now() > deadline {
                break tile;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        };
        let frame = editor.thumb(tile.key).unwrap().expect("stored");
        assert_eq!(frame.rgba.len(), (frame.width * frame.height * 4) as usize);
        assert!(editor.thumb("nothing".to_owned()).unwrap().is_none());
    }

    /// Counts what an export hears and stops it after `stop_at` pictures.
    struct Counter {
        heard: AtomicU64,
        stop_at: u64,
    }

    impl super::ExportProgress for Counter {
        fn progress(&self, done: u64, total: u64) -> bool {
            assert!(done <= total);
            self.heard.fetch_add(1, Ordering::Relaxed);
            done < self.stop_at
        }
    }

    #[test]
    fn the_cut_exports_through_the_binding_and_a_stop_leaves_no_file() {
        let temp = Temp::new();
        let editor = temp.open();
        let movie = temp.0.join("movie.mp4");
        let path = movie.display().to_string();
        let counter = || {
            Arc::new(Counter {
                heard: AtomicU64::new(0),
                stop_at: u64::MAX,
            })
        };
        assert!(editor.export(path.clone(), 720, counter()).is_err());
        editor.append(editor.import(pattern()).unwrap()).unwrap();
        assert!(editor.export(path.clone(), 333, counter()).is_err());

        let stopping = Arc::new(Counter {
            heard: AtomicU64::new(0),
            stop_at: 2,
        });
        let super::EditorError::Failed { message } = editor
            .export(path.clone(), 720, stopping.clone())
            .unwrap_err();
        assert!(message.contains("stopped"), "{message}");
        assert_eq!(stopping.heard.load(Ordering::Relaxed), 2);
        assert!(!movie.exists());

        let all = counter();
        let done = editor.export(path.clone(), 720, all.clone()).unwrap();
        assert_eq!(done.path, path);
        assert_eq!(all.heard.load(Ordering::Relaxed), done.frames);
        assert_eq!(done.width.min(done.height), 720);
        assert!(std::fs::metadata(&movie).unwrap().len() > 0);
        assert_eq!(super::export_sides(), [720, 1080, 2160]);
        // The editor still answers afterwards.
        assert_eq!(editor.lanes().unwrap()[0].clips.len(), 1);
    }

    #[test]
    fn text_crosses_the_boundary_with_its_face() {
        let temp = Temp::new();
        let fonts = temp.0.join("fonts");
        std::fs::create_dir_all(&fonts).unwrap();
        std::fs::copy(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../font/fixtures/sample.ttf"),
            fonts.join("Sample.ttf"),
        )
        .unwrap();
        let editor = temp.open();
        assert!(editor.add_text("A".to_owned(), String::new(), 0).is_err());
        editor
            .set_font_folders(vec![fonts.display().to_string()])
            .unwrap();
        let found = editor.fonts().unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, "Sample.ttf");
        assert_eq!(found[0].label, "Sample");

        let id = editor.add_text("A".to_owned(), String::new(), 0).unwrap();
        let text = editor.text(id.clone()).unwrap().unwrap();
        assert_eq!(text.content, "A");
        assert_eq!(text.face, "Sample.ttf");
        editor.set_text(id.clone(), "AA".to_owned()).unwrap();
        assert_eq!(editor.text(id.clone()).unwrap().unwrap().content, "AA");
        assert!(editor.set_text(id.clone(), String::new()).is_err());
        assert!(
            editor
                .set_text_font(id.clone(), "a/b.ttf".to_owned())
                .is_err()
        );
        assert_eq!(
            editor.clip_bar(id.clone()).unwrap(),
            Some(super::ToolBar::TextClip)
        );
        assert!(editor.text("nope".to_owned()).unwrap().is_none());
    }

    #[test]
    fn every_tool_bar_crosses_the_boundary_and_names_its_tools() {
        use super::{ToolBar, ToolKind, bar_in, bar_out, tool_bar};
        for bar in [
            ToolBar::Root,
            ToolBar::MainClip,
            ToolBar::OverlayClip,
            ToolBar::AudioClip,
            ToolBar::TextClip,
            ToolBar::OtherClip,
            ToolBar::Audio,
            ToolBar::Text,
            ToolBar::Effects,
            ToolBar::Overlay,
        ] {
            assert_eq!(bar_out(bar_in(bar)), bar);
            assert!(tool_bar(bar).iter().all(|tool| !tool.label.is_empty()));
        }
        let root = tool_bar(ToolBar::Root);
        assert_eq!(root[0].kind, ToolKind::Edit);
        assert!(
            root.iter()
                .any(|tool| tool.kind == ToolKind::Audio && tool.opens == Some(ToolBar::Audio))
        );
    }

    #[test]
    fn the_control_curves_cross_the_boundary() {
        assert_eq!(super::volume_at(0.5), 1.0);
        assert_eq!(super::volume_position(1.0), 0.5);
        assert_eq!(super::speed_at(0.2), 1.0);
        assert_eq!(super::speed_position(100.0), 1.0);
    }

    #[test]
    fn play_ticks_the_playhead_and_pause_holds_it() {
        let temp = Temp::new();
        let editor = temp.open();
        editor.play().unwrap();
        assert!(
            !editor.is_playing().unwrap(),
            "an empty project does not play"
        );
        let material = editor.import(pattern()).unwrap();
        editor.append(material).unwrap();
        let end = editor.duration_us().unwrap();
        editor.play().unwrap();
        assert!(editor.is_playing().unwrap());
        assert_eq!(editor.tick(40_000).unwrap(), 40_000);
        editor.pause().unwrap();
        assert_eq!(editor.tick(40_000).unwrap(), 40_000);
        editor.seek(end + 1).unwrap();
        editor.play().unwrap();
        assert_eq!(editor.tick(0).unwrap(), 0, "play from the end starts over");
        editor.seek(end - 10).unwrap();
        assert_eq!(editor.tick(20).unwrap(), end);
        assert!(!editor.is_playing().unwrap());
    }

    #[test]
    fn a_picked_still_lands_on_main_draws_and_saves() {
        let temp = Temp::new();
        let editor = temp.open();
        let lanes = editor.lanes().unwrap();
        assert_eq!(lanes.len(), 1);
        assert!(lanes[0].main && lanes[0].clips.is_empty());

        let material = editor.import(pattern()).unwrap();
        let clip = editor.append(material).unwrap();
        let lanes = editor.lanes().unwrap();
        assert_eq!(lanes[0].kind, LaneKind::Video);
        assert_eq!(lanes[0].clips[0].id, clip);
        assert_eq!(lanes[0].clips[0].name, "pattern.png");
        assert_eq!(editor.duration_us().unwrap(), 3_000_000);
        let frame = editor.frame(500_000).unwrap();
        assert_eq!(
            frame.rgba.len(),
            frame.width as usize * frame.height as usize * 4
        );

        let right = editor.split(clip.clone(), 300_000).unwrap();
        assert_eq!(editor.lanes().unwrap()[0].clips.len(), 2);
        editor.undo().unwrap();
        assert_eq!(editor.lanes().unwrap()[0].clips.len(), 1);
        editor.redo().unwrap();
        editor.remove(right).unwrap();
        editor.save().unwrap();
        drop(editor);

        let opened = temp.open();
        let clips = &opened.lanes().unwrap()[0].clips;
        assert_eq!(clips.len(), 1);
        assert_eq!(clips[0].duration_us, 300_000);
    }

    #[test]
    fn a_refusal_comes_back_as_a_message_and_changes_nothing() {
        let temp = Temp::new();
        let editor = temp.open();
        let error = editor.append("missing".to_owned()).unwrap_err();
        assert_eq!(error.to_string(), "the media is not in the library");
        let notes = temp.0.join("notes.txt");
        std::fs::write(&notes, b"not media").unwrap();
        let error = editor.import(notes.display().to_string()).unwrap_err();
        assert!(
            error.to_string().contains("not a media container"),
            "{error}"
        );
        assert!(editor.frame(-1).is_err());
        assert!(editor.undo().is_err());
        assert_eq!(editor.lanes().unwrap()[0].clips.len(), 0);
    }

    #[test]
    fn a_bundle_that_is_not_a_folder_refuses_open() {
        let temp = Temp::new();
        let error = Editor::open(
            temp.root(),
            temp.0.join("no-bundle").display().to_string(),
            temp.project(),
        );
        assert!(error.is_err());
    }

    #[test]
    fn a_created_project_is_listed_and_opens() {
        let temp = Temp::new();
        let root = temp.root();
        assert!(list_projects(root.clone()).is_empty());
        let made = create_project(root.clone()).unwrap();
        assert!(made.ends_with("Project"));
        let listed = list_projects(root.clone());
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "Project");
        assert_eq!(listed[0].path, made);
        assert_eq!(listed[0].duration_us, 0);
        assert!(listed[0].modified_ms > 0);
        assert!(made.starts_with(&format!("{root}/Projects/")));
        let editor = Editor::open(root, temp.bundle(), made).unwrap();
        assert_eq!(editor.duration_us().unwrap(), 0);
        assert!(temp.0.join("Cutlass/Cache/index.json").is_file());
    }

    #[test]
    fn start_lays_out_the_root_and_writes_the_log_there() {
        let temp = Temp::new();
        start(temp.root()).unwrap();
        for folder in [
            "Projects",
            "Cache",
            "Downloads",
            "Proxies",
            "Models",
            "Logs",
        ] {
            assert!(temp.0.join("Cutlass").join(folder).is_dir(), "{folder}");
        }
        assert!(temp.0.join("Cutlass/Logs/cutlass.log").is_file());
        start(temp.root()).unwrap();
    }

    #[test]
    fn a_relative_root_is_refused_and_lists_nothing() {
        assert!(start("Cutlass".to_owned()).is_err());
        assert!(create_project("Cutlass".to_owned()).is_err());
        assert!(list_projects("Cutlass".to_owned()).is_empty());
        assert!(!std::path::Path::new("Cutlass").exists());
    }

    #[test]
    fn a_project_outside_the_root_is_refused_and_not_created() {
        let temp = Temp::new();
        for outside in [
            temp.0.join("elsewhere"),
            temp.0.join("Cutlass/Projects/a/nested"),
            temp.0.join("Cutlass/Projects/../escape"),
        ] {
            let path = outside.display().to_string();
            let Err(error) = Editor::open(temp.root(), temp.bundle(), path) else {
                panic!("{} opened", outside.display());
            };
            assert!(error.to_string().contains("not a project"), "{error}");
            assert!(!outside.exists());
        }
    }

    #[test]
    fn a_root_that_is_a_file_cannot_hold_a_project() {
        let temp = Temp::new();
        let file = temp.0.join("not-a-folder");
        std::fs::write(&file, b"x").unwrap();
        let root = file.display().to_string();
        assert!(list_projects(root.clone()).is_empty());
        assert!(create_project(root).is_err());
    }
}
