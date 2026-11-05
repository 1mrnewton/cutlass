//! The desktop session.
//!
//! A gesture becomes a [`cutlass_session::Request`] and goes through
//! [`Session::apply`]. Play, pause, and seek stay on the shell, so they do
//! not push an undo step. The picture is `eval::frame` on the software
//! device. Video segments open through the shell's reader factory. A repeated
//! playhead reuses the stored picture. A new playhead seeks the open file.
//! This shell does not composite. Play queues the mixed audio into a software
//! buffer. The window opens this host's output. Export asks eval for the
//! frame at the playhead and writes that PNG.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cutlass_decode::{CodecId, CodecParams, DecodeFactory};
use cutlass_editor::{
    Append, AppendLane, ClipMove, ClipboardMiss, CopiedClip, DropGesture, DropOutcome,
    EFFECT_LENGTH, EffectPlace, GestureError, MediaPlace, MoveGesture, MovePlan, Pointer,
    TITLE_LENGTH, TimelineScale, TitlePlace, TransitionMiss, TrimGesture, TrimPlan, Zoom,
    ZoomBounds, append_placement, copy_clips, drop_gesture, effect_of_card, effect_requests,
    effect_row_for, effect_target, fit_pixel_size, free_row, hit_test, last_accuracy_step,
    material_span, media_requests, minimum_pixel_size, move_requests, paste_requests, plan_move,
    title_requests, title_style, transition_of_card, transition_request, trim_gesture,
};
use cutlass_encode::EncodeFactory;
use cutlass_engine::{Thumb, ThumbKey, Thumbnails, Waveforms};
use cutlass_eval::{ReaderFactory, SceneError, scene_cuts};
use cutlass_export::{ProresProfile, SidecarKind};
use cutlass_model::{
    Crop, Draft, FrameRate, MaterialAudio, MaterialId, MaterialRef, MaterialVideo, MetaType,
    Micros, Segment, SegmentId, TimeRange, TimecodeLabel, TrackAttribute, TrackId, TrackType,
    WordColumns, linkage_from_biz, source_time_at, track_holds_beats, visible_beats, visible_marks,
};
use cutlass_playback::{
    AudioHost, FilmCell, FrameCache, PLAY_FRAMES, Program, Transport, film_cells, media_file,
};
use cutlass_renderer_software::SoftwareDevice;
use cutlass_session::{EditError, ImportedMedium, Request, Session, TrimEdge};
use cutlass_settings::{SeekDuringPlayback, Settings};
use cutlass_storage::Home;
use cutlass_store::{Folder, Hold, StoreError};
use cutlass_transcription::{
    FileSource, HeardClip, HeardMiss, ParakeetUnified, PcmSource, PrepareError, RecognizeError,
    Recognizer, caption_request, columns, fit, words_on_timeline,
};

mod assistant;
mod compound;
mod inspector;
pub use assistant::AgentRun;

pub use inspector::{LetterCase, Panel, SILENT_DB, TextLook, gain_of_db, volume_db};

use crate::face::FaceTable;
use crate::tokens;
use cutlass_probe::{self as probe, Probed};

/// Long-side steps the export dialog offers.
pub const SHORT_SIDES: [i32; 6] = [480, 720, 1080, 1440, 2160, 4320];

pub use cutlass_editor::{TRANSITION_OFFERS, TransitionOffer};

/// How long a notice shows, its fade in and out included.
pub const NOTICE_SHOWN: Duration = Duration::from_millis(2_160);

/// How long a notice takes to fade out at the end of [`NOTICE_SHOWN`].
pub const NOTICE_FADE: Duration = Duration::from_millis(160);

/// The shortest time between two automatic writes of the project.
pub const AUTOSAVE_GAP: Duration = Duration::from_millis(500);

pub use cutlass_playback::ClipWave;

/// The color stored on a marker dropped from the timeline.
pub const MARKER_COLOR: &str = "#E8B931";

#[derive(Debug)]
pub enum ShellError {
    NoProject,
    Store(StoreError),
    Edit(EditError),
    Probe(String),
    MissingFolder(PathBuf),
    Overwrite(PathBuf),
    CodecDeclined(String),
    ExportUnavailable,
    Unavailable(String),
    Eval(String),
    Zoom,
    Locked,
    Io(String),
}

impl std::fmt::Display for ShellError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoProject => formatter.write_str("no project is open"),
            Self::Store(StoreError::Held { .. }) => formatter.write_str(
                "this project is open in another Cutlass window, or another app is holding it",
            ),
            Self::Store(error) => write!(formatter, "{error}"),
            Self::Edit(EditError::EmptyUndo) => formatter.write_str("the undo stack is empty"),
            Self::Edit(EditError::EmptyRedo) => formatter.write_str("the redo stack is empty"),
            Self::Edit(EditError::LockedTrack) => formatter.write_str("the track is locked"),
            Self::Edit(EditError::TransitionWindow) => {
                formatter.write_str("a transition cannot be split")
            }
            Self::Edit(EditError::Overlap) => formatter.write_str("the clips overlap"),
            Self::Edit(EditError::DuplicateMark) => {
                formatter.write_str("a marker is already there")
            }
            Self::Edit(EditError::DuplicateBeat) => formatter.write_str("a beat is already there"),
            Self::Edit(EditError::NotAudio) => formatter.write_str("a beat belongs on audio"),
            Self::Edit(EditError::NotVideo) => formatter.write_str("an extract belongs on video"),
            Self::Edit(EditError::NoSound) => formatter.write_str("the clip has no sound"),
            Self::Edit(EditError::AlreadyExtracted) => {
                formatter.write_str("the sound is already on its own clip")
            }
            Self::Edit(EditError::AlreadyLinked) => {
                formatter.write_str("the clip is already linked")
            }
            Self::Edit(EditError::Angle) => {
                formatter.write_str("the rotation is not a finite number of degrees")
            }
            Self::Edit(EditError::Position) => {
                formatter.write_str("the position is not a finite number")
            }
            Self::Edit(EditError::Scale) => {
                formatter.write_str("the scale must be greater than zero")
            }
            Self::Edit(EditError::Fade) => formatter.write_str("a fade cannot be negative"),
            Self::Edit(EditError::Volume) => formatter.write_str("the volume must be zero or more"),
            Self::Edit(EditError::Opacity) => {
                formatter.write_str("the opacity must be from zero to one")
            }
            Self::Edit(EditError::NotText) => formatter.write_str("a text style belongs on text"),
            Self::Edit(EditError::TextSize) => {
                formatter.write_str("the text size must be greater than zero and at most 512")
            }
            Self::Edit(EditError::TextColor) => {
                formatter.write_str("the text color is not a hex color")
            }
            Self::Edit(EditError::TextAlign) => {
                formatter.write_str("text alignment is not left, center, or right")
            }
            Self::Edit(EditError::LetterSpacing) => {
                formatter.write_str("letter spacing must be from -1 to 4")
            }
            Self::Edit(EditError::LineSpacing) => {
                formatter.write_str("line spacing must be greater than -1 and at most 4")
            }
            Self::Edit(EditError::LineWidth) => {
                formatter.write_str("the line width must be from zero through 8192")
            }
            Self::Edit(EditError::TextFont) => formatter.write_str("the face is not a file name"),
            Self::Edit(EditError::TextEmpty) => formatter.write_str("type something to add text"),
            Self::Edit(EditError::UnknownEffect) => {
                formatter.write_str("that effect is not in this version")
            }
            Self::Edit(EditError::Lut) => formatter.write_str("that file is not a 3D .cube table"),
            Self::Edit(EditError::Mask) => formatter.write_str("a mask number is out of its range"),
            Self::Edit(EditError::Compound) => {
                formatter.write_str("that is not a compound clip played at its own speed")
            }
            Self::Edit(EditError::NotEffect) => {
                formatter.write_str("a strength belongs on an effect, filter, or adjustment")
            }
            Self::Edit(EditError::EffectTarget) => {
                formatter.write_str("an effect goes on a picture clip")
            }
            Self::Edit(EditError::EffectStrength) => {
                formatter.write_str("the strength must be from zero to one")
            }
            Self::Edit(EditError::CanvasColor) => {
                formatter.write_str("the canvas color is not a hex color")
            }
            Self::Edit(EditError::CanvasSize) => {
                formatter.write_str("the canvas side must be from 1 to 7680")
            }
            Self::Edit(EditError::CanvasRatio) => {
                formatter.write_str("the canvas ratio is not a project ratio")
            }
            Self::Edit(EditError::Letterbox) => {
                formatter.write_str("a letterbox belongs on a picture")
            }
            Self::Edit(EditError::FrameRate) => {
                formatter.write_str("the frame rate is not a project rate")
            }
            Self::Edit(EditError::ColorSpace) => {
                formatter.write_str("the color space is not a project color space")
            }
            Self::Edit(error) => write!(formatter, "the edit was refused ({error:?})"),
            Self::Probe(message) | Self::Unavailable(message) | Self::Eval(message) => {
                formatter.write_str(message)
            }
            Self::MissingFolder(path) => {
                write!(formatter, "the folder {} is missing", path.display())
            }
            Self::Overwrite(path) => {
                write!(formatter, "replace {} before writing", path.display())
            }
            Self::CodecDeclined(message) => write!(formatter, "the encoder declined: {message}"),
            Self::ExportUnavailable => formatter.write_str("export is not available"),
            Self::Zoom => formatter.write_str("the timeline has no duration to fit"),
            Self::Locked => formatter.write_str("the track is locked"),
            Self::Io(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for ShellError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Picture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Each lane with its clips, in drawing order.
fn build_lanes(draft: &Draft) -> Vec<LaneView> {
    cutlass_editor::lanes_for(draft, true)
        .into_iter()
        .filter_map(|lane| {
            let track = draft.tracks.get(lane.track_index)?;
            let clips = track
                .segments
                .iter()
                .map(|segment| ClipSpan {
                    id: segment.id.clone(),
                    name: clip_name(draft, segment),
                    start: segment.target.start,
                    duration: segment.target.duration,
                    active: segment.visible,
                    speed_curve: speed_curve(draft, segment)
                        .map(|points| points.iter().map(|point| (point.x, point.y)).collect()),
                })
                .collect();
            Some(LaneView {
                track_id: lane.track_id,
                kind: lane.kind,
                main: lane.main,
                name: (!track.is_default_name && !track.name.is_empty())
                    .then(|| track.name.clone()),
                mute: bit(track.attribute, TrackAttribute::MUTE),
                hide: bit(track.attribute, TrackAttribute::HIDE),
                lock: bit(track.attribute, TrackAttribute::LOCK),
                solo: bit(track.attribute, TrackAttribute::SOLO),
                mute_by_solo: bit(track.attribute, TrackAttribute::MUTE_BY_SOLO),
                clips,
            })
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq)]
pub struct LaneView {
    pub track_id: TrackId,
    pub kind: TrackType,
    pub main: bool,
    /// The name someone gave the track, or `None` for its kind's name.
    pub name: Option<String>,
    pub mute: bool,
    pub hide: bool,
    pub lock: bool,
    pub solo: bool,
    pub mute_by_solo: bool,
    pub clips: Vec<ClipSpan>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClipSpan {
    pub id: SegmentId,
    pub name: String,
    pub start: Micros,
    pub duration: Micros,
    /// Drawn and heard; a deactivated clip is dimmed on its row.
    pub active: bool,
    /// The speed curve the clip runs on: `x` the share of the clip, `y`
    /// the speed there.
    pub speed_curve: Option<Vec<(f64, f64)>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Inspection {
    pub segment_id: SegmentId,
    pub start: Micros,
    pub duration: Micros,
    pub speed: f64,
    pub volume: f64,
    pub scale_x: f64,
    pub scale_y: f64,
    pub rotation: f64,
    pub transform_x: f64,
    pub transform_y: f64,
    pub opacity: f64,
    pub text_size: f64,
    pub text_color: String,
    pub text_align: i32,
    pub letter_spacing: f64,
    pub line_spacing: f64,
    pub line_width: f64,
    pub font: String,
    pub letterbox: String,
    /// How the clip mixes with the picture below it.
    pub blend: cutlass_effects::BlendMode,
    /// The clip's adjust sliders.
    pub adjust: cutlass_effects::Adjust,
    /// The clip's curves, wheels and HSL.
    pub grade: cutlass_effects::Grade,
    /// The clip's LUT: its name and strength.
    pub lut: Option<(String, f32)>,
    /// The clip's chroma key.
    pub chroma: Option<cutlass_effects::Chroma>,
    /// The clip's mask.
    pub mask: Option<cutlass_effects::Mask>,
    /// Where the clip's colour work, or the adjustment layer, shows.
    pub color_mask: Option<cutlass_effects::Mask>,
    /// The speed curve in force, when the clip runs on one.
    pub speed_curve: Option<Vec<cutlass_model::SpeedPoint>>,
    /// The pitch follows the speed.
    pub change_pitch: bool,
    /// Fade-in duration, microseconds.
    pub fade_in: i64,
    /// Fade-out duration, microseconds.
    pub fade_out: i64,
    pub material: String,
    pub text: Option<String>,
    pub words: WordColumns,
    pub locked: bool,
}

#[derive(Clone, Debug)]
pub struct ExportRequest {
    /// The main file: a movie (`.mp4`, `.mov`), the sound alone (`.m4a`,
    /// `.wav`), or a GIF when only the GIF is asked for. A GIF or a caption
    /// file asked for beside it takes the same name and folder.
    pub path: PathBuf,
    pub short_side: i32,
    /// PNG writes the frame at the playhead; H.264, HEVC or ProRes a movie.
    pub codec: CodecId,
    pub replace: bool,
    pub span: ExportSpan,
    /// `None` keeps the project's rate.
    pub frame_rate: Option<FrameRate>,
    pub quality: ExportQuality,
    /// Typed bits a second. `None` takes the rate from `quality`.
    pub bitrate: Option<u32>,
    /// Hold the bitrate over each second instead of on average.
    pub constant_rate: bool,
    pub prores: ProresProfile,
    /// The file has pictures. Off with `sound` on writes the sound alone.
    pub picture: bool,
    /// The file carries the timeline's sound.
    pub sound: bool,
    pub audio: SoundFormat,
    /// Also write an animated GIF.
    pub gif: Option<GifExport>,
    /// Also write the captions as a file.
    pub captions: Option<SidecarKind>,
}

impl Default for ExportRequest {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            short_side: 1080,
            codec: CodecId::H264,
            replace: false,
            span: ExportSpan::Whole,
            frame_rate: None,
            quality: ExportQuality::Recommended,
            bitrate: None,
            constant_rate: false,
            prores: ProresProfile::Standard,
            picture: true,
            sound: true,
            audio: SoundFormat::default(),
            gif: None,
            captions: None,
        }
    }
}

/// The sound of an export: AAC (in an MP4, a movie or an M4A) or 16-bit PCM
/// (in a movie or a WAV).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SoundFormat {
    pub codec: CodecId,
    pub sample_rate: u32,
    pub channels: u16,
    /// Bits a second; PCM ignores it.
    pub bitrate: u32,
}

impl Default for SoundFormat {
    fn default() -> Self {
        Self {
            codec: CodecId::Aac,
            sample_rate: 48_000,
            channels: 2,
            bitrate: 192_000,
        }
    }
}

impl SoundFormat {
    pub const SAMPLE_RATES: [u32; 2] = [44_100, 48_000];
    pub const BITRATES: [u32; 3] = [128_000, 192_000, 320_000];

    fn export(self) -> cutlass_export::AudioExport {
        cutlass_export::AudioExport {
            codec: self.codec,
            bitrate: if self.codec == CodecId::Pcm {
                0
            } else {
                self.bitrate
            },
            sample_rate: self.sample_rate,
            channels: self.channels,
            limiter: 0,
        }
    }

    /// The extension of a file of this sound alone.
    #[must_use]
    pub fn extension(self) -> &'static str {
        if self.codec == CodecId::Pcm {
            "wav"
        } else {
            "m4a"
        }
    }
}

/// An animated GIF beside the export.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GifExport {
    /// One of [`cutlass_export::GIF_SHORT_SIDES`].
    pub short_side: u32,
    pub frame_rate: FrameRate,
    /// Plays over and over; otherwise once.
    pub looped: bool,
}

impl Default for GifExport {
    fn default() -> Self {
        Self {
            short_side: 320,
            frame_rate: FrameRate {
                numerator: 15,
                denominator: 1,
            },
            looped: true,
        }
    }
}

/// GIF frame rates the dialog offers.
pub const GIF_RATES: [u32; 3] = [10, 15, 24];

/// The fastest typed bitrate an export takes, in bits a second.
pub const MAX_BITRATE: u32 = 400_000_000;

/// Said when a project opens that the last app holding it did not close.
pub const RECOVERED_NOTICE: &str =
    "Cutlass did not close last time; this is the project as it was last saved";

/// How much of the timeline a movie holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExportSpan {
    #[default]
    Whole,
    /// From the first selected clip's start to the last one's end.
    Selected,
    /// From the in mark, or the start, to the out mark, or the end.
    Marked,
}

/// How many bits a second a movie spends, around the recommended rate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExportQuality {
    Lower,
    #[default]
    Recommended,
    Higher,
}

impl ExportQuality {
    /// The bitrate for an H.264 or HEVC picture `short_side` pixels on its
    /// short side. HEVC gets the same picture from fewer bits.
    #[must_use]
    pub fn bitrate(self, short_side: u32, codec: CodecId) -> u32 {
        let base = cutlass_export::video_bitrate(short_side);
        let base = if codec == CodecId::Hevc {
            base / 10 * 6
        } else {
            base
        };
        match self {
            Self::Lower => base / 2,
            Self::Recommended => base,
            Self::Higher => base.saturating_mul(2),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Chrome {
    pub project_bar: bool,
    pub library: bool,
    pub preview: bool,
    pub inspector: bool,
    pub timeline: bool,
    pub transport: bool,
    pub export: bool,
    pub projects: bool,
    pub close_requested: bool,
}

pub use cutlass_store::ListedProject;

struct ClipDrag {
    id: SegmentId,
    edge: Option<TrimEdge>,
    press_x: f32,
    press_y: f32,
}

struct PendingTrim {
    id: SegmentId,
    edge: TrimEdge,
    time: Micros,
    snapped: bool,
}

/// A clip kept for paste. `offset` is its start relative to the earliest copied clip.
/// A beat the timeline draws. `source` is the stored source time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BeatView {
    pub segment_id: SegmentId,
    pub source: i64,
    pub time: Micros,
}

/// The beat Delete will remove.
#[derive(Clone, Debug, PartialEq, Eq)]
struct BeatPick {
    segment_id: SegmentId,
    source: i64,
}

/// A marker the timeline draws. `stored_start` is the file value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MarkerView {
    pub segment_id: Option<SegmentId>,
    pub stored_start: i64,
    pub time: Micros,
    pub color: String,
    pub title: String,
}

/// The marker Delete will remove.
#[derive(Clone, Debug, PartialEq, Eq)]
struct MarkerPick {
    segment_id: Option<SegmentId>,
    stored_start: i64,
}

/// The open clip menu, in window points.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipMenu {
    pub x: f32,
    pub y: f32,
}

/// A menu opened over a gap: where it shows, and the row and time it is for.
#[derive(Clone, Debug, PartialEq)]
pub struct GapMenu {
    pub x: f32,
    pub y: f32,
    pub track_id: TrackId,
    pub time: Micros,
}

/// The guide Delete will remove. A time is a vertical line. A row is a
/// horizontal line, in thousandths of the track-sheet height.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GuideMark {
    Time(i64),
    Row(i32),
}

/// Where a clip drag is showing, before the draft is written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MovePreview {
    pub id: SegmentId,
    pub name: String,
    pub start: Micros,
    pub duration: Micros,
    pub track_id: TrackId,
}

/// How far ahead playback opens clips and decodes their first pictures.
const PREROLL: Micros = Micros(1_000_000);
pub struct Shell {
    decoders: DecodeFactory,
    encoders: EncodeFactory,
    readers: ReaderFactory,
    transport: Transport,
    preview_cache: RefCell<FrameCache>,
    /// The lanes as last built, with the revision they were built at.
    lanes_cache: RefCell<Option<(u64, Arc<[LaneView]>)>>,
    settings: Settings,
    /// Where the settings are saved, and the generation on disk.
    settings_file: Option<(PathBuf, u64)>,
    /// Files dropped on the window, imported one job after another.
    waiting_imports: std::collections::VecDeque<PathBuf>,
    session: Option<Session>,
    /// The levels above an open compound clip, the project first.
    outer: Vec<compound::Outer>,
    folder: Option<Folder>,
    /// Library folder edits made, and how many the folder on disk holds.
    /// They are not undo steps, so they count apart from the session.
    library_edits: u64,
    library_saved: u64,
    hold: Option<Hold>,
    /// The session revision the project folder holds, and when it was written.
    saved: Option<(u64, Instant)>,
    path: Option<PathBuf>,
    playhead: Micros,
    playing: bool,
    /// The J/K/L rate while playing: 1 is real time with sound; any other
    /// moves the picture alone, backward below zero.
    shuttle: i32,
    /// The level meter's bars between frames.
    meter: cutlass_playback::LevelMeter,
    /// The I and O marks.
    marks: cutlass_editor::Marks,
    pixel_size: f64,
    /// Left edge of the lanes, in microseconds.
    view_origin: i64,
    lane_width: f64,
    /// The cut has been fit to the lanes since the project opened.
    fitted: bool,
    library_width: f32,
    inspector_width: f32,
    timeline_height: f32,
    /// Clips the user has picked. The last one is the inspector clip.
    selection: Vec<SegmentId>,
    clipboard: Vec<CopiedClip>,
    clip_menu: Option<ClipMenu>,
    /// The Close gap menu, open over a gap on a row.
    gap_menu: Option<GapMenu>,
    /// Vertical guides for this session. They are not stored in the draft.
    guides: Vec<Micros>,
    guide_preview: Option<Micros>,
    /// Horizontal guides, in thousandths of the track-sheet height.
    horizontal: Vec<i32>,
    horizontal_preview: Option<i32>,
    selected_mark: Option<GuideMark>,
    guides_shown: bool,
    guides_locked: bool,
    /// Auto snap. A new window starts on. The draft does not store it.
    snapping: bool,
    /// The split tool cuts a clip at the pointer instead of moving it.
    blade: bool,
    /// Clips on a speed curve draw it on their bar.
    speed_curves: bool,
    /// The marker tool drops a mark on the ruler or on a clip.
    marking: bool,
    selected_marker: Option<MarkerPick>,
    /// The beat tool drops a mark on an audio clip.
    beating: bool,
    selected_beat: Option<BeatPick>,
    clip_drag: Option<ClipDrag>,
    pending_trim: Option<PendingTrim>,
    pending_moves: Vec<ClipMove>,
    notice: String,
    /// When the notice was set; it fades out [`NOTICE_SHOWN`] later.
    notice_since: Instant,
    /// The preview canvas shows black under the picture, or nothing.
    preview_black: bool,
    next_id: u64,
    /// The local speech model, loaded on the first transcript that can hear a clip.
    speech: RefCell<Option<ParakeetUnified>>,
    /// Work running off the window thread, and the project it was started in.
    job: Option<RunningJob>,
    /// The movie the last export wrote.
    exported: Option<PathBuf>,
    /// What the last import that landed added, until the window looks.
    imported: Option<ImportedKind>,
    /// An autosave writing on its own thread.
    saving: Option<Saving>,
    /// The pointer is held down, so like edits join one undo step.
    holding: bool,
    /// The kind, clip and revision of the last edit made while holding.
    held_edit: Option<(HeldEdit, u64)>,
    /// Timeline thumbnails, once the window starts them.
    thumbnails: Option<Thumbnails>,
    /// Thumbnails this frame's clips asked for, nearest the view first.
    film_asks: Vec<ThumbKey>,
    /// Timeline waveforms, once the window starts them.
    waveforms: Option<Waveforms>,
    /// Files this frame's clips asked for peaks of.
    wave_asks: Vec<PathBuf>,
    /// Library card waves by material id, with the peaks, length and width
    /// they were spread from.
    card_waves: std::collections::HashMap<String, CardWave>,
    /// Shared so an export job reads the same faces.
    face: Arc<FaceTable>,
    /// The preview canvas on eframe's GPU device, once the window has one.
    gpu: Option<crate::gpu::Preview>,
    /// The canvas shows a clip's earlier picture while the asked one decodes.
    preview_stale: bool,
    pub chrome: Chrome,
}

impl Shell {
    pub fn new(decoders: DecodeFactory, encoders: EncodeFactory, mut settings: Settings) -> Self {
        settings.ensure_default_shortcuts();
        Self {
            decoders,
            encoders,
            readers: crate::readers::readers_with(
                crate::register_codecs_with(&settings.hardware).0,
            ),
            transport: Transport::new(),
            preview_cache: RefCell::new(FrameCache::new()),
            lanes_cache: RefCell::new(None),
            settings,
            settings_file: None,
            waiting_imports: std::collections::VecDeque::new(),
            session: None,
            outer: Vec::new(),
            folder: None,
            library_edits: 0,
            library_saved: 0,
            hold: None,
            path: None,
            playhead: Micros(0),
            playing: false,
            shuttle: 0,
            meter: cutlass_playback::LevelMeter::new(),
            marks: cutlass_editor::Marks::default(),
            pixel_size: opening_pixel_size(FrameRate::FPS_30),
            view_origin: 0,
            lane_width: 800.0,
            fitted: false,
            library_width: tokens::LIBRARY_WIDTH,
            inspector_width: tokens::INSPECTOR_WIDTH,
            timeline_height: tokens::TIMELINE_HEIGHT,
            selection: Vec::new(),
            clipboard: Vec::new(),
            clip_menu: None,
            gap_menu: None,
            guides: Vec::new(),
            guide_preview: None,
            horizontal: Vec::new(),
            horizontal_preview: None,
            selected_mark: None,
            guides_shown: true,
            guides_locked: false,
            snapping: true,
            blade: false,
            speed_curves: false,
            marking: false,
            selected_marker: None,
            beating: false,
            selected_beat: None,
            clip_drag: None,
            pending_trim: None,
            pending_moves: Vec::new(),
            notice: String::new(),
            notice_since: Instant::now(),
            preview_black: true,
            saved: None,
            next_id: 0,
            speech: RefCell::new(None),
            job: None,
            exported: None,
            imported: None,
            saving: None,
            holding: false,
            held_edit: None,
            thumbnails: None,
            film_asks: Vec::new(),
            waveforms: None,
            wave_asks: Vec::new(),
            card_waves: std::collections::HashMap::new(),
            face: Arc::new(crate::face::installed()),
            gpu: None,
            preview_stale: false,
            chrome: Chrome::default(),
        }
    }

    pub fn codecs_ready(&self) -> bool {
        let picture = CodecParams::video(CodecId::Png, 1, 1);
        self.decoders.open(&picture).is_ok() && self.encoders.open(&picture).is_ok()
    }

    /// Decode timeline thumbnails on their own thread. `ready` runs there
    /// after each one is stored.
    pub fn start_thumbnails(&mut self, ready: impl Fn() + Send + 'static) {
        self.thumbnails = Some(Thumbnails::start(self.readers.clone(), ready));
    }

    /// The filmstrip of clip `id`, drawn `clip_width` by `height` pixels:
    /// the tiles that meet `visible`, pixels from the clip's left edge, each
    /// with its picture once stored. A missing picture is asked for when the
    /// frame ends, by [`Self::send_timeline_asks`]. A clip with no picture file
    /// has no filmstrip.
    pub fn film(
        &mut self,
        id: &SegmentId,
        clip_width: f64,
        height: f64,
        visible: std::ops::Range<f64>,
    ) -> Vec<FilmCell> {
        let (Some(thumbnails), Some(session)) = (self.thumbnails.as_ref(), self.session.as_ref())
        else {
            return Vec::new();
        };
        let cells = film_cells(
            session.draft(),
            self.path.as_deref(),
            thumbnails,
            id,
            clip_width,
            height,
            visible,
            self.pixel_size,
        );
        self.film_asks.extend(
            cells
                .iter()
                .filter(|cell| cell.thumb.is_none())
                .map(|cell| cell.key.clone()),
        );
        cells
    }

    /// Send this frame's thumbnail and waveform asks, and pause both
    /// workers while playing.
    pub fn send_timeline_asks(&mut self) {
        let asks = std::mem::take(&mut self.film_asks);
        if let Some(thumbnails) = self.thumbnails.as_ref() {
            thumbnails.set_paused(self.playing);
            thumbnails.want(asks);
        }
        let asks = std::mem::take(&mut self.wave_asks);
        if let Some(waveforms) = self.waveforms.as_ref() {
            waveforms.set_paused(self.playing);
            waveforms.want(asks);
        }
    }

    /// Read timeline waveforms on their own thread. `ready` runs there after
    /// each publish.
    pub fn start_waveforms(&mut self, ready: impl Fn() + Send + 'static) {
        self.waveforms = Some(Waveforms::start(self.readers.clone(), ready));
    }

    /// The waveform of clip `id`, `clip_width` pixels wide: one level, 0 to
    /// 1, per pixel column of `visible`, pixels from the clip's left edge,
    /// at the clip's volume. A file not read yet is asked for when the frame
    /// ends: an audio clip has no waveform until its peaks arrive, a video
    /// clip an empty one. A video file with no sound has none.
    pub fn wave(
        &mut self,
        id: &SegmentId,
        clip_width: f64,
        visible: std::ops::Range<f64>,
    ) -> Option<ClipWave> {
        let waveforms = self.waveforms.as_ref()?;
        let draft = self.session.as_ref().map(Session::draft)?;
        cutlass_playback::clip_wave(
            draft,
            self.path.as_deref(),
            waveforms,
            id,
            self.pixel_size,
            clip_width,
            visible,
            &mut self.wave_asks,
        )
    }

    /// The library poster of material `id`: a thumbnail of its first frame,
    /// with the key it is stored under. Asked for when the frame ends while
    /// missing. Sound has no poster.
    pub fn poster(&mut self, id: &str) -> Option<(ThumbKey, Option<Arc<Thumb>>)> {
        let thumbnails = self.thumbnails.as_ref()?;
        let draft = self.session.as_ref().map(Session::draft)?;
        let Some(MaterialRef::Video(video)) = draft.materials.get(id) else {
            return None;
        };
        let path = self.media_file(Some(&video.path))?;
        let key = ThumbKey {
            path,
            time: Micros(0),
        };
        let thumb = thumbnails.get(&key);
        if thumb.is_none() {
            self.film_asks.push(key.clone());
        }
        Some((key, thumb))
    }

    /// The library waveform of audio material `id`: `columns` levels, 0 to
    /// 1, across the whole file, once any of it is read. Asked for when the
    /// frame ends while missing or still being read.
    /// Spread again only when the peaks, length or width moved.
    pub fn material_wave(&mut self, id: &str, columns: usize) -> Option<Arc<[f32]>> {
        let waveforms = self.waveforms.as_ref()?;
        let draft = self.session.as_ref().map(Session::draft)?;
        let Some(MaterialRef::Audio(audio)) = draft.materials.get(id) else {
            return None;
        };
        let duration = audio.duration.0;
        let path = self.media_file(Some(&audio.path))?;
        let Some(peaks) = waveforms.get(&path) else {
            self.wave_asks.push(path);
            return None;
        };
        if !peaks.complete {
            self.wave_asks.push(path);
        }
        if let Some(kept) = self.card_waves.get(id)
            && Arc::ptr_eq(&kept.peaks, &peaks)
            && kept.duration == duration
            && kept.columns == columns
        {
            return Some(Arc::clone(&kept.levels));
        }
        let levels: Arc<[f32]> = peaks.spread(duration, columns).into();
        self.card_waves.insert(
            id.to_owned(),
            CardWave {
                peaks,
                duration,
                columns,
                levels: Arc::clone(&levels),
            },
        );
        Some(levels)
    }

    /// A stored media path as a file: inside the project when relative.
    fn media_file(&self, stored: Option<&str>) -> Option<PathBuf> {
        media_file(self.path.as_deref(), stored)
    }

    /// Black under the preview canvas, or nothing so it shows clear.
    pub fn preview_black(&self) -> bool {
        self.preview_black
    }

    pub fn toggle_preview_background(&mut self) {
        self.preview_black = !self.preview_black;
    }

    pub fn notice(&self) -> &str {
        &self.notice
    }

    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    /// Save the settings to `path` from now on. `generation` is what the
    /// file held when it was read.
    pub fn keep_settings_file(&mut self, path: PathBuf, generation: u64) {
        self.settings_file = Some((path, generation));
    }

    /// Write the settings, when they have a file. Another window's write
    /// since the read is taken as the base, and ours is written over it.
    fn save_settings(&mut self) {
        let Some((path, generation)) = self.settings_file.clone() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut written = cutlass_settings::save(&path, generation, &self.settings);
        if let Err(cutlass_settings::SettingsError::GenerationMismatch { actual, .. }) = written {
            written = cutlass_settings::save(&path, actual, &self.settings);
        }
        match written {
            Ok(next) => self.settings_file = Some((path, next)),
            Err(error) => cutlass_logger::warn!("settings: {error}"),
        }
    }

    /// Recent projects that are still there, newest first, leaving out those
    /// under `root`, which the launch list already shows.
    pub fn recent_projects(&self, root: &Path) -> Vec<ListedProject> {
        let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
        self.settings
            .recent_projects
            .iter()
            .filter(|path| {
                !path
                    .canonicalize()
                    .unwrap_or_else(|_| path.to_path_buf())
                    .starts_with(&root)
            })
            .filter_map(|path| cutlass_store::listed_project(path))
            .collect()
    }

    pub fn set_seek_during_playback(&mut self, value: SeekDuringPlayback) {
        self.update_settings(|settings| settings.playback.seek_during_playback = value);
    }

    pub fn set_scrub_audio(&mut self, value: bool) {
        self.update_settings(|settings| settings.playback.scrub_audio = value);
    }

    pub fn project_folder(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub fn project_name(&self) -> String {
        self.session
            .as_ref()
            .map(|session| session.draft().name.clone())
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| "Untitled".to_owned())
    }

    /// The one folder everything Cutlass writes lives under.
    ///
    /// On macOS this is `~/Library/Application Support/Cutlass`. Naming it
    /// does not create it.
    #[must_use]
    pub fn home() -> Home {
        cutlass_storage::desktop_home()
    }

    /// Where new projects are created and where the launch list looks.
    ///
    /// The directory is created when the first project is saved.
    #[must_use]
    pub fn projects_directory() -> PathBuf {
        Self::home().projects()
    }

    /// The user's settings file.
    #[must_use]
    pub fn settings_path() -> PathBuf {
        Self::home().settings_file()
    }

    /// Projects under `root`, newest first. A folder without a draft is skipped.
    #[must_use]
    pub fn list_projects(root: &Path) -> Vec<ListedProject> {
        cutlass_store::list_projects(root)
    }

    /// Write a new empty project under `root` and return its folder.
    pub fn create_project(root: &Path) -> Result<PathBuf, ShellError> {
        cutlass_store::create_project(root).map_err(ShellError::Store)
    }

    /// Write the open project, then leave it so the launch list is showing.
    pub fn close_project(&mut self) -> Result<(), ShellError> {
        if self.session.is_some() {
            self.save()?;
        }
        self.session = None;
        self.waiting_imports.clear();
        self.lanes_cache.replace(None);
        self.card_waves.clear();
        self.folder = None;
        self.hold = None;
        self.saved = None;
        self.path = None;
        self.playhead = Micros(0);
        self.playing = false;
        self.marks.clear();
        self.view_origin = 0;
        self.selection.clear();
        self.clipboard.clear();
        self.clip_menu = None;
        self.clear_guides();
        self.blade = false;
        self.marking = false;
        self.selected_marker = None;
        self.beating = false;
        self.selected_beat = None;
        self.cancel_clip_drag();
        self.fitted = false;
        self.notice.clear();
        self.transport.forget_files();
        self.silence();
        Ok(())
    }

    pub fn playing(&self) -> bool {
        self.playing
    }

    pub fn playhead(&self) -> Micros {
        self.playhead
    }

    pub fn pixel_size(&self) -> f64 {
        self.pixel_size
    }

    /// The time at the left edge of the lanes.
    pub fn view_origin(&self) -> Micros {
        Micros(self.view_origin.max(0))
    }

    pub fn library_width(&self) -> f32 {
        self.library_width
    }

    pub fn inspector_width(&self) -> f32 {
        self.inspector_width
    }

    pub fn timeline_height(&self) -> f32 {
        self.timeline_height
    }

    pub fn set_lane_width(&mut self, width: f32) {
        if width.is_finite() && width > 0.0 {
            self.lane_width = f64::from(width);
        }
    }

    /// Remember the lane width. Until a cut has been fit once, an empty
    /// timeline shows [`EMPTY_SPAN`] across a third of the lanes, and the
    /// first clip, or a saved cut, is fit to three quarters.
    pub fn prepare_lane(&mut self, width: f32) {
        self.set_lane_width(width);
        if self.fitted || self.session.is_none() {
            return;
        }
        if self.cut_duration().0 > 0 {
            self.fitted = self.zoom_fit().is_ok();
        } else if let Some(size) = empty_pixel_size(self.lane_width, self.zoom_bounds().rate) {
            self.pixel_size = size;
        }
    }

    pub fn revision(&self) -> Option<u64> {
        self.session.as_ref().map(Session::revision)
    }

    /// The open session, for a host that edits it elsewhere.
    pub fn session(&self) -> Option<&Session> {
        self.session.as_ref()
    }

    /// Take `session` as the open one, keeping its linkage. A headless host
    /// edits a copy and hands it back before it saves.
    pub fn adopt_session(&mut self, session: Session) -> Result<(), ShellError> {
        if self.session.is_none() {
            return Err(ShellError::NoProject);
        }
        self.session = Some(session);
        self.lanes_cache.replace(None);
        Ok(())
    }

    pub fn draft(&self) -> Option<&Draft> {
        self.session.as_ref().map(Session::draft)
    }

    /// Software output and a live device are open. Linux and a missing device
    /// are not. Pictures still advance either way.
    pub fn audio_output_open(&self) -> bool {
        self.transport.gap().is_none()
    }

    /// Replace the software buffer with the output for `host`.
    pub fn open_preview_audio(&mut self, host: AudioHost) {
        self.transport.open(host);
    }

    #[cfg(test)]
    pub(crate) fn queued_preview(&self) -> &[u8] {
        self.transport.output().queued()
    }

    pub fn open_folder(&mut self, path: &Path) -> Result<(), ShellError> {
        if self.unsaved() {
            self.save()?;
        }
        let opened = cutlass_store::open(path).map_err(ShellError::Store)?;
        let hold = cutlass_store::hold(path).map_err(ShellError::Store)?;
        let rate = opened
            .folder
            .draft
            .frame_rate()
            .unwrap_or(FrameRate::FPS_30);
        self.path = Some(path.to_path_buf());
        let has_pairs = draft_has_linkage(&opened.folder.draft);
        let linkage = linkage_from_biz(
            opened.folder.biz_config.as_ref(),
            opened.folder.draft.id.as_str(),
            has_pairs,
        );
        let mut session = Session::open(opened.folder.draft.clone());
        session.set_linkage(linkage);
        self.outer.clear();
        self.saved = Some((session.revision(), Instant::now()));
        self.session = Some(session);
        self.lanes_cache.replace(None);
        self.folder = Some(opened.folder);
        self.library_edits = 0;
        self.library_saved = 0;
        let unclean = hold.unclean();
        self.hold = Some(hold);
        self.waiting_imports.clear();
        let recent = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        self.settings.note_recent(&recent);
        self.save_settings();
        self.playhead = Micros(0);
        self.playing = false;
        self.marks.clear();
        self.view_origin = 0;
        self.selection.clear();
        self.clipboard.clear();
        self.clip_menu = None;
        self.clear_guides();
        self.blade = false;
        self.marking = false;
        self.selected_marker = None;
        self.beating = false;
        self.selected_beat = None;
        self.cancel_clip_drag();
        self.pixel_size = opening_pixel_size(rate);
        self.fitted = false;
        self.notice.clear();
        if unclean {
            self.set_notice(RECOVERED_NOTICE.to_owned());
        }
        self.transport.forget_files();
        self.silence();
        Ok(())
    }

    pub fn save(&mut self) -> Result<(), ShellError> {
        // A write already under way finishes first, so two never overlap.
        self.finish_background_save();
        let (path, folder, revision) = self.save_snapshot()?;
        let library = self.library_edits;
        folder.save(&path).map_err(ShellError::Store)?;
        self.saved = Some((revision, Instant::now()));
        self.library_saved = library;
        Ok(())
    }

    /// Bring the folder up to the open draft, and hand back where to write
    /// it, a copy to write, and the revision that copy holds.
    fn save_snapshot(&mut self) -> Result<(PathBuf, Folder, u64), ShellError> {
        let path = self.path.clone().ok_or(ShellError::NoProject)?;
        if !path.is_dir() {
            return Err(ShellError::MissingFolder(path));
        }
        let session = self.session.as_ref().ok_or(ShellError::NoProject)?;
        let enabled = session.linkage().enabled;
        let revision = session.revision();
        let draft = self.root_draft().ok_or(ShellError::NoProject)?;
        let folder = self.folder.as_mut().ok_or(ShellError::NoProject)?;
        let main = folder.project.main_timeline_id.clone();
        let name = draft.name.clone();
        folder.draft = draft.clone();
        if let Some(timeline) = folder.timelines.get_mut(&main) {
            *timeline = draft;
        }
        if let Some(entry) = folder
            .project
            .timelines
            .iter_mut()
            .find(|entry| entry.id == main)
        {
            entry.name.clone_from(&name);
        }
        for dock in &mut folder.layout.dock_items {
            if let Some(index) = dock.timeline_ids.iter().position(|id| id == &main)
                && let Some(slot) = dock.timeline_names.get_mut(index)
            {
                slot.clone_from(&name);
            }
        }
        folder.set_linkage_enabled(enabled);
        Ok((path, folder.clone(), revision))
    }

    /// Wait for the background write, if one is under way, and record how
    /// it went.
    fn finish_background_save(&mut self) {
        if let Some(saving) = self.saving.take() {
            let written = saving.handle.join();
            if self.path.as_ref() == Some(&saving.project) {
                if matches!(written, Ok(Ok(()))) {
                    self.library_saved = saving.library;
                }
                self.record_background_save(saving.revision, written);
            } else if let Ok(Err(error)) = written {
                cutlass_logger::error!("autosave of {}: {error}", saving.project.display());
            }
        }
    }

    fn record_background_save(
        &mut self,
        revision: u64,
        written: std::thread::Result<Result<(), StoreError>>,
    ) {
        let error = match written {
            Ok(Ok(())) => {
                self.saved = Some((revision, Instant::now()));
                return;
            }
            Ok(Err(error)) => ShellError::Store(error),
            Err(_) => ShellError::Io("the save stopped unexpectedly".to_owned()),
        };
        self.autosave_failed(&error);
    }

    /// The edits stay unsaved, so quitting still writes them; the clock
    /// restarts so a failing disk is tried once per gap.
    fn autosave_failed(&mut self, error: &ShellError) {
        cutlass_logger::error!("autosave: {error}");
        if let Some((saved, _)) = self.saved {
            self.saved = Some((saved, Instant::now()));
        }
        self.set_notice(format!("Could not save: {error}"));
    }

    /// Write the project when an edit has landed since the last write, at
    /// most once per [`AUTOSAVE_GAP`]. The write runs on its own thread, so
    /// the window does not wait for the disk. Returns how long until the
    /// window should look again.
    pub fn autosave(&mut self) -> Option<Duration> {
        const LOOK_AGAIN: Duration = Duration::from_millis(50);
        if let Some(saving) = self.saving.as_ref() {
            if !saving.handle.is_finished() {
                return Some(LOOK_AGAIN);
            }
            self.finish_background_save();
        }
        if !self.unsaved() {
            return None;
        }
        let (_, at) = self.saved?;
        let wait = AUTOSAVE_GAP.saturating_sub(at.elapsed());
        if !wait.is_zero() {
            return Some(wait);
        }
        let (path, folder, revision) = match self.save_snapshot() {
            Ok(snapshot) => snapshot,
            Err(error) => {
                self.autosave_failed(&error);
                return Some(AUTOSAVE_GAP);
            }
        };
        let project = path.clone();
        let spawned = std::thread::Builder::new()
            .name("cutlass autosave".to_owned())
            .spawn(move || folder.save(&path));
        match spawned {
            Ok(handle) => {
                self.saving = Some(Saving {
                    revision,
                    library: self.library_edits,
                    project,
                    handle,
                });
                Some(LOOK_AGAIN)
            }
            Err(error) => {
                self.autosave_failed(&ShellError::Io(error.to_string()));
                Some(AUTOSAVE_GAP)
            }
        }
    }

    /// An edit has landed since the folder was last written.
    pub fn unsaved(&self) -> bool {
        match (self.session.as_ref(), self.saved) {
            (Some(session), Some((saved, _))) => {
                session.revision() != saved || self.library_edits != self.library_saved
            }
            _ => false,
        }
    }

    /// The library's folders, and the cards taken out of it.
    #[must_use]
    pub fn library_store(&self) -> &cutlass_model::VirtualStore {
        static EMPTY: std::sync::LazyLock<cutlass_model::VirtualStore> =
            std::sync::LazyLock::new(cutlass_model::VirtualStore::default);
        self.folder
            .as_ref()
            .and_then(|folder| folder.virtual_store.as_ref())
            .unwrap_or(&EMPTY)
    }

    /// The file cards of `shelf` in `folder`, through the shared rules.
    #[must_use]
    pub fn library_files(
        &self,
        shelf: cutlass_editor::LibraryShelf,
        folder: &str,
        sort: cutlass_editor::LibrarySort,
        query: &str,
    ) -> Vec<cutlass_editor::LibraryFile> {
        let Some(draft) = self.draft() else {
            return Vec::new();
        };
        cutlass_editor::library_files(draft, self.library_store(), shelf, folder, sort, query)
    }

    /// Change the library's folders. The project then has unsaved changes.
    fn edit_library<R>(
        &mut self,
        edit: impl FnOnce(&mut cutlass_model::VirtualStore) -> R,
    ) -> Result<R, ShellError> {
        let folder = self.folder.as_mut().ok_or(ShellError::NoProject)?;
        let store = folder.virtual_store.get_or_insert_with(Default::default);
        let before = store.clone();
        let result = edit(store);
        if *store != before {
            self.library_edits = self.library_edits.wrapping_add(1);
        }
        Ok(result)
    }

    /// A new folder in `parent`, holding `with`. Returns its id.
    pub fn new_library_folder(
        &mut self,
        parent: &str,
        with: &[String],
    ) -> Result<String, ShellError> {
        let mut id = self.fresh("folder");
        while self
            .library_store()
            .folders
            .iter()
            .any(|folder| folder.id == id)
        {
            id = self.fresh("folder");
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| {
                i64::try_from(since.as_secs()).unwrap_or(i64::MAX)
            });
        let with: Vec<&str> = with.iter().map(String::as_str).collect();
        self.edit_library(|store| cutlass_editor::add_folder(store, &id, parent, &with, now))?;
        Ok(id)
    }

    /// Name a library folder. A blank name keeps the old one.
    pub fn rename_library_folder(&mut self, id: &str, name: &str) -> Result<(), ShellError> {
        self.edit_library(|store| cutlass_editor::rename_folder(store, id, name))?;
        Ok(())
    }

    /// Delete a library folder and the folders in it; their files move up.
    /// Returns where they went.
    pub fn delete_library_folder(&mut self, id: &str) -> Result<Option<String>, ShellError> {
        let name = self.library_folder_name(id);
        let parent = self.edit_library(|store| cutlass_editor::delete_folder(store, id))?;
        if parent.is_some() {
            self.set_notice(format!("Deleted {name}"));
        }
        Ok(parent)
    }

    /// Move a card or folder into `parent`. Says where it went.
    pub fn move_library_item(&mut self, child: &str, parent: &str) -> Result<bool, ShellError> {
        let moved = self.edit_library(|store| cutlass_editor::place(store, child, parent))?;
        if moved {
            let name = self.library_item_name(child);
            let place = if parent.is_empty() {
                "All".to_owned()
            } else {
                self.library_folder_name(parent)
            };
            self.set_notice(format!("Moved {name} to {place}"));
        }
        Ok(moved)
    }

    /// Take file cards out of the library. Their clips stay.
    pub fn hide_library_files(&mut self, ids: &[String]) -> Result<(), ShellError> {
        let paths: Vec<(String, String)> = ids
            .iter()
            .filter_map(|id| Some((id.clone(), self.library_path(id)?)))
            .collect();
        let name = match ids {
            [one] => self.library_item_name(one),
            many => format!("{} items", many.len()),
        };
        self.edit_library(|store| {
            for (id, path) in &paths {
                cutlass_editor::hide_file(store, id, path);
            }
        })?;
        if !paths.is_empty() {
            self.set_notice(format!("Deleted {name}"));
        }
        Ok(())
    }

    /// Deactivate the selection, or activate it again when every picked
    /// clip is already off. One undo step. A deactivated clip is neither
    /// drawn nor heard.
    pub fn toggle_active(&mut self) -> Result<(), ShellError> {
        let picked = self.selection();
        if picked.is_empty() {
            return Err(if self.session.is_none() {
                ShellError::NoProject
            } else {
                ShellError::Unavailable("nothing is selected".to_owned())
            });
        }
        let draft = self.session_draft()?;
        let segments: Vec<&Segment> = draft
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .filter(|segment| picked.contains(&segment.id))
            .collect();
        let active = segments.iter().all(|segment| !segment.visible);
        let requests: Vec<Request> = segments
            .iter()
            .map(|segment| Request::SetActive {
                segment_id: segment.id.clone(),
                active,
            })
            .collect();
        self.apply_requests(&requests)
    }

    /// Every picked clip is deactivated.
    #[must_use]
    pub fn selection_inactive(&self) -> bool {
        let picked = self.selection();
        !picked.is_empty()
            && self.draft().is_some_and(|draft| {
                draft
                    .tracks
                    .iter()
                    .flat_map(|track| &track.segments)
                    .filter(|segment| picked.contains(&segment.id))
                    .all(|segment| !segment.visible)
            })
    }

    /// Show the selected clip's file in the system's file manager.
    pub fn reveal_selected_file(&self) -> Result<(), ShellError> {
        let segment = self
            .selected_segment()
            .ok_or_else(|| ShellError::Unavailable("nothing is selected".to_owned()))?;
        self.reveal_library_file(segment.material_id.as_str())
    }

    /// The selected clip names a file on disk.
    #[must_use]
    pub fn selected_has_file(&self) -> bool {
        self.selected_segment()
            .and_then(|segment| self.library_path(segment.material_id.as_str()))
            .is_some_and(|path| !path.is_empty())
    }

    /// Mark in at the selected clip's start.
    pub fn mark_clip_in(&mut self) -> Result<(), ShellError> {
        let start = self.selected_span()?.0;
        self.marks.set_in(start);
        Ok(())
    }

    /// Mark out at the selected clip's end.
    pub fn mark_clip_out(&mut self) -> Result<(), ShellError> {
        let (start, duration) = self.selected_span()?;
        self.marks
            .set_out(Micros(start.0.saturating_add(duration.0)));
        Ok(())
    }

    fn selected_span(&self) -> Result<(Micros, Micros), ShellError> {
        let segment = self
            .selected_segment()
            .ok_or_else(|| ShellError::Unavailable("nothing is selected".to_owned()))?;
        Ok((segment.target.start, segment.target.duration))
    }

    /// Show a card's file in the system's file manager.
    pub fn reveal_library_file(&self, id: &str) -> Result<(), ShellError> {
        let path = self
            .library_path(id)
            .filter(|path| !path.is_empty())
            .ok_or_else(|| ShellError::Io("this item has no file".to_owned()))?;
        let file = media_file(self.path.as_deref(), Some(&path))
            .ok_or_else(|| ShellError::Io(format!("{path} is not in the project")))?;
        reveal(&file)
    }

    fn library_path(&self, id: &str) -> Option<String> {
        let materials = &self.draft()?.materials;
        materials
            .videos
            .iter()
            .find(|video| video.id.as_str() == id)
            .map(|video| video.path.clone())
            .or_else(|| {
                materials
                    .audios
                    .iter()
                    .find(|audio| audio.id.as_str() == id)
                    .map(|audio| audio.path.clone())
            })
    }

    fn library_folder_name(&self, id: &str) -> String {
        self.library_store()
            .folders
            .iter()
            .find(|folder| folder.id == id)
            .map_or_else(|| "Folder".to_owned(), |folder| folder.display_name.clone())
    }

    fn library_item_name(&self, id: &str) -> String {
        if self
            .library_store()
            .folders
            .iter()
            .any(|folder| folder.id == id)
        {
            return self.library_folder_name(id);
        }
        match self.library_path(id) {
            Some(path) if !path.is_empty() => Path::new(&path)
                .file_name()
                .map_or(path.clone(), |name| name.to_string_lossy().into_owned()),
            _ => "Clip".to_owned(),
        }
    }

    pub fn import_file(&mut self, path: &Path) -> Result<(), ShellError> {
        match self.import_file_in(path) {
            Ok(()) => Ok(()),
            Err(error) => {
                cutlass_logger::error!("import {}: {error}", path.display());
                Err(error)
            }
        }
    }

    fn import_file_in(&mut self, path: &Path) -> Result<(), ShellError> {
        let probed = probe::probe(path).map_err(|error| ShellError::Probe(error.to_string()))?;
        let project = self.path.clone().ok_or(ShellError::NoProject)?;
        let (stored, copied) = place_in_project(&project, path, None)?;
        let imported = self.import_medium(probed, stored);
        if imported.is_err()
            && let Some(copied) = copied
        {
            let _ = std::fs::remove_file(copied);
        }
        imported
    }

    /// Play the file at `path` in the selected clip in place of its media,
    /// keeping the clip's edits. Like an import it is read and copied as a
    /// job; the new media joins the library and the clip takes it in one
    /// undo step.
    pub fn start_replace(&mut self, path: &Path) -> Result<(), ShellError> {
        let project = self.path.clone().ok_or(ShellError::NoProject)?;
        let segment_id = self.selected().cloned().ok_or(ShellError::NoProject)?;
        self.refuse_locked(&segment_id)?;
        self.refuse_busy()?;
        let source = path.to_path_buf();
        let name = source
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .ok_or_else(|| ShellError::Io("the file has no name".to_owned()))?;
        self.start_job("Replace", move |progress| {
            progress.stage(&format!("Reading {name}"));
            let placed = probe::probe(&source)
                .map_err(|error| ShellError::Probe(error.to_string()))
                .and_then(|probed| {
                    progress.stage(&format!("Copying {name}"));
                    let (stored, copied) = place_in_project(&project, &source, Some(progress))?;
                    Ok((probed, stored, CopyGuard(copied)))
                });
            Box::new(move |shell: &mut Shell| {
                let (probed, stored, mut copy) = placed?;
                let import = shell.medium_request(probed, stored);
                let material_id = match &import {
                    Request::Import {
                        medium: ImportedMedium::Video(video),
                    } => video.id.clone(),
                    Request::Import {
                        medium: ImportedMedium::Audio(audio),
                    } => audio.id.clone(),
                    _ => return Err(ShellError::NoProject),
                };
                shell.apply_requests(&[
                    import,
                    Request::ReplaceMedia {
                        segment_id,
                        material_id,
                    },
                ])?;
                copy.keep();
                shell.set_notice(format!("Replaced with {name}"));
                Ok(())
            })
        })
    }

    /// Import `path` as a job: read it, copy it into the project with a
    /// progress bar, then add it to the library when the job lands. A
    /// cancelled import leaves no copy behind.
    pub fn start_import(&mut self, path: &Path) -> Result<(), ShellError> {
        let project = self.path.clone().ok_or(ShellError::NoProject)?;
        self.refuse_busy()?;
        let source = path.to_path_buf();
        let name = source
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .ok_or_else(|| ShellError::Io("the file has no name".to_owned()))?;
        self.start_job("Import", move |progress| {
            progress.stage(&format!("Reading {name}"));
            let placed = probe::probe(&source)
                .map_err(|error| ShellError::Probe(error.to_string()))
                .and_then(|probed| {
                    progress.stage(&format!("Copying {name}"));
                    let (stored, copied) = place_in_project(&project, &source, Some(progress))?;
                    Ok((probed, stored, CopyGuard(copied)))
                });
            Box::new(move |shell: &mut Shell| {
                let (probed, stored, mut copy) = placed.inspect_err(|error| {
                    cutlass_logger::error!("import {name}: {error}");
                })?;
                let before = shell.draft().map(|draft| draft.materials.videos.len());
                shell.import_medium(probed, stored)?;
                copy.keep();
                let video = shell.draft().map(|draft| draft.materials.videos.len()) != before;
                shell.imported = Some(if video {
                    ImportedKind::Video
                } else {
                    ImportedKind::Audio
                });
                shell.set_notice(format!("Imported {name}"));
                Ok(())
            })
        })
    }

    /// What the last import that landed added, once: the window opens that
    /// shelf.
    pub fn take_imported(&mut self) -> Option<ImportedKind> {
        self.imported.take()
    }

    /// The lanes to draw, built again only when the revision moves.
    pub fn lanes(&self) -> Arc<[LaneView]> {
        let Some(session) = self.session.as_ref() else {
            return Arc::from([]);
        };
        let revision = session.revision();
        if let Some((built, lanes)) = self.lanes_cache.borrow().as_ref()
            && *built == revision
        {
            return Arc::clone(lanes);
        }
        let built: Arc<[LaneView]> = build_lanes(session.draft()).into();
        self.lanes_cache
            .replace(Some((revision, Arc::clone(&built))));
        built
    }

    pub fn seek_ruler(&mut self, x: f64) -> Result<Micros, ShellError> {
        let x = self.time_zero_x(x);
        let draft = self.session_draft()?;
        let hit =
            hit_test(draft, self.scale(), Pointer { x, y: 0.0 }).map_err(|_| ShellError::Zoom)?;
        let Some(hit) = hit else {
            return Err(ShellError::Unavailable(
                "the ruler click missed the timeline".to_owned(),
            ));
        };
        self.seek(hit.time);
        Ok(hit.time)
    }

    pub fn seek(&mut self, time: Micros) {
        if self.playing && self.settings.playback.seek_during_playback == SeekDuringPlayback::Pause
        {
            self.playing = false;
            self.silence();
        }
        self.playhead = Micros(time.0.max(0));
        if self.playing && self.shuttle == 1 {
            self.start_stream();
        } else if self.settings.playback.scrub_audio {
            self.queue_preview(PLAY_FRAMES);
        }
        self.reveal_playhead();
    }

    pub fn play(&mut self) {
        let end = self.cut_duration().0;
        if end <= 0 {
            self.playing = false;
            return;
        }
        if self.playhead.0 >= end {
            self.playhead = Micros(0);
        }
        self.playing = true;
        self.shuttle = 1;
        self.start_stream();
        self.preroll();
        self.reveal_playhead();
    }

    pub fn pause(&mut self) {
        self.playing = false;
        self.shuttle = 0;
        self.silence();
    }

    /// The shuttle rate: 0 when stopped, 1 for real time, negative backward.
    #[must_use]
    pub fn shuttle_rate(&self) -> i32 {
        if self.playing { self.shuttle } else { 0 }
    }

    /// J, K or L: play backward or forward, faster each press, or stop.
    /// Any rate but real time forward plays the picture without sound.
    pub fn shuttle(&mut self, key: cutlass_editor::Shuttle) {
        let rate = cutlass_editor::shuttle_rate(self.shuttle_rate(), key);
        let end = self.cut_duration().0;
        match rate {
            0 => self.pause(),
            1 => self.play(),
            _ if end <= 0 || (rate < 0 && self.playhead.0 <= 0) => self.pause(),
            _ => {
                if rate > 0 && self.playhead.0 >= end {
                    self.playhead = Micros(0);
                }
                self.silence();
                self.playing = true;
                self.shuttle = rate;
                self.reveal_playhead();
            }
        }
    }

    /// The I and O marks.
    #[must_use]
    pub fn marks(&self) -> cutlass_editor::Marks {
        self.marks
    }

    /// Mark in at the playhead.
    pub fn mark_in(&mut self) {
        self.marks.set_in(self.playhead);
    }

    /// Mark out at the playhead.
    pub fn mark_out(&mut self) {
        self.marks.set_out(self.playhead);
    }

    pub fn clear_marks(&mut self) {
        self.marks.clear();
    }

    /// The span the marks give on the timeline as cut, when they give one.
    #[must_use]
    pub fn marked_span(&self) -> Option<(Micros, Micros)> {
        self.marks.span(self.cut_duration())
    }

    /// Open the clips playback reaches within [`PREROLL`] and start decoding
    /// them, so a cut does not open a file on the frame it lands.
    fn preroll(&self) {
        if !self.playing {
            return;
        }
        if let Some(session) = self.session.as_ref() {
            self.preview_cache.borrow_mut().preroll(
                session,
                &self.readers,
                self.path.as_deref(),
                self.playhead,
                PREROLL,
            );
        }
    }

    /// Stop every sound: the stream and the scrub slice.
    fn silence(&mut self) {
        self.transport.silence();
    }

    /// Stream audio from the playhead. A host with no stream clock queues
    /// one slice per tick instead.
    fn start_stream(&mut self) {
        let end = self.cut_duration();
        let Some(session) = self.session.as_ref() else {
            self.transport.silence();
            return;
        };
        self.transport.start(
            Program {
                draft: session.draft(),
                revision: session.revision(),
                root: self.path.as_deref(),
                readers: &self.readers,
            },
            self.playhead,
            end,
        );
    }

    pub fn toggle_play(&mut self) {
        if self.playing {
            self.pause();
        } else {
            self.play();
        }
    }

    /// Move to another frame. A positive count goes forward.
    pub fn step_frame(&mut self, frames: i64) -> Result<(), ShellError> {
        let rate = self
            .draft()
            .and_then(Draft::frame_rate)
            .unwrap_or(FrameRate::FPS_30);
        let index = rate.frame_index(self.playhead).unwrap_or(0);
        let next = index.saturating_add(frames).max(0);
        let time = rate.frame_start(next).ok_or(ShellError::Zoom)?;
        let end = self.cut_duration().0.max(0);
        self.seek(Micros(time.0.min(end)));
        Ok(())
    }

    pub fn scrub(&mut self, time: Micros) -> Result<Picture, ShellError> {
        self.seek(time);
        self.preview_bytes()
    }

    /// Move the level meter on by `elapsed`. It reads what the device plays
    /// while playing with sound, and falls to rest otherwise.
    pub fn update_meter(&mut self, elapsed: i64) {
        let peak = if self.playing && self.shuttle == 1 {
            self.transport.heard_peak()
        } else {
            None
        };
        self.meter.update(peak, elapsed);
    }

    /// The left and right channels of the level meter.
    #[must_use]
    pub fn meter(&self) -> [cutlass_playback::ChannelReading; 2] {
        self.meter.reading()
    }

    /// The meter is at rest, so it needs no more frames.
    #[must_use]
    pub fn meter_resting(&self) -> bool {
        self.meter.is_resting()
    }

    /// Move the playhead while playing, by `elapsed`. With a stream clock it
    /// is also pulled toward the heard audio, so the picture stays with the
    /// sound without taking on the device's uneven steps.
    pub fn advance(&mut self, elapsed: i64) {
        if !self.playing || elapsed <= 0 {
            return;
        }
        let end = self.cut_duration();
        if self.shuttle != 1 {
            // The shuttle moves the picture alone, and stops at either end.
            let moved = self
                .playhead
                .0
                .saturating_add(elapsed.saturating_mul(i64::from(self.shuttle)));
            self.playhead = Micros(moved.clamp(0, end.0.max(0)));
            if moved <= 0 || moved >= end.0 {
                self.pause();
            }
            self.reveal_playhead();
            return;
        }
        let Some(session) = self.session.as_ref() else {
            self.playing = false;
            self.silence();
            return;
        };
        let ticked = self.transport.tick(
            Program {
                draft: session.draft(),
                revision: session.revision(),
                root: self.path.as_deref(),
                readers: &self.readers,
            },
            self.playhead,
            elapsed,
            end,
            self.settings.playback.loop_playback,
        );
        self.playhead = ticked.playhead;
        self.playing = ticked.playing;
        if end.0 <= 0 {
            return;
        }
        self.preroll();
        self.reveal_playhead();
    }

    #[cfg(test)]
    pub(crate) fn set_readers(&mut self, readers: ReaderFactory) {
        self.transport.forget_mixer();
        self.readers = readers;
        self.preview_cache.borrow_mut().clear();
        self.transport.forget_files();
    }

    fn queue_preview(&mut self, frames: u32) {
        let Some(session) = self.session.as_ref() else {
            self.transport.silence();
            return;
        };
        self.transport.queue(
            Program {
                draft: session.draft(),
                revision: session.revision(),
                root: self.path.as_deref(),
                readers: &self.readers,
            },
            self.playhead,
            frames,
        );
    }

    /// Draw preview on `state`'s device from now on.
    pub fn attach_gpu(&mut self, state: &eframe::egui_wgpu::RenderState) {
        self.gpu = Some(crate::gpu::Preview::new(state));
    }

    /// The preview as a GPU texture and its size, or `None` before
    /// [`Self::attach_gpu`]. A frame already on the canvas is not drawn again.
    pub fn preview_texture(&mut self) -> Result<Option<(egui::TextureId, [u32; 2])>, ShellError> {
        let time = self.sample_time().0;
        let session = self.session.as_ref().ok_or(ShellError::NoProject)?;
        let Some(gpu) = self.gpu.as_mut() else {
            return Ok(None);
        };
        let key = crate::gpu::Drawn {
            draft: session.draft().id.to_string(),
            revision: session.revision(),
            time,
        };
        if let Some(shown) = gpu.shown(&key) {
            return Ok(Some(shown));
        }
        let mut device = SoftwareDevice::new();
        let mut cache = self.preview_cache.borrow_mut();
        cache.set_patient(false);
        let composed = cache
            .layers(
                session,
                &mut device,
                Micros(time),
                &self.readers,
                self.path.as_deref(),
                &*self.face,
            )
            .map_err(|error| ShellError::Eval(error.to_string()))?;
        let stale = cache.take_stale();
        drop(cache);
        let layers = cutlass_render::frame_layers(composed)
            .map_err(|error| ShellError::Eval(error.to_string()))?;
        let shown = gpu
            .show(layers, key)
            .map_err(|error| ShellError::Eval(error.to_string()))?;
        if stale {
            gpu.forget();
        }
        self.preview_stale = stale;
        Ok(Some(shown))
    }

    /// The preview shows a clip's earlier picture while the one asked for
    /// decodes. The window draws again until this is false.
    #[must_use]
    pub fn preview_pending(&self) -> bool {
        self.preview_stale
    }

    pub fn preview_bytes(&self) -> Result<Picture, ShellError> {
        let session = self.session.as_ref().ok_or(ShellError::NoProject)?;
        let time = self.sample_time();
        let mut device = SoftwareDevice::new();
        let root = self.path.clone();
        let mut cache = self.preview_cache.borrow_mut();
        cache.set_patient(true);
        let picture = cache
            .frame(
                session,
                &mut device,
                time,
                &self.readers,
                root.as_deref(),
                &*self.face,
            )
            .map_err(|error| ShellError::Eval(error.to_string()))?;
        Ok(Picture {
            width: picture.width,
            height: picture.height,
            rgba: picture.rgba,
        })
    }

    /// An edit is there to undo.
    pub fn can_undo(&self) -> bool {
        self.session.as_ref().is_some_and(Session::can_undo)
    }

    /// An undone edit is there to redo.
    pub fn can_redo(&self) -> bool {
        self.session.as_ref().is_some_and(Session::can_redo)
    }

    pub fn undo(&mut self) -> Result<(), ShellError> {
        let session = self.session.as_mut().ok_or(ShellError::NoProject)?;
        session.undo().map_err(ShellError::Edit)?;
        Ok(())
    }

    pub fn redo(&mut self) -> Result<(), ShellError> {
        let session = self.session.as_mut().ok_or(ShellError::NoProject)?;
        session.redo().map_err(ShellError::Edit)?;
        Ok(())
    }

    pub fn run_shortcut(&mut self, keys: &str) -> Result<(), ShellError> {
        let action = self.settings.action_for(keys).map(str::to_owned);
        match action.as_deref() {
            Some("undo") => self.undo(),
            Some("redo" | "redo_y") => self.redo(),
            Some("play") => {
                self.toggle_play();
                Ok(())
            }
            Some("frame_back") => self.step_frame(-1),
            Some("frame_forward") => self.step_frame(1),
            Some("delete") => self.delete_target(),
            Some("split") => self.split_playhead(),
            Some("copy") => self.copy_selected(),
            Some("cut") => self.cut_selected(),
            Some("paste") => self.paste(),
            Some("extract_audio") => self.extract_audio(),
            Some("select_all") => self.select_all(),
            Some("delete_left") => self.delete_side(TrimEdge::Start),
            Some("delete_right") => self.delete_side(TrimEdge::End),
            Some("ripple_delete") => self.ripple_delete(),
            Some("deactivate") => self.toggle_active(),
            Some("compound") => self.create_compound(),
            Some("speed_curves") => {
                self.toggle_speed_curves();
                Ok(())
            }
            Some("shuttle_back") => {
                self.shuttle(cutlass_editor::Shuttle::Back);
                Ok(())
            }
            Some("shuttle_stop") => {
                self.shuttle(cutlass_editor::Shuttle::Stop);
                Ok(())
            }
            Some("shuttle_forward") => {
                self.shuttle(cutlass_editor::Shuttle::Forward);
                Ok(())
            }
            Some("mark_in") => {
                self.mark_in();
                Ok(())
            }
            Some("mark_out") => {
                self.mark_out();
                Ok(())
            }
            Some("clear_marks") => {
                self.clear_marks();
                Ok(())
            }
            _ => Err(ShellError::Unavailable(format!("{keys} is not available"))),
        }
    }

    /// Bind `keys` to `action` and save. Another action that held the
    /// keys loses them; its name comes back.
    pub fn set_shortcut(&mut self, action: &str, keys: &str) -> Option<String> {
        let displaced = self.settings.bind(action, keys);
        self.save_settings();
        displaced
    }

    /// Change the settings and save them.
    pub fn update_settings(&mut self, change: impl FnOnce(&mut Settings)) {
        let before = self.settings.clone();
        change(&mut self.settings);
        if self.settings != before {
            self.save_settings();
        }
    }

    pub fn toggle_attribute(&mut self, track_id: TrackId, bit: i32) -> Result<(), ShellError> {
        let on = !self.attribute_on(&track_id, bit);
        self.apply(Request::SetTrackAttribute { track_id, bit, on })
    }

    /// Add an empty track of `kind`. The shared rules place it: a video
    /// overlay above the main track, audio under it, the rest above video.
    pub fn add_track(&mut self, kind: TrackType) -> Result<TrackId, ShellError> {
        let taken = |shell: &Self, id: &TrackId| {
            shell
                .draft()
                .is_some_and(|draft| draft.tracks.iter().any(|track| &track.id == id))
        };
        let mut track_id = TrackId::new(self.fresh("track"));
        while taken(self, &track_id) {
            track_id = TrackId::new(self.fresh("track"));
        }
        self.apply(Request::InsertTrack {
            track_id: track_id.clone(),
            kind,
        })?;
        Ok(track_id)
    }

    /// Take a track and its clips off the timeline, as one undo step.
    pub fn remove_track(&mut self, track_id: &TrackId) -> Result<(), ShellError> {
        self.apply(Request::RemoveTrack {
            track_id: track_id.clone(),
        })
    }

    /// Name a track. A blank name goes back to the name its kind gives it.
    /// The same name again changes nothing.
    pub fn rename_track(&mut self, track_id: &TrackId, name: &str) -> Result<(), ShellError> {
        let current = self
            .draft()
            .and_then(|draft| draft.tracks.iter().find(|track| &track.id == track_id))
            .map(|track| (track.name.clone(), track.is_default_name));
        let name = name.trim();
        if current.is_some_and(|(old, default)| old == name && default == name.is_empty()) {
            return Ok(());
        }
        self.apply(Request::RenameTrack {
            track_id: track_id.clone(),
            name: name.to_owned(),
        })
    }

    pub fn drag_segment(
        &mut self,
        segment_id: SegmentId,
        x: f64,
        y: f64,
    ) -> Result<(), ShellError> {
        let moves = self.plan_move(segment_id, x, y)?;
        self.pending_moves.clear();
        let requests = move_requests(self.session_draft()?, &moves);
        if requests.is_empty() {
            return Ok(());
        }
        for plan in &moves {
            self.refuse_locked(&plan.segment_id)?;
        }
        self.apply_requests(&requests)?;
        self.reveal_playhead();
        Ok(())
    }

    /// Remember which edge a clip drag started on. A body drag clears a trim preview.
    ///
    /// `press` is the pointer where the drag began, in points.
    pub fn begin_clip_drag(&mut self, id: SegmentId, edge: Option<TrimEdge>, press: (f32, f32)) {
        self.pending_trim = None;
        self.pending_moves.clear();
        if edge.is_some() {
            self.select(Some(id.clone()));
        }
        self.clip_drag = Some(ClipDrag {
            id,
            edge,
            press_x: press.0,
            press_y: press.1,
        });
    }

    /// A body drag that stayed on the press is a click, not a move.
    #[must_use]
    pub fn clip_drag_moved(&self, x: f32, y: f32) -> bool {
        let Some(drag) = &self.clip_drag else {
            return false;
        };
        let dx = f64::from(x - drag.press_x);
        let dy = f64::from(y - drag.press_y);
        dx * dx + dy * dy >= 16.0
    }

    /// The edge this clip's current drag is trimming.
    #[must_use]
    pub fn clip_edge(&self, id: &SegmentId) -> Option<TrimEdge> {
        self.clip_drag
            .as_ref()
            .filter(|drag| &drag.id == id)
            .and_then(|drag| drag.edge)
    }

    /// Follow the pointer without writing the draft. The next paint uses this edge.
    pub fn preview_edge(&mut self, segment_id: SegmentId, edge: TrimEdge, x: f64) {
        self.pending_moves.clear();
        match self.plan_trim(segment_id, edge, x) {
            Ok(plan) => {
                self.pending_trim = Some(PendingTrim {
                    id: plan.segment_id,
                    edge: plan.edge,
                    time: plan.time,
                    snapped: plan.snapped,
                });
            }
            Err(_) => self.pending_trim = None,
        }
    }

    /// The preview edge for this clip, as a timeline time.
    #[must_use]
    pub fn preview_span(&self, id: &SegmentId) -> Option<(TrimEdge, Micros)> {
        self.pending_trim
            .as_ref()
            .filter(|pending| &pending.id == id)
            .map(|pending| (pending.edge, pending.time))
    }

    /// Follow a body drag without writing the draft. The next paint uses this span.
    ///
    /// When the dragged clip is one of several selected clips, each selected
    /// clip keeps its gap and follows by the same amount.
    pub fn preview_move(&mut self, segment_id: SegmentId, x: f64, y: f64) {
        self.pending_trim = None;
        self.pending_moves = self.plan_move(segment_id, x, y).unwrap_or_default();
    }

    /// The dragged clip's copy. A group drag still reports that clip here.
    #[must_use]
    pub fn move_preview(&self) -> Option<MovePreview> {
        self.pending_moves
            .first()
            .map(|pending| self.move_preview_of(pending))
    }

    /// Every copy following the pointer, including the dragged clip.
    #[must_use]
    pub fn move_previews(&self) -> Vec<MovePreview> {
        self.pending_moves
            .iter()
            .map(|pending| self.move_preview_of(pending))
            .collect()
    }

    fn move_preview_of(&self, pending: &ClipMove) -> MovePreview {
        let name = self
            .draft()
            .and_then(|draft| {
                locate(draft, &pending.segment_id).map(|(segment, _)| clip_name(draft, segment))
            })
            .unwrap_or_else(|| pending.segment_id.as_str().to_owned());
        MovePreview {
            id: pending.segment_id.clone(),
            name,
            start: pending.start,
            duration: pending.duration,
            track_id: pending.track_id.clone(),
        }
    }

    /// The original clip is one of the clips being dragged.
    #[must_use]
    pub fn moving(&self, id: &SegmentId) -> bool {
        self.pending_moves
            .iter()
            .any(|pending| &pending.segment_id == id)
    }

    /// A snap target the trim or move preview is holding.
    #[must_use]
    pub fn snap_line(&self) -> Option<Micros> {
        if let Some(pending) = &self.pending_trim
            && pending.snapped
        {
            return Some(pending.time);
        }
        self.pending_moves
            .first()
            .filter(|pending| pending.snapped)
            .map(|pending| pending.start)
    }

    pub fn cancel_clip_drag(&mut self) {
        self.clip_drag = None;
        self.pending_trim = None;
        self.pending_moves.clear();
    }

    /// Finish the drag that `begin_clip_drag` started. An edge trims. The body moves.
    pub fn finish_clip_drag(
        &mut self,
        segment_id: SegmentId,
        x: f64,
        y: f64,
    ) -> Result<(), ShellError> {
        let edge = self
            .clip_drag
            .take()
            .filter(|drag| drag.id == segment_id)
            .and_then(|drag| drag.edge);
        self.pending_trim = None;
        self.pending_moves.clear();
        match edge {
            Some(edge) => self.trim_edge(segment_id, edge, x),
            None => self.drag_segment(segment_id, x, y),
        }
    }

    /// Move one edge. `x` is pixels from the left of the lanes.
    ///
    /// The source follows the timeline through the segment speed. The edge
    /// stops at the media and at the next clip. An edge that does not move
    /// does not push an undo step.
    pub fn trim_edge(
        &mut self,
        segment_id: SegmentId,
        edge: TrimEdge,
        x: f64,
    ) -> Result<(), ShellError> {
        let plan = self.plan_trim(segment_id, edge, x)?;
        self.pending_trim = None;
        self.pending_moves.clear();
        let unchanged = self
            .draft()
            .is_some_and(|draft| trim_is_current(draft, &plan));
        if unchanged {
            return Ok(());
        }
        self.apply(plan.request())?;
        self.reveal_playhead();
        Ok(())
    }

    fn plan_trim(
        &self,
        segment_id: SegmentId,
        edge: TrimEdge,
        x: f64,
    ) -> Result<TrimPlan, ShellError> {
        self.refuse_locked(&segment_id)?;
        let x = self.time_zero_x(x);
        let gesture = TrimGesture {
            segment_id,
            edge,
            pointer: Pointer { x, y: 0.0 },
            scale: self.scale(),
            playhead: self.playhead,
            snap_threshold: self.snap_threshold(),
            guides: self.guide_times(),
        };
        let draft = self.session_draft()?;
        trim_gesture(draft, &gesture).map_err(|error| match error {
            GestureError::UnknownSegment => {
                ShellError::Unavailable("the clip is not on the timeline".to_owned())
            }
            GestureError::BadScale | GestureError::EmptyDuration => {
                ShellError::Unavailable("the drag could not be read".to_owned())
            }
        })
    }

    /// The body drag lands on the pointer time, pulled to a snap within twelve
    /// pixels. Selected clips travel with the dragged one, which comes first.
    /// The draft is not written here.
    fn plan_move(
        &self,
        segment_id: SegmentId,
        x: f64,
        y: f64,
    ) -> Result<Vec<ClipMove>, ShellError> {
        self.refuse_locked(&segment_id)?;
        let gesture = MoveGesture {
            segment_id,
            pointer: Pointer {
                x: self.time_zero_x(x),
                y,
            },
            scale: self.scale(),
            playhead: self.playhead,
            snap_threshold: self.snap_threshold(),
            guides: self.guide_times(),
        };
        let draft = self.session_draft()?;
        match plan_move(draft, &gesture, &self.selection) {
            Ok(MovePlan::Moves(moves)) => Ok(moves),
            Ok(MovePlan::Miss) => Err(ShellError::Unavailable(
                "the drag missed the lanes".to_owned(),
            )),
            Ok(MovePlan::NewLane { .. }) => Err(ShellError::Unavailable(
                "a new lane is not available".to_owned(),
            )),
            Err(GestureError::UnknownSegment) => Err(ShellError::Unavailable(
                "the clip is not on the timeline".to_owned(),
            )),
            Err(_) => Err(ShellError::Unavailable(
                "the drag could not be read".to_owned(),
            )),
        }
    }

    /// Place one library material at the timeline point.
    ///
    /// `x` is pixels from time zero and `y` is pixels from the top lane, on
    /// the same scale as a segment drag. The material stays in the library.
    /// A lane of the material's type is added when the pointer lane and the
    /// upper lane of that type are taken. The source range is the whole
    /// material. One undo removes the segment and a lane this drop added.
    /// The first segment on an empty cut fits that cut to three quarters of
    /// the lane. Another segment keeps the scale.
    pub fn drop_medium(&mut self, material_id: &str, x: f64, y: f64) -> Result<(), ShellError> {
        let x = self.time_zero_x(x);
        let (kind, duration, first_clip) = {
            let draft = self.session_draft()?;
            let span = material_span(draft, material_id).ok_or_else(|| {
                ShellError::Unavailable("the media is not in the library".to_owned())
            })?;
            let first_clip = draft.tracks.iter().all(|track| track.segments.is_empty());
            (span.0, span.1, first_clip)
        };
        if duration.0 <= 0 {
            return Err(ShellError::Unavailable(
                "the media has no duration".to_owned(),
            ));
        }
        let outcome = {
            let gesture = DropGesture {
                kind,
                duration,
                pointer: Pointer { x, y },
                scale: self.scale(),
                playhead: self.playhead,
                snap_threshold: self.snap_threshold(),
                guides: self.guide_times(),
            };
            let draft = self.session_draft()?;
            drop_gesture(draft, &gesture)
                .map_err(|_| ShellError::Unavailable("the drag could not be read".to_owned()))?
        };
        let (track_id, start, lane) = match outcome {
            DropOutcome::Miss => {
                return Err(ShellError::Unavailable(
                    "the drag missed the lanes".to_owned(),
                ));
            }
            DropOutcome::Place { track_id, start } => (track_id, start, None),
            DropOutcome::NewLane { kind, start } => {
                let track_id = TrackId::new(self.fresh("track"));
                (track_id, start, Some(kind))
            }
        };
        self.place_medium(material_id, track_id, start, duration, lane, first_clip)
    }

    /// Add a library item at the end of its lane: main for a picture, else the
    /// first open lane of its kind. A new lane opens when there is none.
    /// The playhead goes to the new clip.
    pub fn add_medium(&mut self, material_id: &str) -> Result<(), ShellError> {
        let (track_id, start, lane, duration, first_clip) = {
            let draft = self.session_draft()?;
            let (kind, duration) = material_span(draft, material_id).ok_or_else(|| {
                ShellError::Unavailable("the media is not in the library".to_owned())
            })?;
            let first_clip = draft.tracks.iter().all(|track| track.segments.is_empty());
            let Append { lane, start } = append_placement(draft, kind);
            match lane {
                AppendLane::Track(track_id) => (track_id, start, None, duration, first_clip),
                AppendLane::New(kind) => (
                    TrackId::new(String::new()),
                    start,
                    Some(kind),
                    duration,
                    first_clip,
                ),
            }
        };
        if duration.0 <= 0 {
            return Err(ShellError::Unavailable(
                "the media has no duration".to_owned(),
            ));
        }
        let track_id = if lane.is_some() {
            TrackId::new(self.fresh("track"))
        } else {
            track_id
        };
        self.place_medium(material_id, track_id, start, duration, lane, first_clip)?;
        self.seek(start);
        Ok(())
    }

    /// Place the whole of `material_id` at `start` on `track_id`, opening the
    /// lane first when `lane` names its kind. Linkage follows the drop rules.
    fn place_medium(
        &mut self,
        material_id: &str,
        track_id: TrackId,
        start: Micros,
        duration: Micros,
        lane: Option<TrackType>,
        first_clip: bool,
    ) -> Result<(), ShellError> {
        let place = MediaPlace {
            track_id,
            segment_id: SegmentId::new(self.fresh("seg")),
            material_id: material_id.to_owned(),
            start,
            duration,
            new_lane: lane,
        };
        let requests = media_requests(self.session_draft()?, &place, self.linkage_enabled());
        let segment_id = place.segment_id;
        self.apply_requests(&requests)?;
        self.select(Some(segment_id));
        if first_clip {
            let _ = self.zoom_fit();
        }
        Ok(())
    }

    /// Add the text shelf's `card_id` at the playhead, on the first text row
    /// with room, else on a new one. The new title is picked.
    pub fn add_title(&mut self, card_id: &str) -> Result<(), ShellError> {
        let start = self.playhead;
        let track_id = free_row(self.session_draft()?, TrackType::Text, start, TITLE_LENGTH)
            .unwrap_or_else(|| TrackId::new(self.fresh("track")));
        self.place_title(card_id, track_id, start)
    }

    /// Place the text shelf's `card_id` where it was dropped on the lanes.
    pub fn drop_title(&mut self, card_id: &str, x: f64, y: f64) -> Result<(), ShellError> {
        let x = self.time_zero_x(x);
        let outcome = {
            let gesture = DropGesture {
                kind: TrackType::Text,
                duration: TITLE_LENGTH,
                pointer: Pointer { x, y },
                scale: self.scale(),
                playhead: self.playhead,
                snap_threshold: self.snap_threshold(),
                guides: self.guide_times(),
            };
            drop_gesture(self.session_draft()?, &gesture)
                .map_err(|_| ShellError::Unavailable("the drag could not be read".to_owned()))?
        };
        let (track_id, start) = match outcome {
            DropOutcome::Miss => {
                return Err(ShellError::Unavailable(
                    "the drag missed the lanes".to_owned(),
                ));
            }
            DropOutcome::Place { track_id, start } => (track_id, start),
            DropOutcome::NewLane { start, .. } => (TrackId::new(self.fresh("track")), start),
        };
        self.place_title(card_id, track_id, start)
    }

    fn place_title(
        &mut self,
        card_id: &str,
        track_id: TrackId,
        start: Micros,
    ) -> Result<(), ShellError> {
        let style = title_style(card_id).ok_or_else(|| {
            ShellError::Unavailable("the title is not on the text shelf".to_owned())
        })?;
        let place = TitlePlace {
            track_id,
            segment_id: SegmentId::new(self.fresh("seg")),
            material_id: MaterialId::new(self.fresh("text")),
            start,
            face: self.face.name().to_owned(),
        };
        self.apply_requests(&title_requests(style, &place))?;
        self.select(Some(place.segment_id));
        Ok(())
    }

    /// Add the effects shelf's `card_id` at the playhead, on the first effect
    /// row with room, else on a new one. It runs over every picture below.
    pub fn add_effect(&mut self, card_id: &str) -> Result<(), ShellError> {
        let start = self.playhead;
        let track_id = free_row(
            self.session_draft()?,
            TrackType::Effect,
            start,
            EFFECT_LENGTH,
        )
        .unwrap_or_else(|| TrackId::new(self.fresh("track")));
        self.place_effect(card_id, track_id, start, EFFECT_LENGTH, None)
    }

    /// Place the effects shelf's `card_id` where it was dropped. On a picture
    /// clip it takes the clip's span and runs over that clip alone; anywhere
    /// else on the lanes it goes on an effect row.
    pub fn drop_effect(&mut self, card_id: &str, x: f64, y: f64) -> Result<(), ShellError> {
        let x = self.time_zero_x(x);
        let on_clip = {
            let draft = self.session_draft()?;
            hit_test(draft, self.scale(), Pointer { x, y })
                .map_err(|_| ShellError::Unavailable("the drag could not be read".to_owned()))?
                .and_then(|hit| hit.segment_id)
                .and_then(|clip| {
                    let (start, duration) = effect_target(draft, &clip)?;
                    Some((clip, start, duration))
                })
        };
        if let Some((clip, start, duration)) = on_clip {
            let track_id = effect_row_for(self.session_draft()?, &clip)
                .unwrap_or_else(|| TrackId::new(self.fresh("track")));
            return self.place_effect(card_id, track_id, start, duration, Some(clip));
        }
        let outcome = {
            let gesture = DropGesture {
                kind: TrackType::Effect,
                duration: EFFECT_LENGTH,
                pointer: Pointer { x, y },
                scale: self.scale(),
                playhead: self.playhead,
                snap_threshold: self.snap_threshold(),
                guides: self.guide_times(),
            };
            drop_gesture(self.session_draft()?, &gesture)
                .map_err(|_| ShellError::Unavailable("the drag could not be read".to_owned()))?
        };
        let (track_id, start) = match outcome {
            DropOutcome::Miss => {
                return Err(ShellError::Unavailable(
                    "the drag missed the lanes".to_owned(),
                ));
            }
            DropOutcome::Place { track_id, start } => (track_id, start),
            DropOutcome::NewLane { start, .. } => (TrackId::new(self.fresh("track")), start),
        };
        self.place_effect(card_id, track_id, start, EFFECT_LENGTH, None)
    }

    /// Add an adjustment layer at the playhead, on the first adjust row
    /// with room, else on a new one. It runs over every picture below.
    pub fn add_adjustment(&mut self) -> Result<(), ShellError> {
        let start = self.playhead;
        let track_id = free_row(
            self.session_draft()?,
            TrackType::Adjust,
            start,
            EFFECT_LENGTH,
        )
        .unwrap_or_else(|| TrackId::new(self.fresh("track")));
        self.place_adjustment(track_id, start)
    }

    /// Add the filters shelf's `card_id` at the playhead, on the first
    /// filter row with room, else on a new one.
    pub fn add_filter(&mut self, card_id: &str) -> Result<(), ShellError> {
        let start = self.playhead;
        let track_id = free_row(
            self.session_draft()?,
            TrackType::Filter,
            start,
            EFFECT_LENGTH,
        )
        .unwrap_or_else(|| TrackId::new(self.fresh("track")));
        self.place_filter(card_id, track_id, start)
    }

    /// Place the filters shelf's `card_id` where it was dropped on the lanes.
    pub fn drop_filter(&mut self, card_id: &str, x: f64, y: f64) -> Result<(), ShellError> {
        let (track_id, start) = self.layer_drop(TrackType::Filter, x, y)?;
        self.place_filter(card_id, track_id, start)
    }

    fn place_filter(
        &mut self,
        card_id: &str,
        track_id: TrackId,
        start: Micros,
    ) -> Result<(), ShellError> {
        let look = cutlass_editor::filter_of_card(card_id)
            .ok_or(ShellError::Edit(EditError::UnknownEffect))?;
        let segment_id = SegmentId::new(self.fresh("seg"));
        let material_id = MaterialId::new(self.fresh("filter"));
        self.apply(Request::AddFilter {
            track_id,
            segment_id: segment_id.clone(),
            material_id,
            start,
            duration: EFFECT_LENGTH,
            filter_id: look.effect_id(),
        })?;
        self.select(Some(segment_id));
        Ok(())
    }

    /// Place an adjustment layer where the card was dropped on the lanes.
    pub fn drop_adjustment(&mut self, x: f64, y: f64) -> Result<(), ShellError> {
        let (track_id, start) = self.layer_drop(TrackType::Adjust, x, y)?;
        self.place_adjustment(track_id, start)
    }

    /// The row and time a card of `kind` lands on, dropped at `x`, `y`.
    fn layer_drop(
        &mut self,
        kind: TrackType,
        x: f64,
        y: f64,
    ) -> Result<(TrackId, Micros), ShellError> {
        let x = self.time_zero_x(x);
        let outcome = {
            let gesture = DropGesture {
                kind,
                duration: EFFECT_LENGTH,
                pointer: Pointer { x, y },
                scale: self.scale(),
                playhead: self.playhead,
                snap_threshold: self.snap_threshold(),
                guides: self.guide_times(),
            };
            drop_gesture(self.session_draft()?, &gesture)
                .map_err(|_| ShellError::Unavailable("the drag could not be read".to_owned()))?
        };
        let (track_id, start) = match outcome {
            DropOutcome::Miss => {
                return Err(ShellError::Unavailable(
                    "the drag missed the lanes".to_owned(),
                ));
            }
            DropOutcome::Place { track_id, start } => (track_id, start),
            DropOutcome::NewLane { start, .. } => (TrackId::new(self.fresh("track")), start),
        };
        Ok((track_id, start))
    }

    fn place_adjustment(&mut self, track_id: TrackId, start: Micros) -> Result<(), ShellError> {
        let segment_id = SegmentId::new(self.fresh("seg"));
        let material_id = MaterialId::new(self.fresh("adjustment"));
        self.apply(Request::AddAdjustment {
            track_id,
            segment_id: segment_id.clone(),
            material_id,
            start,
            duration: EFFECT_LENGTH,
        })?;
        self.select(Some(segment_id));
        Ok(())
    }

    fn place_effect(
        &mut self,
        card_id: &str,
        track_id: TrackId,
        start: Micros,
        duration: Micros,
        on: Option<SegmentId>,
    ) -> Result<(), ShellError> {
        let effect_id =
            effect_of_card(card_id).ok_or(ShellError::Edit(EditError::UnknownEffect))?;
        let place = EffectPlace {
            track_id,
            segment_id: SegmentId::new(self.fresh("seg")),
            material_id: MaterialId::new(self.fresh("effect")),
            start,
            duration,
            on,
        };
        let requests = effect_requests(self.session_draft()?, &effect_id, &place);
        self.apply_requests(&requests)?;
        self.select(Some(place.segment_id));
        Ok(())
    }

    /// Store a blend on the main-track cut nearest the pointer.
    ///
    /// The two clips must touch. The blend does not open a lane and does not
    /// move either clip. A preset longer than the clips is shortened to fit.
    pub fn drop_transition(&mut self, card_id: &str, x: f64, y: f64) -> Result<(), ShellError> {
        let offer = transition_of_card(card_id).ok_or_else(|| {
            ShellError::Unavailable("the transition is not in the library".to_owned())
        })?;
        let pointer = Pointer {
            x: self.time_zero_x(x),
            y,
        };
        let material_id = MaterialId::new(self.fresh("transition"));
        let request = transition_request(
            self.session_draft()?,
            offer,
            pointer,
            self.scale(),
            snap_window(self.pixel_size),
            material_id,
        )
        .map_err(|miss| match miss {
            TransitionMiss::Locked => ShellError::Locked,
            TransitionMiss::BadScale => {
                ShellError::Unavailable("the drag could not be read".to_owned())
            }
            TransitionMiss::OffLanes => {
                ShellError::Unavailable("the drag missed the lanes".to_owned())
            }
            TransitionMiss::NotMain => {
                ShellError::Unavailable("a transition sits on the main track".to_owned())
            }
            TransitionMiss::NoCut => ShellError::Unavailable("the drop missed the cut".to_owned()),
            TransitionMiss::TooShort => {
                ShellError::Unavailable("the clips are too short".to_owned())
            }
        })?;
        self.apply(request)
    }

    pub fn zoom_in(&mut self) -> Result<(), ShellError> {
        self.pixel_size = self
            .stored_zoom()?
            .zoom_in(self.zoom_bounds())
            .ok_or(ShellError::Zoom)?
            .pixel_size();
        self.reveal_playhead();
        Ok(())
    }

    pub fn zoom_out(&mut self) -> Result<(), ShellError> {
        self.pixel_size = self
            .stored_zoom()?
            .zoom_out(self.zoom_bounds())
            .ok_or(ShellError::Zoom)?
            .pixel_size();
        self.reveal_playhead();
        Ok(())
    }

    /// Scale the lanes by a pinch. The timeline time under `x_in_view` stays put.
    ///
    /// `x_in_view` is pixels from the left of the lanes. A factor of one does
    /// nothing. The scale still stops at the two-frame step and the quarter lane.
    pub fn zoom_at(&mut self, factor: f64, x_in_view: f64) -> Result<(), ShellError> {
        if !factor.is_finite() || factor <= 0.0 || !x_in_view.is_finite() {
            return Err(ShellError::Zoom);
        }
        if (factor - 1.0).abs() < 1.0e-4 {
            return Ok(());
        }
        let old = self.pixel_size;
        self.pixel_size = self
            .stored_zoom()?
            .scale(factor, self.zoom_bounds())
            .ok_or(ShellError::Zoom)?
            .pixel_size();
        self.hold_under_pointer(x_in_view, old);
        Ok(())
    }

    pub fn zoom_fit(&mut self) -> Result<(), ShellError> {
        let bounds = self.zoom_bounds();
        if bounds.duration.0 <= 0 || fit_pixel_size(bounds.lane_width, bounds.duration).is_none() {
            return Err(ShellError::Zoom);
        }
        self.pixel_size = self
            .stored_zoom()?
            .zoom_to_fit(bounds)
            .ok_or(ShellError::Zoom)?
            .pixel_size();
        self.view_origin = 0;
        self.reveal_playhead();
        Ok(())
    }

    pub fn minimum_pixel_size(&self) -> Option<f64> {
        minimum_pixel_size(self.zoom_bounds())
    }

    pub fn select(&mut self, id: Option<SegmentId>) {
        self.selected_mark = None;
        self.selected_marker = None;
        self.selected_beat = None;
        self.selection.clear();
        if let Some(id) = id {
            self.selection.push(id);
        }
    }

    /// Add `id` to the selection, or drop it when it is already there.
    pub fn toggle_selected(&mut self, id: SegmentId) {
        self.selected_mark = None;
        self.selected_marker = None;
        self.selected_beat = None;
        if let Some(index) = self.selection.iter().position(|item| item == &id) {
            self.selection.remove(index);
        } else {
            self.selection.push(id);
        }
    }

    /// Replace the selection with `ids`. `keep` adds them to the clips already selected.
    pub fn select_box(&mut self, ids: impl IntoIterator<Item = SegmentId>, keep: bool) {
        self.selected_mark = None;
        self.selected_marker = None;
        self.selected_beat = None;
        if !keep {
            self.selection.clear();
        }
        for id in ids {
            if !self.selection.iter().any(|item| item == &id) {
                self.selection.push(id);
            }
        }
    }

    /// Select every clip on the timeline.
    pub fn select_all(&mut self) -> Result<(), ShellError> {
        let ids: Vec<SegmentId> = self
            .session_draft()?
            .tracks
            .iter()
            .flat_map(|track| track.segments.iter().map(|segment| segment.id.clone()))
            .collect();
        self.select_box(ids, false);
        Ok(())
    }

    pub fn selected(&self) -> Option<&SegmentId> {
        self.selection.last()
    }

    pub fn selection(&self) -> &[SegmentId] {
        &self.selection
    }

    #[must_use]
    pub fn is_selected(&self, id: &SegmentId) -> bool {
        self.selection.iter().any(|item| item == id)
    }

    /// Remove the selected clips, or the selected guide when no clip is selected.
    pub fn delete_target(&mut self) -> Result<(), ShellError> {
        if self.selection.is_empty() {
            if self.selected_mark.is_some() {
                return self.remove_selected_guide();
            }
            if self.selected_marker.is_some() {
                return self.remove_selected_marker();
            }
            if self.selected_beat.is_some() {
                return self.remove_selected_beat();
            }
            return Ok(());
        }
        self.remove_selected()
    }

    /// Guides currently drawn. A hidden list is empty here and is not a snap target.
    #[must_use]
    pub fn guides(&self) -> &[Micros] {
        if self.guides_shown { &self.guides } else { &[] }
    }

    #[must_use]
    pub fn guide_preview(&self) -> Option<Micros> {
        self.guide_preview
    }

    #[must_use]
    pub fn guides_shown(&self) -> bool {
        self.guides_shown
    }

    #[must_use]
    pub fn guides_locked(&self) -> bool {
        self.guides_locked
    }

    /// Auto snap. A new window starts on. The choice lasts until the window closes.
    #[must_use]
    pub fn snap_enabled(&self) -> bool {
        self.snapping
    }

    /// Turn alignment on or off. This does not move a clip and is not an undo step.
    pub fn toggle_snap(&mut self) {
        self.snapping = !self.snapping;
    }

    fn snap_threshold(&self) -> Micros {
        if self.snapping {
            snap_window(self.pixel_size)
        } else {
            Micros(-1)
        }
    }

    pub fn toggle_guides(&mut self) {
        self.guides_shown = !self.guides_shown;
    }

    pub fn toggle_guide_lock(&mut self) {
        self.guides_locked = !self.guides_locked;
    }

    /// The linkage switch. A new project starts on. A folder with no stored
    /// switch and no pairs starts off.
    #[must_use]
    pub fn linkage_enabled(&self) -> bool {
        self.session
            .as_ref()
            .is_some_and(|session| session.linkage().enabled)
    }

    /// The main-track magnet. A new project starts on, and the choice is saved.
    #[must_use]
    pub fn magnet_enabled(&self) -> bool {
        self.session
            .as_ref()
            .is_some_and(|session| session.draft().config.maintrack_adsorb)
    }

    /// Turn packing on or off. Turning it on closes gaps on the main lane.
    ///
    /// A locked main lane refuses the switch. One undo restores the clips and the switch.
    pub fn toggle_magnet(&mut self) -> Result<(), ShellError> {
        let request = cutlass_editor::magnet_request(self.session_draft()?)
            .ok_or_else(|| ShellError::Unavailable("the main track is locked".to_owned()))?;
        self.apply(request)
    }

    /// Store the color under every layer. An empty string clears it.
    /// One undo restores the previous color. The canvas size stays.
    pub fn set_canvas_background(&mut self, color: &str) -> Result<(), ShellError> {
        self.apply(Request::SetCanvasBackground {
            color: color.to_owned(),
        })
    }

    /// The label for the project frame rate. An unreadable rate stays at 30.
    pub fn frame_rate_label(&self) -> String {
        let rate = self
            .draft()
            .and_then(Draft::frame_rate)
            .unwrap_or(FrameRate::FPS_30);
        cutlass_editor::rate_label(rate)
    }

    /// Drop frame is a timecode label for 29.97 and 59.94.
    pub fn drop_frame_available(&self) -> bool {
        self.draft()
            .and_then(Draft::frame_rate)
            .is_some_and(cutlass_editor::is_drop_rate)
    }

    /// Whether timecode labels use the drop-frame separator.
    pub fn drop_frame_enabled(&self) -> bool {
        self.draft()
            .is_some_and(|draft| draft.is_drop_frame_timecode)
    }

    /// Step to the next project rate. 29.97 and 59.94 turn drop frame on.
    /// Every other rate turns it off. One undo restores the rate and the label.
    pub fn cycle_frame_rate(&mut self) -> Result<(), ShellError> {
        let current = self.session_draft()?.frame_rate();
        let next = cutlass_editor::next_rate(current);
        let drop_frame = cutlass_editor::is_drop_rate(next);
        self.apply(Request::SetFrameRate {
            fps: cutlass_editor::rate_fps(next),
            drop_frame,
        })
    }

    /// Step to the next canvas shape and store its frame size.
    ///
    /// 1920×1080 is the 16:9 frame, so a new project steps to 9:16.
    /// One undo restores the size and the ratio. The background stays.
    pub fn cycle_canvas(&mut self) -> Result<(), ShellError> {
        let (width, height) = {
            let canvas = &self.session_draft()?.canvas_config;
            (canvas.width, canvas.height)
        };
        let next = cutlass_editor::next_canvas(width, height);
        self.apply(Request::SetCanvasSize {
            width: next.width,
            height: next.height,
            ratio: next.ratio.to_owned(),
        })
    }

    /// The label for the project color space. A stored int outside the five
    /// spaces is shown as that int.
    pub fn color_space_label(&self) -> String {
        let space = self.draft().map(|draft| draft.color_space).unwrap_or(0);
        cutlass_editor::color_space_name(space)
    }

    /// Store the project name. Surrounding whitespace is dropped. An empty
    /// name stays empty, and the window shows Untitled. The same name is not
    /// an undo step. The canvas stays.
    pub fn set_project_name(&mut self, name: &str) -> Result<(), ShellError> {
        let current = self.session_draft()?.name.clone();
        let name = name.trim();
        if current == name {
            return Ok(());
        }
        self.apply(Request::SetProjectName {
            name: name.to_owned(),
        })
    }

    /// Step to the next project color space. One undo restores the previous
    /// space. The preview is not retinted. The canvas and the frame rate stay.
    pub fn cycle_color_space(&mut self) -> Result<(), ShellError> {
        let current = self.session_draft()?.color_space;
        self.apply(Request::SetColorSpace {
            space: cutlass_editor::next_color_space(current),
        })
    }

    /// The settings the project dialog opens with.
    pub fn project_settings(&self) -> Option<cutlass_editor::ProjectSettings> {
        self.draft().map(cutlass_editor::project_settings)
    }

    /// Save the project dialog: what changed is one undo step, and nothing
    /// changed is no step.
    pub fn save_project_settings(
        &mut self,
        settings: &cutlass_editor::ProjectSettings,
    ) -> Result<(), ShellError> {
        let requests = cutlass_editor::project_requests(self.session_draft()?, settings)
            .map_err(ShellError::Edit)?;
        if requests.is_empty() {
            return Ok(());
        }
        self.apply_requests(&requests)
    }

    /// Turn the drop-frame timecode label on or off. The frame rate stays.
    /// One undo restores the label.
    pub fn toggle_drop_frame(&mut self) -> Result<(), ShellError> {
        let on = self.session_draft()?.is_drop_frame_timecode;
        self.apply(Request::SetDropFrame { on: !on })
    }

    /// Turn follow on or off. Stored pairs stay either way. This is not an undo step.
    pub fn toggle_linkage(&mut self) -> Result<(), ShellError> {
        let session = self.session.as_mut().ok_or(ShellError::NoProject)?;
        let mut settings = session.linkage();
        settings.enabled = !settings.enabled;
        session.set_linkage(settings);
        Ok(())
    }

    pub fn preview_guide(&mut self, time: Micros) {
        self.guide_preview = Some(Micros(time.0.max(0)));
    }

    pub fn clear_guide_preview(&mut self) {
        self.guide_preview = None;
    }

    pub fn take_guide_preview(&mut self) -> Option<Micros> {
        self.guide_preview.take()
    }

    /// Leave a vertical guide. A time already present is selected again.
    pub fn add_guide(&mut self, time: Micros) -> Result<(), ShellError> {
        if self.session.is_none() {
            return Err(ShellError::NoProject);
        }
        if self.guides_locked {
            return Err(ShellError::Unavailable("the guides are locked".to_owned()));
        }
        let time = Micros(time.0.max(0));
        if !self.guides.iter().any(|guide| guide.0 == time.0) {
            self.guides.push(time);
            self.guides.sort_by_key(|guide| guide.0);
        }
        self.selected_marker = None;
        self.selected_beat = None;
        self.selected_mark = Some(GuideMark::Time(time.0));
        Ok(())
    }

    pub fn select_guide(&mut self, time: Micros) {
        if self.guides.iter().any(|guide| guide.0 == time.0) {
            self.selected_mark = Some(GuideMark::Time(time.0));
            self.selected_marker = None;
            self.selected_beat = None;
            self.selection.clear();
        }
    }

    #[must_use]
    pub fn selected_guide(&self) -> Option<Micros> {
        match self.selected_mark {
            Some(GuideMark::Time(time)) => Some(Micros(time)),
            _ => None,
        }
    }

    /// Leave a horizontal guide. `row` is thousandths of the track-sheet height.
    /// A row already present is selected again.
    pub fn add_horizontal(&mut self, row: i32) -> Result<(), ShellError> {
        if self.session.is_none() {
            return Err(ShellError::NoProject);
        }
        if self.guides_locked {
            return Err(ShellError::Unavailable("the guides are locked".to_owned()));
        }
        let row = row.clamp(0, 1_000);
        if !self.horizontal.contains(&row) {
            self.horizontal.push(row);
            self.horizontal.sort_unstable();
        }
        self.selected_marker = None;
        self.selected_beat = None;
        self.selected_mark = Some(GuideMark::Row(row));
        Ok(())
    }

    pub fn preview_horizontal(&mut self, row: i32) {
        self.horizontal_preview = Some(row.clamp(0, 1_000));
    }

    pub fn clear_horizontal_preview(&mut self) {
        self.horizontal_preview = None;
    }

    pub fn take_horizontal_preview(&mut self) -> Option<i32> {
        self.horizontal_preview.take()
    }

    #[must_use]
    pub fn horizontal_preview(&self) -> Option<i32> {
        self.horizontal_preview
    }

    /// Horizontal guides currently drawn, in thousandths of the track-sheet height.
    /// A hidden list is empty here.
    #[must_use]
    pub fn horizontal_guides(&self) -> &[i32] {
        if self.guides_shown {
            &self.horizontal
        } else {
            &[]
        }
    }

    pub fn select_horizontal(&mut self, row: i32) {
        if self.horizontal.contains(&row) {
            self.selected_mark = Some(GuideMark::Row(row));
            self.selected_marker = None;
            self.selected_beat = None;
            self.selection.clear();
        }
    }

    #[must_use]
    pub fn selected_horizontal(&self) -> Option<i32> {
        match self.selected_mark {
            Some(GuideMark::Row(row)) => Some(row),
            _ => None,
        }
    }

    pub fn remove_selected_guide(&mut self) -> Result<(), ShellError> {
        if self.guides_locked {
            return Err(ShellError::Unavailable("the guides are locked".to_owned()));
        }
        match self.selected_mark {
            Some(GuideMark::Time(time)) => self.guides.retain(|guide| guide.0 != time),
            Some(GuideMark::Row(row)) => self.horizontal.retain(|guide| *guide != row),
            None => return Ok(()),
        }
        self.selected_mark = None;
        Ok(())
    }

    /// Timeline time for a pixel measured from the left of the lanes.
    #[must_use]
    pub fn lane_time_at(&self, x_in_view: f64) -> Option<Micros> {
        let x = self.time_zero_x(x_in_view);
        if !self.pixel_size.is_finite() || self.pixel_size <= 0.0 || !x.is_finite() {
            return None;
        }
        let micros = (x / self.pixel_size).round();
        if !(0.0..=i64::MAX as f64).contains(&micros) {
            return None;
        }
        Some(Micros(micros as i64))
    }

    fn guide_times(&self) -> Vec<Micros> {
        if self.guides_shown {
            self.guides.clone()
        } else {
            Vec::new()
        }
    }

    fn clear_guides(&mut self) {
        self.guides.clear();
        self.guide_preview = None;
        self.horizontal.clear();
        self.horizontal_preview = None;
        self.selected_mark = None;
        self.guides_shown = true;
        self.guides_locked = false;
    }

    pub fn inspection(&self) -> Option<Inspection> {
        let draft = self.draft()?;
        let id = self.selected()?;
        let (segment, locked) = locate(draft, id)?;
        let (material, text, words) = material_label(draft, &segment.material_id);
        let (text_size, text_color, text_align, letter_spacing, line_spacing, line_width, font) =
            text_style(draft, &segment.material_id);
        // A keyed clip shows its pose where the playhead is.
        let keyed = cutlass_editor::keyed_pose(draft, id, self.playhead).map(|(_, pose)| pose);
        let (scale_x, scale_y) =
            keyed.map_or(picture_scale(segment), |pose| (pose.scale, pose.scale));
        Some(Inspection {
            segment_id: segment.id.clone(),
            start: segment.target.start,
            duration: segment.target.duration,
            speed: segment.speed,
            volume: segment.volume,
            scale_x,
            scale_y,
            rotation: keyed.map_or(segment.clip.rotation, |pose| pose.rotation),
            transform_x: keyed.map_or(segment.clip.transform.x, |pose| pose.x),
            transform_y: keyed.map_or(segment.clip.transform.y, |pose| pose.y),
            opacity: segment.clip.alpha,
            text_size,
            text_color,
            text_align,
            letter_spacing,
            line_spacing,
            line_width,
            font,
            letterbox: letterbox_color(draft, segment),
            blend: cutlass_effects::segment_blend(draft, segment)
                .unwrap_or(cutlass_effects::BlendMode::SourceOver),
            adjust: cutlass_effects::segment_adjust(draft, segment),
            grade: cutlass_effects::segment_grade(draft, segment),
            lut: cutlass_effects::segment_lut(draft, segment).map(|lut| (lut.name, lut.strength)),
            chroma: cutlass_effects::segment_chroma(draft, segment),
            mask: cutlass_effects::segment_mask(draft, segment),
            color_mask: cutlass_effects::segment_color_mask(draft, segment),
            speed_curve: speed_curve(draft, segment),
            change_pitch: segment.is_tone_modify,
            fade_in: fade_durations(draft, segment).0,
            fade_out: fade_durations(draft, segment).1,
            material,
            text,
            words,
            locked,
        })
    }

    pub fn set_volume(&mut self, volume: f64) -> Result<(), ShellError> {
        let segment_id = self.selected().cloned().ok_or(ShellError::NoProject)?;
        self.refuse_locked(&segment_id)?;
        self.apply(Request::SetVolume { segment_id, volume })
    }

    pub fn set_speed(&mut self, speed: f64) -> Result<(), ShellError> {
        let segment_id = self.selected().cloned().ok_or(ShellError::NoProject)?;
        self.refuse_locked(&segment_id)?;
        self.apply(Request::SetConstantSpeed { segment_id, speed })
    }

    /// Run the selected clip on the speed curve `points`, `x` the share of
    /// the source and `y` the speed there. Its length follows the curve.
    pub fn set_speed_curve(
        &mut self,
        points: &[cutlass_model::SpeedPoint],
    ) -> Result<(), ShellError> {
        let segment_id = self.selected().cloned().ok_or(ShellError::NoProject)?;
        self.refuse_locked(&segment_id)?;
        let speed_id = MaterialId::new(self.fresh("speed"));
        self.apply(Request::SetCurveSpeed {
            segment_id,
            speed_id,
            points: points.to_vec(),
        })
    }

    /// Let the selected clip's pitch follow its speed, or keep it.
    pub fn set_change_pitch(&mut self, on: bool) -> Result<(), ShellError> {
        let segment_id = self.selected().cloned().ok_or(ShellError::NoProject)?;
        self.refuse_locked(&segment_id)?;
        self.apply(Request::SetChangePitch { segment_id, on })
    }

    fn selected_video(&self) -> Option<&MaterialVideo> {
        let draft = self.draft()?;
        let id = self.selected()?;
        let (segment, _) = locate(draft, id)?;
        match draft.materials.get(segment.material_id.as_str()) {
            Some(MaterialRef::Video(video)) => Some(video),
            _ => None,
        }
    }

    /// The selected clip is a video, so Freeze can hold one of its frames.
    #[must_use]
    pub fn freeze_enabled(&self) -> bool {
        self.selected_video().is_some()
    }

    /// The selected video is holding one frame.
    #[must_use]
    pub fn frozen(&self) -> bool {
        self.selected_video()
            .is_some_and(|video| video.freeze.is_some())
    }

    /// Hold the frame under the playhead, or the clip's first frame when the
    /// playhead is elsewhere. A second click plays the clip again.
    ///
    /// The hold is stored on the video. Another clip that uses that same media
    /// holds the same frame. One undo restores the previous picture.
    pub fn toggle_freeze(&mut self) -> Result<(), ShellError> {
        let Some(segment_id) = self.selected().cloned() else {
            return Err(if self.session.is_none() {
                ShellError::NoProject
            } else {
                ShellError::Unavailable("nothing is selected".to_owned())
            });
        };
        let freeze = cutlass_eval::freeze_toggle(self.session_draft()?, &segment_id, self.playhead)
            .map_err(|miss| {
                ShellError::Unavailable(
                    match miss {
                        cutlass_eval::FreezeMiss::UnknownSegment => {
                            "the clip is not on the timeline"
                        }
                        cutlass_eval::FreezeMiss::NotVideo => "a freeze belongs on video",
                        cutlass_eval::FreezeMiss::NoFrame => "the clip has no frame to hold",
                    }
                    .to_owned(),
                )
            })?;
        self.refuse_locked(&segment_id)?;
        self.apply(Request::SetFreeze { segment_id, freeze })
    }

    /// The selected clip is a picture or a sound, so Reverse can flip it.
    #[must_use]
    pub fn reverse_enabled(&self) -> bool {
        let Some(draft) = self.draft() else {
            return false;
        };
        let Some(segment) = self.selected_segment() else {
            return false;
        };
        matches!(
            draft.materials.get(segment.material_id.as_str()),
            Some(MaterialRef::Video(_) | MaterialRef::Audio(_))
        )
    }

    /// The selected picture or sound plays backward.
    #[must_use]
    pub fn reversed(&self) -> bool {
        self.reverse_enabled()
            && self
                .selected_segment()
                .is_some_and(|segment| segment.reverse)
    }

    /// Play the selected clip backward through its own file. A second click
    /// plays it forward. The clip stays where it is. One undo restores the direction.
    pub fn toggle_reverse(&mut self) -> Result<(), ShellError> {
        let Some(segment_id) = self.selected().cloned() else {
            return Err(if self.session.is_none() {
                ShellError::NoProject
            } else {
                ShellError::Unavailable("nothing is selected".to_owned())
            });
        };
        let reverse = {
            let draft = self.session_draft()?;
            let (segment, _) = locate(draft, &segment_id).ok_or_else(|| {
                ShellError::Unavailable("the clip is not on the timeline".to_owned())
            })?;
            let reversible = matches!(
                draft.materials.get(segment.material_id.as_str()),
                Some(MaterialRef::Video(_) | MaterialRef::Audio(_))
            );
            if !reversible {
                return Err(ShellError::Unavailable(
                    "reverse belongs on a picture or a sound".to_owned(),
                ));
            }
            !segment.reverse
        };
        self.refuse_locked(&segment_id)?;
        self.apply(Request::SetReverse {
            segment_id,
            reverse,
            path: String::new(),
        })
    }

    /// The selected clip is a picture that can flip left to right.
    #[must_use]
    pub fn mirror_enabled(&self) -> bool {
        self.selection_can_turn()
    }

    /// The selected picture is flipped left to right.
    #[must_use]
    pub fn mirrored(&self) -> bool {
        self.mirror_enabled()
            && self
                .selected_segment()
                .is_some_and(|segment| segment.clip.flip.horizontal)
    }

    /// Flip the selected picture left to right. A second click restores it.
    /// The clip stays where it is. One undo restores the previous direction.
    pub fn toggle_mirror(&mut self) -> Result<(), ShellError> {
        let Some(segment_id) = self.selected().cloned() else {
            return Err(if self.session.is_none() {
                ShellError::NoProject
            } else {
                ShellError::Unavailable("nothing is selected".to_owned())
            });
        };
        let horizontal = {
            let draft = self.session_draft()?;
            let Some((segment, kind)) = placed(draft, &segment_id) else {
                return Err(ShellError::Unavailable(
                    "the clip is not on the timeline".to_owned(),
                ));
            };
            if !cutlass_editor::can_turn(draft, segment, kind) {
                return Err(ShellError::Unavailable(
                    "a mirror belongs on a picture".to_owned(),
                ));
            }
            !segment.clip.flip.horizontal
        };
        self.refuse_locked(&segment_id)?;
        self.apply(Request::SetMirror {
            segment_id,
            horizontal,
        })
    }

    /// The selected picture is flipped top to bottom.
    #[must_use]
    pub fn flipped(&self) -> bool {
        self.mirror_enabled()
            && self
                .selected_segment()
                .is_some_and(|segment| segment.clip.flip.vertical)
    }

    /// Flip the selected picture top to bottom. A second click restores it.
    /// The clip stays where it is. One undo restores the previous direction.
    pub fn toggle_vertical_flip(&mut self) -> Result<(), ShellError> {
        let segment_id = self.picture_segment("a flip belongs on a picture")?;
        let vertical = self
            .selected_segment()
            .map(|segment| !segment.clip.flip.vertical)
            .ok_or_else(|| ShellError::Unavailable("the clip is not on the timeline".to_owned()))?;
        self.refuse_locked(&segment_id)?;
        self.apply(Request::SetVerticalFlip {
            segment_id,
            vertical,
        })
    }

    /// The selected clip is a picture that can turn.
    #[must_use]
    pub fn rotate_enabled(&self) -> bool {
        self.selection_can_turn()
    }

    /// The selected picture is turned away from its stored upright angle.
    #[must_use]
    pub fn rotated(&self) -> bool {
        self.rotate_enabled()
            && self
                .selected_segment()
                .is_some_and(|segment| segment.clip.rotation.rem_euclid(360.0) != 0.0)
    }

    /// Turn the selected picture clockwise by a quarter turn. Four clicks
    /// bring it upright. The clip stays where it is. One undo restores the angle.
    pub fn turn_clockwise(&mut self) -> Result<(), ShellError> {
        let Some(segment_id) = self.selected().cloned() else {
            return Err(if self.session.is_none() {
                ShellError::NoProject
            } else {
                ShellError::Unavailable("nothing is selected".to_owned())
            });
        };
        let degrees = {
            let draft = self.session_draft()?;
            let Some((segment, kind)) = placed(draft, &segment_id) else {
                return Err(ShellError::Unavailable(
                    "the clip is not on the timeline".to_owned(),
                ));
            };
            if !cutlass_editor::can_turn(draft, segment, kind) {
                return Err(ShellError::Unavailable(
                    "a rotate belongs on a picture".to_owned(),
                ));
            }
            clockwise_quarter(segment.clip.rotation)
        };
        self.refuse_locked(&segment_id)?;
        self.apply(Request::SetRotation {
            segment_id,
            degrees,
        })
    }

    /// The selected clip is a video or a composition, so Crop can inset it.
    #[must_use]
    pub fn crop_enabled(&self) -> bool {
        let Some(draft) = self.draft() else {
            return false;
        };
        let Some(segment) = self.selected_segment() else {
            return false;
        };
        stored_crop(draft, segment).is_some()
    }

    /// The selected picture is keeping less than the whole frame.
    #[must_use]
    pub fn cropped(&self) -> bool {
        let Some(draft) = self.draft() else {
            return false;
        };
        let Some(segment) = self.selected_segment() else {
            return false;
        };
        stored_crop(draft, segment).is_some_and(|crop| crop != Crop::default())
    }

    /// Keep the middle of the selected picture. A second click restores the
    /// whole frame. The clip stays where it is. One undo restores the quad.
    ///
    /// The quad is stored on the video, or on the compound. Another clip that
    /// uses that same material keeps the same middle.
    pub fn toggle_crop(&mut self) -> Result<(), ShellError> {
        let Some(segment_id) = self.selected().cloned() else {
            return Err(if self.session.is_none() {
                ShellError::NoProject
            } else {
                ShellError::Unavailable("nothing is selected".to_owned())
            });
        };
        let crop = {
            let draft = self.session_draft()?;
            let Some((segment, _)) = placed(draft, &segment_id) else {
                return Err(ShellError::Unavailable(
                    "the clip is not on the timeline".to_owned(),
                ));
            };
            let Some(current) = stored_crop(draft, segment) else {
                return Err(ShellError::Unavailable(
                    "a crop belongs on a picture".to_owned(),
                ));
            };
            if current == Crop::default() {
                center_crop()
            } else {
                Crop::default()
            }
        };
        self.refuse_locked(&segment_id)?;
        self.apply(Request::SetCrop { segment_id, crop })
    }

    /// Store the color around the fitted picture. An empty string uses the
    /// project color. One undo restores the previous color. The crop stays.
    pub fn set_letterbox(&mut self, color: &str) -> Result<(), ShellError> {
        let segment_id = self.letterbox_segment()?;
        self.refuse_locked(&segment_id)?;
        let canvas_id = MaterialId::new(self.fresh("canvas"));
        self.apply(Request::SetLetterbox {
            segment_id,
            color: color.to_owned(),
            canvas_id,
        })
    }

    fn letterbox_segment(&self) -> Result<SegmentId, ShellError> {
        let Some(segment_id) = self.selected().cloned() else {
            return Err(if self.session.is_none() {
                ShellError::NoProject
            } else {
                ShellError::Unavailable("nothing is selected".to_owned())
            });
        };
        let draft = self.session_draft()?;
        let Some((segment, _)) = placed(draft, &segment_id) else {
            return Err(ShellError::Unavailable(
                "the clip is not on the timeline".to_owned(),
            ));
        };
        if stored_crop(draft, segment).is_none() {
            return Err(ShellError::Unavailable(
                "a letterbox belongs on a picture".to_owned(),
            ));
        }
        Ok(segment_id)
    }

    /// The selected clip is a picture or a sound, so Transcribe can hear it.
    #[must_use]
    pub fn transcribe_enabled(&self) -> bool {
        let Some(draft) = self.draft() else {
            return false;
        };
        let Some(segment) = self.selected_segment() else {
            return false;
        };
        matches!(
            draft.materials.get(segment.material_id.as_str()),
            Some(MaterialRef::Video(_) | MaterialRef::Audio(_))
        )
    }

    /// Hear the selected clip and place the words on a text track, lined up
    /// with that clip. One undo removes the caption. The local speech model
    /// runs as a job; the caption lands when it finishes, unless the clip
    /// changed meanwhile.
    pub fn transcribe(&mut self) -> Result<(), ShellError> {
        let clip = self.selected_speech()?;
        self.refuse_busy()?;
        let loaded = self.speech.borrow_mut().take();
        if loaded.is_none() && !speech_model_dir().join(SPEECH_MODEL).is_file() {
            return Err(ShellError::Unavailable(
                "the speech model is not on this machine".to_owned(),
            ));
        }
        self.start_job("Transcribe", move |progress| {
            let recognizer = match loaded {
                Some(recognizer) => Ok(recognizer),
                None => {
                    progress.stage("Loading the speech model");
                    load_speech()
                }
            };
            let (recognizer, words) = match recognizer {
                Ok(recognizer) => {
                    progress.stage("Listening");
                    let words = heard_words(&recognizer, &mut FileSource::new(&clip.path), &clip);
                    (Some(recognizer), words)
                }
                Err(error) => (None, Err(error)),
            };
            Box::new(move |shell: &mut Shell| {
                if recognizer.is_some() {
                    shell.speech.replace(recognizer);
                }
                let words = words?;
                shell.still_heard(&clip)?;
                shell.write_caption(&clip, words)
            })
        })
    }

    /// The clip a transcript was heard from is still the same length and
    /// speed, so the words still line up with it.
    fn still_heard(&self, heard: &HeardClip) -> Result<(), ShellError> {
        let now = self.heard_clip(&heard.source_id)?;
        if now.range != heard.range
            || now.target.duration != heard.target.duration
            || now.reverse != heard.reverse
            || now.path != heard.path
        {
            return Err(ShellError::Unavailable(
                "the clip changed while it was heard".to_owned(),
            ));
        }
        Ok(())
    }

    /// Refuse a second job while one runs, or is still stopping.
    fn refuse_busy(&self) -> Result<(), ShellError> {
        match &self.job {
            Some(running) => Err(ShellError::Unavailable(format!(
                "{} is still running",
                running.job.view().label
            ))),
            None => Ok(()),
        }
    }

    /// Run `work` on its own thread. Its landing runs at a later
    /// [`Self::poll_job`], in the project it was started in.
    pub(crate) fn start_job(
        &mut self,
        label: &str,
        work: impl FnOnce(&crate::jobs::Progress) -> Landing + Send + 'static,
    ) -> Result<(), ShellError> {
        self.refuse_busy()?;
        let job = crate::jobs::Job::start(label, work)
            .map_err(|_| ShellError::Unavailable(format!("{label} could not start")))?;
        self.job = Some(RunningJob {
            job,
            project: self.path.clone(),
            stopping: false,
        });
        Ok(())
    }

    /// The job running now, for the window to show. A cancelled job says
    /// so until its thread has stopped.
    pub fn job(&self) -> Option<crate::jobs::JobView> {
        self.job.as_ref().map(|running| {
            let mut view = running.job.view();
            if running.stopping {
                "Stopping".clone_into(&mut view.stage);
                view.fraction = None;
                view.left = None;
            }
            view
        })
    }

    /// Ask the running job to stop. Whatever it was making is dropped; the
    /// job stays until its thread ends, so a new one waits for it.
    pub fn cancel_job(&mut self) {
        if let Some(running) = self.job.as_mut() {
            running.job.cancel();
            running.stopping = true;
        }
    }

    /// Land the job if it finished. True while one still runs.
    pub fn poll_job(&mut self) -> bool {
        let Some(running) = self.job.as_ref() else {
            return false;
        };
        let landing = match running.job.poll() {
            crate::jobs::Poll::Running => return true,
            crate::jobs::Poll::Finished(landing) => Some(landing),
            crate::jobs::Poll::Lost => None,
        };
        let label = running.job.view().label;
        let same_project = running.project == self.path;
        let stopping = running.stopping;
        self.job = None;
        let result = match landing {
            _ if stopping => Ok(()),
            Some(_) if !same_project => Err(ShellError::Unavailable(format!(
                "{label} finished after its project closed"
            ))),
            Some(landing) => landing(self),
            None => Err(ShellError::Unavailable(format!(
                "{label} stopped unexpectedly"
            ))),
        };
        if let Err(error) = result {
            self.set_notice(error.to_string());
        }
        if stopping {
            self.waiting_imports.clear();
        } else if let Some(next) = self.waiting_imports.pop_front() {
            if let Err(error) = self.start_import(&next) {
                self.waiting_imports.clear();
                self.set_notice(error.to_string());
            }
            return self.job.is_some();
        }
        false
    }

    /// What files dropped on the window do: a project folder opens, and
    /// files go into the open project one import after another.
    pub fn drop_files(&mut self, paths: &[PathBuf]) -> Result<(), ShellError> {
        let Some(first) = paths.first() else {
            return Ok(());
        };
        if first.is_dir() && cutlass_store::listed_project(first).is_some() {
            return self.open_folder(first);
        }
        if self.path.is_none() {
            return Err(ShellError::Unavailable(
                "open a project to import into first".to_owned(),
            ));
        }
        let files: Vec<PathBuf> = paths
            .iter()
            .filter(|path| path.is_file())
            .cloned()
            .collect();
        if files.is_empty() {
            return Err(ShellError::Unavailable(
                "only files and project folders can be dropped".to_owned(),
            ));
        }
        self.waiting_imports.extend(files);
        if self.job.is_none()
            && let Some(next) = self.waiting_imports.pop_front()
        {
            let started = self.start_import(&next);
            if started.is_err() {
                self.waiting_imports.clear();
            }
            return started;
        }
        Ok(())
    }

    /// Hear `source` with `recognizer` and place the caption. Tests pass a
    /// stand-in so they do not load the speech model.
    #[cfg(test)]
    fn transcribe_with(
        &mut self,
        recognizer: &dyn Recognizer,
        source: &mut dyn PcmSource,
    ) -> Result<(), ShellError> {
        let clip = self.selected_speech()?;
        let words = heard_words(recognizer, source, &clip)?;
        self.write_caption(&clip, words)
    }

    fn selected_speech(&self) -> Result<HeardClip, ShellError> {
        let Some(segment_id) = self.selected().cloned() else {
            return Err(if self.session.is_none() {
                ShellError::NoProject
            } else {
                ShellError::Unavailable("nothing is selected".to_owned())
            });
        };
        self.refuse_locked(&segment_id)?;
        self.heard_clip(&segment_id)
    }

    /// The sound of clip `segment_id`, and how it maps onto the timeline.
    fn heard_clip(&self, segment_id: &SegmentId) -> Result<HeardClip, ShellError> {
        cutlass_transcription::heard_clip(self.session_draft()?, segment_id).map_err(heard_error)
    }

    fn write_caption(&mut self, clip: &HeardClip, words: WordColumns) -> Result<(), ShellError> {
        let mut next_id = self.next_id;
        let mut fresh = |prefix: &str| {
            next_id = next_id.saturating_add(1);
            format!("{prefix}-{next_id}")
        };
        let request =
            caption_request(self.session_draft()?, clip, words, &mut fresh).map_err(heard_error);
        self.next_id = next_id;
        self.apply(request?)
    }

    /// The selected clip is a video, so its sound can be placed on its own clip.
    #[must_use]
    pub fn extract_audio_enabled(&self) -> bool {
        self.selected_video().is_some()
    }

    /// Place the selected video's sound on an audio track, lined up with that
    /// clip. One undo removes the sound. The window checks the file first.
    pub fn extract_audio(&mut self) -> Result<(), ShellError> {
        let segment_id = self.extract_segment()?;
        let path = {
            let draft = self.session_draft()?;
            let (segment, _) = locate(draft, &segment_id).ok_or_else(|| {
                ShellError::Unavailable("the clip is not on the timeline".to_owned())
            })?;
            let Some(MaterialRef::Video(video)) = draft.materials.get(segment.material_id.as_str())
            else {
                return Err(ShellError::Unavailable(
                    "an extract belongs on video".to_owned(),
                ));
            };
            if segment.reverse && !video.reverse_path.is_empty() {
                video.reverse_path.clone()
            } else {
                video.path.clone()
            }
        };
        let file = self.sound_file(&path)?;
        let heard = self.readers.carries_sound(&file).map_err(|_| {
            ShellError::Unavailable("the clip's sound could not be read".to_owned())
        })?;
        if !heard {
            return Err(ShellError::Unavailable("the clip has no sound".to_owned()));
        }
        self.place_extracted(&segment_id)
    }

    /// Place the selected video's sound without opening the file. Tests use this
    /// so a stand-in path does not have to be a movie.
    #[cfg(test)]
    fn extract_sound(&mut self) -> Result<(), ShellError> {
        let segment_id = self.extract_segment()?;
        self.place_extracted(&segment_id)
    }

    fn extract_segment(&self) -> Result<SegmentId, ShellError> {
        let Some(segment_id) = self.selected().cloned() else {
            return Err(if self.session.is_none() {
                ShellError::NoProject
            } else {
                ShellError::Unavailable("nothing is selected".to_owned())
            });
        };
        let draft = self.session_draft()?;
        let Some((segment, _)) = locate(draft, &segment_id) else {
            return Err(ShellError::Unavailable(
                "the clip is not on the timeline".to_owned(),
            ));
        };
        if !matches!(
            draft.materials.get(segment.material_id.as_str()),
            Some(MaterialRef::Video(_))
        ) {
            return Err(ShellError::Unavailable(
                "an extract belongs on video".to_owned(),
            ));
        }
        self.refuse_locked(&segment_id)?;
        Ok(segment_id)
    }

    fn sound_file(&self, stored: &str) -> Result<PathBuf, ShellError> {
        let path = Path::new(stored);
        if stored.is_empty()
            || path
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            return Err(ShellError::Unavailable("the clip has no sound".to_owned()));
        }
        if path.is_absolute() {
            return Ok(path.to_path_buf());
        }
        Ok(match &self.path {
            Some(root) => root.join(path),
            None => path.to_path_buf(),
        })
    }

    fn place_extracted(&mut self, source_id: &SegmentId) -> Result<(), ShellError> {
        let mut next_id = self.next_id;
        let mut fresh = |prefix: &str| {
            next_id = next_id.saturating_add(1);
            format!("{prefix}-{next_id}")
        };
        let request = cutlass_editor::extract_request(self.session_draft()?, source_id, &mut fresh);
        self.next_id = next_id;
        let request = request
            .ok_or_else(|| ShellError::Unavailable("the clip is not on the timeline".to_owned()))?;
        self.apply(request)
    }

    /// The selected clip is a video, so Split scenes can cut it.
    #[must_use]
    pub fn split_scenes_enabled(&self) -> bool {
        self.selected_video().is_some()
    }

    /// Cut the selected video where its picture changes. One undo restores
    /// the clip. The pictures are read as a job; the cuts land when it
    /// finishes, unless the clip changed meanwhile.
    pub fn split_scenes(&mut self) -> Result<(), ShellError> {
        let segment_id = self.scene_segment()?;
        let (path, range) = {
            let draft = self.session_draft()?;
            let (segment, _) = locate(draft, &segment_id).ok_or_else(|| {
                ShellError::Unavailable("the clip is not on the timeline".to_owned())
            })?;
            let Some(MaterialRef::Video(video)) = draft.materials.get(segment.material_id.as_str())
            else {
                return Err(ShellError::Unavailable(
                    "a scene split belongs on video".to_owned(),
                ));
            };
            if video.freeze.is_some() {
                return Err(ShellError::Unavailable(
                    "a held frame has one scene".to_owned(),
                ));
            }
            if video.path.is_empty() {
                return Err(ShellError::Unavailable(
                    "the clip has no picture".to_owned(),
                ));
            }
            let range = segment
                .source_range
                .clone()
                .filter(|range| range.duration.0 > 0 && range.start.0 >= 0)
                .unwrap_or_else(|| TimeRange::new(0, segment.target.duration.0));
            (video.path.clone(), range)
        };
        let file = self.opened_media(&path)?;
        self.refuse_busy()?;
        let readers = self.readers.clone();
        self.start_job("Split scenes", move |progress| {
            progress.stage("Looking for cuts");
            let cuts = scene_cuts(&readers, &file, &range).map_err(scene_message);
            Box::new(move |shell: &mut Shell| {
                let cuts = cuts?;
                // The cuts are in the file's time; the clip must still show
                // the same stretch of it for them to land where they belong.
                let times = {
                    let draft = shell.session_draft()?;
                    let still = locate(draft, &segment_id).is_some_and(|(segment, _)| {
                        segment
                            .source_range
                            .as_ref()
                            .is_none_or(|now| *now == range)
                    });
                    if !still {
                        return Err(ShellError::Unavailable(
                            "the clip changed while its scenes were found".to_owned(),
                        ));
                    }
                    cutlass_eval::scene_split_times(draft, &segment_id, &range, &cuts)
                };
                shell.split_at_times(&segment_id, &times)
            })
        })
    }

    /// Cut the selected video at `times` on the timeline. Tests pass known
    /// times so they do not decode a picture.
    #[cfg(test)]
    fn split_scenes_at(&mut self, times: &[Micros]) -> Result<(), ShellError> {
        let segment_id = self.scene_segment()?;
        self.split_at_times(&segment_id, times)
    }

    fn scene_segment(&self) -> Result<SegmentId, ShellError> {
        let Some(segment_id) = self.selected().cloned() else {
            return Err(if self.session.is_none() {
                ShellError::NoProject
            } else {
                ShellError::Unavailable("nothing is selected".to_owned())
            });
        };
        self.refuse_locked(&segment_id)?;
        let draft = self.session_draft()?;
        let Some((segment, _)) = locate(draft, &segment_id) else {
            return Err(ShellError::Unavailable(
                "the clip is not on the timeline".to_owned(),
            ));
        };
        if !matches!(
            draft.materials.get(segment.material_id.as_str()),
            Some(MaterialRef::Video(_))
        ) {
            return Err(ShellError::Unavailable(
                "a scene split belongs on video".to_owned(),
            ));
        }
        Ok(segment_id)
    }

    fn opened_media(&self, stored: &str) -> Result<PathBuf, ShellError> {
        let path = Path::new(stored);
        if path
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            return Err(ShellError::Unavailable(
                "the clip has no picture".to_owned(),
            ));
        }
        if path.is_absolute() {
            return Ok(path.to_path_buf());
        }
        Ok(match &self.path {
            Some(root) => root.join(path),
            None => path.to_path_buf(),
        })
    }

    fn split_at_times(
        &mut self,
        segment_id: &SegmentId,
        times: &[Micros],
    ) -> Result<(), ShellError> {
        let kept = {
            let draft = self.session_draft()?;
            let (segment, _) = locate(draft, segment_id).ok_or_else(|| {
                ShellError::Unavailable("the clip is not on the timeline".to_owned())
            })?;
            cutlass_eval::kept_cuts(draft, segment, times)
        };
        if kept.is_empty() {
            return Err(ShellError::Unavailable(
                "no scene change was found".to_owned(),
            ));
        }
        let mut requests = Vec::with_capacity(kept.len());
        for time in kept.into_iter().rev() {
            requests.push(Request::Split {
                segment_id: segment_id.clone(),
                at: time,
                right_id: SegmentId::new(self.fresh("seg")),
            });
        }
        self.apply_requests(&requests)?;
        self.select(Some(segment_id.clone()));
        Ok(())
    }

    fn selection_can_turn(&self) -> bool {
        let Some(draft) = self.draft() else {
            return false;
        };
        let Some(id) = self.selected() else {
            return false;
        };
        let Some((segment, kind)) = placed(draft, id) else {
            return false;
        };
        cutlass_editor::can_turn(draft, segment, kind)
    }

    fn selected_segment(&self) -> Option<&Segment> {
        let draft = self.draft()?;
        let id = self.selected()?;
        locate(draft, id).map(|(segment, _)| segment)
    }

    pub fn set_text(&mut self, content: String) -> Result<(), ShellError> {
        let segment_id = self.selected().cloned().ok_or(ShellError::NoProject)?;
        self.refuse_locked(&segment_id)?;
        self.apply(Request::SetTextContent {
            segment_id,
            content,
        })
    }

    pub fn set_range_start(&mut self, start: Micros) -> Result<(), ShellError> {
        let segment_id = self.selected().cloned().ok_or(ShellError::NoProject)?;
        self.refuse_locked(&segment_id)?;
        let track_id = self.track_of(&segment_id)?.clone();
        self.apply(Request::Move {
            segment_id,
            track_id,
            start,
        })
    }

    /// The selected clip is a picture, so its position and scale can change.
    #[must_use]
    pub fn place_enabled(&self) -> bool {
        self.selection_can_turn()
    }

    /// Move the selected picture. `x` and `y` are fractions of half the canvas,
    /// and positive `y` is up. One undo restores the previous position.
    pub fn set_position(&mut self, x: f64, y: f64) -> Result<(), ShellError> {
        let segment_id = self.picture_segment("a placement belongs on a picture")?;
        self.refuse_locked(&segment_id)?;
        let request = self.keyed_request(&segment_id, |pose| (pose.x, pose.y) = (x, y));
        self.apply(request.unwrap_or(Request::SetPosition { segment_id, x, y }))
    }

    /// Put the selected picture or title where an align button says: an
    /// edge on the canvas edge, or its centre on a centre line, as it sits
    /// at the playhead. One undo restores the previous place.
    pub fn align_selected(&mut self, align: cutlass_editor::StageAlign) -> Result<(), ShellError> {
        let segment_id = self.picture_segment("a placement belongs on a picture")?;
        self.refuse_locked(&segment_id)?;
        let draft = self.session_draft()?;
        let sizes = FaceSizes {
            draft,
            fonts: &*self.face,
        };
        let (x, y) = cutlass_editor::stage_align(draft, &segment_id, self.playhead, &sizes, align)
            .ok_or_else(|| ShellError::Unavailable("the clip has no size to align".to_owned()))?;
        let request = self.keyed_request(&segment_id, |pose| (pose.x, pose.y) = (x, y));
        self.apply(request.unwrap_or(Request::SetPosition { segment_id, x, y }))
    }

    /// The selected clip's in and out animations, when it is a picture or
    /// title on the canvas.
    #[must_use]
    pub fn animations(&self) -> Option<cutlass_editor::ClipAnimations> {
        let segment_id = self.selected()?;
        let draft = self.session_draft().ok()?;
        let sizes = FaceSizes {
            draft,
            fonts: &*self.face,
        };
        cutlass_editor::clip_animations(draft, segment_id, &sizes)
    }

    /// Give the selected clip the in, out or loop animation `resource_id`,
    /// `duration` long, or clear that one with `None`. The session keeps the
    /// in and out inside the clip; a loop's duration is one round. One undo
    /// step.
    pub fn set_animation(
        &mut self,
        kind: cutlass_model::AnimType,
        resource_id: Option<String>,
        duration: Micros,
    ) -> Result<(), ShellError> {
        let Some(segment_id) = self.selected().cloned() else {
            return Err(if self.session.is_none() {
                ShellError::NoProject
            } else {
                ShellError::Unavailable("nothing is selected".to_owned())
            });
        };
        if self.animations().is_none() {
            return Err(ShellError::Unavailable(
                "only a picture or title takes animations".to_owned(),
            ));
        }
        self.refuse_locked(&segment_id)?;
        self.apply(Request::SetAnimation {
            segment_id,
            kind,
            animation: resource_id,
            duration,
        })
    }

    /// Scale the selected picture. One undo restores the previous scale.
    pub fn set_scale(&mut self, scale: f64) -> Result<(), ShellError> {
        let segment_id = self.picture_segment("a placement belongs on a picture")?;
        self.refuse_locked(&segment_id)?;
        let request = self.keyed_request(&segment_id, |pose| pose.scale = scale);
        self.apply(request.unwrap_or(Request::SetScale { segment_id, scale }))
    }

    /// On a clip with keys, with the playhead on it, the request that keys
    /// its pose there after `edit`, as a stage drag does. Elsewhere `None`,
    /// and the edit changes the stored pose.
    fn keyed_request(
        &self,
        segment_id: &SegmentId,
        edit: impl FnOnce(&mut cutlass_editor::Pose),
    ) -> Option<Request> {
        let (offset, mut pose) =
            cutlass_editor::keyed_pose(self.draft()?, segment_id, self.playhead)?;
        edit(&mut pose);
        Some(Request::KeyPose {
            segment_id: segment_id.clone(),
            offset,
            x: pose.x,
            y: pose.y,
            scale: pose.scale,
            rotation: pose.rotation,
        })
    }

    /// Set how solid the selected picture is. `1` is solid and `0` is clear.
    /// One undo restores the previous opacity.
    /// Mix the selected picture with what is below it by `mode`. Normal
    /// takes the blend off. One undo step.
    pub fn set_blend(&mut self, mode: cutlass_effects::BlendMode) -> Result<(), ShellError> {
        let segment_id = self.picture_segment("a blend belongs on a picture")?;
        self.refuse_locked(&segment_id)?;
        let material_id = if mode == cutlass_effects::BlendMode::SourceOver {
            MaterialId::new("")
        } else {
            MaterialId::new(self.fresh("mix"))
        };
        self.apply(Request::SetBlend {
            segment_id,
            material_id,
            mode: mode.package_id().to_owned(),
        })
    }

    /// Set one adjust slider on the selected picture; zero takes it off.
    pub fn set_adjust(
        &mut self,
        key: cutlass_effects::AdjustKey,
        value: f64,
    ) -> Result<(), ShellError> {
        let segment_id = self.adjust_segment()?;
        self.refuse_locked(&segment_id)?;
        let material_id = MaterialId::new(self.fresh("adjust"));
        self.apply(Request::SetAdjust {
            segment_id,
            material_id,
            key: key.kind().to_owned(),
            value,
        })
    }

    /// Set HSL colour `color` (0 red to 7 magenta) on the selected picture:
    /// hue, saturation and lightness from -1 to 1.
    pub fn set_hsl(&mut self, color: u8, values: [f64; 3]) -> Result<(), ShellError> {
        let segment_id = self.adjust_segment()?;
        self.refuse_locked(&segment_id)?;
        let material_id = MaterialId::new(self.fresh("hsl"));
        let [hue, saturation, lightness] = values;
        self.apply(Request::SetHsl {
            segment_id,
            material_id,
            color,
            hue,
            saturation,
            lightness,
        })
    }

    /// Set one curve on the selected picture; no points is the straight line.
    pub fn set_curve(
        &mut self,
        channel: cutlass_effects::CurveChannel,
        points: &[(f32, f32)],
    ) -> Result<(), ShellError> {
        let segment_id = self.adjust_segment()?;
        self.refuse_locked(&segment_id)?;
        let material_id = MaterialId::new(self.fresh("curves"));
        self.apply(Request::SetCurve {
            segment_id,
            material_id,
            channel: channel.key().to_owned(),
            points: points
                .iter()
                .map(|(x, y)| [f64::from(*x), f64::from(*y)])
                .collect(),
        })
    }

    /// Put the `.cube` file at `path` on the selected picture at full
    /// strength, named after the file.
    pub fn load_lut(&mut self, path: &Path) -> Result<(), ShellError> {
        const LARGEST: u64 = 16 * 1024 * 1024;
        let segment_id = self.adjust_segment()?;
        self.refuse_locked(&segment_id)?;
        let unreadable = |reason: String| {
            ShellError::Unavailable(format!("the LUT could not be read: {reason}"))
        };
        let size = std::fs::metadata(path)
            .map_err(|error| unreadable(error.to_string()))?
            .len();
        if size > LARGEST {
            return Err(unreadable("it is larger than 16 MB".to_owned()));
        }
        let text = std::fs::read_to_string(path).map_err(|error| unreadable(error.to_string()))?;
        cutlass_effects::Lut::parse_cube(&text).map_err(|error| unreadable(error.to_string()))?;
        let name = path.file_stem().map_or_else(
            || "LUT".to_owned(),
            |stem| stem.to_string_lossy().into_owned(),
        );
        let material_id = MaterialId::new(self.fresh("lut"));
        self.apply(Request::SetLut {
            segment_id,
            material_id,
            name,
            cube: Some(text),
            strength: 1.0,
        })
    }

    /// Key `color` (`#rrggbb`) out of the selected picture with `strength`
    /// and `shadow`, 0 to 1; an empty colour takes the key off.
    pub fn set_chroma(
        &mut self,
        color: &str,
        strength: f64,
        shadow: f64,
    ) -> Result<(), ShellError> {
        let segment_id = self.picture_segment("a chroma key belongs on a picture")?;
        self.refuse_locked(&segment_id)?;
        let material_id = MaterialId::new(self.fresh("chroma"));
        self.apply(Request::SetChroma {
            segment_id,
            material_id,
            color: color.to_owned(),
            strength,
            shadow,
        })
    }

    /// Put `mask` on the selected picture, in its own frame, or take its
    /// mask off.
    pub fn set_mask(&mut self, mask: Option<cutlass_effects::Mask>) -> Result<(), ShellError> {
        let segment_id = self.picture_segment("a mask belongs on a picture")?;
        self.refuse_locked(&segment_id)?;
        let material_id = MaterialId::new(self.fresh("mask"));
        self.apply(Request::SetMask {
            segment_id,
            material_id,
            mask,
        })
    }

    /// Keep the selected picture's colour work, or the selected adjustment
    /// layer, inside `mask`, or take that limit off.
    pub fn set_color_mask(
        &mut self,
        mask: Option<cutlass_effects::Mask>,
    ) -> Result<(), ShellError> {
        let segment_id = self.adjust_segment()?;
        self.refuse_locked(&segment_id)?;
        let material_id = MaterialId::new(self.fresh("color-mask"));
        self.apply(Request::SetColorMask {
            segment_id,
            material_id,
            mask,
        })
    }

    /// Set the strength of the selected picture's LUT, from 0 to 1.
    pub fn set_lut_strength(&mut self, strength: f64) -> Result<(), ShellError> {
        let segment_id = self.adjust_segment()?;
        self.refuse_locked(&segment_id)?;
        let material_id = MaterialId::new(self.fresh("lut"));
        self.apply(Request::SetLut {
            segment_id,
            material_id,
            name: String::new(),
            cube: None,
            strength,
        })
    }

    /// Take the LUT off the selected picture.
    pub fn remove_lut(&mut self) -> Result<(), ShellError> {
        let segment_id = self.adjust_segment()?;
        self.refuse_locked(&segment_id)?;
        self.apply(Request::SetLut {
            segment_id,
            material_id: MaterialId::new(""),
            name: String::new(),
            cube: Some(String::new()),
            strength: 1.0,
        })
    }

    /// Set one colour wheel on the selected picture.
    pub fn set_wheel(
        &mut self,
        wheel: cutlass_effects::Wheel,
        setting: cutlass_effects::WheelSetting,
    ) -> Result<(), ShellError> {
        let segment_id = self.adjust_segment()?;
        self.refuse_locked(&segment_id)?;
        let material_id = MaterialId::new(self.fresh("wheels"));
        let setting = setting.held();
        self.apply(Request::SetWheel {
            segment_id,
            material_id,
            wheel: wheel.key().to_owned(),
            x: f64::from(setting.x),
            y: f64::from(setting.y),
            luma: f64::from(setting.luma),
        })
    }

    /// Put every adjust slider on the selected picture back to zero, as one
    /// undo step. Nothing set is nothing done.
    pub fn reset_adjust(&mut self) -> Result<(), ShellError> {
        let segment_id = self.adjust_segment()?;
        self.refuse_locked(&segment_id)?;
        let Some(adjust) = self.inspection().map(|inspection| inspection.adjust) else {
            return Ok(());
        };
        let requests: Vec<Request> = cutlass_effects::AdjustKey::ALL
            .into_iter()
            .filter(|key| adjust.get(*key) != 0.0)
            .map(|key| Request::SetAdjust {
                segment_id: segment_id.clone(),
                material_id: MaterialId::new(""),
                key: key.kind().to_owned(),
                value: 0.0,
            })
            .collect();
        if requests.is_empty() {
            return Ok(());
        }
        self.apply_requests(&requests)
    }

    /// Look at the selected picture at the playhead, drawn alone and
    /// without its colour work, and set exposure, contrast, temperature,
    /// tint and saturation to even it out. One undo step.
    pub fn auto_adjust(&mut self) -> Result<(), ShellError> {
        let segment_id = self.picture_segment("auto adjust looks at a picture")?;
        self.refuse_locked(&segment_id)?;
        let own = self.drawn_alone(&segment_id)?;
        let adjust = cutlass_effects::auto_adjust(&own);
        self.set_auto_keys(&segment_id, &adjust)
    }

    /// Bring the selected picture's look toward `reference`'s, each drawn
    /// alone and without its colour work at the playhead. One undo step.
    pub fn match_colour(&mut self, reference: &SegmentId) -> Result<(), ShellError> {
        let segment_id = self.picture_segment("colour match looks at a picture")?;
        self.refuse_locked(&segment_id)?;
        if *reference == segment_id {
            return Err(ShellError::Unavailable(
                "pick another clip to match".to_owned(),
            ));
        }
        let own = self.drawn_alone(&segment_id)?;
        let goal = self.drawn_alone(reference)?;
        let adjust = cutlass_effects::match_adjust(&own, &goal);
        self.set_auto_keys(&segment_id, &adjust)
    }

    /// The other picture clips the selected one can match, by name, in
    /// timeline order.
    pub fn match_choices(&self) -> Vec<(SegmentId, String)> {
        let Some(selected) = self.selected().cloned() else {
            return Vec::new();
        };
        let Ok(draft) = self.session_draft() else {
            return Vec::new();
        };
        let mut choices: Vec<(i64, SegmentId, String)> = draft
            .tracks
            .iter()
            .flat_map(|track| {
                track
                    .segments
                    .iter()
                    .map(move |segment| (&track.kind, segment))
            })
            .filter(|(_, segment)| segment.id != selected)
            .filter(|(kind, segment)| cutlass_editor::can_turn(draft, segment, kind))
            .filter(|(_, segment)| {
                matches!(
                    draft.materials.get(segment.material_id.as_str()),
                    Some(MaterialRef::Video(_))
                )
            })
            .map(|(_, segment)| {
                (
                    segment.target.start.0,
                    segment.id.clone(),
                    clip_name(draft, segment),
                )
            })
            .collect();
        choices.sort_by_key(|(start, _, _)| *start);
        choices
            .into_iter()
            .map(|(_, id, name)| (id, name))
            .collect()
    }

    /// Set the sliders auto adjust and colour match set to `adjust`'s, in
    /// one undo step; the ones already there are left.
    fn set_auto_keys(
        &mut self,
        segment_id: &SegmentId,
        adjust: &cutlass_effects::Adjust,
    ) -> Result<(), ShellError> {
        let current = {
            let draft = self.session_draft()?;
            placed(draft, segment_id)
                .map(|(segment, _)| cutlass_effects::segment_adjust(draft, segment))
                .unwrap_or_default()
        };
        let mut requests = Vec::new();
        for key in cutlass_effects::AUTO_KEYS {
            let value = adjust.get(key);
            if value == current.get(key) {
                continue;
            }
            let material_id = MaterialId::new(self.fresh("adjust"));
            requests.push(Request::SetAdjust {
                segment_id: segment_id.clone(),
                material_id,
                key: key.kind().to_owned(),
                value: f64::from(value),
            });
        }
        if requests.is_empty() {
            return Ok(());
        }
        self.apply_requests(&requests)
    }

    /// `segment_id` drawn alone, without its colour work, on a clear
    /// canvas, at the playhead or the nearest frame of the clip.
    fn drawn_alone(&self, segment_id: &SegmentId) -> Result<Vec<u8>, ShellError> {
        let mut draft = self.session_draft()?.clone();
        let mut span = None;
        for track in &mut draft.tracks {
            track.segments.retain(|segment| segment.id == *segment_id);
            if let Some(segment) = track.segments.first() {
                span = Some(segment.target.clone());
            }
        }
        let Some(span) = span else {
            return Err(ShellError::Unavailable(
                "the clip is not on the timeline".to_owned(),
            ));
        };
        let colour: Vec<MaterialId> = draft
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .flat_map(|segment| &segment.extra_material_refs)
            .filter(|id| {
                draft
                    .materials
                    .get(id.as_str())
                    .is_some_and(cutlass_effects::is_colour_work)
            })
            .cloned()
            .collect();
        for segment in draft
            .tracks
            .iter_mut()
            .flat_map(|track| &mut track.segments)
        {
            segment
                .extra_material_refs
                .retain(|id| !colour.contains(id));
        }
        draft.canvas_config.background = None;
        let end = span.start.0.saturating_add(span.duration.0.max(1) - 1);
        let time = Micros(self.sample_time().0.clamp(span.start.0, end));
        let alone = Session::open(draft);
        let root = self.path.clone();
        let picture = FrameCache::new()
            .frame(
                &alone,
                &mut SoftwareDevice::new(),
                time,
                &self.readers,
                root.as_deref(),
                &*self.face,
            )
            .map_err(|error| ShellError::Eval(error.to_string()))?;
        Ok(picture.rgba)
    }

    pub fn set_opacity(&mut self, opacity: f64) -> Result<(), ShellError> {
        let segment_id = self.picture_segment("an opacity belongs on a picture")?;
        self.refuse_locked(&segment_id)?;
        self.apply(Request::SetOpacity {
            segment_id,
            opacity,
        })
    }

    /// The selected clip is a picture or a sound, so its fade can change.
    #[must_use]
    pub fn fade_enabled(&self) -> bool {
        self.reverse_enabled()
    }

    /// Fade the selected clip in. `seconds` is the duration, and the fade-out
    /// stays. One undo restores the previous fade-in.
    pub fn set_fade_in(&mut self, seconds: f64) -> Result<(), ShellError> {
        let fade_out = self.stored_fade()?.1;
        self.write_fade(fade_micros(seconds)?, fade_out)
    }

    /// Fade the selected clip out. `seconds` is the duration, and the fade-in
    /// stays. One undo restores the previous fade-out.
    pub fn set_fade_out(&mut self, seconds: f64) -> Result<(), ShellError> {
        let fade_in = self.stored_fade()?.0;
        self.write_fade(fade_in, fade_micros(seconds)?)
    }

    fn stored_fade(&self) -> Result<(Micros, Micros), ShellError> {
        let segment_id = self.sound_segment()?;
        self.refuse_locked(&segment_id)?;
        let draft = self.session_draft()?;
        let Some((segment, _)) = placed(draft, &segment_id) else {
            return Err(ShellError::Unavailable(
                "the clip is not on the timeline".to_owned(),
            ));
        };
        let (fade_in, fade_out) = fade_durations(draft, segment);
        Ok((Micros(fade_in), Micros(fade_out)))
    }

    fn write_fade(&mut self, fade_in: Micros, fade_out: Micros) -> Result<(), ShellError> {
        let segment_id = self.sound_segment()?;
        self.refuse_locked(&segment_id)?;
        let fade_id = MaterialId::new(self.fresh("fade"));
        self.apply(Request::SetFade {
            segment_id,
            fade_id,
            fade_in,
            fade_out,
        })
    }

    fn sound_segment(&self) -> Result<SegmentId, ShellError> {
        let Some(segment_id) = self.selected().cloned() else {
            return Err(if self.session.is_none() {
                ShellError::NoProject
            } else {
                ShellError::Unavailable("nothing is selected".to_owned())
            });
        };
        let draft = self.session_draft()?;
        let Some((segment, _)) = placed(draft, &segment_id) else {
            return Err(ShellError::Unavailable(
                "the clip is not on the timeline".to_owned(),
            ));
        };
        let sound = matches!(
            draft.materials.get(segment.material_id.as_str()),
            Some(MaterialRef::Video(_) | MaterialRef::Audio(_))
        );
        if !sound {
            return Err(ShellError::Unavailable(
                "a fade belongs on a picture or a sound".to_owned(),
            ));
        }
        Ok(segment_id)
    }

    /// The selected picture, adjustment layer, or filter: the clips that
    /// take adjust sliders.
    fn adjust_segment(&self) -> Result<SegmentId, ShellError> {
        if let Some(segment_id) = self.selected()
            && let Some((_, TrackType::Adjust | TrackType::Filter)) =
                placed(self.session_draft()?, segment_id)
        {
            return Ok(segment_id.clone());
        }
        self.picture_segment("adjustments belong on a picture")
    }

    fn picture_segment(&self, refusal: &str) -> Result<SegmentId, ShellError> {
        let Some(segment_id) = self.selected().cloned() else {
            return Err(if self.session.is_none() {
                ShellError::NoProject
            } else {
                ShellError::Unavailable("nothing is selected".to_owned())
            });
        };
        let draft = self.session_draft()?;
        let Some((segment, kind)) = placed(draft, &segment_id) else {
            return Err(ShellError::Unavailable(
                "the clip is not on the timeline".to_owned(),
            ));
        };
        if !cutlass_editor::can_turn(draft, segment, kind) {
            return Err(ShellError::Unavailable(refusal.to_owned()));
        }
        Ok(segment_id)
    }

    /// Store the text size in pixels. One undo restores the previous size.
    /// An empty face path takes the editor's face.
    pub fn set_text_size(&mut self, size: f64) -> Result<(), ShellError> {
        let segment_id = self.text_segment()?;
        self.refuse_locked(&segment_id)?;
        let face = self.face.name().to_owned();
        self.apply(Request::SetTextSize {
            segment_id,
            size,
            face,
        })
    }

    /// Store the extra gap between letters, as a fraction of the size.
    /// `0` is the normal gap. One undo restores the previous gap.
    pub fn set_letter_spacing(&mut self, spacing: f64) -> Result<(), ShellError> {
        let segment_id = self.text_segment()?;
        self.refuse_locked(&segment_id)?;
        let face = self.face.name().to_owned();
        self.apply(Request::SetLetterSpacing {
            segment_id,
            spacing,
            face,
        })
    }

    /// Store the wrap width in pixels. `0` keeps one line. One undo restores it.
    pub fn set_line_width(&mut self, width: f64) -> Result<(), ShellError> {
        let segment_id = self.text_segment()?;
        self.refuse_locked(&segment_id)?;
        let face = self.face.name().to_owned();
        self.apply(Request::SetLineWidth {
            segment_id,
            width,
            face,
        })
    }

    /// Store the extra gap between lines, as a fraction of the size.
    /// `0` is the normal gap. One undo restores the previous gap.
    pub fn set_line_spacing(&mut self, spacing: f64) -> Result<(), ShellError> {
        let segment_id = self.text_segment()?;
        self.refuse_locked(&segment_id)?;
        let face = self.face.name().to_owned();
        self.apply(Request::SetLineSpacing {
            segment_id,
            spacing,
            face,
        })
    }

    /// Store the text color. An empty string is white. One undo restores it.
    pub fn set_text_color(&mut self, color: &str) -> Result<(), ShellError> {
        let segment_id = self.text_segment()?;
        self.refuse_locked(&segment_id)?;
        let face = self.face.name().to_owned();
        self.apply(Request::SetTextColor {
            segment_id,
            color: color.to_owned(),
            face,
        })
    }

    /// Step to the next installed face. One undo restores the previous name.
    /// The words and the size stay. The same face is not an undo step.
    pub fn cycle_text_font(&mut self) -> Result<(), ShellError> {
        let segment_id = self.text_segment()?;
        self.refuse_locked(&segment_id)?;
        let current = {
            let draft = self.session_draft()?;
            text_font(draft, &segment_id)
        };
        let names = self.face.names();
        let Some(next) = next_face(&names, &current) else {
            return Err(ShellError::Unavailable("no face is installed".to_owned()));
        };
        if next == current {
            return Ok(());
        }
        self.apply(Request::SetTextFont {
            segment_id,
            face: next,
        })
    }

    /// Cycle left, center, and right. The first step gives the line the canvas
    /// width when it has none. One undo restores the previous alignment.
    pub fn cycle_text_align(&mut self) -> Result<(), ShellError> {
        let segment_id = self.text_segment()?;
        self.refuse_locked(&segment_id)?;
        let alignment = match self.inspection().map(|item| item.text_align).unwrap_or(0) {
            0 => 1,
            1 => 2,
            _ => 0,
        };
        let line_width = f64::from(
            self.draft()
                .map(|draft| draft.canvas_config.width)
                .unwrap_or(0),
        );
        let face = self.face.name().to_owned();
        self.apply(Request::SetTextAlign {
            segment_id,
            alignment,
            face,
            line_width,
        })
    }

    fn text_segment(&self) -> Result<SegmentId, ShellError> {
        let Some(segment_id) = self.selected().cloned() else {
            return Err(if self.session.is_none() {
                ShellError::NoProject
            } else {
                ShellError::Unavailable("nothing is selected".to_owned())
            });
        };
        let draft = self.session_draft()?;
        let Some((segment, _)) = placed(draft, &segment_id) else {
            return Err(ShellError::Unavailable(
                "the clip is not on the timeline".to_owned(),
            ));
        };
        if !matches!(
            draft.materials.get(segment.material_id.as_str()),
            Some(MaterialRef::Text(_))
        ) {
            return Err(ShellError::Unavailable(
                "a text style belongs on text".to_owned(),
            ));
        }
        Ok(segment_id)
    }

    /// Replace the installed face. Tests use this so a frame does not depend
    /// on a font file outside the repo.
    #[cfg(test)]
    pub fn use_face(&mut self, name: &str, bytes: Vec<u8>) {
        self.face = Arc::new(FaceTable::named(name, bytes));
    }

    #[cfg(test)]
    pub fn use_faces(&mut self, faces: Vec<(&str, Vec<u8>)>) {
        self.face = Arc::new(FaceTable::from_pairs(
            faces
                .into_iter()
                .map(|(name, bytes)| (name.to_owned(), bytes))
                .collect(),
        ));
    }

    pub fn remove_selected(&mut self) -> Result<(), ShellError> {
        let ids = {
            let draft = self.session_draft()?;
            self.selection
                .iter()
                .filter(|id| locate(draft, id).is_some())
                .cloned()
                .collect::<Vec<_>>()
        };
        if ids.is_empty() {
            self.selection.clear();
            return Err(ShellError::NoProject);
        }
        for id in &ids {
            self.refuse_locked(id)?;
        }
        let requests: Vec<Request> = ids
            .iter()
            .cloned()
            .map(|segment_id| Request::Remove { segment_id })
            .collect();
        self.apply_requests(&requests)?;
        self.selection.clear();
        if !self.has_clip() {
            self.blade = false;
            self.marking = false;
            self.beating = false;
        }
        Ok(())
    }

    /// Remove the selected clips and pull the later clips on their rows
    /// left into the space. One undo step.
    pub fn ripple_delete(&mut self) -> Result<(), ShellError> {
        let ids: Vec<SegmentId> = {
            let draft = self.session_draft()?;
            self.selection
                .iter()
                .filter(|id| locate(draft, id).is_some())
                .cloned()
                .collect()
        };
        if ids.is_empty() {
            return Err(ShellError::Unavailable("nothing is selected".to_owned()));
        }
        for id in &ids {
            self.refuse_locked(id)?;
        }
        let requests = cutlass_editor::ripple_delete(self.session_draft()?, &ids);
        self.apply_requests(&requests)?;
        self.selection.clear();
        Ok(())
    }

    /// Pull the clips after the gap at `time` on row `track_id` left by the
    /// gap's length. One undo step.
    pub fn close_gap(&mut self, track_id: &TrackId, time: Micros) -> Result<(), ShellError> {
        let draft = self.session_draft()?;
        let locked = draft
            .tracks
            .iter()
            .find(|track| &track.id == track_id)
            .is_some_and(|track| bit(track.attribute, TrackAttribute::LOCK));
        if locked {
            return Err(ShellError::Locked);
        }
        let requests = cutlass_editor::close_gap(draft, track_id, time);
        if requests.is_empty() {
            return Err(ShellError::Unavailable("there is no gap there".to_owned()));
        }
        self.apply_requests(&requests)
    }

    /// Open the Close gap menu at `(x, y)` when `time` on row `track_id` is
    /// in a gap. Answers whether it opened.
    pub fn open_gap_menu(&mut self, track_id: &TrackId, time: Micros, x: f32, y: f32) -> bool {
        let gap = self
            .draft()
            .and_then(|draft| cutlass_editor::gap_at(draft, track_id, time));
        self.gap_menu = gap.map(|_| GapMenu {
            x,
            y,
            track_id: track_id.clone(),
            time,
        });
        self.gap_menu.is_some()
    }

    pub fn close_gap_menu(&mut self) {
        self.gap_menu = None;
    }

    #[must_use]
    pub fn gap_menu(&self) -> Option<&GapMenu> {
        self.gap_menu.as_ref()
    }

    /// Remember the selection for paste. The draft stays as it is.
    pub fn copy_selected(&mut self) -> Result<(), ShellError> {
        if self.selection.is_empty() {
            return Err(ShellError::Unavailable("nothing is selected".to_owned()));
        }
        let copies =
            copy_clips(self.session_draft()?, &self.selection).map_err(|miss| match miss {
                ClipboardMiss::Locked => ShellError::Locked,
                _ => ShellError::Unavailable("nothing is selected".to_owned()),
            })?;
        self.clipboard = copies;
        Ok(())
    }

    /// Copy the selection, then remove it. One undo puts the clips back.
    pub fn cut_selected(&mut self) -> Result<(), ShellError> {
        self.copy_selected()?;
        self.remove_selected()?;
        self.clip_menu = None;
        Ok(())
    }

    /// Place the copied clips at the playhead, keeping the gaps between them.
    pub fn paste(&mut self) -> Result<(), ShellError> {
        let playhead = self.playhead;
        let copies = std::mem::take(&mut self.clipboard);
        let mut next_id = self.next_id;
        let mut fresh = |prefix: &str| {
            next_id = next_id.saturating_add(1);
            format!("{prefix}-{next_id}")
        };
        let planned = paste_requests(self.session_draft()?, &copies, playhead, &mut fresh)
            .map_err(clipboard_error);
        self.next_id = next_id;
        self.clipboard = copies;
        let (requests, ids) = planned?;
        self.apply_requests(&requests)?;
        self.selection = ids;
        self.clip_menu = None;
        Ok(())
    }

    #[must_use]
    pub fn has_clipboard(&self) -> bool {
        !self.clipboard.is_empty()
    }

    pub fn open_clip_menu(&mut self, x: f32, y: f32) {
        self.clip_menu = Some(ClipMenu { x, y });
    }

    pub fn close_clip_menu(&mut self) {
        self.clip_menu = None;
    }

    #[must_use]
    pub fn clip_menu(&self) -> Option<ClipMenu> {
        self.clip_menu
    }

    /// The kind of track the selected clip sits on.
    #[must_use]
    pub fn selected_track_kind(&self) -> Option<TrackType> {
        let id = self.selected()?;
        self.draft()?
            .tracks
            .iter()
            .find(|track| track.segments.iter().any(|segment| &segment.id == id))
            .map(|track| track.kind.clone())
    }

    /// The split tool is armed. A click on a clip cuts it at that time.
    #[must_use]
    pub fn blade(&self) -> bool {
        self.blade
    }

    /// Clips on a speed curve draw it on their bar.
    pub fn speed_curves_shown(&self) -> bool {
        self.speed_curves
    }

    pub fn toggle_speed_curves(&mut self) {
        self.speed_curves = !self.speed_curves;
    }

    pub fn toggle_blade(&mut self) {
        self.blade = !self.blade && self.has_clip();
        if self.blade {
            self.marking = false;
            self.beating = false;
        }
    }

    /// Turn the blade off. Select uses this.
    pub fn clear_blade(&mut self) {
        self.blade = false;
    }

    /// The marker tool is armed. A click on the ruler or a clip drops a mark.
    #[must_use]
    pub fn marking(&self) -> bool {
        self.marking
    }

    pub fn toggle_marker(&mut self) {
        self.marking = !self.marking && self.has_clip();
        if self.marking {
            self.blade = false;
            self.beating = false;
        }
    }

    /// Turn the marker tool off. Select uses this.
    pub fn clear_marker(&mut self) {
        self.marking = false;
    }

    /// Markers currently drawn. A clip mark past the clip is omitted.
    #[must_use]
    pub fn markers(&self) -> Vec<MarkerView> {
        let Some(draft) = self.draft() else {
            return Vec::new();
        };
        visible_marks(draft)
            .into_iter()
            .map(|mark| MarkerView {
                segment_id: mark.segment_id,
                stored_start: mark.stored_start,
                time: mark.time,
                color: mark.color,
                title: mark.title,
            })
            .collect()
    }

    #[must_use]
    pub fn marker_selected(&self, segment_id: &Option<SegmentId>, stored_start: i64) -> bool {
        self.selected_marker
            .as_ref()
            .is_some_and(|pick| &pick.segment_id == segment_id && pick.stored_start == stored_start)
    }

    pub fn select_marker(&mut self, segment_id: Option<SegmentId>, stored_start: i64) {
        let present = self
            .markers()
            .iter()
            .any(|mark| mark.segment_id == segment_id && mark.stored_start == stored_start);
        if !present {
            return;
        }
        self.selected_marker = Some(MarkerPick {
            segment_id,
            stored_start,
        });
        self.selected_beat = None;
        self.selected_mark = None;
        self.selection.clear();
    }

    /// Drop a mark at the pointer. `None` is the ruler. `Some` is that clip.
    ///
    /// A ruler mark is a timeline time. A clip mark is stored as an offset
    /// from the clip start, so a later move carries it.
    pub fn drop_marker(
        &mut self,
        segment_id: Option<SegmentId>,
        x: f64,
        y: f64,
    ) -> Result<(), ShellError> {
        if !self.has_clip() {
            return Err(ShellError::Unavailable(
                "marker is not available".to_owned(),
            ));
        }
        let x = self.time_zero_x(x);
        let scale = self.scale();
        let pointer_y = if segment_id.is_some() { y } else { 0.0 };
        let at = {
            let draft = self.session_draft()?;
            let hit = hit_test(draft, scale, Pointer { x, y: pointer_y }).map_err(|_| {
                ShellError::Unavailable("the marker missed the timeline".to_owned())
            })?;
            let Some(hit) = hit else {
                return Err(ShellError::Unavailable(
                    "the marker missed the timeline".to_owned(),
                ));
            };
            if let Some(id) = &segment_id
                && hit.segment_id.as_ref() != Some(id)
            {
                return Err(ShellError::Unavailable(
                    "the marker missed the clip".to_owned(),
                ));
            }
            Micros(hit.time.0.max(0))
        };
        let material_id = MaterialId::new(self.fresh("time-mark"));
        self.apply(Request::AddTimeMark {
            segment_id,
            at,
            color: MARKER_COLOR.to_owned(),
            material_id,
        })
    }

    fn remove_selected_marker(&mut self) -> Result<(), ShellError> {
        let Some(pick) = self.selected_marker.clone() else {
            return Ok(());
        };
        let present = self.markers().iter().any(|mark| {
            mark.segment_id == pick.segment_id && mark.stored_start == pick.stored_start
        });
        if !present {
            self.selected_marker = None;
            return Ok(());
        }
        self.apply(Request::RemoveTimeMark {
            segment_id: pick.segment_id,
            start: Micros(pick.stored_start),
        })?;
        self.selected_marker = None;
        Ok(())
    }

    /// The beat tool is armed. A click on an audio clip drops a beat.
    #[must_use]
    pub fn beating(&self) -> bool {
        self.beating
    }

    pub fn toggle_beat(&mut self) {
        self.beating = !self.beating && self.has_clip();
        if self.beating {
            self.blade = false;
            self.marking = false;
        }
    }

    /// Turn the beat tool off. Select uses this.
    pub fn clear_beat(&mut self) {
        self.beating = false;
    }

    /// Beats currently drawn. A beat outside the clip is omitted.
    #[must_use]
    pub fn beats(&self) -> Vec<BeatView> {
        let Some(draft) = self.draft() else {
            return Vec::new();
        };
        visible_beats(draft)
            .into_iter()
            .map(|beat| BeatView {
                segment_id: beat.segment_id,
                source: beat.source,
                time: beat.time,
            })
            .collect()
    }

    #[must_use]
    pub fn beat_selected(&self, segment_id: &SegmentId, source: i64) -> bool {
        self.selected_beat
            .as_ref()
            .is_some_and(|pick| &pick.segment_id == segment_id && pick.source == source)
    }

    pub fn select_beat(&mut self, segment_id: SegmentId, source: i64) {
        let present = self
            .beats()
            .iter()
            .any(|beat| beat.segment_id == segment_id && beat.source == source);
        if !present {
            return;
        }
        self.selected_beat = Some(BeatPick { segment_id, source });
        self.selected_marker = None;
        self.selected_mark = None;
        self.selection.clear();
    }

    /// Drop a beat on an audio clip. The stored time is the source time, so a
    /// later move carries the mark.
    pub fn drop_beat(&mut self, segment_id: SegmentId, x: f64, y: f64) -> Result<(), ShellError> {
        if !self.has_clip() {
            return Err(ShellError::Unavailable("beat is not available".to_owned()));
        }
        let x = self.time_zero_x(x);
        let scale = self.scale();
        let source = {
            let draft = self.session_draft()?;
            let hit = hit_test(draft, scale, Pointer { x, y })
                .map_err(|_| ShellError::Unavailable("the beat missed the clip".to_owned()))?;
            let Some(hit) = hit else {
                return Err(ShellError::Unavailable(
                    "the beat missed the clip".to_owned(),
                ));
            };
            if hit.segment_id.as_ref() != Some(&segment_id) {
                return Err(ShellError::Unavailable(
                    "the beat missed the clip".to_owned(),
                ));
            }
            if !track_holds_beats(&hit.lane.kind) {
                return Err(ShellError::Unavailable(
                    "a beat belongs on audio".to_owned(),
                ));
            }
            let (segment, _) = locate(draft, &segment_id)
                .ok_or_else(|| ShellError::Unavailable("the beat missed the clip".to_owned()))?;
            source_time_at(segment, Micros(hit.time.0.max(0)))
                .ok_or_else(|| ShellError::Unavailable("the beat missed the clip".to_owned()))?
        };
        let material_id = MaterialId::new(self.fresh("beats"));
        self.apply(Request::AddBeat {
            segment_id,
            source: Micros(source),
            material_id,
        })
    }

    fn remove_selected_beat(&mut self) -> Result<(), ShellError> {
        let Some(pick) = self.selected_beat.clone() else {
            return Ok(());
        };
        let present = self
            .beats()
            .iter()
            .any(|beat| beat.segment_id == pick.segment_id && beat.source == pick.source);
        if !present {
            self.selected_beat = None;
            return Ok(());
        }
        self.apply(Request::RemoveBeat {
            segment_id: pick.segment_id,
            source: Micros(pick.source),
        })?;
        self.selected_beat = None;
        Ok(())
    }

    /// Any lane has a clip. An empty cut leaves the blade off.
    #[must_use]
    pub fn has_clip(&self) -> bool {
        self.draft()
            .is_some_and(|draft| draft.tracks.iter().any(|track| !track.segments.is_empty()))
    }

    /// Cut the selected clip at the playhead.
    pub fn split_selected(&mut self) -> Result<(), ShellError> {
        let segment_id = self.selected().cloned().ok_or(ShellError::NoProject)?;
        self.split_clip(segment_id, self.playhead)
    }

    /// Cut the clip under the playhead. The selection wins when it covers that time.
    pub fn split_playhead(&mut self) -> Result<(), ShellError> {
        let at = self.playhead;
        let segment_id = self
            .clip_under(at)
            .ok_or_else(|| ShellError::Unavailable("split is not available".to_owned()))?;
        self.split_clip(segment_id, at)
    }

    /// Delete the part of the clip under the playhead before it (`Start`) or after it (`End`).
    ///
    /// The selection wins when it covers the playhead. Each side must keep a
    /// frame, as in a split. After a left delete the playhead goes to the
    /// clip's new start, so the frame under it stays the same with the magnet on.
    pub fn delete_side(&mut self, edge: TrimEdge) -> Result<(), ShellError> {
        let at = self.playhead;
        let request = cutlass_editor::delete_side(self.session_draft()?, &self.selection, at, edge)
            .ok_or_else(|| ShellError::Unavailable("delete is not available".to_owned()))?;
        let Request::Trim { segment_id, .. } = &request else {
            return Err(ShellError::Unavailable(
                "delete is not available".to_owned(),
            ));
        };
        let segment_id = segment_id.clone();
        self.refuse_locked(&segment_id)?;
        self.apply(request)?;
        if edge == TrimEdge::Start
            && let Some((segment, _)) = self.draft().and_then(|draft| locate(draft, &segment_id))
        {
            let start = segment.target.start;
            self.seek(start);
        }
        self.select(Some(segment_id));
        Ok(())
    }

    /// Cut one clip at a lane pixel. `x` is from the left of the lanes.
    pub fn split_pointer(
        &mut self,
        segment_id: SegmentId,
        x: f64,
        y: f64,
    ) -> Result<(), ShellError> {
        let x = self.time_zero_x(x);
        let draft = self.session_draft()?;
        let time = match hit_test(draft, self.scale(), Pointer { x, y }) {
            Ok(Some(hit)) => hit.time,
            Ok(None) | Err(_) => {
                return Err(ShellError::Unavailable("split is not available".to_owned()));
            }
        };
        self.split_clip(segment_id, time)
    }

    fn clip_under(&self, time: Micros) -> Option<SegmentId> {
        cutlass_editor::clip_under(self.draft()?, &self.selection, time)
    }

    /// Cut one clip. Each piece must keep at least one frame. The original id stays selected.
    pub fn split_clip(&mut self, segment_id: SegmentId, at: Micros) -> Result<(), ShellError> {
        self.refuse_locked(&segment_id)?;
        let draft = self.session_draft()?;
        if locate(draft, &segment_id).is_none() {
            return Err(ShellError::Unavailable(
                "the clip is not on the timeline".to_owned(),
            ));
        }
        if !cutlass_editor::can_cut_at(draft, &segment_id, at) {
            return Err(ShellError::Unavailable("split is not available".to_owned()));
        }
        let right_id = SegmentId::new(self.fresh("seg"));
        self.apply(Request::Split {
            segment_id: segment_id.clone(),
            at,
            right_id,
        })?;
        self.select(Some(segment_id));
        Ok(())
    }

    pub fn set_notice(&mut self, message: impl Into<String>) {
        self.notice = message.into();
        self.notice_since = Instant::now();
    }

    pub fn unavailable(&mut self, control: &str) {
        self.set_notice(format!("{control} is not available"));
    }

    /// How long the notice has shown.
    pub fn notice_age(&self) -> Duration {
        self.notice_since.elapsed()
    }

    /// Drop the notice once it has faded out.
    pub fn expire_notice(&mut self) {
        if self.notice_age() >= NOTICE_SHOWN {
            self.notice.clear();
        }
    }

    /// Writes the export `request` names.
    ///
    /// PNG writes the frame at the playhead at once. Anything else runs as a
    /// job: a movie, or the sound alone when the pictures are off, then a
    /// GIF and a caption file beside it when asked. The window shows the
    /// job's progress; Cancel leaves no file. A codec this machine cannot
    /// encode is refused before anything is written.
    pub fn export(&mut self, request: ExportRequest) -> Result<(), ShellError> {
        if request.codec == CodecId::Png {
            return self.export_png(request);
        }
        let plan = self.export_plan(&request)?;
        if !request.replace
            && let Some(taken) = plan.paths().into_iter().find(|path| path.exists())
        {
            return Err(ShellError::Overwrite(taken));
        }
        self.run_export(plan)
    }

    fn export_png(&mut self, request: ExportRequest) -> Result<(), ShellError> {
        if !SHORT_SIDES.contains(&request.short_side) {
            return Err(ShellError::Unavailable(
                "the short side is not one of the steps".to_owned(),
            ));
        }
        if request.path.exists() && !request.replace {
            return Err(ShellError::Overwrite(request.path));
        }
        let root = self.path.clone();
        let draft = self.session_draft()?;
        let (width, height) = self.scaled_canvas(request.short_side)?;
        let params = CodecParams::video(request.codec, width, height);
        if let Err(error) = self.encoders.open(&params) {
            return Err(ShellError::CodecDeclined(error.to_string()));
        }
        let Ok(short_side) = u32::try_from(request.short_side) else {
            return Err(ShellError::Unavailable(
                "the short side is not one of the steps".to_owned(),
            ));
        };
        let rate = draft.frame_rate().unwrap_or(FrameRate::FPS_30);
        let range = playhead_frame(rate, self.playhead)?;
        let export_request = cutlass_export::ExportRequest {
            range,
            short_side,
            codec: CodecId::Png,
            bitrate: 0,
            frame_rate: rate,
            container: cutlass_codec::ContainerId::Still,
            audio: None,
            sidecar: None,
            late_by: 0,
            constant_rate: false,
            prores: ProresProfile::Standard,
        };
        let mut encoders = EncodeFactory::new();
        cutlass_png::register_encoders(&mut encoders);
        let mut mux = cutlass_still::StillMux::new();
        let mut device = SoftwareDevice::new();
        cutlass_export::export(
            cutlass_export::ExportRun {
                draft,
                device: &mut device,
                request: &export_request,
                factory: &encoders,
                readers: &self.readers,
                mux: &mut mux,
                destination: None,
                root: root.as_deref(),
                fonts: &*self.face,
                // The preview's device, so an effect clip is in the file.
                renderer: self.gpu.as_ref().map(crate::gpu::Preview::renderer),
                progress: None,
            },
            |_, _, _, _| Ok(cutlass_eval::ClipPcm::Missing),
        )
        .map_err(export_failure)?;
        let Some(bytes) = mux.bytes() else {
            return Err(ShellError::Unavailable(
                "export did not finish a file".to_owned(),
            ));
        };
        write_replaced(&request.path, bytes)
    }

    /// What a job export writes, checked before it starts.
    fn export_plan(&mut self, request: &ExportRequest) -> Result<ExportPlan, ShellError> {
        if !request.picture && !request.sound && request.gif.is_none() {
            return Err(ShellError::Unavailable(
                "turn on the video, the audio or the GIF".to_owned(),
            ));
        }
        if request.path.is_dir() {
            return Err(ShellError::Io("the export path is a folder".to_owned()));
        }
        if request.path.file_stem().is_none_or(|stem| stem.is_empty())
            || cutlass_export::part_path(&request.path).is_none()
        {
            return Err(ShellError::Io("the export has no file name".to_owned()));
        }
        let draft = self.session_draft()?.clone();
        let range = self.export_range(&draft, request.span)?;
        let frame_rate = request
            .frame_rate
            .or_else(|| draft.frame_rate())
            .unwrap_or(FrameRate::FPS_30);
        let base = cutlass_export::ExportRequest {
            range: range.clone(),
            short_side: 0,
            codec: request.codec,
            bitrate: 0,
            frame_rate,
            container: cutlass_codec::ContainerId::Isobmff,
            audio: None,
            sidecar: None,
            late_by: 0,
            constant_rate: request.constant_rate,
            prores: request.prores,
        };
        let captions = request.captions.map(|kind| cutlass_export::SidecarRequest {
            kind,
            name: request
                .path
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_default(),
        });
        let mut main = None;
        if request.picture {
            main = Some(self.movie_output(request, &base, captions.clone())?);
        } else if request.sound {
            let audio = request.audio.export();
            let container = if audio.codec == CodecId::Pcm {
                cutlass_codec::ContainerId::Riff
            } else {
                cutlass_codec::ContainerId::Isobmff
            };
            main = Some(ExportOutput::Sound {
                path: request.path.with_extension(request.audio.extension()),
                request: cutlass_export::ExportRequest {
                    audio: Some(audio),
                    container,
                    ..base.clone()
                },
            });
        }
        let gif = request.gif.map(|gif| ExportOutput::Gif {
            path: request.path.with_extension("gif"),
            looped: gif.looped,
            request: cutlass_export::ExportRequest {
                short_side: gif.short_side,
                codec: CodecId::Gif,
                frame_rate: gif.frame_rate,
                container: cutlass_codec::ContainerId::Still,
                ..base.clone()
            },
        });
        if let Some(ExportOutput::Gif { request, .. }) = &gif {
            cutlass_export::validate(request).map_err(export_failure)?;
        }
        // Without a movie to carry them, the captions are written alone.
        let captions = match (&main, captions) {
            (Some(ExportOutput::Movie { .. }), _) | (_, None) => None,
            (_, Some(sidecar)) => {
                let body = cutlass_export::sidecar_text(&draft, &range, sidecar.kind)
                    .map_err(export_failure)?;
                Some((sidecar, body))
            }
        };
        Ok(ExportPlan {
            draft,
            outputs: main.into_iter().chain(gif).collect(),
            captions,
            folder: request
                .path
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_default(),
        })
    }

    /// The movie a request names, with its encoder probed on this machine.
    fn movie_output(
        &self,
        request: &ExportRequest,
        base: &cutlass_export::ExportRequest,
        captions: Option<cutlass_export::SidecarRequest>,
    ) -> Result<ExportOutput, ShellError> {
        let Ok(short_side) = u32::try_from(request.short_side) else {
            return Err(ShellError::Unavailable(
                "the short side is not one of the steps".to_owned(),
            ));
        };
        if !SHORT_SIDES.contains(&request.short_side) {
            return Err(ShellError::Unavailable(
                "the short side is not one of the steps".to_owned(),
            ));
        }
        let mov = is_mov(&request.path);
        if request.codec == CodecId::Prores && !mov {
            return Err(ShellError::Unavailable(
                "ProRes is written in a MOV".to_owned(),
            ));
        }
        if request.sound && request.audio.codec == CodecId::Pcm && !mov {
            return Err(ShellError::Unavailable(
                "PCM sound is written in a MOV".to_owned(),
            ));
        }
        let bitrate = match request.bitrate {
            Some(0) => {
                return Err(ShellError::Unavailable(
                    "the bitrate is not a number of bits a second".to_owned(),
                ));
            }
            Some(rate) => rate.min(MAX_BITRATE),
            None => request.quality.bitrate(short_side, request.codec),
        };
        let (width, height) = self.scaled_canvas(request.short_side)?;
        let mut params = CodecParams::with_format(
            request.codec,
            width,
            height,
            cutlass_decode::PixelFormat::Rgba8,
        );
        if request.codec == CodecId::Prores {
            params.extra = request.prores.code().to_vec();
        }
        if let Err(error) = self.encoders.open(&params) {
            return Err(ShellError::CodecDeclined(error.to_string()));
        }
        if !matches!(
            request.codec,
            CodecId::H264 | CodecId::Hevc | CodecId::Prores
        ) {
            return Err(ShellError::ExportUnavailable);
        }
        let movie = cutlass_export::ExportRequest {
            short_side,
            bitrate,
            audio: request.sound.then(|| request.audio.export()),
            sidecar: captions,
            ..base.clone()
        };
        cutlass_export::validate(&movie).map_err(export_failure)?;
        Ok(ExportOutput::Movie {
            path: request.path.clone(),
            request: movie,
        })
    }

    /// Write `plan` as a job. When it lands, [`Self::exported`] names the
    /// main file.
    fn run_export(&mut self, plan: ExportPlan) -> Result<(), ShellError> {
        self.refuse_busy()?;
        let root = self.path.clone();
        let readers = self.readers.clone();
        let encoders = self.encoders.clone();
        let fonts = Arc::clone(&self.face);
        // The preview's device, so an effect clip is in the file.
        let gpu = self.gpu.as_ref().map(|preview| {
            let renderer = preview.renderer();
            (renderer.device().clone(), renderer.queue().clone())
        });
        let first = plan.paths().into_iter().next().unwrap_or_default();
        let name = first
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        self.exported = None;
        self.start_job("Export", move |progress| {
            let renderer = gpu.map(|(device, queue)| cutlass_render::Renderer::new(device, queue));
            // GIF pictures come from the Rust encoder, whose packets the
            // GIF file is built from.
            let mut gifs = EncodeFactory::new();
            cutlass_gif::register_encoders(&mut gifs);
            let steps = plan.outputs.len().max(1);
            let mut written = Ok(());
            for (step, output) in plan.outputs.iter().enumerate() {
                let path = output.path();
                let file = path
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default();
                progress.stage(&format!("Writing {file}"));
                let mut report = |done: u64, total: u64| {
                    if total > 0 {
                        let part = done as f32 / total as f32;
                        progress.fraction((step as f32 + part) / steps as f32);
                    }
                    !progress.cancelled()
                };
                let sources = cutlass_export::MovieSources {
                    draft: &plan.draft,
                    root: root.as_deref(),
                    readers: &readers,
                    fonts: &*fonts,
                    renderer: renderer.as_ref(),
                    encoders: &encoders,
                };
                written = match output {
                    ExportOutput::Movie { request, .. } => {
                        cutlass_export::write_movie(&sources, request, path, &mut report)
                            .map(|_| ())
                    }
                    ExportOutput::Sound { request, .. } => {
                        cutlass_export::write_sound(&sources, request, path, &mut report)
                    }
                    ExportOutput::Gif {
                        request, looped, ..
                    } => cutlass_export::write_gif(
                        &cutlass_export::MovieSources {
                            encoders: &gifs,
                            ..sources
                        },
                        request,
                        *looped,
                        path,
                        &mut report,
                    )
                    .map(|_| ()),
                };
                if written.is_err() {
                    break;
                }
            }
            if written.is_ok()
                && let Some((sidecar, body)) = &plan.captions
            {
                let extension = match sidecar.kind {
                    SidecarKind::Plain => "txt",
                    _ => "srt",
                };
                written = cutlass_export::write_sidecar(
                    &plan.folder,
                    &format!("{}.{extension}", sidecar.name),
                    body,
                )
                .map(|_| ());
            }
            Box::new(move |shell: &mut Shell| {
                written.map_err(export_failure)?;
                shell.set_notice(format!("Exported {name}"));
                shell.exported = Some(first);
                Ok(())
            })
        })
    }

    /// The stretch of the timeline `span` names.
    fn export_range(&self, draft: &Draft, span: ExportSpan) -> Result<TimeRange, ShellError> {
        let range = match span {
            // The stored length can lag the clips, so the end is where the
            // timeline shows the cut ending.
            ExportSpan::Whole => TimeRange::new(0, self.cut_duration().0),
            ExportSpan::Selected => {
                let picked = draft
                    .tracks
                    .iter()
                    .flat_map(|track| &track.segments)
                    .filter(|segment| self.selection.contains(&segment.id))
                    .map(|segment| {
                        (
                            segment.target.start.0,
                            segment.target.start.0 + segment.target.duration.0,
                        )
                    })
                    .reduce(|(start, end), (from, to)| (start.min(from), end.max(to)));
                let Some((start, end)) = picked else {
                    return Err(ShellError::Unavailable("no clip is selected".to_owned()));
                };
                TimeRange::new(start, end - start)
            }
            ExportSpan::Marked => {
                let (start, end) = self.marked_span().ok_or_else(|| {
                    ShellError::Unavailable("mark an in or an out first".to_owned())
                })?;
                TimeRange::new(start.0, end.0 - start.0)
            }
        };
        if range.duration.0 <= 0 {
            return Err(ShellError::Unavailable(
                "there is nothing to export".to_owned(),
            ));
        }
        Ok(range)
    }

    /// An encoder for `codec` opens on this machine.
    pub fn encodes(&self, codec: CodecId) -> bool {
        // Export hands the encoder RGBA pictures, so ask for those.
        self.encoders
            .open(&CodecParams::with_format(
                codec,
                1280,
                720,
                cutlass_decode::PixelFormat::Rgba8,
            ))
            .is_ok()
    }

    /// The movie the last export wrote, until the next export starts.
    pub fn exported(&self) -> Option<&Path> {
        self.exported.as_deref()
    }

    /// Save the picture at the playhead to `path` at the canvas size, as
    /// JPEG for a `.jpg` or `.jpeg` name and PNG otherwise. It runs as a
    /// job; when it lands it is offered like a finished export.
    /// Write a copy of the log, the previous file then the current one, to
    /// `path`. Before the log opened, it is the Cutlass root's logs folder.
    pub fn export_log(path: &Path) -> Result<(), ShellError> {
        if path.is_dir() {
            return Err(ShellError::Io("the log path is a folder".to_owned()));
        }
        let directory = cutlass_logger::directory().unwrap_or_else(|| Self::home().logs());
        cutlass_logger::export(&directory, path)
            .map_err(|error| ShellError::Io(format!("the log was not saved: {error}")))
    }

    pub fn export_still(&mut self, path: &Path) -> Result<(), ShellError> {
        self.refuse_busy()?;
        let draft = self.session_draft()?.clone();
        let jpeg = path.extension().is_some_and(|typed| {
            typed.eq_ignore_ascii_case("jpg") || typed.eq_ignore_ascii_case("jpeg")
        });
        let format = if jpeg {
            cutlass_export::StillFormat::Jpeg
        } else {
            cutlass_export::StillFormat::Png
        };
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .filter(|_| cutlass_export::part_path(path).is_some())
            .ok_or_else(|| ShellError::Io("the still has no file name".to_owned()))?;
        if path.is_dir() {
            return Err(ShellError::Io("the still path is a folder".to_owned()));
        }
        let time = self.playhead;
        let root = self.path.clone();
        let readers = self.readers.clone();
        let encoders = self.encoders.clone();
        let fonts = Arc::clone(&self.face);
        let gpu = self.gpu.as_ref().map(|preview| {
            let renderer = preview.renderer();
            (renderer.device().clone(), renderer.queue().clone())
        });
        let destination = path.to_path_buf();
        self.exported = None;
        self.start_job("Still", move |progress| {
            progress.stage(&format!("Writing {name}"));
            let renderer = gpu.map(|(device, queue)| cutlass_render::Renderer::new(device, queue));
            let written = cutlass_export::write_still(
                &cutlass_export::MovieSources {
                    draft: &draft,
                    root: root.as_deref(),
                    readers: &readers,
                    fonts: &*fonts,
                    renderer: renderer.as_ref(),
                    encoders: &encoders,
                },
                time,
                format,
                &destination,
            );
            Box::new(move |shell: &mut Shell| {
                written.map_err(export_failure)?;
                shell.set_notice(format!("Saved {name}"));
                shell.exported = Some(destination);
                Ok(())
            })
        })
    }

    /// Stop offering the last export.
    pub fn forget_export(&mut self) {
        self.exported = None;
    }

    pub fn drag_timeline(&mut self, delta_y: f32, window_height: f32) {
        let chrome = tokens::HEAD_HEIGHT + tokens::SPLIT + tokens::PREVIEW_MIN_HEIGHT;
        let max_timeline = (window_height - chrome).max(tokens::TIMELINE_MIN);
        self.timeline_height =
            (self.timeline_height - delta_y).clamp(tokens::TIMELINE_MIN, max_timeline);
    }

    pub fn drag_library(&mut self, delta_x: f32, window_width: f32) {
        let max = self.side_max(window_width, self.inspector_width, tokens::LIBRARY_MIN);
        self.library_width = (self.library_width + delta_x).clamp(tokens::LIBRARY_MIN, max);
    }

    pub fn drag_inspector(&mut self, delta_x: f32, window_width: f32) {
        let max = self.side_max(window_width, self.library_width, tokens::INSPECTOR_MIN);
        self.inspector_width = (self.inspector_width - delta_x).clamp(tokens::INSPECTOR_MIN, max);
    }

    pub fn preview_width(&self, window_width: f32) -> f32 {
        window_width - self.library_width - self.inspector_width - tokens::SPLIT * 2.0
    }

    pub fn timecode(&self) -> String {
        self.timecode_at(self.playhead)
    }

    pub fn timecode_at(&self, time: Micros) -> String {
        let Some(draft) = self.draft() else {
            return "00:00:00:00".to_owned();
        };
        draft
            .timecode(time)
            .map(format_timecode)
            .unwrap_or_else(|| "00:00:00:00".to_owned())
    }

    fn import_medium(&mut self, probed: Probed, stored: String) -> Result<(), ShellError> {
        let request = self.medium_request(probed, stored);
        self.apply(request)
    }

    /// The request that adds `probed`, stored at `stored`, to the library.
    fn medium_request(&mut self, probed: Probed, stored: String) -> Request {
        match probed {
            Probed::Video {
                meta,
                width,
                height,
                duration,
            } => {
                let material_id = self.fresh("mat");
                let mut material = MaterialVideo::new(material_id.as_str());
                material.kind = meta.as_str().to_owned();
                material.path = stored;
                material.duration = Micros(duration);
                material.width = width;
                material.height = height;
                Request::Import {
                    medium: ImportedMedium::Video(Box::new(material)),
                }
            }
            Probed::Audio { duration } => {
                let material_id = self.fresh("mat");
                let material = MaterialAudio {
                    id: MaterialId::new(&material_id),
                    kind: MetaType::Music.as_str().to_owned(),
                    path: stored,
                    duration: Micros(duration),
                    ..MaterialAudio::default()
                };
                Request::Import {
                    medium: ImportedMedium::Audio(Box::new(material)),
                }
            }
        }
    }

    fn apply(&mut self, request: Request) -> Result<(), ShellError> {
        self.apply_requests(std::slice::from_ref(&request))
    }

    fn apply_requests(&mut self, requests: &[Request]) -> Result<(), ShellError> {
        let kind = requests
            .first()
            .map(|request| (std::mem::discriminant(request), self.selected().cloned()));
        let session = self.session.as_mut().ok_or(ShellError::NoProject)?;
        let revision = session.revision();
        let joins = self.holding
            && kind.is_some()
            && self
                .held_edit
                .as_ref()
                .is_some_and(|(held, at)| Some(held) == kind.as_ref() && *at == revision);
        let next = if joins {
            session.fold(revision, requests)
        } else {
            session.apply(revision, requests)
        }
        .map_err(ShellError::Edit)?;
        self.held_edit = if self.holding {
            kind.map(|kind| (kind, next))
        } else {
            None
        };
        Ok(())
    }

    /// Whether the pointer is held down this frame. While it is, edits of
    /// one kind to one clip in a row join one undo step, so a slider drag
    /// shows each value and undoes in one go. Letting go ends the step.
    pub fn hold_edits(&mut self, holding: bool) {
        self.holding = holding;
        if !holding {
            self.held_edit = None;
        }
    }

    fn session_draft(&self) -> Result<&Draft, ShellError> {
        self.draft().ok_or(ShellError::NoProject)
    }

    fn fresh(&mut self, prefix: &str) -> String {
        self.next_id = self.next_id.saturating_add(1);
        format!("{prefix}-{}", self.next_id)
    }

    fn scale(&self) -> TimelineScale {
        TimelineScale {
            lane_height: tokens::LANE_HEIGHT,
            pixel_size: self.pixel_size,
            // The desktop adds and removes tracks, so an empty one stays.
            empty_lanes: true,
        }
    }

    fn zoom_bounds(&self) -> ZoomBounds {
        let rate = self
            .draft()
            .and_then(Draft::frame_rate)
            .unwrap_or(FrameRate::FPS_30);
        ZoomBounds {
            lane_width: self.lane_width,
            duration: self.cut_duration(),
            rate,
        }
    }

    fn cut_duration(&self) -> Micros {
        let Some(draft) = self.draft() else {
            return Micros(0);
        };
        let mut end = draft.duration.0;
        for track in &draft.tracks {
            for segment in &track.segments {
                end = end.max(segment.target.end().0);
            }
        }
        Micros(end.max(0))
    }

    /// `x` is pixels from the left of the lanes. Hit testing counts from time zero.
    /// The picture at the cut's end is the last included microsecond.
    ///
    /// A segment covers `[start, end)`. Playback stops on that end, which would
    /// sample past the clip. A time past the end stays past it.
    fn sample_time(&self) -> Micros {
        let time = self.playhead.0.max(0);
        let end = self.cut_duration().0;
        if end > 0 && time == end {
            Micros(end - 1)
        } else {
            Micros(time)
        }
    }

    fn time_zero_x(&self, x_in_view: f64) -> f64 {
        let shift = self.view_origin.max(0) as f64 * self.pixel_size;
        if shift.is_finite() {
            x_in_view + shift
        } else {
            x_in_view
        }
    }

    fn visible_micros(&self) -> Option<i64> {
        if !self.pixel_size.is_finite()
            || self.pixel_size <= 0.0
            || !self.lane_width.is_finite()
            || self.lane_width <= 0.0
        {
            return None;
        }
        let span = self.lane_width / self.pixel_size;
        if !span.is_finite() || span <= 1.0 || span >= i64::MAX as f64 {
            return None;
        }
        Some(span.round() as i64)
    }

    /// Keep the playhead inside the lanes. A cut that already fits stays at the start.
    fn reveal_playhead(&mut self) {
        let Some(visible) = self.visible_micros() else {
            return;
        };
        let end = self.cut_duration().0.max(0);
        if end <= visible {
            self.view_origin = 0;
            return;
        }
        let lead = ((visible as f64) * 0.15).round() as i64;
        let lead = lead.clamp(0, visible.saturating_sub(1));
        let play = self.playhead.0.max(0);
        let mut origin = self.view_origin.max(0);
        let right = origin.saturating_add(visible).saturating_sub(lead);
        if play >= right {
            origin = play.saturating_sub(visible.saturating_sub(lead));
        } else if play < origin {
            origin = play;
        }
        let max_origin = end.saturating_sub(visible).max(0);
        if origin > max_origin {
            origin = max_origin;
        }
        self.view_origin = origin.max(0);
    }

    /// Keep the timeline time under a lane pixel while the pixel size changes.
    fn hold_under_pointer(&mut self, x_in_view: f64, old_pixel_size: f64) {
        if !old_pixel_size.is_finite() || old_pixel_size <= 0.0 {
            return;
        }
        let time = self.view_origin.max(0) as f64 + x_in_view / old_pixel_size;
        if !self.pixel_size.is_finite() || self.pixel_size <= 0.0 || !time.is_finite() {
            return;
        }
        let mut origin = (time - x_in_view / self.pixel_size).round() as i64;
        if origin < 0 {
            origin = 0;
        }
        let Some(visible) = self.visible_micros() else {
            self.view_origin = origin;
            return;
        };
        let end = self.cut_duration().0.max(0);
        let max_origin = end.saturating_sub(visible).max(0);
        self.view_origin = origin.clamp(0, max_origin);
    }

    /// Scroll while a drag sits against the side of the lanes.
    ///
    /// `x` is pixels from the left of the lanes. The left side shows earlier
    /// time. A cut that already fits does not move.
    pub fn drag_scroll(&mut self, x_in_view: f64) -> bool {
        const EDGE: f64 = 36.0;
        let width = self.lane_width;
        if !x_in_view.is_finite() || !width.is_finite() || width <= EDGE * 2.0 {
            return false;
        }
        let (sign, depth) = if x_in_view < EDGE {
            (-1.0, ((EDGE - x_in_view) / EDGE).clamp(0.0, 1.0))
        } else if x_in_view > width - EDGE {
            (1.0, ((x_in_view - (width - EDGE)) / EDGE).clamp(0.0, 1.0))
        } else {
            return false;
        };
        let Some(visible) = self.visible_micros() else {
            return false;
        };
        let step = ((visible as f64) * 0.08 * depth).round() as i64;
        let delta = if sign < 0.0 {
            -step.max(1)
        } else {
            step.max(1)
        };
        self.pan_micros(delta)
    }

    /// Pan the lanes. Positive pixels show later time.
    pub fn pan_pixels(&mut self, delta_px: f64) -> bool {
        if !delta_px.is_finite() || !self.pixel_size.is_finite() || self.pixel_size <= 0.0 {
            return false;
        }
        let micros = (delta_px / self.pixel_size).round() as i64;
        if micros == 0 {
            return false;
        }
        self.pan_micros(micros)
    }

    fn pan_micros(&mut self, delta: i64) -> bool {
        let Some(visible) = self.visible_micros() else {
            return false;
        };
        let end = self.cut_duration().0.max(0);
        let max_origin = end.saturating_sub(visible).max(0);
        let next = self.view_origin.saturating_add(delta).clamp(0, max_origin);
        if next == self.view_origin {
            return false;
        }
        self.view_origin = next;
        true
    }

    fn attribute_on(&self, track_id: &TrackId, flag: i32) -> bool {
        self.draft()
            .and_then(|draft| draft.tracks.iter().find(|track| &track.id == track_id))
            .is_some_and(|track| bit(track.attribute, flag))
    }

    fn refuse_locked(&self, segment_id: &SegmentId) -> Result<(), ShellError> {
        let draft = self.session_draft()?;
        let locked = locate(draft, segment_id).is_some_and(|(_, locked)| locked);
        if locked {
            Err(ShellError::Locked)
        } else {
            Ok(())
        }
    }

    fn track_of(&self, segment_id: &SegmentId) -> Result<TrackId, ShellError> {
        let draft = self.session_draft()?;
        draft
            .tracks
            .iter()
            .find(|track| {
                track
                    .segments
                    .iter()
                    .any(|segment| &segment.id == segment_id)
            })
            .map(|track| track.id.clone())
            .ok_or(ShellError::Unavailable(
                "the segment is not on a lane".to_owned(),
            ))
    }

    /// The size export writes the canvas at, from the shared export rule.
    fn scaled_canvas(&self, short_side: i32) -> Result<(u32, u32), ShellError> {
        let (width, height) = self
            .draft()
            .map(|draft| (draft.canvas_config.width, draft.canvas_config.height))
            .unwrap_or((1920, 1080));
        let side = u32::try_from(short_side).unwrap_or(0);
        cutlass_export::output_size(width, height, side).map_err(export_failure)
    }

    fn side_max(&self, window_width: f32, other: f32, floor: f32) -> f32 {
        let gaps = tokens::SPLIT * 2.0;
        (window_width - other - gaps - tokens::PREVIEW_MIN_WIDTH).max(floor)
    }

    fn stored_zoom(&self) -> Result<Zoom, ShellError> {
        Zoom::new(self.pixel_size).ok_or(ShellError::Zoom)
    }
}

/// What one export job writes.
struct ExportPlan {
    draft: Draft,
    /// The main file first, then a GIF beside it.
    outputs: Vec<ExportOutput>,
    /// A caption file written alone, when there is no movie to carry it.
    captions: Option<(cutlass_export::SidecarRequest, String)>,
    folder: PathBuf,
}

impl ExportPlan {
    fn paths(&self) -> Vec<PathBuf> {
        let mut paths: Vec<PathBuf> = self
            .outputs
            .iter()
            .map(|output| output.path().to_path_buf())
            .collect();
        let captions = self
            .outputs
            .iter()
            .find_map(|output| match output {
                ExportOutput::Movie { request, .. } => request.sidecar.as_ref(),
                _ => None,
            })
            .or(self.captions.as_ref().map(|(sidecar, _)| sidecar));
        if let Some(sidecar) = captions {
            let extension = match sidecar.kind {
                SidecarKind::Plain => "txt",
                _ => "srt",
            };
            paths.push(self.folder.join(format!("{}.{extension}", sidecar.name)));
        }
        paths
    }
}

enum ExportOutput {
    Movie {
        path: PathBuf,
        request: cutlass_export::ExportRequest,
    },
    Sound {
        path: PathBuf,
        request: cutlass_export::ExportRequest,
    },
    Gif {
        path: PathBuf,
        looped: bool,
        request: cutlass_export::ExportRequest,
    },
}

impl ExportOutput {
    fn path(&self) -> &Path {
        match self {
            Self::Movie { path, .. } | Self::Sound { path, .. } | Self::Gif { path, .. } => path,
        }
    }
}

/// A `.mov` name writes a QuickTime movie.
fn is_mov(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("mov"))
}

fn playhead_frame(rate: FrameRate, playhead: Micros) -> Result<TimeRange, ShellError> {
    let index = rate
        .frame_index(Micros(playhead.0.max(0)))
        .ok_or_else(|| ShellError::Unavailable("the playhead is not a frame".to_owned()))?;
    let start = rate
        .frame_start(index)
        .ok_or_else(|| ShellError::Unavailable("the playhead is not a frame".to_owned()))?;
    let next = index
        .checked_add(1)
        .ok_or_else(|| ShellError::Unavailable("the playhead is not a frame".to_owned()))?;
    let end = rate
        .frame_start(next)
        .ok_or_else(|| ShellError::Unavailable("the playhead is not a frame".to_owned()))?;
    let duration = end
        .0
        .checked_sub(start.0)
        .filter(|span| *span > 0)
        .ok_or_else(|| ShellError::Unavailable("the playhead is not a frame".to_owned()))?;
    Ok(TimeRange::new(start.0, duration))
}

fn export_failure(error: cutlass_export::ExportError) -> ShellError {
    match error {
        cutlass_export::ExportError::Encoder(error) => ShellError::CodecDeclined(error.to_string()),
        error => ShellError::Eval(error.to_string()),
    }
}

fn write_replaced(path: &Path, bytes: &[u8]) -> Result<(), ShellError> {
    if path.is_dir() {
        return Err(ShellError::Io("the export path is a folder".to_owned()));
    }
    let parent = match path.parent() {
        Some(parent) if parent.as_os_str().is_empty() => Path::new("."),
        Some(parent) => parent,
        None => {
            return Err(ShellError::Io("the export path has no folder".to_owned()));
        }
    };
    if !parent.is_dir() {
        return Err(ShellError::MissingFolder(parent.to_path_buf()));
    }
    let file_name = path
        .file_name()
        .ok_or_else(|| ShellError::Io("the export path has no name".to_owned()))?;
    let mut temp_name = std::ffi::OsString::from(".");
    temp_name.push(file_name);
    temp_name.push(".part");
    let temp = parent.join(temp_name);
    if let Err(error) = std::fs::write(&temp, bytes) {
        return Err(ShellError::Io(error.to_string()));
    }
    if let Err(error) = std::fs::rename(&temp, path) {
        let _ = std::fs::remove_file(&temp);
        return Err(ShellError::Io(error.to_string()));
    }
    Ok(())
}

fn opening_pixel_size(rate: FrameRate) -> f64 {
    last_accuracy_step(rate).unwrap_or(0.006)
}

/// How much time an empty timeline shows across a third of the lanes.
pub const EMPTY_SPAN: Micros = Micros(10_000_000);

/// [`EMPTY_SPAN`] across a third of `lane_width`, no closer than the
/// finest ruler step.
fn empty_pixel_size(lane_width: f64, rate: FrameRate) -> Option<f64> {
    let size = lane_width / 3.0 / EMPTY_SPAN.0 as f64;
    if !size.is_finite() || size <= 0.0 {
        return None;
    }
    Some(size.min(last_accuracy_step(rate)?))
}

fn bit(attribute: TrackAttribute, flag: i32) -> bool {
    attribute.0 & flag != 0
}

fn heard_error(miss: HeardMiss) -> ShellError {
    ShellError::Unavailable(
        match miss {
            HeardMiss::UnknownSegment => "the clip is not on the timeline",
            HeardMiss::NotHeard => "a transcript belongs on a picture or a sound",
            HeardMiss::NoSound => "the clip has no sound",
            HeardMiss::NothingHeard => "nothing was heard",
            HeardMiss::BadClock => "a word clock does not fit the clip",
        }
        .to_owned(),
    )
}

fn clipboard_error(miss: ClipboardMiss) -> ShellError {
    match miss {
        ClipboardMiss::Empty => ShellError::Unavailable("nothing to paste".to_owned()),
        ClipboardMiss::Locked => ShellError::Locked,
        ClipboardMiss::LaneGone => ShellError::Unavailable("the lane is gone".to_owned()),
    }
}

fn draft_has_linkage(draft: &Draft) -> bool {
    draft
        .relationships
        .iter()
        .any(|item| item.kind == "linkage" && !item.id_to_id.is_empty())
}

/// Twelve pixels, in microseconds. A bad pixel size snaps to nothing.
fn snap_window(pixel_size: f64) -> Micros {
    const PIXELS: f64 = 12.0;
    if !pixel_size.is_finite() || pixel_size <= 0.0 {
        return Micros(0);
    }
    let micros = PIXELS / pixel_size;
    if !micros.is_finite() || micros <= 0.0 {
        return Micros(0);
    }
    if micros >= i64::MAX as f64 {
        return Micros(i64::MAX);
    }
    Micros(micros.round() as i64)
}

fn trim_is_current(draft: &Draft, plan: &TrimPlan) -> bool {
    let Some((segment, _)) = locate(draft, &plan.segment_id) else {
        return false;
    };
    match plan.edge {
        TrimEdge::Start => segment.target.start == plan.time,
        TrimEdge::End => segment.target.end() == plan.time,
    }
}

fn scene_message(error: SceneError) -> ShellError {
    let message = match error {
        SceneError::Missing | SceneError::NoPicture => "the clip has no picture",
        SceneError::Decode => "the clip's picture could not be read",
    };
    ShellError::Unavailable(message.to_owned())
}

/// What the button that opens a file's folder says on this system.
#[cfg(target_os = "macos")]
pub const REVEAL_LABEL: &str = "Show in Finder";
#[cfg(target_os = "windows")]
pub const REVEAL_LABEL: &str = "Show in Explorer";
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub const REVEAL_LABEL: &str = "Show in folder";

/// Open `path`'s folder in the system's file manager, with the file picked
/// where the system can do that.
pub fn reveal(path: &Path) -> Result<(), ShellError> {
    let mut command = if cfg!(target_os = "macos") {
        let mut open = std::process::Command::new("open");
        open.arg("-R").arg(path);
        open
    } else if cfg!(target_os = "windows") {
        let mut explorer = std::process::Command::new("explorer");
        let mut select = std::ffi::OsString::from("/select,");
        select.push(path);
        explorer.arg(select);
        explorer
    } else {
        let mut open = std::process::Command::new("xdg-open");
        open.arg(path.parent().unwrap_or(path));
        open
    };
    command
        .spawn()
        .map(drop)
        .map_err(|error| ShellError::Io(format!("the folder could not be shown: {error}")))
}

/// A job in flight, the project it started in, and whether it was asked
/// to stop.
struct RunningJob {
    job: crate::jobs::Job<Landing>,
    project: Option<PathBuf>,
    stopping: bool,
}

/// What a finished job does on the window thread.
pub(crate) type Landing = Box<dyn FnOnce(&mut Shell) -> Result<(), ShellError> + Send>;

/// The speech model's file inside [`speech_model_dir`].
const SPEECH_MODEL: &str = "parakeet-unified-en-0.6b.nemo";

/// Load the speech model; slow, so a job does it.
fn load_speech() -> Result<ParakeetUnified, ShellError> {
    ParakeetUnified::cpu(&speech_model_dir())
        .map_err(|_| ShellError::Unavailable("the speech model could not be loaded".to_owned()))
}

fn speech_model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../local-files/parakeet-unified-en-0.6b")
}

fn heard_words(
    recognizer: &dyn Recognizer,
    source: &mut dyn PcmSource,
    clip: &HeardClip,
) -> Result<WordColumns, ShellError> {
    let pcm = source.read(&clip.range).map_err(prepare_message)?;
    let pcm = fit(pcm, recognizer.sample_rate()).map_err(prepare_message)?;
    let utterances = recognizer.recognize(&pcm).map_err(recognize_message)?;
    let words = columns(&utterances)
        .map_err(|_| ShellError::Unavailable("a word clock does not fit the clip".to_owned()))?;
    words_on_timeline(words, clip).map_err(heard_error)
}

fn prepare_message(error: PrepareError) -> ShellError {
    let message = match error {
        PrepareError::Open => "the clip has no sound to hear",
        PrepareError::Decode => "the clip's sound could not be read",
        PrepareError::Range
        | PrepareError::SampleRate
        | PrepareError::Samples
        | PrepareError::TooLong
        | PrepareError::NoAudio => "the clip has no sound",
    };
    ShellError::Unavailable(message.to_owned())
}

fn recognize_message(error: RecognizeError) -> ShellError {
    let message = match error {
        RecognizeError::Model | RecognizeError::Search => {
            "the speech model could not hear the clip"
        }
        RecognizeError::SampleRate { .. } | RecognizeError::WordTime => "the clip has no sound",
    };
    ShellError::Unavailable(message.to_owned())
}

fn placed<'a>(draft: &'a Draft, id: &SegmentId) -> Option<(&'a Segment, &'a TrackType)> {
    for track in &draft.tracks {
        if let Some(segment) = track.segments.iter().find(|segment| &segment.id == id) {
            return Some((segment, &track.kind));
        }
    }
    None
}

/// A video, a text, a sticker, or a composition can flip or turn. A sound cannot.
/// The scale the preview uses. Uniform scale wins while that switch is on.
fn picture_scale(segment: &Segment) -> (f64, f64) {
    if segment.uniform_scale.on {
        (segment.uniform_scale.value, segment.uniform_scale.value)
    } else {
        (segment.clip.scale.x, segment.clip.scale.y)
    }
}

fn text_style(
    draft: &Draft,
    material_id: &MaterialId,
) -> (f64, String, i32, f64, f64, f64, String) {
    draft
        .materials
        .texts
        .iter()
        .find(|text| text.id == *material_id)
        .map(|text| {
            (
                text.font_size,
                text.text_color.clone(),
                text.alignment,
                text.letter_spacing,
                text.line_spacing,
                text.line_max_width,
                text.font_path.clone(),
            )
        })
        .unwrap_or((0.0, String::new(), 0, 0.0, 0.0, 0.0, String::new()))
}

fn text_font(draft: &Draft, segment_id: &SegmentId) -> String {
    placed(draft, segment_id)
        .and_then(|(segment, _)| {
            draft
                .materials
                .texts
                .iter()
                .find(|text| text.id == segment.material_id)
        })
        .map(|text| text.font_path.clone())
        .unwrap_or_default()
}

fn next_face(names: &[String], current: &str) -> Option<String> {
    if names.is_empty() {
        return None;
    }
    let index = names
        .iter()
        .position(|name| name == current)
        .map(|index| index.saturating_add(1))
        .unwrap_or(0);
    Some(names[index % names.len()].clone())
}

/// The speed curve `segment` runs on: its curve material's points, when it
/// has two or more.
fn speed_curve(draft: &Draft, segment: &Segment) -> Option<Vec<cutlass_model::SpeedPoint>> {
    segment.extra_material_refs.iter().find_map(|material_id| {
        let speed = draft
            .materials
            .speeds
            .iter()
            .find(|speed| &speed.id == material_id)?;
        let points = &speed.curve_speed.as_ref()?.speed_points;
        (speed.mode == cutlass_model::SpeedMode::Curve && points.len() >= 2).then(|| points.clone())
    })
}

fn letterbox_color(draft: &Draft, segment: &Segment) -> String {
    segment
        .extra_material_refs
        .iter()
        .find_map(|id| match draft.materials.get(id.as_str()) {
            Some(MaterialRef::Canvas(canvas)) => Some(canvas.color.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

fn fade_durations(draft: &Draft, segment: &Segment) -> (i64, i64) {
    for id in &segment.extra_material_refs {
        if let Some(MaterialRef::Fade(fade)) = draft.materials.get(id.as_str()) {
            return (fade.fade_in_duration.0, fade.fade_out_duration.0);
        }
    }
    (0, 0)
}

fn fade_micros(seconds: f64) -> Result<Micros, ShellError> {
    if !seconds.is_finite() || seconds < 0.0 {
        return Err(ShellError::Unavailable(
            "a fade cannot be negative".to_owned(),
        ));
    }
    let microseconds = (seconds * 1_000_000.0).round();
    if microseconds > i64::MAX as f64 {
        return Err(ShellError::Unavailable("a fade is too long".to_owned()));
    }
    Ok(Micros(microseconds as i64))
}

/// One clockwise quarter turn. The preview's positive angle is the other way.
fn stored_crop(draft: &Draft, segment: &Segment) -> Option<Crop> {
    match draft.materials.get(segment.material_id.as_str()) {
        Some(MaterialRef::Video(video)) => Some(video.crop),
        Some(MaterialRef::Draft(compound)) => Some(compound.crop),
        _ => None,
    }
}

/// The middle half of the picture. Coordinates are fractions, Y downward.
fn center_crop() -> Crop {
    Crop {
        upper_left_x: 0.25,
        upper_left_y: 0.25,
        upper_right_x: 0.75,
        upper_right_y: 0.25,
        lower_left_x: 0.25,
        lower_left_y: 0.75,
        lower_right_x: 0.75,
        lower_right_y: 0.75,
    }
}

fn clockwise_quarter(degrees: f64) -> f64 {
    let current = if degrees.is_finite() { degrees } else { 0.0 };
    (current - 90.0).rem_euclid(360.0)
}

fn locate<'a>(draft: &'a Draft, id: &SegmentId) -> Option<(&'a Segment, bool)> {
    for track in &draft.tracks {
        if let Some(segment) = track.segments.iter().find(|segment| &segment.id == id) {
            return Some((segment, bit(track.attribute, TrackAttribute::LOCK)));
        }
    }
    None
}

fn clip_name(draft: &Draft, segment: &Segment) -> String {
    match draft.materials.get(segment.material_id.as_str()) {
        Some(MaterialRef::Video(video)) if !video.path.is_empty() => file_name(&video.path),
        Some(MaterialRef::Audio(audio)) if !audio.path.is_empty() => file_name(&audio.path),
        Some(MaterialRef::Text(text)) if !text.content.is_empty() => text.content.clone(),
        Some(MaterialRef::Draft(compound)) if !compound.name.is_empty() => compound.name.clone(),
        _ => segment.id.as_str().to_owned(),
    }
}

fn material_label(draft: &Draft, id: &MaterialId) -> (String, Option<String>, WordColumns) {
    match draft.materials.get(id.as_str()) {
        Some(MaterialRef::Video(video)) => (video.path.clone(), None, Default::default()),
        Some(MaterialRef::Audio(audio)) => (audio.path.clone(), None, Default::default()),
        Some(MaterialRef::Text(text)) => (
            text.content.clone(),
            Some(text.content.clone()),
            text.words.clone(),
        ),
        Some(MaterialRef::Color(color)) => (color.solid_color.clone(), None, Default::default()),
        _ => (id.as_str().to_owned(), None, Default::default()),
    }
}

fn file_name(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_owned())
}

fn format_timecode(label: TimecodeLabel) -> String {
    let separator = if label.drop_frame { ';' } else { ':' };
    format!(
        "{:02}:{:02}:{:02}{separator}{:02}",
        label.hours, label.minutes, label.seconds, label.frames
    )
}

/// Which request an edit was, and the clip selected for it.
type HeldEdit = (std::mem::Discriminant<Request>, Option<SegmentId>);

/// An autosave in flight and the revision it writes.
struct Saving {
    revision: u64,
    /// The library edit count the write holds.
    library: u64,
    /// The project it writes; a different one may be open when it lands.
    project: PathBuf,
    handle: std::thread::JoinHandle<Result<(), StoreError>>,
}

/// Title sizes from the installed faces, for the shared stage rules.
struct FaceSizes<'a> {
    draft: &'a Draft,
    fonts: &'a dyn cutlass_eval::Fonts,
}

impl cutlass_editor::TextSizes for FaceSizes<'_> {
    fn text_size(&self, segment: &Segment) -> Option<(f64, f64)> {
        let (width, height) = cutlass_eval::text_size(self.draft, self.fonts, segment)?;
        Some((f64::from(width), f64::from(height)))
    }
}

/// A file copied into the project for an import, removed unless the import
/// keeps it.
struct CopyGuard(Option<PathBuf>);

impl CopyGuard {
    fn keep(&mut self) {
        self.0 = None;
    }
}

impl Drop for CopyGuard {
    fn drop(&mut self) {
        if let Some(copied) = self.0.take() {
            let _ = std::fs::remove_file(copied);
        }
    }
}

/// What an import added to the library.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportedKind {
    Video,
    Audio,
}

/// Copy `source` to `destination` a piece at a time, telling `progress`
/// how far it is and stopping when asked. A stopped copy is removed.
fn copy_reporting(
    source: &Path,
    destination: &Path,
    progress: &crate::jobs::Progress,
) -> Result<(), ShellError> {
    use std::io::{Read, Write};
    const PIECE: usize = 4 << 20;
    let io = |error: std::io::Error| ShellError::Io(error.to_string());
    let copied = (|| {
        let mut from = std::fs::File::open(source).map_err(io)?;
        let total = from.metadata().map_err(io)?.len().max(1);
        let mut to = std::fs::File::create(destination).map_err(io)?;
        let mut piece = vec![0; PIECE];
        let mut done = 0_u64;
        loop {
            if progress.cancelled() {
                return Err(ShellError::Unavailable("the import was stopped".to_owned()));
            }
            let read = from.read(&mut piece).map_err(io)?;
            let Some(bytes) = piece.get(..read).filter(|bytes| !bytes.is_empty()) else {
                break;
            };
            to.write_all(bytes).map_err(io)?;
            done += read as u64;
            progress.fraction(done as f32 / total as f32);
        }
        to.sync_all().map_err(io)
    })();
    if copied.is_err() {
        let _ = std::fs::remove_file(destination);
    }
    copied
}

fn place_in_project(
    project: &Path,
    source: &Path,
    progress: Option<&crate::jobs::Progress>,
) -> Result<(String, Option<PathBuf>), ShellError> {
    let project = project
        .canonicalize()
        .map_err(|error| ShellError::Io(error.to_string()))?;
    let source_path = source
        .canonicalize()
        .map_err(|error| ShellError::Io(error.to_string()))?;
    if source_path.starts_with(&project) {
        let relative = source_path
            .strip_prefix(&project)
            .map_err(|error| ShellError::Io(error.to_string()))?;
        return Ok((portable(relative), None));
    }
    let resources = project.join("Resources");
    std::fs::create_dir_all(&resources).map_err(|error| ShellError::Io(error.to_string()))?;
    let name = source
        .file_name()
        .ok_or_else(|| ShellError::Io("the file has no name".to_owned()))?;
    let destination = unique_file(&resources, name)?;
    match progress {
        Some(progress) => copy_reporting(source, &destination, progress)?,
        None => {
            std::fs::copy(source, &destination)
                .map_err(|error| ShellError::Io(error.to_string()))?;
        }
    }
    let stored = destination
        .file_name()
        .map(|file| format!("Resources/{}", file.to_string_lossy()))
        .ok_or_else(|| ShellError::Io("the copy has no name".to_owned()))?;
    Ok((stored, Some(destination)))
}

fn portable(path: &Path) -> String {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

fn unique_file(directory: &Path, name: &std::ffi::OsStr) -> Result<PathBuf, ShellError> {
    let candidate = directory.join(name);
    if !candidate.exists() {
        return Ok(candidate);
    }
    let parsed = Path::new(name);
    let stem = parsed
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| "media".to_owned());
    let extension = parsed
        .extension()
        .map(|ext| ext.to_string_lossy().into_owned());
    for index in 2..10_000 {
        let file_name = match &extension {
            Some(extension) => format!("{stem}-{index}.{extension}"),
            None => format!("{stem}-{index}"),
        };
        let path = directory.join(file_name);
        if !path.exists() {
            return Ok(path);
        }
    }
    Err(ShellError::Io("the project resources are full".to_owned()))
}

/// `peaks` spread over `columns`: the loudest level in each column's share
/// of `duration`, the file's length. Past what was read is silence.
/// A library card's wave and what it was spread from.
struct CardWave {
    peaks: Arc<cutlass_engine::Peaks>,
    duration: i64,
    columns: usize,
    levels: Arc<[f32]>,
}

#[cfg(test)]
mod tests {
    use super::{ExportRequest, SHORT_SIDES, Shell, ShellError};
    use cutlass_codec::{
        CodecId, ContainerId, FrameSize, Packet as FilePacket, StreamCodec, StreamParams,
        Timescale, Timestamp,
    };
    use cutlass_container::{
        Demux, Gap, Landing, OpenError, ProbeResult, ReadError, SeekLanding, SeekTo, Stream,
        TrackId, Trim,
    };
    use cutlass_decode::{
        CodecError, CodecParams, ColorPrimaries, Decoder, Output, Packet, PictureInfo,
        PictureOutput, PixelFormat, Transfer,
    };
    use cutlass_eval::ReaderFactory;
    use cutlass_export::{ProresProfile, SidecarKind};
    use cutlass_model::{
        Crop, Draft, FrameRate, FreezeInfo, MaterialAudio, MaterialColorConfig, MaterialDraft,
        MaterialId, MaterialText, MaterialTransition, MaterialVideo, Micros, RawMaterial, Segment,
        SegmentId, TimeRange, Track, TrackAttribute, TrackType,
    };
    use cutlass_playback::AudioHost;
    use cutlass_session::{EditError, TrimEdge};
    use cutlass_settings::{SeekDuringPlayback, Settings};
    use cutlass_store::Folder;
    use cutlass_transcription::{
        Pcm, PcmSource, PrepareError, RecognizeError, RecognizedWord, Recognizer, Utterance,
    };
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

    static MOVIE_OPENS: AtomicU32 = AtomicU32::new(0);
    static KEPT_OPENS: AtomicU32 = AtomicU32::new(0);

    fn probe_counted(header: &[u8]) -> ProbeResult {
        if header.len() < 8 {
            return ProbeResult::NeedBytes {
                have: header.len(),
                need: 8,
            };
        }
        ProbeResult::Accept {
            family: ContainerId::Isobmff,
            examined: 8,
        }
    }

    fn open_counted(_path: &Path) -> Result<Box<dyn Demux>, OpenError> {
        MOVIE_OPENS.fetch_add(1, Ordering::Relaxed);
        Err(OpenError::Source {
            message: "counted".to_owned(),
        })
    }

    struct Kept {
        streams: Vec<Stream>,
        packet: FilePacket,
        cursor: usize,
        cancelled: bool,
    }

    fn kept_demux() -> Kept {
        let scale = Timescale::new(30);
        let size = FrameSize::try_new(2, 2).expect("size");
        let timestamp = Timestamp::try_new(Some(0), Some(0), Some(1)).expect("time");
        let packet =
            FilePacket::try_new(1, b"unit".to_vec(), timestamp, scale, true).expect("packet");
        Kept {
            streams: vec![Stream::new(
                TrackId::new(1),
                StreamParams::video(StreamCodec::Known(CodecId::Jpeg), size, Vec::new()),
                scale,
                None,
                Vec::new(),
                Trim::default(),
            )],
            packet,
            cursor: 0,
            cancelled: false,
        }
    }

    impl Demux for Kept {
        fn family(&self) -> ContainerId {
            ContainerId::Isobmff
        }

        fn streams(&self) -> &[Stream] {
            &self.streams
        }

        fn gaps(&self) -> &[Gap] {
            &[]
        }

        fn read_packet(&mut self) -> Result<Option<FilePacket>, ReadError> {
            if self.cancelled {
                return Err(ReadError::Cancelled);
            }
            if self.cursor > 0 {
                return Ok(None);
            }
            self.cursor = 1;
            Ok(Some(self.packet.clone()))
        }

        fn seek(&mut self, _to: SeekTo) -> Result<SeekLanding, ReadError> {
            if self.cancelled {
                return Err(ReadError::Cancelled);
            }
            self.cursor = 0;
            Ok(SeekLanding {
                landing: Landing::Packet {
                    ordinal: 0,
                    time: 0,
                    keyframe: true,
                    preroll: 0,
                    byte_position: 0,
                },
                index_used: false,
                payload_scans: 0,
                entries_visited: 0,
            })
        }

        fn cancel(&mut self) {
            self.cancelled = true;
        }

        fn consumed(&self) -> u64 {
            0
        }
    }

    fn open_kept(_path: &Path) -> Result<Box<dyn Demux>, OpenError> {
        KEPT_OPENS.fetch_add(1, Ordering::Relaxed);
        Ok(Box::new(kept_demux()))
    }

    struct Flat;

    impl Decoder for Flat {
        fn codec(&self) -> CodecId {
            CodecId::Jpeg
        }

        fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
            let mut bytes = vec![16; 4];
            bytes.push(128);
            bytes.push(128);
            let picture = PictureOutput::cpu(
                PictureInfo {
                    width: 2,
                    height: 2,
                    pixel: PixelFormat::Yuv420p,
                    primaries: ColorPrimaries::Bt709,
                    transfer: Transfer::Bt709,
                },
                bytes,
                packet.pts,
                packet.timescale,
            )?;
            Ok(vec![Output::Picture(picture)])
        }

        fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
            Ok(Vec::new())
        }

        fn reset(&mut self) {}
    }

    fn open_flat(_params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
        Ok(Box::new(Flat))
    }

    fn digest(bytes: &[u8]) -> u64 {
        let mut hash = 0xcbf29ce484222325_u64;
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    pub(super) fn shell() -> Shell {
        let (decoders, encoders) = crate::register_codecs();
        Shell::new(decoders, encoders, Settings::default())
    }

    /// Four seconds of a 3840x2160 60 fps BT.2020 HLG movie, repainted at
    /// about 120 Hz as the window does: no repaint waits on a late picture.
    /// Run with `--release --ignored --nocapture` for the timing.
    #[test]
    #[ignore = "plays a 4K60 HDR movie in real time"]
    fn a_4k60_hdr_movie_plays_without_late_pictures() {
        let movie = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../local-files/15215997_3840_2160_60fps.mp4");
        if !movie.is_file() {
            eprintln!("skipped: the 4K60 HDR movie is missing");
            return;
        }
        let instance = eframe::wgpu::Instance::new(
            eframe::wgpu::InstanceDescriptor::new_without_display_handle(),
        );
        let Ok(state) = pollster::block_on(eframe::egui_wgpu::RenderState::create(
            &eframe::egui_wgpu::WgpuConfiguration::default(),
            &instance,
            None,
            eframe::egui_wgpu::RendererOptions::default(),
        )) else {
            eprintln!("skipped: no GPU");
            return;
        };
        let temp = Temp::new();
        let mut draft = crate::open_movie::movie_draft(&movie, 1920, 1080, false).unwrap();
        draft.tracks[0].segments[0].target = TimeRange::new(0, 20_000_000);
        draft.tracks[0].segments[0].source_range = Some(TimeRange::new(0, 20_000_000));
        draft.materials.videos[0].duration = Micros(20_000_000);
        draft.duration = Micros(20_000_000);
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.attach_gpu(&state);
        desktop.preview_texture().unwrap().expect("first frame");
        desktop.play();
        let start = std::time::Instant::now();
        let mut last = start;
        let (mut paints, mut stale, mut busy) = (0_u32, 0_u32, std::time::Duration::ZERO);
        let mut worst = std::time::Duration::ZERO;
        while start.elapsed() < std::time::Duration::from_secs(4) {
            let now = std::time::Instant::now();
            desktop.advance(i64::try_from((now - last).as_micros()).unwrap());
            last = now;
            let t = std::time::Instant::now();
            desktop.preview_texture().unwrap();
            let spent = t.elapsed();
            busy += spent;
            worst = worst.max(spent);
            paints += 1;
            if desktop.preview_pending() {
                stale += 1;
            }
            std::thread::sleep(std::time::Duration::from_micros(8_333).saturating_sub(spent));
        }
        eprintln!(
            "playhead {} ms after 4000 ms, {paints} paints, {stale} late, preview {:.2} ms a paint, worst {:.2} ms",
            desktop.playhead.0 / 1000,
            busy.as_secs_f64() * 1000.0 / f64::from(paints),
            worst.as_secs_f64() * 1000.0,
        );
        assert_eq!(stale, 0, "{stale} of {paints} paints showed a late picture");
    }

    #[test]
    fn a_far_jump_shows_the_last_picture_until_the_asked_one_decodes() {
        let movie = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../local-files/12471039-uhd_3840_2160_24fps.mp4");
        if !movie.is_file() {
            eprintln!("skipped: the 4K movie is missing");
            return;
        }
        let instance = eframe::wgpu::Instance::new(
            eframe::wgpu::InstanceDescriptor::new_without_display_handle(),
        );
        let Ok(state) = pollster::block_on(eframe::egui_wgpu::RenderState::create(
            &eframe::egui_wgpu::WgpuConfiguration::default(),
            &instance,
            None,
            eframe::egui_wgpu::RendererOptions::default(),
        )) else {
            eprintln!("skipped: no GPU");
            return;
        };
        let temp = Temp::new();
        let mut draft = crate::open_movie::movie_draft(&movie, 3840, 2160, false).unwrap();
        draft.tracks[0].segments[0].target = TimeRange::new(0, 9_000_000);
        draft.tracks[0].segments[0].source_range = Some(TimeRange::new(0, 9_000_000));
        draft.materials.videos[0].duration = Micros(9_000_000);
        draft.duration = Micros(9_000_000);
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.attach_gpu(&state);
        desktop.preview_texture().unwrap().expect("first frame");
        assert!(!desktop.preview_pending(), "the first frame waits");

        desktop.seek(Micros(8_200_000));
        desktop.preview_texture().unwrap().expect("held frame");
        assert!(desktop.preview_pending(), "a far jump shows the held frame");
        let mut paints = 0;
        while desktop.preview_pending() {
            assert!(paints < 2_000, "the asked frame never arrived");
            std::thread::sleep(std::time::Duration::from_millis(2));
            desktop.preview_texture().unwrap().expect("frame");
            paints += 1;
        }
        let gpu = desktop
            .gpu
            .as_ref()
            .and_then(crate::gpu::Preview::read)
            .unwrap();
        let cpu = desktop.preview_bytes().unwrap().rgba;
        let off = gpu
            .chunks(4)
            .zip(cpu.chunks(4))
            .filter(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| x.abs_diff(*y) > 2))
            .count();
        assert!(off * 1000 <= gpu.len() / 4, "{off} pixels differ");
    }

    #[test]
    fn playback_waits_for_the_sound_then_follows_it_and_hears_an_edit() {
        let movie = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../local-files/Expo - You Don't Need a Mac to Build iOS Apps Anymore.mp4");
        if !movie.is_file() {
            eprintln!("skipped: the movie with sound is missing");
            return;
        }
        let temp = Temp::new();
        let mut draft = crate::open_movie::movie_draft(&movie, 1920, 1080, false).unwrap();
        draft.tracks[0].segments[0].target = TimeRange::new(0, 8_000_000);
        draft.tracks[0].segments[0].source_range = Some(TimeRange::new(0, 8_000_000));
        draft.materials.videos[0].duration = Micros(8_000_000);
        draft.duration = Micros(8_000_000);
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.open_preview_audio(AudioHost::current());
        if desktop.transport.output().clock().is_none() {
            eprintln!("skipped: no audio device");
            return;
        }
        desktop.seek(Micros(1_000_000));
        desktop.play();
        assert!(desktop.transport.is_streaming());
        // Each tick passes the time that really went by, as the window does.
        let mut last = std::time::Instant::now();
        let mut step = |desktop: &mut Shell| {
            std::thread::sleep(std::time::Duration::from_micros(16_667));
            let now = std::time::Instant::now();
            let elapsed = i64::try_from(now.duration_since(last).as_micros()).unwrap();
            last = now;
            desktop.advance(elapsed);
        };
        let mut ticks = 0;
        loop {
            assert!(ticks < 120, "the stream never started");
            step(&mut desktop);
            ticks += 1;
            if desktop.transport.heard_frames() != Some(0) {
                break;
            }
            // Nothing heard after this tick, so nothing was heard during it.
            assert_eq!(desktop.playhead, Micros(1_000_000), "the picture held");
        }
        for _ in 0..30 {
            step(&mut desktop);
        }
        let heard = desktop.transport.heard_time().unwrap();
        assert!(desktop.playhead.0 > 1_300_000, "playback moved");
        assert!(
            (desktop.playhead.0 - heard.0).abs() < 10_000,
            "picture {} audio {}",
            desktop.playhead.0,
            heard.0
        );
        let track = desktop.draft().unwrap().tracks[0].id.clone();
        desktop
            .toggle_attribute(track, TrackAttribute::MUTE)
            .unwrap();
        step(&mut desktop);
        let revision = desktop.session.as_ref().unwrap().revision();
        assert_eq!(desktop.transport.mix_revision(), Some(revision));
        desktop.pause();
        assert!(!desktop.transport.is_streaming());
    }

    #[test]
    fn preview_draws_a_real_movie_on_the_gpu_like_the_cpu() {
        let movie =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../local-files/IMG_8143.MP4");
        if !movie.is_file() {
            eprintln!("skipped: ../local-files/IMG_8143.MP4 is missing");
            return;
        }
        let instance = eframe::wgpu::Instance::new(
            eframe::wgpu::InstanceDescriptor::new_without_display_handle(),
        );
        let state = match pollster::block_on(eframe::egui_wgpu::RenderState::create(
            &eframe::egui_wgpu::WgpuConfiguration::default(),
            &instance,
            None,
            eframe::egui_wgpu::RendererOptions::default(),
        )) {
            Ok(state) => state,
            Err(error) => {
                eprintln!("skipped: {error}");
                return;
            }
        };
        let temp = Temp::new();
        let draft = crate::open_movie::movie_draft(&movie, 848, 608, true).unwrap();
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.attach_gpu(&state);
        desktop.seek(Micros(400_000));

        let (id, size) = desktop.preview_texture().unwrap().expect("gpu preview");
        assert_eq!(size, [848, 608]);
        assert!(state.renderer.read().texture(&id).is_some());
        let again = desktop.preview_texture().unwrap().expect("gpu preview");
        assert_eq!(again, (id, size));

        let gpu = desktop
            .gpu
            .as_ref()
            .and_then(crate::gpu::Preview::read)
            .unwrap();
        let cpu = desktop.preview_bytes().unwrap().rgba;
        let off = gpu
            .chunks(4)
            .zip(cpu.chunks(4))
            .filter(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| x.abs_diff(*y) > 2))
            .count();
        assert!(off * 1000 <= gpu.len() / 4, "{off} pixels differ");

        desktop.seek(Micros(433_334));
        let (next, _) = desktop.preview_texture().unwrap().expect("gpu preview");
        assert_eq!(next, id);
        assert_ne!(
            desktop
                .gpu
                .as_ref()
                .and_then(crate::gpu::Preview::read)
                .unwrap(),
            gpu
        );
    }

    pub(super) struct Temp {
        pub(super) path: PathBuf,
    }

    impl Temp {
        pub(super) fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "cutlass-desktop-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    /// Poll until the job lands, or ten seconds pass.
    fn land(desktop: &mut Shell) {
        let until = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while desktop.poll_job() && std::time::Instant::now() < until {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    #[test]
    fn a_job_lands_on_the_window_thread_and_one_runs_at_a_time() {
        let temp = Temp::new();
        empty_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let (go, gate) = std::sync::mpsc::channel::<()>();
        desktop
            .start_job("Wait", move |progress| {
                progress.stage("Waiting");
                let _ = gate.recv();
                Box::new(|shell: &mut Shell| {
                    shell.set_notice("landed");
                    Ok(())
                })
            })
            .unwrap();
        assert_eq!(desktop.job().map(|job| job.label), Some("Wait".to_owned()));
        assert!(desktop.poll_job(), "still waiting");
        assert_eq!(
            desktop
                .start_job("Second", |_| Box::new(|_: &mut Shell| Ok(())))
                .unwrap_err()
                .to_string(),
            "Wait is still running"
        );
        go.send(()).unwrap();
        land(&mut desktop);
        assert!(desktop.job().is_none());
        assert_eq!(desktop.notice(), "landed");
    }

    #[test]
    fn a_cancelled_job_or_one_from_a_closed_project_does_not_land() {
        let temp = Temp::new();
        empty_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let landing = || -> super::Landing {
            Box::new(|shell: &mut Shell| {
                shell.set_notice("landed");
                Ok(())
            })
        };
        desktop
            .start_job("Cancel me", move |progress| {
                while !progress.cancelled() {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                landing()
            })
            .unwrap();
        desktop.cancel_job();
        assert_eq!(
            desktop.job().map(|job| job.stage),
            Some("Stopping".to_owned())
        );
        land(&mut desktop);
        assert!(desktop.job().is_none());
        assert_ne!(desktop.notice(), "landed");

        let (go, gate) = std::sync::mpsc::channel::<()>();
        desktop
            .start_job("Late", move |_| {
                let _ = gate.recv();
                landing()
            })
            .unwrap();
        desktop.close_project().unwrap();
        go.send(()).unwrap();
        land(&mut desktop);
        assert_eq!(desktop.notice(), "Late finished after its project closed");
    }

    #[test]
    fn transcribe_runs_as_a_job_and_the_caption_lands_when_it_finishes() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
        let sound = root.join("Expo - You Don't Need a Mac to Build iOS Apps Anymore.m4a");
        if !sound.is_file()
            || !super::speech_model_dir()
                .join(super::SPEECH_MODEL)
                .is_file()
        {
            eprintln!("skipped: the sound or the speech model is missing");
            return;
        }
        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("talk"),
            path: sound.display().to_string(),
            duration: Micros(6_000_000),
            ..MaterialAudio::default()
        });
        let mut track = Track::main("sound");
        track.kind = TrackType::Audio;
        let mut segment = Segment::new("talk", "talk", TimeRange::new(0, 6_000_000));
        segment.source_range = Some(TimeRange::new(0, 6_000_000));
        track.segments.push(segment);
        draft.tracks.push(track);
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.select(Some(SegmentId::new("talk")));
        let revision = desktop.revision();
        desktop.transcribe().unwrap();
        assert_eq!(
            desktop.job().map(|job| job.label),
            Some("Transcribe".to_owned())
        );
        // The window is free while the model runs: nothing has landed yet.
        assert_eq!(desktop.revision(), revision);
        let until = std::time::Instant::now() + std::time::Duration::from_secs(300);
        while desktop.poll_job() && std::time::Instant::now() < until {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(desktop.job().is_none(), "the transcript took too long");
        assert_eq!(desktop.notice(), "", "{}", desktop.notice());
        let draft = desktop.draft().unwrap();
        assert!(
            draft
                .tracks
                .iter()
                .any(|track| track.kind == TrackType::Text && !track.segments.is_empty()),
            "no caption landed"
        );
        // The model stays loaded for the next transcript.
        assert!(desktop.speech.borrow().is_some());
    }

    #[test]
    fn split_scenes_runs_as_a_job_and_cuts_a_real_movie_when_it_lands() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
        let movie =
            root.join("Fireship - 5 open source tools that replaced my $320 mo AI stack.mp4");
        let Ok(movie) = movie.canonicalize() else {
            eprintln!("skipped: {} is missing", movie.display());
            return;
        };
        let temp = Temp::new();
        let mut draft = crate::open_movie::movie_draft(&movie, 1920, 1080, false).unwrap();
        draft.tracks[0].segments[0].target = TimeRange::new(0, 30_000_000);
        draft.tracks[0].segments[0].source_range = Some(TimeRange::new(0, 30_000_000));
        draft.duration = Micros(30_000_000);
        let clip = draft.tracks[0].segments[0].id.clone();
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.select(Some(clip));
        let revision = desktop.revision();
        desktop.split_scenes().unwrap();
        assert_eq!(
            desktop.revision(),
            revision,
            "nothing lands before the job ends"
        );
        assert!(desktop.split_scenes().is_err(), "a second job waits");
        let until = std::time::Instant::now() + std::time::Duration::from_secs(120);
        while desktop.poll_job() && std::time::Instant::now() < until {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(desktop.job().is_none(), "the scenes took too long");
        assert_eq!(desktop.notice(), "", "{}", desktop.notice());
        let clips = desktop.draft().unwrap().tracks[0].segments.len();
        assert!(clips > 1, "30 s of a fast-cut video has a scene change");
        desktop.undo().unwrap();
        assert_eq!(desktop.draft().unwrap().tracks[0].segments.len(), 1);
    }

    fn empty_project(dir: &Path) {
        let mut draft = Draft::empty("draft");
        draft.config.maintrack_adsorb = false;
        Folder::new(draft).save(dir).unwrap();
    }

    fn draft_only(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        let mut draft = Draft::empty("draft");
        draft.name = "Only".to_owned();
        std::fs::write(
            dir.join("draft_info.json"),
            cutlass_store::write_draft(&draft).unwrap(),
        )
        .unwrap();
    }

    fn color_project(dir: &Path) {
        let mut draft = Draft::empty("draft");
        draft.name = "Color".to_owned();
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft
            .canvas_config
            .rest
            .insert("kept_field".to_owned(), serde_json::json!({"n": 1}));
        draft.materials.material_colors.push(MaterialColorConfig {
            id: "color".into(),
            is_color_clip: true,
            solid_color: "0xFF336699".to_owned(),
            ..MaterialColorConfig::default()
        });
        draft.tracks[0].segments.push(Segment::new(
            "segment",
            "color",
            TimeRange::new(0, 1_000_000),
        ));
        Folder::new(draft).save(dir).unwrap();
    }

    fn write_wav(path: &Path) {
        let data_bytes = 16_000u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&8_000u32.to_le_bytes());
        bytes.extend_from_slice(&16_000u32.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_bytes.to_le_bytes());
        bytes.extend(std::iter::repeat_n(0u8, data_bytes as usize));
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn an_empty_folder_opens_one_main_lane_and_a_second_open_is_held() {
        let temp = Temp::new();
        draft_only(&temp.path);
        let mut first = shell();
        first.open_folder(&temp.path).unwrap();
        assert_eq!(first.lanes().len(), 1);
        assert!(first.lanes()[0].main);
        assert!(first.lanes()[0].clips.is_empty());
        assert!(temp.path.join(".locked").is_file());
        let mut second = shell();
        let error = second.open_folder(&temp.path).unwrap_err();
        assert!(
            error.to_string().contains("open in another Cutlass window"),
            "{error}"
        );
        assert!(second.draft().is_none());
    }

    #[test]
    fn opening_a_project_notes_it_as_recent_and_says_when_the_last_app_did_not_close() {
        let temp = Temp::new();
        let elsewhere = temp.path.join("elsewhere");
        std::fs::create_dir_all(&elsewhere).unwrap();
        draft_only(&elsewhere);
        let settings_file = temp.path.join("settings").join("settings.json");
        let mut desktop = shell();
        desktop.keep_settings_file(settings_file.clone(), 0);
        desktop.open_folder(&elsewhere).unwrap();
        assert!(desktop.notice().is_empty(), "{}", desktop.notice());
        let saved = cutlass_settings::load(&settings_file).unwrap();
        let canonical = elsewhere.canonicalize().unwrap();
        assert_eq!(saved.settings.recent_projects, vec![canonical.clone()]);
        let projects_root = temp.path.join("projects");
        let recent = desktop.recent_projects(&projects_root);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].path, canonical);
        // A project under the projects folder is already on the launch list.
        assert!(desktop.recent_projects(&temp.path).is_empty());
        desktop.close_project().unwrap();

        // A shell that stopped without closing leaves the folder saying it
        // is open; the next open says so.
        std::fs::write(
            elsewhere.join(".locked"),
            b"{\"address\":null,\"open\":true}\n",
        )
        .unwrap();
        desktop.open_folder(&elsewhere).unwrap();
        assert_eq!(desktop.notice(), super::RECOVERED_NOTICE);
    }

    #[test]
    fn a_newer_protocol_is_named_and_not_opened() {
        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.new_version = "192.0.0".to_owned();
        Folder::new(draft).save(&temp.path).unwrap();
        let error = shell().open_folder(&temp.path).unwrap_err();
        assert!(error.to_string().contains("192.0.0"), "{error}");
    }

    #[test]
    fn save_keeps_an_unknown_field_and_leaves_the_undo_stack_out_of_the_backup() {
        let temp = Temp::new();
        color_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.select(Some(SegmentId::new("segment")));
        desktop.set_volume(0.5).unwrap();
        desktop.save().unwrap();
        let saved = std::fs::read(temp.path.join("draft_info.json")).unwrap();
        let backup = std::fs::read(temp.path.join("draft_info.json.bak")).unwrap();
        assert_eq!(saved, backup);
        assert!(!String::from_utf8_lossy(&backup).contains("\"undo\""));
        let reopened = cutlass_store::open(&temp.path).unwrap();
        assert_eq!(
            reopened.folder.draft.canvas_config.rest["kept_field"]["n"],
            1
        );
        assert_eq!(reopened.folder.draft.tracks[0].segments[0].volume, 0.5);
    }

    #[test]
    fn a_missing_folder_is_named_and_the_session_stays() {
        let temp = Temp::new();
        color_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let moved = temp.path.with_file_name(format!(
            "{}-moved",
            temp.path.file_name().unwrap().to_string_lossy()
        ));
        std::fs::rename(&temp.path, &moved).unwrap();
        let error = desktop.save().unwrap_err();
        assert!(matches!(error, ShellError::MissingFolder(_)), "{error}");
        assert_eq!(desktop.draft().unwrap().name, "Color");
        std::fs::rename(&moved, &temp.path).unwrap();
    }

    #[test]
    fn a_clip_filmstrip_fills_from_the_thumbnail_worker() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
        let movie = root.join("Expo - You Don't Need a Mac to Build iOS Apps Anymore.mp4");
        if !movie.is_file() {
            eprintln!("skipped: {} is missing", movie.display());
            return;
        }
        let temp = Temp::new();
        let mut draft = crate::open_movie::movie_draft(&movie, 1920, 1080, false).unwrap();
        draft.tracks[0].segments[0].target = TimeRange::new(0, 4_000_000);
        draft.tracks[0].segments[0].source_range = Some(TimeRange::new(0, 4_000_000));
        draft.duration = Micros(4_000_000);
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let id = SegmentId::new("movie");
        assert!(desktop.film(&id, 400.0, 48.0, 0.0..400.0).is_empty());

        let (sender, receiver) = std::sync::mpsc::channel();
        desktop.start_thumbnails(move || {
            let _ = sender.send(());
        });
        // 4 s across 400 px: 85 px tiles at 48 rows of 16:9, five of them.
        desktop.pixel_size = 400.0 / 4_000_000.0;
        let cells = desktop.film(&id, 400.0, 48.0, 0.0..400.0);
        assert_eq!(cells.len(), 5);
        assert!(cells.iter().all(|cell| cell.thumb.is_none()));
        desktop.send_timeline_asks();
        for _ in 0..cells.len() {
            receiver
                .recv_timeout(std::time::Duration::from_secs(10))
                .expect("a thumbnail");
        }
        let cells = desktop.film(&id, 400.0, 48.0, 0.0..400.0);
        for cell in &cells {
            let thumb = cell.thumb.as_ref().expect("stored");
            assert_eq!((thumb.width, thumb.height), (284, 160));
            assert!(thumb.rgba.chunks(4).any(|pixel| pixel[..3] != [0, 0, 0]));
        }
        assert_ne!(cells[0].thumb, cells[4].thumb);
    }

    #[test]
    fn a_portrait_clip_gets_tall_narrow_tiles() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
        let movie = root.join("7150172-hd_720_1280_50fps.mp4");
        if !movie.is_file() {
            eprintln!("skipped: {} is missing", movie.display());
            return;
        }
        let temp = Temp::new();
        // Stored as 16:9, as a bad probe might; the file's own shape wins.
        let draft = crate::open_movie::movie_draft(&movie, 1920, 1080, false).unwrap();
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let (sender, receiver) = std::sync::mpsc::channel();
        desktop.start_thumbnails(move || {
            let _ = sender.send(());
        });
        desktop.pixel_size = 400.0 / 1_000_000.0;
        let id = SegmentId::new("movie");
        let wide = desktop.film(&id, 400.0, 48.0, 0.0..400.0);
        assert_eq!(wide[0].tile.width, 48.0 * 16.0 / 9.0);
        desktop.send_timeline_asks();
        receiver
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("a thumbnail");
        let tall = desktop.film(&id, 400.0, 48.0, 0.0..400.0);
        assert_eq!(tall[0].tile.width, 27.0);
        let thumb = tall[0].thumb.as_ref().expect("stored");
        assert_eq!((thumb.width, thumb.height), (90, 160));
    }

    #[test]
    fn the_meter_rises_with_heard_sound_and_falls_when_paused() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
        let sound = root.join("Expo - You Don't Need a Mac to Build iOS Apps Anymore.m4a");
        if !sound.is_file() {
            eprintln!("skipped: {} is missing", sound.display());
            return;
        }
        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.duration = Micros(20_000_000);
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("song"),
            path: sound.display().to_string(),
            duration: Micros(20_000_000),
            ..MaterialAudio::default()
        });
        let mut segment = Segment::new("song", "song", TimeRange::new(0, 20_000_000));
        segment.source_range = Some(TimeRange::new(0, 20_000_000));
        draft.tracks[0].segments.push(segment);
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.seek(Micros(5_000_000));
        desktop.play();
        let mut loudest = cutlass_playback::METER_FLOOR_DB;
        for _ in 0..25 {
            desktop.advance(40_000);
            desktop.update_meter(40_000);
            loudest = loudest.max(desktop.meter()[0].level);
        }
        assert!(loudest > -40.0, "{loudest}");
        desktop.pause();
        desktop.update_meter(10_000_000);
        assert!(desktop.meter_resting());
    }

    #[test]
    fn an_audio_clip_reads_its_waveform_from_the_worker() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
        let sound = root.join("Expo - You Don't Need a Mac to Build iOS Apps Anymore.m4a");
        if !sound.is_file() {
            eprintln!("skipped: {} is missing", sound.display());
            return;
        }
        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.duration = Micros(20_000_000);
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("song"),
            path: sound.display().to_string(),
            duration: Micros(20_000_000),
            ..MaterialAudio::default()
        });
        let mut segment = Segment::new("song", "song", TimeRange::new(0, 20_000_000));
        segment.source_range = Some(TimeRange::new(0, 20_000_000));
        draft.tracks[0].segments.push(segment);
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let (sender, receiver) = std::sync::mpsc::channel();
        desktop.start_waveforms(move || {
            let _ = sender.send(());
        });
        // 20 s across 400 px: one column every 50 ms.
        desktop.pixel_size = 400.0 / 20_000_000.0;
        let id = SegmentId::new("song");
        assert_eq!(desktop.wave(&id, 400.0, 100.0..300.0), None);
        desktop.send_timeline_asks();
        let mut levels = None;
        while receiver
            .recv_timeout(std::time::Duration::from_secs(30))
            .is_ok()
        {
            levels = desktop.wave(&id, 400.0, 100.0..300.0);
            let read = desktop.waveforms.as_ref().unwrap().get(
                &desktop
                    .media_file(Some(&sound.display().to_string()))
                    .unwrap(),
            );
            if read.is_some_and(|peaks| peaks.complete) {
                break;
            }
        }
        let wave = levels.expect("peaks");
        assert!(!wave.under_picture);
        let (first, levels) = (wave.first, wave.levels);
        assert_eq!(first, 100);
        assert_eq!(levels.len(), 200);
        assert!(levels.iter().all(|level| (0.0..=1.0).contains(level)));
        let loud = levels.iter().filter(|level| **level > 0.05).count();
        assert!(loud > 100, "{loud} of {} columns have sound", levels.len());

        // The library card's wave is spread once and kept while the peaks
        // and width stay; a new width spreads again.
        let card = desktop.material_wave("song", 64).expect("card wave");
        assert_eq!(card.len(), 64);
        let again = desktop.material_wave("song", 64).unwrap();
        assert!(std::sync::Arc::ptr_eq(&card, &again));
        assert_eq!(desktop.material_wave("song", 32).unwrap().len(), 32);
    }

    #[test]
    fn a_library_picture_shows_its_first_frame_once_read() {
        let temp = Temp::new();
        Folder::new(Draft::empty("draft")).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let picture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/engine/fixtures/pattern.png");
        desktop.import_file(&picture).unwrap();
        let id = desktop.draft().unwrap().materials.videos[0]
            .id
            .as_str()
            .to_owned();
        assert!(desktop.poster(&id).is_none(), "no worker, no poster");
        let (sender, receiver) = std::sync::mpsc::channel();
        desktop.start_thumbnails(move || {
            let _ = sender.send(());
        });
        let (_, thumb) = desktop.poster(&id).expect("a picture has a poster");
        assert!(thumb.is_none());
        desktop.send_timeline_asks();
        receiver
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("the poster was read");
        let (_, thumb) = desktop.poster(&id).unwrap();
        assert!(thumb.is_some_and(|thumb| thumb.width > 0 && thumb.height > 0));
        assert!(
            desktop.material_wave(&id, 10).is_none(),
            "a picture has no sound"
        );
    }

    #[test]
    fn a_video_clip_with_sound_reads_its_waveform_under_the_picture() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
        let movie = root.join("Expo - You Don't Need a Mac to Build iOS Apps Anymore.mp4");
        if !movie.is_file() {
            eprintln!("skipped: {} is missing", movie.display());
            return;
        }
        let temp = Temp::new();
        let mut draft = crate::open_movie::movie_draft(&movie, 1920, 1080, false).unwrap();
        draft.tracks[0].segments[0].target = TimeRange::new(0, 20_000_000);
        draft.tracks[0].segments[0].source_range = Some(TimeRange::new(0, 20_000_000));
        draft.duration = Micros(20_000_000);
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let (sender, receiver) = std::sync::mpsc::channel();
        desktop.start_waveforms(move || {
            let _ = sender.send(());
        });
        desktop.pixel_size = 400.0 / 20_000_000.0;
        let id = SegmentId::new("movie");
        assert_eq!(
            desktop.wave(&id, 400.0, 0.0..400.0),
            Some(super::ClipWave {
                first: 0,
                levels: Vec::new(),
                gains: Vec::new(),
                under_picture: true,
                muted: false,
            }),
            "a video holds its band open while it is read"
        );
        desktop.send_timeline_asks();
        // Wait until the first 20 s, 2000 buckets, have been read.
        let path = desktop
            .media_file(Some(&movie.display().to_string()))
            .unwrap();
        while desktop
            .waveforms
            .as_ref()
            .unwrap()
            .get(&path)
            .is_none_or(|peaks| peaks.levels.len() < 2_000 && !peaks.complete)
        {
            receiver
                .recv_timeout(std::time::Duration::from_secs(30))
                .expect("peaks");
        }
        let wave = desktop.wave(&id, 400.0, 0.0..400.0).expect("peaks");
        assert!(wave.under_picture);
        assert_eq!(wave.levels.len(), 400);
        let loud = wave.levels.iter().filter(|level| **level > 0.05).count();
        assert!(loud > 200, "{loud} of 400 columns have sound");
    }

    #[test]
    fn a_waveform_follows_the_clip_volume() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
        let sound = root.join("Expo - You Don't Need a Mac to Build iOS Apps Anymore.m4a");
        if !sound.is_file() {
            eprintln!("skipped: {} is missing", sound.display());
            return;
        }
        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.duration = Micros(5_000_000);
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("song"),
            path: sound.display().to_string(),
            duration: Micros(5_000_000),
            ..MaterialAudio::default()
        });
        let mut segment = Segment::new("song", "song", TimeRange::new(0, 5_000_000));
        segment.source_range = Some(TimeRange::new(0, 5_000_000));
        draft.tracks[0].segments.push(segment);
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let (sender, receiver) = std::sync::mpsc::channel();
        desktop.start_waveforms(move || {
            let _ = sender.send(());
        });
        desktop.pixel_size = 400.0 / 5_000_000.0;
        let id = SegmentId::new("song");
        let _ = desktop.wave(&id, 400.0, 0.0..400.0);
        desktop.send_timeline_asks();
        receiver
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("peaks");
        let full = desktop.wave(&id, 400.0, 0.0..400.0).expect("peaks").levels;
        desktop.select(Some(id.clone()));
        desktop.set_volume(0.5).unwrap();
        let half = desktop.wave(&id, 400.0, 0.0..400.0).expect("peaks");
        assert!(half.gains.iter().all(|gain| *gain == 0.5));
        let half = half.levels;
        for (full, half) in full.iter().zip(&half) {
            assert!((full * 0.5 - half).abs() < 1e-6);
        }
        let track = desktop.draft().unwrap().tracks[0].id.clone();
        assert!(!desktop.wave(&id, 400.0, 0.0..400.0).unwrap().muted);
        desktop
            .toggle_attribute(track.clone(), TrackAttribute::MUTE)
            .unwrap();
        assert!(desktop.wave(&id, 400.0, 0.0..400.0).unwrap().muted);
        desktop
            .toggle_attribute(track, TrackAttribute::MUTE)
            .unwrap();
        desktop.set_volume(0.0).unwrap();
        let muted = desktop.wave(&id, 400.0, 0.0..400.0).expect("peaks").levels;
        assert!(muted.iter().all(|level| *level == 0.0));
        assert!(full.iter().any(|level| *level > 0.05));
    }

    #[test]
    fn a_video_clip_without_sound_has_no_waveform() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
        let movie = root.join("7150172-hd_720_1280_50fps.mp4");
        if !movie.is_file() {
            eprintln!("skipped: {} is missing", movie.display());
            return;
        }
        let temp = Temp::new();
        let draft = crate::open_movie::movie_draft(&movie, 1920, 1080, false).unwrap();
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let (sender, receiver) = std::sync::mpsc::channel();
        desktop.start_waveforms(move || {
            let _ = sender.send(());
        });
        desktop.pixel_size = 400.0 / 1_000_000.0;
        let id = SegmentId::new("movie");
        assert_eq!(
            desktop.wave(&id, 400.0, 0.0..400.0),
            Some(super::ClipWave {
                first: 0,
                levels: Vec::new(),
                gains: Vec::new(),
                under_picture: true,
                muted: false,
            }),
            "a video holds its band open while it is read"
        );
        desktop.send_timeline_asks();
        receiver
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("read");
        let path = desktop
            .media_file(Some(&movie.display().to_string()))
            .unwrap();
        let peaks = desktop
            .waveforms
            .as_ref()
            .unwrap()
            .get(&path)
            .expect("read");
        assert!(peaks.complete && peaks.levels.is_empty());
        assert_eq!(desktop.wave(&id, 400.0, 0.0..400.0), None);
    }

    #[test]
    fn the_preview_background_starts_black_and_toggles_clear() {
        let mut desktop = shell();
        assert!(desktop.preview_black());
        desktop.toggle_preview_background();
        assert!(!desktop.preview_black());
        desktop.toggle_preview_background();
        assert!(desktop.preview_black());
    }

    #[test]
    fn an_import_is_written_to_the_folder_without_closing_the_project() {
        let temp = Temp::new();
        empty_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        assert_eq!(desktop.autosave(), None);
        let picture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/engine/fixtures/pattern.png");
        desktop.import_file(&picture).unwrap();
        let wait = desktop.autosave().expect("the write waits for the gap");
        std::thread::sleep(wait);
        // The write runs on its own thread; the window looks again until it
        // has landed.
        let until = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while let Some(wait) = desktop.autosave() {
            assert!(
                std::time::Instant::now() < until,
                "the autosave never landed"
            );
            std::thread::sleep(wait);
        }
        assert!(!desktop.unsaved());
        drop(desktop);

        let mut reopened = shell();
        reopened.open_folder(&temp.path).unwrap();
        let videos = &reopened.draft().unwrap().materials.videos;
        assert_eq!(videos.len(), 1);
        assert!(
            videos[0].path.ends_with("pattern.png"),
            "{}",
            videos[0].path
        );
        assert_eq!(reopened.autosave(), None);
    }

    #[test]
    fn dropped_files_import_one_after_another_and_a_dropped_project_opens() {
        let temp = Temp::new();
        let project = temp.path.join("project");
        std::fs::create_dir_all(&project).unwrap();
        empty_project(&project);
        let picture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/engine/fixtures/pattern.png");
        let second = temp.path.join("second.png");
        std::fs::copy(&picture, &second).unwrap();

        let mut desktop = shell();
        let refused = desktop.drop_files(std::slice::from_ref(&picture));
        assert_eq!(
            refused.unwrap_err().to_string(),
            "open a project to import into first"
        );
        desktop.drop_files(std::slice::from_ref(&project)).unwrap();
        assert_eq!(desktop.project_folder(), Some(project.as_path()));

        desktop
            .drop_files(&[picture.clone(), second.clone()])
            .unwrap();
        assert_eq!(
            desktop.job().map(|job| job.label),
            Some("Import".to_owned())
        );
        land_job(&mut desktop);
        let videos = &desktop.draft().unwrap().materials.videos;
        assert_eq!(videos.len(), 2, "{}", desktop.notice());
        assert!(
            desktop
                .drop_files(&[temp.path.join("missing.png")])
                .is_err()
        );
    }

    #[test]
    fn an_autosave_writes_on_its_own_thread_and_a_save_waits_for_it() {
        let temp = Temp::new();
        empty_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let picture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/engine/fixtures/pattern.png");
        desktop.import_file(&picture).unwrap();
        let wait = desktop.autosave().expect("the write waits for the gap");
        std::thread::sleep(wait);
        // The write starts and the window is told to look again soon.
        assert_eq!(
            desktop.autosave(),
            Some(std::time::Duration::from_millis(50))
        );
        assert!(desktop.saving.is_some());
        // An explicit save, or quitting, waits for it rather than writing
        // the same files at once.
        desktop.save().unwrap();
        assert!(desktop.saving.is_none());
        assert!(!desktop.unsaved());
        assert_eq!(desktop.autosave(), None);

        // A write still under way when another project opens counts for
        // the project it wrote, not the new one.
        desktop.import_file(&picture).unwrap();
        std::thread::sleep(super::AUTOSAVE_GAP);
        assert!(desktop.autosave().is_some());
        let other = Temp::new();
        empty_project(&other.path);
        desktop.open_folder(&other.path).unwrap();
        assert!(!desktop.unsaved());
        let mut reopened = shell();
        reopened.open_folder(&temp.path).unwrap();
        assert_eq!(reopened.draft().unwrap().materials.videos.len(), 2);
    }

    #[test]
    fn a_failed_autosave_keeps_the_edits_unsaved_for_quit() {
        let temp = Temp::new();
        empty_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let picture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/engine/fixtures/pattern.png");
        desktop.import_file(&picture).unwrap();
        std::fs::remove_dir_all(&temp.path).unwrap();
        let wait = desktop.autosave().expect("the write waits for the gap");
        std::thread::sleep(wait);
        assert_eq!(desktop.autosave(), Some(super::AUTOSAVE_GAP));
        assert!(desktop.unsaved());
    }

    #[test]
    fn import_stores_a_still_and_a_sound_without_a_timeline_clip() {
        let temp = Temp::new();
        empty_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let picture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/engine/fixtures/pattern.png");
        desktop.import_file(&picture).unwrap();
        assert!(desktop.lanes().iter().all(|lane| lane.clips.is_empty()));
        assert_eq!(desktop.draft().unwrap().tracks.len(), 1);
        assert!(temp.path.join("Resources").join("pattern.png").is_file());
        assert!(
            desktop.draft().unwrap().materials.videos[0]
                .path
                .starts_with("Resources/")
        );

        let sound = temp.path.join("tone.wav");
        write_wav(&sound);
        desktop.import_file(&sound).unwrap();
        assert!(desktop.lanes().iter().all(|lane| lane.clips.is_empty()));
        assert_eq!(desktop.draft().unwrap().tracks.len(), 1);
        assert!(
            desktop.draft().unwrap().materials.audios[0]
                .path
                .ends_with("tone.wav")
        );

        let junk = temp.path.join("notes.txt");
        std::fs::write(&junk, b"hello").unwrap();
        let before = desktop.draft().unwrap().clone();
        let error = desktop.import_file(&junk).unwrap_err();
        assert!(
            error.to_string().contains("not a media container"),
            "{error}"
        );
        assert_eq!(desktop.draft().unwrap(), &before);

        let truncated = temp.path.join("short.png");
        std::fs::write(
            &truncated,
            [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1A, b'\n'],
        )
        .unwrap();
        let error = desktop.import_file(&truncated).unwrap_err();
        assert!(error.to_string().contains("still"), "{error}");
        assert_eq!(desktop.draft().unwrap(), &before);
        assert!(!temp.path.join("Resources").join("short.png").exists());
    }

    #[test]
    fn an_import_failure_is_written_to_the_log() {
        let log_dir =
            std::env::temp_dir().join(format!("cutlass-import-log-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&log_dir);
        std::fs::create_dir_all(&log_dir).unwrap();
        cutlass_logger::init(&log_dir).unwrap();
        let temp = Temp::new();
        empty_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let junk = temp.path.join("notes.txt");
        std::fs::write(&junk, b"hello").unwrap();
        let error = desktop.import_file(&junk).unwrap_err();
        let log = std::fs::read_to_string(log_dir.join(cutlass_logger::FILE_NAME)).unwrap();
        assert!(log.contains(&error.to_string()), "{log}");
        assert!(log.contains("notes.txt"), "{log}");
        let _ = std::fs::remove_dir_all(&log_dir);
    }

    #[test]
    fn a_fragmented_movie_imports_when_the_header_duration_is_empty() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../local-files/GOOGLE IS BACK! (Gemini 3.8 Flash).mp4");
        if !path.is_file() {
            eprintln!("skipped: the fragmented sample is not in ../local-files");
            return;
        }
        let temp = Temp::new();
        empty_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.import_file(&path).unwrap();
        let video = &desktop.draft().unwrap().materials.videos[0];
        assert_eq!(video.width, 854);
        assert_eq!(video.height, 480);
        assert!(
            (1_009_000_000..1_011_000_000).contains(&video.duration.0),
            "{}",
            video.duration.0
        );
        assert!(desktop.lanes().iter().all(|lane| lane.clips.is_empty()));
    }

    #[test]
    fn an_empty_timeline_opens_at_a_third_and_the_first_clip_fits() {
        let temp = Temp::new();
        empty_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.prepare_lane(900.0);
        assert_eq!(desktop.pixel_size(), 300.0 / 10_000_000.0);
        let picture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/engine/fixtures/pattern.png");
        desktop.import_file(&picture).unwrap();
        let id = desktop.draft().unwrap().materials.videos[0]
            .id
            .as_str()
            .to_owned();
        let y = crate::tokens::LANE_HEIGHT / 2.0;
        desktop.drop_medium(&id, 0.0, y).unwrap();
        desktop.prepare_lane(900.0);
        let cut = desktop.cut_duration().0 as f64;
        let fitted = desktop.pixel_size();
        assert!((fitted * cut - 675.0).abs() < 1.0e-6, "{}", fitted * cut);
        desktop.drop_medium(&id, 0.0, y).unwrap();
        desktop.prepare_lane(900.0);
        assert_eq!(desktop.pixel_size(), fitted);
    }

    #[test]
    fn a_library_drop_places_the_still_and_one_undo_removes_it() {
        let temp = Temp::new();
        empty_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let picture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/engine/fixtures/pattern.png");
        desktop.import_file(&picture).unwrap();
        let id = desktop.draft().unwrap().materials.videos[0]
            .id
            .as_str()
            .to_owned();
        let duration = desktop.draft().unwrap().materials.videos[0].duration;
        desktop.pixel_size = 0.001;
        let clear = desktop.preview_bytes().unwrap();
        desktop
            .drop_medium(&id, 0.0, crate::tokens::LANE_HEIGHT / 2.0)
            .unwrap();
        let lanes = desktop.lanes();
        assert_eq!(lanes.len(), 1);
        assert!(lanes[0].main);
        assert_eq!(lanes[0].clips.len(), 1);
        assert_eq!(lanes[0].clips[0].start, Micros(0));
        assert_eq!(lanes[0].clips[0].duration, duration);
        let segment = &desktop.draft().unwrap().tracks[0].segments[0];
        assert_eq!(segment.source_range, Some(TimeRange::new(0, duration.0)));
        assert_eq!(segment.material_id.as_str(), id);
        assert_eq!(
            desktop.selected().map(|item| item.as_str()),
            Some(segment.id.as_str())
        );
        let placed = desktop.preview_bytes().unwrap();
        assert_ne!(digest(&clear.rgba), digest(&placed.rgba));
        desktop
            .drop_medium(&id, 4_000.0, crate::tokens::LANE_HEIGHT / 2.0)
            .unwrap();
        assert_eq!(desktop.draft().unwrap().tracks.len(), 1);
        assert_eq!(desktop.lanes()[0].clips.len(), 2);
        // Nothing changed, so the lanes are the ones already built.
        assert!(std::sync::Arc::ptr_eq(&desktop.lanes(), &desktop.lanes()));
        let before = desktop.lanes();
        desktop.undo().unwrap();
        assert!(!std::sync::Arc::ptr_eq(&before, &desktop.lanes()));
        assert_eq!(desktop.lanes()[0].clips.len(), 1);
        desktop.undo().unwrap();
        assert!(desktop.lanes().iter().all(|lane| lane.clips.is_empty()));
        desktop.redo().unwrap();
        assert_eq!(desktop.lanes()[0].clips.len(), 1);
    }

    #[test]
    fn align_puts_the_pictures_edge_on_the_canvas_edge() {
        use cutlass_editor::{NoText, StageAlign, stage_box};
        let temp = Temp::new();
        empty_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let picture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/engine/fixtures/pattern.png");
        desktop.import_file(&picture).unwrap();
        let id = desktop.draft().unwrap().materials.videos[0]
            .id
            .as_str()
            .to_owned();
        desktop.pixel_size = 0.001;
        desktop
            .drop_medium(&id, 0.0, crate::tokens::LANE_HEIGHT / 2.0)
            .unwrap();
        desktop.set_scale(0.5).unwrap();
        let clip = desktop.selected().cloned().unwrap();
        let (width, height) = cutlass_editor::canvas_size(desktop.draft().unwrap());
        let edges = |desktop: &Shell| {
            let stage = stage_box(desktop.draft().unwrap(), &clip, Micros(0), &NoText).unwrap();
            let corners = stage.corners();
            let xs = corners.map(|corner| corner.0);
            let ys = corners.map(|corner| corner.1);
            (
                xs.iter().copied().fold(f64::INFINITY, f64::min),
                xs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                ys.iter().copied().fold(f64::INFINITY, f64::min),
                ys.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            )
        };
        let near = |a: f64, b: f64| (a - b).abs() < 1e-6;
        desktop.align_selected(StageAlign::Left).unwrap();
        assert!(near(edges(&desktop).0, 0.0));
        desktop.align_selected(StageAlign::Bottom).unwrap();
        let (left, _, _, bottom) = edges(&desktop);
        assert!(near(left, 0.0) && near(bottom, height));
        desktop.align_selected(StageAlign::Right).unwrap();
        assert!(near(edges(&desktop).1, width));
        desktop.undo().unwrap();
        assert!(near(edges(&desktop).0, 0.0));
    }

    #[test]
    fn a_drop_on_a_taken_span_opens_an_overlay_above_the_main_lane() {
        let temp = Temp::new();
        empty_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let picture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/engine/fixtures/pattern.png");
        desktop.import_file(&picture).unwrap();
        let id = desktop.draft().unwrap().materials.videos[0]
            .id
            .as_str()
            .to_owned();
        desktop.pixel_size = 0.001;
        let y = crate::tokens::LANE_HEIGHT / 2.0;
        desktop.drop_medium(&id, 0.0, y).unwrap();
        desktop.drop_medium(&id, 0.0, y).unwrap();
        let draft = desktop.draft().unwrap();
        assert_eq!(draft.tracks.len(), 2);
        assert!(draft.tracks[0].is_sub_video());
        assert_eq!(draft.tracks[0].segments.len(), 1);
        assert!(draft.tracks[1].is_main());
        assert_eq!(draft.tracks[1].segments.len(), 1);
        desktop.undo().unwrap();
        assert_eq!(desktop.draft().unwrap().tracks.len(), 1);
        assert_eq!(desktop.lanes()[0].clips.len(), 1);
    }

    #[test]
    fn the_magnet_packs_the_main_lane_and_a_lock_refuses_the_switch() {
        let mut bare = shell();
        assert_eq!(
            bare.toggle_magnet().unwrap_err().to_string(),
            "no project is open"
        );

        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.tracks[0]
            .segments
            .push(Segment::new("a", "media", TimeRange::new(500_000, 100_000)));
        draft.tracks[0]
            .segments
            .push(Segment::new("b", "media", TimeRange::new(800_000, 100_000)));
        let mut sound = Track::main("sound");
        sound.kind = TrackType::Audio;
        sound.segments.push(Segment::new(
            "tone",
            "tone",
            TimeRange::new(400_000, 100_000),
        ));
        draft.tracks.push(sound);
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        assert!(desktop.magnet_enabled());
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0].target.start,
            Micros(500_000)
        );

        desktop.toggle_magnet().unwrap();
        assert!(!desktop.magnet_enabled());
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0].target.start,
            Micros(500_000)
        );

        desktop.toggle_magnet().unwrap();
        assert!(desktop.magnet_enabled());
        let packed = desktop.draft().unwrap();
        assert_eq!(packed.tracks[0].segments[0].id, SegmentId::new("a"));
        assert_eq!(
            packed.tracks[0].segments[0].target,
            TimeRange::new(0, 100_000)
        );
        assert_eq!(
            packed.tracks[0].segments[1].target,
            TimeRange::new(100_000, 100_000)
        );
        assert_eq!(packed.tracks[1].segments[0].target.start, Micros(400_000));
        desktop.undo().unwrap();
        assert!(!desktop.magnet_enabled());
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0].target.start,
            Micros(500_000)
        );
        desktop.redo().unwrap();
        assert!(desktop.magnet_enabled());

        let track_id = desktop.draft().unwrap().tracks[0].id.clone();
        desktop
            .toggle_attribute(track_id, TrackAttribute::LOCK)
            .unwrap();
        let locked = desktop.toggle_magnet().unwrap_err();
        assert_eq!(locked.to_string(), "the main track is locked");
        assert!(desktop.magnet_enabled());
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0].target.start,
            Micros(0)
        );
    }

    #[test]
    fn a_new_project_starts_with_linkage_on_and_keeps_the_switch() {
        let root = Temp::new();
        let path = Shell::create_project(&root.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&path).unwrap();
        assert!(desktop.linkage_enabled());
        desktop.toggle_linkage().unwrap();
        assert!(!desktop.linkage_enabled());
        desktop.close_project().unwrap();
        desktop.open_folder(&path).unwrap();
        assert!(!desktop.linkage_enabled());
        let saved = Folder::read(&path).unwrap();
        assert_eq!(
            saved.biz_config.unwrap()["timeline_settings"][saved.draft.id.as_str()]["linkage_enabled"],
            false
        );
    }

    #[test]
    fn a_clip_on_one_main_clip_follows_it() {
        let temp = Temp::new();
        empty_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        assert!(!desktop.linkage_enabled());
        let picture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/engine/fixtures/pattern.png");
        desktop.import_file(&picture).unwrap();
        let id = desktop.draft().unwrap().materials.videos[0]
            .id
            .as_str()
            .to_owned();
        desktop.pixel_size = 0.001;
        desktop.toggle_linkage().unwrap();
        let y = crate::tokens::LANE_HEIGHT / 2.0;
        desktop.drop_medium(&id, 0.0, y).unwrap();
        assert!(desktop.draft().unwrap().relationships.is_empty());
        desktop.drop_medium(&id, 0.0, y).unwrap();
        let draft = desktop.draft().unwrap();
        let owner = draft.tracks[1].segments[0].id.as_str().to_owned();
        let dependent = draft.tracks[0].segments[0].id.as_str().to_owned();
        let duration = draft.tracks[1].segments[0].target.duration.0;
        let pairs: Vec<_> = draft
            .relationships
            .iter()
            .filter(|item| item.kind == "linkage")
            .flat_map(|item| item.id_to_id.iter())
            .map(|pair| {
                (
                    pair.owner.as_str().to_owned(),
                    pair.dependent.as_str().to_owned(),
                )
            })
            .collect();
        assert_eq!(pairs, [(owner.clone(), dependent.clone())]);
        desktop
            .trim_edge(
                SegmentId::new(owner.clone()),
                TrimEdge::End,
                lane_time(&desktop, duration as f64 / 2.0),
            )
            .unwrap();
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0]
                .target
                .duration
                .0,
            duration / 2
        );
        desktop.undo().unwrap();
        desktop
            .drag_segment(
                SegmentId::new(owner.clone()),
                lane_time(&desktop, 500_000.0),
                crate::tokens::LANE_HEIGHT + y,
            )
            .unwrap();
        let draft = desktop.draft().unwrap();
        assert_eq!(draft.tracks[1].segments[0].target.start, Micros(500_000));
        assert_eq!(draft.tracks[0].segments[0].target.start, Micros(500_000));
        desktop.undo().unwrap();
        desktop.toggle_linkage().unwrap();
        desktop
            .drag_segment(
                SegmentId::new(owner.clone()),
                lane_time(&desktop, 500_000.0),
                crate::tokens::LANE_HEIGHT + y,
            )
            .unwrap();
        let draft = desktop.draft().unwrap();
        assert_eq!(draft.tracks[1].segments[0].target.start, Micros(500_000));
        assert_eq!(draft.tracks[0].segments[0].target.start, Micros(0));
        assert_eq!(
            draft.relationships[0].id_to_id[0].dependent.as_str(),
            dependent
        );
        desktop.toggle_linkage().unwrap();
        desktop.undo().unwrap();
        desktop.select(Some(SegmentId::new(owner)));
        desktop.remove_selected().unwrap();
        assert!(
            desktop
                .draft()
                .unwrap()
                .tracks
                .iter()
                .all(|track| track.segments.is_empty())
        );
        desktop.undo().unwrap();
        assert_eq!(desktop.draft().unwrap().tracks[0].segments.len(), 1);
        assert_eq!(desktop.draft().unwrap().tracks[1].segments.len(), 1);
    }

    #[test]
    fn a_clip_on_two_main_clips_is_not_linked() {
        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.config.maintrack_adsorb = false;
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        let mut video = MaterialVideo::new("wide");
        video.duration = Micros(10_000_000);
        draft.materials.videos.push(video);
        draft.tracks[0]
            .segments
            .push(Segment::new("first", "wide", TimeRange::new(0, 1_000_000)));
        draft.tracks[0].segments.push(Segment::new(
            "second",
            "wide",
            TimeRange::new(3_000_000, 500_000),
        ));
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.toggle_linkage().unwrap();
        desktop
            .drop_medium("wide", 0.0, crate::tokens::LANE_HEIGHT / 2.0)
            .unwrap();
        assert!(desktop.draft().unwrap().relationships.is_empty());
        assert_eq!(desktop.draft().unwrap().tracks.len(), 2);
    }

    #[test]
    fn a_marker_lands_on_the_ruler_or_on_a_clip() {
        let mut bare = shell();
        bare.toggle_marker();
        assert!(!bare.marking());

        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.config.maintrack_adsorb = false;
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.tracks[0]
            .segments
            .push(Segment::new("clip", "media", TimeRange::new(0, 2_000_000)));
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.pixel_size = 0.001;
        desktop.toggle_marker();
        assert!(desktop.marking());
        desktop.toggle_blade();
        assert!(desktop.blade());
        assert!(!desktop.marking());
        desktop.clear_blade();
        desktop.toggle_marker();

        let y = crate::tokens::LANE_HEIGHT / 2.0;
        desktop
            .drop_marker(None, lane_time(&desktop, 1_500_000.0), 0.0)
            .unwrap();
        assert!(
            desktop.draft().unwrap().tracks[0].segments[0]
                .extra_material_refs
                .is_empty()
        );
        let ruler = desktop.markers();
        assert_eq!(ruler.len(), 1);
        assert!(ruler[0].segment_id.is_none());
        assert_eq!(ruler[0].time, Micros(1_500_000));
        assert_eq!(ruler[0].color, super::MARKER_COLOR);
        desktop.undo().unwrap();
        assert!(desktop.markers().is_empty());
        desktop.redo().unwrap();

        let id = SegmentId::new("clip");
        desktop
            .drop_marker(Some(id.clone()), lane_time(&desktop, 400_000.0), y)
            .unwrap();
        let clip_mark = desktop
            .markers()
            .into_iter()
            .find(|mark| mark.segment_id.is_some())
            .expect("clip mark");
        assert_eq!(clip_mark.stored_start, 400_000);
        assert_eq!(clip_mark.time, Micros(400_000));
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0].target,
            TimeRange::new(0, 2_000_000)
        );
        let again = desktop
            .drop_marker(Some(id.clone()), lane_time(&desktop, 400_000.0), y)
            .unwrap_err();
        assert_eq!(again.to_string(), "a marker is already there");

        let track_id = desktop.draft().unwrap().tracks[0].id.clone();
        desktop
            .toggle_attribute(track_id.clone(), TrackAttribute::LOCK)
            .unwrap();
        let locked = desktop
            .drop_marker(Some(id.clone()), lane_time(&desktop, 800_000.0), y)
            .unwrap_err();
        assert_eq!(locked.to_string(), "the track is locked");
        desktop
            .toggle_attribute(track_id, TrackAttribute::LOCK)
            .unwrap();

        desktop.select_marker(Some(id.clone()), 400_000);
        desktop.delete_target().unwrap();
        assert!(
            desktop
                .markers()
                .iter()
                .all(|mark| mark.segment_id.is_none())
        );
        desktop
            .drop_marker(Some(id.clone()), lane_time(&desktop, 400_000.0), y)
            .unwrap();
        desktop
            .drag_segment(id.clone(), lane_time(&desktop, 1_000_000.0), y)
            .unwrap();
        let moved = desktop
            .markers()
            .into_iter()
            .find(|mark| mark.segment_id.is_some())
            .expect("moved mark");
        let start = desktop.draft().unwrap().tracks[0].segments[0]
            .target
            .start
            .0;
        assert_eq!(moved.stored_start, 400_000);
        assert_eq!(moved.time.0, start + 400_000);
        assert!(start > 0, "the clip did not move");

        desktop.seek(Micros(start.saturating_add(2_000_000)));
        desktop.select(Some(id));
        desktop.copy_selected().unwrap();
        desktop.paste().unwrap();
        let clips: Vec<_> = desktop
            .markers()
            .into_iter()
            .filter(|mark| mark.segment_id.is_some())
            .collect();
        assert_eq!(clips.len(), 2);
        assert_eq!(clips[0].stored_start, 400_000);
        assert_eq!(clips[1].stored_start, 400_000);
        assert_ne!(clips[0].segment_id, clips[1].segment_id);
        assert_ne!(clips[0].time, clips[1].time);
    }

    #[test]
    fn a_beat_lands_on_an_audio_clip() {
        let mut bare = shell();
        bare.toggle_beat();
        assert!(!bare.beating());

        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.tracks[0].segments.push(Segment::new(
            "picture",
            "media",
            TimeRange::new(0, 2_000_000),
        ));
        let mut sound = Track::main("sound");
        sound.kind = TrackType::Audio;
        sound
            .segments
            .push(Segment::new("tone", "tone", TimeRange::new(0, 2_000_000)));
        draft.tracks.push(sound);
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.pixel_size = 0.001;
        desktop.toggle_beat();
        assert!(desktop.beating());
        desktop.toggle_marker();
        assert!(desktop.marking());
        assert!(!desktop.beating());
        desktop.clear_marker();
        desktop.toggle_beat();
        desktop.toggle_blade();
        assert!(desktop.blade());
        assert!(!desktop.beating());
        desktop.clear_blade();
        desktop.toggle_beat();

        let picture = crate::tokens::LANE_HEIGHT / 2.0;
        let audio = crate::tokens::LANE_HEIGHT * 1.5;
        let refused = desktop
            .drop_beat(
                SegmentId::new("picture"),
                lane_time(&desktop, 400_000.0),
                picture,
            )
            .unwrap_err();
        assert_eq!(refused.to_string(), "a beat belongs on audio");
        assert!(desktop.beats().is_empty());

        let tone = SegmentId::new("tone");
        desktop
            .drop_beat(tone.clone(), lane_time(&desktop, 400_000.0), audio)
            .unwrap();
        let beats = desktop.beats();
        assert_eq!(beats.len(), 1);
        assert_eq!(beats[0].segment_id, tone);
        assert_eq!(beats[0].source, 400_000);
        assert_eq!(beats[0].time, Micros(400_000));
        desktop.undo().unwrap();
        assert!(desktop.beats().is_empty());
        desktop.redo().unwrap();
        let again = desktop
            .drop_beat(tone.clone(), lane_time(&desktop, 400_000.0), audio)
            .unwrap_err();
        assert_eq!(again.to_string(), "a beat is already there");

        let track_id = desktop.draft().unwrap().tracks[1].id.clone();
        desktop
            .toggle_attribute(track_id.clone(), TrackAttribute::LOCK)
            .unwrap();
        let locked = desktop
            .drop_beat(tone.clone(), lane_time(&desktop, 800_000.0), audio)
            .unwrap_err();
        assert_eq!(locked.to_string(), "the track is locked");
        desktop
            .toggle_attribute(track_id, TrackAttribute::LOCK)
            .unwrap();

        desktop.select_beat(tone.clone(), 400_000);
        desktop.delete_target().unwrap();
        assert!(desktop.beats().is_empty());
        desktop
            .drop_beat(tone.clone(), lane_time(&desktop, 400_000.0), audio)
            .unwrap();
        desktop
            .drag_segment(tone.clone(), lane_time(&desktop, 1_000_000.0), audio)
            .unwrap();
        let moved = desktop.beats();
        assert_eq!(moved.len(), 1);
        let start = desktop.draft().unwrap().tracks[1].segments[0]
            .target
            .start
            .0;
        assert_eq!(moved[0].source, 400_000);
        assert_eq!(moved[0].time.0, start + 400_000);
        assert!(start > 0, "the clip did not move");

        desktop.seek(Micros(start.saturating_add(2_000_000)));
        desktop.select(Some(tone));
        desktop.copy_selected().unwrap();
        desktop.paste().unwrap();
        let copied = desktop.beats();
        assert_eq!(copied.len(), 2);
        assert_eq!(copied[0].source, 400_000);
        assert_eq!(copied[1].source, 400_000);
        assert_ne!(copied[0].segment_id, copied[1].segment_id);
        assert_ne!(copied[0].time, copied[1].time);
    }

    fn touching_cut(dir: &Path) {
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.tracks[0]
            .segments
            .push(Segment::new("left", "media", TimeRange::new(0, 1_000_000)));
        draft.tracks[0].segments.push(Segment::new(
            "right",
            "media",
            TimeRange::new(1_000_000, 1_000_000),
        ));
        Folder::new(draft).save(dir).unwrap();
    }

    #[test]
    fn an_effect_dropped_on_a_clip_runs_over_it_alone_and_one_at_the_playhead_over_all() {
        let temp = Temp::new();
        touching_cut(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.pixel_size = 0.001;
        let card = cutlass_editor::effect_cards()[0].id.clone();
        let main_row = crate::tokens::LANE_HEIGHT / 2.0;
        desktop
            .drop_effect(&card, lane_time(&desktop, 1_500_000.0), main_row)
            .unwrap();
        let effects = |desktop: &Shell| -> Vec<Segment> {
            desktop
                .draft()
                .unwrap()
                .tracks
                .iter()
                .filter(|track| track.kind == TrackType::Effect)
                .flat_map(|track| track.segments.clone())
                .collect()
        };
        let bound = effects(&desktop);
        assert_eq!(bound.len(), 1);
        assert_eq!(bound[0].target, TimeRange::new(1_000_000, 1_000_000));
        assert_eq!(bound[0].bound_clip(), Some(SegmentId::new("right")));
        assert_eq!(desktop.selected(), Some(&bound[0].id));

        desktop.seek(Micros(0));
        desktop.add_effect(&card).unwrap();
        let all = effects(&desktop);
        assert_eq!(all.len(), 2);
        let free = all
            .iter()
            .find(|segment| segment.id != bound[0].id)
            .unwrap();
        assert_eq!(free.bound_clip(), None);
        assert_eq!(
            free.target,
            TimeRange::new(0, cutlass_editor::EFFECT_LENGTH.0)
        );
        desktop.undo().unwrap();
        assert_eq!(effects(&desktop).len(), 1);
        assert!(desktop.add_effect("effect:recipe/none").is_err());
    }

    #[test]
    fn v_deactivates_the_picked_clip_dims_it_and_undo_restores_it() {
        let temp = Temp::new();
        touching_cut(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.select(Some(SegmentId::new("right")));
        let active = |desktop: &Shell| {
            desktop
                .lanes()
                .iter()
                .flat_map(|lane| lane.clips.iter())
                .find(|clip| clip.id.as_str() == "right")
                .map(|clip| clip.active)
        };
        assert_eq!(active(&desktop), Some(true));
        desktop.run_shortcut("v").unwrap();
        assert_eq!(active(&desktop), Some(false));
        assert!(desktop.selection_inactive());
        desktop.run_shortcut("v").unwrap();
        assert_eq!(active(&desktop), Some(true));
        desktop.undo().unwrap();
        assert_eq!(active(&desktop), Some(false));
        desktop.undo().unwrap();
        assert_eq!(active(&desktop), Some(true));

        desktop.mark_clip_in().unwrap();
        desktop.mark_clip_out().unwrap();
        assert_eq!(desktop.marks().mark_in, Some(Micros(1_000_000)));
        assert_eq!(desktop.marks().mark_out, Some(Micros(2_000_000)));
        desktop.select(None);
        assert!(desktop.toggle_active().is_err());
    }

    #[test]
    fn a_title_card_lands_at_the_playhead_styled_and_undoes_in_one_step() {
        let temp = Temp::new();
        touching_cut(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.pixel_size = 0.001;
        desktop.seek(Micros(500_000));
        let lower = cutlass_editor::title_style("title:lower-third").unwrap();
        desktop.add_title(&lower.card_id()).unwrap();
        desktop.add_title(&lower.card_id()).unwrap();
        let rows = |desktop: &Shell| {
            desktop
                .draft()
                .unwrap()
                .tracks
                .iter()
                .filter(|track| track.kind == TrackType::Text)
                .count()
        };
        assert_eq!(rows(&desktop), 2, "the second title overlaps the first");
        let draft = desktop.draft().unwrap();
        let text = &draft.materials.texts[0];
        assert_eq!(text.words().unwrap(), "Lower third");
        assert_eq!(text.text_color, "#EFC56A");
        let picked = desktop.selected().cloned().unwrap();
        let segment = draft
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .find(|segment| segment.id == picked)
            .unwrap();
        assert_eq!(segment.target.start, Micros(500_000));
        assert_eq!(segment.target.duration, cutlass_editor::TITLE_LENGTH);
        assert!(segment.clip.transform.y < 0.0);
        desktop.undo().unwrap();
        assert_eq!(rows(&desktop), 1);
        assert_eq!(desktop.draft().unwrap().materials.texts.len(), 1);

        let end = cutlass_editor::title_style("title:end-card").unwrap();
        let text_row = crate::tokens::LANE_HEIGHT / 2.0;
        desktop
            .drop_title(&end.card_id(), lane_time(&desktop, 4_000_000.0), text_row)
            .unwrap();
        assert_eq!(rows(&desktop), 1, "the drop lands after the first title");
        let draft = desktop.draft().unwrap();
        let dropped = draft
            .materials
            .texts
            .iter()
            .find(|text| text.words().as_deref() == Some("End card"))
            .unwrap();
        assert_eq!(dropped.text_color, "#F0E2C4");
        assert!(desktop.drop_title("title:nope", 0.0, text_row).is_err());
    }

    #[test]
    fn a_dropped_transition_shows_in_the_preview_with_no_package() {
        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        for (id, color) in [("red", "0xFFFF0000"), ("blue", "0xFF0000FF")] {
            draft.materials.material_colors.push(MaterialColorConfig {
                id: id.into(),
                is_color_clip: true,
                solid_color: color.to_owned(),
                ..MaterialColorConfig::default()
            });
        }
        draft.tracks[0]
            .segments
            .push(Segment::new("left", "red", TimeRange::new(0, 1_000_000)));
        draft.tracks[0].segments.push(Segment::new(
            "right",
            "blue",
            TimeRange::new(1_000_000, 1_000_000),
        ));
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.pixel_size = 0.001;
        let card = super::TRANSITION_OFFERS
            .iter()
            .find(|offer| offer.id == "dip-white")
            .unwrap()
            .card_id();
        desktop
            .drop_transition(
                &card,
                lane_time(&desktop, 1_000_000.0),
                crate::tokens::LANE_HEIGHT / 2.0,
            )
            .unwrap();
        // The dip runs over the last 0.4s of the red clip: white halfway.
        desktop.seek(Micros(800_000));
        let rgba = desktop.preview_bytes().unwrap().rgba;
        assert_eq!(&rgba[..4], &[255, 255, 255, 255]);
        desktop.seek(Micros(500_000));
        let rgba = desktop.preview_bytes().unwrap().rgba;
        assert_eq!(&rgba[..4], &[255, 0, 0, 255]);
    }

    #[test]
    fn a_transition_lands_on_the_main_cut_and_undo_lifts_it() {
        let temp = Temp::new();
        touching_cut(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.pixel_size = 0.001;
        let y = crate::tokens::LANE_HEIGHT / 2.0;
        let card = super::TRANSITION_OFFERS[0].card_id();
        let missed = desktop
            .drop_transition(&card, lane_time(&desktop, 100_000.0), y)
            .unwrap_err();
        assert_eq!(missed.to_string(), "the drop missed the cut");
        assert!(desktop.draft().unwrap().materials.transitions.is_empty());
        desktop
            .drop_transition(&card, lane_time(&desktop, 1_000_000.0), y)
            .unwrap();
        let draft = desktop.draft().unwrap();
        assert_eq!(draft.tracks.len(), 1);
        assert_eq!(
            draft.tracks[0].segments[0].target,
            TimeRange::new(0, 1_000_000)
        );
        assert_eq!(
            draft.tracks[0].segments[1].target,
            TimeRange::new(1_000_000, 1_000_000)
        );
        assert_eq!(draft.materials.transitions.len(), 1);
        assert_eq!(draft.materials.transitions[0].name, "Dissolve");
        assert_eq!(draft.materials.transitions[0].duration, Micros(500_000));
        assert!(draft.materials.transitions[0].is_overlap);
        assert_eq!(
            draft.tracks[0].segments[0].extra_material_refs,
            vec![draft.materials.transitions[0].id.clone()]
        );
        let error = desktop
            .split_clip(SegmentId::new("left"), Micros(800_000))
            .unwrap_err();
        assert_eq!(error.to_string(), "a transition cannot be split");
        assert_eq!(desktop.draft().unwrap().tracks[0].segments.len(), 2);
        desktop.undo().unwrap();
        assert!(desktop.draft().unwrap().materials.transitions.is_empty());
        assert!(
            desktop.draft().unwrap().tracks[0].segments[0]
                .extra_material_refs
                .is_empty()
        );
    }

    #[test]
    fn a_transition_refuses_a_gap_another_lane_and_a_locked_main_track() {
        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.tracks[0]
            .segments
            .push(Segment::new("left", "media", TimeRange::new(0, 1_000_000)));
        draft.tracks[0].segments.push(Segment::new(
            "right",
            "media",
            TimeRange::new(2_000_000, 1_000_000),
        ));
        let mut sound = Track::main("sound");
        sound.kind = TrackType::Audio;
        sound
            .segments
            .push(Segment::new("tone-a", "tone", TimeRange::new(0, 1_000_000)));
        sound.segments.push(Segment::new(
            "tone-b",
            "tone",
            TimeRange::new(1_000_000, 1_000_000),
        ));
        draft.tracks.push(sound);
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.pixel_size = 0.001;
        let card = super::TRANSITION_OFFERS[1].card_id();
        let gap = desktop
            .drop_transition(
                &card,
                lane_time(&desktop, 1_000_000.0),
                crate::tokens::LANE_HEIGHT / 2.0,
            )
            .unwrap_err();
        assert_eq!(gap.to_string(), "the drop missed the cut");
        let other = desktop
            .drop_transition(
                &card,
                lane_time(&desktop, 1_000_000.0),
                crate::tokens::LANE_HEIGHT + crate::tokens::LANE_HEIGHT / 2.0,
            )
            .unwrap_err();
        assert_eq!(other.to_string(), "a transition sits on the main track");
        assert!(desktop.draft().unwrap().materials.transitions.is_empty());

        let short = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.tracks[0]
            .segments
            .push(Segment::new("left", "media", TimeRange::new(0, 100)));
        draft.tracks[0]
            .segments
            .push(Segment::new("right", "media", TimeRange::new(100, 100)));
        Folder::new(draft).save(&short.path).unwrap();
        desktop.open_folder(&short.path).unwrap();
        desktop.pixel_size = 1.0;
        desktop
            .drop_transition(
                &super::TRANSITION_OFFERS[0].card_id(),
                lane_time(&desktop, 100.0),
                crate::tokens::LANE_HEIGHT / 2.0,
            )
            .unwrap();
        let blend = &desktop.draft().unwrap().materials.transitions[0];
        assert_eq!(blend.duration, Micros(200));
        assert!(blend.is_overlap);
        desktop.undo().unwrap();
        desktop
            .drop_transition(
                &card,
                lane_time(&desktop, 100.0),
                crate::tokens::LANE_HEIGHT / 2.0,
            )
            .unwrap();
        let blend = &desktop.draft().unwrap().materials.transitions[0];
        assert_eq!(blend.name, "Dip to black");
        assert_eq!(blend.effect_id, "cutlass-dip");
        assert_eq!(blend.duration, Micros(100));
        assert!(!blend.is_overlap);

        let track = desktop.draft().unwrap().tracks[0].id.clone();
        desktop
            .toggle_attribute(track, TrackAttribute::LOCK)
            .unwrap();
        let locked = desktop
            .drop_transition(
                &card,
                lane_time(&desktop, 100.0),
                crate::tokens::LANE_HEIGHT / 2.0,
            )
            .unwrap_err();
        assert!(matches!(locked, ShellError::Locked));
    }

    #[test]
    fn an_audio_drop_opens_a_lane_under_the_main_track() {
        let temp = Temp::new();
        empty_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let sound = temp.path.join("tone.wav");
        write_wav(&sound);
        desktop.import_file(&sound).unwrap();
        let id = desktop.draft().unwrap().materials.audios[0]
            .id
            .as_str()
            .to_owned();
        let duration = desktop.draft().unwrap().materials.audios[0].duration;
        desktop.pixel_size = 0.001;
        desktop
            .drop_medium(&id, 0.0, crate::tokens::LANE_HEIGHT / 2.0)
            .unwrap();
        let draft = desktop.draft().unwrap();
        assert_eq!(draft.tracks.len(), 2);
        assert!(draft.tracks[0].is_main());
        assert!(draft.tracks[0].segments.is_empty());
        assert_eq!(draft.tracks[1].kind, TrackType::Audio);
        assert_eq!(draft.tracks[1].segments.len(), 1);
        assert_eq!(
            draft.tracks[1].segments[0].source_range,
            Some(TimeRange::new(0, duration.0))
        );
        assert_eq!(desktop.lanes().len(), 2);
        assert!(desktop.lanes()[0].main);
        assert!(!desktop.lanes()[1].main);
        assert_eq!(desktop.lanes()[1].clips.len(), 1);
        desktop.undo().unwrap();
        assert_eq!(desktop.draft().unwrap().tracks.len(), 1);
        assert!(desktop.lanes().iter().all(|lane| lane.clips.is_empty()));
    }

    #[test]
    fn a_drop_off_the_lanes_and_a_locked_lane_leave_the_draft() {
        let temp = Temp::new();
        empty_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let picture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/engine/fixtures/pattern.png");
        desktop.import_file(&picture).unwrap();
        let id = desktop.draft().unwrap().materials.videos[0]
            .id
            .as_str()
            .to_owned();
        let revision = desktop.revision();
        let missed = desktop.drop_medium(&id, 10.0, -1.0).unwrap_err();
        assert!(missed.to_string().contains("missed"), "{missed}");
        assert_eq!(desktop.revision(), revision);
        assert!(desktop.lanes().iter().all(|lane| lane.clips.is_empty()));
        let unknown = desktop.drop_medium("missing", 0.0, 24.0).unwrap_err();
        assert!(unknown.to_string().contains("library"), "{unknown}");
        desktop
            .toggle_attribute(desktop.lanes()[0].track_id.clone(), TrackAttribute::LOCK)
            .unwrap();
        let locked = desktop
            .drop_medium(&id, 0.0, crate::tokens::LANE_HEIGHT / 2.0)
            .unwrap_err();
        assert!(locked.to_string().contains("locked"), "{locked}");
        assert!(desktop.lanes().iter().all(|lane| lane.clips.is_empty()));
        assert_eq!(desktop.draft().unwrap().tracks.len(), 1);
    }

    #[test]
    fn media_with_no_duration_is_not_placed() {
        let temp = Temp::new();
        empty_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let mut video = MaterialVideo::new("empty");
        video.duration = Micros(0);
        desktop
            .apply(cutlass_session::Request::Import {
                medium: cutlass_session::ImportedMedium::Video(Box::new(video)),
            })
            .unwrap();
        let error = desktop
            .drop_medium("empty", 0.0, crate::tokens::LANE_HEIGHT / 2.0)
            .unwrap_err();
        assert!(error.to_string().contains("no duration"), "{error}");
        assert!(desktop.lanes().iter().all(|lane| lane.clips.is_empty()));
    }

    #[test]
    fn a_real_file_imports_when_local_files_are_present() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
        if !root.is_dir() {
            eprintln!("skipped: ../local-files is missing");
            return;
        }
        let temp = Temp::new();
        empty_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let samples = [
            ("04-motion-graphics.png", true),
            ("sfx_A_sin_20260721_231942.mp3", false),
            ("snaptik_7282700172428086530_v3.mp4", true),
        ];
        let mut imported = 0;
        for (name, video) in samples {
            let path = root.join(name);
            if !path.is_file() {
                eprintln!("skipped: {name} is not in ../local-files");
                continue;
            }
            desktop.import_file(&path).unwrap();
            imported += 1;
            let draft = desktop.draft().unwrap();
            let stored = if video {
                draft
                    .materials
                    .videos
                    .iter()
                    .any(|item| item.path.ends_with(name))
            } else {
                draft
                    .materials
                    .audios
                    .iter()
                    .any(|item| item.path.ends_with(name))
            };
            assert!(stored, "{name}");
            assert!(desktop.lanes().iter().all(|lane| lane.clips.is_empty()));
            assert!(temp.path.join("Resources").join(name).is_file());
        }
        if imported == 0 {
            eprintln!("skipped: ../local-files has none of the sample names");
        }
    }

    #[test]
    fn the_playhead_seeks_from_the_ruler_and_follows_the_playback_setting() {
        let temp = Temp::new();
        color_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let revision = desktop.revision();
        let time = desktop.seek_ruler(1_000.0 * desktop.pixel_size()).unwrap();
        assert_eq!(time, Micros(1_000));
        assert_eq!(desktop.playhead(), Micros(1_000));
        assert_eq!(desktop.revision(), revision);

        desktop.play();
        desktop.set_seek_during_playback(SeekDuringPlayback::Pause);
        desktop.seek(Micros(20));
        assert!(!desktop.playing());
        desktop.play();
        desktop.set_seek_during_playback(SeekDuringPlayback::KeepPlaying);
        desktop.seek(Micros(30));
        assert!(desktop.playing());
        assert_eq!(desktop.revision(), revision);
    }

    /// A project with one picture clip `length` long on the main track.
    fn span_shell(temp: &Temp, length: i64) -> Shell {
        let mut draft = Draft::empty("draft");
        let mut video = MaterialVideo::new("picture");
        video.path = "clip.mp4".to_owned();
        draft.materials.videos.push(video);
        let mut picture = Segment::new("picture", "picture", TimeRange::new(0, length));
        picture.source_range = Some(TimeRange::new(0, length));
        draft.tracks[0].segments.push(picture);
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop
    }

    /// A main row with clips at 0–1, 2–3 and 5–6 seconds, magnet off.
    fn gappy_shell(temp: &Temp) -> Shell {
        let mut draft = Draft::empty("draft");
        draft.config.maintrack_adsorb = false;
        let mut video = MaterialVideo::new("picture");
        video.path = "clip.mp4".to_owned();
        draft.materials.videos.push(video);
        for (id, start) in [("a", 0), ("b", 2_000_000), ("c", 5_000_000)] {
            let mut clip = Segment::new(id, "picture", TimeRange::new(start, 1_000_000));
            clip.source_range = Some(TimeRange::new(0, 1_000_000));
            draft.tracks[0].segments.push(clip);
        }
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop
    }

    fn main_starts(desktop: &Shell) -> Vec<i64> {
        desktop.draft().unwrap().tracks[0]
            .segments
            .iter()
            .map(|segment| segment.target.start.0)
            .collect()
    }

    #[test]
    fn a_clip_key_moves_to_the_nearest_frame_and_not_onto_another_key() {
        let temp = Temp::new();
        let mut desktop = gappy_shell(&temp);
        let b = SegmentId::new("b");
        assert!(desktop.clip_key_times(&b).is_empty());
        desktop.select(Some(b.clone()));
        let picture = desktop.draft().unwrap().tracks[0].segments[1].clone();
        let mut keyed = picture.clone();
        for offset in [0, 500_000] {
            cutlass_model::put_key(&mut keyed, "KFTypePositionX", Micros(offset), 0.0);
        }
        desktop
            .apply(super::Request::SetKeyframes {
                segment_id: b.clone(),
                keyframes: keyed.common_keyframes,
            })
            .unwrap();
        assert_eq!(
            desktop.clip_key_times(&b),
            [Micros(2_000_000), Micros(2_500_000)]
        );
        // 2.71 s lands on the frame at 2.7 s, at 30 frames a second.
        desktop
            .move_clip_key(&b, Micros(2_500_000), Micros(2_710_000))
            .unwrap();
        assert_eq!(
            desktop.clip_key_times(&b),
            [Micros(2_000_000), Micros(2_700_000)]
        );
        assert_eq!(
            desktop
                .move_clip_key(&b, Micros(2_700_000), Micros(2_000_000))
                .unwrap_err()
                .to_string(),
            "a key is already there"
        );
        // Past the clip's end it stops at the end.
        desktop
            .move_clip_key(&b, Micros(2_700_000), Micros(9_000_000))
            .unwrap();
        assert_eq!(desktop.clip_key_times(&b)[1], Micros(3_000_000));
        let revision = desktop.revision();
        desktop
            .move_clip_key(&b, Micros(3_000_000), Micros(3_000_000))
            .unwrap();
        assert_eq!(desktop.revision(), revision);
        desktop.undo().unwrap();
        assert_eq!(desktop.clip_key_times(&b)[1], Micros(2_700_000));
    }

    #[test]
    fn ripple_delete_and_close_gap_pull_the_later_clips_left() {
        let temp = Temp::new();
        let mut desktop = gappy_shell(&temp);
        assert_eq!(
            desktop.ripple_delete().unwrap_err().to_string(),
            "nothing is selected"
        );
        desktop.select(Some(SegmentId::new("a")));
        desktop.run_shortcut("shift+delete").unwrap();
        assert_eq!(main_starts(&desktop), [1_000_000, 4_000_000]);
        assert!(desktop.selection().is_empty());
        desktop.undo().unwrap();
        assert_eq!(main_starts(&desktop), [0, 2_000_000, 5_000_000]);

        let main = desktop.draft().unwrap().tracks[0].id.clone();
        assert!(!desktop.open_gap_menu(&main, Micros(500_000), 10.0, 20.0));
        assert!(desktop.open_gap_menu(&main, Micros(4_000_000), 10.0, 20.0));
        assert_eq!(
            desktop.gap_menu().map(|menu| menu.time),
            Some(Micros(4_000_000))
        );
        desktop.close_gap(&main, Micros(4_000_000)).unwrap();
        assert_eq!(main_starts(&desktop), [0, 2_000_000, 3_000_000]);
        assert_eq!(
            desktop
                .close_gap(&main, Micros(4_000_000))
                .unwrap_err()
                .to_string(),
            "there is no gap there"
        );
        desktop
            .toggle_attribute(main.clone(), TrackAttribute::LOCK)
            .unwrap();
        assert!(matches!(
            desktop.close_gap(&main, Micros(1_500_000)),
            Err(ShellError::Locked)
        ));
    }

    #[test]
    fn the_shuttle_speeds_up_runs_backward_and_stops_at_either_end() {
        use cutlass_editor::Shuttle;
        let temp = Temp::new();
        let mut desktop = span_shell(&temp, 10_000_000);
        desktop.shuttle(Shuttle::Forward);
        assert_eq!(desktop.shuttle_rate(), 1);
        desktop.shuttle(Shuttle::Forward);
        desktop.shuttle(Shuttle::Forward);
        assert_eq!(desktop.shuttle_rate(), 4);
        desktop.advance(1_000_000);
        assert_eq!(desktop.playhead(), Micros(4_000_000));
        desktop.shuttle(Shuttle::Back);
        desktop.shuttle(Shuttle::Back);
        assert_eq!(desktop.shuttle_rate(), -2);
        desktop.advance(1_000_000);
        assert_eq!(desktop.playhead(), Micros(2_000_000));
        desktop.advance(5_000_000);
        assert_eq!(desktop.playhead(), Micros(0));
        assert!(!desktop.playing());
        // Backward from the start has nowhere to go.
        desktop.shuttle(Shuttle::Back);
        assert!(!desktop.playing());
        desktop.shuttle(Shuttle::Forward);
        desktop.shuttle(Shuttle::Forward);
        desktop.advance(20_000_000);
        assert_eq!(desktop.playhead(), Micros(10_000_000));
        assert!(!desktop.playing());
        desktop.shuttle(Shuttle::Forward);
        desktop.shuttle(Shuttle::Stop);
        assert_eq!(desktop.shuttle_rate(), 0);
        assert!(!desktop.playing());
        // Space after a shuttle plays in real time.
        desktop.toggle_play();
        assert_eq!(desktop.shuttle_rate(), 1);
    }

    #[test]
    fn the_keys_mark_in_and_out_and_export_the_marked_span() {
        let temp = Temp::new();
        let mut desktop = span_shell(&temp, 10_000_000);
        for (keys, action) in [
            ("j", "shuttle_back"),
            ("k", "shuttle_stop"),
            ("l", "shuttle_forward"),
            ("i", "mark_in"),
            ("o", "mark_out"),
            ("alt+x", "clear_marks"),
        ] {
            assert!(
                desktop
                    .settings
                    .shortcuts
                    .iter()
                    .any(|shortcut| shortcut.keys == keys && shortcut.action == action),
                "{keys}"
            );
        }
        let draft = desktop.draft().unwrap().clone();
        assert_eq!(
            desktop
                .export_range(&draft, super::ExportSpan::Marked)
                .unwrap_err()
                .to_string(),
            "mark an in or an out first"
        );
        desktop.seek(Micros(2_000_000));
        desktop.run_shortcut("i").unwrap();
        desktop.seek(Micros(6_000_000));
        desktop.run_shortcut("o").unwrap();
        assert_eq!(
            desktop
                .export_range(&draft, super::ExportSpan::Marked)
                .unwrap(),
            TimeRange::new(2_000_000, 4_000_000)
        );
        desktop.run_shortcut("alt+x").unwrap();
        assert_eq!(desktop.marked_span(), None);
        desktop.run_shortcut("o").unwrap();
        assert_eq!(desktop.marked_span(), Some((Micros(0), Micros(6_000_000))));
        desktop.close_project().unwrap();
        assert_eq!(desktop.marks(), cutlass_editor::Marks::default());
    }

    #[test]
    fn play_on_an_empty_cut_stays_at_the_start() {
        let temp = Temp::new();
        empty_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.play();
        assert!(!desktop.playing());
        assert_eq!(desktop.playhead(), Micros(0));
        desktop.playing = true;
        desktop.playhead = Micros(2_000_000);
        desktop.advance(1_000_000);
        assert!(!desktop.playing());
        assert_eq!(desktop.playhead(), Micros(0));
    }

    #[test]
    fn play_at_the_end_starts_again_from_zero() {
        let temp = Temp::new();
        color_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.seek(Micros(1_000_000));
        desktop.play();
        assert!(desktop.playing());
        assert_eq!(desktop.playhead(), Micros(0));
        desktop.advance(1_000_000);
        assert!(!desktop.playing());
        assert_eq!(desktop.playhead(), Micros(1_000_000));
    }

    #[test]
    fn playback_keeps_a_zoomed_playhead_in_the_lane() {
        let temp = Temp::new();
        color_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.set_lane_width(800.0);
        desktop.play();
        desktop.advance(400_000);
        let origin = desktop.view_origin().0;
        let play = desktop.playhead().0;
        let visible = (800.0 / desktop.pixel_size()).round() as i64;
        assert!(origin > 0, "origin {origin}");
        assert!(
            play >= origin && play <= origin + visible,
            "play {play} origin {origin} visible {visible}"
        );
        let at_left = desktop.seek_ruler(0.0).unwrap();
        assert!(at_left.0 > 0, "{at_left:?}");
        assert!(at_left.0 <= desktop.view_origin().0.saturating_add(1));
        desktop.seek(Micros(0));
        assert_eq!(desktop.view_origin(), Micros(0));
        desktop.zoom_fit().unwrap();
        desktop.advance(400_000);
        assert_eq!(
            desktop.view_origin(),
            Micros(0),
            "a cut that fits the lane stays at the start"
        );
    }

    #[test]
    fn the_end_of_the_cut_keeps_the_last_picture() {
        let temp = Temp::new();
        color_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let start = desktop.preview_bytes().unwrap();
        desktop.seek(desktop.cut_duration());
        assert_eq!(desktop.playhead(), Micros(1_000_000));
        let end = desktop.preview_bytes().unwrap();
        assert_eq!(end.width, start.width);
        assert_eq!(end.height, start.height);
        assert_eq!(end.rgba, start.rgba);
    }

    #[test]
    fn space_toggles_play_and_the_arrows_step_one_frame() {
        let temp = Temp::new();
        color_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.run_shortcut("space").unwrap();
        assert!(desktop.playing());
        desktop.run_shortcut("space").unwrap();
        assert!(!desktop.playing());
        desktop.run_shortcut("right").unwrap();
        let frame = FrameRate::FPS_30.frame_start(1).unwrap();
        assert_eq!(desktop.playhead(), frame);
        desktop.run_shortcut("left").unwrap();
        assert_eq!(desktop.playhead(), Micros(0));
        desktop.run_shortcut("left").unwrap();
        assert_eq!(desktop.playhead(), Micros(0));
        desktop.seek(Micros(1_000_000));
        desktop.run_shortcut("right").unwrap();
        assert_eq!(desktop.playhead(), Micros(1_000_000));
        let id = desktop.draft().unwrap().tracks[0].segments[0].id.clone();
        desktop.select(Some(id));
        desktop.run_shortcut("delete").unwrap();
        assert!(desktop.lanes().iter().all(|lane| lane.clips.is_empty()));
        desktop.run_shortcut("delete").unwrap();
    }

    #[test]
    fn play_and_scrub_read_the_software_frame_without_an_undo_step() {
        let temp = Temp::new();
        color_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let revision = desktop.revision();
        desktop.play();
        assert!(desktop.audio_output_open());
        let colored = desktop.preview_bytes().unwrap();
        assert_eq!(colored.width, 2);
        assert_eq!(colored.height, 1);
        assert_eq!(
            colored.rgba,
            [0x33, 0x66, 0x99, 0xFF, 0x33, 0x66, 0x99, 0xFF]
        );
        desktop.advance(2_000_000);
        let clear = desktop.scrub(Micros(2_000_000)).unwrap();
        assert_eq!(clear.rgba, vec![0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(desktop.revision(), revision);
        desktop.pause();
        assert!(!desktop.playing());
        assert_eq!(desktop.revision(), revision);
    }

    fn write_tone(path: &Path) {
        let mut payload = Vec::new();
        for _ in 0..4 {
            payload.extend_from_slice(&32_767_i16.to_le_bytes());
        }
        let data_bytes = u32::try_from(payload.len()).expect("wav size");
        let rate = 48_000_u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&rate.to_le_bytes());
        bytes.extend_from_slice(&(rate * 2).to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_bytes.to_le_bytes());
        bytes.extend_from_slice(&payload);
        std::fs::write(path, bytes).expect("wav");
    }

    fn tone_project(dir: &Path, stored: &str) {
        let mut draft = Draft::empty("draft");
        draft.duration = Micros(1_000_000);
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("tone"),
            path: stored.to_owned(),
            duration: Micros(1_000_000),
            ..MaterialAudio::default()
        });
        let mut track = Track::main("audio");
        track.kind = TrackType::Audio;
        let mut segment = Segment::new("clip", "tone", TimeRange::new(0, 1_000_000));
        segment.volume = 0.5;
        segment.source_range = Some(TimeRange::new(0, 1_000_000));
        track.segments.push(segment);
        draft.tracks.push(track);
        Folder::new(draft).save(dir).unwrap();
    }

    fn preview_samples(bytes: &[u8]) -> Vec<f32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect()
    }

    #[test]
    fn play_queues_the_mixed_tone_and_a_mute_or_linux_host_stays_silent() {
        let temp = Temp::new();
        write_tone(&temp.path.join("tone.wav"));
        tone_project(&temp.path, "tone.wav");
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let revision = desktop.revision();
        desktop.play();
        assert!(desktop.audio_output_open());
        assert_eq!(preview_samples(desktop.queued_preview()), vec![0.5; 8]);
        assert_eq!(desktop.revision(), revision);
        let picture = desktop.preview_bytes().unwrap();
        assert_eq!(picture.width, 2);
        assert_eq!(picture.height, 1);

        desktop.pause();
        desktop.seek(Micros(0));
        assert!(desktop.queued_preview().is_empty());
        desktop.set_scrub_audio(true);
        desktop.seek(Micros(0));
        assert_eq!(preview_samples(desktop.queued_preview()), vec![0.5; 8]);
        assert_eq!(desktop.revision(), revision);

        let audio = desktop
            .lanes()
            .iter()
            .find(|lane| !lane.main)
            .map(|lane| lane.track_id.clone())
            .expect("audio lane");
        desktop
            .toggle_attribute(audio, TrackAttribute::MUTE)
            .unwrap();
        desktop.play();
        assert!(desktop.queued_preview().is_empty());

        desktop.open_preview_audio(AudioHost::Linux);
        assert!(!desktop.audio_output_open());
        desktop.play();
        assert!(desktop.queued_preview().is_empty());
        assert!(desktop.preview_bytes().is_ok());
    }

    #[test]
    fn a_path_that_leaves_the_project_does_not_play() {
        let temp = Temp::new();
        let name = format!(
            "escape-{}.wav",
            temp.path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap()
        );
        let outside = temp.path.join("..").join(&name);
        write_tone(&outside);
        tone_project(&temp.path, &format!("../{name}"));
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let revision = desktop.revision();
        desktop.play();
        assert!(desktop.queued_preview().is_empty());
        assert_eq!(desktop.revision(), revision);
        let _ = std::fs::remove_file(outside);
    }

    #[test]
    fn a_local_wav_queues_when_one_is_present() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
        let Some(tone) = std::fs::read_dir(&root).ok().and_then(|entries| {
            entries.flatten().find_map(|entry| {
                let path = entry.path();
                if path.extension().and_then(|ext| ext.to_str()) != Some("wav") {
                    return None;
                }
                let len = entry.metadata().ok()?.len();
                (len > 0 && len <= 256 * 1024).then_some(path)
            })
        }) else {
            eprintln!("skipped: ../local-files has no small wav");
            return;
        };
        let temp = Temp::new();
        tone_project(&temp.path, &tone.display().to_string());
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let revision = desktop.revision();
        desktop.play();
        assert!(desktop.queued_preview().len().is_multiple_of(8));
        assert_eq!(desktop.revision(), revision);
        assert!(desktop.preview_bytes().is_ok());
    }

    #[test]
    fn a_second_paint_does_not_open_the_movie() {
        MOVIE_OPENS.store(0, Ordering::Relaxed);
        let temp = Temp::new();
        let movie = temp.path.join("clip.bin");
        std::fs::write(&movie, b"12345678").unwrap();
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.materials.material_colors.push(MaterialColorConfig {
            id: "color".into(),
            is_color_clip: true,
            solid_color: "0xFF336699".to_owned(),
            ..MaterialColorConfig::default()
        });
        draft.tracks[0]
            .segments
            .push(Segment::new("color", "color", TimeRange::new(0, 1_000_000)));
        let mut video = MaterialVideo::new("movie");
        video.path = movie.display().to_string();
        draft.materials.videos.push(video);
        let mut segment = Segment::new("movie", "movie", TimeRange::new(0, 1_000_000));
        segment.source_range = Some(TimeRange::new(0, 1_000_000));
        segment.render_index = 1;
        draft.tracks[0].segments.push(segment);
        Folder::new(draft).save(&temp.path).unwrap();

        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let mut readers = ReaderFactory::new();
        readers.register_container(probe_counted, open_counted);
        desktop.set_readers(readers);

        let first = desktop.preview_bytes().unwrap();
        assert_eq!(MOVIE_OPENS.load(Ordering::Relaxed), 1);
        assert_eq!(first.rgba, [0x33, 0x66, 0x99, 0xFF, 0x33, 0x66, 0x99, 0xFF]);
        let second = desktop.preview_bytes().unwrap();
        assert_eq!(MOVIE_OPENS.load(Ordering::Relaxed), 1);
        assert_eq!(second.rgba, first.rgba);

        for (step, time) in [10_i64, 20, 30, 40].into_iter().enumerate() {
            desktop.seek(Micros(time));
            let picture = desktop.preview_bytes().unwrap();
            assert_eq!(picture.rgba, first.rgba);
            assert_eq!(MOVIE_OPENS.load(Ordering::Relaxed), step as u32 + 2);
        }
        desktop.seek(Micros(0));
        let again = desktop.preview_bytes().unwrap();
        assert_eq!(again.rgba, first.rgba);
        assert_eq!(MOVIE_OPENS.load(Ordering::Relaxed), 6);

        desktop.seek(Micros(2_000_000));
        let outside = desktop.preview_bytes().unwrap();
        assert_eq!(outside.rgba, vec![0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(MOVIE_OPENS.load(Ordering::Relaxed), 6);

        let track = desktop.draft().unwrap().tracks[0].id.clone();
        desktop.seek(Micros(0));
        desktop
            .toggle_attribute(track, TrackAttribute::HIDE)
            .unwrap();
        let hidden = desktop.preview_bytes().unwrap();
        assert_eq!(hidden.rgba, vec![0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(MOVIE_OPENS.load(Ordering::Relaxed), 6);
    }

    #[test]
    fn a_new_time_seeks_the_open_movie() {
        KEPT_OPENS.store(0, Ordering::Relaxed);
        let temp = Temp::new();
        let movie = temp.path.join("clip.bin");
        std::fs::write(&movie, b"12345678").unwrap();
        let mut draft = Draft::empty("draft");
        draft.config.maintrack_adsorb = false;
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 2;
        draft.materials.material_colors.push(MaterialColorConfig {
            id: "color".into(),
            is_color_clip: true,
            solid_color: "0xFF336699".to_owned(),
            ..MaterialColorConfig::default()
        });
        draft.tracks[0]
            .segments
            .push(Segment::new("color", "color", TimeRange::new(0, 1_000_000)));
        let mut video = MaterialVideo::new("movie");
        video.path = movie.display().to_string();
        draft.materials.videos.push(video);
        let mut segment = Segment::new("movie", "movie", TimeRange::new(0, 1_000_000));
        segment.source_range = Some(TimeRange::new(0, 1_000_000));
        segment.render_index = 1;
        draft.tracks[0].segments.push(segment);
        Folder::new(draft).save(&temp.path).unwrap();

        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let mut readers = ReaderFactory::new();
        readers.register_container(probe_counted, open_kept);
        readers.decoders_mut().register(CodecId::Jpeg, open_flat);
        desktop.set_readers(readers);

        // The clip and its decode thread each open the movie.
        let first = desktop.preview_bytes().unwrap();
        assert_eq!(KEPT_OPENS.load(Ordering::Relaxed), 4);
        assert_ne!(first.rgba, [0x33, 0x66, 0x99, 0xFF].repeat(4));
        assert!(first.rgba.iter().any(|byte| *byte != 0));

        desktop.seek(Micros(10));
        let later = desktop.preview_bytes().unwrap();
        assert_eq!(KEPT_OPENS.load(Ordering::Relaxed), 4);
        assert_eq!(later.rgba, first.rgba);

        let track = desktop.draft().unwrap().tracks[0].id.clone();
        desktop
            .toggle_attribute(track.clone(), TrackAttribute::MUTE)
            .unwrap();
        desktop.seek(Micros(20));
        let muted = desktop.preview_bytes().unwrap();
        assert_eq!(KEPT_OPENS.load(Ordering::Relaxed), 8);
        assert_eq!(muted.rgba, first.rgba);

        desktop
            .toggle_attribute(track, TrackAttribute::HIDE)
            .unwrap();
        let hidden = desktop.preview_bytes().unwrap();
        assert_eq!(KEPT_OPENS.load(Ordering::Relaxed), 8);
        assert_eq!(hidden.rgba, vec![0; 16]);
    }

    #[test]
    fn undo_uses_the_stored_shortcut_and_does_not_restore_it() {
        let temp = Temp::new();
        color_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        assert!(matches!(
            desktop.undo(),
            Err(ShellError::Edit(EditError::EmptyUndo))
        ));
        desktop.select(Some(SegmentId::new("segment")));
        desktop.set_volume(0.25).unwrap();
        desktop.set_shortcut("undo", "primary+y");
        let error = desktop.run_shortcut("primary+z").unwrap_err();
        assert!(error.to_string().contains("not available"), "{error}");
        assert_eq!(desktop.draft().unwrap().tracks[0].segments[0].volume, 0.25);
        desktop.run_shortcut("primary+y").unwrap();
        assert_eq!(desktop.draft().unwrap().tracks[0].segments[0].volume, 1.0);
        assert_eq!(
            desktop
                .settings()
                .shortcuts
                .iter()
                .find(|shortcut| shortcut.action == "undo")
                .unwrap()
                .keys,
            "primary+y"
        );
    }

    #[test]
    fn lane_header_bits_are_attributes_and_a_locked_lane_refuses_a_drag() {
        let temp = Temp::new();
        color_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let track = desktop.lanes()[0].track_id.clone();
        desktop
            .toggle_attribute(track.clone(), TrackAttribute::MUTE)
            .unwrap();
        assert!(desktop.lanes()[0].mute);
        assert_eq!(desktop.draft().unwrap().tracks[0].segments[0].volume, 1.0);
        desktop
            .toggle_attribute(track, TrackAttribute::LOCK)
            .unwrap();
        assert!(desktop.lanes()[0].lock);
        let start = desktop.lanes()[0].clips[0].start;
        let error = desktop
            .drag_segment(
                SegmentId::new("segment"),
                50_000.0 * desktop.pixel_size(),
                0.0,
            )
            .unwrap_err();
        assert!(
            matches!(
                error,
                ShellError::Edit(EditError::LockedTrack) | ShellError::Locked
            ),
            "{error}"
        );
        assert_eq!(desktop.lanes()[0].clips[0].start, start);
    }

    #[test]
    fn the_meter_rests_while_paused_and_hears_the_mix_while_playing() {
        let temp = Temp::new();
        color_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.update_meter(20_000);
        assert!(desktop.meter_resting());
        desktop.play();
        desktop.advance(20_000);
        desktop.update_meter(20_000);
        // A picture with no sound plays silence, so the meter stays down.
        assert!(desktop.meter_resting());
        desktop.pause();
        desktop.update_meter(20_000);
        assert!(desktop.meter_resting());
    }

    #[test]
    fn library_folders_are_kept_with_the_project() {
        let temp = Temp::new();
        empty_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let picture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/engine/fixtures/pattern.png");
        desktop.import_file(&picture).unwrap();
        let media = |desktop: &Shell, folder: &str| {
            desktop.library_files(
                cutlass_editor::LibraryShelf::Media,
                folder,
                cutlass_editor::LibrarySort::Name,
                "",
            )
        };
        let card = media(&desktop, "")[0].id.clone();
        let folder = desktop
            .new_library_folder("", std::slice::from_ref(&card))
            .unwrap();
        desktop.rename_library_folder(&folder, "Takes").unwrap();
        assert!(media(&desktop, "").is_empty());
        assert_eq!(media(&desktop, &folder).len(), 1);
        assert!(desktop.unsaved());
        desktop.save().unwrap();
        assert!(!desktop.unsaved());
        drop(desktop);

        let mut again = shell();
        again.open_folder(&temp.path).unwrap();
        assert_eq!(again.library_store().folders[0].display_name, "Takes");
        assert_eq!(media(&again, &folder).len(), 1);
        assert!(again.move_library_item(&card, "").unwrap());
        assert_eq!(
            again.notice(),
            format!("Moved {} to All", media(&again, "")[0].name)
        );
        assert_eq!(
            again.delete_library_folder(&folder).unwrap().as_deref(),
            Some("")
        );
        assert!(again.library_store().folders.is_empty());
        assert!(again.unsaved());
    }

    #[test]
    fn a_track_is_added_named_and_deleted_and_stays_while_empty() {
        let temp = Temp::new();
        color_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let main = desktop.lanes()[0].track_id.clone();
        let sound = desktop.add_track(TrackType::Audio).unwrap();
        let over = desktop.add_track(TrackType::Video).unwrap();
        // Empty rows stay on the desktop: the overlay above, the sound under.
        let order: Vec<_> = desktop
            .lanes()
            .iter()
            .map(|lane| lane.track_id.clone())
            .collect();
        assert_eq!(order, [over.clone(), main.clone(), sound.clone()]);

        desktop.rename_track(&sound, "  Voice ").unwrap();
        assert_eq!(desktop.lanes()[2].name.as_deref(), Some("Voice"));
        let revision = desktop.session.as_ref().unwrap().revision();
        desktop.rename_track(&sound, "Voice").unwrap();
        assert_eq!(desktop.session.as_ref().unwrap().revision(), revision);
        desktop.rename_track(&sound, " ").unwrap();
        assert_eq!(desktop.lanes()[2].name, None);

        desktop.remove_track(&over).unwrap();
        assert_eq!(desktop.lanes().len(), 2);
        desktop.undo().unwrap();
        assert_eq!(desktop.lanes().len(), 3);
        assert!(matches!(
            desktop.remove_track(&main).unwrap_err(),
            ShellError::Edit(EditError::MainTrack)
        ));
    }

    #[test]
    fn the_first_clip_fits_the_lane_and_the_next_keeps_the_scale() {
        let temp = Temp::new();
        empty_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.set_lane_width(800.0);
        let picture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/engine/fixtures/pattern.png");
        desktop.import_file(&picture).unwrap();
        let id = desktop.draft().unwrap().materials.videos[0]
            .id
            .as_str()
            .to_owned();
        let duration = desktop.draft().unwrap().materials.videos[0].duration;
        let opening = desktop.pixel_size();
        desktop
            .drop_medium(&id, 0.0, crate::tokens::LANE_HEIGHT / 2.0)
            .unwrap();
        let fitted = cutlass_editor::fit_pixel_size(800.0, duration).unwrap();
        assert!((desktop.pixel_size() - fitted).abs() < 0.000_000_1);
        assert!(desktop.pixel_size() < opening);
        let kept = desktop.pixel_size();
        desktop
            .drop_medium(&id, 0.0, crate::tokens::LANE_HEIGHT / 2.0)
            .unwrap();
        assert_eq!(desktop.pixel_size(), kept);
        assert_eq!(desktop.draft().unwrap().tracks.len(), 2);
    }

    #[test]
    fn a_saved_cut_fits_once_when_the_lane_is_shown() {
        let temp = Temp::new();
        color_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let opening = desktop.pixel_size();
        desktop.prepare_lane(800.0);
        let fitted = cutlass_editor::fit_pixel_size(800.0, Micros(1_000_000)).unwrap();
        assert!((desktop.pixel_size() - fitted).abs() < 0.000_000_1);
        assert!(desktop.pixel_size() < opening);
        desktop.zoom_in().unwrap();
        let closer = desktop.pixel_size();
        desktop.prepare_lane(900.0);
        assert_eq!(desktop.pixel_size(), closer);
    }

    #[test]
    fn zoom_stays_on_the_editor_scale_and_fit_refuses_an_empty_duration() {
        let temp = Temp::new();
        empty_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let size = desktop.pixel_size();
        assert!(desktop.zoom_fit().is_err());
        assert_eq!(desktop.pixel_size(), size);

        color_project(&temp.path);
        // The folder is already held by this shell. A second project is a new folder.
        let longer = Temp::new();
        color_project(&longer.path);
        desktop.open_folder(&longer.path).unwrap();
        desktop.set_lane_width(800.0);
        let before = desktop.pixel_size();
        desktop.zoom_out().unwrap();
        assert!(desktop.pixel_size() < before);
        let floor = desktop.minimum_pixel_size().unwrap();
        for _ in 0..8 {
            desktop.zoom_out().unwrap();
        }
        assert!(desktop.pixel_size() + f64::EPSILON >= floor);
        desktop.zoom_fit().unwrap();
        let fitted = cutlass_editor::fit_pixel_size(800.0, Micros(1_000_000)).unwrap();
        assert!(
            (desktop.pixel_size() - fitted).abs() < 0.000_000_1 || desktop.pixel_size() >= floor
        );
    }

    #[test]
    fn a_picture_moves_and_scales_and_a_sound_does_not() {
        let mut bare = shell();
        assert!(!bare.place_enabled());
        assert_eq!(
            bare.set_position(0.0, 0.0).unwrap_err().to_string(),
            "no project is open"
        );

        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        let mut video = MaterialVideo::new("picture");
        video.path = "clip.mp4".to_owned();
        draft.materials.videos.push(video);
        let mut picture = Segment::new("picture", "picture", TimeRange::new(0, 1_000_000));
        picture.clip.flip.horizontal = true;
        draft.tracks[0].segments.push(picture);
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("tone"),
            path: "tone".to_owned(),
            ..MaterialAudio::default()
        });
        let mut sound = Track::main("tone");
        sound.kind = TrackType::Audio;
        sound
            .segments
            .push(Segment::new("tone", "tone", TimeRange::new(0, 500_000)));
        draft.tracks.push(sound);
        draft.materials.texts.push(MaterialText {
            id: MaterialId::new("caption"),
            content: "hi".to_owned(),
            ..MaterialText::default()
        });
        let mut words = Track::main("words");
        words.kind = TrackType::Text;
        words.segments.push(Segment::new(
            "caption",
            "caption",
            TimeRange::new(0, 500_000),
        ));
        draft.tracks.push(words);
        Folder::new(draft).save(&temp.path).unwrap();

        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.select(None);
        assert_eq!(
            desktop.set_position(0.0, 0.0).unwrap_err().to_string(),
            "nothing is selected"
        );
        desktop.select(Some(SegmentId::new("tone")));
        assert!(!desktop.place_enabled());
        assert_eq!(
            desktop.set_scale(2.0).unwrap_err().to_string(),
            "a placement belongs on a picture"
        );

        desktop.select(Some(SegmentId::new("picture")));
        assert!(desktop.place_enabled());
        desktop.set_position(0.5, -0.25).unwrap();
        let inspection = desktop.inspection().unwrap();
        assert_eq!(inspection.transform_x, 0.5);
        assert_eq!(inspection.transform_y, -0.25);
        assert!(
            desktop.draft().unwrap().tracks[0].segments[0]
                .clip
                .flip
                .horizontal
        );
        desktop.undo().unwrap();
        assert_eq!(desktop.inspection().unwrap().transform_x, 0.0);
        desktop.set_scale(2.0).unwrap();
        assert_eq!(desktop.inspection().unwrap().scale_x, 2.0);
        assert_eq!(desktop.inspection().unwrap().scale_y, 2.0);
        assert_eq!(
            desktop.set_scale(0.0).unwrap_err().to_string(),
            "the scale must be greater than zero"
        );
        assert_eq!(
            desktop.set_position(f64::NAN, 0.0).unwrap_err().to_string(),
            "the position is not a finite number"
        );
        assert_eq!(desktop.inspection().unwrap().scale_x, 2.0);

        let main = desktop.draft().unwrap().tracks[0].id.clone();
        desktop
            .toggle_attribute(main, TrackAttribute::LOCK)
            .unwrap();
        assert_eq!(
            desktop.set_position(1.0, 0.0).unwrap_err().to_string(),
            "the track is locked"
        );

        desktop.select(Some(SegmentId::new("caption")));
        assert!(desktop.place_enabled());
        desktop.set_position(0.1, 0.2).unwrap();
        assert_eq!(desktop.inspection().unwrap().transform_y, 0.2);
    }

    #[test]
    fn a_clip_fades_in_and_out() {
        let mut bare = shell();
        assert!(!bare.fade_enabled());
        assert_eq!(
            bare.set_fade_in(0.5).unwrap_err().to_string(),
            "no project is open"
        );

        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        let mut video = MaterialVideo::new("picture");
        video.path = "clip.mp4".to_owned();
        draft.materials.videos.push(video);
        draft.tracks[0].segments.push(Segment::new(
            "picture",
            "picture",
            TimeRange::new(0, 1_000_000),
        ));
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("tone"),
            path: "tone".to_owned(),
            ..MaterialAudio::default()
        });
        let mut sound = Track::main("tone");
        sound.kind = TrackType::Audio;
        sound
            .segments
            .push(Segment::new("tone", "tone", TimeRange::new(0, 500_000)));
        draft.tracks.push(sound);
        draft.materials.texts.push(MaterialText {
            id: MaterialId::new("caption"),
            content: "hi".to_owned(),
            ..MaterialText::default()
        });
        let mut words = Track::main("words");
        words.kind = TrackType::Text;
        words.segments.push(Segment::new(
            "caption",
            "caption",
            TimeRange::new(0, 500_000),
        ));
        draft.tracks.push(words);
        Folder::new(draft).save(&temp.path).unwrap();

        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.select(None);
        assert_eq!(
            desktop.set_fade_out(0.1).unwrap_err().to_string(),
            "nothing is selected"
        );
        desktop.select(Some(SegmentId::new("caption")));
        assert!(!desktop.fade_enabled());
        assert_eq!(
            desktop.set_fade_in(0.1).unwrap_err().to_string(),
            "a fade belongs on a picture or a sound"
        );

        desktop.select(Some(SegmentId::new("picture")));
        assert!(desktop.fade_enabled());
        assert_eq!(desktop.inspection().unwrap().fade_in, 0);
        desktop.set_fade_in(0.5).unwrap();
        let inspection = desktop.inspection().unwrap();
        assert_eq!(inspection.fade_in, 500_000);
        assert_eq!(inspection.fade_out, 0);
        assert_eq!(
            desktop.draft().unwrap().materials.audio_fades[0].fade_type,
            2
        );
        assert_eq!(desktop.draft().unwrap().tracks[0].segments[0].volume, 1.0);
        desktop.set_fade_out(0.25).unwrap();
        let inspection = desktop.inspection().unwrap();
        assert_eq!(inspection.fade_in, 500_000);
        assert_eq!(inspection.fade_out, 250_000);
        desktop.undo().unwrap();
        assert_eq!(desktop.inspection().unwrap().fade_out, 0);
        assert_eq!(desktop.inspection().unwrap().fade_in, 500_000);
        assert_eq!(
            desktop.set_fade_in(-1.0).unwrap_err().to_string(),
            "a fade cannot be negative"
        );
        assert_eq!(desktop.inspection().unwrap().fade_in, 500_000);

        desktop.select(Some(SegmentId::new("tone")));
        assert!(desktop.fade_enabled());
        desktop.set_fade_out(1.0).unwrap();
        assert_eq!(desktop.inspection().unwrap().fade_out, 1_000_000);
        assert_eq!(desktop.inspection().unwrap().fade_in, 0);
        assert_eq!(desktop.draft().unwrap().materials.audio_fades.len(), 2);
        let sound_track = desktop.draft().unwrap().tracks[1].id.clone();
        desktop
            .toggle_attribute(sound_track, TrackAttribute::LOCK)
            .unwrap();
        assert_eq!(
            desktop.set_fade_in(0.2).unwrap_err().to_string(),
            "the track is locked"
        );
        assert_eq!(desktop.inspection().unwrap().fade_out, 1_000_000);
    }

    #[test]
    fn a_picture_takes_in_and_out_animations_inside_its_length() {
        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        let mut video = MaterialVideo::new("picture");
        video.path = "clip.mp4".to_owned();
        video.width = 640;
        video.height = 360;
        draft.materials.videos.push(video);
        let mut picture = Segment::new("picture", "picture", TimeRange::new(0, 1_000_000));
        picture.source_range = Some(TimeRange::new(0, 1_000_000));
        draft.tracks[0].segments.push(picture);
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("tone"),
            path: "tone".to_owned(),
            ..MaterialAudio::default()
        });
        let mut sound = Track::main("tone");
        sound.kind = TrackType::Audio;
        sound
            .segments
            .push(Segment::new("tone", "tone", TimeRange::new(0, 500_000)));
        draft.tracks.push(sound);
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();

        desktop.select(Some(SegmentId::new("tone")));
        assert_eq!(desktop.animations(), None);
        assert_eq!(
            desktop
                .set_animation(
                    cutlass_model::AnimType::In,
                    Some("cutlass-anim-in-slide-up".to_owned()),
                    Micros(1)
                )
                .unwrap_err()
                .to_string(),
            "only a picture or title takes animations"
        );

        desktop.select(Some(SegmentId::new("picture")));
        // A title's letter animation does not go on a picture.
        assert!(
            desktop
                .set_animation(
                    cutlass_model::AnimType::In,
                    Some("cutlass-anim-in-typewriter".to_owned()),
                    Micros(500_000)
                )
                .is_err()
        );
        let none = desktop.animations().unwrap();
        assert_eq!(
            (none.enter, none.leave, none.length),
            (None, None, Micros(1_000_000))
        );
        desktop
            .set_animation(
                cutlass_model::AnimType::In,
                Some("cutlass-anim-in-slide-up".to_owned()),
                Micros(700_000),
            )
            .unwrap();
        desktop
            .set_animation(
                cutlass_model::AnimType::Out,
                Some("cutlass-anim-out-zoom-out".to_owned()),
                Micros(700_000),
            )
            .unwrap();
        let both = desktop.animations().unwrap();
        let enter = both.enter.unwrap();
        let leave = both.leave.unwrap();
        assert_eq!(enter.resource_id, "cutlass-anim-in-slide-up");
        assert_eq!(enter.duration, Micros(700_000));
        // The two share the clip, so the out side gets what is left.
        assert_eq!(leave.duration, Micros(300_000));
        // A loop plays the whole clip beside them, a round each duration.
        desktop
            .set_animation(
                cutlass_model::AnimType::Loop,
                Some("cutlass-anim-loop-swing".to_owned()),
                Micros(400_000),
            )
            .unwrap();
        let looped = desktop.animations().unwrap().looped.unwrap();
        assert_eq!(looped.duration, Micros(400_000));
        desktop.undo().unwrap();
        assert_eq!(desktop.animations().unwrap().looped, None);

        desktop.undo().unwrap();
        assert_eq!(desktop.animations().unwrap().leave, None);
        desktop
            .set_animation(cutlass_model::AnimType::In, None, Micros(0))
            .unwrap();
        assert_eq!(desktop.animations().unwrap().enter, None);
    }

    #[test]
    fn keys_toggle_step_and_ease_and_a_keyed_clip_keys_its_pose() {
        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        let mut video = MaterialVideo::new("picture");
        video.path = "clip.mp4".to_owned();
        video.width = 640;
        video.height = 360;
        draft.materials.videos.push(video);
        let mut picture = Segment::new("picture", "picture", TimeRange::new(0, 2_000_000));
        picture.source_range = Some(TimeRange::new(0, 2_000_000));
        draft.tracks[0].segments.push(picture);
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.select(Some(SegmentId::new("picture")));

        // With no keys a scale is the stored one.
        desktop.set_scale(0.5).unwrap();
        assert!(desktop.keys().unwrap().times.is_empty());
        desktop.toggle_keyframe().unwrap();
        desktop.seek(Micros(2_000_000));
        desktop.set_scale(1.5).unwrap();
        // A pose edit on a keyed clip keys it at the playhead.
        let keys = desktop.keys().unwrap();
        assert_eq!(keys.times, vec![Micros(0), Micros(2_000_000)]);
        assert_eq!(keys.here, Some(Micros(2_000_000)));
        assert_eq!(desktop.inspection().unwrap().scale_x, 1.5);
        desktop.seek(Micros(1_000_000));
        let halfway = desktop.inspection().unwrap().scale_x;
        assert!(halfway > 0.5 && halfway < 1.5, "halfway {halfway}");

        desktop.step_keyframe(true);
        assert_eq!(desktop.playhead(), Micros(2_000_000));
        desktop.step_keyframe(true);
        assert_eq!(desktop.playhead(), Micros(2_000_000));
        desktop.step_keyframe(false);
        assert_eq!(desktop.playhead(), Micros(0));

        desktop.seek(Micros(1_000_000));
        let curve = cutlass_editor::key_graphs()[0].name.clone();
        desktop.ease_keyframes(Some(curve.clone())).unwrap();
        assert_eq!(desktop.keys().unwrap().graph, Some(curve));
        desktop.undo().unwrap();
        assert_eq!(desktop.keys().unwrap().graph, None);

        desktop.seek(Micros(2_000_000));
        desktop.toggle_keyframe().unwrap();
        assert_eq!(desktop.keys().unwrap().times, vec![Micros(0)]);
        desktop.seek(Micros(5_000_000));
        assert_eq!(
            desktop.toggle_keyframe().unwrap_err().to_string(),
            "move the playhead onto the clip to key it"
        );
    }

    #[test]
    fn a_held_drag_shows_each_value_and_undoes_in_one_step() {
        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        let mut video = MaterialVideo::new("picture");
        video.path = "clip.mp4".to_owned();
        draft.materials.videos.push(video);
        draft.tracks[0].segments.push(Segment::new(
            "picture",
            "picture",
            TimeRange::new(0, 1_000_000),
        ));
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.select(Some(SegmentId::new("picture")));
        desktop.set_scale(1.5).unwrap();

        desktop.hold_edits(true);
        for opacity in [0.9, 0.6, 0.3] {
            desktop.set_opacity(opacity).unwrap();
            assert_eq!(desktop.inspection().unwrap().opacity, opacity);
        }
        // Another kind of edit in the same hold is its own step.
        desktop.set_scale(2.0).unwrap();
        desktop.hold_edits(false);
        // A new hold starts a new step, though it is the same kind.
        desktop.hold_edits(true);
        desktop.set_scale(2.5).unwrap();
        desktop.hold_edits(false);

        desktop.undo().unwrap();
        assert_eq!(desktop.inspection().unwrap().scale_x, 2.0);
        desktop.undo().unwrap();
        assert_eq!(desktop.inspection().unwrap().scale_x, 1.5);
        assert_eq!(desktop.inspection().unwrap().opacity, 0.3);
        desktop.undo().unwrap();
        assert_eq!(desktop.inspection().unwrap().opacity, 1.0);
        assert_eq!(desktop.inspection().unwrap().scale_x, 1.5);
    }

    #[test]
    fn edits_made_without_a_hold_each_undo_on_their_own() {
        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        let mut video = MaterialVideo::new("picture");
        video.path = "clip.mp4".to_owned();
        draft.materials.videos.push(video);
        draft.tracks[0].segments.push(Segment::new(
            "picture",
            "picture",
            TimeRange::new(0, 1_000_000),
        ));
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.select(Some(SegmentId::new("picture")));
        desktop.set_opacity(0.5).unwrap();
        desktop.set_opacity(0.25).unwrap();
        desktop.undo().unwrap();
        assert_eq!(desktop.inspection().unwrap().opacity, 0.5);
    }

    #[test]
    fn a_blend_mode_is_set_shown_saved_and_undone() {
        use cutlass_effects::BlendMode;
        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        let mut video = MaterialVideo::new("picture");
        video.path = "clip.mp4".to_owned();
        draft.materials.videos.push(video);
        draft.tracks[0].segments.push(Segment::new(
            "picture",
            "picture",
            TimeRange::new(0, 1_000_000),
        ));
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.select(Some(SegmentId::new("picture")));
        assert_eq!(desktop.inspection().unwrap().blend, BlendMode::SourceOver);
        desktop.set_blend(BlendMode::Screen).unwrap();
        assert_eq!(desktop.inspection().unwrap().blend, BlendMode::Screen);
        desktop.set_blend(BlendMode::Multiply).unwrap();
        assert_eq!(desktop.inspection().unwrap().blend, BlendMode::Multiply);
        desktop.save().unwrap();
        let saved = Folder::read(&temp.path).unwrap();
        let segment = &saved.draft.tracks[0].segments[0];
        assert_eq!(
            cutlass_effects::segment_blend(&saved.draft, segment),
            Some(BlendMode::Multiply)
        );
        desktop.set_blend(BlendMode::SourceOver).unwrap();
        assert_eq!(desktop.inspection().unwrap().blend, BlendMode::SourceOver);
        desktop.undo().unwrap();
        assert_eq!(desktop.inspection().unwrap().blend, BlendMode::Multiply);
        desktop.undo().unwrap();
        assert_eq!(desktop.inspection().unwrap().blend, BlendMode::Screen);
    }

    #[test]
    fn adjust_sliders_are_set_and_reset_in_one_undo_step() {
        use cutlass_effects::AdjustKey;
        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        let mut video = MaterialVideo::new("picture");
        video.path = "clip.mp4".to_owned();
        draft.materials.videos.push(video);
        draft.tracks[0].segments.push(Segment::new(
            "picture",
            "picture",
            TimeRange::new(0, 1_000_000),
        ));
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.select(Some(SegmentId::new("picture")));
        assert!(desktop.inspection().unwrap().adjust.is_identity());
        desktop.reset_adjust().unwrap();
        assert!(!desktop.can_undo());
        desktop.set_adjust(AdjustKey::Temperature, 0.4).unwrap();
        desktop.set_adjust(AdjustKey::Contrast, -0.2).unwrap();
        let adjust = desktop.inspection().unwrap().adjust;
        assert!((adjust.get(AdjustKey::Temperature) - 0.4).abs() < 1e-6);
        assert!((adjust.get(AdjustKey::Contrast) + 0.2).abs() < 1e-6);
        assert!(desktop.set_adjust(AdjustKey::Grain, -0.5).is_err());
        desktop.reset_adjust().unwrap();
        assert!(desktop.inspection().unwrap().adjust.is_identity());
        desktop.undo().unwrap();
        assert!(
            (desktop
                .inspection()
                .unwrap()
                .adjust
                .get(AdjustKey::Contrast)
                + 0.2)
                .abs()
                < 1e-6
        );
        assert!(
            (desktop
                .inspection()
                .unwrap()
                .adjust
                .get(AdjustKey::Temperature)
                - 0.4)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn auto_adjust_and_colour_match_set_the_sliders_in_one_undo_step() {
        use cutlass_effects::AdjustKey;
        let temp = Temp::new();
        let png = temp.path.join("dark.png");
        image::RgbaImage::from_fn(8, 4, |x, _| {
            let lift = u8::try_from(x * 4).unwrap();
            image::Rgba([0x20 + lift, 0x30 + lift, 0x60 + lift, 0xFF])
        })
        .save(&png)
        .unwrap();
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 8;
        draft.canvas_config.height = 4;
        draft.canvas_config.background = Some("#ffffff".to_owned());
        let mut photo = MaterialVideo::new("photo");
        photo.kind = "photo".to_owned();
        photo.path = png.display().to_string();
        photo.width = 8;
        photo.height = 4;
        draft.materials.videos.push(photo);
        let mut segment = Segment::new("color", "photo", TimeRange::new(0, 1_000_000));
        segment.source_range = Some(TimeRange::new(0, 1_000_000));
        draft.tracks[0].segments.push(segment);
        let warm_png = temp.path.join("warm.png");
        image::RgbaImage::from_fn(8, 4, |x, _| {
            let lift = u8::try_from(x * 8).unwrap();
            image::Rgba([0xB0 + lift, 0x80 + lift, 0x40 + lift, 0xFF])
        })
        .save(&warm_png)
        .unwrap();
        let mut warm = MaterialVideo::new("warm");
        warm.kind = "photo".to_owned();
        warm.path = warm_png.display().to_string();
        warm.width = 8;
        warm.height = 4;
        draft.materials.videos.push(warm);
        let mut segment = Segment::new("warm", "warm", TimeRange::new(1_000_000, 1_000_000));
        segment.source_range = Some(TimeRange::new(0, 1_000_000));
        draft.tracks[0].segments.push(segment);
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.select(Some(SegmentId::new("color")));
        // The clip's own colour work is not what auto adjust looks at.
        desktop.set_adjust(AdjustKey::Exposure, -0.8).unwrap();
        desktop.auto_adjust().unwrap();
        let adjust = desktop.inspection().unwrap().adjust;
        assert!(adjust.get(AdjustKey::Exposure) > 0.3, "{adjust:?}");
        assert!(adjust.get(AdjustKey::Temperature) > 0.2, "{adjust:?}");
        // Again, it lands on the same values and adds no step.
        desktop.auto_adjust().unwrap();
        assert_eq!(desktop.inspection().unwrap().adjust, adjust);
        desktop.undo().unwrap();
        let adjust = desktop.inspection().unwrap().adjust;
        assert!((adjust.get(AdjustKey::Exposure) + 0.8).abs() < 1e-6);
        assert_eq!(adjust.get(AdjustKey::Temperature), 0.0);
        // Colour match warms the blue picture toward the warm one.
        let choices = desktop.match_choices();
        assert_eq!(choices.len(), 1, "{choices:?}");
        assert_eq!(choices[0].0, SegmentId::new("warm"));
        desktop.match_colour(&SegmentId::new("warm")).unwrap();
        let matched = desktop.inspection().unwrap().adjust;
        assert!(matched.get(AdjustKey::Temperature) > 0.5, "{matched:?}");
        assert!(matched.get(AdjustKey::Exposure) > 0.5, "{matched:?}");
        assert!(desktop.match_colour(&SegmentId::new("color")).is_err());
        desktop.undo().unwrap();
        assert_eq!(
            desktop
                .inspection()
                .unwrap()
                .adjust
                .get(AdjustKey::Temperature),
            0.0
        );
        desktop.select(None);
        assert!(desktop.auto_adjust().is_err());
        assert!(desktop.match_choices().is_empty());
    }

    #[test]
    fn an_adjustment_layer_is_added_at_the_playhead_and_adjusted() {
        use cutlass_effects::AdjustKey;
        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        let mut video = MaterialVideo::new("picture");
        video.path = "clip.mp4".to_owned();
        draft.materials.videos.push(video);
        draft.tracks[0].segments.push(Segment::new(
            "picture",
            "picture",
            TimeRange::new(0, 5_000_000),
        ));
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.seek(Micros(1_000_000));
        desktop.add_adjustment().unwrap();
        let layer = desktop.selected().cloned().unwrap();
        let draft = desktop.draft().unwrap();
        let (segment, kind) = super::placed(draft, &layer).unwrap();
        assert_eq!(*kind, TrackType::Adjust);
        assert_eq!(segment.target.start, Micros(1_000_000));
        assert_eq!(desktop.panel(), Some(cutlass_editor::Panel::Adjust));
        desktop.set_adjust(AdjustKey::Exposure, 0.3).unwrap();
        assert!(
            (desktop
                .inspection()
                .unwrap()
                .adjust
                .get(AdjustKey::Exposure)
                - 0.3)
                .abs()
                < 1e-6
        );
        // A second layer at the same time takes a row of its own.
        desktop.add_adjustment().unwrap();
        let rows = desktop
            .draft()
            .unwrap()
            .tracks
            .iter()
            .filter(|track| track.kind == TrackType::Adjust)
            .count();
        assert_eq!(rows, 2);
        desktop.undo().unwrap();
        desktop.undo().unwrap();
        desktop.undo().unwrap();
        assert!(
            !desktop
                .draft()
                .unwrap()
                .tracks
                .iter()
                .any(|track| track.kind == TrackType::Adjust)
        );
    }

    #[test]
    fn a_filter_is_added_from_its_card_with_strength_and_temperature() {
        use cutlass_effects::AdjustKey;
        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        let mut video = MaterialVideo::new("picture");
        video.path = "clip.mp4".to_owned();
        draft.materials.videos.push(video);
        draft.tracks[0].segments.push(Segment::new(
            "picture",
            "picture",
            TimeRange::new(0, 5_000_000),
        ));
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        assert!(desktop.add_filter("filter:look/none").is_err());
        let (card, _) = cutlass_editor::filter_cards()[5].clone();
        desktop.add_filter(&card).unwrap();
        let clip = desktop.selected().cloned().unwrap();
        let draft = desktop.draft().unwrap();
        let (segment, kind) = super::placed(draft, &clip).unwrap();
        assert_eq!(*kind, TrackType::Filter);
        assert!(cutlass_effects::segment_look(draft, segment).is_some());
        assert_eq!(desktop.panel(), Some(cutlass_editor::Panel::Filter));
        assert_eq!(desktop.effect_name().as_deref(), Some("Noir"));
        desktop.set_effect_strength(0.4).unwrap();
        assert!((desktop.effect_strength().unwrap() - 0.4).abs() < 1e-9);
        desktop.set_adjust(AdjustKey::Temperature, 0.2).unwrap();
        assert!(
            (desktop
                .inspection()
                .unwrap()
                .adjust
                .get(AdjustKey::Temperature)
                - 0.2)
                .abs()
                < 1e-6
        );
        desktop.undo().unwrap();
        desktop.undo().unwrap();
        desktop.undo().unwrap();
        assert!(
            desktop
                .draft()
                .unwrap()
                .tracks
                .iter()
                .all(|track| track.kind != TrackType::Filter)
        );
    }

    #[test]
    fn hsl_and_wheels_are_set_on_a_picture() {
        use cutlass_effects::{Wheel, WheelSetting};
        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        let mut video = MaterialVideo::new("picture");
        video.path = "clip.mp4".to_owned();
        draft.materials.videos.push(video);
        draft.tracks[0].segments.push(Segment::new(
            "picture",
            "picture",
            TimeRange::new(0, 1_000_000),
        ));
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.select(Some(SegmentId::new("picture")));
        desktop.set_hsl(4, [0.2, -0.5, 0.0]).unwrap();
        desktop
            .set_wheel(
                Wheel::Gain,
                WheelSetting {
                    x: 3.0,
                    y: 4.0,
                    luma: 0.1,
                },
            )
            .unwrap();
        let grade = desktop.inspection().unwrap().grade;
        assert_eq!(grade.hsl[4], [0.2, -0.5, 0.0]);
        let gain = grade.wheel(Wheel::Gain);
        assert!((gain.x - 0.6).abs() < 1e-6 && (gain.y - 0.8).abs() < 1e-6);
        assert!(desktop.set_hsl(9, [0.1, 0.0, 0.0]).is_err());
        desktop.undo().unwrap();
        desktop.undo().unwrap();
        assert!(desktop.inspection().unwrap().grade.is_identity());
    }

    #[test]
    fn a_lut_file_is_loaded_weakened_and_taken_off() {
        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        let mut video = MaterialVideo::new("picture");
        video.path = "clip.mp4".to_owned();
        draft.materials.videos.push(video);
        draft.tracks[0].segments.push(Segment::new(
            "picture",
            "picture",
            TimeRange::new(0, 1_000_000),
        ));
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.select(Some(SegmentId::new("picture")));
        let bad = temp.path.join("bad.cube");
        std::fs::write(&bad, "LUT_3D_SIZE 3\n0 0 0\n").unwrap();
        let refused = desktop.load_lut(&bad).unwrap_err().to_string();
        assert!(refused.contains("not 27"), "{refused}");
        assert!(desktop.set_lut_strength(0.5).is_err());
        let good = temp.path.join("Warm film.cube");
        std::fs::write(
            &good,
            "LUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n",
        )
        .unwrap();
        desktop.load_lut(&good).unwrap();
        assert_eq!(
            desktop.inspection().unwrap().lut,
            Some(("Warm film".to_owned(), 1.0))
        );
        desktop.set_lut_strength(0.5).unwrap();
        assert_eq!(
            desktop.inspection().unwrap().lut,
            Some(("Warm film".to_owned(), 0.5))
        );
        desktop.remove_lut().unwrap();
        assert_eq!(desktop.inspection().unwrap().lut, None);
        desktop.undo().unwrap();
        assert_eq!(
            desktop
                .inspection()
                .unwrap()
                .lut
                .map(|(_, strength)| strength),
            Some(0.5)
        );
    }

    #[test]
    fn a_picture_changes_opacity_and_a_sound_does_not() {
        let mut bare = shell();
        assert!(!bare.place_enabled());
        assert_eq!(
            bare.set_opacity(0.5).unwrap_err().to_string(),
            "no project is open"
        );

        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        let mut video = MaterialVideo::new("picture");
        video.path = "clip.mp4".to_owned();
        draft.materials.videos.push(video);
        let mut picture = Segment::new("picture", "picture", TimeRange::new(0, 1_000_000));
        picture.clip.rotation = 90.0;
        picture.clip.flip.horizontal = true;
        draft.tracks[0].segments.push(picture);
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("tone"),
            path: "tone".to_owned(),
            ..MaterialAudio::default()
        });
        let mut sound = Track::main("tone");
        sound.kind = TrackType::Audio;
        sound
            .segments
            .push(Segment::new("tone", "tone", TimeRange::new(0, 500_000)));
        draft.tracks.push(sound);
        draft.materials.texts.push(MaterialText {
            id: MaterialId::new("caption"),
            content: "hi".to_owned(),
            ..MaterialText::default()
        });
        let mut words = Track::main("words");
        words.kind = TrackType::Text;
        words.segments.push(Segment::new(
            "caption",
            "caption",
            TimeRange::new(0, 500_000),
        ));
        draft.tracks.push(words);
        Folder::new(draft).save(&temp.path).unwrap();

        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.select(None);
        assert_eq!(
            desktop.set_opacity(0.5).unwrap_err().to_string(),
            "nothing is selected"
        );
        desktop.select(Some(SegmentId::new("tone")));
        assert!(!desktop.place_enabled());
        assert_eq!(
            desktop.set_opacity(0.5).unwrap_err().to_string(),
            "an opacity belongs on a picture"
        );

        desktop.select(Some(SegmentId::new("picture")));
        assert!(desktop.place_enabled());
        assert_eq!(desktop.inspection().unwrap().opacity, 1.0);
        desktop.set_opacity(0.5).unwrap();
        let inspection = desktop.inspection().unwrap();
        assert_eq!(inspection.opacity, 0.5);
        assert_eq!(inspection.rotation, 90.0);
        assert_eq!(inspection.scale_x, 1.0);
        assert!(
            desktop.draft().unwrap().tracks[0].segments[0]
                .clip
                .flip
                .horizontal
        );
        desktop.undo().unwrap();
        assert_eq!(desktop.inspection().unwrap().opacity, 1.0);
        desktop.set_opacity(0.0).unwrap();
        assert_eq!(desktop.inspection().unwrap().opacity, 0.0);
        assert_eq!(
            desktop.set_opacity(1.1).unwrap_err().to_string(),
            "the opacity must be from zero to one"
        );
        assert_eq!(desktop.inspection().unwrap().opacity, 0.0);

        let main = desktop.draft().unwrap().tracks[0].id.clone();
        desktop
            .toggle_attribute(main, TrackAttribute::LOCK)
            .unwrap();
        assert_eq!(
            desktop.set_opacity(1.0).unwrap_err().to_string(),
            "the track is locked"
        );
        assert_eq!(desktop.inspection().unwrap().opacity, 0.0);

        desktop.select(Some(SegmentId::new("caption")));
        assert!(desktop.place_enabled());
        desktop.set_opacity(0.25).unwrap();
        assert_eq!(desktop.inspection().unwrap().opacity, 0.25);
    }

    const SAMPLE_TTF: &[u8] = include_bytes!("../../../crates/font/fixtures/sample.ttf");
    const GRAY_OTF: &[u8] = include_bytes!("../../../crates/font/fixtures/gray-strike.otf");

    #[test]
    fn a_face_changes_the_drawn_letters() {
        let mut bare = shell();
        assert_eq!(
            bare.cycle_text_font().unwrap_err().to_string(),
            "no project is open"
        );

        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 320;
        draft.canvas_config.height = 180;
        draft.materials.videos.push(MaterialVideo::new("picture"));
        draft.tracks[0].segments.push(Segment::new(
            "picture",
            "picture",
            TimeRange::new(0, 1_000_000),
        ));
        draft.materials.texts.push(MaterialText {
            id: MaterialId::new("caption"),
            content: "A".to_owned(),
            ..MaterialText::default()
        });
        let mut words = Track::main("words");
        words.kind = TrackType::Text;
        words.segments.push(Segment::new(
            "caption",
            "caption",
            TimeRange::new(0, 1_000_000),
        ));
        draft.tracks.push(words);
        Folder::new(draft).save(&temp.path).unwrap();

        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.use_faces(vec![
            ("sample.ttf", SAMPLE_TTF.to_vec()),
            ("gray.otf", GRAY_OTF.to_vec()),
        ]);
        desktop.select(Some(SegmentId::new("picture")));
        assert_eq!(
            desktop.cycle_text_font().unwrap_err().to_string(),
            "a text style belongs on text"
        );

        desktop.select(Some(SegmentId::new("caption")));
        desktop.set_text_size(48.0).unwrap();
        assert_eq!(desktop.inspection().unwrap().font, "sample.ttf");
        let sized = desktop.preview_bytes().unwrap();
        desktop.cycle_text_font().unwrap();
        assert_eq!(desktop.inspection().unwrap().font, "gray.otf");
        assert_eq!(desktop.inspection().unwrap().text_size, 48.0);
        assert_eq!(desktop.draft().unwrap().materials.texts[0].content, "A");
        let other = desktop.preview_bytes().unwrap();
        assert_ne!(digest(&sized.rgba), digest(&other.rgba));
        desktop.set_text_size(48.0).unwrap();
        assert_eq!(desktop.inspection().unwrap().font, "gray.otf");
        desktop.undo().unwrap();
        desktop.undo().unwrap();
        assert_eq!(desktop.inspection().unwrap().font, "sample.ttf");
        assert_eq!(
            digest(&desktop.preview_bytes().unwrap().rgba),
            digest(&sized.rgba)
        );

        desktop.use_face("sample.ttf", SAMPLE_TTF.to_vec());
        let revision = desktop.revision();
        desktop.cycle_text_font().unwrap();
        assert_eq!(desktop.revision(), revision);

        let lane = desktop.draft().unwrap().tracks[1].id.clone();
        desktop
            .toggle_attribute(lane, TrackAttribute::LOCK)
            .unwrap();
        assert_eq!(
            desktop.cycle_text_font().unwrap_err().to_string(),
            "the track is locked"
        );
    }

    #[test]
    fn a_text_clip_changes_size_color_and_alignment() {
        let mut bare = shell();
        assert_eq!(
            bare.set_text_size(48.0).unwrap_err().to_string(),
            "no project is open"
        );

        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        let mut video = MaterialVideo::new("picture");
        video.path = "clip.mp4".to_owned();
        draft.materials.videos.push(video);
        draft.tracks[0].segments.push(Segment::new(
            "picture",
            "picture",
            TimeRange::new(0, 1_000_000),
        ));
        draft.materials.texts.push(MaterialText {
            id: MaterialId::new("caption"),
            content: "A".to_owned(),
            words: cutlass_model::WordColumns {
                text: vec!["A".to_owned()],
                start_time: vec![0],
                end_time: vec![100],
                ..cutlass_model::WordColumns::default()
            },
            ..MaterialText::default()
        });
        let mut words = Track::main("words");
        words.kind = TrackType::Text;
        words.segments.push(Segment::new(
            "caption",
            "caption",
            TimeRange::new(0, 1_000_000),
        ));
        draft.tracks.push(words);
        Folder::new(draft).save(&temp.path).unwrap();

        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.use_face("sample.ttf", SAMPLE_TTF.to_vec());
        desktop.select(None);
        assert_eq!(
            desktop.set_text_color("#FFFFFF").unwrap_err().to_string(),
            "nothing is selected"
        );
        desktop.select(Some(SegmentId::new("picture")));
        assert_eq!(
            desktop.set_text_size(48.0).unwrap_err().to_string(),
            "a text style belongs on text"
        );

        desktop.select(Some(SegmentId::new("caption")));
        let before = desktop.preview_bytes().unwrap();
        desktop.set_text_size(64.0).unwrap();
        let inspection = desktop.inspection().unwrap();
        assert_eq!(inspection.text_size, 64.0);
        assert_eq!(inspection.text.as_deref(), Some("A"));
        assert_eq!(inspection.words.text, ["A"]);
        assert_eq!(
            desktop.draft().unwrap().materials.texts[0].font_path,
            "sample.ttf"
        );
        let sized = desktop.preview_bytes().unwrap();
        assert_ne!(digest(&before.rgba), digest(&sized.rgba));

        desktop.set_text_color("#00FF00").unwrap();
        assert_eq!(desktop.inspection().unwrap().text_color, "#00FF00");
        let colored = desktop.preview_bytes().unwrap();
        assert_ne!(digest(&sized.rgba), digest(&colored.rgba));

        desktop.cycle_text_align().unwrap();
        let inspection = desktop.inspection().unwrap();
        assert_eq!(inspection.text_align, 1);
        assert_eq!(
            desktop.draft().unwrap().materials.texts[0].line_max_width,
            1920.0
        );
        desktop.undo().unwrap();
        assert_eq!(desktop.inspection().unwrap().text_align, 0);
        assert_eq!(
            desktop.set_text_size(0.0).unwrap_err().to_string(),
            "the text size must be greater than zero and at most 512"
        );
        assert_eq!(
            desktop.set_text_color("#gg0000").unwrap_err().to_string(),
            "the text color is not a hex color"
        );
        assert_eq!(desktop.inspection().unwrap().text_size, 64.0);
        assert_eq!(desktop.inspection().unwrap().words.text, ["A"]);

        let lane = desktop.draft().unwrap().tracks[1].id.clone();
        desktop
            .toggle_attribute(lane, TrackAttribute::LOCK)
            .unwrap();
        assert_eq!(
            desktop.cycle_text_align().unwrap_err().to_string(),
            "the track is locked"
        );
        assert_eq!(desktop.inspection().unwrap().text_align, 0);
    }

    #[test]
    fn text_spacing_opens_the_gaps_between_letters_and_lines() {
        let mut bare = shell();
        assert_eq!(
            bare.set_letter_spacing(1.0).unwrap_err().to_string(),
            "no project is open"
        );

        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        let mut video = MaterialVideo::new("picture");
        video.path = "clip.mp4".to_owned();
        draft.materials.videos.push(video);
        draft.tracks[0].segments.push(Segment::new(
            "picture",
            "picture",
            TimeRange::new(0, 1_000_000),
        ));
        draft.materials.texts.push(MaterialText {
            id: MaterialId::new("caption"),
            content: "AA".to_owned(),
            ..MaterialText::default()
        });
        let mut words = Track::main("words");
        words.kind = TrackType::Text;
        words.segments.push(Segment::new(
            "caption",
            "caption",
            TimeRange::new(0, 1_000_000),
        ));
        draft.tracks.push(words);
        Folder::new(draft).save(&temp.path).unwrap();

        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.use_face("sample.ttf", SAMPLE_TTF.to_vec());
        desktop.select(Some(SegmentId::new("picture")));
        assert_eq!(
            desktop.set_line_spacing(0.5).unwrap_err().to_string(),
            "a text style belongs on text"
        );

        desktop.select(Some(SegmentId::new("caption")));
        desktop.set_text_size(64.0).unwrap();
        let fitted = desktop.preview_bytes().unwrap();
        desktop.set_letter_spacing(1.0).unwrap();
        let inspection = desktop.inspection().unwrap();
        assert_eq!(inspection.letter_spacing, 1.0);
        assert_eq!(inspection.line_spacing, 0.0);
        assert_eq!(inspection.text_size, 64.0);
        let opened = desktop.preview_bytes().unwrap();
        assert_ne!(digest(&fitted.rgba), digest(&opened.rgba));
        desktop.undo().unwrap();
        assert_eq!(desktop.inspection().unwrap().letter_spacing, 0.0);
        assert_eq!(
            desktop.set_letter_spacing(4.1).unwrap_err().to_string(),
            "letter spacing must be from -1 to 4"
        );
        assert_eq!(
            desktop.set_line_spacing(-1.0).unwrap_err().to_string(),
            "line spacing must be greater than -1 and at most 4"
        );
        assert_eq!(desktop.inspection().unwrap().letter_spacing, 0.0);

        desktop.set_text("A\nA".to_owned()).unwrap();
        let stacked = desktop.preview_bytes().unwrap();
        desktop.set_line_spacing(1.0).unwrap();
        assert_eq!(desktop.inspection().unwrap().line_spacing, 1.0);
        assert_eq!(desktop.inspection().unwrap().text.as_deref(), Some("A\nA"));
        let spaced = desktop.preview_bytes().unwrap();
        assert_ne!(digest(&stacked.rgba), digest(&spaced.rgba));

        let lane = desktop.draft().unwrap().tracks[1].id.clone();
        desktop
            .toggle_attribute(lane, TrackAttribute::LOCK)
            .unwrap();
        assert_eq!(
            desktop.set_letter_spacing(0.5).unwrap_err().to_string(),
            "the track is locked"
        );
        assert_eq!(desktop.inspection().unwrap().line_spacing, 1.0);
    }

    #[test]
    fn the_frame_rate_steps_the_playhead_and_drop_frame_labels_the_clock() {
        let mut bare = shell();
        assert_eq!(
            bare.cycle_frame_rate().unwrap_err().to_string(),
            "no project is open"
        );

        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.tracks[0].segments.push(Segment::new(
            "picture",
            "picture",
            TimeRange::new(0, 1_000_000),
        ));
        Folder::new(draft).save(&temp.path).unwrap();

        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        assert_eq!(desktop.frame_rate_label(), "30 fps");
        desktop.seek(Micros(0));
        desktop.step_frame(1).unwrap();
        assert_eq!(desktop.playhead(), Micros(33_334));
        desktop.seek(Micros(0));
        desktop.cycle_frame_rate().unwrap();
        assert_eq!(desktop.frame_rate_label(), "50 fps");
        assert!(!desktop.drop_frame_available());
        desktop.step_frame(1).unwrap();
        assert_eq!(desktop.playhead(), Micros(20_000));
        assert_eq!(desktop.draft().unwrap().canvas_config.width, 2);

        desktop.cycle_frame_rate().unwrap();
        assert_eq!(desktop.frame_rate_label(), "59.94 fps");
        assert!(desktop.drop_frame_available());
        assert!(desktop.drop_frame_enabled());
        assert_eq!(desktop.timecode_at(Micros(1_000_000)), "00:00:00;59");
        desktop.toggle_drop_frame().unwrap();
        assert_eq!(desktop.frame_rate_label(), "59.94 fps");
        assert_eq!(desktop.timecode_at(Micros(1_000_000)), "00:00:00:59");
        desktop.undo().unwrap();
        assert!(desktop.drop_frame_enabled());
        desktop.undo().unwrap();
        assert_eq!(desktop.frame_rate_label(), "50 fps");
        assert!(!desktop.drop_frame_enabled());

        let lane = desktop.draft().unwrap().tracks[0].id.clone();
        desktop
            .toggle_attribute(lane, TrackAttribute::LOCK)
            .unwrap();
        desktop.cycle_frame_rate().unwrap();
        assert_eq!(desktop.frame_rate_label(), "59.94 fps");
    }

    #[test]
    fn a_letterbox_color_fills_the_bars_around_the_picture() {
        let mut bare = shell();
        assert_eq!(
            bare.set_letterbox("#FF0000").unwrap_err().to_string(),
            "no project is open"
        );

        let temp = Temp::new();
        let picture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/engine/fixtures/pattern.png");
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 16;
        draft.canvas_config.height = 18;
        draft.canvas_config.background = Some("#0000FF".to_owned());
        let mut video = MaterialVideo::new("still");
        video.path = picture.display().to_string();
        video.duration = Micros(1_000_000);
        draft.materials.videos.push(video);
        let mut segment = Segment::new("still", "still", TimeRange::new(0, 1_000_000));
        segment.source_range = Some(TimeRange::new(0, 1_000_000));
        draft.tracks[0].segments.push(segment);
        draft.materials.texts.push(MaterialText::new("caption"));
        let mut words = Track::main("words");
        words.kind = TrackType::Text;
        words.segments.push(Segment::new(
            "caption",
            "caption",
            TimeRange::new(0, 1_000_000),
        ));
        draft.tracks.push(words);
        Folder::new(draft).save(&temp.path).unwrap();

        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.select(Some(SegmentId::new("caption")));
        assert_eq!(
            desktop.set_letterbox("#FF0000").unwrap_err().to_string(),
            "a letterbox belongs on a picture"
        );

        desktop.select(Some(SegmentId::new("still")));
        let before = desktop.preview_bytes().unwrap();
        assert_eq!(&before.rgba[..4], &[0x00, 0x00, 0xFF, 0xFF]);
        desktop.set_letterbox("#FF0000").unwrap();
        assert_eq!(desktop.inspection().unwrap().letterbox, "#FF0000");
        assert_eq!(desktop.inspection().unwrap().scale_x, 1.0);
        let colored = desktop.preview_bytes().unwrap();
        assert_eq!((colored.width, colored.height), (16, 18));
        assert_eq!(&colored.rgba[..4], &[0xFF, 0x00, 0x00, 0xFF]);
        assert_ne!(&colored.rgba[64..68], &[0xFF, 0x00, 0x00, 0xFF]);
        desktop.undo().unwrap();
        assert_eq!(desktop.inspection().unwrap().letterbox, "");
        let restored = desktop.preview_bytes().unwrap();
        assert_eq!(&restored.rgba[..4], &[0x00, 0x00, 0xFF, 0xFF]);
        assert_eq!(
            desktop.set_letterbox("white").unwrap_err().to_string(),
            "the canvas color is not a hex color"
        );

        let lane = desktop.draft().unwrap().tracks[0].id.clone();
        desktop
            .toggle_attribute(lane, TrackAttribute::LOCK)
            .unwrap();
        assert_eq!(
            desktop.set_letterbox("#00FF00").unwrap_err().to_string(),
            "the track is locked"
        );
        assert_eq!(desktop.inspection().unwrap().letterbox, "");
    }

    #[test]
    fn a_project_name_is_the_window_title_and_the_saved_draft() {
        let mut bare = shell();
        assert_eq!(
            bare.set_project_name("Cut").unwrap_err().to_string(),
            "no project is open"
        );

        let root = Temp::new();
        let path = Shell::create_project(&root.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&path).unwrap();
        assert_eq!(desktop.project_name(), "Project");
        let width = desktop.draft().unwrap().canvas_config.width;
        let space = desktop.draft().unwrap().color_space;
        let revision = desktop.revision();
        desktop.set_project_name("Project").unwrap();
        assert_eq!(desktop.revision(), revision);

        desktop.set_project_name("  Cut  ").unwrap();
        assert_eq!(desktop.project_name(), "Cut");
        assert_eq!(desktop.draft().unwrap().canvas_config.width, width);
        assert_eq!(desktop.draft().unwrap().color_space, space);
        desktop.save().unwrap();
        let saved = Folder::read(&path).unwrap();
        assert_eq!(saved.draft.name, "Cut");
        assert_eq!(saved.project.timelines[0].name, "Cut");
        assert_eq!(saved.layout.dock_items[0].timeline_names, ["Cut"]);
        assert_eq!(
            Shell::list_projects(&root.path)
                .into_iter()
                .map(|project| project.name)
                .collect::<Vec<_>>(),
            ["Cut"]
        );

        desktop.undo().unwrap();
        assert_eq!(desktop.project_name(), "Project");
        desktop.set_project_name("   ").unwrap();
        assert_eq!(desktop.draft().unwrap().name, "");
        assert_eq!(desktop.project_name(), "Untitled");

        let lane = desktop.draft().unwrap().tracks[0].id.clone();
        desktop
            .toggle_attribute(lane, TrackAttribute::LOCK)
            .unwrap();
        desktop.set_project_name("Locked").unwrap();
        assert_eq!(desktop.project_name(), "Locked");
    }

    #[test]
    fn the_color_space_steps_and_the_picture_stays() {
        let mut bare = shell();
        assert_eq!(
            bare.cycle_color_space().unwrap_err().to_string(),
            "no project is open"
        );

        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.canvas_config.background = Some("#336699".to_owned());
        Folder::new(draft).save(&temp.path).unwrap();

        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        assert_eq!(desktop.color_space_label(), "None");
        let picture = desktop.preview_bytes().unwrap();
        assert_eq!(picture.rgba, [0x33, 0x66, 0x99, 0xFF].repeat(2));

        let steps = ["SDR 2020", "HLG", "PQ", "Unknown", "None"];
        let spaces = [2, 67, 2084, -1, 0];
        for (label, space) in steps.into_iter().zip(spaces) {
            desktop.cycle_color_space().unwrap();
            assert_eq!(desktop.color_space_label(), label);
            assert_eq!(desktop.draft().unwrap().color_space, space);
            assert_eq!(desktop.draft().unwrap().fps, 30.0);
            assert_eq!(desktop.draft().unwrap().canvas_config.width, 2);
            let next = desktop.preview_bytes().unwrap();
            assert_eq!(next.rgba, picture.rgba);
            let compared = cutlass_export::same_software_pixels(
                desktop.draft().unwrap(),
                next.width,
                next.height,
                &picture.rgba,
                &next.rgba,
            );
            if space == 67 || space == 2084 {
                assert_eq!(compared, Err(cutlass_export::ExportError::HdrNotThisCase));
            } else {
                assert_eq!(compared, Ok(()));
            }
        }
        desktop.undo().unwrap();
        assert_eq!(desktop.draft().unwrap().color_space, -1);

        let lane = desktop.draft().unwrap().tracks[0].id.clone();
        desktop
            .toggle_attribute(lane, TrackAttribute::LOCK)
            .unwrap();
        desktop.cycle_color_space().unwrap();
        assert_eq!(desktop.draft().unwrap().color_space, 0);
    }

    #[test]
    fn the_canvas_shape_changes_the_frame() {
        let mut bare = shell();
        assert_eq!(
            bare.cycle_canvas().unwrap_err().to_string(),
            "no project is open"
        );

        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.canvas_config.background = Some("#336699".to_owned());
        Folder::new(draft).save(&temp.path).unwrap();

        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.cycle_canvas().unwrap();
        assert_eq!(desktop.draft().unwrap().canvas_config.width, 1_080);
        assert_eq!(desktop.draft().unwrap().canvas_config.height, 1_920);
        assert_eq!(desktop.draft().unwrap().canvas_config.ratio, "9:16");
        assert_eq!(
            desktop.draft().unwrap().canvas_config.background.as_deref(),
            Some("#336699")
        );
        assert_eq!(desktop.draft().unwrap().fps, 30.0);
        let picture = desktop.preview_bytes().unwrap();
        assert_eq!((picture.width, picture.height), (1_080, 1_920));
        assert_eq!(&picture.rgba[..4], &[0x33, 0x66, 0x99, 0xFF]);
        desktop.undo().unwrap();
        assert_eq!(desktop.draft().unwrap().canvas_config.width, 1_920);
        assert_eq!(desktop.draft().unwrap().canvas_config.height, 1_080);
        assert_eq!(desktop.draft().unwrap().canvas_config.ratio, "original");

        let lane = desktop.draft().unwrap().tracks[0].id.clone();
        desktop
            .toggle_attribute(lane, TrackAttribute::LOCK)
            .unwrap();
        desktop.cycle_canvas().unwrap();
        assert_eq!(desktop.draft().unwrap().canvas_config.ratio, "9:16");
    }

    #[test]
    fn the_canvas_color_fills_the_frame_and_an_empty_color_clears_it() {
        let mut bare = shell();
        assert_eq!(
            bare.set_canvas_background("#336699")
                .unwrap_err()
                .to_string(),
            "no project is open"
        );

        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        Folder::new(draft).save(&temp.path).unwrap();

        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let clear = desktop.preview_bytes().unwrap();
        assert_eq!(clear.rgba, vec![0, 0, 0, 0, 0, 0, 0, 0]);
        desktop.set_canvas_background("  #336699  ").unwrap();
        let colored = desktop.preview_bytes().unwrap();
        assert_eq!(colored.width, 2);
        assert_eq!(colored.height, 1);
        assert_eq!(colored.rgba, [0x33, 0x66, 0x99, 0xFF].repeat(2));
        assert_eq!(desktop.draft().unwrap().canvas_config.width, 2);
        assert_eq!(desktop.draft().unwrap().canvas_config.height, 1);
        desktop.undo().unwrap();
        assert_eq!(desktop.preview_bytes().unwrap().rgba, vec![0; 8]);

        let lane = desktop.draft().unwrap().tracks[0].id.clone();
        desktop
            .toggle_attribute(lane, TrackAttribute::LOCK)
            .unwrap();
        desktop.set_canvas_background("0xFF336699").unwrap();
        assert_eq!(
            desktop.preview_bytes().unwrap().rgba,
            [0x33, 0x66, 0x99, 0xFF].repeat(2)
        );
        desktop.set_canvas_background("").unwrap();
        assert_eq!(desktop.preview_bytes().unwrap().rgba, vec![0; 8]);
        assert_eq!(
            desktop
                .set_canvas_background("white")
                .unwrap_err()
                .to_string(),
            "the canvas color is not a hex color"
        );
        assert_eq!(desktop.draft().unwrap().canvas_config.background, None);
    }

    #[test]
    fn a_line_width_wraps_the_words_and_zero_keeps_one_line() {
        let mut bare = shell();
        assert_eq!(
            bare.set_line_width(40.0).unwrap_err().to_string(),
            "no project is open"
        );

        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        let mut video = MaterialVideo::new("picture");
        video.path = "clip.mp4".to_owned();
        draft.materials.videos.push(video);
        draft.tracks[0].segments.push(Segment::new(
            "picture",
            "picture",
            TimeRange::new(0, 1_000_000),
        ));
        draft.materials.texts.push(MaterialText {
            id: MaterialId::new("caption"),
            content: "AB".to_owned(),
            ..MaterialText::default()
        });
        let mut words = Track::main("words");
        words.kind = TrackType::Text;
        words.segments.push(Segment::new(
            "caption",
            "caption",
            TimeRange::new(0, 1_000_000),
        ));
        draft.tracks.push(words);
        Folder::new(draft).save(&temp.path).unwrap();

        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.use_face("sample.ttf", SAMPLE_TTF.to_vec());
        desktop.select(Some(SegmentId::new("picture")));
        assert_eq!(
            desktop.set_line_width(40.0).unwrap_err().to_string(),
            "a text style belongs on text"
        );

        desktop.select(Some(SegmentId::new("caption")));
        desktop.set_text_size(64.0).unwrap();
        let one_line = desktop.preview_bytes().unwrap();
        desktop.set_line_width(1.0).unwrap();
        let inspection = desktop.inspection().unwrap();
        assert_eq!(inspection.line_width, 1.0);
        assert_eq!(inspection.text_align, 0);
        assert_eq!(inspection.text_size, 64.0);
        assert_eq!(inspection.text.as_deref(), Some("AB"));
        let wrapped = desktop.preview_bytes().unwrap();
        assert_ne!(digest(&one_line.rgba), digest(&wrapped.rgba));
        desktop.undo().unwrap();
        assert_eq!(desktop.inspection().unwrap().line_width, 0.0);
        assert_eq!(
            desktop.set_line_width(-1.0).unwrap_err().to_string(),
            "the line width must be from zero through 8192"
        );
        assert_eq!(
            desktop.set_line_width(8_193.0).unwrap_err().to_string(),
            "the line width must be from zero through 8192"
        );
        desktop.set_line_width(8_192.0).unwrap();
        assert_eq!(desktop.inspection().unwrap().line_width, 8_192.0);

        let lane = desktop.draft().unwrap().tracks[1].id.clone();
        desktop
            .toggle_attribute(lane, TrackAttribute::LOCK)
            .unwrap();
        assert_eq!(
            desktop.set_line_width(40.0).unwrap_err().to_string(),
            "the track is locked"
        );
        assert_eq!(desktop.inspection().unwrap().line_width, 8_192.0);
    }

    #[test]
    fn the_inspector_edits_through_requests_and_leaves_words_on_an_empty_string() {
        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        let mut text = MaterialText::new("caption");
        text.content = "Hello".to_owned();
        text.words.text = vec!["Hello".to_owned()];
        text.words.start_time = vec![0];
        text.words.end_time = vec![100];
        draft.materials.texts.push(text);
        draft.tracks[0].segments.push(Segment::new(
            "caption-segment",
            "caption",
            TimeRange::new(0, 1_000_000),
        ));
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.select(Some(SegmentId::new("caption-segment")));
        let inspection = desktop.inspection().unwrap();
        assert_eq!(inspection.text.as_deref(), Some("Hello"));
        assert_eq!(inspection.words.text, ["Hello"]);
        desktop.set_text(String::new()).unwrap();
        let inspection = desktop.inspection().unwrap();
        assert_eq!(inspection.text.as_deref(), Some(""));
        assert_eq!(inspection.words.text, ["Hello"]);
        desktop
            .toggle_attribute(desktop.lanes()[0].track_id.clone(), TrackAttribute::LOCK)
            .unwrap();
        assert!(desktop.set_text("nope".to_owned()).is_err());
        assert_eq!(desktop.inspection().unwrap().text.as_deref(), Some(""));
    }

    #[test]
    fn export_writes_the_playhead_png_and_refuses_a_codec_it_cannot_encode() {
        let temp = Temp::new();
        let missing = shell().export(ExportRequest {
            path: temp.path.join("missing.png"),
            short_side: 480,
            codec: CodecId::Png,
            replace: false,
            ..ExportRequest::default()
        });
        assert!(matches!(missing, Err(ShellError::NoProject)));
        assert!(!temp.path.join("missing.png").exists());

        color_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let before = desktop.draft().unwrap().clone();
        let destination = temp.path.join("out.png");
        let declined = desktop.export(ExportRequest {
            path: destination.clone(),
            short_side: 480,
            codec: CodecId::Vp8,
            replace: false,
            ..ExportRequest::default()
        });
        assert!(matches!(declined, Err(ShellError::CodecDeclined(_))));
        assert!(!destination.exists());
        assert_eq!(desktop.draft().unwrap(), &before);

        desktop
            .export(ExportRequest {
                path: destination.clone(),
                short_side: 480,
                codec: CodecId::Png,
                replace: false,
                ..ExportRequest::default()
            })
            .unwrap();
        let written = std::fs::read(&destination).unwrap();
        let (width, height, samples) = png_samples(&written);
        // The short side is 480; the 2:1 canvas keeps its shape.
        assert_eq!((width, height), (960, 480));
        assert!(
            samples
                .chunks(4)
                .all(|pixel| pixel == [0x33, 0x66, 0x99, 0xFF])
        );
        assert_eq!(desktop.draft().unwrap(), &before);

        let kept = std::fs::read(&destination).unwrap();
        let overwrite = desktop.export(ExportRequest {
            path: destination.clone(),
            short_side: 480,
            codec: CodecId::Png,
            replace: false,
            ..ExportRequest::default()
        });
        assert!(matches!(overwrite, Err(ShellError::Overwrite(_))));
        assert_eq!(std::fs::read(&destination).unwrap(), kept);

        desktop.seek(Micros(2_000_000));
        desktop
            .export(ExportRequest {
                path: destination.clone(),
                short_side: 480,
                codec: CodecId::Png,
                replace: true,
                ..ExportRequest::default()
            })
            .unwrap();
        let (width, height, samples) = png_samples(&std::fs::read(&destination).unwrap());
        assert_eq!((width, height), (960, 480));
        assert!(samples.iter().all(|sample| *sample == 0));
        assert_eq!(desktop.draft().unwrap(), &before);

        let step = desktop.export(ExportRequest {
            path: destination.clone(),
            short_side: 1000,
            codec: CodecId::Png,
            replace: true,
            ..ExportRequest::default()
        });
        assert!(matches!(step, Err(ShellError::Unavailable(_))));
        assert_eq!(SHORT_SIDES, [480, 720, 1080, 1440, 2160, 4320]);
    }

    /// A two-second color project, 64 by 32, at 30 frames a second.
    fn movie_project(dir: &Path) {
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 64;
        draft.canvas_config.height = 32;
        draft.materials.material_colors.push(MaterialColorConfig {
            id: "color".into(),
            is_color_clip: true,
            solid_color: "0xFF336699".to_owned(),
            ..MaterialColorConfig::default()
        });
        draft.tracks[0]
            .segments
            .push(Segment::new("one", "color", TimeRange::new(0, 1_000_000)));
        draft.tracks[0].segments.push(Segment::new(
            "two",
            "color",
            TimeRange::new(1_000_000, 1_000_000),
        ));
        draft.duration = Micros(2_000_000);
        Folder::new(draft).save(dir).unwrap();
    }

    fn land_job(desktop: &mut Shell) {
        // A debug Rust encoder on a small CI runner takes well over a minute.
        let until = std::time::Instant::now() + std::time::Duration::from_secs(300);
        while desktop.poll_job() && std::time::Instant::now() < until {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(desktop.job().is_none(), "the job did not finish");
    }

    #[test]
    fn an_import_job_copies_the_file_and_lands_it_in_the_library() {
        let temp = Temp::new();
        empty_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let picture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/engine/fixtures/pattern.png");
        desktop.start_import(&picture).unwrap();
        assert_eq!(
            desktop.job().map(|job| job.label),
            Some("Import".to_owned())
        );
        land_job(&mut desktop);
        assert_eq!(desktop.notice(), "Imported pattern.png");
        assert!(temp.path.join("Resources/pattern.png").is_file());
        assert_eq!(desktop.draft().unwrap().materials.videos.len(), 1);
        assert_eq!(desktop.take_imported(), Some(super::ImportedKind::Video));
        assert_eq!(desktop.take_imported(), None);
    }

    #[test]
    fn a_cancelled_import_leaves_no_copy_in_the_project() {
        let temp = Temp::new();
        empty_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let picture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/engine/fixtures/pattern.png");
        desktop.start_import(&picture).unwrap();
        desktop.cancel_job();
        land_job(&mut desktop);
        assert!(!temp.path.join("Resources/pattern.png").exists());
        assert!(desktop.draft().unwrap().materials.videos.is_empty());
        assert_eq!(desktop.take_imported(), None);

        // A file that is not media is refused when the job lands, and
        // nothing is copied.
        let text = temp.path.join("notes.txt");
        std::fs::write(&text, "not a picture").unwrap();
        let outside = Temp::new();
        let moved = outside.path.join("notes.txt");
        std::fs::rename(&text, &moved).unwrap();
        desktop.start_import(&moved).unwrap();
        land_job(&mut desktop);
        assert!(!desktop.notice().is_empty());
        assert!(!temp.path.join("Resources/notes.txt").exists());
    }

    #[test]
    fn a_movie_export_runs_as_a_job_and_lands_an_mp4() {
        let temp = Temp::new();
        movie_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let destination = temp.path.join("cut.mp4");
        desktop
            .export(ExportRequest {
                path: destination.clone(),
                short_side: 480,
                ..ExportRequest::default()
            })
            .unwrap();
        assert_eq!(
            desktop.job().map(|job| job.label),
            Some("Export".to_owned())
        );
        land_job(&mut desktop);
        assert_eq!(desktop.notice(), "Exported cut.mp4");
        assert_eq!(desktop.exported(), Some(destination.as_path()));
        let bytes = std::fs::read(&destination).unwrap();
        assert_eq!(&bytes[4..8], b"ftyp");
        assert!(!temp.path.join(".cut.mp4.part").exists());

        // The selected clip alone is half as long, so the file is smaller.
        desktop.select(Some(SegmentId::new("two")));
        let half = temp.path.join("half.mp4");
        desktop
            .export(ExportRequest {
                path: half.clone(),
                short_side: 480,
                span: super::ExportSpan::Selected,
                sound: false,
                ..ExportRequest::default()
            })
            .unwrap();
        assert!(
            desktop.exported().is_none(),
            "a new export forgets the last"
        );
        land_job(&mut desktop);
        assert_eq!(desktop.exported(), Some(half.as_path()));
        assert!(std::fs::metadata(&half).unwrap().len() < bytes.len() as u64);

        // HEVC is offered only where an encoder opens, and then it writes.
        if desktop.encodes(CodecId::Hevc) {
            let hevc = temp.path.join("hevc.mp4");
            desktop
                .export(ExportRequest {
                    path: hevc.clone(),
                    short_side: 480,
                    codec: CodecId::Hevc,
                    ..ExportRequest::default()
                })
                .unwrap();
            land_job(&mut desktop);
            assert_eq!(desktop.notice(), "Exported hevc.mp4");
            assert_eq!(&std::fs::read(&hevc).unwrap()[4..8], b"ftyp");
        }
    }

    #[test]
    fn an_export_writes_a_mov_the_sound_alone_a_gif_and_a_caption_file() {
        let temp = Temp::new();
        movie_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let run = |desktop: &mut Shell, request: ExportRequest| {
            desktop.export(request).unwrap();
            land_job(desktop);
            assert!(
                desktop.notice().starts_with("Exported"),
                "{}",
                desktop.notice()
            );
        };

        // A QuickTime movie with PCM sound, a typed bitrate held each
        // second, and its captions beside it.
        let mov = temp.path.join("cut.mov");
        run(
            &mut desktop,
            ExportRequest {
                path: mov.clone(),
                short_side: 480,
                bitrate: Some(3_000_000),
                constant_rate: true,
                audio: super::SoundFormat {
                    codec: CodecId::Pcm,
                    ..super::SoundFormat::default()
                },
                captions: Some(SidecarKind::Srt),
                ..ExportRequest::default()
            },
        );
        let bytes = std::fs::read(&mov).unwrap();
        assert_eq!(&bytes[4..12], b"ftypqt  ");
        assert!(bytes.windows(4).any(|w| w == b"sowt"));
        assert!(temp.path.join("cut.srt").is_file());

        // PCM and ProRes are refused in an MP4.
        let pcm_mp4 = desktop.export(ExportRequest {
            path: temp.path.join("pcm.mp4"),
            short_side: 480,
            audio: super::SoundFormat {
                codec: CodecId::Pcm,
                ..super::SoundFormat::default()
            },
            ..ExportRequest::default()
        });
        assert_eq!(
            pcm_mp4.unwrap_err().to_string(),
            "PCM sound is written in a MOV"
        );
        let prores_mp4 = desktop.export(ExportRequest {
            path: temp.path.join("prores.mp4"),
            short_side: 480,
            codec: CodecId::Prores,
            ..ExportRequest::default()
        });
        assert_eq!(
            prores_mp4.unwrap_err().to_string(),
            "ProRes is written in a MOV"
        );
        if desktop.encodes(CodecId::Prores) {
            let prores = temp.path.join("prores.mov");
            run(
                &mut desktop,
                ExportRequest {
                    path: prores.clone(),
                    short_side: 480,
                    codec: CodecId::Prores,
                    prores: ProresProfile::High,
                    ..ExportRequest::default()
                },
            );
            assert!(
                std::fs::read(&prores)
                    .unwrap()
                    .windows(4)
                    .any(|w| w == b"apch")
            );
        }

        // With the pictures off, the sound alone: an M4A for AAC, a WAV for
        // PCM, whatever extension the name had.
        run(
            &mut desktop,
            ExportRequest {
                path: temp.path.join("sound.mp4"),
                picture: false,
                ..ExportRequest::default()
            },
        );
        let m4a = std::fs::read(temp.path.join("sound.m4a")).unwrap();
        assert_eq!(&m4a[4..12], b"ftypM4A ");
        assert_eq!(
            desktop.exported(),
            Some(temp.path.join("sound.m4a").as_path())
        );
        run(
            &mut desktop,
            ExportRequest {
                path: temp.path.join("sound.wav"),
                picture: false,
                audio: super::SoundFormat {
                    codec: CodecId::Pcm,
                    sample_rate: 44_100,
                    channels: 1,
                    bitrate: 0,
                },
                ..ExportRequest::default()
            },
        );
        assert_eq!(
            &std::fs::read(temp.path.join("sound.wav")).unwrap()[..4],
            b"RIFF"
        );

        // A GIF alone, and then again beside a movie, which finds it taken.
        let gif = super::GifExport {
            short_side: 240,
            ..super::GifExport::default()
        };
        run(
            &mut desktop,
            ExportRequest {
                path: temp.path.join("loop.gif"),
                picture: false,
                sound: false,
                gif: Some(gif),
                ..ExportRequest::default()
            },
        );
        let bytes = std::fs::read(temp.path.join("loop.gif")).unwrap();
        assert_eq!(&bytes[..6], b"GIF89a");
        assert!(bytes.windows(11).any(|w| w == b"NETSCAPE2.0"));
        let taken = desktop.export(ExportRequest {
            path: temp.path.join("loop.mp4"),
            short_side: 480,
            gif: Some(gif),
            ..ExportRequest::default()
        });
        assert!(matches!(taken, Err(ShellError::Overwrite(path)) if path.ends_with("loop.gif")));
        let nothing = desktop.export(ExportRequest {
            path: temp.path.join("none.mp4"),
            picture: false,
            sound: false,
            ..ExportRequest::default()
        });
        assert!(nothing.is_err());
        assert!(!temp.path.join("loop.mp4").exists());
    }

    #[test]
    fn a_still_lands_the_canvas_at_the_playhead_as_png_or_jpeg() {
        let temp = Temp::new();
        movie_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.seek(Micros(1_500_000));
        let png = temp.path.join("frame.png");
        desktop.export_still(&png).unwrap();
        assert_eq!(desktop.job().map(|job| job.label), Some("Still".to_owned()));
        land_job(&mut desktop);
        assert_eq!(desktop.notice(), "Saved frame.png");
        assert_eq!(desktop.exported(), Some(png.as_path()));
        assert!(!temp.path.join(".frame.png.part").exists());
        let picture = image::open(&png).unwrap().to_rgba8();
        assert_eq!(picture.dimensions(), (64, 32));
        let [red, green, blue, alpha] = picture.get_pixel(32, 16).0;
        let near = |got: u8, want: u8| got.abs_diff(want) <= 3;
        assert!(
            near(red, 0x33) && near(green, 0x66) && near(blue, 0x99) && alpha == 255,
            "{:?}",
            (red, green, blue, alpha)
        );

        let jpeg = temp.path.join("frame.JPG");
        desktop.export_still(&jpeg).unwrap();
        land_job(&mut desktop);
        let bytes = std::fs::read(&jpeg).unwrap();
        assert_eq!(&bytes[..2], &[0xff, 0xd8]);

        assert_eq!(
            desktop.export_still(&temp.path).unwrap_err().to_string(),
            ShellError::Io("the still path is a folder".to_owned()).to_string()
        );
    }

    #[test]
    fn a_cancelled_movie_export_leaves_no_file_and_nothing_selected_is_refused() {
        let temp = Temp::new();
        movie_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.select(None);
        let destination = temp.path.join("cut.mp4");
        let nothing = desktop.export(ExportRequest {
            path: destination.clone(),
            short_side: 480,
            span: super::ExportSpan::Selected,
            ..ExportRequest::default()
        });
        assert_eq!(nothing.unwrap_err().to_string(), "no clip is selected");
        assert!(desktop.job().is_none());

        desktop
            .export(ExportRequest {
                path: destination.clone(),
                short_side: 480,
                ..ExportRequest::default()
            })
            .unwrap();
        desktop.cancel_job();
        // The worker sees the cancel at its next frame and removes its part;
        // the job stays until then.
        land_job(&mut desktop);
        assert!(!temp.path.join(".cut.mp4.part").exists());
        assert!(!destination.exists());
        assert!(desktop.exported().is_none());
    }

    #[test]
    fn export_writes_a_placed_still() {
        let temp = Temp::new();
        let picture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/engine/fixtures/pattern.png");
        export_still(&temp.path, &picture);

        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../local-files");
        let Some(local) = smallest_png(&root) else {
            return;
        };
        let local_dir = Temp::new();
        export_still(&local_dir.path, &local);
    }

    fn png_samples(bytes: &[u8]) -> (u32, u32, Vec<u16>) {
        assert!(bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]));
        let mut decoder = cutlass_png::PngDecoder::open(&CodecParams::video(CodecId::Png, 1, 1))
            .expect("decoder");
        let picture = decoder
            .decode_picture(&Packet::video(bytes.to_vec(), 0, 1, 30, true))
            .expect("png");
        let image = &picture[0].image;
        assert_eq!(image.color, cutlass_decode::ColorType::Rgba);
        (image.width, image.height, image.samples.clone())
    }

    /// A grey frame with a vignette clip over it, exported as a PNG at the
    /// first frame: with eframe's GPU the corners are darker than the middle.
    #[test]
    fn an_export_draws_an_effect_clip_through_the_preview_gpu() {
        let instance = eframe::wgpu::Instance::new(
            eframe::wgpu::InstanceDescriptor::new_without_display_handle(),
        );
        let state = match pollster::block_on(eframe::egui_wgpu::RenderState::create(
            &eframe::egui_wgpu::WgpuConfiguration::default(),
            &instance,
            None,
            eframe::egui_wgpu::RendererOptions::default(),
        )) {
            Ok(state) => state,
            Err(error) if std::env::var_os(cutlass_render::REQUIRE_GPU).is_some() => {
                panic!("a GPU is required: {error}")
            }
            Err(_) => {
                eprintln!("skipped: no GPU");
                return;
            }
        };
        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 16;
        draft.canvas_config.height = 16;
        draft.materials.material_colors.push(MaterialColorConfig {
            id: "grey".into(),
            is_color_clip: true,
            solid_color: "0xFF808080".to_owned(),
            ..MaterialColorConfig::default()
        });
        draft.tracks[0]
            .segments
            .push(Segment::new("grey", "grey", TimeRange::new(0, 1_000_000)));
        draft
            .materials
            .video_effects
            .push(cutlass_model::MaterialVideoEffect {
                id: "fx".into(),
                effect_id: "recipe/vignette".to_owned(),
                ..cutlass_model::MaterialVideoEffect::default()
            });
        let mut row = Track::main("effects");
        row.kind = TrackType::Effect;
        let mut effect = Segment::new("fx", "fx", TimeRange::new(0, 1_000_000));
        effect.render_index = 11_000;
        row.segments.push(effect);
        draft.tracks.push(row);
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.attach_gpu(&state);
        let destination = temp.path.join("out.png");
        desktop
            .export(ExportRequest {
                path: destination.clone(),
                short_side: 480,
                codec: CodecId::Png,
                replace: false,
                ..ExportRequest::default()
            })
            .unwrap();
        let (width, height, samples) = png_samples(&std::fs::read(&destination).unwrap());
        let at = |x: u32, y: u32| samples[usize::try_from((y * width + x) * 4).unwrap()];
        let (middle, corner) = (at(width / 2, height / 2), at(0, 0));
        assert!(corner + 40 < middle, "corner {corner}, middle {middle}");
    }

    fn export_still(dir: &Path, media: &Path) {
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 16;
        draft.canvas_config.height = 16;
        let mut video = MaterialVideo::new("still");
        video.path = media.display().to_string();
        video.duration = Micros(1_000_000);
        draft.materials.videos.push(video);
        let mut segment = Segment::new("still", "still", TimeRange::new(0, 1_000_000));
        segment.source_range = Some(TimeRange::new(0, 1_000_000));
        draft.tracks[0].segments.push(segment);
        Folder::new(draft).save(dir).unwrap();
        let mut desktop = shell();
        desktop.open_folder(dir).unwrap();
        let preview = desktop.preview_bytes().unwrap();
        assert!(
            preview.rgba.iter().any(|byte| *byte != 0),
            "the still did not open"
        );
        let destination = dir.join("out.png");
        desktop
            .export(ExportRequest {
                path: destination.clone(),
                short_side: 480,
                codec: CodecId::Png,
                replace: false,
                ..ExportRequest::default()
            })
            .unwrap();
        let (width, height, samples) = png_samples(&std::fs::read(&destination).unwrap());
        assert_eq!((width, height), (preview.width * 30, preview.height * 30));
        for y in 0..height {
            for x in 0..width {
                let sx = x * preview.width / width;
                let sy = y * preview.height / height;
                let from = usize::try_from((sy * preview.width + sx) * 4).unwrap();
                let to = usize::try_from((y * width + x) * 4).unwrap();
                let pixel = &preview.rgba[from..from + 4];
                assert_eq!(
                    samples[to..to + 4],
                    [
                        u16::from(pixel[0]),
                        u16::from(pixel[1]),
                        u16::from(pixel[2]),
                        u16::from(pixel[3]),
                    ]
                );
            }
        }
    }

    fn smallest_png(root: &Path) -> Option<PathBuf> {
        let entries = root.read_dir().ok()?;
        let mut best: Option<(u64, PathBuf)> = None;
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if !path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("png"))
            {
                continue;
            }
            let Ok(meta) = entry.metadata() else {
                continue;
            };
            if !meta.is_file() || meta.len() == 0 || meta.len() > 8_000_000 {
                continue;
            }
            if best.as_ref().is_none_or(|current| meta.len() < current.0) {
                best = Some((meta.len(), path));
            }
        }
        best.map(|(_, path)| path)
    }

    fn trim_project(dir: &Path) {
        let mut draft = Draft::empty("draft");
        draft.config.maintrack_adsorb = false;
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        let mut video = MaterialVideo::new("media");
        video.duration = Micros(10_000_000);
        draft.materials.videos.push(video);
        let mut first = Segment::new("clip", "media", TimeRange::new(0, 1_000_000));
        first.source_range = Some(TimeRange::new(0, 1_000_000));
        let mut second = Segment::new("next", "media", TimeRange::new(3_000_000, 500_000));
        second.source_range = Some(TimeRange::new(0, 500_000));
        draft.tracks[0].segments.push(first);
        draft.tracks[0].segments.push(second);
        Folder::new(draft).save(dir).unwrap();
    }

    fn lane_time(desktop: &Shell, time: f64) -> f64 {
        let origin = desktop.view_origin().0 as f64;
        (time - origin) * desktop.pixel_size()
    }

    #[test]
    fn a_guide_pulls_a_move_and_delete_removes_it() {
        let temp = Temp::new();
        trim_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.add_guide(Micros(1_500_000)).unwrap();
        let id = desktop.draft().unwrap().tracks[0].segments[0].id.clone();
        desktop.preview_move(id.clone(), lane_time(&desktop, 1_501_000.0), 24.0);
        let preview = desktop.move_preview().expect("preview");
        assert_eq!(preview.start, Micros(1_500_000));
        desktop
            .trim_edge(id.clone(), TrimEdge::End, lane_time(&desktop, 1_501_000.0))
            .unwrap();
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0]
                .target
                .duration,
            Micros(1_500_000)
        );
        desktop.undo().unwrap();
        desktop.select(Some(id));
        desktop.select_guide(Micros(1_500_000));
        assert!(desktop.selection().is_empty());
        desktop.run_shortcut("delete").unwrap();
        assert!(desktop.guides().is_empty());
        desktop.add_guide(Micros(2_000_000)).unwrap();
        desktop.toggle_guide_lock();
        let locked = desktop.add_guide(Micros(2_100_000)).unwrap_err();
        assert_eq!(locked.to_string(), "the guides are locked");
        assert_eq!(desktop.guides(), &[Micros(2_000_000)]);
        desktop.toggle_guide_lock();
        desktop.toggle_guides();
        assert!(desktop.guides().is_empty());
        desktop
            .trim_edge(
                desktop.draft().unwrap().tracks[0].segments[0].id.clone(),
                TrimEdge::End,
                lane_time(&desktop, 2_001_000.0),
            )
            .unwrap();
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0]
                .target
                .duration,
            Micros(2_001_000)
        );
        desktop.toggle_guides();
        desktop
            .trim_edge(
                desktop.draft().unwrap().tracks[0].segments[0].id.clone(),
                TrimEdge::End,
                lane_time(&desktop, 2_000_500.0),
            )
            .unwrap();
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0]
                .target
                .duration,
            Micros(2_000_000)
        );
        desktop.close_project().unwrap();
        desktop.open_folder(&temp.path).unwrap();
        assert!(desktop.guides().is_empty());
    }

    #[test]
    fn a_horizontal_guide_stays_until_delete() {
        let temp = Temp::new();
        trim_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.add_horizontal(400).unwrap();
        desktop.add_horizontal(400).unwrap();
        assert_eq!(desktop.horizontal_guides(), &[400]);
        desktop.add_guide(Micros(1_000_000)).unwrap();
        assert!(desktop.selected_horizontal().is_none());
        desktop.select_horizontal(400);
        assert!(desktop.selected_guide().is_none());
        let id = desktop.draft().unwrap().tracks[0].segments[0].id.clone();
        desktop.select(Some(id));
        desktop.select_horizontal(400);
        assert!(desktop.selection().is_empty());
        desktop.run_shortcut("delete").unwrap();
        assert!(desktop.horizontal_guides().is_empty());
        assert_eq!(desktop.guides(), &[Micros(1_000_000)]);
        desktop.add_horizontal(250).unwrap();
        desktop.toggle_guide_lock();
        let locked = desktop.add_horizontal(500).unwrap_err();
        assert_eq!(locked.to_string(), "the guides are locked");
        desktop.toggle_guide_lock();
        desktop.toggle_guides();
        assert!(desktop.horizontal_guides().is_empty());
        desktop.toggle_guides();
        assert_eq!(desktop.horizontal_guides(), &[250]);
        desktop.close_project().unwrap();
        desktop.open_folder(&temp.path).unwrap();
        assert!(desktop.horizontal_guides().is_empty());
        assert!(desktop.guides().is_empty());
    }

    #[test]
    fn trimming_an_edge_updates_the_source_and_a_locked_lane_refuses() {
        let temp = Temp::new();
        trim_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let id = desktop.draft().unwrap().tracks[0].segments[0].id.clone();
        let revision = desktop.revision();
        desktop
            .trim_edge(id.clone(), TrimEdge::End, lane_time(&desktop, 1_000_000.0))
            .unwrap();
        assert_eq!(desktop.revision(), revision);
        desktop
            .trim_edge(id.clone(), TrimEdge::End, lane_time(&desktop, 2_000_000.0))
            .unwrap();
        let segment = &desktop.draft().unwrap().tracks[0].segments[0];
        assert_eq!(segment.target, TimeRange::new(0, 2_000_000));
        assert_eq!(segment.source_range, Some(TimeRange::new(0, 2_000_000)));
        desktop.undo().unwrap();
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0]
                .target
                .duration,
            Micros(1_000_000)
        );
        desktop
            .trim_edge(id.clone(), TrimEdge::End, lane_time(&desktop, 4_000_000.0))
            .unwrap();
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0].target.end(),
            Micros(3_000_000)
        );
        desktop.undo().unwrap();
        desktop.seek(Micros(1_500_000));
        desktop.preview_edge(id.clone(), TrimEdge::End, lane_time(&desktop, 1_500_500.0));
        assert_eq!(desktop.snap_line(), Some(Micros(1_500_000)));
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0]
                .target
                .duration,
            Micros(1_000_000)
        );
        desktop.cancel_clip_drag();
        let track = desktop.lanes()[0].track_id.clone();
        desktop
            .toggle_attribute(track, TrackAttribute::LOCK)
            .unwrap();
        let error = desktop
            .trim_edge(id, TrimEdge::End, lane_time(&desktop, 2_000_000.0))
            .unwrap_err();
        assert_eq!(error.to_string(), "the track is locked");
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0]
                .target
                .duration,
            Micros(1_000_000)
        );
    }

    #[test]
    fn crop_keeps_the_middle_of_a_picture() {
        let mut bare = shell();
        assert!(!bare.crop_enabled());
        assert_eq!(
            bare.toggle_crop().unwrap_err().to_string(),
            "no project is open"
        );

        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 4;
        draft.canvas_config.height = 4;
        draft.config.maintrack_adsorb = false;
        let mut video = MaterialVideo::new("picture");
        video.path = "clip".to_owned();
        video.duration = Micros(2_000_000);
        video.crop_ratio = "16:9".to_owned();
        video.crop_scale = 2.0;
        draft.materials.videos.push(video);
        draft.tracks[0].segments.push(Segment::new(
            "picture",
            "picture",
            TimeRange::new(0, 2_000_000),
        ));
        draft.tracks[0].segments.push(Segment::new(
            "again",
            "picture",
            TimeRange::new(3_000_000, 1_000_000),
        ));
        draft.tracks[0].segments.push(Segment::new(
            "plain",
            "plain",
            TimeRange::new(5_000_000, 100_000),
        ));
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("tone"),
            ..MaterialAudio::default()
        });
        let mut sound = Track::main("sound");
        sound.kind = TrackType::Audio;
        sound
            .segments
            .push(Segment::new("tone", "tone", TimeRange::new(0, 1_000_000)));
        draft.tracks.push(sound);
        draft.materials.texts.push(MaterialText {
            id: MaterialId::new("caption"),
            content: "hi".to_owned(),
            ..MaterialText::default()
        });
        let mut words = Track::main("words");
        words.kind = TrackType::Text;
        words.segments.push(Segment::new(
            "caption",
            "caption",
            TimeRange::new(0, 1_000_000),
        ));
        draft.tracks.push(words);
        draft.materials.drafts.push(MaterialDraft {
            id: MaterialId::new("compound"),
            ..MaterialDraft::default()
        });
        let mut compound = Track::main("compound");
        compound.kind = TrackType::Composition;
        compound.segments.push(Segment::new(
            "compound",
            "compound",
            TimeRange::new(0, 1_000_000),
        ));
        draft.tracks.push(compound);
        Folder::new(draft).save(&temp.path).unwrap();

        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.select(Some(SegmentId::new("plain")));
        assert!(!desktop.crop_enabled());
        assert_eq!(
            desktop.toggle_crop().unwrap_err().to_string(),
            "a crop belongs on a picture"
        );
        desktop.select(Some(SegmentId::new("tone")));
        assert!(!desktop.crop_enabled());
        assert!(desktop.reverse_enabled());
        desktop.select(Some(SegmentId::new("caption")));
        assert!(desktop.rotate_enabled());
        assert!(!desktop.crop_enabled());

        desktop.select(Some(SegmentId::new("picture")));
        assert!(desktop.crop_enabled());
        assert!(!desktop.cropped());
        desktop.toggle_crop().unwrap();
        assert!(desktop.cropped());
        // "again" shows the same video; the crop is this clip's alone.
        let draft = desktop.draft().unwrap();
        let own = &draft.tracks[0].segments[0].material_id;
        let video = draft
            .materials
            .videos
            .iter()
            .find(|video| &video.id == own)
            .unwrap();
        assert_eq!(video.crop, super::center_crop());
        assert_eq!(video.crop_ratio, "16:9");
        assert_eq!(video.crop_scale, 2.0);
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0].target,
            TimeRange::new(0, 2_000_000)
        );
        desktop.select(Some(SegmentId::new("again")));
        assert!(!desktop.cropped());
        desktop.select(Some(SegmentId::new("picture")));
        desktop.undo().unwrap();
        assert!(!desktop.cropped());
        desktop.redo().unwrap();
        assert!(desktop.cropped());
        desktop.toggle_crop().unwrap();
        assert!(!desktop.cropped());
        assert_eq!(
            desktop.draft().unwrap().materials.videos[0].crop,
            Crop::default()
        );

        let track = desktop.draft().unwrap().tracks[0].id.clone();
        desktop
            .toggle_attribute(track, TrackAttribute::LOCK)
            .unwrap();
        let locked = desktop.toggle_crop().unwrap_err();
        assert_eq!(locked.to_string(), "the track is locked");
        assert!(!desktop.cropped());

        desktop.select(Some(SegmentId::new("compound")));
        assert!(desktop.crop_enabled());
        assert!(desktop.mirror_enabled());
        assert!(!desktop.freeze_enabled());
        desktop.toggle_crop().unwrap();
        assert!(desktop.cropped());
        assert_eq!(
            desktop.draft().unwrap().materials.drafts[0].crop,
            super::center_crop()
        );
        // The shared video and the picture's own copy; a compound adds none.
        assert_eq!(desktop.draft().unwrap().materials.videos.len(), 2);
        desktop.toggle_crop().unwrap();
        assert_eq!(
            desktop.draft().unwrap().materials.drafts[0].crop,
            Crop::default()
        );
    }

    struct CaptionStandIn {
        words: Vec<RecognizedWord>,
    }

    impl Recognizer for CaptionStandIn {
        fn sample_rate(&self) -> u32 {
            16_000
        }

        fn recognize(&self, pcm: &Pcm) -> Result<Vec<Utterance>, RecognizeError> {
            if pcm.sample_rate != self.sample_rate() || pcm.samples.is_empty() {
                return Err(RecognizeError::SampleRate {
                    expected: self.sample_rate(),
                    actual: pcm.sample_rate,
                });
            }
            Ok(vec![Utterance {
                words: self.words.clone(),
            }])
        }
    }

    struct CaptionTone;

    impl PcmSource for CaptionTone {
        fn read(&mut self, range: &TimeRange) -> Result<Pcm, PrepareError> {
            if range.duration.0 <= 0 || range.start.0 < 0 {
                return Err(PrepareError::Range);
            }
            Ok(Pcm {
                sample_rate: 16_000,
                samples: vec![0.0; 16],
            })
        }
    }

    #[test]
    fn split_scenes_cuts_a_video_where_the_picture_changes() {
        let mut bare = shell();
        assert!(!bare.split_scenes_enabled());
        assert_eq!(
            bare.split_scenes().unwrap_err().to_string(),
            "no project is open"
        );

        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.config.maintrack_adsorb = false;
        let mut video = MaterialVideo::new("picture");
        video.path = "clip".to_owned();
        video.duration = Micros(1_200_000);
        draft.materials.videos.push(video);
        let mut picture = Segment::new("picture", "picture", TimeRange::new(0, 1_200_000));
        picture.source_range = Some(TimeRange::new(0, 1_200_000));
        draft.tracks[0].segments.push(picture);
        let mut joined = Segment::new("joined", "picture", TimeRange::new(8_000_000, 1_200_000));
        joined.source_range = Some(TimeRange::new(0, 1_200_000));
        joined.extra_material_refs.push(MaterialId::new("blend"));
        draft.tracks[0].segments.push(joined);
        draft.materials.transitions.push(MaterialTransition {
            id: MaterialId::new("blend"),
            duration: Micros(500_000),
            is_overlap: false,
            ..MaterialTransition::default()
        });
        let mut held = MaterialVideo::new("held");
        held.path = "clip".to_owned();
        held.duration = Micros(1_000_000);
        held.freeze = Some(FreezeInfo::default());
        draft.materials.videos.push(held);
        draft.tracks[0].segments.push(Segment::new(
            "held",
            "held",
            TimeRange::new(2_000_000, 1_000_000),
        ));
        let mut silent = MaterialVideo::new("silent");
        silent.duration = Micros(1_000_000);
        draft.materials.videos.push(silent);
        draft.tracks[0].segments.push(Segment::new(
            "silent",
            "silent",
            TimeRange::new(4_000_000, 1_000_000),
        ));
        draft.tracks[0].segments.push(Segment::new(
            "plain",
            "plain",
            TimeRange::new(6_000_000, 100_000),
        ));
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("tone"),
            path: "tone".to_owned(),
            ..MaterialAudio::default()
        });
        let mut sound = Track::main("sound");
        sound.kind = TrackType::Audio;
        sound
            .segments
            .push(Segment::new("tone", "tone", TimeRange::new(0, 1_000_000)));
        draft.tracks.push(sound);
        draft.materials.texts.push(MaterialText {
            id: MaterialId::new("caption"),
            content: "hi".to_owned(),
            ..MaterialText::default()
        });
        let mut words = Track::main("words");
        words.kind = TrackType::Text;
        words.segments.push(Segment::new(
            "caption",
            "caption",
            TimeRange::new(0, 1_000_000),
        ));
        draft.tracks.push(words);
        Folder::new(draft).save(&temp.path).unwrap();

        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.select(None);
        assert_eq!(
            desktop.split_scenes().unwrap_err().to_string(),
            "nothing is selected"
        );
        desktop.select(Some(SegmentId::new("plain")));
        assert!(!desktop.split_scenes_enabled());
        assert_eq!(
            desktop.split_scenes().unwrap_err().to_string(),
            "a scene split belongs on video"
        );
        desktop.select(Some(SegmentId::new("tone")));
        assert!(!desktop.split_scenes_enabled());
        desktop.select(Some(SegmentId::new("caption")));
        assert!(!desktop.split_scenes_enabled());

        desktop.select(Some(SegmentId::new("silent")));
        assert!(desktop.split_scenes_enabled());
        assert_eq!(
            desktop.split_scenes().unwrap_err().to_string(),
            "the clip has no picture"
        );
        desktop.select(Some(SegmentId::new("held")));
        assert_eq!(
            desktop.split_scenes().unwrap_err().to_string(),
            "a held frame has one scene"
        );

        let main = desktop.draft().unwrap().tracks[0].id.clone();
        desktop
            .toggle_attribute(main.clone(), TrackAttribute::LOCK)
            .unwrap();
        desktop.select(Some(SegmentId::new("picture")));
        assert_eq!(
            desktop.split_scenes().unwrap_err().to_string(),
            "the track is locked"
        );
        desktop
            .toggle_attribute(main, TrackAttribute::LOCK)
            .unwrap();

        assert_eq!(
            desktop
                .split_scenes_at(&[Micros(1_000)])
                .unwrap_err()
                .to_string(),
            "no scene change was found"
        );
        assert_eq!(desktop.draft().unwrap().tracks[0].segments.len(), 5);

        desktop
            .split_scenes_at(&[Micros(400_000), Micros(800_000)])
            .unwrap();
        let segments = &desktop.draft().unwrap().tracks[0].segments;
        assert_eq!(segments.len(), 7);
        assert_eq!(segments[0].id.as_str(), "picture");
        assert_eq!(segments[0].target, TimeRange::new(0, 400_000));
        assert_eq!(segments[0].source_range, Some(TimeRange::new(0, 400_000)));
        assert_eq!(segments[1].target, TimeRange::new(400_000, 400_000));
        assert_eq!(
            segments[1].source_range,
            Some(TimeRange::new(400_000, 400_000))
        );
        assert_eq!(segments[2].target, TimeRange::new(800_000, 400_000));
        assert_eq!(
            segments[2].source_range,
            Some(TimeRange::new(800_000, 400_000))
        );
        desktop.undo().unwrap();
        assert_eq!(desktop.draft().unwrap().tracks[0].segments.len(), 5);
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0].target,
            TimeRange::new(0, 1_200_000)
        );

        desktop.select(Some(SegmentId::new("joined")));
        desktop
            .split_scenes_at(&[Micros(8_400_000), Micros(8_800_000)])
            .unwrap();
        let joined = desktop.draft().unwrap().tracks[0]
            .segments
            .iter()
            .find(|segment| segment.id.as_str() == "joined")
            .expect("joined")
            .clone();
        assert_eq!(joined.target, TimeRange::new(8_000_000, 400_000));
        assert_eq!(
            desktop.draft().unwrap().tracks[0]
                .segments
                .iter()
                .filter(|segment| segment.target.start == Micros(8_800_000))
                .count(),
            0
        );
    }

    #[test]
    fn q_and_w_delete_either_side_of_the_playhead_and_command_a_selects_every_clip() {
        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.tracks[0]
            .segments
            .push(Segment::new("a", "media", TimeRange::new(0, 1_000_000)));
        draft.tracks[0].segments.push(Segment::new(
            "b",
            "media",
            TimeRange::new(1_000_000, 1_000_000),
        ));
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        assert!(desktop.magnet_enabled());
        let targets = |desktop: &Shell| -> Vec<TimeRange> {
            desktop.draft().unwrap().tracks[0]
                .segments
                .iter()
                .map(|segment| segment.target.clone())
                .collect()
        };

        desktop.seek(Micros(400_000));
        desktop.run_shortcut("q").unwrap();
        assert_eq!(
            targets(&desktop),
            [
                TimeRange::new(0, 600_000),
                TimeRange::new(600_000, 1_000_000)
            ]
        );
        assert_eq!(desktop.playhead(), Micros(0));
        assert_eq!(desktop.selection(), [SegmentId::new("a")]);

        desktop.seek(Micros(300_000));
        desktop.run_shortcut("w").unwrap();
        assert_eq!(
            targets(&desktop),
            [
                TimeRange::new(0, 300_000),
                TimeRange::new(300_000, 1_000_000)
            ]
        );
        assert_eq!(desktop.playhead(), Micros(300_000));

        desktop.seek(Micros(310_000));
        assert_eq!(
            desktop.run_shortcut("q").unwrap_err().to_string(),
            "delete is not available"
        );

        desktop.undo().unwrap();
        assert_eq!(targets(&desktop)[0], TimeRange::new(0, 600_000));
        let redo = if cfg!(target_os = "macos") {
            "ctrl+y"
        } else {
            "primary+y"
        };
        desktop.run_shortcut(redo).unwrap();
        assert_eq!(targets(&desktop)[0], TimeRange::new(0, 300_000));

        desktop.run_shortcut("primary+a").unwrap();
        assert_eq!(
            desktop.selection(),
            [SegmentId::new("a"), SegmentId::new("b")]
        );
    }

    #[test]
    fn the_card_plus_adds_at_the_end_of_the_matching_lane() {
        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        let mut picture = MaterialVideo::new("picture");
        picture.duration = Micros(500_000);
        draft.materials.videos.push(picture);
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("tone"),
            duration: Micros(300_000),
            ..MaterialAudio::default()
        });
        draft.tracks[0]
            .segments
            .push(Segment::new("a", "picture", TimeRange::new(0, 1_000_000)));
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();

        desktop.add_medium("picture").unwrap();
        let main = &desktop.draft().unwrap().tracks[0];
        assert_eq!(main.segments[1].target, TimeRange::new(1_000_000, 500_000));
        assert_eq!(desktop.playhead(), Micros(1_000_000));

        desktop.add_medium("tone").unwrap();
        desktop.add_medium("tone").unwrap();
        let draft = desktop.draft().unwrap();
        let sound: Vec<&Track> = draft
            .tracks
            .iter()
            .filter(|track| track.kind == TrackType::Audio)
            .collect();
        assert_eq!(sound.len(), 1);
        assert_eq!(
            sound[0]
                .segments
                .iter()
                .map(|segment| segment.target.clone())
                .collect::<Vec<_>>(),
            [TimeRange::new(0, 300_000), TimeRange::new(300_000, 300_000)]
        );
        desktop.undo().unwrap();
        desktop.undo().unwrap();
        assert!(
            desktop
                .draft()
                .unwrap()
                .tracks
                .iter()
                .all(|track| track.kind != TrackType::Audio)
        );
        assert_eq!(
            desktop.add_medium("missing").unwrap_err().to_string(),
            "the media is not in the library"
        );
    }

    #[test]
    fn shift_command_s_extracts_the_audio() {
        let mut bare = shell();
        assert_eq!(
            bare.run_shortcut("primary+shift+s")
                .unwrap_err()
                .to_string(),
            "no project is open"
        );
    }

    #[test]
    fn extract_audio_places_the_sound_on_its_own_clip() {
        let mut bare = shell();
        assert!(!bare.extract_audio_enabled());
        assert_eq!(
            bare.extract_audio().unwrap_err().to_string(),
            "no project is open"
        );

        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.config.maintrack_adsorb = false;
        let mut video = MaterialVideo::new("picture");
        video.path = "clip.mp4".to_owned();
        video.duration = Micros(1_000_000);
        draft.materials.videos.push(video);
        let mut picture = Segment::new("picture", "picture", TimeRange::new(0, 1_000_000));
        picture.source_range = Some(TimeRange::new(0, 1_000_000));
        picture.speed = 2.0;
        draft.tracks[0].segments.push(picture);
        let mut silent = MaterialVideo::new("silent");
        silent.duration = Micros(500_000);
        draft.materials.videos.push(silent);
        draft.tracks[0].segments.push(Segment::new(
            "silent",
            "silent",
            TimeRange::new(2_000_000, 500_000),
        ));
        let mut parent = MaterialVideo::new("parent");
        parent.path = "../clip.mp4".to_owned();
        parent.duration = Micros(500_000);
        draft.materials.videos.push(parent);
        draft.tracks[0].segments.push(Segment::new(
            "parent",
            "parent",
            TimeRange::new(3_000_000, 500_000),
        ));
        draft.tracks[0].segments.push(Segment::new(
            "plain",
            "plain",
            TimeRange::new(4_000_000, 100_000),
        ));
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("tone"),
            path: "tone".to_owned(),
            ..MaterialAudio::default()
        });
        let mut sound = Track::main("tone");
        sound.kind = TrackType::Audio;
        sound
            .segments
            .push(Segment::new("tone", "tone", TimeRange::new(0, 500_000)));
        draft.tracks.push(sound);
        draft.materials.texts.push(MaterialText {
            id: MaterialId::new("caption"),
            content: "hi".to_owned(),
            ..MaterialText::default()
        });
        let mut words = Track::main("words");
        words.kind = TrackType::Text;
        words.segments.push(Segment::new(
            "caption",
            "caption",
            TimeRange::new(0, 500_000),
        ));
        draft.tracks.push(words);
        Folder::new(draft).save(&temp.path).unwrap();

        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.select(None);
        assert_eq!(
            desktop.extract_audio().unwrap_err().to_string(),
            "nothing is selected"
        );
        desktop.select(Some(SegmentId::new("plain")));
        assert!(!desktop.extract_audio_enabled());
        assert_eq!(
            desktop.extract_audio().unwrap_err().to_string(),
            "an extract belongs on video"
        );
        desktop.select(Some(SegmentId::new("tone")));
        assert!(!desktop.extract_audio_enabled());
        desktop.select(Some(SegmentId::new("caption")));
        assert!(!desktop.extract_audio_enabled());

        desktop.select(Some(SegmentId::new("silent")));
        assert!(desktop.extract_audio_enabled());
        assert_eq!(
            desktop.extract_audio().unwrap_err().to_string(),
            "the clip has no sound"
        );
        desktop.select(Some(SegmentId::new("parent")));
        assert_eq!(
            desktop.extract_audio().unwrap_err().to_string(),
            "the clip has no sound"
        );

        let main = desktop.draft().unwrap().tracks[0].id.clone();
        desktop
            .toggle_attribute(main.clone(), TrackAttribute::LOCK)
            .unwrap();
        desktop.select(Some(SegmentId::new("picture")));
        assert_eq!(
            desktop.extract_sound().unwrap_err().to_string(),
            "the track is locked"
        );
        desktop
            .toggle_attribute(main, TrackAttribute::LOCK)
            .unwrap();

        desktop.extract_sound().unwrap();
        let draft = desktop.draft().unwrap();
        let lane = draft
            .tracks
            .iter()
            .find(|track| track.kind == TrackType::Audio && track.id.as_str() != "tone")
            .expect("sound");
        assert_eq!(lane.segments.len(), 1);
        assert_eq!(lane.segments[0].target, TimeRange::new(0, 1_000_000));
        assert_eq!(lane.segments[0].speed, 2.0);
        assert_eq!(draft.materials.audios.len(), 2);
        assert!(
            draft
                .materials
                .audios
                .iter()
                .any(|audio| audio.kind == "video_original_sound"
                    && audio.video_id.as_str() == "picture")
        );
        assert_eq!(
            desktop.extract_sound().unwrap_err().to_string(),
            "the sound is already on its own clip"
        );
        desktop.undo().unwrap();
        assert_eq!(desktop.draft().unwrap().materials.audios.len(), 1);
        desktop.extract_sound().unwrap();
        assert_eq!(desktop.draft().unwrap().materials.audios.len(), 2);
    }

    #[test]
    fn transcribe_places_words_on_a_text_track() {
        let mut bare = shell();
        assert!(!bare.transcribe_enabled());
        assert_eq!(
            bare.transcribe().unwrap_err().to_string(),
            "no project is open"
        );

        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.config.maintrack_adsorb = false;
        let mut video = MaterialVideo::new("picture");
        video.path = "clip".to_owned();
        video.duration = Micros(2_000_000);
        draft.materials.videos.push(video);
        draft.tracks[0].segments.push(Segment::new(
            "picture",
            "picture",
            TimeRange::new(0, 2_000_000),
        ));
        let mut fast = Segment::new("fast", "picture", TimeRange::new(3_000_000, 1_000_000));
        fast.speed = 2.0;
        fast.source_range = Some(TimeRange::new(0, 2_000_000));
        draft.tracks[0].segments.push(fast);
        let mut backward =
            Segment::new("backward", "picture", TimeRange::new(5_000_000, 2_000_000));
        backward.reverse = true;
        backward.source_range = Some(TimeRange::new(0, 2_000_000));
        draft.tracks[0].segments.push(backward);
        let mut odd = Segment::new("odd", "picture", TimeRange::new(8_000_000, 1_000_000));
        odd.speed = 0.0;
        odd.source_range = Some(TimeRange::new(0, 2_000_000));
        draft.tracks[0].segments.push(odd);
        let mut silent = MaterialVideo::new("silent");
        silent.duration = Micros(1_000_000);
        draft.materials.videos.push(silent);
        draft.tracks[0].segments.push(Segment::new(
            "silent",
            "silent",
            TimeRange::new(10_000_000, 1_000_000),
        ));
        draft.tracks[0].segments.push(Segment::new(
            "plain",
            "plain",
            TimeRange::new(12_000_000, 100_000),
        ));
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("tone"),
            path: "tone.wav".to_owned(),
            duration: Micros(1_000_000),
            ..MaterialAudio::default()
        });
        let mut sound = Track::main("sound");
        sound.kind = TrackType::Audio;
        sound.segments.push(Segment::new(
            "tone",
            "tone",
            TimeRange::new(14_000_000, 1_000_000),
        ));
        draft.tracks.push(sound);
        draft.materials.texts.push(MaterialText {
            id: MaterialId::new("caption"),
            content: "hi".to_owned(),
            ..MaterialText::default()
        });
        let mut words = Track::main("words");
        words.kind = TrackType::Text;
        words.segments.push(Segment::new(
            "caption",
            "caption",
            TimeRange::new(0, 1_000_000),
        ));
        draft.tracks.push(words);
        draft.materials.drafts.push(MaterialDraft {
            id: MaterialId::new("compound"),
            ..MaterialDraft::default()
        });
        let mut compound = Track::main("compound");
        compound.kind = TrackType::Composition;
        compound.segments.push(Segment::new(
            "compound",
            "compound",
            TimeRange::new(16_000_000, 1_000_000),
        ));
        draft.tracks.push(compound);
        Folder::new(draft).save(&temp.path).unwrap();

        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.select(None);
        assert!(!desktop.transcribe_enabled());
        assert_eq!(
            desktop.transcribe().unwrap_err().to_string(),
            "nothing is selected"
        );
        desktop.select(Some(SegmentId::new("plain")));
        assert!(!desktop.transcribe_enabled());
        assert_eq!(
            desktop.transcribe().unwrap_err().to_string(),
            "a transcript belongs on a picture or a sound"
        );
        desktop.select(Some(SegmentId::new("caption")));
        assert!(!desktop.transcribe_enabled());
        desktop.select(Some(SegmentId::new("compound")));
        assert!(!desktop.transcribe_enabled());

        desktop.select(Some(SegmentId::new("silent")));
        assert!(desktop.transcribe_enabled());
        assert_eq!(
            desktop.transcribe().unwrap_err().to_string(),
            "the clip has no sound"
        );
        assert_eq!(desktop.draft().unwrap().materials.texts.len(), 1);

        let main = desktop.draft().unwrap().tracks[0].id.clone();
        desktop
            .toggle_attribute(main.clone(), TrackAttribute::LOCK)
            .unwrap();
        desktop.select(Some(SegmentId::new("picture")));
        assert_eq!(
            desktop.transcribe().unwrap_err().to_string(),
            "the track is locked"
        );
        desktop
            .toggle_attribute(main, TrackAttribute::LOCK)
            .unwrap();
        assert_eq!(desktop.draft().unwrap().materials.texts.len(), 1);

        let mut tone = CaptionTone;
        let quiet = CaptionStandIn { words: Vec::new() };
        assert_eq!(
            desktop
                .transcribe_with(&quiet, &mut tone)
                .unwrap_err()
                .to_string(),
            "nothing was heard"
        );
        assert_eq!(desktop.draft().unwrap().materials.texts.len(), 1);

        let heard = CaptionStandIn {
            words: vec![
                RecognizedWord {
                    text: "hi".to_owned(),
                    start_ms: 0,
                    end_ms: 400,
                },
                RecognizedWord {
                    text: "there".to_owned(),
                    start_ms: 400,
                    end_ms: 900,
                },
            ],
        };
        desktop.transcribe_with(&heard, &mut tone).unwrap();
        let draft = desktop.draft().unwrap();
        let caption_track = draft
            .tracks
            .iter()
            .find(|track| {
                track.kind == TrackType::Text
                    && track.id.as_str() != "words"
                    && track.segments.len() == 1
            })
            .expect("caption track");
        assert_eq!(
            caption_track.segments[0].target,
            TimeRange::new(0, 2_000_000)
        );
        let material = draft
            .materials
            .texts
            .iter()
            .find(|text| text.content == "hi there")
            .expect("words");
        assert_eq!(material.words.start_time, [0, 400]);
        assert_eq!(material.words.end_time, [400, 900]);
        assert_eq!(material.current_words.text, ["hi", "there"]);
        let pair = draft
            .relationships
            .iter()
            .find(|item| item.kind == "linkage")
            .expect("link")
            .id_to_id
            .iter()
            .find(|pair| pair.owner.as_str() == "picture")
            .expect("pair");
        assert_eq!(pair.dependent, caption_track.segments[0].id);

        let fast_words = CaptionStandIn {
            words: vec![RecognizedWord {
                text: "quick".to_owned(),
                start_ms: 0,
                end_ms: 1_000,
            }],
        };
        desktop.select(Some(SegmentId::new("fast")));
        desktop.transcribe_with(&fast_words, &mut tone).unwrap();
        let draft = desktop.draft().unwrap();
        let reused = draft
            .tracks
            .iter()
            .find(|track| {
                track.kind == TrackType::Text
                    && track
                        .segments
                        .iter()
                        .any(|segment| segment.target == TimeRange::new(3_000_000, 1_000_000))
            })
            .expect("reused");
        assert_eq!(reused.segments.len(), 2);
        assert_eq!(
            draft
                .tracks
                .iter()
                .find(|track| track.id.as_str() == "words")
                .expect("words")
                .segments
                .len(),
            1
        );
        let quick = draft
            .materials
            .texts
            .iter()
            .find(|text| text.content == "quick")
            .expect("quick");
        assert_eq!(quick.words.start_time, [0]);
        assert_eq!(quick.words.end_time, [500]);

        desktop.undo().unwrap();
        assert!(
            desktop
                .draft()
                .unwrap()
                .tracks
                .iter()
                .filter(|track| track.kind == TrackType::Text && track.id.as_str() != "words")
                .all(|track| track.segments.len() == 1)
        );
        desktop.undo().unwrap();
        assert!(
            desktop
                .draft()
                .unwrap()
                .tracks
                .iter()
                .all(|track| track.id.as_str() == "words" || track.kind != TrackType::Text)
        );
        assert!(
            desktop
                .draft()
                .unwrap()
                .materials
                .texts
                .iter()
                .all(|text| text.content == "hi")
        );

        let backward_words = CaptionStandIn {
            words: vec![RecognizedWord {
                text: "back".to_owned(),
                start_ms: 0,
                end_ms: 1_000,
            }],
        };
        desktop.select(Some(SegmentId::new("backward")));
        desktop.transcribe_with(&backward_words, &mut tone).unwrap();
        let back = desktop
            .draft()
            .unwrap()
            .materials
            .texts
            .iter()
            .find(|text| text.content == "back")
            .expect("back")
            .clone();
        assert_eq!(back.words.start_time, [1_000]);
        assert_eq!(back.words.end_time, [2_000]);

        let odd_words = CaptionStandIn {
            words: vec![RecognizedWord {
                text: "odd".to_owned(),
                start_ms: 0,
                end_ms: 400,
            }],
        };
        desktop.select(Some(SegmentId::new("odd")));
        desktop.transcribe_with(&odd_words, &mut tone).unwrap();
        let odd = desktop
            .draft()
            .unwrap()
            .materials
            .texts
            .iter()
            .find(|text| text.content == "odd")
            .expect("odd");
        assert_eq!(odd.words.start_time, [0]);
        assert_eq!(odd.words.end_time, [400]);

        desktop.select(Some(SegmentId::new("tone")));
        assert!(desktop.transcribe_enabled());
        assert!(!desktop.crop_enabled());
        let sound_words = CaptionStandIn {
            words: vec![RecognizedWord {
                text: "sound".to_owned(),
                start_ms: 0,
                end_ms: 200,
            }],
        };
        desktop.transcribe_with(&sound_words, &mut tone).unwrap();
        let draft = desktop.draft().unwrap();
        let sound_text = draft
            .materials
            .texts
            .iter()
            .find(|text| text.content == "sound")
            .expect("sound");
        assert_eq!(sound_text.words.end_time, [200]);
        let sound_segment = draft
            .tracks
            .iter()
            .flat_map(|track| track.segments.iter())
            .find(|segment| segment.material_id == sound_text.id)
            .expect("sound clip");
        assert_eq!(sound_segment.target, TimeRange::new(14_000_000, 1_000_000));
        desktop.undo().unwrap();
        assert!(
            desktop
                .draft()
                .unwrap()
                .materials
                .texts
                .iter()
                .all(|text| text.content != "sound")
        );
    }

    #[test]
    fn rotate_turns_a_picture_clockwise_by_a_quarter() {
        let mut bare = shell();
        assert!(!bare.rotate_enabled());
        assert_eq!(
            bare.turn_clockwise().unwrap_err().to_string(),
            "no project is open"
        );

        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.config.maintrack_adsorb = false;
        let mut video = MaterialVideo::new("picture");
        video.path = "clip".to_owned();
        video.duration = Micros(2_000_000);
        draft.materials.videos.push(video);
        let mut picture = Segment::new("picture", "picture", TimeRange::new(0, 2_000_000));
        picture.clip.rotation = 15.0;
        picture.clip.flip.horizontal = true;
        draft.tracks[0].segments.push(picture);
        draft.tracks[0].segments.push(Segment::new(
            "plain",
            "plain",
            TimeRange::new(3_000_000, 100_000),
        ));
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("tone"),
            path: "tone".to_owned(),
            duration: Micros(2_000_000),
            ..MaterialAudio::default()
        });
        let mut sound = Track::main("sound");
        sound.kind = TrackType::Audio;
        sound
            .segments
            .push(Segment::new("tone", "tone", TimeRange::new(0, 2_000_000)));
        draft.tracks.push(sound);
        draft.materials.texts.push(MaterialText {
            id: MaterialId::new("caption"),
            content: "hi".to_owned(),
            ..MaterialText::default()
        });
        let mut words = Track::main("words");
        words.kind = TrackType::Text;
        words.segments.push(Segment::new(
            "caption",
            "caption",
            TimeRange::new(0, 1_000_000),
        ));
        draft.tracks.push(words);
        draft.materials.stickers.push(RawMaterial {
            id: MaterialId::new("badge"),
            kind: "sticker".to_owned(),
            ..RawMaterial::default()
        });
        let mut stickers = Track::main("stickers");
        stickers.kind = TrackType::Sticker;
        stickers
            .segments
            .push(Segment::new("badge", "badge", TimeRange::new(0, 1_000_000)));
        draft.tracks.push(stickers);
        draft.materials.drafts.push(MaterialDraft {
            id: MaterialId::new("compound"),
            ..MaterialDraft::default()
        });
        let mut compound = Track::main("compound");
        compound.kind = TrackType::Composition;
        compound.segments.push(Segment::new(
            "compound",
            "compound",
            TimeRange::new(0, 1_000_000),
        ));
        draft.tracks.push(compound);
        Folder::new(draft).save(&temp.path).unwrap();

        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.select(Some(SegmentId::new("plain")));
        assert!(!desktop.rotate_enabled());
        assert_eq!(
            desktop.turn_clockwise().unwrap_err().to_string(),
            "a rotate belongs on a picture"
        );
        desktop.select(Some(SegmentId::new("tone")));
        assert!(!desktop.rotate_enabled());
        assert!(desktop.reverse_enabled());
        assert_eq!(
            desktop.turn_clockwise().unwrap_err().to_string(),
            "a rotate belongs on a picture"
        );

        desktop.select(Some(SegmentId::new("picture")));
        assert!(desktop.rotate_enabled());
        assert!(desktop.rotated());
        desktop.turn_clockwise().unwrap();
        assert_eq!(picture_rotation(&desktop, "picture"), 285.0);
        assert!(
            desktop.draft().unwrap().tracks[0].segments[0]
                .clip
                .flip
                .horizontal
        );
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0].target,
            TimeRange::new(0, 2_000_000)
        );
        desktop.undo().unwrap();
        assert_eq!(picture_rotation(&desktop, "picture"), 15.0);
        desktop.redo().unwrap();
        desktop.turn_clockwise().unwrap();
        desktop.turn_clockwise().unwrap();
        desktop.turn_clockwise().unwrap();
        assert_eq!(picture_rotation(&desktop, "picture"), 15.0);

        let track = desktop.draft().unwrap().tracks[0].id.clone();
        desktop
            .toggle_attribute(track, TrackAttribute::LOCK)
            .unwrap();
        let locked = desktop.turn_clockwise().unwrap_err();
        assert_eq!(locked.to_string(), "the track is locked");
        assert_eq!(picture_rotation(&desktop, "picture"), 15.0);

        desktop.select(Some(SegmentId::new("caption")));
        assert!(desktop.rotate_enabled());
        assert!(!desktop.freeze_enabled());
        assert!(!desktop.reverse_enabled());
        assert!(!desktop.rotated());
        desktop.turn_clockwise().unwrap();
        assert_eq!(picture_rotation(&desktop, "caption"), 270.0);
        assert!(desktop.rotated());
        desktop.turn_clockwise().unwrap();
        desktop.turn_clockwise().unwrap();
        desktop.turn_clockwise().unwrap();
        assert_eq!(picture_rotation(&desktop, "caption"), 0.0);
        assert!(!desktop.rotated());
        desktop.select(Some(SegmentId::new("badge")));
        assert!(desktop.rotate_enabled());
        desktop.select(Some(SegmentId::new("compound")));
        assert!(desktop.rotate_enabled());
        assert!(!desktop.rotated());
    }

    fn picture_rotation(desktop: &Shell, id: &str) -> f64 {
        desktop
            .draft()
            .unwrap()
            .tracks
            .iter()
            .flat_map(|track| track.segments.iter())
            .find(|segment| segment.id.as_str() == id)
            .expect("clip")
            .clip
            .rotation
    }

    #[test]
    fn mirror_flips_a_picture_left_to_right() {
        let mut bare = shell();
        assert!(!bare.mirror_enabled());
        assert_eq!(
            bare.toggle_mirror().unwrap_err().to_string(),
            "no project is open"
        );

        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.config.maintrack_adsorb = false;
        let mut video = MaterialVideo::new("picture");
        video.path = "clip".to_owned();
        video.duration = Micros(2_000_000);
        draft.materials.videos.push(video);
        let mut picture = Segment::new("picture", "picture", TimeRange::new(0, 2_000_000));
        picture.clip.flip.vertical = true;
        draft.tracks[0].segments.push(picture);
        draft.tracks[0].segments.push(Segment::new(
            "plain",
            "plain",
            TimeRange::new(3_000_000, 100_000),
        ));
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("tone"),
            path: "tone".to_owned(),
            duration: Micros(2_000_000),
            ..MaterialAudio::default()
        });
        let mut sound = Track::main("sound");
        sound.kind = TrackType::Audio;
        sound
            .segments
            .push(Segment::new("tone", "tone", TimeRange::new(0, 2_000_000)));
        draft.tracks.push(sound);
        draft.materials.texts.push(MaterialText {
            id: MaterialId::new("caption"),
            content: "hi".to_owned(),
            ..MaterialText::default()
        });
        let mut words = Track::main("words");
        words.kind = TrackType::Text;
        words.segments.push(Segment::new(
            "caption",
            "caption",
            TimeRange::new(0, 1_000_000),
        ));
        draft.tracks.push(words);
        draft.materials.stickers.push(RawMaterial {
            id: MaterialId::new("badge"),
            kind: "sticker".to_owned(),
            ..RawMaterial::default()
        });
        let mut stickers = Track::main("stickers");
        stickers.kind = TrackType::Sticker;
        stickers
            .segments
            .push(Segment::new("badge", "badge", TimeRange::new(0, 1_000_000)));
        draft.tracks.push(stickers);
        draft.materials.drafts.push(MaterialDraft {
            id: MaterialId::new("compound"),
            ..MaterialDraft::default()
        });
        let mut compound = Track::main("compound");
        compound.kind = TrackType::Composition;
        compound.segments.push(Segment::new(
            "compound",
            "compound",
            TimeRange::new(0, 1_000_000),
        ));
        draft.tracks.push(compound);
        Folder::new(draft).save(&temp.path).unwrap();

        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.select(Some(SegmentId::new("plain")));
        assert!(!desktop.mirror_enabled());
        assert_eq!(
            desktop.toggle_mirror().unwrap_err().to_string(),
            "a mirror belongs on a picture"
        );
        desktop.select(Some(SegmentId::new("tone")));
        assert!(!desktop.mirror_enabled());
        assert!(desktop.reverse_enabled());
        assert_eq!(
            desktop.toggle_mirror().unwrap_err().to_string(),
            "a mirror belongs on a picture"
        );

        desktop.select(Some(SegmentId::new("picture")));
        assert!(desktop.mirror_enabled());
        assert!(!desktop.mirrored());
        desktop.toggle_mirror().unwrap();
        assert!(desktop.mirrored());
        let clip = &desktop.draft().unwrap().tracks[0].segments[0].clip;
        assert!(clip.flip.horizontal);
        assert!(clip.flip.vertical);
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0].target,
            TimeRange::new(0, 2_000_000)
        );
        desktop.undo().unwrap();
        assert!(!desktop.mirrored());
        desktop.redo().unwrap();
        desktop.toggle_mirror().unwrap();
        assert!(!desktop.mirrored());

        let track = desktop.draft().unwrap().tracks[0].id.clone();
        desktop
            .toggle_attribute(track, TrackAttribute::LOCK)
            .unwrap();
        let locked = desktop.toggle_mirror().unwrap_err();
        assert_eq!(locked.to_string(), "the track is locked");
        assert!(!desktop.mirrored());

        desktop.select(Some(SegmentId::new("caption")));
        assert!(desktop.mirror_enabled());
        assert!(!desktop.freeze_enabled());
        assert!(!desktop.reverse_enabled());
        desktop.toggle_mirror().unwrap();
        assert!(desktop.mirrored());
        desktop.select(Some(SegmentId::new("badge")));
        assert!(desktop.mirror_enabled());
        desktop.select(Some(SegmentId::new("compound")));
        assert!(desktop.mirror_enabled());
        assert!(!desktop.mirrored());
    }

    #[test]
    fn a_vertical_flip_turns_the_picture_over() {
        let mut bare = shell();
        assert!(!bare.flipped());
        assert_eq!(
            bare.toggle_vertical_flip().unwrap_err().to_string(),
            "no project is open"
        );

        let temp = Temp::new();
        let picture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/engine/fixtures/pattern.png");
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 16;
        draft.canvas_config.height = 16;
        draft.config.maintrack_adsorb = false;
        let mut video = MaterialVideo::new("still");
        video.path = picture.display().to_string();
        video.duration = Micros(1_000_000);
        draft.materials.videos.push(video);
        let mut segment = Segment::new("still", "still", TimeRange::new(0, 1_000_000));
        segment.source_range = Some(TimeRange::new(0, 1_000_000));
        draft.tracks[0].segments.push(segment);
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("tone"),
            path: "tone".to_owned(),
            duration: Micros(1_000_000),
            ..MaterialAudio::default()
        });
        let mut sound = Track::main("sound");
        sound.kind = TrackType::Audio;
        sound
            .segments
            .push(Segment::new("tone", "tone", TimeRange::new(0, 1_000_000)));
        draft.tracks.push(sound);
        draft.materials.texts.push(MaterialText {
            id: MaterialId::new("caption"),
            content: "A".to_owned(),
            ..MaterialText::default()
        });
        let mut words = Track::main("words");
        words.kind = TrackType::Text;
        words.segments.push(Segment::new(
            "caption",
            "caption",
            TimeRange::new(0, 1_000_000),
        ));
        draft.tracks.push(words);
        Folder::new(draft).save(&temp.path).unwrap();

        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.select(Some(SegmentId::new("tone")));
        assert!(!desktop.flipped());
        assert_eq!(
            desktop.toggle_vertical_flip().unwrap_err().to_string(),
            "a flip belongs on a picture"
        );

        desktop.select(Some(SegmentId::new("still")));
        assert!(desktop.mirror_enabled());
        assert!(!desktop.flipped());
        let before = desktop.preview_bytes().unwrap();
        assert_eq!((before.width, before.height), (16, 16));
        let row = (before.width as usize) * 4;
        let last = before.rgba.len() - row;
        assert_ne!(&before.rgba[..row], &before.rgba[last..]);
        desktop.toggle_vertical_flip().unwrap();
        assert!(desktop.flipped());
        assert!(!desktop.mirrored());
        let clip = &desktop.draft().unwrap().tracks[0].segments[0].clip;
        assert!(clip.flip.vertical);
        assert!(!clip.flip.horizontal);
        assert_eq!(clip.transform.x, 0.0);
        assert_eq!(clip.scale.x, 1.0);
        let turned = desktop.preview_bytes().unwrap();
        assert_eq!(&turned.rgba[..row], &before.rgba[last..]);
        assert_eq!(&turned.rgba[last..], &before.rgba[..row]);
        desktop.toggle_vertical_flip().unwrap();
        assert!(!desktop.flipped());
        let restored = desktop.preview_bytes().unwrap();
        assert_eq!(restored.rgba, before.rgba);
        desktop.undo().unwrap();
        assert!(desktop.flipped());
        desktop.undo().unwrap();
        assert!(!desktop.flipped());

        desktop.select(Some(SegmentId::new("caption")));
        desktop.toggle_vertical_flip().unwrap();
        assert!(desktop.flipped());
        assert!(!desktop.mirrored());

        desktop.select(Some(SegmentId::new("still")));
        let lane = desktop.draft().unwrap().tracks[0].id.clone();
        desktop
            .toggle_attribute(lane, TrackAttribute::LOCK)
            .unwrap();
        assert_eq!(
            desktop.toggle_vertical_flip().unwrap_err().to_string(),
            "the track is locked"
        );
        assert!(!desktop.flipped());
    }

    #[test]
    fn reverse_plays_a_picture_or_a_sound_backward() {
        let mut bare = shell();
        assert!(!bare.reverse_enabled());
        assert_eq!(
            bare.toggle_reverse().unwrap_err().to_string(),
            "no project is open"
        );

        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.config.maintrack_adsorb = false;
        let mut video = MaterialVideo::new("picture");
        video.path = "clip".to_owned();
        video.duration = Micros(2_000_000);
        draft.materials.videos.push(video);
        let mut picture = Segment::new("picture", "picture", TimeRange::new(0, 2_000_000));
        picture.source_range = Some(TimeRange::new(0, 2_000_000));
        draft.tracks[0].segments.push(picture);
        draft.tracks[0].segments.push(Segment::new(
            "plain",
            "plain",
            TimeRange::new(3_000_000, 100_000),
        ));
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("tone"),
            path: "tone".to_owned(),
            duration: Micros(2_000_000),
            ..MaterialAudio::default()
        });
        let mut sound = Track::main("sound");
        sound.kind = TrackType::Audio;
        let mut tone = Segment::new("tone", "tone", TimeRange::new(0, 2_000_000));
        tone.source_range = Some(TimeRange::new(0, 2_000_000));
        sound.segments.push(tone);
        draft.tracks.push(sound);
        Folder::new(draft).save(&temp.path).unwrap();

        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.select(Some(SegmentId::new("plain")));
        assert!(!desktop.reverse_enabled());
        assert_eq!(
            desktop.toggle_reverse().unwrap_err().to_string(),
            "reverse belongs on a picture or a sound"
        );

        desktop.select(Some(SegmentId::new("picture")));
        assert!(desktop.reverse_enabled());
        assert!(!desktop.reversed());
        desktop.toggle_reverse().unwrap();
        assert!(desktop.reversed());
        assert_eq!(
            desktop.draft().unwrap().materials.videos[0].reverse_path,
            ""
        );
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0].target,
            TimeRange::new(0, 2_000_000)
        );
        assert_eq!(source_at_start(&desktop, "picture"), 1_999_999);
        desktop.undo().unwrap();
        assert!(!desktop.reversed());
        assert_eq!(source_at_start(&desktop, "picture"), 0);
        desktop.redo().unwrap();
        desktop.toggle_reverse().unwrap();
        assert!(!desktop.reversed());

        let track = desktop.draft().unwrap().tracks[0].id.clone();
        desktop
            .toggle_attribute(track, TrackAttribute::LOCK)
            .unwrap();
        let locked = desktop.toggle_reverse().unwrap_err();
        assert_eq!(locked.to_string(), "the track is locked");
        assert!(!desktop.reversed());

        desktop.select(Some(SegmentId::new("tone")));
        assert!(desktop.reverse_enabled());
        assert!(!desktop.freeze_enabled());
        desktop.toggle_reverse().unwrap();
        assert!(desktop.reversed());
        assert_eq!(desktop.draft().unwrap().materials.videos.len(), 1);
        assert_eq!(source_at_start(&desktop, "tone"), 1_999_999);
        desktop.toggle_reverse().unwrap();
        assert_eq!(source_at_start(&desktop, "tone"), 0);
    }

    fn source_at_start(desktop: &Shell, id: &str) -> i64 {
        let draft = desktop.draft().unwrap();
        let segment = draft
            .tracks
            .iter()
            .flat_map(|track| track.segments.iter())
            .find(|segment| segment.id.as_str() == id)
            .expect("clip");
        let map = cutlass_engine::placement_of(draft, segment).expect("map");
        cutlass_engine::map_timeline(&map, segment.target.start)
            .expect("source")
            .0
    }

    #[test]
    fn a_freeze_holds_the_frame_under_the_playhead() {
        let mut bare = shell();
        assert!(!bare.freeze_enabled());
        assert_eq!(
            bare.toggle_freeze().unwrap_err().to_string(),
            "no project is open"
        );

        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        draft.config.maintrack_adsorb = false;
        let mut video = MaterialVideo::new("picture");
        video.path = "clip".to_owned();
        video.duration = Micros(10_000_000);
        draft.materials.videos.push(video);
        let mut picture = Segment::new("picture", "picture", TimeRange::new(0, 2_000_000));
        picture.source_range = Some(TimeRange::new(0, 2_000_000));
        draft.tracks[0].segments.push(picture);
        let mut sound = Track::main("sound");
        sound.kind = TrackType::Audio;
        sound
            .segments
            .push(Segment::new("tone", "tone", TimeRange::new(0, 2_000_000)));
        draft.tracks.push(sound);
        Folder::new(draft).save(&temp.path).unwrap();

        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        assert!(!desktop.freeze_enabled());
        desktop.select(Some(SegmentId::new("tone")));
        assert!(!desktop.freeze_enabled());
        assert_eq!(
            desktop.toggle_freeze().unwrap_err().to_string(),
            "a freeze belongs on video"
        );

        desktop.select(Some(SegmentId::new("picture")));
        assert!(desktop.freeze_enabled());
        assert!(!desktop.frozen());
        desktop.seek(Micros(500_000));
        desktop.toggle_freeze().unwrap();
        let held = desktop.draft().unwrap().materials.videos[0]
            .freeze
            .clone()
            .expect("held frame");
        assert_eq!(held.timestamp, Micros(500_000));
        assert_eq!(held.source_material_id, "picture");
        assert_eq!(held.source_material_path, "clip");
        assert!(desktop.frozen());
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0].target,
            TimeRange::new(0, 2_000_000)
        );
        desktop.undo().unwrap();
        assert!(
            desktop.draft().unwrap().materials.videos[0]
                .freeze
                .is_none()
        );
        desktop.redo().unwrap();
        assert!(desktop.frozen());
        desktop.toggle_freeze().unwrap();
        assert!(!desktop.frozen());

        desktop.seek(Micros(5_000_000));
        desktop.toggle_freeze().unwrap();
        assert_eq!(
            desktop.draft().unwrap().materials.videos[0]
                .freeze
                .as_ref()
                .expect("first frame")
                .timestamp,
            Micros(0)
        );
        desktop.toggle_freeze().unwrap();
        desktop.set_speed(2.0).unwrap();
        desktop.seek(Micros(400_000));
        desktop.toggle_freeze().unwrap();
        assert_eq!(
            desktop.draft().unwrap().materials.videos[0]
                .freeze
                .as_ref()
                .expect("sped frame")
                .timestamp,
            Micros(800_000)
        );

        let track = desktop.draft().unwrap().tracks[0].id.clone();
        desktop
            .toggle_attribute(track, TrackAttribute::LOCK)
            .unwrap();
        let locked = desktop.toggle_freeze().unwrap_err();
        assert_eq!(locked.to_string(), "the track is locked");
        assert!(desktop.frozen());
    }

    #[test]
    fn snapping_starts_on_and_a_drag_keeps_the_pointer_when_it_is_off() {
        let mut bare = shell();
        assert!(bare.snap_enabled());
        bare.toggle_snap();
        assert!(!bare.snap_enabled());

        let temp = Temp::new();
        trim_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        assert!(desktop.snap_enabled());
        let id = desktop.draft().unwrap().tracks[0].segments[0].id.clone();
        let y = crate::tokens::LANE_HEIGHT / 2.0;
        desktop.seek(Micros(1_500_000));
        desktop.preview_move(id.clone(), lane_time(&desktop, 1_501_000.0), y);
        assert_eq!(
            desktop.move_preview().expect("the drag is showing").start,
            Micros(1_500_000)
        );
        let revision = desktop.revision();
        desktop.toggle_snap();
        assert!(!desktop.snap_enabled());
        assert_eq!(desktop.revision(), revision);
        desktop.preview_move(id.clone(), lane_time(&desktop, 1_501_000.0), y);
        assert_eq!(
            desktop.move_preview().expect("the drag is showing").start,
            Micros(1_501_000)
        );
        assert!(desktop.snap_line().is_none());
        desktop
            .drag_segment(id.clone(), lane_time(&desktop, 1_501_000.0), y)
            .unwrap();
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0].target.start,
            Micros(1_501_000)
        );
        desktop.undo().unwrap();
        assert!(!desktop.snap_enabled());
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0].target.start,
            Micros(0)
        );

        desktop.preview_edge(id.clone(), TrimEdge::End, lane_time(&desktop, 1_501_000.0));
        assert_eq!(
            desktop.preview_span(&id).expect("the edge is showing").1,
            Micros(1_501_000)
        );
        desktop.toggle_snap();
        desktop.seek(Micros(1_500_000));
        desktop.preview_edge(id.clone(), TrimEdge::End, lane_time(&desktop, 1_501_000.0));
        assert_eq!(
            desktop.preview_span(&id).expect("the edge is showing").1,
            Micros(1_500_000)
        );
    }

    #[test]
    fn dragging_a_clip_follows_the_pointer_and_snaps_without_writing() {
        let temp = Temp::new();
        trim_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let id = desktop.draft().unwrap().tracks[0].segments[0].id.clone();
        let y = crate::tokens::LANE_HEIGHT / 2.0;
        desktop.seek(Micros(2_000_000));
        let revision = desktop.revision();
        desktop.preview_move(id.clone(), lane_time(&desktop, 2_001_000.0), y);
        let preview = desktop.move_preview().expect("the drag is showing");
        assert_eq!(preview.start, Micros(2_000_000));
        assert_eq!(preview.duration, Micros(1_000_000));
        assert!(desktop.moving(&id));
        assert_eq!(desktop.snap_line(), Some(Micros(2_000_000)));
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0].target.start,
            Micros(0)
        );
        assert_eq!(desktop.revision(), revision);
        desktop
            .drag_segment(id.clone(), lane_time(&desktop, 2_001_000.0), y)
            .unwrap();
        assert!(desktop.move_preview().is_none());
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0].target.start,
            Micros(2_000_000)
        );
        let held = desktop.revision();
        desktop
            .drag_segment(id.clone(), lane_time(&desktop, 2_000_000.0), y)
            .unwrap();
        assert_eq!(desktop.revision(), held);
        desktop.undo().unwrap();
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0].target.start,
            Micros(0)
        );
        desktop.seek(Micros(0));
        desktop.preview_move(id.clone(), lane_time(&desktop, 500_000.0), y);
        assert_eq!(
            desktop.move_preview().expect("the drag is showing").start,
            Micros(500_000)
        );
        assert!(desktop.snap_line().is_none());
        desktop
            .drag_segment(id.clone(), lane_time(&desktop, 500_000.0), y)
            .unwrap();
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0].target.start,
            Micros(500_000)
        );
        let track = desktop.lanes()[0].track_id.clone();
        desktop
            .toggle_attribute(track, TrackAttribute::LOCK)
            .unwrap();
        desktop.preview_move(id.clone(), lane_time(&desktop, 800_000.0), y);
        assert!(desktop.move_preview().is_none());
        let error = desktop
            .drag_segment(id, lane_time(&desktop, 800_000.0), y)
            .unwrap_err();
        assert_eq!(error.to_string(), "the track is locked");
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0].target.start,
            Micros(500_000)
        );
    }

    #[test]
    fn splitting_keeps_a_frame_on_each_piece_and_the_key_uses_the_playhead() {
        let mut desktop = shell();
        desktop.toggle_blade();
        assert!(!desktop.blade());
        let temp = Temp::new();
        trim_project(&temp.path);
        desktop.open_folder(&temp.path).unwrap();
        let id = desktop.draft().unwrap().tracks[0].segments[0].id.clone();
        let margin = desktop
            .draft()
            .unwrap()
            .frame_rate()
            .unwrap()
            .frame_start(1)
            .unwrap();
        desktop.toggle_blade();
        assert!(desktop.blade());
        desktop.clear_blade();
        desktop.select(Some(id.clone()));
        desktop.seek(Micros(margin.0 - 1));
        let revision = desktop.revision();
        let error = desktop.split_selected().unwrap_err();
        assert_eq!(error.to_string(), "split is not available");
        assert_eq!(desktop.revision(), revision);
        desktop.seek(Micros(500_000));
        desktop.split_selected().unwrap();
        {
            let segments = &desktop.draft().unwrap().tracks[0].segments;
            assert_eq!(segments.len(), 3);
            assert_eq!(segments[0].target, TimeRange::new(0, 500_000));
            assert_eq!(segments[0].source_range, Some(TimeRange::new(0, 500_000)));
            assert_eq!(segments[1].target, TimeRange::new(500_000, 500_000));
            assert_eq!(
                segments[1].source_range,
                Some(TimeRange::new(500_000, 500_000))
            );
        }
        assert_eq!(desktop.selected(), Some(&id));
        desktop.undo().unwrap();
        assert_eq!(desktop.draft().unwrap().tracks[0].segments.len(), 2);
        desktop.select(None);
        desktop.seek(Micros(2_000_000));
        let error = desktop.run_shortcut("s").unwrap_err();
        assert_eq!(error.to_string(), "split is not available");
        desktop.seek(Micros(500_000));
        desktop.run_shortcut("s").unwrap();
        assert_eq!(desktop.draft().unwrap().tracks[0].segments.len(), 3);
        desktop.undo().unwrap();
        let y = crate::tokens::LANE_HEIGHT / 2.0;
        desktop
            .split_pointer(id.clone(), lane_time(&desktop, 500_000.0), y)
            .unwrap();
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0]
                .target
                .duration,
            Micros(500_000)
        );
        desktop.undo().unwrap();
        assert_eq!(
            ShellError::Edit(EditError::TransitionWindow).to_string(),
            "a transition cannot be split"
        );
        let track = desktop.lanes()[0].track_id.clone();
        desktop
            .toggle_attribute(track, TrackAttribute::LOCK)
            .unwrap();
        desktop.seek(Micros(500_000));
        desktop.select(Some(id));
        let error = desktop.split_playhead().unwrap_err();
        assert_eq!(error.to_string(), "the track is locked");
        assert_eq!(desktop.draft().unwrap().tracks[0].segments.len(), 2);
    }

    #[test]
    fn a_drag_at_the_lane_edge_scrolls_and_stops_at_the_cut() {
        let temp = Temp::new();
        trim_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.set_lane_width(800.0);
        assert!(!desktop.drag_scroll(400.0));
        assert_eq!(desktop.view_origin(), Micros(0));
        assert!(desktop.drag_scroll(790.0));
        let scrolled = desktop.view_origin().0;
        assert!(scrolled > 0);
        assert!(desktop.drag_scroll(0.0));
        assert!(desktop.view_origin().0 < scrolled);
        for _ in 0..8 {
            if desktop.view_origin().0 == 0 {
                break;
            }
            let _ = desktop.drag_scroll(0.0);
        }
        assert_eq!(desktop.view_origin(), Micros(0));
        assert!(!desktop.drag_scroll(0.0));
        let mut reached_end = false;
        for _ in 0..2_000 {
            if !desktop.drag_scroll(800.0) {
                reached_end = true;
                break;
            }
        }
        assert!(reached_end);
        assert!(!desktop.drag_scroll(800.0));
        assert!(desktop.view_origin().0 > 0);
        let at_end = desktop.view_origin();
        assert!(desktop.pan_pixels(-4_000.0));
        assert!(desktop.view_origin().0 < at_end.0);
    }

    #[test]
    fn a_box_selection_deletes_every_clip_in_one_step() {
        let temp = Temp::new();
        trim_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let ids: Vec<_> = desktop.draft().unwrap().tracks[0]
            .segments
            .iter()
            .map(|segment| segment.id.clone())
            .collect();
        desktop.select_box(ids.clone(), false);
        assert_eq!(desktop.selection(), ids.as_slice());
        assert!(desktop.is_selected(&ids[0]));
        desktop.toggle_selected(ids[0].clone());
        assert!(!desktop.is_selected(&ids[0]));
        assert_eq!(desktop.selected(), Some(&ids[1]));
        desktop.toggle_selected(ids[0].clone());
        desktop.select(Some(ids[0].clone()));
        assert_eq!(desktop.selection(), std::slice::from_ref(&ids[0]));
        desktop.select_box(ids.iter().cloned(), true);
        assert_eq!(desktop.selection().len(), 2);
        let track = desktop.lanes()[0].track_id.clone();
        desktop
            .toggle_attribute(track.clone(), TrackAttribute::LOCK)
            .unwrap();
        assert_eq!(
            desktop.remove_selected().unwrap_err().to_string(),
            "the track is locked"
        );
        assert_eq!(desktop.draft().unwrap().tracks[0].segments.len(), 2);
        desktop
            .toggle_attribute(track, TrackAttribute::LOCK)
            .unwrap();
        desktop.remove_selected().unwrap();
        assert!(desktop.selection().is_empty());
        assert!(desktop.draft().unwrap().tracks[0].segments.is_empty());
        desktop.undo().unwrap();
        assert_eq!(desktop.draft().unwrap().tracks[0].segments.len(), 2);
    }

    #[test]
    fn a_drag_moves_every_selected_clip_and_keeps_the_gap() {
        let temp = Temp::new();
        trim_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let ids: Vec<_> = desktop.draft().unwrap().tracks[0]
            .segments
            .iter()
            .map(|segment| segment.id.clone())
            .collect();
        desktop.select_box(ids.clone(), false);
        let y = crate::tokens::LANE_HEIGHT / 2.0;
        desktop.preview_move(ids[1].clone(), lane_time(&desktop, 0.0), y);
        let held = desktop.move_previews();
        assert_eq!(held.len(), 2);
        assert_eq!(held[0].id, ids[1]);
        assert_eq!(
            held.iter()
                .find(|preview| preview.id == ids[0])
                .unwrap()
                .start,
            Micros(0)
        );
        assert_eq!(
            held.iter()
                .find(|preview| preview.id == ids[1])
                .unwrap()
                .start,
            Micros(3_000_000)
        );
        desktop.seek(Micros(2_000_000));
        desktop.preview_move(ids[0].clone(), lane_time(&desktop, 2_001_000.0), y);
        let moving = desktop.move_previews();
        assert_eq!(
            moving
                .iter()
                .find(|preview| preview.id == ids[0])
                .unwrap()
                .start,
            Micros(2_000_000)
        );
        assert_eq!(
            moving
                .iter()
                .find(|preview| preview.id == ids[1])
                .unwrap()
                .start,
            Micros(5_000_000)
        );
        assert!(desktop.moving(&ids[0]) && desktop.moving(&ids[1]));
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0].target.start,
            Micros(0)
        );
        desktop
            .drag_segment(ids[0].clone(), lane_time(&desktop, 2_001_000.0), y)
            .unwrap();
        let starts: Vec<_> = desktop.draft().unwrap().tracks[0]
            .segments
            .iter()
            .map(|segment| segment.target.start)
            .collect();
        assert_eq!(starts, [Micros(2_000_000), Micros(5_000_000)]);
        desktop.undo().unwrap();
        let starts: Vec<_> = desktop.draft().unwrap().tracks[0]
            .segments
            .iter()
            .map(|segment| segment.target.start)
            .collect();
        assert_eq!(starts, [Micros(0), Micros(3_000_000)]);
        let track = desktop.lanes()[0].track_id.clone();
        desktop
            .toggle_attribute(track, TrackAttribute::LOCK)
            .unwrap();
        let error = desktop
            .drag_segment(ids[0].clone(), lane_time(&desktop, 500_000.0), y)
            .unwrap_err();
        assert_eq!(error.to_string(), "the track is locked");
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[1].target.start,
            Micros(3_000_000)
        );
    }

    #[test]
    fn paste_lands_at_the_playhead_and_keeps_the_gap() {
        let temp = Temp::new();
        trim_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        assert_eq!(
            desktop.copy_selected().unwrap_err().to_string(),
            "nothing is selected"
        );
        let ids: Vec<_> = desktop.draft().unwrap().tracks[0]
            .segments
            .iter()
            .map(|segment| segment.id.clone())
            .collect();
        desktop.select_box(ids, false);
        desktop.copy_selected().unwrap();
        assert!(desktop.has_clipboard());
        assert_eq!(
            desktop.paste().unwrap_err().to_string(),
            "the clips overlap"
        );
        assert_eq!(desktop.draft().unwrap().tracks[0].segments.len(), 2);
        desktop.seek(Micros(4_000_000));
        desktop.paste().unwrap();
        let draft = desktop.draft().unwrap();
        assert_eq!(draft.tracks[0].segments.len(), 4);
        assert_eq!(
            draft.tracks[0].segments[2].target,
            TimeRange::new(4_000_000, 1_000_000)
        );
        assert_eq!(
            draft.tracks[0].segments[3].target,
            TimeRange::new(7_000_000, 500_000)
        );
        assert_eq!(desktop.selection().len(), 2);
        desktop.undo().unwrap();
        assert_eq!(desktop.draft().unwrap().tracks[0].segments.len(), 2);
        let ids: Vec<_> = desktop.draft().unwrap().tracks[0]
            .segments
            .iter()
            .map(|segment| segment.id.clone())
            .collect();
        desktop.select_box(ids, false);
        desktop.run_shortcut("primary+x").unwrap();
        assert!(desktop.draft().unwrap().tracks[0].segments.is_empty());
        desktop.seek(Micros(0));
        desktop.run_shortcut("primary+v").unwrap();
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0].target.start,
            Micros(0)
        );
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[1].target.start,
            Micros(3_000_000)
        );
        desktop.undo().unwrap();
        desktop.undo().unwrap();
        assert_eq!(desktop.draft().unwrap().tracks[0].segments.len(), 2);
        let ids: Vec<_> = desktop.draft().unwrap().tracks[0]
            .segments
            .iter()
            .map(|segment| segment.id.clone())
            .collect();
        desktop.select_box(ids, false);
        let track = desktop.lanes()[0].track_id.clone();
        desktop
            .toggle_attribute(track, TrackAttribute::LOCK)
            .unwrap();
        assert_eq!(
            desktop.copy_selected().unwrap_err().to_string(),
            "the track is locked"
        );
        assert!(desktop.has_clipboard());
        assert_eq!(desktop.draft().unwrap().tracks[0].segments.len(), 2);
    }

    #[test]
    fn a_pinch_keeps_the_time_under_the_pointer() {
        let temp = Temp::new();
        trim_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        desktop.set_lane_width(800.0);
        let pixel = desktop.pixel_size();
        assert!(desktop.pan_pixels(400.0));
        let x = 200.0;
        let time = desktop.view_origin().0 as f64 + x / pixel;
        desktop.zoom_at(0.5, x).unwrap();
        assert!((desktop.pixel_size() - pixel * 0.5).abs() < 1.0e-9);
        let after = desktop.view_origin().0 as f64 + x / desktop.pixel_size();
        assert!(
            (after - time).abs() <= 1.0,
            "the time moved from {time} to {after}"
        );
        desktop.zoom_at(2.0, x).unwrap();
        assert!((desktop.pixel_size() - pixel).abs() < 1.0e-9);
        let back = desktop.view_origin().0 as f64 + x / desktop.pixel_size();
        assert!((back - time).abs() <= 1.0);
        let size = desktop.pixel_size();
        let origin = desktop.view_origin();
        desktop.zoom_at(2.0, x).unwrap();
        assert_eq!(desktop.pixel_size(), size);
        assert_eq!(desktop.view_origin(), origin);
        desktop.zoom_at(1.0, x).unwrap();
        assert_eq!(desktop.pixel_size(), size);
    }

    #[test]
    fn the_launch_list_is_the_saved_projects() {
        let root = Temp::new();
        let missing = root.path.join("missing");
        assert!(Shell::list_projects(&missing).is_empty());
        assert!(!missing.exists());
        std::fs::create_dir(root.path.join("notes")).unwrap();
        let first = Shell::create_project(&root.path).unwrap();
        let second = Shell::create_project(&root.path).unwrap();
        let mut names: Vec<_> = Shell::list_projects(&root.path)
            .into_iter()
            .map(|project| project.name)
            .collect();
        names.sort();
        assert_eq!(names, ["Project", "Project 2"]);
        assert!(first.ends_with("Project"));
        assert!(second.ends_with("Project 2"));
        assert!(Folder::read(&first).unwrap().draft.name == "Project");
    }

    #[test]
    fn leaving_a_project_writes_the_folder() {
        let root = Temp::new();
        let path = Shell::create_project(&root.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&path).unwrap();
        let track = desktop.draft().unwrap().tracks[0].id.clone();
        desktop
            .toggle_attribute(track, TrackAttribute::LOCK)
            .unwrap();
        desktop.close_project().unwrap();
        assert!(desktop.draft().is_none());
        let saved = Folder::read(&path).unwrap();
        assert!(
            saved.draft.tracks[0]
                .attribute
                .contains(TrackAttribute::LOCK)
        );
        assert_eq!(Shell::list_projects(&root.path).len(), 1);
    }

    #[test]
    fn naming_the_projects_directory_does_not_create_it() {
        let path = Shell::projects_directory();
        assert_eq!(path, Shell::home().projects());
        assert!(path.starts_with(Shell::home().root()));
    }
}
