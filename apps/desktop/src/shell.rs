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
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use cutlass_decode::{CodecId, CodecParams, DecodeFactory};
use cutlass_editor::{
    DropGesture, DropOutcome, FilmTile, GestureError, GestureOutcome, MoveGesture, Pointer,
    TimelineScale, TrimGesture, TrimPlan, Zoom, ZoomBounds, drop_gesture, film_tiles,
    fit_pixel_size, hit_test, lane_category, lanes, last_accuracy_step, minimum_pixel_size,
    move_gesture, snap_guides, trim_gesture,
};
use cutlass_encode::EncodeFactory;
use cutlass_engine::{Thumb, ThumbKey, Thumbnails, Waveforms};
use cutlass_eval::{Heard, ReaderFactory, SceneError, scene_cuts, source_on_timeline};
use cutlass_model::{
    Crop, Draft, FrameRate, FreezeInfo, MaterialAudio, MaterialId, MaterialRef, MaterialTransition,
    MaterialVideo, MetaType, Micros, Segment, SegmentId, TimeMarkItem, TimeRange, TimecodeLabel,
    TrackAttribute, TrackId, TrackType, WordColumns, beat_timeline_time, blend_window,
    linkage_from_biz, segment_beats, segment_time_mark, source_time_at, time_mark_items,
    track_holds_beats, user_beats, visible_beats, visible_marks,
};
use cutlass_playback::{
    AudioHost, AudioOutput, FrameCache, MixJob, Mixer, PLAY_FRAMES, PREVIEW_RATE, PreviewMix,
    preview_frames, queue_mix,
};
use cutlass_renderer_software::SoftwareDevice;
use cutlass_session::{EditError, ImportedMedium, Request, Session, TrimEdge};
use cutlass_settings::{SeekDuringPlayback, Settings, Shortcut};
use cutlass_store::{Folder, Hold, StoreError};
use cutlass_transcription::{
    FileSource, ParakeetUnified, PcmSource, PrepareError, RecognizeError, Recognizer, columns, fit,
};

use crate::face::FaceTable;
use crate::probe::{self, Probed};
use crate::tokens;

/// Long-side steps the export dialog offers.
pub const LONG_SIDES: [i32; 6] = [480, 720, 1080, 1440, 2160, 4320];

/// A blend the Transitions shelf can drop onto a main-track cut.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransitionOffer {
    pub id: &'static str,
    pub name: &'static str,
    /// Preset length in microseconds. A short clip stores a shorter blend.
    pub duration: i64,
    /// `true` splits the window across the cut. `false` keeps it on the outgoing clip.
    pub overlap: bool,
}

impl TransitionOffer {
    #[must_use]
    pub fn card_id(self) -> String {
        format!("transition:{}", self.id)
    }
}

/// How long a notice shows, its fade in and out included.
pub const NOTICE_SHOWN: Duration = Duration::from_millis(2_160);

/// How long a notice takes to fade out at the end of [`NOTICE_SHOWN`].
pub const NOTICE_FADE: Duration = Duration::from_millis(160);

/// The shortest time between two automatic writes of the project.
pub const AUTOSAVE_GAP: Duration = Duration::from_millis(500);

/// A clip's waveform over its visible pixel columns.
#[derive(Clone, Debug, PartialEq)]
pub struct ClipWave {
    /// The first column, pixels from the clip's left edge.
    pub first: u32,
    /// One level, 0 to 1, per column.
    pub levels: Vec<f32>,
    /// The clip's gain at each column, its volume keyframes applied: 1 is
    /// the file as recorded.
    pub gains: Vec<f32>,
    /// The sound of a video clip, drawn in a band under its picture.
    pub under_picture: bool,
    /// The clip's track is muted, or silenced by another track's solo.
    pub muted: bool,
}

/// One filmstrip tile of a clip, and its picture once stored.
pub struct FilmCell {
    pub tile: FilmTile,
    pub key: ThumbKey,
    pub thumb: Option<std::sync::Arc<Thumb>>,
}

/// The color stored on a marker dropped from the timeline.
pub const MARKER_COLOR: &str = "#E8B931";

/// Dissolve crosses the cut. Dip stays on the outgoing clip.
pub const TRANSITION_OFFERS: &[TransitionOffer] = &[
    TransitionOffer {
        id: "dissolve",
        name: "Dissolve",
        duration: 500_000,
        overlap: true,
    },
    TransitionOffer {
        id: "dip",
        name: "Dip",
        duration: 400_000,
        overlap: false,
    },
];

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

#[derive(Clone, Debug, PartialEq)]
pub struct LaneView {
    pub track_id: TrackId,
    pub kind: TrackType,
    pub main: bool,
    pub mute: bool,
    pub hide: bool,
    pub lock: bool,
    pub solo: bool,
    pub mute_by_solo: bool,
    pub clips: Vec<ClipSpan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClipSpan {
    pub id: SegmentId,
    pub name: String,
    pub start: Micros,
    pub duration: Micros,
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
    pub path: PathBuf,
    pub long_side: i32,
    pub codec: CodecId,
    pub replace: bool,
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

/// One saved project in the launch list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ListedProject {
    pub path: PathBuf,
    pub name: String,
    pub duration: Micros,
    pub modified: SystemTime,
}

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

struct PendingMove {
    id: SegmentId,
    name: String,
    start: Micros,
    duration: Micros,
    track_id: TrackId,
    snapped: bool,
}

/// A clip kept for paste. `offset` is its start relative to the earliest copied clip.
#[derive(Clone)]
struct CopiedClip {
    track_id: TrackId,
    offset: i64,
    segment: Segment,
    /// Offsets from the clip start. The paste stores them on a new material.
    marks: Vec<TimeMarkItem>,
    /// Source times. The paste stores them on a new material.
    beats: Vec<i64>,
}

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
/// How far the picture may drift from the heard audio before it jumps to it.
const AUDIO_SNAP: i64 = 60_000;
/// The share of the drift corrected each tick. Small enough that the device's
/// buffer-sized clock steps do not reach the picture.
const AUDIO_PULL: i64 = 16;

/// The playhead after one tick: moved by the frame's elapsed time, then
/// pulled a little toward the heard audio, or onto it when far off.
fn follow_audio(wall: i64, heard: i64) -> i64 {
    let drift = heard.saturating_sub(wall);
    if drift.abs() > AUDIO_SNAP {
        heard
    } else {
        wall.saturating_add(drift / AUDIO_PULL)
    }
}

/// The audio stream playback follows. The playhead is where its heard
/// frames reach, so the picture follows the sound. The mixer thread fills it.
#[derive(Clone, Copy, Debug)]
struct AudioStream {
    /// Timeline time of the first streamed frame.
    start: Micros,
    /// The device's played count when this stream started.
    played: u64,
    /// The device has played this stream's first frame.
    heard: bool,
}

impl AudioStream {
    fn time_at(self, frames: u64) -> Micros {
        let micros = i128::from(frames) * 1_000_000 / i128::from(PREVIEW_RATE);
        Micros(
            self.start
                .0
                .saturating_add(i64::try_from(micros).unwrap_or(i64::MAX)),
        )
    }
}

pub struct Shell {
    decoders: DecodeFactory,
    encoders: EncodeFactory,
    readers: ReaderFactory,
    output: AudioOutput,
    heard: Heard,
    preview_cache: RefCell<FrameCache>,
    settings: Settings,
    session: Option<Session>,
    folder: Option<Folder>,
    hold: Option<Hold>,
    /// The session revision the project folder holds, and when it was written.
    saved: Option<(u64, Instant)>,
    path: Option<PathBuf>,
    playhead: Micros,
    playing: bool,
    /// Present while playing on a device with a stream clock.
    stream: Option<AudioStream>,
    /// Mixes the stream on its own thread. Started on the first play.
    mixer: Option<Mixer>,
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
    /// The marker tool drops a mark on the ruler or on a clip.
    marking: bool,
    selected_marker: Option<MarkerPick>,
    /// The beat tool drops a mark on an audio clip.
    beating: bool,
    selected_beat: Option<BeatPick>,
    clip_drag: Option<ClipDrag>,
    pending_trim: Option<PendingTrim>,
    pending_moves: Vec<PendingMove>,
    notice: String,
    /// When the notice was set; it fades out [`NOTICE_SHOWN`] later.
    notice_since: Instant,
    /// The preview canvas shows black under the picture, or nothing.
    preview_black: bool,
    next_id: u64,
    /// The local speech model, loaded on the first transcript that can hear a clip.
    speech: RefCell<Option<ParakeetUnified>>,
    /// Timeline thumbnails, once the window starts them.
    thumbnails: Option<Thumbnails>,
    /// Thumbnails this frame's clips asked for, nearest the view first.
    film_asks: Vec<ThumbKey>,
    /// Timeline waveforms, once the window starts them.
    waveforms: Option<Waveforms>,
    /// Files this frame's clips asked for peaks of.
    wave_asks: Vec<PathBuf>,
    face: FaceTable,
    /// The preview canvas on eframe's GPU device, once the window has one.
    gpu: Option<crate::gpu::Preview>,
    /// The canvas shows a clip's earlier picture while the asked one decodes.
    preview_stale: bool,
    pub chrome: Chrome,
}

impl Shell {
    pub fn new(decoders: DecodeFactory, encoders: EncodeFactory, mut settings: Settings) -> Self {
        ensure_shortcut(&mut settings, "undo", "primary+z");
        ensure_shortcut(&mut settings, "redo", "primary+shift+z");
        ensure_shortcut(&mut settings, "play", "space");
        ensure_shortcut(&mut settings, "frame_back", "left");
        ensure_shortcut(&mut settings, "frame_forward", "right");
        ensure_shortcut(&mut settings, "delete", "delete");
        ensure_shortcut(&mut settings, "split", "s");
        ensure_shortcut(&mut settings, "copy", "primary+c");
        ensure_shortcut(&mut settings, "cut", "primary+x");
        ensure_shortcut(&mut settings, "paste", "primary+v");
        ensure_shortcut(&mut settings, "extract_audio", "primary+shift+s");
        Self {
            decoders,
            encoders,
            readers: crate::readers::readers_with(
                crate::register_codecs_with(&settings.hardware).0,
            ),
            output: AudioOutput::open(AudioHost::Software),
            heard: Heard::new(),
            preview_cache: RefCell::new(FrameCache::new()),
            settings,
            session: None,
            folder: None,
            hold: None,
            path: None,
            playhead: Micros(0),
            playing: false,
            stream: None,
            mixer: None,
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
            guides: Vec::new(),
            guide_preview: None,
            horizontal: Vec::new(),
            horizontal_preview: None,
            selected_mark: None,
            guides_shown: true,
            guides_locked: false,
            snapping: true,
            blade: false,
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
            thumbnails: None,
            film_asks: Vec::new(),
            waveforms: None,
            wave_asks: Vec::new(),
            face: FaceTable::load(),
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
        let Some(thumbnails) = self.thumbnails.as_ref() else {
            return Vec::new();
        };
        let Some(draft) = self.session.as_ref().map(Session::draft) else {
            return Vec::new();
        };
        let Some(segment) = draft
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .find(|segment| &segment.id == id)
        else {
            return Vec::new();
        };
        let Some(MaterialRef::Video(video)) = draft.materials.get(segment.material_id.as_str())
        else {
            return Vec::new();
        };
        let Some(path) = self.media_file(cutlass_engine::media_path(draft, segment)) else {
            return Vec::new();
        };
        let Ok(map) = cutlass_engine::placement_of(draft, segment) else {
            return Vec::new();
        };
        let still = video.kind == MetaType::Photo.as_str();
        // The file's upright shape once the worker has opened it; the stored
        // material size until then.
        let (shape_w, shape_h) = thumbnails.shape(&path).unwrap_or((
            u32::try_from(video.width).unwrap_or(0),
            u32::try_from(video.height).unwrap_or(0),
        ));
        let aspect = if shape_w > 0 && shape_h > 0 {
            f64::from(shape_w) / f64::from(shape_h)
        } else {
            16.0 / 9.0
        };
        let rate = draft.frame_rate().unwrap_or(FrameRate::FPS_30);
        let mut cells = Vec::new();
        for tile in film_tiles(clip_width, height * aspect, visible, self.pixel_size) {
            let time = Micros(segment.target.start.0.saturating_add(tile.offset.0));
            let Ok(source) = cutlass_engine::map_timeline(&map, time) else {
                continue;
            };
            let source = if still {
                Micros(0)
            } else {
                rate.frame_index(source)
                    .and_then(|frame| rate.frame_start(frame))
                    .unwrap_or(source)
            };
            let key = ThumbKey {
                path: path.clone(),
                time: source,
            };
            let thumb = thumbnails.get(&key);
            if thumb.is_none() {
                self.film_asks.push(key.clone());
            }
            cells.push(FilmCell { tile, key, thumb });
        }
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
        let (track, segment) = draft.tracks.iter().find_map(|track| {
            track
                .segments
                .iter()
                .find(|segment| &segment.id == id)
                .map(|segment| (track, segment))
        })?;
        let muted = track.attribute.contains(TrackAttribute::MUTE)
            || track.attribute.contains(TrackAttribute::MUTE_BY_SOLO);
        let under_picture = match draft.materials.get(segment.material_id.as_str()) {
            Some(MaterialRef::Audio(_)) => false,
            Some(MaterialRef::Video(video)) if video.kind != MetaType::Photo.as_str() => true,
            _ => return None,
        };
        let path = self.media_file(cutlass_engine::media_path(draft, segment))?;
        let map = cutlass_engine::placement_of(draft, segment).ok()?;
        let Some(peaks) = waveforms.get(&path) else {
            self.wave_asks.push(path);
            // Hold a video's band open while it is read, so the filmstrip
            // does not jump when the sound arrives.
            return under_picture.then(|| ClipWave {
                first: 0,
                levels: Vec::new(),
                gains: Vec::new(),
                under_picture,
                muted,
            });
        };
        if !peaks.complete {
            self.wave_asks.push(path);
        } else if peaks.levels.is_empty() {
            return None;
        }
        let pixel_size = self.pixel_size;
        let usable = |value: f64| value.is_finite() && value > 0.0;
        if !usable(pixel_size) || !usable(clip_width) {
            return None;
        }
        let first = visible.start.max(0.0).floor();
        let last = visible.end.min(clip_width).ceil();
        let start = segment.target.start.0;
        let end = segment.target.end().0.saturating_sub(1).max(start);
        let at = |x: f64| {
            let time = Micros((start as f64 + x / pixel_size) as i64).min(Micros(end));
            cutlass_engine::map_timeline(&map, time).ok()
        };
        let mut levels = Vec::new();
        let mut gains = Vec::new();
        let mut x = first;
        while x < last && levels.len() < 16_384 {
            let level = match (at(x), at(x + 1.0)) {
                (Some(from), Some(to)) => peaks.level(from, to),
                _ => 0.0,
            };
            let offset = Micros((x / pixel_size) as i64);
            let gain = cutlass_engine::clip_volume(segment, offset).max(0.0) as f32;
            levels.push((level * gain).min(1.0));
            gains.push(gain);
            x += 1.0;
        }
        Some(ClipWave {
            first: first as u32,
            levels,
            gains,
            under_picture,
            muted,
        })
    }

    /// A stored media path as a file: inside the project when relative.
    fn media_file(&self, stored: Option<&str>) -> Option<PathBuf> {
        let stored = stored?;
        if Path::new(stored).is_absolute() {
            return Some(PathBuf::from(stored));
        }
        cutlass_engine::media_in_project(self.path.as_deref()?, stored)
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

    pub fn set_seek_during_playback(&mut self, value: SeekDuringPlayback) {
        self.settings.playback.seek_during_playback = value;
    }

    pub fn set_scrub_audio(&mut self, value: bool) {
        self.settings.playback.scrub_audio = value;
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

    /// Where new projects are created and where the launch list looks.
    ///
    /// On macOS this is `~/Library/Application Support/Cutlass/Projects`.
    /// The directory is created when the first project is saved.
    #[must_use]
    pub fn projects_directory() -> PathBuf {
        application_support()
            .map(|base| base.join("Cutlass").join("Projects"))
            .unwrap_or_else(|| std::env::temp_dir().join("cutlass-projects"))
    }

    /// The user's settings file.
    ///
    /// On macOS this is `~/Library/Application Support/Cutlass/settings.json`.
    #[must_use]
    pub fn settings_path() -> PathBuf {
        application_support()
            .map(|base| base.join("Cutlass"))
            .unwrap_or_else(|| std::env::temp_dir().join("cutlass"))
            .join("settings.json")
    }

    /// Projects under `root`, newest first. A folder without a draft is skipped.
    #[must_use]
    pub fn list_projects(root: &Path) -> Vec<ListedProject> {
        let Ok(entries) = std::fs::read_dir(root) else {
            return Vec::new();
        };
        let mut projects = Vec::new();
        for entry in entries {
            let Ok(entry) = entry else {
                continue;
            };
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let Ok(folder) = Folder::read(&path) else {
                continue;
            };
            let modified = entry
                .metadata()
                .and_then(|metadata| metadata.modified())
                .unwrap_or(UNIX_EPOCH);
            let name = if folder.draft.name.is_empty() {
                entry.file_name().to_string_lossy().into_owned()
            } else {
                folder.draft.name
            };
            projects.push(ListedProject {
                path,
                name,
                duration: folder.draft.duration,
                modified,
            });
        }
        projects.sort_by(|left, right| {
            right
                .modified
                .cmp(&left.modified)
                .then_with(|| left.name.cmp(&right.name))
        });
        projects
    }

    /// Write a new empty project under `root` and return its folder.
    pub fn create_project(root: &Path) -> Result<PathBuf, ShellError> {
        std::fs::create_dir_all(root).map_err(|error| ShellError::Io(error.to_string()))?;
        let (name, path, id) = next_project_slot(root);
        let mut draft = Draft::empty(id);
        draft.name = name;
        let now = unix_millis();
        draft.create_time = now;
        draft.update_time = now;
        let mut folder = Folder::new(draft);
        folder.set_linkage_enabled(true);
        folder.save(&path).map_err(ShellError::Store)?;
        Ok(path)
    }

    /// Write the open project, then leave it so the launch list is showing.
    pub fn close_project(&mut self) -> Result<(), ShellError> {
        if self.session.is_some() {
            self.save()?;
        }
        self.session = None;
        self.folder = None;
        self.hold = None;
        self.saved = None;
        self.path = None;
        self.playhead = Micros(0);
        self.playing = false;
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
        self.heard.clear();
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

    pub fn draft(&self) -> Option<&Draft> {
        self.session.as_ref().map(Session::draft)
    }

    /// Software output and a live device are open. Linux and a missing device
    /// are not. Pictures still advance either way.
    pub fn audio_output_open(&self) -> bool {
        self.output.gap().is_none()
    }

    /// Replace the software buffer with the output for `host`.
    pub fn open_preview_audio(&mut self, host: AudioHost) {
        self.stream = None;
        self.mixer = None;
        self.output = AudioOutput::open(host);
    }

    #[cfg(test)]
    pub(crate) fn queued_preview(&self) -> &[u8] {
        self.output.queued()
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
        self.saved = Some((session.revision(), Instant::now()));
        self.session = Some(session);
        self.folder = Some(opened.folder);
        self.hold = Some(hold);
        self.playhead = Micros(0);
        self.playing = false;
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
        self.heard.clear();
        self.silence();
        Ok(())
    }

    pub fn save(&mut self) -> Result<(), ShellError> {
        let path = self.path.clone().ok_or(ShellError::NoProject)?;
        if !path.is_dir() {
            return Err(ShellError::MissingFolder(path));
        }
        let enabled = self
            .session
            .as_ref()
            .ok_or(ShellError::NoProject)?
            .linkage()
            .enabled;
        let draft = self
            .session
            .as_ref()
            .ok_or(ShellError::NoProject)?
            .draft()
            .clone();
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
        folder.save(&path).map_err(ShellError::Store)?;
        let revision = self.session.as_ref().map_or(0, Session::revision);
        self.saved = Some((revision, Instant::now()));
        Ok(())
    }

    /// Write the project when an edit has landed since the last write, at
    /// most once per [`AUTOSAVE_GAP`]. Returns how long until a held-back
    /// write is due, so the window can wake for it.
    pub fn autosave(&mut self) -> Option<Duration> {
        if !self.unsaved() {
            return None;
        }
        let (_, at) = self.saved?;
        let revision = self.session.as_ref()?.revision();
        let wait = AUTOSAVE_GAP.saturating_sub(at.elapsed());
        if !wait.is_zero() {
            return Some(wait);
        }
        if let Err(error) = self.save() {
            cutlass_logger::error!("autosave: {error}");
            self.saved = Some((revision, Instant::now()));
            self.set_notice(format!("Could not save: {error}"));
        }
        None
    }

    /// An edit has landed since the folder was last written.
    pub fn unsaved(&self) -> bool {
        match (self.session.as_ref(), self.saved) {
            (Some(session), Some((saved, _))) => session.revision() != saved,
            _ => false,
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
        let (stored, copied) = place_in_project(&project, path)?;
        let imported = self.import_medium(probed, stored);
        if imported.is_err()
            && let Some(copied) = copied
        {
            let _ = std::fs::remove_file(copied);
        }
        imported
    }

    pub fn lanes(&self) -> Vec<LaneView> {
        let Some(draft) = self.draft() else {
            return Vec::new();
        };
        lanes(draft)
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
                    })
                    .collect();
                Some(LaneView {
                    track_id: lane.track_id,
                    kind: lane.kind,
                    main: lane.main,
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
        if self.playing {
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
        self.start_stream();
        self.preroll();
        self.reveal_playhead();
    }

    pub fn pause(&mut self) {
        self.playing = false;
        self.silence();
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
        self.stream = None;
        if let Some(mixer) = &self.mixer {
            mixer.stop();
        }
        self.output.clear_stream();
        self.output.queue(&[]);
    }

    /// Stream audio from the playhead on the mixer thread. A host with no
    /// stream clock queues one slice per tick instead.
    fn start_stream(&mut self) {
        self.output.clear_stream();
        let (Some(clock), Some(session)) = (self.output.clock(), self.session.as_ref()) else {
            self.stream = None;
            self.queue_preview(PLAY_FRAMES);
            return;
        };
        if self.mixer.is_none() {
            self.mixer = Mixer::start(&self.output, self.readers.clone());
        }
        let Some(mixer) = &self.mixer else {
            self.stream = None;
            self.queue_preview(PLAY_FRAMES);
            return;
        };
        mixer.play(MixJob {
            draft: session.draft().clone(),
            revision: session.revision(),
            root: self.path.clone(),
            start: self.playhead,
            end: self.cut_duration(),
        });
        self.stream = Some(AudioStream {
            start: self.playhead,
            played: clock.played,
            heard: false,
        });
    }

    /// Restart the stream from the playhead when the draft changed while
    /// playing, so the mixer hears the edit.
    fn follow_edits(&mut self) {
        let (Some(mixer), Some(session)) = (&self.mixer, self.session.as_ref()) else {
            return;
        };
        if mixer.revision() != Some(session.revision()) {
            self.start_stream();
        }
    }

    /// Where the stream's heard frames reach, or `None` without a stream.
    #[cfg(test)]
    fn heard_time(&self) -> Option<Micros> {
        let stream = self.stream?;
        Some(stream.time_at(self.heard_frames()?))
    }

    /// Frames of the stream the device has played.
    fn heard_frames(&self) -> Option<u64> {
        let stream = self.stream?;
        let clock = self.output.clock()?;
        let mixed = self.mixer.as_ref()?.mixed();
        Some(clock.heard().saturating_sub(stream.played).min(mixed))
    }

    /// The playhead after `elapsed` while streaming. Before the device plays
    /// the stream's first frame the picture holds, so it starts with the
    /// sound. On that first frame it lands on the heard time, and after that
    /// it moves by `elapsed` and leans toward the audio.
    fn streamed_playhead(&mut self, wall: i64) -> Option<i64> {
        let heard_frames = self.heard_frames()?;
        let stream = self.stream.as_mut()?;
        let heard = stream.time_at(heard_frames).0;
        if heard_frames == 0 {
            return Some(stream.start.0);
        }
        if !stream.heard {
            stream.heard = true;
            return Some(heard);
        }
        Some(follow_audio(wall, heard))
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

    /// Move the playhead while playing, by `elapsed`. With a stream clock it
    /// is also pulled toward the heard audio, so the picture stays with the
    /// sound without taking on the device's uneven steps.
    pub fn advance(&mut self, elapsed: i64) {
        if !self.playing || elapsed <= 0 {
            return;
        }
        let end = self.cut_duration().0;
        if end <= 0 {
            self.playing = false;
            self.playhead = Micros(0);
            self.silence();
            return;
        }
        let wall = self.playhead.0.saturating_add(elapsed);
        let mut next = match self.streamed_playhead(wall) {
            Some(streamed) => streamed.max(self.playhead.0),
            None => wall,
        };
        let mut wrapped = false;
        if next >= end {
            if self.settings.playback.loop_playback {
                next %= end;
                wrapped = true;
            } else {
                next = end;
                self.playing = false;
            }
        }
        self.playhead = Micros(next);
        self.preroll();
        if !self.playing {
            self.silence();
        } else if wrapped && self.stream.is_some() {
            self.start_stream();
        } else if self.stream.is_some() {
            self.follow_edits();
        } else {
            self.queue_preview(preview_frames(elapsed));
        }
        self.reveal_playhead();
    }

    #[cfg(test)]
    pub(crate) fn set_readers(&mut self, readers: ReaderFactory) {
        self.mixer = None;
        self.readers = readers;
        self.preview_cache.borrow_mut().clear();
        self.heard.clear();
    }

    fn queue_preview(&mut self, frames: u32) {
        let Some(session) = self.session.as_ref() else {
            self.output.queue(&[]);
            return;
        };
        let root = self.path.clone();
        queue_mix(
            &mut self.output,
            PreviewMix {
                draft: session.draft(),
                revision: session.revision(),
                time: self.playhead,
                frames,
                readers: &self.readers,
                heard: &mut self.heard,
                root: root.as_deref(),
            },
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
                &self.face,
            )
            .map_err(|error| ShellError::Eval(error.to_string()))?;
        let stale = cache.take_stale();
        drop(cache);
        let layers = match composed {
            cutlass_eval::Composed::Layers(layers) => layers,
            cutlass_eval::Composed::Frame(frame) => cutlass_eval::FrameLayers::pixels(frame)
                .ok_or_else(|| ShellError::Eval("the frame has no pixels".to_owned()))?,
        };
        let shown = gpu
            .show(&layers, key)
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
                &self.face,
            )
            .map_err(|error| ShellError::Eval(error.to_string()))?;
        Ok(Picture {
            width: picture.width,
            height: picture.height,
            rgba: picture.rgba,
        })
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
        let action = self
            .settings
            .shortcuts
            .iter()
            .find(|shortcut| shortcut.keys == keys)
            .map(|shortcut| shortcut.action.clone());
        match action.as_deref() {
            Some("undo") => self.undo(),
            Some("redo") => self.redo(),
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
            _ => Err(ShellError::Unavailable(format!("{keys} is not available"))),
        }
    }

    pub fn set_shortcut(&mut self, action: &str, keys: &str) {
        if let Some(shortcut) = self
            .settings
            .shortcuts
            .iter_mut()
            .find(|shortcut| shortcut.action == action)
        {
            shortcut.keys = keys.to_owned();
        }
    }

    pub fn toggle_attribute(&mut self, track_id: TrackId, bit: i32) -> Result<(), ShellError> {
        let on = !self.attribute_on(&track_id, bit);
        self.apply(Request::SetTrackAttribute { track_id, bit, on })
    }

    pub fn drag_segment(
        &mut self,
        segment_id: SegmentId,
        x: f64,
        y: f64,
    ) -> Result<(), ShellError> {
        let plan = self.plan_move(segment_id, x, y)?;
        self.pending_moves.clear();
        let plans = self.ordered_moves(self.group_plans(plan));
        if plans.iter().all(|plan| self.move_is_current(plan)) {
            return Ok(());
        }
        for plan in &plans {
            self.refuse_locked(&plan.id)?;
        }
        let requests: Vec<Request> = plans
            .into_iter()
            .map(|plan| Request::Move {
                segment_id: plan.id,
                track_id: plan.track_id,
                start: plan.start,
            })
            .collect();
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
        self.pending_moves = match self.plan_move(segment_id, x, y) {
            Ok(plan) => self.group_plans(plan),
            Err(_) => Vec::new(),
        };
    }

    /// The dragged clip's copy. A group drag still reports that clip here.
    #[must_use]
    pub fn move_preview(&self) -> Option<MovePreview> {
        self.pending_moves.first().map(|pending| MovePreview {
            id: pending.id.clone(),
            name: pending.name.clone(),
            start: pending.start,
            duration: pending.duration,
            track_id: pending.track_id.clone(),
        })
    }

    /// Every copy following the pointer, including the dragged clip.
    #[must_use]
    pub fn move_previews(&self) -> Vec<MovePreview> {
        self.pending_moves
            .iter()
            .map(|pending| MovePreview {
                id: pending.id.clone(),
                name: pending.name.clone(),
                start: pending.start,
                duration: pending.duration,
                track_id: pending.track_id.clone(),
            })
            .collect()
    }

    /// The original clip is one of the clips being dragged.
    #[must_use]
    pub fn moving(&self, id: &SegmentId) -> bool {
        self.pending_moves.iter().any(|pending| &pending.id == id)
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

    /// The body drag lands on the pointer time, pulled to a snap within twelve pixels.
    ///
    /// A start that matches the clip does not push an undo step. The draft is
    /// not written here.
    fn plan_move(&self, segment_id: SegmentId, x: f64, y: f64) -> Result<PendingMove, ShellError> {
        self.refuse_locked(&segment_id)?;
        let x = self.time_zero_x(x);
        let scale = self.scale();
        let pointer = Pointer { x, y };
        let guides = self.guide_times();
        let gesture = MoveGesture {
            segment_id: segment_id.clone(),
            pointer,
            scale,
            playhead: self.playhead,
            snap_threshold: self.snap_threshold(),
            guides: guides.clone(),
        };
        let draft = self.session_draft()?;
        let raw = match hit_test(draft, scale, pointer) {
            Ok(Some(hit)) => hit.time,
            Ok(None) => {
                return Err(ShellError::Unavailable(
                    "the drag missed the lanes".to_owned(),
                ));
            }
            Err(_) => {
                return Err(ShellError::Unavailable(
                    "the drag could not be read".to_owned(),
                ));
            }
        };
        if self.is_selected(&segment_id) && self.selection.len() > 1 {
            let snapped = snap_guides(
                draft,
                raw,
                self.playhead,
                self.snap_threshold(),
                Some(&segment_id),
                &guides,
            );
            let start = snapped.map(|target| target.time).unwrap_or(raw);
            let start = Micros(start.0.max(0));
            let (name, duration, track_id) = draft
                .tracks
                .iter()
                .find_map(|track| {
                    track.segments.iter().find_map(|segment| {
                        (segment.id == segment_id).then(|| {
                            (
                                clip_name(draft, segment),
                                segment.target.duration,
                                track.id.clone(),
                            )
                        })
                    })
                })
                .ok_or_else(|| {
                    ShellError::Unavailable("the clip is not on the timeline".to_owned())
                })?;
            return Ok(PendingMove {
                id: segment_id,
                name,
                start,
                duration,
                track_id,
                snapped: start != raw,
            });
        }
        let outcome = move_gesture(draft, &gesture)
            .map_err(|_| ShellError::Unavailable("the drag could not be read".to_owned()))?;
        let (segment_id, track_id, start) = match outcome {
            GestureOutcome::Edit(Request::Move {
                segment_id,
                track_id,
                start,
            }) => (segment_id, track_id, start),
            GestureOutcome::Edit(_) => {
                return Err(ShellError::Unavailable(
                    "the drag could not be read".to_owned(),
                ));
            }
            GestureOutcome::Miss => {
                return Err(ShellError::Unavailable(
                    "the drag missed the lanes".to_owned(),
                ));
            }
            GestureOutcome::NewLane { .. } => {
                return Err(ShellError::Unavailable(
                    "a new lane is not available".to_owned(),
                ));
            }
        };
        let (name, duration) = locate(draft, &segment_id)
            .map(|(segment, _)| (clip_name(draft, segment), segment.target.duration))
            .ok_or_else(|| ShellError::Unavailable("the clip is not on the timeline".to_owned()))?;
        Ok(PendingMove {
            id: segment_id,
            name,
            start,
            duration,
            track_id,
            snapped: start != raw,
        })
    }

    /// Selected clips travel with the dragged one. A clip already at the left
    /// edge stops the whole group there, so the gaps stay.
    fn group_plans(&self, dragged: PendingMove) -> Vec<PendingMove> {
        let Some(draft) = self.draft() else {
            return vec![dragged];
        };
        let Some((segment, _)) = locate(draft, &dragged.id) else {
            return vec![dragged];
        };
        if !self.is_selected(&dragged.id) || self.selection.len() < 2 {
            return vec![dragged];
        }
        let origin = segment.target.start.0;
        let mut members = Vec::new();
        let mut min_start = origin;
        for track in &draft.tracks {
            for segment in &track.segments {
                if !self.is_selected(&segment.id) {
                    continue;
                }
                min_start = min_start.min(segment.target.start.0);
                members.push(PendingMove {
                    id: segment.id.clone(),
                    name: clip_name(draft, segment),
                    start: segment.target.start,
                    duration: segment.target.duration,
                    track_id: track.id.clone(),
                    snapped: false,
                });
            }
        }
        if members.len() < 2 {
            return vec![dragged];
        }
        let mut delta = dragged.start.0.saturating_sub(origin);
        if min_start.saturating_add(delta) < 0 {
            delta = min_start.saturating_neg();
        }
        let dragged_id = dragged.id.clone();
        let snapped_at = dragged.start;
        let was_snapped = dragged.snapped;
        for member in &mut members {
            member.start = Micros(member.start.0.saturating_add(delta));
            if member.id == dragged_id {
                member.snapped = was_snapped && member.start == snapped_at;
            }
        }
        let dragged_first = |member: &PendingMove| member.id != dragged_id;
        members.sort_by_key(dragged_first);
        members
    }

    /// Later clips move first when the group is heading right, so a clip does
    /// not land on a selected clip that has not moved yet.
    fn ordered_moves(&self, mut plans: Vec<PendingMove>) -> Vec<PendingMove> {
        let later = self.draft().is_some_and(|draft| {
            plans.iter().any(|plan| {
                locate(draft, &plan.id)
                    .is_some_and(|(segment, _)| plan.start.0 > segment.target.start.0)
            })
        });
        if later {
            plans.sort_by_key(|plan| std::cmp::Reverse(plan.start.0));
        } else {
            plans.sort_by_key(|plan| plan.start.0);
        }
        plans
    }

    fn move_is_current(&self, plan: &PendingMove) -> bool {
        let Some(draft) = self.draft() else {
            return false;
        };
        draft.tracks.iter().any(|track| {
            track.id == plan.track_id
                && track
                    .segments
                    .iter()
                    .any(|segment| segment.id == plan.id && segment.target.start == plan.start)
        })
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
            let span = medium_span(draft, material_id).ok_or_else(|| {
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
        let segment_id = SegmentId::new(self.fresh("seg"));
        let mut requests = Vec::new();
        if let Some(kind) = lane {
            requests.push(Request::InsertTrack {
                track_id: track_id.clone(),
                kind,
            });
        }
        let mut segment = Segment::new(
            segment_id.clone(),
            material_id,
            TimeRange::new(start.0, duration.0),
        );
        segment.source_range = Some(TimeRange::new(0, duration.0));
        let link_owner = if self.linkage_enabled() {
            let draft = self.session_draft()?;
            let on_main = draft
                .tracks
                .iter()
                .any(|track| track.id == track_id && track.is_main());
            if on_main {
                None
            } else {
                exclusive_main_owner(draft, start, duration)
            }
        } else {
            None
        };
        requests.push(Request::Place {
            track_id,
            segment: Box::new(segment),
        });
        if let Some(owner) = link_owner {
            requests.push(Request::Link {
                owner,
                dependent: segment_id.clone(),
            });
        }
        self.apply_requests(&requests)?;
        self.select(Some(segment_id));
        if first_clip {
            let _ = self.zoom_fit();
        }
        Ok(())
    }

    /// Store a blend on the main-track cut nearest the pointer.
    ///
    /// The two clips must touch. The blend does not open a lane and does not
    /// move either clip. A preset longer than the clips is shortened to fit.
    pub fn drop_transition(&mut self, card_id: &str, x: f64, y: f64) -> Result<(), ShellError> {
        let offer = TRANSITION_OFFERS
            .iter()
            .find(|offer| offer.card_id() == card_id)
            .copied()
            .ok_or_else(|| {
                ShellError::Unavailable("the transition is not in the library".to_owned())
            })?;
        let x = self.time_zero_x(x);
        let scale = self.scale();
        let threshold = snap_window(self.pixel_size);
        let (segment_id, fitted) = {
            let draft = self.session_draft()?;
            let hit = hit_test(draft, scale, Pointer { x, y })
                .map_err(|_| ShellError::Unavailable("the drag could not be read".to_owned()))?;
            let Some(hit) = hit else {
                return Err(ShellError::Unavailable(
                    "the drag missed the lanes".to_owned(),
                ));
            };
            if !hit.lane.main {
                return Err(ShellError::Unavailable(
                    "a transition sits on the main track".to_owned(),
                ));
            }
            let (segment_id, outgoing, incoming) = main_cut(draft, hit.time, threshold)?;
            let fitted = fit_transition(offer.duration, outgoing, incoming, offer.overlap)
                .ok_or_else(|| ShellError::Unavailable("the clips are too short".to_owned()))?;
            (segment_id, fitted)
        };
        let transition = MaterialTransition {
            id: MaterialId::new(self.fresh("transition")),
            kind: "transition".to_owned(),
            name: offer.name.to_owned(),
            duration: Micros(fitted),
            is_overlap: offer.overlap,
            ..Default::default()
        };
        self.apply(Request::SetTransition {
            segment_id,
            transition: Box::new(transition),
        })
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
        let Some(session) = self.session.as_ref() else {
            return Err(ShellError::NoProject);
        };
        if session
            .draft()
            .main_track()
            .is_some_and(|track| track.attribute.contains(TrackAttribute::LOCK))
        {
            return Err(ShellError::Unavailable(
                "the main track is locked".to_owned(),
            ));
        }
        let on = !session.draft().config.maintrack_adsorb;
        self.apply(Request::SetMagnet { on })
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
        rate_label(rate)
    }

    /// Drop frame is a timecode label for 29.97 and 59.94.
    pub fn drop_frame_available(&self) -> bool {
        self.draft()
            .and_then(Draft::frame_rate)
            .is_some_and(drop_rate)
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
        let next = next_rate(current);
        let drop_frame = drop_rate(next);
        self.apply(Request::SetFrameRate {
            fps: stored_fps(next),
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
        let next = next_canvas(width, height);
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
        color_space_name(space)
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
            space: next_color_space(current),
        })
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
        Some(Inspection {
            segment_id: segment.id.clone(),
            start: segment.target.start,
            duration: segment.target.duration,
            speed: segment.speed,
            volume: segment.volume,
            scale_x: picture_scale(segment).0,
            scale_y: picture_scale(segment).1,
            rotation: segment.clip.rotation,
            transform_x: segment.clip.transform.x,
            transform_y: segment.clip.transform.y,
            opacity: segment.clip.alpha,
            text_size,
            text_color,
            text_align,
            letter_spacing,
            line_spacing,
            line_width,
            font,
            letterbox: letterbox_color(draft, segment),
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
        let playhead = self.playhead;
        let freeze = {
            let draft = self.session_draft()?;
            let (segment, _) = locate(draft, &segment_id).ok_or_else(|| {
                ShellError::Unavailable("the clip is not on the timeline".to_owned())
            })?;
            let Some(MaterialRef::Video(video)) = draft.materials.get(segment.material_id.as_str())
            else {
                return Err(ShellError::Unavailable(
                    "a freeze belongs on video".to_owned(),
                ));
            };
            if video.freeze.is_some() {
                None
            } else {
                let timestamp = held_frame(draft, segment, playhead)?;
                Some(FreezeInfo {
                    source_material_id: video.id.as_str().to_owned(),
                    source_material_path: video.path.clone(),
                    reverse_material_path: video.reverse_path.clone(),
                    timestamp,
                    rest: Default::default(),
                })
            }
        };
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
            if !can_turn(draft, segment, kind) {
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
            if !can_turn(draft, segment, kind) {
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
    /// with that clip. One undo removes the caption. The window waits while
    /// the local speech model runs.
    pub fn transcribe(&mut self) -> Result<(), ShellError> {
        let clip = self.selected_speech()?;
        self.ensure_speech()?;
        let mut source = FileSource::new(&clip.path);
        let words = {
            let cell = self.speech.borrow();
            let Some(recognizer) = cell.as_ref() else {
                return Err(ShellError::Unavailable(
                    "the speech model is not on this machine".to_owned(),
                ));
            };
            heard_words(recognizer, &mut source, &clip)?
        };
        self.write_caption(&clip, words)
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
        let draft = self.session_draft()?;
        let Some((segment, _)) = locate(draft, &segment_id) else {
            return Err(ShellError::Unavailable(
                "the clip is not on the timeline".to_owned(),
            ));
        };
        let path = match draft.materials.get(segment.material_id.as_str()) {
            Some(MaterialRef::Video(video)) => video.path.clone(),
            Some(MaterialRef::Audio(audio)) => audio.path.clone(),
            _ => {
                return Err(ShellError::Unavailable(
                    "a transcript belongs on a picture or a sound".to_owned(),
                ));
            }
        };
        if path.is_empty() {
            return Err(ShellError::Unavailable("the clip has no sound".to_owned()));
        }
        let range = segment
            .source_range
            .clone()
            .filter(|range| range.duration.0 > 0 && range.start.0 >= 0)
            .unwrap_or_else(|| TimeRange::new(0, segment.target.duration.0));
        if range.duration.0 <= 0 || segment.target.duration.0 <= 0 {
            return Err(ShellError::Unavailable("the clip has no sound".to_owned()));
        }
        let source_us = range.duration.0;
        let target_us = segment.target.duration.0;
        Ok(HeardClip {
            source_id: segment_id,
            path,
            range,
            source_us,
            target_us,
            reverse: segment.reverse,
            speed_holds: segment.speed.is_finite() && segment.speed > 0.0,
            target: segment.target.clone(),
        })
    }

    fn ensure_speech(&self) -> Result<(), ShellError> {
        if self.speech.borrow().is_some() {
            return Ok(());
        }
        let dir = speech_model_dir();
        if !dir.join("parakeet-unified-en-0.6b.nemo").is_file() {
            return Err(ShellError::Unavailable(
                "the speech model is not on this machine".to_owned(),
            ));
        }
        let model = ParakeetUnified::cpu(&dir).map_err(|_| {
            ShellError::Unavailable("the speech model could not be loaded".to_owned())
        })?;
        *self.speech.borrow_mut() = Some(model);
        Ok(())
    }

    fn write_caption(&mut self, clip: &HeardClip, words: WordColumns) -> Result<(), ShellError> {
        if words.text.is_empty() {
            return Err(ShellError::Unavailable("nothing was heard".to_owned()));
        }
        let existing = {
            let draft = self.session_draft()?;
            caption_track(draft, &clip.target)
        };
        let track_id = existing.unwrap_or_else(|| TrackId::new(self.fresh("captions")));
        let segment_id = SegmentId::new(self.fresh("caption"));
        let material_id = MaterialId::new(self.fresh("caption"));
        self.apply(Request::AddCaption {
            source_id: clip.source_id.clone(),
            track_id,
            segment_id,
            material_id,
            words,
        })
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
        let existing = {
            let draft = self.session_draft()?;
            let (segment, _) = locate(draft, source_id).ok_or_else(|| {
                ShellError::Unavailable("the clip is not on the timeline".to_owned())
            })?;
            let target = segment.target.clone();
            draft
                .tracks
                .iter()
                .find(|track| {
                    track.kind == TrackType::Audio
                        && !track.attribute.contains(TrackAttribute::LOCK)
                        && track
                            .segments
                            .iter()
                            .all(|other| other.target.overlaps(&target).ok() == Some(false))
                })
                .map(|track| track.id.clone())
        };
        let track_id = existing.unwrap_or_else(|| TrackId::new(self.fresh("sound")));
        let segment_id = SegmentId::new(self.fresh("sound"));
        let material_id = MaterialId::new(self.fresh("sound"));
        self.apply(Request::ExtractAudio {
            source_id: source_id.clone(),
            track_id,
            segment_id,
            material_id,
        })
    }

    /// The selected clip is a video, so Split scenes can cut it.
    #[must_use]
    pub fn split_scenes_enabled(&self) -> bool {
        self.selected_video().is_some()
    }

    /// Cut the selected video where its picture changes. One undo restores
    /// the clip. The window waits while the pictures are read.
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
        let cuts = scene_cuts(&self.readers, &file, &range).map_err(scene_message)?;
        let times = {
            let draft = self.session_draft()?;
            timeline_cuts(draft, &segment_id, &range, &cuts)
        };
        self.split_at_times(&segment_id, &times)
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
            kept_cuts(draft, segment, times)
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
        can_turn(draft, segment, kind)
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
        self.apply(Request::SetPosition { segment_id, x, y })
    }

    /// Scale the selected picture. One undo restores the previous scale.
    pub fn set_scale(&mut self, scale: f64) -> Result<(), ShellError> {
        let segment_id = self.picture_segment("a placement belongs on a picture")?;
        self.refuse_locked(&segment_id)?;
        self.apply(Request::SetScale { segment_id, scale })
    }

    /// Set how solid the selected picture is. `1` is solid and `0` is clear.
    /// One undo restores the previous opacity.
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
        if !can_turn(draft, segment, kind) {
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
        self.face = FaceTable::named(name, bytes);
    }

    #[cfg(test)]
    pub fn use_faces(&mut self, faces: Vec<(&str, Vec<u8>)>) {
        self.face = FaceTable::from_pairs(
            faces
                .into_iter()
                .map(|(name, bytes)| (name.to_owned(), bytes))
                .collect(),
        );
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

    /// Remember the selection for paste. The draft stays as it is.
    pub fn copy_selected(&mut self) -> Result<(), ShellError> {
        if self.selection.is_empty() {
            return Err(ShellError::Unavailable("nothing is selected".to_owned()));
        }
        let draft = self.session_draft()?;
        let mut copies = Vec::new();
        for track in &draft.tracks {
            for segment in &track.segments {
                if !self.is_selected(&segment.id) {
                    continue;
                }
                if bit(track.attribute, TrackAttribute::LOCK) {
                    return Err(ShellError::Locked);
                }
                let owned = segment_time_mark(&draft.materials, segment).map(|raw| raw.id.clone());
                let marks = segment_time_mark(&draft.materials, segment)
                    .map(time_mark_items)
                    .unwrap_or_default();
                let beat_owned = segment_beats(&draft.materials, segment).map(|raw| raw.id.clone());
                let beats = if track_holds_beats(&track.kind) {
                    segment_beats(&draft.materials, segment)
                        .map(user_beats)
                        .unwrap_or_default()
                } else {
                    Vec::new()
                };
                let mut segment = segment.clone();
                if let Some(id) = owned {
                    segment.extra_material_refs.retain(|ref_id| ref_id != &id);
                }
                if let Some(id) = beat_owned {
                    segment.extra_material_refs.retain(|ref_id| ref_id != &id);
                }
                copies.push(CopiedClip {
                    track_id: track.id.clone(),
                    offset: segment.target.start.0,
                    segment,
                    marks,
                    beats,
                });
            }
        }
        if copies.is_empty() {
            return Err(ShellError::Unavailable("nothing is selected".to_owned()));
        }
        let earliest = copies.iter().map(|item| item.offset).min().unwrap_or(0);
        for item in &mut copies {
            item.offset = item.offset.saturating_sub(earliest);
        }
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
        if self.clipboard.is_empty() {
            return Err(ShellError::Unavailable("nothing to paste".to_owned()));
        }
        {
            let draft = self.session_draft()?;
            for item in &self.clipboard {
                let Some(track) = draft.tracks.iter().find(|track| track.id == item.track_id)
                else {
                    return Err(ShellError::Unavailable("the lane is gone".to_owned()));
                };
                if bit(track.attribute, TrackAttribute::LOCK) {
                    return Err(ShellError::Locked);
                }
            }
        }
        let playhead = self.playhead.0.max(0);
        let copies = self.clipboard.clone();
        let mut requests = Vec::new();
        let mut ids = Vec::new();
        let mut marks = Vec::new();
        let mut beats = Vec::new();
        for item in copies {
            let mut segment = item.segment.clone();
            let id = SegmentId::new(self.fresh("seg"));
            segment.id = id.clone();
            let start = playhead.saturating_add(item.offset);
            segment.target = TimeRange::new(start, segment.target.duration.0);
            let duration = segment.target.duration.0;
            let beat_sources: Vec<i64> = item
                .beats
                .iter()
                .copied()
                .filter(|source| beat_timeline_time(&segment, *source).is_some())
                .collect();
            requests.push(Request::Place {
                track_id: item.track_id.clone(),
                segment: Box::new(segment),
            });
            for mark in item.marks {
                if mark.start.0 < 0 || mark.start.0 >= duration {
                    continue;
                }
                let at = start.saturating_add(mark.start.0);
                marks.push((id.clone(), at, mark.color));
            }
            for source in beat_sources {
                beats.push((id.clone(), source));
            }
            ids.push(id);
        }
        for (segment_id, at, color) in marks {
            requests.push(Request::AddTimeMark {
                segment_id: Some(segment_id),
                at: Micros(at),
                color,
                material_id: MaterialId::new(self.fresh("time-mark")),
            });
        }
        for (segment_id, source) in beats {
            requests.push(Request::AddBeat {
                segment_id,
                source: Micros(source),
                material_id: MaterialId::new(self.fresh("beats")),
            });
        }
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
        let draft = self.draft()?;
        if let Some(id) = self
            .selection
            .iter()
            .rev()
            .find(|id| locate(draft, id).is_some_and(|(segment, _)| covers(segment, time)))
        {
            return Some(id.clone());
        }
        if let Some(id) = draft
            .main_track()
            .and_then(|track| clip_on_track(track, time))
        {
            return Some(id);
        }
        draft
            .tracks
            .iter()
            .find_map(|track| clip_on_track(track, time))
    }

    /// Cut one clip. Each piece must keep at least one frame. The original id stays selected.
    pub fn split_clip(&mut self, segment_id: SegmentId, at: Micros) -> Result<(), ShellError> {
        self.refuse_locked(&segment_id)?;
        let draft = self.session_draft()?;
        let (segment, _) = locate(draft, &segment_id)
            .ok_or_else(|| ShellError::Unavailable("the clip is not on the timeline".to_owned()))?;
        let start = segment.target.start.0;
        let end = start.saturating_add(segment.target.duration.0);
        let margin = split_margin(draft);
        if at.0 < start.saturating_add(margin) || at.0 > end.saturating_sub(margin) {
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

    /// Writes a PNG of the frame at the playhead.
    ///
    /// The long side is one of [`LONG_SIDES`]. A PNG is one picture, so the
    /// export range is that frame. The software encoder writes it, and the
    /// still mux checks the file before it replaces `request.path`. A codec
    /// this shell cannot mux is refused after the encoder probe.
    pub fn export(&mut self, request: ExportRequest) -> Result<(), ShellError> {
        if !LONG_SIDES.contains(&request.long_side) {
            return Err(ShellError::Unavailable(
                "the long side is not one of the steps".to_owned(),
            ));
        }
        if request.path.exists() && !request.replace {
            return Err(ShellError::Overwrite(request.path));
        }
        let root = self.path.clone();
        let draft = self.session_draft()?;
        let (width, height) = self.scaled_canvas(request.long_side);
        let params = CodecParams::video(request.codec, width, height);
        if let Err(error) = self.encoders.open(&params) {
            return Err(ShellError::CodecDeclined(error.to_string()));
        }
        if request.codec != CodecId::Png {
            return Err(ShellError::ExportUnavailable);
        }
        let Ok(long_side) = u32::try_from(request.long_side) else {
            return Err(ShellError::Unavailable(
                "the long side is not one of the steps".to_owned(),
            ));
        };
        let rate = draft.frame_rate().unwrap_or(FrameRate::FPS_30);
        let range = playhead_frame(rate, self.playhead)?;
        let export_request = cutlass_export::ExportRequest {
            range,
            long_side,
            codec: CodecId::Png,
            bitrate: 0,
            frame_rate: rate,
            container: cutlass_codec::ContainerId::Still,
            audio: None,
            sidecar: None,
            late_by: 0,
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
                fonts: &self.face,
            },
            |_, _, _| Ok(cutlass_eval::ClipPcm::Missing),
        )
        .map_err(export_failure)?;
        let Some(bytes) = mux.bytes() else {
            return Err(ShellError::Unavailable(
                "export did not finish a file".to_owned(),
            ));
        };
        write_replaced(&request.path, bytes)
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
                self.apply(Request::Import {
                    medium: ImportedMedium::Video(Box::new(material)),
                })
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
                self.apply(Request::Import {
                    medium: ImportedMedium::Audio(Box::new(material)),
                })
            }
        }
    }

    fn apply(&mut self, request: Request) -> Result<(), ShellError> {
        self.apply_requests(std::slice::from_ref(&request))
    }

    fn apply_requests(&mut self, requests: &[Request]) -> Result<(), ShellError> {
        let session = self.session.as_mut().ok_or(ShellError::NoProject)?;
        let revision = session.revision();
        session
            .apply(revision, requests)
            .map_err(ShellError::Edit)?;
        Ok(())
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

    fn scaled_canvas(&self, long_side: i32) -> (u32, u32) {
        let (width, height) = self
            .draft()
            .map(|draft| (draft.canvas_config.width, draft.canvas_config.height))
            .unwrap_or((1920, 1080));
        let width = f64::from(width.max(1));
        let height = f64::from(height.max(1));
        let scale = f64::from(long_side.max(1)) / width.max(height);
        let scaled_width = (width * scale).round().max(1.0);
        let scaled_height = (height * scale).round().max(1.0);
        (
            u32::try_from(scaled_width as i64).unwrap_or(1),
            u32::try_from(scaled_height as i64).unwrap_or(1),
        )
    }

    fn side_max(&self, window_width: f32, other: f32, floor: f32) -> f32 {
        let gaps = tokens::SPLIT * 2.0;
        (window_width - other - gaps - tokens::PREVIEW_MIN_WIDTH).max(floor)
    }

    fn stored_zoom(&self) -> Result<Zoom, ShellError> {
        Zoom::new(self.pixel_size).ok_or(ShellError::Zoom)
    }
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

fn ensure_shortcut(settings: &mut Settings, action: &str, keys: &str) {
    if settings
        .shortcuts
        .iter()
        .any(|shortcut| shortcut.action == action)
    {
        return;
    }
    settings.shortcuts.push(Shortcut {
        action: action.to_owned(),
        keys: keys.to_owned(),
        ..Shortcut::default()
    });
}

fn bit(attribute: TrackAttribute, flag: i32) -> bool {
    attribute.0 & flag != 0
}

/// The outgoing clip of the touching main-track cut nearest `time`.
///
/// A gap is not a cut. The last two numbers are the outgoing and incoming
/// durations.
fn main_cut(
    draft: &Draft,
    time: Micros,
    threshold: Micros,
) -> Result<(SegmentId, i64, i64), ShellError> {
    let track = draft
        .tracks
        .iter()
        .find(|track| track.is_main())
        .ok_or_else(|| ShellError::Unavailable("the drop missed the cut".to_owned()))?;
    if track.attribute.contains(TrackAttribute::LOCK) {
        return Err(ShellError::Locked);
    }
    let mut ordered: Vec<_> = track.segments.iter().collect();
    ordered.sort_by_key(|segment| segment.target.start.0);
    let mut best: Option<(i64, SegmentId, i64, i64)> = None;
    for pair in ordered.windows(2) {
        let left = pair[0];
        let right = pair[1];
        if left.target.duration.0 <= 0 || right.target.duration.0 <= 0 {
            continue;
        }
        let cut = left.target.end().0;
        if right.target.start.0 != cut {
            continue;
        }
        let distance = (time.0 - cut).saturating_abs();
        if distance > threshold.0 {
            continue;
        }
        let closer = best
            .as_ref()
            .is_none_or(|(best_distance, _, _, _)| distance < *best_distance);
        if closer {
            best = Some((
                distance,
                left.id.clone(),
                left.target.duration.0,
                right.target.duration.0,
            ));
        }
    }
    best.map(|(_, id, outgoing, incoming)| (id, outgoing, incoming))
        .ok_or_else(|| ShellError::Unavailable("the drop missed the cut".to_owned()))
}

/// Shorten `preset` so the blend window stays inside both clips.
fn fit_transition(preset: i64, outgoing: i64, incoming: i64, overlap: bool) -> Option<i64> {
    if preset <= 0 || outgoing <= 0 || incoming <= 0 {
        return None;
    }
    let fitted = if overlap {
        let even = outgoing.min(incoming).saturating_mul(2);
        let odd = outgoing
            .min(incoming.saturating_sub(1))
            .saturating_mul(2)
            .saturating_add(1);
        even.max(odd).min(preset)
    } else {
        preset.min(outgoing)
    };
    if fitted <= 0 { None } else { Some(fitted) }
}

/// The one main-track clip `span` sits on, when it sits on no other.
/// A locked main lane is not an owner. Touching at an endpoint is not an overlap.
fn exclusive_main_owner(draft: &Draft, start: Micros, duration: Micros) -> Option<SegmentId> {
    let end = start.0.checked_add(duration.0)?;
    if duration.0 <= 0 {
        return None;
    }
    let mut owner = None;
    for track in &draft.tracks {
        if !track.is_main() || track.attribute.contains(TrackAttribute::LOCK) {
            continue;
        }
        for segment in &track.segments {
            let segment_end = segment
                .target
                .start
                .0
                .checked_add(segment.target.duration.0)?;
            let overlaps = start.0 < segment_end && segment.target.start.0 < end;
            if !overlaps {
                continue;
            }
            if owner.is_some() {
                return None;
            }
            owner = Some(segment.id.clone());
        }
    }
    owner
}

fn draft_has_linkage(draft: &Draft) -> bool {
    draft
        .relationships
        .iter()
        .any(|item| item.kind == "linkage" && !item.id_to_id.is_empty())
}

fn medium_span(draft: &Draft, id: &str) -> Option<(TrackType, Micros)> {
    if let Some(video) = draft
        .materials
        .videos
        .iter()
        .find(|item| item.id.as_str() == id)
    {
        let kind = MetaType::parse(&video.kind)
            .and_then(lane_category)
            .unwrap_or(TrackType::Video);
        return Some((kind, video.duration));
    }
    let audio = draft
        .materials
        .audios
        .iter()
        .find(|item| item.id.as_str() == id)?;
    Some((TrackType::Audio, audio.duration))
}

fn held_frame(draft: &Draft, segment: &Segment, playhead: Micros) -> Result<Micros, ShellError> {
    let start = segment.target.start;
    let end = segment.target.end().0;
    let time = if playhead.0 >= start.0 && playhead.0 < end {
        playhead
    } else {
        start
    };
    cutlass_eval::freeze_timestamp(draft, segment, time)
        .ok_or_else(|| ShellError::Unavailable("the clip has no frame to hold".to_owned()))
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

fn split_margin(draft: &Draft) -> i64 {
    draft
        .frame_rate()
        .and_then(|rate| rate.frame_start(1))
        .map(|time| time.0.max(1))
        .unwrap_or(1)
}

fn scene_message(error: SceneError) -> ShellError {
    let message = match error {
        SceneError::Missing | SceneError::NoPicture => "the clip has no picture",
        SceneError::Decode => "the clip's picture could not be read",
    };
    ShellError::Unavailable(message.to_owned())
}

fn timeline_cuts(draft: &Draft, id: &SegmentId, range: &TimeRange, cuts: &[Micros]) -> Vec<Micros> {
    let Some((segment, _)) = locate(draft, id) else {
        return Vec::new();
    };
    let mut times = Vec::new();
    for cut in cuts {
        let time = source_on_timeline(draft, segment, *cut)
            .or_else(|| proportional(&segment.target, range, cut.0).map(Micros));
        if let Some(time) = time {
            times.push(time);
        }
    }
    times
}

fn proportional(target: &TimeRange, range: &TimeRange, source: i64) -> Option<i64> {
    let into = source.checked_sub(range.start.0)?;
    if range.duration.0 <= 0 || into < 0 || into >= range.duration.0 {
        return None;
    }
    let scaled =
        i128::from(into).checked_mul(i128::from(target.duration.0))? / i128::from(range.duration.0);
    target.start.0.checked_add(i64::try_from(scaled).ok()?)
}

fn kept_cuts(draft: &Draft, segment: &Segment, times: &[Micros]) -> Vec<Micros> {
    let start = segment.target.start.0;
    let end = start.saturating_add(segment.target.duration.0);
    let margin = split_margin(draft);
    let mut ordered: Vec<i64> = times.iter().map(|time| time.0).collect();
    ordered.sort_unstable();
    ordered.dedup();
    let mut kept = Vec::new();
    let mut last = start;
    for time in ordered {
        if time < start.saturating_add(margin) || time > end.saturating_sub(margin) {
            continue;
        }
        if time < last.saturating_add(margin) {
            continue;
        }
        if blend_blocks(draft, &segment.id, time) {
            continue;
        }
        kept.push(Micros(time));
        last = time;
    }
    kept
}

fn blend_blocks(draft: &Draft, segment_id: &SegmentId, at: i64) -> bool {
    let Some(track) = draft.tracks.iter().find(|track| {
        track
            .segments
            .iter()
            .any(|segment| &segment.id == segment_id)
    }) else {
        return false;
    };
    for segment in &track.segments {
        let Some(cut) = segment
            .target
            .start
            .0
            .checked_add(segment.target.duration.0)
        else {
            continue;
        };
        for material_id in &segment.extra_material_refs {
            let Some(MaterialRef::Transition(transition)) =
                draft.materials.get(material_id.as_str())
            else {
                continue;
            };
            let Some((window_start, window_end)) =
                blend_window(Micros(cut), transition.duration, transition.is_overlap)
            else {
                continue;
            };
            if at >= window_start.0 && at < window_end.0 {
                return true;
            }
        }
    }
    false
}

fn covers(segment: &Segment, time: Micros) -> bool {
    let start = segment.target.start.0;
    let end = start.saturating_add(segment.target.duration.0);
    time.0 >= start && time.0 < end
}

fn clip_on_track(track: &cutlass_model::Track, time: Micros) -> Option<SegmentId> {
    track
        .segments
        .iter()
        .find(|segment| covers(segment, time))
        .map(|segment| segment.id.clone())
}

struct HeardClip {
    source_id: SegmentId,
    path: String,
    range: TimeRange,
    source_us: i64,
    target_us: i64,
    reverse: bool,
    speed_holds: bool,
    target: TimeRange,
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
    map_words(words, clip)
}

fn map_words(words: WordColumns, clip: &HeardClip) -> Result<WordColumns, ShellError> {
    if words.text.is_empty() {
        return Err(ShellError::Unavailable("nothing was heard".to_owned()));
    }
    let mut start_time = Vec::with_capacity(words.text.len());
    let mut end_time = Vec::with_capacity(words.text.len());
    for (start_ms, end_ms) in words.start_time.iter().zip(&words.end_time) {
        let (start, end) = timeline_millis(*start_ms, *end_ms, clip)?;
        start_time.push(start);
        end_time.push(end);
    }
    Ok(WordColumns {
        start_time,
        end_time,
        text: words.text,
        rest: words.rest,
    })
}

fn timeline_millis(start_ms: i64, end_ms: i64, clip: &HeardClip) -> Result<(i64, i64), ShellError> {
    let start_us = millis_to_us(start_ms)?;
    let end_us = millis_to_us(end_ms)?;
    let (start_us, end_us) = if clip.reverse {
        let flipped_start = clip.source_us.checked_sub(end_us).ok_or_else(|| {
            ShellError::Unavailable("a word clock does not fit the clip".to_owned())
        })?;
        let flipped_end = clip.source_us.checked_sub(start_us).ok_or_else(|| {
            ShellError::Unavailable("a word clock does not fit the clip".to_owned())
        })?;
        (flipped_start, flipped_end)
    } else {
        (start_us, end_us)
    };
    let (start_us, end_us) = if clip.speed_holds {
        (
            scale_us(start_us, clip.source_us, clip.target_us)?,
            scale_us(end_us, clip.source_us, clip.target_us)?,
        )
    } else {
        (start_us, end_us)
    };
    let start_ms = start_us / 1_000;
    let end_ms = (end_us / 1_000).max(start_ms);
    Ok((start_ms, end_ms))
}

fn millis_to_us(millis: i64) -> Result<i64, ShellError> {
    millis
        .checked_mul(1_000)
        .ok_or_else(|| ShellError::Unavailable("a word clock does not fit the clip".to_owned()))
}

fn scale_us(value: i64, source_us: i64, target_us: i64) -> Result<i64, ShellError> {
    if source_us <= 0 {
        return Err(ShellError::Unavailable(
            "a word clock does not fit the clip".to_owned(),
        ));
    }
    let scaled = i128::from(value)
        .checked_mul(i128::from(target_us))
        .ok_or_else(|| ShellError::Unavailable("a word clock does not fit the clip".to_owned()))?;
    i64::try_from(scaled / i128::from(source_us))
        .map_err(|_| ShellError::Unavailable("a word clock does not fit the clip".to_owned()))
}

fn caption_track(draft: &Draft, range: &TimeRange) -> Option<TrackId> {
    draft
        .tracks
        .iter()
        .find(|track| {
            matches!(track.kind, TrackType::Text | TrackType::TextToVideoText)
                && !bit(track.attribute, TrackAttribute::LOCK)
                && track.segments.iter().all(|segment| {
                    segment
                        .target
                        .overlaps(range)
                        .is_ok_and(|overlaps| !overlaps)
                })
        })
        .map(|track| track.id.clone())
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

fn can_turn(draft: &Draft, segment: &Segment, kind: &TrackType) -> bool {
    match draft.materials.get(segment.material_id.as_str()) {
        Some(MaterialRef::Video(_) | MaterialRef::Text(_) | MaterialRef::Draft(_)) => true,
        Some(MaterialRef::Other(_))
            if draft
                .materials
                .stickers
                .iter()
                .any(|item| item.id == segment.material_id) =>
        {
            true
        }
        Some(_) => false,
        None => matches!(
            kind,
            TrackType::Text
                | TrackType::TextToVideoText
                | TrackType::Sticker
                | TrackType::Composition
        ),
    }
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

#[derive(Clone, Copy)]
struct CanvasShape {
    ratio: &'static str,
    width: i32,
    height: i32,
}

/// Shapes the preview can show. A new project is 1920×1080, so the next shape is 9:16.
const CANVAS_SHAPES: [CanvasShape; 13] = [
    CanvasShape {
        ratio: "16:9",
        width: 1920,
        height: 1080,
    },
    CanvasShape {
        ratio: "9:16",
        width: 1080,
        height: 1920,
    },
    CanvasShape {
        ratio: "1:1",
        width: 1080,
        height: 1080,
    },
    CanvasShape {
        ratio: "4:3",
        width: 1440,
        height: 1080,
    },
    CanvasShape {
        ratio: "3:4",
        width: 1080,
        height: 1440,
    },
    CanvasShape {
        ratio: "4:5",
        width: 1080,
        height: 1350,
    },
    CanvasShape {
        ratio: "2:1",
        width: 1920,
        height: 960,
    },
    CanvasShape {
        ratio: "21:9",
        width: 2520,
        height: 1080,
    },
    CanvasShape {
        ratio: "2.35:1",
        width: 1920,
        height: 817,
    },
    CanvasShape {
        ratio: "1.85:1",
        width: 1920,
        height: 1038,
    },
    CanvasShape {
        ratio: "1:2",
        width: 960,
        height: 1920,
    },
    CanvasShape {
        ratio: "128:27",
        width: 5120,
        height: 1080,
    },
    CanvasShape {
        ratio: "1.125:2.436",
        width: 1125,
        height: 2436,
    },
];

fn next_canvas(width: i32, height: i32) -> CanvasShape {
    let index = CANVAS_SHAPES
        .iter()
        .position(|shape| shape.width == width && shape.height == height)
        .map(|index| index.saturating_add(1))
        .unwrap_or(0);
    CANVAS_SHAPES[index % CANVAS_SHAPES.len()]
}

const COLOR_SPACES: [(i32, &str); 5] = [
    (-1, "Unknown"),
    (0, "None"),
    (2, "SDR 2020"),
    (67, "HLG"),
    (2084, "PQ"),
];

fn color_space_name(space: i32) -> String {
    COLOR_SPACES
        .iter()
        .find(|item| item.0 == space)
        .map(|item| item.1.to_owned())
        .unwrap_or_else(|| space.to_string())
}

fn next_color_space(current: i32) -> i32 {
    let index = COLOR_SPACES
        .iter()
        .position(|item| item.0 == current)
        .map(|index| index.saturating_add(1))
        .unwrap_or(0);
    COLOR_SPACES[index % COLOR_SPACES.len()].0
}

const PROJECT_RATES: [FrameRate; 9] = [
    FrameRate::FPS_23_976,
    FrameRate::FPS_24,
    FrameRate::FPS_25,
    FrameRate::FPS_29_97,
    FrameRate::FPS_30,
    FrameRate::FPS_50,
    FrameRate::FPS_59_94,
    FrameRate::FPS_60,
    FrameRate::FPS_120,
];

fn next_rate(current: Option<FrameRate>) -> FrameRate {
    let index = current
        .and_then(|rate| PROJECT_RATES.iter().position(|item| *item == rate))
        .map(|index| index.saturating_add(1))
        .unwrap_or(0);
    PROJECT_RATES[index % PROJECT_RATES.len()]
}

fn drop_rate(rate: FrameRate) -> bool {
    matches!(
        (rate.numerator, rate.denominator),
        (30_000, 1001) | (60_000, 1001)
    )
}

fn stored_fps(rate: FrameRate) -> f64 {
    f64::from(rate.numerator) / f64::from(rate.denominator)
}

fn rate_label(rate: FrameRate) -> String {
    let name = match (rate.numerator, rate.denominator) {
        (24_000, 1001) => "23.976",
        (30_000, 1001) => "29.97",
        (60_000, 1001) => "59.94",
        (numerator, 1) => return format!("{numerator} fps"),
        _ => "Frame rate",
    };
    format!("{name} fps")
}

fn format_timecode(label: TimecodeLabel) -> String {
    let separator = if label.drop_frame { ';' } else { ':' };
    format!(
        "{:02}:{:02}:{:02}{separator}{:02}",
        label.hours, label.minutes, label.seconds, label.frames
    )
}

fn place_in_project(
    project: &Path,
    source: &Path,
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
    std::fs::copy(source, &destination).map_err(|error| ShellError::Io(error.to_string()))?;
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

#[cfg(target_os = "macos")]
fn application_support() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(
        PathBuf::from(home)
            .join("Library")
            .join("Application Support"),
    )
}

#[cfg(target_os = "windows")]
fn application_support() -> Option<PathBuf> {
    std::env::var_os("APPDATA").map(PathBuf::from)
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
fn application_support() -> Option<PathBuf> {
    if let Some(xdg) = std::env::var_os("XDG_DATA_HOME") {
        return Some(PathBuf::from(xdg));
    }
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".local").join("share"))
}

fn next_project_slot(root: &Path) -> (String, PathBuf, String) {
    for index in 1..100_000 {
        let name = if index == 1 {
            "Project".to_owned()
        } else {
            format!("Project {index}")
        };
        let path = root.join(&name);
        if !path.exists() {
            return (name, path, format!("project-{index}"));
        }
    }
    let stamp = unix_millis();
    let name = format!("Project {stamp}");
    let path = root.join(&name);
    (name, path, format!("project-{stamp}"))
}

fn unix_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| i64::try_from(duration.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::{ExportRequest, LONG_SIDES, Shell, ShellError};
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

    fn shell() -> Shell {
        let (decoders, encoders) = crate::register_codecs();
        Shell::new(decoders, encoders, Settings::default())
    }

    #[test]
    fn the_playhead_moves_by_elapsed_time_and_leans_toward_the_audio() {
        assert_eq!(super::follow_audio(1_000_000, 1_000_000), 1_000_000);
        assert_eq!(super::follow_audio(1_000_000, 1_016_000), 1_001_000);
        assert_eq!(super::follow_audio(1_000_000, 984_000), 999_000);
        assert_eq!(super::follow_audio(1_000_000, 1_100_000), 1_100_000);
        let stream = super::AudioStream {
            start: Micros(2_000_000),
            played: 0,
            heard: true,
        };
        assert_eq!(
            stream.time_at(u64::from(cutlass_playback::MIX_WINDOW)),
            Micros(2_020_000)
        );
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
        if desktop.output.clock().is_none() {
            eprintln!("skipped: no audio device");
            return;
        }
        desktop.seek(Micros(1_000_000));
        desktop.play();
        assert!(desktop.mixer.is_some());
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
            if desktop.heard_frames() != Some(0) {
                break;
            }
            // Nothing heard after this tick, so nothing was heard during it.
            assert_eq!(desktop.playhead, Micros(1_000_000), "the picture held");
        }
        for _ in 0..30 {
            step(&mut desktop);
        }
        let heard = desktop.heard_time().unwrap();
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
        assert_eq!(
            desktop
                .mixer
                .as_ref()
                .and_then(cutlass_playback::Mixer::revision),
            Some(revision)
        );
        desktop.pause();
        assert!(desktop.stream.is_none());
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

    struct Temp {
        path: PathBuf,
    }

    impl Temp {
        fn new() -> Self {
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
        assert!(error.to_string().contains("already held"), "{error}");
        assert!(second.draft().is_none());
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
            assert_eq!((thumb.width, thumb.height), (171, 96));
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
        assert_eq!((thumb.width, thumb.height), (54, 96));
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
        assert_eq!(desktop.autosave(), None);
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
        desktop.undo().unwrap();
        assert_eq!(desktop.lanes()[0].clips.len(), 1);
        desktop.undo().unwrap();
        assert!(desktop.lanes().iter().all(|lane| lane.clips.is_empty()));
        desktop.redo().unwrap();
        assert_eq!(desktop.lanes()[0].clips.len(), 1);
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
        assert_eq!(blend.name, "Dip");
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
            .into_iter()
            .find(|lane| !lane.main)
            .expect("audio lane");
        desktop
            .toggle_attribute(audio.track_id, TrackAttribute::MUTE)
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
    fn export_writes_the_playhead_png_and_refuses_a_codec_it_cannot_mux() {
        let temp = Temp::new();
        let missing = shell().export(ExportRequest {
            path: temp.path.join("missing.png"),
            long_side: 480,
            codec: CodecId::Png,
            replace: false,
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
            long_side: 480,
            codec: CodecId::Vp8,
            replace: false,
        });
        assert!(matches!(declined, Err(ShellError::CodecDeclined(_))));
        assert!(!destination.exists());
        assert_eq!(desktop.draft().unwrap(), &before);

        desktop
            .export(ExportRequest {
                path: destination.clone(),
                long_side: 480,
                codec: CodecId::Png,
                replace: false,
            })
            .unwrap();
        let written = std::fs::read(&destination).unwrap();
        let (width, height, samples) = png_samples(&written);
        assert_eq!((width, height), (480, 240));
        assert!(
            samples
                .chunks(4)
                .all(|pixel| pixel == [0x33, 0x66, 0x99, 0xFF])
        );
        assert_eq!(desktop.draft().unwrap(), &before);

        let kept = std::fs::read(&destination).unwrap();
        let overwrite = desktop.export(ExportRequest {
            path: destination.clone(),
            long_side: 480,
            codec: CodecId::Png,
            replace: false,
        });
        assert!(matches!(overwrite, Err(ShellError::Overwrite(_))));
        assert_eq!(std::fs::read(&destination).unwrap(), kept);

        desktop.seek(Micros(2_000_000));
        desktop
            .export(ExportRequest {
                path: destination.clone(),
                long_side: 480,
                codec: CodecId::Png,
                replace: true,
            })
            .unwrap();
        let (width, height, samples) = png_samples(&std::fs::read(&destination).unwrap());
        assert_eq!((width, height), (480, 240));
        assert!(samples.iter().all(|sample| *sample == 0));
        assert_eq!(desktop.draft().unwrap(), &before);
        let clear = std::fs::read(&destination).unwrap();
        let movie = desktop.export(ExportRequest {
            path: destination.clone(),
            long_side: 480,
            codec: CodecId::H264,
            replace: true,
        });
        assert!(matches!(
            movie,
            Err(ShellError::CodecDeclined(_) | ShellError::ExportUnavailable)
        ));
        assert_eq!(std::fs::read(&destination).unwrap(), clear);

        let step = desktop.export(ExportRequest {
            path: destination.clone(),
            long_side: 1000,
            codec: CodecId::Png,
            replace: true,
        });
        assert!(matches!(step, Err(ShellError::Unavailable(_))));
        assert_eq!(LONG_SIDES, [480, 720, 1080, 1440, 2160, 4320]);
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
                long_side: 480,
                codec: CodecId::Png,
                replace: false,
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
        let video = &desktop.draft().unwrap().materials.videos[0];
        assert_eq!(video.crop, super::center_crop());
        assert_eq!(video.crop_ratio, "16:9");
        assert_eq!(video.crop_scale, 2.0);
        assert_eq!(
            desktop.draft().unwrap().tracks[0].segments[0].target,
            TimeRange::new(0, 2_000_000)
        );
        desktop.select(Some(SegmentId::new("again")));
        assert!(desktop.cropped());
        desktop.undo().unwrap();
        assert!(!desktop.cropped());
        desktop.redo().unwrap();
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
        assert_eq!(desktop.draft().unwrap().materials.videos.len(), 1);
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
        let under_support = path.ends_with(Path::new("Cutlass").join("Projects"));
        let fallback = path.ends_with("cutlass-projects");
        assert!(under_support || fallback, "{}", path.display());
    }
}
