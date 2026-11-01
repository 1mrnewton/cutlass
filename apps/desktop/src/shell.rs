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
use std::time::{SystemTime, UNIX_EPOCH};

use cutlass_decode::{CodecId, CodecParams, DecodeFactory};
use cutlass_editor::{
    DropGesture, DropOutcome, GestureError, GestureOutcome, MoveGesture, Pointer, TimelineScale,
    TrimGesture, TrimPlan, Zoom, ZoomBounds, drop_gesture, fit_pixel_size, hit_test, lane_category,
    lanes, last_accuracy_step, minimum_pixel_size, move_gesture, snap_guides, trim_gesture,
};
use cutlass_encode::EncodeFactory;
use cutlass_eval::{Heard, ReaderFactory};
use cutlass_model::{
    Draft, FrameRate, MaterialAudio, MaterialId, MaterialRef, MaterialVideo, MetaType, Micros,
    Segment, SegmentId, TimeRange, TimecodeLabel, TrackAttribute, TrackId, TrackType,
};
use cutlass_playback::{
    AudioHost, AudioOutput, FrameCache, PLAY_FRAMES, PreviewMix, preview_frames, queue_mix,
};
use cutlass_renderer_software::SoftwareDevice;
use cutlass_session::{EditError, ImportedMedium, Request, Session, TrimEdge};
use cutlass_settings::{SeekDuringPlayback, Settings, Shortcut};
use cutlass_store::{Folder, Hold, StoreError};

use crate::probe::{self, Probed};
use crate::tokens;

/// Long-side steps the export dialog offers.
pub const LONG_SIDES: [i32; 6] = [480, 720, 1080, 1440, 2160, 4320];

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
    pub material: String,
    pub text: Option<String>,
    pub words: cutlass_model::WordColumns,
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
}

/// The open clip menu, in window points.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipMenu {
    pub x: f32,
    pub y: f32,
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
    path: Option<PathBuf>,
    playhead: Micros,
    playing: bool,
    pixel_size: f64,
    /// Left edge of the lanes, in microseconds.
    view_origin: i64,
    lane_width: f64,
    /// Fit the cut on the next lane layout. An empty cut stays at the opening scale.
    fit_on_layout: bool,
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
    selected_guide: Option<i64>,
    guides_shown: bool,
    guides_locked: bool,
    /// The split tool cuts a clip at the pointer instead of moving it.
    blade: bool,
    clip_drag: Option<ClipDrag>,
    pending_trim: Option<PendingTrim>,
    pending_moves: Vec<PendingMove>,
    notice: String,
    next_id: u64,
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
        Self {
            decoders,
            encoders,
            readers: crate::readers::installed_readers(),
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
            pixel_size: opening_pixel_size(FrameRate::FPS_30),
            view_origin: 0,
            lane_width: 800.0,
            fit_on_layout: false,
            library_width: tokens::LIBRARY_WIDTH,
            inspector_width: tokens::INSPECTOR_WIDTH,
            timeline_height: tokens::TIMELINE_HEIGHT,
            selection: Vec::new(),
            clipboard: Vec::new(),
            clip_menu: None,
            guides: Vec::new(),
            guide_preview: None,
            selected_guide: None,
            guides_shown: true,
            guides_locked: false,
            blade: false,
            clip_drag: None,
            pending_trim: None,
            pending_moves: Vec::new(),
            notice: String::new(),
            next_id: 0,
            chrome: Chrome::default(),
        }
    }

    pub fn codecs_ready(&self) -> bool {
        let picture = CodecParams::video(CodecId::Png, 1, 1);
        self.decoders.open(&picture).is_ok() && self.encoders.open(&picture).is_ok()
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
        Folder::new(draft).save(&path).map_err(ShellError::Store)?;
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
        self.path = None;
        self.playhead = Micros(0);
        self.playing = false;
        self.view_origin = 0;
        self.selection.clear();
        self.clipboard.clear();
        self.clip_menu = None;
        self.clear_guides();
        self.blade = false;
        self.cancel_clip_drag();
        self.fit_on_layout = false;
        self.notice.clear();
        self.heard.clear();
        self.output.queue(&[]);
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

    /// Remember the lane width, and fit a saved cut the first time it is shown.
    pub fn prepare_lane(&mut self, width: f32) {
        self.set_lane_width(width);
        if self.fit_on_layout {
            self.fit_on_layout = false;
            let _ = self.zoom_fit();
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
        self.output = AudioOutput::open(host);
    }

    #[cfg(test)]
    pub(crate) fn queued_preview(&self) -> &[u8] {
        self.output.queued()
    }

    pub fn open_folder(&mut self, path: &Path) -> Result<(), ShellError> {
        let opened = cutlass_store::open(path).map_err(ShellError::Store)?;
        let hold = cutlass_store::hold(path).map_err(ShellError::Store)?;
        let rate = opened
            .folder
            .draft
            .frame_rate()
            .unwrap_or(FrameRate::FPS_30);
        self.path = Some(path.to_path_buf());
        self.session = Some(Session::open(opened.folder.draft.clone()));
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
        self.cancel_clip_drag();
        self.pixel_size = opening_pixel_size(rate);
        self.fit_on_layout = self.cut_duration().0 > 0;
        self.notice.clear();
        self.heard.clear();
        self.output.queue(&[]);
        Ok(())
    }

    pub fn save(&mut self) -> Result<(), ShellError> {
        let path = self.path.clone().ok_or(ShellError::NoProject)?;
        if !path.is_dir() {
            return Err(ShellError::MissingFolder(path));
        }
        let draft = self
            .session
            .as_ref()
            .ok_or(ShellError::NoProject)?
            .draft()
            .clone();
        let folder = self.folder.as_mut().ok_or(ShellError::NoProject)?;
        let main = folder.project.main_timeline_id.clone();
        folder.draft = draft.clone();
        if let Some(timeline) = folder.timelines.get_mut(&main) {
            *timeline = draft;
        }
        folder.save(&path).map_err(ShellError::Store)?;
        Ok(())
    }

    pub fn import_file(&mut self, path: &Path) -> Result<(), ShellError> {
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
        }
        self.playhead = Micros(time.0.max(0));
        if self.playing || self.settings.playback.scrub_audio {
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
        self.queue_preview(PLAY_FRAMES);
        self.reveal_playhead();
    }

    pub fn pause(&mut self) {
        self.playing = false;
        self.output.queue(&[]);
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

    pub fn advance(&mut self, elapsed: i64) {
        if !self.playing || elapsed <= 0 {
            return;
        }
        let end = self.cut_duration().0;
        if end <= 0 {
            self.playing = false;
            self.playhead = Micros(0);
            self.output.queue(&[]);
            return;
        }
        let mut next = self.playhead.0.saturating_add(elapsed);
        if next >= end {
            if self.settings.playback.loop_playback {
                next %= end;
            } else {
                next = end;
                self.playing = false;
            }
        }
        self.playhead = Micros(next);
        if self.playing {
            self.queue_preview(preview_frames(elapsed));
        } else {
            self.output.queue(&[]);
        }
        self.reveal_playhead();
    }

    #[cfg(test)]
    pub(crate) fn set_readers(&mut self, readers: ReaderFactory) {
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

    pub fn preview_bytes(&self) -> Result<Picture, ShellError> {
        let session = self.session.as_ref().ok_or(ShellError::NoProject)?;
        let time = self.sample_time();
        let mut device = SoftwareDevice::new();
        let root = self.path.clone();
        let picture = self
            .preview_cache
            .borrow_mut()
            .frame(session, &mut device, time, &self.readers, root.as_deref())
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
            snap_threshold: snap_window(self.pixel_size),
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
            snap_threshold: snap_window(self.pixel_size),
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
                snap_window(self.pixel_size),
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
                snap_threshold: snap_window(self.pixel_size),
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
        requests.push(Request::Place {
            track_id,
            segment: Box::new(segment),
        });
        self.apply_requests(&requests)?;
        self.select(Some(segment_id));
        if first_clip {
            let _ = self.zoom_fit();
        }
        Ok(())
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
        self.selected_guide = None;
        self.selection.clear();
        if let Some(id) = id {
            self.selection.push(id);
        }
    }

    /// Add `id` to the selection, or drop it when it is already there.
    pub fn toggle_selected(&mut self, id: SegmentId) {
        self.selected_guide = None;
        if let Some(index) = self.selection.iter().position(|item| item == &id) {
            self.selection.remove(index);
        } else {
            self.selection.push(id);
        }
    }

    /// Replace the selection with `ids`. `keep` adds them to the clips already selected.
    pub fn select_box(&mut self, ids: impl IntoIterator<Item = SegmentId>, keep: bool) {
        self.selected_guide = None;
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
            if self.selected_guide.is_some() {
                return self.remove_selected_guide();
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

    pub fn toggle_guides(&mut self) {
        self.guides_shown = !self.guides_shown;
    }

    pub fn toggle_guide_lock(&mut self) {
        self.guides_locked = !self.guides_locked;
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
        self.selected_guide = Some(time.0);
        Ok(())
    }

    pub fn select_guide(&mut self, time: Micros) {
        if self.guides.iter().any(|guide| guide.0 == time.0) {
            self.selected_guide = Some(time.0);
            self.selection.clear();
        }
    }

    #[must_use]
    pub fn selected_guide(&self) -> Option<Micros> {
        self.selected_guide.map(Micros)
    }

    pub fn remove_selected_guide(&mut self) -> Result<(), ShellError> {
        if self.guides_locked {
            return Err(ShellError::Unavailable("the guides are locked".to_owned()));
        }
        let Some(time) = self.selected_guide else {
            return Ok(());
        };
        self.guides.retain(|guide| guide.0 != time);
        self.selected_guide = None;
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
        self.selected_guide = None;
        self.guides_shown = true;
        self.guides_locked = false;
    }

    pub fn inspection(&self) -> Option<Inspection> {
        let draft = self.draft()?;
        let id = self.selected()?;
        let (segment, locked) = locate(draft, id)?;
        let (material, text, words) = material_label(draft, &segment.material_id);
        Some(Inspection {
            segment_id: segment.id.clone(),
            start: segment.target.start,
            duration: segment.target.duration,
            speed: segment.speed,
            volume: segment.volume,
            scale_x: segment.clip.scale.x,
            scale_y: segment.clip.scale.y,
            rotation: segment.clip.rotation,
            transform_x: segment.clip.transform.x,
            transform_y: segment.clip.transform.y,
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

    pub fn edit_transform(&mut self) -> Result<(), ShellError> {
        Err(ShellError::Unavailable(
            "a transform edit is not available".to_owned(),
        ))
    }

    pub fn edit_style(&mut self) -> Result<(), ShellError> {
        Err(ShellError::Unavailable(
            "a text style edit is not available".to_owned(),
        ))
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
                copies.push(CopiedClip {
                    track_id: track.id.clone(),
                    offset: segment.target.start.0,
                    segment: segment.clone(),
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
        for item in copies {
            let mut segment = item.segment.clone();
            let id = SegmentId::new(self.fresh("seg"));
            segment.id = id.clone();
            segment.target = TimeRange::new(
                playhead.saturating_add(item.offset),
                segment.target.duration.0,
            );
            requests.push(Request::Place {
                track_id: item.track_id.clone(),
                segment: Box::new(segment),
            });
            ids.push(id);
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

    /// The split tool is armed. A click on a clip cuts it at that time.
    #[must_use]
    pub fn blade(&self) -> bool {
        self.blade
    }

    pub fn toggle_blade(&mut self) {
        self.blade = !self.blade && self.has_clip();
    }

    /// Turn the blade off. Select uses this.
    pub fn clear_blade(&mut self) {
        self.blade = false;
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
    }

    pub fn unavailable(&mut self, control: &str) {
        self.notice = format!("{control} is not available");
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

fn material_label(
    draft: &Draft,
    id: &MaterialId,
) -> (String, Option<String>, cutlass_model::WordColumns) {
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
        Draft, FrameRate, MaterialAudio, MaterialColorConfig, MaterialId, MaterialVideo, Micros,
        Segment, SegmentId, TimeRange, Track, TrackAttribute, TrackType,
    };
    use cutlass_playback::AudioHost;
    use cutlass_session::{EditError, TrimEdge};
    use cutlass_settings::{SeekDuringPlayback, Settings};
    use cutlass_store::Folder;
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
        Folder::new(Draft::empty("draft")).save(dir).unwrap();
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

        let first = desktop.preview_bytes().unwrap();
        assert_eq!(KEPT_OPENS.load(Ordering::Relaxed), 2);
        assert_ne!(first.rgba, [0x33, 0x66, 0x99, 0xFF].repeat(4));
        assert!(first.rgba.iter().any(|byte| *byte != 0));

        desktop.seek(Micros(10));
        let later = desktop.preview_bytes().unwrap();
        assert_eq!(KEPT_OPENS.load(Ordering::Relaxed), 2);
        assert_eq!(later.rgba, first.rgba);

        let track = desktop.draft().unwrap().tracks[0].id.clone();
        desktop
            .toggle_attribute(track.clone(), TrackAttribute::MUTE)
            .unwrap();
        desktop.seek(Micros(20));
        let muted = desktop.preview_bytes().unwrap();
        assert_eq!(KEPT_OPENS.load(Ordering::Relaxed), 4);
        assert_eq!(muted.rgba, first.rgba);

        desktop
            .toggle_attribute(track, TrackAttribute::HIDE)
            .unwrap();
        let hidden = desktop.preview_bytes().unwrap();
        assert_eq!(KEPT_OPENS.load(Ordering::Relaxed), 4);
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
    fn the_inspector_edits_through_requests_and_leaves_words_on_an_empty_string() {
        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 2;
        draft.canvas_config.height = 1;
        let mut text = cutlass_model::MaterialText::new("caption");
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
        let error = desktop.edit_style().unwrap_err();
        assert!(error.to_string().contains("not available"), "{error}");
        assert_eq!(desktop.inspection().unwrap().words.text, ["Hello"]);
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
