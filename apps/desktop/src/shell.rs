//! The desktop session.
//!
//! A gesture becomes a [`cutlass_session::Request`] and goes through
//! [`Session::apply`]. Play, pause, and seek stay on the shell, so they do
//! not push an undo step. The picture is `eval::frame` on the software
//! device. This shell does not composite, and it does not open an audio device.

use std::path::{Path, PathBuf};

use cutlass_decode::{CodecId, CodecParams, DecodeFactory};
use cutlass_editor::{
    GestureOutcome, MoveGesture, Pointer, TimelineScale, Zoom, ZoomBounds, fit_pixel_size,
    hit_test, lane_category, lanes, last_accuracy_step, minimum_pixel_size, move_gesture,
};
use cutlass_encode::EncodeFactory;
use cutlass_eval::{Snapshot, frame};
use cutlass_model::{
    Draft, FrameRate, MaterialAudio, MaterialId, MaterialRef, MaterialVideo, MetaType, Micros,
    Segment, SegmentId, TimeRange, TimecodeLabel, TrackAttribute, TrackId, TrackType,
};
use cutlass_renderer_software::SoftwareDevice;
use cutlass_session::{EditError, ImportedMedium, Request, Session};
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
    pub close_requested: bool,
}

pub struct Shell {
    decoders: DecodeFactory,
    encoders: EncodeFactory,
    settings: Settings,
    session: Option<Session>,
    folder: Option<Folder>,
    hold: Option<Hold>,
    path: Option<PathBuf>,
    playhead: Micros,
    playing: bool,
    pixel_size: f64,
    lane_width: f64,
    library_width: f32,
    inspector_width: f32,
    timeline_height: f32,
    selected: Option<SegmentId>,
    notice: String,
    next_id: u64,
    pub chrome: Chrome,
}

impl Shell {
    pub fn new(decoders: DecodeFactory, encoders: EncodeFactory, mut settings: Settings) -> Self {
        ensure_shortcut(&mut settings, "undo", "primary+z");
        ensure_shortcut(&mut settings, "redo", "primary+shift+z");
        Self {
            decoders,
            encoders,
            settings,
            session: None,
            folder: None,
            hold: None,
            path: None,
            playhead: Micros(0),
            playing: false,
            pixel_size: opening_pixel_size(FrameRate::FPS_30),
            lane_width: 800.0,
            library_width: tokens::LIBRARY_WIDTH,
            inspector_width: tokens::INSPECTOR_WIDTH,
            timeline_height: tokens::TIMELINE_HEIGHT,
            selected: None,
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

    pub fn project_name(&self) -> String {
        self.session
            .as_ref()
            .map(|session| session.draft().name.clone())
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| "Untitled".to_owned())
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

    pub fn revision(&self) -> Option<u64> {
        self.session.as_ref().map(Session::revision)
    }

    pub fn draft(&self) -> Option<&Draft> {
        self.session.as_ref().map(Session::draft)
    }

    /// This checkout opens no audio device. The picture still advances.
    pub fn audio_output_open(&self) -> bool {
        false
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
        self.selected = None;
        self.pixel_size = opening_pixel_size(rate);
        self.notice.clear();
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
    }

    pub fn play(&mut self) {
        self.playing = true;
    }

    pub fn pause(&mut self) {
        self.playing = false;
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
        let mut next = self.playhead.0.saturating_add(elapsed);
        if end > 0 && next >= end {
            if self.settings.playback.loop_playback {
                next %= end;
            } else {
                next = end;
                self.playing = false;
            }
        }
        self.playhead = Micros(next);
    }

    pub fn preview_bytes(&self) -> Result<Picture, ShellError> {
        let draft = self.session_draft()?;
        let mut device = SoftwareDevice::new();
        let picture = frame(&Snapshot::new(draft), self.playhead, &mut device)
            .map_err(|error| ShellError::Eval(error.to_string()))?;
        Ok(Picture {
            width: picture.extent.width,
            height: picture.extent.height,
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
        let gesture = MoveGesture {
            segment_id,
            pointer: Pointer { x, y },
            scale: self.scale(),
            playhead: self.playhead,
            snap_threshold: Micros(0),
        };
        let draft = self.session_draft()?;
        let outcome = move_gesture(draft, &gesture)
            .map_err(|_| ShellError::Unavailable("the drag could not be read".to_owned()))?;
        match outcome {
            GestureOutcome::Edit(request) => self.apply(request),
            GestureOutcome::Miss => Err(ShellError::Unavailable(
                "the drag missed the lanes".to_owned(),
            )),
            GestureOutcome::NewLane { .. } => Err(ShellError::Unavailable(
                "a new lane is not available".to_owned(),
            )),
        }
    }

    pub fn zoom_in(&mut self) -> Result<(), ShellError> {
        self.pixel_size = self
            .stored_zoom()?
            .zoom_in(self.zoom_bounds())
            .ok_or(ShellError::Zoom)?
            .pixel_size();
        Ok(())
    }

    pub fn zoom_out(&mut self) -> Result<(), ShellError> {
        self.pixel_size = self
            .stored_zoom()?
            .zoom_out(self.zoom_bounds())
            .ok_or(ShellError::Zoom)?
            .pixel_size();
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
        Ok(())
    }

    pub fn minimum_pixel_size(&self) -> Option<f64> {
        minimum_pixel_size(self.zoom_bounds())
    }

    pub fn select(&mut self, id: Option<SegmentId>) {
        self.selected = id;
    }

    pub fn selected(&self) -> Option<&SegmentId> {
        self.selected.as_ref()
    }

    pub fn inspection(&self) -> Option<Inspection> {
        let draft = self.draft()?;
        let id = self.selected.as_ref()?;
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
        let segment_id = self.selected.clone().ok_or(ShellError::NoProject)?;
        self.refuse_locked(&segment_id)?;
        self.apply(Request::SetVolume { segment_id, volume })
    }

    pub fn set_speed(&mut self, speed: f64) -> Result<(), ShellError> {
        let segment_id = self.selected.clone().ok_or(ShellError::NoProject)?;
        self.refuse_locked(&segment_id)?;
        self.apply(Request::SetConstantSpeed { segment_id, speed })
    }

    pub fn set_text(&mut self, content: String) -> Result<(), ShellError> {
        let segment_id = self.selected.clone().ok_or(ShellError::NoProject)?;
        self.refuse_locked(&segment_id)?;
        self.apply(Request::SetTextContent {
            segment_id,
            content,
        })
    }

    pub fn set_range_start(&mut self, start: Micros) -> Result<(), ShellError> {
        let segment_id = self.selected.clone().ok_or(ShellError::NoProject)?;
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
        let segment_id = self.selected.clone().ok_or(ShellError::NoProject)?;
        self.refuse_locked(&segment_id)?;
        self.apply(Request::Remove { segment_id })
    }

    pub fn split_selected(&mut self) -> Result<(), ShellError> {
        let segment_id = self.selected.clone().ok_or(ShellError::NoProject)?;
        self.refuse_locked(&segment_id)?;
        let inspection = self.inspection().ok_or(ShellError::NoProject)?;
        let at = self.playhead;
        let end = inspection.start.0.saturating_add(inspection.duration.0);
        if at.0 <= inspection.start.0 || at.0 >= end {
            return Err(ShellError::Unavailable("split is not available".to_owned()));
        }
        let right_id = SegmentId::new(self.fresh("seg"));
        self.apply(Request::Split {
            segment_id,
            at,
            right_id,
        })
    }

    pub fn set_notice(&mut self, message: impl Into<String>) {
        self.notice = message.into();
    }

    pub fn unavailable(&mut self, control: &str) {
        self.notice = format!("{control} is not available");
    }

    pub fn export(&mut self, request: ExportRequest) -> Result<(), ShellError> {
        if !LONG_SIDES.contains(&request.long_side) {
            return Err(ShellError::Unavailable(
                "the long side is not one of the steps".to_owned(),
            ));
        }
        if request.path.exists() && !request.replace {
            return Err(ShellError::Overwrite(request.path));
        }
        let (width, height) = self.scaled_canvas(request.long_side);
        let params = CodecParams::video(request.codec, width, height);
        if let Err(error) = self.encoders.open(&params) {
            return Err(ShellError::CodecDeclined(error.to_string()));
        }
        Err(ShellError::ExportUnavailable)
    }

    pub fn drag_timeline(&mut self, delta_y: f32, window_height: f32) {
        let chrome = tokens::HEAD_HEIGHT + tokens::SPLIT * 2.0 + tokens::PREVIEW_MIN_HEIGHT;
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
                let kind = lane_category(meta)
                    .ok_or_else(|| ShellError::Unavailable("the picture has no lane".to_owned()))?;
                let track_id = self.main_track()?;
                let end = self.track_end(&track_id);
                let material_id = self.fresh("mat");
                let segment_id = self.fresh("seg");
                let mut material = MaterialVideo::new(material_id.as_str());
                material.kind = meta.as_str().to_owned();
                material.path = stored;
                material.duration = Micros(duration);
                material.width = width;
                material.height = height;
                let segment = placed(&segment_id, &material_id, end, duration);
                self.apply(Request::Import {
                    track_id,
                    kind,
                    medium: ImportedMedium::Video(Box::new(material)),
                    segment: Box::new(segment),
                })
            }
            Probed::Audio { duration } => {
                let track_id = self.audio_track();
                let end = self.track_end(&track_id);
                let material_id = self.fresh("mat");
                let segment_id = self.fresh("seg");
                let material = MaterialAudio {
                    id: MaterialId::new(&material_id),
                    kind: MetaType::Music.as_str().to_owned(),
                    path: stored,
                    duration: Micros(duration),
                    ..MaterialAudio::default()
                };
                let segment = placed(&segment_id, &material_id, end, duration);
                self.apply(Request::Import {
                    track_id,
                    kind: TrackType::Audio,
                    medium: ImportedMedium::Audio(Box::new(material)),
                    segment: Box::new(segment),
                })
            }
        }
    }

    fn apply(&mut self, request: Request) -> Result<(), ShellError> {
        let session = self.session.as_mut().ok_or(ShellError::NoProject)?;
        let revision = session.revision();
        session
            .apply(revision, &[request])
            .map_err(ShellError::Edit)?;
        Ok(())
    }

    fn session_draft(&self) -> Result<&Draft, ShellError> {
        self.draft().ok_or(ShellError::NoProject)
    }

    fn main_track(&self) -> Result<TrackId, ShellError> {
        self.session_draft()?
            .main_track()
            .map(|track| track.id.clone())
            .ok_or(ShellError::Unavailable(
                "the project has no main lane".to_owned(),
            ))
    }

    fn audio_track(&self) -> TrackId {
        self.draft()
            .and_then(|draft| {
                draft
                    .tracks
                    .iter()
                    .find(|track| track.kind == TrackType::Audio)
                    .map(|track| track.id.clone())
            })
            .unwrap_or_else(|| TrackId::new("audio-1"))
    }

    fn track_end(&self, track_id: &TrackId) -> i64 {
        self.draft()
            .and_then(|draft| draft.tracks.iter().find(|track| &track.id == track_id))
            .and_then(|track| {
                track
                    .segments
                    .iter()
                    .map(|segment| segment.target.end().0)
                    .max()
            })
            .unwrap_or(0)
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

fn placed(segment_id: &str, material_id: &str, start: i64, duration: i64) -> Segment {
    let mut segment = Segment::new(segment_id, material_id, TimeRange::new(start, duration));
    segment.source_range = Some(TimeRange::new(0, duration));
    segment
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

#[cfg(test)]
mod tests {
    use super::{ExportRequest, LONG_SIDES, Shell, ShellError};
    use cutlass_decode::CodecId;
    use cutlass_model::{
        Draft, MaterialColorConfig, Micros, Segment, SegmentId, TimeRange, TrackAttribute,
    };
    use cutlass_session::EditError;
    use cutlass_settings::{SeekDuringPlayback, Settings};
    use cutlass_store::Folder;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

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
    fn import_places_a_still_a_movie_and_a_sound_and_refuses_the_rest() {
        let temp = Temp::new();
        empty_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let picture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/engine/fixtures/pattern.png");
        desktop.import_file(&picture).unwrap();
        let main = desktop.lanes().into_iter().find(|lane| lane.main).unwrap();
        assert_eq!(main.clips.len(), 1);
        assert!(temp.path.join("Resources").join("pattern.png").is_file());
        assert!(
            desktop.draft().unwrap().materials.videos[0]
                .path
                .starts_with("Resources/")
        );

        let sound = temp.path.join("tone.wav");
        write_wav(&sound);
        desktop.import_file(&sound).unwrap();
        assert!(
            desktop
                .lanes()
                .iter()
                .any(|lane| !lane.main && lane.clips.len() == 1)
        );
        let audio = desktop.lanes().into_iter().find(|lane| !lane.main).unwrap();
        desktop
            .toggle_attribute(audio.track_id, TrackAttribute::SOLO)
            .unwrap();
        assert!(
            desktop
                .lanes()
                .iter()
                .find(|lane| lane.main)
                .unwrap()
                .mute_by_solo
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
        for (name, on_main) in samples {
            let path = root.join(name);
            if !path.is_file() {
                eprintln!("skipped: {name} is not in ../local-files");
                continue;
            }
            desktop.import_file(&path).unwrap();
            imported += 1;
            let lane = desktop
                .lanes()
                .into_iter()
                .find(|lane| {
                    lane.main == on_main && lane.clips.iter().any(|clip| clip.name == name)
                })
                .unwrap_or_else(|| panic!("{name}"));
            assert!(lane.clips.iter().any(|clip| clip.name == name));
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
    fn play_and_scrub_read_the_software_frame_without_an_undo_step() {
        let temp = Temp::new();
        color_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let revision = desktop.revision();
        desktop.play();
        assert!(!desktop.audio_output_open());
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
    fn export_asks_before_replacing_and_writes_nothing_when_encode_is_unavailable() {
        let temp = Temp::new();
        empty_project(&temp.path);
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let destination = temp.path.join("out.png");
        let declined = desktop.export(ExportRequest {
            path: destination.clone(),
            long_side: 1080,
            codec: CodecId::Vp8,
            replace: false,
        });
        assert!(matches!(declined, Err(ShellError::CodecDeclined(_))));
        assert!(!destination.exists());

        let unavailable = desktop.export(ExportRequest {
            path: destination.clone(),
            long_side: 1080,
            codec: CodecId::Png,
            replace: false,
        });
        assert!(matches!(unavailable, Err(ShellError::ExportUnavailable)));
        assert!(!destination.exists());

        std::fs::write(&destination, b"keep").unwrap();
        let overwrite = desktop.export(ExportRequest {
            path: destination.clone(),
            long_side: 720,
            codec: CodecId::Png,
            replace: false,
        });
        assert!(matches!(overwrite, Err(ShellError::Overwrite(_))));
        assert_eq!(std::fs::read(&destination).unwrap(), b"keep");
        assert_eq!(LONG_SIDES, [480, 720, 1080, 1440, 2160, 4320]);
    }
}
