//! Cutlass desktop shell: a Slint UI driving the headless [`Engine`].
//!
//! Renders the composited frame at the playhead, mirrors the timeline into the
//! Slint model, and turns user gestures into [`Command`]s applied through the
//! engine.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::cell::RefCell;
use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

use cutlass_commands::{Command, EditCommand, EditOutcome, ProjectCommand};
use cutlass_engine::{ApplyOutcome, Engine, EngineConfig, ExportStats, RgbaFrame, export_project};
use cutlass_models::{Clip, ClipId, ClipSource, Generator, RationalTime, TrackId, TrackKind};
use slint::{
    ComponentHandle, Image, ModelRc, Rgba8Pixel, SharedPixelBuffer, SharedString, Timer, TimerMode,
    VecModel,
};
use tracing::warn;
use tracing_subscriber::EnvFilter;

slint::include_modules!();

const PLAYBACK_TICK: Duration = Duration::from_millis(16);
const IDLE_TICK: Duration = Duration::from_millis(300);

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
    if let Err(e) = run() {
        warn!(error = %e, "cutlass-ui exited with an error");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    cutlass_engine::init();

    let config = EngineConfig {
        cache_dir: PathBuf::from(".cutlass/cache"),
        ..EngineConfig::default()
    };
    let ui = AppWindow::new()?;
    let app = Rc::new(RefCell::new(App::new(config)?));

    bind_callbacks(&ui, &app);

    let play_timer = Timer::default();
    {
        let app = app.clone();
        let weak = ui.as_weak();
        play_timer.start(TimerMode::Repeated, PLAYBACK_TICK, move || {
            if let Some(ui) = weak.upgrade()
                && let Ok(mut app) = app.try_borrow_mut()
            {
                app.tick_playback(&ui);
            }
        });
    }

    let idle_timer = Timer::default();
    {
        let app = app.clone();
        let weak = ui.as_weak();
        idle_timer.start(TimerMode::Repeated, IDLE_TICK, move || {
            if let Some(ui) = weak.upgrade()
                && let Ok(mut app) = app.try_borrow_mut()
            {
                app.idle_tick(&ui);
            }
        });
    }

    if let Some(arg) = std::env::args().nth(1) {
        let path = PathBuf::from(arg);
        if path.is_file() {
            app.borrow_mut().import_path(&ui, path);
        } else {
            warn!(?path, "ignoring CLI argument: file does not exist");
        }
    }

    app.borrow_mut().sync(&ui);
    ui.run()?;
    Ok(())
}

fn bind_callbacks(ui: &AppWindow, app: &Rc<RefCell<App>>) {
    {
        let app = app.clone();
        let weak = ui.as_weak();
        ui.on_import(move || {
            let picked = rfd::FileDialog::new()
                .add_filter("Video", &["mp4", "mov", "mkv", "webm", "m4v", "avi"])
                .set_title("Import video")
                .pick_file();
            if let Some(path) = picked {
                if let Some(ui) = weak.upgrade() {
                    app.borrow_mut().import_path(&ui, path);
                }
            }
        });
    }
    {
        let app = app.clone();
        let weak = ui.as_weak();
        ui.on_export(move || {
            let picked = rfd::FileDialog::new()
                .add_filter("MP4 video", &["mp4"])
                .set_title("Export video")
                .set_file_name("export.mp4")
                .save_file();
            if let Some(path) = picked {
                if let Some(ui) = weak.upgrade() {
                    app.borrow_mut().start_export(&ui, path);
                }
            }
        });
    }
    {
        let app = app.clone();
        let weak = ui.as_weak();
        ui.on_scrub(move |frame| {
            if let Some(ui) = weak.upgrade() {
                app.borrow_mut().scrub(frame, &ui);
            }
        });
    }
    {
        let app = app.clone();
        let weak = ui.as_weak();
        ui.on_toggle_play(move || {
            if let Some(ui) = weak.upgrade() {
                app.borrow_mut().toggle_play(&ui);
            }
        });
    }
    {
        let app = app.clone();
        let weak = ui.as_weak();
        ui.on_select_clip(move |handle| {
            if let Some(ui) = weak.upgrade() {
                app.borrow_mut().select(handle, &ui);
            }
        });
    }
    {
        let app = app.clone();
        let weak = ui.as_weak();
        ui.on_move_clip(move |handle, start| {
            if let Some(ui) = weak.upgrade() {
                app.borrow_mut().move_clip(handle, start, &ui);
            }
        });
    }
    {
        let app = app.clone();
        let weak = ui.as_weak();
        ui.on_split(move || {
            if let Some(ui) = weak.upgrade() {
                app.borrow_mut().split(&ui);
            }
        });
    }
    {
        let app = app.clone();
        let weak = ui.as_weak();
        ui.on_delete_clip(move || {
            if let Some(ui) = weak.upgrade() {
                app.borrow_mut().delete_selected(&ui, false);
            }
        });
    }
    {
        let app = app.clone();
        let weak = ui.as_weak();
        ui.on_ripple_delete(move || {
            if let Some(ui) = weak.upgrade() {
                app.borrow_mut().delete_selected(&ui, true);
            }
        });
    }
    {
        let app = app.clone();
        let weak = ui.as_weak();
        ui.on_do_undo(move || {
            if let Some(ui) = weak.upgrade() {
                app.borrow_mut().undo(&ui);
            }
        });
    }
    {
        let app = app.clone();
        let weak = ui.as_weak();
        ui.on_do_redo(move || {
            if let Some(ui) = weak.upgrade() {
                app.borrow_mut().redo(&ui);
            }
        });
    }
}

struct App {
    engine: Engine,
    playhead: i64,
    playing: bool,
    play_anchor: Option<(Instant, i64)>,
    selected: Option<ClipId>,
    dirty: bool,
    handles: HashMap<i32, ClipId>,
    export_rx: Option<Receiver<ExportMsg>>,
}

enum ExportMsg {
    Done(Result<ExportStats, String>),
}

impl App {
    fn new(config: EngineConfig) -> std::io::Result<Self> {
        Ok(Self {
            engine: Engine::new(config)?,
            playhead: 0,
            playing: false,
            play_anchor: None,
            selected: None,
            dirty: false,
            handles: HashMap::new(),
            export_rx: None,
        })
    }

    fn timeline_rate(&self) -> cutlass_models::Rational {
        self.engine.project().timeline().frame_rate
    }

    fn rt(&self, tick: i64) -> RationalTime {
        RationalTime::new(tick, self.timeline_rate())
    }

    fn duration_ticks(&self) -> i64 {
        self.engine.project().timeline().duration().value
    }

    fn import_path(&mut self, ui: &AppWindow, path: PathBuf) {
        let media = match self.engine.apply(Command::Project(ProjectCommand::Import { path })) {
            Ok(ApplyOutcome::Imported { media }) => media,
            Ok(other) => {
                ui.set_status(format!("Import failed: unexpected {other:?}").into());
                return;
            }
            Err(e) => {
                ui.set_status(format!("Import failed: {e}").into());
                return;
            }
        };

        let track = match self.first_video_track() {
            Ok(t) => t,
            Err(e) => {
                ui.set_status(format!("Could not create track: {e}").into());
                return;
            }
        };

        let (source, start_tick) = {
            let project = self.engine.project();
            let media_src = match project.media(media) {
                Some(m) => m,
                None => {
                    ui.set_status("Import failed: media missing from pool".into());
                    return;
                }
            };
            let start_tick = project
                .timeline()
                .track(track)
                .map(|t| t.content_end())
                .unwrap_or(0);
            (media_src.full_range(), start_tick)
        };

        if let Err(e) = self.engine.apply(Command::Edit(EditCommand::AddClip {
            track,
            media,
            source,
            start: self.rt(start_tick),
        })) {
            ui.set_status(format!("Could not place clip: {e}").into());
            return;
        }

        self.playhead = start_tick;
        self.sync(ui);
    }

    fn first_video_track(&mut self) -> Result<TrackId, cutlass_engine::EngineError> {
        if let Some(track) = self
            .engine
            .project()
            .timeline()
            .tracks_ordered()
            .find(|t| t.kind == TrackKind::Video)
        {
            return Ok(track.id);
        }
        match self.engine.apply(Command::Edit(EditCommand::AddTrack {
            kind: TrackKind::Video,
            name: "V1".into(),
        })) {
            Ok(ApplyOutcome::Edited(EditOutcome::CreatedTrack(id))) => Ok(id),
            Ok(other) => Err(cutlass_engine::EngineError::Export(format!(
                "add track failed: {other:?}"
            ))),
            Err(e) => Err(e),
        }
    }

    fn start_export(&mut self, ui: &AppWindow, path: PathBuf) {
        if self.export_rx.is_some() {
            return;
        }
        if self.duration_ticks() <= 0 {
            ui.set_status("Nothing to export — import a video first".into());
            return;
        }
        if self.playing {
            self.playing = false;
            self.play_anchor = None;
            ui.set_playing(false);
        }

        let project = self.engine.project().clone();
        let color_convert = self.engine.config().color_convert;
        let (tx, rx) = mpsc::channel();
        self.export_rx = Some(rx);

        std::thread::spawn(move || {
            let result = export_project(&project, &path, color_convert).map_err(|e| e.to_string());
            let _ = tx.send(ExportMsg::Done(result));
        });

        ui.set_exporting(true);
        ui.set_export_progress(0.0);
        ui.set_status("Exporting…".into());
    }

    fn poll_export(&mut self, ui: &AppWindow) {
        let Some(rx) = self.export_rx.as_ref() else {
            return;
        };
        let mut finished = None;
        while let Ok(msg) = rx.try_recv() {
            let ExportMsg::Done(result) = msg;
            finished = Some(result);
        }
        if let Some(result) = finished {
            self.export_rx = None;
            ui.set_exporting(false);
            ui.set_export_progress(-1.0);
            match result {
                Ok(stats) => ui.set_status(
                    format!(
                        "Exported {} frames ({}×{})",
                        stats.frames, stats.width, stats.height
                    )
                    .into(),
                ),
                Err(e) => ui.set_status(format!("Export failed: {e}").into()),
            }
        }
    }

    fn scrub(&mut self, frame: i32, ui: &AppWindow) {
        self.playhead = (frame as i64).clamp(0, self.duration_ticks().max(0));
        if self.playing {
            self.play_anchor = Some((Instant::now(), self.playhead));
        }
        ui.set_playhead(self.playhead as i32);
        self.dirty = true;
    }

    fn toggle_play(&mut self, ui: &AppWindow) {
        if self.duration_ticks() <= 0 {
            return;
        }
        self.playing = !self.playing;
        if self.playing {
            self.play_anchor = Some((Instant::now(), self.playhead));
        } else {
            self.play_anchor = None;
        }
        ui.set_playing(self.playing);
    }

    fn tick_playback(&mut self, ui: &AppWindow) {
        if self.playing {
            let Some((started, from_frame)) = self.play_anchor else {
                return;
            };
            let dur = self.duration_ticks();
            let fps = self.timeline_rate().as_f64();
            let elapsed = started.elapsed().as_secs_f64();
            let advanced = (elapsed * fps).floor() as i64;
            let target = from_frame + advanced;

            if target >= dur {
                self.playhead = dur.max(0);
                self.playing = false;
                self.play_anchor = None;
                self.dirty = false;
                ui.set_playing(false);
                ui.set_playhead(self.playhead as i32);
                self.render(ui);
                return;
            }
            if target != self.playhead {
                self.playhead = target;
                ui.set_playhead(self.playhead as i32);
                self.render(ui);
            }
            return;
        }

        if self.dirty {
            self.dirty = false;
            self.render(ui);
        }
    }

    fn select(&mut self, handle: i32, ui: &AppWindow) {
        self.selected = self.handles.get(&handle).copied();
        self.sync(ui);
    }

    fn move_clip(&mut self, handle: i32, new_start: i32, ui: &AppWindow) {
        let Some(clip) = self.handles.get(&handle).copied() else {
            return;
        };
        let Some(to_track) = self.engine.project().timeline().track_of(clip) else {
            return;
        };
        let _ = self.engine.apply(Command::Edit(EditCommand::MoveClip {
            clip,
            to_track,
            start: self.rt(new_start as i64),
        }));
        self.selected = Some(clip);
        self.sync(ui);
    }

    fn split(&mut self, ui: &AppWindow) {
        let Some(clip) = self.selected else {
            return;
        };
        match self.engine.apply(Command::Edit(EditCommand::SplitClip {
            clip,
            at: self.rt(self.playhead),
        })) {
            Ok(_) => self.sync(ui),
            Err(_) => ui.set_status("Move the playhead inside the clip to split".into()),
        }
    }

    fn delete_selected(&mut self, ui: &AppWindow, ripple: bool) {
        let Some(clip) = self.selected else {
            return;
        };
        let cmd = if ripple {
            EditCommand::RippleDelete { clip }
        } else {
            EditCommand::RemoveClip { clip }
        };
        if self.engine.apply(Command::Edit(cmd)).is_ok() {
            self.selected = None;
            self.clamp_playhead();
            self.sync(ui);
        }
    }

    fn undo(&mut self, ui: &AppWindow) {
        if self.engine.undo() {
            self.selected = None;
            self.clamp_playhead();
            self.sync(ui);
        }
    }

    fn redo(&mut self, ui: &AppWindow) {
        if self.engine.redo() {
            self.selected = None;
            self.clamp_playhead();
            self.sync(ui);
        }
    }

    fn idle_tick(&mut self, ui: &AppWindow) {
        self.poll_export(ui);
    }

    fn clamp_playhead(&mut self) {
        self.playhead = self.playhead.clamp(0, self.duration_ticks().max(0));
    }

    fn sync(&mut self, ui: &AppWindow) {
        let order: Vec<TrackId> = self.engine.project().timeline().order().to_vec();

        let mut tracks = Vec::with_capacity(order.len());
        let mut clips = Vec::new();
        self.handles.clear();
        let mut next_handle: i32 = 0;
        let mut selected_handle: i32 = -1;

        for (idx, track_id) in order.iter().enumerate() {
            let Some(track) = self.engine.project().timeline().track(*track_id) else {
                continue;
            };
            tracks.push(TrackData {
                name: SharedString::from(track.name.as_str()),
                video: track.kind == TrackKind::Video,
            });
            for clip in track.clips_ordered() {
                let handle = next_handle;
                next_handle += 1;
                self.handles.insert(handle, clip.id);
                let is_selected = self.selected == Some(clip.id);
                if is_selected {
                    selected_handle = handle;
                }
                clips.push(ClipData {
                    handle,
                    label: SharedString::from(self.clip_label(clip)),
                    start: clip.start().value as i32,
                    duration: clip.timeline.duration.value as i32,
                    track: idx as i32,
                    generated: clip.is_generated(),
                    selected: is_selected,
                });
            }
        }

        ui.set_tracks(ModelRc::new(VecModel::from(tracks)));
        ui.set_clips(ModelRc::new(VecModel::from(clips)));
        ui.set_selected(selected_handle);
        ui.set_duration(self.duration_ticks() as i32);
        ui.set_fps(self.timeline_rate().as_f64() as f32);
        ui.set_playhead(self.playhead as i32);
        ui.set_playing(self.playing);
        ui.set_has_media(self.engine.project().media_count() > 0);
        ui.set_can_undo(self.engine.can_undo());
        ui.set_can_redo(self.engine.can_redo());
        ui.set_proxy_progress(-1.0);

        self.refresh_status(ui);
        self.render(ui);
    }

    fn clip_label(&self, clip: &Clip) -> String {
        match &clip.content {
            ClipSource::Media { media, .. } => self
                .engine
                .project()
                .media(*media)
                .and_then(|m| m.path.file_name())
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "clip".to_string()),
            ClipSource::Generated(Generator::Text { .. }) => "Text".to_string(),
            ClipSource::Generated(Generator::SolidColor { .. }) => "Color".to_string(),
            ClipSource::Generated(Generator::Shape { .. }) => "Shape".to_string(),
            ClipSource::Generated(Generator::Adjustment) => "Adjustment".to_string(),
        }
    }

    fn refresh_status(&self, ui: &AppWindow) {
        let clips = self.engine.project().timeline().clip_count();
        let media = self.engine.project().media_count();
        let status = if media > 0 {
            format!("{media} source(s)  ·  {clips} clip(s)")
        } else {
            "No media imported — click Import to add a video".to_string()
        };
        ui.set_status(SharedString::from(status));
    }

    fn render(&mut self, ui: &AppWindow) {
        match self.engine.get_frame(self.rt(self.playhead)) {
            Ok(frame) => ui.set_preview(to_slint_image(&frame)),
            Err(_) => {}
        }
    }
}

fn to_slint_image(frame: &RgbaFrame) -> Image {
    let mut buffer = SharedPixelBuffer::<Rgba8Pixel>::new(frame.width, frame.height);
    buffer.make_mut_bytes().copy_from_slice(&frame.bytes);
    Image::from_rgba8(buffer)
}
