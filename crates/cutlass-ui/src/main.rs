//! Cutlass desktop shell: a Slint UI driving the headless [`Engine`].
//!
//! This is the front-end half of the editor. It owns an [`Engine`], renders the
//! composited frame at the playhead into the preview, mirrors the timeline into
//! the Slint model, and turns user gestures (scrub, drag, split, delete,
//! undo/redo) into [`EditCommand`]s applied through the engine. All media work —
//! decode, cache, proxies — lives in the engine; the UI only renders and edits.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::cell::RefCell;
use std::collections::HashMap;
use std::error::Error;
use std::path::Path;
use std::rc::Rc;
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

use cutlass_compositor::{CompositeLayer, RgbaImage, composite};
use cutlass_decode::Decoder;
use cutlass_engines::{
    EditCommand, Engine, ExportSettings, ExportStats, ProxyStatus, RenderedContent, RenderedLayer,
};
use cutlass_models::{
    Clip, ClipId, ClipSource, Generator, MediaSource, Rational, TimeRange, TrackId, TrackKind,
};
use slint::{
    ComponentHandle, Image, ModelRc, Rgba8Pixel, SharedPixelBuffer, SharedString, Timer, TimerMode,
    VecModel,
};
use tracing::warn;
use tracing_subscriber::EnvFilter;

slint::include_modules!();

/// Vertical resolution the preview composites at. The compositor scales every
/// source to this canvas, so a modest height keeps scrub/playback cheap while
/// staying sharp enough to edit by.
const PREVIEW_HEIGHT: u32 = 1080;

/// Playback clock tick. Playback advances by wall-clock time, not by tick count,
/// so the actual frame shown is always correct for the timeline rate regardless
/// of this interval; ~60 Hz just keeps motion smooth.
const PLAYBACK_TICK: Duration = Duration::from_millis(16);

/// How often idle work runs: install finished proxies, refresh build progress.
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
    let ui = AppWindow::new()?;
    let app = Rc::new(RefCell::new(App::new()));

    bind_callbacks(&ui, &app);

    // Playback clock: always running, cheap when paused (an early return).
    let play_timer = Timer::default();
    {
        let app = app.clone();
        let weak = ui.as_weak();
        play_timer.start(TimerMode::Repeated, PLAYBACK_TICK, move || {
            // `try_borrow_mut` skips the tick if a UI callback is mid-edit (or a
            // modal dialog is pumping the loop); the next tick catches up.
            if let Some(ui) = weak.upgrade()
                && let Ok(mut app) = app.try_borrow_mut()
            {
                app.tick_playback(&ui);
            }
        });
    }

    // Idle proxy maintenance + progress refresh.
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

    // Optional: `cutlass-ui <video>` opens with that file already imported.
    if let Some(arg) = std::env::args().nth(1) {
        let path = std::path::PathBuf::from(arg);
        if path.exists() {
            app.borrow_mut().import_path(&ui, path);
        } else {
            warn!(?path, "ignoring CLI argument: file does not exist");
        }
    }

    app.borrow_mut().sync(&ui);
    ui.run()?;
    Ok(())
}

/// Wire every Slint callback to the corresponding [`App`] method. Each closure
/// upgrades the window weak-ref and borrows the shared app for the duration of
/// the edit; Slint runs these on the single UI thread, so the borrows never nest.
fn bind_callbacks(ui: &AppWindow, app: &Rc<RefCell<App>>) {
    {
        let app = app.clone();
        let weak = ui.as_weak();
        ui.on_import(move || {
            let Some(ui) = weak.upgrade() else { return };
            // Run the (modal, loop-pumping) file dialog *before* borrowing, so
            // timer ticks that fire while it's open don't hit a held borrow.
            let picked = rfd::FileDialog::new()
                .add_filter("Video", &["mp4", "mov", "mkv", "webm", "m4v", "avi"])
                .set_title("Import video")
                .pick_file();
            if let Some(path) = picked {
                app.borrow_mut().import_path(&ui, path);
            }
        });
    }
    {
        let app = app.clone();
        let weak = ui.as_weak();
        ui.on_export(move || {
            let Some(ui) = weak.upgrade() else { return };
            // Suggest a default name; pick the destination before borrowing so a
            // timer tick during the (modal) dialog doesn't hit a held borrow.
            let picked = rfd::FileDialog::new()
                .add_filter("MP4 video", &["mp4"])
                .set_title("Export video")
                .set_file_name("export.mp4")
                .save_file();
            if let Some(path) = picked {
                app.borrow_mut().start_export(&ui, path);
            }
        });
    }
    {
        let app = app.clone();
        let weak = ui.as_weak();
        ui.on_scrub(move |frame| {
            if let Some(ui) = weak.upgrade() {
                app.borrow_mut().scrub(frame as i64, &ui);
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
                app.borrow_mut().move_clip(handle, start as i64, &ui);
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

/// The UI-side editing session: the engine plus view state (playhead, selection)
/// and the per-sync handle map that ties Slint clip handles to real `ClipId`s.
struct App {
    engine: Engine,
    fps: Rational,
    playhead: i64,
    playing: bool,
    /// `(when playback started, playhead at that moment)`; `None` when paused.
    play_anchor: Option<(Instant, i64)>,
    selected: Option<ClipId>,
    /// Set when the playhead moved but the preview hasn't been re-rendered yet.
    /// The ~60 Hz timer renders the latest position, so a burst of scrub events
    /// collapses to one decode+composite instead of one per event.
    dirty: bool,
    /// Rebuilt every `sync`: Slint `int` handle -> the clip it stands for.
    handles: HashMap<i32, ClipId>,
    /// `Some` while an export runs on a worker thread: progress/result arrive
    /// here and are drained on the idle tick.
    export_rx: Option<Receiver<ExportMsg>>,
}

/// Messages from the background export thread to the UI.
enum ExportMsg {
    Progress { done: i64, total: i64 },
    Done(Result<ExportStats, String>),
}

impl App {
    fn new() -> Self {
        let fps = Rational::FPS_30;
        Self {
            engine: Engine::new("Untitled", fps),
            fps,
            playhead: 0,
            playing: false,
            play_anchor: None,
            selected: None,
            dirty: false,
            handles: HashMap::new(),
            export_rx: None,
        }
    }

    // --- commands ---------------------------------------------------------

    fn import_path(&mut self, ui: &AppWindow, path: std::path::PathBuf) {
        let probe = match probe(&path) {
            Ok(p) => p,
            Err(e) => {
                ui.set_status(format!("Import failed: {e}").into());
                return;
            }
        };

        // First import sets the timeline rate to the source rate, so a
        // single-source edit maps frames 1:1 and plays back at native speed.
        if self.engine.project().media_count() == 0
            && self.engine.project().timeline().clip_count() == 0
        {
            self.fps = probe.frame_rate;
            self.engine = Engine::new("Untitled", self.fps);
        }

        let media = MediaSource::new(
            &path,
            probe.width,
            probe.height,
            probe.frame_rate,
            probe.duration_frames,
            false,
        );
        let media_id = match self.engine.import_media(media) {
            Ok(id) => id,
            Err(e) => {
                ui.set_status(format!("Import failed: {e}").into());
                return;
            }
        };

        let track = self.first_video_track_or_create();
        // Append after existing content on the track so the placement is legal.
        let start = self
            .engine
            .project()
            .timeline()
            .track(track)
            .map(|t| t.content_end())
            .unwrap_or(0);
        if let Err(e) = self.engine.apply(EditCommand::AddClip {
            track,
            media: media_id,
            source: TimeRange::new(0, probe.duration_frames),
            start,
        }) {
            ui.set_status(format!("Could not place clip: {e}").into());
            return;
        }

        self.playhead = start;
        tracing::info!(
            width = probe.width,
            height = probe.height,
            fps = probe.frame_rate.as_f64(),
            duration_frames = probe.duration_frames,
            timeline_frames = self.engine.duration(),
            "imported media and placed clip"
        );
        self.sync(ui);
    }

    /// Kick off an export on a background thread.
    ///
    /// The thread gets a *clone* of the project and opens its own source
    /// decoders ([`Engine::for_export`]), so the live session keeps running and
    /// the UI never blocks. Progress and the final result stream back over a
    /// channel drained in [`poll_export`]. While it runs, background proxy
    /// builds in the live engine are paused so the export gets the CPU.
    fn start_export(&mut self, ui: &AppWindow, path: std::path::PathBuf) {
        if self.export_rx.is_some() {
            return; // an export is already running.
        }
        if self.engine.duration() <= 0 {
            ui.set_status("Nothing to export — import a video first".into());
            return;
        }
        if self.playing {
            self.playing = false;
            self.play_anchor = None;
            ui.set_playing(false);
        }

        let (w, h) = self.export_dimensions();
        let settings = ExportSettings::new(w, h);
        let project = self.engine.project().clone();
        let (tx, rx) = mpsc::channel();
        self.export_rx = Some(rx);
        self.engine.set_background_paused(true);

        std::thread::spawn(move || {
            let result = (|| -> Result<ExportStats, String> {
                let mut engine = Engine::for_export(project).map_err(|e| e.to_string())?;
                let tx = tx.clone();
                let mut on_progress = |done: i64, total: i64| {
                    let _ = tx.send(ExportMsg::Progress { done, total });
                };
                engine
                    .export_with(&path, settings, Some(&mut on_progress))
                    .map_err(|e| e.to_string())
            })();
            // The receiver may be gone if the window closed; ignore send errors.
            let _ = tx.send(ExportMsg::Done(result));
        });

        ui.set_exporting(true);
        ui.set_export_progress(0.0);
        ui.set_status(format!("Exporting {w}×{h}…").into());
    }

    /// Drain export progress/result from the worker thread (called on idle).
    fn poll_export(&mut self, ui: &AppWindow) {
        let Some(rx) = self.export_rx.as_ref() else {
            return;
        };
        let mut finished: Option<Result<ExportStats, String>> = None;
        while let Ok(msg) = rx.try_recv() {
            match msg {
                ExportMsg::Progress { done, total } => {
                    let frac = if total > 0 { done as f32 / total as f32 } else { 0.0 };
                    ui.set_export_progress(frac);
                    ui.set_status(format!("Exporting… {:.0}%", frac * 100.0).into());
                }
                ExportMsg::Done(result) => finished = Some(result),
            }
        }

        if let Some(result) = finished {
            self.export_rx = None;
            self.engine.set_background_paused(false);
            ui.set_exporting(false);
            ui.set_export_progress(-1.0);
            match result {
                Ok(stats) => ui.set_status(
                    format!("Exported {} frames ({}×{})", stats.frames, stats.width, stats.height)
                        .into(),
                ),
                Err(e) => ui.set_status(format!("Export failed: {e}").into()),
            }
        }
    }

    /// Output resolution for an export: the largest imported source dimensions
    /// (rounded to even for H.264), or 1080p when the timeline is generators-only.
    fn export_dimensions(&self) -> (u32, u32) {
        let dims = self
            .engine
            .project()
            .media_iter()
            .filter(|m| m.width > 0 && m.height > 0)
            .map(|m| (m.width, m.height))
            .max_by_key(|(w, h)| (*w as u64) * (*h as u64));
        match dims {
            Some((w, h)) => (w & !1, h & !1),
            None => (1920, 1080),
        }
    }

    fn scrub(&mut self, frame: i64, ui: &AppWindow) {
        self.playhead = frame.clamp(0, self.engine.duration().max(0));
        // Keep the playback clock anchored to the new position if mid-play.
        if self.playing {
            self.play_anchor = Some((Instant::now(), self.playhead));
        }
        // Nudge the proxy queue toward whatever is under the playhead.
        self.engine.set_playhead(self.playhead);
        // Move the playhead line now (cheap), but defer the decode+composite to
        // the next timer tick so a fast drag doesn't render every intermediate
        // frame — only the latest one the user lands on.
        ui.set_playhead(self.playhead as i32);
        self.dirty = true;
    }

    fn toggle_play(&mut self, ui: &AppWindow) {
        if self.engine.duration() <= 0 {
            return;
        }
        self.playing = !self.playing;
        if self.playing {
            self.play_anchor = Some((Instant::now(), self.playhead));
            // Don't let background transcodes steal cycles from live playback.
            self.engine.set_background_paused(true);
        } else {
            self.play_anchor = None;
            self.engine.set_background_paused(false);
        }
        ui.set_playing(self.playing);
    }

    /// The ~60 Hz UI tick. While playing, advance the playhead to match elapsed
    /// wall-clock time; while paused, render a pending scrub at most once per
    /// tick. Either way the decode+composite runs ≤ once per tick, which is what
    /// caps CPU during a fast drag (many scrub events → one render here).
    fn tick_playback(&mut self, ui: &AppWindow) {
        if self.playing {
            let Some((started, from_frame)) = self.play_anchor else {
                return;
            };
            let dur = self.engine.duration();
            let elapsed = started.elapsed().as_secs_f64();
            let advanced = (elapsed * self.fps.as_f64()).floor() as i64;
            let target = from_frame + advanced;

            if target >= dur {
                self.playhead = dur.max(0);
                self.playing = false;
                self.play_anchor = None;
                self.dirty = false;
                self.engine.set_background_paused(false);
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

        // Paused: coalesce scrub renders to this tick.
        if self.dirty {
            self.dirty = false;
            self.render(ui);
        }
    }

    fn select(&mut self, handle: i32, ui: &AppWindow) {
        self.selected = self.handles.get(&handle).copied();
        self.sync(ui);
    }

    fn move_clip(&mut self, handle: i32, new_start: i64, ui: &AppWindow) {
        let Some(clip) = self.handles.get(&handle).copied() else {
            return;
        };
        let Some(to_track) = self.engine.project().timeline().track_of(clip) else {
            return;
        };
        // A rejected move (overlap / negative) is a no-op; sync snaps it back.
        let _ = self.engine.apply(EditCommand::MoveClip {
            clip,
            to_track,
            start: new_start.max(0),
        });
        self.selected = Some(clip);
        self.sync(ui);
    }

    fn split(&mut self, ui: &AppWindow) {
        let Some(clip) = self.selected else { return };
        match self.engine.apply(EditCommand::SplitClip {
            clip,
            at: self.playhead,
        }) {
            Ok(_) => self.sync(ui),
            Err(_) => ui.set_status("Move the playhead inside the clip to split".into()),
        }
    }

    fn delete_selected(&mut self, ui: &AppWindow, ripple: bool) {
        let Some(clip) = self.selected else { return };
        let cmd = if ripple {
            EditCommand::RippleDelete { clip }
        } else {
            EditCommand::RemoveClip { clip }
        };
        if self.engine.apply(cmd).is_ok() {
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
        // An in-flight export owns the status line and the CPU; just pump its
        // progress and skip proxy maintenance / re-rendering until it's done.
        if self.export_rx.is_some() {
            self.poll_export(ui);
            return;
        }
        self.engine.poll_proxies();
        self.refresh_status(ui);
        // When idle, re-render so a freshly installed proxy frame replaces the
        // source-decoded one under the playhead.
        if !self.playing && self.engine.project().media_count() > 0 {
            self.render(ui);
        }
    }

    // --- helpers ----------------------------------------------------------

    fn first_video_track_or_create(&mut self) -> TrackId {
        let existing = self
            .engine
            .project()
            .timeline()
            .tracks_ordered()
            .find(|t| t.kind == TrackKind::Video)
            .map(|t| t.id);
        existing.unwrap_or_else(|| self.engine.project_mut().add_track(TrackKind::Video, "V1"))
    }

    fn clamp_playhead(&mut self) {
        self.playhead = self.playhead.clamp(0, self.engine.duration().max(0));
    }

    /// Rebuild the Slint model from engine state and re-render the preview.
    fn sync(&mut self, ui: &AppWindow) {
        let order: Vec<TrackId> = self.engine.project().timeline().order().to_vec();

        let mut tracks: Vec<TrackData> = Vec::with_capacity(order.len());
        let mut clips: Vec<ClipData> = Vec::new();
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
                    start: clip.start() as i32,
                    duration: clip.timeline.duration as i32,
                    track: idx as i32,
                    generated: clip.is_generated(),
                    selected: is_selected,
                });
            }
        }

        ui.set_tracks(ModelRc::new(VecModel::from(tracks)));
        ui.set_clips(ModelRc::new(VecModel::from(clips)));
        ui.set_selected(selected_handle);
        ui.set_duration(self.engine.duration() as i32);
        ui.set_fps(self.fps.as_f64() as f32);
        ui.set_playhead(self.playhead as i32);
        ui.set_playing(self.playing);
        ui.set_has_media(self.engine.project().media_count() > 0);
        ui.set_can_undo(self.engine.can_undo());
        ui.set_can_redo(self.engine.can_redo());

        self.refresh_status(ui);
        self.render(ui);
    }

    fn clip_label(&self, clip: &Clip) -> String {
        match &clip.content {
            ClipSource::Media { media, .. } => self
                .engine
                .project()
                .media(*media)
                .and_then(|m| m.path().file_name())
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "clip".to_string()),
            ClipSource::Generated(Generator::Text { .. }) => "Text".to_string(),
            ClipSource::Generated(Generator::SolidColor { .. }) => "Color".to_string(),
            ClipSource::Generated(Generator::Shape { .. }) => "Shape".to_string(),
            ClipSource::Generated(Generator::Adjustment) => "Adjustment".to_string(),
        }
    }

    fn refresh_status(&self, ui: &AppWindow) {
        let mut total = 0usize;
        let mut ready = 0usize;
        let mut building: Option<f32> = None;
        for media in self.engine.project().media_iter() {
            total += 1;
            match self.engine.proxy_status(media.id) {
                Some(ProxyStatus::Ready(_)) => ready += 1,
                Some(ProxyStatus::Building { progress }) => {
                    building = Some(building.map_or(*progress, |b| b.min(*progress)));
                }
                _ => {}
            }
        }

        let clips = self.engine.project().timeline().clip_count();
        let status = match building {
            Some(p) => format!("Building proxies… {:.0}%  ·  {clips} clips", p * 100.0),
            None if total > 0 => format!("{ready}/{total} proxies ready  ·  {clips} clips"),
            None => "No media imported — click Import to add a video".to_string(),
        };
        ui.set_status(SharedString::from(status));
        ui.set_proxy_progress(building.unwrap_or(-1.0));
    }

    /// Composite the layer stack at the playhead and push it to the preview.
    fn render(&mut self, ui: &AppWindow) {
        let layers = self.engine.frame_at(self.playhead).unwrap_or_default();
        if layers.is_empty() {
            return;
        }
        let (cw, ch) = pick_canvas(&layers, PREVIEW_HEIGHT);
        let composite_layers = to_composite_layers(&layers);
        let image = composite(cw, ch, &composite_layers);
        ui.set_preview(to_slint_image(&image));
    }
}

/// Probed source facts needed to register media with the engine.
struct Probe {
    width: u32,
    height: u32,
    frame_rate: Rational,
    duration_frames: i64,
}

fn probe(path: &Path) -> Result<Probe, Box<dyn Error>> {
    let decoder = Decoder::open(path)?;
    let info = decoder.info();
    let (num, den) = info.frame_rate_parts();
    let frame_rate = Rational::new(num, den);
    if !frame_rate.is_valid() {
        return Err("source has an invalid frame rate".into());
    }
    let duration_frames = decoder
        .duration()
        .map(|d| (d.as_secs_f64() * frame_rate.as_f64()).round() as i64)
        .filter(|&n| n > 0)
        .unwrap_or(1_000_000);
    Ok(Probe {
        width: info.width,
        height: info.height,
        frame_rate,
        duration_frames,
    })
}

/// Map the engine's resolved layers onto the compositor's layer type. Media
/// frames become sampled layers; solid generators become fills; everything else
/// the CPU compositor can't draw yet is skipped.
fn to_composite_layers(layers: &[RenderedLayer]) -> Vec<CompositeLayer<'_>> {
    let mut out = Vec::with_capacity(layers.len());
    for layer in layers {
        match &layer.content {
            RenderedContent::Media(frame) => out.push(CompositeLayer::Frame(frame.as_ref())),
            RenderedContent::Generated(Generator::SolidColor { rgba }) => {
                out.push(CompositeLayer::Solid(*rgba))
            }
            RenderedContent::Generated(_) => {}
        }
    }
    out
}

/// Choose a preview canvas: `target_h` tall, aspect taken from the topmost media
/// frame in the stack (falling back to 16:9 for generated-only frames).
fn pick_canvas(layers: &[RenderedLayer], target_h: u32) -> (u32, u32) {
    let dims = layers.iter().rev().find_map(|l| match &l.content {
        RenderedContent::Media(f) if f.width > 0 && f.height > 0 => Some((f.width, f.height)),
        _ => None,
    });
    let h = target_h.max(2);
    match dims {
        Some((sw, sh)) => {
            let mut w = ((sw as u64 * h as u64) / sh as u64) as u32;
            if w % 2 == 1 {
                w += 1;
            }
            (w.max(2), h)
        }
        None => (h * 16 / 9, h),
    }
}

fn to_slint_image(image: &RgbaImage) -> Image {
    let mut buffer = SharedPixelBuffer::<Rgba8Pixel>::new(image.width, image.height);
    buffer.make_mut_bytes().copy_from_slice(&image.pixels);
    Image::from_rgba8(buffer)
}
