//! The desktop editor. It holds one project and draws the window in `docs/desktop-layout.md`.

#![deny(unsafe_code)]

use std::path::PathBuf;

use eframe::egui::{self, Align2, Button, Rect, RichText, Sense, TextStyle, Vec2, pos2};

use cutlass_project::{History, Project, SegmentId};

mod app_settings;
mod chrome;
mod export;
mod frames;
mod gaps;
mod hear;
mod icons;
mod import;
mod inspector;
mod keys;
mod library;
mod logs;
mod media_bin;
mod menu;
mod modify;
mod preview;
#[cfg(target_vendor = "apple")]
mod preview_file;
mod projects;
mod ruler;
mod theme;
mod timeline;
mod tip;
mod words;

fn main() -> eframe::Result {
    logs::open();
    let viewport = egui::ViewportBuilder::default()
        .with_title("Cutlass")
        .with_app_id("cutlass")
        .with_inner_size([1280.0, 800.0])
        .with_min_inner_size([1100.0, 700.0])
        .with_fullsize_content_view(true)
        .with_titlebar_shown(false)
        .with_title_shown(false);
    let mut native = eframe::NativeOptions {
        viewport,
        renderer: eframe::Renderer::Wgpu,
        // The recent list is the only thing stored. The window is not restored.
        persist_window: false,
        ..Default::default()
    };
    request_preview_features(&mut native);

    eframe::run_native(
        "Cutlass",
        native,
        Box::new(|cc| Ok(Box::new(CutlassApp::new(cc)))),
    )
}

/// A 10-bit frame is P010. The window device has to ask before it can sample one.
fn request_preview_features(native: &mut eframe::NativeOptions) {
    let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &mut native.wgpu_options.wgpu_setup else {
        return;
    };
    let previous = std::sync::Arc::clone(&setup.device_descriptor);
    setup.device_descriptor = std::sync::Arc::new(move |adapter| {
        let mut descriptor = previous(adapter);
        descriptor.required_features |= cutlass_compose::sampling_features(adapter.features());
        descriptor
    });
}

struct CutlassApp {
    history: History,
    project: Project,
    path: Option<PathBuf>,
    checkpoint: String,
    recent: Vec<PathBuf>,
    pending: Option<projects::Replace>,
    draft_slot: String,
    draft_text: Option<String>,
    recovery: Option<projects::Recovery>,
    recovery_shown: bool,
    playhead_us: i64,
    play: Option<cutlass_engine::Run>,
    hear: hear::Hear,
    reel: frames::Reel,
    shelf: library::Shelf,
    media: library::MediaSource,
    bin: media_bin::Bin,
    selected_segments: Vec<SegmentId>,
    clip_drag: Option<timeline::ClipDrag>,
    trim_drag: Option<timeline::TrimDrag>,
    split: bool,
    export: Option<export::Sheet>,
    modify: Option<modify::Draft>,
    app: app_settings::AppSettings,
    app_dialog: Option<app_settings::Draft>,
    language: words::Language,
    panes: gaps::Panes,
}

impl CutlassApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        theme::install(&cc.egui_ctx);
        icons::install(&cc.egui_ctx);
        let loaded = projects::blank();
        logs::project(loaded.project.name());
        let gpu = cc.wgpu_render_state.as_ref().map(|state| {
            cutlass_compose::Gpu::with_device(state.device.clone(), state.queue.clone())
        });
        if gpu.is_none() {
            log::warn!("The window has no GPU. The preview stays the background plate.");
        }
        let ours = std::process::id();
        let recovery = projects::draft_dir()
            .and_then(|dir| projects::find_recovery(&dir, ours, projects::process_running));
        Self {
            history: History::new(),
            project: loaded.project,
            path: None,
            checkpoint: loaded.checkpoint,
            recent: projects::load_recent(cc.storage),
            pending: None,
            draft_slot: ours.to_string(),
            draft_text: None,
            recovery,
            recovery_shown: false,
            playhead_us: 0,
            play: None,
            hear: hear::Hear::new(),
            reel: frames::Reel::new(gpu),
            shelf: library::Shelf::Media,
            media: library::MediaSource::Project,
            bin: media_bin::Bin::new(),
            selected_segments: Vec::new(),
            clip_drag: None,
            trim_drag: None,
            split: false,
            export: None,
            modify: None,
            app: app_settings::AppSettings::default(),
            app_dialog: None,
            language: words::language_from_tag(&words::system_tag()),
            panes: gaps::Panes::new(),
        }
    }

    fn delete_pressed(&mut self, ui: &egui::Ui) {
        let typing = ui.memory(|memory| memory.focused().is_some());
        let pressed = ui.input(|input| {
            input.key_pressed(egui::Key::Delete) || input.key_pressed(egui::Key::Backspace)
        });
        if typing || !pressed {
            return;
        }
        self.remove_selection(ui);
    }

    fn remove_selection(&mut self, ui: &egui::Ui) {
        let lanes = timeline::lanes_of(&self.project);
        let had = !self.selected_segments.is_empty();
        let removed = timeline::delete_selected(
            &mut self.history,
            &mut self.project,
            &mut self.selected_segments,
            &lanes,
        );
        if had && !removed {
            let text = if self
                .selected_segments
                .iter()
                .all(|id| timeline::edit_refused(&self.project, &lanes, *id))
            {
                words::of(ui, words::Phrase::TrackIsLocked)
            } else {
                words::of(ui, words::Phrase::EditMissed)
            };
            timeline::show_note(ui, text);
        }
    }

    fn history_shortcut(&mut self, ui: &egui::Ui) {
        let Some(step) = keys::history(ui) else {
            return;
        };
        self.step_history(ui, step == keys::Shortcut::Undo);
    }

    fn step_history(&mut self, ui: &egui::Ui, undo: bool) {
        let result = if undo {
            self.history.undo(&mut self.project)
        } else {
            self.history.redo(&mut self.project)
        };
        if result.is_err() {
            timeline::show_note(
                ui,
                words::of(
                    ui,
                    if undo {
                        words::Phrase::NothingToUndo
                    } else {
                        words::Phrase::NothingToRedo
                    },
                ),
            );
        }
    }

    fn relink(&mut self, ui: &egui::Ui, id: cutlass_project::AssetId) {
        let Some(path) = import::pick_one() else {
            return;
        };
        let replacement = match import::from_path(&path) {
            Ok(asset) => asset,
            Err(import::ImportError::Empty) => {
                timeline::show_note(ui, words::of(ui, words::Phrase::ImportEmpty));
                return;
            }
            Err(import::ImportError::Unread) => {
                timeline::show_note(ui, words::of(ui, words::Phrase::ImportUnread));
                return;
            }
        };
        let Some(current) = self.project.asset(id) else {
            timeline::show_note(ui, words::of(ui, words::Phrase::EditMissed));
            return;
        };
        if !import::same_kind(current.kind(), replacement.kind()) {
            timeline::show_note(ui, words::of(ui, words::Phrase::RelinkKind));
            return;
        }
        if current.source() == replacement.source() {
            return;
        }
        let source = replacement.source().to_owned();
        if let Err(error) = self.history.apply(
            &mut self.project,
            &[cutlass_project::Command::SetSource { asset: id, source }],
        ) {
            timeline::show_note(ui, timeline::apply_note(&error, words::active(ui)));
        }
    }

    fn import_chosen(&mut self, ui: &egui::Ui) {
        self.add_imports(ui, &import::pick());
    }

    fn add_imports(&mut self, ui: &egui::Ui, paths: &[std::path::PathBuf]) {
        let (assets, failed) = import::chosen(paths);
        if !assets.is_empty() {
            let commands: Vec<_> = assets
                .into_iter()
                .map(cutlass_project::Command::AddAsset)
                .collect();
            match self.history.apply(&mut self.project, &commands) {
                Ok(outcomes) => {
                    let folder = self.bin.open();
                    for outcome in outcomes {
                        if let cutlass_project::Outcome::Asset(id) = outcome {
                            self.bin.move_file(id, folder);
                        }
                    }
                }
                Err(error) => {
                    timeline::show_note(ui, timeline::apply_note(&error, words::active(ui)));
                    return;
                }
            }
        }
        let Some(failed) = failed else {
            return;
        };
        let phrase = match failed {
            import::ImportError::Unread => words::Phrase::ImportUnread,
            import::ImportError::Empty => words::Phrase::ImportEmpty,
        };
        timeline::show_note(ui, words::of(ui, phrase));
    }

    fn blocked(&self) -> bool {
        self.export.is_some()
            || self.modify.is_some()
            || self.app_dialog.is_some()
            || self.pending.is_some()
            || self.recovery.is_some()
    }

    fn unsaved(&self) -> bool {
        projects::changed(&self.project, &self.checkpoint)
    }

    /// `true` when the unsaved dialog opened on this frame.
    fn handle_file(&mut self, ui: &egui::Ui, ask: Option<projects::Ask>) -> bool {
        let Some(ask) = ask else {
            return false;
        };
        if self.blocked() {
            return false;
        }
        let replacing = match &ask {
            projects::Ask::Save => {
                self.save_current(ui);
                return false;
            }
            projects::Ask::SaveAs => {
                self.save_as(ui);
                return false;
            }
            projects::Ask::New => projects::Replace::New,
            projects::Ask::Open => projects::Replace::Open,
            projects::Ask::Recent(path) => projects::Replace::Recent(path.clone()),
        };
        self.begin_replace(ui, &ask, replacing)
    }

    fn begin_replace(
        &mut self,
        ui: &egui::Ui,
        ask: &projects::Ask,
        next: projects::Replace,
    ) -> bool {
        if projects::needs_guard(self.unsaved(), ask) {
            self.pending = Some(next);
            true
        } else {
            self.replace_now(ui, next);
            false
        }
    }

    fn finish_guard(&mut self, ui: &egui::Ui, choice: Option<projects::GuardChoice>) {
        let Some(choice) = choice else {
            return;
        };
        let Some(next) = self.pending.clone() else {
            return;
        };
        match choice {
            projects::GuardChoice::Cancel => self.pending = None,
            projects::GuardChoice::Discard => {
                self.pending = None;
                self.replace_now(ui, next);
            }
            projects::GuardChoice::Save => {
                if self.save_current(ui) {
                    self.pending = None;
                    self.replace_now(ui, next);
                }
            }
        }
    }

    fn replace_now(&mut self, ui: &egui::Ui, next: projects::Replace) {
        match next {
            projects::Replace::New => self.install(projects::blank(), None),
            projects::Replace::Open => {
                if let Some(path) = projects::pick_open() {
                    self.open_path(ui, path);
                }
            }
            projects::Replace::Recent(path) => self.open_path(ui, path),
        }
    }

    fn open_path(&mut self, ui: &egui::Ui, path: PathBuf) {
        match projects::read(&path) {
            Ok(loaded) => {
                projects::remember(&mut self.recent, &path);
                self.install(loaded, Some(path));
            }
            Err(projects::ReadError::Missing) => {
                projects::forget(&mut self.recent, &path);
                timeline::show_note(ui, words::of(ui, words::Phrase::ProjectUnread));
            }
            Err(projects::ReadError::Unread) => {
                timeline::show_note(ui, words::of(ui, words::Phrase::ProjectUnread));
            }
        }
    }

    fn save_current(&mut self, ui: &egui::Ui) -> bool {
        match self.path.clone() {
            Some(path) => self.write_path(ui, path),
            None => self.save_as(ui),
        }
    }

    fn save_as(&mut self, ui: &egui::Ui) -> bool {
        let Some(path) = projects::pick_save(self.project.name()) else {
            return false;
        };
        self.write_path(ui, path)
    }

    fn write_path(&mut self, ui: &egui::Ui, path: PathBuf) -> bool {
        match projects::write(&path, &self.project) {
            Ok(checkpoint) => {
                self.checkpoint = checkpoint;
                projects::remember(&mut self.recent, &path);
                self.path = Some(path);
                self.clear_draft();
                true
            }
            Err(()) => {
                timeline::show_note(ui, words::of(ui, words::Phrase::ProjectUnsaved));
                false
            }
        }
    }

    /// The undo stack belongs to the cut that was open. A new cut starts empty.
    fn install(&mut self, loaded: projects::Loaded, path: Option<PathBuf>) {
        self.history = History::new();
        self.project = loaded.project;
        self.checkpoint = loaded.checkpoint;
        self.path = path;
        self.pending = None;
        self.playhead_us = 0;
        self.play = None;
        self.hear.stop();
        logs::project(self.project.name());
        self.selected_segments.clear();
        self.clip_drag = None;
        self.trim_drag = None;
        self.split = false;
        self.bin = media_bin::Bin::new();
        self.export = None;
        self.modify = None;
        self.app_dialog = None;
        self.reel.clear();
        self.clear_draft();
    }

    fn finish_recovery(&mut self, ui: &egui::Ui, choice: Option<projects::RecoveryChoice>) {
        let Some(choice) = choice else {
            return;
        };
        let Some(found) = self.recovery.clone() else {
            return;
        };
        match choice {
            projects::RecoveryChoice::Discard => {
                if let Some(dir) = projects::draft_dir() {
                    projects::remove_draft(&dir, &found.slot);
                }
                self.recovery = None;
            }
            projects::RecoveryChoice::Restore => {
                if self.restore_cut(ui, &found) {
                    self.recovery = None;
                }
            }
        }
    }

    /// The user's file stays the last Save. The old slot is removed only after
    /// this process has its own copy.
    fn restore_cut(&mut self, ui: &egui::Ui, found: &projects::Recovery) -> bool {
        let Ok(loaded) = projects::read(&found.cutlass) else {
            timeline::show_note(ui, words::of(ui, words::Phrase::ProjectUnread));
            return false;
        };
        let checkpoint = projects::saved_checkpoint(found.source.as_deref());
        let source = found.source.clone();
        self.install(loaded, source);
        self.checkpoint = checkpoint;
        let Some(dir) = projects::draft_dir() else {
            return true;
        };
        let Ok(text) = self.project.to_json() else {
            return true;
        };
        let name = self.project.name().to_owned();
        if projects::write_draft(&dir, &self.draft_slot, &text, &name, self.path.as_deref()).is_ok()
        {
            self.draft_text = Some(text);
            projects::remove_draft(&dir, &found.slot);
        }
        true
    }

    /// A changed cut is copied beside the app. The user's file is not written.
    fn flush_draft(&mut self) {
        let Ok(text) = self.project.to_json() else {
            return;
        };
        if text == self.checkpoint {
            if self.draft_text.is_some() {
                self.clear_draft();
            }
            return;
        }
        if !projects::draft_needed(&self.checkpoint, &text, self.draft_text.as_deref()) {
            return;
        }
        let Some(dir) = projects::draft_dir() else {
            return;
        };
        let name = self.project.name().to_owned();
        let source = self.path.clone();
        if projects::write_draft(&dir, &self.draft_slot, &text, &name, source.as_deref()).is_ok() {
            self.draft_text = Some(text);
        }
    }

    fn clear_draft(&mut self) {
        if let Some(dir) = projects::draft_dir() {
            projects::remove_draft(&dir, &self.draft_slot);
        }
        self.draft_text = None;
    }

    fn begin_export(&mut self) {
        if self.project.duration().value <= 0 {
            log::warn!("The cut is empty. Nothing was exported.");
            self.export = None;
            return;
        }
        let name = self
            .export
            .as_ref()
            .map(|sheet| sheet.name().to_owned())
            .unwrap_or_default();
        let Some(path) = export::pick_path(&name) else {
            return;
        };
        let project = self.project.clone();
        let started = std::thread::Builder::new()
            .name("cutlass-export".to_owned())
            .spawn(move || match cutlass_export::write(&project, &path) {
                Ok(()) => log::info!("Exported {}.", path.display()),
                Err(error) => log::warn!("Export failed ({}): {error}.", path.display()),
            });
        if started.is_err() {
            log::warn!("Export did not start.");
            return;
        }
        self.export = None;
    }

    fn toggle_play(&mut self) {
        let rate = self.project.settings().rate().rate();
        let duration = self.project.duration().value;
        self.playhead_us = cutlass_engine::toggle(&mut self.play, self.playhead_us, duration, rate);
        if self.play.is_some() {
            self.hear.start(&self.project, self.playhead_us);
        } else {
            self.hear.stop();
        }
    }

    fn advance_play(&mut self, ctx: &egui::Context) {
        let Some((origin, elapsed, rate, duration)) = self.play.as_ref().map(|run| {
            (
                run.origin_us(),
                run.elapsed(),
                self.project.settings().rate().rate(),
                self.project.duration().value,
            )
        }) else {
            return;
        };
        match self.hear.follow(&self.project, duration, rate) {
            cutlass_engine::Lead::Silent => {
                match cutlass_engine::advance(origin, elapsed, duration, rate) {
                    cutlass_engine::Motion::Running(us) => {
                        self.playhead_us = us;
                        ctx.request_repaint_after(cutlass_engine::frame_wait(rate));
                    }
                    cutlass_engine::Motion::Finished(us) => {
                        self.playhead_us = us;
                        self.play = None;
                    }
                    cutlass_engine::Motion::Idle => self.play = None,
                }
            }
            cutlass_engine::Lead::Waiting => {
                ctx.request_repaint();
            }
            cutlass_engine::Lead::Running(us) => {
                self.playhead_us = us;
                ctx.request_repaint_after(cutlass_engine::frame_wait(rate));
            }
            cutlass_engine::Lead::Finished(us) => {
                self.playhead_us = us;
                self.play = None;
                self.hear.stop();
            }
        }
    }
}

impl eframe::App for CutlassApp {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        theme::GAP.to_normalized_gamma_f32()
    }

    /// The recent paths, and a copy of a changed cut. This does not write the user's file.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        projects::store_recent(storage, &self.recent);
        self.flush_draft();
    }

    /// How often that copy is made. The recent list is stored at the same time.
    fn auto_save_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs(30)
    }

    /// The cloud key stays in memory. Widget state is not written.
    fn persist_egui_memory(&self) -> bool {
        false
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        logs::project(self.project.name());
        words::install(ui.ctx(), self.language);
        chrome::pass_clicks(frame);
        let key_play = !self.blocked() && keys::play(ui);
        if self.blocked() {
            if self.play.is_some() {
                self.hear.stop();
            }
            self.play = None;
        }
        self.advance_play(ui.ctx());
        let driven = self.playhead_us;
        let dialog = self.blocked();
        keys::set_blocked(ui, dialog);
        if !dialog {
            let dropped = import::dropped(ui);
            if !dropped.is_empty() {
                self.add_imports(ui, &dropped);
            }
            self.history_shortcut(ui);
            self.delete_pressed(ui);
        }

        let mut bar = Bar {
            export: false,
            settings: false,
            file: None,
        };
        egui::Panel::top("head")
            .exact_size(theme::HEADER_HEIGHT)
            .frame(theme::bar())
            .show_separator_line(false)
            .show(ui, |ui| {
                bar = header(ui, self.project.name(), &self.recent);
            });
        let opened = bar.export && !self.blocked();
        if opened {
            self.export = Some(export::Sheet::open(
                self.project.name(),
                self.project.settings().rate().rate(),
            ));
        }
        let opened_guard = self.handle_file(ui, bar.file);

        let body = ui.available_rect_before_wrap();
        gaps::drag(ui, &mut self.panes, body);

        egui::Panel::bottom("timeline")
            .resizable(false)
            .exact_size(self.panes.timeline())
            .frame(theme::gap_above())
            .show_separator_line(false)
            .show(ui, |ui| {
                let commands = timeline::show(
                    ui,
                    &self.project,
                    &mut self.playhead_us,
                    &mut self.selected_segments,
                    &mut self.clip_drag,
                    &mut self.trim_drag,
                    timeline::Tools {
                        split: &mut self.split,
                        can_undo: self.history.can_undo(),
                        can_redo: self.history.can_redo(),
                        playing: self.play.is_some(),
                    },
                );
                match timeline::take_tool(ui) {
                    Some(timeline::ToolPress::Undo) => self.step_history(ui, true),
                    Some(timeline::ToolPress::Redo) => self.step_history(ui, false),
                    Some(timeline::ToolPress::Delete) => self.remove_selection(ui),
                    None => {}
                }
                if !commands.is_empty() {
                    let select_placed = timeline::take_placed(ui);
                    match self.history.apply(&mut self.project, &commands) {
                        Ok(outcomes) => {
                            if select_placed && let Some(id) = timeline::placed_segment(&outcomes) {
                                self.selected_segments = vec![id];
                            }
                        }
                        Err(error) => {
                            timeline::show_note(
                                ui,
                                timeline::apply_note(&error, words::active(ui)),
                            );
                        }
                    }
                }
            });
        if self.play.is_some() && self.playhead_us != driven {
            self.play = None;
            self.hear.stop();
        }

        let mut import = false;
        let mut placed = Vec::new();
        egui::Panel::left("library")
            .resizable(false)
            .exact_size(self.panes.library())
            .frame(theme::gap_right())
            .show_separator_line(false)
            .show(ui, |ui| {
                let action = library::show(
                    ui,
                    &self.project,
                    &mut self.shelf,
                    &mut self.media,
                    &mut self.bin,
                );
                import = action.import;
                placed = action.commands;
            });
        if import && !self.blocked() {
            self.import_chosen(ui);
        }
        if !placed.is_empty() && !self.blocked() {
            match self.history.apply(&mut self.project, &placed) {
                Ok(outcomes) => {
                    if let Some(id) = timeline::placed_segment(&outcomes) {
                        self.selected_segments = vec![id];
                    }
                }
                Err(error) => {
                    timeline::show_note(ui, timeline::apply_note(&error, words::active(ui)));
                }
            }
        }
        if let Some(id) = library::take_relink(ui.ctx())
            && !self.blocked()
        {
            self.relink(ui, id);
        }

        egui::Panel::right("inspector")
            .resizable(false)
            .exact_size(self.panes.inspector())
            .frame(theme::gap_left())
            .show_separator_line(false)
            .show(ui, |ui| {
                let shown = match self.selected_segments.as_slice() {
                    [id] => Some(*id),
                    _ => None,
                };
                let commands = inspector::show(ui, &self.project, shown, self.playhead_us);
                if !commands.is_empty()
                    && let Err(error) = self.history.apply(&mut self.project, &commands)
                {
                    timeline::show_note(ui, timeline::apply_note(&error, words::active(ui)));
                }
            });

        egui::CentralPanel::default()
            .frame(theme::panel())
            .show(ui, |ui| {
                let hidden = timeline::hidden_tracks(&self.project);
                let pictures = self.reel.images(
                    ui.ctx(),
                    frame.wgpu_render_state(),
                    &self.project,
                    self.playhead_us,
                    &hidden,
                    self.play.is_some(),
                );
                let commands = preview::show(
                    ui,
                    &self.project,
                    self.playhead_us,
                    self.play.is_some(),
                    &mut self.selected_segments,
                    &pictures,
                );
                if !commands.is_empty()
                    && let Err(error) = self.history.apply(&mut self.project, &commands)
                {
                    timeline::show_note(ui, timeline::apply_note(&error, words::active(ui)));
                }
            });
        if key_play || (!self.blocked() && preview::take_play(ui)) {
            self.toggle_play();
        }
        if self.play.is_some() {
            let rate = self.project.settings().rate().rate();
            ui.ctx()
                .request_repaint_after(cutlass_engine::frame_wait(rate));
        }
        if self.reel.waiting() {
            ui.ctx().request_repaint();
        }
        gaps::show(ui, body, &self.panes);

        let span = self.project.duration();
        let label = export::duration_label(span.value, span.timescale.get());
        let dismiss = self
            .export
            .as_mut()
            .map(|sheet| export::show(ui, sheet, &label));
        if !opened {
            match dismiss {
                Some(export::Dismiss::Close) => self.export = None,
                Some(export::Dismiss::Export) => self.begin_export(),
                Some(export::Dismiss::Stay) | None => {}
            }
        }

        let asked = modify::take_ask(ui.ctx());
        if asked && !self.blocked() {
            self.modify = Some(modify::Draft::open(&self.project));
        }
        let outcome = self
            .modify
            .as_mut()
            .map(|draft| modify::show(ui, &self.project, draft));
        match outcome {
            Some(modify::Outcome::Saved(commands)) => {
                if !commands.is_empty()
                    && let Err(error) = self.history.apply(&mut self.project, &commands)
                {
                    timeline::show_note(ui, timeline::apply_note(&error, words::active(ui)));
                }
                self.modify = None;
            }
            Some(modify::Outcome::Closed) => self.modify = None,
            Some(modify::Outcome::Open) | None => {}
        }

        let mut opened_settings = false;
        if bar.settings && !self.blocked() {
            self.app_dialog = Some(app_settings::Draft::open(&self.app));
            opened_settings = true;
        }
        let outcome = self
            .app_dialog
            .as_mut()
            .map(|draft| app_settings::show(ui, draft));
        match outcome {
            Some(app_settings::Outcome::Saved(settings)) if !opened_settings => {
                self.app = settings;
                self.app_dialog = None;
            }
            Some(app_settings::Outcome::Closed) if !opened_settings => self.app_dialog = None,
            _ => {}
        }
        if self.pending.is_some() {
            let choice = projects::show_guard(ui);
            if !opened_guard {
                self.finish_guard(ui, choice);
            }
        }
        let mut opened_recovery = false;
        if self.recovery.is_some() && !self.recovery_shown {
            opened_recovery = true;
            self.recovery_shown = true;
        }
        if let Some(found) = self.recovery.clone() {
            let choice = projects::show_recovery(ui, &found.name);
            if !opened_recovery {
                self.finish_recovery(ui, choice);
            }
        }
        if !self.blocked() {
            import::show_hover(ui);
        }
    }
}

struct Bar {
    export: bool,
    settings: bool,
    file: Option<projects::Ask>,
}

/// Project bar. The Mac close, minimize, and maximize buttons are the window's.
/// Dragging the name moves the window. File, Export, and Settings do not.
fn header(ui: &mut egui::Ui, name: &str, recent: &[PathBuf]) -> Bar {
    crate::tip::region(ui, words::of(ui, words::Phrase::RegionProject));
    ui.take_available_height();
    let rect = ui.max_rect();
    let font = TextStyle::Button.resolve(ui.style());
    let export = words::of(ui, words::Phrase::Export);
    let settings = words::of(ui, words::Phrase::Settings);
    let file = words::of(ui, words::Phrase::File);
    let export_w = words::control_width(ui, export, font.clone(), 20.0, 72.0);
    let settings_w = words::control_width(ui, settings, font.clone(), 20.0, 72.0);
    let file_w = words::control_width(ui, file, font.clone(), 20.0, 64.0);
    let export_rect = Rect::from_min_size(
        rect.right_center() - Vec2::new(12.0 + export_w, 13.0),
        Vec2::new(export_w, 26.0),
    );
    let settings_rect = Rect::from_min_size(
        pos2(export_rect.left() - 8.0 - settings_w, export_rect.top()),
        Vec2::new(settings_w, 26.0),
    );
    let file_rect = Rect::from_min_size(
        pos2(rect.left() + 80.0, export_rect.top()),
        Vec2::new(file_w, 26.0),
    );
    let drag = ui.interact(rect, ui.scope_id().with("project-bar"), Sense::drag());
    let pointer = ui.input(|input| input.pointer.interact_pos());
    let on_control = pointer.is_some_and(|point| {
        export_rect.contains(point) || settings_rect.contains(point) || file_rect.contains(point)
    });
    if drag.drag_started() && !on_control {
        ui.ctx().send_viewport_cmd(egui::ViewportCommand::StartDrag);
    }
    ui.painter()
        .text(rect.center(), Align2::CENTER_CENTER, name, font, theme::INK);
    let name_rect = Rect::from_min_max(
        pos2(file_rect.right() + 8.0, rect.center().y - 12.0),
        pos2(settings_rect.left() - 8.0, rect.center().y + 12.0),
    );
    if name_rect.is_positive() {
        crate::tip::read(ui, name_rect, ui.scope_id().with("project-name"), name);
    }

    let file_ask = projects::file_button(ui, file_rect, recent);
    let settings_clicked = theme::with_face(ui, theme::BAR, theme::INK, |ui| {
        ui.put(settings_rect, Button::new(RichText::new(settings)))
    })
    .clicked();
    let export_clicked = theme::with_face(ui, theme::EXPORT, theme::EXPORT_INK, |ui| {
        ui.put(
            export_rect,
            Button::new(RichText::new(export).strong()).corner_radius(8),
        )
    })
    .clicked();
    Bar {
        export: export_clicked,
        settings: settings_clicked,
        file: file_ask,
    }
}

#[cfg(test)]
mod preview_device {
    use super::request_preview_features;

    #[test]
    fn the_window_asks_for_ten_bit_samples_when_the_adapter_has_them() {
        let instance = eframe::wgpu::Instance::new(
            eframe::wgpu::InstanceDescriptor::new_without_display_handle(),
        );
        let Ok(adapter) = pollster::block_on(
            instance.request_adapter(&eframe::wgpu::RequestAdapterOptions::default()),
        ) else {
            return;
        };
        if !adapter
            .features()
            .contains(eframe::wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
        {
            return;
        }
        let mut native = eframe::NativeOptions::default();
        request_preview_features(&mut native);
        let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &native.wgpu_options.wgpu_setup else {
            panic!("the window opens its own device");
        };
        let descriptor = (setup.device_descriptor)(&adapter);
        assert!(
            descriptor
                .required_features
                .contains(eframe::wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
        );
    }
}
