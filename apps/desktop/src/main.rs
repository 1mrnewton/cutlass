//! The desktop editor. It holds one project and draws the window in `docs/desktop-layout.md`.

#![deny(unsafe_code)]

use eframe::egui::{self, Align2, Button, Rect, RichText, Sense, TextStyle, Vec2, pos2};

use cutlass_project::{History, Project, ProjectId, SegmentId};

mod app_settings;
mod chrome;
mod export;
mod gaps;
mod icons;
mod import;
mod inspector;
mod keys;
mod library;
mod media_bin;
mod modify;
mod preview;
mod ruler;
mod theme;
mod timeline;
mod tip;
mod words;

fn main() -> eframe::Result {
    let viewport = egui::ViewportBuilder::default()
        .with_title("Cutlass")
        .with_app_id("cutlass")
        .with_inner_size([1280.0, 800.0])
        .with_min_inner_size([1100.0, 700.0])
        .with_fullsize_content_view(true)
        .with_titlebar_shown(false)
        .with_title_shown(false);

    eframe::run_native(
        "Cutlass",
        eframe::NativeOptions {
            viewport,
            ..Default::default()
        },
        Box::new(|cc| Ok(Box::new(CutlassApp::new(cc)))),
    )
}

struct CutlassApp {
    history: History,
    project: Project,
    playhead_us: i64,
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
        let id = ProjectId::from_bytes(uuid::Uuid::new_v4().into_bytes());
        Self {
            history: History::new(),
            project: Project::new(id, "Untitled"),
            playhead_us: 0,
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
        let lanes = timeline::lane_memory(ui);
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
        let undo = step == keys::Shortcut::Undo;
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
}

impl eframe::App for CutlassApp {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        theme::GAP.to_normalized_gamma_f32()
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        words::install(ui.ctx(), self.language);
        chrome::pass_clicks(frame);
        let dialog = self.export.is_some() || self.modify.is_some() || self.app_dialog.is_some();
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
        };
        egui::Panel::top("head")
            .exact_size(theme::HEADER_HEIGHT)
            .frame(theme::bar())
            .show_separator_line(false)
            .show(ui, |ui| {
                bar = header(ui, self.project.name());
            });
        let busy = self.export.is_some() || self.modify.is_some() || self.app_dialog.is_some();
        let opened = bar.export && !busy;
        if opened {
            self.export = Some(export::Sheet::open(
                self.project.name(),
                self.project.settings().rate().rate(),
            ));
        }

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
                    &mut self.split,
                );
                if !commands.is_empty()
                    && let Err(error) = self.history.apply(&mut self.project, &commands)
                {
                    timeline::show_note(ui, timeline::apply_note(&error, words::active(ui)));
                }
            });

        let mut import = false;
        egui::Panel::left("library")
            .resizable(false)
            .exact_size(self.panes.library())
            .frame(theme::gap_right())
            .show_separator_line(false)
            .show(ui, |ui| {
                import = library::show(
                    ui,
                    &self.project,
                    &mut self.shelf,
                    &mut self.media,
                    &mut self.bin,
                );
            });
        if import && self.export.is_none() && self.modify.is_none() && self.app_dialog.is_none() {
            self.import_chosen(ui);
        }
        if let Some(id) = library::take_relink(ui.ctx())
            && self.export.is_none()
            && self.modify.is_none()
            && self.app_dialog.is_none()
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
                let commands = preview::show(
                    ui,
                    &self.project,
                    self.playhead_us,
                    &mut self.selected_segments,
                );
                if !commands.is_empty()
                    && let Err(error) = self.history.apply(&mut self.project, &commands)
                {
                    timeline::show_note(ui, timeline::apply_note(&error, words::active(ui)));
                }
            });
        gaps::show(ui, body, &self.panes);

        let span = self.project.duration();
        let label = export::duration_label(span.value, span.timescale.get());
        let close = self
            .export
            .as_mut()
            .is_some_and(|sheet| export::show(ui, sheet, &label));
        if close && !opened {
            self.export = None;
        }

        let asked = modify::take_ask(ui.ctx());
        if asked && self.export.is_none() && self.modify.is_none() && self.app_dialog.is_none() {
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
        if bar.settings
            && self.export.is_none()
            && self.modify.is_none()
            && self.app_dialog.is_none()
        {
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
        if self.export.is_none() && self.modify.is_none() && self.app_dialog.is_none() {
            import::show_hover(ui);
        }
    }
}

struct Bar {
    export: bool,
    settings: bool,
}

/// Project bar. The Mac close, minimize, and maximize buttons are the window's.
fn header(ui: &mut egui::Ui, name: &str) -> Bar {
    crate::tip::region(ui, words::of(ui, words::Phrase::RegionProject));
    ui.take_available_height();
    let rect = ui.max_rect();
    let font = TextStyle::Button.resolve(ui.style());
    let export = words::of(ui, words::Phrase::Export);
    let settings = words::of(ui, words::Phrase::Settings);
    let export_w = words::control_width(ui, export, font.clone(), 20.0, 72.0);
    let settings_w = words::control_width(ui, settings, font.clone(), 20.0, 72.0);
    let export_rect = Rect::from_min_size(
        rect.right_center() - Vec2::new(12.0 + export_w, 13.0),
        Vec2::new(export_w, 26.0),
    );
    let settings_rect = Rect::from_min_size(
        pos2(export_rect.left() - 8.0 - settings_w, export_rect.top()),
        Vec2::new(settings_w, 26.0),
    );
    let drag = ui.interact(rect, ui.scope_id().with("project-bar"), Sense::drag());
    let pointer = ui.input(|input| input.pointer.interact_pos());
    let on_control =
        pointer.is_some_and(|point| export_rect.contains(point) || settings_rect.contains(point));
    if drag.drag_started() && !on_control {
        ui.ctx().send_viewport_cmd(egui::ViewportCommand::StartDrag);
    }
    ui.painter()
        .text(rect.center(), Align2::CENTER_CENTER, name, font, theme::INK);
    let name_rect = Rect::from_min_max(
        pos2(rect.left() + 80.0, rect.center().y - 12.0),
        pos2(settings_rect.left() - 8.0, rect.center().y + 12.0),
    );
    if name_rect.is_positive() {
        crate::tip::read(ui, name_rect, ui.scope_id().with("project-name"), name);
    }

    let settings_clicked = ui
        .put(
            settings_rect,
            Button::new(RichText::new(settings).color(theme::INK)).fill(theme::BAR),
        )
        .clicked();
    let export_clicked = ui
        .put(
            export_rect,
            Button::new(RichText::new(export).color(theme::EXPORT_INK).strong())
                .fill(theme::EXPORT)
                .corner_radius(8),
        )
        .clicked();
    Bar {
        export: export_clicked,
        settings: settings_clicked,
    }
}
