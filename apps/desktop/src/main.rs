//! The desktop editor. It holds one project and draws the window in `docs/desktop-layout.md`.

#![deny(unsafe_code)]

use eframe::egui::{self, Align2, Button, Rect, RichText, Sense, TextStyle, Vec2, pos2};

use cutlass_project::{History, Project, ProjectId, SegmentId};

mod chrome;
mod export;
mod gaps;
mod icons;
mod inspector;
mod keys;
mod library;
mod media_bin;
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
}

impl eframe::App for CutlassApp {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        theme::GAP.to_normalized_gamma_f32()
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        words::install(ui.ctx(), self.language);
        chrome::pass_clicks(frame);
        keys::set_blocked(ui, self.export.is_some());
        if self.export.is_none() {
            self.history_shortcut(ui);
            self.delete_pressed(ui);
        }

        let mut open_export = false;
        egui::Panel::top("head")
            .exact_size(theme::HEADER_HEIGHT)
            .frame(theme::bar())
            .show_separator_line(false)
            .show(ui, |ui| {
                open_export = header(ui, self.project.name());
            });
        let opened = open_export && self.export.is_none();
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

        egui::Panel::left("library")
            .resizable(false)
            .exact_size(self.panes.library())
            .frame(theme::gap_right())
            .show_separator_line(false)
            .show(ui, |ui| {
                library::show(
                    ui,
                    &self.project,
                    &mut self.shelf,
                    &mut self.media,
                    &mut self.bin,
                );
            });

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
                inspector::show(ui, &self.project, shown);
            });

        egui::CentralPanel::default()
            .frame(theme::panel())
            .show(ui, |ui| {
                preview::show(
                    ui,
                    &self.project,
                    self.playhead_us,
                    &mut self.selected_segments,
                );
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
    }
}

/// Project bar. The Mac close, minimize, and maximize buttons are the window's.
fn header(ui: &mut egui::Ui, name: &str) -> bool {
    crate::tip::region(ui, words::of(ui, words::Phrase::RegionProject));
    ui.take_available_height();
    let rect = ui.max_rect();
    let export = words::of(ui, words::Phrase::Export);
    let width = words::control_width(
        ui,
        export,
        TextStyle::Button.resolve(ui.style()),
        20.0,
        72.0,
    );
    let size = Vec2::new(width, 26.0);
    let origin = rect.right_center() - Vec2::new(12.0 + size.x, size.y / 2.0);
    let button = Rect::from_min_size(origin, size);
    let drag = ui.interact(rect, ui.scope_id().with("project-bar"), Sense::drag());
    let pointer = ui.input(|input| input.pointer.interact_pos());
    if drag.drag_started() && pointer.is_none_or(|point| !button.contains(point)) {
        ui.ctx().send_viewport_cmd(egui::ViewportCommand::StartDrag);
    }
    ui.painter().text(
        rect.center(),
        Align2::CENTER_CENTER,
        name,
        TextStyle::Button.resolve(ui.style()),
        theme::INK,
    );
    let name_rect = Rect::from_min_max(
        pos2(rect.left() + 80.0, rect.center().y - 12.0),
        pos2(button.left() - 8.0, rect.center().y + 12.0),
    );
    if name_rect.is_positive() {
        crate::tip::read(ui, name_rect, ui.scope_id().with("project-name"), name);
    }

    ui.put(
        button,
        Button::new(RichText::new(export).color(theme::EXPORT_INK).strong())
            .fill(theme::EXPORT)
            .corner_radius(8),
    )
    .clicked()
}
