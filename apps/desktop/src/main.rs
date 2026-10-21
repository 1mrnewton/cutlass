//! The desktop editor. It holds one project and draws the window in `docs/desktop-layout.md`.

#![deny(unsafe_code)]

use eframe::egui::{self, Align2, Button, Rect, RichText, TextStyle, Vec2};

use cutlass_project::{AssetId, History, Project, ProjectId, SegmentId};

mod inspector;
mod library;
mod preview;
mod ruler;
mod theme;
mod timeline;

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
    selected_asset: Option<AssetId>,
    selected_segment: Option<SegmentId>,
    clip_drag: Option<timeline::ClipDrag>,
}

impl CutlassApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        theme::install(&cc.egui_ctx);
        let id = ProjectId::from_bytes(uuid::Uuid::new_v4().into_bytes());
        Self {
            history: History::new(),
            project: Project::new(id, "Untitled"),
            playhead_us: 0,
            shelf: library::Shelf::Media,
            selected_asset: None,
            selected_segment: None,
            clip_drag: None,
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
        timeline::delete_selected(
            &mut self.history,
            &mut self.project,
            &mut self.selected_segment,
        );
    }
}

impl eframe::App for CutlassApp {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        theme::GAP.to_normalized_gamma_f32()
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.delete_pressed(ui);

        egui::Panel::top("head")
            .exact_size(theme::HEADER_HEIGHT)
            .frame(theme::bar())
            .show_separator_line(false)
            .show(ui, |ui| header(ui, self.project.name()));

        let timeline_max =
            (ui.available_height() - theme::TIMELINE_RESERVE).max(theme::TIMELINE_MIN);

        egui::Panel::bottom("timeline")
            .resizable(true)
            .default_size(420.0)
            .size_range(theme::TIMELINE_MIN..=timeline_max)
            .frame(theme::gap_above())
            .show_separator_line(false)
            .show(ui, |ui| {
                if let Some(command) = timeline::show(
                    ui,
                    &self.project,
                    &mut self.playhead_us,
                    &mut self.selected_segment,
                    &mut self.clip_drag,
                ) {
                    let _ = self.history.apply(&mut self.project, &[command]);
                }
            });

        egui::Panel::left("library")
            .resizable(true)
            .default_size(368.0)
            .min_size(220.0)
            .frame(theme::gap_right())
            .show_separator_line(false)
            .show(ui, |ui| {
                library::show(ui, &self.project, &mut self.shelf, &mut self.selected_asset);
            });

        egui::Panel::right("inspector")
            .resizable(true)
            .default_size(300.0)
            .min_size(240.0)
            .frame(theme::gap_left())
            .show_separator_line(false)
            .show(ui, |ui| {
                inspector::show(ui, &self.project, self.selected_segment);
            });

        egui::CentralPanel::default()
            .frame(theme::panel())
            .show(ui, |ui| {
                preview::show(ui, &self.project, self.playhead_us);
            });
    }
}

/// Project bar. The Mac close, minimize, and maximize buttons are the window's.
fn header(ui: &mut egui::Ui, name: &str) {
    ui.take_available_height();
    let rect = ui.max_rect();
    ui.painter().text(
        rect.center(),
        Align2::CENTER_CENTER,
        name,
        TextStyle::Button.resolve(ui.style()),
        theme::INK,
    );

    let size = Vec2::new(72.0, 26.0);
    let origin = rect.right_center() - Vec2::new(12.0 + size.x, size.y / 2.0);
    ui.put(
        Rect::from_min_size(origin, size),
        Button::new(RichText::new("Export").color(theme::EXPORT_INK).strong())
            .fill(theme::EXPORT)
            .corner_radius(8),
    );
}
