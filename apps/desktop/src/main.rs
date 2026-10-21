//! The desktop editor. It holds one project and draws the window in `docs/desktop-layout.md`.

#![deny(unsafe_code)]

use eframe::egui::{self, Align2, Button, Rect, RichText, TextStyle, Vec2};

use cutlass_project::{Project, ProjectId};

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
    project: Project,
}

impl CutlassApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        theme::install(&cc.egui_ctx);
        let id = ProjectId::from_bytes(uuid::Uuid::new_v4().into_bytes());
        Self {
            project: Project::new(id, "Untitled"),
        }
    }
}

impl eframe::App for CutlassApp {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        theme::GAP.to_normalized_gamma_f32()
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
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
                timeline::show(ui, &self.project);
            });

        egui::Panel::left("library")
            .resizable(true)
            .default_size(368.0)
            .min_size(220.0)
            .frame(theme::gap_right())
            .show_separator_line(false)
            .show(ui, |ui| {
                ui.painter().rect_filled(ui.max_rect(), 0.0, theme::PANEL);
            });

        egui::Panel::right("inspector")
            .resizable(true)
            .default_size(300.0)
            .min_size(240.0)
            .frame(theme::gap_left())
            .show_separator_line(false)
            .show(ui, |ui| {
                ui.painter().rect_filled(ui.max_rect(), 0.0, theme::PANEL);
            });

        egui::CentralPanel::default()
            .frame(theme::panel())
            .show(ui, |_| {});
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
