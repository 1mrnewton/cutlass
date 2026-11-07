//! Shared controls. A panel uses these instead of picking a color or drawing an icon.

use egui::{Align2, CornerRadius, FontId, Rect, Sense, Ui, Vec2};

use crate::tokens;
use crate::ui;

pub struct HeaderFlags {
    pub mute: bool,
    pub hide: bool,
    pub lock: bool,
    pub solo: bool,
}

pub struct HeaderPress {
    pub mute: bool,
    pub hide: bool,
    pub lock: bool,
    pub solo: bool,
}

pub fn icon_button(ui: &mut Ui, name: &str, tip: &str) -> egui::Response {
    ui.add(
        ui::Button::icon(name)
            .kind(ui::ButtonKind::Secondary)
            .min_size(Vec2::splat(tokens::ICON_BUTTON_SIZE))
            .icon_size(18.0)
            .tip(tip),
    )
}

pub fn toolbar_button(ui: &mut Ui, label: &str) -> egui::Response {
    ui.add(ui::Button::new(label).tip(label))
}

pub fn export_button(ui: &mut Ui) -> egui::Response {
    ui.add(
        ui::Button::primary("Export")
            .min_size(Vec2::new(72.0, 26.0))
            .tip("Export"),
    )
}

pub fn field(ui: &mut Ui, text: &mut String) -> egui::Response {
    ui::Field::new(text).width(160.0).show(ui).response
}

pub fn lane_header(ui: &mut Ui, flags: HeaderFlags) -> HeaderPress {
    ui.horizontal(|ui| HeaderPress {
        mute: mark(ui, "Mute", flags.mute),
        hide: mark(ui, "Hide", flags.hide),
        lock: mark(ui, "Lock", flags.lock),
        solo: mark(ui, "Solo", flags.solo),
    })
    .inner
}

fn mark(ui: &mut Ui, label: &str, on: bool) -> bool {
    ui.add(ui::Button::new(label).on(on).tip(label)).clicked()
}

pub fn dialog<R>(ui: &mut Ui, title: &str, add: impl FnOnce(&mut Ui) -> R) -> (R, bool) {
    ui::Surface::scrim().paint(ui.painter(), ui.max_rect());
    let screen = ui.max_rect();
    let width = 420.0;
    let top = (screen.center().y - 180.0).max(screen.top() + 16.0);
    let origin = egui::pos2(screen.center().x - width * 0.5, top);
    // The body grows with what it holds, to the window's foot, and scrolls
    // past that.
    let room = (screen.bottom() - 16.0 - top).max(160.0);
    let body = ui.painter().add(egui::Shape::Noop);
    let mut closed = false;
    let inner = ui
        .scope_builder(
            egui::UiBuilder::new().max_rect(Rect::from_min_size(origin, Vec2::new(width, room))),
            |ui| {
                let rect = ui.max_rect();
                let head = Rect::from_min_size(rect.min, Vec2::new(rect.width(), 40.0));
                ui.painter().rect_filled(
                    head,
                    CornerRadius::same(tokens::DIALOG_RADIUS),
                    tokens::DIALOG_HEAD,
                );
                ui.painter().text(
                    head.left_center() + Vec2::new(16.0, 0.0),
                    Align2::LEFT_CENTER,
                    title,
                    FontId::proportional(tokens::TYPE_SIZE),
                    tokens::TEXT,
                );
                let close = Rect::from_center_size(
                    head.right_center() - Vec2::new(20.0, 0.0),
                    Vec2::splat(tokens::WINDOW_BUTTON),
                );
                ui.painter()
                    .circle_filled(close.center(), 6.0, tokens::CLOSE);
                let close_response = ui.allocate_rect(close, Sense::click());
                closed = close_response.clicked();
                ui.add_space(48.0);
                let inner = ui::Scroll::vertical(("dialog", title))
                    .max_height(room - 64.0)
                    .auto_shrink([false, true])
                    .show(ui, add);
                ui.add_space(16.0);
                let height = ui.min_rect().height().clamp(360.0_f32.min(room), room);
                ui.painter().set(
                    body,
                    ui::Surface::dialog()
                        .shape(Rect::from_min_size(rect.min, Vec2::new(width, height))),
                );
                inner
            },
        )
        .inner;
    (inner, closed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Color32, Rect, Ui};

    fn paint(mut add: impl FnMut(&mut Ui)) -> egui::FullOutput {
        let ctx = egui::Context::default();
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(800.0, 600.0),
                )),
                ..Default::default()
            },
            |ui| add(ui),
        );
        output.textures_delta.clear();
        output
    }

    fn has_color(output: &egui::FullOutput, color: Color32) -> bool {
        output
            .shapes
            .iter()
            .any(|clipped| shape_has(&clipped.shape, color))
    }

    fn shape_has(shape: &egui::Shape, color: Color32) -> bool {
        match shape {
            egui::Shape::Rect(rect) => rect.fill == color || rect.stroke.color == color,
            egui::Shape::Path(path) => {
                path.fill == color
                    || matches!(path.stroke.color, egui::epaint::ColorMode::Solid(solid) if solid == color)
            }
            egui::Shape::Text(text) => text.fallback_color == color,
            egui::Shape::Circle(circle) => circle.fill == color || circle.stroke.color == color,
            egui::Shape::Vec(shapes) => shapes.iter().any(|shape| shape_has(shape, color)),
            _ => false,
        }
    }

    #[test]
    fn shared_widgets_use_the_token_colors() {
        let output = paint(|ui| {
            assert!(export_button(ui).on_hover_text("Export").rect.width() > 0.0);
            icon_button(ui, "play", "Play");
            icon_button(ui, "missing-icon", "Missing");
            let mut value = "field".to_owned();
            field(ui, &mut value);
            lane_header(
                ui,
                HeaderFlags {
                    mute: true,
                    hide: false,
                    lock: false,
                    solo: false,
                },
            );
        });
        assert!(has_color(&output, tokens::EXPORT));
        assert!(has_color(&output, tokens::ICON_BUTTON));
    }

    #[test]
    fn a_dialog_uses_the_shared_chrome() {
        let output = paint(|ui| {
            let _ = dialog(ui, "Export", |ui| {
                ui.label("Short side");
            });
        });
        assert!(has_color(&output, tokens::PROJECT_BAR) || has_color(&output, tokens::DIALOG_HEAD));
    }
}
