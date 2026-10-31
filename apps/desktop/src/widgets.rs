//! Shared controls. A panel uses these instead of picking a color or drawing an icon.

use egui::{Align2, CornerRadius, FontId, Rect, RichText, Sense, TextEdit, Ui, Vec2};

use crate::icons;
use crate::tokens;

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
    let (rect, response) =
        ui.allocate_exact_size(Vec2::splat(tokens::ICON_BUTTON_SIZE), Sense::click());
    if ui.is_rect_visible(rect) {
        ui.painter().rect_filled(
            rect,
            CornerRadius::same(tokens::ICON_RADIUS),
            tokens::ICON_BUTTON,
        );
        let mark = Rect::from_center_size(rect.center(), Vec2::splat(18.0));
        if let Err(error) = icons::paint(ui.painter(), mark, name, tokens::TEXT) {
            ui.painter().text(
                rect.center(),
                Align2::CENTER_CENTER,
                error.to_string(),
                FontId::proportional(9.0),
                tokens::FAINT,
            );
        }
    }
    response.on_hover_text(tip)
}

pub fn toolbar_button(ui: &mut Ui, label: &str) -> egui::Response {
    let response = ui.add(
        egui::Button::new(
            RichText::new(label)
                .size(tokens::TYPE_SIZE)
                .color(tokens::TEXT),
        )
        .fill(tokens::ICON_BUTTON)
        .corner_radius(CornerRadius::same(tokens::ICON_RADIUS)),
    );
    response.on_hover_text(label)
}

pub fn export_button(ui: &mut Ui) -> egui::Response {
    let font = FontId::proportional(tokens::TYPE_SIZE);
    let galley = ui
        .ctx()
        .fonts_mut(|fonts| fonts.layout_no_wrap("Export".to_owned(), font, tokens::EXPORT_INK));
    let pad = Vec2::new(
        f32::from(tokens::EXPORT_PAD_X),
        f32::from(tokens::EXPORT_PAD_Y),
    );
    let (rect, response) = ui.allocate_exact_size(galley.size() + pad * 2.0, Sense::click());
    if ui.is_rect_visible(rect) {
        ui.painter().rect_filled(
            rect,
            CornerRadius::same(tokens::EXPORT_RADIUS),
            tokens::EXPORT,
        );
        ui.painter()
            .galley(rect.min + pad, galley, tokens::EXPORT_INK);
    }
    response.on_hover_text("Export")
}

pub fn field(ui: &mut Ui, text: &mut String) -> egui::Response {
    ui.add(
        TextEdit::singleline(text)
            .text_color(tokens::TEXT)
            .font(egui::TextStyle::Body)
            .desired_width(160.0),
    )
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
    let color = if on { tokens::TEXT } else { tokens::FAINT };
    ui.add(
        egui::Button::new(RichText::new(label).size(tokens::TYPE_SIZE).color(color))
            .fill(tokens::ICON_BUTTON)
            .corner_radius(CornerRadius::same(tokens::ICON_RADIUS)),
    )
    .on_hover_text(label)
    .clicked()
}

pub fn dialog<R>(ui: &mut Ui, title: &str, add: impl FnOnce(&mut Ui) -> R) -> (R, bool) {
    ui.painter()
        .rect_filled(ui.max_rect(), CornerRadius::ZERO, tokens::SCRIM);
    let width = 420.0;
    let origin = ui.max_rect().center() - Vec2::new(width * 0.5, 180.0);
    let mut closed = false;
    let inner = ui
        .scope_builder(
            egui::UiBuilder::new().max_rect(Rect::from_min_size(origin, Vec2::new(width, 360.0))),
            |ui| {
                let rect = ui.max_rect();
                ui.painter().rect_filled(
                    rect,
                    CornerRadius::same(tokens::DIALOG_RADIUS),
                    tokens::PROJECT_BAR,
                );
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
                add(ui)
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
                ui.label("Long side");
            });
        });
        assert!(has_color(&output, tokens::PROJECT_BAR) || has_color(&output, tokens::DIALOG_HEAD));
    }
}
