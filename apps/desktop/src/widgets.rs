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
    let (rect, response) = ui.allocate_exact_size(Vec2::new(72.0, 26.0), Sense::click());
    if ui.is_rect_visible(rect) {
        ui.painter().rect_filled(
            rect,
            CornerRadius::same(tokens::EXPORT_RADIUS),
            tokens::EXPORT,
        );
        ui.painter().text(
            rect.center(),
            Align2::CENTER_CENTER,
            "Export",
            FontId::proportional(tokens::TYPE_SIZE),
            tokens::EXPORT_INK,
        );
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

/// Dim the window and take the pointer, so nothing under a dialog reacts.
fn scrim(ui: &mut Ui, title: &str) {
    let screen = ui.max_rect();
    ui.painter()
        .rect_filled(screen, CornerRadius::ZERO, tokens::SCRIM);
    let _ = ui.interact(
        screen,
        egui::Id::unique(("scrim", title)),
        Sense::click_and_drag(),
    );
}

pub fn dialog<R>(ui: &mut Ui, title: &str, add: impl FnOnce(&mut Ui) -> R) -> (R, bool) {
    scrim(ui, title);
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
                let inner = egui::ScrollArea::vertical()
                    .id_salt(("dialog", title))
                    .max_height(room - 64.0)
                    .auto_shrink([false, true])
                    .show(ui, add)
                    .inner;
                ui.add_space(16.0);
                let height = ui.min_rect().height().clamp(360.0_f32.min(room), room);
                ui.painter().set(
                    body,
                    egui::Shape::rect_filled(
                        Rect::from_min_size(rect.min, Vec2::new(width, height)),
                        CornerRadius::same(tokens::DIALOG_RADIUS),
                        tokens::PROJECT_BAR,
                    ),
                );
                inner
            },
        )
        .inner;
    (inner, closed)
}

/// How a sheet ended this frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SheetEnd {
    Open,
    Cancel,
    Confirm,
}

const SHEET_HEAD: f32 = 52.0;
const SHEET_FOOT: f32 = 64.0;
const SHEET_PAD: f32 = 32.0;

/// A settings sheet: a centered title with the close dot on its left, a
/// scrolling body, and a foot with Cancel and `confirm`. Escape cancels.
pub fn sheet<R>(
    ui: &mut Ui,
    title: &str,
    width: f32,
    confirm: &str,
    add: impl FnOnce(&mut Ui) -> R,
) -> (R, SheetEnd) {
    let screen = ui.max_rect();
    scrim(ui, title);
    // The height the sheet took last frame centers it; the first frame
    // guesses.
    let id = egui::Id::unique(("sheet", title));
    let last: f32 = ui.data(|data| data.get_temp(id)).unwrap_or(480.0);
    let width = width.min(screen.width() - 32.0);
    let room = (screen.height() - 32.0).max(SHEET_HEAD + SHEET_FOOT + 80.0);
    let height = last.min(room);
    let origin = egui::pos2(
        screen.center().x - width * 0.5,
        (screen.center().y - height * 0.5).max(screen.top() + 16.0),
    );
    let radius = tokens::DIALOG_RADIUS;
    let body = ui.painter().add(egui::Shape::Noop);
    let mut end = SheetEnd::Open;
    let (inner, used) = ui
        .scope_builder(
            egui::UiBuilder::new().max_rect(Rect::from_min_size(origin, Vec2::new(width, room))),
            |ui| {
                let rect = ui.max_rect();
                let head = Rect::from_min_size(rect.min, Vec2::new(width, SHEET_HEAD));
                ui.painter().rect_filled(
                    head,
                    CornerRadius {
                        nw: radius,
                        ne: radius,
                        sw: 0,
                        se: 0,
                    },
                    tokens::DIALOG_HEAD,
                );
                ui.painter().text(
                    head.center(),
                    Align2::CENTER_CENTER,
                    title,
                    FontId::proportional(15.0),
                    tokens::TEXT,
                );
                let close = Rect::from_center_size(
                    head.left_center() + Vec2::new(24.0, 0.0),
                    Vec2::splat(18.0),
                );
                let close_response = ui.allocate_rect(close, Sense::click()).on_hover_text("Close");
                ui.painter().circle_filled(close.center(), 7.0, tokens::CLOSE);
                if close_response.hovered() {
                    let stroke = egui::Stroke::new(1.3, tokens::EXPORT_INK);
                    let c = close.center();
                    ui.painter()
                        .line_segment([c + Vec2::splat(-2.5), c + Vec2::splat(2.5)], stroke);
                    ui.painter().line_segment(
                        [c + Vec2::new(-2.5, 2.5), c + Vec2::new(2.5, -2.5)],
                        stroke,
                    );
                }
                if close_response.clicked() {
                    end = SheetEnd::Cancel;
                }
                let scroll_room = room - SHEET_HEAD - SHEET_FOOT;
                let body_top = head.bottom();
                let mut inner = None;
                ui.scope_builder(
                    egui::UiBuilder::new().max_rect(Rect::from_min_size(
                        egui::pos2(rect.left(), body_top),
                        Vec2::new(width, scroll_room),
                    )),
                    |ui| {
                        inner = Some(
                            egui::ScrollArea::vertical()
                                .id_salt(("sheet", title))
                                .max_height(scroll_room)
                                .auto_shrink([false, true])
                                .show(ui, |ui| {
                                    egui::Frame::new()
                                        .inner_margin(egui::Margin {
                                            left: SHEET_PAD as i8,
                                            right: SHEET_PAD as i8,
                                            top: 20,
                                            bottom: 16,
                                        })
                                        .show(ui, |ui| {
                                            ui.set_width(width - SHEET_PAD * 2.0);
                                            add(ui)
                                        })
                                        .inner
                                })
                                .inner,
                        );
                    },
                );
                let body_bottom = ui.min_rect().bottom().min(body_top + scroll_room);
                let foot = Rect::from_min_size(
                    egui::pos2(rect.left(), body_bottom),
                    Vec2::new(width, SHEET_FOOT),
                );
                ui.painter().rect_filled(
                    foot,
                    CornerRadius {
                        nw: 0,
                        ne: 0,
                        sw: radius,
                        se: radius,
                    },
                    tokens::DIALOG_HEAD,
                );
                let save = Rect::from_min_size(
                    foot.right_center() + Vec2::new(-24.0 - 88.0, -17.0),
                    Vec2::new(88.0, 34.0),
                );
                if accent_button(ui, save, confirm).clicked() {
                    end = SheetEnd::Confirm;
                }
                let cancel = Rect::from_min_size(
                    save.left_top() - Vec2::new(88.0, 0.0),
                    Vec2::new(80.0, 34.0),
                );
                if quiet_button(ui, cancel, "Cancel").clicked() {
                    end = SheetEnd::Cancel;
                }
                let used = foot.bottom() - rect.top();
                ui.painter().set(
                    body,
                    egui::Shape::rect_filled(
                        Rect::from_min_size(rect.min, Vec2::new(width, used)),
                        CornerRadius::same(radius),
                        tokens::PROJECT_BAR,
                    ),
                );
                (inner, used)
            },
        )
        .inner;
    if (used - last).abs() > 0.5 {
        ui.data_mut(|data| data.insert_temp(id, used));
        ui.ctx().request_repaint();
    }
    if end == SheetEnd::Open && ui.input(|input| input.key_pressed(egui::Key::Escape)) {
        end = SheetEnd::Cancel;
    }
    let Some(inner) = inner else {
        unreachable!("the sheet body always runs");
    };
    (inner, end)
}

/// The filled accent button, the sheet's confirm.
fn accent_button(ui: &mut Ui, rect: Rect, label: &str) -> egui::Response {
    let response = ui.allocate_rect(rect, Sense::click());
    let fill = if response.hovered() {
        tokens::EXPORT.gamma_multiply(0.88)
    } else {
        tokens::EXPORT
    };
    ui.painter()
        .rect_filled(rect, CornerRadius::same(tokens::EXPORT_RADIUS), fill);
    ui.painter().text(
        rect.center(),
        Align2::CENTER_CENTER,
        label,
        FontId::proportional(14.0),
        tokens::EXPORT_INK,
    );
    response
}

/// A text-only button.
fn quiet_button(ui: &mut Ui, rect: Rect, label: &str) -> egui::Response {
    let response = ui.allocate_rect(rect, Sense::click());
    let color = if response.hovered() {
        tokens::TEXT
    } else {
        tokens::MUTED
    };
    ui.painter().text(
        rect.center(),
        Align2::CENTER_CENTER,
        label,
        FontId::proportional(14.0),
        color,
    );
    response
}

const FORM_LABEL: f32 = 150.0;
const FORM_FIELD: f32 = 36.0;

/// A small capital heading over a group of rows.
pub fn form_section(ui: &mut Ui, title: &str) {
    ui.add_space(8.0);
    ui.label(
        RichText::new(title.to_uppercase())
            .size(11.0)
            .extra_letter_spacing(1.5)
            .color(tokens::SECTION),
    );
    ui.add_space(10.0);
}

/// A label on the left and its control filling the rest of the row.
pub fn form_row<R>(ui: &mut Ui, label: &str, add: impl FnOnce(&mut Ui) -> R) -> R {
    let inner = ui
        .horizontal(|ui| {
            let (rect, _) =
                ui.allocate_exact_size(Vec2::new(FORM_LABEL, FORM_FIELD), Sense::hover());
            ui.painter().text(
                rect.left_center(),
                Align2::LEFT_CENTER,
                label,
                FontId::proportional(tokens::TYPE_SIZE + 1.0),
                tokens::MUTED,
            );
            ui.vertical(|ui| {
                ui.set_min_height(FORM_FIELD);
                add(ui)
            })
            .inner
        })
        .inner;
    ui.add_space(10.0);
    inner
}

/// A line of small text under a row's control.
pub fn form_hint(ui: &mut Ui, text: &str) {
    ui.horizontal(|ui| {
        ui.add_space(FORM_LABEL + ui.spacing().item_spacing.x);
        ui.add(
            egui::Label::new(RichText::new(text).size(12.0).color(tokens::FAINT)).wrap(),
        );
    });
    ui.add_space(10.0);
}

/// A full-width filled text box.
pub fn text_box(ui: &mut Ui, text: &mut String) -> egui::Response {
    egui::Frame::new()
        .fill(tokens::FIELD)
        .corner_radius(CornerRadius::same(tokens::EXPORT_RADIUS))
        .inner_margin(egui::Margin::symmetric(12, 9))
        .show(ui, |ui| {
            ui.add(
                TextEdit::singleline(text)
                    .frame(egui::Frame::NONE)
                    .text_color(tokens::TEXT)
                    .font(FontId::proportional(tokens::TYPE_SIZE + 1.0))
                    .desired_width(f32::INFINITY),
            )
        })
        .inner
}

/// A full-width picker showing `current`, opening `options` below it.
/// Returns the index picked this frame.
pub fn select(ui: &mut Ui, id: &str, current: &str, options: &[&str]) -> Option<usize> {
    let (rect, response) = ui.allocate_exact_size(
        Vec2::new(ui.available_width(), FORM_FIELD),
        Sense::click(),
    );
    let fill = if response.hovered() {
        tokens::MENU
    } else {
        tokens::FIELD
    };
    ui.painter()
        .rect_filled(rect, CornerRadius::same(tokens::EXPORT_RADIUS), fill);
    ui.painter().text(
        rect.left_center() + Vec2::new(12.0, 0.0),
        Align2::LEFT_CENTER,
        current,
        FontId::proportional(tokens::TYPE_SIZE + 1.0),
        tokens::TEXT,
    );
    let mark = rect.right_center() - Vec2::new(18.0, 0.0);
    let stroke = egui::Stroke::new(1.5, tokens::MENU_CHEVRON);
    ui.painter()
        .line_segment([mark + Vec2::new(-4.5, -2.0), mark + Vec2::new(0.0, 2.5)], stroke);
    ui.painter()
        .line_segment([mark + Vec2::new(0.0, 2.5), mark + Vec2::new(4.5, -2.0)], stroke);
    let mut picked = None;
    egui::Popup::menu(&response)
        .id(egui::Id::unique(("select", id)))
        .width(rect.width())
        .gap(4.0)
        .frame(
            egui::Frame::new()
                .fill(tokens::MENU)
                .corner_radius(CornerRadius::same(tokens::EXPORT_RADIUS))
                .inner_margin(egui::Margin::same(4)),
        )
        .show(|ui| {
            ui.set_width(rect.width() - 8.0);
            for (index, option) in options.iter().enumerate() {
                let (row, row_response) = ui.allocate_exact_size(
                    Vec2::new(ui.available_width(), 30.0),
                    Sense::click(),
                );
                if row_response.hovered() {
                    ui.painter()
                        .rect_filled(row, CornerRadius::same(6), tokens::MENU_HOVER);
                }
                let chosen = *option == current;
                ui.painter().text(
                    row.left_center() + Vec2::new(10.0, 0.0),
                    Align2::LEFT_CENTER,
                    *option,
                    FontId::proportional(tokens::TYPE_SIZE + 1.0),
                    if chosen { tokens::EXPORT } else { tokens::TEXT },
                );
                if row_response.clicked() {
                    picked = Some(index);
                    ui.close();
                }
            }
        });
    picked
}

/// An on/off switch at the right of the row. Off and not `enabled`, it
/// cannot be clicked.
pub fn switch(ui: &mut Ui, on: &mut bool, enabled: bool) -> egui::Response {
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        ui.set_min_height(FORM_FIELD);
        let sense = if enabled { Sense::click() } else { Sense::hover() };
        let (rect, mut response) = ui.allocate_exact_size(Vec2::new(44.0, 24.0), sense);
        if response.clicked() {
            *on = !*on;
            response.mark_changed();
        }
        let shown = ui.ctx().animate_bool(response.id, *on);
        let track = if *on {
            tokens::EXPORT
        } else {
            tokens::SWITCH_OFF
        };
        let alpha = if enabled { 1.0 } else { 0.4 };
        ui.painter().rect_filled(
            rect,
            CornerRadius::same(12),
            track.gamma_multiply(alpha),
        );
        let x = egui::lerp(rect.left() + 12.0..=rect.right() - 12.0, shown);
        ui.painter().circle_filled(
            egui::pos2(x, rect.center().y),
            9.0,
            egui::Color32::WHITE.gamma_multiply(if enabled { 1.0 } else { 0.5 }),
        );
        response
    })
    .inner
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
