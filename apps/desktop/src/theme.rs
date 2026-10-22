//! Chrome colors from `docs/desktop-layout.md`.

use std::sync::Arc;

use eframe::egui::{self, Color32, Frame, Margin, Stroke};

pub const GAP: Color32 = Color32::from_rgb(0x14, 0x14, 0x14);
pub const PANEL: Color32 = Color32::from_rgb(0x1c, 0x1c, 0x1c);
pub const BAR: Color32 = Color32::from_rgb(0x26, 0x26, 0x26);
pub const INK: Color32 = Color32::from_rgb(0xf3, 0xf3, 0xf3);
pub const MUTED: Color32 = Color32::from_rgb(0xa3, 0xa3, 0xa3);
pub const LANE: Color32 = Color32::from_rgb(0x24, 0x24, 0x24);
pub const LANE_LINE: Color32 = Color32::from_rgb(0x2e, 0x2e, 0x2e);
/// Active library shelf. `docs/library.md`.
pub const ACTIVE: Color32 = Color32::from_rgb(0x33, 0x33, 0x33);
/// Idle library-rail label.
pub const FAINT: Color32 = Color32::from_rgb(0x73, 0x73, 0x73);
pub const VIDEO: Color32 = Color32::from_rgb(0x5b, 0x8a, 0xa3);
pub const AUDIO: Color32 = Color32::from_rgb(0x4e, 0x8f, 0x7c);
pub const TEXT: Color32 = Color32::from_rgb(0x9a, 0x86, 0xb0);
pub const STICKER: Color32 = Color32::from_rgb(0xc4, 0xa1, 0x5a);
pub const EFFECT: Color32 = Color32::from_rgb(0xc4, 0x7a, 0x5a);
pub const FILTER: Color32 = Color32::from_rgb(0x6a, 0x8f, 0x72);
pub const ADJUST: Color32 = Color32::from_rgb(0x8a, 0x7e, 0x9a);
pub const EXPORT: Color32 = Color32::from_rgb(0x00, 0xc8, 0xd7);
pub const EXPORT_INK: Color32 = Color32::from_rgb(0x06, 0x22, 0x26);
/// Right-click plate. `docs/desktop-layout.md`.
pub const MENU: Color32 = Color32::from_rgb(0x2c, 0x2c, 0x2c);
pub const MENU_HOVER: Color32 = Color32::from_rgb(0x3a, 0x3a, 0x3a);
pub const MENU_OFF: Color32 = Color32::from_rgb(0x6f, 0x6f, 0x6f);
pub const MENU_KEY: Color32 = Color32::from_rgb(0x8d, 0x8d, 0x8d);
pub const MENU_KEY_OFF: Color32 = Color32::from_rgb(0x5c, 0x5c, 0x5c);
pub const MENU_BADGE: Color32 = Color32::from_rgb(0x6d, 0x5e, 0xfc);
/// Plate a clear button gains while the pointer is on it.
const HOVER_PLATE: Color32 = Color32::from_rgb(0x2e, 0x2e, 0x2e);

/// Dark gap between panels. That gap is the drag handle.
pub const GAP_PX: i8 = 6;

/// Project bar height. The Mac window buttons sit in this bar.
pub const HEADER_HEIGHT: f32 = 38.0;

/// Timeline cannot cover the preview. Head, frame, and transport stay visible.
pub const TIMELINE_RESERVE: f32 = 268.0;
pub const TIMELINE_MIN: f32 = 160.0;

pub fn install(ctx: &egui::Context) {
    ctx.set_theme(egui::ThemePreference::Dark);
    ctx.style_mut_of(egui::Theme::Dark, |style| {
        style.interaction.selectable_labels = false;
        // A hover name waits until the pointer has been still. `docs/desktop-layout.md`.
        style.interaction.tooltip_delay = 0.32;
        style.visuals.override_text_color = Some(INK);
        style.visuals.panel_fill = PANEL;
        style.visuals.window_fill = PANEL;
        style.visuals.extreme_bg_color = GAP;
        style.visuals.faint_bg_color = BAR;
        style.visuals.widgets.noninteractive.bg_stroke = Stroke::NONE;
        // No frame. A control that does not name its own fill steps to this plate.
        style.visuals.widgets.inactive.bg_stroke = Stroke::NONE;
        style.visuals.widgets.hovered.bg_stroke = Stroke::NONE;
        style.visuals.widgets.active.bg_stroke = Stroke::NONE;
        style.visuals.widgets.open.bg_stroke = Stroke::NONE;
        style.visuals.widgets.hovered.weak_bg_fill = Color32::from_rgb(0x3a, 0x3a, 0x3a);
        style.visuals.widgets.active.weak_bg_fill = Color32::from_rgb(0x3a, 0x3a, 0x3a);
        style.visuals.widgets.open.weak_bg_fill = Color32::from_rgb(0x3a, 0x3a, 0x3a);
        // A panel draws its resize line from these strokes. Width 0 leaves the
        // gap with no color. The color is the scroll bar, so it stays faint.
        style.visuals.widgets.inactive.fg_stroke.color = FAINT;
        style.visuals.widgets.hovered.fg_stroke.color = FAINT;
        style.visuals.widgets.active.fg_stroke.color = FAINT;
        style.visuals.widgets.hovered.fg_stroke.width = 0.0;
        style.visuals.widgets.active.fg_stroke.width = 0.0;
        let scroll = &mut style.spacing.scroll;
        scroll.bar_width = 3.0;
        scroll.floating_width = 3.0;
        scroll.active_background_opacity = 0.0;
        scroll.interact_background_opacity = 0.0;
    });
}

/// The pointer is on the control, or the button is held.
pub fn hot(response: &egui::Response) -> bool {
    response.hovered() || response.is_pointer_button_down_on()
}

/// Resting colors, or the same colors stepped lighter while `hot`.
/// A clear fill becomes `#2e2e2e` instead of a near-black wash.
pub fn face(fill: Color32, ink: Color32, hot: bool) -> (Color32, Color32) {
    if hot {
        (hover_fill(fill), hover_ink(ink))
    } else {
        (fill, ink)
    }
}

/// Run `add` with a button face that lifts on hover and press.
/// `Button::fill` would freeze the color, so the face lives on the widget style.
pub fn with_face(
    ui: &mut egui::Ui,
    fill: Color32,
    ink: Color32,
    add: impl FnOnce(&mut egui::Ui) -> egui::Response,
) -> egui::Response {
    let previous = Arc::clone(ui.style());
    tune(ui, fill, ink);
    let response = add(ui);
    ui.set_style(previous);
    response
}

/// Rows in the File menu. Clear at rest. A plate and lighter text while hovered.
pub fn tune_menu(ui: &mut egui::Ui) {
    tune(ui, Color32::TRANSPARENT, INK);
}

fn tune(ui: &mut egui::Ui, fill: Color32, ink: Color32) {
    let hot_fill = hover_fill(fill);
    let hot_ink = hover_ink(ink);
    let visuals = ui.visuals_mut();
    paint_face(&mut visuals.widgets.inactive, fill, ink);
    paint_face(&mut visuals.widgets.hovered, hot_fill, hot_ink);
    paint_face(&mut visuals.widgets.active, hot_fill, hot_ink);
    paint_face(&mut visuals.widgets.open, hot_fill, hot_ink);
}

fn paint_face(visual: &mut egui::style::WidgetVisuals, fill: Color32, ink: Color32) {
    visual.weak_bg_fill = fill;
    visual.fg_stroke.color = ink;
    visual.bg_stroke = Stroke::NONE;
}

fn hover_fill(color: Color32) -> Color32 {
    if color.a() == 0 {
        HOVER_PLATE
    } else {
        lift(color, 0x12)
    }
}

fn hover_ink(color: Color32) -> Color32 {
    lift(color, 0x1c)
}

fn lift(color: Color32, by: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(
        color.r().saturating_add(by),
        color.g().saturating_add(by),
        color.b().saturating_add(by),
        color.a(),
    )
}

pub fn bar() -> Frame {
    Frame::NONE.fill(BAR)
}

pub fn panel() -> Frame {
    Frame::NONE.fill(PANEL)
}

pub fn gap_right() -> Frame {
    panel().outer_margin(Margin {
        left: 0,
        right: GAP_PX,
        top: 0,
        bottom: 0,
    })
}

pub fn gap_left() -> Frame {
    panel().outer_margin(Margin {
        left: GAP_PX,
        right: 0,
        top: 0,
        bottom: 0,
    })
}

pub fn gap_above() -> Frame {
    panel().outer_margin(Margin {
        left: 0,
        right: 0,
        top: GAP_PX,
        bottom: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::{BAR, INK, face};

    #[test]
    fn a_hovered_button_steps_lighter_and_a_clear_one_gains_a_plate() {
        let (fill, ink) = face(BAR, INK, true);
        assert!(fill.r() > BAR.r() && fill.g() > BAR.g() && fill.b() > BAR.b());
        assert!(ink.r() > INK.r());
        let (clear, _) = face(eframe::egui::Color32::TRANSPARENT, INK, true);
        assert_eq!(clear, eframe::egui::Color32::from_rgb(0x2e, 0x2e, 0x2e));
        let (rest, rest_ink) = face(BAR, INK, false);
        assert_eq!((rest, rest_ink), (BAR, INK));
    }
}
