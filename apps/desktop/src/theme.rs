//! Chrome colors from `docs/desktop-layout.md`.

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
        // Hover is a light frame and press is a white one. The shell paints its own fills.
        style.visuals.widgets.inactive.bg_stroke = Stroke::NONE;
        style.visuals.widgets.hovered.bg_stroke = Stroke::NONE;
        style.visuals.widgets.active.bg_stroke = Stroke::NONE;
        style.visuals.widgets.open.bg_stroke = Stroke::NONE;
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
