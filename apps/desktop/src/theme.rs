//! Chrome colors from `docs/desktop-layout.md`.

use eframe::egui::{self, Color32, Frame, Margin, Stroke};

pub const GAP: Color32 = Color32::from_rgb(0x14, 0x14, 0x14);
pub const PANEL: Color32 = Color32::from_rgb(0x1c, 0x1c, 0x1c);
pub const BAR: Color32 = Color32::from_rgb(0x26, 0x26, 0x26);
pub const INK: Color32 = Color32::from_rgb(0xf3, 0xf3, 0xf3);
pub const EXPORT: Color32 = Color32::from_rgb(0x00, 0xc8, 0xd7);
pub const EXPORT_INK: Color32 = Color32::from_rgb(0x06, 0x22, 0x26);

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
        style.visuals.override_text_color = Some(INK);
        style.visuals.panel_fill = PANEL;
        style.visuals.window_fill = PANEL;
        style.visuals.extreme_bg_color = GAP;
        style.visuals.faint_bg_color = BAR;
        style.visuals.widgets.noninteractive.bg_stroke = Stroke::NONE;
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
