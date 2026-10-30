//! Colors, type, and chrome for the desktop window.
//!
//! Panels read these values. A panel does not pick its own color.

use egui::Color32;

/// The gap between regions, and the hairline inside a region.
pub const GAP: Color32 = Color32::from_rgb(0x14, 0x14, 0x14);
/// Library, preview, inspector, and timeline.
pub const PANEL: Color32 = Color32::from_rgb(0x1c, 0x1c, 0x1c);
/// The project bar and the dialog body.
pub const PROJECT_BAR: Color32 = Color32::from_rgb(0x26, 0x26, 0x26);
/// The export control.
pub const EXPORT: Color32 = Color32::from_rgb(0x00, 0xc8, 0xd7);
/// Ink on the export control.
pub const EXPORT_INK: Color32 = Color32::from_rgb(0x06, 0x22, 0x26);
/// Primary text.
pub const TEXT: Color32 = Color32::from_rgb(0xf3, 0xf3, 0xf3);
/// Secondary text.
pub const MUTED: Color32 = Color32::from_rgb(0xa3, 0xa3, 0xa3);
/// Tertiary text.
pub const FAINT: Color32 = Color32::from_rgb(0x73, 0x73, 0x73);
/// The playhead.
pub const PLAYHEAD: Color32 = Color32::WHITE;
/// An icon button.
pub const ICON_BUTTON: Color32 = Color32::from_rgb(0x2a, 0x2a, 0x2a);
/// A selected clip edge.
pub const SELECTED: Color32 = Color32::WHITE;
/// The dialog title row.
pub const DIALOG_HEAD: Color32 = Color32::from_rgb(0x32, 0x32, 0x32);
/// Close, minimize, and zoom, in that order.
pub const CLOSE: Color32 = Color32::from_rgb(0xff, 0x5f, 0x57);
pub const MINIMIZE: Color32 = Color32::from_rgb(0xfe, 0xbc, 0x2e);
pub const MAXIMIZE: Color32 = Color32::from_rgb(0x28, 0xc8, 0x40);
/// The scrim behind a dialog.
pub const SCRIM: Color32 = Color32::from_black_alpha(140);

pub const TYPE_SIZE: f32 = 13.0;
pub const TIMECODE_SIZE: f32 = 12.0;
pub const EXPORT_RADIUS: u8 = 8;
pub const EXPORT_PAD_X: i8 = 12;
pub const EXPORT_PAD_Y: i8 = 5;
pub const DIALOG_RADIUS: u8 = 12;
pub const ICON_RADIUS: u8 = 6;
pub const SELECTED_EDGE: f32 = 1.0;
pub const STROKE: f32 = 1.5;

pub const HEAD_HEIGHT: f32 = 38.0;
pub const SPLIT: f32 = 6.0;
pub const HAIRLINE: f32 = 1.0;
pub const TOOLBAR_HEIGHT: f32 = 36.0;
pub const RULER_HEIGHT: f32 = 44.0;
pub const RAIL_WIDTH: f32 = 44.0;
pub const COVER_WIDTH: f32 = 72.0;
pub const TRANSPORT_HEIGHT: f32 = 40.0;
pub const ICON_BUTTON_SIZE: f32 = 28.0;
pub const WINDOW_BUTTON: f32 = 12.0;

pub const LIBRARY_WIDTH: f32 = 280.0;
pub const INSPECTOR_WIDTH: f32 = 300.0;
pub const TIMELINE_HEIGHT: f32 = 280.0;
pub const LIBRARY_MIN: f32 = 160.0;
pub const INSPECTOR_MIN: f32 = 200.0;
pub const TIMELINE_MIN: f32 = 160.0;
pub const PREVIEW_MIN_WIDTH: f32 = 320.0;
pub const PREVIEW_MIN_HEIGHT: f32 = 180.0;
pub const MIN_WINDOW_WIDTH: f32 = 1100.0;
pub const MIN_WINDOW_HEIGHT: f32 = 700.0;

/// Lane height shared by drawing and hit testing.
pub const LANE_HEIGHT: f64 = 48.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_desktop_tokens_match_the_picture() {
        assert_eq!(GAP, Color32::from_rgb(0x14, 0x14, 0x14));
        assert_eq!(PANEL, Color32::from_rgb(0x1c, 0x1c, 0x1c));
        assert_eq!(PROJECT_BAR, Color32::from_rgb(0x26, 0x26, 0x26));
        assert_eq!(EXPORT, Color32::from_rgb(0x00, 0xc8, 0xd7));
        assert_eq!(EXPORT_INK, Color32::from_rgb(0x06, 0x22, 0x26));
        assert_eq!(TEXT, Color32::from_rgb(0xf3, 0xf3, 0xf3));
        assert_eq!(MUTED, Color32::from_rgb(0xa3, 0xa3, 0xa3));
        assert_eq!(FAINT, Color32::from_rgb(0x73, 0x73, 0x73));
        assert_eq!(PLAYHEAD, Color32::WHITE);
        assert_eq!(ICON_BUTTON, Color32::from_rgb(0x2a, 0x2a, 0x2a));
        assert_eq!(SELECTED, Color32::WHITE);
        assert_eq!(TYPE_SIZE, 13.0);
        assert_eq!(TIMECODE_SIZE, 12.0);
        assert_eq!(EXPORT_RADIUS, 8);
        assert_eq!((EXPORT_PAD_X, EXPORT_PAD_Y), (12, 5));
        assert_eq!(DIALOG_RADIUS, 12);
        assert_eq!(DIALOG_HEAD, Color32::from_rgb(0x32, 0x32, 0x32));
        assert_eq!(SPLIT, 6.0);
        assert_eq!(SELECTED_EDGE, 1.0);
        assert_eq!(STROKE, 1.5);
    }
}
