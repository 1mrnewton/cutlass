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
/// A ruler tick.
pub const RULER_TICK: Color32 = Color32::from_rgb(0x5a, 0x5a, 0x5a);
/// The playhead.
pub const PLAYHEAD: Color32 = Color32::WHITE;
/// An icon button, and a toolbar button that is on.
pub const ICON_BUTTON: Color32 = Color32::from_rgb(0x2a, 0x2a, 0x2a);
/// The active library shelf.
pub const ACTIVE: Color32 = Color32::from_rgb(0x33, 0x33, 0x33);
/// The main lane.
pub const LANE: Color32 = Color32::from_rgb(0x24, 0x24, 0x24);
/// The empty main lane edge.
pub const LANE_LINE: Color32 = Color32::from_rgb(0x2e, 0x2e, 0x2e);
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
/// A major ruler tick. A minor tick is half as tall.
pub const MAJOR_TICK: f32 = 12.0;
pub const RAIL_WIDTH: f32 = 76.0;
pub const COVER_WIDTH: f32 = 72.0;
pub const TRACK_HEAD: f32 = 130.0;
pub const TRANSPORT_HEIGHT: f32 = 52.0;
pub const ICON_BUTTON_SIZE: f32 = 28.0;
pub const WINDOW_BUTTON: f32 = 12.0;

pub const LIBRARY_WIDTH: f32 = 368.0;
pub const INSPECTOR_WIDTH: f32 = 300.0;
pub const TIMELINE_HEIGHT: f32 = 420.0;
pub const LIBRARY_MIN: f32 = 260.0;
pub const INSPECTOR_MIN: f32 = 240.0;
pub const TIMELINE_MIN: f32 = 160.0;
pub const PREVIEW_MIN_WIDTH: f32 = 320.0;
pub const PREVIEW_MIN_HEIGHT: f32 = 268.0;
pub const MIN_WINDOW_WIDTH: f32 = 1100.0;
pub const MIN_WINDOW_HEIGHT: f32 = 700.0;

/// Lane height shared by drawing and hit testing.
pub const LANE_HEIGHT: f64 = 48.0;

/// A video clip, measured from CapCut 9.6.0: its body, the name and length
/// chips, the waveform bars, and the volume line across them.
pub const CLIP_VIDEO: Color32 = Color32::from_rgb(30, 76, 81);
pub const CLIP_VIDEO_CHIP: Color32 = Color32::from_rgb(47, 94, 99);
pub const CLIP_VIDEO_BAR: Color32 = Color32::from_rgb(51, 117, 125);
pub const CLIP_VIDEO_LINE: Color32 = Color32::from_rgb(80, 122, 126);
/// An audio clip, the same parts in CapCut's navy.
pub const CLIP_AUDIO: Color32 = Color32::from_rgb(29, 54, 94);
pub const CLIP_AUDIO_CHIP: Color32 = Color32::from_rgb(45, 72, 114);
pub const CLIP_AUDIO_BAR: Color32 = Color32::from_rgb(58, 130, 200);
pub const CLIP_AUDIO_LINE: Color32 = Color32::from_rgb(100, 138, 186);
/// The text in a clip's chips.
pub const CLIP_TEXT: Color32 = Color32::from_rgb(224, 224, 224);
/// The row across the top of a main-track or audio clip that holds its
/// name chip, and on the main track its length chip.
pub const CLIP_HEADER: f32 = 18.0;

/// The cap on a loud waveform bar, and the tip on one that clips.
pub const WAVE_LOUD: Color32 = Color32::from_rgb(226, 143, 72);
pub const WAVE_CLIPPED: Color32 = Color32::from_rgb(215, 68, 50);

/// A main-track clip, CapCut's 76 track height less its border: a header
/// row, then the filmstrip over the sound band.
pub const MAIN_CLIP_HEIGHT: f32 = 74.0;
/// An audio clip: a header row over its waveform.
pub const AUDIO_CLIP_HEIGHT: f32 = 46.0;
/// Any other clip.
pub const CLIP_HEIGHT: f32 = 28.0;
/// The gap above and below a main-track clip inside its lane row.
pub const MAIN_CLIP_INSET: f32 = 8.0;
/// The gap above and below any other clip.
pub const CLIP_INSET: f32 = 6.0;

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
        assert_eq!(ACTIVE, Color32::from_rgb(0x33, 0x33, 0x33));
        assert_eq!(LANE, Color32::from_rgb(0x24, 0x24, 0x24));
        assert_eq!(LANE_LINE, Color32::from_rgb(0x2e, 0x2e, 0x2e));
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
