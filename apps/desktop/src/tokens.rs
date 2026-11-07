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
/// The export control under the pointer.
pub const EXPORT_HOVER: Color32 = Color32::from_rgb(0x2a, 0xd4, 0xe0);
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
/// A card, bubble, or field that sits on a panel.
pub const RAISED: Color32 = Color32::from_rgb(0x2a, 0x2a, 0x2a);
/// A well sunk into a panel, such as the inspector's page bar.
pub const INSET: Color32 = Color32::from_rgb(0x26, 0x26, 0x26);
/// A button held down.
pub const PRESSED: Color32 = Color32::from_rgb(0x3a, 0x3a, 0x3a);
/// The faint lane a scroll thumb runs in, shown with the thumb.
pub const SCROLL_TRACK: Color32 = Color32::from_rgba_premultiplied(10, 10, 10, 10);
/// A scroll thumb at rest: see-through, so it sits on any panel.
pub const SCROLL_THUMB: Color32 = Color32::from_rgba_premultiplied(56, 56, 56, 56);
/// A scroll thumb under the pointer.
pub const SCROLL_THUMB_HOVER: Color32 = Color32::from_rgba_premultiplied(96, 96, 96, 96);
/// A scroll thumb being dragged.
pub const SCROLL_THUMB_DRAG: Color32 = Color32::from_rgba_premultiplied(136, 136, 136, 136);
/// The ruler's band between the in and out marks.
/// A text field's fill, and its edge when hovered and when typing in it.
pub const FIELD: Color32 = RAISED;
pub const FIELD_EDGE_HOVER: Color32 = Color32::from_rgb(0x3a, 0x3a, 0x3a);
pub const FIELD_EDGE_FOCUS: Color32 = Color32::from_rgb(0x6a, 0x6a, 0x6a);
/// Selected text in a field: the accent, see-through.
pub const TEXT_SELECTION: Color32 = Color32::from_rgba_premultiplied(0x00, 0x4a, 0x50, 0x60);
/// A choice's fill when it is the one picked, and when hovered.
pub const CHOICE_ON: Color32 = Color32::from_rgb(0x3a, 0x3a, 0x3a);
pub const CHOICE_HOVER: Color32 = Color32::from_rgb(0x30, 0x30, 0x30);
/// A switch's track when off, and when off and hovered.
pub const SWITCH_OFF: Color32 = Color32::from_rgb(0x3a, 0x3a, 0x3a);
pub const SWITCH_OFF_HOVER: Color32 = Color32::from_rgb(0x48, 0x48, 0x48);
pub const MARKED: Color32 = Color32::from_rgba_premultiplied(0x00, 0x32, 0x36, 0x40);

pub const TYPE_SIZE: f32 = 13.0;
pub const TIMECODE_SIZE: f32 = 12.0;
pub const EXPORT_RADIUS: u8 = 8;
pub const EXPORT_PAD_X: i8 = 12;
pub const EXPORT_PAD_Y: i8 = 5;
/// Corner radii, smallest first. A dialog uses `DIALOG_RADIUS`.
pub const RADIUS_SM: u8 = 6;
pub const RADIUS_MD: u8 = 8;
pub const RADIUS_LG: u8 = 10;
pub const DIALOG_RADIUS: u8 = 12;
pub const ICON_RADIUS: u8 = 6;
pub const SELECTED_EDGE: f32 = 1.0;
pub const STROKE: f32 = 1.5;

/// A scroll bar is this wide and stays this wide; hover changes only its color.
pub const SCROLL_BAR: f32 = 6.0;
/// The gap between a scroll bar's ends and the ends of its area.
pub const SCROLL_INSET: f32 = 2.0;
/// The gap between a scroll bar and the side of its area: none, so the bar
/// sits on the panel's edge.
pub const SCROLL_EDGE: f32 = 0.0;
/// The gap between the thumb and the sides of its track.
pub const SCROLL_THUMB_INSET: f32 = 1.0;
/// Seconds a scroll bar takes to fade in or out. Only its opacity changes.
pub const SCROLL_FADE: f32 = 0.15;
/// Seconds a scroll bar stays after the content last moved.
pub const SCROLL_LINGER: f64 = 0.8;
/// The shortest a scroll thumb gets on long content.
pub const SCROLL_MIN_THUMB: f32 = 24.0;

/// A text button's padding, and the least height of any button.
pub const BUTTON_PAD_X: f32 = 10.0;
pub const BUTTON_PAD_Y: f32 = 4.0;
pub const BUTTON_HEIGHT: f32 = 24.0;
/// The gap between a button's icon and its label.
pub const BUTTON_GAP: f32 = 6.0;

/// A text field's padding and its one-line height.
pub const FIELD_PAD_X: f32 = 8.0;
pub const FIELD_PAD_Y: f32 = 5.0;
pub const FIELD_HEIGHT: f32 = 26.0;
/// A choice's size and the well around a row of them.
pub const CHOICE_HEIGHT: f32 = 22.0;
pub const CHOICE_PAD_X: f32 = 10.0;
pub const CHOICE_TEXT: f32 = 12.0;
pub const SEGMENT_PAD: i8 = 2;
pub const SEGMENT_GAP: f32 = 2.0;
/// Check boxes and switches: sizes, the gap to a label, and the row they
/// sit in.
pub const CHECK_SIZE: f32 = 16.0;
pub const CHECK_RADIUS: u8 = 4;
pub const TOGGLE_GAP: f32 = 8.0;
pub const TOGGLE_ROW: f32 = 22.0;
pub const SWITCH_WIDTH: f32 = 32.0;
pub const SWITCH_HEIGHT: f32 = 18.0;
pub const SWITCH_KNOB_INSET: f32 = 2.0;
/// Seconds for a switch's knob to slide across.
pub const SWITCH_SLIDE: f32 = 0.12;
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

/// A video clip: its body, the name and length
/// chips, the waveform bars, and the volume line across them.
pub const CLIP_VIDEO: Color32 = Color32::from_rgb(30, 76, 81);
pub const CLIP_VIDEO_CHIP: Color32 = Color32::from_rgb(47, 94, 99);
pub const CLIP_VIDEO_BAR: Color32 = Color32::from_rgb(51, 117, 125);
pub const CLIP_VIDEO_LINE: Color32 = Color32::from_rgb(80, 122, 126);
/// An audio clip, the same parts in navy.
pub const CLIP_AUDIO: Color32 = Color32::from_rgb(29, 54, 94);
pub const CLIP_AUDIO_CHIP: Color32 = Color32::from_rgb(45, 72, 114);
pub const CLIP_AUDIO_BAR: Color32 = Color32::from_rgb(58, 130, 200);
pub const CLIP_AUDIO_LINE: Color32 = Color32::from_rgb(100, 138, 186);
/// The text in a clip's chips.
pub const CLIP_TEXT: Color32 = Color32::from_rgb(224, 224, 224);
/// The row across the top of a main-track or audio clip that holds its
/// name chip, and on the main track its length chip.
pub const CLIP_HEADER: f32 = 18.0;

/// The share of a main-track clip's body under the header that its sound
/// band takes: the stored `wave_ratio` fallback, 30.
pub const WAVE_RATIO: f32 = 0.3;

/// The cap on a loud waveform bar, and the tip on one that clips.
pub const WAVE_LOUD: Color32 = Color32::from_rgb(226, 143, 72);
pub const WAVE_CLIPPED: Color32 = Color32::from_rgb(215, 68, 50);

/// A main-track clip, the 76 track height less its border: a header
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
/// The level meter's column at the timeline's right edge.
pub const METER_WIDTH: f32 = 40.0;
/// The meter's bars: quiet, loud (from -12 dB), and near full scale (from
/// -3 dB), and the mark a clipped channel shows.
pub const METER_LOW: Color32 = Color32::from_rgb(0x3d, 0xd6, 0x8c);
pub const METER_LOUD: Color32 = Color32::from_rgb(0xf5, 0xc5, 0x42);
pub const METER_HOT: Color32 = Color32::from_rgb(0xff, 0x5f, 0x57);
/// The shortest a clip gets in a row dragged short.
pub const MIN_CLIP_HEIGHT: f32 = 12.0;
/// How short and how tall a track row can be dragged.
pub const MIN_LANE_HEIGHT: f32 = 24.0;
pub const MAX_LANE_HEIGHT: f32 = 240.0;

/// A right-click menu.
pub const MENU: Color32 = Color32::from_rgb(0x2c, 0x2c, 0x2c);
/// The menu row under the pointer, and the row whose submenu is open.
pub const MENU_HOVER: Color32 = Color32::from_rgb(0x3a, 0x3a, 0x3a);
/// A menu row that cannot run now.
pub const MENU_DISABLED: Color32 = Color32::from_rgb(0x6f, 0x6f, 0x6f);
/// A menu shortcut.
pub const MENU_KEY: Color32 = Color32::from_rgb(0x8d, 0x8d, 0x8d);
/// The shortcut of a row that cannot run now.
pub const MENU_KEY_DISABLED: Color32 = Color32::from_rgb(0x5c, 0x5c, 0x5c);
/// The arrow of a row with a submenu.
pub const MENU_CHEVRON: Color32 = Color32::from_rgb(0xa3, 0xa3, 0xa3);
/// The line between menu groups.
pub const MENU_SEPARATOR: Color32 = Color32::from_rgba_premultiplied(20, 20, 20, 20);
/// The badge on a menu row this build cannot do yet.
pub const MENU_BADGE: Color32 = Color32::from_rgb(0x6d, 0x5e, 0xfc);
/// The narrowest a menu is drawn.
pub const MENU_WIDTH: f32 = 248.0;

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
