use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::clip::Clip;
use super::color::Color;

/// Whether a lane carries video or audio clips. Drives two invariants:
///
///   * Visual placement — audio lanes are always rendered **below** all
///     video lanes (`Sequence::track_order` is kept sorted so every
///     `Video` track precedes every `Audio` track).
///   * Cross-lane drops — clips can only be moved to a lane of the
///     same kind. The gesture layer enforces this when picking a drop
///     target; the command layer rejects mismatched targets defensively.
///
/// New kinds (`Text`, `Caption`, …) can be added later; they will need
/// a slot in [`TrackKind::default_palette`] and a position in the
/// "video at top, audio at bottom" ordering rule.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrackKind {
    #[default]
    Video,
    Audio,
}

/// Cool tones for video — blue, teal, indigo, slate, lavender.
/// Hoisted to a `const` (not a fresh array in `match`) so the
/// returned slice has `'static` lifetime.
const VIDEO_PALETTE: &[Color] = &[
    Color::rgb(0x4A, 0x6F, 0xA5),
    Color::rgb(0x5E, 0x8B, 0x7E),
    Color::rgb(0x6C, 0x5B, 0x7B),
    Color::rgb(0x54, 0x7A, 0x8F),
    Color::rgb(0x7D, 0x8A, 0xA8),
];

/// Warm tones for audio — amber, terracotta, ochre, coral.
const AUDIO_PALETTE: &[Color] = &[
    Color::rgb(0xC9, 0x98, 0x46),
    Color::rgb(0xBF, 0x6F, 0x4A),
    Color::rgb(0xA6, 0x6B, 0x5F),
    Color::rgb(0xC7, 0x7F, 0x4D),
];

impl TrackKind {
    /// Per-kind clip-color palette. New lanes spawned at edit time
    /// pick from here so they look visually distinct without forcing
    /// the user to assign a color themselves. The agent can override
    /// by emitting a future `SetTrackColor` command.
    ///
    /// Palettes intentionally don't overlap between kinds — at a
    /// glance, cool tones read as video and warm tones as audio.
    pub fn default_palette(self) -> &'static [Color] {
        match self {
            TrackKind::Video => VIDEO_PALETTE,
            TrackKind::Audio => AUDIO_PALETTE,
        }
    }

    /// Color for the `n`-th lane of this kind (zero-based). Wraps
    /// around so adding a sixth video lane reuses the first video
    /// color — visually fine in practice, and we'd rather repeat a
    /// hue than ship a brown lane.
    pub fn palette_color(self, index: usize) -> Color {
        let palette = self.default_palette();
        // `default_palette()` is hand-authored above and is never
        // empty, but the cheap guard keeps a future regression from
        // panicking the editor.
        if palette.is_empty() {
            return Color::rgb(0x37, 0x37, 0x37);
        }
        palette[index % palette.len()]
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Track {
    pub id: String,
    pub name: String,
    pub kind: TrackKind,
    /// Color shared by every clip on this lane. Choosing the color
    /// per-lane (not per-clip) makes the invariant the user expects —
    /// "one type, one color" — true by construction: a clip's color
    /// changes when it moves to a different lane.
    pub color: Color,
    /// Stable iteration order of clips within this lane. Updated by
    /// structural commands (insert/remove); intra-lane reposition does
    /// not touch this vector.
    pub clip_order: Vec<String>,
    pub clips: HashMap<String, Clip>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_and_audio_palettes_dont_overlap() {
        let v: std::collections::HashSet<_> =
            TrackKind::Video.default_palette().iter().collect();
        let a: std::collections::HashSet<_> =
            TrackKind::Audio.default_palette().iter().collect();
        assert!(
            v.intersection(&a).next().is_none(),
            "video/audio palettes overlap — colors should disambiguate kind at a glance",
        );
    }

    #[test]
    fn palette_color_wraps_on_index_overflow() {
        let first = TrackKind::Video.palette_color(0);
        let wrapped = TrackKind::Video
            .palette_color(TrackKind::Video.default_palette().len());
        assert_eq!(first, wrapped);
    }
}
