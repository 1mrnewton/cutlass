//! Derived per-clip capabilities: what inspector sections and context-menu
//! actions apply to a clip on a given lane.
//!
//! This is **not** persisted — it is computed from [`Clip`] + [`TrackKind`]
//! and mirrors the engine's edit-command rejection rules. The desktop UI
//! projects these flags onto the Slint clip model; the AI agent can use the
//! same descriptor to pre-validate commands before dispatching them.

use crate::clip::{Clip, ClipSource, Generator};
use crate::track::TrackKind;

/// What a clip supports for inspector panels and context-menu actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ClipCapabilities {
    // --- inspector sections -------------------------------------------------
    /// Spatial transform (position, scale, rotation, opacity).
    pub has_transform: bool,
    /// Crop and flip framing.
    pub has_crop: bool,
    /// Volume, fades, denoise, duck, beats — the audio inspector block.
    pub has_audio: bool,
    /// Constant speed, reverse, and speed ramps (media-backed only).
    pub has_speed: bool,
    /// Text content and styling.
    pub has_text: bool,
    /// Shape size and fill (generator shapes on sticker lanes).
    pub has_shape: bool,
    /// Effect chain sliders.
    pub has_effects: bool,
    /// Filter / adjustment / look properties (mask, chroma, grade).
    pub has_filter_adjust: bool,

    // --- context-menu / edit actions ----------------------------------------
    /// Split at the playhead (all clip kinds).
    pub can_split: bool,
    /// Toggle reverse playback (`SetClipSpeed` with flipped `reversed`).
    pub can_reverse: bool,
    /// Delete and close the gap on the clip's lane (`RippleDelete`).
    pub can_ripple_delete: bool,
}

/// A timeline edit the UI or agent may offer for a selected clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipAction {
    Split,
    Copy,
    Duplicate,
    Delete,
    RippleDelete,
    Reverse,
}

impl ClipCapabilities {
    /// Derive capabilities from the clip's content and the lane it sits on.
    pub fn for_clip(clip: &Clip, kind: TrackKind) -> Self {
        let is_visual = kind.is_visual();
        let is_media = clip.source_range().is_some();

        let (has_text, has_shape) = match &clip.content {
            ClipSource::Generated(Generator::Text { .. }) => (true, false),
            ClipSource::Generated(Generator::Shape { .. }) => (false, true),
            _ => (false, false),
        };

        Self {
            has_transform: is_visual,
            has_crop: is_visual,
            has_audio: kind == TrackKind::Audio,
            has_speed: is_media,
            has_text,
            has_shape,
            has_effects: is_visual,
            has_filter_adjust: is_visual,
            can_split: true,
            can_reverse: is_media,
            can_ripple_delete: true,
        }
    }

    /// Whether this clip kind allows `action`. Selection-scoped actions
    /// (`Copy`, `Duplicate`, `Delete`) are always permitted when a clip is
    /// selected — gating happens UI-side on selection emptiness.
    pub fn allows(self, action: ClipAction) -> bool {
        match action {
            ClipAction::Split => self.can_split,
            ClipAction::Reverse => self.can_reverse,
            ClipAction::RippleDelete => self.can_ripple_delete,
            ClipAction::Copy | ClipAction::Duplicate | ClipAction::Delete => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::{Clip, Generator, Shape};
    use crate::ids::MediaId;
    use crate::time::{Rational, TimeRange};

    const R24: Rational = Rational::FPS_24;

    fn tr(start: i64, duration: i64) -> TimeRange {
        TimeRange::at_rate(start, duration, R24)
    }

    fn media_clip() -> Clip {
        Clip::from_media(MediaId::from_raw(1), tr(0, 48), tr(0, 48))
    }

    fn assert_caps(clip: &Clip, kind: TrackKind, expect: ClipCapabilities) {
        assert_eq!(ClipCapabilities::for_clip(clip, kind), expect);
    }

    #[test]
    fn media_video_clip_caps() {
        let clip = media_clip();
        assert_caps(
            &clip,
            TrackKind::Video,
            ClipCapabilities {
                has_transform: true,
                has_crop: true,
                has_audio: false,
                has_speed: true,
                has_text: false,
                has_shape: false,
                has_effects: true,
                has_filter_adjust: true,
                can_split: true,
                can_reverse: true,
                can_ripple_delete: true,
            },
        );
    }

    #[test]
    fn media_audio_clip_caps() {
        let clip = media_clip();
        assert_caps(
            &clip,
            TrackKind::Audio,
            ClipCapabilities {
                has_transform: false,
                has_crop: false,
                has_audio: true,
                has_speed: true,
                has_text: false,
                has_shape: false,
                has_effects: false,
                has_filter_adjust: false,
                can_split: true,
                can_reverse: true,
                can_ripple_delete: true,
            },
        );
    }

    #[test]
    fn text_clip_caps() {
        let clip = Clip::generated(Generator::text("Hello"), tr(0, 48));
        assert_caps(
            &clip,
            TrackKind::Text,
            ClipCapabilities {
                has_transform: true,
                has_crop: true,
                has_audio: false,
                has_speed: false,
                has_text: true,
                has_shape: false,
                has_effects: true,
                has_filter_adjust: true,
                can_split: true,
                can_reverse: false,
                can_ripple_delete: true,
            },
        );
    }

    #[test]
    fn shape_clip_caps() {
        let clip = Clip::generated(
            Generator::shape(Shape::Rectangle, [255, 0, 0, 255]),
            tr(0, 48),
        );
        assert_caps(
            &clip,
            TrackKind::Sticker,
            ClipCapabilities {
                has_transform: true,
                has_crop: true,
                has_audio: false,
                has_speed: false,
                has_text: false,
                has_shape: true,
                has_effects: true,
                has_filter_adjust: true,
                can_split: true,
                can_reverse: false,
                can_ripple_delete: true,
            },
        );
    }

    #[test]
    fn solid_and_adjustment_caps() {
        let solid = Clip::generated(
            Generator::SolidColor {
                rgba: [0, 0, 0, 255],
            },
            tr(0, 24),
        );
        assert_caps(
            &solid,
            TrackKind::Sticker,
            ClipCapabilities {
                has_transform: true,
                has_crop: true,
                has_effects: true,
                has_filter_adjust: true,
                can_split: true,
                can_ripple_delete: true,
                ..Default::default()
            },
        );

        let adj = Clip::generated(Generator::Adjustment, tr(0, 24));
        assert_caps(
            &adj,
            TrackKind::Adjustment,
            ClipCapabilities {
                has_transform: true,
                has_crop: true,
                has_effects: true,
                has_filter_adjust: true,
                can_split: true,
                can_reverse: false,
                can_ripple_delete: true,
                ..Default::default()
            },
        );
    }

    #[test]
    fn allows_action_matches_flags() {
        let video = ClipCapabilities::for_clip(&media_clip(), TrackKind::Video);
        assert!(video.allows(ClipAction::Reverse));
        assert!(video.allows(ClipAction::Split));

        let text = ClipCapabilities::for_clip(
            &Clip::generated(Generator::text("Hi"), tr(0, 24)),
            TrackKind::Text,
        );
        assert!(!text.allows(ClipAction::Reverse));
        assert!(text.allows(ClipAction::Copy));
    }
}
