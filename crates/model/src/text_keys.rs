//! A text's style settings that key on their own: its size, spacing, wrap
//! width, outline, background and shadow. Each is one number on the text
//! material, keyed on the segment under a Cutlass name.

use crate::keyframe::{KeyframeGraph, sample};
use crate::materials::MaterialText;
use crate::segment::Segment;
use crate::time::Micros;

/// The key names, in [`TextStyleKey::ALL`] order.
pub static TEXT_STYLE_KEYS: [&str; 10] = [
    "KFTypeTextSize",
    "KFTypeTextLetterSpacing",
    "KFTypeTextLineSpacing",
    "KFTypeTextLineWidth",
    "KFTypeTextStrokeWidth",
    "KFTypeTextBackgroundAlpha",
    "KFTypeTextBackgroundRound",
    "KFTypeTextShadowDistance",
    "KFTypeTextShadowAngle",
    "KFTypeTextShadowBlur",
];

/// One keyed text style setting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextStyleKey {
    /// The size in pixels.
    Size,
    /// Extra space between letters, as a share of the size.
    LetterSpacing,
    /// Extra space between lines, as a share of the size.
    LineSpacing,
    /// The wrap width in pixels.
    LineWidth,
    /// The outline's width in ems.
    StrokeWidth,
    /// The background box's opacity.
    BackgroundAlpha,
    /// How round the background box's corners are.
    BackgroundRound,
    /// The shadow's distance in hundredths of an em.
    ShadowDistance,
    /// The shadow's angle in degrees.
    ShadowAngle,
    /// The shadow's blur.
    ShadowBlur,
}

impl TextStyleKey {
    /// Every setting.
    pub const ALL: [Self; 10] = [
        Self::Size,
        Self::LetterSpacing,
        Self::LineSpacing,
        Self::LineWidth,
        Self::StrokeWidth,
        Self::BackgroundAlpha,
        Self::BackgroundRound,
        Self::ShadowDistance,
        Self::ShadowAngle,
        Self::ShadowBlur,
    ];

    fn index(self) -> usize {
        Self::ALL
            .iter()
            .position(|key| *key == self)
            .unwrap_or_default()
    }

    /// The one key name the setting is stored under.
    #[must_use]
    pub fn names(self) -> &'static [&'static str] {
        std::slice::from_ref(&TEXT_STYLE_KEYS[self.index()])
    }

    /// The setting a key name stores, when it is one.
    #[must_use]
    pub fn of(property: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|key| key.names()[0] == property)
    }

    /// Whether `value` is one the setting's edit takes.
    #[must_use]
    pub fn takes(self, value: f64) -> bool {
        if !value.is_finite() {
            return false;
        }
        match self {
            Self::Size => value > 0.0 && value <= 512.0,
            Self::LetterSpacing => (-1.0..=4.0).contains(&value),
            Self::LineSpacing => value > -1.0 && value <= 4.0,
            Self::LineWidth => (0.0..=8_192.0).contains(&value),
            Self::StrokeWidth => (0.0..=0.25).contains(&value),
            Self::BackgroundAlpha | Self::BackgroundRound | Self::ShadowBlur => {
                (0.0..=1.0).contains(&value)
            }
            Self::ShadowDistance => (0.0..=100.0).contains(&value),
            Self::ShadowAngle => true,
        }
    }

    /// The setting's value on `text`.
    #[must_use]
    pub fn get(self, text: &MaterialText) -> f64 {
        match self {
            Self::Size => text.font_size,
            Self::LetterSpacing => text.letter_spacing,
            Self::LineSpacing => text.line_spacing,
            Self::LineWidth => text.line_max_width,
            Self::StrokeWidth => text.border_width,
            Self::BackgroundAlpha => text.background_alpha,
            Self::BackgroundRound => text.background_round_radius,
            Self::ShadowDistance => text.shadow_distance,
            Self::ShadowAngle => text.shadow_angle,
            Self::ShadowBlur => text.shadow_smoothing,
        }
    }

    /// Set the setting on `text` to `value`. A value the edit would refuse
    /// is held to the nearest one it takes.
    pub fn set(self, text: &mut MaterialText, value: f64) {
        let value = match self {
            Self::Size => value.clamp(1.0, 512.0),
            Self::LetterSpacing => value.clamp(-1.0, 4.0),
            Self::LineSpacing => value.clamp(-0.99, 4.0),
            Self::LineWidth => value.clamp(0.0, 8_192.0),
            Self::StrokeWidth => value.clamp(0.0, 0.25),
            Self::BackgroundAlpha | Self::BackgroundRound | Self::ShadowBlur => {
                value.clamp(0.0, 1.0)
            }
            Self::ShadowDistance => value.clamp(0.0, 100.0),
            Self::ShadowAngle => value,
        };
        let field = match self {
            Self::Size => &mut text.font_size,
            Self::LetterSpacing => &mut text.letter_spacing,
            Self::LineSpacing => &mut text.line_spacing,
            Self::LineWidth => &mut text.line_max_width,
            Self::StrokeWidth => &mut text.border_width,
            Self::BackgroundAlpha => &mut text.background_alpha,
            Self::BackgroundRound => &mut text.background_round_radius,
            Self::ShadowDistance => &mut text.shadow_distance,
            Self::ShadowAngle => &mut text.shadow_angle,
            Self::ShadowBlur => &mut text.shadow_smoothing,
        };
        *field = value;
    }
}

/// `text` with each style setting `segment` keys set as keyed at timeline
/// `time`; `None` when it keys none, so the stored text can be drawn and
/// kept as it is.
#[must_use]
pub fn text_style_at(
    text: &MaterialText,
    segment: &Segment,
    graphs: &[KeyframeGraph],
    time: Micros,
) -> Option<MaterialText> {
    if !segment
        .common_keyframes
        .iter()
        .any(|group| TEXT_STYLE_KEYS.contains(&group.property_type.as_str()))
    {
        return None;
    }
    let offset = time.0.saturating_sub(segment.target.start.0);
    let mut keyed = text.clone();
    for key in TextStyleKey::ALL {
        if let Some(Some(value)) = sample(&segment.common_keyframes, graphs, key.names(), offset) {
            key.set(&mut keyed, value);
        }
    }
    Some(keyed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keyframe::{is_keyable, put_key};
    use crate::time::TimeRange;

    #[test]
    fn a_keyed_style_moves_along_the_clip_and_leaves_the_rest() {
        let text = MaterialText {
            font_size: 40.0,
            shadow_angle: -45.0,
            ..MaterialText::default()
        };
        let mut segment = Segment::new("s", "t", TimeRange::new(1_000, 2_000));
        assert_eq!(text_style_at(&text, &segment, &[], Micros(1_000)), None);
        let size = TextStyleKey::Size.names()[0];
        put_key(&mut segment, size, Micros(0), 20.0);
        put_key(&mut segment, size, Micros(2_000), 60.0);
        let keyed = text_style_at(&text, &segment, &[], Micros(2_000)).unwrap();
        assert_eq!(keyed.font_size, 40.0);
        assert_eq!(keyed.shadow_angle, -45.0);
        let start = text_style_at(&text, &segment, &[], Micros(1_000)).unwrap();
        assert_eq!(start.font_size, 20.0);
        // A key the edit would refuse is held to its range.
        put_key(
            &mut segment,
            TextStyleKey::ShadowBlur.names()[0],
            Micros(0),
            3.0,
        );
        let held = text_style_at(&text, &segment, &[], Micros(1_000)).unwrap();
        assert_eq!(held.shadow_smoothing, 1.0);
    }

    #[test]
    fn every_setting_has_its_own_name_range_and_field() {
        for key in TextStyleKey::ALL {
            let name = key.names()[0];
            assert_eq!(TextStyleKey::of(name), Some(key));
            assert!(is_keyable(name), "{name}");
            assert!(!key.takes(f64::NAN), "{name}");
            let mut text = MaterialText::default();
            let value = if key == TextStyleKey::Size {
                30.0
            } else {
                0.125
            };
            assert!(key.takes(value), "{name}");
            key.set(&mut text, value);
            assert_eq!(key.get(&text), value, "{name}");
            for other in TextStyleKey::ALL.into_iter().filter(|other| *other != key) {
                assert_eq!(
                    other.get(&text),
                    other.get(&MaterialText::default()),
                    "{name}"
                );
            }
        }
        assert!(!TextStyleKey::Size.takes(0.0));
        assert!(!TextStyleKey::LineSpacing.takes(-1.0));
        assert!(TextStyleKey::ShadowAngle.takes(720.0));
        assert_eq!(TextStyleKey::of("KFTypeTextColorR"), None);
    }
}
