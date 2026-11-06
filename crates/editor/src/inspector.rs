//! The inspector model every shell shares: which panel a clip opens, how a
//! title's outline, shadow, box and glow read, volume in decibels, letter
//! case, the canvas-pixel units the position fields show, and the title
//! style presets.

use cutlass_model::{
    Draft, MaterialEffect, MaterialRef, MaterialText, Segment, Track, TrackFlag, TrackType,
    segment_bloom,
};

/// The panel a clip opens in the inspector, from its material and its row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Panel {
    /// A picture on the main track. `still` is a photo or a color clip.
    Main {
        still: bool,
    },
    /// A picture on an overlay row.
    Overlay {
        still: bool,
    },
    Audio,
    Text,
    /// A text clip with recognized words, or one on a caption or lyric row.
    Caption,
    Effect,
    Filter,
    Adjust,
    Sticker,
    /// A clip this build has no panel for.
    Other,
}

impl Panel {
    #[must_use]
    pub fn title(self) -> &'static str {
        match self {
            Self::Main { still: false } => "Video",
            Self::Main { still: true } => "Photo",
            Self::Overlay { still: false } => "Overlay video",
            Self::Overlay { still: true } => "Overlay photo",
            Self::Audio => "Audio",
            Self::Text => "Text",
            Self::Caption => "Caption",
            Self::Effect => "Effect",
            Self::Filter => "Filter",
            Self::Adjust => "Adjustment",
            Self::Sticker => "Sticker",
            Self::Other => "Clip",
        }
    }
}

/// How a title's outline, shadow, box, and glow look. A part with an empty
/// color is off; its numbers are what it returns with.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TextLook {
    pub outline_color: String,
    /// Ems.
    pub outline_width: f64,
    pub shadow_color: String,
    /// Hundredths of an em.
    pub shadow_distance: f64,
    /// 0 to 1.
    pub shadow_blur: f64,
    /// Degrees. A title with no shadow reads the angle a first one takes.
    pub shadow_angle: f64,
    pub background_color: String,
    /// 0 to 1.
    pub background_alpha: f64,
    /// 0 square to 1 round.
    pub background_round: f64,
    pub glow_color: String,
    /// 0 to 1.
    pub glow_strength: f64,
    /// 0 to 1.
    pub glow_range: f64,
    /// The bubble's key; empty for a plain box or none.
    pub bubble: String,
}

/// What a part takes when its color is first picked and its numbers are
/// still zero. These are the phone's Styles starting values.
pub const OUTLINE_START: f64 = 0.06;
pub const SHADOW_DISTANCE_START: f64 = 8.0;
pub const SHADOW_BLUR_START: f64 = 0.3;
pub const BACKGROUND_ALPHA_START: f64 = 1.0;
pub const BACKGROUND_ROUND_START: f64 = 0.3;
pub const GLOW_STRENGTH_START: f64 = 0.6;
pub const GLOW_RANGE_START: f64 = 0.5;
/// Where a new shadow falls: down and to the right.
pub const SHADOW_ANGLE: f64 = -45.0;

/// The panel for `segment` on `track`. The material decides first; the row
/// decides for materials that only say what they are through their row.
#[must_use]
pub fn panel_of(draft: &Draft, track: &Track, segment: &Segment) -> Panel {
    let picture = |still: bool| {
        if track.kind == TrackType::Sticker {
            Panel::Sticker
        } else if track.is_main() {
            Panel::Main { still }
        } else {
            Panel::Overlay { still }
        }
    };
    match draft.materials.get(segment.material_id.as_str()) {
        Some(MaterialRef::Video(video)) => picture(video.kind == "photo"),
        Some(MaterialRef::Draft(_)) => picture(false),
        Some(MaterialRef::Color(_)) => picture(true),
        Some(MaterialRef::Audio(_)) => Panel::Audio,
        Some(MaterialRef::Text(text)) if is_caption(track, text) => Panel::Caption,
        Some(MaterialRef::Text(_)) => Panel::Text,
        Some(MaterialRef::VideoEffect(_) | MaterialRef::Plugin(_)) => Panel::Effect,
        _ => match track.kind {
            TrackType::Audio | TrackType::TextToVideoTextAudio => Panel::Audio,
            TrackType::Text | TrackType::TextToVideoText => Panel::Text,
            TrackType::Sticker => Panel::Sticker,
            TrackType::Effect | TrackType::PluginEffect => Panel::Effect,
            TrackType::Filter => Panel::Filter,
            TrackType::Adjust => Panel::Adjust,
            _ if draft
                .materials
                .stickers
                .iter()
                .any(|item| item.id == segment.material_id) =>
            {
                Panel::Sticker
            }
            _ => Panel::Other,
        },
    }
}

/// True when `text` on `track` is a caption: its row is a caption or
/// lyric row, its kind says so, or it has recognized words.
#[must_use]
pub fn is_caption(track: &Track, text: &MaterialText) -> bool {
    matches!(
        track.flag,
        TrackFlag::TextSubtitle
            | TrackFlag::TextLyric
            | TrackFlag::VideoSubtitle
            | TrackFlag::AudioSubtitle
            | TrackFlag::AiScriptSubtitle
    ) || matches!(text.kind.as_str(), "subtitle" | "lyrics")
        || !text.words.text.is_empty()
}

/// The title's outline, shadow, box and glow; `None` when it is not text.
#[must_use]
pub fn text_look(draft: &Draft, segment: &Segment) -> Option<TextLook> {
    let Some(MaterialRef::Text(text)) = draft.materials.get(segment.material_id.as_str()) else {
        return None;
    };
    let on = |on: bool, color: &str| if on { color.to_owned() } else { String::new() };
    // No glow reads as no amounts, so a first glow color takes the starts.
    let (glow_color, glow_strength, glow_range) = segment_bloom(&draft.materials, segment)
        .and_then(MaterialEffect::bloom)
        .map_or((String::new(), 0.0, 0.0), |glow| {
            (glow.color, glow.strength, glow.range)
        });
    Some(TextLook {
        outline_color: text.border_color.clone(),
        outline_width: text.border_width,
        shadow_color: on(text.has_shadow, &text.shadow_color),
        shadow_distance: text.shadow_distance,
        shadow_blur: text.shadow_smoothing,
        shadow_angle: if text.has_shadow && text.shadow_angle.is_finite() {
            text.shadow_angle
        } else {
            SHADOW_ANGLE
        },
        background_color: on(text.background_style != 0, &text.background_color),
        background_alpha: text.background_alpha,
        background_round: text.background_round_radius,
        glow_color,
        glow_strength,
        glow_range,
        bubble: text.bubble.clone(),
    })
}

/// The quietest level the volume slider shows; it and below are silence.
pub const SILENT_DB: f64 = -60.0;

/// A gain in decibels, [`SILENT_DB`] for silence.
#[must_use]
pub fn volume_db(gain: f64) -> f64 {
    if gain <= 0.0 || !gain.is_finite() {
        return SILENT_DB;
    }
    (20.0 * gain.log10()).max(SILENT_DB)
}

/// The gain for `db` decibels; [`SILENT_DB`] and below is zero.
#[must_use]
pub fn gain_of_db(db: f64) -> f64 {
    if db <= SILENT_DB {
        0.0
    } else {
        10f64.powf(db / 20.0)
    }
}

/// How the Case buttons rewrite a title's words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LetterCase {
    Upper,
    Lower,
    /// The first letter of each word up, the rest down.
    Title,
}

impl LetterCase {
    #[must_use]
    pub fn apply(self, words: &str) -> String {
        match self {
            Self::Upper => words.to_uppercase(),
            Self::Lower => words.to_lowercase(),
            Self::Title => {
                let mut out = String::with_capacity(words.len());
                let mut start = true;
                for letter in words.chars() {
                    if letter.is_whitespace() {
                        start = true;
                        out.push(letter);
                    } else if start {
                        start = false;
                        out.extend(letter.to_uppercase());
                    } else {
                        out.extend(letter.to_lowercase());
                    }
                }
                out
            }
        }
    }
}

/// `value`, or `start` when a part is being turned on with nothing set.
#[must_use]
pub fn starting(color: &str, value: f64, start: f64) -> f64 {
    if !color.is_empty() && value == 0.0 {
        start
    } else {
        value
    }
}

/// Title style presets: name, fill color, box color. "None" clears both.
pub const TEXT_PRESETS: [(&str, &str, &str); 16] = [
    ("None", "", ""),
    ("Plain", "#FFFFFF", "#3A3A3A"),
    ("Yellow", "#FFE14A", "#3A3A3A"),
    ("Red", "#FF4D4D", "#3A3A3A"),
    ("Orange", "#FF9A3C", "#3A3A3A"),
    ("Blue", "#5AA7FF", "#3A3A3A"),
    ("Green", "#3DDC6A", "#1C1C1C"),
    ("Gray", "#D0D0D0", "#2A2A2A"),
    ("Ink", "#1A1A1A", "#F3F3F3"),
    ("Gold", "#F2C14E", "#2A2418"),
    ("Violet", "#FFFFFF", "#7A3CFF"),
    ("Paper", "#7A3CFF", "#FFFFFF"),
    ("Night", "#FFFFFF", "#111111"),
    ("Lime", "#D6FF4A", "#111111"),
    ("Rose", "#FF6A9A", "#2A1218"),
    ("Cyan", "#3DE0FF", "#102028"),
];

/// A clip's stored position (-1 to 1 each way, Y up) as canvas pixels from
/// the middle, Y down, rounded, as the position fields show it.
#[must_use]
pub fn canvas_pixels(transform: (f64, f64), canvas: (i32, i32)) -> (f64, f64) {
    let (half_width, half_height) = halves(canvas);
    (
        (transform.0 * half_width).round(),
        (-transform.1 * half_height).round(),
    )
}

/// Canvas pixels from the middle, Y down, as a stored position.
#[must_use]
pub fn stored_position(pixels: (f64, f64), canvas: (i32, i32)) -> (f64, f64) {
    let (half_width, half_height) = halves(canvas);
    (pixels.0 / half_width, -pixels.1 / half_height)
}

/// Half the canvas, falling back to 1920×1080 for an empty one.
fn halves(canvas: (i32, i32)) -> (f64, f64) {
    if canvas.0 > 0 && canvas.1 > 0 {
        (f64::from(canvas.0) / 2.0, f64::from(canvas.1) / 2.0)
    } else {
        (960.0, 540.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decibels_round_trip_and_floor_at_silence() {
        assert_eq!(volume_db(1.0), 0.0);
        assert_eq!(volume_db(0.0), SILENT_DB);
        assert!((gain_of_db(volume_db(0.5)) - 0.5).abs() < 1e-9);
        assert_eq!(gain_of_db(SILENT_DB), 0.0);
    }

    #[test]
    fn letter_case_rewrites_words() {
        assert_eq!(
            LetterCase::Title.apply("hELLO big world"),
            "Hello Big World"
        );
        assert_eq!(LetterCase::Upper.apply("ok"), "OK");
    }

    #[test]
    fn canvas_pixels_count_from_the_middle_with_y_down() {
        assert_eq!(canvas_pixels((0.5, 0.5), (1920, 1080)), (480.0, -270.0));
        assert_eq!(stored_position((480.0, -270.0), (1920, 1080)), (0.5, 0.5));
        assert_eq!(canvas_pixels((1.0, 0.0), (0, 0)), (960.0, 0.0));
    }

    #[test]
    fn a_picture_on_a_sticker_row_opens_the_sticker_panel() {
        let mut draft = Draft::empty("D");
        let mut heart = cutlass_model::MaterialVideo::new("heart");
        heart.kind = "sticker".to_owned();
        draft.materials.videos.push(heart);
        let segment = Segment::new("a", "heart", cutlass_model::TimeRange::new(0, 1));
        let mut row = Track::main("stickers");
        row.kind = TrackType::Sticker;
        assert_eq!(panel_of(&draft, &row, &segment), Panel::Sticker);
        row.kind = TrackType::Video;
        assert_eq!(
            panel_of(&draft, &row, &segment),
            Panel::Main { still: false }
        );
    }

    #[test]
    fn a_caption_without_words_still_opens_the_caption_panel() {
        let mut draft = Draft::empty("D");
        let mut line = MaterialText::new("line");
        line.kind = "subtitle".to_owned();
        draft.materials.texts.push(line);
        let mut title = MaterialText::new("title");
        title.kind = "text".to_owned();
        draft.materials.texts.push(title);
        let mut row = Track::main("captions");
        row.kind = TrackType::Text;
        let caption = Segment::new("a", "line", cutlass_model::TimeRange::new(0, 1));
        let plain = Segment::new("b", "title", cutlass_model::TimeRange::new(0, 1));
        assert_eq!(panel_of(&draft, &row, &caption), Panel::Caption);
        assert_eq!(panel_of(&draft, &row, &plain), Panel::Text);
        row.flag = TrackFlag::TextSubtitle;
        assert_eq!(panel_of(&draft, &row, &plain), Panel::Caption);
    }

    #[test]
    fn a_starting_value_fills_in_only_when_turning_on() {
        assert_eq!(starting("#fff", 0.0, 0.3), 0.3);
        assert_eq!(starting("", 0.0, 0.3), 0.0);
        assert_eq!(starting("#fff", 0.5, 0.3), 0.5);
    }
}
