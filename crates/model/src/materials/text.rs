use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::words::WordColumns;
use crate::id::MaterialId;

fn one() -> f64 {
    1.0
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterialText {
    #[serde(default)]
    pub id: MaterialId,
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub base_content: String,
    #[serde(default)]
    pub text_color: String,
    #[serde(default = "one")]
    pub text_alpha: f64,
    #[serde(default)]
    pub font_size: f64,
    #[serde(default)]
    pub font_name: String,
    #[serde(default)]
    pub font_path: String,
    #[serde(default)]
    pub font_title: String,
    #[serde(default)]
    pub alignment: i32,
    #[serde(default)]
    pub typesetting: i32,
    #[serde(default)]
    pub line_spacing: f64,
    #[serde(default)]
    pub letter_spacing: f64,
    #[serde(default)]
    pub line_max_width: f64,
    #[serde(default)]
    pub background_color: String,
    #[serde(default = "one")]
    pub background_alpha: f64,
    #[serde(default)]
    pub background_style: i32,
    #[serde(default)]
    pub background_round_radius: f64,
    #[serde(default)]
    pub background_width: f64,
    #[serde(default)]
    pub background_height: f64,
    #[serde(default)]
    pub border_color: String,
    #[serde(default = "one")]
    pub border_alpha: f64,
    #[serde(default)]
    pub border_width: f64,
    #[serde(default)]
    pub has_shadow: bool,
    #[serde(default)]
    pub shadow_color: String,
    #[serde(default)]
    pub shadow_alpha: f64,
    #[serde(default)]
    pub shadow_angle: f64,
    #[serde(default)]
    pub shadow_distance: f64,
    #[serde(default)]
    pub shadow_smoothing: f64,
    #[serde(default = "one")]
    pub global_alpha: f64,
    #[serde(default)]
    pub bold_width: f64,
    #[serde(default)]
    pub italic_degree: i32,
    #[serde(default)]
    pub underline: bool,
    #[serde(default)]
    pub is_rich_text: bool,
    #[serde(default)]
    pub text_size: i32,
    #[serde(default)]
    pub check_flag: i32,
    #[serde(default)]
    pub words: WordColumns,
    #[serde(default)]
    pub current_words: WordColumns,
    #[serde(default)]
    pub text_to_audio_ids: Vec<String>,
    #[serde(default)]
    pub fixed_width: f64,
    #[serde(default)]
    pub fixed_height: f64,
    #[serde(default)]
    pub group_id: String,
    #[serde(default)]
    pub sub_type: i32,
    #[serde(default)]
    pub recognize_type: i32,
    /// The [`Bubble`] the background is shaped as, by its key; empty is a
    /// plain box. The app's own: the reference editor does not read it.
    #[serde(
        default,
        rename = "cutlass_bubble",
        skip_serializing_if = "String::is_empty"
    )]
    pub bubble: String,
    /// The caption style its words are drawn in, by key; empty is plain.
    /// The app's own, like `bubble`.
    #[serde(
        default,
        rename = "cutlass_caption_style",
        skip_serializing_if = "String::is_empty"
    )]
    pub caption_style: String,
    /// Words the caption marks, by index into its word columns: drawn in
    /// the style's spoken look while they show. The app's own.
    #[serde(
        default,
        rename = "cutlass_caption_keywords",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub caption_keywords: Vec<usize>,
    /// Milliseconds each letter of a word starts after the one before it,
    /// so a word's look runs across its letters; 0 changes them together.
    /// The app's own.
    #[serde(
        default,
        rename = "cutlass_caption_stagger_ms",
        skip_serializing_if = "is_zero"
    )]
    pub caption_stagger_ms: i64,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

impl MaterialText {
    /// Resource-mapping fallbacks. An absent key stays at 1, and so does [`Default`].
    pub const MAPPING_TEXT_ALPHA: f64 = 0.0;
    pub const MAPPING_BACKGROUND_ALPHA: f64 = 0.0;
    pub const MAPPING_BORDER_ALPHA: f64 = 0.0;
    pub const MAPPING_GLOBAL_ALPHA: f64 = 0.0;

    pub fn new(id: impl Into<MaterialId>) -> Self {
        Self {
            id: id.into(),
            ..Self::default()
        }
    }
}

/// Serde's skip test for a zero stagger.
#[allow(clippy::trivially_copy_pass_by_ref)]
const fn is_zero(value: &i64) -> bool {
    *value == 0
}

impl Default for MaterialText {
    fn default() -> Self {
        Self {
            id: MaterialId::default(),
            kind: String::new(),
            name: String::new(),
            content: String::new(),
            base_content: String::new(),
            text_color: String::new(),
            text_alpha: 1.0,
            font_size: 0.0,
            font_name: String::new(),
            font_path: String::new(),
            font_title: String::new(),
            alignment: 0,
            typesetting: 0,
            line_spacing: 0.0,
            letter_spacing: 0.0,
            line_max_width: 0.0,
            background_color: String::new(),
            background_alpha: 1.0,
            background_style: 0,
            background_round_radius: 0.0,
            background_width: 0.0,
            background_height: 0.0,
            border_color: String::new(),
            border_alpha: 1.0,
            border_width: 0.0,
            has_shadow: false,
            shadow_color: String::new(),
            shadow_alpha: 0.0,
            shadow_angle: 0.0,
            shadow_distance: 0.0,
            shadow_smoothing: 0.0,
            global_alpha: 1.0,
            bold_width: 0.0,
            italic_degree: 0,
            underline: false,
            is_rich_text: false,
            text_size: 0,
            check_flag: 0,
            words: WordColumns::default(),
            current_words: WordColumns::default(),
            text_to_audio_ids: Vec::new(),
            fixed_width: 0.0,
            fixed_height: 0.0,
            group_id: String::new(),
            sub_type: 0,
            recognize_type: 0,
            bubble: String::new(),
            caption_style: String::new(),
            caption_keywords: Vec::new(),
            caption_stagger_ms: 0,
            rest: Map::new(),
        }
    }
}

/// The keys a caption's style may have, as `cutlass_captions` draws them.
pub const CAPTION_STYLE_KEYS: [&str; 10] = [
    "classic",
    "karaoke",
    "highlight",
    "pop",
    "word-by-word",
    "one-word",
    "neon",
    "bold-box",
    "fade",
    "rise",
];

/// A shape a title's background takes instead of a plain box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bubble {
    /// A box with fully round ends.
    Pill,
    /// A round box with a tail down to the left, as someone speaks.
    Speech,
    /// A cloud with two small puffs trailing down to the left.
    Thought,
    /// A spiky star around the letters.
    Burst,
    /// A strip with notched ends.
    Banner,
}

impl Bubble {
    /// Every bubble, in the order the panel shows them.
    pub const ALL: [Self; 5] = [
        Self::Pill,
        Self::Speech,
        Self::Thought,
        Self::Burst,
        Self::Banner,
    ];

    /// The key a title stores it by.
    #[must_use]
    pub const fn key(self) -> &'static str {
        match self {
            Self::Pill => "pill",
            Self::Speech => "speech",
            Self::Thought => "thought",
            Self::Burst => "burst",
            Self::Banner => "banner",
        }
    }

    /// The name the panel shows.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Pill => "Pill",
            Self::Speech => "Speech",
            Self::Thought => "Thought",
            Self::Burst => "Burst",
            Self::Banner => "Banner",
        }
    }

    /// The bubble stored as `key`, if it is one.
    #[must_use]
    pub fn of(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|bubble| bubble.key() == key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_alphas_stay_opaque() {
        let text = MaterialText::default();
        let parsed: MaterialText = serde_json::from_str("{}").unwrap();
        assert_eq!(text, parsed);
        assert_eq!(text.text_alpha, 1.0);
        assert_eq!(text.background_alpha, 1.0);
        assert_eq!(text.border_alpha, 1.0);
        assert_eq!(text.global_alpha, 1.0);
        assert_ne!(text.text_alpha, MaterialText::MAPPING_TEXT_ALPHA);
        assert_ne!(
            text.background_alpha,
            MaterialText::MAPPING_BACKGROUND_ALPHA
        );
        assert_ne!(text.border_alpha, MaterialText::MAPPING_BORDER_ALPHA);
        assert_ne!(text.global_alpha, MaterialText::MAPPING_GLOBAL_ALPHA);
    }

    #[test]
    fn a_bubble_is_kept_by_its_key_and_left_out_when_there_is_none() {
        let mut text = MaterialText::new("t");
        assert!(!serde_json::to_string(&text).unwrap().contains("bubble"));
        text.bubble = Bubble::Speech.key().to_owned();
        let json = serde_json::to_string(&text).unwrap();
        assert!(json.contains(r#""cutlass_bubble":"speech""#));
        let back: MaterialText = serde_json::from_str(&json).unwrap();
        assert_eq!(Bubble::of(&back.bubble), Some(Bubble::Speech));
        assert!(back.rest.is_empty());
        assert_eq!(Bubble::of("cloud"), None);
    }
}
