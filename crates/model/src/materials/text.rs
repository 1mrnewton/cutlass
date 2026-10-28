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
            rest: Map::new(),
        }
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
}
