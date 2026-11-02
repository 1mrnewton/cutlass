use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::Crop;
use crate::draft::Draft;
use crate::id::MaterialId;

/// A compound clip. `draft` is the nested document, when the file inlines it.
/// `combination_type` names AI-music roles. It does not define nesting.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct MaterialDraft {
    #[serde(default)]
    pub id: MaterialId,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub formula_id: String,
    #[serde(default)]
    pub combination_id: String,
    #[serde(default)]
    pub combination_type: String,
    #[serde(default)]
    pub precompile_combination: bool,
    #[serde(default)]
    pub draft: Option<Box<Draft>>,
    #[serde(default)]
    pub draft_file_path: String,
    #[serde(default)]
    pub draft_cover_path: String,
    #[serde(default)]
    pub draft_config_path: String,
    /// The quad sampled from the child picture. Zero keeps the whole picture.
    #[serde(default, skip_serializing_if = "crop_is_clear")]
    pub crop: Crop,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

fn crop_is_clear(crop: &Crop) -> bool {
    *crop == Crop::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_uncropped_compound_omits_the_quad() {
        let material = MaterialDraft::default();
        let json = serde_json::to_value(&material).unwrap();
        assert!(json.get("crop").is_none());
        let parsed: MaterialDraft = serde_json::from_str("{}").unwrap();
        assert_eq!(parsed.crop, Crop::default());

        let mut cropped = MaterialDraft::default();
        cropped.crop.upper_left_x = 0.25;
        let json = serde_json::to_value(&cropped).unwrap();
        assert_eq!(json["crop"]["upper_left_x"], 0.25);
    }
}
