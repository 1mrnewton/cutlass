use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::effect::MaterialEffect;
use crate::id::MaterialId;

/// One HSL record. The mapping source platform of 1 is not applied to an absent key.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct MaterialHsl {
    #[serde(default)]
    pub id: MaterialId,
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub constant_material_id: String,
    #[serde(default)]
    pub hsl_color_type: i32,
    #[serde(default)]
    pub hue: i32,
    #[serde(default)]
    pub saturation: i32,
    #[serde(default)]
    pub lightness: i32,
    #[serde(default)]
    pub interacting: bool,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub visible: bool,
    #[serde(default)]
    pub lumi_hub_path: String,
    #[serde(default)]
    pub custom_color: String,
    #[serde(default)]
    pub resource_id: String,
    #[serde(default)]
    pub source_platform: i32,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

impl MaterialHsl {
    pub const MAPPING_SOURCE_PLATFORM: i32 = 1;
}

/// A picture-adjust material. The 56 material pools do not include a pool for
/// this record, so it is not stored as one of those arrays.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct MaterialPictureAdjust {
    #[serde(default)]
    pub id: MaterialId,
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub brightness: Option<MaterialEffect>,
    #[serde(default)]
    pub contrast: Option<MaterialEffect>,
    #[serde(default)]
    pub saturation: Option<MaterialEffect>,
    #[serde(default)]
    pub sharpening: Option<MaterialEffect>,
    #[serde(default)]
    pub highlight: Option<MaterialEffect>,
    #[serde(default)]
    pub shadow: Option<MaterialEffect>,
    #[serde(default)]
    pub white: Option<MaterialEffect>,
    #[serde(default)]
    pub black: Option<MaterialEffect>,
    #[serde(default)]
    pub clear: Option<MaterialEffect>,
    #[serde(default)]
    pub color_temperature: Option<MaterialEffect>,
    #[serde(default)]
    pub hue: Option<MaterialEffect>,
    #[serde(default)]
    pub fade: Option<MaterialEffect>,
    #[serde(default)]
    pub light_sensation: Option<MaterialEffect>,
    #[serde(default)]
    pub vignetting: Option<MaterialEffect>,
    #[serde(default)]
    pub particle: Option<MaterialEffect>,
    #[serde(default)]
    pub lut: Option<MaterialEffect>,
    #[serde(default)]
    pub hsl: Option<MaterialHsl>,
    #[serde(default)]
    pub enable_lut: bool,
    #[serde(default)]
    pub enable_adjust: bool,
    #[serde(default)]
    pub enable_hsl: bool,
    #[serde(default)]
    pub enable_color_curves: bool,
    #[serde(default)]
    pub color_curves: Option<Value>,
    #[serde(default)]
    pub enable_hsl_curves: bool,
    #[serde(default)]
    pub hsl_curves: Option<Value>,
    #[serde(default)]
    pub enable_color_wheels: bool,
    #[serde(default)]
    pub primary_color_wheels: Option<Value>,
    #[serde(default)]
    pub log_color_wheels: Option<Value>,
    #[serde(default)]
    pub filter: Option<MaterialEffect>,
    #[serde(default)]
    pub adjust_mask: Option<Value>,
    #[serde(default)]
    pub adjust_preset_resource_id: String,
    #[serde(default)]
    pub smart_adjust_color: Option<MaterialEffect>,
    #[serde(default)]
    pub color_match: Option<MaterialEffect>,
    #[serde(default)]
    pub enable_color_match_adjust: bool,
    #[serde(default)]
    pub color_correct: Option<MaterialEffect>,
    #[serde(default)]
    pub enable_adjust_mask: bool,
    #[serde(default)]
    pub color_correct_alg_result: String,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_absent_hsl_platform_is_not_the_mapping_value() {
        let hsl: MaterialHsl = serde_json::from_str("{}").unwrap();
        assert_eq!(hsl.source_platform, 0);
        assert_ne!(hsl.source_platform, MaterialHsl::MAPPING_SOURCE_PLATFORM);
    }

    #[test]
    fn a_picture_adjust_keeps_its_slots_and_unnamed_keys() {
        let adjust: MaterialPictureAdjust = serde_json::from_str(
            r#"{"id":"a","brightness":{"value":0.2},"enable_adjust":true,"future_adjust":1}"#,
        )
        .unwrap();
        assert!(adjust.enable_adjust);
        assert_eq!(adjust.brightness.as_ref().unwrap().value, Some(0.2));
        assert_eq!(serde_json::to_value(&adjust).unwrap()["future_adjust"], 1);
    }
}
