use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::enums::AnimType;
use crate::id::MaterialId;
use crate::time::{Micros, TimeRange};

/// An effect package. The mapping fallback for `value` is 0.5 and is not applied
/// when the key is absent.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct MaterialEffect {
    #[serde(default)]
    pub id: MaterialId,
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub intensity_key: String,
    #[serde(default)]
    pub effect_id: String,
    #[serde(default)]
    pub resource_id: String,
    #[serde(default)]
    pub third_resource_id: String,
    #[serde(default)]
    pub item_effect_type: i64,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub sub_type: Option<Value>,
    #[serde(default)]
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    #[serde(default)]
    pub category_id: String,
    #[serde(default)]
    pub category_name: String,
    #[serde(default)]
    pub sub_category_id: String,
    #[serde(default)]
    pub sub_category_name: String,
    #[serde(default)]
    pub platform: String,
    #[serde(default)]
    pub source_platform: i32,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub adjust_params: Option<Value>,
    #[serde(default)]
    pub formula_id: String,
    #[serde(default)]
    pub enable_skin_tone_correction: bool,
    #[serde(default)]
    pub algorithm_artifact_path: String,
    #[serde(default)]
    pub face_adjust_params: Option<Value>,
    #[serde(default)]
    pub exclusion_group: String,
    #[serde(default)]
    pub panel_id: String,
    #[serde(default)]
    pub bloom_params: Option<Value>,
    #[serde(default)]
    pub request_id: String,
    #[serde(default)]
    pub color_match_info: Option<Value>,
    #[serde(default)]
    pub lumi_hub_path: String,
    #[serde(default)]
    pub covering_relation_change: Option<Value>,
    #[serde(default)]
    pub beauty_face_auto_preset_id: String,
    #[serde(default)]
    pub beauty_body_auto_preset_id: String,
    #[serde(default)]
    pub smart_color_mode: i32,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

impl MaterialEffect {
    pub const MAPPING_VALUE: f64 = 0.5;
    pub const MAPPING_SOURCE_PLATFORM: i32 = 1;

    pub fn new(id: impl Into<MaterialId>) -> Self {
        Self {
            id: id.into(),
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct MaterialVideoEffect {
    #[serde(default)]
    pub id: MaterialId,
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub effect_id: String,
    #[serde(default)]
    pub resource_id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub value: f64,
    #[serde(default)]
    pub category_id: String,
    #[serde(default)]
    pub category_name: String,
    #[serde(default)]
    pub item_effect_type: i64,
    #[serde(default)]
    pub platform: String,
    #[serde(default)]
    pub source_platform: i32,
    #[serde(default)]
    pub sub_type: Option<Value>,
    #[serde(default)]
    pub bind_segment_id: String,
    #[serde(default)]
    pub transparent_params: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub adjust_params: Option<Value>,
    #[serde(default)]
    pub formula_id: String,
    #[serde(default)]
    pub render_index: i32,
    #[serde(default)]
    pub track_render_index: Option<Value>,
    #[serde(default)]
    pub common_keyframes: Option<Value>,
    #[serde(default)]
    pub request_id: String,
    #[serde(default)]
    pub algorithm_artifact_path: String,
    #[serde(default)]
    pub disable_effect_faces: String,
    #[serde(default)]
    pub enable_mask: bool,
    #[serde(default)]
    pub effect_mask: Option<Value>,
    #[serde(default)]
    pub covering_relation_change: Option<Value>,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

impl MaterialVideoEffect {
    pub const MAPPING_SOURCE_PLATFORM: i32 = 1;
}

/// One in, out, loop, caption, or group animation.
///
/// A missing `duration` stays missing. It is not zero, and it is not the
/// mapping fallback of three seconds. A missing `platform` stays empty.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct StickerAnimation {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub effect_id: String,
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub start: Micros,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<Micros>,
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub platform: String,
    #[serde(default)]
    pub resource_id: String,
    #[serde(default)]
    pub third_resource_id: String,
    #[serde(default)]
    pub source_platform: i32,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub category_id: String,
    #[serde(default)]
    pub category_name: String,
    #[serde(default)]
    pub panel: String,
    #[serde(default)]
    pub material_type: String,
    #[serde(default)]
    pub anim_adjust_params: Option<Value>,
    #[serde(default)]
    pub request_id: String,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

impl StickerAnimation {
    pub const MAPPING_DURATION: Micros = Micros(3_000_000);
    pub const MAPPING_PLATFORM: &'static str = "all";
    pub const MAPPING_SOURCE_PLATFORM: i32 = 1;

    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ..Self::default()
        }
    }

    #[must_use]
    pub fn anim_type(&self) -> Option<AnimType> {
        AnimType::parse(&self.kind)
    }
}

#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct MaterialAnimations {
    #[serde(default)]
    pub id: MaterialId,
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub animations: Vec<StickerAnimation>,
    #[serde(default)]
    pub multi_language_current: String,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct MaterialPluginEffect {
    #[serde(default)]
    pub id: MaterialId,
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub seg_id: String,
    #[serde(default)]
    pub plugin_id: String,
    #[serde(default)]
    pub plugin_version: String,
    #[serde(default)]
    pub effect_name: String,
    #[serde(default)]
    pub dir_name: String,
    #[serde(default)]
    pub effect_abs_path: String,
    #[serde(default)]
    pub time_range: Option<TimeRange>,
    #[serde(default)]
    pub apply_time_range: Option<TimeRange>,
    #[serde(default)]
    pub effect_params: String,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_effect_value_is_not_the_mapping_fallback() {
        let effect = MaterialEffect::new("e");
        let parsed: MaterialEffect = serde_json::from_str("{}").unwrap();
        assert_eq!(effect.value, None);
        assert_eq!(parsed.value, None);
        assert_eq!(parsed.source_platform, 0);
        assert_ne!(parsed.value, Some(MaterialEffect::MAPPING_VALUE));
        assert_ne!(
            parsed.source_platform,
            MaterialEffect::MAPPING_SOURCE_PLATFORM
        );
        let stored: MaterialEffect = serde_json::from_str(r#"{"value":0.0}"#).unwrap();
        assert_eq!(stored.value, Some(0.0));
        assert!(serde_json::to_value(parsed).unwrap().get("value").is_none());
    }

    #[test]
    fn a_missing_animation_duration_is_neither_zero_nor_three_seconds() {
        let anim = StickerAnimation::new("a");
        let parsed: StickerAnimation = serde_json::from_str(
            r#"{"effect_id":"e","start":0,"path":"p","name":"n","panel":"","material_type":"in"}"#,
        )
        .unwrap();
        assert!(anim.duration.is_none());
        assert!(parsed.duration.is_none());
        assert_eq!(parsed.platform, "");
        assert_ne!(parsed.duration, Some(Micros(0)));
        assert_ne!(parsed.duration, Some(StickerAnimation::MAPPING_DURATION));
        assert_ne!(parsed.platform, StickerAnimation::MAPPING_PLATFORM);
        assert_eq!(parsed.source_platform, 0);
        assert_ne!(
            parsed.source_platform,
            StickerAnimation::MAPPING_SOURCE_PLATFORM
        );
        assert!(parsed.anim_type().is_none());
        let json = serde_json::to_value(&parsed).unwrap();
        assert!(json.get("duration").is_none());
        assert_eq!(json["platform"], "");
        assert_eq!(json["effect_id"], "e");

        let zero: StickerAnimation = serde_json::from_str(r#"{"duration":0}"#).unwrap();
        assert_eq!(zero.duration, Some(Micros(0)));
        let typed: StickerAnimation = serde_json::from_str(r#"{"type":"in"}"#).unwrap();
        assert_eq!(typed.anim_type(), Some(AnimType::In));
    }
}
