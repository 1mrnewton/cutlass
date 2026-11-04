use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::enums::AnimType;
use crate::id::MaterialId;
use crate::materials::Materials;
use crate::segment::Segment;
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
    /// The `type` of a title's glow.
    pub const BLOOM: &'static str = "bloom";

    pub fn new(id: impl Into<MaterialId>) -> Self {
        Self {
            id: id.into(),
            ..Self::default()
        }
    }

    /// A title's glow effect storing `bloom`.
    pub fn glow(id: impl Into<MaterialId>, bloom: &BloomParams) -> Self {
        let mut effect = Self::new(id);
        effect.kind = Self::BLOOM.to_owned();
        effect.set_bloom(bloom);
        effect
    }

    /// Store `bloom` as this effect's `bloom_params`.
    pub fn set_bloom(&mut self, bloom: &BloomParams) {
        self.bloom_params = serde_json::to_value(bloom).ok();
    }

    /// The glow this effect stores, when its `bloom_params` read as one.
    #[must_use]
    pub fn bloom(&self) -> Option<BloomParams> {
        serde_json::from_value(self.bloom_params.clone()?).ok()
    }
}

/// A title's glow, as `bloom_params` stores it. `strength` and `range` are
/// read from 0 to 1; the direction is kept but not drawn.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BloomParams {
    #[serde(default)]
    pub color: String,
    #[serde(default)]
    pub strength: f64,
    #[serde(default = "full_range")]
    pub range: f64,
    #[serde(default)]
    pub dir_x: f64,
    #[serde(default)]
    pub dir_y: f64,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

impl Default for BloomParams {
    fn default() -> Self {
        Self {
            color: String::new(),
            strength: 0.0,
            range: full_range(),
            dir_x: 0.0,
            dir_y: 0.0,
            rest: Map::new(),
        }
    }
}

const fn full_range() -> f64 {
    1.0
}

/// The glow effect among `segment`'s sidecars.
#[must_use]
pub fn segment_bloom<'a>(
    materials: &'a Materials,
    segment: &Segment,
) -> Option<&'a MaterialEffect> {
    segment.extra_material_refs.iter().find_map(|id| {
        materials
            .effects
            .iter()
            .find(|effect| &effect.id == id && effect.kind == MaterialEffect::BLOOM)
    })
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
    fn a_glow_reads_its_params_and_is_found_on_its_clip() {
        let mut effect = MaterialEffect::new("glow");
        effect.kind = MaterialEffect::BLOOM.to_owned();
        effect.bloom_params = Some(serde_json::json!({"color":"#FFCC00","strength":0.6,"extra":1}));
        let bloom = effect.bloom().unwrap();
        assert_eq!(bloom.color, "#FFCC00");
        assert_eq!(bloom.strength, 0.6);
        assert_eq!(bloom.range, 1.0);
        assert_eq!(bloom.rest.get("extra"), Some(&serde_json::json!(1)));

        let mut other = MaterialEffect::new("other");
        other.kind = "video_effect".to_owned();
        let materials = Materials {
            effects: vec![other, effect],
            ..Materials::default()
        };
        let mut segment = Segment::new("s", "t", TimeRange::new(0, 1));
        segment.extra_material_refs = vec![MaterialId::new("other")];
        assert!(segment_bloom(&materials, &segment).is_none());
        segment.extra_material_refs.push(MaterialId::new("glow"));
        assert_eq!(
            segment_bloom(&materials, &segment).unwrap().id.as_str(),
            "glow"
        );

        let made = MaterialEffect::glow("g", &bloom);
        assert_eq!(made.kind, MaterialEffect::BLOOM);
        assert_eq!(made.bloom(), Some(bloom));

        let mut broken = MaterialEffect::new("b");
        broken.bloom_params = Some(serde_json::json!({"strength":"high"}));
        assert!(broken.bloom().is_none());
    }

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
