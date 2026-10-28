use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::clip::{Clip, UniformScale};
use crate::enums::{SegmentSource, TrackAttribute};
use crate::id::{MaterialId, SegmentId};
use crate::keyframe::CommonKeyframeGroup;
use crate::time::TimeRange;

fn one() -> f64 {
    1.0
}

fn yes() -> bool {
    true
}

fn default_scene() -> String {
    "default".to_owned()
}

/// How a text or sticker segment follows another layer. Both local drafts stored this object.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct ResponsiveLayout {
    #[serde(default)]
    pub enable: bool,
    #[serde(default)]
    pub target_follow: String,
    #[serde(default)]
    pub size_layout: i32,
    #[serde(default)]
    pub horizontal_pos_layout: i32,
    #[serde(default)]
    pub vertical_pos_layout: i32,
    /// Keys this model does not name.
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// `hdr_settings` on a segment. `mode` is [`crate::enums::HdrDisplayMode`].
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct HdrSettings {
    #[serde(default)]
    pub mode: i32,
    #[serde(default)]
    pub intensity: f64,
    #[serde(default)]
    pub nits: f64,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// One clip on a track, as `draft_info.json` writes it.
///
/// Video, text, audio, and the other kinds share this record. The kind is the track's
/// type plus the material `type`, not a tag on the segment. Sidecars (speed, animation,
/// canvas, beats) are ids in `extra_material_refs`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct Segment {
    pub id: SegmentId,
    pub material_id: MaterialId,
    #[serde(rename = "target_timerange")]
    pub target: TimeRange,
    #[serde(default, rename = "source_timerange")]
    pub source_range: Option<TimeRange>,
    #[serde(default, rename = "render_timerange")]
    pub render: TimeRange,
    #[serde(default)]
    pub desc: String,
    #[serde(default)]
    pub state: i32,
    #[serde(default = "one")]
    pub speed: f64,
    #[serde(default)]
    pub is_loop: bool,
    #[serde(default)]
    pub is_tone_modify: bool,
    #[serde(default)]
    pub reverse: bool,
    #[serde(default)]
    pub intensifies_audio: bool,
    #[serde(default)]
    pub cartoon: bool,
    #[serde(default = "one")]
    pub volume: f64,
    #[serde(default = "one")]
    pub last_nonzero_volume: f64,
    #[serde(default, deserialize_with = "crate::serde_util::null_as_default")]
    pub clip: Clip,
    #[serde(default, deserialize_with = "crate::serde_util::null_as_default")]
    pub uniform_scale: UniformScale,
    #[serde(default)]
    pub extra_material_refs: Vec<MaterialId>,
    #[serde(default)]
    pub render_index: i32,
    #[serde(default)]
    pub keyframe_refs: Vec<String>,
    #[serde(default)]
    pub enable_lut: bool,
    #[serde(default)]
    pub enable_adjust: bool,
    #[serde(default)]
    pub enable_hsl: bool,
    #[serde(default = "yes")]
    pub visible: bool,
    #[serde(default)]
    pub group_id: String,
    #[serde(default)]
    pub enable_color_curves: bool,
    #[serde(default)]
    pub enable_hsl_curves: bool,
    #[serde(default)]
    pub track_render_index: i32,
    #[serde(default)]
    pub hdr_settings: Option<HdrSettings>,
    #[serde(default)]
    pub hdr_vivid_settings: Option<Value>,
    #[serde(default)]
    pub enable_color_wheels: bool,
    #[serde(default)]
    pub track_attribute: TrackAttribute,
    #[serde(default)]
    pub is_placeholder: bool,
    #[serde(default)]
    pub template_id: String,
    #[serde(default)]
    pub enable_smart_color_adjust: bool,
    #[serde(default = "default_scene")]
    pub template_scene: String,
    #[serde(default)]
    pub common_keyframes: Vec<CommonKeyframeGroup>,
    #[serde(default)]
    pub caption_info: Option<Value>,
    #[serde(default, deserialize_with = "crate::serde_util::null_as_default")]
    pub responsive_layout: ResponsiveLayout,
    #[serde(default)]
    pub enable_color_match_adjust: bool,
    #[serde(default)]
    pub enable_color_correct_adjust: bool,
    #[serde(default)]
    pub enable_adjust_mask: bool,
    #[serde(default)]
    pub raw_segment_id: SegmentId,
    #[serde(default)]
    pub lyric_keyframes: Option<Value>,
    #[serde(default)]
    pub enable_video_mask: bool,
    #[serde(default)]
    pub digital_human_template_group_id: String,
    #[serde(default)]
    pub color_correct_alg_result: String,
    #[serde(default)]
    pub source: SegmentSource,
    #[serde(default)]
    pub enable_mask_stroke: bool,
    #[serde(default)]
    pub enable_mask_shadow: bool,
    #[serde(default)]
    pub enable_color_adjust_pro: bool,
    #[serde(default)]
    pub segment_color_tag: String,
    /// Keys this model does not name yet. Video-only fields land here.
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

impl Segment {
    pub fn new(
        id: impl Into<SegmentId>,
        material_id: impl Into<MaterialId>,
        target: TimeRange,
    ) -> Self {
        Self {
            id: id.into(),
            material_id: material_id.into(),
            target,
            source_range: None,
            render: TimeRange::default(),
            desc: String::new(),
            state: 0,
            speed: 1.0,
            is_loop: false,
            is_tone_modify: false,
            reverse: false,
            intensifies_audio: false,
            cartoon: false,
            volume: 1.0,
            last_nonzero_volume: 1.0,
            clip: Clip::default(),
            uniform_scale: UniformScale::default(),
            extra_material_refs: Vec::new(),
            render_index: 0,
            keyframe_refs: Vec::new(),
            enable_lut: false,
            enable_adjust: false,
            enable_hsl: false,
            visible: true,
            group_id: String::new(),
            enable_color_curves: false,
            enable_hsl_curves: false,
            track_render_index: 0,
            hdr_settings: None,
            hdr_vivid_settings: None,
            enable_color_wheels: false,
            track_attribute: TrackAttribute::default(),
            is_placeholder: false,
            template_id: String::new(),
            enable_smart_color_adjust: false,
            template_scene: default_scene(),
            common_keyframes: Vec::new(),
            caption_info: None,
            responsive_layout: ResponsiveLayout::default(),
            enable_color_match_adjust: false,
            enable_color_correct_adjust: false,
            enable_adjust_mask: false,
            raw_segment_id: SegmentId::default(),
            lyric_keyframes: None,
            enable_video_mask: false,
            digital_human_template_group_id: String::new(),
            color_correct_alg_result: String::new(),
            source: SegmentSource::Normal,
            enable_mask_stroke: false,
            enable_mask_shadow: false,
            enable_color_adjust_pro: false,
            segment_color_tag: String::new(),
            rest: Map::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_file_spells_the_range_timerange() {
        let segment: Segment = serde_json::from_str(
            r#"{"id":"s","material_id":"m","target_timerange":{"start":0,"duration":5000000},"source_timerange":null,"clip":null,"render_index":14000,"extra_material_refs":["a"],"template_scene":"default","source":"segmentsourcenormal"}"#,
        )
        .unwrap();
        assert_eq!(segment.target, TimeRange::new(0, 5_000_000));
        assert!(segment.source_range.is_none());
        assert_eq!(segment.clip, Clip::default());
        assert_eq!(segment.render_index, 14_000);
        assert_eq!(segment.extra_material_refs, vec![MaterialId::new("a")]);
        assert_eq!(segment.speed, 1.0);
        let json = serde_json::to_value(&segment).unwrap();
        assert!(json.get("target_timerange").is_some());
        assert!(json.get("target_time_range").is_none());
    }

    #[test]
    fn responsive_layout_keeps_unnamed_keys() {
        let layout: ResponsiveLayout = serde_json::from_str(r#"{"future_layout":1}"#).unwrap();
        assert_eq!(serde_json::to_value(layout).unwrap()["future_layout"], 1);
    }
}
