use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::id::MaterialId;

mod adjust;
mod audio;
mod beat;
mod caption;
mod color;
mod draft;
mod effect;
mod mark;
mod rich;
mod shape;
mod speed;
mod text;
mod transition;
mod video;
mod words;

pub use adjust::{MaterialHsl, MaterialPictureAdjust};
pub use audio::{MaterialAudio, MaterialAudioFade};
pub use beat::{
    VisibleBeat, beat_timeline_time, new_beats, segment_beats, set_user_beats, source_time_at,
    track_holds_beats, user_beats, visible_beats,
};
pub use caption::{CaptionTemplateInfo, Kept, SubtitleKeywords, TextRange, keywords_in_ranges};
pub use color::{MaterialCanvas, MaterialColorConfig};
pub use draft::MaterialDraft;
pub use effect::{
    BloomParams, MaterialAnimations, MaterialEffect, MaterialPluginEffect, MaterialVideoEffect,
    StickerAnimation, segment_bloom,
};
pub use mark::{
    TimeMarkItem, VisibleMark, clip_mark_time, new_time_mark, referenced_time_marks,
    segment_time_mark, set_time_mark_items, time_mark_items, visible_marks,
};
pub use rich::{
    RichFill, RichFont, RichPaint, RichShadow, RichSolid, RichStroke, RichStyle, RichText,
};
pub use shape::{
    MaterialShape, SHAPE_ARROW, SHAPE_CONTOUR, SHAPE_FILL, SHAPE_LINE, SHAPE_LINE_WIDTH, SHAPE_PEN,
    SHAPE_SHADOW, SHAPE_SIZE, SHAPE_STROKE_WIDTH, ShapeKind, ShapePoint,
};
pub use speed::{
    ConstantSpeedError, CurveSpeed, MaterialSpeed, SpeedPoint, SpeedPreset,
    constant_target_duration, curve_speed_at, curve_speed_factor, curve_target_duration,
    split_curve,
};
pub use text::{Bubble, CAPTION_STYLE_KEYS, MaterialText};
pub use transition::{MaterialTransition, blend_extents, blend_window};
pub use video::{Crop, FreezeInfo, MaterialVideo, VideoStable};
pub use words::WordColumns;

/// A material this model does not name field by field. `id` and `type` are lifted out.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct RawMaterial {
    #[serde(default)]
    pub id: MaterialId,
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(flatten, default)]
    pub fields: Map<String, Value>,
}

/// Every pool written under `materials`. Empty pools are still present in the file.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Materials {
    #[serde(default)]
    pub videos: Vec<MaterialVideo>,
    #[serde(default)]
    pub audios: Vec<MaterialAudio>,
    #[serde(default)]
    pub texts: Vec<MaterialText>,
    #[serde(default)]
    pub speeds: Vec<MaterialSpeed>,
    #[serde(default)]
    pub audio_fades: Vec<MaterialAudioFade>,
    #[serde(default)]
    pub transitions: Vec<MaterialTransition>,
    #[serde(default)]
    pub material_animations: Vec<MaterialAnimations>,
    #[serde(default)]
    pub canvases: Vec<MaterialCanvas>,
    #[serde(default)]
    pub drafts: Vec<MaterialDraft>,
    #[serde(default)]
    pub plugin_effects: Vec<MaterialPluginEffect>,
    #[serde(default)]
    pub flowers: Vec<RawMaterial>,
    #[serde(default)]
    pub tail_leaders: Vec<RawMaterial>,
    #[serde(default)]
    pub images: Vec<RawMaterial>,
    #[serde(default)]
    pub effects: Vec<MaterialEffect>,
    #[serde(default)]
    pub stickers: Vec<RawMaterial>,
    #[serde(default)]
    pub audio_effects: Vec<RawMaterial>,
    #[serde(default)]
    pub beats: Vec<RawMaterial>,
    #[serde(default)]
    pub placeholders: Vec<RawMaterial>,
    #[serde(default)]
    pub placeholder_infos: Vec<RawMaterial>,
    #[serde(default)]
    pub common_mask: Vec<RawMaterial>,
    #[serde(default)]
    pub chromas: Vec<RawMaterial>,
    #[serde(default)]
    pub text_templates: Vec<RawMaterial>,
    #[serde(default)]
    pub realtime_denoises: Vec<RawMaterial>,
    #[serde(default)]
    pub audio_pannings: Vec<RawMaterial>,
    #[serde(default)]
    pub audio_pitch_shifts: Vec<RawMaterial>,
    #[serde(default)]
    pub video_trackings: Vec<RawMaterial>,
    #[serde(default)]
    pub hsl: Vec<MaterialHsl>,
    #[serde(default)]
    pub color_curves: Vec<RawMaterial>,
    #[serde(default)]
    pub hsl_curves: Vec<RawMaterial>,
    #[serde(default)]
    pub primary_color_wheels: Vec<RawMaterial>,
    #[serde(default)]
    pub log_color_wheels: Vec<RawMaterial>,
    #[serde(default)]
    pub video_effects: Vec<MaterialVideoEffect>,
    #[serde(default)]
    pub ai_text_effects: Vec<RawMaterial>,
    #[serde(default)]
    pub ai_generate_holders: Vec<RawMaterial>,
    #[serde(default)]
    pub audio_balances: Vec<RawMaterial>,
    #[serde(default)]
    pub handwrites: Vec<RawMaterial>,
    #[serde(default)]
    pub manual_deformations: Vec<RawMaterial>,
    #[serde(default)]
    pub manual_beautys: Vec<RawMaterial>,
    #[serde(default)]
    pub sound_channel_mappings: Vec<RawMaterial>,
    #[serde(default)]
    pub green_screens: Vec<RawMaterial>,
    #[serde(default)]
    pub shapes: Vec<MaterialShape>,
    #[serde(default)]
    pub material_colors: Vec<MaterialColorConfig>,
    #[serde(default)]
    pub digital_humans: Vec<RawMaterial>,
    #[serde(default)]
    pub digital_human_model_dressing: Vec<RawMaterial>,
    #[serde(default)]
    pub smart_crops: Vec<RawMaterial>,
    #[serde(default)]
    pub ai_translates: Vec<RawMaterial>,
    #[serde(default)]
    pub audio_track_indexes: Vec<RawMaterial>,
    #[serde(default)]
    pub loudnesses: Vec<RawMaterial>,
    #[serde(default)]
    pub vocal_beautifys: Vec<RawMaterial>,
    #[serde(default)]
    pub vocal_separations: Vec<RawMaterial>,
    #[serde(default)]
    pub smart_relights: Vec<RawMaterial>,
    #[serde(default)]
    pub time_marks: Vec<RawMaterial>,
    #[serde(default)]
    pub multi_language_refs: Vec<RawMaterial>,
    #[serde(default)]
    pub video_shadows: Vec<RawMaterial>,
    #[serde(default)]
    pub video_strokes: Vec<RawMaterial>,
    #[serde(default)]
    pub video_radius: Vec<RawMaterial>,
    /// Pool names this build has not listed.
    #[serde(flatten, default)]
    pub extra_pools: Map<String, Value>,
}

/// A material found by id, in this draft only.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MaterialRef<'a> {
    Video(&'a MaterialVideo),
    Audio(&'a MaterialAudio),
    Text(&'a MaterialText),
    Speed(&'a MaterialSpeed),
    Fade(&'a MaterialAudioFade),
    Transition(&'a MaterialTransition),
    Animations(&'a MaterialAnimations),
    Canvas(&'a MaterialCanvas),
    Draft(&'a MaterialDraft),
    Plugin(&'a MaterialPluginEffect),
    Effect(&'a MaterialEffect),
    Hsl(&'a MaterialHsl),
    VideoEffect(&'a MaterialVideoEffect),
    Shape(&'a MaterialShape),
    Color(&'a MaterialColorConfig),
    Other(&'a RawMaterial),
    /// An item in a pool this model does not list.
    Extra(&'a Value),
}

/// Calls `$then!` with every typed pool field and how `get` wraps its items.
macro_rules! each_pool {
    ($then:ident) => {
        $then!(
                videos, MaterialRef::Video;
                audios, MaterialRef::Audio;
                texts, MaterialRef::Text;
                speeds, MaterialRef::Speed;
                audio_fades, MaterialRef::Fade;
                transitions, MaterialRef::Transition;
                material_animations, MaterialRef::Animations;
                canvases, MaterialRef::Canvas;
                drafts, MaterialRef::Draft;
                plugin_effects, MaterialRef::Plugin;
                flowers, MaterialRef::Other;
                tail_leaders, MaterialRef::Other;
                images, MaterialRef::Other;
                effects, MaterialRef::Effect;
                stickers, MaterialRef::Other;
                audio_effects, MaterialRef::Other;
                beats, MaterialRef::Other;
                placeholders, MaterialRef::Other;
                placeholder_infos, MaterialRef::Other;
                common_mask, MaterialRef::Other;
                chromas, MaterialRef::Other;
                text_templates, MaterialRef::Other;
                realtime_denoises, MaterialRef::Other;
                audio_pannings, MaterialRef::Other;
                audio_pitch_shifts, MaterialRef::Other;
                video_trackings, MaterialRef::Other;
                hsl, MaterialRef::Hsl;
                color_curves, MaterialRef::Other;
                hsl_curves, MaterialRef::Other;
                primary_color_wheels, MaterialRef::Other;
                log_color_wheels, MaterialRef::Other;
                video_effects, MaterialRef::VideoEffect;
                ai_text_effects, MaterialRef::Other;
                ai_generate_holders, MaterialRef::Other;
                audio_balances, MaterialRef::Other;
                handwrites, MaterialRef::Other;
                manual_deformations, MaterialRef::Other;
                manual_beautys, MaterialRef::Other;
                sound_channel_mappings, MaterialRef::Other;
                green_screens, MaterialRef::Other;
                shapes, MaterialRef::Shape;
                material_colors, MaterialRef::Color;
                digital_humans, MaterialRef::Other;
                digital_human_model_dressing, MaterialRef::Other;
                smart_crops, MaterialRef::Other;
                ai_translates, MaterialRef::Other;
                audio_track_indexes, MaterialRef::Other;
                loudnesses, MaterialRef::Other;
                vocal_beautifys, MaterialRef::Other;
                vocal_separations, MaterialRef::Other;
                smart_relights, MaterialRef::Other;
                time_marks, MaterialRef::Other;
                multi_language_refs, MaterialRef::Other;
                video_shadows, MaterialRef::Other;
                video_strokes, MaterialRef::Other;
                video_radius, MaterialRef::Other
        )
    };
}

impl Materials {
    /// The material with this id, searching this draft's pools. Child drafts are separate.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<MaterialRef<'_>> {
        if id.is_empty() {
            return None;
        }
        macro_rules! take {
            ($($field:ident, $wrap:path);* $(;)?) => {
                $(
                    if let Some(item) = self.$field.iter().find(|item| item.id.as_str() == id) {
                        return Some($wrap(item));
                    }
                )*
            };
        }
        each_pool!(take);
        for pool in self.extra_pools.values() {
            let Some(items) = pool.as_array() else {
                continue;
            };
            for item in items {
                if item.get("id").and_then(Value::as_str) == Some(id) {
                    return Some(MaterialRef::Extra(item));
                }
            }
        }
        None
    }

    /// Whether the videos `a` and `b` show the same file: the same id, or
    /// one a clip's own copy of the other.
    #[must_use]
    pub fn same_video(&self, a: &str, b: &str) -> bool {
        if a == b {
            return true;
        }
        let path = |id: &str| {
            self.videos
                .iter()
                .find(|video| video.id.as_str() == id)
                .map(|video| video.path.as_str())
                .filter(|path| !path.is_empty())
        };
        path(a).is_some_and(|left| path(b) == Some(left))
    }

    /// Keep only the materials whose id `keep` answers true for.
    pub fn retain(&mut self, keep: impl Fn(&str) -> bool) {
        macro_rules! keep_in {
            ($($field:ident, $wrap:path);* $(;)?) => {
                $(
                    self.$field.retain(|item| keep(item.id.as_str()));
                )*
            };
        }
        each_pool!(keep_in);
        for pool in self.extra_pools.values_mut() {
            if let Some(items) = pool.as_array_mut() {
                items.retain(|item| item.get("id").and_then(Value::as_str).is_none_or(&keep));
            }
        }
    }

    /// Move every material of `other` into these pools, after the ones here.
    pub fn absorb(&mut self, other: Materials) {
        let mut other = other;
        macro_rules! move_in {
            ($($field:ident, $wrap:path);* $(;)?) => {
                $(
                    self.$field.append(&mut other.$field);
                )*
            };
        }
        each_pool!(move_in);
        for (name, pool) in other.extra_pools {
            let Value::Array(mut items) = pool else {
                continue;
            };
            match self
                .extra_pools
                .get_mut(&name)
                .and_then(Value::as_array_mut)
            {
                Some(here) => here.append(&mut items),
                None => {
                    self.extra_pools.insert(name, Value::Array(items));
                }
            }
        }
    }

    /// Take the material `id` out of whichever pool holds it. Whether one did.
    pub fn remove(&mut self, id: &str) -> bool {
        if id.is_empty() {
            return false;
        }
        macro_rules! drop_from {
            ($($field:ident, $wrap:path);* $(;)?) => {
                $(
                    if let Some(index) = self.$field.iter().position(|item| item.id.as_str() == id) {
                        self.$field.remove(index);
                        return true;
                    }
                )*
            };
        }
        each_pool!(drop_from);
        for pool in self.extra_pools.values_mut() {
            let Some(items) = pool.as_array_mut() else {
                continue;
            };
            if let Some(index) = items
                .iter()
                .position(|item| item.get("id").and_then(Value::as_str) == Some(id))
            {
                items.remove(index);
                return true;
            }
        }
        false
    }

    /// Add a copy of the material `id` under `new_id`, in the same pool.
    /// Returns `false` when there is no material `id`, or `new_id` is
    /// empty or taken.
    pub fn copy_as(&mut self, id: &str, new_id: &MaterialId) -> bool {
        if id.is_empty() || new_id.is_empty() || self.get(new_id.as_str()).is_some() {
            return false;
        }
        macro_rules! copy {
            ($($field:ident, $wrap:path);* $(;)?) => {
                $(
                    if let Some(item) = self.$field.iter().find(|item| item.id.as_str() == id) {
                        let mut item = item.clone();
                        item.id = new_id.clone();
                        self.$field.push(item);
                        return true;
                    }
                )*
            };
        }
        each_pool!(copy);
        for pool in self.extra_pools.values_mut() {
            let Some(items) = pool.as_array_mut() else {
                continue;
            };
            if let Some(item) = items
                .iter()
                .find(|item| item.get("id").and_then(Value::as_str) == Some(id))
            {
                let mut item = item.clone();
                if let Some(fields) = item.as_object_mut() {
                    fields.insert("id".to_owned(), Value::String(new_id.as_str().to_owned()));
                }
                items.push(item);
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_copy_lands_in_its_pool_under_the_new_id_and_a_taken_id_is_refused() {
        let mut materials: Materials = serde_json::from_str(
            r#"{"audio_fades":[{"id":"f","fade_in_duration":5}],"future_pool":[{"id":"z","n":1}]}"#,
        )
        .unwrap();
        assert!(materials.copy_as("f", &"g".into()));
        assert!(
            matches!(materials.get("g"), Some(MaterialRef::Fade(fade)) if fade.fade_in_duration.0 == 5)
        );
        assert!(materials.copy_as("z", &"y".into()));
        assert!(matches!(materials.get("y"), Some(MaterialRef::Extra(item)) if item["n"] == 1));
        assert!(!materials.copy_as("f", &"y".into()));
        assert!(!materials.copy_as("missing", &"w".into()));
        assert!(!materials.copy_as("f", &"".into()));
    }

    #[test]
    fn lookup_finds_typed_pools_and_keeps_unnamed_keys() {
        let materials: Materials = serde_json::from_str(
            r#"{"texts":[{"id":"t","type":"text","content":"hi","autoAdaptCanvasEnabled":true}],"speeds":[{"id":"s","type":"speed","mode":0,"speed":1.0}],"shapes":[{"id":"sh","roundness":[0.2]}],"effects":[{"id":"ef","value":0.2}],"material_colors":[{"id":"co","solid_color":"white"}],"hsl":[{"id":"h","hue":12}],"video_effects":[{"id":"ve","name":"glow"}]}"#,
        )
        .unwrap();
        assert!(matches!(
            materials.get("t"),
            Some(MaterialRef::Text(text))
                if text.content == "hi" && text.rest["autoAdaptCanvasEnabled"] == true
        ));
        assert!(matches!(materials.get("s"), Some(MaterialRef::Speed(_))));
        assert!(matches!(
            materials.get("sh"),
            Some(MaterialRef::Shape(shape)) if shape.roundness == vec![0.2]
        ));
        assert!(matches!(
            materials.get("ef"),
            Some(MaterialRef::Effect(effect)) if effect.value == Some(0.2)
        ));
        assert!(matches!(materials.get("co"), Some(MaterialRef::Color(_))));
        assert!(matches!(materials.get("h"), Some(MaterialRef::Hsl(_))));
        assert!(matches!(
            materials.get("ve"),
            Some(MaterialRef::VideoEffect(_))
        ));
        assert!(materials.get("").is_none());
    }

    #[test]
    fn lookup_does_not_descend_into_a_child_draft() {
        let materials: Materials = serde_json::from_str(
            r#"{"drafts":[{"id":"nest","draft":{"id":"child","version":1,"materials":{"texts":[{"id":"inner","content":"x"}]}}}]}"#,
        )
        .unwrap();
        assert!(materials.get("inner").is_none());
        assert!(matches!(materials.get("nest"), Some(MaterialRef::Draft(_))));
    }
}
