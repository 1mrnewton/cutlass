use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::id::MaterialId;

mod adjust;
mod audio;
mod color;
mod draft;
mod effect;
mod shape;
mod speed;
mod text;
mod transition;
mod video;
mod words;

pub use adjust::{MaterialHsl, MaterialPictureAdjust};
pub use audio::{MaterialAudio, MaterialAudioFade};
pub use color::{MaterialCanvas, MaterialColorConfig};
pub use draft::MaterialDraft;
pub use effect::{
    MaterialAnimations, MaterialEffect, MaterialPluginEffect, MaterialVideoEffect, StickerAnimation,
};
pub use shape::MaterialShape;
pub use speed::{
    ConstantSpeedError, CurveSpeed, MaterialSpeed, SpeedPoint, constant_target_duration,
    curve_speed_factor, curve_target_duration,
};
pub use text::MaterialText;
pub use transition::{MaterialTransition, blend_extents, blend_window};
pub use video::MaterialVideo;
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
        take!(
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
        );
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
