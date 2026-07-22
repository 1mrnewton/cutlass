//! Animated scalar-channel catalog for the keyframe graph editor.
//! Colors are omitted; vec2 params expose separate X/Y (or W/H) channels.

use cutlass_models::{Keyframe, Param};
use slint::{Model, SharedString};

use super::easing_of;
use crate::{Clip, GraphChannel, ParamKeyframe};

/// One animatable scalar (or vec2-axis) channel descriptor.
#[derive(Clone, Copy)]
struct ChannelDesc {
    key: &'static str,
    channel: i32,
    label: &'static str,
}

macro_rules! ch {
    ($key:literal, $ch:expr, $label:literal) => {
        ChannelDesc {
            key: $key,
            channel: $ch,
            label: $label,
        }
    };
}

/// Scalar / vec2 channels the graph can display (colors omitted).
const CHANNEL_CATALOG: &[ChannelDesc] = &[
    ch!("position", 0, "Position X"),
    ch!("position", 1, "Position Y"),
    ch!("anchor", 0, "Anchor X"),
    ch!("anchor", 1, "Anchor Y"),
    ch!("scale", 0, "Scale X"),
    ch!("scale", 1, "Scale Y"),
    ch!("rotation", 0, "Rotation"),
    ch!("opacity", 0, "Opacity"),
    ch!("volume", 0, "Volume"),
    ch!("pan", 0, "Pan"),
    ch!("text_size", 0, "Text Size"),
    ch!("text_letter_spacing", 0, "Letter Spacing"),
    ch!("text_line_spacing", 0, "Line Spacing"),
    ch!("text_stroke_width", 0, "Stroke Width"),
    ch!("text_background_radius", 0, "BG Radius"),
    ch!("text_shadow_blur", 0, "Shadow Blur"),
    ch!("text_shadow_distance", 0, "Shadow Distance"),
    ch!("look_filter_intensity", 0, "Filter Intensity"),
    ch!("look_lut_intensity", 0, "LUT Intensity"),
    ch!("look_adjust_brightness", 0, "Brightness"),
    ch!("look_adjust_contrast", 0, "Contrast"),
    ch!("look_adjust_saturation", 0, "Saturation"),
    ch!("look_adjust_exposure", 0, "Exposure"),
    ch!("look_adjust_temperature", 0, "Temperature"),
    ch!("look_adjust_tint", 0, "Tint"),
    ch!("look_adjust_hue", 0, "Hue"),
    ch!("look_adjust_highlights", 0, "Highlights"),
    ch!("look_adjust_shadows", 0, "Shadows"),
    ch!("look_adjust_sharpness", 0, "Sharpness"),
    ch!("look_adjust_vignette", 0, "Vignette"),
    ch!("style_shadow_offset", 0, "Shadow Offset X"),
    ch!("style_shadow_offset", 1, "Shadow Offset Y"),
    ch!("style_shadow_blur", 0, "Shadow Blur"),
    ch!("style_glow_radius", 0, "Glow Radius"),
    ch!("style_glow_intensity", 0, "Glow Intensity"),
    ch!("style_outline_width", 0, "Outline Width"),
    ch!("style_background_padding", 0, "BG Padding"),
    ch!("style_background_radius", 0, "BG Radius"),
    ch!("look_mask_feather", 0, "Mask Feather"),
    ch!("look_mask_center", 0, "Mask Center X"),
    ch!("look_mask_center", 1, "Mask Center Y"),
    ch!("look_mask_size", 0, "Mask Size W"),
    ch!("look_mask_size", 1, "Mask Size H"),
    ch!("look_mask_rotation", 0, "Mask Rotation"),
    ch!("look_mask_roundness", 0, "Mask Roundness"),
    ch!("look_chroma_strength", 0, "Chroma Strength"),
    ch!("look_chroma_shadow", 0, "Chroma Shadow"),
];

pub(super) fn kf_list<'a>(clip: &'a Clip, key: &str) -> Option<&'a slint::ModelRc<ParamKeyframe>> {
    Some(match key {
        "position" => &clip.kf_position,
        "anchor" => &clip.kf_anchor,
        "scale" => &clip.kf_scale,
        "rotation" => &clip.kf_rotation,
        "opacity" => &clip.kf_opacity,
        "volume" => &clip.kf_volume,
        "pan" => &clip.kf_pan,
        "text_size" => &clip.kf_text_size,
        "text_letter_spacing" => &clip.kf_text_letter_spacing,
        "text_line_spacing" => &clip.kf_text_line_spacing,
        "text_stroke_width" => &clip.kf_text_stroke_width,
        "text_background_radius" => &clip.kf_text_background_radius,
        "text_shadow_blur" => &clip.kf_text_shadow_blur,
        "text_shadow_distance" => &clip.kf_text_shadow_distance,
        "look_filter_intensity" => &clip.kf_look_filter_intensity,
        "look_lut_intensity" => &clip.kf_look_lut_intensity,
        "look_adjust_brightness" => &clip.kf_look_adjust_brightness,
        "look_adjust_contrast" => &clip.kf_look_adjust_contrast,
        "look_adjust_saturation" => &clip.kf_look_adjust_saturation,
        "look_adjust_exposure" => &clip.kf_look_adjust_exposure,
        "look_adjust_temperature" => &clip.kf_look_adjust_temperature,
        "look_adjust_tint" => &clip.kf_look_adjust_tint,
        "look_adjust_hue" => &clip.kf_look_adjust_hue,
        "look_adjust_highlights" => &clip.kf_look_adjust_highlights,
        "look_adjust_shadows" => &clip.kf_look_adjust_shadows,
        "look_adjust_sharpness" => &clip.kf_look_adjust_sharpness,
        "look_adjust_vignette" => &clip.kf_look_adjust_vignette,
        "style_shadow_offset" => &clip.kf_style_shadow_offset,
        "style_shadow_blur" => &clip.kf_style_shadow_blur,
        "style_glow_radius" => &clip.kf_style_glow_radius,
        "style_glow_intensity" => &clip.kf_style_glow_intensity,
        "style_outline_width" => &clip.kf_style_outline_width,
        "style_background_padding" => &clip.kf_style_background_padding,
        "style_background_radius" => &clip.kf_style_background_radius,
        "look_mask_feather" => &clip.kf_look_mask_feather,
        "look_mask_center" => &clip.kf_look_mask_center,
        "look_mask_size" => &clip.kf_look_mask_size,
        "look_mask_rotation" => &clip.kf_look_mask_rotation,
        "look_mask_roundness" => &clip.kf_look_mask_roundness,
        "look_chroma_strength" => &clip.kf_look_chroma_strength,
        "look_chroma_shadow" => &clip.kf_look_chroma_shadow,
        _ => return None,
    })
}

fn is_vec2_key(key: &str) -> bool {
    matches!(
        key,
        "position"
            | "anchor"
            | "scale"
            | "style_shadow_offset"
            | "look_mask_center"
            | "look_mask_size"
    )
}

/// Animated scalar channels on `clip` (non-empty kf lists only; colors skipped).
pub fn animated_channels(clip: &Clip) -> Vec<GraphChannel> {
    let mut out = Vec::new();
    for desc in CHANNEL_CATALOG {
        if !is_vec2_key(desc.key) && desc.channel != 0 {
            continue;
        }
        let Some(kfs) = kf_list(clip, desc.key) else {
            continue;
        };
        if kfs.row_count() == 0 {
            continue;
        }
        out.push(GraphChannel {
            key: SharedString::from(desc.key),
            channel: desc.channel,
            label: SharedString::from(desc.label),
        });
    }
    out
}

fn scalar_from_kfs(kfs: &slint::ModelRc<ParamKeyframe>, channel: i32) -> Param<f32> {
    let ch = if channel <= 0 { 0 } else { 1 };
    let keyframes: Vec<Keyframe<f32>> = kfs
        .iter()
        .map(|kf| Keyframe {
            tick: i64::from(kf.tick),
            value: if ch == 0 { kf.value_x } else { kf.value_y },
            easing: easing_of(&kf),
            tangents: None,
        })
        .collect();
    if keyframes.is_empty() {
        Param::Constant(0.0)
    } else {
        Param::Keyframed { keyframes }
    }
}

/// Rebuild a scalar `Param` for the graph channel (`channel` 0/1 on vec2).
pub fn channel_param(clip: &Clip, key: &str, channel: i32) -> Option<Param<f32>> {
    let kfs = kf_list(clip, key)?;
    if kfs.row_count() == 0 {
        return None;
    }
    Some(scalar_from_kfs(kfs, channel))
}
