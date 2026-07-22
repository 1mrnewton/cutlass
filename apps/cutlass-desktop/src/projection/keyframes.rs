use super::*;

/// Project one animatable property's keyframes for the UI: clip-relative
/// engine ticks become ABSOLUTE sequence ticks (start + offset), easing is
/// flattened to the Slint encoding, and `split` maps the value into the
/// `(value-x, value-y)` pair (scalars leave y at 0). Empty ⇔ constant.
pub(super) fn keyframes_to_slint<T: Lerp>(
    param: &Param<T>,
    clip_start: i64,
    split: impl Fn(&T) -> (f32, f32),
) -> ModelRc<ParamKeyframe> {
    let rows: Vec<ParamKeyframe> = param
        .keyframes()
        .iter()
        .map(|kf: &Keyframe<T>| {
            let (value_x, value_y) = split(&kf.value);
            let (easing, [bez_x1, bez_y1, bez_x2, bez_y2]) = easing_to_ui(kf.easing);
            let (has_tangents, out_tx, out_ty, in_tx, in_ty) = match kf.tangents {
                Some(t) => (true, t.out_t[0], t.out_t[1], t.in_t[0], t.in_t[1]),
                None => (false, 0.0, 0.0, 0.0, 0.0),
            };
            ParamKeyframe {
                tick: clamp_i32(clip_start + kf.tick),
                value_x,
                value_y,
                easing,
                bez_x1,
                bez_y1,
                bez_x2,
                bez_y2,
                has_tangents,
                out_tx,
                out_ty,
                in_tx,
                in_ty,
            }
        })
        .collect();
    model(rows)
}

pub(super) fn empty_keyframes() -> ModelRc<ParamKeyframe> {
    model(Vec::new())
}

pub(super) fn text_keyframes(
    clip: &EngineClip,
    clip_start: i64,
    param: impl FnOnce(&EngineTextStyle) -> &Param<f32>,
) -> ModelRc<ParamKeyframe> {
    match &clip.content {
        ClipSource::Generated(Generator::Text { style, .. }) => {
            keyframes_to_slint(param(style), clip_start, |v| (*v, 0.0))
        }
        _ => empty_keyframes(),
    }
}

pub(super) fn text_color_keyframes(
    clip: &EngineClip,
    clip_start: i64,
    param: impl FnOnce(&EngineTextStyle) -> &Param<[u8; 4]>,
) -> ModelRc<ParamKeyframe> {
    match &clip.content {
        ClipSource::Generated(Generator::Text { style, .. }) => {
            keyframes_to_slint(param(style), clip_start, |_| (0.0, 0.0))
        }
        _ => empty_keyframes(),
    }
}

pub(super) fn text_stroke_keyframes(
    clip: &EngineClip,
    clip_start: i64,
    param: impl FnOnce(&cutlass_models::TextStroke) -> &Param<f32>,
) -> ModelRc<ParamKeyframe> {
    match &clip.content {
        ClipSource::Generated(Generator::Text { style, .. }) => style
            .stroke
            .as_ref()
            .map_or_else(empty_keyframes, |stroke| {
                keyframes_to_slint(param(stroke), clip_start, |v| (*v, 0.0))
            }),
        _ => empty_keyframes(),
    }
}

pub(super) fn text_stroke_color_keyframes(
    clip: &EngineClip,
    clip_start: i64,
    param: impl FnOnce(&cutlass_models::TextStroke) -> &Param<[u8; 4]>,
) -> ModelRc<ParamKeyframe> {
    match &clip.content {
        ClipSource::Generated(Generator::Text { style, .. }) => style
            .stroke
            .as_ref()
            .map_or_else(empty_keyframes, |stroke| {
                keyframes_to_slint(param(stroke), clip_start, |_| (0.0, 0.0))
            }),
        _ => empty_keyframes(),
    }
}

pub(super) fn text_background_keyframes(
    clip: &EngineClip,
    clip_start: i64,
    param: impl FnOnce(&cutlass_models::TextBackground) -> &Param<f32>,
) -> ModelRc<ParamKeyframe> {
    match &clip.content {
        ClipSource::Generated(Generator::Text { style, .. }) => style
            .background
            .as_ref()
            .map_or_else(empty_keyframes, |background| {
                keyframes_to_slint(param(background), clip_start, |v| (*v, 0.0))
            }),
        _ => empty_keyframes(),
    }
}

pub(super) fn text_background_color_keyframes(
    clip: &EngineClip,
    clip_start: i64,
    param: impl FnOnce(&cutlass_models::TextBackground) -> &Param<[u8; 4]>,
) -> ModelRc<ParamKeyframe> {
    match &clip.content {
        ClipSource::Generated(Generator::Text { style, .. }) => style
            .background
            .as_ref()
            .map_or_else(empty_keyframes, |background| {
                keyframes_to_slint(param(background), clip_start, |_| (0.0, 0.0))
            }),
        _ => empty_keyframes(),
    }
}

pub(super) fn text_shadow_keyframes(
    clip: &EngineClip,
    clip_start: i64,
    param: impl FnOnce(&cutlass_models::TextShadow) -> &Param<f32>,
) -> ModelRc<ParamKeyframe> {
    match &clip.content {
        ClipSource::Generated(Generator::Text { style, .. }) => style
            .shadow
            .as_ref()
            .map_or_else(empty_keyframes, |shadow| {
                keyframes_to_slint(param(shadow), clip_start, |v| (*v, 0.0))
            }),
        _ => empty_keyframes(),
    }
}

pub(super) fn text_shadow_color_keyframes(
    clip: &EngineClip,
    clip_start: i64,
    param: impl FnOnce(&cutlass_models::TextShadow) -> &Param<[u8; 4]>,
) -> ModelRc<ParamKeyframe> {
    match &clip.content {
        ClipSource::Generated(Generator::Text { style, .. }) => style
            .shadow
            .as_ref()
            .map_or_else(empty_keyframes, |shadow| {
                keyframes_to_slint(param(shadow), clip_start, |_| (0.0, 0.0))
            }),
        _ => empty_keyframes(),
    }
}
