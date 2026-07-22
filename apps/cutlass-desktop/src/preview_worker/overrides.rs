use super::*;

/// Fit/fill helper (M1 canvas settings): compute the centered fit (scale
/// 1.0) or cover transform for a clip and commit it through the regular
/// `SetClipTransform` path, so it keyframes at the playhead on animated
/// clips and undoes in one step like any gesture.
pub(super) fn fit_clip_and_publish(
    engine: &mut Engine,
    clip: &str,
    fill: bool,
    tick: i64,
    tl_rate: Rational,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "fit/fill ignored: unparsable clip id");
        return;
    };
    let Some(transform) = fit_clip_transform(engine, clip_id, fill, tick) else {
        error!(%clip_id, "fit/fill ignored: unknown clip or degenerate content");
        return;
    };
    let at = RationalTime::new(tick, tl_rate);
    set_transform_and_publish(engine, clip, transform, at, ui);
}

/// The transform that centers a clip at aspect-fit (scale 1.0 by the
/// placement convention) or at the cover scale that fills the canvas — the
/// crop's kept region is what aspect-fits, so it is also what must cover.
/// Rotation and opacity keep their playhead-sampled values; position resets
/// to center (CapCut fit/fill semantics).
pub(super) fn fit_clip_transform(
    engine: &Engine,
    clip_id: ClipId,
    fill: bool,
    tick: i64,
) -> Option<ClipTransform> {
    let project = engine.project();
    let clip = project.clip(clip_id)?;
    let (canvas_w, canvas_h) = cutlass_render::canvas_size(project);
    let (content_w, content_h) = match clip.media() {
        Some(media_id) => {
            let media = project.media(media_id)?;
            (media.width, media.height)
        }
        // Generators raster at canvas size: fit and fill are both 1.0.
        None => (canvas_w, canvas_h),
    };
    let crop = clip.crop.sample(clip.animation_tick(tick));
    let (w, h) = (content_w as f32 * crop.w, content_h as f32 * crop.h);
    if w <= 0.0 || h <= 0.0 || canvas_w == 0 || canvas_h == 0 {
        return None;
    }
    let (cw, ch) = (canvas_w as f32, canvas_h as f32);
    let fit = (cw / w).min(ch / h);
    let cover = (cw / w).max(ch / h);
    // Fit/fill writes a uniform scale (both axes).
    let scale = cutlass_models::Scale2::uniform(if fill { cover / fit } else { 1.0 });
    let sampled = clip.transform.sample_at(clip.animation_tick_f(tick as f64));
    Some(ClipTransform {
        position: [0.0, 0.0],
        anchor_point: sampled.anchor_point,
        scale,
        rotation: sampled.rotation,
        opacity: sampled.opacity,
    })
}

/// Commit a transform gesture as one undoable `SetClipTransform`, keyframing
/// at `at` (the playhead) when the property is animated.
pub(super) fn set_transform_and_publish(
    engine: &mut Engine,
    clip: &str,
    transform: ClipTransform,
    at: RationalTime,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-transform ignored: unparsable clip id");
        return;
    };
    // CapCut compose semantics: on a clip with animated properties this
    // commit writes keyframes at the playhead instead of flattening. Note it
    // before applying so the UI can surface "a gesture added a keyframe".
    let wrote_keyframe = engine
        .project()
        .clip(clip_id)
        .is_some_and(|c| c.transform.is_animated());
    match engine.apply(Command::Edit(EditCommand::SetClipTransform {
        clip: clip_id,
        transform,
        at: Some(at),
    })) {
        Ok(_) => {
            info!(%clip_id, ?transform, "set clip transform");
            if wrote_keyframe {
                bump_keyframe_commit_epoch(ui);
            }
            publish_projection(engine, ui);
        }
        Err(e) => {
            error!(%clip_id, "set transform failed: {e}");
            publish_projection(engine, ui);
        }
    }
}

/// Point the engine's transform override at `clip` (raw id) for the next
/// renders — the live preview of an in-flight gesture. Unparsable ids are
/// dropped (stale projection race).
pub(super) fn apply_transform_override(engine: &mut Engine, clip: &str, transform: ClipTransform) {
    match parse_raw_id(clip).map(ClipId::from_raw) {
        Some(id) => engine.set_transform_override(Some((id, transform))),
        None => error!(clip, "transform override ignored: unparsable clip id"),
    }
}

/// Point the engine's generator override at `clip` (raw id) for the next
/// renders — the live preview of an uncommitted inspector edit. Unparsable
/// ids are dropped (stale projection race), same as the transform override.
pub(super) fn apply_generator_override(engine: &mut Engine, clip: &str, generator: Generator) {
    match parse_raw_id(clip).map(ClipId::from_raw) {
        Some(id) => engine.set_generator_override(Some((id, generator))),
        None => error!(clip, "generator override ignored: unparsable clip id"),
    }
}

/// Point the engine's look override at `clip` (raw id) for the next renders —
/// the live preview of an uncommitted filter/adjustment edit. Unparsable ids
/// are dropped (stale projection race), same as the other overrides.
pub(super) fn apply_look_override(
    engine: &mut Engine,
    clip: &str,
    filter_id: &str,
    intensity: f32,
    adjust: &ColorAdjustments,
) {
    match parse_raw_id(clip).map(ClipId::from_raw) {
        Some(id) => engine.set_look_override(Some((
            id,
            filter_from_ui(filter_id, intensity),
            sanitize_adjustments(adjust),
        ))),
        None => error!(clip, "look override ignored: unparsable clip id"),
    }
}

/// Point the engine's styles override at `clip` (raw id) for the next renders —
/// the live preview of an uncommitted layer-style edit. Unparsable ids are
/// dropped (stale projection race), same as the other overrides.
pub(super) fn apply_styles_override(engine: &mut Engine, clip: &str, styles: LayerStyles) {
    match parse_raw_id(clip).map(ClipId::from_raw) {
        Some(id) => engine.set_styles_override(Some((id, styles))),
        None => error!(clip, "styles override ignored: unparsable clip id"),
    }
}

pub(super) fn filter_from_ui(filter_id: &str, intensity: f32) -> Option<Filter> {
    let id = filter_id.trim();
    if id.is_empty() {
        return None;
    }
    Some(Filter {
        id: id.to_string(),
        intensity: clamp_unit(intensity).into(),
    })
}

pub(super) fn sanitize_adjustments(adjust: &ColorAdjustments) -> ColorAdjustments {
    ColorAdjustments {
        brightness: clamp_signed_unit(adjust.brightness.sample(0)).into(),
        contrast: clamp_signed_unit(adjust.contrast.sample(0)).into(),
        saturation: clamp_signed_unit(adjust.saturation.sample(0)).into(),
        exposure: clamp_signed_unit(adjust.exposure.sample(0)).into(),
        temperature: clamp_signed_unit(adjust.temperature.sample(0)).into(),
        tint: clamp_signed_unit(adjust.tint.sample(0)).into(),
        hue: clamp_signed_unit(adjust.hue.sample(0)).into(),
        highlights: clamp_signed_unit(adjust.highlights.sample(0)).into(),
        shadows: clamp_signed_unit(adjust.shadows.sample(0)).into(),
        sharpness: clamp_unit(adjust.sharpness.sample(0)).into(),
        vignette: clamp_unit(adjust.vignette.sample(0)).into(),
    }
}

pub(super) fn clamp_unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

pub(super) fn clamp_signed_unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Signal the inspector that a transform gesture just wrote keyframes (the
/// transient "keyframe added" chip): bump `EditorStore.keyframe-commit-epoch`.
pub(super) fn bump_keyframe_commit_epoch(ui: &UiSink) {
    let editor_weak = ui.editor.clone();
    if let Err(e) = slint::invoke_from_event_loop(move || {
        if let Some(store) = editor_weak.upgrade() {
            store.set_keyframe_commit_epoch(store.get_keyframe_commit_epoch().wrapping_add(1));
        }
    }) {
        error!("failed to bump keyframe commit epoch: {e}");
    }
}

/// Insert or replace one property keyframe at `at` (absolute playhead
/// position) as one undoable edit (keyframes roadmap Phase 1: the inspector
/// diamond / easing picker). Engine-rejected positions (playhead outside the
/// clip — the UI gates, but a stale projection can race) only log.
/// Style-param commits clear a live styles override first.
pub(super) fn set_param_keyframe_and_publish(
    engine: &mut Engine,
    clip: &str,
    param: ClipParam,
    at: RationalTime,
    value: ParamValue,
    easing: Easing,
    ui: &UiSink,
) {
    if matches!(param, ClipParam::Style { .. }) {
        engine.set_styles_override(None);
    }
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-param-keyframe ignored: unparsable clip id");
        return;
    };
    match engine.apply(Command::Edit(EditCommand::SetParamKeyframe {
        clip: clip_id,
        param,
        at,
        value,
        easing,
        tangents: None,
    })) {
        Ok(_) => {
            info!(%clip_id, ?param, tick = at.value, "set param keyframe");
            publish_projection(engine, ui);
        }
        Err(e) => error!(%clip_id, ?param, "set param keyframe failed: {e}"),
    }
}

/// Replace one animatable property with a constant (inspector slider commit
/// on a non-animated row). Drops any existing keyframes on that property.
/// Style-param commits clear a live styles override first (look clears on
/// filter/adjust commit the same way).
pub(super) fn set_param_constant_and_publish(
    engine: &mut Engine,
    clip: &str,
    param: ClipParam,
    value: ParamValue,
    ui: &UiSink,
) {
    if matches!(param, ClipParam::Style { .. }) {
        engine.set_styles_override(None);
    }
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-param-constant ignored: unparsable clip id");
        return;
    };
    match engine.apply(Command::Edit(EditCommand::SetParamConstant {
        clip: clip_id,
        param,
        value,
    })) {
        Ok(_) => {
            info!(%clip_id, ?param, "set param constant");
            publish_projection(engine, ui);
        }
        Err(e) => error!(%clip_id, ?param, "set param constant failed: {e}"),
    }
}

/// Remove the keyframe at exactly `at` on one property (inspector diamond
/// toggled off). The engine rejects when nothing sits there.
pub(super) fn remove_param_keyframe_and_publish(
    engine: &mut Engine,
    clip: &str,
    param: ClipParam,
    at: RationalTime,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "remove-param-keyframe ignored: unparsable clip id");
        return;
    };
    match engine.apply(Command::Edit(EditCommand::RemoveParamKeyframe {
        clip: clip_id,
        param,
        at,
    })) {
        Ok(_) => {
            info!(%clip_id, ?param, tick = at.value, "removed param keyframe");
            publish_projection(engine, ui);
        }
        Err(e) => error!(%clip_id, ?param, "remove param keyframe failed: {e}"),
    }
}

/// Every animated property with a keyframe exactly at the clip-relative
/// `rel_tick`, with that keyframe's value and easing — the slice of one
/// merged timeline diamond (the timeline draws one diamond per tick across
/// all properties, CapCut-style).
pub(super) fn keyframes_at(
    transform: &AnimatedTransform,
    rel_tick: i64,
) -> Vec<(ClipParam, ParamValue, Easing)> {
    let mut hits = Vec::new();
    if let Some(kf) = transform
        .position
        .keyframes()
        .iter()
        .find(|k| k.tick == rel_tick)
    {
        hits.push((ClipParam::Position, ParamValue::Vec2(kf.value), kf.easing));
    }
    if let Some(kf) = transform
        .scale
        .keyframes()
        .iter()
        .find(|k| k.tick == rel_tick)
    {
        // Prefer Scalar when uniform so diamond retimes stay byte-compat with
        // older agent/UI paths; Vec2 when axes are split.
        let value = if kf.value.is_uniform() {
            ParamValue::Scalar(kf.value.x)
        } else {
            ParamValue::Vec2([kf.value.x, kf.value.y])
        };
        hits.push((ClipParam::Scale, value, kf.easing));
    }
    let scalars = [
        (ClipParam::Rotation, &transform.rotation),
        (ClipParam::Opacity, &transform.opacity),
    ];
    for (param, p) in scalars {
        if let Some(kf) = p.keyframes().iter().find(|k| k.tick == rel_tick) {
            hits.push((param, ParamValue::Scalar(kf.value), kf.easing));
        }
    }
    hits
}

/// Move every keyframe at `from_tick` to `to_tick` (timeline diamond drag,
/// keyframes roadmap Phase 2): per property a remove + re-set with the same
/// value and easing, all in one history group so a single undo puts the
/// diamond back. A keyframe already sitting at the destination on the same
/// property is replaced (the diamonds merge, like CapCut). The engine
/// re-validates that `to_tick` falls inside the clip.
pub(super) fn retime_keyframes_and_publish(
    engine: &mut Engine,
    clip: &str,
    from_tick: i64,
    to_tick: i64,
    tl_rate: Rational,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "retime-keyframes ignored: unparsable clip id");
        return;
    };
    if from_tick == to_tick {
        return;
    }
    let Some(model) = engine.project().clip(clip_id) else {
        error!(%clip_id, "retime-keyframes ignored: clip not on the timeline");
        return;
    };
    let moved = keyframes_at(&model.transform, from_tick - model.timeline.start.value);
    if moved.is_empty() {
        error!(%clip_id, from_tick, "retime-keyframes ignored: no keyframes at tick");
        return;
    }

    engine.begin_group();
    for (param, value, easing) in moved {
        if let Err(e) = engine.apply(Command::Edit(EditCommand::RemoveParamKeyframe {
            clip: clip_id,
            param,
            at: RationalTime::new(from_tick, tl_rate),
        })) {
            error!(%clip_id, ?param, "retime keyframes failed removing: {e}");
            engine.rollback_group();
            return;
        }
        if let Err(e) = engine.apply(Command::Edit(EditCommand::SetParamKeyframe {
            clip: clip_id,
            param,
            at: RationalTime::new(to_tick, tl_rate),
            value,
            easing,
            tangents: None,
        })) {
            error!(%clip_id, ?param, "retime keyframes failed setting: {e}");
            engine.rollback_group();
            return;
        }
    }
    engine.commit_group();
    info!(%clip_id, from_tick, to_tick, "retimed keyframes");
    publish_projection(engine, ui);
}

/// Remove every property's keyframe at `tick` (timeline diamond
/// right-click) as one history group — one undo restores the whole merged
/// diamond.
pub(super) fn remove_keyframes_at_and_publish(
    engine: &mut Engine,
    clip: &str,
    tick: i64,
    tl_rate: Rational,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "remove-keyframes ignored: unparsable clip id");
        return;
    };
    let Some(model) = engine.project().clip(clip_id) else {
        error!(%clip_id, "remove-keyframes ignored: clip not on the timeline");
        return;
    };
    let hits = keyframes_at(&model.transform, tick - model.timeline.start.value);
    if hits.is_empty() {
        error!(%clip_id, tick, "remove-keyframes ignored: no keyframes at tick");
        return;
    }

    engine.begin_group();
    for (param, _, _) in hits {
        if let Err(e) = engine.apply(Command::Edit(EditCommand::RemoveParamKeyframe {
            clip: clip_id,
            param,
            at: RationalTime::new(tick, tl_rate),
        })) {
            error!(%clip_id, ?param, "remove keyframes failed: {e}");
            engine.rollback_group();
            return;
        }
    }
    engine.commit_group();
    info!(%clip_id, tick, "removed keyframes at tick");
    publish_projection(engine, ui);
}

// PORT: main warmed the disk cache here with a `prefetch_ahead` read-ahead
// after idle frames; this branch's engine keeps decode read-ahead internal
// to its native decoders, so the worker sends nothing.
