use super::*;

/// Append a catalog effect to a clip's chain (M4). One undoable entry; the
/// composite repaints because effects are visual.
pub(super) fn add_effect_and_publish(
    engine: &mut Engine,
    clip: &str,
    effect_id: &str,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "add-effect ignored: unparsable clip id");
        return;
    };
    if let Err(e) = engine.apply(Command::Edit(EditCommand::AddEffect {
        clip: clip_id,
        effect_id: effect_id.to_string(),
    })) {
        error!(%clip_id, effect_id, "add effect failed: {e}");
        return;
    }
    info!(%clip_id, effect_id, "added effect");
    publish_projection(engine, ui);
}

/// Remove the effect at `index` from a clip's chain (M4).
pub(super) fn remove_effect_and_publish(engine: &mut Engine, clip: &str, index: u32, ui: &UiSink) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "remove-effect ignored: unparsable clip id");
        return;
    };
    if let Err(e) = engine.apply(Command::Edit(EditCommand::RemoveEffect {
        clip: clip_id,
        index: index as usize,
    })) {
        error!(%clip_id, index, "remove effect failed: {e}");
        return;
    }
    info!(%clip_id, index, "removed effect");
    publish_projection(engine, ui);
}

/// Set one effect parameter to a constant (M4). The inspector addresses the
/// parameter by its catalog name; resolve it to the uniform slot index the
/// command expects from the clip's current effect. Scalars use
/// [`EditCommand::SetEffectParam`]; colors and vec2s use
/// [`EditCommand::SetParamConstant`] with [`ClipParam::Effect`].
pub(super) fn set_effect_param_and_publish(
    engine: &mut Engine,
    clip: &str,
    index: u32,
    param: &str,
    value: ParamValue,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-effect-param ignored: unparsable clip id");
        return;
    };
    let slot = engine
        .project()
        .clip(clip_id)
        .and_then(|c| c.effects.get(index as usize))
        .and_then(|fx| cutlass_models::effect_spec(&fx.effect_id))
        .and_then(|spec| spec.params.iter().position(|p| p.name == param));
    let Some(slot) = slot else {
        error!(%clip_id, index, param, "set-effect-param ignored: unknown param");
        return;
    };
    let edit = match value {
        ParamValue::Scalar(v) => EditCommand::SetEffectParam {
            clip: clip_id,
            index: index as usize,
            param: slot,
            value: v,
        },
        other => EditCommand::SetParamConstant {
            clip: clip_id,
            param: ClipParam::Effect {
                effect: index,
                param: slot as u32,
            },
            value: other,
        },
    };
    if let Err(e) = engine.apply(Command::Edit(edit)) {
        error!(%clip_id, index, param, ?value, "set effect param failed: {e}");
        return;
    }
    info!(%clip_id, index, param, ?value, "set effect param");
    publish_projection(engine, ui);
}

/// Add a catalog transition at the junction after `clip` (M4). Requires a
/// right-neighbor clip that abuts; the engine rejects otherwise.
pub(super) fn add_transition_and_publish(
    engine: &mut Engine,
    clip: &str,
    transition_id: &str,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "add-transition ignored: unparsable clip id");
        return;
    };
    if let Err(e) = engine.apply(Command::Edit(EditCommand::AddTransition {
        clip: clip_id,
        transition_id: transition_id.to_string(),
    })) {
        error!(%clip_id, transition_id, "add transition failed: {e}");
        return;
    }
    info!(%clip_id, transition_id, "added transition");
    publish_projection(engine, ui);
}

/// Remove the transition at `clip`'s right junction (M4).
pub(super) fn remove_transition_and_publish(engine: &mut Engine, clip: &str, ui: &UiSink) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "remove-transition ignored: unparsable clip id");
        return;
    };
    if let Err(e) = engine.apply(Command::Edit(EditCommand::RemoveTransition {
        clip: clip_id,
    })) {
        error!(%clip_id, "remove transition failed: {e}");
        return;
    }
    info!(%clip_id, "removed transition");
    publish_projection(engine, ui);
}

/// Set the window length (timeline ticks) of the transition after `clip` (M4).
pub(super) fn set_transition_and_publish(
    engine: &mut Engine,
    clip: &str,
    duration: i64,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-transition ignored: unparsable clip id");
        return;
    };
    if let Err(e) = engine.apply(Command::Edit(EditCommand::SetTransition {
        clip: clip_id,
        duration,
    })) {
        error!(%clip_id, duration, "set transition failed: {e}");
        return;
    }
    info!(%clip_id, duration, "set transition duration");
    publish_projection(engine, ui);
}
