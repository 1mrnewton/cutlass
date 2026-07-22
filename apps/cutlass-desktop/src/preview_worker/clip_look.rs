use super::*;

/// Set the project canvas settings (M1): aspect preset + background color
/// in one undoable history entry. An out-of-range preset index falls back
/// to auto (defensive — the dialog's list is index-aligned with the model).
pub(super) fn set_canvas_and_publish(
    engine: &mut Engine,
    aspect_index: i32,
    background: [u8; 3],
    ui: &UiSink,
) {
    let aspect = usize::try_from(aspect_index)
        .ok()
        .and_then(|i| cutlass_models::CanvasAspect::ALL.get(i).copied())
        .unwrap_or_default();
    match engine.apply(Command::Edit(EditCommand::SetCanvas { aspect, background })) {
        Ok(_) => {
            info!(aspect = aspect.name(), ?background, "set canvas settings");
            publish_projection(engine, ui);
        }
        Err(e) => error!("set canvas failed: {e}"),
    }
}

/// Set a visual clip's crop window + mirroring (CapCut crop, M1). One
/// undoable history entry; the engine validates the rect and rejects
/// audio-lane clips, so a failure here just logs (the inspector only shows
/// crop controls for visual clips — a rejection is a stale-projection race).
pub(super) fn set_clip_crop_and_publish(
    engine: &mut Engine,
    clip: &str,
    crop: CropRect,
    flip_h: bool,
    flip_v: bool,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-clip-crop ignored: unparsable clip id");
        return;
    };
    if let Err(e) = engine.apply(Command::Edit(EditCommand::SetClipCrop {
        clip: clip_id,
        crop,
        flip_h,
        flip_v,
    })) {
        error!(%clip_id, "set clip crop failed: {e}");
        return;
    }
    info!(
        %clip_id,
        x = crop.x, y = crop.y, w = crop.w, h = crop.h, flip_h, flip_v,
        "set clip crop"
    );
    publish_projection(engine, ui);
}

/// Set or clear a visual clip's filter preset. A live look drag may have left
/// an override in place; clear it first so the commit becomes authoritative.
pub(super) fn set_clip_filter_and_publish(
    engine: &mut Engine,
    clip: &str,
    filter_id: &str,
    intensity: f32,
    ui: &UiSink,
) {
    engine.set_look_override(None);
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-clip-filter ignored: unparsable clip id");
        return;
    };
    let filter = filter_from_ui(filter_id, intensity);
    if let Err(e) = engine.apply(Command::Edit(EditCommand::SetClipFilter {
        clip: clip_id,
        filter: filter.clone(),
    })) {
        error!(%clip_id, filter_id, intensity, "set clip filter failed: {e}");
        return;
    }
    info!(%clip_id, ?filter, "set clip filter");
    publish_projection(engine, ui);
}

/// Set or clear a visual clip's `.cube` LUT (empty path clears). Intensity
/// blends the looked-up color over the original in the LUT pass itself.
pub(super) fn set_clip_lut_and_publish(
    engine: &mut Engine,
    clip: &str,
    path: &str,
    intensity: f32,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-clip-lut ignored: unparsable clip id");
        return;
    };
    let lut = (!path.is_empty()).then(|| Lut {
        path: path.to_string(),
        intensity: intensity.clamp(0.0, 1.0),
    });
    if let Err(e) = engine.apply(Command::Edit(EditCommand::SetClipLut {
        clip: clip_id,
        lut: lut.clone(),
    })) {
        error!(%clip_id, path, intensity, "set clip LUT failed: {e}");
        return;
    }
    info!(%clip_id, ?lut, "set clip LUT");
    publish_projection(engine, ui);
}

/// Set all manual color adjustments on a visual clip in one undoable edit.
/// Release commits clear the live look override first, mirroring generator
/// and transform preview semantics.
pub(super) fn set_clip_adjust_and_publish(
    engine: &mut Engine,
    clip: &str,
    adjust: ColorAdjustments,
    ui: &UiSink,
) {
    engine.set_look_override(None);
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-clip-adjust ignored: unparsable clip id");
        return;
    };
    let adjust = sanitize_adjustments(adjust);
    if let Err(e) = engine.apply(Command::Edit(EditCommand::SetClipAdjustments {
        clip: clip_id,
        adjust,
    })) {
        error!(%clip_id, ?adjust, "set clip adjustments failed: {e}");
        return;
    }
    info!(%clip_id, ?adjust, "set clip adjustments");
    publish_projection(engine, ui);
}

pub(super) fn set_clip_animation_and_publish(
    engine: &mut Engine,
    clip: &str,
    slot: &str,
    animation_id: &str,
    speed: f32,
    intensity: f32,
    stagger: f32,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-clip-animation ignored: unparsable clip id");
        return;
    };
    let Some(animation_slot) = parse_animation_slot(slot) else {
        error!(slot, "set-clip-animation ignored: unknown slot");
        return;
    };
    let animation = if animation_id.is_empty() {
        None
    } else {
        Some(cutlass_models::AnimationRef {
            id: animation_id.to_string(),
            speed,
            intensity,
            stagger,
        })
    };
    if let Err(e) = engine.apply(Command::Edit(EditCommand::SetClipAnimation {
        clip: clip_id,
        slot: animation_slot,
        animation,
    })) {
        error!(%clip_id, slot, animation_id, "set clip animation failed: {e}");
        return;
    }
    info!(%clip_id, slot, animation_id, speed, intensity, stagger, "set clip animation");
    publish_projection(engine, ui);
}

pub(super) fn parse_animation_slot(slot: &str) -> Option<cutlass_models::AnimationSlot> {
    match slot {
        "in" => Some(cutlass_models::AnimationSlot::In),
        "out" => Some(cutlass_models::AnimationSlot::Out),
        "combo" => Some(cutlass_models::AnimationSlot::Combo),
        _ => None,
    }
}
