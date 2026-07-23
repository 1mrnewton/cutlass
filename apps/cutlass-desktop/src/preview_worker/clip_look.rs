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
///
/// When crop is already keyframed, `at` (the playhead) writes a keyframe
/// instead of flattening — same compose semantics as
/// [`super::overrides::set_transform_and_publish`].
pub(super) fn set_clip_crop_and_publish(
    engine: &mut Engine,
    clip: &str,
    crop: CropRect,
    flip_h: bool,
    flip_v: bool,
    at: RationalTime,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-clip-crop ignored: unparsable clip id");
        return;
    };
    // Clear a live crop override before the commit so the next frame never
    // flashes the stale drag value (same order as set_param_constant).
    clear_param_override(engine, clip, ClipParam::Crop, Some(&ui.audio));
    let wrote_keyframe = engine
        .project()
        .clip(clip_id)
        .is_some_and(|c| c.crop.is_animated());
    if let Err(e) = engine.apply(Command::Edit(EditCommand::SetClipCrop {
        clip: clip_id,
        crop,
        flip_h,
        flip_v,
        at: Some(at),
    })) {
        error!(%clip_id, "set clip crop failed: {e}");
        return;
    }
    info!(
        %clip_id,
        x = crop.x, y = crop.y, w = crop.w, h = crop.h, flip_h, flip_v,
        "set clip crop"
    );
    if wrote_keyframe {
        bump_keyframe_commit_epoch(ui);
    }
    publish_projection(engine, ui);
}

/// Set a visual clip's blend mode (CapCut "Blend"). Unknown mode ids are
/// ignored (the inspector only offers catalog entries).
pub(super) fn set_blend_mode_and_publish(engine: &mut Engine, clip: &str, mode: &str, ui: &UiSink) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-blend-mode ignored: unparsable clip id");
        return;
    };
    let Some(mode) = BlendMode::from_id(mode) else {
        error!(clip, mode, "set-blend-mode ignored: unknown mode");
        return;
    };
    if let Err(e) = engine.apply(Command::Edit(EditCommand::SetClipBlendMode {
        clip: clip_id,
        mode,
    })) {
        error!(%clip_id, ?mode, "set clip blend mode failed: {e}");
        return;
    }
    info!(%clip_id, ?mode, "set clip blend mode");
    publish_projection(engine, ui);
}

/// Set per-clip transform motion blur. Validation lives in the model
/// setter; failures just log (stale-projection race).
pub(super) fn set_motion_blur_and_publish(
    engine: &mut Engine,
    clip: &str,
    motion_blur: MotionBlur,
    ui: &UiSink,
) {
    // Clear a live shutter/samples override before the commit so the next
    // frame never flashes the stale drag value.
    engine.set_motion_blur_override(None);
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-motion-blur ignored: unparsable clip id");
        return;
    };
    let enabled = motion_blur.enabled;
    let shutter = motion_blur.shutter_deg;
    let samples = motion_blur.samples;
    if let Err(e) = engine.apply(Command::Edit(EditCommand::SetClipMotionBlur {
        clip: clip_id,
        motion_blur,
    })) {
        error!(%clip_id, "set clip motion blur failed: {e}");
        return;
    }
    info!(%clip_id, enabled, shutter, samples, "set clip motion blur");
    publish_projection(engine, ui);
}

/// Replace a visual clip's layer styles (CapCut shadow/glow/outline/background).
/// A live styles drag may have left an override in place; clear it first so
/// the commit becomes authoritative (mirrors look filter/adjust commits).
pub(super) fn set_layer_styles_and_publish(
    engine: &mut Engine,
    clip: &str,
    styles: LayerStyles,
    ui: &UiSink,
) {
    engine.set_styles_override(None);
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-layer-styles ignored: unparsable clip id");
        return;
    };
    if let Err(e) = engine.apply(Command::Edit(EditCommand::SetClipLayerStyles {
        clip: clip_id,
        styles: styles.clone(),
    })) {
        error!(%clip_id, "set clip layer styles failed: {e}");
        return;
    }
    info!(%clip_id, empty = styles.is_empty(), "set clip layer styles");
    publish_projection(engine, ui);
}

/// Set or clear a visual clip's shaped alpha mask (CapCut mask panel).
pub(super) fn set_mask_and_publish(
    engine: &mut Engine,
    clip: &str,
    mask: Option<Mask>,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-mask ignored: unparsable clip id");
        return;
    };
    let kind = mask.as_ref().map(|m| m.kind.id());
    if let Err(e) = engine.apply(Command::Edit(EditCommand::SetClipMask {
        clip: clip_id,
        mask,
    })) {
        error!(%clip_id, "set clip mask failed: {e}");
        return;
    }
    info!(%clip_id, ?kind, "set clip mask");
    publish_projection(engine, ui);
}

/// Switch mask kind (or clear with empty `kind`), preserving feather / invert /
/// geometry from the clip's committed mask. Reads engine state — no UI snapshot.
pub(super) fn set_mask_kind_and_publish(engine: &mut Engine, clip: &str, kind: &str, ui: &UiSink) {
    let Some(mask) = mask_with_kind(engine, clip, kind) else {
        return;
    };
    set_mask_and_publish(engine, clip, mask, ui);
}

/// Build the mask for a kind switch against the clip's committed state.
/// `None` return means the edit was dropped (bad id / unknown kind).
pub(super) fn mask_with_kind(engine: &Engine, clip: &str, kind: &str) -> Option<Option<Mask>> {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-mask-kind ignored: unparsable clip id");
        return None;
    };
    let Some(clip_ref) = engine.project().clip(clip_id) else {
        error!(clip, "set-mask-kind ignored: unknown clip");
        return None;
    };
    if kind.is_empty() {
        return Some(None);
    }
    let Some(spec) = cutlass_models::mask_catalog()
        .iter()
        .find(|s| s.kind.id() == kind)
    else {
        error!(kind, "set-mask-kind ignored: unknown kind");
        return None;
    };
    // Preserve feather / invert / geometry when switching kind;
    // enable-from-none uses Mask::new defaults.
    let mut mask = clip_ref
        .mask
        .clone()
        .unwrap_or_else(|| Mask::new(spec.kind));
    mask.kind = spec.kind;
    // Switching onto Mirror with the historical full-layer size[0]=1 yields
    // a no-op band — seed CapCut-parity half-width thickness instead.
    if spec.kind == MaskKind::Mirror
        && let Some([w, h]) = mask.size.constant()
        && (w - 1.0).abs() < f32::EPSILON
    {
        mask.size = Param::Constant([0.5, h]);
    }
    Some(Some(mask))
}

/// Toggle invert on the clip's existing mask. No-op (logged) when mask is off.
pub(super) fn set_mask_invert_and_publish(
    engine: &mut Engine,
    clip: &str,
    invert: bool,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-mask-invert ignored: unparsable clip id");
        return;
    };
    let Some(clip_ref) = engine.project().clip(clip_id) else {
        error!(clip, "set-mask-invert ignored: unknown clip");
        return;
    };
    let Some(mut mask) = clip_ref.mask.clone() else {
        error!(clip, "set-mask-invert ignored: clip has no mask");
        return;
    };
    mask.invert = invert;
    set_mask_and_publish(engine, clip, Some(mask), ui);
}

/// Set or clear chroma keying (CapCut green screen).
pub(super) fn set_chroma_and_publish(
    engine: &mut Engine,
    clip: &str,
    chroma: Option<ChromaKey>,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-chroma ignored: unparsable clip id");
        return;
    };
    // Drop a live chroma-color override so enable/disable never flashes it.
    engine.set_chroma_color_override(None);
    let enabled = chroma.is_some();
    if let Err(e) = engine.apply(Command::Edit(EditCommand::SetClipChroma {
        clip: clip_id,
        chroma,
    })) {
        error!(%clip_id, "set clip chroma failed: {e}");
        return;
    }
    info!(%clip_id, enabled, "set clip chroma");
    publish_projection(engine, ui);
}

/// Set chroma RGB on a clip that already has chroma enabled.
pub(super) fn set_chroma_color_and_publish(
    engine: &mut Engine,
    clip: &str,
    rgb: [u8; 3],
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-chroma-color ignored: unparsable clip id");
        return;
    };
    let Some(clip_ref) = engine.project().clip(clip_id) else {
        error!(clip, "set-chroma-color ignored: unknown clip");
        return;
    };
    let Some(mut chroma) = clip_ref.chroma_key.clone() else {
        error!(clip, "set-chroma-color ignored: chroma off");
        return;
    };
    // Clear a live chroma-color override before the commit so the next frame
    // never flashes the stale drag value.
    engine.set_chroma_color_override(None);
    chroma.rgb = rgb;
    set_chroma_and_publish(engine, clip, Some(chroma), ui);
}

/// Install a session-only chroma RGB override for live color-well preview.
pub(super) fn apply_chroma_color_override(engine: &mut Engine, clip: &str, rgb: [u8; 3]) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "chroma-color preview ignored: unparsable clip id");
        return;
    };
    if engine
        .project()
        .clip(clip_id)
        .and_then(|c| c.chroma_key.as_ref())
        .is_none()
    {
        error!(clip, "chroma-color preview ignored: chroma off");
        return;
    }
    engine.set_chroma_color_override(Some((clip_id, rgb)));
}

/// Enable/disable one layer-style block, merging against committed styles.
pub(super) fn toggle_layer_style_and_publish(
    engine: &mut Engine,
    clip: &str,
    block: &str,
    enabled: bool,
    ui: &UiSink,
) {
    let Some(styles) = styles_with_toggled_block(engine, clip, block, enabled) else {
        return;
    };
    set_layer_styles_and_publish(engine, clip, styles, ui);
}

/// Merge a style-block toggle against the clip's committed styles.
pub(super) fn styles_with_toggled_block(
    engine: &Engine,
    clip: &str,
    block: &str,
    enabled: bool,
) -> Option<LayerStyles> {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "toggle-layer-style ignored: unparsable clip id");
        return None;
    };
    let Some(clip_ref) = engine.project().clip(clip_id) else {
        error!(clip, "toggle-layer-style ignored: unknown clip");
        return None;
    };
    let mut styles = clip_ref.styles.clone();
    match block {
        "shadow" => {
            styles.shadow = enabled.then(cutlass_models::LayerShadow::default);
        }
        "glow" => {
            styles.glow = enabled.then(cutlass_models::LayerGlow::default);
        }
        "outline" => {
            styles.outline = enabled.then(cutlass_models::LayerOutline::default);
        }
        "background" => {
            styles.background = enabled.then(cutlass_models::LayerBackground::default);
        }
        other => {
            error!(block = other, "toggle-layer-style ignored: unknown block");
            return None;
        }
    }
    Some(styles)
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
    // Intensity slider drags leave a LookParam::LutIntensity override; clear
    // it before the structural SetClipLut commit.
    clear_param_override(
        engine,
        clip,
        ClipParam::Look {
            param: cutlass_models::LookParam::LutIntensity,
        },
        Some(&ui.audio),
    );
    let lut = (!path.is_empty()).then(|| Lut {
        path: path.to_string(),
        intensity: intensity.clamp(0.0, 1.0).into(),
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
    let adjust = sanitize_adjustments(&adjust);
    if let Err(e) = engine.apply(Command::Edit(EditCommand::SetClipAdjustments {
        clip: clip_id,
        adjust: adjust.clone(),
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
    animation: Option<cutlass_models::AnimationRef>,
    ui: &UiSink,
) {
    // Clear a live speed/intensity/stagger override before the commit.
    engine.set_animation_override(None);
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-clip-animation ignored: unparsable clip id");
        return;
    };
    let Some(animation_slot) = parse_animation_slot(slot) else {
        error!(slot, "set-clip-animation ignored: unknown slot");
        return;
    };
    if let Err(e) = engine.apply(Command::Edit(EditCommand::SetClipAnimation {
        clip: clip_id,
        slot: animation_slot,
        animation: animation.clone(),
    })) {
        error!(%clip_id, slot, ?animation, "set clip animation failed: {e}");
        return;
    }
    info!(%clip_id, slot, ?animation, "set clip animation");
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
