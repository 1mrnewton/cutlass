//! Phase I look edits: mask, chroma key, stabilization, filter, adjustments,
//! animations, and the audio role tag.
//!
//! All seven follow the pitch/denoise pattern — the model validates and
//! stores, no duration changes, so there is no neighbor check and the inverse
//! is a full-clip restore. The properties are render-neutral this milestone
//! (persisted + surfaced in `ui_state`, drawn later).

use cutlass_models::{
    AnimationRef, AnimationSlot, AudioRole, BlendMode, ChromaKey, Clip, ClipId, ColorAdjustments,
    Filter, LayerStyles, Lut, Mask, ModelError, MotionBlur, Project, StabilizeLevel,
};

use crate::action::edit::restore_clip::RestoreClipAction;
use crate::action::{ApplyContext, EditAction};
use crate::error::EngineError;

/// Snapshot `clip`, run `mutate` against the project, and hand back the
/// restore inverse — the shared shape of every look edit.
fn with_restore(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    mutate: impl FnOnce(&mut Project) -> Result<(), ModelError>,
) -> Result<Box<dyn EditAction>, EngineError> {
    let before: Clip = ctx
        .project
        .clip(clip)
        .cloned()
        .ok_or(ModelError::UnknownClip(clip))?;
    mutate(ctx.project)?;
    Ok(Box::new(RestoreClipAction { clip: before }))
}

pub fn set_mask(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    mask: Option<Mask>,
) -> Result<Box<dyn EditAction>, EngineError> {
    with_restore(ctx, clip, |p| p.set_clip_mask(clip, mask))
}

pub fn set_blend_mode(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    mode: BlendMode,
) -> Result<Box<dyn EditAction>, EngineError> {
    with_restore(ctx, clip, |p| p.set_blend_mode(clip, mode))
}

pub fn set_motion_blur(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    motion_blur: MotionBlur,
) -> Result<Box<dyn EditAction>, EngineError> {
    with_restore(ctx, clip, |p| p.set_motion_blur(clip, motion_blur))
}

pub fn set_layer_styles(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    styles: LayerStyles,
) -> Result<Box<dyn EditAction>, EngineError> {
    with_restore(ctx, clip, |p| p.set_layer_styles(clip, styles))
}

pub fn set_chroma(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    chroma: Option<ChromaKey>,
) -> Result<Box<dyn EditAction>, EngineError> {
    with_restore(ctx, clip, |p| p.set_clip_chroma_key(clip, chroma))
}

pub fn set_stabilize(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    stabilize: Option<StabilizeLevel>,
) -> Result<Box<dyn EditAction>, EngineError> {
    with_restore(ctx, clip, |p| p.set_clip_stabilize(clip, stabilize))
}

pub fn set_filter(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    filter: Option<Filter>,
) -> Result<Box<dyn EditAction>, EngineError> {
    with_restore(ctx, clip, |p| p.set_clip_filter(clip, filter))
}

pub fn set_lut(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    lut: Option<Lut>,
) -> Result<Box<dyn EditAction>, EngineError> {
    with_restore(ctx, clip, |p| p.set_clip_lut(clip, lut))
}

pub fn set_adjustments(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    adjust: ColorAdjustments,
) -> Result<Box<dyn EditAction>, EngineError> {
    with_restore(ctx, clip, |p| p.set_clip_adjustments(clip, adjust))
}

pub fn set_animation(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    slot: AnimationSlot,
    animation: Option<AnimationRef>,
) -> Result<Box<dyn EditAction>, EngineError> {
    with_restore(ctx, clip, |p| p.set_clip_animation(clip, slot, animation))
}

pub fn set_audio_role(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    role: Option<AudioRole>,
) -> Result<Box<dyn EditAction>, EngineError> {
    with_restore(ctx, clip, |p| p.set_clip_audio_role(clip, role))
}
