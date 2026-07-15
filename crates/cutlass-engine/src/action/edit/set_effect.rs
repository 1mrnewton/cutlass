use cutlass_models::{ClipId, ModelError};

use crate::action::edit::restore_clip::RestoreClipAction;
use crate::action::{ApplyContext, EditAction};
use crate::error::EngineError;

// Effect-chain edits (M4). All four share the `set_crop` inverse shape: a
// full-clip snapshot restore, since the chain is small and a snapshot rolls
// back add / remove / reorder / param-edit unconditionally.

pub fn add_effect(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    effect_id: &str,
) -> Result<Box<dyn EditAction>, EngineError> {
    let before = snapshot(ctx, clip)?;
    ctx.project.add_effect(clip, effect_id)?;
    Ok(Box::new(RestoreClipAction { clip: before }))
}

pub fn remove_effect(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    index: usize,
) -> Result<Box<dyn EditAction>, EngineError> {
    let before = snapshot(ctx, clip)?;
    ctx.project.remove_effect(clip, index)?;
    Ok(Box::new(RestoreClipAction { clip: before }))
}

pub fn move_effect(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    from_index: usize,
    to_index: usize,
) -> Result<Box<dyn EditAction>, EngineError> {
    let before = snapshot(ctx, clip)?;
    ctx.project.move_effect(clip, from_index, to_index)?;
    Ok(Box::new(RestoreClipAction { clip: before }))
}

pub fn set_effect_param(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    index: usize,
    param: usize,
    value: f32,
) -> Result<Box<dyn EditAction>, EngineError> {
    let before = snapshot(ctx, clip)?;
    ctx.project.set_effect_param(clip, index, param, value)?;
    Ok(Box::new(RestoreClipAction { clip: before }))
}

fn snapshot(ctx: &ApplyContext<'_>, clip: ClipId) -> Result<cutlass_models::Clip, EngineError> {
    Ok(ctx
        .project
        .clip(clip)
        .cloned()
        .ok_or(ModelError::UnknownClip(clip))?)
}
