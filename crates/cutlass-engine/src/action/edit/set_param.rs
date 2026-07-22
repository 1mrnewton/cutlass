use cutlass_models::{
    ClipId, ClipParam, Easing, ModelError, ParamValue, RationalTime, SpatialTangents,
};

use crate::action::edit::restore_clip::RestoreClipAction;
use crate::action::{ApplyContext, EditAction};
use crate::error::EngineError;

// Parameter keyframe edits (M2). All three commands share the same inverse
// shape: a full-clip restore, like `SetTransformAction` — parameter state
// is tiny and the restore is unconditionally correct.

pub fn set_keyframe(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    param: ClipParam,
    at: RationalTime,
    value: ParamValue,
    easing: Easing,
    tangents: Option<SpatialTangents>,
) -> Result<Box<dyn EditAction>, EngineError> {
    let before = snapshot(ctx, clip)?;
    ctx.project
        .set_param_keyframe(clip, param, at, value, easing, tangents)?;
    Ok(Box::new(RestoreClipAction { clip: before }))
}

pub fn remove_keyframe(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    param: ClipParam,
    at: RationalTime,
) -> Result<Box<dyn EditAction>, EngineError> {
    let before = snapshot(ctx, clip)?;
    ctx.project.remove_param_keyframe(clip, param, at)?;
    Ok(Box::new(RestoreClipAction { clip: before }))
}

pub fn set_constant(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    param: ClipParam,
    value: ParamValue,
) -> Result<Box<dyn EditAction>, EngineError> {
    let before = snapshot(ctx, clip)?;
    ctx.project.set_param_constant(clip, param, value)?;
    Ok(Box::new(RestoreClipAction { clip: before }))
}

fn snapshot(ctx: &ApplyContext<'_>, clip: ClipId) -> Result<cutlass_models::Clip, EngineError> {
    Ok(ctx
        .project
        .clip(clip)
        .cloned()
        .ok_or(ModelError::UnknownClip(clip))?)
}
