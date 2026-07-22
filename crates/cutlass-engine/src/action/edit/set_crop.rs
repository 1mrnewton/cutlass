use cutlass_models::{ClipId, CropRect, ModelError, RationalTime};

use crate::action::edit::restore_clip::RestoreClipAction;
use crate::action::{ApplyContext, EditAction};
use crate::error::EngineError;

/// Set a clip's framing (CapCut crop + flips, M1). The model validates the
/// rect and the visual-track requirement. The inverse is a full-clip
/// restore — crop and both flips roll back in one shot, like the speed and
/// transform edits.
///
/// `at: Some(playhead)` keyframes an already-animated crop (M2 compose).
pub fn set_crop(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    crop: CropRect,
    flip_h: bool,
    flip_v: bool,
    at: Option<RationalTime>,
) -> Result<Box<dyn EditAction>, EngineError> {
    let before = ctx
        .project
        .clip(clip)
        .cloned()
        .ok_or(ModelError::UnknownClip(clip))?;
    ctx.project.set_clip_crop(clip, crop, flip_h, flip_v, at)?;
    Ok(Box::new(RestoreClipAction { clip: before }))
}
