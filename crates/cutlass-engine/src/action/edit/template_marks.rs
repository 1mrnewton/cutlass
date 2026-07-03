//! Template authoring markers: replaceable slots and editable texts.
//!
//! Both are render-inert metadata the [`Template`](cutlass_models::Template)
//! surface scans; the model validates eagerly at mark time so authoring
//! mistakes surface here, not at template-apply time.

use cutlass_models::{ClipId, ModelError, Replaceable};

use crate::action::edit::restore_clip::RestoreClipAction;
use crate::action::{ApplyContext, EditAction};
use crate::error::EngineError;

/// Mark (or unmark) a media clip as a user-replaceable template slot.
pub fn set_replaceable(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    replaceable: Option<Replaceable>,
) -> Result<Box<dyn EditAction>, EngineError> {
    let before = ctx
        .project
        .clip(clip)
        .cloned()
        .ok_or(ModelError::UnknownClip(clip))?;
    ctx.project.set_replaceable(clip, replaceable)?;
    Ok(Box::new(RestoreClipAction { clip: before }))
}

/// Mark a text clip's wording as user-editable in template use.
pub fn set_text_editable(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    editable: bool,
) -> Result<Box<dyn EditAction>, EngineError> {
    let before = ctx
        .project
        .clip(clip)
        .cloned()
        .ok_or(ModelError::UnknownClip(clip))?;
    ctx.project.set_text_editable(clip, editable)?;
    Ok(Box::new(RestoreClipAction { clip: before }))
}
