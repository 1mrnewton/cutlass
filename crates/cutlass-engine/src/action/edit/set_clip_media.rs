use cutlass_models::{ClipId, MediaId, ModelError, TimeRange};

use crate::action::edit::restore_clip::RestoreClipAction;
use crate::action::{ApplyContext, EditAction};
use crate::error::EngineError;

/// Replace a clip's content with a trimmed window of pooled media (template
/// music swap, slot in-point re-pick). The model validates the media, the
/// source window, and the track's content kind. The inverse is a full-clip
/// restore, so the previous content (media or windowing) rolls back in one
/// shot.
pub fn execute(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    media: MediaId,
    source: TimeRange,
) -> Result<Box<dyn EditAction>, EngineError> {
    let before = ctx
        .project
        .clip(clip)
        .cloned()
        .ok_or(ModelError::UnknownClip(clip))?;
    ctx.project.set_clip_media(clip, media, source)?;
    Ok(Box::new(RestoreClipAction { clip: before }))
}
