use cutlass_models::{Clip, ClipId, ModelError, RationalTime, TrackId};

use crate::action::{ApplyContext, EditAction};
use crate::error::EngineError;

/// Undo of a split: drop the tail and restore the original (pre-split) left
/// clip. The produced inverse redoes the split **without allocating a new id**
/// (see [`ReapplySplitAction`]) so deeper redo entries that reference the tail
/// id keep resolving.
pub struct MergeSplitAction {
    pub left_id: ClipId,
    /// The original clip as it was before the split, restored on undo.
    pub restored: Clip,
    pub right_id: ClipId,
}

/// Redo of a split: reproduce the post-split state verbatim by shrinking the
/// left half back down and re-inserting the captured tail under its original
/// id. Oscillates with [`MergeSplitAction`].
pub struct ReapplySplitAction {
    pub left_id: ClipId,
    /// The left half as it looked immediately after the split.
    pub left_after: Clip,
    /// The tail clip (carries its own `right_id`), re-inserted verbatim.
    pub right: Clip,
    /// The track the tail lived on when the split happened.
    pub track: TrackId,
    pub right_id: ClipId,
}

pub fn execute(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    at: RationalTime,
) -> Result<(ClipId, Box<dyn EditAction>), EngineError> {
    let restored = ctx
        .project
        .clip(clip)
        .cloned()
        .ok_or(ModelError::UnknownClip(clip))?;
    let right_id = ctx.project.split_clip(clip, at)?;
    Ok((
        right_id,
        Box::new(MergeSplitAction {
            left_id: clip,
            restored,
            right_id,
        }),
    ))
}

impl EditAction for MergeSplitAction {
    fn apply(
        self: Box<Self>,
        ctx: &mut ApplyContext<'_>,
    ) -> Result<Box<dyn EditAction>, EngineError> {
        // Snapshot the post-split state before undoing so redo can restore it
        // byte-for-byte (crucially, keeping the tail's id stable).
        let track = ctx
            .project
            .timeline()
            .track_of(self.right_id)
            .ok_or(ModelError::UnknownClip(self.right_id))?;
        let right = ctx
            .project
            .remove_clip(self.right_id)
            .ok_or(ModelError::UnknownClip(self.right_id))?;
        let left_after = ctx
            .project
            .clip(self.left_id)
            .cloned()
            .ok_or(ModelError::UnknownClip(self.left_id))?;
        // Tail is gone, so restoring the left half to its full pre-split range
        // can't overlap anything.
        *ctx.project
            .timeline_mut()
            .clip_mut(self.left_id)
            .ok_or(ModelError::UnknownClip(self.left_id))? = self.restored;
        Ok(Box::new(ReapplySplitAction {
            left_id: self.left_id,
            left_after,
            right,
            track,
            right_id: self.right_id,
        }))
    }
}

impl EditAction for ReapplySplitAction {
    fn apply(
        self: Box<Self>,
        ctx: &mut ApplyContext<'_>,
    ) -> Result<Box<dyn EditAction>, EngineError> {
        let restored = ctx
            .project
            .clip(self.left_id)
            .cloned()
            .ok_or(ModelError::UnknownClip(self.left_id))?;
        // Shrink the left half first: it currently spans the full pre-split
        // range, which would overlap the tail we're about to re-insert.
        *ctx.project
            .timeline_mut()
            .clip_mut(self.left_id)
            .ok_or(ModelError::UnknownClip(self.left_id))? = self.left_after;
        ctx.project
            .timeline_mut()
            .add_clip(self.track, self.right)?;
        Ok(Box::new(MergeSplitAction {
            left_id: self.left_id,
            restored,
            right_id: self.right_id,
        }))
    }
}
