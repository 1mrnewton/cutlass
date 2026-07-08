//! Reorder a track in the stack as one undoable unit.

use cutlass_models::{ModelError, TrackId};

use crate::action::{ApplyContext, EditAction};
use crate::error::EngineError;

pub fn execute(
    ctx: &mut ApplyContext<'_>,
    track: TrackId,
    index: usize,
) -> Result<Box<dyn EditAction>, EngineError> {
    let timeline = ctx.project.timeline();
    let from_index = timeline
        .order()
        .iter()
        .position(|&id| id == track)
        .ok_or(ModelError::UnknownTrack(track))?;
    ctx.project.timeline_mut().move_track(track, index)?;
    Ok(Box::new(MoveTrackAction {
        track_id: track,
        from_index,
        to_index: index,
    }))
}

pub struct MoveTrackAction {
    pub track_id: TrackId,
    pub from_index: usize,
    pub to_index: usize,
}

impl EditAction for MoveTrackAction {
    fn apply(
        self: Box<Self>,
        ctx: &mut ApplyContext<'_>,
    ) -> Result<Box<dyn EditAction>, EngineError> {
        ctx.project
            .timeline_mut()
            .move_track(self.track_id, self.from_index)?;
        Ok(Box::new(MoveTrackAction {
            track_id: self.track_id,
            from_index: self.to_index,
            to_index: self.from_index,
        }))
    }
}
