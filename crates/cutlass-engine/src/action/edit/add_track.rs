use cutlass_models::{ModelError, Track, TrackId, TrackKind};

use crate::action::{ApplyContext, EditAction};
use crate::error::EngineError;

pub struct RemoveTrackAction {
    pub track_id: TrackId,
}

pub struct InsertTrackAction {
    pub track: Track,
    pub order_index: usize,
}

pub fn execute(
    ctx: &mut ApplyContext<'_>,
    kind: TrackKind,
    name: impl Into<String>,
) -> Result<(TrackId, Box<dyn EditAction>), EngineError> {
    let id = ctx.project.add_track(kind, name);
    Ok((id, Box::new(RemoveTrackAction { track_id: id })))
}

impl EditAction for RemoveTrackAction {
    fn apply(self: Box<Self>, ctx: &mut ApplyContext<'_>) -> Result<Box<dyn EditAction>, EngineError> {
        let order_index = ctx
            .project
            .timeline()
            .order()
            .iter()
            .position(|&id| id == self.track_id)
            .ok_or(ModelError::UnknownTrack(self.track_id))?;
        let track = ctx
            .project
            .timeline_mut()
            .remove_track(self.track_id)
            .ok_or(ModelError::UnknownTrack(self.track_id))?;
        Ok(Box::new(InsertTrackAction {
            track,
            order_index,
        }))
    }
}

impl EditAction for InsertTrackAction {
    fn apply(self: Box<Self>, ctx: &mut ApplyContext<'_>) -> Result<Box<dyn EditAction>, EngineError> {
        let id = self.track.id;
        ctx.project
            .timeline_mut()
            .restore_track(self.track, self.order_index)?;
        Ok(Box::new(RemoveTrackAction { track_id: id }))
    }
}
