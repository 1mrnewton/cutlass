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
    index: Option<usize>,
    pinned: bool,
) -> Result<(TrackId, Box<dyn EditAction>), EngineError> {
    let mut track = Track::new(kind, name);
    track.pinned = pinned;
    let timeline = ctx.project.timeline_mut();
    let id = match index {
        Some(order_index) => timeline.insert_track(track, order_index),
        None => timeline.add_track(track),
    };
    Ok((id, Box::new(RemoveTrackAction { track_id: id })))
}

impl EditAction for RemoveTrackAction {
    fn apply(
        self: Box<Self>,
        ctx: &mut ApplyContext<'_>,
    ) -> Result<Box<dyn EditAction>, EngineError> {
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
        Ok(Box::new(InsertTrackAction { track, order_index }))
    }
}

impl EditAction for InsertTrackAction {
    fn apply(
        self: Box<Self>,
        ctx: &mut ApplyContext<'_>,
    ) -> Result<Box<dyn EditAction>, EngineError> {
        let id = self.track.id;
        ctx.project
            .timeline_mut()
            .restore_track(self.track, self.order_index)?;
        Ok(Box::new(RemoveTrackAction { track_id: id }))
    }
}
