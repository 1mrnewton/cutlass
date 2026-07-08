//! Track rename: swap a lane's display name as one undoable unit.

use cutlass_models::{ModelError, TrackId};

use crate::action::{ApplyContext, EditAction};
use crate::error::EngineError;

pub fn set_track_name(
    ctx: &mut ApplyContext<'_>,
    track: TrackId,
    name: String,
) -> Result<Box<dyn EditAction>, EngineError> {
    let track_mut = ctx
        .project
        .timeline_mut()
        .track_mut(track)
        .ok_or(ModelError::UnknownTrack(track))?;
    let before = std::mem::replace(&mut track_mut.name, name);
    Ok(Box::new(RestoreTrackNameAction {
        track_id: track,
        name: before,
    }))
}

struct RestoreTrackNameAction {
    track_id: TrackId,
    name: String,
}

impl EditAction for RestoreTrackNameAction {
    fn apply(
        self: Box<Self>,
        ctx: &mut ApplyContext<'_>,
    ) -> Result<Box<dyn EditAction>, EngineError> {
        let track_mut = ctx
            .project
            .timeline_mut()
            .track_mut(self.track_id)
            .ok_or(ModelError::UnknownTrack(self.track_id))?;
        let current = std::mem::replace(&mut track_mut.name, self.name);
        Ok(Box::new(RestoreTrackNameAction {
            track_id: self.track_id,
            name: current,
        }))
    }
}
