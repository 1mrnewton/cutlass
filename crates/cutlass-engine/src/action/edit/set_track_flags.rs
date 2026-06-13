use cutlass_models::{ModelError, TrackId};

use crate::action::{ApplyContext, EditAction};
use crate::error::EngineError;

/// The undoable unit for a track's on/off flags (enabled / muted / locked /
/// duck source).
///
/// Each `SetTrack*` command changes exactly one flag; the action snapshots all
/// of the *previous* values so its inverse is a plain swap that oscillates
/// like trim's `RestoreClip`. Only the supplied flag(s) are overwritten —
/// `None` leaves a flag as it is.
pub struct SetTrackFlagsAction {
    pub track: TrackId,
    pub enabled: bool,
    pub muted: bool,
    pub locked: bool,
    pub duck_source: bool,
}

pub fn execute(
    ctx: &mut ApplyContext<'_>,
    track: TrackId,
    enabled: Option<bool>,
    muted: Option<bool>,
    locked: Option<bool>,
    duck_source: Option<bool>,
) -> Result<Box<dyn EditAction>, EngineError> {
    let t = ctx
        .project
        .timeline_mut()
        .track_mut(track)
        .ok_or(ModelError::UnknownTrack(track))?;
    // Snapshot the pre-edit flags into the inverse (a full restore).
    let inverse = Box::new(SetTrackFlagsAction {
        track,
        enabled: t.enabled,
        muted: t.muted,
        locked: t.locked,
        duck_source: t.duck_source,
    });
    if let Some(v) = enabled {
        t.enabled = v;
    }
    if let Some(v) = muted {
        t.muted = v;
    }
    if let Some(v) = locked {
        t.locked = v;
    }
    if let Some(v) = duck_source {
        t.duck_source = v;
    }
    Ok(inverse)
}

impl EditAction for SetTrackFlagsAction {
    fn apply(self: Box<Self>, ctx: &mut ApplyContext<'_>) -> Result<Box<dyn EditAction>, EngineError> {
        execute(
            ctx,
            self.track,
            Some(self.enabled),
            Some(self.muted),
            Some(self.locked),
            Some(self.duck_source),
        )
    }
}
