//! Ruler-marker edits (M1 markers): add / remove / set, each with an
//! exact inverse so marker gestures oscillate under undo/redo like every
//! other timeline edit.

use cutlass_models::{Marker, MarkerColor, MarkerId, ModelError, RationalTime};

use crate::action::{ApplyContext, EditAction};
use crate::error::EngineError;

/// Add a marker at `at`. An omitted color cycles the fixed palette by the
/// current marker count, so consecutive unstyled markers stay distinct.
pub fn add(
    ctx: &mut ApplyContext<'_>,
    at: RationalTime,
    name: String,
    color: Option<MarkerColor>,
) -> Result<(MarkerId, Box<dyn EditAction>), EngineError> {
    let timeline = ctx.project.timeline_mut();
    let color = color.unwrap_or_else(|| MarkerColor::cycle(timeline.marker_count()));
    let id = timeline.add_marker(Marker::new(at, name, color))?;
    Ok((id, Box::new(RemoveMarkerAction { marker: id })))
}

/// Move / rename / recolor a marker; the inverse restores its prior state.
pub fn set(
    ctx: &mut ApplyContext<'_>,
    marker: MarkerId,
    at: RationalTime,
    name: String,
    color: MarkerColor,
) -> Result<Box<dyn EditAction>, EngineError> {
    let timeline = ctx.project.timeline_mut();
    let before = timeline
        .marker(marker)
        .cloned()
        .ok_or(ModelError::UnknownMarker(marker))?;
    timeline.set_marker(marker, at, name, color)?;
    Ok(Box::new(ReplaceMarkerAction { before }))
}

/// Remove a marker; applying returns the restore (same id) as the inverse.
pub struct RemoveMarkerAction {
    pub marker: MarkerId,
}

impl EditAction for RemoveMarkerAction {
    fn apply(self: Box<Self>, ctx: &mut ApplyContext<'_>) -> Result<Box<dyn EditAction>, EngineError> {
        let removed = ctx
            .project
            .timeline_mut()
            .remove_marker(self.marker)
            .ok_or(ModelError::UnknownMarker(self.marker))?;
        Ok(Box::new(RestoreMarkerAction { marker: removed }))
    }
}

/// Re-insert a removed marker snapshot (undo of remove).
struct RestoreMarkerAction {
    marker: Marker,
}

impl EditAction for RestoreMarkerAction {
    fn apply(self: Box<Self>, ctx: &mut ApplyContext<'_>) -> Result<Box<dyn EditAction>, EngineError> {
        let id = ctx.project.timeline_mut().add_marker(self.marker)?;
        Ok(Box::new(RemoveMarkerAction { marker: id }))
    }
}

/// Swap a marker back to a captured snapshot (set-marker undo/redo).
struct ReplaceMarkerAction {
    before: Marker,
}

impl EditAction for ReplaceMarkerAction {
    fn apply(self: Box<Self>, ctx: &mut ApplyContext<'_>) -> Result<Box<dyn EditAction>, EngineError> {
        let id = self.before.id;
        let timeline = ctx.project.timeline_mut();
        let current = timeline
            .marker(id)
            .cloned()
            .ok_or(ModelError::UnknownMarker(id))?;
        timeline.set_marker(id, self.before.tick, self.before.name, self.before.color)?;
        Ok(Box::new(ReplaceMarkerAction { before: current }))
    }
}
