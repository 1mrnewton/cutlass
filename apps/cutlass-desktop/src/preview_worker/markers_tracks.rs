use super::*;

/// Drop a ruler marker (M1). Empty `color` cycles the palette; one undoable
/// history entry.
pub(super) fn add_marker_and_publish(
    engine: &mut Engine,
    at_tick: i64,
    name: &str,
    color: &str,
    tl_rate: Rational,
    ui: &UiSink,
) {
    let at = RationalTime::new(at_tick.max(0), tl_rate);
    let color = parse_marker_color(color);
    match engine.apply(Command::Edit(EditCommand::AddMarker {
        at,
        name: name.to_string(),
        color,
    })) {
        Ok(ApplyOutcome::Edited(EditOutcome::CreatedMarker(id))) => {
            info!(%id, at_tick, "added timeline marker");
            publish_projection(engine, ui);
        }
        Ok(other) => error!(at_tick, "unexpected add-marker outcome: {other:?}"),
        Err(e) => error!(at_tick, "add marker failed: {e}"),
    }
}

/// Remove a ruler marker by raw id (M1). One undoable history entry.
pub(super) fn remove_marker_and_publish(engine: &mut Engine, marker: &str, ui: &UiSink) {
    let Some(marker_id) = parse_raw_id(marker).map(MarkerId::from_raw) else {
        error!(marker, "remove-marker ignored: unparsable marker id");
        return;
    };
    match engine.apply(Command::Edit(EditCommand::RemoveMarker {
        marker: marker_id,
    })) {
        Ok(_) => {
            info!(%marker_id, "removed timeline marker");
            publish_projection(engine, ui);
        }
        Err(e) => error!(%marker_id, "remove marker failed: {e}"),
    }
}

/// Move / rename / recolor a ruler marker (M1). One undoable history entry.
pub(super) fn set_marker_and_publish(
    engine: &mut Engine,
    marker: &str,
    at_tick: i64,
    name: &str,
    color: &str,
    tl_rate: Rational,
    ui: &UiSink,
) {
    let Some(marker_id) = parse_raw_id(marker).map(MarkerId::from_raw) else {
        error!(marker, "set-marker ignored: unparsable marker id");
        return;
    };
    let at = RationalTime::new(at_tick.max(0), tl_rate);
    let color = parse_marker_color(color)
        .or_else(|| {
            engine
                .project()
                .timeline()
                .marker(marker_id)
                .map(|m| m.color)
        })
        .unwrap_or(MarkerColor::Teal);
    match engine.apply(Command::Edit(EditCommand::SetMarker {
        marker: marker_id,
        at,
        name: name.to_string(),
        color,
    })) {
        Ok(_) => {
            info!(%marker_id, at_tick, "updated timeline marker");
            publish_projection(engine, ui);
        }
        Err(e) => error!(%marker_id, "set marker failed: {e}"),
    }
}

pub(super) fn remove_track_manual_and_publish(engine: &mut Engine, track: &str, ui: &UiSink) {
    let Some(track_id) = parse_raw_id(track).map(TrackId::from_raw) else {
        error!(track, "remove-track ignored: unparsable track id");
        return;
    };
    // CapCut never deletes the main track (the UI hides the menu item; this
    // guards races where the projection lagged a main-lane promotion).
    if Some(track_id) == main_video_track(engine) {
        info!(%track_id, "remove-track ignored: main track is permanent");
        return;
    }
    match engine.apply(Command::Edit(EditCommand::RemoveTrack { track: track_id })) {
        Ok(_) => {
            info!(%track_id, "removed track");
            publish_projection(engine, ui);
        }
        Err(e) => error!(%track_id, "remove track failed: {e}"),
    }
}

pub(super) fn move_track_manual_and_publish(
    engine: &mut Engine,
    track: &str,
    index: usize,
    ui: &UiSink,
) {
    let Some(track_id) = parse_raw_id(track).map(TrackId::from_raw) else {
        error!(track, "move-track ignored: unparsable track id");
        return;
    };
    match engine.apply(Command::Edit(EditCommand::MoveTrack {
        track: track_id,
        index,
    })) {
        Ok(_) => {
            info!(%track_id, index, "moved track");
            publish_projection(engine, ui);
        }
        Err(e) => error!(%track_id, index, "move track failed: {e}"),
    }
}

pub(super) fn set_track_name_and_publish(
    engine: &mut Engine,
    track: &str,
    name: &str,
    ui: &UiSink,
) {
    let Some(track_id) = parse_raw_id(track).map(TrackId::from_raw) else {
        error!(track, "set-track-name ignored: unparsable track id");
        return;
    };
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return;
    }
    match engine.apply(Command::Edit(EditCommand::SetTrackName {
        track: track_id,
        name: trimmed.to_string(),
    })) {
        Ok(_) => {
            info!(%track_id, name = trimmed, "renamed track");
            publish_projection(engine, ui);
        }
        Err(e) => error!(%track_id, "rename track failed: {e}"),
    }
}

pub(super) fn parse_marker_color(name: &str) -> Option<MarkerColor> {
    MarkerColor::parse_token(name)
}
