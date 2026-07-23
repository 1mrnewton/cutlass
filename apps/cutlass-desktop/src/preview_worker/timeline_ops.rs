use super::*;

/// Build a shape generator with new reference-pixel dimensions, preserving the
/// clip's shape kind and fill. `None` when the clip is missing or not a shape.
///
/// Dimensions are floored at 1px and non-finite input is rejected: the slider
/// stays in `8..=1920`, but a typed entry or double-click reset can deliver
/// anything, and a zero/negative extent would collapse the raster's `Rect` to
/// an invisible shape.
pub(super) fn shape_size_from_engine(
    engine: &Engine,
    clip: &str,
    width: f32,
    height: f32,
) -> Option<Generator> {
    if !width.is_finite() || !height.is_finite() {
        return None;
    }
    let clip_id = parse_raw_id(clip).map(ClipId::from_raw)?;
    let generator = match &engine.project().timeline().clip(clip_id)?.content {
        ClipSource::Generated(g) => g,
        ClipSource::Media { .. } => return None,
    };
    match generator {
        // A slider commit sets an absolute size, so the animated params
        // collapse to constants (matching the pre-keyframe behavior); corner
        // rounding and stroke ride along untouched.
        Generator::Shape {
            shape,
            rgba,
            corner_radius,
            stroke,
            ..
        } => Some(Generator::Shape {
            shape: shape.clone(),
            rgba: rgba.clone(),
            width: Param::Constant(width.max(1.0)),
            height: Param::Constant(height.max(1.0)),
            corner_radius: corner_radius.clone(),
            stroke: stroke.clone(),
        }),
        _ => None,
    }
}

/// Rebuild a solid or shape generator with a new fill color, preserving every
/// other field. `None` when the clip is missing or not a solid/shape.
///
/// A color commit collapses an animated shape fill to a constant (same as
/// width/height sliders); solids are a bare `[u8; 4]`.
pub(super) fn generator_fill_from_engine(
    engine: &Engine,
    clip: &str,
    rgba: [u8; 4],
) -> Option<Generator> {
    let clip_id = parse_raw_id(clip).map(ClipId::from_raw)?;
    let generator = match &engine.project().timeline().clip(clip_id)?.content {
        ClipSource::Generated(g) => g,
        ClipSource::Media { .. } => return None,
    };
    match generator {
        Generator::SolidColor { .. } => Some(Generator::SolidColor { rgba }),
        Generator::Shape {
            shape,
            width,
            height,
            corner_radius,
            stroke,
            ..
        } => Some(Generator::Shape {
            shape: shape.clone(),
            rgba: Param::Constant(rgba),
            width: width.clone(),
            height: height.clone(),
            corner_radius: corner_radius.clone(),
            stroke: stroke.clone(),
        }),
        _ => None,
    }
}

/// Replace a generated clip's content (inspector title edit). One history
/// entry per committed edit; the engine rejects non-generated clips.
pub(super) fn set_generator_and_publish(
    engine: &mut Engine,
    clip: &str,
    generator: Generator,
    ui: &UiSink,
) {
    // A live font-size drag may have left an override in place; the commit is
    // the authoritative value, so clear it (the next render is identical — no
    // flicker between drag end and commit, mirroring `SetTransform`).
    engine.set_generator_override(None);
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "set-generator ignored: unparsable clip id");
        return;
    };
    match engine.apply(Command::Edit(EditCommand::SetGenerator {
        clip: clip_id,
        generator,
    })) {
        Ok(_) => {
            info!(%clip_id, "updated generated clip content");
            publish_projection(engine, ui);
        }
        Err(e) => error!(%clip_id, "set generator failed: {e}"),
    }
}

/// Whether `[start, start + duration)` overlaps no clip on `track`.
pub(super) fn span_free(track: &Track, start: i64, duration: i64) -> bool {
    let end = start + duration;
    track
        .clips_ordered()
        .iter()
        .all(|c| c.timeline.end_tick() <= start || c.timeline.start.value >= end)
}

/// `track` (raw id from the Slint projection) when it names an existing lane
/// of `kind`.
pub(super) fn lane_of_kind(engine: &Engine, track: &str, kind: TrackKind) -> Option<TrackId> {
    let id = TrackId::from_raw(parse_raw_id(track)?);
    engine
        .project()
        .timeline()
        .track(id)
        .is_some_and(|t| t.kind == kind)
        .then_some(id)
}

/// The main lane while it's still empty — CapCut lands the first video
/// dropped *anywhere* on the main track rather than spawning an overlay lane.
pub(super) fn empty_main_lane(engine: &Engine, kind: TrackKind) -> Option<TrackId> {
    if kind != TrackKind::Video {
        return None;
    }
    let timeline = engine.project().timeline();
    let main = timeline.main_track()?;
    timeline
        .track(main)
        .is_some_and(cutlass_models::Track::is_empty)
        .then_some(main)
}

/// Move a dragged clip to its resolved landing spot: an existing lane
/// (`track` set) or a new lane of the clip's kind inserted at `insert_row`.
/// A cross-lane move that empties its source lane removes that lane
/// (CapCut deletes overlay tracks that empty out). With `insert` (main-track
/// magnet) the landing is an insertion on the main lane; with the magnet on,
/// a move *off* the main lane also closes the gap it leaves. Every variant
/// is one history group, so one undo reverts the whole gesture.
#[allow(clippy::too_many_arguments)]
pub(super) fn move_clip_and_publish(
    engine: &mut Engine,
    clip: &str,
    track: &str,
    insert_row: i64,
    start_tick: i64,
    insert: bool,
    main_magnet: bool,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "move ignored: unparsable clip id");
        return;
    };
    let Some(source_track) = engine.project().timeline().track_of(clip_id) else {
        error!(%clip_id, "move ignored: clip not on the timeline");
        return;
    };
    let kind = engine
        .project()
        .timeline()
        .track(source_track)
        .expect("track_of returned an existing track")
        .kind;
    let placed = engine
        .project()
        .clip(clip_id)
        .expect("track_of returned a placed clip")
        .timeline;
    let tl_rate = engine.project().timeline().frame_rate;
    // Decided before the gesture mutates anything: a new lane created below
    // the stack would become the bottom video lane and steal main status.
    let source_is_main = main_magnet && Some(source_track) == main_video_track(engine);

    if insert {
        // Main-track magnet: the resolver targets the existing main lane.
        let Some(to_track) = parse_raw_id(track).map(TrackId::from_raw) else {
            error!(%clip_id, track, "insert-move ignored: unparsable track id");
            return;
        };
        engine.begin_group();
        let result = if to_track == source_track {
            ripple_reorder(engine, clip_id, to_track, start_tick.max(0))
        } else {
            ripple_move_in(engine, clip_id, source_track, to_track, start_tick.max(0))
        };
        match result {
            Ok(()) => {
                engine.commit_group();
                info!(%clip_id, %to_track, start_tick, "ripple-inserted moved clip");
            }
            Err(e) => {
                error!(%clip_id, %to_track, start_tick, "insert move failed: {e}");
                engine.rollback_group();
            }
        }
        publish_projection(engine, ui);
        return;
    }

    // One history entry per move, including a created destination lane and a
    // removed emptied source lane.
    engine.begin_group();
    let to_track = match parse_raw_id(track).map(TrackId::from_raw) {
        Some(id) => id,
        None => match create_track(engine, kind, insert_row) {
            Ok(id) => id,
            Err(e) => {
                error!(%clip_id, "move failed creating {kind:?} track: {e}");
                engine.rollback_group();
                return;
            }
        },
    };

    match engine.apply(Command::Edit(EditCommand::MoveClip {
        clip: clip_id,
        to_track,
        start: RationalTime::new(start_tick.max(0), tl_rate),
    })) {
        Ok(ApplyOutcome::Edited(EditOutcome::Updated(_))) => {
            let mut completed = true;
            if source_track != to_track {
                // Leaving the main lane with the magnet on closes the gap
                // the clip vacated (CapCut ripple). Can't collide: the first
                // shifted clip lands exactly where the moved clip started.
                if source_is_main {
                    completed = apply_edit(
                        engine,
                        EditCommand::ShiftClips {
                            track: source_track,
                            from: placed.start,
                            delta: RationalTime::new(-placed.duration.value, tl_rate),
                        },
                    )
                    .map_err(|e| error!(%clip_id, "move failed closing main-lane gap: {e}"))
                    .is_ok();
                }
                if completed {
                    remove_track_if_empty(engine, source_track);
                }
            }
            if completed {
                engine.commit_group();
                info!(%clip_id, %to_track, start_tick, "moved clip");
            } else {
                engine.rollback_group();
            }
            publish_projection(engine, ui);
        }
        Ok(other) => {
            error!(%clip_id, "unexpected move-clip outcome: {other:?}");
            engine.rollback_group();
            publish_projection(engine, ui);
        }
        // The drag resolver previewed a valid spot; the engine still rejects
        // atomically if the projection raced a concurrent edit. Rolling back
        // removes a lane this move just created.
        Err(e) => {
            error!(%clip_id, %to_track, start_tick, "move clip failed: {e}");
            engine.rollback_group();
            publish_projection(engine, ui);
        }
    }
}

/// Reorder within the main lane as one group of four commands: park the clip
/// past the lane's content end (never rendered — the projection publishes
/// only after the group resolves), close its old gap, open the new hole at
/// `at` (post-close space, straight from the drag resolver), and land in it.
pub(super) fn ripple_reorder(
    engine: &mut Engine,
    clip_id: ClipId,
    track: TrackId,
    at: i64,
) -> Result<(), String> {
    let tl_rate = engine.project().timeline().frame_rate;
    let placed = engine
        .project()
        .clip(clip_id)
        .ok_or("clip not on the timeline")?
        .timeline;
    let duration = placed.duration.value;
    let park = engine
        .project()
        .timeline()
        .track(track)
        .ok_or("main lane missing")?
        .content_end();

    apply_edit(
        engine,
        EditCommand::MoveClip {
            clip: clip_id,
            to_track: track,
            start: RationalTime::new(park, tl_rate),
        },
    )?;
    // Both shifts also carry the parked clip along (its start stays past the
    // rest of the lane), so it never collides with the clips in between.
    apply_edit(
        engine,
        EditCommand::ShiftClips {
            track,
            from: placed.start,
            delta: RationalTime::new(-duration, tl_rate),
        },
    )?;
    apply_edit(
        engine,
        EditCommand::ShiftClips {
            track,
            from: RationalTime::new(at, tl_rate),
            delta: RationalTime::new(duration, tl_rate),
        },
    )?;
    apply_edit(
        engine,
        EditCommand::MoveClip {
            clip: clip_id,
            to_track: track,
            start: RationalTime::new(at, tl_rate),
        },
    )
}

/// Cross-lane move onto the main lane: open the hole at `at`, move the clip
/// in, and drop the source lane when this emptied it (same overlay policy as
/// freeform moves).
pub(super) fn ripple_move_in(
    engine: &mut Engine,
    clip_id: ClipId,
    source_track: TrackId,
    to_track: TrackId,
    at: i64,
) -> Result<(), String> {
    let tl_rate = engine.project().timeline().frame_rate;
    let duration = engine
        .project()
        .clip(clip_id)
        .ok_or("clip not on the timeline")?
        .timeline
        .duration
        .value;

    apply_edit(
        engine,
        EditCommand::ShiftClips {
            track: to_track,
            from: RationalTime::new(at, tl_rate),
            delta: RationalTime::new(duration, tl_rate),
        },
    )?;
    apply_edit(
        engine,
        EditCommand::MoveClip {
            clip: clip_id,
            to_track,
            start: RationalTime::new(at, tl_rate),
        },
    )?;
    remove_track_if_empty(engine, source_track);
    Ok(())
}

/// Re-place a trimmed clip at its resolved extent. The trim resolver already
/// clamped to neighbors and source headroom, so this should always apply; the
/// engine still validates atomically (overlap, source bounds) and we surface
/// any rejection rather than mutating the projection optimistically.
///
/// With `linkage` on, the same edge delta applies to every clip in the
/// trimmed clip's link group (the resolver intersected the clamps, so the
/// partners' extents are valid too) — one history entry for the group.
///
/// With the main-track magnet on and the trim touching the main lane, the
/// trim *ripples* instead of leaving/eating a gap: downstream clips follow
/// the dragged edge (timeline roadmap Phase 7's deliberate gap). See
/// [`commit_trims`]; still one history entry — a single undo restores the
/// trim and every shifted clip.
pub(super) fn trim_clip_and_publish(
    engine: &mut Engine,
    clip: &str,
    start_tick: i64,
    duration_ticks: i64,
    linkage: bool,
    main_magnet: bool,
    ui: &UiSink,
) {
    let Some(clip_id) = parse_raw_id(clip).map(ClipId::from_raw) else {
        error!(clip, "trim ignored: unparsable clip id");
        return;
    };
    let Some(placed) = engine.project().clip(clip_id).map(|c| c.timeline) else {
        error!(%clip_id, "trim ignored: clip not on the timeline");
        return;
    };
    let tl_rate = engine.project().timeline().frame_rate;
    let start = start_tick.max(0);
    let duration = duration_ticks.max(1);
    // The same edge motion, expressed as deltas the partners can replay.
    let delta_start = start - placed.start.value;
    let delta_duration = duration - placed.duration.value;

    let mut trims = vec![(clip_id, TimeRange::at_rate(start, duration, tl_rate))];
    if linkage {
        for partner in link_group_ids(engine, clip_id) {
            if partner == clip_id {
                continue;
            }
            let Some(extent) = engine.project().clip(partner).map(|c| c.timeline) else {
                continue;
            };
            trims.push((
                partner,
                TimeRange::at_rate(
                    extent.start.value + delta_start,
                    (extent.duration.value + delta_duration).max(1),
                    tl_rate,
                ),
            ));
        }
    }

    match commit_trims(engine, &trims, main_magnet) {
        Ok(ripple) => info!(%clip_id, start_tick, duration_ticks, linkage, ripple, "trimmed clip"),
        Err(e) => error!(%clip_id, "trim clip failed: {e}"),
    }
    publish_projection(engine, ui);
}

/// Apply a resolved set of member trims as one history group.
///
/// With the main-track magnet on and any member sitting on the main lane
/// (dragging the audio half of a linked pair must still keep the main lane
/// gapless), every member's trim ripples on its own lane — linked pairs and
/// their downstream neighbors all shift by the same duration delta, so
/// cross-lane alignment survives. Otherwise members get plain `TrimClip`s.
///
/// A rejected step rolls the whole group back — no half-applied ripple.
/// Returns whether the group rippled.
pub(super) fn commit_trims(
    engine: &mut Engine,
    trims: &[(ClipId, TimeRange)],
    main_magnet: bool,
) -> Result<bool, String> {
    let main = main_video_track(engine);
    let ripple = main_magnet
        && main.is_some_and(|m| {
            trims
                .iter()
                .any(|&(id, _)| engine.project().timeline().track_of(id) == Some(m))
        });

    engine.begin_group();
    for &(id, timeline) in trims {
        let result = if ripple {
            apply_ripple_trim(engine, id, timeline)
        } else {
            apply_edit(engine, EditCommand::TrimClip { clip: id, timeline })
        };
        if let Err(e) = result {
            engine.rollback_group();
            return Err(format!("clip {id}: {e}"));
        }
    }
    engine.commit_group();
    Ok(ripple)
}

/// One member's ripple trim: `TrimClip` + `ShiftClips` composed on the
/// member's own lane, ordered so the engine's atomic validation accepts the
/// intermediate state (open room before growing into it, trim before
/// closing the gap behind a shrink).
///
/// Semantics (CapCut): the trimmed clip stays anchored at its old start and
/// every downstream clip shifts by the duration delta — the lane neither
/// leaves nor eats a gap.
/// - Trailing edge: only the duration changes; downstream (clips starting at
///   or after the old end) shifts by the delta.
/// - Leading edge: the resolved extent moves the start — that start delta is
///   what the engine derives the new source in-point from — and the shift
///   then re-anchors the clip at its old start, carrying downstream along.
///   A leading grow shifts everything from the old start right first, then
///   trims anchored there, which yields the same negative start delta.
///
/// The caller wraps members in one history group and rolls back on error,
/// so a rejected step never leaves a half-applied ripple.
pub(super) fn apply_ripple_trim(
    engine: &mut Engine,
    clip: ClipId,
    timeline: TimeRange,
) -> Result<(), String> {
    let Some(old) = engine.project().clip(clip).map(|c| c.timeline) else {
        return Err("clip is not on the timeline".into());
    };
    let Some(track) = engine.project().timeline().track_of(clip) else {
        return Err("clip has no track".into());
    };
    let tl_rate = engine.project().timeline().frame_rate;
    let delta_dur = timeline.duration.value - old.duration.value;
    let trim = EditCommand::TrimClip { clip, timeline };

    if timeline.start.value != old.start.value {
        // Leading edge (the resolver anchors the end, so the start moved).
        if delta_dur > 0 {
            // Grow: open room first (the clip and everything after it move
            // right), then trim anchored at the old start.
            apply_edit(
                engine,
                EditCommand::ShiftClips {
                    track,
                    from: old.start,
                    delta: RationalTime::new(delta_dur, tl_rate),
                },
            )?;
            apply_edit(
                engine,
                EditCommand::TrimClip {
                    clip,
                    timeline: TimeRange::at_rate(old.start.value, timeline.duration.value, tl_rate),
                },
            )
        } else {
            // Shrink: trim to the resolved extent (a gap opens at the old
            // start), then slide the clip and downstream left into it.
            apply_edit(engine, trim)?;
            apply_edit(
                engine,
                EditCommand::ShiftClips {
                    track,
                    from: timeline.start,
                    delta: RationalTime::new(old.start.value - timeline.start.value, tl_rate),
                },
            )
        }
    } else if delta_dur > 0 {
        // Trailing grow: push downstream right, then extend into the hole.
        apply_edit(
            engine,
            EditCommand::ShiftClips {
                track,
                from: RationalTime::new(old.end_tick(), tl_rate),
                delta: RationalTime::new(delta_dur, tl_rate),
            },
        )?;
        apply_edit(engine, trim)
    } else if delta_dur < 0 {
        // Trailing shrink: pull the edge in, then close the gap behind it.
        apply_edit(engine, trim)?;
        apply_edit(
            engine,
            EditCommand::ShiftClips {
                track,
                from: RationalTime::new(old.end_tick(), tl_rate),
                delta: RationalTime::new(delta_dur, tl_rate),
            },
        )
    } else {
        // No edge moved (defensive — the UI skips noop trims).
        apply_edit(engine, trim)
    }
}

/// Every clip sharing `clip`'s link group (including itself); just the clip
/// when it's unlinked. O(total clips) — cold per-gesture path.
pub(super) fn link_group_ids(engine: &Engine, clip: ClipId) -> Vec<ClipId> {
    let Some(link) = engine.project().clip(clip).and_then(|c| c.link) else {
        return vec![clip];
    };
    engine
        .project()
        .timeline()
        .tracks_ordered()
        .flat_map(|t| t.clips_ordered())
        .filter(|c| c.link == Some(link))
        .map(|c| c.id)
        .collect()
}

/// Toggle a track header flag (hide/mute/lock). Undoable like any edit; the
/// republished projection carries the new flag to the lane header. Disabling
/// a visual track drops it from the composite (the engine skips `!enabled`
/// visual tracks), so the preview catches up on the next scrub.
pub(super) fn set_track_flag_and_publish(
    engine: &mut Engine,
    track: &str,
    flag: TrackFlag,
    value: bool,
    ui: &UiSink,
) {
    let Some(track_id) = parse_raw_id(track).map(TrackId::from_raw) else {
        error!(track, "set-track-flag ignored: unparsable track id");
        return;
    };
    let command = match flag {
        TrackFlag::Enabled => EditCommand::SetTrackEnabled {
            track: track_id,
            enabled: value,
        },
        TrackFlag::Muted => EditCommand::SetTrackMuted {
            track: track_id,
            muted: value,
        },
        TrackFlag::Locked => EditCommand::SetTrackLocked {
            track: track_id,
            locked: value,
        },
        TrackFlag::DuckSource => EditCommand::SetTrackDuckSource {
            track: track_id,
            duck_source: value,
        },
    };
    match engine.apply(Command::Edit(command)) {
        Ok(ApplyOutcome::Edited(EditOutcome::UpdatedTrack(_))) => {
            info!(%track_id, value, "set track flag");
            publish_projection(engine, ui);
        }
        Ok(other) => error!(%track_id, "unexpected set-track-flag outcome: {other:?}"),
        Err(e) => error!(%track_id, "set track flag failed: {e}"),
    }
}
