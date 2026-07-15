use cutlass_models::{
    Clip, ClipId, ClipSource, MediaKind, ModelError, Project, RationalTime, TimeRange, TrackId,
    TrackKind, check_same_rate,
};

use crate::action::edit::insert_clip::InsertClipAction;
use crate::action::edit::link_clips::SetClipLinksAction;
use crate::action::edit::{shift_clips, split_clip};
use crate::action::{ApplyContext, CompoundAction, EditAction};
use crate::error::EngineError;

struct FreezePlan {
    source: ClipId,
    track: TrackId,
    at: RationalTime,
    duration: RationalTime,
    freeze: Clip,
    split_source: bool,
    clear_orphan_link: bool,
}

/// Insert a held video frame as one atomic structural edit.
///
/// The rejection-prone source/link/rate/edge policy is resolved before the
/// first mutation. The internal link-clear, optional split, ripple shift, and
/// insertion actions then contribute one compound inverse. If any action
/// fails, already-applied actions are rolled back immediately.
pub fn execute(
    ctx: &mut ApplyContext<'_>,
    clip: ClipId,
    at: RationalTime,
    duration: RationalTime,
) -> Result<(ClipId, Box<dyn EditAction>), EngineError> {
    let plan = build_plan(ctx.project, clip, at, duration)?;
    let freeze_id = plan.freeze.id;
    let mut inverses: Vec<Box<dyn EditAction>> = Vec::with_capacity(4);

    if plan.clear_orphan_link {
        apply_and_record(
            ctx,
            &mut inverses,
            Box::new(SetClipLinksAction {
                links: vec![(plan.source, None)],
            }),
        )?;
    }

    if plan.split_source {
        match split_clip::execute(ctx, plan.source, plan.at) {
            Ok((_tail, inverse)) => inverses.push(inverse),
            Err(error) => {
                rollback(ctx, &mut inverses, &error);
                return Err(error);
            }
        }
    }

    match shift_clips::execute(ctx, plan.track, plan.at, plan.duration) {
        Ok(inverse) => inverses.push(inverse),
        Err(error) => {
            rollback(ctx, &mut inverses, &error);
            return Err(error);
        }
    }

    apply_and_record(
        ctx,
        &mut inverses,
        Box::new(InsertClipAction {
            track: plan.track,
            clip: plan.freeze,
        }),
    )?;

    Ok((freeze_id, Box::new(CompoundAction { actions: inverses })))
}

fn build_plan(
    project: &Project,
    clip_id: ClipId,
    at: RationalTime,
    duration: RationalTime,
) -> Result<FreezePlan, ModelError> {
    let timeline = project.timeline();
    let timeline_rate = timeline.frame_rate;
    check_same_rate(at.rate, timeline_rate)?;
    check_same_rate(duration.rate, timeline_rate)?;
    if !timeline_rate.is_valid() || duration.value <= 0 {
        return Err(ModelError::InvalidRange);
    }

    let track_id = timeline
        .track_of(clip_id)
        .ok_or(ModelError::UnknownClip(clip_id))?;
    let track = timeline
        .track(track_id)
        .ok_or(ModelError::UnknownTrack(track_id))?;
    if track.kind != TrackKind::Video {
        return Err(ModelError::InvalidParam(format!(
            "freeze frame requires a clip on a video track (clip {clip_id} is on {track_id})"
        )));
    }
    if track.locked {
        return Err(ModelError::InvalidParam(format!(
            "source video track {track_id} is locked"
        )));
    }
    let ordered = track.clips_ordered();
    for candidate in &ordered {
        check_same_rate(candidate.timeline.start.rate, timeline_rate)?;
        check_same_rate(candidate.timeline.duration.rate, timeline_rate)?;
        if candidate.timeline.is_empty() {
            return Err(ModelError::InvalidRange);
        }
        candidate.timeline.end()?;
    }
    for pair in ordered.windows(2) {
        if pair[0].timeline.end()?.value > pair[1].timeline.start.value {
            return Err(ModelError::Overlap(track_id));
        }
    }

    let source = timeline
        .clip(clip_id)
        .cloned()
        .ok_or(ModelError::UnknownClip(clip_id))?;
    let ClipSource::Media { media, .. } = &source.content else {
        return Err(ModelError::InvalidParam(
            "freeze frame requires a media-backed clip".into(),
        ));
    };
    if source.freeze_frame {
        return Err(ModelError::InvalidParam(format!(
            "clip {clip_id} is already a freeze frame"
        )));
    }
    let media_source = project
        .media(*media)
        .ok_or(ModelError::UnknownMedia(*media))?;
    if media_source.kind() != MediaKind::Video {
        return Err(ModelError::InvalidParam(format!(
            "clip {clip_id} does not reference video media"
        )));
    }

    let source_end = source.timeline.end()?;
    if at.value < source.timeline.start.value || at.value > source_end.value {
        return Err(ModelError::InvalidRange);
    }
    let at_start = at.value == source.timeline.start.value;
    let at_end = at.value == source_end.value;
    if at_start && (source.animation_in.is_some() || source.animation_combo.is_some()) {
        return Err(ModelError::InvalidParam(
            "cannot freeze at a clip start with an active entrance/combo animation".into(),
        ));
    }
    if at_end && (source.animation_out.is_some() || source.animation_combo.is_some()) {
        return Err(ModelError::InvalidParam(
            "cannot freeze at a clip end with an active exit/combo animation".into(),
        ));
    }

    let clear_orphan_link = if let Some(link) = source.link {
        if let Some(partner) = timeline
            .tracks_ordered()
            .flat_map(|candidate_track| candidate_track.clips())
            .find(|candidate| candidate.id != clip_id && candidate.link == Some(link))
        {
            return Err(ModelError::InvalidParam(format!(
                "clip {clip_id} is linked to {}; unlink the group before freezing a frame",
                partner.id
            )));
        }
        true
    } else {
        false
    };
    for shifted in track
        .clips()
        .filter(|candidate| candidate.timeline.start.value >= at.value)
    {
        let Some(link) = shifted.link else {
            continue;
        };
        let group_moves_together = timeline
            .tracks_ordered()
            .flat_map(|candidate_track| candidate_track.clips())
            .filter(|candidate| candidate.link == Some(link))
            .all(|member| {
                timeline.track_of(member.id) == Some(track_id)
                    && member.timeline.start.value >= at.value
            });
        if !group_moves_together {
            return Err(ModelError::InvalidParam(format!(
                "freezing would desynchronize linked clip {}; unlink the group first",
                shifted.id
            )));
        }
    }

    let resolved_at = if at_end {
        RationalTime::new(
            at.value.checked_sub(1).ok_or(ModelError::TimeOverflow)?,
            timeline_rate,
        )
    } else {
        at
    };
    let source_time = source
        .source_time_at(resolved_at)?
        .ok_or(ModelError::InvalidRange)?;
    let freeze_timeline = TimeRange::new(at, duration)?;
    freeze_timeline.end()?;

    // Positive ripple shifts cannot collide, but checked arithmetic must be
    // validated up front because Project::shift_clips mutates clips in place.
    for candidate in track.clips() {
        if candidate.timeline.start.value >= at.value {
            candidate
                .timeline
                .start
                .value
                .checked_add(duration.value)
                .ok_or(ModelError::TimeOverflow)?;
            candidate
                .timeline
                .end()?
                .value
                .checked_add(duration.value)
                .ok_or(ModelError::TimeOverflow)?;
        }
    }
    let split_source = !at_start && !at_end;
    if split_source {
        source_end
            .value
            .checked_add(duration.value)
            .ok_or(ModelError::TimeOverflow)?;
    }

    let freeze = source.frozen_frame(source_time, freeze_timeline)?;
    Ok(FreezePlan {
        source: clip_id,
        track: track_id,
        at,
        duration,
        freeze,
        split_source,
        clear_orphan_link,
    })
}

fn apply_and_record(
    ctx: &mut ApplyContext<'_>,
    inverses: &mut Vec<Box<dyn EditAction>>,
    action: Box<dyn EditAction>,
) -> Result<(), EngineError> {
    match action.apply(ctx) {
        Ok(inverse) => {
            inverses.push(inverse);
            Ok(())
        }
        Err(error) => {
            rollback(ctx, inverses, &error);
            Err(error)
        }
    }
}

fn rollback(
    ctx: &mut ApplyContext<'_>,
    inverses: &mut Vec<Box<dyn EditAction>>,
    cause: &EngineError,
) {
    while let Some(inverse) = inverses.pop() {
        if let Err(rollback_error) = inverse.apply(ctx) {
            tracing::error!("freeze-frame rollback failed after {cause}: {rollback_error}");
            break;
        }
    }
}
