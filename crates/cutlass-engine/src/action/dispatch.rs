use cutlass_commands::{Command, EditCommand, EditOutcome, ProjectCommand};

use super::edit::add_track::RemoveTrackAction;
use super::edit::remove_media::RemoveMediaAction;
use super::edit::restore_transitions::RestoreTransitionsAction;
use super::edit::{self, remove_clip::RemoveClipAction};
use super::project::{self, import};
use super::{ApplyContext, CompoundAction, EditAction};
use crate::error::EngineError;
use cutlass_models::{TrackId, Transition};

/// Capture every track's transitions before a structural edit, but only when
/// some exist — the common case (no transitions) pays nothing.
fn transitions_guard(ctx: &ApplyContext<'_>) -> Option<Vec<(TrackId, Vec<Transition>)>> {
    ctx.project
        .has_transitions()
        .then(|| ctx.project.transitions_snapshot())
}

/// After a structural edit, prune junctions whose abutment broke. If anything
/// was pruned, fold a transitions-restore into the inverse so undo brings the
/// dropped junctions back; otherwise the primary inverse stands alone.
fn finalize_structural(
    ctx: &mut ApplyContext<'_>,
    guard: Option<Vec<(TrackId, Vec<Transition>)>>,
    primary: Box<dyn EditAction>,
) -> Box<dyn EditAction> {
    let Some(snapshot) = guard else {
        return primary;
    };
    if ctx.project.prune_dead_transitions() {
        Box::new(CompoundAction {
            actions: vec![primary, Box::new(RestoreTransitionsAction { snapshot })],
        })
    } else {
        primary
    }
}

/// Result of applying a wire [`Command`] through the engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApplyOutcome {
    Imported { media: cutlass_models::MediaId },
    RemovedMedia { media: cutlass_models::MediaId },
    Saved,
    Opened,
    Loaded,
    Relinked { media: cutlass_models::MediaId },
    Exported { stats: cutlass_encoder::ExportStats },
    Edited(EditOutcome),
}

pub fn dispatch(
    command: Command,
    ctx: &mut ApplyContext<'_>,
) -> Result<(ApplyOutcome, Option<Box<dyn EditAction>>), EngineError> {
    match command {
        Command::Project(project) => dispatch_project(project, ctx),
        Command::Edit(edit) => dispatch_edit(edit, ctx),
    }
}

fn dispatch_project(
    command: ProjectCommand,
    ctx: &mut ApplyContext<'_>,
) -> Result<(ApplyOutcome, Option<Box<dyn EditAction>>), EngineError> {
    match command {
        ProjectCommand::Import { path } => {
            let (media, inverse) = import::execute(ctx, &path)?;
            Ok((ApplyOutcome::Imported { media }, inverse))
        }
        ProjectCommand::RemoveMedia { media } => {
            let inverse = Box::new(RemoveMediaAction { media }).apply(ctx)?;
            Ok((ApplyOutcome::RemovedMedia { media }, Some(inverse)))
        }
        ProjectCommand::Save { path } => {
            project::save::execute(ctx, path)?;
            Ok((ApplyOutcome::Saved, None))
        }
        ProjectCommand::Open { path } => {
            project::open::execute(ctx, path)?;
            Ok((ApplyOutcome::Opened, None))
        }
        ProjectCommand::Load { path } => {
            project::load::execute(ctx, path)?;
            Ok((ApplyOutcome::Loaded, None))
        }
        ProjectCommand::RelinkMedia { media, path } => {
            project::relink::execute(ctx, media, &path)?;
            Ok((ApplyOutcome::Relinked { media }, None))
        }
        ProjectCommand::Export { .. } => Err(EngineError::Export(
            "export is handled by Engine::apply, not dispatch".into(),
        )),
    }
}

fn dispatch_edit(
    edit: EditCommand,
    ctx: &mut ApplyContext<'_>,
) -> Result<(ApplyOutcome, Option<Box<dyn EditAction>>), EngineError> {
    match edit {
        EditCommand::AddTrack { kind, name, index } => {
            let (id, inverse) = edit::add_track::execute(ctx, kind, name, index)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::CreatedTrack(id)),
                Some(inverse),
            ))
        }
        EditCommand::AddClip {
            track,
            media,
            source,
            start,
        } => {
            let (id, inverse) = edit::add_clip::execute(ctx, track, media, source, start)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::Created(id)),
                Some(inverse),
            ))
        }
        EditCommand::AddGenerated {
            track,
            generator,
            timeline,
        } => {
            let (id, inverse) = edit::add_generated::execute(ctx, track, generator, timeline)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::Created(id)),
                Some(inverse),
            ))
        }
        EditCommand::SetGenerator { clip, generator } => {
            let inverse = edit::set_generator::execute(ctx, clip, generator)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::Updated(clip)),
                Some(inverse),
            ))
        }
        EditCommand::SetClipTransform {
            clip,
            transform,
            at,
        } => {
            let inverse = edit::set_transform::execute(ctx, clip, transform, at)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::Updated(clip)),
                Some(inverse),
            ))
        }
        EditCommand::SetParamKeyframe {
            clip,
            param,
            at,
            value,
            easing,
        } => {
            let inverse = edit::set_param::set_keyframe(ctx, clip, param, at, value, easing)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::Updated(clip)),
                Some(inverse),
            ))
        }
        EditCommand::RemoveParamKeyframe { clip, param, at } => {
            let inverse = edit::set_param::remove_keyframe(ctx, clip, param, at)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::Updated(clip)),
                Some(inverse),
            ))
        }
        EditCommand::SetParamConstant { clip, param, value } => {
            let inverse = edit::set_param::set_constant(ctx, clip, param, value)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::Updated(clip)),
                Some(inverse),
            ))
        }
        EditCommand::SetClipSpeed {
            clip,
            speed,
            reversed,
        } => {
            let inverse = edit::set_speed::set_speed(ctx, clip, speed, reversed)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::Updated(clip)),
                Some(inverse),
            ))
        }
        EditCommand::SetSpeedCurve { clip, curve } => {
            let inverse = edit::set_speed_curve::set_speed_curve(ctx, clip, curve)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::Updated(clip)),
                Some(inverse),
            ))
        }
        EditCommand::SetClipPitch {
            clip,
            preserve_pitch,
        } => {
            let inverse = edit::set_pitch::set_pitch(ctx, clip, preserve_pitch)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::Updated(clip)),
                Some(inverse),
            ))
        }
        EditCommand::SetClipCrop {
            clip,
            crop,
            flip_h,
            flip_v,
        } => {
            let inverse = edit::set_crop::set_crop(ctx, clip, crop, flip_h, flip_v)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::Updated(clip)),
                Some(inverse),
            ))
        }
        EditCommand::SetClipAudio {
            clip,
            volume,
            fade_in,
            fade_out,
        } => {
            let inverse = edit::set_audio::set_audio(ctx, clip, volume, fade_in, fade_out)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::Updated(clip)),
                Some(inverse),
            ))
        }
        EditCommand::SetClipDenoise { clip, denoise } => {
            let inverse = edit::set_denoise::set_denoise(ctx, clip, denoise)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::Updated(clip)),
                Some(inverse),
            ))
        }
        EditCommand::AddEffect { clip, effect_id } => {
            let inverse = edit::set_effect::add_effect(ctx, clip, &effect_id)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::Updated(clip)),
                Some(inverse),
            ))
        }
        EditCommand::RemoveEffect { clip, index } => {
            let inverse = edit::set_effect::remove_effect(ctx, clip, index)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::Updated(clip)),
                Some(inverse),
            ))
        }
        EditCommand::SetEffectParam {
            clip,
            index,
            param,
            value,
        } => {
            let inverse = edit::set_effect::set_effect_param(ctx, clip, index, param, value)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::Updated(clip)),
                Some(inverse),
            ))
        }
        EditCommand::AddTransition {
            clip,
            transition_id,
        } => {
            let inverse = edit::set_transition::add_transition(ctx, clip, &transition_id)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::Updated(clip)),
                Some(inverse),
            ))
        }
        EditCommand::RemoveTransition { clip } => {
            let inverse = edit::set_transition::remove_transition(ctx, clip)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::Updated(clip)),
                Some(inverse),
            ))
        }
        EditCommand::SetTransition { clip, duration } => {
            let inverse = edit::set_transition::set_transition(ctx, clip, duration)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::Updated(clip)),
                Some(inverse),
            ))
        }
        EditCommand::SplitClip { clip, at } => {
            let guard = transitions_guard(ctx);
            let (id, primary) = edit::split_clip::execute(ctx, clip, at)?;
            let inverse = finalize_structural(ctx, guard, primary);
            Ok((
                ApplyOutcome::Edited(EditOutcome::Created(id)),
                Some(inverse),
            ))
        }
        EditCommand::TrimClip { clip, timeline } => {
            let guard = transitions_guard(ctx);
            let primary = edit::trim_clip::execute(ctx, clip, timeline)?;
            let inverse = finalize_structural(ctx, guard, primary);
            Ok((
                ApplyOutcome::Edited(EditOutcome::Updated(clip)),
                Some(inverse),
            ))
        }
        EditCommand::MoveClip {
            clip,
            to_track,
            start,
        } => {
            let guard = transitions_guard(ctx);
            let primary = edit::move_clip::execute(ctx, clip, to_track, start)?;
            let inverse = finalize_structural(ctx, guard, primary);
            Ok((
                ApplyOutcome::Edited(EditOutcome::Updated(clip)),
                Some(inverse),
            ))
        }
        EditCommand::RemoveClip { clip } => {
            let guard = transitions_guard(ctx);
            let primary = Box::new(RemoveClipAction { clip }).apply(ctx)?;
            let inverse = finalize_structural(ctx, guard, primary);
            Ok((
                ApplyOutcome::Edited(EditOutcome::Removed(clip)),
                Some(inverse),
            ))
        }
        EditCommand::RemoveTrack { track } => {
            let guard = transitions_guard(ctx);
            let primary = Box::new(RemoveTrackAction { track_id: track }).apply(ctx)?;
            let inverse = finalize_structural(ctx, guard, primary);
            Ok((
                ApplyOutcome::Edited(EditOutcome::RemovedTrack(track)),
                Some(inverse),
            ))
        }
        EditCommand::SetTrackEnabled { track, enabled } => {
            let inverse =
                edit::set_track_flags::execute(ctx, track, Some(enabled), None, None, None)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::UpdatedTrack(track)),
                Some(inverse),
            ))
        }
        EditCommand::SetTrackMuted { track, muted } => {
            let inverse =
                edit::set_track_flags::execute(ctx, track, None, Some(muted), None, None)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::UpdatedTrack(track)),
                Some(inverse),
            ))
        }
        EditCommand::SetTrackLocked { track, locked } => {
            let inverse =
                edit::set_track_flags::execute(ctx, track, None, None, Some(locked), None)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::UpdatedTrack(track)),
                Some(inverse),
            ))
        }
        EditCommand::SetTrackDuckSource { track, duck_source } => {
            let inverse =
                edit::set_track_flags::execute(ctx, track, None, None, None, Some(duck_source))?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::UpdatedTrack(track)),
                Some(inverse),
            ))
        }
        EditCommand::RippleDelete { clip } => {
            let guard = transitions_guard(ctx);
            let primary = edit::ripple_delete::execute(ctx, clip)?;
            let inverse = finalize_structural(ctx, guard, primary);
            Ok((
                ApplyOutcome::Edited(EditOutcome::Removed(clip)),
                Some(inverse),
            ))
        }
        EditCommand::ShiftClips { track, from, delta } => {
            let guard = transitions_guard(ctx);
            let primary = edit::shift_clips::execute(ctx, track, from, delta)?;
            let inverse = finalize_structural(ctx, guard, primary);
            Ok((
                ApplyOutcome::Edited(EditOutcome::ShiftedTrack(track)),
                Some(inverse),
            ))
        }
        EditCommand::RippleInsert {
            track,
            media,
            source,
            at,
        } => {
            let guard = transitions_guard(ctx);
            let (id, primary) = edit::ripple_insert::execute(ctx, track, media, source, at)?;
            let inverse = finalize_structural(ctx, guard, primary);
            Ok((
                ApplyOutcome::Edited(EditOutcome::Created(id)),
                Some(inverse),
            ))
        }
        EditCommand::LinkClips { clips } => {
            let first = clips
                .first()
                .copied()
                .ok_or_else(|| EngineError::from(cutlass_models::ModelError::InvalidRange))?;
            let inverse = edit::link_clips::execute(ctx, &clips)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::Updated(first)),
                Some(inverse),
            ))
        }
        EditCommand::DuckLanes {
            voice,
            music,
            threshold,
            amount,
            attack,
            release,
        } => {
            let (clip, inverse) =
                edit::duck::duck(ctx, &voice, &music, threshold, amount, attack, release)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::Updated(clip)),
                Some(inverse),
            ))
        }
        EditCommand::DetectBeats { clip } => {
            let inverse = edit::detect_beats::detect(ctx, clip)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::Updated(clip)),
                Some(inverse),
            ))
        }
        EditCommand::ClearBeats { clip } => {
            let inverse = edit::detect_beats::clear(ctx, clip)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::Updated(clip)),
                Some(inverse),
            ))
        }
        EditCommand::RemoveSilences {
            clip,
            threshold,
            min_silence,
            padding,
        } => {
            let guard = transitions_guard(ctx);
            let (track, primary) =
                edit::remove_silences::remove(ctx, clip, threshold, min_silence, padding)?;
            let inverse = finalize_structural(ctx, guard, primary);
            Ok((
                ApplyOutcome::Edited(EditOutcome::ShiftedTrack(track)),
                Some(inverse),
            ))
        }
        EditCommand::AddMarker { at, name, color } => {
            let (id, inverse) = edit::marker::add(ctx, at, name, color)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::CreatedMarker(id)),
                Some(inverse),
            ))
        }
        EditCommand::RemoveMarker { marker } => {
            let inverse = Box::new(edit::marker::RemoveMarkerAction { marker }).apply(ctx)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::RemovedMarker(marker)),
                Some(inverse),
            ))
        }
        EditCommand::SetMarker {
            marker,
            at,
            name,
            color,
        } => {
            let inverse = edit::marker::set(ctx, marker, at, name, color)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::UpdatedMarker(marker)),
                Some(inverse),
            ))
        }
        EditCommand::SetCanvas { aspect, background } => {
            let inverse = edit::set_canvas::set_canvas(ctx, aspect, background)?;
            Ok((
                ApplyOutcome::Edited(EditOutcome::UpdatedCanvas),
                Some(inverse),
            ))
        }
    }
}
