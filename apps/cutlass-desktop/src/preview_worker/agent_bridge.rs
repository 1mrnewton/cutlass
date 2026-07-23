use super::*;

/// Replay a rehearsed agent plan on the live engine, re-validated step by
/// step, with sandbox-allocated ids remapped onto the ids the live engine
/// hands out. Each phase (see `PromptOutcome::phase_breaks`) commits as
/// its own history group — one undo step per phase. `after_step` runs
/// after every applied step (the worker publishes there, so the user
/// watches the plan land) and after the final commit or a rollback. A
/// failure rolls back the failing phase only and stops: earlier phases
/// were valid and committed, so they stay, each independently undoable —
/// the error says how many landed. Id maps persist across phases (a later
/// phase may address a clip an earlier one created). Each phase also
/// enforces the desktop's empty-lane invariant before it commits; a phase
/// checkpoint therefore must leave a coherent timeline on its own.
pub(crate) fn agent_replay(
    engine: &mut Engine,
    phases: Vec<Vec<AgentPlanStep>>,
    mut after_step: impl FnMut(&mut Engine),
) -> Result<(), String> {
    use std::collections::HashMap as Map;
    let mut clip_map: Map<u64, u64> = Map::new();
    let mut track_map: Map<u64, u64> = Map::new();
    let mut marker_map: Map<u64, u64> = Map::new();
    let phase_count = phases.len();
    let mut applied = 0usize;
    for (phase_index, steps) in phases.into_iter().enumerate() {
        let total = steps.len();
        let mut cleanup_lanes = Vec::new();
        engine.begin_group();
        for (index, mut step) in steps.into_iter().enumerate() {
            step.command.remap_ids(&clip_map, &track_map, &marker_map);
            let cleanup_lane = agent_cleanup_source_lane(engine, &step.command);
            let outcome = cutlass_ai::validate(&step.command, engine.project())
                .map_err(|r| r.message)
                .and_then(|lowered| engine.apply(lowered).map_err(|e| e.to_string()));
            match outcome {
                Ok(ApplyOutcome::Edited(edited)) => {
                    if let Some(lane) = cleanup_lane {
                        cleanup_lanes.push(lane);
                    }
                    match (step.created, &edited) {
                        (Some(AgentCreated::Clip(sandbox)), EditOutcome::Created(live)) => {
                            clip_map.insert(sandbox, live.raw());
                        }
                        (Some(AgentCreated::Track(sandbox)), EditOutcome::CreatedTrack(live)) => {
                            track_map.insert(sandbox, live.raw());
                        }
                        (Some(AgentCreated::Marker(sandbox)), EditOutcome::CreatedMarker(live)) => {
                            marker_map.insert(sandbox, live.raw());
                        }
                        _ => {}
                    }
                    after_step(engine);
                }
                Ok(other) => {
                    engine.rollback_group();
                    after_step(engine);
                    return Err(replay_failure(
                        phase_index,
                        phase_count,
                        index,
                        total,
                        &format!("unexpected engine outcome {other:?}"),
                    ));
                }
                Err(reason) => {
                    engine.rollback_group();
                    after_step(engine);
                    return Err(replay_failure(
                        phase_index,
                        phase_count,
                        index,
                        total,
                        &reason,
                    ));
                }
            }
        }
        // Agent commands bypass the desktop gesture helpers, so mirror
        // their CapCut lane policy at every commit boundary. A phase is an
        // independently undoable state and must satisfy desktop invariants;
        // if a plan wants to empty and refill a lane, those edits belong in
        // the same phase.
        cleanup_lanes.sort();
        cleanup_lanes.dedup();
        for lane in cleanup_lanes {
            remove_track_if_empty(engine, lane);
        }
        engine.commit_group();
        applied += total;
    }
    info!(steps = applied, phases = phase_count, "agent plan applied");
    after_step(engine);
    Ok(())
}

/// Replay failures name the phase (when there are several), the failing
/// step, and how much of the plan already landed — the transcript line
/// must let the user judge what a failure left behind.
pub(super) fn replay_failure(
    phase_index: usize,
    phase_count: usize,
    step_index: usize,
    steps_in_phase: usize,
    reason: &str,
) -> String {
    let step = format!("step {}/{}", step_index + 1, steps_in_phase);
    let location = if phase_count > 1 {
        format!("phase {}/{} {step}", phase_index + 1, phase_count)
    } else {
        step
    };
    let landed = match phase_index {
        0 => "nothing was applied".to_string(),
        1 => format!("phase 1 of {phase_count} was applied and stays undoable"),
        n => format!("phases 1–{n} of {phase_count} were applied and stay undoable"),
    };
    format!("{location}: {reason} — {landed}")
}

/// Source lane that may become empty after an agent structural edit.
pub(super) fn agent_cleanup_source_lane(
    engine: &Engine,
    command: &cutlass_ai::WireCommand,
) -> Option<TrackId> {
    let clip = match command {
        cutlass_ai::WireCommand::MoveClip(args) => args.clip,
        cutlass_ai::WireCommand::RemoveClip(args) => args.clip,
        cutlass_ai::WireCommand::RippleDelete(args) => args.clip,
        _ => return None,
    };
    engine.project().timeline().track_of(ClipId::from_raw(clip))
}

/// Distinct error token when Apply's expected seed revision no longer
/// matches [`Engine::revision`]. Matched by the agent worker — not shown
/// verbatim in the transcript.
pub(crate) const STALE_PLAN_SEED_ERROR: &str = "__stale_plan_seed_revision__";

/// Apply a rehearsed plan when `expected_seed_revision` still matches the
/// live engine. Compared inside the worker on the same ordered channel as
/// mutations so a concurrent user edit cannot sneak past the check.
pub(crate) fn agent_apply_with_seed(
    engine: &mut Engine,
    phases: Vec<Vec<AgentPlanStep>>,
    expected_seed_revision: u64,
    after_step: impl FnMut(&mut Engine),
) -> Result<(), String> {
    if engine.revision() != expected_seed_revision {
        return Err(STALE_PLAN_SEED_ERROR.to_string());
    }
    agent_replay(engine, phases, after_step)
}

pub(super) fn agent_apply_and_publish(
    engine: &mut Engine,
    phases: Vec<Vec<AgentPlanStep>>,
    expected_seed_revision: u64,
    ui: &UiSink,
) -> Result<(), String> {
    agent_apply_with_seed(engine, phases, expected_seed_revision, |engine| {
        publish_projection(engine, ui)
    })
}
