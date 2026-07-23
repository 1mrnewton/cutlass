//! Generic per-param live preview override lane.
//!
//! UI/inspector drags send [`WorkerMsg::ParamOverride`]; the worker stores
//! `(clip, param) → value` on the engine and re-renders. Volume/pan also
//! mirror onto the preview audio mixer so drag is audible without a project
//! republish. Release commits via `SetParamConstant` / `SetParamKeyframe`
//! and clears with [`WorkerMsg::ClearParamOverride`]. No history, revision,
//! or projection.

use super::*;

/// Point the engine's param-override map at `(clip, param)` for the next
/// renders. Volume/pan also update the preview audio mixer. Unparsable ids
/// are dropped (stale projection race).
pub(super) fn apply_param_override(
    engine: &mut Engine,
    clip: &str,
    param: ClipParam,
    value: ParamValue,
    audio: Option<&crate::audio::AudioHandle>,
) {
    match parse_raw_id(clip).map(ClipId::from_raw) {
        Some(id) => {
            engine.set_param_override(id, param, value);
            if let Some(audio) = audio {
                audio.set_param_override(id, param, value);
            }
        }
        None => error!(clip, "param override ignored: unparsable clip id"),
    }
}

/// Drop every live param override for `clip` (and any mirrored audio
/// volume/pan override).
pub(super) fn clear_param_overrides(
    engine: &mut Engine,
    clip: &str,
    audio: Option<&crate::audio::AudioHandle>,
) {
    match parse_raw_id(clip).map(ClipId::from_raw) {
        Some(id) => {
            engine.clear_param_overrides(id);
            if let Some(audio) = audio {
                audio.clear_param_overrides(id);
            }
        }
        None => error!(clip, "clear param override ignored: unparsable clip id"),
    }
}

/// Drop one live param override after that param is committed (and any
/// mirrored audio volume/pan override).
pub(super) fn clear_param_override(
    engine: &mut Engine,
    clip: &str,
    param: ClipParam,
    audio: Option<&crate::audio::AudioHandle>,
) {
    if let Some(id) = parse_raw_id(clip).map(ClipId::from_raw) {
        engine.clear_param_override(id, param);
        if let Some(audio) = audio {
            audio.clear_param_override(id, param);
        }
    }
}

/// Pending `(clip, param) → value` accumulator for a coalesce burst.
type PendingParamOverrides = HashMap<(String, ClipParam), ParamValue>;

/// One mid-burst non-override: flush this batch (if any) onto the engine,
/// then dispatch `msg`. Produced by [`drain_param_override_queue`].
struct ParamOverrideDrainStep {
    /// Overrides to apply *before* `msg`. Empty when the accumulator was
    /// already flushed (or never dirtied) since the last step.
    flush: PendingParamOverrides,
    msg: WorkerMsg,
}

/// Result of [`drain_param_override_queue`]: playhead tick, whether `pending`
/// still needs a final flush, the leftover accumulator, and mid-burst steps.
type ParamOverrideDrain = (
    i64,
    bool,
    PendingParamOverrides,
    Vec<ParamOverrideDrainStep>,
);

/// Drain the worker inbox for a param-override coalesce burst.
///
/// Returns `(tick, dirty, pending, steps)`. Each step's `flush` map is a
/// snapshot taken *before* the accumulator is cleared — callers must apply
/// `flush` then dispatch `msg`. Leaving entries in `pending` across a
/// clearing dispatch is what resurrects committed overrides
/// (see `coalesce_does_not_resurrect_committed_override`).
fn drain_param_override_queue(
    mut pending: PendingParamOverrides,
    mut dirty: bool,
    mut tick: i64,
    req_rx: &Receiver<WorkerMsg>,
) -> ParamOverrideDrain {
    let mut steps = Vec::new();
    while let Ok(next) = req_rx.try_recv() {
        match next {
            WorkerMsg::Frame(latest) => tick = latest,
            WorkerMsg::ParamOverride {
                clip: c,
                param: p,
                value: v,
                tick: at,
            } => {
                pending.insert((c, p), v);
                tick = at;
                dirty = true;
            }
            other => {
                let flush = if std::mem::take(&mut dirty) {
                    // Clear so a later final flush cannot resurrect overrides
                    // that `other` is about to commit+clear.
                    std::mem::take(&mut pending)
                } else {
                    HashMap::new()
                };
                steps.push(ParamOverrideDrainStep { flush, msg: other });
            }
        }
    }
    (tick, dirty, pending, steps)
}

/// Coalesce a burst of [`WorkerMsg::ParamOverride`] messages: latest value
/// per `(clip, param)` wins, then at most one frame build. Mutating messages
/// drained from the queue are dispatched in order; a pending override is
/// applied *before* a drained mutation that might clear it (same rule as
/// [`WorkerMsg::TransformOverride`]).
#[allow(clippy::too_many_arguments)]
pub(super) fn coalesce_param_overrides(
    engine: &mut Engine,
    clipboard: &mut Option<Vec<ClipboardClip>>,
    main_magnet: &mut bool,
    linkage: &mut bool,
    clip: String,
    param: ClipParam,
    value: ParamValue,
    tick: i64,
    req_rx: &Receiver<WorkerMsg>,
    tl_rate: Rational,
    preview_weak: &slint::Weak<PreviewStore<'static>>,
    fit: &FrameFit,
    cache: &FrameCache,
    sprite_mode: &Cell<bool>,
    export_state: &ExportJobState,
    ui: &UiSink,
) -> i64 {
    // Pending map: latest value per (clip, param). Seeded with the head msg.
    let mut pending = PendingParamOverrides::new();
    pending.insert((clip, param), value);
    let (tick, dirty, pending, steps) = drain_param_override_queue(pending, true, tick, req_rx);

    for step in steps {
        if !step.flush.is_empty() {
            flush_param_overrides(engine, &step.flush, Some(&ui.audio));
        }
        dispatch(
            engine,
            clipboard,
            main_magnet,
            linkage,
            step.msg,
            tl_rate,
            preview_weak,
            fit,
            cache,
            sprite_mode,
            export_state,
            ui,
        );
    }

    if dirty {
        flush_param_overrides(engine, &pending, Some(&ui.audio));
    }
    tick
}

fn flush_param_overrides(
    engine: &mut Engine,
    pending: &HashMap<(String, ClipParam), ParamValue>,
    audio: Option<&crate::audio::AudioHandle>,
) {
    for ((clip, param), value) in pending {
        apply_param_override(engine, clip, *param, *value, audio);
    }
}

#[cfg(test)]
mod tests;
