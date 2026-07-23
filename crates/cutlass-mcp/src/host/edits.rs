//! Validated wire-edit apply / undo / redo on the engine host thread.
//!
//! External agents never construct [`EditCommand`]s. Every mutation goes
//! `WireCommand` → [`cutlass_ai::validate`] → [`Engine::apply`], the same
//! gate the in-app agent uses. A batch is one undo group (all-or-nothing).

use cutlass_ai::agent::describe_action;
use cutlass_ai::{WireCommand, validate};
use cutlass_commands::EditOutcome;
use cutlass_engine::{ApplyOutcome, Engine};
use serde::Serialize;
use serde_json::{Value, json};

use super::{Meta, eng_err, meta_of, require_engine_mut};

/// One successfully applied wire edit inside a batch.
#[derive(Debug, Serialize)]
pub struct AppliedEdit {
    /// Wire tool name (`split_clip`, `add_generated`, …).
    pub command: String,
    /// Editor-language transcript line (same format as in-app undo tooltips).
    pub action: String,
    /// Compact outcome so agents can chain created ids into follow-up edits.
    pub outcome: Value,
}

/// Result of an all-or-nothing `apply_edits` batch.
#[derive(Debug, Serialize)]
pub struct AppliedBatch {
    pub applied: Vec<AppliedEdit>,
    pub meta: Meta,
}

/// Result of `undo` / `redo`. `changed: false` means the stack was empty —
/// that is success, not an error.
#[derive(Debug, Serialize)]
pub struct UndoResult {
    pub changed: bool,
    pub meta: Meta,
}

pub(super) fn do_apply_edits(
    slot: &mut Option<Engine>,
    edits: Vec<Value>,
) -> Result<AppliedBatch, String> {
    if edits.is_empty() {
        return Err("empty edits batch — nothing to apply".into());
    }
    let engine = require_engine_mut(slot)?;
    engine.begin_group();

    let mut applied = Vec::with_capacity(edits.len());
    for (index, edit) in edits.into_iter().enumerate() {
        match apply_one(engine, edit) {
            Ok(row) => applied.push(row),
            Err(msg) => {
                engine.rollback_group();
                return Err(format!(
                    "edit[{index}] failed: {msg} — whole batch rolled back; project unchanged"
                ));
            }
        }
    }

    engine.commit_group();
    Ok(AppliedBatch {
        applied,
        meta: meta_of(engine),
    })
}

fn apply_one(engine: &mut Engine, edit: Value) -> Result<AppliedEdit, String> {
    let mut obj = match edit {
        Value::Object(map) => map,
        other => {
            return Err(format!(
                "each edit must be a JSON object with a \"command\" key, got {other}"
            ));
        }
    };
    let name = match obj.remove("command") {
        Some(Value::String(s)) => s,
        Some(other) => {
            return Err(format!(
                "\"command\" must be a string tool name, got {other}"
            ));
        }
        None => {
            return Err(
                "missing \"command\" key — expected {\"command\": \"<tool>\", ...arguments}".into(),
            );
        }
    };
    // Remaining keys are the tool arguments; from_tool_call gives hint-rich
    // parse errors (and "did you mean"-style unknown-name messages).
    let args = Value::Object(obj);
    let wire = WireCommand::from_tool_call(&name, args)?;
    let command = validate(&wire, engine.project()).map_err(|r| r.message)?;
    let outcome = match engine.apply(command).map_err(eng_err)? {
        ApplyOutcome::Edited(outcome) => outcome,
        other => {
            return Err(format!("unexpected apply outcome for {name}: {other:?}"));
        }
    };
    let action = describe_action(&wire, Some(&outcome));
    Ok(AppliedEdit {
        command: name,
        action,
        outcome: outcome_json(&outcome),
    })
}

pub(super) fn do_undo(slot: &mut Option<Engine>) -> Result<UndoResult, String> {
    let engine = require_engine_mut(slot)?;
    // Must not run inside an open group — apply_edits always commits/rolls back.
    let changed = engine.undo();
    Ok(UndoResult {
        changed,
        meta: meta_of(engine),
    })
}

pub(super) fn do_redo(slot: &mut Option<Engine>) -> Result<UndoResult, String> {
    let engine = require_engine_mut(slot)?;
    let changed = engine.redo();
    Ok(UndoResult {
        changed,
        meta: meta_of(engine),
    })
}

/// Compact agent-facing outcome shape (`{"kind":"created","clip":5}`).
fn outcome_json(outcome: &EditOutcome) -> Value {
    match outcome {
        EditOutcome::Created(id) => json!({ "kind": "created", "clip": id.raw() }),
        EditOutcome::CreatedTrack(id) => {
            json!({ "kind": "created_track", "track": id.raw() })
        }
        EditOutcome::Updated(id) => json!({ "kind": "updated", "clip": id.raw() }),
        EditOutcome::Removed(id) => json!({ "kind": "removed", "clip": id.raw() }),
        EditOutcome::RemovedTrack(id) => {
            json!({ "kind": "removed_track", "track": id.raw() })
        }
        EditOutcome::ShiftedTrack(id) => {
            json!({ "kind": "shifted_track", "track": id.raw() })
        }
        EditOutcome::UpdatedTrack(id) => {
            json!({ "kind": "updated_track", "track": id.raw() })
        }
        EditOutcome::CreatedMarker(id) => {
            json!({ "kind": "created_marker", "marker": id.raw() })
        }
        EditOutcome::UpdatedMarker(id) => {
            json!({ "kind": "updated_marker", "marker": id.raw() })
        }
        EditOutcome::RemovedMarker(id) => {
            json!({ "kind": "removed_marker", "marker": id.raw() })
        }
        EditOutcome::UpdatedCanvas => json!({ "kind": "updated_canvas" }),
        EditOutcome::UpdatedProject => json!({ "kind": "updated_project" }),
    }
}
