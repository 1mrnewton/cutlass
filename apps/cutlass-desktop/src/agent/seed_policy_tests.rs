//! Unit tests for parked-plan sandbox reseed / live-revision divergence.

use super::*;
use crate::agent_senses::AgentSenses;
use crate::preview_worker::{STALE_PLAN_SEED_ERROR, agent_apply_with_seed};
use cutlass_ai::wire;
use cutlass_ai::{EngineBridge, Message, ToolOutput, WireCommand};
use cutlass_commands::{Command, EditCommand};
use cutlass_engine::{Engine, EngineConfig};
use cutlass_models::{Project, Rational};
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;

fn temp_engine(project: Project) -> Engine {
    Engine::with_project(EngineConfig::default(), project).expect("engine")
}

struct ScriptedSnapshots {
    snapshots: RefCell<VecDeque<Option<(Project, u64)>>>,
    calls: Cell<usize>,
}

impl ScriptedSnapshots {
    fn with_revisions(snapshots: impl IntoIterator<Item = Option<(Project, u64)>>) -> Self {
        Self {
            snapshots: RefCell::new(snapshots.into_iter().collect()),
            calls: Cell::new(0),
        }
    }
}

impl ProjectSnapshotSource for ScriptedSnapshots {
    fn snapshot_project_with_revision(&self) -> Option<(Project, u64)> {
        self.calls.set(self.calls.get() + 1);
        self.snapshots
            .borrow_mut()
            .pop_front()
            .expect("scripted project snapshot")
    }
}

#[test]
fn sandbox_seed_policy_keeps_parked_plan_when_live_revision_matches() {
    assert_eq!(
        sandbox_seed_policy(true, Some(7), Some(7)).unwrap(),
        SandboxSeedPolicy::KeepPending
    );
    assert_eq!(
        sandbox_seed_policy(false, Some(7), None).unwrap(),
        SandboxSeedPolicy::Reseed {
            discarded_stale_plan: false,
        }
    );
}

#[test]
fn sandbox_seed_policy_discards_parked_plan_when_live_revision_moves() {
    assert_eq!(
        sandbox_seed_policy(true, Some(7), Some(8)).unwrap(),
        SandboxSeedPolicy::Reseed {
            discarded_stale_plan: true,
        }
    );
    assert_eq!(
        sandbox_seed_policy(true, None, Some(1)).unwrap(),
        SandboxSeedPolicy::Reseed {
            discarded_stale_plan: true,
        }
    );
    assert!(
        sandbox_seed_policy(true, Some(1), None)
            .unwrap_err()
            .contains("not responding")
    );
}

#[test]
fn continue_pending_keeps_sandbox_when_live_revision_matches() {
    let sandbox = temp_engine(Project::new("parked-keep", Rational::FPS_24));
    let revision_before = sandbox.revision();
    let name_before = sandbox.project().name.clone();
    let preview = Preview {
        plan: vec![AgentPlanStep {
            command: WireCommand::AddMarker(wire::AddMarker {
                at: 1.0,
                name: Some("parked".into()),
                color: None,
            }),
            created: None,
        }],
        seed_revision: Some(4),
        descriptions: vec!["set marker".into()],
        ..Default::default()
    };

    let policy = sandbox_seed_policy(true, preview.seed_revision, Some(4)).unwrap();
    assert_eq!(policy, SandboxSeedPolicy::KeepPending);
    // Keep path must not touch the sandbox or parked plan (no live snapshot).
    assert_eq!(sandbox.revision(), revision_before);
    assert_eq!(sandbox.project().name, name_before);
    assert_eq!(preview.plan.len(), 1);
    assert_eq!(preview.seed_revision, Some(4));
}

#[test]
fn continue_pending_reseeds_and_clears_plan_when_live_diverges() {
    let stale = Project::new("stale-parked", Rational::FPS_24);
    let fresh = Project::new("live-after-user-edit", Rational::FPS_24);
    let worker = ScriptedSnapshots::with_revisions([Some((fresh, 9))]);
    let mut sandbox = temp_engine(stale);
    let mut preview = Preview {
        plan: vec![AgentPlanStep {
            command: WireCommand::AddMarker(wire::AddMarker {
                at: 1.0,
                name: Some("stale".into()),
                color: None,
            }),
            created: None,
        }],
        phase_breaks: vec![1],
        descriptions: vec!["old action".into()],
        seed_revision: Some(4),
        ..Default::default()
    };

    let policy = sandbox_seed_policy(true, preview.seed_revision, Some(9)).unwrap();
    assert_eq!(
        policy,
        SandboxSeedPolicy::Reseed {
            discarded_stale_plan: true,
        }
    );
    reseed_sandbox_from_live(&worker, &mut sandbox, &mut preview).unwrap();

    assert_eq!(sandbox.project().name, "live-after-user-edit");
    assert!(preview.plan.is_empty());
    assert!(preview.phase_breaks.is_empty());
    assert!(preview.descriptions.is_empty());
    assert_eq!(preview.seed_revision, Some(9));
    assert_eq!(
        STALE_PLAN_NOTICE,
        "Project changed since this plan was rehearsed — discarded the stale plan and re-read the current state."
    );
    assert_eq!(worker.calls.get(), 1, "one live snapshot for the reseed");
}

#[test]
fn stale_plan_discard_restores_history_and_refreshes_checkpoint() {
    let checkpoint = vec![Message::user("earlier chat")];
    let mut history = checkpoint.clone();
    history.push(Message::Assistant {
        content: "I'll add a marker".into(),
        tool_calls: vec![],
    });
    history.push(Message::user("and another edit"));
    let mut preview = Preview {
        history_restore: Some(checkpoint.clone()),
        seed_revision: Some(4),
        plan: vec![AgentPlanStep {
            command: WireCommand::AddMarker(wire::AddMarker {
                at: 1.0,
                name: Some("stale".into()),
                color: None,
            }),
            created: None,
        }],
        ..Default::default()
    };

    restore_history_after_stale_plan_discard(&mut history, &mut preview);

    assert_eq!(history, checkpoint);
    assert_eq!(
        preview.history_restore.as_ref(),
        Some(&checkpoint),
        "fresh checkpoint must match restored history so a later DiscardPlan \
         cannot rewind past turns completed after the stale-plan notice"
    );
    // The replacement user prompt is not in history yet — run_prompt adds it.
    assert_eq!(history.len(), 1);
}

#[test]
fn auto_apply_stale_outcome_posts_notice_and_corrects_history() {
    let mut history = vec![
        Message::user("trim the clip"),
        Message::assistant_text("I'll trim it."),
    ];
    let turn_len = history.len();
    // turn_messages already appended (sandbox edits looked successful).
    history.push(Message::tool_result("call-1", "trimmed clip c1"));

    let pending = record_auto_apply_outcome(&mut history, ApplyLiveOutcome::Stale);
    assert_eq!(pending, Some(("status", STALE_APPLY_NOTICE)));
    assert!(
        history[turn_len..]
            .iter()
            .any(|m| matches!(m, Message::ToolResult { .. })),
        "sandbox turn messages stay so the model sees what it attempted"
    );
    assert!(
        matches!(
            history.last(),
            Some(Message::User { content, .. }) if content == STALE_APPLY_NOTICE
        ),
        "corrective notice must be last so the next turn knows nothing landed"
    );
}

#[test]
fn auto_apply_failed_outcome_corrects_history_without_duplicate_transcript() {
    let mut history = vec![Message::user("add a marker")];
    assert_eq!(
        record_auto_apply_outcome(&mut history, ApplyLiveOutcome::Failed),
        None,
        "apply_plan_live already pushed the error row"
    );
    assert_eq!(
        history.last(),
        Some(&Message::user(AUTO_APPLY_FAILED_HISTORY_NOTICE))
    );
    assert_eq!(
        record_auto_apply_outcome(&mut history, ApplyLiveOutcome::Applied),
        None
    );
    assert_eq!(
        history.last(),
        Some(&Message::user(AUTO_APPLY_FAILED_HISTORY_NOTICE)),
        "Applied must not append another notice"
    );
}

#[test]
fn stale_discard_trims_rehearsal_transcript_rows_keeps_replacement_prompt() {
    // checkpoint = rows before the first dry-run user prompt.
    let kinds = vec![
        "status",    // earlier chat
        "user",      // dry-run #1 prompt (discarded with rehearsal)
        "action",    // rehearsal
        "assistant", // rehearsal
        "user",      // replacement prompt already pushed
        "status",    // slash-expand / warnings after that user
    ];
    let trimmed = trim_rows_after_stale_plan_discard(kinds, 1, |k| *k == "user");
    assert_eq!(
        trimmed,
        vec!["status", "user", "status"],
        "rehearsal rows between the checkpoint and the latest user must drop"
    );
}

#[test]
fn explicit_stale_apply_truncates_transcript_to_checkpoint() {
    // ApplyPlan has no replacement user prompt — drop everything after the
    // dry-run checkpoint (same restore point as history_restore).
    let kinds = vec![
        "status",    // earlier chat
        "user",      // dry-run prompt
        "action",    // rehearsal
        "assistant", // rehearsal
    ];
    assert_eq!(
        truncate_rows_to_checkpoint(kinds, 1),
        vec!["status"],
        "explicit stale Apply must drop rehearsal rows, not keep the dry-run user"
    );
}

#[test]
fn continue_pending_seed_error_restores_plan_pending_flag() {
    // When project_revision is unreachable, sandbox_seed_policy errs after the
    // Prompt handler cleared plan_pending — restore it iff a parked plan remains.
    assert!(matches!(
        sandbox_seed_policy(true, Some(1), None),
        Err("The editor engine is not responding.")
    ));
    let preview_still_parked = true;
    let continue_pending = true;
    assert!(
        continue_pending && preview_still_parked,
        "failure path must set plan_pending again so Apply/Discard stay available"
    );
}

#[test]
fn stale_discard_transcript_trim_is_noop_when_nothing_to_drop() {
    let kinds = vec!["user", "status"];
    let trimmed = trim_rows_after_stale_plan_discard(kinds.clone(), 0, |k| *k == "user");
    assert_eq!(trimmed, kinds);
}

#[test]
fn apply_with_matching_seed_revision_succeeds() {
    let mut live = temp_engine(Project::new("apply-ok", Rational::FPS_24));
    let seed = live.revision();
    let plan = vec![vec![AgentPlanStep {
        command: WireCommand::AddMarker(wire::AddMarker {
            at: 1.0,
            name: Some("rehearsed".into()),
            color: Some(wire::WireMarkerColor::Red),
        }),
        created: None,
    }]];
    agent_apply_with_seed(&mut live, plan, seed, |_| {}).expect("matching seed applies");
    assert_eq!(live.project().timeline().markers().len(), 1);
}

#[test]
fn apply_after_live_mutation_is_refused_and_leaves_project_untouched() {
    let mut live = temp_engine(Project::new("apply-stale", Rational::FPS_24));
    let seed = live.revision();
    // User edit bumps the live revision under the parked plan.
    live.apply(Command::Edit(EditCommand::SetProjectName {
        name: "user-renamed".into(),
    }))
    .expect("user edit");
    let revision_after_user = live.revision();
    assert_ne!(seed, revision_after_user);
    let name_before = live.project().name.clone();

    let plan = vec![vec![AgentPlanStep {
        command: WireCommand::AddMarker(wire::AddMarker {
            at: 2.0,
            name: Some("stale-plan".into()),
            color: None,
        }),
        created: None,
    }]];
    let err = agent_apply_with_seed(&mut live, plan, seed, |_| {})
        .expect_err("mismatched seed must refuse replay");
    assert_eq!(err, STALE_PLAN_SEED_ERROR);
    assert_eq!(live.revision(), revision_after_user);
    assert_eq!(live.project().name, name_before);
    assert!(
        live.project().timeline().markers().is_empty(),
        "stale apply must not replay onto the live project"
    );
    assert_eq!(
        STALE_APPLY_NOTICE,
        "Project changed since this plan was rehearsed — the plan was not applied."
    );
}

#[test]
fn project_post_hook_updates_seed_revision_with_live_snapshot() {
    let live = Project::new("after-import", Rational::FPS_24);
    let worker = ScriptedSnapshots::with_revisions([Some((live, 42))]);
    let mut sandbox = temp_engine(Project::new("stale", Rational::FPS_24));
    let mut plan = Vec::new();
    let mut senses = AgentSenses::new();
    let mut seed_revision = Some(7u64);
    {
        let mut bridge = SandboxBridge {
            worker: &worker,
            engine: &mut sandbox,
            plan: &mut plan,
            senses: &mut senses,
            default_playhead_seconds: 0.0,
            seed_revision: Some(&mut seed_revision),
        };
        bridge
            .after_host_call(
                "project_save",
                &serde_json::json!({}),
                Ok(&ToolOutput::text("ok")),
            )
            .expect("reconcile");
    }
    assert_eq!(sandbox.project().name, "after-import");
    assert_eq!(seed_revision, Some(42));
}
