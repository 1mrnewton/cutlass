//! Unit tests for parked-plan sandbox reseed / live-revision divergence.

use super::*;
use crate::agent_senses::AgentSenses;
use cutlass_ai::wire;
use cutlass_ai::{EngineBridge, ToolOutput, WireCommand};
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
