use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_ai::{
    EngineBridge, HostToolSpec, ProjectSummary, ToolOutput, WireCommand, summarize, validate,
};
use cutlass_commands::EditOutcome;
use cutlass_engine::{ApplyOutcome, Engine, EngineConfig};

use crate::agent_senses::AgentSenses;
use crate::preview_worker::WorkerHandle;

use super::types::{AgentCreated, AgentPlanStep};

pub(crate) fn sandbox_engine() -> Result<Engine, String> {
    Engine::new(EngineConfig::default())
        .map_err(|e| format!("agent sandbox engine failed to start: {e}"))
}

pub(crate) trait ProjectSnapshotSource {
    fn snapshot_project(&self) -> Option<cutlass_models::Project>;
}

impl ProjectSnapshotSource for WorkerHandle {
    fn snapshot_project(&self) -> Option<cutlass_models::Project> {
        WorkerHandle::snapshot_project(self)
    }
}

pub(crate) struct SandboxBridge<'a, W: ProjectSnapshotSource + ?Sized> {
    pub(crate) worker: &'a W,
    pub(crate) engine: &'a mut Engine,
    pub(crate) plan: &'a mut Vec<AgentPlanStep>,
    pub(crate) senses: &'a mut AgentSenses,
    pub(crate) default_playhead_seconds: f64,
}

impl<W: ProjectSnapshotSource + ?Sized> EngineBridge for SandboxBridge<'_, W> {
    fn summary(&mut self) -> ProjectSummary {
        summarize(self.engine.project())
    }

    fn sense_tools(&self) -> Vec<HostToolSpec> {
        AgentSenses::specs()
    }

    fn sense(
        &mut self,
        name: &str,
        arguments: &serde_json::Value,
        cancel: &AtomicBool,
    ) -> Result<ToolOutput, String> {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled before the media sense could run".into());
        }
        let output = self.senses.call(
            self.engine.project(),
            self.default_playhead_seconds,
            name,
            arguments,
        )?;
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled while the media sense was running".into());
        }
        Ok(output)
    }

    fn before_host_call(
        &mut self,
        name: &str,
        _arguments: &serde_json::Value,
    ) -> Result<(), String> {
        if crate::agent_project::mutates_live_project(name) && !self.plan.is_empty() {
            return Err(format!(
                "{name} cannot run while timeline edits are staged; project operations must \
                 happen before staged edits, or after the user applies or discards the pending \
                 plan"
            ));
        }
        Ok(())
    }

    fn after_host_call(
        &mut self,
        name: &str,
        _arguments: &serde_json::Value,
        _result: Result<&ToolOutput, &str>,
    ) -> Result<(), String> {
        if !crate::agent_project::mutates_live_project(name) {
            return Ok(());
        }

        let snapshot = self.worker.snapshot_project().ok_or_else(|| {
            format!(
                "could not reconcile the agent sandbox after project host call '{name}': \
                 the editor engine did not reply with a live project snapshot"
            )
        })?;
        self.plan.clear();
        self.engine.reset_project(snapshot);
        // `reset_project` clears history, including the prompt's pending
        // group. Reopen it immediately so any later staged edit is still
        // covered by the core loop's normal abort rollback.
        self.engine.begin_group();
        Ok(())
    }

    fn apply(&mut self, command: &WireCommand) -> Result<EditOutcome, String> {
        let lowered = validate(command, self.engine.project()).map_err(|r| r.message)?;
        match self.engine.apply(lowered) {
            Ok(ApplyOutcome::Edited(outcome)) => {
                let created = match &outcome {
                    EditOutcome::Created(id) => Some(AgentCreated::Clip(id.raw())),
                    EditOutcome::CreatedTrack(id) => Some(AgentCreated::Track(id.raw())),
                    EditOutcome::CreatedMarker(id) => Some(AgentCreated::Marker(id.raw())),
                    _ => None,
                };
                self.plan.push(AgentPlanStep {
                    command: command.clone(),
                    created,
                });
                Ok(outcome)
            }
            Ok(other) => Err(format!("unexpected engine outcome: {other:?}")),
            Err(e) => Err(e.to_string()),
        }
    }

    fn check(&mut self, command: &WireCommand) -> Result<(), String> {
        validate(command, self.engine.project())
            .map(|_| ())
            .map_err(|r| r.message)
    }

    fn begin_group(&mut self) {
        self.engine.begin_group();
    }

    fn end_group(&mut self) {
        self.engine.commit_group();
    }

    fn rollback_group(&mut self) {
        self.engine.rollback_group();
    }
}
