//! AI agent worker: runs prompts against a sandbox engine, then replays
//! the validated plan on the live engine as one undoable history group.
//!
//! Why a sandbox? The agent loop holds a conversation across network
//! waits, and the engine's history groups don't nest — an open group on
//! the live engine would swallow any user edit made while the model
//! thinks. Instead the prompt edits a throwaway [`Engine`] seeded with a
//! snapshot of the live project: tool calls really apply (so the model
//! sees created clip/track ids and the world it changed), and nothing
//! touches the user's timeline until the plan replays atomically via
//! [`WorkerHandle::agent_apply_plan`]. Replay re-validates every step
//! against the live project and remaps ids the sandbox allocated, so a
//! mid-prompt user edit can only fail the plan loudly — never corrupt it.
//!
//! With the dry-run toggle on (the default), the plan is parked here and
//! the chat panel shows an Apply / Discard card instead of auto-applying.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, Sender, unbounded};
use cutlass_ai::providers::openai_compat::OpenAiCompatProvider;
use cutlass_ai::{
    AgentConfig, AgentEvent, AgentExtensions, EditorContext, EngineBridge, Message, ProjectSummary,
    PromptStatus, WireCommand, compose_rules, expand_slash_command, load_agent_dir, merge_skills,
    run_prompt, summarize, validate,
};
use cutlass_commands::EditOutcome;
use cutlass_engine::{ApplyOutcome, Engine, EngineConfig};
use slint::{Model, ModelRc, SharedString, VecModel};
use tracing::{error, info, warn};

use crate::preview_worker::WorkerHandle;
use crate::{AgentEntry, AgentStore};

/// An entity id the sandbox allocated while rehearsing a command. Replay
/// maps it onto the id the live engine allocates for the same step.
#[derive(Debug, Clone, Copy)]
pub enum AgentCreated {
    Clip(u64),
    Track(u64),
    Marker(u64),
}

/// One rehearsed command, ready for live replay.
#[derive(Debug, Clone)]
pub struct AgentPlanStep {
    pub command: WireCommand,
    /// Sandbox id this step created (`split_clip`'s right half,
    /// `add_track`'s lane, …), if any.
    pub created: Option<AgentCreated>,
}

enum AgentRequest {
    Prompt {
        prompt: String,
        context: EditorContext,
        dry_run: bool,
    },
    ApplyPlan,
    DiscardPlan,
    /// Forget the conversation so far (the project was replaced wholesale —
    /// open / new / restore — and prior turns name clips that are gone).
    ResetHistory,
}

#[derive(Clone)]
pub struct AgentHandle {
    tx: Sender<AgentRequest>,
    cancel: Arc<AtomicBool>,
}

impl AgentHandle {
    pub fn prompt(&self, prompt: String, context: EditorContext, dry_run: bool) {
        let _ = self.tx.send(AgentRequest::Prompt {
            prompt,
            context,
            dry_run,
        });
    }

    pub fn apply_plan(&self) {
        let _ = self.tx.send(AgentRequest::ApplyPlan);
    }

    pub fn discard_plan(&self) {
        let _ = self.tx.send(AgentRequest::DiscardPlan);
    }

    /// Drop the multi-turn conversation memory; the next prompt starts a
    /// fresh dialogue. Fired when the project is replaced wholesale.
    pub fn reset_history(&self) {
        let _ = self.tx.send(AgentRequest::ResetHistory);
    }

    /// Cooperative cancel: the provider checks this flag between stream
    /// chunks, so a running prompt aborts within one network read.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

pub struct AgentWorker {
    handle: AgentHandle,
    _join: JoinHandle<()>,
}

impl AgentWorker {
    pub fn spawn(
        worker: WorkerHandle,
        store: slint::Weak<AgentStore<'static>>,
    ) -> Result<Self, String> {
        let (tx, rx) = unbounded();
        let cancel = Arc::new(AtomicBool::new(false));
        let thread_cancel = cancel.clone();
        let join = std::thread::Builder::new()
            .name("cutlass-agent".into())
            .spawn(move || agent_main(worker, store, rx, thread_cancel))
            .map_err(|e| e.to_string())?;
        Ok(Self {
            handle: AgentHandle { tx, cancel },
            _join: join,
        })
    }

    pub fn handle(&self) -> AgentHandle {
        self.handle.clone()
    }
}

fn entry(kind: &str, text: impl Into<SharedString>) -> AgentEntry {
    AgentEntry {
        kind: kind.into(),
        text: text.into(),
    }
}

fn with_transcript(
    store: &slint::Weak<AgentStore<'static>>,
    f: impl FnOnce(&VecModel<AgentEntry>) + Send + 'static,
) {
    let store = store.clone();
    let _ = slint::invoke_from_event_loop(move || {
        if let Some(store) = store.upgrade() {
            let transcript = store.get_transcript();
            if let Some(model) = transcript.as_any().downcast_ref::<VecModel<AgentEntry>>() {
                f(model);
            }
        }
    });
}

fn push_entry(store: &slint::Weak<AgentStore<'static>>, kind: &'static str, text: String) {
    with_transcript(store, move |model| model.push(entry(kind, text)));
}

fn append_assistant_text(store: &slint::Weak<AgentStore<'static>>, delta: String) {
    with_transcript(store, move |model| {
        let last = model.row_count().wrapping_sub(1);
        match model.row_data(last) {
            Some(e) if e.kind == "assistant" => {
                let mut e = e;
                e.text = format!("{}{}", e.text, delta).into();
                model.set_row_data(last, e);
            }
            _ => model.push(entry("assistant", delta)),
        }
    });
}

fn with_store(
    store: &slint::Weak<AgentStore<'static>>,
    f: impl FnOnce(AgentStore<'_>) + Send + 'static,
) {
    let store = store.clone();
    let _ = slint::invoke_from_event_loop(move || {
        if let Some(store) = store.upgrade() {
            f(store);
        }
    });
}

fn sandbox_engine() -> Result<Engine, String> {
    Engine::new(EngineConfig::default())
        .map_err(|e| format!("agent sandbox engine failed to start: {e}"))
}

struct SandboxBridge<'a> {
    engine: &'a mut Engine,
    plan: &'a mut Vec<AgentPlanStep>,
}

impl EngineBridge for SandboxBridge<'_> {
    fn summary(&mut self) -> ProjectSummary {
        summarize(self.engine.project())
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

#[derive(Default)]
struct Preview {
    plan: Vec<AgentPlanStep>,
    descriptions: Vec<SharedString>,
    history_restore: Option<Vec<Message>>,
}

impl Preview {
    fn is_pending(&self) -> bool {
        !self.plan.is_empty()
    }

    fn clear(&mut self) {
        self.plan.clear();
        self.descriptions.clear();
        self.history_restore = None;
    }
}

fn agent_main(
    worker: WorkerHandle,
    store: slint::Weak<AgentStore<'static>>,
    rx: Receiver<AgentRequest>,
    cancel: Arc<AtomicBool>,
) {
    let mut sandbox: Option<Engine> = None;
    let mut preview = Preview::default();
    let mut history: Vec<Message> = Vec::new();

    let config_path = cutlass_settings::default_config_path();
    let configured = cutlass_settings::load(&config_path)
        .map(|s| s.ai.is_configured())
        .unwrap_or(false);
    let path_text: SharedString = config_path.display().to_string().into();
    with_store(&store, move |s| {
        s.set_configured(configured);
        s.set_config_path(path_text);
    });

    while let Ok(req) = rx.recv() {
        match req {
            AgentRequest::Prompt {
                prompt,
                context,
                dry_run,
            } => {
                cancel.store(false, Ordering::Relaxed);
                if dry_run {
                    if !preview.is_pending() {
                        preview.history_restore = Some(history.clone());
                    }
                } else if preview.is_pending() {
                    if let Some(saved) = preview.history_restore.take() {
                        history = saved;
                    }
                    preview.clear();
                    push_entry(&store, "status", "Pending preview discarded.".into());
                }
                with_store(&store, |s| {
                    s.set_running(true);
                    s.set_plan_pending(false);
                    s.set_undo_offered(false);
                });
                push_entry(&store, "user", prompt.clone());

                // Reload ~/.cutlass/agent every prompt (tiny files) so
                // rule/skill/command edits apply without a restart.
                let agent_dir = load_agent_dir(&cutlass_settings::agent_dir());
                for warning in &agent_dir.warnings {
                    warn!(warning, "agent extension file skipped");
                    push_entry(&store, "status", warning.clone());
                }
                // Slash commands expand client-side; the transcript keeps
                // what was typed, the model sees the template.
                let sent = match expand_slash_command(&prompt, &agent_dir.commands) {
                    Some(expanded) => {
                        let name = prompt[1..].split_whitespace().next().unwrap_or("");
                        push_entry(&store, "status", format!("Expanded /{name}."));
                        expanded
                    }
                    None => prompt.clone(),
                };

                run_one_prompt(
                    &worker,
                    &store,
                    &mut sandbox,
                    &mut preview,
                    &mut history,
                    &sent,
                    context,
                    agent_dir,
                    dry_run,
                    &cancel,
                );

                with_store(&store, |s| s.set_running(false));
            }
            AgentRequest::ApplyPlan => {
                let plan = std::mem::take(&mut preview.plan);
                preview.clear();
                with_store(&store, |s| s.set_plan_pending(false));
                if plan.is_empty() {
                    continue;
                }
                apply_plan_live(&worker, &store, plan);
            }
            AgentRequest::DiscardPlan => {
                if preview.is_pending() {
                    if let Some(saved) = preview.history_restore.take() {
                        history = saved;
                    }
                    preview.clear();
                    push_entry(
                        &store,
                        "status",
                        "Plan discarded — nothing was applied.".into(),
                    );
                }
                with_store(&store, |s| s.set_plan_pending(false));
            }
            AgentRequest::ResetHistory => {
                history.clear();
                preview.clear();
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_one_prompt(
    worker: &WorkerHandle,
    store: &slint::Weak<AgentStore<'static>>,
    sandbox: &mut Option<Engine>,
    preview: &mut Preview,
    history: &mut Vec<Message>,
    prompt: &str,
    context: EditorContext,
    agent_dir: cutlass_ai::AgentDir,
    dry_run: bool,
    cancel: &AtomicBool,
) {
    let config_path = cutlass_settings::default_config_path();
    let section = match cutlass_settings::load(&config_path) {
        Ok(settings) => settings.ai,
        Err(e) => {
            push_entry(store, "error", e);
            return;
        }
    };
    if !section.is_configured() {
        with_store(store, |s| s.set_configured(false));
        push_entry(
            store,
            "error",
            format!(
                "No AI provider configured. Add an endpoint and model in \
                 Settings (or an [ai] table in {}), then send again.",
                config_path.display()
            ),
        );
        return;
    }
    // The third provider mode: "Cutlass account" routes through the
    // backend's OpenAI-compatible managed proxy with the keychain session
    // as the bearer (model pinned server-side; credits metered there).
    let provider = if section.use_account {
        let token = match crate::account::managed_access_token() {
            Ok(token) => token,
            Err(e) => {
                push_entry(store, "error", e);
                return;
            }
        };
        with_store(store, |s| s.set_configured(true));
        OpenAiCompatProvider::new(
            &format!("{}/v1/generate", crate::account::base_url()),
            "cutlass-managed",
            Some(token),
        )
    } else {
        let api_key = match cutlass_ai::config::resolve_api_key(
            section.api_key.as_deref(),
            section.api_key_env.as_deref(),
        ) {
            Ok(key) => key,
            Err(e) => {
                push_entry(store, "error", e);
                return;
            }
        };
        with_store(store, |s| s.set_configured(true));
        OpenAiCompatProvider::new(&section.base_url, &section.model, api_key)
    };

    let sandbox_existed = sandbox.is_some();
    let engine = match sandbox {
        Some(engine) => engine,
        None => match sandbox_engine() {
            Ok(engine) => sandbox.insert(engine),
            Err(e) => {
                push_entry(store, "error", e);
                return;
            }
        },
    };

    let continue_pending = preview.is_pending() && sandbox_existed;
    if !continue_pending {
        let Some(snapshot) = worker.snapshot_project() else {
            push_entry(
                store,
                "error",
                "The editor engine is not responding.".into(),
            );
            return;
        };
        engine.reset_project(snapshot);
        preview.plan.clear();
        preview.descriptions.clear();
    }

    // Compose rules after the snapshot reset so per-project rules read
    // from the project this prompt actually edits (imported projects
    // included — the panel shows them via EditorStore.project.agent-rules).
    let mut sections: Vec<(String, String)> = agent_dir
        .rules
        .into_iter()
        .map(|(stem, text)| (format!("user rule: {stem}"), text))
        .collect();
    let project_rules = engine.project().metadata().agent_rules.clone();
    if !project_rules.trim().is_empty() {
        sections.push(("project rules".into(), project_rules));
    }
    let (rules, truncated) = compose_rules(&sections);
    with_store(store, move |s| s.set_rules_truncated(truncated));
    if truncated {
        push_entry(
            store,
            "status",
            "Rules exceed the size cap and were truncated.".into(),
        );
    }
    let extensions = AgentExtensions {
        rules,
        skills: merge_skills(agent_dir.skills),
    };

    let mut plan: Vec<AgentPlanStep> = preview.plan.clone();
    let mut bridge = SandboxBridge {
        engine,
        plan: &mut plan,
    };
    let event_store = store.clone();
    let mut on_event = move |event: AgentEvent| match event {
        AgentEvent::TextDelta(delta) => append_assistant_text(&event_store, delta),
        AgentEvent::Action(action) => push_entry(&event_store, "action", action.description),
    };

    info!(prompt, dry_run, "agent prompt started");
    let outcome = run_prompt(
        &provider,
        &mut bridge,
        &context,
        &extensions,
        history,
        prompt,
        &AgentConfig::default(),
        cancel,
        &mut on_event,
    );

    match outcome.status {
        PromptStatus::Aborted(reason) => {
            warn!(reason, "agent prompt aborted");
            push_entry(
                store,
                "error",
                if reason == "cancelled" {
                    "Stopped — nothing was applied.".to_string()
                } else if reason.contains("402") {
                    // The managed proxy's out-of-credits answer.
                    "Out of Cutlass credits — buy a pack in Settings > Account. \
                     Nothing was applied."
                        .to_string()
                } else {
                    format!("{reason} — nothing was applied.")
                },
            );
        }
        PromptStatus::Completed | PromptStatus::DryRun => {
            info!(actions = plan.len(), "agent prompt completed");
            history.extend(outcome.turn_messages);
            trim_history(history);
            if dry_run {
                preview.plan = plan;
                preview.descriptions.extend(
                    outcome
                        .actions
                        .iter()
                        .map(|a| SharedString::from(a.description.clone())),
                );
            } else if !plan.is_empty() {
                apply_plan_live(worker, store, plan);
            }
        }
    }

    let pending = preview.is_pending();
    let descriptions = preview.descriptions.clone();
    with_store(store, move |s| {
        if pending {
            s.set_plan_actions(ModelRc::new(VecModel::from(descriptions)));
        }
        s.set_plan_pending(pending);
    });
}

fn apply_plan_live(
    worker: &WorkerHandle,
    store: &slint::Weak<AgentStore<'static>>,
    plan: Vec<AgentPlanStep>,
) {
    let count = plan.len();
    match worker.agent_apply_plan(plan) {
        Some(Ok(())) => {
            push_entry(
                store,
                "applied",
                format!(
                    "Applied {count} edit{} as one undo step.",
                    if count == 1 { "" } else { "s" }
                ),
            );
            with_store(store, |s| s.set_undo_offered(true));
        }
        Some(Err(e)) => {
            error!(error = e, "agent plan replay failed");
            push_entry(
                store,
                "error",
                format!("Could not apply the plan: {e}. Nothing was changed."),
            );
        }
        None => push_entry(
            store,
            "error",
            "The editor engine is not responding.".into(),
        ),
    }
}

const HISTORY_CHAR_BUDGET: usize = 24_000;

fn trim_history(history: &mut Vec<Message>) {
    while history_chars(history) > HISTORY_CHAR_BUDGET {
        let next_turn = history
            .iter()
            .enumerate()
            .skip(1)
            .find(|(_, m)| matches!(m, Message::User { .. }))
            .map(|(i, _)| i);
        match next_turn {
            Some(i) => {
                history.drain(0..i);
            }
            None => break,
        }
    }
}

fn history_chars(history: &[Message]) -> usize {
    history.iter().map(message_chars).sum()
}

fn message_chars(m: &Message) -> usize {
    match m {
        Message::System { content } | Message::User { content } => content.len(),
        Message::Assistant {
            content,
            tool_calls,
        } => {
            content.len()
                + tool_calls
                    .iter()
                    .map(|c| c.name.len() + c.arguments.to_string().len())
                    .sum::<usize>()
        }
        Message::ToolResult { content, .. } => content.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preview_worker::agent_replay;
    use cutlass_ai::wire;
    use cutlass_models::{MediaSource, Project, Rational};

    fn fixture_project() -> (Project, u64) {
        let mut project = Project::new("agent-ui-fixture", Rational::FPS_24);
        let media = project
            .add_media(MediaSource::new(
                "/tmp/agent-ui-fixture.mp4",
                1920,
                1080,
                Rational::FPS_24,
                60 * 24,
                false,
            ))
            .raw();
        (project, media)
    }

    fn temp_engine(project: Project) -> Engine {
        Engine::with_project(EngineConfig::default(), project).expect("engine")
    }

    #[test]
    fn rehearsed_plan_replays_with_id_remapping_and_single_undo() {
        let (project, media) = fixture_project();
        let mut sandbox = temp_engine(project.clone());
        let mut live = temp_engine(project);

        let mut plan: Vec<AgentPlanStep> = Vec::new();
        let mut bridge = SandboxBridge {
            engine: &mut sandbox,
            plan: &mut plan,
        };
        bridge.begin_group();
        let track = match bridge
            .apply(&WireCommand::AddTrack(wire::AddTrack {
                kind: wire::WireTrackKind::Video,
                name: "V1".into(),
                index: None,
            }))
            .expect("add track")
        {
            EditOutcome::CreatedTrack(id) => id.raw(),
            other => panic!("expected created track, got {other:?}"),
        };
        let head = match bridge
            .apply(&WireCommand::AddClip(wire::AddClip {
                track,
                media,
                source_start: 0.0,
                source_duration: 10.0,
                start: 0.0,
            }))
            .expect("add clip")
        {
            EditOutcome::Created(id) => id.raw(),
            other => panic!("expected created clip, got {other:?}"),
        };
        let right = match bridge
            .apply(&WireCommand::SplitClip(wire::SplitClip {
                clip: head,
                at: 4.0,
            }))
            .expect("split clip")
        {
            EditOutcome::Created(id) => id.raw(),
            other => panic!("expected created clip, got {other:?}"),
        };
        bridge
            .apply(&WireCommand::TrimClip(wire::TrimClip {
                clip: right,
                start: 4.0,
                duration: 2.0,
            }))
            .expect("trim clip");
        bridge.end_group();
        assert_eq!(plan.len(), 4);

        agent_replay(&mut live, plan, |_| {}).expect("replay");

        let timeline = live.project().timeline();
        assert_eq!(timeline.track_count(), 1);
        assert_eq!(timeline.clip_count(), 2);

        assert!(live.undo(), "the plan is one undo entry");
        assert_eq!(live.project().timeline().track_count(), 0);
        assert!(!live.undo(), "nothing left to undo");
    }

    #[test]
    fn stale_plan_rolls_back_and_reports() {
        let (project, _media) = fixture_project();
        let mut live = temp_engine(project);

        let plan = vec![AgentPlanStep {
            command: WireCommand::TrimClip(wire::TrimClip {
                clip: 999_999,
                start: 0.0,
                duration: 1.0,
            }),
            created: None,
        }];
        let err = agent_replay(&mut live, plan, |_| {}).expect_err("stale plan must fail");
        assert!(err.contains("step 1/1"), "names the failing step: {err}");
        assert!(!live.undo(), "rollback leaves no history entry");
    }
}
