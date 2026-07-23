use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crossbeam_channel::{Receiver, unbounded};
use cutlass_ai::{
    AgentConfig, AgentEvent, AgentExtensions, EditorContext, Message, PromptStatus, compose_rules,
    expand_slash_command, load_agent_dir, merge_skills, run_prompt_with_host,
};
use cutlass_engine::Engine;
use cutlass_jobs::JobManager;
use slint::{ModelRc, SharedString, VecModel};
use tracing::{error, info, warn};

use crate::agent_senses::AgentSenses;
use crate::agent_session::AgentSession;
use crate::cache_registry::CacheRegistry;
use crate::preview_worker::WorkerHandle;
use crate::{AgentStore, AppWindow};

use super::sandbox::{SandboxBridge, sandbox_engine};
use super::token_usage::format_usage_line;
use super::tool_host::{DesktopToolHandles, DesktopToolHost, abort_status_message};
use super::transcript::{
    append_assistant_text, append_reasoning_text, attach_usage_line, persist_session,
    publish_chat_list, push_entry, push_image_entry, replace_transcript, with_store,
};
use super::types::{
    AgentHandle, AgentPlanStep, AgentRequest, AgentRuntimeHandles, AgentWorker, ApprovalDecision,
};

impl AgentWorker {
    pub fn spawn(
        worker: WorkerHandle,
        store: slint::Weak<AgentStore<'static>>,
        app: slint::Weak<AppWindow>,
        cache_registry: CacheRegistry,
        job_manager: JobManager,
    ) -> Result<Self, String> {
        let (tx, rx) = unbounded();
        let (approval_tx, approval_rx) = unbounded();
        let cancel = Arc::new(AtomicBool::new(false));
        let thread_cancel = cancel.clone();
        let pending_approval_id = Arc::new(AtomicU64::new(0));
        let thread_pending_approval_id = pending_approval_id.clone();
        let approval_id_allocator = Arc::new(AtomicU64::new(0));
        let join = std::thread::Builder::new()
            .name("cutlass-agent".into())
            .spawn(move || {
                agent_main(
                    worker,
                    AgentRuntimeHandles {
                        store,
                        app,
                        cache_registry,
                        job_manager,
                    },
                    rx,
                    thread_cancel,
                    approval_rx,
                    thread_pending_approval_id,
                    approval_id_allocator,
                )
            })
            .map_err(|e| e.to_string())?;
        Ok(Self {
            handle: AgentHandle {
                tx,
                cancel,
                approval_tx,
                pending_approval_id,
            },
            _join: join,
        })
    }

    pub fn handle(&self) -> AgentHandle {
        self.handle.clone()
    }
}

#[derive(Default)]
pub(crate) struct Preview {
    pub(crate) plan: Vec<AgentPlanStep>,
    /// Phase boundaries within `plan` (exclusive step indices) from
    /// `commit_progress` — live replay commits one undo group per phase.
    pub(crate) phase_breaks: Vec<usize>,
    pub(crate) descriptions: Vec<SharedString>,
    pub(crate) history_restore: Option<Vec<Message>>,
}

impl Preview {
    fn is_pending(&self) -> bool {
        !self.plan.is_empty()
    }

    fn clear(&mut self) {
        self.plan.clear();
        self.phase_breaks.clear();
        self.descriptions.clear();
        self.history_restore = None;
    }
}

pub(crate) fn agent_main(
    worker: WorkerHandle,
    runtime: AgentRuntimeHandles,
    rx: Receiver<AgentRequest>,
    cancel: Arc<AtomicBool>,
    approval_rx: Receiver<ApprovalDecision>,
    pending_approval_id: Arc<AtomicU64>,
    approval_id_allocator: Arc<AtomicU64>,
) {
    let store = runtime.store.clone();
    let mut sandbox: Option<Engine> = None;
    let mut senses = AgentSenses::new();
    let mut preview = Preview::default();
    let mut history: Vec<Message> = Vec::new();
    let mut current_project: Option<PathBuf> = None;
    let mut current_chat_id: Option<String> = None;

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
                if current_chat_id.is_none() {
                    current_chat_id = current_project.as_deref().and_then(|project| {
                        match crate::agent_session::allocate_chat_id(project) {
                            Ok(id) => Some(id),
                            Err(error) => {
                                warn!(error, project = %project.display(), "agent chat id was not allocated");
                                None
                            }
                        }
                    });
                    publish_chat_list(
                        &store,
                        current_project.as_deref(),
                        current_chat_id.as_deref(),
                    );
                }
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
                    &runtime,
                    &mut sandbox,
                    &mut senses,
                    &mut preview,
                    &mut history,
                    &sent,
                    context,
                    agent_dir,
                    dry_run,
                    &cancel,
                    &approval_rx,
                    &pending_approval_id,
                    &approval_id_allocator,
                );

                with_store(&store, |s| s.set_running(false));
                persist_session(
                    current_project.as_deref(),
                    current_chat_id.as_deref(),
                    &history,
                    &store,
                );
                publish_chat_list(
                    &store,
                    current_project.as_deref(),
                    current_chat_id.as_deref(),
                );
            }
            AgentRequest::ApplyPlan => {
                let plan = std::mem::take(&mut preview.plan);
                let phase_breaks = std::mem::take(&mut preview.phase_breaks);
                preview.clear();
                with_store(&store, |s| s.set_plan_pending(false));
                if plan.is_empty() {
                    continue;
                }
                apply_plan_live(&worker, &store, plan, &phase_breaks);
                persist_session(
                    current_project.as_deref(),
                    current_chat_id.as_deref(),
                    &history,
                    &store,
                );
                publish_chat_list(
                    &store,
                    current_project.as_deref(),
                    current_chat_id.as_deref(),
                );
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
                persist_session(
                    current_project.as_deref(),
                    current_chat_id.as_deref(),
                    &history,
                    &store,
                );
                publish_chat_list(
                    &store,
                    current_project.as_deref(),
                    current_chat_id.as_deref(),
                );
            }
            AgentRequest::NewChat => {
                if let Some(saved) = preview.history_restore.take() {
                    history = saved;
                }
                preview.clear();
                with_store(&store, |s| s.set_plan_pending(false));
                persist_session(
                    current_project.as_deref(),
                    current_chat_id.as_deref(),
                    &history,
                    &store,
                );

                current_chat_id = current_project.as_deref().and_then(|project| {
                    match crate::agent_session::allocate_chat_id(project) {
                        Ok(id) => Some(id),
                        Err(error) => {
                            warn!(error, project = %project.display(), "agent chat id was not allocated");
                            None
                        }
                    }
                });
                history.clear();
                replace_transcript(&store, Vec::new(), None);
                publish_chat_list(
                    &store,
                    current_project.as_deref(),
                    current_chat_id.as_deref(),
                );
            }
            AgentRequest::SelectChat { id } => {
                if current_chat_id.as_deref() == Some(id.as_str()) {
                    publish_chat_list(
                        &store,
                        current_project.as_deref(),
                        current_chat_id.as_deref(),
                    );
                    continue;
                }
                if let Some(saved) = preview.history_restore.take() {
                    history = saved;
                }
                preview.clear();
                with_store(&store, |s| s.set_plan_pending(false));
                persist_session(
                    current_project.as_deref(),
                    current_chat_id.as_deref(),
                    &history,
                    &store,
                );

                let loaded = current_project
                    .as_deref()
                    .ok_or_else(|| "no project is open".to_string())
                    .and_then(|project| crate::agent_session::load_chat(project, &id));
                match loaded {
                    Ok(session) => {
                        history = session.history;
                        replace_transcript(&store, session.transcript, None);
                        current_chat_id = Some(id);
                    }
                    Err(error) => {
                        warn!(error, chat_id = id, "agent chat could not be restored");
                        push_entry(
                            &store,
                            "error",
                            "That chat could not be restored.".to_string(),
                        );
                    }
                }
                publish_chat_list(
                    &store,
                    current_project.as_deref(),
                    current_chat_id.as_deref(),
                );
            }
            AgentRequest::SwitchProject { path } => {
                // A parked dry-run conversation names edits that never
                // landed. Restore the history checkpoint before persisting
                // the draft we are leaving.
                if let Some(saved) = preview.history_restore.take() {
                    history = saved;
                }
                preview.clear();
                with_store(&store, |s| s.set_plan_pending(false));
                persist_session(
                    current_project.as_deref(),
                    current_chat_id.as_deref(),
                    &history,
                    &store,
                );

                let (next_chat_id, session, restore_error) = match path.as_deref() {
                    Some(project) => match crate::agent_session::list_chats(project) {
                        Ok(chats) => match chats.first() {
                            Some(chat) => {
                                match crate::agent_session::load_chat(project, &chat.id) {
                                    Ok(session) => (Some(chat.id.clone()), session, None),
                                    Err(error) => (
                                        Some(chat.id.clone()),
                                        AgentSession::default(),
                                        Some(error),
                                    ),
                                }
                            }
                            None => match crate::agent_session::allocate_chat_id(project) {
                                Ok(id) => (Some(id), AgentSession::default(), None),
                                Err(error) => (None, AgentSession::default(), Some(error)),
                            },
                        },
                        Err(error) => match crate::agent_session::allocate_chat_id(project) {
                            Ok(id) => (Some(id), AgentSession::default(), Some(error)),
                            Err(allocation_error) => (
                                None,
                                AgentSession::default(),
                                Some(format!("{error}; {allocation_error}")),
                            ),
                        },
                    },
                    None => (None, AgentSession::default(), None),
                };
                history = session.history;
                replace_transcript(&store, session.transcript, restore_error);
                current_project = path;
                current_chat_id = next_chat_id;
                publish_chat_list(
                    &store,
                    current_project.as_deref(),
                    current_chat_id.as_deref(),
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_one_prompt(
    worker: &WorkerHandle,
    runtime: &AgentRuntimeHandles,
    sandbox: &mut Option<Engine>,
    senses: &mut AgentSenses,
    preview: &mut Preview,
    history: &mut Vec<Message>,
    prompt: &str,
    context: EditorContext,
    agent_dir: cutlass_ai::AgentDir,
    dry_run: bool,
    cancel: &AtomicBool,
    approval_rx: &Receiver<ApprovalDecision>,
    pending_approval_id: &Arc<AtomicU64>,
    approval_id_allocator: &Arc<AtomicU64>,
) {
    let store = &runtime.store;
    let config_path = cutlass_settings::default_config_path();
    let section = match cutlass_settings::load(&config_path) {
        Ok(settings) => settings.ai,
        Err(e) => {
            push_entry(store, "error", e);
            return;
        }
    };
    let provider = match cutlass_ai::config::provider_from_ai(&section) {
        Ok(provider) => provider,
        Err(e) => {
            // The settings may look structurally complete, but no provider can
            // be built from them — every prompt would error, so surface the
            // unconfigured state instead of a chat box that can't work.
            with_store(store, |s| s.set_configured(false));
            push_entry(
                store,
                "error",
                format!(
                    "{e} Configure AI in Settings (or edit {}).",
                    config_path.display()
                ),
            );
            return;
        }
    };
    with_store(store, |s| s.set_configured(true));

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
        preview.phase_breaks.clear();
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

    // A continued preview accumulates steps; the outcome's phase breaks
    // index this prompt's actions only, so offset them into the plan.
    let plan_base = preview.plan.len();
    let mut plan: Vec<AgentPlanStep> = preview.plan.clone();
    let mut bridge = SandboxBridge {
        worker,
        engine,
        plan: &mut plan,
        senses,
        default_playhead_seconds: context.playhead_seconds,
    };
    let mut tool_host = DesktopToolHost::new(
        section.autonomy,
        DesktopToolHandles::from_runtime(runtime, worker),
        approval_rx.clone(),
        pending_approval_id.clone(),
        approval_id_allocator.clone(),
    );
    let event_store = store.clone();
    let mut on_event = move |event: AgentEvent| match event {
        AgentEvent::TextDelta(delta) => append_assistant_text(&event_store, delta),
        AgentEvent::ReasoningDelta(delta) => append_reasoning_text(&event_store, delta),
        AgentEvent::Action(action) => push_entry(&event_store, "action", action.description),
        AgentEvent::HostAction { name, summary } => {
            push_entry(&event_store, "action", format!("{name}: {summary}"))
        }
        AgentEvent::Image(image) => push_image_entry(&event_store, image),
        // No live status string on the store; the finished line lands on the
        // exchange when the prompt completes (success or abort).
        AgentEvent::Usage(usage) => info!(
            input = usage.input_tokens,
            cached = usage.cached_input_tokens,
            output = usage.output_tokens,
            cost = ?usage.cost,
            "agent token usage"
        ),
    };

    info!(prompt, dry_run, "agent prompt started");
    let outcome = run_prompt_with_host(
        &provider,
        &mut bridge,
        &mut tool_host,
        &context,
        &extensions,
        history,
        prompt,
        &AgentConfig::default(),
        cancel,
        &mut on_event,
    );
    let ordinary_host_call_attempted = tool_host.ordinary_host_call_attempted();

    match outcome.status {
        PromptStatus::Aborted(reason) => {
            warn!(reason, "agent prompt aborted");
            push_entry(
                store,
                "error",
                abort_status_message(&reason, ordinary_host_call_attempted),
            );
        }
        PromptStatus::Completed | PromptStatus::DryRun => {
            info!(actions = plan.len(), "agent prompt completed");
            history.extend(outcome.turn_messages);
            trim_history(history);
            if dry_run {
                preview.plan = plan;
                preview
                    .phase_breaks
                    .extend(outcome.phase_breaks.iter().map(|b| plan_base + b));
                preview.descriptions.extend(
                    outcome
                        .actions
                        .iter()
                        .map(|a| SharedString::from(a.description.clone())),
                );
            } else if !plan.is_empty() {
                // Auto-apply never extends a parked preview (any pending one
                // was discarded above), so the breaks are plan-relative.
                apply_plan_live(worker, store, plan, &outcome.phase_breaks);
            }
        }
    }

    if let Some(line) = format_usage_line(&outcome.usage) {
        attach_usage_line(store, line);
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

/// Split a rehearsed plan at its phase breaks (exclusive step indices,
/// strictly increasing). The remainder past the last break is the final
/// phase; a break flush with the plan's end leaves no empty group behind.
pub(crate) fn split_plan_phases(
    mut plan: Vec<AgentPlanStep>,
    breaks: &[usize],
) -> Vec<Vec<AgentPlanStep>> {
    let mut phases = Vec::with_capacity(breaks.len() + 1);
    for &at in breaks.iter().rev() {
        if at < plan.len() {
            phases.push(plan.split_off(at));
        }
    }
    if !plan.is_empty() {
        phases.push(plan);
    }
    phases.reverse();
    phases
}

pub(crate) fn apply_plan_live(
    worker: &WorkerHandle,
    store: &slint::Weak<AgentStore<'static>>,
    plan: Vec<AgentPlanStep>,
    phase_breaks: &[usize],
) {
    let count = plan.len();
    let phases = split_plan_phases(plan, phase_breaks);
    let phase_count = phases.len();
    match worker.agent_apply_plan(phases) {
        Some(Ok(())) => {
            push_entry(
                store,
                "applied",
                if phase_count > 1 {
                    format!("Applied {count} edits in {phase_count} undo steps.")
                } else {
                    format!(
                        "Applied {count} edit{} as one undo step.",
                        if count == 1 { "" } else { "s" }
                    )
                },
            );
            with_store(store, |s| s.set_undo_offered(true));
        }
        Some(Err(e)) => {
            error!(error = e, "agent plan replay failed");
            // The replay error already says how much (if anything) landed.
            push_entry(store, "error", format!("Could not apply the plan: {e}."));
        }
        None => push_entry(
            store,
            "error",
            "The editor engine is not responding.".into(),
        ),
    }
}

pub(crate) const HISTORY_CHAR_BUDGET: usize = 24_000;

pub(crate) fn trim_history(history: &mut Vec<Message>) {
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

pub(crate) fn history_chars(history: &[Message]) -> usize {
    history.iter().map(message_chars).sum()
}

pub(crate) fn message_chars(m: &Message) -> usize {
    match m {
        Message::System { content } => content.len(),
        // History is text-only (turns strip images to labeled placeholders
        // before they land here), but count any payload that does appear so
        // the budget can never be blown by raw image bytes.
        Message::User { content, images } => {
            content.len() + images.iter().map(|i| i.data.len()).sum::<usize>()
        }
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
        Message::ToolResult {
            content, images, ..
        } => content.len() + images.iter().map(|i| i.data.len()).sum::<usize>(),
    }
}
