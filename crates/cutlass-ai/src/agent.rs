//! The agent loop: prompt in, validated-and-applied command group out,
//! every step observable.
//!
//! The loop's whole world is the [`EngineBridge`] — it cannot name a file
//! path, a socket, or a UI type. One prompt = one history group: the
//! bridge's group markers wrap the run, failed individual commands are
//! reported back to the model (which may correct course), and the group
//! rolls back only when the prompt aborts (cancellation, provider error,
//! turn or host-call cap exceeded). Reaching the edit cap is gentler:
//! further edits are refused and the run ends keeping everything already
//! applied. In dry-run mode nothing is applied; the validated plan comes
//! back for the UI's preview card.
//!
//! Beyond edits, the embedder can wire a [`ToolHost`] of app tools, while
//! the bridge can expose strictly read-only senses of its exact project
//! state. The latter is what lets a model inspect edits inside rehearsal
//! without accidentally looking at the untouched live project. Both are
//! dispatched by exact name and charged against the host-call cap. The
//! built-in `commit_progress` tool records phase breaks so a long task's
//! live replay can land as several undo steps
//! ([`PromptOutcome::phase_breaks`]).

use std::collections::HashSet;
use std::sync::atomic::AtomicBool;

use cutlass_commands::EditOutcome;

use crate::describe::{EditorContext, ProjectSummary};
use crate::extend::AgentExtensions;
use crate::provider::{
    ChatProvider, ChatRequest, FinishReason, ImagePart, Message, ProviderError, ProviderStreamEvent,
};
use crate::tools::{HostToolSpec, ToolHost, is_host_tool_name};
use crate::wire::{self, WireCommand};

/// The loop's only view of the engine. The UI implements this over a
/// sandbox engine whose validated plan replays onto the live one
/// (`cutlass-ui/src/agent.rs`); tests implement it over a plain `Engine`.
pub trait EngineBridge {
    /// Fresh summary of the project as it stands.
    fn summary(&mut self) -> ProjectSummary;
    /// Read-only tools that inspect this exact engine state. Unlike ordinary
    /// host tools, these travel with the sandbox bridge so a screenshot taken
    /// after an edit observes the rehearsed project, not the live project.
    ///
    /// The loop accepts only [`crate::tools::ToolTier::ReadOnly`] specs here.
    fn sense_tools(&self) -> Vec<HostToolSpec> {
        Vec::new()
    }
    /// Execute one tool previously returned by [`EngineBridge::sense_tools`].
    /// Implementations must not mutate project state.
    fn sense(
        &mut self,
        name: &str,
        _arguments: &serde_json::Value,
        _cancel: &AtomicBool,
    ) -> Result<crate::tools::ToolOutput, String> {
        Err(format!("unknown engine sense '{name}'"))
    }
    /// Prepare for an ordinary registered [`ToolHost`] call.
    ///
    /// The loop invokes this after charging the host-call cap, but before
    /// authorization or dispatch. Returning `Err` rejects the call without
    /// invoking either [`ToolHost::authorize`], [`ToolHost::call`], or
    /// [`EngineBridge::after_host_call`]. Bridge-owned read-only senses do
    /// not pass through this hook.
    fn before_host_call(
        &mut self,
        _name: &str,
        _arguments: &serde_json::Value,
    ) -> Result<(), String> {
        Ok(())
    }
    /// Reconcile bridge state after an ordinary host dispatch was attempted.
    ///
    /// The loop invokes this exactly once after [`ToolHost::call`] returns,
    /// for both success and failure, and before exposing that result to the
    /// model. Authorization failures and pre-call rejections do not invoke
    /// it. `result` borrows the host result so implementations can inspect
    /// success or failure without cloning [`crate::tools::ToolOutput`].
    ///
    /// Host calls may have partial side effects even when they return `Err`.
    /// This hook is therefore the bridge's reconciliation boundary. A hook
    /// failure aborts the prompt and rolls back its sandbox edit group, but
    /// cannot promise to undo effects the host already performed.
    fn after_host_call(
        &mut self,
        _name: &str,
        _arguments: &serde_json::Value,
        _result: Result<&crate::tools::ToolOutput, &str>,
    ) -> Result<(), String> {
        Ok(())
    }
    /// Validate + apply one wire command. `Err` is a model-readable reason
    /// (validation rejection or engine error); state is unchanged on `Err`.
    fn apply(&mut self, command: &WireCommand) -> Result<EditOutcome, String>;
    /// Validate only — the dry-run path. State must not change.
    fn check(&mut self, command: &WireCommand) -> Result<(), String>;
    fn begin_group(&mut self);
    fn end_group(&mut self);
    fn rollback_group(&mut self);
}

/// Guardrail knobs.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Hard cap on edit-tool calls per prompt (the runaway-loop fuse).
    /// Reaching it does not fail the prompt: over-cap edits are refused
    /// and the run completes keeping the edits already applied.
    pub max_tool_calls: usize,
    /// Hard cap on host-tool calls per prompt. A separate fuse: senses
    /// and app control must not starve editing, nor the reverse.
    pub max_host_calls: usize,
    /// Hard cap on provider turns per prompt.
    pub max_turns: usize,
    /// Hard cap on images carried by one request, newest kept. Screenshot
    /// tools bound each image's dimensions, so count × bounded size caps
    /// the whole vision payload.
    pub max_images: usize,
    /// Hard cap on total encoded image bytes carried by one request. This
    /// protects the provider boundary even when an extensible host tool does
    /// not honor the normal screenshot dimension limits.
    pub max_image_bytes: usize,
    /// Validate and collect the plan without applying anything.
    pub dry_run: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_tool_calls: 1000,
            max_host_calls: 200,
            max_turns: 200,
            max_images: 25,
            max_image_bytes: 24 * 1024 * 1024,
            dry_run: false,
        }
    }
}

/// One command the agent ran (or, in dry-run, plans to run).
#[derive(Debug, Clone, PartialEq)]
pub struct ActionLogEntry {
    pub command: WireCommand,
    /// Human-readable line for the transcript / undo tooltip / eval
    /// assertions, e.g. `split clip 7 at 12.40s (new clip 21)`.
    pub description: String,
}

/// Streamed progress for the chat panel.
#[derive(Debug, Clone, PartialEq)]
pub enum AgentEvent {
    /// Assistant text, as it streams.
    TextDelta(String),
    /// A provider-generated reasoning summary, kept out of model history.
    ReasoningDelta(String),
    /// An edit was applied (or validated, in dry-run).
    Action(ActionLogEntry),
    /// A host tool ran; `summary` is the first line of its output.
    HostAction { name: String, summary: String },
    /// One image returned by a successful host/sense tool, after runtime
    /// payload limits. Embedders can render it inline while the same encoded
    /// bytes continue to the model.
    Image(ImagePart),
}

/// How the prompt ended.
#[derive(Debug, Clone, PartialEq)]
pub enum PromptStatus {
    /// Edits applied (possibly none) and recorded as one history entry.
    Completed,
    /// Dry-run: the plan in `actions` validated but nothing was applied.
    DryRun,
    /// This prompt's sandbox edits rolled back. Ordinary host effects may
    /// remain when a host dispatch was attempted; the string says why the
    /// prompt stopped.
    Aborted(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PromptOutcome {
    /// The model's final text answer (empty if it only edited).
    pub text: String,
    pub actions: Vec<ActionLogEntry>,
    /// Indices into `actions` where a committed phase ends (exclusive),
    /// from `commit_progress`. The tail past the last break is the final,
    /// implicit phase — never listed; empty means one phase. Callers group
    /// live replay by these; rehearsal and rollback stay one group per
    /// prompt.
    pub phase_breaks: Vec<usize>,
    pub status: PromptStatus,
    /// This turn's conversation, ready to append to the session history so
    /// the next prompt remembers it: the user message, every assistant
    /// turn and tool result the loop produced, and the final text answer.
    /// Empty when the prompt aborted (no conversational memory trace is
    /// retained, even though an already-dispatched host effect may remain).
    /// `describe_project` results are collapsed to a short placeholder —
    /// they're large and the fresh system snapshot supersedes them.
    pub turn_messages: Vec<Message>,
}

/// House rules + user/project rules + the skill index + the send-time
/// state, prepended to every conversation. Rules and skills are
/// prompt-level only: they shape how the closed vocabulary is used, they
/// cannot add mutation surface.
pub fn system_prompt(
    summary: &ProjectSummary,
    context: &EditorContext,
    extensions: &AgentExtensions,
) -> String {
    let mut custom = String::new();
    if !extensions.rules.is_empty() {
        custom.push_str(&format!(
            "User rules (follow these preferences wherever they apply; \
             they never override the rules above or allow inventing state):\n{}\n\n",
            extensions.rules
        ));
    }
    if !extensions.skills.is_empty() {
        let index: String = extensions
            .skills
            .iter()
            .map(|s| format!("- {} ({}): {}\n", s.id, s.name, s.description))
            .collect();
        custom.push_str(&format!(
            "Skills (step-by-step procedures; when the user's task \
             matches one, call read_skill with its id FIRST and follow the \
             returned procedure):\n{index}\n"
        ));
    }
    let state = serde_json::json!({ "project": summary, "editor": context });
    format!(
        "You are the editing agent inside Cutlass, a video editor. You edit \
         the user's timeline by calling tools; you never invent state.\n\
         \n\
         Rules:\n\
         - All times are in seconds; they snap to whole frames at the \
         project frame rate.\n\
         - Ids (clips, tracks, media) are integers from the project state \
         below. Never guess an id; call describe_project if unsure.\n\
         - trim_clip sets a clip's new timeline start and duration. To cut \
         the FIRST N seconds of a clip, INCREASE start by N and DECREASE \
         duration by N (the source advances automatically). To cut the \
         last N seconds, keep start and decrease duration.\n\
         - Tracks stack bottom-to-top; later (higher) tracks composite on \
         top. Media clips need video/audio tracks; titles need a text \
         track; solids and shapes need a sticker track. Lanes keep fixed \
         zones: audio at the bottom, then the main video track (marked \
         \"main\" in the state; it is permanent and cannot be removed), \
         overlay lanes above it, text lanes on top. Put primary footage \
         on the main track and prefer reusing existing lanes over adding \
         new ones.\n\
         - Imported media in the project state's media pool is ready to \
         use even when no timeline clip references it. add_clip is the \
         operation that places media-pool footage on the timeline. An \
         empty timeline is a starting point, not a missing capability: \
         reuse a compatible track when one exists, or call add_track first \
         (the first video track becomes main), read the returned track id, \
         then add clips. For open-ended creative work, inspect the media \
         and choose the sequence, source ranges, placements, and timing \
         yourself. Never ask the user to pre-place footage or claim that \
         media-pool placement is unsupported.\n\
         - If a tool call is rejected, read the error and correct course; \
         do not repeat the identical call.\n\
         - The state below is a fresh snapshot of the project as it is \
         now: it already reflects every edit applied so far, including \
         ones made earlier in this conversation. Trust it over anything \
         said earlier; use the conversation only to understand what the \
         user is referring to.\n\
         - describe_project returns this same state, kept current as you \
         edit. When the user only asks a question, answer directly from \
         the state below — do not call describe_project first. But once \
         you have applied edits that move, resize, split, add, or remove \
         clips this turn, call describe_project to read the new positions \
         and ids before any further edit that depends on them: recompute, \
         do not assume, and do not give up. Name clips and tracks by id \
         and content so answers stay checkable; if the state cannot \
         answer a question, say what is missing instead of guessing. \
         Unknown source-footage content is not missing project state: \
         when media inspection tools are available, use them instead of \
         declining the task or asking the user to place footage first.\n\
         - Clips on one track can never overlap, and a clip can only grow \
         into free space. To lengthen a clip or insert into a packed \
         track, first make room: move or shift the later clips on that \
         track to the right (shift_clips, move_clip, or ripple_insert), \
         then resize. If a tool call is rejected for an overlap or for \
         exceeding the source media, read the error, re-inspect the \
         current state, and adjust the plan — never abandon the task for \
         lack of state you can fetch.\n\
         \n\
         {custom}\
         Current state (the user's selection and playhead are in \
         'editor'):\n{state}"
    )
}

/// Appended to the system message only when the embedder wires host
/// tools; `system_prompt` itself stays host-agnostic (its signature and
/// output are relied on by other callers and tests).
const HOST_TOOLS_RULES: &str = "\n\nHost tools: tools named {namespace}_{tool} (app_…, media_…, python_…) \
     reach the surrounding application rather than the timeline. Read-only \
     and workspace tools run immediately. Tools with system-wide effects may \
     pause for the user's confirmation and can be declined. Treat a decline \
     as an instruction to change course, not an error to retry.";

fn engine_sense_rules(specs: &[HostToolSpec]) -> String {
    let names = specs
        .iter()
        .map(|spec| spec.name.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "\n\nRehearsal senses ({names}) inspect the complete current project snapshot and sandbox, \
         including edits already completed in this prompt. Source footage is available through \
         these senses: do not claim that you cannot browse or verify it without attempting the \
         relevant sense. Open-ended creative requests such as freestyle edits, montages, or \
         \"make something interesting\" are fully actionable. Survey the media pool with \
         media_pool_sheet when it is listed, inspect promising sources with media_asset_strip, \
         then use visual evidence and editorial judgment to make concrete edits rather than \
         declining or asking the user to choose placements and ranges. Media-pool sources need no \
         pre-placement: when the timeline is empty, create the required tracks with add_track and \
         build the sequence with add_clip. Before finalizing visual or timing work, use the \
         cheapest relevant sense to verify it: prefer a schematic timeline map for placement and \
         timing, and a composited preview frame only when appearance or layering matters. Never \
         claim a check succeeded if a sense failed."
    )
}

/// The phase marker (a loop concern, not a wire command): lets a long
/// task land as several undo steps instead of one monolith.
fn commit_progress_spec() -> wire::ToolSpec {
    wire::ToolSpec {
        name: "commit_progress".into(),
        description: "Mark the edits so far as one completed phase so they land as \
                      their own undo step. Call this between logical stages of a \
                      long task; costs nothing."
            .into(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {},
        }),
    }
}

/// Run one prompt with only the validated edit vocabulary.
///
/// Kept as the compatibility/default entry point for embedders that do not
/// expose application tools. Use [`run_prompt_with_host`] to add senses,
/// app control, jobs, or other namespaced capabilities.
#[allow(clippy::too_many_arguments)]
pub fn run_prompt(
    provider: &dyn ChatProvider,
    bridge: &mut dyn EngineBridge,
    context: &EditorContext,
    extensions: &AgentExtensions,
    history: &[Message],
    prompt: &str,
    config: &AgentConfig,
    cancel: &AtomicBool,
    on_event: &mut dyn FnMut(AgentEvent),
) -> PromptOutcome {
    run_prompt_with_host(
        provider,
        bridge,
        &mut crate::tools::NullToolHost,
        context,
        extensions,
        history,
        prompt,
        config,
        cancel,
        on_event,
    )
}

/// Run one prompt to completion against `bridge` and `host`.
///
/// `context` is the send-time editor snapshot (selection, playhead);
/// `history` is the prior conversation in this session (the caller's
/// accumulated `turn_messages`, with no system message — a fresh one is
/// regenerated here so the current project state always wins); `host` is
/// the embedder's tool surface (pass [`crate::tools::NullToolHost`] when
/// there is none); `on_event` receives streamed text and applied actions
/// for the UI. The returned [`PromptOutcome::turn_messages`] is this
/// turn's contribution to append.
#[allow(clippy::too_many_arguments)]
pub fn run_prompt_with_host(
    provider: &dyn ChatProvider,
    bridge: &mut dyn EngineBridge,
    host: &mut dyn ToolHost,
    context: &EditorContext,
    extensions: &AgentExtensions,
    history: &[Message],
    prompt: &str,
    config: &AgentConfig,
    cancel: &AtomicBool,
    on_event: &mut dyn FnMut(AgentEvent),
) -> PromptOutcome {
    let summary = bridge.summary();
    let mut tools = wire::tool_specs();
    tools.push(wire::describe_project_spec());
    if !extensions.skills.is_empty() {
        tools.push(crate::extend::read_skill_spec());
    }
    tools.push(commit_progress_spec());
    // Built-in names always win: a colliding host spec is dropped here —
    // never sent, never dispatched — so a host can neither shadow the edit
    // vocabulary nor the loop's own tools. (`read_skill` stays reserved
    // even when no skills are loaded.)
    let mut seen_sense_names = HashSet::new();
    let sense_specs: Vec<HostToolSpec> = bridge
        .sense_tools()
        .into_iter()
        .filter(|spec| {
            is_host_tool_name(&spec.name) && spec.tier == crate::tools::ToolTier::ReadOnly
        })
        .filter(|spec| spec.name != "read_skill" && tools.iter().all(|t| t.name != spec.name))
        .filter(|spec| seen_sense_names.insert(spec.name.clone()))
        .collect();
    tools.extend(sense_specs.iter().map(|spec| wire::ToolSpec {
        name: spec.name.clone(),
        description: spec.description.clone(),
        parameters: spec.parameters.clone(),
    }));

    let mut seen_host_names = HashSet::new();
    let host_specs: Vec<HostToolSpec> = host
        .tools()
        .into_iter()
        .filter(|spec| is_host_tool_name(&spec.name))
        .filter(|spec| tools.iter().all(|tool| tool.name != spec.name))
        .filter(|spec| seen_host_names.insert(spec.name.clone()))
        .collect();
    tools.extend(host_specs.iter().map(|spec| wire::ToolSpec {
        name: spec.name.clone(),
        description: spec.description.clone(),
        parameters: spec.parameters.clone(),
    }));

    let mut system = system_prompt(&summary, context, extensions);
    if !sense_specs.is_empty() || !host_specs.is_empty() {
        system.push_str(HOST_TOOLS_RULES);
    }
    if !sense_specs.is_empty() {
        system.push_str(&engine_sense_rules(&sense_specs));
    }
    let mut messages = Vec::with_capacity(history.len() + 2);
    messages.push(Message::System { content: system });
    messages.extend_from_slice(history);
    // This turn's own messages start here (the user prompt and everything
    // the loop appends), kept so we can hand them back as `turn_messages`.
    let turn_start = messages.len();
    messages.push(Message::user(prompt));

    let mut actions: Vec<ActionLogEntry> = Vec::new();
    let mut phase_breaks: Vec<usize> = Vec::new();
    let mut edit_calls = 0usize;
    let mut host_calls = 0usize;
    // Set when an edit call lands past `max_tool_calls`: the run ends after
    // the current turn's calls are answered, keeping everything applied,
    // rather than burning turns until the turn cap rolls the prompt back.
    let mut edit_cap_tripped = false;
    let mut final_text = String::new();
    // The first image-bearing tool result is appended after the current user
    // message. Images are surfaced only after the whole request budget has run,
    // immediately before those exact attachments are sent to the provider.
    let mut image_event_cursor = messages.len();
    // Call ids of `describe_project` results, collapsed in `turn_messages`
    // so the session history never carries a full stale project blob.
    let mut describe_call_ids: Vec<String> = Vec::new();

    if !config.dry_run {
        bridge.begin_group();
    }
    let abort = |bridge: &mut dyn EngineBridge, actions: Vec<ActionLogEntry>, reason: String| {
        if !config.dry_run {
            bridge.rollback_group();
        }
        PromptOutcome {
            text: String::new(),
            actions,
            // Rolled back ⇒ no phases survive to group.
            phase_breaks: Vec::new(),
            status: PromptStatus::Aborted(reason),
            turn_messages: Vec::new(),
        }
    };

    for _turn in 0..config.max_turns {
        enforce_image_budget(&mut messages, config.max_images, config.max_image_bytes);
        for message in messages.iter().skip(image_event_cursor) {
            if let Message::ToolResult { images, .. } = message {
                for image in images {
                    on_event(AgentEvent::Image(image.clone()));
                }
            }
        }
        image_event_cursor = messages.len();
        let turn = {
            let mut forward = |event: ProviderStreamEvent<'_>| match event {
                ProviderStreamEvent::TextDelta(delta) => {
                    on_event(AgentEvent::TextDelta(delta.to_string()));
                }
                ProviderStreamEvent::ReasoningSummaryDelta(delta) => {
                    on_event(AgentEvent::ReasoningDelta(delta.to_string()));
                }
            };
            match provider.chat(
                &ChatRequest {
                    messages: &messages,
                    tools: &tools,
                },
                cancel,
                &mut forward,
            ) {
                Ok(turn) => turn,
                Err(ProviderError::Cancelled) => {
                    return abort(bridge, actions, "cancelled".to_string());
                }
                Err(e) => return abort(bridge, actions, e.to_string()),
            }
        };

        if turn.tool_calls.is_empty() {
            final_text = turn.text;
            if turn.finish == FinishReason::Length {
                return abort(
                    bridge,
                    actions,
                    "the model ran out of tokens mid-answer".to_string(),
                );
            }
            break;
        }

        let tool_calls = turn.tool_calls.clone();
        messages.push(Message::Assistant {
            content: turn.text,
            tool_calls: turn.tool_calls,
        });

        for call in tool_calls {
            // Only host successes attach images; every other path is text.
            let mut images: Vec<ImagePart> = Vec::new();
            let result: String = if call.name == "describe_project" {
                describe_call_ids.push(call.id.clone());
                let state = serde_json::json!({
                    "project": bridge.summary(),
                    "editor": context,
                });
                state.to_string()
            } else if call.name == "read_skill" && !extensions.skills.is_empty() {
                // Read-only like describe_project: answered from the
                // preloaded skill set, no dispatch, no edit-cap charge.
                read_skill_result(&extensions.skills, &call.arguments)
            } else if call.name == "commit_progress" {
                // Free (charges neither cap): marking a phase must never
                // compete with the work it delimits.
                let committed = phase_breaks.last().copied().unwrap_or(0);
                if actions.len() > committed {
                    phase_breaks.push(actions.len());
                    format!(
                        "ok: committed phase {} ({} edits)",
                        phase_breaks.len(),
                        actions.len() - committed
                    )
                } else {
                    // No break recorded — an empty phase would replay as an
                    // empty undo group.
                    "nothing new to commit — make edits first".to_string()
                }
            } else if sense_specs.iter().any(|spec| spec.name == call.name) {
                host_calls += 1;
                if host_calls > config.max_host_calls {
                    return abort(
                        bridge,
                        actions,
                        format!(
                            "exceeded the {}-host-call cap for one prompt",
                            config.max_host_calls
                        ),
                    );
                }
                match bridge.sense(&call.name, &call.arguments, cancel) {
                    Err(reason) => format!("rejected: {reason}"),
                    Ok(output) => {
                        let mut content = if output.text.is_empty() {
                            "ok".to_string()
                        } else {
                            output.text
                        };
                        images = output.images;
                        enforce_tool_output_image_budget(
                            &mut content,
                            &mut images,
                            config.max_images,
                            config.max_image_bytes,
                        );
                        on_event(AgentEvent::HostAction {
                            name: call.name.clone(),
                            summary: host_action_summary(&content),
                        });
                        content
                    }
                }
            } else if let Some(spec) = host_specs.iter().find(|spec| spec.name == call.name) {
                host_calls += 1;
                if host_calls > config.max_host_calls {
                    return abort(
                        bridge,
                        actions,
                        format!(
                            "exceeded the {}-host-call cap for one prompt",
                            config.max_host_calls
                        ),
                    );
                }
                match bridge.before_host_call(&call.name, &call.arguments) {
                    Err(reason) => format!("rejected: {reason}"),
                    Ok(()) => {
                        match host.authorize(&call.name, &call.arguments, spec.tier, cancel) {
                            Err(reason) => format!("rejected: {reason}"),
                            Ok(()) => {
                                let host_result = host.call(&call.name, &call.arguments, cancel);
                                let borrowed_result = match &host_result {
                                    Ok(output) => Ok(output),
                                    Err(reason) => Err(reason.as_str()),
                                };
                                if let Err(reason) = bridge.after_host_call(
                                    &call.name,
                                    &call.arguments,
                                    borrowed_result,
                                ) {
                                    return abort(
                                        bridge,
                                        actions,
                                        format!(
                                            "host tool '{}' was dispatched, but reconciliation \
                                             failed: {reason}; host effects may already have \
                                             occurred",
                                            call.name
                                        ),
                                    );
                                }
                                match host_result {
                                    Err(reason) => format!("rejected: {reason}"),
                                    Ok(output) => {
                                        let mut content = if output.text.is_empty() {
                                            "ok".to_string()
                                        } else {
                                            output.text
                                        };
                                        images = output.images;
                                        enforce_tool_output_image_budget(
                                            &mut content,
                                            &mut images,
                                            config.max_images,
                                            config.max_image_bytes,
                                        );
                                        on_event(AgentEvent::HostAction {
                                            name: call.name.clone(),
                                            summary: host_action_summary(&content),
                                        });
                                        content
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                edit_calls += 1;
                if edit_calls > config.max_tool_calls {
                    // The runaway fuse is not a failure: edits already applied
                    // stay, this and further edit calls are refused, and the
                    // run ends after this turn's calls are answered.
                    edit_cap_tripped = true;
                    messages.push(Message::ToolResult {
                        call_id: call.id,
                        content: format!(
                            "rejected: reached the {}-edit cap for one prompt; the edits \
                             already applied are kept",
                            config.max_tool_calls
                        ),
                        images,
                    });
                    continue;
                }
                match WireCommand::from_tool_call(&call.name, call.arguments.clone()) {
                    Err(reason) => format!("rejected: {reason}"),
                    Ok(command) => {
                        let applied = if config.dry_run {
                            bridge.check(&command).map(|()| None)
                        } else {
                            bridge.apply(&command).map(Some)
                        };
                        match applied {
                            Err(reason) => format!("rejected: {reason}"),
                            Ok(outcome) => {
                                let description = describe_action(&command, outcome.as_ref());
                                let entry = ActionLogEntry {
                                    command,
                                    description: description.clone(),
                                };
                                on_event(AgentEvent::Action(entry.clone()));
                                actions.push(entry);
                                if config.dry_run {
                                    format!("validated (dry run, not yet applied): {description}")
                                } else {
                                    format!("ok: {description}")
                                }
                            }
                        }
                    }
                }
            };
            messages.push(Message::ToolResult {
                call_id: call.id,
                content: result,
                images,
            });
        }

        if edit_cap_tripped {
            final_text = format!(
                "Reached the {}-edit cap for one prompt; kept the {} edit{} already applied.",
                config.max_tool_calls,
                actions.len(),
                if actions.len() == 1 { "" } else { "s" }
            );
            break;
        }

        if _turn + 1 == config.max_turns {
            return abort(
                bridge,
                actions,
                format!("exceeded the {}-turn cap for one prompt", config.max_turns),
            );
        }
    }

    let turn_messages =
        collect_turn_messages(messages, turn_start, &describe_call_ids, &final_text);
    if config.dry_run {
        return PromptOutcome {
            text: final_text,
            actions,
            phase_breaks,
            status: PromptStatus::DryRun,
            turn_messages,
        };
    }
    bridge.end_group();
    PromptOutcome {
        text: final_text,
        actions,
        phase_breaks,
        status: PromptStatus::Completed,
        turn_messages,
    }
}

/// Transcript line for one host call: the first line of its output,
/// capped so the panel never renders a wall of tool text.
fn host_action_summary(text: &str) -> String {
    const MAX_CHARS: usize = 120;
    let line = text.lines().next().unwrap_or("").trim();
    let mut summary: String = line.chars().take(MAX_CHARS).collect();
    if line.chars().count() > MAX_CHARS {
        summary.push('…');
    }
    summary
}

/// Answer a `read_skill` call from the preloaded skill set. Unknown ids
/// get a model-readable rejection listing what exists.
fn read_skill_result(skills: &[crate::extend::Skill], arguments: &serde_json::Value) -> String {
    let id = arguments.get("id").and_then(|v| v.as_str()).unwrap_or("");
    match skills.iter().find(|s| s.id == id) {
        Some(skill) => format!("# {} ({})\n\n{}", skill.name, skill.id, skill.body),
        None => format!(
            "rejected: unknown skill '{id}'; available skills: {}",
            skills
                .iter()
                .map(|s| s.id.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// Bound a single extensible tool result before it reaches either the
/// transcript or the request history. Count and encoded-byte limits both keep
/// the newest attachments, matching the whole-request policy below.
fn enforce_tool_output_image_budget(
    content: &mut String,
    images: &mut Vec<ImagePart>,
    max_images: usize,
    max_bytes: usize,
) {
    let mut count = images.len();
    let mut bytes = images
        .iter()
        .map(|image| image.data.len())
        .fold(0usize, usize::saturating_add);
    let mut drop_count = 0usize;
    for image in images.iter() {
        if count <= max_images && bytes <= max_bytes {
            break;
        }
        count = count.saturating_sub(1);
        bytes = bytes.saturating_sub(image.data.len());
        drop_count += 1;
    }
    for dropped in images.drain(..drop_count) {
        content.push_str(&format!(
            "\n[image not attached because it exceeded the request budget: {}]",
            dropped.label
        ));
    }
}

/// Keep only the newest `max_images` images across the request; older
/// ones are dropped in place and noted with a text placeholder carrying
/// the label, so the model knows what it saw and can re-request it.
/// Newest-wins matches how the agent works with vision: screenshot, look,
/// act — a stale frame is cheaper to re-take than to carry.
fn enforce_image_budget(messages: &mut [Message], max_images: usize, max_bytes: usize) {
    let mut image_total: usize = messages.iter().map(image_count).sum();
    let mut byte_total: usize = messages
        .iter()
        .flat_map(message_images)
        .map(|image| image.data.len())
        .fold(0usize, usize::saturating_add);
    if image_total <= max_images && byte_total <= max_bytes {
        return;
    }

    // Oldest first. Count how much of each image vector to drain before
    // mutating it, keeping this O(number of images) rather than repeatedly
    // removing index zero.
    for message in messages.iter_mut() {
        if image_total <= max_images && byte_total <= max_bytes {
            break;
        }
        let (content, images) = match message {
            Message::User { content, images } => (content, images),
            Message::ToolResult {
                content, images, ..
            } => (content, images),
            _ => continue,
        };
        let mut drop_count = 0usize;
        for image in images.iter() {
            if image_total <= max_images && byte_total <= max_bytes {
                break;
            }
            image_total = image_total.saturating_sub(1);
            byte_total = byte_total.saturating_sub(image.data.len());
            drop_count += 1;
        }
        for dropped in images.drain(..drop_count) {
            content.push_str(&format!("\n[image no longer attached: {}]", dropped.label));
        }
    }
}

fn image_count(message: &Message) -> usize {
    match message {
        Message::User { images, .. } | Message::ToolResult { images, .. } => images.len(),
        _ => 0,
    }
}

fn message_images(message: &Message) -> &[ImagePart] {
    match message {
        Message::User { images, .. } | Message::ToolResult { images, .. } => images,
        _ => &[],
    }
}

/// Session history is text-only: raw image bytes would bloat every later
/// request and the persisted session file for no benefit — the agent can
/// always re-screenshot the *current* state. A labeled placeholder keeps
/// the narrative ("looked at the timeline here") without the payload.
fn strip_images(content: &mut String, images: &mut Vec<ImagePart>) {
    for image in images.drain(..) {
        content.push_str(&format!("\n[image: {}]", image.label));
    }
}

/// This turn's slice of the conversation (`messages[turn_start..]`: the
/// user prompt plus every assistant/tool message the loop appended), with
/// the final text answer added (it isn't pushed during the loop),
/// `describe_project` results collapsed to a placeholder, and images
/// stripped to labels (history is text-only). This is what the session
/// appends to its history so the next prompt remembers the turn.
fn collect_turn_messages(
    messages: Vec<Message>,
    turn_start: usize,
    describe_call_ids: &[String],
    final_text: &str,
) -> Vec<Message> {
    let mut turn: Vec<Message> = messages.into_iter().skip(turn_start).collect();
    for message in &mut turn {
        match message {
            Message::ToolResult {
                call_id,
                content,
                images,
            } => {
                if describe_call_ids.iter().any(|id| id == call_id) {
                    *content =
                        "(project state omitted — see the current state in the system message)"
                            .to_string();
                }
                strip_images(content, images);
            }
            Message::User { content, images } => strip_images(content, images),
            _ => {}
        }
    }
    if !final_text.is_empty() {
        turn.push(Message::Assistant {
            content: final_text.to_string(),
            tool_calls: Vec::new(),
        });
    }
    turn
}

// --- action log ---------------------------------------------------------

fn secs(v: f64) -> String {
    format!("{v:.2}s")
}

fn rgba(c: [u8; 4]) -> String {
    format!("#{:02x}{:02x}{:02x}{:02x}", c[0], c[1], c[2], c[3])
}

fn param_name(param: &wire::WireClipParam) -> String {
    match param {
        wire::WireClipParam::Position => "position".into(),
        wire::WireClipParam::AnchorPoint => "anchor point".into(),
        wire::WireClipParam::Scale => "scale".into(),
        wire::WireClipParam::Rotation => "rotation".into(),
        wire::WireClipParam::Opacity => "opacity".into(),
        wire::WireClipParam::Crop => "crop".into(),
        wire::WireClipParam::Volume => "volume".into(),
        wire::WireClipParam::Pan => "pan".into(),
        wire::WireClipParam::Speed => "speed".into(),
        wire::WireClipParam::Effect { index, param } => format!("effect {index} {param}"),
        wire::WireClipParam::Shape { param } => format!("shape {param:?}").to_lowercase(),
        wire::WireClipParam::Text { param } => format!("text {param:?}").to_lowercase(),
        wire::WireClipParam::Look { param } => format!("look {param:?}").to_lowercase(),
        wire::WireClipParam::Style { param } => format!("style {param:?}").to_lowercase(),
    }
}

/// The keyframed value in editor language: "scale 150%", "rotation 90°",
/// "[0.25, -0.10]". Falls back to "?" when the call omitted the value (the
/// validation rejection carries the real message).
fn param_value_phrase(
    param: &wire::WireClipParam,
    value: Option<f64>,
    position: Option<[f64; 2]>,
    color: Option<[u8; 4]>,
    rect: Option<[f64; 4]>,
) -> String {
    match param {
        wire::WireClipParam::Position | wire::WireClipParam::AnchorPoint => position
            .map(|p| format!("[{:.2}, {:.2}]", p[0], p[1]))
            .unwrap_or_else(|| "?".into()),
        wire::WireClipParam::Crop => rect
            .map(|r| format!("[{:.2}, {:.2}, {:.2}, {:.2}]", r[0], r[1], r[2], r[3]))
            .unwrap_or_else(|| "?".into()),
        wire::WireClipParam::Scale | wire::WireClipParam::Opacity | wire::WireClipParam::Volume => {
            value
                .map(|v| format!("{:.0}%", v * 100.0))
                .unwrap_or_else(|| "?".into())
        }
        wire::WireClipParam::Pan => value
            .map(|v| format!("{v:.2}"))
            .unwrap_or_else(|| "?".into()),
        wire::WireClipParam::Rotation => value
            .map(|v| format!("{v:.0}°"))
            .unwrap_or_else(|| "?".into()),
        wire::WireClipParam::Shape {
            param: wire::WireShapeParam::Fill | wire::WireShapeParam::StrokeColor,
        }
        | wire::WireClipParam::Text {
            param:
                wire::WireTextParam::Fill
                | wire::WireTextParam::StrokeColor
                | wire::WireTextParam::ShadowColor,
        }
        | wire::WireClipParam::Style {
            param:
                wire::WireStyleParam::ShadowColor
                | wire::WireStyleParam::GlowColor
                | wire::WireStyleParam::OutlineColor
                | wire::WireStyleParam::BackgroundColor,
        } => color.map(rgba).unwrap_or_else(|| "?".into()),
        wire::WireClipParam::Style {
            param: wire::WireStyleParam::ShadowOffset,
        }
        | wire::WireClipParam::Look {
            param: wire::WireLookParam::MaskCenter | wire::WireLookParam::MaskSize,
        } => position
            .map(|p| format!("[{:.2}, {:.2}]", p[0], p[1]))
            .unwrap_or_else(|| "?".into()),
        _ => value.map(|v| v.to_string()).unwrap_or_else(|| "?".into()),
    }
}

fn generator_phrase(generator: &wire::WireGenerator) -> String {
    match generator {
        wire::WireGenerator::Text { content } => format!("text '{content}'"),
        wire::WireGenerator::Solid { rgba: c } => format!("solid {}", rgba(*c)),
        wire::WireGenerator::Shape {
            shape,
            rgba: c,
            width,
            height,
        } => {
            let name = match shape {
                wire::WireShape::Rectangle => "rectangle",
                wire::WireShape::Ellipse => "ellipse",
            };
            let size = match (width, height) {
                (Some(w), Some(h)) => format!(" {w:.0}×{h:.0} ref px"),
                _ => String::new(),
            };
            format!("{} {}{}", rgba(*c), name, size)
        }
    }
}

/// One transcript line per command: what happened, in editor language.
/// `outcome` is `None` for dry-run (planned, not applied).
pub fn describe_action(command: &WireCommand, outcome: Option<&EditOutcome>) -> String {
    let mut line = match command {
        WireCommand::AddTrack(a) => format!("added {:?} track '{}'", a.kind, a.name).to_lowercase(),
        WireCommand::AddClip(a) => format!(
            "placed media {} ({}–{} of source) at {} on track {}",
            a.media,
            secs(a.source_start),
            secs(a.source_start + a.source_duration),
            secs(a.start),
            a.track,
        ),
        WireCommand::ExtractAudio(a) => {
            format!(
                "extracted audio from clip {} onto track {}",
                a.clip, a.track
            )
        }
        WireCommand::DuplicateClip(a) => format!(
            "duplicated clip {} onto track {} at {}",
            a.clip,
            a.to_track,
            secs(a.start),
        ),
        WireCommand::AddGenerated(a) => format!(
            "added {} at {} for {} on track {}",
            generator_phrase(&a.generator),
            secs(a.start),
            secs(a.duration),
            a.track,
        ),
        WireCommand::SetGenerator(a) => format!(
            "changed clip {} to {}",
            a.clip,
            generator_phrase(&a.generator)
        ),
        WireCommand::SetClipTransform(a) => {
            let mut parts = Vec::new();
            if a.position_x.is_some() || a.position_y.is_some() {
                parts.push("position".to_string());
            }
            if let Some(s) = a.scale {
                parts.push(format!("scale {:.0}%", s * 100.0));
            }
            if let Some(r) = a.rotation {
                parts.push(format!("rotation {r:.0}°"));
            }
            if let Some(o) = a.opacity {
                parts.push(format!("opacity {:.0}%", o * 100.0));
            }
            format!("set clip {} {}", a.clip, parts.join(", "))
        }
        WireCommand::SetClipCrop(a) => {
            let mut parts = Vec::new();
            let edges: Vec<String> = [
                ("left", a.left),
                ("top", a.top),
                ("right", a.right),
                ("bottom", a.bottom),
            ]
            .iter()
            .filter_map(|(name, v)| v.map(|v| format!("{name} {:.0}%", v * 100.0)))
            .collect();
            if !edges.is_empty() {
                parts.push(format!("cropped {}", edges.join(", ")));
            }
            if let Some(h) = a.flip_h {
                parts.push(
                    if h {
                        "flipped horizontally"
                    } else {
                        "unflipped horizontally"
                    }
                    .into(),
                );
            }
            if let Some(v) = a.flip_v {
                parts.push(
                    if v {
                        "flipped vertically"
                    } else {
                        "unflipped vertically"
                    }
                    .into(),
                );
            }
            if parts.is_empty() {
                parts.push("framing unchanged".into());
            }
            format!("set clip {} {}", a.clip, parts.join(", "))
        }
        WireCommand::SetParamKeyframe(a) => format!(
            "keyframed clip {} {} = {} at {}",
            a.clip,
            param_name(&a.param),
            param_value_phrase(&a.param, a.value, a.position, a.rgba, a.rect),
            secs(a.at),
        ),
        WireCommand::RemoveParamKeyframe(a) => format!(
            "removed clip {} {} keyframe at {}",
            a.clip,
            param_name(&a.param),
            secs(a.at),
        ),
        WireCommand::SetParamConstant(a) => format!(
            "set clip {} {} to {} (animation cleared)",
            a.clip,
            param_name(&a.param),
            param_value_phrase(&a.param, a.value, a.position, a.rgba, a.rect),
        ),
        WireCommand::SetClipSpeed(a) => {
            let mut parts = Vec::new();
            if let Some(s) = a.speed {
                parts.push(format!("speed {s}x"));
            }
            if let Some(r) = a.reversed {
                parts.push(if r {
                    "reversed".into()
                } else {
                    "forward".to_string()
                });
            }
            if parts.is_empty() {
                parts.push("retiming unchanged".into());
            }
            format!("set clip {} {}", a.clip, parts.join(", "))
        }
        WireCommand::SetSpeedCurve(a) => match &a.preset {
            Some(preset) => format!("applied {preset} speed ramp to clip {}", a.clip),
            None => format!("cleared speed ramp on clip {}", a.clip),
        },
        WireCommand::SetClipPitch(a) => format!(
            "set clip {} pitch to {}",
            a.clip,
            if a.preserve_pitch {
                "preserved"
            } else {
                "follow speed"
            }
        ),
        WireCommand::SetDenoise(a) => format!(
            "turned noise reduction {} on clip {}",
            if a.denoise { "on" } else { "off" },
            a.clip
        ),
        WireCommand::SetClipMask(a) => match &a.mask {
            Some(mask) => format!(
                "set {} mask on clip {}",
                match mask.kind {
                    crate::wire::WireMaskKind::Linear => "linear",
                    crate::wire::WireMaskKind::Mirror => "mirror",
                    crate::wire::WireMaskKind::Circle => "circle",
                    crate::wire::WireMaskKind::Rectangle => "rectangle",
                    crate::wire::WireMaskKind::Heart => "heart",
                    crate::wire::WireMaskKind::Star => "star",
                },
                a.clip
            ),
            None => format!("cleared mask on clip {}", a.clip),
        },
        WireCommand::SetClipChroma(a) => match &a.chroma {
            Some(_) => format!("set chroma key on clip {}", a.clip),
            None => format!("cleared chroma key on clip {}", a.clip),
        },
        WireCommand::SetClipStabilize(a) => match a.level {
            Some(level) => format!(
                "set {} stabilization on clip {}",
                match level {
                    crate::wire::WireStabilizeLevel::Recommended => "recommended",
                    crate::wire::WireStabilizeLevel::Smooth => "smooth",
                    crate::wire::WireStabilizeLevel::MaxSmooth => "max smooth",
                },
                a.clip
            ),
            None => format!("cleared stabilization on clip {}", a.clip),
        },
        WireCommand::SetClipFilter(a) => match &a.filter {
            Some(filter) => format!("set {} filter on clip {}", filter.id, a.clip),
            None => format!("cleared filter on clip {}", a.clip),
        },
        WireCommand::SetClipBlendMode(a) => format!(
            "set clip {} blend mode to {}",
            a.clip,
            match a.mode {
                crate::wire::WireBlendMode::Normal => "normal",
                crate::wire::WireBlendMode::Darken => "darken",
                crate::wire::WireBlendMode::Multiply => "multiply",
                crate::wire::WireBlendMode::ColorBurn => "color_burn",
                crate::wire::WireBlendMode::Lighten => "lighten",
                crate::wire::WireBlendMode::Screen => "screen",
                crate::wire::WireBlendMode::ColorDodge => "color_dodge",
                crate::wire::WireBlendMode::Add => "add",
                crate::wire::WireBlendMode::Overlay => "overlay",
                crate::wire::WireBlendMode::SoftLight => "soft_light",
                crate::wire::WireBlendMode::HardLight => "hard_light",
                crate::wire::WireBlendMode::Difference => "difference",
                crate::wire::WireBlendMode::Exclusion => "exclusion",
            }
        ),
        WireCommand::SetClipLayerStyles(a) => {
            let mut blocks = Vec::new();
            if a.styles.shadow.is_some() {
                blocks.push("shadow");
            }
            if a.styles.glow.is_some() {
                blocks.push("glow");
            }
            if a.styles.outline.is_some() {
                blocks.push("outline");
            }
            if a.styles.background.is_some() {
                blocks.push("background");
            }
            if blocks.is_empty() {
                format!("cleared layer styles on clip {}", a.clip)
            } else {
                format!(
                    "set layer styles ({}) on clip {}",
                    blocks.join(", "),
                    a.clip
                )
            }
        }
        WireCommand::SetClipAdjustments(a) => {
            let mut parts = Vec::new();
            if let Some(v) = a.brightness {
                parts.push(format!("brightness {v:.2}"));
            }
            if let Some(v) = a.contrast {
                parts.push(format!("contrast {v:.2}"));
            }
            if let Some(v) = a.saturation {
                parts.push(format!("saturation {v:.2}"));
            }
            if let Some(v) = a.exposure {
                parts.push(format!("exposure {v:.2}"));
            }
            if let Some(v) = a.temperature {
                parts.push(format!("temperature {v:.2}"));
            }
            if let Some(v) = a.tint {
                parts.push(format!("tint {v:.2}"));
            }
            if let Some(v) = a.hue {
                parts.push(format!("hue {v:.2}"));
            }
            if let Some(v) = a.highlights {
                parts.push(format!("highlights {v:.2}"));
            }
            if let Some(v) = a.shadows {
                parts.push(format!("shadows {v:.2}"));
            }
            if let Some(v) = a.sharpness {
                parts.push(format!("sharpness {v:.2}"));
            }
            if let Some(v) = a.vignette {
                parts.push(format!("vignette {v:.2}"));
            }
            if parts.is_empty() {
                parts.push("adjustments unchanged".into());
            }
            format!("set clip {} {}", a.clip, parts.join(", "))
        }
        WireCommand::SetClipAnimation(a) => {
            let slot = match a.slot {
                crate::wire::WireAnimationSlot::In => "in",
                crate::wire::WireAnimationSlot::Out => "out",
                crate::wire::WireAnimationSlot::Combo => "combo",
            };
            match &a.animation {
                Some(id) => format!("set {} animation on clip {} ({} slot)", id, a.clip, slot),
                None => format!("cleared {} animation on clip {}", slot, a.clip),
            }
        }
        WireCommand::SetAudioRole(a) => match a.role {
            Some(role) => format!(
                "tagged clip {} as {}",
                a.clip,
                match role {
                    crate::wire::WireAudioRole::Music => "music",
                    crate::wire::WireAudioRole::Sfx => "sfx",
                    crate::wire::WireAudioRole::Voiceover => "voiceover",
                    crate::wire::WireAudioRole::Extracted => "extracted",
                }
            ),
            None => format!("cleared audio role on clip {}", a.clip),
        },
        WireCommand::SetClipAudio(a) => {
            let mut parts = Vec::new();
            if let Some(v) = a.volume {
                parts.push(if v == 0.0 {
                    "muted".to_string()
                } else {
                    format!("volume {:.0}%", v * 100.0)
                });
            }
            if let Some(f) = a.fade_in {
                parts.push(format!("fade in {}", secs(f)));
            }
            if let Some(f) = a.fade_out {
                parts.push(format!("fade out {}", secs(f)));
            }
            if parts.is_empty() {
                parts.push("audio unchanged".into());
            }
            format!("set clip {} {}", a.clip, parts.join(", "))
        }
        WireCommand::AddEffect(a) => format!("added {} effect to clip {}", a.effect, a.clip),
        WireCommand::RemoveEffect(a) => {
            format!("removed effect {} from clip {}", a.index, a.clip)
        }
        WireCommand::MoveEffect(a) => format!(
            "moved effect {} to {} on clip {}",
            a.from_index, a.to_index, a.clip
        ),
        WireCommand::SetEffectParam(a) => {
            let shown = if let Some(c) = a.rgba {
                rgba(c)
            } else if let Some(p) = a.position {
                format!("[{:.2}, {:.2}]", p[0], p[1])
            } else {
                a.value.map(|v| v.to_string()).unwrap_or_else(|| "?".into())
            };
            format!(
                "set clip {} effect {} {} = {}",
                a.clip, a.index, a.param, shown
            )
        }
        WireCommand::AddTransition(a) => {
            format!("added {} transition after clip {}", a.transition, a.clip)
        }
        WireCommand::RemoveTransition(a) => {
            format!("removed transition after clip {}", a.clip)
        }
        WireCommand::SetTransition(a) => {
            format!("set transition after clip {} to {}s", a.clip, a.seconds)
        }
        WireCommand::SplitClip(a) => format!("split clip {} at {}", a.clip, secs(a.at)),
        WireCommand::TrimClip(a) => format!(
            "trimmed clip {} to {}–{}",
            a.clip,
            secs(a.start),
            secs(a.start + a.duration)
        ),
        WireCommand::MoveClip(a) => {
            format!(
                "moved clip {} to {} on track {}",
                a.clip,
                secs(a.start),
                a.to_track
            )
        }
        WireCommand::RemoveClip(a) => format!("removed clip {}", a.clip),
        WireCommand::RemoveTrack(a) => format!("removed track {}", a.track),
        WireCommand::SetTrackEnabled(a) => format!(
            "{} track {}",
            if a.enabled { "showed" } else { "hid" },
            a.track
        ),
        WireCommand::SetTrackMuted(a) => format!(
            "{} track {}",
            if a.muted { "muted" } else { "unmuted" },
            a.track
        ),
        WireCommand::SetTrackLocked(a) => format!(
            "{} track {}",
            if a.locked { "locked" } else { "unlocked" },
            a.track
        ),
        WireCommand::RippleDelete(a) => {
            format!(
                "ripple-deleted clip {} (later clips closed the gap)",
                a.clip
            )
        }
        WireCommand::ShiftClips(a) => format!(
            "shifted clips on track {} from {} by {:+.2}s",
            a.track,
            secs(a.from),
            a.delta
        ),
        WireCommand::RippleInsert(a) => format!(
            "ripple-inserted media {} at {} on track {} (later clips moved right)",
            a.media,
            secs(a.at),
            a.track
        ),
        WireCommand::LinkClips(a) => format!(
            "linked clips {}",
            a.clips
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        WireCommand::UnlinkClips(a) => format!(
            "unlinked complete groups touched by clips {}",
            a.clips
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        WireCommand::AddMarker(a) => {
            let name = match &a.name {
                Some(name) if !name.is_empty() => format!(" '{name}'"),
                _ => String::new(),
            };
            let color = a
                .color
                .map(|c| format!(" ({c:?})").to_lowercase())
                .unwrap_or_default();
            format!("added marker{name} at {}{color}", secs(a.at))
        }
        WireCommand::RemoveMarker(a) => format!("removed marker {}", a.marker),
        WireCommand::SetMarker(a) => {
            let mut parts = Vec::new();
            if let Some(at) = a.at {
                parts.push(format!("moved to {}", secs(at)));
            }
            if let Some(name) = &a.name {
                parts.push(format!("named '{name}'"));
            }
            if let Some(color) = a.color {
                parts.push(format!("colored {color:?}").to_lowercase());
            }
            if parts.is_empty() {
                parts.push("unchanged".into());
            }
            format!("set marker {} {}", a.marker, parts.join(", "))
        }
        WireCommand::SetCanvas(a) => {
            let mut parts = Vec::new();
            if let Some(aspect) = a.aspect {
                parts.push(format!("aspect {}", aspect.name()));
            }
            if let Some([r, g, b]) = a.background {
                parts.push(format!("background rgb({r}, {g}, {b})"));
            }
            if parts.is_empty() {
                parts.push("unchanged".into());
            }
            format!("set canvas {}", parts.join(", "))
        }
    };
    match outcome {
        Some(EditOutcome::Created(id)) => line.push_str(&format!(" (new clip {})", id.raw())),
        Some(EditOutcome::CreatedTrack(id)) => line.push_str(&format!(" (track {})", id.raw())),
        Some(EditOutcome::CreatedMarker(id)) => line.push_str(&format!(" (marker {})", id.raw())),
        _ => {}
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_models::ClipId;

    #[test]
    fn action_lines_read_like_an_edit_log() {
        let split = WireCommand::SplitClip(wire::SplitClip { clip: 7, at: 12.4 });
        assert_eq!(
            describe_action(&split, Some(&EditOutcome::Created(ClipId::from_raw(21)))),
            "split clip 7 at 12.40s (new clip 21)"
        );

        let move_effect = WireCommand::MoveEffect(wire::MoveEffect {
            clip: 7,
            from_index: 0,
            to_index: 2,
        });
        assert_eq!(
            describe_action(&move_effect, None),
            "moved effect 0 to 2 on clip 7"
        );

        let extract = WireCommand::ExtractAudio(wire::ExtractAudio { clip: 7, track: 3 });
        assert_eq!(
            describe_action(&extract, Some(&EditOutcome::Created(ClipId::from_raw(22)))),
            "extracted audio from clip 7 onto track 3 (new clip 22)"
        );

        let duplicate = WireCommand::DuplicateClip(wire::DuplicateClip {
            clip: 7,
            to_track: 3,
            start: 12.5,
        });
        assert_eq!(
            describe_action(
                &duplicate,
                Some(&EditOutcome::Created(ClipId::from_raw(23)))
            ),
            "duplicated clip 7 onto track 3 at 12.50s (new clip 23)"
        );

        let trim = WireCommand::TrimClip(wire::TrimClip {
            clip: 12,
            start: 3.0,
            duration: 7.0,
        });
        assert_eq!(
            describe_action(&trim, Some(&EditOutcome::Updated(ClipId::from_raw(12)))),
            "trimmed clip 12 to 3.00s–10.00s"
        );

        let title = WireCommand::AddGenerated(wire::AddGenerated {
            track: 3,
            generator: wire::WireGenerator::Text {
                content: "INTRO".into(),
            },
            start: 0.0,
            duration: 3.0,
        });
        assert_eq!(
            describe_action(&title, None),
            "added text 'INTRO' at 0.00s for 3.00s on track 3"
        );

        let canvas = WireCommand::SetCanvas(wire::SetCanvas {
            aspect: Some(wire::WireCanvasAspect::Tall9x16),
            background: Some([20, 20, 28]),
        });
        assert_eq!(
            describe_action(&canvas, Some(&EditOutcome::UpdatedCanvas)),
            "set canvas aspect 9:16, background rgb(20, 20, 28)"
        );
    }

    #[test]
    fn system_prompt_carries_state_and_trim_rule() {
        let summary = ProjectSummary {
            name: "demo".into(),
            frame_rate_fps: 24.0,
            duration_seconds: 10.0,
            tracks: vec![],
            markers: vec![],
            canvas: None,
            media: vec![],
        };
        let ctx = EditorContext {
            selected_clips: vec![12],
            playhead_seconds: 3.5,
            ..Default::default()
        };
        let prompt = system_prompt(&summary, &ctx, &AgentExtensions::default());
        assert!(prompt.contains("\"selected_clips\":[12]"));
        assert!(prompt.contains("INCREASE start"));
        assert!(prompt.contains("\"name\":\"demo\""));
        // The Q&A rule: answer from the pushed state, no tool calls.
        assert!(prompt.contains("answer directly from"));
        // The re-inspect rule: after edits, read the new state, don't give up.
        assert!(prompt.contains("call describe_project to read the new"));
        // Unknown footage content is fetchable, not grounds to refuse.
        assert!(prompt.contains("Unknown source-footage content is not missing project state"));
        assert!(prompt.contains("instead of declining the task"));
        // An empty timeline can be constructed directly from media-pool items.
        assert!(prompt.contains("add_clip is the operation that places media-pool footage"));
        assert!(prompt.contains("An empty timeline is a starting point"));
        assert!(prompt.contains("Never ask the user to pre-place footage"));
        // The overlap rule: make room before growing into a packed track.
        assert!(prompt.contains("Clips on one track can never overlap"));
        // No extensions ⇒ no rules or skills sections.
        assert!(!prompt.contains("User rules"));
        assert!(!prompt.contains("read_skill"));
    }

    #[test]
    fn system_prompt_injects_rules_and_skill_index_only() {
        let summary = ProjectSummary {
            name: "demo".into(),
            frame_rate_fps: 24.0,
            duration_seconds: 10.0,
            tracks: vec![],
            markers: vec![],
            canvas: None,
            media: vec![],
        };
        let extensions = AgentExtensions {
            rules: "[user]\nalways vertical 9:16".into(),
            skills: vec![crate::extend::Skill {
                id: "podcast-cleanup".into(),
                name: "Podcast cleanup".into(),
                description: "Clean up a talk recording.".into(),
                body: "SECRET BODY".into(),
            }],
        };
        let prompt = system_prompt(&summary, &EditorContext::default(), &extensions);
        assert!(prompt.contains("always vertical 9:16"));
        assert!(prompt.contains("podcast-cleanup (Podcast cleanup): Clean up a talk recording."));
        // Only the index enters the prompt — bodies load through read_skill.
        assert!(!prompt.contains("SECRET BODY"));
    }

    #[test]
    fn engine_sense_rules_make_open_ended_creative_work_actionable() {
        let rules = engine_sense_rules(&[
            HostToolSpec {
                name: "media_pool_sheet".into(),
                description: "Survey imported visual media.".into(),
                parameters: serde_json::json!({"type": "object"}),
                tier: crate::tools::ToolTier::ReadOnly,
            },
            HostToolSpec {
                name: "media_asset_strip".into(),
                description: "Inspect one source over time.".into(),
                parameters: serde_json::json!({"type": "object"}),
                tier: crate::tools::ToolTier::ReadOnly,
            },
        ]);

        assert!(rules.contains("complete current project snapshot"));
        assert!(rules.contains("freestyle edits"));
        assert!(rules.contains("media_pool_sheet"));
        assert!(rules.contains("media_asset_strip"));
        assert!(rules.contains("rather than declining"));
        assert!(rules.contains("create the required tracks with add_track"));
        assert!(rules.contains("build the sequence with add_clip"));
    }

    #[test]
    fn image_budget_drops_oldest_and_leaves_labeled_placeholders() {
        let mut messages = vec![
            Message::system("s"),
            Message::User {
                content: "look at these".into(),
                images: vec![
                    ImagePart::png(vec![1], "timeline at 2.00s"),
                    ImagePart::png(vec![2], "timeline at 5.00s"),
                ],
            },
            Message::ToolResult {
                call_id: "call_1".into(),
                content: "screenshot taken".into(),
                images: vec![ImagePart::jpeg(vec![3], "preview at 8.00s")],
            },
        ];

        enforce_image_budget(&mut messages, 1, usize::MAX);

        match &messages[1] {
            Message::User { content, images } => {
                assert!(images.is_empty(), "both older images dropped");
                assert!(content.contains("no longer attached: timeline at 2.00s"));
                assert!(content.contains("no longer attached: timeline at 5.00s"));
            }
            other => panic!("unexpected {other:?}"),
        }
        match &messages[2] {
            Message::ToolResult {
                content, images, ..
            } => {
                assert_eq!(images.len(), 1, "the newest image survives");
                assert_eq!(images[0].label, "preview at 8.00s");
                assert!(!content.contains("no longer attached"));
            }
            other => panic!("unexpected {other:?}"),
        }

        // Under budget: untouched.
        let mut under = vec![Message::User {
            content: "one".into(),
            images: vec![ImagePart::png(vec![1], "a")],
        }];
        enforce_image_budget(&mut under, 8, usize::MAX);
        assert_eq!(image_count(&under[0]), 1);
    }

    #[test]
    fn image_byte_budget_keeps_the_newest_payload_that_fits() {
        let mut messages = vec![
            Message::User {
                content: "old".into(),
                images: vec![ImagePart::png(vec![1; 6], "old six bytes")],
            },
            Message::ToolResult {
                call_id: "call_1".into(),
                content: "new".into(),
                images: vec![
                    ImagePart::png(vec![2; 4], "new four bytes"),
                    ImagePart::png(vec![3; 5], "newest five bytes"),
                ],
            },
        ];

        enforce_image_budget(&mut messages, 8, 9);

        let Message::User { content, images } = &messages[0] else {
            panic!("user message");
        };
        assert!(images.is_empty());
        assert!(content.contains("old six bytes"));
        let Message::ToolResult {
            content, images, ..
        } = &messages[1]
        else {
            panic!("tool result");
        };
        assert_eq!(
            images
                .iter()
                .map(|image| image.label.as_str())
                .collect::<Vec<_>>(),
            vec!["new four bytes", "newest five bytes"]
        );
        assert!(!content.contains("no longer attached"));

        enforce_image_budget(&mut messages, 8, 3);
        let Message::ToolResult {
            content, images, ..
        } = &messages[1]
        else {
            panic!("tool result");
        };
        assert!(
            images.is_empty(),
            "an individually oversized newest image drops"
        );
        assert!(content.contains("new four bytes"));
        assert!(content.contains("newest five bytes"));
    }

    #[test]
    fn tool_output_budget_drops_before_transcript_delivery() {
        let mut content = "frames ready".to_string();
        let mut images = vec![
            ImagePart::png(vec![1; 5], "old"),
            ImagePart::png(vec![2; 4], "middle"),
            ImagePart::png(vec![3; 3], "new"),
        ];

        enforce_tool_output_image_budget(&mut content, &mut images, 2, 7);

        assert_eq!(
            images
                .iter()
                .map(|image| image.label.as_str())
                .collect::<Vec<_>>(),
            vec!["middle", "new"]
        );
        assert!(content.contains("request budget: old"), "{content}");
    }

    #[test]
    fn turn_messages_strip_images_to_labels() {
        let messages = vec![
            Message::system("s"),
            Message::User {
                content: "what's here?".into(),
                images: vec![ImagePart::png(vec![1], "frame at 0.00s")],
            },
            Message::Assistant {
                content: String::new(),
                tool_calls: Vec::new(),
            },
            Message::ToolResult {
                call_id: "call_1".into(),
                content: "took the shot".into(),
                images: vec![ImagePart::jpeg(vec![2], "preview at 3.00s")],
            },
        ];

        let turn = collect_turn_messages(messages, 1, &[], "done");

        for message in &turn {
            assert_eq!(image_count(message), 0, "history is text-only: {message:?}");
        }
        match &turn[0] {
            Message::User { content, .. } => {
                assert!(content.contains("[image: frame at 0.00s]"), "{content}");
            }
            other => panic!("unexpected {other:?}"),
        }
        match &turn[2] {
            Message::ToolResult { content, .. } => {
                assert!(content.contains("[image: preview at 3.00s]"), "{content}");
            }
            other => panic!("unexpected {other:?}"),
        }
        assert_eq!(
            turn.last(),
            Some(&Message::assistant_text("done")),
            "the final answer is appended"
        );
    }

    #[test]
    fn host_action_summary_keeps_the_first_line_capped() {
        assert_eq!(host_action_summary("saved\ndetails follow"), "saved");
        let long = "x".repeat(200);
        let summary = host_action_summary(&long);
        assert_eq!(summary.chars().count(), 121, "120 chars + ellipsis");
        assert!(summary.ends_with('…'));
    }

    #[test]
    fn read_skill_returns_body_or_lists_available() {
        let skills = vec![crate::extend::Skill {
            id: "podcast-cleanup".into(),
            name: "Podcast cleanup".into(),
            description: "d".into(),
            body: "Step 1: denoise.".into(),
        }];
        let ok = read_skill_result(&skills, &serde_json::json!({ "id": "podcast-cleanup" }));
        assert!(ok.contains("Step 1: denoise."));
        let missing = read_skill_result(&skills, &serde_json::json!({ "id": "nope" }));
        assert!(missing.starts_with("rejected: unknown skill 'nope'"));
        assert!(missing.contains("podcast-cleanup"));
    }
}
