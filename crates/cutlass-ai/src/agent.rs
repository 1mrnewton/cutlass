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
    TokenUsage,
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
    /// Cumulative token usage for this prompt so far (sum of every provider
    /// turn that reported usage). Emitted after each such turn.
    Usage(TokenUsage),
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
    /// Cumulative provider-reported token usage across every turn of this
    /// prompt (including turns before an abort — tokens already spent stay).
    pub usage: TokenUsage,
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
    let mut usage = TokenUsage::default();
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
    let abort =
        |bridge: &mut dyn EngineBridge, actions: Vec<ActionLogEntry>, usage: TokenUsage, reason: String| {
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
                usage,
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
                    return abort(bridge, actions, usage, "cancelled".to_string());
                }
                Err(e) => return abort(bridge, actions, usage, e.to_string()),
            }
        };
        if let Some(turn_usage) = &turn.usage {
            usage.add(turn_usage);
            on_event(AgentEvent::Usage(usage));
        }

        if turn.tool_calls.is_empty() {
            final_text = turn.text;
            if turn.finish == FinishReason::Length {
                return abort(
                    bridge,
                    actions,
                    usage,
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
                        usage,
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
                        usage,
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
                                        usage,
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
                usage,
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
            usage,
        };
    }
    bridge.end_group();
    PromptOutcome {
        text: final_text,
        actions,
        phase_breaks,
        status: PromptStatus::Completed,
        turn_messages,
        usage,
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

mod action_log;
pub use action_log::describe_action;

#[cfg(test)]
mod tests;
