//! The agent loop: prompt in, validated-and-applied command group out,
//! every step observable.
//!
//! The loop's whole world is the [`EngineBridge`] — it cannot name a file
//! path, a socket, or a UI type. One prompt = one history group: the
//! bridge's group markers wrap the run, failed individual commands are
//! reported back to the model (which may correct course), and the group
//! rolls back only when the prompt aborts (cancellation, provider error,
//! cap exceeded). In dry-run mode nothing is applied; the validated plan
//! comes back for the UI's preview card.

use std::sync::atomic::AtomicBool;

use cutlass_commands::EditOutcome;

use crate::describe::{EditorContext, ProjectSummary};
use crate::extend::AgentExtensions;
use crate::provider::{ChatProvider, ChatRequest, FinishReason, ImagePart, Message, ProviderError};
use crate::wire::{self, WireCommand};

/// The loop's only view of the engine. The UI implements this over a
/// sandbox engine whose validated plan replays onto the live one
/// (`cutlass-ui/src/agent.rs`); tests implement it over a plain `Engine`.
pub trait EngineBridge {
    /// Fresh summary of the project as it stands.
    fn summary(&mut self) -> ProjectSummary;
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
    pub max_tool_calls: usize,
    /// Hard cap on provider turns per prompt.
    pub max_turns: usize,
    /// Hard cap on images carried by one request, newest kept. Screenshot
    /// tools bound each image's dimensions, so count × bounded size caps
    /// the whole vision payload.
    pub max_images: usize,
    /// Validate and collect the plan without applying anything.
    pub dry_run: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_tool_calls: 32,
            max_turns: 16,
            max_images: 8,
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
    /// An edit was applied (or validated, in dry-run).
    Action(ActionLogEntry),
}

/// How the prompt ended.
#[derive(Debug, Clone, PartialEq)]
pub enum PromptStatus {
    /// Edits applied (possibly none) and recorded as one history entry.
    Completed,
    /// Dry-run: the plan in `actions` validated but nothing was applied.
    DryRun,
    /// Rolled back; nothing from this prompt remains. The string says why.
    Aborted(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PromptOutcome {
    /// The model's final text answer (empty if it only edited).
    pub text: String,
    pub actions: Vec<ActionLogEntry>,
    pub status: PromptStatus,
    /// This turn's conversation, ready to append to the session history so
    /// the next prompt remembers it: the user message, every assistant
    /// turn and tool result the loop produced, and the final text answer.
    /// Empty when the prompt aborted (nothing applied → no memory trace).
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
         answer a question, say what is missing instead of guessing.\n\
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

/// Run one prompt to completion against `bridge`.
///
/// `context` is the send-time editor snapshot (selection, playhead);
/// `history` is the prior conversation in this session (the caller's
/// accumulated `turn_messages`, with no system message — a fresh one is
/// regenerated here so the current project state always wins); `on_event`
/// receives streamed text and applied actions for the UI. The returned
/// [`PromptOutcome::turn_messages`] is this turn's contribution to append.
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
    let summary = bridge.summary();
    let mut messages = Vec::with_capacity(history.len() + 2);
    messages.push(Message::System {
        content: system_prompt(&summary, context, extensions),
    });
    messages.extend_from_slice(history);
    // This turn's own messages start here (the user prompt and everything
    // the loop appends), kept so we can hand them back as `turn_messages`.
    let turn_start = messages.len();
    messages.push(Message::user(prompt));
    let mut tools = wire::tool_specs();
    tools.push(wire::describe_project_spec());
    if !extensions.skills.is_empty() {
        tools.push(crate::extend::read_skill_spec());
    }

    let mut actions: Vec<ActionLogEntry> = Vec::new();
    let mut edit_calls = 0usize;
    let mut final_text = String::new();
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
            status: PromptStatus::Aborted(reason),
            turn_messages: Vec::new(),
        }
    };

    for _turn in 0..config.max_turns {
        enforce_image_budget(&mut messages, config.max_images);
        let turn = {
            let mut forward = |delta: &str| on_event(AgentEvent::TextDelta(delta.to_string()));
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
            } else {
                edit_calls += 1;
                if edit_calls > config.max_tool_calls {
                    return abort(
                        bridge,
                        actions,
                        format!(
                            "exceeded the {}-edit cap for one prompt",
                            config.max_tool_calls
                        ),
                    );
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
            messages.push(Message::tool_result(call.id, result));
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
            status: PromptStatus::DryRun,
            turn_messages,
        };
    }
    bridge.end_group();
    PromptOutcome {
        text: final_text,
        actions,
        status: PromptStatus::Completed,
        turn_messages,
    }
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

/// Keep only the newest `max_images` images across the request; older
/// ones are dropped in place and noted with a text placeholder carrying
/// the label, so the model knows what it saw and can re-request it.
/// Newest-wins matches how the agent works with vision: screenshot, look,
/// act — a stale frame is cheaper to re-take than to carry.
fn enforce_image_budget(messages: &mut [Message], max_images: usize) {
    let total: usize = messages.iter().map(image_count).sum();
    let mut to_drop = total.saturating_sub(max_images);
    if to_drop == 0 {
        return;
    }
    // Oldest first: walk from the front, draining each message's images
    // front-first until the excess is gone.
    for message in messages.iter_mut() {
        if to_drop == 0 {
            break;
        }
        let (content, images) = match message {
            Message::User { content, images } => (content, images),
            Message::ToolResult {
                content, images, ..
            } => (content, images),
            _ => continue,
        };
        while to_drop > 0 && !images.is_empty() {
            let dropped = images.remove(0);
            content.push_str(&format!("\n[image no longer attached: {}]", dropped.label));
            to_drop -= 1;
        }
    }
}

fn image_count(message: &Message) -> usize {
    match message {
        Message::User { images, .. } | Message::ToolResult { images, .. } => images.len(),
        _ => 0,
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

fn param_name(param: wire::WireClipParam) -> &'static str {
    match param {
        wire::WireClipParam::Position => "position",
        wire::WireClipParam::Scale => "scale",
        wire::WireClipParam::Rotation => "rotation",
        wire::WireClipParam::Opacity => "opacity",
        wire::WireClipParam::Volume => "volume",
    }
}

/// The keyframed value in editor language: "scale 150%", "rotation 90°",
/// "[0.25, -0.10]". Falls back to "?" when the call omitted the value (the
/// validation rejection carries the real message).
fn param_value_phrase(
    param: wire::WireClipParam,
    value: Option<f64>,
    position: Option<[f64; 2]>,
) -> String {
    match param {
        wire::WireClipParam::Position => position
            .map(|p| format!("[{:.2}, {:.2}]", p[0], p[1]))
            .unwrap_or_else(|| "?".into()),
        wire::WireClipParam::Scale | wire::WireClipParam::Opacity | wire::WireClipParam::Volume => {
            value
                .map(|v| format!("{:.0}%", v * 100.0))
                .unwrap_or_else(|| "?".into())
        }
        wire::WireClipParam::Rotation => value
            .map(|v| format!("{v:.0}°"))
            .unwrap_or_else(|| "?".into()),
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
            param_name(a.param),
            param_value_phrase(a.param, a.value, a.position),
            secs(a.at),
        ),
        WireCommand::RemoveParamKeyframe(a) => format!(
            "removed clip {} {} keyframe at {}",
            a.clip,
            param_name(a.param),
            secs(a.at),
        ),
        WireCommand::SetParamConstant(a) => format!(
            "set clip {} {} to {} (animation cleared)",
            a.clip,
            param_name(a.param),
            param_value_phrase(a.param, a.value, a.position),
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
        WireCommand::SetEffectParam(a) => {
            format!(
                "set clip {} effect {} {} = {}",
                a.clip, a.index, a.param, a.value
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

        enforce_image_budget(&mut messages, 1);

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
        enforce_image_budget(&mut under, 8);
        assert_eq!(image_count(&under[0]), 1);
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
