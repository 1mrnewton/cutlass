//! The eval harness: scripted prompts against a real engine, zero network.
//!
//! Each test scripts the provider's turns, runs the full agent loop
//! through an `EngineBridge` backed by a real `Engine`, and asserts on
//! the final timeline, the action log, and the undo history. This is how
//! agent regressions get caught in CI without a live model.

use std::sync::atomic::AtomicBool;

use cutlass_ai::agent::{
    AgentConfig, AgentEvent, EngineBridge, PromptStatus, run_prompt_with_host,
};
use cutlass_ai::provider::{ChatTurn, FinishReason, ImagePart, Message, ToolCall};
use cutlass_ai::providers::ScriptedProvider;
use cutlass_ai::tools::{HostToolSpec, NullToolHost, ToolHost, ToolOutput, ToolTier};
use cutlass_ai::{EditorContext, ProjectSummary, WireCommand, summarize, validate};
use cutlass_commands::EditOutcome;
use cutlass_engine::{ApplyOutcome, Engine, EngineConfig};
use cutlass_models::{
    Generator, LinkId, MediaSource, Project, Rational, RationalTime, TimeRange, TrackKind,
};

const R24: Rational = Rational::FPS_24;

/// A real engine behind the loop's bridge.
struct EngineHost {
    engine: Engine,
    sense_specs: Vec<HostToolSpec>,
    sense_outputs: std::collections::VecDeque<Result<ToolOutput, String>>,
    sense_calls: Vec<(String, serde_json::Value)>,
    sense_clip_counts: Vec<usize>,
    before_host_outputs: std::collections::VecDeque<Result<(), String>>,
    after_host_outputs: std::collections::VecDeque<Result<(), String>>,
    before_host_calls: Vec<(String, serde_json::Value)>,
    after_host_calls: Vec<(String, serde_json::Value, bool)>,
}

impl EngineHost {
    fn new(project: Project) -> Self {
        let config = EngineConfig { undo_limit: 64 };
        Self {
            engine: Engine::with_project(config, project).expect("engine"),
            sense_specs: Vec::new(),
            sense_outputs: std::collections::VecDeque::new(),
            sense_calls: Vec::new(),
            sense_clip_counts: Vec::new(),
            before_host_outputs: std::collections::VecDeque::new(),
            after_host_outputs: std::collections::VecDeque::new(),
            before_host_calls: Vec::new(),
            after_host_calls: Vec::new(),
        }
    }
}

impl EngineBridge for EngineHost {
    fn summary(&mut self) -> ProjectSummary {
        summarize(self.engine.project())
    }

    fn sense_tools(&self) -> Vec<HostToolSpec> {
        self.sense_specs.clone()
    }

    fn sense(
        &mut self,
        name: &str,
        arguments: &serde_json::Value,
        _cancel: &AtomicBool,
    ) -> Result<ToolOutput, String> {
        self.sense_clip_counts
            .push(self.engine.project().timeline().clip_count());
        self.sense_calls.push((name.to_string(), arguments.clone()));
        self.sense_outputs
            .pop_front()
            .unwrap_or_else(|| Err("scripted engine sense ran out of outputs".into()))
    }

    fn before_host_call(
        &mut self,
        name: &str,
        arguments: &serde_json::Value,
    ) -> Result<(), String> {
        self.before_host_calls
            .push((name.to_string(), arguments.clone()));
        self.before_host_outputs.pop_front().unwrap_or(Ok(()))
    }

    fn after_host_call(
        &mut self,
        name: &str,
        arguments: &serde_json::Value,
        result: Result<&ToolOutput, &str>,
    ) -> Result<(), String> {
        self.after_host_calls
            .push((name.to_string(), arguments.clone(), result.is_ok()));
        self.after_host_outputs.pop_front().unwrap_or(Ok(()))
    }

    fn apply(&mut self, command: &WireCommand) -> Result<EditOutcome, String> {
        let lowered = validate(command, self.engine.project()).map_err(|r| r.message)?;
        match self.engine.apply(lowered) {
            Ok(ApplyOutcome::Edited(outcome)) => Ok(outcome),
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

/// 24 fps project, one video track, one 10 s clip (of a 60 s source) at 0 s.
/// Built directly on the `Project` so the engine starts with empty history.
fn fixture() -> (EngineHost, u64, u64, u64) {
    let mut project = Project::new("eval", R24);
    let media = project.add_media(MediaSource::new(
        "/tmp/eval.mp4",
        1920,
        1080,
        R24,
        60 * 24,
        true,
    ));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip = project
        .add_clip(
            track,
            media,
            TimeRange::at_rate(0, 240, R24),
            RationalTime::new(0, R24),
        )
        .unwrap();
    (
        EngineHost::new(project),
        media.raw(),
        track.raw(),
        clip.raw(),
    )
}

fn tool_turn(calls: Vec<(&str, &str, serde_json::Value)>) -> ChatTurn {
    ChatTurn {
        text: String::new(),
        reasoning_summary: String::new(),
        tool_calls: calls
            .into_iter()
            .map(|(id, name, arguments)| ToolCall {
                id: id.to_string(),
                name: name.to_string(),
                arguments,
            })
            .collect(),
        finish: FinishReason::ToolCalls,
    }
}

fn text_turn(text: &str) -> ChatTurn {
    ChatTurn {
        text: text.to_string(),
        reasoning_summary: String::new(),
        tool_calls: Vec::new(),
        finish: FinishReason::Stop,
    }
}

fn run_with(
    provider: &dyn cutlass_ai::provider::ChatProvider,
    host: &mut EngineHost,
    tool_host: &mut dyn ToolHost,
    context: &EditorContext,
    prompt: &str,
    config: &AgentConfig,
) -> (cutlass_ai::PromptOutcome, Vec<AgentEvent>) {
    let cancel = AtomicBool::new(false);
    let mut events = Vec::new();
    let outcome = run_prompt_with_host(
        provider,
        host,
        tool_host,
        context,
        &cutlass_ai::AgentExtensions::default(),
        &[],
        prompt,
        config,
        &cancel,
        &mut |e| events.push(e),
    );
    (outcome, events)
}

fn run(
    provider: &ScriptedProvider,
    host: &mut EngineHost,
    context: &EditorContext,
    prompt: &str,
    config: &AgentConfig,
) -> (cutlass_ai::PromptOutcome, Vec<AgentEvent>) {
    run_with(provider, host, &mut NullToolHost, context, prompt, config)
}

/// Scripted [`ToolHost`] double: canned outputs in call order, every
/// call recorded for assertions.
struct ScriptedHost {
    specs: Vec<HostToolSpec>,
    outputs: std::collections::VecDeque<Result<ToolOutput, String>>,
    authorizations: Vec<(String, serde_json::Value, ToolTier)>,
    calls: Vec<(String, serde_json::Value)>,
}

impl ScriptedHost {
    fn new(specs: Vec<HostToolSpec>, outputs: Vec<Result<ToolOutput, String>>) -> Self {
        Self {
            specs,
            outputs: outputs.into(),
            authorizations: Vec::new(),
            calls: Vec::new(),
        }
    }
}

impl ToolHost for ScriptedHost {
    fn tools(&self) -> Vec<HostToolSpec> {
        self.specs.clone()
    }

    fn authorize(
        &mut self,
        name: &str,
        arguments: &serde_json::Value,
        tier: ToolTier,
        _cancel: &AtomicBool,
    ) -> Result<(), String> {
        self.authorizations
            .push((name.to_string(), arguments.clone(), tier));
        match tier {
            ToolTier::ReadOnly | ToolTier::Workspace => Ok(()),
            ToolTier::System => Err(format!(
                "system tool '{name}' requires confirmation, but this host has no approval broker"
            )),
        }
    }

    fn call(
        &mut self,
        name: &str,
        arguments: &serde_json::Value,
        _cancel: &AtomicBool,
    ) -> Result<ToolOutput, String> {
        self.calls.push((name.to_string(), arguments.clone()));
        self.outputs
            .pop_front()
            .unwrap_or_else(|| Err("scripted host ran out of outputs".into()))
    }
}

fn host_spec(name: &'static str) -> HostToolSpec {
    HostToolSpec {
        name: name.into(),
        description: "test tool".into(),
        parameters: serde_json::json!({ "type": "object", "properties": {} }),
        tier: ToolTier::ReadOnly,
    }
}

#[test]
fn cut_the_first_three_seconds() {
    let (mut host, _, _, clip) = fixture();
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![(
            "call_1",
            "trim_clip",
            serde_json::json!({ "clip": clip, "start": 3.0, "duration": 7.0 }),
        )]),
        text_turn("Cut the first 3 seconds; the clip now runs 3.00s–10.00s."),
    ]);

    let context = EditorContext {
        selected_clips: vec![clip],
        ..Default::default()
    };
    let (outcome, events) = run(
        &provider,
        &mut host,
        &context,
        "cut the first 3 seconds of the selected clip",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(outcome.actions.len(), 1);
    assert_eq!(
        outcome.actions[0].description,
        format!("trimmed clip {clip} to 3.00s–10.00s")
    );
    assert!(outcome.text.contains("3.00s–10.00s"));
    assert!(events.iter().any(|e| matches!(e, AgentEvent::Action(_))));

    // The edit landed, frame-snapped, with the source in-point advanced.
    let placed = host
        .engine
        .project()
        .clip(cutlass_models::ClipId::from_raw(clip))
        .unwrap();
    assert_eq!(placed.timeline, TimeRange::at_rate(72, 168, R24));
    assert_eq!(placed.source_range().unwrap().start.value, 72);

    // One prompt = one history entry: a single undo restores everything.
    assert!(host.engine.undo());
    let restored = host
        .engine
        .project()
        .clip(cutlass_models::ClipId::from_raw(clip))
        .unwrap();
    assert_eq!(restored.timeline, TimeRange::at_rate(0, 240, R24));
    assert!(!host.engine.undo(), "exactly one history entry per prompt");

    // The system prompt carried the send-time selection.
    let first_request = &provider.requests()[0];
    match &first_request[0] {
        Message::System { content } => {
            assert!(content.contains(&format!("\"selected_clips\":[{clip}]")));
        }
        other => panic!("expected system message, got {other:?}"),
    }
}

#[test]
fn duplicate_selected_clip_and_one_prompt_undo_removes_the_copy() {
    let (mut host, _, track, clip) = fixture();
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![(
            "call_1",
            "duplicate_clip",
            serde_json::json!({
                "clip": clip,
                "to_track": track,
                "start": 12.0,
            }),
        )]),
        text_turn("Duplicated the selected clip at 12 seconds."),
    ]);
    let context = EditorContext {
        selected_clips: vec![clip],
        ..Default::default()
    };

    let (outcome, _) = run(
        &provider,
        &mut host,
        &context,
        "duplicate the selected clip at 12 seconds",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(outcome.actions.len(), 1);
    assert_eq!(
        outcome.actions[0].command,
        WireCommand::DuplicateClip(cutlass_ai::wire::DuplicateClip {
            clip,
            to_track: track,
            start: 12.0,
        })
    );
    let duplicate = host
        .engine
        .project()
        .timeline()
        .tracks_ordered()
        .flat_map(|track| track.clips())
        .find(|candidate| candidate.id.raw() != clip)
        .expect("duplicated clip")
        .clone();
    assert_eq!(duplicate.timeline, TimeRange::at_rate(288, 240, R24));
    assert_eq!(duplicate.link, None);
    assert_eq!(
        outcome.actions[0].description,
        format!(
            "duplicated clip {clip} onto track {track} at 12.00s (new clip {})",
            duplicate.id.raw()
        )
    );

    assert!(host.engine.undo(), "one prompt is one undo entry");
    assert!(host.engine.project().clip(duplicate.id).is_none());
    assert!(
        host.engine
            .project()
            .clip(cutlass_models::ClipId::from_raw(clip))
            .is_some()
    );
    assert_eq!(host.engine.project().timeline().clip_count(), 1);
    assert!(!host.engine.undo(), "the fixture itself created no history");
}

#[test]
fn unlink_one_member_commits_the_complete_group_as_one_phase() {
    let mut project = Project::new("eval-unlink", R24);
    let track = project.add_track(TrackKind::Sticker, "Overlays");
    let clips = [
        project
            .add_generated(
                track,
                Generator::SolidColor {
                    rgba: [255, 0, 0, 255],
                },
                TimeRange::at_rate(0, 24, R24),
            )
            .unwrap(),
        project
            .add_generated(
                track,
                Generator::SolidColor {
                    rgba: [0, 0, 255, 255],
                },
                TimeRange::at_rate(24, 24, R24),
            )
            .unwrap(),
    ];
    let link = LinkId::next();
    for clip in clips {
        project.timeline_mut().clip_mut(clip).unwrap().link = Some(link);
    }
    let mut host = EngineHost::new(project);
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![(
            "call_1",
            "unlink_clips",
            serde_json::json!({ "clips": [clips[1].raw()] }),
        )]),
        tool_turn(vec![("call_2", "commit_progress", serde_json::json!({}))]),
        text_turn("Unlinked the selected group."),
    ]);

    let (outcome, _) = run(
        &provider,
        &mut host,
        &EditorContext {
            selected_clips: vec![clips[1].raw()],
            ..Default::default()
        },
        "unlink the selected clip group and commit",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(outcome.actions.len(), 1);
    assert_eq!(outcome.phase_breaks, vec![1]);
    assert_eq!(
        outcome.actions[0].command,
        WireCommand::UnlinkClips(cutlass_ai::wire::UnlinkClips {
            clips: vec![clips[1].raw()]
        })
    );
    assert_eq!(
        outcome.actions[0].description,
        format!(
            "unlinked complete groups touched by clips {}",
            clips[1].raw()
        )
    );
    for clip in clips {
        assert_eq!(host.engine.project().clip(clip).unwrap().link, None);
    }

    assert!(host.engine.undo(), "the committed phase is one undo step");
    for clip in clips {
        assert_eq!(host.engine.project().clip(clip).unwrap().link, Some(link));
    }
    assert!(!host.engine.undo(), "the fixture itself created no history");
}

#[test]
fn model_corrects_course_after_a_rejection() {
    let (mut host, _, _, clip) = fixture();
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![(
            "call_1",
            "remove_clip",
            serde_json::json!({ "clip": 999 }),
        )]),
        tool_turn(vec![(
            "call_2",
            "remove_clip",
            serde_json::json!({ "clip": clip }),
        )]),
        text_turn("Removed the clip."),
    ]);

    let (outcome, _) = run(
        &provider,
        &mut host,
        &EditorContext::default(),
        "delete the clip",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(outcome.actions.len(), 1, "only the corrected call applied");
    assert_eq!(host.engine.project().timeline().clip_count(), 0);

    // The rejection went back to the model as the tool result, naming the
    // ids that do exist.
    let second_request = &provider.requests()[1];
    let last = second_request.last().unwrap();
    match last {
        Message::ToolResult {
            call_id, content, ..
        } => {
            assert_eq!(call_id, "call_1");
            assert!(
                content.contains("rejected: clip 999 does not exist"),
                "{content}"
            );
            assert!(content.contains(&clip.to_string()), "{content}");
        }
        other => panic!("expected tool result, got {other:?}"),
    }
}

#[test]
fn cap_trip_keeps_the_edits_already_applied() {
    let (mut host, _, _, clip) = fixture();
    let provider = ScriptedProvider::new(vec![tool_turn(vec![
        (
            "call_1",
            "split_clip",
            serde_json::json!({ "clip": clip, "at": 5.0 }),
        ),
        (
            "call_2",
            "add_track",
            serde_json::json!({ "kind": "text", "name": "Titles" }),
        ),
    ])]);

    let config = AgentConfig {
        max_tool_calls: 1,
        ..Default::default()
    };
    let (outcome, _) = run(
        &provider,
        &mut host,
        &EditorContext::default(),
        "go wild",
        &config,
    );

    // The cap does not fail the prompt: the run completes keeping the
    // in-cap edits, refusing the rest.
    assert_eq!(outcome.status, PromptStatus::Completed);
    assert!(outcome.text.contains("1-edit cap"), "{}", outcome.text);
    assert_eq!(outcome.actions.len(), 1, "only the in-cap edit applied");
    assert_eq!(host.engine.project().timeline().clip_count(), 2);
    assert_eq!(
        host.engine.project().timeline().track_count(),
        1,
        "the over-cap add_track was refused"
    );
    // The kept edits are one undoable history entry, like any prompt.
    assert!(host.engine.undo());
    assert_eq!(host.engine.project().timeline().clip_count(), 1);
    assert!(!host.engine.undo(), "exactly one history entry per prompt");
}

#[test]
fn questions_answer_without_editing() {
    let (mut host, _, _, _) = fixture();
    let provider = ScriptedProvider::new(vec![text_turn("The timeline is 10.00s long.")]);

    let (outcome, events) = run(
        &provider,
        &mut host,
        &EditorContext::default(),
        "how long is the timeline?",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert!(outcome.actions.is_empty());
    assert_eq!(outcome.text, "The timeline is 10.00s long.");
    assert!(events.iter().all(|e| matches!(e, AgentEvent::TextDelta(_))));
    assert!(
        !host.engine.undo(),
        "answering a question records no history"
    );
}

#[test]
fn which_clips_have_no_audio_answers_from_pushed_state() {
    // Two sources, one silent. The summary pushed into the system prompt
    // must already carry the facts ("which clips have no audio?" is the
    // roadmap's canonical Q&A example) so the model answers in one turn,
    // no tool calls.
    let mut project = Project::new("eval-audio", R24);
    let talk = project.add_media(MediaSource::new(
        "/tmp/talk.mp4",
        1920,
        1080,
        R24,
        60 * 24,
        true,
    ));
    let broll = project.add_media(MediaSource::new(
        "/tmp/broll.mp4",
        1920,
        1080,
        R24,
        60 * 24,
        false,
    ));
    let track = project.add_track(TrackKind::Video, "V1");
    project
        .add_clip(
            track,
            talk,
            TimeRange::at_rate(0, 120, R24),
            RationalTime::new(0, R24),
        )
        .unwrap();
    let silent_clip = project
        .add_clip(
            track,
            broll,
            TimeRange::at_rate(0, 120, R24),
            RationalTime::new(120, R24),
        )
        .unwrap()
        .raw();
    let mut host = EngineHost::new(project);

    let provider = ScriptedProvider::new(vec![text_turn(&format!(
        "Only clip {silent_clip} (broll.mp4, 5.00s–10.00s) has no audio."
    ))]);

    let (outcome, _) = run(
        &provider,
        &mut host,
        &EditorContext::default(),
        "which clips have no audio?",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert!(outcome.actions.is_empty());
    assert!(outcome.text.contains("broll.mp4"));
    assert!(!host.engine.undo(), "answering records no history");

    // One provider turn was enough, and the pushed state held the facts
    // plus the rule that says to answer from it.
    let requests = provider.requests();
    assert_eq!(requests.len(), 1, "answered without tool calls");
    match &requests[0][0] {
        Message::System { content } => {
            assert!(content.contains("\"has_audio\":false"), "{content}");
            assert!(content.contains("broll.mp4"), "{content}");
            assert!(content.contains("answer directly from"));
        }
        other => panic!("expected system message, got {other:?}"),
    }
}

#[test]
fn answer_only_turn_in_dry_run_yields_no_plan() {
    // With the preview toggle on, the UI shows an Apply/Discard card only
    // for a non-empty plan; a question must come back as zero actions so
    // no empty card (and no "Applied 0 edits") ever renders.
    let (mut host, _, _, _) = fixture();
    let provider = ScriptedProvider::new(vec![text_turn("The timeline runs 10.00s.")]);

    let config = AgentConfig {
        dry_run: true,
        ..Default::default()
    };
    let (outcome, events) = run(
        &provider,
        &mut host,
        &EditorContext::default(),
        "how long is the timeline?",
        &config,
    );

    assert_eq!(outcome.status, PromptStatus::DryRun);
    assert!(outcome.actions.is_empty());
    assert_eq!(outcome.text, "The timeline runs 10.00s.");
    assert!(events.iter().all(|e| matches!(e, AgentEvent::TextDelta(_))));
    assert!(!host.engine.undo(), "dry-run Q&A records no history");
}

#[test]
fn describe_project_feeds_state_back_without_counting_as_an_edit() {
    let (mut host, _, _, _) = fixture();
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![("call_1", "describe_project", serde_json::json!({}))]),
        text_turn("There is one clip on one video track."),
    ]);

    let (outcome, _) = run(
        &provider,
        &mut host,
        &EditorContext::default(),
        "what's in this project?",
        &AgentConfig {
            max_tool_calls: 0, // describe_project must not count against the cap
            ..Default::default()
        },
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert!(outcome.actions.is_empty());

    let second_request = &provider.requests()[1];
    match second_request.last().unwrap() {
        Message::ToolResult { content, .. } => {
            assert!(content.contains("\"project\""), "{content}");
            assert!(content.contains("eval.mp4"), "{content}");
        }
        other => panic!("expected tool result, got {other:?}"),
    }
}

#[test]
fn reasoning_summaries_stream_across_tool_rounds_without_entering_history() {
    let (mut host, _, _, clip) = fixture();
    let first_summary = "I should inspect the current timeline.";
    let second_summary = "The selected clip can be split safely.";
    let final_summary = "The edit succeeded, so I can answer.";
    let provider = ScriptedProvider::new(vec![
        ChatTurn {
            text: String::new(),
            reasoning_summary: first_summary.into(),
            tool_calls: vec![ToolCall {
                id: "inspect".into(),
                name: "describe_project".into(),
                arguments: serde_json::json!({}),
            }],
            finish: FinishReason::ToolCalls,
        },
        ChatTurn {
            text: String::new(),
            reasoning_summary: second_summary.into(),
            tool_calls: vec![ToolCall {
                id: "edit".into(),
                name: "split_clip".into(),
                arguments: serde_json::json!({"clip": clip, "at": 5.0}),
            }],
            finish: FinishReason::ToolCalls,
        },
        ChatTurn {
            text: "Split the clip at 5 seconds.".into(),
            reasoning_summary: final_summary.into(),
            tool_calls: Vec::new(),
            finish: FinishReason::Stop,
        },
    ]);

    let (outcome, events) = run(
        &provider,
        &mut host,
        &EditorContext::default(),
        "split the clip after inspecting it",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(outcome.actions.len(), 1);
    assert_eq!(host.engine.project().timeline().clip_count(), 2);
    let summaries = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ReasoningDelta(delta) => Some(delta.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(summaries, [first_summary, second_summary, final_summary]);
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::Action(_)))
    );
    assert!(events.iter().any(|event| {
        matches!(event, AgentEvent::TextDelta(delta) if delta == "Split the clip at 5 seconds.")
    }));

    let requests = provider.requests();
    for message in requests
        .iter()
        .flatten()
        .chain(outcome.turn_messages.iter())
    {
        let content = match message {
            Message::System { content }
            | Message::User { content, .. }
            | Message::Assistant { content, .. }
            | Message::ToolResult { content, .. } => content,
        };
        for summary in [first_summary, second_summary, final_summary] {
            assert!(
                !content.contains(summary),
                "reasoning summary leaked into model history: {content}"
            );
        }
    }
}

#[test]
fn dry_run_collects_the_plan_without_touching_the_engine() {
    let (mut host, _, _, clip) = fixture();
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![(
            "call_1",
            "trim_clip",
            serde_json::json!({ "clip": clip, "start": 3.0, "duration": 7.0 }),
        )]),
        text_turn("Planned one trim."),
    ]);

    let config = AgentConfig {
        dry_run: true,
        ..Default::default()
    };
    let (outcome, _) = run(
        &provider,
        &mut host,
        &EditorContext::default(),
        "cut the first 3 seconds",
        &config,
    );

    assert_eq!(outcome.status, PromptStatus::DryRun);
    assert_eq!(outcome.actions.len(), 1);
    assert_eq!(
        outcome.actions[0].command,
        WireCommand::TrimClip(cutlass_ai::wire::TrimClip {
            clip,
            start: 3.0,
            duration: 7.0,
        })
    );

    // Untouched: original placement, no history.
    let placed = host
        .engine
        .project()
        .clip(cutlass_models::ClipId::from_raw(clip))
        .unwrap();
    assert_eq!(placed.timeline, TimeRange::at_rate(0, 240, R24));
    assert!(!host.engine.undo());
}

/// A model simulator: creates a text track, reads the new track's id out
/// of the tool result (the way a real model does), then places the title
/// on it. Static scripts can't thread runtime ids; this can.
struct TitleAddingModel;

impl cutlass_ai::provider::ChatProvider for TitleAddingModel {
    fn chat(
        &self,
        request: &cutlass_ai::provider::ChatRequest<'_>,
        _cancel: &AtomicBool,
        _on_event: &mut dyn FnMut(cutlass_ai::provider::ProviderStreamEvent<'_>),
    ) -> Result<ChatTurn, cutlass_ai::provider::ProviderError> {
        let last = request.messages.last().unwrap();
        Ok(match last {
            Message::User { .. } => tool_turn(vec![(
                "call_1",
                "add_track",
                serde_json::json!({ "kind": "text", "name": "Titles" }),
            )]),
            Message::ToolResult {
                call_id, content, ..
            } if call_id == "call_1" => {
                // "ok: added text track 'Titles' (track 42)"
                let id: u64 = content
                    .rsplit("(track ")
                    .next()
                    .and_then(|s| s.trim_end_matches(')').parse().ok())
                    .expect("track id in tool result");
                tool_turn(vec![(
                    "call_2",
                    "add_generated",
                    serde_json::json!({
                        "track": id,
                        "generator": { "type": "text", "content": "INTRO" },
                        "start": 0.0,
                        "duration": 3.0,
                    }),
                )])
            }
            _ => text_turn("Added the INTRO title."),
        })
    }
}

#[test]
fn add_a_title_that_says_intro() {
    let (mut host, _, _, _) = fixture();
    let (outcome, _) = run_with(
        &TitleAddingModel,
        &mut host,
        &mut NullToolHost,
        &EditorContext::default(),
        "add a title that says INTRO",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(outcome.actions.len(), 2);
    assert!(
        outcome.actions[1]
            .description
            .starts_with("added text 'INTRO' at 0.00s for 3.00s"),
        "{}",
        outcome.actions[1].description
    );

    let summary = summarize(host.engine.project());
    let titles = summary
        .tracks
        .iter()
        .find(|t| t.name == "Titles")
        .expect("titles track");
    assert_eq!(titles.clips.len(), 1);

    // One undo removes both the clip and the track.
    assert!(host.engine.undo());
    assert!(
        summarize(host.engine.project())
            .tracks
            .iter()
            .all(|t| t.name != "Titles")
    );
    assert!(!host.engine.undo());
}

/// Creates the required audio lane first, then threads that runtime track id
/// into the explicit-target extraction tool call.
struct AudioExtractingModel {
    clip: u64,
}

impl cutlass_ai::provider::ChatProvider for AudioExtractingModel {
    fn chat(
        &self,
        request: &cutlass_ai::provider::ChatRequest<'_>,
        _cancel: &AtomicBool,
        _on_event: &mut dyn FnMut(cutlass_ai::provider::ProviderStreamEvent<'_>),
    ) -> Result<ChatTurn, cutlass_ai::provider::ProviderError> {
        let last = request.messages.last().unwrap();
        Ok(match last {
            Message::User { .. } => tool_turn(vec![(
                "extract_track",
                "add_track",
                serde_json::json!({ "kind": "audio", "name": "Extracted" }),
            )]),
            Message::ToolResult {
                call_id, content, ..
            } if call_id == "extract_track" => {
                let track: u64 = content
                    .rsplit("(track ")
                    .next()
                    .and_then(|text| text.trim_end_matches(')').parse().ok())
                    .expect("created track id in tool result");
                tool_turn(vec![(
                    "extract_clip",
                    "extract_audio",
                    serde_json::json!({ "clip": self.clip, "track": track }),
                )])
            }
            _ => text_turn("Extracted the clip's audio."),
        })
    }
}

#[test]
fn extract_audio_after_creating_its_explicit_track_is_one_prompt_undo() {
    let (mut host, _, _, clip) = fixture();
    let model = AudioExtractingModel { clip };
    let (outcome, _) = run_with(
        &model,
        &mut host,
        &mut NullToolHost,
        &EditorContext {
            selected_clips: vec![clip],
            ..Default::default()
        },
        "extract the selected clip's audio",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(outcome.actions.len(), 2);
    assert!(matches!(
        outcome.actions[1].command,
        WireCommand::ExtractAudio(_)
    ));
    assert!(
        outcome.actions[1]
            .description
            .starts_with(&format!("extracted audio from clip {clip} onto track"))
    );
    let audio = host
        .engine
        .project()
        .timeline()
        .tracks_ordered()
        .find(|track| track.kind == TrackKind::Audio)
        .and_then(|track| track.clips().next())
        .expect("extracted audio");
    assert_eq!(audio.audio_role, Some(cutlass_models::AudioRole::Extracted));
    assert!(
        !host
            .engine
            .project()
            .timeline()
            .carries_own_audio(cutlass_models::ClipId::from_raw(clip))
    );

    assert!(host.engine.undo(), "one prompt is one undo entry");
    assert!(
        host.engine
            .project()
            .timeline()
            .tracks_ordered()
            .all(|track| track.kind != TrackKind::Audio)
    );
    assert!(
        host.engine
            .project()
            .timeline()
            .carries_own_audio(cutlass_models::ClipId::from_raw(clip))
    );
    assert!(!host.engine.undo());
}

#[test]
fn delete_every_clip_on_the_music_track() {
    // Fixture with a second (audio) track holding three clips.
    let mut project = Project::new("eval-music", R24);
    let media = project.add_media(MediaSource::new(
        "/tmp/music.mp3",
        0,
        0,
        R24,
        120 * 24,
        true,
    ));
    project.add_track(TrackKind::Video, "V1");
    let music = project.add_track(TrackKind::Audio, "Music");
    let clips: Vec<u64> = (0..3)
        .map(|i| {
            project
                .add_clip(
                    music,
                    media,
                    TimeRange::at_rate(0, 120, R24),
                    RationalTime::new(i * 150, R24),
                )
                .unwrap()
                .raw()
        })
        .collect();
    let mut host = EngineHost::new(project);

    let provider = ScriptedProvider::new(vec![
        tool_turn(
            clips
                .iter()
                .enumerate()
                .map(|(i, clip)| {
                    (
                        match i {
                            0 => "call_1",
                            1 => "call_2",
                            _ => "call_3",
                        },
                        "remove_clip",
                        serde_json::json!({ "clip": clip }),
                    )
                })
                .collect(),
        ),
        text_turn("Cleared the music track."),
    ]);

    let (outcome, _) = run(
        &provider,
        &mut host,
        &EditorContext::default(),
        "delete every clip on the music track",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(outcome.actions.len(), 3);
    let summary = summarize(host.engine.project());
    let music_track = summary.tracks.iter().find(|t| t.name == "Music").unwrap();
    assert!(music_track.clips.is_empty());

    // One undo brings all three back.
    assert!(host.engine.undo());
    let summary = summarize(host.engine.project());
    let music_track = summary.tracks.iter().find(|t| t.name == "Music").unwrap();
    assert_eq!(music_track.clips.len(), 3);
}

#[test]
fn lower_music_volume_with_fades() {
    // Fixture with an audio lane holding one music clip.
    let mut project = Project::new("eval-volume", R24);
    let media = project.add_media(MediaSource::new(
        "/tmp/music.mp3",
        0,
        0,
        R24,
        120 * 24,
        true,
    ));
    project.add_track(TrackKind::Video, "V1");
    let music = project.add_track(TrackKind::Audio, "Music");
    let clip = project
        .add_clip(
            music,
            media,
            TimeRange::at_rate(0, 240, R24),
            RationalTime::new(0, R24),
        )
        .unwrap()
        .raw();
    let mut host = EngineHost::new(project);

    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![(
            "call_1",
            "set_clip_audio",
            serde_json::json!({
                "clip": clip, "volume": 0.5, "fade_in": 1.0, "fade_out": 2.0,
            }),
        )]),
        text_turn("Lowered the music to 50% with fades."),
    ]);

    let (outcome, _) = run(
        &provider,
        &mut host,
        &EditorContext::default(),
        "lower the music to half volume and fade it in and out",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(outcome.actions.len(), 1);
    assert_eq!(
        outcome.actions[0].description,
        format!("set clip {clip} volume 50%, fade in 1.00s, fade out 2.00s")
    );

    let clip_id = cutlass_models::ClipId::from_raw(clip);
    let c = host.engine.project().clip(clip_id).unwrap();
    assert_eq!(c.volume.constant(), Some(0.5));
    assert_eq!((c.fade_in, c.fade_out), (24, 48));
    // The summary the next prompt would see carries the new mix.
    let summary = summarize(host.engine.project());
    let summarized = &summary
        .tracks
        .iter()
        .find(|t| t.name == "Music")
        .unwrap()
        .clips[0];
    assert_eq!(summarized.volume, Some(0.5));
    assert_eq!(summarized.fade_in, Some(1.0));

    // One undo restores the default mix.
    assert!(host.engine.undo());
    assert!(
        !host
            .engine
            .project()
            .clip(clip_id)
            .unwrap()
            .has_custom_audio()
    );
}

#[test]
fn volume_envelope_with_keyframes() {
    // Fixture with an audio lane holding one music clip (10s at 24fps).
    let mut project = Project::new("eval-envelope", R24);
    let media = project.add_media(MediaSource::new(
        "/tmp/music.mp3",
        0,
        0,
        R24,
        120 * 24,
        true,
    ));
    let music = project.add_track(TrackKind::Audio, "Music");
    let clip = project
        .add_clip(
            music,
            media,
            TimeRange::at_rate(0, 240, R24),
            RationalTime::new(0, R24),
        )
        .unwrap()
        .raw();
    let mut host = EngineHost::new(project);

    // "duck the music down to 20% between 2s and 4s" → a volume envelope.
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![
            (
                "call_1",
                "set_param_keyframe",
                serde_json::json!({ "clip": clip, "param": "volume", "at": 2.0, "value": 1.0 }),
            ),
            (
                "call_2",
                "set_param_keyframe",
                serde_json::json!({ "clip": clip, "param": "volume", "at": 3.0, "value": 0.2 }),
            ),
            (
                "call_3",
                "set_param_keyframe",
                serde_json::json!({ "clip": clip, "param": "volume", "at": 4.0, "value": 1.0 }),
            ),
        ]),
        text_turn("Ducked the music to 20% from 2 to 4 seconds."),
    ]);

    let (outcome, _) = run(
        &provider,
        &mut host,
        &EditorContext::default(),
        "duck the music to 20% between 2 and 4 seconds",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(outcome.actions.len(), 3);
    assert_eq!(
        outcome.actions[1].description,
        format!("keyframed clip {clip} volume = 20% at 3.00s")
    );

    // The envelope landed: a keyframed volume that dips at 3s (72 ticks).
    let clip_id = cutlass_models::ClipId::from_raw(clip);
    let c = host.engine.project().clip(clip_id).unwrap();
    assert!(c.has_volume_envelope());
    assert_eq!(c.volume.keyframes().len(), 3);
    assert_eq!(c.volume.sample(48), 1.0);
    assert_eq!(c.volume.sample(72), 0.2);
    assert_eq!(c.volume.sample(96), 1.0);

    // One prompt = one undo: the whole envelope disappears as a unit.
    assert!(host.engine.undo());
    assert!(
        !host
            .engine
            .project()
            .clip(clip_id)
            .unwrap()
            .has_volume_envelope()
    );
}

#[test]
fn pan_envelope_with_keyframes() {
    let mut project = Project::new("eval-pan", R24);
    let media = project.add_media(MediaSource::new(
        "/tmp/music.mp3",
        0,
        0,
        R24,
        120 * 24,
        true,
    ));
    let music = project.add_track(TrackKind::Audio, "Music");
    let clip = project
        .add_clip(
            music,
            media,
            TimeRange::at_rate(0, 240, R24),
            RationalTime::new(0, R24),
        )
        .unwrap()
        .raw();
    let mut host = EngineHost::new(project);

    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![
            (
                "call_1",
                "set_param_keyframe",
                serde_json::json!({ "clip": clip, "param": "pan", "at": 0.0, "value": -1.0 }),
            ),
            (
                "call_2",
                "set_param_keyframe",
                serde_json::json!({ "clip": clip, "param": "pan", "at": 5.0, "value": 1.0 }),
            ),
        ]),
        text_turn("Swept the music from left to right."),
    ]);

    let (outcome, _) = run(
        &provider,
        &mut host,
        &EditorContext::default(),
        "pan the music from left to right over five seconds",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(outcome.actions.len(), 2);
    assert_eq!(
        outcome.actions[0].description,
        format!("keyframed clip {clip} pan = -1.00 at 0.00s")
    );

    let clip_id = cutlass_models::ClipId::from_raw(clip);
    let c = host.engine.project().clip(clip_id).unwrap();
    assert!(c.has_pan_envelope());
    assert_eq!(c.pan.sample(0), -1.0);
    assert_eq!(c.pan.sample(120), 1.0);
}

#[test]
fn fade_in_with_opacity_keyframes() {
    let (mut host, _, _, clip) = fixture();
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![
            (
                "call_1",
                "set_param_keyframe",
                serde_json::json!({
                    "clip": clip, "param": "opacity", "at": 0.0,
                    "value": 0.0, "easing": "ease_in_out",
                }),
            ),
            (
                "call_2",
                "set_param_keyframe",
                serde_json::json!({
                    "clip": clip, "param": "opacity", "at": 1.0, "value": 1.0,
                }),
            ),
        ]),
        text_turn("Added a 1-second fade-in."),
    ]);

    let (outcome, _) = run(
        &provider,
        &mut host,
        &EditorContext::default(),
        "fade the clip in over the first second",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(outcome.actions.len(), 2);
    assert_eq!(
        outcome.actions[0].description,
        format!("keyframed clip {clip} opacity = 0% at 0.00s")
    );
    assert_eq!(
        outcome.actions[1].description,
        format!("keyframed clip {clip} opacity = 100% at 1.00s")
    );

    // The curve landed: 0 → 1 over the first 24 ticks, eased.
    let placed = host
        .engine
        .project()
        .clip(cutlass_models::ClipId::from_raw(clip))
        .unwrap();
    assert!(placed.transform.is_animated());
    assert_eq!(placed.transform.opacity.keyframes().len(), 2);
    assert_eq!(placed.transform.sample(0).opacity, 0.0);
    assert_eq!(placed.transform.sample(24).opacity, 1.0);

    // One prompt = one undo: the animation disappears as a unit.
    assert!(host.engine.undo());
    let restored = host
        .engine
        .project()
        .clip(cutlass_models::ClipId::from_raw(clip))
        .unwrap();
    assert!(!restored.transform.is_animated());
    assert!(!host.engine.undo());
}

#[test]
fn add_fade_in_animation_preset() {
    let (mut host, _, _, clip) = fixture();
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![(
            "call_1",
            "set_clip_animation",
            serde_json::json!({
                "clip": clip,
                "slot": "in",
                "animation": "fade_in",
            }),
        )]),
        text_turn("Added a fade-in animation."),
    ]);

    let (outcome, _) = run(
        &provider,
        &mut host,
        &EditorContext::default(),
        "add a fade in animation to the clip",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(outcome.actions.len(), 1);
    assert_eq!(
        outcome.actions[0].description,
        format!("set fade_in animation on clip {clip} (in slot)")
    );

    let placed = host
        .engine
        .project()
        .clip(cutlass_models::ClipId::from_raw(clip))
        .unwrap();
    assert_eq!(
        placed.animation_in.as_ref().map(|a| a.id.as_str()),
        Some("fade_in")
    );

    assert!(host.engine.undo());
    let restored = host
        .engine
        .project()
        .clip(cutlass_models::ClipId::from_raw(clip))
        .unwrap();
    assert!(restored.animation_in.is_none());
    assert!(!host.engine.undo());
}

#[test]
fn speed_up_and_reverse_clip() {
    let (mut host, _, _, clip) = fixture();
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![(
            "call_1",
            "set_clip_speed",
            serde_json::json!({ "clip": clip, "speed": 2.0, "reversed": true }),
        )]),
        text_turn("Doubled the speed and reversed it."),
    ]);

    let (outcome, _) = run(
        &provider,
        &mut host,
        &EditorContext::default(),
        "play the clip backwards at double speed",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(outcome.actions.len(), 1);
    assert_eq!(
        outcome.actions[0].description,
        format!("set clip {clip} speed 2x, reversed")
    );

    // 10 s of source at 2x occupies 5 s of timeline (120 ticks @ 24 fps),
    // and the retiming shows up in the next describe() the model sees.
    let placed = host
        .engine
        .project()
        .clip(cutlass_models::ClipId::from_raw(clip))
        .unwrap();
    assert_eq!(placed.timeline.duration.value, 120);
    assert!(placed.reversed);
    let summary = summarize(host.engine.project());
    let described = &summary.tracks[0].clips[0];
    assert_eq!(described.speed, Some(2.0));
    assert_eq!(described.reversed, Some(true));

    // One undo restores the original 1x forward placement.
    assert!(host.engine.undo());
    let restored = host
        .engine
        .project()
        .clip(cutlass_models::ClipId::from_raw(clip))
        .unwrap();
    assert_eq!(restored.timeline.duration.value, 240);
    assert!(!restored.is_retimed());
}

#[test]
fn apply_and_clear_speed_ramp_on_clip() {
    let (mut host, _, _, clip) = fixture();
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![(
            "call_1",
            "set_speed_curve",
            serde_json::json!({ "clip": clip, "preset": "montage" }),
        )]),
        text_turn("Added a montage speed ramp."),
    ]);

    let (outcome, _) = run(
        &provider,
        &mut host,
        &EditorContext::default(),
        "give the clip a montage speed ramp",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(outcome.actions.len(), 1);
    assert_eq!(
        outcome.actions[0].description,
        format!("applied montage speed ramp to clip {clip}")
    );

    // The ramp lands, retimes the clip (montage averages faster than 1×, so
    // shorter than the 240-tick original), and surfaces in the next describe.
    let placed = host
        .engine
        .project()
        .clip(cutlass_models::ClipId::from_raw(clip))
        .unwrap();
    assert!(placed.has_speed_curve() && placed.is_retimed());
    assert!(placed.timeline.duration.value < 240);
    let summary = summarize(host.engine.project());
    assert_eq!(summary.tracks[0].clips[0].speed_ramp, Some(true));

    // One undo restores the original constant-speed placement.
    assert!(host.engine.undo());
    let restored = host
        .engine
        .project()
        .clip(cutlass_models::ClipId::from_raw(clip))
        .unwrap();
    assert_eq!(restored.timeline.duration.value, 240);
    assert!(!restored.is_retimed());
}

#[test]
fn crop_to_center_and_mirror_clip() {
    let (mut host, _, _, clip) = fixture();
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![(
            "call_1",
            "set_clip_crop",
            serde_json::json!({ "clip": clip, "left": 0.25, "right": 0.25, "flip_h": true }),
        )]),
        text_turn("Cropped to the middle half and mirrored it."),
    ]);

    let (outcome, _) = run(
        &provider,
        &mut host,
        &EditorContext::default(),
        "crop to the center half and mirror it",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(outcome.actions.len(), 1);
    assert_eq!(
        outcome.actions[0].description,
        format!("set clip {clip} cropped left 25%, right 25%, flipped horizontally")
    );

    // The kept region and flip land on the model, and the next describe()
    // surfaces them so the model can reason about current framing.
    let placed = host
        .engine
        .project()
        .clip(cutlass_models::ClipId::from_raw(clip))
        .unwrap();
    assert_eq!(placed.crop.sample(0).x, 0.25);
    assert_eq!(placed.crop.sample(0).w, 0.5);
    assert!(placed.flip_h && !placed.flip_v);
    let summary = summarize(host.engine.project());
    let described = &summary.tracks[0].clips[0];
    assert_eq!(described.crop, Some([0.25, 0.0, 0.25, 0.0]));
    assert_eq!(described.flip_h, Some(true));
    assert_eq!(described.flip_v, None);

    // One undo restores the full frame.
    assert!(host.engine.undo());
    let restored = host
        .engine
        .project()
        .clip(cutlass_models::ClipId::from_raw(clip))
        .unwrap();
    assert!(!restored.has_custom_crop());
}

#[test]
fn add_a_blur_to_the_clip() {
    let (mut host, _, _, clip) = fixture();
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![
            (
                "call_1",
                "add_effect",
                serde_json::json!({ "clip": clip, "effect": "gaussian_blur" }),
            ),
            (
                "call_2",
                "set_effect_param",
                serde_json::json!({ "clip": clip, "index": 0, "param": "radius", "value": 6.0 }),
            ),
        ]),
        text_turn("Added a gaussian blur and set its radius to 6."),
    ]);

    let context = EditorContext {
        selected_clips: vec![clip],
        ..Default::default()
    };
    let (outcome, _) = run(
        &provider,
        &mut host,
        &context,
        "add a blur to the clip",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(outcome.actions.len(), 2);
    assert_eq!(
        outcome.actions[0].description,
        format!("added gaussian_blur effect to clip {clip}")
    );

    // The effect landed and surfaces in the summary the next prompt sees,
    // with the index the model would address.
    let placed = host
        .engine
        .project()
        .clip(cutlass_models::ClipId::from_raw(clip))
        .unwrap();
    assert_eq!(placed.effects.len(), 1);
    assert_eq!(placed.effects[0].effect_id, "gaussian_blur");
    let summary = summarize(host.engine.project());
    let described = &summary.tracks[0].clips[0];
    assert_eq!(described.effects.len(), 1);
    assert_eq!(described.effects[0].effect, "gaussian_blur");
    assert_eq!(described.effects[0].params.get("radius"), Some(&6.0));

    // One prompt = one undo: the whole effect disappears as a unit.
    assert!(host.engine.undo());
    let restored = host
        .engine
        .project()
        .clip(cutlass_models::ClipId::from_raw(clip))
        .unwrap();
    assert!(restored.effects.is_empty());
    assert!(!host.engine.undo());
}

#[test]
fn crossfade_between_two_clips() {
    let mut project = Project::new("eval", R24);
    let media = project.add_media(MediaSource::new(
        "/tmp/eval.mp4",
        1920,
        1080,
        R24,
        60 * 24,
        true,
    ));
    let track = project.add_track(TrackKind::Video, "V1");
    let first = project
        .add_clip(
            track,
            media,
            TimeRange::at_rate(0, 120, R24),
            RationalTime::new(0, R24),
        )
        .unwrap()
        .raw();
    let _second = project
        .add_clip(
            track,
            media,
            TimeRange::at_rate(0, 120, R24),
            RationalTime::new(120, R24),
        )
        .unwrap();
    let mut host = EngineHost::new(project);

    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![
            (
                "call_1",
                "add_transition",
                serde_json::json!({ "clip": first, "transition": "crossfade" }),
            ),
            (
                "call_2",
                "set_transition",
                serde_json::json!({ "clip": first, "seconds": 0.5 }),
            ),
        ]),
        text_turn("Added a half-second crossfade between the two clips."),
    ]);

    let (outcome, _) = run(
        &provider,
        &mut host,
        &EditorContext::default(),
        "crossfade between the two clips",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(outcome.actions.len(), 2);
    assert_eq!(
        outcome.actions[0].description,
        format!("added crossfade transition after clip {first}")
    );

    // The junction landed and surfaces on the left clip in the next summary.
    let summary = summarize(host.engine.project());
    let described = &summary.tracks[0].clips[0];
    assert_eq!(described.transition.as_deref(), Some("crossfade"));

    // One prompt = one undo: the whole transition disappears as a unit.
    assert!(host.engine.undo());
    let summary = summarize(host.engine.project());
    assert_eq!(summary.tracks[0].clips[0].transition, None);
    assert!(!host.engine.undo());
}

#[test]
fn add_marker_at_playhead() {
    let (mut host, _, _, _) = fixture();
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![(
            "call_1",
            "add_marker",
            serde_json::json!({ "at": 5.0, "name": "beat drop", "color": "red" }),
        )]),
        text_turn("Dropped a red marker at the beat."),
    ]);

    let (outcome, _) = run(
        &provider,
        &mut host,
        &EditorContext::default(),
        "mark the beat drop at 5 seconds",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(outcome.actions.len(), 1);
    assert!(
        outcome.actions[0]
            .description
            .contains("added marker 'beat drop' at 5.00s"),
        "{}",
        outcome.actions[0].description
    );
    assert!(
        outcome.actions[0].description.contains("(red)"),
        "{}",
        outcome.actions[0].description
    );

    let markers = host.engine.project().timeline().markers();
    assert_eq!(markers.len(), 1);
    assert_eq!(markers[0].tick.value, 120);
    assert_eq!(markers[0].name, "beat drop");
    assert_eq!(markers[0].color, cutlass_models::MarkerColor::Red);

    let summary = summarize(host.engine.project());
    assert_eq!(summary.markers.len(), 1);
    assert_eq!(summary.markers[0].name, "beat drop");
    assert_eq!(summary.markers[0].color, "red");

    assert!(host.engine.undo());
    assert!(host.engine.project().timeline().markers().is_empty());
}

#[test]
fn keyframe_outside_clip_is_rejected_with_extent() {
    let (mut host, _, _, clip) = fixture();
    // First call misses the clip (it ends at 10 s); the model corrects.
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![(
            "call_1",
            "set_param_keyframe",
            serde_json::json!({
                "clip": clip, "param": "scale", "at": 30.0, "value": 2.0,
            }),
        )]),
        tool_turn(vec![(
            "call_2",
            "set_param_keyframe",
            serde_json::json!({
                "clip": clip, "param": "scale", "at": 9.0, "value": 2.0,
            }),
        )]),
        text_turn("Keyframed the zoom at 9 seconds."),
    ]);

    let (outcome, _) = run(
        &provider,
        &mut host,
        &EditorContext::default(),
        "zoom in at the end of the clip",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(outcome.actions.len(), 1, "only the corrected call applied");

    // The rejection named the clip's extent so the model could correct.
    let requests = provider.requests();
    let tool_results: Vec<&str> = requests
        .iter()
        .flat_map(|msgs| msgs.iter())
        .filter_map(|m| match m {
            Message::ToolResult { content, .. } => Some(content.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        tool_results
            .iter()
            .any(|r| r.contains("outside clip") && r.contains("10.000s")),
        "rejection names the extent: {tool_results:?}"
    );
}

#[test]
fn provider_failure_mid_prompt_rolls_back() {
    let (mut host, _, _, clip) = fixture();
    // One successful edit turn, then the script runs dry — which the loop
    // sees as a provider error on the next turn.
    let provider = ScriptedProvider::new(vec![tool_turn(vec![(
        "call_1",
        "split_clip",
        serde_json::json!({ "clip": clip, "at": 5.0 }),
    )])]);

    let (outcome, _) = run(
        &provider,
        &mut host,
        &EditorContext::default(),
        "split the clip",
        &AgentConfig::default(),
    );

    assert!(matches!(outcome.status, PromptStatus::Aborted(_)));
    assert_eq!(
        host.engine.project().timeline().clip_count(),
        1,
        "the applied split must be rolled back"
    );
    assert!(!host.engine.undo());
}

fn message_kind(m: &Message) -> &'static str {
    match m {
        Message::System { .. } => "system",
        Message::User { .. } => "user",
        Message::Assistant { .. } => "assistant",
        Message::ToolResult { .. } => "tool",
    }
}

/// Multi-turn memory: the first prompt's `turn_messages` carry the whole
/// turn, and threading them into the next prompt puts the prior dialogue
/// into the request — behind a freshly regenerated system prompt — so the
/// model can answer "what did you just do?".
#[test]
fn session_history_threads_prior_turns_into_the_next_prompt() {
    let (mut host, _media, _track, clip) = fixture();
    let ctx = EditorContext {
        selected_clips: vec![clip],
        ..Default::default()
    };
    let cancel = AtomicBool::new(false);

    let first = ScriptedProvider::new(vec![
        tool_turn(vec![(
            "call_1",
            "split_clip",
            serde_json::json!({ "clip": clip, "at": 4.0 }),
        )]),
        text_turn("Split the clip at 4.00s into two."),
    ]);
    let outcome1 = run_prompt_with_host(
        &first,
        &mut host,
        &mut NullToolHost,
        &ctx,
        &cutlass_ai::AgentExtensions::default(),
        &[],
        "split the selected clip in half",
        &AgentConfig::default(),
        &cancel,
        &mut |_| {},
    );
    assert_eq!(outcome1.status, PromptStatus::Completed);

    // The turn carries: the user prompt, the assistant's tool call, the
    // tool result, and the final text answer.
    let kinds: Vec<&str> = outcome1.turn_messages.iter().map(message_kind).collect();
    assert_eq!(kinds, ["user", "assistant", "tool", "assistant"]);

    let second = ScriptedProvider::new(vec![text_turn("I split it into two clips.")]);
    let _ = run_prompt_with_host(
        &second,
        &mut host,
        &mut NullToolHost,
        &ctx,
        &cutlass_ai::AgentExtensions::default(),
        &outcome1.turn_messages,
        "what did you just do?",
        &AgentConfig::default(),
        &cancel,
        &mut |_| {},
    );

    let sent = second.requests();
    assert_eq!(sent.len(), 1, "the second prompt is one provider call");
    let convo = &sent[0];
    assert!(
        matches!(convo[0], Message::System { .. }),
        "a fresh system prompt leads every request"
    );
    assert!(
        convo.iter().any(|m| matches!(
            m,
            Message::User { content, .. } if content == "split the selected clip in half"
        )),
        "the prior user turn is remembered"
    );
    assert!(
        matches!(
            convo.last().unwrap(),
            Message::User { content, .. } if content == "what did you just do?"
        ),
        "the newest user message comes last"
    );
}

/// Skill retrieval-and-follow: with skills loaded, the index (names +
/// descriptions only) enters the system prompt, `read_skill` returns the
/// body without counting as an edit, and the model then edits per the
/// procedure — the whole prompt still one undo group.
#[test]
fn read_skill_feeds_the_procedure_then_edits_follow() {
    let (mut host, _media, _track, clip) = fixture();
    let cancel = AtomicBool::new(false);
    let extensions = cutlass_ai::AgentExtensions {
        rules: "[user]\nprefer tight cuts".into(),
        skills: vec![cutlass_ai::Skill {
            id: "tight-open".into(),
            name: "Tight opening".into(),
            description: "Trim the first seconds off the opening clip.".into(),
            body: "Step 1: trim_clip the first clip, raising start by 3.".into(),
        }],
    };
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![(
            "call_1",
            "read_skill",
            serde_json::json!({ "id": "tight-open" }),
        )]),
        tool_turn(vec![(
            "call_2",
            "trim_clip",
            serde_json::json!({ "clip": clip, "start": 3.0, "duration": 7.0 }),
        )]),
        text_turn("Tightened the opening per the skill."),
    ]);

    let cfg = AgentConfig {
        max_tool_calls: 1, // read_skill must not consume the only edit slot
        ..AgentConfig::default()
    };
    let mut events = Vec::new();
    let outcome = run_prompt_with_host(
        &provider,
        &mut host,
        &mut NullToolHost,
        &EditorContext::default(),
        &extensions,
        &[],
        "tighten the opening",
        &cfg,
        &cancel,
        &mut |e| events.push(e),
    );
    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(outcome.actions.len(), 1);
    assert!(outcome.actions[0].description.starts_with("trimmed clip"));

    // The system prompt carried the rules and the index, not the body.
    let sent = provider.requests();
    let Message::System { content } = &sent[0][0] else {
        panic!("first message is the system prompt");
    };
    assert!(content.contains("prefer tight cuts"));
    assert!(content.contains("tight-open (Tight opening)"));
    assert!(!content.contains("Step 1: trim_clip"));

    // The read_skill tool result delivered the body verbatim.
    let body_result = sent[1].iter().find_map(|m| match m {
        Message::ToolResult { content, .. } => Some(content.clone()),
        _ => None,
    });
    assert!(
        body_result
            .expect("read_skill result in the second request")
            .contains("Step 1: trim_clip"),
    );

    assert!(host.engine.undo(), "one undo entry for the whole prompt");
}

/// `describe_project` results are large and the fresh system snapshot
/// supersedes them, so history keeps only a placeholder — never a full
/// stale project blob.
#[test]
fn describe_project_results_are_collapsed_in_history() {
    let (mut host, _media, _track, _clip) = fixture();
    let cancel = AtomicBool::new(false);
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![("call_1", "describe_project", serde_json::json!({}))]),
        text_turn("There is one clip on one video track."),
    ]);
    let outcome = run_prompt_with_host(
        &provider,
        &mut host,
        &mut NullToolHost,
        &EditorContext::default(),
        &cutlass_ai::AgentExtensions::default(),
        &[],
        "what's on the timeline?",
        &AgentConfig::default(),
        &cancel,
        &mut |_| {},
    );
    assert_eq!(outcome.status, PromptStatus::Completed);

    let tool_result = outcome
        .turn_messages
        .iter()
        .find_map(|m| match m {
            Message::ToolResult { content, .. } => Some(content.clone()),
            _ => None,
        })
        .expect("the describe_project tool result");
    assert!(
        tool_result.contains("project state omitted"),
        "the blob is collapsed: {tool_result}"
    );
    assert!(
        !tool_result.contains("\"tracks\""),
        "no full project json survives in history"
    );
}

/// A bridge-owned sense runs against the already-mutated rehearsal state,
/// which is the key distinction from a host screenshot of the live project.
#[test]
fn engine_sense_observes_rehearsed_edits_and_returns_images() {
    let (mut host, _, _, clip) = fixture();
    host.sense_specs = vec![host_spec("media_screenshot_preview")];
    host.sense_outputs.push_back(Ok(ToolOutput {
        text: "sandbox preview at 5.00s".into(),
        images: vec![ImagePart::png(vec![7, 8, 9], "sandbox preview")],
    }));
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![(
            "call_1",
            "split_clip",
            serde_json::json!({ "clip": clip, "at": 5.0 }),
        )]),
        tool_turn(vec![(
            "call_2",
            "media_screenshot_preview",
            serde_json::json!({ "at": 5.0 }),
        )]),
        text_turn("The rehearsed split looks correct."),
    ]);

    let (outcome, events) = run(
        &provider,
        &mut host,
        &EditorContext::default(),
        "split the clip and check the result",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(outcome.actions.len(), 1);
    assert_eq!(host.sense_clip_counts, vec![2]);
    let requests = provider.requests();
    let Message::System { content } = &requests[0][0] else {
        panic!("first message should be the system prompt");
    };
    assert!(
        content.contains("inspect the complete current project snapshot and sandbox"),
        "{content}"
    );
    assert!(content.contains("schematic timeline map"), "{content}");
    assert!(content.contains("media_screenshot_preview"), "{content}");
    assert_eq!(
        host.sense_calls,
        vec![(
            "media_screenshot_preview".into(),
            serde_json::json!({ "at": 5.0 })
        )]
    );
    match requests[2].last().unwrap() {
        Message::ToolResult {
            content, images, ..
        } => {
            assert_eq!(content, "sandbox preview at 5.00s");
            assert_eq!(images[0].label, "sandbox preview");
            assert_eq!(*images[0].data, vec![7, 8, 9]);
        }
        other => panic!("expected sense result, got {other:?}"),
    }
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::HostAction { name, .. } if name == "media_screenshot_preview"
    )));
    assert!(events.iter().any(
        |event| matches!(event, AgentEvent::Image(image) if image.label == "sandbox preview")
    ));
}

#[test]
fn media_pool_sheet_sense_dispatches_and_survives_the_image_budget() {
    let (mut host, _, _, _) = fixture();
    host.sense_specs = vec![host_spec("media_pool_sheet")];
    host.sense_outputs.push_back(Ok(ToolOutput {
        text: "Media-pool contact sheet page 1 of 2.".into(),
        images: vec![ImagePart::png(
            vec![0x89, 0x50, 0x4e, 0x47],
            "media pool sheet page 1 of 2",
        )],
    }));
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![(
            "call_1",
            "media_pool_sheet",
            serde_json::json!({ "page": 1, "max_width": 1024 }),
        )]),
        text_turn("I found footage for the freestyle edit."),
    ]);
    let config = AgentConfig {
        max_images: 1,
        max_image_bytes: 16,
        ..AgentConfig::default()
    };

    let (outcome, events) = run(
        &provider,
        &mut host,
        &EditorContext::default(),
        "freestyle an edit from my media library",
        &config,
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(
        host.sense_calls,
        vec![(
            "media_pool_sheet".into(),
            serde_json::json!({ "page": 1, "max_width": 1024 })
        )]
    );
    let requests = provider.requests();
    let Message::System { content } = &requests[0][0] else {
        panic!("first message should be the system prompt");
    };
    assert!(content.contains("freestyle edits"), "{content}");
    assert!(content.contains("media_pool_sheet"), "{content}");
    match requests[1].last().unwrap() {
        Message::ToolResult {
            content, images, ..
        } => {
            assert_eq!(content, "Media-pool contact sheet page 1 of 2.");
            assert_eq!(images.len(), 1);
            assert_eq!(images[0].label, "media pool sheet page 1 of 2");
            assert_eq!(*images[0].data, vec![0x89, 0x50, 0x4e, 0x47]);
        }
        other => panic!("expected media-pool sheet result, got {other:?}"),
    }
    assert!(events.iter().any(
        |event| matches!(event, AgentEvent::Image(image) if image.label == "media pool sheet page 1 of 2")
    ));
}

#[test]
fn engine_senses_are_read_only_filtered_and_outrank_host_collisions() {
    let (mut bridge, _, _, _) = fixture();
    bridge.sense_specs = vec![
        host_spec("media_frame"),
        host_spec("media_frame"),
        {
            let mut spec = host_spec("media_mutate");
            spec.tier = ToolTier::Workspace;
            spec
        },
        host_spec("invalid_name"),
    ];
    bridge
        .sense_outputs
        .push_back(Ok(ToolOutput::text("sandbox frame")));
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![("call_1", "media_frame", serde_json::json!({}))]),
        text_turn("Checked."),
    ]);
    let mut tool_host = ScriptedHost::new(
        vec![host_spec("media_frame")],
        vec![Ok(ToolOutput::text("wrong live frame"))],
    );

    let (outcome, _) = run_with(
        &provider,
        &mut bridge,
        &mut tool_host,
        &EditorContext::default(),
        "check the frame",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(bridge.sense_calls.len(), 1);
    assert!(tool_host.calls.is_empty(), "sandbox sense wins the name");
    let offered = &provider.tool_names()[0];
    assert_eq!(
        offered.iter().filter(|name| *name == "media_frame").count(),
        1
    );
    assert!(!offered.iter().any(|name| name == "media_mutate"));
    assert!(!offered.iter().any(|name| name == "invalid_name"));
}

/// Host dispatch round-trip: the call reaches the host with its exact
/// arguments, the output's text and image ride back as the tool result,
/// and the transcript hears about both its action and bounded image.
#[test]
fn host_tool_round_trip_carries_arguments_images_and_events() {
    let (mut host, _, _, _) = fixture();
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![(
            "call_1",
            "media_screenshot_preview",
            serde_json::json!({ "at": 1.5 }),
        )]),
        text_turn("Here's the preview."),
    ]);
    let mut tool_host = ScriptedHost::new(
        vec![host_spec("media_screenshot_preview")],
        vec![Ok(ToolOutput {
            text: "screenshot taken at 1.50s\n(1280x720 png)".into(),
            images: vec![ImagePart::png(vec![1, 2, 3], "preview at 1.50s")],
        })],
    );

    let (outcome, events) = run_with(
        &provider,
        &mut host,
        &mut tool_host,
        &EditorContext::default(),
        "look at the preview",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert!(outcome.actions.is_empty(), "host calls are not edits");
    assert_eq!(
        tool_host.calls,
        vec![(
            "media_screenshot_preview".to_string(),
            serde_json::json!({ "at": 1.5 })
        )]
    );

    // The next request carries the result — text and image both.
    let second = &provider.requests()[1];
    match second.last().unwrap() {
        Message::ToolResult {
            call_id,
            content,
            images,
        } => {
            assert_eq!(call_id, "call_1");
            assert_eq!(content, "screenshot taken at 1.50s\n(1280x720 png)");
            assert_eq!(images.len(), 1);
            assert_eq!(images[0].label, "preview at 1.50s");
            assert_eq!(*images[0].data, vec![1, 2, 3]);
        }
        other => panic!("expected tool result, got {other:?}"),
    }

    assert!(
        events.iter().any(|e| matches!(
            e,
            AgentEvent::HostAction { name, summary }
                if name == "media_screenshot_preview" && summary == "screenshot taken at 1.50s"
        )),
        "{events:?}"
    );
    assert!(events.iter().any(
        |event| matches!(event, AgentEvent::Image(image) if image.label == "preview at 1.50s")
    ));
}

#[test]
fn host_pre_hook_rejection_skips_authorization_dispatch_and_post() {
    let (mut bridge, _, _, _) = fixture();
    bridge
        .before_host_outputs
        .push_back(Err("project operations must run before staged edits".into()));
    let arguments = serde_json::json!({ "path": "/tmp/new.mp4" });
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![("call_1", "project_import_media", arguments.clone())]),
        text_turn("I need to import before editing."),
    ]);
    let mut tool_host = ScriptedHost::new(
        vec![host_spec("project_import_media")],
        vec![Ok(ToolOutput::text("imported"))],
    );

    let (outcome, _) = run_with(
        &provider,
        &mut bridge,
        &mut tool_host,
        &EditorContext::default(),
        "import this",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(
        bridge.before_host_calls,
        vec![("project_import_media".into(), arguments)]
    );
    assert!(bridge.after_host_calls.is_empty());
    assert!(tool_host.authorizations.is_empty());
    assert!(tool_host.calls.is_empty());
    match provider.requests()[1].last().unwrap() {
        Message::ToolResult { content, .. } => {
            assert!(
                content.contains("rejected: project operations must run before staged edits"),
                "{content}"
            );
        }
        other => panic!("expected tool result, got {other:?}"),
    }
}

#[test]
fn host_post_hook_runs_after_success_and_failure_without_changing_outputs() {
    let (mut bridge, _, _, _) = fixture();
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![
            (
                "call_1",
                "media_screenshot_preview",
                serde_json::json!({ "at": 1.0 }),
            ),
            (
                "call_2",
                "media_screenshot_preview",
                serde_json::json!({ "at": 2.0 }),
            ),
        ]),
        text_turn("Checked both frames."),
    ]);
    let mut tool_host = ScriptedHost::new(
        vec![host_spec("media_screenshot_preview")],
        vec![
            Ok(ToolOutput {
                text: "first frame".into(),
                images: vec![ImagePart::png(vec![4, 5, 6], "first")],
            }),
            Err("second frame unavailable".into()),
        ],
    );

    let (outcome, events) = run_with(
        &provider,
        &mut bridge,
        &mut tool_host,
        &EditorContext::default(),
        "check two frames",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(
        bridge
            .after_host_calls
            .iter()
            .map(|(name, arguments, succeeded)| (
                name.as_str(),
                arguments["at"].as_f64(),
                *succeeded
            ))
            .collect::<Vec<_>>(),
        vec![
            ("media_screenshot_preview", Some(1.0), true),
            ("media_screenshot_preview", Some(2.0), false),
        ]
    );
    let requests = provider.requests();
    let results = requests[1]
        .iter()
        .filter_map(|message| match message {
            Message::ToolResult {
                call_id,
                content,
                images,
            } => Some((call_id.as_str(), content.as_str(), images.as_slice())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, "call_1");
    assert_eq!(results[0].1, "first frame");
    assert_eq!(results[0].2[0].label, "first");
    assert_eq!(results[1].0, "call_2");
    assert_eq!(results[1].1, "rejected: second frame unavailable");
    assert!(results[1].2.is_empty());
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::Image(image) if image.label == "first"))
    );
}

#[test]
fn host_post_hook_failure_aborts_and_rolls_back_before_more_work() {
    let (mut bridge, _, _, clip) = fixture();
    bridge
        .after_host_outputs
        .push_back(Err("live project snapshot did not reply".into()));
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![(
            "edit_1",
            "split_clip",
            serde_json::json!({ "clip": clip, "at": 5.0 }),
        )]),
        tool_turn(vec![
            ("host_1", "app_ping", serde_json::json!({})),
            (
                "edit_2",
                "add_track",
                serde_json::json!({ "kind": "text", "name": "Too late" }),
            ),
        ]),
        text_turn("This turn must never run."),
    ]);
    let mut tool_host = ScriptedHost::new(
        vec![host_spec("app_ping")],
        vec![Ok(ToolOutput::text("pong"))],
    );

    let (outcome, events) = run_with(
        &provider,
        &mut bridge,
        &mut tool_host,
        &EditorContext::default(),
        "split, ping, then add a track",
        &AgentConfig::default(),
    );

    match &outcome.status {
        PromptStatus::Aborted(reason) => {
            assert!(reason.contains("reconciliation failed"), "{reason}");
            assert!(
                reason.contains("host effects may already have occurred"),
                "{reason}"
            );
        }
        other => panic!("expected abort, got {other:?}"),
    }
    assert_eq!(provider.requests().len(), 2, "no later provider turn");
    assert_eq!(outcome.actions.len(), 1, "the later edit never ran");
    assert_eq!(tool_host.calls.len(), 1, "host dispatch ran exactly once");
    assert_eq!(bridge.after_host_calls.len(), 1);
    assert_eq!(
        bridge.engine.project().timeline().clip_count(),
        1,
        "the staged split was rolled back"
    );
    assert_eq!(bridge.engine.project().timeline().track_count(), 1);
    assert!(!bridge.engine.undo(), "the aborted group left no history");
    assert!(
        events
            .iter()
            .all(|event| !matches!(event, AgentEvent::HostAction { .. })),
        "a result that could not be reconciled is not surfaced as completed"
    );
}

#[test]
fn inline_image_events_match_the_request_wide_budget() {
    let (mut bridge, _, _, _) = fixture();
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![
            ("call_1", "media_frame", serde_json::json!({ "at": 1 })),
            ("call_2", "media_frame", serde_json::json!({ "at": 2 })),
        ]),
        text_turn("Compared the frames."),
    ]);
    let mut tool_host = ScriptedHost::new(
        vec![host_spec("media_frame")],
        vec![
            Ok(ToolOutput {
                text: "old frame".into(),
                images: vec![ImagePart::png(vec![1, 2, 3], "old")],
            }),
            Ok(ToolOutput {
                text: "new frame".into(),
                images: vec![ImagePart::png(vec![4, 5, 6], "new")],
            }),
        ],
    );
    let config = AgentConfig {
        max_images: 1,
        max_image_bytes: 10,
        ..AgentConfig::default()
    };

    let (outcome, events) = run_with(
        &provider,
        &mut bridge,
        &mut tool_host,
        &EditorContext::default(),
        "compare two frames",
        &config,
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    let requests = provider.requests();
    let results: Vec<_> = requests[1]
        .iter()
        .filter_map(|message| match message {
            Message::ToolResult {
                content, images, ..
            } => Some((content, images)),
            _ => None,
        })
        .collect();
    assert_eq!(results.len(), 2);
    assert!(results[0].0.contains("image no longer attached: old"));
    assert!(results[0].1.is_empty());
    assert_eq!(results[1].1[0].label, "new");

    let labels: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::Image(image) => Some(image.label.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(labels, vec!["new"]);
}

#[test]
fn app_controls_use_fresh_state_for_a_looped_playback_workflow() {
    let (mut bridge, _, _, _) = fixture();
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![("state", "app_state", serde_json::json!({}))]),
        tool_turn(vec![
            (
                "loop",
                "app_loop_range",
                serde_json::json!({
                    "action": "set",
                    "start_seconds": 115.0,
                    "end_seconds": 120.0
                }),
            ),
            (
                "panel",
                "app_panel",
                serde_json::json!({ "panel": "library", "action": "hide" }),
            ),
            (
                "theme",
                "app_theme",
                serde_json::json!({ "theme": "dark-blue" }),
            ),
            (
                "play",
                "app_playback",
                serde_json::json!({ "action": "play" }),
            ),
        ]),
        text_turn("Playing the last five seconds on a loop with the library hidden."),
    ]);
    let specs = [
        "app_state",
        "app_loop_range",
        "app_panel",
        "app_theme",
        "app_playback",
    ]
    .into_iter()
    .map(host_spec)
    .collect();
    let mut tool_host = ScriptedHost::new(
        specs,
        vec![
            Ok(ToolOutput::text(
                r#"{"sequence":{"duration_seconds":120.0}}"#,
            )),
            Ok(ToolOutput::text(
                r#"{"state":{"transport":{"loop_enabled":true,"range_in_seconds":115.0,"range_out_seconds":120.0}}}"#,
            )),
            Ok(ToolOutput::text(
                r#"{"state":{"panels":{"library":false}}}"#,
            )),
            Ok(ToolOutput::text(r#"{"state":{"theme":"dark-blue"}}"#)),
            Ok(ToolOutput::text(
                r#"{"state":{"transport":{"playing":true}}}"#,
            )),
        ],
    );

    let (outcome, events) = run_with(
        &provider,
        &mut bridge,
        &mut tool_host,
        &EditorContext::default(),
        "play the last 5 seconds looped, hide the library, and use the dark theme",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert!(outcome.actions.is_empty());
    assert_eq!(
        tool_host.calls,
        vec![
            ("app_state".into(), serde_json::json!({})),
            (
                "app_loop_range".into(),
                serde_json::json!({
                    "action": "set",
                    "start_seconds": 115.0,
                    "end_seconds": 120.0
                }),
            ),
            (
                "app_panel".into(),
                serde_json::json!({ "panel": "library", "action": "hide" }),
            ),
            (
                "app_theme".into(),
                serde_json::json!({ "theme": "dark-blue" }),
            ),
            (
                "app_playback".into(),
                serde_json::json!({ "action": "play" }),
            ),
        ]
    );
    let host_actions = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::HostAction { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        host_actions,
        vec![
            "app_state",
            "app_loop_range",
            "app_panel",
            "app_theme",
            "app_playback"
        ]
    );
    let requests = provider.requests();
    let final_tool_results = requests[2]
        .iter()
        .filter_map(|message| match message {
            Message::ToolResult {
                call_id, content, ..
            } if call_id != "state" => Some((call_id.as_str(), content.as_str())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(final_tool_results.len(), 4);
    assert!(
        final_tool_results
            .iter()
            .any(|(id, content)| *id == "play" && content.contains("\"playing\":true"))
    );
}

#[test]
fn host_rejection_reports_and_the_prompt_still_completes() {
    let (mut host, _, _, _) = fixture();
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![(
            "call_1",
            "media_screenshot_preview",
            serde_json::json!({}),
        )]),
        text_turn("Couldn't grab the preview."),
    ]);
    let mut tool_host = ScriptedHost::new(
        vec![host_spec("media_screenshot_preview")],
        vec![Err("preview is not rendered yet".into())],
    );

    let (outcome, events) = run_with(
        &provider,
        &mut host,
        &mut tool_host,
        &EditorContext::default(),
        "look at the preview",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    let second = &provider.requests()[1];
    match second.last().unwrap() {
        Message::ToolResult {
            content, images, ..
        } => {
            assert_eq!(content, "rejected: preview is not rendered yet");
            assert!(images.is_empty());
        }
        other => panic!("expected tool result, got {other:?}"),
    }
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, AgentEvent::HostAction { .. })),
        "a rejected call is not an action"
    );
}

#[test]
fn system_host_tools_fail_closed_without_an_approval_broker() {
    let (mut host, _, _, _) = fixture();
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![(
            "call_1",
            "system_cache_clear",
            serde_json::json!({ "cache": "proxies" }),
        )]),
        text_turn("I couldn't clear it without approval."),
    ]);
    let mut spec = host_spec("system_cache_clear");
    spec.tier = ToolTier::System;
    let mut tool_host = ScriptedHost::new(vec![spec], vec![Ok(ToolOutput::text("cache cleared"))]);

    let (outcome, _) = run_with(
        &provider,
        &mut host,
        &mut tool_host,
        &EditorContext::default(),
        "clear the proxy cache",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert!(
        tool_host.calls.is_empty(),
        "execution never runs before authorization"
    );
    let second = &provider.requests()[1];
    match second.last().unwrap() {
        Message::ToolResult { content, .. } => {
            assert!(content.contains("requires confirmation"), "{content}");
            assert!(content.contains("no approval broker"), "{content}");
        }
        other => panic!("expected tool result, got {other:?}"),
    }
}

/// The two fuses are independent: an edit cap of zero still lets host
/// tools run, and the host cap aborts with a full rollback.
#[test]
fn host_calls_have_their_own_cap_separate_from_the_edit_fuse() {
    // Edit cap 0, one host call: completes.
    let (mut host, _, _, _) = fixture();
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![(
            "call_1",
            "media_screenshot_preview",
            serde_json::json!({}),
        )]),
        text_turn("Looked."),
    ]);
    let mut tool_host = ScriptedHost::new(
        vec![host_spec("media_screenshot_preview")],
        vec![Ok(ToolOutput::text("looks fine"))],
    );
    let config = AgentConfig {
        max_tool_calls: 0,
        ..Default::default()
    };
    let (outcome, _) = run_with(
        &provider,
        &mut host,
        &mut tool_host,
        &EditorContext::default(),
        "look at the preview",
        &config,
    );
    assert_eq!(outcome.status, PromptStatus::Completed);

    // Host cap 1, one applied edit then two host calls: the second host
    // call trips the cap and the whole prompt rolls back.
    let (mut host, _, _, clip) = fixture();
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![(
            "call_1",
            "split_clip",
            serde_json::json!({ "clip": clip, "at": 5.0 }),
        )]),
        tool_turn(vec![
            ("call_2", "media_screenshot_preview", serde_json::json!({})),
            ("call_3", "media_screenshot_preview", serde_json::json!({})),
        ]),
    ]);
    let mut tool_host = ScriptedHost::new(
        vec![host_spec("media_screenshot_preview")],
        vec![Ok(ToolOutput::text("looks fine"))],
    );
    let config = AgentConfig {
        max_host_calls: 1,
        ..Default::default()
    };
    let (outcome, _) = run_with(
        &provider,
        &mut host,
        &mut tool_host,
        &EditorContext::default(),
        "split then stare at the preview forever",
        &config,
    );
    match &outcome.status {
        PromptStatus::Aborted(reason) => assert!(reason.contains("1-host-call cap"), "{reason}"),
        other => panic!("expected abort, got {other:?}"),
    }
    assert_eq!(tool_host.calls.len(), 1, "the cap trips before the call");
    assert_eq!(
        host.engine.project().timeline().clip_count(),
        1,
        "the applied split was rolled back"
    );
    assert!(!host.engine.undo(), "an aborted prompt leaves no history");
}

/// Defense in depth: a host spec reusing a built-in name is neither
/// offered to the provider nor dispatched — the built-in wins.
#[test]
fn host_tool_name_collisions_lose_to_the_built_ins() {
    let (mut host, _, _, clip) = fixture();
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![(
            "call_1",
            "split_clip",
            serde_json::json!({ "clip": clip, "at": 5.0 }),
        )]),
        tool_turn(vec![("call_2", "describe_project", serde_json::json!({}))]),
        text_turn("Split it."),
    ]);
    let mut tool_host = ScriptedHost::new(
        vec![
            host_spec("split_clip"),
            host_spec("describe_project"),
            host_spec("app_ping"),
            host_spec("app_ping"),
            host_spec("app-Bad"),
        ],
        Vec::new(),
    );

    let (outcome, _) = run_with(
        &provider,
        &mut host,
        &mut tool_host,
        &EditorContext::default(),
        "split the clip",
        &AgentConfig::default(),
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert!(
        tool_host.calls.is_empty(),
        "both colliding names dispatched as built-ins"
    );
    assert_eq!(outcome.actions.len(), 1, "the real split applied");
    assert_eq!(host.engine.project().timeline().clip_count(), 2);

    // The offered tool list carries each name once and keeps the
    // legitimately-namespaced host tool.
    let offered = &provider.tool_names()[0];
    assert_eq!(
        offered
            .iter()
            .filter(|n| n.as_str() == "split_clip")
            .count(),
        1
    );
    assert_eq!(
        offered
            .iter()
            .filter(|n| n.as_str() == "describe_project")
            .count(),
        1
    );
    assert_eq!(
        offered.iter().filter(|n| n.as_str() == "app_ping").count(),
        1,
        "duplicate host specs are offered once"
    );
    assert!(
        !offered.iter().any(|n| n == "app-Bad"),
        "malformed host names never reach the model"
    );
}

/// `commit_progress` records phase breaks between edits, refuses empty
/// phases, and charges neither the edit fuse nor the host cap.
#[test]
fn commit_progress_records_phase_breaks_without_charging_caps() {
    let (mut host, _, _, _) = fixture();
    let provider = ScriptedProvider::new(vec![
        // Nothing to commit yet.
        tool_turn(vec![("c0", "commit_progress", serde_json::json!({}))]),
        tool_turn(vec![
            ("c1", "add_marker", serde_json::json!({ "at": 1.0 })),
            ("c2", "add_marker", serde_json::json!({ "at": 2.0 })),
        ]),
        // First commit lands; the immediate second one is empty.
        tool_turn(vec![
            ("c3", "commit_progress", serde_json::json!({})),
            ("c4", "commit_progress", serde_json::json!({})),
        ]),
        tool_turn(vec![("c5", "add_marker", serde_json::json!({ "at": 3.0 }))]),
        text_turn("Three markers, two phases."),
    ]);

    // Exactly three edit slots and zero host slots: commits must charge
    // neither cap for this to complete.
    let config = AgentConfig {
        max_tool_calls: 3,
        max_host_calls: 0,
        ..Default::default()
    };
    let (outcome, _) = run(
        &provider,
        &mut host,
        &EditorContext::default(),
        "mark the beats in phases",
        &config,
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(outcome.actions.len(), 3);
    assert_eq!(outcome.phase_breaks, vec![2]);

    let results: Vec<(String, String)> = provider
        .requests()
        .iter()
        .flat_map(|msgs| msgs.iter())
        .filter_map(|m| match m {
            Message::ToolResult {
                call_id, content, ..
            } => Some((call_id.clone(), content.clone())),
            _ => None,
        })
        .collect();
    let result_for = |id: &str| {
        results
            .iter()
            .find(|(call_id, _)| call_id == id)
            .map(|(_, content)| content.as_str())
            .unwrap_or_else(|| panic!("no result for {id}"))
    };
    assert_eq!(result_for("c0"), "nothing new to commit — make edits first");
    assert_eq!(result_for("c3"), "ok: committed phase 1 (2 edits)");
    assert_eq!(result_for("c4"), "nothing new to commit — make edits first");
}

#[test]
fn system_prompt_gains_host_tool_rules_only_when_host_tools_exist() {
    // Without host tools: no paragraph.
    let (mut host, _, _, _) = fixture();
    let provider = ScriptedProvider::new(vec![text_turn("Hi.")]);
    let _ = run(
        &provider,
        &mut host,
        &EditorContext::default(),
        "hello",
        &AgentConfig::default(),
    );
    match &provider.requests()[0][0] {
        Message::System { content } => assert!(!content.contains("Host tools:"), "{content}"),
        other => panic!("expected system message, got {other:?}"),
    }

    // With host tools: the paragraph and the spec both show up.
    let (mut host, _, _, _) = fixture();
    let provider = ScriptedProvider::new(vec![text_turn("Hi.")]);
    let mut tool_host = ScriptedHost::new(vec![host_spec("app_ping")], Vec::new());
    let _ = run_with(
        &provider,
        &mut host,
        &mut tool_host,
        &EditorContext::default(),
        "hello",
        &AgentConfig::default(),
    );
    match &provider.requests()[0][0] {
        Message::System { content } => {
            assert!(content.contains("Host tools:"), "{content}");
            assert!(content.contains("declined"), "{content}");
        }
        other => panic!("expected system message, got {other:?}"),
    }
    assert!(provider.tool_names()[0].iter().any(|n| n == "app_ping"));
}
