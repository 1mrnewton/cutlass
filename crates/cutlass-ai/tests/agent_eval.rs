//! The eval harness: scripted prompts against a real engine, zero network.
//!
//! Each test scripts the provider's turns, runs the full agent loop
//! through an `EngineBridge` backed by a real `Engine`, and asserts on
//! the final timeline, the action log, and the undo history. This is how
//! agent regressions get caught in CI without a live model.

use std::sync::atomic::AtomicBool;

use cutlass_ai::agent::{run_prompt, AgentConfig, AgentEvent, EngineBridge, PromptStatus};
use cutlass_ai::provider::{ChatTurn, FinishReason, Message, ToolCall};
use cutlass_ai::providers::ScriptedProvider;
use cutlass_ai::{summarize, validate, EditorContext, ProjectSummary, WireCommand};
use cutlass_commands::EditOutcome;
use cutlass_engine::{ApplyOutcome, ColorConvertPath, Engine, EngineConfig};
use cutlass_models::{MediaSource, Project, Rational, RationalTime, TimeRange, TrackKind};

const R24: Rational = Rational::FPS_24;

/// A real engine behind the loop's bridge.
struct EngineHost {
    engine: Engine,
    _dir: tempfile::TempDir,
}

impl EngineHost {
    fn new(project: Project) -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = EngineConfig {
            cache_dir: dir.path().join("cache"),
            cache_budget_bytes: 16 * 1024 * 1024,
            undo_limit: 64,
            color_convert: ColorConvertPath::Gpu,
        };
        Self {
            engine: Engine::with_project(config, project).expect("engine"),
            _dir: dir,
        }
    }
}

impl EngineBridge for EngineHost {
    fn summary(&mut self) -> ProjectSummary {
        summarize(self.engine.project())
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
        tool_calls: Vec::new(),
        finish: FinishReason::Stop,
    }
}

fn run_with(
    provider: &dyn cutlass_ai::provider::ChatProvider,
    host: &mut EngineHost,
    context: &EditorContext,
    prompt: &str,
    config: &AgentConfig,
) -> (cutlass_ai::PromptOutcome, Vec<AgentEvent>) {
    let cancel = AtomicBool::new(false);
    let mut events = Vec::new();
    let outcome = run_prompt(
        provider,
        host,
        context,
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
    run_with(provider, host, context, prompt, config)
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
        Message::ToolResult { call_id, content } => {
            assert_eq!(call_id, "call_1");
            assert!(content.contains("rejected: clip 999 does not exist"), "{content}");
            assert!(content.contains(&clip.to_string()), "{content}");
        }
        other => panic!("expected tool result, got {other:?}"),
    }
}

#[test]
fn cap_trip_rolls_the_whole_prompt_back() {
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

    match &outcome.status {
        PromptStatus::Aborted(reason) => assert!(reason.contains("1-edit cap"), "{reason}"),
        other => panic!("expected abort, got {other:?}"),
    }
    // The split that did apply was rolled back; nothing remains.
    assert_eq!(host.engine.project().timeline().clip_count(), 1);
    assert_eq!(host.engine.project().timeline().track_count(), 1);
    assert!(!host.engine.undo(), "a rolled-back prompt leaves no history");
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
    assert!(events
        .iter()
        .all(|e| matches!(e, AgentEvent::TextDelta(_))));
    assert!(!host.engine.undo(), "answering a question records no history");
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
        .add_clip(track, talk, TimeRange::at_rate(0, 120, R24), RationalTime::new(0, R24))
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
            assert!(content.contains("answer in text directly from it"));
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
        _on_text: &mut dyn FnMut(&str),
    ) -> Result<ChatTurn, cutlass_ai::provider::ProviderError> {
        let last = request.messages.last().unwrap();
        Ok(match last {
            Message::User { .. } => tool_turn(vec![(
                "call_1",
                "add_track",
                serde_json::json!({ "kind": "text", "name": "Titles" }),
            )]),
            Message::ToolResult { call_id, content } if call_id == "call_1" => {
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
    assert!(summarize(host.engine.project())
        .tracks
        .iter()
        .all(|t| t.name != "Titles"));
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
