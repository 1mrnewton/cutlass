//! End-to-end motion animation evals: scripted agent → real engine → sampled frames.
//!
//! Sibling of `agent_eval.rs` (kept separate so that file stays under the
//! size ceiling). Reuses the same EngineBridge + ScriptedProvider patterns.

use std::sync::atomic::AtomicBool;

use cutlass_ai::agent::{
    AgentConfig, AgentEvent, EngineBridge, PromptStatus, run_prompt_with_host,
};
use cutlass_ai::provider::{ChatTurn, FinishReason, Message, ToolCall};
use cutlass_ai::providers::ScriptedProvider;
use cutlass_ai::tools::NullToolHost;
use cutlass_ai::{EditorContext, ProjectSummary, WireCommand, summarize, validate};
use cutlass_commands::EditOutcome;
use cutlass_engine::{ApplyOutcome, Engine, EngineConfig};
use cutlass_models::{
    ClipId, ClipParam, Easing, MediaSource, ParamValue, Project, Rational, RationalTime, TimeRange,
    TrackKind,
};

const R24: Rational = Rational::FPS_24;

struct EngineHost {
    engine: Engine,
}

impl EngineHost {
    fn new(project: Project) -> Self {
        let config = EngineConfig { undo_limit: 64 };
        Self {
            engine: Engine::with_project(config, project).expect("engine"),
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

/// 24 fps project, one video track, one 10 s clip (of a 60 s source).
fn fixture_starting_at(start_seconds: f64) -> (EngineHost, u64) {
    let mut project = Project::new("motion-eval", R24);
    let media = project.add_media(MediaSource::new(
        "/tmp/motion-eval.mp4",
        1920,
        1080,
        R24,
        60 * 24,
        true,
    ));
    let track = project.add_track(TrackKind::Video, "V1");
    let start_ticks = (start_seconds * f64::from(R24.num) / f64::from(R24.den)).round() as i64;
    let clip = project
        .add_clip(
            track,
            media,
            TimeRange::at_rate(0, 240, R24),
            RationalTime::new(start_ticks, R24),
        )
        .unwrap();
    (EngineHost::new(project), clip.raw())
}

/// Clip at timeline 0 — used by evals that do not exercise clip-start offset.
fn fixture() -> (EngineHost, u64) {
    fixture_starting_at(0.0)
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
        usage: None,
    }
}

fn text_turn(text: &str) -> ChatTurn {
    ChatTurn {
        text: text.to_string(),
        reasoning_summary: String::new(),
        tool_calls: Vec::new(),
        finish: FinishReason::Stop,
        usage: None,
    }
}

fn run(
    provider: &ScriptedProvider,
    host: &mut EngineHost,
    prompt: &str,
) -> (cutlass_ai::PromptOutcome, Vec<AgentEvent>) {
    let cancel = AtomicBool::new(false);
    let mut events = Vec::new();
    let outcome = run_prompt_with_host(
        provider,
        host,
        &mut NullToolHost,
        &EditorContext::default(),
        &cutlass_ai::AgentExtensions::default(),
        &[],
        prompt,
        &AgentConfig::default(),
        &cancel,
        &mut |e| events.push(e),
    );
    (outcome, events)
}

fn tool_result_contents(provider: &ScriptedProvider) -> Vec<String> {
    provider
        .requests()
        .iter()
        .flat_map(|msgs| msgs.iter())
        .filter_map(|m| match m {
            Message::ToolResult { content, .. } => Some(content.clone()),
            _ => None,
        })
        .collect()
}

/// Clip-relative tick for an offset from the clip's timeline start.
fn tick_at(seconds: f64) -> i64 {
    (seconds * f64::from(R24.num) / f64::from(R24.den)).round() as i64
}

#[test]
fn slide_in_pan_keyframes_sample_monotonic_to_center() {
    // Non-zero clip start exercises absolute `at` → clip-relative tick mapping.
    const CLIP_START: f64 = 2.0;
    let (mut host, clip) = fixture_starting_at(CLIP_START);
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![
            (
                "call_1",
                "set_param_keyframe",
                serde_json::json!({
                    "clip": clip,
                    "param": "position",
                    "at": CLIP_START,
                    "position": [-1.0, 0.0],
                    // Easing shapes the segment *leaving* this keyframe.
                    "easing": "ease_out",
                }),
            ),
            (
                "call_2",
                "set_param_keyframe",
                serde_json::json!({
                    "clip": clip,
                    "param": "position",
                    "at": CLIP_START + 1.0,
                    "position": [0.0, 0.0],
                }),
            ),
        ]),
        text_turn("Slid the clip in from the left over one second."),
    ]);

    let (outcome, _) = run(
        &provider,
        &mut host,
        "slide the clip in from the left over one second",
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(outcome.actions.len(), 2);

    let placed = host.engine.project().clip(ClipId::from_raw(clip)).unwrap();
    let kfs = placed.transform.position.keyframes();
    assert_eq!(kfs.len(), 2);
    assert_eq!(kfs[0].tick, 0, "keyframes are clip-relative");
    assert_eq!(kfs[0].value, [-1.0, 0.0]);
    assert_eq!(kfs[0].easing, Easing::EaseOut);
    assert_eq!(kfs[1].tick, 24);
    assert_eq!(kfs[1].value, [0.0, 0.0]);

    // Sample at start, +0.25s, +0.5s, +0.75s, +1s — x rises to on-canvas center.
    let samples: Vec<[f32; 2]> = [0.0, 0.25, 0.5, 0.75, 1.0]
        .into_iter()
        .map(|s| placed.transform.sample(tick_at(s)).position)
        .collect();
    assert_eq!(samples[0], [-1.0, 0.0]);
    assert_eq!(
        samples[4],
        [0.0, 0.0],
        "final position must be canvas center"
    );
    for window in samples.windows(2) {
        assert!(
            window[0][0] < window[1][0],
            "x must increase monotonically: {samples:?}"
        );
        assert_eq!(window[0][1], 0.0);
        assert_eq!(window[1][1], 0.0);
    }
    // ease_out is ahead of linear at the midpoint (−0.25 vs −0.5).
    let linear_mid = -0.5f32;
    assert!(
        samples[2][0] > linear_mid + 0.1,
        "ease_out midpoint must be further along than linear ({linear_mid}): got {}",
        samples[2][0]
    );

    let summary = summarize(host.engine.project());
    let position = summary.tracks[0].clips[0]
        .keyframes
        .as_ref()
        .expect("keyframes")
        .get("position")
        .expect("position");
    assert_eq!(position[0].at, CLIP_START);
    assert_eq!(position[1].at, CLIP_START + 1.0);
}

#[test]
fn zoom_in_scale_keyframes_sample_monotonic_and_describe() {
    let (mut host, clip) = fixture();
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![
            (
                "call_1",
                "set_param_keyframe",
                serde_json::json!({
                    "clip": clip, "param": "scale", "at": 0.0, "value": 1.0,
                }),
            ),
            (
                "call_2",
                "set_param_keyframe",
                serde_json::json!({
                    "clip": clip, "param": "scale", "at": 1.0, "value": 1.3,
                }),
            ),
        ]),
        text_turn("Zoomed in from 1.0 to 1.3 over the first second."),
    ]);

    let (outcome, _) = run(
        &provider,
        &mut host,
        "zoom the clip in over the first second",
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(outcome.actions.len(), 2);

    let placed = host.engine.project().clip(ClipId::from_raw(clip)).unwrap();
    let kfs = placed.transform.scale.keyframes();
    assert_eq!(kfs.len(), 2);
    assert_eq!(kfs[0].tick, 0);
    assert_eq!(kfs[0].value.x, 1.0);
    assert_eq!(kfs[0].value.y, 1.0);
    assert_eq!(kfs[1].tick, 24);
    assert!((kfs[1].value.x - 1.3).abs() < 1e-5);
    assert!((kfs[1].value.y - 1.3).abs() < 1e-5);

    let scales: Vec<f32> = [0.0, 0.25, 0.5, 0.75, 1.0]
        .into_iter()
        .map(|s| {
            let sc = placed.transform.sample(tick_at(s)).scale;
            assert!(
                (sc.x - sc.y).abs() < 1e-5,
                "scale should stay uniform: {:?}",
                sc
            );
            sc.x
        })
        .collect();
    assert!((scales[0] - 1.0).abs() < 1e-5);
    assert!((scales[4] - 1.3).abs() < 1e-5);
    for window in scales.windows(2) {
        assert!(
            window[0] < window[1],
            "scale must increase monotonically: {scales:?}"
        );
    }

    // Describe exposes absolute timeline seconds under keyframes.scale.
    let summary = summarize(host.engine.project());
    let clip_summary = &summary.tracks[0].clips[0];
    assert_eq!(
        clip_summary.scale, None,
        "animated scale omits static field"
    );
    let scale_kfs = clip_summary
        .keyframes
        .as_ref()
        .expect("keyframes present")
        .get("scale")
        .expect("scale keyframes");
    assert_eq!(scale_kfs.len(), 2);
    assert_eq!(scale_kfs[0].at, 0.0);
    assert_eq!(scale_kfs[0].value, serde_json::json!(1.0));
    assert_eq!(scale_kfs[1].at, 1.0);
    assert_eq!(scale_kfs[1].value, serde_json::json!(1.3));
}

#[test]
fn percent_mistake_teaching_error_reaches_model_and_recovers() {
    let (mut host, clip) = fixture();
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![(
            "call_1",
            "set_param_keyframe",
            serde_json::json!({
                "clip": clip, "param": "scale", "at": 0.0, "value": 130,
            }),
        )]),
        tool_turn(vec![(
            "call_2",
            "set_param_keyframe",
            serde_json::json!({
                "clip": clip, "param": "scale", "at": 0.0, "value": 1.3,
            }),
        )]),
        text_turn("Set scale to 1.3 (130%)."),
    ]);

    let (outcome, _) = run(&provider, &mut host, "zoom the clip to 130%");

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(
        outcome.actions.len(),
        1,
        "only the corrected call should apply"
    );

    let results = tool_result_contents(&provider);
    assert!(
        results.iter().any(|r| r.contains("for 130% send 1.3")),
        "teaching rejection must reach the model as a tool result: {results:?}"
    );

    let placed = host.engine.project().clip(ClipId::from_raw(clip)).unwrap();
    let kfs = placed.transform.scale.keyframes();
    assert_eq!(kfs.len(), 1);
    assert_eq!(kfs[0].tick, 0);
    assert!((kfs[0].value.x - 1.3).abs() < 1e-5);
}

#[test]
fn set_clip_transform_flatten_guard_preserves_position_keyframes() {
    let mut project = Project::new("motion-flatten", R24);
    let media = project.add_media(MediaSource::new(
        "/tmp/motion-flatten.mp4",
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
    let clip_raw = clip.raw();

    // Seed position keyframes before the agent runs.
    project
        .set_param_keyframe(
            clip,
            ClipParam::Position,
            RationalTime::new(0, R24),
            ParamValue::Vec2([-0.5, 0.0]),
            Easing::Linear,
            None,
        )
        .unwrap();
    project
        .set_param_keyframe(
            clip,
            ClipParam::Position,
            RationalTime::new(48, R24),
            ParamValue::Vec2([0.5, 0.0]),
            Easing::Linear,
            None,
        )
        .unwrap();
    let before = project
        .clip(clip)
        .unwrap()
        .transform
        .position
        .keyframes()
        .to_vec();
    assert_eq!(before.len(), 2);

    let mut host = EngineHost::new(project);
    let provider = ScriptedProvider::new(vec![
        tool_turn(vec![(
            "call_1",
            "set_clip_transform",
            serde_json::json!({
                "clip": clip_raw,
                "scale": 1.2,
            }),
        )]),
        text_turn("Could not flatten the animated transform."),
    ]);

    let (outcome, _) = run(
        &provider,
        &mut host,
        "set the clip scale to 1.2 without touching position",
    );

    assert_eq!(outcome.status, PromptStatus::Completed);
    assert_eq!(
        outcome.actions.len(),
        0,
        "set_clip_transform must not apply over keyframes"
    );

    let results = tool_result_contents(&provider);
    assert!(
        results.iter().any(|r| {
            (r.contains("keyframes") || r.contains("keyframed")) && r.contains("set_param_keyframe")
        }),
        "flatten-guard rejection must reach the model: {results:?}"
    );

    let after = host
        .engine
        .project()
        .clip(ClipId::from_raw(clip_raw))
        .unwrap()
        .transform
        .position
        .keyframes();
    assert_eq!(after.len(), before.len());
    assert_eq!(after[0].tick, before[0].tick);
    assert_eq!(after[0].value, before[0].value);
    assert_eq!(after[1].tick, before[1].tick);
    assert_eq!(after[1].value, before[1].value);
    assert!(
        !host
            .engine
            .project()
            .clip(ClipId::from_raw(clip_raw))
            .unwrap()
            .transform
            .scale
            .is_animated()
    );
    let scale = host
        .engine
        .project()
        .clip(ClipId::from_raw(clip_raw))
        .unwrap()
        .transform
        .scale
        .sample(0);
    assert_eq!(scale.x, 1.0);
    assert_eq!(scale.y, 1.0);
}
