//! Offline request-payload size ratchet for the Chat Completions wire body.
//!
//! These ceilings catch regressions (>~15% growth) in tool schemas and
//! transcript bloat. When schema-shrink / transcript-trim work lands, ratchet
//! the ceilings *down* to the new measured values (+ ~15% headroom).

use cutlass_models::{MediaSource, Project, Rational, RationalTime, TimeRange, TrackKind};

use crate::agent::system_prompt;
use crate::describe::{EditorContext, summarize};
use crate::extend::AgentExtensions;
use crate::provider::{ChatRequest, Message, ToolCall};
use crate::providers::openai_compat::OpenAiCompatProvider;
use crate::wire::{self, ToolSpec};

/// Match the hostless tool list assembled in `run_prompt_with_host`.
fn hostless_tools() -> Vec<ToolSpec> {
    let mut tools = wire::tool_specs();
    tools.push(wire::describe_project_spec());
    tools.push(ToolSpec {
        name: "commit_progress".into(),
        description: "Mark the edits so far as one completed phase so they land as \
                      their own undo step. Call this between logical stages of a \
                      long task; costs nothing."
            .into(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {},
        }),
    });
    tools
}

fn body_bytes(provider: &OpenAiCompatProvider, messages: &[Message], tools: &[ToolSpec]) -> usize {
    provider
        .request_body(&ChatRequest {
            messages,
            tools,
            session_id: None,
        })
        .to_string()
        .len()
}

/// ~10 clips on 3 tracks (main video, overlay video, audio).
fn fixture_project() -> (Project, EditorContext) {
    const R24: Rational = Rational::FPS_24;
    let mut project = Project::new("request-size", R24);
    let media_a = project.add_media(MediaSource::new(
        "/tmp/size-a.mp4",
        1920,
        1080,
        R24,
        60 * 24,
        true,
    ));
    let media_b = project.add_media(MediaSource::new(
        "/tmp/size-b.mp4",
        1920,
        1080,
        R24,
        60 * 24,
        true,
    ));
    let media_audio = project.add_media(MediaSource::new(
        "/tmp/size-audio.wav",
        0,
        0,
        R24,
        60 * 24,
        true,
    ));

    let main = project.add_track(TrackKind::Video, "V1");
    let overlay = project.add_track(TrackKind::Video, "V2");
    let audio = project.add_track(TrackKind::Audio, "A1");

    // 5 clips on main, 3 on overlay, 2 on audio → 10 clips (no overlaps).
    let mut first_clip = None;
    for i in 0..5i64 {
        let start = i * 120;
        let clip = project
            .add_clip(
                main,
                media_a,
                TimeRange::at_rate(start, 120, R24),
                RationalTime::new(start, R24),
            )
            .unwrap();
        if first_clip.is_none() {
            first_clip = Some(clip);
        }
    }
    for i in 0..3i64 {
        let start = i * 160;
        project
            .add_clip(
                overlay,
                media_b,
                TimeRange::at_rate(0, 120, R24),
                RationalTime::new(start, R24),
            )
            .unwrap();
    }
    for i in 0..2i64 {
        let start = i * 360;
        project
            .add_clip(
                audio,
                media_audio,
                TimeRange::at_rate(0, 300, R24),
                RationalTime::new(start, R24),
            )
            .unwrap();
    }

    let context = EditorContext {
        selected_clips: vec![first_clip.expect("main clips").raw()],
        playhead_seconds: 0.0,
        ..Default::default()
    };
    (project, context)
}

fn describe_project_dump(project: &Project, context: &EditorContext) -> String {
    serde_json::json!({
        "project": summarize(project),
        "editor": context,
    })
    .to_string()
}

#[test]
fn tools_only_body_stays_under_ceiling() {
    let provider = OpenAiCompatProvider::new("http://localhost:11434/v1", "bench-model", None);
    let tools = hostless_tools();
    let messages = vec![Message::user(
        "cut the first 3 seconds of the selected clip",
    )];
    let bytes = body_bytes(&provider, &messages, &tools);

    // Measured 2026-07-23 (tool schema v47 Mirror feather docs): 52_531 bytes
    // (tools dominate). Ceiling = measured + ~3% headroom; ratchet down on shrinks.
    const TOOLS_ONLY_CEILING: usize = 54_100;
    assert!(
        bytes < TOOLS_ONLY_CEILING,
        "tools-only request body grew to {bytes} bytes (ceiling {TOOLS_ONLY_CEILING}); \
         measured at harness write time — see comment above. Ratchet the ceiling down \
         when schemas shrink."
    );
    // Keep the measured number visible in failures / bless output.
    eprintln!("request_size tools-only: {bytes} bytes (ceiling {TOOLS_ONLY_CEILING})");
}

#[test]
fn transcript_growth_last_turn_stays_under_ceiling() {
    let provider = OpenAiCompatProvider::new("http://localhost:11434/v1", "bench-model", None);
    let tools = hostless_tools();
    let (project, context) = fixture_project();
    let summary = summarize(&project);
    let system = system_prompt(&summary, &context, &AgentExtensions::default());
    let dump = describe_project_dump(&project, &context);

    // Six provider turns' worth of history culminating in the *next* request:
    // system + user, then five assistant/tool rounds (two of which are full
    // describe_project dumps), then a follow-up user message for turn 6.
    let messages = vec![
        Message::system(system),
        Message::user("cut the first 3 seconds of the selected clip and slide it in"),
        Message::Assistant {
            content: String::new(),
            tool_calls: vec![ToolCall {
                id: "c1".into(),
                name: "describe_project".into(),
                arguments: serde_json::json!({}),
            }],
        },
        Message::tool_result("c1", dump.clone()),
        Message::Assistant {
            content: String::new(),
            tool_calls: vec![ToolCall {
                id: "c2".into(),
                name: "trim_clip".into(),
                arguments: serde_json::json!({
                    "clip": context.selected_clips[0],
                    "start": 3.0,
                    "duration": 7.0,
                }),
            }],
        },
        Message::tool_result("c2", "ok: trimmed clip"),
        Message::Assistant {
            content: String::new(),
            tool_calls: vec![ToolCall {
                id: "c3".into(),
                name: "describe_project".into(),
                arguments: serde_json::json!({}),
            }],
        },
        Message::tool_result("c3", dump),
        Message::Assistant {
            content: String::new(),
            tool_calls: vec![ToolCall {
                id: "c4".into(),
                name: "set_clip_animation".into(),
                arguments: serde_json::json!({
                    "clip": context.selected_clips[0],
                    "slot": "entrance",
                    "animation": "slide_left",
                }),
            }],
        },
        Message::tool_result("c4", "ok: animation set"),
        Message::Assistant {
            content: "Done — trimmed and animated.".into(),
            tool_calls: vec![],
        },
        Message::user("also fade the audio track in over the first second"),
    ];

    let total = body_bytes(&provider, &messages, &tools);
    let tools_bytes = body_bytes(&provider, &[], &tools);
    let messages_bytes = body_bytes(&provider, &messages, &[]);

    // Measured 2026-07-23 (post speed schema drop): total=65_562,
    // tools≈50_846, messages≈14_807. Ceiling still holds with prior +15%.
    const TRANSCRIPT_CEILING: usize = 68_400;
    assert!(
        total < TRANSCRIPT_CEILING,
        "last-turn request body grew to {total} bytes \
         (tools≈{tools_bytes}, messages≈{messages_bytes}; ceiling {TRANSCRIPT_CEILING}). \
         Ratchet the ceiling down when transcript trimming lands."
    );
    eprintln!(
        "request_size transcript last-turn: total={total} tools≈{tools_bytes} \
         messages≈{messages_bytes} (ceiling {TRANSCRIPT_CEILING})"
    );
}
