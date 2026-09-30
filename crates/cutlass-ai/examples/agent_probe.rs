//! Hand-run end-to-end agent probe: a real prompt, the configured live
//! model, and a real engine editing a fixture project. Not a test.
//!
//! ```bash
//! cargo run -p cutlass-ai --example agent_probe -- "cut the first 3 seconds of the selected clip"
//! ```

use std::sync::atomic::AtomicBool;

use cutlass_ai::agent::{AgentConfig, AgentEvent, EngineBridge, run_prompt};
use cutlass_ai::config::provider_from_ai;
use cutlass_ai::{EditorContext, ProjectSummary, WireCommand, summarize, validate};
use cutlass_commands::EditOutcome;
use cutlass_engine::{ApplyOutcome, Engine, EngineConfig};
use cutlass_models::{MediaSource, Project, Rational, RationalTime, TimeRange, TrackKind};

struct EngineHost {
    engine: Engine,
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

fn main() {
    const R24: Rational = Rational::FPS_24;
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "cut the first 3 seconds of the selected clip".to_string());

    let path = cutlass_settings::default_config_path();
    let ai = match cutlass_settings::load(&path) {
        Ok(settings) => settings.ai,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    let provider = provider_from_ai(&ai).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });

    // Fixture: one 10s clip (of a 60s source) on V1, currently selected.
    let mut project = Project::new("probe", R24);
    let media = project.add_media(MediaSource::new(
        "/tmp/probe.mp4",
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

    let engine = Engine::with_project(EngineConfig { undo_limit: 64 }, project).unwrap();
    let mut host = EngineHost { engine };

    let context = EditorContext {
        selected_clips: vec![clip.raw()],
        playhead_seconds: 0.0,
        ..Default::default()
    };

    println!("model: {}  prompt: {prompt:?}\n", ai.model);
    let cancel = AtomicBool::new(false);
    let outcome = run_prompt(
        &provider,
        &mut host,
        &context,
        &cutlass_ai::AgentExtensions::default(),
        &[],
        &prompt,
        &AgentConfig::default(),
        &cancel,
        &mut |event| match event {
            AgentEvent::TextDelta(t) => {
                print!("{t}");
                use std::io::Write;
                std::io::stdout().flush().ok();
            }
            AgentEvent::ReasoningDelta(t) => eprintln!("  reasoning: {t}"),
            AgentEvent::Action(a) => println!("  ⚙ {}", a.description),
            AgentEvent::HostAction { name, summary } => println!("  ⚙ {name}: {summary}"),
            AgentEvent::Image(image) => println!("  ◫ {}", image.label),
            AgentEvent::Usage(u) => {
                let cost = u
                    .cost
                    .map(|c| format!("${c:.4}"))
                    .unwrap_or_else(|| "n/a".into());
                println!(
                    "  tokens: in={} (cached {}) out={} cost={}",
                    u.input_tokens, u.cached_input_tokens, u.output_tokens, cost
                );
            }
        },
    );

    println!("\nstatus: {:?}", outcome.status);
    let placed = host.engine.project().clip(clip).unwrap();
    println!(
        "clip {} now: timeline {:?} ticks, source start {:?}",
        clip.raw(),
        (placed.timeline.start.value, placed.timeline.duration.value),
        placed.source_range().map(|s| s.start.value),
    );
}
