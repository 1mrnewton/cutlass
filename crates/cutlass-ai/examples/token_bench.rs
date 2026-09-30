//! Live token/cost bench for one or more agent prompts against a fixture project.
//!
//! Not a test — needs a configured provider (`~/.cutlass/config.toml`) and network.
//!
//! ```bash
//! cargo run -p cutlass-ai --example token_bench
//! cargo run -p cutlass-ai --example token_bench -- "prompt 1" "prompt 2"
//! ```

use std::sync::atomic::AtomicBool;

use cutlass_ai::agent::{AgentConfig, AgentEvent, EngineBridge, run_prompt};
use cutlass_ai::config::provider_from_ai;
use cutlass_ai::provider::TokenUsage;
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

fn format_cost(cost: Option<f64>) -> String {
    match cost {
        Some(c) => format!("${c:.4}"),
        None => "n/a".into(),
    }
}

fn print_usage_line(prefix: &str, u: &TokenUsage) {
    println!(
        "{prefix} in={} (cached {}) out={} cost={}",
        u.input_tokens,
        u.cached_input_tokens,
        u.output_tokens,
        format_cost(u.cost)
    );
}

fn fixture_project() -> (Project, u64) {
    const R24: Rational = Rational::FPS_24;
    // Fixture: 3 tracks — main video (2 clips), overlay video (1), audio (1);
    // 60s source media at 1920x1080; one clip selected; playhead 0.
    let mut project = Project::new("token-bench", R24);
    let video = project.add_media(MediaSource::new(
        "/tmp/token-bench.mp4",
        1920,
        1080,
        R24,
        60 * 24,
        true,
    ));
    let audio_media = project.add_media(MediaSource::new(
        "/tmp/token-bench-audio.wav",
        0,
        0,
        R24,
        60 * 24,
        true,
    ));

    let main = project.add_track(TrackKind::Video, "V1");
    let overlay = project.add_track(TrackKind::Video, "V2");
    let audio = project.add_track(TrackKind::Audio, "A1");

    let selected = project
        .add_clip(
            main,
            video,
            TimeRange::at_rate(0, 240, R24),
            RationalTime::new(0, R24),
        )
        .unwrap();
    project
        .add_clip(
            main,
            video,
            TimeRange::at_rate(240, 240, R24),
            RationalTime::new(240, R24),
        )
        .unwrap();
    project
        .add_clip(
            overlay,
            video,
            TimeRange::at_rate(0, 120, R24),
            RationalTime::new(48, R24),
        )
        .unwrap();
    project
        .add_clip(
            audio,
            audio_media,
            TimeRange::at_rate(0, 480, R24),
            RationalTime::new(0, R24),
        )
        .unwrap();

    (project, selected.raw())
}

fn default_prompts() -> Vec<String> {
    vec![
        "cut the first 3 seconds of the selected clip".into(),
        "animate the selected clip to slide in from the left over the first second".into(),
    ]
}

fn main() {
    let prompts: Vec<String> = {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if args.is_empty() {
            default_prompts()
        } else {
            args
        }
    };

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

    let (project, selected) = fixture_project();
    let engine = Engine::with_project(EngineConfig { undo_limit: 64 }, project).unwrap();
    let mut host = EngineHost { engine };

    let context = EditorContext {
        selected_clips: vec![selected],
        playhead_seconds: 0.0,
        ..Default::default()
    };

    println!("model: {}\n", ai.model);

    let mut grand = TokenUsage::default();
    for (i, prompt) in prompts.iter().enumerate() {
        println!("── prompt {}/{}: {prompt:?} ──", i + 1, prompts.len());
        let cancel = AtomicBool::new(false);
        let outcome = run_prompt(
            &provider,
            &mut host,
            &context,
            &cutlass_ai::AgentExtensions::default(),
            &[],
            prompt,
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
                    print_usage_line("turn usage →", &u);
                }
            },
        );

        if !outcome.text.is_empty() && !outcome.text.ends_with('\n') {
            println!();
        }

        println!("── summary ──");
        println!("prompt: {prompt:?}");
        println!("status: {:?}", outcome.status);
        print!("provider-reported total: ");
        print_usage_line("", &outcome.usage);
        if outcome.usage.cost.is_none() {
            println!("cost: not reported");
        }
        println!("actions applied: {}", outcome.actions.len());
        println!();

        grand.add(&outcome.usage);
    }

    print!("grand total: ");
    print_usage_line("", &grand);
    if grand.cost.is_none() {
        println!("cost: not reported");
    }
}
