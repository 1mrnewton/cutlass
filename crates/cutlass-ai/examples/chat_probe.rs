//! Hand-run provider probe: one real completion through the configured
//! endpoint, tools attached. Not a test — needs a live endpoint.
//!
//! ```bash
//! # ~/.cutlass/config.toml:  [ai] base_url/model (see config.rs docs)
//! cargo run -p cutlass-ai --example chat_probe -- "what tools do you have?"
//! ```

use std::sync::atomic::AtomicBool;

use cutlass_ai::config::provider_from_ai;
use cutlass_ai::provider::{ChatProvider, ChatRequest, Message, ProviderStreamEvent};

fn main() {
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Reply with one sentence: what kind of assistant are you?".to_string());

    let path = cutlass_settings::default_config_path();
    let ai = match cutlass_settings::load(&path) {
        Ok(settings) => settings.ai,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    println!(
        "source: {}  endpoint: {}  model: {}\n",
        ai.source.key(),
        ai.base_url,
        ai.model
    );

    let provider = provider_from_ai(&ai).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });

    let messages = vec![
        Message::system(
            "You are the editing agent inside the Cutlass video editor. \
             You edit the timeline by calling tools.",
        ),
        Message::user(prompt),
    ];
    let mut tools = cutlass_ai::tool_specs();
    tools.push(cutlass_ai::wire::describe_project_spec());

    let cancel = AtomicBool::new(false);
    let turn = provider
        .chat(
            &ChatRequest {
                messages: &messages,
                tools: &tools,
            },
            &cancel,
            &mut |event| {
                use std::io::Write;
                match event {
                    ProviderStreamEvent::TextDelta(delta) => {
                        print!("{delta}");
                        std::io::stdout().flush().ok();
                    }
                    ProviderStreamEvent::ReasoningSummaryDelta(delta) => {
                        eprint!("{delta}");
                        std::io::stderr().flush().ok();
                    }
                }
            },
        )
        .unwrap_or_else(|e| {
            eprintln!("\nprovider error: {e}");
            std::process::exit(1);
        });

    println!("\n\nfinish: {:?}", turn.finish);
    for call in &turn.tool_calls {
        println!("tool call: {}({})", call.name, call.arguments);
    }
}
