//! The local runner's language: Rhai, run in a child process.
//!
//! Rhai has no file, socket, process or environment calls, and module
//! imports are compiled out, so a script reaches nothing but its input.
//! Limits on operations, depth, strings, arrays and maps stop a runaway
//! script; the parent also kills the child after a wall-clock limit
//! ([`ProcessExecutor`](crate::ProcessExecutor)).

use std::io::{Read, Write};

use rhai::{Dynamic, Engine, Scope};
use serde_json::Value;

use crate::sandbox::{SandboxError, ScriptCall};

fn engine() -> Engine {
    let mut engine = Engine::new();
    engine.set_max_operations(50_000_000);
    engine.set_max_call_levels(64);
    engine.set_max_expr_depths(128, 64);
    engine.set_max_string_size(4 << 20);
    engine.set_max_array_size(1 << 20);
    engine.set_max_map_size(1 << 16);
    // `print` and `debug` would write into the answer; they say nothing.
    engine.on_print(|_| {});
    engine.on_debug(|_, _, _| {});
    engine.disable_symbol("eval");
    engine
}

/// Run `source` with `input` (JSON) bound to the variable `input`; the
/// script's last value comes back as JSON text.
pub fn run_script(source: &str, input_json: &str) -> Result<String, SandboxError> {
    let input: Value = serde_json::from_str(input_json).map_err(|_| SandboxError::NotJson)?;
    let input: Dynamic = rhai::serde::to_dynamic(&input)
        .map_err(|error| SandboxError::Executor(error.to_string()))?;
    let mut scope = Scope::new();
    scope.push("input", input);
    let value: Dynamic = engine()
        .eval_with_scope(&mut scope, source)
        .map_err(|error| SandboxError::Executor(error.to_string()))?;
    let value: Value = rhai::serde::from_dynamic(&value)
        .map_err(|error| SandboxError::Executor(error.to_string()))?;
    Ok(value.to_string())
}

/// The child's side: one [`ScriptCall`] as JSON on stdin, and on stdout the
/// result as `{"ok": <json>}` or `{"error": "<message>"}`.
pub fn serve_script(input: &mut dyn Read, output: &mut dyn Write) -> std::io::Result<()> {
    let mut text = String::new();
    input.read_to_string(&mut text)?;
    let answer = match serde_json::from_str::<ScriptCall>(&text) {
        Ok(call) => match run_script(&call.source, &call.input) {
            Ok(json) => format!("{{\"ok\":{json}}}"),
            Err(SandboxError::Executor(message)) => {
                serde_json::json!({ "error": message }).to_string()
            }
            Err(error) => serde_json::json!({ "error": error.to_string() }).to_string(),
        },
        Err(error) => serde_json::json!({ "error": format!("bad call: {error}") }).to_string(),
    };
    output.write_all(answer.as_bytes())?;
    output.flush()
}

#[cfg(test)]
mod tests {
    use super::{run_script, serve_script};
    use crate::sandbox::SandboxError;

    #[test]
    fn a_script_reads_its_input_and_returns_json() {
        let source = r#"
            let gaps = [];
            let clips = input.clips;
            for i in 1..clips.len() {
                let end = clips[i - 1].start + clips[i - 1].length;
                if clips[i].start > end {
                    gaps.push(#{ start: end, length: clips[i].start - end });
                }
            }
            #{ gaps: gaps, count: gaps.len() }
        "#;
        let input = r#"{"clips":[{"start":0,"length":10},{"start":15,"length":5},{"start":20,"length":1}]}"#;
        let out: serde_json::Value =
            serde_json::from_str(&run_script(source, input).unwrap()).unwrap();
        assert_eq!(
            out,
            serde_json::json!({"gaps":[{"start":10,"length":5}],"count":1})
        );
    }

    #[test]
    fn a_script_has_no_files_no_imports_no_eval_and_stops_when_it_runs_away() {
        for source in [
            r#"import "x" as x; 1"#,
            r#"eval("1")"#,
            r#"open_file("/etc/passwd")"#,
            "loop { }",
            "fn f(n) { f(n + 1) } f(0)",
        ] {
            assert!(
                matches!(run_script(source, "{}"), Err(SandboxError::Executor(_))),
                "{source}"
            );
        }
        assert_eq!(run_script("1", "not json"), Err(SandboxError::NotJson));
        // Printing does not leak into the answer.
        assert_eq!(run_script(r#"print("hi"); 2"#, "{}").unwrap(), "2");
    }

    #[test]
    fn the_child_answers_ok_or_error() {
        let mut out = Vec::new();
        serve_script(
            &mut &br#"{"source":"input.a * 2","input":"{\"a\":21}"}"#[..],
            &mut out,
        )
        .unwrap();
        assert_eq!(String::from_utf8(out).unwrap(), r#"{"ok":42}"#);
        let mut out = Vec::new();
        serve_script(&mut &b"{}"[..], &mut out).unwrap();
        assert!(String::from_utf8(out).unwrap().contains("bad call"));
    }
}
