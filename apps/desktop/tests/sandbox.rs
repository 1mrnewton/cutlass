//! The local sandbox end to end: the real `cutlass sandbox` child.

#![cfg(test)]

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::Duration;

use cutlass::sandbox::local_sandbox;
use cutlass_agent::{ProcessExecutor, Sandbox, SandboxError, SandboxExecutor};

fn program() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_cutlass"))
}

#[test]
fn a_script_runs_in_its_own_process_and_answers_json() {
    let mut sandbox = local_sandbox(program(), Some(Path::new("/projects/My Film")));
    let source = r#"
        let gaps = [];
        for i in 1..input.len() {
            let end = input[i - 1].start + input[i - 1].length;
            if input[i].start > end { gaps.push(#{ start: end, length: input[i].start - end }); }
        }
        gaps
    "#;
    let input = r#"[{"start":0,"length":10},{"start":15,"length":5}]"#;
    let answer: serde_json::Value =
        serde_json::from_str(&sandbox.run(source, input).expect("run")).expect("json");
    assert_eq!(answer, serde_json::json!([{"start": 10, "length": 5}]));
    // The project folder is refused by name before anything runs.
    assert_eq!(
        sandbox.run(r#"let p = "/projects/My Film/draft_info.json"; 1"#, "{}"),
        Err(SandboxError::ProjectPath)
    );
    // A script that never ends is stopped by its operation limit.
    let stopped = sandbox.run("loop { }", "{}").expect_err("stopped");
    assert!(stopped.to_string().contains("operations"), "{stopped}");
    // The child has no environment: nothing to read even if Rhai could.
    let error = sandbox.run(r#"open("x")"#, "{}").expect_err("no files");
    assert!(error.to_string().contains("open"), "{error}");
}

#[test]
fn a_runner_past_its_time_is_killed() {
    let mut slow = ProcessExecutor::new(
        program(),
        vec![OsString::from("sandbox")],
        Duration::from_millis(0),
    );
    let error = slow.execute("1", "{}").expect_err("killed");
    assert!(error.to_string().contains("stopped"), "{error}");
}
