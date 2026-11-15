//! `cutlass mcp` end to end: the real process, stopped the way a client
//! that quits stops it.

#![cfg(test)]
// Signals are sent with `kill`.
#![cfg(unix)]

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

/// A home folder and a project in it, both removed when the test ends, so
/// the child reads no settings of the user's and writes its log here.
struct Home(PathBuf);

impl Home {
    fn new(name: &str) -> (Self, PathBuf) {
        let root =
            std::env::temp_dir().join(format!("cutlass-mcp-signal-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("home");
        let project = cutlass_store::create_project(&root).expect("project");
        (Self(root), project)
    }
}

impl Drop for Home {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Start `cutlass mcp` on `project`, rename the project through it, and
/// wait for the answer, so the child holds the project with an edit saved.
/// Its stdin stays open in the child that is returned.
fn serving(home: &Home, project: &Path) -> Child {
    let mut child = Command::new(env!("CARGO_BIN_EXE_cutlass"))
        .arg("mcp")
        .arg(project)
        .env("HOME", &home.0)
        .env_remove("XDG_DATA_HOME")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("cutlass mcp");
    let rename = serde_json::json!({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{
        "name":"draft_apply",
        "arguments":{"expected_revision":0,
            "requests":[{"op":"set_project_name","name":"kept"}]}}});
    let stdin = child.stdin.as_mut().expect("stdin");
    stdin
        .write_all(format!("{rename}\n").as_bytes())
        .and_then(|()| stdin.flush())
        .expect("write");
    let stdout = child.stdout.take().expect("stdout");
    let (answers, answered) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut line = String::new();
        if BufReader::new(stdout).read_line(&mut line).is_ok() {
            let _ = answers.send(line);
        }
    });
    let answer = answered
        .recv_timeout(Duration::from_secs(60))
        .expect("an answer to the rename");
    assert!(answer.contains("revision=1 committed=true"), "{answer}");
    child
}

/// Send `signal` to `child` and wait for it to end.
fn stopped_by(child: &mut Child, signal: &str) -> ExitStatus {
    let sent = Command::new("kill")
        .arg(format!("-{signal}"))
        .arg(child.id().to_string())
        .status()
        .expect("kill");
    assert!(sent.success(), "kill -{signal}");
    let until = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(status) = child.try_wait().expect("wait") {
            return status;
        }
        if Instant::now() >= until {
            let _ = child.kill();
            panic!("cutlass mcp kept running after {signal}");
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn a_signal_closes_the_project_while_stdin_is_still_open() {
    for signal in ["INT", "TERM", "HUP"] {
        let (home, project) = Home::new(signal);
        let mut child = serving(&home, &project);
        let status = stopped_by(&mut child, signal);
        assert!(status.success(), "{signal}: {status}");
        let hold = cutlass_store::hold(&project).expect("the lock was let go");
        assert!(!hold.unclean(), "{signal} left the project not closed");
        drop(hold);
        let opened = cutlass_store::open(&project).expect("open");
        assert_eq!(opened.folder.draft.name, "kept", "{signal}");
    }
}

#[test]
fn a_kill_leaves_the_project_marked_and_its_last_save_whole() {
    // The one stop no process can answer: the edit was saved when it was
    // made, and the next open is told the folder was not closed.
    let (home, project) = Home::new("KILL");
    let mut child = serving(&home, &project);
    let status = stopped_by(&mut child, "KILL");
    assert!(!status.success(), "{status}");
    let hold = cutlass_store::hold(&project).expect("a dead holder's lock is free");
    assert!(hold.unclean());
    drop(hold);
    let opened = cutlass_store::open(&project).expect("open");
    assert_eq!(opened.folder.draft.name, "kept");
}
