//! `cutlass mcp /path/to/project`: the agent tools on stdio, with no window.
//!
//! The project opens and stays held while the client is connected. Each
//! change the client makes is saved at once, so a crash loses nothing, and
//! the lock is let go when stdin closes. The model loop stays in the
//! client. Approvals need a person, so this host refuses them, and the
//! client sees the refusal as a tool error.

use std::io::{BufRead, Write};
use std::path::Path;

use cutlass_agent::{
    ApprovalAction, Approver, Catalog, DraftTools, Export, Forward, McpClient, McpError, McpServer,
    McpTransport, Media, MediaInfo, Packages, Player, Registry, Sandbox, SandboxError,
    SettingsTool, Speech, ToolSink,
};

use crate::shell::{Shell, ShellError};

/// Serve the project at `path` on stdin and stdout until stdin closes.
pub fn run(path: Option<&Path>) -> Result<(), String> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    serve(path, stdin.lock(), stdout.lock())
}

/// Serve `path`, reading messages from `input` and writing answers to
/// `output`, a line each.
pub fn serve(path: Option<&Path>, input: impl BufRead, output: impl Write) -> Result<(), String> {
    let Some(path) = path else {
        return Err("name the project folder: cutlass mcp /path/to/project".to_owned());
    };
    let settings_path = Shell::settings_path();
    let loaded = crate::load_settings_file(&settings_path);
    let (decoders, encoders) = crate::register_codecs_with(&loaded.settings.hardware);
    let mut shell = Shell::new(decoders, encoders, loaded.settings);
    match shell.open_folder(path) {
        Ok(()) => {}
        Err(ShellError::Store(cutlass_store::StoreError::Held { .. })) => {
            // The window holds it: send each call there.
            let Some((address, token)) = cutlass_store::held_address(path) else {
                return Err(format!(
                    "{} is open in another Cutlass that does not take calls; close it there first",
                    path.display()
                ));
            };
            let forward = Forward::connect(&address, &token)?;
            cutlass_logger::info!("mcp: forwarding {} to {address}", path.display());
            let mut server = McpServer::new(forward, "cutlass", env!("CARGO_PKG_VERSION"));
            return pump(&mut server, input, output, |_| Ok(()));
        }
        Err(other) => return Err(other.to_string()),
    }
    let session = shell
        .session()
        .cloned()
        .ok_or_else(|| "the project did not open".to_owned())?;
    let mut saved = session.revision();
    let settings = SettingsTool::open(&settings_path).map_err(|error| error.to_string())?;
    let registry = Registry::new(
        DraftTools::new(session),
        settings,
        NoPlayer,
        Probe,
        Unbuilt(NOT_HERE),
        Unbuilt(NOT_HERE),
        (
            Unbuilt(NOT_HERE),
            Unbuilt(NOT_HERE),
            NoSandbox,
            Refuse,
            McpClient::new(NoServers),
        ),
    );
    let mut server = McpServer::new(registry, "cutlass", env!("CARGO_PKG_VERSION"));
    cutlass_logger::info!("mcp: serving {}", path.display());
    pump(&mut server, input, output, |server| {
        let session = server.sink().draft.session();
        if session.revision() != saved {
            saved = session.revision();
            shell
                .adopt_session(session.clone())
                .and_then(|()| shell.save())
                .map_err(|error| format!("the project was not saved: {error}"))?;
        }
        Ok(())
    })
}

/// Answer each line of `input` on `output` until it closes, calling
/// `after` once each message is handled.
fn pump<S: ToolSink>(
    server: &mut McpServer<S>,
    input: impl BufRead,
    mut output: impl Write,
    mut after: impl FnMut(&McpServer<S>) -> Result<(), String>,
) -> Result<(), String> {
    for line in input.lines() {
        let line = line.map_err(|error| error.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let answer = server.handle(&line);
        after(server)?;
        if let Some(answer) = answer {
            output
                .write_all(answer.as_bytes())
                .and_then(|()| output.write_all(b"\n"))
                .and_then(|()| output.flush())
                .map_err(|error| error.to_string())?;
        }
    }
    cutlass_logger::info!("mcp: the client closed stdin");
    Ok(())
}

const NOT_HERE: &str = "not available from cutlass mcp; open the project in the Cutlass window";

struct NoPlayer;

impl Player for NoPlayer {
    fn seek(&mut self, _time_us: i64) -> Result<(), String> {
        Err(format!("the player is {NOT_HERE}"))
    }

    fn play(&mut self) -> Result<(), String> {
        Err(format!("the player is {NOT_HERE}"))
    }

    fn pause(&mut self) -> Result<(), String> {
        Err(format!("the player is {NOT_HERE}"))
    }
}

/// Reads a file's header for its length, size and streams.
pub(crate) struct Probe;

impl Media for Probe {
    fn probe(&mut self, path: &str) -> Result<MediaInfo, String> {
        match cutlass_probe::probe(Path::new(path)).map_err(|error| error.to_string())? {
            cutlass_probe::Probed::Video {
                width,
                height,
                duration,
                ..
            } => Ok(MediaInfo {
                duration_us: duration,
                width,
                height,
                streams: vec!["video".to_owned()],
            }),
            cutlass_probe::Probed::Audio { duration } => Ok(MediaInfo {
                duration_us: duration,
                width: 0,
                height: 0,
                streams: vec!["audio".to_owned()],
            }),
        }
    }
}

/// Transcription, the catalog, packages and export, which this host does
/// not run yet; the text says where they are.
pub(crate) struct Unbuilt(pub(crate) &'static str);

impl Speech for Unbuilt {
    fn start(&mut self, _start_us: i64, _duration_us: i64) -> Result<String, String> {
        Err(format!("transcription is {}", self.0))
    }

    fn poll(&mut self, _job_id: &str) -> Result<String, String> {
        Err(format!("transcription is {}", self.0))
    }
}

impl Catalog for Unbuilt {
    fn search(&mut self, _query: &str) -> Result<Vec<String>, String> {
        Err(format!("the catalog is {}", self.0))
    }
}

impl Packages for Unbuilt {
    fn ensure(&mut self, _package_id: &str) -> Result<(), String> {
        Err(format!("packages are {}", self.0))
    }
}

impl Export for Unbuilt {
    fn start(&mut self, _path: &str) -> Result<String, String> {
        Err(format!("export is {}", self.0))
    }
}

pub(crate) struct NoSandbox;

impl Sandbox for NoSandbox {
    fn run(&mut self, _source: &str, _input_json: &str) -> Result<String, SandboxError> {
        Err(SandboxError::Executor(
            "no sandbox runner is installed".to_owned(),
        ))
    }
}

/// No one is here to allow anything.
struct Refuse;

impl Approver for Refuse {
    fn decide(&mut self, action: &ApprovalAction) -> bool {
        cutlass_logger::info!("mcp: refused {action:?}, as no one can allow it here");
        false
    }
}

struct NoServers;

impl McpTransport for NoServers {
    fn call(
        &mut self,
        server_name: &str,
        _tool: &str,
        _arguments: &str,
    ) -> Result<String, McpError> {
        Err(McpError::Transport(format!(
            "cutlass mcp does not call other servers, so {server_name} was not reached"
        )))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::serve;

    struct Project(std::path::PathBuf);

    impl Project {
        fn new(name: &str) -> Self {
            let path =
                std::env::temp_dir().join(format!("cutlass-mcp-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            let mut draft = cutlass_model::Draft::empty("draft");
            draft.tracks[0].id = cutlass_model::TrackId::new("main");
            cutlass_store::Folder::new(draft).save(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for Project {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn lines(messages: &[Value]) -> String {
        messages
            .iter()
            .map(|message| format!("{message}\n"))
            .collect()
    }

    #[test]
    fn a_client_reads_edits_and_the_edit_is_saved() {
        let project = Project::new("edit");
        let place = json!({
            "expected_revision": 0,
            "requests": [{
                "op": "place",
                "track_id": "main",
                "segment": serde_json::to_value(cutlass_model::Segment::new(
                    "a",
                    "media",
                    cutlass_model::TimeRange::new(0, 1_000_000),
                ))
                .unwrap(),
            }],
        });
        let input = lines(&[
            json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
            json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
            json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}),
            json!({"jsonrpc":"2.0","id":3,"method":"tools/call",
                "params":{"name":"draft_query","arguments":{"input":"project"}}}),
            json!({"jsonrpc":"2.0","id":4,"method":"tools/call",
                "params":{"name":"draft_apply","arguments":place}}),
            json!({"jsonrpc":"2.0","id":5,"method":"tools/call",
                "params":{"name":"player_play","arguments":{}}}),
        ]);
        let mut output = Vec::new();
        serve(Some(&project.0), input.as_bytes(), &mut output).unwrap();
        let answers: Vec<Value> = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(answers.len(), 5);
        assert_eq!(answers[0]["result"]["serverInfo"]["name"], "cutlass");
        assert!(answers[1]["result"]["tools"].as_array().unwrap().len() > 10);
        let read = answers[2]["result"]["content"][0]["text"].as_str().unwrap();
        assert!(read.contains("revision=0"), "{read}");
        assert_eq!(
            answers[3]["result"]["content"][0]["text"],
            "revision=1 committed=true"
        );
        assert_eq!(answers[4]["result"]["isError"], true);
        let opened = cutlass_store::open(&project.0).unwrap();
        assert_eq!(opened.folder.draft.tracks[0].segments.len(), 1);
        // The lock was let go when stdin closed.
        assert!(cutlass_store::hold(&project.0).is_ok());
    }

    #[test]
    fn a_held_project_or_no_path_is_refused() {
        let project = Project::new("held");
        let _hold = cutlass_store::hold(&project.0).unwrap();
        let error = serve(Some(&project.0), "".as_bytes(), Vec::new()).unwrap_err();
        assert!(error.contains("does not take calls"), "{error}");
        let error = serve(None, "".as_bytes(), Vec::new()).unwrap_err();
        assert!(error.contains("cutlass mcp /path/to/project"), "{error}");
    }

    #[test]
    fn a_project_open_in_the_window_is_served_through_it() {
        let project = Project::new("forward");
        let (decoders, encoders) = crate::register_codecs();
        let mut desktop =
            crate::shell::Shell::new(decoders, encoders, cutlass_settings::Settings::default());
        desktop.open_folder(&project.0).unwrap();
        desktop.listen(std::sync::Arc::new(|| {}));
        let place = json!({
            "expected_revision": 0,
            "requests": [{
                "op": "place",
                "track_id": "main",
                "segment": serde_json::to_value(cutlass_model::Segment::new(
                    "a",
                    "media",
                    cutlass_model::TimeRange::new(0, 1_000_000),
                ))
                .unwrap(),
            }],
        });
        let input = lines(&[
            json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
            json!({"jsonrpc":"2.0","id":2,"method":"tools/call",
                "params":{"name":"draft_apply","arguments":place}}),
        ]);
        let path = project.0.clone();
        let client = std::thread::spawn(move || {
            let mut output = Vec::new();
            serve(Some(&path), input.as_bytes(), &mut output).map(|()| output)
        });
        let until = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while !client.is_finished() && std::time::Instant::now() < until {
            desktop.serve_forwarded();
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let output = client.join().unwrap().unwrap();
        let answers: Vec<Value> = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(answers.len(), 2);
        assert_eq!(
            answers[1]["result"]["content"][0]["text"],
            "revision=1 committed=true"
        );
        // The window holds the edit, as one undo step.
        assert_eq!(desktop.draft().unwrap().tracks[0].segments.len(), 1);
        desktop.undo().unwrap();
        assert!(desktop.draft().unwrap().tracks[0].segments.is_empty());
    }
}
