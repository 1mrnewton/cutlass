//! `cutlass mcp /path/to/project`: the agent tools on stdio, with no window.
//!
//! The project opens and stays held while the client is connected. Each
//! change the client makes is saved at once, so a crash loses nothing, and
//! the lock is let go when stdin closes. The model loop stays in the
//! client. Approvals need a person, so this host refuses them, and the
//! client sees the refusal as a tool error.
//!
//! With no window there is no screen to play on: seek moves the playhead a
//! `.png` export draws, and an export or a track runs to the end before it
//! answers.

use std::cell::RefCell;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use cutlass_agent::{
    ApprovalAction, Approver, Catalog, DraftTools, Export, Forward, McpClient, McpError, McpServer,
    McpTransport, Media, MediaInfo, Packages, Player, Registry, SettingsTool, Speech, ToolSink,
    TrackAsk, Tracking,
};

use crate::shell::{ExportRequest, Shell, ShellError};

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
    let shell = Rc::new(RefCell::new(shell));
    let registry = Registry::new(
        DraftTools::new(session),
        settings,
        Headless(Rc::clone(&shell)),
        Probe,
        Headless(Rc::clone(&shell)),
        Bundled,
        (
            Bundled,
            Headless(Rc::clone(&shell)),
            crate::sandbox::local_sandbox(crate::sandbox::this_program(), Some(path)),
            Refuse,
            McpClient::new(NoServers),
            Headless(Rc::clone(&shell)),
        ),
    );
    let mut server = McpServer::new(Logged(registry), "cutlass", env!("CARGO_PKG_VERSION"));
    cutlass_logger::info!("mcp: serving {}", path.display());
    pump(&mut server, input, output, |server| {
        let session = server.sink().0.draft.session();
        if session.revision() != saved {
            saved = session.revision();
            let mut shell = shell.borrow_mut();
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

/// A tool sink that logs each call it runs, by name and outcome.
struct Logged<S>(S);

impl<S: ToolSink> ToolSink for Logged<S> {
    fn run_tool(&mut self, call: &cutlass_agent::ToolCall) -> cutlass_agent::ToolOutcome {
        let started = std::time::Instant::now();
        let outcome = self.0.run_tool(call);
        crate::shell::log_tool_call("mcp", call, &outcome, started.elapsed());
        outcome
    }
}

/// The player and export on the project this host holds.
struct Headless(Rc<RefCell<Shell>>);

impl Player for Headless {
    fn seek(&mut self, time_us: i64) -> Result<(), String> {
        if time_us < 0 {
            return Err("the time is before the start".to_owned());
        }
        self.0.borrow_mut().seek(cutlass_model::Micros(time_us));
        Ok(())
    }

    fn play(&mut self) -> Result<(), String> {
        Err(NO_SCREEN.to_owned())
    }

    fn pause(&mut self) -> Result<(), String> {
        Err(NO_SCREEN.to_owned())
    }
}

const NO_SCREEN: &str = "cutlass mcp has no screen to play on; seek moves the playhead that a .png export draws, and the Cutlass window plays";

impl Speech for Headless {
    fn start(&mut self, start_us: i64, duration_us: i64) -> Result<String, String> {
        self.0
            .borrow_mut()
            .start_hearing(start_us, duration_us)
            .map_err(|error| error.to_string())
    }

    fn poll(&mut self, job_id: &str) -> Result<Option<String>, String> {
        self.0
            .borrow_mut()
            .hearing(job_id)
            .map_err(|error| error.to_string())
    }
}

impl Tracking for Headless {
    /// Tracks to the end before it answers, as an export does, so the
    /// next `track.poll` has the path.
    fn start(&mut self, ask: &TrackAsk) -> Result<String, String> {
        let mut shell = self.0.borrow_mut();
        let id = shell.agent_tracks_next_id();
        shell
            .start_agent_track(id.clone(), ask)
            .map_err(|error| error.to_string())?;
        while shell.poll_job() {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        Ok(id)
    }

    fn poll(&mut self, job_id: &str) -> Result<Option<cutlass_model::TrackPath>, String> {
        self.0
            .borrow_mut()
            .agent_track(job_id)
            .map_err(|error| error.to_string())
    }

    fn clip_path(
        &self,
        draft: &cutlass_model::Draft,
        clip: &cutlass_model::SegmentId,
        path: &cutlass_model::TrackPath,
    ) -> Option<cutlass_model::TrackPath> {
        crate::shell::clip_path(draft, clip, path)
    }
}

impl Export for Headless {
    /// Writes the file before it answers: a `.png` is the frame at the
    /// playhead, anything else the whole project as H.264. An existing
    /// file is not replaced, as no one is here to allow it.
    fn start(&mut self, path: &str) -> Result<String, String> {
        if path.trim().is_empty() {
            return Err("name the file to export to".to_owned());
        }
        let path = PathBuf::from(path);
        let codec = if path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("png"))
        {
            cutlass_codec::CodecId::Png
        } else {
            cutlass_codec::CodecId::H264
        };
        let mut shell = self.0.borrow_mut();
        shell
            .export(ExportRequest {
                path: path.clone(),
                codec,
                ..ExportRequest::default()
            })
            .map_err(|error| error.to_string())?;
        while shell.poll_job() {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        if codec != cutlass_codec::CodecId::Png && shell.exported() != Some(path.as_path()) {
            return Err(shell.notice().to_owned());
        }
        Ok(format!("finished, wrote {}", path.display()))
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

/// The effects, filters, transitions and animations Cutlass ships.
pub(crate) struct Bundled;

impl Catalog for Bundled {
    fn search(&mut self, query: &str) -> Result<Vec<String>, String> {
        Ok(cutlass_editor::catalog_search(query))
    }
}

/// Every package ships in the app, so one is ready when Cutlass has it,
/// and there is nothing to fetch when it does not.
impl Packages for Bundled {
    fn ensure(&mut self, package_id: &str) -> Result<(), String> {
        if cutlass_editor::is_bundled(package_id) {
            Ok(())
        } else {
            Err(format!(
                "Cutlass has no package `{package_id}`; every package ships in the app, \
                 and catalog.search lists their ids"
            ))
        }
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
                "op": "import",
                "medium": {"video": {"id": "media", "type": "video", "duration": 1_000_000}},
            }, {
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

    fn answers(output: Vec<u8>) -> Vec<Value> {
        String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    #[test]
    fn with_no_window_seek_and_export_write_files() {
        let project = Project::new("export");
        // A fifth of a second of blue, 64 by 32.
        let mut draft = cutlass_model::Draft::empty("draft");
        draft.canvas_config.width = 64;
        draft.canvas_config.height = 32;
        draft
            .materials
            .material_colors
            .push(cutlass_model::MaterialColorConfig {
                id: "color".into(),
                is_color_clip: true,
                solid_color: "0xFF336699".to_owned(),
                ..cutlass_model::MaterialColorConfig::default()
            });
        draft.tracks[0].segments.push(cutlass_model::Segment::new(
            "blue",
            "color",
            cutlass_model::TimeRange::new(0, 200_000),
        ));
        draft.duration = cutlass_model::Micros(200_000);
        cutlass_store::Folder::new(draft).save(&project.0).unwrap();
        let still = project.0.join("still.png");
        let movie = project.0.join("cut.mp4");
        let call = |id: i64, name: &str, arguments: Value| {
            json!({"jsonrpc":"2.0","id":id,"method":"tools/call",
                "params":{"name":name,"arguments":arguments}})
        };
        let input = lines(&[
            json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
            call(2, "player_seek", json!({"input": "100000"})),
            call(
                3,
                "export_start",
                json!({"input": format!("new {}", still.display())}),
            ),
            call(
                4,
                "export_start",
                json!({"input": format!("new {}", movie.display())}),
            ),
            call(
                5,
                "export_start",
                json!({"input": format!("new {}", movie.display())}),
            ),
            call(6, "player_play", json!({})),
            call(7, "catalog_search", json!({"input": "dissolve"})),
            call(8, "speech_start", json!({"input": "0 100000"})),
        ]);
        let mut output = Vec::new();
        serve(Some(&project.0), input.as_bytes(), &mut output).unwrap();
        let answers = answers(output);
        let text = |at: usize| {
            answers[at]["result"]["content"][0]["text"]
                .as_str()
                .unwrap()
                .to_owned()
        };
        assert_eq!(answers.len(), 8);
        // A color clip has no sound to hear.
        assert!(text(7).contains("nothing with sound"), "{}", text(7));
        assert!(text(6).contains("transition cutlass-"), "{}", text(6));
        assert_eq!(answers[1]["result"]["isError"], false, "{}", text(1));
        assert!(text(2).contains("still.png"), "{}", text(2));
        assert_eq!(&std::fs::read(&still).unwrap()[1..4], b"PNG");
        assert!(text(3).contains("cut.mp4"), "{}", text(3));
        assert_eq!(&std::fs::read(&movie).unwrap()[4..8], b"ftyp");
        // An existing file needs an allow no one can give here.
        assert_eq!(answers[4]["result"]["isError"], true, "{}", text(4));
        assert_eq!(answers[5]["result"]["isError"], true);
        assert!(text(5).contains("no screen"), "{}", text(5));
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
                "op": "import",
                "medium": {"video": {"id": "media", "type": "video", "duration": 1_000_000}},
            }, {
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

    #[test]
    #[ignore = "decodes seconds of a real movie; slow in a debug build"]
    fn with_no_window_a_track_runs_to_the_end_and_is_saved() {
        let movie = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../local-files/16929689_1920_1080_30fps.mp4");
        if !movie.is_file() {
            eprintln!("skipped: the movie is missing");
            return;
        }
        let project = Project::new("track");
        let mut draft = crate::open_movie::movie_draft(&movie, 1920, 1080, false).unwrap();
        draft.tracks[0].segments[0].target = cutlass_model::TimeRange::new(0, 4_000_000);
        draft.tracks[0].segments[0].source_range =
            Some(cutlass_model::TimeRange::new(0, 4_000_000));
        draft.materials.videos[0].duration = cutlass_model::Micros(4_000_000);
        draft.duration = cutlass_model::Micros(4_000_000);
        let target = draft.tracks[0].segments[0].id.clone();
        let material = draft.tracks[0].segments[0].material_id.clone();
        let mut sticker = cutlass_model::Segment::new(
            "follower",
            material.as_str(),
            cutlass_model::TimeRange::new(0, 4_000_000),
        );
        sticker.clip.transform.y = 0.3;
        sticker.uniform_scale.value = 0.25;
        let mut lane = cutlass_model::Track::main("stickers");
        lane.kind = cutlass_model::TrackType::Sticker;
        lane.segments.push(sticker);
        draft.tracks.push(lane);
        let _ = std::fs::remove_dir_all(&project.0);
        cutlass_store::Folder::new(draft).save(&project.0).unwrap();
        // The shirt at one second, in canvas units.
        let unit = |x: f64, y: f64| (2.0 * x / 1920.0 - 1.0, 1.0 - 2.0 * y / 1080.0);
        let (a, b) = (unit(915.0, 435.0), unit(1080.0, 600.0));
        let input = lines(&[
            json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
            json!({"jsonrpc":"2.0","id":2,"method":"tools/call",
                "params":{"name":"track_start","arguments":{
                    "target": target.as_str(), "at_us": 1_000_000,
                    "box": [a.0, a.1, b.0, b.1], "followers": ["follower"],
                    "masks": [target.as_str()]}}}),
            json!({"jsonrpc":"2.0","id":3,"method":"tools/call",
                "params":{"name":"track_poll","arguments":{"input":"track-1"}}}),
        ]);
        let mut output = Vec::new();
        serve(Some(&project.0), input.as_bytes(), &mut output).unwrap();
        let answers = answers(output);
        let started = answers[1]["result"]["content"][0]["text"].as_str().unwrap();
        assert_eq!(started, "job=track-1 revision=0");
        let polled = answers[2]["result"]["content"][0]["text"].as_str().unwrap();
        eprintln!("{polled}");
        assert!(polled.starts_with("revision=1 frames="), "{polled}");
        let opened = cutlass_store::open(&project.0).unwrap();
        let draft = &opened.folder.draft;
        assert_eq!(draft.materials.video_trackings.len(), 1);
        let follower = &draft.tracks[1].segments[0];
        assert!(cutlass_model::key_offsets(follower).len() > 2);
        // The movie's own mask is a box on the shirt that shrinks with it.
        let movie = &draft.tracks[0].segments[0];
        let mask = cutlass_effects::segment_mask(draft, movie).unwrap();
        let size = |t: i64| {
            mask.at(movie, &draft.keyframe_graph_list, cutlass_model::Micros(t))
                .width
        };
        assert!(size(3_500_000) < size(1_000_000) * 0.9);
    }
}
