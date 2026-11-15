//! `cutlass mcp /path/to/project`: the agent tools on stdio, with no window.
//!
//! The project opens and stays held while the client is connected. Each
//! change the client makes is saved at once, so a crash loses nothing, and
//! the lock is let go when stdin closes or the process is told to stop: a
//! client that quits signals the server it started, and does not always
//! close stdin first. The model loop stays in the client. Approvals need
//! a person, so this host refuses them, and the client sees the refusal as
//! a tool error.
//!
//! A project another app holds is not opened: the calls go to that app
//! instead. This host listens too, so a second `cutlass mcp` on the same
//! project attaches to it, and when the Cutlass window opens the project
//! this host saves, lets it go and sends its later calls to the window. A
//! holder that goes away is found again on the next call: the project's
//! new holder, or this host itself.
//!
//! With no window there is no screen to play on: seek moves the playhead a
//! `.png` export draws, and an export or a track runs to the end before it
//! answers. An export draws on a GPU device opened with no window, as the
//! window draws on its own, so an effect clip is in the file. On a machine
//! that gives none the file is written without its effect clips, and the
//! answer names them.

use std::cell::RefCell;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::mpsc::{Sender, channel};
use std::time::{Duration, Instant};

use cutlass_agent::{
    ApprovalAction, Approver, Catalog, DraftTools, Export, Forward, LocalSandbox, McpClient,
    McpServer, Media, MediaInfo, Packages, Player, ProcessExecutor, Registry, SettingsTool,
    SkillLibrary, Speech, Switched, ToolCall, ToolOutcome, ToolSink, TrackAsk, Tracking, Wake,
};

use crate::mcp_servers::McpServers;
use crate::shell::{ExportRequest, Shell, ShellError};

/// Serve the project at `path` on stdin and stdout until stdin closes or
/// the process is told to stop.
pub fn run(path: Option<&Path>) -> Result<(), String> {
    let stdout = std::io::stdout();
    serve_with(
        path,
        &Shell::settings_path(),
        std::io::BufReader::new(std::io::stdin()),
        stdout.lock(),
        stop_on_signal,
    )
}

/// Ends the serving loop from outside it, as a closed stdin ends it from
/// inside: the project is closed and its lock let go.
struct Stop(Sender<Event>);

impl Stop {
    fn stop(&self) {
        let _ = self.0.send(Event::Stopped);
    }
}

/// End the loop when the process is told to stop: SIGINT, SIGTERM or
/// SIGHUP, and on Windows Ctrl-C, Ctrl-Break or a closed console. Left to
/// end the process themselves, they would leave the project marked as not
/// closed, and its next open would say Cutlass did not close.
fn stop_on_signal(stop: Stop) {
    if let Err(error) = ctrlc::set_handler(move || stop.stop()) {
        cutlass_logger::warn!("mcp: a signal will not close the project: {error}");
    }
}

/// What the serving loop waits for.
enum Event {
    /// A line from the client.
    Line(String),
    /// Bytes from the client that are not text.
    Garbled,
    /// The client closed its end.
    Closed,
    /// The process was told to stop.
    Stopped,
    /// Another host sent a call to the project socket.
    Socket,
}

/// Serve `path`, reading messages from `input` and writing answers to
/// `output`, a line each.
pub fn serve(
    path: Option<&Path>,
    input: impl BufRead + Send + 'static,
    output: impl Write,
) -> Result<(), String> {
    serve_with(path, &Shell::settings_path(), input, output, |_| {})
}

/// [`serve`], with the user's settings read from and written to
/// `settings`. `stopper` is handed what ends the loop from outside, before
/// the project opens, so a signal that comes while it opens closes it too.
fn serve_with(
    path: Option<&Path>,
    settings: &Path,
    input: impl BufRead + Send + 'static,
    mut output: impl Write,
    stopper: impl FnOnce(Stop),
) -> Result<(), String> {
    let Some(path) = path else {
        return Err("name the project folder: cutlass mcp /path/to/project".to_owned());
    };
    let (events, waiting) = channel();
    stopper(Stop(events.clone()));
    let socket = events.clone();
    let wake: Wake = Arc::new(move || {
        let _ = socket.send(Event::Socket);
    });
    let mut host = Host {
        path: path.to_path_buf(),
        settings: settings.to_path_buf(),
        place: Place::Unset,
        let_go: None,
        wake,
    };
    host.attach()?;
    let mut server = McpServer::new(Logged(host), "cutlass", env!("CARGO_PKG_VERSION"));
    std::thread::Builder::new()
        .name("mcp client".to_owned())
        .spawn(move || read_lines(input, &events))
        .map_err(|error| error.to_string())?;
    for event in waiting {
        let answer = match event {
            Event::Line(line) if line.is_empty() => None,
            Event::Line(line) => server.handle(&line),
            // The answer `handle` gives a line that is not JSON.
            Event::Garbled => server.handle("\u{fffd}"),
            Event::Socket => {
                server.sink_mut().0.serve_socket();
                None
            }
            Event::Closed => {
                cutlass_logger::info!("mcp: the client closed stdin");
                break;
            }
            Event::Stopped => {
                cutlass_logger::info!("mcp: told to stop; closing the project");
                break;
            }
        };
        if let Some(answer) = answer {
            output
                .write_all(answer.as_bytes())
                .and_then(|()| output.write_all(b"\n"))
                .and_then(|()| output.flush())
                .map_err(|error| error.to_string())?;
        }
    }
    Ok(())
}

/// Send each line of `input` to the serving loop, then that it closed. A
/// line that is not text is told apart, so one bad byte does not end the
/// session.
fn read_lines(mut input: impl BufRead, events: &Sender<Event>) {
    let mut bytes = Vec::new();
    loop {
        bytes.clear();
        let event = match input.read_until(b'\n', &mut bytes) {
            Ok(0) | Err(_) => Event::Closed,
            Ok(_) => match std::str::from_utf8(&bytes) {
                Ok(line) => Event::Line(line.trim().to_owned()),
                Err(_) => Event::Garbled,
            },
        };
        let closed = matches!(event, Event::Closed);
        if events.send(event).is_err() || closed {
            return;
        }
    }
}

/// The registry on a project this host holds.
type OwnRegistry = Registry<
    Headless,
    Probe,
    Headless,
    Bundled,
    Bundled,
    Headless,
    Switched<LocalSandbox<ProcessExecutor>>,
    Refuse,
    McpServers,
    Headless,
>;

/// A project this host holds, and its tools.
struct Own {
    shell: Rc<RefCell<Shell>>,
    registry: OwnRegistry,
    /// The draft revision the folder was last saved at.
    saved: u64,
    /// The settings generation the servers, sandbox and skills were set up
    /// from.
    settings: u64,
    path: PathBuf,
}

/// Why this host did not take the project.
enum OpenError {
    /// Another app holds it.
    Held,
    Other(String),
}

impl Own {
    fn open(path: &Path, settings_path: &Path, wake: &Wake) -> Result<Self, OpenError> {
        let loaded = crate::load_settings_file(settings_path);
        let (decoders, encoders) = crate::register_codecs_with(&loaded.settings.hardware);
        let mut shell = Shell::new(decoders, encoders, loaded.settings);
        match shell.open_folder(path) {
            Ok(()) => {}
            Err(ShellError::Store(cutlass_store::StoreError::Held { .. })) => {
                return Err(OpenError::Held);
            }
            Err(other) => return Err(OpenError::Other(other.to_string())),
        }
        // Other hosts on this project send their calls here.
        shell.listen(Arc::clone(wake));
        // No window gives this host a GPU device, so its first export opens one.
        shell.draw_exports_offscreen(cutlass_render::Renderer::headless);
        let session = shell
            .session()
            .cloned()
            .ok_or_else(|| OpenError::Other("the project did not open".to_owned()))?;
        let saved = session.revision();
        let settings = SettingsTool::open(settings_path)
            .map_err(|error| OpenError::Other(error.to_string()))?;
        let generation = settings.generation();
        let mut servers = McpServers::default();
        servers.configure(&settings.settings().mcp_servers);
        let sandbox = Switched::from_settings(
            settings.settings(),
            crate::sandbox::local_sandbox(crate::sandbox::this_program(), Some(path)),
        );
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
                sandbox,
                Refuse,
                McpClient::new(servers),
                Headless(Rc::clone(&shell)),
            ),
        );
        Ok(Self {
            shell,
            registry,
            saved,
            settings: generation,
            path: path.to_path_buf(),
        })
    }

    /// Run `call`, then save what it changed and take up a settings change.
    fn run(&mut self, call: &ToolCall) -> ToolOutcome {
        let outcome = self.registry.run_tool(call);
        match self.settle() {
            Ok(()) => outcome,
            Err(error) => failed(error),
        }
    }

    /// Save the folder when the draft moved, and set the servers, sandbox
    /// and skills up again when the settings did, so a change made through
    /// `settings.set` holds from the next call on.
    fn settle(&mut self) -> Result<(), String> {
        let generation = self.registry.settings.generation();
        if generation != self.settings {
            self.settings = generation;
            let settings = self.registry.settings.settings().clone();
            self.registry
                .mcp
                .transport_mut()
                .configure(&settings.mcp_servers);
            self.registry.sandbox = Switched::from_settings(
                &settings,
                crate::sandbox::local_sandbox(crate::sandbox::this_program(), Some(&self.path)),
            );
            self.registry.skills = SkillLibrary::bundled().with_folders(&settings.skill_folders);
        }
        let session = self.registry.draft.session();
        if session.revision() != self.saved {
            let mut shell = self.shell.borrow_mut();
            shell
                .adopt_session(session.clone())
                .and_then(|()| shell.save())
                .map_err(|error| {
                    format!("the change was made but the project was not saved: {error}")
                })?;
            self.saved = session.revision();
        }
        Ok(())
    }
}

/// Where this host's calls run.
enum Place {
    /// On the project it holds.
    Own(Box<Own>),
    /// In the app that holds the project.
    Forward(Forward),
    /// Nowhere yet: the next call finds the project's holder.
    Unset,
}

/// The tools of one project, wherever it is held now.
struct Host {
    path: PathBuf,
    /// The user's settings file.
    settings: PathBuf,
    place: Place,
    /// When this host let the project go to a window that asked for it.
    let_go: Option<Instant>,
    wake: Wake,
}

/// How long, after letting the project go, this host waits for the window
/// to take it before it would take it back itself.
const HANDOVER_WAIT: Duration = Duration::from_secs(5);

impl Host {
    /// Find where the project is: held by another app, whose address is in
    /// its lock file, or free for this host to open.
    fn attach(&mut self) -> Result<(), String> {
        // Just let go: the window that asked is about to hold it.
        if let Some(since) = self.let_go.take() {
            while since.elapsed() < HANDOVER_WAIT {
                if self.forward().is_ok() {
                    return Ok(());
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        match Own::open(&self.path, &self.settings, &self.wake) {
            Ok(own) => {
                cutlass_logger::info!("mcp: serving {}", self.path.display());
                self.place = Place::Own(Box::new(own));
                Ok(())
            }
            Err(OpenError::Held) => self.forward(),
            Err(OpenError::Other(error)) => Err(error),
        }
    }

    /// Send the calls to the app that holds the project.
    fn forward(&mut self) -> Result<(), String> {
        let Some((address, token)) = cutlass_store::held_address(&self.path) else {
            return Err(format!(
                "{} is open in another Cutlass that does not take calls; close it there first",
                self.path.display()
            ));
        };
        let forward = Forward::connect(&address, &token)?;
        cutlass_logger::info!("mcp: forwarding {} to {address}", self.path.display());
        self.place = Place::Forward(forward);
        Ok(())
    }

    /// Run the calls other hosts sent to the project this host holds. A
    /// window's ask to let the project go is answered by saving it and
    /// letting go; this host's next call then goes to that window.
    fn serve_socket(&mut self) {
        let Place::Own(own) = &self.place else {
            return;
        };
        let mut waiting = own.shell.borrow_mut().take_forwarded().into_iter();
        while let Some(forwarded) = waiting.next() {
            let Place::Own(own) = &mut self.place else {
                return;
            };
            if forwarded.call.name != cutlass_agent::YIELD {
                let started = Instant::now();
                let outcome = own.run(&forwarded.call);
                crate::shell::log_tool_call("mcp", &forwarded.call, &outcome, started.elapsed());
                forwarded.answer(outcome);
                continue;
            }
            if let Err(error) = own.settle() {
                forwarded.answer(failed(error));
                continue;
            }
            // Dropping the project lets its lock go and closes its socket.
            self.place = Place::Unset;
            self.let_go = Some(Instant::now());
            cutlass_logger::info!(
                "mcp: let {} go to the Cutlass window that asked for it",
                self.path.display()
            );
            forwarded.answer(ToolOutcome {
                content: "let go".to_owned(),
                is_error: false,
                picture: None,
            });
            for late in waiting {
                late.answer(failed(
                    "this host let the project go to the Cutlass window; call again".to_owned(),
                ));
            }
            return;
        }
    }
}

impl ToolSink for Host {
    fn run_tool(&mut self, call: &ToolCall) -> ToolOutcome {
        if matches!(self.place, Place::Unset)
            && let Err(error) = self.attach()
        {
            return failed(error);
        }
        match &mut self.place {
            Place::Own(own) => own.run(call),
            Place::Forward(forward) => match forward.try_call(call) {
                Ok(outcome) => outcome,
                Err(error) => {
                    // The holder is gone. The next call finds the project
                    // where it is now; this one is not sent twice, as the
                    // holder may have run it before it went.
                    cutlass_logger::warn!("mcp: the project's holder stopped answering: {error}");
                    self.place = Place::Unset;
                    failed(format!(
                        "the app that held the project stopped answering ({error}), so this \
                         call did not come back. Call again: the project is found where it is \
                         now. Read it first, as its revision starts again."
                    ))
                }
            },
            Place::Unset => failed("the project is not open".to_owned()),
        }
    }
}

fn failed(content: String) -> ToolOutcome {
    ToolOutcome {
        content,
        is_error: true,
        picture: None,
    }
}

/// A tool sink that logs each call it runs, by name and outcome.
struct Logged<S>(S);

impl<S: ToolSink> ToolSink for Logged<S> {
    fn run_tool(&mut self, call: &ToolCall) -> ToolOutcome {
        let started = Instant::now();
        let mut outcome = self.0.run_tool(call);
        // A job still running is asked again for a few seconds before the
        // client hears "pending", so it does not poll in a tight loop.
        while crate::assistant::still_pending(call, &outcome) && started.elapsed() < PENDING_WAIT {
            std::thread::sleep(Duration::from_millis(200));
            outcome = self.0.run_tool(call);
        }
        crate::shell::log_tool_call("mcp", call, &outcome, started.elapsed());
        outcome
    }
}

/// How long a poll of a running job waits before it answers pending.
const PENDING_WAIT: Duration = Duration::from_secs(5);

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
            std::thread::sleep(Duration::from_millis(10));
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
    /// file is not replaced, as no one is here to allow it. Effect clips
    /// the file was written without, for want of a GPU, are named.
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
            std::thread::sleep(Duration::from_millis(10));
        }
        if codec != cutlass_codec::CodecId::Png && shell.exported() != Some(path.as_path()) {
            return Err(shell.notice().to_owned());
        }
        if !path.is_file() {
            return Err(format!("{} was not written", path.display()));
        }
        let id = shell.agent_export_next();
        let wrote = shell.export_wrote(&path);
        shell.agent_export_began(id.clone(), path, Some(Ok(wrote)));
        Ok(id)
    }

    fn poll(&mut self, job_id: &str) -> Result<Option<String>, String> {
        match self.0.borrow_mut().agent_export_now() {
            Some((id, state)) if id == job_id => state,
            _ => Err(format!("no export {job_id}; export.start names its job")),
        }
    }
}

/// Reads a file's header for its length, size and streams.
pub(crate) struct Probe;

impl Media for Probe {
    fn probe(&mut self, path: &str) -> Result<MediaInfo, String> {
        if path.starts_with('~') {
            return Err(format!(
                "{path}: give the file's full path, from the root, not one starting with ~"
            ));
        }
        match cutlass_probe::probe(Path::new(path)).map_err(|error| error.to_string())? {
            cutlass_probe::Probed::Video {
                meta,
                width,
                height,
                duration,
                has_audio,
            } => {
                let mut streams = vec!["video".to_owned()];
                if has_audio {
                    streams.push("audio".to_owned());
                }
                Ok(MediaInfo {
                    kind: meta.as_str().to_owned(),
                    duration_us: duration,
                    width,
                    height,
                    streams,
                })
            }
            cutlass_probe::Probed::Audio { duration } => Ok(MediaInfo {
                kind: cutlass_model::MetaType::Music.as_str().to_owned(),
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
        serve(
            Some(&project.0),
            std::io::Cursor::new(input.into_bytes()),
            &mut output,
        )
        .unwrap();
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
    fn the_probe_names_the_import_type_and_a_movies_sound() {
        use cutlass_agent::Media;
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../crates");
        let movie = root.join("containers/mpegts/tests/fixtures/h264_aac.ts");
        let info = super::Probe.probe(movie.to_str().unwrap()).unwrap();
        assert_eq!(info.kind, "video");
        assert_eq!(info.streams, ["video", "audio"]);
        let still = root.join("engine/fixtures/pattern.png");
        let info = super::Probe.probe(still.to_str().unwrap()).unwrap();
        assert_eq!(info.kind, "image");
        assert_eq!(info.streams, ["video"]);
        let photo = root.join("engine/fixtures/pattern.jpg");
        assert_eq!(
            super::Probe.probe(photo.to_str().unwrap()).unwrap().kind,
            "photo"
        );
        assert!(super::Probe.probe("/no/such/file.mp4").is_err());
        let home = super::Probe.probe("~/Movies/a.mp4").unwrap_err();
        assert!(home.contains("full path"), "{home}");
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
            call(9, "export_poll", json!({"input": "export-2"})),
            call(
                10,
                "export_start",
                json!({"input": format!("new {}", project.0.join("frame.jpg").display())}),
            ),
        ]);
        let mut output = Vec::new();
        serve(
            Some(&project.0),
            std::io::Cursor::new(input.into_bytes()),
            &mut output,
        )
        .unwrap();
        let answers = answers(output);
        let text = |at: usize| {
            answers[at]["result"]["content"][0]["text"]
                .as_str()
                .unwrap()
                .to_owned()
        };
        assert_eq!(answers.len(), 10);
        // The export answered when it was written, and a poll says so again.
        assert!(
            text(2).starts_with("job=export-1 finished: wrote "),
            "{}",
            text(2)
        );
        assert!(text(8).starts_with("finished: wrote "), "{}", text(8));
        assert!(text(8).contains("cut.mp4"), "{}", text(8));
        // A .jpg would be a movie under a photo's name: refused.
        assert_eq!(answers[9]["result"]["isError"], true, "{}", text(9));
        assert!(!project.0.join("frame.jpg").exists());
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

    /// A grey frame with a vignette clip over it, exported by a host with
    /// no window: with a GPU the corners of the still and of the movie are
    /// darker than the middle, and without one the answers name the clip.
    #[test]
    fn with_no_window_an_export_shows_an_effect_clip_or_names_it() {
        let gpu = cutlass_render::Renderer::for_test()
            .expect("a GPU is required")
            .is_some();
        let project = Project::new("effect");
        cutlass_store::Folder::new(crate::shell::vignette_draft())
            .save(&project.0)
            .unwrap();
        let still = project.0.join("still.png");
        let movie = project.0.join("cut.mp4");
        let call = |id: i64, path: &std::path::Path| {
            json!({"jsonrpc":"2.0","id":id,"method":"tools/call",
                "params":{"name":"export_start",
                    "arguments":{"input": format!("new {}", path.display())}}})
        };
        let input = lines(&[
            json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
            call(2, &still),
            call(3, &movie),
        ]);
        let mut output = Vec::new();
        serve(
            Some(&project.0),
            std::io::Cursor::new(input.into_bytes()),
            &mut output,
        )
        .unwrap();
        let answers = answers(output);
        let text = |at: usize| {
            answers[at]["result"]["content"][0]["text"]
                .as_str()
                .unwrap()
                .to_owned()
        };
        assert_eq!(answers.len(), 3);
        assert!(still.is_file(), "{}", text(1));
        assert!(movie.is_file(), "{}", text(2));
        if !gpu {
            eprintln!("no GPU: the answers name the clip");
            for (at, path) in [(1, &still), (2, &movie)] {
                let named = format!(
                    "wrote {}, but 1 effect clip is not in the picture (fx): no GPU device",
                    path.display()
                );
                assert!(text(at).contains(&named), "{}", text(at));
            }
            return;
        }
        assert_eq!(
            text(1),
            format!("job=export-1 finished: wrote {}", still.display())
        );
        assert_eq!(
            text(2),
            format!("job=export-2 finished: wrote {}", movie.display())
        );
        let (middle, corner) = crate::shell::still_middle_and_corner(&still);
        assert!(corner + 40 < middle, "corner {corner}, middle {middle}");
        let (middle, corner) = crate::shell::movie_middle_and_corner(&movie);
        assert!(corner + 40 < middle, "corner {corner}, middle {middle}");
    }

    #[test]
    fn with_no_gpu_the_export_answer_names_the_effect_clips_left_out() {
        use cutlass_agent::Export;
        let project = Project::new("no-gpu");
        cutlass_store::Folder::new(crate::shell::vignette_draft())
            .save(&project.0)
            .unwrap();
        let (decoders, encoders) = crate::register_codecs();
        let mut shell =
            crate::shell::Shell::new(decoders, encoders, cutlass_settings::Settings::default());
        shell.open_folder(&project.0).unwrap();
        shell.draw_exports_offscreen(crate::shell::absent);
        let mut host = super::Headless(std::rc::Rc::new(std::cell::RefCell::new(shell)));
        for name in ["still.png", "cut.mp4"] {
            let path = project.0.join(name);
            let id = Export::start(&mut host, path.to_str().unwrap()).unwrap();
            assert!(path.is_file());
            // The answer when it was written, and each poll after, say so.
            assert_eq!(
                Export::poll(&mut host, &id).unwrap().unwrap(),
                format!(
                    "wrote {}, but 1 effect clip is not in the picture (fx): no GPU device: \
                     none here. Effects are drawn on a GPU, and none could be opened on this \
                     machine.",
                    path.display()
                )
            );
        }
        // Past the effect clip the still is whole, and the answer is plain.
        host.0.borrow_mut().seek(cutlass_model::Micros(150_000));
        let late = project.0.join("late.png");
        let id = Export::start(&mut host, late.to_str().unwrap()).unwrap();
        assert_eq!(
            Export::poll(&mut host, &id).unwrap().unwrap(),
            format!("wrote {}", late.display())
        );
    }

    #[test]
    fn a_held_project_or_no_path_is_refused() {
        let project = Project::new("held");
        let _hold = cutlass_store::hold(&project.0).unwrap();
        let error = serve(Some(&project.0), std::io::empty(), Vec::new()).unwrap_err();
        assert!(error.contains("does not take calls"), "{error}");
        let error = serve(None, std::io::empty(), Vec::new()).unwrap_err();
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
            serve(
                Some(&path),
                std::io::Cursor::new(input.into_bytes()),
                &mut output,
            )
            .map(|()| output)
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

    /// A client this test writes to line by line: `cutlass mcp` reads it as
    /// its stdin until the sender is dropped.
    struct Typed(std::sync::mpsc::Receiver<Vec<u8>>, Vec<u8>);

    impl std::io::Read for Typed {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            if self.1.is_empty() {
                match self.0.recv() {
                    Ok(bytes) => self.1 = bytes,
                    Err(_) => return Ok(0),
                }
            }
            let count = out.len().min(self.1.len());
            out[..count].copy_from_slice(&self.1[..count]);
            self.1.drain(..count);
            Ok(count)
        }
    }

    /// What a client wrote to: the answers so far, read from another thread.
    #[derive(Clone, Default)]
    struct Heard(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for Heard {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl Heard {
        /// The answer to the message with `id`, waiting for it; `tick` runs
        /// between looks, for a window that must serve the call meanwhile.
        fn answer(&self, id: i64, mut tick: impl FnMut()) -> Value {
            let until = std::time::Instant::now() + std::time::Duration::from_secs(30);
            loop {
                let text = String::from_utf8(self.0.lock().unwrap().clone()).unwrap();
                let found = text
                    .lines()
                    .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                    .find(|answer| answer["id"] == id);
                if let Some(found) = found {
                    return found;
                }
                assert!(
                    std::time::Instant::now() < until,
                    "no answer to {id}: {text}"
                );
                tick();
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }
    }

    /// A `cutlass mcp` on `path` that the test types to; dropping the
    /// sender closes its stdin.
    fn typed_host(
        path: &std::path::Path,
    ) -> (
        std::sync::mpsc::Sender<Vec<u8>>,
        Heard,
        std::thread::JoinHandle<Result<(), String>>,
    ) {
        typed_host_with(path, &crate::shell::Shell::settings_path())
    }

    /// [`typed_host`], with its settings kept in `settings`.
    fn typed_host_with(
        path: &std::path::Path,
        settings: &std::path::Path,
    ) -> (
        std::sync::mpsc::Sender<Vec<u8>>,
        Heard,
        std::thread::JoinHandle<Result<(), String>>,
    ) {
        let (lines, typed) = std::sync::mpsc::channel();
        let heard = Heard::default();
        let (path, settings, output) = (path.to_path_buf(), settings.to_path_buf(), heard.clone());
        let host = std::thread::spawn(move || {
            super::serve_with(
                Some(&path),
                &settings,
                std::io::BufReader::new(Typed(typed, Vec::new())),
                output,
                |_| {},
            )
        });
        (lines, heard, host)
    }

    fn call_line(id: i64, name: &str, arguments: Value) -> Vec<u8> {
        format!(
            "{}\n",
            json!({"jsonrpc":"2.0","id":id,"method":"tools/call",
                "params":{"name":name,"arguments":arguments}})
        )
        .into_bytes()
    }

    fn said(answer: &Value) -> String {
        answer["result"]["content"][0]["text"]
            .as_str()
            .unwrap_or_default()
            .to_owned()
    }

    fn rename(revision: u64, name: &str) -> Value {
        json!({"expected_revision": revision,
            "requests": [{"op": "set_project_name", "name": name}]})
    }

    #[test]
    fn told_to_stop_with_stdin_still_open_the_host_closes_the_project() {
        let project = Project::new("stopped");
        let (lines, typed) = std::sync::mpsc::channel();
        let (stops, stop) = std::sync::mpsc::channel();
        let heard = Heard::default();
        let (path, output) = (project.0.clone(), heard.clone());
        let host = std::thread::spawn(move || {
            super::serve_with(
                Some(&path),
                &crate::shell::Shell::settings_path(),
                std::io::BufReader::new(Typed(typed, Vec::new())),
                output,
                move |stop| stops.send(stop).unwrap(),
            )
        });
        lines
            .send(call_line(1, "draft_apply", rename(0, "kept")))
            .unwrap();
        assert_eq!(said(&heard.answer(1, || {})), "revision=1 committed=true");
        // What a signal does. The client still holds stdin open: `lines`
        // is alive until the test ends.
        stop.recv().unwrap().stop();
        let until = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while !host.is_finished() {
            assert!(
                std::time::Instant::now() < until,
                "the host kept serving after it was told to stop"
            );
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        host.join().unwrap().unwrap();
        let hold = cutlass_store::hold(&project.0).expect("the lock was let go");
        assert!(!hold.unclean(), "the project was closed, not left");
        drop(hold);
        let opened = cutlass_store::open(&project.0).unwrap();
        assert_eq!(opened.folder.draft.name, "kept");
        drop(lines);
    }

    #[test]
    fn a_second_client_attaches_to_the_host_that_holds_the_project() {
        let project = Project::new("second");
        let (first, heard, host) = typed_host(&project.0);
        first
            .send(call_line(1, "draft_query", json!({"input": "project"})))
            .unwrap();
        assert!(said(&heard.answer(1, || {})).contains("revision=0"));
        // The second finds the project held, and its edit lands on the
        // first's session: one writer, one undo stack.
        let input = lines(&[
            json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
            json!({"jsonrpc":"2.0","id":2,"method":"tools/call",
                "params":{"name":"draft_apply","arguments":rename(0, "from the second")}}),
        ]);
        let mut output = Vec::new();
        serve(
            Some(&project.0),
            std::io::Cursor::new(input.into_bytes()),
            &mut output,
        )
        .unwrap();
        let second = answers(output);
        assert_eq!(said(&second[1]), "revision=1 committed=true");
        first
            .send(call_line(2, "draft_query", json!({"input": "project"})))
            .unwrap();
        let read = said(&heard.answer(2, || {}));
        assert!(read.contains("revision=1 name=from the second"), "{read}");
        first.send(call_line(3, "draft_undo", json!({}))).unwrap();
        assert_eq!(heard.answer(3, || {})["result"]["isError"], false);
        drop(first);
        host.join().unwrap().unwrap();
        assert!(
            cutlass_store::hold(&project.0).is_ok(),
            "the lock was let go"
        );
    }

    #[test]
    fn the_window_takes_the_project_from_a_host_with_no_window_which_then_forwards() {
        let project = Project::new("handover");
        let (client, heard, host) = typed_host(&project.0);
        client
            .send(call_line(
                1,
                "draft_apply",
                rename(0, "edited with no window"),
            ))
            .unwrap();
        assert_eq!(said(&heard.answer(1, || {})), "revision=1 committed=true");
        // The window opens the same project: the host saves, lets go, and
        // the window reads what it saved.
        let (decoders, encoders) = crate::register_codecs();
        let mut desktop =
            crate::shell::Shell::new(decoders, encoders, cutlass_settings::Settings::default());
        desktop.listen(std::sync::Arc::new(|| {}));
        desktop.open_folder(&project.0).unwrap();
        assert_eq!(desktop.draft().unwrap().name, "edited with no window");
        // The client's next calls run in the window, on its undo stack.
        client
            .send(call_line(2, "draft_query", json!({"input": "project"})))
            .unwrap();
        let read = said(&heard.answer(2, || {
            desktop.serve_forwarded();
        }));
        assert!(
            read.contains("revision=0 name=edited with no window"),
            "{read}"
        );
        client
            .send(call_line(
                3,
                "draft_apply",
                rename(0, "edited in the window"),
            ))
            .unwrap();
        heard.answer(3, || {
            desktop.serve_forwarded();
        });
        assert_eq!(desktop.draft().unwrap().name, "edited in the window");
        desktop.undo().unwrap();
        assert_eq!(desktop.draft().unwrap().name, "edited with no window");
        // A second window does not take it from the first.
        let asked = std::thread::scope(|scope| {
            let asking = scope.spawn(|| {
                let (decoders, encoders) = crate::register_codecs();
                let mut other = crate::shell::Shell::new(
                    decoders,
                    encoders,
                    cutlass_settings::Settings::default(),
                );
                other.listen(std::sync::Arc::new(|| {}));
                other.open_folder(&project.0).is_err()
            });
            while !asking.is_finished() {
                desktop.serve_forwarded();
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            asking.join().unwrap()
        });
        assert!(asked, "a window kept its project");
        drop(client);
        host.join().unwrap().unwrap();
    }

    #[test]
    fn a_forwarder_whose_window_closed_finds_the_project_again() {
        let project = Project::new("gone");
        let (decoders, encoders) = crate::register_codecs();
        let mut desktop =
            crate::shell::Shell::new(decoders, encoders, cutlass_settings::Settings::default());
        desktop.open_folder(&project.0).unwrap();
        desktop.listen(std::sync::Arc::new(|| {}));
        let (client, heard, host) = typed_host(&project.0);
        client
            .send(call_line(1, "draft_apply", rename(0, "in the window")))
            .unwrap();
        heard.answer(1, || {
            desktop.serve_forwarded();
        });
        desktop.save().unwrap();
        // The window closes. The call in flight says so; the next one opens
        // the project here and reads what the window saved.
        drop(desktop);
        client
            .send(call_line(2, "draft_query", json!({"input": "project"})))
            .unwrap();
        let lost = heard.answer(2, || {});
        assert_eq!(lost["result"]["isError"], true);
        assert!(said(&lost).contains("Call again"), "{}", said(&lost));
        client
            .send(call_line(3, "draft_query", json!({"input": "project"})))
            .unwrap();
        let read = said(&heard.answer(3, || {}));
        assert!(read.contains("revision=0 name=in the window"), "{read}");
        drop(client);
        host.join().unwrap().unwrap();
        assert!(cutlass_store::hold(&project.0).is_ok());
    }

    #[test]
    fn a_settings_change_holds_from_the_next_call_and_bad_bytes_do_not_end_the_session() {
        let project = Project::new("settle");
        // The settings are a file of this test's own, not the user's.
        let settings = project.0.join("settings-home").join("settings.json");
        std::fs::create_dir_all(settings.parent().unwrap()).unwrap();
        let (client, heard, host) = typed_host_with(&project.0, &settings);
        client.send(b"\xff\xfe not text\n".to_vec()).unwrap();
        client
            .send(call_line(1, "settings_get", json!({})))
            .unwrap();
        let settings: Value = serde_json::from_str(&said(&heard.answer(1, || {}))).unwrap();
        let generation = settings["generation"].as_u64().unwrap();
        assert_eq!(settings["settings"]["sandbox"]["local"], true);
        // The bad line got a parse error, and the session went on.
        let text = String::from_utf8(heard.0.lock().unwrap().clone()).unwrap();
        assert!(text.contains("-32700"), "{text}");
        let set = |on: bool, generation: u64| json!({"expected_generation": generation, "changes": {"sandbox": {"local": on}}});
        client
            .send(call_line(2, "settings_set", set(false, generation)))
            .unwrap();
        assert_eq!(heard.answer(2, || {})["result"]["isError"], false);
        client
            .send(call_line(3, "sandbox_run", json!({"source": "1 + 1"})))
            .unwrap();
        let off = heard.answer(3, || {});
        assert_eq!(off["result"]["isError"], true, "{}", said(&off));
        assert!(said(&off).contains("running code is off"), "{}", said(&off));
        client
            .send(call_line(4, "settings_set", set(true, generation + 1)))
            .unwrap();
        assert_eq!(heard.answer(4, || {})["result"]["isError"], false);
        client
            .send(call_line(5, "sandbox_run", json!({"source": "1 + 1"})))
            .unwrap();
        // On again, the call reaches the runner (which, under test, is this
        // test program and not `cutlass`, so it has no answer).
        let on = said(&heard.answer(5, || {}));
        assert!(!on.contains("running code is off"), "{on}");
        drop(client);
        host.join().unwrap().unwrap();
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
        serve(
            Some(&project.0),
            std::io::Cursor::new(input.into_bytes()),
            &mut output,
        )
        .unwrap();
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
