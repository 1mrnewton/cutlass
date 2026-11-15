//! The assistant's tools on the open project. Each call runs on the
//! window's thread against the session the window holds, so an edit the
//! assistant makes is one undo step like any other, and the player moves
//! the window's playhead.

use cutlass_agent::{
    ApprovalAction, Approver, DraftTools, Export, McpClient, Player, Registry, SettingsTool,
    Speech, ToolCall, ToolOutcome, ToolSink, TrackAsk, Tracking,
};
use cutlass_codec::CodecId;
use cutlass_model::Micros;

use super::{ExportRequest, Shell};
use crate::mcp::{Bundled, Probe};

/// Player moves, made on the window once the call is done.
#[derive(Default)]
struct Moves {
    seek: Option<i64>,
    play: Option<bool>,
}

impl Player for Moves {
    fn seek(&mut self, time_us: i64) -> Result<(), String> {
        if time_us < 0 {
            return Err("the time is before the start".to_owned());
        }
        self.seek = Some(time_us);
        Ok(())
    }

    fn play(&mut self) -> Result<(), String> {
        self.play = Some(true);
        Ok(())
    }

    fn pause(&mut self) -> Result<(), String> {
        self.play = Some(false);
        Ok(())
    }
}

/// Allows what the user already allowed for this call; anything else is
/// kept, so the window can ask, and refused for now.
struct Asked<'a> {
    allowed: &'a [ApprovalAction],
    wanted: Option<ApprovalAction>,
}

impl Approver for Asked<'_> {
    fn decide(&mut self, action: &ApprovalAction) -> bool {
        if self.allowed.contains(action) {
            return true;
        }
        self.wanted.get_or_insert_with(|| action.clone());
        false
    }
}

/// Transcription as a window job; an earlier call's job is polled here.
struct Hearing<'a> {
    shell: &'a mut Shell,
}

impl Speech for Hearing<'_> {
    fn start(&mut self, start_us: i64, duration_us: i64) -> Result<String, String> {
        self.shell
            .start_hearing(start_us, duration_us)
            .map_err(|error| error.to_string())
    }

    fn poll(&mut self, job_id: &str) -> Result<Option<String>, String> {
        self.shell
            .hearing(job_id)
            .map_err(|error| error.to_string())
    }
}

/// Tracking: a job the window starts once the call is done, and the
/// paths of jobs as they stood when the call began.
struct Tracks {
    now: super::follow::AgentTracks,
    started: Option<(String, TrackAsk)>,
}

impl Tracking for Tracks {
    fn start(&mut self, ask: &TrackAsk) -> Result<String, String> {
        if self.started.is_some() {
            return Err("one tracking job at a time".to_owned());
        }
        let id = self.now.next_id();
        self.started = Some((id.clone(), ask.clone()));
        Ok(id)
    }

    fn poll(&mut self, job_id: &str) -> Result<Option<cutlass_model::TrackPath>, String> {
        self.now.path(job_id)
    }

    fn clip_path(
        &self,
        draft: &cutlass_model::Draft,
        clip: &cutlass_model::SegmentId,
        path: &cutlass_model::TrackPath,
    ) -> Option<cutlass_model::TrackPath> {
        super::clip_path(draft, clip, path)
    }
}

/// An export the assistant asked for: its job id, the file, and what
/// became of it once that is known.
pub(super) struct AgentExport {
    id: String,
    path: std::path::PathBuf,
    /// `None` while it is still writing.
    done: Option<Result<String, String>>,
}

/// An export the window starts once the call is done, and the last one's
/// state as it stood when the call began.
struct Exports {
    /// The id the next export gets.
    next: String,
    path: Option<String>,
    /// The last export's id and what a poll of it answers.
    now: Option<(String, Result<Option<String>, String>)>,
}

impl Export for Exports {
    fn start(&mut self, path: &str) -> Result<String, String> {
        if path.trim().is_empty() {
            return Err("name the file to export to".to_owned());
        }
        self.path = Some(path.to_owned());
        Ok(self.next.clone())
    }

    fn poll(&mut self, job_id: &str) -> Result<Option<String>, String> {
        match &self.now {
            Some((id, state)) if id == job_id => state.clone(),
            // The one this call asked for starts when the call is done.
            _ if self.path.is_some() && job_id == self.next => Ok(None),
            _ => Err(format!("no export {job_id}; export.start names its job")),
        }
    }
}

/// What one assistant tool call did, and the approval it waits for when
/// it needs one the user has not given.
pub struct AgentRun {
    pub outcome: ToolOutcome,
    pub wants: Option<ApprovalAction>,
}

/// Error text kept in the log for a failed tool call. The rest may be a
/// script's output.
const TOOL_ERROR_LOGGED: usize = 300;

/// Log one finished tool call from `source`: its name and id, how long it
/// took, and the start of its error. The arguments stay out, as
/// `settings.set` can carry a key and `draft.apply` the user's words.
pub(crate) fn log_tool_call(
    source: &str,
    call: &ToolCall,
    outcome: &ToolOutcome,
    took: std::time::Duration,
) {
    if outcome.is_error {
        let mut error = outcome.content.as_str();
        if error.len() > TOOL_ERROR_LOGGED {
            let mut end = TOOL_ERROR_LOGGED;
            while !error.is_char_boundary(end) {
                end -= 1;
            }
            error = &error[..end];
        }
        cutlass_logger::warn!(
            "{source} tool {}{} failed after {took:.1?}: {error}",
            call.name,
            call_id(call)
        );
    } else {
        cutlass_logger::info!(
            "{source} tool {}{} ran in {took:.1?}",
            call.name,
            call_id(call)
        );
    }
}

/// ` (id)`, or nothing for a call that came without one.
fn call_id(call: &ToolCall) -> String {
    if call.id.is_empty() {
        String::new()
    } else {
        format!(" ({})", call.id)
    }
}

impl Shell {
    /// The id the assistant's next export gets.
    pub(crate) fn agent_export_next(&self) -> String {
        format!("export-{}", self.agent_exports.saturating_add(1))
    }

    /// Note that the assistant's export to `path` began, as job `id`.
    /// `done` is its result when it was written before this returned.
    pub(crate) fn agent_export_began(
        &mut self,
        id: String,
        path: std::path::PathBuf,
        done: Option<Result<String, String>>,
    ) {
        self.agent_exports = self.agent_exports.saturating_add(1);
        self.agent_export = Some(AgentExport { id, path, done });
    }

    /// The assistant's last export and what a poll of it answers: `None`
    /// while it writes, the file once written, or why it failed.
    pub(crate) fn agent_export_now(&mut self) -> Option<(String, Result<Option<String>, String>)> {
        self.poll_job();
        let writing = self.job().is_some_and(|job| job.label == "Export");
        let written = self.exported.clone();
        let notice = self.notice.clone();
        let export = self.agent_export.as_mut()?;
        if export.done.is_none() && !writing {
            export.done = Some(if written.as_deref() == Some(export.path.as_path()) {
                Ok(format!("wrote {}", export.path.display()))
            } else if notice.is_empty() {
                Err("the export stopped before it finished".to_owned())
            } else {
                Err(notice)
            });
        }
        let state = match &export.done {
            None => Ok(None),
            Some(Ok(wrote)) => Ok(Some(wrote.clone())),
            Some(Err(why)) => Err(why.clone()),
        };
        Some((export.id.clone(), state))
    }

    /// Run one tool call from the assistant on the open project. `allowed`
    /// is what the user allowed for this call; a call that needs anything
    /// else does nothing and says what it wants.
    pub fn run_agent_tool(&mut self, call: &ToolCall, allowed: &[ApprovalAction]) -> AgentRun {
        let started = std::time::Instant::now();
        let run = self.run_agent_tool_in(call, allowed);
        match &run.wants {
            Some(action) => cutlass_logger::info!(
                "assistant tool {}{} waits for the user's allow: {action:?}",
                call.name,
                call_id(call)
            ),
            None => log_tool_call("assistant", call, &run.outcome, started.elapsed()),
        }
        run
    }

    fn run_agent_tool_in(&mut self, call: &ToolCall, allowed: &[ApprovalAction]) -> AgentRun {
        let Some(session) = self.session.clone() else {
            return AgentRun {
                outcome: ToolOutcome {
                    content: "no project is open".to_owned(),
                    is_error: true,
                },
                wants: None,
            };
        };
        let before = session.revision();
        let (path, generation) = self
            .settings_file
            .clone()
            .unwrap_or_else(|| (Shell::settings_path(), 0));
        let settings = SettingsTool::held(path.clone(), generation, self.settings.clone());
        let jobs = std::mem::take(&mut self.agent_speech_jobs);
        let track_jobs = std::mem::take(&mut self.agent_track_jobs);
        let tracks = Tracks {
            now: self.agent_tracks_now(),
            started: None,
        };
        let exports = Exports {
            next: self.agent_export_next(),
            path: None,
            now: self.agent_export_now(),
        };
        let mut servers = std::mem::take(&mut self.mcp_servers);
        servers.configure(&self.settings.mcp_servers);
        let sandbox = cutlass_agent::Switched::from_settings(
            &self.settings,
            crate::sandbox::local_sandbox(crate::sandbox::this_program(), self.project_folder()),
        );
        let mut draft = DraftTools::new(session);
        draft.set_remembered(std::mem::take(&mut self.agent_request_ids));
        let mut registry = Registry::new(
            draft,
            settings,
            Moves::default(),
            Probe,
            Hearing { shell: self },
            Bundled,
            (
                Bundled,
                exports,
                sandbox,
                Asked {
                    allowed,
                    wanted: None,
                },
                McpClient::new(servers),
                tracks,
            ),
        );
        registry.set_speech_jobs(jobs);
        registry.set_track_jobs(track_jobs);
        let mut outcome = registry.run_tool(call);
        let wants = registry.approver.wanted.take();
        let settings_now = registry.settings.get();
        let servers = std::mem::take(registry.mcp.transport_mut());
        let session = registry.draft.session().clone();
        let request_ids = registry.draft.take_remembered();
        let moves = std::mem::take(&mut registry.player);
        let exported = registry.export.path.take();
        let jobs = registry.take_speech_jobs();
        let mut track_jobs = registry.take_track_jobs();
        let tracks = std::mem::replace(
            &mut registry.tracking,
            Tracks {
                now: super::follow::AgentTracks::default(),
                started: None,
            },
        );
        drop(registry);
        self.mcp_servers = servers;
        self.agent_request_ids = request_ids;
        if settings_now.generation != generation {
            // The settings tool already wrote the file at this generation.
            cutlass_logger::info!(
                "settings changed by the assistant: {}",
                super::changed_settings(&self.settings, &settings_now.settings).join(", ")
            );
            self.settings = settings_now.settings;
            self.settings_file = Some((path, settings_now.generation));
        }
        self.agent_speech_jobs = jobs;
        self.keep_track_ids(&tracks.now);
        if let Some((id, ask)) = tracks.started
            && let Err(error) = self.start_agent_track(id.clone(), &ask)
        {
            track_jobs.remove(&id);
            outcome = ToolOutcome {
                content: error.to_string(),
                is_error: true,
            };
        }
        self.agent_track_jobs = track_jobs;
        if session.revision() != before {
            self.session = Some(session);
            self.lanes_cache.replace(None);
            self.bars_cache.replace(None);
        }
        if let Some(time) = moves.seek {
            self.seek(Micros(time));
        }
        match moves.play {
            Some(true) => self.play(),
            Some(false) => self.pause(),
            None => {}
        }
        if let Some(path) = exported {
            let replace = call.arguments.trim_start().starts_with("replace");
            let still = path.to_ascii_lowercase().ends_with(".png");
            let codec = if still { CodecId::Png } else { CodecId::H264 };
            let id = self.agent_export_next();
            let file = std::path::PathBuf::from(&path);
            let started = self.export(ExportRequest {
                path: file.clone(),
                codec,
                replace,
                ..ExportRequest::default()
            });
            match started {
                Err(error) => {
                    outcome = ToolOutcome {
                        content: error.to_string(),
                        is_error: true,
                    };
                }
                // A still is written by the time `export` returns.
                Ok(()) if still => {
                    let wrote = format!("wrote {path}");
                    outcome.content = format!("job={id} finished: {wrote}");
                    self.agent_export_began(id, file, Some(Ok(wrote)));
                }
                // A movie is a job: `export.poll` says when it lands.
                Ok(()) => self.agent_export_began(id, file, None),
            }
        }
        AgentRun { outcome, wants }
    }
}

#[cfg(test)]
mod tests {
    use cutlass_agent::ToolCall;
    use cutlass_model::{Draft, Micros, Segment, TimeRange, TrackId};
    use cutlass_store::Folder;
    use serde_json::json;

    use crate::shell::tests::{Temp, shell};

    fn call(name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: "1".to_owned(),
            name: name.to_owned(),
            arguments: arguments.to_owned(),
        }
    }

    #[test]
    fn a_tool_call_is_logged_by_name_and_outcome_without_its_arguments() {
        crate::test_log::path();
        let ok = ToolCall {
            id: "log-call-ok".to_owned(),
            name: "draft.query".to_owned(),
            arguments: "args-stay-out".to_owned(),
        };
        let done = cutlass_agent::ToolOutcome {
            content: "the project".to_owned(),
            is_error: false,
        };
        super::log_tool_call("assistant", &ok, &done, std::time::Duration::from_millis(3));
        let failed = ToolCall {
            id: "log-call-failed".to_owned(),
            ..ok.clone()
        };
        let long = cutlass_agent::ToolOutcome {
            content: format!("bad-start{}", "é".repeat(400)),
            is_error: true,
        };
        super::log_tool_call("mcp", &failed, &long, std::time::Duration::from_millis(3));
        let log = crate::test_log::text();
        let line = |id: &str| {
            log.lines()
                .find(|line| line.contains(id))
                .unwrap_or_default()
                .to_owned()
        };
        let ran = line("log-call-ok");
        assert!(ran.contains(" INFO "), "{log}");
        assert!(
            ran.contains("assistant tool draft.query (log-call-ok) ran in"),
            "{log}"
        );
        let refused = line("log-call-failed");
        assert!(refused.contains(" WARN "), "{log}");
        assert!(
            refused.contains("mcp tool draft.query (log-call-failed) failed"),
            "{log}"
        );
        assert!(refused.contains("bad-start"), "{log}");
        assert!(refused.len() < 600, "the error is cut: {refused}");
        assert!(!log.contains("args-stay-out"), "{log}");
        let mut desktop = shell();
        let none = desktop.run_agent_tool(
            &ToolCall {
                id: "log-call-no-project".to_owned(),
                ..ok
            },
            &[],
        );
        assert!(none.outcome.is_error);
        let log = crate::test_log::text();
        assert!(
            log.contains("assistant tool draft.query (log-call-no-project) failed"),
            "{log}"
        );
    }

    #[test]
    fn the_assistant_reads_edits_and_moves_the_playhead_in_the_window() {
        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.tracks[0].id = TrackId::new("main");
        draft.tracks[0]
            .segments
            .push(Segment::new("a", "missing", TimeRange::new(0, 2_000_000)));
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        let none = desktop
            .run_agent_tool(&call("draft.query", "project"), &[])
            .outcome;
        assert!(none.is_error);
        desktop.open_folder(&temp.path).unwrap();

        let read = desktop
            .run_agent_tool(&call("draft.query", "project"), &[])
            .outcome;
        assert!(!read.is_error, "{}", read.content);
        let revision: u64 = read
            .content
            .split_whitespace()
            .find_map(|word| word.strip_prefix("revision="))
            .and_then(|value| value.parse().ok())
            .unwrap();

        let split = json!({
            "expected_revision": revision,
            "requests": [{
                "op": "split",
                "segment_id": "a",
                "at": 500_000,
                "right_id": "b",
            }],
        });
        let applied = desktop
            .run_agent_tool(&call("draft.apply", &split.to_string()), &[])
            .outcome;
        assert!(!applied.is_error, "{}", applied.content);
        let count =
            |desktop: &crate::shell::Shell| desktop.draft().unwrap().tracks[0].segments.len();
        assert_eq!(count(&desktop), 2);
        // The window undoes the assistant's edit like its own.
        desktop.undo().unwrap();
        assert_eq!(count(&desktop), 1);

        let stale = desktop
            .run_agent_tool(&call("draft.apply", &split.to_string()), &[])
            .outcome;
        assert!(stale.is_error, "a stale revision is refused");
        // A combo sent again with its request id is answered as the first
        // time, and runs once.
        let named = json!({
            "expected_revision": desktop.session().unwrap().revision(),
            "request_id": "once",
            "requests": [{"op": "set_project_name", "name": "named once"}],
        });
        let first = desktop
            .run_agent_tool(&call("draft.apply", &named.to_string()), &[])
            .outcome;
        assert!(!first.is_error, "{}", first.content);
        let revision = desktop.session().unwrap().revision();
        let again = desktop
            .run_agent_tool(&call("draft.apply", &named.to_string()), &[])
            .outcome;
        assert_eq!(again, first);
        assert_eq!(desktop.session().unwrap().revision(), revision);

        let seek = desktop
            .run_agent_tool(&call("player.seek", "1000000"), &[])
            .outcome;
        assert!(!seek.is_error, "{}", seek.content);
        assert_eq!(desktop.playhead(), Micros(1_000_000));

        // Replacing a file waits for the user's allow, and touches nothing
        // before it.
        let taken = temp.path.join("taken.png");
        std::fs::write(&taken, b"old").unwrap();
        let replace = call("export.start", &format!("replace {}", taken.display()));
        let asked = desktop.run_agent_tool(&replace, &[]);
        assert!(asked.outcome.is_error);
        let wanted = cutlass_agent::ApprovalAction::ExportReplace {
            path: taken.display().to_string(),
        };
        assert_eq!(asked.wants, Some(wanted.clone()));
        assert_eq!(std::fs::read(&taken).unwrap(), b"old");
        let allowed = desktop.run_agent_tool(&replace, &[wanted]);
        assert!(allowed.wants.is_none());
        assert!(!allowed.outcome.is_error, "{}", allowed.outcome.content);
        assert_ne!(
            std::fs::read(&taken).unwrap(),
            b"old",
            "the still was written"
        );
        // A new file that is taken is refused without asking.
        let new = desktop.run_agent_tool(
            &call("export.start", &format!("new {}", taken.display())),
            &[],
        );
        assert!(new.wants.is_none());
        assert!(new.outcome.is_error);
        let settings = desktop
            .run_agent_tool(&call("settings.get", ""), &[])
            .outcome;
        assert!(!settings.is_error, "{}", settings.content);

        // A settings change lands in the window's settings and its file.
        let file = temp.path.join("settings-home").join("settings.json");
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        desktop.keep_settings_file(file.clone(), 0);
        let set = desktop
            .run_agent_tool(
                &call(
                    "settings.set",
                    r#"{"expected_generation":0,"changes":{"step_cap":40}}"#,
                ),
                &[],
            )
            .outcome;
        assert!(!set.is_error, "{}", set.content);
        assert_eq!(desktop.settings().step_cap, 40);
        assert_eq!(cutlass_settings::load(&file).unwrap().settings.step_cap, 40);
        // Turning on the cloud model waits for the user's allow.
        let cloud = desktop.run_agent_tool(
            &call(
                "settings.set",
                r#"{"expected_generation":1,"changes":{"model":{"cloud":true}}}"#,
            ),
            &[],
        );
        assert_eq!(
            cloud.wants,
            Some(cutlass_agent::ApprovalAction::EnableCloudModel)
        );
        assert!(!desktop.settings().model.cloud);
    }

    #[test]
    fn the_assistant_follows_a_movie_export_to_its_end() {
        let temp = Temp::new();
        // A fifth of a second of blue, 64 by 32.
        let mut draft = Draft::empty("draft");
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
        draft.tracks[0]
            .segments
            .push(Segment::new("blue", "color", TimeRange::new(0, 200_000)));
        draft.duration = Micros(200_000);
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        // A still is written at once, and the start says so.
        let still = temp.path.join("frame.png");
        let frame = desktop
            .run_agent_tool(
                &call("export.start", &format!("new {}", still.display())),
                &[],
            )
            .outcome;
        assert_eq!(
            frame.content,
            format!("job=export-1 finished: wrote {}", still.display())
        );
        // A movie is a job: the start names it and a poll follows it.
        let movie = temp.path.join("cut.mp4");
        let started = desktop
            .run_agent_tool(
                &call("export.start", &format!("new {}", movie.display())),
                &[],
            )
            .outcome;
        assert_eq!(started.content, "job=export-2", "{}", started.content);
        let until = std::time::Instant::now() + std::time::Duration::from_secs(120);
        let polled = loop {
            let polled = desktop
                .run_agent_tool(&call("export.poll", "export-2"), &[])
                .outcome;
            if polled.content != "pending" || std::time::Instant::now() > until {
                break polled;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        };
        assert!(!polled.is_error, "{}", polled.content);
        assert_eq!(
            polled.content,
            format!("finished: wrote {}", movie.display())
        );
        assert_eq!(&std::fs::read(&movie).unwrap()[4..8], b"ftyp");
        // An older job is not this one, and an ending that is neither a
        // still nor a movie is refused before anything is written.
        let old = desktop
            .run_agent_tool(&call("export.poll", "export-1"), &[])
            .outcome;
        assert!(old.is_error, "{}", old.content);
        let photo = temp.path.join("frame.jpg");
        let refused = desktop
            .run_agent_tool(
                &call("export.start", &format!("new {}", photo.display())),
                &[],
            )
            .outcome;
        assert!(refused.is_error && refused.content.contains(".png"));
        assert!(!photo.exists());
    }

    #[test]
    fn the_assistant_tracks_as_a_window_job_and_hears_its_errors() {
        let temp = Temp::new();
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 1920;
        draft.canvas_config.height = 1080;
        draft.materials.videos.push(cutlass_model::MaterialVideo {
            id: "m".into(),
            width: 1920,
            height: 1080,
            path: "missing.mp4".into(),
            ..cutlass_model::MaterialVideo::default()
        });
        draft.tracks[0]
            .segments
            .push(Segment::new("v", "m", TimeRange::new(0, 2_000_000)));
        let mut lane = cutlass_model::Track::main("stickers");
        lane.kind = cutlass_model::TrackType::Sticker;
        lane.segments
            .push(Segment::new("s", "m", TimeRange::new(0, 2_000_000)));
        draft.tracks.push(lane);
        Folder::new(draft).save(&temp.path).unwrap();
        let mut desktop = shell();
        desktop.open_folder(&temp.path).unwrap();
        let ask = |target: &str| {
            json!({
                "target": target,
                "at_us": 500_000,
                "box": [-0.2, 0.2, 0.2, -0.2],
                "followers": ["s"],
            })
            .to_string()
        };
        let refused = desktop
            .run_agent_tool(&call("track.start", &ask("s")), &[])
            .outcome;
        assert!(refused.is_error, "a sticker is not tracked");
        assert!(
            refused.content.contains("video clip"),
            "{}",
            refused.content
        );

        let started = desktop
            .run_agent_tool(&call("track.start", &ask("v")), &[])
            .outcome;
        assert!(!started.is_error, "{}", started.content);
        let job = started
            .content
            .strip_prefix("job=")
            .and_then(|rest| rest.split_whitespace().next())
            .unwrap()
            .to_owned();
        // The file is missing, so the job ends with that, and the poll says so.
        let until = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let polled = loop {
            let polled = desktop
                .run_agent_tool(&call("track.poll", &job), &[])
                .outcome;
            if polled.content != "pending" || std::time::Instant::now() > until {
                break polled;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        };
        assert!(polled.is_error, "{}", polled.content);
        assert!(
            polled.content.contains("could not be opened"),
            "{}",
            polled.content
        );
        let again = desktop
            .run_agent_tool(&call("track.poll", &job), &[])
            .outcome;
        assert!(again.is_error && again.content.contains("unknown tracking job"));
        assert!(
            desktop
                .draft()
                .unwrap()
                .materials
                .video_trackings
                .is_empty()
        );
    }
}
