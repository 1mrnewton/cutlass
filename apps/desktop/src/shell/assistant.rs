//! The assistant's tools on the open project. Each call runs on the
//! window's thread against the session the window holds, so an edit the
//! assistant makes is one undo step like any other, and the player moves
//! the window's playhead.

use cutlass_agent::{
    ApprovalAction, Approver, DraftTools, Export, McpClient, McpError, McpTransport, Player,
    Registry, SettingsTool, Speech, ToolCall, ToolOutcome, ToolSink,
};
use cutlass_codec::CodecId;
use cutlass_model::Micros;

use super::{ExportRequest, Shell};
use crate::mcp::{Bundled, Probe, Unbuilt};

/// Where the tools this window does not hand the assistant yet are.
const IN_THE_WINDOW: &str = "not open to the assistant yet; use the window for it";

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

/// An export the window starts once the call is done.
#[derive(Default)]
struct Exports {
    path: Option<String>,
}

impl Export for Exports {
    fn start(&mut self, path: &str) -> Result<String, String> {
        if path.trim().is_empty() {
            return Err("name the file to export to".to_owned());
        }
        self.path = Some(path.to_owned());
        Ok("export".to_owned())
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
            "the assistant does not call other servers yet, so {server_name} was not reached"
        )))
    }
}

/// What one assistant tool call did, and the approval it waits for when
/// it needs one the user has not given.
pub struct AgentRun {
    pub outcome: ToolOutcome,
    pub wants: Option<ApprovalAction>,
}

impl Shell {
    /// Run one tool call from the assistant on the open project. `allowed`
    /// is what the user allowed for this call; a call that needs anything
    /// else does nothing and says what it wants.
    pub fn run_agent_tool(&mut self, call: &ToolCall, allowed: &[ApprovalAction]) -> AgentRun {
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
        let settings = SettingsTool::held(path, generation, self.settings.clone());
        let jobs = std::mem::take(&mut self.agent_speech_jobs);
        let sandbox =
            crate::sandbox::local_sandbox(crate::sandbox::this_program(), self.project_folder());
        let mut registry = Registry::new(
            DraftTools::new(session),
            settings,
            Moves::default(),
            Probe,
            Hearing { shell: self },
            Bundled,
            (
                Unbuilt(IN_THE_WINDOW),
                Exports::default(),
                sandbox,
                Asked {
                    allowed,
                    wanted: None,
                },
                McpClient::new(NoServers),
            ),
        );
        registry.set_speech_jobs(jobs);
        let mut outcome = registry.run_tool(call);
        let wants = registry.approver.wanted.take();
        let session = registry.draft.session().clone();
        let moves = std::mem::take(&mut registry.player);
        let exported = registry.export.path.take();
        let jobs = registry.take_speech_jobs();
        drop(registry);
        self.agent_speech_jobs = jobs;
        if session.revision() != before {
            self.session = Some(session);
            self.lanes_cache.replace(None);
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
            let codec = if path.to_ascii_lowercase().ends_with(".png") {
                CodecId::Png
            } else {
                CodecId::H264
            };
            let started = self.export(ExportRequest {
                path: path.into(),
                codec,
                replace,
                ..ExportRequest::default()
            });
            if let Err(error) = started {
                outcome = ToolOutcome {
                    content: error.to_string(),
                    is_error: true,
                };
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
    }
}
