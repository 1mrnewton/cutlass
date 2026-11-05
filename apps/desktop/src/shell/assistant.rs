//! The assistant's tools on the open project. Each call runs on the
//! window's thread against the session the window holds, so an edit the
//! assistant makes is one undo step like any other, and the player moves
//! the window's playhead.

use cutlass_agent::{
    ApprovalAction, Approver, DraftTools, McpClient, McpError, McpTransport, Player, Registry,
    SettingsTool, ToolCall, ToolOutcome, ToolSink,
};
use cutlass_model::Micros;

use super::Shell;
use crate::mcp::{NoSandbox, Probe, Unbuilt};

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

/// The assistant does not ask for approvals yet, so anything that needs
/// one is refused, and the model sees why.
struct Unasked;

impl Approver for Unasked {
    fn decide(&mut self, action: &ApprovalAction) -> bool {
        cutlass_logger::info!("assistant: refused {action:?}; approvals are made in Settings");
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
            "the assistant does not call other servers yet, so {server_name} was not reached"
        )))
    }
}

impl Shell {
    /// Run one tool call from the assistant on the open project.
    pub fn run_agent_tool(&mut self, call: &ToolCall) -> ToolOutcome {
        let Some(session) = self.session.clone() else {
            return ToolOutcome {
                content: "no project is open".to_owned(),
                is_error: true,
            };
        };
        let before = session.revision();
        let (path, generation) = self
            .settings_file
            .clone()
            .unwrap_or_else(|| (Shell::settings_path(), 0));
        let settings = SettingsTool::held(path, generation, self.settings.clone());
        let mut registry = Registry::new(
            DraftTools::new(session),
            settings,
            Moves::default(),
            Probe,
            Unbuilt(IN_THE_WINDOW),
            Unbuilt(IN_THE_WINDOW),
            (
                Unbuilt(IN_THE_WINDOW),
                Unbuilt(IN_THE_WINDOW),
                NoSandbox,
                Unasked,
                McpClient::new(NoServers),
            ),
        );
        let outcome = registry.run_tool(call);
        let session = registry.draft.session();
        if session.revision() != before {
            self.session = Some(session.clone());
            self.lanes_cache.replace(None);
        }
        if let Some(time) = registry.player.seek {
            self.seek(Micros(time));
        }
        match registry.player.play {
            Some(true) => self.play(),
            Some(false) => self.pause(),
            None => {}
        }
        outcome
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
        let none = desktop.run_agent_tool(&call("draft.query", "project"));
        assert!(none.is_error);
        desktop.open_folder(&temp.path).unwrap();

        let read = desktop.run_agent_tool(&call("draft.query", "project"));
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
        let applied = desktop.run_agent_tool(&call("draft.apply", &split.to_string()));
        assert!(!applied.is_error, "{}", applied.content);
        let count =
            |desktop: &crate::shell::Shell| desktop.draft().unwrap().tracks[0].segments.len();
        assert_eq!(count(&desktop), 2);
        // The window undoes the assistant's edit like its own.
        desktop.undo().unwrap();
        assert_eq!(count(&desktop), 1);

        let stale = desktop.run_agent_tool(&call("draft.apply", &split.to_string()));
        assert!(stale.is_error, "a stale revision is refused");

        let seek = desktop.run_agent_tool(&call("player.seek", "1000000"));
        assert!(!seek.is_error, "{}", seek.content);
        assert_eq!(desktop.playhead(), Micros(1_000_000));

        let export = desktop.run_agent_tool(&call("export.start", "new /tmp/x.mp4"));
        assert!(export.is_error);
        assert!(
            export.content.contains("not open to the assistant"),
            "{}",
            export.content
        );
        let settings = desktop.run_agent_tool(&call("settings.get", ""));
        assert!(!settings.is_error, "{}", settings.content);
    }
}
