//! A host's stand-in for the tools it cannot serve: each call fails with
//! one plain sentence, so the model hears why and moves on.
//!
//! A phone serves the draft, settings, skills and scripts; playback,
//! export, speech, tracking, media probes, the catalog, packages and MCP
//! servers stay with the desktop for now. [`tools_for`] keeps them out of
//! what the model is offered at all.

use cutlass_model::{Draft, SegmentId, TrackPath};

use crate::approval::{ApprovalAction, Approver};
use crate::mcp::{McpError, McpTransport};
use crate::model::ToolDefinition;
#[cfg(feature = "script")]
use crate::sandbox::{SandboxError, SandboxExecutor};
use crate::tools::{
    Catalog, Export, Media, MediaInfo, Packages, Player, Speech, TrackAsk, Tracking,
};

/// Every call fails with `0` as its reason.
#[derive(Clone, Copy, Debug)]
pub struct Unavailable(pub &'static str);

impl Unavailable {
    fn no<T>(self) -> Result<T, String> {
        Err(self.0.to_owned())
    }
}

impl Player for Unavailable {
    fn seek(&mut self, _time_us: i64) -> Result<(), String> {
        self.no()
    }

    fn play(&mut self) -> Result<(), String> {
        self.no()
    }

    fn pause(&mut self) -> Result<(), String> {
        self.no()
    }
}

impl Media for Unavailable {
    fn probe(&mut self, _path: &str) -> Result<MediaInfo, String> {
        self.no()
    }
}

impl Speech for Unavailable {
    fn start(&mut self, _start_us: i64, _duration_us: i64) -> Result<String, String> {
        self.no()
    }

    fn poll(&mut self, _job_id: &str) -> Result<Option<String>, String> {
        self.no()
    }
}

impl Tracking for Unavailable {
    fn start(&mut self, _ask: &TrackAsk) -> Result<String, String> {
        self.no()
    }

    fn poll(&mut self, _job_id: &str) -> Result<Option<TrackPath>, String> {
        self.no()
    }

    fn clip_path(&self, _draft: &Draft, _clip: &SegmentId, _path: &TrackPath) -> Option<TrackPath> {
        None
    }
}

impl Catalog for Unavailable {
    fn search(&mut self, _query: &str) -> Result<Vec<String>, String> {
        self.no()
    }
}

impl Packages for Unavailable {
    fn ensure(&mut self, _package_id: &str) -> Result<(), String> {
        self.no()
    }
}

impl Export for Unavailable {
    fn start(&mut self, _path: &str) -> Result<String, String> {
        self.no()
    }
}

impl McpTransport for Unavailable {
    fn call(&mut self, _server: &str, _tool: &str, _arguments: &str) -> Result<String, McpError> {
        Err(McpError::Transport(self.0.to_owned()))
    }
}

/// Says no to every allow: a host with no one to ask.
impl Approver for Unavailable {
    fn decide(&mut self, _action: &ApprovalAction) -> bool {
        false
    }
}

/// Scripts run in this process, under the script engine's own limits on
/// operations, depth and size. For a host that cannot start a child
/// process, as a phone cannot.
#[cfg(feature = "script")]
#[derive(Clone, Copy, Debug, Default)]
pub struct InProcess;

#[cfg(feature = "script")]
impl SandboxExecutor for InProcess {
    fn execute(&mut self, source: &str, input_json: &str) -> Result<String, SandboxError> {
        crate::script::run_script(source, input_json)
    }
}

/// The definitions of `tools` only, in the order the registry lists them.
#[must_use]
pub fn tools_for(tools: &[&str]) -> Vec<ToolDefinition> {
    crate::tools::tool_definitions()
        .into_iter()
        .filter(|definition| tools.contains(&definition.name.as_str()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{Unavailable, tools_for};
    use crate::tools::{Export, Player};

    #[test]
    fn every_call_says_why_and_only_named_tools_are_offered() {
        let mut host = Unavailable("not on phones yet");
        assert_eq!(Player::seek(&mut host, 0).unwrap_err(), "not on phones yet");
        assert_eq!(
            Export::start(&mut host, "a.mp4").unwrap_err(),
            "not on phones yet"
        );
        let offered: Vec<String> = tools_for(&["skill.load", "draft.query"])
            .into_iter()
            .map(|definition| definition.name)
            .collect();
        assert_eq!(offered, ["draft.query", "skill.load"]);
    }
}
