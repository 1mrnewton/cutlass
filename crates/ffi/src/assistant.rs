//! The assistant's tools on a phone: the same registry the desktop runs,
//! over this project's session.
//!
//! The phone serves the draft, settings, skills and scripts (in this
//! process, under the script engine's limits). Playback, export, speech,
//! tracking, media probes, the catalog, packages and MCP servers stay with
//! the desktop for now; they are not offered to the model, and a call that
//! names one anyway hears why. Nothing that needs an allow runs here: the
//! phone has no allow dialog yet, so those calls are refused.

use std::path::Path;

use cutlass_agent::{
    DraftTools, InProcess, LocalSandbox, McpClient, Registry, SandboxPolicy, SettingsTool,
    Switched, ToolCall, ToolDefinition, ToolOutcome, ToolSink, Unavailable,
};

use crate::PhoneHost;

/// The tools a phone offers the model.
pub const PHONE_TOOLS: &[&str] = &[
    "draft.query",
    "draft.apply",
    "draft.undo",
    "draft.redo",
    "settings.get",
    "settings.set",
    "skill.load",
    "sandbox.run",
];

/// Why a desktop tool fails here.
const NOT_HERE: Unavailable = Unavailable("this tool is on the desktop app only for now");

/// The definitions of [`PHONE_TOOLS`], for the model.
#[must_use]
pub fn phone_tools() -> Vec<ToolDefinition> {
    cutlass_agent::tools_for(PHONE_TOOLS)
}

impl PhoneHost {
    /// Run one of the assistant's tool calls on the open project, with the
    /// settings in `settings_file`. A change to the draft is one undo step,
    /// as any other edit is, and is saved.
    pub fn run_agent_tool(&mut self, call: &ToolCall, settings_file: &Path) -> ToolOutcome {
        if !PHONE_TOOLS.contains(&call.name.as_str()) {
            return ToolOutcome {
                content: format!("{} is on the desktop app only for now", call.name),
                is_error: true,
            };
        }
        let settings = match SettingsTool::open(settings_file) {
            Ok(settings) => settings,
            Err(error) => {
                return ToolOutcome {
                    content: format!("the settings do not read: {error}"),
                    is_error: true,
                };
            }
        };
        let before = self.revision();
        let sandbox = Switched::from_settings(
            settings.settings(),
            LocalSandbox::new(
                SandboxPolicy {
                    project_path: self.project.root().to_string_lossy().into_owned(),
                    home_path: String::new(),
                },
                InProcess,
            ),
        );
        let mut registry = Registry::new(
            DraftTools::new(self.session().clone()),
            settings,
            NOT_HERE,
            NOT_HERE,
            NOT_HERE,
            NOT_HERE,
            (
                NOT_HERE,
                NOT_HERE,
                sandbox,
                NOT_HERE,
                McpClient::new(NOT_HERE),
                NOT_HERE,
            ),
        );
        let outcome = registry.run_tool(call);
        let session = registry.draft.session().clone();
        drop(registry);
        if session.revision() != before {
            self.commands.adopt(session);
            if let Err(error) = self.save() {
                cutlass_logger::warn!("assistant edit not saved: {error}");
            }
        }
        outcome
    }
}
