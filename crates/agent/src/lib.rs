//! The harness for the open project.
//!
//! One ask becomes model rounds. Tool results go back to the model. Draft
//! writes are one session combo. Settings writes use the settings generation.
//! The desktop injects the player, media, speech, catalog, packages, export,
//! sandbox executor, and MCP transport. This crate does not open media, start
//! a download, or open a socket.

#![forbid(unsafe_code)]

mod approval;
mod draft;
mod r#loop;
mod mcp;
mod model;
mod sandbox;
mod settings_tool;
mod skill;
mod subagent;
mod tools;

pub use approval::{ApprovalAction, ApprovalError, Approver, authorize, authorize_then};
pub use draft::{
    ApplyInput, ApplyMode, ApplyResult, DraftError, DraftQuery, DraftTools, FullSnapshot,
    ProjectSnapshot, QueryView, RangeSnapshot, SegmentSnapshot, TrackSnapshot, request_name,
};
pub use r#loop::{LoopError, ToolOutcome, ToolSink, Turn, cap_from_settings, run};
pub use mcp::{McpClient, McpError, McpTransport};
pub use model::{
    CloudModel, Completion, CompletionRequest, LocalModel, Message, Model, ModelError, ToolCall,
    ToolDefinition,
};
pub use sandbox::{
    LocalSandbox, OnlineSandbox, Sandbox, SandboxError, SandboxExecutor, SandboxPolicy,
    SandboxRoute, route_for,
};
pub use settings_tool::{SettingsTool, SettingsToolError, SettingsView};
pub use skill::{Skill, SkillError, SkillLibrary, SkillPlan, parse_skill_md};
pub use subagent::{Subagent, SubagentError, SubagentResult, allowed_tools, parent_apply};
pub use tools::{
    Catalog, Export, Invocation, Media, MediaInfo, Packages, Player, Registry, Speech, SpeechJob,
    SpeechPoll, TOOL_NAMES, ToolError, tool_definitions,
};

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use cutlass_model::Draft;
    use cutlass_session::Session;
    use cutlass_settings::Settings;

    use crate::approval::{ApprovalAction, Approver};
    use crate::draft::DraftTools;
    use crate::r#loop::{cap_from_settings, run};
    use crate::mcp::{McpClient, McpError, McpTransport};
    use crate::model::{Completion, LocalModel, Message, ToolCall};
    use crate::sandbox::{Sandbox, SandboxError};
    use crate::settings_tool::SettingsTool;
    use crate::tools::{
        Catalog, Export, Media, MediaInfo, Packages, Player, Registry, Speech, tool_definitions,
    };

    struct Temp {
        dir: PathBuf,
    }

    impl Temp {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let dir =
                std::env::temp_dir().join(format!("cutlass-agent-turn-{}-{n}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            Self { dir }
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    struct Host;

    impl Player for Host {
        fn seek(&mut self, _time_us: i64) -> Result<(), String> {
            Ok(())
        }

        fn play(&mut self) -> Result<(), String> {
            Ok(())
        }

        fn pause(&mut self) -> Result<(), String> {
            Ok(())
        }
    }

    impl Media for Host {
        fn probe(&mut self, _path: &str) -> Result<MediaInfo, String> {
            Ok(MediaInfo {
                duration_us: 0,
                width: 0,
                height: 0,
                streams: Vec::new(),
            })
        }
    }

    impl Speech for Host {
        fn start(&mut self, _start_us: i64, _duration_us: i64) -> Result<String, String> {
            Ok("job".to_owned())
        }

        fn poll(&mut self, _job_id: &str) -> Result<String, String> {
            Ok(String::new())
        }
    }

    impl Catalog for Host {
        fn search(&mut self, _query: &str) -> Result<Vec<String>, String> {
            Ok(Vec::new())
        }
    }

    impl Packages for Host {
        fn ensure(&mut self, _package_id: &str) -> Result<(), String> {
            Ok(())
        }
    }

    impl Export for Host {
        fn start(&mut self, _path: &str) -> Result<String, String> {
            Ok("export".to_owned())
        }
    }

    impl Sandbox for Host {
        fn run(&mut self, _source: &str, input_json: &str) -> Result<String, SandboxError> {
            Ok(input_json.to_owned())
        }
    }

    impl Approver for Host {
        fn decide(&mut self, _action: &ApprovalAction) -> bool {
            false
        }
    }

    impl McpTransport for Host {
        fn call(
            &mut self,
            _server_name: &str,
            _tool: &str,
            _arguments: &str,
        ) -> Result<String, McpError> {
            Ok("ok".to_owned())
        }
    }

    #[test]
    fn a_turn_reads_the_draft_through_the_registry() {
        let temp = Temp::new();
        let settings = Settings::default();
        let mut registry = Registry::new(
            DraftTools::new(Session::open(Draft::empty("D"))),
            SettingsTool::open(temp.dir.join("settings.json")).unwrap(),
            Host,
            Host,
            Host,
            Host,
            (Host, Host, Host, Host, McpClient::new(Host)),
        );
        let mut model = LocalModel::scripted(vec![
            Completion {
                text: String::new(),
                calls: vec![ToolCall {
                    id: "1".to_owned(),
                    name: "draft.query".to_owned(),
                    arguments: "project".to_owned(),
                }],
                proposed: Vec::new(),
            },
            Completion {
                text: "done".to_owned(),
                calls: Vec::new(),
                proposed: Vec::new(),
            },
        ]);
        let turn = run(
            &mut model,
            &mut registry,
            cap_from_settings(&settings),
            "what is the revision?",
            &tool_definitions(),
        )
        .unwrap();
        assert_eq!(settings.step_cap, 32);
        assert_eq!(turn.rounds, 2);
        assert_eq!(registry.draft.session().revision(), 0);
        let seen = model.seen();
        assert_eq!(seen.len(), 2);
        assert!(seen[1].messages.iter().any(|message| {
            matches!(
                message,
                Message::Tool {
                    name,
                    content,
                    is_error: false,
                    ..
                } if name == "draft.query" && content.contains("revision=0")
            )
        }));
    }
}
