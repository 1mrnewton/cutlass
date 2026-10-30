//! The user's settings: shortcuts, panels, playback preferences, model
//! choice, sandbox choice, MCP servers, and skill folders.
//!
//! The local model switch is on. The cloud model switch is off. The local
//! sandbox is on. The online sandbox is off. This crate has no draft types
//! and does not open a socket.
//!
//! A load returns the generation stored in the file. A save sends that
//! generation. A mismatch writes nothing. A successful save stores the
//! next generation.

#![forbid(unsafe_code)]

mod file;
mod shortcuts;

pub use file::{Loaded, SettingsError, load, save};
pub use shortcuts::Shortcut;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// How many model rounds one turn may take. The user can raise it.
pub const DEFAULT_STEP_CAP: u32 = 32;

const fn default_on() -> bool {
    true
}

const fn default_step_cap() -> u32 {
    DEFAULT_STEP_CAP
}

fn default_stdio() -> String {
    McpServer::STDIO.to_owned()
}

/// Shortcuts, panels, playback, models, servers, and skill folders.
///
/// A field this file does not name stays in [`Settings::rest`] and does not
/// clear the fields it does name.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Settings {
    #[serde(default)]
    pub shortcuts: Vec<Shortcut>,
    #[serde(default)]
    pub panels: Vec<Panel>,
    #[serde(default)]
    pub playback: Playback,
    #[serde(default)]
    pub model: ModelChoice,
    #[serde(default)]
    pub sandbox: SandboxChoice,
    #[serde(default)]
    pub mcp_servers: Vec<McpServer>,
    #[serde(default)]
    pub skill_folders: Vec<SkillFolder>,
    /// Rounds one turn may take before the harness stops.
    #[serde(default = "default_step_cap")]
    pub step_cap: u32,
    /// Local notes. They leave the machine only while the cloud model is on.
    #[serde(default = "default_on")]
    pub memory: bool,
    /// Keys this file does not name.
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            shortcuts: Vec::new(),
            panels: Vec::new(),
            playback: Playback::default(),
            model: ModelChoice::default(),
            sandbox: SandboxChoice::default(),
            mcp_servers: Vec::new(),
            skill_folders: Vec::new(),
            step_cap: DEFAULT_STEP_CAP,
            memory: true,
            rest: Map::new(),
        }
    }
}

/// One panel the shell can show.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Panel {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub open: bool,
    /// Keys this record does not name.
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// What a seek does while playback is already running.
///
/// The two values are stored. A seek does not invent its own branch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeekDuringPlayback {
    /// The playhead moves and playback continues.
    #[default]
    KeepPlaying,
    /// The playhead moves and playback stops.
    Pause,
}

/// How the open player behaves. These are not draft fields.
///
/// Preview can play audio while the playhead is dragged. That switch is
/// separate from normal playback. A seek while playing follows
/// [`Playback::seek_during_playback`].
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Playback {
    /// Start again at the beginning when playback reaches the end.
    #[serde(default)]
    pub loop_playback: bool,
    /// Play audio while the playhead is dragged.
    #[serde(default)]
    pub scrub_audio: bool,
    /// Keep playing or pause when the playhead is seeked during playback.
    #[serde(default)]
    pub seek_during_playback: SeekDuringPlayback,
    /// Keys this record does not name.
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// Which model runners are on.
///
/// The two switches are independent. A cloud model does not turn the
/// online sandbox on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelChoice {
    /// Weights and the prompt stay on the machine.
    #[serde(default = "default_on")]
    pub local: bool,
    /// Off until an account exists and the user enables it.
    #[serde(default)]
    pub cloud: bool,
    /// Keys this record does not name.
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

impl Default for ModelChoice {
    fn default() -> Self {
        Self {
            local: true,
            cloud: false,
            rest: Map::new(),
        }
    }
}

/// Which code runners are on.
///
/// The local runner ships. The online runner stays off until the user
/// turns it on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SandboxChoice {
    #[serde(default = "default_on")]
    pub local: bool,
    #[serde(default)]
    pub online: bool,
    /// Keys this record does not name.
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

impl Default for SandboxChoice {
    fn default() -> Self {
        Self {
            local: true,
            online: false,
            rest: Map::new(),
        }
    }
}

/// One server the user added.
///
/// A local stdio server is the kind that ships. A remote URL is stored
/// and stays disabled until the user enables that server. This crate
/// does not contact the URL.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct McpServer {
    #[serde(default)]
    pub name: String,
    /// Off until the user enables this server.
    #[serde(default)]
    pub enabled: bool,
    /// [`Self::STDIO`] or [`Self::REMOTE`]. Any other word is kept.
    #[serde(default = "default_stdio")]
    pub kind: String,
    #[serde(default)]
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub url: String,
    /// Keys this record does not name.
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

impl McpServer {
    /// A local process. Its arguments stay on the machine.
    pub const STDIO: &'static str = "stdio";
    /// A remote URL. Enabling it is a separate settings change.
    pub const REMOTE: &'static str = "remote";
}

impl Default for McpServer {
    fn default() -> Self {
        Self {
            name: String::new(),
            enabled: false,
            kind: Self::STDIO.to_owned(),
            command: String::new(),
            args: Vec::new(),
            url: String::new(),
            rest: Map::new(),
        }
    }
}

/// A folder of user skills. The path is stored and not opened.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct SkillFolder {
    #[serde(default)]
    pub path: String,
    /// Keys this record does not name.
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_STEP_CAP, McpServer, ModelChoice, Playback, SandboxChoice, SeekDuringPlayback,
        Settings,
    };

    #[test]
    fn defaults_match_an_empty_object() {
        let settings = Settings::default();
        assert!(settings.model.local);
        assert!(!settings.model.cloud);
        assert!(settings.sandbox.local);
        assert!(!settings.sandbox.online);
        assert!(settings.memory);
        assert_eq!(settings.step_cap, DEFAULT_STEP_CAP);
        assert_eq!(settings.step_cap, 32);
        assert!(settings.shortcuts.is_empty());
        assert!(settings.panels.is_empty());
        assert_eq!(settings.playback, Playback::default());
        assert!(!settings.playback.loop_playback);
        assert!(!settings.playback.scrub_audio);
        assert_eq!(
            settings.playback.seek_during_playback,
            SeekDuringPlayback::KeepPlaying
        );
        assert!(settings.mcp_servers.is_empty());
        assert!(settings.skill_folders.is_empty());
        let parsed: Settings = serde_json::from_str("{}").unwrap();
        assert_eq!(parsed, settings);
        assert_eq!(parsed.model, ModelChoice::default());
        assert_eq!(parsed.sandbox, SandboxChoice::default());
    }

    #[test]
    fn a_stored_switch_beats_the_default() {
        let settings: Settings = serde_json::from_str(
            r#"{"model":{"local":false,"cloud":true},"sandbox":{"local":false,"online":true},"memory":false,"step_cap":0,"playback":{"loop_playback":true,"scrub_audio":true}}"#,
        )
        .unwrap();
        assert!(!settings.model.local);
        assert!(settings.model.cloud);
        assert!(!settings.sandbox.local);
        assert!(settings.sandbox.online);
        assert!(!settings.memory);
        assert_eq!(settings.step_cap, 0);
        assert!(settings.playback.loop_playback);
        assert!(settings.playback.scrub_audio);
        assert_eq!(
            settings.playback.seek_during_playback,
            SeekDuringPlayback::KeepPlaying
        );
    }

    #[test]
    fn seek_during_playback_stores_both_choices() {
        let keep: Settings =
            serde_json::from_str(r#"{"playback":{"seek_during_playback":"keep_playing"}}"#)
                .unwrap();
        let pause: Settings =
            serde_json::from_str(r#"{"playback":{"seek_during_playback":"pause"}}"#).unwrap();
        assert_eq!(
            keep.playback.seek_during_playback,
            SeekDuringPlayback::KeepPlaying
        );
        assert_eq!(
            pause.playback.seek_during_playback,
            SeekDuringPlayback::Pause
        );
        let written = serde_json::to_value(&pause).unwrap();
        assert_eq!(written["playback"]["seek_during_playback"], "pause");
        assert!(!pause.playback.scrub_audio);
        let omitted: Settings =
            serde_json::from_str(r#"{"playback":{"scrub_audio":true}}"#).unwrap();
        assert!(omitted.playback.scrub_audio);
        assert_eq!(
            omitted.playback.seek_during_playback,
            SeekDuringPlayback::KeepPlaying
        );
    }

    #[test]
    fn an_omitted_model_field_keeps_its_default() {
        let settings: Settings =
            serde_json::from_str(r#"{"model":{"cloud":true},"sandbox":{"online":true}}"#).unwrap();
        assert!(settings.model.local);
        assert!(settings.model.cloud);
        assert!(settings.sandbox.local);
        assert!(settings.sandbox.online);
    }

    #[test]
    fn an_unknown_field_leaves_the_known_fields() {
        let settings: Settings = serde_json::from_str(
            r#"{"shortcuts":[{"action":"split","keys":"s","future_key":1}],"panels":[{"id":"media","open":true,"future_panel":2}],"model":{"cloud":false,"future_model":"kept"},"playback":{"loop_playback":true,"future_playback":3},"mcp_servers":[{"name":"far","kind":"remote","url":"http://127.0.0.1:9/mcp","future_server":4}],"skill_folders":[{"path":"../skills/demo","future_skill":true}],"future_switch":true}"#,
        )
        .unwrap();
        assert_eq!(settings.shortcuts[0].action, "split");
        assert_eq!(settings.shortcuts[0].keys, "s");
        assert_eq!(settings.shortcuts[0].rest["future_key"], 1);
        assert_eq!(settings.panels[0].id, "media");
        assert!(settings.panels[0].open);
        assert_eq!(settings.panels[0].rest["future_panel"], 2);
        assert!(settings.model.local);
        assert!(!settings.model.cloud);
        assert_eq!(settings.model.rest["future_model"], "kept");
        assert!(settings.playback.loop_playback);
        assert_eq!(settings.playback.rest["future_playback"], 3);
        assert_eq!(settings.mcp_servers[0].name, "far");
        assert_eq!(settings.mcp_servers[0].kind, McpServer::REMOTE);
        assert!(!settings.mcp_servers[0].enabled);
        assert_eq!(settings.mcp_servers[0].url, "http://127.0.0.1:9/mcp");
        assert_eq!(settings.mcp_servers[0].rest["future_server"], 4);
        assert_eq!(settings.skill_folders[0].path, "../skills/demo");
        assert_eq!(settings.skill_folders[0].rest["future_skill"], true);
        assert_eq!(settings.rest["future_switch"], true);

        let value = serde_json::to_value(&settings).unwrap();
        assert_eq!(value["shortcuts"][0]["action"], "split");
        assert_eq!(value["shortcuts"][0]["future_key"], 1);
        assert_eq!(value["panels"][0]["id"], "media");
        assert_eq!(value["model"]["local"], true);
        assert_eq!(value["model"]["cloud"], false);
        assert_eq!(value["model"]["future_model"], "kept");
        assert_eq!(value["playback"]["future_playback"], 3);
        assert_eq!(value["mcp_servers"][0]["url"], "http://127.0.0.1:9/mcp");
        assert_eq!(value["mcp_servers"][0]["enabled"], false);
        assert_eq!(value["skill_folders"][0]["path"], "../skills/demo");
        assert_eq!(value["future_switch"], true);
        assert!(settings.memory);
        assert_eq!(settings.step_cap, 32);
    }

    #[test]
    fn a_local_server_defaults_to_stdio_and_stays_off() {
        let settings: Settings =
            serde_json::from_str(r#"{"mcp_servers":[{"name":"tools","command":"tool"}]}"#).unwrap();
        let server = &settings.mcp_servers[0];
        assert_eq!(server.kind, McpServer::STDIO);
        assert!(!server.enabled);
        assert_eq!(server.command, "tool");
        assert!(server.url.is_empty());
        assert_eq!(McpServer::default().kind, McpServer::STDIO);
        assert!(!McpServer::default().enabled);
    }

    #[test]
    fn a_null_switch_is_refused() {
        let error = serde_json::from_str::<Settings>(r#"{"model":{"local":null}}"#);
        assert!(error.is_err());
        let error = serde_json::from_str::<Settings>(r#"{"shortcuts":"split"}"#);
        assert!(error.is_err());
        let error = serde_json::from_str::<Settings>("[]");
        assert!(error.is_err());
    }
}
