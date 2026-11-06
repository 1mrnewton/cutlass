//! The user's settings: shortcuts, panels, playback preferences, hardware
//! codecs, model choice, sandbox choice, MCP servers, skill folders, and
//! the update check.
//!
//! Hardware decode and hardware encode are on.
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
pub use shortcuts::{ACTIONS, Shortcut, action_label, default_keys, keys_label};

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
    pub hardware: Hardware,
    #[serde(default)]
    pub import: Import,
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
    /// Project folders opened lately, newest first, at most
    /// [`RECENT_LIMIT`].
    #[serde(default)]
    pub recent_projects: Vec<std::path::PathBuf>,
    #[serde(default)]
    pub updates: Updates,
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
            hardware: Hardware::default(),
            import: Import::default(),
            model: ModelChoice::default(),
            sandbox: SandboxChoice::default(),
            mcp_servers: Vec::new(),
            skill_folders: Vec::new(),
            step_cap: DEFAULT_STEP_CAP,
            memory: true,
            recent_projects: Vec::new(),
            updates: Updates::default(),
            rest: Map::new(),
        }
    }
}

/// Whether Cutlass looks for a newer release, and when it last looked.
///
/// Off, nothing asks the release feed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Updates {
    /// Look for a newer release at launch, at most once a day.
    #[serde(default = "default_on")]
    pub check: bool,
    /// When the last check started, in seconds since 1970; 0 is never.
    #[serde(default)]
    pub last_check: u64,
    /// Keys this record does not name.
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

impl Default for Updates {
    fn default() -> Self {
        Self {
            check: true,
            last_check: 0,
            rest: Map::new(),
        }
    }
}

/// How many recent projects the settings keep.
pub const RECENT_LIMIT: usize = 10;

impl Settings {
    /// Whether the panel `id` shows. A panel the file does not name shows.
    #[must_use]
    pub fn panel_open(&self, id: &str) -> bool {
        self.panels
            .iter()
            .find(|panel| panel.id == id)
            .is_none_or(|panel| panel.open)
    }

    /// Show or hide the panel `id`.
    pub fn set_panel_open(&mut self, id: &str, open: bool) {
        match self.panels.iter_mut().find(|panel| panel.id == id) {
            Some(panel) => panel.open = open,
            None => self.panels.push(Panel {
                id: id.to_owned(),
                open,
                ..Panel::default()
            }),
        }
    }

    /// Put `path` first among the recent projects, once, and drop the
    /// oldest past [`RECENT_LIMIT`].
    pub fn note_recent(&mut self, path: &std::path::Path) {
        self.recent_projects.retain(|recent| recent != path);
        self.recent_projects.insert(0, path.to_path_buf());
        self.recent_projects.truncate(RECENT_LIMIT);
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
/// What an import does with the file.
///
/// By default the file is copied into the project, so the project folder
/// holds everything it plays and can move as one. Linked, the project
/// points at the file where it is: nothing is copied, and moving or deleting
/// the file takes it from the project.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Import {
    /// Point at the file where it is instead of copying it in.
    #[serde(default)]
    pub link: bool,
    /// Keys this record does not name.
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// Preview can play audio while the playhead is dragged. That switch is
/// separate from normal playback. A seek while playing follows
/// [`Playback::seek_during_playback`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    /// Play a clip's proxy in the preview when it has one. Export always
    /// reads the source.
    #[serde(default = "default_on")]
    pub proxies: bool,
    /// Keys this record does not name.
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

impl Default for Playback {
    fn default() -> Self {
        Self {
            loop_playback: false,
            scrub_audio: false,
            seek_during_playback: SeekDuringPlayback::default(),
            proxies: true,
            rest: Map::new(),
        }
    }
}

/// Whether the platform codecs may run.
///
/// Off, the platform video opener is not registered and the Rust codec
/// for the same id is used. A codec with no Rust opener is then refused.
/// Audio and still codecs are not covered by these switches.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hardware {
    /// Decode pictures with the platform decoder, such as VideoToolbox.
    #[serde(default = "default_on")]
    pub decode: bool,
    /// Encode pictures with the platform encoder, such as VideoToolbox.
    #[serde(default = "default_on")]
    pub encode: bool,
    /// Keys this record does not name.
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

impl Default for Hardware {
    fn default() -> Self {
        Self {
            decode: true,
            encode: true,
            rest: Map::new(),
        }
    }
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
    /// Where the local runner answers chat completions, on this machine.
    #[serde(default = "default_local_url")]
    pub local_url: String,
    /// The model the local runner loads, by its name there.
    #[serde(default = "default_local_name")]
    pub local_name: String,
    /// Keys this record does not name.
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// Ollama's chat completions address.
pub const DEFAULT_LOCAL_URL: &str = "http://127.0.0.1:11434/v1/chat/completions";

/// A small model that calls tools.
pub const DEFAULT_LOCAL_NAME: &str = "qwen2.5:7b";

fn default_local_url() -> String {
    DEFAULT_LOCAL_URL.to_owned()
}

fn default_local_name() -> String {
    DEFAULT_LOCAL_NAME.to_owned()
}

impl Default for ModelChoice {
    fn default() -> Self {
        Self {
            local: true,
            cloud: false,
            local_url: default_local_url(),
            local_name: default_local_name(),
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
        DEFAULT_STEP_CAP, Hardware, McpServer, ModelChoice, Playback, SandboxChoice,
        SeekDuringPlayback, Settings,
    };

    #[test]
    fn a_panel_shows_until_it_is_hidden() {
        let mut settings = Settings::default();
        assert!(settings.panel_open("library"));
        settings.set_panel_open("library", false);
        assert!(!settings.panel_open("library"));
        settings.set_panel_open("library", true);
        assert!(settings.panel_open("library"));
        assert_eq!(settings.panels.len(), 1);
    }

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
        assert!(settings.playback.proxies, "proxies play by default");
        assert!(settings.hardware.decode);
        assert!(settings.hardware.encode);
        assert!(!settings.import.link, "imports copy by default");
        assert!(settings.mcp_servers.is_empty());
        assert!(settings.skill_folders.is_empty());
        assert!(settings.updates.check, "updates are checked by default");
        assert_eq!(settings.updates.last_check, 0);
        let parsed: Settings = serde_json::from_str("{}").unwrap();
        assert_eq!(parsed, settings);
        let linked: Settings = serde_json::from_str(r#"{"import":{"link":true}}"#).unwrap();
        assert!(linked.import.link);
        let without: Settings =
            serde_json::from_str(r#"{"playback":{"loop_playback":true}}"#).unwrap();
        assert!(without.playback.proxies && without.playback.loop_playback);
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
    fn each_hardware_switch_is_stored_on_its_own() {
        let settings: Settings = serde_json::from_str(r#"{"hardware":{"decode":false}}"#).unwrap();
        assert!(!settings.hardware.decode);
        assert!(settings.hardware.encode);
        let written = serde_json::to_value(&settings).unwrap();
        assert_eq!(written["hardware"]["decode"], false);
        assert_eq!(written["hardware"]["encode"], true);
        let parsed: Settings = serde_json::from_str("{}").unwrap();
        assert_eq!(parsed.hardware, Hardware::default());
    }

    #[test]
    fn an_omitted_model_field_keeps_its_default() {
        let settings: Settings =
            serde_json::from_str(r#"{"model":{"cloud":true},"sandbox":{"online":true}}"#).unwrap();
        assert!(settings.model.local);
        assert!(settings.model.cloud);
        assert_eq!(settings.model.local_url, super::DEFAULT_LOCAL_URL);
        assert_eq!(settings.model.local_name, super::DEFAULT_LOCAL_NAME);
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
    fn the_update_switch_and_last_check_are_stored() {
        let settings: Settings =
            serde_json::from_str(r#"{"updates":{"check":false,"future_update":1}}"#).unwrap();
        assert!(!settings.updates.check);
        assert_eq!(settings.updates.last_check, 0);
        let mut settings = settings;
        settings.updates.last_check = 1_800_000_000;
        let written = serde_json::to_value(&settings).unwrap();
        assert_eq!(written["updates"]["check"], false);
        assert_eq!(written["updates"]["last_check"], 1_800_000_000u64);
        assert_eq!(written["updates"]["future_update"], 1);
        let read: Settings = serde_json::from_value(written).unwrap();
        assert_eq!(read, settings);
        assert!(serde_json::from_str::<Settings>(r#"{"updates":{"last_check":-1}}"#).is_err());
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

    #[test]
    fn a_recent_project_moves_to_the_front_once_and_the_list_is_capped() {
        let mut settings = Settings::default();
        for n in 0..12 {
            settings.note_recent(std::path::Path::new(&format!("/p/{n}")));
        }
        assert_eq!(settings.recent_projects.len(), super::RECENT_LIMIT);
        assert_eq!(settings.recent_projects[0], std::path::Path::new("/p/11"));
        settings.note_recent(std::path::Path::new("/p/5"));
        assert_eq!(settings.recent_projects[0], std::path::Path::new("/p/5"));
        assert_eq!(settings.recent_projects.len(), super::RECENT_LIMIT);
        assert_eq!(
            settings
                .recent_projects
                .iter()
                .filter(|path| path.ends_with("5"))
                .count(),
            1
        );
    }
}
