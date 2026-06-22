//! Cutlass user settings: the typed model and IO for `~/.cutlass/config.toml`.
//!
//! This crate is the **single owner** of the user config file. Everything the
//! app persists between runs that isn't project data or the recents/autosave
//! sidecars (those live in the OS data dir, see `cutlass-ui::paths`) lives
//! here: the AI provider, the theme, the autosave cadence, and the cache
//! limits. Keys never live in project files — the `[ai]` table is the
//! historical home for the API key and stays here.
//!
//! Two design rules carried over from the rest of the app:
//!
//! - **Loading is tolerant.** A missing file is all-defaults — a fresh
//!   install is a normal state, never an error (the `recent.json`
//!   philosophy). Only a *malformed* file is an `Err`, so callers can choose
//!   to surface the parse problem (the agent does) or fall back to defaults
//!   (app startup does, via `unwrap_or_default`).
//! - **Writing is format-preserving.** [`save`] round-trips the existing file
//!   through `toml_edit`, so hand-written comments, key ordering, and any
//!   tables a newer build added all survive a save from an older one. We only
//!   ever touch the keys we own.
//!
//! ```toml
//! [ai]
//! base_url = "http://localhost:11434/v1"   # Ollama
//! model = "qwen3:14b"
//! # api_key = "sk-..."             # literal key, or:
//! # api_key_env = "OPENAI_API_KEY"  # read from the environment
//!
//! [appearance]
//! theme = "dark-blue"              # "default" | "ember" | "dark-blue"
//!
//! [cache]
//! budget_mb = 51200               # 50 GiB
//! # dir = "/path/to/cache"        # absent ⇒ the OS cache dir
//! ```

use std::path::{Path, PathBuf};

use toml_edit::{DocumentMut, Item, Table, value};

/// Default on-disk frame cache budget in MiB (50 GiB), mirroring
/// `cutlass_engine::DEFAULT_CACHE_BUDGET_BYTES` so an unconfigured install
/// matches the engine default exactly.
pub const DEFAULT_CACHE_BUDGET_MB: u64 = 50 * 1024;

/// Which bundled theme the shell renders. The variant order matches the
/// `index()` the UI dropdown uses; `DarkBlue` is the shipped default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThemeChoice {
    /// Cool graphite + teal (`DefaultTheme`).
    Default,
    /// Warm ember (`EmberTheme`).
    Ember,
    /// The shipped look (`DarkBlueTheme`).
    #[default]
    DarkBlue,
}

impl ThemeChoice {
    /// Every choice, in dropdown order.
    pub const ALL: [ThemeChoice; 3] = [
        ThemeChoice::Default,
        ThemeChoice::Ember,
        ThemeChoice::DarkBlue,
    ];

    /// The stable string written to `config.toml`.
    pub fn key(self) -> &'static str {
        match self {
            ThemeChoice::Default => "default",
            ThemeChoice::Ember => "ember",
            ThemeChoice::DarkBlue => "dark-blue",
        }
    }

    /// Parse a `config.toml` value; `None` for anything unrecognized (the
    /// caller keeps the default rather than failing the whole load).
    pub fn from_key(s: &str) -> Option<Self> {
        match s {
            "default" => Some(ThemeChoice::Default),
            "ember" => Some(ThemeChoice::Ember),
            "dark-blue" | "dark_blue" => Some(ThemeChoice::DarkBlue),
            _ => None,
        }
    }

    /// 0-based index for the Slint dropdown.
    pub fn index(self) -> i32 {
        match self {
            ThemeChoice::Default => 0,
            ThemeChoice::Ember => 1,
            ThemeChoice::DarkBlue => 2,
        }
    }

    /// Inverse of [`index`](Self::index); out-of-range falls back to the
    /// shipped default.
    pub fn from_index(i: i32) -> Self {
        match i {
            0 => ThemeChoice::Default,
            1 => ThemeChoice::Ember,
            _ => ThemeChoice::DarkBlue,
        }
    }
}

/// The `[ai]` table: how the agent reaches an OpenAI-compatible endpoint.
/// Plain data — key *resolution* (the `api_key_env` indirection) is an
/// AI-domain concern and lives in `cutlass_ai::config`. The default (empty
/// endpoint + model) is the "not configured" state
/// [`is_configured`](Self::is_configured) reports.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AiSettings {
    /// OpenAI-compatible endpoint root, e.g. `http://localhost:11434/v1`.
    pub base_url: String,
    /// Model name as the endpoint knows it, e.g. `qwen3:14b` or `gpt-4o`.
    pub model: String,
    /// Literal API key. Local servers usually need none.
    pub api_key: Option<String>,
    /// Name of an environment variable holding the key (preferred over a
    /// literal for cloud providers).
    pub api_key_env: Option<String>,
}

impl AiSettings {
    /// Whether enough is set to attempt a prompt. An endpoint and a model are
    /// the floor; the key is optional (local servers need none). The agent
    /// panel keys its "connect a provider" state off this.
    pub fn is_configured(&self) -> bool {
        !self.base_url.trim().is_empty() && !self.model.trim().is_empty()
    }
}

/// The `[appearance]` table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AppearanceSettings {
    /// The active bundled theme.
    pub theme: ThemeChoice,
}

/// The `[cache]` table: limits on the on-disk frame cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheSettings {
    /// Override for the cache directory; `None` uses the OS cache dir
    /// (`cutlass-ui::paths::cache_dir`). Applied at the next launch.
    pub dir: Option<PathBuf>,
    /// LRU eviction budget in MiB. Applied at the next launch.
    pub budget_mb: u64,
}

impl Default for CacheSettings {
    fn default() -> Self {
        Self {
            dir: None,
            budget_mb: DEFAULT_CACHE_BUDGET_MB,
        }
    }
}

/// The whole user config, one struct per table. [`Settings::default`] is the
/// state of a fresh install (no file on disk).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Settings {
    /// `[appearance]`.
    pub appearance: AppearanceSettings,
    /// `[ai]`.
    pub ai: AiSettings,
    /// `[cache]`.
    pub cache: CacheSettings,
}

/// `~/.cutlass/config.toml` — the user's home dir on every platform
/// (`C:\Users\<name>\.cutlass\config.toml` on Windows, where `HOME` is
/// unset). Falls back to the temp dir only if the home dir can't be resolved;
/// never the working directory, which is the read-only install folder on
/// Windows.
pub fn default_config_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join(".cutlass")
        .join("config.toml")
}

/// Load settings from `path`. A missing file yields [`Settings::default`]
/// (not an error); a malformed file is an `Err` naming the problem so the
/// caller can surface it. Unknown keys/tables are ignored, and any key we
/// don't recognize keeps its default — a partially-written file still loads.
pub fn load(path: &Path) -> Result<Settings, String> {
    let raw = match std::fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Settings::default()),
        Err(e) => return Err(format!("could not read {}: {e}", path.display())),
    };
    let doc = raw
        .parse::<DocumentMut>()
        .map_err(|e| format!("could not parse {}: {e}", path.display()))?;
    Ok(Settings::from_document(&doc))
}

/// Persist `settings` to `path`, preserving everything we don't own.
///
/// Reads the existing file (if any) into a `toml_edit` document, patches the
/// keys this crate manages, and writes it back. Comments, blank lines, key
/// order, and unknown tables survive. The parent directory is created on
/// demand. A malformed existing file is an `Err` rather than silently
/// clobbered — refusing to overwrite a file we couldn't understand.
pub fn save(path: &Path, settings: &Settings) -> Result<(), String> {
    let mut doc = match std::fs::read_to_string(path) {
        Ok(raw) => raw
            .parse::<DocumentMut>()
            .map_err(|e| format!("could not parse {}: {e}", path.display()))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => DocumentMut::new(),
        Err(e) => return Err(format!("could not read {}: {e}", path.display())),
    };

    settings.write_into(&mut doc);

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("could not create {}: {e}", parent.display()))?;
    }
    std::fs::write(path, doc.to_string())
        .map_err(|e| format!("could not write {}: {e}", path.display()))
}

impl Settings {
    fn from_document(doc: &DocumentMut) -> Self {
        let mut s = Settings::default();

        if let Some(t) = section(doc, "ai") {
            if let Some(v) = string_at(t, "base_url") {
                s.ai.base_url = v;
            }
            if let Some(v) = string_at(t, "model") {
                s.ai.model = v;
            }
            s.ai.api_key = string_at(t, "api_key");
            s.ai.api_key_env = string_at(t, "api_key_env");
        }

        if let Some(t) = section(doc, "appearance") {
            if let Some(theme) = string_at(t, "theme")
                .as_deref()
                .and_then(ThemeChoice::from_key)
            {
                s.appearance.theme = theme;
            }
        }

        if let Some(t) = section(doc, "cache") {
            if let Some(v) = int_at(t, "budget_mb") {
                s.cache.budget_mb = v.max(0) as u64;
            }
            s.cache.dir = string_at(t, "dir")
                .filter(|v| !v.trim().is_empty())
                .map(PathBuf::from);
        }

        s
    }

    fn write_into(&self, doc: &mut DocumentMut) {
        {
            let t = ensure_table(doc, "ai");
            set_str(t, "base_url", &self.ai.base_url);
            set_str(t, "model", &self.ai.model);
            set_optional(t, "api_key", self.ai.api_key.as_deref());
            set_optional(t, "api_key_env", self.ai.api_key_env.as_deref());
        }
        {
            let t = ensure_table(doc, "appearance");
            set_str(t, "theme", self.appearance.theme.key());
        }
        {
            let dir = self.cache.dir.as_ref().map(|p| p.to_string_lossy());
            let t = ensure_table(doc, "cache");
            set_int(t, "budget_mb", self.cache.budget_mb as i64);
            set_optional(t, "dir", dir.as_deref());
        }
    }
}

// --- toml_edit helpers ------------------------------------------------------

fn section<'a>(doc: &'a DocumentMut, key: &str) -> Option<&'a Table> {
    doc.get(key).and_then(Item::as_table)
}

fn string_at(table: &Table, key: &str) -> Option<String> {
    table.get(key).and_then(Item::as_str).map(str::to_owned)
}

fn int_at(table: &Table, key: &str) -> Option<i64> {
    table.get(key).and_then(Item::as_integer)
}

/// Borrow (creating if absent or if the existing item isn't a table) the
/// named top-level table. Replacing a non-table is the only way a corrupt
/// hand-edit (`ai = 3`) could otherwise wedge the writer.
fn ensure_table<'a>(doc: &'a mut DocumentMut, key: &str) -> &'a mut Table {
    if doc.get(key).and_then(Item::as_table).is_none() {
        doc.insert(key, Item::Table(Table::new()));
    }
    doc.get_mut(key)
        .and_then(Item::as_table_mut)
        .expect("table ensured above")
}

/// Write a string only when it differs from what's there. Skipping an
/// unchanged key leaves its decor (inline comments, spacing) untouched — the
/// core of the format-preserving promise.
fn set_str(table: &mut Table, key: &str, v: &str) {
    if table.get(key).and_then(Item::as_str) != Some(v) {
        table[key] = value(v);
    }
}

fn set_int(table: &mut Table, key: &str, v: i64) {
    if table.get(key).and_then(Item::as_integer) != Some(v) {
        table[key] = value(v);
    }
}

/// Set `key` to `val`, or remove it entirely when `None`, so a cleared field
/// leaves no stale literal behind.
fn set_optional(table: &mut Table, key: &str, val: Option<&str>) {
    match val {
        Some(v) => set_str(table, key, v),
        None => {
            table.remove(key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_file_is_all_defaults() {
        let s = load(Path::new("/nonexistent/cutlass/config.toml")).unwrap();
        assert_eq!(s, Settings::default());
        assert!(!s.ai.is_configured());
        assert_eq!(s.appearance.theme, ThemeChoice::DarkBlue);
        assert_eq!(s.cache.budget_mb, DEFAULT_CACHE_BUDGET_MB);
    }

    #[test]
    fn parses_each_section_and_tolerates_unknown_tables() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
[editor]
something_else = true

[ai]
base_url = "http://localhost:11434/v1"
model = "qwen3:14b"
api_key_env = "OPENAI_API_KEY"

[appearance]
theme = "ember"

[cache]
budget_mb = 1024
dir = "/tmp/cutlass-cache"
"#,
        )
        .unwrap();

        let s = load(&path).unwrap();
        assert_eq!(s.ai.base_url, "http://localhost:11434/v1");
        assert_eq!(s.ai.model, "qwen3:14b");
        assert_eq!(s.ai.api_key_env.as_deref(), Some("OPENAI_API_KEY"));
        assert!(s.ai.is_configured());
        assert_eq!(s.appearance.theme, ThemeChoice::Ember);
        assert_eq!(s.cache.budget_mb, 1024);
        assert_eq!(
            s.cache.dir.as_deref(),
            Some(Path::new("/tmp/cutlass-cache"))
        );
    }

    #[test]
    fn malformed_file_is_an_error_not_a_default() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "[ai]\nbase_url = \n").unwrap();
        assert!(load(&path).unwrap_err().contains("could not parse"));
    }

    #[test]
    fn save_round_trips_and_preserves_comments_and_unknown_tables() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            "# my cutlass config\n[ai]\nbase_url = \"http://x/v1\"  # local\nmodel = \"m\"\n\n[plugins]\nkeep = true\n",
        )
        .unwrap();

        let mut s = load(&path).unwrap();
        s.appearance.theme = ThemeChoice::Default;
        s.ai.model = "qwen3:14b".into();
        save(&path, &s).unwrap();

        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(raw.contains("# my cutlass config"), "leading comment kept");
        assert!(raw.contains("# local"), "inline comment kept");
        assert!(raw.contains("[plugins]"), "unknown table kept");
        assert!(raw.contains("keep = true"));

        let reloaded = load(&path).unwrap();
        assert_eq!(reloaded.ai.model, "qwen3:14b");
        assert_eq!(reloaded.appearance.theme, ThemeChoice::Default);
    }

    #[test]
    fn clearing_an_optional_key_removes_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");

        let mut s = Settings::default();
        s.ai.base_url = "http://x/v1".into();
        s.ai.model = "m".into();
        s.ai.api_key = Some("sk-secret".into());
        save(&path, &s).unwrap();
        assert!(std::fs::read_to_string(&path).unwrap().contains("api_key"));

        s.ai.api_key = None;
        save(&path, &s).unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(
            !raw.contains("api_key"),
            "cleared key left no literal: {raw}"
        );
        assert_eq!(load(&path).unwrap().ai.api_key, None);
    }

    #[test]
    fn save_creates_parent_dir() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("deep").join("config.toml");
        save(&path, &Settings::default()).unwrap();
        assert!(path.exists());
        assert_eq!(load(&path).unwrap(), Settings::default());
    }

    #[test]
    fn theme_key_index_round_trip() {
        for theme in ThemeChoice::ALL {
            assert_eq!(ThemeChoice::from_key(theme.key()), Some(theme));
            assert_eq!(ThemeChoice::from_index(theme.index()), theme);
        }
        assert_eq!(ThemeChoice::from_key("nonsense"), None);
        assert_eq!(ThemeChoice::from_index(99), ThemeChoice::DarkBlue);
    }

    #[test]
    fn corrupt_non_table_section_is_replaced_on_save() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "ai = 3\n").unwrap();
        // `ai = 3` parses fine; saving must overwrite it with a real table
        // rather than panic.
        let mut s = Settings::default();
        s.ai.base_url = "http://x/v1".into();
        s.ai.model = "m".into();
        save(&path, &s).unwrap();
        assert!(load(&path).unwrap().ai.is_configured());
    }
}
