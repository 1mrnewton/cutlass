//! Cutlass user settings: the typed model and IO for `~/.cutlass/config.toml`.
//!
//! This crate is the **single owner** of the user config file. Everything the
//! app persists between runs that isn't project data or the recents/autosave
//! sidecars (those live in the OS data dir, see `cutlass-desktop::paths`)
//! lives here: AI providers, the theme, cloud API endpoint, and storage
//! locations/quotas. Keys never live in project files — the `[ai]` table is
//! the historical home for the API key and stays here.
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
//! source = "local"                 # "local" | "openrouter" | "custom"
//! base_url = "http://localhost:11434/v1"   # Local / Advanced
//! model = "qwen3:14b"              # curated id, OR slug, or freeform
//! # api_key = "sk-..."             # OpenRouter / Advanced, or:
//! # api_key_env = "OPENROUTER_API_KEY"
//! # api_protocol = "responses"      # Advanced only; default chat_completions
//! # reasoning_summary = "off"       # default: "auto" in Responses mode
//! # autonomy = "full"              # skip destructive-tool confirmations
//!
//! [appearance]
//! theme = "dark-blue"              # "default" | "ember" | "dark-blue"
//!
//! [storage]
//! root = "/Volumes/Media/Cutlass"  # optional; absolute paths only
//! download_quota_mib = 2048        # default: 2 GiB
//!
//! [storage.paths]
//! proxies = "/Volumes/Scratch/Cutlass/proxies"
//! # analysis = "/Volumes/Scratch/Cutlass/analysis"
//! # ai_models = "/Volumes/Scratch/Cutlass/ai-models"
//! # download = "/Volumes/Scratch/Cutlass/download"
//! # catalog = "/Volumes/Scratch/Cutlass/catalog"
//! # luts = "/Volumes/Scratch/Cutlass/luts"
//! # lottie = "/Volumes/Scratch/Cutlass/lottie"
//! # templates = "/Volumes/Scratch/Cutlass/templates"
//!
//! # BYOK provider keys (stock, generation, TTS) — same literal-or-env
//! # pattern as [ai]. A configured key routes calls direct to the provider,
//! # bypassing the Cutlass backend entirely.
//! [providers.pexels]
//! api_key_env = "PEXELS_API_KEY"
//!
//! [providers.elevenlabs]
//! api_key = "sk-..."
//!
//! # Anonymous cloud API host (stock, catalogs, update check).
//! [cloud]
//! base_url = "https://api.cutlass.sh"     # API override; empty = default
//! ```

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use toml_edit::{DocumentMut, Item, Table, value};

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

/// How much the agent may do without asking. Read by the desktop's tool
/// host when it executes destructive (System-tier) agent tools; the
/// validated edit vocabulary is not affected (it has its own preview/undo
/// flow).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Autonomy {
    /// Destructive tools (clear caches, run scripts, overwrite files)
    /// require a per-call confirmation in the agent panel.
    #[default]
    Ask,
    /// Run everything without confirmations.
    Full,
}

impl Autonomy {
    /// The stable string written to `config.toml`.
    pub fn key(self) -> &'static str {
        match self {
            Autonomy::Ask => "ask",
            Autonomy::Full => "full",
        }
    }

    /// Parse a `config.toml` value; `None` for anything unrecognized (the
    /// caller keeps the default rather than failing the whole load).
    /// "confirm" is accepted as an alias for [`Autonomy::Ask`].
    pub fn from_key(s: &str) -> Option<Self> {
        match s {
            "ask" | "confirm" => Some(Autonomy::Ask),
            "full" => Some(Autonomy::Full),
            _ => None,
        }
    }
}

/// OpenAI-compatible HTTP protocol used by the editing agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AiApiProtocol {
    /// Broadly supported by Ollama, LM Studio, llama.cpp, and older gateways.
    #[default]
    ChatCompletions,
    /// OpenAI Responses API, required for reasoning models that call tools.
    Responses,
}

impl AiApiProtocol {
    pub fn key(self) -> &'static str {
        match self {
            Self::ChatCompletions => "chat_completions",
            Self::Responses => "responses",
        }
    }

    /// Parse a persisted value. Aliases tolerate hand-written configs while
    /// keeping one canonical key on save.
    pub fn from_key(value: &str) -> Option<Self> {
        match value {
            "chat_completions" | "chat-completions" | "chat" => Some(Self::ChatCompletions),
            "responses" | "response" => Some(Self::Responses),
            _ => None,
        }
    }
}

/// Whether Responses requests ask the provider for a safe reasoning summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReasoningSummary {
    /// Ask for the most detailed provider-supported summary.
    #[default]
    Auto,
    /// Do not request or display reasoning summaries.
    Off,
}

impl ReasoningSummary {
    pub fn key(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Off => "off",
        }
    }

    pub fn from_key(value: &str) -> Option<Self> {
        match value {
            "auto" | "on" => Some(Self::Auto),
            "off" | "none" => Some(Self::Off),
            _ => None,
        }
    }
}

/// Which AI path Settings / the agent should use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AiSource {
    /// Local OpenAI-compatible server (Ollama, LM Studio). Base URL + curated model.
    #[default]
    Local,
    /// OpenRouter cloud gateway. One key + curated OR slug.
    OpenRouter,
    /// Freeform OpenAI-compatible endpoint (escape hatch).
    Custom,
}

impl AiSource {
    pub fn key(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::OpenRouter => "openrouter",
            Self::Custom => "custom",
        }
    }

    pub fn from_key(value: &str) -> Option<Self> {
        match value {
            "local" => Some(Self::Local),
            "openrouter" | "or" => Some(Self::OpenRouter),
            "custom" | "advanced" | "endpoint" => Some(Self::Custom),
            _ => None,
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
    /// Local / OpenRouter / freeform Advanced.
    pub source: AiSource,
    /// OpenAI-compatible endpoint root. Used by Local and Custom; ignored for
    /// OpenRouter (fixed gateway URL in `cutlass-ai`).
    pub base_url: String,
    /// Model id: curated local / OR slug / freeform for Custom.
    pub model: String,
    /// HTTP API shape. Local and OpenRouter always use Chat Completions;
    /// Custom may select Responses.
    pub api_protocol: AiApiProtocol,
    /// Provider-generated summary visibility for Responses reasoning models.
    pub reasoning_summary: ReasoningSummary,
    /// Literal API key. Local servers usually need none; OpenRouter requires one.
    pub api_key: Option<String>,
    /// Name of an environment variable holding the key (preferred over a
    /// literal for cloud providers).
    pub api_key_env: Option<String>,
    /// Confirmation policy for destructive agent tools. Orthogonal to
    /// [`is_configured`](Self::is_configured) — it gates tool *execution*,
    /// not provider reachability.
    pub autonomy: Autonomy,
}

impl AiSettings {
    /// Whether enough is set to attempt a prompt for the active [`AiSource`].
    /// Allowlist membership is enforced by the desktop / `cutlass-ai` catalog
    /// at save and prompt time; this only checks structural readiness.
    pub fn is_configured(&self) -> bool {
        let model = !self.model.trim().is_empty();
        match self.source {
            AiSource::Local => !self.base_url.trim().is_empty() && model,
            AiSource::OpenRouter => model && self.has_api_key(),
            AiSource::Custom => !self.base_url.trim().is_empty() && model,
        }
    }

    /// Literal or env-named key present (not resolved).
    pub fn has_api_key(&self) -> bool {
        self.api_key
            .as_deref()
            .is_some_and(|k| !k.trim().is_empty())
            || self
                .api_key_env
                .as_deref()
                .is_some_and(|k| !k.trim().is_empty())
    }
}

/// The `[appearance]` table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AppearanceSettings {
    /// The active bundled theme.
    pub theme: ThemeChoice,
}

/// One `[providers.<name>]` entry: a BYOK key for a third-party service
/// (stock search, image/video generation, TTS). Same literal-or-env shape
/// as [`AiSettings`]; key *resolution* stays with the caller (the env
/// indirection is a use-site concern). A configured provider routes calls
/// direct — the Cutlass backend never sees the key.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProviderSettings {
    /// Literal API key.
    pub api_key: Option<String>,
    /// Name of an environment variable holding the key (preferred).
    pub api_key_env: Option<String>,
}

impl ProviderSettings {
    /// Whether either key form is present (the BYOK routing predicate).
    pub fn is_configured(&self) -> bool {
        self.api_key
            .as_deref()
            .is_some_and(|k| !k.trim().is_empty())
            || self
                .api_key_env
                .as_deref()
                .is_some_and(|k| !k.trim().is_empty())
    }

    /// Resolve the key: literal wins, else the named environment variable.
    pub fn resolve_key(&self) -> Option<String> {
        if let Some(key) = self.api_key.as_deref().filter(|k| !k.trim().is_empty()) {
            return Some(key.to_string());
        }
        self.api_key_env
            .as_deref()
            .filter(|name| !name.trim().is_empty())
            .and_then(|name| std::env::var(name).ok())
            .filter(|v| !v.is_empty())
    }
}

/// The `[cloud]` table: anonymous API host override for stock, catalogs,
/// and the update check. Empty `base_url` means the shipped default.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CloudSettings {
    /// Backend (API) base URL override; empty = the shipped default.
    pub base_url: String,
}

/// Default quota for downloaded, re-fetchable assets: 2048 MiB (2 GiB).
pub const DEFAULT_DOWNLOAD_QUOTA_MIB: u64 = 2_048;

/// Smallest accepted download-cache quota.
pub const MIN_DOWNLOAD_QUOTA_MIB: u64 = 1;

/// Largest accepted download-cache quota: 1 TiB.
///
/// Keeping a finite upper bound makes conversion to bytes and downstream
/// accounting predictable even when the config file was hand-edited.
pub const MAX_DOWNLOAD_QUOTA_MIB: u64 = 1_048_576;

/// Known per-cache overrides in `[storage.paths]`.
///
/// Every populated path is absolute. Loading ignores empty, relative, and
/// wrongly-typed values rather than rejecting the rest of the config.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StoragePathOverrides {
    /// Generated media proxies.
    pub proxies: Option<PathBuf>,
    /// Regenerable media-analysis state.
    pub analysis: Option<PathBuf>,
    /// Downloaded AI model weights for transcription, vision, and embeddings.
    pub ai_models: Option<PathBuf>,
    /// Downloaded stock and generated assets.
    pub download: Option<PathBuf>,
    /// Cached asset-catalog responses and metadata.
    pub catalog: Option<PathBuf>,
    /// Downloaded LUT packs.
    pub luts: Option<PathBuf>,
    /// Downloaded Lottie assets.
    pub lottie: Option<PathBuf>,
    /// Downloaded and installed template bundles.
    pub templates: Option<PathBuf>,
}

impl StoragePathOverrides {
    /// Return a known override by its exact TOML key.
    ///
    /// Unknown keys return `None`; they are still preserved in the TOML
    /// document when [`save`] patches a file.
    pub fn get(&self, key: &str) -> Option<&Path> {
        match key {
            "proxies" => self.proxies.as_deref(),
            "analysis" => self.analysis.as_deref(),
            "ai_models" => self.ai_models.as_deref(),
            "download" => self.download.as_deref(),
            "catalog" => self.catalog.as_deref(),
            "luts" => self.luts.as_deref(),
            "lottie" => self.lottie.as_deref(),
            "templates" => self.templates.as_deref(),
            _ => None,
        }
    }
}

/// The `[storage]` table: optional storage roots and the download-cache quota.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageSettings {
    /// Optional absolute root for Cutlass-managed storage.
    pub root: Option<PathBuf>,
    /// Download-cache quota in MiB.
    ///
    /// The default is [`DEFAULT_DOWNLOAD_QUOTA_MIB`] (2048 MiB); accepted
    /// values are [`MIN_DOWNLOAD_QUOTA_MIB`] through
    /// [`MAX_DOWNLOAD_QUOTA_MIB`], inclusive.
    pub download_quota_mib: u64,
    /// Absolute per-cache overrides from `[storage.paths]`.
    pub paths: StoragePathOverrides,
}

impl StorageSettings {
    /// Whether `value` is safe to use as a download-cache quota.
    pub fn is_valid_download_quota_mib(value: u64) -> bool {
        (MIN_DOWNLOAD_QUOTA_MIB..=MAX_DOWNLOAD_QUOTA_MIB).contains(&value)
    }
}

impl Default for StorageSettings {
    fn default() -> Self {
        Self {
            root: None,
            download_quota_mib: DEFAULT_DOWNLOAD_QUOTA_MIB,
            paths: StoragePathOverrides::default(),
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
    /// `[providers.<name>]` — BYOK keys by provider name ("pexels",
    /// "pixabay", "elevenlabs", …). Sorted map so saves are deterministic.
    pub providers: BTreeMap<String, ProviderSettings>,
    /// `[cloud]`.
    pub cloud: CloudSettings,
    /// `[storage]`.
    pub storage: StorageSettings,
}

impl Settings {
    /// The named provider's settings, defaulting to unconfigured.
    pub fn provider(&self, name: &str) -> ProviderSettings {
        self.providers.get(name).cloned().unwrap_or_default()
    }
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

/// `~/.cutlass/agent/` — the user's AI-assistant extension dir
/// (`rules/*.md`, `skills/<id>/SKILL.md`, `commands/*.md`), reloaded by
/// the desktop before every prompt so edits apply without a restart.
pub fn agent_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join(".cutlass")
        .join("agent")
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
        Ok(raw) => raw.parse::<DocumentMut>().map_err(|_| {
            format!(
                "could not parse {}: the existing configuration is malformed",
                path.display()
            )
        })?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => DocumentMut::new(),
        Err(e) => return Err(format!("could not read {}: {e}", path.display())),
    };

    settings.write_into(&mut doc)?;

    // Materialize and validate the complete output before creating directories,
    // temporary files, or otherwise changing filesystem state.
    let serialized = doc.to_string();
    let _: DocumentMut = serialized.parse().map_err(|_| {
        format!(
            "could not validate generated configuration for {}",
            path.display()
        )
    })?;

    persist_serialized(path, serialized.as_bytes())
}

const UNIQUE_PATH_ATTEMPTS: usize = 128;
static UNIQUE_PATH_COUNTER: AtomicU64 = AtomicU64::new(0);

trait PersistenceFs {
    fn rename(&self, from: &Path, to: &Path) -> io::Result<()>;
    fn remove_file(&self, path: &Path) -> io::Result<()>;
    fn symlink_metadata(&self, path: &Path) -> io::Result<std::fs::Metadata>;
}

struct StdPersistenceFs;

impl PersistenceFs for StdPersistenceFs {
    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        std::fs::rename(from, to)
    }

    fn remove_file(&self, path: &Path) -> io::Result<()> {
        std::fs::remove_file(path)
    }

    fn symlink_metadata(&self, path: &Path) -> io::Result<std::fs::Metadata> {
        std::fs::symlink_metadata(path)
    }
}

fn persist_serialized(destination: &Path, contents: &[u8]) -> Result<(), String> {
    let parent = destination_parent(destination)?;
    std::fs::create_dir_all(parent)
        .map_err(|e| format!("could not create {}: {e}", parent.display()))?;

    let permissions = match std::fs::metadata(destination) {
        Ok(metadata) => Some(metadata.permissions()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => None,
        Err(e) => {
            return Err(format!(
                "could not read permissions for {}: {e}",
                destination.display()
            ));
        }
    };

    let temporary = write_synced_temp(destination, contents, permissions)?;
    install_temp_with_ops(destination, &temporary, &StdPersistenceFs)
}

fn write_synced_temp(
    destination: &Path,
    contents: &[u8],
    permissions: Option<std::fs::Permissions>,
) -> Result<PathBuf, String> {
    let (temporary, mut file) = create_unique_temp(destination)?;

    if let Err(e) = file.write_all(contents) {
        drop(file);
        return Err(cleanup_temp_after_error(
            &StdPersistenceFs,
            &temporary,
            format!(
                "could not write temporary configuration for {}: {e}",
                destination.display()
            ),
        ));
    }

    if let Some(permissions) = permissions
        && let Err(e) = file.set_permissions(permissions)
    {
        drop(file);
        return Err(cleanup_temp_after_error(
            &StdPersistenceFs,
            &temporary,
            format!(
                "could not preserve permissions for {}: {e}",
                destination.display()
            ),
        ));
    }

    if let Err(e) = file.sync_all() {
        drop(file);
        return Err(cleanup_temp_after_error(
            &StdPersistenceFs,
            &temporary,
            format!(
                "could not sync temporary configuration for {}: {e}",
                destination.display()
            ),
        ));
    }
    drop(file);

    Ok(temporary)
}

fn create_unique_temp(destination: &Path) -> Result<(PathBuf, std::fs::File), String> {
    for _ in 0..UNIQUE_PATH_ATTEMPTS {
        let candidate = unique_sibling_path(destination, "tmp")?;
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        match options.open(&candidate) {
            Ok(file) => return Ok((candidate, file)),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
            Err(e) => {
                return Err(format!(
                    "could not create a temporary configuration beside {}: {e}",
                    destination.display()
                ));
            }
        }
    }

    Err(format!(
        "could not allocate a unique temporary configuration beside {}",
        destination.display()
    ))
}

fn install_temp_with_ops(
    destination: &Path,
    temporary: &Path,
    fs: &impl PersistenceFs,
) -> Result<(), String> {
    let atomic_error = match fs.rename(temporary, destination) {
        Ok(()) => return Ok(()),
        Err(e) => e,
    };

    match fs.symlink_metadata(destination) {
        Ok(_) => {}
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return Err(cleanup_temp_after_error(
                fs,
                temporary,
                format!(
                    "could not install configuration at {} with an atomic rename: {atomic_error}; \
                     the destination was absent, so no existing configuration required recovery",
                    destination.display()
                ),
            ));
        }
        Err(e) => {
            return Err(cleanup_temp_after_error(
                fs,
                temporary,
                format!(
                    "could not install configuration at {} with an atomic rename: {atomic_error}; \
                     could not inspect the existing destination for recovery: {e}",
                    destination.display()
                ),
            ));
        }
    }

    let backup = match vacant_backup_path(destination, fs) {
        Ok(path) => path,
        Err(e) => {
            return Err(cleanup_temp_after_error(
                fs,
                temporary,
                format!(
                    "could not replace configuration at {} after atomic rename failed \
                     ({atomic_error}): {e}; the existing configuration was left in place",
                    destination.display()
                ),
            ));
        }
    };

    if let Err(backup_error) = fs.rename(destination, &backup) {
        return Err(cleanup_temp_after_error(
            fs,
            temporary,
            format!(
                "could not replace configuration at {}: atomic rename failed ({atomic_error}); \
                 moving the existing configuration to a backup also failed: {backup_error}; \
                 the existing configuration was left in place",
                destination.display()
            ),
        ));
    }

    match fs.rename(temporary, destination) {
        Ok(()) => {
            // Installation is the commit point. A stale backup is preferable
            // to reporting failure after callers have already persisted the
            // new state; relocation transactions use `save` as their commit
            // callback and would otherwise roll data back out from under the
            // newly installed configuration.
            let _ = fs.remove_file(&backup);
            Ok(())
        }
        Err(install_error) => match fs.rename(&backup, destination) {
            Ok(()) => Err(cleanup_temp_after_error(
                fs,
                temporary,
                format!(
                    "could not install the new configuration at {} after backing up the existing \
                     file: {install_error}; the original configuration was restored",
                    destination.display()
                ),
            )),
            Err(rollback_error) => Err(cleanup_temp_after_error(
                fs,
                temporary,
                format!(
                    "could not install the new configuration at {} after backing up the existing \
                     file: {install_error}; rollback failed: {rollback_error}; the destination may \
                     be missing, and the original configuration backup was retained at {}",
                    destination.display(),
                    backup.display()
                ),
            )),
        },
    }
}

fn vacant_backup_path(destination: &Path, fs: &impl PersistenceFs) -> Result<PathBuf, String> {
    for _ in 0..UNIQUE_PATH_ATTEMPTS {
        let candidate = unique_sibling_path(destination, "backup")?;
        match fs.symlink_metadata(&candidate) {
            Ok(_) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(candidate),
            Err(e) => {
                return Err(format!(
                    "could not inspect a backup path beside {}: {e}",
                    destination.display()
                ));
            }
        }
    }

    Err(format!(
        "could not allocate a unique backup beside {}",
        destination.display()
    ))
}

fn unique_sibling_path(destination: &Path, role: &str) -> Result<PathBuf, String> {
    let parent = destination_parent(destination)?;
    let file_name = destination.file_name().ok_or_else(|| {
        format!(
            "configuration path {} has no file name",
            destination.display()
        )
    })?;
    let nonce = UNIQUE_PATH_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut name = OsString::from(".");
    name.push(file_name);
    name.push(format!(".cutlass-{role}-{}-{nonce}", std::process::id()));
    Ok(parent.join(name))
}

fn destination_parent(destination: &Path) -> Result<&Path, String> {
    let parent = destination.parent().ok_or_else(|| {
        format!(
            "configuration path {} has no parent directory",
            destination.display()
        )
    })?;
    if parent.as_os_str().is_empty() {
        Ok(Path::new("."))
    } else {
        Ok(parent)
    }
}

fn cleanup_temp_after_error(
    fs: &impl PersistenceFs,
    temporary: &Path,
    primary_error: String,
) -> String {
    match fs.remove_file(temporary) {
        Ok(()) => primary_error,
        Err(e) if e.kind() == io::ErrorKind::NotFound => primary_error,
        Err(e) => format!(
            "{primary_error}; temporary-file cleanup also failed for {}: {e}; \
             the temporary file may remain",
            temporary.display()
        ),
    }
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
            if let Some(protocol) = string_at(t, "api_protocol")
                .as_deref()
                .and_then(AiApiProtocol::from_key)
            {
                s.ai.api_protocol = protocol;
            }
            if let Some(summary) = string_at(t, "reasoning_summary")
                .as_deref()
                .and_then(ReasoningSummary::from_key)
            {
                s.ai.reasoning_summary = summary;
            }
            s.ai.api_key = string_at(t, "api_key");
            s.ai.api_key_env = string_at(t, "api_key_env");
            // Legacy `use_account` is ignored (Cutlass account removed).
            if let Some(autonomy) = string_at(t, "autonomy")
                .as_deref()
                .and_then(Autonomy::from_key)
            {
                s.ai.autonomy = autonomy;
            }
            // Explicit `source` wins. Otherwise migrate: a pre-source config
            // with URL+model was freeform Advanced; empty stays Local default.
            if let Some(source) = string_at(t, "source")
                .as_deref()
                .and_then(AiSource::from_key)
            {
                s.ai.source = source;
            } else if !s.ai.base_url.trim().is_empty() && !s.ai.model.trim().is_empty() {
                s.ai.source = AiSource::Custom;
            }
        }

        if let Some(t) = section(doc, "appearance")
            && let Some(theme) = string_at(t, "theme")
                .as_deref()
                .and_then(ThemeChoice::from_key)
        {
            s.appearance.theme = theme;
        }

        if let Some(t) = section(doc, "storage") {
            s.storage.root = absolute_path_at(t, "root");
            if let Some(quota) = t
                .get("download_quota_mib")
                .and_then(Item::as_integer)
                .and_then(|quota| u64::try_from(quota).ok())
                .filter(|quota| StorageSettings::is_valid_download_quota_mib(*quota))
            {
                s.storage.download_quota_mib = quota;
            }

            if let Some(paths) = t.get("paths").and_then(Item::as_table) {
                s.storage.paths.proxies = absolute_path_at(paths, "proxies");
                s.storage.paths.analysis = absolute_path_at(paths, "analysis");
                s.storage.paths.ai_models = absolute_path_at(paths, "ai_models");
                s.storage.paths.download = absolute_path_at(paths, "download");
                s.storage.paths.catalog = absolute_path_at(paths, "catalog");
                s.storage.paths.luts = absolute_path_at(paths, "luts");
                s.storage.paths.lottie = absolute_path_at(paths, "lottie");
                s.storage.paths.templates = absolute_path_at(paths, "templates");
            }
        }

        if let Some(t) = section(doc, "providers") {
            for (name, item) in t.iter() {
                if let Some(entry) = item.as_table() {
                    s.providers.insert(
                        name.to_string(),
                        ProviderSettings {
                            api_key: string_at(entry, "api_key"),
                            api_key_env: string_at(entry, "api_key_env"),
                        },
                    );
                }
            }
        }

        // Prefer `[cloud]`; fall back to legacy `[account].base_url`.
        if let Some(t) = section(doc, "cloud") {
            if let Some(v) = string_at(t, "base_url") {
                s.cloud.base_url = v;
            }
        } else if let Some(t) = section(doc, "account")
            && let Some(v) = string_at(t, "base_url")
        {
            s.cloud.base_url = v;
        }

        s
    }

    fn write_into(&self, doc: &mut DocumentMut) -> Result<(), String> {
        if !StorageSettings::is_valid_download_quota_mib(self.storage.download_quota_mib) {
            return Err(format!(
                "storage.download_quota_mib must be between \
                 {MIN_DOWNLOAD_QUOTA_MIB} and {MAX_DOWNLOAD_QUOTA_MIB} MiB"
            ));
        }
        let storage_root = storage_path_for_save(self.storage.root.as_deref(), "root")?;
        let storage_paths = [
            (
                "proxies",
                storage_path_for_save(self.storage.paths.proxies.as_deref(), "paths.proxies")?,
            ),
            (
                "analysis",
                storage_path_for_save(self.storage.paths.analysis.as_deref(), "paths.analysis")?,
            ),
            (
                "ai_models",
                storage_path_for_save(self.storage.paths.ai_models.as_deref(), "paths.ai_models")?,
            ),
            (
                "download",
                storage_path_for_save(self.storage.paths.download.as_deref(), "paths.download")?,
            ),
            (
                "catalog",
                storage_path_for_save(self.storage.paths.catalog.as_deref(), "paths.catalog")?,
            ),
            (
                "luts",
                storage_path_for_save(self.storage.paths.luts.as_deref(), "paths.luts")?,
            ),
            (
                "lottie",
                storage_path_for_save(self.storage.paths.lottie.as_deref(), "paths.lottie")?,
            ),
            (
                "templates",
                storage_path_for_save(self.storage.paths.templates.as_deref(), "paths.templates")?,
            ),
        ];

        {
            let t = ensure_table(doc, "ai");
            if self.ai.source == AiSource::default() {
                t.remove("source");
            } else {
                set_str(t, "source", self.ai.source.key());
            }
            set_str(t, "base_url", &self.ai.base_url);
            set_str(t, "model", &self.ai.model);
            if self.ai.api_protocol == AiApiProtocol::default() {
                t.remove("api_protocol");
            } else {
                set_str(t, "api_protocol", self.ai.api_protocol.key());
            }
            if self.ai.reasoning_summary == ReasoningSummary::default() {
                t.remove("reasoning_summary");
            } else {
                set_str(t, "reasoning_summary", self.ai.reasoning_summary.key());
            }
            set_optional(t, "api_key", self.ai.api_key.as_deref());
            set_optional(t, "api_key_env", self.ai.api_key_env.as_deref());
            // Drop legacy managed-account key if present.
            t.remove("use_account");
            // Default autonomy is absence, so a fresh config stays minimal.
            if self.ai.autonomy == Autonomy::default() {
                t.remove("autonomy");
            } else {
                set_str(t, "autonomy", self.ai.autonomy.key());
            }
        }
        {
            let t = ensure_table(doc, "appearance");
            set_str(t, "theme", self.appearance.theme.key());
        }
        {
            let has_path_overrides = storage_paths.iter().any(|(_, value)| value.is_some());
            let has_storage_values = storage_root.is_some()
                || self.storage.download_quota_mib != DEFAULT_DOWNLOAD_QUOTA_MIB
                || has_path_overrides;

            if has_storage_values {
                ensure_table(doc, "storage");
            }
            if let Some(t) = doc.get_mut("storage").and_then(Item::as_table_mut) {
                set_optional(t, "root", storage_root);
                if self.storage.download_quota_mib == DEFAULT_DOWNLOAD_QUOTA_MIB {
                    t.remove("download_quota_mib");
                } else {
                    set_integer(
                        t,
                        "download_quota_mib",
                        self.storage.download_quota_mib as i64,
                    );
                }

                if has_path_overrides {
                    ensure_child_table(t, "paths");
                }
                let remove_paths =
                    if let Some(paths) = t.get_mut("paths").and_then(Item::as_table_mut) {
                        for (key, value) in storage_paths {
                            set_optional(paths, key, value);
                        }
                        paths.is_empty()
                    } else {
                        false
                    };
                if remove_paths {
                    t.remove("paths");
                }
            }

            let remove_storage = doc
                .get("storage")
                .and_then(Item::as_table)
                .is_some_and(Table::is_empty);
            if remove_storage {
                doc.remove("storage");
            }
        }
        {
            // Only write providers we hold; hand-added entries under
            // `[providers.*]` that we loaded are re-written unchanged, and
            // ones we never parsed (non-table junk) are left alone.
            for (name, provider) in &self.providers {
                let t = ensure_table(doc, "providers");
                if t.get(name).and_then(Item::as_table).is_none() {
                    t.insert(name, Item::Table(Table::new()));
                }
                let entry = t
                    .get_mut(name)
                    .and_then(Item::as_table_mut)
                    .expect("provider table ensured above");
                set_optional(entry, "api_key", provider.api_key.as_deref());
                set_optional(entry, "api_key_env", provider.api_key_env.as_deref());
            }
            // Dropping a provider from the map removes its table.
            if let Some(t) = doc.get_mut("providers").and_then(Item::as_table_mut) {
                let stale: Vec<String> = t
                    .iter()
                    .filter(|(name, item)| {
                        item.as_table().is_some() && !self.providers.contains_key(*name)
                    })
                    .map(|(name, _)| name.to_string())
                    .collect();
                for name in stale {
                    t.remove(&name);
                }
                if t.is_empty() {
                    doc.remove("providers");
                }
            }
        }
        {
            // Empty override removes the key (and a now-empty table).
            // Migrate off legacy `[account]` when writing.
            doc.remove("account");
            if self.cloud.base_url.is_empty() {
                if let Some(t) = doc.get_mut("cloud").and_then(Item::as_table_mut) {
                    t.remove("base_url");
                }
            } else {
                let t = ensure_table(doc, "cloud");
                set_str(t, "base_url", &self.cloud.base_url);
            }
            if let Some(t) = doc.get_mut("cloud").and_then(Item::as_table_mut)
                && t.is_empty()
            {
                doc.remove("cloud");
            }
        }
        Ok(())
    }
}

// --- toml_edit helpers ------------------------------------------------------

fn section<'a>(doc: &'a DocumentMut, key: &str) -> Option<&'a Table> {
    doc.get(key).and_then(Item::as_table)
}

fn string_at(table: &Table, key: &str) -> Option<String> {
    table.get(key).and_then(Item::as_str).map(str::to_owned)
}

fn absolute_path_at(table: &Table, key: &str) -> Option<PathBuf> {
    let raw = table.get(key).and_then(Item::as_str)?;
    if raw.trim().is_empty() {
        return None;
    }
    let path = PathBuf::from(raw);
    path.is_absolute().then_some(path)
}

fn storage_path_for_save<'a>(path: Option<&'a Path>, key: &str) -> Result<Option<&'a str>, String> {
    let Some(path) = path else {
        return Ok(None);
    };
    if !path.is_absolute() {
        return Err(format!("storage.{key} must be an absolute path"));
    }
    path.to_str()
        .map(Some)
        .ok_or_else(|| format!("storage.{key} is not valid UTF-8 and cannot be written to TOML"))
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

/// Borrow a real child table, replacing a wrongly-typed item only when a
/// caller has a value that must be written there.
fn ensure_child_table<'a>(table: &'a mut Table, key: &str) -> &'a mut Table {
    if table.get(key).and_then(Item::as_table).is_none() {
        table.insert(key, Item::Table(Table::new()));
    }
    table
        .get_mut(key)
        .and_then(Item::as_table_mut)
        .expect("child table ensured above")
}

/// Write a string only when it differs from what's there. Skipping an
/// unchanged key leaves its decor (inline comments, spacing) untouched — the
/// core of the format-preserving promise.
fn set_str(table: &mut Table, key: &str, v: &str) {
    if table.get(key).and_then(Item::as_str) != Some(v) {
        table[key] = value(v);
    }
}

fn set_integer(table: &mut Table, key: &str, v: i64) {
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
mod tests;
