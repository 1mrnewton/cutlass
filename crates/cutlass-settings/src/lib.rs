//! Cutlass user settings: the typed model and IO for `~/.cutlass/config.toml`.
//!
//! This crate is the **single owner** of the user config file. Everything the
//! app persists between runs that isn't project data or the recents/autosave
//! sidecars (those live in the OS data dir, see `cutlass-desktop::paths`)
//! lives here: AI providers, the theme, account endpoints, and storage
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
//! base_url = "http://localhost:11434/v1"   # Ollama
//! model = "qwen3:14b"
//! # api_key = "sk-..."             # literal key, or:
//! # api_key_env = "OPENAI_API_KEY"  # read from the environment
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
//! # Cutlass account plumbing. The session token itself is NEVER here —
//! # it lives in the OS keychain (see cutlass-cloud's token store).
//! [account]
//! base_url = "https://api.cutlass.sh"     # API override; empty = default
//! auth_base_url = "https://cutlass.sh"    # website/auth override; empty = default
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
    /// Route the assistant through the Cutlass account (managed chat
    /// proxy, credits-metered) instead of the endpoint above. The three
    /// provider modes: local/BYOK endpoint (fields above), or this.
    pub use_account: bool,
    /// Confirmation policy for destructive agent tools. Orthogonal to
    /// [`is_configured`](Self::is_configured) — it gates tool *execution*,
    /// not provider reachability.
    pub autonomy: Autonomy,
}

impl AiSettings {
    /// Whether enough is set to attempt a prompt. An endpoint and a model are
    /// the floor; the key is optional (local servers need none). The agent
    /// panel keys its "connect a provider" state off this.
    pub fn is_configured(&self) -> bool {
        self.use_account || (!self.base_url.trim().is_empty() && !self.model.trim().is_empty())
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

/// The `[account]` table. Deliberately tiny: the base-URL overrides are
/// the only account state that belongs in a plain file — the session
/// token lives in the OS keychain, never here.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AccountSettings {
    /// Backend (API) base URL override; empty = the shipped default.
    pub base_url: String,
    /// Auth base URL override — the website hosting better-auth
    /// (`/api/auth/*`, `/device`, `/account`); empty = the shipped
    /// default.
    pub auth_base_url: String,
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
    /// `[account]`.
    pub account: AccountSettings,
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

    if let Some(permissions) = permissions {
        if let Err(e) = file.set_permissions(permissions) {
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
            s.ai.api_key = string_at(t, "api_key");
            s.ai.api_key_env = string_at(t, "api_key_env");
            s.ai.use_account = t
                .get("use_account")
                .and_then(Item::as_bool)
                .unwrap_or(false);
            if let Some(autonomy) = string_at(t, "autonomy")
                .as_deref()
                .and_then(Autonomy::from_key)
            {
                s.ai.autonomy = autonomy;
            }
        }

        if let Some(t) = section(doc, "appearance") {
            if let Some(theme) = string_at(t, "theme")
                .as_deref()
                .and_then(ThemeChoice::from_key)
            {
                s.appearance.theme = theme;
            }
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

        if let Some(t) = section(doc, "account") {
            if let Some(v) = string_at(t, "base_url") {
                s.account.base_url = v;
            }
            if let Some(v) = string_at(t, "auth_base_url") {
                s.account.auth_base_url = v;
            }
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
            set_str(t, "base_url", &self.ai.base_url);
            set_str(t, "model", &self.ai.model);
            set_optional(t, "api_key", self.ai.api_key.as_deref());
            set_optional(t, "api_key_env", self.ai.api_key_env.as_deref());
            if self.ai.use_account {
                t.insert("use_account", toml_edit::value(true));
            } else {
                t.remove("use_account");
            }
            // Same convention as `use_account`: the default is absence, so a
            // fresh config stays minimal.
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
            // Empty overrides remove their keys (and a now-empty table),
            // so a fresh config stays minimal.
            for (key, value) in [
                ("base_url", &self.account.base_url),
                ("auth_base_url", &self.account.auth_base_url),
            ] {
                if value.is_empty() {
                    if let Some(t) = doc.get_mut("account").and_then(Item::as_table_mut) {
                        t.remove(key);
                    }
                } else {
                    let t = ensure_table(doc, "account");
                    set_str(t, key, value);
                }
            }
            if let Some(t) = doc.get_mut("account").and_then(Item::as_table_mut) {
                if t.is_empty() {
                    doc.remove("account");
                }
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
mod tests {
    use std::cell::Cell;

    use super::*;

    #[derive(Default)]
    struct FaultFs {
        rename_calls: Cell<usize>,
        remove_calls: Cell<usize>,
        failed_renames: Vec<usize>,
        failed_removals: Vec<usize>,
    }

    impl FaultFs {
        fn failing(failed_renames: &[usize], failed_removals: &[usize]) -> Self {
            Self {
                failed_renames: failed_renames.to_vec(),
                failed_removals: failed_removals.to_vec(),
                ..Self::default()
            }
        }

        fn next_call(counter: &Cell<usize>) -> usize {
            let call = counter.get() + 1;
            counter.set(call);
            call
        }
    }

    impl PersistenceFs for FaultFs {
        fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
            let call = Self::next_call(&self.rename_calls);
            if self.failed_renames.contains(&call) {
                return Err(io::Error::other(format!("injected rename failure #{call}")));
            }
            std::fs::rename(from, to)
        }

        fn remove_file(&self, path: &Path) -> io::Result<()> {
            let call = Self::next_call(&self.remove_calls);
            if self.failed_removals.contains(&call) {
                return Err(io::Error::other(format!(
                    "injected removal failure #{call}"
                )));
            }
            std::fs::remove_file(path)
        }

        fn symlink_metadata(&self, path: &Path) -> io::Result<std::fs::Metadata> {
            std::fs::symlink_metadata(path)
        }
    }

    fn transaction_artifacts(directory: &Path) -> Vec<PathBuf> {
        let mut artifacts: Vec<_> = std::fs::read_dir(directory)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.file_name().is_some_and(|name| {
                    let name = name.to_string_lossy();
                    name.contains(".cutlass-tmp-") || name.contains(".cutlass-backup-")
                })
            })
            .collect();
        artifacts.sort();
        artifacts
    }

    fn assert_no_transaction_artifacts(directory: &Path) {
        let artifacts = transaction_artifacts(directory);
        assert!(
            artifacts.is_empty(),
            "unexpected transaction artifacts: {artifacts:?}"
        );
    }

    #[test]
    fn missing_file_is_all_defaults() {
        let s = load(Path::new("/nonexistent/cutlass/config.toml")).unwrap();
        assert_eq!(s, Settings::default());
        assert!(!s.ai.is_configured());
        assert_eq!(s.appearance.theme, ThemeChoice::DarkBlue);
        assert_eq!(s.storage, StorageSettings::default());
        assert_eq!(s.storage.download_quota_mib, DEFAULT_DOWNLOAD_QUOTA_MIB);
    }

    #[test]
    fn storage_valid_values_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let mut s = Settings::default();
        s.storage.root = Some(dir.path().join("storage"));
        s.storage.download_quota_mib = 4_096;
        s.storage.paths = StoragePathOverrides {
            proxies: Some(dir.path().join("proxies")),
            download: Some(dir.path().join("download")),
            catalog: Some(dir.path().join("catalog")),
            luts: Some(dir.path().join("luts")),
            lottie: Some(dir.path().join("lottie")),
            templates: Some(dir.path().join("templates")),
        };

        save(&path, &s).unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(raw.contains("[storage]"), "{raw}");
        assert!(raw.contains("download_quota_mib = 4096"), "{raw}");
        assert!(raw.contains("[storage.paths]"), "{raw}");
        for key in [
            "proxies",
            "download",
            "catalog",
            "luts",
            "lottie",
            "templates",
        ] {
            assert!(raw.contains(&format!("{key} = ")), "missing {key}: {raw}");
        }

        let loaded = load(&path).unwrap();
        assert_eq!(loaded.storage, s.storage);
        for key in [
            "proxies",
            "download",
            "catalog",
            "luts",
            "lottie",
            "templates",
        ] {
            assert!(loaded.storage.paths.get(key).is_some(), "{key}");
        }
        assert_eq!(loaded.storage.paths.get("future"), None);
    }

    #[test]
    fn storage_paths_must_be_absolute() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let absolute_download = dir.path().join("download");
        std::fs::write(
            &path,
            format!(
                "[storage]\n\
                 root = \"relative/root\"\n\
                 [storage.paths]\n\
                 proxies = \"relative/proxies\"\n\
                 download = {:?}\n\
                 catalog = 3\n\
                 luts = \"\"\n",
                absolute_download.to_str().unwrap()
            ),
        )
        .unwrap();

        let mut s = load(&path).unwrap();
        assert_eq!(s.storage.root, None);
        assert_eq!(s.storage.paths.proxies, None);
        assert_eq!(
            s.storage.paths.download.as_deref(),
            Some(absolute_download.as_path())
        );
        assert_eq!(s.storage.paths.catalog, None);
        assert_eq!(s.storage.paths.luts, None);

        let original = std::fs::read_to_string(&path).unwrap();
        s.storage.root = Some(PathBuf::from("relative/root"));
        let error = save(&path, &s).unwrap_err();
        assert!(error.contains("absolute path"), "{error}");
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            original,
            "failed validation must not rewrite the file"
        );
        assert_no_transaction_artifacts(dir.path());
    }

    #[test]
    fn invalid_storage_quotas_fall_back_to_default() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        for value in [
            "0".to_string(),
            "-1".to_string(),
            (MAX_DOWNLOAD_QUOTA_MIB + 1).to_string(),
            "\"2048\"".to_string(),
            "3.5".to_string(),
        ] {
            std::fs::write(&path, format!("[storage]\ndownload_quota_mib = {value}\n")).unwrap();
            assert_eq!(
                load(&path).unwrap().storage.download_quota_mib,
                DEFAULT_DOWNLOAD_QUOTA_MIB,
                "value {value} should fall back"
            );
        }

        for value in [MIN_DOWNLOAD_QUOTA_MIB, MAX_DOWNLOAD_QUOTA_MIB] {
            std::fs::write(&path, format!("[storage]\ndownload_quota_mib = {value}\n")).unwrap();
            assert_eq!(load(&path).unwrap().storage.download_quota_mib, value);
        }

        let original = std::fs::read_to_string(&path).unwrap();
        let mut s = Settings::default();
        s.storage.download_quota_mib = 0;
        let error = save(&path, &s).unwrap_err();
        assert!(error.contains("download_quota_mib"), "{error}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), original);
        assert_no_transaction_artifacts(dir.path());
    }

    #[test]
    fn default_storage_values_are_omitted() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");

        save(&path, &Settings::default()).unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(!raw.contains("[storage]"), "{raw}");
        assert!(!raw.contains("download_quota_mib"), "{raw}");

        std::fs::write(
            &path,
            "[storage]\n\
             root = \"\"\n\
             download_quota_mib = 2048\n\
             [storage.paths]\n\
             proxies = \"\"\n",
        )
        .unwrap();
        let s = load(&path).unwrap();
        assert_eq!(s.storage, StorageSettings::default());
        save(&path, &s).unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(!raw.contains("[storage]"), "{raw}");
    }

    #[test]
    fn clearing_storage_values_preserves_unknown_nested_values() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let absolute = dir.path().join("cache");
        let absolute = absolute.to_str().unwrap();
        std::fs::write(
            &path,
            format!(
                "# storage heading\n\
                 [storage]\n\
                 root = {absolute:?}\n\
                 download_quota_mib = 4096\n\
                 future_policy = \"keep\" # future policy comment\n\
                 [storage.paths]\n\
                 proxies = {absolute:?}\n\
                 download = {absolute:?}\n\
                 catalog = {absolute:?}\n\
                 luts = {absolute:?}\n\
                 lottie = {absolute:?}\n\
                 templates = {absolute:?}\n\
                 future_cache = {absolute:?} # future cache comment\n"
            ),
        )
        .unwrap();

        let mut s = load(&path).unwrap();
        s.storage = StorageSettings::default();
        save(&path, &s).unwrap();

        let raw = std::fs::read_to_string(&path).unwrap();
        let doc = raw.parse::<DocumentMut>().unwrap();
        let storage = section(&doc, "storage").unwrap();
        assert!(storage.get("root").is_none(), "{raw}");
        assert!(storage.get("download_quota_mib").is_none(), "{raw}");
        assert_eq!(
            storage.get("future_policy").and_then(Item::as_str),
            Some("keep")
        );
        let paths = storage.get("paths").and_then(Item::as_table).unwrap();
        for key in [
            "proxies",
            "download",
            "catalog",
            "luts",
            "lottie",
            "templates",
        ] {
            assert!(paths.get(key).is_none(), "{key} was not cleared: {raw}");
        }
        assert_eq!(
            paths.get("future_cache").and_then(Item::as_str),
            Some(absolute)
        );
        assert!(raw.contains("# storage heading"), "{raw}");
        assert!(raw.contains("# future policy comment"), "{raw}");
        assert!(raw.contains("# future cache comment"), "{raw}");
    }

    #[test]
    fn storage_save_preserves_comments_unknown_keys_and_unrelated_tables() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let absolute = dir.path().join("cache");
        let absolute = absolute.to_str().unwrap();
        std::fs::write(
            &path,
            format!(
                "# config heading\n\
                 [storage]\n\
                 root = {absolute:?} # root comment\n\
                 download_quota_mib = 4096 # quota comment\n\
                 future_policy = \"keep\" # storage unknown\n\
                 [storage.paths]\n\
                 proxies = {absolute:?} # proxy comment\n\
                 future_cache = {absolute:?} # paths unknown\n\
                 [plugins]\n\
                 enabled = true # unrelated\n"
            ),
        )
        .unwrap();

        let mut s = load(&path).unwrap();
        s.ai.model = "changed-elsewhere".into();
        save(&path, &s).unwrap();

        let raw = std::fs::read_to_string(&path).unwrap();
        for comment in [
            "# config heading",
            "# root comment",
            "# quota comment",
            "# storage unknown",
            "# proxy comment",
            "# paths unknown",
            "# unrelated",
        ] {
            assert!(raw.contains(comment), "lost {comment}: {raw}");
        }
        assert!(raw.contains("future_policy = \"keep\""), "{raw}");
        assert!(raw.contains("future_cache = "), "{raw}");
        assert!(raw.contains("[plugins]"), "{raw}");
        assert!(raw.contains("enabled = true"), "{raw}");
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_storage_path_returns_error_without_rewriting() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let original = "# keep me\n[future]\nvalue = true\n";
        std::fs::write(&path, original).unwrap();

        let mut s = Settings::default();
        s.storage.root = Some(PathBuf::from(OsString::from_vec(
            b"/tmp/cutlass-\xff".to_vec(),
        )));
        let error = save(&path, &s).unwrap_err();
        assert!(error.contains("UTF-8"), "{error}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), original);
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
"#,
        )
        .unwrap();

        let s = load(&path).unwrap();
        assert_eq!(s.ai.base_url, "http://localhost:11434/v1");
        assert_eq!(s.ai.model, "qwen3:14b");
        assert_eq!(s.ai.api_key_env.as_deref(), Some("OPENAI_API_KEY"));
        assert!(s.ai.is_configured());
        assert_eq!(s.appearance.theme, ThemeChoice::Ember);
    }

    #[test]
    fn malformed_file_is_an_error_not_a_default() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let secret = "sk-must-not-appear-in-errors";
        let malformed = format!("[ai]\napi_key = {secret}\n");
        std::fs::write(&path, &malformed).unwrap();
        assert!(load(&path).unwrap_err().contains("could not parse"));
        let save_error = save(&path, &Settings::default()).unwrap_err();
        assert!(save_error.contains("could not parse"), "{save_error}");
        assert!(!save_error.contains(secret), "{save_error}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), malformed);
        assert_no_transaction_artifacts(dir.path());
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
    fn preserves_tables_from_other_builds() {
        // A config written by a build that still had a `[cache]` table (or any
        // future section) must survive a save from this one untouched.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "[cache]\nbudget_mb = 1024\n").unwrap();

        let mut s = load(&path).unwrap();
        s.ai.base_url = "http://x/v1".into();
        s.ai.model = "m".into();
        save(&path, &s).unwrap();

        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(raw.contains("[cache]"), "unowned table kept: {raw}");
        assert!(raw.contains("budget_mb = 1024"));
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
        assert_no_transaction_artifacts(path.parent().unwrap());
    }

    #[test]
    fn save_replaces_existing_file_without_transaction_artifacts() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let original = "# keep this comment\n[ai]\nmodel = \"old\"\n\n[future]\nkeep = true\n";
        std::fs::write(&path, original).unwrap();

        let mut settings = load(&path).unwrap();
        settings.ai.model = "new".into();
        save(&path, &settings).unwrap();

        let saved = std::fs::read_to_string(&path).unwrap();
        assert!(saved.contains("model = \"new\""), "{saved}");
        assert!(saved.contains("# keep this comment"), "{saved}");
        assert!(saved.contains("[future]"), "{saved}");
        assert_no_transaction_artifacts(dir.path());
    }

    #[cfg(unix)]
    #[test]
    fn save_preserves_existing_file_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "[ai]\nmodel = \"old\"\n").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();

        let mut settings = load(&path).unwrap();
        settings.ai.model = "new".into();
        save(&path, &settings).unwrap();

        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o640);
        assert_no_transaction_artifacts(dir.path());
    }

    #[cfg(unix)]
    #[test]
    fn new_temporary_config_is_private_before_installation() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let destination = dir.path().join("config.toml");
        let temporary = write_synced_temp(&destination, b"secret = \"value\"\n", None).unwrap();

        let mode = std::fs::metadata(&temporary).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        std::fs::remove_file(temporary).unwrap();
        assert_no_transaction_artifacts(dir.path());
    }

    #[test]
    fn fallback_swap_replaces_existing_file_and_cleans_backup() {
        let dir = tempfile::tempdir().unwrap();
        let destination = dir.path().join("config.toml");
        let original = b"version = \"old\"\n";
        let replacement = b"version = \"new\"\n";
        std::fs::write(&destination, original).unwrap();
        let temporary = write_synced_temp(&destination, replacement, None).unwrap();
        let fs = FaultFs::failing(&[1], &[]);

        install_temp_with_ops(&destination, &temporary, &fs).unwrap();

        assert_eq!(std::fs::read(&destination).unwrap(), replacement);
        assert_no_transaction_artifacts(dir.path());
    }

    #[test]
    fn fallback_install_failure_restores_original_and_cleans_temp() {
        let dir = tempfile::tempdir().unwrap();
        let destination = dir.path().join("config.toml");
        let original = b"version = \"old\"\n";
        let replacement = b"version = \"new\"\n";
        std::fs::write(&destination, original).unwrap();
        let temporary = write_synced_temp(&destination, replacement, None).unwrap();
        let fs = FaultFs::failing(&[1, 3], &[]);

        let error = install_temp_with_ops(&destination, &temporary, &fs).unwrap_err();

        assert!(
            error.contains("original configuration was restored"),
            "{error}"
        );
        assert_eq!(std::fs::read(&destination).unwrap(), original);
        assert_no_transaction_artifacts(dir.path());
    }

    #[test]
    fn fallback_reports_failed_rollback_and_retains_original_backup() {
        let dir = tempfile::tempdir().unwrap();
        let destination = dir.path().join("config.toml");
        let original = b"version = \"old\"\n";
        let replacement = b"version = \"new\"\n";
        std::fs::write(&destination, original).unwrap();
        let temporary = write_synced_temp(&destination, replacement, None).unwrap();
        let fs = FaultFs::failing(&[1, 3, 4], &[]);

        let error = install_temp_with_ops(&destination, &temporary, &fs).unwrap_err();

        assert!(error.contains("rollback failed"), "{error}");
        assert!(error.contains("backup was retained"), "{error}");
        assert!(!destination.exists());
        let artifacts = transaction_artifacts(dir.path());
        assert_eq!(artifacts.len(), 1, "{artifacts:?}");
        assert!(
            artifacts[0]
                .file_name()
                .unwrap()
                .to_string_lossy()
                .contains(".cutlass-backup-")
        );
        assert_eq!(std::fs::read(&artifacts[0]).unwrap(), original);
    }

    #[test]
    fn fallback_backup_cleanup_failure_does_not_uncommit_installation() {
        let dir = tempfile::tempdir().unwrap();
        let destination = dir.path().join("config.toml");
        let original = b"version = \"old\"\n";
        let replacement = b"version = \"new\"\n";
        std::fs::write(&destination, original).unwrap();
        let temporary = write_synced_temp(&destination, replacement, None).unwrap();
        let fs = FaultFs::failing(&[1], &[1]);

        install_temp_with_ops(&destination, &temporary, &fs).unwrap();

        assert_eq!(std::fs::read(&destination).unwrap(), replacement);
        let artifacts = transaction_artifacts(dir.path());
        assert_eq!(artifacts.len(), 1, "{artifacts:?}");
        assert!(
            artifacts[0]
                .file_name()
                .unwrap()
                .to_string_lossy()
                .contains(".cutlass-backup-")
        );
        assert_eq!(std::fs::read(&artifacts[0]).unwrap(), original);
    }

    #[test]
    fn providers_and_account_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");

        let mut s = Settings::default();
        s.providers.insert(
            "pexels".into(),
            ProviderSettings {
                api_key: None,
                api_key_env: Some("PEXELS_API_KEY".into()),
            },
        );
        s.providers.insert(
            "elevenlabs".into(),
            ProviderSettings {
                api_key: Some("sk-11".into()),
                api_key_env: None,
            },
        );
        s.account.base_url = "https://staging.api.cutlass.sh".into();
        save(&path, &s).unwrap();

        let loaded = load(&path).unwrap();
        assert_eq!(
            loaded.provider("pexels").api_key_env.as_deref(),
            Some("PEXELS_API_KEY")
        );
        assert!(loaded.provider("pexels").is_configured());
        assert_eq!(
            loaded.provider("elevenlabs").api_key.as_deref(),
            Some("sk-11")
        );
        assert!(!loaded.provider("nonexistent").is_configured());
        assert_eq!(loaded.account.base_url, "https://staging.api.cutlass.sh");

        // Dropping a provider removes its table; clearing the account
        // override removes the key.
        let mut s = loaded;
        s.providers.remove("elevenlabs");
        s.account.base_url.clear();
        save(&path, &s).unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(!raw.contains("elevenlabs"), "{raw}");
        assert!(!raw.contains("base_url = \"https://staging"), "{raw}");
        assert!(raw.contains("[providers.pexels]"), "{raw}");
    }

    #[test]
    fn provider_key_resolution_prefers_literal() {
        let p = ProviderSettings {
            api_key: Some("literal".into()),
            api_key_env: Some("SOME_ENV_THAT_IS_UNSET_12345".into()),
        };
        assert_eq!(p.resolve_key().as_deref(), Some("literal"));
        let p = ProviderSettings {
            api_key: None,
            api_key_env: Some("SOME_ENV_THAT_IS_UNSET_12345".into()),
        };
        assert_eq!(p.resolve_key(), None);
        assert!(p.is_configured(), "env-named key counts as configured");
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
    fn autonomy_parses_and_tolerates_missing_or_garbage_values() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");

        std::fs::write(&path, "[ai]\nmodel = \"m\"\nautonomy = \"full\"\n").unwrap();
        assert_eq!(load(&path).unwrap().ai.autonomy, Autonomy::Full);

        // Missing key keeps the default.
        std::fs::write(&path, "[ai]\nmodel = \"m\"\n").unwrap();
        assert_eq!(load(&path).unwrap().ai.autonomy, Autonomy::Ask);

        // Unrecognized value keeps the default rather than failing the load.
        std::fs::write(&path, "[ai]\nautonomy = \"yolo\"\n").unwrap();
        assert_eq!(load(&path).unwrap().ai.autonomy, Autonomy::Ask);

        // "confirm" is the tolerated alias for Ask.
        std::fs::write(&path, "[ai]\nautonomy = \"confirm\"\n").unwrap();
        assert_eq!(load(&path).unwrap().ai.autonomy, Autonomy::Ask);
    }

    #[test]
    fn autonomy_round_trips_and_preserves_unrelated_keys() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            "# my cutlass config\n[ai]\nbase_url = \"http://x/v1\"  # local\nmodel = \"m\"\n",
        )
        .unwrap();

        let mut s = load(&path).unwrap();
        s.ai.autonomy = Autonomy::Full;
        save(&path, &s).unwrap();

        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(raw.contains("autonomy = \"full\""), "{raw}");
        assert!(raw.contains("# my cutlass config"), "leading comment kept");
        assert!(raw.contains("# local"), "inline comment kept");
        assert_eq!(load(&path).unwrap().ai.autonomy, Autonomy::Full);

        // Back to the default removes the key (the `use_account` convention).
        let mut s = load(&path).unwrap();
        s.ai.autonomy = Autonomy::Ask;
        save(&path, &s).unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(!raw.contains("autonomy"), "default left no literal: {raw}");
        assert_eq!(load(&path).unwrap().ai.autonomy, Autonomy::Ask);
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
