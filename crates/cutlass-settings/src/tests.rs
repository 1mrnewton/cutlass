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
        analysis: Some(dir.path().join("analysis")),
        ai_models: Some(dir.path().join("ai-models")),
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
        "analysis",
        "ai_models",
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
        "analysis",
        "ai_models",
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
                 analysis = \"relative/analysis\"\n\
                 ai_models = \"relative/ai-models\"\n\
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
    assert_eq!(s.storage.paths.analysis, None);
    assert_eq!(s.storage.paths.ai_models, None);
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

    s.storage.root = None;
    s.storage.paths.analysis = Some(PathBuf::from("relative/analysis"));
    let error = save(&path, &s).unwrap_err();
    assert!(error.contains("paths.analysis"), "{error}");
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        original,
        "failed analysis-path validation must not rewrite the file"
    );
    assert_no_transaction_artifacts(dir.path());

    s.storage.paths.analysis = None;
    s.storage.paths.ai_models = Some(PathBuf::from("relative/ai-models"));
    let error = save(&path, &s).unwrap_err();
    assert!(error.contains("paths.ai_models"), "{error}");
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        original,
        "failed AI-models-path validation must not rewrite the file"
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
                 analysis = {absolute:?}\n\
                 ai_models = {absolute:?}\n\
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
        "analysis",
        "ai_models",
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
                 analysis = {absolute:?} # analysis comment\n\
                 ai_models = {absolute:?} # AI models comment\n\
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
        "# analysis comment",
        "# AI models comment",
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
    // Pre-source configs with URL+model migrate to freeform Advanced.
    assert_eq!(s.ai.source, AiSource::Custom);
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
fn recent_colors_round_trip_and_tolerate_bad_entries() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.toml");
    std::fs::write(
        &path,
        r#"
[appearance]
theme = "ember"
recent_colors = ["FF0000", "not-a-color", "00FF00", "FF0000", "0000FF80"]
"#,
    )
    .unwrap();

    let s = load(&path).unwrap();
    assert_eq!(
        s.appearance.recent_colors,
        vec![[0xFF, 0, 0, 0xFF], [0, 0xFF, 0, 0xFF], [0, 0, 0xFF, 0x80]]
    );

    let mut next = s;
    push_recent(&mut next.appearance.recent_colors, [1, 2, 3, 255]);
    save(&path, &next).unwrap();

    let raw = std::fs::read_to_string(&path).unwrap();
    assert!(raw.contains("recent_colors"));
    assert!(raw.contains("010203"));

    let reloaded = load(&path).unwrap();
    assert_eq!(reloaded.appearance.recent_colors[0], [1, 2, 3, 255]);
    assert_eq!(reloaded.appearance.theme, ThemeChoice::Ember);

    // Clearing the list removes the key entirely.
    let mut cleared = reloaded;
    cleared.appearance.recent_colors.clear();
    save(&path, &cleared).unwrap();
    let cleared_raw = std::fs::read_to_string(&path).unwrap();
    assert!(
        !cleared_raw.contains("recent_colors"),
        "empty list should drop the key: {cleared_raw}"
    );
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
fn providers_and_cloud_round_trip() {
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
    s.cloud.base_url = "https://staging.api.cutlass.sh".into();
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
    assert_eq!(loaded.cloud.base_url, "https://staging.api.cutlass.sh");
    let raw = std::fs::read_to_string(&path).unwrap();
    assert!(raw.contains("[cloud]"), "{raw}");

    // Dropping a provider removes its table; clearing the cloud
    // override removes the key.
    let mut s = loaded;
    s.providers.remove("elevenlabs");
    s.cloud.base_url.clear();
    save(&path, &s).unwrap();
    let raw = std::fs::read_to_string(&path).unwrap();
    assert!(!raw.contains("elevenlabs"), "{raw}");
    assert!(!raw.contains("base_url = \"https://staging"), "{raw}");
    assert!(raw.contains("[providers.pexels]"), "{raw}");
}

#[test]
fn legacy_account_base_url_loads_into_cloud() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.toml");
    std::fs::write(
        &path,
        "[account]\nbase_url = \"https://legacy.api.cutlass.sh\"\nauth_base_url = \"https://cutlass.sh\"\n",
    )
    .unwrap();
    let loaded = load(&path).unwrap();
    assert_eq!(loaded.cloud.base_url, "https://legacy.api.cutlass.sh");

    // Saving migrates to `[cloud]` and drops `[account]`.
    save(&path, &loaded).unwrap();
    let raw = std::fs::read_to_string(&path).unwrap();
    assert!(raw.contains("[cloud]"), "{raw}");
    assert!(!raw.contains("[account]"), "{raw}");
    assert!(!raw.contains("auth_base_url"), "{raw}");
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

    // Back to the default removes the key (absence = default).
    let mut s = load(&path).unwrap();
    s.ai.autonomy = Autonomy::Ask;
    save(&path, &s).unwrap();
    let raw = std::fs::read_to_string(&path).unwrap();
    assert!(!raw.contains("autonomy"), "default left no literal: {raw}");
    assert_eq!(load(&path).unwrap().ai.autonomy, Autonomy::Ask);
}

#[test]
fn ai_protocol_and_reasoning_summary_are_tolerant() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.toml");

    std::fs::write(
        &path,
        "[ai]\napi_protocol = \"responses\"\nreasoning_summary = \"off\"\n",
    )
    .unwrap();
    let loaded = load(&path).unwrap();
    assert_eq!(loaded.ai.api_protocol, AiApiProtocol::Responses);
    assert_eq!(loaded.ai.reasoning_summary, ReasoningSummary::Off);

    std::fs::write(
        &path,
        "[ai]\napi_protocol = \"chat-completions\"\nreasoning_summary = \"on\"\n",
    )
    .unwrap();
    let loaded = load(&path).unwrap();
    assert_eq!(loaded.ai.api_protocol, AiApiProtocol::ChatCompletions);
    assert_eq!(loaded.ai.reasoning_summary, ReasoningSummary::Auto);

    std::fs::write(
        &path,
        "[ai]\napi_protocol = \"future\"\nreasoning_summary = \"verbose\"\n",
    )
    .unwrap();
    let loaded = load(&path).unwrap();
    assert_eq!(loaded.ai.api_protocol, AiApiProtocol::default());
    assert_eq!(loaded.ai.reasoning_summary, ReasoningSummary::default());
}

#[test]
fn ai_protocol_and_reasoning_summary_round_trip_canonical_keys() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.toml");
    std::fs::write(&path, "# keep me\n[ai]\nmodel = \"gpt-5\"\n").unwrap();

    let mut settings = load(&path).unwrap();
    settings.ai.api_protocol = AiApiProtocol::Responses;
    settings.ai.reasoning_summary = ReasoningSummary::Off;
    save(&path, &settings).unwrap();

    let raw = std::fs::read_to_string(&path).unwrap();
    assert!(raw.contains("api_protocol = \"responses\""), "{raw}");
    assert!(raw.contains("reasoning_summary = \"off\""), "{raw}");
    assert!(raw.contains("# keep me"), "{raw}");
    let loaded = load(&path).unwrap();
    assert_eq!(loaded.ai.api_protocol, AiApiProtocol::Responses);
    assert_eq!(loaded.ai.reasoning_summary, ReasoningSummary::Off);

    settings.ai.api_protocol = AiApiProtocol::default();
    settings.ai.reasoning_summary = ReasoningSummary::default();
    save(&path, &settings).unwrap();
    let raw = std::fs::read_to_string(&path).unwrap();
    assert!(!raw.contains("api_protocol"), "{raw}");
    assert!(!raw.contains("reasoning_summary"), "{raw}");
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

#[test]
fn ai_source_parses_migrates_and_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.toml");

    // Missing source + empty AI → Local default, not configured.
    std::fs::write(&path, "[ai]\n").unwrap();
    let loaded = load(&path).unwrap();
    assert_eq!(loaded.ai.source, AiSource::Local);
    assert!(!loaded.ai.is_configured());

    // Explicit openrouter needs a key to be configured.
    std::fs::write(
        &path,
        "[ai]\nsource = \"openrouter\"\nmodel = \"openai/gpt-5.6-sol\"\napi_key = \"sk-or\"\n",
    )
    .unwrap();
    let loaded = load(&path).unwrap();
    assert_eq!(loaded.ai.source, AiSource::OpenRouter);
    assert!(loaded.ai.is_configured());

    std::fs::write(
        &path,
        "[ai]\nsource = \"openrouter\"\nmodel = \"openai/gpt-5.6-sol\"\n",
    )
    .unwrap();
    assert!(!load(&path).unwrap().ai.is_configured());

    // Round-trip non-default source.
    let mut settings = Settings::default();
    settings.ai.source = AiSource::OpenRouter;
    settings.ai.model = "openai/gpt-5.6-sol".into();
    settings.ai.api_key = Some("sk-or".into());
    save(&path, &settings).unwrap();
    let raw = std::fs::read_to_string(&path).unwrap();
    assert!(raw.contains("source = \"openrouter\""), "{raw}");
    assert_eq!(load(&path).unwrap().ai.source, AiSource::OpenRouter);

    // Default Local omits the key.
    settings.ai.source = AiSource::Local;
    settings.ai.base_url = "http://localhost:11434/v1".into();
    settings.ai.model = "qwen3:14b".into();
    settings.ai.api_key = None;
    save(&path, &settings).unwrap();
    let raw = std::fs::read_to_string(&path).unwrap();
    assert!(!raw.contains("source"), "{raw}");
}
