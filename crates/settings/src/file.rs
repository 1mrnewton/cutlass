//! Generation, load, and save.
//!
//! A load returns the generation stored in the file. A save sends that
//! generation. A mismatch writes nothing. A successful save stores the
//! next generation. An absent file is generation 0 and is not itself a
//! document. A file that cannot be parsed is left as it was.

use std::fmt;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use crate::Settings;

/// A settings file that was read.
///
/// [`Loaded::generation`] is what a later [`save`] must send.
#[derive(Clone, Debug, PartialEq)]
pub struct Loaded {
    pub generation: u64,
    pub settings: Settings,
}

/// Why a settings file could not be read or written.
#[derive(Debug, PartialEq, Eq)]
pub enum SettingsError {
    /// The path is not a file.
    Missing { path: PathBuf },
    /// The bytes are not the JSON this loader accepts.
    Json { path: PathBuf, message: String },
    /// The caller read a different generation than the file.
    GenerationMismatch { expected: u64, actual: u64 },
    /// The generation cannot advance, so the file was left as it was.
    GenerationExhausted,
    /// The filesystem refused the read or the write.
    Io { path: PathBuf, message: String },
}

impl fmt::Display for SettingsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing { path } => {
                write!(formatter, "missing settings file {}", path.display())
            }
            Self::Json { path, message } => {
                write!(formatter, "could not read {}: {message}", path.display())
            }
            Self::GenerationMismatch { expected, actual } => {
                write!(
                    formatter,
                    "settings generation {actual} does not match {expected}, so nothing was written"
                )
            }
            Self::GenerationExhausted => {
                write!(
                    formatter,
                    "settings generation cannot advance, so nothing was written"
                )
            }
            Self::Io { path, message } => {
                write!(formatter, "could not access {}: {message}", path.display())
            }
        }
    }
}

impl std::error::Error for SettingsError {}

/// Read the settings file.
///
/// An absent path is [`SettingsError::Missing`]. Truncated or hostile JSON
/// is [`SettingsError::Json`]. An omitted generation is 0.
pub fn load(path: &Path) -> Result<Loaded, SettingsError> {
    let bytes = read_required(path)?;
    parse(&bytes, path)
}

/// Write `settings` when `expected` is the generation [`load`] returned.
///
/// A mismatch, a generation that cannot advance, or a file that cannot be
/// parsed writes nothing and returns the cause. A successful write stores
/// the next generation and returns it. An absent file has generation 0, so
/// the first write sends 0 and stores 1. Unnamed fields survive when
/// `settings` still carries them. This function does not create a missing
/// parent directory and does not open a skill folder or a server URL.
pub fn save(path: &Path, expected: u64, settings: &Settings) -> Result<u64, SettingsError> {
    let actual = generation_on_disk(path)?;
    if actual != expected {
        return Err(SettingsError::GenerationMismatch { expected, actual });
    }
    let Some(next) = expected.checked_add(1) else {
        return Err(SettingsError::GenerationExhausted);
    };
    let bytes = encode(next, settings, path)?;
    write_bytes(path, &bytes)?;
    Ok(next)
}

fn generation_on_disk(path: &Path) -> Result<u64, SettingsError> {
    match fs::read(path) {
        Ok(bytes) => Ok(parse(&bytes, path)?.generation),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(0),
        Err(error) => Err(io_error(path, error)),
    }
}

fn read_required(path: &Path) -> Result<Vec<u8>, SettingsError> {
    match fs::read(path) {
        Ok(bytes) => Ok(bytes),
        Err(error) if error.kind() == ErrorKind::NotFound => Err(SettingsError::Missing {
            path: path.to_path_buf(),
        }),
        Err(error) => Err(io_error(path, error)),
    }
}

fn parse(bytes: &[u8], path: &Path) -> Result<Loaded, SettingsError> {
    let value: Value =
        serde_json::from_slice(bytes).map_err(|error| json_error(path, error.to_string()))?;
    let Value::Object(mut object) = value else {
        return Err(json_error(path, "settings file is not an object"));
    };
    let generation = take_generation(&mut object, path)?;
    let settings = serde_json::from_value(Value::Object(object))
        .map_err(|error| json_error(path, error.to_string()))?;
    Ok(Loaded {
        generation,
        settings,
    })
}

fn take_generation(object: &mut Map<String, Value>, path: &Path) -> Result<u64, SettingsError> {
    match object.remove("generation") {
        None => Ok(0),
        Some(Value::Number(number)) => number
            .as_u64()
            .ok_or_else(|| json_error(path, "generation is out of range")),
        Some(_) => Err(json_error(path, "generation is not a number")),
    }
}

fn encode(generation: u64, settings: &Settings, path: &Path) -> Result<Vec<u8>, SettingsError> {
    let value =
        serde_json::to_value(settings).map_err(|error| json_error(path, error.to_string()))?;
    let Value::Object(object) = value else {
        return Err(json_error(path, "settings did not encode as an object"));
    };
    let mut ordered = Map::new();
    ordered.insert("generation".to_owned(), Value::from(generation));
    for (key, item) in object {
        if key != "generation" {
            ordered.insert(key, item);
        }
    }
    let mut bytes = serde_json::to_vec(&Value::Object(ordered))
        .map_err(|error| json_error(path, error.to_string()))?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn write_bytes(path: &Path, bytes: &[u8]) -> Result<(), SettingsError> {
    fs::write(path, bytes).map_err(|error| io_error(path, error))
}

fn json_error(path: &Path, message: impl Into<String>) -> SettingsError {
    SettingsError::Json {
        path: path.to_path_buf(),
        message: message.into(),
    }
}

fn io_error(path: &Path, error: std::io::Error) -> SettingsError {
    SettingsError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use serde_json::Value;

    use super::{SettingsError, load, save};
    use crate::{Panel, Settings, Shortcut};

    struct Temp {
        path: PathBuf,
    }

    impl Temp {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("cutlass-settings-{}-{n}", std::process::id()));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn a_save_round_trips_and_advances_the_generation() {
        let temp = Temp::new();
        let path = temp.path.join("settings.json");
        let mut settings = Settings::default();
        settings.playback.loop_playback = true;
        settings.shortcuts.push(Shortcut {
            action: "split".to_owned(),
            keys: "s".to_owned(),
            ..Shortcut::default()
        });
        let generation = save(&path, 0, &settings).unwrap();
        assert_eq!(generation, 1);
        let mut loaded = load(&path).unwrap();
        assert_eq!(loaded.generation, 1);
        assert!(loaded.settings.model.local);
        assert!(!loaded.settings.model.cloud);
        assert!(loaded.settings.playback.loop_playback);
        assert_eq!(loaded.settings.shortcuts[0].action, "split");
        assert_eq!(loaded.settings.shortcuts[0].keys, "s");
        assert!(loaded.settings.rest.get("generation").is_none());

        loaded.settings.playback.scrub_audio = true;
        let before = fs::read(&path).unwrap();
        let next = save(&path, loaded.generation, &loaded.settings).unwrap();
        assert_eq!(next, 2);
        assert_ne!(fs::read(&path).unwrap(), before);
        let again = load(&path).unwrap();
        assert_eq!(again.generation, 2);
        assert!(again.settings.playback.loop_playback);
        assert!(again.settings.playback.scrub_audio);
        assert!(again.settings.model.local);
        assert!(!again.settings.model.cloud);
        let value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(value["generation"], 2);
        assert_eq!(value["shortcuts"][0]["action"], "split");
        assert_eq!(value["playback"]["scrub_audio"], true);
    }

    #[test]
    fn a_stale_generation_writes_nothing() {
        let temp = Temp::new();
        let path = temp.path.join("settings.json");
        let mut settings = Settings::default();
        settings.panels.push(Panel {
            id: "media".to_owned(),
            open: true,
            ..Panel::default()
        });
        save(&path, 0, &settings).unwrap();
        let before = fs::read(&path).unwrap();
        let error = save(&path, 0, &Settings::default()).unwrap_err();
        assert_eq!(
            error,
            SettingsError::GenerationMismatch {
                expected: 0,
                actual: 1
            }
        );
        assert!(error.to_string().contains("nothing was written"));
        assert_eq!(fs::read(&path).unwrap(), before);
        let loaded = load(&path).unwrap();
        assert_eq!(loaded.generation, 1);
        assert_eq!(loaded.settings.panels[0].id, "media");
        assert!(loaded.settings.panels[0].open);
    }

    #[test]
    fn an_absent_file_is_missing_and_the_first_write_stores_generation_one() {
        let temp = Temp::new();
        let path = temp.path.join("settings.json");
        let error = load(&path).unwrap_err();
        assert_eq!(error, SettingsError::Missing { path: path.clone() });
        assert!(error.to_string().contains("missing settings file"));
        let error = save(&path, 1, &Settings::default()).unwrap_err();
        assert_eq!(
            error,
            SettingsError::GenerationMismatch {
                expected: 1,
                actual: 0
            }
        );
        assert!(!path.exists());
        let generation = save(&path, 0, &Settings::default()).unwrap();
        assert_eq!(generation, 1);
        let loaded = load(&path).unwrap();
        assert_eq!(loaded.generation, 1);
        assert!(loaded.settings.model.local);
        assert!(!loaded.settings.model.cloud);
        assert!(loaded.settings.sandbox.local);
        assert!(!loaded.settings.sandbox.online);
    }

    #[test]
    fn a_truncated_file_is_refused_and_left_in_place() {
        let temp = Temp::new();
        let path = temp.path.join("settings.json");
        for bytes in [
            b"{".as_slice(),
            b"",
            b"[]",
            b"{\"generation\":null}",
            b"{\"generation\":-1}",
        ] {
            fs::write(&path, bytes).unwrap();
            let before = fs::read(&path).unwrap();
            assert!(matches!(load(&path), Err(SettingsError::Json { .. })));
            let error = save(&path, 0, &Settings::default()).unwrap_err();
            assert!(matches!(error, SettingsError::Json { .. }));
            assert_eq!(fs::read(&path).unwrap(), before);
        }
    }

    #[test]
    fn hostile_json_is_left_unchanged() {
        let temp = Temp::new();
        let path = temp.path.join("settings.json");
        let raw = br#"{"generation":2,"shortcuts":"split","model":{"local":true,"cloud":false}}"#;
        fs::write(&path, raw).unwrap();
        let before = fs::read(&path).unwrap();
        assert!(matches!(load(&path), Err(SettingsError::Json { .. })));
        let error = save(&path, 2, &Settings::default()).unwrap_err();
        assert!(matches!(error, SettingsError::Json { .. }));
        assert_eq!(fs::read(&path).unwrap(), before);
    }

    #[test]
    fn an_unknown_field_survives_a_save_and_does_not_wipe_known_fields() {
        let temp = Temp::new();
        let path = temp.path.join("settings.json");
        let raw = br#"{"generation":4,"shortcuts":[{"action":"split","keys":"s","future_key":1}],"model":{"local":true,"cloud":false,"future_model":"kept"},"mcp_servers":[{"name":"far","kind":"remote","url":"http://127.0.0.1:9/mcp","enabled":false}],"skill_folders":[{"path":"../skills/demo","future_skill":true}],"future_switch":true}"#;
        fs::write(&path, raw).unwrap();
        let mut loaded = load(&path).unwrap();
        assert_eq!(loaded.generation, 4);
        assert_eq!(loaded.settings.shortcuts[0].action, "split");
        assert_eq!(loaded.settings.shortcuts[0].keys, "s");
        assert_eq!(loaded.settings.shortcuts[0].rest["future_key"], 1);
        assert!(loaded.settings.model.local);
        assert!(!loaded.settings.model.cloud);
        assert_eq!(loaded.settings.model.rest["future_model"], "kept");
        assert!(!loaded.settings.mcp_servers[0].enabled);
        assert_eq!(loaded.settings.mcp_servers[0].url, "http://127.0.0.1:9/mcp");
        assert_eq!(loaded.settings.skill_folders[0].path, "../skills/demo");
        assert_eq!(loaded.settings.skill_folders[0].rest["future_skill"], true);
        assert_eq!(loaded.settings.rest["future_switch"], true);
        assert!(loaded.settings.rest.get("generation").is_none());
        assert!(loaded.settings.memory);
        assert_eq!(loaded.settings.step_cap, 32);

        loaded.settings.playback.scrub_audio = true;
        let generation = save(&path, 4, &loaded.settings).unwrap();
        assert_eq!(generation, 5);
        let again = load(&path).unwrap();
        assert_eq!(again.generation, 5);
        assert_eq!(again.settings.shortcuts[0].action, "split");
        assert_eq!(again.settings.shortcuts[0].rest["future_key"], 1);
        assert!(again.settings.model.local);
        assert!(!again.settings.model.cloud);
        assert_eq!(again.settings.model.rest["future_model"], "kept");
        assert!(again.settings.playback.scrub_audio);
        assert_eq!(again.settings.skill_folders[0].path, "../skills/demo");
        assert!(!temp.path.join("skills").exists());
        let value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(value["generation"], 5);
        assert_eq!(value["future_switch"], true);
        assert_eq!(value["shortcuts"][0]["action"], "split");
        assert_eq!(value["shortcuts"][0]["future_key"], 1);
        assert_eq!(value["model"]["local"], true);
        assert_eq!(value["model"]["cloud"], false);
        assert_eq!(value["model"]["future_model"], "kept");
        assert_eq!(value["skill_folders"][0]["future_skill"], true);
        assert_eq!(value["playback"]["scrub_audio"], true);
        assert_eq!(value["mcp_servers"][0]["url"], "http://127.0.0.1:9/mcp");
    }

    #[test]
    fn a_file_with_only_a_generation_keeps_the_model_defaults() {
        let temp = Temp::new();
        let path = temp.path.join("settings.json");
        fs::write(&path, b"{\"generation\":3}\n").unwrap();
        let loaded = load(&path).unwrap();
        assert_eq!(loaded.generation, 3);
        assert!(loaded.settings.model.local);
        assert!(!loaded.settings.model.cloud);
        assert!(loaded.settings.sandbox.local);
        assert!(!loaded.settings.sandbox.online);
        assert_eq!(loaded.settings.step_cap, 32);
        assert!(loaded.settings.memory);
        assert!(loaded.settings.shortcuts.is_empty());
    }

    #[test]
    fn a_generation_that_cannot_advance_writes_nothing() {
        let temp = Temp::new();
        let path = temp.path.join("settings.json");
        let raw = format!(
            r#"{{"generation":{},"model":{{"local":true,"cloud":false}}}}"#,
            u64::MAX
        );
        fs::write(&path, raw.as_bytes()).unwrap();
        let before = fs::read(&path).unwrap();
        let error = save(&path, u64::MAX, &Settings::default()).unwrap_err();
        assert_eq!(error, SettingsError::GenerationExhausted);
        assert!(error.to_string().contains("nothing was written"));
        assert_eq!(fs::read(&path).unwrap(), before);
        let loaded = load(&path).unwrap();
        assert_eq!(loaded.generation, u64::MAX);
        assert!(loaded.settings.model.local);
        assert!(!loaded.settings.model.cloud);
    }

    #[test]
    fn a_missing_parent_directory_is_not_created() {
        let temp = Temp::new();
        let path = temp.path.join("missing").join("settings.json");
        let error = save(&path, 0, &Settings::default()).unwrap_err();
        assert!(matches!(error, SettingsError::Io { .. }));
        assert!(!temp.path.join("missing").exists());
    }

    #[test]
    fn a_directory_is_not_a_settings_file() {
        let temp = Temp::new();
        let error = load(&temp.path).unwrap_err();
        assert!(matches!(error, SettingsError::Io { .. }));
        let before = fs::read_dir(&temp.path).unwrap().count();
        let error = save(&temp.path, 0, &Settings::default()).unwrap_err();
        assert!(matches!(error, SettingsError::Io { .. }));
        assert_eq!(fs::read_dir(&temp.path).unwrap().count(), before);
    }
}
