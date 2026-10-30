//! The user's settings file and its own undo stack.
//!
//! The generation is separate from the open draft. Undo of one leaves the
//! other as it is. A missing file is generation 0 until the first save.

use std::fmt;
use std::path::{Path, PathBuf};

use cutlass_settings::{Settings, SettingsError, load, save};

/// The settings generation and the settings at that generation.
#[derive(Clone, Debug, PartialEq)]
pub struct SettingsView {
    /// The generation held in memory.
    pub generation: u64,
    /// The settings held in memory.
    pub settings: Settings,
}

/// Why a settings tool did not load or save.
#[derive(Debug, PartialEq, Eq)]
pub enum SettingsToolError {
    /// The settings file could not be read or written.
    File(SettingsError),
    /// The caller read a different generation than the one held in memory.
    GenerationMismatch {
        /// The generation the caller sent.
        expected: u64,
        /// The generation this tool holds, or the generation on disk.
        actual: u64,
    },
    /// Undo was asked for and this tool has no settings undo step.
    EmptyUndo,
    /// Redo was asked for and this tool has no settings redo step.
    EmptyRedo,
}

impl fmt::Display for SettingsToolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::File(error) => write!(formatter, "{error}"),
            Self::GenerationMismatch { expected, actual } => write!(
                formatter,
                "settings generation {actual} does not match {expected}, so nothing was written"
            ),
            Self::EmptyUndo => write!(formatter, "nothing to undo"),
            Self::EmptyRedo => write!(formatter, "nothing to redo"),
        }
    }
}

impl std::error::Error for SettingsToolError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::File(error) => Some(error),
            Self::GenerationMismatch { .. } | Self::EmptyUndo | Self::EmptyRedo => None,
        }
    }
}

/// Settings file, generation, and a separate undo stack.
#[derive(Debug)]
pub struct SettingsTool {
    path: PathBuf,
    generation: u64,
    settings: Settings,
    undo: Vec<Settings>,
    redo: Vec<Settings>,
}

impl SettingsTool {
    /// Load the file. A missing file is generation 0 and does not create it.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, SettingsToolError> {
        let path = path.into();
        match load(&path) {
            Ok(loaded) => Ok(Self::ready(path, loaded.generation, loaded.settings)),
            Err(SettingsError::Missing { .. }) => Ok(Self::ready(path, 0, Settings::default())),
            Err(error) => Err(SettingsToolError::File(error)),
        }
    }

    /// The generation and settings held in memory.
    #[must_use]
    pub fn get(&self) -> SettingsView {
        SettingsView {
            generation: self.generation,
            settings: self.settings.clone(),
        }
    }

    /// Save `settings` when `expected` is the current generation.
    pub fn set(&mut self, expected: u64, settings: Settings) -> Result<u64, SettingsToolError> {
        if expected != self.generation {
            return Err(SettingsToolError::GenerationMismatch {
                expected,
                actual: self.generation,
            });
        }
        let generation = save(&self.path, expected, &settings).map_err(map_save)?;
        let previous = std::mem::replace(&mut self.settings, settings);
        self.undo.push(previous);
        self.redo.clear();
        self.generation = generation;
        Ok(generation)
    }

    /// Save the previous settings and return the generation that write stored.
    pub fn undo(&mut self) -> Result<u64, SettingsToolError> {
        let Some(previous) = self.undo.pop() else {
            return Err(SettingsToolError::EmptyUndo);
        };
        match save(&self.path, self.generation, &previous) {
            Ok(generation) => {
                let current = std::mem::replace(&mut self.settings, previous);
                self.redo.push(current);
                self.generation = generation;
                Ok(generation)
            }
            Err(error) => {
                self.undo.push(previous);
                Err(map_save(error))
            }
        }
    }

    /// Save the settings that undo removed.
    pub fn redo(&mut self) -> Result<u64, SettingsToolError> {
        let Some(next) = self.redo.pop() else {
            return Err(SettingsToolError::EmptyRedo);
        };
        match save(&self.path, self.generation, &next) {
            Ok(generation) => {
                let current = std::mem::replace(&mut self.settings, next);
                self.undo.push(current);
                self.generation = generation;
                Ok(generation)
            }
            Err(error) => {
                self.redo.push(next);
                Err(map_save(error))
            }
        }
    }

    /// The generation held in memory.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The settings held in memory.
    #[must_use]
    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    /// The file these settings are loaded from.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn ready(path: PathBuf, generation: u64, settings: Settings) -> Self {
        Self {
            path,
            generation,
            settings,
            undo: Vec::new(),
            redo: Vec::new(),
        }
    }
}

fn map_save(error: SettingsError) -> SettingsToolError {
    match error {
        SettingsError::GenerationMismatch { expected, actual } => {
            SettingsToolError::GenerationMismatch { expected, actual }
        }
        other => SettingsToolError::File(other),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use cutlass_model::{Draft, Micros, Segment, SegmentId, TimeRange, TrackId};
    use cutlass_session::{Request, Session};
    use cutlass_settings::{Settings, SettingsError};

    use super::{SettingsTool, SettingsToolError, SettingsView};

    struct Temp {
        path: PathBuf,
    }

    impl Temp {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("cutlass-agent-settings-{}-{n}", std::process::id()));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }

        fn file(&self) -> PathBuf {
            self.path.join("settings.json")
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn session() -> (Session, TrackId) {
        let mut draft = Draft::empty("D");
        let track_id = TrackId::new("main");
        draft.tracks[0].id = track_id.clone();
        (Session::open(draft), track_id)
    }

    fn place(track_id: &TrackId, id: &str) -> Request {
        Request::Place {
            track_id: track_id.clone(),
            segment: Box::new(Segment::new(id, "media", TimeRange::new(0, 1_000_000))),
        }
    }

    #[test]
    fn missing_file_is_generation_zero() {
        let temp = Temp::new();
        let path = temp.file();
        assert!(!path.exists());
        let mut tool = SettingsTool::open(&path).unwrap();
        assert_eq!(tool.generation(), 0);
        assert_eq!(tool.settings(), &Settings::default());
        assert_eq!(tool.path(), path.as_path());
        assert!(!path.exists());
        let view = tool.get();
        assert_eq!(view.generation, 0);
        assert_eq!(view.settings, Settings::default());
        assert!(!view.settings.model.cloud);
        assert_eq!(tool.undo().unwrap_err(), SettingsToolError::EmptyUndo);
        assert!(!path.exists());
    }

    #[test]
    fn set_advances_generation() {
        let temp = Temp::new();
        let path = temp.file();
        let mut tool = SettingsTool::open(&path).unwrap();
        assert_eq!(tool.generation(), 0);
        let mut settings = tool.settings().clone();
        settings.step_cap = 40;
        settings.playback.loop_playback = true;
        settings.model.cloud = true;
        let generation = tool.set(0, settings).unwrap();
        assert_eq!(generation, 1);
        assert_eq!(tool.generation(), 1);
        assert_eq!(tool.settings().step_cap, 40);
        assert!(tool.settings().playback.loop_playback);
        assert!(tool.settings().model.cloud);
        let text = String::from_utf8(fs::read(&path).unwrap()).unwrap();
        assert!(text.contains("\"generation\":1,"));
        assert!(text.contains("\"step_cap\":40"));
        assert!(text.contains("\"loop_playback\":true"));
        assert!(text.contains("\"cloud\":true"));
        let loaded = SettingsTool::open(&path).unwrap();
        assert_eq!(loaded.generation(), 1);
        assert_eq!(loaded.settings(), tool.settings());
        assert!(loaded.settings().model.cloud);
    }

    #[test]
    fn generation_mismatch_writes_nothing() {
        let temp = Temp::new();
        let path = temp.file();
        let mut tool = SettingsTool::open(&path).unwrap();
        let before: SettingsView = tool.get();
        let mut settings = tool.settings().clone();
        settings.step_cap = 40;
        settings.playback.loop_playback = true;
        let error = tool.set(9, settings).unwrap_err();
        assert_eq!(
            error,
            SettingsToolError::GenerationMismatch {
                expected: 9,
                actual: 0,
            }
        );
        assert!(error.to_string().contains("nothing was written"));
        assert!(std::error::Error::source(&error).is_none());
        assert!(!path.exists());
        assert_eq!(tool.get(), before);
        assert_eq!(tool.generation(), 0);
        assert_eq!(tool.settings().step_cap, Settings::default().step_cap);
    }

    #[test]
    fn undo_settings_restores_the_previous_value() {
        let temp = Temp::new();
        let path = temp.file();
        let mut tool = SettingsTool::open(&path).unwrap();
        assert_eq!(tool.undo().unwrap_err(), SettingsToolError::EmptyUndo);
        assert!(!path.exists());
        let mut written = tool.settings().clone();
        written.step_cap = 40;
        written.playback.loop_playback = true;
        tool.set(0, written).unwrap();
        let generation = tool.undo().unwrap();
        assert_eq!(generation, 2);
        assert_eq!(tool.generation(), 2);
        assert_eq!(tool.settings(), &Settings::default());
        assert!(!tool.settings().model.cloud);
        let text = String::from_utf8(fs::read(&path).unwrap()).unwrap();
        assert!(text.contains("\"generation\":2,"));
        assert!(text.contains("\"step_cap\":32"));
        assert!(text.contains("\"loop_playback\":false"));
        let loaded = SettingsTool::open(&path).unwrap();
        assert_eq!(loaded.generation(), 2);
        assert_eq!(loaded.settings(), &Settings::default());
        let bytes = fs::read(&path).unwrap();
        assert_eq!(tool.undo().unwrap_err(), SettingsToolError::EmptyUndo);
        assert_eq!(fs::read(&path).unwrap(), bytes);
    }

    #[test]
    fn redo_settings_restores_the_written_value() {
        let temp = Temp::new();
        let mut tool = SettingsTool::open(temp.file()).unwrap();
        assert_eq!(tool.redo().unwrap_err(), SettingsToolError::EmptyRedo);
        let mut written = tool.settings().clone();
        written.step_cap = 40;
        written.playback.loop_playback = true;
        tool.set(0, written.clone()).unwrap();
        tool.undo().unwrap();
        assert_eq!(tool.settings(), &Settings::default());
        let generation = tool.redo().unwrap();
        assert_eq!(generation, 3);
        assert_eq!(tool.generation(), 3);
        assert_eq!(tool.settings(), &written);
        assert_eq!(tool.redo().unwrap_err(), SettingsToolError::EmptyRedo);
        let text = String::from_utf8(fs::read(tool.path()).unwrap()).unwrap();
        assert!(text.contains("\"generation\":3,"));
        assert!(text.contains("\"step_cap\":40"));
        assert!(text.contains("\"loop_playback\":true"));

        tool.undo().unwrap();
        let mut next = tool.settings().clone();
        next.step_cap = 7;
        tool.set(tool.generation(), next).unwrap();
        assert_eq!(tool.settings().step_cap, 7);
        assert_eq!(tool.redo().unwrap_err(), SettingsToolError::EmptyRedo);
        tool.undo().unwrap();
        assert_eq!(tool.settings(), &Settings::default());
    }

    #[test]
    fn settings_write_is_not_a_draft_undo_step() {
        let temp = Temp::new();
        let (mut session, track) = session();
        session.apply(0, &[place(&track, "a")]).unwrap();
        assert_eq!(session.revision(), 1);
        let mut tool = SettingsTool::open(temp.file()).unwrap();
        let mut settings = tool.settings().clone();
        settings.step_cap = 40;
        settings.playback.loop_playback = true;
        tool.set(0, settings).unwrap();
        assert_eq!(session.revision(), 1);
        assert_eq!(session.draft().tracks[0].segments.len(), 1);
        assert_eq!(
            session.draft().tracks[0].segments[0].id,
            SegmentId::new("a")
        );
        assert_eq!(tool.settings().step_cap, 40);
        assert!(tool.settings().playback.loop_playback);
        assert!(!tool.settings().model.cloud);
    }

    #[test]
    fn undo_of_the_draft_leaves_settings() {
        let temp = Temp::new();
        let (mut session, track) = session();
        session.apply(0, &[place(&track, "a")]).unwrap();
        let mut tool = SettingsTool::open(temp.file()).unwrap();
        let mut settings = tool.settings().clone();
        settings.step_cap = 40;
        settings.playback.loop_playback = true;
        tool.set(0, settings).unwrap();
        assert_eq!(session.undo().unwrap(), 2);
        assert!(session.draft().tracks[0].segments.is_empty());
        assert_eq!(session.draft().duration, Micros(0));
        assert_eq!(tool.generation(), 1);
        assert_eq!(tool.settings().step_cap, 40);
        assert!(tool.settings().playback.loop_playback);
        let loaded = SettingsTool::open(tool.path()).unwrap();
        assert_eq!(loaded.settings().step_cap, 40);
        assert!(loaded.settings().playback.loop_playback);
    }

    #[test]
    fn undo_of_settings_leaves_the_draft() {
        let temp = Temp::new();
        let (mut session, track) = session();
        let mut tool = SettingsTool::open(temp.file()).unwrap();
        session.apply(0, &[place(&track, "a")]).unwrap();
        let mut settings = tool.settings().clone();
        settings.step_cap = 40;
        settings.playback.loop_playback = true;
        tool.set(tool.generation(), settings).unwrap();
        assert_eq!(session.undo().unwrap(), 2);
        session
            .apply(session.revision(), &[place(&track, "a")])
            .unwrap();
        assert_eq!(session.revision(), 3);
        let revision = session.revision();
        let generation = tool.undo().unwrap();
        assert_eq!(generation, 2);
        assert_eq!(tool.settings(), &Settings::default());
        assert!(!tool.settings().playback.loop_playback);
        assert_eq!(session.revision(), revision);
        assert_eq!(session.draft().tracks[0].segments.len(), 1);
        assert_eq!(
            session.draft().tracks[0].segments[0].id,
            SegmentId::new("a")
        );
        assert_eq!(session.draft().duration, Micros(1_000_000));
    }

    #[test]
    fn hostile_settings_file_is_an_error() {
        let temp = Temp::new();
        let path = temp.file();
        fs::write(&path, b"{").unwrap();
        let before = fs::read(&path).unwrap();
        let error = SettingsTool::open(&path).unwrap_err();
        assert!(matches!(
            error,
            SettingsToolError::File(SettingsError::Json { .. })
        ));
        assert!(error.to_string().contains("could not read"));
        assert!(std::error::Error::source(&error).is_some());
        assert_eq!(fs::read(&path).unwrap(), before);
    }
}
