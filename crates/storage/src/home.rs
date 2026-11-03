//! The one folder everything Cutlass writes lives under.
//!
//! A shell names the root and nothing else. The projects, the cache, the
//! downloads, the proxies, the models, the logs, and the settings file are
//! laid out here, the same on every platform. Bundled resources are not under
//! the root: they ship read-only inside the app package.

use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

use crate::StorageError;

const PROJECTS: &str = "Projects";
const CACHE: &str = "Cache";
const DOWNLOADS: &str = "Downloads";
const PROXIES: &str = "Proxies";
const MODELS: &str = "Models";
const LOGS: &str = "Logs";
const SETTINGS: &str = "settings.json";

/// The Cutlass root and the folders under it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Home {
    root: PathBuf,
}

impl Home {
    /// Name the layout under `root` without touching the disk.
    ///
    /// The root must be absolute, so the folders do not move with the
    /// working directory.
    pub fn at(root: impl Into<PathBuf>) -> Result<Self, StorageError> {
        let root = root.into();
        if !root.is_absolute() {
            return Err(StorageError::NotAbsolute { path: root });
        }
        Ok(Self { root })
    }

    /// Name the layout under `root` and create every folder in it.
    ///
    /// The settings file is not written. A root that is a file, or a folder
    /// under it that is a file, is refused.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, StorageError> {
        let home = Self::at(root)?;
        for folder in home.folders() {
            fs::create_dir_all(&folder).map_err(|error| {
                if folder.exists() && !folder.is_dir() {
                    StorageError::NotADirectory {
                        path: folder.clone(),
                    }
                } else {
                    StorageError::Io {
                        path: folder.clone(),
                        message: error.to_string(),
                    }
                }
            })?;
        }
        Ok(home)
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// One folder per project.
    #[must_use]
    pub fn projects(&self) -> PathBuf {
        self.root.join(PROJECTS)
    }

    /// Previews, thumbnails, and the index. Eviction may empty it.
    #[must_use]
    pub fn cache(&self) -> PathBuf {
        self.root.join(CACHE)
    }

    /// Stock footage and packages the user kept. Never evicted.
    #[must_use]
    pub fn downloads(&self) -> PathBuf {
        self.root.join(DOWNLOADS)
    }

    /// Smaller copies of source media for editing.
    #[must_use]
    pub fn proxies(&self) -> PathBuf {
        self.root.join(PROXIES)
    }

    /// Speech and other model weights.
    #[must_use]
    pub fn models(&self) -> PathBuf {
        self.root.join(MODELS)
    }

    #[must_use]
    pub fn logs(&self) -> PathBuf {
        self.root.join(LOGS)
    }

    #[must_use]
    pub fn settings_file(&self) -> PathBuf {
        self.root.join(SETTINGS)
    }

    fn folders(&self) -> [PathBuf; 6] {
        [
            self.projects(),
            self.cache(),
            self.downloads(),
            self.proxies(),
            self.models(),
            self.logs(),
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Host {
    Mac,
    Windows,
    Other,
}

const fn current_host() -> Host {
    if cfg!(target_os = "macos") {
        Host::Mac
    } else if cfg!(target_os = "windows") {
        Host::Windows
    } else {
        Host::Other
    }
}

/// The desktop's private root when the user has not chosen one.
///
/// macOS: `~/Library/Application Support/Cutlass`. Windows:
/// `%APPDATA%\Cutlass`. Elsewhere: `$XDG_DATA_HOME/cutlass`, else
/// `~/.local/share/cutlass`. `None` when the variables it needs are unset.
/// A phone passes its app container instead.
#[must_use]
pub fn desktop_root() -> Option<PathBuf> {
    root_from(
        current_host(),
        variable("HOME").as_deref(),
        variable("APPDATA").as_deref(),
        variable("XDG_DATA_HOME").as_deref(),
    )
}

fn root_from(
    host: Host,
    home: Option<&Path>,
    app_data: Option<&Path>,
    xdg_data: Option<&Path>,
) -> Option<PathBuf> {
    let root = match host {
        Host::Mac => home?
            .join("Library")
            .join("Application Support")
            .join("Cutlass"),
        Host::Windows => match app_data {
            Some(app_data) => app_data.join("Cutlass"),
            None => home?.join("AppData").join("Roaming").join("Cutlass"),
        },
        Host::Other => match xdg_data {
            Some(xdg) => xdg.join("cutlass"),
            None => home?.join(".local").join("share").join("cutlass"),
        },
    };
    root.is_absolute().then_some(root)
}

fn variable(name: &str) -> Option<PathBuf> {
    env::var_os(name)
        .filter(|value: &OsString| !value.is_empty())
        .map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::{Home, Host, desktop_root, root_from};
    use crate::StorageError;

    struct Temp(PathBuf);

    impl Temp {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("cutlass-home-{}-{n}", std::process::id()));
            let _ = fs::remove_dir_all(&path);
            Self(path)
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn every_folder_is_under_the_one_root() {
        let home = Home::at("/data/cutlass").unwrap();
        let root = Path::new("/data/cutlass");
        assert_eq!(home.root(), root);
        assert_eq!(home.projects(), root.join("Projects"));
        assert_eq!(home.cache(), root.join("Cache"));
        assert_eq!(home.downloads(), root.join("Downloads"));
        assert_eq!(home.proxies(), root.join("Proxies"));
        assert_eq!(home.models(), root.join("Models"));
        assert_eq!(home.logs(), root.join("Logs"));
        assert_eq!(home.settings_file(), root.join("settings.json"));
    }

    #[test]
    fn naming_the_layout_does_not_create_it() {
        let temp = Temp::new();
        Home::at(&temp.0).unwrap();
        assert!(!temp.0.exists());
    }

    #[test]
    fn open_creates_every_folder_and_opens_again() {
        let temp = Temp::new();
        let home = Home::open(&temp.0).unwrap();
        for folder in home.folders() {
            assert!(folder.is_dir(), "{}", folder.display());
        }
        assert!(!home.settings_file().exists());
        fs::write(home.projects().join("kept"), b"x").unwrap();
        let again = Home::open(&temp.0).unwrap();
        assert_eq!(again, home);
        assert!(again.projects().join("kept").exists());
    }

    #[test]
    fn a_relative_root_is_refused() {
        let Err(error) = Home::at("Cutlass") else {
            panic!("a relative root opened");
        };
        assert!(matches!(error, StorageError::NotAbsolute { .. }));
        assert!(Home::open("").is_err());
    }

    #[test]
    fn a_root_or_a_folder_that_is_a_file_is_refused() {
        let temp = Temp::new();
        fs::create_dir_all(&temp.0).unwrap();
        let file = temp.0.join("root");
        fs::write(&file, b"x").unwrap();
        assert!(Home::open(&file).is_err());

        let root = temp.0.join("home");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("Cache"), b"x").unwrap();
        let Err(error) = Home::open(&root) else {
            panic!("a file named Cache opened");
        };
        assert!(matches!(error, StorageError::NotADirectory { .. }));
    }

    #[test]
    fn removing_the_root_and_opening_again_starts_empty() {
        let temp = Temp::new();
        let home = Home::open(&temp.0).unwrap();
        fs::write(home.cache().join("old"), b"x").unwrap();
        fs::remove_dir_all(&temp.0).unwrap();
        let home = Home::open(&temp.0).unwrap();
        assert!(home.cache().is_dir());
        assert!(!home.cache().join("old").exists());
    }

    #[test]
    fn each_desktop_has_its_private_root() {
        let home = Path::new("/Users/someone");
        assert_eq!(
            root_from(Host::Mac, Some(home), None, None).unwrap(),
            home.join("Library/Application Support/Cutlass")
        );
        assert_eq!(root_from(Host::Mac, None, None, None), None);
        let roaming = Path::new("/roaming");
        assert_eq!(
            root_from(Host::Windows, Some(home), Some(roaming), None).unwrap(),
            roaming.join("Cutlass")
        );
        assert_eq!(
            root_from(Host::Windows, Some(home), None, None).unwrap(),
            home.join("AppData/Roaming/Cutlass")
        );
        let xdg = Path::new("/xdg/data");
        assert_eq!(
            root_from(Host::Other, Some(home), None, Some(xdg)).unwrap(),
            xdg.join("cutlass")
        );
        assert_eq!(
            root_from(Host::Other, Some(home), None, None).unwrap(),
            home.join(".local/share/cutlass")
        );
    }

    #[test]
    fn a_relative_variable_does_not_become_a_root() {
        assert_eq!(
            root_from(Host::Other, None, None, Some(Path::new("data"))),
            None
        );
        assert_eq!(
            root_from(Host::Mac, Some(Path::new("someone")), None, None),
            None
        );
    }

    #[test]
    fn this_desktop_root_is_absolute_when_known() {
        if let Some(root) = desktop_root() {
            assert!(root.is_absolute());
            assert!(Home::at(root).is_ok());
        }
    }
}
