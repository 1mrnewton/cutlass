//! The open project folder.
//!
//! Create, open, and save go through the store. A path that leaves the folder
//! is refused.

use std::io;
use std::path::{Component, Path, PathBuf};

use cutlass_session::LinkageSettings;
use cutlass_store::{Draft, Folder, RepairNote, StoreError};

use crate::FfiError;

/// An open project folder.
#[derive(Debug)]
pub struct Project {
    root: PathBuf,
    folder: Folder,
    report: Vec<RepairNote>,
}

impl Project {
    /// Create `root` as an empty project and open it.
    pub fn create(root: &Path, draft_id: &str) -> Result<Self, FfiError> {
        Folder::new(Draft::empty(draft_id)).save(root)?;
        Self::open(root)
    }

    /// Parse, upgrade, and report. Does not push history.
    pub fn open(root: &Path) -> Result<Self, FfiError> {
        let opened = cutlass_store::open(root)?;
        let root = canonicalize(root)?;
        Ok(Self {
            root,
            folder: opened.folder,
            report: opened.report,
        })
    }

    pub fn save(&self) -> Result<(), FfiError> {
        self.folder.save(&self.root)?;
        Ok(())
    }

    /// Canonical project root.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    #[must_use]
    pub fn draft(&self) -> &Draft {
        &self.folder.draft
    }

    #[must_use]
    pub fn report(&self) -> &[RepairNote] {
        &self.report
    }

    #[must_use]
    pub fn main_timeline_id(&self) -> &str {
        &self.folder.project.main_timeline_id
    }

    /// Replace the in-memory cut.
    ///
    /// Also replace `timelines[main id]` when that key exists, so the two copies
    /// stay the same. Does not write disk and does not touch history.
    pub fn set_draft(&mut self, draft: Draft) {
        let main_id = self.main_timeline_id().to_owned();
        if let Some(timeline) = self.folder.timelines.get_mut(&main_id) {
            *timeline = draft.clone();
        }
        self.folder.draft = draft;
    }

    /// Switches from `draft_biz_config.json` via `cutlass_session::linkage_from_biz`.
    ///
    /// `has_linkage_pairs` is any relationship whose `kind` is `linkage`.
    /// The timeline id is this project's main timeline.
    #[must_use]
    pub fn stored_linkage(&self) -> LinkageSettings {
        let has_linkage_pairs = self
            .folder
            .draft
            .relationships
            .iter()
            .any(|relationship| relationship.kind == "linkage");
        cutlass_session::linkage_from_biz(
            self.folder.biz_config.as_ref(),
            self.main_timeline_id(),
            has_linkage_pairs,
        )
    }

    /// A path inside the project. Does not create files and does not read bytes.
    pub fn resolve(&self, relative: &Path) -> Result<PathBuf, FfiError> {
        if leaves_lexically(relative) {
            return Err(path_escapes(relative));
        }
        let joined = self.root.join(relative);
        if path_exists(&joined)? {
            if let Some(canonical) = inside_canonical(&self.root, &joined)? {
                return Ok(canonical);
            }
            return Err(path_escapes(relative));
        }
        let Some(ancestor) = nearest_ancestor(&joined)? else {
            return Err(path_escapes(relative));
        };
        if inside_canonical(&self.root, &ancestor)?.is_some() {
            Ok(self.root.join(relative))
        } else {
            Err(path_escapes(relative))
        }
    }
}

/// `..`, an absolute path, and an empty path are refused before the filesystem is touched.
fn leaves_lexically(relative: &Path) -> bool {
    if relative.as_os_str().is_empty() || relative.is_absolute() {
        return true;
    }
    relative
        .components()
        .any(|component| !matches!(component, Component::Normal(_) | Component::CurDir))
}

fn path_escapes(relative: &Path) -> FfiError {
    FfiError::PathEscapes {
        name: relative.display().to_string(),
    }
}

fn path_exists(path: &Path) -> Result<bool, FfiError> {
    path.try_exists().map_err(|error| io_failure(path, error))
}

fn nearest_ancestor(path: &Path) -> Result<Option<PathBuf>, FfiError> {
    let mut ancestor = path.to_path_buf();
    loop {
        match ancestor.try_exists() {
            Ok(true) => return Ok(Some(ancestor)),
            Ok(false) => {}
            Err(error) => return Err(io_failure(&ancestor, error)),
        }
        let Some(parent) = ancestor.parent() else {
            return Ok(None);
        };
        if parent == ancestor {
            return Ok(None);
        }
        ancestor = parent.to_path_buf();
    }
}

fn inside_canonical(root: &Path, path: &Path) -> Result<Option<PathBuf>, FfiError> {
    let canonical = canonicalize(path)?;
    Ok(canonical.starts_with(root).then_some(canonical))
}

fn canonicalize(path: &Path) -> Result<PathBuf, FfiError> {
    path.canonicalize().map_err(|error| io_failure(path, error))
}

fn io_failure(path: &Path, error: io::Error) -> FfiError {
    FfiError::Store(StoreError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    use cutlass_store::{StoreError, TimelineEntry};

    use super::Project;
    use crate::FfiError;

    struct Temp {
        path: PathBuf,
    }

    impl Temp {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("cutlass-ffi-project-{}-{n}", std::process::id()));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn open_empty(temp: &Temp) -> Project {
        Project::create(&temp.path.join("project"), "D").unwrap()
    }

    #[test]
    fn create_writes_an_empty_project() {
        let temp = Temp::new();
        let root = temp.path.join("project");
        let project = Project::create(&root, "D").unwrap();
        assert_eq!(project.root(), root.canonicalize().unwrap());
        assert!(project.root().join("draft_info.json").is_file());
        assert!(project.root().join("project.json").is_file());
        assert!(
            project
                .root()
                .join("Timelines")
                .join("D")
                .join("draft_info.json")
                .is_file()
        );
        let draft = project.draft();
        assert_eq!(draft.id.as_str(), "D");
        assert_eq!(project.main_timeline_id(), "D");
        assert_eq!(draft.tracks.len(), 1);
        assert!(draft.tracks[0].is_main());
        assert_eq!(draft.duration, cutlass_session::Micros(0));
        assert_eq!(draft.version, 360_000);
        assert_eq!(draft.new_version, "189.0.0");
        assert!(draft.config.maintrack_adsorb);
        assert!(project.report().is_empty());
        assert!(!project.stored_linkage().enabled);
    }

    #[test]
    fn open_reads_the_same_draft_id() {
        let temp = Temp::new();
        let project = open_empty(&temp);
        let opened = Project::open(project.root()).unwrap();
        assert_eq!(opened.draft().id.as_str(), "D");
        assert_eq!(opened.main_timeline_id(), "D");
    }

    #[test]
    fn a_renamed_draft_survives_save_and_open() {
        let temp = Temp::new();
        let mut project = open_empty(&temp);
        let before = fs::read(project.root().join("draft_info.json")).unwrap();
        let mut draft = project.draft().clone();
        draft.name = "Renamed".to_owned();
        project.set_draft(draft);
        assert_eq!(
            fs::read(project.root().join("draft_info.json")).unwrap(),
            before
        );
        let main_id = project.main_timeline_id().to_owned();
        assert_eq!(project.draft().name, "Renamed");
        assert_eq!(
            project
                .folder
                .timelines
                .get(&main_id)
                .map(|draft| draft.name.as_str()),
            Some("Renamed")
        );
        project.save().unwrap();
        let opened = Project::open(project.root()).unwrap();
        assert_eq!(opened.draft().name, "Renamed");
        assert_eq!(opened.draft().id.as_str(), "D");
    }

    #[test]
    fn resolve_returns_a_file_inside_the_project() {
        let temp = Temp::new();
        let project = open_empty(&temp);
        let resolved = project.resolve(Path::new("draft_info.json")).unwrap();
        assert!(resolved.is_file());
        assert!(resolved.starts_with(project.root()));
        assert_eq!(
            resolved,
            project
                .root()
                .join("draft_info.json")
                .canonicalize()
                .unwrap()
        );
        let missing = Path::new("brand-new/file.bin");
        let resolved = project.resolve(missing).unwrap();
        assert_eq!(resolved, project.root().join(missing));
        assert!(!resolved.exists());
        assert!(!project.root().join("brand-new").exists());
    }

    #[test]
    fn resolve_refuses_a_path_that_leaves_the_project() {
        let temp = Temp::new();
        let project = open_empty(&temp);
        let secret = temp.path.join("secret");
        let bytes = b"known-secret";
        fs::write(&secret, bytes).unwrap();
        let error = project.resolve(Path::new("../secret")).unwrap_err();
        assert!(
            matches!(&error, FfiError::PathEscapes { name } if name == "../secret"),
            "{error:?}"
        );
        assert_eq!(fs::read(&secret).unwrap(), bytes);
        let absolute = secret.canonicalize().unwrap();
        let error = project.resolve(&absolute).unwrap_err();
        assert!(
            matches!(&error, FfiError::PathEscapes { name } if name == &absolute.display().to_string()),
            "{error:?}"
        );
        assert_eq!(fs::read(&secret).unwrap(), bytes);
        let draft = fs::read(project.root().join("draft_info.json")).unwrap();
        let error = project
            .resolve(Path::new("Timelines/../draft_info.json"))
            .unwrap_err();
        assert!(
            matches!(
                &error,
                FfiError::PathEscapes { name } if name == "Timelines/../draft_info.json"
            ),
            "{error:?}"
        );
        assert_eq!(
            fs::read(project.root().join("draft_info.json")).unwrap(),
            draft
        );
        let error = project.resolve(Path::new("")).unwrap_err();
        assert!(
            matches!(&error, FfiError::PathEscapes { name } if name.is_empty()),
            "{error:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn resolve_refuses_a_symlink_that_leaves_the_project() {
        let temp = Temp::new();
        let project = open_empty(&temp);
        let secret = temp.path.join("secret");
        let bytes = b"known-secret";
        fs::write(&secret, bytes).unwrap();
        std::os::unix::fs::symlink(&secret, project.root().join("leak")).unwrap();
        let error = project.resolve(Path::new("leak")).unwrap_err();
        assert!(
            matches!(&error, FfiError::PathEscapes { name } if name == "leak"),
            "{error:?}"
        );
        assert_eq!(fs::read(&secret).unwrap(), bytes);
    }

    #[test]
    fn save_refuses_a_timeline_id_that_leaves_the_folder() {
        let temp = Temp::new();
        let mut project = open_empty(&temp);
        project.folder.project.timelines.push(TimelineEntry {
            id: "../outside".to_owned(),
            ..Default::default()
        });
        let error = project.save().unwrap_err();
        assert!(
            matches!(
                &error,
                FfiError::Store(StoreError::PathEscapes { name }) if name == "../outside"
            ),
            "{error:?}"
        );
        assert!(!project.root().with_file_name("outside").exists());
        assert!(!temp.path.join("outside").exists());
    }

    #[test]
    fn create_refuses_a_draft_id_that_leaves_the_folder() {
        let temp = Temp::new();
        let root = temp.path.join("project");
        let error = Project::create(&root, "../x").unwrap_err();
        assert!(
            matches!(
                &error,
                FfiError::Store(StoreError::PathEscapes { name }) if name == "../x"
            ),
            "{error:?}"
        );
        assert!(!root.exists());
        assert!(!temp.path.join("x").exists());
    }

    #[test]
    fn open_of_a_missing_directory_is_an_error() {
        let temp = Temp::new();
        let missing = temp.path.join("missing");
        let error = Project::open(&missing).unwrap_err();
        assert!(
            matches!(&error, FfiError::Store(StoreError::MissingDraft { .. })),
            "{error:?}"
        );
        assert!(!missing.exists());
    }

    #[test]
    fn linkage_pairs_without_a_biz_file_enable_linkage() {
        let temp = Temp::new();
        let mut project = open_empty(&temp);
        assert!(!project.root().join("draft_biz_config.json").exists());
        let mut draft = project.draft().clone();
        draft.relationships.push(Default::default());
        draft.relationships.last_mut().unwrap().kind = "linkage".to_owned();
        project.set_draft(draft);
        assert!(project.stored_linkage().enabled);
        assert!(!project.root().join("draft_biz_config.json").exists());
    }
}
