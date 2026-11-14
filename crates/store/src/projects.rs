//! The projects directory: the launch list, a new empty project, and a
//! project renamed, copied or deleted.
//!
//! Each project is a folder under one root. The desktop and the phones list
//! and create them the same way.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use cutlass_model::{Draft, Micros};

use crate::folder::DRAFT_FILE;
use crate::{Folder, StoreError};

/// One saved project in the launch list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ListedProject {
    pub path: PathBuf,
    pub name: String,
    pub duration: Micros,
    pub modified: SystemTime,
}

/// When the project at `path` was last saved: its draft file's time. Other
/// files the app keeps in the folder, such as a card picture, leave it be,
/// where the folder's own time would move.
#[must_use]
pub fn saved_at(path: &Path) -> Option<SystemTime> {
    fs::metadata(path.join(DRAFT_FILE))
        .and_then(|metadata| metadata.modified())
        .ok()
}

/// The card for one project folder, or `None` when it holds no draft.
#[must_use]
pub fn listed_project(path: &Path) -> Option<ListedProject> {
    if !path.is_dir() {
        return None;
    }
    let folder = Folder::read(path).ok()?;
    let modified = saved_at(path).unwrap_or(UNIX_EPOCH);
    let name = if folder.draft.name.is_empty() {
        path.file_name()?.to_string_lossy().into_owned()
    } else {
        folder.draft.name
    };
    Some(ListedProject {
        path: path.to_path_buf(),
        name,
        duration: folder.draft.duration,
        modified,
    })
}

/// Projects under `root`, newest first. A folder without a readable draft
/// is skipped, and a missing root lists nothing.
#[must_use]
pub fn list_projects(root: &Path) -> Vec<ListedProject> {
    let Ok(entries) = fs::read_dir(root) else {
        return Vec::new();
    };
    let mut projects = Vec::new();
    for entry in entries.flatten() {
        if let Some(project) = listed_project(&entry.path()) {
            projects.push(project);
        }
    }
    projects.sort_by(|left, right| {
        right
            .modified
            .cmp(&left.modified)
            .then_with(|| left.name.cmp(&right.name))
    });
    projects
}

/// Write a new empty project under `root` and return its folder.
///
/// The first free name of "Project", "Project 2", … is taken. Linkage starts on.
pub fn create_project(root: &Path) -> Result<PathBuf, StoreError> {
    fs::create_dir_all(root).map_err(|error| StoreError::Io {
        path: root.to_path_buf(),
        message: error.to_string(),
    })?;
    let (name, path, id) = next_slot(root);
    let mut draft = Draft::empty(id);
    draft.name = name;
    let now = unix_millis();
    draft.create_time = now;
    draft.update_time = now;
    let mut folder = Folder::new(draft);
    folder.set_linkage_enabled(true);
    folder.save(&path)?;
    Ok(path)
}

/// Give the project at `path` the name `name`, trimmed. Its folder keeps
/// its own name, so nothing that points into it moves.
pub fn rename_project(path: &Path, name: &str) -> Result<(), StoreError> {
    let name = name.trim();
    if name.is_empty() {
        return Err(StoreError::EmptyName);
    }
    let mut folder = Folder::read(path)?;
    if folder.draft.name == name {
        return Ok(());
    }
    folder.draft.name = name.to_owned();
    folder.draft.update_time = unix_millis();
    folder.save(path)
}

/// Copy the project at `path` beside it, named "… copy" (then "… copy
/// 2", …) in a new folder, and return the new folder. Its media is copied
/// too: a project keeps its files inside its folder.
pub fn duplicate_project(path: &Path) -> Result<PathBuf, StoreError> {
    let folder = Folder::read(path)?;
    let root = path.parent().ok_or_else(|| StoreError::Io {
        path: path.to_path_buf(),
        message: "the project has no parent folder".to_owned(),
    })?;
    let stem = format!("{} copy", folder.draft.name);
    let (name, copy) = (1..100_000)
        .map(|index| {
            let name = if index == 1 {
                stem.clone()
            } else {
                format!("{stem} {index}")
            };
            let copy = root.join(&name);
            (name, copy)
        })
        .find(|(_, copy)| !copy.exists())
        .ok_or_else(|| StoreError::Io {
            path: root.to_path_buf(),
            message: "no free name for the copy".to_owned(),
        })?;
    if let Err(error) = copy_tree(path, &copy) {
        let _ = fs::remove_dir_all(&copy);
        return Err(error);
    }
    let mut copied = match Folder::read(&copy) {
        Ok(copied) => copied,
        Err(error) => {
            let _ = fs::remove_dir_all(&copy);
            return Err(error);
        }
    };
    copied.draft.name = name;
    let now = unix_millis();
    copied.draft.create_time = now;
    copied.draft.update_time = now;
    if let Err(error) = copied.save(&copy) {
        let _ = fs::remove_dir_all(&copy);
        return Err(error);
    }
    Ok(copy)
}

/// Every file and folder under `from` into a new `to`. A link is not
/// followed: it would reach outside the project. The hold on the folder
/// stays with it.
fn copy_tree(from: &Path, to: &Path) -> Result<(), StoreError> {
    let io = |path: &Path, error: std::io::Error| StoreError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    };
    fs::create_dir(to).map_err(|error| io(to, error))?;
    for entry in fs::read_dir(from).map_err(|error| io(from, error))? {
        let entry = entry.map_err(|error| io(from, error))?;
        let kind = entry
            .file_type()
            .map_err(|error| io(&entry.path(), error))?;
        if entry.file_name() == crate::hold::LOCK_FILE {
            continue;
        }
        let target = to.join(entry.file_name());
        if kind.is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else if kind.is_file() {
            fs::copy(entry.path(), &target).map_err(|error| io(&target, error))?;
        }
    }
    Ok(())
}

/// Delete the project at `path` and everything in its folder. A folder
/// that holds no project is refused, so nothing else is removed.
pub fn delete_project(path: &Path) -> Result<(), StoreError> {
    if listed_project(path).is_none() {
        return Err(StoreError::MissingDraft {
            path: path.join("draft_info.json"),
        });
    }
    fs::remove_dir_all(path).map_err(|error| StoreError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })
}

fn next_slot(root: &Path) -> (String, PathBuf, String) {
    for index in 1..100_000 {
        let name = if index == 1 {
            "Project".to_owned()
        } else {
            format!("Project {index}")
        };
        let path = root.join(&name);
        if !path.exists() {
            return (name, path, format!("project-{index}"));
        }
    }
    let stamp = unix_millis();
    let name = format!("Project {stamp}");
    let path = root.join(&name);
    (name, path, format!("project-{stamp}"))
}

fn unix_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| i64::try_from(duration.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Duration, SystemTime};

    use super::{
        create_project, delete_project, duplicate_project, list_projects, listed_project,
        rename_project, saved_at,
    };
    use crate::Folder;

    struct Temp {
        path: PathBuf,
    }

    impl Temp {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("cutlass-store-projects-{}-{n}", std::process::id()));
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
    fn a_missing_root_lists_nothing_and_is_not_created() {
        let temp = Temp::new();
        let missing = temp.path.join("missing");
        assert!(list_projects(&missing).is_empty());
        assert!(!missing.exists());
    }

    #[test]
    fn new_projects_take_the_next_free_name_and_are_listed() {
        let temp = Temp::new();
        let root = temp.path.join("Projects");
        let first = create_project(&root).unwrap();
        let second = create_project(&root).unwrap();
        assert!(first.ends_with("Project"));
        assert!(second.ends_with("Project 2"));
        let folder = Folder::read(&second).unwrap();
        assert_eq!(folder.draft.name, "Project 2");
        assert_eq!(folder.draft.id, "project-2".into());
        assert!(folder.draft.create_time > 0);
        let mut names: Vec<_> = list_projects(&root)
            .into_iter()
            .map(|project| project.name)
            .collect();
        names.sort();
        assert_eq!(names, ["Project", "Project 2"]);
    }

    #[test]
    fn a_file_or_a_folder_without_a_draft_is_skipped_and_its_name_is_not_reused() {
        let temp = Temp::new();
        fs::write(temp.path.join("Project"), b"not a folder").unwrap();
        fs::create_dir(temp.path.join("notes")).unwrap();
        fs::create_dir(temp.path.join("Project 2")).unwrap();
        fs::write(temp.path.join("Project 2/draft_info.json"), b"{ not json").unwrap();
        assert!(list_projects(&temp.path).is_empty());
        let made = create_project(&temp.path).unwrap();
        assert!(made.ends_with("Project 3"));
        assert_eq!(list_projects(&temp.path).len(), 1);
    }

    #[test]
    fn a_project_dates_from_its_last_save_not_from_other_files_in_its_folder() {
        let temp = Temp::new();
        let project = create_project(&temp.path).unwrap();
        let saved = saved_at(&project).unwrap();
        assert_eq!(listed_project(&project).unwrap().modified, saved);
        // A card picture written beside the draft, as the app does through
        // a file it renames into place, moves the folder's time only.
        std::thread::sleep(Duration::from_millis(20));
        fs::write(project.join(".picture.jpg.part"), b"picture").unwrap();
        fs::rename(
            project.join(".picture.jpg.part"),
            project.join("picture.jpg"),
        )
        .unwrap();
        assert_ne!(fs::metadata(&project).unwrap().modified().unwrap(), saved);
        assert_eq!(listed_project(&project).unwrap().modified, saved);
        // A save moves it.
        std::thread::sleep(Duration::from_millis(20));
        rename_project(&project, "Renamed").unwrap();
        assert!(listed_project(&project).unwrap().modified > saved);
        assert_eq!(saved_at(&temp.path.join("missing")), None);
    }

    #[test]
    fn the_newest_project_is_listed_first() {
        let temp = Temp::new();
        let older = create_project(&temp.path).unwrap();
        let newer = create_project(&temp.path).unwrap();
        let past = SystemTime::now() - Duration::from_secs(3_600);
        fs::File::open(&older)
            .and_then(|folder| folder.set_modified(past))
            .unwrap();
        let listed = list_projects(&temp.path);
        assert_eq!(listed[0].path, newer);
        assert_eq!(listed[1].path, older);
    }

    #[test]
    fn a_project_is_renamed_copied_with_its_media_and_deleted() {
        let temp = Temp::new();
        let first = create_project(&temp.path).unwrap();
        fs::create_dir_all(first.join("Resources")).unwrap();
        fs::write(first.join("Resources/clip.mp4"), b"movie").unwrap();
        fs::write(first.join(".locked"), b"{}").unwrap();

        rename_project(&first, "  Holiday  ").unwrap();
        assert_eq!(Folder::read(&first).unwrap().draft.name, "Holiday");
        assert!(first.ends_with("Project"), "the folder keeps its name");
        assert!(matches!(
            rename_project(&first, "  "),
            Err(crate::StoreError::EmptyName)
        ));

        let copy = duplicate_project(&first).unwrap();
        assert!(copy.ends_with("Holiday copy"));
        assert_eq!(Folder::read(&copy).unwrap().draft.name, "Holiday copy");
        assert_eq!(fs::read(copy.join("Resources/clip.mp4")).unwrap(), b"movie");
        assert!(!copy.join(".locked").exists());
        let again = duplicate_project(&first).unwrap();
        assert!(again.ends_with("Holiday copy 2"));
        assert_eq!(list_projects(&temp.path).len(), 3);

        delete_project(&copy).unwrap();
        assert!(!copy.exists());
        assert_eq!(list_projects(&temp.path).len(), 2);
        // A folder with no project in it is left alone.
        let notes = temp.path.join("notes");
        fs::create_dir(&notes).unwrap();
        assert!(delete_project(&notes).is_err());
        assert!(notes.exists());
    }
}
