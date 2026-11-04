//! The projects directory: the launch list and a new empty project.
//!
//! Each project is a folder under one root. The desktop and the phones list
//! and create them the same way.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use cutlass_model::{Draft, Micros};

use crate::{Folder, StoreError};

/// One saved project in the launch list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ListedProject {
    pub path: PathBuf,
    pub name: String,
    pub duration: Micros,
    pub modified: SystemTime,
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
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Ok(folder) = Folder::read(&path) else {
            continue;
        };
        let modified = entry
            .metadata()
            .and_then(|metadata| metadata.modified())
            .unwrap_or(UNIX_EPOCH);
        let name = if folder.draft.name.is_empty() {
            entry.file_name().to_string_lossy().into_owned()
        } else {
            folder.draft.name
        };
        projects.push(ListedProject {
            path,
            name,
            duration: folder.draft.duration,
            modified,
        });
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

    use super::{create_project, list_projects};
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
}
