//! Recovery copies beside the project files.
//!
//! `draft_info.json.bak` is a byte copy of `draft_info.json`.
//! `project.json.bak` is a byte copy of `project.json`. No other name is
//! copied. Neither file is the undo stack.

use std::path::Path;

use crate::StoreError;

const BACKED_UP: &[&str] = &["draft_info.json", "project.json"];

/// Copy `draft_info.json` or `project.json` onto the `.bak` file beside it.
pub fn copy_backup(path: &Path) -> Result<(), StoreError> {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return Err(StoreError::BackupRefused {
            path: path.to_path_buf(),
        });
    };
    if !BACKED_UP.contains(&name) {
        return Err(StoreError::BackupRefused {
            path: path.to_path_buf(),
        });
    }
    let bak = path.with_file_name(format!("{name}.bak"));
    std::fs::copy(path, &bak).map_err(|error| StoreError::Io {
        path: bak,
        message: error.to_string(),
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::copy_backup;
    use crate::StoreError;

    struct Temp {
        path: PathBuf,
    }

    impl Temp {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("cutlass-store-{}-{n}", std::process::id()));
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
    fn the_backup_is_a_byte_copy_of_the_draft_file() {
        let temp = Temp::new();
        let draft = temp.path.join("draft_info.json");
        let bytes = b"{\"id\":\"D\"}\n";
        fs::write(&draft, bytes).unwrap();
        copy_backup(&draft).unwrap();
        assert_eq!(
            fs::read(temp.path.join("draft_info.json.bak")).unwrap(),
            bytes
        );
    }

    #[test]
    fn the_project_file_backup_is_a_byte_copy() {
        let temp = Temp::new();
        let project = temp.path.join("project.json");
        let bytes = b"{\"main_timeline_id\":\"D\"}\n";
        fs::write(&project, bytes).unwrap();
        copy_backup(&project).unwrap();
        assert_eq!(fs::read(temp.path.join("project.json.bak")).unwrap(), bytes);
        assert!(!temp.path.join("draft_info.json.bak").exists());
    }

    #[test]
    fn another_file_is_not_backed_up() {
        let temp = Temp::new();
        let notes = temp.path.join("notes.json");
        fs::write(&notes, b"{}\n").unwrap();
        let error = copy_backup(&notes).unwrap_err();
        assert!(matches!(error, StoreError::BackupRefused { .. }));
        assert!(!temp.path.join("notes.json.bak").exists());
        assert!(!temp.path.join("draft_info.json.bak").exists());
        assert!(!temp.path.join("project.json.bak").exists());
    }

    #[test]
    fn a_missing_draft_file_fails_and_writes_no_backup() {
        let temp = Temp::new();
        let draft = temp.path.join("draft_info.json");
        let error = copy_backup(&draft).unwrap_err();
        assert!(matches!(error, StoreError::Io { .. }));
        assert!(!temp.path.join("draft_info.json.bak").exists());
    }
}
