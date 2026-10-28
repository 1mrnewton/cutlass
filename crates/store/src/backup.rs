//! A recovery copy of the draft file.
//!
//! `.bak` is a byte copy of `draft_info.json` and of no other file. It is not
//! the undo stack.

use std::path::Path;

use crate::StoreError;

const DRAFT_FILE: &str = "draft_info.json";
const BACKUP_FILE: &str = "draft_info.json.bak";

/// Copy `draft_info.json` onto `draft_info.json.bak` beside it.
pub fn copy_draft_backup(draft_path: &Path) -> Result<(), StoreError> {
    let Some(name) = draft_path.file_name().and_then(|name| name.to_str()) else {
        return Err(StoreError::BackupRefused {
            path: draft_path.to_path_buf(),
        });
    };
    if name != DRAFT_FILE {
        return Err(StoreError::BackupRefused {
            path: draft_path.to_path_buf(),
        });
    }
    let bak = draft_path.with_file_name(BACKUP_FILE);
    std::fs::copy(draft_path, &bak).map_err(|error| StoreError::Io {
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

    use super::copy_draft_backup;
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
        fs::write(temp.path.join("project.json"), b"{\"other\":true}\n").unwrap();
        copy_draft_backup(&draft).unwrap();
        assert_eq!(
            fs::read(temp.path.join("draft_info.json.bak")).unwrap(),
            bytes
        );
        assert!(!temp.path.join("project.json.bak").exists());
    }

    #[test]
    fn another_file_is_not_backed_up() {
        let temp = Temp::new();
        let project = temp.path.join("project.json");
        fs::write(&project, b"{}\n").unwrap();
        let error = copy_draft_backup(&project).unwrap_err();
        assert!(matches!(error, StoreError::BackupRefused { .. }));
        assert!(!temp.path.join("project.json.bak").exists());
        assert!(!temp.path.join("draft_info.json.bak").exists());
    }

    #[test]
    fn a_missing_draft_file_fails_and_writes_no_backup() {
        let temp = Temp::new();
        let draft = temp.path.join("draft_info.json");
        let error = copy_draft_backup(&draft).unwrap_err();
        assert!(matches!(error, StoreError::Io { .. }));
        assert!(!temp.path.join("draft_info.json.bak").exists());
    }
}
