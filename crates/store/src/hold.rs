//! The exclusive hold on `.locked`.
//!
//! Open reads the folder. This hold is the separate step that keeps a second
//! process out. The file stores a socket address. The socket framing is not
//! open yet, so the address is null. The exclusive lock is what "held" means.
//! Dropping the hold releases the lock and leaves the file in place.

use std::fs::{File, OpenOptions, TryLockError};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::StoreError;

const LOCK_FILE: &str = ".locked";
const ABSENT_ADDRESS: &[u8] = b"{\"address\":null}\n";

/// A process that currently holds `.locked`.
#[derive(Debug)]
pub struct Hold {
    file: File,
    path: PathBuf,
}

impl Hold {
    /// The lock file. It stays inside the project folder.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for Hold {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

/// Take `.locked` in `path`. A folder that is already held is refused.
pub fn hold(path: &Path) -> Result<Hold, StoreError> {
    let root = path.canonicalize().map_err(|error| StoreError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    if !root.is_dir() {
        return Err(StoreError::Io {
            path: root,
            message: "the project path is not a folder".to_owned(),
        });
    }
    let lock_path = root.join(LOCK_FILE);
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .map_err(|error| StoreError::Io {
            path: lock_path.clone(),
            message: error.to_string(),
        })?;
    let canonical = lock_path.canonicalize().map_err(|error| StoreError::Io {
        path: lock_path.clone(),
        message: error.to_string(),
    })?;
    if !canonical.starts_with(&root) {
        return Err(StoreError::PathEscapes {
            name: LOCK_FILE.to_owned(),
        });
    }
    match file.try_lock() {
        Ok(()) => {}
        Err(TryLockError::WouldBlock) => {
            return Err(StoreError::Held { path: root });
        }
        Err(TryLockError::Error(error)) => {
            return Err(StoreError::Io {
                path: lock_path,
                message: error.to_string(),
            });
        }
    }
    file.set_len(0).map_err(|error| StoreError::Io {
        path: lock_path.clone(),
        message: error.to_string(),
    })?;
    file.seek(SeekFrom::Start(0))
        .map_err(|error| StoreError::Io {
            path: lock_path.clone(),
            message: error.to_string(),
        })?;
    file.write_all(ABSENT_ADDRESS)
        .map_err(|error| StoreError::Io {
            path: lock_path.clone(),
            message: error.to_string(),
        })?;
    Ok(Hold {
        file,
        path: lock_path,
    })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    struct Temp {
        path: PathBuf,
    }

    impl Temp {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("cutlass-store-hold-{}-{n}", std::process::id()));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn a_second_hold_on_the_same_folder_is_refused() {
        let temp = Temp::new();
        let root = temp.path().join("project");
        fs::create_dir(&root).unwrap();
        let first = hold(&root).unwrap();
        let canonical = root.canonicalize().unwrap();
        assert_eq!(first.path(), canonical.join(".locked"));
        let error = hold(&root).unwrap_err();
        let StoreError::Held { path } = error else {
            panic!("held");
        };
        assert_eq!(path, root.canonicalize().unwrap());
        let message = StoreError::Held { path: path.clone() }.to_string();
        assert!(message.contains("already held"));
        assert_eq!(fs::read(first.path()).unwrap(), b"{\"address\":null}\n");
        drop(first);
        let second = hold(&root).unwrap();
        assert_eq!(fs::read(second.path()).unwrap(), b"{\"address\":null}\n");
    }

    #[test]
    fn a_lock_file_that_leaves_the_folder_is_refused() {
        let temp = Temp::new();
        let root = temp.path().join("project");
        let outside = temp.path().join("outside");
        fs::create_dir(&root).unwrap();
        fs::create_dir(&outside).unwrap();
        let external = outside.join("held");
        fs::write(&external, b"keep").unwrap();
        std::os::unix::fs::symlink(&external, root.join(".locked")).unwrap();
        let error = hold(&root).unwrap_err();
        assert!(matches!(error, StoreError::PathEscapes { .. }));
        assert_eq!(fs::read(&external).unwrap(), b"keep");
    }

    #[test]
    fn a_missing_folder_is_named() {
        let temp = Temp::new();
        let missing = temp.path().join("gone");
        let error = hold(&missing).unwrap_err();
        let StoreError::Io { path, .. } = error else {
            panic!("io");
        };
        assert_eq!(path, missing);
    }
}
