//! The writable cache directory.
//!
//! A relative path stays inside the root it was joined to. A symlink is
//! refused. The index file is `index.json` in this directory and nowhere else.

use std::path::{Component, Path, PathBuf};

use cutlass_download::hash_file;

use crate::StorageError;
use crate::index::{Index, Place, Snapshot};

const INDEX_FILE: &str = "index.json";
const INDEX_TMP: &str = "index.json.tmp";

pub(crate) fn directory(path: &Path) -> Result<PathBuf, StorageError> {
    let metadata = std::fs::metadata(path).map_err(|error| StorageError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    if !metadata.is_dir() {
        return Err(StorageError::NotADirectory {
            path: path.to_path_buf(),
        });
    }
    std::fs::canonicalize(path).map_err(|error| StorageError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })
}

pub(crate) fn ensure_cache(path: &Path) -> Result<PathBuf, StorageError> {
    std::fs::create_dir_all(path).map_err(|error| StorageError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let root = directory(path)?;
    std::fs::create_dir_all(root.join("files")).map_err(|error| StorageError::Io {
        path: root.join("files"),
        message: error.to_string(),
    })?;
    std::fs::create_dir_all(root.join("incoming")).map_err(|error| StorageError::Io {
        path: root.join("incoming"),
        message: error.to_string(),
    })?;
    Ok(root)
}

/// Join `relative` onto `root`. `..`, an absolute path, and a symlink are refused.
pub(crate) fn resolve_file(
    root: &Path,
    relative: &Path,
    must_exist: bool,
) -> Result<PathBuf, StorageError> {
    if relative.as_os_str().is_empty() || relative.is_absolute() {
        return Err(StorageError::LeavesRoot {
            path: relative.to_path_buf(),
        });
    }
    let mut current = root.to_path_buf();
    let components: Vec<_> = relative.components().collect();
    if components.is_empty() {
        return Err(StorageError::LeavesRoot {
            path: relative.to_path_buf(),
        });
    }
    for (index, component) in components.iter().enumerate() {
        let Component::Normal(name) = component else {
            return Err(StorageError::LeavesRoot {
                path: relative.to_path_buf(),
            });
        };
        current.push(name);
        let last = index + 1 == components.len();
        match std::fs::symlink_metadata(&current) {
            Ok(meta) if meta.file_type().is_symlink() => {
                return Err(StorageError::Symlink { path: current });
            }
            Ok(meta) if last && meta.is_file() => {}
            Ok(meta) if !last && meta.is_dir() => {}
            Ok(_) => return Err(StorageError::NotAFile { path: current }),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound && last && !must_exist => {}
            Err(error) => {
                return Err(StorageError::Io {
                    path: current,
                    message: error.to_string(),
                });
            }
        }
    }
    if !current.starts_with(root) {
        return Err(StorageError::LeavesRoot {
            path: relative.to_path_buf(),
        });
    }
    Ok(current)
}

pub(crate) fn load(bundle_root: &Path, cache_root: &Path) -> Result<Index, StorageError> {
    let path = cache_root.join(INDEX_FILE);
    let bytes = match std::fs::symlink_metadata(&path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(Index::empty());
        }
        Err(error) => {
            return Err(StorageError::Io {
                path,
                message: error.to_string(),
            });
        }
        Ok(meta) if meta.file_type().is_symlink() => {
            return Err(StorageError::Symlink { path });
        }
        Ok(meta) if !meta.is_file() => {
            return Err(StorageError::NotAFile { path });
        }
        Ok(_) => std::fs::read(&path).map_err(|error| StorageError::Io {
            path: path.clone(),
            message: error.to_string(),
        })?,
    };
    let snapshot: Snapshot =
        serde_json::from_slice(&bytes).map_err(|error| StorageError::Index {
            path: path.clone(),
            message: error.to_string(),
        })?;
    let index = Index::from_snapshot(snapshot).map_err(|error| match error {
        StorageError::Index { message, .. } => StorageError::Index {
            path: path.clone(),
            message,
        },
        other => other,
    })?;
    for record in index.records() {
        let root = match record.place() {
            Place::Bundle => bundle_root,
            Place::Cache => cache_root,
        };
        let full = resolve_file(root, record.relative(), true)?;
        let size = std::fs::metadata(&full)
            .map_err(|error| StorageError::Io {
                path: full.clone(),
                message: error.to_string(),
            })?
            .len();
        if size != record.size() {
            return Err(StorageError::Index {
                path: path.clone(),
                message: format!(
                    "{} is {size} bytes, and the index says {}",
                    record.id(),
                    record.size()
                ),
            });
        }
        if record.is_partial() {
            continue;
        }
        if let Some(expected) = record.checksum() {
            let (checksum, hashed) = hash_file(&full).map_err(|error| StorageError::Io {
                path: full.clone(),
                message: error.to_string(),
            })?;
            if hashed != size || checksum != expected {
                return Err(StorageError::ChecksumMismatch {
                    id: record.id().clone(),
                });
            }
        }
    }
    Ok(index)
}

pub(crate) fn persist(cache_root: &Path, index: &Index) -> Result<(), StorageError> {
    let path = cache_root.join(INDEX_FILE);
    let temporary = cache_root.join(INDEX_TMP);
    let mut bytes = serde_json::to_vec(&index.snapshot()).map_err(|error| StorageError::Index {
        path: path.clone(),
        message: error.to_string(),
    })?;
    bytes.push(b'\n');
    std::fs::write(&temporary, &bytes).map_err(|error| StorageError::Io {
        path: temporary.clone(),
        message: error.to_string(),
    })?;
    replace_file(&temporary, &path)
}

fn replace_file(from: &Path, to: &Path) -> Result<(), StorageError> {
    match std::fs::rename(from, to) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            std::fs::remove_file(to).map_err(|error| StorageError::Io {
                path: to.to_path_buf(),
                message: error.to_string(),
            })?;
            std::fs::rename(from, to).map_err(|error| StorageError::Io {
                path: to.to_path_buf(),
                message: error.to_string(),
            })
        }
        Err(error) => Err(StorageError::Io {
            path: to.to_path_buf(),
            message: error.to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{ensure_cache, resolve_file};
    use crate::StorageError;
    use crate::Temp;

    #[test]
    fn a_relative_file_stays_in_the_root() {
        let temp = Temp::new("cache-path");
        let root = ensure_cache(temp.path()).unwrap();
        let file = root.join("files").join("glow");
        fs::write(&file, b"glow").unwrap();
        let resolved = resolve_file(&root, std::path::Path::new("files/glow"), true).unwrap();
        assert_eq!(resolved, file);
        let missing = resolve_file(&root, std::path::Path::new("files/next"), false).unwrap();
        assert_eq!(missing, root.join("files").join("next"));
        assert!(!missing.exists());
    }

    #[test]
    fn a_path_that_leaves_the_root_is_refused() {
        let temp = Temp::new("cache-escape");
        let root = ensure_cache(temp.path()).unwrap();
        let outside = temp.path().join("secret");
        fs::write(&outside, b"secret").unwrap();
        for relative in ["../secret", "files/../../secret", "/etc/passwd"] {
            let error = resolve_file(&root, std::path::Path::new(relative), true).unwrap_err();
            assert!(
                matches!(
                    error,
                    StorageError::LeavesRoot { .. } | StorageError::Io { .. }
                ),
                "{relative}: {error}"
            );
        }
        assert_eq!(fs::read(&outside).unwrap(), b"secret");
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_that_leaves_the_root_is_refused() {
        let temp = Temp::new("cache-link");
        let root = ensure_cache(temp.path()).unwrap();
        let outside = temp.path().join("secret");
        fs::write(&outside, b"secret").unwrap();
        let link = root.join("files").join("link");
        std::os::unix::fs::symlink(&outside, &link).unwrap();
        let error = resolve_file(&root, std::path::Path::new("files/link"), true).unwrap_err();
        assert!(matches!(error, StorageError::Symlink { .. }));
        assert_eq!(fs::read(&outside).unwrap(), b"secret");
    }
}
