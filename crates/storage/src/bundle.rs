//! The read-only resources directory.
//!
//! A bundled file is indexed from the package. It is not copied into the
//! cache, and this module does not write the package.

use std::path::{Component, Path, PathBuf};

use cutlass_download::{Checksum, hash_file};

use crate::StorageError;
use crate::cache::resolve_file;
use crate::index::{Index, Place, Record, ResourceId};

pub(crate) fn mount(
    index: &mut Index,
    bundle_root: &Path,
    id: &ResourceId,
    relative: &Path,
    checksum: Checksum,
) -> Result<bool, StorageError> {
    let stored = stored_relative(relative)?;
    let full = resolve_file(bundle_root, Path::new(&stored), true)?;
    let (found, size) = hash_file(&full).map_err(|error| StorageError::Io {
        path: full,
        message: error.to_string(),
    })?;
    if found != checksum {
        return Err(StorageError::ChecksumMismatch { id: id.clone() });
    }
    if let Some(existing) = index.get(id) {
        let same = existing.place() == Place::Bundle
            && existing.relative() == Path::new(&stored)
            && existing.checksum() == Some(checksum)
            && !existing.is_partial()
            && !existing.is_preview();
        if same {
            return Ok(false);
        }
        return Err(StorageError::AlreadyStored { id: id.clone() });
    }
    index.insert(Record::bundle(
        id.clone(),
        PathBuf::from(stored),
        size,
        checksum,
    ));
    Ok(true)
}

fn stored_relative(relative: &Path) -> Result<String, StorageError> {
    if relative.as_os_str().is_empty() || relative.is_absolute() {
        return Err(StorageError::LeavesRoot {
            path: relative.to_path_buf(),
        });
    }
    let mut parts = Vec::new();
    for component in relative.components() {
        match component {
            Component::Normal(name) => {
                let Some(text) = name.to_str() else {
                    return Err(StorageError::LeavesRoot {
                        path: relative.to_path_buf(),
                    });
                };
                if text.is_empty() || text == "." || text == ".." {
                    return Err(StorageError::LeavesRoot {
                        path: relative.to_path_buf(),
                    });
                }
                parts.push(text);
            }
            _ => {
                return Err(StorageError::LeavesRoot {
                    path: relative.to_path_buf(),
                });
            }
        }
    }
    if parts.is_empty() {
        return Err(StorageError::LeavesRoot {
            path: relative.to_path_buf(),
        });
    }
    Ok(parts.join("/"))
}
