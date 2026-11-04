//! Writes that a crash cannot leave half done.
//!
//! The bytes go to a temporary file beside the target, are flushed to the
//! disk, and then take the target's name in one rename. A crash leaves the
//! old file or the new one, never a torn one.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

/// Replace `path` with `bytes` in one step.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "no file name"))?;
    let temp = path.with_file_name(format!(".{name}.saving"));
    let written = (|| {
        let mut file = File::create(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&temp, path)
    })();
    if written.is_err() {
        let _ = fs::remove_file(&temp);
    }
    written?;
    sync_parent(path);
    Ok(())
}

/// Flush the folder entry for the rename where the system allows opening
/// a folder; elsewhere the rename alone stands.
fn sync_parent(path: &Path) {
    if let Some(parent) = path.parent()
        && let Ok(folder) = File::open(parent)
    {
        let _ = folder.sync_all();
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::write_atomic;

    #[test]
    fn a_write_replaces_the_file_and_leaves_no_temporary_behind() {
        let dir = std::env::temp_dir().join(format!("cutlass-atomic-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("draft_info.json");
        fs::write(&path, b"old").unwrap();
        write_atomic(&path, b"new").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"new");
        let names: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(names, ["draft_info.json"]);
        fs::remove_dir_all(&dir).unwrap();
    }
}
