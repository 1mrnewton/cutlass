//! Per-user, platform-correct locations for Cutlass's writable data.
//!
//! On Windows the app installs into `C:\Program Files\Cutlass` (read-only for
//! normal users) and its shortcut launches with that as the working directory,
//! so any *relative* path or any `$HOME`-derived path (`HOME` is unset on
//! Windows) lands in a folder the app can't write to — the historic "Cutlass
//! crashes on launch unless you run as administrator" bug.
//!
//! This helper resolves to the OS-blessed per-user data directory instead,
//! which is always writable without elevation:
//!
//! | role  | Windows    | macOS                           | Linux (XDG)      |
//! |-------|------------|---------------------------------|------------------|
//! | data  | `%APPDATA%`| `~/Library/Application Support` | `~/.local/share` |
//!
//! Callers create the directory lazily (the draft store `create_dir_all`s its
//! target), so this function only computes a path.

use std::path::PathBuf;

/// Application folder nested under the OS data root.
const APP_DIR: &str = "Cutlass";

/// Writable **data** root for things the user would miss if they vanished:
/// the app-owned project drafts (`<os-data>/Cutlass`, see [`crate::drafts`]).
pub fn data_dir() -> PathBuf {
    app_root(dirs::data_dir())
}

/// `<base>/Cutlass`, where `base` is the OS dir when known, else the user's
/// home, else the temp dir. Never the working directory — on Windows that is
/// the read-only install folder, the very thing this module exists to avoid.
fn app_root(base: Option<PathBuf>) -> PathBuf {
    base.or_else(dirs::home_dir)
        .unwrap_or_else(std::env::temp_dir)
        .join(APP_DIR)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_dir_is_absolute_and_namespaced() {
        let data = data_dir();
        assert!(data.is_absolute(), "data dir must be absolute: {data:?}");
        assert!(data.ends_with("Cutlass"));
    }
}
