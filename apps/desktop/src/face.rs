//! The faces the desktop offers for text clips.
//!
//! The table itself is shared with the phones; this module says where a
//! desktop keeps its fonts: the system's folders, the person's own, the
//! `Fonts` folder of the Cutlass home, and a `fonts` folder shipped beside
//! the app. Only names are read at the start; a face's file is read when a
//! frame first asks for it.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

pub use cutlass_playback::FaceTable;

/// The faces new text starts in, best first: the first one found is the
/// default. These draw Latin with kerning and ligatures from OpenType
/// tables on each system.
const DEFAULT_FACES: &[&str] = &[
    "Arial.ttf",
    "arial.ttf",
    "Geneva.ttf",
    "NewYork.ttf",
    "DejaVuSans.ttf",
    "LiberationSans-Regular.ttf",
];

/// The folders fonts are looked for in, on this system.
fn folders() -> Vec<PathBuf> {
    let mut folders = Vec::new();
    let home = std::env::var_os("HOME").map(PathBuf::from);
    if cfg!(target_os = "macos") {
        folders.push(PathBuf::from("/System/Library/Fonts"));
        folders.push(PathBuf::from("/Library/Fonts"));
        folders.extend(home.map(|home| home.join("Library/Fonts")));
    } else if cfg!(target_os = "windows") {
        let windows =
            std::env::var_os("WINDIR").map_or_else(|| PathBuf::from("C:\\Windows"), PathBuf::from);
        folders.push(windows.join("Fonts"));
        folders.extend(
            std::env::var_os("LOCALAPPDATA")
                .map(|local| PathBuf::from(local).join("Microsoft\\Windows\\Fonts")),
        );
    } else {
        folders.push(PathBuf::from("/usr/share/fonts"));
        folders.push(PathBuf::from("/usr/local/share/fonts"));
        if let Some(home) = home {
            folders.push(home.join(".fonts"));
            folders.push(home.join(".local/share/fonts"));
        }
    }
    folders.push(cutlass_storage::desktop_home().fonts());
    // Fonts shipped with the app: beside the program, or in a bundle's
    // resources.
    if let Ok(program) = std::env::current_exe()
        && let Some(beside) = program.parent()
    {
        folders.push(beside.join("fonts"));
        folders.push(beside.join("../Resources/fonts"));
    }
    folders
}

/// The faces found on this system, looked for once.
#[must_use]
pub fn installed() -> Arc<FaceTable> {
    static FACES: OnceLock<Arc<FaceTable>> = OnceLock::new();
    Arc::clone(
        FACES.get_or_init(|| Arc::new(FaceTable::scan(&folders()).preferring(DEFAULT_FACES))),
    )
}

#[cfg(test)]
mod tests {
    use super::{folders, installed};

    #[test]
    fn the_folders_hold_the_homes_own_and_the_apps() {
        let folders = folders();
        assert!(folders.contains(&cutlass_storage::desktop_home().fonts()));
        assert!(folders.iter().any(|folder| folder.ends_with("fonts")));
        assert!(folders.len() >= 4);
    }

    #[test]
    fn the_faces_are_looked_for_once() {
        let first = installed();
        let again = installed();
        assert!(std::sync::Arc::ptr_eq(&first, &again));
        // Whatever the system has, the default is one of its faces.
        assert_eq!(first.name().is_empty(), first.names().is_empty());
    }

    /// How long the scan takes and what it finds, on this machine.
    #[test]
    #[ignore = "prints what this machine's fonts give"]
    fn what_this_machine_gives() {
        let started = std::time::Instant::now();
        let table = cutlass_playback::FaceTable::scan(&folders()).preferring(super::DEFAULT_FACES);
        let took = started.elapsed();
        let names = table.names();
        let line = format!(
            "{} faces in {took:?}; default {}; fallbacks {:?}\n",
            names.len(),
            table.name(),
            table.fallback_names()
        );
        std::io::Write::write_all(&mut std::io::stderr(), line.as_bytes()).expect("stderr");
        // A fallback face is read when a character first needs it.
        use cutlass_eval::Fonts as _;
        for (character, emoji) in [('\u{1F600}', true), ('\u{4E2D}', false), ('\u{915}', false)] {
            let started = std::time::Instant::now();
            let size = table.fallback(character, emoji).map(<[u8]>::len);
            let line = format!(
                "U+{:04X}: {size:?} bytes in {:?}\n",
                u32::from(character),
                started.elapsed()
            );
            std::io::Write::write_all(&mut std::io::stderr(), line.as_bytes()).expect("stderr");
        }
    }
}
