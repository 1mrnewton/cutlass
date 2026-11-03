//! Installed faces for text clips, shared by every shell.
//!
//! The draft stores a face's file name. Evaluation does not open that path;
//! this table supplies the bytes when the name matches a face it knows. A
//! shell names the files it trusts ([`FaceTable::load_from`]) or the folders
//! its system keeps fonts in ([`FaceTable::scan`]). A scanned face is read
//! the first time a frame asks for it, so a long font folder costs little.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use cutlass_eval::Fonts;

/// The most faces a scan keeps.
pub const MAX_FACES: usize = 400;
/// The largest face file a scan keeps, in bytes.
pub const MAX_FACE_BYTES: u64 = 40 * 1024 * 1024;
/// Faces new text should start in, best first. A scan puts the first of
/// these it finds ahead of the rest. Helvetica only counts as a single-face
/// file; the system's collection is left out like every collection.
pub const PREFERRED_FACES: &[&str] = &[
    "Helvetica.ttf",
    "HelveticaNeue.ttf",
    "Arial.ttf",
    "SFNS.ttf",
];
/// How many folders deep a scan looks below each folder it is given.
const MAX_DEPTH: usize = 3;

enum Bytes {
    Read(Vec<u8>),
    /// Read on the first ask. `None` once it failed to read as one face.
    Later(PathBuf, OnceLock<Option<Vec<u8>>>),
}

/// Faces by file name. An empty table supplies nothing.
#[derive(Default)]
pub struct FaceTable {
    faces: Vec<(String, Bytes)>,
}

impl FaceTable {
    /// Every path that is a single readable face, in that order, read now.
    /// A second file with the same name is skipped.
    #[must_use]
    pub fn load_from<P: AsRef<Path>>(paths: &[P]) -> Self {
        let mut table = Self::default();
        for path in paths {
            let path = path.as_ref();
            let Ok(bytes) = std::fs::read(path) else {
                continue;
            };
            if cutlass_font::read(&bytes).is_err() {
                continue;
            }
            if let Some(name) = file_name(path) {
                table.push(name, Bytes::Read(bytes));
            }
        }
        table
    }

    /// Faces given as bytes, in order. Bytes that are not one face are kept
    /// out.
    #[must_use]
    pub fn from_pairs(faces: Vec<(String, Vec<u8>)>) -> Self {
        let mut table = Self::default();
        for (name, bytes) in faces {
            if cutlass_font::read(&bytes).is_ok() {
                table.push(name, Bytes::Read(bytes));
            }
        }
        table
    }

    /// One face given as bytes.
    #[must_use]
    pub fn named(name: impl Into<String>, bytes: Vec<u8>) -> Self {
        Self::from_pairs(vec![(name.into(), bytes)])
    }

    /// The single-face TrueType and OpenType files in `folders` and the
    /// folders below them, by name. Collections are left out. At most
    /// [`MAX_FACES`] are kept; a file over [`MAX_FACE_BYTES`] is skipped.
    /// Only each file's first bytes are read now. The first of
    /// [`PREFERRED_FACES`] found comes first; the rest go by name.
    #[must_use]
    pub fn scan<P: AsRef<Path>>(folders: &[P]) -> Self {
        let mut found = Vec::new();
        for folder in folders {
            walk(folder.as_ref(), 0, &mut found);
        }
        found.sort_by(|a: &(String, PathBuf), b| a.0.cmp(&b.0));
        if let Some(at) = PREFERRED_FACES
            .iter()
            .find_map(|preferred| found.iter().position(|(name, _)| name == preferred))
        {
            let face = found.remove(at);
            found.insert(0, face);
        }
        let mut table = Self::default();
        for (name, path) in found {
            if table.faces.len() == MAX_FACES {
                break;
            }
            table.push(name, Bytes::Later(path, OnceLock::new()));
        }
        table
    }

    /// The first face, which new text starts in. An empty table has an
    /// empty name.
    #[must_use]
    pub fn name(&self) -> &str {
        self.faces.first().map_or("", |(name, _)| name.as_str())
    }

    /// File names, in order.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.faces.iter().map(|(name, _)| name.clone()).collect()
    }

    fn push(&mut self, name: String, bytes: Bytes) {
        if name.is_empty() || self.faces.iter().any(|(have, _)| have == &name) {
            return;
        }
        self.faces.push((name, bytes));
    }
}

impl Fonts for FaceTable {
    fn face(&self, path: &str) -> Option<&[u8]> {
        let (_, bytes) = self.faces.iter().find(|(name, _)| name == path)?;
        match bytes {
            Bytes::Read(bytes) => Some(bytes),
            Bytes::Later(path, cell) => cell
                .get_or_init(|| {
                    let bytes = std::fs::read(path).ok()?;
                    cutlass_font::read(&bytes).ok()?;
                    Some(bytes)
                })
                .as_deref(),
        }
    }
}

/// A face's name for people: the file name without its extension, with
/// dashes and underscores as spaces.
#[must_use]
pub fn face_label(name: &str) -> String {
    let stem = Path::new(name)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(name);
    stem.replace(['-', '_'], " ")
}

fn walk(folder: &Path, depth: usize, found: &mut Vec<(String, PathBuf)>) {
    let Ok(entries) = std::fs::read_dir(folder) else {
        return;
    };
    for entry in entries.flatten() {
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        let path = entry.path();
        if kind.is_dir() {
            if depth < MAX_DEPTH {
                walk(&path, depth + 1, found);
            }
        } else if kind.is_file()
            && single_face(&path)
            && let Some(name) = file_name(&path)
        {
            found.push((name, path));
        }
    }
}

/// A `.ttf` or `.otf` file under the size cap whose first bytes name one
/// TrueType or OpenType face.
fn single_face(path: &Path) -> bool {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase);
    if !matches!(extension.as_deref(), Some("ttf" | "otf")) {
        return false;
    }
    let Ok(mut file) = File::open(path) else {
        return false;
    };
    if file
        .metadata()
        .map_or(true, |meta| meta.len() > MAX_FACE_BYTES)
    {
        return false;
    }
    let mut tag = [0_u8; 4];
    if file.read_exact(&mut tag).is_err() {
        return false;
    }
    matches!(&tag, [0, 1, 0, 0] | b"OTTO" | b"true")
}

fn file_name(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    (!name.contains(['/', '\\', '\0'])).then(|| name.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &[u8] = include_bytes!("../../font/fixtures/sample.ttf");

    struct Folder(PathBuf);

    impl Folder {
        fn new(tag: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "cutlass-faces-{tag}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("folder");
            Self(path)
        }
    }

    impl Drop for Folder {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn named_files_are_read_now_and_junk_is_kept_out() {
        let folder = Folder::new("load");
        let face = folder.0.join("Face.ttf");
        let junk = folder.0.join("Junk.ttf");
        std::fs::write(&face, SAMPLE).unwrap();
        std::fs::write(&junk, b"not a font").unwrap();
        let table = FaceTable::load_from(&[&face, &junk, &face]);
        assert_eq!(table.names(), ["Face.ttf"]);
        assert_eq!(table.name(), "Face.ttf");
        assert_eq!(table.face("Face.ttf"), Some(SAMPLE));
        assert!(table.face("Junk.ttf").is_none());

        let empty = FaceTable::load_from::<&Path>(&[]);
        assert!(empty.name().is_empty());
        assert!(
            FaceTable::named("x.ttf", b"junk".to_vec())
                .names()
                .is_empty()
        );
    }

    #[test]
    fn a_scan_finds_single_faces_below_a_folder_and_reads_them_when_asked() {
        let folder = Folder::new("scan");
        let deep = folder.0.join("a/b");
        std::fs::create_dir_all(&deep).unwrap();
        std::fs::write(folder.0.join("Zed.TTF"), SAMPLE).unwrap();
        std::fs::write(deep.join("Alpha.otf"), SAMPLE).unwrap();
        // A collection, a fake face and another kind of file stay out.
        std::fs::write(folder.0.join("Both.ttc"), SAMPLE).unwrap();
        std::fs::write(folder.0.join("Fake.ttf"), b"wOFFnope").unwrap();
        std::fs::write(folder.0.join("notes.txt"), SAMPLE).unwrap();
        let too_deep = folder.0.join("1/2/3/4");
        std::fs::create_dir_all(&too_deep).unwrap();
        std::fs::write(too_deep.join("Lost.ttf"), SAMPLE).unwrap();

        let table = FaceTable::scan(&[&folder.0, &folder.0.join("missing")]);
        assert_eq!(table.names(), ["Alpha.otf", "Zed.TTF"]);
        assert_eq!(table.face("Alpha.otf"), Some(SAMPLE));
        assert!(table.face("Lost.ttf").is_none());
    }

    #[test]
    fn a_scan_puts_the_best_preferred_face_first() {
        let folder = Folder::new("preferred");
        for name in ["Zed.ttf", "Arial.ttf", "Alpha.otf"] {
            std::fs::write(folder.0.join(name), SAMPLE).unwrap();
        }
        let table = FaceTable::scan(&[&folder.0]);
        assert_eq!(table.names(), ["Arial.ttf", "Alpha.otf", "Zed.ttf"]);
        assert_eq!(table.name(), "Arial.ttf");

        std::fs::write(folder.0.join("Helvetica.ttf"), SAMPLE).unwrap();
        // A collection never wins, even under a preferred stem.
        std::fs::write(folder.0.join("HelveticaNeue.ttc"), SAMPLE).unwrap();
        let table = FaceTable::scan(&[&folder.0]);
        assert_eq!(
            table.names(),
            ["Helvetica.ttf", "Alpha.otf", "Arial.ttf", "Zed.ttf"]
        );

        std::fs::remove_file(folder.0.join("Helvetica.ttf")).unwrap();
        std::fs::remove_file(folder.0.join("Arial.ttf")).unwrap();
        assert_eq!(FaceTable::scan(&[&folder.0]).name(), "Alpha.otf");
    }

    #[test]
    fn a_scanned_face_that_turns_out_bad_supplies_nothing() {
        let folder = Folder::new("bad");
        let mut header = vec![0, 1, 0, 0];
        header.extend_from_slice(b"truncated");
        std::fs::write(folder.0.join("Broken.ttf"), &header).unwrap();
        let table = FaceTable::scan(&[&folder.0]);
        assert_eq!(table.names(), ["Broken.ttf"]);
        assert!(table.face("Broken.ttf").is_none());
        assert!(table.face("Broken.ttf").is_none());
    }

    #[test]
    fn a_label_is_the_name_without_its_extension() {
        assert_eq!(face_label("DINCondensed-bold.ttf"), "DINCondensed bold");
        assert_eq!(face_label("Courier_New.otf"), "Courier New");
        assert_eq!(face_label("plain"), "plain");
    }
}
