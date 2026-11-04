//! Installed faces for text clips, shared by every shell.
//!
//! The draft stores a face's file name. Evaluation does not open that path;
//! this table supplies the bytes when the name matches a face it knows. A
//! shell names the files it trusts ([`FaceTable::load_from`]) or the folders
//! its system keeps fonts in ([`FaceTable::scan`]). A scanned face is read
//! the first time a frame asks for it, so a long font folder costs little.
//!
//! A face of a collection is named `File.ttc#n`, face `n` of that file, and
//! labelled with its full name. Frames get it copied out as its own font.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use cutlass_eval::Fonts;

/// The most faces a scan keeps.
pub const MAX_FACES: usize = 400;
/// The largest face file a scan keeps, in bytes.
pub const MAX_FACE_BYTES: u64 = 40 * 1024 * 1024;
/// Faces new text should start in, best first. A scan puts the first of
/// these it finds ahead of the rest.
pub const PREFERRED_FACES: &[&str] = &[
    "Helvetica.ttf",
    "Helvetica.ttc#0",
    "HelveticaNeue.ttf",
    "HelveticaNeue.ttc#0",
    "Arial.ttf",
    "SFNS.ttf",
];
/// How many folders deep a scan looks below each folder it is given.
const MAX_DEPTH: usize = 3;

enum Bytes {
    Read(Vec<u8>),
    /// Read on the first ask, face `index` copied out of a collection. `None`
    /// once it failed to read as one face.
    Later(PathBuf, Option<u32>, OnceLock<Option<Vec<u8>>>),
}

struct Face {
    name: String,
    label: String,
    bytes: Bytes,
}

/// A face found by a scan, before the table keeps it.
struct Found {
    name: String,
    label: String,
    path: PathBuf,
    index: Option<u32>,
}

/// Faces by name. An empty table supplies nothing.
#[derive(Default)]
pub struct FaceTable {
    faces: Vec<Face>,
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
                table.push(name.clone(), face_label(&name), Bytes::Read(bytes));
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
                table.push(name.clone(), face_label(&name), Bytes::Read(bytes));
            }
        }
        table
    }

    /// One face given as bytes.
    #[must_use]
    pub fn named(name: impl Into<String>, bytes: Vec<u8>) -> Self {
        Self::from_pairs(vec![(name.into(), bytes)])
    }

    /// The TrueType and OpenType faces in `folders` and the folders below
    /// them, each face of a collection on its own, by label. At most
    /// [`MAX_FACES`] are kept; a file over [`MAX_FACE_BYTES`] is skipped.
    /// Only headers and names are read now. The first of
    /// [`PREFERRED_FACES`] found comes first.
    #[must_use]
    pub fn scan<P: AsRef<Path>>(folders: &[P]) -> Self {
        let mut found = Vec::new();
        for folder in folders {
            walk(folder.as_ref(), 0, &mut found);
        }
        found.sort_by(|a, b| {
            a.label
                .to_lowercase()
                .cmp(&b.label.to_lowercase())
                .then_with(|| a.name.cmp(&b.name))
        });
        if let Some(at) = PREFERRED_FACES
            .iter()
            .find_map(|preferred| found.iter().position(|face| face.name == *preferred))
        {
            let face = found.remove(at);
            found.insert(0, face);
        }
        let mut table = Self::default();
        for face in found {
            if table.faces.len() == MAX_FACES {
                break;
            }
            let bytes = Bytes::Later(face.path, face.index, OnceLock::new());
            table.push(face.name, face.label, bytes);
        }
        table
    }

    /// The first face, which new text starts in. An empty table has an
    /// empty name.
    #[must_use]
    pub fn name(&self) -> &str {
        self.faces.first().map_or("", |face| face.name.as_str())
    }

    /// Names, in order.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.faces.iter().map(|face| face.name.clone()).collect()
    }

    /// Names with their labels for people, in order.
    #[must_use]
    pub fn labelled(&self) -> Vec<(String, String)> {
        self.faces
            .iter()
            .map(|face| (face.name.clone(), face.label.clone()))
            .collect()
    }

    fn push(&mut self, name: String, label: String, bytes: Bytes) {
        if name.is_empty() || self.faces.iter().any(|face| face.name == name) {
            return;
        }
        self.faces.push(Face { name, label, bytes });
    }
}

impl Fonts for FaceTable {
    fn face(&self, path: &str) -> Option<&[u8]> {
        let face = self.faces.iter().find(|face| face.name == path)?;
        match &face.bytes {
            Bytes::Read(bytes) => Some(bytes),
            Bytes::Later(path, index, cell) => cell
                .get_or_init(|| {
                    let bytes = read_capped(path)?;
                    match index {
                        None => {
                            cutlass_font::read(&bytes).ok()?;
                            Some(bytes)
                        }
                        Some(index) => cutlass_font::single_face(&bytes, *index).ok(),
                    }
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

fn walk(folder: &Path, depth: usize, found: &mut Vec<Found>) {
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
        } else if kind.is_file() {
            faces_in(&path, found);
        }
    }
}

/// The faces of a `.ttf`, `.otf` or `.ttc` file under the size cap. A file
/// that does not read as fonts adds nothing, and neither does a hidden file
/// or a face whose name starts with a dot, which systems keep for their own
/// interface.
fn faces_in(path: &Path, found: &mut Vec<Found>) {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase);
    if !matches!(extension.as_deref(), Some("ttf" | "otf" | "ttc")) {
        return;
    }
    let Some(name) = file_name(path).filter(|name| !name.starts_with('.')) else {
        return;
    };
    let Ok(mut file) = File::open(path) else {
        return;
    };
    if file
        .metadata()
        .map_or(true, |meta| meta.len() > MAX_FACE_BYTES)
    {
        return;
    }
    let Ok(faces) = cutlass_font::list_faces(|offset, len| {
        file.seek(SeekFrom::Start(offset)).ok()?;
        let mut bytes = vec![0; len];
        file.read_exact(&mut bytes).ok()?;
        Some(bytes)
    }) else {
        return;
    };
    for face in faces {
        if face
            .full_name
            .as_deref()
            .is_some_and(|full| full.starts_with('.'))
        {
            continue;
        }
        let (name, label) = match face.index {
            None => (name.clone(), face_label(&name)),
            Some(index) => (
                format!("{name}#{index}"),
                face.full_name
                    .unwrap_or_else(|| format!("{} {}", face_label(&name), index + 1)),
            ),
        };
        found.push(Found {
            name,
            label,
            path: path.to_owned(),
            index: face.index,
        });
    }
}

/// The whole file when it is under the size cap; it may have grown since the
/// scan.
fn read_capped(path: &Path) -> Option<Vec<u8>> {
    let file = File::open(path).ok()?;
    if file.metadata().ok()?.len() > MAX_FACE_BYTES {
        return None;
    }
    let mut bytes = Vec::new();
    file.take(MAX_FACE_BYTES + 1).read_to_end(&mut bytes).ok()?;
    (bytes.len() as u64 <= MAX_FACE_BYTES).then_some(bytes)
}

fn file_name(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    (!name.contains(['/', '\\', '\0'])).then(|| name.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &[u8] = include_bytes!("../../font/fixtures/sample.ttf");
    const SAMPLE_TTC: &[u8] = include_bytes!("../../font/fixtures/sample.ttc");

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
        // A fake face and another kind of file stay out.
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
    fn each_face_of_a_collection_is_listed_by_its_full_name_and_draws_alone() {
        let folder = Folder::new("ttc");
        std::fs::write(folder.0.join("Zed.ttf"), SAMPLE).unwrap();
        std::fs::write(folder.0.join("Both.ttc"), SAMPLE_TTC).unwrap();
        let table = FaceTable::scan(&[&folder.0]);
        assert_eq!(table.names(), ["Both.ttc#0", "Both.ttc#1", "Zed.ttf"]);
        assert_eq!(table.labelled()[0].1, "Cutlass Sample");
        assert_eq!(table.labelled()[2].1, "Zed");
        for name in ["Both.ttc#0", "Both.ttc#1"] {
            let bytes = table.face(name).expect("copied out");
            assert!(cutlass_font::read(bytes).is_ok(), "{name}");
        }
        // The file name alone or a face past the end names nothing.
        assert!(table.face("Both.ttc").is_none());
        assert!(table.face("Both.ttc#2").is_none());
    }

    #[test]
    fn hidden_files_and_dot_named_system_faces_stay_out() {
        let folder = Folder::new("hidden");
        std::fs::write(folder.0.join(".Hidden.ttf"), SAMPLE).unwrap();
        let mut private = SAMPLE_TTC.to_vec();
        let utf16: Vec<u8> = "Cutlass"
            .encode_utf16()
            .flat_map(u16::to_be_bytes)
            .collect();
        let dotted: Vec<u8> = ".utlass"
            .encode_utf16()
            .flat_map(u16::to_be_bytes)
            .collect();
        for (from, to) in [(b"Cutlass".to_vec(), b".utlass".to_vec()), (utf16, dotted)] {
            while let Some(at) = private.windows(from.len()).position(|w| w == from) {
                private[at..at + to.len()].copy_from_slice(&to);
            }
        }
        std::fs::write(folder.0.join("System.ttc"), &private).unwrap();
        std::fs::write(folder.0.join("Shown.ttf"), SAMPLE).unwrap();
        assert_eq!(FaceTable::scan(&[&folder.0]).names(), ["Shown.ttf"]);
    }

    #[test]
    fn a_collection_that_changes_after_the_scan_supplies_nothing() {
        let folder = Folder::new("ttc-gone");
        let path = folder.0.join("Both.ttc");
        std::fs::write(&path, SAMPLE_TTC).unwrap();
        let table = FaceTable::scan(&[&folder.0]);
        std::fs::write(&path, &SAMPLE_TTC[..64]).unwrap();
        assert!(table.face("Both.ttc#1").is_none());
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

        // The first face of a preferred collection beats a later choice.
        std::fs::write(folder.0.join("HelveticaNeue.ttc"), SAMPLE_TTC).unwrap();
        assert_eq!(FaceTable::scan(&[&folder.0]).name(), "HelveticaNeue.ttc#0");

        std::fs::write(folder.0.join("Helvetica.ttf"), SAMPLE).unwrap();
        let table = FaceTable::scan(&[&folder.0]);
        assert_eq!(table.name(), "Helvetica.ttf");
        assert_eq!(table.names().len(), 6);

        for name in ["Helvetica.ttf", "Arial.ttf", "HelveticaNeue.ttc"] {
            std::fs::remove_file(folder.0.join(name)).unwrap();
        }
        assert_eq!(FaceTable::scan(&[&folder.0]).name(), "Alpha.otf");
    }

    #[test]
    fn a_scanned_face_that_turns_out_bad_supplies_nothing() {
        let folder = Folder::new("bad");
        // The directory reads, the tables it points at are cut off.
        std::fs::write(folder.0.join("Broken.ttf"), &SAMPLE[..12 + 11 * 16]).unwrap();
        // A directory that does not read keeps the file out.
        let mut header = vec![0, 1, 0, 0];
        header.extend_from_slice(b"truncated");
        std::fs::write(folder.0.join("Torn.ttf"), &header).unwrap();
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
