//! Installed faces for text segments.
//!
//! The draft stores the file name. Evaluation does not open that path. This
//! table supplies the bytes when the name matches a face that was read.

use std::path::Path;

use cutlass_eval::Fonts;

const FACE_CANDIDATES: &[&str] = &[
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Geneva.ttf",
    "/System/Library/Fonts/NewYork.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
];

/// Bytes for the faces that were read. An empty table supplies nothing.
pub struct FaceTable {
    faces: Vec<(String, Vec<u8>)>,
}

impl FaceTable {
    /// Every candidate that is a single readable face, in that order.
    ///
    /// A second file with the same name is skipped.
    #[must_use]
    pub fn load() -> Self {
        Self::load_from(FACE_CANDIDATES)
    }

    #[cfg(test)]
    #[must_use]
    pub fn named(name: impl Into<String>, bytes: Vec<u8>) -> Self {
        let mut table = Self { faces: Vec::new() };
        table.push_face(name.into(), bytes);
        table
    }

    #[cfg(test)]
    #[must_use]
    pub fn from_pairs(faces: Vec<(String, Vec<u8>)>) -> Self {
        let mut table = Self { faces: Vec::new() };
        for (name, bytes) in faces {
            table.push_face(name, bytes);
        }
        table
    }

    /// The first face. An empty table has an empty name.
    #[must_use]
    pub fn name(&self) -> &str {
        self.faces
            .first()
            .map(|(name, _)| name.as_str())
            .unwrap_or("")
    }

    /// File names, in the order they were read.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.faces.iter().map(|(name, _)| name.clone()).collect()
    }

    fn load_from(paths: &[&str]) -> Self {
        let mut table = Self { faces: Vec::new() };
        for path in paths {
            let Ok(bytes) = std::fs::read(path) else {
                continue;
            };
            if cutlass_font::read(&bytes).is_err() {
                continue;
            }
            let Some(name) = Path::new(path).file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            table.push_face(name.to_owned(), bytes);
        }
        table
    }

    fn push_face(&mut self, name: String, bytes: Vec<u8>) {
        if name.is_empty() || self.faces.iter().any(|(have, _)| have == &name) {
            return;
        }
        self.faces.push((name, bytes));
    }
}

impl Fonts for FaceTable {
    fn face(&self, path: &str) -> Option<&[u8]> {
        self.faces
            .iter()
            .find(|(name, _)| name == path)
            .map(|(_, bytes)| bytes.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &[u8] = include_bytes!("../../../crates/font/fixtures/sample.ttf");

    #[test]
    fn an_empty_list_supplies_nothing_and_a_face_file_is_kept() {
        let empty = FaceTable::load_from(&[]);
        assert!(empty.name().is_empty());
        assert!(empty.face("sample.ttf").is_none());

        let directory = std::env::temp_dir();
        let path = directory.join(format!(
            "cutlass-face-{}-{}.ttf",
            std::process::id(),
            SAMPLE.len()
        ));
        std::fs::write(&path, SAMPLE).expect("write");
        let loaded = FaceTable::load_from(&[path.to_str().expect("utf-8")]);
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            loaded.name(),
            path.file_name().and_then(|name| name.to_str()).unwrap()
        );
        assert!(loaded.face(loaded.name()).is_some());
        assert!(loaded.face("other.ttf").is_none());

        let second = directory.join(format!(
            "cutlass-face-b-{}-{}.otf",
            std::process::id(),
            SAMPLE.len()
        ));
        std::fs::write(&path, SAMPLE).expect("write");
        std::fs::write(&second, SAMPLE).expect("write");
        let both = FaceTable::load_from(&[
            path.to_str().expect("utf-8"),
            second.to_str().expect("utf-8"),
            second.to_str().expect("utf-8"),
        ]);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&second);
        assert_eq!(both.names().len(), 2);
        assert!(both.face(both.names()[0].as_str()).is_some());
        assert!(both.face(both.names()[1].as_str()).is_some());

        let junk = directory.join(format!("cutlass-face-junk-{}", std::process::id()));
        std::fs::write(&junk, b"not a font").expect("write");
        let skipped = FaceTable::load_from(&[junk.to_str().expect("utf-8")]);
        let _ = std::fs::remove_file(&junk);
        assert!(skipped.name().is_empty());
    }
}
