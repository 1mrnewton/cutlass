//! One installed face for text segments.
//!
//! The draft stores the file name. Evaluation does not open that path. This
//! table supplies the bytes when the name matches the face that was read.

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

/// Bytes for one face. An empty name supplies nothing.
pub struct FaceTable {
    name: String,
    bytes: Vec<u8>,
}

impl FaceTable {
    /// The first candidate that is a single readable face.
    #[must_use]
    pub fn load() -> Self {
        Self::load_from(FACE_CANDIDATES)
    }

    #[cfg(test)]
    #[must_use]
    pub fn named(name: impl Into<String>, bytes: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            bytes,
        }
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    fn load_from(paths: &[&str]) -> Self {
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
            return Self {
                name: name.to_owned(),
                bytes,
            };
        }
        Self {
            name: String::new(),
            bytes: Vec::new(),
        }
    }
}

impl Fonts for FaceTable {
    fn face(&self, path: &str) -> Option<&[u8]> {
        if self.name.is_empty() || path != self.name {
            None
        } else {
            Some(self.bytes.as_slice())
        }
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

        let junk = directory.join(format!("cutlass-face-junk-{}", std::process::id()));
        std::fs::write(&junk, b"not a font").expect("write");
        let skipped = FaceTable::load_from(&[junk.to_str().expect("utf-8")]);
        let _ = std::fs::remove_file(&junk);
        assert!(skipped.name().is_empty());
    }
}
