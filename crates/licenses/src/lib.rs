//! The notices for the third-party crates the app ships.
//!
//! `third_party.json` is written by `cargo run -p licenses` from the lock
//! file and each crate's own license files, so every app shows the same
//! list. A license text several crates share is stored once.

#![forbid(unsafe_code)]

use serde::Deserialize;

/// Cutlass's own license.
pub const CUTLASS_LICENSE: &str = "MIT OR Apache-2.0";

const THIRD_PARTY: &str = include_str!("../third_party.json");

/// One crate the app is built from.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct Crate {
    pub name: String,
    pub version: String,
    /// The SPDX expression the crate declares.
    pub license: String,
    pub repository: String,
    /// Indexes into [`Notices::texts`].
    texts: Vec<usize>,
}

/// Every crate and the license texts they ship with.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct Notices {
    pub crates: Vec<Crate>,
    texts: Vec<String>,
}

impl Notices {
    /// The license files `krate` ships, word for word.
    pub fn texts_of<'a>(&'a self, krate: &'a Crate) -> impl Iterator<Item = &'a str> + 'a {
        krate
            .texts
            .iter()
            .filter_map(|index| self.texts.get(*index).map(String::as_str))
    }

    /// The whole list as one text file: each crate, its license, and its
    /// files.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut out = format!("Cutlass is licensed {CUTLASS_LICENSE}.\n\nIt is built with:\n");
        for krate in &self.crates {
            out.push_str(&format!(
                "\n==== {} {} ({})\n",
                krate.name, krate.version, krate.license
            ));
            if !krate.repository.is_empty() {
                out.push_str(&krate.repository);
                out.push('\n');
            }
            for text in self.texts_of(krate) {
                out.push('\n');
                out.push_str(text);
                out.push('\n');
            }
        }
        out
    }
}

/// The notices bundled with this build. A file that does not read is an
/// empty list.
#[must_use]
pub fn notices() -> Notices {
    serde_json::from_str(THIRD_PARTY).unwrap_or(Notices {
        crates: Vec::new(),
        texts: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bundled_list_reads_and_every_crate_has_a_license() {
        let notices = notices();
        assert!(notices.crates.len() > 50, "{}", notices.crates.len());
        for krate in &notices.crates {
            assert!(!krate.license.is_empty(), "{}", krate.name);
        }
        let egui = notices
            .crates
            .iter()
            .find(|krate| krate.name == "egui")
            .expect("egui is shipped");
        assert!(notices.texts_of(egui).count() > 0);
        let text = notices.to_text();
        assert!(text.starts_with("Cutlass is licensed MIT OR Apache-2.0."));
        assert!(text.contains("==== egui "));
    }
}
