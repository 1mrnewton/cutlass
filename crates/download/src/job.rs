//! One URL, the partial it writes, and how soon it should run.

use std::fmt;
use std::path::PathBuf;

use crate::Checksum;

/// The id of one transfer. The downloader assigns it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct JobId(pub u64);

impl fmt::Display for JobId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// A row on screen runs before a file nothing is showing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Priority {
    /// The panel is showing this row.
    Visible,
    /// Nothing on screen is waiting for this file.
    Background,
}

/// What `storage` asks this crate to fetch.
///
/// `checksum` is absent for a preview until the host provides one.
/// The partial stays at `destination` until `storage` publishes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Job {
    pub url: String,
    pub destination: PathBuf,
    pub checksum: Option<Checksum>,
    pub priority: Priority,
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{Job, Priority};
    use crate::Checksum;

    #[test]
    fn a_preview_can_omit_the_checksum_and_a_visible_row_is_marked() {
        let preview = Job {
            url: "https://example.test/poster.mp4".to_owned(),
            destination: PathBuf::from("incoming/poster"),
            checksum: None,
            priority: Priority::Visible,
        };
        assert!(preview.checksum.is_none());
        assert_eq!(preview.priority, Priority::Visible);
        let package = Job {
            url: preview.url.clone(),
            destination: preview.destination.clone(),
            checksum: Some(Checksum::of(b"package")),
            priority: Priority::Background,
        };
        assert_ne!(package.checksum, preview.checksum);
        assert_ne!(package.priority, preview.priority);
    }
}
