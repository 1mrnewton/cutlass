//! Updates: the release feed, the package for this platform, and putting it
//! in place of the running app.
//!
//! The feed is GitHub's latest release. A release offers the package named
//! for this platform and a `<package>.sha256` beside it; pre-releases and
//! drafts are ignored. The package downloads over the download crate and
//! counts only when its checksum matches. Installing swaps the macOS bundle,
//! runs the Windows installer once the app quits, or renames the new
//! AppImage over the old one. The relaunch is a command the caller starts.

#![forbid(unsafe_code)]

mod feed;
mod fetch;
mod install;
mod version;

use std::fmt;

pub use feed::{Asset, FEED_LIMIT, FEED_URL, Offer, Platform, due, offer, parse_checksum};
pub use fetch::{check, download, get_text};
pub use install::{Installed, Relaunch, install};
pub use version::Version;

/// Why a check, download or install failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UpdateError {
    /// The server could not be reached or answered with an error.
    Network(String),
    /// The feed or a checksum file did not read.
    Feed(String),
    /// The download was stopped; its pieces are kept.
    Stopped,
    /// This copy of the app cannot update itself.
    NotInstalled(String),
    /// The package could not be put in place. The old app is left.
    Install(String),
}

impl fmt::Display for UpdateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Network(message)
            | Self::Feed(message)
            | Self::NotInstalled(message)
            | Self::Install(message) => formatter.write_str(message),
            Self::Stopped => formatter.write_str("the download was stopped"),
        }
    }
}

impl std::error::Error for UpdateError {}
