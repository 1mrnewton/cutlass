//! The FFmpeg oracle, and what a test does when it or a corpus file is
//! missing: it says so on stderr and skips, unless
//! `MPEG2_REQUIRE_ORACLES` is set, which turns the skip into a failure.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The flag that makes a missing oracle or corpus file fail the test.
pub const REQUIRE: &str = "MPEG2_REQUIRE_ORACLES";

/// Panics when oracles are required; otherwise says what was skipped.
pub fn skip(what: &str) {
    assert!(
        std::env::var_os(REQUIRE).is_none(),
        "{what} is required by {REQUIRE}"
    );
    eprintln!("skipped: {what} is missing; set {REQUIRE}=1 to fail instead");
}

/// The first FFmpeg that runs, or `None` after [`skip`].
pub fn ffmpeg() -> Option<PathBuf> {
    let found = [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "ffmpeg",
    ]
    .into_iter()
    .map(PathBuf::from)
    .find(|path| {
        Command::new(path)
            .arg("-version")
            .output()
            .is_ok_and(|o| o.status.success())
    });
    if found.is_none() {
        skip("ffmpeg");
    }
    found
}

/// The `ffprobe` beside `ffmpeg`, or `None` after [`skip`].
pub fn ffprobe(ffmpeg: &Path) -> Option<PathBuf> {
    let probe = ffmpeg.with_file_name("ffprobe");
    let runs = Command::new(&probe)
        .arg("-version")
        .output()
        .is_ok_and(|o| o.status.success());
    if runs {
        Some(probe)
    } else {
        skip("ffprobe");
        None
    }
}
