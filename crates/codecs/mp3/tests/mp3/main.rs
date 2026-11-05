//! Integration tests for `cutlass-mp3`, through `cutlass-mp3-container`.
//!
//! Fixtures and their provenance are in `tests/fixtures/README.md`.
//! Reference decoders run when they are installed and are skipped, with a
//! line on stderr, when they are not. Set `MP3_REQUIRE_ORACLES=1` to turn a
//! missing reference decoder into a failure. Media under
//! `../local-files` (next to the checkout) is used when it is present.

// `clippy.toml` allows `expect` and `panic` in `cfg(test)` code; the
// helpers live outside `#[test]` functions, so the modules say so.
#[cfg(test)]
mod common;
#[cfg(test)]
mod decoder;
