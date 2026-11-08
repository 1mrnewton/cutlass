//! Integration tests for `cutlass-ac3`.
//!
//! Fixtures and their provenance are in `tests/fixtures/README.md`.
//! Reference decoders run when they are installed and are skipped, with a
//! line on stderr, when they are not. Set `AC3_REQUIRE_ORACLES=1` to turn a
//! missing reference decoder into a failure. `AC3_FATE_SAMPLES` names the
//! FFmpeg FATE sample folder for the tools FFmpeg's encoders do not write.

// `clippy.toml` allows `expect` and `panic` in `cfg(test)` code; the
// helpers live outside `#[test]` functions, so the modules say so.
#[cfg(test)]
mod common;
#[cfg(test)]
mod decoder;
#[cfg(test)]
mod encoder;
#[cfg(test)]
mod fate;
