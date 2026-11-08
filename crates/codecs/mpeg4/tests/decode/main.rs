//! The MPEG-4 Visual decoder through the engine's `Decoder` trait.

// Test-only code, so the helpers may unwrap and panic like the tests.
#[cfg(test)]
mod common;
#[cfg(test)]
mod corpus;
#[cfg(test)]
mod lifecycle;
#[cfg(test)]
mod packed;
