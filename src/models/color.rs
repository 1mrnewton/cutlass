//! 8-bit-per-channel sRGB color, kept Slint-independent so the
//! domain model never reaches into the Slint crate. The projector
//! converts to `slint::Color` exactly once per track at projection
//! time.
//!
//! We chose explicit channels (not a packed `u32`) so the serde wire
//! format the agent will eventually produce reads as
//! `{ "r": 74, "g": 111, "b": 165, "a": 255 }` instead of an opaque
//! integer.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    /// Opaque RGB triple — the common case for track colors.
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }
}
