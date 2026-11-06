//! Sample width for an 8-bit or 10-bit reconstruction.
//!
//! An 8-bit value stays in `0..=255`. A 10-bit value stays in `0..=1023`.
//! The 8-bit path passes depth 8, so the clamp is the same bound as before.

/// One reconstructed sample. `u8` is 8-bit. `u16` holds 10-bit and 12-bit.
pub trait Pix: Copy + Clone + Default + Send + Sync + std::fmt::Debug {
    fn to_i32(self) -> i32;
    fn from_i32(value: i32) -> Self;
}

impl Pix for u8 {
    fn to_i32(self) -> i32 {
        i32::from(self)
    }

    fn from_i32(value: i32) -> Self {
        value as u8
    }
}

impl Pix for u16 {
    fn to_i32(self) -> i32 {
        i32::from(self)
    }

    fn from_i32(value: i32) -> Self {
        value as u16
    }
}

/// Clamp one residual sum into the legal sample range for `depth`.
#[must_use]
pub fn clip_sample(value: i32, depth: u8) -> i32 {
    let max = (1i32 << u32::from(depth.min(16))) - 1;
    value.clamp(0, max)
}

/// `1 << (BitDepth - 1)`. 128 at 8-bit, 512 at 10-bit.
#[must_use]
pub fn sample_mid(depth: u8) -> i32 {
    128i32 << u32::from(depth.saturating_sub(8))
}

/// `BitDepth - 8`. Zero at 8-bit, so a shift by it leaves an 8-bit value alone.
#[must_use]
pub fn depth_shift(depth: u8) -> u32 {
    u32::from(depth.saturating_sub(8))
}
