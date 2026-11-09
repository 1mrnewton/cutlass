//! Sample-level kernels for the AV1 decoder and encoder.
//!
//! Every kernel is generic over [`Pixel`]: `u8` for 8-bit streams and `u16`
//! for 10- and 12-bit streams. The bit depth is passed at run time as `bd`
//! (8, 10, or 12). The kernels are bit exact to the AV1 specification
//! (v1.0.0-errata1). Buffers are slices with an explicit stride in samples.

pub mod cdef;
pub mod fg;
pub mod ipred;
pub mod itx;
pub mod lf;
pub mod lr;
pub mod mc;
pub mod superres;

/// One picture sample. `u8` holds 8-bit, `u16` holds 10- and 12-bit.
pub trait Pixel:
    Copy + Default + Send + Sync + PartialEq + Eq + core::fmt::Debug + 'static
{
    /// True for `u16`.
    const HBD: bool;
    /// Converts a value already inside `0..=(1 << bd) - 1`.
    fn from_i32(v: i32) -> Self;
    fn to_i32(self) -> i32;
}

impl Pixel for u8 {
    const HBD: bool = false;
    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v as u8
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        i32::from(self)
    }
}

impl Pixel for u16 {
    const HBD: bool = true;
    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v as u16
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        i32::from(self)
    }
}

/// Clamps `v` into `0..=(1 << bd) - 1` and converts.
#[inline(always)]
pub fn clip_px<P: Pixel>(v: i32, bd: u32) -> P {
    P::from_i32(v.clamp(0, (1 << bd) - 1))
}

/// `Round2` from the specification for a non-negative shift.
#[inline(always)]
pub fn round2(x: i32, n: u32) -> i32 {
    if n == 0 { x } else { (x + (1 << (n - 1))) >> n }
}

/// `Round2` on 64-bit values.
#[inline(always)]
pub fn round2_64(x: i64, n: u32) -> i64 {
    if n == 0 { x } else { (x + (1 << (n - 1))) >> n }
}

/// `Round2Signed` from the specification.
#[inline(always)]
pub fn round2_signed(x: i32, n: u32) -> i32 {
    if x >= 0 { round2(x, n) } else { -round2(-x, n) }
}
