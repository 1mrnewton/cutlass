//! The inverse transform of a block of an 8-bit stream (AV1 specification
//! 7.13.3), added to the prediction: 16-bit values, eight to a vector.
//!
//! The row transforms run on eight rows at a time and the column
//! transforms on eight columns at a time, so an entry of the array `T` a
//! one-dimensional transform works on (7.13.2) is one vector. Each
//! transform is straight-line code made from its table of steps
//! ([`steps`]), once for each count of inputs that can be non-zero: a
//! step on entries known to be zero is no code, and a rotation of one
//! non-zero entry is two multiplies.
//!
//! In an 8-bit stream every intermediate value fits 16 bits, so the
//! clamps of 7.13.2 and 7.13.3 are saturating adds or nothing at all. A
//! stream that breaks that promise gets meaningless samples here, as it
//! does from any decoder.

#[macro_use]
mod macros;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;
mod scalar;
pub mod steps;
#[cfg(test)]
mod tests;

/// A one-dimensional transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Dct,
    Adst,
    Identity,
}

/// A transform block and the samples it is added to.
#[derive(Debug)]
pub struct Block<'a> {
    /// The prediction: `h` rows of `w` samples, `stride` apart.
    pub dst: &'a mut [u8],
    pub stride: usize,
    /// The dequantized coefficients in rows of `min(w, 32)`; only the
    /// first `nz_rows` rows and `nz_cols` columns can be non-zero. A
    /// coefficient is taken as its low 16 bits.
    pub coefs: &'a [i32],
    pub w: usize,
    pub h: usize,
    /// The transform of the rows, then of the columns.
    pub row: Kind,
    pub col: Kind,
    pub nz_rows: usize,
    pub nz_cols: usize,
    /// The rounding shift after the row transforms (0 to 2).
    pub shift: u32,
    /// The columns go on bottom up, the rows right to left (the flipped
    /// ADST).
    pub flip_ud: bool,
    pub flip_lr: bool,
}

/// Whether [`add`] is a vector kernel in this build.
pub const ACCELERATED: bool = cfg!(all(target_arch = "aarch64", target_feature = "neon"));

/// Whether there is a transform `kind` of `n` values here.
const fn has(kind: Kind, n: usize) -> bool {
    matches!((kind, n), (Kind::Dct, 8 | 16 | 32 | 64))
}

/// Whether [`add`] takes a block of this size and these transforms.
#[must_use]
pub const fn supports(w: usize, h: usize, row: Kind, col: Kind) -> bool {
    has(row, w) && has(col, h)
}

/// Whether the slices of `b` hold the block.
fn fits(b: &Block<'_>) -> bool {
    let (tw, th) = (b.w.min(32), b.h.min(32));
    supports(b.w, b.h, b.row, b.col)
        && b.shift <= 2
        && b.stride >= b.w
        && b.coefs.len() >= tw * th
        && b.dst.len() >= (b.h - 1) * b.stride + b.w
}

/// Transforms the block and adds it to its prediction. Returns `false`,
/// with nothing done, for a block [`supports`] refuses or one its slices
/// are too short for.
#[inline]
pub fn add(b: &mut Block<'_>) -> bool {
    if !fits(b) {
        return false;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // SAFETY: the `cfg` above proves NEON is enabled for this build,
        // and the kernel names AES only where the build has it too.
        #[allow(unsafe_code)]
        unsafe {
            neon::add(b);
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    scalar::add(b);
    true
}
