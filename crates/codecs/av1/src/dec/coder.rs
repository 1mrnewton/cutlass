//! The entropy coding direction of the tile syntax.
//!
//! The tile code in this module reads or writes every syntax element through
//! [`Coder`]. When decoding, the value argument is ignored and the decoded
//! value is returned. When encoding, the value is written and returned, so the
//! same code updates contexts and reconstructs the picture exactly as a
//! decoder will.

use super::msac::Msac;

/// [`Coder::mark`] tags with this bit set are superblock starts, followed by
/// the superblock's mi row (bits 15..30) and column (bits 0..15); other tags
/// are `cdef_idx` positions.
pub const SB_MARK: u32 = 1 << 31;

pub trait Coder {
    /// True when the coder writes the values it is given.
    const WRITE: bool;
    fn bool_adapt(&mut self, cdf: &mut [u16; 2], v: bool) -> bool;
    fn bool_prob(&mut self, f: u32, v: bool) -> bool;
    fn symbol<const N: usize>(&mut self, cdf: &mut [u16; N], v: usize) -> usize;
    fn literal(&mut self, n: u32, v: u32) -> u32;
    fn golomb(&mut self, v: u32) -> u32;
    fn ns(&mut self, n: u32, v: u32) -> u32;
    /// Tags the current stream position (an encoder hook; see
    /// `MsacEnc::mark`).
    #[inline(always)]
    fn mark(&mut self, _tag: u32) {}
    /// Runs `f` on the coder; the decoder runs it on a copy of its state
    /// held in a local, which a loop can keep in registers.
    #[inline(always)]
    fn with_local<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        f(self)
    }
    /// Runs `f` on the decoder itself, for loops written for decoding only;
    /// `None` when the coder writes.
    #[inline(always)]
    fn decoder<R>(&mut self, f: impl FnOnce(&mut Msac<'_>) -> R) -> Option<R> {
        let _ = f;
        None
    }
}

impl Coder for Msac<'_> {
    #[inline(always)]
    fn decoder<R>(&mut self, f: impl FnOnce(&mut Msac<'_>) -> R) -> Option<R> {
        Some(f(self))
    }
    const WRITE: bool = false;
    #[inline(always)]
    fn bool_adapt(&mut self, cdf: &mut [u16; 2], _v: bool) -> bool {
        Msac::bool_adapt(self, cdf)
    }
    #[inline(always)]
    fn bool_prob(&mut self, f: u32, _v: bool) -> bool {
        Msac::bool_prob(self, f)
    }
    #[inline(always)]
    fn symbol<const N: usize>(&mut self, cdf: &mut [u16; N], _v: usize) -> usize {
        Msac::symbol(self, cdf)
    }
    #[inline(always)]
    fn literal(&mut self, n: u32, _v: u32) -> u32 {
        Msac::literal(self, n)
    }
    #[inline(always)]
    fn golomb(&mut self, _v: u32) -> u32 {
        Msac::golomb(self)
    }
    #[inline(always)]
    fn ns(&mut self, n: u32, _v: u32) -> u32 {
        Msac::ns(self, n)
    }
    #[inline(always)]
    fn with_local<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        Msac::with_local(self, f)
    }
}
