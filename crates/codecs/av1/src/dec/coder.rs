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

/// The inverted probability `bool_prob` takes for the event "the symbol is
/// in `mask`" (bit `i`: symbol `i`) under the inverted CDF `cdf` (its last
/// entry is the adaptation counter).
#[inline(always)]
pub fn masked_prob<const N: usize>(cdf: &[u16; N], mask: u16) -> u32 {
    let n = N - 1;
    // Cumulative probability of the symbols below `i`, i in 0..=n.
    let below = |i: usize| -> u32 {
        if i == 0 {
            0
        } else if i > n {
            32768
        } else {
            32768 - u32::from(cdf[i - 1])
        }
    };
    let mut s = 0;
    for i in 0..16 {
        if mask & (1 << i) != 0 {
            s += below(i + 1) - below(i);
        }
    }
    s
}

pub trait Coder {
    /// True when the coder writes the values it is given.
    const WRITE: bool;
    fn bool_adapt(&mut self, cdf: &mut [u16; 2], v: bool) -> bool;
    fn bool_prob(&mut self, f: u32, v: bool) -> bool;
    fn symbol<const N: usize>(&mut self, cdf: &mut [u16; N], v: usize) -> usize;
    fn literal(&mut self, n: u32, v: u32) -> u32;
    fn golomb(&mut self, v: u32) -> u32;
    fn ns(&mut self, n: u32, v: u32) -> u32;
    /// A boolean whose probability is the sum of the probabilities of the
    /// symbols in `mask` (bit `i`: symbol `i`) of the inverted CDF `cdf`
    /// (see [`masked_prob`]); `cdf` does not adapt.
    #[inline(always)]
    fn bool_masked<const N: usize>(&mut self, cdf: &[u16; N], mask: u16, v: bool) -> bool {
        self.bool_prob(masked_prob(cdf, mask), v)
    }
    /// [`Coder::symbol`] on a [`super::cdf::pack4`] CDF.
    #[inline(always)]
    fn symbol_packed(&mut self, cdf: &mut u64, v: usize) -> usize {
        let mut c = super::cdf::unpack4(*cdf);
        let s = self.symbol(&mut c, v);
        *cdf = super::cdf::pack4(c);
        s
    }
    /// Tags the current stream position (an encoder hook; see
    /// `MsacEnc::mark`).
    #[inline(always)]
    fn mark(&mut self, _tag: u32) {}
    /// Encoder trials: see `MsacEnc::checkpoint` (`None` when decoding).
    #[inline(always)]
    fn checkpoint(
        &self,
        _base: Option<&crate::enc::msac_enc::MsacState>,
    ) -> Option<crate::enc::msac_enc::MsacState> {
        None
    }
    /// Returns to a [`Coder::checkpoint`].
    #[inline(always)]
    fn restore(&mut self, _s: &crate::enc::msac_enc::MsacState) {}
    /// Bits written so far (0 when decoding).
    #[inline(always)]
    fn bits_written(&self) -> u64 {
        0
    }
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
