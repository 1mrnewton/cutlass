//! One entropy-coder interface for the parts of CELT that the encoder and
//! decoder share symbol for symbol (`ec_ctx` in the reference).
//!
//! Each method takes the value the encoder writes and returns the value the
//! decoder read; the encoder returns its input unchanged.

use crate::range::{RangeDecoder, RangeEncoder};

pub(crate) enum Coder<'c, 'a> {
    Dec(&'c mut RangeDecoder<'a>),
    Enc(&'c mut RangeEncoder),
}

impl Coder<'_, '_> {
    #[inline]
    pub(crate) fn encoding(&self) -> bool {
        matches!(self, Self::Enc(_))
    }

    #[inline]
    pub(crate) fn tell(&self) -> i32 {
        match self {
            Self::Dec(d) => d.tell(),
            Self::Enc(e) => e.tell(),
        }
    }

    #[inline]
    pub(crate) fn tell_frac(&self) -> i32 {
        match self {
            Self::Dec(d) => d.tell_frac() as i32,
            Self::Enc(e) => e.tell_frac() as i32,
        }
    }

    /// Bytes the frame may use.
    #[inline]
    pub(crate) fn storage(&self) -> i32 {
        let bytes = match self {
            Self::Dec(d) => d.storage(),
            Self::Enc(e) => e.storage(),
        };
        i32::try_from(bytes).unwrap_or(i32::MAX)
    }

    #[inline]
    pub(crate) fn bit_logp(&mut self, value: bool, logp: u32) -> bool {
        match self {
            Self::Dec(d) => d.bit_logp(logp),
            Self::Enc(e) => {
                e.bit_logp(value, logp);
                value
            }
        }
    }

    /// A uniform integer in `0..ft`.
    #[inline]
    pub(crate) fn uint(&mut self, value: u32, ft: u32) -> u32 {
        match self {
            Self::Dec(d) => d.dec_uint(ft),
            Self::Enc(e) => {
                e.enc_uint(value, ft);
                value
            }
        }
    }

    /// `bits` raw bits.
    #[inline]
    pub(crate) fn bits(&mut self, value: u32, bits: u32) -> u32 {
        match self {
            Self::Dec(d) => d.dec_bits(bits),
            Self::Enc(e) => {
                e.enc_bits(value, bits);
                value
            }
        }
    }

    #[inline]
    pub(crate) fn icdf(&mut self, value: usize, icdf: &[u8], ftb: u32) -> usize {
        match self {
            Self::Dec(d) => d.dec_icdf(icdf, ftb),
            Self::Enc(e) => {
                e.enc_icdf(value, icdf, ftb);
                value
            }
        }
    }
}
