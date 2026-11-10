//! CABAC, clause 9.3. Context variables `m` and `n` are Tables 9-12 to 9-33.

use cutlass_decode::CodecError;

use crate::bits::Bits;
use crate::tables::{CABAC_MN, LAST_8X8, RANGE_LPS, SIG_8X8, TRANS_LPS, TRANS_MPS};

/// Contexts 0..1023 are clause 9.3. Context 1040 is `mb_skip_run_type`.
pub(crate) const CONTEXTS: usize = 1053;

pub struct Cabac<'a> {
    data: &'a [u8],
    /// Next byte that has not been pulled into `cache`.
    index: usize,
    cache: u64,
    /// Valid low bits of `cache`.
    bits: u32,
    range: u32,
    offset: u32,
    /// Each context is `pStateIdx << 1 | valMPS`.
    ctx: [u8; CONTEXTS],
}

/// `rangeTabLPS` by packed context.
static LPS_RANGE: [[u8; 4]; 128] = {
    let mut table = [[0u8; 4]; 128];
    let mut packed = 0;
    while packed < 128 {
        table[packed] = RANGE_LPS[packed >> 1];
        packed += 1;
    }
    table
};

/// The packed context after one bin. The first half is after its most
/// probable symbol, the second half after its least probable one, where
/// state 0 also swaps which symbol is the most probable.
static AFTER: [u8; 256] = {
    let mut table = [0u8; 256];
    let mut packed = 0;
    while packed < 128 {
        let mps = packed as u8 & 1;
        table[packed] = (TRANS_MPS[packed >> 1] << 1) | mps;
        let swapped = if packed >> 1 == 0 { 1 - mps } else { mps };
        table[packed + 128] = (TRANS_LPS[packed >> 1] << 1) | swapped;
        packed += 1;
    }
    table
};

impl<'a> Cabac<'a> {
    pub fn start(bits: &mut Bits<'a>, qp: i32, init: usize) -> Result<Self, CodecError> {
        bits.cabac_align()?;
        let data = bits.rest();
        if data.is_empty() {
            return Err(bits.truncated("cabac"));
        }
        let mut eng = Self {
            data,
            index: 0,
            cache: 0,
            bits: 0,
            range: 510,
            offset: 0,
            ctx: [0; CONTEXTS],
        };
        eng.offset = eng.read_bits(9)?;
        eng.init_states(qp, init);
        Ok(eng)
    }

    fn at_bit(&self) -> u64 {
        u64::try_from(self.index.saturating_mul(8)).unwrap_or(u64::MAX) - u64::from(self.bits)
    }

    /// Pulls bytes into the cache: four at once while the stream has them,
    /// then one at a time. Called with fewer than 32 bits held.
    fn fill(&mut self) {
        if self.bits <= 32
            && let Some(&[a, b, c, d]) = self.data.get(self.index..self.index + 4)
        {
            self.cache = (self.cache << 32) | u64::from(u32::from_be_bytes([a, b, c, d]));
            self.index += 4;
            self.bits += 32;
            return;
        }
        while self.bits <= 56 && self.index < self.data.len() {
            self.cache = (self.cache << 8) | u64::from(self.data[self.index]);
            self.index += 1;
            self.bits += 8;
        }
    }

    /// `n` is at most 9, and may be 0.
    #[inline(always)]
    fn read_bits(&mut self, n: u32) -> Result<u32, CodecError> {
        if self.bits < n {
            self.fill();
            if self.bits < n {
                return Err(CodecError::Truncated {
                    at_bit: self.at_bit(),
                    what: "cabac",
                });
            }
        }
        self.bits -= n;
        let mask = (1u64 << n) - 1;
        Ok(((self.cache >> self.bits) & mask) as u32)
    }

    #[inline(always)]
    fn renorm(&mut self) -> Result<(), CodecError> {
        if self.range >= 256 {
            return Ok(());
        }
        // One shift brings `range` back to 256..510. That is the same bits the
        // one-at-a-time loop used to read.
        let shift = self.range.leading_zeros() - 23;
        self.range <<= shift;
        self.offset = (self.offset << shift) | self.read_bits(shift)?;
        Ok(())
    }

    #[inline(always)]
    pub fn decision(&mut self, ctx: usize) -> Result<u8, CodecError> {
        // Which symbol comes out is as good as random to the CPU, so
        // nothing here branches on it: the two outcomes differ by a mask
        // and a table half.
        let packed = usize::from(self.ctx[ctx] & 127);
        let range_lps = u32::from(LPS_RANGE[packed][((self.range >> 6) & 3) as usize]);
        let range_mps = self.range - range_lps;
        let lps = u32::from(self.offset >= range_mps);
        let mask = lps.wrapping_neg();
        self.ctx[ctx] = AFTER[packed + ((lps as usize) << 7)];
        let offset = self.offset - (range_mps & mask);
        let range = range_mps ^ ((range_mps ^ range_lps) & mask);
        // One shift brings `range` back to 256..510; it is 0 when `range`
        // is already there.
        let shift = range.leading_zeros() - 23;
        self.range = range << shift;
        self.offset = (offset << shift) | self.read_bits(shift)?;
        Ok((packed as u8 & 1) ^ lps as u8)
    }

    #[inline(always)]
    pub fn bypass(&mut self) -> Result<u8, CodecError> {
        self.offset = (self.offset << 1) | self.read_bits(1)?;
        if self.offset >= self.range {
            self.offset -= self.range;
            Ok(1)
        } else {
            Ok(0)
        }
    }

    pub fn terminate(&mut self) -> Result<bool, CodecError> {
        self.range -= 2;
        if self.offset >= self.range {
            Ok(true)
        } else {
            self.renorm()?;
            Ok(false)
        }
    }

    /// Clause 9.3.1.2 after the samples of an I_PCM macroblock: the
    /// arithmetic decoder starts again on `data`. The context variables are
    /// not set up again (clause 9.3.1 does that at the start of a slice
    /// only), so they carry on from the macroblocks before.
    pub fn restart(&mut self, data: &'a [u8]) -> Result<(), CodecError> {
        if data.is_empty() {
            return Err(CodecError::Truncated {
                at_bit: 0,
                what: "cabac",
            });
        }
        self.data = data;
        self.index = 0;
        self.cache = 0;
        self.bits = 0;
        self.range = 510;
        self.offset = self.read_bits(9)?;
        Ok(())
    }

    pub fn tail(&self) -> &'a [u8] {
        let consumed = self.index * 8 - usize::try_from(self.bits).unwrap_or(0);
        let byte = consumed.div_ceil(8);
        self.data.get(byte..).unwrap_or(&[])
    }

    fn init_states(&mut self, qp: i32, init: usize) {
        let slice_qp = qp.clamp(0, 51);
        let mut state = [0u8; CONTEXTS];
        let mut mps = [0u8; CONTEXTS];
        for (ctx, &(m, n)) in CABAC_MN[init].iter().enumerate() {
            if ctx == 276 {
                state[276] = 63;
                mps[276] = 0;
                continue;
            }
            let pre = (((i32::from(m) * slice_qp) >> 4) + i32::from(n)).clamp(1, 126);
            if pre <= 63 {
                state[ctx] = (63 - pre) as u8;
                mps[ctx] = 0;
            } else {
                state[ctx] = (pre - 64) as u8;
                mps[ctx] = 1;
            }
        }
        init_rle(&mut state, &mut mps, slice_qp);
        for (packed, (state, mps)) in self.ctx.iter_mut().zip(state.iter().zip(&mps)) {
            *packed = (state << 1) | mps;
        }
    }
}

/// Table J-7. `mb_skip_run_type` is context 1040, `m` 18 and `n` 64, for every
/// `cabac_init_idc`.
pub(crate) fn init_rle(state: &mut [u8], mps: &mut [u8], qp: i32) {
    let Some(slot) = state.get_mut(1040) else {
        return;
    };
    let Some(bit) = mps.get_mut(1040) else {
        return;
    };
    let pre = ((18 * qp) >> 4) + 64;
    if pre <= 63 {
        *slot = u8::try_from(63 - pre).unwrap_or(0);
        *bit = 0;
    } else {
        *slot = u8::try_from(pre - 64).unwrap_or(0);
        *bit = 1;
    }
}

/// Significant-coefficient contexts for an 8x8 block, Table 9-43 frame column.
pub fn sig8(index: usize) -> usize {
    usize::from(SIG_8X8[index])
}

/// Table 9-43 field column. A field macroblock uses these instead of [`sig8`].
pub fn sig8_field(index: usize) -> usize {
    const FIELD: [u8; 63] = [
        0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 7, 7, 8, 4, 5, 6, 9, 10, 10, 8, 11, 12, 11, 9, 9, 10, 10,
        8, 11, 12, 11, 9, 9, 10, 10, 8, 11, 12, 11, 9, 9, 10, 10, 8, 13, 13, 9, 9, 10, 10, 8, 13,
        13, 9, 9, 10, 10, 14, 14, 14, 14, 14,
    ];
    usize::from(FIELD[index])
}

pub fn last8(index: usize) -> usize {
    usize::from(LAST_8X8[index])
}

#[cfg(test)]
mod tests {
    use super::Cabac;
    use crate::bits::Bits;

    /// Clause 9.3.1.2 runs again after the samples of an I_PCM macroblock,
    /// and clause 9.3.1.1 does not: the arithmetic decoder starts over on the
    /// bytes that follow, and every context keeps what it has learned.
    #[test]
    fn restart_after_pcm_keeps_the_context_variables() {
        let data = [0x5a, 0xc3, 0x3c, 0x99, 0x71, 0x0f, 0xe2, 0x48];
        let next = [0b1010_0101, 0b1100_0000, 0, 0];
        let mut bits = Bits::new(&data);
        let mut cab = Cabac::start(&mut bits, 26, 0).unwrap();
        let fresh = cab.ctx;
        for ctx in [5usize, 5, 5, 73, 73, 105, 105, 105] {
            cab.decision(ctx).unwrap();
        }
        let learned = cab.ctx;
        assert_ne!(learned, fresh, "the decisions moved no context");
        cab.restart(&next).unwrap();
        assert_eq!(cab.ctx, learned, "the contexts were set up again");
        assert_eq!(cab.range, 510);
        assert_eq!(cab.offset, 0b1_0100_1011);
        assert!(cab.restart(&[]).is_err());
    }
}
