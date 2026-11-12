//! CABAC, clause 9.3. Context variables `m` and `n` are Tables 9-12 to 9-33.

use cutlass_decode::CodecError;

use crate::bits::Bits;
use crate::tables::{CABAC_MN, LAST_8X8, RANGE_LPS, SIG_8X8, TRANS_LPS, TRANS_MPS};

/// Contexts 0..1023 are clause 9.3. Context 1040 is `mb_skip_run_type`.
pub(crate) const CONTEXTS: usize = 1053;

pub struct Cabac<'a> {
    data: &'a [u8],
    /// Next byte that has not been pulled into `value`. It runs past the
    /// end of `data` by the zero bytes read there.
    index: usize,
    /// `codIOffset` from bit 54, with room above for the one more bit a
    /// bypass bin shifts in, then the stream bits read ahead of it, then
    /// one marker bit, then zeros. Shifting the value left reads the bits
    /// under the offset; when the marker climbs to bit 46, fewer than eight
    /// are left and [`Self::refill`] adds more.
    value: u64,
    range: u32,
    /// Each context is `pStateIdx << 1 | valMPS`.
    ctx: [u8; CONTEXTS],
}

/// [`Cabac`] over a run of bins, from [`Cabac::run`]: its bins are the
/// same, with the offset and the range held here until the run ends.
pub struct Run<'c, 'a> {
    cab: &'c mut Cabac<'a>,
    value: u64,
    range: u32,
}

impl Run<'_, '_> {
    /// `codIRange` shifted back to 256..510 and the offset with it.
    #[inline(always)]
    fn renorm_by(&mut self, range: u32, value: u64) {
        let shift = range.leading_zeros() - 23;
        self.range = range << shift;
        self.value = value << shift;
        if self.value & AHEAD == 0 {
            self.value = refilled(self.cab, self.value);
        }
    }

    /// One bin decoded with context `ctx`, clause 9.3.3.2.1.
    #[inline(always)]
    pub fn decision(&mut self, ctx: usize) -> Result<u8, CodecError> {
        // Which symbol comes out is as good as random to the CPU, so
        // nothing here branches on it: the two outcomes differ by a mask
        // and a table half.
        let packed = usize::from(self.cab.ctx[ctx] & 127);
        let range_lps = u32::from(LPS_RANGE[packed][((self.range >> 6) & 3) as usize]);
        let range_mps = self.range - range_lps;
        let scaled = u64::from(range_mps) << OFFSET_AT;
        let lps = u64::from(self.value >= scaled);
        let mask = lps.wrapping_neg();
        self.cab.ctx[ctx] = AFTER[packed + ((lps as usize) << 7)];
        let value = self.value - (scaled & mask);
        let range = range_mps ^ ((range_mps ^ range_lps) & mask as u32);
        self.renorm_by(range, value);
        Ok((packed as u8 & 1) ^ lps as u8)
    }

    /// One bin decoded in bypass mode, clause 9.3.3.2.3.
    #[inline(always)]
    pub fn bypass(&mut self) -> Result<u8, CodecError> {
        let mut value = self.value << 1;
        let scaled = u64::from(self.range) << OFFSET_AT;
        let one = value >= scaled;
        if one {
            value -= scaled;
        }
        self.value = value;
        if value & AHEAD == 0 {
            self.value = refilled(self.cab, value);
        }
        Ok(u8::from(one))
    }
}

/// `value` as the offset of `cab` after [`Cabac::refill`]. Only the offset
/// is handed over, not the [`Run`], so the run stays where the compiler
/// put it.
#[cold]
#[inline(never)]
fn refilled(cab: &mut Cabac<'_>, value: u64) -> u64 {
    cab.value = value;
    cab.refill();
    cab.value
}

impl Drop for Run<'_, '_> {
    #[inline(always)]
    fn drop(&mut self) {
        self.cab.value = self.value;
        self.cab.range = self.range;
    }
}

/// Where `codIOffset` starts in [`Cabac::value`].
const OFFSET_AT: u32 = 54;
/// The bits under the marker when at least eight stream bits are ahead.
const AHEAD: u64 = (1 << (OFFSET_AT - 8)) - 1;

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
            value: 0,
            range: 510,
            ctx: [0; CONTEXTS],
        };
        eng.load();
        eng.init_states(qp, init);
        Ok(eng)
    }

    /// Clause 9.3.1.2 on the bytes from `index`: the first nine bits are
    /// the offset, and the rest of the first four bytes are read ahead.
    fn load(&mut self) {
        let word = self.word();
        self.value = (u64::from(word) << 31) | (1 << 30);
    }

    /// The next four bytes, zeros past the end of the data.
    #[inline(always)]
    fn word(&mut self) -> u32 {
        let at = self.index;
        self.index += 4;
        if let Some(&[a, b, c, d]) = self.data.get(at..at + 4) {
            return u32::from_be_bytes([a, b, c, d]);
        }
        let byte = |i: usize| self.data.get(at + i).copied().unwrap_or(0);
        u32::from_be_bytes([byte(0), byte(1), byte(2), byte(3)])
    }

    /// The stream bits read ahead of the offset.
    fn ahead(&self) -> u32 {
        OFFSET_AT - 1 - self.value.trailing_zeros().min(OFFSET_AT - 1)
    }

    /// Bits of the data the offset has taken in so far.
    fn at_bit(&self) -> u64 {
        let read = u64::try_from(self.index.saturating_mul(8)).unwrap_or(u64::MAX);
        read.saturating_sub(u64::from(self.ahead()))
    }

    /// Whether the offset has taken in bits past the end of the data.
    fn overran(&self) -> bool {
        self.at_bit() > u64::try_from(self.data.len().saturating_mul(8)).unwrap_or(u64::MAX)
    }

    /// Four more bytes after the read-ahead bits, the first where the
    /// marker was, and the marker below them. Called with the marker at bit 46 or above.
    #[cold]
    #[inline(never)]
    fn refill(&mut self) {
        let marker = self.value.trailing_zeros();
        let word = u64::from(self.word());
        self.value = (self.value ^ (1 << marker)) | (word << (marker - 31)) | (1 << (marker - 32));
    }

    /// `codIRange` shifted back to 256..510 and the offset with it.
    #[inline(always)]
    fn renorm_by(&mut self, range: u32, value: u64) {
        let shift = range.leading_zeros() - 23;
        self.range = range << shift;
        self.value = value << shift;
        if self.value & AHEAD == 0 {
            self.refill();
        }
    }

    #[inline(always)]
    pub fn decision(&mut self, ctx: usize) -> Result<u8, CodecError> {
        // Which symbol comes out is as good as random to the CPU, so
        // nothing here branches on it: the two outcomes differ by a mask
        // and a table half.
        let packed = usize::from(self.ctx[ctx] & 127);
        let range_lps = u32::from(LPS_RANGE[packed][((self.range >> 6) & 3) as usize]);
        let range_mps = self.range - range_lps;
        let scaled = u64::from(range_mps) << OFFSET_AT;
        let lps = u64::from(self.value >= scaled);
        let mask = lps.wrapping_neg();
        self.ctx[ctx] = AFTER[packed + ((lps as usize) << 7)];
        let value = self.value - (scaled & mask);
        let range = range_mps ^ ((range_mps ^ range_lps) & mask as u32);
        self.renorm_by(range, value);
        Ok((packed as u8 & 1) ^ lps as u8)
    }

    /// Only the tests read a bypass bin outside a [`Run`].
    #[cfg(test)]
    #[inline(always)]
    pub fn bypass(&mut self) -> Result<u8, CodecError> {
        let mut value = self.value << 1;
        let scaled = u64::from(self.range) << OFFSET_AT;
        let one = value >= scaled;
        if one {
            value -= scaled;
        }
        self.value = value;
        if value & AHEAD == 0 {
            self.refill();
        }
        Ok(u8::from(one))
    }

    /// The decoder with its offset and range held apart, where the
    /// compiler can keep them in registers over a run of bins rather than
    /// store and load them for each; they are put back when the run ends.
    #[inline(always)]
    pub fn run(&mut self) -> Run<'_, 'a> {
        Run {
            value: self.value,
            range: self.range,
            cab: self,
        }
    }

    /// `end_of_slice_flag` and the like. Bits read past the end of the data
    /// come out here as an error: the bins before only see zeros there.
    pub fn terminate(&mut self) -> Result<bool, CodecError> {
        if self.overran() {
            return Err(CodecError::Truncated {
                at_bit: self.at_bit(),
                what: "cabac",
            });
        }
        self.range -= 2;
        let scaled = u64::from(self.range) << OFFSET_AT;
        if self.value >= scaled {
            Ok(true)
        } else {
            self.renorm_by(self.range, self.value);
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
        self.range = 510;
        self.load();
        Ok(())
    }

    /// The data after the byte holding the last bit the offset took in.
    pub fn tail(&self) -> &'a [u8] {
        let consumed = usize::try_from(self.at_bit()).unwrap_or(usize::MAX);
        self.data.get(consumed.div_ceil(8)..).unwrap_or(&[])
    }

    /// `codIOffset`.
    #[cfg(test)]
    fn offset(&self) -> u32 {
        (self.value >> OFFSET_AT) as u32
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
    use crate::tables::{RANGE_LPS, TRANS_LPS, TRANS_MPS};

    /// Clause 9.3.3.2 as written: one bit read at a time while the range
    /// is below 256.
    struct Plain<'a> {
        data: &'a [u8],
        bit: usize,
        range: u32,
        offset: u32,
        ctx: Vec<u8>,
    }

    impl Plain<'_> {
        fn read(&mut self) -> u32 {
            let byte = self.data.get(self.bit / 8).copied().unwrap_or(0);
            self.bit += 1;
            u32::from(byte >> (7 - (self.bit - 1) % 8)) & 1
        }

        fn renorm(&mut self) {
            while self.range < 256 {
                self.range <<= 1;
                self.offset = (self.offset << 1) | self.read();
            }
        }

        fn decision(&mut self, ctx: usize) -> u8 {
            let state = usize::from(self.ctx[ctx] >> 1);
            let mps = self.ctx[ctx] & 1;
            let lps = u32::from(RANGE_LPS[state][((self.range >> 6) & 3) as usize]);
            self.range -= lps;
            let bin = if self.offset >= self.range {
                self.offset -= self.range;
                self.range = lps;
                let swapped = if state == 0 { 1 - mps } else { mps };
                self.ctx[ctx] = (TRANS_LPS[state] << 1) | swapped;
                1 - mps
            } else {
                self.ctx[ctx] = (TRANS_MPS[state] << 1) | mps;
                mps
            };
            self.renorm();
            bin
        }

        fn bypass(&mut self) -> u8 {
            self.offset = (self.offset << 1) | self.read();
            if self.offset >= self.range {
                self.offset -= self.range;
                1
            } else {
                0
            }
        }

        fn terminate(&mut self) -> u8 {
            self.range -= 2;
            if self.offset >= self.range {
                1
            } else {
                self.renorm();
                0
            }
        }
    }

    /// The engine reads its bits ahead in bulk; every bin, range and offset
    /// must still be the clause's, through refills and past the end.
    #[test]
    fn bins_match_the_plain_engine() {
        let mut seed = 99u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) as u32
        };
        for round in 0..2_000 {
            let len = 1 + (next() % 300) as usize;
            let mut data: Vec<u8> = (0..len).map(|_| next() as u8).collect();
            // The first nine bits are below 510 in a stream.
            data[0] &= 0x7f;
            let mut bits = Bits::new(&data);
            let mut fast = Cabac::start(&mut bits, 26, 0).unwrap();
            let mut plain = Plain {
                data: &data,
                bit: 9,
                range: 510,
                offset: u32::from(data[0]) << 1 | u32::from(data.get(1).copied().unwrap_or(0) >> 7),
                ctx: fast.ctx.to_vec(),
            };
            for step in 0..4_000 {
                let r = next();
                let ctx = (r % 460) as usize;
                let (got, want) = match r % 50 {
                    0..=9 => (fast.bypass().unwrap(), plain.bypass()),
                    10 => match fast.terminate() {
                        Ok(bin) => (u8::from(bin), plain.terminate()),
                        Err(_) => {
                            assert!(plain.bit > data.len() * 8, "round {round} step {step}");
                            break;
                        }
                    },
                    // A run of bins with the offset and range held
                    // apart, each checked as it comes.
                    11..=20 => {
                        let mut run = fast.run();
                        let mut last = (0, 0);
                        for k in 0..(r >> 8) % 12 + 1 {
                            let ctx = (ctx + k as usize) % 460;
                            last = if k % 3 == 2 {
                                (run.bypass().unwrap(), plain.bypass())
                            } else {
                                (run.decision(ctx).unwrap(), plain.decision(ctx))
                            };
                            assert_eq!(last.0, last.1, "round {round} step {step} bin {k}");
                        }
                        last
                    }
                    _ => (fast.decision(ctx).unwrap(), plain.decision(ctx)),
                };
                assert_eq!(got, want, "round {round} step {step}");
                assert_eq!((fast.range, fast.offset()), (plain.range, plain.offset));
                if r % 50 == 10 && got == 1 {
                    assert_eq!(fast.at_bit(), plain.bit as u64);
                    break;
                }
            }
        }
    }

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
        assert_eq!(cab.offset(), 0b1_0100_1011);
        assert!(cab.restart(&[]).is_err());
    }
}
