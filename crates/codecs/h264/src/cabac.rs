//! CABAC, clause 9.3. Context variables `m` and `n` are Tables 9-12 to 9-33.

use cutlass_decode::CodecError;

use crate::bits::Bits;
use crate::tables::{CABAC_MN, LAST_8X8, RANGE_LPS, SIG_8X8, TRANS_LPS, TRANS_MPS};

pub struct Cabac<'a> {
    data: &'a [u8],
    /// Next byte that has not been pulled into `cache`.
    index: usize,
    cache: u64,
    /// Valid low bits of `cache`.
    bits: u32,
    range: u32,
    offset: u32,
    state: [u8; 1024],
    mps: [u8; 1024],
}

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
            state: [0; 1024],
            mps: [0; 1024],
        };
        eng.offset = eng.read_bits(9)?;
        let slice_qp = qp.clamp(0, 51);
        for (ctx, &(m, n)) in CABAC_MN[init].iter().enumerate() {
            if ctx == 276 {
                eng.state[276] = 63;
                eng.mps[276] = 0;
                continue;
            }
            let pre = (((i32::from(m) * slice_qp) >> 4) + i32::from(n)).clamp(1, 126);
            if pre <= 63 {
                eng.state[ctx] = (63 - pre) as u8;
                eng.mps[ctx] = 0;
            } else {
                eng.state[ctx] = (pre - 64) as u8;
                eng.mps[ctx] = 1;
            }
        }
        Ok(eng)
    }

    fn at_bit(&self) -> u64 {
        u64::try_from(self.index.saturating_mul(8)).unwrap_or(u64::MAX) - u64::from(self.bits)
    }

    fn fill(&mut self) {
        while self.bits <= 56 && self.index < self.data.len() {
            self.cache = (self.cache << 8) | u64::from(self.data[self.index]);
            self.index += 1;
            self.bits += 8;
        }
    }

    #[inline(always)]
    fn read_bits(&mut self, n: u32) -> Result<u32, CodecError> {
        self.fill();
        if self.bits < n {
            return Err(CodecError::Truncated {
                at_bit: self.at_bit(),
                what: "cabac",
            });
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
        let state = usize::from(self.state[ctx]);
        let mps = self.mps[ctx];
        let range_lps = u32::from(RANGE_LPS[state][((self.range >> 6) & 3) as usize]);
        self.range -= range_lps;
        let bin = if self.offset >= self.range {
            let bin = 1 - mps;
            self.offset -= self.range;
            self.range = range_lps;
            if state == 0 {
                self.mps[ctx] = 1 - mps;
            }
            self.state[ctx] = TRANS_LPS[state];
            bin
        } else {
            self.state[ctx] = TRANS_MPS[state];
            mps
        };
        self.renorm()?;
        Ok(bin)
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

    pub fn open(data: &'a [u8], qp: i32, init: usize) -> Result<Self, CodecError> {
        if data.is_empty() {
            return Err(CodecError::Truncated {
                at_bit: 0,
                what: "cabac",
            });
        }
        let mut eng = Self {
            data,
            index: 0,
            cache: 0,
            bits: 0,
            range: 510,
            offset: 0,
            state: [0; 1024],
            mps: [0; 1024],
        };
        eng.offset = eng.read_bits(9)?;
        eng.init_states(qp, init);
        Ok(eng)
    }

    pub fn tail(&self) -> &'a [u8] {
        let consumed = self.index * 8 - usize::try_from(self.bits).unwrap_or(0);
        let byte = consumed.div_ceil(8);
        self.data.get(byte..).unwrap_or(&[])
    }

    fn init_states(&mut self, qp: i32, init: usize) {
        let slice_qp = qp.clamp(0, 51);
        for (ctx, &(m, n)) in CABAC_MN[init].iter().enumerate() {
            if ctx == 276 {
                self.state[276] = 63;
                self.mps[276] = 0;
                continue;
            }
            let pre = (((i32::from(m) * slice_qp) >> 4) + i32::from(n)).clamp(1, 126);
            if pre <= 63 {
                self.state[ctx] = (63 - pre) as u8;
                self.mps[ctx] = 0;
            } else {
                self.state[ctx] = (pre - 64) as u8;
                self.mps[ctx] = 1;
            }
        }
    }
}

/// Significant-coefficient contexts for an 8x8 block, Table 9-43 frame column.
pub fn sig8(index: usize) -> usize {
    usize::from(SIG_8X8[index])
}

pub fn last8(index: usize) -> usize {
    usize::from(LAST_8X8[index])
}
