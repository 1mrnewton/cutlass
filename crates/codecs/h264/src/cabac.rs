//! CABAC, clause 9.3. Context variables `m` and `n` are Tables 9-12 to 9-33.

use cutlass_decode::CodecError;

use crate::bits::Bits;
use crate::tables::{CABAC_MN, LAST_8X8, RANGE_LPS, SIG_8X8, TRANS_LPS, TRANS_MPS};

pub struct Cabac<'a> {
    data: &'a [u8],
    bit: usize,
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
            bit: 0,
            range: 510,
            offset: 0,
            state: [0; 1024],
            mps: [0; 1024],
        };
        for _ in 0..9 {
            eng.offset = (eng.offset << 1) | eng.read_bit()?;
        }
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

    fn read_bit(&mut self) -> Result<u32, CodecError> {
        let byte_index = self.bit / 8;
        // Clause 9.3.3.2.2 reads a zero once the slice is exhausted. cabac_zero_word is that padding.
        if byte_index >= self.data.len() {
            return Err(CodecError::Truncated {
                at_bit: self.bit as u64,
                what: "cabac",
            });
        }
        let shift = 7 - (self.bit % 8);
        let bit = u32::from((self.data[byte_index] >> shift) & 1);
        self.bit += 1;
        Ok(bit)
    }

    fn renorm(&mut self) -> Result<(), CodecError> {
        while self.range < 256 {
            self.range <<= 1;
            self.offset = (self.offset << 1) | self.read_bit()?;
        }
        Ok(())
    }

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

    pub fn bypass(&mut self) -> Result<u8, CodecError> {
        self.offset = (self.offset << 1) | self.read_bit()?;
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
            bit: 0,
            range: 510,
            offset: 0,
            state: [0; 1024],
            mps: [0; 1024],
        };
        for _ in 0..9 {
            eng.offset = (eng.offset << 1) | eng.read_bit()?;
        }
        eng.init_states(qp, init);
        Ok(eng)
    }

    pub fn tail(&self) -> &'a [u8] {
        let byte = self.bit.div_ceil(8);
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
