//! Multisymbol arithmetic decoder (specification parsing process).
//!
//! CDFs are stored in the specification's forward form: for an alphabet of
//! `n` symbols the slice has length `n + 1`, `cdf[n - 1] == 32768`, and
//! `cdf[n]` is the adaptation counter (saturated at 32). This is not libaom's
//! inverse CDF.

use crate::bits::BitReader;
use crate::error::DecodeError;
use crate::math::floor_log2;

const EC_PROB_SHIFT: u32 = 6;
const EC_MIN_PROB: u32 = 4;

/// Tile symbol decoder over a byte-aligned window of `sz` bytes.
pub struct SymbolDecoder<'a> {
    bits: BitReader<'a>,
    value: u32,
    range: u32,
    max_bits: i32,
    /// Bit position in the parent buffer where this window started.
    origin: usize,
    update: bool,
}

impl<'a> SymbolDecoder<'a> {
    /// `init_symbol(sz)`. `bits` must be positioned at the first tile byte.
    pub fn init(bits: BitReader<'a>, sz: usize, update: bool) -> Result<Self, DecodeError> {
        if sz > (i32::MAX as usize) / 8 {
            return Err(DecodeError::malformed("tile symbol size"));
        }
        let origin = bits.position();
        let mut bits = bits;
        let num_bits = (sz * 8).min(15) as u32;
        let buf = bits.f(num_bits)?;
        let padded = buf << (15 - num_bits);
        let value = ((1u32 << 15) - 1) ^ padded;
        let max_bits = (sz as i32) * 8 - 15;
        Ok(Self {
            bits,
            value,
            range: 1 << 15,
            max_bits,
            origin,
            update,
        })
    }

    pub fn read_symbol(&mut self, cdf: &mut [u16]) -> Result<u32, DecodeError> {
        if cdf.len() < 3 {
            return Err(DecodeError::malformed("CDF shorter than a binary symbol"));
        }
        let n = cdf.len() - 1;
        if cdf[n - 1] != 1 << 15 {
            return Err(DecodeError::malformed("CDF is missing the 32768 sentinel"));
        }
        let mut cur = self.range;
        let mut symbol: i32 = -1;
        let mut prev;
        loop {
            symbol += 1;
            if symbol as usize >= n {
                return Err(DecodeError::malformed("symbol decoder failed to select"));
            }
            prev = cur;
            let f = (1u32 << 15) - u32::from(cdf[symbol as usize]);
            cur = ((self.range >> 8) * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT);
            cur += EC_MIN_PROB * (n as u32 - symbol as u32 - 1);
            if self.value >= cur {
                break;
            }
        }
        self.range = prev - cur;
        self.value -= cur;
        self.renorm()?;
        if self.update {
            adapt(cdf, symbol as usize);
        }
        Ok(symbol as u32)
    }

    /// `read_bool`. The temporary CDF is not adapted.
    pub fn read_bool(&mut self) -> Result<u32, DecodeError> {
        let saved = self.update;
        self.update = false;
        let mut cdf = [1 << 14, 1 << 15, 0];
        let bit = self.read_symbol(&mut cdf)?;
        self.update = saved;
        Ok(bit)
    }

    /// `read_literal(n)`.
    pub fn read_literal(&mut self, n: u32) -> Result<u32, DecodeError> {
        let mut x = 0u32;
        for _ in 0..n {
            x = (x << 1) | self.read_bool()?;
        }
        Ok(x)
    }

    /// `NS(n)` using arithmetic literals.
    #[allow(dead_code)]
    pub fn ns(&mut self, n: u32) -> Result<u32, DecodeError> {
        if n == 0 {
            return Err(DecodeError::malformed("NS(0)"));
        }
        let w = floor_log2(n) + 1;
        let m = (1u32 << w) - n;
        let v = self.read_literal(w - 1)?;
        if v < m {
            Ok(v)
        } else {
            let extra = self.read_literal(1)?;
            Ok((v << 1) - m + extra)
        }
    }

    /// `exit_symbol`. Returns the number of bits consumed from the window
    /// start, including the trailing marker the exit process skips.
    pub fn exit(mut self) -> Result<usize, DecodeError> {
        if self.max_bits < -14 {
            return Err(DecodeError::malformed(
                "symbol decoder exit with too much padding",
            ));
        }
        // trailingBitPosition = get_position() - Min(15, SymbolMaxBits + 15)
        let pos = self.bits.position();
        let min_term = 15.min(self.max_bits + 15);
        if pos < self.origin || (pos - self.origin) < min_term as usize && min_term > 0 {
            // The position is absolute in the parent buffer. The bits already
            // read include the initial 15 (or fewer) bits.
        }
        let trailing = pos as i32 - min_term;
        if trailing < 0 {
            return Err(DecodeError::malformed("trailing bit position"));
        }
        // Advance by Max(0, SymbolMaxBits) from the *current* position. The
        // spec's get_position already includes bits read during renorm, and
        // SymbolMaxBits is what remains. Skipping Max(0, SymbolMaxBits) lands
        // on paddingEndPosition.
        let skip = self.max_bits.max(0) as usize;
        self.bits.skip_bits(skip)?;
        let padding_end = self.bits.position();
        if !padding_end.is_multiple_of(8) {
            return Err(DecodeError::malformed(
                "symbol decoder exit is not byte aligned",
            ));
        }
        // Check the trailing one and zeros. Those bits sit in the original
        // window; re-read them from a fresh cursor.
        let window_end = padding_end;
        let check_from = trailing as usize;
        if check_from >= window_end {
            return Err(DecodeError::malformed("trailing bit is past the tile"));
        }
        let mut checker = BitReader::new(self.bits_data());
        // `bits_data` is the whole parent. Seek to trailing.
        checker.seek_bit(check_from)?;
        let marker = checker.read_bit()?;
        if marker != 1 {
            return Err(DecodeError::malformed("tile trailing bit is not 1"));
        }
        while checker.position() < window_end {
            if checker.read_bit()? != 0 {
                return Err(DecodeError::malformed("tile padding bit is not 0"));
            }
        }
        Ok(window_end - self.origin)
    }

    fn bits_data(&self) -> &'a [u8] {
        self.bits.data_slice()
    }

    fn renorm(&mut self) -> Result<(), DecodeError> {
        if self.range == 0 {
            return Err(DecodeError::malformed("symbol range collapsed"));
        }
        let bits = 15 - floor_log2(self.range);
        self.range <<= bits;
        let num_bits = bits.min(self.max_bits.max(0) as u32);
        let new_data = self.bits.f(num_bits)?;
        let padded = new_data << (bits - num_bits);
        self.value = padded ^ (((self.value + 1) << bits) - 1);
        self.max_bits -= bits as i32;
        Ok(())
    }
}

fn adapt(cdf: &mut [u16], symbol: usize) {
    let n = cdf.len() - 1;
    let count = u32::from(cdf[n]);
    let rate = 3 + u32::from(count > 15) + u32::from(count > 31) + floor_log2(n as u32).min(2);
    let mut tmp = 0u32;
    for (i, slot) in cdf.iter_mut().take(n - 1).enumerate() {
        if i == symbol {
            tmp = 1 << 15;
        }
        let cur = u32::from(*slot);
        if tmp < cur {
            *slot = (cur - ((cur - tmp) >> rate)) as u16;
        } else {
            *slot = (cur + ((tmp - cur) >> rate)) as u16;
        }
    }
    if cdf[n] < 32 {
        cdf[n] += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_binary_symbol_on_zero_byte() {
        // See the initialization worked example: one zero byte selects symbol 0
        // from cdf [16384, 32768, 0] without adaptation.
        let data = [0u8];
        let bits = BitReader::new(&data);
        let mut dec = SymbolDecoder::init(bits, 1, false).unwrap();
        let mut cdf = [16384u16, 32768, 0];
        assert_eq!(dec.read_symbol(&mut cdf).unwrap(), 0);
    }

    #[test]
    fn adaptation_moves_the_selected_symbol() {
        let data = [0u8, 0, 0];
        let bits = BitReader::new(&data);
        let mut dec = SymbolDecoder::init(bits, 3, true).unwrap();
        let mut cdf = [16384u16, 32768, 0];
        let _ = dec.read_symbol(&mut cdf).unwrap();
        assert!(cdf[0] > 16384);
        assert_eq!(cdf[2], 1);
    }
}
