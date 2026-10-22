//! Multisymbol range encoder.
//!
//! This is the inverse of [`crate::symbol::SymbolDecoder`], ported from
//! libaom `od_ec_encode_cdf_q15` / `od_ec_enc_done`. It builds tile payloads
//! for tests. It is not an AV1 video encoder.

use crate::error::DecodeError;
use crate::math::floor_log2;

/// Arithmetic encoder over one tile window.
pub struct SymbolEncoder {
    buf: Vec<u8>,
    offs: usize,
    low: u64,
    rng: u32,
    cnt: i32,
}

impl SymbolEncoder {
    pub fn new() -> Self {
        Self {
            buf: vec![0; 64],
            offs: 0,
            low: 0,
            rng: 0x8000,
            cnt: -9,
        }
    }

    /// Encode `symbol` with a specification forward CDF (`cdf[n-1] == 32768`).
    pub fn encode_symbol(&mut self, symbol: usize, cdf: &[u16]) -> Result<(), DecodeError> {
        let nsyms = cdf.len() - 1;
        if nsyms < 2 || symbol >= nsyms || cdf[nsyms - 1] != 32768 {
            return Err(DecodeError::malformed("encoder symbol"));
        }
        let icdf = |i: usize| -> u32 {
            if i + 1 == nsyms {
                0
            } else {
                32768 - u32::from(cdf[i])
            }
        };
        let fl = if symbol > 0 { icdf(symbol - 1) } else { 32768 };
        let fh = icdf(symbol);
        self.encode_q15(fl, fh, symbol, nsyms)
    }

    /// `L(1)` as a boolean symbol with probability 1/2. Not adapted.
    pub fn encode_bool(&mut self, bit: u32) -> Result<(), DecodeError> {
        if bit > 1 {
            return Err(DecodeError::malformed("encoder bool"));
        }
        self.encode_symbol(bit as usize, &[16384, 32768, 0])
    }

    /// Flush the minimum bits that decode to the encoded symbols.
    pub fn finish(mut self) -> Result<Vec<u8>, DecodeError> {
        let l = self.low;
        let c = self.cnt;
        let mut s = 10 + c;
        let m = 0x3FFFu64;
        let mut e = ((l + m) & !m) | (m + 1);
        let mut offs = self.offs;
        if s > 0 {
            if !(0..64).contains(&(c + 16)) {
                return Err(DecodeError::malformed("encoder termination shift"));
            }
            let mut shift = c + 16;
            let mut n = (1u64 << shift) - 1;
            while s > 0 {
                if offs + 1 > self.buf.len() {
                    self.buf.resize(offs + 8, 0);
                }
                let val = (e >> shift) as u16;
                self.buf[offs] = (val & 0xff) as u8;
                if val & 0x0100 != 0 {
                    propagate(&mut self.buf, offs)?;
                }
                offs += 1;
                e &= n;
                s -= 8;
                shift -= 8;
                n >>= 8;
            }
        }
        self.buf.truncate(offs);
        Ok(self.buf)
    }

    fn encode_q15(&mut self, fl: u32, fh: u32, s: usize, nsyms: usize) -> Result<(), DecodeError> {
        let mut low = self.low;
        let mut rng = self.rng;
        let n = (nsyms - 1) as u32;
        if fl < 32768 {
            if s == 0 {
                return Err(DecodeError::malformed("encoder fl"));
            }
            let u = (((rng >> 8) * (fl >> 6)) >> 1) + 4 * (n - (s as u32 - 1));
            let v = (((rng >> 8) * (fh >> 6)) >> 1) + 4 * (n - s as u32);
            if u < v || rng < u {
                return Err(DecodeError::malformed("encoder range split"));
            }
            low += u64::from(rng - u);
            rng = u - v;
        } else {
            let sub = (((rng >> 8) * (fh >> 6)) >> 1) + 4 * (n - s as u32);
            if rng < sub {
                return Err(DecodeError::malformed("encoder range split"));
            }
            rng -= sub;
        }
        self.normalize(low, rng)
    }

    fn normalize(&mut self, mut low: u64, rng: u32) -> Result<(), DecodeError> {
        if rng == 0 || rng > 65535 {
            return Err(DecodeError::malformed("encoder range"));
        }
        let mut c = self.cnt;
        let d = 15 - floor_log2(rng) as i32;
        let mut s = c + d;
        if s >= 40 {
            let num_bytes_ready = ((s >> 3) + 1) as usize;
            if num_bytes_ready == 0 || num_bytes_ready > 8 {
                return Err(DecodeError::malformed("encoder flush width"));
            }
            c += 24 - (num_bytes_ready as i32) * 8;
            if !(0..64).contains(&c) {
                return Err(DecodeError::malformed("encoder count"));
            }
            let mut output = low >> c;
            low &= (1u64 << c) - 1;
            let carry_bit = 1u64 << (num_bytes_ready * 8);
            let carry = output & carry_bit != 0;
            output &= carry_bit - 1;
            self.write_ready(output, carry, num_bytes_ready)?;
            s = c + d - 24;
        }
        self.low = low << d;
        self.rng = rng << d;
        self.cnt = s;
        Ok(())
    }

    fn write_ready(&mut self, output: u64, carry: bool, num: usize) -> Result<(), DecodeError> {
        if self.offs + 8 > self.buf.len() {
            self.buf.resize(self.offs * 2 + 16, 0);
        }
        let reg = output << ((8 - num) * 8);
        let bytes = reg.to_be_bytes();
        let offs = self.offs;
        self.buf[offs..offs + 8].copy_from_slice(&bytes);
        if carry {
            propagate(&mut self.buf, offs)?;
        }
        self.offs = offs + num;
        Ok(())
    }
}

fn propagate(buf: &mut [u8], offs: usize) -> Result<(), DecodeError> {
    if offs == 0 {
        return Err(DecodeError::malformed("encoder carry"));
    }
    let mut offs = offs - 1;
    loop {
        let sum = u16::from(buf[offs]) + 1;
        buf[offs] = sum as u8;
        if sum < 256 {
            return Ok(());
        }
        if offs == 0 {
            return Err(DecodeError::malformed("encoder carry"));
        }
        offs -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitReader;
    use crate::symbol::SymbolDecoder;

    #[test]
    fn encoded_symbols_round_trip_and_exit() {
        let cdf = [16384u16, 24576, 32768, 0];
        let mut enc = SymbolEncoder::new();
        enc.encode_symbol(0, &cdf).unwrap();
        enc.encode_symbol(2, &cdf).unwrap();
        enc.encode_symbol(1, &cdf).unwrap();
        enc.encode_bool(1).unwrap();
        let bytes = enc.finish().unwrap();
        assert!(!bytes.is_empty());
        let mut dec = SymbolDecoder::init(BitReader::new(&bytes), bytes.len(), false).unwrap();
        let mut row = cdf;
        assert_eq!(dec.read_symbol(&mut row).unwrap(), 0);
        row = cdf;
        assert_eq!(dec.read_symbol(&mut row).unwrap(), 2);
        row = cdf;
        assert_eq!(dec.read_symbol(&mut row).unwrap(), 1);
        assert_eq!(dec.read_bool().unwrap(), 1);
        dec.exit().unwrap();
    }
}
