//! Start-code search, emulation-prevention removal, and an MSB-first bit
//! reader for the few header fields framing needs.

/// Position of the next `00 00 01` at or after `from`.
///
/// Reads about one byte in three on data without zeros, the way decoders
/// commonly skip: a byte above 1 cannot be any of the three positions of a
/// start code that ends within two bytes of it.
#[must_use]
pub fn find_start_code(buf: &[u8], from: usize) -> Option<usize> {
    let mut at = from.checked_add(2)?;
    let len = buf.len();
    while at < len {
        let byte = buf[at];
        if byte > 1 {
            at += 3;
        } else if buf[at - 1] != 0 {
            at += 2;
        } else if buf[at - 2] != 0 || byte != 1 {
            at += 1;
        } else {
            return Some(at - 2);
        }
    }
    None
}

/// `true` when `buf` begins with `00 00 01` or `00 00 00 01`.
#[must_use]
pub fn starts_with_start_code(buf: &[u8]) -> bool {
    matches!(buf, [0, 0, 1, ..] | [0, 0, 0, 1, ..])
}

/// Copies up to `limit` bytes of RBSP from a NAL unit, dropping each
/// `emulation_prevention_three_byte` (H.264 7.4.1, H.265 7.4.2).
pub fn unescape(nal: &[u8], limit: usize, out: &mut Vec<u8>) {
    out.clear();
    let mut zeros = 0;
    for &byte in nal {
        if out.len() >= limit {
            break;
        }
        if zeros >= 2 && byte == 3 {
            zeros = 0;
            continue;
        }
        zeros = if byte == 0 { zeros + 1 } else { 0 };
        out.push(byte);
    }
}

/// Reads bits most significant first. Every read past the end is `None`.
pub struct Bits<'a> {
    bytes: &'a [u8],
    bit: usize,
}

impl<'a> Bits<'a> {
    #[must_use]
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, bit: 0 }
    }

    pub fn bit(&mut self) -> Option<bool> {
        let byte = *self.bytes.get(self.bit / 8)?;
        let value = (byte >> (7 - self.bit % 8)) & 1;
        self.bit += 1;
        Some(value == 1)
    }

    /// Up to 32 bits.
    pub fn bits(&mut self, count: u32) -> Option<u32> {
        let mut value = 0_u32;
        for _ in 0..count.min(32) {
            value = (value << 1) | u32::from(self.bit()?);
        }
        Some(value)
    }

    pub fn skip(&mut self, count: usize) -> Option<()> {
        let end = self.bit.checked_add(count)?;
        if end > self.bytes.len() * 8 {
            return None;
        }
        self.bit = end;
        Some(())
    }

    /// Unsigned Exp-Golomb, up to 31 leading zeros.
    pub fn ue(&mut self) -> Option<u32> {
        let mut zeros = 0;
        while !self.bit()? {
            zeros += 1;
            if zeros > 31 {
                return None;
            }
        }
        let rest = self.bits(zeros)?;
        let value = (1_u64 << zeros) - 1 + u64::from(rest);
        u32::try_from(value).ok()
    }

    /// Signed Exp-Golomb.
    pub fn se(&mut self) -> Option<i32> {
        let code = i64::from(self.ue()?);
        let value = if code % 2 == 1 {
            (code + 1) / 2
        } else {
            -(code / 2)
        };
        i32::try_from(value).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::{Bits, find_start_code, starts_with_start_code, unescape};

    fn naive(buf: &[u8], from: usize) -> Option<usize> {
        (from..buf.len().saturating_sub(2))
            .find(|&i| buf[i] == 0 && buf[i + 1] == 0 && buf[i + 2] == 1)
    }

    #[test]
    fn the_skipping_search_finds_what_a_byte_walk_finds() {
        let mut state = 0x1234_5678_u32;
        for round in 0..400 {
            let len = 8 + round % 90;
            let buf: Vec<u8> = (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    // Mostly zeros and ones so start codes are common.
                    [0, 0, 0, 1, 2, 0xFF][(state % 6) as usize]
                })
                .collect();
            let mut from = 0;
            loop {
                let found = find_start_code(&buf, from);
                assert_eq!(found, naive(&buf, from), "{buf:?} from {from}");
                match found {
                    Some(at) => from = at + 1,
                    None => break,
                }
            }
        }
        assert_eq!(find_start_code(&[0, 0, 1], 0), Some(0));
        assert_eq!(find_start_code(&[0, 1], 0), None);
        assert_eq!(find_start_code(&[], usize::MAX), None);
        assert!(starts_with_start_code(&[0, 0, 0, 1, 9]));
        assert!(!starts_with_start_code(&[0, 0, 2]));
    }

    #[test]
    fn emulation_prevention_bytes_are_dropped() {
        let mut out = Vec::new();
        unescape(&[0x65, 0, 0, 3, 1, 0, 0, 3, 0, 7], 64, &mut out);
        assert_eq!(out, vec![0x65, 0, 0, 1, 0, 0, 0, 7]);
        unescape(&[1, 2, 3, 4], 2, &mut out);
        assert_eq!(out, vec![1, 2]);
    }

    #[test]
    fn exp_golomb_codes_read_back() {
        // 1, 010, 011, 00100, 00101 => 0, 1, 2, 3, 4.
        let mut bits = Bits::new(&[0b1010_0110, 0b0100_0010, 0b1000_0000]);
        assert_eq!(bits.ue(), Some(0));
        assert_eq!(bits.ue(), Some(1));
        assert_eq!(bits.ue(), Some(2));
        assert_eq!(bits.ue(), Some(3));
        assert_eq!(bits.ue(), Some(4));
        let mut signed = Bits::new(&[0b0100_1100, 0b1000_0000]);
        assert_eq!(signed.se(), Some(1));
        assert_eq!(signed.se(), Some(-1));
        assert_eq!(signed.se(), Some(2));
        let mut empty = Bits::new(&[0]);
        assert_eq!(empty.ue(), None);
        assert_eq!(Bits::new(&[0xAB]).bits(4), Some(0xA));
    }
}
