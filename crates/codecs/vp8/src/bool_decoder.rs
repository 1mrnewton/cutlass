//! The boolean entropy decoder of RFC 6386 section 7.
//!
//! `value` holds two bytes: the high byte is compared against the split,
//! the low byte is the next eight bits. A read past the end of the
//! partition shifts in zeros and counts them in `overrun`; one such byte is
//! the look-ahead a well-formed partition always needs at its end, so the
//! caller treats two or more as a truncated partition.

use cutlass_decode::CodecError;

pub(crate) struct BoolDecoder<'a> {
    data: &'a [u8],
    at: usize,
    value: u32,
    range: u32,
    bit_count: u32,
    overrun: usize,
}

impl<'a> BoolDecoder<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        let mut decoder = Self {
            data,
            at: 0,
            value: 0,
            range: 255,
            bit_count: 0,
            overrun: 0,
        };
        let high = decoder.next_byte();
        let low = decoder.next_byte();
        decoder.value = (high << 8) | low;
        decoder
    }

    fn next_byte(&mut self) -> u32 {
        if let Some(&byte) = self.data.get(self.at) {
            self.at += 1;
            u32::from(byte)
        } else {
            self.overrun = self.overrun.saturating_add(1);
            0
        }
    }

    /// One bool whose chance of being zero is `prob / 256`.
    pub(crate) fn read_bool(&mut self, prob: u8) -> bool {
        let split = 1 + (((self.range - 1) * u32::from(prob)) >> 8);
        let big_split = split << 8;
        let bit = if self.value >= big_split {
            self.range -= split;
            self.value -= big_split;
            true
        } else {
            self.range = split;
            false
        };
        while self.range < 128 {
            self.value <<= 1;
            self.range <<= 1;
            self.bit_count += 1;
            if self.bit_count == 8 {
                self.bit_count = 0;
                self.value |= self.next_byte();
            }
        }
        bit
    }

    /// An `n`-bit unsigned literal, most significant bit first, each bit at
    /// even odds (`L(n)` in the RFC).
    pub(crate) fn read_literal(&mut self, bits: u32) -> u32 {
        let mut value = 0;
        for _ in 0..bits {
            value = (value << 1) | u32::from(self.read_bool(128));
        }
        value
    }

    pub(crate) fn read_flag(&mut self) -> bool {
        self.read_bool(128)
    }

    /// A magnitude of `bits` bits then a sign bit, as the header's signed
    /// deltas are stored.
    pub(crate) fn read_signed(&mut self, bits: u32) -> i32 {
        let magnitude = i32::try_from(self.read_literal(bits)).unwrap_or(0);
        if self.read_flag() {
            -magnitude
        } else {
            magnitude
        }
    }

    /// An optional signed delta: a presence flag, then [`Self::read_signed`].
    pub(crate) fn read_optional_signed(&mut self, bits: u32) -> i32 {
        if self.read_flag() {
            self.read_signed(bits)
        } else {
            0
        }
    }

    /// A value read with a tree in the RFC's layout: a positive entry is the
    /// index of the next pair, a negative or zero entry is minus the leaf.
    pub(crate) fn read_tree(&mut self, tree: &[i8], probs: &[u8], start: usize) -> u8 {
        let mut index = start;
        loop {
            let prob = probs.get(index >> 1).copied().unwrap_or(128);
            let next = tree
                .get(index + usize::from(self.read_bool(prob)))
                .copied()
                .unwrap_or(0);
            if next <= 0 {
                return next.unsigned_abs();
            }
            index = usize::from(next.unsigned_abs());
        }
    }

    /// Fail when the partition ran out: more than the one byte of
    /// look-ahead was made up.
    pub(crate) fn check(&self, what: &'static str) -> Result<(), CodecError> {
        if self.overrun >= 2 {
            return Err(CodecError::Truncated {
                at_bit: u64::try_from(self.data.len())
                    .unwrap_or(u64::MAX)
                    .saturating_mul(8),
                what,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::BoolDecoder;

    /// The RFC 6386 section 7.3 bool encoder, written out for the test so
    /// the decoder is checked against an independent writer.
    struct Encoder {
        out: Vec<u8>,
        range: u32,
        bottom: u32,
        bit_count: i32,
    }

    impl Encoder {
        fn new() -> Self {
            Self {
                out: Vec::new(),
                range: 255,
                bottom: 0,
                bit_count: 24,
            }
        }

        fn add_one(&mut self) {
            let mut at = self.out.len();
            while at > 0 {
                at -= 1;
                if self.out[at] == 255 {
                    self.out[at] = 0;
                } else {
                    self.out[at] += 1;
                    break;
                }
            }
        }

        fn write(&mut self, prob: u8, bit: bool) {
            let split = 1 + (((self.range - 1) * u32::from(prob)) >> 8);
            if bit {
                self.bottom = self.bottom.wrapping_add(split);
                self.range -= split;
            } else {
                self.range = split;
            }
            while self.range < 128 {
                self.range <<= 1;
                if self.bottom & (1 << 31) != 0 {
                    self.add_one();
                }
                self.bottom <<= 1;
                self.bit_count -= 1;
                if self.bit_count == 0 {
                    self.out.push((self.bottom >> 24) as u8);
                    self.bottom &= (1 << 24) - 1;
                    self.bit_count = 8;
                }
            }
        }

        fn finish(mut self) -> Vec<u8> {
            for _ in 0..32 {
                self.write(128, false);
            }
            self.out
        }
    }

    #[test]
    fn bools_at_every_probability_come_back() {
        let mut seed = 0x1234_5678_u32;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let mut written = Vec::new();
        let mut encoder = Encoder::new();
        for _ in 0..20_000 {
            let prob = u8::try_from(next() % 255 + 1).unwrap();
            // Skew the bits toward the probability so long runs happen.
            let bit = (next() % 256) >= u32::from(prob);
            encoder.write(prob, bit);
            written.push((prob, bit));
        }
        let bytes = encoder.finish();
        let mut decoder = BoolDecoder::new(&bytes);
        for (n, (prob, bit)) in written.into_iter().enumerate() {
            assert_eq!(decoder.read_bool(prob), bit, "bool {n}");
        }
        decoder.check("test").unwrap();
    }

    #[test]
    fn literals_trees_and_signs_read_in_order() {
        let mut encoder = Encoder::new();
        // 0b1011 as a four-bit literal.
        for bit in [true, false, true, true] {
            encoder.write(128, bit);
        }
        // Magnitude 5 in four bits, then the minus sign.
        for bit in [false, true, false, true, true] {
            encoder.write(128, bit);
        }
        // The key frame luma tree: "1 1 0" is H_PRED (leaf 2 of 4..7).
        let tree: [i8; 8] = [-4, 2, 4, 6, -0, -1, -2, -3];
        let probs = [145, 156, 163, 128];
        encoder.write(145, true);
        encoder.write(156, true);
        encoder.write(163, false);
        let bytes = encoder.finish();
        let mut decoder = BoolDecoder::new(&bytes);
        assert_eq!(decoder.read_literal(4), 0b1011);
        assert_eq!(decoder.read_signed(4), -5);
        assert_eq!(decoder.read_tree(&tree, &probs, 0), 2);
    }

    #[test]
    fn running_out_is_a_truncation_not_a_panic() {
        // One byte short of the look-ahead is still whole; eight more bits
        // past it are not.
        let mut decoder = BoolDecoder::new(&[0xFF]);
        decoder.check("short").unwrap();
        for _ in 0..64 {
            let _ = decoder.read_bool(128);
        }
        assert!(decoder.check("short").is_err());
        let mut empty = BoolDecoder::new(&[]);
        for _ in 0..1000 {
            let _ = empty.read_bool(1);
        }
        assert!(empty.check("empty").is_err());
        let one = [0x55, 0xAA];
        let mut decoder = BoolDecoder::new(&one);
        decoder.check("two bytes").unwrap();
        for _ in 0..64 {
            let _ = decoder.read_bool(128);
        }
        assert!(decoder.check("drained").is_err());
    }
}
