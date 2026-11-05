//! The boolean entropy decoder of RFC 6386 section 7.
//!
//! Arithmetic is the RFC's. Bytes are loaded only when a bit needs them,
//! and a bit that needs a byte past the end sets [`BoolDecoder::eof`] and
//! reads zeros, as libwebp does. The caller refuses the frame when `eof` is
//! set after a macroblock, so a short partition is an error, never a guess.

pub(crate) struct BoolDecoder<'a> {
    data: &'a [u8],
    next: usize,
    /// Loaded bits. The comparison window is `value >> bits`.
    value: u64,
    /// Loaded bits beyond the 8-bit window; negative when a load is due.
    bits: i32,
    /// The range minus one, 127 to 254 between bits.
    range: u32,
    eof: bool,
}

impl<'a> BoolDecoder<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        let mut decoder = Self {
            data,
            next: 0,
            value: 0,
            bits: -8,
            range: 254,
            eof: false,
        };
        decoder.load();
        decoder
    }

    /// A bit needed a byte past the end of the partition.
    pub(crate) fn eof(&self) -> bool {
        self.eof
    }

    fn load(&mut self) {
        if let Some(chunk) = self.data.get(self.next..self.next + 8) {
            // Seven bytes at a time while eight remain.
            let word = u64::from_be_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]);
            self.value = (self.value << 56) | (word >> 8);
            self.bits += 56;
            self.next += 7;
        } else if let Some(&byte) = self.data.get(self.next) {
            self.value = (self.value << 8) | u64::from(byte);
            self.bits += 8;
            self.next += 1;
        } else if !self.eof {
            self.value <<= 8;
            self.bits += 8;
            self.eof = true;
        } else {
            // Past the end twice: the frame is refused; keep shifts in range.
            self.bits = 0;
        }
    }

    /// One bit that is 1 with probability `(256 - prob) / 256`.
    #[inline]
    pub(crate) fn bit(&mut self, prob: u8) -> bool {
        if self.bits < 0 {
            self.load();
        }
        let shift = self.bits.max(0).unsigned_abs();
        let split = (self.range * u32::from(prob)) >> 8;
        // `value >> shift` stays below 256 by construction.
        let value = (self.value >> shift) as u32;
        let (bit, range) = if value > split {
            self.value -= u64::from(split + 1) << shift;
            (true, self.range - split)
        } else {
            (false, split + 1)
        };
        // `range` is 1 to 255 here; bring it back to 128 to 255.
        let renorm = 7 ^ (31 - range.leading_zeros());
        self.range = (range << renorm) - 1;
        self.bits -= renorm.cast_signed();
        bit
    }

    /// An unsigned `count`-bit literal, most significant bit first.
    pub(crate) fn literal(&mut self, count: u32) -> u32 {
        let mut value = 0;
        for _ in 0..count {
            value = (value << 1) | u32::from(self.bit(128));
        }
        value
    }

    /// A `count`-bit magnitude and then its sign.
    pub(crate) fn signed(&mut self, count: u32) -> i32 {
        let magnitude = self.literal(count).cast_signed();
        if self.bit(128) { -magnitude } else { magnitude }
    }

    /// A flag, then a signed value when the flag is set.
    pub(crate) fn optional_signed(&mut self, count: u32) -> i32 {
        if self.bit(128) { self.signed(count) } else { 0 }
    }

    pub(crate) fn flag(&mut self) -> bool {
        self.bit(128)
    }

    /// Walk a section 8.1 tree.
    pub(crate) fn tree(&mut self, tree: &[i8], probs: &[u8]) -> usize {
        let mut index = 0usize;
        loop {
            let next = tree[index + usize::from(self.bit(probs[index >> 1]))];
            if next <= 0 {
                return usize::from(next.unsigned_abs());
            }
            index = usize::from(next.unsigned_abs());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BoolDecoder;
    use crate::lossy::bool_encoder::BoolEncoder;

    #[test]
    fn bits_written_by_the_rfc_encoder_come_back() {
        let mut encoder = BoolEncoder::new();
        let mut expected = Vec::new();
        let mut seed = 12345u32;
        for _ in 0..5000 {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            let prob = ((seed >> 8) & 0xff) as u8;
            let prob = prob.max(1);
            let bit = (seed >> 20) % 256 >= u32::from(prob);
            encoder.put(bit, prob);
            expected.push((prob, bit));
        }
        let bytes = encoder.finish();
        let mut decoder = BoolDecoder::new(&bytes);
        for (index, (prob, bit)) in expected.into_iter().enumerate() {
            assert_eq!(decoder.bit(prob), bit, "bit {index}");
        }
        assert!(!decoder.eof());
    }

    #[test]
    fn reading_past_the_end_sets_eof() {
        let mut decoder = BoolDecoder::new(&[0x55]);
        for _ in 0..64 {
            decoder.bit(128);
        }
        assert!(decoder.eof());
    }
}
