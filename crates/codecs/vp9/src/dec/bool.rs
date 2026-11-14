//! The boolean decoder of the tile data (clause 9.2 of the VP9
//! specification), built for speed: the window holds up to 64 bits of the
//! stream, refilled eight bytes at a time, and a bit costs one multiply,
//! one compare and one shift. Reading past the end of the data reads zero
//! bits, as libvpx does; [`Bool::overrun`] tells whether that happened by
//! more than the padding a valid stream may use.

/// The arithmetic decoder of one tile, or of the compressed header.
#[derive(Clone, Copy)]
pub struct Bool<'a> {
    data: &'a [u8],
    pos: usize,
    /// The top `count + 8` bits are stream bits not yet consumed.
    value: u64,
    /// How many valid bits `value` holds past its top 8.
    count: i32,
    range: u32,
}

impl<'a> Bool<'a> {
    /// Starts on `data` and reads the marker bit, which must be 0.
    pub fn new(data: &'a [u8]) -> Option<Self> {
        if data.is_empty() {
            return None;
        }
        let mut b = Self {
            data,
            pos: 0,
            value: 0,
            count: -8,
            range: 255,
        };
        b.fill();
        if b.read(128) {
            return None;
        }
        Some(b)
    }

    #[inline(always)]
    fn fill(&mut self) {
        let free = 56 - self.count; // bits that fit under the valid ones, less 8
        if let Some(chunk) = self.data.get(self.pos..).and_then(|s| s.first_chunk::<8>()) {
            let big = u64::from_be_bytes(*chunk);
            let bytes = (free >> 3) as usize;
            let keep = (bytes * 8) as u32;
            let taken = big & !u64::MAX.checked_shr(keep).unwrap_or(0);
            self.value |= taken >> (self.count + 8);
            self.count += keep as i32;
            self.pos += bytes;
        } else {
            self.fill_slow();
        }
    }

    #[cold]
    fn fill_slow(&mut self) {
        let mut shift = 48 - self.count;
        while shift >= 0 {
            let byte = match self.data.get(self.pos) {
                Some(&b) => {
                    self.pos += 1;
                    b
                }
                None => {
                    // Past the end: zero bits, and count them as read so
                    // that `overrun` can see it.
                    self.pos += 1;
                    0
                }
            };
            self.value |= u64::from(byte) << shift;
            self.count += 8;
            shift -= 8;
        }
    }

    /// One bit whose probability of being 0 is `prob / 256`.
    #[inline(always)]
    pub fn read(&mut self, prob: u8) -> bool {
        if self.count < 0 {
            self.fill();
        }
        let split = 1 + (((self.range - 1) * u32::from(prob)) >> 8);
        let big = u64::from(split) << 56;
        let bit = self.value >= big;
        if bit {
            self.range -= split;
            self.value -= big;
        } else {
            self.range = split;
        }
        // `range` is 1 to 255 here.
        let shift = self.range.leading_zeros() - 24;
        self.range <<= shift;
        self.value <<= shift;
        self.count -= shift as i32;
        bit
    }

    /// A bit of probability one half.
    #[inline(always)]
    pub fn bit(&mut self) -> bool {
        self.read(128)
    }

    /// A symbol of a tree (libvpx `vpx_read_tree`): `tree[i]` is a leaf
    /// `-symbol` when not positive, else the index of the next pair.
    #[inline]
    pub fn tree(&mut self, tree: &[i8], probs: &[u8]) -> u8 {
        let mut i = 0usize;
        loop {
            let p = probs.get(i >> 1).copied().unwrap_or(128);
            let next = tree
                .get(i + usize::from(self.read(p)))
                .copied()
                .unwrap_or(0);
            if next <= 0 {
                return (-next) as u8;
            }
            i = next as usize;
        }
    }

    /// Whether the decoder read further past the end of its data than the
    /// padding a valid stream can need.
    pub fn overrun(&self) -> bool {
        // Bits consumed = bytes loaded * 8 - bits still held.
        let held = (self.count + 8).max(0) as usize;
        (self.pos * 8).saturating_sub(held) > self.data.len() * 8 + 16
    }
}

#[cfg(test)]
mod tests {
    use super::Bool;
    use crate::bits::{BoolDecoder, BoolWriter};

    #[test]
    fn matches_the_reference_decoder() {
        let mut seed = 12345u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            seed >> 8
        };
        let mut w = BoolWriter::new();
        let mut bits = Vec::new();
        for _ in 0..20000 {
            let prob = (rnd() % 255 + 1) as u8;
            let bit = (rnd() % 256) >= u32::from(prob);
            bits.push((prob, bit));
            w.write(u32::from(bit), u32::from(prob));
        }
        let data = w.finish().expect("bytes");
        let mut fast = Bool::new(&data).expect("marker");
        let mut slow = BoolDecoder::new(&data).expect("marker");
        for &(prob, bit) in &bits {
            let a = fast.read(prob);
            let b = slow.read(u32::from(prob)).expect("bit") == 1;
            assert_eq!(a, b);
            assert_eq!(a, bit);
        }
        assert!(!fast.overrun());
    }
}
