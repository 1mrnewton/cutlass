//! The boolean entropy encoder of RFC 6386 section 7.3, and the cost in
//! bits of coding a bit with a given probability.

use std::sync::OnceLock;

pub(crate) struct BoolEncoder {
    out: Vec<u8>,
    range: u32,
    bottom: u32,
    bit_count: i32,
}

impl BoolEncoder {
    #[cfg(test)]
    pub(crate) fn new() -> Self {
        Self::with_capacity(0)
    }

    pub(crate) fn with_capacity(bytes: usize) -> Self {
        Self {
            out: Vec::with_capacity(bytes),
            range: 255,
            bottom: 0,
            bit_count: 24,
        }
    }

    /// Carry into the bytes already written.
    fn add_one(&mut self) {
        for byte in self.out.iter_mut().rev() {
            if *byte == 255 {
                *byte = 0;
            } else {
                *byte += 1;
                break;
            }
        }
    }

    /// One bit that is 0 with probability `prob / 256`.
    #[inline]
    pub(crate) fn put(&mut self, bit: bool, prob: u8) {
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

    pub(crate) fn flag(&mut self, bit: bool) {
        self.put(bit, 128);
    }

    /// An unsigned `count`-bit literal, most significant bit first.
    pub(crate) fn literal(&mut self, value: u32, count: u32) {
        for shift in (0..count).rev() {
            self.flag((value >> shift) & 1 != 0);
        }
    }

    /// A `count`-bit magnitude and then its sign.
    pub(crate) fn signed(&mut self, value: i32, count: u32) {
        self.literal(value.unsigned_abs(), count);
        self.flag(value < 0);
    }

    /// A flag, then the signed value when it is not zero.
    pub(crate) fn optional_signed(&mut self, value: i32, count: u32) {
        self.flag(value != 0);
        if value != 0 {
            self.signed(value, count);
        }
    }

    /// The leaf `value` of a section 8.1 tree.
    pub(crate) fn tree(&mut self, tree: &[i8], probs: &[u8], value: usize) {
        for (node, bit) in tree_path(tree, value) {
            self.put(bit, probs[node >> 1]);
        }
    }

    pub(crate) fn finish(mut self) -> Vec<u8> {
        for _ in 0..32 {
            self.flag(false);
        }
        self.out
    }
}

/// The nodes and bits from the root of `tree` to the leaf `value`.
pub(crate) fn tree_path(tree: &[i8], value: usize) -> Vec<(usize, bool)> {
    fn walk(tree: &[i8], index: usize, value: usize, path: &mut Vec<(usize, bool)>) -> bool {
        for bit in [false, true] {
            let next = tree[index + usize::from(bit)];
            path.push((index, bit));
            if next <= 0 {
                if usize::from(next.unsigned_abs()) == value {
                    return true;
                }
            } else if walk(tree, usize::from(next.unsigned_abs()), value, path) {
                return true;
            }
            path.pop();
        }
        false
    }
    let mut path = Vec::new();
    walk(tree, 0, value, &mut path);
    path
}

/// Bits, in 1/256ths, of coding a 0 with probability `prob / 256`;
/// a 1 costs `COSTS[255 - prob]` plus a little, see [`cost`].
fn costs() -> &'static [u32; 257] {
    static COSTS: OnceLock<[u32; 257]> = OnceLock::new();
    COSTS.get_or_init(|| {
        std::array::from_fn(|i| {
            let p = (i as f64).max(0.5) / 256.0;
            (-p.log2() * 256.0).round() as u32
        })
    })
}

/// Bits, in 1/256ths, of coding `bit` with probability `prob / 256` of 0.
#[inline]
pub(crate) fn cost(bit: bool, prob: u8) -> u32 {
    let table = costs();
    if bit {
        table[256 - usize::from(prob)]
    } else {
        table[usize::from(prob)]
    }
}

#[cfg(test)]
mod tests {
    use super::{BoolEncoder, cost, tree_path};
    use crate::lossy::bool_decoder::BoolDecoder;
    use crate::lossy::tables::B_MODE_TREE;

    #[test]
    fn literals_signed_values_and_trees_read_back() {
        let mut encoder = BoolEncoder::new();
        encoder.literal(0x5a, 7);
        encoder.signed(-9, 4);
        encoder.optional_signed(0, 6);
        encoder.optional_signed(13, 6);
        let probs = [120u8, 90, 200, 30, 66, 140, 180, 20, 250];
        for mode in 0..10 {
            encoder.tree(&B_MODE_TREE, &probs, mode);
        }
        let bytes = encoder.finish();
        let mut decoder = BoolDecoder::new(&bytes);
        assert_eq!(decoder.literal(7), 0x5a);
        assert_eq!(decoder.signed(4), -9);
        assert_eq!(decoder.optional_signed(6), 0);
        assert_eq!(decoder.optional_signed(6), 13);
        for mode in 0..10 {
            assert_eq!(decoder.tree(&B_MODE_TREE, &probs), mode);
        }
    }

    #[test]
    fn every_mode_has_a_path_and_costs_fall_with_probability() {
        for mode in 0..10 {
            assert!(!tree_path(&B_MODE_TREE, mode).is_empty());
        }
        assert_eq!(cost(false, 128), 256);
        assert_eq!(cost(true, 128), 256);
        assert!(cost(false, 250) < cost(false, 10));
        assert!(cost(true, 250) > cost(true, 10));
        assert_eq!(cost(true, 0), 0);
    }
}
