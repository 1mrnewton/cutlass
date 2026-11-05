//! The boolean entropy coder of RFC 6386 Section 7.
//!
//! The decoder keeps up to 64 bits of the partition in a window, as the
//! reference decoder's two-byte value does one byte at a time. Bytes past
//! the end of a partition read as zero, the way every shipping decoder
//! reads them. [`BoolDecoder::overread`] says how far past the end the
//! coder went.

/// Reads bools from one partition.
#[derive(Clone, Debug)]
pub struct BoolDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    /// The coded value, left-aligned: bit 63 is the next bit to compare.
    value: u64,
    /// Valid bits in `value` past the top 8. Below zero means a refill.
    count: i32,
    range: u32,
    /// Zero bytes supplied past the end of `data`.
    zeros: usize,
}

impl<'a> BoolDecoder<'a> {
    #[must_use]
    pub fn new(data: &'a [u8]) -> Self {
        let mut decoder = Self {
            data,
            pos: 0,
            value: 0,
            count: -8,
            range: 255,
            zeros: 0,
        };
        decoder.fill();
        decoder
    }

    fn fill(&mut self) {
        // Bits wanted below the 8-bit compare window.
        let mut shift = 64 - 8 - (self.count + 8);
        while shift >= 0 {
            let byte = if let Some(&byte) = self.data.get(self.pos) {
                self.pos += 1;
                byte
            } else {
                self.zeros += 1;
                0
            };
            self.value |= u64::from(byte) << shift;
            self.count += 8;
            shift -= 8;
        }
    }

    /// One bool whose probability of being zero is `prob / 256`.
    #[inline]
    pub fn read(&mut self, prob: u8) -> bool {
        let split = 1 + (((self.range - 1) * u32::from(prob)) >> 8);
        if self.count < 0 {
            self.fill();
        }
        let big = u64::from(split) << 56;
        let bit = if self.value >= big {
            self.range -= split;
            self.value -= big;
            true
        } else {
            self.range = split;
            false
        };
        let shift = self.range.leading_zeros().saturating_sub(24);
        self.range <<= shift;
        self.value <<= shift;
        // `shift` is at most 7.
        self.count -= i32::try_from(shift).unwrap_or(7);
        bit
    }

    #[inline]
    pub fn bit(&mut self) -> bool {
        self.read(128)
    }

    /// An unsigned literal, most significant bit first.
    pub fn literal(&mut self, bits: u32) -> u32 {
        let mut value = 0;
        for _ in 0..bits {
            value = (value << 1) | u32::from(self.bit());
        }
        value
    }

    /// A magnitude of `bits` bits followed by a sign bit.
    pub fn signed(&mut self, bits: u32) -> i32 {
        let magnitude = i32::try_from(self.literal(bits)).unwrap_or(0);
        if self.bit() { -magnitude } else { magnitude }
    }

    /// A flag, then a signed value when the flag is set; 0 otherwise.
    pub fn maybe_signed(&mut self, bits: u32) -> i32 {
        if self.bit() { self.signed(bits) } else { 0 }
    }

    /// Walks a tree whose leaves are stored as non-positive values.
    #[inline]
    pub fn tree(&mut self, tree: &[i8], probs: &[u8]) -> u8 {
        self.tree_from(tree, probs, 0)
    }

    /// Walks a tree from node index `start`.
    #[inline]
    pub fn tree_from(&mut self, tree: &[i8], probs: &[u8], start: usize) -> u8 {
        let mut index = start;
        loop {
            let prob = probs.get(index >> 1).copied().unwrap_or(128);
            let next = tree
                .get(index + usize::from(self.read(prob)))
                .copied()
                .unwrap_or(0);
            if next <= 0 {
                return next.unsigned_abs();
            }
            index = usize::from(next.unsigned_abs());
        }
    }

    /// Zero bytes the coder has supplied past the end of its partition.
    /// The window reads ahead, so a few are normal at the end.
    #[must_use]
    pub fn overread(&self) -> usize {
        self.zeros
    }
}

/// Writes bools into one partition, as RFC 6386 Section 7.3 does.
#[derive(Clone, Debug, Default)]
pub struct BoolEncoder {
    out: Vec<u8>,
    range: u32,
    bottom: u32,
    bit_count: i32,
}

impl BoolEncoder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            out: Vec::new(),
            range: 255,
            bottom: 0,
            bit_count: 24,
        }
    }

    fn carry(&mut self) {
        for byte in self.out.iter_mut().rev() {
            if *byte == 255 {
                *byte = 0;
            } else {
                *byte += 1;
                return;
            }
        }
    }

    /// One bool whose probability of being zero is `prob / 256`.
    #[inline]
    pub fn write(&mut self, prob: u8, bit: bool) {
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
                self.carry();
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

    #[inline]
    pub fn bit(&mut self, bit: bool) {
        self.write(128, bit);
    }

    /// An unsigned literal, most significant bit first.
    pub fn literal(&mut self, bits: u32, value: u32) {
        for shift in (0..bits).rev() {
            self.bit((value >> shift) & 1 != 0);
        }
    }

    /// A magnitude of `bits` bits followed by a sign bit.
    pub fn signed(&mut self, bits: u32, value: i32) {
        self.literal(bits, value.unsigned_abs());
        self.bit(value < 0);
    }

    /// A flag, then the signed value when it is not zero.
    pub fn maybe_signed(&mut self, bits: u32, value: i32) {
        self.bit(value != 0);
        if value != 0 {
            self.signed(bits, value);
        }
    }

    /// Writes `symbol` through a tree whose leaves are non-positive values.
    pub fn tree(&mut self, tree: &[i8], probs: &[u8], symbol: u8) {
        self.tree_from(tree, probs, symbol, 0);
    }

    /// Writes `symbol` through a tree from node index `start`.
    pub fn tree_from(&mut self, tree: &[i8], probs: &[u8], symbol: u8, start: usize) {
        let mut path = [(0usize, false); 16];
        let mut depth = 0;
        if find_path(tree, start, symbol, &mut path, &mut depth) {
            for &(node, bit) in path.iter().take(depth) {
                let prob = probs.get(node >> 1).copied().unwrap_or(128);
                self.write(prob, bit);
            }
        }
    }

    /// Bytes written so far, not counting the bits still in the coder.
    #[must_use]
    pub fn len(&self) -> usize {
        self.out.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.out.is_empty()
    }

    /// Flushes the coder and returns the partition.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        let mut c = self.bit_count;
        let mut v = self.bottom;
        if c < 32 && v & (1u32 << (32 - c)) != 0 {
            self.carry();
        }
        v <<= c & 7;
        c >>= 3;
        while c > 0 {
            v <<= 8;
            c -= 1;
        }
        for _ in 0..4 {
            self.out.push((v >> 24) as u8);
            v <<= 8;
        }
        self.out
    }
}

/// Depth-first search for the bits that reach `symbol`.
fn find_path(
    tree: &[i8],
    node: usize,
    symbol: u8,
    path: &mut [(usize, bool); 16],
    depth: &mut usize,
) -> bool {
    if *depth >= path.len() {
        return false;
    }
    for bit in [false, true] {
        let Some(&next) = tree.get(node + usize::from(bit)) else {
            return false;
        };
        path[*depth] = (node, bit);
        *depth += 1;
        if next <= 0 {
            if next.unsigned_abs() == symbol {
                return true;
            }
        } else if find_path(tree, usize::from(next.unsigned_abs()), symbol, path, depth) {
            return true;
        }
        *depth -= 1;
    }
    false
}

/// Cost of one bool in 1/256 bits, `-log2(p) * 256`, by 8-bit probability
/// of the coded value. Index 0 is never used.
pub fn bit_cost(prob_of_value: u8) -> u32 {
    COST[usize::from(prob_of_value)]
}

/// Cost of coding `bit` at probability `prob` of zero.
#[inline]
pub fn cost(prob: u8, bit: bool) -> u32 {
    if bit {
        // 256 - prob is in 1..=255 because prob is never 0.
        bit_cost(255 - prob.max(1) + 1)
    } else {
        bit_cost(prob)
    }
}

const COST: [u32; 256] = cost_table();

const fn cost_table() -> [u32; 256] {
    // -log2(p / 256) * 256 by a fixed-point series: log2 by repeated
    // squaring with 16 fractional bits.
    let mut table = [0u32; 256];
    let mut p = 1;
    while p < 256 {
        // log2(256 / p) = 8 - log2(p).
        let log = log2_fixed(p as u64);
        let bits = (8u64 << 16).saturating_sub(log);
        table[p] = ((bits + 128) >> 8) as u32;
        p += 1;
    }
    table[0] = table[1];
    table
}

/// log2(x) with 16 fractional bits, for 1 <= x < 2^32.
const fn log2_fixed(x: u64) -> u64 {
    let mut int = 0u64;
    let mut v = x;
    while v >= 2 {
        v >>= 1;
        int += 1;
    }
    // Normalize x to [1, 2) with 32 fractional bits.
    let mut m = ((x as u128) << 32) >> int;
    let mut frac = 0u64;
    let mut bit = 1u64 << 15;
    while bit > 0 {
        m = (m * m) >> 32;
        if m >= (2u128 << 32) {
            m >>= 1;
            frac |= bit;
        }
        bit >>= 1;
    }
    (int << 16) | frac
}

#[cfg(test)]
mod tests {
    use super::{BoolDecoder, BoolEncoder, bit_cost, cost};

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (self.0 >> 33) as u32
        }
    }

    #[test]
    fn random_bools_round_trip_at_every_probability() {
        let mut rng = Lcg(7);
        let mut symbols = Vec::new();
        let mut encoder = BoolEncoder::new();
        for _ in 0..200_000 {
            let prob = (rng.next() % 255 + 1) as u8;
            let bit = rng.next() % 256 >= u32::from(prob);
            encoder.write(prob, bit);
            symbols.push((prob, bit));
        }
        let data = encoder.finish();
        let mut decoder = BoolDecoder::new(&data);
        for (index, &(prob, bit)) in symbols.iter().enumerate() {
            assert_eq!(decoder.read(prob), bit, "symbol {index}");
        }
    }

    #[test]
    fn literals_trees_and_signed_values_round_trip() {
        let tree = crate::tables::B_MODE_TREE;
        let probs = crate::tables::B_MODE_PROBS;
        let mut encoder = BoolEncoder::new();
        encoder.literal(7, 93);
        encoder.signed(4, -11);
        encoder.maybe_signed(6, 0);
        encoder.maybe_signed(6, 40);
        for mode in 0..10u8 {
            encoder.tree(&tree, &probs, mode);
        }
        let data = encoder.finish();
        let mut decoder = BoolDecoder::new(&data);
        assert_eq!(decoder.literal(7), 93);
        assert_eq!(decoder.signed(4), -11);
        assert_eq!(decoder.maybe_signed(6), 0);
        assert_eq!(decoder.maybe_signed(6), 40);
        for mode in 0..10u8 {
            assert_eq!(decoder.tree(&tree, &probs), mode);
        }
    }

    #[test]
    fn a_long_run_of_certain_ones_carries_into_written_bytes() {
        let mut encoder = BoolEncoder::new();
        for _ in 0..10_000 {
            encoder.write(1, true);
        }
        encoder.write(255, false);
        let data = encoder.finish();
        let mut decoder = BoolDecoder::new(&data);
        for index in 0..10_000 {
            assert!(decoder.read(1), "bit {index}");
        }
        assert!(!decoder.read(255));
    }

    #[test]
    fn an_empty_partition_reads_zero_bits_without_panicking() {
        let mut decoder = BoolDecoder::new(&[]);
        for _ in 0..1000 {
            assert!(!decoder.read(128));
        }
        assert!(decoder.overread() > 100);
    }

    #[test]
    fn the_reference_two_byte_decoder_agrees() {
        // RFC 6386 Section 7.3 `read_bool`, bit by bit.
        let mut rng = Lcg(99);
        let data: Vec<u8> = (0..4096).map(|_| rng.next() as u8).collect();
        let probs: Vec<u8> = (0..20_000).map(|_| (rng.next() % 255 + 1) as u8).collect();
        let mut fast = BoolDecoder::new(&data);
        let mut value = (u32::from(data[0]) << 8) | u32::from(data[1]);
        let mut input = 2usize;
        let mut range = 255u32;
        let mut bit_count = 0;
        for &prob in &probs {
            let split = 1 + (((range - 1) * u32::from(prob)) >> 8);
            let big = split << 8;
            let bit = if value >= big {
                range -= split;
                value -= big;
                true
            } else {
                range = split;
                false
            };
            while range < 128 {
                value <<= 1;
                range <<= 1;
                bit_count += 1;
                if bit_count == 8 {
                    bit_count = 0;
                    value |= u32::from(data.get(input).copied().unwrap_or(0));
                    input += 1;
                }
            }
            assert_eq!(fast.read(prob), bit);
        }
    }

    #[test]
    fn costs_are_minus_log2_in_256ths() {
        assert_eq!(bit_cost(128), 256);
        assert_eq!(bit_cost(64), 512);
        assert!(bit_cost(255) < 2);
        assert!(cost(250, true) > cost(250, false));
    }
}
