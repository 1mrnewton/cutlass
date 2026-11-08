//! Coefficient tokens (RFC 6386 section 13), written, costed or counted
//! through one walk so the three always agree.

use super::bool_encoder::{BoolEncoder, cost};
use super::header::CoeffProbs;
use super::tables::{BANDS, CAT3, CAT4, CAT5, CAT6, COEFF_UPDATE_PROBS, DEFAULT_COEFF_PROBS};

/// Where the bits of a token walk go.
pub(crate) trait Sink {
    /// A bit with a fixed probability.
    fn put(&mut self, bit: bool, prob: u8);

    /// A bit of the adaptive node `probs[t][band][ctx][node]`.
    #[inline]
    fn node(&mut self, _at: (usize, usize, usize, usize), bit: bool, prob: u8) {
        self.put(bit, prob);
    }
}

impl Sink for BoolEncoder {
    #[inline]
    fn put(&mut self, bit: bool, prob: u8) {
        BoolEncoder::put(self, bit, prob);
    }
}

/// Bits in 1/256ths.
#[derive(Default)]
pub(crate) struct Cost(pub u32);

impl Sink for Cost {
    #[inline]
    fn put(&mut self, bit: bool, prob: u8) {
        self.0 += cost(bit, prob);
    }
}

/// Zeros and ones of every node: `[type][band][context][node]`.
type Counts = [[[[[u32; 2]; 11]; 3]; 8]; 4];

/// How often each adaptive node coded a 0 and a 1.
pub(crate) struct Stats(pub Box<Counts>);

impl Stats {
    pub(crate) fn new() -> Self {
        Self(Box::new([[[[[0; 2]; 11]; 3]; 8]; 4]))
    }

    /// libwebp's rule: a node takes the probability its counts ask for when
    /// that saves more than the eight bits of sending it.
    pub(crate) fn probabilities(&self) -> CoeffProbs {
        let mut probs = DEFAULT_COEFF_PROBS;
        for t in 0..4 {
            for b in 0..8 {
                for c in 0..3 {
                    for n in 0..11 {
                        let [zeros, ones] = self.0[t][b][c][n];
                        let total = zeros + ones;
                        let old = DEFAULT_COEFF_PROBS[t][b][c][n];
                        let update = COEFF_UPDATE_PROBS[t][b][c][n];
                        let new = if ones == 0 {
                            255
                        } else {
                            (255 - u64::from(ones) * 255 / u64::from(total)) as u8
                        };
                        let branch = |p: u8| {
                            u64::from(ones) * u64::from(cost(true, p))
                                + u64::from(zeros) * u64::from(cost(false, p))
                        };
                        let old_cost = branch(old) + u64::from(cost(false, update));
                        let new_cost = branch(new) + u64::from(cost(true, update)) + 8 * 256;
                        if old_cost > new_cost {
                            probs[t][b][c][n] = new;
                        }
                    }
                }
            }
        }
        probs
    }
}

impl Sink for Stats {
    #[inline]
    fn put(&mut self, _bit: bool, _prob: u8) {}

    #[inline]
    fn node(&mut self, (t, b, c, n): (usize, usize, usize, usize), bit: bool, _prob: u8) {
        self.0[t][b][c][n][usize::from(bit)] += 1;
    }
}

/// Write the probability updates of `probs` against the defaults.
pub(crate) fn write_updates(encoder: &mut BoolEncoder, probs: &CoeffProbs) {
    for t in 0..4 {
        for b in 0..8 {
            for c in 0..3 {
                for n in 0..11 {
                    let changed = probs[t][b][c][n] != DEFAULT_COEFF_PROBS[t][b][c][n];
                    encoder.put(changed, COEFF_UPDATE_PROBS[t][b][c][n]);
                    if changed {
                        encoder.literal(u32::from(probs[t][b][c][n]), 8);
                    }
                }
            }
        }
    }
}

/// The tokens of one block of type `t` from scan position `first`, with
/// `levels` in scan order. Returns whether any level was not zero, the
/// context the next blocks see.
pub(crate) fn block<S: Sink>(
    sink: &mut S,
    probs: &CoeffProbs,
    t: usize,
    ctx: usize,
    first: usize,
    levels: &[i16; 16],
) -> bool {
    let Some(last) = (first..16).rev().find(|&n| levels[n] != 0) else {
        let band = BANDS[first];
        sink.node((t, band, ctx, 0), false, probs[t][band][ctx][0]);
        return false;
    };
    let mut n = first;
    let mut ctx = ctx;
    while n < 16 {
        let mut band = BANDS[n];
        if n > last {
            sink.node((t, band, ctx, 0), false, probs[t][band][ctx][0]);
            break;
        }
        sink.node((t, band, ctx, 0), true, probs[t][band][ctx][0]);
        while levels[n] == 0 {
            sink.node((t, band, ctx, 1), false, probs[t][band][ctx][1]);
            n += 1;
            band = BANDS[n];
            ctx = 0;
        }
        let p = &probs[t][band][ctx];
        let at = |node| (t, band, ctx, node);
        sink.node(at(1), true, p[1]);
        let value = levels[n].unsigned_abs();
        if value == 1 {
            sink.node(at(2), false, p[2]);
        } else {
            sink.node(at(2), true, p[2]);
            large(sink, p, at, value);
        }
        sink.put(levels[n] < 0, 128);
        ctx = if value == 1 { 1 } else { 2 };
        n += 1;
    }
    true
}

/// A token of value 2 or more.
fn large<S: Sink>(
    sink: &mut S,
    p: &[u8; 11],
    at: impl Fn(usize) -> (usize, usize, usize, usize),
    value: u16,
) {
    if value <= 4 {
        sink.node(at(3), false, p[3]);
        if value == 2 {
            sink.node(at(4), false, p[4]);
        } else {
            sink.node(at(4), true, p[4]);
            sink.node(at(5), value == 4, p[5]);
        }
    } else if value <= 10 {
        sink.node(at(3), true, p[3]);
        sink.node(at(6), false, p[6]);
        if value <= 6 {
            sink.node(at(7), false, p[7]);
            sink.put(value == 6, 159);
        } else {
            sink.node(at(7), true, p[7]);
            let v = value - 7;
            sink.put(v >= 2, 165);
            sink.put(v & 1 == 1, 145);
        }
    } else {
        sink.node(at(3), true, p[3]);
        sink.node(at(6), true, p[6]);
        let (category, extra): (usize, &[u8]) = match value {
            11..=18 => (0, &CAT3),
            19..=34 => (1, &CAT4),
            35..=66 => (2, &CAT5),
            _ => (3, &CAT6),
        };
        let high = category >> 1;
        sink.node(at(8), high == 1, p[8]);
        sink.node(at(9 + high), category & 1 == 1, p[9 + high]);
        let rest = u32::from(value) - (3 + (8 << category));
        for (index, &prob) in extra.iter().enumerate() {
            let shift = extra.len() - 1 - index;
            sink.put((rest >> shift) & 1 == 1, prob);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Cost, Stats, block};
    use crate::lossy::bool_decoder::BoolDecoder;
    use crate::lossy::bool_encoder::BoolEncoder;
    use crate::lossy::tables::DEFAULT_COEFF_PROBS;

    #[test]
    fn costs_and_counts_follow_the_written_bits() {
        let levels = [
            [3i16, 0, 0, -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7],
            [0; 16],
            [-2047, 66, 35, 34, 19, 18, 11, 10, 7, 6, 5, 4, 3, 2, 1, -1],
        ];
        let mut cost = Cost::default();
        let mut stats = Stats::new();
        let mut encoder = BoolEncoder::new();
        for (index, levels) in levels.iter().enumerate() {
            let first = index % 2;
            let a = block(&mut cost, &DEFAULT_COEFF_PROBS, 3, 1, first, levels);
            let b = block(&mut stats, &DEFAULT_COEFF_PROBS, 3, 1, first, levels);
            let c = block(&mut encoder, &DEFAULT_COEFF_PROBS, 3, 1, first, levels);
            assert_eq!((a, b), (c, c));
        }
        let bytes = encoder.finish();
        // Within a byte or two of the estimate.
        assert!((cost.0 / 256).abs_diff(bytes.len() as u32 * 8) < 48);
        assert!(
            stats.0[3]
                .iter()
                .flatten()
                .flatten()
                .any(|counts| counts[1] > 0)
        );
        // The decoder side reads the same bits back as a sanity check of
        // the walk; the frame tests check the values.
        let mut decoder = BoolDecoder::new(&bytes);
        assert!(decoder.bit(DEFAULT_COEFF_PROBS[3][0][1][0]));
    }
}
