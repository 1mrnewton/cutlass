//! Quantization, token writing, token counts, and token costs.

use crate::bool_coder::{BoolEncoder, cost};
use crate::header::CoeffProbs;
use crate::tables::{CAT_BASE, CAT_BITS, CAT_PROBS, COEFF_BANDS, COEFF_UPDATE_PROBS, ZIGZAG};

/// Largest magnitude a token can carry.
pub const MAX_LEVEL: i32 = 67 + 2047;

/// One quantized block: levels in zigzag order and the index past the
/// last nonzero level (`first` when the block is empty).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QBlock {
    pub levels: [i16; 16],
    pub end: u8,
}

/// Token for a magnitude: 0..4 are literal, 5..10 are categories.
#[inline]
fn token_of(v: i32) -> (usize, usize) {
    match v {
        0..=4 => (v as usize, 0),
        5..=6 => (5, 0),
        7..=10 => (6, 1),
        11..=18 => (7, 2),
        19..=34 => (8, 3),
        35..=66 => (9, 4),
        _ => (10, 5),
    }
}

/// Branches of the coefficient tree for a token, as (node, bit).
/// Node 0 is the end-of-block check, which the caller handles.
#[inline]
fn token_path(token: usize) -> &'static [(usize, bool)] {
    match token {
        0 => &[(1, false)],
        1 => &[(1, true), (2, false)],
        2 => &[(1, true), (2, true), (3, false), (4, false)],
        3 => &[(1, true), (2, true), (3, false), (4, true), (5, false)],
        4 => &[(1, true), (2, true), (3, false), (4, true), (5, true)],
        5 => &[(1, true), (2, true), (3, true), (6, false), (7, false)],
        6 => &[(1, true), (2, true), (3, true), (6, false), (7, true)],
        7 => &[(1, true), (2, true), (3, true), (6, true), (8, false), (9, false)],
        8 => &[(1, true), (2, true), (3, true), (6, true), (8, false), (9, true)],
        9 => &[(1, true), (2, true), (3, true), (6, true), (8, true), (10, false)],
        _ => &[(1, true), (2, true), (3, true), (6, true), (8, true), (10, true)],
    }
}

/// What walking a block's tokens produces.
pub enum Event {
    /// A tree branch with an adaptive probability.
    Node { band: usize, ctx: usize, node: usize, bit: bool },
    /// An extra bit or sign with a fixed probability.
    Fixed { prob: u8, bit: bool },
}

/// Walks a block's tokens in coding order (Section 13).
#[inline]
pub fn walk(q: &QBlock, ctx: usize, first: usize, mut f: impl FnMut(Event)) {
    let end = usize::from(q.end);
    let mut c = first;
    let mut ctx = ctx;
    let mut check_eob = true;
    while c < end {
        let band = COEFF_BANDS[c];
        if check_eob {
            f(Event::Node { band, ctx, node: 0, bit: true });
        }
        let level = i32::from(q.levels[c]);
        let v = level.abs().min(MAX_LEVEL);
        let (token, cat) = token_of(v);
        for &(node, bit) in token_path(token) {
            f(Event::Node { band, ctx, node, bit });
        }
        if token >= 5 {
            let extra = v - CAT_BASE[cat];
            let bits = CAT_BITS[cat];
            for (i, &prob) in CAT_PROBS[cat][..bits].iter().enumerate() {
                f(Event::Fixed {
                    prob,
                    bit: (extra >> (bits - 1 - i)) & 1 == 1,
                });
            }
        }
        if v != 0 {
            f(Event::Fixed { prob: 128, bit: level < 0 });
        }
        ctx = match v {
            0 => 0,
            1 => 1,
            _ => 2,
        };
        check_eob = v != 0;
        c += 1;
    }
    if c < 16 {
        f(Event::Node { band: COEFF_BANDS[c], ctx, node: 0, bit: false });
    }
}

/// Writes a block's tokens.
pub fn write_block(w: &mut BoolEncoder, probs: &[[[u8; 11]; 3]; 8], ctx: usize, first: usize, q: &QBlock) {
    walk(q, ctx, first, |event| match event {
        Event::Node { band, ctx, node, bit } => w.write(probs[band][ctx][node], bit),
        Event::Fixed { prob, bit } => w.write(prob, bit),
    });
}

/// Branch counts by type, band, context, and node: `[zeros, ones]`.
pub type Counts = [[[[[u32; 2]; 11]; 3]; 8]; 4];

pub fn count_block(counts: &mut [[[[u32; 2]; 11]; 3]; 8], ctx: usize, first: usize, q: &QBlock) {
    walk(q, ctx, first, |event| {
        if let Event::Node { band, ctx, node, bit } = event {
            counts[band][ctx][node][usize::from(bit)] += 1;
        }
    });
}

/// Cost in 1/256 bit of `zeros` and `ones` coded at `prob`.
#[must_use]
pub fn branch_cost(prob: u8, counts: [u32; 2]) -> u64 {
    u64::from(counts[0]) * u64::from(cost(prob, false)) + u64::from(counts[1]) * u64::from(cost(prob, true))
}

/// The probability that codes `counts` best.
#[must_use]
pub fn best_prob(counts: [u32; 2]) -> u8 {
    let total = u64::from(counts[0]) + u64::from(counts[1]);
    if total == 0 {
        return 128;
    }
    let p = (u64::from(counts[0]) * 256 + total / 2) / total;
    p.clamp(1, 255) as u8
}

/// Picks the coefficient probabilities for a frame: each node takes the
/// count-optimal probability where that pays for its update.
#[must_use]
pub fn choose_coeff_probs(base: &CoeffProbs, counts: &Counts) -> CoeffProbs {
    let mut out = *base;
    for i in 0..4 {
        for j in 0..8 {
            for k in 0..3 {
                for l in 0..11 {
                    let c = counts[i][j][k][l];
                    let old = base[i][j][k][l];
                    let new = best_prob(c);
                    let update = COEFF_UPDATE_PROBS[i][j][k][l];
                    let keep = branch_cost(old, c) + u64::from(cost(update, false));
                    let change = branch_cost(new, c) + u64::from(cost(update, true)) + 8 * 256;
                    if new != old && change < keep {
                        out[i][j][k][l] = new;
                    }
                }
            }
        }
    }
    out
}

/// Token costs for one block type under the current probabilities.
#[derive(Clone, Debug)]
pub struct TypeCosts {
    /// `[band][ctx][token]` after an end-of-block check that said "more".
    pub token: [[[u16; 11]; 3]; 8],
    /// `[band][ctx][token]` right after a zero, with no end-of-block check.
    pub token_no_eob: [[[u16; 11]; 3]; 8],
    /// `[band][ctx]` cost of ending the block.
    pub eob: [[u16; 3]; 8],
}

/// Costs of every block type, plus the extra-bit cost of each magnitude.
#[derive(Clone, Debug)]
pub struct Costs {
    pub types: [TypeCosts; 4],
    /// Extra bits and sign of a magnitude, in 1/256 bit.
    pub extra: Vec<u16>,
}

impl Costs {
    #[must_use]
    pub fn new(probs: &CoeffProbs) -> Self {
        let types = std::array::from_fn(|t| {
            let mut tc = TypeCosts {
                token: [[[0; 11]; 3]; 8],
                token_no_eob: [[[0; 11]; 3]; 8],
                eob: [[0; 3]; 8],
            };
            for band in 0..8 {
                for ctx in 0..3 {
                    let p = &probs[t][band][ctx];
                    tc.eob[band][ctx] = cost(p[0], false) as u16;
                    for token in 0..11 {
                        let path: u32 = token_path(token).iter().map(|&(n, b)| cost(p[n], b)).sum();
                        tc.token_no_eob[band][ctx][token] = path.min(u32::from(u16::MAX)) as u16;
                        tc.token[band][ctx][token] = (path + cost(p[0], true)).min(u32::from(u16::MAX)) as u16;
                    }
                }
            }
            tc
        });
        let mut extra = vec![0u16; MAX_LEVEL as usize + 1];
        for (v, slot) in extra.iter_mut().enumerate().skip(1) {
            let v = v as i32;
            let (token, cat) = token_of(v);
            let mut bits = 256u32;
            if token >= 5 {
                let e = v - CAT_BASE[cat];
                let n = CAT_BITS[cat];
                for (i, &prob) in CAT_PROBS[cat][..n].iter().enumerate() {
                    bits += cost(prob, (e >> (n - 1 - i)) & 1 == 1);
                }
            }
            *slot = bits.min(u32::from(u16::MAX)) as u16;
        }
        Self { types, extra }
    }

    /// Cost of a block's tokens in 1/256 bit.
    #[must_use]
    pub fn block(&self, ty: usize, ctx: usize, first: usize, q: &QBlock) -> u32 {
        let tc = &self.types[ty];
        let end = usize::from(q.end);
        let mut c = first;
        let mut ctx = ctx;
        let mut after_zero = false;
        let mut total = 0u32;
        while c < end {
            let band = COEFF_BANDS[c];
            let v = i32::from(q.levels[c]).abs().min(MAX_LEVEL);
            let (token, _) = token_of(v);
            total += u32::from(if after_zero {
                tc.token_no_eob[band][ctx][token]
            } else {
                tc.token[band][ctx][token]
            });
            if v != 0 {
                total += u32::from(self.extra[v as usize]);
            }
            ctx = match v {
                0 => 0,
                1 => 1,
                _ => 2,
            };
            after_zero = v == 0;
            c += 1;
        }
        if c < 16 {
            total += u32::from(tc.eob[COEFF_BANDS[c]][ctx]);
        }
        total
    }
}

/// Rounding added before dividing by the step, in 1/128 of the step.
#[derive(Clone, Copy, Debug)]
pub struct Rounding {
    pub dc: i32,
    pub ac: i32,
}

/// Quantizes a block of raster-order coefficients. Returns the levels and
/// writes the dequantized raster coefficients the decoder will see.
#[must_use]
pub fn quantize(coeffs: &[i16; 16], dq: [i32; 2], round: Rounding, first: usize, deq: &mut [i16; 16]) -> QBlock {
    let mut q = QBlock {
        levels: [0; 16],
        end: first as u8,
    };
    *deq = [0; 16];
    for c in first..16 {
        let pos = ZIGZAG[c];
        let x = i32::from(coeffs[pos]);
        let step = dq[usize::from(c > 0)];
        let r = if c == 0 { round.dc } else { round.ac };
        let level = ((x.abs() * 128 + r * step) / (step * 128)).min(MAX_LEVEL);
        if level != 0 {
            let level = if x < 0 { -level } else { level };
            q.levels[c] = level as i16;
            deq[pos] = (level * step) as i16;
            q.end = (c + 1) as u8;
        }
    }
    q
}

#[cfg(test)]
mod tests {
    use super::{Costs, QBlock, Rounding, count_block, quantize, write_block};
    use crate::bool_coder::{BoolDecoder, BoolEncoder};
    use crate::header::Probs;
    use crate::tables::ZIGZAG;
    use crate::tokens::{Context, Dequant, MbCoeffs, read_mb};

    #[test]
    fn written_tokens_read_back_through_the_decoder() {
        let probs = Probs::default().coeff;
        let dq = Dequant::new(30, &crate::header::QuantIndices::default());
        let mut seed = 9u32;
        let mut rand = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            seed >> 16
        };
        for _ in 0..300 {
            // 16 Y blocks with a Y2 block, then 8 chroma.
            let mut blocks = [QBlock::default(); 25];
            for (b, block) in blocks.iter_mut().enumerate() {
                let first = usize::from(b < 16);
                block.end = first as u8;
                for c in first..16 {
                    let r = rand() % 100;
                    let v = match r {
                        0..=60 => 0,
                        61..=80 => 1,
                        81..=95 => (rand() % 20) as i16,
                        _ => (rand() % 2114) as i16,
                    };
                    let v = if rand() % 2 == 0 { -v } else { v };
                    if rand() % 7 == 0 {
                        break;
                    }
                    block.levels[c] = v;
                    if v != 0 {
                        block.end = (c + 1) as u8;
                    }
                }
            }
            let mut w = BoolEncoder::new();
            let mut left: Context = [0; 9];
            let mut above: Context = [0; 9];
            let costs = Costs::new(&probs);
            let mut predicted = 0u32;
            let mut counts = [[[[[0u32; 2]; 11]; 3]; 8]; 4];
            let mut write = |ty: usize, slot_l: usize, slot_a: usize, first: usize, q: &QBlock, w: &mut BoolEncoder| {
                let ctx = usize::from(left[slot_l] + above[slot_a]);
                predicted += costs.block(ty, ctx, first, q);
                count_block(&mut counts[ty], ctx, first, q);
                write_block(w, &probs[ty], ctx, first, q);
                let has = u8::from(usize::from(q.end) > first);
                left[slot_l] = has;
                above[slot_a] = has;
            };
            write(1, 8, 8, 0, &blocks[24], &mut w);
            for b in 0..16 {
                write(0, b >> 2, b & 3, 1, &blocks[b], &mut w);
            }
            for i in 0..8 {
                let l = 4 + (i >> 2) * 2 + ((i & 3) >> 1);
                let a = 4 + (i >> 2) * 2 + (i & 1);
                write(2, l, a, 0, &blocks[16 + i], &mut w);
            }
            let before = w.len();
            let data = w.finish();
            assert!(predicted / 256 <= (data.len() as u32 + 4) * 8 && (before as u32) * 8 <= predicted / 256 + 64);
            let mut bits = BoolDecoder::new(&data);
            let mut out = MbCoeffs::default();
            let (mut l, mut a) = ([0u8; 9], [0u8; 9]);
            read_mb(&mut bits, &probs, &mut l, &mut a, true, &dq, &mut out);
            for (b, block) in blocks.iter().enumerate() {
                let index = if b < 16 { b } else if b < 24 { b } else { 24 };
                let factors = if b == 24 { dq.y2 } else if b >= 16 { dq.uv } else { dq.y1 };
                for c in 0..16 {
                    let want = (i32::from(block.levels[c]) * factors[usize::from(c > 0)]) as i16;
                    let got = out.blocks[index][ZIGZAG[c]];
                    if b < 16 && c == 0 {
                        continue;
                    }
                    assert_eq!(got, want, "block {b} coeff {c}");
                }
            }
        }
    }

    #[test]
    fn quantization_rounds_by_the_given_bias() {
        let mut coeffs = [0i16; 16];
        coeffs[0] = 100;
        coeffs[1] = -29;
        coeffs[4] = 14;
        let mut deq = [0i16; 16];
        let q = quantize(&coeffs, [20, 10], Rounding { dc: 64, ac: 64 }, 0, &mut deq);
        assert_eq!(q.levels[0], 5);
        assert_eq!(q.levels[1], -3);
        assert_eq!(q.levels[2], 1);
        assert_eq!(q.end, 3);
        assert_eq!(deq[0], 100);
        assert_eq!(deq[1], -30);
        assert_eq!(deq[4], 10);
    }
}
