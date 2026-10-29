//! Coefficient token encoder.
//!
//! The bits are the inverse of `decode_coefs`: an end-of-block bit, a zero
//! run, then a magnitude. Category bits are written most-significant-bit first.

use crate::bits::BoolWriter;
use crate::prob::FrameContext;
use crate::tables::{
    BAND_4X4, BAND_8X8_PLUS, CAT1_PROB, CAT2_PROB, CAT3_PROB, CAT4_PROB, CAT5_PROB, CAT6_PROB,
    PARETO8,
};

const CAT6_MAX: u32 = 67 + (1 << 14) - 1;

pub struct TxTokens {
    pub tx: u8,
    pub plane: usize,
    pub inter: bool,
    pub ctx: usize,
    pub tokens: Vec<i32>,
}

pub fn write_tokens(
    w: &mut BoolWriter,
    fc: &FrameContext,
    tx: &TxTokens,
    scan: &[u16],
    neighbors: &[u16],
) {
    let mut coder = Coder {
        w,
        fc,
        tx: usize::from(tx.tx),
        plane: usize::from(tx.plane > 0),
        inter: usize::from(tx.inter),
        scan,
        neighbors,
        tokens: &tx.tokens,
        cache: vec![0u8; scan.len()],
    };
    coder.write(tx.ctx);
}

struct Coder<'a> {
    w: &'a mut BoolWriter,
    fc: &'a FrameContext,
    tx: usize,
    plane: usize,
    inter: usize,
    scan: &'a [u16],
    neighbors: &'a [u16],
    tokens: &'a [i32],
    cache: Vec<u8>,
}

impl Coder<'_> {
    fn write(&mut self, mut ctx: usize) {
        let mut c = 0usize;
        while c < self.scan.len() {
            if self.tail_zero(c) {
                self.bit(c, ctx, 0, 0);
                break;
            }
            self.bit(c, ctx, 0, 1);
            if self.token(c) == 0 {
                self.bit(c, ctx, 1, 0);
                c = self.zeros(c, &mut ctx);
                if c >= self.scan.len() {
                    return;
                }
            } else {
                self.bit(c, ctx, 1, 1);
            }
            let energy = self.magnitude(c, ctx);
            self.remember(c, energy);
            c += 1;
            if c < self.scan.len() {
                ctx = self.neighbor(c);
            }
        }
    }

    fn zeros(&mut self, mut c: usize, ctx: &mut usize) -> usize {
        loop {
            self.remember(c, 0);
            c += 1;
            if c >= self.scan.len() {
                return c;
            }
            *ctx = self.neighbor(c);
            if self.token(c) == 0 {
                self.bit(c, *ctx, 1, 0);
            } else {
                self.bit(c, *ctx, 1, 1);
                return c;
            }
        }
    }

    fn magnitude(&mut self, c: usize, ctx: usize) -> u8 {
        let mag = self.token(c).unsigned_abs().min(CAT6_MAX);
        let negative = self.token(c) < 0;
        let pivot = self.prob(c, ctx, 2);
        if mag <= 1 {
            self.bit_prob(pivot, 0);
            self.sign(negative);
            return 1;
        }
        self.bit_prob(pivot, 1);
        let extra = PARETO8[usize::from(pivot.saturating_sub(1))];
        self.category(mag, &extra);
        self.sign(negative);
        energy(mag)
    }

    fn category(&mut self, mag: u32, extra: &[u8; 8]) {
        if mag <= 4 {
            self.bit_prob(extra[0], 0);
            if mag == 2 {
                self.bit_prob(extra[1], 0);
            } else {
                self.bit_prob(extra[1], 1);
                self.bit_prob(extra[2], u32::from(mag == 4));
            }
            return;
        }
        self.bit_prob(extra[0], 1);
        let (base, bits, probs, node_a, node_b) = match mag {
            5..=6 => (5, 1u32, &CAT1_PROB[..], 0, 0),
            7..=10 => (7, 2, &CAT2_PROB[..], 0, 1),
            11..=18 => (11, 3, &CAT3_PROB[..], 1, 0),
            19..=34 => (19, 4, &CAT4_PROB[..], 1, 1),
            35..=66 => (35, 5, &CAT5_PROB[..], 2, 0),
            _ => (67, 14, &CAT6_PROB[..], 2, 1),
        };
        self.pareto_branch(extra, node_a, node_b);
        let extra_bits = mag - base;
        for shift in (0..bits).rev() {
            let bit = (extra_bits >> shift) & 1;
            let prob = probs[usize::try_from(bits - 1 - shift).unwrap_or(0)];
            self.bit_prob(prob, bit);
        }
    }

    fn pareto_branch(&mut self, extra: &[u8; 8], kind: u8, second: u8) {
        match kind {
            0 => {
                self.bit_prob(extra[3], 0);
                self.bit_prob(extra[4], u32::from(second));
            }
            1 => {
                self.bit_prob(extra[3], 1);
                self.bit_prob(extra[5], 0);
                self.bit_prob(extra[6], u32::from(second));
            }
            _ => {
                self.bit_prob(extra[3], 1);
                self.bit_prob(extra[5], 1);
                self.bit_prob(extra[7], u32::from(second));
            }
        }
    }

    fn sign(&mut self, negative: bool) {
        self.w.write(u32::from(negative), 128);
    }

    fn tail_zero(&self, start: usize) -> bool {
        (start..self.scan.len()).all(|c| self.token(c) == 0)
    }

    fn token(&self, c: usize) -> i32 {
        self.tokens
            .get(usize::from(self.scan[c]))
            .copied()
            .unwrap_or(0)
    }

    fn remember(&mut self, c: usize, energy: u8) {
        let at = usize::from(self.scan[c]);
        if let Some(slot) = self.cache.get_mut(at) {
            *slot = energy;
        }
    }

    fn neighbor(&self, c: usize) -> usize {
        let left = self.cached(self.neighbors.get(c * 2).copied().unwrap_or(0));
        let above = self.cached(self.neighbors.get(c * 2 + 1).copied().unwrap_or(0));
        usize::from(1 + left + above) >> 1
    }

    fn cached(&self, index: u16) -> u8 {
        self.cache.get(usize::from(index)).copied().unwrap_or(0)
    }

    fn bit(&mut self, c: usize, ctx: usize, node: usize, value: u32) {
        let prob = self.prob(c, ctx, node);
        self.w.write(value, u32::from(prob));
    }

    fn bit_prob(&mut self, prob: u8, value: u32) {
        self.w.write(value, u32::from(prob));
    }

    fn prob(&self, c: usize, ctx: usize, node: usize) -> u8 {
        let band = self.band(c);
        self.fc.coef[self.tx][self.plane][self.inter][band][ctx.min(5)][node]
    }

    fn band(&self, c: usize) -> usize {
        let table: &[u8] = if self.tx == 0 {
            &BAND_4X4
        } else {
            &BAND_8X8_PLUS
        };
        usize::from(table.get(c).copied().unwrap_or(5))
    }
}

fn energy(mag: u32) -> u8 {
    match mag {
        0 => 0,
        1 => 1,
        2 => 2,
        3 | 4 => 3,
        5..=10 => 4,
        _ => 5,
    }
}

#[cfg(test)]
mod tests {
    use super::TxTokens;
    use super::write_tokens;
    use crate::bits::{BoolDecoder, BoolWriter};
    use crate::coef::decode_coefs;
    use crate::prob::{Counts, FrameContext};
    use crate::tables::{
        NEIGHBORS_DEFAULT_4X4, NEIGHBORS_DEFAULT_8X8, SCAN_DEFAULT_4X4, SCAN_DEFAULT_8X8,
    };

    #[test]
    fn mixed_tokens_match_the_decoder() {
        let mut tokens = vec![0i32; 16];
        tokens[0] = 5;
        tokens[1] = -2;
        tokens[4] = 1;
        tokens[15] = 40;
        round_trip(0, &tokens, &SCAN_DEFAULT_4X4, &NEIGHBORS_DEFAULT_4X4, false);
        let mut larger = vec![0i32; 64];
        larger[0] = -7;
        larger[8] = 3;
        larger[63] = 1;
        round_trip(1, &larger, &SCAN_DEFAULT_8X8, &NEIGHBORS_DEFAULT_8X8, true);
    }

    fn round_trip(tx: u8, tokens: &[i32], scan: &[u16], neighbors: &[u16], inter: bool) {
        let coded = TxTokens {
            tx,
            plane: 0,
            inter,
            ctx: 1,
            tokens: tokens.to_vec(),
        };
        let fc = FrameContext::defaults();
        let mut writer = BoolWriter::new();
        write_tokens(&mut writer, &fc, &coded, scan, neighbors);
        let bytes = writer.finish().expect("tokens");
        let mut reader = BoolDecoder::new(&bytes).expect("decoder");
        let mut counts = Counts::new();
        let mut coef = vec![0i32; tokens.len()];
        decode_coefs(
            &mut reader,
            &fc,
            &mut counts,
            false,
            usize::from(tx),
            0,
            inter,
            [1, 1],
            1,
            scan,
            neighbors,
            &mut coef,
        )
        .expect("coefs");
        reader.finish().expect("padding");
        assert_eq!(coef, tokens);
    }
}
