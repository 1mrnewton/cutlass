//! Coefficient token decode.
//!
//! The walk matches libvpx `decode_coefs`: an end-of-block bit, a run of
//! zeros, then a value. Category tokens are counted as a two for adaptation.

#![allow(dead_code)]

use cutlass_decode::CodecError;

use crate::bits::BoolDecoder;
use crate::prob::{Counts, FrameContext};
use crate::tables::{
    BAND_4X4, BAND_8X8_PLUS, CAT1_PROB, CAT2_PROB, CAT3_PROB, CAT4_PROB, CAT5_PROB, CAT6_PROB,
    PARETO8,
};

const EOB_TOKEN: usize = 3;
const ZERO_TOKEN: usize = 0;
const ONE_TOKEN: usize = 1;
const TWO_TOKEN: usize = 2;

struct CoefReader<'a, 'b> {
    reader: &'a mut BoolDecoder<'b>,
    fc: &'a FrameContext,
    counts: &'a mut Counts,
    collect: bool,
    tx: usize,
    plane: usize,
    reference: usize,
    scan: &'a [u16],
    neighbors: &'a [u16],
    coef: &'a mut [i32],
    bands: &'a [u8],
    cache: Vec<u8>,
    dq_dc: i32,
    dq_ac: i32,
    shift: u32,
}

#[allow(clippy::too_many_arguments)]
pub fn decode_coefs(
    reader: &mut BoolDecoder<'_>,
    fc: &FrameContext,
    counts: &mut Counts,
    collect: bool,
    tx: usize,
    plane: usize,
    is_inter: bool,
    dq: [i16; 2],
    ctx: usize,
    scan: &[u16],
    neighbors: &[u16],
    coef: &mut [i32],
) -> Result<usize, CodecError> {
    let bands: &[u8] = if tx == 0 { &BAND_4X4 } else { &BAND_8X8_PLUS };
    let mut walk = CoefReader {
        reader,
        fc,
        counts,
        collect,
        tx,
        plane,
        reference: usize::from(is_inter),
        scan,
        neighbors,
        coef,
        bands,
        cache: vec![0u8; scan.len()],
        dq_dc: i32::from(dq[0]),
        dq_ac: i32::from(dq[1]),
        shift: u32::from(tx == 3),
    };
    walk.walk(ctx)
}

impl CoefReader<'_, '_> {
    fn walk(&mut self, mut ctx: usize) -> Result<usize, CodecError> {
        let n = self.scan.len();
        let mut c = 0usize;
        let mut dqv = self.dq_dc;
        while c < n {
            let band = self.band(c);
            self.branch(band, ctx);
            if self.bit(self.prob(band, ctx, 0))? == 0 {
                self.tally(band, ctx, EOB_TOKEN);
                break;
            }
            if self.bit(self.prob(band, ctx, 1))? == 0 {
                (c, ctx) = self.zeros(c, ctx, &mut dqv)?;
                if c >= n {
                    return Ok(c);
                }
            }
            let band = self.band(c);
            let (value, energy) = self.magnitude(band, ctx, dqv)?;
            self.store(c, value, energy)?;
            c += 1;
            if c < n {
                ctx = self.neighbor_ctx(c);
            }
            dqv = self.dq_ac;
        }
        Ok(c)
    }

    /// The zero bit at `c` is already clear. Count zeros until a value.
    fn zeros(
        &mut self,
        mut c: usize,
        mut ctx: usize,
        dqv: &mut i32,
    ) -> Result<(usize, usize), CodecError> {
        loop {
            let band = self.band(c);
            self.tally(band, ctx, ZERO_TOKEN);
            *dqv = self.dq_ac;
            self.store_energy(c, 0);
            c += 1;
            if c >= self.scan.len() {
                return Ok((c, ctx));
            }
            ctx = self.neighbor_ctx(c);
            let band = self.band(c);
            if self.bit(self.prob(band, ctx, 1))? != 0 {
                return Ok((c, ctx));
            }
        }
    }

    fn magnitude(&mut self, band: usize, ctx: usize, dqv: i32) -> Result<(i32, u8), CodecError> {
        let pivot = self.prob(band, ctx, 2);
        if self.bit(pivot)? == 0 {
            self.tally(band, ctx, ONE_TOKEN);
            let mag = dqv >> self.shift;
            return Ok((self.signed(mag)?, 1));
        }
        self.tally(band, ctx, TWO_TOKEN);
        let extra = PARETO8[usize::from(pivot.saturating_sub(1))];
        if self.bit(extra[0])? == 0 {
            return self.small(extra, dqv);
        }
        self.category(extra, dqv)
    }

    fn small(&mut self, extra: [u8; 8], dqv: i32) -> Result<(i32, u8), CodecError> {
        if self.bit(extra[1])? == 0 {
            let mag = (2 * dqv) >> self.shift;
            return Ok((self.signed(mag)?, 2));
        }
        let ones = i32::try_from(self.bit(extra[2])?).unwrap_or(0);
        let mag = ((3 + ones) * dqv) >> self.shift;
        Ok((self.signed(mag)?, 3))
    }

    fn category(&mut self, extra: [u8; 8], dqv: i32) -> Result<(i32, u8), CodecError> {
        let (base, bits, probs, energy) = if self.bit(extra[3])? == 0 {
            if self.bit(extra[4])? == 0 {
                (5, 1u32, &CAT1_PROB[..], 4)
            } else {
                (7, 2, &CAT2_PROB[..], 4)
            }
        } else if self.bit(extra[5])? == 0 {
            if self.bit(extra[6])? == 0 {
                (11, 3, &CAT3_PROB[..], 5)
            } else {
                (19, 4, &CAT4_PROB[..], 5)
            }
        } else if self.bit(extra[7])? == 0 {
            (35, 5, &CAT5_PROB[..], 5)
        } else {
            (67, 14, &CAT6_PROB[..], 5)
        };
        let extra_bits = self.cat_bits(probs, bits)?;
        let mag = ((base + extra_bits) * dqv) >> self.shift;
        Ok((self.signed(mag)?, energy))
    }

    fn cat_bits(&mut self, probs: &[u8], n: u32) -> Result<i32, CodecError> {
        let mut value = 0i32;
        for i in 0..n {
            let bit = i32::try_from(self.bit(probs[usize::try_from(i).unwrap_or(0)])?).unwrap_or(0);
            value = (value << 1) | bit;
        }
        Ok(value)
    }

    fn signed(&mut self, mag: i32) -> Result<i32, CodecError> {
        if self.bit(128)? == 1 {
            Ok(-mag)
        } else {
            Ok(mag)
        }
    }

    fn store(&mut self, c: usize, value: i32, energy: u8) -> Result<(), CodecError> {
        let at = usize::from(self.scan[c]);
        let slot = self.coef.get_mut(at).ok_or(CodecError::Malformed {
            at_bit: 0,
            what: "coef scan",
        })?;
        *slot = value;
        self.store_energy(c, energy);
        Ok(())
    }

    fn store_energy(&mut self, c: usize, energy: u8) {
        let at = usize::from(self.scan[c]);
        if let Some(slot) = self.cache.get_mut(at) {
            *slot = energy;
        }
    }

    fn neighbor_ctx(&self, c: usize) -> usize {
        let left = self.cached(self.neighbors.get(c * 2).copied().unwrap_or(0));
        let above = self.cached(self.neighbors.get(c * 2 + 1).copied().unwrap_or(0));
        usize::from(1 + left + above) >> 1
    }

    fn cached(&self, index: u16) -> u8 {
        self.cache.get(usize::from(index)).copied().unwrap_or(0)
    }

    fn band(&self, c: usize) -> usize {
        usize::from(self.bands[c])
    }

    fn prob(&self, band: usize, ctx: usize, node: usize) -> u8 {
        self.fc.coef[self.tx][self.plane][self.reference][band][ctx.min(5)][node]
    }

    fn bit(&mut self, prob: u8) -> Result<u32, CodecError> {
        self.reader.read(u32::from(prob))
    }

    fn branch(&mut self, band: usize, ctx: usize) {
        if self.collect {
            let slot = &mut self.counts.eob[self.tx][self.plane][self.reference][band][ctx];
            *slot = slot.saturating_add(1);
        }
    }

    fn tally(&mut self, band: usize, ctx: usize, token: usize) {
        if self.collect {
            let slot = &mut self.counts.coef[self.tx][self.plane][self.reference][band][ctx][token];
            *slot = slot.saturating_add(1);
        }
    }
}
