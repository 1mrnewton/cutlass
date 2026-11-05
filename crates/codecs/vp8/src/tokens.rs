//! DCT token partitions (RFC 6386 Section 13).

use crate::bool_coder::BoolDecoder;
use crate::header::CoeffProbs;
use crate::tables::{CAT_BASE, CAT_BITS, CAT_PROBS, COEFF_BANDS, ZIGZAG};

/// Block types for coefficient probabilities.
pub const TYPE_Y_AFTER_Y2: usize = 0;
pub const TYPE_Y2: usize = 1;
pub const TYPE_CHROMA: usize = 2;
pub const TYPE_Y_WITH_DC: usize = 3;

/// Dequantization factors for one segment, as `[dc, ac]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dequant {
    pub y1: [i32; 2],
    pub y2: [i32; 2],
    pub uv: [i32; 2],
}

impl Dequant {
    /// Factors for a quantizer index and the frame's deltas (Section 14.1).
    #[must_use]
    pub fn new(q: i32, deltas: &crate::header::QuantIndices) -> Self {
        use crate::tables::{AC_Q, DC_Q};
        let dc = |delta: i8| i32::from(DC_Q[(q + i32::from(delta)).clamp(0, 127) as usize]);
        let ac = |delta: i8| i32::from(AC_Q[(q + i32::from(delta)).clamp(0, 127) as usize]);
        Self {
            y1: [dc(deltas.y_dc_delta), ac(0)],
            y2: [
                dc(deltas.y2_dc_delta) * 2,
                (ac(deltas.y2_ac_delta) * 155 / 100).max(8),
            ],
            uv: [dc(deltas.uv_dc_delta).min(132), ac(deltas.uv_ac_delta)],
        }
    }
}

/// "Has data" flags carried across macroblocks: Y columns or rows 0..4,
/// U 4..6, V 6..8, Y2 at 8.
pub type Context = [u8; 9];

/// Coefficients of one macroblock: 16 Y, 4 U, 4 V, then Y2.
#[derive(Clone, Debug)]
pub struct MbCoeffs {
    pub blocks: [[i16; 16]; 25],
    /// Per block: 0 when empty, 1 when only the first coefficient may be
    /// set, 2 otherwise.
    pub shape: [u8; 25],
    pub any: bool,
}

impl Default for MbCoeffs {
    fn default() -> Self {
        Self {
            blocks: [[0; 16]; 25],
            shape: [0; 25],
            any: false,
        }
    }
}

impl MbCoeffs {
    pub fn clear(&mut self) {
        for (block, shape) in self.blocks.iter_mut().zip(self.shape.iter_mut()) {
            if *shape != 0 {
                *block = [0; 16];
                *shape = 0;
            }
        }
        self.any = false;
    }
}

/// Reads one block's tokens. Returns the index past the last token read,
/// equal to `first` when the block is empty.
#[inline]
fn read_block(
    bits: &mut BoolDecoder<'_>,
    probs: &[[[u8; 11]; 3]; 8],
    ctx: usize,
    first: usize,
    dq: [i32; 2],
    out: &mut [i16; 16],
) -> usize {
    let mut c = first;
    let mut p = &probs[COEFF_BANDS[c]][ctx];
    if !bits.read(p[0]) {
        return c;
    }
    loop {
        if !bits.read(p[1]) {
            // A zero, and no end-of-block check before the next token.
            c += 1;
            if c == 16 {
                return c;
            }
            p = &probs[COEFF_BANDS[c]][0];
            continue;
        }
        let (value, next) = if !bits.read(p[2]) {
            (1, 1)
        } else if !bits.read(p[3]) {
            let v = if !bits.read(p[4]) {
                2
            } else if !bits.read(p[5]) {
                3
            } else {
                4
            };
            (v, 2)
        } else {
            let cat = if !bits.read(p[6]) {
                usize::from(bits.read(p[7]))
            } else if !bits.read(p[8]) {
                2 + usize::from(bits.read(p[9]))
            } else {
                4 + usize::from(bits.read(p[10]))
            };
            let mut extra = 0;
            for &prob in &CAT_PROBS[cat][..CAT_BITS[cat]] {
                extra = (extra << 1) | i32::from(bits.read(prob));
            }
            (CAT_BASE[cat] + extra, 2)
        };
        let signed = if bits.bit() { -value } else { value };
        out[ZIGZAG[c]] = (signed * dq[usize::from(c > 0)]) as i16;
        c += 1;
        if c == 16 {
            return c;
        }
        p = &probs[COEFF_BANDS[c]][next];
        if !bits.read(p[0]) {
            return c;
        }
    }
}

/// Reads a macroblock's tokens and updates the contexts.
pub fn read_mb(
    bits: &mut BoolDecoder<'_>,
    probs: &CoeffProbs,
    left: &mut Context,
    above: &mut Context,
    has_y2: bool,
    dq: &Dequant,
    out: &mut MbCoeffs,
) {
    let mut any = false;
    let (y_type, first) = if has_y2 {
        let ctx = usize::from(left[8] + above[8]);
        let end = read_block(bits, &probs[TYPE_Y2], ctx, 0, dq.y2, &mut out.blocks[24]);
        let has = u8::from(end > 0);
        left[8] = has;
        above[8] = has;
        out.shape[24] = if end > 1 { 2 } else { has };
        any |= has != 0;
        (TYPE_Y_AFTER_Y2, 1)
    } else {
        (TYPE_Y_WITH_DC, 0)
    };
    for b in 0..16 {
        let (l, a) = (b >> 2, b & 3);
        let ctx = usize::from(left[l] + above[a]);
        let end = read_block(bits, &probs[y_type], ctx, first, dq.y1, &mut out.blocks[b]);
        let has = u8::from(end > first);
        left[l] = has;
        above[a] = has;
        out.shape[b] = if end > 1 { 2 } else { has };
        any |= has != 0;
    }
    for b in 16..24 {
        let i = b - 16;
        let (l, a) = (
            4 + (i >> 2) * 2 + ((i & 3) >> 1),
            4 + (i >> 2) * 2 + (i & 1),
        );
        let ctx = usize::from(left[l] + above[a]);
        let end = read_block(bits, &probs[TYPE_CHROMA], ctx, 0, dq.uv, &mut out.blocks[b]);
        let has = u8::from(end > 0);
        left[l] = has;
        above[a] = has;
        out.shape[b] = if end > 1 { 2 } else { has };
        any |= has != 0;
    }
    out.any = any;
}

/// Clears the contexts of a macroblock without tokens. Y2's context stays
/// when the macroblock has no Y2 block.
pub fn skip_mb(left: &mut Context, above: &mut Context, has_y2: bool) {
    left[..8].fill(0);
    above[..8].fill(0);
    if has_y2 {
        left[8] = 0;
        above[8] = 0;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn chroma_context_slots_follow_the_reference_tables() {
        // Section 13.3's left and above context slots for blocks 16..24.
        let left = [4, 4, 5, 5, 6, 6, 7, 7];
        let above = [4, 5, 4, 5, 6, 7, 6, 7];
        for i in 0..8 {
            let l = 4 + (i >> 2) * 2 + ((i & 3) >> 1);
            let a = 4 + (i >> 2) * 2 + (i & 1);
            assert_eq!((l, a), (left[i], above[i]), "block {}", 16 + i);
        }
    }
}
