//! Transform-block coefficient syntax.
//!
//! `read_4x4` implements `coeffs()` for `TX_4X4`, including `all_zero`, EOB,
//! base/BR/Golomb levels, and DC/AC signs. The scan is the default 4×4 scan
//! because lossless forces `DCT_DCT`. Larger transforms are rejected by the
//! caller until their scans are wired.

use crate::coeff_cdf::{
    COEFF_BASE, COEFF_BASE_EOB, COEFF_BR, DC_SIGN, EOB_EXTRA, EOB_PT_16, TXB_SKIP,
};
use crate::error::DecodeError;
use crate::symbol::SymbolDecoder;

const NUM_BASE_LEVELS: i32 = 2;
const COEFF_BASE_RANGE: i32 = 12;
const BR_CDF_SIZE: i32 = 4;
const SIG_COEF_CONTEXTS: i32 = 42;
const SIG_COEF_CONTEXTS_EOB: i32 = 4;

/// `Default_Scan_4x4`.
const SCAN_4X4: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

/// Quantizer-index bucket for coefficient CDF initialization.
pub fn qctx(base_q_idx: u8) -> usize {
    match base_q_idx {
        0..=20 => 0,
        21..=60 => 1,
        61..=120 => 2,
        _ => 3,
    }
}

/// Per-plane level and DC-sign contexts at 4-sample granularity.
pub struct LevelMaps {
    pub above_level: [Vec<u8>; 3],
    pub left_level: [Vec<u8>; 3],
    pub above_dc: [Vec<u8>; 3],
    pub left_dc: [Vec<u8>; 3],
}

impl LevelMaps {
    pub fn new(mi_cols: usize, mi_rows: usize) -> Self {
        Self {
            above_level: [vec![0; mi_cols], vec![0; mi_cols], vec![0; mi_cols]],
            left_level: [vec![0; mi_rows], vec![0; mi_rows], vec![0; mi_rows]],
            above_dc: [vec![0; mi_cols], vec![0; mi_cols], vec![0; mi_cols]],
            left_dc: [vec![0; mi_rows], vec![0; mi_rows], vec![0; mi_rows]],
        }
    }
}

/// Context for `all_zero` / `txb_skip`.
pub fn txb_skip_ctx(
    plane: usize,
    tx_w: usize,
    tx_h: usize,
    block_w: usize,
    block_h: usize,
    x4: usize,
    y4: usize,
    max_x4: usize,
    max_y4: usize,
    above_level: &[u8],
    left_level: &[u8],
    above_dc: &[u8],
    left_dc: &[u8],
) -> Result<usize, DecodeError> {
    let w4 = tx_w / 4;
    let h4 = tx_h / 4;
    if plane == 0 {
        let mut top = 0u8;
        let mut left = 0u8;
        for k in 0..w4 {
            if x4 + k < max_x4 {
                top = top.max(
                    *above_level
                        .get(x4 + k)
                        .ok_or_else(|| DecodeError::malformed("above level context"))?,
                );
            }
        }
        for k in 0..h4 {
            if y4 + k < max_y4 {
                left = left.max(
                    *left_level
                        .get(y4 + k)
                        .ok_or_else(|| DecodeError::malformed("left level context"))?,
                );
            }
        }
        let ctx = if block_w == tx_w && block_h == tx_h {
            0
        } else if top == 0 && left == 0 {
            1
        } else if top == 0 || left == 0 {
            2 + usize::from(top.max(left) > 3)
        } else if top.max(left) <= 3 {
            4
        } else if top.min(left) <= 3 {
            5
        } else {
            6
        };
        Ok(ctx)
    } else {
        let mut above = 0u8;
        let mut left = 0u8;
        for i in 0..w4 {
            if x4 + i < max_x4 {
                let level = *above_level
                    .get(x4 + i)
                    .ok_or_else(|| DecodeError::malformed("above level context"))?;
                let dc = *above_dc
                    .get(x4 + i)
                    .ok_or_else(|| DecodeError::malformed("above dc context"))?;
                above |= level;
                above |= dc;
            }
        }
        for i in 0..h4 {
            if y4 + i < max_y4 {
                let level = *left_level
                    .get(y4 + i)
                    .ok_or_else(|| DecodeError::malformed("left level context"))?;
                let dc = *left_dc
                    .get(y4 + i)
                    .ok_or_else(|| DecodeError::malformed("left dc context"))?;
                left |= level;
                left |= dc;
            }
        }
        let mut ctx = usize::from(above != 0) + usize::from(left != 0) + 7;
        if block_w * block_h > tx_w * tx_h {
            ctx += 3;
        }
        Ok(ctx)
    }
}

/// Read one 4×4 coefficient block. Returns signed levels in scan-raster order.
/// An all-zero block returns sixteen zeros and still updates level contexts.
pub fn read_4x4(
    sym: &mut SymbolDecoder<'_>,
    base_q_idx: u8,
    plane: usize,
    block_w: usize,
    block_h: usize,
    x4: usize,
    y4: usize,
    max_x4: usize,
    max_y4: usize,
    maps: &mut LevelMaps,
) -> Result<[i32; 16], DecodeError> {
    if plane > 2 {
        return Err(DecodeError::malformed("coefficient plane"));
    }
    let q = qctx(base_q_idx);
    let tx_sz_ctx = 0usize;
    let ptype = usize::from(plane > 0);
    let ctx = txb_skip_ctx(
        plane,
        4,
        4,
        block_w,
        block_h,
        x4,
        y4,
        max_x4,
        max_y4,
        &maps.above_level[plane],
        &maps.left_level[plane],
        &maps.above_dc[plane],
        &maps.left_dc[plane],
    )?;
    if ctx >= 13 {
        return Err(DecodeError::malformed("txb_skip context"));
    }
    let mut skip_cdf = TXB_SKIP[q][tx_sz_ctx][ctx];
    let all_zero = sym.read_symbol(&mut skip_cdf)?;
    let mut quant = [0i32; 16];
    let mut cul = 0i32;
    let mut dc_cat = 0u8;
    if all_zero == 0 {
        let eob = read_eob_16(sym, q, ptype, tx_sz_ctx)?;
        if eob == 0 || eob > 16 {
            return Err(DecodeError::malformed("4x4 eob"));
        }
        for c in (0..eob).rev() {
            let pos = SCAN_4X4[c];
            let mut level = if c + 1 == eob {
                let bctx = coeff_base_ctx(&quant, pos, c, true)?;
                let eob_ctx = bctx as i32 - SIG_COEF_CONTEXTS + SIG_COEF_CONTEXTS_EOB;
                if !(0..4).contains(&eob_ctx) {
                    return Err(DecodeError::malformed("coeff_base_eob context"));
                }
                let mut cdf = COEFF_BASE_EOB[q][tx_sz_ctx][ptype][eob_ctx as usize];
                sym.read_symbol(&mut cdf)? as i32 + 1
            } else {
                let bctx = coeff_base_ctx(&quant, pos, c, false)?;
                if bctx >= 42 {
                    return Err(DecodeError::malformed("coeff_base context"));
                }
                let mut cdf = COEFF_BASE[q][tx_sz_ctx][ptype][bctx];
                sym.read_symbol(&mut cdf)? as i32
            };
            if level > NUM_BASE_LEVELS {
                let passes = COEFF_BASE_RANGE / (BR_CDF_SIZE - 1);
                for _ in 0..passes {
                    let br_ctx = coeff_br_ctx(&quant, pos)?;
                    let mut cdf = COEFF_BR[q][tx_sz_ctx][ptype][br_ctx];
                    let br = sym.read_symbol(&mut cdf)? as i32;
                    level += br;
                    if br < BR_CDF_SIZE - 1 {
                        break;
                    }
                }
            }
            quant[pos] = level;
        }
        for (c, &pos) in SCAN_4X4[..eob].iter().enumerate() {
            if quant[pos] == 0 {
                continue;
            }
            let sign = if c == 0 {
                let sctx = dc_sign_ctx(plane, x4, y4, max_x4, max_y4, maps)?;
                let mut cdf = DC_SIGN[q][ptype][sctx];
                sym.read_symbol(&mut cdf)?
            } else {
                sym.read_bool()?
            };
            if quant[pos] > NUM_BASE_LEVELS + COEFF_BASE_RANGE {
                quant[pos] = read_golomb(sym)?;
            }
            if pos == 0 && quant[pos] > 0 {
                dc_cat = if sign == 0 { 2 } else { 1 };
            }
            quant[pos] &= 0xF_FFFF;
            cul += quant[pos];
            if sign != 0 {
                quant[pos] = -quant[pos];
            }
        }
        cul = cul.min(63);
    }
    store_context(maps, plane, x4, y4, cul as u8, dc_cat)?;
    Ok(quant)
}

fn read_eob_16(
    sym: &mut SymbolDecoder<'_>,
    q: usize,
    ptype: usize,
    tx_sz_ctx: usize,
) -> Result<usize, DecodeError> {
    // 4×4 is TX_CLASS_2D under the lossless DCT_DCT rule, so the EOB context is 0.
    let mut cdf = EOB_PT_16[q][ptype][0];
    let eob_pt = sym.read_symbol(&mut cdf)? as i32 + 1;
    let mut eob = if eob_pt < 2 {
        eob_pt
    } else {
        (1 << (eob_pt - 2)) + 1
    };
    let eob_shift = (eob_pt - 3).max(-1);
    if eob_shift >= 0 {
        let extra_ctx = (eob_pt - 3) as usize;
        if extra_ctx >= 9 {
            return Err(DecodeError::malformed("eob_extra context"));
        }
        let mut extra = EOB_EXTRA[q][tx_sz_ctx][ptype][extra_ctx];
        if sym.read_symbol(&mut extra)? != 0 {
            eob += 1 << eob_shift;
        }
        let n = (eob_pt - 2).max(0);
        for i in 1..n {
            let shift = n - 1 - i;
            if sym.read_bool()? != 0 {
                eob += 1 << shift;
            }
        }
    }
    if eob < 0 {
        return Err(DecodeError::malformed("negative eob"));
    }
    Ok(eob as usize)
}

fn read_golomb(sym: &mut SymbolDecoder<'_>) -> Result<i32, DecodeError> {
    let mut length = 0i32;
    loop {
        length += 1;
        if length > 20 {
            return Err(DecodeError::malformed("golomb length"));
        }
        let bit = sym.read_bool()?;
        if length == 20 && bit == 0 {
            return Err(DecodeError::malformed("golomb length 20 must end"));
        }
        if bit != 0 {
            break;
        }
    }
    let mut x = 1i32;
    let bits = length - 1;
    for _ in 0..bits {
        let bit = sym.read_bool()?;
        x = (x << 1) | bit as i32;
    }
    Ok(x + COEFF_BASE_RANGE + NUM_BASE_LEVELS)
}

fn store_context(
    maps: &mut LevelMaps,
    plane: usize,
    x4: usize,
    y4: usize,
    cul: u8,
    dc_cat: u8,
) -> Result<(), DecodeError> {
    let above_l = maps.above_level[plane]
        .get_mut(x4)
        .ok_or_else(|| DecodeError::malformed("level context store"))?;
    *above_l = cul;
    maps.above_dc[plane][x4] = dc_cat;
    let left_l = maps.left_level[plane]
        .get_mut(y4)
        .ok_or_else(|| DecodeError::malformed("level context store"))?;
    *left_l = cul;
    maps.left_dc[plane][y4] = dc_cat;
    Ok(())
}

fn dc_sign_ctx(
    plane: usize,
    x4: usize,
    y4: usize,
    max_x4: usize,
    max_y4: usize,
    maps: &LevelMaps,
) -> Result<usize, DecodeError> {
    let mut dc_sign = 0i32;
    if x4 < max_x4 {
        dc_sign += sign_delta(
            *maps.above_dc[plane]
                .get(x4)
                .ok_or_else(|| DecodeError::malformed("dc sign context"))?,
        );
    }
    if y4 < max_y4 {
        dc_sign += sign_delta(
            *maps.left_dc[plane]
                .get(y4)
                .ok_or_else(|| DecodeError::malformed("dc sign context"))?,
        );
    }
    Ok(if dc_sign < 0 {
        1
    } else if dc_sign > 0 {
        2
    } else {
        0
    })
}

fn sign_delta(category: u8) -> i32 {
    match category {
        1 => -1,
        2 => 1,
        _ => 0,
    }
}

fn coeff_base_ctx(
    quant: &[i32; 16],
    pos: usize,
    c: usize,
    is_eob: bool,
) -> Result<usize, DecodeError> {
    let bwl = 2usize;
    let width = 4usize;
    let height = 4usize;
    if is_eob {
        if c == 0 {
            return Ok((SIG_COEF_CONTEXTS - 4) as usize);
        }
        if c <= (height << bwl) / 8 {
            return Ok((SIG_COEF_CONTEXTS - 3) as usize);
        }
        if c <= (height << bwl) / 4 {
            return Ok((SIG_COEF_CONTEXTS - 2) as usize);
        }
        return Ok((SIG_COEF_CONTEXTS - 1) as usize);
    }
    let row = pos >> bwl;
    let col = pos - (row << bwl);
    if row >= height || col >= width {
        return Err(DecodeError::malformed("coefficient position"));
    }
    let mut mag = 0i32;
    const OFF: [(i32, i32); 5] = [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0)];
    for (dr, dc) in OFF {
        let rr = row as i32 + dr;
        let cc = col as i32 + dc;
        if rr >= 0 && cc >= 0 && (rr as usize) < height && (cc as usize) < width {
            mag += quant[(rr as usize) * width + cc as usize].abs().min(3);
        }
    }
    let ctx = ((mag + 1) >> 1).min(4);
    if row == 0 && col == 0 {
        return Ok(0);
    }
    const OFFSET: [[i32; 5]; 5] = [
        [0, 1, 6, 6, 0],
        [1, 6, 6, 21, 0],
        [6, 6, 21, 21, 0],
        [6, 21, 21, 21, 0],
        [0, 0, 0, 0, 0],
    ];
    Ok((ctx + OFFSET[row.min(4)][col.min(4)]) as usize)
}

fn coeff_br_ctx(quant: &[i32; 16], pos: usize) -> Result<usize, DecodeError> {
    let row = pos >> 2;
    let col = pos & 3;
    if row >= 4 {
        return Err(DecodeError::malformed("coefficient position"));
    }
    let mut mag = 0i32;
    const OFF: [(i32, i32); 3] = [(0, 1), (1, 0), (1, 1)];
    for (dr, dc) in OFF {
        let rr = row as i32 + dr;
        let cc = col as i32 + dc;
        if rr >= 0 && cc >= 0 && rr < 4 && cc < 4 {
            mag += quant[(rr as usize) * 4 + cc as usize].min(15);
        }
    }
    let mag = ((mag + 1) >> 1).min(6) as usize;
    if pos == 0 {
        Ok(mag)
    } else if row < 2 && col < 2 {
        Ok(mag + 7)
    } else {
        Ok(mag + 14)
    }
}
