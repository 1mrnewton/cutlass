//! Clause 8.6. SP and SI replace the prediction instead of adding a residual.
//!
//! Chroma DC is scaled with equation 8-431 on both paths. Clause 8.5.12 leaves a
//! chroma DC coefficient unscaled, so the switching copy in 8-441 would skip the
//! scale that a flat prediction needs in order to round-trip.

use crate::recon::{self, idct4, scale4};
use crate::set::ZIGZAG4;

/// QPY or QPC scales a non-switching residual. QSY or QSC quantises the sum.
#[derive(Clone, Copy)]
pub(crate) struct SpQuant {
    pub from: i32,
    pub to: i32,
    pub switching: bool,
}

const AIJ: [i32; 3] = [16, 25, 20];

/// Equation 8-419. Columns are the two-even, two-odd, and mixed classes.
const LEVEL2: [[i32; 3]; 6] = [
    [13107, 5243, 8066],
    [11916, 4660, 7490],
    [10082, 4194, 6554],
    [9362, 3647, 5825],
    [8192, 3355, 5243],
    [7282, 2893, 4559],
];

/// One luma 4×4. `levels` is scan order. The result replaces the prediction.
pub(crate) fn luma_block(
    pred: &[i32; 16],
    levels: &[i32; 16],
    quant: SpQuant,
    weight: &[u8; 16],
) -> [i32; 16] {
    let freq = forward(pred);
    let raster = place(levels);
    let coeffs = quant_freq(&freq, &raster, quant, weight);
    inverse(&coeffs, quant.to, weight, None)
}

/// One chroma component, 8×8, four 4×4 blocks. `ac[n]` is the 15 AC levels in scan
/// order. `dc` is `ChromaDCLevel` in block order.
pub(crate) fn chroma_plane(
    pred: &[i32; 64],
    ac: &[[i32; 16]; 4],
    dc: &[i32; 4],
    quant: SpQuant,
    weight: &[u8; 16],
) -> [i32; 64] {
    let mut freq = [[0i32; 16]; 4];
    let mut coeffs = [[0i32; 16]; 4];
    let mut pred_dc = [0i32; 4];
    for block in 0..4 {
        let samples = block_of(pred, block);
        freq[block] = forward(&samples);
        pred_dc[block] = freq[block][0];
        let raster = place_ac(&ac[block]);
        coeffs[block] = quant_freq(&freq[block], &raster, quant, weight);
    }
    let dc = chroma_dc(pred_dc, dc, quant, weight);
    let mut out = [0i32; 64];
    for block in 0..4 {
        let samples = inverse(&coeffs[block], quant.to, weight, Some(dc[block]));
        let bx = block & 1;
        let by = block >> 1;
        for row in 0..4 {
            for col in 0..4 {
                out[(by * 4 + row) * 8 + bx * 4 + col] = samples[row * 4 + col];
            }
        }
    }
    out
}

fn block_of(plane: &[i32; 64], block: usize) -> [i32; 16] {
    let bx = block & 1;
    let by = block >> 1;
    let mut out = [0i32; 16];
    for row in 0..4 {
        for col in 0..4 {
            out[row * 4 + col] = plane[(by * 4 + row) * 8 + bx * 4 + col];
        }
    }
    out
}

fn place(scan: &[i32; 16]) -> [i32; 16] {
    let mut raster = [0i32; 16];
    for (index, &pos) in ZIGZAG4.iter().enumerate() {
        raster[pos] = scan[index];
    }
    raster
}

fn place_ac(ac: &[i32; 16]) -> [i32; 16] {
    let mut scan = [0i32; 16];
    scan[1..16].copy_from_slice(&ac[..15]);
    place(&scan)
}

fn quant_freq(
    freq: &[i32; 16],
    levels: &[i32; 16],
    quant: SpQuant,
    weight: &[u8; 16],
) -> [i32; 16] {
    let mut out = [0i32; 16];
    for row in 0..4 {
        for col in 0..4 {
            let index = row * 4 + col;
            let class = class4(row, col);
            let weight = i32::from(weight[index]);
            out[index] = if quant.switching {
                quant_level(i64::from(freq[index]), quant.to, class, false)
                    .saturating_add(levels[index])
            } else {
                let added = scaled_level(levels[index], quant.from, row, col, weight);
                quant_level(i64::from(freq[index]) + added, quant.to, class, false)
            };
        }
    }
    out
}

fn inverse(coeffs: &[i32; 16], qs: i32, weight: &[u8; 16], chroma_dc: Option<i32>) -> [i32; 16] {
    let mut block = [0i32; 16];
    for row in 0..4 {
        for col in 0..4 {
            let index = row * 4 + col;
            block[index] = if row == 0 && col == 0 {
                chroma_dc.unwrap_or_else(|| {
                    scale4(coeffs[index], qs, col, row, i32::from(weight[index]))
                })
            } else {
                scale4(coeffs[index], qs, col, row, i32::from(weight[index]))
            };
        }
    }
    idct4(&mut block);
    block
}

fn scaled_level(level: i32, qp: i32, row: usize, col: usize, weight: i32) -> i64 {
    let scale = i64::from(recon::level_scale4(qp, row, col, weight));
    let factor = i64::from(AIJ[class4(row, col)]);
    let shift = qp.div_euclid(6);
    ((i64::from(level) * scale * factor) << shift) >> 10
}

fn quant_level(value: i64, qs: i32, class: usize, chroma_dc: bool) -> i32 {
    if value == 0 {
        return 0;
    }
    let negative = value < 0;
    let abs = value.saturating_abs();
    let row = qs.rem_euclid(6) as usize;
    let shift_base = qs.div_euclid(6);
    let extra = i32::from(chroma_dc);
    let scale = i64::from(LEVEL2[row][class]);
    let round = 1i64 << (14 + shift_base + extra);
    let mag = (abs * scale + round) >> (15 + shift_base + extra);
    let signed = if negative { -mag } else { mag };
    i32::try_from(signed).unwrap_or(if negative { i32::MIN } else { i32::MAX })
}

fn chroma_dc(pred_dc: [i32; 4], levels: &[i32; 4], quant: SpQuant, weight: &[u8; 16]) -> [i32; 4] {
    let freq = hadamard(pred_dc);
    let weight00 = i32::from(weight[0]);
    let mixed = if quant.switching {
        let mut out = [0i32; 4];
        for index in 0..4 {
            out[index] = quant_level(i64::from(freq[index]), quant.to, 0, true)
                .saturating_add(levels[index]);
        }
        out
    } else {
        let mut out = [0i32; 4];
        let scale = i64::from(recon::level_scale4(quant.from, 0, 0, weight00));
        let shift = quant.from.div_euclid(6);
        for index in 0..4 {
            let added = ((i64::from(levels[index]) * scale * 16) << shift) >> 9;
            out[index] = quant_level(i64::from(freq[index]) + added, quant.to, 0, true);
        }
        out
    };
    let spatial = hadamard(mixed);
    let scale = i64::from(recon::level_scale4(quant.to, 0, 0, weight00));
    let shift = quant.to.div_euclid(6);
    let mut out = [0i32; 4];
    for index in 0..4 {
        let scaled = (i64::from(spatial[index]) * scale) << shift;
        out[index] = i32::try_from(scaled >> 5).unwrap_or(0);
    }
    out
}

fn hadamard(coeff: [i32; 4]) -> [i32; 4] {
    let c = coeff.map(i64::from);
    let mixed = [
        c[0] + c[1] + c[2] + c[3],
        c[0] - c[1] + c[2] - c[3],
        c[0] + c[1] - c[2] - c[3],
        c[0] - c[1] - c[2] + c[3],
    ];
    mixed.map(|value| i32::try_from(value).unwrap_or(0))
}

fn class4(row: usize, col: usize) -> usize {
    if row.is_multiple_of(2) && col.is_multiple_of(2) {
        0
    } else if row % 2 == 1 && col % 2 == 1 {
        1
    } else {
        2
    }
}

fn forward(block: &[i32; 16]) -> [i32; 16] {
    let mut out = *block;
    for row in 0..4 {
        let index = row * 4;
        let s0 = out[index] + out[index + 3];
        let s3 = out[index] - out[index + 3];
        let s1 = out[index + 1] + out[index + 2];
        let s2 = out[index + 1] - out[index + 2];
        out[index] = s0 + s1;
        out[index + 1] = (s3 << 1) + s2;
        out[index + 2] = s0 - s1;
        out[index + 3] = s3 - (s2 << 1);
    }
    for col in 0..4 {
        let s0 = out[col] + out[12 + col];
        let s3 = out[col] - out[12 + col];
        let s1 = out[4 + col] + out[8 + col];
        let s2 = out[4 + col] - out[8 + col];
        out[col] = s0 + s1;
        out[4 + col] = (s3 << 1) + s2;
        out[8 + col] = s0 - s1;
        out[12 + col] = s3 - (s2 << 1);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{SpQuant, chroma_plane, luma_block};

    const WEIGHT: [u8; 16] = [16; 16];

    #[test]
    fn a_flat_prediction_round_trips_at_qs_0_and_drops_one_at_qs_26() {
        let pred = [128i32; 16];
        let zero = [0i32; 16];
        for switching in [false, true] {
            let kept = luma_block(&pred, &zero, quant(26, 0, switching), &WEIGHT);
            assert!(kept.iter().all(|sample| *sample == 128), "{switching}");
            let dropped = luma_block(&pred, &zero, quant(26, 26, switching), &WEIGHT);
            assert!(dropped.iter().all(|sample| *sample == 127), "{switching}");
        }
    }

    #[test]
    fn a_dc_residual_takes_the_switching_path_apart_from_the_other() {
        let pred = [128i32; 16];
        let mut levels = [0i32; 16];
        levels[0] = 1;
        let ordinary = luma_block(&pred, &levels, quant(26, 0, false), &WEIGHT);
        let switching = luma_block(&pred, &levels, quant(26, 0, true), &WEIGHT);
        assert!(ordinary.iter().all(|sample| *sample == 131));
        assert!(switching.iter().all(|sample| *sample == 128));
    }

    #[test]
    fn chroma_dc_uses_the_same_scale_on_both_paths() {
        let pred = [128i32; 64];
        let ac = [[0i32; 16]; 4];
        let zero = [0i32; 4];
        for switching in [false, true] {
            let kept = chroma_plane(&pred, &ac, &zero, quant(26, 0, switching), &WEIGHT);
            assert!(kept.iter().all(|sample| *sample == 128), "{switching}");
        }
        let mut dc = [0i32; 4];
        dc[0] = 1;
        let ordinary = chroma_plane(&pred, &ac, &dc, quant(26, 0, false), &WEIGHT);
        let switching = chroma_plane(&pred, &ac, &dc, quant(26, 0, true), &WEIGHT);
        assert!(ordinary.iter().all(|sample| *sample == 130));
        assert!(switching.iter().all(|sample| *sample == 128));
    }

    fn quant(from: i32, to: i32, switching: bool) -> SpQuant {
        SpQuant {
            from,
            to,
            switching,
        }
    }
}
