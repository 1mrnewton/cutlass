//! Constrained directional enhancement, applied after the loop filter.
//!
//! The filter reads the deblocked picture and writes a separate picture, so a
//! filtered sample is never used as a neighbor in the same pass.

use crate::bits::floor_log2;
use crate::header::CdefParams;

const DIV_TABLE: [i64; 9] = [0, 840, 420, 280, 210, 168, 140, 120, 105];
const PRI_TAPS: [[i32; 2]; 2] = [[4, 2], [3, 3]];
const SEC_TAPS: [[i32; 2]; 2] = [[2, 1], [2, 1]];
const DIRECTIONS: [[[i32; 2]; 2]; 8] = [
    [[-1, 1], [-2, 2]],
    [[0, 1], [-1, 2]],
    [[0, 1], [0, 2]],
    [[0, 1], [1, 2]],
    [[1, 1], [2, 2]],
    [[1, 0], [2, 1]],
    [[1, 0], [2, 0]],
    [[1, 0], [2, -1]],
];
const UV_DIR: [[[u8; 8]; 2]; 2] = [
    [[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 2, 2, 3, 4, 6, 0]],
    [[7, 0, 2, 4, 5, 6, 6, 6], [0, 1, 2, 3, 4, 5, 6, 7]],
];

pub struct CdefInput<'a> {
    pub y: &'a mut [u8],
    pub u: &'a mut [u8],
    pub v: &'a mut [u8],
    pub y_stride: usize,
    pub c_stride: usize,
    pub mi_cols: usize,
    pub mi_rows: usize,
    pub skips: &'a [u8],
    pub index: &'a [i8],
    pub params: CdefParams,
}

pub fn apply(frame: &mut CdefInput<'_>) {
    // A strength pair of zero is the identity. The whole frame is left as the
    // loop filter wrote it, including the copies this pass would otherwise make.
    if idle(&frame.params) {
        return;
    }
    let src_y = frame.y.to_vec();
    let src_u = frame.u.to_vec();
    let src_v = frame.v.to_vec();
    let mut row = 0;
    while row < frame.mi_rows {
        let mut col = 0;
        while col < frame.mi_cols {
            let base_row = row & !15;
            let base_col = col & !15;
            let slot = base_row * frame.mi_cols + base_col;
            let idx = frame.index.get(slot).copied().unwrap_or(-1);
            if idx >= 0 && !skipped(frame.skips, frame.mi_cols, frame.mi_rows, row, col) {
                let idx = idx as usize;
                let y_pri = i32::from(frame.params.y_pri.get(idx).copied().unwrap_or(0));
                let y_sec = i32::from(frame.params.y_sec.get(idx).copied().unwrap_or(0));
                let uv_pri = i32::from(frame.params.uv_pri.get(idx).copied().unwrap_or(0));
                let uv_sec = i32::from(frame.params.uv_sec.get(idx).copied().unwrap_or(0));
                if y_pri != 0 || y_sec != 0 || uv_pri != 0 || uv_sec != 0 {
                    let (dir, var) = if y_pri == 0 && uv_pri == 0 {
                        (0, 0)
                    } else {
                        direction(&src_y, frame.y_stride, col * 4, row * 4)
                    };
                    if y_pri != 0 || y_sec != 0 {
                        filter_luma(frame, &src_y, row, col, idx, dir, var);
                    }
                    if uv_pri != 0 || uv_sec != 0 {
                        filter_chroma(frame, &src_u, &src_v, row, col, idx, dir);
                    }
                }
            }
            col += 2;
        }
        row += 2;
    }
}

fn idle(params: &CdefParams) -> bool {
    params.y_pri.iter().all(|strength| *strength == 0)
        && params.y_sec.iter().all(|strength| *strength == 0)
        && params.uv_pri.iter().all(|strength| *strength == 0)
        && params.uv_sec.iter().all(|strength| *strength == 0)
}

fn skipped(skips: &[u8], mi_cols: usize, mi_rows: usize, row: usize, col: usize) -> bool {
    for dy in 0..2 {
        for dx in 0..2 {
            let y = row + dy;
            let x = col + dx;
            if y >= mi_rows || x >= mi_cols {
                continue;
            }
            if skips.get(y * mi_cols + x).copied().unwrap_or(0) == 0 {
                return false;
            }
        }
    }
    true
}

fn filter_luma(
    frame: &mut CdefInput<'_>,
    src: &[u8],
    row: usize,
    col: usize,
    idx: usize,
    dir: usize,
    var: u32,
) {
    let mut pri = i32::from(frame.params.y_pri.get(idx).copied().unwrap_or(0));
    let sec = i32::from(frame.params.y_sec.get(idx).copied().unwrap_or(0));
    let use_dir = if pri == 0 { 0 } else { dir };
    pri = adjust(pri, var);
    filter_rect(Rect {
        src,
        dst: frame.y,
        stride: frame.y_stride,
        x0: col * 4,
        y0: row * 4,
        width: 8,
        height: 8,
        dir: use_dir,
        pri,
        sec,
        damping: i32::from(frame.params.damping),
        sub_x: false,
        sub_y: false,
        mi_cols: frame.mi_cols,
        mi_rows: frame.mi_rows,
    });
}

fn filter_chroma(
    frame: &mut CdefInput<'_>,
    src_u: &[u8],
    src_v: &[u8],
    row: usize,
    col: usize,
    idx: usize,
    y_dir: usize,
) {
    let pri = i32::from(frame.params.uv_pri.get(idx).copied().unwrap_or(0));
    let sec = i32::from(frame.params.uv_sec.get(idx).copied().unwrap_or(0));
    let dir = if pri == 0 {
        0
    } else {
        usize::from(UV_DIR[1][1][y_dir])
    };
    let damping = i32::from(frame.params.damping).saturating_sub(1);
    filter_rect(Rect {
        src: src_u,
        dst: frame.u,
        stride: frame.c_stride,
        x0: col * 2,
        y0: row * 2,
        width: 4,
        height: 4,
        dir,
        pri,
        sec,
        damping,
        sub_x: true,
        sub_y: true,
        mi_cols: frame.mi_cols,
        mi_rows: frame.mi_rows,
    });
    filter_rect(Rect {
        src: src_v,
        dst: frame.v,
        stride: frame.c_stride,
        x0: col * 2,
        y0: row * 2,
        width: 4,
        height: 4,
        dir,
        pri,
        sec,
        damping,
        sub_x: true,
        sub_y: true,
        mi_cols: frame.mi_cols,
        mi_rows: frame.mi_rows,
    });
}

struct Rect<'a> {
    src: &'a [u8],
    dst: &'a mut [u8],
    stride: usize,
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
    dir: usize,
    pri: i32,
    sec: i32,
    damping: i32,
    sub_x: bool,
    sub_y: bool,
    mi_cols: usize,
    mi_rows: usize,
}

fn adjust(pri: i32, var: u32) -> i32 {
    if var == 0 {
        return 0;
    }
    let var_str = if var >> 6 == 0 {
        0
    } else {
        i32::try_from(floor_log2(var >> 6).min(12)).unwrap_or(0)
    };
    (pri * (4 + var_str) + 8) >> 4
}

fn direction(src: &[u8], stride: usize, x0: usize, y0: usize) -> (usize, u32) {
    let mut partial = [[0i64; 15]; 8];
    for i in 0..8 {
        for j in 0..8 {
            let sample = i64::from(at(src, stride, x0 + j, y0 + i)) - 128;
            partial[0][i + j] += sample;
            partial[1][i + j / 2] += sample;
            partial[2][i] += sample;
            partial[3][3 + i - j / 2] += sample;
            partial[4][7 + i - j] += sample;
            partial[5][3 - i / 2 + j] += sample;
            partial[6][j] += sample;
            partial[7][i / 2 + j] += sample;
        }
    }
    let mut cost = [0i64; 8];
    for bin in partial[2].iter().take(8) {
        cost[2] += bin * bin;
    }
    for bin in partial[6].iter().take(8) {
        cost[6] += bin * bin;
    }
    cost[2] *= DIV_TABLE[8];
    cost[6] *= DIV_TABLE[8];
    for i in 0..7 {
        let div = DIV_TABLE[i + 1];
        cost[0] += (partial[0][i] * partial[0][i] + partial[0][14 - i] * partial[0][14 - i]) * div;
        cost[4] += (partial[4][i] * partial[4][i] + partial[4][14 - i] * partial[4][14 - i]) * div;
    }
    cost[0] += partial[0][7] * partial[0][7] * DIV_TABLE[8];
    cost[4] += partial[4][7] * partial[4][7] * DIV_TABLE[8];
    for i in (1..8).step_by(2) {
        for j in 0..5 {
            cost[i] += partial[i][3 + j] * partial[i][3 + j];
        }
        cost[i] *= DIV_TABLE[8];
        for j in 0..3 {
            let div = DIV_TABLE[2 * j + 2];
            cost[i] +=
                (partial[i][j] * partial[i][j] + partial[i][10 - j] * partial[i][10 - j]) * div;
        }
    }
    let mut best = 0i64;
    let mut dir = 0usize;
    for (i, value) in cost.iter().enumerate() {
        if *value > best {
            best = *value;
            dir = i;
        }
    }
    let var = (best - cost[(dir + 4) & 7]) >> 10;
    (dir, u32::try_from(var).unwrap_or(0))
}

fn filter_rect(block: Rect<'_>) {
    let tap = usize::try_from(block.pri & 1).unwrap_or(0);
    for i in 0..block.height {
        for j in 0..block.width {
            let x = i32::from(at(block.src, block.stride, block.x0 + j, block.y0 + i));
            let mut sum = 0i32;
            let mut max = x;
            let mut min = x;
            for k in 0..2 {
                for sign in [-1, 1] {
                    let mut running = Running { sum, min, max };
                    accumulate(
                        &block,
                        i,
                        j,
                        x,
                        Tap {
                            dir: block.dir,
                            k,
                            sign,
                            primary: true,
                            tap,
                        },
                        &mut running,
                    );
                    for dir_off in [-2, 2] {
                        accumulate(
                            &block,
                            i,
                            j,
                            x,
                            Tap {
                                dir: (block.dir as i32 + dir_off).rem_euclid(8) as usize,
                                k,
                                sign,
                                primary: false,
                                tap,
                            },
                            &mut running,
                        );
                    }
                    sum = running.sum;
                    min = running.min;
                    max = running.max;
                }
            }
            let rounded = (8 + sum - i32::from(sum < 0)) >> 4;
            let value = (x + rounded).clamp(min, max);
            put(block.dst, block.stride, block.x0 + j, block.y0 + i, value);
        }
    }
}

struct Running {
    sum: i32,
    min: i32,
    max: i32,
}

struct Tap {
    dir: usize,
    k: usize,
    sign: i32,
    primary: bool,
    tap: usize,
}

fn accumulate(block: &Rect<'_>, i: usize, j: usize, x: i32, tap: Tap, running: &mut Running) {
    let Some(sample) = neighbor(block, i, j, tap.dir, tap.k, tap.sign) else {
        return;
    };
    let threshold = if tap.primary { block.pri } else { block.sec };
    let weight = if tap.primary {
        PRI_TAPS[tap.tap][tap.k]
    } else {
        SEC_TAPS[tap.tap][tap.k]
    };
    running.sum += weight * constrain(sample - x, threshold, block.damping);
    running.max = running.max.max(sample);
    running.min = running.min.min(sample);
}

fn neighbor(block: &Rect<'_>, i: usize, j: usize, dir: usize, k: usize, sign: i32) -> Option<i32> {
    let dy = sign * DIRECTIONS[dir][k][0];
    let dx = sign * DIRECTIONS[dir][k][1];
    let y = block.y0 as i32 + i as i32 + dy;
    let x = block.x0 as i32 + j as i32 + dx;
    let candidate_r = (y << i32::from(block.sub_y)) >> 2;
    let candidate_c = (x << i32::from(block.sub_x)) >> 2;
    if candidate_c < 0
        || candidate_r < 0
        || candidate_c >= block.mi_cols as i32
        || candidate_r >= block.mi_rows as i32
        || x < 0
        || y < 0
    {
        return None;
    }
    Some(i32::from(at(
        block.src,
        block.stride,
        x as usize,
        y as usize,
    )))
}

fn constrain(diff: i32, threshold: i32, damping: i32) -> i32 {
    if threshold == 0 {
        return 0;
    }
    let shift = (damping - i32::try_from(floor_log2(threshold as u32)).unwrap_or(0)).max(0);
    let sign = if diff < 0 { -1 } else { 1 };
    let abs = diff.abs();
    sign * abs.min(threshold - (abs >> shift)).max(0)
}

fn at(src: &[u8], stride: usize, x: usize, y: usize) -> u8 {
    src.get(y * stride + x).copied().unwrap_or(0)
}

fn put(dst: &mut [u8], stride: usize, x: usize, y: usize, value: i32) {
    if let Some(slot) = dst.get_mut(y * stride + x) {
        *slot = u8::try_from(value).unwrap_or(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_strength_leaves_the_picture() {
        let mut y = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let mut u = vec![11, 22, 33, 44];
        let mut v = vec![55, 66, 77, 88];
        let before_y = y.clone();
        let before_u = u.clone();
        let before_v = v.clone();
        apply(&mut CdefInput {
            y: &mut y,
            u: &mut u,
            v: &mut v,
            y_stride: 4,
            c_stride: 2,
            mi_cols: 2,
            mi_rows: 2,
            skips: &[0, 0, 0, 0],
            index: &[0, -1, -1, -1],
            params: CdefParams {
                damping: 3,
                bits: 0,
                y_pri: [0; 8],
                y_sec: [0; 8],
                uv_pri: [0; 8],
                uv_sec: [0; 8],
            },
        });
        assert_eq!(y, before_y);
        assert_eq!(u, before_u);
        assert_eq!(v, before_v);
    }
}
