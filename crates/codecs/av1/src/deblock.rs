//! In-loop deblocking from section 7.14 of the AV1 specification.
//!
//! Vertical edges are filtered before horizontal edges. Segmentation and
//! per-block filter deltas are not used; frames that enable them are rejected
//! before this runs.

use crate::dims::{MAX_TX_SIZE_RECT, TX_HEIGHT, TX_WIDTH, subsampled_size};
use crate::header::FrameInfo;

const MAX_LOOP_FILTER: i32 = 63;

pub struct Edges<'a> {
    pub sizes: &'a [u8],
    pub tx: &'a [u8],
    pub mi_cols: usize,
    pub mi_rows: usize,
    pub sub_x: bool,
    pub sub_y: bool,
    pub frame: &'a FrameInfo,
}

pub fn deblock_plane(
    samples: &mut [u8],
    width: usize,
    height: usize,
    plane: usize,
    edges: &Edges<'_>,
) {
    // A frame with both luma levels at zero is left unchanged. A reference
    // delta is not applied on top of that pair of zeros.
    if edges.frame.lf_level[0] == 0 && edges.frame.lf_level[1] == 0 {
        return;
    }
    let sub_x = usize::from(plane > 0 && edges.sub_x);
    let sub_y = usize::from(plane > 0 && edges.sub_y);
    if plane > 0 && edges.frame.lf_level[plane + 1] == 0 {
        return;
    }
    let row_step = 1usize << sub_y;
    let col_step = 1usize << sub_x;
    for pass in 0..2 {
        let mut row = 0;
        while row < edges.mi_rows {
            let mut col = 0;
            while col < edges.mi_cols {
                filter_edge(
                    samples,
                    width,
                    height,
                    plane,
                    pass,
                    Mi {
                        row,
                        col,
                        sub_x,
                        sub_y,
                    },
                    edges,
                );
                col += col_step;
            }
            row += row_step;
        }
    }
}

struct Mi {
    row: usize,
    col: usize,
    sub_x: usize,
    sub_y: usize,
}

#[derive(Clone, Copy)]
struct Axis {
    x: i32,
    y: i32,
    dx: i32,
    dy: i32,
}

#[derive(Clone, Copy)]
struct Strength {
    limit: i32,
    blimit: i32,
    thresh: i32,
}

fn filter_edge(
    samples: &mut [u8],
    width: usize,
    height: usize,
    plane: usize,
    pass: usize,
    mi: Mi,
    edges: &Edges<'_>,
) {
    let Mi {
        mut row,
        mut col,
        sub_x,
        sub_y,
    } = mi;
    let (dx, dy) = if pass == 0 { (1i32, 0i32) } else { (0, 1) };
    let x = col * 4;
    let y = row * 4;
    row |= sub_y;
    col |= sub_x;
    if !on_screen(x, y, width << sub_x, height << sub_y, pass) {
        return;
    }
    if row >= edges.mi_rows || col >= edges.mi_cols {
        return;
    }
    let x_p = x >> sub_x;
    let y_p = y >> sub_y;
    let prev_row = row.wrapping_sub((dy as usize) << sub_y);
    let prev_col = col.wrapping_sub((dx as usize) << sub_x);
    if prev_row >= edges.mi_rows || prev_col >= edges.mi_cols {
        return;
    }
    let tx = tx_at(edges, plane, row, col);
    let prev_tx = tx_at(edges, plane, prev_row, prev_col);
    let tx_edge = if pass == 0 {
        x_p.is_multiple_of(TX_WIDTH[tx])
    } else {
        y_p.is_multiple_of(TX_HEIGHT[tx])
    };
    // Intra frames filter every transform edge. Inter prediction boundaries are not used here.
    if !tx_edge {
        return;
    }
    let filter_size = filter_size(pass, plane, tx, prev_tx);
    let lvl = level(edges.frame, plane, pass);
    if lvl == 0 {
        return;
    }
    let (limit, blimit, thresh) = limits(lvl, edges.frame.lf_sharpness);
    let strength = Strength {
        limit,
        blimit,
        thresh,
    };
    for i in 0..4 {
        let axis = Axis {
            x: x_p as i32 + dy * i,
            y: y_p as i32 + dx * i,
            dx,
            dy,
        };
        filter_sample(samples, width, height, axis, strength, filter_size, plane);
    }
}

fn on_screen(x: usize, y: usize, frame_w: usize, frame_h: usize, pass: usize) -> bool {
    if x >= frame_w || y >= frame_h {
        return false;
    }
    if pass == 0 && x == 0 {
        return false;
    }
    if pass == 1 && y == 0 {
        return false;
    }
    true
}

fn tx_at(edges: &Edges<'_>, plane: usize, row: usize, col: usize) -> usize {
    if plane == 0 {
        return edges.tx[row * edges.mi_cols + col] as usize;
    }
    let bsize = edges.sizes[row * edges.mi_cols + col] as usize;
    let uv = MAX_TX_SIZE_RECT[subsampled_size(bsize, edges.sub_x, edges.sub_y)];
    if TX_WIDTH[uv] == 64 || TX_HEIGHT[uv] == 64 {
        if TX_WIDTH[uv] == 16 {
            return 9;
        }
        if TX_HEIGHT[uv] == 16 {
            return 10;
        }
        return 3;
    }
    uv
}

fn filter_size(pass: usize, plane: usize, tx: usize, prev: usize) -> usize {
    let base = if pass == 0 {
        TX_WIDTH[prev].min(TX_WIDTH[tx])
    } else {
        TX_HEIGHT[prev].min(TX_HEIGHT[tx])
    };
    if plane == 0 {
        base.min(16)
    } else {
        base.min(8)
    }
}

fn level(frame: &FrameInfo, plane: usize, pass: usize) -> i32 {
    let index = if plane == 0 { pass } else { plane + 1 };
    let mut lvl = i32::from(frame.lf_level[index]).clamp(0, MAX_LOOP_FILTER);
    if frame.lf_delta_enabled {
        let shift = lvl >> 5;
        lvl += i32::from(frame.lf_ref_deltas[0]) << shift;
        lvl = lvl.clamp(0, MAX_LOOP_FILTER);
    }
    lvl
}

fn limits(lvl: i32, sharpness: u8) -> (i32, i32, i32) {
    let sharp = i32::from(sharpness);
    let shift = if sharp > 4 {
        2
    } else if sharp > 0 {
        1
    } else {
        0
    };
    let limit = if sharp > 0 {
        (lvl >> shift).clamp(1, 9 - sharp)
    } else {
        (lvl >> shift).max(1)
    };
    let blimit = 2 * (lvl + 2) + limit;
    let thresh = lvl >> 4;
    (limit, blimit, thresh)
}

fn filter_sample(
    samples: &mut [u8],
    width: usize,
    height: usize,
    axis: Axis,
    strength: Strength,
    filter_size: usize,
    plane: usize,
) {
    let Some(mask) = masks(samples, width, height, axis, strength, filter_size, plane) else {
        return;
    };
    if !mask.filter {
        return;
    }
    if filter_size == 4 || !mask.flat {
        narrow(samples, width, axis, mask.hev);
    } else if filter_size == 8 || !mask.flat2 {
        wide(samples, width, axis, 3, plane);
    } else {
        wide(samples, width, axis, 4, plane);
    }
}

struct Mask {
    hev: bool,
    filter: bool,
    flat: bool,
    flat2: bool,
}

fn masks(
    samples: &[u8],
    width: usize,
    height: usize,
    axis: Axis,
    strength: Strength,
    filter_size: usize,
    plane: usize,
) -> Option<Mask> {
    let Axis { x, y, dx, dy } = axis;
    let Strength {
        limit,
        blimit,
        thresh,
    } = strength;
    let at = |p: i32| sample(samples, width, height, x + p * dx, y + p * dy);
    let q0 = at(0)?;
    let q1 = at(1)?;
    let p0 = at(-1)?;
    let p1 = at(-2)?;
    let hev = (p1 - p0).abs() > thresh || (q1 - q0).abs() > thresh;
    let filter_len = if filter_size == 4 {
        4
    } else if plane != 0 {
        6
    } else if filter_size == 8 {
        8
    } else {
        16
    };
    let mut mask = (p1 - p0).abs() > limit || (q1 - q0).abs() > limit;
    mask |= (p0 - q0).abs() * 2 + (p1 - q1).abs() / 2 > blimit;
    if filter_len >= 6 {
        let p2 = at(-3)?;
        let q2 = at(2)?;
        mask |= (p2 - p1).abs() > limit || (q2 - q1).abs() > limit;
    }
    if filter_len >= 8 {
        let p2 = at(-3)?;
        let q2 = at(2)?;
        let p3 = at(-4)?;
        let q3 = at(3)?;
        mask |= (p3 - p2).abs() > limit || (q3 - q2).abs() > limit;
    }
    let flat = if filter_size >= 8 {
        let p2 = at(-3)?;
        let q2 = at(2)?;
        let mut flat_mask = (p1 - p0).abs() > 1 || (q1 - q0).abs() > 1;
        flat_mask |= (p2 - p0).abs() > 1 || (q2 - q0).abs() > 1;
        if filter_len >= 8 {
            let p3 = at(-4)?;
            let q3 = at(3)?;
            flat_mask |= (p3 - p0).abs() > 1 || (q3 - q0).abs() > 1;
        }
        !flat_mask
    } else {
        false
    };
    let flat2 = if filter_size >= 16 {
        let mut flat_mask = false;
        for p in [4, 5, 6] {
            flat_mask |= (at(-1 - p)? - p0).abs() > 1 || (at(p)? - q0).abs() > 1;
        }
        !flat_mask
    } else {
        false
    };
    Some(Mask {
        hev,
        filter: !mask,
        flat,
        flat2,
    })
}

fn narrow(samples: &mut [u8], width: usize, axis: Axis, hev: bool) {
    let Axis { x, y, dx, dy } = axis;
    let Some(q0) = sample(samples, width, usize::MAX, x, y) else {
        return;
    };
    let Some(q1) = sample(samples, width, usize::MAX, x + dx, y + dy) else {
        return;
    };
    let Some(p0) = sample(samples, width, usize::MAX, x - dx, y - dy) else {
        return;
    };
    let Some(p1) = sample(samples, width, usize::MAX, x - 2 * dx, y - 2 * dy) else {
        return;
    };
    let ps1 = p1 - 128;
    let ps0 = p0 - 128;
    let qs0 = q0 - 128;
    let qs1 = q1 - 128;
    let mut filter = if hev { clamp4(ps1 - qs1) } else { 0 };
    filter = clamp4(filter + 3 * (qs0 - ps0));
    let filter1 = clamp4(filter + 4) >> 3;
    let filter2 = clamp4(filter + 3) >> 3;
    store(samples, width, x, y, clamp4(qs0 - filter1) + 128);
    store(samples, width, x - dx, y - dy, clamp4(ps0 + filter2) + 128);
    if !hev {
        let filter = round2(filter1, 1);
        store(samples, width, x + dx, y + dy, clamp4(qs1 - filter) + 128);
        store(
            samples,
            width,
            x - 2 * dx,
            y - 2 * dy,
            clamp4(ps1 + filter) + 128,
        );
    }
}

fn wide(samples: &mut [u8], width: usize, axis: Axis, log2_size: u32, plane: usize) {
    let Axis { x, y, dx, dy } = axis;
    let (n, n2) = if log2_size == 4 {
        (6i32, 1i32)
    } else if plane == 0 {
        (3, 0)
    } else {
        (2, 1)
    };
    let mut filtered = [0i32; 12];
    for i in -n..n {
        let mut total = 0i32;
        for j in -n..=n {
            let p = (i + j).clamp(-(n + 1), n);
            let tap = if j.abs() <= n2 { 2 } else { 1 };
            let Some(value) = sample(samples, width, usize::MAX, x + p * dx, y + p * dy) else {
                return;
            };
            total += value * tap;
        }
        filtered[(i + n) as usize] = round2(total, log2_size);
    }
    for i in -n..n {
        store(
            samples,
            width,
            x + i * dx,
            y + i * dy,
            filtered[(i + n) as usize],
        );
    }
}

fn sample(samples: &[u8], width: usize, height: usize, x: i32, y: i32) -> Option<i32> {
    if x < 0 || y < 0 {
        return None;
    }
    let (x, y) = (x as usize, y as usize);
    if x >= width || y >= height {
        return None;
    }
    samples.get(y * width + x).copied().map(i32::from)
}

fn store(samples: &mut [u8], width: usize, x: i32, y: i32, value: i32) {
    if x < 0 || y < 0 {
        return;
    }
    let (x, y) = (x as usize, y as usize);
    if x >= width {
        return;
    }
    if let Some(slot) = samples.get_mut(y * width + x) {
        *slot = value.clamp(0, 255) as u8;
    }
}

fn clamp4(value: i32) -> i32 {
    value.clamp(-128, 127)
}

fn round2(x: i32, n: u32) -> i32 {
    if n == 0 { x } else { (x + (1 << (n - 1))) >> n }
}
