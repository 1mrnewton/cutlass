//! The plain version of every kernel. It is what a CPU without a vector
//! version runs, and what each vector version is tested against.

use crate::deblock::{ChromaRows, Lines, Rows};

/// See [`crate::deblock::luma_normal_rows`].
pub fn luma_normal_rows(rows: &mut Rows, alpha: i32, beta: i32, tc0: [i8; 4]) {
    for column in 0..16 {
        let tc0 = i32::from(tc0[column / 4]);
        if tc0 < 0 {
            continue;
        }
        let [_, p2, p1, p0, q0, q1, q2, _] = line(rows, column);
        if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
            continue;
        }
        let ap = (p2 - p0).abs();
        let aq = (q2 - q0).abs();
        let tc = tc0 + i32::from(ap < beta) + i32::from(aq < beta);
        let delta = ((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
        rows[3][column] = clip(p0 + delta);
        rows[4][column] = clip(q0 - delta);
        if ap < beta {
            let d = ((p2 + ((p0 + q0 + 1) >> 1) - (p1 << 1)) >> 1).clamp(-tc0, tc0);
            rows[2][column] = clip(p1 + d);
        }
        if aq < beta {
            let d = ((q2 + ((p0 + q0 + 1) >> 1) - (q1 << 1)) >> 1).clamp(-tc0, tc0);
            rows[5][column] = clip(q1 + d);
        }
    }
}

/// See [`crate::deblock::luma_strong_rows`].
pub fn luma_strong_rows(rows: &mut Rows, alpha: i32, beta: i32) {
    for column in 0..16 {
        let [p3, p2, p1, p0, q0, q1, q2, q3] = line(rows, column);
        if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
            continue;
        }
        let small = (p0 - q0).abs() < ((alpha >> 2) + 2);
        if small && (p2 - p0).abs() < beta {
            rows[3][column] = clip((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3);
            rows[2][column] = clip((p2 + p1 + p0 + q0 + 2) >> 2);
            rows[1][column] = clip((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3);
        } else {
            rows[3][column] = clip((2 * p1 + p0 + q1 + 2) >> 2);
        }
        if small && (q2 - q0).abs() < beta {
            rows[4][column] = clip((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3);
            rows[5][column] = clip((p0 + q0 + q1 + q2 + 2) >> 2);
            rows[6][column] = clip((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3);
        } else {
            rows[4][column] = clip((2 * q1 + q0 + p1 + 2) >> 2);
        }
    }
}

/// See [`crate::deblock::luma_normal_lines`].
pub fn luma_normal_lines(lines: &mut Lines, alpha: i32, beta: i32, tc0: [i8; 4]) {
    let mut rows = turned(lines);
    luma_normal_rows(&mut rows, alpha, beta, tc0);
    turn_back(&rows, lines);
}

/// See [`crate::deblock::luma_strong_lines`].
pub fn luma_strong_lines(lines: &mut Lines, alpha: i32, beta: i32) {
    let mut rows = turned(lines);
    luma_strong_rows(&mut rows, alpha, beta);
    turn_back(&rows, lines);
}

fn line(rows: &Rows, column: usize) -> [i32; 8] {
    let mut line = [0i32; 8];
    for (sample, row) in line.iter_mut().zip(rows) {
        *sample = i32::from(row[column]);
    }
    line
}

fn clip(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

fn turned(lines: &Lines) -> Rows {
    let mut rows = [[0u8; 16]; 8];
    for (index, line) in lines.iter().enumerate() {
        for (row, &sample) in rows.iter_mut().zip(line) {
            row[index] = sample;
        }
    }
    rows
}

fn turn_back(rows: &Rows, lines: &mut Lines) {
    for (index, line) in lines.iter_mut().enumerate() {
        for (sample, row) in line.iter_mut().zip(rows) {
            *sample = row[index];
        }
    }
}

/// The six-tap filter's sum at one sample.
fn six(taps: [i32; 6]) -> i32 {
    taps[0] - 5 * taps[1] + 20 * taps[2] + 20 * taps[3] - 5 * taps[4] + taps[5]
}

fn six_at<T: Copy + Into<i32>, const N: usize>(taps: [&[T; N]; 6], index: usize) -> i32 {
    six(taps.map(|line| line[index].into()))
}

/// See [`crate::mc::half16`].
pub fn half16(taps: [&[u8; 16]; 6]) -> [u8; 16] {
    core::array::from_fn(|index| clip((six_at(taps, index) + 16) >> 5))
}

/// See [`crate::mc::half8`].
pub fn half8(taps: [&[u8; 8]; 6]) -> [u8; 8] {
    core::array::from_fn(|index| clip((six_at(taps, index) + 16) >> 5))
}

/// See [`crate::mc::raw16`].
pub fn raw16(taps: [&[u8; 16]; 6]) -> [i16; 16] {
    core::array::from_fn(|index| six_at(taps, index) as i16)
}

/// See [`crate::mc::raw8`].
pub fn raw8(taps: [&[u8; 8]; 6]) -> [i16; 8] {
    core::array::from_fn(|index| six_at(taps, index) as i16)
}

/// See [`crate::mc::centre16`].
pub fn centre16(taps: [&[i16; 16]; 6]) -> [u8; 16] {
    core::array::from_fn(|index| clip((six_at(taps, index) + 512) >> 10))
}

/// See [`crate::mc::centre8`].
pub fn centre8(taps: [&[i16; 8]; 6]) -> [u8; 8] {
    core::array::from_fn(|index| clip((six_at(taps, index) + 512) >> 10))
}

/// See [`crate::mc::avg16`].
pub fn avg16(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    core::array::from_fn(|index| ((u16::from(a[index]) + u16::from(b[index]) + 1) >> 1) as u8)
}

/// See [`crate::mc::avg8`].
pub fn avg8(a: &[u8; 8], b: &[u8; 8]) -> [u8; 8] {
    core::array::from_fn(|index| ((u16::from(a[index]) + u16::from(b[index]) + 1) >> 1) as u8)
}

/// See [`crate::mc::chroma8`].
pub fn chroma8(top: &[u8; 9], bottom: &[u8; 9], xf: u8, yf: u8) -> [u8; 8] {
    let (xf, yf) = (i32::from(xf.min(7)), i32::from(yf.min(7)));
    core::array::from_fn(|index| {
        let value = (8 - xf) * (8 - yf) * i32::from(top[index])
            + xf * (8 - yf) * i32::from(top[index + 1])
            + (8 - xf) * yf * i32::from(bottom[index])
            + xf * yf * i32::from(bottom[index + 1]);
        clip((value + 32) >> 6)
    })
}

/// See [`crate::deblock::chroma_normal`].
pub fn chroma_normal(rows: &mut ChromaRows, alpha: i32, beta: i32, tc0: [i8; 4]) {
    for column in 0..8 {
        let tc0 = i32::from(tc0[column / 2]);
        if tc0 < 0 {
            continue;
        }
        let [p1, p0, q0, q1] = chroma_line(rows, column);
        if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
            continue;
        }
        let tc = tc0 + 1;
        let delta = ((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
        rows[1][column] = clip(p0 + delta);
        rows[2][column] = clip(q0 - delta);
    }
}

/// See [`crate::deblock::chroma_strong`].
pub fn chroma_strong(rows: &mut ChromaRows, alpha: i32, beta: i32) {
    for column in 0..8 {
        let [p1, p0, q0, q1] = chroma_line(rows, column);
        if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
            continue;
        }
        rows[1][column] = clip((2 * p1 + p0 + q1 + 2) >> 2);
        rows[2][column] = clip((2 * q1 + q0 + p1 + 2) >> 2);
    }
}

fn chroma_line(rows: &ChromaRows, column: usize) -> [i32; 4] {
    let mut line = [0i32; 4];
    for (sample, row) in line.iter_mut().zip(rows) {
        *sample = i32::from(row[column]);
    }
    line
}

/// See [`crate::mc::weight_block`]; sixteen samples of it.
pub fn weight16(a: &[u8; 16], b: &[u8; 16], weight: crate::mc_block::BiWeight) -> [u8; 16] {
    core::array::from_fn(|index| crate::mc_block::weight_one(a[index], b[index], weight))
}

/// [`weight16`] for eight samples.
pub fn weight8(a: &[u8; 8], b: &[u8; 8], weight: crate::mc_block::BiWeight) -> [u8; 8] {
    core::array::from_fn(|index| crate::mc_block::weight_one(a[index], b[index], weight))
}

crate::mc_block::block_kernels!(pub,);
crate::mc_block::predict_kernels!(pub,);

/// Eight lines of four samples as four rows of eight.
fn chroma_lines(lines: &[[u8; 4]; 8]) -> ChromaRows {
    let mut rows = [[0u8; 8]; 4];
    for (index, line) in lines.iter().enumerate() {
        for (row, &sample) in rows.iter_mut().zip(line) {
            row[index] = sample;
        }
    }
    rows
}

crate::edge::edge_kernels!(pub,);

/// See [`crate::deblock::luma_mb`]: each edge in turn.
pub fn luma_mb(plane: &mut [u8], edges: &crate::edge::MbEdges) -> bool {
    if !crate::edge::mb_fits(plane.len(), edges, 16, 4) {
        return false;
    }
    crate::edge::each_edge(edges, 4, |edge| {
        luma_edge(plane, edge);
    });
    true
}

/// See [`crate::deblock::chroma_mb`]: each edge in turn.
pub fn chroma_mb(plane: &mut [u8], edges: &crate::edge::MbEdges) -> bool {
    if !crate::edge::mb_fits(plane.len(), edges, 8, 2) {
        return false;
    }
    crate::edge::each_edge(edges, 2, |edge| {
        chroma_edge(plane, edge);
    });
    true
}

/// The transform of clause 8.5.12.2, then each sample plus its residual,
/// clipped.
fn idct4_rows(block: &[i32; 16], rows: [[u8; 4]; 4]) -> [[u8; 4]; 4] {
    let mut mid = [0i32; 16];
    for i in 0..4 {
        let [d0, d1, d2, d3] = [
            block[i * 4],
            block[i * 4 + 1],
            block[i * 4 + 2],
            block[i * 4 + 3],
        ];
        let e0 = d0.wrapping_add(d2);
        let e1 = d0.wrapping_sub(d2);
        let e2 = (d1 >> 1).wrapping_sub(d3);
        let e3 = d1.wrapping_add(d3 >> 1);
        mid[i * 4] = e0.wrapping_add(e3);
        mid[i * 4 + 1] = e1.wrapping_add(e2);
        mid[i * 4 + 2] = e1.wrapping_sub(e2);
        mid[i * 4 + 3] = e0.wrapping_sub(e3);
    }
    let mut out = rows;
    for j in 0..4 {
        let [d0, d1, d2, d3] = [mid[j], mid[4 + j], mid[8 + j], mid[12 + j]];
        let e0 = d0.wrapping_add(d2);
        let e1 = d0.wrapping_sub(d2);
        let e2 = (d1 >> 1).wrapping_sub(d3);
        let e3 = d1.wrapping_add(d3 >> 1);
        let column = [
            e0.wrapping_add(e3),
            e1.wrapping_add(e2),
            e1.wrapping_sub(e2),
            e0.wrapping_sub(e3),
        ];
        for (row, value) in out.iter_mut().zip(column) {
            let residual = value.wrapping_add(32) >> 6;
            row[j] = clip(i32::from(row[j]).saturating_add(residual));
        }
    }
    out
}

/// `dc` added to each of the eight samples in `samples`, each held to 0 to
/// 255.
fn dc_row(samples: u64, dc: i32) -> u64 {
    u64::from_le_bytes(
        samples
            .to_le_bytes()
            .map(|sample| (i32::from(sample) + dc.clamp(-255, 255)).clamp(0, 255) as u8),
    )
}

crate::residual::residual_kernels!(pub,);

/// See [`crate::mc::luma_pair`]: each block made whole on its own, then the
/// two averaged.
pub fn luma_pair(
    dst: &mut [u8],
    dst_stride: usize,
    first: (crate::mc_block::Block<'_>, (u8, u8)),
    second: Option<(crate::mc_block::Block<'_>, (u8, u8))>,
) -> bool {
    pair(dst, dst_stride, first, second, luma_block)
}

/// See [`crate::mc::chroma_pair`].
pub fn chroma_pair(
    dst: &mut [u8],
    dst_stride: usize,
    first: (crate::mc_block::Block<'_>, (u8, u8)),
    second: Option<(crate::mc_block::Block<'_>, (u8, u8))>,
) -> bool {
    pair(dst, dst_stride, first, second, chroma_block)
}

type Predictor = fn(&mut [u8], usize, crate::mc_block::Block<'_>, (u8, u8)) -> bool;

fn pair(
    dst: &mut [u8],
    dst_stride: usize,
    first: (crate::mc_block::Block<'_>, (u8, u8)),
    second: Option<(crate::mc_block::Block<'_>, (u8, u8))>,
    predict: Predictor,
) -> bool {
    let (w, h) = first.0.size;
    if w == 0
        || h == 0
        || w > 16
        || h > 16
        || dst_stride < w
        || (h - 1) * dst_stride + w > dst.len()
    {
        return false;
    }
    let mut one = [0u8; 256];
    if !predict(&mut one, w, first.0, first.1) {
        return false;
    }
    let mut two = [0u8; 256];
    if let Some((block, frac)) = second
        && (block.size != first.0.size || !predict(&mut two, w, block, frac))
    {
        return false;
    }
    for row in 0..h {
        let out = &mut dst[row * dst_stride..row * dst_stride + w];
        let a = &one[row * w..row * w + w];
        if second.is_some() {
            let b = &two[row * w..row * w + w];
            for ((sample, &x), &y) in out.iter_mut().zip(a).zip(b) {
                *sample = ((u16::from(x) + u16::from(y) + 1) >> 1) as u8;
            }
        } else {
            out.copy_from_slice(a);
        }
    }
    true
}
