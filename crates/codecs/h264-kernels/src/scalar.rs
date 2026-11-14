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

/// See [`crate::deblock::mb`].
pub fn mb_planes(
    planes: (&mut [u8], &mut [u8], &mut [u8]),
    edges: &[crate::edge::MbEdges; 3],
    some: [bool; 3],
) -> bool {
    let (luma, cb, cr) = planes;
    let [for_luma, for_cb, for_cr] = edges;
    let [some_luma, some_cb, some_cr] = some;
    if (some_luma && !for_luma.fits(luma.len(), 16))
        || (some_cb && !for_cb.fits(cb.len(), 8))
        || (some_cr && !for_cr.fits(cr.len(), 8))
    {
        return false;
    }
    (!some_luma || luma_mb(luma, for_luma))
        && (!some_cb || chroma_mb(cb, for_cb))
        && (!some_cr || chroma_mb(cr, for_cr))
}

/// See [`crate::deblock::mb_twin`]: [`mb_planes`] with `chroma` for both
/// chroma planes.
pub fn mb_twin(
    planes: (&mut [u8], &mut [u8], &mut [u8]),
    luma: &crate::edge::MbEdges,
    chroma: &crate::edge::MbEdges,
    some: [bool; 2],
) -> bool {
    let [some_luma, some_chroma] = some;
    mb_planes(
        planes,
        &[*luma, *chroma, *chroma],
        [some_luma, some_chroma, some_chroma],
    )
}

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

/// One line of the 8×8 transform, clause 8.5.13.2, wrapping as the
/// decoder's 32-bit arithmetic does.
fn idct8_line(d: [i32; 8]) -> [i32; 8] {
    let a0 = d[0].wrapping_add(d[4]);
    let a2 = d[0].wrapping_sub(d[4]);
    let a4 = (d[2] >> 1).wrapping_sub(d[6]);
    let a6 = (d[6] >> 1).wrapping_add(d[2]);
    let b0 = a0.wrapping_add(a6);
    let b2 = a2.wrapping_add(a4);
    let b4 = a2.wrapping_sub(a4);
    let b6 = a0.wrapping_sub(a6);
    let a1 = d[5]
        .wrapping_sub(d[3])
        .wrapping_sub(d[7])
        .wrapping_sub(d[7] >> 1);
    let a3 = d[1]
        .wrapping_add(d[7])
        .wrapping_sub(d[3])
        .wrapping_sub(d[3] >> 1);
    let a5 = d[7]
        .wrapping_sub(d[1])
        .wrapping_add(d[5])
        .wrapping_add(d[5] >> 1);
    let a7 = d[3]
        .wrapping_add(d[5])
        .wrapping_add(d[1])
        .wrapping_add(d[1] >> 1);
    let b1 = (a7 >> 2).wrapping_add(a1);
    let b3 = a3.wrapping_add(a5 >> 2);
    let b5 = (a3 >> 2).wrapping_sub(a5);
    let b7 = a7.wrapping_sub(a1 >> 2);
    [
        b0.wrapping_add(b7),
        b2.wrapping_add(b5),
        b4.wrapping_add(b3),
        b6.wrapping_add(b1),
        b6.wrapping_sub(b1),
        b4.wrapping_sub(b3),
        b2.wrapping_sub(b5),
        b0.wrapping_sub(b7),
    ]
}

/// The transform of clause 8.5.13 along each row and then down each
/// column, then each sample plus its residual, clipped.
fn idct8_rows(block: &[i32; 64], rows: [[u8; 8]; 8]) -> [[u8; 8]; 8] {
    let mut mid = [[0i32; 8]; 8];
    for (i, line) in mid.iter_mut().enumerate() {
        *line = idct8_line(core::array::from_fn(|k| block[i * 8 + k]));
    }
    let mut out = rows;
    for j in 0..8 {
        let column = idct8_line(core::array::from_fn(|i| mid[i][j]));
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

/// See [`crate::residual::store`]: row by row.
pub fn store<const N: usize>(
    plane: &mut [u8],
    at: usize,
    down: usize,
    rows: &[[u8; N]; N],
) -> bool {
    if down < N
        || (N.saturating_sub(1))
            .checked_mul(down)
            .and_then(|span| span.checked_add(at))
            .and_then(|start| start.checked_add(N))
            .is_none_or(|end| end > plane.len())
    {
        return false;
    }
    for (index, row) in rows.iter().enumerate() {
        let start = at + index * down;
        plane[start..start + N].copy_from_slice(row);
    }
    true
}

crate::residual::residual_kernels!(pub,);
crate::residual::dc_add_rows!(pub,);

/// See [`crate::residual::dc_add_quads`]: each sample of each quarter
/// moved by its DC and held to 0 to 255.
pub fn dc_add_quads(plane: &mut [u8], at: usize, down: usize, dcs: [i32; 4]) -> bool {
    if down < 8
        || 7usize
            .checked_mul(down)
            .and_then(|rows| rows.checked_add(8))
            .and_then(|span| span.checked_add(at))
            .is_none_or(|end| end > plane.len())
    {
        return false;
    }
    for row in 0..8 {
        for column in 0..8 {
            let dc = dcs[(row / 4) * 2 + column / 4].clamp(-255, 255);
            let sample = &mut plane[at + row * down + column];
            *sample = (i32::from(*sample) + dc).clamp(0, 255) as u8;
        }
    }
    true
}

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

/// See [`crate::mc::chroma_planes`].
pub fn chroma_planes(
    dst: (&mut [u8], &mut [u8]),
    dst_stride: usize,
    size: (usize, usize),
    first: crate::mc_block::ChromaSource<'_>,
    second: Option<crate::mc_block::ChromaSource<'_>>,
) -> bool {
    let (w, h) = size;
    if (w != 4 && w != 8) || h == 0 || h > 8 || h % 2 != 0 {
        return false;
    }
    // Both planes are made aside first, so a failure writes nothing.
    let (dst_cb, dst_cr) = dst;
    let mut made = [[0u8; 64]; 2];
    for (cb, out) in [(true, 0), (false, 1)] {
        let one = (first.plane(cb, size), first.frac);
        let two = second.map(|source| (source.plane(cb, size), source.frac));
        if !chroma_pair(&mut made[out], w, one, two) {
            return false;
        }
    }
    if dst_stride < w || (h - 1) * dst_stride + w > dst_cb.len().min(dst_cr.len()) {
        return false;
    }
    for (dst, made) in [(dst_cb, &made[0]), (dst_cr, &made[1])] {
        for row in 0..h {
            dst[row * dst_stride..row * dst_stride + w]
                .copy_from_slice(&made[row * w..row * w + w]);
        }
    }
    true
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

/// See [`crate::mc::weight_onto`].
pub fn weight_onto(
    dst: &mut [u8],
    dst_stride: usize,
    b: &[u8],
    size: (usize, usize),
    weight: crate::mc_block::BiWeight,
) -> bool {
    if !crate::mc_block::onto_fits(dst, dst_stride, b, size, weight) {
        return false;
    }
    let (w, h) = size;
    for row in 0..h {
        let out = &mut dst[row * dst_stride..row * dst_stride + w];
        for (sample, &right) in out.iter_mut().zip(&b[row * w..row * w + w]) {
            *sample = crate::mc_block::weight_one(*sample, right, weight);
        }
    }
    true
}

/// See [`crate::mc::scale_block`].
pub fn scale_block(
    dst: &mut [u8],
    dst_stride: usize,
    size: (usize, usize),
    weight: crate::mc_block::UniWeight,
) -> bool {
    if !weight.fits() || !crate::mc_block::scale_fits(dst, dst_stride, size) {
        return false;
    }
    let (w, h) = size;
    for row in 0..h {
        for sample in &mut dst[row * dst_stride..row * dst_stride + w] {
            let product = i32::from(*sample) * weight.weight;
            let scaled = if weight.shift > 0 {
                (product + (1 << (weight.shift - 1))) >> weight.shift
            } else {
                product
            };
            *sample = (scaled + weight.offset).clamp(0, 255) as u8;
        }
    }
    true
}

/// A cache hint, which has no plain counterpart: it does nothing here.
pub fn prefetch(_plane: &[u8], _at: usize, _stride: usize, _rows: usize) {}

/// A cache hint, which has no plain counterpart: it does nothing here.
pub fn prefetch_item<T>(_items: &[T], _at: usize) {}

/// See [`crate::mc::predict_420`]: the same checks, and the blocks made by
/// [`luma_pair`] and [`chroma_planes`].
pub fn predict_420<'r>(
    dst: &mut crate::mc_block::Planes420Mut<'_>,
    block: (usize, usize, usize, usize),
    first: (&crate::mc_block::Planes420<'r>, [i16; 2]),
    second: Option<(&crate::mc_block::Planes420<'r>, [i16; 2])>,
) -> (bool, bool) {
    use crate::mc_block::{Block, ChromaSource, Planes420};
    let (x, y, w, h) = block;
    if !dst.holds((x, y), (w, h)) {
        return (false, false);
    }
    let (dst_y, dst_u, dst_v, stride) = dst.planes();
    let luma = (w == 8 || w == 16) && (1..=16).contains(&h) && {
        let from = |(planes, mv): (&Planes420<'r>, [i16; 2])| {
            let at = (
                x as i64 + i64::from(mv[0] >> 2),
                y as i64 + i64::from(mv[1] >> 2),
            );
            let frac = ((mv[0] & 3) as u8, (mv[1] & 3) as u8);
            planes.luma_from(at, (w, h), frac)?;
            let block = Block {
                src: planes.y(),
                stride: planes.stride(),
                at: (at.0 as usize, at.1 as usize),
                size: (w, h),
            };
            Some((block, frac))
        };
        match (from(first), second.map(from)) {
            (Some(one), None) => luma_pair(&mut dst_y[y * stride + x..], stride, one, None),
            (Some(one), Some(Some(two))) => {
                luma_pair(&mut dst_y[y * stride + x..], stride, one, Some(two))
            }
            _ => false,
        }
    };
    let (cw, ch) = (w / 2, h / 2);
    let chroma = (cw == 4 || cw == 8) && ch != 0 && ch <= 8 && ch % 2 == 0 && {
        let (cx, cy) = (x / 2, y / 2);
        let from = |(planes, mv): (&Planes420<'r>, [i16; 2])| {
            let at = (
                cx as i64 + i64::from(mv[0] >> 3),
                cy as i64 + i64::from(mv[1] >> 3),
            );
            let frac = ((mv[0] & 7) as u8, (mv[1] & 7) as u8);
            planes.chroma_from(at, (cw, ch), frac)?;
            Some(ChromaSource {
                cb: planes.cb(),
                cr: planes.cr(),
                stride: planes.stride() / 2,
                at: (at.0 as usize, at.1 as usize),
                frac,
            })
        };
        let stride = stride / 2;
        let out = cy * stride + cx;
        let planes = (&mut dst_u[out..], &mut dst_v[out..]);
        match (from(first), second.map(from)) {
            (Some(one), None) => chroma_planes(planes, stride, (cw, ch), one, None),
            (Some(one), Some(Some(two))) => chroma_planes(planes, stride, (cw, ch), one, Some(two)),
            _ => false,
        }
    };
    (luma, chroma)
}

/// The samples beside an `N`×`N` block of a plane: the corner, the row
/// above and the column to the left.
fn beside<const N: usize>(plane: &[u8], start: usize, down: usize) -> (i32, [i32; N], [i32; N]) {
    (
        i32::from(plane[start]),
        core::array::from_fn(|x| i32::from(plane[start + 1 + x])),
        core::array::from_fn(|y| i32::from(plane[start + (y + 1) * down])),
    )
}

/// The plane prediction's gradient along `samples`, equations 8-128 and
/// 8-129 and their chroma twins: the sample before the first is `corner`.
fn gradient<const N: usize>(samples: &[i32; N], corner: i32) -> i32 {
    let half = N / 2;
    (0..half)
        .map(|step| {
            let before = if step == half - 1 {
                corner
            } else {
                samples[half - 2 - step]
            };
            (step as i32 + 1) * (samples[half + step] - before)
        })
        .sum()
}

/// See [`crate::intra::pred16`]: each sample worked out on its own.
pub fn pred16(plane: &mut [u8], at: usize, down: usize, mode: u8, near: (bool, bool)) -> bool {
    let Some(start) = crate::intra::corner_at::<16>(plane.len(), at, down) else {
        return false;
    };
    let (corner, up, side) = beside::<16>(plane, start, down);
    let (above, left) = near;
    let dc = match (above, left) {
        (true, true) => (up.iter().sum::<i32>() + side.iter().sum::<i32>() + 16) >> 5,
        (true, false) => (up.iter().sum::<i32>() + 8) >> 4,
        (false, true) => (side.iter().sum::<i32>() + 8) >> 4,
        (false, false) => 128,
    };
    let (h, v) = (gradient(&up, corner), gradient(&side, corner));
    let (a, b, c) = (
        16 * (up[15] + side[15]),
        (5 * h + 32) >> 6,
        (5 * v + 32) >> 6,
    );
    for y in 0..16 {
        for x in 0..16 {
            let sample = match mode {
                0 if above => up[x],
                1 if left => side[y],
                2 => dc,
                3 if above && left => (a + b * (x as i32 - 7) + c * (y as i32 - 7) + 16) >> 5,
                _ => 128,
            };
            plane[at + y * down + x] = sample.clamp(0, 255) as u8;
        }
    }
    true
}

/// See [`crate::intra::pred_chroma`]: each sample worked out on its own.
pub fn pred_chroma(plane: &mut [u8], at: usize, down: usize, mode: u8, near: (bool, bool)) -> bool {
    let Some(start) = crate::intra::corner_at::<8>(plane.len(), at, down) else {
        return false;
    };
    let (corner, up, side) = beside::<8>(plane, start, down);
    let (above, left) = near;
    let sum = |samples: &[i32; 8], half: usize| samples[half * 4..half * 4 + 4].iter().sum::<i32>();
    // The DC of each 4×4 quarter: from both edges beside it when the
    // quarter touches both, else from the one edge it touches, else from
    // the other.
    let quarter = |x: usize, y: usize| {
        let (along, down_side) = (sum(&up, x), sum(&side, y));
        let both = (along + down_side + 4) >> 3;
        match (above, left) {
            (true, true) if x == y => both,
            (true, true) if y == 0 => (along + 2) >> 2,
            (true, true) => (down_side + 2) >> 2,
            (true, false) => (along + 2) >> 2,
            (false, true) => (down_side + 2) >> 2,
            (false, false) => 128,
        }
    };
    let dcs = [
        [quarter(0, 0), quarter(1, 0)],
        [quarter(0, 1), quarter(1, 1)],
    ];
    let (h, v) = (gradient(&up, corner), gradient(&side, corner));
    let (a, b, c) = (
        16 * (up[7] + side[7]),
        (17 * h + 16) >> 5,
        (17 * v + 16) >> 5,
    );
    for y in 0..8 {
        for x in 0..8 {
            let sample = match mode {
                1 if left => side[y],
                2 if above => up[x],
                3 if above && left => (a + b * (x as i32 - 3) + c * (y as i32 - 3) + 16) >> 5,
                _ => dcs[y / 4][x / 4],
            };
            plane[at + y * down + x] = sample.clamp(0, 255) as u8;
        }
    }
    true
}

/// The filtered edges of the 8×8 block at `at`, as the decoder gathers and
/// filters them: missing samples are 128 and stay unfiltered, missing
/// samples above and right repeat the last one above, and a missing corner
/// is left out of the filter.
fn edges8(
    plane: &[u8],
    (start, at, down): (usize, usize, usize),
    near: crate::intra::Near8,
) -> crate::intra::Edges8 {
    use crate::intra::avg3;
    let mut raw_above = [128u8; 16];
    if near.above {
        raw_above[..8].copy_from_slice(&plane[start + 1..start + 9]);
        if near.above_right {
            raw_above[8..].copy_from_slice(&plane[start + 9..start + 17]);
        } else {
            let last = raw_above[7];
            raw_above[8..].fill(last);
        }
    }
    let mut raw_left = [128u8; 8];
    if near.left {
        for (y, sample) in raw_left.iter_mut().enumerate() {
            *sample = plane[at - 1 + y * down];
        }
    }
    let corner = if near.corner { plane[start] } else { 128 };
    let mut above = raw_above;
    if near.above {
        let first = if near.corner { corner } else { raw_above[0] };
        above[0] = avg3(first, raw_above[0], raw_above[1]);
        for i in 1..15 {
            above[i] = avg3(raw_above[i - 1], raw_above[i], raw_above[i + 1]);
        }
        above[15] = avg3(raw_above[14], raw_above[15], raw_above[15]);
    }
    let mut left = raw_left;
    if near.left {
        let first = if near.corner { corner } else { raw_left[0] };
        left[0] = avg3(first, raw_left[0], raw_left[1]);
        for i in 1..7 {
            left[i] = avg3(raw_left[i - 1], raw_left[i], raw_left[i + 1]);
        }
        left[7] = avg3(raw_left[6], raw_left[7], raw_left[7]);
    }
    let corner = match (near.corner, near.above, near.left) {
        (true, true, true) => avg3(raw_above[0], corner, raw_left[0]),
        (true, true, false) => avg3(corner, corner, raw_above[0]),
        (true, false, true) => avg3(corner, corner, raw_left[0]),
        _ => corner,
    };
    crate::intra::Edges8 {
        above,
        left,
        corner,
    }
}

/// See [`crate::intra::pred8`]: each sample worked out on its own.
pub fn pred8(
    plane: &mut [u8],
    at: usize,
    down: usize,
    mode: u8,
    near: crate::intra::Near8,
) -> bool {
    use crate::intra::{avg2, avg3, sample8};
    let Some(start) = crate::intra::corner8_at(plane.len(), at, down) else {
        return false;
    };
    let edges = edges8(plane, (start, at, down), near);
    let (above, left, corner) = (&edges.above, &edges.left, edges.corner);
    let dc = match (near.above, near.left) {
        (true, true) => {
            let sum: u32 = above[..8].iter().chain(left).map(|&s| u32::from(s)).sum();
            ((sum + 8) >> 4) as u8
        }
        (true, false) => ((above[..8].iter().map(|&s| u32::from(s)).sum::<u32>() + 4) >> 3) as u8,
        (false, true) => ((left.iter().map(|&s| u32::from(s)).sum::<u32>() + 4) >> 3) as u8,
        (false, false) => 128,
    };
    let both = near.above && near.left;
    for y in 0..8 {
        for x in 0..8 {
            let sample = match mode {
                0 if near.above => above[x],
                1 if near.left => left[y],
                3 if near.above => {
                    if x == 7 && y == 7 {
                        avg3(above[14], above[15], above[15])
                    } else {
                        avg3(above[x + y], above[x + y + 1], above[x + y + 2])
                    }
                }
                4 if both => {
                    let at = |i: isize| match usize::try_from(i) {
                        Ok(i) => above[i],
                        Err(_) => corner,
                    };
                    let beside = |i: isize| match usize::try_from(i) {
                        Ok(i) => left[i],
                        Err(_) => corner,
                    };
                    let (x, y) = (x as isize, y as isize);
                    if x > y {
                        avg3(at(x - y - 2), at(x - y - 1), at(x - y))
                    } else if x < y {
                        avg3(beside(y - x - 2), beside(y - x - 1), beside(y - x))
                    } else {
                        avg3(above[0], corner, left[0])
                    }
                }
                5 | 6 if both => sample8(mode, x, y, &edges),
                7 if near.above => {
                    let base = x + (y >> 1);
                    if y % 2 == 0 {
                        avg2(above[base], above[base + 1])
                    } else {
                        avg3(above[base], above[base + 1], above[base + 2])
                    }
                }
                8 if near.left => sample8(mode, x, y, &edges),
                _ => dc,
            };
            plane[at + y * down + x] = sample;
        }
    }
    true
}

/// See [`crate::intra::pred4`]: each sample worked out on its own.
pub fn pred4(
    plane: &mut [u8],
    at: usize,
    down: usize,
    mode: u8,
    near: crate::intra::Near4,
) -> bool {
    use crate::intra::{dc4, edges4, sample4};
    let Some(start) = crate::intra::corner4_at(plane.len(), at, down) else {
        return false;
    };
    let Some(&top) = plane[start..].first_chunk::<9>() else {
        return false;
    };
    let side = core::array::from_fn(|y| plane[at - 1 + y * down]);
    let (above, left, corner) = edges4(&top, side, near);
    let directional = match mode {
        0 | 3 | 7 => near.above,
        1 | 8 => near.left,
        4..=6 => near.above && near.left,
        _ => false,
    };
    let dc = dc4(&above, &left, near);
    for y in 0..4 {
        for x in 0..4 {
            plane[at + y * down + x] = if directional {
                sample4(mode, x, y, &above, &left, corner)
            } else {
                dc
            };
        }
    }
    true
}
