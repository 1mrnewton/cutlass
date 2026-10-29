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

pub struct Picture<'a> {
    pub y: &'a mut [u8],
    pub y_w: usize,
    pub y_h: usize,
    pub u: &'a mut [u8],
    pub v: &'a mut [u8],
    pub c_w: usize,
    pub c_h: usize,
}

/// Loop-filter every plane on a shared set of workers.
///
/// Vertical edges finish before horizontal edges inside each plane. Row bands
/// of one plane run together, and the chroma planes run beside luma, so the
/// cores that finished a tile pick up filter rows instead of waiting.
pub fn deblock_picture(picture: Picture<'_>, edges: &Edges<'_>) {
    if edges.frame.lf_level[0] == 0 && edges.frame.lf_level[1] == 0 {
        return;
    }
    let Picture {
        y,
        y_w,
        y_h,
        u,
        v,
        c_w,
        c_h,
    } = picture;
    let cores = parallelism(edges.mi_rows);
    let luma_workers = (cores * 2 / 3).max(1);
    let chroma_workers = ((cores - luma_workers) / 2).max(1);
    std::thread::scope(|scope| {
        scope.spawn(|| filter_plane(y, y_w, y_h, 0, edges, luma_workers));
        scope.spawn(|| filter_plane(u, c_w, c_h, 1, edges, chroma_workers));
        scope.spawn(|| filter_plane(v, c_w, c_h, 2, edges, chroma_workers));
    });
}

fn filter_plane(
    samples: &mut [u8],
    width: usize,
    height: usize,
    plane: usize,
    edges: &Edges<'_>,
    workers: usize,
) {
    if edges.frame.lf_level[0] == 0 && edges.frame.lf_level[1] == 0 {
        return;
    }
    let sub_x = usize::from(plane > 0 && edges.sub_x);
    let sub_y = usize::from(plane > 0 && edges.sub_y);
    if plane > 0 && edges.frame.lf_level[plane + 1] == 0 {
        return;
    }
    if width == 0 || height == 0 || edges.mi_cols == 0 || edges.mi_rows == 0 {
        return;
    }
    let row_step = 1usize << sub_y;
    let col_step = 1usize << sub_x;
    for pass in 0..2 {
        let lvl = level(edges.frame, plane, pass);
        if lvl == 0 {
            continue;
        }
        let job = Job {
            width,
            plane_h: height,
            plane,
            pass,
            edges,
            sub_x,
            sub_y,
            col_step,
            lvl,
        };
        if pass == 0 {
            run_bands(
                samples,
                &vertical_bands(edges, sub_y, height, row_step, workers),
                &job,
            );
        } else {
            // Horizontal taps overlap adjacent rows. Edges 16 pixels apart do not.
            for phase in [0usize, 4, 8, 12] {
                run_bands(
                    samples,
                    &horizontal_bands(edges, sub_y, height, row_step, phase, workers),
                    &job,
                );
            }
        }
    }
}

struct Job<'a> {
    width: usize,
    plane_h: usize,
    plane: usize,
    pass: usize,
    edges: &'a Edges<'a>,
    sub_x: usize,
    sub_y: usize,
    col_step: usize,
    lvl: i32,
}

#[derive(Clone, Copy)]
struct Band {
    y0: usize,
    y1: usize,
    row0: usize,
    row1: usize,
    row_step: usize,
}

fn parallelism(rows: usize) -> usize {
    if rows < 16 {
        return 1;
    }
    let cores = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    cores.max(1)
}

fn pixel_row(mi_row: usize, sub_y: usize) -> usize {
    (mi_row.saturating_mul(4)) >> sub_y
}

fn vertical_bands(
    edges: &Edges<'_>,
    sub_y: usize,
    plane_h: usize,
    row_step: usize,
    workers: usize,
) -> Vec<Band> {
    let workers = workers.max(1).min(edges.mi_rows.max(1));
    let rows_per = edges.mi_rows.div_ceil(workers);
    let rows_per = rows_per.div_ceil(row_step).max(1).saturating_mul(row_step);
    let mut bands = Vec::new();
    let mut row = 0;
    while row < edges.mi_rows {
        let row1 = (row + rows_per).min(edges.mi_rows);
        let y0 = pixel_row(row, sub_y).min(plane_h);
        let y1 = pixel_row(row1, sub_y).min(plane_h);
        if y1 > y0 {
            bands.push(Band {
                y0,
                y1,
                row0: row,
                row1,
                row_step,
            });
        }
        row = row1;
    }
    bands
}

fn horizontal_bands(
    edges: &Edges<'_>,
    sub_y: usize,
    plane_h: usize,
    row_step: usize,
    phase: usize,
    workers: usize,
) -> Vec<Band> {
    let wave_step = 4usize << sub_y;
    let mut rows = Vec::new();
    let mut row = 0;
    while row < edges.mi_rows {
        let y = pixel_row(row, sub_y);
        if y < plane_h && y % 16 == phase {
            rows.push(row);
        }
        row += row_step;
    }
    if rows.is_empty() {
        return Vec::new();
    }
    let workers = workers.max(1).min(rows.len());
    let per = rows.len().div_ceil(workers);
    let mut bands = Vec::new();
    let mut start = 0;
    while start < rows.len() {
        let end = (start + per).min(rows.len());
        let first = rows[start];
        let last = rows[end - 1];
        let y_first = pixel_row(first, sub_y);
        let y_last = pixel_row(last, sub_y);
        let y0 = y_first.saturating_sub(7);
        let y1 = (y_last + 7).min(plane_h);
        if y1 > y0 {
            bands.push(Band {
                y0,
                y1,
                row0: first,
                row1: last.saturating_add(1),
                row_step: wave_step,
            });
        }
        start = end;
    }
    bands
}

fn run_bands(samples: &mut [u8], bands: &[Band], job: &Job<'_>) {
    if bands.is_empty() {
        return;
    }
    // One band, or windows that share pixels, filters against the whole plane.
    // Coordinates stay absolute. A parallel band later subtracts its origin.
    if bands.len() == 1 || !sliceable(bands, job.width, samples.len()) {
        for band in bands {
            filter_rows(samples, 0, job.plane_h, band, job);
        }
        return;
    }
    let width = job.width;
    std::thread::scope(|scope| {
        let mut rest = &mut *samples;
        let mut cursor = 0usize;
        for band in bands {
            let start = band.y0 * width;
            let end = band.y1 * width;
            let skip = start - cursor;
            let (_, tail) = rest.split_at_mut(skip);
            let (chunk, tail) = tail.split_at_mut(end - start);
            let band = *band;
            scope.spawn(move || filter_rows(chunk, band.y0, band.y1 - band.y0, &band, job));
            rest = tail;
            cursor = end;
        }
    });
}

fn sliceable(bands: &[Band], width: usize, len: usize) -> bool {
    if !disjoint(bands) {
        return false;
    }
    let mut cursor = 0usize;
    for band in bands {
        let Some(start) = band.y0.checked_mul(width) else {
            return false;
        };
        let Some(end) = band.y1.checked_mul(width) else {
            return false;
        };
        if start < cursor || end < start || end > len {
            return false;
        }
        cursor = end;
    }
    true
}

fn disjoint(bands: &[Band]) -> bool {
    let mut cursor = 0usize;
    for band in bands {
        if band.y0 < cursor || band.y1 <= band.y0 {
            return false;
        }
        cursor = band.y1;
    }
    true
}

fn filter_rows(samples: &mut [u8], band_y: usize, band_h: usize, band: &Band, job: &Job<'_>) {
    let mut row = band.row0;
    while row < band.row1 && row < job.edges.mi_rows {
        let mut col = 0;
        while col < job.edges.mi_cols {
            filter_edge(
                samples,
                job,
                band_y,
                band_h,
                Mi {
                    row,
                    col,
                    sub_x: job.sub_x,
                    sub_y: job.sub_y,
                },
            );
            col += column_advance(job, row, col);
        }
        row += band.row_step;
    }
}

/// Vertical edges exist on transform boundaries. Interior columns of a wide
/// transform have nothing to filter, so the scan jumps to the next boundary.
fn column_advance(job: &Job<'_>, row: usize, col: usize) -> usize {
    if job.pass != 0 || row >= job.edges.mi_rows || col >= job.edges.mi_cols {
        return job.col_step;
    }
    let tx = tx_at(job.edges, job.plane, row, col);
    let span = TX_WIDTH.get(tx).copied().unwrap_or(4).max(4);
    let x_p = (col.saturating_mul(4)) >> job.sub_x;
    let phase = x_p % span;
    let remaining = span - phase;
    ((remaining << job.sub_x) / 4).max(job.col_step)
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

fn filter_edge(samples: &mut [u8], job: &Job<'_>, band_y: usize, band_h: usize, mi: Mi) {
    let Mi {
        mut row,
        mut col,
        sub_x,
        sub_y,
    } = mi;
    let (dx, dy) = if job.pass == 0 { (1i32, 0i32) } else { (0, 1) };
    let x = col * 4;
    let y = row * 4;
    row |= sub_y;
    col |= sub_x;
    if !on_screen(x, y, job.width << sub_x, job.plane_h << sub_y, job.pass) {
        return;
    }
    if row >= job.edges.mi_rows || col >= job.edges.mi_cols {
        return;
    }
    let x_p = x >> sub_x;
    let y_p = y >> sub_y;
    let prev_row = row.wrapping_sub((dy as usize) << sub_y);
    let prev_col = col.wrapping_sub((dx as usize) << sub_x);
    if prev_row >= job.edges.mi_rows || prev_col >= job.edges.mi_cols {
        return;
    }
    let tx = tx_at(job.edges, job.plane, row, col);
    let prev_tx = tx_at(job.edges, job.plane, prev_row, prev_col);
    let tx_edge = if job.pass == 0 {
        x_p.is_multiple_of(TX_WIDTH[tx])
    } else {
        y_p.is_multiple_of(TX_HEIGHT[tx])
    };
    // Intra frames filter every transform edge. Inter prediction boundaries are not used here.
    if !tx_edge {
        return;
    }
    let filter_size = filter_size(job.pass, job.plane, tx, prev_tx);
    let (limit, blimit, thresh) = limits(job.lvl, job.edges.frame.lf_sharpness);
    let strength = Strength {
        limit,
        blimit,
        thresh,
    };
    let y_base = band_y as i32;
    for i in 0..4 {
        let axis = Axis {
            x: x_p as i32 + dy * i,
            y: y_p as i32 + dx * i - y_base,
            dx,
            dy,
        };
        filter_sample(
            samples,
            job.width,
            band_h,
            axis,
            strength,
            filter_size,
            job.plane,
        );
    }
}

/// Pixels on one side of an edge. Index `origin` is `q0`; `origin + p` is the
/// sample `p` steps into the block.
struct Line {
    values: [i32; 16],
    origin: i32,
    from: i32,
    to: i32,
}

fn filter_window(filter_size: usize, plane: usize) -> i32 {
    if filter_size == 4 {
        2
    } else if plane != 0 {
        3
    } else if filter_size == 8 {
        4
    } else {
        7
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
    let radius = filter_window(filter_size, plane);
    let Some(mut line) = load_line(samples, width, height, axis, radius) else {
        return;
    };
    let mask = masks(&line, strength, filter_size, plane);
    if !mask.filter {
        return;
    }
    if filter_size == 4 || !mask.flat {
        narrow(&mut line, mask.hev);
    } else if filter_size == 8 || !mask.flat2 {
        wide(&mut line, 3, plane);
    } else {
        wide(&mut line, 4, plane);
    }
    store_line(samples, width, axis, &line);
}

struct Mask {
    hev: bool,
    filter: bool,
    flat: bool,
    flat2: bool,
}

fn load_line(samples: &[u8], width: usize, height: usize, axis: Axis, radius: i32) -> Option<Line> {
    let origin = radius;
    let count = (radius * 2) as usize;
    if count > 16 || radius <= 0 {
        return None;
    }
    if let Some(values) = load_window(samples, width, height, axis, radius, count) {
        return Some(Line {
            values,
            origin,
            from: -radius,
            to: radius,
        });
    }
    let mut values = [0i32; 16];
    let mut p = -radius;
    while p < radius {
        values[(origin + p) as usize] = sample(
            samples,
            width,
            height,
            axis.x + p * axis.dx,
            axis.y + p * axis.dy,
        )?;
        p += 1;
    }
    Some(Line {
        values,
        origin,
        from: -radius,
        to: radius,
    })
}

/// Copy a window that sits fully inside the plane. The edge filter is the hot
/// path, and the per-sample bounds check is the same answer for every tap.
fn load_window(
    samples: &[u8],
    width: usize,
    height: usize,
    axis: Axis,
    radius: i32,
    count: usize,
) -> Option<[i32; 16]> {
    if axis.dx == 1 && axis.dy == 0 {
        let y = usize::try_from(axis.y).ok()?;
        let x0 = usize::try_from(axis.x - radius).ok()?;
        if y >= height || x0.saturating_add(count) > width {
            return None;
        }
        let start = y.saturating_mul(width).saturating_add(x0);
        let row = samples.get(start..start + count)?;
        let mut values = [0i32; 16];
        for (slot, sample) in values.iter_mut().zip(row) {
            *slot = i32::from(*sample);
        }
        return Some(values);
    }
    if axis.dx == 0 && axis.dy == 1 {
        let x = usize::try_from(axis.x).ok()?;
        let y0 = usize::try_from(axis.y - radius).ok()?;
        if x >= width || y0.saturating_add(count) > height {
            return None;
        }
        let mut values = [0i32; 16];
        for (i, slot) in values.iter_mut().take(count).enumerate() {
            let at = (y0 + i).saturating_mul(width).saturating_add(x);
            *slot = i32::from(*samples.get(at)?);
        }
        return Some(values);
    }
    None
}

fn store_line(samples: &mut [u8], width: usize, axis: Axis, line: &Line) {
    let count = (line.to - line.from) as usize;
    if store_window(samples, width, axis, line, count) {
        return;
    }
    let mut p = line.from;
    while p < line.to {
        store(
            samples,
            width,
            axis.x + p * axis.dx,
            axis.y + p * axis.dy,
            line.values[(line.origin + p) as usize],
        );
        p += 1;
    }
}

fn store_window(samples: &mut [u8], width: usize, axis: Axis, line: &Line, count: usize) -> bool {
    if count > 16 {
        return false;
    }
    if axis.dx == 1 && axis.dy == 0 {
        let Ok(y) = usize::try_from(axis.y) else {
            return false;
        };
        let Ok(x0) = usize::try_from(axis.x + line.from * axis.dx) else {
            return false;
        };
        let start = y.saturating_mul(width).saturating_add(x0);
        let Some(row) = samples.get_mut(start..start + count) else {
            return false;
        };
        for (slot, p) in row.iter_mut().zip(line.from..line.to) {
            let value = line.values[(line.origin + p) as usize];
            *slot = value.clamp(0, 255) as u8;
        }
        return true;
    }
    if axis.dx == 0 && axis.dy == 1 {
        let Ok(x) = usize::try_from(axis.x) else {
            return false;
        };
        let Ok(y0) = usize::try_from(axis.y + line.from) else {
            return false;
        };
        if x >= width || count == 0 {
            return false;
        }
        for (i, p) in (line.from..line.to).enumerate() {
            let at = (y0 + i).saturating_mul(width).saturating_add(x);
            let Some(slot) = samples.get_mut(at) else {
                return false;
            };
            let value = line.values[(line.origin + p) as usize];
            *slot = value.clamp(0, 255) as u8;
        }
        return true;
    }
    false
}

fn at(line: &Line, p: i32) -> i32 {
    line.values
        .get((line.origin + p) as usize)
        .copied()
        .unwrap_or(0)
}

fn put(line: &mut Line, p: i32, value: i32) {
    if let Some(slot) = line.values.get_mut((line.origin + p) as usize) {
        *slot = value;
    }
}

fn masks(line: &Line, strength: Strength, filter_size: usize, plane: usize) -> Mask {
    let Strength {
        limit,
        blimit,
        thresh,
    } = strength;
    let q0 = at(line, 0);
    let q1 = at(line, 1);
    let p0 = at(line, -1);
    let p1 = at(line, -2);
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
        let p2 = at(line, -3);
        let q2 = at(line, 2);
        mask |= (p2 - p1).abs() > limit || (q2 - q1).abs() > limit;
    }
    if filter_len >= 8 {
        let p2 = at(line, -3);
        let q2 = at(line, 2);
        let p3 = at(line, -4);
        let q3 = at(line, 3);
        mask |= (p3 - p2).abs() > limit || (q3 - q2).abs() > limit;
    }
    let flat = if filter_size >= 8 {
        let p2 = at(line, -3);
        let q2 = at(line, 2);
        let mut flat_mask = (p1 - p0).abs() > 1 || (q1 - q0).abs() > 1;
        flat_mask |= (p2 - p0).abs() > 1 || (q2 - q0).abs() > 1;
        if filter_len >= 8 {
            let p3 = at(line, -4);
            let q3 = at(line, 3);
            flat_mask |= (p3 - p0).abs() > 1 || (q3 - q0).abs() > 1;
        }
        !flat_mask
    } else {
        false
    };
    let flat2 = if filter_size >= 16 {
        let mut flat_mask = false;
        for p in [4, 5, 6] {
            flat_mask |= (at(line, -1 - p) - p0).abs() > 1 || (at(line, p) - q0).abs() > 1;
        }
        !flat_mask
    } else {
        false
    };
    Mask {
        hev,
        filter: !mask,
        flat,
        flat2,
    }
}

fn narrow(line: &mut Line, hev: bool) {
    let q0 = at(line, 0);
    let q1 = at(line, 1);
    let p0 = at(line, -1);
    let p1 = at(line, -2);
    let ps1 = p1 - 128;
    let ps0 = p0 - 128;
    let qs0 = q0 - 128;
    let qs1 = q1 - 128;
    let mut filter = if hev { clamp4(ps1 - qs1) } else { 0 };
    filter = clamp4(filter + 3 * (qs0 - ps0));
    let filter1 = clamp4(filter + 4) >> 3;
    let filter2 = clamp4(filter + 3) >> 3;
    put(line, 0, clamp4(qs0 - filter1) + 128);
    put(line, -1, clamp4(ps0 + filter2) + 128);
    if !hev {
        let filter = round2(filter1, 1);
        put(line, 1, clamp4(qs1 - filter) + 128);
        put(line, -2, clamp4(ps1 + filter) + 128);
    }
}

fn wide(line: &mut Line, log2_size: u32, plane: usize) {
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
            total += at(line, p) * tap;
        }
        filtered[(i + n) as usize] = round2(total, log2_size);
    }
    for i in -n..n {
        put(line, i, filtered[(i + n) as usize]);
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
