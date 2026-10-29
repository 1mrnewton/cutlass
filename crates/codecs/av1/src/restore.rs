//! Loop restoration from section 7.17 of the AV1 specification.
//!
//! Wiener, self-guided, and switchable units are read at the start of each
//! superblock and applied after CDEF, on the upscaled picture.

use cutlass_decode::CodecError;

use crate::header::{FrameInfo, RESTORE_NONE, RESTORE_SGRPROJ, RESTORE_SWITCHABLE, RESTORE_WIENER};
use crate::symbol::SymbolDecoder;

const SGR_PARAMS: [[i32; 4]; 16] = [
    [2, 12, 1, 4],
    [2, 15, 1, 6],
    [2, 18, 1, 8],
    [2, 21, 1, 9],
    [2, 24, 1, 10],
    [2, 29, 1, 11],
    [2, 36, 1, 12],
    [2, 45, 1, 13],
    [2, 56, 1, 14],
    [2, 68, 1, 15],
    [0, 0, 1, 5],
    [0, 0, 1, 8],
    [0, 0, 1, 11],
    [0, 0, 1, 14],
    [2, 30, 0, 0],
    [2, 75, 0, 0],
];

const XQD_MIN: [i32; 2] = [-96, -32];
const XQD_MAX: [i32; 2] = [31, 95];
const XQD_REF: [i32; 2] = [-32, 31];
const WIENER_MIN: [i32; 3] = [-5, -23, -17];
const WIENER_MAX: [i32; 3] = [10, 8, 46];
const WIENER_K: [i32; 3] = [1, 2, 3];
const WIENER_MID: [i32; 3] = [3, -7, 15];
const PRJ_BITS: i32 = 7;
const RST_BITS: i32 = 4;
const MTABLE_BITS: i32 = 20;
const RECIP_BITS: i32 = 12;
const SGR_BITS: i32 = 8;

/// One restoration unit after the tile symbols for it have been read.
#[derive(Clone, Copy)]
pub struct Unit {
    pub kind: u8,
    pub set: u8,
    pub xqd: [i32; 2],
    /// Vertical coefficients, then horizontal. Each pass stores the three
    /// signaled taps. The center tap is derived so the seven taps sum to 128.
    pub coeff: [[i32; 3]; 2],
}

/// Restoration units for the planes that use the self-guided filter.
pub struct Grid {
    cols: [usize; 3],
    rows: [usize; 3],
    unit_size: [usize; 3],
    units: Vec<Unit>,
    reference: [[i32; 2]; 3],
    wiener_ref: [[[i32; 3]; 2]; 3],
    active: bool,
}

impl Grid {
    pub fn new(frame: &FrameInfo) -> Result<Self, CodecError> {
        let mut cols = [0usize; 3];
        let mut rows = [0usize; 3];
        let mut unit_size = [0usize; 3];
        let mut count = 0usize;
        for plane in 0..3 {
            let kind = frame.restoration.plane[plane];
            if kind == RESTORE_NONE {
                continue;
            }
            if kind != RESTORE_SGRPROJ && kind != RESTORE_WIENER && kind != RESTORE_SWITCHABLE {
                return Err(CodecError::Unsupported {
                    what: "loop restoration",
                });
            }
            let sub = usize::from(plane > 0);
            let width = (frame.upscaled_width as usize).div_ceil(1 << sub);
            let height = (frame.height as usize).div_ceil(1 << sub);
            let size = usize::from(frame.restoration.unit[plane]).max(1);
            cols[plane] = count_units(size, width);
            rows[plane] = count_units(size, height);
            unit_size[plane] = size;
            count = count.saturating_add(cols[plane].saturating_mul(rows[plane]));
        }
        Ok(Self {
            cols,
            rows,
            unit_size,
            units: vec![
                Unit {
                    kind: RESTORE_NONE,
                    set: 0,
                    xqd: [0; 2],
                    coeff: [[0; 3]; 2],
                };
                count
            ],
            reference: [XQD_REF; 3],
            wiener_ref: [[WIENER_MID; 2]; 3],
            active: count > 0,
        })
    }

    pub fn active(&self) -> bool {
        self.active
    }

    fn index(&self, plane: usize, row: usize, col: usize) -> Option<usize> {
        if plane >= 3 || row >= self.rows[plane] || col >= self.cols[plane] {
            return None;
        }
        let mut base = 0usize;
        for earlier in 0..plane {
            base = base.saturating_add(self.cols[earlier].saturating_mul(self.rows[earlier]));
        }
        Some(base + row * self.cols[plane] + col)
    }
}

fn count_units(unit: usize, frame: usize) -> usize {
    (frame.saturating_add(unit >> 1) / unit.max(1)).max(1)
}

/// Read the restoration units covered by one superblock.
pub fn read_superblock(
    symbols: &mut SymbolDecoder<'_>,
    update: bool,
    frame: &FrameInfo,
    grid: &mut Grid,
    row: usize,
    col: usize,
    blocks: usize,
) -> Result<(), CodecError> {
    if !grid.active {
        return Ok(());
    }
    for plane in 0..3 {
        let kind = frame.restoration.plane[plane];
        if kind != RESTORE_SGRPROJ && kind != RESTORE_WIENER && kind != RESTORE_SWITCHABLE {
            continue;
        }
        let sub = usize::from(plane > 0);
        let unit = grid.unit_size[plane].max(1);
        let row_start = (row * (4 >> sub)).div_ceil(unit);
        let row_end = grid.rows[plane].min(((row + blocks) * (4 >> sub)).div_ceil(unit));
        let col_start = (col * (4 >> sub)).div_ceil(unit);
        let col_end = grid.cols[plane].min(((col + blocks) * (4 >> sub)).div_ceil(unit));
        for unit_row in row_start..row_end {
            for unit_col in col_start..col_end {
                let decoded = match kind {
                    RESTORE_WIENER => {
                        read_wiener_unit(symbols, update, plane > 0, &mut grid.wiener_ref[plane])?
                    }
                    RESTORE_SGRPROJ => read_sgr_unit(symbols, update, &mut grid.reference[plane])?,
                    _ => read_switchable_unit(
                        symbols,
                        update,
                        plane > 0,
                        &mut grid.wiener_ref[plane],
                        &mut grid.reference[plane],
                    )?,
                };
                if let Some(slot) = grid.index(plane, unit_row, unit_col) {
                    grid.units[slot] = decoded;
                }
            }
        }
    }
    Ok(())
}

fn read_wiener_unit(
    symbols: &mut SymbolDecoder<'_>,
    update: bool,
    chroma: bool,
    reference: &mut [[i32; 3]; 2],
) -> Result<Unit, CodecError> {
    let mut cdf = [11570u16, 32768, 0];
    if symbols.read(&mut cdf, update)? == 0 {
        return Ok(empty_unit());
    }
    read_wiener_coeffs(symbols, chroma, reference)
}

fn read_switchable_unit(
    symbols: &mut SymbolDecoder<'_>,
    update: bool,
    chroma: bool,
    wiener: &mut [[i32; 3]; 2],
    sgr: &mut [i32; 2],
) -> Result<Unit, CodecError> {
    let mut cdf = [9413u16, 22581, 32768, 0];
    match symbols.read(&mut cdf, update)? {
        1 => read_wiener_coeffs(symbols, chroma, wiener),
        2 => read_sgr_coeffs(symbols, sgr),
        _ => Ok(empty_unit()),
    }
}

fn empty_unit() -> Unit {
    Unit {
        kind: RESTORE_NONE,
        set: 0,
        xqd: [0; 2],
        coeff: [[0; 3]; 2],
    }
}

fn read_wiener_coeffs(
    symbols: &mut SymbolDecoder<'_>,
    chroma: bool,
    reference: &mut [[i32; 3]; 2],
) -> Result<Unit, CodecError> {
    let mut coeff = [[0i32; 3]; 2];
    for pass in 0..2 {
        let first = usize::from(chroma);
        for index in first..3 {
            let value = signed_subexp(
                symbols,
                WIENER_MIN[index],
                WIENER_MAX[index] + 1,
                WIENER_K[index],
                reference[pass][index],
            )?;
            coeff[pass][index] = value;
            reference[pass][index] = value;
        }
    }
    Ok(Unit {
        kind: RESTORE_WIENER,
        set: 0,
        xqd: [0; 2],
        coeff,
    })
}

fn read_sgr_unit(
    symbols: &mut SymbolDecoder<'_>,
    update: bool,
    reference: &mut [i32; 2],
) -> Result<Unit, CodecError> {
    let mut cdf = [16855u16, 32768, 0];
    if symbols.read(&mut cdf, update)? == 0 {
        return Ok(empty_unit());
    }
    read_sgr_coeffs(symbols, reference)
}

fn read_sgr_coeffs(
    symbols: &mut SymbolDecoder<'_>,
    reference: &mut [i32; 2],
) -> Result<Unit, CodecError> {
    let set = symbols.read_literal(4)? as u8;
    let mut xqd = [0i32; 2];
    for pass in 0..2 {
        let radius = SGR_PARAMS[usize::from(set.min(15))][pass * 2];
        let value = if radius == 0 {
            if pass == 1 {
                clip3(XQD_MIN[1], XQD_MAX[1], (1 << PRJ_BITS) - reference[0])
            } else {
                0
            }
        } else {
            signed_subexp(
                symbols,
                XQD_MIN[pass],
                XQD_MAX[pass] + 1,
                4,
                reference[pass],
            )?
        };
        xqd[pass] = value;
        reference[pass] = value;
    }
    Ok(Unit {
        kind: RESTORE_SGRPROJ,
        set,
        xqd,
        coeff: [[0; 3]; 2],
    })
}

/// Seven symmetric taps. The signaled values occupy both ends, and the center
/// keeps a unit DC gain.
pub fn wiener_taps(coeff: [i32; 3]) -> [i32; 7] {
    let mut filter = [0i32; 7];
    filter[3] = 128;
    for (index, tap) in coeff.iter().enumerate() {
        filter[index] = *tap;
        filter[6 - index] = *tap;
        filter[3] -= 2 * *tap;
    }
    filter
}

fn signed_subexp(
    symbols: &mut SymbolDecoder<'_>,
    low: i32,
    high: i32,
    k: i32,
    reference: i32,
) -> Result<i32, CodecError> {
    let mx = high - low;
    let recentered = unsigned_subexp(symbols, mx, k, reference - low)?;
    Ok(recentered + low)
}

fn unsigned_subexp(
    symbols: &mut SymbolDecoder<'_>,
    mx: i32,
    k: i32,
    reference: i32,
) -> Result<i32, CodecError> {
    let value = subexp(symbols, mx, k)?;
    if (reference << 1) <= mx {
        Ok(inverse_recenter(reference, value))
    } else {
        Ok(mx - 1 - inverse_recenter(mx - 1 - reference, value))
    }
}

fn subexp(symbols: &mut SymbolDecoder<'_>, num_syms: i32, k: i32) -> Result<i32, CodecError> {
    let mut i = 0i32;
    let mut mk = 0i32;
    loop {
        let bits = if i == 0 { k } else { k + i - 1 };
        let a = 1i32 << bits.max(0);
        if num_syms <= mk + 3 * a {
            return Ok(ns(symbols, (num_syms - mk).max(0) as u32)? as i32 + mk);
        }
        if symbols.read_bool()? {
            i += 1;
            mk += a;
        } else {
            return Ok(symbols.read_literal(bits.max(0) as u32)? as i32 + mk);
        }
    }
}

fn ns(symbols: &mut SymbolDecoder<'_>, n: u32) -> Result<u32, CodecError> {
    if n <= 1 {
        return Ok(0);
    }
    let w = u32::ilog2(n) + 1;
    let m = (1u32 << w) - n;
    let v = symbols.read_literal(w - 1)?;
    if v < m {
        return Ok(v);
    }
    let extra = symbols.read_literal(1)?;
    Ok((v << 1) - m + extra)
}

fn inverse_recenter(reference: i32, value: i32) -> i32 {
    if value > 2 * reference {
        value
    } else if value & 1 == 1 {
        reference - ((value + 1) >> 1)
    } else {
        reference + (value >> 1)
    }
}

fn clip3(min: i32, max: i32, value: i32) -> i32 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

/// Filter one plane. `cdef` is the picture after CDEF. `before` is the picture
/// before CDEF, used for the two rows outside a restoration stripe.
pub fn apply_sgr(
    cdef: &[u8],
    before: &[u8],
    width: usize,
    height: usize,
    grid: &Grid,
    plane: usize,
) -> Vec<u8> {
    let mut out = cdef.to_vec();
    if width == 0 || height == 0 || plane >= 3 || grid.rows[plane] == 0 {
        return out;
    }
    let mut y = 0usize;
    while y < height {
        let stripe = (y + 8) / 64;
        let start = stripe as i32 * 64 - 8;
        let end = start + 63;
        let last = (end.max(0) as usize).min(height - 1);
        if last < y {
            break;
        }
        filter_rows(
            &mut Rows {
                out: &mut out,
                cdef,
                before,
                width,
                grid,
                plane,
            },
            Span {
                y0: y,
                y1: last,
                stripe_start: start,
                stripe_end: end,
            },
        );
        y = last + 1;
    }
    out
}

struct Rows<'a> {
    out: &'a mut [u8],
    cdef: &'a [u8],
    before: &'a [u8],
    width: usize,
    grid: &'a Grid,
    plane: usize,
}

struct Span {
    y0: usize,
    y1: usize,
    stripe_start: i32,
    stripe_end: i32,
}

fn filter_rows(rows: &mut Rows<'_>, span: Span) {
    let width = rows.width;
    let mut x = 0usize;
    while x < width {
        let unit_col =
            (x / rows.grid.unit_size[rows.plane].max(1)).min(rows.grid.cols[rows.plane] - 1);
        let mut x1 = x + 1;
        while x1 < width {
            let next =
                (x1 / rows.grid.unit_size[rows.plane].max(1)).min(rows.grid.cols[rows.plane] - 1);
            if next != unit_col {
                break;
            }
            x1 += 1;
        }
        let unit_row =
            (span.y0 / rows.grid.unit_size[rows.plane].max(1)).min(rows.grid.rows[rows.plane] - 1);
        let Some(slot) = rows.grid.index(rows.plane, unit_row, unit_col) else {
            x = x1;
            continue;
        };
        let unit = rows.grid.units[slot];
        if unit.kind == RESTORE_SGRPROJ || unit.kind == RESTORE_WIENER {
            let source = Source {
                cdef: rows.cdef,
                before: rows.before,
                width,
                height: rows.cdef.len() / width.max(1),
                stripe_start: span.stripe_start,
                stripe_end: span.stripe_end,
            };
            let block = Block {
                x,
                y: span.y0,
                w: x1 - x,
                h: span.y1 - span.y0 + 1,
            };
            if unit.kind == RESTORE_WIENER {
                wiener_block(rows.out, &source, block, unit.coeff[0], unit.coeff[1]);
            } else {
                project(rows.out, &source, block, unit.set, unit.xqd);
            }
        }
        x = x1;
    }
}

struct Source<'a> {
    cdef: &'a [u8],
    before: &'a [u8],
    width: usize,
    height: usize,
    stripe_start: i32,
    stripe_end: i32,
}

impl Source<'_> {
    fn sample(&self, x: i32, y: i32) -> i32 {
        let last_x = self.width.saturating_sub(1) as i32;
        let last_y = self.height.saturating_sub(1) as i32;
        let x = x.clamp(0, last_x);
        let mut y = y.clamp(0, last_y);
        let plane = if y < self.stripe_start {
            y = y.max(self.stripe_start - 2);
            self.before
        } else if y > self.stripe_end {
            y = y.min(self.stripe_end + 2);
            self.before
        } else {
            self.cdef
        };
        let y = y.clamp(0, last_y) as usize;
        let x = x as usize;
        i32::from(*plane.get(y * self.width + x).unwrap_or(&0))
    }
}

struct Block {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
}

fn project(out: &mut [u8], source: &Source<'_>, block: Block, set: u8, xqd: [i32; 2]) {
    let params = SGR_PARAMS[usize::from(set.min(15))];
    let w0 = xqd[0];
    let w1 = xqd[1];
    let w2 = (1 << PRJ_BITS) - w0 - w1;
    for row in 0..block.h {
        for col in 0..block.w {
            let sample = source.sample((block.x + col) as i32, (block.y + row) as i32);
            let u = i64::from(sample) << RST_BITS;
            let mut v = i64::from(w1) * u;
            if params[0] != 0 {
                v += i64::from(w0)
                    * i64::from(box_value(
                        source,
                        block.x + col,
                        block.y + row,
                        row,
                        params[0],
                        params[1],
                        0,
                    ));
            } else {
                v += i64::from(w0) * u;
            }
            if params[2] != 0 {
                v += i64::from(w2)
                    * i64::from(box_value(
                        source,
                        block.x + col,
                        block.y + row,
                        row,
                        params[2],
                        params[3],
                        1,
                    ));
            } else {
                v += i64::from(w2) * u;
            }
            let rounded = round2(v, RST_BITS + PRJ_BITS);
            if let Some(slot) = out.get_mut((block.y + row) * source.width + block.x + col) {
                *slot = u8::try_from(rounded.clamp(0, 255)).unwrap_or(0);
            }
        }
    }
}

fn box_value(
    source: &Source<'_>,
    frame_x: usize,
    frame_y: usize,
    row: usize,
    radius: i32,
    eps: i32,
    pass: i32,
) -> i32 {
    let n = (2 * radius + 1) * (2 * radius + 1);
    let n2e = n * n * eps;
    if n2e == 0 {
        return 0;
    }
    let scale = ((1i64 << MTABLE_BITS) + i64::from(n2e) / 2) / i64::from(n2e);
    let one_over_n = ((1i32 << RECIP_BITS) + n / 2) / n;
    let mut a_sum = 0i64;
    let mut b_sum = 0i64;
    for dy in -1..=1 {
        for dx in -1..=1 {
            let weight = neighbor_weight(pass, row as i32, dy, dx);
            if weight == 0 {
                continue;
            }
            let (a2, b2) = box_ab(
                source,
                frame_x as i32 + dx,
                frame_y as i32 + dy,
                radius,
                n,
                scale,
                one_over_n,
            );
            a_sum += i64::from(weight) * i64::from(a2);
            b_sum += i64::from(weight) * i64::from(b2);
        }
    }
    let shift = if pass == 0 && (row & 1) == 1 { 4 } else { 5 };
    let sample = source.sample(frame_x as i32, frame_y as i32);
    round2(
        a_sum * i64::from(sample) + b_sum,
        SGR_BITS + shift - RST_BITS,
    ) as i32
}

fn neighbor_weight(pass: i32, row: i32, dy: i32, dx: i32) -> i32 {
    if pass == 0 {
        if (row + dy) & 1 == 1 {
            if dx == 0 { 6 } else { 5 }
        } else {
            0
        }
    } else if dx == 0 || dy == 0 {
        4
    } else {
        3
    }
}

fn box_ab(
    source: &Source<'_>,
    x: i32,
    y: i32,
    radius: i32,
    n: i32,
    s: i64,
    one_over_n: i32,
) -> (i32, i32) {
    let mut a = 0i64;
    let mut b = 0i64;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let sample = i64::from(source.sample(x + dx, y + dy));
            a += sample * sample;
            b += sample;
        }
    }
    let d = b;
    let p = (a * i64::from(n) - d * d).max(0);
    let z = round2(p * s, MTABLE_BITS);
    let a2 = if z >= 255 {
        256
    } else if z == 0 {
        1
    } else {
        ((z << SGR_BITS) + z / 2) / (z + 1)
    };
    let b2 = (i64::from(1i32 << SGR_BITS) - a2) * b * i64::from(one_over_n);
    (a2 as i32, round2(b2, RECIP_BITS) as i32)
}

/// Filter one plane with the Wiener restoration filter.
///
/// `cdef` is the picture after CDEF. `before` is the deblocked picture, used
/// for the two rows outside a restoration stripe. Pass 0 of `coeff` is the
/// vertical filter. Pass 1 is the horizontal filter, and it runs first.
pub fn apply_wiener(
    cdef: &[u8],
    before: &[u8],
    width: usize,
    height: usize,
    grid: &Grid,
    plane: usize,
) -> Vec<u8> {
    let mut out = cdef.to_vec();
    if width == 0 || height == 0 || plane >= 3 || grid.rows[plane] == 0 {
        return out;
    }
    let mut y = 0usize;
    while y < height {
        let stripe = (y + 8) / 64;
        let start = stripe as i32 * 64 - 8;
        let end = start + 63;
        let last = (end.max(0) as usize).min(height - 1);
        if last < y {
            break;
        }
        let mut x = 0usize;
        while x < width {
            let size = grid.unit_size[plane].max(1);
            let unit_col = (x / size).min(grid.cols[plane] - 1);
            let mut x1 = x + 1;
            while x1 < width && (x1 / size).min(grid.cols[plane] - 1) == unit_col {
                x1 += 1;
            }
            let unit_row = (y / size).min(grid.rows[plane] - 1);
            if let Some(slot) = grid.index(plane, unit_row, unit_col) {
                let unit = grid.units[slot];
                if unit.kind == RESTORE_WIENER {
                    let source = Source {
                        cdef,
                        before,
                        width,
                        height: cdef.len() / width.max(1),
                        stripe_start: start,
                        stripe_end: end,
                    };
                    wiener_block(
                        &mut out,
                        &source,
                        Block {
                            x,
                            y,
                            w: x1 - x,
                            h: last - y + 1,
                        },
                        unit.coeff[0],
                        unit.coeff[1],
                    );
                }
            }
            x = x1;
        }
        y = last + 1;
    }
    out
}

fn wiener_block(
    out: &mut [u8],
    source: &Source<'_>,
    block: Block,
    vertical: [i32; 3],
    horizontal: [i32; 3],
) {
    let hfilter = wiener_taps(horizontal);
    let vfilter = wiener_taps(vertical);
    let rows = block.h + 6;
    let mut intermediate = vec![0i32; rows.saturating_mul(block.w)];
    for row in 0..rows {
        for col in 0..block.w {
            let mut sum = 0i32;
            for (tap_index, tap) in hfilter.iter().enumerate() {
                let sample = source.sample(
                    block.x as i32 + col as i32 + tap_index as i32 - 3,
                    block.y as i32 + row as i32 - 3,
                );
                sum += *tap * sample;
            }
            let rounded = round2(i64::from(sum), 3) as i32;
            if let Some(slot) = intermediate.get_mut(row * block.w + col) {
                *slot = clip3(-2048, 6143, rounded);
            }
        }
    }
    for row in 0..block.h {
        for col in 0..block.w {
            let mut sum = 0i32;
            for (tap_index, tap) in vfilter.iter().enumerate() {
                let above = intermediate
                    .get((row + tap_index) * block.w + col)
                    .copied()
                    .unwrap_or(0);
                sum += *tap * above;
            }
            let rounded = round2(i64::from(sum), 11);
            if let Some(slot) = out.get_mut((block.y + row) * source.width + block.x + col) {
                *slot = u8::try_from(rounded.clamp(0, 255)).unwrap_or(0);
            }
        }
    }
}

fn round2(value: i64, bits: i32) -> i64 {
    if bits <= 0 {
        return value;
    }
    (value + (1i64 << (bits - 1))) >> bits
}

#[cfg(test)]
mod tests {
    use super::{
        Grid, RESTORE_WIENER, SGR_PARAMS, Unit, WIENER_MID, XQD_REF, apply_sgr, apply_wiener,
        wiener_taps,
    };

    #[test]
    fn every_self_guided_set_has_four_parameters() {
        assert_eq!(SGR_PARAMS.len(), 16);
        assert_eq!(SGR_PARAMS[0], [2, 12, 1, 4]);
        assert_eq!(SGR_PARAMS[15], [2, 75, 0, 0]);
    }

    #[test]
    fn a_constant_plane_stays_constant() {
        let width = 16usize;
        let height = 8usize;
        let samples = vec![40u8; width * height];
        let grid = Grid {
            cols: [1, 0, 0],
            rows: [1, 0, 0],
            unit_size: [256, 0, 0],
            units: vec![Unit {
                kind: super::RESTORE_SGRPROJ,
                set: 0,
                xqd: [-32, 31],
                coeff: [[0; 3]; 2],
            }],
            reference: [XQD_REF; 3],
            wiener_ref: [[WIENER_MID; 2]; 3],
            active: true,
        };
        let out = apply_sgr(&samples, &samples, width, height, &grid, 0);
        assert!(out.iter().all(|sample| *sample == 40));
    }

    #[test]
    fn wiener_taps_keep_unit_gain() {
        assert_eq!(wiener_taps([0, 0, 0]), [0, 0, 0, 128, 0, 0, 0]);
        let taps = wiener_taps([3, -7, 15]);
        assert_eq!(taps.iter().sum::<i32>(), 128);
        assert_eq!(taps[0], 3);
        assert_eq!(taps[6], 3);
    }

    #[test]
    fn a_constant_plane_stays_constant_under_wiener() {
        let width = 16usize;
        let height = 8usize;
        let samples = vec![40u8; width * height];
        let grid = Grid {
            cols: [1, 0, 0],
            rows: [1, 0, 0],
            unit_size: [256, 0, 0],
            units: vec![Unit {
                kind: RESTORE_WIENER,
                set: 0,
                xqd: [0; 2],
                coeff: [[3, -7, 15], [3, -7, 15]],
            }],
            reference: [XQD_REF; 3],
            wiener_ref: [[WIENER_MID; 2]; 3],
            active: true,
        };
        let out = apply_wiener(&samples, &samples, width, height, &grid, 0);
        assert!(out.iter().all(|sample| *sample == 40));
    }
}
