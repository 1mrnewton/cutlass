//! Film grain synthesis for the displayed picture.
//!
//! The model is Recommendation ITU-T H.274, equations 22 through 32, carried
//! in the H.264 film-grain SEI. Equation 31, the unitary matrix, is the
//! orthonormal inverse DCT: a column is one basis vector. The uniform source
//! for the Box-Muller transform is not specified, so this module draws it from
//! a fixed xorshift. The same picture and the same message always produce the
//! same samples.
//!
//! Grain is a display step. Callers apply it to a copy of the reconstructed
//! planes. The reference picture stays clean.

use crate::recon::Sample;

/// Seed for the xorshift32 sequence that feeds equation 28.
const SEED: u32 = 0xA341_316C;

/// One intensity interval and its six model values, with the missing values
/// already inferred.
#[derive(Clone, Debug)]
pub(crate) struct Interval {
    pub(crate) low: u8,
    pub(crate) high: u8,
    pub(crate) value: [i32; 6],
}

/// The intervals of one colour component.
#[derive(Clone, Debug)]
pub(crate) struct Component {
    pub(crate) intervals: Vec<Interval>,
}

/// A film-grain model that applies to one displayed picture.
#[derive(Clone, Debug)]
pub(crate) struct FilmGrain {
    /// 0 is frequency filtering. 1 is auto-regression.
    pub(crate) model: u8,
    /// 0 is additive. 1 is multiplicative.
    pub(crate) blend: u8,
    pub(crate) scale: u8,
    pub(crate) comp: [Option<Component>; 3],
}

struct Rng {
    state: u32,
}

impl Rng {
    fn new() -> Self {
        Self { state: SEED }
    }

    fn next(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    /// A uniform value in (0, 1).
    fn uniform(&mut self) -> f64 {
        (f64::from(self.next()) + 1.0) / (f64::from(u32::MAX) + 2.0)
    }

    fn gaussian(&mut self) -> f64 {
        let u0 = self.uniform();
        let u1 = self.uniform();
        let value = (-2.0 * u0.ln()).sqrt() * (2.0 * std::f64::consts::PI * u1).cos();
        if value.is_finite() { value } else { 0.0 }
    }
}

struct Prev {
    g: Vec<i32>,
    w: usize,
    h: usize,
}

impl Prev {
    fn none() -> Self {
        Self {
            g: Vec::new(),
            w: 0,
            h: 0,
        }
    }

    fn at(&self, x: usize, y: usize, w: usize, h: usize) -> i32 {
        if self.w != w || self.h != h {
            return 0;
        }
        self.g
            .get(y.saturating_mul(w).saturating_add(x))
            .copied()
            .unwrap_or(0)
    }
}

/// Blend `grain` onto the reconstructed planes. `chroma` is 1, 2, or 3.
pub(crate) fn apply<S: Sample>(
    y: &mut [S],
    u: &mut [S],
    v: &mut [S],
    width: usize,
    height: usize,
    chroma: u8,
    grain: &FilmGrain,
) {
    if width == 0 || height == 0 {
        return;
    }
    let mut rng = Rng::new();
    let prev = paint(&mut rng, y, width, height, 0, grain, &Prev::none());
    let (cw, ch) = chroma_size(chroma, width, height);
    if cw == 0 || ch == 0 {
        return;
    }
    let prev = paint(&mut rng, u, cw, ch, 1, grain, &prev);
    let _ = paint(&mut rng, v, cw, ch, 2, grain, &prev);
}

fn chroma_size(chroma: u8, width: usize, height: usize) -> (usize, usize) {
    match chroma {
        3 => (width, height),
        2 => (width / 2, height),
        _ => (width / 2, height / 2),
    }
}

fn paint<S: Sample>(
    rng: &mut Rng,
    plane: &mut [S],
    w: usize,
    h: usize,
    component: usize,
    grain: &FilmGrain,
    prev: &Prev,
) -> Prev {
    let Some(comp) = grain.comp.get(component).and_then(Option::as_ref) else {
        return Prev::none();
    };
    if plane.len() < w.saturating_mul(h) {
        return Prev::none();
    }
    let g = if grain.model == 0 {
        frequency(rng, plane, w, h, comp, prev, grain.scale)
    } else {
        autoreg(rng, plane, w, h, comp, prev, grain.scale)
    };
    blend(plane, &g, grain.blend);
    Prev { g, w, h }
}

fn frequency<S: Sample>(
    rng: &mut Rng,
    plane: &[S],
    w: usize,
    h: usize,
    comp: &Component,
    prev: &Prev,
    scale: u8,
) -> Vec<i32> {
    let matrix = dct_matrix();
    let mut g = vec![0; w.saturating_mul(h)];
    let shift = u32::from(scale).min(31);
    for ty in 0..h.div_ceil(16) {
        for tx in 0..w.div_ceil(16) {
            let raw = noise_tile(rng);
            let mut cache: Vec<(usize, [[i32; 16]; 16])> = Vec::new();
            let y0 = ty.saturating_mul(16);
            let x0 = tx.saturating_mul(16);
            let mut band = Band {
                raw: &raw,
                matrix: &matrix,
                cache: &mut cache,
                comp,
                prev,
            };
            write_tile(
                plane,
                &mut g,
                Tile {
                    x0,
                    y0,
                    w,
                    h,
                    shift,
                },
                &mut band,
            );
        }
    }
    g
}

fn noise_tile(rng: &mut Rng) -> [[f64; 16]; 16] {
    let mut raw = [[0.0; 16]; 16];
    for row in &mut raw {
        for sample in row {
            *sample = rng.gaussian();
        }
    }
    raw
}

#[derive(Clone, Copy)]
struct Tile {
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    shift: u32,
}

struct Band<'a> {
    raw: &'a [[f64; 16]; 16],
    matrix: &'a [[f64; 16]; 16],
    cache: &'a mut Vec<(usize, [[i32; 16]; 16])>,
    comp: &'a Component,
    prev: &'a Prev,
}

fn write_tile<S: Sample>(plane: &[S], g: &mut [i32], tile: Tile, band: &mut Band<'_>) {
    for by in 0..2 {
        for bx in 0..2 {
            let sy = tile.y0 + by * 8;
            let sx = tile.x0 + bx * 8;
            if sy >= tile.h || sx >= tile.w {
                continue;
            }
            let avg = average(plane, tile.w, tile.h, sx, sy);
            write_block(g, tile, Block { sx, sy, avg }, band);
        }
    }
}

struct Block {
    sx: usize,
    sy: usize,
    avg: i32,
}

fn write_block(g: &mut [i32], tile: Tile, block: Block, band: &mut Band<'_>) {
    for row in 0..8 {
        for col in 0..8 {
            let y = block.sy + row;
            let x = block.sx + col;
            if y >= tile.h || x >= tile.w {
                continue;
            }
            let local_y = y - tile.y0;
            let local_x = x - tile.x0;
            let hits = matching(band.comp, block.avg);
            let mut acc = 0i64;
            for index in hits {
                let value = band.comp.intervals[index].value;
                let q = filtered(band, index);
                let sample = q[local_y][local_x];
                let older = band.prev.at(x, y, tile.w, tile.h);
                acc += (i64::from(value[0]) * i64::from(sample)
                    + i64::from(value[5]) * i64::from(older))
                    >> tile.shift;
                g[y * tile.w + x] = narrow(acc);
            }
        }
    }
}

fn matching(comp: &Component, avg: i32) -> Vec<usize> {
    comp.intervals
        .iter()
        .enumerate()
        .filter(|(_, interval)| avg >= i32::from(interval.low) && avg <= i32::from(interval.high))
        .map(|(index, _)| index)
        .collect()
}

fn filtered(band: &mut Band<'_>, index: usize) -> [[i32; 16]; 16] {
    if let Some(pos) = band.cache.iter().position(|(got, _)| *got == index) {
        return band.cache[pos].1;
    }
    let interval = band.comp.intervals[index].clone();
    let q = idct(&bandpass(band.raw, &interval), band.matrix);
    band.cache.push((index, q));
    q
}

fn bandpass(raw: &[[f64; 16]; 16], interval: &Interval) -> [[f64; 16]; 16] {
    let mut z = *raw;
    let high_h = interval.value[1];
    let high_v = interval.value[2];
    let low_h = interval.value[3];
    let low_v = interval.value[4];
    for (y, row) in z.iter_mut().enumerate() {
        for (x, sample) in row.iter_mut().enumerate() {
            let fx = i32::try_from(x).unwrap_or(i32::MAX);
            let fy = i32::try_from(y).unwrap_or(i32::MAX);
            if (fx < low_h && fy < low_v) || fx > high_h || fy > high_v {
                *sample = 0.0;
            }
        }
    }
    z
}

fn autoreg<S: Sample>(
    rng: &mut Rng,
    plane: &[S],
    w: usize,
    h: usize,
    comp: &Component,
    prev: &Prev,
    scale: u8,
) -> Vec<i32> {
    let mut g = vec![0; w.saturating_mul(h)];
    let shift = u32::from(scale).min(31);
    for y in 0..h {
        for x in 0..w {
            let Some(sample) = plane.get(y * w + x) else {
                continue;
            };
            let key: i32 = (*sample).into();
            let key = intensity8::<S>(key);
            let n = rng.gaussian();
            let mut acc = 0i64;
            let mut hit = false;
            for interval in &comp.intervals {
                if key < i32::from(interval.low) || key > i32::from(interval.high) {
                    continue;
                }
                hit = true;
                acc += i64::from(ar_one(interval, n, &g, prev, Site { x, y, w, h }, shift));
            }
            if hit {
                g[y * w + x] = narrow(acc);
            }
        }
    }
    g
}

struct Site {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
}

fn ar_one(interval: &Interval, n: f64, g: &[i32], prev: &Prev, site: Site, shift: u32) -> i32 {
    let Site { x, y, w, h } = site;
    let v = interval.value;
    let left = neighbor(g, x.checked_sub(1), Some(y), w, h);
    let up = neighbor(g, Some(x), y.checked_sub(1), w, h);
    let diag_l = neighbor(g, x.checked_sub(1), y.checked_sub(1), w, h);
    let diag_r = neighbor(g, x.checked_add(1), y.checked_sub(1), w, h);
    let left2 = neighbor(g, x.checked_sub(2), Some(y), w, h);
    let up2 = neighbor(g, Some(x), y.checked_sub(2), w, h);
    let aspect = i64::from(v[4]);
    let cross = shift.saturating_mul(2).min(62);
    let term1 = i64::from(left) + ((aspect * i64::from(up)) >> shift);
    let term3 = (aspect * (i64::from(diag_l) + i64::from(diag_r))) >> shift;
    let term5 = i64::from(left2) + ((aspect * aspect * i64::from(up2)) >> cross);
    let noise = round_i64(f64::from(v[0]) * n);
    let sum = noise
        + i64::from(v[1]) * term1
        + i64::from(v[3]) * term3
        + i64::from(v[5]) * term5
        + i64::from(v[2]) * i64::from(prev.at(x, y, w, h));
    narrow(sum >> shift)
}

fn neighbor(g: &[i32], x: Option<usize>, y: Option<usize>, w: usize, h: usize) -> i32 {
    let (Some(x), Some(y)) = (x, y) else {
        return 0;
    };
    if x >= w || y >= h {
        return 0;
    }
    g.get(y.saturating_mul(w).saturating_add(x))
        .copied()
        .unwrap_or(0)
}

fn average<S: Sample>(plane: &[S], stride: usize, height: usize, x0: usize, y0: usize) -> i32 {
    let mut sum = 0i64;
    for row in 0..8 {
        for col in 0..8 {
            let y = y0 + row;
            let x = x0 + col;
            if y >= height || x >= stride {
                continue;
            }
            let Some(sample) = plane.get(y * stride + x) else {
                continue;
            };
            let value: i32 = (*sample).into();
            sum += i64::from(value);
        }
    }
    let bits = S::BITS;
    let shift = bits.saturating_sub(2).min(62);
    let add = 1i64 << bits.saturating_sub(3).min(62);
    narrow((sum + add) >> shift).clamp(0, 255)
}

fn intensity8<S: Sample>(sample: i32) -> i32 {
    let bits = S::BITS;
    if bits <= 8 {
        return sample.clamp(0, 255);
    }
    let shift = bits.saturating_sub(8).min(31);
    let add = 1i32 << bits.saturating_sub(9).min(30);
    (sample.saturating_add(add) >> shift).clamp(0, 255)
}

fn blend<S: Sample>(plane: &mut [S], g: &[i32], mode: u8) {
    let denom = i64::from(S::LIMIT.max(1));
    for (slot, grain) in plane.iter_mut().zip(g.iter().copied()) {
        if grain == 0 {
            continue;
        }
        let value: i32 = (*slot).into();
        let next = if mode == 0 {
            value.saturating_add(grain)
        } else {
            let product = i64::from(value) * i64::from(grain);
            value.saturating_add(div_round(product, denom))
        };
        *slot = S::clip(next);
    }
}

fn div_round(num: i64, den: i64) -> i32 {
    if den <= 0 {
        return 0;
    }
    let half = den / 2;
    let rounded = if num >= 0 {
        num.saturating_add(half) / den
    } else {
        num.saturating_sub(half) / den
    };
    narrow(rounded)
}

fn narrow(value: i64) -> i32 {
    i32::try_from(value).unwrap_or(if value > 0 { i32::MAX } else { i32::MIN })
}

fn round_i64(value: f64) -> i64 {
    if !value.is_finite() {
        return 0;
    }
    let rounded = if value >= 0.0 {
        value + 0.5
    } else {
        value - 0.5
    };
    if rounded >= f64::from(i32::MAX) {
        i64::from(i32::MAX)
    } else if rounded <= f64::from(i32::MIN) {
        i64::from(i32::MIN)
    } else {
        i64::from(rounded as i32)
    }
}

fn round_i32(value: f64) -> i32 {
    narrow(round_i64(value))
}

/// Columns of `r` are the orthonormal DCT basis. `r * z * r^T` is the inverse.
fn dct_matrix() -> [[f64; 16]; 16] {
    let mut r = [[0.0; 16]; 16];
    let n: f64 = 16.0;
    for (spatial, row) in r.iter_mut().enumerate() {
        for (freq, cell) in row.iter_mut().enumerate() {
            let alpha = if freq == 0 {
                (1.0 / n).sqrt()
            } else {
                (2.0 / n).sqrt()
            };
            let angle = std::f64::consts::PI
                * f64::from(u32::try_from(freq).unwrap_or(0))
                * (2.0 * f64::from(u32::try_from(spatial).unwrap_or(0)) + 1.0)
                / (2.0 * n);
            *cell = alpha * angle.cos();
        }
    }
    r
}

fn idct(z: &[[f64; 16]; 16], r: &[[f64; 16]; 16]) -> [[i32; 16]; 16] {
    let mut out = [[0; 16]; 16];
    for y in 0..16 {
        for x in 0..16 {
            let mut sum = 0.0;
            for i in 0..16 {
                for j in 0..16 {
                    sum += r[y][i] * z[i][j] * r[x][j];
                }
            }
            out[y][x] = round_i32(sum);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{dct_matrix, idct};

    #[test]
    fn unitary_dc_is_a_constant_one() {
        let matrix = dct_matrix();
        let mut z = [[0.0; 16]; 16];
        z[0][0] = 16.0;
        let q = idct(&z, &matrix);
        assert!(q.iter().flatten().all(|sample| *sample == 1));
    }

    #[test]
    fn horizontal_frequency_varies_along_x() {
        let matrix = dct_matrix();
        let mut z = [[0.0; 16]; 16];
        z[0][1] = 16.0;
        let q = idct(&z, &matrix);
        assert!(q.iter().all(|row| *row == q[0]));
        assert!(q[0].iter().any(|sample| *sample != q[0][0]));
    }
}
