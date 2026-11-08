//! Motion compensated prediction (Sections 7.6 and 7.8).
//!
//! Reference samples outside the decoded area repeat the nearest edge
//! sample, which is the padding the standard specifies for a rectangular
//! VOP. Field prediction reads every other frame row from the padded
//! frame.

/// A reference plane with the edge its padding repeats.
#[derive(Clone, Copy)]
pub(crate) struct RefPlane<'a> {
    pub(crate) data: &'a [u8],
    pub(crate) stride: usize,
    /// Columns and rows of real samples; reads beyond repeat the last.
    pub(crate) edge_w: i32,
    pub(crate) edge_h: i32,
}

/// Which rows a fetch reads.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Rows {
    Frame,
    /// Rows of one field: 0 top, 1 bottom. Coordinates are field rows.
    Field(u8),
}

impl RefPlane<'_> {
    /// Copies the `w` by `h` block whose top-left sample is `(x, y)` into
    /// `out` (stride `w`).
    pub(crate) fn fetch(&self, x: i32, y: i32, w: usize, h: usize, rows: Rows, out: &mut [u8]) {
        // The frame is padded, then a field is read from it: past the
        // bottom a top field reads the last frame row, which is the
        // bottom field's.
        let (step, parity) = match rows {
            Rows::Frame => (1, 0),
            Rows::Field(p) => (2, i32::from(p)),
        };
        let last_x = self.edge_w - 1;
        let last_y = self.edge_h - 1;
        let (top, bottom) = (step * y + parity, step * (y + h as i32 - 1) + parity);
        let inside = x >= 0 && x + w as i32 - 1 <= last_x && top >= 0 && bottom <= last_y;
        for j in 0..h {
            let Some(dst) = out.get_mut(j * w..j * w + w) else {
                return;
            };
            let fy = (step * (y + j as i32) + parity).clamp(0, last_y.max(0));
            let base = fy as usize * self.stride;
            if inside {
                let start = base + x as usize;
                if let Some(src) = self.data.get(start..start + w) {
                    dst.copy_from_slice(src);
                }
            } else {
                for (i, d) in dst.iter_mut().enumerate() {
                    let fx = (x + i as i32).clamp(0, last_x.max(0)) as usize;
                    *d = self.data.get(base + fx).copied().unwrap_or(128);
                }
            }
        }
    }
}

/// Half-sample interpolation of a `w` by `h` block from `src` (stride
/// `w + 1`, `h + 1` rows) at half offsets `dx`, `dy` (0 or 1), with the
/// VOP's rounding control.
pub(crate) fn hpel(
    src: &[u8],
    w: usize,
    h: usize,
    dx: i32,
    dy: i32,
    rounding: bool,
    dst: &mut [u8],
) {
    let s = w + 1;
    let r = i32::from(rounding);
    let at = |i: usize| i32::from(src.get(i).copied().unwrap_or(0));
    for y in 0..h {
        for x in 0..w {
            let i = y * s + x;
            let v = match (dx, dy) {
                (0, 0) => at(i),
                (1, 0) => (at(i) + at(i + 1) + 1 - r) >> 1,
                (0, _) => (at(i) + at(i + s) + 1 - r) >> 1,
                _ => (at(i) + at(i + 1) + at(i + s) + at(i + s + 1) + 2 - r) >> 2,
            };
            if let Some(d) = dst.get_mut(y * w + x) {
                *d = v as u8;
            }
        }
    }
}

/// The 8-tap half-sample filter over `n + 1` samples `s` with the block
/// edge mirrored (Section 7.6.2.2), giving the `n` samples between them.
#[inline]
fn lowpass(s: &[i32], n: usize, rounding: bool, out: &mut [i32]) {
    let at = |i: isize| -> i32 {
        let max = n as isize;
        let j = if i < 0 {
            -1 - i
        } else if i > max {
            2 * max + 1 - i
        } else {
            i
        };
        s.get(j as usize).copied().unwrap_or(0)
    };
    let round = if rounding { 15 } else { 16 };
    for (k, o) in out.iter_mut().enumerate().take(n) {
        let i = k as isize;
        let v = (at(i) + at(i + 1)) * 20 - (at(i - 1) + at(i + 2)) * 6
            + (at(i - 2) + at(i + 3)) * 3
            - (at(i - 3) + at(i + 4));
        *o = ((v + round) >> 5).clamp(0, 255);
    }
}

/// Quarter-sample interpolation of a `w` by `h` block (`w`, `h` at most
/// 16) from `src` (stride `w + 1`, `h + 1` rows) at quarter offsets `dx`,
/// `dy` (0..=3). The horizontal stage forms the quarter positions of each
/// row; the vertical stage filters those.
pub(crate) fn qpel(
    src: &[u8],
    w: usize,
    h: usize,
    dx: i32,
    dy: i32,
    rounding: bool,
    dst: &mut [u8],
) {
    let s = w + 1;
    let r = i32::from(rounding);
    let avg = |a: i32, b: i32| (a + b + 1 - r) >> 1;
    // Horizontal: h + 1 rows of w samples.
    let mut hx = [0i32; 17 * 16];
    let mut row = [0i32; 17];
    let mut half = [0i32; 16];
    for y in 0..=h {
        for (x, v) in row.iter_mut().enumerate().take(s) {
            *v = i32::from(src.get(y * s + x).copied().unwrap_or(0));
        }
        if dx != 0 {
            lowpass(&row, w, rounding, &mut half);
        }
        for x in 0..w {
            let v = match dx {
                0 => row[x],
                1 => avg(row[x], half[x]),
                2 => half[x],
                _ => avg(row[x + 1], half[x]),
            };
            hx[y * 16 + x] = v;
        }
    }
    let mut col = [0i32; 17];
    let mut vhalf = [0i32; 16];
    for x in 0..w {
        for (y, v) in col.iter_mut().enumerate().take(h + 1) {
            *v = hx[y * 16 + x];
        }
        if dy != 0 {
            lowpass(&col, h, rounding, &mut vhalf);
        }
        for y in 0..h {
            let v = match dy {
                0 => col[y],
                1 => avg(col[y], vhalf[y]),
                2 => vhalf[y],
                _ => avg(col[y + 1], vhalf[y]),
            };
            if let Some(d) = dst.get_mut(y * w + x) {
                *d = v as u8;
            }
        }
    }
}

/// The quarter-sample interpolation of streams from libavcodec builds
/// before 4653. Positions with a quarter horizontal offset and a nonzero
/// vertical one average the surrounding full, half and centre samples
/// instead of filtering in two stages.
pub(crate) fn qpel_legacy(
    src: &[u8],
    w: usize,
    h: usize,
    dx: i32,
    dy: i32,
    rounding: bool,
    dst: &mut [u8],
) {
    if dx & 1 == 0 || dy == 0 {
        qpel(src, w, h, dx, dy, rounding, dst);
        return;
    }
    let s = w + 1;
    let r = i32::from(rounding);
    let mut full = [0i32; 17 * 17];
    for (i, v) in full.iter_mut().enumerate().take(s * (h + 1)) {
        *v = i32::from(src.get(i).copied().unwrap_or(0));
    }
    // Half planes: H (h + 1 rows), V, and the centre HV.
    let mut hh = [0i32; 17 * 16];
    let mut vv = [0i32; 16 * 17];
    let mut hv = [0i32; 16 * 16];
    let mut tmp = [0i32; 17];
    let mut out = [0i32; 16];
    for y in 0..=h {
        lowpass(&full[y * s..], w, rounding, &mut out);
        hh[y * 16..y * 16 + w].copy_from_slice(&out[..w]);
    }
    for x in 0..=w.min(16) {
        for y in 0..=h {
            tmp[y] = full[y * s + x];
        }
        lowpass(&tmp, h, rounding, &mut out);
        for y in 0..h {
            vv[y * 17 + x] = out[y];
        }
    }
    for x in 0..w {
        for y in 0..=h {
            tmp[y] = hh[y * 16 + x];
        }
        lowpass(&tmp, h, rounding, &mut out);
        for y in 0..h {
            hv[y * 16 + x] = out[y];
        }
    }
    let avg2 = |a: i32, b: i32| (a + b + 1 - r) >> 1;
    let avg4 = |a: i32, b: i32, c: i32, d: i32| (a + b + c + d + 2 - r) >> 2;
    let ox = usize::from(dx == 3);
    let oy = usize::from(dy == 3);
    for y in 0..h {
        for x in 0..w {
            let v = if dy == 2 {
                avg2(vv[y * 17 + x + ox], hv[y * 16 + x])
            } else {
                avg4(
                    full[(y + oy) * s + x + ox],
                    hh[(y + oy) * 16 + x],
                    vv[y * 17 + x + ox],
                    hv[y * 16 + x],
                )
            };
            if let Some(d) = dst.get_mut(y * w + x) {
                *d = v as u8;
            }
        }
    }
}

/// Chroma vector of a 16x16 half-sample luma vector component: half the
/// vector, rounded toward the half position.
#[inline]
pub(crate) fn chroma_hpel(m: i32) -> i32 {
    (m >> 1) | (m & 1)
}

/// Chroma vector from the sum of four 8x8 luma vector components, in half
/// samples (Table 7-6's sixteenth rounding).
#[inline]
pub(crate) fn chroma_4mv(sum: i32) -> i32 {
    const ROUND: [i32; 16] = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1];
    ROUND[(sum & 15) as usize] + (sum >> 3)
}

/// The interpolation one prediction uses.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Pel {
    Half,
    Quarter,
    /// Quarter-sample streams from libavcodec builds before 4653.
    QuarterLegacy,
}

/// Predicts a `w` by `h` luma block at full-sample position `(x, y)` plus
/// vector `(mx, my)` in the units `pel` names.
#[allow(clippy::too_many_arguments)]
pub(crate) fn luma_block(
    r: &RefPlane<'_>,
    x: i32,
    y: i32,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    pel: Pel,
    rows: Rows,
    rounding: bool,
    dst: &mut [u8],
) {
    let mut src = [0u8; 17 * 17];
    match pel {
        Pel::Half => {
            r.fetch(x + (mx >> 1), y + (my >> 1), w + 1, h + 1, rows, &mut src);
            hpel(&src, w, h, mx & 1, my & 1, rounding, dst);
        }
        Pel::Quarter | Pel::QuarterLegacy => {
            r.fetch(x + (mx >> 2), y + (my >> 2), w + 1, h + 1, rows, &mut src);
            if pel == Pel::Quarter {
                qpel(&src, w, h, mx & 3, my & 3, rounding, dst);
            } else {
                qpel_legacy(&src, w, h, mx & 3, my & 3, rounding, dst);
            }
        }
    }
}

/// Predicts a `w` by `h` chroma block at `(x, y)` plus a half-sample
/// chroma vector.
#[allow(clippy::too_many_arguments)]
pub(crate) fn chroma_block(
    r: &RefPlane<'_>,
    x: i32,
    y: i32,
    w: usize,
    h: usize,
    cx: i32,
    cy: i32,
    rows: Rows,
    rounding: bool,
    dst: &mut [u8],
) {
    let mut src = [0u8; 9 * 9];
    r.fetch(x + (cx >> 1), y + (cy >> 1), w + 1, h + 1, rows, &mut src);
    hpel(&src, w, h, cx & 1, cy & 1, rounding, dst);
}

/// Averages `other` into `dst` with upward rounding, for bidirectional
/// prediction.
pub(crate) fn average(dst: &mut [u8], other: &[u8]) {
    for (d, &o) in dst.iter_mut().zip(other) {
        *d = ((u16::from(*d) + u16::from(o) + 1) >> 1) as u8;
    }
}

/// Global motion parameters of one S-VOP (Section 7.8.4), in the form the
/// per-sample warp uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Warp {
    /// 1/(2 << accuracy) sample offsets: luma `[0]`, chroma `[1]`.
    pub(crate) offset: [[i64; 2]; 2],
    /// `[component][axis]`.
    pub(crate) delta: [[i64; 2]; 2],
    /// Luma and chroma shifts of the affine form.
    pub(crate) shift: [u32; 2],
    pub(crate) accuracy: u32,
    /// The warp is a translation: the offset alone describes it.
    pub(crate) translation: bool,
}

fn rounded_div(a: i64, b: i64) -> i64 {
    if b == 0 {
        return 0;
    }
    if a >= 0 {
        (a + (b >> 1)) / b
    } else {
        (a - (b >> 1)) / b
    }
}

impl Warp {
    /// Derives the warp from the decoded trajectory of `points` warping
    /// points over a `width` by `height` VOP. `divx_413` selects the
    /// sprite reference formula one DivX 5 build used. Returns `None`
    /// when the parameters overflow the per-sample arithmetic.
    pub(crate) fn new(
        trajectory: &[(i32, i32); 3],
        points: u8,
        accuracy: u8,
        width: u32,
        height: u32,
        divx_413: bool,
    ) -> Option<Self> {
        let a: i64 = 2 << accuracy;
        let rho = 3 - i64::from(accuracy);
        let r = 16 / a;
        let (w, h) = (i64::from(width), i64::from(height));
        if w <= 0 || h <= 0 {
            return None;
        }
        let d: Vec<(i64, i64)> = trajectory
            .iter()
            .map(|&(x, y)| (i64::from(x), i64::from(y)))
            .collect();
        let d = |i: usize| d.get(i).copied().unwrap_or((0, 0));
        let vop = [(0i64, 0i64), (w, 0), (0, h)];
        let mut alpha = 1i64;
        while (1 << alpha) < w {
            alpha += 1;
        }
        let mut beta = 0i64;
        while (1 << beta) < h {
            beta += 1;
        }
        let (w2, h2) = (1i64 << alpha, 1i64 << beta);
        let mut sref = [(0i64, 0i64); 3];
        for (i, s) in sref.iter_mut().enumerate() {
            let (sx, sy) = if i == 0 {
                d(0)
            } else {
                (d(0).0 + d(i).0, d(0).1 + d(i).1)
            };
            *s = if divx_413 {
                (a * vop[i].0 + sx, a * vop[i].1 + sy)
            } else {
                (
                    (a >> 1) * (2 * vop[i].0 + sx),
                    (a >> 1) * (2 * vop[i].1 + sy),
                )
            };
        }
        let virt = [
            (
                16 * (vop[0].0 + w2)
                    + rounded_div(
                        (w - w2) * (r * sref[0].0 - 16 * vop[0].0)
                            + w2 * (r * sref[1].0 - 16 * vop[1].0),
                        w,
                    ),
                16 * vop[0].1
                    + rounded_div(
                        (w - w2) * (r * sref[0].1 - 16 * vop[0].1)
                            + w2 * (r * sref[1].1 - 16 * vop[1].1),
                        w,
                    ),
            ),
            (
                16 * vop[0].0
                    + rounded_div(
                        (h - h2) * (r * sref[0].0 - 16 * vop[0].0)
                            + h2 * (r * sref[2].0 - 16 * vop[2].0),
                        h,
                    ),
                16 * (vop[0].1 + h2)
                    + rounded_div(
                        (h - h2) * (r * sref[0].1 - 16 * vop[0].1)
                            + h2 * (r * sref[2].1 - 16 * vop[2].1),
                        h,
                    ),
            ),
        ];
        let mut offset = [[0i64; 2]; 2];
        let mut delta = [[a, 0], [0, a]];
        let mut shift = [0u32; 2];
        match points {
            0 => {}
            1 => {
                offset[0] = [sref[0].0 - a * vop[0].0, sref[0].1 - a * vop[0].1];
                offset[1] = [
                    ((sref[0].0 >> 1) | (sref[0].0 & 1)) - a * (vop[0].0 / 2),
                    ((sref[0].1 >> 1) | (sref[0].1 & 1)) - a * (vop[0].1 / 2),
                ];
            }
            2 => {
                let ar = alpha + rho;
                let (dxx, dxy) = (-r * sref[0].0 + virt[0].0, r * sref[0].1 - virt[0].1);
                let (dyx, dyy) = (-r * sref[0].1 + virt[0].1, -r * sref[0].0 + virt[0].0);
                offset[0] = [
                    sref[0].0 * (1 << ar) + dxx * -vop[0].0 + dxy * -vop[0].1 + (1 << (ar - 1)),
                    sref[0].1 * (1 << ar) + dyx * -vop[0].0 + dyy * -vop[0].1 + (1 << (ar - 1)),
                ];
                offset[1] = [
                    dxx * (-2 * vop[0].0 + 1) + dxy * (-2 * vop[0].1 + 1) + 2 * w2 * r * sref[0].0
                        - 16 * w2
                        + (1 << (ar + 1)),
                    dyx * (-2 * vop[0].0 + 1) + dyy * (-2 * vop[0].1 + 1) + 2 * w2 * r * sref[0].1
                        - 16 * w2
                        + (1 << (ar + 1)),
                ];
                delta = [[dxx, dxy], [dyx, dyy]];
                shift = [ar as u32, (ar + 2) as u32];
            }
            3 => {
                let min_ab = alpha.min(beta);
                let (w3, h3) = (w2 >> min_ab, h2 >> min_ab);
                let s = alpha + beta + rho - min_ab;
                let dxx = (-r * sref[0].0 + virt[0].0) * h3;
                let dxy = (-r * sref[0].0 + virt[1].0) * w3;
                let dyx = (-r * sref[0].1 + virt[0].1) * h3;
                let dyy = (-r * sref[0].1 + virt[1].1) * w3;
                offset[0] = [
                    sref[0].0 * (1 << s) + dxx * -vop[0].0 + dxy * -vop[0].1 + (1 << (s - 1)),
                    sref[0].1 * (1 << s) + dyx * -vop[0].0 + dyy * -vop[0].1 + (1 << (s - 1)),
                ];
                offset[1] = [
                    dxx * (-2 * vop[0].0 + 1)
                        + dxy * (-2 * vop[0].1 + 1)
                        + 2 * w2 * h3 * r * sref[0].0
                        - 16 * w2 * h3
                        + (1 << (s + 1)),
                    dyx * (-2 * vop[0].0 + 1)
                        + dyy * (-2 * vop[0].1 + 1)
                        + 2 * w2 * h3 * r * sref[0].1
                        - 16 * w2 * h3
                        + (1 << (s + 1)),
                ];
                delta = [[dxx, dxy], [dyx, dyy]];
                shift = [s as u32, (s + 2) as u32];
            }
            _ => return None,
        }
        let accuracy = u32::from(accuracy);
        if delta[0][0] == a << shift[0]
            && delta[0][1] == 0
            && delta[1][0] == 0
            && delta[1][1] == a << shift[0]
        {
            for c in 0..2 {
                offset[0][c] >>= shift[0];
                offset[1][c] >>= shift[1];
            }
            return Some(Self {
                offset,
                delta: [[a, 0], [0, a]],
                shift: [0, 0],
                accuracy,
                translation: true,
            });
        }
        // Scale to 16 fractional bits for the per-sample warp, refusing
        // parameters the 32-bit sample arithmetic of the warp cannot hold.
        let shift_y = 16 - i64::from(shift[0]);
        let shift_c = 16 - i64::from(shift[1]);
        if shift_y < 0 || shift_c < 0 {
            return None;
        }
        let limit = i64::from(i32::MAX);
        for c in 0..2 {
            if offset[0][c].abs() >= limit >> shift_y
                || offset[1][c].abs() >= limit >> shift_c
                || delta[0][c].abs() >= limit >> shift_y
                || delta[1][c].abs() >= limit >> shift_y
            {
                return None;
            }
        }
        for c in 0..2 {
            offset[0][c] <<= shift_y;
            offset[1][c] <<= shift_c;
            delta[0][c] <<= shift_y;
            delta[1][c] <<= shift_y;
        }
        for c in 0..2 {
            let sd = [delta[c][0] - (a << 16), delta[c][1] - (a << 16)];
            let checks = [
                offset[0][c] + delta[c][0] * (w + 16),
                offset[0][c] + delta[c][1] * (h + 16),
                offset[0][c] + delta[c][0] * (w + 16) + delta[c][1] * (h + 16),
                delta[c][0] * (w + 16),
                delta[c][1] * (h + 16),
                sd[0],
                sd[1],
                offset[0][c] + sd[0] * (w + 16),
                offset[0][c] + sd[1] * (h + 16),
                offset[0][c] + sd[0] * (w + 16) + sd[1] * (h + 16),
            ];
            if checks.iter().any(|v| v.abs() >= limit) {
                return None;
            }
        }
        Some(Self {
            offset,
            delta,
            shift: [16, 16],
            accuracy,
            translation: false,
        })
    }

    /// The vector a GMC macroblock at `(mb_x, mb_y)` contributes to motion
    /// vector prediction: the average warp over its luma samples, in half
    /// or quarter samples, clamped to the `fcode` range (Section 7.8.7.3).
    pub(crate) fn average_vector(
        &self,
        mb_x: u32,
        mb_y: u32,
        quarter: bool,
        fcode: u32,
        divx_413: bool,
    ) -> (i32, i32) {
        let a = i64::from(self.accuracy);
        let q = i64::from(quarter);
        let len = 1i64 << (fcode + 4);
        let mut out = [0i64; 2];
        for (n, o) in out.iter_mut().enumerate() {
            let sum = if self.translation {
                if divx_413 && a >= q {
                    self.offset[0][n] / (1 << (a - q))
                } else {
                    rshift(self.offset[0][n] << q, a)
                }
            } else {
                let shift = i64::from(self.shift[0]);
                let mut dx = self.delta[n][0];
                let mut dy = self.delta[n][1];
                if n == 1 {
                    dy -= 1 << (shift + a + 1);
                } else {
                    dx -= 1 << (shift + a + 1);
                }
                let mb_v =
                    self.offset[0][n] + dx * i64::from(mb_x) * 16 + dy * i64::from(mb_y) * 16;
                let mut sum = 0i64;
                for y in 0..16 {
                    let mut v = mb_v + dy * y;
                    for _ in 0..16 {
                        sum += (v as i32 >> shift) as i64;
                        v += dx;
                    }
                }
                rshift(sum, a + 8 - q)
            };
            *o = sum.clamp(-len, len - 1);
        }
        (out[0] as i32, out[1] as i32)
    }
}

/// Shift right rounding to nearest, halves away from zero.
fn rshift(a: i64, b: i64) -> i64 {
    if b <= 0 {
        return a;
    }
    if a > 0 {
        (a + ((1 << b) >> 1)) >> b
    } else {
        (a + ((1 << b) >> 1) - 1) >> b
    }
}

/// Bilinear sixteenth-sample prediction for a translational warp: an
/// `w` by `h` block from `src` (stride `w + 1`, `h + 1` rows).
fn bilinear16(src: &[u8], w: usize, h: usize, fx: i32, fy: i32, rounding: bool, dst: &mut [u8]) {
    let s = w + 1;
    let a = (16 - fx) * (16 - fy);
    let b = fx * (16 - fy);
    let c = (16 - fx) * fy;
    let d = fx * fy;
    let round = 128 - i32::from(rounding);
    let at = |i: usize| i32::from(src.get(i).copied().unwrap_or(0));
    for y in 0..h {
        for x in 0..w {
            let i = y * s + x;
            let v = (a * at(i) + b * at(i + 1) + c * at(i + s) + d * at(i + s + 1) + round) >> 8;
            if let Some(o) = dst.get_mut(y * w + x) {
                *o = v as u8;
            }
        }
    }
}

/// Global motion compensated prediction of the macroblock at `(mb_x,
/// mb_y)` into `y` (16x16), `u` and `v` (8x8).
#[allow(clippy::too_many_arguments)]
pub(crate) fn gmc_macroblock(
    warp: &Warp,
    luma: &RefPlane<'_>,
    cb: &RefPlane<'_>,
    cr: &RefPlane<'_>,
    mb_x: u32,
    mb_y: u32,
    rounding: bool,
    out_y: &mut [u8],
    out_u: &mut [u8],
    out_v: &mut [u8],
) {
    let (bx, by) = (mb_x as i32 * 16, mb_y as i32 * 16);
    let a = warp.accuracy;
    if warp.translation {
        let mut src = [0u8; 17 * 17];
        let (ox, oy) = (warp.offset[0][0] as i32, warp.offset[0][1] as i32);
        let sx = bx + (ox >> (a + 1));
        let sy = by + (oy >> (a + 1));
        let fx = (ox << (3 - a)) & 15;
        let fy = (oy << (3 - a)) & 15;
        luma.fetch(sx, sy, 17, 17, Rows::Frame, &mut src);
        if (fx | fy) & 7 != 0 {
            bilinear16(&src, 16, 16, fx, fy, rounding, out_y);
        } else {
            hpel(&src, 16, 16, fx >> 3, fy >> 3, rounding, out_y);
        }
        let (ox, oy) = (warp.offset[1][0] as i32, warp.offset[1][1] as i32);
        let sx = bx / 2 + (ox >> (a + 1));
        let sy = by / 2 + (oy >> (a + 1));
        let fx = (ox << (3 - a)) & 15;
        let fy = (oy << (3 - a)) & 15;
        for (plane, out) in [(cb, &mut *out_u), (cr, &mut *out_v)] {
            let mut src = [0u8; 9 * 9];
            plane.fetch(sx, sy, 9, 9, Rows::Frame, &mut src);
            bilinear16(&src, 8, 8, fx, fy, rounding, out);
        }
        return;
    }
    let shift = a + 1;
    let round = (1i64 << (2 * a + 1)) - i64::from(rounding);
    let warp_block = |plane: &RefPlane<'_>, ox: i64, oy: i64, n: usize, out: &mut [u8]| {
        let s = 1i64 << shift;
        let last_x = i64::from(plane.edge_w - 1);
        let last_y = i64::from(plane.edge_h - 1);
        let at = |x: i64, y: i64| -> i64 {
            let x = x.clamp(0, last_x.max(0)) as usize;
            let y = y.clamp(0, last_y.max(0)) as usize;
            i64::from(plane.data.get(y * plane.stride + x).copied().unwrap_or(128))
        };
        for y in 0..n {
            for x in 0..n {
                let vx = ox + warp.delta[0][0] * x as i64 + warp.delta[0][1] * y as i64;
                let vy = oy + warp.delta[1][0] * x as i64 + warp.delta[1][1] * y as i64;
                let px = (vx as i32 >> 16) as i64;
                let py = (vy as i32 >> 16) as i64;
                let fx = px & (s - 1);
                let fy = py & (s - 1);
                let (ix, iy) = (px >> shift, py >> shift);
                let v = ((at(ix, iy) * (s - fx) + at(ix + 1, iy) * fx) * (s - fy)
                    + (at(ix, iy + 1) * (s - fx) + at(ix + 1, iy + 1) * fx) * fy
                    + round)
                    >> (2 * shift);
                if let Some(o) = out.get_mut(y * n + x) {
                    *o = v as u8;
                }
            }
        }
    };
    let (bx, by) = (i64::from(bx), i64::from(by));
    let ox = warp.offset[0][0] + warp.delta[0][0] * bx + warp.delta[0][1] * by;
    let oy = warp.offset[0][1] + warp.delta[1][0] * bx + warp.delta[1][1] * by;
    warp_block(luma, ox, oy, 16, out_y);
    let ox = warp.offset[1][0] + warp.delta[0][0] * bx / 2 + warp.delta[0][1] * by / 2;
    let oy = warp.offset[1][1] + warp.delta[1][0] * bx / 2 + warp.delta[1][1] * by / 2;
    warp_block(cb, ox, oy, 8, out_u);
    warp_block(cr, ox, oy, 8, out_v);
}

#[cfg(test)]
mod tests {
    use super::{Pel, RefPlane, Rows, chroma_4mv, chroma_hpel, luma_block, qpel, qpel_legacy};

    fn ramp(w: usize, h: usize) -> Vec<u8> {
        (0..w * h)
            .map(|i| ((i % w) * 3 + (i / w) * 5) as u8)
            .collect()
    }

    #[test]
    fn fetch_repeats_edges_and_reads_fields() {
        let data = ramp(16, 16);
        let r = RefPlane {
            data: &data,
            stride: 16,
            edge_w: 16,
            edge_h: 16,
        };
        let mut out = [0u8; 4];
        r.fetch(-3, -3, 2, 2, Rows::Frame, &mut out);
        assert_eq!(out, [data[0]; 4]);
        r.fetch(15, 0, 2, 1, Rows::Frame, &mut out[..2]);
        assert_eq!(out[..2], [data[15], data[15]]);
        // Bottom field row 1 is frame row 3.
        r.fetch(0, 1, 1, 1, Rows::Field(1), &mut out[..1]);
        assert_eq!(out[0], data[3 * 16]);
        // Past the bottom both fields read the padded frame's last row,
        // and above the top its first.
        r.fetch(0, 9, 1, 1, Rows::Field(0), &mut out[..1]);
        assert_eq!(out[0], data[15 * 16]);
        r.fetch(0, 9, 1, 1, Rows::Field(1), &mut out[..1]);
        assert_eq!(out[0], data[15 * 16]);
        r.fetch(0, -2, 1, 1, Rows::Field(1), &mut out[..1]);
        assert_eq!(out[0], data[0]);
        // Inside, a field reads its own rows.
        r.fetch(0, 7, 1, 1, Rows::Field(0), &mut out[..1]);
        assert_eq!(out[0], data[14 * 16]);
    }

    #[test]
    fn whole_sample_vectors_copy() {
        let data = ramp(32, 32);
        let r = RefPlane {
            data: &data,
            stride: 32,
            edge_w: 32,
            edge_h: 32,
        };
        let mut out = [0u8; 64];
        for pel in [Pel::Half, Pel::Quarter, Pel::QuarterLegacy] {
            let scale = if pel == Pel::Half { 2 } else { 4 };
            luma_block(
                &r,
                8,
                8,
                8,
                8,
                2 * scale,
                3 * scale,
                pel,
                Rows::Frame,
                false,
                &mut out,
            );
            for y in 0..8 {
                for x in 0..8 {
                    assert_eq!(out[y * 8 + x], data[(11 + y) * 32 + 10 + x]);
                }
            }
        }
    }

    #[test]
    fn quarter_sample_of_a_flat_area_is_flat() {
        let src = [77u8; 17 * 17];
        let mut out = [0u8; 256];
        for dx in 0..4 {
            for dy in 0..4 {
                for rounding in [false, true] {
                    qpel(&src, 16, 16, dx, dy, rounding, &mut out);
                    assert!(out.iter().all(|&v| v == 77));
                    qpel_legacy(&src, 16, 16, dx, dy, rounding, &mut out);
                    assert!(out.iter().all(|&v| v == 77));
                }
            }
        }
    }

    #[test]
    fn chroma_vectors_round_as_the_standard_says() {
        // 16x16: 1.5 luma samples is 0.75 chroma, rounded to the half.
        assert_eq!(chroma_hpel(3), 1);
        assert_eq!(chroma_hpel(4), 2);
        assert_eq!(chroma_hpel(-1), -1);
        // Four equal vectors give the 16x16 result.
        for m in -40..40 {
            assert_eq!(chroma_4mv(4 * m), chroma_hpel(m), "{m}");
        }
    }
}
