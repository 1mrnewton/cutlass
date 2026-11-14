//! The loop filter (RFC 6386 Section 15).

/// What the loop filter needs to know about one macroblock.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MbFilter {
    /// Filter level after segment and delta adjustments, 0..63.
    pub level: u8,
    /// Interior limit, at least 1.
    pub interior: u8,
    /// High-edge-variance threshold.
    pub hev: u8,
    /// Whether the edges inside the macroblock are filtered.
    pub inner: bool,
}

/// Interior limit for a level and sharpness (Section 15.2).
#[must_use]
pub fn interior_limit(level: u8, sharpness: u8) -> u8 {
    let mut limit = level;
    if sharpness > 0 {
        limit >>= if sharpness > 4 { 2 } else { 1 };
        limit = limit.min(9 - sharpness.min(8));
    }
    limit.max(1)
}

/// High-edge-variance threshold (Section 15.2).
#[must_use]
pub fn hev_threshold(level: u8, key: bool) -> u8 {
    let mut hev = u8::from(level >= 15);
    if level >= 40 {
        hev += 1;
    }
    if level >= 20 && !key {
        hev += 1;
    }
    hev
}

#[inline]
fn s8(x: i32) -> i32 {
    x.clamp(-128, 127)
}

#[inline]
fn u8c(x: i32) -> u8 {
    x.clamp(0, 255) as u8
}

/// Pixels across one edge: `p[0..4]` are p3..p0, `q[0..4]` are q0..q3.
/// `at` is q0 and `step` the distance between taps.
#[inline]
fn px(buf: &[u8], at: usize, step: usize, k: isize) -> i32 {
    i32::from(buf[(at as isize + k * step as isize) as usize])
}

#[inline]
fn simple_ok(buf: &[u8], at: usize, step: usize, limit: i32) -> bool {
    let p1 = px(buf, at, step, -2);
    let p0 = px(buf, at, step, -1);
    let q0 = px(buf, at, step, 0);
    let q1 = px(buf, at, step, 1);
    (p0 - q0).abs() * 2 + ((p1 - q1).abs() >> 1) <= limit
}

#[inline]
fn common(buf: &mut [u8], at: usize, step: usize, outer: bool) {
    let p1 = px(buf, at, step, -2);
    let p0 = px(buf, at, step, -1);
    let q0 = px(buf, at, step, 0);
    let q1 = px(buf, at, step, 1);
    let mut a = 3 * (q0 - p0);
    if outer {
        a += s8(p1 - q1);
    }
    let a = s8(a);
    let f1 = (a + 4).min(127) >> 3;
    let f2 = (a + 3).min(127) >> 3;
    buf[at - step] = u8c(p0 + f2);
    buf[at] = u8c(q0 - f1);
    if !outer {
        let a = (f1 + 1) >> 1;
        buf[at - 2 * step] = u8c(p1 + a);
        buf[at + step] = u8c(q1 - a);
    }
}

/// Filters 16 positions across a normal edge at once. Row `k` of `r`
/// holds tap `k` (p3, p2, p1, p0, q0, q1, q2, q3) of every position
/// (Sections 15.3 and 15.4). Every lane takes the same steps and a
/// select, so the loop vectorizes.
#[inline]
fn normal_lanes(r: &mut [[u8; 16]; 8], edge: i16, interior: i16, hev_t: i16, macroblock: bool) {
    let s8 = |x: i16| x.clamp(-128, 127);
    let u8c = |x: i16| x.clamp(0, 255) as u8;
    for i in 0..16 {
        let p3 = i16::from(r[0][i]);
        let p2 = i16::from(r[1][i]);
        let p1 = i16::from(r[2][i]);
        let p0 = i16::from(r[3][i]);
        let q0 = i16::from(r[4][i]);
        let q1 = i16::from(r[5][i]);
        let q2 = i16::from(r[6][i]);
        let q3 = i16::from(r[7][i]);
        let on = ((p0 - q0).abs() * 2 + ((p1 - q1).abs() >> 1) <= edge)
            & ((p3 - p2).abs() <= interior)
            & ((p2 - p1).abs() <= interior)
            & ((p1 - p0).abs() <= interior)
            & ((q3 - q2).abs() <= interior)
            & ((q2 - q1).abs() <= interior)
            & ((q1 - q0).abs() <= interior);
        let high = ((p1 - p0).abs() > hev_t) | ((q1 - q0).abs() > hev_t);
        let w = s8(s8(p1 - q1) + 3 * (q0 - p0));
        // The whole-macroblock filter, for a low-variance macroblock edge.
        let wide = macroblock & !high;
        let a27 = (27 * w + 63) >> 7;
        let a18 = (18 * w + 63) >> 7;
        let a9 = (9 * w + 63) >> 7;
        // The common filter: with p1 - q1 on a high-variance edge, and
        // moving p1 and q1 as well on a low-variance inner edge.
        let a = if high { w } else { s8(3 * (q0 - p0)) };
        let f1 = (a + 4).min(127) >> 3;
        let f2 = (a + 3).min(127) >> 3;
        let f3 = (f1 + 1) >> 1;
        // Selects, not branches, and unconditional stores.
        let d1 = if wide { a9 } else { 0 };
        let d2 = if wide {
            a18
        } else if high {
            0
        } else {
            f3
        };
        let dp0 = if wide { a27 } else { f2 };
        let dq0 = if wide { a27 } else { f1 };
        let (d1, d2, dp0, dq0) = if on { (d1, d2, dp0, dq0) } else { (0, 0, 0, 0) };
        r[1][i] = u8c(p2 + d1);
        r[2][i] = u8c(p1 + d2);
        r[3][i] = u8c(p0 + dp0);
        r[4][i] = u8c(q0 - dq0);
        r[5][i] = u8c(q1 - d2);
        r[6][i] = u8c(q2 - d1);
    }
}

/// Filters `count` (at most 16) positions along an edge. `at` is the
/// first q0 pixel, `step` crosses the edge, and `along` walks it.
#[allow(clippy::too_many_arguments)]
#[expect(
    clippy::needless_range_loop,
    reason = "a vertical edge transposes rows into lanes by index"
)]
fn normal_edge(
    buf: &mut [u8],
    at: usize,
    step: usize,
    along: usize,
    count: usize,
    edge: i32,
    interior: i32,
    hev_t: i32,
    macroblock: bool,
) {
    let count = count.min(16);
    let (edge, interior, hev_t) = (edge as i16, interior as i16, hev_t as i16);
    let mut r = [[0u8; 16]; 8];
    if step == 1 {
        // A vertical edge: each position is eight bytes of one row,
        // transposed into lanes and back.
        for k in 0..count {
            let start = at + k * along - 4;
            for (j, &v) in buf[start..start + 8].iter().enumerate() {
                r[j][k] = v;
            }
        }
        normal_lanes(&mut r, edge, interior, hev_t, macroblock);
        for k in 0..count {
            let start = at + k * along - 4;
            for (j, v) in buf[start + 1..start + 7].iter_mut().enumerate() {
                *v = r[j + 1][k];
            }
        }
        return;
    }
    // A horizontal edge: each tap is a run of one row.
    let base = at - 4 * step;
    for (k, row) in r.iter_mut().enumerate() {
        let from = base + k * step;
        row[..count].copy_from_slice(&buf[from..from + count]);
    }
    normal_lanes(&mut r, edge, interior, hev_t, macroblock);
    for (k, row) in r.iter().enumerate().skip(1).take(6) {
        let to = base + k * step;
        buf[to..to + count].copy_from_slice(&row[..count]);
    }
}

fn simple_edge(buf: &mut [u8], mut at: usize, step: usize, along: usize, limit: i32) {
    for _ in 0..16 {
        if simple_ok(buf, at, step, limit) {
            common(buf, at, step, true);
        }
        at += along;
    }
}

/// Planes of a frame being filtered.
pub struct Planes<'a> {
    pub y: &'a mut [u8],
    pub u: &'a mut [u8],
    pub v: &'a mut [u8],
    pub mbw: usize,
    pub mbh: usize,
}

/// Filters a whole frame in raster order. `mbs` holds one entry per
/// macroblock.
pub fn filter_frame(planes: &mut Planes<'_>, mbs: &[MbFilter], simple: bool) {
    let ys = planes.mbw * 16;
    let cs = planes.mbw * 8;
    for my in 0..planes.mbh {
        for mx in 0..planes.mbw {
            let mb = mbs[my * planes.mbw + mx];
            if mb.level == 0 {
                continue;
            }
            let level = i32::from(mb.level);
            let interior = i32::from(mb.interior);
            let hev_t = i32::from(mb.hev);
            let mb_limit = (level + 2) * 2 + interior;
            let sub_limit = level * 2 + interior;
            let y0 = my * 16 * ys + mx * 16;
            let c0 = my * 8 * cs + mx * 8;
            if simple {
                if mx > 0 {
                    simple_edge(planes.y, y0, 1, ys, mb_limit);
                }
                if mb.inner {
                    for x in [4, 8, 12] {
                        simple_edge(planes.y, y0 + x, 1, ys, sub_limit);
                    }
                }
                if my > 0 {
                    simple_edge(planes.y, y0, ys, 1, mb_limit);
                }
                if mb.inner {
                    for r in [4, 8, 12] {
                        simple_edge(planes.y, y0 + r * ys, ys, 1, sub_limit);
                    }
                }
                continue;
            }
            if mx > 0 {
                normal_edge(planes.y, y0, 1, ys, 16, mb_limit, interior, hev_t, true);
                normal_edge(planes.u, c0, 1, cs, 8, mb_limit, interior, hev_t, true);
                normal_edge(planes.v, c0, 1, cs, 8, mb_limit, interior, hev_t, true);
            }
            if mb.inner {
                for x in [4, 8, 12] {
                    normal_edge(
                        planes.y,
                        y0 + x,
                        1,
                        ys,
                        16,
                        sub_limit,
                        interior,
                        hev_t,
                        false,
                    );
                }
                normal_edge(
                    planes.u,
                    c0 + 4,
                    1,
                    cs,
                    8,
                    sub_limit,
                    interior,
                    hev_t,
                    false,
                );
                normal_edge(
                    planes.v,
                    c0 + 4,
                    1,
                    cs,
                    8,
                    sub_limit,
                    interior,
                    hev_t,
                    false,
                );
            }
            if my > 0 {
                normal_edge(planes.y, y0, ys, 1, 16, mb_limit, interior, hev_t, true);
                normal_edge(planes.u, c0, cs, 1, 8, mb_limit, interior, hev_t, true);
                normal_edge(planes.v, c0, cs, 1, 8, mb_limit, interior, hev_t, true);
            }
            if mb.inner {
                for r in [4, 8, 12] {
                    normal_edge(
                        planes.y,
                        y0 + r * ys,
                        ys,
                        1,
                        16,
                        sub_limit,
                        interior,
                        hev_t,
                        false,
                    );
                }
                normal_edge(
                    planes.u,
                    c0 + 4 * cs,
                    cs,
                    1,
                    8,
                    sub_limit,
                    interior,
                    hev_t,
                    false,
                );
                normal_edge(
                    planes.v,
                    c0 + 4 * cs,
                    cs,
                    1,
                    8,
                    sub_limit,
                    interior,
                    hev_t,
                    false,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{hev_threshold, interior_limit};

    #[test]
    fn limits_follow_section_15() {
        assert_eq!(interior_limit(0, 0), 1);
        assert_eq!(interior_limit(63, 0), 63);
        assert_eq!(interior_limit(63, 1), 8);
        assert_eq!(interior_limit(20, 4), 5);
        assert_eq!(interior_limit(20, 5), 4);
        assert_eq!(interior_limit(63, 7), 2);
        assert_eq!(hev_threshold(14, true), 0);
        assert_eq!(hev_threshold(15, true), 1);
        assert_eq!(hev_threshold(20, false), 2);
        assert_eq!(hev_threshold(40, true), 2);
        assert_eq!(hev_threshold(40, false), 3);
    }
}
