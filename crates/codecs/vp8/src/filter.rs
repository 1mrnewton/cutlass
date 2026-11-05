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
fn normal_ok(buf: &[u8], at: usize, step: usize, edge: i32, interior: i32) -> bool {
    if !simple_ok(buf, at, step, edge) {
        return false;
    }
    let p3 = px(buf, at, step, -4);
    let p2 = px(buf, at, step, -3);
    let p1 = px(buf, at, step, -2);
    let p0 = px(buf, at, step, -1);
    let q0 = px(buf, at, step, 0);
    let q1 = px(buf, at, step, 1);
    let q2 = px(buf, at, step, 2);
    let q3 = px(buf, at, step, 3);
    (p3 - p2).abs() <= interior
        && (p2 - p1).abs() <= interior
        && (p1 - p0).abs() <= interior
        && (q3 - q2).abs() <= interior
        && (q2 - q1).abs() <= interior
        && (q1 - q0).abs() <= interior
}

#[inline]
fn hev(buf: &[u8], at: usize, step: usize, threshold: i32) -> bool {
    let p1 = px(buf, at, step, -2);
    let p0 = px(buf, at, step, -1);
    let q0 = px(buf, at, step, 0);
    let q1 = px(buf, at, step, 1);
    (p1 - p0).abs() > threshold || (q1 - q0).abs() > threshold
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

#[inline]
fn mb_edge(buf: &mut [u8], at: usize, step: usize) {
    let p2 = px(buf, at, step, -3);
    let p1 = px(buf, at, step, -2);
    let p0 = px(buf, at, step, -1);
    let q0 = px(buf, at, step, 0);
    let q1 = px(buf, at, step, 1);
    let q2 = px(buf, at, step, 2);
    let w = s8(s8(p1 - q1) + 3 * (q0 - p0));
    let a = (27 * w + 63) >> 7;
    buf[at - step] = u8c(p0 + a);
    buf[at] = u8c(q0 - a);
    let a = (18 * w + 63) >> 7;
    buf[at - 2 * step] = u8c(p1 + a);
    buf[at + step] = u8c(q1 - a);
    let a = (9 * w + 63) >> 7;
    buf[at - 3 * step] = u8c(p2 + a);
    buf[at + 2 * step] = u8c(q2 - a);
}

/// Filters `count` positions along an edge. `at` is the first q0 pixel,
/// `step` crosses the edge, and `along` walks it.
#[allow(clippy::too_many_arguments)]
fn normal_edge(
    buf: &mut [u8],
    mut at: usize,
    step: usize,
    along: usize,
    count: usize,
    edge: i32,
    interior: i32,
    hev_t: i32,
    macroblock: bool,
) {
    for _ in 0..count {
        if normal_ok(buf, at, step, edge, interior) {
            let high = hev(buf, at, step, hev_t);
            if macroblock {
                if high {
                    common(buf, at, step, true);
                } else {
                    mb_edge(buf, at, step);
                }
            } else {
                common(buf, at, step, high);
            }
        }
        at += along;
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
