//! The loop filter of RFC 6386 section 15.
//!
//! It runs over the whole reconstructed frame, macroblock by macroblock in
//! raster order. For each: the left macroblock edge, the inner vertical
//! edges, the top macroblock edge, then the inner horizontal edges. Every
//! edge is filtered one line of pixels across it at a time.

/// The filter strength of one macroblock.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Strength {
    /// The filter level, 0 to 63. Zero leaves the macroblock alone.
    pub level: u8,
    pub interior: u8,
    pub hev_threshold: u8,
    /// Filter the edges inside the macroblock too.
    pub inner: bool,
}

/// Interior limit and high-edge-variance threshold for a key frame
/// macroblock at `level` (section 15.2).
pub(crate) fn strength(level: u8, sharpness: u8, inner: bool) -> Strength {
    let mut interior = level;
    if sharpness > 0 {
        interior >>= if sharpness > 4 { 2 } else { 1 };
        interior = interior.min(9 - sharpness.min(9));
    }
    let hev_threshold = if level >= 40 {
        2
    } else {
        u8::from(level >= 15)
    };
    Strength {
        level,
        interior: interior.max(1),
        hev_threshold,
        inner,
    }
}

/// A plane and where its edge pixels sit: `at` is `q0`, `across` steps
/// from `p0` to `q0`.
struct Line<'a> {
    plane: &'a mut [u8],
    at: usize,
    across: usize,
}

impl Line<'_> {
    /// Pixel `offset` steps across the edge from `q0` (`-1` is `p0`).
    fn get(&self, offset: isize) -> i32 {
        self.index(offset)
            .and_then(|index| self.plane.get(index))
            .map_or(0, |&v| i32::from(v))
    }

    fn set(&mut self, offset: isize, value: i32) {
        if let Some(index) = self.index(offset)
            && let Some(pixel) = self.plane.get_mut(index)
        {
            *pixel = u8::try_from(value.clamp(0, 255)).unwrap_or(u8::MAX);
        }
    }

    fn index(&self, offset: isize) -> Option<usize> {
        let step = self.across.checked_mul(offset.unsigned_abs())?;
        if offset < 0 {
            self.at.checked_sub(step)
        } else {
            self.at.checked_add(step)
        }
    }
}

fn s8(value: i32) -> i32 {
    value.clamp(-128, 127)
}

fn simple_ok(line: &Line<'_>, limit: i32) -> bool {
    let (p1, p0, q0, q1) = (line.get(-2), line.get(-1), line.get(0), line.get(1));
    (p0 - q0).abs() * 2 + ((p1 - q1).abs() >> 1) <= limit
}

fn normal_ok(line: &Line<'_>, edge: i32, interior: i32) -> bool {
    if !simple_ok(line, edge) {
        return false;
    }
    let px: [i32; 8] = [-4, -3, -2, -1, 0, 1, 2, 3].map(|o| line.get(o));
    // p3-p2, p2-p1, p1-p0, then q1-q0, q2-q1, q3-q2.
    [(0, 1), (1, 2), (2, 3), (5, 4), (6, 5), (7, 6)]
        .iter()
        .all(|&(a, b)| (px[a] - px[b]).abs() <= interior)
}

fn high_variance(line: &Line<'_>, threshold: i32) -> bool {
    (line.get(-2) - line.get(-1)).abs() > threshold || (line.get(1) - line.get(0)).abs() > threshold
}

/// The common adjustment: `p0` and `q0` move toward each other; without
/// the outer taps, `p1` and `q1` follow by half as much.
fn common(line: &mut Line<'_>, outer_taps: bool) {
    let (p1, p0, q0, q1) = (line.get(-2), line.get(-1), line.get(0), line.get(1));
    let mut a = 3 * (q0 - p0);
    if outer_taps {
        a += s8(p1 - q1);
    }
    let a = s8(a);
    let f1 = s8(a + 4) >> 3;
    let f2 = s8(a + 3) >> 3;
    line.set(-1, p0 + f2);
    line.set(0, q0 - f1);
    if !outer_taps {
        let half = (f1 + 1) >> 1;
        line.set(-2, p1 + half);
        line.set(1, q1 - half);
    }
}

/// The wider macroblock-edge adjustment over three pixels each side.
fn macroblock(line: &mut Line<'_>) {
    let px: [i32; 6] = [-3, -2, -1, 0, 1, 2].map(|o| line.get(o));
    let w = s8(s8(px[1] - px[4]) + 3 * (px[3] - px[2]));
    for (weight, offset) in [(27, 1_isize), (18, 2), (9, 3)] {
        let a = s8((weight * w + 63) >> 7);
        let p = px[3 - offset.unsigned_abs()];
        let q = px[2 + offset.unsigned_abs()];
        line.set(-offset, p + a);
        line.set(offset - 1, q - a);
    }
}

/// Which kind of edge, for the normal filter.
#[derive(Clone, Copy)]
enum Edge {
    Macroblock,
    Inner,
}

/// Filter `count` lines across one edge. `at` is the first `q0`, `across`
/// steps over the edge, `along` steps to the next line.
fn normal_edge(
    plane: &mut [u8],
    at: usize,
    across: usize,
    along: usize,
    count: usize,
    edge: Edge,
    strength: Strength,
) {
    let level = i32::from(strength.level);
    let interior = i32::from(strength.interior);
    let hev = i32::from(strength.hev_threshold);
    let limit = match edge {
        Edge::Macroblock => (level + 2) * 2 + interior,
        Edge::Inner => level * 2 + interior,
    };
    for n in 0..count {
        let mut line = Line {
            plane: &mut *plane,
            at: at + n * along,
            across,
        };
        if !normal_ok(&line, limit, interior) {
            continue;
        }
        let high = high_variance(&line, hev);
        match edge {
            Edge::Macroblock if !high => macroblock(&mut line),
            Edge::Macroblock => common(&mut line, true),
            Edge::Inner => common(&mut line, high),
        }
    }
}

fn simple_edge(plane: &mut [u8], at: usize, across: usize, along: usize, limit: i32) {
    for n in 0..16 {
        let mut line = Line {
            plane: &mut *plane,
            at: at + n * along,
            across,
        };
        if simple_ok(&line, limit) {
            common(&mut line, true);
        }
    }
}

/// The planes of one frame, macroblock-aligned.
pub(crate) struct Planes<'a> {
    pub y: &'a mut [u8],
    pub u: &'a mut [u8],
    pub v: &'a mut [u8],
    pub mb_cols: usize,
    pub mb_rows: usize,
}

/// Run the loop filter over the whole frame. `strengths` holds one entry
/// per macroblock in raster order.
pub(crate) fn filter_frame(planes: &mut Planes<'_>, strengths: &[Strength], simple: bool) {
    let y_stride = planes.mb_cols * 16;
    let uv_stride = planes.mb_cols * 8;
    for row in 0..planes.mb_rows {
        for col in 0..planes.mb_cols {
            let Some(&strength) = strengths.get(row * planes.mb_cols + col) else {
                continue;
            };
            if strength.level == 0 {
                continue;
            }
            let y = row * 16 * y_stride + col * 16;
            if simple {
                filter_simple(planes.y, y, y_stride, row, col, strength);
                continue;
            }
            let uv = row * 8 * uv_stride + col * 8;
            for (plane, at, stride, size) in [
                (&mut *planes.y, y, y_stride, 16),
                (&mut *planes.u, uv, uv_stride, 8),
                (&mut *planes.v, uv, uv_stride, 8),
            ] {
                if col > 0 {
                    normal_edge(plane, at, 1, stride, size, Edge::Macroblock, strength);
                }
                if strength.inner {
                    for x in (4..size).step_by(4) {
                        normal_edge(plane, at + x, 1, stride, size, Edge::Inner, strength);
                    }
                }
                if row > 0 {
                    normal_edge(plane, at, stride, 1, size, Edge::Macroblock, strength);
                }
                if strength.inner {
                    for y in (4..size).step_by(4) {
                        normal_edge(
                            plane,
                            at + y * stride,
                            stride,
                            1,
                            size,
                            Edge::Inner,
                            strength,
                        );
                    }
                }
            }
        }
    }
}

fn filter_simple(
    plane: &mut [u8],
    at: usize,
    stride: usize,
    row: usize,
    col: usize,
    strength: Strength,
) {
    let level = i32::from(strength.level);
    let interior = i32::from(strength.interior);
    let edge_limit = (level + 2) * 2 + interior;
    let inner_limit = level * 2 + interior;
    if col > 0 {
        simple_edge(plane, at, 1, stride, edge_limit);
    }
    if strength.inner {
        for x in [4, 8, 12] {
            simple_edge(plane, at + x, 1, stride, inner_limit);
        }
    }
    if row > 0 {
        simple_edge(plane, at, stride, 1, edge_limit);
    }
    if strength.inner {
        for y in [4, 8, 12] {
            simple_edge(plane, at + y * stride, stride, 1, inner_limit);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strength_follows_level_and_sharpness() {
        assert_eq!(strength(10, 0, true).interior, 10);
        assert_eq!(strength(10, 0, true).hev_threshold, 0);
        assert_eq!(strength(15, 0, true).hev_threshold, 1);
        assert_eq!(strength(40, 0, true).hev_threshold, 2);
        // Sharpness 3 halves and caps at 6; sharpness 6 quarters and caps at 3.
        assert_eq!(strength(30, 3, true).interior, 6);
        assert_eq!(strength(30, 6, true).interior, 3);
        assert_eq!(strength(1, 7, true).interior, 1);
    }

    #[test]
    fn a_small_step_is_smoothed_and_a_real_edge_is_kept() {
        // One row of a 16-wide plane with a step of 6 at the inner edge x = 8.
        let mut plane = [100_u8; 16 * 16];
        for row in plane.as_chunks_mut::<16>().0 {
            row[8..].fill(106);
        }
        let s = strength(20, 0, true);
        normal_edge(&mut plane, 8, 1, 16, 16, Edge::Inner, s);
        let row = &plane[..16];
        assert!(row[7] > 100 && row[8] < 106, "{row:?}");
        assert!(row.windows(2).all(|w| w[0] <= w[1]), "{row:?}");
        // A step far above the limit is a picture edge: untouched.
        let mut sharp = [0_u8; 16 * 16];
        for row in sharp.as_chunks_mut::<16>().0 {
            row[8..].fill(200);
        }
        let before = sharp;
        normal_edge(&mut sharp, 8, 1, 16, 16, Edge::Macroblock, s);
        assert_eq!(sharp, before);
    }

    #[test]
    fn the_macroblock_filter_spreads_over_three_pixels() {
        let mut plane = [0_u8; 8];
        plane[..4].fill(100);
        plane[4..].fill(104);
        let mut line = Line {
            plane: &mut plane,
            at: 4,
            across: 1,
        };
        macroblock(&mut line);
        // w = (p1 - q1) + 3 * (q0 - p0) = 8: a = 2, then 1, then 1.
        assert_eq!(plane, [100, 101, 101, 102, 102, 103, 103, 104]);
    }
}
