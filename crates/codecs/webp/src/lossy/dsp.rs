//! Inverse transforms (RFC 6386 section 14) and the loop filter (section
//! 15), in the integer arithmetic of the reference decoder.

use std::num::Wrapping;

// The arithmetic wraps in 32 bits, as it does in libwebp's C on the
// machines it runs on. A legal stream never gets there; a hostile one must
// not panic, and still decodes the way `dwebp` decodes it.
fn mul1(a: Wrapping<i32>) -> Wrapping<i32> {
    ((a * Wrapping(20_091)) >> 16) + a
}

fn mul2(a: Wrapping<i32>) -> Wrapping<i32> {
    (a * Wrapping(35_468)) >> 16
}

/// Section 14.3: add the inverse DCT of `coeffs` to the 4x4 block at `at`.
pub(crate) fn add_idct(coeffs: &[i16; 16], buf: &mut [u8], at: usize, stride: usize) {
    let mut tmp = [Wrapping(0i32); 16];
    for col in 0..4 {
        let c0 = Wrapping(i32::from(coeffs[col]));
        let c1 = Wrapping(i32::from(coeffs[4 + col]));
        let c2 = Wrapping(i32::from(coeffs[8 + col]));
        let c3 = Wrapping(i32::from(coeffs[12 + col]));
        let a = c0 + c2;
        let b = c0 - c2;
        let c = mul2(c1) - mul1(c3);
        let d = mul1(c1) + mul2(c3);
        tmp[col * 4] = a + d;
        tmp[col * 4 + 1] = b + c;
        tmp[col * 4 + 2] = b - c;
        tmp[col * 4 + 3] = a - d;
    }
    for row in 0..4 {
        let dc = tmp[row] + Wrapping(4);
        let a = dc + tmp[8 + row];
        let b = dc - tmp[8 + row];
        let c = mul2(tmp[4 + row]) - mul1(tmp[12 + row]);
        let d = mul1(tmp[4 + row]) + mul2(tmp[12 + row]);
        let line = &mut buf[at + row * stride..at + row * stride + 4];
        for (pixel, value) in line.iter_mut().zip([a + d, b + c, b - c, a - d]) {
            *pixel = (i32::from(*pixel) + (value.0 >> 3)).clamp(0, 255) as u8;
        }
    }
}

/// Section 14.3: the inverse Walsh-Hadamard transform of the Y2 block.
/// Returns the DC of each of the sixteen luma blocks in raster order.
pub(crate) fn inverse_wht(input: &[i16; 16]) -> [i16; 16] {
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let a0 = i32::from(input[i]) + i32::from(input[12 + i]);
        let a1 = i32::from(input[4 + i]) + i32::from(input[8 + i]);
        let a2 = i32::from(input[4 + i]) - i32::from(input[8 + i]);
        let a3 = i32::from(input[i]) - i32::from(input[12 + i]);
        tmp[i] = a0 + a1;
        tmp[8 + i] = a0 - a1;
        tmp[4 + i] = a3 + a2;
        tmp[12 + i] = a3 - a2;
    }
    let mut out = [0i16; 16];
    for i in 0..4 {
        let dc = tmp[i * 4] + 3;
        let a0 = dc + tmp[i * 4 + 3];
        let a1 = tmp[i * 4 + 1] + tmp[i * 4 + 2];
        let a2 = tmp[i * 4 + 1] - tmp[i * 4 + 2];
        let a3 = dc - tmp[i * 4 + 3];
        // Stored in 16 bits, as the reference decoder does.
        out[i * 4] = ((a0 + a1) >> 3) as i16;
        out[i * 4 + 1] = ((a3 + a2) >> 3) as i16;
        out[i * 4 + 2] = ((a0 - a1) >> 3) as i16;
        out[i * 4 + 3] = ((a3 - a2) >> 3) as i16;
    }
    out
}

/// Clamp to a signed byte, the `c()` of section 15.2.
fn sclip1(v: i32) -> i32 {
    v.clamp(-128, 127)
}

fn sclip2(v: i32) -> i32 {
    v.clamp(-16, 15)
}

fn clip_pixel(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// One line of pixels across an edge: `p3 p2 p1 p0 | q0 q1 q2 q3`, the
/// edge between `at - step` and `at`.
struct Line {
    at: usize,
    step: usize,
}

impl Line {
    fn get(&self, buf: &[u8], offset: isize) -> i32 {
        let index = self
            .at
            .wrapping_add_signed(offset * self.step.cast_signed());
        i32::from(buf[index])
    }

    fn set(&self, buf: &mut [u8], offset: isize, value: i32) {
        let index = self
            .at
            .wrapping_add_signed(offset * self.step.cast_signed());
        buf[index] = clip_pixel(value);
    }
}

fn filter2(buf: &mut [u8], line: &Line) {
    let p1 = line.get(buf, -2);
    let p0 = line.get(buf, -1);
    let q0 = line.get(buf, 0);
    let q1 = line.get(buf, 1);
    let a = 3 * (q0 - p0) + sclip1(p1 - q1);
    let a1 = sclip2((a + 4) >> 3);
    let a2 = sclip2((a + 3) >> 3);
    line.set(buf, -1, p0 + a2);
    line.set(buf, 0, q0 - a1);
}

fn filter4(buf: &mut [u8], line: &Line) {
    let p1 = line.get(buf, -2);
    let p0 = line.get(buf, -1);
    let q0 = line.get(buf, 0);
    let q1 = line.get(buf, 1);
    let a = 3 * (q0 - p0);
    let a1 = sclip2((a + 4) >> 3);
    let a2 = sclip2((a + 3) >> 3);
    let a3 = (a1 + 1) >> 1;
    line.set(buf, -2, p1 + a3);
    line.set(buf, -1, p0 + a2);
    line.set(buf, 0, q0 - a1);
    line.set(buf, 1, q1 - a3);
}

fn filter6(buf: &mut [u8], line: &Line) {
    let p2 = line.get(buf, -3);
    let p1 = line.get(buf, -2);
    let p0 = line.get(buf, -1);
    let q0 = line.get(buf, 0);
    let q1 = line.get(buf, 1);
    let q2 = line.get(buf, 2);
    let a = sclip1(3 * (q0 - p0) + sclip1(p1 - q1));
    let a1 = (27 * a + 63) >> 7;
    let a2 = (18 * a + 63) >> 7;
    let a3 = (9 * a + 63) >> 7;
    line.set(buf, -3, p2 + a3);
    line.set(buf, -2, p1 + a2);
    line.set(buf, -1, p0 + a1);
    line.set(buf, 0, q0 - a1);
    line.set(buf, 1, q1 - a2);
    line.set(buf, 2, q2 - a3);
}

fn high_variance(buf: &[u8], line: &Line, threshold: i32) -> bool {
    let p1 = line.get(buf, -2);
    let p0 = line.get(buf, -1);
    let q0 = line.get(buf, 0);
    let q1 = line.get(buf, 1);
    (p1 - p0).abs() > threshold || (q1 - q0).abs() > threshold
}

fn edge_ok(buf: &[u8], line: &Line, limit2: i32) -> bool {
    let p1 = line.get(buf, -2);
    let p0 = line.get(buf, -1);
    let q0 = line.get(buf, 0);
    let q1 = line.get(buf, 1);
    4 * (p0 - q0).abs() + (p1 - q1).abs() <= limit2
}

fn normal_ok(buf: &[u8], line: &Line, limit2: i32, interior: i32) -> bool {
    if !edge_ok(buf, line, limit2) {
        return false;
    }
    let v: [i32; 8] = std::array::from_fn(|i| line.get(buf, i as isize - 4));
    v.windows(2)
        .enumerate()
        .filter(|(i, _)| *i != 3)
        .all(|(_, pair)| (pair[1] - pair[0]).abs() <= interior)
}

/// One edge of `length` pixels. `step` crosses the edge and `pitch` runs
/// along it.
#[derive(Clone, Copy)]
pub(crate) struct Edge {
    pub at: usize,
    pub step: usize,
    pub pitch: usize,
    pub length: usize,
}

impl Edge {
    fn lines(self) -> impl Iterator<Item = Line> {
        (0..self.length).map(move |i| Line {
            at: self.at + i * self.pitch,
            step: self.step,
        })
    }
}

/// Section 15.2: the simple filter on one luma edge.
pub(crate) fn simple(buf: &mut [u8], edge: Edge, limit: i32) {
    let limit2 = 2 * limit + 1;
    for line in edge.lines() {
        if edge_ok(buf, &line, limit2) {
            filter2(buf, &line);
        }
    }
}

/// Section 15.3: the normal filter on one edge. A macroblock edge changes
/// up to three pixels each side, a sub-block edge two.
pub(crate) fn normal(
    buf: &mut [u8],
    edge: Edge,
    limit: i32,
    interior: i32,
    hev: i32,
    macroblock_edge: bool,
) {
    let limit2 = 2 * limit + 1;
    for line in edge.lines() {
        if normal_ok(buf, &line, limit2, interior) {
            if high_variance(buf, &line, hev) {
                filter2(buf, &line);
            } else if macroblock_edge {
                filter6(buf, &line);
            } else {
                filter4(buf, &line);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{add_idct, inverse_wht};

    #[test]
    fn coefficients_at_the_sixteen_bit_edges_wrap_and_do_not_panic() {
        for value in [i16::MAX, i16::MIN] {
            let mut buf = [128u8; 16];
            add_idct(&[value; 16], &mut buf, 0, 4);
            let mut alternating = [value; 16];
            for (index, c) in alternating.iter_mut().enumerate() {
                if index % 3 == 0 {
                    *c = c.wrapping_neg();
                }
            }
            add_idct(&alternating, &mut buf, 0, 4);
            let _ = inverse_wht(&[value; 16]);
        }
    }

    #[test]
    fn a_lone_dc_adds_a_flat_offset() {
        let mut coeffs = [0i16; 16];
        coeffs[0] = 80;
        let mut buf = [100u8; 16];
        add_idct(&coeffs, &mut buf, 0, 4);
        assert_eq!(buf, [110; 16]);
    }

    #[test]
    fn a_lone_y2_dc_spreads_to_every_block() {
        let mut input = [0i16; 16];
        input[0] = 21;
        assert_eq!(inverse_wht(&input), [3; 16]);
    }
}
