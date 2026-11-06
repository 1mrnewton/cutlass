//! Intra prediction, RFC 6386 section 12.
//!
//! Every predictor writes a block at `at` in a buffer whose row pitch is
//! `stride`, reading the row above and the column to the left from the same
//! buffer. The caller fills those edges, with 127 above the picture and 129
//! left of it, before predicting.

/// Sub-block modes in the order of section 11.2 (`B_DC_PRED` first). A
/// whole-block mode uses the first four numbers: DC, TM, V, H.
pub(crate) const B_DC: u8 = 0;
pub(crate) const B_TM: u8 = 1;
pub(crate) const B_VE: u8 = 2;
pub(crate) const B_HE: u8 = 3;
const B_LD: u8 = 4;
const B_RD: u8 = 5;
const B_VR: u8 = 6;
const B_VL: u8 = 7;
const B_HD: u8 = 8;

fn avg2(a: u8, b: u8) -> u8 {
    ((u16::from(a) + u16::from(b) + 1) >> 1) as u8
}

fn avg3(a: u8, b: u8, c: u8) -> u8 {
    ((u16::from(a) + 2 * u16::from(b) + u16::from(c) + 2) >> 2) as u8
}

fn fill(buf: &mut [u8], at: usize, stride: usize, size: usize, value: u8) {
    for row in 0..size {
        buf[at + row * stride..at + row * stride + size].fill(value);
    }
}

fn true_motion(buf: &mut [u8], at: usize, stride: usize, size: usize) {
    let corner = i32::from(buf[at - stride - 1]);
    for row in 0..size {
        let left = i32::from(buf[at + row * stride - 1]) - corner;
        for col in 0..size {
            let above = i32::from(buf[at - stride + col]);
            buf[at + row * stride + col] = (above + left).clamp(0, 255) as u8;
        }
    }
}

/// Which edges a whole-block DC prediction may read.
#[derive(Clone, Copy)]
pub(crate) struct Edges {
    pub above: bool,
    pub left: bool,
}

/// A 16x16 luma or 8x8 chroma block, `size` being 16 or 8.
pub(crate) fn block(buf: &mut [u8], at: usize, stride: usize, size: usize, mode: u8, edges: Edges) {
    match mode {
        B_TM => true_motion(buf, at, stride, size),
        B_VE => {
            for row in 0..size {
                buf.copy_within(at - stride..at - stride + size, at + row * stride);
            }
        }
        B_HE => {
            for row in 0..size {
                let left = buf[at + row * stride - 1];
                fill_row(buf, at + row * stride, size, left);
            }
        }
        _ => {
            let shift = size.trailing_zeros();
            let above: u32 = (0..size).map(|col| u32::from(buf[at - stride + col])).sum();
            let left: u32 = (0..size)
                .map(|row| u32::from(buf[at + row * stride - 1]))
                .sum();
            let value = match (edges.above, edges.left) {
                (true, true) => (above + left + (1 << shift)) >> (shift + 1),
                (true, false) => (above + (1 << (shift - 1))) >> shift,
                (false, true) => (left + (1 << (shift - 1))) >> shift,
                (false, false) => 128,
            };
            fill(buf, at, stride, size, value as u8);
        }
    }
}

fn fill_row(buf: &mut [u8], at: usize, size: usize, value: u8) {
    buf[at..at + size].fill(value);
}

/// One 4x4 luma sub-block. The row above extends four pixels to the right.
pub(crate) fn sub_block(buf: &mut [u8], at: usize, stride: usize, mode: u8) {
    let top = |buf: &[u8], col: usize| buf[at - stride + col];
    let left = |buf: &[u8], row: usize| buf[at + row * stride - 1];
    let corner = buf[at - stride - 1];
    let a: [u8; 8] = std::array::from_fn(|col| top(buf, col));
    let l: [u8; 4] = std::array::from_fn(|row| left(buf, row));
    let mut out = [[0u8; 4]; 4];
    match mode {
        B_DC => {
            let sum: u32 = a[..4]
                .iter()
                .chain(&l)
                .map(|&value| u32::from(value))
                .sum::<u32>();
            out = [[((sum + 4) >> 3) as u8; 4]; 4];
        }
        B_TM => {
            true_motion(buf, at, stride, 4);
            return;
        }
        B_VE => {
            let row = [
                avg3(corner, a[0], a[1]),
                avg3(a[0], a[1], a[2]),
                avg3(a[1], a[2], a[3]),
                avg3(a[2], a[3], a[4]),
            ];
            out = [row; 4];
        }
        B_HE => {
            out[0] = [avg3(corner, l[0], l[1]); 4];
            out[1] = [avg3(l[0], l[1], l[2]); 4];
            out[2] = [avg3(l[1], l[2], l[3]); 4];
            out[3] = [avg3(l[2], l[3], l[3]); 4];
        }
        B_LD => {
            for (row, line) in out.iter_mut().enumerate() {
                for (col, pixel) in line.iter_mut().enumerate() {
                    let i = row + col;
                    *pixel = if i == 6 {
                        avg3(a[6], a[7], a[7])
                    } else {
                        avg3(a[i], a[i + 1], a[i + 2])
                    };
                }
            }
        }
        B_RD => {
            // The edge from bottom-left, up the left column, across the top.
            let e = [l[3], l[2], l[1], l[0], corner, a[0], a[1], a[2], a[3]];
            for (row, line) in out.iter_mut().enumerate() {
                for (col, pixel) in line.iter_mut().enumerate() {
                    let i = 3 - row + col;
                    *pixel = avg3(e[i], e[i + 1], e[i + 2]);
                }
            }
        }
        B_VR => {
            let e = [l[3], l[2], l[1], l[0], corner, a[0], a[1], a[2], a[3]];
            out[3][0] = avg3(e[1], e[2], e[3]);
            out[2][0] = avg3(e[2], e[3], e[4]);
            out[3][1] = avg3(e[3], e[4], e[5]);
            out[1][0] = out[3][1];
            out[2][1] = avg2(e[4], e[5]);
            out[0][0] = out[2][1];
            out[3][2] = avg3(e[4], e[5], e[6]);
            out[1][1] = out[3][2];
            out[2][2] = avg2(e[5], e[6]);
            out[0][1] = out[2][2];
            out[3][3] = avg3(e[5], e[6], e[7]);
            out[1][2] = out[3][3];
            out[2][3] = avg2(e[6], e[7]);
            out[0][2] = out[2][3];
            out[1][3] = avg3(e[6], e[7], e[8]);
            out[0][3] = avg2(e[7], e[8]);
        }
        B_VL => {
            out[0][0] = avg2(a[0], a[1]);
            out[1][0] = avg3(a[0], a[1], a[2]);
            out[2][0] = avg2(a[1], a[2]);
            out[0][1] = out[2][0];
            out[1][1] = avg3(a[1], a[2], a[3]);
            out[3][0] = out[1][1];
            out[2][1] = avg2(a[2], a[3]);
            out[0][2] = out[2][1];
            out[3][1] = avg3(a[2], a[3], a[4]);
            out[1][2] = out[3][1];
            out[2][2] = avg2(a[3], a[4]);
            out[0][3] = out[2][2];
            out[3][2] = avg3(a[3], a[4], a[5]);
            out[1][3] = out[3][2];
            out[2][3] = avg3(a[4], a[5], a[6]);
            out[3][3] = avg3(a[5], a[6], a[7]);
        }
        B_HD => {
            let e = [l[3], l[2], l[1], l[0], corner, a[0], a[1], a[2], a[3]];
            out[3][0] = avg2(e[0], e[1]);
            out[3][1] = avg3(e[0], e[1], e[2]);
            out[2][0] = avg2(e[1], e[2]);
            out[3][2] = out[2][0];
            out[2][1] = avg3(e[1], e[2], e[3]);
            out[3][3] = out[2][1];
            out[2][2] = avg2(e[2], e[3]);
            out[1][0] = out[2][2];
            out[2][3] = avg3(e[2], e[3], e[4]);
            out[1][1] = out[2][3];
            out[1][2] = avg2(e[3], e[4]);
            out[0][0] = out[1][2];
            out[1][3] = avg3(e[3], e[4], e[5]);
            out[0][1] = out[1][3];
            out[0][2] = avg3(e[4], e[5], e[6]);
            out[0][3] = avg3(e[5], e[6], e[7]);
        }
        _ => {
            // B_HU_PRED.
            out[0][0] = avg2(l[0], l[1]);
            out[0][1] = avg3(l[0], l[1], l[2]);
            out[0][2] = avg2(l[1], l[2]);
            out[1][0] = out[0][2];
            out[0][3] = avg3(l[1], l[2], l[3]);
            out[1][1] = out[0][3];
            out[1][2] = avg2(l[2], l[3]);
            out[2][0] = out[1][2];
            out[1][3] = avg3(l[2], l[3], l[3]);
            out[2][1] = out[1][3];
            out[2][2] = l[3];
            out[2][3] = l[3];
            out[3] = [l[3]; 4];
        }
    }
    for (row, line) in out.iter().enumerate() {
        buf[at + row * stride..at + row * stride + 4].copy_from_slice(line);
    }
}
