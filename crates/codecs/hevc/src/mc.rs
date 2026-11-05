//! Fractional sample interpolation (clause 8.5.3.3.3) and weighted sample
//! prediction (clause 8.5.3.3.4).

/// Luma 8-tap filter `fL` for quarter positions 1..=3.
const LUMA: [[i32; 8]; 4] = [
    [0, 0, 0, 64, 0, 0, 0, 0],
    [-1, 4, -10, 58, 17, -5, 1, 0],
    [-1, 4, -11, 40, 40, -11, 4, -1],
    [0, 1, -5, 17, 58, -10, 4, -1],
];

/// Chroma 4-tap filter `fC` for eighth positions 1..=7.
const CHROMA: [[i32; 4]; 8] = [
    [0, 64, 0, 0],
    [-2, 58, 10, -2],
    [-4, 54, 16, -2],
    [-6, 46, 28, -4],
    [-4, 36, 36, -4],
    [-4, 28, 46, -6],
    [-2, 16, 54, -4],
    [-2, 10, 58, -2],
];

/// A reference plane.
pub struct Plane<'a> {
    pub samples: &'a [u16],
    pub width: usize,
    pub height: usize,
}

/// Interpolate a `w`×`h` block whose integer origin is (`x`, `y`) at
/// fraction (`fx`, `fy`) into `out` (stride `w`), 14-bit intermediate
/// precision. Luma uses the 8-tap filter in quarters, chroma the 4-tap
/// filter in eighths. Samples outside the plane repeat its edge.
pub fn interpolate(
    plane: &Plane<'_>,
    block: Block,
    bit_depth: u32,
    out: &mut [i16],
    scratch: &mut Scratch,
) {
    if block.luma {
        let taps = [&LUMA[block.fx & 3], &LUMA[block.fy & 3]];
        filter::<8>(plane, block, bit_depth, taps, (out, scratch));
    } else {
        let taps = [&CHROMA[block.fx & 7], &CHROMA[block.fy & 7]];
        filter::<4>(plane, block, bit_depth, taps, (out, scratch));
    }
}

/// Buffers the interpolation reuses from block to block.
pub struct Scratch {
    win: Vec<u16>,
    tmp: Vec<i16>,
}

impl Default for Scratch {
    fn default() -> Self {
        Self {
            win: vec![0; WINDOW * WINDOW],
            tmp: vec![0; WINDOW * 64],
        }
    }
}

/// Largest window a 64×64 block with an 8-tap filter reads, per edge.
const WINDOW: usize = 64 + 7;

fn filter<const T: usize>(
    plane: &Plane<'_>,
    block: Block,
    bit_depth: u32,
    [cx, cy]: [&[i32; T]; 2],
    (out, scratch): (&mut [i16], &mut Scratch),
) {
    let Block {
        x, y, fx, fy, w, h, ..
    } = block;
    let (w, h) = (w.min(64), h.min(64));
    let half = (T / 2 - 1) as isize;
    let shift1 = bit_depth.saturating_sub(8).min(4);
    let shift3 = 14u32.saturating_sub(bit_depth).max(2);
    let sw = w + T - 1;
    let sh = h + T - 1;
    let (pw, ph) = (plane.width as isize, plane.height as isize);
    let (x0, y0) = (x - half, y - half);
    if pw == 0 || ph == 0 || plane.samples.len() < plane.width * plane.height {
        out.iter_mut().for_each(|v| *v = 0);
        return;
    }
    let inside = x0 >= 0 && y0 >= 0 && x0 + sw as isize <= pw && y0 + sh as isize <= ph;
    let Scratch { win, tmp } = scratch;
    // The source rows: the reference itself, or a window with its edges
    // repeated when the block reaches outside.
    let (src, stride, base): (&[u16], usize, usize) = if inside {
        (
            plane.samples,
            plane.width,
            y0 as usize * plane.width + x0 as usize,
        )
    } else {
        let win = &mut win[..sw * sh];
        for j in 0..sh {
            let row = &mut win[j * sw..j * sw + sw];
            let sy = (y0 + j as isize).clamp(0, ph - 1) as usize * plane.width;
            for (i, v) in row.iter_mut().enumerate() {
                let sx = (x0 + i as isize).clamp(0, pw - 1) as usize;
                *v = plane.samples[sy + sx];
            }
        }
        (&*win, sw, 0)
    };
    let row = |j: usize| -> &[u16] { &src[base + j * stride..][..sw] };
    let h0 = T / 2 - 1;
    let out = &mut out[..w * h];
    let mut acc = [0i32; 64];
    let acc = &mut acc[..w];
    match (fx != 0, fy != 0) {
        (false, false) => {
            for (j, dst) in out.chunks_exact_mut(w).enumerate() {
                for (d, &s) in dst.iter_mut().zip(&row(j + h0)[h0..]) {
                    *d = (i32::from(s) << shift3) as i16;
                }
            }
        }
        (true, false) => {
            for (j, dst) in out.chunks_exact_mut(w).enumerate() {
                let line = row(j + h0);
                taps(cx, std::array::from_fn(|k| &line[k..]), acc);
                for (d, &a) in dst.iter_mut().zip(acc.iter()) {
                    *d = (a >> shift1) as i16;
                }
            }
        }
        (false, true) => {
            for (j, dst) in out.chunks_exact_mut(w).enumerate() {
                taps(cy, std::array::from_fn(|k| &row(j + k)[h0..]), acc);
                for (d, &a) in dst.iter_mut().zip(acc.iter()) {
                    *d = (a >> shift1) as i16;
                }
            }
        }
        (true, true) => {
            let tmp = &mut tmp[..sh * w];
            for (j, dst) in tmp.chunks_exact_mut(w).enumerate() {
                let line = row(j);
                taps(cx, std::array::from_fn(|k| &line[k..]), acc);
                for (d, &a) in dst.iter_mut().zip(acc.iter()) {
                    *d = (a >> shift1) as i16;
                }
            }
            let tmp = &*tmp;
            for (j, dst) in out.chunks_exact_mut(w).enumerate() {
                taps(cy, std::array::from_fn(|k| &tmp[(j + k) * w..]), acc);
                for (d, &a) in dst.iter_mut().zip(acc.iter()) {
                    *d = (a >> 6) as i16;
                }
            }
        }
    }
}

/// `out[i]` = the sum over `k` of `c[k] · lines[k][i]`, four outputs at a
/// time with two left over at most (block widths are even).
#[inline(always)]
fn taps<const T: usize, S: Copy>(c: &[i32; T], lines: [&[S]; T], out: &mut [i32])
where
    i32: From<S>,
{
    let (quads, rest) = out.as_chunks_mut::<4>();
    let tail = quads.len() * 4;
    for (q, quad) in quads.iter_mut().enumerate() {
        let i = q * 4;
        let mut acc = [0i32; 4];
        for (&c, line) in c.iter().zip(&lines) {
            let Some(src) = line.get(i..i + 4) else {
                continue;
            };
            for (a, &s) in acc.iter_mut().zip(src) {
                *a += c * i32::from(s);
            }
        }
        *quad = acc;
    }
    for (n, o) in rest.iter_mut().enumerate() {
        *o = c
            .iter()
            .zip(&lines)
            .map(|(&c, line)| line.get(tail + n).map_or(0, |&s| c * i32::from(s)))
            .sum();
    }
}

/// Where and how large one prediction block is in a reference plane.
#[derive(Clone, Copy, Debug)]
pub struct Block {
    pub x: isize,
    pub y: isize,
    pub fx: usize,
    pub fy: usize,
    pub w: usize,
    pub h: usize,
    pub luma: bool,
}

/// Explicit weight of one list: weight, offset (already scaled to the bit
/// depth), and `log2Wd` without the 14-bit shift.
#[derive(Clone, Copy, Debug)]
pub struct Wp {
    pub weight: i32,
    pub offset: i32,
    pub log2_denom: u32,
}

/// Write one predicted block: default or explicit weighting of one or two
/// interpolated blocks (clause 8.5.3.3.4).
pub fn store(
    dst: &mut [u16],
    stride: usize,
    w: usize,
    h: usize,
    bit_depth: u32,
    pred: [Option<&[i16]>; 2],
    weights: Option<[Wp; 2]>,
) {
    let max = (1i32 << bit_depth) - 1;
    let shift1 = 14 - bit_depth;
    let rows = dst.chunks_mut(stride).take(h);
    match (pred, weights) {
        ([Some(a), Some(b)], None) => {
            let shift2 = 15 - bit_depth;
            let add = 1i32 << (shift2 - 1);
            for (row, (a, b)) in rows.zip(a.chunks_exact(w).zip(b.chunks_exact(w))) {
                for (out, (&a, &b)) in row[..w].iter_mut().zip(a.iter().zip(b)) {
                    let v = (i32::from(a) + i32::from(b) + add) >> shift2;
                    *out = v.clamp(0, max) as u16;
                }
            }
        }
        ([Some(a), Some(b)], Some([w0, w1])) => {
            let log2wd = w0.log2_denom + shift1;
            let add = (w0.offset + w1.offset + 1) << log2wd;
            for (row, (a, b)) in rows.zip(a.chunks_exact(w).zip(b.chunks_exact(w))) {
                for (out, (&a, &b)) in row[..w].iter_mut().zip(a.iter().zip(b)) {
                    let v =
                        (i32::from(a) * w0.weight + i32::from(b) * w1.weight + add) >> (log2wd + 1);
                    *out = v.clamp(0, max) as u16;
                }
            }
        }
        ([Some(a), None] | [None, Some(a)], None) => {
            let add = 1i32 << (shift1 - 1);
            for (row, a) in rows.zip(a.chunks_exact(w)) {
                for (out, &a) in row[..w].iter_mut().zip(a) {
                    let v = (i32::from(a) + add) >> shift1;
                    *out = v.clamp(0, max) as u16;
                }
            }
        }
        ([Some(a), None], Some([wp, _])) | ([None, Some(a)], Some([_, wp])) => {
            let log2wd = wp.log2_denom + shift1;
            for (row, a) in rows.zip(a.chunks_exact(w)) {
                for (out, &a) in row[..w].iter_mut().zip(a) {
                    let p = i32::from(a);
                    let v = if log2wd >= 1 {
                        ((p * wp.weight + (1 << (log2wd - 1))) >> log2wd) + wp.offset
                    } else {
                        p * wp.weight + wp.offset
                    };
                    *out = v.clamp(0, max) as u16;
                }
            }
        }
        ([None, None], _) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::{Block, Plane, Scratch, Wp, interpolate, store};

    #[test]
    fn whole_sample_motion_copies_the_reference() {
        let samples: Vec<u16> = (0..64).map(|v| v as u16).collect();
        let plane = Plane {
            samples: &samples,
            width: 8,
            height: 8,
        };
        let mut out = vec![0i16; 4];
        let block = Block {
            x: 2,
            y: 3,
            fx: 0,
            fy: 0,
            w: 2,
            h: 2,
            luma: true,
        };
        interpolate(&plane, block, 8, &mut out, &mut Scratch::default());
        let mut dst = vec![0u16; 4];
        store(&mut dst, 2, 2, 2, 8, [Some(&out), None], None);
        assert_eq!(dst, vec![26, 27, 34, 35]);
    }

    #[test]
    fn half_sample_of_a_flat_plane_is_flat_and_edges_clamp() {
        let samples = vec![100u16; 16];
        let plane = Plane {
            samples: &samples,
            width: 4,
            height: 4,
        };
        let mut out = vec![0i16; 16];
        let block = Block {
            x: -9,
            y: 7,
            fx: 2,
            fy: 3,
            w: 4,
            h: 4,
            luma: true,
        };
        interpolate(&plane, block, 10, &mut out, &mut Scratch::default());
        let mut dst = vec![0u16; 16];
        store(&mut dst, 4, 4, 4, 10, [Some(&out), Some(&out)], None);
        assert!(dst.iter().all(|&v| v == 100));
        let wp = Wp {
            weight: 2,
            offset: 4,
            log2_denom: 1,
        };
        store(&mut dst, 4, 4, 4, 10, [Some(&out), None], Some([wp, wp]));
        assert!(dst.iter().all(|&v| v == 104));
    }
}
