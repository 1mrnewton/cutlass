//! The loop filters across one edge (clause 8.8.2, libvpx
//! `vpx_dsp/loopfilter.c`), for any bit depth: thresholds and sample
//! offsets are those of 8 bits shifted by `bd - 8`.

use super::pixel::Pixel;

/// The thresholds of one filter level, at 8 bits.
#[derive(Clone, Copy, Default, Debug)]
pub struct Thresh {
    pub blimit: u8,
    pub limit: u8,
    pub hev: u8,
}

/// Which filter: 4, 8 or 16 samples wide (taps on each side 4, 4, 8).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Wd {
    W4,
    W8,
    W16,
}

/// Filters `n` lines across one edge. Line `i` has its first sample past
/// the edge (`q0`) at `at + i * line`, and its samples step by `step`
/// (1 for a vertical edge, the stride for a horizontal one).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn filter<P: Pixel>(
    buf: &mut [P],
    at: usize,
    step: usize,
    line: usize,
    n: usize,
    wd: Wd,
    t: Thresh,
    bd: u32,
) {
    let sh = bd - 8;
    let blimit = i32::from(t.blimit) << sh;
    let limit = i32::from(t.limit) << sh;
    let hev_t = i32::from(t.hev) << sh;
    let one = 1i32 << sh;
    let half = 0x80i32 << sh;
    let (lo, hi) = (-(128i32 << sh), (128i32 << sh) - 1);
    let clamp = |v: i32| v.clamp(lo, hi);
    for i in 0..n {
        let q0i = at + i * line;
        let px = |k: isize| -> i32 { buf[(q0i as isize + k * step as isize) as usize].get() };
        let (p3, p2, p1, p0) = (px(-4), px(-3), px(-2), px(-1));
        let (q0, q1, q2, q3) = (px(0), px(1), px(2), px(3));
        let mask = (p3 - p2).abs() <= limit
            && (p2 - p1).abs() <= limit
            && (p1 - p0).abs() <= limit
            && (q1 - q0).abs() <= limit
            && (q2 - q1).abs() <= limit
            && (q3 - q2).abs() <= limit
            && (p0 - q0).abs() * 2 + (p1 - q1).abs() / 2 <= blimit;
        if !mask {
            continue;
        }
        let flat = wd != Wd::W4
            && (p1 - p0).abs() <= one
            && (q1 - q0).abs() <= one
            && (p2 - p0).abs() <= one
            && (q2 - q0).abs() <= one
            && (p3 - p0).abs() <= one
            && (q3 - q0).abs() <= one;
        let set = |buf: &mut [P], k: isize, v: i32| {
            buf[(q0i as isize + k * step as isize) as usize] = P::new(v);
        };
        if flat && wd == Wd::W16 {
            let (p7, p6, p5, p4) = (px(-8), px(-7), px(-6), px(-5));
            let (q4, q5, q6, q7) = (px(4), px(5), px(6), px(7));
            let flat2 = (p4 - p0).abs() <= one
                && (q4 - q0).abs() <= one
                && (p5 - p0).abs() <= one
                && (q5 - q0).abs() <= one
                && (p6 - p0).abs() <= one
                && (q6 - q0).abs() <= one
                && (p7 - p0).abs() <= one
                && (q7 - q0).abs() <= one;
            if flat2 {
                let s = [
                    p7, p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6, q7,
                ];
                // Running sum of the 15-tap filter, as libvpx writes it out.
                let mut sum = p7 * 7 + p6 * 2 + p5 + p4 + p3 + p2 + p1 + p0 + q0;
                let mut out = [0i32; 14];
                out[0] = (sum + 8) >> 4;
                for k in 1usize..14 {
                    // Output k (p6 + ... ) moves the window: drop the
                    // oldest left tap, add the next right tap.
                    let drop_a = s[k.saturating_sub(7)];
                    let drop_b = s[k];
                    let add_a = s[k + 1];
                    let add_b = s[(k + 8).min(15)];
                    sum = sum - drop_a - drop_b + add_a + add_b;
                    out[k] = (sum + 8) >> 4;
                }
                for (k, v) in out.iter().enumerate() {
                    set(buf, k as isize - 7, *v);
                }
                continue;
            }
        }
        if flat {
            set(buf, -3, (p3 + p3 + p3 + 2 * p2 + p1 + p0 + q0 + 4) >> 3);
            set(buf, -2, (p3 + p3 + p2 + 2 * p1 + p0 + q0 + q1 + 4) >> 3);
            set(buf, -1, (p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2 + 4) >> 3);
            set(buf, 0, (p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3 + 4) >> 3);
            set(buf, 1, (p1 + p0 + q0 + 2 * q1 + q2 + q3 + q3 + 4) >> 3);
            set(buf, 2, (p0 + q0 + q1 + 2 * q2 + q3 + q3 + q3 + 4) >> 3);
            continue;
        }
        let ps1 = p1 - half;
        let ps0 = p0 - half;
        let qs0 = q0 - half;
        let qs1 = q1 - half;
        let hev = (p1 - p0).abs() > hev_t || (q1 - q0).abs() > hev_t;
        let mut f = if hev { clamp(ps1 - qs1) } else { 0 };
        f = clamp(f + 3 * (qs0 - ps0));
        let f1 = clamp(f + 4) >> 3;
        let f2 = clamp(f + 3) >> 3;
        set(buf, 0, clamp(qs0 - f1) + half);
        set(buf, -1, clamp(ps0 + f2) + half);
        if !hev {
            let f = (f1 + 1) >> 1;
            set(buf, 1, clamp(qs1 - f) + half);
            set(buf, -2, clamp(ps1 + f) + half);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Thresh, Wd, filter};

    /// The running sum against the 15-tap filter written out term by term.
    #[test]
    fn wide_filter_matches_its_definition() {
        let s: [i32; 16] = [
            10, 11, 10, 11, 10, 11, 10, 11, 11, 10, 11, 10, 11, 10, 11, 10,
        ];
        let (p7, p6, p5, p4, p3, p2, p1, p0) = (s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]);
        let (q0, q1, q2, q3, q4, q5, q6, q7) =
            (s[8], s[9], s[10], s[11], s[12], s[13], s[14], s[15]);
        let r = |v: i32| (v + 8) >> 4;
        let want = [
            r(p7 * 7 + p6 * 2 + p5 + p4 + p3 + p2 + p1 + p0 + q0),
            r(p7 * 6 + p6 + p5 * 2 + p4 + p3 + p2 + p1 + p0 + q0 + q1),
            r(p7 * 5 + p6 + p5 + p4 * 2 + p3 + p2 + p1 + p0 + q0 + q1 + q2),
            r(p7 * 4 + p6 + p5 + p4 + p3 * 2 + p2 + p1 + p0 + q0 + q1 + q2 + q3),
            r(p7 * 3 + p6 + p5 + p4 + p3 + p2 * 2 + p1 + p0 + q0 + q1 + q2 + q3 + q4),
            r(p7 * 2 + p6 + p5 + p4 + p3 + p2 + p1 * 2 + p0 + q0 + q1 + q2 + q3 + q4 + q5),
            r(p7 + p6 + p5 + p4 + p3 + p2 + p1 + p0 * 2 + q0 + q1 + q2 + q3 + q4 + q5 + q6),
            r(p6 + p5 + p4 + p3 + p2 + p1 + p0 + q0 * 2 + q1 + q2 + q3 + q4 + q5 + q6 + q7),
            r(p5 + p4 + p3 + p2 + p1 + p0 + q0 + q1 * 2 + q2 + q3 + q4 + q5 + q6 + q7 * 2),
            r(p4 + p3 + p2 + p1 + p0 + q0 + q1 + q2 * 2 + q3 + q4 + q5 + q6 + q7 * 3),
            r(p3 + p2 + p1 + p0 + q0 + q1 + q2 + q3 * 2 + q4 + q5 + q6 + q7 * 4),
            r(p2 + p1 + p0 + q0 + q1 + q2 + q3 + q4 * 2 + q5 + q6 + q7 * 5),
            r(p1 + p0 + q0 + q1 + q2 + q3 + q4 + q5 * 2 + q6 + q7 * 6),
            r(p0 + q0 + q1 + q2 + q3 + q4 + q5 + q6 * 2 + q7 * 7),
        ];
        let mut buf: Vec<u8> = s.iter().map(|&v| v as u8).collect();
        filter(
            &mut buf,
            8,
            1,
            16,
            1,
            Wd::W16,
            Thresh {
                blimit: 60,
                limit: 10,
                hev: 2,
            },
            8,
        );
        let got: Vec<i32> = buf[1..15].iter().map(|&v| i32::from(v)).collect();
        assert_eq!(got, want);
    }
}
