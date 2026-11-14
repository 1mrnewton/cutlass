//! The kernel against the luma filter as clauses 8.7.2.5.3 and 8.7.2.5.7
//! write it, one segment at a time.

use super::{Segment, luma8, luma8_lines, luma16, luma16_lines};

/// The filter of one segment of four lines `l[line][i]`, as the clauses do it.
fn reference(l: &mut [[i32; 8]; 4], s: Segment) {
    reference_at(l, s, 255);
}

/// [`reference`] for samples up to `max`.
fn reference_at(l: &mut [[i32; 8]; 4], s: Segment, max: i32) {
    let (beta, tc) = (i32::from(s.beta), i32::from(s.tc));
    let dp = |v: &[i32; 8]| (v[1] - 2 * v[2] + v[3]).abs();
    let dq = |v: &[i32; 8]| (v[6] - 2 * v[5] + v[4]).abs();
    let (dp0, dp3, dq0, dq3) = (dp(&l[0]), dp(&l[3]), dq(&l[0]), dq(&l[3]));
    let (d0, d3) = (dp0 + dq0, dp3 + dq3);
    if d0 + d3 >= beta {
        return;
    }
    let strong_line = |v: &[i32; 8], d: i32| {
        2 * d < (beta >> 2)
            && (v[0] - v[3]).abs() + (v[4] - v[7]).abs() < (beta >> 3)
            && (v[3] - v[4]).abs() < ((5 * tc + 1) >> 1)
    };
    let clip = |v: i32| v.clamp(0, max);
    if strong_line(&l[0], d0) && strong_line(&l[3], d3) {
        let tc2 = 2 * tc;
        for v in l.iter_mut() {
            let [p3, p2, p1, p0, q0, q1, q2, q3] = *v;
            if !s.no_p {
                v[3] =
                    clip(((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3).clamp(p0 - tc2, p0 + tc2));
                v[2] = clip(((p2 + p1 + p0 + q0 + 2) >> 2).clamp(p1 - tc2, p1 + tc2));
                v[1] = clip(((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3).clamp(p2 - tc2, p2 + tc2));
            }
            if !s.no_q {
                v[4] =
                    clip(((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3).clamp(q0 - tc2, q0 + tc2));
                v[5] = clip(((p0 + q0 + q1 + q2 + 2) >> 2).clamp(q1 - tc2, q1 + tc2));
                v[6] = clip(((p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3).clamp(q2 - tc2, q2 + tc2));
            }
        }
        return;
    }
    let side = (beta + (beta >> 1)) >> 3;
    let (fp1, fq1) = (dp0 + dp3 < side, dq0 + dq3 < side);
    let half = tc >> 1;
    for v in l.iter_mut() {
        let [_, p2, p1, p0, q0, q1, q2, _] = *v;
        let delta = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
        if delta.abs() >= tc * 10 {
            continue;
        }
        let delta = delta.clamp(-tc, tc);
        if !s.no_p {
            v[3] = clip(p0 + delta);
            if fp1 {
                v[2] = clip(p1 + ((((p2 + p0 + 1) >> 1) - p1 + delta) >> 1).clamp(-half, half));
            }
        }
        if !s.no_q {
            v[4] = clip(q0 - delta);
            if fq1 {
                v[5] = clip(q1 + ((((q2 + q0 + 1) >> 1) - q1 - delta) >> 1).clamp(-half, half));
            }
        }
    }
}

#[test]
fn eight_lines_match_the_clauses() {
    if !crate::wide::has_avx2() {
        return;
    }
    let mut seed = 0x5eed_u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for round in 0..20_000 {
        // Smooth lines with a step, so that every branch occurs.
        let base = (next() % 256) as i32;
        let step = (next() % 64) as i32 - 32;
        let noise = 1 + (next() % 16) as i32;
        let lines: [[i32; 8]; 8] = core::array::from_fn(|_| {
            core::array::from_fn(|i| {
                let v = base + if i >= 4 { step } else { 0 } + (next() % noise as u64) as i32;
                v.clamp(0, 255)
            })
        });
        let mut seg = || {
            let k = next();
            Segment {
                beta: (k % 65) as i16 * if round % 7 == 0 { 0 } else { 1 },
                tc: ((k >> 8) % 25) as i16,
                no_p: (k >> 16) % 5 == 0,
                no_q: (k >> 24) % 7 == 0,
            }
        };
        let segs = [seg(), seg()];
        let mut want = lines;
        for (s, sg) in segs.iter().enumerate() {
            let mut l: [[i32; 8]; 4] = core::array::from_fn(|k| want[4 * s + k]);
            reference(&mut l, *sg);
            for k in 0..4 {
                want[4 * s + k] = l[k];
            }
        }
        // The kernel takes the samples across the edge as rows.
        let mut rows: [[u8; 8]; 8] =
            core::array::from_fn(|i| core::array::from_fn(|l| lines[l][i] as u8));
        assert!(luma8(&mut rows, segs));
        for l in 0..8 {
            for i in 0..8 {
                assert_eq!(
                    i32::from(rows[i][l]),
                    want[l][i],
                    "round {round} line {l} sample {i}"
                );
            }
        }
        // And by lines.
        let mut by_line: [[u8; 8]; 8] =
            core::array::from_fn(|l| core::array::from_fn(|i| lines[l][i] as u8));
        assert!(luma8_lines(&mut by_line, segs));
        for l in 0..8 {
            for i in 0..8 {
                assert_eq!(i32::from(by_line[l][i]), want[l][i], "lines: round {round}");
            }
        }
    }
}

#[test]
fn eight_deep_lines_match_the_clauses() {
    if !crate::wide::has_avx2() {
        return;
    }
    let mut seed = 0xdeeb_u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for bd in [9u32, 10, 11] {
        let (max, sh) = ((1i32 << bd) - 1, bd - 8);
        for round in 0..20_000 {
            // Smooth lines with a step, and lines at the ends of the range.
            let base = (next() % (max as u64 + 1)) as i32;
            let step = ((next() % 64) as i32 - 32) << sh;
            let noise = (1 + (next() % 16) as i32) << sh;
            let lines: [[i32; 8]; 8] = core::array::from_fn(|_| {
                core::array::from_fn(|i| {
                    let v = base + if i >= 4 { step } else { 0 } + (next() % noise as u64) as i32;
                    if round % 11 == 0 {
                        if i >= 4 { max } else { 0 }
                    } else {
                        v.clamp(0, max)
                    }
                })
            });
            let mut seg = || {
                let k = next();
                Segment {
                    beta: ((k % 65) as i16 * i16::from(round % 7 != 0)) << sh,
                    tc: (((k >> 8) % 25) as i16) << sh,
                    no_p: (k >> 16) % 5 == 0,
                    no_q: (k >> 24) % 7 == 0,
                }
            };
            let segs = [seg(), seg()];
            let mut want = lines;
            for (s, sg) in segs.iter().enumerate() {
                let mut l: [[i32; 8]; 4] = core::array::from_fn(|k| want[4 * s + k]);
                reference_at(&mut l, *sg, max);
                for k in 0..4 {
                    want[4 * s + k] = l[k];
                }
            }
            let mut rows: [[u16; 8]; 8] =
                core::array::from_fn(|i| core::array::from_fn(|l| lines[l][i] as u16));
            assert!(luma16(&mut rows, segs, bd));
            let mut by_line: [[u16; 8]; 8] =
                core::array::from_fn(|l| core::array::from_fn(|i| lines[l][i] as u16));
            assert!(luma16_lines(&mut by_line, segs, bd));
            for l in 0..8 {
                for i in 0..8 {
                    assert_eq!(
                        i32::from(rows[i][l]),
                        want[l][i],
                        "bd {bd} round {round} line {l} sample {i}"
                    );
                    assert_eq!(
                        i32::from(by_line[l][i]),
                        want[l][i],
                        "lines: bd {bd} round {round}"
                    );
                }
            }
        }
    }
}
