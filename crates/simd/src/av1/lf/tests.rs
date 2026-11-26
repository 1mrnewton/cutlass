use super::{
    IDLE, Limits, Size, by_line, by_line_16, by_sample, cols_16_in_place, cols_in_place,
    rows_in_place, scalar,
};

/// A small fixed-seed generator: the same lines on every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
}

const SIZES: [Size; 4] = [Size::Four, Size::Six, Size::Eight, Size::Sixteen];

/// A line of 14 samples: smooth sides with a step at the edge, so every
/// branch of the filter is taken by some line.
fn line(rng: &mut Rng) -> [u8; 14] {
    let kind = rng.below(6);
    let base = rng.below(256) as i32;
    let step = match kind {
        0 => 0,
        1 => rng.below(4) as i32,
        2 => rng.below(24) as i32,
        _ => rng.below(256) as i32 - 128,
    };
    let noise = [1, 1, 2, 3, 12, 256][kind as usize];
    core::array::from_fn(|k| {
        let side = if k < 7 { 0 } else { step };
        let n = rng.below(noise) as i32 - (noise as i32) / 2;
        (base + side + n).clamp(0, 255) as u8
    })
}

fn limits(rng: &mut Rng) -> Limits {
    // What `lvl` 1..=63 and `sharpness` 0..=7 give (7.14.4).
    let lvl = 1 + rng.below(63) as i32;
    let sharp = rng.below(8) as i32;
    let shift = if sharp > 4 { 2 } else { i32::from(sharp > 0) };
    let limit = if sharp > 0 {
        (lvl >> shift).clamp(1, 9 - sharp)
    } else {
        (lvl >> shift).max(1)
    };
    Limits {
        limit: limit as u8,
        blimit: (2 * (lvl + 2) + limit) as u8,
        thresh: (lvl >> 4) as u8,
    }
}

/// What the specification gives for one line, and whether it passed.
fn expect(l: &[u8; 14], size: Size, lim: Limits) -> ([u8; 14], bool) {
    let mut px = l.map(i32::from);
    let on = scalar::line(&mut px, size, lim);
    (px.map(|v| v as u8), on)
}

#[test]
fn every_layout_matches_the_specification() {
    let mut rng = Rng(0x0D_EB10C);
    // Lines that passed, were flat (moved outside p1..q1), and were flat
    // over the whole of a size-16 line.
    let (mut passed, mut wide, mut widest) = (0, 0, 0);
    for round in 0..6000 {
        let size = SIZES[round % 4];
        let lim = limits(&mut rng);
        let lines: [[u8; 14]; 16] = core::array::from_fn(|_| line(&mut rng));
        let want: Vec<_> = lines.iter().map(|l| expect(l, size, lim)).collect();
        let any = want.iter().any(|w| w.1);
        for (l, w) in lines.iter().zip(&want) {
            passed += usize::from(w.1);
            wide += usize::from(l[4] != w.0[4] || l[9] != w.0[9]);
            widest += usize::from(l[1] != w.0[1] || l[12] != w.0[12]);
        }

        let mut taps = [[0u8; 16]; 14];
        for (l, line) in lines.iter().enumerate() {
            for (row, &v) in taps.iter_mut().zip(line) {
                row[l] = v;
            }
        }
        assert_eq!(by_sample(&mut taps, size, lim), any);
        for (l, w) in want.iter().enumerate() {
            for (k, row) in taps.iter().enumerate() {
                assert_eq!(row[l], w.0[k], "by sample {size:?} line {l} tap {k}");
            }
        }

        if size == Size::Sixteen {
            let mut wide_lines: [[u8; 16]; 16] = core::array::from_fn(|l| {
                core::array::from_fn(|k| match k {
                    0 => 7,
                    15 => 9,
                    _ => lines[l][k - 1],
                })
            });
            assert_eq!(by_line_16(&mut wide_lines, lim), any);
            for (l, w) in want.iter().enumerate() {
                assert_eq!(wide_lines[l][1..15], w.0, "by line 16, line {l}");
                assert_eq!((wide_lines[l][0], wide_lines[l][15]), (7, 9));
            }
        } else {
            let mut short: [[u8; 8]; 16] =
                core::array::from_fn(|l| core::array::from_fn(|k| lines[l][3 + k]));
            assert_eq!(by_line(&mut short, size, lim), any);
            for (l, w) in want.iter().enumerate() {
                assert_eq!(short[l], w.0[3..11], "by line {size:?} line {l}");
            }
        }
    }
    assert!(passed > 10_000, "{passed} lines passed the mask");
    assert!(wide > 2_000, "{wide} lines took a wide filter");
    assert!(widest > 300, "{widest} lines took the widest filter");
}

#[test]
fn a_filter_touches_only_the_samples_its_size_names() {
    let mut rng = Rng(77);
    for round in 0..400 {
        let size = SIZES[round % 4];
        let lim = Limits {
            limit: 40,
            blimit: 190,
            thresh: 2,
        };
        let before: [[u8; 16]; 14] =
            core::array::from_fn(|_| core::array::from_fn(|_| 100 + rng.below(3) as u8));
        let mut taps = before;
        by_sample(&mut taps, size, lim);
        let (lo, hi) = size.writes();
        for k in (0..lo).chain(hi..14) {
            assert_eq!(taps[k], before[k], "{size:?} tap {k}");
        }
    }
}

#[test]
fn idle_lines_never_change_and_never_count() {
    let lim = Limits {
        limit: 63,
        blimit: 193,
        thresh: 3,
    };
    for size in SIZES {
        let mut taps: [[u8; 16]; 14] =
            core::array::from_fn(|k| [if k < 7 { IDLE.0 } else { IDLE.1 }; 16]);
        let before = taps;
        assert!(!by_sample(&mut taps, size, lim), "{size:?}");
        assert_eq!(taps, before);
    }
    let mut lines = [[
        IDLE.0, IDLE.0, IDLE.0, IDLE.0, IDLE.1, IDLE.1, IDLE.1, IDLE.1,
    ]; 16];
    assert!(!by_line(&mut lines, Size::Eight, lim));
    let mut wide: [[u8; 16]; 16] =
        [core::array::from_fn(|k| if k < 8 { IDLE.0 } else { IDLE.1 }); 16];
    assert!(!by_line_16(&mut wide, lim));
}

#[test]
fn the_narrow_layout_refuses_the_widest_filter() {
    let mut lines = [[100u8; 8]; 16];
    let lim = Limits {
        limit: 63,
        blimit: 193,
        thresh: 0,
    };
    assert!(!by_line(&mut lines, Size::Sixteen, lim));
    assert_eq!(lines, [[100u8; 8]; 16]);
}

#[test]
fn a_step_inside_the_limits_is_smoothed() {
    // Flat sides four apart: the size-8 filter ramps across the edge. Each
    // output is the mean of the seven samples around it and itself again.
    let mut taps: [[u8; 16]; 14] = core::array::from_fn(|k| [if k < 7 { 100 } else { 104 }; 16]);
    let lim = Limits {
        limit: 10,
        blimit: 40,
        thresh: 1,
    };
    assert!(by_sample(&mut taps, Size::Eight, lim));
    let got: Vec<u8> = (3..11).map(|k| taps[k][0]).collect();
    assert_eq!(got, [100, 101, 101, 102, 103, 103, 104, 104]);
}

/// A picture whose rows are lines of [`line`] across a horizontal edge at
/// row 8 and, along each row, across vertical edges every 16 columns.
fn picture(rng: &mut Rng, stride: usize) -> Vec<u8> {
    let mut buf = vec![0u8; stride * 24];
    for x in 0..stride {
        let l = line(rng);
        for (k, &v) in l.iter().enumerate() {
            buf[(1 + k) * stride + x] = v;
        }
    }
    for y in 0..24 {
        for x0 in (0..stride).step_by(16) {
            if rng.below(2) == 0 {
                let l = line(rng);
                buf[y * stride + x0..][..14].copy_from_slice(&l);
            }
        }
    }
    buf
}

#[test]
fn in_place_kernels_match_their_scalar_twins() {
    let mut rng = Rng(0x01A9_1ACE);
    let stride = 48;
    let mut moved = 0;
    for round in 0..3000 {
        let size = SIZES[round % 4];
        let lim = limits(&mut rng);
        let pic = picture(&mut rng, stride);
        // A horizontal edge at row 8, sixteen columns from 3.
        let (mut a, mut b) = (pic.clone(), pic.clone());
        let at = 8 * stride + 3;
        assert_eq!(
            rows_in_place(&mut a, stride, at, size, lim),
            scalar::rows_in_place(&mut b, stride, at, size, lim)
        );
        assert_eq!(a, b, "rows, round {round}");
        // The vertical edge of the lines from column 16 (q0 at column 23)
        // on rows 2..18.
        let (mut a, mut b) = (pic.clone(), pic.clone());
        let at = 2 * stride + 23;
        if size == Size::Sixteen {
            assert_eq!(
                cols_16_in_place(&mut a, stride, at, lim),
                scalar::cols_16_in_place(&mut b, stride, at, lim)
            );
        } else {
            assert_eq!(
                cols_in_place(&mut a, stride, at, size, lim),
                scalar::cols_in_place(&mut b, stride, at, size, lim)
            );
        }
        assert_eq!(a, b, "columns, round {round}");
        moved += usize::from(a != pic);
    }
    assert!(moved > 300, "the edges filter some lines: {moved}");
}

/// The limits of level `lvl` and sharpness `sharp` (7.14.4).
fn limits_of(lvl: i32, sharp: i32) -> Limits {
    let shift = if sharp > 4 { 2 } else { i32::from(sharp > 0) };
    let limit = if sharp > 0 {
        (lvl >> shift).clamp(1, 9 - sharp)
    } else {
        (lvl >> shift).max(1)
    };
    Limits {
        limit: limit as u8,
        blimit: (2 * (lvl + 2) + limit) as u8,
        thresh: (lvl >> 4) as u8,
    }
}

/// Sixteen lines of every kind in one call: idle lines, lines flat all
/// along, lines flat near the edge only, lines at the ends of the sample
/// range (where the signed steps saturate), and the lines of [`line`].
fn mixed(rng: &mut Rng, turn: usize) -> [[u8; 14]; 16] {
    core::array::from_fn(|l| match (l + turn) % 8 {
        0 => core::array::from_fn(|k| if k < 7 { IDLE.0 } else { IDLE.1 }),
        1 => {
            let b = rng.below(254) as u8;
            core::array::from_fn(|k| b + u8::from(k >= 7) + u8::from(rng.below(8) == 0))
        }
        2 => {
            let b = 2 + rng.below(250) as u8;
            core::array::from_fn(|k| match k {
                0..3 => b.wrapping_add(rng.below(64) as u8),
                11.. => b.wrapping_sub(rng.below(64) as u8),
                _ => b + rng.below(2) as u8,
            })
        }
        3 => {
            let top = rng.below(2) == 0;
            core::array::from_fn(|_| {
                let n = rng.below(4) as u8;
                if top { 255 - n } else { n }
            })
        }
        _ => line(rng),
    })
}

#[test]
fn every_level_and_sharpness_matches_with_all_kinds_of_line_in_one_call() {
    // Each lane must take its own filter, whatever its neighbours take.
    let mut rng = Rng(0x00F1_A7ED);
    // Calls with a narrow line beside a wide one, and with a line flat
    // near the edge beside one flat all along.
    let (mut narrow_and_wide, mut both_wide) = (0, 0);
    for lvl in 0..=63 {
        for sharp in 0..8 {
            let lim = limits_of(lvl, sharp);
            for &size in &SIZES {
                let lines = mixed(&mut rng, lvl as usize);
                let want: Vec<_> = lines.iter().map(|l| expect(l, size, lim)).collect();
                let is_wide = |l: &[u8; 14], w: &[u8; 14]| l[4] != w[4] || l[9] != w[9];
                let is_widest = |l: &[u8; 14], w: &[u8; 14]| l[1] != w[1] || l[12] != w[12];
                let pairs = || lines.iter().zip(&want);
                let narrow = pairs().any(|(l, w)| w.1 && !is_wide(l, &w.0));
                let wide = pairs().any(|(l, w)| is_wide(l, &w.0));
                let wide_only = pairs().any(|(l, w)| is_wide(l, &w.0) && !is_widest(l, &w.0));
                let widest = pairs().any(|(l, w)| is_widest(l, &w.0));
                narrow_and_wide += usize::from(narrow && wide);
                both_wide += usize::from(wide_only && widest);
                let any = want.iter().any(|w| w.1);
                let mut taps = [[0u8; 16]; 14];
                for (l, line) in lines.iter().enumerate() {
                    for (row, &v) in taps.iter_mut().zip(line) {
                        row[l] = v;
                    }
                }
                assert_eq!(by_sample(&mut taps, size, lim), any);
                for (l, w) in want.iter().enumerate() {
                    for (k, row) in taps.iter().enumerate() {
                        assert_eq!(
                            row[l], w.0[k],
                            "by sample {size:?} level {lvl} sharpness {sharp} line {l} tap {k}"
                        );
                    }
                }
                if size == Size::Sixteen {
                    let mut all: [[u8; 16]; 16] = core::array::from_fn(|l| {
                        core::array::from_fn(|k| match k {
                            0 => 200,
                            15 => 3,
                            _ => lines[l][k - 1],
                        })
                    });
                    assert_eq!(by_line_16(&mut all, lim), any);
                    for (l, w) in want.iter().enumerate() {
                        assert_eq!(all[l][1..15], w.0, "by line 16, level {lvl} line {l}");
                        assert_eq!((all[l][0], all[l][15]), (200, 3));
                    }
                } else {
                    let mut short: [[u8; 8]; 16] =
                        core::array::from_fn(|l| core::array::from_fn(|k| lines[l][3 + k]));
                    assert_eq!(by_line(&mut short, size, lim), any);
                    for (l, w) in want.iter().enumerate() {
                        assert_eq!(
                            short[l],
                            w.0[3..11],
                            "by line {size:?} level {lvl} line {l}"
                        );
                    }
                }
            }
        }
    }
    assert!(narrow_and_wide > 200, "{narrow_and_wide}");
    assert!(both_wide > 20, "{both_wide}");
}

#[test]
fn in_place_kernels_take_a_buffer_that_ends_with_their_last_sample() {
    // The slices hold exactly the samples the kernels are documented to
    // read: nothing before the first, nothing after the last.
    let mut rng = Rng(0x7167_0001);
    for round in 0..400 {
        let size = SIZES[round % 4];
        let lim = limits(&mut rng);
        let stride = 16 + round % 5;
        // Rows: the first row read starts the buffer, the last ends it.
        let (lo, hi) = size.reads();
        let mut a: Vec<u8> = vec![0; (hi - lo - 1) * stride + 16];
        for x in 0..16 {
            let l = line(&mut rng);
            for k in lo..hi {
                a[(k - lo) * stride + x] = l[k];
            }
        }
        let mut b = a.clone();
        // `q0` is row 7 of the line, row `7 - lo` of the buffer.
        let at = (7 - lo) * stride;
        assert_eq!(
            rows_in_place(&mut a, stride, at, size, lim),
            scalar::rows_in_place(&mut b, stride, at, size, lim)
        );
        assert_eq!(a, b, "rows {size:?}, round {round}");
        // Columns: sixteen rows of 8 (or 16) samples.
        let n = if size == Size::Sixteen { 16 } else { 8 };
        let mut a: Vec<u8> = vec![0; 15 * stride + n];
        for y in 0..16 {
            let l = line(&mut rng);
            for k in 0..n {
                a[y * stride + k] = if n == 16 { l[(k + 13) % 14] } else { l[3 + k] };
            }
        }
        let mut b = a.clone();
        if size == Size::Sixteen {
            assert_eq!(
                cols_16_in_place(&mut a, stride, 8, lim),
                scalar::cols_16_in_place(&mut b, stride, 8, lim)
            );
        } else {
            assert_eq!(
                cols_in_place(&mut a, stride, 4, size, lim),
                scalar::cols_in_place(&mut b, stride, 4, size, lim)
            );
        }
        assert_eq!(a, b, "columns {size:?}, round {round}");
    }
}

#[test]
fn a_call_with_no_line_inside_the_limits_changes_nothing() {
    let mut rng = Rng(9);
    let lim = Limits {
        limit: 1,
        blimit: 5,
        thresh: 0,
    };
    for &size in &SIZES {
        // A step of at least 100 across every edge.
        let mut pic: Vec<u8> = (0..24 * 40).map(|_| rng.below(50) as u8).collect();
        for y in 0..24 {
            for x in 0..40 {
                if (y >= 8) != (x >= 20) {
                    pic[y * 40 + x] += 150;
                }
            }
        }
        let was = pic.clone();
        assert!(!rows_in_place(&mut pic, 40, 8 * 40 + 2, size, lim));
        if size == Size::Sixteen {
            assert!(!cols_16_in_place(&mut pic, 40, 40 + 20, lim));
        } else {
            assert!(!cols_in_place(&mut pic, 40, 40 + 20, size, lim));
        }
        assert_eq!(pic, was);
    }
}
