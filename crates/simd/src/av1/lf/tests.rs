use super::{IDLE, Limits, Size, by_line, by_line_16, by_sample, scalar};

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
