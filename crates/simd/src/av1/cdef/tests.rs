use super::{
    CENTRE, Filter, STRIDE, UNAVAILABLE, WINDOW, direction_u8, direction_u16, filter_inner_u8,
    filter_u8, filter_u16, scalar,
};

/// A small fixed-seed generator: the same blocks on every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
}

/// Blocks that reach the extremes of the sums, then textured random ones.
fn blocks(bd: u32, n: usize) -> Vec<[[u16; 8]; 8]> {
    let max = (1u16 << bd) - 1;
    let mut out = vec![[[0; 8]; 8], [[max; 8]; 8], [[max / 2 + 1; 8]; 8]];
    // One edge in each direction, both polarities.
    for k in 0..16 {
        let mut b = [[0u16; 8]; 8];
        for (i, row) in b.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                let side = match k % 8 {
                    0 => i + j < 8,
                    1 => 2 * i + j < 12,
                    2 => i < 4,
                    3 => 2 * i + 7 - j < 12,
                    4 => i + 7 - j < 8,
                    5 => i + 2 * (7 - j) < 12,
                    6 => j < 4,
                    _ => i + 2 * j < 12,
                };
                *v = if side == (k < 8) { max } else { 0 };
            }
        }
        out.push(b);
    }
    let mut rng = Rng(0x00C0_FFEE ^ u64::from(bd));
    for t in 0..n {
        let mut b = [[0u16; 8]; 8];
        // Narrow ranges make near ties between directions.
        let span = [u32::from(max) + 1, 2, 4, 37][t % 4];
        let base = rng.next() % (u32::from(max) + 2 - span);
        for row in &mut b {
            for v in row {
                *v = (base + rng.next() % span) as u16;
            }
        }
        out.push(b);
    }
    out
}

#[test]
fn direction_u8_matches_the_specification() {
    for b in blocks(8, 20_000) {
        let b8 = b.map(|r| r.map(|v| v as u8));
        assert_eq!(direction_u8(&b8), scalar::direction(&b, 8), "{b:?}");
    }
}

#[test]
fn direction_u16_matches_the_specification_at_every_depth() {
    for bd in [8, 10, 12] {
        for b in blocks(bd, 20_000) {
            assert_eq!(
                direction_u16(&b, bd),
                scalar::direction(&b, bd),
                "{bd} {b:?}"
            );
        }
    }
}

#[test]
fn a_flat_block_has_no_direction_and_no_variance() {
    assert_eq!(direction_u8(&[[77; 8]; 8]), (0, 0));
    // Every cost is equal, so the first direction stays.
    assert_eq!(direction_u16(&[[1023; 8]; 8], 10), (0, 0));
}

#[test]
fn an_edge_gives_its_direction() {
    // Every row is flat, so the sums along the rows stand out: direction 2.
    let mut rows = [[0u8; 8]; 8];
    for (i, r) in rows.iter_mut().enumerate() {
        *r = [if i < 4 { 255 } else { 0 }; 8];
    }
    let (dir, var) = direction_u8(&rows);
    assert_eq!(dir, 2);
    assert!(var > 0);
    // Every column is flat: direction 6.
    let cols = [[255, 255, 255, 255, 0, 0, 0, 0]; 8];
    assert_eq!(direction_u8(&cols).0, 6);
}

#[test]
fn a_depth_outside_the_range_is_clamped() {
    let b = blocks(12, 4);
    assert_eq!(direction_u16(&b[5], 99), direction_u16(&b[5], 12));
    assert_eq!(
        direction_u16(&[[200; 8]; 8], 0),
        direction_u16(&[[200; 8]; 8], 8)
    );
}

/// `Cdef_Directions` of the specification: `(row, column)` steps.
const DIRECTIONS: [[[i32; 2]; 2]; 8] = [
    [[-1, 1], [-2, 2]],
    [[0, 1], [-1, 2]],
    [[0, 1], [0, 2]],
    [[0, 1], [1, 2]],
    [[1, 1], [2, 2]],
    [[1, 0], [2, 1]],
    [[1, 0], [2, 0]],
    [[1, 0], [2, -1]],
];

/// The filter for strengths `pri`, `sec` (already scaled), a direction, a
/// damping and a bit depth, as 7.15.3 derives it.
fn filter_for(pri: u16, sec: u16, dir: usize, damping: u32, bd: u32) -> Filter {
    let at = |d: usize, k: usize| {
        let [dy, dx] = DIRECTIONS[d & 7][k];
        (CENTRE as i32 + dy * STRIDE as i32 + dx) as usize
    };
    let shift = |v: u16| {
        if v == 0 {
            0
        } else {
            damping.saturating_sub(15 - v.leading_zeros())
        }
    };
    Filter {
        pri,
        sec,
        pri_shift: shift(pri),
        sec_shift: shift(sec),
        pri_taps: if (pri >> (bd - 8)) & 1 == 1 {
            [3, 3]
        } else {
            [4, 2]
        },
        pri_at: [at(dir, 0), at(dir, 1)],
        sec_at: [
            at(dir + 2, 0),
            at(dir + 6, 0),
            at(dir + 2, 1),
            at(dir + 6, 1),
        ],
    }
}

/// A window of textured samples of `bd` bits, with `edges` choosing which
/// sides lie outside the picture.
fn window(rng: &mut Rng, bd: u32, edges: u32) -> [u16; WINDOW] {
    let max = (1u32 << bd) - 1;
    let span = [max + 1, 3, 9, 60][(rng.next() % 4) as usize].min(max + 1);
    let base = rng.next() % (max + 2 - span);
    core::array::from_fn(|n| {
        let (row, col) = (n / STRIDE, n % STRIDE);
        let out = (edges & 1 != 0 && row < 2)
            || (edges & 2 != 0 && row >= 10)
            || (edges & 4 != 0 && col < 2)
            || (edges & 8 != 0 && (10..12).contains(&col));
        if out {
            UNAVAILABLE
        } else {
            (base + rng.next() % span) as u16
        }
    })
}

#[test]
fn the_filter_matches_the_specification() {
    let mut rng = Rng(0x00F1_17E4);
    let mut changed = 0;
    for round in 0..6000u32 {
        let bd = [8, 8, 10, 12][(round % 4) as usize];
        let edges = if round % 3 == 0 { rng.next() % 16 } else { 0 };
        let win = window(&mut rng, bd, edges);
        let (pri, sec) = match round % 5 {
            0 => (rng.next() % 16, 0),
            1 => (0, [1, 2, 4][(rng.next() % 3) as usize]),
            _ => (rng.next() % 16, [0, 1, 2, 4][(rng.next() % 4) as usize]),
        };
        let (pri, sec) = ((pri << (bd - 8)) as u16, (sec << (bd - 8)) as u16);
        if pri == 0 && sec == 0 {
            continue;
        }
        let damping = 3 + rng.next() % 4 + (bd - 8);
        let f = filter_for(pri, sec, (rng.next() % 8) as usize, damping, bd);
        let rows = [8, 8, 4][(round % 3) as usize];
        let want = scalar::filter(&win, &f, rows);
        assert_eq!(filter_u16(&win, &f, rows), want, "{bd} bits {f:?}");
        if bd == 8 {
            assert_eq!(filter_u8(&win, &f, rows), want.map(|r| r.map(|v| v as u8)));
        }
        changed += usize::from(want[0][0] != win[CENTRE]);
    }
    assert!(
        changed > 1000,
        "{changed} blocks changed their first sample"
    );
}

#[test]
fn rows_past_the_count_come_back_zero() {
    let mut rng = Rng(5);
    let win = window(&mut rng, 8, 0);
    let f = filter_for(7, 2, 3, 5, 8);
    let got = filter_u8(&win, &f, 4);
    assert_eq!(got[4..], [[0u8; 8]; 4]);
    assert_eq!(got[..4], filter_u8(&win, &f, 8)[..4]);
    // More rows than a block has are the block's eight.
    assert_eq!(filter_u8(&win, &f, 99), filter_u8(&win, &f, 8));
}

#[test]
fn a_block_alone_in_its_picture_is_unchanged() {
    // Every tap is outside the picture: nothing to pull the samples toward.
    let win: [u16; WINDOW] = core::array::from_fn(|n| {
        let (row, col) = (n / STRIDE, n % STRIDE);
        if (2..10).contains(&row) && (2..10).contains(&col) {
            200
        } else {
            UNAVAILABLE
        }
    });
    let got = filter_u8(&win, &filter_for(15, 4, 0, 6, 8), 8);
    assert_eq!(got, [[200u8; 8]; 8]);
}

#[test]
fn tap_positions_out_of_range_are_clamped() {
    let mut rng = Rng(9);
    let win = window(&mut rng, 8, 0);
    let mut f = filter_for(6, 2, 1, 5, 8);
    f.pri_at = [usize::MAX, 5000];
    f.sec_shift = 400;
    let mut clamped = f;
    clamped.pri_at = [2 * CENTRE; 2];
    clamped.sec_shift = 15;
    assert_eq!(filter_u8(&win, &f, 8), filter_u8(&win, &clamped, 8));
}

/// The byte filter's block as rows of 8, from a destination with rows
/// `stride` apart that started as `fill`.
fn inner(win: &[u16; WINDOW], f: &Filter, wide: bool, rows: usize) -> (Vec<u8>, bool) {
    let mut dst = vec![0xEE; 8 * 11];
    let done = filter_inner_u8(&mut dst, 11, win, f, wide, rows);
    (dst, done)
}

#[test]
fn the_byte_filter_matches_the_specification_inside_the_picture() {
    let mut rng = Rng(0x000B_17E5);
    let mut changed = 0;
    for round in 0..8000u32 {
        // Every sample available: the byte filter's whole contract.
        let win = window(&mut rng, 8, 0);
        let (pri, sec) = match round % 5 {
            0 => (1 + rng.next() % 15, 0),
            1 => (0, [1, 2, 4][(rng.next() % 3) as usize]),
            _ => (rng.next() % 16, [0, 1, 2, 4][(rng.next() % 4) as usize]),
        };
        if pri == 0 && sec == 0 {
            continue;
        }
        let damping = 3 + rng.next() % 4;
        let f = filter_for(
            pri as u16,
            sec as u16,
            (rng.next() % 8) as usize,
            damping,
            8,
        );
        let want = scalar::filter(&win, &f, 8).map(|r| r.map(|v| v as u8));
        for (wide, w) in [(true, 8), (false, 4)] {
            for rows in [8, 4] {
                let (dst, done) = inner(&win, &f, wide, rows);
                assert!(done);
                for (r, row) in dst.chunks(11).enumerate() {
                    if r < rows {
                        assert_eq!(row[..w], want[r][..w], "{w} wide row {r} of {rows}, {f:?}");
                    }
                    // Past the block's columns and rows nothing is written.
                    let from = if r < rows { w } else { 0 };
                    assert!(row[from..].iter().all(|&v| v == 0xEE), "{w} wide row {r}");
                }
            }
        }
        changed += usize::from(u16::from(want[0][0]) != win[CENTRE]);
    }
    assert!(
        changed > 1500,
        "{changed} blocks changed their first sample"
    );
}

#[test]
fn the_byte_filter_clamps_what_the_caller_derived() {
    let mut rng = Rng(11);
    let win = window(&mut rng, 8, 0);
    let mut f = filter_for(9, 2, 5, 5, 8);
    f.pri_at = [usize::MAX, 5000];
    f.sec_shift = 400;
    let mut clamped = f;
    clamped.pri_at = [2 * CENTRE; 2];
    clamped.sec_shift = 15;
    for wide in [true, false] {
        assert_eq!(inner(&win, &f, wide, 99), inner(&win, &clamped, wide, 8));
    }
    // A flat block stays flat whatever the strengths.
    let flat = [77u16; WINDOW];
    let (dst, _) = inner(&flat, &filter_for(15, 4, 3, 6, 8), true, 8);
    assert!(dst.chunks(11).all(|r| r[..8] == [77u8; 8]));
}

#[test]
fn the_byte_filter_declines_a_destination_too_short() {
    let mut rng = Rng(12);
    let win = window(&mut rng, 8, 0);
    let f = filter_for(9, 0, 5, 5, 8);
    let mut dst = vec![5u8; 3 * 11 + 7];
    assert!(!filter_inner_u8(&mut dst, 11, &win, &f, true, 4));
    assert!(!filter_inner_u8(&mut dst, 11, &win, &f, false, 0));
    assert!(dst.iter().all(|&v| v == 5));
    assert!(filter_inner_u8(&mut dst, 11, &win, &f, false, 4));
}
