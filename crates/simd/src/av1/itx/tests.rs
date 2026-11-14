use super::steps::{Prog, table};
use super::{Block, Kind, add, add_flat, scalar, supports};

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

    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }

    /// A value within `mag` of zero.
    fn signed(&mut self, mag: u32) -> i32 {
        self.below(2 * mag + 1) as i32 - mag as i32
    }
}

/// Every transform there is: its kind, length, step table, where the
/// inputs go in `T`, where the outputs come from and whether odd outputs
/// are negated.
type Transform = (Kind, usize, Prog<256>, [u8; 64], [u8; 64], bool);

fn transforms() -> Vec<Transform> {
    vec![
        (
            Kind::Dct,
            4,
            table::DCT4,
            table::DCT4_IN,
            table::SAME,
            false,
        ),
        (
            Kind::Dct,
            8,
            table::DCT8,
            table::DCT8_IN,
            table::SAME,
            false,
        ),
        (
            Kind::Dct,
            16,
            table::DCT16,
            table::DCT16_IN,
            table::SAME,
            false,
        ),
        (
            Kind::Dct,
            32,
            table::DCT32,
            table::DCT32_IN,
            table::SAME,
            false,
        ),
        (
            Kind::Dct,
            64,
            table::DCT64,
            table::DCT64_IN,
            table::SAME,
            false,
        ),
        (
            Kind::Adst,
            8,
            table::ADST8,
            table::ADST8_IN,
            table::ADST8_OUT,
            true,
        ),
        (
            Kind::Adst,
            16,
            table::ADST16,
            table::ADST16_IN,
            table::ADST16_OUT,
            true,
        ),
    ]
}

/// The transform run as its whole table of steps on one column.
fn by_the_table(tr: &Transform, input: &[i16]) -> Vec<i16> {
    let (_, n, prog, at, from, negate_odd) = tr;
    let round = |x: i32| ((x + 2048) >> 12) as i16;
    let mut t = [0i16; 64];
    for (k, &v) in input.iter().enumerate() {
        t[usize::from(at[k])] = v;
    }
    assert!(prog.len <= 256, "the table holds every step");
    for s in &prog.ops[..prog.len] {
        let (a, b) = (usize::from(s.a), usize::from(s.b));
        if s.had {
            let (a, b) = if s.flip { (b, a) } else { (a, b) };
            let (x, y) = (t[a], t[b]);
            t[a] = x.saturating_add(y);
            t[b] = x.saturating_sub(y);
        } else {
            let (x, y) = (i32::from(t[a]), i32::from(t[b]));
            let (c, n) = (i32::from(s.cos), i32::from(s.sin));
            let (p, q) = (round(x * c - y * n), round(x * n + y * c));
            (t[a], t[b]) = if s.flip { (q, p) } else { (p, q) };
        }
    }
    (0..*n)
        .map(|i| {
            let v = t[usize::from(from[i])];
            if *negate_odd && i & 1 == 1 {
                v.saturating_neg()
            } else {
                v
            }
        })
        .collect()
}

#[test]
fn a_transform_with_zero_inputs_left_out_matches_its_whole_table() {
    let mut rng = Rng(11);
    for tr in transforms() {
        let (kind, n) = (tr.0, tr.1);
        for nz in [1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 64] {
            if nz > n.min(32) {
                continue;
            }
            for round in 0..20 {
                // Small values, then ones large enough to saturate a sum.
                let mag = if round < 14 { 2000 } else { 32767 };
                let mut t = [[0i16; 8]; 64];
                for v in t.iter_mut().take(nz) {
                    *v = std::array::from_fn(|_| rng.signed(mag) as i16);
                }
                let input = t;
                assert!(scalar::one_dimension(kind, n, nz, &mut t));
                for l in 0..8 {
                    let column: Vec<i16> = input.iter().take(nz).map(|v| v[l]).collect();
                    let got: Vec<i16> = t.iter().take(n).map(|v| v[l]).collect();
                    assert_eq!(got, by_the_table(&tr, &column), "{kind:?} {n} nz {nz}");
                }
            }
        }
    }
}

#[test]
fn a_dct_of_its_first_input_alone_is_flat() {
    for n in [4, 8, 16, 32, 64] {
        let mut t = [[0i16; 8]; 64];
        t[0] = [4096, -4096, 1, -1, 1000, 77, 0, 32767];
        let want = t[0].map(|v| ((i32::from(v) * 2896 + 2048) >> 12) as i16);
        assert!(scalar::one_dimension(Kind::Dct, n, 1, &mut t));
        for (i, v) in t.iter().enumerate().take(n) {
            assert_eq!(*v, want, "length {n} output {i}");
        }
    }
}

#[test]
fn the_adst_of_four_values_follows_the_specification() {
    // 7.13.2.6 with one input each: the sines times it, rounded.
    let round = |x: i32| ((x + 2048) >> 12) as i16;
    let cases: [(usize, [i32; 4]); 4] = [
        (0, [1321, 2482, 3344, 3803]),
        (1, [3344, 3344, 0, -3344]),
        (2, [3803, -1321, -3344, 2482]),
        (3, [2482, -3803, 3344, -1321]),
    ];
    for (input, sines) in cases {
        let mut t = [[0i16; 8]; 64];
        // Not -32768: the specification keeps `t0 - t2 + t3` in 16 bits.
        t[input] = [1, -1, 100, -100, 4096, 9999, -32767, 32767];
        let x = t[input];
        assert!(scalar::one_dimension(Kind::Adst, 4, 4, &mut t));
        for (i, s) in sines.into_iter().enumerate() {
            assert_eq!(
                t[i],
                x.map(|v| round(i32::from(v) * s)),
                "input {input} output {i}"
            );
        }
    }
}

/// A block of the given size with random coefficients in its first rows
/// and columns, over a random prediction.
struct Case {
    w: usize,
    h: usize,
    stride: usize,
    coefs: Vec<i32>,
    dst: Vec<u8>,
    nz: (usize, usize),
    shift: u32,
    flips: (bool, bool),
}

fn case(rng: &mut Rng, w: usize, h: usize) -> Case {
    let (tw, th) = (w.min(32), h.min(32));
    let nz = match rng.below(4) {
        0 => (1 + rng.below(2) as usize, 1 + rng.below(2) as usize),
        1 => (1 + rng.below(th as u32) as usize, 1),
        2 => (1, 1 + rng.below(tw as u32) as usize),
        _ => (
            1 + rng.below(th as u32) as usize,
            1 + rng.below(tw as u32) as usize,
        ),
    };
    let mag = [40, 600, 5000, 32767][rng.below(4) as usize];
    let mut coefs = vec![0i32; tw * th];
    for r in 0..nz.0 {
        for c in 0..nz.1 {
            if rng.below(3) != 0 {
                coefs[r * tw + c] = rng.signed(mag);
            }
        }
    }
    let stride = w + rng.below(9) as usize;
    Case {
        w,
        h,
        stride,
        coefs,
        dst: (0..stride * h).map(|_| rng.below(256) as u8).collect(),
        nz,
        shift: rng.below(3),
        flips: (rng.below(2) == 0, rng.below(2) == 0),
    }
}

fn block<'a>(c: &'a Case, dst: &'a mut [u8], row: Kind, col: Kind) -> Block<'a> {
    Block {
        dst,
        stride: c.stride,
        coefs: &c.coefs,
        w: c.w,
        h: c.h,
        row,
        col,
        nz_rows: c.nz.0,
        nz_cols: c.nz.1,
        shift: c.shift,
        flip_ud: c.flips.0,
        flip_lr: c.flips.1,
    }
}

#[test]
fn the_kernel_matches_the_scalar_block_transform() {
    let mut rng = Rng(5);
    let sizes = [4usize, 8, 16, 32, 64];
    let kinds = [Kind::Dct, Kind::Adst, Kind::Identity];
    let mut blocks = 0;
    for &w in &sizes {
        for &h in &sizes {
            for row in kinds {
                for col in kinds {
                    if !supports(w, h, row, col) {
                        continue;
                    }
                    for _ in 0..30 {
                        let c = case(&mut rng, w, h);
                        let mut want = c.dst.clone();
                        scalar::add(&mut block(&c, &mut want, row, col));
                        let mut got = c.dst.clone();
                        assert!(add(&mut block(&c, &mut got, row, col)));
                        assert_eq!(got, want, "{w}x{h} {row:?} {col:?} nz {:?}", c.nz);
                        blocks += 1;
                    }
                }
            }
        }
    }
    // 4 sizes each way: 16 blocks of DCTs, 4 of ADSTs, 9 of identity
    // transforms, and the mixes.
    assert!(blocks >= (16 + 4 + 9) * 30);
}

#[test]
fn a_block_the_kernel_does_not_take_or_that_does_not_fit_is_left_alone() {
    let mut rng = Rng(9);
    let c = case(&mut rng, 16, 16);
    // A transform there is none of at this length.
    assert!(!supports(64, 16, Kind::Adst, Kind::Dct));
    let mut dst = c.dst.clone();
    let mut b = block(&c, &mut dst, Kind::Dct, Kind::Dct);
    b.w = 12;
    assert!(!add(&mut b));
    // Too few samples, too few coefficients, a stride under the width
    // and a shift past 2.
    let last = c.dst.len() - (c.stride - c.w) - 1;
    let mut b = block(&c, &mut dst[..last], Kind::Dct, Kind::Dct);
    assert!(!add(&mut b));
    b.dst = &mut [];
    assert!(!add(&mut b));
    let mut b = block(&c, &mut dst, Kind::Dct, Kind::Dct);
    b.coefs = &c.coefs[..255];
    assert!(!add(&mut b));
    b.coefs = &c.coefs;
    b.stride = 15;
    assert!(!add(&mut b));
    b.stride = c.stride;
    b.shift = 3;
    assert!(!add(&mut b));
    assert_eq!(dst, c.dst);
    // Exactly the samples of the block are enough.
    let mut dst = c.dst.clone();
    let enough = 15 * c.stride + 16;
    assert!(add(&mut block(
        &c,
        &mut dst[..enough],
        Kind::Dct,
        Kind::Dct
    )));
}

#[test]
fn a_flat_residual_is_added_to_every_sample_and_clamped() {
    let mut rng = Rng(3);
    for w in [4usize, 8, 16, 32, 64] {
        for h in [4usize, 8, 16, 64] {
            for v in [
                0,
                1,
                -1,
                7,
                -40,
                200,
                -200,
                255,
                -255,
                256,
                -4000,
                4000,
                i32::MAX,
                i32::MIN,
            ] {
                let stride = w + rng.below(5) as usize;
                let old: Vec<u8> = (0..stride * h).map(|_| rng.below(256) as u8).collect();
                let mut got = old.clone();
                assert!(add_flat(&mut got, stride, w, h, v));
                let mut want = old.clone();
                scalar::add_flat(&mut want, stride, w, h, v);
                assert_eq!(got, want, "{w}x{h} {v}");
                // The samples between the rows of the block stay.
                for y in 0..h {
                    for x in 0..stride {
                        let at = y * stride + x;
                        let add = if x < w { i64::from(v) } else { 0 };
                        assert_eq!(i64::from(got[at]), (i64::from(old[at]) + add).clamp(0, 255));
                    }
                }
            }
        }
    }
    // Another width, or too few samples: nothing is done.
    let mut dst = [9u8; 64];
    assert!(!add_flat(&mut dst, 8, 12, 4, 5));
    assert!(!add_flat(&mut dst, 4, 8, 4, 5));
    assert!(!add_flat(&mut dst[..31], 8, 8, 4, 5));
    assert!(!add_flat(&mut dst, 8, 8, 0, 5));
    assert_eq!(dst, [9u8; 64]);
}
