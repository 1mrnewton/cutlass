use super::{Block, prep_hv_u8, put_hv_u8, scalar};

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

/// Filter rows: regular, sharp, smooth and bilinear rows of the AV1
/// subpel filters, and one whose taps do not add up to 128.
const FILTERS: [[i8; 8]; 8] = [
    [0, 2, -6, 126, 8, -2, 0, 0],
    [0, 2, -14, 110, 38, -10, 2, 0],
    [0, 4, -20, 78, 78, -20, 4, 0],
    [-2, 6, -12, 124, 16, -6, 4, -2],
    [-4, 12, -24, 80, 80, -24, 12, -4],
    [0, 2, 28, 62, 34, 2, 0, 0],
    [0, 0, 0, 64, 64, 0, 0, 0],
    [0, 0, 0, 8, 120, 0, 0, 0],
];

const STRIDE: usize = 160;

/// A source with flat, extreme and noisy areas.
fn source(rng: &mut Rng) -> Vec<u8> {
    (0..STRIDE * 150)
        .map(|i| match (i / STRIDE / 24 + i % STRIDE / 40) % 4 {
            0 => 255,
            1 => 0,
            2 => (rng.next() % 256) as u8,
            _ => 100 + (rng.next() % 8) as u8,
        })
        .collect()
}

#[test]
fn both_outputs_match_the_specification_for_every_block_size() {
    let mut rng = Rng(0x00C0_DEC5);
    let src = source(&mut rng);
    for (w, h) in [
        (8, 8),
        (8, 4),
        (8, 32),
        (16, 16),
        (16, 64),
        (32, 8),
        (32, 32),
        (64, 64),
        (128, 128),
        (64, 3),
    ] {
        for round in 0..12 {
            let fh = &FILTERS[(rng.next() % 8) as usize];
            let fv = &FILTERS[(round + rng.next() as usize) % 8];
            let start = (rng.next() as usize % 8) * STRIDE + rng.next() as usize % 16;
            let stride = w + round % 3;
            let mut got = vec![7i16; stride * h];
            let mut want = got.clone();
            assert!(prep_hv_u8(
                &mut got,
                stride,
                &Block {
                    src: &src,
                    start,
                    src_stride: STRIDE,
                    w,
                    h,
                    fh,
                    fv
                },
            ));
            scalar::prep_hv(
                &mut want,
                stride,
                &Block {
                    src: &src,
                    start,
                    src_stride: STRIDE,
                    w,
                    h,
                    fh,
                    fv,
                },
            );
            assert_eq!(got, want, "prep {w}x{h} {fh:?} {fv:?}");
            let mut got = vec![9u8; stride * h];
            let mut want = got.clone();
            assert!(put_hv_u8(
                &mut got,
                stride,
                &Block {
                    src: &src,
                    start,
                    src_stride: STRIDE,
                    w,
                    h,
                    fh,
                    fv
                },
            ));
            scalar::put_hv(
                &mut want,
                stride,
                &Block {
                    src: &src,
                    start,
                    src_stride: STRIDE,
                    w,
                    h,
                    fh,
                    fv,
                },
            );
            assert_eq!(got, want, "put {w}x{h} {fh:?} {fv:?}");
        }
    }
}

#[test]
fn a_flat_block_stays_flat() {
    // The taps add up to 128: a flat source comes back as it is, scaled
    // by 16 in the compound intermediate.
    let src = vec![200u8; STRIDE * 40];
    let mut tmp = vec![0i16; 256];
    assert!(prep_hv_u8(
        &mut tmp,
        16,
        &Block {
            src: &src,
            start: 5,
            src_stride: STRIDE,
            w: 16,
            h: 16,
            fh: &FILTERS[3],
            fv: &FILTERS[1]
        },
    ));
    assert!(tmp.iter().all(|&v| v == 200 * 16));
    let mut out = vec![0u8; 256];
    assert!(put_hv_u8(
        &mut out,
        16,
        &Block {
            src: &src,
            start: 5,
            src_stride: STRIDE,
            w: 16,
            h: 16,
            fh: &FILTERS[3],
            fv: &FILTERS[1]
        },
    ));
    assert!(out.iter().all(|&v| v == 200));
}

#[test]
fn the_final_samples_are_clipped_to_8_bits() {
    // A sharp filter over a step from 0 to 255 overshoots both ways.
    let src: Vec<u8> = (0..STRIDE * 40)
        .map(|i| if i % STRIDE >= 12 { 255 } else { 0 })
        .collect();
    let mut out = vec![9u8; 16 * 8];
    assert!(put_hv_u8(
        &mut out,
        16,
        &Block {
            src: &src,
            start: 0,
            src_stride: STRIDE,
            w: 16,
            h: 8,
            fh: &FILTERS[4],
            fv: &FILTERS[4]
        },
    ));
    let mut want = vec![0u8; 16 * 8];
    scalar::put_hv(
        &mut want,
        16,
        &Block {
            src: &src,
            start: 0,
            src_stride: STRIDE,
            w: 16,
            h: 8,
            fh: &FILTERS[4],
            fv: &FILTERS[4],
        },
    );
    assert_eq!(out, want);
    assert!(out.contains(&0) && out.contains(&255));
}

#[test]
fn a_block_it_does_not_take_is_left_alone() {
    let src = vec![50u8; STRIDE * 40];
    let mut tmp = vec![3i16; 64 * 64];
    let (fh, fv) = (&FILTERS[0], &FILTERS[1]);
    // Widths the kernel has no strip for, and no rows.
    for (w, h) in [(4, 8), (2, 2), (12, 8), (8, 0), (256, 8)] {
        assert!(
            !prep_hv_u8(
                &mut tmp,
                64,
                &Block {
                    src: &src,
                    start: 0,
                    src_stride: STRIDE,
                    w,
                    h,
                    fh,
                    fv
                },
            ),
            "{w}x{h}"
        );
    }
    // A source or a destination too short for the block.
    assert!(!prep_hv_u8(
        &mut tmp,
        64,
        &Block {
            src: &src[..STRIDE * 10],
            start: 0,
            src_stride: STRIDE,
            w: 8,
            h: 8,
            fh,
            fv
        },
    ));
    assert!(!prep_hv_u8(
        &mut tmp[..60],
        64,
        &Block {
            src: &src,
            start: 0,
            src_stride: STRIDE,
            w: 8,
            h: 8,
            fh,
            fv
        },
    ));
    assert!(!prep_hv_u8(
        &mut tmp,
        64,
        &Block {
            src: &src,
            start: usize::MAX - 3,
            src_stride: STRIDE,
            w: 8,
            h: 8,
            fh,
            fv
        },
    ));
    assert!(tmp.iter().all(|&v| v == 3));
    let mut out = vec![3u8; 64];
    assert!(!put_hv_u8(
        &mut out,
        8,
        &Block {
            src: &src,
            start: 0,
            src_stride: STRIDE,
            w: 4,
            h: 8,
            fh,
            fv
        },
    ));
    assert!(out.iter().all(|&v| v == 3));
}

#[test]
fn a_source_must_hold_one_sample_past_the_last_one_read() {
    // The kernel loads sixteen samples for the last eight columns, one
    // more than the filters read: a source that ends with the last
    // sample read is declined, and one sample more is enough.
    let (w, h) = (16, 8);
    let len = (h + 6) * STRIDE + w + 7;
    let mut rng = Rng(3);
    let src: Vec<u8> = (0..len + 1).map(|_| (rng.next() % 256) as u8).collect();
    let mut got = vec![0i16; w * h];
    let mut want = got.clone();
    let block = |src| Block {
        src,
        start: 0,
        src_stride: STRIDE,
        w,
        h,
        fh: &FILTERS[3],
        fv: &FILTERS[4],
    };
    assert!(!prep_hv_u8(&mut got, w, &block(&src[..len])));
    assert!(got.iter().all(|&v| v == 0), "declined: nothing written");
    assert!(prep_hv_u8(&mut got, w, &block(&src)));
    scalar::prep_hv(&mut want, w, &block(&src));
    assert_eq!(got, want);
}
