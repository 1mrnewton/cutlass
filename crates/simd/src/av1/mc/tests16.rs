use super::{
    Block16, prep_h_u16, prep_hv_u16, prep_v_u16, put_h_u16, put_hv_u16, put_v_u16, scalar16,
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

/// A source of `bd` bits with flat, extreme and noisy areas.
fn source(rng: &mut Rng, bd: u32) -> Vec<u16> {
    let max = (1u32 << bd) - 1;
    (0..STRIDE * 150)
        .map(|i| match (i / STRIDE / 24 + i % STRIDE / 40) % 4 {
            0 => max as u16,
            1 => 0,
            2 => (rng.next() % (max + 1)) as u16,
            _ => (max / 3 + rng.next() % 8) as u16,
        })
        .collect()
}

type Prep = fn(&mut [i16], usize, &Block16<'_>) -> bool;
type Put = fn(&mut [u16], usize, &Block16<'_>) -> bool;
type PrepTwin = fn(&mut [i16], usize, &Block16<'_>);
type PutTwin = fn(&mut [u16], usize, &Block16<'_>);

#[test]
fn every_entry_matches_the_specification_at_10_and_12_bits() {
    let preps: [(Prep, PrepTwin, &str); 3] = [
        (prep_hv_u16, scalar16::prep_hv, "prep hv"),
        (prep_h_u16, scalar16::prep_h, "prep h"),
        (prep_v_u16, scalar16::prep_v, "prep v"),
    ];
    let puts: [(Put, PutTwin, &str); 3] = [
        (put_hv_u16, scalar16::put_hv, "put hv"),
        (put_h_u16, scalar16::put_h, "put h"),
        (put_v_u16, scalar16::put_v, "put v"),
    ];
    for bd in [10, 12] {
        let mut rng = Rng(0x00C0_DEC5 + u64::from(bd));
        let src = source(&mut rng, bd);
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
            (16, 1),
        ] {
            for round in 0..12 {
                let fh = &FILTERS[(rng.next() % 8) as usize];
                let fv = &FILTERS[(round + rng.next() as usize) % 8];
                let start = (rng.next() as usize % 8) * STRIDE + rng.next() as usize % 16;
                let stride = w + round % 3;
                let b = Block16 {
                    src: &src,
                    start,
                    src_stride: STRIDE,
                    w,
                    h,
                    fh,
                    fv,
                    bd,
                };
                for (entry, twin, name) in preps {
                    let mut got = vec![7i16; stride * h];
                    let mut want = got.clone();
                    assert!(entry(&mut got, stride, &b));
                    twin(&mut want, stride, &b);
                    assert_eq!(got, want, "{name} {bd}-bit {w}x{h} {fh:?} {fv:?}");
                }
                for (entry, twin, name) in puts {
                    let mut got = vec![9u16; stride * h];
                    let mut want = got.clone();
                    assert!(entry(&mut got, stride, &b));
                    twin(&mut want, stride, &b);
                    assert_eq!(got, want, "{name} {bd}-bit {w}x{h} {fh:?} {fv:?}");
                }
            }
        }
    }
}

#[test]
fn a_flat_block_stays_flat() {
    // The taps add up to 128: a flat source comes back as it is. The
    // compound intermediate is the sample at 14 bits, less 8192.
    for (bd, sample) in [(10, 800u16), (12, 3200)] {
        let src = vec![sample; STRIDE * 40];
        let b = Block16 {
            src: &src,
            start: 5,
            src_stride: STRIDE,
            w: 16,
            h: 16,
            fh: &FILTERS[3],
            fv: &FILTERS[1],
            bd,
        };
        let scaled = (i32::from(sample) << (14 - bd)) - 8192;
        for entry in [prep_hv_u16, prep_h_u16, prep_v_u16] {
            let mut tmp = vec![0i16; 256];
            assert!(entry(&mut tmp, 16, &b));
            assert!(tmp.iter().all(|&v| i32::from(v) == scaled), "{bd}-bit");
        }
        for entry in [put_hv_u16, put_h_u16, put_v_u16] {
            let mut out = vec![0u16; 256];
            assert!(entry(&mut out, 16, &b));
            assert!(out.iter().all(|&v| v == sample), "{bd}-bit");
        }
    }
}

#[test]
fn the_final_samples_are_clipped_to_the_bit_depth() {
    // A sharp filter over a step from 0 to the largest sample overshoots
    // both ways.
    for bd in [10u32, 12] {
        let max = ((1u32 << bd) - 1) as u16;
        let src: Vec<u16> = (0..STRIDE * 40)
            .map(|i| if i % STRIDE >= 12 { max } else { 0 })
            .collect();
        let b = Block16 {
            src: &src,
            start: 0,
            src_stride: STRIDE,
            w: 16,
            h: 8,
            fh: &FILTERS[4],
            fv: &FILTERS[4],
            bd,
        };
        let mut out = vec![9u16; 16 * 8];
        assert!(put_hv_u16(&mut out, 16, &b));
        let mut want = vec![0u16; 16 * 8];
        scalar16::put_hv(&mut want, 16, &b);
        assert_eq!(out, want);
        assert!(out.contains(&0) && out.contains(&max));
        assert!(out.iter().all(|&v| v <= max));
    }
}

#[test]
fn a_block_the_kernel_does_not_take_is_left_alone() {
    let src = vec![500u16; STRIDE * 40];
    let block = |w: usize, h: usize, bd: u32, len: usize| Block16 {
        src: &src[..len],
        start: 0,
        src_stride: STRIDE,
        w,
        h,
        fh: &FILTERS[0],
        fv: &FILTERS[0],
        bd,
    };
    let all = src.len();
    let mut out = vec![7u16; 64 * 64];
    // A width of 4, no rows, 8-bit samples, a destination too small, and a
    // source that ends before the last sample read: an 8x8 block reads 14
    // rows and one sample past the 15 of its last row.
    assert!(!put_hv_u16(&mut out, 64, &block(4, 8, 10, all)));
    assert!(!put_hv_u16(&mut out, 64, &block(8, 0, 10, all)));
    assert!(!put_hv_u16(&mut out, 64, &block(8, 8, 8, all)));
    assert!(!put_hv_u16(&mut out[..63], 8, &block(8, 8, 10, all)));
    assert!(!put_hv_u16(
        &mut out,
        64,
        &block(8, 8, 10, 14 * STRIDE + 15)
    ));
    assert!(out.iter().all(|&v| v == 7));
    assert!(put_hv_u16(&mut out, 64, &block(8, 8, 10, 14 * STRIDE + 16)));
    let mut tmp = vec![7i16; 64];
    assert!(!prep_v_u16(&mut tmp, 8, &block(8, 9, 12, all)));
    assert!(tmp.iter().all(|&v| v == 7));
}
