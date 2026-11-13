//! The kernel against the two matrix products of clause 8.6.4.2.

use super::{DCT32, DST4, inverse, inverse4};

fn reference(c: &[i32], n: usize, bit_depth: u32) -> Vec<i32> {
    let m = |k: usize, i: usize| i32::from(DCT32[k * (32 / n)][i]);
    let mut mid = vec![0i32; n * n];
    for x in 0..n {
        for y in 0..n {
            let s: i64 = (0..n)
                .map(|k| i64::from(m(k, y)) * i64::from(c[k * n + x]))
                .sum();
            mid[y * n + x] = ((s + 64) >> 7).clamp(-32768, 32767) as i32;
        }
    }
    let shift = 20 - bit_depth;
    let mut out = vec![0i32; n * n];
    for y in 0..n {
        for x in 0..n {
            let s: i64 = (0..n)
                .map(|k| i64::from(m(k, x)) * i64::from(mid[y * n + k]))
                .sum();
            out[y * n + x] = ((s + (1 << (shift - 1))) >> shift) as i32;
        }
    }
    out
}

#[test]
fn matches_the_matrix_products() {
    if !crate::wide::has_avx2() {
        return;
    }
    let mut seed = 0xdc7_u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for n in [8usize, 16, 32] {
        for round in 0..300 {
            let cols = 1 + (next() as usize) % n;
            let rows = 1 + (next() as usize) % n;
            let big = round % 5 == 0;
            let mut c = vec![0i32; n * n];
            for y in 0..rows {
                for x in 0..cols {
                    if next() % 3 != 0 {
                        let r = next() as i32;
                        c[y * n + x] = if big { r >> 16 } else { r % 600 };
                    }
                }
            }
            for bd in [8u32, 10] {
                let want = reference(&c, n, bd);
                let mut got = vec![0i32; n * n];
                assert!(inverse(&c, &mut got, n, (cols, rows), bd));
                assert_eq!(got, want, "{n}x{n} cols {cols} rows {rows} bit depth {bd}");
            }
        }
    }
}

fn reference4(c: &[i32], dst: bool, bit_depth: u32) -> Vec<i32> {
    let m = |k: usize, i: usize| i32::from(if dst { DST4[k][i] } else { DCT32[k * 8][i] });
    let mut mid = [0i32; 16];
    for x in 0..4 {
        for y in 0..4 {
            let s: i64 = (0..4)
                .map(|k| i64::from(m(k, y)) * i64::from(c[k * 4 + x]))
                .sum();
            mid[y * 4 + x] = ((s + 64) >> 7).clamp(-32768, 32767) as i32;
        }
    }
    let shift = 20 - bit_depth;
    (0..16)
        .map(|i| {
            let (y, x) = (i / 4, i % 4);
            let s: i64 = (0..4)
                .map(|k| i64::from(m(k, x)) * i64::from(mid[y * 4 + k]))
                .sum();
            ((s + (1 << (shift - 1))) >> shift) as i32
        })
        .collect()
}

#[test]
fn four_by_four_matches_the_matrix_products() {
    if !crate::wide::has_avx2() {
        return;
    }
    let mut seed = 0x4444_u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for round in 0..2000 {
        let big = round % 4 == 0;
        let c: Vec<i32> = (0..16)
            .map(|_| {
                let r = next() as i32;
                if big { r >> 16 } else { r % 700 }
            })
            .collect();
        for dst in [false, true] {
            for bd in [8u32, 10] {
                let mut got = vec![0i32; 16];
                assert!(inverse4(&c, &mut got, dst, bd));
                assert_eq!(got, reference4(&c, dst, bd), "dst {dst} bit depth {bd}");
            }
        }
    }
}
