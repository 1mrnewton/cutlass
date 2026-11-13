//! The kernels against clause 8.7.3 sample by sample.

use super::{band_row, edge_row};

fn noise(n: usize, mut seed: u64) -> Vec<u8> {
    (0..n)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed as u8
        })
        .collect()
}

#[test]
fn edge_and_band_offsets_match_the_specification() {
    if !crate::wide::has_avx2() {
        return;
    }
    let raw = noise(4 * 200, 99);
    // Smooth runs too, so that equal neighbours occur often.
    let smooth: Vec<u8> = raw.iter().map(|&v| 120 + (v & 3)).collect();
    let extremes: Vec<u8> = raw
        .iter()
        .map(|&v| if v & 1 == 0 { 2 } else { 253 })
        .collect();
    for samples in [&raw, &smooth, &extremes] {
        for n in [1usize, 7, 31, 32, 33, 64, 100, 150] {
            let cur = &samples[..n];
            let a = &samples[200..200 + n];
            let b = &samples[400..400 + n];
            for offsets in [[7, 3, -2, -7], [-7, -7, 7, 7], [1, 0, 0, -1]] {
                let mut got = vec![0u8; n];
                assert!(edge_row(&mut got, cur, (a, b), offsets));
                for x in 0..n {
                    let v = i32::from(cur[x]);
                    let e = 2 + (v - i32::from(a[x])).signum() + (v - i32::from(b[x])).signum();
                    let o = match e {
                        0 => offsets[0],
                        1 => offsets[1],
                        3 => offsets[2],
                        4 => offsets[3],
                        _ => 0,
                    };
                    assert_eq!(
                        i32::from(got[x]),
                        (v + i32::from(o)).clamp(0, 255),
                        "edge n {n} x {x}"
                    );
                }
                for position in [0u8, 5, 15, 29, 31] {
                    assert!(band_row(&mut got, cur, position, offsets));
                    for x in 0..n {
                        let v = i32::from(cur[x]);
                        let k = (((v >> 3) - i32::from(position)) & 31) as usize;
                        let o = if k < 4 { i32::from(offsets[k]) } else { 0 };
                        assert_eq!(i32::from(got[x]), (v + o).clamp(0, 255), "band n {n} x {x}");
                    }
                }
            }
        }
    }
}

#[test]
fn deep_edge_and_band_offsets_match_the_specification() {
    use super::{band_row16, edge_row16};
    if !crate::wide::has_avx2() {
        return;
    }
    for bd in [9u32, 10, 12] {
        let top = (1i32 << bd) - 1;
        let raw = noise(2 * 600, u64::from(bd) + 5);
        let words = |f: &dyn Fn(u16) -> u16| -> Vec<u16> {
            raw.chunks(2)
                .map(|p| f(u16::from_le_bytes([p[0], p[1]]) & top as u16))
                .collect()
        };
        let noisy = words(&|v| v);
        let smooth = words(&|v| (top as u16 / 2) + (v & 3));
        let extremes = words(&|v| if v & 1 == 0 { 3 } else { top as u16 - 3 });
        let big = 7i16 << (bd.min(10) - 5);
        for samples in [&noisy, &smooth, &extremes] {
            for n in [1usize, 7, 15, 16, 17, 64, 100, 150] {
                let cur = &samples[..n];
                let a = &samples[200..200 + n];
                let b = &samples[400..400 + n];
                for offsets in [[big, 3, -2, -big], [-big, -big, big, big], [1, 0, 0, -1]] {
                    let mut got = vec![0u16; n];
                    assert!(edge_row16(&mut got, cur, (a, b), offsets, bd));
                    for x in 0..n {
                        let v = i32::from(cur[x]);
                        let e = 2 + (v - i32::from(a[x])).signum() + (v - i32::from(b[x])).signum();
                        let o = match e {
                            0 => offsets[0],
                            1 => offsets[1],
                            3 => offsets[2],
                            4 => offsets[3],
                            _ => 0,
                        };
                        assert_eq!(
                            i32::from(got[x]),
                            (v + i32::from(o)).clamp(0, top),
                            "edge bd {bd} n {n} x {x}"
                        );
                    }
                    for position in [0u8, 5, 15, 29, 31] {
                        assert!(band_row16(&mut got, cur, position, offsets, bd));
                        for x in 0..n {
                            let v = i32::from(cur[x]);
                            let k = (((v >> (bd - 5)) - i32::from(position)) & 31) as usize;
                            let o = if k < 4 { i32::from(offsets[k]) } else { 0 };
                            assert_eq!(
                                i32::from(got[x]),
                                (v + o).clamp(0, top),
                                "band bd {bd} n {n} x {x}"
                            );
                        }
                    }
                }
            }
        }
    }
}
