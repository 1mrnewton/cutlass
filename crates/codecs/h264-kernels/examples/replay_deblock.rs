//! Temporary: replays macroblocks recorded from a real decode through the
//! deblocking kernels, to time them on real samples and thresholds.
#![allow(clippy::print_stdout)]

use cutlass_h264_kernels::deblock::{Filter, MbEdges, chroma_mb, luma_mb};

const REC: usize = 768;

fn edges(bytes: &[u8]) -> ([Option<Filter>; 4], [Option<Filter>; 4]) {
    let one = |x: &[u8]| match x[0] {
        0 => None,
        kind => Some(Filter {
            alpha: x[1],
            beta: x[2],
            tc0: (kind == 1).then(|| [x[3] as i8, x[4] as i8, x[5] as i8, x[6] as i8]),
        }),
    };
    let v = [0, 1, 2, 3].map(|e| one(&bytes[7 * e..]));
    let h = [0, 1, 2, 3].map(|e| one(&bytes[28 + 7 * e..]));
    (v, h)
}

fn main() -> Result<(), String> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: replay_deblock <records> [loops] [mode]")?;
    let loops: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let mode = std::env::args().nth(3).unwrap_or_default();
    let copy_only = mode == "1";
    let (do_luma, do_chroma) = (mode != "3", mode != "2");
    let data = std::fs::read(&path).map_err(|err| format!("{path}: {err}"))?;
    let n = data.len() / REC;
    let records: Vec<_> = (0..n)
        .map(|i| {
            let r = &data[i * REC..];
            let (lv, lh) = edges(r);
            let (bv, bh) = edges(&r[456..]);
            let (cv, ch) = edges(&r[612..]);
            (i, (lv, lh), (bv, bh), (cv, ch))
        })
        .collect();
    let (ls, cs) = (64usize, 32usize);
    let mut lp = vec![0u8; ls * 24];
    let mut bp = vec![0u8; cs * 12];
    let mut rp = vec![0u8; cs * 12];
    let mut sum = 0xcbf2_9ce4_8422_2325u64;
    for loop_index in 0..loops {
        for &(i, (lv, lh), (bv, bh), (cv, ch)) in &records {
            let r = &data[i * REC..];
            let (lw, bw, rw) = (&r[56..456], &r[512..612], &r[668..768]);
            for y in 0..20 {
                lp[y * ls..y * ls + 20].copy_from_slice(&lw[y * 20..y * 20 + 20]);
            }
            for y in 0..10 {
                bp[y * cs..y * cs + 10].copy_from_slice(&bw[y * 10..y * 10 + 10]);
                rp[y * cs..y * cs + 10].copy_from_slice(&rw[y * 10..y * 10 + 10]);
            }
            if !copy_only {
                let l = MbEdges {
                    at: 4 * ls + 4,
                    down: ls,
                    vertical: lv,
                    horizontal: lh,
                };
                let b = MbEdges {
                    at: 2 * cs + 2,
                    down: cs,
                    vertical: bv,
                    horizontal: bh,
                };
                let c = MbEdges {
                    at: 2 * cs + 2,
                    down: cs,
                    vertical: cv,
                    horizontal: ch,
                };
                if do_luma {
                    assert!(luma_mb(&mut lp, &l));
                }
                if do_chroma {
                    assert!(chroma_mb(&mut bp, &b));
                    assert!(chroma_mb(&mut rp, &c));
                }
            }
            if loop_index == 0 {
                for y in 0..20 {
                    for x in 0..20 {
                        sum = (sum ^ u64::from(lp[y * ls + x])).wrapping_mul(0x100_0000_01b3);
                    }
                }
                for y in 0..10 {
                    for x in 0..10 {
                        sum = (sum ^ u64::from(bp[y * cs + x])).wrapping_mul(0x100_0000_01b3);
                        sum = (sum ^ u64::from(rp[y * cs + x])).wrapping_mul(0x100_0000_01b3);
                    }
                }
            }
        }
    }
    println!("{n} mbs sum {sum:016x}");
    Ok(())
}
