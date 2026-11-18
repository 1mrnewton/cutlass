//! Temporary: calls one MC kernel many times so instructions per call can be counted.
#![allow(clippy::print_stdout)]
use cutlass_h264_kernels::mc::{
    BiWeight, Block, avg_block, chroma_block, chroma_pair, luma_block, luma_pair, weight_block,
};

fn main() {
    let which = std::env::args().nth(1).unwrap_or_default();
    let calls: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let stride = 256usize;
    let mut plane = vec![0u8; stride * 64];
    let mut seed = 0x1234_5678_9abc_def0u64;
    for sample in plane.iter_mut() {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *sample = (seed >> 56) as u8;
    }
    let mut dst = vec![0u8; 256 * 16];
    let parts: Vec<&str> = which.split(',').collect();
    let w: usize = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(16);
    let xf: u8 = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let yf: u8 = parts.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let cr: Vec<u8> = plane.iter().map(|v| v ^ 0x5a).collect();
    let mut dst2 = vec![0u8; 256 * 16];
    let mode = [
        "lpair", "lold", "cpair", "avg", "weight", "luma", "planes", "cpair2",
    ]
    .iter()
    .position(|m| *m == parts[0])
    .unwrap_or(9);
    let mut sum = 0u64;
    for i in 0..calls {
        let block = Block {
            src: &plane,
            stride,
            at: (8 + (i & 7), 8),
            size: (w, w),
        };
        let a = &plane[(i & 7)..(i & 7) + 256];
        let b = &plane[512 + (i & 7)..512 + (i & 7) + 256];
        let other = Block {
            src: &plane,
            stride,
            at: (12 + (i & 7), 20),
            size: (w, w),
        };
        let ok = if mode == 6 || mode == 7 {
            let src = |at: (usize, usize)| cutlass_h264_kernels::mc::ChromaSource {
                cb: &plane,
                cr: &cr,
                stride,
                at,
                frac: (xf, yf),
            };
            let one = std::hint::black_box(src((8 + (i & 7), 8)));
            let two = (parts.get(4) == Some(&"2")).then(|| src((12 + (i & 7), 20)));
            if mode == 6 {
                cutlass_h264_kernels::mc::chroma_planes((&mut dst, &mut dst2), 16, (w, w), one, two)
            } else {
                let a = chroma_pair(
                    &mut dst,
                    16,
                    (one.plane(true, (w, w)), one.frac),
                    two.map(|t| (t.plane(true, (w, w)), t.frac)),
                );
                let b = chroma_pair(
                    &mut dst2,
                    16,
                    (one.plane(false, (w, w)), one.frac),
                    two.map(|t| (t.plane(false, (w, w)), t.frac)),
                );
                a && b
            }
        } else if mode == 0 {
            luma_pair(
                &mut dst,
                16,
                std::hint::black_box((block, (xf, yf))),
                Some((other, (xf, yf))),
            )
        } else if mode == 1 {
            let mut t0 = [0u8; 256];
            let mut t1 = [0u8; 256];
            luma_block(&mut t0, w, std::hint::black_box(block), (xf, yf));
            luma_block(&mut t1, w, std::hint::black_box(other), (xf, yf));
            avg_block(&mut dst, 16, &t0, &t1, (w, w))
        } else if mode == 2 {
            chroma_pair(
                &mut dst,
                16,
                std::hint::black_box((block, (xf, yf))),
                Some((other, (xf, yf))),
            )
        } else if mode == 3 {
            avg_block(&mut dst, 16, std::hint::black_box(a), b, (w, w))
        } else if mode == 4 {
            let weight = BiWeight {
                w0: 40,
                w1: 24,
                shift: 6,
                offset: 1,
            };
            weight_block(&mut dst, 16, std::hint::black_box(a), b, (w, w), weight)
        } else if mode == 5 {
            luma_block(&mut dst, 256, std::hint::black_box(block), (xf, yf))
        } else {
            chroma_block(&mut dst, 256, std::hint::black_box(block), (xf, yf))
        };
        sum += u64::from(ok) + u64::from(dst[5]);
    }
    println!("{sum}");
}
