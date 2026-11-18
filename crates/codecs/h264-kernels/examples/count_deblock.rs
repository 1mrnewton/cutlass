//! Temporary: calls one deblocking kernel many times, so the instructions
//! each call costs can be counted from outside (`/usr/bin/time -l`).
#![allow(clippy::print_stdout)]

use cutlass_h264_kernels::deblock::{
    Edge, Filter, MbEdges, chroma_edge, chroma_mb, luma_edge, luma_mb,
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
    for (index, sample) in plane.iter_mut().enumerate() {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Smooth, like a picture, with a little noise.
        *sample = (100
            + (index % stride) / 8
            + (seed
                >> std::env::var("SHIFT")
                    .ok()
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(61)) as usize) as u8;
    }
    let (vertical, tc0, chroma) = match which.as_str() {
        "luma_v_normal" => (true, Some([1i8, 1, 2, 0]), false),
        "luma_h_normal" => (false, Some([1i8, 1, 2, 0]), false),
        "luma_v_strong" => (true, None, false),
        "luma_h_strong" => (false, None, false),
        "chroma_v_normal" => (true, Some([1i8, 1, 2, 0]), true),
        "chroma_h_normal" => (false, Some([1i8, 1, 2, 0]), true),
        "chroma_v_strong" => (true, None, true),
        "chroma_h_strong" => (false, None, true),
        _ => (true, None, false),
    };
    // Whole macroblocks: every edge at a strength below 4, an intra one
    // with an 8x8 transform (strength 4 outside, 3 inside, no edge at 4 or
    // 12), and chroma's two and two.
    let normal = Some(Filter {
        alpha: 40,
        beta: 10,
        tc0: Some([1, 1, 2, 0]),
    });
    let strong = Some(Filter {
        alpha: 40,
        beta: 10,
        tc0: None,
    });
    let whole = match which.as_str() {
        "luma_mb_normal" => Some((true, [normal; 4], [normal; 4])),
        "luma_mb_intra8" => Some((
            true,
            [strong, None, normal, None],
            [strong, None, normal, None],
        )),
        "luma_mb_intra4" => Some((
            true,
            [strong, normal, normal, normal],
            [strong, normal, normal, normal],
        )),
        "mb_none" => Some((true, [None; 4], [None; 4])),
        "luma_mb_t8" => Some((
            true,
            [normal, None, normal, None],
            [normal, None, normal, None],
        )),
        "mb_v0h0" => Some((true, [normal, None, None, None], [normal, None, None, None])),
        "mb_v_inner" => Some((true, [None, normal, normal, normal], [None; 4])),
        "mb_h_inner" => Some((true, [None; 4], [None, normal, normal, normal])),
        "mb_v0_normal" => Some((true, [normal, None, None, None], [None; 4])),
        "mb_v0_strong" => Some((true, [strong, None, None, None], [None; 4])),
        "mb_h0_normal" => Some((true, [None; 4], [normal, None, None, None])),
        "mb_h0_strong" => Some((true, [None; 4], [strong, None, None, None])),
        "chroma_mb_normal" => Some((
            false,
            [normal, None, normal, None],
            [normal, None, normal, None],
        )),
        "chroma_v0" => Some((false, [normal, None, None, None], [None; 4])),
        "chroma_h0" => Some((false, [None; 4], [normal, None, None, None])),
        "chroma_v0h0" => Some((
            false,
            [normal, None, None, None],
            [normal, None, None, None],
        )),
        "chroma_v0_strong" => Some((false, [strong, None, None, None], [None; 4])),
        "chroma_mb_intra" => Some((
            false,
            [strong, None, normal, None],
            [strong, None, normal, None],
        )),
        _ => None,
    };
    if let Some((luma, vertical, horizontal)) = whole {
        let mut done = 0u64;
        for call in 0..calls {
            let x = 16 + (call % 12) * 16;
            let y = 16 + (call / 12 % 2) * 16;
            let edges = MbEdges {
                at: y * stride + x,
                down: stride,
                vertical,
                horizontal,
            };
            let ok = if luma {
                luma_mb(&mut plane, &edges)
            } else {
                chroma_mb(&mut plane, &edges)
            };
            done += u64::from(ok);
        }
        println!(
            "{which}: {done} macroblocks, sample {}",
            plane[stride * 16 + 16]
        );
        return;
    }
    let mut done = 0u64;
    for call in 0..calls {
        let x = 16 + (call % 12) * 16;
        let y = 16 + (call / 12 % 2) * 16;
        let edge = Edge {
            at: y * stride + x,
            down: stride,
            vertical,
            alpha: 40,
            beta: 10,
            tc0,
        };
        let ok = if chroma {
            chroma_edge(&mut plane, edge)
        } else {
            luma_edge(&mut plane, edge)
        };
        done += u64::from(ok);
    }
    println!("{which}: {done} edges, sample {}", plane[stride * 16 + 16]);
}
