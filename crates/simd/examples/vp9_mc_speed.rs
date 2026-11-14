//! Throughput of the VP9 8-bit interpolation kernels on blocks that stay
//! in cache: `cargo run --release -p cutlass-simd --example vp9_mc_speed`.

use cutlass_simd::vp9::mc::{Block, predict};

fn main() {
    let stride = 128;
    let src: Vec<u8> = (0..stride * 128).map(|i| (i * 7 % 251) as u8).collect();
    let mut dst = vec![0u8; 64 * 64];
    let mut tmp = vec![0u8; cutlass_simd::vp9::mc::TMP];
    let f = Some([-1i16, 6, -19, 78, 78, -19, 6, -1]);
    for &(w, h) in &[(8usize, 8usize), (16, 16), (32, 32), (64, 64)] {
        for (name, fx, fy) in [("copy", None, None), ("h", f, None), ("v", None, f), ("hv", f, f)] {
            let n = 4_000_000 / (w * h) + 1;
            let t = std::time::Instant::now();
            for i in 0..n {
                let at = (8 + i % 16) * stride + 8;
                let _ = predict(Block { dst: &mut dst, dst_stride: 64, src: &src, at, src_stride: stride, w, h, fx, fy, avg: false, tmp: &mut tmp });
            }
            let ns = t.elapsed().as_nanos() as f64 / (n * w * h) as f64;
            println!("{w}x{h} {name}: {ns:.3} ns/px");
        }
    }
}
