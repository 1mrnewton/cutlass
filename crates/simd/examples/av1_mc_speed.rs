//! Throughput of the AV1 8-bit motion compensation entries on blocks that
//! stay in cache: `cargo run --release -p cutlass-simd --example
//! av1_mc_speed [scalar]`. With `scalar` the run-time kernels are forced
//! off, which times the scalar twins.

#![allow(clippy::print_stdout)]

use cutlass_simd::av1::mc::{
    Block, prep_h_u8, prep_hv_u8, prep_v_u8, put_h_u8, put_hv_u8, put_v_u8,
};

fn main() {
    if std::env::args().nth(1).as_deref() == Some("scalar") {
        cutlass_simd::av1::cpu::force_scalar(true);
    }
    let stride = 160;
    let src: Vec<u8> = (0..stride * 150).map(|i| (i * 7 % 251) as u8).collect();
    let six: [i8; 8] = [0, 2, -14, 110, 38, -10, 2, 0];
    let eight: [i8; 8] = [-4, 12, -24, 80, 80, -24, 12, -4];
    let mut prep = vec![0i16; 128 * 128];
    let mut put = vec![0u8; 128 * 128];
    let mut sink = 0u64;
    for (taps, f) in [("6", &six), ("8", &eight)] {
        for &(w, h) in &[(8usize, 8usize), (16, 16), (32, 32), (64, 64), (128, 128)] {
            let n = 4_000_000 / (w * h) + 1;
            macro_rules! time {
                ($name:literal, $entry:ident, $dst:ident) => {{
                    let t = std::time::Instant::now();
                    for i in 0..n {
                        let b = Block {
                            src: &src,
                            start: (i % 8) * stride + i % 16,
                            src_stride: stride,
                            w,
                            h,
                            fh: f,
                            fv: f,
                        };
                        let _ = $entry(&mut $dst, w, &b);
                        sink = sink.wrapping_add($dst[i % (w * h)] as u64);
                    }
                    let ns = t.elapsed().as_nanos() as f64 / (n * w * h) as f64;
                    println!("{w}x{h} {taps}-tap {}: {ns:.3} ns/px", $name);
                }};
            }
            time!("prep hv", prep_hv_u8, prep);
            time!("put hv", put_hv_u8, put);
            time!("prep h", prep_h_u8, prep);
            time!("put h", put_h_u8, put);
            time!("prep v", prep_v_u8, prep);
            time!("put v", put_v_u8, put);
        }
    }
    println!("checksum {sink}");
}
