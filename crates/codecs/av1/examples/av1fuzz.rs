//! Mutation fuzzer: corrupts real IVF streams and checks that decoding never panics.
//!
//! Usage: av1fuzz <iterations> <file.ivf>...
//!
//! Odd iterations decode with `Av1Core::decode_delayed` and `drain`
//! (frames reconstructed beside the parsing of the next ones); panics on
//! any thread count.

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::sync::atomic::{AtomicUsize, Ordering};

use cutlass_av1::dec::decoder::Av1Core;

fn ivf_frames(data: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    if data.len() < 32 {
        return out;
    }
    let mut pos = u16::from_le_bytes([data[6], data[7]]) as usize;
    while pos + 12 <= data.len() {
        let size =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 12;
        let end = (pos + size).min(data.len());
        out.push(data[pos..end].to_vec());
        pos = end;
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let iters: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    static PANICS: AtomicUsize = AtomicUsize::new(0);
    std::panic::set_hook(Box::new(|info| {
        PANICS.fetch_add(1, Ordering::Relaxed);
        eprintln!("PANIC: {info}");
    }));
    for path in &args[2..] {
        let Ok(data) = std::fs::read(path) else {
            continue;
        };
        let frames = ivf_frames(&data);
        for it in 0..iters {
            let mut fr = frames.clone();
            let n = fr.len();
            if n == 0 {
                break;
            }
            let mode = rnd() % 4;
            for _ in 0..(1 + rnd() % 8) {
                let f = (rnd() as usize) % n;
                let len = fr[f].len();
                if len == 0 {
                    continue;
                }
                match mode {
                    0 => {
                        let i = (rnd() as usize) % len;
                        fr[f][i] ^= 1 << (rnd() % 8);
                    }
                    1 => {
                        let i = (rnd() as usize) % len;
                        fr[f][i] = rnd() as u8;
                    }
                    2 => {
                        let cut = (rnd() as usize) % len;
                        fr[f].truncate(cut);
                    }
                    _ => {
                        // Header region corruption.
                        let i = (rnd() as usize) % len.min(24);
                        fr[f][i] = rnd() as u8;
                    }
                }
            }
            let r = std::panic::catch_unwind(|| {
                let mut core = Av1Core::new();
                core.max_pixels = 1 << 20;
                let delayed = it % 2 == 1;
                for (i, f) in fr.iter().enumerate() {
                    if f.is_empty() {
                        continue;
                    }
                    if delayed {
                        let _ = core.decode_delayed(f, i as u64);
                    } else {
                        let _ = core.decode(f);
                    }
                }
                while delayed && core.drain().is_err() {}
            });
            if r.is_err() {
                eprintln!("  in {path} iteration {it}");
            }
        }
    }
    println!("panics: {}", PANICS.load(Ordering::Relaxed));
}
