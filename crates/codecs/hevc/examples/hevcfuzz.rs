//! Mutation fuzzer: corrupts the access units of real HEVC streams and
//! checks that decoding never panics and never hangs, on one thread and on
//! several.
//!
//! Usage: hevcfuzz <iterations> <in.mp4|in.mov|in.hevc>...
//!
//! Each iteration flips bits, overwrites bytes, cuts access units short or
//! scrambles their first bytes (the NAL and slice headers), then decodes
//! the result with 1, 2 or 4 threads in turn and drains the decoder. An
//! iteration that has not finished after 60 seconds counts as a hang: the
//! fuzzer reports it and stops, since its threads cannot be taken back.
//! The exit status is non-zero when any iteration panicked or hung.

#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use cutlass_decode::{Packet, VideoDecoder};
use cutlass_hevc::HevcDecoder;

mod common;

fn main() -> std::process::ExitCode {
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
    let mut runs = 0usize;
    for path in args.iter().skip(2) {
        let (params, units) = match common::read(path) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("{e}");
                continue;
            }
        };
        let n = units.len();
        if n == 0 {
            continue;
        }
        for it in 0..iters {
            let mut au = units.clone();
            let mode = rnd() % 4;
            for _ in 0..(1 + rnd() % 8) {
                let f = (rnd() as usize) % n;
                let len = au[f].len();
                if len == 0 {
                    continue;
                }
                match mode {
                    0 => {
                        let i = (rnd() as usize) % len;
                        au[f][i] ^= 1 << (rnd() % 8);
                    }
                    1 => {
                        let i = (rnd() as usize) % len;
                        au[f][i] = rnd() as u8;
                    }
                    2 => {
                        let cut = (rnd() as usize) % len;
                        au[f].truncate(cut);
                    }
                    _ => {
                        let i = (rnd() as usize) % len.min(24);
                        au[f][i] = rnd() as u8;
                    }
                }
            }
            let threads = [1, 2, 4][it % 3];
            let params = params.clone();
            let (done, finished) = mpsc::channel();
            std::thread::spawn(move || {
                let r = std::panic::catch_unwind(move || {
                    let Ok(mut dec) = HevcDecoder::open_with_threads(&params, threads) else {
                        return;
                    };
                    for (i, unit) in au.into_iter().enumerate() {
                        if unit.is_empty() {
                            continue;
                        }
                        let _ = dec.push(&Packet::video(unit, i as i64, 1, 30, false));
                    }
                    let _ = dec.drain();
                });
                let _ = done.send(r.is_ok());
            });
            runs += 1;
            match finished.recv_timeout(Duration::from_secs(60)) {
                Ok(true) => {}
                Ok(false) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                    eprintln!("  in {path} iteration {it} ({threads} threads)");
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    eprintln!("HANG in {path} iteration {it} ({threads} threads)");
                    println!(
                        "runs: {runs} panics: {} hangs: 1",
                        PANICS.load(Ordering::Relaxed)
                    );
                    return std::process::ExitCode::FAILURE;
                }
            }
        }
    }
    let panics = PANICS.load(Ordering::Relaxed);
    println!("runs: {runs} panics: {panics} hangs: 0");
    if panics == 0 {
        std::process::ExitCode::SUCCESS
    } else {
        std::process::ExitCode::FAILURE
    }
}
