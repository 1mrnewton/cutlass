//! End-to-end preview-path benchmark: decode → composite → readback.
//!
//! Measures what the preview worker pays per frame — a sequential decode in
//! CPU vs zero-copy GPU output mode, each frame composited full-canvas and
//! read back — so the decode→composite seam (CPU plane upload vs shared
//! texture import) is included, unlike `decode_bench` which stops at the
//! decoder.
//!
//! Run:
//!   `cargo run --release -p cutlass-render --example composite_bench -- <media> [frames]`

use std::path::PathBuf;
use std::time::Instant;

use cutlass_compositor::{
    CompositeLayer, Compositor, CompositorConfig, GpuContext, LayerPlacement,
};
use cutlass_decoder::{OutputMode, open_video_decoder};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .map(PathBuf::from)
        .expect("usage: composite_bench <media> [frames]");
    let frames: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(150);

    let gpu = GpuContext::new_headless_blocking().expect("gpu");
    let info = gpu.adapter.get_info();
    println!("adapter: {} ({:?})", info.name, info.backend);

    let mut comp = Compositor::new(&gpu);

    for (mode, label) in [
        (OutputMode::Cpu, "cpu decode + upload"),
        (OutputMode::Gpu, "gpu decode + zero-copy import"),
    ] {
        let mut decoder = match open_video_decoder(&path, mode) {
            Ok(d) => d,
            Err(e) => {
                println!("{label}: skipped ({e})");
                continue;
            }
        };
        let config = CompositorConfig::new(1920, 1080);
        let mut samples = Vec::with_capacity(frames);
        let mut gpu_frames = 0usize;
        let total = Instant::now();
        for _ in 0..frames {
            let start = Instant::now();
            let frame = match decoder.next_frame() {
                Ok(Some(frame)) => frame,
                Ok(None) => break,
                Err(e) => {
                    println!("{label}: decode failed ({e})");
                    break;
                }
            };
            if frame.is_gpu() {
                gpu_frames += 1;
            }
            let layer = CompositeLayer::frame(&frame, LayerPlacement::full_canvas(&config));
            let image = comp.render(&gpu, &config, &[layer]).expect("composite");
            assert_eq!((image.width, image.height), (1920, 1080));
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        let total_s = total.elapsed().as_secs_f64();
        samples.sort_by(|a, b| a.total_cmp(b));
        let n = samples.len().max(1);
        println!(
            "{label:<32} {:>4} frames ({gpu_frames} gpu) | {:>7.2} ms mean | {:>7.2} ms p50 | {:>7.2} ms p95 | {:>6.1} eff fps",
            samples.len(),
            samples.iter().sum::<f64>() / n as f64,
            samples.get(n / 2).copied().unwrap_or(0.0),
            samples
                .get((n * 95 / 100).min(n - 1))
                .copied()
                .unwrap_or(0.0),
            if total_s > 0.0 {
                samples.len() as f64 / total_s
            } else {
                0.0
            },
        );
    }
}
