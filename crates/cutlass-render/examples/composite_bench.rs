//! Compositor pipeline benchmark over a real media file.
//!
//! Measures decode, compositor-only (warm), and decode+composite end-to-end at
//! several canvas sizes — the breakdown that decides whether preview keeps cadence.
//!
//! Run:
//!   `cargo run --release -p cutlass-render --example composite_bench -- <media> [frames]`
//!
//! `frames` caps how many sequential samples each phase takes (default 60).

use std::path::{Path, PathBuf};
use std::time::Instant;

use cutlass_compositor::{
    ColorGrade, CompositeLayer, Compositor, CompositorConfig, GpuContext, LayerChromaKey,
    LayerEffects, LayerMask, LayerPlacement, PassInstance, mask_kind,
};
use cutlass_core::{RationalTime, RgbaImage};
use cutlass_decoder::{OutputMode, open_video_decoder, probe};
use cutlass_models::AnimationSlot;
use cutlass_render::{TextAnimation, text_anim_bench};
use cutlass_text::{ClusterBox, ShapedText};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .map(PathBuf::from)
        .expect("usage: composite_bench <media> [frames]");
    let frames: i64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(60);

    // CPU-only glyph-instance path — useful even when GPU / media is unavailable.
    bench_animated_text_cpu(frames.max(1) as usize);
    println!();

    let gpu = match GpuContext::new_headless_blocking() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no GPU adapter: {e}");
            return;
        }
    };
    let backend = gpu.adapter.get_info().backend;
    println!("gpu backend: {backend:?}\n");

    let info = probe(&path).expect("probe failed");
    let fps = info.frame_rate;
    println!("media: {}", path.display());
    println!(
        "  {}x{} @ {:.3} fps, {} frames total",
        info.width,
        info.height,
        fps.as_f64(),
        info.frame_count,
    );
    println!("  benching {frames} frames per phase\n");

    bench_decode(&path, frames);
    println!();

    let canvas_sizes = [
        ("preview_960x540", 960u32, 540u32),
        ("1080p", 1920, 1080),
        ("native_4k", info.width, info.height),
    ];

    for (mode, label) in [
        (OutputMode::Cpu, "cpu_upload"),
        (OutputMode::Gpu, "gpu_import"),
    ] {
        println!("--- {label} ---");
        if let Some(frame) = prime_frame(&path, mode) {
            bench_compositor_only(&gpu, &frame, &canvas_sizes, frames, label);
            bench_effect_overhead(&gpu, &frame, frames, label);
            bench_compositor_with_look(&gpu, &frame, &canvas_sizes, frames, label);
        }
        bench_decode_and_composite(
            &gpu,
            &path,
            mode,
            &canvas_sizes,
            fps,
            frames,
            info.frame_count.max(1),
            label,
        );
        println!();
    }
}

/// Per-cluster text animation (shape → deltas → glyph instances). No GPU.
fn bench_animated_text_cpu(iters: usize) {
    println!("=== animated text (CPU: cluster_deltas + place_clusters) ===");
    const N: usize = 128;
    let shaped = ShapedText {
        extent: ((N as u32) * 10, 24),
        clusters: (0..N)
            .map(|i| ClusterBox {
                text_range: i..i + 1,
                line: 0,
                offset: [i as f32 * 10.0, 0.0],
                baseline: 18.0,
                image: RgbaImage::new(8, 16, vec![255; 8 * 16 * 4]),
            })
            .collect(),
    };
    let anim = TextAnimation {
        id: "wave".into(),
        slot: AnimationSlot::Combo,
        t: 0.42,
        intensity: 1.1,
        stagger: 0.9,
    };

    let mut samples = Vec::with_capacity(iters);
    let total = Instant::now();
    for i in 0..iters {
        let mut a = anim.clone();
        a.t = (i as f32 / iters as f32).fract();
        let start = Instant::now();
        let deltas = text_anim_bench::cluster_deltas(&shaped, &a);
        let instances =
            text_anim_bench::place_clusters(&shaped, &deltas, [32.0, 48.0], [1.0, 1.0], 0.0, 1.0);
        std::hint::black_box(instances.len());
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let stats = Stats::from_samples(samples, total.elapsed().as_secs_f64());
    stats.print(&format!("  wave {N} clusters × {iters}"));
}

fn prime_frame(path: &Path, mode: OutputMode) -> Option<cutlass_core::VideoFrame> {
    let mut dec = match open_video_decoder(path, mode) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("prime ({mode:?}): {e}");
            return None;
        }
    };
    match dec.next_frame() {
        Ok(Some(f)) => Some(f),
        Ok(None) => {
            eprintln!("prime ({mode:?}): empty stream");
            None
        }
        Err(e) => {
            eprintln!("prime ({mode:?}): {e}");
            None
        }
    }
}

/// Raw decode throughput (no compositor).
fn bench_decode(path: &Path, frames: i64) {
    println!("=== decode only (baseline) ===");
    for (mode, label) in [(OutputMode::Cpu, "cpu"), (OutputMode::Gpu, "gpu")] {
        let _ = bench_sequential_decode(path, mode, label, frames);
    }
}

fn bench_sequential_decode(
    path: &Path,
    mode: OutputMode,
    label: &str,
    frames: i64,
) -> Option<Stats> {
    let mut dec = match open_video_decoder(path, mode) {
        Ok(d) => d,
        Err(e) => {
            println!("decode ({label}): skipped ({e})");
            return None;
        }
    };
    let mut samples = Vec::with_capacity(frames as usize);
    let total = Instant::now();
    for _ in 0..frames {
        let start = Instant::now();
        match dec.next_frame() {
            Ok(Some(frame)) => {
                samples.push(start.elapsed().as_secs_f64() * 1000.0);
                drop(frame);
            }
            Ok(None) => break,
            Err(e) => {
                println!("decode ({label}): skipped ({e})");
                return None;
            }
        }
    }
    let stats = Stats::from_samples(samples, total.elapsed().as_secs_f64());
    stats.print(&format!("decode sequential ({label})"));
    Some(stats)
}

/// Hold one decoded frame; time compositor + readback only (warm target cache).
fn bench_compositor_only(
    gpu: &GpuContext,
    frame: &cutlass_core::VideoFrame,
    sizes: &[(&str, u32, u32)],
    iters: i64,
    import: &str,
) {
    println!("compositor-only ({import}):");
    let fx_effects = LayerEffects {
        mask: Some(LayerMask::new(mask_kind::CIRCLE)),
        chroma_key: Some(LayerChromaKey {
            rgb: [0.0, 1.0, 0.0],
            strength: 0.5,
            shadow: 0.0,
        }),
    };
    let mut comp = Compositor::new(gpu);
    for &(label, w, h) in sizes {
        let config = CompositorConfig::new(w, h);
        let placement = LayerPlacement::full_canvas(&config);
        for tag in ["baseline", "fx"] {
            let layer = if tag == "baseline" {
                CompositeLayer::frame(frame, placement)
            } else {
                CompositeLayer::frame(frame, placement).with_fx(fx_effects)
            };
            // Prime pipeline + target cache.
            let _ = comp.render(gpu, &config, &[layer]).expect("prime");

            let mut samples = Vec::with_capacity(iters as usize);
            let total = Instant::now();
            for _ in 0..iters {
                let start = Instant::now();
                let layer = if tag == "baseline" {
                    CompositeLayer::frame(frame, placement)
                } else {
                    CompositeLayer::frame(frame, placement).with_fx(fx_effects)
                };
                let img = comp.render(gpu, &config, &[layer]).expect("render");
                samples.push(start.elapsed().as_secs_f64() * 1000.0);
                std::hint::black_box(img.pixel(0, 0));
            }
            let stats = Stats::from_samples(samples, total.elapsed().as_secs_f64());
            stats.print(&format!("  composite {label} {w}x{h} ({tag})"));
        }
    }
}

/// Effect-free vs passthrough-chain vs blur at preview resolution.
fn bench_effect_overhead(
    gpu: &GpuContext,
    frame: &cutlass_core::VideoFrame,
    iters: i64,
    import: &str,
) {
    let (w, h) = (960u32, 540u32);
    println!("effect overhead ({import}) @ {w}x{h}:");
    let config = CompositorConfig::new(w, h);
    let placement = LayerPlacement::full_canvas(&config);
    let mut comp = Compositor::new(gpu);

    let baseline_layer = CompositeLayer::frame(frame, placement);
    let _ = comp
        .render(gpu, &config, &[baseline_layer])
        .expect("prime baseline");

    let passthrough = [PassInstance {
        id: "sharpen",
        params: &[0.5],
    }];
    let blur = [PassInstance {
        id: "gaussian_blur",
        params: &[8.0],
    }];

    let cases: [(&str, &[PassInstance]); 3] = [
        ("effect_free", &[]),
        ("passthrough_sharpen", &passthrough),
        ("blur_r8", &blur),
    ];

    let mut baseline_mean = 0.0f64;
    for (i, (name, effects)) in cases.iter().enumerate() {
        let mut samples = Vec::with_capacity(iters as usize);
        let total = Instant::now();
        for _ in 0..iters {
            let layer = CompositeLayer::frame(frame, placement).with_effects(effects);
            let start = Instant::now();
            let img = comp.render(gpu, &config, &[layer]).expect("render");
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
            std::hint::black_box(img.pixel(0, 0));
        }
        let stats = Stats::from_samples(samples, total.elapsed().as_secs_f64());
        if i == 0 {
            baseline_mean = stats.mean;
        }
        let delta = stats.mean - baseline_mean;
        stats.print(&format!("  {name:<22} Δ{delta:+.2} ms vs baseline"));
    }
}

/// Same as compositor-only, but every layer carries a non-identity color grade.
fn bench_compositor_with_look(
    gpu: &GpuContext,
    frame: &cutlass_core::VideoFrame,
    sizes: &[(&str, u32, u32)],
    iters: i64,
    import: &str,
) {
    println!("compositor-only with look ({import}):");
    let grade = ColorGrade {
        saturation: 0.35,
        contrast: 0.2,
        exposure: 0.05,
        ..ColorGrade::default()
    };
    let mut comp = Compositor::new(gpu);
    for &(label, w, h) in sizes {
        let config = CompositorConfig::new(w, h);
        let placement = LayerPlacement::full_canvas(&config);
        let make_layer = || CompositeLayer::frame(frame, placement).with_color_grade(Some(grade));
        let _ = comp.render(gpu, &config, &[make_layer()]).expect("prime");

        let mut samples = Vec::with_capacity(iters as usize);
        let total = Instant::now();
        for _ in 0..iters {
            let start = Instant::now();
            let img = comp.render(gpu, &config, &[make_layer()]).expect("render");
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
            std::hint::black_box(img.pixel(0, 0));
        }
        let stats = Stats::from_samples(samples, total.elapsed().as_secs_f64());
        stats.print(&format!("  composite+look {label} {w}x{h}"));
    }
}

/// Decode + composite end-to-end: sequential playback and random scrub.
#[allow(clippy::too_many_arguments)]
fn bench_decode_and_composite(
    gpu: &GpuContext,
    path: &Path,
    mode: OutputMode,
    sizes: &[(&str, u32, u32)],
    fps: cutlass_core::Rational,
    frames: i64,
    total_frames: i64,
    import: &str,
) {
    println!("decode + composite ({import}):");
    for &(label, w, h) in sizes {
        let Ok(mut dec) = open_video_decoder(path, mode) else {
            return;
        };
        let config = CompositorConfig::new(w, h);
        let mut comp = Compositor::new(gpu);
        let mut samples = Vec::with_capacity(frames as usize);
        let total = Instant::now();
        for i in 0..frames {
            let target = RationalTime::new(i, fps);
            let start = Instant::now();
            let frame = dec.frame_at(target).expect("frame_at").expect("frame");
            let layer = CompositeLayer::frame(&frame, LayerPlacement::full_canvas(&config));
            let img = comp.render(gpu, &config, &[layer]).expect("render");
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
            std::hint::black_box(img.pixel(0, 0));
        }
        let stats = Stats::from_samples(samples, total.elapsed().as_secs_f64());
        stats.print(&format!("  playback {label} {w}x{h}"));
    }

    // Random scrub at preview resolution — mostly real seeks.
    let (label, w, h) = sizes[0];
    let Ok(mut dec) = open_video_decoder(path, mode) else {
        return;
    };
    let config = CompositorConfig::new(w, h);
    let mut comp = Compositor::new(gpu);
    let jumps = 48usize;
    let mut state: u64 = 0x243F_6A88_85A3_08D3;
    let mut samples = Vec::with_capacity(jumps);
    let total = Instant::now();
    for _ in 0..jumps {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let tick = ((state >> 33) % total_frames.max(1) as u64) as i64;
        let target = RationalTime::new(tick, fps);
        let start = Instant::now();
        let frame = match dec.frame_at(target) {
            Ok(Some(f)) => f,
            Ok(None) => continue,
            Err(e) => {
                println!("  scrub: {e}");
                return;
            }
        };
        let layer = CompositeLayer::frame(&frame, LayerPlacement::full_canvas(&config));
        let img = comp.render(gpu, &config, &[layer]).expect("render");
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(img.pixel(0, 0));
    }
    let stats = Stats::from_samples(samples, total.elapsed().as_secs_f64());
    stats.print(&format!("  random scrub {label} {w}x{h} (x{jumps})"));
}

#[derive(Clone, Copy)]
struct Stats {
    mean: f64,
    p50: f64,
    p95: f64,
    max: f64,
    count: usize,
    eff_fps: f64,
}

impl Stats {
    fn from_samples(mut samples_ms: Vec<f64>, total_s: f64) -> Self {
        if samples_ms.is_empty() {
            return Self {
                mean: 0.0,
                p50: 0.0,
                p95: 0.0,
                max: 0.0,
                count: 0,
                eff_fps: 0.0,
            };
        }
        samples_ms.sort_by(|a, b| a.total_cmp(b));
        let n = samples_ms.len();
        Self {
            mean: samples_ms.iter().sum::<f64>() / n as f64,
            p50: samples_ms[n / 2],
            p95: samples_ms[(n * 95 / 100).min(n - 1)],
            max: samples_ms[n - 1],
            count: n,
            eff_fps: n as f64 / total_s.max(f64::EPSILON),
        }
    }

    fn print(&self, label: &str) {
        println!(
            "{label:<48} {:>4} | {:>7.2} ms mean | {:>7.2} ms p50 | {:>7.2} ms p95 | {:>7.2} ms max | {:>6.1} fps",
            self.count, self.mean, self.p50, self.p95, self.max, self.eff_fps,
        );
    }
}
