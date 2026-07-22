//! Per-cluster text animation hot path (no GPU).
//!
//! Run: `cargo bench -p cutlass-render --bench text_anim`

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use cutlass_core::RgbaImage;
use cutlass_models::AnimationSlot;
use cutlass_render::{TextAnimation, text_anim_bench};
use cutlass_text::{ClusterBox, ShapedText};

const CLUSTER_COUNT: usize = 256;

fn fake_shaped(n: usize) -> ShapedText {
    ShapedText {
        extent: ((n as u32) * 10, 24),
        clusters: (0..n)
            .map(|i| ClusterBox {
                text_range: i..i + 1,
                line: 0,
                offset: [i as f32 * 10.0, 0.0],
                baseline: 18.0,
                image: RgbaImage::new(8, 16, vec![255; 8 * 16 * 4]),
            })
            .collect(),
    }
}

fn anim() -> TextAnimation {
    TextAnimation {
        id: "wave".into(),
        slot: AnimationSlot::Combo,
        t: 0.37,
        intensity: 1.25,
        stagger: 0.8,
    }
}

fn bench_cluster_pipeline(c: &mut Criterion) {
    let shaped = fake_shaped(CLUSTER_COUNT);
    let animation = anim();
    let mut group = c.benchmark_group("text_anim");
    group.throughput(Throughput::Elements(CLUSTER_COUNT as u64));

    group.bench_function("cluster_deltas", |b| {
        b.iter(|| {
            black_box(text_anim_bench::cluster_deltas(
                black_box(&shaped),
                black_box(&animation),
            ))
        })
    });

    let deltas = text_anim_bench::cluster_deltas(&shaped, &animation);
    group.bench_function("place_clusters", |b| {
        b.iter(|| {
            black_box(text_anim_bench::place_clusters(
                black_box(&shaped),
                black_box(&deltas),
                black_box([40.0, 60.0]),
                black_box(1.0),
                black_box(0.1),
                black_box(1.0),
            ))
        })
    });

    group.bench_function("deltas_then_place", |b| {
        b.iter(|| {
            let deltas = text_anim_bench::cluster_deltas(&shaped, &animation);
            black_box(text_anim_bench::place_clusters(
                &shaped,
                &deltas,
                [40.0, 60.0],
                1.0,
                0.1,
                1.0,
            ))
        })
    });

    group.finish();
}

criterion_group!(benches, bench_cluster_pipeline);
criterion_main!(benches);
