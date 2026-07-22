//! Hot-path Param / AnimatedTransform sampling.
//!
//! Run: `cargo bench -p cutlass-models --bench param_sample`

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use cutlass_models::{AnimatedTransform, Easing, Keyframe, Param};

fn keyframed_scalar() -> Param<f32> {
    Param::Keyframed {
        keyframes: vec![
            Keyframe {
                tick: 0,
                value: 0.0,
                easing: Easing::EaseInOut,
            },
            Keyframe {
                tick: 24,
                value: 1.0,
                easing: Easing::from_preset_id("snappy").unwrap(),
            },
            Keyframe {
                tick: 48,
                value: 0.25,
                easing: Easing::Bezier {
                    points: [0.42, 0.0, 0.58, 1.0],
                },
            },
            Keyframe {
                tick: 96,
                value: 1.0,
                easing: Easing::Linear,
            },
        ],
    }
}

fn keyframed_transform() -> AnimatedTransform {
    let mut t = AnimatedTransform::identity();
    t.opacity = keyframed_scalar();
    t.scale = Param::Keyframed {
        keyframes: vec![
            Keyframe {
                tick: 0,
                value: 0.5,
                easing: Easing::EaseOut,
            },
            Keyframe {
                tick: 60,
                value: 1.25,
                easing: Easing::from_preset_id("overshoot").unwrap(),
            },
            Keyframe {
                tick: 120,
                value: 1.0,
                easing: Easing::Linear,
            },
        ],
    };
    t.position = Param::Keyframed {
        keyframes: vec![
            Keyframe {
                tick: 0,
                value: [-0.2, 0.0],
                easing: Easing::from_preset_id("anticipate").unwrap(),
            },
            Keyframe {
                tick: 90,
                value: [0.2, 0.1],
                easing: Easing::EaseInOut,
            },
        ],
    };
    t
}

fn bench_param_sample(c: &mut Criterion) {
    let param = keyframed_scalar();
    c.bench_function("param_f32_sample_mid", |b| {
        b.iter(|| black_box(param.sample(black_box(37))))
    });
    c.bench_function("param_f32_sample_sweep", |b| {
        b.iter(|| {
            let mut acc = 0.0f32;
            for tick in 0..128 {
                acc += param.sample(tick);
            }
            black_box(acc)
        })
    });
}

fn bench_transform_sample(c: &mut Criterion) {
    let xform = keyframed_transform();
    c.bench_function("animated_transform_sample_mid", |b| {
        b.iter(|| black_box(xform.sample(black_box(45))))
    });
    c.bench_function("animated_transform_sample_sweep", |b| {
        b.iter(|| {
            let mut acc = 0.0f32;
            for tick in 0..128 {
                acc += xform.sample(tick).opacity;
            }
            black_box(acc)
        })
    });
}

criterion_group!(benches, bench_param_sample, bench_transform_sample);
criterion_main!(benches);
