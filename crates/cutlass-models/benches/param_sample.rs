//! Hot-path Param / AnimatedTransform / LayerStyles sampling.
//!
//! Run: `cargo bench -p cutlass-models --bench param_sample`

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use cutlass_models::{AnimatedTransform, Easing, Keyframe, LayerShadow, LayerStyles, Param};

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
                value: 0.5.into(),
                easing: Easing::EaseOut,
            },
            Keyframe {
                tick: 60,
                value: 1.25.into(),
                easing: Easing::from_preset_id("overshoot").unwrap(),
            },
            Keyframe {
                tick: 120,
                value: 1.0.into(),
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

/// Per-frame cost of sampling a keyframed layer-style shadow (blur + offset
/// + color) — what resolve pays before handing styles to the compositor.
fn keyframed_layer_styles() -> LayerStyles {
    LayerStyles {
        shadow: Some(LayerShadow {
            rgba: Param::Keyframed {
                keyframes: vec![
                    Keyframe {
                        tick: 0,
                        value: [0, 0, 0, 64],
                        easing: Easing::Linear,
                    },
                    Keyframe {
                        tick: 48,
                        value: [0, 0, 0, 200],
                        easing: Easing::EaseInOut,
                    },
                    Keyframe {
                        tick: 96,
                        value: [20, 10, 0, 128],
                        easing: Easing::from_preset_id("snappy").unwrap(),
                    },
                ],
            },
            offset: Param::Keyframed {
                keyframes: vec![
                    Keyframe {
                        tick: 0,
                        value: [0.0, 0.0],
                        easing: Easing::EaseOut,
                    },
                    Keyframe {
                        tick: 60,
                        value: [12.0, 8.0],
                        easing: Easing::from_preset_id("overshoot").unwrap(),
                    },
                    Keyframe {
                        tick: 120,
                        value: [4.0, 4.0],
                        easing: Easing::Linear,
                    },
                ],
            },
            blur: Param::Keyframed {
                keyframes: vec![
                    Keyframe {
                        tick: 0,
                        value: 0.0,
                        easing: Easing::EaseIn,
                    },
                    Keyframe {
                        tick: 40,
                        value: 24.0,
                        easing: Easing::EaseInOut,
                    },
                    Keyframe {
                        tick: 100,
                        value: 8.0,
                        easing: Easing::Linear,
                    },
                ],
            },
        }),
        ..Default::default()
    }
}

fn sample_styles_shadow(styles: &LayerStyles, tick: i64) -> f32 {
    let shadow = styles.shadow.as_ref().expect("shadow");
    let blur = shadow.blur.sample(tick);
    let offset = shadow.offset.sample(tick);
    let rgba = shadow.rgba.sample(tick);
    blur + offset[0] + offset[1] + f32::from(rgba[3])
}

fn bench_layer_styles_sample(c: &mut Criterion) {
    let styles = keyframed_layer_styles();
    c.bench_function("layer_styles_shadow_sample_mid", |b| {
        b.iter(|| black_box(sample_styles_shadow(&styles, black_box(45))))
    });
    c.bench_function("layer_styles_shadow_sample_sweep", |b| {
        b.iter(|| {
            let mut acc = 0.0f32;
            for tick in 0..128 {
                acc += sample_styles_shadow(&styles, tick);
            }
            black_box(acc)
        })
    });
}

criterion_group!(
    benches,
    bench_param_sample,
    bench_transform_sample,
    bench_layer_styles_sample
);
criterion_main!(benches);
