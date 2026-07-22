use super::*;

fn kf(tick: i64, value: f32) -> Keyframe<f32> {
    Keyframe {
        tick,
        value,
        easing: Easing::Linear,
    }
}

// --- sampling -----------------------------------------------------------

#[test]
fn constant_samples_everywhere() {
    let p = Param::Constant(2.5f32);
    assert_eq!(p.sample(-100), 2.5);
    assert_eq!(p.sample(0), 2.5);
    assert_eq!(p.sample(i64::MAX), 2.5);
}

#[test]
fn keyframed_clamps_outside_range() {
    let p = Param::Keyframed {
        keyframes: vec![kf(10, 1.0), kf(20, 3.0)],
    };
    assert_eq!(p.sample(0), 1.0);
    assert_eq!(p.sample(10), 1.0);
    assert_eq!(p.sample(20), 3.0);
    assert_eq!(p.sample(1000), 3.0);
}

#[test]
fn linear_interpolation_between_keyframes() {
    let p = Param::Keyframed {
        keyframes: vec![kf(0, 0.0), kf(10, 10.0)],
    };
    assert_eq!(p.sample(5), 5.0);
    assert_eq!(p.sample(1), 1.0);
    assert_eq!(p.sample(9), 9.0);
}

#[test]
fn single_keyframe_acts_constant() {
    let p = Param::Keyframed {
        keyframes: vec![kf(50, 7.0)],
    };
    assert_eq!(p.sample(0), 7.0);
    assert_eq!(p.sample(50), 7.0);
    assert_eq!(p.sample(100), 7.0);
}

#[test]
fn multi_segment_picks_correct_pair() {
    let p = Param::Keyframed {
        keyframes: vec![kf(0, 0.0), kf(10, 100.0), kf(30, 0.0)],
    };
    assert_eq!(p.sample(5), 50.0);
    assert_eq!(p.sample(10), 100.0);
    assert_eq!(p.sample(20), 50.0);
}

#[test]
fn fractional_sampling_interpolates_between_ticks() {
    let p = Param::Keyframed {
        keyframes: vec![kf(0, 0.0), kf(10, 10.0)],
    };
    // Whole ticks agree with the integer path.
    assert_eq!(p.sample_at(5.0), p.sample(5));
    // Sub-tick positions land between frame values.
    assert_eq!(p.sample_at(5.5), 5.5);
    assert_eq!(p.sample_at(0.25), 0.25);
    // Clamping matches the integer path on both sides.
    assert_eq!(p.sample_at(-3.7), 0.0);
    assert_eq!(p.sample_at(10.4), 10.0);
    assert_eq!(Param::Constant(2.5f32).sample_at(1.5), 2.5);
}

#[test]
fn vec2_lerp() {
    let p = Param::Keyframed {
        keyframes: vec![
            Keyframe {
                tick: 0,
                value: [0.0, 0.0],
                easing: Easing::Linear,
            },
            Keyframe {
                tick: 10,
                value: [1.0, -1.0],
                easing: Easing::Linear,
            },
        ],
    };
    assert_eq!(p.sample(5), [0.5, -0.5]);
}

// --- easing ---------------------------------------------------------------

#[test]
fn easing_endpoints_are_exact() {
    for easing in [
        Easing::Linear,
        Easing::EaseIn,
        Easing::EaseOut,
        Easing::EaseInOut,
        Easing::Bezier {
            points: [0.42, 0.0, 0.58, 1.0],
        },
    ] {
        assert_eq!(easing.apply(0.0), 0.0, "{easing:?} at 0");
        assert!((easing.apply(1.0) - 1.0).abs() < 1e-4, "{easing:?} at 1");
    }
}

#[test]
fn ease_in_starts_slow_ease_out_starts_fast() {
    assert!(Easing::EaseIn.apply(0.25) < 0.25);
    assert!(Easing::EaseOut.apply(0.25) > 0.25);
    let mid = Easing::EaseInOut.apply(0.5);
    assert!((mid - 0.5).abs() < 1e-6);
}

#[test]
fn named_easing_presets_roundtrip_and_validate() {
    assert_eq!(EASING_PRESETS.len(), 3);
    for preset in EASING_PRESETS {
        let easing = Easing::from_preset_id(preset.id).expect(preset.id);
        assert_eq!(easing.preset_id(), Some(preset.id));
        assert!(easing.validate().is_ok(), "{}", preset.id);
        assert_eq!(easing.apply(0.0), 0.0);
        assert!((easing.apply(1.0) - 1.0).abs() < 1e-4);
    }
    assert!(Easing::from_preset_id("bounce").is_none());
    assert_eq!(
        Easing::Bezier {
            points: [0.42, 0.0, 0.58, 1.0],
        }
        .preset_id(),
        None
    );
}

#[test]
fn bezier_matches_css_ease_in_out_shape() {
    // cubic-bezier(0.42, 0, 0.58, 1) — CSS "ease-in-out".
    let e = Easing::Bezier {
        points: [0.42, 0.0, 0.58, 1.0],
    };
    assert!(e.apply(0.1) < 0.1);
    assert!(e.apply(0.9) > 0.9);
    assert!((e.apply(0.5) - 0.5).abs() < 1e-3);
    // Monotonic over a sweep.
    let mut prev = 0.0;
    for i in 0..=100 {
        let v = e.apply(i as f32 / 100.0);
        assert!(v >= prev - 1e-4, "non-monotonic at {i}");
        prev = v;
    }
}

#[test]
fn easing_subsegments_preserve_original_progress() {
    const FROM: f32 = 0.2;
    const TO: f32 = 0.8;
    for easing in [
        Easing::Linear,
        Easing::EaseIn,
        Easing::EaseOut,
        Easing::EaseInOut,
        Easing::Bezier {
            points: [0.42, 0.0, 0.58, 1.0],
        },
    ] {
        let sliced = easing.subsegment(FROM, TO).unwrap();
        let start = easing.apply(FROM);
        let span = easing.apply(TO) - start;
        for step in 0..=20 {
            let u = step as f32 / 20.0;
            let original = easing.apply(FROM + (TO - FROM) * u);
            let expected = (original - start) / span;
            assert!(
                (sliced.apply(u) - expected).abs() < 2e-4,
                "{easing:?} slice diverged at {u}"
            );
        }
    }
}

#[test]
fn easing_integrals_match_closed_form_endpoints() {
    // ∫₀¹ of each easing over the unit interval.
    assert!((Easing::Linear.integral_to(1.0) - 0.5).abs() < 1e-6);
    assert!((Easing::EaseIn.integral_to(1.0) - 1.0 / 3.0).abs() < 1e-6);
    assert!((Easing::EaseOut.integral_to(1.0) - 2.0 / 3.0).abs() < 1e-6);
    assert!((Easing::EaseInOut.integral_to(1.0) - 0.5).abs() < 1e-6);
    // The symmetric CSS ease-in-out bezier integrates to ½ by symmetry.
    let e = Easing::Bezier {
        points: [0.42, 0.0, 0.58, 1.0],
    };
    assert!((e.integral_to(1.0) - 0.5).abs() < 1e-3);
    // Integral is 0 at t=0 and monotonic increasing.
    for easing in [
        Easing::Linear,
        Easing::EaseIn,
        Easing::EaseOut,
        Easing::EaseInOut,
    ] {
        assert_eq!(easing.integral_to(0.0), 0.0);
        let mut prev = 0.0;
        for i in 0..=20 {
            let v = easing.integral_to(i as f32 / 20.0);
            assert!(v >= prev - 1e-6, "{easing:?} non-monotonic integral");
            prev = v;
        }
    }
}

#[test]
fn bezier_validation_rejects_bad_x() {
    assert!(
        Easing::Bezier {
            points: [1.5, 0.0, 0.5, 1.0]
        }
        .validate()
        .is_err()
    );
    assert!(
        Easing::Bezier {
            points: [0.5, 0.0, -0.1, 1.0]
        }
        .validate()
        .is_err()
    );
    assert!(
        Easing::Bezier {
            points: [0.5, f32::NAN, 0.5, 1.0]
        }
        .validate()
        .is_err()
    );
    // Overshooting y is allowed (CSS semantics).
    assert!(
        Easing::Bezier {
            points: [0.3, -0.5, 0.7, 1.5]
        }
        .validate()
        .is_ok()
    );
}

// --- mutation ---------------------------------------------------------------

#[test]
fn set_keyframe_on_constant_becomes_curve() {
    let mut p = Param::Constant(1.0f32);
    p.set_keyframe(10, 2.0, Easing::Linear);
    assert!(p.is_animated());
    assert_eq!(p.keyframes().len(), 1);
    assert_eq!(p.sample(10), 2.0);
}

#[test]
fn set_keyframe_inserts_sorted_and_replaces() {
    let mut p = Param::Constant(0.0f32);
    p.set_keyframe(20, 2.0, Easing::Linear);
    p.set_keyframe(0, 0.0, Easing::Linear);
    p.set_keyframe(10, 1.0, Easing::Linear);
    let ticks: Vec<i64> = p.keyframes().iter().map(|k| k.tick).collect();
    assert_eq!(ticks, vec![0, 10, 20]);

    p.set_keyframe(10, 5.0, Easing::EaseIn);
    assert_eq!(p.keyframes().len(), 3);
    assert_eq!(p.sample(10), 5.0);
}

#[test]
fn remove_keyframe_collapses_last_to_constant() {
    let mut p = Param::Constant(0.0f32);
    p.set_keyframe(10, 7.0, Easing::Linear);
    assert!(!p.remove_keyframe(5), "no keyframe at 5");
    assert!(p.remove_keyframe(10));
    assert!(!p.is_animated());
    assert_eq!(p.constant(), Some(7.0));
}

#[test]
fn set_constant_wipes_keyframes() {
    let mut p = Param::Constant(0.0f32);
    p.set_keyframe(10, 1.0, Easing::Linear);
    p.set_keyframe(20, 2.0, Easing::Linear);
    p.set_constant(9.0);
    assert_eq!(p.constant(), Some(9.0));
    assert!(p.keyframes().is_empty());
}

// --- serde -----------------------------------------------------------------

#[test]
fn constant_serializes_as_bare_value() {
    let p = Param::Constant(1.5f32);
    assert_eq!(serde_json::to_string(&p).unwrap(), "1.5");
    let v: Param<[f32; 2]> = Param::Constant([0.0, 0.25]);
    assert_eq!(serde_json::to_string(&v).unwrap(), "[0.0,0.25]");
}

#[test]
fn bare_value_deserializes_as_constant() {
    let p: Param<f32> = serde_json::from_str("2.0").unwrap();
    assert_eq!(p, Param::Constant(2.0));
    let v: Param<[f32; 2]> = serde_json::from_str("[0.1,0.2]").unwrap();
    assert_eq!(v, Param::Constant([0.1, 0.2]));
}

#[test]
fn keyframed_roundtrips_compactly() {
    let p = Param::Keyframed {
        keyframes: vec![
            Keyframe {
                tick: 0,
                value: 1.0f32,
                easing: Easing::Linear,
            },
            Keyframe {
                tick: 24,
                value: 2.0,
                easing: Easing::EaseInOut,
            },
        ],
    };
    let json = serde_json::to_string(&p).unwrap();
    // Linear easing is elided; non-linear spelled out.
    assert_eq!(
        json,
        r#"{"kf":[{"t":0,"v":1.0},{"t":24,"v":2.0,"e":"ease_in_out"}]}"#
    );
    let back: Param<f32> = serde_json::from_str(&json).unwrap();
    assert_eq!(back, p);
}

#[test]
fn bezier_easing_roundtrips() {
    let p = Param::Keyframed {
        keyframes: vec![
            Keyframe {
                tick: 0,
                value: 0.0f32,
                easing: Easing::Bezier {
                    points: [0.42, 0.0, 0.58, 1.0],
                },
            },
            Keyframe {
                tick: 10,
                value: 1.0,
                easing: Easing::Linear,
            },
        ],
    };
    let json = serde_json::to_string(&p).unwrap();
    let back: Param<f32> = serde_json::from_str(&json).unwrap();
    assert_eq!(back, p);
}

#[test]
fn vec2_keyframed_roundtrips() {
    let p = Param::Keyframed {
        keyframes: vec![
            Keyframe {
                tick: 0,
                value: [0.0f32, 0.0],
                easing: Easing::Linear,
            },
            Keyframe {
                tick: 48,
                value: [0.5, -0.5],
                easing: Easing::EaseOut,
            },
        ],
    };
    let json = serde_json::to_string(&p).unwrap();
    let back: Param<[f32; 2]> = serde_json::from_str(&json).unwrap();
    assert_eq!(back, p);
}

// --- validation -------------------------------------------------------------

#[test]
fn validate_shape_rejects_unsorted_and_empty() {
    let unsorted: Param<f32> = Param::Keyframed {
        keyframes: vec![kf(10, 1.0), kf(5, 2.0)],
    };
    assert!(unsorted.validate_shape().is_err());

    let dup: Param<f32> = Param::Keyframed {
        keyframes: vec![kf(10, 1.0), kf(10, 2.0)],
    };
    assert!(dup.validate_shape().is_err());

    let empty: Param<f32> = Param::Keyframed { keyframes: vec![] };
    assert!(empty.validate_shape().is_err());

    let ok: Param<f32> = Param::Keyframed {
        keyframes: vec![kf(0, 1.0), kf(10, 2.0)],
    };
    assert!(ok.validate_shape().is_ok());
    assert!(Param::Constant(1.0f32).validate_shape().is_ok());
}
