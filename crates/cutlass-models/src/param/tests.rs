use super::*;

fn kf(tick: i64, value: f32) -> Keyframe<f32> {
    Keyframe::new(tick, value, Easing::Linear)
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
                tangents: None,
            },
            Keyframe {
                tick: 10,
                value: [1.0, -1.0],
                easing: Easing::Linear,
                tangents: None,
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
fn hold_keeps_value_until_next_keyframe() {
    let p = Param::Keyframed {
        keyframes: vec![
            Keyframe {
                tick: 0,
                value: 1.0f32,
                easing: Easing::Hold,
                tangents: None,
            },
            Keyframe {
                tick: 10,
                value: 5.0,
                easing: Easing::Linear,
                tangents: None,
            },
        ],
    };
    // k0's value holds through the whole open interval…
    assert_eq!(p.sample(0), 1.0);
    assert_eq!(p.sample(5), 1.0);
    assert_eq!(p.sample(9), 1.0);
    assert_eq!(p.sample_at(9.99), 1.0);
    // …and jumps to k1's value exactly at k1's tick.
    assert_eq!(p.sample(10), 5.0);
    assert_eq!(p.sample(11), 5.0);
}

#[test]
fn hold_between_curved_segments() {
    // linear → hold → linear: neighbors are unaffected by the step.
    let p = Param::Keyframed {
        keyframes: vec![
            Keyframe {
                tick: 0,
                value: 0.0f32,
                easing: Easing::Linear,
                tangents: None,
            },
            Keyframe {
                tick: 10,
                value: 10.0,
                easing: Easing::Hold,
                tangents: None,
            },
            Keyframe {
                tick: 20,
                value: 20.0,
                easing: Easing::Linear,
                tangents: None,
            },
            Keyframe {
                tick: 30,
                value: 30.0,
                easing: Easing::Linear,
                tangents: None,
            },
        ],
    };
    assert_eq!(p.sample(5), 5.0); // linear segment interpolates
    assert_eq!(p.sample(10), 10.0);
    assert_eq!(p.sample(15), 10.0); // hold segment stays at 10
    assert_eq!(p.sample(19), 10.0);
    assert_eq!(p.sample(20), 20.0); // jump lands on the next keyframe
    assert_eq!(p.sample(25), 25.0); // trailing linear segment unaffected
}

#[test]
fn hold_easing_roundtrips_as_hold_tag() {
    let p = Param::Keyframed {
        keyframes: vec![
            Keyframe {
                tick: 0,
                value: 1.0f32,
                easing: Easing::Hold,
                tangents: None,
            },
            Keyframe {
                tick: 12,
                value: 2.0,
                easing: Easing::Linear,
                tangents: None,
            },
        ],
    };
    let json = serde_json::to_string(&p).unwrap();
    assert_eq!(
        json,
        r#"{"kf":[{"t":0,"v":1.0,"e":"hold"},{"t":12,"v":2.0}]}"#
    );
    let back: Param<f32> = serde_json::from_str(&json).unwrap();
    assert_eq!(back, p);
}

#[test]
fn hold_subsegment_stays_hold() {
    // Interior slice: no jump inside, still a hold.
    assert_eq!(Easing::Hold.subsegment(0.2, 0.8).unwrap(), Easing::Hold);
    // Slice reaching the endpoint keeps the jump at its own end.
    assert_eq!(Easing::Hold.subsegment(0.5, 1.0).unwrap(), Easing::Hold);
    // Invalid ranges still fail closed.
    assert!(Easing::Hold.subsegment(0.8, 0.2).is_err());
}

#[test]
fn hold_integral_is_zero() {
    assert_eq!(Easing::Hold.integral_to(0.0), 0.0);
    assert_eq!(Easing::Hold.integral_to(0.5), 0.0);
    assert_eq!(Easing::Hold.integral_to(1.0), 0.0);
}

#[test]
fn hold_validates_ok() {
    assert!(Easing::Hold.validate().is_ok());
    assert_eq!(Easing::Hold.preset_id(), None);
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
                tangents: None,
            },
            Keyframe {
                tick: 24,
                value: 2.0,
                easing: Easing::EaseInOut,
                tangents: None,
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
                tangents: None,
            },
            Keyframe {
                tick: 10,
                value: 1.0,
                easing: Easing::Linear,
                tangents: None,
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
                tangents: None,
            },
            Keyframe {
                tick: 48,
                value: [0.5, -0.5],
                easing: Easing::EaseOut,
                tangents: None,
            },
        ],
    };
    let json = serde_json::to_string(&p).unwrap();
    let back: Param<[f32; 2]> = serde_json::from_str(&json).unwrap();
    assert_eq!(back, p);
}

// --- spatial tangents (motion paths) ----------------------------------------

#[test]
fn legacy_vec2_keyframe_json_omits_tangents_slot() {
    let json = r#"{"kf":[{"t":0,"v":[0.0,0.0]},{"t":10,"v":[1.0,1.0]}]}"#;
    let p: Param<[f32; 2]> = serde_json::from_str(json).unwrap();
    assert!(p.keyframes().iter().all(|kf| kf.tangents.is_none()));
    assert_eq!(serde_json::to_string(&p).unwrap(), json);
}

#[test]
fn spatial_tangents_roundtrip_compactly() {
    let p = Param::Keyframed {
        keyframes: vec![
            Keyframe {
                tick: 0,
                value: [0.0, 0.0],
                easing: Easing::Linear,
                tangents: Some(SpatialTangents {
                    out_t: [0.0, 0.55],
                    in_t: [0.0, 0.0],
                }),
            },
            Keyframe {
                tick: 10,
                value: [1.0, 1.0],
                easing: Easing::Linear,
                tangents: Some(SpatialTangents {
                    out_t: [0.0, 0.0],
                    in_t: [-0.55, 0.0],
                }),
            },
        ],
    };
    let json = serde_json::to_string(&p).unwrap();
    assert!(json.contains(r#""s":{"o":[0.0,0.55]}"#));
    assert!(json.contains(r#""s":{"i":[-0.55,0.0]}"#));
    let back: Param<[f32; 2]> = serde_json::from_str(&json).unwrap();
    assert_eq!(back, p);
}

#[test]
fn straight_line_position_matches_legacy_lerp_goldens() {
    // No tangents → bit-identical to the pre-motion-path straight lerp.
    let p = Param::Keyframed {
        keyframes: vec![
            Keyframe::new(0, [0.0, 0.0], Easing::Linear),
            Keyframe::new(10, [1.0, -1.0], Easing::Linear),
        ],
    };
    assert_eq!(p.sample(0), [0.0, 0.0]);
    assert_eq!(p.sample(5), [0.5, -0.5]);
    assert_eq!(p.sample(10), [1.0, -1.0]);
    assert_eq!(p.sample_at(2.5), [0.25, -0.25]);
}

#[test]
fn quarter_circle_motion_path_leaves_the_diagonal() {
    // kf0 (0,0) out (0, 0.55), kf1 (1,1) in (-0.55, 0) — approximate
    // quarter-circle: mid-arc sits above the straight diagonal.
    let p = Param::Keyframed {
        keyframes: vec![
            Keyframe {
                tick: 0,
                value: [0.0, 0.0],
                easing: Easing::Linear,
                tangents: Some(SpatialTangents {
                    out_t: [0.0, 0.55],
                    in_t: [0.0, 0.0],
                }),
            },
            Keyframe {
                tick: 10,
                value: [1.0, 1.0],
                easing: Easing::Linear,
                tangents: Some(SpatialTangents {
                    out_t: [0.0, 0.0],
                    in_t: [-0.55, 0.0],
                }),
            },
        ],
    };
    let mid = p.sample(5);
    let line_y = mid[0]; // straight diagonal is y = x
    assert!(
        mid[1] > line_y + 0.05,
        "midpoint {mid:?} should sit above the straight diagonal"
    );
    // Rough AE-style quarter bulge — off-diagonal and in the upper half.
    assert!(
        (0.2..=0.6).contains(&mid[0]) && (0.65..=0.95).contains(&mid[1]),
        "unexpected mid-arc position {mid:?}"
    );
    // Monotone progress along the path (x and arc both increase).
    let mut prev = p.sample(0);
    for tick in 1..=10 {
        let cur = p.sample(tick);
        let d_prev = (prev[0] * prev[0] + prev[1] * prev[1]).sqrt();
        let d_cur = (cur[0] * cur[0] + cur[1] * cur[1]).sqrt();
        assert!(
            d_cur + 1e-4 >= d_prev,
            "non-monotone radial progress at {tick}: {prev:?} -> {cur:?}"
        );
        prev = cur;
    }
}

#[test]
fn arc_length_normalization_keeps_equal_t_steps_even() {
    // Asymmetric handles so chord-length ≠ parameter u.
    let p = Param::Keyframed {
        keyframes: vec![
            Keyframe {
                tick: 0,
                value: [0.0, 0.0],
                easing: Easing::Linear,
                tangents: Some(SpatialTangents {
                    out_t: [0.8, 0.0],
                    in_t: [0.0, 0.0],
                }),
            },
            Keyframe {
                tick: 100,
                value: [1.0, 1.0],
                easing: Easing::Linear,
                tangents: Some(SpatialTangents {
                    out_t: [0.0, 0.0],
                    in_t: [0.0, -0.8],
                }),
            },
        ],
    };
    let mut points = Vec::new();
    for i in 0..=10 {
        points.push(p.sample(i * 10));
    }
    let mut lengths = Vec::new();
    for w in points.windows(2) {
        let dx = w[1][0] - w[0][0];
        let dy = w[1][1] - w[0][1];
        lengths.push((dx * dx + dy * dy).sqrt());
    }
    let mean = lengths.iter().sum::<f32>() / lengths.len() as f32;
    for (i, len) in lengths.iter().enumerate() {
        let rel = (len - mean).abs() / mean;
        assert!(
            rel <= 0.10,
            "step {i} arc length {len} diverges from mean {mean} by {rel:.3}"
        );
    }
}

#[test]
fn set_keyframe_tangents_errors_when_missing() {
    let mut p = Param::Constant([0.0f32, 0.0]);
    assert!(
        p.set_keyframe_tangents(
            0,
            Some(SpatialTangents {
                out_t: [0.1, 0.0],
                in_t: [0.0, 0.0],
            })
        )
        .is_err()
    );
    p.set_keyframe(0, [0.0, 0.0], Easing::Linear);
    assert!(
        p.set_keyframe_tangents(
            5,
            Some(SpatialTangents {
                out_t: [0.1, 0.0],
                in_t: [0.0, 0.0],
            })
        )
        .is_err()
    );
    assert!(
        p.set_keyframe_tangents(
            0,
            Some(SpatialTangents {
                out_t: [0.1, 0.2],
                in_t: [0.0, 0.0],
            })
        )
        .is_ok()
    );
    assert_eq!(
        p.keyframes()[0].tangents,
        Some(SpatialTangents {
            out_t: [0.1, 0.2],
            in_t: [0.0, 0.0],
        })
    );
}

// --- piecewise easing presets ----------------------------------------------

#[test]
fn bounce_out_expansion_count_monotone_ticks_ends_at_target() {
    let from = Keyframe::new(0, 0.0, Easing::Linear);
    let to = Keyframe::new(100, 1.0, Easing::Linear);
    let kfs = expand_preset(PiecewiseEasingPreset::BounceOut, &from, &to);
    assert_eq!(kfs.len(), 8);
    assert_eq!(kfs.first().map(|k| (k.tick, k.value)), Some((0, 0.0)));
    assert_eq!(kfs.last().map(|k| (k.tick, k.value)), Some((100, 1.0)));
    for w in kfs.windows(2) {
        assert!(w[1].tick > w[0].tick, "ticks must be strictly increasing");
    }
    // Landings / valleys at expected value fractions.
    assert!((kfs[1].value - 1.0).abs() < 1e-5);
    assert!((kfs[2].value - 0.75).abs() < 1e-5);
    assert!((kfs[3].value - 1.0).abs() < 1e-5);
}

#[test]
fn elastic_out_overshoots_then_converges() {
    let from = Keyframe::new(0, 0.0, Easing::Linear);
    let to = Keyframe::new(100, 10.0, Easing::Linear);
    let kfs = expand_preset(PiecewiseEasingPreset::ElasticOut, &from, &to);
    assert_eq!(kfs.len(), 6);
    let peak = kfs[1].value;
    assert!(peak > 10.0, "first extremum should overshoot: {peak}");
    assert!((kfs.last().unwrap().value - 10.0).abs() < 1e-5);
    // Overshoot magnitude decays toward the target.
    let overs: Vec<f32> = kfs[1..kfs.len() - 1]
        .iter()
        .map(|k| (k.value - 10.0).abs())
        .collect();
    for w in overs.windows(2) {
        assert!(w[1] < w[0] + 1e-5, "overshoot should decay");
    }
}

#[test]
fn back_out_overshoots_ten_percent() {
    let from = Keyframe::new(0, 0.0, Easing::Linear);
    let to = Keyframe::new(100, 1.0, Easing::Linear);
    let kfs = expand_preset(PiecewiseEasingPreset::BackOut, &from, &to);
    assert_eq!(kfs.len(), 3);
    assert!((kfs[1].value - 1.1).abs() < 1e-5);
    assert!((kfs[2].value - 1.0).abs() < 1e-5);
}

#[test]
fn short_segment_preset_is_noop() {
    let from = Keyframe::new(0, 0.0, Easing::EaseIn);
    let to = Keyframe::new(5, 1.0, Easing::Linear);
    let kfs = expand_preset(PiecewiseEasingPreset::BounceOut, &from, &to);
    assert_eq!(kfs.len(), 2);
    assert_eq!(kfs[0].easing, Easing::EaseIn);
    assert_eq!((kfs[0].tick, kfs[1].tick), (0, 5));
}

#[test]
fn color_params_do_not_implement_extrapolate() {
    // Compile-time gate: `[u8; 4]` is Lerp but not Extrapolate, so
    // `expand_preset` cannot be called on color keyframes.
    fn assert_extrapolate<T: Extrapolate>() {}
    assert_extrapolate::<f32>();
    assert_extrapolate::<[f32; 2]>();
    // Uncommenting the next line must fail to compile:
    // assert_extrapolate::<[u8; 4]>();
}

#[test]
fn apply_easing_preset_expands_outgoing_segment() {
    let mut p = Param::Keyframed {
        keyframes: vec![
            Keyframe::new(0, 0.0, Easing::Linear),
            Keyframe::new(100, 1.0, Easing::Linear),
            Keyframe::new(200, 0.0, Easing::Linear),
        ],
    };
    p.apply_easing_preset(0, PiecewiseEasingPreset::BackOut)
        .unwrap();
    let ticks: Vec<i64> = p.keyframes().iter().map(|k| k.tick).collect();
    assert_eq!(ticks[0], 0);
    assert!(ticks.contains(&100));
    assert!(ticks.contains(&200), "later segments untouched");
    // BackOut: from + overshoot + to, plus the trailing keyframe at 200.
    assert_eq!(p.keyframes().len(), 4);
    let mid = p.keyframes().iter().find(|k| k.tick > 0 && k.tick < 100);
    assert!(mid.is_some(), "overshoot keyframe inserted");
    assert!((mid.unwrap().value - 1.1).abs() < 1e-4);
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
