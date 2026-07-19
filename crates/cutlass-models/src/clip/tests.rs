use super::*;
use crate::time::Rational;

const R24: Rational = Rational::FPS_24;
const R30: Rational = Rational::FPS_30;

fn rt(value: i64, rate: Rational) -> RationalTime {
    RationalTime::new(value, rate)
}

fn tr(start: i64, duration: i64, rate: Rational) -> TimeRange {
    TimeRange::at_rate(start, duration, rate)
}

fn media_clip(media: MediaId, source: TimeRange, timeline: TimeRange) -> Clip {
    Clip::from_media(media, source, timeline)
}

#[test]
fn extracted_audio_companion_copies_only_audio_and_retime_state() {
    let media = MediaId::from_raw(41);
    let source_range = tr(12, 120, R24);
    let timeline = tr(36, 80, R24);
    let mut source = Clip::from_media(media, source_range, timeline);

    source.link = Some(LinkId::from_raw(9));
    source.speed = Rational::new(3, 2);
    source.reversed = true;
    source.speed_curve = speed_preset("hero").unwrap();
    source.preserve_pitch = false;
    source.volume = Param::Keyframed {
        keyframes: vec![
            Keyframe {
                tick: 0,
                value: 0.25,
                easing: Easing::EaseIn,
            },
            Keyframe {
                tick: 79,
                value: 0.8,
                easing: Easing::Linear,
            },
        ],
    };
    source.fade_in = 7;
    source.fade_out = 11;
    source.denoise = true;
    source.beats = vec![12, 36, 72];

    // Non-audio state is intentionally noisy: none of it may leak to the
    // audio-lane companion.
    source.transform = ClipTransform {
        position: [0.2, -0.3],
        anchor_point: [0.1, 0.9],
        scale: 1.5,
        rotation: 20.0,
        opacity: 0.6,
    }
    .into();
    source.crop = CropRect {
        x: 0.1,
        y: 0.2,
        w: 0.7,
        h: 0.6,
    };
    source.flip_h = true;
    source.filter = Some(crate::look::Filter {
        id: "vivid".into(),
        intensity: 0.4,
    });
    source.animation_in = Some(crate::look::AnimationRef::new("fade_in"));
    source.replaceable = Some(Replaceable::new(3));
    source.text_editable = true;

    let companion = source.extracted_audio_companion().unwrap();
    let mut expected = Clip::from_media(media, source_range, timeline);
    expected.id = companion.id;
    expected.speed = source.speed;
    expected.reversed = source.reversed;
    expected.speed_curve = source.speed_curve.clone();
    expected.preserve_pitch = source.preserve_pitch;
    expected.volume = source.volume.clone();
    expected.fade_in = source.fade_in;
    expected.fade_out = source.fade_out;
    expected.denoise = source.denoise;
    expected.beats = source.beats.clone();
    expected.audio_role = Some(crate::look::AudioRole::Extracted);

    assert_eq!(companion, expected);
    assert_ne!(companion.id, source.id);
    assert_eq!(companion.timeline, source.timeline);
    assert_eq!(companion.content, source.content);
    assert_eq!(companion.link, None);
}

// --- generator serde compat -------------------------------------------

#[test]
fn legacy_bare_sticker_string_deserializes_to_empty_asset() {
    let g: Generator = serde_json::from_value(serde_json::json!("Sticker")).unwrap();
    assert_eq!(
        g,
        Generator::Sticker {
            asset: String::new()
        }
    );
    // Payload form without the field also defaults to empty.
    let g: Generator = serde_json::from_value(serde_json::json!({"Sticker": {}})).unwrap();
    assert_eq!(
        g,
        Generator::Sticker {
            asset: String::new()
        }
    );
}

#[test]
fn generator_roundtrips_through_custom_deserialize() {
    // One of each variant — pins the Deserialize mirror to the real enum.
    let all = [
        Generator::text("hi"),
        Generator::SolidColor { rgba: [1, 2, 3, 4] },
        Generator::shape(Shape::Ellipse, [9, 8, 7, 6]),
        Generator::sticker("heart"),
        Generator::lottie("/tmp/confetti.json", 256, 256),
        Generator::Effect,
        Generator::Filter,
        Generator::Adjustment,
    ];
    for g in all {
        let json = serde_json::to_value(&g).unwrap();
        let back: Generator = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(back, g, "round-trip of {json}");
    }
}

#[test]
fn unknown_generator_variant_errors_with_variant_name() {
    let err = serde_json::from_value::<Generator>(serde_json::json!("Wombat")).unwrap_err();
    assert!(err.to_string().contains("Wombat"), "{err}");
}

#[test]
fn sticker_validate_accepts_catalog_and_empty_ids_only() {
    assert!(Generator::sticker("heart").validate().is_ok());
    assert!(Generator::sticker("").validate().is_ok());
    assert!(Generator::sticker("nope").validate().is_err());
}

// --- constructors -----------------------------------------------------

#[test]
fn from_media_wires_content_and_timeline() {
    let media = MediaId::from_raw(42);
    let source = tr(100, 50, R30);
    let timeline = tr(10, 40, R24);
    let clip = media_clip(media, source, timeline);

    assert_eq!(clip.content, ClipSource::Media { media, source });
    assert_eq!(clip.timeline, timeline);
    assert!(!clip.is_generated());
}

#[test]
fn from_media_assigns_distinct_ids() {
    let media = MediaId::from_raw(1);
    let source = tr(0, 10, R24);
    let timeline = tr(0, 10, R24);
    let a = media_clip(media, source, timeline);
    let b = media_clip(media, source, timeline);
    assert_ne!(a.id, b.id);
}

#[test]
fn generated_text_clip() {
    let timeline = tr(0, 48, R24);
    let clip = Clip::generated(Generator::text("Hello"), timeline);
    assert_eq!(
        clip.content,
        ClipSource::Generated(Generator::text("Hello"))
    );
    assert_eq!(clip.timeline, timeline);
    assert!(clip.is_generated());
}

#[test]
fn generated_all_variants() {
    let timeline = tr(0, 10, R24);

    let solid = Clip::generated(
        Generator::SolidColor {
            rgba: [255, 0, 0, 255],
        },
        timeline,
    );
    assert!(matches!(
        solid.content,
        ClipSource::Generated(Generator::SolidColor { .. })
    ));

    let shape = Clip::generated(
        Generator::shape(Shape::Ellipse, [0, 128, 255, 255]),
        timeline,
    );
    assert!(matches!(
        shape.content,
        ClipSource::Generated(Generator::Shape {
            shape: Shape::Ellipse,
            ..
        })
    ));

    let adj = Clip::generated(Generator::Adjustment, timeline);
    assert!(matches!(
        adj.content,
        ClipSource::Generated(Generator::Adjustment)
    ));
}

#[test]
fn generated_assigns_distinct_ids() {
    let timeline = tr(0, 10, R24);
    let a = Clip::generated(Generator::Adjustment, timeline);
    let b = Clip::generated(Generator::Adjustment, timeline);
    assert_ne!(a.id, b.id);
}

// --- beats (M8 Phase 6) -----------------------------------------------

#[test]
fn beat_timeline_ticks_map_proportionally_within_window() {
    // Source window [0, 100) at 24fps maps onto a [10, 60) timeline span
    // (50 ticks): a beat at source 50 (half-way) lands at 10 + 25 = 35.
    let mut clip = media_clip(MediaId::from_raw(1), tr(0, 100, R24), tr(10, 50, R24));
    clip.beats = vec![0, 50, 99];
    assert_eq!(clip.beat_timeline_ticks(), vec![10, 35, 60]);
}

#[test]
fn beat_timeline_ticks_skip_out_of_window_beats() {
    // Window [40, 80): beats before/after it (left by a trim/split) drop.
    let mut clip = media_clip(MediaId::from_raw(1), tr(40, 40, R24), tr(0, 40, R24));
    clip.beats = vec![10, 40, 60, 80, 100];
    assert_eq!(clip.beat_timeline_ticks(), vec![0, 20]);
}

#[test]
fn beat_timeline_ticks_mirror_when_reversed() {
    // Reversed: a beat near the source window end appears near clip start.
    let mut clip = media_clip(MediaId::from_raw(1), tr(0, 100, R24), tr(0, 100, R24));
    clip.reversed = true;
    clip.beats = vec![25, 75];
    assert_eq!(clip.beat_timeline_ticks(), vec![25, 75]);
    clip.beats = vec![10];
    assert_eq!(clip.beat_timeline_ticks(), vec![90]);
}

#[test]
fn beats_serialize_only_when_present() {
    let clip = media_clip(MediaId::from_raw(1), tr(0, 100, R24), tr(0, 100, R24));
    let json = serde_json::to_value(&clip).unwrap();
    assert!(
        json.get("beats").is_none(),
        "beat-free clips serialize without the field"
    );
    // Pre-beats files (no key) deserialize to an empty list.
    let back: Clip = serde_json::from_value(json).unwrap();
    assert!(!back.has_beats());

    let mut beaten = clip.clone();
    beaten.beats = vec![10, 20, 30];
    let json = serde_json::to_value(&beaten).unwrap();
    assert_eq!(json["beats"], serde_json::json!([10, 20, 30]));
    let round: Clip = serde_json::from_value(json).unwrap();
    assert_eq!(round.beats, vec![10, 20, 30]);
}

// --- accessors --------------------------------------------------------

#[test]
fn media_clip_accessors() {
    let media = MediaId::from_raw(7);
    let source = tr(50, 25, R24);
    let timeline = tr(100, 25, R24);
    let clip = media_clip(media, source, timeline);

    assert_eq!(clip.media(), Some(media));
    assert_eq!(clip.source_range(), Some(source));
    assert_eq!(clip.start(), rt(100, R24));
    assert_eq!(clip.end().unwrap(), rt(125, R24));
}

#[test]
fn generated_clip_accessors_are_none() {
    let clip = Clip::generated(Generator::text("x"), tr(5, 10, R24));
    assert_eq!(clip.media(), None);
    assert_eq!(clip.source_range(), None);
    assert_eq!(clip.start().value, 5);
    assert_eq!(clip.end().unwrap().value, 15);
}

#[test]
fn clip_clone_and_eq() {
    let media = MediaId::from_raw(1);
    let source = tr(0, 10, R24);
    let timeline = tr(0, 10, R24);
    let a = media_clip(media, source, timeline);
    let b = a.clone();
    assert_eq!(a, b);
    assert_eq!(a.id, b.id);
}

// --- source_time_at: same-rate media ----------------------------------

#[test]
fn source_time_at_same_rate_maps_one_to_one() {
    // source [100, 110) placed at timeline [10, 20) — 1:1 at 24fps.
    let clip = media_clip(MediaId::from_raw(1), tr(100, 10, R24), tr(10, 10, R24));

    assert_eq!(
        clip.source_time_at(rt(15, R24)).unwrap(),
        Some(rt(105, R24))
    );
    assert_eq!(
        clip.source_time_at(rt(10, R24)).unwrap(),
        Some(rt(100, R24))
    );
    assert_eq!(
        clip.source_time_at(rt(19, R24)).unwrap(),
        Some(rt(109, R24))
    );
}

#[test]
fn source_time_at_half_open_boundaries() {
    let clip = media_clip(MediaId::from_raw(1), tr(0, 10, R24), tr(10, 10, R24));

    // Exclusive end is not contained.
    assert_eq!(clip.source_time_at(rt(20, R24)).unwrap(), None);
    // Before start.
    assert_eq!(clip.source_time_at(rt(9, R24)).unwrap(), None);
    // After end.
    assert_eq!(clip.source_time_at(rt(21, R24)).unwrap(), None);
}

#[test]
fn source_time_at_generated_always_none() {
    let clip = Clip::generated(Generator::text("title"), tr(0, 100, R24));
    assert_eq!(clip.source_time_at(rt(50, R24)).unwrap(), None);
}

// --- source_time_at: mixed rates ------------------------------------

#[test]
fn source_time_at_resamples_across_rates() {
    // 120 source ticks @ 30fps -> 96 timeline ticks @ 24fps.
    let clip = media_clip(MediaId::from_raw(1), tr(0, 120, R30), tr(0, 96, R24));

    // Timeline midpoint should land near source midpoint after resample.
    let src = clip.source_time_at(rt(48, R24)).unwrap().unwrap();
    assert_eq!(src.rate, R30);
    // 48 @ 24fps = 60 @ 30fps offset from source start 0.
    assert_eq!(src.value, 60);

    // Timeline start maps to source start regardless of rate.
    assert_eq!(clip.source_time_at(rt(0, R24)).unwrap(), Some(rt(0, R30)));
}

#[test]
fn source_time_at_offset_from_nonzero_source_start() {
    // source [200, 300) @ 30fps at timeline [0, 80) @ 24fps.
    let clip = media_clip(MediaId::from_raw(1), tr(200, 100, R30), tr(0, 80, R24));

    let at_start = clip.source_time_at(rt(0, R24)).unwrap().unwrap();
    assert_eq!(at_start, rt(200, R30));

    // 40 timeline ticks @ 24fps -> 50 source ticks @ 30fps from in-point.
    let mid = clip.source_time_at(rt(40, R24)).unwrap().unwrap();
    assert_eq!(mid, rt(250, R30));
}

// --- source_time_at: speed & reverse (M1) -----------------------------

#[test]
fn source_time_at_scales_by_speed() {
    // 2× speed: source [100, 200) occupies timeline [0, 50).
    let mut clip = media_clip(MediaId::from_raw(1), tr(100, 100, R24), tr(0, 50, R24));
    clip.speed = Rational::new(2, 1);
    assert!(clip.is_retimed());
    assert_eq!(clip.source_time_at(rt(0, R24)).unwrap(), Some(rt(100, R24)));
    assert_eq!(
        clip.source_time_at(rt(20, R24)).unwrap(),
        Some(rt(140, R24))
    );
    assert_eq!(
        clip.source_time_at(rt(49, R24)).unwrap(),
        Some(rt(198, R24))
    );
}

#[test]
fn source_time_at_half_speed_holds_frames() {
    // ½ speed: source [0, 50) stretches over timeline [0, 100); each
    // source frame holds for two timeline ticks.
    let mut clip = media_clip(MediaId::from_raw(1), tr(0, 50, R24), tr(0, 100, R24));
    clip.speed = Rational::new(1, 2);
    assert_eq!(clip.source_time_at(rt(0, R24)).unwrap(), Some(rt(0, R24)));
    assert_eq!(clip.source_time_at(rt(50, R24)).unwrap(), Some(rt(25, R24)));
    assert_eq!(clip.source_time_at(rt(51, R24)).unwrap(), Some(rt(25, R24)));
    assert_eq!(clip.source_time_at(rt(99, R24)).unwrap(), Some(rt(49, R24)));
}

#[test]
fn source_time_at_reversed_walks_backward() {
    let mut clip = media_clip(MediaId::from_raw(1), tr(100, 50, R24), tr(0, 50, R24));
    clip.reversed = true;
    assert!(clip.is_retimed());
    assert_eq!(clip.source_time_at(rt(0, R24)).unwrap(), Some(rt(149, R24)));
    assert_eq!(
        clip.source_time_at(rt(25, R24)).unwrap(),
        Some(rt(124, R24))
    );
    assert_eq!(
        clip.source_time_at(rt(49, R24)).unwrap(),
        Some(rt(100, R24))
    );
}

#[test]
fn source_time_at_reversed_double_speed() {
    // 2× + reverse: timeline [0, 25) covers source [100, 150) backward.
    let mut clip = media_clip(MediaId::from_raw(1), tr(100, 50, R24), tr(0, 25, R24));
    clip.speed = Rational::new(2, 1);
    clip.reversed = true;
    assert_eq!(clip.source_time_at(rt(0, R24)).unwrap(), Some(rt(149, R24)));
    assert_eq!(
        clip.source_time_at(rt(10, R24)).unwrap(),
        Some(rt(129, R24))
    );
    assert_eq!(
        clip.source_time_at(rt(24, R24)).unwrap(),
        Some(rt(101, R24))
    );
}

#[test]
fn source_time_at_clamps_rounding_into_the_window() {
    // src dur 3 ÷ (2/3 speed) = 4.5 → 4 timeline ticks (truncating);
    // every timeline tick must still land inside the source window.
    let mut clip = media_clip(MediaId::from_raw(1), tr(10, 3, R24), tr(0, 4, R24));
    clip.speed = Rational::new(2, 3);
    for t in 0..4 {
        let src = clip.source_time_at(rt(t, R24)).unwrap().unwrap().value;
        assert!((10..13).contains(&src), "tick {t} mapped to {src}");
    }
}

// --- speed serde shape --------------------------------------------------

#[test]
fn never_retimed_clips_serialize_without_speed_fields() {
    let clip = media_clip(MediaId::from_raw(1), tr(0, 10, R24), tr(0, 10, R24));
    let value = serde_json::to_value(&clip).expect("serialize");
    let map = value.as_object().expect("clip serializes to a map");
    assert!(!map.contains_key("speed"), "1× speed must stay absent");
    assert!(!map.contains_key("reversed"), "forward must stay absent");

    // And a pre-speed save (no fields) loads as forward 1×.
    let loaded: Clip = serde_json::from_value(value).expect("deserialize");
    assert_eq!(loaded.speed, Rational::new(1, 1));
    assert!(!loaded.reversed);
    assert!(!loaded.is_retimed());
}

#[test]
fn retimed_clip_roundtrips_speed_through_serde() {
    let mut clip = media_clip(MediaId::from_raw(1), tr(0, 10, R24), tr(0, 5, R24));
    clip.speed = Rational::new(2, 1);
    clip.reversed = true;
    let json = serde_json::to_string(&clip).expect("serialize");
    let loaded: Clip = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(loaded.speed, Rational::new(2, 1));
    assert!(loaded.reversed);
}

// --- pitch lock (M8 Phase 3) --------------------------------------------

#[test]
fn pitch_lock_defaults_on_and_is_omitted_from_saves() {
    let clip = media_clip(MediaId::from_raw(1), tr(0, 10, R24), tr(0, 5, R24));
    assert!(clip.preserve_pitch, "pitch is locked by default");
    let map = serde_json::to_value(&clip).unwrap();
    assert!(
        !map.as_object().unwrap().contains_key("preserve_pitch"),
        "the locked default stays absent so old files are byte-identical"
    );
    // A pre-Phase-3 save (no field) loads pitch-locked.
    let loaded: Clip = serde_json::from_value(map).unwrap();
    assert!(loaded.preserve_pitch);
}

#[test]
fn pitch_unlock_roundtrips_and_drives_the_transpose_factor() {
    let mut clip = media_clip(MediaId::from_raw(1), tr(0, 10, R24), tr(0, 5, R24));
    clip.speed = Rational::new(2, 1);
    // Locked: no pitch shift regardless of speed.
    assert_eq!(clip.audio_pitch_factor(), 1.0);
    // Unlocked (chipmunk): pitch rides the 2× speed.
    clip.preserve_pitch = false;
    assert!((clip.audio_pitch_factor() - 2.0).abs() < 1e-6);
    let json = serde_json::to_string(&clip).expect("serialize");
    assert!(json.contains("preserve_pitch"), "the off state is saved");
    let loaded: Clip = serde_json::from_str(&json).expect("deserialize");
    assert!(!loaded.preserve_pitch);
    assert!((loaded.audio_pitch_factor() - 2.0).abs() < 1e-6);
}

// --- speed curves (M2) ---------------------------------------------------

fn linear_ramp(v0: f32, v1: f32) -> Param<f32> {
    Param::Keyframed {
        keyframes: vec![
            Keyframe {
                tick: 0,
                value: v0,
                easing: Easing::Linear,
            },
            Keyframe {
                tick: SPEED_CURVE_SCALE,
                value: v1,
                easing: Easing::Linear,
            },
        ],
    }
}

#[test]
fn flat_curve_is_not_retimed_and_omitted_from_saves() {
    let clip = media_clip(MediaId::from_raw(1), tr(0, 10, R24), tr(0, 10, R24));
    assert!(!clip.has_speed_curve());
    assert_eq!(clip.speed_curve_average(), 1.0);
    let map = serde_json::to_value(&clip).unwrap();
    assert!(!map.as_object().unwrap().contains_key("speed_curve"));
}

#[test]
fn curve_integral_matches_analytic_linear_ramp() {
    let mut clip = media_clip(MediaId::from_raw(1), tr(0, 100, R24), tr(0, 100, R24));
    // Rate ramps 1 → 3 linearly; average = 2, ∫₀ᵖ (1+2q) dq = p + p².
    clip.speed_curve = linear_ramp(1.0, 3.0);
    assert!(clip.has_speed_curve());
    assert!((clip.speed_curve_average() - 2.0).abs() < 1e-6);
    assert!((clip.speed_curve_integral(0.5) - (0.5 + 0.25)).abs() < 1e-6);
    assert!((clip.speed_curve_integral(1.0) - 2.0).abs() < 1e-6);
    // Outside the unit range clamps.
    assert_eq!(clip.speed_curve_integral(0.0), 0.0);
    assert!((clip.speed_curve_integral(2.0) - 2.0).abs() < 1e-6);
}

#[test]
fn source_fraction_is_normalized_and_matches_source_time_at() {
    // Linear 1 → 3 ramp: ∫₀ᵖ (1+2q) dq / 2 = (p + p²)/2.
    let curve = linear_ramp(1.0, 3.0);
    assert_eq!(speed_curve_source_fraction(&curve, 0.0), 0.0);
    assert!(
        (speed_curve_source_fraction(&curve, 1.0) - 1.0).abs() < 1e-9,
        "ends at 1"
    );
    assert!(
        (speed_curve_source_fraction(&curve, 0.5) - (0.5 + 0.25) / 2.0).abs() < 1e-9,
        "midpoint sweeps the analytic fraction"
    );
    // It is exactly the placement `source_time_at` uses for the picture, so
    // the audio render warps in lockstep: fraction · window == source offset.
    let clip = {
        let mut c = media_clip(MediaId::from_raw(1), tr(0, 100, R24), tr(0, 100, R24));
        c.speed_curve = curve.clone();
        c
    };
    let src_dur = clip.source_range().unwrap().duration.value as f64;
    let mid = clip.start().value + clip.timeline.duration.value / 2;
    let src = clip
        .source_time_at(RationalTime::new(mid, R24))
        .unwrap()
        .unwrap();
    let expected = (src_dur * speed_curve_source_fraction(&curve, 0.5)).round() as i64;
    assert_eq!(
        src.value, expected,
        "audio fraction matches the video mapping"
    );
}

#[test]
fn curve_integral_holds_flat_outside_keyframes() {
    let mut clip = media_clip(MediaId::from_raw(1), tr(0, 100, R24), tr(0, 100, R24));
    // One mid keyframe: constant 2.0 everywhere (flat extrapolation).
    clip.speed_curve = Param::Keyframed {
        keyframes: vec![Keyframe {
            tick: SPEED_CURVE_SCALE / 2,
            value: 2.0,
            easing: Easing::Linear,
        }],
    };
    assert!((clip.speed_curve_integral(0.25) - 0.5).abs() < 1e-6);
    assert!((clip.speed_curve_average() - 2.0).abs() < 1e-6);
}

#[test]
fn source_time_at_curve_sweeps_full_window_symmetrically() {
    // A symmetric slow-fast-slow ramp must still consume the whole source
    // window across the clip, and the midpoint sweeps exactly half.
    let mut clip = media_clip(MediaId::from_raw(1), tr(0, 100, R24), tr(0, 100, R24));
    clip.speed_curve = Param::Keyframed {
        keyframes: vec![
            Keyframe {
                tick: 0,
                value: 0.5,
                easing: Easing::Linear,
            },
            Keyframe {
                tick: SPEED_CURVE_SCALE / 2,
                value: 2.0,
                easing: Easing::Linear,
            },
            Keyframe {
                tick: SPEED_CURVE_SCALE,
                value: 0.5,
                easing: Easing::Linear,
            },
        ],
    };
    let start = clip.source_time_at(rt(0, R24)).unwrap().unwrap();
    let mid = clip.source_time_at(rt(50, R24)).unwrap().unwrap();
    let endish = clip.source_time_at(rt(99, R24)).unwrap().unwrap();
    assert_eq!(start.value, 0);
    // By symmetry the middle of the clip is the middle of the source.
    assert_eq!(mid.value, 50);
    // The last frame clamps to the final source frame (window fully swept).
    assert_eq!(endish.value, 99);
}

#[test]
fn source_time_at_flat_curve_matches_constant_speed_exact_path() {
    // A clip with a flat curve and a constant 2× must map identically to
    // the exact rational path (no f64 drift).
    let mut curved = media_clip(MediaId::from_raw(1), tr(0, 100, R24), tr(0, 50, R24));
    curved.speed = Rational::new(2, 1);
    for tick in [0, 7, 23, 49] {
        let got = curved.source_time_at(rt(tick, R24)).unwrap().unwrap();
        assert_eq!(got.value, (tick * 2).min(99), "tick {tick}");
    }
}

#[test]
fn validate_speed_curve_rejects_out_of_range_values_and_ticks() {
    // Value below the floor.
    assert!(validate_speed_curve(&linear_ramp(0.0, 1.0)).is_err());
    // Tick outside the normalized span.
    let bad_tick = Param::Keyframed {
        keyframes: vec![Keyframe {
            tick: SPEED_CURVE_SCALE + 1,
            value: 1.0,
            easing: Easing::Linear,
        }],
    };
    assert!(validate_speed_curve(&bad_tick).is_err());
    // A sane ramp passes.
    assert!(validate_speed_curve(&linear_ramp(0.5, 2.0)).is_ok());
}

#[test]
fn speed_presets_are_valid_curves() {
    for name in ["ramp_up", "ramp_down", "montage", "hero", "bullet"] {
        let curve = speed_preset(name).unwrap_or_else(|| panic!("missing preset {name}"));
        validate_speed_curve(&curve).unwrap_or_else(|e| panic!("{name} invalid: {e:?}"));
    }
    assert!(speed_preset("nope").is_none());
}

#[test]
fn curve_roundtrips_through_serde_and_marks_retimed() {
    let mut clip = media_clip(MediaId::from_raw(1), tr(0, 100, R24), tr(0, 100, R24));
    clip.speed_curve = speed_preset("montage").unwrap();
    assert!(clip.is_retimed());
    let json = serde_json::to_string(&clip).unwrap();
    let loaded: Clip = serde_json::from_str(&json).unwrap();
    assert_eq!(loaded.speed_curve, clip.speed_curve);
    assert!(loaded.has_speed_curve());
}

// --- audio mix: volume + fades (M1) --------------------------------------

#[test]
fn default_audio_serializes_without_fields() {
    let clip = media_clip(MediaId::from_raw(1), tr(0, 10, R24), tr(0, 10, R24));
    assert!(!clip.has_custom_audio());
    let value = serde_json::to_value(&clip).expect("serialize");
    let map = value.as_object().expect("clip serializes to a map");
    assert!(!map.contains_key("volume"), "unit volume must stay absent");
    assert!(!map.contains_key("fade_in"), "zero fade must stay absent");
    assert!(!map.contains_key("fade_out"), "zero fade must stay absent");

    // And a pre-volume save loads with the defaults.
    let loaded: Clip = serde_json::from_value(value).expect("deserialize");
    assert_eq!(loaded.volume, Param::Constant(1.0));
    assert_eq!((loaded.fade_in, loaded.fade_out), (0, 0));
}

#[test]
fn custom_audio_roundtrips_through_serde() {
    let mut clip = media_clip(MediaId::from_raw(1), tr(0, 48, R24), tr(0, 48, R24));
    clip.volume = Param::Constant(0.5);
    clip.fade_in = 12;
    clip.fade_out = 24;
    assert!(clip.has_custom_audio());
    let json = serde_json::to_string(&clip).expect("serialize");
    let loaded: Clip = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(loaded.volume, Param::Constant(0.5));
    assert_eq!((loaded.fade_in, loaded.fade_out), (12, 24));
}

#[test]
fn constant_volume_serializes_as_a_bare_value() {
    // M8 migrated `volume` to a `Param`, but a constant gain must stay
    // byte-identical to the pre-M8 bare-`f32` shape so old files load
    // unchanged and constant-only saves never grow a `{"kf":..}` wrapper.
    let mut clip = media_clip(MediaId::from_raw(1), tr(0, 48, R24), tr(0, 48, R24));
    clip.volume = Param::Constant(0.5);
    let value = serde_json::to_value(&clip).expect("serialize");
    assert_eq!(value.get("volume"), Some(&serde_json::json!(0.5)));
    // A pre-M8 bare value still loads as a constant.
    let loaded: Clip = serde_json::from_value(value).expect("deserialize");
    assert_eq!(loaded.volume, Param::Constant(0.5));
}

#[test]
fn volume_envelope_roundtrips_and_validates() {
    let mut clip = media_clip(MediaId::from_raw(1), tr(0, 48, R24), tr(0, 48, R24));
    clip.volume = Param::Keyframed {
        keyframes: vec![
            Keyframe {
                tick: 0,
                value: 0.0,
                easing: Easing::Linear,
            },
            Keyframe {
                tick: 24,
                value: 1.0,
                easing: Easing::EaseOut,
            },
        ],
    };
    assert!(clip.has_volume_envelope());
    assert!(!clip.is_silent(), "an envelope is non-zero somewhere");
    validate_volume_envelope(&clip.volume).expect("in-range envelope");
    let json = serde_json::to_string(&clip).expect("serialize");
    let loaded: Clip = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(loaded.volume, clip.volume);

    // Out-of-range gain is rejected.
    let hot = Param::Keyframed {
        keyframes: vec![Keyframe {
            tick: 0,
            value: MAX_CLIP_VOLUME + 1.0,
            easing: Easing::Linear,
        }],
    };
    assert!(validate_volume_envelope(&hot).is_err());
}

#[test]
fn audio_gain_ramps_linearly_at_both_edges() {
    let vol = |v: f32| Param::Constant(v);
    // No fades: flat volume everywhere.
    assert_eq!(audio_gain_at(0, 100, &vol(0.8), 0, 0), 0.8);
    assert_eq!(audio_gain_at(99, 100, &vol(0.8), 0, 0), 0.8);

    // Fade-in over the first 10: silence at 0, half at 5, full at 10.
    assert_eq!(audio_gain_at(0, 100, &vol(1.0), 10, 0), 0.0);
    assert_eq!(audio_gain_at(5, 100, &vol(1.0), 10, 0), 0.5);
    assert_eq!(audio_gain_at(10, 100, &vol(1.0), 10, 0), 1.0);

    // Fade-out over the last 10: full until 90, half at 95, ~0 at the end.
    assert_eq!(audio_gain_at(90, 100, &vol(1.0), 0, 10), 1.0);
    assert_eq!(audio_gain_at(95, 100, &vol(1.0), 0, 10), 0.5);
    assert!(audio_gain_at(99, 100, &vol(1.0), 0, 10) <= 0.11);

    // Ramps scale by the volume and overlapping fades multiply.
    assert_eq!(audio_gain_at(5, 100, &vol(2.0), 10, 0), 1.0);
    assert_eq!(audio_gain_at(5, 10, &vol(1.0), 10, 10), 0.25);

    // Out-of-span positions never go negative.
    assert_eq!(audio_gain_at(-3, 100, &vol(1.0), 10, 0), 0.0);
    assert_eq!(audio_gain_at(105, 100, &vol(1.0), 0, 10), 0.0);
}

#[test]
fn audio_gain_follows_a_keyframed_envelope() {
    // A 0→1 ramp envelope over the span: the gain tracks the curve.
    let env = Param::Keyframed {
        keyframes: vec![
            Keyframe {
                tick: 0,
                value: 0.0,
                easing: Easing::Linear,
            },
            Keyframe {
                tick: 100,
                value: 1.0,
                easing: Easing::Linear,
            },
        ],
    };
    assert_eq!(audio_gain_at(0, 100, &env, 0, 0), 0.0);
    assert_eq!(audio_gain_at(50, 100, &env, 0, 0), 0.5);
    assert_eq!(audio_gain_at(100, 100, &env, 0, 0), 1.0);
    // Fades still multiply on top of the sampled envelope value.
    assert_eq!(audio_gain_at(50, 100, &env, 0, 20), 0.5);
    assert_eq!(audio_gain_at(90, 100, &env, 0, 20), 0.9 * 0.5);
}

// --- crop & flip (M1) ----------------------------------------------------

#[test]
fn default_crop_serializes_without_fields() {
    let clip = media_clip(MediaId::from_raw(1), tr(0, 10, R24), tr(0, 10, R24));
    assert!(!clip.has_custom_crop());
    let value = serde_json::to_value(&clip).expect("serialize");
    let map = value.as_object().expect("clip serializes to a map");
    assert!(!map.contains_key("crop"), "full crop must stay absent");
    assert!(!map.contains_key("flip_h"), "no flip must stay absent");
    assert!(!map.contains_key("flip_v"), "no flip must stay absent");

    // And a pre-crop save loads with the defaults.
    let loaded: Clip = serde_json::from_value(value).expect("deserialize");
    assert_eq!(loaded.crop, CropRect::FULL);
    assert!(!loaded.flip_h && !loaded.flip_v);
}

#[test]
fn custom_crop_roundtrips_through_serde() {
    let mut clip = media_clip(MediaId::from_raw(1), tr(0, 10, R24), tr(0, 10, R24));
    clip.crop = CropRect {
        x: 0.1,
        y: 0.2,
        w: 0.5,
        h: 0.25,
    };
    clip.flip_h = true;
    assert!(clip.has_custom_crop());
    let json = serde_json::to_string(&clip).expect("serialize");
    let loaded: Clip = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(loaded.crop, clip.crop);
    assert!(loaded.flip_h && !loaded.flip_v);
}

#[test]
fn crop_rect_validation() {
    assert!(CropRect::FULL.validate().is_ok());
    assert!(
        CropRect {
            x: 0.25,
            y: 0.0,
            w: 0.5,
            h: 1.0,
        }
        .validate()
        .is_ok()
    );

    // Degenerate extents.
    for (w, h) in [(0.0, 1.0), (1.0, 0.0), (0.001, 1.0)] {
        assert!(
            CropRect {
                x: 0.0,
                y: 0.0,
                w,
                h
            }
            .validate()
            .is_err(),
            "w={w} h={h} must be rejected"
        );
    }
    // Out of frame.
    assert!(
        CropRect {
            x: -0.1,
            y: 0.0,
            w: 0.5,
            h: 0.5
        }
        .validate()
        .is_err()
    );
    assert!(
        CropRect {
            x: 0.6,
            y: 0.0,
            w: 0.5,
            h: 0.5
        }
        .validate()
        .is_err()
    );
    assert!(
        CropRect {
            x: 0.0,
            y: 0.9,
            w: 0.5,
            h: 0.2
        }
        .validate()
        .is_err()
    );
    // Non-finite.
    assert!(
        CropRect {
            x: f32::NAN,
            y: 0.0,
            w: 1.0,
            h: 1.0
        }
        .validate()
        .is_err()
    );
}

// --- transform ----------------------------------------------------------

#[test]
fn new_clips_have_identity_transform() {
    let clip = Clip::generated(Generator::Adjustment, tr(0, 10, R24));
    assert!(clip.transform.is_identity());
    assert_eq!(clip.transform, AnimatedTransform::default());
}

#[test]
fn clip_without_transform_field_deserializes_to_identity() {
    // A clip serialized before transforms existed: no `transform` key.
    let clip = Clip::generated(Generator::text("old"), tr(0, 10, R24));
    let mut value = serde_json::to_value(&clip).expect("serialize");
    value
        .as_object_mut()
        .expect("clip serializes to a map")
        .remove("transform")
        .expect("transform field present");

    let loaded: Clip = serde_json::from_value(value).expect("deserialize legacy clip");
    assert!(loaded.transform.is_identity());
    assert_eq!(loaded.content, clip.content);
}

#[test]
fn transform_roundtrips_through_serde() {
    let mut clip = Clip::generated(Generator::Adjustment, tr(0, 10, R24));
    clip.transform = ClipTransform {
        position: [-0.25, 0.5],
        scale: 1.5,
        rotation: 90.0,
        opacity: 0.25,
        ..ClipTransform::IDENTITY
    }
    .into();
    let json = serde_json::to_string(&clip).expect("serialize");
    let loaded: Clip = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(loaded.transform, clip.transform);
}

#[test]
fn legacy_plain_transform_json_deserializes_as_constants() {
    // The exact shape every pre-M2 save wrote: bare values per property.
    let json = r#"{
        "id": 1,
        "content": { "Generated": { "Text": { "content": "t" } } },
        "timeline": { "start": { "value": 0, "rate": { "num": 24, "den": 1 } },
                      "duration": { "value": 24, "rate": { "num": 24, "den": 1 } } },
        "transform": { "position": [0.25, -0.1], "scale": 2.0,
                       "rotation": 45.0, "opacity": 0.5 }
    }"#;
    let clip: Clip = serde_json::from_str(json).expect("deserialize pre-M2 transform");
    assert!(!clip.transform.is_animated());
    assert_eq!(
        clip.transform.sample(0),
        ClipTransform {
            position: [0.25, -0.1],
            scale: 2.0,
            rotation: 45.0,
            opacity: 0.5,
            ..ClipTransform::IDENTITY
        }
    );
}

#[test]
fn constant_transform_serializes_in_pre_m2_shape() {
    let mut clip = Clip::generated(Generator::Adjustment, tr(0, 10, R24));
    clip.transform = ClipTransform {
        position: [0.25, 0.5],
        scale: 1.5,
        rotation: 0.0,
        opacity: 1.0,
        ..ClipTransform::IDENTITY
    }
    .into();
    let value = serde_json::to_value(&clip).expect("serialize");
    // Bare values, not {"kf": ...} wrappers — byte-compatible with old readers.
    assert_eq!(value["transform"]["scale"], 1.5);
    assert_eq!(value["transform"]["position"][0], 0.25);
}

#[test]
fn keyframed_transform_roundtrips() {
    let mut clip = Clip::generated(Generator::Adjustment, tr(0, 48, R24));
    clip.transform
        .set_param_keyframe(
            ClipParam::Opacity,
            0,
            ParamValue::Scalar(0.0),
            Easing::Linear,
        )
        .unwrap();
    clip.transform
        .set_param_keyframe(
            ClipParam::Opacity,
            24,
            ParamValue::Scalar(1.0),
            Easing::EaseOut,
        )
        .unwrap();
    let json = serde_json::to_string(&clip).expect("serialize");
    let loaded: Clip = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(loaded.transform, clip.transform);
    assert!(loaded.transform.is_animated());
    // Segment 0→24 leaves the linear keyframe at tick 0: halfway = 0.5.
    assert_eq!(loaded.transform.sample(12).opacity, 0.5);
}

#[test]
fn animated_transform_samples_per_property() {
    let mut t = AnimatedTransform::identity();
    t.set_param_keyframe(ClipParam::Scale, 0, ParamValue::Scalar(1.0), Easing::Linear)
        .unwrap();
    t.set_param_keyframe(
        ClipParam::Scale,
        10,
        ParamValue::Scalar(2.0),
        Easing::Linear,
    )
    .unwrap();
    // Scale animates; everything else stays constant.
    let mid = t.sample(5);
    assert_eq!(mid.scale, 1.5);
    assert_eq!(mid.position, [0.0, 0.0]);
    assert_eq!(mid.opacity, 1.0);
}

#[test]
fn compose_at_writes_keyframe_only_on_animated_properties() {
    let mut t = AnimatedTransform::identity();
    t.set_param_keyframe(ClipParam::Scale, 0, ParamValue::Scalar(1.0), Easing::Linear)
        .unwrap();
    t.set_param_keyframe(
        ClipParam::Scale,
        20,
        ParamValue::Scalar(3.0),
        Easing::Linear,
    )
    .unwrap();

    let edit = ClipTransform {
        position: [0.3, 0.0],
        scale: 2.0,
        rotation: 0.0,
        opacity: 1.0,
        ..ClipTransform::IDENTITY
    };
    t.compose_at(edit, 10);

    // Scale gained a keyframe at tick 10; the curve still animates.
    assert_eq!(t.scale.keyframes().len(), 3);
    assert_eq!(t.sample(10).scale, 2.0);
    assert_eq!(t.sample(0).scale, 1.0);
    assert_eq!(t.sample(20).scale, 3.0);
    // Position was constant and stays constant.
    assert!(!t.position.is_animated());
    assert_eq!(t.sample(0).position, [0.3, 0.0]);
}

#[test]
fn remove_param_keyframe_errors_when_absent() {
    let mut t = AnimatedTransform::identity();
    assert!(t.remove_param_keyframe(ClipParam::Scale, 5).is_err());
    t.set_param_keyframe(ClipParam::Scale, 5, ParamValue::Scalar(2.0), Easing::Linear)
        .unwrap();
    assert!(t.remove_param_keyframe(ClipParam::Scale, 5).is_ok());
    assert!(!t.scale.is_animated());
    assert_eq!(t.scale.constant(), Some(2.0));
}

#[test]
fn param_kind_mismatch_rejected() {
    let mut t = AnimatedTransform::identity();
    assert!(matches!(
        t.set_param_keyframe(
            ClipParam::Scale,
            0,
            ParamValue::Vec2([1.0, 1.0]),
            Easing::Linear
        ),
        Err(ModelError::InvalidParam(_))
    ));
    assert!(matches!(
        t.set_param_constant(ClipParam::Position, ParamValue::Scalar(1.0)),
        Err(ModelError::InvalidParam(_))
    ));
}

#[test]
fn param_values_validated_per_property() {
    let mut t = AnimatedTransform::identity();
    assert!(
        t.set_param_keyframe(
            ClipParam::Scale,
            0,
            ParamValue::Scalar(-1.0),
            Easing::Linear
        )
        .is_err()
    );
    assert!(
        t.set_param_keyframe(
            ClipParam::Opacity,
            0,
            ParamValue::Scalar(1.5),
            Easing::Linear
        )
        .is_err()
    );
    assert!(
        t.set_param_constant(ClipParam::Position, ParamValue::Vec2([f32::NAN, 0.0]))
            .is_err()
    );
}

#[test]
fn animation_tick_clamps_into_clip() {
    let clip = Clip::generated(Generator::Adjustment, tr(100, 50, R24));
    assert_eq!(clip.animation_tick(100), 0);
    assert_eq!(clip.animation_tick(125), 25);
    assert_eq!(clip.animation_tick(149), 49);
    assert_eq!(clip.animation_tick(90), 0);
    assert_eq!(clip.animation_tick(500), 49);
    // The fractional variant keeps sub-frame offsets and clamps the
    // same way.
    assert!((clip.animation_tick_f(125.4) - 25.4).abs() < 1e-9);
    assert_eq!(clip.animation_tick_f(99.5), 0.0);
    assert_eq!(clip.animation_tick_f(149.6), 49.0);
}

// --- text style ---------------------------------------------------------

#[test]
fn legacy_text_clip_without_style_loads_default() {
    // A title serialized before styling existed: the Text variant only had
    // a `content` field.
    let json = r#"{
        "id": 1,
        "content": { "Generated": { "Text": { "content": "old title" } } },
        "timeline": { "start": { "value": 0, "rate": { "num": 24, "den": 1 } },
                      "duration": { "value": 24, "rate": { "num": 24, "den": 1 } } }
    }"#;
    let clip: Clip = serde_json::from_str(json).expect("deserialize legacy text clip");
    match clip.content {
        ClipSource::Generated(Generator::Text { content, style }) => {
            assert_eq!(content, "old title");
            assert_eq!(style, TextStyle::default());
        }
        other => panic!("expected text generator, got {other:?}"),
    }
}

#[test]
fn legacy_shape_without_dimensions_loads_legacy_defaults() {
    let json = r#"{
        "id": 1,
        "content": { "Generated": { "Shape": {
            "shape": "Rectangle",
            "rgba": [255, 0, 0, 255]
        } } },
        "timeline": { "start": { "value": 0, "rate": { "num": 24, "den": 1 } },
                      "duration": { "value": 24, "rate": { "num": 24, "den": 1 } } }
    }"#;
    let clip: Clip = serde_json::from_str(json).expect("deserialize legacy shape clip");
    match clip.content {
        ClipSource::Generated(Generator::Shape {
            rgba,
            width,
            height,
            corner_radius,
            stroke,
            ..
        }) => {
            // Bare legacy values load as constants (the M2 Param trick).
            assert_eq!(rgba.constant(), Some([255, 0, 0, 255]));
            assert_eq!(width.constant(), Some(960.0));
            assert_eq!(height.constant(), Some(540.0));
            // Fields that postdate the file default to "not set".
            assert_eq!(corner_radius.constant(), Some(0.0));
            assert!(stroke.is_none());
        }
        other => panic!("expected shape generator, got {other:?}"),
    }
}

#[test]
fn never_touched_shape_serializes_byte_identical_to_legacy() {
    // A constant shape must not leak the new fields into saves: `Param`
    // constants serialize bare, corner_radius/stroke are elided.
    let generator = Generator::shape(Shape::Rectangle, [10, 20, 30, 255]);
    let json = serde_json::to_value(&generator).unwrap();
    let obj = &json["Shape"];
    assert_eq!(obj["rgba"], serde_json::json!([10, 20, 30, 255]));
    assert_eq!(obj["width"], serde_json::json!(200.0));
    assert!(
        obj.get("corner_radius").is_none(),
        "zero corner radius must be elided: {obj}"
    );
    assert!(obj.get("stroke").is_none(), "absent stroke must be elided");
}

#[test]
fn new_shape_kinds_roundtrip_with_keyframed_params() {
    let mut inner = Param::Constant(0.5);
    inner.set_keyframe(0, 0.2, Easing::Linear);
    inner.set_keyframe(24, 0.9, Easing::EaseInOut);
    let generator = Generator::Shape {
        shape: Shape::Star {
            points: 5,
            inner_ratio: inner,
        },
        rgba: Param::Constant([255, 0, 0, 255]),
        width: Param::Constant(300.0),
        height: Param::Constant(300.0),
        corner_radius: Param::Constant(4.0),
        stroke: Some(ShapeStroke::new([0, 0, 0, 255], 8.0)),
    };
    generator.validate().expect("valid star");
    let json = serde_json::to_string(&generator).unwrap();
    let back: Generator = serde_json::from_str(&json).unwrap();
    assert_eq!(back, generator);

    let path = Generator::Shape {
        shape: Shape::Path(ShapePath {
            points: vec![
                ShapePathPoint::corner([-50.0, 0.0]),
                ShapePathPoint {
                    anchor: [50.0, 0.0],
                    handle_in: [0.0, -60.0],
                    handle_out: [50.0, 0.0],
                },
            ],
            closed: false,
        }),
        rgba: Param::Constant([255, 255, 255, 255]),
        width: Param::Constant(100.0),
        height: Param::Constant(100.0),
        corner_radius: Param::Constant(0.0),
        stroke: Some(ShapeStroke::new([255, 255, 255, 255], 4.0)),
    };
    path.validate().expect("valid path");
    let json = serde_json::to_string(&path).unwrap();
    let back: Generator = serde_json::from_str(&json).unwrap();
    assert_eq!(back, path);
}

#[test]
fn generator_validate_rejects_bad_shapes() {
    let bad = |shape: Shape| Generator::shape(shape, [255, 255, 255, 255]).validate();
    assert!(bad(Shape::Polygon { sides: 2 }).is_err());
    assert!(
        bad(Shape::Star {
            points: 99,
            inner_ratio: Param::Constant(0.5)
        })
        .is_err()
    );
    assert!(
        bad(Shape::Star {
            points: 5,
            inner_ratio: Param::Constant(1.5)
        })
        .is_err(),
        "inner_ratio above 1 must be rejected"
    );
    assert!(
        bad(Shape::Path(ShapePath {
            points: vec![ShapePathPoint::corner([0.0, 0.0])],
            closed: true
        }))
        .is_err(),
        "single-point path is not drawable"
    );
    assert!(
        bad(Shape::Path(ShapePath {
            points: vec![
                ShapePathPoint::corner([0.0, f32::NAN]),
                ShapePathPoint::corner([10.0, 0.0]),
            ],
            closed: true,
        }))
        .is_err(),
        "non-finite path coordinate must be rejected"
    );

    let mut wide = Generator::shape(Shape::Rectangle, [255, 255, 255, 255]);
    if let Generator::Shape { width, .. } = &mut wide {
        *width = Param::Constant(MAX_SHAPE_DIM * 2.0);
    }
    assert!(wide.validate().is_err(), "oversized width must be rejected");

    let mut hot_stroke = Generator::shape(Shape::Ellipse, [255, 255, 255, 255]);
    if let Generator::Shape { stroke, .. } = &mut hot_stroke {
        *stroke = Some(ShapeStroke::new([0, 0, 0, 255], MAX_STROKE_WIDTH + 1.0));
    }
    assert!(hot_stroke.validate().is_err());
}

#[test]
fn shape_param_routing_sets_and_samples() {
    let mut g = Generator::shape(Shape::Rectangle, [255, 255, 255, 255]);
    g.set_shape_param_keyframe(
        ShapeParam::Width,
        0,
        ParamValue::Scalar(100.0),
        Easing::Linear,
    )
    .unwrap();
    g.set_shape_param_keyframe(
        ShapeParam::Width,
        10,
        ParamValue::Scalar(300.0),
        Easing::Linear,
    )
    .unwrap();
    g.set_shape_param_keyframe(
        ShapeParam::Fill,
        0,
        ParamValue::Color([0, 0, 0, 255]),
        Easing::Linear,
    )
    .unwrap();
    g.set_shape_param_keyframe(
        ShapeParam::Fill,
        10,
        ParamValue::Color([255, 0, 0, 255]),
        Easing::Linear,
    )
    .unwrap();
    let Generator::Shape { width, rgba, .. } = &g else {
        panic!()
    };
    assert_eq!(width.sample(5), 200.0);
    assert_eq!(rgba.sample(5), [128, 0, 0, 255], "colors lerp per channel");

    // Removing down to zero keyframes collapses to a constant.
    g.remove_shape_param_keyframe(ShapeParam::Width, 0).unwrap();
    g.remove_shape_param_keyframe(ShapeParam::Width, 10)
        .unwrap();
    let Generator::Shape { width, .. } = &g else {
        panic!()
    };
    assert!(!width.is_animated());
}

#[test]
fn shape_param_routing_rejects_wrong_targets() {
    let mut rect = Generator::shape(Shape::Rectangle, [255, 255, 255, 255]);
    // Star-only param on a rectangle.
    assert!(
        rect.set_shape_param_constant(ShapeParam::InnerRatio, ParamValue::Scalar(0.5))
            .is_err()
    );
    // Stroke params with no stroke set.
    assert!(
        rect.set_shape_param_constant(ShapeParam::StrokeWidth, ParamValue::Scalar(4.0))
            .is_err()
    );
    // Kind mismatches both ways.
    assert!(
        rect.set_shape_param_constant(ShapeParam::Width, ParamValue::Color([0, 0, 0, 255]))
            .is_err()
    );
    assert!(
        rect.set_shape_param_constant(ShapeParam::Fill, ParamValue::Scalar(1.0))
            .is_err()
    );
    // Out-of-range values are rejected at the routing boundary.
    assert!(
        rect.set_shape_param_constant(ShapeParam::Width, ParamValue::Scalar(-5.0))
            .is_err()
    );
    // Non-shape generators reject everything.
    let mut text = Generator::text("hi");
    assert!(
        text.set_shape_param_constant(ShapeParam::Width, ParamValue::Scalar(10.0))
            .is_err()
    );
}

#[test]
fn text_style_roundtrips_through_serde() {
    let style = TextStyle {
        font: "Helvetica".into(),
        size: 120.0,
        bold: true,
        italic: true,
        underline: true,
        case: TextCase::Upper,
        fill: [10, 20, 30, 255],
        letter_spacing: 3.0,
        line_spacing: 1.5,
        align_h: TextAlignH::Right,
        align_v: TextAlignV::Bottom,
        wrap: false,
        stroke: Some(TextStroke {
            rgba: [0, 0, 0, 255],
            width: 8.0,
        }),
        background: Some(TextBackground {
            rgba: [255, 255, 0, 200],
            radius: 0.5,
        }),
        shadow: Some(TextShadow {
            rgba: [0, 0, 0, 230],
            blur: 0.25,
            distance: 12.0,
        }),
        effect_preset: None,
    };
    let clip = Clip::generated(
        Generator::Text {
            content: "Styled".into(),
            style: style.clone(),
        },
        tr(0, 24, R24),
    );
    let json = serde_json::to_string(&clip).expect("serialize");
    let loaded: Clip = serde_json::from_str(&json).expect("deserialize");
    match loaded.content {
        ClipSource::Generated(Generator::Text {
            content,
            style: got,
        }) => {
            assert_eq!(content, "Styled");
            assert_eq!(got, style);
        }
        other => panic!("expected text generator, got {other:?}"),
    }
}

#[test]
fn text_generator_validation_rejects_unsafe_metrics() {
    let invalid = |style| {
        Generator::Text {
            content: "Unsafe".into(),
            style,
        }
        .validate()
        .is_err()
    };

    assert!(invalid(TextStyle {
        size: f32::NAN,
        ..Default::default()
    }));
    assert!(invalid(TextStyle {
        line_spacing: 0.0,
        ..Default::default()
    }));
    assert!(invalid(TextStyle {
        stroke: Some(TextStroke {
            rgba: [0, 0, 0, 255],
            width: 513.0,
        }),
        ..Default::default()
    }));
    assert!(invalid(TextStyle {
        background: Some(TextBackground {
            rgba: [0, 0, 0, 255],
            radius: 1.01,
        }),
        ..Default::default()
    }));
    assert!(invalid(TextStyle {
        shadow: Some(TextShadow {
            rgba: [0, 0, 0, 255],
            blur: 0.2,
            distance: -1.0,
        }),
        ..Default::default()
    }));

    Generator::Text {
        content: "Safe".into(),
        style: TextStyle {
            stroke: Some(TextStroke {
                rgba: [0, 0, 0, 255],
                width: 40.0,
            }),
            background: Some(TextBackground {
                rgba: [0, 0, 0, 200],
                radius: 1.0,
            }),
            shadow: Some(TextShadow {
                rgba: [0, 0, 0, 230],
                blur: 1.0,
                distance: 50.0,
            }),
            ..Default::default()
        },
    }
    .validate()
    .expect("inspector effect limits are valid");
}

#[test]
fn text_case_apply() {
    assert_eq!(TextCase::Normal.apply("Hello World"), "Hello World");
    assert_eq!(TextCase::Upper.apply("Hello World"), "HELLO WORLD");
    assert_eq!(TextCase::Lower.apply("Hello World"), "hello world");
    assert_eq!(TextCase::Title.apply("hello world"), "Hello World");
    assert_eq!(TextCase::Title.apply("hELLO  wORLD"), "Hello  World");
}

#[test]
fn transform_validation() {
    assert!(ClipTransform::IDENTITY.validate().is_ok());
    assert!(
        ClipTransform {
            position: [0.4, -0.4],
            scale: 3.0,
            rotation: -720.0,
            opacity: 0.0,
            ..ClipTransform::IDENTITY
        }
        .validate()
        .is_ok()
    );

    let bad_scale = ClipTransform {
        scale: -0.5,
        ..ClipTransform::IDENTITY
    };
    assert!(matches!(
        bad_scale.validate(),
        Err(ModelError::InvalidTransform(_))
    ));

    let bad_opacity = ClipTransform {
        opacity: -0.1,
        ..ClipTransform::IDENTITY
    };
    assert!(matches!(
        bad_opacity.validate(),
        Err(ModelError::InvalidTransform(_))
    ));

    let bad_position = ClipTransform {
        position: [0.0, f32::NAN],
        ..ClipTransform::IDENTITY
    };
    assert!(matches!(
        bad_position.validate(),
        Err(ModelError::InvalidTransform(_))
    ));
}

// --- source_time_at: errors -------------------------------------------

#[test]
fn source_time_at_rate_mismatch_errors() {
    let clip = media_clip(MediaId::from_raw(1), tr(0, 10, R24), tr(0, 10, R24));
    let err = clip.source_time_at(rt(5, R30)).unwrap_err();
    assert_eq!(
        err,
        ModelError::RateMismatch {
            expected: R30,
            got: R24,
        }
    );
}
