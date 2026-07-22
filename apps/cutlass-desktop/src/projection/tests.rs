use super::*;

fn t(value: i64, num: i32, den: i32) -> EngineTime {
    EngineTime::new(value, EngineRational { num, den })
}

#[test]
fn projects_clip_blend_mode() {
    use cutlass_models::{BlendMode, MediaId, MediaSource, RationalTime, TimeRange, TrackKind};
    use slint::Model;
    use std::collections::HashMap;

    let mut project = EngineProject::new("blend", EngineRational::FPS_24);
    let media = project.add_media(MediaSource::new(
        "/tmp/blend.mp4",
        1920,
        1080,
        EngineRational::FPS_24,
        480,
        true,
    ));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip_id = project
        .add_clip(
            track,
            media,
            TimeRange::at_rate(0, 48, EngineRational::FPS_24),
            RationalTime::new(0, EngineRational::FPS_24),
        )
        .expect("clip");
    project
        .set_blend_mode(clip_id, BlendMode::Multiply)
        .expect("set blend");

    let projected = project_to_slint(&project, &HashMap::new(), &HashSet::new());
    let clips = projected.sequence.tracks.row_data(0).unwrap().clips;
    let clip = clips.row_data(0).unwrap();
    assert_eq!(clip.blend_mode.as_str(), "multiply");
    assert_eq!(clip.blend_label.as_str(), "Multiply");

    // Default is Normal when unset.
    let flat = clip_to_slint(
        &project,
        &cutlass_models::Clip::from_media(
            MediaId::from_raw(media.raw()),
            TimeRange::at_rate(0, 48, EngineRational::FPS_24),
            TimeRange::at_rate(0, 48, EngineRational::FPS_24),
        ),
        EngineKind::Video,
        &HashMap::new(),
    );
    assert_eq!(flat.blend_mode.as_str(), "normal");
    assert_eq!(flat.blend_label.as_str(), "Normal");
}

#[test]
fn projects_clip_layer_styles() {
    use cutlass_models::{
        LayerShadow, LayerStyles, MediaSource, RationalTime, TimeRange, TrackKind,
    };
    use slint::Model;
    use std::collections::HashMap;

    let mut project = EngineProject::new("styles", EngineRational::FPS_24);
    let media = project.add_media(MediaSource::new(
        "/tmp/styles.mp4",
        1920,
        1080,
        EngineRational::FPS_24,
        480,
        true,
    ));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip_id = project
        .add_clip(
            track,
            media,
            TimeRange::at_rate(0, 48, EngineRational::FPS_24),
            RationalTime::new(0, EngineRational::FPS_24),
        )
        .expect("clip");
    project
        .set_layer_styles(
            clip_id,
            LayerStyles {
                shadow: Some(LayerShadow::default()),
                ..Default::default()
            },
        )
        .expect("set styles");

    let projected = project_to_slint(&project, &HashMap::new(), &HashSet::new());
    let clips = projected.sequence.tracks.row_data(0).unwrap().clips;
    let clip = clips.row_data(0).unwrap();
    assert!(clip.style_shadow_enabled);
    assert!(!clip.style_glow_enabled);
    assert!((clip.style_shadow_blur - 8.0).abs() < f32::EPSILON);
    assert!((clip.style_shadow_offset_x - 4.0).abs() < f32::EPSILON);
    assert_eq!(clip.style_shadow_color.red(), 0);
    assert_eq!(clip.style_shadow_color.alpha(), 128);
    assert_eq!(clip.kf_style_shadow_blur.row_count(), 0);

    project
        .set_layer_styles(clip_id, LayerStyles::default())
        .expect("clear styles");
    let projected = project_to_slint(&project, &HashMap::new(), &HashSet::new());
    let clip = projected
        .sequence
        .tracks
        .row_data(0)
        .unwrap()
        .clips
        .row_data(0)
        .unwrap();
    assert!(!clip.style_shadow_enabled);
}

#[test]
fn projects_new_adjust_sliders_and_keyframes() {
    use cutlass_models::{
        ClipParam, ColorAdjustments, Easing, LookParam, MediaSource, ParamValue, RationalTime,
        TimeRange, TrackKind,
    };
    use slint::Model;
    use std::collections::HashMap;

    let mut project = EngineProject::new("adjust", EngineRational::FPS_24);
    let media = project.add_media(MediaSource::new(
        "/tmp/adjust.mp4",
        1920,
        1080,
        EngineRational::FPS_24,
        480,
        true,
    ));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip_id = project
        .add_clip(
            track,
            media,
            TimeRange::at_rate(0, 48, EngineRational::FPS_24),
            RationalTime::new(0, EngineRational::FPS_24),
        )
        .expect("clip");

    project
        .set_clip_adjustments(
            clip_id,
            ColorAdjustments {
                tint: 0.25.into(),
                hue: (-0.5).into(),
                highlights: 0.1.into(),
                shadows: (-0.2).into(),
                sharpness: 0.75.into(),
                vignette: 0.4.into(),
                ..Default::default()
            },
        )
        .expect("adjust");
    project
        .set_param_keyframe(
            clip_id,
            ClipParam::Look {
                param: LookParam::AdjustHue,
            },
            RationalTime::new(0, EngineRational::FPS_24),
            ParamValue::Scalar(-1.0),
            Easing::Linear,
        )
        .expect("kf0");
    project
        .set_param_keyframe(
            clip_id,
            ClipParam::Look {
                param: LookParam::AdjustHue,
            },
            RationalTime::new(24, EngineRational::FPS_24),
            ParamValue::Scalar(1.0),
            Easing::Linear,
        )
        .expect("kf1");

    let projected = project_to_slint(&project, &HashMap::new(), &HashSet::new());
    let clip = projected
        .sequence
        .tracks
        .row_data(0)
        .unwrap()
        .clips
        .row_data(0)
        .unwrap();
    assert!((clip.adjust_tint - 0.25).abs() < f32::EPSILON);
    assert!((clip.adjust_highlights - 0.1).abs() < f32::EPSILON);
    assert!((clip.adjust_shadows - (-0.2)).abs() < f32::EPSILON);
    assert!((clip.adjust_sharpness - 0.75).abs() < f32::EPSILON);
    assert!((clip.adjust_vignette - 0.4).abs() < f32::EPSILON);
    assert_eq!(clip.kf_look_adjust_hue.row_count(), 2);
    assert_eq!(clip.kf_look_adjust_tint.row_count(), 0);
}

#[test]
fn projects_clip_mask() {
    use cutlass_models::{Mask, MaskKind, MediaSource, Param, RationalTime, TimeRange, TrackKind};
    use slint::Model;
    use std::collections::HashMap;

    let mut project = EngineProject::new("mask", EngineRational::FPS_24);
    let media = project.add_media(MediaSource::new(
        "/tmp/mask.mp4",
        1920,
        1080,
        EngineRational::FPS_24,
        480,
        true,
    ));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip_id = project
        .add_clip(
            track,
            media,
            TimeRange::at_rate(0, 48, EngineRational::FPS_24),
            RationalTime::new(0, EngineRational::FPS_24),
        )
        .expect("clip");

    let flat = project_to_slint(&project, &HashMap::new(), &HashSet::new())
        .sequence
        .tracks
        .row_data(0)
        .unwrap()
        .clips
        .row_data(0)
        .unwrap();
    assert_eq!(flat.mask_kind.as_str(), "");

    let mut mask = Mask::new(MaskKind::Heart);
    mask.feather = Param::Constant(0.25);
    mask.invert = true;
    mask.rotation = Param::Constant(30.0);
    project
        .set_clip_mask(clip_id, Some(mask))
        .expect("set mask");

    let projected = project_to_slint(&project, &HashMap::new(), &HashSet::new());
    let clip = projected
        .sequence
        .tracks
        .row_data(0)
        .unwrap()
        .clips
        .row_data(0)
        .unwrap();
    assert_eq!(clip.mask_kind.as_str(), "heart");
    assert_eq!(clip.mask_label.as_str(), "Heart");
    assert!(clip.mask_invert);
    assert!((clip.mask_feather - 0.25).abs() < f32::EPSILON);
    assert!((clip.mask_rotation - 30.0).abs() < f32::EPSILON);
    assert_eq!(clip.kf_look_mask_feather.row_count(), 0);

    project.set_clip_mask(clip_id, None).expect("clear");
    let clip = project_to_slint(&project, &HashMap::new(), &HashSet::new())
        .sequence
        .tracks
        .row_data(0)
        .unwrap()
        .clips
        .row_data(0)
        .unwrap();
    assert_eq!(clip.mask_kind.as_str(), "");
}

#[test]
fn projects_clip_chroma() {
    use cutlass_models::{ChromaKey, MediaSource, Param, RationalTime, TimeRange, TrackKind};
    use slint::Model;
    use std::collections::HashMap;

    let mut project = EngineProject::new("chroma", EngineRational::FPS_24);
    let media = project.add_media(MediaSource::new(
        "/tmp/chroma.mp4",
        1920,
        1080,
        EngineRational::FPS_24,
        480,
        true,
    ));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip_id = project
        .add_clip(
            track,
            media,
            TimeRange::at_rate(0, 48, EngineRational::FPS_24),
            RationalTime::new(0, EngineRational::FPS_24),
        )
        .expect("clip");

    let flat = project_to_slint(&project, &HashMap::new(), &HashSet::new())
        .sequence
        .tracks
        .row_data(0)
        .unwrap()
        .clips
        .row_data(0)
        .unwrap();
    assert!(!flat.chroma_enabled);

    project
        .set_clip_chroma_key(
            clip_id,
            Some(ChromaKey {
                rgb: [0, 255, 0],
                strength: Param::Constant(0.5),
                shadow: Param::Constant(0.1),
            }),
        )
        .expect("set chroma");

    let clip = project_to_slint(&project, &HashMap::new(), &HashSet::new())
        .sequence
        .tracks
        .row_data(0)
        .unwrap()
        .clips
        .row_data(0)
        .unwrap();
    assert!(clip.chroma_enabled);
    assert_eq!(clip.chroma_color.green(), 255);
    assert!((clip.chroma_strength - 0.5).abs() < f32::EPSILON);
    assert!((clip.chroma_shadow - 0.1).abs() < f32::EPSILON);

    project.set_clip_chroma_key(clip_id, None).expect("clear");
    let clip = project_to_slint(&project, &HashMap::new(), &HashSet::new())
        .sequence
        .tracks
        .row_data(0)
        .unwrap()
        .clips
        .row_data(0)
        .unwrap();
    assert!(!clip.chroma_enabled);
}

#[test]
fn sticker_card_gets_catalog_label_and_composited_tag() {
    use cutlass_models::{Clip as MClip, Generator, Rational, TimeRange};
    let span = TimeRange::at_rate(0, 100, Rational::FPS_24);
    let heart = MClip::generated(Generator::sticker("heart"), span);
    assert_eq!(clip_generator_visual(&heart).0, "sticker");
    // Legacy payload-less stickers draw nothing: no composited tag.
    let legacy = MClip::generated(Generator::sticker(""), span);
    assert_eq!(clip_generator_visual(&legacy).0, "");
}

#[test]
fn duration_label_uses_seconds_under_a_minute() {
    assert_eq!(clip_duration_label(t(90, 30, 1)), "3.0s");
    assert_eq!(clip_duration_label(t(101, 30, 1)), "3.4s");
    assert_eq!(clip_duration_label(t(0, 30, 1)), "0.0s");
}

#[test]
fn duration_label_switches_to_timecode_at_a_minute() {
    assert_eq!(clip_duration_label(t(1800, 30, 1)), "1:00");
    // 1h 0m 23s at 30fps.
    assert_eq!(clip_duration_label(t(30 * 3623, 30, 1)), "1:00:23");
}

#[test]
fn duration_label_handles_ntsc_rates() {
    // Exactly 60 logical frames at 29.97: just under 60.06s.
    assert_eq!(clip_duration_label(t(1800, 30000, 1001)), "1:00");
}

#[test]
fn time_to_seconds_is_rate_exact() {
    assert_eq!(time_to_seconds(t(48, 24, 1)), 2.0);
    assert_eq!(time_to_seconds(t(500, 1000, 1)), 0.5);
    assert_eq!(time_to_seconds(t(1, 0, 1)), 0.0, "degenerate rate is safe");
}

#[test]
fn speed_label_formats_retimes() {
    use cutlass_models::{Clip as MClip, MediaId, TimeRange};
    let mut clip = MClip::from_media(
        MediaId::from_raw(1),
        TimeRange::at_rate(0, 48, EngineRational::FPS_24),
        TimeRange::at_rate(0, 48, EngineRational::FPS_24),
    );
    assert_eq!(speed_label(&clip), "", "1× forward has no badge");

    clip.speed = EngineRational::new(2, 1);
    assert_eq!(speed_label(&clip), "2x");
    clip.speed = EngineRational::new(1, 2);
    assert_eq!(speed_label(&clip), "0.5x");
    clip.speed = EngineRational::new(3, 4);
    assert_eq!(speed_label(&clip), "0.75x");

    clip.reversed = true;
    assert_eq!(speed_label(&clip), "0.75x R");
    clip.speed = EngineRational::new(1, 1);
    assert_eq!(speed_label(&clip), "R");
}

#[test]
fn phantom_lanes_are_not_projected() {
    use slint::Model;

    let mut project = EngineProject::new("test", EngineRational::FPS_24);
    project.add_track(cutlass_models::TrackKind::Video, "V1");
    project.add_track(cutlass_models::TrackKind::Effect, "FX1");
    project.add_track(cutlass_models::TrackKind::Filter, "F1");
    project.add_track(cutlass_models::TrackKind::Adjustment, "ADJ1");
    project.add_track(cutlass_models::TrackKind::Sticker, "ST1");

    let projected = project_to_slint(&project, &HashMap::new(), &HashSet::new());
    let tracks = &projected.sequence.tracks;
    // Top-first: sticker, adjustment (M4), effect (standalone effect
    // segments), then the main video lane; only the filter lane stays
    // model-only (M0 "hide phantom kinds", its engine lands in M5).
    assert_eq!(tracks.row_count(), 4);
    assert_eq!(tracks.row_data(0).unwrap().kind, TrackKind::Sticker);
    assert_eq!(tracks.row_data(1).unwrap().kind, TrackKind::Adjustment);
    assert_eq!(tracks.row_data(2).unwrap().kind, TrackKind::Effect);
    assert_eq!(tracks.row_data(3).unwrap().kind, TrackKind::Video);
    assert!(tracks.row_data(3).unwrap().is_main, "main flag projected");
}

#[test]
fn media_pool_flags_missing_entries() {
    use cutlass_models::MediaSource;
    use slint::Model;

    let mut project = EngineProject::new("test", EngineRational::FPS_24);
    let here = project.add_media(MediaSource::new(
        "/tmp/a.mp4",
        1920,
        1080,
        EngineRational::FPS_24,
        48,
        true,
    ));
    let gone = project.add_media(MediaSource::new(
        "/tmp/b.mp4",
        1920,
        1080,
        EngineRational::FPS_24,
        48,
        true,
    ));

    let missing: HashSet<u64> = [gone.raw()].into();
    let projected = project_to_slint(&project, &HashMap::new(), &missing);
    let media = &projected.media;
    assert_eq!(media.row_count(), 2);
    // The pool is sorted by raw id, so rows follow insertion here.
    let first = media.row_data(0).unwrap();
    let second = media.row_data(1).unwrap();
    assert_eq!(first.id.as_str(), here.raw().to_string());
    assert!(!first.is_missing);
    assert!(second.is_missing);
    assert_eq!(
        second.path.as_str(),
        "/tmp/b.mp4",
        "dialog shows where the file used to be"
    );
}

#[test]
fn media_pool_reports_clip_usage_counts() {
    use cutlass_models::{MediaSource, RationalTime, TimeRange, TrackKind};
    use slint::Model;

    let mut project = EngineProject::new("test", EngineRational::FPS_24);
    let used = project.add_media(MediaSource::new(
        "/tmp/used.mp4",
        1920,
        1080,
        EngineRational::FPS_24,
        48,
        true,
    ));
    let unused = project.add_media(MediaSource::new(
        "/tmp/unused.mp4",
        1920,
        1080,
        EngineRational::FPS_24,
        48,
        true,
    ));

    // Two abutting clips reference `used`; `unused` is referenced by none.
    let track = project.add_track(TrackKind::Video, "V1");
    let src = TimeRange::at_rate(0, 24, EngineRational::FPS_24);
    project
        .add_clip(
            track,
            used,
            src,
            RationalTime::new(0, EngineRational::FPS_24),
        )
        .expect("first clip");
    project
        .add_clip(
            track,
            used,
            src,
            RationalTime::new(24, EngineRational::FPS_24),
        )
        .expect("second clip");

    let projected = project_to_slint(&project, &HashMap::new(), &HashSet::new());
    let media = &projected.media;
    assert_eq!(media.row_count(), 2);
    let by_id = |id: &str| {
        (0..media.row_count())
            .map(|r| media.row_data(r).unwrap())
            .find(|m| m.id.as_str() == id)
            .expect("media row")
    };
    assert_eq!(by_id(&used.raw().to_string()).usage_count, 2);
    assert_eq!(by_id(&unused.raw().to_string()).usage_count, 0);
}

#[test]
fn keyframes_publish_absolute_ticks_and_easing() {
    use cutlass_models::{Easing, Keyframe, Param};
    use slint::Model;

    let constant: Param<f32> = Param::Constant(1.0);
    assert_eq!(
        keyframes_to_slint(&constant, 100, |v| (*v, 0.0)).row_count(),
        0
    );

    let param = Param::Keyframed {
        keyframes: vec![
            Keyframe {
                tick: 0,
                value: 0.5f32,
                easing: Easing::EaseOut,
            },
            Keyframe {
                tick: 24,
                value: 1.0,
                easing: Easing::Bezier {
                    points: [0.42, 0.0, 0.58, 1.0],
                },
            },
        ],
    };
    let rows = keyframes_to_slint(&param, 100, |v| (*v, 0.0));
    assert_eq!(rows.row_count(), 2);
    let first = rows.row_data(0).unwrap();
    assert_eq!((first.tick, first.value_x, first.easing), (100, 0.5, 2));
    let second = rows.row_data(1).unwrap();
    assert_eq!((second.tick, second.easing), (124, 4));
    assert_eq!(
        [second.bez_x1, second.bez_y1, second.bez_x2, second.bez_y2],
        [0.42, 0.0, 0.58, 1.0]
    );
}

// --- Phase 4 tick audit: i64 → i32 projection saturates, never wraps. ---

#[test]
fn clamp_i32_saturates_at_the_bounds() {
    assert_eq!(clamp_i32(0), 0);
    assert_eq!(clamp_i32(1_000), 1_000);
    // Above/below i32 range pin to the edge instead of wrapping (a naive
    // `as i32` would alias these to small / negative ticks).
    assert_eq!(clamp_i32(i64::from(i32::MAX) + 1), i32::MAX);
    assert_eq!(clamp_i32(i64::MAX), i32::MAX);
    assert_eq!(clamp_i32(i64::from(i32::MIN) - 1), i32::MIN);
    assert_eq!(clamp_i32(i64::MIN), i32::MIN);
}

#[test]
fn rational_time_saturates_huge_ticks() {
    // A tick parked past the i32 ceiling clamps to the edge of the
    // addressable timeline rather than teleporting to a negative frame.
    let huge = rational_time(t(i64::from(i32::MAX) + 5_000, 30, 1));
    assert_eq!(huge.value, i32::MAX);
    assert_eq!((huge.rate.num, huge.rate.den), (30, 1));
    // In-range ticks pass through untouched.
    assert_eq!(rational_time(t(123, 30, 1)).value, 123);
}

#[test]
fn speed_label_marks_ramps_with_a_tilde() {
    use cutlass_models::{Clip as MClip, MediaId, TimeRange, speed_preset};
    let mut clip = MClip::from_media(
        MediaId::from_raw(1),
        TimeRange::at_rate(0, 48, EngineRational::FPS_24),
        TimeRange::at_rate(0, 48, EngineRational::FPS_24),
    );
    clip.speed_curve = speed_preset("montage").unwrap();
    let label = speed_label(&clip);
    assert!(
        label.starts_with('~'),
        "ramp badge is tilde-prefixed: {label}"
    );
    assert!(
        label.ends_with('x'),
        "ramp badge reports an effective rate: {label}"
    );
}

#[test]
fn speed_curve_projects_dense_samples_and_handles() {
    use cutlass_models::{MediaId, MediaSource, TimeRange, speed_preset};
    use slint::Model;

    let mut project = EngineProject::new("test", EngineRational::FPS_24);
    let media = project.add_media(MediaSource::new(
        "/tmp/a.mp4",
        1920,
        1080,
        EngineRational::FPS_24,
        480,
        true,
    ));
    let _ = media;
    let mut clip = cutlass_models::Clip::from_media(
        MediaId::from_raw(media.raw()),
        TimeRange::at_rate(0, 240, EngineRational::FPS_24),
        TimeRange::at_rate(0, 240, EngineRational::FPS_24),
    );
    // Flat clip: no ramp data projected.
    let flat = clip_to_slint(&project, &clip, EngineKind::Video, &HashMap::new());
    assert!(!flat.has_speed_curve);
    assert_eq!(flat.kf_speed_curve.row_count(), 0);
    assert_eq!(flat.speed_curve_samples.row_count(), 0);

    // Montage ramp: handles mirror the curve's control points (normalized
    // ticks, no clip-start offset), and the dense sample strip fills in.
    clip.speed_curve = speed_preset("montage").unwrap();
    let ramped = clip_to_slint(&project, &clip, EngineKind::Video, &HashMap::new());
    assert!(ramped.has_speed_curve);
    assert_eq!(ramped.kf_speed_curve.row_count(), 3);
    assert_eq!(ramped.kf_speed_curve.row_data(0).unwrap().tick, 0);
    assert_eq!(
        ramped.kf_speed_curve.row_data(2).unwrap().tick,
        cutlass_models::SPEED_CURVE_SCALE as i32
    );
    assert_eq!(ramped.speed_curve_samples.row_count(), SPEED_GRAPH_SAMPLES);
    assert!(ramped.speed_curve_avg > 0.0);
}

#[test]
fn projects_typed_effect_params_and_color_roundtrips() {
    use cutlass_models::{ClipParam, Generator, ParamValue, TimeRange, TrackKind};
    use slint::Model;
    use std::collections::HashMap;

    let mut project = EngineProject::new("effects", EngineRational::FPS_24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    let clip_id = project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [128, 128, 128, 255],
            },
            TimeRange::at_rate(0, 48, EngineRational::FPS_24),
        )
        .expect("clip");
    project.add_effect(clip_id, "duotone").expect("duotone");

    let projected = project_to_slint(&project, &HashMap::new(), &HashSet::new());
    let clip = projected
        .sequence
        .tracks
        .row_data(0)
        .unwrap()
        .clips
        .row_data(0)
        .unwrap();
    assert_eq!(clip.effects.row_count(), 1);
    let fx = clip.effects.row_data(0).unwrap();
    assert_eq!(fx.effect_id.as_str(), "duotone");
    assert_eq!(fx.params.row_count(), 3);

    let shadow = fx.params.row_data(0).unwrap();
    assert_eq!(shadow.name.as_str(), "shadow_color");
    assert_eq!(shadow.kind.as_str(), "color");
    assert_eq!(shadow.color.red(), 20);
    assert_eq!(shadow.color.green(), 16);
    assert_eq!(shadow.color.blue(), 60);
    assert_eq!(shadow.color.alpha(), 255);

    let intensity = fx.params.row_data(2).unwrap();
    assert_eq!(intensity.kind.as_str(), "scalar");
    assert!((intensity.value - 1.0).abs() < f32::EPSILON);

    project
        .set_param_constant(
            clip_id,
            ClipParam::Effect {
                effect: 0,
                param: 0,
            },
            ParamValue::Color([10, 20, 30, 255]),
        )
        .expect("set shadow");

    let projected = project_to_slint(&project, &HashMap::new(), &HashSet::new());
    let shadow = projected
        .sequence
        .tracks
        .row_data(0)
        .unwrap()
        .clips
        .row_data(0)
        .unwrap()
        .effects
        .row_data(0)
        .unwrap()
        .params
        .row_data(0)
        .unwrap();
    assert_eq!(shadow.kind.as_str(), "color");
    assert_eq!(shadow.color.red(), 10);
    assert_eq!(shadow.color.green(), 20);
    assert_eq!(shadow.color.blue(), 30);
    assert_eq!(shadow.color.alpha(), 255);

    // color_overlay also projects its vec2 offset.
    project
        .add_effect(clip_id, "color_overlay")
        .expect("overlay");
    let projected = project_to_slint(&project, &HashMap::new(), &HashSet::new());
    let overlay = projected
        .sequence
        .tracks
        .row_data(0)
        .unwrap()
        .clips
        .row_data(0)
        .unwrap()
        .effects
        .row_data(1)
        .unwrap();
    let offset = overlay.params.row_data(1).unwrap();
    assert_eq!(offset.name.as_str(), "offset");
    assert_eq!(offset.kind.as_str(), "vec2");
    assert!((offset.vec2_x - 0.0).abs() < f32::EPSILON);
    assert!((offset.vec2_y - 0.0).abs() < f32::EPSILON);
}
