use cutlass_models::{
    AnimationRef, AudioRole, ChromaKey, Clip, ClipCapabilities, ColorAdjustments, CropRect, Easing,
    EffectInstance, Filter, Keyframe, LinkId, Lut, Mask, MaskKind, MediaId, MediaSource, Param,
    Project, Rational, RationalTime, Replaceable, StabilizeLevel, TimeRange, TrackKind,
};

const FPS: Rational = Rational::FPS_24;

fn rt(value: i64) -> RationalTime {
    RationalTime::new(value, FPS)
}

fn tr(start: i64, duration: i64) -> TimeRange {
    TimeRange::at_rate(start, duration, FPS)
}

fn video_project() -> (Project, MediaId) {
    let mut project = Project::new("freeze", FPS);
    let media = project.add_media(MediaSource::new(
        "/tmp/freeze-model.mp4",
        1920,
        1080,
        FPS,
        1_000,
        true,
    ));
    (project, media)
}

#[test]
fn freeze_marker_is_additive_and_omitted_while_false() {
    let ordinary = Clip::from_media(MediaId::from_raw(7), tr(10, 20), tr(30, 20));
    let ordinary_json = serde_json::to_string(&ordinary).unwrap();
    assert!(
        !ordinary_json.contains("freeze_frame"),
        "ordinary clip bytes must retain their pre-freeze shape"
    );

    let loaded: Clip = serde_json::from_str(&ordinary_json).unwrap();
    assert!(!loaded.freeze_frame);
    assert_eq!(serde_json::to_string(&loaded).unwrap(), ordinary_json);

    let mut frozen = ordinary;
    frozen.freeze_frame = true;
    let value = serde_json::to_value(&frozen).unwrap();
    assert_eq!(value["freeze_frame"], true);
    assert!(serde_json::from_value::<Clip>(value).unwrap().freeze_frame);
}

#[test]
fn freeze_helper_bakes_visual_state_and_clears_dynamic_semantics() {
    let mut source = Clip::from_media(MediaId::from_raw(9), tr(100, 120), tr(10, 80));
    source.link = Some(LinkId::next());
    source
        .transform
        .position
        .set_keyframe(0, [0.0, 0.0], Easing::Linear);
    source
        .transform
        .position
        .set_keyframe(40, [1.0, -0.5], Easing::Linear);
    source
        .transform
        .rotation
        .set_keyframe(0, 10.0, Easing::Linear);
    source
        .transform
        .rotation
        .set_keyframe(40, 50.0, Easing::Linear);
    source.speed = Rational::new(3, 2);
    source.reversed = true;
    source.speed_curve = Param::Keyframed {
        keyframes: vec![
            Keyframe {
                tick: 0,
                value: 0.5,
                easing: Easing::Linear,
            },
            Keyframe {
                tick: 1_000,
                value: 1.5,
                easing: Easing::Linear,
            },
        ],
    };
    source.preserve_pitch = false;
    source.volume = Param::Constant(0.25);
    source.fade_in = 4;
    source.fade_out = 6;
    source.denoise = true;
    source.beats = vec![105, 140, 180];
    source.audio_role = Some(AudioRole::Music);
    source.crop = CropRect {
        x: 0.1,
        y: 0.2,
        w: 0.7,
        h: 0.6,
    };
    source.flip_h = true;
    source.flip_v = true;
    source.mask = Some(Mask::new(MaskKind::Circle));
    source.chroma_key = Some(ChromaKey {
        rgb: [0, 255, 0],
        strength: 0.7.into(),
        shadow: 0.2.into(),
    });
    source.stabilize = Some(StabilizeLevel::Smooth);
    source.filter = Some(Filter::new("vivid"));
    source.lut = Some(Lut::new("/tmp/look.cube"));
    source.adjust = ColorAdjustments {
        brightness: 0.2.into(),
        contrast: (-0.1).into(),
        saturation: 0.3.into(),
        exposure: 0.1.into(),
        temperature: (-0.2).into(),
    };
    source.animation_in = Some(AnimationRef::new("fade_in"));
    source.animation_out = Some(AnimationRef::new("fade_out"));
    source.animation_combo = Some(AnimationRef::new("pulse"));
    source.replaceable = Some(Replaceable::new(3).with_label("Original slot"));
    source.text_editable = true;

    let mut effect = EffectInstance::new("gaussian_blur");
    effect
        .set_param_keyframe(0, 0, 2.0, Easing::Linear)
        .unwrap();
    effect
        .set_param_keyframe(0, 40, 18.0, Easing::Linear)
        .unwrap();
    source.effects.push(effect);

    // The placement begins at source timeline tick 30, i.e. clip-relative 20.
    let expected_transform = source.transform.sample(20);
    let expected_radius = source.effects[0].params["radius"].sample(20);
    let source_id = source.id;
    let frozen = source.frozen_frame(rt(140), tr(30, 50)).unwrap();

    assert_ne!(frozen.id, source_id);
    assert_eq!(frozen.media(), source.media());
    assert_eq!(frozen.source_range(), Some(tr(140, 1)));
    assert_eq!(frozen.timeline, tr(30, 50));
    assert!(frozen.freeze_frame);
    assert_eq!(frozen.link, None);

    assert_eq!(frozen.transform.sample(0), expected_transform);
    assert!(!frozen.transform.is_animated());
    assert_eq!(
        frozen.effects[0].params["radius"],
        Param::Constant(expected_radius)
    );
    assert_eq!(frozen.effects[0].effect_id, source.effects[0].effect_id);

    assert_eq!(frozen.speed, Rational::new(1, 1));
    assert!(!frozen.reversed);
    assert_eq!(frozen.speed_curve, Param::Constant(1.0));
    assert!(frozen.preserve_pitch);
    assert_eq!(frozen.volume, Param::Constant(1.0));
    assert_eq!((frozen.fade_in, frozen.fade_out), (0, 0));
    assert!(!frozen.denoise);
    assert!(frozen.beats.is_empty());
    assert_eq!(frozen.audio_role, None);
    assert!(frozen.is_silent());

    assert_eq!(frozen.crop, source.crop);
    assert_eq!((frozen.flip_h, frozen.flip_v), (true, true));
    assert_eq!(frozen.mask, source.mask);
    assert_eq!(frozen.chroma_key, source.chroma_key);
    assert_eq!(frozen.stabilize, source.stabilize);
    assert_eq!(frozen.filter, source.filter);
    assert_eq!(frozen.lut, source.lut);
    assert_eq!(frozen.adjust, source.adjust);
    assert_eq!(frozen.animation_in, None);
    assert_eq!(frozen.animation_out, None);
    assert_eq!(frozen.animation_combo, None);
    assert_eq!(frozen.replaceable, None);
    assert!(!frozen.text_editable);

    for tick in [30, 31, 55, 79] {
        assert_eq!(frozen.source_time_at(rt(tick)).unwrap(), Some(rt(140)));
    }
    assert_eq!(frozen.source_time_at(rt(80)).unwrap(), None);
}

#[test]
fn frozen_clip_is_silent_and_has_no_retime_or_extract_capabilities() {
    let (mut project, media) = video_project();
    let track = project.add_track(TrackKind::Video, "V1");
    let source = Clip::from_media(media, tr(20, 80), tr(0, 80));
    let frozen = source.frozen_frame(rt(40), tr(10, 30)).unwrap();
    let id = project
        .timeline_mut()
        .add_clip(track, frozen)
        .expect("place frozen clip");
    let clip = project.clip(id).unwrap();

    assert!(!project.timeline().carries_own_audio(id));
    let capabilities = ClipCapabilities::for_clip(&project, clip, TrackKind::Video);
    assert!(!capabilities.has_audio);
    assert!(!capabilities.has_speed);
    assert!(!capabilities.can_reverse);
    assert!(!capabilities.can_extract_audio);
    assert!(capabilities.can_split);
}

#[test]
fn trim_extend_and_split_keep_the_held_source_frame() {
    let (mut project, media) = video_project();
    let track = project.add_track(TrackKind::Video, "V1");
    let source = Clip::from_media(media, tr(100, 100), tr(0, 100));
    let frozen = source.frozen_frame(rt(142), tr(20, 20)).unwrap();
    let id = project.timeline_mut().add_clip(track, frozen).unwrap();
    let held = Some(tr(142, 1));

    project.trim_clip(id, tr(25, 10)).unwrap();
    assert_eq!(project.clip(id).unwrap().source_range(), held);
    project.trim_clip(id, tr(10, 40)).unwrap();
    assert_eq!(project.clip(id).unwrap().source_range(), held);

    let tail = project.split_clip(id, rt(25)).unwrap();
    let left = project.clip(id).unwrap();
    let right = project.clip(tail).unwrap();
    assert_eq!(left.timeline, tr(10, 15));
    assert_eq!(right.timeline, tr(25, 25));
    assert!(left.freeze_frame && right.freeze_frame);
    assert_eq!(left.source_range(), held);
    assert_eq!(right.source_range(), held);

    let left_before = left.clone();
    assert!(
        project
            .set_clip_speed(id, Rational::new(2, 1), false)
            .is_err()
    );
    assert_eq!(project.clip(id).unwrap(), &left_before);
    assert!(
        project
            .set_clip_speed_curve(id, Some(Param::Constant(2.0)))
            .is_err()
    );
    assert_eq!(project.clip(id).unwrap(), &left_before);
}
