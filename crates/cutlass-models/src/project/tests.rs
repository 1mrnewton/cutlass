use super::*;
use std::path::Path;

use crate::clip::Shape;

const R24: Rational = Rational::FPS_24;
const R30: Rational = Rational::FPS_30;

fn rt(value: i64) -> RationalTime {
    RationalTime::new(value, R24)
}

fn tr(start: i64, duration: i64) -> TimeRange {
    TimeRange::at_rate(start, duration, R24)
}

fn tr_at(start: i64, duration: i64, rate: Rational) -> TimeRange {
    TimeRange::at_rate(start, duration, rate)
}

fn sample_media(fps: Rational, duration: i64) -> MediaSource {
    MediaSource::new("/tmp/sample.mp4", 1920, 1080, fps, duration, true)
}

fn project_with_media(duration: i64) -> (Project, MediaId, TrackId) {
    let mut project = Project::new("test", R24);
    let media_id = project.add_media(sample_media(R24, duration));
    let track = project.add_track(TrackKind::Video, "V1");
    (project, media_id, track)
}

// --- shape params -------------------------------------------------------

#[test]
fn shape_params_keyframe_through_project_api() {
    let mut project = Project::new("p", R24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    let clip = project
        .add_generated(
            track,
            Generator::shape(Shape::Ellipse, [255, 255, 255, 255]),
            tr(100, 48),
        )
        .unwrap();

    // Keyframe ticks are clip-relative: timeline 100/140 → ticks 0/40.
    let width = ClipParam::Shape {
        param: crate::ShapeParam::Width,
    };
    project
        .set_param_keyframe(
            clip,
            width,
            rt(100),
            ParamValue::Scalar(100.0),
            Easing::Linear,
        )
        .unwrap();
    project
        .set_param_keyframe(
            clip,
            width,
            rt(140),
            ParamValue::Scalar(500.0),
            Easing::Linear,
        )
        .unwrap();
    let ClipSource::Generated(Generator::Shape { width: w, .. }) =
        &project.clip(clip).unwrap().content
    else {
        panic!("expected shape");
    };
    assert_eq!(w.sample(20), 300.0);

    // Constants flatten; out-of-range and wrong-kind values are rejected
    // with the clip unchanged.
    project
        .set_param_constant(clip, width, ParamValue::Scalar(250.0))
        .unwrap();
    assert!(
        project
            .set_param_keyframe(
                clip,
                width,
                rt(100),
                ParamValue::Color([0; 4]),
                Easing::Linear
            )
            .is_err()
    );
    assert!(
        project
            .set_param_constant(clip, width, ParamValue::Scalar(f32::NAN))
            .is_err()
    );

    // Media clips reject shape params outright.
    let media = project.add_media(sample_media(R24, 500));
    let vtrack = project.add_track(TrackKind::Video, "V1");
    let mclip = project.add_clip(vtrack, media, tr(0, 100), rt(0)).unwrap();
    assert!(
        project
            .set_param_constant(mclip, width, ParamValue::Scalar(10.0))
            .is_err()
    );
}

#[test]
fn add_generated_rejects_invalid_shape() {
    let mut project = Project::new("p", R24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    let bad = Generator::shape(Shape::Polygon { sides: 2 }, [255, 255, 255, 255]);
    assert!(matches!(
        project.add_generated(track, bad, tr(0, 48)),
        Err(ModelError::InvalidParam(_))
    ));
}

// --- transitions (M4) -------------------------------------------------

/// Two abutting solid clips on one sticker track (a lane kind that
/// supports transitions); returns `(project, left, right, track)`.
fn project_with_abutting_pair() -> (Project, ClipId, ClipId, TrackId) {
    let mut project = Project::new("test", R24);
    let track = project.add_track(TrackKind::Sticker, "S1");
    let left = project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [255, 0, 0, 255],
            },
            tr(0, 24),
        )
        .unwrap();
    let right = project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [0, 0, 255, 255],
            },
            tr(24, 24),
        )
        .unwrap();
    (project, left, right, track)
}

#[test]
fn add_transition_links_abutting_pair() {
    let (mut project, left, right, track) = project_with_abutting_pair();
    project.add_transition(left, "crossfade").unwrap();
    let t = project
        .timeline()
        .track(track)
        .unwrap()
        .transition_at(left)
        .unwrap();
    assert_eq!(t.right, right);
    assert_eq!(t.transition_id, "crossfade");
    assert_eq!(t.duration, crate::transition::DEFAULT_TRANSITION_TICKS);
}

#[test]
fn add_transition_rejects_unknown_id_and_non_abutting() {
    let (mut project, left, _right, _track) = project_with_abutting_pair();
    assert!(matches!(
        project.add_transition(left, "warp_speed"),
        Err(ModelError::InvalidParam(_))
    ));
    // A lone clip with no right neighbor cannot take a transition.
    let track = project.add_track(TrackKind::Sticker, "S2");
    let lone = project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [0, 255, 0, 255],
            },
            tr(0, 24),
        )
        .unwrap();
    assert!(matches!(
        project.add_transition(lone, "crossfade"),
        Err(ModelError::InvalidParam(_))
    ));
}

#[test]
fn add_transition_rejects_canvas_pass_lanes() {
    // Effect/filter/adjustment segments resolve to canvas-wide passes,
    // which the renderer can't nest inside a transition — the model
    // refuses the junction outright.
    let mut project = Project::new("test", R24);
    let track = project.add_track(TrackKind::Adjustment, "FX");
    let left = project
        .add_generated(track, Generator::Adjustment, tr(0, 24))
        .unwrap();
    project
        .add_generated(track, Generator::Adjustment, tr(24, 24))
        .unwrap();
    assert!(matches!(
        project.add_transition(left, "crossfade"),
        Err(ModelError::InvalidParam(_))
    ));
    assert!(!project.has_transitions());
}

#[test]
fn set_and_remove_transition_duration() {
    let (mut project, left, _right, track) = project_with_abutting_pair();
    project.add_transition(left, "wipe_left").unwrap();
    project.set_transition_duration(left, 12).unwrap();
    assert_eq!(
        project
            .timeline()
            .track(track)
            .unwrap()
            .transition_at(left)
            .unwrap()
            .duration,
        12
    );
    project.remove_transition(left).unwrap();
    assert!(
        project
            .timeline()
            .track(track)
            .unwrap()
            .transition_at(left)
            .is_none()
    );
    assert!(matches!(
        project.remove_transition(left),
        Err(ModelError::InvalidParam(_))
    ));
}

#[test]
fn prune_drops_transition_when_junction_breaks() {
    let (mut project, left, _right, track) = project_with_abutting_pair();
    project.add_transition(left, "slide").unwrap();
    assert!(project.has_transitions());

    // Move the left clip so the pair no longer abuts.
    project.move_clip(left, track, rt(100)).unwrap();
    assert!(project.prune_dead_transitions());
    assert!(!project.has_transitions());
    // Idempotent: a second prune finds nothing to do.
    assert!(!project.prune_dead_transitions());
}

#[test]
fn transitions_snapshot_round_trips() {
    let (mut project, left, _right, _track) = project_with_abutting_pair();
    project.add_transition(left, "dip_to_black").unwrap();
    let snapshot = project.transitions_snapshot();

    project.remove_transition(left).unwrap();
    assert!(!project.has_transitions());

    project.restore_transitions(snapshot);
    assert!(project.has_transitions());
}

// --- Project::new -----------------------------------------------------

#[test]
fn new_creates_empty_project_at_frame_rate() {
    let project = Project::new("my edit", R24);
    assert_eq!(project.schema, ProjectSchema::current());
    assert_eq!(project.metadata, ProjectMetadata::default());
    assert_eq!(project.name, "my edit");
    assert_eq!(project.timeline().frame_rate, R24);
    assert_eq!(project.media_count(), 0);
    assert_eq!(project.timeline().track_count(), 0);
    assert_eq!(project.timeline().clip_count(), 0);
    assert!(project.id.raw() >= 1);
}

#[test]
fn new_accepts_string_name() {
    let owned = Project::new(String::from("owned"), R24);
    assert_eq!(owned.name, "owned");
}

// --- media pool -------------------------------------------------------

#[test]
fn add_media_returns_id_and_lookup_works() {
    let mut project = Project::new("test", R24);
    let media = sample_media(R24, 500);
    let id = project.add_media(media.clone());

    assert_eq!(project.media_count(), 1);
    assert_eq!(project.media(id).unwrap().path(), media.path());
}

#[test]
fn media_iter_visits_all_sources() {
    let mut project = Project::new("test", R24);
    project.add_media(sample_media(R24, 10));
    project.add_media(sample_media(R24, 20));
    let durations: Vec<i64> = project.media_iter().map(|m| m.duration.value).collect();
    assert_eq!(durations.len(), 2);
    assert!(durations.contains(&10));
    assert!(durations.contains(&20));
}

#[test]
fn find_media_by_path_matches_same_file() {
    let mut project = Project::new("test", R24);
    let id = project.add_media(sample_media(R24, 10));
    assert_eq!(
        project.find_media_by_path(Path::new("/tmp/sample.mp4")),
        Some(id)
    );
    assert_eq!(
        project.find_media_by_path(Path::new("/tmp/other.mp4")),
        None
    );
}

#[test]
fn remove_media_unknown_errors() {
    let mut project = Project::new("test", R24);
    let missing = MediaId::from_raw(99);
    assert_eq!(
        project.remove_media(missing),
        Err(ModelError::UnknownMedia(missing))
    );
}

#[test]
fn is_media_referenced_reflects_clip_usage() {
    let (mut project, media_id, track) = project_with_media(100);
    assert!(!project.is_media_referenced(media_id));

    let clip = project.add_clip(track, media_id, tr(0, 50), rt(0)).unwrap();
    assert!(project.is_media_referenced(media_id));

    project.remove_clip(clip);
    assert!(!project.is_media_referenced(media_id));
}

#[test]
fn remove_media_succeeds_when_unreferenced() {
    let mut project = Project::new("test", R24);
    let id = project.add_media(sample_media(R24, 10));
    let removed = project.remove_media(id).unwrap();
    assert_eq!(removed.duration.value, 10);
    assert_eq!(project.media_count(), 0);
}

// --- add_clip ---------------------------------------------------------

#[test]
fn add_clip_places_media_with_rate_conform() {
    let mut project = Project::new("test", R24);
    let media_id = project.add_media(sample_media(R30, 1000));
    let track = project.add_track(TrackKind::Video, "V1");

    let clip = project
        .add_clip(track, media_id, tr_at(0, 120, R30), rt(10))
        .unwrap();

    let placed = project.clip(clip).unwrap();
    assert_eq!(placed.start().value, 10);
    assert_eq!(placed.timeline.duration.value, 96);
    assert_eq!(placed.source_range(), Some(tr_at(0, 120, R30)));
}

#[test]
fn add_clip_rejects_unknown_track_and_media() {
    let (mut project, media_id, _) = project_with_media(100);
    let missing_track = TrackId::from_raw(999);
    let missing_media = MediaId::from_raw(999);

    assert_eq!(
        project.add_clip(missing_track, media_id, tr(0, 10), rt(0)),
        Err(ModelError::UnknownTrack(missing_track))
    );
    let track = project.add_track(TrackKind::Video, "V2");
    assert_eq!(
        project.add_clip(track, missing_media, tr(0, 10), rt(0)),
        Err(ModelError::UnknownMedia(missing_media))
    );
}

#[test]
fn add_clip_rejects_empty_source_and_out_of_bounds() {
    let (mut project, media_id, track) = project_with_media(100);

    assert_eq!(
        project.add_clip(track, media_id, tr(0, 0), rt(0)),
        Err(ModelError::InvalidRange)
    );
    assert_eq!(
        project.add_clip(track, media_id, tr(-1, 10), rt(0)),
        Err(ModelError::SourceOutOfBounds)
    );
    assert_eq!(
        project.add_clip(track, media_id, tr(95, 10), rt(0)),
        Err(ModelError::SourceOutOfBounds)
    );
}

#[test]
fn add_clip_rejects_rate_mismatches() {
    let (mut project, media_id, track) = project_with_media(100);

    assert_eq!(
        project.add_clip(track, media_id, tr_at(0, 10, R30), rt(0)),
        Err(ModelError::RateMismatch {
            expected: R24,
            got: R30,
        })
    );
    let bad_start = RationalTime::new(0, R30);
    assert_eq!(
        project.add_clip(track, media_id, tr(0, 10), bad_start),
        Err(ModelError::RateMismatch {
            expected: R24,
            got: R30,
        })
    );
}

#[test]
fn add_clip_rejects_overlap() {
    let (mut project, media_id, track) = project_with_media(1000);
    project
        .add_clip(track, media_id, tr(0, 100), rt(0))
        .unwrap();
    assert_eq!(
        project.add_clip(track, media_id, tr(0, 50), rt(50)),
        Err(ModelError::Overlap(track))
    );
}

// --- add_generated ----------------------------------------------------

#[test]
fn add_generated_without_media() {
    let mut project = Project::new("test", R24);
    let track = project.add_track(TrackKind::Text, "Titles");
    let clip = project
        .add_generated(track, Generator::text("Hi"), tr(0, 24))
        .unwrap();

    assert!(project.clip(clip).unwrap().is_generated());
    assert_eq!(project.media_count(), 0);
}

#[test]
fn add_generated_rejects_empty_and_wrong_rate() {
    let mut project = Project::new("test", R24);
    let track = project.add_track(TrackKind::Video, "V1");

    assert_eq!(
        project.add_generated(track, Generator::Adjustment, tr(0, 0)),
        Err(ModelError::InvalidRange)
    );
    assert_eq!(
        project.add_generated(
            track,
            Generator::shape(Shape::Rectangle, [255, 255, 255, 255]),
            tr_at(0, 10, R30),
        ),
        Err(ModelError::RateMismatch {
            expected: R24,
            got: R30,
        })
    );
}

// --- duplicate_clip ---------------------------------------------------

#[test]
fn duplicate_clip_preserves_full_property_snapshot() {
    let (mut project, media_id, source_track) = project_with_media(1_000);
    let destination_track = project.add_track(TrackKind::Video, "V2");
    let source_id = project
        .add_clip(source_track, media_id, tr(17, 80), rt(10))
        .unwrap();
    let source_link = crate::LinkId::next();

    {
        let clip = project.timeline_mut().clip_mut(source_id).unwrap();
        clip.link = Some(source_link);

        clip.transform.position.set_keyframe(
            0,
            [0.1, -0.2],
            Easing::Bezier {
                points: [0.2, 0.8, 0.7, 0.1],
            },
        );
        clip.transform
            .position
            .set_keyframe(60, [0.7, 0.4], Easing::EaseOut);
        clip.transform.anchor_point = Param::Constant([0.25, 0.75]);
        clip.transform.scale.set_keyframe(0, 0.8, Easing::EaseIn);
        clip.transform
            .scale
            .set_keyframe(60, 1.6, Easing::EaseInOut);
        clip.transform.rotation = Param::Constant(27.5);
        clip.transform.opacity = Param::Constant(0.65);

        clip.speed = Rational::new(3, 2);
        clip.reversed = true;
        clip.speed_curve.set_keyframe(0, 0.75, Easing::EaseIn);
        clip.speed_curve
            .set_keyframe(crate::SPEED_CURVE_SCALE, 1.5, Easing::EaseOut);
        clip.preserve_pitch = false;

        clip.volume.set_keyframe(0, 0.2, Easing::EaseIn);
        clip.volume.set_keyframe(60, 1.4, Easing::EaseOut);
        clip.fade_in = 9;
        clip.fade_out = 11;
        clip.denoise = true;

        clip.crop = CropRect {
            x: 0.1,
            y: 0.2,
            w: 0.7,
            h: 0.6,
        };
        clip.flip_h = true;
        clip.flip_v = true;

        let mut effect = EffectInstance::new("gaussian_blur");
        effect
            .set_param_keyframe(0, 0, 2.0, Easing::EaseIn)
            .unwrap();
        effect
            .set_param_keyframe(0, 60, 12.0, Easing::EaseOut)
            .unwrap();
        clip.effects = vec![effect];
        clip.beats = vec![19, 31, 47];

        clip.mask = Some(Mask {
            kind: MaskKind::Heart,
            feather: 0.35.into(),
            invert: true,
        });
        clip.chroma_key = Some(ChromaKey {
            rgb: [3, 240, 17],
            strength: 0.72.into(),
            shadow: 0.18.into(),
        });
        clip.stabilize = Some(StabilizeLevel::MaxSmooth);
        clip.filter = Some(Filter {
            id: "vivid".into(),
            intensity: 0.63.into(),
        });
        clip.lut = Some(crate::look::Lut {
            path: "/tmp/cinematic.cube".into(),
            intensity: 0.71.into(),
        });
        clip.adjust = ColorAdjustments {
            brightness: 0.1.into(),
            contrast: (-0.2).into(),
            saturation: 0.3.into(),
            exposure: (-0.4).into(),
            temperature: 0.5.into(),
        };
        clip.animation_in = Some(AnimationRef::new("zoom_in"));
        clip.animation_out = Some(AnimationRef::new("drop"));
        clip.animation_combo = Some(AnimationRef::new("pulse"));
        clip.audio_role = Some(AudioRole::Voiceover);
        clip.replaceable = Some(
            Replaceable::new(7)
                .with_accepts(SlotMedia::VideoOnly)
                .with_label("Hero"),
        );
        clip.text_editable = true;
    }

    let source_before = project.clip(source_id).unwrap().clone();
    let destination_start = rt(240);
    let duplicate_id = project
        .duplicate_clip(source_id, destination_track, destination_start)
        .unwrap();
    let duplicate = project.clip(duplicate_id).unwrap();

    let mut expected = source_before.clone();
    expected.id = duplicate_id;
    expected.timeline.start = destination_start;
    expected.link = None;
    assert_eq!(
        duplicate, &expected,
        "normalizing only id/start/link makes the full snapshots equal"
    );
    assert_eq!(
        project.clip(source_id).unwrap(),
        &source_before,
        "duplication never mutates the source"
    );
    assert_ne!(duplicate_id, source_id);
    assert_eq!(
        project.timeline().track_of(duplicate_id),
        Some(destination_track)
    );
}

#[test]
fn duplicate_clip_clones_generated_content_and_template_flag() {
    let mut project = Project::new("test", R24);
    let source_track = project.add_track(TrackKind::Text, "Titles");
    let destination_track = project.add_track(TrackKind::Text, "Titles 2");
    let source_id = project
        .add_generated(source_track, Generator::text("Animated title"), tr(5, 40))
        .unwrap();
    project.set_text_editable(source_id, true).unwrap();
    project
        .set_param_keyframe(
            source_id,
            ClipParam::Scale,
            rt(5),
            ParamValue::Scalar(0.75),
            Easing::EaseInOut,
        )
        .unwrap();

    let source_before = project.clip(source_id).unwrap().clone();
    let start = rt(100);
    let duplicate_id = project
        .duplicate_clip(source_id, destination_track, start)
        .unwrap();

    let mut expected = source_before.clone();
    expected.id = duplicate_id;
    expected.timeline.start = start;
    expected.link = None;
    assert_eq!(project.clip(duplicate_id).unwrap(), &expected);
    assert!(project.clip(duplicate_id).unwrap().text_editable);
    assert_eq!(project.clip(source_id).unwrap(), &source_before);
}

#[test]
fn duplicate_clip_unlinks_copy_without_changing_source_group() {
    let (mut project, media_id, source_track) = project_with_media(300);
    let destination_track = project.add_track(TrackKind::Video, "V2");
    let first = project
        .add_clip(source_track, media_id, tr(0, 20), rt(0))
        .unwrap();
    let second = project
        .add_clip(source_track, media_id, tr(20, 20), rt(20))
        .unwrap();
    let group = crate::LinkId::next();
    for id in [first, second] {
        project.timeline_mut().clip_mut(id).unwrap().link = Some(group);
    }

    let duplicate = project
        .duplicate_clip(first, destination_track, rt(100))
        .unwrap();

    assert_eq!(project.clip(first).unwrap().link, Some(group));
    assert_eq!(project.clip(second).unwrap().link, Some(group));
    assert_eq!(project.clip(duplicate).unwrap().link, None);
}

#[test]
fn duplicate_clip_rejections_are_atomic() {
    let (mut project, media_id, source_track) = project_with_media(500);
    let destination_track = project.add_track(TrackKind::Video, "V2");
    let incompatible_track = project.add_track(TrackKind::Text, "Titles");
    let source = project
        .add_clip(source_track, media_id, tr(0, 40), rt(0))
        .unwrap();
    let blocker = project
        .add_clip(destination_track, media_id, tr(40, 40), rt(100))
        .unwrap();
    let source_before = project.clip(source).unwrap().clone();
    let blocker_before = project.clip(blocker).unwrap().clone();
    let clip_count_before = project.timeline().clip_count();
    let project_before = serde_json::to_value(&project).unwrap();

    macro_rules! assert_atomic_rejection {
        ($result:expr, $expected:expr) => {{
            assert_eq!($result, Err($expected));
            assert_eq!(project.timeline().clip_count(), clip_count_before);
            assert_eq!(project.clip(source).unwrap(), &source_before);
            assert_eq!(project.clip(blocker).unwrap(), &blocker_before);
            assert_eq!(
                serde_json::to_value(&project).unwrap(),
                project_before,
                "the complete serialized project remains unchanged"
            );
        }};
    }

    let equivalent_but_noncanonical_rate = Rational::new(48, 2);
    assert_atomic_rejection!(
        project.duplicate_clip(
            source,
            destination_track,
            RationalTime::new(200, equivalent_but_noncanonical_rate)
        ),
        ModelError::RateMismatch {
            expected: R24,
            got: equivalent_but_noncanonical_rate,
        }
    );
    assert_atomic_rejection!(
        project.duplicate_clip(source, destination_track, rt(-1)),
        ModelError::InvalidRange
    );

    let missing_clip = ClipId::from_raw(u64::MAX - 1);
    assert_atomic_rejection!(
        project.duplicate_clip(missing_clip, destination_track, rt(200)),
        ModelError::UnknownClip(missing_clip)
    );

    let missing_track = TrackId::from_raw(u64::MAX - 1);
    assert_atomic_rejection!(
        project.duplicate_clip(source, missing_track, rt(200)),
        ModelError::UnknownTrack(missing_track)
    );
    assert_atomic_rejection!(
        project.duplicate_clip(source, incompatible_track, rt(200)),
        ModelError::IncompatibleTrackKind {
            track: incompatible_track,
            kind: TrackKind::Text,
        }
    );
    assert_atomic_rejection!(
        project.duplicate_clip(source, destination_track, rt(110)),
        ModelError::Overlap(destination_track)
    );
}

// --- remove_clip ------------------------------------------------------

#[test]
fn remove_clip_returns_clip_and_leaves_gap() {
    let (mut project, media_id, track) = project_with_media(200);
    let a = project.add_clip(track, media_id, tr(0, 50), rt(0)).unwrap();
    let b = project
        .add_clip(track, media_id, tr(50, 50), rt(100))
        .unwrap();

    let removed = project.remove_clip(a).unwrap();
    assert_eq!(removed.id, a);
    assert!(project.clip(a).is_none());
    assert_eq!(project.clip(b).unwrap().start().value, 100);
    assert_eq!(project.timeline().clip_count(), 1);
}

#[test]
fn remove_clip_unknown_returns_none() {
    let (mut project, _, _) = project_with_media(10);
    assert!(project.remove_clip(ClipId::from_raw(404)).is_none());
}

// --- split_clip -------------------------------------------------------

#[test]
fn split_clip_divides_media_and_generated() {
    let (mut project, media_id, track) = project_with_media(500);
    let media_clip = project
        .add_clip(track, media_id, tr(100, 100), rt(0))
        .unwrap();
    let right = project.split_clip(media_clip, rt(40)).unwrap();
    assert_eq!(project.clip(media_clip).unwrap().timeline, tr(0, 40));
    assert_eq!(project.clip(right).unwrap().timeline, tr(40, 60));

    let fx = project.add_track(TrackKind::Adjustment, "FX");
    let generated = project
        .add_generated(fx, Generator::Adjustment, tr(200, 100))
        .unwrap();
    let generated_right = project.split_clip(generated, rt(250)).unwrap();
    assert_eq!(project.clip(generated).unwrap().timeline, tr(200, 50));
    assert_eq!(project.clip(generated_right).unwrap().timeline, tr(250, 50));
}

#[test]
fn split_clip_preserves_full_property_snapshot_and_rebases_tail_curves() {
    let (mut project, media_id, track) = project_with_media(1_000);
    let clip_id = project
        .add_clip(track, media_id, tr(17, 180), rt(10))
        .unwrap();
    project
        .set_clip_speed(clip_id, Rational::new(3, 2), true)
        .unwrap();
    let link = crate::LinkId::next();

    {
        let clip = project.timeline_mut().clip_mut(clip_id).unwrap();
        clip.link = Some(link);
        clip.transform.position.set_keyframe(
            0,
            [0.1, -0.2],
            Easing::Bezier {
                points: [0.2, 0.8, 0.7, 0.1],
            },
        );
        clip.transform
            .position
            .set_keyframe(100, [0.7, 0.4], Easing::EaseOut);
        clip.transform.anchor_point = Param::Constant([0.25, 0.75]);
        clip.transform.scale.set_keyframe(0, 0.8, Easing::EaseIn);
        clip.transform
            .scale
            .set_keyframe(100, 1.6, Easing::EaseInOut);
        clip.transform.rotation = Param::Constant(27.5);
        clip.transform.opacity = Param::Constant(0.65);

        // A normalized, non-default representation whose value is flat
        // keeps this property-focused test on the exact constant-rate
        // source path while proving the ramp field survives.
        clip.speed_curve.set_keyframe(0, 1.0, Easing::Linear);
        clip.speed_curve
            .set_keyframe(crate::SPEED_CURVE_SCALE, 1.0, Easing::Linear);
        clip.preserve_pitch = false;

        clip.volume.set_keyframe(0, 0.2, Easing::EaseIn);
        clip.volume.set_keyframe(100, 1.4, Easing::EaseOut);
        clip.fade_in = 9;
        clip.fade_out = 11;
        clip.denoise = true;

        clip.crop = CropRect {
            x: 0.1,
            y: 0.2,
            w: 0.7,
            h: 0.6,
        };
        clip.flip_h = true;
        clip.flip_v = true;

        let mut effect = EffectInstance::new("gaussian_blur");
        effect
            .set_param_keyframe(0, 0, 2.0, Easing::EaseIn)
            .unwrap();
        effect
            .set_param_keyframe(0, 100, 12.0, Easing::EaseOut)
            .unwrap();
        clip.effects = vec![effect];
        clip.beats = vec![19, 31, 47, 139];

        clip.mask = Some(Mask {
            kind: crate::look::MaskKind::Heart,
            feather: 0.35.into(),
            invert: true,
        });
        clip.chroma_key = Some(ChromaKey {
            rgb: [3, 240, 17],
            strength: 0.72.into(),
            shadow: 0.18.into(),
        });
        clip.stabilize = Some(StabilizeLevel::MaxSmooth);
        clip.filter = Some(Filter {
            id: "vivid".into(),
            intensity: 0.63.into(),
        });
        clip.lut = Some(Lut {
            path: "/tmp/cinematic.cube".into(),
            intensity: 0.71.into(),
        });
        clip.adjust = ColorAdjustments {
            brightness: 0.1.into(),
            contrast: (-0.2).into(),
            saturation: 0.3.into(),
            exposure: (-0.4).into(),
            temperature: 0.5.into(),
        };
        clip.animation_in = Some(AnimationRef::new("zoom_in"));
        clip.animation_out = Some(AnimationRef::new("drop"));
        clip.audio_role = Some(AudioRole::Voiceover);
        clip.replaceable = Some(
            Replaceable::new(7)
                .with_accepts(SlotMedia::VideoOnly)
                .with_label("Hero"),
        );
        clip.text_editable = true;
    }

    let before = project.clip(clip_id).unwrap().clone();
    let (expected_left_curve, expected_right_curve) =
        split_speed_curve(&before.speed_curve, 0.5).unwrap();
    let right_id = project.split_clip(clip_id, rt(70)).unwrap();

    let mut expected_left = before.clone();
    expected_left.timeline = tr(10, 60);
    expected_left.content = ClipSource::Media {
        media: media_id,
        source: tr(107, 90),
    };
    expected_left.fade_out = 0;
    expected_left.animation_out = None;
    expected_left.speed_curve = expected_left_curve;
    assert_eq!(
        project.clip(clip_id).unwrap(),
        &expected_left,
        "the head differs only in its split geometry and right-edge properties"
    );

    let mut expected_right = before.clone();
    expected_right.id = right_id;
    expected_right.timeline = tr(70, 60);
    expected_right.content = ClipSource::Media {
        media: media_id,
        source: tr(17, 90),
    };
    expected_right.link = None;
    expected_right.shift_timeline_params(-60).unwrap();
    expected_right.speed_curve = expected_right_curve;
    expected_right.fade_in = 0;
    expected_right.animation_in = None;
    assert_eq!(
        project.clip(right_id).unwrap(),
        &expected_right,
        "normalizing only split-specific fields makes the full snapshots equal"
    );

    let right = project.clip(right_id).unwrap();
    for tail_tick in [0, 17, 59] {
        assert_eq!(
            right.transform.sample(tail_tick),
            before.transform.sample(60 + tail_tick)
        );
        assert_eq!(
            right.volume.sample(tail_tick),
            before.volume.sample(60 + tail_tick)
        );
        assert_eq!(
            right.effects[0].sample_param("radius", tail_tick as f64),
            before.effects[0].sample_param("radius", (60 + tail_tick) as f64)
        );
    }
    assert_eq!(project.clip(clip_id).unwrap().link, Some(link));
    assert_eq!(
        right.link, None,
        "callers remain responsible for relinking tails"
    );
}

#[test]
fn split_clip_uses_forward_constant_speed_source_boundary() {
    let (mut project, media_id, track) = project_with_media(1_000);
    let clip_id = project
        .add_clip(track, media_id, tr(100, 120), rt(10))
        .unwrap();
    project
        .set_clip_speed(clip_id, Rational::new(3, 2), false)
        .unwrap();
    let before = project.clip(clip_id).unwrap().clone();

    let right_id = project.split_clip(clip_id, rt(42)).unwrap();
    let left = project.clip(clip_id).unwrap();
    let right = project.clip(right_id).unwrap();
    assert_eq!(left.timeline, tr(10, 32));
    assert_eq!(right.timeline, tr(42, 48));
    assert_eq!(left.source_range(), Some(tr(100, 48)));
    assert_eq!(right.source_range(), Some(tr(148, 72)));

    for tick in [10, 25, 41] {
        assert_eq!(
            left.source_time_at(rt(tick)),
            before.source_time_at(rt(tick))
        );
    }
    for tick in [42, 55, 75, 89] {
        assert_eq!(
            right.source_time_at(rt(tick)),
            before.source_time_at(rt(tick))
        );
    }
}

#[test]
fn split_clip_uses_reversed_constant_speed_source_boundary() {
    let (mut project, media_id, track) = project_with_media(1_000);
    let clip_id = project
        .add_clip(track, media_id, tr(100, 120), rt(10))
        .unwrap();
    project
        .set_clip_speed(clip_id, Rational::new(3, 2), true)
        .unwrap();
    let before = project.clip(clip_id).unwrap().clone();

    let right_id = project.split_clip(clip_id, rt(42)).unwrap();
    let left = project.clip(clip_id).unwrap();
    let right = project.clip(right_id).unwrap();
    assert_eq!(left.source_range(), Some(tr(172, 48)));
    assert_eq!(right.source_range(), Some(tr(100, 72)));

    for tick in [10, 25, 41] {
        assert_eq!(
            left.source_time_at(rt(tick)),
            before.source_time_at(rt(tick))
        );
    }
    for tick in [42, 55, 75, 89] {
        assert_eq!(
            right.source_time_at(rt(tick)),
            before.source_time_at(rt(tick))
        );
    }
}

#[test]
fn split_clip_segments_speed_curve_and_preserves_source_mapping() {
    let (mut project, media_id, track) = project_with_media(1_000);
    let clip_id = project
        .add_clip(track, media_id, tr(100, 200), rt(0))
        .unwrap();
    let mut curve = Param::Constant(1.0);
    curve.set_keyframe(0, 1.0, Easing::Linear);
    curve.set_keyframe(crate::SPEED_CURVE_SCALE, 3.0, Easing::Linear);
    project.set_clip_speed_curve(clip_id, Some(curve)).unwrap();
    let before = project.clip(clip_id).unwrap().clone();
    assert_eq!(before.timeline, tr(0, 100));

    let right_id = project.split_clip(clip_id, rt(50)).unwrap();
    let left = project.clip(clip_id).unwrap();
    let right = project.clip(right_id).unwrap();
    assert_eq!(left.source_range(), Some(tr(100, 75)));
    assert_eq!(right.source_range(), Some(tr(175, 125)));
    assert_eq!(
        left.source_range().unwrap().end_tick(),
        right.source_range().unwrap().start.value,
        "forward source windows stay contiguous at the split"
    );
    assert_eq!(
        left.speed_curve.sample_at(500.0),
        before.speed_curve.sample_at(250.0)
    );
    assert_eq!(
        right.speed_curve.sample_at(500.0),
        before.speed_curve.sample_at(750.0)
    );

    for tick in [0, 10, 25, 49] {
        assert_eq!(
            left.source_time_at(rt(tick)),
            before.source_time_at(rt(tick))
        );
    }
    for tick in [50, 60, 75, 99] {
        assert_eq!(
            right.source_time_at(rt(tick)),
            before.source_time_at(rt(tick))
        );
    }
    assert_eq!(
        right.source_time_at(rt(50)).unwrap(),
        before.source_time_at(rt(50)).unwrap(),
        "the first tail frame is the original frame at the boundary"
    );
}

#[test]
fn split_clip_rejects_a_cut_inside_one_source_frame() {
    let (mut project, media_id, track) = project_with_media(100);
    let clip_id = project
        .add_clip(track, media_id, tr(0, 100), rt(0))
        .unwrap();
    project
        .set_clip_speed(clip_id, Rational::new(1, 2), false)
        .unwrap();
    let before = project.clip(clip_id).unwrap().clone();

    assert_eq!(
        project.split_clip(clip_id, rt(1)),
        Err(ModelError::InvalidRange)
    );
    assert_eq!(project.clip(clip_id), Some(&before));
    assert_eq!(project.timeline().clip_count(), 1);
}

#[test]
fn split_after_reload_allocates_a_fresh_tail_id() {
    // Regression for the cross-session id-collision bug: a clip persisted
    // in one session must not have its id re-handed to a clip created in
    // the next. Round-tripping through JSON (as loading a project does)
    // reseeds the id allocator, so the split's tail id is distinct from
    // every id already in the project.
    let (mut project, media_id, track) = project_with_media(500);
    let persisted = project
        .add_clip(track, media_id, tr(0, 100), rt(0))
        .unwrap();

    let json = serde_json::to_string(&project).unwrap();
    let mut reloaded: Project = serde_json::from_str(&json).unwrap();

    let tail = reloaded.split_clip(persisted, rt(40)).unwrap();
    assert_ne!(tail, persisted, "tail must not reuse the persisted clip id");
    assert_eq!(reloaded.timeline().clip_count(), 2);
    // Both halves are addressable and distinct — the overwrite that used
    // to drop the left half can no longer happen.
    assert_eq!(reloaded.clip(persisted).unwrap().timeline, tr(0, 40));
    assert_eq!(reloaded.clip(tail).unwrap().timeline, tr(40, 60));
}

#[test]
fn split_clip_rejects_at_boundaries() {
    let (mut project, media_id, track) = project_with_media(500);
    let clip = project
        .add_clip(track, media_id, tr(0, 100), rt(10))
        .unwrap();
    assert_eq!(
        project.split_clip(clip, rt(10)),
        Err(ModelError::InvalidRange)
    );
    assert_eq!(
        project.split_clip(clip, rt(110)),
        Err(ModelError::InvalidRange)
    );
}

#[test]
fn split_clip_fails_closed_when_time_anchored_state_cannot_continue() {
    let (mut project, media_id, track) = project_with_media(500);
    let clip_id = project
        .add_clip(track, media_id, tr(0, 100), rt(0))
        .unwrap();

    project.timeline_mut().clip_mut(clip_id).unwrap().fade_in = 40;
    let before = project.clip(clip_id).unwrap().clone();
    assert!(matches!(
        project.split_clip(clip_id, rt(20)),
        Err(ModelError::InvalidParam(message)) if message.contains("fade-in")
    ));
    assert_eq!(project.clip(clip_id), Some(&before));
    assert_eq!(project.timeline().clip_count(), 1);

    let clip = project.timeline_mut().clip_mut(clip_id).unwrap();
    clip.fade_in = 0;
    clip.animation_combo = Some(AnimationRef::new("pulse"));
    let before = project.clip(clip_id).unwrap().clone();
    assert!(matches!(
        project.split_clip(clip_id, rt(25)),
        Err(ModelError::InvalidParam(message)) if message.contains("phase boundary")
    ));
    assert_eq!(project.clip(clip_id), Some(&before));
    assert_eq!(project.timeline().clip_count(), 1);
}

#[test]
fn split_clip_rejects_unknown_clip() {
    let (mut project, _, _) = project_with_media(10);
    let unknown = ClipId::from_raw(999);
    assert_eq!(
        project.split_clip(unknown, rt(5)),
        Err(ModelError::UnknownClip(unknown))
    );
}

#[test]
fn split_clip_rejects_wrong_timeline_rate() {
    let (mut project, media_id, track) = project_with_media(100);
    let clip = project.add_clip(track, media_id, tr(0, 50), rt(0)).unwrap();
    assert_eq!(
        project.split_clip(clip, RationalTime::new(25, R30)),
        Err(ModelError::RateMismatch {
            expected: R24,
            got: R30,
        })
    );
}

// --- trim_clip --------------------------------------------------------

#[test]
fn trim_clip_tail_shortens_media_source() {
    let (mut project, media_id, track) = project_with_media(500);
    let clip = project
        .add_clip(track, media_id, tr(0, 100), rt(0))
        .unwrap();

    project.trim_clip(clip, tr(0, 60)).unwrap();
    let trimmed = project.clip(clip).unwrap();
    assert_eq!(trimmed.timeline, tr(0, 60));
    assert_eq!(trimmed.source_range(), Some(tr(0, 60)));
}

#[test]
fn trim_extends_image_clips_past_the_default_window() {
    let mut project = Project::new("stills", R24);
    let media_id = project.add_media(MediaSource::image("/tmp/a.png", 800, 600));
    let track = project.add_track(TrackKind::Video, "V1");
    let full = project.media(media_id).unwrap().full_range();
    let clip = project.add_clip(track, media_id, full, rt(0)).unwrap();
    // The 5s default placement at 24 fps.
    assert_eq!(project.clip(clip).unwrap().timeline, tr(0, 120));

    // A still stretches to any length: 10s here, double its pool entry.
    project.trim_clip(clip, tr(0, 240)).unwrap();
    let stretched = project.clip(clip).unwrap();
    assert_eq!(stretched.timeline, tr(0, 240));
    let source = stretched.source_range().unwrap();
    assert_eq!(source.start.value, 0);
    assert!(source.duration.value > crate::media::STILL_DEFAULT_DURATION_TICKS);
}

#[test]
fn add_clip_allows_oversized_image_windows() {
    let mut project = Project::new("stills", R24);
    let media_id = project.add_media(MediaSource::image("/tmp/a.png", 800, 600));
    let track = project.add_track(TrackKind::Video, "V1");
    // Place a 20s clip from the 5s pool entry directly (agent add_clip).
    let window = TimeRange::at_rate(0, 20_000, crate::media::STILL_TICK_RATE);
    let clip = project.add_clip(track, media_id, window, rt(0)).unwrap();
    assert_eq!(project.clip(clip).unwrap().timeline, tr(0, 480));

    // Video media keeps its real material bound.
    let video = project.add_media(sample_media(R24, 100));
    assert!(matches!(
        project.add_clip(track, video, tr(0, 101), rt(600)),
        Err(ModelError::SourceOutOfBounds)
    ));
}

#[test]
fn trim_clip_generated_only_moves_timeline() {
    let mut project = Project::new("test", R24);
    let track = project.add_track(TrackKind::Adjustment, "FX");
    let clip = project
        .add_generated(track, Generator::Adjustment, tr(0, 100))
        .unwrap();

    project.trim_clip(clip, tr(20, 50)).unwrap();
    let trimmed = project.clip(clip).unwrap();
    assert_eq!(trimmed.timeline, tr(20, 50));
    assert_eq!(trimmed.source_range(), None);
}

// --- set_clip_speed (M1) -----------------------------------------------

#[test]
fn set_clip_speed_rescales_timeline_duration() {
    let (mut project, media_id, track) = project_with_media(500);
    let clip = project
        .add_clip(track, media_id, tr(0, 100), rt(0))
        .unwrap();

    // 2× halves the footprint; the source window is untouched.
    project
        .set_clip_speed(clip, Rational::new(2, 1), false)
        .unwrap();
    let c = project.clip(clip).unwrap();
    assert_eq!(c.timeline, tr(0, 50));
    assert_eq!(c.source_range(), Some(tr(0, 100)));
    assert_eq!(c.speed, Rational::new(2, 1));

    // Back to 1× restores the original footprint.
    project
        .set_clip_speed(clip, Rational::new(1, 1), false)
        .unwrap();
    assert_eq!(project.clip(clip).unwrap().timeline, tr(0, 100));

    // Slow motion stretches it.
    project
        .set_clip_speed(clip, Rational::new(1, 2), false)
        .unwrap();
    assert_eq!(project.clip(clip).unwrap().timeline, tr(0, 200));
}

#[test]
fn set_clip_pitch_toggles_the_flag_without_moving_the_clip() {
    let (mut project, media_id, track) = project_with_media(500);
    let clip = project
        .add_clip(track, media_id, tr(0, 100), rt(0))
        .unwrap();
    project
        .set_clip_speed(clip, Rational::new(2, 1), false)
        .unwrap();
    assert!(
        project.clip(clip).unwrap().preserve_pitch,
        "locked by default"
    );

    // Unlocking pitch keeps the retimed footprint (pure audio property).
    project.set_clip_pitch(clip, false).unwrap();
    let c = project.clip(clip).unwrap();
    assert!(!c.preserve_pitch);
    assert_eq!(c.timeline, tr(0, 50));

    project.set_clip_pitch(clip, true).unwrap();
    assert!(project.clip(clip).unwrap().preserve_pitch);
}

#[test]
fn set_clip_pitch_rejects_generated_clips() {
    let mut project = Project::new("p", R24);
    let track = project.add_track(TrackKind::Sticker, "T1");
    let clip = project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [0, 0, 0, 255],
            },
            tr(0, 10),
        )
        .unwrap();
    assert!(
        project.set_clip_pitch(clip, false).is_err(),
        "no audio to retime"
    );
}

#[test]
fn set_clip_denoise_toggles_and_returns_the_previous_flag() {
    let (mut project, media_id, track) = project_with_media(500);
    let clip = project
        .add_clip(track, media_id, tr(0, 100), rt(0))
        .unwrap();
    assert!(!project.clip(clip).unwrap().denoise, "off by default");

    // Turning it on returns the old value (false); turning it off again
    // returns true — the engine uses this for the undo inverse.
    assert!(!project.set_clip_denoise(clip, true).unwrap());
    assert!(project.clip(clip).unwrap().denoise);
    assert!(project.set_clip_denoise(clip, false).unwrap());
    assert!(!project.clip(clip).unwrap().denoise);
}

#[test]
fn set_clip_denoise_rejects_generated_clips() {
    let mut project = Project::new("p", R24);
    let track = project.add_track(TrackKind::Sticker, "T1");
    let clip = project
        .add_generated(
            track,
            Generator::SolidColor {
                rgba: [0, 0, 0, 255],
            },
            tr(0, 10),
        )
        .unwrap();
    assert!(
        project.set_clip_denoise(clip, true).is_err(),
        "no audio to clean"
    );
}

#[test]
fn set_clip_speed_curve_rederives_duration_from_average() {
    let (mut project, media_id, track) = project_with_media(500);
    let clip = project
        .add_clip(track, media_id, tr(0, 100), rt(0))
        .unwrap();

    // Average 2× ramp halves the footprint (source ÷ avg), like constant 2×.
    let ramp = crate::clip::speed_preset("montage").unwrap();
    let avg = {
        let mut probe = project.clip(clip).unwrap().clone();
        probe.speed_curve = ramp.clone();
        probe.speed_curve_average()
    };
    project.set_clip_speed_curve(clip, Some(ramp)).unwrap();
    let expected = (100.0 / avg).round() as i64;
    assert_eq!(project.clip(clip).unwrap().timeline, tr(0, expected));
    assert!(project.clip(clip).unwrap().has_speed_curve());

    // Clearing the ramp restores the original footprint exactly.
    project.set_clip_speed_curve(clip, None).unwrap();
    assert_eq!(project.clip(clip).unwrap().timeline, tr(0, 100));
    assert!(!project.clip(clip).unwrap().has_speed_curve());
}

#[test]
fn set_clip_speed_curve_rejects_bad_targets() {
    let (mut project, media_id, track) = project_with_media(500);
    let clip = project
        .add_clip(track, media_id, tr(0, 100), rt(0))
        .unwrap();
    // Out-of-range ramp value.
    let bad = Param::Keyframed {
        keyframes: vec![
            crate::param::Keyframe {
                tick: 0,
                value: 0.0,
                easing: Easing::Linear,
            },
            crate::param::Keyframe {
                tick: 1000,
                value: 1.0,
                easing: Easing::Linear,
            },
        ],
    };
    assert!(project.set_clip_speed_curve(clip, Some(bad)).is_err());
    // Generated clips cannot be retimed.
    let sticker = project.add_track(TrackKind::Sticker, "S");
    let generated = project
        .add_generated(
            sticker,
            Generator::SolidColor {
                rgba: [0, 0, 0, 255],
            },
            tr(200, 10),
        )
        .unwrap();
    assert!(matches!(
        project.set_clip_speed_curve(generated, Some(crate::clip::speed_preset("hero").unwrap())),
        Err(ModelError::InvalidParam(_))
    ));
}

#[test]
fn set_clip_speed_reverse_keeps_duration() {
    let (mut project, media_id, track) = project_with_media(500);
    let clip = project
        .add_clip(track, media_id, tr(0, 100), rt(0))
        .unwrap();

    project
        .set_clip_speed(clip, Rational::new(1, 1), true)
        .unwrap();
    let c = project.clip(clip).unwrap();
    assert_eq!(c.timeline, tr(0, 100));
    assert!(c.reversed && c.is_retimed());
}

#[test]
fn set_clip_speed_rejects_invalid_targets() {
    let (mut project, media_id, track) = project_with_media(500);
    let clip = project
        .add_clip(track, media_id, tr(0, 100), rt(0))
        .unwrap();

    assert!(matches!(
        project.set_clip_speed(clip, Rational::new(0, 1), false),
        Err(ModelError::InvalidParam(_))
    ));
    assert!(matches!(
        project.set_clip_speed(clip, Rational::new(-2, 1), false),
        Err(ModelError::InvalidParam(_))
    ));

    let fx = project.add_track(TrackKind::Adjustment, "FX");
    let generated = project
        .add_generated(fx, Generator::Adjustment, tr(0, 100))
        .unwrap();
    assert!(matches!(
        project.set_clip_speed(generated, Rational::new(2, 1), false),
        Err(ModelError::InvalidParam(_))
    ));

    assert_eq!(
        project.set_clip_speed(ClipId::from_raw(404), Rational::new(2, 1), false),
        Err(ModelError::UnknownClip(ClipId::from_raw(404)))
    );
}

#[test]
fn set_clip_speed_rejects_overlap_with_neighbor() {
    let (mut project, media_id, track) = project_with_media(500);
    let clip = project
        .add_clip(track, media_id, tr(0, 100), rt(0))
        .unwrap();
    // Neighbor right behind the clip: slowing to ½× (200 ticks) collides.
    project
        .add_clip(track, media_id, tr(0, 50), rt(100))
        .unwrap();

    assert_eq!(
        project.set_clip_speed(clip, Rational::new(1, 2), false),
        Err(ModelError::Overlap(track))
    );
    // The clip is untouched after the rejection.
    let c = project.clip(clip).unwrap();
    assert_eq!(c.timeline, tr(0, 100));
    assert_eq!(c.speed, Rational::new(1, 1));
}

// --- set_clip_beats (M8 Phase 6) -----------------------------------------

#[test]
fn set_clip_beats_sorts_dedups_clamps_and_returns_previous() {
    let (mut project, media_id, track) = project_with_media(500);
    let clip = project
        .add_clip(track, media_id, tr(0, 100), rt(0))
        .unwrap();

    // Out-of-window (>=100), duplicate, and unsorted input is normalized.
    let prev = project
        .set_clip_beats(clip, vec![60, 10, 10, 200, 30])
        .unwrap();
    assert!(prev.is_empty(), "no beats before the first detect");
    assert_eq!(project.clip(clip).unwrap().beats, vec![10, 30, 60]);

    // Replacing returns the old list (the inverse snapshot).
    let prev = project.set_clip_beats(clip, vec![5]).unwrap();
    assert_eq!(prev, vec![10, 30, 60]);
    assert_eq!(project.clip(clip).unwrap().beats, vec![5]);
}

#[test]
fn set_clip_beats_rejects_generated_clips() {
    let mut project = Project::new("test", R24);
    let track = project.add_track(TrackKind::Text, "T");
    let clip = project
        .add_generated(track, Generator::text("hi"), tr(0, 48))
        .unwrap();
    assert!(matches!(
        project.set_clip_beats(clip, vec![1, 2]),
        Err(ModelError::InvalidParam(_))
    ));
}

#[test]
fn split_keeps_beats_on_both_halves() {
    let (mut project, media_id, track) = project_with_media(500);
    let clip = project
        .add_clip(track, media_id, tr(0, 100), rt(0))
        .unwrap();
    project.set_clip_beats(clip, vec![10, 60, 90]).unwrap();

    let right = project.split_clip(clip, rt(50)).unwrap();
    // Both halves keep the full source-tick list; each shows only the
    // beats inside its own window via `beat_timeline_ticks`.
    assert_eq!(project.clip(clip).unwrap().beats, vec![10, 60, 90]);
    assert_eq!(project.clip(right).unwrap().beats, vec![10, 60, 90]);
    // Left window [0,50): only the beat at 10 maps in.
    assert_eq!(project.clip(clip).unwrap().beat_timeline_ticks(), vec![10]);
    // Right window [50,100) at timeline [50,100): beats 60, 90 map in.
    assert_eq!(
        project.clip(right).unwrap().beat_timeline_ticks(),
        vec![60, 90]
    );
}

// --- set_clip_audio (M1) -------------------------------------------------

#[test]
fn set_clip_audio_sets_volume_and_fades() {
    let (mut project, media_id, track) = project_with_media(500);
    let clip = project
        .add_clip(track, media_id, tr(0, 100), rt(0))
        .unwrap();

    project
        .set_clip_audio(clip, Some(0.5), rt(10), rt(20))
        .unwrap();
    let c = project.clip(clip).unwrap();
    assert_eq!(c.volume.constant(), Some(0.5));
    assert_eq!((c.fade_in, c.fade_out), (10, 20));
    assert!(c.has_custom_audio());

    // A `None` volume keeps the gain and only moves the fades.
    project.set_clip_audio(clip, None, rt(5), rt(0)).unwrap();
    let c = project.clip(clip).unwrap();
    assert_eq!(c.volume.constant(), Some(0.5));
    assert_eq!((c.fade_in, c.fade_out), (5, 0));

    // Back to defaults clears the custom-audio state.
    project
        .set_clip_audio(clip, Some(1.0), rt(0), rt(0))
        .unwrap();
    assert!(!project.clip(clip).unwrap().has_custom_audio());
}

#[test]
fn set_clip_audio_none_volume_preserves_envelope() {
    let (mut project, media_id, track) = project_with_media(500);
    let clip = project
        .add_clip(track, media_id, tr(0, 100), rt(0))
        .unwrap();
    // A keyframed gain envelope (dips to 0.2 mid-clip).
    project
        .set_param_keyframe(
            clip,
            ClipParam::Volume,
            rt(0),
            ParamValue::Scalar(1.0),
            Easing::Linear,
        )
        .unwrap();
    project
        .set_param_keyframe(
            clip,
            ClipParam::Volume,
            rt(50),
            ParamValue::Scalar(0.2),
            Easing::Linear,
        )
        .unwrap();
    assert!(project.clip(clip).unwrap().has_volume_envelope());

    // Setting only fades (None volume) must not flatten the envelope.
    project.set_clip_audio(clip, None, rt(10), rt(20)).unwrap();
    let c = project.clip(clip).unwrap();
    assert!(c.has_volume_envelope());
    assert_eq!(c.volume.sample(50), 0.2);
    assert_eq!((c.fade_in, c.fade_out), (10, 20));
}

#[test]
fn set_clip_audio_rejects_invalid_inputs() {
    let (mut project, media_id, track) = project_with_media(500);
    let clip = project
        .add_clip(track, media_id, tr(0, 100), rt(0))
        .unwrap();

    for volume in [-0.1, 11.0, f32::NAN, f32::INFINITY] {
        assert!(matches!(
            project.set_clip_audio(clip, Some(volume), rt(0), rt(0)),
            Err(ModelError::InvalidParam(_))
        ));
    }
    // Negative or longer-than-the-clip fades.
    assert!(matches!(
        project.set_clip_audio(clip, Some(1.0), rt(-1), rt(0)),
        Err(ModelError::InvalidParam(_))
    ));
    assert!(matches!(
        project.set_clip_audio(clip, Some(1.0), rt(0), rt(101)),
        Err(ModelError::InvalidParam(_))
    ));

    let fx = project.add_track(TrackKind::Adjustment, "FX");
    let generated = project
        .add_generated(fx, Generator::Adjustment, tr(0, 100))
        .unwrap();
    assert!(matches!(
        project.set_clip_audio(generated, Some(0.5), rt(0), rt(0)),
        Err(ModelError::InvalidParam(_))
    ));

    assert_eq!(
        project.set_clip_audio(ClipId::from_raw(404), Some(0.5), rt(0), rt(0)),
        Err(ModelError::UnknownClip(ClipId::from_raw(404)))
    );
}

#[test]
fn split_keeps_volume_and_partitions_fades() {
    let (mut project, media_id, track) = project_with_media(500);
    let clip = project
        .add_clip(track, media_id, tr(0, 100), rt(0))
        .unwrap();
    project
        .set_clip_audio(clip, Some(0.5), rt(10), rt(20))
        .unwrap();

    let right = project.split_clip(clip, rt(60)).unwrap();
    let left = project.clip(clip).unwrap();
    let right = project.clip(right).unwrap();
    // Volume rides both halves; the fade-in stays with the head, the
    // fade-out moves to the tail.
    assert_eq!(
        (left.volume.constant(), right.volume.constant()),
        (Some(0.5), Some(0.5))
    );
    assert_eq!((left.fade_in, left.fade_out), (10, 0));
    assert_eq!((right.fade_in, right.fade_out), (0, 20));
}

// --- set_clip_crop (M1) ---------------------------------------------------

#[test]
fn set_clip_crop_sets_framing() {
    let (mut project, media_id, track) = project_with_media(500);
    let clip = project
        .add_clip(track, media_id, tr(0, 100), rt(0))
        .unwrap();

    let crop = CropRect {
        x: 0.25,
        y: 0.0,
        w: 0.5,
        h: 1.0,
    };
    project.set_clip_crop(clip, crop, true, false).unwrap();
    let c = project.clip(clip).unwrap();
    assert_eq!(c.crop, crop);
    assert!(c.flip_h && !c.flip_v);
    assert!(c.has_custom_crop());

    // Back to the full frame clears the custom-framing state.
    project
        .set_clip_crop(clip, CropRect::FULL, false, false)
        .unwrap();
    assert!(!project.clip(clip).unwrap().has_custom_crop());
}

#[test]
fn set_clip_crop_rejects_invalid_targets() {
    let (mut project, media_id, track) = project_with_media(500);
    let clip = project
        .add_clip(track, media_id, tr(0, 100), rt(0))
        .unwrap();

    // Invalid rects bounce.
    assert!(matches!(
        project.set_clip_crop(
            clip,
            CropRect {
                x: 0.8,
                y: 0.0,
                w: 0.5,
                h: 1.0
            },
            false,
            false
        ),
        Err(ModelError::InvalidParam(_))
    ));

    // Audio lanes have no frame to crop.
    let audio = project.add_track(TrackKind::Audio, "A1");
    let audio_clip = project.add_clip(audio, media_id, tr(0, 50), rt(0)).unwrap();
    assert!(matches!(
        project.set_clip_crop(audio_clip, CropRect::FULL, true, false),
        Err(ModelError::IncompatibleTrackKind { .. })
    ));

    assert_eq!(
        project.set_clip_crop(ClipId::from_raw(404), CropRect::FULL, false, false),
        Err(ModelError::UnknownClip(ClipId::from_raw(404)))
    );
}

#[test]
fn split_keeps_crop_and_flips_on_both_halves() {
    let (mut project, media_id, track) = project_with_media(500);
    let clip = project
        .add_clip(track, media_id, tr(0, 100), rt(0))
        .unwrap();
    let crop = CropRect {
        x: 0.1,
        y: 0.1,
        w: 0.8,
        h: 0.8,
    };
    project.set_clip_crop(clip, crop, true, true).unwrap();

    let right = project.split_clip(clip, rt(60)).unwrap();
    for id in [clip, right] {
        let c = project.clip(id).unwrap();
        assert_eq!(c.crop, crop);
        assert!(c.flip_h && c.flip_v);
    }
}

#[test]
fn trim_scales_source_consumption_by_speed() {
    let (mut project, media_id, track) = project_with_media(500);
    let clip = project
        .add_clip(track, media_id, tr(100, 100), rt(0))
        .unwrap();
    project
        .set_clip_speed(clip, Rational::new(2, 1), false)
        .unwrap();
    // Now timeline [0, 50) over source [100, 200).

    // Head trim by 10 timeline ticks eats 20 source ticks.
    project.trim_clip(clip, tr(10, 40)).unwrap();
    let c = project.clip(clip).unwrap();
    assert_eq!(c.source_range(), Some(tr(120, 80)));

    // Tail trim to 20 timeline ticks keeps 40 source ticks.
    project.trim_clip(clip, tr(10, 20)).unwrap();
    assert_eq!(
        project.clip(clip).unwrap().source_range(),
        Some(tr(120, 40))
    );
}

#[test]
fn trim_reversed_clip_mirrors_source_window() {
    let (mut project, media_id, track) = project_with_media(500);
    let clip = project
        .add_clip(track, media_id, tr(100, 100), rt(0))
        .unwrap();
    project
        .set_clip_speed(clip, Rational::new(1, 1), true)
        .unwrap();

    // The timeline head shows the source END: a head trim by 10 drops
    // the top 10 source ticks, keeping the bottom.
    project.trim_clip(clip, tr(10, 90)).unwrap();
    assert_eq!(
        project.clip(clip).unwrap().source_range(),
        Some(tr(100, 90))
    );

    // A tail trim drops the source BOTTOM.
    project.trim_clip(clip, tr(10, 80)).unwrap();
    assert_eq!(
        project.clip(clip).unwrap().source_range(),
        Some(tr(110, 80))
    );
}

#[test]
fn split_retimed_clip_inherits_speed_and_partitions_source() {
    let (mut project, media_id, track) = project_with_media(500);
    let clip = project
        .add_clip(track, media_id, tr(0, 100), rt(0))
        .unwrap();
    project
        .set_clip_speed(clip, Rational::new(2, 1), false)
        .unwrap();
    // Timeline [0, 50) over source [0, 100).

    let right = project.split_clip(clip, rt(20)).unwrap();
    let left = project.clip(clip).unwrap();
    let right = project.clip(right).unwrap();
    assert_eq!(left.timeline, tr(0, 20));
    assert_eq!(left.source_range(), Some(tr(0, 40)));
    assert_eq!(right.timeline, tr(20, 30));
    assert_eq!(right.source_range(), Some(tr(40, 60)));
    assert_eq!(right.speed, Rational::new(2, 1));
}

#[test]
fn split_reversed_clip_hands_window_top_to_the_left() {
    let (mut project, media_id, track) = project_with_media(500);
    let clip = project
        .add_clip(track, media_id, tr(100, 100), rt(0))
        .unwrap();
    project
        .set_clip_speed(clip, Rational::new(1, 1), true)
        .unwrap();

    let right = project.split_clip(clip, rt(30)).unwrap();
    let left = project.clip(clip).unwrap();
    let right = project.clip(right).unwrap();
    // Left timeline half plays the source top backward; right half the
    // bottom.
    assert_eq!(left.source_range(), Some(tr(170, 30)));
    assert_eq!(right.source_range(), Some(tr(100, 70)));
    assert!(right.reversed);
}

// --- move_clip --------------------------------------------------------

#[test]
fn move_clip_repositions_on_same_track() {
    let (mut project, media_id, track) = project_with_media(200);
    let clip = project.add_clip(track, media_id, tr(0, 50), rt(0)).unwrap();

    project.move_clip(clip, track, rt(80)).unwrap();
    assert_eq!(project.clip(clip).unwrap().timeline, tr(80, 50));
    assert_eq!(project.timeline().track_of(clip), Some(track));
}

#[test]
fn move_clip_rejects_negative_start_and_unknown_track() {
    let (mut project, media_id, track) = project_with_media(200);
    let clip = project.add_clip(track, media_id, tr(0, 50), rt(0)).unwrap();
    let missing = TrackId::from_raw(77);

    assert_eq!(
        project.move_clip(clip, track, rt(-1)),
        Err(ModelError::InvalidRange)
    );
    assert_eq!(
        project.move_clip(clip, missing, rt(0)),
        Err(ModelError::UnknownTrack(missing))
    );
}

// --- shift_clips ------------------------------------------------------

fn packed_track() -> (Project, MediaId, TrackId, [ClipId; 3]) {
    let (mut project, media_id, track) = project_with_media(1000);
    let a = project.add_clip(track, media_id, tr(0, 50), rt(0)).unwrap();
    let b = project
        .add_clip(track, media_id, tr(50, 50), rt(50))
        .unwrap();
    let c = project
        .add_clip(track, media_id, tr(100, 50), rt(100))
        .unwrap();
    (project, media_id, track, [a, b, c])
}

#[test]
fn shift_clips_right_moves_suffix_only() {
    let (mut project, _, track, [a, b, c]) = packed_track();
    let first = project.shift_clips(track, rt(50), rt(30)).unwrap();
    assert_eq!(first, Some(rt(80)));
    assert_eq!(project.clip(a).unwrap().start().value, 0);
    assert_eq!(project.clip(b).unwrap().start().value, 80);
    assert_eq!(project.clip(c).unwrap().start().value, 130);
}

#[test]
fn shift_clips_left_closes_gap() {
    let (mut project, _, track, [a, b, c]) = packed_track();
    project.shift_clips(track, rt(50), rt(30)).unwrap();
    let first = project.shift_clips(track, rt(80), rt(-30)).unwrap();
    assert_eq!(first, Some(rt(50)));
    assert_eq!(project.clip(a).unwrap().start().value, 0);
    assert_eq!(project.clip(b).unwrap().start().value, 50);
    assert_eq!(project.clip(c).unwrap().start().value, 100);
}

#[test]
fn shift_clips_left_rejects_collision_and_negative() {
    let (mut project, _, track, [_, b, _]) = packed_track();
    assert_eq!(
        project.shift_clips(track, rt(50), rt(-10)),
        Err(ModelError::Overlap(track))
    );
    assert_eq!(
        project.shift_clips(track, rt(0), rt(-1)),
        Err(ModelError::InvalidRange)
    );
    // Validation failures must not mutate anything.
    assert_eq!(project.clip(b).unwrap().start().value, 50);
}

#[test]
fn shift_clips_past_content_is_noop() {
    let (mut project, _, track, [a, b, c]) = packed_track();
    assert_eq!(project.shift_clips(track, rt(999), rt(40)).unwrap(), None);
    for (clip, start) in [(a, 0), (b, 50), (c, 100)] {
        assert_eq!(project.clip(clip).unwrap().start().value, start);
    }
}

#[test]
fn shift_clips_rejects_zero_delta_and_unknown_track() {
    let (mut project, _, track, _) = packed_track();
    assert_eq!(
        project.shift_clips(track, rt(0), rt(0)),
        Err(ModelError::InvalidRange)
    );
    let missing = TrackId::from_raw(404);
    assert_eq!(
        project.shift_clips(missing, rt(0), rt(10)),
        Err(ModelError::UnknownTrack(missing))
    );
}

// --- ripple_delete ----------------------------------------------------

#[test]
fn ripple_delete_shifts_later_clips() {
    let (mut project, media_id, track) = project_with_media(300);
    let a = project.add_clip(track, media_id, tr(0, 50), rt(0)).unwrap();
    let b = project
        .add_clip(track, media_id, tr(50, 50), rt(50))
        .unwrap();
    let c = project
        .add_clip(track, media_id, tr(100, 50), rt(150))
        .unwrap();

    project.ripple_delete(b).unwrap();
    assert!(project.clip(b).is_none());
    assert_eq!(project.clip(a).unwrap().start().value, 0);
    assert_eq!(project.clip(c).unwrap().start().value, 100);
}

// --- timeline accessors -----------------------------------------------

#[test]
fn timeline_mut_allows_direct_timeline_edits() {
    let mut project = Project::new("test", R24);
    let track = project.add_track(TrackKind::Audio, "A1");
    assert_eq!(project.timeline_mut().track_count(), 1);
    assert_eq!(
        project.timeline().track(track).unwrap().kind,
        TrackKind::Audio
    );
}

// --- Clone ------------------------------------------------------------

#[test]
fn project_clone_is_independent_snapshot() {
    let (mut project, media_id, track) = project_with_media(100);
    let clip = project.add_clip(track, media_id, tr(0, 50), rt(0)).unwrap();

    let mut cloned = project.clone();
    assert_eq!(cloned.clip(clip).unwrap().timeline, tr(0, 50));

    cloned.remove_clip(clip);
    assert!(cloned.clip(clip).is_none());
    assert!(project.clip(clip).is_some());
}

// --- Phase I look properties --------------------------------------------

use crate::look::{
    AnimationRef, AnimationSlot, AudioRole, ChromaKey, ColorAdjustments, Filter, Mask, MaskKind,
    StabilizeLevel,
};

#[test]
fn look_setters_persist_on_a_media_clip() {
    let (mut project, media_id, track) = project_with_media(100);
    let clip = project.add_clip(track, media_id, tr(0, 50), rt(0)).unwrap();

    project
        .set_clip_mask(clip, Some(Mask::new(MaskKind::Circle)))
        .unwrap();
    project
        .set_clip_chroma_key(
            clip,
            Some(ChromaKey {
                rgb: [0, 255, 0],
                strength: 0.7.into(),
                shadow: 0.2.into(),
            }),
        )
        .unwrap();
    project
        .set_clip_stabilize(clip, Some(StabilizeLevel::Smooth))
        .unwrap();
    project
        .set_clip_filter(clip, Some(Filter::new("vivid")))
        .unwrap();
    project
        .set_clip_adjustments(
            clip,
            ColorAdjustments {
                brightness: 0.25.into(),
                ..Default::default()
            },
        )
        .unwrap();

    let c = project.clip(clip).unwrap();
    assert_eq!(c.mask.as_ref().unwrap().kind, MaskKind::Circle);
    assert_eq!(c.chroma_key.as_ref().unwrap().strength, 0.7.into());
    assert_eq!(c.stabilize, Some(StabilizeLevel::Smooth));
    assert_eq!(c.filter.as_ref().unwrap().id, "vivid");
    assert_eq!(c.adjust.brightness, 0.25.into());

    // Clearing works and the fields vanish from saves again.
    project.set_clip_mask(clip, None).unwrap();
    assert!(project.clip(clip).unwrap().mask.is_none());
}

#[test]
fn filter_intensity_keyframes_sample_at_clip_local_ticks() {
    let (mut project, media_id, track) = project_with_media(100);
    let clip = project.add_clip(track, media_id, tr(0, 50), rt(0)).unwrap();
    project
        .set_clip_filter(clip, Some(Filter::new("vivid")))
        .unwrap();
    let param = ClipParam::Look {
        param: crate::LookParam::FilterIntensity,
    };
    project
        .set_param_keyframe(clip, param, rt(0), ParamValue::Scalar(0.0), Easing::Linear)
        .unwrap();
    project
        .set_param_keyframe(clip, param, rt(10), ParamValue::Scalar(1.0), Easing::Linear)
        .unwrap();

    let intensity = project
        .clip(clip)
        .unwrap()
        .filter
        .as_ref()
        .unwrap()
        .intensity
        .sample(5);
    assert!((intensity - 0.5).abs() < f32::EPSILON);
}

#[test]
fn look_setters_reject_wrong_content() {
    let (mut project, media_id, _) = project_with_media(100);

    // Generated clips have no source pixels to mask/key/stabilize.
    let fx = project.add_track(TrackKind::Adjustment, "FX");
    let bar = project
        .add_generated(fx, Generator::Adjustment, tr(0, 24))
        .unwrap();
    assert!(
        project
            .set_clip_mask(bar, Some(Mask::new(MaskKind::Linear)))
            .is_err()
    );
    assert!(
        project
            .set_clip_stabilize(bar, Some(StabilizeLevel::Recommended))
            .is_err()
    );
    // …but filters and adjustments are exactly what lane bars persist.
    assert!(
        project
            .set_clip_filter(bar, Some(Filter::new("noir")))
            .is_ok()
    );
    assert!(
        project
            .set_clip_adjustments(
                bar,
                ColorAdjustments {
                    contrast: (-0.5).into(),
                    ..Default::default()
                }
            )
            .is_ok()
    );

    // Audio-lane clips show no pixels.
    let audio = project.add_track(TrackKind::Audio, "A1");
    let aclip = project.add_clip(audio, media_id, tr(0, 50), rt(0)).unwrap();
    assert!(
        project
            .set_clip_filter(aclip, Some(Filter::new("warm")))
            .is_err()
    );

    // Stills have no motion to stabilize.
    let image = project.add_media(MediaSource::image("/tmp/photo.png", 800, 600));
    let video = project.add_track(TrackKind::Video, "V2");
    let still = project
        .add_clip(
            video,
            image,
            tr_at(0, 100, crate::media::STILL_TICK_RATE),
            rt(0),
        )
        .unwrap();
    assert!(
        project
            .set_clip_stabilize(still, Some(StabilizeLevel::Smooth))
            .is_err()
    );
    assert!(
        project
            .set_clip_mask(still, Some(Mask::new(MaskKind::Heart)))
            .is_ok(),
        "masks apply to stills"
    );

    // Unknown ids and out-of-range values are rejected.
    let (mut p2, m2, t2) = project_with_media(100);
    let clip = p2.add_clip(t2, m2, tr(0, 50), rt(0)).unwrap();
    assert!(p2.set_clip_filter(clip, Some(Filter::new("nope"))).is_err());
    let mut hot = Filter::new("vivid");
    hot.intensity = 1.5.into();
    assert!(p2.set_clip_filter(clip, Some(hot)).is_err());
}

#[test]
fn animation_slots_enforce_capcut_exclusivity() {
    let (mut project, media_id, track) = project_with_media(100);
    let clip = project.add_clip(track, media_id, tr(0, 50), rt(0)).unwrap();

    project
        .set_clip_animation(clip, AnimationSlot::In, Some(AnimationRef::new("fade_in")))
        .unwrap();
    project
        .set_clip_animation(clip, AnimationSlot::Out, Some(AnimationRef::new("drop")))
        .unwrap();
    let c = project.clip(clip).unwrap();
    assert_eq!(c.animation_in.as_ref().unwrap().id, "fade_in");
    assert_eq!(c.animation_out.as_ref().unwrap().id, "drop");

    // A combo replaces both sides…
    project
        .set_clip_animation(clip, AnimationSlot::Combo, Some(AnimationRef::new("pulse")))
        .unwrap();
    let c = project.clip(clip).unwrap();
    assert!(c.animation_in.is_none() && c.animation_out.is_none());
    assert_eq!(c.animation_combo.as_ref().unwrap().id, "pulse");

    // …and an entrance evicts the combo again.
    project
        .set_clip_animation(clip, AnimationSlot::In, Some(AnimationRef::new("zoom_in")))
        .unwrap();
    let c = project.clip(clip).unwrap();
    assert!(c.animation_combo.is_none());

    // Wrong slot, unknown id, and text-only presets on media are rejected.
    assert!(
        project
            .set_clip_animation(clip, AnimationSlot::Out, Some(AnimationRef::new("fade_in")))
            .is_err()
    );
    assert!(
        project
            .set_clip_animation(clip, AnimationSlot::In, Some(AnimationRef::new("nope")))
            .is_err()
    );
    assert!(
        project
            .set_clip_animation(
                clip,
                AnimationSlot::Combo,
                Some(AnimationRef::new("typewriter"))
            )
            .is_err()
    );

    // Text clips accept the text-only presets.
    let text = project.add_track(TrackKind::Text, "T1");
    let title = project
        .add_generated(text, Generator::text("Hi"), tr(0, 24))
        .unwrap();
    assert!(
        project
            .set_clip_animation(
                title,
                AnimationSlot::Combo,
                Some(AnimationRef::new("typewriter"))
            )
            .is_ok()
    );
}

#[test]
fn audio_role_tags_audio_lane_clips_only() {
    let (mut project, media_id, video) = project_with_media(100);
    let vclip = project.add_clip(video, media_id, tr(0, 50), rt(0)).unwrap();
    assert!(
        project
            .set_clip_audio_role(vclip, Some(AudioRole::Music))
            .is_err()
    );

    let audio = project.add_track(TrackKind::Audio, "A1");
    let aclip = project.add_clip(audio, media_id, tr(0, 50), rt(0)).unwrap();
    project
        .set_clip_audio_role(aclip, Some(AudioRole::Voiceover))
        .unwrap();
    assert_eq!(
        project.clip(aclip).unwrap().audio_role,
        Some(AudioRole::Voiceover)
    );
    project.set_clip_audio_role(aclip, None).unwrap();
    assert!(project.clip(aclip).unwrap().audio_role.is_none());
}

#[test]
fn text_effect_presets_bake_onto_the_style() {
    let mut project = Project::new("t", R24);
    let track = project.add_track(TrackKind::Text, "T1");

    let style = crate::TextStyle {
        effect_preset: Some("neon".into()),
        ..Default::default()
    };
    let clip = project
        .add_generated(
            track,
            Generator::Text {
                content: "Glow".into(),
                style,
            },
            tr(0, 24),
        )
        .unwrap();

    let ClipSource::Generated(Generator::Text { style, .. }) = &project.clip(clip).unwrap().content
    else {
        panic!("expected text");
    };
    assert_eq!(style.effect_preset.as_deref(), Some("neon"));
    let spec = crate::look::text_effect_spec("neon").unwrap();
    assert_eq!(style.stroke, spec.stroke);
    assert_eq!(style.shadow, spec.shadow);
    assert_eq!(style.background, spec.background);

    // Unknown presets are rejected at the door.
    let bad = crate::TextStyle {
        effect_preset: Some("nope".into()),
        ..Default::default()
    };
    assert!(
        project
            .set_generator(
                clip,
                Generator::Text {
                    content: "Glow".into(),
                    style: bad,
                }
            )
            .is_err()
    );

    // Clearing the preset keeps whatever treatments the style carries.
    let manual = crate::TextStyle {
        effect_preset: None,
        stroke: Some(crate::TextStroke {
            rgba: [1, 2, 3, 255].into(),
            width: 2.0.into(),
        }),
        ..Default::default()
    };
    project
        .set_generator(
            clip,
            Generator::Text {
                content: "Glow".into(),
                style: manual.clone(),
            },
        )
        .unwrap();
    let ClipSource::Generated(Generator::Text { style, .. }) = &project.clip(clip).unwrap().content
    else {
        panic!("expected text");
    };
    assert_eq!(style.stroke, manual.stroke);
}

#[test]
fn look_fields_roundtrip_and_stay_off_the_wire_when_unset() {
    let (mut project, media_id, track) = project_with_media(100);
    let plain = project.add_clip(track, media_id, tr(0, 50), rt(0)).unwrap();
    let styled = project
        .add_clip(track, media_id, tr(50, 50), rt(50))
        .unwrap();
    project
        .set_clip_mask(styled, Some(Mask::new(MaskKind::Star)))
        .unwrap();
    project
        .set_clip_filter(styled, Some(Filter::new("sunset")))
        .unwrap();

    let plain_json = serde_json::to_value(project.clip(plain).unwrap()).unwrap();
    for key in [
        "mask",
        "chroma_key",
        "stabilize",
        "filter",
        "adjust",
        "audio_role",
    ] {
        assert!(
            plain_json.get(key).is_none(),
            "untouched clip should not serialize `{key}`"
        );
    }
    let styled_json = serde_json::to_value(project.clip(styled).unwrap()).unwrap();
    assert_eq!(styled_json["mask"]["kind"], "star");
    assert_eq!(styled_json["filter"]["id"], "sunset");

    // The whole project (with looks) survives a save/load roundtrip.
    let json = serde_json::to_value(&project).unwrap();
    let back: Project = serde_json::from_value(json).unwrap();
    assert_eq!(
        back.clip(styled).unwrap().mask,
        project.clip(styled).unwrap().mask
    );
    assert_eq!(
        back.clip(styled).unwrap().filter,
        project.clip(styled).unwrap().filter
    );
    assert!(back.clip(plain).unwrap().mask.is_none());
}

#[test]
fn speed_preset_catalog_ids_resolve_and_match_back() {
    for spec in crate::speed_preset_catalog() {
        let curve = crate::speed_preset(spec.id)
            .unwrap_or_else(|| panic!("catalog id '{}' has no curve", spec.id));
        crate::validate_speed_curve(&curve)
            .unwrap_or_else(|e| panic!("preset '{}' curve invalid: {e}", spec.id));
        assert_eq!(crate::speed_preset_id(&curve), Some(spec.id));
    }
    assert_eq!(crate::speed_preset_id(&Param::Constant(1.0)), None);
}
