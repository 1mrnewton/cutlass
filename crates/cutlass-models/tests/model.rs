//! Integration tests for project-level invariants and cross-module behavior.

mod common;

use common::{rt, sample_media, tr, tr_at, FPS_24, FPS_30};
use cutlass_models::{
    ClipTransform, Generator, MediaSource, ModelError, Project, Shape, TrackKind,
};

#[test]
fn build_project_and_query_by_id() {
    let mut project = Project::new("demo", FPS_24);

    let media = sample_media(FPS_24, 1000);
    let media_id = project.add_media(media);

    let v1 = project.add_track(TrackKind::Video, "V1");

    let c1 = project
        .add_clip(v1, media_id, tr(0, 100), rt(0))
        .expect("first clip");
    let c2 = project
        .add_clip(v1, media_id, tr(200, 100), rt(100))
        .expect("second clip");

    assert_eq!(
        project.clip(c1).unwrap().source_range(),
        Some(tr(0, 100))
    );
    assert_eq!(project.clip(c1).unwrap().media(), Some(media_id));
    assert_eq!(project.clip(c2).unwrap().start().value, 100);
    assert_eq!(project.timeline().track_of(c1), Some(v1));

    assert_eq!(project.timeline().duration().value, 200);
    assert_eq!(project.timeline().clip_count(), 2);
}

#[test]
fn generated_clips_need_no_media() {
    let mut project = Project::new("demo", FPS_24);
    let title = project.add_track(TrackKind::Text, "Titles");
    let gfx = project.add_track(TrackKind::Sticker, "GFX");

    let text = project
        .add_generated(
            title,
            Generator::text("Hello"),
            tr(0, 48),
        )
        .unwrap();
    let shape = project
        .add_generated(
            gfx,
            Generator::shape(Shape::Rectangle, [255, 255, 255, 255]),
            tr(48, 48),
        )
        .unwrap();

    assert_eq!(project.clip(text).unwrap().media(), None);
    assert!(project.clip(text).unwrap().is_generated());
    assert_eq!(project.clip(shape).unwrap().source_range(), None);
    assert_eq!(project.media_count(), 0);
    assert_eq!(project.timeline().duration().value, 96);
}

#[test]
fn set_transform_updates_visual_and_rejects_audio() {
    let mut project = Project::new("demo", FPS_24);
    let media_id = project.add_media(sample_media(FPS_24, 1000));
    let v1 = project.add_track(TrackKind::Video, "V1");
    let a1 = project.add_track(TrackKind::Audio, "A1");

    let video = project.add_clip(v1, media_id, tr(0, 48), rt(0)).unwrap();
    let audio = project.add_clip(a1, media_id, tr(0, 48), rt(0)).unwrap();

    // New clips start at identity (aspect-fit, centered).
    assert!(project.clip(video).unwrap().transform.is_identity());

    let t = ClipTransform {
        position: [0.1, 0.2],
        scale: 2.0,
        rotation: -30.0,
        opacity: 0.5,
        ..ClipTransform::IDENTITY
    };
    project.set_transform(video, t, None).expect("set transform");
    assert_eq!(project.clip(video).unwrap().transform, t.into());

    // Audio clips have nothing to place on the canvas.
    assert!(matches!(
        project.set_transform(audio, t, None),
        Err(ModelError::IncompatibleTrackKind { .. })
    ));

    // Invalid values are rejected and leave the stored transform unchanged.
    for bad in [
        ClipTransform {
            scale: 0.0,
            ..ClipTransform::IDENTITY
        },
        ClipTransform {
            opacity: 1.5,
            ..ClipTransform::IDENTITY
        },
        ClipTransform {
            position: [f32::NAN, 0.0],
            ..ClipTransform::IDENTITY
        },
        ClipTransform {
            rotation: f32::INFINITY,
            ..ClipTransform::IDENTITY
        },
    ] {
        assert!(matches!(
            project.set_transform(video, bad, None),
            Err(ModelError::InvalidTransform(_))
        ));
    }
    assert_eq!(project.clip(video).unwrap().transform, t.into());
}

#[test]
fn overlap_is_rejected() {
    let mut project = Project::new("demo", FPS_24);
    let media_id = project.add_media(sample_media(FPS_24, 1000));
    let v1 = project.add_track(TrackKind::Video, "V1");

    project.add_clip(v1, media_id, tr(0, 100), rt(0)).unwrap();
    let err = project
        .add_clip(v1, media_id, tr(0, 100), rt(50))
        .unwrap_err();
    assert_eq!(err, ModelError::Overlap(v1));
}

#[test]
fn unknown_refs_error() {
    let mut project = Project::new("demo", FPS_24);
    let v1 = project.add_track(TrackKind::Video, "V1");
    let media_id = project.add_media(sample_media(FPS_24, 1000));

    let bad_media = MediaSource::new("/x", 1, 1, FPS_24, 10, false).id;
    assert!(matches!(
        project.add_clip(v1, bad_media, tr(0, 10), rt(0)),
        Err(ModelError::UnknownMedia(_))
    ));

    assert_eq!(
        project.add_clip(v1, media_id, tr(900, 200), rt(0)),
        Err(ModelError::SourceOutOfBounds)
    );
}

#[test]
fn rate_conform_adjusts_timeline_duration() {
    let mut project = Project::new("demo", FPS_24);
    let media_id = project.add_media(sample_media(FPS_30, 1000));
    let v1 = project.add_track(TrackKind::Video, "V1");

    let clip_id = project
        .add_clip(
            v1,
            media_id,
            tr_at(0, 120, FPS_30),
            rt(0),
        )
        .unwrap();
    assert_eq!(project.clip(clip_id).unwrap().timeline.duration.value, 96);
}

#[test]
fn removing_referenced_media_fails_then_succeeds() {
    let mut project = Project::new("demo", FPS_24);
    let media_id = project.add_media(sample_media(FPS_24, 1000));
    let v1 = project.add_track(TrackKind::Video, "V1");
    let clip_id = project
        .add_clip(v1, media_id, tr(0, 100), rt(0))
        .unwrap();

    assert_eq!(
        project.remove_media(media_id),
        Err(ModelError::MediaReferenced(media_id))
    );

    project.timeline_mut().remove_clip(clip_id).unwrap();
    assert!(project.remove_media(media_id).is_ok());
    assert_eq!(project.media_count(), 0);
}

#[test]
fn audio_tracks_stack_below_visual_lanes() {
    let mut project = Project::new("demo", FPS_24);
    let v1 = project.add_track(TrackKind::Video, "V1");
    let v2 = project.add_track(TrackKind::Video, "V2");
    let a1 = project.add_track(TrackKind::Audio, "A1");

    // Visual lanes keep their insertion order; audio sinks to the bottom of the
    // stack (index 0) so it always renders at the bottom of the timeline.
    assert_eq!(project.timeline().order(), &[a1, v1, v2]);
    let names: Vec<&str> = project
        .timeline()
        .tracks_ordered()
        .map(|t| t.name.as_str())
        .collect();
    assert_eq!(names, ["A1", "V1", "V2"]);
}

#[test]
fn clip_at_and_source_mapping() {
    let mut project = Project::new("demo", FPS_24);
    let media_id = project.add_media(sample_media(FPS_24, 1000));
    let v1 = project.add_track(TrackKind::Video, "V1");
    let id = project
        .add_clip(v1, media_id, tr(100, 10), rt(10))
        .unwrap();

    let track = project.timeline().track(v1).unwrap();
    assert_eq!(
        track.clip_at(rt(15)).unwrap().map(|c| c.id),
        Some(id)
    );
    assert!(track.clip_at(rt(25)).unwrap().is_none());
    assert_eq!(
        project
            .clip(id)
            .unwrap()
            .source_time_at(rt(15))
            .unwrap()
            .map(|t| t.value),
        Some(105)
    );
}

#[test]
fn split_media_clip_divides_timeline_and_source() {
    let mut project = Project::new("demo", FPS_24);
    let media_id = project.add_media(sample_media(FPS_24, 1000));
    let v1 = project.add_track(TrackKind::Video, "V1");
    let left = project
        .add_clip(v1, media_id, tr(100, 100), rt(0))
        .unwrap();

    let right = project.split_clip(left, rt(40)).expect("split inside the clip");
    assert_ne!(left, right);

    let l = project.clip(left).unwrap();
    assert_eq!(l.timeline, tr(0, 40));
    assert_eq!(l.source_range(), Some(tr(100, 40)));
    let r = project.clip(right).unwrap();
    assert_eq!(r.timeline, tr(40, 60));
    assert_eq!(r.source_range(), Some(tr(140, 60)));
    assert_eq!(project.timeline().duration().value, 100);
    assert_eq!(project.timeline().clip_count(), 2);
}

#[test]
fn split_at_or_outside_boundary_is_rejected() {
    let mut project = Project::new("demo", FPS_24);
    let media_id = project.add_media(sample_media(FPS_24, 1000));
    let v1 = project.add_track(TrackKind::Video, "V1");
    let clip = project
        .add_clip(v1, media_id, tr(0, 100), rt(10))
        .unwrap();

    assert_eq!(project.split_clip(clip, rt(10)), Err(ModelError::InvalidRange));
    assert_eq!(project.split_clip(clip, rt(110)), Err(ModelError::InvalidRange));
    assert_eq!(project.split_clip(clip, rt(200)), Err(ModelError::InvalidRange));
}

#[test]
fn trim_head_advances_source_in_point() {
    let mut project = Project::new("demo", FPS_24);
    let media_id = project.add_media(sample_media(FPS_24, 1000));
    let v1 = project.add_track(TrackKind::Video, "V1");
    let clip = project
        .add_clip(v1, media_id, tr(100, 100), rt(0))
        .unwrap();

    project
        .trim_clip(clip, tr(30, 70))
        .expect("head trim within bounds");
    let c = project.clip(clip).unwrap();
    assert_eq!(c.timeline, tr(30, 70));
    assert_eq!(c.source_range(), Some(tr(130, 70)));
}

#[test]
fn trim_past_source_bounds_is_rejected() {
    let mut project = Project::new("demo", FPS_24);
    let media_id = project.add_media(sample_media(FPS_24, 100));
    let v1 = project.add_track(TrackKind::Video, "V1");
    let clip = project
        .add_clip(v1, media_id, tr(90, 10), rt(0))
        .unwrap();

    assert_eq!(
        project.trim_clip(clip, tr(0, 40)),
        Err(ModelError::SourceOutOfBounds)
    );
}

#[test]
fn trim_into_neighbour_is_rejected() {
    let mut project = Project::new("demo", FPS_24);
    let media_id = project.add_media(sample_media(FPS_24, 1000));
    let v1 = project.add_track(TrackKind::Video, "V1");
    let a = project
        .add_clip(v1, media_id, tr(0, 100), rt(0))
        .unwrap();
    project
        .add_clip(v1, media_id, tr(0, 100), rt(100))
        .unwrap();

    assert_eq!(
        project.trim_clip(a, tr(0, 150)),
        Err(ModelError::Overlap(v1))
    );
}

#[test]
fn move_clip_across_tracks_and_rejects_overlap() {
    let mut project = Project::new("demo", FPS_24);
    let media_id = project.add_media(sample_media(FPS_24, 1000));
    let v1 = project.add_track(TrackKind::Video, "V1");
    let v2 = project.add_track(TrackKind::Video, "V2");
    let clip = project
        .add_clip(v1, media_id, tr(0, 100), rt(0))
        .unwrap();
    project
        .add_clip(v2, media_id, tr(0, 100), rt(0))
        .unwrap();

    assert_eq!(project.move_clip(clip, v2, rt(0)), Err(ModelError::Overlap(v2)));
    assert_eq!(project.timeline().track_of(clip), Some(v1));

    project.move_clip(clip, v2, rt(200)).unwrap();
    assert_eq!(project.timeline().track_of(clip), Some(v2));
    assert_eq!(project.clip(clip).unwrap().timeline, tr(200, 100));
}

#[test]
fn move_clip_rejects_incompatible_track_kind() {
    let mut project = Project::new("demo", FPS_24);
    let media_id = project.add_media(sample_media(FPS_24, 1000));
    let video = project.add_track(TrackKind::Video, "V1");
    let overlay = project.add_track(TrackKind::Text, "T1");
    let clip = project
        .add_clip(video, media_id, tr(0, 100), rt(0))
        .unwrap();

    assert_eq!(
        project.move_clip(clip, overlay, rt(0)),
        Err(ModelError::IncompatibleTrackKind {
            track: overlay,
            kind: TrackKind::Text,
        })
    );
}

#[test]
fn generated_clip_rejects_wrong_track_kind() {
    let mut project = Project::new("demo", FPS_24);
    let video = project.add_track(TrackKind::Video, "V1");

    assert_eq!(
        project.add_generated(
            video,
            Generator::text("nope"),
            tr(0, 24),
        ),
        Err(ModelError::IncompatibleTrackKind {
            track: video,
            kind: TrackKind::Video,
        })
    );
}

#[test]
fn ripple_delete_closes_the_gap() {
    let mut project = Project::new("demo", FPS_24);
    let media_id = project.add_media(sample_media(FPS_24, 1000));
    let v1 = project.add_track(TrackKind::Video, "V1");
    let a = project
        .add_clip(v1, media_id, tr(0, 100), rt(0))
        .unwrap();
    let b = project
        .add_clip(v1, media_id, tr(100, 100), rt(100))
        .unwrap();
    let c = project
        .add_clip(v1, media_id, tr(200, 100), rt(200))
        .unwrap();

    project.ripple_delete(b).unwrap();
    assert!(project.clip(b).is_none());
    assert_eq!(project.clip(a).unwrap().start().value, 0);
    assert_eq!(project.clip(c).unwrap().start().value, 100);
    assert_eq!(project.timeline().duration().value, 200);
}

#[test]
fn editing_unknown_clip_errors() {
    let mut project = Project::new("demo", FPS_24);
    let media_id = project.add_media(sample_media(FPS_24, 1000));
    let v1 = project.add_track(TrackKind::Video, "V1");
    let clip = project
        .add_clip(v1, media_id, tr(0, 100), rt(0))
        .unwrap();
    let gone = project.ripple_delete(clip).unwrap().id;

    assert!(matches!(
        project.split_clip(gone, rt(5)),
        Err(ModelError::UnknownClip(_))
    ));
    assert!(matches!(
        project.trim_clip(gone, tr(0, 10)),
        Err(ModelError::UnknownClip(_))
    ));
    assert!(matches!(
        project.move_clip(gone, v1, rt(0)),
        Err(ModelError::UnknownClip(_))
    ));
}

#[test]
fn media_pool_lookup_and_iteration() {
    let mut project = Project::new("pool", FPS_24);
    let a = project.add_media(sample_media(FPS_24, 100));
    let b = project.add_media(sample_media(FPS_30, 200));

    assert_eq!(project.media(a).unwrap().duration.value, 100);
    assert_eq!(project.media(b).unwrap().frame_rate, FPS_30);

    let mut durations: Vec<i64> = project.media_iter().map(|m| m.duration.value).collect();
    durations.sort();
    assert_eq!(durations, vec![100, 200]);
}

#[test]
fn resample_preserved_across_split_trim_chain() {
    let mut project = Project::new("chain", FPS_24);
    let media_id = project.add_media(sample_media(FPS_30, 600));
    let track = project.add_track(TrackKind::Video, "V1");

    let clip = project
        .add_clip(track, media_id, tr_at(0, 300, FPS_30), rt(0))
        .unwrap();
    assert_eq!(project.clip(clip).unwrap().timeline.duration.value, 240);

    let right = project.split_clip(clip, rt(120)).unwrap();
    project.trim_clip(right, tr(120, 96)).unwrap();

    assert_eq!(project.clip(clip).unwrap().timeline, tr(0, 120));
    assert_eq!(project.clip(right).unwrap().timeline, tr(120, 96));
    assert_eq!(project.timeline().duration().value, 216);
}

#[test]
fn audio_and_video_tracks_hold_independent_clips() {
    let mut project = Project::new("av", FPS_24);
    let media_id = project.add_media(sample_media(FPS_24, 500));
    let video = project.add_track(TrackKind::Video, "V1");
    let audio = project.add_track(TrackKind::Audio, "A1");

    let picture = project
        .add_clip(video, media_id, tr(0, 200), rt(0))
        .unwrap();
    let sound = project
        .add_clip(audio, media_id, tr(0, 200), rt(0))
        .unwrap();

    assert_ne!(picture, sound);
    assert_eq!(project.timeline().track_of(picture), Some(video));
    assert_eq!(project.timeline().track_of(sound), Some(audio));
    assert_eq!(project.timeline().clip_count(), 2);
}
