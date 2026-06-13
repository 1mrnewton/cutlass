//! End-to-end editing workflows across project, timeline, track, clip, and time.

mod common;

use common::{
    project_with_media, rt, rt_at, sample_media, tr, tr_at, FPS_23_976, FPS_24, FPS_30,
};
use cutlass_models::{
    Generator, ModelError, Project, Shape, TimeRange, TrackKind,
};

#[test]
fn realistic_edit_session() {
    let mut project = Project::new("documentary", FPS_24);

    let interview = project.add_media(sample_media(FPS_24, 2_400));
    let broll = project.add_media(sample_media(FPS_30, 900));
    let v1 = project.add_track(TrackKind::Video, "A-roll");
    let v2 = project.add_track(TrackKind::Video, "B-roll");

    let intro = project
        .add_clip(v1, interview, tr(0, 240), rt(0))
        .unwrap();
    let body = project
        .add_clip(v1, interview, tr(240, 480), rt(240))
        .unwrap();
    project
        .add_clip(v2, broll, tr_at(0, 120, FPS_30), rt(0))
        .unwrap();

    // Split the body clip and ripple-delete the tail segment.
    let tail = project.split_clip(body, rt(360)).unwrap();
    project.ripple_delete(tail).unwrap();

    // Trim intro head, move surviving body earlier to close the ripple gap.
    project.trim_clip(intro, tr(24, 216)).unwrap();
    project.move_clip(body, v1, rt(240)).unwrap();

    assert_eq!(project.timeline().clip_count(), 3);
    assert_eq!(project.clip(intro).unwrap().timeline, tr(24, 216));
    assert_eq!(project.clip(body).unwrap().timeline, tr(240, 120));
    assert_eq!(project.timeline().track_of(body), Some(v1));
    assert!(project.clip(tail).is_none());
    assert_eq!(project.media_count(), 2);
    assert_eq!(project.timeline().duration().value, 360);
}

#[test]
fn remove_clip_leaves_gap_ripple_closes_it() {
    let (mut project, media_id, track) = project_with_media(500);

    let a = project.add_clip(track, media_id, tr(0, 100), rt(0)).unwrap();
    let b = project.add_clip(track, media_id, tr(100, 100), rt(100)).unwrap();
    let c = project.add_clip(track, media_id, tr(200, 100), rt(200)).unwrap();

    project.remove_clip(b).unwrap();
    assert_eq!(project.clip(c).unwrap().start().value, 200);
    assert_eq!(project.timeline().duration().value, 300);

    project.ripple_delete(a).unwrap();
    assert_eq!(project.clip(c).unwrap().start().value, 100);
    assert_eq!(project.timeline().duration().value, 200);
}

#[test]
fn ntsc_media_on_film_timeline_split_and_trim() {
    let mut project = Project::new("ntsc", FPS_24);
    let media_id = project.add_media(sample_media(FPS_23_976, 2_402));
    let track = project.add_track(TrackKind::Video, "V1");

    // 1001 source ticks (~41.7s) -> 1002 timeline ticks at 24fps.
    let clip = project
        .add_clip(
            track,
            media_id,
            tr_at(0, 1_001, FPS_23_976),
            rt(0),
        )
        .unwrap();
    assert_eq!(project.clip(clip).unwrap().timeline.duration.value, 1_002);

    let right = project.split_clip(clip, rt(400)).unwrap();
    project
        .trim_clip(right, tr(400, 500))
        .unwrap();

    let left = project.clip(clip).unwrap();
    let r = project.clip(right).unwrap();
    assert_eq!(left.timeline, tr(0, 400));
    assert_eq!(r.timeline, tr(400, 500));
    assert_eq!(project.timeline().duration().value, 900);
}

#[test]
fn mixed_generated_and_media_layers() {
    let mut project = Project::new("mixed", FPS_24);
    let media_id = project.add_media(sample_media(FPS_24, 1_000));
    let video = project.add_track(TrackKind::Video, "Video");
    let titles = project.add_track(TrackKind::Text, "Titles");
    let gfx = project.add_track(TrackKind::Sticker, "GFX");

    let footage = project
        .add_clip(video, media_id, tr(0, 500), rt(0))
        .unwrap();
    let title = project
        .add_generated(
            titles,
            Generator::text("Act I"),
            tr(0, 48),
        )
        .unwrap();
    let lower_third = project
        .add_generated(
            gfx,
            Generator::SolidColor {
                rgba: [0, 0, 0, 180],
            },
            tr(100, 72),
        )
        .unwrap();

    assert_eq!(project.timeline().clip_count(), 3);
    assert_eq!(project.clip(footage).unwrap().media(), Some(media_id));
    assert!(project.clip(title).unwrap().is_generated());
    assert_eq!(
        project.timeline().track_of(lower_third),
        Some(gfx)
    );
    assert_eq!(project.timeline().duration().value, 500);
}

#[test]
fn multiple_clips_reference_same_media() {
    let mut project = Project::new("reuse", FPS_24);
    let media_id = project.add_media(sample_media(FPS_24, 1_000));
    let track = project.add_track(TrackKind::Video, "V1");

    let a = project
        .add_clip(track, media_id, tr(0, 100), rt(0))
        .unwrap();
    let b = project
        .add_clip(track, media_id, tr(200, 150), rt(100))
        .unwrap();

    assert_eq!(project.clip(a).unwrap().media(), Some(media_id));
    assert_eq!(project.clip(b).unwrap().media(), Some(media_id));
    assert!(project.is_media_referenced(media_id));
    assert_eq!(project.media_count(), 1);
}

#[test]
fn project_clone_captures_undo_snapshot() {
    let (mut project, media_id, track) = project_with_media(300);
    let clip = project.add_clip(track, media_id, tr(0, 100), rt(0)).unwrap();
    let snapshot = project.clone();

    project.ripple_delete(clip).unwrap();
    assert_eq!(project.timeline().clip_count(), 0);

    assert_eq!(snapshot.timeline().clip_count(), 1);
    assert_eq!(snapshot.clip(clip).unwrap().timeline, tr(0, 100));
    assert_eq!(snapshot.media_count(), 1);
}

#[test]
fn timeline_remove_track_purges_clip_index() {
    let mut project = Project::new("tracks", FPS_24);
    let media_id = project.add_media(sample_media(FPS_24, 500));
    let keep = project.add_track(TrackKind::Video, "Keep");
    let drop = project.add_track(TrackKind::Video, "Drop");

    let on_drop = project
        .add_clip(drop, media_id, tr(0, 100), rt(0))
        .unwrap();
    let on_keep = project
        .add_clip(keep, media_id, tr(0, 50), rt(0))
        .unwrap();

    project.timeline_mut().remove_track(drop).unwrap();
    assert!(project.clip(on_drop).is_none());
    assert!(project.timeline().track_of(on_drop).is_none());
    assert_eq!(project.clip(on_keep).unwrap().id, on_keep);
    assert_eq!(project.timeline().track_count(), 1);
    assert_eq!(project.timeline().clip_count(), 1);
}

#[test]
fn shape_generator_and_move_preserves_source() {
    let mut project = Project::new("gfx", FPS_24);
    let gfx = project.add_track(TrackKind::Sticker, "GFX");
    let alt = project.add_track(TrackKind::Sticker, "Alt");

    let shape = project
        .add_generated(
            gfx,
            Generator::shape(Shape::Ellipse, [255, 255, 255, 255]),
            tr(50, 100),
        )
        .unwrap();

    project.move_clip(shape, alt, rt(200)).unwrap();
    let placed = project.clip(shape).unwrap();
    assert_eq!(placed.timeline, tr(200, 100));
    assert_eq!(placed.source_range(), None);
    assert_eq!(project.timeline().track_of(shape), Some(alt));
}

#[test]
fn rate_mismatch_surfaces_throughout_edit_ops() {
    let (mut project, media_id, track) = project_with_media(200);
    let clip = project.add_clip(track, media_id, tr(0, 100), rt(0)).unwrap();
    let bad_time = rt_at(50, FPS_30);

    assert_eq!(
        project.split_clip(clip, bad_time),
        Err(ModelError::RateMismatch {
            expected: FPS_24,
            got: FPS_30,
        })
    );
    assert_eq!(
        project.move_clip(clip, track, bad_time),
        Err(ModelError::RateMismatch {
            expected: FPS_24,
            got: FPS_30,
        })
    );
    assert_eq!(
        project.add_generated(
            track,
            Generator::Adjustment,
            TimeRange::at_rate(0, 10, FPS_30),
        ),
        Err(ModelError::RateMismatch {
            expected: FPS_24,
            got: FPS_30,
        })
    );
}
