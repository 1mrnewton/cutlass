use cutlass_models::{
    Generator, MediaSource, ModelError, Project, Rational, Shape, TimeRange, TrackKind,
};

fn sample_media(fps: Rational, duration: i64) -> MediaSource {
    MediaSource::new("/tmp/sample.mp4", 3840, 2160, fps, duration, true)
}

#[test]
fn build_project_and_query_by_id() {
    let mut project = Project::new("demo", Rational::FPS_24);

    let media = sample_media(Rational::FPS_24, 1000);
    let media_id = project.add_media(media);

    let v1 = project.add_track(TrackKind::Video, "V1");

    // Two non-overlapping clips on the same track.
    let c1 = project
        .add_clip(v1, media_id, TimeRange::new(0, 100), 0)
        .expect("first clip");
    let c2 = project
        .add_clip(v1, media_id, TimeRange::new(200, 100), 100)
        .expect("second clip");

    // O(1) lookup by id across the timeline.
    assert_eq!(
        project.clip(c1).unwrap().source_range(),
        Some(TimeRange::new(0, 100))
    );
    assert_eq!(project.clip(c1).unwrap().media(), Some(media_id));
    assert_eq!(project.clip(c2).unwrap().start(), 100);
    assert_eq!(project.timeline().track_of(c1), Some(v1));

    // Timeline duration = end of the last clip.
    assert_eq!(project.timeline().duration(), 200);
    assert_eq!(project.timeline().clip_count(), 2);
}

#[test]
fn generated_clips_need_no_media() {
    let mut project = Project::new("demo", Rational::FPS_24);
    let title = project.add_track(TrackKind::Video, "Titles");

    // A text and a shape clip, neither backed by media.
    let text = project
        .add_generated(
            title,
            Generator::Text {
                content: "Hello".into(),
            },
            TimeRange::new(0, 48),
        )
        .unwrap();
    let shape = project
        .add_generated(
            title,
            Generator::Shape {
                shape: Shape::Rectangle,
            },
            TimeRange::new(48, 48),
        )
        .unwrap();

    assert_eq!(project.clip(text).unwrap().media(), None);
    assert!(project.clip(text).unwrap().is_generated());
    assert_eq!(project.clip(shape).unwrap().source_range(), None);
    assert_eq!(project.media_count(), 0);
    assert_eq!(project.timeline().duration(), 96);
}

#[test]
fn overlap_is_rejected() {
    let mut project = Project::new("demo", Rational::FPS_24);
    let media_id = project.add_media(sample_media(Rational::FPS_24, 1000));
    let v1 = project.add_track(TrackKind::Video, "V1");

    project
        .add_clip(v1, media_id, TimeRange::new(0, 100), 0)
        .unwrap();
    let err = project
        .add_clip(v1, media_id, TimeRange::new(0, 100), 50)
        .unwrap_err();
    assert_eq!(err, ModelError::Overlap(v1));
}

#[test]
fn unknown_refs_error() {
    let mut project = Project::new("demo", Rational::FPS_24);
    let v1 = project.add_track(TrackKind::Video, "V1");
    let media_id = project.add_media(sample_media(Rational::FPS_24, 1000));

    // Unknown media.
    let bad_media = MediaSource::new("/x", 1, 1, Rational::FPS_24, 10, false).id;
    assert!(matches!(
        project.add_clip(v1, bad_media, TimeRange::new(0, 10), 0),
        Err(ModelError::UnknownMedia(_))
    ));

    // Source range past the media bounds.
    assert_eq!(
        project.add_clip(v1, media_id, TimeRange::new(900, 200), 0),
        Err(ModelError::SourceOutOfBounds)
    );
}

#[test]
fn rate_conform_adjusts_timeline_duration() {
    // 30fps source on a 24fps timeline: 120 source frames (4s) -> 96 timeline frames.
    let mut project = Project::new("demo", Rational::FPS_24);
    let media_id = project.add_media(sample_media(Rational::FPS_30, 1000));
    let v1 = project.add_track(TrackKind::Video, "V1");

    let clip_id = project
        .add_clip(v1, media_id, TimeRange::new(0, 120), 0)
        .unwrap();
    assert_eq!(project.clip(clip_id).unwrap().timeline.duration, 96);
}

#[test]
fn removing_referenced_media_fails_then_succeeds() {
    let mut project = Project::new("demo", Rational::FPS_24);
    let media_id = project.add_media(sample_media(Rational::FPS_24, 1000));
    let v1 = project.add_track(TrackKind::Video, "V1");
    let clip_id = project
        .add_clip(v1, media_id, TimeRange::new(0, 100), 0)
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
fn track_stacking_order_is_preserved() {
    let mut project = Project::new("demo", Rational::FPS_24);
    let v1 = project.add_track(TrackKind::Video, "V1");
    let v2 = project.add_track(TrackKind::Video, "V2");
    let a1 = project.add_track(TrackKind::Audio, "A1");

    assert_eq!(project.timeline().order(), &[v1, v2, a1]);
    let names: Vec<&str> = project
        .timeline()
        .tracks_ordered()
        .map(|t| t.name.as_str())
        .collect();
    assert_eq!(names, ["V1", "V2", "A1"]);
}

#[test]
fn clip_at_and_source_mapping() {
    // Same-rate (24/24) so source duration maps 1:1 to timeline frames:
    // source [100,110) placed at timeline_start=10 -> timeline [10,20).
    let mut project = Project::new("demo", Rational::FPS_24);
    let media_id = project.add_media(sample_media(Rational::FPS_24, 1000));
    let v1 = project.add_track(TrackKind::Video, "V1");
    let id = project
        .add_clip(v1, media_id, TimeRange::new(100, 10), 10)
        .unwrap();

    let track = project.timeline().track(v1).unwrap();
    assert_eq!(track.clip_at(15).map(|c| c.id), Some(id));
    assert!(track.clip_at(25).is_none());
    assert_eq!(project.clip(id).unwrap().source_frame_at(15), Some(105));
}
