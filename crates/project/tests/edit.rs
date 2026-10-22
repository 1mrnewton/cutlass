//! Edits that cross the project, the timeline, and the segments.

use std::num::NonZeroU32;

use cutlass_core::Timestamp;
use cutlass_project::{
    AspectRatio, Asset, AssetId, Blend, Channels, Color, ColorSpace, Command, Crop, Easing, Error,
    FrameRate, History, Keyframe, KeyframeProperty, KeyframeValue, Outcome, Project, ProjectId,
    Segment, SegmentId, SequenceRate, Speed, TimeRange, TrackId, TrackKind, Transform,
};

fn scale(value: u32) -> NonZeroU32 {
    match NonZeroU32::new(value) {
        Some(scale) => scale,
        None => panic!("timescale is non-zero"),
    }
}

fn expect_err<T>(result: Result<T, Error>, expected: Error) {
    match result {
        Err(error) if error == expected => {}
        Err(error) => panic!("expected {expected}, got {error}"),
        Ok(_) => panic!("expected {expected}"),
    }
}

struct Cut {
    history: History,
    project: Project,
    main: TrackId,
}

fn bare() -> Cut {
    let project = Project::new(ProjectId::from_bytes([1; 16]), "cut");
    let main = project.timeline().main_track().id();
    Cut {
        history: History::new(),
        project,
        main,
    }
}

fn run(cut: &mut Cut, command: Command) -> Result<Outcome, Error> {
    let mut outcomes = cut.history.apply(&mut cut.project, &[command])?;
    if outcomes.len() != 1 {
        panic!("expected one outcome, got {}", outcomes.len());
    }
    match outcomes.pop() {
        Some(outcome) => Ok(outcome),
        None => panic!("expected one outcome"),
    }
}

fn asset_id(outcome: Outcome) -> AssetId {
    match outcome {
        Outcome::Asset(id) => id,
        other => panic!("expected an asset, got {other:?}"),
    }
}

fn segment_id(outcome: Outcome) -> SegmentId {
    match outcome {
        Outcome::Segment(id) | Outcome::Opened { segment: id, .. } => id,
        other => panic!("expected a segment, got {other:?}"),
    }
}

fn same_track(outcome: Outcome) -> SegmentId {
    match outcome {
        Outcome::Segment(id) => id,
        other => panic!("expected the same track, got {other:?}"),
    }
}

fn opened(outcome: Outcome) -> (SegmentId, TrackId) {
    match outcome {
        Outcome::Opened { segment, track } => (segment, track),
        other => panic!("expected a new track, got {other:?}"),
    }
}

fn video(seconds: i64, width: u32, height: u32) -> Result<Asset, Error> {
    Asset::video(
        format!("file:///v{seconds}.mp4"),
        Timestamp::new(seconds * 30_000, scale(30_000)),
        width,
        height,
        true,
    )
}

fn add_video(cut: &mut Cut, seconds: i64, width: u32, height: u32) -> Result<AssetId, Error> {
    Ok(asset_id(run(
        cut,
        Command::AddAsset(video(seconds, width, height)?),
    )?))
}

fn load() -> Result<(Cut, AssetId), Error> {
    let mut cut = bare();
    let asset = add_video(&mut cut, 10, 1280, 720)?;
    Ok((cut, asset))
}

fn media(
    project: &Project,
    asset: AssetId,
    start_us: i64,
    source_ticks: i64,
    channels: Channels,
) -> Result<Segment, Error> {
    let asset = project.asset(asset).ok_or(Error::UnknownAsset(asset))?;
    let stored = asset.duration().ok_or(Error::ImageSource)?;
    let source = TimeRange::new(
        Timestamp::new(0, stored.timescale),
        Timestamp::new(source_ticks, stored.timescale),
    )?;
    Segment::media(asset, source, start_us, channels, Speed::ONE)
}

fn place_main(
    cut: &mut Cut,
    asset: AssetId,
    start_us: i64,
    ticks: i64,
) -> Result<SegmentId, Error> {
    let channels = cut
        .project
        .asset(asset)
        .ok_or(Error::UnknownAsset(asset))?
        .default_channels();
    let segment = media(&cut.project, asset, start_us, ticks, channels)?;
    let main = cut.main;
    Ok(segment_id(run(
        cut,
        Command::Place {
            track: Some(main),
            segment,
        },
    )?))
}

fn place_on(
    cut: &mut Cut,
    track: TrackId,
    asset: AssetId,
    start_us: i64,
    ticks: i64,
) -> Result<Outcome, Error> {
    let channels = cut
        .project
        .asset(asset)
        .ok_or(Error::UnknownAsset(asset))?
        .default_channels();
    let segment = media(&cut.project, asset, start_us, ticks, channels)?;
    run(
        cut,
        Command::Place {
            track: Some(track),
            segment,
        },
    )
}

fn open_picture(
    cut: &mut Cut,
    asset: AssetId,
    start_us: i64,
    ticks: i64,
) -> Result<(SegmentId, TrackId), Error> {
    let channels = cut
        .project
        .asset(asset)
        .ok_or(Error::UnknownAsset(asset))?
        .default_channels();
    let segment = media(&cut.project, asset, start_us, ticks, channels)?;
    Ok(opened(run(
        cut,
        Command::Place {
            track: None,
            segment,
        },
    )?))
}

fn open_sound(cut: &mut Cut, asset: AssetId, start_us: i64) -> Result<(SegmentId, TrackId), Error> {
    let segment = media(&cut.project, asset, start_us, 30_000, Channels::Sound)?;
    Ok(opened(run(
        cut,
        Command::Place {
            track: None,
            segment,
        },
    )?))
}

fn start(cut: &Cut, id: SegmentId) -> Result<i64, Error> {
    Ok(cut
        .project
        .timeline()
        .segment(id)
        .ok_or(Error::UnknownSegment(id))?
        .timeline()
        .start()
        .value)
}

fn on_track(cut: &Cut, id: SegmentId) -> TrackId {
    cut.project
        .timeline()
        .tracks()
        .iter()
        .find(|track| track.segments().iter().any(|segment| segment.id() == id))
        .expect("segment is on a track")
        .id()
}

fn kinds(cut: &Cut) -> Vec<TrackKind> {
    cut.project
        .timeline()
        .tracks()
        .iter()
        .map(cutlass_project::Track::kind)
        .collect()
}

#[test]
fn magnet_packs_a_main_clip_to_the_start() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let id = place_main(&mut cut, asset, 5_000_000, 30_000)?;
    assert_eq!(start(&cut, id)?, 0);
    assert_eq!(cut.project.duration().value, 1_000_000);
    assert_eq!(cut.project.timeline().tracks().len(), 1);
    let segment = cut
        .project
        .timeline()
        .segment(id)
        .ok_or(Error::UnknownSegment(id))?;
    assert_eq!(
        segment
            .source()
            .ok_or(Error::ImageSource)?
            .start()
            .timescale
            .get(),
        30_000
    );
    assert_eq!(segment.timeline().duration().timescale.get(), 1_000_000);
    Ok(())
}

#[test]
fn magnet_off_keeps_the_gap() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    run(&mut cut, Command::SetMagnet(false))?;
    let id = place_main(&mut cut, asset, 5_000_000, 30_000)?;
    assert_eq!(start(&cut, id)?, 5_000_000);
    assert_eq!(cut.project.duration().value, 6_000_000);
    Ok(())
}

#[test]
fn magnet_does_not_pack_an_overlay() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let (id, _) = open_picture(&mut cut, asset, 5_000_000, 30_000)?;
    assert_eq!(start(&cut, id)?, 5_000_000);
    assert_eq!(cut.project.duration().value, 6_000_000);
    assert!(cut.project.timeline().main_track().segments().is_empty());
    assert_eq!(kinds(&cut), vec![TrackKind::Video, TrackKind::Video]);
    assert!(!cut.project.timeline().tracks()[1].is_main());
    Ok(())
}

#[test]
fn turning_magnet_on_packs_and_linkage_keeps_the_offset() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    run(&mut cut, Command::SetMagnet(false))?;
    let main = place_main(&mut cut, asset, 1_000_000, 30_000)?;
    let (overlay, _) = open_picture(&mut cut, asset, 1_000_500, 30_000)?;
    run(&mut cut, Command::SetMagnet(true))?;
    assert_eq!(start(&cut, main)?, 0);
    assert_eq!(start(&cut, overlay)?, 500);
    Ok(())
}

#[test]
fn turning_magnet_off_leaves_the_packed_clips() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let id = place_main(&mut cut, asset, 5_000_000, 30_000)?;
    run(&mut cut, Command::SetMagnet(false))?;
    assert_eq!(start(&cut, id)?, 0);
    assert!(!cut.project.settings().magnet());
    Ok(())
}

#[test]
fn linkage_off_does_not_move_the_overlay() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    run(&mut cut, Command::SetMagnet(false))?;
    run(&mut cut, Command::SetLinkage(false))?;
    let main = place_main(&mut cut, asset, 1_000_000, 30_000)?;
    let (overlay, _) = open_picture(&mut cut, asset, 1_000_500, 30_000)?;
    run(&mut cut, Command::SetMagnet(true))?;
    assert_eq!(start(&cut, main)?, 0);
    assert_eq!(start(&cut, overlay)?, 1_000_500);
    Ok(())
}

#[test]
fn an_existing_link_survives_a_new_main_clip() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let first = place_main(&mut cut, asset, 0, 30_000)?;
    let (overlay, _) = open_picture(&mut cut, asset, 100, 30_000)?;
    let second = place_main(&mut cut, asset, 0, 30_000)?;
    assert_eq!(start(&cut, second)?, 0);
    assert_eq!(start(&cut, first)?, 1_000_000);
    assert_eq!(start(&cut, overlay)?, 1_000_100);
    assert_eq!(cut.project.timeline().tracks().len(), 2);
    Ok(())
}

#[test]
fn overlap_opens_another_track_of_the_same_kind() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    run(&mut cut, Command::SetMagnet(false))?;
    let first = place_main(&mut cut, asset, 0, 30_000)?;
    let main = cut.main;
    let (second, track) = opened(place_on(&mut cut, main, asset, 0, 30_000)?);
    assert_ne!(track, cut.main);
    assert_eq!(kinds(&cut), vec![TrackKind::Video, TrackKind::Video]);
    assert!(cut.project.timeline().tracks()[0].is_main());
    assert!(!cut.project.timeline().tracks()[1].is_main());
    assert_eq!(start(&cut, first)?, 0);
    assert_eq!(start(&cut, second)?, 0);
    Ok(())
}

#[test]
fn a_free_time_stays_on_the_same_track() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    run(&mut cut, Command::SetMagnet(false))?;
    place_main(&mut cut, asset, 0, 30_000)?;
    let main = cut.main;
    let second = same_track(place_on(&mut cut, main, asset, 2_000_000, 30_000)?);
    assert_eq!(cut.project.timeline().tracks().len(), 1);
    assert_eq!(start(&cut, second)?, 2_000_000);
    Ok(())
}

#[test]
fn linkage_deletes_a_segment_that_sits_on_one_main_segment() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let main = place_main(&mut cut, asset, 0, 30_000)?;
    let (overlay, _) = open_picture(&mut cut, asset, 100, 30_000)?;
    run(&mut cut, Command::RemoveSegment(main))?;
    assert!(cut.project.timeline().segment(overlay).is_none());
    assert_eq!(cut.project.timeline().tracks().len(), 1);
    assert!(cut.project.timeline().main_track().segments().is_empty());
    assert_eq!(cut.project.duration().value, 0);
    Ok(())
}

#[test]
fn linkage_keeps_a_segment_that_sits_on_two_main_segments() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    run(&mut cut, Command::SetMagnet(false))?;
    let first = place_main(&mut cut, asset, 0, 30_000)?;
    let second = place_main(&mut cut, asset, 1_000_000, 30_000)?;
    let (overlay, _) = open_picture(&mut cut, asset, 500_000, 30_000)?;
    run(&mut cut, Command::RemoveSegment(first))?;
    assert_eq!(start(&cut, overlay)?, 500_000);
    assert_eq!(start(&cut, second)?, 1_000_000);
    Ok(())
}

#[test]
fn linkage_off_keeps_the_segment_when_the_main_segment_is_deleted() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    run(&mut cut, Command::SetLinkage(false))?;
    let main = place_main(&mut cut, asset, 0, 30_000)?;
    let (overlay, _) = open_picture(&mut cut, asset, 0, 30_000)?;
    run(&mut cut, Command::RemoveSegment(main))?;
    assert_eq!(start(&cut, overlay)?, 0);
    assert!(cut.project.timeline().main_track().segments().is_empty());
    assert_eq!(cut.project.timeline().tracks().len(), 2);
    Ok(())
}

#[test]
fn deleting_a_main_clip_packs_the_rest_and_moves_what_follows_it() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let main = place_main(&mut cut, asset, 0, 30_000)?;
    let next = place_main(&mut cut, asset, 1_000_000, 30_000)?;
    let (sound, _) = open_sound(&mut cut, asset, 1_000_100)?;
    run(&mut cut, Command::RemoveSegment(main))?;
    assert!(cut.project.timeline().segment(main).is_none());
    assert_eq!(start(&cut, next)?, 0);
    assert_eq!(start(&cut, sound)?, 100);
    assert_eq!(kinds(&cut), vec![TrackKind::Audio, TrackKind::Video]);
    Ok(())
}

#[test]
fn magnet_off_delete_leaves_the_gap() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    run(&mut cut, Command::SetMagnet(false))?;
    let main = place_main(&mut cut, asset, 0, 30_000)?;
    let next = place_main(&mut cut, asset, 2_000_000, 30_000)?;
    let (overlay, _) = open_picture(&mut cut, asset, 2_000_000, 30_000)?;
    run(&mut cut, Command::RemoveSegment(main))?;
    assert_eq!(start(&cut, next)?, 2_000_000);
    assert_eq!(start(&cut, overlay)?, 2_000_000);
    assert_eq!(cut.project.duration().value, 3_000_000);
    Ok(())
}

#[test]
fn removing_the_last_overlay_segment_removes_the_track() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let (overlay, _) = open_picture(&mut cut, asset, 0, 30_000)?;
    run(&mut cut, Command::RemoveSegment(overlay))?;
    assert_eq!(cut.project.timeline().tracks().len(), 1);
    assert!(cut.project.timeline().main_track().is_main());
    Ok(())
}

#[test]
fn duration_is_the_end_of_the_latest_segment() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    place_main(&mut cut, asset, 0, 30_000)?;
    open_picture(&mut cut, asset, 2_000_000, 30_000)?;
    assert_eq!(cut.project.duration().value, 3_000_000);
    assert_eq!(cut.project.duration().timescale.get(), 1_000_000);
    Ok(())
}

#[test]
fn the_first_video_replaces_an_original_canvas() -> Result<(), Error> {
    let mut cut = bare();
    let image = Asset::image("file:///s.png", 100, 50)?;
    run(&mut cut, Command::AddAsset(image))?;
    assert_eq!(cut.project.settings().width(), 1920);
    assert_eq!(cut.project.settings().height(), 1080);

    run(
        &mut cut,
        Command::SetCanvas {
            width: 100,
            height: 80,
            ratio: AspectRatio::Original,
        },
    )?;
    add_video(&mut cut, 10, 1280, 720)?;
    assert_eq!(cut.project.settings().width(), 1280);
    assert_eq!(cut.project.settings().height(), 720);
    assert_eq!(cut.project.settings().ratio(), AspectRatio::Original);

    add_video(&mut cut, 12, 640, 360)?;
    assert_eq!(cut.project.settings().width(), 1280);
    assert_eq!(cut.project.settings().height(), 720);
    Ok(())
}

#[test]
fn a_fixed_ratio_keeps_the_canvas_size() -> Result<(), Error> {
    let mut cut = bare();
    run(
        &mut cut,
        Command::SetCanvas {
            width: 1080,
            height: 1920,
            ratio: AspectRatio::R9x16,
        },
    )?;
    add_video(&mut cut, 10, 1280, 720)?;
    assert_eq!(cut.project.settings().width(), 1080);
    assert_eq!(cut.project.settings().height(), 1920);
    assert_eq!(cut.project.settings().ratio(), AspectRatio::R9x16);
    Ok(())
}

#[test]
fn a_zero_canvas_is_rejected_and_the_size_stays() -> Result<(), Error> {
    let mut cut = bare();
    expect_err(
        run(
            &mut cut,
            Command::SetCanvas {
                width: 0,
                height: 10,
                ratio: AspectRatio::Square,
            },
        ),
        Error::ZeroCanvas,
    );
    assert_eq!(cut.project.settings().width(), 1920);
    assert_eq!(cut.project.settings().ratio(), AspectRatio::Original);
    assert_eq!(
        cut.history.undo(&mut cut.project).unwrap_err(),
        Error::NothingToUndo
    );
    Ok(())
}

#[test]
fn rate_color_and_background_are_stored() -> Result<(), Error> {
    let mut cut = bare();
    run(
        &mut cut,
        Command::SetRate(SequenceRate::new(FrameRate::Fps2997, false)?),
    )?;
    run(&mut cut, Command::SetColorSpace(ColorSpace::Rec2100Pq))?;
    run(&mut cut, Command::SetBackground(Color::new(1, 2, 3)))?;
    assert_eq!(cut.project.settings().rate().rate(), FrameRate::Fps2997);
    assert!(!cut.project.settings().rate().drop_frame());
    assert_eq!(cut.project.settings().color_space(), ColorSpace::Rec2100Pq);
    assert_eq!(cut.project.settings().background().channels(), (1, 2, 3));
    Ok(())
}

#[test]
fn double_speed_halves_the_packed_duration() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let id = place_main(&mut cut, asset, 0, 30_000)?;
    run(
        &mut cut,
        Command::SetSpeed {
            segment: id,
            speed: Speed::new(2, 1)?,
        },
    )?;
    let segment = cut
        .project
        .timeline()
        .segment(id)
        .ok_or(Error::UnknownSegment(id))?;
    assert_eq!(segment.speed().numerator(), 2);
    assert_eq!(segment.timeline().duration().value, 500_000);
    assert_eq!(start(&cut, id)?, 0);
    Ok(())
}

#[test]
fn trim_pulls_the_next_main_clip_and_its_follower_in() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let first = place_main(&mut cut, asset, 0, 30_000)?;
    let second = place_main(&mut cut, asset, 1_000_000, 30_000)?;
    let (overlay, _) = open_picture(&mut cut, asset, 1_000_100, 30_000)?;
    let source = TimeRange::new(
        Timestamp::new(0, scale(30_000)),
        Timestamp::new(15_000, scale(30_000)),
    )?;
    run(
        &mut cut,
        Command::Trim {
            segment: first,
            source,
        },
    )?;
    assert_eq!(
        cut.project
            .timeline()
            .segment(first)
            .ok_or(Error::UnknownSegment(first))?
            .timeline()
            .duration()
            .value,
        500_000
    );
    assert_eq!(start(&cut, second)?, 500_000);
    assert_eq!(start(&cut, overlay)?, 500_100);
    let kept = cut
        .project
        .timeline()
        .segment(first)
        .ok_or(Error::UnknownSegment(first))?;
    assert_eq!(
        kept.source().ok_or(Error::ImageSource)?.duration().value,
        15_000
    );
    assert_eq!(
        kept.source()
            .ok_or(Error::ImageSource)?
            .duration()
            .timescale
            .get(),
        30_000
    );
    Ok(())
}

#[test]
fn a_trim_that_would_drop_a_keyframe_changes_nothing() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let mut segment = media(&cut.project, asset, 0, 30_000, Channels::Both)?;
    segment.push_keyframe(Keyframe::new(1_000_000, KeyframeValue::Opacity(0.0))?)?;
    let main = cut.main;
    let id = segment_id(run(
        &mut cut,
        Command::Place {
            track: Some(main),
            segment,
        },
    )?);
    let source = TimeRange::new(
        Timestamp::new(0, scale(30_000)),
        Timestamp::new(15_000, scale(30_000)),
    )?;
    expect_err(
        run(
            &mut cut,
            Command::Trim {
                segment: id,
                source,
            },
        ),
        Error::KeyframeOutOfRange,
    );
    let kept = cut
        .project
        .timeline()
        .segment(id)
        .ok_or(Error::UnknownSegment(id))?;
    assert_eq!(kept.timeline().duration().value, 1_000_000);
    assert_eq!(kept.keyframes().len(), 1);
    assert_eq!(kept.keyframes()[0].offset().value, 1_000_000);
    Ok(())
}

#[test]
fn a_trim_that_would_overlap_changes_nothing_when_magnet_is_off() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    run(&mut cut, Command::SetMagnet(false))?;
    let first = place_main(&mut cut, asset, 0, 30_000)?;
    let second = place_main(&mut cut, asset, 1_500_000, 30_000)?;
    let source = TimeRange::new(
        Timestamp::new(0, scale(30_000)),
        Timestamp::new(60_000, scale(30_000)),
    )?;
    expect_err(
        run(
            &mut cut,
            Command::Trim {
                segment: first,
                source,
            },
        ),
        Error::Overlap,
    );
    assert_eq!(
        cut.project
            .timeline()
            .segment(first)
            .ok_or(Error::UnknownSegment(first))?
            .timeline()
            .duration()
            .value,
        1_000_000
    );
    assert_eq!(start(&cut, second)?, 1_500_000);
    Ok(())
}

#[test]
fn an_image_keeps_its_timeline_duration_and_has_no_speed() -> Result<(), Error> {
    let mut cut = bare();
    let image = Asset::image("content://media/still", 400, 300)?;
    let image = asset_id(run(&mut cut, Command::AddAsset(image))?);
    let segment = {
        let asset = cut.project.asset(image).ok_or(Error::UnknownAsset(image))?;
        Segment::image(asset, 5_000_000, 5_000_000)?
    };
    let main = cut.main;
    let id = segment_id(run(
        &mut cut,
        Command::Place {
            track: Some(main),
            segment,
        },
    )?);
    let segment = cut
        .project
        .timeline()
        .segment(id)
        .ok_or(Error::UnknownSegment(id))?;
    assert!(segment.source().is_none());
    assert_eq!(segment.timeline().duration().value, 5_000_000);
    assert_eq!(start(&cut, id)?, 0);
    expect_err(
        run(
            &mut cut,
            Command::SetSpeed {
                segment: id,
                speed: Speed::new(2, 1)?,
            },
        ),
        Error::ImageSource,
    );
    assert_eq!(
        cut.project
            .timeline()
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .timeline()
            .duration()
            .value,
        5_000_000
    );
    assert_eq!(cut.project.settings().width(), 1920);
    Ok(())
}

#[test]
fn sound_opens_an_audio_track_below_main_and_follows_it() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    run(&mut cut, Command::SetMagnet(false))?;
    let main = place_main(&mut cut, asset, 1_000_000, 30_000)?;
    let (sound, _) = open_sound(&mut cut, asset, 1_000_200)?;
    assert_eq!(kinds(&cut), vec![TrackKind::Audio, TrackKind::Video]);
    run(&mut cut, Command::SetMagnet(true))?;
    assert_eq!(start(&cut, main)?, 0);
    assert_eq!(start(&cut, sound)?, 200);
    assert_eq!(kinds(&cut), vec![TrackKind::Audio, TrackKind::Video]);
    Ok(())
}

#[test]
fn a_segment_refuses_the_wrong_track() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let wav = Asset::audio("file:///a.wav", Timestamp::new(30_000, scale(30_000)))?;
    let wav = asset_id(run(&mut cut, Command::AddAsset(wav))?);
    let (_, audio_track) = open_sound(&mut cut, asset, 0)?;
    let before = cut.project.next_id();
    let sound = media(&cut.project, wav, 0, 30_000, Channels::Sound)?;
    let main = cut.main;
    expect_err(
        run(
            &mut cut,
            Command::Place {
                track: Some(main),
                segment: sound,
            },
        ),
        Error::WrongTrack,
    );
    let picture = media(&cut.project, asset, 0, 30_000, Channels::Both)?;
    expect_err(
        run(
            &mut cut,
            Command::Place {
                track: Some(audio_track),
                segment: picture,
            },
        ),
        Error::WrongTrack,
    );
    assert_eq!(cut.project.timeline().main_track().segments().len(), 0);
    assert_eq!(cut.project.timeline().tracks().len(), 2);
    assert_eq!(cut.project.next_id(), before);
    Ok(())
}

#[test]
fn a_linked_move_that_would_overlap_changes_nothing() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    run(&mut cut, Command::SetMagnet(false))?;
    let first = place_main(&mut cut, asset, 0, 30_000)?;
    let second = place_main(&mut cut, asset, 2_000_000, 30_000)?;
    let (held, track) = open_picture(&mut cut, asset, 1_000_000, 30_000)?;
    let moved = same_track(place_on(&mut cut, track, asset, 2_000_000, 30_000)?);
    expect_err(
        run(&mut cut, Command::SetMagnet(true)),
        Error::LinkageOverlap(moved),
    );
    assert_eq!(start(&cut, first)?, 0);
    assert_eq!(start(&cut, second)?, 2_000_000);
    assert_eq!(start(&cut, moved)?, 2_000_000);
    assert_eq!(start(&cut, held)?, 1_000_000);
    assert!(!cut.project.settings().magnet());
    Ok(())
}

#[test]
fn a_linked_move_past_zero_changes_nothing() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    run(&mut cut, Command::SetMagnet(false))?;
    let main = place_main(&mut cut, asset, 1_000_000, 30_000)?;
    let (overlay, _) = open_picture(&mut cut, asset, 100_000, 30_000)?;
    expect_err(run(&mut cut, Command::SetMagnet(true)), Error::NegativeTime);
    assert_eq!(start(&cut, main)?, 1_000_000);
    assert_eq!(start(&cut, overlay)?, 100_000);
    assert!(!cut.project.settings().magnet());
    Ok(())
}

#[test]
fn an_unknown_asset_or_track_refuses_the_place() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let other = video(11, 320, 180)?;
    let segment = {
        let stored = other.duration().ok_or(Error::ImageSource)?;
        let source = TimeRange::new(
            Timestamp::new(0, stored.timescale),
            Timestamp::new(30_000, stored.timescale),
        )?;
        Segment::media(&other, source, 0, Channels::Both, Speed::ONE)?
    };
    let main = cut.main;
    match run(
        &mut cut,
        Command::Place {
            track: Some(main),
            segment,
        },
    ) {
        Err(Error::UnknownAsset(id)) if id.get() == 0 => {}
        other => panic!("expected an unknown asset, got {other:?}"),
    }
    let missing = TrackId::new(99)?;
    let segment = media(&cut.project, asset, 0, 30_000, Channels::Both)?;
    expect_err(
        run(
            &mut cut,
            Command::Place {
                track: Some(missing),
                segment,
            },
        ),
        Error::UnknownTrack(missing),
    );
    assert!(cut.project.timeline().main_track().segments().is_empty());
    Ok(())
}

#[test]
fn two_creates_receive_different_ids() -> Result<(), Error> {
    let (mut cut, first_asset) = load()?;
    let second_asset = add_video(&mut cut, 4, 320, 180)?;
    assert_ne!(first_asset, second_asset);
    let first = place_main(&mut cut, first_asset, 0, 30_000)?;
    let second = place_main(&mut cut, first_asset, 1_000_000, 30_000)?;
    assert_ne!(first, second);
    assert_ne!(first.get(), first_asset.get());
    assert_eq!(cut.project.timeline().main_track().segments().len(), 2);
    Ok(())
}

#[test]
fn magnet_off_moves_a_clip_to_the_new_start() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    run(&mut cut, Command::SetMagnet(false))?;
    let id = place_main(&mut cut, asset, 0, 30_000)?;
    assert_eq!(
        run(
            &mut cut,
            Command::Move {
                segment: id,
                track: None,
                start: 2_000_000,
            },
        )?,
        Outcome::Unchanged
    );
    assert_eq!(start(&cut, id)?, 2_000_000);
    Ok(())
}

#[test]
fn magnet_uses_the_start_to_choose_the_order() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let first = place_main(&mut cut, asset, 0, 30_000)?;
    let second = place_main(&mut cut, asset, 1_000_000, 30_000)?;
    run(
        &mut cut,
        Command::Move {
            segment: second,
            track: None,
            start: 0,
        },
    )?;
    assert_eq!(start(&cut, second)?, 0);
    assert_eq!(start(&cut, first)?, 1_000_000);
    Ok(())
}

#[test]
fn moving_a_main_clip_moves_the_segment_that_sits_on_it() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    run(&mut cut, Command::SetMagnet(false))?;
    let main = place_main(&mut cut, asset, 1_000_000, 30_000)?;
    let (overlay, _) = open_picture(&mut cut, asset, 1_000_500, 30_000)?;
    run(
        &mut cut,
        Command::Move {
            segment: main,
            track: None,
            start: 3_000_000,
        },
    )?;
    assert_eq!(start(&cut, main)?, 3_000_000);
    assert_eq!(start(&cut, overlay)?, 3_000_500);
    Ok(())
}

#[test]
fn a_move_onto_a_taken_time_changes_nothing() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    run(&mut cut, Command::SetMagnet(false))?;
    let first = place_main(&mut cut, asset, 0, 30_000)?;
    let second = place_main(&mut cut, asset, 2_000_000, 30_000)?;
    expect_err(
        run(
            &mut cut,
            Command::Move {
                segment: first,
                track: None,
                start: 2_000_000,
            },
        ),
        Error::Overlap,
    );
    assert_eq!(start(&cut, first)?, 0);
    assert_eq!(start(&cut, second)?, 2_000_000);
    Ok(())
}

#[test]
fn a_sound_segment_cannot_move_onto_the_main_track() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let (sound, track) = open_sound(&mut cut, asset, 0)?;
    let main = cut.main;
    expect_err(
        run(
            &mut cut,
            Command::Move {
                segment: sound,
                track: Some(main),
                start: 0,
            },
        ),
        Error::WrongTrack,
    );
    assert_eq!(on_track(&cut, sound), track);
    Ok(())
}

#[test]
fn a_main_clip_can_move_onto_an_overlay() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    run(&mut cut, Command::SetMagnet(false))?;
    let main_clip = place_main(&mut cut, asset, 0, 30_000)?;
    let (overlay, track) = open_picture(&mut cut, asset, 5_000_000, 30_000)?;
    run(
        &mut cut,
        Command::Move {
            segment: main_clip,
            track: Some(track),
            start: 0,
        },
    )?;
    assert_eq!(on_track(&cut, main_clip), track);
    assert_eq!(start(&cut, main_clip)?, 0);
    assert_eq!(start(&cut, overlay)?, 5_000_000);
    assert!(cut.project.timeline().main_track().segments().is_empty());
    Ok(())
}

#[test]
fn leaving_the_main_track_packs_what_remains_and_moves_its_follower() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let first = place_main(&mut cut, asset, 0, 30_000)?;
    let second = place_main(&mut cut, asset, 1_000_000, 30_000)?;
    let (follower, _) = open_picture(&mut cut, asset, 100, 15_000)?;
    let (_, destination) = open_picture(&mut cut, asset, 8_000_000, 30_000)?;
    run(
        &mut cut,
        Command::Move {
            segment: first,
            track: Some(destination),
            start: 5_000_000,
        },
    )?;
    assert_eq!(on_track(&cut, first), destination);
    assert_eq!(start(&cut, first)?, 5_000_000);
    assert_eq!(start(&cut, second)?, 0);
    assert_eq!(start(&cut, follower)?, 5_000_100);
    Ok(())
}

#[test]
fn split_keeps_the_left_piece_and_returns_the_right() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    run(&mut cut, Command::SetMagnet(false))?;
    let left = place_main(&mut cut, asset, 0, 30_000)?;
    let right = segment_id(run(
        &mut cut,
        Command::Split {
            segment: left,
            at: 500_000,
        },
    )?);
    assert_ne!(left, right);
    assert_eq!(on_track(&cut, right), cut.main);
    let left_segment = cut
        .project
        .timeline()
        .segment(left)
        .ok_or(Error::UnknownSegment(left))?;
    let right_segment = cut
        .project
        .timeline()
        .segment(right)
        .ok_or(Error::UnknownSegment(right))?;
    assert_eq!(left_segment.timeline().duration().value, 500_000);
    assert_eq!(right_segment.timeline().start().value, 500_000);
    assert_eq!(right_segment.timeline().duration().value, 500_000);
    assert_eq!(
        left_segment
            .source()
            .ok_or(Error::ImageSource)?
            .duration()
            .value,
        15_000
    );
    assert_eq!(
        right_segment
            .source()
            .ok_or(Error::ImageSource)?
            .start()
            .value,
        15_000
    );
    assert_eq!(
        right_segment
            .source()
            .ok_or(Error::ImageSource)?
            .duration()
            .value,
        15_000
    );
    Ok(())
}

#[test]
fn a_split_on_the_edge_changes_nothing() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    run(&mut cut, Command::SetMagnet(false))?;
    let id = place_main(&mut cut, asset, 0, 30_000)?;
    expect_err(
        run(&mut cut, Command::Split { segment: id, at: 0 }),
        Error::SplitOutside,
    );
    expect_err(
        run(
            &mut cut,
            Command::Split {
                segment: id,
                at: 1_000_000,
            },
        ),
        Error::SplitOutside,
    );
    assert_eq!(
        cut.project
            .timeline()
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .timeline()
            .duration()
            .value,
        1_000_000
    );
    assert_eq!(cut.project.timeline().main_track().segments().len(), 1);
    Ok(())
}

#[test]
fn a_keyframe_past_the_cut_moves_to_the_right_piece() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    run(&mut cut, Command::SetMagnet(false))?;
    let mut segment = media(&cut.project, asset, 0, 30_000, Channels::Both)?;
    segment.push_keyframe(Keyframe::new(200_000, KeyframeValue::Opacity(0.25))?)?;
    segment.push_keyframe(
        Keyframe::new(700_000, KeyframeValue::Opacity(0.5))?.with_easing(Easing::EaseIn),
    )?;
    let main = cut.main;
    let left = segment_id(run(
        &mut cut,
        Command::Place {
            track: Some(main),
            segment,
        },
    )?);
    let right = segment_id(run(
        &mut cut,
        Command::Split {
            segment: left,
            at: 500_000,
        },
    )?);
    let left_keys = cut
        .project
        .timeline()
        .segment(left)
        .ok_or(Error::UnknownSegment(left))?
        .keyframes();
    let right_keys = cut
        .project
        .timeline()
        .segment(right)
        .ok_or(Error::UnknownSegment(right))?
        .keyframes();
    assert_eq!(left_keys.len(), 1);
    assert_eq!(left_keys[0].offset().value, 200_000);
    assert_eq!(right_keys.len(), 1);
    assert_eq!(right_keys[0].offset().value, 200_000);
    assert_eq!(right_keys[0].easing(), Easing::EaseIn);
    assert_eq!(left_keys[0].easing(), Easing::Linear);
    Ok(())
}

#[test]
fn splitting_the_main_clip_leaves_the_next_one_packed_against_it() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let left = place_main(&mut cut, asset, 0, 30_000)?;
    let next = place_main(&mut cut, asset, 1_000_000, 30_000)?;
    let right = segment_id(run(
        &mut cut,
        Command::Split {
            segment: left,
            at: 500_000,
        },
    )?);
    assert_eq!(start(&cut, right)?, 500_000);
    assert_eq!(start(&cut, next)?, 1_000_000);
    assert_eq!(cut.project.timeline().main_track().segments().len(), 3);
    Ok(())
}

#[test]
fn an_image_splits_on_the_timeline_without_a_source() -> Result<(), Error> {
    let mut cut = bare();
    run(&mut cut, Command::SetMagnet(false))?;
    let image = asset_id(run(
        &mut cut,
        Command::AddAsset(Asset::image("content://media/still", 8, 8)?),
    )?);
    let segment = {
        let asset = cut.project.asset(image).ok_or(Error::UnknownAsset(image))?;
        Segment::image(asset, 1_000_000, 4_000_000)?
    };
    let main = cut.main;
    let left = segment_id(run(
        &mut cut,
        Command::Place {
            track: Some(main),
            segment,
        },
    )?);
    let right = segment_id(run(
        &mut cut,
        Command::Split {
            segment: left,
            at: 2_000_000,
        },
    )?);
    let left = cut
        .project
        .timeline()
        .segment(left)
        .ok_or(Error::UnknownSegment(left))?;
    let right = cut
        .project
        .timeline()
        .segment(right)
        .ok_or(Error::UnknownSegment(right))?;
    assert!(left.source().is_none());
    assert!(right.source().is_none());
    assert_eq!(left.timeline().duration().value, 1_000_000);
    assert_eq!(right.timeline().start().value, 2_000_000);
    assert_eq!(right.timeline().duration().value, 3_000_000);
    Ok(())
}

#[test]
fn transform_moves_the_picture_and_undo_puts_it_back() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let id = place_main(&mut cut, asset, 0, 30_000)?;
    let transform = Transform::new((0.25, -0.1), (2.0, 0.5), 15.0, 0.4, true, false)?;
    assert_eq!(
        run(
            &mut cut,
            Command::SetTransform {
                segment: id,
                transform,
            },
        )?,
        Outcome::Unchanged
    );
    let placed = cut
        .project
        .timeline()
        .segment(id)
        .ok_or(Error::UnknownSegment(id))?;
    assert_eq!(placed.transform(), transform);
    assert_eq!(placed.timeline().start().value, 0);
    cut.history.undo(&mut cut.project)?;
    let restored = cut
        .project
        .timeline()
        .segment(id)
        .ok_or(Error::UnknownSegment(id))?;
    assert_eq!(restored.transform(), Transform::IDENTITY);
    cut.history.redo(&mut cut.project)?;
    let again = cut
        .project
        .timeline()
        .segment(id)
        .ok_or(Error::UnknownSegment(id))?;
    assert_eq!(again.transform(), transform);
    let missing = SegmentId::new(99).expect("id");
    expect_err(
        run(
            &mut cut,
            Command::SetTransform {
                segment: missing,
                transform: Transform::IDENTITY,
            },
        ),
        Error::UnknownSegment(missing),
    );
    Ok(())
}

#[test]
fn a_keyframe_is_stored_and_undo_removes_it() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let id = place_main(&mut cut, asset, 0, 30_000)?;
    let keyframe = Keyframe::new(0, KeyframeValue::Opacity(0.5))?;
    assert_eq!(
        run(
            &mut cut,
            Command::AddKeyframe {
                segment: id,
                keyframe,
            },
        )?,
        Outcome::Unchanged
    );
    let scale = Keyframe::new(0, KeyframeValue::Scale { x: 2.0, y: 1.0 })?;
    run(
        &mut cut,
        Command::AddKeyframe {
            segment: id,
            keyframe: scale,
        },
    )?;
    let placed = cut
        .project
        .timeline()
        .segment(id)
        .ok_or(Error::UnknownSegment(id))?;
    assert_eq!(placed.transform(), Transform::IDENTITY);
    assert_eq!(placed.keyframes().len(), 2);
    assert_eq!(placed.keyframes()[0].value(), KeyframeValue::Opacity(0.5));
    cut.history.undo(&mut cut.project)?;
    let undone = cut
        .project
        .timeline()
        .segment(id)
        .ok_or(Error::UnknownSegment(id))?;
    assert_eq!(undone.keyframes().len(), 1);
    cut.history.undo(&mut cut.project)?;
    let cleared = cut
        .project
        .timeline()
        .segment(id)
        .ok_or(Error::UnknownSegment(id))?;
    assert!(cleared.keyframes().is_empty());
    cut.history.redo(&mut cut.project)?;
    run(
        &mut cut,
        Command::RemoveKeyframe {
            segment: id,
            offset_us: 0,
            property: KeyframeProperty::Opacity,
        },
    )?;
    let removed = cut
        .project
        .timeline()
        .segment(id)
        .ok_or(Error::UnknownSegment(id))?;
    assert!(removed.keyframes().is_empty());
    Ok(())
}

#[test]
fn a_bad_keyframe_changes_nothing() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let id = place_main(&mut cut, asset, 0, 30_000)?;
    let keyframe = Keyframe::new(0, KeyframeValue::Opacity(0.5))?;
    run(
        &mut cut,
        Command::AddKeyframe {
            segment: id,
            keyframe,
        },
    )?;
    expect_err(
        run(
            &mut cut,
            Command::AddKeyframe {
                segment: id,
                keyframe,
            },
        ),
        Error::DuplicateKeyframe,
    );
    expect_err(
        run(
            &mut cut,
            Command::AddKeyframe {
                segment: id,
                keyframe: Keyframe::new(1_000_001, KeyframeValue::Rotation(10.0))?,
            },
        ),
        Error::KeyframeOutOfRange,
    );
    expect_err(
        run(
            &mut cut,
            Command::RemoveKeyframe {
                segment: id,
                offset_us: 0,
                property: KeyframeProperty::Volume,
            },
        ),
        Error::MissingKeyframe,
    );
    expect_err(
        run(
            &mut cut,
            Command::RemoveKeyframe {
                segment: id,
                offset_us: -1,
                property: KeyframeProperty::Opacity,
            },
        ),
        Error::NegativeTime,
    );
    let missing = SegmentId::new(99).expect("id");
    expect_err(
        run(
            &mut cut,
            Command::AddKeyframe {
                segment: missing,
                keyframe,
            },
        ),
        Error::UnknownSegment(missing),
    );
    let kept = cut
        .project
        .timeline()
        .segment(id)
        .ok_or(Error::UnknownSegment(id))?;
    assert_eq!(kept.keyframes(), &[keyframe]);
    Ok(())
}

fn decorated(cut: &Cut, asset: AssetId) -> Result<Segment, Error> {
    let stored = cut.project.asset(asset).ok_or(Error::UnknownAsset(asset))?;
    let duration = stored.duration().ok_or(Error::ImageSource)?;
    let source = TimeRange::new(
        Timestamp::new(0, duration.timescale),
        Timestamp::new(30_000, duration.timescale),
    )?;
    let mut segment = Segment::media(stored, source, 2_000_000, Channels::Both, Speed::new(2, 1)?)?;
    segment.set_visible(false);
    segment.set_volume(0.25)?;
    segment.set_transform(Transform::new(
        (0.1, -0.2),
        (2.0, 3.0),
        15.0,
        0.4,
        true,
        false,
    )?);
    segment.push_keyframe(Keyframe::new(100_000, KeyframeValue::Opacity(0.2))?)?;
    Ok(segment)
}

#[test]
fn a_copy_keeps_the_clip_and_lands_where_it_ended() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    run(&mut cut, Command::SetMagnet(false))?;
    let segment = decorated(&cut, asset)?;
    let transform = segment.transform();
    let keyframes = segment.keyframes().to_vec();
    let source = segment.source();
    let main = cut.main;
    let id = segment_id(run(
        &mut cut,
        Command::Place {
            track: Some(main),
            segment,
        },
    )?);
    let original = cut
        .project
        .timeline()
        .segment(id)
        .ok_or(Error::UnknownSegment(id))?
        .clone();
    let copy = original.duplicate(original.timeline().end().value)?;
    let placed = same_track(run(
        &mut cut,
        Command::Place {
            track: Some(main),
            segment: copy,
        },
    )?);
    assert_ne!(placed, id);
    assert_eq!(start(&cut, id)?, 2_000_000);
    assert_eq!(start(&cut, placed)?, 2_500_000);
    let kept = cut
        .project
        .timeline()
        .segment(id)
        .ok_or(Error::UnknownSegment(id))?;
    assert!(!kept.visible());
    assert_eq!(kept.volume(), 0.25);
    let next = cut
        .project
        .timeline()
        .segment(placed)
        .ok_or(Error::UnknownSegment(placed))?;
    assert_eq!(next.asset_id(), asset);
    assert_eq!(next.source(), source);
    assert_eq!(next.channels(), Channels::Both);
    assert_eq!(next.speed(), Speed::new(2, 1)?);
    assert!(!next.visible());
    assert_eq!(next.volume(), 0.25);
    assert_eq!(next.transform(), transform);
    assert_eq!(next.keyframes(), keyframes.as_slice());
    Ok(())
}

#[test]
fn a_copy_still_places_after_the_original_is_removed() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    run(&mut cut, Command::SetMagnet(false))?;
    let id = place_main(&mut cut, asset, 0, 30_000)?;
    let original = cut
        .project
        .timeline()
        .segment(id)
        .ok_or(Error::UnknownSegment(id))?
        .clone();
    let copy = original.duplicate(original.timeline().end().value)?;
    run(&mut cut, Command::RemoveSegment(id))?;
    assert!(cut.project.timeline().segment(id).is_none());
    let main = cut.main;
    let placed = same_track(run(
        &mut cut,
        Command::Place {
            track: Some(main),
            segment: copy,
        },
    )?);
    assert_ne!(placed, id);
    assert_eq!(start(&cut, placed)?, 1_000_000);
    let next = cut
        .project
        .timeline()
        .segment(placed)
        .ok_or(Error::UnknownSegment(placed))?;
    assert_eq!(next.source(), original.source());
    assert_eq!(next.asset_id(), asset);
    Ok(())
}

#[test]
fn a_taken_time_opens_a_track_for_the_copy() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    run(&mut cut, Command::SetMagnet(false))?;
    let first = place_main(&mut cut, asset, 0, 30_000)?;
    let second = place_main(&mut cut, asset, 1_500_000, 30_000)?;
    let original = cut
        .project
        .timeline()
        .segment(first)
        .ok_or(Error::UnknownSegment(first))?
        .clone();
    let copy = original.duplicate(original.timeline().end().value)?;
    let main = cut.main;
    let (placed, track) = opened(run(
        &mut cut,
        Command::Place {
            track: Some(main),
            segment: copy,
        },
    )?);
    assert_ne!(track, main);
    assert_eq!(start(&cut, placed)?, 1_000_000);
    assert_eq!(start(&cut, first)?, 0);
    assert_eq!(start(&cut, second)?, 1_500_000);
    assert_eq!(on_track(&cut, first), main);
    assert_eq!(on_track(&cut, placed), track);
    Ok(())
}

#[test]
fn magnet_packs_the_copy_after_the_main_clip() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let id = place_main(&mut cut, asset, 5_000_000, 30_000)?;
    assert_eq!(start(&cut, id)?, 0);
    let original = cut
        .project
        .timeline()
        .segment(id)
        .ok_or(Error::UnknownSegment(id))?
        .clone();
    let copy = original.duplicate(original.timeline().end().value)?;
    let main = cut.main;
    let placed = same_track(run(
        &mut cut,
        Command::Place {
            track: Some(main),
            segment: copy,
        },
    )?);
    assert_eq!(start(&cut, id)?, 0);
    assert_eq!(start(&cut, placed)?, 1_000_000);
    assert_eq!(cut.project.duration().value, 2_000_000);
    Ok(())
}

#[test]
fn duplicating_a_main_clip_leaves_its_overlay() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let main_id = place_main(&mut cut, asset, 0, 30_000)?;
    let (overlay, overlay_track) = open_picture(&mut cut, asset, 100, 30_000)?;
    let original = cut
        .project
        .timeline()
        .segment(main_id)
        .ok_or(Error::UnknownSegment(main_id))?
        .clone();
    let copy = original.duplicate(original.timeline().end().value)?;
    let main = cut.main;
    let placed = same_track(run(
        &mut cut,
        Command::Place {
            track: Some(main),
            segment: copy,
        },
    )?);
    assert_eq!(start(&cut, main_id)?, 0);
    assert_eq!(start(&cut, placed)?, 1_000_000);
    assert_eq!(start(&cut, overlay)?, 100);
    assert_eq!(on_track(&cut, overlay), overlay_track);
    Ok(())
}

#[test]
fn volume_sets_the_gain_and_undo_puts_it_back() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let id = place_main(&mut cut, asset, 0, 30_000)?;
    let keyframe = Keyframe::new(0, KeyframeValue::Volume(0.5))?;
    run(
        &mut cut,
        Command::AddKeyframe {
            segment: id,
            keyframe,
        },
    )?;
    assert_eq!(
        run(
            &mut cut,
            Command::SetVolume {
                segment: id,
                volume: 0.25,
            },
        )?,
        Outcome::Unchanged
    );
    let placed = cut
        .project
        .timeline()
        .segment(id)
        .ok_or(Error::UnknownSegment(id))?;
    assert_eq!(placed.volume(), 0.25);
    assert!(placed.visible());
    assert_eq!(placed.timeline().start().value, 0);
    assert_eq!(placed.keyframes(), &[keyframe]);
    cut.history.undo(&mut cut.project)?;
    let restored = cut
        .project
        .timeline()
        .segment(id)
        .ok_or(Error::UnknownSegment(id))?;
    assert_eq!(restored.volume(), 1.0);
    cut.history.redo(&mut cut.project)?;
    let again = cut
        .project
        .timeline()
        .segment(id)
        .ok_or(Error::UnknownSegment(id))?;
    assert_eq!(again.volume(), 0.25);
    let missing = SegmentId::new(99).expect("id");
    expect_err(
        run(
            &mut cut,
            Command::SetVolume {
                segment: missing,
                volume: 0.0,
            },
        ),
        Error::UnknownSegment(missing),
    );
    assert_eq!(
        cut.project
            .timeline()
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .volume(),
        0.25
    );
    Ok(())
}

#[test]
fn a_bad_volume_changes_nothing() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let id = place_main(&mut cut, asset, 0, 30_000)?;
    run(
        &mut cut,
        Command::SetVolume {
            segment: id,
            volume: 0.25,
        },
    )?;
    expect_err(
        run(
            &mut cut,
            Command::SetVolume {
                segment: id,
                volume: -0.1,
            },
        ),
        Error::NegativeVolume,
    );
    expect_err(
        run(
            &mut cut,
            Command::SetVolume {
                segment: id,
                volume: f64::NAN,
            },
        ),
        Error::NonFinite,
    );
    expect_err(
        run(
            &mut cut,
            Command::SetVolume {
                segment: id,
                volume: f64::INFINITY,
            },
        ),
        Error::NonFinite,
    );
    let kept = cut
        .project
        .timeline()
        .segment(id)
        .ok_or(Error::UnknownSegment(id))?;
    assert_eq!(kept.volume(), 0.25);
    assert!(kept.visible());
    Ok(())
}

#[test]
fn visible_hides_the_segment_and_undo_shows_it() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let id = place_main(&mut cut, asset, 0, 30_000)?;
    assert_eq!(
        run(
            &mut cut,
            Command::SetVisible {
                segment: id,
                visible: false,
            },
        )?,
        Outcome::Unchanged
    );
    let placed = cut
        .project
        .timeline()
        .segment(id)
        .ok_or(Error::UnknownSegment(id))?;
    assert!(!placed.visible());
    assert_eq!(placed.volume(), 1.0);
    assert_eq!(placed.timeline().start().value, 0);
    assert_eq!(on_track(&cut, id), cut.main);
    cut.history.undo(&mut cut.project)?;
    assert!(
        cut.project
            .timeline()
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .visible()
    );
    cut.history.redo(&mut cut.project)?;
    assert!(
        !cut.project
            .timeline()
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .visible()
    );
    let missing = SegmentId::new(99).expect("id");
    expect_err(
        run(
            &mut cut,
            Command::SetVisible {
                segment: missing,
                visible: false,
            },
        ),
        Error::UnknownSegment(missing),
    );
    Ok(())
}

#[test]
fn easing_changes_the_blend_and_undo_restores_it() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let id = place_main(&mut cut, asset, 0, 30_000)?;
    run(
        &mut cut,
        Command::AddKeyframe {
            segment: id,
            keyframe: Keyframe::new(1_000_000, KeyframeValue::Opacity(1.0))?,
        },
    )?;
    run(
        &mut cut,
        Command::AddKeyframe {
            segment: id,
            keyframe: Keyframe::new(0, KeyframeValue::Opacity(0.0))?,
        },
    )?;
    let blend = |cut: &Cut| -> Result<KeyframeValue, Error> {
        cut.project
            .timeline()
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .sample(KeyframeProperty::Opacity, 500_000)
    };
    assert_eq!(blend(&cut)?, KeyframeValue::Opacity(0.5));
    assert_eq!(
        run(
            &mut cut,
            Command::SetEasing {
                segment: id,
                offset_us: 0,
                property: KeyframeProperty::Opacity,
                easing: Easing::EaseIn,
            },
        )?,
        Outcome::Unchanged
    );
    assert_eq!(blend(&cut)?, KeyframeValue::Opacity(0.25));
    cut.history.undo(&mut cut.project)?;
    assert_eq!(blend(&cut)?, KeyframeValue::Opacity(0.5));
    cut.history.redo(&mut cut.project)?;
    assert_eq!(blend(&cut)?, KeyframeValue::Opacity(0.25));
    expect_err(
        run(
            &mut cut,
            Command::SetEasing {
                segment: id,
                offset_us: 1,
                property: KeyframeProperty::Opacity,
                easing: Easing::EaseOut,
            },
        ),
        Error::MissingKeyframe,
    );
    assert_eq!(blend(&cut)?, KeyframeValue::Opacity(0.25));
    expect_err(
        run(
            &mut cut,
            Command::SetEasing {
                segment: id,
                offset_us: -1,
                property: KeyframeProperty::Volume,
                easing: Easing::Linear,
            },
        ),
        Error::NegativeTime,
    );
    let missing = SegmentId::new(99).expect("id");
    expect_err(
        run(
            &mut cut,
            Command::SetEasing {
                segment: missing,
                offset_us: 0,
                property: KeyframeProperty::Opacity,
                easing: Easing::EaseIn,
            },
        ),
        Error::UnknownSegment(missing),
    );
    assert_eq!(blend(&cut)?, KeyframeValue::Opacity(0.25));
    Ok(())
}

#[test]
fn fades_scale_the_gain_and_undo_clears_them() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let id = place_main(&mut cut, asset, 0, 30_000)?;
    let gain = |cut: &Cut, at: i64| -> Result<f64, Error> {
        cut.project
            .timeline()
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .gain(at)
    };
    assert_eq!(gain(&cut, 250_000)?, 1.0);
    assert_eq!(
        run(
            &mut cut,
            Command::SetFades {
                segment: id,
                fade_in: 500_000,
                fade_out: 500_000,
            },
        )?,
        Outcome::Unchanged
    );
    assert_eq!(gain(&cut, 0)?, 0.0);
    assert_eq!(gain(&cut, 250_000)?, 0.5);
    assert_eq!(gain(&cut, 750_000)?, 0.5);
    cut.history.undo(&mut cut.project)?;
    assert_eq!(gain(&cut, 250_000)?, 1.0);
    cut.history.redo(&mut cut.project)?;
    assert_eq!(gain(&cut, 250_000)?, 0.5);
    expect_err(
        run(
            &mut cut,
            Command::SetFades {
                segment: id,
                fade_in: 1_000_001,
                fade_out: 0,
            },
        ),
        Error::FadeOutOfRange,
    );
    expect_err(
        run(
            &mut cut,
            Command::SetFades {
                segment: id,
                fade_in: -1,
                fade_out: 0,
            },
        ),
        Error::NegativeTime,
    );
    assert_eq!(
        cut.project
            .timeline()
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .fade_in(),
        500_000
    );
    let missing = SegmentId::new(99).expect("id");
    expect_err(
        run(
            &mut cut,
            Command::SetFades {
                segment: missing,
                fade_in: 0,
                fade_out: 0,
            },
        ),
        Error::UnknownSegment(missing),
    );
    Ok(())
}

#[test]
fn a_split_keeps_each_fade_on_its_edge() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    run(&mut cut, Command::SetMagnet(false))?;
    let id = place_main(&mut cut, asset, 0, 30_000)?;
    run(
        &mut cut,
        Command::SetFades {
            segment: id,
            fade_in: 600_000,
            fade_out: 100_000,
        },
    )?;
    let right = segment_id(run(
        &mut cut,
        Command::Split {
            segment: id,
            at: 400_000,
        },
    )?);
    let left = cut
        .project
        .timeline()
        .segment(id)
        .ok_or(Error::UnknownSegment(id))?;
    assert_eq!(left.fade_in(), 400_000);
    assert_eq!(left.fade_out(), 0);
    let right = cut
        .project
        .timeline()
        .segment(right)
        .ok_or(Error::UnknownSegment(right))?;
    assert_eq!(right.fade_in(), 200_000);
    assert_eq!(right.fade_out(), 100_000);
    Ok(())
}

#[test]
fn a_faster_clip_clamps_its_fades() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let id = place_main(&mut cut, asset, 0, 30_000)?;
    run(
        &mut cut,
        Command::SetFades {
            segment: id,
            fade_in: 800_000,
            fade_out: 700_000,
        },
    )?;
    run(
        &mut cut,
        Command::SetSpeed {
            segment: id,
            speed: Speed::new(2, 1)?,
        },
    )?;
    let clip = cut
        .project
        .timeline()
        .segment(id)
        .ok_or(Error::UnknownSegment(id))?;
    assert_eq!(clip.timeline().duration().value, 500_000);
    assert_eq!(clip.fade_in(), 500_000);
    assert_eq!(clip.fade_out(), 500_000);
    Ok(())
}

#[test]
fn denoise_turns_on_for_sound_and_undo_turns_it_off() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let id = place_main(&mut cut, asset, 0, 30_000)?;
    assert_eq!(
        run(
            &mut cut,
            Command::SetDenoise {
                segment: id,
                on: true,
            },
        )?,
        Outcome::Unchanged
    );
    assert!(
        cut.project
            .timeline()
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .denoise()
    );
    cut.history.undo(&mut cut.project)?;
    assert!(
        !cut.project
            .timeline()
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .denoise()
    );
    cut.history.redo(&mut cut.project)?;
    assert!(
        cut.project
            .timeline()
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .denoise()
    );
    let right = segment_id(run(
        &mut cut,
        Command::Split {
            segment: id,
            at: 500_000,
        },
    )?);
    assert!(
        cut.project
            .timeline()
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .denoise()
    );
    assert!(
        cut.project
            .timeline()
            .segment(right)
            .ok_or(Error::UnknownSegment(right))?
            .denoise()
    );

    let picture = media(&cut.project, asset, 2_000_000, 30_000, Channels::Picture)?;
    let main = cut.main;
    let picture = segment_id(run(
        &mut cut,
        Command::Place {
            track: Some(main),
            segment: picture,
        },
    )?);
    expect_err(
        run(
            &mut cut,
            Command::SetDenoise {
                segment: picture,
                on: true,
            },
        ),
        Error::NoSound,
    );
    assert!(
        !cut.project
            .timeline()
            .segment(picture)
            .ok_or(Error::UnknownSegment(picture))?
            .denoise()
    );
    let missing = SegmentId::new(99).expect("id");
    expect_err(
        run(
            &mut cut,
            Command::SetDenoise {
                segment: missing,
                on: true,
            },
        ),
        Error::UnknownSegment(missing),
    );
    Ok(())
}

#[test]
fn keep_pitch_turns_off_and_a_speed_change_leaves_it() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let id = place_main(&mut cut, asset, 0, 30_000)?;
    assert!(
        cut.project
            .timeline()
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .keep_pitch()
    );
    assert_eq!(
        run(
            &mut cut,
            Command::SetKeepPitch {
                segment: id,
                keep: false,
            },
        )?,
        Outcome::Unchanged
    );
    cut.history.undo(&mut cut.project)?;
    assert!(
        cut.project
            .timeline()
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .keep_pitch()
    );
    cut.history.redo(&mut cut.project)?;
    run(
        &mut cut,
        Command::SetSpeed {
            segment: id,
            speed: Speed::new(2, 1)?,
        },
    )?;
    let sped = cut
        .project
        .timeline()
        .segment(id)
        .ok_or(Error::UnknownSegment(id))?;
    assert_eq!(sped.timeline().duration().value, 500_000);
    assert!(!sped.keep_pitch());
    let right = segment_id(run(
        &mut cut,
        Command::Split {
            segment: id,
            at: 250_000,
        },
    )?);
    assert!(
        !cut.project
            .timeline()
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .keep_pitch()
    );
    assert!(
        !cut.project
            .timeline()
            .segment(right)
            .ok_or(Error::UnknownSegment(right))?
            .keep_pitch()
    );

    let picture = media(&cut.project, asset, 2_000_000, 30_000, Channels::Picture)?;
    let main = cut.main;
    let picture = segment_id(run(
        &mut cut,
        Command::Place {
            track: Some(main),
            segment: picture,
        },
    )?);
    expect_err(
        run(
            &mut cut,
            Command::SetKeepPitch {
                segment: picture,
                keep: false,
            },
        ),
        Error::NoSound,
    );
    assert!(
        cut.project
            .timeline()
            .segment(picture)
            .ok_or(Error::UnknownSegment(picture))?
            .keep_pitch()
    );
    let missing = SegmentId::new(99).expect("id");
    expect_err(
        run(
            &mut cut,
            Command::SetKeepPitch {
                segment: missing,
                keep: false,
            },
        ),
        Error::UnknownSegment(missing),
    );
    Ok(())
}

#[test]
fn crop_insets_the_picture_and_undo_restores_the_whole() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let id = place_main(&mut cut, asset, 0, 30_000)?;
    assert!(
        cut.project
            .timeline()
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .crop()
            .is_whole()
    );
    let crop = Crop::new(0.1, 0.2, 0.0, 0.25)?;
    assert_eq!(
        run(&mut cut, Command::SetCrop { segment: id, crop },)?,
        Outcome::Unchanged
    );
    cut.history.undo(&mut cut.project)?;
    assert!(
        cut.project
            .timeline()
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .crop()
            .is_whole()
    );
    cut.history.redo(&mut cut.project)?;
    run(
        &mut cut,
        Command::SetSpeed {
            segment: id,
            speed: Speed::new(2, 1)?,
        },
    )?;
    let sped = cut
        .project
        .timeline()
        .segment(id)
        .ok_or(Error::UnknownSegment(id))?;
    assert_eq!(sped.timeline().duration().value, 500_000);
    assert_eq!(sped.crop(), crop);
    let right = segment_id(run(
        &mut cut,
        Command::Split {
            segment: id,
            at: 250_000,
        },
    )?);
    assert_eq!(
        cut.project
            .timeline()
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .crop(),
        crop
    );
    assert_eq!(
        cut.project
            .timeline()
            .segment(right)
            .ok_or(Error::UnknownSegment(right))?
            .crop(),
        crop
    );

    let sound = media(&cut.project, asset, 2_000_000, 30_000, Channels::Sound)?;
    let sound = segment_id(run(
        &mut cut,
        Command::Place {
            track: None,
            segment: sound,
        },
    )?);
    expect_err(
        run(
            &mut cut,
            Command::SetCrop {
                segment: sound,
                crop,
            },
        ),
        Error::NoPicture,
    );
    assert!(
        cut.project
            .timeline()
            .segment(sound)
            .ok_or(Error::UnknownSegment(sound))?
            .crop()
            .is_whole()
    );
    run(
        &mut cut,
        Command::SetCrop {
            segment: sound,
            crop: Crop::WHOLE,
        },
    )?;
    let missing = SegmentId::new(99).expect("id");
    expect_err(
        run(
            &mut cut,
            Command::SetCrop {
                segment: missing,
                crop,
            },
        ),
        Error::UnknownSegment(missing),
    );
    Ok(())
}

#[test]
fn blend_sets_the_mode_and_undo_restores_normal() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let id = place_main(&mut cut, asset, 0, 30_000)?;
    assert_eq!(
        cut.project
            .timeline()
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .blend(),
        Blend::Normal
    );
    assert_eq!(
        run(
            &mut cut,
            Command::SetBlend {
                segment: id,
                blend: Blend::Multiply,
            },
        )?,
        Outcome::Unchanged
    );
    cut.history.undo(&mut cut.project)?;
    assert_eq!(
        cut.project
            .timeline()
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .blend(),
        Blend::Normal
    );
    cut.history.redo(&mut cut.project)?;
    run(
        &mut cut,
        Command::SetSpeed {
            segment: id,
            speed: Speed::new(2, 1)?,
        },
    )?;
    let sped = cut
        .project
        .timeline()
        .segment(id)
        .ok_or(Error::UnknownSegment(id))?;
    assert_eq!(sped.timeline().duration().value, 500_000);
    assert_eq!(sped.blend(), Blend::Multiply);
    let right = segment_id(run(
        &mut cut,
        Command::Split {
            segment: id,
            at: 250_000,
        },
    )?);
    assert_eq!(
        cut.project
            .timeline()
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .blend(),
        Blend::Multiply
    );
    assert_eq!(
        cut.project
            .timeline()
            .segment(right)
            .ok_or(Error::UnknownSegment(right))?
            .blend(),
        Blend::Multiply
    );

    let sound = media(&cut.project, asset, 2_000_000, 30_000, Channels::Sound)?;
    let sound = segment_id(run(
        &mut cut,
        Command::Place {
            track: None,
            segment: sound,
        },
    )?);
    expect_err(
        run(
            &mut cut,
            Command::SetBlend {
                segment: sound,
                blend: Blend::Screen,
            },
        ),
        Error::NoPicture,
    );
    assert_eq!(
        cut.project
            .timeline()
            .segment(sound)
            .ok_or(Error::UnknownSegment(sound))?
            .blend(),
        Blend::Normal
    );
    run(
        &mut cut,
        Command::SetBlend {
            segment: sound,
            blend: Blend::Normal,
        },
    )?;
    let missing = SegmentId::new(99).expect("id");
    expect_err(
        run(
            &mut cut,
            Command::SetBlend {
                segment: missing,
                blend: Blend::Add,
            },
        ),
        Error::UnknownSegment(missing),
    );
    Ok(())
}

#[test]
fn detach_audio_adds_a_sound_segment_of_the_same_asset() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    let id = place_main(&mut cut, asset, 0, 30_000)?;
    let crop = Crop::new(0.1, 0.0, 0.0, 0.0)?;
    let transform = Transform::new((0.2, 0.0), (1.0, 1.0), 0.0, 1.0, false, false)?;
    run(
        &mut cut,
        Command::SetVolume {
            segment: id,
            volume: 0.25,
        },
    )?;
    run(
        &mut cut,
        Command::SetFades {
            segment: id,
            fade_in: 100_000,
            fade_out: 50_000,
        },
    )?;
    run(
        &mut cut,
        Command::SetDenoise {
            segment: id,
            on: true,
        },
    )?;
    run(
        &mut cut,
        Command::SetKeepPitch {
            segment: id,
            keep: false,
        },
    )?;
    run(&mut cut, Command::SetCrop { segment: id, crop })?;
    run(
        &mut cut,
        Command::SetBlend {
            segment: id,
            blend: Blend::Multiply,
        },
    )?;
    run(
        &mut cut,
        Command::SetTransform {
            segment: id,
            transform,
        },
    )?;
    run(
        &mut cut,
        Command::SetVisible {
            segment: id,
            visible: false,
        },
    )?;
    run(
        &mut cut,
        Command::AddKeyframe {
            segment: id,
            keyframe: Keyframe::new(0, KeyframeValue::Opacity(0.2))?,
        },
    )?;
    run(
        &mut cut,
        Command::AddKeyframe {
            segment: id,
            keyframe: Keyframe::new(100_000, KeyframeValue::Volume(0.5))?
                .with_easing(Easing::EaseIn),
        },
    )?;

    let (sound, track) = opened(run(&mut cut, Command::DetachAudio { segment: id })?);
    assert_eq!(
        cut.project
            .timeline()
            .track(track)
            .ok_or(Error::UnknownTrack(track))?
            .kind(),
        TrackKind::Audio
    );
    assert_eq!(kinds(&cut), vec![TrackKind::Audio, TrackKind::Video]);
    let picture = cut
        .project
        .timeline()
        .segment(id)
        .ok_or(Error::UnknownSegment(id))?;
    let heard = cut
        .project
        .timeline()
        .segment(sound)
        .ok_or(Error::UnknownSegment(sound))?;
    assert_eq!(picture.channels(), Channels::Picture);
    assert_eq!(picture.volume(), 1.0);
    assert_eq!(picture.fade_in(), 0);
    assert_eq!(picture.fade_out(), 0);
    assert!(!picture.denoise());
    assert!(picture.keep_pitch());
    assert_eq!(picture.crop(), crop);
    assert_eq!(picture.blend(), Blend::Multiply);
    assert_eq!(picture.transform(), transform);
    assert!(!picture.visible());
    assert_eq!(picture.keyframes().len(), 1);
    assert_eq!(picture.keyframes()[0].property(), KeyframeProperty::Opacity);
    assert_eq!(heard.channels(), Channels::Sound);
    assert_eq!(heard.asset_id(), asset);
    assert_eq!(heard.source(), picture.source());
    assert_eq!(heard.timeline(), picture.timeline());
    assert_eq!(heard.speed(), picture.speed());
    assert_eq!(heard.volume(), 0.25);
    assert_eq!(heard.fade_in(), 100_000);
    assert_eq!(heard.fade_out(), 50_000);
    assert!(heard.denoise());
    assert!(!heard.keep_pitch());
    assert!(heard.crop().is_whole());
    assert_eq!(heard.blend(), Blend::Normal);
    assert_eq!(heard.transform(), Transform::IDENTITY);
    assert!(!heard.visible());
    assert_eq!(heard.keyframes().len(), 1);
    assert_eq!(heard.keyframes()[0].easing(), Easing::EaseIn);
    let loaded = Project::from_json(&cut.project.to_json()?)?;
    assert_eq!(
        loaded
            .timeline()
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .channels(),
        Channels::Picture
    );
    assert_eq!(
        loaded
            .timeline()
            .segment(sound)
            .ok_or(Error::UnknownSegment(sound))?
            .channels(),
        Channels::Sound
    );

    cut.history.undo(&mut cut.project)?;
    assert_eq!(
        cut.project
            .timeline()
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .channels(),
        Channels::Both
    );
    assert_eq!(
        cut.project
            .timeline()
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .volume(),
        0.25
    );
    assert!(cut.project.timeline().segment(sound).is_none());
    assert_eq!(kinds(&cut), vec![TrackKind::Video]);
    cut.history.redo(&mut cut.project)?;
    assert_eq!(
        cut.project
            .timeline()
            .segment(sound)
            .ok_or(Error::UnknownSegment(sound))?
            .channels(),
        Channels::Sound
    );
    expect_err(
        run(&mut cut, Command::DetachAudio { segment: id }),
        Error::NoSound,
    );
    expect_err(
        run(&mut cut, Command::DetachAudio { segment: sound }),
        Error::NoPicture,
    );
    let missing = SegmentId::new(99).expect("id");
    expect_err(
        run(&mut cut, Command::DetachAudio { segment: missing }),
        Error::UnknownSegment(missing),
    );
    assert_eq!(
        cut.project
            .timeline()
            .segment(id)
            .ok_or(Error::UnknownSegment(id))?
            .channels(),
        Channels::Picture
    );
    assert_eq!(
        cut.project
            .timeline()
            .segment(sound)
            .ok_or(Error::UnknownSegment(sound))?
            .channels(),
        Channels::Sound
    );
    Ok(())
}

#[test]
fn a_free_audio_track_keeps_the_sound_and_a_taken_one_opens_another() -> Result<(), Error> {
    let (mut cut, asset) = load()?;
    run(&mut cut, Command::SetMagnet(false))?;
    let first = place_main(&mut cut, asset, 1_000_000, 30_000)?;
    let (sound, audio) = opened(run(&mut cut, Command::DetachAudio { segment: first })?);
    assert_eq!(start(&cut, first)?, 1_000_000);
    assert_eq!(start(&cut, sound)?, 1_000_000);

    let later = place_main(&mut cut, asset, 3_000_000, 30_000)?;
    let later_sound = same_track(run(&mut cut, Command::DetachAudio { segment: later })?);
    assert_eq!(on_track(&cut, later_sound), audio);
    assert_eq!(start(&cut, later_sound)?, 3_000_000);
    assert_eq!(kinds(&cut), vec![TrackKind::Audio, TrackKind::Video]);

    run(
        &mut cut,
        Command::Move {
            segment: first,
            track: None,
            start: 1_500_000,
        },
    )?;
    assert_eq!(start(&cut, first)?, 1_500_000);
    assert_eq!(start(&cut, sound)?, 1_500_000);
    assert_eq!(start(&cut, later)?, 3_000_000);
    assert_eq!(start(&cut, later_sound)?, 3_000_000);

    let (overlay, _) = open_picture(&mut cut, asset, 1_500_000, 30_000)?;
    let (_, second) = opened(run(&mut cut, Command::DetachAudio { segment: overlay })?);
    assert_ne!(second, audio);
    assert_eq!(
        cut.project
            .timeline()
            .track(second)
            .ok_or(Error::UnknownTrack(second))?
            .kind(),
        TrackKind::Audio
    );
    assert_eq!(
        kinds(&cut),
        vec![
            TrackKind::Audio,
            TrackKind::Audio,
            TrackKind::Video,
            TrackKind::Video
        ]
    );
    Ok(())
}
