use std::collections::HashMap;

use super::clip::Clip;
use super::project::Project;
use super::rational::Rational;
use super::rational_time::RationalTime;
use super::sequence::Sequence;
use super::time_range::TimeRange;
use super::track::{Track, TrackKind};

fn rt(value: i32, fps: &Rational) -> RationalTime {
    RationalTime {
        value,
        rate: fps.clone(),
    }
}

fn clip(
    id: &str,
    name: &str,
    timeline_start: i32,
    src_start: i32,
    duration: i32,
    fps: &Rational,
) -> Clip {
    Clip {
        id: id.into(),
        name: name.into(),
        timeline_start: rt(timeline_start, fps),
        source_range: TimeRange {
            start: rt(src_start, fps),
            duration: rt(duration, fps),
        },
    }
}

fn track(
    id: &str,
    name: &str,
    kind: TrackKind,
    kind_index: usize,
    clips: Vec<Clip>,
) -> Track {
    let clip_order: Vec<String> = clips.iter().map(|c| c.id.clone()).collect();
    let clips: HashMap<String, Clip> = clips.into_iter().map(|c| (c.id.clone(), c)).collect();
    Track {
        id: id.into(),
        name: name.into(),
        kind,
        color: kind.palette_color(kind_index),
        clip_order,
        clips,
    }
}

/// Demo project — same content that used to live in `editor-store.slint`.
pub fn sample_project() -> Project {
    let fps = Rational { num: 24, den: 1 };

    // Track_order invariant: every Video lane precedes every Audio
    // lane. This is what makes "audio is always at the bottom" hold
    // in the timeline iteration. The `kind_index` argument picks the
    // lane's color from its kind's palette so adjacent lanes of the
    // same kind never share a color in the sample.
    let tracks = vec![
        track(
            "1",
            "V1",
            TrackKind::Video,
            0,
            vec![clip("1", "Clip 1", 10, 10, 100, &fps)],
        ),
        track(
            "2",
            "V2",
            TrackKind::Video,
            1,
            vec![
                clip("2", "Clip 2", 0, 0, 80, &fps),
                clip("3", "Clip 3", 120, 0, 60, &fps),
            ],
        ),
        track(
            "3",
            "V3",
            TrackKind::Video,
            2,
            vec![clip("4", "Clip 4", 50, 0, 90, &fps)],
        ),
        track(
            "4",
            "A1",
            TrackKind::Audio,
            0,
            vec![
                clip("5", "Clip 5", 30, 5, 70, &fps),
                clip("6", "Clip 6", 150, 0, 45, &fps),
            ],
        ),
        track(
            "5",
            "A2",
            TrackKind::Audio,
            1,
            vec![clip("7", "Clip 7", 0, 0, 200, &fps)],
        ),
    ];

    let track_order: Vec<String> = tracks.iter().map(|t| t.id.clone()).collect();
    let tracks: HashMap<String, Track> = tracks.into_iter().map(|t| (t.id.clone(), t)).collect();

    let next_track_id = (tracks.len() as u32) + 1;
    Project {
        id: "1".into(),
        title: "Project 1".into(),
        sequence: Sequence {
            id: "1".into(),
            name: "Sequence 1".into(),
            fps,
            drop_frame: false,
            track_order,
            tracks,
            next_track_id,
            width: 1080.0,
            height: 1920.0,
        },
    }
}
