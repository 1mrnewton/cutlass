//! Which clip the Auto Captions dialog would transcribe.
//!
//! Resolution reads the same projected `Project` the timeline draws, so the
//! dialog's source line and the job's audio can't disagree. The dialog asks for
//! a display view; `wire_captions` asks for the request the job needs.

use slint::Model;

use crate::auto_captions::AutoCaptionRequest;
use crate::{CaptionAutoSource, Clip, Media, Project, TrackKind};

/// Characters per line the dialog offers, clamped into the model's range.
const MIN_LINE_LENGTH: u16 = 12;
const MAX_LINE_LENGTH: u16 = 120;

/// The clip to transcribe: the selection when it has audio, otherwise whatever
/// sits under the playhead.
///
/// A refusal always explains itself, since "Generate captions" being greyed out
/// with no reason is the worst version of this dialog.
pub fn auto_source(project: Project, selected_clip: &str, playhead_tick: i32) -> CaptionAutoSource {
    let Some((clip, media)) = pick(&project, selected_clip, playhead_tick) else {
        return refusal(if selected_clip.is_empty() {
            "Select a clip with audio, or park the playhead over one"
        } else {
            "That clip has no audio to transcribe"
        });
    };

    if media.is_missing {
        return refusal("This clip's file is missing — relink it first");
    }
    if media.path.is_empty() {
        return refusal("This clip's file cannot be read");
    }

    CaptionAutoSource {
        clip_id: clip.id,
        name: media.name,
        duration_label: clip.duration_label,
        reason: Default::default(),
    }
}

/// Everything the transcription job needs for `clip_id`, or `None` when the
/// clip or its media has gone (the projection can change between the dialog
/// opening and the button being pressed).
pub fn auto_request(
    project: &Project,
    clip_id: &str,
    max_chars_per_line: i32,
    template: &str,
) -> Option<AutoCaptionRequest> {
    let (clip, media) = clips(project)
        .find(|(clip, _)| clip.id == clip_id)
        .and_then(|(clip, _)| media_of(project, &clip).map(|media| (clip, media)))?;
    if media.is_missing || media.path.is_empty() || !media.has_audio {
        return None;
    }

    let layout = cutlass_models::CaptionLayout {
        max_chars_per_line: max_chars_per_line
            .clamp(i32::from(MIN_LINE_LENGTH), i32::from(MAX_LINE_LENGTH))
            as u16,
        ..cutlass_models::CaptionLayout::default()
    };
    Some(AutoCaptionRequest {
        path: std::path::PathBuf::from(media.path.as_str()),
        media: clip.media_id.to_string(),
        name: media.name.to_string(),
        start_tick: i64::from(clip.timeline_start.value),
        duration_ticks: i64::from(clip.source_range.duration.value),
        source_in_seconds: f64::from(clip.source_in_s),
        speed: if clip.speed > 0.0 {
            f64::from(clip.speed)
        } else {
            1.0
        },
        rate: cutlass_models::Rational::new(project.sequence.fps.num, project.sequence.fps.den),
        // `ggml-base.en` is English-only, so pinning the language beats letting
        // detection wander on a model that has one answer anyway.
        language: Some("en".to_owned()),
        template: template.to_owned(),
        layout,
    })
}

/// The selected clip when it can be transcribed, else the best clip under the
/// playhead: the main video lane first (what the viewer is watching), then any
/// other visual lane, then audio.
fn pick(project: &Project, selected_clip: &str, playhead_tick: i32) -> Option<(Clip, Media)> {
    if !selected_clip.is_empty()
        && let Some((clip, _)) = clips(project).find(|(clip, _)| clip.id == selected_clip)
        && let Some(media) = transcribable(project, &clip)
    {
        return Some((clip, media));
    }

    clips(project)
        .filter(|(clip, _)| covers(clip, playhead_tick))
        .filter_map(|(clip, lane)| {
            let media = transcribable(project, &clip)?;
            let rank = match (lane.is_main, lane.kind) {
                (true, _) => 0,
                (_, TrackKind::Video) => 1,
                (_, TrackKind::Audio) => 2,
                _ => return None,
            };
            Some((rank, clip, media))
        })
        .min_by_key(|(rank, _, _)| *rank)
        .map(|(_, clip, media)| (clip, media))
}

/// The clip's pool entry, when the clip has one with sound in it.
fn transcribable(project: &Project, clip: &Clip) -> Option<Media> {
    media_of(project, clip).filter(|media| media.has_audio)
}

fn media_of(project: &Project, clip: &Clip) -> Option<Media> {
    if clip.media_id.is_empty() {
        return None;
    }
    (0..project.media.row_count())
        .filter_map(|index| project.media.row_data(index))
        .find(|media| media.id == clip.media_id)
}

fn covers(clip: &Clip, tick: i32) -> bool {
    let start = clip.timeline_start.value;
    let end = start.saturating_add(clip.source_range.duration.value);
    tick >= start && tick < end
}

/// What a lane contributes to source ranking — the whole `Track` would be
/// cloned per clip for two fields.
#[derive(Debug, Clone, Copy)]
struct Lane {
    kind: TrackKind,
    is_main: bool,
}

/// Every clip in the sequence with the lane it sits on.
fn clips(project: &Project) -> impl Iterator<Item = (Clip, Lane)> + '_ {
    (0..project.sequence.tracks.row_count())
        .filter_map(|index| project.sequence.tracks.row_data(index))
        .flat_map(|track| {
            let lane = Lane {
                kind: track.kind,
                is_main: track.is_main,
            };
            (0..track.clips.row_count())
                .filter_map(|index| track.clips.row_data(index))
                .map(move |clip| (clip, lane))
                .collect::<Vec<_>>()
        })
}

fn refusal(reason: &str) -> CaptionAutoSource {
    CaptionAutoSource {
        reason: reason.into(),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Rational, RationalTime, Sequence, TimeRange, Track};

    use slint::{ModelRc, SharedString, VecModel};
    use std::rc::Rc;

    fn rate() -> Rational {
        Rational { num: 30, den: 1 }
    }

    fn clip(id: &str, media: &str, start: i32, duration: i32) -> Clip {
        Clip {
            id: SharedString::from(id),
            media_id: SharedString::from(media),
            duration_label: SharedString::from("0:10"),
            speed: 1.0,
            timeline_start: RationalTime {
                value: start,
                rate: rate(),
            },
            source_range: TimeRange {
                start: RationalTime {
                    value: 0,
                    rate: rate(),
                },
                duration: RationalTime {
                    value: duration,
                    rate: rate(),
                },
            },
            ..Default::default()
        }
    }

    fn media(id: &str, has_audio: bool) -> Media {
        Media {
            id: SharedString::from(id),
            name: SharedString::from(format!("{id}.mp4")),
            path: SharedString::from(format!("/tmp/{id}.mp4")),
            has_audio,
            ..Default::default()
        }
    }

    fn project(tracks: Vec<Track>, pool: Vec<Media>) -> Project {
        Project {
            sequence: Sequence {
                fps: rate(),
                tracks: ModelRc::from(Rc::new(VecModel::from(tracks))),
                ..Default::default()
            },
            media: ModelRc::from(Rc::new(VecModel::from(pool))),
            ..Default::default()
        }
    }

    fn track(id: &str, kind: TrackKind, is_main: bool, clips: Vec<Clip>) -> Track {
        Track {
            id: SharedString::from(id),
            kind,
            is_main,
            clips: ModelRc::from(Rc::new(VecModel::from(clips))),
            ..Default::default()
        }
    }

    fn one_clip_project() -> Project {
        project(
            vec![track(
                "v1",
                TrackKind::Video,
                true,
                vec![clip("c1", "m1", 0, 300)],
            )],
            vec![media("m1", true)],
        )
    }

    #[test]
    fn the_selection_wins_when_it_has_audio() {
        let source = auto_source(one_clip_project(), "c1", 900);
        assert_eq!(source.clip_id, "c1");
        assert_eq!(source.name, "m1.mp4");
        assert_eq!(source.reason, "");
    }

    #[test]
    fn the_playhead_picks_the_main_lane_over_an_overlay() {
        let project = project(
            vec![
                track(
                    "v2",
                    TrackKind::Video,
                    false,
                    vec![clip("overlay", "m2", 0, 300)],
                ),
                track(
                    "v1",
                    TrackKind::Video,
                    true,
                    vec![clip("main", "m1", 0, 300)],
                ),
            ],
            vec![media("m1", true), media("m2", true)],
        );
        assert_eq!(auto_source(project, "", 60).clip_id, "main");
    }

    #[test]
    fn a_selection_without_audio_explains_itself() {
        let project = project(
            vec![track(
                "t1",
                TrackKind::Text,
                false,
                vec![clip("title", "", 0, 300)],
            )],
            vec![],
        );
        let source = auto_source(project, "title", 60);
        assert_eq!(source.clip_id, "");
        assert!(source.reason.contains("no audio"), "{}", source.reason);
    }

    #[test]
    fn nothing_under_the_playhead_asks_for_a_selection() {
        let source = auto_source(one_clip_project(), "", 9_000);
        assert_eq!(source.clip_id, "");
        assert!(source.reason.contains("Select a clip"), "{}", source.reason);
    }

    #[test]
    fn a_missing_file_is_named_as_the_blocker() {
        let project = project(
            vec![track(
                "v1",
                TrackKind::Video,
                true,
                vec![clip("c1", "m1", 0, 300)],
            )],
            vec![Media {
                is_missing: true,
                ..media("m1", true)
            }],
        );
        let source = auto_source(project, "c1", 60);
        assert_eq!(source.clip_id, "");
        assert!(source.reason.contains("missing"), "{}", source.reason);
    }

    #[test]
    fn a_request_carries_the_clips_window_and_a_clamped_line_length() {
        let mut clip = clip("c1", "m1", 90, 300);
        clip.source_in_s = 2.5;
        clip.speed = 2.0;
        let project = project(
            vec![track("v1", TrackKind::Video, true, vec![clip])],
            vec![media("m1", true)],
        );

        let request =
            auto_request(&project, "c1", 4, "karaoke_pop").expect("clip is transcribable");
        assert_eq!(request.start_tick, 90);
        assert_eq!(request.duration_ticks, 300);
        assert_eq!(request.source_in_seconds, 2.5);
        assert_eq!(request.speed, 2.0);
        assert_eq!(request.media, "m1");
        assert_eq!(request.template, "karaoke_pop");
        assert_eq!(request.layout.max_chars_per_line, MIN_LINE_LENGTH);
    }

    #[test]
    fn a_request_for_a_vanished_clip_is_refused() {
        assert!(auto_request(&one_clip_project(), "gone", 42, "clean").is_none());
    }
}
