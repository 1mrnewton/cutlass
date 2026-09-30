//! Pure caption lookups for the inspector's caption sheet and cue list.
//!
//! These read the projected Slint `Sequence` (no engine access) the same way
//! [`crate::inspector`] does, so the cue list always agrees with what the
//! timeline is showing. Every mutation lives in `wire_captions.rs`.

mod auto;

use slint::{Model, ModelRc};

pub use auto::{auto_request, auto_source};

use crate::{CaptionCueRow, CaptionGroupView, Clip, Sequence, TrackKind};

/// Transcription confidence below which a cue is flagged for review. Whisper's
/// per-segment averages sit near 0.9 on clean speech, so 0.6 marks the lines
/// worth a second look without lighting up the whole list.
const LOW_CONFIDENCE: f32 = 0.6;

/// The caption group with this id, or a default-valued group (empty `id`) when
/// there is none — the caption inspector tests `group.id != ""`.
pub fn group(sequence: Sequence, group_id: &str) -> CaptionGroupView {
    if group_id.is_empty() {
        return CaptionGroupView::default();
    }
    (0..sequence.caption_groups.row_count())
        .filter_map(|i| sequence.caption_groups.row_data(i))
        .find(|group| group.id == group_id)
        .unwrap_or_default()
}

/// Every cue of `group_id`, in timeline order.
pub fn cues(sequence: Sequence, group_id: &str) -> ModelRc<CaptionCueRow> {
    if group_id.is_empty() {
        return ModelRc::default();
    }
    let fps = sequence.fps.clone();
    let mut cues: Vec<CaptionCueRow> = text_clips(&sequence)
        .filter(|clip| clip.caption_group == group_id)
        .map(|clip| cue_row(&clip, fps.num, fps.den))
        .collect();
    cues.sort_by_key(|cue| (cue.start_tick, cue.number));
    ModelRc::from(std::rc::Rc::new(slint::VecModel::from(cues)))
}

/// Clips on every text lane. Captions only ever live on text lanes (the model
/// enforces it), so this is the whole search space for a cue lookup.
fn text_clips(sequence: &Sequence) -> impl Iterator<Item = Clip> + '_ {
    (0..sequence.tracks.row_count())
        .filter_map(|i| sequence.tracks.row_data(i))
        .filter(|track| track.kind == TrackKind::Text)
        .flat_map(|track| {
            (0..track.clips.row_count())
                .filter_map(move |i| track.clips.row_data(i))
                .collect::<Vec<_>>()
        })
}

fn cue_row(clip: &Clip, fps_num: i32, fps_den: i32) -> CaptionCueRow {
    let start = clip.timeline_start.value;
    let duration = clip.source_range.duration.value;
    CaptionCueRow {
        clip_id: clip.id.clone(),
        number: clip.caption_index,
        text: clip.text_content.clone(),
        start_tick: start,
        duration_ticks: duration,
        timecode: format!(
            "{} → {}",
            clock(start, fps_num, fps_den),
            clock(start + duration, fps_num, fps_den)
        )
        .into(),
        speaker: clip.caption_speaker.clone(),
        confidence: clip.caption_confidence,
        // Zero means "unknown" (hand-typed or imported), which is not the same
        // as a weak transcription.
        low_confidence: clip.caption_confidence > 0.0 && clip.caption_confidence < LOW_CONFIDENCE,
        has_words: clip.caption_has_words,
        style_override: clip.caption_style_override,
    }
}

/// A tick as `m:ss.cs` — subtitle-style, not SMPTE: cue lists are read as
/// speech timings, and hundredths carry more information here than a frame
/// index does.
fn clock(tick: i32, fps_num: i32, fps_den: i32) -> String {
    if fps_num <= 0 || fps_den <= 0 {
        return "0:00.00".to_owned();
    }
    let hundredths = i64::from(tick) * 100 * i64::from(fps_den) / (i64::from(fps_num));
    let (hundredths, sign) = if hundredths < 0 {
        (-hundredths, "-")
    } else {
        (hundredths, "")
    };
    let minutes = hundredths / 6000;
    let seconds = (hundredths / 100) % 60;
    let centis = hundredths % 100;
    format!("{sign}{minutes}:{seconds:02}.{centis:02}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Rational, RationalTime, TimeRange, Track};

    use slint::{SharedString, VecModel};
    use std::rc::Rc;

    fn rate() -> Rational {
        Rational { num: 30, den: 1 }
    }

    fn cue_clip(id: &str, group: &str, index: i32, start: i32, duration: i32) -> Clip {
        Clip {
            id: SharedString::from(id),
            text_content: SharedString::from(format!("line {index}")),
            caption_group: SharedString::from(group),
            caption_index: index,
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

    fn sequence(clips: Vec<Clip>, groups: Vec<CaptionGroupView>) -> Sequence {
        Sequence {
            fps: rate(),
            tracks: ModelRc::from(Rc::new(VecModel::from(vec![Track {
                id: SharedString::from("t1"),
                kind: TrackKind::Text,
                clips: ModelRc::from(Rc::new(VecModel::from(clips))),
                ..Default::default()
            }]))),
            caption_groups: ModelRc::from(Rc::new(VecModel::from(groups))),
            ..Default::default()
        }
    }

    #[test]
    fn cues_come_back_in_timeline_order_with_subtitle_timecodes() {
        // Deliberately out of order in the model: the timeline's clip vectors
        // are not sorted, so the cue list has to sort them itself.
        let sequence = sequence(
            vec![
                cue_clip("c2", "7", 2, 90, 45),
                cue_clip("c1", "7", 1, 0, 60),
                cue_clip("other", "8", 1, 10, 60),
            ],
            vec![],
        );

        let rows = cues(sequence, "7");
        let rows: Vec<_> = (0..rows.row_count())
            .filter_map(|i| rows.row_data(i))
            .collect();

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].clip_id, "c1");
        assert_eq!(rows[0].timecode, "0:00.00 → 0:02.00");
        assert_eq!(rows[1].clip_id, "c2");
        assert_eq!(rows[1].timecode, "0:03.00 → 0:04.50");
    }

    #[test]
    fn unknown_group_yields_no_cues_and_a_blank_group() {
        let sequence = sequence(vec![cue_clip("c1", "7", 1, 0, 60)], vec![]);
        assert_eq!(cues(sequence.clone(), "9").row_count(), 0);
        assert_eq!(group(sequence.clone(), "9").id, "");
        assert_eq!(cues(sequence, "").row_count(), 0);
    }

    #[test]
    fn only_weak_transcriptions_flag_for_review() {
        let mut typed = cue_clip("c1", "7", 1, 0, 60);
        let mut weak = cue_clip("c2", "7", 2, 60, 60);
        weak.caption_confidence = 0.3;
        typed.caption_confidence = 0.0;

        let rows = cues(sequence(vec![typed, weak], vec![]), "7");
        assert!(!rows.row_data(0).expect("first cue").low_confidence);
        assert!(rows.row_data(1).expect("second cue").low_confidence);
    }
}
