//! A clip's heard words as a caption: which stretch of the file is heard,
//! how its word times land on the timeline, and the caption row.

use cutlass_model::{
    AnimType, Draft, MaterialId, MaterialRef, Micros, SegmentId, TimeRange, TrackAttribute,
    TrackId, TrackType, WordColumns, segment_animations,
};
use cutlass_session::{Request, SubtitleCue};

use crate::lines::heard_lines;

/// The sound a clip plays, and how it maps onto the timeline.
#[derive(Clone, Debug, PartialEq)]
pub struct HeardClip {
    pub source_id: SegmentId,
    /// The stored media path.
    pub path: String,
    /// The stretch of the file the clip plays.
    pub range: TimeRange,
    pub reverse: bool,
    /// The clip has a usable constant speed, so word times scale to its length.
    pub speed_holds: bool,
    /// Where the clip sits on the timeline.
    pub target: TimeRange,
}

/// Why a clip has no caption.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeardMiss {
    UnknownSegment,
    /// The clip is not a picture or a sound.
    NotHeard,
    /// The clip has no file or no length.
    NoSound,
    /// The recognizer heard no words.
    NothingHeard,
    /// A word time overflows the clip's clock.
    BadClock,
    /// The caption is not linked to a clip it was heard from, as an
    /// imported subtitle file is not.
    NoSource,
    /// The clip is not a caption.
    NotCaption,
}

/// The sound of clip `segment_id`.
pub fn heard_clip(draft: &Draft, segment_id: &SegmentId) -> Result<HeardClip, HeardMiss> {
    let segment = draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .find(|segment| &segment.id == segment_id)
        .ok_or(HeardMiss::UnknownSegment)?;
    let path = match draft.materials.get(segment.material_id.as_str()) {
        Some(MaterialRef::Video(video)) => video.path.clone(),
        Some(MaterialRef::Audio(audio)) => audio.path.clone(),
        _ => return Err(HeardMiss::NotHeard),
    };
    let range = segment
        .source_range
        .clone()
        .filter(|range| range.duration.0 > 0 && range.start.0 >= 0)
        .unwrap_or_else(|| TimeRange::new(0, segment.target.duration.0));
    if path.is_empty() || range.duration.0 <= 0 || segment.target.duration.0 <= 0 {
        return Err(HeardMiss::NoSound);
    }
    Ok(HeardClip {
        source_id: segment_id.clone(),
        path,
        range,
        reverse: segment.reverse,
        speed_holds: segment.speed.is_finite() && segment.speed > 0.0,
        target: segment.target.clone(),
    })
}

/// Word times, in milliseconds from the start of `clip.range`, moved to
/// milliseconds from the start of the clip on the timeline. A reversed clip
/// flips them; a clip with a speed scales them to its length.
pub fn words_on_timeline(words: WordColumns, clip: &HeardClip) -> Result<WordColumns, HeardMiss> {
    if words.text.is_empty() {
        return Err(HeardMiss::NothingHeard);
    }
    let mut start_time = Vec::with_capacity(words.text.len());
    let mut end_time = Vec::with_capacity(words.text.len());
    for (start_ms, end_ms) in words.start_time.iter().zip(&words.end_time) {
        let (start, end) = timeline_millis(*start_ms, *end_ms, clip)?;
        start_time.push(start);
        end_time.push(end);
    }
    Ok(WordColumns {
        start_time,
        end_time,
        text: words.text,
        rest: words.rest,
    })
}

/// The first unlocked text row with room for `range`.
#[must_use]
pub fn caption_row(draft: &Draft, range: &TimeRange) -> Option<TrackId> {
    draft
        .tracks
        .iter()
        .find(|track| {
            matches!(track.kind, TrackType::Text | TrackType::TextToVideoText)
                && !track.attribute.contains(TrackAttribute::LOCK)
                && track.segments.iter().all(|segment| {
                    segment
                        .target
                        .overlaps(range)
                        .is_ok_and(|overlaps| !overlaps)
                })
        })
        .map(|track| track.id.clone())
}

/// The requests that write `words` (already on the timeline clock) as
/// captions under `clip`, one clip per line of [`heard_lines`], back to back
/// on a free text row or a new one, each linked to `clip`. Applied together
/// they are one undo step. `fresh` makes an unused id from a prefix.
pub fn caption_requests(
    draft: &Draft,
    clip: &HeardClip,
    words: WordColumns,
    fresh: &mut dyn FnMut(&str) -> String,
) -> Result<Vec<Request>, HeardMiss> {
    let lines = lines_on_clip(clip, &words)?;
    let track_id =
        caption_row(draft, &clip.target).unwrap_or_else(|| TrackId::new(fresh("captions")));
    Ok(lines
        .into_iter()
        .map(|(span, words)| Request::AddCaption {
            source_id: clip.source_id.clone(),
            track_id: track_id.clone(),
            segment_id: SegmentId::new(fresh("caption")),
            material_id: MaterialId::new(fresh("caption")),
            words,
            span: Some(Box::new(span)),
        })
        .collect())
}

/// `words` split into lines, each with its place on the timeline.
fn lines_on_clip(
    clip: &HeardClip,
    words: &WordColumns,
) -> Result<Vec<(TimeRange, WordColumns)>, HeardMiss> {
    if words.text.is_empty() {
        return Err(HeardMiss::NothingHeard);
    }
    let start = clip.target.start.0;
    let length_us = clip.target.duration.0;
    let lines = heard_lines(words, length_us / 1_000);
    if lines.is_empty() {
        return Err(HeardMiss::NoSound);
    }
    Ok(lines
        .into_iter()
        .map(|line| {
            let from = line.start_ms * 1_000;
            // A line that reaches the clip's end ends with it, to the
            // microsecond.
            let to = if line.end_ms * 1_000 >= length_us {
                length_us
            } else {
                line.end_ms * 1_000
            };
            (TimeRange::new(start + from, to - from), line.words)
        })
        .collect())
}

/// The clip caption `caption_id` was heard from: the owner it was linked
/// to when it was made.
pub fn caption_source(draft: &Draft, caption_id: &SegmentId) -> Result<SegmentId, HeardMiss> {
    draft
        .relationships
        .iter()
        .filter(|relationship| relationship.kind == "linkage")
        .flat_map(|relationship| &relationship.id_to_id)
        .find(|pair| &pair.dependent == caption_id)
        .map(|pair| pair.owner.clone())
        .filter(|owner| heard_clip(draft, owner).is_ok())
        .ok_or(HeardMiss::NoSource)
}

/// The requests that give the clip caption `caption_id` was heard from new
/// captions from `words` heard again, split into lines as Transcribe splits
/// them. Every caption linked to the clip on that lane comes off; the lines
/// take `caption_id`'s look and its animations. One undo step. `fresh`
/// makes an unused id from a prefix.
pub fn rerun_requests(
    draft: &Draft,
    caption_id: &SegmentId,
    clip: &HeardClip,
    words: &WordColumns,
    fresh: &mut dyn FnMut(&str) -> String,
) -> Result<Vec<Request>, HeardMiss> {
    let segment = draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .find(|segment| &segment.id == caption_id)
        .ok_or(HeardMiss::UnknownSegment)?;
    if !matches!(
        draft.materials.get(segment.material_id.as_str()),
        Some(MaterialRef::Text(_))
    ) {
        return Err(HeardMiss::NotCaption);
    }
    let lines: Vec<SubtitleCue> = lines_on_clip(clip, words)?
        .into_iter()
        .map(|(target, words)| SubtitleCue {
            segment_id: SegmentId::new(fresh("caption")),
            material_id: MaterialId::new(fresh("caption")),
            target,
            words,
        })
        .collect();
    let animations: Vec<(AnimType, String, Micros)> = segment_animations(draft, segment)
        .map(|group| {
            group
                .animations
                .iter()
                .filter_map(|row| {
                    let kind = match row.kind.as_str() {
                        "in" => AnimType::In,
                        "out" => AnimType::Out,
                        "loop" => AnimType::Loop,
                        "caption" => AnimType::Caption,
                        _ => return None,
                    };
                    Some((
                        kind,
                        row.resource_id.clone(),
                        row.duration.unwrap_or(Micros(0)),
                    ))
                })
                .collect()
        })
        .unwrap_or_default();
    let mut requests = Vec::with_capacity(1 + lines.len() * animations.len());
    let ids: Vec<SegmentId> = lines.iter().map(|line| line.segment_id.clone()).collect();
    requests.push(Request::RecaptionClip {
        source_id: clip.source_id.clone(),
        like: caption_id.clone(),
        lines,
    });
    for id in ids {
        for (kind, animation, duration) in &animations {
            requests.push(Request::SetAnimation {
                segment_id: id.clone(),
                kind: *kind,
                animation: Some(animation.clone()),
                duration: *duration,
            });
        }
    }
    Ok(requests)
}

fn timeline_millis(start_ms: i64, end_ms: i64, clip: &HeardClip) -> Result<(i64, i64), HeardMiss> {
    let source_us = clip.range.duration.0;
    let target_us = clip.target.duration.0;
    let start_us = start_ms.checked_mul(1_000).ok_or(HeardMiss::BadClock)?;
    let end_us = end_ms.checked_mul(1_000).ok_or(HeardMiss::BadClock)?;
    let (start_us, end_us) = if clip.reverse {
        (
            source_us.checked_sub(end_us).ok_or(HeardMiss::BadClock)?,
            source_us.checked_sub(start_us).ok_or(HeardMiss::BadClock)?,
        )
    } else {
        (start_us, end_us)
    };
    let (start_us, end_us) = if clip.speed_holds {
        (
            scale(start_us, source_us, target_us)?,
            scale(end_us, source_us, target_us)?,
        )
    } else {
        (start_us, end_us)
    };
    let start_ms = start_us / 1_000;
    Ok((start_ms, (end_us / 1_000).max(start_ms)))
}

fn scale(value: i64, source_us: i64, target_us: i64) -> Result<i64, HeardMiss> {
    if source_us <= 0 {
        return Err(HeardMiss::BadClock);
    }
    let scaled = i128::from(value)
        .checked_mul(i128::from(target_us))
        .ok_or(HeardMiss::BadClock)?;
    i64::try_from(scaled / i128::from(source_us)).map_err(|_| HeardMiss::BadClock)
}

#[cfg(test)]
mod tests {
    use cutlass_model::{MaterialAudio, Segment, Track};

    use super::*;

    fn draft(speed: f64, reverse: bool) -> Draft {
        let mut draft = Draft::empty("D");
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("m"),
            path: "/talk.wav".to_owned(),
            ..Default::default()
        });
        let mut segment = Segment::new("a", "m", TimeRange::new(5_000_000, 2_000_000));
        segment.source_range = Some(TimeRange::new(0, 4_000_000));
        segment.speed = speed;
        segment.reverse = reverse;
        draft.tracks[0].segments.push(segment);
        draft
    }

    fn words() -> WordColumns {
        WordColumns {
            start_time: vec![0, 1_000],
            end_time: vec![1_000, 4_000],
            text: vec!["hi".to_owned(), "there".to_owned()],
            rest: Default::default(),
        }
    }

    #[test]
    fn word_times_follow_the_speed_and_reverse() {
        let clip = heard_clip(&draft(2.0, false), &SegmentId::new("a")).unwrap();
        let mapped = words_on_timeline(words(), &clip).unwrap();
        assert_eq!(mapped.start_time, vec![0, 500]);
        assert_eq!(mapped.end_time, vec![500, 2_000]);
        let clip = heard_clip(&draft(2.0, true), &SegmentId::new("a")).unwrap();
        let mapped = words_on_timeline(words(), &clip).unwrap();
        assert_eq!(mapped.start_time, vec![1_500, 0]);
        assert_eq!(mapped.end_time, vec![2_000, 1_500]);
    }

    #[test]
    fn a_caption_takes_a_free_text_row_or_opens_one() {
        let mut draft = draft(1.0, false);
        let clip = heard_clip(&draft, &SegmentId::new("a")).unwrap();
        let mut next = 0;
        let mut fresh = |prefix: &str| {
            next += 1;
            format!("{prefix}-{next}")
        };
        let requests = caption_requests(&draft, &clip, words(), &mut fresh).unwrap();
        let Some(Request::AddCaption { track_id, .. }) = requests.first().cloned() else {
            panic!("a caption request");
        };
        assert_eq!(track_id, TrackId::new("captions-1"));
        let mut text = Track::main(TrackId::new("text"));
        text.kind = TrackType::Text;
        draft.tracks.push(text);
        assert_eq!(
            caption_row(&draft, &clip.target),
            Some(TrackId::new("text"))
        );
    }

    #[test]
    fn nothing_heard_and_no_sound_refuse() {
        let draft = draft(1.0, false);
        let clip = heard_clip(&draft, &SegmentId::new("a")).unwrap();
        let empty = WordColumns::default();
        assert_eq!(
            words_on_timeline(empty, &clip),
            Err(HeardMiss::NothingHeard)
        );
        assert_eq!(
            heard_clip(&draft, &SegmentId::new("z")),
            Err(HeardMiss::UnknownSegment)
        );
    }

    /// A long talk heard on a 20 s clip: three sentences, a pause, and a
    /// run-on that needs a length break. Each word lasts 250 ms.
    fn talk() -> WordColumns {
        let script = "This is good. And it's actually good so I'm gonna show you the \
                      benchmarks in a second. Then we will look at what it costs";
        let mut words = WordColumns::default();
        let mut at = 0;
        for (index, word) in script.split_whitespace().enumerate() {
            if index == 17 {
                at += 1_000;
            }
            words.start_time.push(at);
            words.end_time.push(at + 250);
            words.text.push(word.to_owned());
            at += 300;
        }
        words
    }

    fn talk_session() -> (cutlass_session::Session, HeardClip, Vec<Request>) {
        let mut draft = Draft::empty("D");
        draft.materials.audios.push(MaterialAudio {
            id: MaterialId::new("m"),
            path: "/talk.wav".to_owned(),
            ..Default::default()
        });
        draft.tracks[0]
            .segments
            .push(Segment::new("a", "m", TimeRange::new(0, 20_000_000)));
        let session = cutlass_session::Session::open(draft);
        let clip = heard_clip(session.draft(), &SegmentId::new("a")).unwrap();
        let mut next = 0;
        let mut fresh = |prefix: &str| {
            next += 1;
            format!("{prefix}-{next}")
        };
        let requests = caption_requests(session.draft(), &clip, talk(), &mut fresh).unwrap();
        (session, clip, requests)
    }

    fn lane(draft: &Draft) -> Vec<(String, TimeRange, WordColumns)> {
        let track = draft
            .tracks
            .iter()
            .find(|track| track.flag == cutlass_model::TrackFlag::TextSubtitle)
            .unwrap();
        track
            .segments
            .iter()
            .map(|segment| {
                let Some(MaterialRef::Text(text)) =
                    draft.materials.get(segment.material_id.as_str())
                else {
                    panic!("text");
                };
                (
                    text.content.clone(),
                    segment.target.clone(),
                    text.words.clone(),
                )
            })
            .collect()
    }

    #[test]
    fn a_long_talk_lands_as_a_caption_per_line_back_to_back_in_one_step() {
        let (mut session, _, requests) = talk_session();
        assert_eq!(requests.len(), 4);
        session.apply(0, &requests).unwrap();
        let rows = lane(session.draft());
        let texts: Vec<&str> = rows.iter().map(|row| row.0.as_str()).collect();
        assert_eq!(
            texts,
            [
                "This is good.",
                "And it's actually good so I'm gonna show you the",
                "benchmarks in a second.",
                "Then we will look at what it costs",
            ]
        );
        // Back to back, but the pause before "Then" stays open.
        assert_eq!(rows[0].1, TimeRange::new(0, 900_000));
        assert_eq!(rows[1].1, TimeRange::new(900_000, 3_000_000));
        assert_eq!(rows[2].1, TimeRange::new(3_900_000, 1_150_000));
        assert_eq!(rows[3].1, TimeRange::new(6_100_000, 2_350_000));
        // Each caption's words are on its own clock.
        assert_eq!(rows[1].2.start_time[0], 0);
        assert_eq!(rows[1].2.start_time[1], 300);
        assert_eq!(rows[3].2.end_time.last(), Some(&2_350));
        let draft = session.draft();
        let pairs: Vec<_> = draft
            .relationships
            .iter()
            .flat_map(|relationship| &relationship.id_to_id)
            .collect();
        assert_eq!(pairs.len(), 4);
        assert!(pairs.iter().all(|pair| pair.owner.as_str() == "a"));
        session.undo().unwrap();
        assert!(session.draft().materials.texts.is_empty());
    }

    #[test]
    fn a_rerun_splits_the_new_words_the_same_way_and_keeps_the_look() {
        let (mut session, clip, requests) = talk_session();
        session.apply(0, &requests).unwrap();
        let first = session
            .draft()
            .tracks
            .iter()
            .find(|track| track.flag == cutlass_model::TrackFlag::TextSubtitle)
            .unwrap()
            .segments[0]
            .id
            .clone();
        assert_eq!(
            caption_source(session.draft(), &first),
            Ok(SegmentId::new("a"))
        );
        let mut styled = session.draft().clone();
        for text in &mut styled.materials.texts {
            "pop".clone_into(&mut text.caption_style);
        }
        let mut session = cutlass_session::Session::open(styled);
        session
            .apply(
                0,
                &[Request::SetAnimation {
                    segment_id: first.clone(),
                    kind: AnimType::Caption,
                    animation: Some("cutlass-anim-loop-pulse".to_owned()),
                    duration: Micros(0),
                }],
            )
            .unwrap();
        let mut next = 100;
        let mut fresh = |prefix: &str| {
            next += 1;
            format!("{prefix}-{next}")
        };
        let mut again = talk();
        "great.".clone_into(&mut again.text[2]);
        let requests = rerun_requests(session.draft(), &first, &clip, &again, &mut fresh).unwrap();
        session.apply(1, &requests).unwrap();
        let draft = session.draft();
        let rows = lane(draft);
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0].0, "This is great.");
        assert_eq!(rows[1].1, TimeRange::new(900_000, 3_000_000));
        assert_eq!(draft.materials.texts.len(), 4);
        assert!(
            draft
                .materials
                .texts
                .iter()
                .all(|text| text.caption_style == "pop")
        );
        let subtitles = draft
            .tracks
            .iter()
            .find(|track| track.flag == cutlass_model::TrackFlag::TextSubtitle)
            .unwrap();
        for segment in &subtitles.segments {
            let row = segment_animations(draft, segment).expect("animation");
            assert_eq!(row.animations[0].kind, "caption");
        }
        session.undo().unwrap();
        assert_eq!(lane(session.draft())[0].0, "This is good.");
    }

    #[test]
    fn a_rerun_refuses_a_caption_with_no_heard_clip_or_no_words_in_it() {
        let draft = draft(1.0, false);
        let clip = heard_clip(&draft, &SegmentId::new("a")).unwrap();
        let mut fresh = |prefix: &str| prefix.to_owned();
        assert_eq!(
            caption_source(&draft, &SegmentId::new("a")),
            Err(HeardMiss::NoSource)
        );
        assert_eq!(
            rerun_requests(&draft, &SegmentId::new("z"), &clip, &words(), &mut fresh),
            Err(HeardMiss::UnknownSegment)
        );
        assert_eq!(
            rerun_requests(&draft, &SegmentId::new("a"), &clip, &words(), &mut fresh),
            Err(HeardMiss::NotCaption)
        );
        assert_eq!(
            caption_requests(&draft, &clip, WordColumns::default(), &mut fresh),
            Err(HeardMiss::NothingHeard)
        );
    }
}
