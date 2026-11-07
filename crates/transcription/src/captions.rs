//! A clip's heard words as a caption: which stretch of the file is heard,
//! how its word times land on the timeline, and the caption row.

use cutlass_model::{
    Draft, MaterialId, MaterialRef, SegmentId, TimeRange, TrackAttribute, TrackId, TrackType,
    WordColumns,
};
use cutlass_session::Request;

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

/// The request that writes `words` (already on the timeline clock) as a
/// caption under `clip`, on a free text row or a new one. `fresh` makes an
/// unused id from a prefix.
pub fn caption_request(
    draft: &Draft,
    clip: &HeardClip,
    words: WordColumns,
    fresh: &mut dyn FnMut(&str) -> String,
) -> Result<Request, HeardMiss> {
    if words.text.is_empty() {
        return Err(HeardMiss::NothingHeard);
    }
    let track_id =
        caption_row(draft, &clip.target).unwrap_or_else(|| TrackId::new(fresh("captions")));
    Ok(Request::AddCaption {
        source_id: clip.source_id.clone(),
        track_id,
        segment_id: SegmentId::new(fresh("caption")),
        material_id: MaterialId::new(fresh("caption")),
        words,
    })
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

/// The request that gives caption `caption_id` the words heard again from
/// `clip`, already on the clip's timeline clock. The words are moved to the
/// caption's own clock, and only those it spans are kept, so a caption that
/// was trimmed or split keeps its own stretch. Its look stays.
pub fn rerun_request(
    draft: &Draft,
    caption_id: &SegmentId,
    clip: &HeardClip,
    words: &WordColumns,
) -> Result<Request, HeardMiss> {
    let segment = draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .find(|segment| &segment.id == caption_id)
        .ok_or(HeardMiss::UnknownSegment)?;
    let Some(MaterialRef::Text(text)) = draft.materials.get(segment.material_id.as_str()) else {
        return Err(HeardMiss::NotCaption);
    };
    let offset = (clip.target.start.0 - segment.target.start.0) / 1_000;
    let length = segment.target.duration.0 / 1_000;
    let mut kept = WordColumns::default();
    for ((word, start), end) in words
        .text
        .iter()
        .zip(&words.start_time)
        .zip(&words.end_time)
    {
        let start = start.checked_add(offset).ok_or(HeardMiss::BadClock)?;
        let end = end.checked_add(offset).ok_or(HeardMiss::BadClock)?;
        if end <= 0 || start >= length {
            continue;
        }
        let start = start.clamp(0, length);
        kept.start_time.push(start);
        kept.end_time.push(end.clamp(start, length));
        kept.text.push(word.clone());
    }
    if kept.text.is_empty() {
        return Err(HeardMiss::NothingHeard);
    }
    Ok(Request::SetWords {
        material_id: text.id.clone(),
        words: kept,
    })
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
        let Request::AddCaption { track_id, .. } =
            caption_request(&draft, &clip, words(), &mut fresh).unwrap()
        else {
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

    #[test]
    fn a_rerun_finds_the_heard_clip_and_keeps_the_captions_own_stretch() {
        let mut session = cutlass_session::Session::open(draft(1.0, false));
        let clip = heard_clip(session.draft(), &SegmentId::new("a")).unwrap();
        let mut next = 0;
        let mut fresh = |prefix: &str| {
            next += 1;
            format!("{prefix}-{next}")
        };
        let request = caption_request(session.draft(), &clip, words(), &mut fresh).unwrap();
        session.apply(0, &[request]).unwrap();
        // The main row closed its gap, so the clip is heard where it is now.
        let clip = heard_clip(session.draft(), &SegmentId::new("a")).unwrap();
        assert_eq!(clip.target.start.0, 0);
        let caption = SegmentId::new("caption-2");
        assert_eq!(
            caption_source(session.draft(), &caption),
            Ok(SegmentId::new("a"))
        );

        let heard = WordColumns {
            start_time: vec![0, 900, 1_500],
            end_time: vec![800, 1_400, 1_900],
            text: vec!["hello".to_owned(), "out".to_owned(), "there".to_owned()],
            rest: Default::default(),
        };
        let request = rerun_request(session.draft(), &caption, &clip, &heard).unwrap();
        let Request::SetWords { words, .. } = &request else {
            panic!("set words");
        };
        assert_eq!(words, &heard);

        // A caption that starts a second later keeps the words it spans,
        // on its own clock.
        let mut moved = session.draft().clone();
        let segment = moved
            .tracks
            .iter_mut()
            .flat_map(|track| &mut track.segments)
            .find(|segment| segment.id == caption)
            .unwrap();
        segment.target = TimeRange::new(1_000_000, 1_000_000);
        let Request::SetWords { words, .. } =
            rerun_request(&moved, &caption, &clip, &heard).unwrap()
        else {
            panic!("set words");
        };
        assert_eq!(words.text, ["out", "there"]);
        assert_eq!(words.start_time, [0, 500]);
        assert_eq!(words.end_time, [400, 900]);

        session.apply(1, &[request]).unwrap();
        session.undo().unwrap();
    }

    #[test]
    fn a_rerun_refuses_a_caption_with_no_heard_clip_or_no_words_in_it() {
        let draft = draft(1.0, false);
        let clip = heard_clip(&draft, &SegmentId::new("a")).unwrap();
        assert_eq!(
            caption_source(&draft, &SegmentId::new("a")),
            Err(HeardMiss::NoSource)
        );
        assert_eq!(
            rerun_request(&draft, &SegmentId::new("z"), &clip, &words()),
            Err(HeardMiss::UnknownSegment)
        );
        assert_eq!(
            rerun_request(&draft, &SegmentId::new("a"), &clip, &words()),
            Err(HeardMiss::NotCaption)
        );
    }
}
