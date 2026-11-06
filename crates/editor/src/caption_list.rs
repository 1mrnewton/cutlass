//! The caption list every shell shows: each caption on a caption's lane in
//! timeline order, the one under the playhead, a search, and the requests
//! that edit a line, add one after it, and style every caption at once.

use cutlass_captions::rematch;
use cutlass_model::{
    Draft, MaterialId, MaterialRef, Micros, SegmentId, TimeRange, Track, TrackFlag, TrackId,
    WordColumns,
};
use cutlass_session::{Request, SubtitleCue, TrimEdge};

use crate::inspector::is_caption;

/// What a caption added after a row says until it is typed over.
pub const NEW_CAPTION_TEXT: &str = "New caption";
/// The longest caption the add button places in a gap.
pub const NEW_CAPTION_LENGTH: Micros = Micros(2_000_000);
/// The shortest gap the add button fills; below it the row's time is split.
pub const NEW_CAPTION_GAP: Micros = Micros(300_000);

/// One caption as the list shows it.
#[derive(Clone, Debug, PartialEq)]
pub struct CaptionRow {
    pub segment_id: SegmentId,
    pub material_id: MaterialId,
    pub target: TimeRange,
    /// What the caption says.
    pub text: String,
}

/// Why a caption list edit made no request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptionListMiss {
    /// The clip is not a caption on the timeline.
    NotCaption,
    /// A caption needs at least one word; delete it instead.
    Empty,
    /// The edit is longer than words can be matched over.
    TooLong,
    /// There is no room after the row, and the row is too short to split.
    NoRoom,
}

fn lane_of<'a>(draft: &'a Draft, segment_id: &SegmentId) -> Option<&'a Track> {
    draft.tracks.iter().find(|track| {
        track
            .segments
            .iter()
            .any(|segment| &segment.id == segment_id)
    })
}

/// Every caption on the lane of caption `segment_id`, in timeline order.
/// Empty when the clip is not a caption.
#[must_use]
pub fn caption_rows(draft: &Draft, segment_id: &SegmentId) -> Vec<CaptionRow> {
    let Some(track) = lane_of(draft, segment_id) else {
        return Vec::new();
    };
    let mut rows: Vec<CaptionRow> = track
        .segments
        .iter()
        .filter_map(
            |segment| match draft.materials.get(segment.material_id.as_str()) {
                Some(MaterialRef::Text(text)) if is_caption(track, text) => Some(CaptionRow {
                    segment_id: segment.id.clone(),
                    material_id: text.id.clone(),
                    target: segment.target.clone(),
                    text: text.content.clone(),
                }),
                _ => None,
            },
        )
        .collect();
    if !rows.iter().any(|row| &row.segment_id == segment_id) {
        return Vec::new();
    }
    rows.sort_by_key(|row| row.target.start.0);
    rows
}

/// The row showing at `time`, by index into `rows`.
#[must_use]
pub fn row_at(rows: &[CaptionRow], time: Micros) -> Option<usize> {
    rows.iter().position(|row| {
        row.target.start.0 <= time.0
            && time.0 < row.target.start.0.saturating_add(row.target.duration.0)
    })
}

/// The rows whose text holds `query`, ignoring case, by index into
/// `rows`. A blank query keeps every row.
#[must_use]
pub fn rows_matching(rows: &[CaptionRow], query: &str) -> Vec<usize> {
    let query = query.trim().to_lowercase();
    rows.iter()
        .enumerate()
        .filter(|(_, row)| query.is_empty() || row.text.to_lowercase().contains(&query))
        .map(|(index, _)| index)
        .collect()
}

/// The words caption `segment_id` has: its material id, its words, and its
/// length in milliseconds.
fn caption_words(
    draft: &Draft,
    segment_id: &SegmentId,
) -> Result<(MaterialId, WordColumns, TimeRange), CaptionListMiss> {
    let track = lane_of(draft, segment_id).ok_or(CaptionListMiss::NotCaption)?;
    let segment = track
        .segments
        .iter()
        .find(|segment| &segment.id == segment_id)
        .ok_or(CaptionListMiss::NotCaption)?;
    match draft.materials.get(segment.material_id.as_str()) {
        Some(MaterialRef::Text(text)) if is_caption(track, text) => {
            Ok((text.id.clone(), text.words.clone(), segment.target.clone()))
        }
        _ => Err(CaptionListMiss::NotCaption),
    }
}

/// Words saying `text`, spread over `length_ms` by their letters.
#[must_use]
pub fn words_over(text: &str, length_ms: i64) -> WordColumns {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let letters: i64 = tokens
        .iter()
        .map(|word| i64::try_from(word.chars().count()).unwrap_or(i64::MAX))
        .fold(0, i64::saturating_add)
        .max(1);
    let at = |done: i64| {
        let scaled = i128::from(length_ms.max(0)) * i128::from(done) / i128::from(letters);
        i64::try_from(scaled).unwrap_or(length_ms)
    };
    let mut words = WordColumns::default();
    let mut done = 0_i64;
    for word in tokens {
        words.start_time.push(at(done));
        done = done.saturating_add(i64::try_from(word.chars().count()).unwrap_or(i64::MAX));
        words.end_time.push(at(done));
        words.text.push(word.to_owned());
    }
    words
}

/// Words saying `edited` in place of `words`, keeping the times of the
/// words that stay. A caption with no words spreads them over its length.
pub fn rematched(
    words: &WordColumns,
    edited: &str,
    length_ms: i64,
) -> Result<WordColumns, CaptionListMiss> {
    if edited.trim().is_empty() {
        return Err(CaptionListMiss::Empty);
    }
    if words.text.is_empty() {
        return Ok(words_over(edited, length_ms));
    }
    rematch(words, edited).map_err(|_| CaptionListMiss::TooLong)
}

/// The request that makes caption `segment_id` say `text`, its words
/// rematched so the ones that stay keep their times. `None` when it already
/// says that.
pub fn caption_text_request(
    draft: &Draft,
    segment_id: &SegmentId,
    text: &str,
) -> Result<Option<Request>, CaptionListMiss> {
    let (material_id, words, target) = caption_words(draft, segment_id)?;
    let current = draft
        .materials
        .texts
        .iter()
        .find(|item| item.id == material_id)
        .map(|item| item.content.as_str());
    if current == Some(text) {
        return Ok(None);
    }
    let words = rematched(&words, text, target.duration.0 / 1_000)?;
    Ok(Some(Request::SetWords { material_id, words }))
}

/// The requests that add a caption after `segment_id`, looking like it.
///
/// A gap of at least [`NEW_CAPTION_GAP`] after the row takes a caption up
/// to [`NEW_CAPTION_LENGTH`] long saying [`NEW_CAPTION_TEXT`]. With no such
/// gap the row's time is split in half: the row ends at the middle, and the
/// words that start after it move to the new caption. `fresh` makes an
/// unused id from a prefix.
pub fn add_after_requests(
    draft: &Draft,
    segment_id: &SegmentId,
    fresh: &mut dyn FnMut(&str) -> String,
) -> Result<Vec<Request>, CaptionListMiss> {
    let (material_id, words, target) = caption_words(draft, segment_id)?;
    let track = lane_of(draft, segment_id).ok_or(CaptionListMiss::NotCaption)?;
    let end = target.start.0.saturating_add(target.duration.0);
    let next = track
        .segments
        .iter()
        .map(|segment| segment.target.start.0)
        .filter(|start| *start >= end)
        .min();
    let gap = next.map_or(NEW_CAPTION_LENGTH.0, |next| next - end);
    let new_ids = |fresh: &mut dyn FnMut(&str) -> String| {
        (
            SegmentId::new(fresh("caption")),
            MaterialId::new(fresh("caption")),
        )
    };
    if gap >= NEW_CAPTION_GAP.0 {
        let length = gap.min(NEW_CAPTION_LENGTH.0);
        let (new_segment, new_material) = new_ids(fresh);
        return Ok(vec![Request::InsertCaption {
            like: segment_id.clone(),
            cue: SubtitleCue {
                segment_id: new_segment,
                material_id: new_material,
                target: TimeRange::new(end, length),
                words: words_over(NEW_CAPTION_TEXT, length / 1_000),
            },
        }]);
    }
    let half = target.duration.0 / 2;
    if half < NEW_CAPTION_GAP.0 {
        return Err(CaptionListMiss::NoRoom);
    }
    let middle_ms = half / 1_000;
    let mut kept = WordColumns::default();
    let mut moved = WordColumns::default();
    for index in 0..words.text.len() {
        let start = words.start_time.get(index).copied().unwrap_or_default();
        let finish = words.end_time.get(index).copied().unwrap_or(start);
        let text = words.text[index].clone();
        if start < middle_ms {
            kept.start_time.push(start);
            kept.end_time.push(finish.min(middle_ms));
            kept.text.push(text);
        } else if !(moved.text.is_empty() && text == crate::LINE_BREAK) {
            moved.start_time.push(start - middle_ms);
            moved.end_time.push(finish - middle_ms);
            moved.text.push(text);
        }
    }
    while kept
        .text
        .last()
        .is_some_and(|word| word == crate::LINE_BREAK)
    {
        kept.start_time.pop();
        kept.end_time.pop();
        kept.text.pop();
    }
    if kept.text.is_empty() {
        kept = words_over(NEW_CAPTION_TEXT, middle_ms);
    }
    if moved.text.is_empty() {
        moved = words_over(NEW_CAPTION_TEXT, (target.duration.0 - half) / 1_000);
    }
    let (new_segment, new_material) = new_ids(fresh);
    Ok(vec![
        Request::Trim {
            segment_id: segment_id.clone(),
            edge: TrimEdge::End,
            time: Micros(target.start.0 + half),
        },
        Request::SetWords {
            material_id,
            words: kept,
        },
        Request::InsertCaption {
            like: segment_id.clone(),
            cue: SubtitleCue {
                segment_id: new_segment,
                material_id: new_material,
                target: TimeRange::new(target.start.0 + half, target.duration.0 - half),
                words: moved,
            },
        },
    ])
}

/// The request that takes caption `segment_id` off the timeline.
#[must_use]
pub fn remove_caption_request(segment_id: &SegmentId) -> Request {
    Request::Remove {
        segment_id: segment_id.clone(),
    }
}

/// The subtitle lane caption `segment_id` sits on, when it is one: the
/// lane "Apply to all main captions" styles together.
#[must_use]
pub fn main_caption_lane(draft: &Draft, segment_id: &SegmentId) -> Option<TrackId> {
    let track = lane_of(draft, segment_id)?;
    (track.flag == TrackFlag::TextSubtitle).then(|| track.id.clone())
}

/// `request` for every caption on the subtitle lane of the clip it names,
/// so a style change reaches all of them in one undo step.
///
/// Only a look or place change fans out: size, color, alignment, spacing,
/// wrap width, face, outline, shadow, background, caption style, stagger,
/// position and scale. Any other request, or a clip that is not on a
/// subtitle lane, comes back alone.
#[must_use]
pub fn for_every_caption(draft: &Draft, request: &Request) -> Vec<Request> {
    let Some(segment_id) = styled_clip(request) else {
        return vec![request.clone()];
    };
    if main_caption_lane(draft, segment_id).is_none() {
        return vec![request.clone()];
    }
    let rows = caption_rows(draft, segment_id);
    if rows.is_empty() {
        return vec![request.clone()];
    }
    rows.iter()
        .map(|row| with_clip(request, &row.segment_id))
        .collect()
}

fn styled_clip(request: &Request) -> Option<&SegmentId> {
    match request {
        Request::SetTextSize { segment_id, .. }
        | Request::SetTextColor { segment_id, .. }
        | Request::SetTextAlign { segment_id, .. }
        | Request::SetLetterSpacing { segment_id, .. }
        | Request::SetLineSpacing { segment_id, .. }
        | Request::SetLineWidth { segment_id, .. }
        | Request::SetTextFont { segment_id, .. }
        | Request::SetTextOutline { segment_id, .. }
        | Request::SetTextShadow { segment_id, .. }
        | Request::SetTextBackground { segment_id, .. }
        | Request::SetCaptionStyle { segment_id, .. }
        | Request::SetCaptionStagger { segment_id, .. }
        | Request::SetPosition { segment_id, .. }
        | Request::SetScale { segment_id, .. } => Some(segment_id),
        _ => None,
    }
}

fn with_clip(request: &Request, clip: &SegmentId) -> Request {
    let mut request = request.clone();
    match &mut request {
        Request::SetTextSize { segment_id, .. }
        | Request::SetTextColor { segment_id, .. }
        | Request::SetTextAlign { segment_id, .. }
        | Request::SetLetterSpacing { segment_id, .. }
        | Request::SetLineSpacing { segment_id, .. }
        | Request::SetLineWidth { segment_id, .. }
        | Request::SetTextFont { segment_id, .. }
        | Request::SetTextOutline { segment_id, .. }
        | Request::SetTextShadow { segment_id, .. }
        | Request::SetTextBackground { segment_id, .. }
        | Request::SetCaptionStyle { segment_id, .. }
        | Request::SetCaptionStagger { segment_id, .. }
        | Request::SetPosition { segment_id, .. }
        | Request::SetScale { segment_id, .. } => clip.clone_into(segment_id),
        _ => {}
    }
    request
}

#[cfg(test)]
mod tests {
    use cutlass_model::Draft;
    use cutlass_session::Session;

    use super::*;

    fn counter() -> impl FnMut(&str) -> String {
        let mut next = 0;
        move |prefix: &str| {
            next += 1;
            format!("{prefix}-new-{next}")
        }
    }

    /// Three captions on a subtitle lane: "one two" 0–1 s, "three" 3–4 s,
    /// "four five" 4–5 s, stored out of order.
    fn session() -> Session {
        let srt = "00:00:04,000 --> 00:00:05,000\nfour five\n\n00:00:00,000 --> 00:00:01,000\none two\n\n00:00:03,000 --> 00:00:04,000\nthree\n";
        let mut ids = 0;
        let request = crate::subtitle_request(srt, crate::SubtitleFormat::Srt, &mut |p: &str| {
            ids += 1;
            format!("{p}-{ids}")
        })
        .unwrap();
        let mut session = Session::open(Draft::empty("D"));
        session.apply(0, &[request]).unwrap();
        session
    }

    fn ids(rows: &[CaptionRow]) -> Vec<String> {
        rows.iter().map(|row| row.text.clone()).collect()
    }

    fn first(session: &Session) -> SegmentId {
        let lane = session
            .draft()
            .tracks
            .iter()
            .find(|track| track.flag == TrackFlag::TextSubtitle)
            .unwrap();
        lane.segments[0].id.clone()
    }

    #[test]
    fn the_list_holds_the_lane_in_time_order_and_finds_the_playhead() {
        let session = session();
        let rows = caption_rows(session.draft(), &first(&session));
        assert_eq!(ids(&rows), ["one two", "three", "four five"]);
        assert_eq!(row_at(&rows, Micros(3_500_000)), Some(1));
        assert_eq!(row_at(&rows, Micros(4_000_000)), Some(2));
        assert_eq!(row_at(&rows, Micros(2_000_000)), None);
        assert_eq!(rows_matching(&rows, "  FOUR "), [2]);
        assert_eq!(rows_matching(&rows, ""), [0, 1, 2]);
        assert!(rows_matching(&rows, "zzz").is_empty());
        assert!(caption_rows(session.draft(), &SegmentId::new("nobody")).is_empty());
    }

    #[test]
    fn an_edited_line_keeps_the_times_of_the_words_that_stay() {
        let mut session = session();
        let id = first(&session);
        let before = session.draft().texts_words(&id);
        let request = caption_text_request(session.draft(), &id, "one too")
            .unwrap()
            .unwrap();
        session.apply(1, &[request]).unwrap();
        let after = session.draft().texts_words(&id);
        assert_eq!(after.text, ["one", "too"]);
        assert_eq!(after.start_time[0], before.start_time[0]);
        assert_eq!(after.end_time[1], before.end_time[1]);
        assert_eq!(
            caption_text_request(session.draft(), &id, "one too"),
            Ok(None)
        );
        assert_eq!(
            caption_text_request(session.draft(), &id, "  "),
            Err(CaptionListMiss::Empty)
        );
    }

    #[test]
    fn add_fills_the_gap_after_a_row_or_splits_its_time() {
        let mut session = session();
        let mut fresh = counter();
        let id = first(&session);
        let requests = add_after_requests(session.draft(), &id, &mut fresh).unwrap();
        session.apply(1, &requests).unwrap();
        let rows = caption_rows(session.draft(), &id);
        assert_eq!(
            ids(&rows),
            ["one two", NEW_CAPTION_TEXT, "three", "four five"]
        );
        assert_eq!(rows[1].target, TimeRange::new(1_000_000, 2_000_000));

        // "three" runs into "four five": its time is split.
        let three = rows[2].segment_id.clone();
        let requests = add_after_requests(session.draft(), &three, &mut fresh).unwrap();
        session.apply(2, &requests).unwrap();
        let rows = caption_rows(session.draft(), &id);
        assert_eq!(
            ids(&rows),
            [
                "one two",
                NEW_CAPTION_TEXT,
                "three",
                NEW_CAPTION_TEXT,
                "four five"
            ]
        );
        assert_eq!(rows[2].target, TimeRange::new(3_000_000, 500_000));
        assert_eq!(rows[3].target, TimeRange::new(3_500_000, 500_000));

        // "four five" splits between its words after a neighbour lands.
        let mut session = session_with_adjacent_pair();
        let id = first(&session);
        let requests = add_after_requests(session.draft(), &id, &mut fresh).unwrap();
        session.apply(1, &requests).unwrap();
        let rows = caption_rows(session.draft(), &id);
        assert_eq!(ids(&rows), ["alpha", "beta", "next"]);
        undo_restores(&mut session);
    }

    fn session_with_adjacent_pair() -> Session {
        let srt =
            "00:00:00,000 --> 00:00:01,000\nalpha beta\n\n00:00:01,000 --> 00:00:02,000\nnext\n";
        let request =
            crate::subtitle_request(srt, crate::SubtitleFormat::Srt, &mut counter()).unwrap();
        let mut session = Session::open(Draft::empty("D"));
        session.apply(0, &[request]).unwrap();
        session
    }

    fn undo_restores(session: &mut Session) {
        session.undo().unwrap();
        let id = first(session);
        assert_eq!(
            ids(&caption_rows(session.draft(), &id)),
            ["alpha beta", "next"]
        );
    }

    #[test]
    fn a_short_row_with_no_room_after_it_adds_nothing() {
        let srt = "00:00:00,000 --> 00:00:00,400\nhi\n\n00:00:00,400 --> 00:00:01,000\nthere\n";
        let request =
            crate::subtitle_request(srt, crate::SubtitleFormat::Srt, &mut counter()).unwrap();
        let mut session = Session::open(Draft::empty("D"));
        session.apply(0, &[request]).unwrap();
        let id = first(&session);
        assert_eq!(
            add_after_requests(session.draft(), &id, &mut counter()),
            Err(CaptionListMiss::NoRoom)
        );
        assert_eq!(
            add_after_requests(session.draft(), &SegmentId::new("x"), &mut counter()),
            Err(CaptionListMiss::NotCaption)
        );
    }

    #[test]
    fn a_style_change_reaches_every_caption_on_a_subtitle_lane_only() {
        let mut session = session();
        let id = first(&session);
        let size = Request::SetTextSize {
            segment_id: id.clone(),
            size: 30.0,
            face: String::new(),
        };
        let all = for_every_caption(session.draft(), &size);
        assert_eq!(all.len(), 3);
        session.apply(1, &all).unwrap();
        assert!(
            session
                .draft()
                .materials
                .texts
                .iter()
                .all(|t| t.font_size == 30.0)
        );
        session.undo().unwrap();
        assert!(
            session
                .draft()
                .materials
                .texts
                .iter()
                .all(|t| t.font_size != 30.0)
        );

        // A words edit stays on its clip.
        let words = Request::SetCaptionKeyword {
            segment_id: id.clone(),
            word: 0,
            on: true,
        };
        assert_eq!(
            for_every_caption(session.draft(), &words),
            std::slice::from_ref(&words)
        );

        // Off a subtitle lane, the clip is styled alone.
        let mut draft = session.draft().clone();
        for track in &mut draft.tracks {
            track.flag = TrackFlag::None;
        }
        assert_eq!(
            for_every_caption(&draft, &size),
            std::slice::from_ref(&size)
        );
        assert_eq!(main_caption_lane(&draft, &id), None);
    }

    trait Words {
        fn texts_words(&self, segment_id: &SegmentId) -> WordColumns;
    }

    impl Words for Draft {
        fn texts_words(&self, segment_id: &SegmentId) -> WordColumns {
            caption_words(self, segment_id).unwrap().1
        }
    }
}
