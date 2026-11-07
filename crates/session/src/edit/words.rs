use cutlass_model::{
    Draft, MaterialId, MaterialRef, MaterialText, Segment, SegmentId, TrackFlag, TrackId,
    TrackType, WordColumns,
};

use super::EditError;
use crate::request::SubtitleCue;

/// Join word text with single spaces. Empty pieces are skipped.
fn caption_text(words: &[String]) -> String {
    let mut text = String::new();
    let mut broken = false;
    for word in words {
        if word.is_empty() {
            continue;
        }
        // A line or page break marker breaks the line instead of a space.
        if word.trim_matches('\n').is_empty() {
            broken = !text.is_empty();
            continue;
        }
        if broken {
            text.push('\n');
        } else if !text.is_empty() {
            text.push(' ');
        }
        broken = false;
        text.push_str(word);
    }
    text
}

pub fn set_words(
    draft: &mut Draft,
    material_id: &MaterialId,
    words: WordColumns,
) -> Result<(), EditError> {
    if words.start_time.len() != words.end_time.len() || words.start_time.len() != words.text.len()
    {
        return Err(EditError::WordShape);
    }
    for (start, end) in words.start_time.iter().zip(&words.end_time) {
        if *start < 0 || *end < *start {
            return Err(EditError::WordShape);
        }
    }
    let content = caption_text(&words.text);
    let text = draft
        .materials
        .texts
        .iter_mut()
        .find(|text| &text.id == material_id)
        .ok_or(EditError::MissingMaterial)?;
    // A marked word stays marked where its start now falls.
    let marked_at: Vec<i64> = text
        .caption_keywords
        .iter()
        .filter_map(|word| text.words.start_time.get(*word).copied())
        .collect();
    let mut keywords: Vec<usize> = marked_at
        .iter()
        .filter_map(|at| {
            (0..words.text.len()).find(|index| {
                words.start_time[*index] <= *at
                    && *at < words.end_time[*index].max(words.start_time[*index] + 1)
                    && words.text[*index] != "\n"
            })
        })
        .collect();
    keywords.sort_unstable();
    keywords.dedup();
    text.caption_keywords = keywords;
    text.words = words.clone();
    text.current_words = words;
    text.content.clone_from(&content);
    text.base_content = content;
    Ok(())
}

/// Material kind of a caption: a text clip that came from speech.
pub const CAPTION_KIND: &str = "subtitle";

/// True when every clip on the lane is a caption, or the lane is empty.
fn holds_only_captions(draft: &Draft, track_index: usize) -> bool {
    draft.tracks[track_index].segments.iter().all(|segment| {
        matches!(
            draft.materials.get(segment.material_id.as_str()),
            Some(MaterialRef::Text(text)) if text.kind == CAPTION_KIND
        )
    })
}

/// Place one caption on `track_id` for the source clip and store `words`.
///
/// A missing track is added as a text lane flagged as subtitles. An unflagged
/// text lane that holds only captions takes that flag too. The material kind
/// is `subtitle`, so the clip stays a caption after its words are cleared.
/// The caption keeps the source clip's timeline range. A link is recorded
/// when the source can own one.
pub fn add_caption(
    draft: &mut Draft,
    source_id: &SegmentId,
    track_id: &TrackId,
    segment_id: &SegmentId,
    material_id: &MaterialId,
    words: WordColumns,
) -> Result<(), EditError> {
    if words.text.is_empty() {
        return Err(EditError::WordShape);
    }
    let (source_track, source_index) = super::locate(draft, source_id)?;
    super::ensure_unlocked(draft.tracks[source_track].attribute)?;
    let target = draft.tracks[source_track].segments[source_index]
        .target
        .clone();
    if draft.tracks.iter().any(|track| {
        track
            .segments
            .iter()
            .any(|segment| &segment.id == segment_id)
    }) {
        return Err(EditError::DuplicateSegment);
    }
    if draft.materials.get(material_id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    if let Some(index) = draft.tracks.iter().position(|track| &track.id == track_id) {
        let track = &draft.tracks[index];
        if !matches!(track.kind, TrackType::Text | TrackType::TextToVideoText) {
            return Err(EditError::TrackKind);
        }
        super::ensure_unlocked(track.attribute)?;
        if track.flag == TrackFlag::None && holds_only_captions(draft, index) {
            draft.tracks[index].flag = TrackFlag::TextSubtitle;
        }
    } else {
        super::track::insert_track(draft, track_id, &TrackType::Text)?;
        if let Some(track) = draft.tracks.iter_mut().find(|track| &track.id == track_id) {
            track.flag = TrackFlag::TextSubtitle;
        }
    }
    let mut text = MaterialText::new(material_id.clone());
    text.kind = CAPTION_KIND.to_owned();
    draft.materials.texts.push(text);
    let mut segment = Segment::new(segment_id.clone(), material_id.clone(), target);
    segment.render_index = TrackType::Text.render_floor().unwrap_or(0);
    super::place::place(draft, track_id, segment)?;
    set_words(draft, material_id, words)?;
    match super::linkage::link(draft, source_id, segment_id) {
        Ok(()) | Err(EditError::AlreadyLinked) => Ok(()),
        Err(error) => Err(error),
    }
}

/// Place `cues` as captions on a new subtitle lane `track_id`.
///
/// Each cue becomes a `subtitle` text clip at its target with its words.
/// An existing lane, no cues, a cue with no words, and cues that overlap are
/// refused; the session drops the partly changed draft.
pub fn add_subtitles(
    draft: &mut Draft,
    track_id: &TrackId,
    cues: &[SubtitleCue],
) -> Result<(), EditError> {
    if cues.is_empty() || cues.iter().any(|cue| cue.words.text.is_empty()) {
        return Err(EditError::WordShape);
    }
    if draft.tracks.iter().any(|track| &track.id == track_id) {
        return Err(EditError::TrackKind);
    }
    super::track::insert_track(draft, track_id, &TrackType::Text)?;
    if let Some(track) = draft.tracks.iter_mut().find(|track| &track.id == track_id) {
        track.flag = TrackFlag::TextSubtitle;
    }
    for cue in cues {
        if draft.materials.get(cue.material_id.as_str()).is_some() {
            return Err(EditError::DuplicateMaterial);
        }
        let mut text = MaterialText::new(cue.material_id.clone());
        text.kind = CAPTION_KIND.to_owned();
        draft.materials.texts.push(text);
        let mut segment = Segment::new(
            cue.segment_id.clone(),
            cue.material_id.clone(),
            cue.target.clone(),
        );
        segment.render_index = TrackType::Text.render_floor().unwrap_or(0);
        super::place::place(draft, track_id, segment)?;
        set_words(draft, &cue.material_id, cue.words.clone())?;
    }
    Ok(())
}

/// Place `cue` on the lane of caption `like`, with `like`'s text look and
/// canvas place. See [`crate::Request::InsertCaption`].
pub fn insert_caption(
    draft: &mut Draft,
    like: &SegmentId,
    cue: &SubtitleCue,
) -> Result<(), EditError> {
    if cue.words.text.is_empty() {
        return Err(EditError::WordShape);
    }
    let (track_index, segment_index) = super::locate(draft, like)?;
    let track = &draft.tracks[track_index];
    if !matches!(track.kind, TrackType::Text | TrackType::TextToVideoText) {
        return Err(EditError::TrackKind);
    }
    super::ensure_unlocked(track.attribute)?;
    let track_id = track.id.clone();
    let model = &track.segments[segment_index];
    let Some(MaterialRef::Text(look)) = draft.materials.get(model.material_id.as_str()) else {
        return Err(EditError::TrackKind);
    };
    if draft.materials.get(cue.material_id.as_str()).is_some() {
        return Err(EditError::DuplicateMaterial);
    }
    let mut text = look.clone();
    text.id = cue.material_id.clone();
    text.caption_keywords.clear();
    let mut segment = Segment::new(
        cue.segment_id.clone(),
        cue.material_id.clone(),
        cue.target.clone(),
    );
    segment.clip = model.clip.clone();
    segment.uniform_scale = model.uniform_scale.clone();
    segment.render_index = model.render_index;
    draft.materials.texts.push(text);
    super::place::place(draft, &track_id, segment)?;
    set_words(draft, &cue.material_id, cue.words.clone())
}

#[cfg(test)]
mod tests {
    use cutlass_model::MaterialText;

    use super::*;

    fn draft_with_text() -> (Draft, MaterialId) {
        let mut draft = Draft::empty("D");
        let id = MaterialId::new("caption");
        draft.materials.texts.push(MaterialText::new(id.clone()));
        (draft, id)
    }

    fn columns(start: i64, end: i64, text: &str) -> WordColumns {
        WordColumns {
            start_time: vec![start],
            end_time: vec![end],
            text: vec![text.to_owned()],
            rest: Default::default(),
        }
    }

    #[test]
    fn a_marked_word_stays_marked_when_words_are_joined_or_split() {
        let (mut draft, id) = draft_with_text();
        let three = WordColumns {
            start_time: vec![0, 100, 200],
            end_time: vec![100, 200, 300],
            text: vec!["a".to_owned(), "b".to_owned(), "c".to_owned()],
            rest: Default::default(),
        };
        set_words(&mut draft, &id, three).unwrap();
        draft.materials.texts[0].caption_keywords = vec![2];
        // Joining the first two moves the third word to index 1.
        let joined = WordColumns {
            start_time: vec![0, 200],
            end_time: vec![200, 300],
            text: vec!["a b".to_owned(), "c".to_owned()],
            rest: Default::default(),
        };
        set_words(&mut draft, &id, joined).unwrap();
        assert_eq!(draft.materials.texts[0].caption_keywords, [1]);
        // A word whose time is gone loses its mark.
        set_words(&mut draft, &id, columns(0, 100, "a")).unwrap();
        assert!(draft.materials.texts[0].caption_keywords.is_empty());
    }

    #[test]
    fn recognized_words_fill_both_clocks_and_the_string() {
        let (mut draft, id) = draft_with_text();
        set_words(&mut draft, &id, columns(0, 400, "hello")).unwrap();
        let text = &draft.materials.texts[0];
        assert_eq!(text.words.text, ["hello"]);
        assert_eq!(text.current_words.start_time, [0]);
        assert_eq!(text.current_words.end_time, [400]);
        assert_eq!(text.content, "hello");
        assert_eq!(text.base_content, "hello");
    }

    #[test]
    fn a_missing_text_material_is_refused() {
        let mut draft = Draft::empty("D");
        let error =
            set_words(&mut draft, &MaterialId::new("gone"), columns(0, 1, "x")).unwrap_err();
        assert_eq!(error, EditError::MissingMaterial);
        assert!(draft.materials.texts.is_empty());
    }

    #[test]
    fn a_word_that_ends_before_it_starts_is_refused() {
        let (mut draft, id) = draft_with_text();
        let error = set_words(&mut draft, &id, columns(20, 10, "x")).unwrap_err();
        assert_eq!(error, EditError::WordShape);
        assert!(draft.materials.texts[0].content.is_empty());
    }

    #[test]
    fn columns_of_different_lengths_are_refused() {
        let (mut draft, id) = draft_with_text();
        let words = WordColumns {
            start_time: vec![0, 10],
            end_time: vec![10],
            text: vec!["a".to_owned()],
            rest: Default::default(),
        };
        let error = set_words(&mut draft, &id, words).unwrap_err();
        assert_eq!(error, EditError::WordShape);
        assert!(draft.materials.texts[0].words.text.is_empty());
    }

    #[test]
    fn a_caption_lines_up_with_the_clip_and_keeps_its_words() {
        use cutlass_model::{Segment, TimeRange, TrackAttribute, TrackId};

        use crate::request::Request;
        use crate::session::Session;

        let mut draft = Draft::empty("D");
        draft.tracks[0].id = TrackId::new("main");
        draft.tracks[0].segments.push(Segment::new(
            "picture",
            "picture",
            TimeRange::new(1_000_000, 2_000_000),
        ));
        let mut session = Session::open(draft);
        let words = WordColumns {
            start_time: vec![0, 400],
            end_time: vec![400, 900],
            text: vec!["hello".to_owned(), "there".to_owned()],
            rest: Default::default(),
        };
        session
            .apply(
                0,
                &[Request::AddCaption {
                    source_id: "picture".into(),
                    track_id: "captions".into(),
                    segment_id: "line".into(),
                    material_id: "caption".into(),
                    words: words.clone(),
                }],
            )
            .unwrap();
        let draft = session.draft();
        let track = draft
            .tracks
            .iter()
            .find(|track| track.id.as_str() == "captions")
            .expect("track");
        assert_eq!(track.kind, TrackType::Text);
        assert_eq!(track.flag, TrackFlag::TextSubtitle);
        let source = draft
            .tracks
            .iter()
            .find(|track| track.id.as_str() == "main")
            .expect("main")
            .segments[0]
            .target
            .clone();
        assert_eq!(source, TimeRange::new(0, 2_000_000));
        assert_eq!(track.segments[0].target, source);
        assert_eq!(track.segments[0].render_index, 14_000);
        let text = &draft.materials.texts[0];
        assert_eq!(text.kind, CAPTION_KIND);
        assert_eq!(text.content, "hello there");
        assert_eq!(text.words, words);
        assert_eq!(text.current_words, words);
        let pair = &draft
            .relationships
            .iter()
            .find(|item| item.kind == "linkage")
            .expect("link")
            .id_to_id[0];
        assert_eq!(pair.owner.as_str(), "picture");
        assert_eq!(pair.dependent.as_str(), "line");

        session.undo().unwrap();
        assert!(session.draft().materials.texts.is_empty());
        assert!(
            session
                .draft()
                .tracks
                .iter()
                .all(|track| track.id.as_str() != "captions")
        );
        session.redo().unwrap();

        session
            .apply(
                3,
                &[Request::SetTrackAttribute {
                    track_id: TrackId::new("main"),
                    bit: TrackAttribute::LOCK,
                    on: true,
                }],
            )
            .unwrap();
        let error = session
            .apply(
                4,
                &[Request::AddCaption {
                    source_id: "picture".into(),
                    track_id: "captions".into(),
                    segment_id: "other".into(),
                    material_id: "other".into(),
                    words,
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::LockedTrack);
        assert_eq!(session.draft().materials.texts.len(), 1);
        assert_eq!(
            session
                .draft()
                .tracks
                .iter()
                .find(|track| track.id.as_str() == "captions")
                .expect("track")
                .segments
                .len(),
            1
        );

        let before = session.draft().clone();
        let error = session
            .apply(
                4,
                &[Request::AddCaption {
                    source_id: "picture".into(),
                    track_id: "captions".into(),
                    segment_id: "empty".into(),
                    material_id: "empty".into(),
                    words: WordColumns::default(),
                }],
            )
            .unwrap_err();
        assert_eq!(error, EditError::WordShape);
        assert_eq!(session.draft(), &before);
    }

    fn caption_on(draft: &mut Draft, track: &str, id: &str) -> Result<(), EditError> {
        add_caption(
            draft,
            &SegmentId::new("picture"),
            &TrackId::new(track),
            &SegmentId::new(id),
            &MaterialId::new(id),
            columns(0, 400, "hi"),
        )
    }

    fn draft_with_picture() -> Draft {
        let mut draft = Draft::empty("D");
        draft.tracks[0].segments.push(Segment::new(
            "picture",
            "picture",
            cutlass_model::TimeRange::new(0, 2_000_000),
        ));
        draft
    }

    #[test]
    fn a_caption_stays_a_caption_when_its_words_are_cleared() {
        let mut draft = draft_with_picture();
        caption_on(&mut draft, "captions", "line").unwrap();
        set_words(&mut draft, &MaterialId::new("line"), WordColumns::default()).unwrap();
        let text = &draft.materials.texts[0];
        assert!(text.words.text.is_empty());
        assert_eq!(text.kind, CAPTION_KIND);
        let track = draft
            .tracks
            .iter()
            .find(|track| track.id.as_str() == "captions")
            .expect("track");
        assert_eq!(track.flag, TrackFlag::TextSubtitle);
    }

    #[test]
    fn a_text_lane_takes_the_subtitle_flag_only_when_it_holds_only_captions() {
        let mut draft = draft_with_picture();
        text_lane(&mut draft, "empty");
        caption_on(&mut draft, "empty", "first").unwrap();
        assert_eq!(flag(&draft, "empty"), TrackFlag::TextSubtitle);

        text_lane(&mut draft, "titles");
        let mut title = MaterialText::new("title");
        title.kind = "text".to_owned();
        draft.materials.texts.push(title);
        let lane = draft
            .tracks
            .iter_mut()
            .find(|track| track.id.as_str() == "titles")
            .expect("lane");
        lane.segments.push(Segment::new(
            "title",
            "title",
            cutlass_model::TimeRange::new(3_000_000, 1_000_000),
        ));
        caption_on(&mut draft, "titles", "second").unwrap();
        assert_eq!(flag(&draft, "titles"), TrackFlag::None);

        text_lane(&mut draft, "lyrics");
        draft
            .tracks
            .iter_mut()
            .find(|track| track.id.as_str() == "lyrics")
            .expect("lane")
            .flag = TrackFlag::TextLyric;
        caption_on(&mut draft, "lyrics", "third").unwrap();
        assert_eq!(flag(&draft, "lyrics"), TrackFlag::TextLyric);
    }

    fn text_lane(draft: &mut Draft, id: &str) {
        super::super::track::insert_track(draft, &TrackId::new(id), &TrackType::Text).unwrap();
    }

    fn flag(draft: &Draft, id: &str) -> TrackFlag {
        draft
            .tracks
            .iter()
            .find(|track| track.id.as_str() == id)
            .expect("lane")
            .flag
    }

    fn cue(id: &str, start: i64, duration: i64, text: &str) -> SubtitleCue {
        SubtitleCue {
            segment_id: SegmentId::new(id),
            material_id: MaterialId::new(id),
            target: cutlass_model::TimeRange::new(start, duration),
            words: columns(0, duration / 1_000, text),
        }
    }

    #[test]
    fn subtitle_cues_land_as_captions_on_a_new_subtitle_lane_in_one_step() {
        use crate::request::Request;
        use crate::session::Session;

        let mut session = Session::open(Draft::empty("D"));
        let before = session.draft().clone();
        let cues = vec![
            cue("one", 0, 1_000_000, "hello"),
            cue("two", 1_500_000, 500_000, "there"),
        ];
        session
            .apply(
                0,
                &[Request::AddSubtitles {
                    track_id: TrackId::new("subs"),
                    cues: cues.clone(),
                }],
            )
            .unwrap();
        let draft = session.draft();
        let lane = draft
            .tracks
            .iter()
            .find(|track| track.id.as_str() == "subs")
            .expect("lane");
        assert_eq!(lane.kind, TrackType::Text);
        assert_eq!(lane.flag, TrackFlag::TextSubtitle);
        assert_eq!(lane.segments.len(), 2);
        assert_eq!(lane.segments[1].target, cues[1].target);
        assert_eq!(lane.segments[1].render_index, 14_000);
        let two = draft
            .materials
            .texts
            .iter()
            .find(|text| text.id.as_str() == "two")
            .expect("two");
        assert_eq!(two.kind, CAPTION_KIND);
        assert_eq!(two.content, "there");
        assert_eq!(two.words, cues[1].words);
        assert_eq!(two.current_words, cues[1].words);

        session.undo().unwrap();
        assert_eq!(session.draft(), &before);
    }

    #[test]
    fn overlapping_empty_or_clashing_subtitles_leave_the_draft() {
        use crate::request::Request;
        use crate::session::Session;

        let mut session = Session::open(Draft::empty("D"));
        let before = session.draft().clone();
        let refused = |session: &mut Session, track: &str, cues: Vec<SubtitleCue>| {
            session
                .apply(
                    0,
                    &[Request::AddSubtitles {
                        track_id: TrackId::new(track),
                        cues,
                    }],
                )
                .unwrap_err()
        };
        let overlap = vec![
            cue("one", 0, 1_000_000, "a"),
            cue("two", 500_000, 1_000_000, "b"),
        ];
        assert_eq!(refused(&mut session, "subs", overlap), EditError::Overlap);
        assert_eq!(
            refused(&mut session, "subs", Vec::new()),
            EditError::WordShape
        );
        let mut silent = cue("one", 0, 1_000_000, "a");
        silent.words = WordColumns::default();
        assert_eq!(
            refused(&mut session, "subs", vec![silent]),
            EditError::WordShape
        );
        let twice = vec![
            cue("one", 0, 100_000, "a"),
            cue("one", 200_000, 100_000, "b"),
        ];
        assert_eq!(
            refused(&mut session, "subs", twice),
            EditError::DuplicateMaterial
        );
        let main = session.draft().tracks[0].id.as_str().to_owned();
        assert_eq!(
            refused(&mut session, &main, vec![cue("one", 0, 100_000, "a")]),
            EditError::TrackKind
        );
        assert_eq!(session.draft(), &before);
    }

    #[test]
    fn an_inserted_caption_looks_like_its_neighbour_and_takes_its_words() {
        use crate::request::Request;
        use crate::session::Session;

        let mut session = Session::open(Draft::empty("D"));
        session
            .apply(
                0,
                &[Request::AddSubtitles {
                    track_id: TrackId::new("subs"),
                    cues: vec![
                        cue("one", 0, 1_000_000, "hello"),
                        cue("three", 3_000_000, 1_000_000, "bye"),
                    ],
                }],
            )
            .unwrap();
        let mut draft = session.draft().clone();
        let one = draft
            .materials
            .texts
            .iter_mut()
            .find(|t| t.id.as_str() == "one")
            .unwrap();
        one.font_size = 42.0;
        one.caption_keywords = vec![0];
        draft
            .tracks
            .iter_mut()
            .find(|t| t.id.as_str() == "subs")
            .unwrap()
            .segments[0]
            .clip
            .transform
            .y = -0.8;
        let mut session = Session::open(draft);
        let before = session.draft().clone();
        session
            .apply(
                0,
                &[Request::InsertCaption {
                    like: SegmentId::new("one"),
                    cue: cue("two", 1_000_000, 2_000_000, "middle words"),
                }],
            )
            .unwrap();
        let draft = session.draft();
        let lane = draft
            .tracks
            .iter()
            .find(|t| t.id.as_str() == "subs")
            .unwrap();
        let ids: Vec<_> = lane.segments.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, ["one", "two", "three"]);
        assert_eq!(lane.segments[1].clip.transform.y, -0.8);
        let two = draft
            .materials
            .texts
            .iter()
            .find(|t| t.id.as_str() == "two")
            .unwrap();
        assert_eq!(two.font_size, 42.0);
        assert_eq!(two.kind, CAPTION_KIND);
        assert_eq!(two.content, "middle words");
        assert!(two.caption_keywords.is_empty());

        // No room, an unknown neighbour, or no words leave the draft.
        let mut session = Session::open(before.clone());
        for (like, cue) in [
            ("one", cue("x", 500_000, 1_000_000, "x")),
            ("nobody", cue("x", 1_000_000, 500_000, "x")),
            (
                "one",
                SubtitleCue {
                    words: WordColumns::default(),
                    ..cue("x", 1_000_000, 500_000, "x")
                },
            ),
            ("one", cue("three", 1_000_000, 500_000, "x")),
        ] {
            assert!(
                session
                    .apply(
                        0,
                        &[Request::InsertCaption {
                            like: SegmentId::new(like),
                            cue
                        }]
                    )
                    .is_err()
            );
            assert_eq!(session.draft(), &before);
        }
    }
}
