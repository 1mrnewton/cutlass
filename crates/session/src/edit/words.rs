use cutlass_model::{
    Draft, MaterialId, MaterialText, Segment, SegmentId, TrackId, TrackType, WordColumns,
};

use super::EditError;

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
    text.words = words.clone();
    text.current_words = words;
    text.content.clone_from(&content);
    text.base_content = content;
    Ok(())
}

/// Place one caption on `track_id` for the source clip and store `words`.
///
/// A missing track is added as text. The caption keeps the source clip's
/// timeline range. A link is recorded when the source can own one.
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
    let (source_track, source_index) = super::speed::locate(draft, source_id)?;
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
    if let Some(track) = draft.tracks.iter().find(|track| &track.id == track_id) {
        if !matches!(track.kind, TrackType::Text | TrackType::TextToVideoText) {
            return Err(EditError::TrackKind);
        }
        super::ensure_unlocked(track.attribute)?;
    } else {
        super::track::insert_track(draft, track_id, &TrackType::Text)?;
    }
    let mut text = MaterialText::new(material_id.clone());
    text.kind = "text".to_owned();
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
}
