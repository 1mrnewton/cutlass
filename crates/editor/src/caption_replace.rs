//! Find and replace across every caption on a lane, as one undo step.
//!
//! Matching ignores case, as the Captions list's search does. Each caption
//! that changes is rematched, so the words that stay keep their times.

use cutlass_model::{Draft, SegmentId};
use cutlass_session::Request;

use crate::caption_list::{CaptionListMiss, caption_rows, caption_text_request};

/// What a find and replace across the captions comes to.
#[derive(Clone, Debug, PartialEq)]
pub struct Replaced {
    /// One request per caption that changes.
    pub requests: Vec<Request>,
    /// How many times `find` was replaced.
    pub count: usize,
}

/// `text` with every match of `find`, case ignored, replaced by `with`,
/// and how many there were. An empty `find` matches nothing.
#[must_use]
pub fn replace_ignoring_case(text: &str, find: &str, with: &str) -> (String, usize) {
    let needle: Vec<char> = find.chars().flat_map(char::to_lowercase).collect();
    if needle.is_empty() {
        return (text.to_owned(), 0);
    }
    let mut out = String::with_capacity(text.len());
    let mut count = 0;
    let mut rest = text;
    while let Some(first) = rest.chars().next() {
        match matched_len(rest, &needle) {
            Some(len) => {
                out.push_str(with);
                count += 1;
                rest = &rest[len..];
            }
            None => {
                out.push(first);
                rest = &rest[first.len_utf8()..];
            }
        }
    }
    (out, count)
}

/// The byte length of the start of `text` that is `needle`, case ignored.
fn matched_len(text: &str, needle: &[char]) -> Option<usize> {
    let mut want = needle.iter();
    let mut pending = want.next()?;
    for (at, letter) in text.char_indices() {
        for lower in letter.to_lowercase() {
            if lower != *pending {
                return None;
            }
            match want.next() {
                Some(next) => pending = next,
                None => return Some(at + letter.len_utf8()),
            }
        }
    }
    None
}

/// The requests that replace `find` with `with` in every caption on
/// caption `segment_id`'s lane, as one undo step. A caption left with no
/// words refuses the whole replace, so none is emptied by accident.
pub fn replace_in_captions(
    draft: &Draft,
    segment_id: &SegmentId,
    find: &str,
    with: &str,
) -> Result<Replaced, CaptionListMiss> {
    let rows = caption_rows(draft, segment_id);
    if rows.is_empty() {
        return Err(CaptionListMiss::NotCaption);
    }
    let mut requests = Vec::new();
    let mut count = 0;
    for row in rows {
        let (text, found) = replace_ignoring_case(&row.text, find, with);
        if found == 0 {
            continue;
        }
        count += found;
        if let Some(request) = caption_text_request(draft, &row.segment_id, text.trim())? {
            requests.push(request);
        }
    }
    Ok(Replaced { requests, count })
}

#[cfg(test)]
mod tests {
    use cutlass_model::{MaterialRef, TrackFlag};
    use cutlass_session::Session;

    use super::*;

    fn session() -> Session {
        let srt = "00:00:00,000 --> 00:00:02,000\nthe cat sat\n\n00:00:03,000 --> 00:00:05,000\nThe Cat ran\n\n00:00:06,000 --> 00:00:07,000\ndog\n";
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

    fn first(session: &Session) -> SegmentId {
        session
            .draft()
            .tracks
            .iter()
            .find(|track| track.flag == TrackFlag::TextSubtitle)
            .unwrap()
            .segments[0]
            .id
            .clone()
    }

    fn texts(session: &Session) -> Vec<String> {
        caption_rows(session.draft(), &first(session))
            .into_iter()
            .map(|row| row.text)
            .collect()
    }

    #[test]
    fn a_match_ignores_case_and_keeps_the_rest() {
        assert_eq!(
            replace_ignoring_case("The cat, the CAT", "cat", "dog"),
            ("The dog, the dog".to_owned(), 2)
        );
        assert_eq!(
            replace_ignoring_case("aaa", "aa", "b"),
            ("ba".to_owned(), 1)
        );
        assert_eq!(
            replace_ignoring_case("Straße", "STRASSE", "x"),
            ("Straße".to_owned(), 0)
        );
        assert_eq!(
            replace_ignoring_case("ÉTÉ été", "été", "x"),
            ("x x".to_owned(), 2)
        );
        assert_eq!(replace_ignoring_case("cat", "", "x"), ("cat".to_owned(), 0));
        assert_eq!(replace_ignoring_case("", "cat", "x"), (String::new(), 0));
    }

    #[test]
    fn a_replace_reaches_every_caption_in_one_step_and_keeps_word_times() {
        let mut session = session();
        let before = session.draft().clone();
        let replaced =
            replace_in_captions(session.draft(), &first(&session), "cat", "dog").unwrap();
        assert_eq!(replaced.count, 2);
        assert_eq!(replaced.requests.len(), 2);
        let starts = |session: &Session| {
            let draft = session.draft();
            let segment = &draft
                .tracks
                .iter()
                .find(|track| track.flag == TrackFlag::TextSubtitle)
                .unwrap()
                .segments[0];
            match draft.materials.get(segment.material_id.as_str()) {
                Some(MaterialRef::Text(text)) => text.words.start_time.clone(),
                _ => panic!("not text"),
            }
        };
        let times = starts(&session);
        session
            .apply(session.revision(), &replaced.requests)
            .unwrap();
        assert_eq!(texts(&session), ["the dog sat", "The dog ran", "dog"]);
        assert_eq!(starts(&session), times);
        session.undo().unwrap();
        assert_eq!(session.draft(), &before);
    }

    #[test]
    fn a_replace_that_finds_nothing_or_empties_a_caption_changes_nothing() {
        let session = session();
        let id = first(&session);
        let none = replace_in_captions(session.draft(), &id, "bird", "x").unwrap();
        assert_eq!(
            none,
            Replaced {
                requests: Vec::new(),
                count: 0
            }
        );
        assert_eq!(
            replace_in_captions(session.draft(), &id, "dog", " "),
            Err(CaptionListMiss::Empty)
        );
        assert_eq!(
            replace_in_captions(session.draft(), &SegmentId::from("nope"), "cat", "x"),
            Err(CaptionListMiss::NotCaption)
        );
    }
}
