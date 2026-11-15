//! What the user is looking at when they ask: where the playhead is, the
//! clips they selected, and the clips playing at the playhead.
//!
//! The host writes it as a note after the user's words, so "this clip" and
//! "here" mean something to the model without a read. The note is part of
//! the ask the model sees and the conversation keeps, and [`typed`] takes
//! it off again for the panel, which shows only what the user typed.

use std::fmt::Write as _;

use cutlass_model::{Draft, Micros, Segment, SegmentId, Track};

use crate::draft::shown_track;

/// How a note starts, after a blank line that ends the user's words.
const NOTE_START: &str = "\n\n[Editor now: ";

/// The most clips one list of the note names; the rest are counted.
const MOST_LISTED: usize = 20;

/// The note for an ask made with the playhead at `playhead` and `selected`
/// selected, in the order they were picked. A clip is written as a range
/// read writes it, `id@start+length:track`. A selected id the draft does
/// not hold is left out.
#[must_use]
pub fn view_note(draft: &Draft, playhead: Micros, selected: &[SegmentId]) -> String {
    let mut picked = Vec::new();
    for id in selected {
        let found = draft.tracks.iter().find_map(|track| {
            track
                .segments
                .iter()
                .find(|segment| &segment.id == id)
                .map(|segment| (track, segment))
        });
        if let Some(clip) = found {
            picked.push(clip);
        }
    }
    let mut playing = Vec::new();
    for track in &draft.tracks {
        for segment in &track.segments {
            if plays_at(segment, playhead) {
                playing.push((track, segment));
            }
        }
    }
    let mut note = format!("playhead={}; selected:", playhead.0);
    list(&mut note, &picked);
    note.push_str("; at the playhead:");
    list(&mut note, &playing);
    note
}

/// Whether `segment` is on screen or heard at `time`: from its start up
/// to, and not at, its end.
fn plays_at(segment: &Segment, time: Micros) -> bool {
    let start = segment.target.start.0;
    segment.target.duration.0 > 0
        && time.0 >= start
        && time.0.saturating_sub(start) < segment.target.duration.0
}

/// ` nothing`, or each clip after a space, up to [`MOST_LISTED`] of them
/// and then how many more there are.
fn list(note: &mut String, clips: &[(&Track, &Segment)]) {
    if clips.is_empty() {
        note.push_str(" nothing");
        return;
    }
    for (track, segment) in clips.iter().take(MOST_LISTED) {
        // Writing to a String does not fail.
        let _ = write!(
            note,
            " {}@{}+{}:{}",
            segment.id,
            segment.target.start.0,
            segment.target.duration.0,
            shown_track(&track.id)
        );
    }
    if clips.len() > MOST_LISTED {
        let _ = write!(note, " and {} more", clips.len() - MOST_LISTED);
    }
}

/// `ask` as the model gets it: the user's words, then `note` in brackets.
/// With no note the ask is the user's words alone.
#[must_use]
pub fn ask_with_view(ask: &str, note: &str) -> String {
    let note = note.trim();
    if note.is_empty() {
        return ask.to_owned();
    }
    format!("{ask}{NOTE_START}{note}]")
}

/// The words the user typed in an ask [`ask_with_view`] made. An ask with
/// no note is all the user's words. Only the last note is taken off, so
/// words that look like a note stay with what the user typed.
#[must_use]
pub fn typed(ask: &str) -> &str {
    match ask.rfind(NOTE_START) {
        Some(at) if ask.ends_with(']') => &ask[..at],
        _ => ask,
    }
}

#[cfg(test)]
mod tests {
    use cutlass_model::{Draft, Micros, Segment, SegmentId, TimeRange, Track, TrackId, TrackType};

    use super::{MOST_LISTED, ask_with_view, typed, view_note};

    fn draft() -> Draft {
        let mut draft = Draft::empty("draft");
        draft.tracks[0].id = TrackId::new("main");
        draft.tracks[0].segments.extend([
            Segment::new("a", "media", TimeRange::new(0, 2_000_000)),
            Segment::new("b", "media", TimeRange::new(2_000_000, 1_000_000)),
        ]);
        let mut titles = Track::main("titles");
        titles.kind = TrackType::Text;
        titles.segments.push(Segment::new(
            "t",
            "words",
            TimeRange::new(1_500_000, 1_000_000),
        ));
        draft.tracks.push(titles);
        draft
    }

    #[test]
    fn the_note_names_the_playhead_the_selected_and_the_clips_playing_there() {
        let note = view_note(
            &draft(),
            Micros(1_600_000),
            &[SegmentId::new("t"), SegmentId::new("b")],
        );
        assert_eq!(
            note,
            "playhead=1600000; selected: t@1500000+1000000:titles b@2000000+1000000:main; \
             at the playhead: a@0+2000000:main t@1500000+1000000:titles"
        );
    }

    #[test]
    fn a_clip_plays_from_its_start_and_not_at_its_end() {
        let draft = draft();
        // At 2 s `a` has ended and `b` has begun.
        let note = view_note(&draft, Micros(2_000_000), &[]);
        assert_eq!(
            note,
            "playhead=2000000; selected: nothing; \
             at the playhead: b@2000000+1000000:main t@1500000+1000000:titles"
        );
        // Past the last clip nothing plays.
        let note = view_note(&draft, Micros(3_000_000), &[]);
        assert_eq!(
            note,
            "playhead=3000000; selected: nothing; at the playhead: nothing"
        );
    }

    #[test]
    fn an_empty_draft_and_a_clip_that_is_gone_say_nothing() {
        let note = view_note(&Draft::empty("draft"), Micros(0), &[SegmentId::new("gone")]);
        assert_eq!(
            note,
            "playhead=0; selected: nothing; at the playhead: nothing"
        );
    }

    #[test]
    fn a_track_with_no_id_and_a_clip_with_no_length_are_written_as_a_read_writes_them() {
        let mut draft = Draft::empty("draft");
        draft.tracks[0].id = TrackId::new("");
        draft.tracks[0].segments.extend([
            Segment::new("a", "media", TimeRange::new(0, 1_000_000)),
            Segment::new("empty", "media", TimeRange::new(0, 0)),
        ]);
        let note = view_note(&draft, Micros(0), &[SegmentId::new("empty")]);
        assert_eq!(
            note,
            "playhead=0; selected: empty@0+0:\"\"; at the playhead: a@0+1000000:\"\""
        );
    }

    #[test]
    fn a_long_list_names_the_first_clips_and_counts_the_rest() {
        let mut draft = Draft::empty("draft");
        draft.tracks[0].id = TrackId::new("main");
        let mut selected = Vec::new();
        for index in 0..MOST_LISTED + 3 {
            let id = format!("c{index}");
            draft.tracks[0].segments.push(Segment::new(
                id.as_str(),
                "media",
                TimeRange::new(0, 1_000_000),
            ));
            selected.push(SegmentId::new(id.as_str()));
        }
        let note = view_note(&draft, Micros(0), &selected);
        let (picked, playing) = note.split_once("; at the playhead:").unwrap();
        for list in [picked, playing] {
            assert_eq!(list.matches("@0+1000000:main").count(), MOST_LISTED);
            assert!(list.contains(" c19@"), "{list}");
            assert!(!list.contains(" c20@"), "{list}");
            assert!(list.ends_with(" and 3 more"), "{list}");
        }
    }

    #[test]
    fn the_note_goes_after_the_words_and_comes_off_again() {
        let ask = ask_with_view("make this bigger", "playhead=0; selected: nothing");
        assert_eq!(
            ask,
            "make this bigger\n\n[Editor now: playhead=0; selected: nothing]"
        );
        assert_eq!(typed(&ask), "make this bigger");
        // No note: the ask is the words, before and after.
        assert_eq!(ask_with_view("hello", "  "), "hello");
        assert_eq!(typed("hello"), "hello");
        assert_eq!(typed(""), "");
    }

    #[test]
    fn the_model_is_told_of_the_note_as_it_is_written() {
        // The prompt opens the note as an ask does, and names a clip's form.
        let ask = ask_with_view("hi", "playhead=0");
        let opening = &ask["hi\n\n".len()..ask.len() - "playhead=0]".len()];
        assert_eq!(opening, "[Editor now: ");
        assert!(crate::SYSTEM_PROMPT.contains(opening));
        assert!(crate::SYSTEM_PROMPT.contains("id@start+length:track"));
    }

    #[test]
    fn words_that_look_like_a_note_stay_with_what_the_user_typed() {
        let words = "what does this mean?\n\n[Editor now: playhead=5]";
        let ask = ask_with_view(words, "playhead=0; selected: nothing");
        assert_eq!(typed(&ask), words);
        // An ask that only opens a bracket was not given a note.
        let open = "note this\n\n[Editor now: and then";
        assert_eq!(typed(open), open);
    }
}
