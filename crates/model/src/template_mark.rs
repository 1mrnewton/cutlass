//! What a clip is in a project template.
//!
//! A project template is a finished edit with some clips marked. A slot is
//! a clip the person using the template fills with their own photo or
//! video; editable text is a text clip whose words they change; music is
//! the one sound they may swap. Everything else stays as the author made
//! it. The mark is the app's own: the reference editor does not read it.

use serde::{Deserialize, Serialize};

use crate::draft::Draft;
use crate::id::SegmentId;
use crate::materials::MaterialRef;
use crate::meta::MetaType;
use crate::time::Micros;

/// What a slot takes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotTakes {
    /// A photo or a video.
    #[default]
    Either,
    /// A video only.
    Video,
    /// A photo only.
    Photo,
}

impl SlotTakes {
    /// Whether a medium takes this slot; `photo` is a still.
    #[must_use]
    pub const fn accepts(self, photo: bool) -> bool {
        match self {
            Self::Either => true,
            Self::Video => !photo,
            Self::Photo => photo,
        }
    }
}

/// A clip's mark in a project template, stored on the segment as
/// `cutlass_template`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum TemplateMark {
    /// A photo or video clip the user fills with their own media.
    Slot {
        #[serde(default)]
        takes: SlotTakes,
    },
    /// A text clip whose words the user may change.
    Text,
    /// The sound the user may swap for their own.
    Music,
}

/// One slot of a template, as the person filling it sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemplateSlot {
    pub segment_id: SegmentId,
    pub takes: SlotTakes,
    /// Where the slot starts on the timeline.
    pub start: Micros,
    /// How long it plays on the timeline.
    pub length: Micros,
    /// How much of a video it plays: its length times its speed. A video
    /// shorter than this does not fill it; a photo always does.
    pub needs: Micros,
    /// The sample in it is a photo.
    pub sample_is_photo: bool,
}

/// The slots of `draft` in the order a person fills them: by start time,
/// then by the draft's row order. Slots inside a compound clip are not
/// counted: a compound clip plays as one piece.
#[must_use]
pub fn template_slots(draft: &Draft) -> Vec<TemplateSlot> {
    let mut slots: Vec<(usize, TemplateSlot)> = Vec::new();
    for (row, track) in draft.tracks.iter().enumerate() {
        for segment in &track.segments {
            let Some(TemplateMark::Slot { takes }) = segment.template_mark else {
                continue;
            };
            let Some(MaterialRef::Video(video)) = draft.materials.get(segment.material_id.as_str())
            else {
                continue;
            };
            let needs = segment
                .source_range
                .as_ref()
                .map_or(segment.target.duration, |range| range.duration);
            slots.push((
                row,
                TemplateSlot {
                    segment_id: segment.id.clone(),
                    takes,
                    start: segment.target.start,
                    length: segment.target.duration,
                    needs,
                    sample_is_photo: video.kind == MetaType::Photo.as_str(),
                },
            ));
        }
    }
    slots.sort_by_key(|(row, slot)| (slot.start.0, *row));
    slots.into_iter().map(|(_, slot)| slot).collect()
}

/// The text clips of `draft` whose words the person may change, by start time.
#[must_use]
pub fn template_texts(draft: &Draft) -> Vec<SegmentId> {
    marked(draft, TemplateMark::Text)
}

/// The music clip of `draft` the person may swap, if one is marked.
#[must_use]
pub fn template_music(draft: &Draft) -> Option<SegmentId> {
    marked(draft, TemplateMark::Music).into_iter().next()
}

fn marked(draft: &Draft, mark: TemplateMark) -> Vec<SegmentId> {
    let mut found: Vec<(i64, usize, SegmentId)> = draft
        .tracks
        .iter()
        .enumerate()
        .flat_map(|(row, track)| track.segments.iter().map(move |segment| (row, segment)))
        .filter(|(_, segment)| segment.template_mark == Some(mark))
        .map(|(row, segment)| (segment.target.start.0, row, segment.id.clone()))
        .collect();
    found.sort();
    found.into_iter().map(|(_, _, id)| id).collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{SlotTakes, TemplateMark, template_music, template_slots, template_texts};
    use crate::{Draft, MaterialVideo, MetaType, Segment, TimeRange, Track};

    fn slot(id: &str, material: &str, start: i64) -> Segment {
        let mut segment = Segment::new(id, material, TimeRange::new(start, 2_000_000));
        segment.template_mark = Some(TemplateMark::Slot {
            takes: SlotTakes::Either,
        });
        segment
    }

    #[test]
    fn slots_fill_by_start_then_row_and_say_what_they_need() {
        let mut draft = Draft::empty("D");
        let mut video = MaterialVideo::new("v");
        video.kind = MetaType::Video.as_str().to_owned();
        let mut photo = MaterialVideo::new("p");
        photo.kind = MetaType::Photo.as_str().to_owned();
        draft.materials.videos.extend([video, photo]);
        let mut fast = slot("late", "v", 4_000_000);
        fast.speed = 2.0;
        fast.source_range = Some(TimeRange::new(0, 4_000_000));
        draft.tracks[0].segments.extend([
            fast,
            slot("first", "p", 0),
            Segment::new("plain", "v", TimeRange::new(2_000_000, 2_000_000)),
        ]);
        let mut overlay = Track::main("over");
        overlay.segments.push(slot("same-start", "v", 0));
        // A slot on something that is not a photo or video is not counted.
        overlay.segments.push(slot("odd", "missing", 9_000_000));
        draft.tracks.push(overlay);
        let slots = template_slots(&draft);
        let ids: Vec<&str> = slots.iter().map(|slot| slot.segment_id.as_str()).collect();
        assert_eq!(ids, ["first", "same-start", "late"]);
        assert!(slots[0].sample_is_photo);
        assert_eq!(slots[2].needs.0, 4_000_000);
        assert_eq!(slots[2].length.0, 2_000_000);
        assert_eq!(slots[0].needs.0, 2_000_000);
    }

    #[test]
    fn texts_and_music_are_found_by_their_marks() {
        let mut draft = Draft::empty("D");
        let mut words = Segment::new("w2", "t", TimeRange::new(5, 1));
        words.template_mark = Some(TemplateMark::Text);
        let mut early = Segment::new("w1", "t", TimeRange::new(0, 1));
        early.template_mark = Some(TemplateMark::Text);
        let mut song = Segment::new("song", "a", TimeRange::new(0, 1));
        song.template_mark = Some(TemplateMark::Music);
        draft.tracks[0].segments.extend([words, early, song]);
        let texts: Vec<String> = template_texts(&draft)
            .into_iter()
            .map(|id| id.as_str().to_owned())
            .collect();
        assert_eq!(texts, ["w1", "w2"]);
        assert_eq!(template_music(&draft).unwrap().as_str(), "song");
        assert!(template_slots(&Draft::empty("E")).is_empty());
    }

    #[test]
    fn marks_store_as_a_role_and_read_back() {
        let marks = [
            (
                TemplateMark::Slot {
                    takes: SlotTakes::Video,
                },
                json!({"role": "slot", "takes": "video"}),
            ),
            (TemplateMark::Text, json!({"role": "text"})),
            (TemplateMark::Music, json!({"role": "music"})),
        ];
        for (mark, stored) in marks {
            assert_eq!(serde_json::to_value(mark).unwrap(), stored);
            assert_eq!(
                serde_json::from_value::<TemplateMark>(stored).unwrap(),
                mark
            );
        }
    }

    #[test]
    fn a_slot_without_takes_takes_either() {
        let mark: TemplateMark = serde_json::from_value(json!({"role": "slot"})).unwrap();
        assert_eq!(
            mark,
            TemplateMark::Slot {
                takes: SlotTakes::Either
            }
        );
    }

    #[test]
    fn an_unknown_role_is_refused() {
        assert!(serde_json::from_value::<TemplateMark>(json!({"role": "logo"})).is_err());
    }

    #[test]
    fn what_each_slot_accepts() {
        assert!(SlotTakes::Either.accepts(true) && SlotTakes::Either.accepts(false));
        assert!(SlotTakes::Video.accepts(false) && !SlotTakes::Video.accepts(true));
        assert!(SlotTakes::Photo.accepts(true) && !SlotTakes::Photo.accepts(false));
    }
}
