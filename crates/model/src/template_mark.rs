//! What a clip is in a project template.
//!
//! A project template is a finished edit with some clips marked. A slot is
//! a clip the person using the template fills with their own photo or
//! video; editable text is a text clip whose words they change; music is
//! the one sound they may swap. Everything else stays as the author made
//! it. The mark is the app's own: the reference editor does not read it.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::draft::Draft;
use crate::id::{MaterialId, SegmentId};
use crate::materials::{MaterialDraft, MaterialRef};
use crate::segment::Segment;
use crate::time::{Micros, TimeRange};

#[allow(clippy::trivially_copy_pass_by_ref)]
const fn is_false(value: &bool) -> bool {
    !*value
}

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
        /// The person put their own media here; the sample is gone.
        #[serde(default, skip_serializing_if = "is_false")]
        filled: bool,
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
    /// The person already put their own media here.
    pub filled: bool,
}

/// The slots of `draft` in the order a person fills them: by start time,
/// then by the draft's row order. Slots inside a compound clip are not
/// counted: a compound clip plays as one piece.
#[must_use]
pub fn template_slots(draft: &Draft) -> Vec<TemplateSlot> {
    let mut slots: Vec<(usize, TemplateSlot)> = Vec::new();
    for (row, track) in draft.tracks.iter().enumerate() {
        for segment in &track.segments {
            let Some(TemplateMark::Slot { takes, filled }) = segment.template_mark else {
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
                    sample_is_photo: video.is_still(),
                    filled,
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

/// The template clip `segment_id` of `draft`: its material's id and the
/// draft inside it. `None` for any other clip, a compound clip included.
#[must_use]
pub fn template_clip<'a>(
    draft: &'a Draft,
    segment_id: &SegmentId,
) -> Option<(&'a MaterialDraft, &'a Draft)> {
    let segment = draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .find(|segment| &segment.id == segment_id)?;
    let MaterialRef::Draft(material) = draft.materials.get(segment.material_id.as_str())? else {
        return None;
    };
    if !material.template {
        return None;
    }
    Some((material, material.draft.as_deref()?))
}

/// How long `draft` plays: the end of its last clip.
#[must_use]
pub fn template_length(draft: &Draft) -> Micros {
    Micros(
        draft
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .map(|segment| {
                segment
                    .target
                    .start
                    .0
                    .saturating_add(segment.target.duration.0)
            })
            .max()
            .unwrap_or_default(),
    )
}

/// The material a template clip plays: `inside` as one closed piece named
/// `name`. The inside keeps its clips' gaps, like any nested draft.
///
/// The inside takes `id` as its own id, and every clip and material in it
/// has its id put under `id`, as `id.old`. A template comes from another
/// project, so its ids could otherwise be ones this project already uses,
/// and sound and pictures are looked up by clip id.
#[must_use]
pub fn template_material(id: MaterialId, name: &str, inside: Draft) -> MaterialDraft {
    let mut inside = under(inside, id.as_str());
    inside.id = id.as_str().into();
    inside.config.maintrack_adsorb = false;
    inside.duration = template_length(&inside);
    MaterialDraft {
        id,
        name: name.to_owned(),
        combination_type: "compound".to_owned(),
        draft: Some(Box::new(inside)),
        template: true,
        ..MaterialDraft::default()
    }
}

/// `draft` with every clip and material id, at any depth, renamed to
/// `prefix.id`, and every mention of one renamed with it. The draft comes
/// back as it was if it cannot be read as JSON and back.
fn under(draft: Draft, prefix: &str) -> Draft {
    fn take(item: &Value, ids: &mut BTreeSet<String>) {
        if let Some(id) = item.get("id").and_then(Value::as_str)
            && !id.is_empty()
        {
            ids.insert(id.to_owned());
        }
    }
    fn list(value: Option<&Value>) -> impl Iterator<Item = &Value> {
        value.and_then(Value::as_array).into_iter().flatten()
    }
    fn collect(draft: &Value, ids: &mut BTreeSet<String>) {
        for track in list(draft.get("tracks")) {
            for segment in list(track.get("segments")) {
                take(segment, ids);
            }
        }
        let pools = draft.get("materials").and_then(Value::as_object);
        for pool in pools.into_iter().flat_map(|pools| pools.values()) {
            for item in list(Some(pool)) {
                take(item, ids);
                if let Some(child) = item.get("draft").filter(|child| child.is_object()) {
                    collect(child, ids);
                }
            }
        }
    }
    fn rename(value: &mut Value, ids: &BTreeSet<String>, prefix: &str) {
        match value {
            Value::String(text) if ids.contains(text.as_str()) => {
                *text = format!("{prefix}.{text}");
            }
            Value::Array(items) => {
                for item in items {
                    rename(item, ids, prefix);
                }
            }
            Value::Object(map) => {
                for (key, item) in map.iter_mut() {
                    // What a thing is, never which one: an id that reads
                    // like a kind must not rename the kind.
                    if !(matches!(key.as_str(), "type" | "role") && item.is_string()) {
                        rename(item, ids, prefix);
                    }
                }
            }
            _ => {}
        }
    }
    let Ok(mut value) = serde_json::to_value(&draft) else {
        return draft;
    };
    let mut ids = BTreeSet::new();
    collect(&value, &mut ids);
    rename(&mut value, &ids, prefix);
    serde_json::from_value(value).unwrap_or(draft)
}

/// The clip that plays the whole of the template `material` from `start`.
/// `None` when the template plays nothing.
#[must_use]
pub fn template_segment(id: SegmentId, material: &MaterialDraft, start: Micros) -> Option<Segment> {
    let length = template_length(material.draft.as_deref()?);
    if length.0 <= 0 {
        return None;
    }
    let mut segment = Segment::new(id, material.id.clone(), TimeRange::new(start.0, length.0));
    segment.source_range = Some(TimeRange::new(0, length.0));
    Some(segment)
}

/// How many slots of `draft` still hold their sample.
#[must_use]
pub fn template_unfilled(draft: &Draft) -> usize {
    template_slots(draft)
        .iter()
        .filter(|slot| !slot.filled)
        .count()
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

impl TemplateMark {
    /// A slot that takes `takes` and still holds its sample.
    #[must_use]
    pub const fn slot(takes: SlotTakes) -> Self {
        Self::Slot {
            takes,
            filled: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        SlotTakes, TemplateMark, template_clip, template_material, template_music,
        template_segment, template_slots, template_texts, template_unfilled,
    };
    use crate::{Draft, MaterialVideo, MetaType, Segment, TimeRange, Track};

    fn slot(id: &str, material: &str, start: i64) -> Segment {
        let mut segment = Segment::new(id, material, TimeRange::new(start, 2_000_000));
        segment.template_mark = Some(TemplateMark::slot(SlotTakes::Either));
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
                TemplateMark::slot(SlotTakes::Video),
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
        assert_eq!(mark, TemplateMark::slot(SlotTakes::Either));
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

    #[test]
    fn a_template_clip_plays_the_whole_inside_as_one_closed_piece() {
        let mut inside = Draft::empty("in");
        let mut video = MaterialVideo::new("v");
        video.kind = MetaType::Video.as_str().to_owned();
        inside.materials.videos.push(video);
        // An id that reads like a mark's role leaves the role alone.
        let mut filled = slot("slot", "v", 0);
        filled.template_mark = Some(TemplateMark::Slot {
            takes: SlotTakes::Either,
            filled: true,
        });
        inside.tracks[0]
            .segments
            .extend([filled, slot("b", "v", 3_000_000)]);
        let material = template_material("tpl".into(), "Trip", inside);
        assert!(material.template);
        let segment = template_segment("clip".into(), &material, crate::Micros(1_000_000)).unwrap();
        assert_eq!(segment.target, TimeRange::new(1_000_000, 5_000_000));
        assert_eq!(segment.source_range, Some(TimeRange::new(0, 5_000_000)));
        let mut draft = Draft::empty("D");
        draft.materials.drafts.push(material);
        draft.tracks[0].segments.push(segment);
        let (found, inside) = template_clip(&draft, &"clip".into()).unwrap();
        assert_eq!(found.name, "Trip");
        assert_eq!(template_unfilled(inside), 1);
        // The project's own slots are not the template's.
        assert!(template_slots(&draft).is_empty());
        let empty = template_material("none".into(), "Empty", Draft::empty("e"));
        assert!(template_segment("x".into(), &empty, crate::Micros(0)).is_none());
    }
}
