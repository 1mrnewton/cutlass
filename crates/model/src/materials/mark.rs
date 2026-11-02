use serde_json::Value;

use crate::id::{MaterialId, SegmentId};
use crate::materials::{Materials, RawMaterial};
use crate::meta::MetaType;
use crate::segment::Segment;
use crate::time::Micros;

/// One marker. `start` is a timeline time on the ruler, or an offset from
/// the clip's target start. `duration` is microseconds. Zero is a point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimeMarkItem {
    pub start: Micros,
    pub duration: Micros,
    pub color: String,
    pub title: String,
}

impl TimeMarkItem {
    #[must_use]
    pub fn point(start: Micros, color: impl Into<String>) -> Self {
        Self {
            start,
            duration: Micros(0),
            color: color.into(),
            title: String::new(),
        }
    }
}

/// A marker the timeline can draw. `stored_start` is the value in the file.
/// `time` is where it sits on the timeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisibleMark {
    pub segment_id: Option<SegmentId>,
    pub stored_start: i64,
    pub time: Micros,
    pub color: String,
    pub title: String,
}

/// A new time-mark material with one item.
#[must_use]
pub fn new_time_mark(id: MaterialId, item: TimeMarkItem) -> RawMaterial {
    let mut raw = RawMaterial {
        id,
        kind: MetaType::TimeMark.as_str().to_owned(),
        ..RawMaterial::default()
    };
    set_time_mark_items(&mut raw, &[item]);
    raw
}

/// Replace `mark_items`. Other fields on the material stay.
pub fn set_time_mark_items(raw: &mut RawMaterial, items: &[TimeMarkItem]) {
    let values = items.iter().map(item_value).collect();
    raw.fields
        .insert("mark_items".to_owned(), Value::Array(values));
}

/// Items stored on a time-mark material. An item without a start is skipped.
#[must_use]
pub fn time_mark_items(raw: &RawMaterial) -> Vec<TimeMarkItem> {
    let Some(items) = raw.fields.get("mark_items").and_then(Value::as_array) else {
        return Vec::new();
    };
    items.iter().filter_map(one_item).collect()
}

/// The time-mark material this clip points at, when it points at one.
#[must_use]
pub fn segment_time_mark<'a>(
    materials: &'a Materials,
    segment: &Segment,
) -> Option<&'a RawMaterial> {
    segment
        .extra_material_refs
        .iter()
        .find_map(|id| materials.time_marks.iter().find(|item| &item.id == id))
}

/// Timeline time of a clip marker. An offset outside the clip is `None`.
#[must_use]
pub fn clip_mark_time(segment: &Segment, item: &TimeMarkItem) -> Option<Micros> {
    if item.start.0 < 0 || item.start.0 >= segment.target.duration.0 {
        return None;
    }
    segment.target.start.0.checked_add(item.start.0).map(Micros)
}

/// Time-mark materials a clip points at. A ruler mark is not in this list.
#[must_use]
pub fn referenced_time_marks(
    materials: &Materials,
    tracks: &[crate::track::Track],
) -> Vec<MaterialId> {
    let mut ids = Vec::new();
    for track in tracks {
        for segment in &track.segments {
            for id in &segment.extra_material_refs {
                if materials.time_marks.iter().any(|item| &item.id == id)
                    && !ids.iter().any(|seen| seen == id)
                {
                    ids.push(id.clone());
                }
            }
        }
    }
    ids
}

/// Ruler marks, and clip marks that still sit on their clip.
#[must_use]
pub fn visible_marks(draft: &crate::draft::Draft) -> Vec<VisibleMark> {
    let owned = referenced_time_marks(&draft.materials, &draft.tracks);
    let mut marks = Vec::new();
    for track in &draft.tracks {
        for segment in &track.segments {
            let Some(raw) = segment_time_mark(&draft.materials, segment) else {
                continue;
            };
            for item in time_mark_items(raw) {
                let Some(time) = clip_mark_time(segment, &item) else {
                    continue;
                };
                marks.push(VisibleMark {
                    segment_id: Some(segment.id.clone()),
                    stored_start: item.start.0,
                    time,
                    color: item.color.clone(),
                    title: item.title.clone(),
                });
            }
        }
    }
    for raw in &draft.materials.time_marks {
        if owned.iter().any(|id| id == &raw.id) {
            continue;
        }
        for item in time_mark_items(raw) {
            marks.push(VisibleMark {
                segment_id: None,
                stored_start: item.start.0,
                time: item.start,
                color: item.color.clone(),
                title: item.title.clone(),
            });
        }
    }
    marks
}

fn item_value(item: &TimeMarkItem) -> Value {
    serde_json::json!({
        "time_range": {
            "start": item.start.0,
            "duration": item.duration.0,
        },
        "color": item.color,
        "title": item.title,
    })
}

fn one_item(item: &Value) -> Option<TimeMarkItem> {
    let start = item
        .get("time_range")
        .and_then(|range| range.get("start"))
        .and_then(Value::as_i64)?;
    let duration = item
        .get("time_range")
        .and_then(|range| range.get("duration"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let color = item
        .get("color")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    let title = item
        .get("title")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    Some(TimeMarkItem {
        start: Micros(start),
        duration: Micros(duration),
        color,
        title,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draft::Draft;
    use crate::time::TimeRange;

    #[test]
    fn a_clip_mark_is_an_offset_and_a_ruler_mark_is_a_time() {
        let item = TimeMarkItem::point(Micros(250_000), "#ffffff");
        let raw = new_time_mark(MaterialId::new("marks"), item.clone());
        assert_eq!(raw.kind, "time_mark");
        assert_eq!(time_mark_items(&raw), vec![item]);

        let mut clip = Segment::new("clip", "media", TimeRange::new(1_000_000, 1_000_000));
        clip.extra_material_refs.push(MaterialId::new("marks"));
        let mut draft = Draft::empty("D");
        draft.tracks[0].segments.push(clip);
        draft.materials.time_marks.push(raw);

        let marks = visible_marks(&draft);
        assert_eq!(marks.len(), 1);
        assert_eq!(
            marks[0].segment_id.as_ref().map(SegmentId::as_str),
            Some("clip")
        );
        assert_eq!(marks[0].stored_start, 250_000);
        assert_eq!(marks[0].time, Micros(1_250_000));

        draft.tracks[0].segments[0].extra_material_refs.clear();
        let marks = visible_marks(&draft);
        assert!(marks[0].segment_id.is_none());
        assert_eq!(marks[0].time, Micros(250_000));
    }

    #[test]
    fn a_mark_past_the_clip_is_kept_off_the_timeline() {
        let raw = new_time_mark(
            MaterialId::new("marks"),
            TimeMarkItem::point(Micros(2_000_000), "#ffffff"),
        );
        let mut clip = Segment::new("clip", "media", TimeRange::new(0, 1_000_000));
        clip.extra_material_refs.push(raw.id.clone());
        let mut draft = Draft::empty("D");
        draft.tracks[0].segments.push(clip);
        draft.materials.time_marks.push(raw);
        assert!(visible_marks(&draft).is_empty());
        assert_eq!(time_mark_items(&draft.materials.time_marks[0]).len(), 1);
    }
}
