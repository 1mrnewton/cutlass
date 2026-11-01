use serde_json::Value;

use crate::enums::TrackType;
use crate::id::{MaterialId, SegmentId};
use crate::materials::{Materials, RawMaterial};
use crate::meta::MetaType;
use crate::segment::Segment;
use crate::time::Micros;

/// A beat the timeline can draw. `source` is the stored source time.
/// `time` is where it sits on the timeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisibleBeat {
    pub segment_id: SegmentId,
    pub source: i64,
    pub time: Micros,
}

/// A beat belongs on an audio lane.
#[must_use]
pub fn track_holds_beats(kind: &TrackType) -> bool {
    matches!(kind, TrackType::Audio | TrackType::TextToVideoTextAudio)
}

/// A new beats material with one source time.
#[must_use]
pub fn new_beats(id: MaterialId, source: i64) -> RawMaterial {
    let mut raw = RawMaterial {
        id,
        kind: MetaType::Beats.as_str().to_owned(),
        ..RawMaterial::default()
    };
    raw.fields
        .insert("enable_ai_beats".to_owned(), Value::Bool(false));
    set_user_beats(&mut raw, &[source]);
    raw
}

/// Replace `user_beats`. Other fields on the material stay.
pub fn set_user_beats(raw: &mut RawMaterial, beats: &[i64]) {
    let values = beats.iter().copied().map(Value::from).collect();
    raw.fields
        .insert("user_beats".to_owned(), Value::Array(values));
}

/// Source times stored on a beats material.
///
/// A single integer and an array of integers are both read. Anything else
/// contributes nothing.
#[must_use]
pub fn user_beats(raw: &RawMaterial) -> Vec<i64> {
    let Some(value) = raw.fields.get("user_beats") else {
        return Vec::new();
    };
    let mut beats = Vec::new();
    push_beats(value, &mut beats);
    beats
}

/// The beats material this clip points at, when it points at one.
#[must_use]
pub fn segment_beats<'a>(materials: &'a Materials, segment: &Segment) -> Option<&'a RawMaterial> {
    segment
        .extra_material_refs
        .iter()
        .find_map(|id| materials.beats.iter().find(|item| &item.id == id))
}

/// Timeline time of one stored beat.
///
/// `source` is a source time. With no source range it is an offset from zero
/// across the target duration. A beat outside that window is `None`. Speed
/// other than 1 divides the source offset, toward zero.
#[must_use]
pub fn beat_timeline_time(segment: &Segment, source: i64) -> Option<Micros> {
    let (source_start, source_end) = source_window(segment)?;
    if source < source_start || source >= source_end {
        return None;
    }
    let delta = source.checked_sub(source_start)?;
    let delta_target = timeline_offset(delta, segment.speed)?;
    segment.target.start.0.checked_add(delta_target).map(Micros)
}

/// The source time that lands on `at`.
///
/// `None` when `at` is outside the clip, or when this speed cannot draw a
/// beat on that exact timeline time.
#[must_use]
pub fn source_time_at(segment: &Segment, at: Micros) -> Option<i64> {
    let (source_start, source_end) = source_window(segment)?;
    let start = segment.target.start.0;
    let end = start.checked_add(segment.target.duration.0)?;
    if at.0 < start || at.0 >= end {
        return None;
    }
    let offset = at.0.checked_sub(start)?;
    let delta = source_offset(offset, segment.speed)?;
    let source = source_start.checked_add(delta)?;
    if source < source_start || source >= source_end {
        return None;
    }
    let shown = beat_timeline_time(segment, source)?;
    if shown.0 == at.0 { Some(source) } else { None }
}

/// Beats that still sit on their clip.
#[must_use]
pub fn visible_beats(draft: &crate::draft::Draft) -> Vec<VisibleBeat> {
    let mut beats = Vec::new();
    for track in &draft.tracks {
        for segment in &track.segments {
            let Some(raw) = segment_beats(&draft.materials, segment) else {
                continue;
            };
            for source in user_beats(raw) {
                let Some(time) = beat_timeline_time(segment, source) else {
                    continue;
                };
                beats.push(VisibleBeat {
                    segment_id: segment.id.clone(),
                    source,
                    time,
                });
            }
        }
    }
    beats
}

fn source_window(segment: &Segment) -> Option<(i64, i64)> {
    if let Some(source) = &segment.source_range {
        if source.duration.0 <= 0 {
            return None;
        }
        let end = source.start.0.checked_add(source.duration.0)?;
        return Some((source.start.0, end));
    }
    if segment.target.duration.0 <= 0 {
        return None;
    }
    Some((0, segment.target.duration.0))
}

fn timeline_offset(delta: i64, speed: f64) -> Option<i64> {
    if delta < 0 || !speed.is_finite() || speed <= 0.0 {
        return None;
    }
    if speed == 1.0 {
        return Some(delta);
    }
    let mapped = (delta as f64) / speed;
    if !mapped.is_finite() || mapped < 0.0 || mapped >= (1_u64 << 63) as f64 {
        return None;
    }
    Some(mapped.trunc() as i64)
}

fn source_offset(offset: i64, speed: f64) -> Option<i64> {
    if offset < 0 || !speed.is_finite() || speed <= 0.0 {
        return None;
    }
    if speed == 1.0 {
        return Some(offset);
    }
    if offset.unsigned_abs() > (1_u64 << 53) {
        return None;
    }
    let low = (offset as f64) * speed;
    if !low.is_finite() || low < 0.0 || low >= (1_u64 << 63) as f64 {
        return None;
    }
    let mut delta = low.ceil() as i64;
    if (delta as f64) < low {
        delta = delta.saturating_add(1);
    }
    for _ in 0..4 {
        if timeline_offset(delta, speed) == Some(offset) {
            return Some(delta);
        }
        delta = delta.checked_add(1)?;
    }
    None
}

fn push_beats(value: &Value, beats: &mut Vec<i64>) {
    if let Some(beat) = value.as_i64() {
        beats.push(beat);
        return;
    }
    let Some(items) = value.as_array() else {
        return;
    };
    for item in items {
        if let Some(beat) = item.as_i64() {
            beats.push(beat);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draft::Draft;
    use crate::time::TimeRange;

    #[test]
    fn a_beat_is_a_source_time_and_double_speed_uses_half() {
        let raw = new_beats(MaterialId::new("beats"), 200_000);
        assert_eq!(raw.kind, "beats");
        assert_eq!(user_beats(&raw), vec![200_000]);
        assert_eq!(
            raw.fields.get("enable_ai_beats").and_then(Value::as_bool),
            Some(false)
        );

        let mut clip = Segment::new("clip", "media", TimeRange::new(1_000_000, 500_000));
        clip.speed = 2.0;
        clip.source_range = Some(TimeRange::new(0, 1_000_000));
        assert_eq!(beat_timeline_time(&clip, 200_000), Some(Micros(1_100_000)));
        assert_eq!(source_time_at(&clip, Micros(1_100_000)), Some(200_000));
        assert!(source_time_at(&clip, Micros(1_500_000)).is_none());

        let single = RawMaterial {
            id: MaterialId::new("one"),
            kind: "beats".to_owned(),
            fields: serde_json::from_str(r#"{"user_beats":200000}"#).expect("json"),
        };
        assert_eq!(user_beats(&single), vec![200_000]);

        clip.extra_material_refs.push(raw.id.clone());
        let mut draft = Draft::empty("D");
        draft.tracks[0].segments.push(clip);
        draft.materials.beats.push(raw);
        let beats = visible_beats(&draft);
        assert_eq!(beats.len(), 1);
        assert_eq!(beats[0].source, 200_000);
        assert_eq!(beats[0].time, Micros(1_100_000));
    }

    #[test]
    fn a_beat_past_the_source_is_kept_off_the_timeline() {
        let raw = new_beats(MaterialId::new("beats"), 2_000_000);
        let mut clip = Segment::new("clip", "media", TimeRange::new(0, 1_000_000));
        clip.extra_material_refs.push(raw.id.clone());
        let mut draft = Draft::empty("D");
        draft.tracks[0].segments.push(clip);
        draft.materials.beats.push(raw);
        assert!(visible_beats(&draft).is_empty());
        assert_eq!(user_beats(&draft.materials.beats[0]), vec![2_000_000]);
    }
}
