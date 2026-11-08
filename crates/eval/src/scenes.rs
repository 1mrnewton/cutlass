//! Where a scene split cuts a clip: the file's scene changes mapped to the
//! timeline, then thinned so every piece keeps a frame and no cut lands in
//! a transition's blend window.

use cutlass_model::{Draft, MaterialRef, Micros, Segment, SegmentId, TimeRange, blend_window};

use crate::source_on_timeline;

/// The timeline times to split `segment_id` at, earliest first.
///
/// `source_cuts` are scene changes in the file's time, found over `range`.
/// A cut the speed map cannot place lands in proportion to the clip. A cut
/// too close to the clip's ends or to the previous kept cut, or inside a
/// blend window on the clip's row, is dropped.
#[must_use]
pub fn scene_split_times(
    draft: &Draft,
    segment_id: &SegmentId,
    range: &TimeRange,
    source_cuts: &[Micros],
) -> Vec<Micros> {
    let Some(segment) = draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .find(|segment| &segment.id == segment_id)
    else {
        return Vec::new();
    };
    let times: Vec<Micros> = source_cuts
        .iter()
        .filter_map(|cut| {
            source_on_timeline(draft, segment, *cut)
                .or_else(|| proportional(&segment.target, range, cut.0).map(Micros))
        })
        .collect();
    kept_cuts(draft, segment, &times)
}

/// `times` that leave at least a frame on every piece and miss every blend
/// window on the clip's row, earliest first and without repeats.
#[must_use]
pub fn kept_cuts(draft: &Draft, segment: &Segment, times: &[Micros]) -> Vec<Micros> {
    let start = segment.target.start.0;
    let end = start.saturating_add(segment.target.duration.0);
    let margin = draft
        .frame_rate()
        .and_then(|rate| rate.frame_start(1))
        .map_or(1, |time| time.0.max(1));
    let mut ordered: Vec<i64> = times.iter().map(|time| time.0).collect();
    ordered.sort_unstable();
    ordered.dedup();
    let mut kept = Vec::new();
    let mut last = start;
    for time in ordered {
        let inside = time >= start.saturating_add(margin) && time <= end.saturating_sub(margin);
        if !inside || time < last.saturating_add(margin) || blend_blocks(draft, &segment.id, time) {
            continue;
        }
        kept.push(Micros(time));
        last = time;
    }
    kept
}

/// `at` falls inside a transition's blend window on `segment_id`'s row.
#[must_use]
pub fn blend_blocks(draft: &Draft, segment_id: &SegmentId, at: i64) -> bool {
    let Some(track) = draft.tracks.iter().find(|track| {
        track
            .segments
            .iter()
            .any(|segment| &segment.id == segment_id)
    }) else {
        return false;
    };
    track.segments.iter().any(|segment| {
        let Some(cut) = segment
            .target
            .start
            .0
            .checked_add(segment.target.duration.0)
        else {
            return false;
        };
        segment.extra_material_refs.iter().any(|material_id| {
            let Some(MaterialRef::Transition(transition)) =
                draft.materials.get(material_id.as_str())
            else {
                return false;
            };
            blend_window(Micros(cut), transition.duration, transition.is_overlap)
                .is_some_and(|(from, to)| at >= from.0 && at < to.0)
        })
    })
}

fn proportional(target: &TimeRange, range: &TimeRange, source: i64) -> Option<i64> {
    let into = source.checked_sub(range.start.0)?;
    if range.duration.0 <= 0 || into < 0 || into >= range.duration.0 {
        return None;
    }
    let scaled =
        i128::from(into).checked_mul(i128::from(target.duration.0))? / i128::from(range.duration.0);
    target.start.0.checked_add(i64::try_from(scaled).ok()?)
}

#[cfg(test)]
mod tests {
    use cutlass_model::{MaterialId, MaterialTransition};

    use super::*;

    fn draft() -> Draft {
        let mut draft = Draft::empty("D");
        let mut a = Segment::new("a", "m", TimeRange::new(0, 4_000_000));
        a.source_range = Some(TimeRange::new(1_000_000, 4_000_000));
        draft.tracks[0].segments.push(a);
        draft
    }

    #[test]
    fn file_cuts_land_on_the_timeline_and_the_ends_are_dropped() {
        let draft = draft();
        let range = TimeRange::new(1_000_000, 4_000_000);
        let cuts = [
            Micros(1_000_000),
            Micros(2_000_000),
            Micros(2_000_000),
            Micros(4_500_000),
        ];
        assert_eq!(
            scene_split_times(&draft, &SegmentId::new("a"), &range, &cuts),
            vec![Micros(1_000_000), Micros(3_500_000)]
        );
    }

    #[test]
    fn a_cut_inside_a_blend_window_is_dropped() {
        let mut draft = draft();
        draft.tracks[0]
            .segments
            .push(Segment::new("b", "m", TimeRange::new(4_000_000, 1_000_000)));
        draft.materials.transitions.push(MaterialTransition {
            id: MaterialId::new("t"),
            duration: Micros(1_000_000),
            is_overlap: true,
            ..Default::default()
        });
        draft.tracks[0].segments[0]
            .extra_material_refs
            .push(MaterialId::new("t"));
        let segment = draft.tracks[0].segments[0].clone();
        assert!(blend_blocks(&draft, &segment.id, 3_800_000));
        assert_eq!(
            kept_cuts(&draft, &segment, &[Micros(1_000_000), Micros(3_800_000)]),
            vec![Micros(1_000_000)]
        );
    }
}
