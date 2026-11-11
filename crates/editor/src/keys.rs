//! A clip's keyframes as the timeline shows them, the keyframe button and
//! the steps between keys, and the Graphs panel's curves.

use cutlass_model::{
    Draft, GRAPH_PRESETS, GraphPreset, KeyframeGraph, is_pose_key, key_offsets, key_offsets_where,
};
use cutlass_session::{Micros, Request, SegmentId};

use crate::stage::{TextSizes, stage_box};

/// A clip's keyframes, as timeline times.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClipKeys {
    /// Each keyframe's timeline time, earliest first.
    pub times: Vec<Micros>,
    /// The keyframe within a frame of the time asked, when one is.
    pub here: Option<Micros>,
    /// The later keyframe of the span the time asked is in: the one after
    /// the key under it, or after the last key before it. None past the
    /// last key or before the first.
    pub span: Option<Micros>,
    /// The Graphs panel curve easing that span, when it has one.
    pub graph: Option<String>,
    start: Micros,
    duration: Micros,
    /// A frame, the nearness that counts as on a key.
    reach: i64,
}

/// The keyframes of the picture or title clip `segment_id`, and the one
/// within a frame of `time`. Other clips have none to show.
#[must_use]
pub fn clip_keys(
    draft: &Draft,
    segment_id: &SegmentId,
    time: Micros,
    sizes: &dyn TextSizes,
) -> Option<ClipKeys> {
    stage_box(draft, segment_id, time, sizes)?;
    ClipKeys::of(draft, segment_id, time)
}

/// The keys clip `segment_id` has on the settings `names`, such as the
/// mask's or the volume's, and the one within a frame of `time`.
#[must_use]
pub fn value_keys(
    draft: &Draft,
    segment_id: &SegmentId,
    time: Micros,
    names: &[&str],
) -> Option<ClipKeys> {
    ClipKeys::of_where(draft, segment_id, time, &|property| {
        names.contains(&property)
    })
}

impl ClipKeys {
    fn of(draft: &Draft, segment_id: &SegmentId, time: Micros) -> Option<Self> {
        Self::of_where(draft, segment_id, time, &is_pose_key)
    }

    fn of_where(
        draft: &Draft,
        segment_id: &SegmentId,
        time: Micros,
        keep: &dyn Fn(&str) -> bool,
    ) -> Option<Self> {
        let segment = draft
            .tracks
            .iter()
            .flat_map(|track| &track.segments)
            .find(|segment| segment.id == *segment_id)?;
        let start = segment.target.start;
        let times: Vec<Micros> = key_offsets_where(segment, keep)
            .into_iter()
            .map(|offset| Micros(start.0.saturating_add(offset.0)))
            .collect();
        let reach = frame(draft.fps);
        let here = times
            .iter()
            .copied()
            .filter(|key| (key.0 - time.0).abs() <= reach)
            .min_by_key(|key| (key.0 - time.0).abs());
        let after = time.0.saturating_add(reach);
        let span = times
            .iter()
            .copied()
            .find(|key| key.0 > after)
            .filter(|_| times.first().is_some_and(|first| first.0 <= after));
        let graph = span.and_then(|end| {
            let offset = Micros(end.0 - start.0);
            segment
                .common_keyframes
                .iter()
                .filter(|group| keep(&group.property_type))
                .flat_map(|group| &group.keyframe_list)
                .filter(|key| key.time_offset == offset && !key.graph_id.is_empty())
                .find_map(|key| {
                    draft
                        .keyframe_graph_list
                        .iter()
                        .find(|graph| graph.id == key.graph_id)
                })
                .map(|graph| graph.resource_name.clone())
                .filter(|name| !name.is_empty())
        });
        // Keys a split or trim left outside the clip still shape it, but
        // only those on it are marked.
        let end = start.0.saturating_add(segment.target.duration.0);
        let times = times
            .into_iter()
            .filter(|key| (start.0..=end).contains(&key.0))
            .collect();
        Some(Self {
            times,
            here,
            span,
            graph,
            start,
            duration: segment.target.duration,
            reach,
        })
    }

    /// The request the keyframe button makes at `time`: remove the key here,
    /// else add one when `time` is on the clip.
    #[must_use]
    pub fn toggle(&self, segment_id: &SegmentId, time: Micros) -> Option<Request> {
        if let Some(here) = self.here {
            return Some(Request::RemoveKeyframe {
                segment_id: segment_id.clone(),
                offset: Micros(here.0 - self.start.0),
            });
        }
        let offset = time.0.checked_sub(self.start.0)?;
        (0..=self.duration.0)
            .contains(&offset)
            .then(|| Request::AddKeyframe {
                segment_id: segment_id.clone(),
                offset: Micros(offset),
                properties: Vec::new(),
            })
    }
}

impl ClipKeys {
    /// The offset from the clip's start of the key under the playhead, or
    /// where a key at `time` would go when it is on the clip.
    #[must_use]
    pub fn offset_at(&self, time: Micros) -> Option<Micros> {
        if let Some(here) = self.here {
            return Some(Micros(here.0 - self.start.0));
        }
        let offset = time.0.checked_sub(self.start.0)?;
        (0..=self.duration.0)
            .contains(&offset)
            .then_some(Micros(offset))
    }

    /// The last key more than a frame before `time`, for the ◀ button.
    #[must_use]
    pub fn before(&self, time: Micros) -> Option<Micros> {
        self.times
            .iter()
            .rev()
            .copied()
            .find(|key| key.0 < time.0.saturating_sub(self.reach))
    }

    /// The first key more than a frame after `time`, for the ▶ button.
    #[must_use]
    pub fn after(&self, time: Micros) -> Option<Micros> {
        self.times
            .iter()
            .copied()
            .find(|key| key.0 > time.0.saturating_add(self.reach))
    }

    /// The request that eases the span the playhead is in along `graph`,
    /// or makes it straight with `None`.
    #[must_use]
    pub fn ease(&self, segment_id: &SegmentId, graph: Option<String>) -> Option<Request> {
        self.ease_only(segment_id, graph, &[])
    }

    /// [`ClipKeys::ease`] on the keys named `properties` only; none eases
    /// every pose key.
    #[must_use]
    pub fn ease_only(
        &self,
        segment_id: &SegmentId,
        graph: Option<String>,
        properties: &[&str],
    ) -> Option<Request> {
        let end = self.span?;
        Some(Request::SetKeyframeGraph {
            segment_id: segment_id.clone(),
            offset: Micros(end.0 - self.start.0),
            graph,
            properties: properties.iter().map(|name| (*name).to_owned()).collect(),
        })
    }
}

/// The request that moves every key at `from` from the clip's start to
/// `to`, a drag of the key's mark. `None` when there is no key at `from`,
/// `to` is off the clip, or another key is already at `to`.
#[must_use]
pub fn move_key(
    draft: &Draft,
    segment_id: &SegmentId,
    from: Micros,
    to: Micros,
) -> Option<Request> {
    let segment = draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .find(|segment| segment.id == *segment_id)?;
    let offsets = key_offsets(segment);
    let on = (0..=segment.target.duration.0).contains(&to.0);
    if from == to || !on || !offsets.contains(&from) || offsets.contains(&to) {
        return None;
    }
    let mut keyframes = segment.common_keyframes.clone();
    for group in &mut keyframes {
        for key in &mut group.keyframe_list {
            if key.time_offset == from {
                key.time_offset = to;
            }
        }
        group.keyframe_list.sort_by_key(|key| key.time_offset.0);
    }
    Some(Request::SetKeyframes {
        segment_id: segment_id.clone(),
        keyframes,
    })
}

/// The request that removes every key at `at` from the clip's start,
/// whatever it keys: a key's mark deleted. `None` when the clip is not on
/// the timeline or has no key there.
#[must_use]
pub fn remove_key(draft: &Draft, segment_id: &SegmentId, at: Micros) -> Option<Request> {
    let mut segment = draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .find(|segment| segment.id == *segment_id)?
        .clone();
    cutlass_model::remove_keys_where(&mut segment, at, |_| true).then(|| Request::SetKeyframes {
        segment_id: segment_id.clone(),
        keyframes: segment.common_keyframes,
    })
}

/// The request that moves the keys at `offsets` from the clip's start by
/// `delta` together, several marks dragged as one. `None` when none of
/// the offsets has a key, `delta` is nothing, a key would leave the clip,
/// or one would land on a key that is not moving.
#[must_use]
pub fn move_keys(
    draft: &Draft,
    segment_id: &SegmentId,
    offsets: &[Micros],
    delta: i64,
) -> Option<Request> {
    let segment = draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .find(|segment| segment.id == *segment_id)?;
    let all = key_offsets(segment);
    let moving: Vec<Micros> = all
        .iter()
        .copied()
        .filter(|offset| offsets.contains(offset))
        .collect();
    if moving.is_empty() || delta == 0 {
        return None;
    }
    for offset in &moving {
        let to = offset.0.checked_add(delta)?;
        let on = (0..=segment.target.duration.0).contains(&to);
        let taken = all.contains(&Micros(to)) && !moving.contains(&Micros(to));
        if !on || taken {
            return None;
        }
    }
    let mut keyframes = segment.common_keyframes.clone();
    for group in &mut keyframes {
        for key in &mut group.keyframe_list {
            if moving.contains(&key.time_offset) {
                key.time_offset = Micros(key.time_offset.0 + delta);
            }
        }
        group.keyframe_list.sort_by_key(|key| key.time_offset.0);
    }
    Some(Request::SetKeyframes {
        segment_id: segment_id.clone(),
        keyframes,
    })
}

/// The request that removes every key at each of `offsets` from the
/// clip's start: several marks deleted as one. `None` when the clip is
/// not on the timeline or has a key at none of them.
#[must_use]
pub fn remove_keys(draft: &Draft, segment_id: &SegmentId, offsets: &[Micros]) -> Option<Request> {
    let mut segment = draft
        .tracks
        .iter()
        .flat_map(|track| &track.segments)
        .find(|segment| segment.id == *segment_id)?
        .clone();
    let mut removed = false;
    for offset in offsets {
        removed |= cutlass_model::remove_keys_where(&mut segment, *offset, |_| true);
    }
    removed.then(|| Request::SetKeyframes {
        segment_id: segment_id.clone(),
        keyframes: segment.common_keyframes,
    })
}

/// A Graphs panel curve to show: its name and its line through the unit
/// square, x the share of time and y the share of the change.
#[derive(Clone, Debug, PartialEq)]
pub struct KeyGraph {
    pub name: String,
    pub points: Vec<(f64, f64)>,
}

/// Steps drawn along each Bézier piece of a curve.
const STEPS: usize = 12;

/// The Graphs panel curves, in the panel's order after None.
#[must_use]
pub fn key_graphs() -> Vec<KeyGraph> {
    GRAPH_PRESETS.iter().map(drawn).collect()
}

fn drawn(preset: &GraphPreset) -> KeyGraph {
    let graph = KeyframeGraph::of("", preset);
    let at = |index: usize| {
        let point = &graph.graph_points[index].point;
        (point.x, point.y)
    };
    let mut points = vec![at(0)];
    for piece in 0..(graph.graph_points.len() - 1) / 3 {
        let [p0, p1, p2, p3] = [0, 1, 2, 3].map(|step| at(piece * 3 + step));
        for step in 1..=STEPS {
            #[expect(clippy::cast_precision_loss, reason = "a dozen steps")]
            let t = step as f64 / STEPS as f64;
            let along = |a: f64, b: f64, c: f64, d: f64| {
                let u = 1.0 - t;
                u * u * u * a + 3.0 * u * u * t * b + 3.0 * u * t * t * c + t * t * t * d
            };
            points.push((along(p0.0, p1.0, p2.0, p3.0), along(p0.1, p1.1, p2.1, p3.1)));
        }
    }
    KeyGraph {
        name: preset.name.to_owned(),
        points,
    }
}

/// One frame at `fps` in microseconds; 30 when the rate is not usable.
fn frame(fps: f64) -> i64 {
    let fps = if fps.is_finite() && fps >= 1.0 {
        fps
    } else {
        30.0
    };
    // A frame of at least one frame per second fits easily.
    #[expect(clippy::cast_possible_truncation, reason = "at most a million")]
    let frame = (1_000_000.0 / fps).round() as i64;
    frame
}

#[cfg(test)]
mod tests {
    use cutlass_model::{MaterialVideo, Segment, TimeRange, put_key};

    use super::*;
    use crate::stage::NoText;

    #[test]
    fn several_keys_move_together_and_are_removed_together() {
        let mut draft = Draft::empty("D");
        let mut segment = Segment::new("c", "v", TimeRange::new(0, 4_000_000));
        for offset in [0, 1_000_000, 2_000_000, 3_000_000] {
            put_key(&mut segment, "KFTypePositionX", Micros(offset), 0.0);
        }
        put_key(&mut segment, "KFTypeAlpha", Micros(2_000_000), 0.5);
        draft.tracks[0].segments.push(segment);
        let id = SegmentId::new("c");
        let times = |request: Option<Request>| -> Option<Vec<Vec<i64>>> {
            let Request::SetKeyframes { keyframes, .. } = request? else {
                return None;
            };
            Some(
                keyframes
                    .iter()
                    .map(|group| {
                        group
                            .keyframe_list
                            .iter()
                            .map(|key| key.time_offset.0)
                            .collect()
                    })
                    .collect(),
            )
        };
        let picked = [Micros(1_000_000), Micros(2_000_000)];
        // Half a second later, both keys and the opacity key with them.
        assert_eq!(
            times(move_keys(&draft, &id, &picked, 500_000)),
            Some(vec![
                vec![0, 1_500_000, 2_500_000, 3_000_000],
                vec![2_500_000]
            ])
        );
        // One second later the first lands where the second was, which is
        // moving too, but the second lands on a key that is not.
        assert_eq!(move_keys(&draft, &id, &picked, 1_000_000), None);
        // A key may not leave the clip, and nothing moves by nothing.
        assert_eq!(move_keys(&draft, &id, &picked, -1_500_000), None);
        assert_eq!(move_keys(&draft, &id, &picked, 2_500_000), None);
        assert_eq!(move_keys(&draft, &id, &picked, 0), None);
        assert_eq!(move_keys(&draft, &id, &picked, i64::MAX), None);
        assert_eq!(move_keys(&draft, &id, &[Micros(7)], 1), None);
        assert_eq!(move_keys(&draft, &SegmentId::new("gone"), &picked, 1), None);

        assert_eq!(
            times(remove_keys(&draft, &id, &picked)),
            Some(vec![vec![0, 3_000_000]])
        );
        // A time with no key among them is passed over.
        assert_eq!(
            times(remove_keys(&draft, &id, &[Micros(7), Micros(0)])),
            Some(vec![vec![1_000_000, 2_000_000, 3_000_000], vec![2_000_000]])
        );
        assert_eq!(remove_keys(&draft, &id, &[Micros(7)]), None);
        assert_eq!(remove_keys(&draft, &id, &[]), None);
        assert_eq!(remove_keys(&draft, &SegmentId::new("gone"), &picked), None);
    }

    #[test]
    fn a_key_is_removed_from_every_setting_keyed_at_its_time() {
        let mut draft = Draft::empty("D");
        let mut segment = Segment::new("c", "v", TimeRange::new(0, 3_000_000));
        for offset in [0, 1_000_000] {
            put_key(&mut segment, "KFTypePositionX", Micros(offset), 0.0);
        }
        put_key(&mut segment, "KFTypeAlpha", Micros(1_000_000), 0.5);
        draft.tracks[0].segments.push(segment);
        let id = SegmentId::new("c");
        let Some(Request::SetKeyframes { keyframes, .. }) =
            remove_key(&draft, &id, Micros(1_000_000))
        else {
            std::panic::resume_unwind(Box::new("no removal"));
        };
        let left: Vec<i64> = keyframes
            .iter()
            .flat_map(|group| &group.keyframe_list)
            .map(|key| key.time_offset.0)
            .collect();
        assert_eq!(left, [0]);
        // Nothing to remove: a time with no key, and a clip that is gone.
        assert!(remove_key(&draft, &id, Micros(500_000)).is_none());
        assert!(remove_key(&draft, &SegmentId::new("gone"), Micros(0)).is_none());
        // The draft itself is untouched until the request is applied.
        assert_eq!(draft.tracks[0].segments[0].common_keyframes.len(), 2);
    }

    #[test]
    fn a_key_moves_to_a_free_time_on_the_clip_and_keeps_its_order() {
        let mut draft = Draft::empty("D");
        let mut segment = Segment::new("c", "v", TimeRange::new(0, 3_000_000));
        for (offset, value) in [(0, 0.0), (1_000_000, 0.5), (2_000_000, 1.0)] {
            put_key(&mut segment, "KFTypePositionX", Micros(offset), value);
        }
        draft.tracks[0].segments.push(segment);
        let id = SegmentId::new("c");
        let Some(Request::SetKeyframes { keyframes, .. }) =
            move_key(&draft, &id, Micros(1_000_000), Micros(2_500_000))
        else {
            std::panic::resume_unwind(Box::new("no move"));
        };
        let moved: Vec<(i64, f64)> = keyframes[0]
            .keyframe_list
            .iter()
            .map(|key| (key.time_offset.0, key.values[0]))
            .collect();
        assert_eq!(moved, [(0, 0.0), (2_000_000, 1.0), (2_500_000, 0.5)]);
        // Onto another key, off the clip, or from no key: nothing.
        assert!(move_key(&draft, &id, Micros(1_000_000), Micros(2_000_000)).is_none());
        assert!(move_key(&draft, &id, Micros(1_000_000), Micros(4_000_000)).is_none());
        assert!(move_key(&draft, &id, Micros(500_000), Micros(600_000)).is_none());
    }

    #[test]
    fn the_steps_go_to_the_keys_more_than_a_frame_away() {
        let mut draft = Draft::empty("D");
        draft.fps = 30.0;
        let mut video = MaterialVideo::new("v");
        video.width = 16;
        video.height = 9;
        draft.materials.videos.push(video);
        let mut segment = Segment::new("c", "v", TimeRange::new(1_000_000, 3_000_000));
        segment.source_range = Some(TimeRange::new(0, 3_000_000));
        for offset in [0, 1_000_000, 2_000_000] {
            put_key(&mut segment, "KFTypePositionX", Micros(offset), 0.0);
        }
        draft.tracks[0].segments.push(segment);
        let id = SegmentId::new("c");
        let keys = clip_keys(&draft, &id, Micros(2_000_000), &NoText).unwrap();
        assert_eq!(keys.here, Some(Micros(2_000_000)));
        assert_eq!(keys.before(Micros(2_000_000)), Some(Micros(1_000_000)));
        assert_eq!(keys.after(Micros(2_000_000)), Some(Micros(3_000_000)));
        // A key within a frame is the one under the playhead, not a step.
        assert_eq!(keys.after(Micros(2_990_000)), None);
        assert_eq!(keys.before(Micros(1_010_000)), None);
        assert_eq!(
            keys.toggle(&id, Micros(2_000_000)),
            Some(Request::RemoveKeyframe {
                segment_id: id.clone(),
                offset: Micros(1_000_000),
            })
        );
    }
}
