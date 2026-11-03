use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::clip::{Clip, Point, UniformScale};
use crate::enums::KeyframeCurve;
use crate::id::MaterialId;
use crate::segment::Segment;
use crate::time::Micros;

/// One property's keys. `property_type` is stored as written.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct CommonKeyframeGroup {
    #[serde(default)]
    pub property_type: String,
    #[serde(default)]
    pub material_id: MaterialId,
    #[serde(default)]
    pub keyframe_list: Vec<CommonKeyframe>,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct CommonKeyframe {
    #[serde(default)]
    pub id: String,
    /// `Line`, `CurveIn`, `CurveOut`, `CurveInOut`, or one of the free curves.
    #[serde(default, rename = "curveType")]
    pub curve_type: String,
    #[serde(default)]
    pub time_offset: Micros,
    #[serde(default)]
    pub values: Vec<f64>,
    #[serde(default)]
    pub string_value: String,
    #[serde(default, rename = "graphID")]
    pub graph_id: String,
    #[serde(default)]
    pub left_control: Point,
    #[serde(default)]
    pub right_control: Point,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// The clip at `time`, with keyframes measured from the segment start.
///
/// A key before the first sample holds that sample. A key after the last
/// holds the last. The same offset keeps the later key. A line, an empty
/// curve, `none`, and an unknown curve are a straight line, and their handles
/// are ignored. Any other curve is a cubic. The earlier key's right handle
/// and the later key's left handle are offsets from those samples. The sample
/// solves that cubic in time, and stops when the time error is under a
/// thousandth of the span. A handle that is not finite refuses the clip. A
/// graph id is not sampled. `KFTypeScale` drives both axes through uniform
/// scale when that switch is on, and through `clip.scale` when it is off.
/// With uniform scale on and no `KFTypeScale`, `KFTypeScaleX` drives it, as
/// the reference editor keys both axes even then. With it off,
/// `KFTypeScaleX` and `KFTypeScaleY` replace one axis each.
pub fn clip_at(segment: &Segment, time: Micros) -> Option<(Clip, UniformScale)> {
    let offset = time.0.saturating_sub(segment.target.start.0);
    let mut clip = segment.clip.clone();
    let mut uniform = segment.uniform_scale.clone();
    let groups = &segment.common_keyframes;
    if let Some(value) = sample(groups, &["KFTypeAlpha", "alpha"], offset)? {
        clip.alpha = value;
    }
    if let Some(value) = sample(groups, &["KFTypeRotation", "rotation"], offset)? {
        clip.rotation = value;
    }
    if let Some(value) = sample(groups, &["KFTypePositionX"], offset)? {
        clip.transform.x = value;
    }
    if let Some(value) = sample(groups, &["KFTypePositionY"], offset)? {
        clip.transform.y = value;
    }
    let both = sample(groups, &["KFTypeScale"], offset)?;
    if uniform.on {
        let value = match both {
            Some(value) => Some(value),
            None => sample(groups, &["KFTypeScaleX"], offset)?,
        };
        if let Some(value) = value {
            uniform.value = value;
        }
    } else {
        if let Some(value) = both {
            clip.scale.x = value;
            clip.scale.y = value;
        }
        if let Some(value) = sample(groups, &["KFTypeScaleX"], offset)? {
            clip.scale.x = value;
        }
        if let Some(value) = sample(groups, &["KFTypeScaleY"], offset)? {
            clip.scale.y = value;
        }
    }
    Some((clip, uniform))
}

#[derive(Clone)]
struct KeySample {
    time: i64,
    value: f64,
    order: usize,
    bends: bool,
    left_x: f64,
    left_y: f64,
    right_x: f64,
    right_y: f64,
}

pub fn sample(groups: &[CommonKeyframeGroup], names: &[&str], offset: i64) -> Option<Option<f64>> {
    let mut keys = Vec::new();
    for group in groups {
        if !names.contains(&group.property_type.as_str()) {
            continue;
        }
        for key in &group.keyframe_list {
            let Some(value) = key.values.first().copied() else {
                continue;
            };
            if !value.is_finite() {
                return None;
            }
            let bends = bends(&key.curve_type);
            if bends
                && !(key.left_control.x.is_finite()
                    && key.left_control.y.is_finite()
                    && key.right_control.x.is_finite()
                    && key.right_control.y.is_finite())
            {
                return None;
            }
            let (left_x, left_y, right_x, right_y) = if bends {
                (
                    key.left_control.x,
                    key.left_control.y,
                    key.right_control.x,
                    key.right_control.y,
                )
            } else {
                (0.0, 0.0, 0.0, 0.0)
            };
            keys.push(KeySample {
                time: key.time_offset.0,
                value,
                order: keys.len(),
                bends,
                left_x,
                left_y,
                right_x,
                right_y,
            });
        }
    }
    Some(sample_keys(&mut keys, offset))
}

fn bends(curve_type: &str) -> bool {
    match KeyframeCurve::parse(curve_type) {
        Some(
            KeyframeCurve::CurveOut
            | KeyframeCurve::CurveIn
            | KeyframeCurve::CurveInOut
            | KeyframeCurve::FreeCurveIn
            | KeyframeCurve::FreeCurveOut
            | KeyframeCurve::FreeCurveInOut,
        ) => true,
        Some(KeyframeCurve::Line | KeyframeCurve::None) | None => false,
    }
}

fn sample_keys(keys: &mut [KeySample], at: i64) -> Option<f64> {
    if keys.is_empty() {
        return None;
    }
    keys.sort_by(|left, right| {
        left.time
            .cmp(&right.time)
            .then(left.order.cmp(&right.order))
    });
    let mut collapsed: Vec<KeySample> = Vec::new();
    for key in keys.iter().cloned() {
        if let Some(last) = collapsed.last_mut()
            && last.time == key.time
        {
            *last = key;
            continue;
        }
        collapsed.push(key);
    }
    let first = collapsed.first()?;
    if at <= first.time {
        return Some(first.value);
    }
    let last = collapsed.last()?;
    if at >= last.time {
        return Some(last.value);
    }
    for pair in collapsed.windows(2) {
        if at >= pair[0].time && at <= pair[1].time {
            return Some(blend(&pair[0], &pair[1], at));
        }
    }
    Some(last.value)
}

fn blend(left: &KeySample, right: &KeySample, at: i64) -> f64 {
    let span = (right.time - left.time) as f64;
    if span == 0.0 {
        return right.value;
    }
    let along = (at - left.time) as f64 / span;
    let linear = left.value + (right.value - left.value) * along;
    if !left.bends && !right.bends {
        return linear;
    }
    let t = parameter(left.right_x / span, 1.0 + right.left_x / span, along);
    let curved = bezier(
        left.value,
        left.value + left.right_y,
        right.value + right.left_y,
        right.value,
        t,
    );
    if curved.is_finite() { curved } else { linear }
}

/// Cubic parameter whose horizontal bezier equals `along`.
///
/// Eight Newton steps from `along`, then bisection. A step whose time error
/// is under a thousandth of the span is kept. A query on either sample
/// returns that end.
fn parameter(cx1: f64, cx2: f64, along: f64) -> f64 {
    if along < 1e-6 {
        return 0.0;
    }
    if along > 1.0 || (1.0 - along) < 1e-6 {
        return 1.0;
    }
    let mut t = along;
    for _ in 0..8 {
        let x = horizontal(cx1, cx2, t);
        if (x - along).abs() < 1e-3 {
            return t;
        }
        let slope = horizontal_slope(cx1, cx2, t);
        if slope.abs() < 1e-6 {
            break;
        }
        t = (t - (x - along) / slope).clamp(0.0, 1.0);
    }
    let mut low = 0.0;
    let mut high = 1.0;
    for _ in 0..64 {
        if low >= high || (high - low) <= 1e-6 {
            return t.min(1.0);
        }
        let mid = low + (high - low) * 0.5;
        let x = horizontal(cx1, cx2, mid);
        if x < along {
            low = mid;
        } else {
            high = mid;
        }
        if (x - along).abs() < 1e-3 {
            return mid;
        }
    }
    t.min(1.0)
}

fn horizontal(cx1: f64, cx2: f64, t: f64) -> f64 {
    let (cubic, square, linear) = horizontal_terms(cx1, cx2);
    ((cubic * t + square) * t + linear) * t
}

fn horizontal_slope(cx1: f64, cx2: f64, t: f64) -> f64 {
    let (cubic, square, linear) = horizontal_terms(cx1, cx2);
    (3.0 * cubic * t + 2.0 * square) * t + linear
}

fn horizontal_terms(cx1: f64, cx2: f64) -> (f64, f64, f64) {
    (
        3.0 * cx1 - 3.0 * cx2 + 1.0,
        3.0 * cx2 - 6.0 * cx1,
        3.0 * cx1,
    )
}

fn bezier(y0: f64, y1: f64, y2: f64, y3: f64, t: f64) -> f64 {
    let rest = 1.0 - t;
    rest * rest * rest * y0 + 3.0 * rest * rest * t * y1 + 3.0 * rest * t * t * y2 + t * t * t * y3
}

/// The properties a pose key holds, as the reference editor writes them
/// when a preview gesture keys a clip.
pub const POSE_KEYS: [&str; 5] = [
    "KFTypePositionX",
    "KFTypePositionY",
    "KFTypeRotation",
    "KFTypeScaleX",
    "KFTypeScaleY",
];
/// The opacity property.
pub const ALPHA_KEY: &str = "KFTypeAlpha";

/// The offsets from the segment start that hold a key, earliest first, each
/// once.
#[must_use]
pub fn key_offsets(segment: &Segment) -> Vec<Micros> {
    let mut offsets: Vec<Micros> = segment
        .common_keyframes
        .iter()
        .flat_map(|group| &group.keyframe_list)
        .map(|key| key.time_offset)
        .collect();
    offsets.sort_unstable_by_key(|offset| offset.0);
    offsets.dedup();
    offsets
}

/// Set `property` to `value` at `offset`: a key there takes it, else a
/// straight key is added in time order. A missing group is added.
pub fn put_key(segment: &mut Segment, property: &str, offset: Micros, value: f64) {
    let id = format!("{}-{property}-{}", segment.id, offset.0);
    let at = match segment
        .common_keyframes
        .iter()
        .position(|group| group.property_type == property)
    {
        Some(at) => at,
        None => {
            segment.common_keyframes.push(CommonKeyframeGroup {
                property_type: property.to_owned(),
                rest: id_field(&format!("{}-{property}", segment.id)),
                ..CommonKeyframeGroup::default()
            });
            segment.common_keyframes.len() - 1
        }
    };
    let keys = &mut segment.common_keyframes[at].keyframe_list;
    if let Some(key) = keys.iter_mut().find(|key| key.time_offset == offset) {
        key.values = vec![value];
        return;
    }
    let place = keys
        .iter()
        .position(|key| key.time_offset.0 > offset.0)
        .unwrap_or(keys.len());
    keys.insert(
        place,
        CommonKeyframe {
            id,
            curve_type: "Line".to_owned(),
            time_offset: offset,
            values: vec![value],
            ..CommonKeyframe::default()
        },
    );
}

/// Key the clip's place, size, turn, and opacity as they are at `offset`
/// from the segment start. A clip whose keys do not read is refused.
#[must_use]
pub fn add_keys(segment: &mut Segment, offset: Micros) -> Option<()> {
    let at = Micros(segment.target.start.0.saturating_add(offset.0));
    let (clip, uniform) = clip_at(segment, at)?;
    let (scale_x, scale_y) = if uniform.on {
        (uniform.value, uniform.value)
    } else {
        (clip.scale.x, clip.scale.y)
    };
    let values = [
        clip.transform.x,
        clip.transform.y,
        clip.rotation,
        scale_x,
        scale_y,
    ];
    for (property, value) in POSE_KEYS.iter().zip(values) {
        put_key(segment, property, offset, value);
    }
    put_key(segment, ALPHA_KEY, offset, clip.alpha);
    Some(())
}

/// Remove every key at `offset`. A group left with no keys is dropped.
/// Whether any key was there.
pub fn remove_keys(segment: &mut Segment, offset: Micros) -> bool {
    let mut removed = false;
    for group in &mut segment.common_keyframes {
        let before = group.keyframe_list.len();
        group.keyframe_list.retain(|key| key.time_offset != offset);
        removed |= group.keyframe_list.len() != before;
    }
    segment
        .common_keyframes
        .retain(|group| !group.keyframe_list.is_empty());
    removed
}

fn id_field(id: &str) -> Map<String, Value> {
    let mut rest = Map::new();
    rest.insert("id".to_owned(), Value::String(id.to_owned()));
    rest
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_alpha_key_keeps_the_stored_names() {
        let group: CommonKeyframeGroup = serde_json::from_str(
            r#"{"property_type":"alpha","material_id":"m","keyframe_list":[{"time_offset":0,"values":[1.0],"curveType":"Line","graphID":"","left_control":{"x":0,"y":0},"right_control":{"x":0,"y":0}}]}"#,
        )
        .unwrap();
        assert_eq!(group.property_type, "alpha");
        assert_eq!(group.keyframe_list[0].curve_type, "Line");
        assert_eq!(
            KeyframeCurve::parse(&group.keyframe_list[0].curve_type),
            Some(KeyframeCurve::Line)
        );
    }

    fn segment() -> Segment {
        let mut segment = Segment::new("a", "m", crate::TimeRange::new(1_000, 4_000));
        segment.uniform_scale.on = true;
        segment
    }

    #[test]
    fn a_key_holds_the_pose_sampled_between_its_neighbours() {
        let mut segment = segment();
        put_key(&mut segment, "KFTypePositionX", Micros(0), 0.0);
        put_key(&mut segment, "KFTypePositionX", Micros(2_000), 1.0);
        put_key(&mut segment, "KFTypeScaleX", Micros(2_000), 2.0);
        segment.clip.alpha = 0.5;
        add_keys(&mut segment, Micros(1_000)).unwrap();
        assert_eq!(
            key_offsets(&segment),
            [Micros(0), Micros(1_000), Micros(2_000)]
        );
        let value = |property: &str| {
            let group = segment
                .common_keyframes
                .iter()
                .find(|group| group.property_type == property)
                .unwrap();
            let key = group
                .keyframe_list
                .iter()
                .find(|key| key.time_offset == Micros(1_000))
                .unwrap();
            assert_eq!(key.curve_type, "Line");
            key.values[0]
        };
        assert_eq!(value("KFTypePositionX"), 0.5);
        assert_eq!(value("KFTypeScaleX"), 2.0);
        assert_eq!(value("KFTypeScaleY"), 2.0);
        assert_eq!(value(ALPHA_KEY), 0.5);
        let offsets: Vec<i64> = segment.common_keyframes[0]
            .keyframe_list
            .iter()
            .map(|key| key.time_offset.0)
            .collect();
        assert_eq!(offsets, [0, 1_000, 2_000]);
    }

    #[test]
    fn a_key_is_replaced_in_place_and_removing_it_drops_empty_groups() {
        let mut segment = segment();
        put_key(&mut segment, "KFTypeRotation", Micros(500), 10.0);
        put_key(&mut segment, "KFTypeRotation", Micros(500), 20.0);
        put_key(&mut segment, ALPHA_KEY, Micros(900), 1.0);
        assert_eq!(segment.common_keyframes[0].keyframe_list.len(), 1);
        assert_eq!(segment.common_keyframes[0].keyframe_list[0].values, [20.0]);
        assert!(remove_keys(&mut segment, Micros(500)));
        assert!(!remove_keys(&mut segment, Micros(500)));
        assert_eq!(segment.common_keyframes.len(), 1);
        assert_eq!(key_offsets(&segment), [Micros(900)]);
    }
}
