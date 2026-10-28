use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::id::MaterialId;
use crate::time::Micros;

/// A transition material named by the outgoing segment.
///
/// `duration` is microseconds. Zero is a hard cut. A negative duration is
/// reported and played as a hard cut. `is_overlap` false places the window
/// on the outgoing segment: `[cut - duration, cut)`. `is_overlap` true gives
/// the outgoing side `duration / 2`, rounded toward zero, and the incoming
/// side the remainder: `[cut - left, cut + right)`. The two target ranges
/// still only touch. Load reports a window that runs past a segment and
/// leaves this material as stored.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct MaterialTransition {
    #[serde(default)]
    pub id: MaterialId,
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub effect_id: String,
    #[serde(default)]
    pub resource_id: String,
    #[serde(default)]
    pub third_resource_id: String,
    #[serde(default)]
    pub source_platform: i32,
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub duration: Micros,
    #[serde(default)]
    pub is_overlap: bool,
    #[serde(default)]
    pub platform: String,
    #[serde(default)]
    pub category_id: String,
    #[serde(default)]
    pub category_name: String,
    #[serde(default)]
    pub request_id: String,
    #[serde(default)]
    pub is_ai_transition: bool,
    #[serde(default)]
    pub video_path: String,
    #[serde(default)]
    pub task_id: String,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// Outgoing and incoming extents of a blend window.
///
/// `None` is a hard cut. A duration of 0 is a hard cut, and a negative
/// duration is played as one. `is_overlap` false keeps the whole duration
/// on the outgoing side. `is_overlap` true gives the outgoing side
/// `duration / 2`, toward zero, and the incoming side the remainder.
#[must_use]
pub fn blend_extents(duration: Micros, is_overlap: bool) -> Option<(Micros, Micros)> {
    if duration.0 <= 0 {
        return None;
    }
    if is_overlap {
        let left = duration.0 / 2;
        Some((Micros(left), Micros(duration.0 - left)))
    } else {
        Some((duration, Micros(0)))
    }
}

/// Half-open window `[cut - left, cut + right)`.
#[must_use]
pub fn blend_window(cut: Micros, duration: Micros, is_overlap: bool) -> Option<(Micros, Micros)> {
    let (left, right) = blend_extents(duration, is_overlap)?;
    let start = cut.0.checked_sub(left.0)?;
    let end = cut.0.checked_add(right.0)?;
    Some((Micros(start), Micros(end)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_window_is_half_open_around_the_cut() {
        assert_eq!(blend_window(Micros(1_000_000), Micros(0), false), None);
        assert_eq!(blend_window(Micros(1_000_000), Micros(-1), true), None);
        assert_eq!(
            blend_window(Micros(1_000_000), Micros(200_000), false),
            Some((Micros(800_000), Micros(1_000_000)))
        );
        assert_eq!(
            blend_extents(Micros(101), true),
            Some((Micros(50), Micros(51)))
        );
        assert_eq!(
            blend_window(Micros(1_000_000), Micros(101), true),
            Some((Micros(999_950), Micros(1_000_051)))
        );
    }
}
