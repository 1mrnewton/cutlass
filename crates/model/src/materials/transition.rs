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
