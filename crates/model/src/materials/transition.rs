use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::id::MaterialId;
use crate::time::Micros;

/// A transition material. `is_overlap` is the stored flag.
/// How a cut keeps a duration while segments on a track do not overlap is still open.
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
