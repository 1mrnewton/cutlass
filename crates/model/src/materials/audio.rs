use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::id::MaterialId;
use crate::time::Micros;

#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct MaterialAudio {
    #[serde(default)]
    pub id: MaterialId,
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub duration: Micros,
    /// Set when this audio is the picture's detached original sound.
    #[serde(default)]
    pub video_id: MaterialId,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct MaterialAudioFade {
    #[serde(default)]
    pub id: MaterialId,
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub fade_type: i32,
    #[serde(default)]
    pub fade_in_duration: Micros,
    #[serde(default)]
    pub fade_out_duration: Micros,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}
