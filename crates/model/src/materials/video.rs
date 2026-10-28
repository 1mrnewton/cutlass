use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::id::MaterialId;
use crate::time::Micros;

fn one() -> f64 {
    1.0
}

fn free_crop() -> String {
    "free".to_owned()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterialVideo {
    #[serde(default)]
    pub id: MaterialId,
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub width: i32,
    #[serde(default)]
    pub height: i32,
    #[serde(default)]
    pub duration: Micros,
    #[serde(default)]
    pub has_audio: bool,
    #[serde(default = "free_crop")]
    pub crop_ratio: String,
    #[serde(default = "one")]
    pub crop_scale: f64,
    #[serde(default)]
    pub reverse_path: String,
    #[serde(default)]
    pub reverse_intensifies_path: String,
    /// Held source frame. Absent and null stay unset.
    #[serde(default)]
    pub freeze: Option<FreezeInfo>,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// The source frame a freeze holds across the segment's target range.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct FreezeInfo {
    #[serde(default)]
    pub source_material_id: String,
    #[serde(default)]
    pub source_material_path: String,
    #[serde(default)]
    pub reverse_material_path: String,
    #[serde(default)]
    pub timestamp: Micros,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

impl MaterialVideo {
    /// Resource-mapping fallback for `duration`. An absent key stays zero.
    pub const MAPPING_DURATION: Micros = Micros(3_000_000);

    pub fn new(id: impl Into<MaterialId>) -> Self {
        Self {
            id: id.into(),
            ..Self::default()
        }
    }
}

impl Default for MaterialVideo {
    fn default() -> Self {
        Self {
            id: MaterialId::default(),
            kind: String::new(),
            path: String::new(),
            width: 0,
            height: 0,
            duration: Micros(0),
            has_audio: false,
            crop_ratio: free_crop(),
            crop_scale: 1.0,
            reverse_path: String::new(),
            reverse_intensifies_path: String::new(),
            freeze: None,
            rest: Map::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_matches_a_missing_object_and_not_the_mapping_duration() {
        let video = MaterialVideo::default();
        let parsed: MaterialVideo = serde_json::from_str("{}").unwrap();
        assert_eq!(video, parsed);
        assert_eq!(video.crop_ratio, "free");
        assert_eq!(video.crop_scale, 1.0);
        assert_eq!(video.duration, Micros(0));
        assert_ne!(video.duration, MaterialVideo::MAPPING_DURATION);
        assert_eq!(MaterialVideo::new("clip").id.as_str(), "clip");
        assert_eq!(MaterialVideo::new("clip").duration, Micros(0));
        assert!(video.freeze.is_none());
        let cleared: MaterialVideo = serde_json::from_str(r#"{"freeze":null}"#).unwrap();
        assert!(cleared.freeze.is_none());
        let held: MaterialVideo = serde_json::from_str(
            r#"{"freeze":{"source_material_id":"src","source_material_path":"clip","reverse_material_path":"","timestamp":1500}}"#,
        )
        .unwrap();
        let freeze = held.freeze.unwrap();
        assert_eq!(freeze.source_material_id, "src");
        assert_eq!(freeze.timestamp, Micros(1500));
    }
}
