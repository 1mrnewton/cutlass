use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::id::MaterialId;

/// A solid or gradient color clip.
///
/// `gradient_colors` and `gradient_percents` are arrays. The mapping table
/// records a string and one double; a lone JSON value is read as a one-element array.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct MaterialColorConfig {
    #[serde(default)]
    pub id: MaterialId,
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub is_color_clip: bool,
    #[serde(default)]
    pub is_gradient: bool,
    #[serde(default)]
    pub solid_color: String,
    #[serde(default, deserialize_with = "crate::serde_util::one_or_many_string")]
    pub gradient_colors: Vec<String>,
    #[serde(default, deserialize_with = "crate::serde_util::one_or_many_f64")]
    pub gradient_percents: Vec<f64>,
    #[serde(default)]
    pub gradient_angle: f64,
    #[serde(default)]
    pub width: f64,
    #[serde(default)]
    pub height: f64,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct MaterialCanvas {
    #[serde(default)]
    pub id: MaterialId,
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub color: String,
    #[serde(default)]
    pub blur: f64,
    #[serde(default)]
    pub image: String,
    #[serde(default)]
    pub album_image: String,
    #[serde(default)]
    pub image_id: String,
    #[serde(default)]
    pub image_name: String,
    #[serde(default)]
    pub source_platform: i32,
    #[serde(default)]
    pub team_id: String,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_stops_stay_arrays() {
        let absent: MaterialColorConfig = serde_json::from_str("{}").unwrap();
        assert!(absent.gradient_colors.is_empty());
        assert!(absent.gradient_percents.is_empty());
        let color: MaterialColorConfig = serde_json::from_str(
            r#"{"solid_color":"white","gradient_colors":"black","gradient_percents":[0.0,1.0]}"#,
        )
        .unwrap();
        assert_eq!(color.gradient_colors, vec!["black"]);
        assert_eq!(color.gradient_percents, vec![0.0, 1.0]);
        assert_eq!(
            serde_json::to_value(&color).unwrap()["gradient_colors"],
            serde_json::json!(["black"])
        );
    }
}
