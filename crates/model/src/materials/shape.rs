use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::id::MaterialId;

/// A shape. Roundness, custom points, size, scale, and the in and out tangents
/// are arrays of doubles. A lone JSON number is read as a one-element array.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct MaterialShape {
    #[serde(default)]
    pub id: MaterialId,
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub shape_type: i32,
    #[serde(default, deserialize_with = "crate::serde_util::one_or_many_f64")]
    pub roundness: Vec<f64>,
    #[serde(default, deserialize_with = "crate::serde_util::one_or_many_f64")]
    pub custom_points: Vec<f64>,
    #[serde(default, deserialize_with = "crate::serde_util::one_or_many_f64")]
    pub shape_size: Vec<f64>,
    #[serde(default)]
    pub global_alpha: f64,
    #[serde(default)]
    pub color: String,
    #[serde(default)]
    pub border_line_style: i32,
    #[serde(default)]
    pub border_width: f64,
    #[serde(default)]
    pub border_color: String,
    #[serde(default)]
    pub shadow_color: String,
    #[serde(default)]
    pub shadow_alpha: f64,
    #[serde(default)]
    pub shadow_distance: f64,
    #[serde(default)]
    pub shadow_angle: f64,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub combo_info: Option<Value>,
    #[serde(default, deserialize_with = "crate::serde_util::one_or_many_f64")]
    pub shape_scale: Vec<f64>,
    #[serde(default, deserialize_with = "crate::serde_util::one_or_many_f64")]
    pub custom_points_in: Vec<f64>,
    #[serde(default, deserialize_with = "crate::serde_util::one_or_many_f64")]
    pub custom_points_out: Vec<f64>,
    #[serde(default)]
    pub border_alpha: f64,
    #[serde(default)]
    pub shadow_ambiguity: f64,
    #[serde(default)]
    pub endpoint_left_style: i32,
    #[serde(default)]
    pub endpoint_right_style: i32,
    #[serde(default)]
    pub line_style: i32,
    #[serde(default)]
    pub fill_render_style: Option<Value>,
    #[serde(default)]
    pub constant_material_id: String,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contour_fields_are_arrays_and_absence_is_not_a_zero() {
        let absent: MaterialShape = serde_json::from_str("{}").unwrap();
        assert!(absent.roundness.is_empty());
        assert!(absent.shape_size.is_empty());
        assert_ne!(absent.roundness, vec![0.0]);

        let shape: MaterialShape = serde_json::from_str(
            r#"{"id":"sh","roundness":0.25,"custom_points":[0.0,1.0],"shape_size":[10.0,20.0],"shape_scale":1.0,"custom_points_in":[0.0],"custom_points_out":[1.0],"future_shape":true}"#,
        )
        .unwrap();
        assert_eq!(shape.roundness, vec![0.25]);
        assert_eq!(shape.custom_points, vec![0.0, 1.0]);
        assert_eq!(shape.shape_size, vec![10.0, 20.0]);
        assert_eq!(shape.shape_scale, vec![1.0]);
        let json = serde_json::to_value(&shape).unwrap();
        assert_eq!(json["roundness"], serde_json::json!([0.25]));
        assert_eq!(json["future_shape"], true);
    }
}
