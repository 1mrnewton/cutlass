use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::clip::Point;
use crate::id::MaterialId;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enums::KeyframeCurve;

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
}
