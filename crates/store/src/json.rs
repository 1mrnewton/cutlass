//! Draft JSON.
//!
//! Four default policies stay separate here. A mapping fallback is a named
//! constant on the model, and a missing key is not filled with it. A
//! constructor is `Draft::empty` or a material `new`. An absent key is what
//! the model deserializes when the file omits the key. `Default` is the Rust
//! value. Loading does not call a constructor, and it does not replace a
//! stored value with either of the other two.

use std::path::Path;

use cutlass_model::Draft;
use serde_json::Value;

use crate::StoreError;

/// Read one draft. Unknown keys, including keys inside a known record, stay on the draft.
pub fn parse_draft(bytes: &[u8], path: &Path) -> Result<Draft, StoreError> {
    serde_json::from_slice(bytes).map_err(|error| StoreError::Json {
        path: path.to_path_buf(),
        message: error.to_string(),
    })
}

/// Write one draft. A missing animation duration stays missing.
pub fn write_draft(draft: &Draft) -> Result<Vec<u8>, StoreError> {
    let mut bytes = serde_json::to_vec(draft).map_err(|error| StoreError::Json {
        path: Path::new("draft_info.json").to_path_buf(),
        message: error.to_string(),
    })?;
    bytes.push(b'\n');
    Ok(bytes)
}

/// Read one JSON value stored beside the draft.
pub fn parse_value(bytes: &[u8], path: &Path) -> Result<Value, StoreError> {
    serde_json::from_slice(bytes).map_err(|error| StoreError::Json {
        path: path.to_path_buf(),
        message: error.to_string(),
    })
}

/// Write one JSON value and a trailing newline.
pub fn write_value(value: &Value, path: &Path) -> Result<Vec<u8>, StoreError> {
    let mut bytes = serde_json::to_vec(value).map_err(|error| StoreError::Json {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    bytes.push(b'\n');
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use cutlass_model::{Draft, Micros, StickerAnimation};
    use serde_json::Value;

    use super::{parse_draft, write_draft};

    #[test]
    fn an_unknown_field_inside_a_known_record_round_trips() {
        let path = Path::new("draft_info.json");
        let raw = br#"{"id":"D","version":360000,"new_version":"189.0.0","canvas_config":{"ratio":"original","width":1920,"height":1080,"future_canvas":1},"tracks":[{"id":"t","type":"video","flag":0,"segments":[{"id":"s","material_id":"m","target_timerange":{"start":0,"duration":1,"future_span":true},"clip":{"scale":{"x":1.0,"y":1.0},"rotation":0.0,"transform":{"x":0.0,"y":0.0},"flip":{"horizontal":false,"vertical":false},"alpha":1.0,"future_clip":2}}]}]}"#;
        let draft = parse_draft(raw, path).unwrap();
        assert_eq!(draft.canvas_config.rest["future_canvas"], 1);
        let back = write_draft(&draft).unwrap();
        let value: Value = serde_json::from_slice(&back).unwrap();
        assert_eq!(value["canvas_config"]["future_canvas"], 1);
        assert_eq!(
            value["tracks"][0]["segments"][0]["target_timerange"]["future_span"],
            true
        );
        assert_eq!(value["tracks"][0]["segments"][0]["clip"]["future_clip"], 2);
        assert_eq!(value["tracks"][0]["segments"][0]["id"], "s");
    }

    #[test]
    fn the_four_default_policies_stay_apart() {
        let path = Path::new("draft_info.json");
        let mapping = StickerAnimation::MAPPING_DURATION;
        assert_eq!(mapping, Micros(3_000_000));

        let constructed = StickerAnimation::new("anim");
        assert_eq!(constructed.id, "anim");
        assert!(constructed.duration.is_none());

        let absent = parse_draft(
            br#"{"id":"D","version":1,"materials":{"material_animations":[{"id":"pack","animations":[{"id":"anim"}]}]}}"#,
            path,
        )
        .unwrap();
        let anim = &absent.materials.material_animations[0].animations[0];
        assert!(anim.duration.is_none());
        assert_ne!(anim.duration, Some(mapping));
        assert_ne!(anim.duration, Some(Micros(0)));

        let defaulted = StickerAnimation::default();
        assert!(defaulted.id.is_empty());
        assert_ne!(constructed.id, defaulted.id);

        let empty = Draft::empty("D");
        assert!(empty.config.maintrack_adsorb);
        let parsed = parse_draft(br#"{"id":"D","version":1}"#, path).unwrap();
        assert!(!parsed.config.maintrack_adsorb);
    }

    #[test]
    fn a_stored_zero_duration_stays_zero() {
        let path = Path::new("draft_info.json");
        let draft = parse_draft(
            br#"{"id":"D","version":1,"materials":{"material_animations":[{"animations":[{"duration":0}]}]}}"#,
            path,
        )
        .unwrap();
        let anim = &draft.materials.material_animations[0].animations[0];
        assert_eq!(anim.duration, Some(Micros(0)));
        let value: Value = serde_json::from_slice(&write_draft(&draft).unwrap()).unwrap();
        assert_eq!(
            value["materials"]["material_animations"][0]["animations"][0]["duration"],
            0
        );
    }

    #[test]
    fn truncated_json_is_refused() {
        let path = Path::new("draft_info.json");
        let error = parse_draft(b"{", path).unwrap_err();
        assert!(matches!(error, crate::StoreError::Json { .. }));
        let error = parse_draft(b"", path).unwrap_err();
        assert!(matches!(error, crate::StoreError::Json { .. }));
        let error = parse_draft(b"[]", path).unwrap_err();
        assert!(matches!(error, crate::StoreError::Json { .. }));
    }
}
