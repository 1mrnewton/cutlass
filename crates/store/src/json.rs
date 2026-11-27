//! Draft JSON.
//!
//! Four default policies stay separate here. A mapping fallback is a named
//! constant on the model, and a missing key is not filled with it. A
//! constructor is `Draft::empty` or a material `new`. An absent key is what
//! the model deserializes when the file omits the key. `Default` is the Rust
//! value. Loading does not call a constructor, and it does not replace a
//! stored value with either of the other two.
//!
//! A project file comes from outside, so reading it is bounded. A file past
//! [`MAX_FILE_BYTES`] is refused before it is read. A string past
//! [`MAX_STRING_BYTES`], or an array or object past [`MAX_ITEMS`], is refused
//! before anything is built from the file. Nesting is bounded by serde_json's
//! own limit of 128 levels, and a deeper file is refused as
//! [`StoreError::Json`]. Each refusal is an error, not a panic.

use std::cell::Cell;
use std::fmt;
use std::path::Path;

use cutlass_model::Draft;
use serde::de::{self, DeserializeOwned, DeserializeSeed, Deserializer, MapAccess, SeqAccess};
use serde_json::Value;

use crate::StoreError;

/// The largest project file read, in bytes: 256 MiB.
///
/// A draft holds the edit, not the media. A long project with word-timed
/// captions and a keyframe on every frame of an hour's clip still writes a
/// draft in the tens of megabytes, so a real project stays well under this.
/// The size is taken from the file's metadata before a byte is read, and the
/// read stops one byte past it in case the file grew in between, so a larger
/// file costs no memory.
pub(crate) const MAX_FILE_BYTES: u64 = 256 * 1024 * 1024;

/// The longest single string read, in bytes once escapes are decoded: 16 MiB.
///
/// The longest strings a draft holds are a caption's styled text and a file
/// path, kilobytes at most. A map key counts as a string.
pub(crate) const MAX_STRING_BYTES: usize = 16 * 1024 * 1024;

/// The most items a single array, or members a single object, may hold: four million.
///
/// The longest lists a draft holds are keyframes and word timings. Four
/// million is a keyframe on every frame of over 18 hours at 60 frames a second.
pub(crate) const MAX_ITEMS: usize = 4_000_000;

/// Which bound a project file went past.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JsonLimit {
    /// The file is larger than `MAX_FILE_BYTES`, 256 MiB.
    File,
    /// One string or map key is longer than `MAX_STRING_BYTES`, 16 MiB.
    String,
    /// One array or object holds more than `MAX_ITEMS`, four million.
    Items,
}

impl fmt::Display for JsonLimit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::File => write!(formatter, "the file is larger than {MAX_FILE_BYTES} bytes"),
            Self::String => write!(
                formatter,
                "a string is longer than {MAX_STRING_BYTES} bytes"
            ),
            Self::Items => write!(
                formatter,
                "an array or object holds more than {MAX_ITEMS} items"
            ),
        }
    }
}

/// Read one draft. Unknown keys, including keys inside a known record, stay on the draft.
///
/// A caption that marks no words of its own takes the words its
/// `subtitle_keywords` spans, so keywords marked in the reference editor show.
pub fn parse_draft(bytes: &[u8], path: &Path) -> Result<Draft, StoreError> {
    let mut draft: Draft = parse_struct(bytes, path)?;
    for text in &mut draft.materials.texts {
        text.adopt_subtitle_keywords();
    }
    Ok(draft)
}

/// Read one typed file of the project, inside the same bounds as the draft.
pub(crate) fn parse_struct<T: DeserializeOwned>(
    bytes: &[u8],
    path: &Path,
) -> Result<T, StoreError> {
    check_limits(bytes, path)?;
    serde_json::from_slice(bytes).map_err(|error| StoreError::Json {
        path: path.to_path_buf(),
        message: error.to_string(),
    })
}

/// Refuse `bytes` when one string, array, or object in it is past its bound.
///
/// This walks the JSON once with serde_json's own parser, so the grammar and
/// the nesting limit are the ones the typed parse uses, and it keeps nothing:
/// a string without escapes is borrowed from `bytes`. Bytes it cannot walk are
/// refused here too, so nothing past a bad token goes unchecked.
fn check_limits(bytes: &[u8], path: &Path) -> Result<(), StoreError> {
    let hit = Cell::new(None);
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    match (Walk { hit: &hit }).deserialize(&mut deserializer) {
        Ok(()) => Ok(()),
        Err(error) => Err(match hit.get() {
            Some(limit) => StoreError::TooLarge {
                path: path.to_path_buf(),
                limit,
            },
            None => StoreError::Json {
                path: path.to_path_buf(),
                message: error.to_string(),
            },
        }),
    }
}

/// One step of [`check_limits`]. `hit` names the bound that stopped the walk.
#[derive(Clone, Copy)]
struct Walk<'a> {
    hit: &'a Cell<Option<JsonLimit>>,
}

impl Walk<'_> {
    fn refuse<E: de::Error>(self, limit: JsonLimit) -> E {
        self.hit.set(Some(limit));
        E::custom(limit)
    }

    fn count<E: de::Error>(self, items: usize) -> Result<(), E> {
        if items > MAX_ITEMS {
            Err(self.refuse(JsonLimit::Items))
        } else {
            Ok(())
        }
    }
}

impl<'de> DeserializeSeed<'de> for Walk<'_> {
    type Value = ();

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        deserializer.deserialize_any(self)
    }
}

impl<'de> de::Visitor<'de> for Walk<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value")
    }

    fn visit_bool<E: de::Error>(self, _: bool) -> Result<(), E> {
        Ok(())
    }

    fn visit_i64<E: de::Error>(self, _: i64) -> Result<(), E> {
        Ok(())
    }

    fn visit_u64<E: de::Error>(self, _: u64) -> Result<(), E> {
        Ok(())
    }

    fn visit_f64<E: de::Error>(self, _: f64) -> Result<(), E> {
        Ok(())
    }

    fn visit_unit<E: de::Error>(self) -> Result<(), E> {
        Ok(())
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<(), E> {
        if value.len() > MAX_STRING_BYTES {
            Err(self.refuse(JsonLimit::String))
        } else {
            Ok(())
        }
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
        let mut items = 0_usize;
        while seq.next_element_seed(self)?.is_some() {
            items += 1;
            self.count(items)?;
        }
        Ok(())
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        let mut items = 0_usize;
        while map.next_key_seed(self)?.is_some() {
            map.next_value_seed(self)?;
            items += 1;
            self.count(items)?;
        }
        Ok(())
    }
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

/// Read one JSON value stored beside the draft, inside the same bounds as the draft.
pub fn parse_value(bytes: &[u8], path: &Path) -> Result<Value, StoreError> {
    parse_struct(bytes, path)
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

    use super::{
        JsonLimit, MAX_ITEMS, MAX_STRING_BYTES, check_limits, parse_draft, parse_value, write_draft,
    };

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
    fn marked_spans_load_as_caption_keywords_and_stay_on_the_draft() {
        let path = Path::new("draft_info.json");
        let raw = br#"{"id":"D","version":1,"materials":{"texts":[{"id":"t","type":"subtitle","content":"so very good","subtitle_keywords":{"range":[{"location":8,"length":4,"source_type":0}]},"words":{"start_time":[0,100,200],"end_time":[100,200,300],"text":["so","very","good"]}}]}}"#;
        let draft = parse_draft(raw, path).unwrap();
        assert_eq!(draft.materials.texts[0].caption_keywords, [2]);
        let value: Value = serde_json::from_slice(&write_draft(&draft).unwrap()).unwrap();
        let text = &value["materials"]["texts"][0];
        assert_eq!(text["cutlass_caption_keywords"], serde_json::json!([2]));
        assert_eq!(
            text["subtitle_keywords"],
            serde_json::json!({"range":[{"location":8,"length":4,"source_type":0}]})
        );
    }

    fn limit_of(bytes: &[u8]) -> Option<JsonLimit> {
        match check_limits(bytes, Path::new("draft_info.json")) {
            Ok(()) => None,
            Err(crate::StoreError::TooLarge { limit, .. }) => Some(limit),
            Err(error) => panic!("refused for another reason: {error}"),
        }
    }

    fn string_of(len: usize) -> Vec<u8> {
        let mut bytes = vec![b'a'; len + 2];
        bytes[0] = b'"';
        bytes[len + 1] = b'"';
        bytes
    }

    fn array_of(items: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(items * 2 + 1);
        bytes.push(b'[');
        for item in 0..items {
            if item > 0 {
                bytes.push(b',');
            }
            bytes.push(b'0');
        }
        bytes.push(b']');
        bytes
    }

    #[test]
    fn a_string_at_its_bound_passes_and_one_byte_more_is_refused() {
        assert_eq!(limit_of(&string_of(MAX_STRING_BYTES)), None);
        assert_eq!(
            limit_of(&string_of(MAX_STRING_BYTES + 1)),
            Some(JsonLimit::String)
        );
    }

    #[test]
    fn a_long_key_and_a_long_escaped_string_are_refused() {
        let mut key = b"{".to_vec();
        key.extend_from_slice(&string_of(MAX_STRING_BYTES + 1));
        key.extend_from_slice(b":0}");
        assert_eq!(limit_of(&key), Some(JsonLimit::String));

        let mut escaped = b"\"".to_vec();
        for _ in 0..=MAX_STRING_BYTES {
            escaped.extend_from_slice(br"\n");
        }
        escaped.push(b'"');
        assert_eq!(limit_of(&escaped), Some(JsonLimit::String));
    }

    #[test]
    fn an_array_at_its_bound_passes_and_one_item_more_is_refused() {
        assert_eq!(limit_of(&array_of(MAX_ITEMS)), None);
        assert_eq!(limit_of(&array_of(MAX_ITEMS + 1)), Some(JsonLimit::Items));
    }

    #[test]
    fn an_object_with_too_many_members_is_refused() {
        let mut object = b"{".to_vec();
        for member in 0..=MAX_ITEMS {
            if member > 0 {
                object.push(b',');
            }
            object.extend_from_slice(br#""":0"#);
        }
        object.push(b'}');
        assert_eq!(limit_of(&object), Some(JsonLimit::Items));
    }

    #[test]
    fn a_bound_is_refused_wherever_it_sits_and_through_parse_value() {
        let mut nested = br#"{"a":[{"b":"#.to_vec();
        nested.extend_from_slice(&string_of(MAX_STRING_BYTES + 1));
        nested.extend_from_slice(b"}]}");
        let error = parse_value(&nested, Path::new("project.json")).unwrap_err();
        assert!(matches!(
            error,
            crate::StoreError::TooLarge { limit: JsonLimit::String, ref path }
                if path.ends_with("project.json")
        ));
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
