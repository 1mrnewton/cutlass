//! The library's folders: `draft_virtual_store.json`.
//!
//! The file holds one array, `draft_virtual_store`, of typed lists. Type 0
//! is the folders, type 1 the parent links, type 2 the path rows. A link
//! puts a folder or a library item under a folder; an empty parent is the
//! top of the library. A path row with `is_hidden` keeps an item out of the
//! library. Keys this build does not know, in every row and around the
//! lists, are kept as they were read.

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

const FOLDERS: i64 = 0;
const LINKS: i64 = 1;
const PATHS: i64 = 2;

/// The library's folders, where each folder and item sits, and the items
/// taken out of it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VirtualStore {
    pub folders: Vec<BinFolder>,
    pub links: Vec<BinLink>,
    pub paths: Vec<BinPath>,
    /// Keys beside `type` and `value` on the folder, link and path lists.
    pub list_rest: [Map<String, Value>; 3],
    /// Lists of a type this build does not know, kept in their place.
    pub other: Vec<Value>,
    /// Keys beside `draft_virtual_store`.
    pub rest: Map<String, Value>,
}

/// A library folder. A row with an empty id is the library's own top, as
/// the reference editor saves it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BinFolder {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub display_name: String,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// `child_id` sits in the folder `parent_id`; an empty parent is the top.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BinLink {
    #[serde(default)]
    pub child_id: String,
    #[serde(default)]
    pub parent_id: String,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// A library item's file. A hidden row keeps the item out of the library.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BinPath {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub path: String,
    #[serde(default, deserialize_with = "flag")]
    pub is_encrypted: bool,
    #[serde(default, deserialize_with = "flag")]
    pub is_hidden: bool,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

impl BinFolder {
    /// A new folder with the keys the reference editor writes for one.
    /// `now` is seconds since 1970.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>, now: i64) -> Self {
        let mut rest = Map::new();
        for (key, value) in [
            ("creation_time", now),
            ("import_time", now),
            ("import_time_us", now.saturating_mul(1_000_000)),
            ("sort_type", 0),
            ("sort_sub_type", 0),
            ("filter_type", 0),
            ("subdraft_filter_type", 0),
            ("material_color_tag", 0),
        ] {
            rest.insert(key.to_owned(), Value::from(value));
        }
        Self {
            id: id.into(),
            display_name: name.into(),
            rest,
        }
    }
}

/// A bool the file may store as true or false, or as 0 or 1.
fn flag<'de, D: Deserializer<'de>>(deserializer: D) -> Result<bool, D::Error> {
    match Value::deserialize(deserializer)? {
        Value::Bool(on) => Ok(on),
        Value::Number(number) => Ok(number.as_f64().is_some_and(|value| value != 0.0)),
        Value::Null => Ok(false),
        other => Err(D::Error::custom(format!("expected a flag, found {other}"))),
    }
}

#[derive(Serialize, Deserialize)]
struct RawStore {
    #[serde(default)]
    draft_virtual_store: Vec<RawList>,
    #[serde(flatten, default)]
    rest: Map<String, Value>,
}

#[derive(Serialize, Deserialize)]
struct RawList {
    #[serde(rename = "type")]
    kind: i64,
    #[serde(default)]
    value: Vec<Value>,
    #[serde(flatten, default)]
    rest: Map<String, Value>,
}

fn items<T: for<'de> Deserialize<'de>>(values: Vec<Value>) -> Result<Vec<T>, serde_json::Error> {
    values.into_iter().map(serde_json::from_value).collect()
}

fn values<T: Serialize>(items: &[T]) -> Result<Vec<Value>, serde_json::Error> {
    items.iter().map(serde_json::to_value).collect()
}

impl<'de> Deserialize<'de> for VirtualStore {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = RawStore::deserialize(deserializer)?;
        let mut store = Self {
            rest: raw.rest,
            ..Self::default()
        };
        for list in raw.draft_virtual_store {
            let slot = match list.kind {
                FOLDERS => {
                    store.folders = items(list.value).map_err(D::Error::custom)?;
                    0
                }
                LINKS => {
                    store.links = items(list.value).map_err(D::Error::custom)?;
                    1
                }
                PATHS => {
                    store.paths = items(list.value).map_err(D::Error::custom)?;
                    2
                }
                _ => {
                    store
                        .other
                        .push(serde_json::to_value(list).map_err(D::Error::custom)?);
                    continue;
                }
            };
            store.list_rest[slot] = list.rest;
        }
        Ok(store)
    }
}

impl Serialize for VirtualStore {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::Error as _;
        let mut lists = vec![
            RawList {
                kind: FOLDERS,
                value: values(&self.folders).map_err(S::Error::custom)?,
                rest: self.list_rest[0].clone(),
            },
            RawList {
                kind: LINKS,
                value: values(&self.links).map_err(S::Error::custom)?,
                rest: self.list_rest[1].clone(),
            },
            RawList {
                kind: PATHS,
                value: values(&self.paths).map_err(S::Error::custom)?,
                rest: self.list_rest[2].clone(),
            },
        ];
        for other in &self.other {
            lists.push(serde_json::from_value(other.clone()).map_err(S::Error::custom)?);
        }
        RawStore {
            draft_virtual_store: lists,
            rest: self.rest.clone(),
        }
        .serialize(serializer)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn the_three_lists_read_and_write_back_with_unknown_keys_kept() {
        let file = json!({
            "draft_virtual_store": [
                {"type": 0, "value": [{
                    "id": "", "display_name": "", "creation_time": 0,
                    "import_time": 0, "import_time_us": 0, "sort_type": 0,
                    "sort_sub_type": 0, "filter_type": 0,
                    "subdraft_filter_type": 0, "material_color_tag": 0
                }]},
                {"type": 1, "value": [{"child_id": "a", "parent_id": ""}], "note": 3},
                {"type": 2, "value": [{"id": "m", "path": "/x.mov", "is_encrypted": 0, "is_hidden": 1}]},
                {"type": 7, "value": [1, 2]}
            ],
            "draft_materials": []
        });
        let store: VirtualStore = serde_json::from_value(file.clone()).unwrap();
        assert_eq!(store.folders.len(), 1);
        assert_eq!(store.folders[0].rest["sort_type"], json!(0));
        assert_eq!(store.links[0].child_id, "a");
        assert!(store.paths[0].is_hidden);
        assert!(!store.paths[0].is_encrypted);
        assert_eq!(store.other.len(), 1);
        assert_eq!(store.list_rest[1]["note"], json!(3));
        assert!(store.rest.contains_key("draft_materials"));
        let written = serde_json::to_value(&store).unwrap();
        let again: VirtualStore = serde_json::from_value(written.clone()).unwrap();
        assert_eq!(again, store);
        assert_eq!(
            written["draft_virtual_store"][3],
            json!({"type": 7, "value": [1, 2]})
        );
        assert_eq!(written["draft_virtual_store"][1]["note"], json!(3));
        assert_eq!(written["draft_virtual_store"].as_array().unwrap().len(), 4);
        assert_eq!(
            written["draft_virtual_store"][2]["value"][0]["is_hidden"],
            json!(true)
        );
    }

    #[test]
    fn an_empty_file_is_an_empty_library_and_a_new_folder_has_its_keys() {
        let store: VirtualStore = serde_json::from_value(json!({})).unwrap();
        assert_eq!(store, VirtualStore::default());
        let folder = BinFolder::new("f", "Untitled folder", 10);
        assert_eq!(folder.rest["import_time_us"], json!(10_000_000));
        assert_eq!(folder.rest.len(), 8);
    }
}
