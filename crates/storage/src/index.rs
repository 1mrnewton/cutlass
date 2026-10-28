//! The index of bundled and cached resources.
//!
//! A record is an id, a location, a size, a checksum, and whether the bytes
//! are still partial. A preview may omit the checksum until the host provides
//! one. Paths in the file are relative. An id is not a path.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

use cutlass_download::Checksum;
use serde::{Deserialize, Serialize};

use crate::StorageError;

/// This build's index. A different version is refused.
pub(crate) const VERSION: u32 = 1;

/// A stable resource id. It is a single file name, not a path.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceId(String);

impl ResourceId {
    pub fn new(id: &str) -> Result<Self, StorageError> {
        if id.is_empty() {
            return Err(StorageError::EmptyId);
        }
        let bytes = id.as_bytes();
        let name = bytes.len() <= 128
            && bytes
                .iter()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
            && id != "."
            && id != "..";
        if !name {
            return Err(StorageError::InvalidId { id: id.to_owned() });
        }
        Ok(Self(id.to_owned()))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ResourceId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Where the bytes live.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Place {
    Bundle,
    Cache,
}

/// One indexed file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    id: ResourceId,
    place: Place,
    relative: PathBuf,
    size: u64,
    checksum: Option<Checksum>,
    partial: bool,
    preview: bool,
    age: u64,
}

impl Record {
    pub(crate) fn bundle(id: ResourceId, relative: PathBuf, size: u64, checksum: Checksum) -> Self {
        Self {
            id,
            place: Place::Bundle,
            relative,
            size,
            checksum: Some(checksum),
            partial: false,
            preview: false,
            age: 0,
        }
    }

    #[must_use]
    pub fn id(&self) -> &ResourceId {
        &self.id
    }

    #[must_use]
    pub fn place(&self) -> Place {
        self.place
    }

    #[must_use]
    pub fn relative(&self) -> &Path {
        &self.relative
    }

    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    #[must_use]
    pub fn checksum(&self) -> Option<Checksum> {
        self.checksum
    }

    #[must_use]
    pub fn is_partial(&self) -> bool {
        self.partial
    }

    #[must_use]
    pub fn is_preview(&self) -> bool {
        self.preview
    }

    /// Lowest age is the oldest cached preview. A package has no age.
    #[must_use]
    pub fn age(&self) -> u64 {
        self.age
    }

    pub(crate) fn cached(
        id: ResourceId,
        relative: PathBuf,
        size: u64,
        checksum: Option<Checksum>,
        partial: bool,
        preview: bool,
        age: u64,
    ) -> Self {
        Self {
            id,
            place: Place::Cache,
            relative,
            size,
            checksum,
            partial,
            preview,
            age,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Index {
    records: BTreeMap<ResourceId, Record>,
    next_age: u64,
}

impl Index {
    pub(crate) fn empty() -> Self {
        Self {
            records: BTreeMap::new(),
            next_age: 1,
        }
    }

    pub(crate) fn get(&self, id: &ResourceId) -> Option<&Record> {
        self.records.get(id)
    }

    pub(crate) fn insert(&mut self, record: Record) {
        self.records.insert(record.id.clone(), record);
    }

    pub(crate) fn remove(&mut self, id: &ResourceId) -> Option<Record> {
        self.records.remove(id)
    }

    pub(crate) fn set_checksum(&mut self, id: &ResourceId, checksum: Checksum) -> bool {
        let Some(record) = self.records.get_mut(id) else {
            return false;
        };
        record.checksum = Some(checksum);
        true
    }

    /// The next preview age. The counter moves only after the age is returned.
    pub(crate) fn allocate_age(&mut self) -> Result<u64, StorageError> {
        let age = self.next_age;
        self.next_age = self
            .next_age
            .checked_add(1)
            .ok_or_else(|| StorageError::Index {
                path: PathBuf::from("index.json"),
                message: "the cache age counter is exhausted".to_owned(),
            })?;
        Ok(age)
    }

    pub(crate) fn used_except(&self, id: &ResourceId) -> u64 {
        self.records
            .values()
            .filter(|record| record.place == Place::Cache && &record.id != id)
            .fold(0u64, |total, record| total.saturating_add(record.size))
    }

    pub(crate) fn used_bytes(&self) -> u64 {
        self.records
            .values()
            .filter(|record| record.place == Place::Cache)
            .fold(0u64, |total, record| total.saturating_add(record.size))
    }

    pub(crate) fn records(&self) -> impl Iterator<Item = &Record> {
        self.records.values()
    }

    pub(crate) fn snapshot(&self) -> Snapshot {
        Snapshot {
            version: VERSION,
            next_age: self.next_age,
            records: self
                .records
                .values()
                .map(|record| StoredRecord {
                    id: record.id.as_str().to_owned(),
                    place: record.place,
                    path: relative_text(&record.relative),
                    size: record.size,
                    checksum: record.checksum,
                    partial: record.partial,
                    preview: record.preview,
                    age: record.age,
                })
                .collect(),
        }
    }

    pub(crate) fn from_snapshot(snapshot: Snapshot) -> Result<Self, StorageError> {
        if snapshot.version != VERSION {
            return Err(StorageError::IndexVersion {
                found: snapshot.version,
            });
        }
        if snapshot.next_age == 0 {
            return Err(StorageError::Index {
                path: PathBuf::from("index.json"),
                message: "the next age starts at 1".to_owned(),
            });
        }
        let mut records = BTreeMap::new();
        for stored in snapshot.records {
            let record = stored.into_record()?;
            if record.age >= snapshot.next_age {
                return Err(StorageError::Index {
                    path: PathBuf::from("index.json"),
                    message: format!("{} has an age this index would reuse", record.id),
                });
            }
            if records.insert(record.id.clone(), record).is_some() {
                return Err(StorageError::Index {
                    path: PathBuf::from("index.json"),
                    message: "a resource id is listed twice".to_owned(),
                });
            }
        }
        Ok(Self {
            records,
            next_age: snapshot.next_age,
        })
    }
}

fn relative_text(path: &Path) -> String {
    let mut parts = Vec::new();
    for component in path.components() {
        if let Component::Normal(name) = component
            && let Some(text) = name.to_str()
        {
            parts.push(text);
        }
    }
    parts.join("/")
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Snapshot {
    version: u32,
    next_age: u64,
    records: Vec<StoredRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredRecord {
    id: String,
    place: Place,
    path: String,
    size: u64,
    checksum: Option<Checksum>,
    partial: bool,
    preview: bool,
    age: u64,
}

impl StoredRecord {
    fn into_record(self) -> Result<Record, StorageError> {
        let id = ResourceId::new(&self.id)?;
        let relative = relative_path(&self.path)?;
        if self.place == Place::Bundle && (self.partial || self.preview) {
            return Err(StorageError::Index {
                path: PathBuf::from("index.json"),
                message: format!("{id} is bundled, so it is not a cached preview"),
            });
        }
        if self.preview && self.age == 0 {
            return Err(StorageError::Index {
                path: PathBuf::from("index.json"),
                message: format!("{id} is a preview with no age"),
            });
        }
        if !self.preview && self.age != 0 {
            return Err(StorageError::Index {
                path: PathBuf::from("index.json"),
                message: format!("{id} is a package with an eviction age"),
            });
        }
        if !self.partial && !self.preview && self.checksum.is_none() {
            return Err(StorageError::Index {
                path: PathBuf::from("index.json"),
                message: format!("{id} is a package with no checksum"),
            });
        }
        Ok(Record {
            id,
            place: self.place,
            relative,
            size: self.size,
            checksum: self.checksum,
            partial: self.partial,
            preview: self.preview,
            age: self.age,
        })
    }
}

pub(crate) fn relative_path(text: &str) -> Result<PathBuf, StorageError> {
    let path = Path::new(text);
    if text.is_empty() || path.is_absolute() {
        return Err(StorageError::LeavesRoot {
            path: path.to_path_buf(),
        });
    }
    let mut count = 0u32;
    for component in path.components() {
        match component {
            Component::Normal(name) => {
                let Some(piece) = name.to_str() else {
                    return Err(StorageError::LeavesRoot {
                        path: path.to_path_buf(),
                    });
                };
                if piece.is_empty() || piece == "." || piece == ".." {
                    return Err(StorageError::LeavesRoot {
                        path: path.to_path_buf(),
                    });
                }
                count += 1;
            }
            _ => {
                return Err(StorageError::LeavesRoot {
                    path: path.to_path_buf(),
                });
            }
        }
    }
    if count == 0 {
        return Err(StorageError::LeavesRoot {
            path: path.to_path_buf(),
        });
    }
    Ok(path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::{Index, Place, Record, ResourceId, VERSION};
    use crate::StorageError;
    use cutlass_download::Checksum;

    #[test]
    fn an_empty_id_or_a_path_is_refused() {
        assert!(matches!(ResourceId::new(""), Err(StorageError::EmptyId)));
        for id in ["../secret", "a/b", "a\\b", ".", "..", "has space", "ü"] {
            assert!(
                matches!(ResourceId::new(id), Err(StorageError::InvalidId { .. })),
                "{id}"
            );
        }
        assert_eq!(ResourceId::new("glow-1").unwrap().as_str(), "glow-1");
    }

    #[test]
    fn a_bundle_record_does_not_count_toward_the_cache() {
        let mut index = Index::empty();
        index.insert(Record::bundle(
            ResourceId::new("glow").unwrap(),
            "effects/glow.bin".into(),
            4,
            Checksum::of(b"glow"),
        ));
        assert_eq!(index.used_bytes(), 0);
        let record = index.get(&ResourceId::new("glow").unwrap()).unwrap();
        assert_eq!(record.place(), Place::Bundle);
        assert!(!record.is_partial());
        assert!(!record.is_preview());
        assert!(record.checksum().is_some());
        let snapshot = index.snapshot();
        assert_eq!(snapshot.version, VERSION);
        let back = Index::from_snapshot(snapshot).unwrap();
        assert_eq!(
            back.get(&ResourceId::new("glow").unwrap()).unwrap().size(),
            4
        );
    }

    #[test]
    fn a_path_that_leaves_the_root_is_refused() {
        for path in [
            "",
            "..",
            "../secret",
            "/etc/passwd",
            "files/../../secret",
            "files/../x",
        ] {
            assert!(
                matches!(
                    super::relative_path(path),
                    Err(StorageError::LeavesRoot { .. })
                ),
                "{path}"
            );
        }
        assert_eq!(
            super::relative_path("effects/glow.bin").unwrap(),
            std::path::Path::new("effects/glow.bin")
        );
    }

    #[test]
    fn a_different_index_version_or_a_duplicate_id_is_refused() {
        let mut snapshot = Index::empty().snapshot();
        snapshot.version = 2;
        assert!(matches!(
            Index::from_snapshot(snapshot),
            Err(StorageError::IndexVersion { found: 2 })
        ));
        let mut snapshot = Index::empty().snapshot();
        snapshot.version = 0;
        assert!(matches!(
            Index::from_snapshot(snapshot),
            Err(StorageError::IndexVersion { found: 0 })
        ));
        let mut snapshot = Index::empty().snapshot();
        let record = super::StoredRecord {
            id: "glow".to_owned(),
            place: Place::Bundle,
            path: "effects/glow.bin".to_owned(),
            size: 1,
            checksum: Some(Checksum::of(b"g")),
            partial: false,
            preview: false,
            age: 0,
        };
        snapshot.records.push(record.clone());
        snapshot.records.push(record);
        assert!(matches!(
            Index::from_snapshot(snapshot),
            Err(StorageError::Index { .. })
        ));
    }
}
