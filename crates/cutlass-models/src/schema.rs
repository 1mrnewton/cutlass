//! Project document schema identity and version metadata.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Stable format family for a Cutlass timeline project.
pub const PROJECT_SCHEMA_KIND: &str = "cutlass.project";

/// Current [`ProjectSchema::version`] for newly created projects.
///
/// History:
/// - **1** — alpha format through M0/M3.
/// - **2** — M2 animatable params: transform properties may serialize as
///   `{"kf": [...]}` keyframe curves instead of bare values. v2 readers
///   accept v1 files unchanged (constant params share the v1 shape); v1
///   builds refuse v2 files rather than half-parse keyframes.
/// - **3** — Mirror mask is a parallel band (`size[0]` = thickness). v2
///   files that stored the unused default `[1,1]` (legacy half-plane
///   ignored size) are rewritten to `size[0] = 0.5` so they stay visible.
///
/// Versioning policy (v1 roadmap M0 — the rules for changing the format):
///
/// - **Adding an optional field?** Don't bump. Ship it with
///   `#[serde(default)]` + skip-if-default on save (e.g. `Clip::volume`,
///   `MediaSource::is_image`). Same-version readers tolerate fields they
///   don't know and drop them on resave.
/// - **Changing a shape, renaming, or making a field required?** Bump this
///   constant and add the matching `migrate_vN_to_vN1` step to
///   `migrate_document` in `persist.rs` (a test fails if the step is
///   missing).
/// - **Files newer than this build are refused** on load with
///   [`ModelError::UnsupportedProjectSchema`](crate::ModelError) — never
///   guessed at.
pub const PROJECT_SCHEMA_VERSION: u32 = 3;

/// Identifies the serialized shape of a [`Project`](crate::Project).
///
/// Kept as a structured object so future formats can carry kind, extensions,
/// and migration hints without another top-level field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectSchema {
    /// Monotonic document version; bump when project fields change.
    pub version: u32,
    /// Stable format identifier (e.g. [`PROJECT_SCHEMA_KIND`]).
    pub kind: String,
    /// Optional capability tags present in this document (`"nested_timelines"`, …).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extensions: Vec<String>,
}

impl ProjectSchema {
    /// Schema stamped on new projects and written on save.
    pub fn current() -> Self {
        Self {
            version: PROJECT_SCHEMA_VERSION,
            kind: PROJECT_SCHEMA_KIND.into(),
            extensions: Vec::new(),
        }
    }

    /// Whether this engine build can load the schema without migration.
    /// Every version up to the current one reads forward (the M2+ policy:
    /// newer fields are additive or tolerated as bare-value constants).
    pub fn is_supported(&self) -> bool {
        self.kind == PROJECT_SCHEMA_KIND
            && self.version >= 1
            && self.version <= PROJECT_SCHEMA_VERSION
    }
}

pub fn serialize<S>(schema: &ProjectSchema, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    schema.serialize(serializer)
}

/// Accept a full schema object or a legacy bare version integer.
pub fn deserialize<'de, D>(deserializer: D) -> Result<ProjectSchema, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Repr {
        Full(ProjectSchema),
        LegacyVersion(u32),
    }

    match Repr::deserialize(deserializer)? {
        Repr::Full(schema) => Ok(schema),
        Repr::LegacyVersion(version) => Ok(ProjectSchema {
            version,
            kind: PROJECT_SCHEMA_KIND.into(),
            extensions: Vec::new(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_is_supported() {
        let schema = ProjectSchema::current();
        assert!(schema.is_supported());
        assert_eq!(schema.version, PROJECT_SCHEMA_VERSION);
        assert_eq!(schema.kind, PROJECT_SCHEMA_KIND);
    }

    #[test]
    fn deserialize_legacy_integer_version() {
        #[derive(Deserialize)]
        struct Holder {
            #[serde(deserialize_with = "super::deserialize")]
            schema: ProjectSchema,
        }
        let holder: Holder = serde_json::from_value(serde_json::json!({ "schema": 1 })).unwrap();
        assert_eq!(holder.schema.version, 1);
        assert_eq!(holder.schema.kind, PROJECT_SCHEMA_KIND);
        assert!(holder.schema.extensions.is_empty());
    }

    #[test]
    fn deserialize_full_object() {
        let schema: ProjectSchema = serde_json::from_value(serde_json::json!({
            "version": 1,
            "kind": "cutlass.project",
            "extensions": ["draft"]
        }))
        .unwrap();
        assert_eq!(schema.extensions, vec!["draft"]);
    }
}
