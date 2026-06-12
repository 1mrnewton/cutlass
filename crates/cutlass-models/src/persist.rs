//! On-disk project files (`.cutlass` JSON).

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Write};
use std::path::Path;

use serde::Deserialize;

use crate::error::ModelError;
use crate::project::Project;
use crate::schema::{ProjectSchema, PROJECT_SCHEMA_VERSION};

/// Recommended extension for Cutlass project files.
pub const PROJECT_FILE_EXTENSION: &str = "cutlass";

/// Numeric version from [`ProjectSchema::current`] for simple callers.
pub const PROJECT_FILE_VERSION: u32 = PROJECT_SCHEMA_VERSION;

/// Legacy envelope used by early v1 saves (`{ "version", "project" }`).
#[derive(Debug, Deserialize)]
struct LegacyProjectFile {
    version: u32,
    project: Project,
}

impl Project {
    /// Serialize this project to a `.cutlass` JSON file.
    ///
    /// The document is stamped with this build's schema version regardless
    /// of what was loaded: the writer defines the format. (A project opened
    /// from a v1 file may now hold v2-only data like keyframes; persisting
    /// it as "v1" would lie to older readers.)
    pub fn save_to_file(&self, path: &Path) -> io::Result<()> {
        let mut doc = self.clone();
        doc.schema.version = PROJECT_SCHEMA_VERSION;
        doc.schema.kind = crate::schema::PROJECT_SCHEMA_KIND.into();
        let json = serde_json::to_string_pretty(&doc).map_err(io::Error::other)?;
        let mut writer = BufWriter::new(File::create(path)?);
        writer.write_all(json.as_bytes())?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        Ok(())
    }

    /// Deserialize a project from a `.cutlass` JSON file.
    pub fn load_from_file(path: &Path) -> Result<Project, ModelError> {
        let file = File::open(path).map_err(|e| ModelError::InvalidProjectFile(e.to_string()))?;
        let reader = BufReader::new(file);
        let mut project = match serde_json::from_reader::<_, Project>(reader) {
            Ok(project) => project,
            Err(_) => {
                let file = File::open(path).map_err(|e| ModelError::InvalidProjectFile(e.to_string()))?;
                let reader = BufReader::new(file);
                let legacy: LegacyProjectFile = serde_json::from_reader(reader)
                    .map_err(|e| ModelError::InvalidProjectFile(e.to_string()))?;
                let mut project = legacy.project;
                if project.schema.version == 0 {
                    project.schema.version = legacy.version;
                }
                project
            }
        };
        normalize_legacy_schema(&mut project.schema);
        validate_schema(&project.schema)?;
        Ok(project)
    }
}

fn normalize_legacy_schema(schema: &mut ProjectSchema) {
    if schema.kind.is_empty() {
        schema.kind = crate::schema::PROJECT_SCHEMA_KIND.into();
    }
}

fn validate_schema(found: &ProjectSchema) -> Result<(), ModelError> {
    let expected = ProjectSchema::current();
    if !found.is_supported() {
        return Err(ModelError::UnsupportedProjectSchema {
            found: found.clone(),
            expected,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::{Clip, Generator};
    use crate::time::{Rational, RationalTime, TimeRange};
    use crate::track::TrackKind;

    const R24: Rational = Rational::FPS_24;

    #[test]
    fn roundtrip_save_load_preserves_timeline_and_metadata() {
        let mut project = Project::new("demo", R24);
        project.metadata_mut().description = "rough cut".into();
        project.metadata_mut().author = Some("editor".into());
        let media_id = project.add_media(crate::MediaSource::new(
            "/tmp/clip.mp4",
            1920,
            1080,
            R24,
            240,
            true,
        ));
        let track = project.add_track(TrackKind::Video, "V1");
        project
            .add_clip(
                track,
                media_id,
                TimeRange::at_rate(0, 48, R24),
                RationalTime::new(0, R24),
            )
            .unwrap();
        let overlay = project.add_track(TrackKind::Text, "T1");
        project
            .timeline_mut()
            .add_clip(
                overlay,
                Clip::generated(Generator::text("hi"), TimeRange::at_rate(48, 24, R24)),
            )
            .unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.cutlass");
        project.save_to_file(&path).unwrap();
        let loaded = Project::load_from_file(&path).unwrap();

        assert_eq!(loaded.schema, ProjectSchema::current());
        assert_eq!(loaded.id, project.id);
        assert_eq!(loaded.name, "demo");
        assert_eq!(loaded.metadata, project.metadata);
        assert_eq!(loaded.media_count(), 1);
        assert_eq!(loaded.timeline().clip_count(), 2);
    }

    #[test]
    fn saved_file_writes_schema_object() {
        let project = Project::new("shape", R24);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shape.cutlass");
        project.save_to_file(&path).unwrap();
        let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let schema = json.get("schema").expect("schema object");
        assert_eq!(schema["version"], PROJECT_SCHEMA_VERSION);
        assert_eq!(schema["kind"], "cutlass.project");
    }

    #[test]
    fn load_accepts_v1_schema_files() {
        // Every pre-M2 alpha save is a v1 file; v2 readers open them as-is.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v1.cutlass");
        std::fs::write(
            &path,
            r#"{"schema":{"version":1,"kind":"cutlass.project"},"id":1,"name":"v1","metadata":{},"media":[],"timeline":{"frame_rate":{"num":24,"den":1},"tracks":[],"order":[],"clip_index":[]}}"#,
        )
        .unwrap();
        let loaded = Project::load_from_file(&path).unwrap();
        assert_eq!(loaded.schema.version, 1);

        // Re-saving a v1 project writes the current format version.
        let resaved = dir.path().join("resaved.cutlass");
        loaded.save_to_file(&resaved).unwrap();
        let reloaded = Project::load_from_file(&resaved).unwrap();
        assert_eq!(reloaded.schema.version, PROJECT_SCHEMA_VERSION);
    }

    #[test]
    fn keyframed_transform_survives_save_load() {
        use crate::clip::{ClipParam, ParamValue};
        use crate::param::Easing;

        let mut project = Project::new("anim", R24);
        let track = project.add_track(TrackKind::Text, "T1");
        let clip = project
            .timeline_mut()
            .add_clip(
                track,
                Clip::generated(Generator::text("fade"), TimeRange::at_rate(0, 48, R24)),
            )
            .unwrap();
        project
            .set_param_keyframe(
                clip,
                ClipParam::Opacity,
                RationalTime::new(0, R24),
                ParamValue::Scalar(0.0),
                Easing::EaseInOut,
            )
            .unwrap();
        project
            .set_param_keyframe(
                clip,
                ClipParam::Opacity,
                RationalTime::new(24, R24),
                ParamValue::Scalar(1.0),
                Easing::Linear,
            )
            .unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("anim.cutlass");
        project.save_to_file(&path).unwrap();
        let loaded = Project::load_from_file(&path).unwrap();
        let transform = &loaded.clip(clip).unwrap().transform;
        assert!(transform.is_animated());
        assert_eq!(transform.opacity.keyframes().len(), 2);
        assert_eq!(transform.sample(24).opacity, 1.0);
    }

    #[test]
    fn load_rejects_unknown_schema_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.cutlass");
        std::fs::write(
            &path,
            r#"{"schema":{"version":99,"kind":"cutlass.project"},"id":1,"name":"x","metadata":{},"media":[],"timeline":{"frame_rate":{"num":24,"den":1},"tracks":[],"order":[],"clip_index":[]}}"#,
        )
        .unwrap();
        let err = Project::load_from_file(&path).unwrap_err();
        assert!(matches!(
            err,
            ModelError::UnsupportedProjectSchema { .. }
        ));
    }

    #[test]
    fn load_accepts_legacy_integer_schema_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy-int.cutlass");
        std::fs::write(
            &path,
            r#"{"schema_version":1,"id":1,"name":"legacy","metadata":{},"media":[],"timeline":{"frame_rate":{"num":24,"den":1},"tracks":[],"order":[],"clip_index":[]}}"#,
        )
        .unwrap();
        let loaded = Project::load_from_file(&path).unwrap();
        assert_eq!(loaded.schema.version, 1);
        assert_eq!(loaded.schema.kind, crate::schema::PROJECT_SCHEMA_KIND);
    }

    #[test]
    fn load_accepts_legacy_envelope() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy.cutlass");
        std::fs::write(
            &path,
            r#"{"version":1,"project":{"schema":{"version":1,"kind":"cutlass.project"},"id":1,"name":"legacy","metadata":{"description":"old envelope"},"media":[],"timeline":{"frame_rate":{"num":24,"den":1},"tracks":[],"order":[],"clip_index":[]}}}"#,
        )
        .unwrap();
        let loaded = Project::load_from_file(&path).unwrap();
        assert_eq!(loaded.name, "legacy");
        assert_eq!(loaded.metadata.description, "old envelope");
    }
}
