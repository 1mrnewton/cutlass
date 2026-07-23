use std::cell::{Cell, RefCell};

use super::*;
use crate::clip::{Clip, Generator};
use crate::time::{Rational, RationalTime, TimeRange};
use crate::track::TrackKind;

const R24: Rational = Rational::FPS_24;

#[derive(Default)]
struct FaultFs {
    operations: RefCell<Vec<&'static str>>,
    create_new_calls: Cell<usize>,
    create_dir_calls: Cell<usize>,
    rename_calls: Cell<usize>,
    remove_file_calls: Cell<usize>,
    temp_collisions: usize,
    backup_collisions: usize,
    fail_write: bool,
    fail_flush: bool,
    fail_permissions: bool,
    fail_sync: bool,
    failed_renames: Vec<usize>,
    failed_remove_files: Vec<usize>,
}

impl FaultFs {
    fn next_call(counter: &Cell<usize>) -> usize {
        let call = counter.get() + 1;
        counter.set(call);
        call
    }

    fn record(&self, operation: &'static str) {
        self.operations.borrow_mut().push(operation);
    }

    fn injected(operation: &str, call: usize) -> io::Error {
        io::Error::other(format!("injected {operation} failure #{call}"))
    }
}

impl PersistenceFs for FaultFs {
    fn create_new(&self, path: &Path) -> io::Result<File> {
        self.record("create_new");
        let call = Self::next_call(&self.create_new_calls);
        if call <= self.temp_collisions {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "injected temporary-path collision",
            ));
        }
        open_new_temp(path)
    }

    fn write_all(&self, file: &mut File, contents: &[u8]) -> io::Result<()> {
        self.record("write_all");
        if self.fail_write {
            return Err(Self::injected("write", 1));
        }
        Write::write_all(file, contents)
    }

    fn flush(&self, file: &mut File) -> io::Result<()> {
        self.record("flush");
        if self.fail_flush {
            return Err(Self::injected("flush", 1));
        }
        Write::flush(file)
    }

    fn set_permissions(&self, file: &File, permissions: Permissions) -> io::Result<()> {
        self.record("set_permissions");
        if self.fail_permissions {
            return Err(Self::injected("permission", 1));
        }
        file.set_permissions(permissions)
    }

    fn sync_all(&self, file: &File) -> io::Result<()> {
        self.record("sync_all");
        if self.fail_sync {
            return Err(Self::injected("sync", 1));
        }
        file.sync_all()
    }

    fn symlink_metadata(&self, path: &Path) -> io::Result<std::fs::Metadata> {
        std::fs::symlink_metadata(path)
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        self.record("rename");
        let call = Self::next_call(&self.rename_calls);
        if self.failed_renames.contains(&call) {
            return Err(Self::injected("rename", call));
        }
        std::fs::rename(from, to)
    }

    fn create_dir(&self, path: &Path) -> io::Result<()> {
        self.record("create_dir");
        let call = Self::next_call(&self.create_dir_calls);
        if call <= self.backup_collisions {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "injected backup-path collision",
            ));
        }
        create_private_dir(path)
    }

    fn remove_file(&self, path: &Path) -> io::Result<()> {
        self.record("remove_file");
        let call = Self::next_call(&self.remove_file_calls);
        if self.failed_remove_files.contains(&call) {
            return Err(Self::injected("remove-file", call));
        }
        std::fs::remove_file(path)
    }

    fn remove_dir(&self, path: &Path) -> io::Result<()> {
        self.record("remove_dir");
        std::fs::remove_dir(path)
    }
}

fn transaction_artifacts(directory: &Path) -> Vec<PathBuf> {
    let mut artifacts: Vec<_> = std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name().is_some_and(|name| {
                let name = name.to_string_lossy();
                name.starts_with(".cutlass-project-tmp-")
                    || name.starts_with(".cutlass-project-backup-")
            })
        })
        .collect();
    artifacts.sort();
    artifacts
}

fn assert_no_transaction_artifacts(directory: &Path) {
    let artifacts = transaction_artifacts(directory);
    assert!(
        artifacts.is_empty(),
        "unexpected project-save artifacts: {artifacts:?}"
    );
}

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
    assert_no_transaction_artifacts(dir.path());
}

#[test]
fn save_preserves_pretty_json_with_trailing_newline() {
    let project = Project::new("format", R24);
    let mut expected_document = project.clone();
    expected_document.schema.version = PROJECT_SCHEMA_VERSION;
    expected_document.schema.kind = crate::schema::PROJECT_SCHEMA_KIND.into();
    let mut expected = serde_json::to_string_pretty(&expected_document).unwrap();
    expected.push('\n');

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("format.cutlass");
    project.save_to_file(&path).unwrap();

    assert_eq!(std::fs::read_to_string(path).unwrap(), expected);
    assert_no_transaction_artifacts(dir.path());
}

#[test]
fn atomic_save_replaces_and_roundtrips_without_artifacts() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("replace.cutlass");
    Project::new("old", R24).save_to_file(&path).unwrap();

    let mut replacement = Project::new("new", R24);
    replacement.metadata_mut().description = "replacement".into();
    replacement.save_to_file(&path).unwrap();

    let loaded = Project::load_from_file(&path).unwrap();
    assert_eq!(loaded.name, "new");
    assert_eq!(loaded.metadata.description, "replacement");
    assert_no_transaction_artifacts(dir.path());
}

#[cfg(unix)]
#[test]
fn serialization_failure_precedes_destination_access() {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("existing.cutlass");
    let original = b"known-good project";
    std::fs::write(&path, original).unwrap();

    let mut project = Project::new("invalid path", R24);
    project.add_media(crate::MediaSource::new(
        PathBuf::from(OsString::from_vec(vec![0xff])),
        1920,
        1080,
        R24,
        24,
        false,
    ));

    project.save_to_file(&path).unwrap_err();

    assert_eq!(std::fs::read(&path).unwrap(), original);
    assert_no_transaction_artifacts(dir.path());
}

#[test]
fn sync_failure_keeps_old_project_and_prevents_install() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sync.cutlass");
    let original = b"known-good project";
    std::fs::write(&path, original).unwrap();
    let fs = FaultFs {
        fail_sync: true,
        ..FaultFs::default()
    };

    let error = persist_serialized(&path, b"replacement", &fs).unwrap_err();

    assert!(error.to_string().contains("could not sync"), "{error}");
    assert_eq!(std::fs::read(&path).unwrap(), original);
    assert_eq!(
        *fs.operations.borrow(),
        [
            "create_new",
            "write_all",
            "flush",
            "set_permissions",
            "sync_all",
            "remove_file"
        ]
    );
    assert_eq!(fs.rename_calls.get(), 0);
    assert_no_transaction_artifacts(dir.path());
}

#[test]
fn fallback_swap_replaces_existing_project_and_cleans_backup() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fallback.cutlass");
    let original = b"old project";
    let replacement = b"new project";
    std::fs::write(&path, original).unwrap();
    let fs = FaultFs {
        failed_renames: vec![1],
        ..FaultFs::default()
    };

    persist_serialized(&path, replacement, &fs).unwrap();

    assert_eq!(std::fs::read(&path).unwrap(), replacement);
    assert_eq!(fs.rename_calls.get(), 3);
    assert_no_transaction_artifacts(dir.path());
}

#[test]
fn fallback_install_failure_restores_original_project() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rollback.cutlass");
    let original = b"old project";
    std::fs::write(&path, original).unwrap();
    let fs = FaultFs {
        failed_renames: vec![1, 3],
        ..FaultFs::default()
    };

    let error = persist_serialized(&path, b"new project", &fs).unwrap_err();

    assert!(
        error.to_string().contains("original project was restored"),
        "{error}"
    );
    assert_eq!(std::fs::read(&path).unwrap(), original);
    assert_eq!(fs.rename_calls.get(), 4);
    assert_no_transaction_artifacts(dir.path());
}

#[test]
fn failed_rollback_retains_the_only_good_copy_in_backup() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("retained.cutlass");
    let original = b"old project";
    std::fs::write(&path, original).unwrap();
    let fs = FaultFs {
        failed_renames: vec![1, 3, 4],
        ..FaultFs::default()
    };

    let error = persist_serialized(&path, b"new project", &fs).unwrap_err();

    assert!(
        error.to_string().contains("retained in a sibling backup"),
        "{error}"
    );
    assert!(!path.exists());
    let artifacts = transaction_artifacts(dir.path());
    assert_eq!(artifacts.len(), 1, "{artifacts:?}");
    assert_eq!(
        std::fs::read(artifacts[0].join("original")).unwrap(),
        original
    );
}

#[test]
fn committed_save_ignores_backup_cleanup_failure() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("committed.cutlass");
    let original = b"old project";
    let replacement = b"new project";
    std::fs::write(&path, original).unwrap();
    let fs = FaultFs {
        failed_renames: vec![1],
        failed_remove_files: vec![1],
        ..FaultFs::default()
    };

    persist_serialized(&path, replacement, &fs).unwrap();

    assert_eq!(std::fs::read(&path).unwrap(), replacement);
    let artifacts = transaction_artifacts(dir.path());
    assert_eq!(artifacts.len(), 1, "{artifacts:?}");
    assert_eq!(
        std::fs::read(artifacts[0].join("original")).unwrap(),
        original
    );
    std::fs::remove_dir_all(&artifacts[0]).unwrap();
}

#[test]
fn unique_path_allocation_is_bounded_under_collisions() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("collision.cutlass");

    let temp_fs = FaultFs {
        temp_collisions: UNIQUE_PATH_ATTEMPTS,
        ..FaultFs::default()
    };
    let error = create_unique_temp(&path, &temp_fs).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
    assert_eq!(temp_fs.create_new_calls.get(), UNIQUE_PATH_ATTEMPTS);

    let backup_fs = FaultFs {
        backup_collisions: UNIQUE_PATH_ATTEMPTS,
        ..FaultFs::default()
    };
    let error = create_unique_backup(&path, &backup_fs).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
    assert_eq!(backup_fs.create_dir_calls.get(), UNIQUE_PATH_ATTEMPTS);
    assert_no_transaction_artifacts(dir.path());
}

#[cfg(unix)]
#[test]
fn save_preserves_existing_file_permissions() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("permissions.cutlass");
    std::fs::write(&path, b"old project").unwrap();
    std::fs::set_permissions(&path, Permissions::from_mode(0o640)).unwrap();

    Project::new("permissions", R24)
        .save_to_file(&path)
        .unwrap();

    let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o640);
    assert_no_transaction_artifacts(dir.path());
}

#[cfg(unix)]
#[test]
fn save_rejects_symlink_destination_without_touching_target() {
    use std::os::unix::fs::symlink;

    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("target.cutlass");
    let destination = dir.path().join("link.cutlass");
    let original = b"known-good project";
    std::fs::write(&target, original).unwrap();
    symlink(&target, &destination).unwrap();

    let error = Project::new("replacement", R24)
        .save_to_file(&destination)
        .unwrap_err();

    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    assert!(
        !error
            .to_string()
            .contains(dir.path().to_string_lossy().as_ref()),
        "error exposed the project directory: {error}"
    );
    assert_eq!(std::fs::read(&target).unwrap(), original);
    assert!(
        std::fs::symlink_metadata(&destination)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert_no_transaction_artifacts(dir.path());
}

#[test]
fn saved_file_writes_schema_object() {
    let project = Project::new("shape", R24);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shape.cutlass");
    project.save_to_file(&path).unwrap();
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
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
            None,
        )
        .unwrap();
    project
        .set_param_keyframe(
            clip,
            ClipParam::Opacity,
            RationalTime::new(24, R24),
            ParamValue::Scalar(1.0),
            Easing::Linear,
            None,
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
    assert!(matches!(err, ModelError::UnsupportedProjectSchema { .. }));
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
fn load_tolerates_unknown_optional_fields() {
    // The versioning policy: additive optional fields don't bump the
    // schema version, so a same-version file written by a newer build
    // may carry fields this build doesn't know. Loading must succeed
    // and keep everything this build *does* know.
    let mut project = Project::new("tolerant", R24);
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

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tolerant.cutlass");
    project.save_to_file(&path).unwrap();

    // Doctor the saved JSON with unknown fields at every level a future
    // build is likely to extend.
    let mut doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    doc["color_management"] = serde_json::json!({ "working_space": "bt709" });
    // The media pool serializes as `[id, source]` pairs.
    doc["media"][0][1]["pixel_aspect"] = serde_json::json!(1.0);
    doc["timeline"]["markers_future"] = serde_json::json!([{ "tick": 5, "name": "beat" }]);
    doc["timeline"]["tracks"][0][1]["clips"][0][1]["future_field"] = serde_json::json!(true);
    std::fs::write(&path, serde_json::to_string(&doc).unwrap()).unwrap();

    let loaded = Project::load_from_file(&path).unwrap();
    assert_eq!(loaded.name, "tolerant");
    assert_eq!(loaded.media_count(), 1);
    assert_eq!(loaded.timeline().clip_count(), 1);

    // The other half of the contract: unknown fields are ignored, not
    // preserved — a resave by this build drops them.
    let resaved = dir.path().join("resaved.cutlass");
    loaded.save_to_file(&resaved).unwrap();
    let json = std::fs::read_to_string(&resaved).unwrap();
    assert!(!json.contains("color_management"));
    assert!(!json.contains("markers_future"));
}

#[test]
fn migration_chain_covers_every_supported_version() {
    // Walking from the oldest supported version must find a step arm
    // for every gap — `migrate_document` panics if a schema bump landed
    // without its migration step.
    for from in 1..=PROJECT_SCHEMA_VERSION {
        let mut doc = serde_json::json!({});
        migrate_document(&mut doc, from);
        // Empty docs have nothing to rewrite.
        assert_eq!(doc, serde_json::json!({}));
    }
}

#[test]
fn migrate_v2_mirror_default_size_to_half_width_band() {
    let mut doc = serde_json::json!({
        "timeline": {
            "tracks": [[1, {
                "clips": [[10, {
                    "mask": { "kind": "mirror" }
                }], [11, {
                    "mask": {
                        "kind": "mirror",
                        "size": [1.0, 1.0]
                    }
                }], [12, {
                    "mask": {
                        "kind": "mirror",
                        "size": [0.25, 1.0]
                    }
                }], [13, {
                    "mask": {
                        "kind": "circle",
                        "size": [1.0, 1.0]
                    }
                }]]
            }]]
        }
    });
    migrate_document(&mut doc, 2);
    let clips = &doc["timeline"]["tracks"][0][1]["clips"];
    assert_eq!(clips[0][1]["mask"]["size"], serde_json::json!([0.5, 1.0]));
    assert_eq!(clips[1][1]["mask"]["size"], serde_json::json!([0.5, 1.0]));
    assert_eq!(
        clips[2][1]["mask"]["size"],
        serde_json::json!([0.25, 1.0]),
        "custom thickness must be preserved"
    );
    assert_eq!(
        clips[3][1]["mask"]["size"],
        serde_json::json!([1.0, 1.0]),
        "non-mirror kinds are untouched"
    );
}

#[test]
fn unknown_root_project_field_does_not_trip_envelope_unwrap() {
    // A current-version file may carry unknown root fields named
    // "project"/"version" (tolerance policy); the legacy-envelope
    // detection must not unwrap those.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("notenvelope.cutlass");
    std::fs::write(
        &path,
        r#"{"schema":{"version":2,"kind":"cutlass.project"},"id":7,"name":"keep","metadata":{},"media":[],"timeline":{"frame_rate":{"num":24,"den":1},"tracks":[],"order":[],"clip_index":[]},"project":"unknown-extension","version":9}"#,
    )
    .unwrap();
    let loaded = Project::load_from_file(&path).unwrap();
    assert_eq!(loaded.name, "keep");
    // Provenance keeps the file's schema version; save re-stamps current.
    assert_eq!(loaded.schema.version, 2);
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
