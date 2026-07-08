//! Distributable template bundles (`.cutlassb`).
//!
//! A raw `.cutlasst` references its sample media by **local absolute path**
//! and is therefore not distributable — on any other machine every slot is
//! missing media. A bundle is the shippable form: one uncompressed tar
//! archive (sample media is already-compressed mp4/jpg; the JSON is small)
//! containing
//!
//! ```text
//! bundle.json          manifest: name, min_schema_version, format version
//! template.cutlasst    the template, media paths rewritten to media/<n>.<ext>
//! media/0.mp4          sample media at relative paths
//! media/1.jpg
//! ```
//!
//! The manifest exists so a gallery (or an older app) can decide whether it
//! can open the bundle **without** parsing the full template document:
//! `min_schema_version` is the embedded project's schema version, refused
//! by builds older than it.
//!
//! [`write`] rewrites pool paths to the relative `media/` form while
//! archiving; [`install`] extracts into a per-template directory and
//! rewrites them back to absolute, yielding a directory whose
//! `template.cutlasst` loads with the ordinary [`Template::load_from_file`].
//! Extraction validates entry names (the tar path-traversal classic) and
//! refuses anything outside `media/` and the two known files.

use std::io::Read;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::ModelError;
use crate::schema::PROJECT_SCHEMA_VERSION;
use crate::template::Template;

/// Recommended extension for template bundles.
pub const BUNDLE_FILE_EXTENSION: &str = "cutlassb";

/// Bumped only if the *archive layout* changes (not the template schema —
/// that is what `min_schema_version` tracks).
pub const BUNDLE_FORMAT_VERSION: u32 = 1;

/// The `bundle.json` manifest at the head of every bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleManifest {
    /// Archive layout version ([`BUNDLE_FORMAT_VERSION`]).
    pub format_version: u32,
    /// Display name (mirrors the template's meta, for gallery listings).
    pub name: String,
    /// The embedded project-schema version; apps older than this must
    /// refuse the bundle instead of half-parsing it.
    pub min_schema_version: u32,
}

/// Write `template` as a distributable bundle at `path`. Every pool media
/// file is read from its current (absolute) path and archived under
/// `media/`; the archived template references those relative paths.
///
/// Fails if any pool media file is missing on disk — a bundle with absent
/// sample media would be broken on every machine, so authoring refuses it
/// here rather than shipping the problem.
pub fn write(template: &Template, path: &Path) -> Result<(), ModelError> {
    let io_err = |e: std::io::Error| ModelError::InvalidProjectFile(e.to_string());

    // Plan the rewrite first: media id -> (source path, archive name).
    let mut entries: Vec<(crate::ids::MediaId, PathBuf, String)> = Vec::new();
    for (index, media) in template.project().media_iter().enumerate() {
        let source = media.path().to_path_buf();
        if !source.is_file() {
            return Err(ModelError::InvalidProjectFile(format!(
                "bundle refused: sample media missing on disk: {}",
                source.display()
            )));
        }
        let ext = source
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("bin")
            .to_lowercase();
        entries.push((media.id, source, format!("media/{index}.{ext}")));
    }

    // The archived template carries the relative paths.
    let mut doc = template.clone();
    for (id, _, archive_name) in &entries {
        if let Some(media) = doc.project.media_mut(*id) {
            media.path = PathBuf::from(archive_name);
        }
    }

    let manifest = BundleManifest {
        format_version: BUNDLE_FORMAT_VERSION,
        name: template.meta().name.clone(),
        min_schema_version: PROJECT_SCHEMA_VERSION,
    };

    let file = std::fs::File::create(path).map_err(io_err)?;
    let mut builder = tar::Builder::new(std::io::BufWriter::new(file));

    let manifest_json = serde_json::to_vec_pretty(&manifest)
        .map_err(|e| ModelError::InvalidProjectFile(e.to_string()))?;
    append_bytes(&mut builder, "bundle.json", &manifest_json).map_err(io_err)?;

    let template_json = serde_json::to_vec_pretty(&doc)
        .map_err(|e| ModelError::InvalidProjectFile(e.to_string()))?;
    append_bytes(&mut builder, "template.cutlasst", &template_json).map_err(io_err)?;

    for (_, source, archive_name) in &entries {
        builder
            .append_path_with_name(source, archive_name)
            .map_err(io_err)?;
    }
    builder.into_inner().map_err(io_err)?;
    Ok(())
}

/// Read just the manifest — the cheap compatibility check a gallery runs
/// before downloading nothing else or offering "requires a newer Cutlass".
pub fn read_manifest(path: &Path) -> Result<BundleManifest, ModelError> {
    let io_err = |e: std::io::Error| ModelError::InvalidProjectFile(e.to_string());
    let file = std::fs::File::open(path).map_err(io_err)?;
    let mut archive = tar::Archive::new(std::io::BufReader::new(file));
    for entry in archive.entries().map_err(io_err)? {
        let mut entry = entry.map_err(io_err)?;
        if entry.path().map_err(io_err)?.as_ref() == Path::new("bundle.json") {
            let mut raw = String::new();
            entry.read_to_string(&mut raw).map_err(io_err)?;
            return serde_json::from_str(&raw)
                .map_err(|e| ModelError::InvalidProjectFile(format!("bad bundle manifest: {e}")));
        }
    }
    Err(ModelError::InvalidProjectFile(
        "not a template bundle: no bundle.json".into(),
    ))
}

/// Extract a bundle into `dest_dir` (created; expected empty or absent) and
/// return the loaded, render-ready template whose media paths point into
/// `dest_dir/media/`. The rewritten `template.cutlasst` is left in
/// `dest_dir`, so subsequent sessions load it directly with
/// [`Template::load_from_file`] — install once, open forever.
pub fn install(path: &Path, dest_dir: &Path) -> Result<Template, ModelError> {
    let io_err = |e: std::io::Error| ModelError::InvalidProjectFile(e.to_string());

    let manifest = read_manifest(path)?;
    if manifest.format_version > BUNDLE_FORMAT_VERSION {
        return Err(ModelError::InvalidProjectFile(format!(
            "bundle format v{} is newer than this build supports (v{BUNDLE_FORMAT_VERSION})",
            manifest.format_version
        )));
    }
    if manifest.min_schema_version > PROJECT_SCHEMA_VERSION {
        return Err(ModelError::InvalidProjectFile(format!(
            "template requires a newer Cutlass (schema v{}, this build supports v{PROJECT_SCHEMA_VERSION})",
            manifest.min_schema_version
        )));
    }

    std::fs::create_dir_all(dest_dir).map_err(io_err)?;
    let file = std::fs::File::open(path).map_err(io_err)?;
    let mut archive = tar::Archive::new(std::io::BufReader::new(file));
    for entry in archive.entries().map_err(io_err)? {
        let mut entry = entry.map_err(io_err)?;
        let name = entry.path().map_err(io_err)?.into_owned();
        if !safe_entry_name(&name) {
            return Err(ModelError::InvalidProjectFile(format!(
                "bundle refused: unsafe entry path {:?}",
                name
            )));
        }
        let dest = dest_dir.join(&name);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(io_err)?;
        }
        entry.unpack(&dest).map_err(io_err)?;
    }

    // Rewrite the extracted template's relative media paths to absolute.
    let template_path = dest_dir.join("template.cutlasst");
    let mut template = Template::load_from_file(&template_path)?;
    let ids: Vec<crate::ids::MediaId> = template.project().media_iter().map(|m| m.id).collect();
    for id in ids {
        if let Some(media) = template.project.media_mut(id) {
            if media.path.is_relative() {
                media.path = dest_dir.join(&media.path);
            }
        }
    }
    template.save_to_file(&template_path).map_err(io_err)?;
    Ok(template)
}

/// Only the two known files and normal-component paths under `media/` may
/// unpack (no absolute paths, no `..`, no other roots).
fn safe_entry_name(name: &Path) -> bool {
    if name == Path::new("bundle.json") || name == Path::new("template.cutlasst") {
        return true;
    }
    let mut components = name.components();
    if components.next() != Some(Component::Normal("media".as_ref())) {
        return false;
    }
    components.all(|c| matches!(c, Component::Normal(_)))
}

fn append_bytes<W: std::io::Write>(
    builder: &mut tar::Builder<W>,
    name: &str,
    bytes: &[u8],
) -> std::io::Result<()> {
    let mut header = tar::Header::new_gnu();
    header.set_size(bytes.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();
    builder.append_data(&mut header, name, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::Replaceable;
    use crate::template::TemplateMeta;
    use crate::time::{Rational, RationalTime, TimeRange};
    use crate::track::TrackKind;
    use crate::{MediaSource, Project};

    const R24: Rational = Rational::FPS_24;

    /// A one-slot template whose sample media is a real file on disk.
    fn template_with_real_media(dir: &Path) -> Template {
        let sample_path = dir.join("sample.mp4");
        std::fs::write(&sample_path, b"not really an mp4, but bytes travel").unwrap();

        let mut project = Project::new("Bundled", R24);
        let media = project.add_media(MediaSource::new(&sample_path, 1920, 1080, R24, 240, true));
        let track = project.add_track(TrackKind::Video, "V1");
        let slot = project
            .add_clip(
                track,
                media,
                TimeRange::at_rate(0, 24, R24),
                RationalTime::new(0, R24),
            )
            .unwrap();
        project
            .set_replaceable(slot, Some(Replaceable::new(0)))
            .unwrap();
        Template::from_project(project, TemplateMeta::new("Bundled"))
    }

    #[test]
    fn bundle_roundtrips_and_installs_with_absolute_paths() {
        let dir = tempfile::tempdir().unwrap();
        let template = template_with_real_media(dir.path());
        let bundle = dir.path().join("bundled.cutlassb");
        write(&template, &bundle).unwrap();

        let manifest = read_manifest(&bundle).unwrap();
        assert_eq!(manifest.name, "Bundled");
        assert_eq!(manifest.min_schema_version, PROJECT_SCHEMA_VERSION);

        let install_dir = dir.path().join("installed/bundled");
        let installed = install(&bundle, &install_dir).unwrap();
        assert_eq!(installed.slot_count(), 1);
        for media in installed.project().media_iter() {
            assert!(media.path.is_absolute(), "rewritten: {:?}", media.path);
            assert!(media.path.is_file(), "extracted: {:?}", media.path);
            assert!(media.path.starts_with(&install_dir));
        }

        // Install once, open forever: the rewritten file loads directly.
        let reloaded = Template::load_from_file(&install_dir.join("template.cutlasst")).unwrap();
        assert_eq!(reloaded.meta().name, "Bundled");
        assert!(reloaded.project().media_iter().all(|m| m.path.is_file()));
    }

    #[test]
    fn write_refuses_missing_sample_media() {
        let mut project = Project::new("Broken", R24);
        project.add_media(MediaSource::new(
            "/definitely/not/here.mp4",
            1920,
            1080,
            R24,
            240,
            true,
        ));
        let template = Template::from_project(project, TemplateMeta::new("Broken"));
        let dir = tempfile::tempdir().unwrap();
        let err = write(&template, &dir.path().join("broken.cutlassb")).unwrap_err();
        assert!(err.to_string().contains("sample media missing"));
    }

    #[test]
    fn install_refuses_future_schema() {
        let dir = tempfile::tempdir().unwrap();
        let template = template_with_real_media(dir.path());
        let bundle = dir.path().join("future.cutlassb");
        write(&template, &bundle).unwrap();

        // Rewrite the manifest to demand a future schema.
        let raw = std::fs::read(&bundle).unwrap();
        let mut archive = tar::Archive::new(raw.as_slice());
        let out = dir.path().join("rebuild.cutlassb");
        let mut builder = tar::Builder::new(std::fs::File::create(&out).unwrap());
        for entry in archive.entries().unwrap() {
            let mut entry = entry.unwrap();
            let name = entry.path().unwrap().into_owned();
            let mut bytes = Vec::new();
            entry.read_to_end(&mut bytes).unwrap();
            if name == Path::new("bundle.json") {
                let mut manifest: BundleManifest = serde_json::from_slice(&bytes).unwrap();
                manifest.min_schema_version = 99;
                bytes = serde_json::to_vec(&manifest).unwrap();
            }
            let mut header = tar::Header::new_gnu();
            header.set_size(bytes.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder
                .append_data(&mut header, name.to_str().unwrap(), bytes.as_slice())
                .unwrap();
        }
        builder.into_inner().unwrap();

        let err = install(&out, &dir.path().join("nope")).unwrap_err();
        assert!(err.to_string().contains("newer Cutlass"), "{err}");
    }

    #[test]
    fn install_refuses_path_traversal_entries() {
        let dir = tempfile::tempdir().unwrap();
        let evil = dir.path().join("evil.cutlassb");
        let mut builder = tar::Builder::new(std::fs::File::create(&evil).unwrap());
        let manifest = BundleManifest {
            format_version: BUNDLE_FORMAT_VERSION,
            name: "Evil".into(),
            min_schema_version: 1,
        };
        let bytes = serde_json::to_vec(&manifest).unwrap();
        let mut header = tar::Header::new_gnu();
        header.set_size(bytes.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder
            .append_data(&mut header, "bundle.json", bytes.as_slice())
            .unwrap();
        // (tar::Builder itself refuses `..` in entry names, so the archived
        // probe uses an unknown root; the `..` form is covered below on the
        // validator directly.)
        let payload = b"pwned".to_vec();
        let mut header = tar::Header::new_gnu();
        header.set_size(payload.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder
            .append_data(&mut header, "not-media/escape.txt", payload.as_slice())
            .unwrap();
        builder.into_inner().unwrap();

        let err = install(&evil, &dir.path().join("safe")).unwrap_err();
        assert!(err.to_string().contains("unsafe entry"), "{err}");
        assert!(!dir.path().join("safe/not-media/escape.txt").exists());
    }

    #[test]
    fn entry_name_validator_rejects_traversal_and_absolutes() {
        assert!(safe_entry_name(Path::new("bundle.json")));
        assert!(safe_entry_name(Path::new("template.cutlasst")));
        assert!(safe_entry_name(Path::new("media/0.mp4")));
        assert!(safe_entry_name(Path::new("media/sub/1.jpg")));
        assert!(!safe_entry_name(Path::new("media/../escape.txt")));
        assert!(!safe_entry_name(Path::new("../escape.txt")));
        assert!(!safe_entry_name(Path::new("/etc/passwd")));
        assert!(!safe_entry_name(Path::new("other.txt")));
    }

    #[test]
    fn manifest_of_non_bundle_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let not_bundle = dir.path().join("plain.tar");
        let mut builder = tar::Builder::new(std::fs::File::create(&not_bundle).unwrap());
        let bytes = b"hello".to_vec();
        let mut header = tar::Header::new_gnu();
        header.set_size(bytes.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder
            .append_data(&mut header, "readme.txt", bytes.as_slice())
            .unwrap();
        builder.into_inner().unwrap();
        assert!(read_manifest(&not_bundle).is_err());
    }
}
