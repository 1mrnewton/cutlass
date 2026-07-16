use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CommittedRelocation {
    pub(super) report: cutlass_storage::RelocationReport,
    pub(super) cleanup_warning: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CacheRelocationFailure(String);

impl CacheRelocationFailure {
    pub(super) fn from_message(message: impl Into<String>) -> Self {
        Self(bounded_message(&message.into()))
    }

    pub(super) fn with_detail(prefix: &str, detail: &str) -> Self {
        Self(bounded_message(&format!("{prefix}: {detail}")))
    }
}

impl fmt::Display for CacheRelocationFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for CacheRelocationFailure {}

pub(super) fn ensure_cache_can_be_relocated(id: CacheId) -> Result<(), String> {
    if !cache_descriptors()
        .iter()
        .any(|descriptor| descriptor.id == id)
    {
        return Err("unknown cache cannot be relocated".into());
    }
    let descriptor = id.descriptor();
    if descriptor.kind == CacheKind::Memory {
        return Err("memory cache cannot be relocated".into());
    }
    if descriptor.tier == CacheTier::UserData {
        return Err("user data cannot be relocated through the cache registry".into());
    }
    if !matches!(
        id,
        CacheId::Proxies
            | CacheId::Analysis
            | CacheId::AiModels
            | CacheId::Download
            | CacheId::Catalog
            | CacheId::Luts
            | CacheId::Lottie
            | CacheId::Templates
    ) {
        return Err("cache target is not relocatable".into());
    }
    Ok(())
}

pub(super) fn validate_relocation_paths(old_path: &Path, new_path: &Path) -> Result<(), String> {
    if old_path == new_path || old_path.starts_with(new_path) || new_path.starts_with(old_path) {
        return Err("cache relocation source and destination must not overlap".into());
    }
    Ok(())
}

pub(super) fn validate_relocation_destination(destination: &Path) -> Result<(), String> {
    match std::fs::symlink_metadata(destination) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            Err("cache relocation destination cannot be a symbolic link".into())
        }
        Ok(_) => Err("cache relocation destination already exists".into()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(_) => Err("cache relocation destination could not be inspected".into()),
    }
}

pub(super) fn set_storage_path_override(
    settings: &mut cutlass_settings::Settings,
    id: CacheId,
    path: PathBuf,
) -> Result<(), String> {
    ensure_cache_can_be_relocated(id)?;
    let field = match id {
        CacheId::Proxies => &mut settings.storage.paths.proxies,
        CacheId::Analysis => &mut settings.storage.paths.analysis,
        CacheId::AiModels => &mut settings.storage.paths.ai_models,
        CacheId::Download => &mut settings.storage.paths.download,
        CacheId::Catalog => &mut settings.storage.paths.catalog,
        CacheId::Luts => &mut settings.storage.paths.luts,
        CacheId::Lottie => &mut settings.storage.paths.lottie,
        CacheId::Templates => &mut settings.storage.paths.templates,
        _ => return Err("cache target has no storage settings field".into()),
    };
    *field = Some(path);
    Ok(())
}

pub(super) fn persist_relocation_settings(
    config_path: &Path,
    settings: &cutlass_settings::Settings,
) -> Result<(), String> {
    cutlass_settings::save(config_path, settings)
        .map_err(|_| "storage settings could not be saved".to_string())
}

pub(super) fn validate_relocation_references(
    id: CacheId,
    saved: &DraftReferenceReport,
    live: &CacheReferenceReport,
) -> Result<(), CacheRelocationFailure> {
    let has_persisted_project_references = match id {
        CacheId::Proxies | CacheId::Analysis | CacheId::AiModels | CacheId::Catalog => false,
        CacheId::Download | CacheId::Luts | CacheId::Lottie | CacheId::Templates => true,
        CacheId::PreviewFrames
        | CacheId::LibraryThumbnails
        | CacheId::TimelineFilmstrips
        | CacheId::TimelineWaveforms => {
            return Err(CacheRelocationFailure::from_message(
                "memory cache has no relocation reference policy",
            ));
        }
    };
    if !saved.is_complete() {
        return Err(CacheRelocationFailure::from_message(
            "saved project reference inventory is incomplete; cache relocation was refused",
        ));
    }
    if !live.is_complete() {
        return Err(CacheRelocationFailure::from_message(
            "current project reference inventory is incomplete; cache relocation was refused",
        ));
    }
    if !has_persisted_project_references {
        return Ok(());
    }

    let saved_references = saved.references.by_cache.get(&id).ok_or_else(|| {
        CacheRelocationFailure::from_message(
            "saved project reference inventory omitted the target cache",
        )
    })?;
    let live_references = live.by_cache.get(&id).ok_or_else(|| {
        CacheRelocationFailure::from_message(
            "current project reference inventory omitted the target cache",
        )
    })?;
    if !saved_references.is_empty() || !live_references.is_empty() {
        return Err(CacheRelocationFailure::from_message(
            "project files reference this cache; cache relocation was refused",
        ));
    }
    Ok(())
}

pub(super) fn relocate_disk_root<F>(
    old_path: &Path,
    new_path: &Path,
    cancel: &AtomicBool,
    persist: F,
) -> Result<CommittedRelocation, CacheRelocationFailure>
where
    F: FnOnce(&Path) -> Result<(), String>,
{
    ensure_not_cancelled(cancel, "cancelled before creating the cache source")
        .map_err(CacheRelocationFailure::from_message)?;
    ensure_relocation_source_exists(old_path)?;
    let cancellation = || cancel.load(Ordering::Acquire);
    classify_relocation_result(relocate_cache(old_path, new_path, &cancellation, persist)).map_err(
        |error| {
            CacheRelocationFailure::with_detail(
                "cache filesystem relocation failed",
                &error.to_string(),
            )
        },
    )
}

pub(super) fn relocate_download_root<F>(
    cache: &DownloadCache,
    expected_old_path: &Path,
    new_path: &Path,
    cancel: &AtomicBool,
    persist: F,
) -> Result<CommittedRelocation, CacheRelocationFailure>
where
    F: FnOnce(&Path) -> Result<(), String>,
{
    let mut committed = None;
    cache
        .switch_root(new_path, |old_path, destination| {
            if old_path != expected_old_path || destination != new_path {
                return Err(CacheRelocationFailure::from_message(
                    "download cache root changed before relocation",
                ));
            }
            committed = Some(relocate_disk_root(old_path, destination, cancel, persist)?);
            Ok(())
        })
        .map_err(|error| {
            CacheRelocationFailure::with_detail(
                "download cache root could not be switched",
                &error.to_string(),
            )
        })?;
    committed.ok_or_else(|| {
        CacheRelocationFailure::from_message(
            "download cache root switched without relocation accounting",
        )
    })
}

fn ensure_relocation_source_exists(path: &Path) -> Result<(), CacheRelocationFailure> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => std::fs::create_dir_all(path)
            .map_err(|error| {
                CacheRelocationFailure::with_detail(
                    "missing cache source could not be created",
                    &error.to_string(),
                )
            }),
        Err(error) => Err(CacheRelocationFailure::with_detail(
            "cache source could not be inspected",
            &error.to_string(),
        )),
    }
}

pub(super) fn classify_relocation_result(
    result: Result<cutlass_storage::RelocationReport, StorageError>,
) -> Result<CommittedRelocation, StorageError> {
    match result {
        Ok(report) => Ok(CommittedRelocation {
            report,
            cleanup_warning: None,
        }),
        Err(error) => {
            let Some(report) = error.committed_relocation() else {
                return Err(error);
            };
            Ok(CommittedRelocation {
                report,
                cleanup_warning: Some(bounded_message(&format!(
                    "cache relocation committed, but old-copy cleanup did not finish: {error}"
                ))),
            })
        }
    }
}
