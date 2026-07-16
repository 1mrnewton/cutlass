use super::*;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct CacheUsage {
    pub(super) bytes: u64,
    pub(super) entries: u64,
    pub(super) files: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct UiCacheUsage {
    pub(super) thumbnails: CacheUsage,
    pub(super) filmstrips: CacheUsage,
    pub(super) waveforms: CacheUsage,
}

/// One cache's current usage and immutable descriptor metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CacheSnapshot {
    pub(crate) id: CacheId,
    pub(crate) label: &'static str,
    pub(crate) kind: CacheKind,
    pub(crate) tier: CacheTier,
    pub(crate) path: Option<PathBuf>,
    pub(crate) bytes: u64,
    pub(crate) entries: u64,
    pub(crate) files: u64,
}

impl Serialize for CacheSnapshot {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut fields = serializer
            .serialize_struct("CacheSnapshot", if self.path.is_some() { 8 } else { 7 })?;
        fields.serialize_field("cache_id", self.id.as_str())?;
        fields.serialize_field("label", self.label)?;
        fields.serialize_field("kind", cache_kind_key(self.kind))?;
        fields.serialize_field("tier", cache_tier_key(self.tier))?;
        if let Some(path) = &self.path {
            fields.serialize_field("path", &path.to_string_lossy())?;
        }
        fields.serialize_field("bytes", &self.bytes)?;
        fields.serialize_field("entries", &self.entries)?;
        fields.serialize_field("files", &self.files)?;
        fields.end()
    }
}

/// Exact pre-clear accounting plus a current snapshot when it was practical
/// to collect one after the operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CacheClearReport {
    pub(crate) id: CacheId,
    pub(crate) removed_bytes: u64,
    pub(crate) removed_entries: u64,
    pub(crate) removed_files: u64,
    pub(crate) current: Option<CacheSnapshot>,
}

impl Serialize for CacheClearReport {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut fields = serializer.serialize_struct(
            "CacheClearReport",
            if self.current.is_some() { 5 } else { 4 },
        )?;
        fields.serialize_field("cache_id", self.id.as_str())?;
        fields.serialize_field("removed_bytes", &self.removed_bytes)?;
        fields.serialize_field("removed_entries", &self.removed_entries)?;
        fields.serialize_field("removed_files", &self.removed_files)?;
        if let Some(current) = &self.current {
            fields.serialize_field("cache", current)?;
        }
        fields.end()
    }
}

/// Committed relocation accounting and the newly published cache generation.
#[allow(dead_code)] // Public(crate) DTO for the following UI/agent wiring slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CacheRelocationReport {
    pub(crate) id: CacheId,
    pub(crate) old_path: PathBuf,
    pub(crate) new_path: PathBuf,
    pub(crate) bytes: u64,
    pub(crate) files: u64,
    pub(crate) used_copy_fallback: bool,
    pub(crate) cleanup_warning: Option<String>,
    pub(crate) generation: u64,
    pub(crate) current: Option<CacheSnapshot>,
}

impl Serialize for CacheRelocationReport {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut fields = serializer.serialize_struct(
            "CacheRelocationReport",
            7 + usize::from(self.cleanup_warning.is_some()) + usize::from(self.current.is_some()),
        )?;
        fields.serialize_field("cache_id", self.id.as_str())?;
        fields.serialize_field("old_path", &self.old_path.to_string_lossy())?;
        fields.serialize_field("new_path", &self.new_path.to_string_lossy())?;
        fields.serialize_field("bytes", &self.bytes)?;
        fields.serialize_field("files", &self.files)?;
        fields.serialize_field("used_copy_fallback", &self.used_copy_fallback)?;
        if let Some(warning) = &self.cleanup_warning {
            fields.serialize_field("cleanup_warning", warning)?;
        }
        fields.serialize_field("generation", &self.generation)?;
        if let Some(current) = &self.current {
            fields.serialize_field("cache", current)?;
        }
        fields.end()
    }
}

/// Failures that can prevent a coordinated disk-cache callback from starting.
#[derive(Debug)]
pub(crate) enum CacheCoordinationError {
    /// Cooperative cancellation was requested, including by a panicking
    /// cancellation callback.
    Cancelled,
    /// Another cache operation held the gate for longer than the bounded wait.
    TimedOut,
    /// The operation gate was poisoned.
    GateUnavailable,
    /// The requested cache is memory-only.
    MemoryCache,
    /// The leased storage generation failed point-in-time filesystem
    /// validation.
    InvalidLayout { source: StorageError },
    /// A registered disk cache did not resolve to an absolute path.
    DiskPathUnavailable,
}

impl fmt::Display for CacheCoordinationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => formatter.write_str("coordinated cache operation cancelled"),
            Self::TimedOut => formatter.write_str("another cache operation did not finish in time"),
            Self::GateUnavailable => {
                formatter.write_str("cache operation coordination is unavailable")
            }
            Self::MemoryCache => formatter.write_str("memory cache has no coordinated disk root"),
            Self::InvalidLayout { .. } => {
                formatter.write_str("cache layout failed filesystem validation")
            }
            Self::DiskPathUnavailable => {
                formatter.write_str("disk cache has no valid storage path")
            }
        }
    }
}

impl Error for CacheCoordinationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidLayout { source } => Some(source),
            Self::Cancelled
            | Self::TimedOut
            | Self::GateUnavailable
            | Self::MemoryCache
            | Self::DiskPathUnavailable => None,
        }
    }
}

/// A coordinated disk-cache failure, preserving callback errors separately
/// from cancellation and registry coordination failures.
#[derive(Debug)]
pub(crate) enum CoordinatedCacheError<E> {
    Coordination(CacheCoordinationError),
    Callback(E),
}

impl<E> fmt::Display for CoordinatedCacheError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Coordination(error) => error.fmt(formatter),
            Self::Callback(_) => formatter.write_str("coordinated cache callback failed"),
        }
    }
}

impl<E> Error for CoordinatedCacheError<E>
where
    E: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Coordination(error) => Some(error),
            Self::Callback(error) => Some(error),
        }
    }
}

const fn cache_kind_key(kind: CacheKind) -> &'static str {
    match kind {
        CacheKind::Memory => "memory",
        CacheKind::Disk => "disk",
    }
}

const fn cache_tier_key(tier: CacheTier) -> &'static str {
    match tier {
        CacheTier::Disposable => "disposable",
        CacheTier::Redownloadable => "redownloadable",
        CacheTier::UserData => "user_data",
    }
}
