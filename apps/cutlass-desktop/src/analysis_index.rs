//! Stable media identity and short-lived access to the desktop moments index.
//!
//! Every API in this module is synchronous and intended only for a background
//! job or worker thread. Media hashing and SQLite must never run on the Slint
//! UI thread or the preview worker.

use std::error::Error;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, Read};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

use cutlass_analysis::moments::{
    MediaContentKey, MomentBatchKey, MomentQuery, MomentReplacementBatch, MomentsIndex,
    MomentsIndexError, MomentsQueryResult,
};
use cutlass_storage::CacheId;
use same_file::Handle as FileIdentity;
use sha2::{Digest, Sha256};

use crate::cache_registry::{CacheCoordinationError, CacheRegistry, CoordinatedCacheError};

const MEDIA_HASH_BUFFER_BYTES: usize = 128 * 1024;

/// A media-identity, cache-coordination, or moments-index failure.
///
/// Display messages are bounded, stable, and never contain media or cache
/// paths. Underlying I/O and SQLite errors remain available through
/// [`Error::source`].
#[derive(Debug)]
pub(crate) enum AnalysisIndexError {
    /// Cooperative cancellation was requested. A panicking callback is also
    /// treated as cancellation.
    Cancelled,
    /// The canonical source target was not a regular file.
    NotRegularFile,
    /// The source mapping, file identity, or file-version snapshot changed.
    SourceChanged,
    /// A media-source filesystem operation failed before a stable token could
    /// be produced.
    MediaIo {
        operation: &'static str,
        source: io::Error,
    },
    /// The cache registry could not establish a coordinated Analysis-root
    /// operation.
    CacheCoordination(CacheCoordinationError),
    /// The short-lived moments index operation failed.
    MomentsIndex(MomentsIndexError),
}

impl fmt::Display for AnalysisIndexError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => formatter.write_str("media analysis operation cancelled"),
            Self::NotRegularFile => formatter.write_str("media source is not a regular file"),
            Self::SourceChanged => {
                formatter.write_str("media source changed during identity validation")
            }
            Self::MediaIo { operation, .. } => {
                write!(formatter, "media source {operation} failed")
            }
            Self::CacheCoordination(error) => error.fmt(formatter),
            Self::MomentsIndex(error) => error.fmt(formatter),
        }
    }
}

impl Error for AnalysisIndexError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::MediaIo { source, .. } => Some(source),
            Self::CacheCoordination(error) => Some(error),
            Self::MomentsIndex(error) => Some(error),
            Self::Cancelled | Self::NotRegularFile | Self::SourceChanged => None,
        }
    }
}

impl From<MomentsIndexError> for AnalysisIndexError {
    fn from(error: MomentsIndexError) -> Self {
        Self::MomentsIndex(error)
    }
}

impl From<CacheCoordinationError> for AnalysisIndexError {
    fn from(error: CacheCoordinationError) -> Self {
        match error {
            CacheCoordinationError::Cancelled => Self::Cancelled,
            error => Self::CacheCoordination(error),
        }
    }
}

/// An opaque point-in-time proof that a regular media file hashed to one
/// content key without an observed mapping, identity, or version change.
///
/// Validation takes `O(file bytes)` time and `O(1)` auxiliary memory. The
/// retained open identity handle and version snapshot permit inexpensive
/// later revalidation, but they do not lock the source. A file can still
/// change immediately after a successful check, and a mutation that is made
/// and fully restored between metadata observations may evade detection on a
/// filesystem with insufficient timestamp fidelity.
pub(crate) struct ValidatedMediaContent {
    content_key: MediaContentKey,
    canonical_path: PathBuf,
    identity: FileIdentity,
    version: FileVersion,
}

impl ValidatedMediaContent {
    /// The SHA-256 media content key.
    pub(crate) const fn content_key(&self) -> MediaContentKey {
        self.content_key
    }

    /// The canonical source path reserved for the future decoder job.
    pub(crate) fn path(&self) -> &Path {
        &self.canonical_path
    }

    /// Recheck the retained object and the canonical path's current target.
    ///
    /// This is a point-in-time check with the same post-success limitation as
    /// token construction. It does not rehash the media bytes.
    #[allow(dead_code)] // For an analyzer that must revalidate mid-job.
    pub(crate) fn verify_unchanged(&self) -> Result<(), AnalysisIndexError> {
        verify_open_identity_version(&self.identity, &self.version)?;
        let remapped = fs::canonicalize(&self.canonical_path)
            .map_err(|_| AnalysisIndexError::SourceChanged)?;
        if remapped != self.canonical_path {
            return Err(AnalysisIndexError::SourceChanged);
        }
        verify_current_path_identity(&self.canonical_path, &self.identity, &self.version)
    }
}

/// Stream a regular media file into SHA-256 and return an opaque stable token.
///
/// The operation is `O(file bytes)` in time and uses a fixed 128 KiB buffer,
/// so auxiliary memory is `O(1)`. The path is canonicalized before opening and
/// its mapping, the opened object identity, and strong available file-version
/// metadata are all revalidated immediately after hashing.
///
/// Cancellation is checked before filesystem access, before and after every
/// blocking read, and before success. A panic from `cancelled` fails closed as
/// [`AnalysisIndexError::Cancelled`].
///
/// A successful result remains a point-in-time observation: the source can
/// change immediately afterward, and a fully restored mutation between
/// metadata snapshots can evade detection on filesystems whose timestamps do
/// not distinguish it.
pub(crate) fn validate_media_content_with_cancel(
    path: &Path,
    cancelled: &dyn Fn() -> bool,
) -> Result<ValidatedMediaContent, AnalysisIndexError> {
    validate_media_content_impl(path, cancelled, |_| {})
}

fn validate_media_content_impl(
    path: &Path,
    cancelled: &dyn Fn() -> bool,
    mut after_chunk: impl FnMut(usize),
) -> Result<ValidatedMediaContent, AnalysisIndexError> {
    check_cancelled(cancelled)?;
    let canonical_path =
        fs::canonicalize(path).map_err(|source| media_io("canonicalization", source))?;
    check_cancelled(cancelled)?;

    let path_metadata =
        fs::metadata(&canonical_path).map_err(|source| media_io("inspection", source))?;
    if !path_metadata.is_file() {
        return Err(AnalysisIndexError::NotRegularFile);
    }
    check_cancelled(cancelled)?;

    let mut file = File::open(&canonical_path).map_err(|source| media_io("open", source))?;
    check_cancelled(cancelled)?;
    let opened_metadata = file
        .metadata()
        .map_err(|source| media_io("opened-file inspection", source))?;
    if !opened_metadata.is_file() {
        return Err(AnalysisIndexError::NotRegularFile);
    }
    let version = FileVersion::from_metadata(&opened_metadata);
    let identity_file = file
        .try_clone()
        .map_err(|source| media_io("identity-handle cloning", source))?;
    let identity = FileIdentity::from_file(identity_file)
        .map_err(|source| media_io("identity capture", source))?;

    let content_key = hash_reader_with_cancel(&mut file, cancelled, &mut after_chunk)?;
    verify_after_hash(path, &canonical_path, &identity, &version)?;
    check_cancelled(cancelled)?;

    Ok(ValidatedMediaContent {
        content_key,
        canonical_path,
        identity,
        version,
    })
}

fn hash_reader_with_cancel(
    reader: &mut impl Read,
    cancelled: &dyn Fn() -> bool,
    mut after_chunk: impl FnMut(usize),
) -> Result<MediaContentKey, AnalysisIndexError> {
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; MEDIA_HASH_BUFFER_BYTES];
    loop {
        check_cancelled(cancelled)?;
        let read_result = reader.read(&mut buffer);
        check_cancelled(cancelled)?;
        let count = match read_result {
            Ok(count) => count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(source) => return Err(media_io("read", source)),
        };
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
        after_chunk(count);
    }

    let digest: [u8; 32] = hasher.finalize().into();
    Ok(MediaContentKey::from_bytes(digest))
}

fn verify_after_hash(
    original_path: &Path,
    canonical_path: &Path,
    expected_identity: &FileIdentity,
    expected_version: &FileVersion,
) -> Result<(), AnalysisIndexError> {
    verify_open_identity_version(expected_identity, expected_version)?;
    let remapped =
        fs::canonicalize(original_path).map_err(|_| AnalysisIndexError::SourceChanged)?;
    if remapped != canonical_path {
        return Err(AnalysisIndexError::SourceChanged);
    }
    verify_current_path_identity(canonical_path, expected_identity, expected_version)
}

fn verify_open_identity_version(
    identity: &FileIdentity,
    expected_version: &FileVersion,
) -> Result<(), AnalysisIndexError> {
    let metadata = identity
        .as_file()
        .metadata()
        .map_err(|_| AnalysisIndexError::SourceChanged)?;
    if !metadata.is_file() || FileVersion::from_metadata(&metadata) != *expected_version {
        return Err(AnalysisIndexError::SourceChanged);
    }
    Ok(())
}

fn verify_current_path_identity(
    canonical_path: &Path,
    expected_identity: &FileIdentity,
    expected_version: &FileVersion,
) -> Result<(), AnalysisIndexError> {
    let current_file = File::open(canonical_path).map_err(|_| AnalysisIndexError::SourceChanged)?;
    let current_identity =
        FileIdentity::from_file(current_file).map_err(|_| AnalysisIndexError::SourceChanged)?;
    let current_metadata = current_identity
        .as_file()
        .metadata()
        .map_err(|_| AnalysisIndexError::SourceChanged)?;
    if !current_metadata.is_file()
        || &current_identity != expected_identity
        || FileVersion::from_metadata(&current_metadata) != *expected_version
    {
        return Err(AnalysisIndexError::SourceChanged);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FileVersion {
    length: u64,
    #[cfg(unix)]
    modified_seconds: i64,
    #[cfg(unix)]
    modified_nanoseconds: i64,
    #[cfg(unix)]
    changed_seconds: i64,
    #[cfg(unix)]
    changed_nanoseconds: i64,
    #[cfg(windows)]
    creation_time: u64,
    #[cfg(windows)]
    last_write_time: u64,
    #[cfg(not(any(unix, windows)))]
    modified: Option<std::time::SystemTime>,
}

impl FileVersion {
    fn from_metadata(metadata: &fs::Metadata) -> Self {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt as _;

            Self {
                length: metadata.len(),
                modified_seconds: metadata.mtime(),
                modified_nanoseconds: metadata.mtime_nsec(),
                changed_seconds: metadata.ctime(),
                changed_nanoseconds: metadata.ctime_nsec(),
            }
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt as _;

            Self {
                length: metadata.len(),
                creation_time: metadata.creation_time(),
                last_write_time: metadata.last_write_time(),
            }
        }
        #[cfg(not(any(unix, windows)))]
        {
            Self {
                length: metadata.len(),
                modified: metadata.modified().ok(),
            }
        }
    }
}

fn check_cancelled(cancelled: &dyn Fn() -> bool) -> Result<(), AnalysisIndexError> {
    let requested = catch_unwind(AssertUnwindSafe(cancelled)).unwrap_or(true);
    if requested {
        Err(AnalysisIndexError::Cancelled)
    } else {
        Ok(())
    }
}

fn media_io(operation: &'static str, source: io::Error) -> AnalysisIndexError {
    AnalysisIndexError::MediaIo { operation, source }
}

/// Cheap cloneable access to the moments database in the active Analysis
/// cache generation.
///
/// Each call acquires the cache registry's operation gate and storage-layout
/// lease, opens one [`MomentsIndex`] inside the callback, and drops the index
/// before either guard is released. Methods are synchronous and must run only
/// on a job or worker thread.
#[derive(Clone)]
pub(crate) struct AnalysisIndexService {
    registry: CacheRegistry,
}

impl AnalysisIndexService {
    pub(crate) fn new(registry: CacheRegistry) -> Self {
        Self { registry }
    }

    pub(crate) fn replace_batch(
        &self,
        batch: &MomentReplacementBatch,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<(), AnalysisIndexError> {
        self.with_analysis_root(cancelled, |root| {
            replace_batch_in_root(root, batch, cancelled)
        })
    }

    pub(crate) fn batch_exists(
        &self,
        key: &MomentBatchKey,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<bool, AnalysisIndexError> {
        self.with_analysis_root(cancelled, |root| batch_exists_in_root(root, key, cancelled))
    }

    pub(crate) fn query_with_limit(
        &self,
        query: &MomentQuery,
        limit: usize,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<MomentsQueryResult, AnalysisIndexError> {
        self.with_analysis_root(cancelled, |root| {
            query_in_root(root, query, limit, cancelled)
        })
    }

    /// Drop one analyzer's results for one asset — the path a "re-analyze from
    /// scratch" action takes.
    #[allow(dead_code)] // No UI invalidates a single batch yet.
    pub(crate) fn delete_batch(
        &self,
        key: &MomentBatchKey,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<bool, AnalysisIndexError> {
        self.with_analysis_root(cancelled, |root| delete_batch_in_root(root, key, cancelled))
    }

    fn with_analysis_root<T>(
        &self,
        cancelled: &dyn Fn() -> bool,
        operation: impl FnOnce(&Path) -> Result<T, AnalysisIndexError>,
    ) -> Result<T, AnalysisIndexError> {
        match self
            .registry
            .with_disk_cache_root(CacheId::Analysis, cancelled, operation)
        {
            Ok(value) => Ok(value),
            Err(CoordinatedCacheError::Coordination(error)) => Err(error.into()),
            Err(CoordinatedCacheError::Callback(error)) => Err(error),
        }
    }
}

fn with_index_in_root<T>(
    root: &Path,
    cancelled: &dyn Fn() -> bool,
    operation: impl FnOnce(&MomentsIndex) -> Result<T, AnalysisIndexError>,
) -> Result<T, AnalysisIndexError> {
    check_cancelled(cancelled)?;
    let index = MomentsIndex::open_in_cache_root(root)?;
    let result = operation(&index);
    drop(index);
    result
}

fn replace_batch_in_root(
    root: &Path,
    batch: &MomentReplacementBatch,
    cancelled: &dyn Fn() -> bool,
) -> Result<(), AnalysisIndexError> {
    with_index_in_root(root, cancelled, |index| {
        check_cancelled(cancelled)?;
        index.replace_batch(batch).map_err(Into::into)
    })
}

fn batch_exists_in_root(
    root: &Path,
    key: &MomentBatchKey,
    cancelled: &dyn Fn() -> bool,
) -> Result<bool, AnalysisIndexError> {
    with_index_in_root(root, cancelled, |index| {
        check_cancelled(cancelled)?;
        let exists = index.batch_exists(key)?;
        check_cancelled(cancelled)?;
        Ok(exists)
    })
}

fn query_in_root(
    root: &Path,
    query: &MomentQuery,
    limit: usize,
    cancelled: &dyn Fn() -> bool,
) -> Result<MomentsQueryResult, AnalysisIndexError> {
    with_index_in_root(root, cancelled, |index| {
        check_cancelled(cancelled)?;
        let result = index.query_with_limit(query, limit)?;
        check_cancelled(cancelled)?;
        Ok(result)
    })
}

fn delete_batch_in_root(
    root: &Path,
    key: &MomentBatchKey,
    cancelled: &dyn Fn() -> bool,
) -> Result<bool, AnalysisIndexError> {
    with_index_in_root(root, cancelled, |index| {
        check_cancelled(cancelled)?;
        index.delete_batch(key).map_err(Into::into)
    })
}

#[cfg(test)]
mod tests {
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use cutlass_analysis::TimeSpan;
    use cutlass_analysis::moments::{
        AnalyzerIdentity, AnalyzerVersion, MEDIA_CONTENT_KEY_BYTES,
        MOMENTS_INDEX_DATABASE_FILENAME, MomentConfidence, MomentKind, MomentPayload, MomentRecord,
    };

    use super::*;

    const HELLO_SHA256: &str = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";

    #[test]
    fn hashes_known_vector_and_equal_bytes_to_equal_content_keys() {
        let temporary = tempfile::tempdir().unwrap();
        let first_path = temporary.path().join("first.mp4");
        let second_path = temporary.path().join("second.mov");
        fs::write(&first_path, b"hello").unwrap();
        fs::write(&second_path, b"hello").unwrap();

        let first = validate_media_content_with_cancel(&first_path, &|| false).unwrap();
        let second = validate_media_content_with_cancel(&second_path, &|| false).unwrap();

        assert_eq!(first.content_key().to_string(), HELLO_SHA256);
        assert_eq!(first.content_key(), second.content_key());
        assert_eq!(first.path(), fs::canonicalize(&first_path).unwrap());
    }

    #[test]
    fn streaming_hash_uses_multiple_bounded_reads() {
        let bytes = vec![0x5a; MEDIA_HASH_BUFFER_BYTES * 3 + 17];
        let mut reader = ObservingReader::new(bytes);
        let key = hash_reader_with_cancel(&mut reader, &|| false, |_| {}).unwrap();

        assert_ne!(
            key,
            MediaContentKey::from_bytes([0_u8; MEDIA_CONTENT_KEY_BYTES])
        );
        assert!(reader.read_calls >= 4);
        assert_eq!(reader.maximum_request, MEDIA_HASH_BUFFER_BYTES);
    }

    #[test]
    fn cancellation_before_access_and_during_read_is_typed() {
        let missing = Path::new("distinctive-path-that-must-not-be-accessed.mp4");
        assert!(matches!(
            validate_media_content_with_cancel(missing, &|| true),
            Err(AnalysisIndexError::Cancelled)
        ));

        let cancelled = Arc::new(AtomicBool::new(false));
        let mut reader = CancellingReader {
            cancelled: Arc::clone(&cancelled),
        };
        let error =
            hash_reader_with_cancel(&mut reader, &|| cancelled.load(Ordering::Acquire), |_| {})
                .unwrap_err();
        assert!(matches!(error, AnalysisIndexError::Cancelled));
    }

    #[test]
    fn panicking_cancellation_callback_fails_closed() {
        let error = validate_media_content_with_cancel(
            Path::new("distinctive-path-that-must-not-be-accessed.mp4"),
            &|| panic!("injected cancellation panic"),
        );
        let error = expect_validation_error(error);
        assert!(matches!(error, AnalysisIndexError::Cancelled));
    }

    #[test]
    fn directories_are_not_media_files() {
        let temporary = tempfile::tempdir().unwrap();
        let error = expect_validation_error(validate_media_content_with_cancel(
            temporary.path(),
            &|| false,
        ));
        assert!(matches!(error, AnalysisIndexError::NotRegularFile));
    }

    #[test]
    fn in_place_mutation_during_hash_is_rejected() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("source.mp4");
        fs::write(&path, vec![0x2a; MEDIA_HASH_BUFFER_BYTES * 2]).unwrap();
        let mutation_path = path.clone();
        let mut mutated = false;

        let result = validate_media_content_impl(&path, &|| false, |_| {
            if !mutated {
                mutated = true;
                let mut file = OpenOptions::new()
                    .append(true)
                    .open(&mutation_path)
                    .unwrap();
                file.write_all(b"!").unwrap();
                file.flush().unwrap();
            }
        });
        let error = expect_validation_error(result);

        assert!(matches!(error, AnalysisIndexError::SourceChanged));
    }

    #[cfg(unix)]
    #[test]
    fn same_path_replacement_during_hash_is_rejected() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("source.mp4");
        let displaced = temporary.path().join("source-original.mp4");
        fs::write(&path, vec![0x31; MEDIA_HASH_BUFFER_BYTES * 2]).unwrap();
        let replacement_path = path.clone();
        let mut replaced = false;

        let result = validate_media_content_impl(&path, &|| false, |_| {
            if !replaced {
                replaced = true;
                fs::rename(&replacement_path, &displaced).unwrap();
                fs::write(&replacement_path, vec![0x32; MEDIA_HASH_BUFFER_BYTES * 2]).unwrap();
            }
        });
        let error = expect_validation_error(result);

        assert!(matches!(error, AnalysisIndexError::SourceChanged));
    }

    #[test]
    fn display_errors_never_expose_media_paths() {
        let temporary = tempfile::tempdir().unwrap();
        let secret = "media-secret-7f43a90c";
        let path = temporary.path().join(secret).join("missing.mp4");
        let error = expect_validation_error(validate_media_content_with_cancel(&path, &|| false));
        let display = error.to_string();

        assert!(!display.contains(secret), "{display}");
        assert!(display.len() < 128);
    }

    #[test]
    fn token_revalidation_detects_later_modification() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("source.mp4");
        fs::write(&path, b"stable bytes").unwrap();
        let validated = validate_media_content_with_cancel(&path, &|| false).unwrap();

        validated.verify_unchanged().unwrap();
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(b"!").unwrap();
        file.flush().unwrap();
        assert!(matches!(
            validated.verify_unchanged(),
            Err(AnalysisIndexError::SourceChanged)
        ));
    }

    #[cfg(unix)]
    #[test]
    fn token_revalidation_detects_later_replacement() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("source.mp4");
        let displaced = temporary.path().join("source-original.mp4");
        fs::write(&path, b"stable bytes").unwrap();
        let validated = validate_media_content_with_cancel(&path, &|| false).unwrap();

        fs::rename(&path, displaced).unwrap();
        fs::write(&path, b"stable bytes").unwrap();
        assert!(matches!(
            validated.verify_unchanged(),
            Err(AnalysisIndexError::SourceChanged)
        ));
    }

    #[test]
    fn index_root_helpers_round_trip_and_release_the_database() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("analysis");
        let key = MediaContentKey::from_bytes([0x17; MEDIA_CONTENT_KEY_BYTES]);
        let analyzer = AnalyzerIdentity::new("cutlass.test").unwrap();
        let version = AnalyzerVersion::new(1).unwrap();
        let batch_key = MomentBatchKey::new(key, analyzer.clone(), version);
        let record = MomentRecord::new(
            key,
            TimeSpan::new(1.25, 2.5).unwrap(),
            MomentKind::VISION_MOMENT,
            MomentConfidence::new(0.75).unwrap(),
            analyzer,
            version,
            Some(MomentPayload::new("fixture").unwrap()),
        );
        let batch = MomentReplacementBatch::new(batch_key.clone(), vec![record.clone()]).unwrap();

        replace_batch_in_root(&root, &batch, &|| false).unwrap();
        assert!(
            root.join(MOMENTS_INDEX_DATABASE_FILENAME).is_file(),
            "index must use the standard cache-root filename"
        );
        assert!(batch_exists_in_root(&root, &batch_key, &|| false).unwrap());
        let result = query_in_root(&root, &MomentQuery::new(key), 16, &|| false).unwrap();
        assert_eq!(result.records(), &[record]);
        assert!(delete_batch_in_root(&root, &batch_key, &|| false).unwrap());
        assert!(!batch_exists_in_root(&root, &batch_key, &|| false).unwrap());

        let moved = temporary.path().join("analysis-moved");
        fs::rename(&root, &moved)
            .expect("every short-lived SQLite connection must be dropped before return");
        assert!(moved.join(MOMENTS_INDEX_DATABASE_FILENAME).is_file());
    }

    #[test]
    fn pre_cancelled_index_operation_does_not_create_database_root() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("analysis");
        let key = MediaContentKey::from_bytes([0x23; MEDIA_CONTENT_KEY_BYTES]);

        let error = query_in_root(&root, &MomentQuery::new(key), 16, &|| true).unwrap_err();

        assert!(matches!(error, AnalysisIndexError::Cancelled));
        assert!(!root.exists());
    }

    fn expect_validation_error(
        result: Result<ValidatedMediaContent, AnalysisIndexError>,
    ) -> AnalysisIndexError {
        match result {
            Ok(_) => panic!("media validation unexpectedly succeeded"),
            Err(error) => error,
        }
    }

    struct ObservingReader {
        bytes: Vec<u8>,
        offset: usize,
        read_calls: usize,
        maximum_request: usize,
    }

    impl ObservingReader {
        fn new(bytes: Vec<u8>) -> Self {
            Self {
                bytes,
                offset: 0,
                read_calls: 0,
                maximum_request: 0,
            }
        }
    }

    impl Read for ObservingReader {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            self.read_calls += 1;
            self.maximum_request = self.maximum_request.max(buffer.len());
            let remaining = self.bytes.len().saturating_sub(self.offset);
            let count = remaining.min(buffer.len()).min(31 * 1024);
            if count == 0 {
                return Ok(0);
            }
            buffer[..count].copy_from_slice(&self.bytes[self.offset..self.offset + count]);
            self.offset += count;
            Ok(count)
        }
    }

    struct CancellingReader {
        cancelled: Arc<AtomicBool>,
    }

    impl Read for CancellingReader {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            buffer[..5].copy_from_slice(b"hello");
            self.cancelled.store(true, Ordering::Release);
            Ok(5)
        }
    }
}
