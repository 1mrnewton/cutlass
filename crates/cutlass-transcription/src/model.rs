use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, TryLockError, Weak};
use std::time::Duration;

use cap_std::ambient_authority;
use cap_std::fs::{
    Dir as CapabilityDir, File as CapabilityFile, OpenOptions as CapabilityOpenOptions,
};
use same_file::Handle as FileIdentity;
use sha2::{Digest, Sha256};
use thiserror::Error;

const BASE_EN_SPEC: ModelSpec = ModelSpec::new(
    "base.en",
    "ggml-base.en.bin",
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
    147_964_211,
    "a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002",
    true,
    "about 148 MB on disk and about 388 MB RAM during inference",
);

const WHISPER_MODELS: &[WhisperModel] = &[WhisperModel::BaseEn];
const DOWNLOAD_BUFFER_BYTES: usize = 64 * 1024;
const TEMP_FILE_ATTEMPTS: u64 = 256;
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_TOTAL_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const MODEL_LOCK_RETRY_DELAY: Duration = Duration::from_millis(5);

static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);
static MODEL_LOCKS: OnceLock<Mutex<ModelLockMap>> = OnceLock::new();

type ModelLockMap = HashMap<PathBuf, Weak<Mutex<()>>>;

/// A built-in Whisper model known to Cutlass.
///
/// This enum, rather than a caller-provided filename, is used by every public
/// [`ModelManager`] file operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum WhisperModel {
    /// The official English-only Whisper base model.
    BaseEn,
}

impl WhisperModel {
    /// Returns all models in stable catalog order.
    #[must_use]
    pub const fn catalog() -> &'static [Self] {
        WHISPER_MODELS
    }

    /// Returns the immutable catalog specification for this model.
    #[must_use]
    pub const fn spec(self) -> &'static ModelSpec {
        match self {
            Self::BaseEn => &BASE_EN_SPEC,
        }
    }
}

/// Immutable metadata for one verified model artifact.
///
/// Construction is intentionally private. Public model-manager operations can
/// therefore use only filenames compiled into this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSpec {
    id: &'static str,
    filename: &'static str,
    url: &'static str,
    exact_bytes: u64,
    sha256: &'static str,
    english_only: bool,
    resource_description: &'static str,
}

impl ModelSpec {
    const fn new(
        id: &'static str,
        filename: &'static str,
        url: &'static str,
        exact_bytes: u64,
        sha256: &'static str,
        english_only: bool,
        resource_description: &'static str,
    ) -> Self {
        Self {
            id,
            filename,
            url,
            exact_bytes,
            sha256,
            english_only,
            resource_description,
        }
    }

    /// Returns the stable application-facing model identifier.
    #[must_use]
    pub const fn id(&self) -> &'static str {
        self.id
    }

    /// Returns the fixed local filename.
    #[must_use]
    pub const fn filename(&self) -> &'static str {
        self.filename
    }

    /// Returns the official HTTPS artifact URL.
    #[must_use]
    pub const fn url(&self) -> &'static str {
        self.url
    }

    /// Returns the exact accepted artifact length.
    #[must_use]
    pub const fn exact_bytes(&self) -> u64 {
        self.exact_bytes
    }

    /// Returns the lowercase SHA-256 digest in hexadecimal.
    #[must_use]
    pub const fn sha256(&self) -> &'static str {
        self.sha256
    }

    /// Returns whether this model can recognize only English speech.
    #[must_use]
    pub const fn is_english_only(&self) -> bool {
        self.english_only
    }

    /// Returns an approximate disk and inference-memory description.
    #[must_use]
    pub const fn resource_description(&self) -> &'static str {
        self.resource_description
    }
}

/// A readable stream returned by a [`ModelDownloader`].
pub type DownloadReader = Box<dyn Read + Send + 'static>;

/// An injectable source for model bytes.
///
/// Tests and embedding applications can provide deterministic readers without
/// network access. Integrity and length enforcement remain the model manager's
/// responsibility.
pub trait ModelDownloader: Send + Sync {
    /// Opens a streaming response for `spec`.
    ///
    /// The returned stream is consumed at most through the catalogued exact
    /// byte count plus one detecting read.
    fn download(&self, spec: &ModelSpec) -> Result<DownloadReader, DownloadError>;
}

impl<F> ModelDownloader for F
where
    F: Fn(&ModelSpec) -> Result<DownloadReader, DownloadError> + Send + Sync,
{
    fn download(&self, spec: &ModelSpec) -> Result<DownloadReader, DownloadError> {
        self(spec)
    }
}

/// A failure to establish a model download stream.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DownloadError {
    /// The server returned a non-success HTTP status.
    #[error("model server returned HTTP status {status}")]
    HttpStatus {
        /// The rejected HTTP status code.
        status: u16,
    },
    /// The HTTP transport failed before a usable stream was returned.
    #[error("model download transport failed: {message}")]
    Transport {
        /// A transport diagnostic without response content.
        message: String,
    },
    /// One or more configured timeouts were zero.
    #[error("model download timeouts must all be greater than zero")]
    InvalidTimeout,
    /// A custom downloader rejected the request.
    #[error("model downloader failed: {message}")]
    Custom {
        /// A custom downloader diagnostic.
        message: String,
    },
}

impl DownloadError {
    /// Creates an error for an injected downloader.
    #[must_use]
    pub fn custom(message: impl Into<String>) -> Self {
        Self::Custom {
            message: message.into(),
        }
    }
}

/// Blocking HTTPS model downloader backed by `ureq`.
///
/// The default uses a 15-second connect timeout, a 30-second read timeout, and
/// a 30-minute whole-request deadline. Redirects are handled by `ureq`; the
/// final response must be a 2xx status.
#[derive(Debug, Clone)]
pub struct HttpDownloader {
    agent: ureq::Agent,
}

impl HttpDownloader {
    /// Creates a downloader with explicit non-zero timeout bounds.
    ///
    /// # Errors
    ///
    /// Returns [`DownloadError::InvalidTimeout`] if any timeout is zero.
    pub fn with_timeouts(
        connect_timeout: Duration,
        read_timeout: Duration,
        total_timeout: Duration,
    ) -> Result<Self, DownloadError> {
        if connect_timeout.is_zero() || read_timeout.is_zero() || total_timeout.is_zero() {
            return Err(DownloadError::InvalidTimeout);
        }

        let agent = ureq::AgentBuilder::new()
            .https_only(true)
            .timeout_connect(connect_timeout)
            .timeout_read(read_timeout)
            .timeout(total_timeout)
            .build();
        Ok(Self { agent })
    }
}

impl Default for HttpDownloader {
    fn default() -> Self {
        Self::with_timeouts(
            DEFAULT_CONNECT_TIMEOUT,
            DEFAULT_READ_TIMEOUT,
            DEFAULT_TOTAL_TIMEOUT,
        )
        .expect("default model download timeouts are non-zero")
    }
}

impl ModelDownloader for HttpDownloader {
    fn download(&self, spec: &ModelSpec) -> Result<DownloadReader, DownloadError> {
        let response = match self
            .agent
            .get(spec.url())
            .set("Accept", "application/octet-stream")
            .set("Accept-Encoding", "identity")
            .set(
                "User-Agent",
                concat!("cutlass-transcription/", env!("CARGO_PKG_VERSION")),
            )
            .call()
        {
            Ok(response) => response,
            Err(ureq::Error::Status(status, _)) => {
                return Err(DownloadError::HttpStatus { status });
            }
            Err(ureq::Error::Transport(error)) => {
                return Err(DownloadError::Transport {
                    message: error.to_string(),
                });
            }
        };

        if !(200..300).contains(&response.status()) {
            return Err(DownloadError::HttpStatus {
                status: response.status(),
            });
        }
        Ok(response.into_reader())
    }
}

/// Why a regular model file failed exact integrity verification.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ModelIntegrityError {
    /// The file or stream length differed from the catalog.
    #[error("expected exactly {expected} bytes, found {actual}")]
    WrongSize {
        /// Exact catalogued bytes.
        expected: u64,
        /// Observed bytes.
        actual: u64,
    },
    /// The SHA-256 digest differed from the catalog.
    #[error("expected SHA-256 {expected}, found {actual}")]
    ChecksumMismatch {
        /// Catalogued lowercase SHA-256.
        expected: String,
        /// Observed lowercase SHA-256.
        actual: String,
    },
}

/// Current integrity state of one catalogued model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelStatus {
    /// No model file exists.
    Missing,
    /// A regular file passed exact size and SHA-256 verification.
    Ready {
        /// Verified model path.
        path: PathBuf,
    },
    /// A regular file exists but failed integrity verification.
    Invalid {
        /// Invalid model path.
        path: PathBuf,
        /// Exact verification failure.
        reason: ModelIntegrityError,
    },
}

/// A model catalog, filesystem, download, or integrity failure.
#[derive(Debug, Error)]
pub enum ModelManagerError {
    /// Cooperative model installation cancellation was requested.
    #[error("model installation cancelled")]
    Cancelled,
    /// The configured model root was relative.
    #[error("model root must be absolute: `{root}`")]
    RootNotAbsolute {
        /// Rejected root.
        root: PathBuf,
    },
    /// The configured model root was a filesystem root.
    #[error("model root must not be a filesystem root: `{root}`")]
    FilesystemRoot {
        /// Rejected root.
        root: PathBuf,
    },
    /// The root contained lexical traversal.
    #[error("model root must not contain `.` or `..` traversal: `{root}`")]
    RootTraversal {
        /// Rejected root.
        root: PathBuf,
    },
    /// A compiled catalog entry violated path or digest invariants.
    #[error("invalid model catalog entry `{model_id}`: {detail}")]
    InvalidCatalogEntry {
        /// Stable catalog identifier.
        model_id: &'static str,
        /// Rejected invariant.
        detail: &'static str,
    },
    /// The model root was a link, Windows reparse point, or non-directory.
    #[error("unsafe model root `{root}`: {detail}")]
    UnsafeRoot {
        /// Rejected root path.
        root: PathBuf,
        /// Rejected filesystem type.
        detail: &'static str,
    },
    /// The lexical root stopped naming the pinned physical directory.
    #[error(
        "model root mapping changed while operating on `{root}`; expected pinned directory `{pinned_root}`"
    )]
    RootMappingChanged {
        /// Configured lexical root.
        root: PathBuf,
        /// Canonical directory pinned at operation start.
        pinned_root: PathBuf,
    },
    /// A known model path was a link, Windows reparse point, or non-regular file.
    #[error("unsafe model file `{path}`: {detail}")]
    UnsafeModelFile {
        /// Rejected known model path.
        path: PathBuf,
        /// Rejected filesystem type.
        detail: &'static str,
    },
    /// A model or temporary file name changed objects during an operation.
    #[error("{object} changed during the operation: `{path}`")]
    FilesystemObjectChanged {
        /// Path used for diagnostics.
        path: PathBuf,
        /// Kind of filesystem object that changed.
        object: &'static str,
    },
    /// A required model file did not exist.
    #[error("model file is missing: `{path}`")]
    MissingModel {
        /// Expected known model path.
        path: PathBuf,
    },
    /// A model file or completed download failed integrity verification.
    #[error("model integrity check failed for `{path}`: {reason}")]
    Integrity {
        /// Known model path.
        path: PathBuf,
        /// Exact verification failure.
        #[source]
        reason: ModelIntegrityError,
    },
    /// A stream exceeded the catalogued byte maximum.
    #[error("model download exceeded {expected} bytes after observing {observed}")]
    DownloadTooLarge {
        /// Exact catalogued maximum.
        expected: u64,
        /// Bytes observed when overflow was detected.
        observed: u64,
    },
    /// A downloader could not produce a stream.
    #[error(transparent)]
    Download(#[from] DownloadError),
    /// A bounded filesystem or stream operation failed.
    #[error("failed to {operation} `{path}`: {source}")]
    Io {
        /// Operation being attempted.
        operation: &'static str,
        /// Relevant filesystem path.
        path: PathBuf,
        /// Underlying I/O failure.
        #[source]
        source: io::Error,
    },
    /// Installation completed, but the containing directory could not be synced.
    #[error(
        "verified model was installed at `{path}`, but durability confirmation for its containing directory failed; retrying ensure is safe: {source}"
    )]
    InstalledButDirectorySyncFailed {
        /// Installed verified model path.
        path: PathBuf,
        /// Directory synchronization failure.
        #[source]
        source: io::Error,
    },
    /// Unique temporary filenames could not be allocated.
    #[error("could not allocate a unique temporary model file in `{root}`")]
    TemporaryNameExhausted {
        /// Model directory containing collisions.
        root: PathBuf,
    },
}

/// Safe, transactional storage for built-in Whisper models.
///
/// Every disk operation pins a canonical root directory handle, performs
/// model I/O relative to that handle, and revalidates that the configured root
/// still names the pinned directory before reporting success or mutating a
/// known model name.
#[derive(Debug, Clone)]
pub struct ModelManager {
    root: PathBuf,
}

impl ModelManager {
    /// Validates and stores an absolute, non-root model directory.
    ///
    /// This constructor performs lexical validation only and does not create
    /// the directory. The final root itself is rejected later if it is a
    /// symlink or non-directory object.
    ///
    /// # Errors
    ///
    /// Returns [`ModelManagerError`] for a relative path, filesystem root, or
    /// explicit lexical traversal.
    pub fn new(root: impl AsRef<Path>) -> Result<Self, ModelManagerError> {
        Ok(Self {
            root: validate_root(root.as_ref())?,
        })
    }

    /// Returns the validated model directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Returns the fixed path for a built-in model without touching the disk.
    ///
    /// # Errors
    ///
    /// Returns an error only if a compiled catalog invariant is invalid.
    pub fn model_path(&self, model: WhisperModel) -> Result<PathBuf, ModelManagerError> {
        self.path_for_spec(model.spec())
    }

    /// Verifies the known file, returning missing, ready, or invalid status.
    ///
    /// Links, Windows reparse points, and non-regular model files are rejected
    /// as errors rather than represented as ordinary corruption.
    ///
    /// # Errors
    ///
    /// Returns [`ModelManagerError`] for unsafe filesystem objects or I/O
    /// failures.
    pub fn status(&self, model: WhisperModel) -> Result<ModelStatus, ModelManagerError> {
        self.status_spec(model.spec())
    }

    /// Requires exact size and SHA-256 verification and returns the model path.
    ///
    /// # Errors
    ///
    /// Returns [`ModelManagerError::MissingModel`], an integrity error, an
    /// unsafe-file error, or an I/O error.
    pub fn verify(&self, model: WhisperModel) -> Result<PathBuf, ModelManagerError> {
        self.verify_spec(model.spec())
    }

    /// Ensures a verified model using the bounded default HTTPS downloader.
    ///
    /// # Errors
    ///
    /// Returns [`ModelManagerError`] when verification, downloading, durable
    /// writing, or atomic installation fails.
    pub fn ensure(&self, model: WhisperModel) -> Result<PathBuf, ModelManagerError> {
        self.ensure_with(model, &HttpDownloader::default())
    }

    /// Ensures a verified model using an injected streaming downloader.
    ///
    /// Concurrent calls for the same canonical model path are serialized
    /// within the process, including calls through lexical directory aliases.
    /// Existing files are reused only after full verification.
    ///
    /// # Errors
    ///
    /// Returns [`ModelManagerError`] when verification, downloading, durable
    /// writing, or atomic installation fails.
    pub fn ensure_with(
        &self,
        model: WhisperModel,
        downloader: &dyn ModelDownloader,
    ) -> Result<PathBuf, ModelManagerError> {
        self.ensure_with_cancellation(model, downloader, &never_cancelled)
    }

    /// Ensures a verified model with cooperative cancellation.
    ///
    /// `cancelled` is borrowed for this call, so it may directly borrow a job
    /// context. It can be called many times and should be inexpensive. A panic
    /// from the callback is treated as a cancellation request.
    ///
    /// Cancellation is checked around root setup, lock acquisition, existing
    /// model verification, stream establishment, each download read, and
    /// durable temporary-file writes. A single blocking transport or reader
    /// call, and the existing-model hash pass, cannot be interrupted midway.
    ///
    /// The final cancellation check is used as this standalone API's commit
    /// decision. Once it passes, cancellation is no longer consulted:
    /// replacement, durability confirmation, and root-mapping checks finish
    /// and return their actual result. An outer job that must atomically
    /// coordinate cancellation with publication should instead use
    /// [`Self::ensure_with_cancellation_and_commit`] and bind its commit
    /// callback to the job owner's atomic commit handshake.
    ///
    /// # Errors
    ///
    /// Returns [`ModelManagerError::Cancelled`] when cancellation is observed
    /// before commit. Other failures retain the same meaning as
    /// [`Self::ensure_with`].
    pub fn ensure_with_cancellation(
        &self,
        model: WhisperModel,
        downloader: &dyn ModelDownloader,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<PathBuf, ModelManagerError> {
        let begin_commit = || check_cancelled(cancelled).is_ok();
        self.ensure_with_cancellation_and_commit(model, downloader, cancelled, &begin_commit)
    }

    /// Ensures a verified model with cancellation and an atomic commit handshake.
    ///
    /// Both callbacks are borrowed for this call and may directly borrow an
    /// outer job context. `cancelled` has the same cooperative semantics as in
    /// [`Self::ensure_with_cancellation`]: it may be called many times before
    /// publication and a panic is treated as cancellation.
    ///
    /// `begin_commit` is not called when a verified existing model is reused.
    /// For a download or replacement, it is called exactly once, after the
    /// complete stream has passed exact length and SHA-256 verification, the
    /// temporary file has been flushed and synced, temporary-file identity and
    /// replacement-target safety have been checked, and the configured root
    /// mapping has been revalidated. The call occurs immediately before the
    /// first target removal or rename.
    ///
    /// Returning `false`, or panicking, fails closed with
    /// [`ModelManagerError::Cancelled`]. The temporary file is cleaned up and
    /// an existing target is left unchanged. Returning `true` is the point of
    /// no return: neither callback is consulted again, and replacement,
    /// directory durability, and final root-mapping checks return their actual
    /// result.
    ///
    /// An outer job manager can remove the cancellation/publication race by
    /// making `begin_commit` perform its atomic transition from cancellable to
    /// committing (for example, by calling its `try_begin_commit` operation).
    ///
    /// # Errors
    ///
    /// Returns [`ModelManagerError::Cancelled`] when cancellation is observed
    /// before commit or the commit callback rejects or panics. Other failures
    /// retain the same meaning as [`Self::ensure_with`].
    pub fn ensure_with_cancellation_and_commit(
        &self,
        model: WhisperModel,
        downloader: &dyn ModelDownloader,
        cancelled: &dyn Fn() -> bool,
        begin_commit: &dyn Fn() -> bool,
    ) -> Result<PathBuf, ModelManagerError> {
        self.ensure_spec_with_cancellation_and_commit(
            model.spec(),
            downloader,
            cancelled,
            begin_commit,
        )
    }

    /// Removes only the known regular model file.
    ///
    /// Returns `true` when a file was removed and `false` when it was already
    /// absent. Directories are never removed. Link and Windows reparse-point
    /// model files are rejected and left untouched.
    ///
    /// # Errors
    ///
    /// Returns [`ModelManagerError`] for unsafe filesystem objects or I/O
    /// failures.
    pub fn remove(&self, model: WhisperModel) -> Result<bool, ModelManagerError> {
        self.remove_spec(model.spec())
    }

    fn status_spec(&self, spec: &ModelSpec) -> Result<ModelStatus, ModelManagerError> {
        validate_spec(spec)?;
        let Some(snapshot) = self.capture_existing_root()? else {
            return Ok(ModelStatus::Missing);
        };
        let physical_path = snapshot.physical_model_path(spec);
        let lock = model_lock(&physical_path);
        let _guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        snapshot.verify_mapping()?;
        self.inspect_path(spec, &snapshot)
    }

    fn verify_spec(&self, spec: &ModelSpec) -> Result<PathBuf, ModelManagerError> {
        match self.status_spec(spec)? {
            ModelStatus::Missing => Err(ModelManagerError::MissingModel {
                path: self.path_for_spec(spec)?,
            }),
            ModelStatus::Ready { path } => Ok(path),
            ModelStatus::Invalid { path, reason } => {
                Err(ModelManagerError::Integrity { path, reason })
            }
        }
    }

    #[cfg(test)]
    fn ensure_spec(
        &self,
        spec: &ModelSpec,
        downloader: &dyn ModelDownloader,
    ) -> Result<PathBuf, ModelManagerError> {
        self.ensure_spec_with_cancellation(spec, downloader, &never_cancelled)
    }

    #[cfg(test)]
    fn ensure_spec_with_cancellation(
        &self,
        spec: &ModelSpec,
        downloader: &dyn ModelDownloader,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<PathBuf, ModelManagerError> {
        let begin_commit = || check_cancelled(cancelled).is_ok();
        self.ensure_spec_with_cancellation_and_commit(spec, downloader, cancelled, &begin_commit)
    }

    fn ensure_spec_with_cancellation_and_commit(
        &self,
        spec: &ModelSpec,
        downloader: &dyn ModelDownloader,
        cancelled: &dyn Fn() -> bool,
        begin_commit: &dyn Fn() -> bool,
    ) -> Result<PathBuf, ModelManagerError> {
        validate_spec(spec)?;
        check_cancelled(cancelled)?;
        let snapshot = self.create_and_capture_root()?;
        let physical_path = snapshot.physical_model_path(spec);
        let reported_path = snapshot.configured_model_path(spec);
        let lock = model_lock(&physical_path);
        let _guard = lock_model_with_cancellation(lock.as_ref(), cancelled)?;
        snapshot.verify_mapping()?;

        check_cancelled(cancelled)?;
        let status = self.inspect_path(spec, &snapshot);
        check_cancelled(cancelled)?;
        if matches!(status?, ModelStatus::Ready { .. }) {
            return Ok(reported_path);
        }

        let mut temporary = TemporaryDownload::create(&snapshot, spec.filename())?;
        let temporary_path = temporary.path.clone();
        check_cancelled(cancelled)?;
        let download = downloader.download(spec);
        check_cancelled(cancelled)?;
        let mut reader = download?;
        let mut hasher = Sha256::new();
        let mut total = 0_u64;
        let mut buffer = [0_u8; DOWNLOAD_BUFFER_BYTES];

        loop {
            let read_length = next_read_length(spec.exact_bytes(), total, buffer.len());
            check_cancelled(cancelled)?;
            let read = reader.read(&mut buffer[..read_length]);
            check_cancelled(cancelled)?;
            let count = read.map_err(|source| ModelManagerError::Io {
                operation: "read model download",
                path: temporary_path.clone(),
                source,
            })?;
            if count == 0 {
                break;
            }
            let observed = total.saturating_add(count as u64);
            if observed > spec.exact_bytes() {
                return Err(ModelManagerError::DownloadTooLarge {
                    expected: spec.exact_bytes(),
                    observed,
                });
            }

            temporary
                .file_mut()
                .write_all(&buffer[..count])
                .map_err(|source| ModelManagerError::Io {
                    operation: "write model download",
                    path: temporary_path.clone(),
                    source,
                })?;
            hasher.update(&buffer[..count]);
            total = observed;
        }

        if total != spec.exact_bytes() {
            return Err(integrity_error(
                reported_path.clone(),
                ModelIntegrityError::WrongSize {
                    expected: spec.exact_bytes(),
                    actual: total,
                },
            ));
        }

        let actual_sha256 = digest_to_hex(&hasher.finalize());
        if actual_sha256 != spec.sha256() {
            return Err(integrity_error(
                reported_path.clone(),
                ModelIntegrityError::ChecksumMismatch {
                    expected: spec.sha256().to_owned(),
                    actual: actual_sha256,
                },
            ));
        }

        check_cancelled(cancelled)?;
        temporary
            .file_mut()
            .flush()
            .map_err(|source| ModelManagerError::Io {
                operation: "flush model download",
                path: temporary_path.clone(),
                source,
            })?;
        check_cancelled(cancelled)?;
        temporary
            .file_mut()
            .sync_all()
            .map_err(|source| ModelManagerError::Io {
                operation: "sync model download",
                path: temporary_path,
                source,
            })?;
        check_cancelled(cancelled)?;

        temporary.verify_current()?;
        let target_exists = ensure_safe_replace_target(&snapshot, spec)?;
        snapshot.verify_mapping()?;
        temporary.commit(spec.filename(), target_exists, begin_commit)?;
        sync_containing_directory(&snapshot, &physical_path)?;
        snapshot.verify_mapping()?;
        Ok(reported_path)
    }

    fn remove_spec(&self, spec: &ModelSpec) -> Result<bool, ModelManagerError> {
        validate_spec(spec)?;
        let Some(snapshot) = self.capture_existing_root()? else {
            return Ok(false);
        };
        let physical_path = snapshot.physical_model_path(spec);
        let lock = model_lock(&physical_path);
        let _guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        snapshot.verify_mapping()?;
        self.remove_from_snapshot(spec, &snapshot)
    }

    fn remove_from_snapshot(
        &self,
        spec: &ModelSpec,
        snapshot: &RootSnapshot,
    ) -> Result<bool, ModelManagerError> {
        let path = snapshot.physical_model_path(spec);
        let metadata = match snapshot.dir.symlink_metadata(spec.filename()) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                snapshot.verify_mapping()?;
                return Ok(false);
            }
            Err(source) => {
                return Err(ModelManagerError::Io {
                    operation: "inspect model file",
                    path: path.clone(),
                    source,
                });
            }
        };
        ensure_cap_regular_model_file(&path, &metadata)?;
        snapshot.verify_mapping()?;
        snapshot
            .dir
            .remove_file(spec.filename())
            .map_err(|source| ModelManagerError::Io {
                operation: "remove model file",
                path: path.clone(),
                source,
            })?;
        snapshot.verify_mapping()?;
        Ok(true)
    }

    fn path_for_spec(&self, spec: &ModelSpec) -> Result<PathBuf, ModelManagerError> {
        validate_spec(spec)?;
        Ok(self.root.join(spec.filename()))
    }

    fn inspect_path(
        &self,
        spec: &ModelSpec,
        snapshot: &RootSnapshot,
    ) -> Result<ModelStatus, ModelManagerError> {
        let path = snapshot.physical_model_path(spec);
        let reported_path = snapshot.configured_model_path(spec);
        let metadata = match snapshot.dir.symlink_metadata(spec.filename()) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                snapshot.verify_mapping()?;
                return Ok(ModelStatus::Missing);
            }
            Err(source) => {
                return Err(ModelManagerError::Io {
                    operation: "inspect model file",
                    path,
                    source,
                });
            }
        };
        ensure_cap_regular_model_file(&path, &metadata)?;
        if metadata.len() != spec.exact_bytes() {
            snapshot.verify_mapping()?;
            return Ok(ModelStatus::Invalid {
                path: reported_path,
                reason: ModelIntegrityError::WrongSize {
                    expected: spec.exact_bytes(),
                    actual: metadata.len(),
                },
            });
        }

        let mut file =
            snapshot
                .dir
                .open(spec.filename())
                .map_err(|source| ModelManagerError::Io {
                    operation: "open model file",
                    path: path.clone(),
                    source,
                })?;
        let identity_file = file
            .try_clone()
            .map(CapabilityFile::into_std)
            .map_err(|source| ModelManagerError::Io {
                operation: "clone opened model file",
                path: path.clone(),
                source,
            })?;
        let opened_metadata = identity_file
            .metadata()
            .map_err(|source| ModelManagerError::Io {
                operation: "inspect opened model file",
                path: path.clone(),
                source,
            })?;
        ensure_regular_model_file(&path, &opened_metadata)?;
        let opened_version = FileVersion::from_metadata(&opened_metadata);
        let opened_identity =
            FileIdentity::from_file(identity_file).map_err(|source| ModelManagerError::Io {
                operation: "identify opened model file",
                path: path.clone(),
                source,
            })?;

        let mut hasher = Sha256::new();
        let mut total = 0_u64;
        let mut buffer = [0_u8; DOWNLOAD_BUFFER_BYTES];
        loop {
            let read_length = next_read_length(spec.exact_bytes(), total, buffer.len());
            let count =
                file.read(&mut buffer[..read_length])
                    .map_err(|source| ModelManagerError::Io {
                        operation: "read model file",
                        path: path.clone(),
                        source,
                    })?;
            if count == 0 {
                break;
            }
            total = total.saturating_add(count as u64);
            if total > spec.exact_bytes() {
                verify_model_file_unchanged(snapshot, spec, &opened_identity, &opened_version)?;
                return Ok(ModelStatus::Invalid {
                    path: reported_path,
                    reason: ModelIntegrityError::WrongSize {
                        expected: spec.exact_bytes(),
                        actual: total,
                    },
                });
            }
            hasher.update(&buffer[..count]);
        }
        let actual_sha256 = digest_to_hex(&hasher.finalize());
        let final_metadata =
            verify_model_file_unchanged(snapshot, spec, &opened_identity, &opened_version)?;
        if total != spec.exact_bytes() {
            return Ok(ModelStatus::Invalid {
                path: reported_path,
                reason: ModelIntegrityError::WrongSize {
                    expected: spec.exact_bytes(),
                    actual: total,
                },
            });
        }

        if actual_sha256 != spec.sha256() {
            return Ok(ModelStatus::Invalid {
                path: reported_path,
                reason: ModelIntegrityError::ChecksumMismatch {
                    expected: spec.sha256().to_owned(),
                    actual: actual_sha256,
                },
            });
        }

        if final_metadata.len() != spec.exact_bytes() {
            return Ok(ModelStatus::Invalid {
                path: reported_path,
                reason: ModelIntegrityError::WrongSize {
                    expected: spec.exact_bytes(),
                    actual: final_metadata.len(),
                },
            });
        }

        snapshot.verify_mapping()?;
        Ok(ModelStatus::Ready {
            path: reported_path,
        })
    }

    fn capture_existing_root(&self) -> Result<Option<RootSnapshot>, ModelManagerError> {
        let metadata = match fs::symlink_metadata(&self.root) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(source) => {
                return Err(ModelManagerError::Io {
                    operation: "inspect model root",
                    path: self.root.clone(),
                    source,
                });
            }
        };
        ensure_directory_root(&self.root, &metadata)?;
        RootSnapshot::capture(&self.root).map(Some)
    }

    fn create_and_capture_root(&self) -> Result<RootSnapshot, ModelManagerError> {
        fs::create_dir_all(&self.root).map_err(|source| ModelManagerError::Io {
            operation: "create model root",
            path: self.root.clone(),
            source,
        })?;
        RootSnapshot::capture(&self.root)
    }
}

struct RootSnapshot {
    lexical_root: PathBuf,
    canonical_root: PathBuf,
    dir: CapabilityDir,
    identity: FileIdentity,
}

impl RootSnapshot {
    fn capture(lexical_root: &Path) -> Result<Self, ModelManagerError> {
        let metadata =
            fs::symlink_metadata(lexical_root).map_err(|source| ModelManagerError::Io {
                operation: "inspect model root",
                path: lexical_root.to_path_buf(),
                source,
            })?;
        ensure_directory_root(lexical_root, &metadata)?;

        let canonical_root =
            fs::canonicalize(lexical_root).map_err(|source| ModelManagerError::Io {
                operation: "canonicalize model root",
                path: lexical_root.to_path_buf(),
                source,
            })?;
        let dir = CapabilityDir::open_ambient_dir(&canonical_root, ambient_authority()).map_err(
            |source| ModelManagerError::Io {
                operation: "open canonical model root",
                path: canonical_root.clone(),
                source,
            },
        )?;
        let identity_file =
            dir.try_clone()
                .map(CapabilityDir::into_std_file)
                .map_err(|source| ModelManagerError::Io {
                    operation: "clone canonical model root handle",
                    path: canonical_root.clone(),
                    source,
                })?;
        let opened_metadata = identity_file
            .metadata()
            .map_err(|source| ModelManagerError::Io {
                operation: "inspect canonical model root handle",
                path: canonical_root.clone(),
                source,
            })?;
        ensure_directory_root(&canonical_root, &opened_metadata)?;
        let identity =
            FileIdentity::from_file(identity_file).map_err(|source| ModelManagerError::Io {
                operation: "identify canonical model root",
                path: canonical_root.clone(),
                source,
            })?;

        let snapshot = Self {
            lexical_root: lexical_root.to_path_buf(),
            canonical_root,
            dir,
            identity,
        };
        snapshot.verify_mapping()?;
        Ok(snapshot)
    }

    fn physical_model_path(&self, spec: &ModelSpec) -> PathBuf {
        self.canonical_root.join(spec.filename())
    }

    fn configured_model_path(&self, spec: &ModelSpec) -> PathBuf {
        self.lexical_root.join(spec.filename())
    }

    fn verify_mapping(&self) -> Result<(), ModelManagerError> {
        let unchanged = || -> Option<bool> {
            let metadata = fs::symlink_metadata(&self.lexical_root).ok()?;
            if ensure_directory_root(&self.lexical_root, &metadata).is_err() {
                return Some(false);
            }

            let canonical = fs::canonicalize(&self.lexical_root).ok()?;
            if canonical != self.canonical_root {
                return Some(false);
            }

            let identity = FileIdentity::from_path(&self.lexical_root).ok()?;
            if identity != self.identity {
                return Some(false);
            }

            let final_metadata = fs::symlink_metadata(&self.lexical_root).ok()?;
            if ensure_directory_root(&self.lexical_root, &final_metadata).is_err() {
                return Some(false);
            }
            let final_canonical = fs::canonicalize(&self.lexical_root).ok()?;
            Some(final_canonical == self.canonical_root)
        };

        if unchanged() == Some(true) {
            Ok(())
        } else {
            Err(ModelManagerError::RootMappingChanged {
                root: self.lexical_root.clone(),
                pinned_root: self.canonical_root.clone(),
            })
        }
    }
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

fn verify_model_file_unchanged(
    snapshot: &RootSnapshot,
    spec: &ModelSpec,
    expected_identity: &FileIdentity,
    expected_version: &FileVersion,
) -> Result<cap_std::fs::Metadata, ModelManagerError> {
    let path = snapshot.physical_model_path(spec);
    let metadata = snapshot
        .dir
        .symlink_metadata(spec.filename())
        .map_err(|source| {
            if source.kind() == io::ErrorKind::NotFound {
                ModelManagerError::FilesystemObjectChanged {
                    path: path.clone(),
                    object: "model file",
                }
            } else {
                ModelManagerError::Io {
                    operation: "reinspect model file",
                    path: path.clone(),
                    source,
                }
            }
        })?;
    ensure_cap_regular_model_file(&path, &metadata)?;

    let current_file = snapshot
        .dir
        .open(spec.filename())
        .map(CapabilityFile::into_std)
        .map_err(|source| ModelManagerError::Io {
            operation: "reopen model file for identity verification",
            path: path.clone(),
            source,
        })?;
    let current_metadata = current_file
        .metadata()
        .map_err(|source| ModelManagerError::Io {
            operation: "reinspect opened model file",
            path: path.clone(),
            source,
        })?;
    ensure_regular_model_file(&path, &current_metadata)?;
    let current_version = FileVersion::from_metadata(&current_metadata);
    let current_identity =
        FileIdentity::from_file(current_file).map_err(|source| ModelManagerError::Io {
            operation: "reidentify model file",
            path: path.clone(),
            source,
        })?;
    if &current_identity != expected_identity || &current_version != expected_version {
        return Err(ModelManagerError::FilesystemObjectChanged {
            path,
            object: "model file",
        });
    }

    snapshot.verify_mapping()?;
    Ok(metadata)
}

fn validate_root(root: &Path) -> Result<PathBuf, ModelManagerError> {
    if !root.is_absolute() {
        return Err(ModelManagerError::RootNotAbsolute {
            root: root.to_path_buf(),
        });
    }

    let mut has_normal_component = false;
    for component in root.components() {
        match component {
            Component::ParentDir | Component::CurDir => {
                return Err(ModelManagerError::RootTraversal {
                    root: root.to_path_buf(),
                });
            }
            Component::Normal(_) => has_normal_component = true,
            Component::Prefix(_) | Component::RootDir => {}
        }
    }
    if !has_normal_component {
        return Err(ModelManagerError::FilesystemRoot {
            root: root.to_path_buf(),
        });
    }

    Ok(root.components().collect())
}

fn validate_spec(spec: &ModelSpec) -> Result<(), ModelManagerError> {
    let filename = spec.filename();
    if filename.is_empty()
        || filename == "."
        || filename == ".."
        || filename.contains('/')
        || filename.contains('\\')
    {
        return Err(ModelManagerError::InvalidCatalogEntry {
            model_id: spec.id(),
            detail: "filename must be one plain path component",
        });
    }
    let mut components = Path::new(filename).components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err(ModelManagerError::InvalidCatalogEntry {
            model_id: spec.id(),
            detail: "filename must be one plain path component",
        });
    }
    if spec.exact_bytes() == 0 {
        return Err(ModelManagerError::InvalidCatalogEntry {
            model_id: spec.id(),
            detail: "exact byte count must be non-zero",
        });
    }
    if spec.sha256().len() != 64
        || !spec
            .sha256()
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(ModelManagerError::InvalidCatalogEntry {
            model_id: spec.id(),
            detail: "SHA-256 must be 64 lowercase hexadecimal characters",
        });
    }
    Ok(())
}

fn ensure_directory_root(root: &Path, metadata: &fs::Metadata) -> Result<(), ModelManagerError> {
    if metadata.file_type().is_symlink() || std_metadata_is_reparse_point(metadata) {
        return Err(ModelManagerError::UnsafeRoot {
            root: root.to_path_buf(),
            detail: "root is a symlink or reparse point",
        });
    }
    if !metadata.is_dir() {
        return Err(ModelManagerError::UnsafeRoot {
            root: root.to_path_buf(),
            detail: "root is not a directory",
        });
    }
    Ok(())
}

fn ensure_regular_model_file(
    path: &Path,
    metadata: &fs::Metadata,
) -> Result<(), ModelManagerError> {
    if metadata.file_type().is_symlink() || std_metadata_is_reparse_point(metadata) {
        return Err(ModelManagerError::UnsafeModelFile {
            path: path.to_path_buf(),
            detail: "model path is a symlink or reparse point",
        });
    }
    if !metadata.is_file() {
        return Err(ModelManagerError::UnsafeModelFile {
            path: path.to_path_buf(),
            detail: "model path is not a regular file",
        });
    }
    Ok(())
}

fn ensure_cap_regular_model_file(
    path: &Path,
    metadata: &cap_std::fs::Metadata,
) -> Result<(), ModelManagerError> {
    if metadata.is_symlink() || cap_metadata_is_reparse_point(metadata) {
        return Err(ModelManagerError::UnsafeModelFile {
            path: path.to_path_buf(),
            detail: "model path is a symlink or reparse point",
        });
    }
    if !metadata.is_file() {
        return Err(ModelManagerError::UnsafeModelFile {
            path: path.to_path_buf(),
            detail: "model path is not a regular file",
        });
    }
    Ok(())
}

fn ensure_safe_replace_target(
    snapshot: &RootSnapshot,
    spec: &ModelSpec,
) -> Result<bool, ModelManagerError> {
    let path = snapshot.physical_model_path(spec);
    let metadata = match snapshot.dir.symlink_metadata(spec.filename()) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(source) => {
            return Err(ModelManagerError::Io {
                operation: "inspect model replacement target",
                path,
                source,
            });
        }
    };
    ensure_cap_regular_model_file(&path, &metadata)?;
    Ok(true)
}

#[cfg(windows)]
const WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;

#[cfg(windows)]
fn windows_attributes_are_reparse_point(attributes: u32) -> bool {
    attributes & WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT != 0
}

#[cfg(windows)]
fn std_metadata_is_reparse_point(metadata: &fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt as _;

    windows_attributes_are_reparse_point(metadata.file_attributes())
}

#[cfg(not(windows))]
fn std_metadata_is_reparse_point(_metadata: &fs::Metadata) -> bool {
    false
}

#[cfg(windows)]
fn cap_metadata_is_reparse_point(metadata: &cap_std::fs::Metadata) -> bool {
    use cap_std::fs::MetadataExt as _;

    windows_attributes_are_reparse_point(metadata.file_attributes())
}

#[cfg(not(windows))]
fn cap_metadata_is_reparse_point(_metadata: &cap_std::fs::Metadata) -> bool {
    false
}

#[cfg(unix)]
fn sync_containing_directory(
    snapshot: &RootSnapshot,
    installed_path: &Path,
) -> Result<(), ModelManagerError> {
    // cap-std opens directories with O_PATH on Linux, and fsync on an O_PATH
    // descriptor fails with EBADF, so reopen a regular read-only descriptor
    // and confirm it still names the pinned root before syncing through it.
    let sync_failed = |source| ModelManagerError::InstalledButDirectorySyncFailed {
        path: installed_path.to_path_buf(),
        source,
    };
    let directory = fs::File::open(&snapshot.canonical_root).map_err(sync_failed)?;
    let identity = FileIdentity::from_file(directory).map_err(sync_failed)?;
    if identity != snapshot.identity {
        return Err(ModelManagerError::RootMappingChanged {
            root: snapshot.lexical_root.clone(),
            pinned_root: snapshot.canonical_root.clone(),
        });
    }
    identity.as_file().sync_all().map_err(sync_failed)
}

#[cfg(not(unix))]
fn sync_containing_directory(
    _snapshot: &RootSnapshot,
    _installed_path: &Path,
) -> Result<(), ModelManagerError> {
    Ok(())
}

fn integrity_error(path: PathBuf, reason: ModelIntegrityError) -> ModelManagerError {
    ModelManagerError::Integrity { path, reason }
}

fn next_read_length(expected: u64, observed: u64, buffer_capacity: usize) -> usize {
    let remaining = expected.saturating_sub(observed);
    remaining.min((buffer_capacity - 1) as u64) as usize + 1
}

fn digest_to_hex(digest: &[u8]) -> String {
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut output, "{byte:02x}").expect("writing hexadecimal to a string cannot fail");
    }
    output
}

fn never_cancelled() -> bool {
    false
}

fn check_cancelled(cancelled: &dyn Fn() -> bool) -> Result<(), ModelManagerError> {
    let requested =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(cancelled)).unwrap_or(true);
    if requested {
        Err(ModelManagerError::Cancelled)
    } else {
        Ok(())
    }
}

fn check_begin_commit(begin_commit: &dyn Fn() -> bool) -> Result<(), ModelManagerError> {
    let accepted =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(begin_commit)).unwrap_or(false);
    if accepted {
        Ok(())
    } else {
        Err(ModelManagerError::Cancelled)
    }
}

fn lock_model_with_cancellation<'a>(
    lock: &'a Mutex<()>,
    cancelled: &dyn Fn() -> bool,
) -> Result<MutexGuard<'a, ()>, ModelManagerError> {
    loop {
        check_cancelled(cancelled)?;
        match lock.try_lock() {
            Ok(guard) => {
                check_cancelled(cancelled)?;
                return Ok(guard);
            }
            Err(TryLockError::Poisoned(poisoned)) => {
                let guard = poisoned.into_inner();
                check_cancelled(cancelled)?;
                return Ok(guard);
            }
            Err(TryLockError::WouldBlock) => {
                std::thread::yield_now();
                std::thread::sleep(MODEL_LOCK_RETRY_DELAY);
            }
        }
    }
}

fn model_lock(path: &Path) -> Arc<Mutex<()>> {
    let locks = MODEL_LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut locks = locks
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    locks.retain(|_, lock| lock.strong_count() > 0);
    if let Some(lock) = locks.get(path).and_then(Weak::upgrade) {
        return lock;
    }

    let lock = Arc::new(Mutex::new(()));
    locks.insert(path.to_path_buf(), Arc::downgrade(&lock));
    lock
}

struct TemporaryDownload {
    root: CapabilityDir,
    name: PathBuf,
    path: PathBuf,
    file: Option<CapabilityFile>,
    identity: Option<FileIdentity>,
    committed: bool,
}

impl TemporaryDownload {
    fn create(snapshot: &RootSnapshot, model_filename: &str) -> Result<Self, ModelManagerError> {
        let root = snapshot
            .dir
            .try_clone()
            .map_err(|source| ModelManagerError::Io {
                operation: "clone pinned model root",
                path: snapshot.canonical_root.clone(),
                source,
            })?;
        for _ in 0..TEMP_FILE_ATTEMPTS {
            let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let name = PathBuf::from(format!(
                ".{model_filename}.download-{}-{sequence}.tmp",
                std::process::id()
            ));
            let path = snapshot.canonical_root.join(&name);
            let mut options = CapabilityOpenOptions::new();
            options.write(true).create_new(true);
            match root.open_with(&name, &options) {
                Ok(file) => {
                    let mut temporary = Self {
                        root,
                        name,
                        path,
                        file: Some(file),
                        identity: None,
                        committed: false,
                    };
                    let identity_file = temporary
                        .file
                        .as_ref()
                        .expect("new temporary file is open")
                        .try_clone()
                        .map(CapabilityFile::into_std)
                        .map_err(|source| ModelManagerError::Io {
                            operation: "clone temporary model file",
                            path: temporary.path.clone(),
                            source,
                        })?;
                    let metadata =
                        identity_file
                            .metadata()
                            .map_err(|source| ModelManagerError::Io {
                                operation: "inspect temporary model file",
                                path: temporary.path.clone(),
                                source,
                            })?;
                    ensure_regular_model_file(&temporary.path, &metadata)?;
                    temporary.identity =
                        Some(FileIdentity::from_file(identity_file).map_err(|source| {
                            ModelManagerError::Io {
                                operation: "identify temporary model file",
                                path: temporary.path.clone(),
                                source,
                            }
                        })?);
                    return Ok(temporary);
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(source) => {
                    return Err(ModelManagerError::Io {
                        operation: "create temporary model file",
                        path,
                        source,
                    });
                }
            }
        }

        Err(ModelManagerError::TemporaryNameExhausted {
            root: snapshot.canonical_root.clone(),
        })
    }

    fn file_mut(&mut self) -> &mut CapabilityFile {
        self.file
            .as_mut()
            .expect("temporary model file is open until commit")
    }

    fn verify_current(&self) -> Result<(), ModelManagerError> {
        let metadata =
            self.root
                .symlink_metadata(&self.name)
                .map_err(|source| ModelManagerError::Io {
                    operation: "reinspect temporary model file",
                    path: self.path.clone(),
                    source,
                })?;
        ensure_cap_regular_model_file(&self.path, &metadata)?;

        let current_file = self
            .root
            .open(&self.name)
            .map(CapabilityFile::into_std)
            .map_err(|source| ModelManagerError::Io {
                operation: "reopen temporary model file",
                path: self.path.clone(),
                source,
            })?;
        let current_metadata = current_file
            .metadata()
            .map_err(|source| ModelManagerError::Io {
                operation: "reinspect opened temporary model file",
                path: self.path.clone(),
                source,
            })?;
        ensure_regular_model_file(&self.path, &current_metadata)?;
        let current_identity =
            FileIdentity::from_file(current_file).map_err(|source| ModelManagerError::Io {
                operation: "reidentify temporary model file",
                path: self.path.clone(),
                source,
            })?;
        if self.identity.as_ref() != Some(&current_identity) {
            return Err(ModelManagerError::FilesystemObjectChanged {
                path: self.path.clone(),
                object: "temporary model file",
            });
        }
        Ok(())
    }

    fn commit(
        mut self,
        target_name: &str,
        target_exists: bool,
        begin_commit: &dyn Fn() -> bool,
    ) -> Result<(), ModelManagerError> {
        drop(self.identity.take());
        drop(self.file.take());

        check_begin_commit(begin_commit)?;

        #[cfg(windows)]
        if target_exists {
            self.root
                .remove_file(target_name)
                .map_err(|source| ModelManagerError::Io {
                    operation: "remove invalid model before replacement",
                    path: self.root_path(target_name),
                    source,
                })?;
        }
        #[cfg(not(windows))]
        let _ = target_exists;

        self.root
            .rename(&self.name, &self.root, target_name)
            .map_err(|source| ModelManagerError::Io {
                operation: "atomically install model file",
                path: self.root_path(target_name),
                source,
            })?;
        self.committed = true;
        Ok(())
    }

    fn root_path(&self, name: &str) -> PathBuf {
        self.path
            .parent()
            .expect("temporary model path has a parent")
            .join(name)
    }
}

impl Drop for TemporaryDownload {
    fn drop(&mut self) {
        if !self.committed {
            drop(self.identity.take());
            drop(self.file.take());
            let _ = self.root.remove_file(&self.name);
        }
    }
}

#[cfg(test)]
mod tests;
