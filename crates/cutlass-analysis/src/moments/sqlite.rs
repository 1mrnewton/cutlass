//! SQLite persistence for validated media moments.

use std::error::Error;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::str;
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use rusqlite::ffi::ErrorCode;
use rusqlite::limits::Limit;
use rusqlite::types::ValueRef;
use rusqlite::{Connection, OpenFlags, OptionalExtension, Row, TransactionBehavior, params};

use super::{
    AnalyzerIdentity, AnalyzerVersion, MAX_ANALYZER_IDENTITY_BYTES, MAX_MOMENT_KIND_BYTES,
    MAX_MOMENT_PAYLOAD_BYTES, MEDIA_CONTENT_KEY_BYTES, MediaContentKey, MomentBatchKey,
    MomentConfidence, MomentKind, MomentPayload, MomentQuery, MomentRecord, MomentReplacementBatch,
};
use crate::TimeSpan;

/// Filename used by [`MomentsIndex::open_in_cache_root`].
pub const MOMENTS_INDEX_DATABASE_FILENAME: &str = "moments.sqlite3";

/// Current SQLite schema version stored in `PRAGMA user_version`.
pub const MOMENTS_INDEX_SCHEMA_VERSION: u32 = 1;

/// Default maximum number of records returned by one query.
pub const DEFAULT_MOMENTS_QUERY_LIMIT: usize = 256;

/// Absolute upper bound for a configured or caller-requested query limit.
pub const MAX_MOMENTS_QUERY_LIMIT: usize = 4_096;

/// Default duration SQLite waits for a contended database lock.
pub const DEFAULT_MOMENTS_BUSY_TIMEOUT: Duration = Duration::from_secs(5);

/// Longest lock wait accepted by [`MomentsIndexConfig`].
pub const MAX_MOMENTS_BUSY_TIMEOUT: Duration = Duration::from_secs(60);

/// Largest time boundary accepted by the SQLite schema.
///
/// This is the largest finite `f64`, preserving the full validated
/// [`TimeSpan`] domain while excluding SQLite infinities.
pub const MAX_MOMENT_TIME_SECONDS: f64 = f64::MAX;

const SQLITE_LIMIT_LENGTH_BYTES: i32 = 1_048_576;
const SQLITE_LIMIT_SQL_BYTES: i32 = 65_536;
const SQLITE_LIMIT_COLUMNS: i32 = 64;
const SQLITE_LIMIT_EXPRESSION_DEPTH: i32 = 100;
const SQLITE_LIMIT_COMPOUND_SELECTS: i32 = 8;
const SQLITE_LIMIT_VM_OPERATIONS: i32 = 100_000;
const SQLITE_LIMIT_FUNCTION_ARGUMENTS: i32 = 32;
const SQLITE_LIMIT_ATTACHED_DATABASES: i32 = 0;
const SQLITE_LIMIT_LIKE_PATTERN_BYTES: i32 = 1_024;
const SQLITE_LIMIT_VARIABLES: i32 = 32;
const SQLITE_LIMIT_TRIGGER_DEPTH: i32 = 8;
const SQLITE_LIMIT_WORKER_THREADS: i32 = 2;

const UPSERT_BATCH_SQL: &str = "
    INSERT INTO moment_batches (
        content_key,
        analyzer_identity,
        analyzer_version
    ) VALUES (?1, ?2, ?3)
    ON CONFLICT (content_key, analyzer_identity, analyzer_version) DO NOTHING
";

const DELETE_MOMENTS_SQL: &str = "
    DELETE FROM moments
    WHERE content_key = ?1
      AND analyzer_identity = ?2
      AND analyzer_version = ?3
";

const INSERT_MOMENT_SQL: &str = "
    INSERT INTO moments (
        content_key,
        analyzer_identity,
        analyzer_version,
        start_seconds,
        end_seconds,
        kind,
        confidence,
        payload_present,
        payload_text
    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
";

const QUERY_MOMENTS_SQL: &str = "
    SELECT
        content_key,
        start_seconds,
        end_seconds,
        kind,
        confidence,
        analyzer_identity,
        analyzer_version,
        payload_present,
        payload_text
    FROM moments
    WHERE content_key = ?1
      AND (?2 IS NULL OR kind = ?2)
      AND confidence >= ?3
      AND (
          ?4 = 0
          OR (
              ?5 < ?6
              AND (
                  (
                      start_seconds = end_seconds
                      AND ?5 <= start_seconds
                      AND start_seconds < ?6
                  )
                  OR (
                      start_seconds < end_seconds
                      AND start_seconds < ?6
                      AND end_seconds > ?5
                  )
              )
          )
      )
    ORDER BY
        start_seconds ASC,
        end_seconds ASC,
        kind ASC,
        analyzer_identity ASC,
        analyzer_version ASC,
        confidence DESC,
        payload_present ASC,
        payload_text ASC
    LIMIT ?7
";

/// Validated operational limits for a [`MomentsIndex`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MomentsIndexConfig {
    default_query_limit: usize,
    maximum_query_limit: usize,
    busy_timeout: Duration,
}

impl MomentsIndexConfig {
    /// Creates a validated index configuration.
    ///
    /// Query limits must be non-zero, the default must not exceed the
    /// configured maximum, and the maximum must not exceed
    /// [`MAX_MOMENTS_QUERY_LIMIT`]. `busy_timeout` may be zero and must not
    /// exceed [`MAX_MOMENTS_BUSY_TIMEOUT`].
    ///
    /// # Errors
    ///
    /// Returns [`MomentsIndexError::InvalidConfig`] for the first invalid
    /// field.
    pub fn new(
        default_query_limit: usize,
        maximum_query_limit: usize,
        busy_timeout: Duration,
    ) -> Result<Self, MomentsIndexError> {
        let config = Self {
            default_query_limit,
            maximum_query_limit,
            busy_timeout,
        };
        config.validate()?;
        Ok(config)
    }

    /// Number of records requested by [`MomentsIndex::query`].
    #[must_use]
    pub const fn default_query_limit(self) -> usize {
        self.default_query_limit
    }

    /// Largest limit accepted by [`MomentsIndex::query_with_limit`].
    #[must_use]
    pub const fn maximum_query_limit(self) -> usize {
        self.maximum_query_limit
    }

    /// Duration SQLite waits for a contended database lock.
    #[must_use]
    pub const fn busy_timeout(self) -> Duration {
        self.busy_timeout
    }

    fn validate(self) -> Result<(), MomentsIndexError> {
        if self.maximum_query_limit == 0 {
            return Err(MomentsIndexError::InvalidConfig {
                field: "maximum_query_limit",
                reason: "must be non-zero",
            });
        }
        if self.maximum_query_limit > MAX_MOMENTS_QUERY_LIMIT {
            return Err(MomentsIndexError::InvalidConfig {
                field: "maximum_query_limit",
                reason: "exceeds the absolute query limit",
            });
        }
        if self.default_query_limit == 0 {
            return Err(MomentsIndexError::InvalidConfig {
                field: "default_query_limit",
                reason: "must be non-zero",
            });
        }
        if self.default_query_limit > self.maximum_query_limit {
            return Err(MomentsIndexError::InvalidConfig {
                field: "default_query_limit",
                reason: "must not exceed maximum_query_limit",
            });
        }
        if self.busy_timeout > MAX_MOMENTS_BUSY_TIMEOUT {
            return Err(MomentsIndexError::InvalidConfig {
                field: "busy_timeout",
                reason: "exceeds the maximum busy timeout",
            });
        }
        Ok(())
    }
}

impl Default for MomentsIndexConfig {
    fn default() -> Self {
        Self {
            default_query_limit: DEFAULT_MOMENTS_QUERY_LIMIT,
            maximum_query_limit: MAX_MOMENTS_QUERY_LIMIT,
            busy_timeout: DEFAULT_MOMENTS_BUSY_TIMEOUT,
        }
    }
}

/// Result of one bounded moments query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MomentsQueryResult {
    records: Vec<MomentRecord>,
    truncated: bool,
}

impl MomentsQueryResult {
    /// Returned records in [`MomentRecord`]'s canonical order.
    #[must_use]
    pub fn records(&self) -> &[MomentRecord] {
        &self.records
    }

    /// Consumes the result and returns its records.
    #[must_use]
    pub fn into_records(self) -> Vec<MomentRecord> {
        self.records
    }

    /// Whether at least one additional matching record exists.
    #[must_use]
    pub const fn truncated(&self) -> bool {
        self.truncated
    }

    /// Number of returned records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether no records were returned.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

/// Failures produced by the SQLite moments adapter.
#[derive(Debug)]
#[non_exhaustive]
pub enum MomentsIndexError {
    /// An index configuration field violated its documented bounds.
    InvalidConfig {
        /// Stable configuration field name.
        field: &'static str,
        /// Stable explanation that does not contain caller data.
        reason: &'static str,
    },
    /// A requested query limit was zero or exceeded the configured maximum.
    InvalidQueryLimit {
        /// Caller-requested limit.
        requested: usize,
        /// Configured maximum accepted by this index.
        maximum: usize,
    },
    /// The database was created by a newer schema version.
    UnsupportedSchema {
        /// Version found in `PRAGMA user_version`.
        found: i64,
        /// Newest version supported by this adapter.
        supported: u32,
    },
    /// The declared schema version did not match the required schema objects.
    SchemaMismatch {
        /// Stable mismatch category without database-provided text.
        reason: &'static str,
    },
    /// A returned row could not be reconstructed through domain constructors.
    CorruptRow {
        /// Zero-based position within the bounded SQLite result.
        row_index: usize,
        /// Stable field name.
        field: &'static str,
        /// Stable reason that never includes stored field contents.
        reason: &'static str,
    },
    /// The connection mutex was poisoned by an earlier panic.
    MutexPoisoned,
    /// Creating the database parent directory failed.
    FileSystem {
        /// Stable filesystem operation name.
        operation: &'static str,
        /// Underlying standard-library error.
        source: io::Error,
    },
    /// SQLite could not proceed because the database or a table was locked.
    DatabaseLocked {
        /// Underlying SQLite error.
        source: rusqlite::Error,
    },
    /// SQLite reported a malformed or non-database file.
    DatabaseCorrupt {
        /// Underlying SQLite error.
        source: rusqlite::Error,
    },
    /// SQLite rejected a value, trigger, size, or table constraint.
    ConstraintViolation {
        /// Underlying SQLite error.
        source: rusqlite::Error,
    },
    /// SQLite reported an operating-system, access, capacity, or open failure.
    DatabaseIo {
        /// Underlying SQLite error.
        source: rusqlite::Error,
    },
    /// Another SQLite failure occurred.
    Database {
        /// Underlying SQLite error.
        source: rusqlite::Error,
    },
}

impl fmt::Display for MomentsIndexError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig { field, reason } => {
                write!(
                    formatter,
                    "invalid moments index config `{field}`: {reason}"
                )
            }
            Self::InvalidQueryLimit { requested, maximum } => write!(
                formatter,
                "invalid moments query limit {requested}; expected 1..={maximum}"
            ),
            Self::UnsupportedSchema { found, supported } => write!(
                formatter,
                "moments index schema version {found} is newer than supported version {supported}"
            ),
            Self::SchemaMismatch { reason } => {
                write!(formatter, "moments index schema mismatch: {reason}")
            }
            Self::CorruptRow {
                row_index,
                field,
                reason,
            } => write!(
                formatter,
                "corrupt moments row {row_index}, field `{field}`: {reason}"
            ),
            Self::MutexPoisoned => {
                formatter.write_str("moments index connection mutex is poisoned")
            }
            Self::FileSystem { operation, source } => {
                write!(formatter, "moments index {operation} failed: {source}")
            }
            Self::DatabaseLocked { source } => {
                write!(formatter, "moments database is locked: {source}")
            }
            Self::DatabaseCorrupt { source } => {
                write!(formatter, "moments database is corrupt: {source}")
            }
            Self::ConstraintViolation { source } => {
                write!(formatter, "moments database constraint failed: {source}")
            }
            Self::DatabaseIo { source } => {
                write!(formatter, "moments database I/O failed: {source}")
            }
            Self::Database { source } => write!(formatter, "moments database failed: {source}"),
        }
    }
}

impl Error for MomentsIndexError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::FileSystem { source, .. } => Some(source),
            Self::DatabaseLocked { source }
            | Self::DatabaseCorrupt { source }
            | Self::ConstraintViolation { source }
            | Self::DatabaseIo { source }
            | Self::Database { source } => Some(source),
            Self::InvalidConfig { .. }
            | Self::InvalidQueryLimit { .. }
            | Self::UnsupportedSchema { .. }
            | Self::SchemaMismatch { .. }
            | Self::CorruptRow { .. }
            | Self::MutexPoisoned => None,
        }
    }
}

/// Thread-safe SQLite persistence for complete analyzer moment batches.
///
/// A single rusqlite connection is serialized by a mutex. Replacement batches
/// use `BEGIN IMMEDIATE`, so readers never observe a partially replaced batch.
pub struct MomentsIndex {
    path: PathBuf,
    config: MomentsIndexConfig,
    connection: Mutex<Connection>,
}

impl MomentsIndex {
    /// Opens or creates an index at an explicit database path.
    ///
    /// Missing parent directories are created. Existing files are never
    /// deleted or replaced when opening, migrating, or reporting corruption.
    ///
    /// # Errors
    ///
    /// Returns a typed [`MomentsIndexError`] for filesystem, SQLite, schema,
    /// or configuration failures.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, MomentsIndexError> {
        Self::open_with_config(path, MomentsIndexConfig::default())
    }

    /// Opens or creates the standard database beneath a cache root.
    ///
    /// The path is `cache_root /` [`MOMENTS_INDEX_DATABASE_FILENAME`].
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`MomentsIndex::open`].
    pub fn open_in_cache_root(cache_root: impl AsRef<Path>) -> Result<Self, MomentsIndexError> {
        Self::open_in_cache_root_with_config(cache_root, MomentsIndexConfig::default())
    }

    /// Opens an explicit database path with validated operational limits.
    ///
    /// # Errors
    ///
    /// Returns a typed [`MomentsIndexError`] for filesystem, SQLite, schema,
    /// or configuration failures.
    pub fn open_with_config(
        path: impl AsRef<Path>,
        config: MomentsIndexConfig,
    ) -> Result<Self, MomentsIndexError> {
        config.validate()?;
        let path = path.as_ref().to_path_buf();
        if path.as_os_str().is_empty() {
            return Err(MomentsIndexError::SchemaMismatch {
                reason: "database path must not be empty",
            });
        }

        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).map_err(|source| MomentsIndexError::FileSystem {
                operation: "parent directory creation",
                source,
            })?;
        }

        let flags = OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_CREATE
            | OpenFlags::SQLITE_OPEN_NO_MUTEX;
        let mut connection = Connection::open_with_flags(&path, flags).map_err(map_sqlite_error)?;
        initialize_connection(&mut connection, config)?;

        Ok(Self {
            path,
            config,
            connection: Mutex::new(connection),
        })
    }

    /// Opens the standard cache-root database with explicit configuration.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`MomentsIndex::open_with_config`].
    pub fn open_in_cache_root_with_config(
        cache_root: impl AsRef<Path>,
        config: MomentsIndexConfig,
    ) -> Result<Self, MomentsIndexError> {
        Self::open_with_config(
            cache_root.as_ref().join(MOMENTS_INDEX_DATABASE_FILENAME),
            config,
        )
    }

    /// Exact database path supplied to the open operation.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Operational configuration used by this index.
    #[must_use]
    pub const fn config(&self) -> MomentsIndexConfig {
        self.config
    }

    /// Atomically replaces one analyzer's complete result batch.
    ///
    /// The batch marker is retained even when `batch` contains no records.
    /// Any failure before commit rolls back the marker, deletion, and all
    /// inserts, leaving the prior complete batch unchanged.
    ///
    /// # Errors
    ///
    /// Returns a typed lock, SQLite, constraint, or mutex error.
    pub fn replace_batch(&self, batch: &MomentReplacementBatch) -> Result<(), MomentsIndexError> {
        let key = batch.key();
        let content_key = key.content_key();
        let analyzer_version = i64::from(key.analyzer_version().get());
        let mut connection = self.lock_connection()?;
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(map_sqlite_error)?;

        transaction
            .execute(
                UPSERT_BATCH_SQL,
                params![
                    &content_key.as_bytes()[..],
                    key.analyzer_identity().as_str(),
                    analyzer_version,
                ],
            )
            .map_err(map_sqlite_error)?;
        transaction
            .execute(
                DELETE_MOMENTS_SQL,
                params![
                    &content_key.as_bytes()[..],
                    key.analyzer_identity().as_str(),
                    analyzer_version,
                ],
            )
            .map_err(map_sqlite_error)?;

        {
            let mut insert = transaction
                .prepare_cached(INSERT_MOMENT_SQL)
                .map_err(map_sqlite_error)?;
            for record in batch.records() {
                let (payload_present, payload_text) = record
                    .payload()
                    .map_or((0_i64, ""), |payload| (1_i64, payload.as_str()));
                insert
                    .execute(params![
                        &content_key.as_bytes()[..],
                        key.analyzer_identity().as_str(),
                        analyzer_version,
                        record.span().start_seconds(),
                        record.span().end_seconds(),
                        record.kind().as_str(),
                        f64::from(record.confidence().get()),
                        payload_present,
                        payload_text,
                    ])
                    .map_err(map_sqlite_error)?;
            }
        }

        transaction.commit().map_err(map_sqlite_error)
    }

    /// Returns whether a completed batch marker exists for `key`.
    ///
    /// This returns `true` for a successfully persisted empty replacement.
    ///
    /// # Errors
    ///
    /// Returns a typed SQLite or mutex error.
    pub fn batch_exists(&self, key: &MomentBatchKey) -> Result<bool, MomentsIndexError> {
        let content_key = key.content_key();
        let connection = self.lock_connection()?;
        connection
            .query_row(
                "
                    SELECT EXISTS (
                        SELECT 1
                        FROM moment_batches
                        WHERE content_key = ?1
                          AND analyzer_identity = ?2
                          AND analyzer_version = ?3
                    )
                ",
                params![
                    &content_key.as_bytes()[..],
                    key.analyzer_identity().as_str(),
                    i64::from(key.analyzer_version().get()),
                ],
                |row| row.get::<_, bool>(0),
            )
            .map_err(map_sqlite_error)
    }

    /// Deletes a completed batch marker and its cascading child rows.
    ///
    /// The returned boolean is `true` when a marker existed.
    ///
    /// # Errors
    ///
    /// Returns a typed SQLite or mutex error.
    pub fn delete_batch(&self, key: &MomentBatchKey) -> Result<bool, MomentsIndexError> {
        let content_key = key.content_key();
        let connection = self.lock_connection()?;
        let deleted = connection
            .execute(
                "
                    DELETE FROM moment_batches
                    WHERE content_key = ?1
                      AND analyzer_identity = ?2
                      AND analyzer_version = ?3
                ",
                params![
                    &content_key.as_bytes()[..],
                    key.analyzer_identity().as_str(),
                    i64::from(key.analyzer_version().get()),
                ],
            )
            .map_err(map_sqlite_error)?;
        Ok(deleted != 0)
    }

    /// Runs a bounded query using the configured default limit.
    ///
    /// # Errors
    ///
    /// Returns a typed SQLite, corrupt-row, or mutex error.
    pub fn query(&self, query: &MomentQuery) -> Result<MomentsQueryResult, MomentsIndexError> {
        self.query_with_limit(query, self.config.default_query_limit)
    }

    /// Runs a bounded query with an explicit positive limit.
    ///
    /// One sentinel row beyond `limit` is read and validated. `truncated` is
    /// therefore set only when an additional matching row actually exists.
    ///
    /// # Errors
    ///
    /// Returns [`MomentsIndexError::InvalidQueryLimit`] for zero or a value
    /// above this index's configured maximum. SQLite, corrupt-row, and mutex
    /// failures are also returned.
    pub fn query_with_limit(
        &self,
        query: &MomentQuery,
        limit: usize,
    ) -> Result<MomentsQueryResult, MomentsIndexError> {
        if limit == 0 || limit > self.config.maximum_query_limit {
            return Err(MomentsIndexError::InvalidQueryLimit {
                requested: limit,
                maximum: self.config.maximum_query_limit,
            });
        }
        let fetch_limit = limit
            .checked_add(1)
            .ok_or(MomentsIndexError::InvalidQueryLimit {
                requested: limit,
                maximum: self.config.maximum_query_limit,
            })?;
        let fetch_limit_sql =
            i64::try_from(fetch_limit).map_err(|_| MomentsIndexError::InvalidQueryLimit {
                requested: limit,
                maximum: self.config.maximum_query_limit,
            })?;

        let (has_time_range, query_start, query_end) =
            query.time_range().map_or((0_i64, 0.0, 0.0), |range| {
                (1_i64, range.start_seconds(), range.end_seconds())
            });
        let content_key = query.content_key();
        let kind = query.kind().map(MomentKind::as_str);
        let minimum_confidence = f64::from(query.minimum_confidence().get());

        let connection = self.lock_connection()?;
        let mut statement = connection
            .prepare_cached(QUERY_MOMENTS_SQL)
            .map_err(map_sqlite_error)?;
        let mut rows = statement
            .query(params![
                &content_key.as_bytes()[..],
                kind,
                minimum_confidence,
                has_time_range,
                query_start,
                query_end,
                fetch_limit_sql,
            ])
            .map_err(map_sqlite_error)?;

        let mut records = Vec::with_capacity(fetch_limit);
        while let Some(row) = rows.next().map_err(map_sqlite_error)? {
            records.push(hydrate_record(row, records.len())?);
        }

        let truncated = records.len() > limit;
        if truncated {
            records.pop();
        }
        Ok(MomentsQueryResult { records, truncated })
    }

    fn lock_connection(&self) -> Result<MutexGuard<'_, Connection>, MomentsIndexError> {
        self.connection
            .lock()
            .map_err(|_| MomentsIndexError::MutexPoisoned)
    }
}

fn initialize_connection(
    connection: &mut Connection,
    config: MomentsIndexConfig,
) -> Result<(), MomentsIndexError> {
    connection
        .busy_timeout(config.busy_timeout)
        .map_err(map_sqlite_error)?;
    configure_sqlite_limits(connection)?;
    connection
        .pragma_update(None, "foreign_keys", "ON")
        .map_err(map_sqlite_error)?;
    let foreign_keys_enabled: i64 = connection
        .pragma_query_value(None, "foreign_keys", |row| row.get(0))
        .map_err(map_sqlite_error)?;
    if foreign_keys_enabled != 1 {
        return Err(MomentsIndexError::SchemaMismatch {
            reason: "SQLite foreign keys could not be enabled",
        });
    }

    let schema_version: i64 = connection
        .pragma_query_value(None, "user_version", |row| row.get(0))
        .map_err(map_sqlite_error)?;
    if schema_version < 0 {
        return Err(MomentsIndexError::SchemaMismatch {
            reason: "schema version must not be negative",
        });
    }
    if schema_version > i64::from(MOMENTS_INDEX_SCHEMA_VERSION) {
        return Err(MomentsIndexError::UnsupportedSchema {
            found: schema_version,
            supported: MOMENTS_INDEX_SCHEMA_VERSION,
        });
    }
    configure_durability(connection)?;

    if schema_version == 0 {
        initialize_schema(connection)?;
    }
    validate_schema(connection)
}

fn configure_sqlite_limits(connection: &Connection) -> Result<(), MomentsIndexError> {
    let limits = [
        (Limit::SQLITE_LIMIT_LENGTH, SQLITE_LIMIT_LENGTH_BYTES),
        (Limit::SQLITE_LIMIT_SQL_LENGTH, SQLITE_LIMIT_SQL_BYTES),
        (Limit::SQLITE_LIMIT_COLUMN, SQLITE_LIMIT_COLUMNS),
        (
            Limit::SQLITE_LIMIT_EXPR_DEPTH,
            SQLITE_LIMIT_EXPRESSION_DEPTH,
        ),
        (
            Limit::SQLITE_LIMIT_COMPOUND_SELECT,
            SQLITE_LIMIT_COMPOUND_SELECTS,
        ),
        (Limit::SQLITE_LIMIT_VDBE_OP, SQLITE_LIMIT_VM_OPERATIONS),
        (
            Limit::SQLITE_LIMIT_FUNCTION_ARG,
            SQLITE_LIMIT_FUNCTION_ARGUMENTS,
        ),
        (
            Limit::SQLITE_LIMIT_ATTACHED,
            SQLITE_LIMIT_ATTACHED_DATABASES,
        ),
        (
            Limit::SQLITE_LIMIT_LIKE_PATTERN_LENGTH,
            SQLITE_LIMIT_LIKE_PATTERN_BYTES,
        ),
        (Limit::SQLITE_LIMIT_VARIABLE_NUMBER, SQLITE_LIMIT_VARIABLES),
        (
            Limit::SQLITE_LIMIT_TRIGGER_DEPTH,
            SQLITE_LIMIT_TRIGGER_DEPTH,
        ),
        (
            Limit::SQLITE_LIMIT_WORKER_THREADS,
            SQLITE_LIMIT_WORKER_THREADS,
        ),
    ];

    for (limit, maximum) in limits {
        connection
            .set_limit(limit, maximum)
            .map_err(map_sqlite_error)?;
        let effective = connection.limit(limit).map_err(map_sqlite_error)?;
        if effective > maximum {
            return Err(MomentsIndexError::SchemaMismatch {
                reason: "SQLite runtime limit could not be bounded",
            });
        }
    }
    Ok(())
}

fn configure_durability(connection: &Connection) -> Result<(), MomentsIndexError> {
    let wal_result = connection.query_row("PRAGMA journal_mode = WAL", [], |row| {
        row.get::<_, String>(0)
    });
    if let Err(source) = wal_result
        && !is_locking_error(&source)
    {
        return Err(map_sqlite_error(source));
    }

    connection
        .pragma_update(None, "synchronous", "NORMAL")
        .map_err(map_sqlite_error)
}

fn require_empty_unversioned_database(connection: &Connection) -> Result<(), MomentsIndexError> {
    let object_count: i64 = connection
        .query_row(
            "
                SELECT COUNT(*)
                FROM sqlite_schema
                WHERE name NOT LIKE 'sqlite_%'
            ",
            [],
            |row| row.get(0),
        )
        .map_err(map_sqlite_error)?;
    if object_count != 0 {
        return Err(MomentsIndexError::SchemaMismatch {
            reason: "unversioned database is not empty",
        });
    }
    Ok(())
}

fn initialize_schema(connection: &mut Connection) -> Result<(), MomentsIndexError> {
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(map_sqlite_error)?;
    let schema_version: i64 = transaction
        .pragma_query_value(None, "user_version", |row| row.get(0))
        .map_err(map_sqlite_error)?;
    if schema_version > i64::from(MOMENTS_INDEX_SCHEMA_VERSION) {
        return Err(MomentsIndexError::UnsupportedSchema {
            found: schema_version,
            supported: MOMENTS_INDEX_SCHEMA_VERSION,
        });
    }
    if schema_version == i64::from(MOMENTS_INDEX_SCHEMA_VERSION) {
        transaction.commit().map_err(map_sqlite_error)?;
        return Ok(());
    }
    if schema_version != 0 {
        return Err(MomentsIndexError::SchemaMismatch {
            reason: "schema version is not supported",
        });
    }
    require_empty_unversioned_database(&transaction)?;

    transaction
        .execute(&create_batches_sql(), [])
        .map_err(map_sqlite_error)?;
    transaction
        .execute(&create_moments_sql(), [])
        .map_err(map_sqlite_error)?;
    transaction
        .execute(&create_query_index_sql(), [])
        .map_err(map_sqlite_error)?;
    transaction
        .pragma_update(
            None,
            "user_version",
            i64::from(MOMENTS_INDEX_SCHEMA_VERSION),
        )
        .map_err(map_sqlite_error)?;
    transaction.commit().map_err(map_sqlite_error)
}

fn validate_schema(connection: &Connection) -> Result<(), MomentsIndexError> {
    validate_schema_object(
        connection,
        "table",
        "moment_batches",
        &create_batches_sql(),
        "moment_batches table is missing or incompatible",
    )?;
    validate_schema_object(
        connection,
        "table",
        "moments",
        &create_moments_sql(),
        "moments table is missing or incompatible",
    )?;
    validate_schema_object(
        connection,
        "index",
        "moments_query_order",
        &create_query_index_sql(),
        "moments query index is missing or incompatible",
    )
}

fn validate_schema_object(
    connection: &Connection,
    object_type: &str,
    name: &str,
    expected_sql: &str,
    reason: &'static str,
) -> Result<(), MomentsIndexError> {
    let actual_sql = connection
        .query_row(
            "
                SELECT sql
                FROM sqlite_schema
                WHERE type = ?1 AND name = ?2
            ",
            params![object_type, name],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(map_sqlite_error)?;
    let Some(actual_sql) = actual_sql else {
        return Err(MomentsIndexError::SchemaMismatch { reason });
    };
    if normalize_schema_sql(&actual_sql) != normalize_schema_sql(expected_sql) {
        return Err(MomentsIndexError::SchemaMismatch { reason });
    }
    Ok(())
}

fn normalize_schema_sql(sql: &str) -> String {
    sql.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn create_batches_sql() -> String {
    format!(
        "
        CREATE TABLE moment_batches (
            content_key BLOB NOT NULL
                CHECK (
                    typeof(content_key) = 'blob'
                    AND length(content_key) = {content_key_bytes}
                ),
            analyzer_identity TEXT NOT NULL
                CHECK (
                    typeof(analyzer_identity) = 'text'
                    AND length(CAST(analyzer_identity AS BLOB))
                        BETWEEN 1 AND {maximum_identity_bytes}
                    AND substr(analyzer_identity, 1, 1) GLOB '[a-z0-9]'
                    AND analyzer_identity NOT GLOB '*[^a-z0-9._/-]*'
                ),
            analyzer_version INTEGER NOT NULL
                CHECK (
                    typeof(analyzer_version) = 'integer'
                    AND analyzer_version BETWEEN 1 AND {maximum_analyzer_version}
                ),
            PRIMARY KEY (
                content_key,
                analyzer_identity,
                analyzer_version
            )
        ) WITHOUT ROWID
        ",
        content_key_bytes = MEDIA_CONTENT_KEY_BYTES,
        maximum_identity_bytes = MAX_ANALYZER_IDENTITY_BYTES,
        maximum_analyzer_version = u32::MAX,
    )
}

fn create_moments_sql() -> String {
    format!(
        "
        CREATE TABLE moments (
            content_key BLOB NOT NULL
                CHECK (
                    typeof(content_key) = 'blob'
                    AND length(content_key) = {content_key_bytes}
                ),
            analyzer_identity TEXT NOT NULL
                CHECK (
                    typeof(analyzer_identity) = 'text'
                    AND length(CAST(analyzer_identity AS BLOB))
                        BETWEEN 1 AND {maximum_identity_bytes}
                    AND substr(analyzer_identity, 1, 1) GLOB '[a-z0-9]'
                    AND analyzer_identity NOT GLOB '*[^a-z0-9._/-]*'
                ),
            analyzer_version INTEGER NOT NULL
                CHECK (
                    typeof(analyzer_version) = 'integer'
                    AND analyzer_version BETWEEN 1 AND {maximum_analyzer_version}
                ),
            start_seconds REAL NOT NULL
                CHECK (
                    typeof(start_seconds) = 'real'
                    AND start_seconds >= 0.0
                    AND start_seconds <= {maximum_time:e}
                ),
            end_seconds REAL NOT NULL
                CHECK (
                    typeof(end_seconds) = 'real'
                    AND end_seconds >= start_seconds
                    AND end_seconds <= {maximum_time:e}
                ),
            kind TEXT NOT NULL
                CHECK (
                    typeof(kind) = 'text'
                    AND length(CAST(kind AS BLOB))
                        BETWEEN 1 AND {maximum_kind_bytes}
                    AND substr(kind, 1, 1) GLOB '[a-z]'
                    AND kind NOT GLOB '*[^a-z0-9._-]*'
                ),
            confidence REAL NOT NULL
                CHECK (
                    typeof(confidence) = 'real'
                    AND confidence >= 0.0
                    AND confidence <= 1.0
                ),
            payload_present INTEGER NOT NULL
                CHECK (
                    typeof(payload_present) = 'integer'
                    AND payload_present IN (0, 1)
                ),
            payload_text TEXT NOT NULL
                CHECK (
                    typeof(payload_text) = 'text'
                    AND (
                        (
                            payload_present = 0
                            AND length(CAST(payload_text AS BLOB)) = 0
                        )
                        OR (
                            payload_present = 1
                            AND length(CAST(payload_text AS BLOB))
                                BETWEEN 1 AND {maximum_payload_bytes}
                        )
                    )
                ),
            PRIMARY KEY (
                content_key,
                analyzer_identity,
                analyzer_version,
                start_seconds,
                end_seconds,
                kind,
                confidence,
                payload_present,
                payload_text
            ),
            FOREIGN KEY (
                content_key,
                analyzer_identity,
                analyzer_version
            ) REFERENCES moment_batches (
                content_key,
                analyzer_identity,
                analyzer_version
            ) ON DELETE CASCADE
        ) WITHOUT ROWID
        ",
        content_key_bytes = MEDIA_CONTENT_KEY_BYTES,
        maximum_identity_bytes = MAX_ANALYZER_IDENTITY_BYTES,
        maximum_analyzer_version = u32::MAX,
        maximum_time = MAX_MOMENT_TIME_SECONDS,
        maximum_kind_bytes = MAX_MOMENT_KIND_BYTES,
        maximum_payload_bytes = MAX_MOMENT_PAYLOAD_BYTES,
    )
}

fn create_query_index_sql() -> String {
    "
        CREATE INDEX moments_query_order ON moments (
            content_key,
            start_seconds,
            end_seconds,
            kind,
            analyzer_identity,
            analyzer_version,
            confidence DESC,
            payload_present,
            payload_text
        )
    "
    .to_owned()
}

fn hydrate_record(row: &Row<'_>, row_index: usize) -> Result<MomentRecord, MomentsIndexError> {
    let content_key =
        MediaContentKey::from_slice(blob_column(row, 0, row_index, "content_key")?)
            .map_err(|_| corrupt_row(row_index, "content_key", "invalid media content key"))?;

    let start_seconds = real_column(row, 1, row_index, "start_seconds")?;
    let end_seconds = real_column(row, 2, row_index, "end_seconds")?;
    let span = TimeSpan::new(start_seconds, end_seconds)
        .map_err(|_| corrupt_row(row_index, "time_span", "invalid time boundaries"))?;

    let kind_text = text_column(row, 3, row_index, "kind")?;
    let kind = MomentKind::new(kind_text)
        .map_err(|_| corrupt_row(row_index, "kind", "invalid moment kind"))?;

    let stored_confidence = real_column(row, 4, row_index, "confidence")?;
    if !stored_confidence.is_finite() || !(0.0..=1.0).contains(&stored_confidence) {
        return Err(corrupt_row(
            row_index,
            "confidence",
            "value is outside the finite domain",
        ));
    }
    #[allow(clippy::cast_possible_truncation)]
    let confidence_f32 = stored_confidence as f32;
    if f64::from(confidence_f32).to_bits() != stored_confidence.to_bits() {
        return Err(corrupt_row(
            row_index,
            "confidence",
            "value is not an exact f32",
        ));
    }
    let confidence = MomentConfidence::new(confidence_f32)
        .map_err(|_| corrupt_row(row_index, "confidence", "invalid confidence"))?;

    let analyzer_text = text_column(row, 5, row_index, "analyzer_identity")?;
    let analyzer_identity = AnalyzerIdentity::new(analyzer_text)
        .map_err(|_| corrupt_row(row_index, "analyzer_identity", "invalid analyzer identity"))?;

    let stored_version = integer_column(row, 6, row_index, "analyzer_version")?;
    let version_u32 = u32::try_from(stored_version)
        .map_err(|_| corrupt_row(row_index, "analyzer_version", "value is outside u32"))?;
    let analyzer_version = AnalyzerVersion::new(version_u32)
        .map_err(|_| corrupt_row(row_index, "analyzer_version", "version must be non-zero"))?;

    let payload_present = integer_column(row, 7, row_index, "payload_present")?;
    let payload_text = text_column(row, 8, row_index, "payload_text")?;
    let payload = match payload_present {
        0 if payload_text.is_empty() => None,
        0 => {
            return Err(corrupt_row(
                row_index,
                "payload_text",
                "absent payload must use the empty sentinel",
            ));
        }
        1 => Some(
            MomentPayload::new(payload_text)
                .map_err(|_| corrupt_row(row_index, "payload_text", "invalid moment payload"))?,
        ),
        _ => {
            return Err(corrupt_row(
                row_index,
                "payload_present",
                "marker must be zero or one",
            ));
        }
    };

    Ok(MomentRecord::new(
        content_key,
        span,
        kind,
        confidence,
        analyzer_identity,
        analyzer_version,
        payload,
    ))
}

fn blob_column<'row>(
    row: &'row Row<'_>,
    column: usize,
    row_index: usize,
    field: &'static str,
) -> Result<&'row [u8], MomentsIndexError> {
    match row
        .get_ref(column)
        .map_err(|_| corrupt_row(row_index, field, "column could not be read"))?
    {
        ValueRef::Blob(value) => Ok(value),
        _ => Err(corrupt_row(row_index, field, "unexpected SQLite type")),
    }
}

fn text_column<'row>(
    row: &'row Row<'_>,
    column: usize,
    row_index: usize,
    field: &'static str,
) -> Result<&'row str, MomentsIndexError> {
    match row
        .get_ref(column)
        .map_err(|_| corrupt_row(row_index, field, "column could not be read"))?
    {
        ValueRef::Text(value) => str::from_utf8(value)
            .map_err(|_| corrupt_row(row_index, field, "text is not valid UTF-8")),
        _ => Err(corrupt_row(row_index, field, "unexpected SQLite type")),
    }
}

fn real_column(
    row: &Row<'_>,
    column: usize,
    row_index: usize,
    field: &'static str,
) -> Result<f64, MomentsIndexError> {
    match row
        .get_ref(column)
        .map_err(|_| corrupt_row(row_index, field, "column could not be read"))?
    {
        ValueRef::Real(value) => Ok(value),
        _ => Err(corrupt_row(row_index, field, "unexpected SQLite type")),
    }
}

fn integer_column(
    row: &Row<'_>,
    column: usize,
    row_index: usize,
    field: &'static str,
) -> Result<i64, MomentsIndexError> {
    match row
        .get_ref(column)
        .map_err(|_| corrupt_row(row_index, field, "column could not be read"))?
    {
        ValueRef::Integer(value) => Ok(value),
        _ => Err(corrupt_row(row_index, field, "unexpected SQLite type")),
    }
}

fn corrupt_row(row_index: usize, field: &'static str, reason: &'static str) -> MomentsIndexError {
    MomentsIndexError::CorruptRow {
        row_index,
        field,
        reason,
    }
}

fn map_sqlite_error(source: rusqlite::Error) -> MomentsIndexError {
    match sqlite_error_code(&source) {
        Some(
            ErrorCode::DatabaseBusy
            | ErrorCode::DatabaseLocked
            | ErrorCode::FileLockingProtocolFailed,
        ) => MomentsIndexError::DatabaseLocked { source },
        Some(ErrorCode::DatabaseCorrupt | ErrorCode::NotADatabase) => {
            MomentsIndexError::DatabaseCorrupt { source }
        }
        Some(ErrorCode::ConstraintViolation | ErrorCode::TypeMismatch | ErrorCode::TooBig) => {
            MomentsIndexError::ConstraintViolation { source }
        }
        Some(
            ErrorCode::PermissionDenied
            | ErrorCode::ReadOnly
            | ErrorCode::SystemIoFailure
            | ErrorCode::DiskFull
            | ErrorCode::CannotOpen
            | ErrorCode::NoLargeFileSupport,
        ) => MomentsIndexError::DatabaseIo { source },
        _ => MomentsIndexError::Database { source },
    }
}

fn sqlite_error_code(source: &rusqlite::Error) -> Option<ErrorCode> {
    match source {
        rusqlite::Error::SqliteFailure(error, _) => Some(error.code),
        rusqlite::Error::SqlInputError { error, .. } => Some(error.code),
        _ => None,
    }
}

fn is_locking_error(source: &rusqlite::Error) -> bool {
    matches!(
        sqlite_error_code(source),
        Some(
            ErrorCode::DatabaseBusy
                | ErrorCode::DatabaseLocked
                | ErrorCode::FileLockingProtocolFailed
        )
    )
}
