//! Shared desktop cache inventory, clearing, and coordinated relocation.
//!
//! The registry owns the live storage layout and serializes every destructive
//! cache operation. All public operations are synchronous and intended for an
//! off-UI worker thread.

mod coordination;
mod registry;
mod relocate;
#[cfg(test)]
mod tests;
mod types;

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::io;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvTimeoutError, bounded};
use cutlass_cloud::cache::DownloadCache;
use cutlass_storage::{
    CacheDescriptor, CacheId, CacheKind, CacheTier, SharedStorageLayout, StorageError,
    StorageLayout, cache_descriptors, clear_cache, measure_disk_usage, relocate_cache,
};
use serde::ser::SerializeStruct;
use serde::{Serialize, Serializer};

use crate::AppWindow;
use crate::cache_references::{CacheReferenceReport, DraftReferenceReport};
use crate::preview_worker::{PreviewCacheStats, ProjectMaintenanceGuard, WorkerHandle};

// Submodule items are imported for sibling `use super::*` visibility and for
// the crate-visible re-exports below.
#[allow(unused_imports)]
use coordination::*;
#[allow(unused_imports)]
use registry::*;
#[allow(unused_imports)]
use relocate::*;
#[allow(unused_imports)]
use types::*;

pub(crate) use registry::{CacheRegistry, cache_can_be_cleared};
pub(crate) use types::{
    CacheClearReport, CacheCoordinationError, CacheRelocationReport, CacheSnapshot,
    CoordinatedCacheError,
};
