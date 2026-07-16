use super::*;

pub(super) const GATE_WAIT_SLICE: Duration = Duration::from_millis(25);
pub(super) const GATE_WAIT_TIMEOUT: Duration = Duration::from_secs(10);
pub(super) const UI_WAIT_SLICE: Duration = Duration::from_millis(25);
pub(super) const UI_WAIT_TIMEOUT: Duration = Duration::from_secs(2);
pub(super) const MAX_ERROR_CHARS: usize = 240;
pub(super) const UI_OPERATION_PENDING: u8 = 0;
pub(super) const UI_OPERATION_RUNNING: u8 = 1;
pub(super) const UI_OPERATION_ABANDONED: u8 = 2;

pub(super) fn try_with_operation_gate<T>(
    gate: &Mutex<()>,
    operation: impl FnOnce() -> T,
) -> Result<T, String> {
    let _guard = gate.try_lock().map_err(|error| match error {
        TryLockError::WouldBlock => {
            "a cache operation is in progress; settings were not saved".to_string()
        }
        TryLockError::Poisoned(_) => {
            "cache operation coordination is unavailable; settings were not saved".to_string()
        }
    })?;
    Ok(operation())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn with_coordinated_disk_cache_root<T, E>(
    layout: &SharedStorageLayout,
    gate: &Mutex<()>,
    id: CacheId,
    cancelled: &dyn Fn() -> bool,
    timeout: Duration,
    wait_slice: Duration,
    operation: impl FnOnce(&Path) -> Result<T, E>,
) -> Result<T, CoordinatedCacheError<E>> {
    check_coordinated_cancelled(cancelled).map_err(CoordinatedCacheError::Coordination)?;
    if id.descriptor().kind != CacheKind::Disk {
        return Err(CoordinatedCacheError::Coordination(
            CacheCoordinationError::MemoryCache,
        ));
    }

    let _operation = acquire_operation_gate_with_cancel(gate, cancelled, timeout, wait_slice)
        .map_err(CoordinatedCacheError::Coordination)?;
    let layout = layout.lease();
    check_coordinated_cancelled(cancelled).map_err(CoordinatedCacheError::Coordination)?;
    layout.layout().validate_filesystem().map_err(|source| {
        CoordinatedCacheError::Coordination(CacheCoordinationError::InvalidLayout { source })
    })?;
    check_coordinated_cancelled(cancelled).map_err(CoordinatedCacheError::Coordination)?;

    let root = layout.resolve(id).filter(|path| path.is_absolute()).ok_or(
        CoordinatedCacheError::Coordination(CacheCoordinationError::DiskPathUnavailable),
    )?;
    check_coordinated_cancelled(cancelled).map_err(CoordinatedCacheError::Coordination)?;
    operation(&root).map_err(CoordinatedCacheError::Callback)
}

fn check_coordinated_cancelled(cancelled: &dyn Fn() -> bool) -> Result<(), CacheCoordinationError> {
    let requested = catch_unwind(AssertUnwindSafe(cancelled)).unwrap_or(true);
    if requested {
        Err(CacheCoordinationError::Cancelled)
    } else {
        Ok(())
    }
}

fn acquire_operation_gate_with_cancel<'a>(
    gate: &'a Mutex<()>,
    cancelled: &dyn Fn() -> bool,
    timeout: Duration,
    wait_slice: Duration,
) -> Result<MutexGuard<'a, ()>, CacheCoordinationError> {
    let started = Instant::now();
    loop {
        check_coordinated_cancelled(cancelled)?;
        match gate.try_lock() {
            Ok(guard) => {
                check_coordinated_cancelled(cancelled)?;
                return Ok(guard);
            }
            Err(TryLockError::Poisoned(_)) => {
                return Err(CacheCoordinationError::GateUnavailable);
            }
            Err(TryLockError::WouldBlock) => {}
        }
        let remaining = timeout.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            return Err(CacheCoordinationError::TimedOut);
        }
        std::thread::sleep(wait_slice.min(remaining));
    }
}

pub(super) fn acquire_operation_gate<'a>(
    gate: &'a Mutex<()>,
    cancel: &AtomicBool,
    timeout: Duration,
    wait_slice: Duration,
) -> Result<MutexGuard<'a, ()>, String> {
    let started = Instant::now();
    loop {
        ensure_not_cancelled(
            cancel,
            "cancelled while waiting for another cache operation",
        )?;
        match gate.try_lock() {
            Ok(guard) => return Ok(guard),
            Err(TryLockError::Poisoned(_)) => {
                return Err("cache operation gate is unavailable".into());
            }
            Err(TryLockError::WouldBlock) => {}
        }
        let remaining = timeout.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            return Err("another cache operation did not finish in time".into());
        }
        std::thread::sleep(wait_slice.min(remaining));
    }
}

pub(super) fn wait_for_ui_response<T>(
    receiver: &Receiver<Result<T, String>>,
    cancel: &AtomicBool,
    state: &AtomicU8,
    timeout: Duration,
    wait_slice: Duration,
) -> Result<T, String> {
    let started = Instant::now();
    loop {
        if cancel.load(Ordering::Acquire)
            && state
                .compare_exchange(
                    UI_OPERATION_PENDING,
                    UI_OPERATION_ABANDONED,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
        {
            return Err("cancelled while waiting for cache statistics".into());
        }
        let remaining = timeout.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            let abandoned_while_queued = state
                .compare_exchange(
                    UI_OPERATION_PENDING,
                    UI_OPERATION_ABANDONED,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok();
            return Err(if abandoned_while_queued {
                "the Cutlass UI did not respond to the cache operation in time".into()
            } else {
                "the cache UI operation started but did not finish in time".into()
            });
        }
        match receiver.recv_timeout(wait_slice.min(remaining)) {
            // Once the UI has completed an operation, return its result even
            // if cancellation raced with delivery. In particular, a clear is
            // a commit point and must not be reported as though it never ran.
            Ok(response) => return response,
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                let _ = state.compare_exchange(
                    UI_OPERATION_PENDING,
                    UI_OPERATION_ABANDONED,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
                return Err("the cache UI response channel closed".into());
            }
        }
    }
}

pub(super) fn ensure_not_cancelled(
    cancel: &AtomicBool,
    message: &'static str,
) -> Result<(), String> {
    if cancel.load(Ordering::Acquire) {
        Err(message.into())
    } else {
        Ok(())
    }
}

pub(super) fn bounded_error(prefix: &str, error: &str) -> String {
    let mut bounded: String = error.chars().take(MAX_ERROR_CHARS).collect();
    if error.chars().count() > MAX_ERROR_CHARS {
        bounded.push('…');
    }
    format!("{prefix}: {bounded}")
}

pub(super) fn bounded_message(message: &str) -> String {
    let mut bounded: String = message.chars().take(MAX_ERROR_CHARS).collect();
    if message.chars().count() > MAX_ERROR_CHARS {
        bounded.push('…');
    }
    bounded
}
