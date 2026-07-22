use super::*;

impl WorkerHandle {
    /// Return exact preview-frame cache usage, ordered with all worker
    /// requests submitted before this call.
    #[allow(dead_code)] // Public(crate) seam for the follow-up registry wiring.
    pub(crate) fn preview_cache_stats(&self) -> Result<PreviewCacheStats, String> {
        self.preview_cache_stats_with_cancel(&AtomicBool::new(false))
    }

    pub(crate) fn preview_cache_stats_with_cancel(
        &self,
        cancel: &AtomicBool,
    ) -> Result<PreviewCacheStats, String> {
        self.preview_cache_rpc(
            "stats",
            PREVIEW_CACHE_RPC_TIMEOUT,
            cancel,
            |reply, operation| WorkerMsg::GetPreviewCacheStats { reply, operation },
        )
    }

    /// Clear only composited preview frames and return their exact pre-clear
    /// usage. The cache is empty when the worker produces the reply; later
    /// queued renders may refill it.
    #[allow(dead_code)] // Public(crate) seam for the follow-up registry wiring.
    pub(crate) fn clear_preview_cache(&self) -> Result<PreviewCacheStats, String> {
        self.clear_preview_cache_with_cancel(&AtomicBool::new(false))
    }

    pub(crate) fn clear_preview_cache_with_cancel(
        &self,
        cancel: &AtomicBool,
    ) -> Result<PreviewCacheStats, String> {
        self.preview_cache_rpc(
            "clear",
            PREVIEW_CACHE_RPC_TIMEOUT,
            cancel,
            |reply, operation| WorkerMsg::ClearPreviewCache { reply, operation },
        )
    }

    #[allow(dead_code)] // Reachable through the intentionally staged APIs above.
    pub(in crate::preview_worker) fn preview_cache_rpc(
        &self,
        operation: &'static str,
        timeout: Duration,
        cancel: &AtomicBool,
        request: impl FnOnce(Sender<PreviewCacheStats>, Arc<WorkerRpcOperation>) -> WorkerMsg,
    ) -> Result<PreviewCacheStats, String> {
        if cancel.load(Ordering::Acquire) {
            return Err(format!("preview cache {operation} request was cancelled"));
        }
        let (reply, response) = bounded(1);
        let operation_state = Arc::new(WorkerRpcOperation::pending());
        self.tx
            .send(request(reply, Arc::clone(&operation_state)))
            .map_err(|_| {
                operation_state.abandon();
                format!("preview cache {operation} request failed: preview worker is not running")
            })?;

        let started = Instant::now();
        loop {
            if cancel.load(Ordering::Acquire) && operation_state.abandon() {
                return Err(format!("preview cache {operation} request was cancelled"));
            }
            let remaining = timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                let detail = if operation_state.abandon() {
                    "while still queued"
                } else {
                    "after the worker started it"
                };
                return Err(format!(
                    "preview cache {operation} request timed out {detail} after {} ms",
                    timeout.as_millis()
                ));
            }
            match response.recv_timeout(PREVIEW_CACHE_RPC_WAIT_SLICE.min(remaining)) {
                Ok(stats) => return Ok(stats),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    operation_state.abandon();
                    return Err(format!(
                        "preview cache {operation} request failed: preview worker stopped before replying"
                    ));
                }
            }
        }
    }

    /// Freeze the preview worker in queue order and return a coherent clone of
    /// its live project. Dropping the returned guard resumes message handling.
    ///
    /// This synchronous RPC must run on a background maintenance thread,
    /// never the UI thread. Acquisition is cancellation-aware and bounded;
    /// after acquisition there is deliberately no lease timeout, because a
    /// legitimate cache relocation may take arbitrarily long.
    #[allow(dead_code)] // Public(crate) seam for cache relocation wiring.
    pub(crate) fn begin_project_maintenance_with_cancel(
        &self,
        cancel: &AtomicBool,
    ) -> Result<ProjectMaintenanceGuard, String> {
        self.begin_project_maintenance_with_timeout(cancel, PROJECT_MAINTENANCE_ACQUIRE_TIMEOUT)
    }

    #[allow(dead_code)] // Testable timeout seam used by the public(crate) API.
    pub(in crate::preview_worker) fn begin_project_maintenance_with_timeout(
        &self,
        cancel: &AtomicBool,
        timeout: Duration,
    ) -> Result<ProjectMaintenanceGuard, String> {
        if cancel.load(Ordering::Acquire) {
            return Err("project maintenance request was cancelled before worker claim".into());
        }
        let (reply, response) = bounded(1);
        // The guard owns the sole sender. Its one typed action fits without
        // waiting; sender drop also wakes the worker as an ordinary resume.
        let (resume, wait_for_resume) = bounded::<ProjectMaintenanceResumeAction>(1);
        let operation = Arc::new(WorkerRpcOperation::pending());
        self.tx
            .send(WorkerMsg::BeginProjectMaintenance {
                reply,
                resume: wait_for_resume,
                operation: Arc::clone(&operation),
            })
            .map_err(|_| {
                operation.abandon();
                "project maintenance request failed: preview worker is not running".to_string()
            })?;

        let started = Instant::now();
        loop {
            // A delivered grant wins a cancellation race after worker claim.
            // Returning it transfers the only resume sender into the guard.
            match response.try_recv() {
                Ok(reply) => return project_maintenance_result(reply, resume),
                Err(TryRecvError::Disconnected) => {
                    return Err(
                        "project maintenance request failed: preview worker stopped before replying"
                            .into(),
                    );
                }
                Err(TryRecvError::Empty) => {}
            }

            if cancel.load(Ordering::Acquire) && operation.abandon() {
                return Err("project maintenance request was cancelled before worker claim".into());
            }

            let remaining = timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                // Close the smallest race between the timeout check and a
                // grant already entering the bounded response channel.
                match response.try_recv() {
                    Ok(reply) => return project_maintenance_result(reply, resume),
                    Err(TryRecvError::Disconnected) => {
                        return Err(
                            "project maintenance request failed: preview worker stopped before replying"
                                .into(),
                        );
                    }
                    Err(TryRecvError::Empty) => {}
                }
                let detail = if operation.abandon() {
                    "while still queued"
                } else {
                    "after worker claim"
                };
                return Err(format!(
                    "project maintenance request timed out {detail} after {} ms",
                    timeout.as_millis()
                ));
            }

            match response.recv_timeout(PREVIEW_CACHE_RPC_WAIT_SLICE.min(remaining)) {
                Ok(reply) => return project_maintenance_result(reply, resume),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    operation.abandon();
                    return Err(
                        "project maintenance request failed: preview worker stopped before replying"
                            .into(),
                    );
                }
            }
        }
    }
}
