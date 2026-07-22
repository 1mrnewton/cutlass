use super::*;

impl WorkerHandle {
    pub fn import(&self, path: PathBuf) {
        let _ = self.tx.send(WorkerMsg::Import(path));
    }

    /// Import media in queue order and wait for an acknowledged result.
    #[allow(dead_code)] // Phase 2c foundation; consumed by agent project RPCs next.
    pub(crate) fn import_media_rpc(&self, path: PathBuf) -> Result<ImportMediaRpcResult, String> {
        self.import_media_rpc_with_cancel(path, &AtomicBool::new(false))
    }

    #[allow(dead_code)] // Phase 2c foundation; consumed by agent project RPCs next.
    pub(crate) fn import_media_rpc_with_cancel(
        &self,
        path: PathBuf,
        cancel: &AtomicBool,
    ) -> Result<ImportMediaRpcResult, String> {
        self.project_rpc(
            "import media",
            PROJECT_RPC_TIMEOUT,
            cancel,
            |reply, operation| WorkerMsg::ImportMediaRpc {
                path,
                reply,
                operation,
            },
        )
    }

    /// OS file drop: import `paths` and, when `target` names a timeline
    /// landing spot (lane row, tick), place them end-to-end from there.
    pub fn drop_files(&self, paths: Vec<PathBuf>, target: Option<(i64, i64)>) {
        let _ = self.tx.send(WorkerMsg::DropFiles { paths, target });
    }

    /// A preview proxy landed for pool media `media_id` (raw id), generated
    /// from the source file at `source`. Called from the proxy worker thread.
    pub fn proxy_ready(&self, media_id: u64, source: PathBuf, proxy: PathBuf) {
        let _ = self.tx.send(WorkerMsg::ProxyReady {
            media_id,
            source,
            proxy,
        });
    }

    pub fn save_project(&self, path: Option<PathBuf>) {
        let _ = self.tx.send(WorkerMsg::SaveProject { path });
    }

    /// Save in queue order and return the engine's actual bound path.
    #[allow(dead_code)] // Phase 2c foundation; consumed by agent project RPCs next.
    pub(crate) fn save_project_rpc(
        &self,
        path: Option<PathBuf>,
    ) -> Result<SaveProjectRpcResult, String> {
        self.save_project_rpc_with_cancel(path, &AtomicBool::new(false))
    }

    #[allow(dead_code)] // Phase 2c foundation; consumed by agent project RPCs next.
    pub(crate) fn save_project_rpc_with_cancel(
        &self,
        path: Option<PathBuf>,
        cancel: &AtomicBool,
    ) -> Result<SaveProjectRpcResult, String> {
        self.project_rpc(
            "save project",
            PROJECT_RPC_TIMEOUT,
            cancel,
            |reply, operation| WorkerMsg::SaveProjectRpc {
                path,
                reply,
                operation,
            },
        )
    }

    pub fn open_project(&self, path: PathBuf) {
        let _ = self.tx.send(WorkerMsg::OpenProject { path });
    }

    /// Open a project in queue order and return its actual bound path and
    /// acknowledged session state.
    #[allow(dead_code)] // Phase 2c foundation; consumed by agent project RPCs next.
    pub(crate) fn open_project_rpc(&self, path: PathBuf) -> Result<OpenProjectRpcResult, String> {
        self.open_project_rpc_with_cancel(path, &AtomicBool::new(false))
    }

    #[allow(dead_code)] // Phase 2c foundation; consumed by agent project RPCs next.
    pub(crate) fn open_project_rpc_with_cancel(
        &self,
        path: PathBuf,
        cancel: &AtomicBool,
    ) -> Result<OpenProjectRpcResult, String> {
        self.project_rpc(
            "open project",
            PROJECT_RPC_TIMEOUT,
            cancel,
            |reply, operation| WorkerMsg::OpenProjectRpc {
                path,
                reply,
                operation,
            },
        )
    }

    pub fn apply_template(&self, path: PathBuf, picks: Vec<TemplatePick>) {
        let _ = self.tx.send(WorkerMsg::ApplyTemplate { path, picks });
    }

    /// Apply and bind a template in queue order. `Ok` confirms the actual
    /// app-owned draft path; a post-apply binding failure is returned as an
    /// explicit uncertain/partially committed outcome.
    #[allow(dead_code)] // Phase 2c foundation; consumed by agent project RPCs next.
    pub(crate) fn apply_template_rpc(
        &self,
        path: PathBuf,
        picks: Vec<TemplatePick>,
    ) -> Result<ApplyTemplateRpcResult, String> {
        self.apply_template_rpc_with_cancel(path, picks, &AtomicBool::new(false))
    }

    #[allow(dead_code)] // Phase 2c foundation; consumed by agent project RPCs next.
    pub(crate) fn apply_template_rpc_with_cancel(
        &self,
        path: PathBuf,
        picks: Vec<TemplatePick>,
        cancel: &AtomicBool,
    ) -> Result<ApplyTemplateRpcResult, String> {
        self.project_rpc(
            "apply template",
            PROJECT_RPC_TIMEOUT,
            cancel,
            |reply, operation| WorkerMsg::ApplyTemplateRpc {
                path,
                picks,
                reply,
                operation,
            },
        )
    }

    pub fn new_project(&self) {
        let _ = self.tx.send(WorkerMsg::NewProject);
    }

    /// Replace the live session with a fresh unbound project in queue order.
    /// Binding remains a separate acknowledged save owned by the caller.
    #[allow(dead_code)] // Phase 2c foundation; consumed by agent project RPCs next.
    pub(crate) fn new_project_rpc(&self) -> Result<NewProjectRpcResult, String> {
        self.new_project_rpc_with_cancel(&AtomicBool::new(false))
    }

    #[allow(dead_code)] // Phase 2c foundation; consumed by agent project RPCs next.
    pub(crate) fn new_project_rpc_with_cancel(
        &self,
        cancel: &AtomicBool,
    ) -> Result<NewProjectRpcResult, String> {
        self.project_rpc(
            "new project",
            PROJECT_RPC_TIMEOUT,
            cancel,
            |reply, operation| WorkerMsg::NewProjectRpc { reply, operation },
        )
    }

    pub fn rename_project(&self, name: String) {
        let _ = self.tx.send(WorkerMsg::RenameProject { name });
    }

    /// Send one queue-ordered project mutation and wait for its bounded reply.
    ///
    /// Cancellation abandons only a still-pending request, which guarantees
    /// the worker cannot later claim or mutate for it. If the worker has
    /// already claimed the request, cancellation no longer returns early:
    /// the actual operation result wins whenever it arrives before `timeout`.
    /// A timeout or disconnect after claim says "outcome unknown" explicitly,
    /// because the mutation may have happened even if its reply was lost.
    pub(in crate::preview_worker) fn project_rpc<T>(
        &self,
        name: &'static str,
        timeout: Duration,
        cancel: &AtomicBool,
        request: impl FnOnce(Sender<Result<T, String>>, Arc<WorkerRpcOperation>) -> WorkerMsg,
    ) -> Result<T, String> {
        if cancel.load(Ordering::Acquire) {
            return Err(format!(
                "{name} request cancelled before worker claim; not started"
            ));
        }
        let (reply, response) = bounded(1);
        let operation = Arc::new(WorkerRpcOperation::pending());
        self.tx
            .send(request(reply, Arc::clone(&operation)))
            .map_err(|_| {
                operation.abandon();
                format!("{name} request failed: preview worker is not running; not started")
            })?;

        let started = Instant::now();
        loop {
            // A delivered result always wins cancellation/deadline races.
            match response.try_recv() {
                Ok(result) => return result,
                Err(TryRecvError::Disconnected) => {
                    let detail = if operation.abandon() {
                        "not started"
                    } else {
                        "outcome unknown after worker claim"
                    };
                    return Err(format!(
                        "{name} request failed: preview worker stopped before replying; {detail}"
                    ));
                }
                Err(TryRecvError::Empty) => {}
            }

            if cancel.load(Ordering::Acquire) && operation.abandon() {
                return Err(format!(
                    "{name} request cancelled before worker claim; not started"
                ));
            }

            let remaining = timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                // Close the race where a reply entered the channel between
                // the first try_recv and the deadline observation.
                match response.try_recv() {
                    Ok(result) => return result,
                    Err(TryRecvError::Disconnected) => {
                        let detail = if operation.abandon() {
                            "not started"
                        } else {
                            "outcome unknown after worker claim"
                        };
                        return Err(format!(
                            "{name} request failed: preview worker stopped before replying; {detail}"
                        ));
                    }
                    Err(TryRecvError::Empty) => {}
                }
                let detail = if operation.abandon() {
                    "before worker claim; not started"
                } else {
                    "after worker claim; outcome unknown"
                };
                return Err(format!(
                    "{name} request timed out {detail} after {} ms",
                    timeout.as_millis()
                ));
            }

            match response.recv_timeout(PREVIEW_CACHE_RPC_WAIT_SLICE.min(remaining)) {
                Ok(result) => return result,
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    let detail = if operation.abandon() {
                        "not started"
                    } else {
                        "outcome unknown after worker claim"
                    };
                    return Err(format!(
                        "{name} request failed: preview worker stopped before replying; {detail}"
                    ));
                }
            }
        }
    }

    pub fn relink_media(&self, media: String, path: PathBuf) {
        let _ = self.tx.send(WorkerMsg::RelinkMedia { media, path });
    }

    /// Relink one pool entry in queue order and return its resulting path.
    #[allow(dead_code)] // Phase 2c foundation; consumed by agent project RPCs next.
    pub(crate) fn relink_media_rpc(
        &self,
        media: String,
        path: PathBuf,
    ) -> Result<RelinkMediaRpcResult, String> {
        self.relink_media_rpc_with_cancel(media, path, &AtomicBool::new(false))
    }

    #[allow(dead_code)] // Phase 2c foundation; consumed by agent project RPCs next.
    pub(crate) fn relink_media_rpc_with_cancel(
        &self,
        media: String,
        path: PathBuf,
        cancel: &AtomicBool,
    ) -> Result<RelinkMediaRpcResult, String> {
        self.project_rpc(
            "relink media",
            PROJECT_RPC_TIMEOUT,
            cancel,
            |reply, operation| WorkerMsg::RelinkMediaRpc {
                media,
                path,
                reply,
                operation,
            },
        )
    }

    pub fn relink_folder(&self, folder: PathBuf) {
        let _ = self.tx.send(WorkerMsg::RelinkFolder { folder });
    }

    /// Relink matching missing media in queue order. The returned entries are
    /// sorted by raw media id.
    #[allow(dead_code)] // Phase 2c foundation; consumed by agent project RPCs next.
    pub(crate) fn relink_folder_rpc(
        &self,
        folder: PathBuf,
    ) -> Result<RelinkFolderRpcResult, String> {
        self.relink_folder_rpc_with_cancel(folder, &AtomicBool::new(false))
    }

    #[allow(dead_code)] // Phase 2c foundation; consumed by agent project RPCs next.
    pub(crate) fn relink_folder_rpc_with_cancel(
        &self,
        folder: PathBuf,
        cancel: &AtomicBool,
    ) -> Result<RelinkFolderRpcResult, String> {
        self.project_rpc(
            "relink folder",
            PROJECT_RPC_TIMEOUT,
            cancel,
            |reply, operation| WorkerMsg::RelinkFolderRpc {
                folder,
                reply,
                operation,
            },
        )
    }

    pub fn remove_media(&self, media: String, force: bool) {
        let _ = self.tx.send(WorkerMsg::RemoveMedia { media, force });
    }
}
