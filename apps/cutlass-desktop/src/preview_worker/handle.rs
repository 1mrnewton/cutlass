use super::*;

impl WorkerHandle {
    pub fn request_frame(&self, tick: i64) {
        let _ = self.tx.send(WorkerMsg::Frame(tick));
    }

    /// Report the preview panel's on-screen size (physical px). The worker
    /// fits every subsequent render inside it and repaints the current frame.
    pub fn set_viewport(&self, width: u32, height: u32) {
        let _ = self.tx.send(WorkerMsg::Viewport { width, height });
    }

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
    pub(super) fn project_rpc<T>(
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
    pub(super) fn preview_cache_rpc(
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
    pub(super) fn begin_project_maintenance_with_timeout(
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

    /// Synchronous round-trip: clone of the live project as of every edit
    /// sent before this call. `None` only if the worker thread is gone.
    pub fn snapshot_project(&self) -> Option<Project> {
        let (reply, rx) = bounded(1);
        self.tx.send(WorkerMsg::SnapshotProject { reply }).ok()?;
        rx.recv().ok()
    }

    /// Synchronous round-trip: replay a rehearsed agent plan, one undo
    /// entry per phase. `None` only if the worker thread is gone.
    pub fn agent_apply_plan(&self, phases: Vec<Vec<AgentPlanStep>>) -> Option<Result<(), String>> {
        let (reply, rx) = bounded(1);
        self.tx
            .send(WorkerMsg::AgentApplyPlan { phases, reply })
            .ok()?;
        rx.recv().ok()
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

    pub fn export(&self, request: ExportRequest) {
        let _ = self.tx.send(WorkerMsg::Export(request));
    }

    pub fn cancel_export(&self) {
        let _ = self.tx.send(WorkerMsg::CancelExport);
    }

    pub fn add_clip(
        &self,
        media: String,
        track: String,
        start_tick: i64,
        drop_row: i64,
        insert: bool,
    ) {
        let _ = self.tx.send(WorkerMsg::AddClip {
            media,
            track,
            start_tick,
            drop_row,
            insert,
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_generated(
        &self,
        generator: Generator,
        track: String,
        start_tick: i64,
        duration_ticks: i64,
        drop_row: i64,
        effect: Option<String>,
        animations: Vec<(String, String)>,
    ) {
        let _ = self.tx.send(WorkerMsg::AddGenerated {
            generator,
            track,
            start_tick,
            duration_ticks,
            drop_row,
            effect,
            animations,
        });
    }

    pub fn move_clip(
        &self,
        clip: String,
        track: String,
        insert_row: i64,
        start_tick: i64,
        insert: bool,
    ) {
        let _ = self.tx.send(WorkerMsg::MoveClip {
            clip,
            track,
            insert_row,
            start_tick,
            insert,
        });
    }

    pub fn move_group(&self, moves: Vec<GroupMove>) {
        let _ = self.tx.send(WorkerMsg::MoveGroup { moves });
    }

    pub fn trim_clip(&self, clip: String, start_tick: i64, duration_ticks: i64) {
        let _ = self.tx.send(WorkerMsg::TrimClip {
            clip,
            start_tick,
            duration_ticks,
        });
    }

    pub fn remove_clips(&self, clips: Vec<String>) {
        let _ = self.tx.send(WorkerMsg::RemoveClips { clips });
    }

    pub fn ripple_delete_clips(&self, clips: Vec<String>) {
        let _ = self.tx.send(WorkerMsg::RippleDeleteClips { clips });
    }

    pub fn reverse_clip(&self, clip: String) {
        let _ = self.tx.send(WorkerMsg::ReverseClip { clip });
    }

    pub fn extract_audio(&self, clip: String) {
        let _ = self.tx.send(WorkerMsg::ExtractAudio { clip });
    }

    pub fn split_clip(&self, clip: String, at_tick: i64) {
        let _ = self.tx.send(WorkerMsg::SplitClip { clip, at_tick });
    }

    pub fn add_marker(&self, at_tick: i64, name: String, color: String) {
        let _ = self.tx.send(WorkerMsg::AddMarker {
            at_tick,
            name,
            color,
        });
    }

    pub fn remove_marker(&self, marker: String) {
        let _ = self.tx.send(WorkerMsg::RemoveMarker { marker });
    }

    pub fn set_marker(&self, marker: String, at_tick: i64, name: String, color: String) {
        let _ = self.tx.send(WorkerMsg::SetMarker {
            marker,
            at_tick,
            name,
            color,
        });
    }

    pub fn remove_track_manual(&self, track: String) {
        let _ = self.tx.send(WorkerMsg::RemoveTrackManual { track });
    }

    pub fn move_track_manual(&self, track: String, index: usize) {
        let _ = self.tx.send(WorkerMsg::MoveTrackManual { track, index });
    }

    pub fn set_track_name(&self, track: String, name: String) {
        let _ = self.tx.send(WorkerMsg::SetTrackName { track, name });
    }

    pub fn set_generator(&self, clip: String, generator: Generator) {
        let _ = self.tx.send(WorkerMsg::SetGenerator { clip, generator });
    }

    pub fn set_shape_size(&self, clip: String, width: f32, height: f32) {
        let _ = self.tx.send(WorkerMsg::SetShapeSize {
            clip,
            width,
            height,
        });
    }

    pub fn set_clip_speed(&self, clip: String, num: i32, den: i32, reversed: bool) {
        let _ = self.tx.send(WorkerMsg::SetClipSpeed {
            clip,
            num,
            den,
            reversed,
        });
    }

    pub fn set_clip_pitch(&self, clip: String, preserve: bool) {
        let _ = self.tx.send(WorkerMsg::SetClipPitch { clip, preserve });
    }

    pub fn set_denoise(&self, clip: String, denoise: bool) {
        let _ = self.tx.send(WorkerMsg::SetDenoise { clip, denoise });
    }

    /// Resolve a speed-ramp preset name (CapCut speed curves, M2) and dispatch
    /// the edit. `""` / `"none"` / `"normal"` clears the ramp; an unknown name
    /// is dropped with a warning so a stray UI string can't apply garbage.
    pub fn set_speed_curve(&self, clip: String, preset: String) {
        let curve = match preset.trim() {
            "" | "none" | "normal" => None,
            name => match cutlass_models::speed_preset(name) {
                Some(curve) => Some(curve),
                None => {
                    warn!(preset = name, "set-speed-curve ignored: unknown preset");
                    return;
                }
            },
        };
        let _ = self.tx.send(WorkerMsg::SetSpeedCurve { clip, curve });
    }

    pub fn set_speed_curve_point(&self, clip: String, index: i32, value: f32) {
        let Ok(index) = usize::try_from(index) else {
            warn!(index, "set-speed-curve-point ignored: negative index");
            return;
        };
        let _ = self
            .tx
            .send(WorkerMsg::SetSpeedCurvePoint { clip, index, value });
    }

    /// Set the flat volume level + fades (CapCut's basic slider): `volume` is
    /// `Some`, flattening any envelope.
    pub fn set_clip_audio(&self, clip: String, volume: f32, fade_in_s: f32, fade_out_s: f32) {
        let _ = self.tx.send(WorkerMsg::SetClipAudio {
            clip,
            volume: Some(volume),
            fade_in_s,
            fade_out_s,
        });
    }

    /// Duck `clip` (a music clip) under the voice-tagged lanes (M8 Phase 4).
    pub fn duck_under_voice(&self, clip: String) {
        let _ = self.tx.send(WorkerMsg::DuckUnderVoice { clip });
    }

    /// Detect beat markers on `clip` (CapCut "Beat", M8 Phase 6).
    pub fn detect_beats(&self, clip: String) {
        let _ = self.tx.send(WorkerMsg::DetectBeats { clip });
    }

    /// Clear `clip`'s detected beat markers (M8 Phase 6).
    pub fn clear_beats(&self, clip: String) {
        let _ = self.tx.send(WorkerMsg::ClearBeats { clip });
    }

    /// Set only the fades, preserving the clip's gain (constant or a
    /// keyframed M8 envelope) — `volume` lowers to `None`.
    pub fn set_clip_fades(&self, clip: String, fade_in_s: f32, fade_out_s: f32) {
        let _ = self.tx.send(WorkerMsg::SetClipAudio {
            clip,
            volume: None,
            fade_in_s,
            fade_out_s,
        });
    }

    pub fn set_clip_crop(&self, clip: String, crop: CropRect, flip_h: bool, flip_v: bool) {
        let _ = self.tx.send(WorkerMsg::SetClipCrop {
            clip,
            crop,
            flip_h,
            flip_v,
        });
    }

    pub fn set_clip_filter(&self, clip: String, filter_id: String, intensity: f32) {
        let _ = self.tx.send(WorkerMsg::SetClipFilter {
            clip,
            filter_id,
            intensity,
        });
    }

    pub fn set_clip_lut(&self, clip: String, path: String, intensity: f32) {
        let _ = self.tx.send(WorkerMsg::SetClipLut {
            clip,
            path,
            intensity,
        });
    }

    pub fn set_clip_adjust(&self, clip: String, adjust: ColorAdjustments) {
        let _ = self.tx.send(WorkerMsg::SetClipAdjust { clip, adjust });
    }

    pub fn set_agent_rules(&self, rules: String) {
        let _ = self.tx.send(WorkerMsg::SetAgentRules { rules });
    }

    pub fn set_clip_animation(
        &self,
        clip: String,
        slot: String,
        animation_id: String,
        speed: f32,
        intensity: f32,
        stagger: f32,
    ) {
        let _ = self.tx.send(WorkerMsg::SetClipAnimation {
            clip,
            slot,
            animation_id,
            speed,
            intensity,
            stagger,
        });
    }

    pub fn preview_clip_look(
        &self,
        clip: String,
        filter_id: String,
        intensity: f32,
        adjust: ColorAdjustments,
        tick: i64,
    ) {
        let _ = self.tx.send(WorkerMsg::PreviewClipLook {
            clip,
            filter_id,
            intensity,
            adjust,
            tick,
        });
    }

    pub fn add_effect(&self, clip: String, effect_id: String) {
        let _ = self.tx.send(WorkerMsg::AddEffect { clip, effect_id });
    }

    pub fn remove_effect(&self, clip: String, index: u32) {
        let _ = self.tx.send(WorkerMsg::RemoveEffect { clip, index });
    }

    pub fn set_effect_param(&self, clip: String, index: u32, param: String, value: f32) {
        let _ = self.tx.send(WorkerMsg::SetEffectParam {
            clip,
            index,
            param,
            value,
        });
    }

    pub fn add_transition(&self, clip: String, transition_id: String) {
        let _ = self.tx.send(WorkerMsg::AddTransition {
            clip,
            transition_id,
        });
    }

    pub fn remove_transition(&self, clip: String) {
        let _ = self.tx.send(WorkerMsg::RemoveTransition { clip });
    }

    pub fn set_transition(&self, clip: String, duration: i64) {
        let _ = self.tx.send(WorkerMsg::SetTransition { clip, duration });
    }

    pub fn set_canvas(&self, aspect_index: i32, background: [u8; 3]) {
        let _ = self.tx.send(WorkerMsg::SetCanvas {
            aspect_index,
            background,
        });
    }

    pub fn fit_clip(&self, clip: String, fill: bool, tick: i64) {
        let _ = self.tx.send(WorkerMsg::FitClip { clip, fill, tick });
    }

    pub fn set_param_keyframe(
        &self,
        clip: String,
        param: ClipParam,
        tick: i64,
        value: ParamValue,
        easing: Easing,
    ) {
        let _ = self.tx.send(WorkerMsg::SetParamKeyframe {
            clip,
            param,
            tick,
            value,
            easing,
        });
    }

    pub fn remove_param_keyframe(&self, clip: String, param: ClipParam, tick: i64) {
        let _ = self
            .tx
            .send(WorkerMsg::RemoveParamKeyframe { clip, param, tick });
    }

    pub fn retime_keyframes(&self, clip: String, from_tick: i64, to_tick: i64) {
        let _ = self.tx.send(WorkerMsg::RetimeKeyframes {
            clip,
            from_tick,
            to_tick,
        });
    }

    pub fn remove_keyframes_at(&self, clip: String, tick: i64) {
        let _ = self.tx.send(WorkerMsg::RemoveKeyframesAt { clip, tick });
    }

    pub fn set_transform(&self, clip: String, transform: ClipTransform, tick: i64) {
        let _ = self.tx.send(WorkerMsg::SetTransform {
            clip,
            transform,
            tick,
        });
    }

    pub fn transform_override(&self, clip: String, transform: ClipTransform, tick: i64) {
        let _ = self.tx.send(WorkerMsg::TransformOverride {
            clip,
            transform,
            tick,
        });
    }

    pub fn begin_transform_gesture(&self, clip: String, tick: i64) {
        let _ = self
            .tx
            .send(WorkerMsg::BeginTransformGesture { clip, tick });
    }

    pub fn end_transform_gesture(&self) {
        let _ = self.tx.send(WorkerMsg::EndTransformGesture);
    }

    pub fn clear_transform_override(&self, tick: i64) {
        let _ = self.tx.send(WorkerMsg::ClearTransformOverride { tick });
    }

    pub fn generator_override(&self, clip: String, generator: Generator, tick: i64) {
        let _ = self.tx.send(WorkerMsg::GeneratorOverride {
            clip,
            generator,
            tick,
        });
    }

    pub fn clear_generator_override(&self, tick: i64) {
        let _ = self.tx.send(WorkerMsg::ClearGeneratorOverride { tick });
    }

    pub fn preview_shape_size(&self, clip: String, width: f32, height: f32, tick: i64) {
        let _ = self.tx.send(WorkerMsg::PreviewShapeSize {
            clip,
            width,
            height,
            tick,
        });
    }

    pub fn undo(&self) {
        let _ = self.tx.send(WorkerMsg::Undo);
    }

    pub fn redo(&self) {
        let _ = self.tx.send(WorkerMsg::Redo);
    }

    pub fn copy_clips(&self, clips: Vec<String>) {
        let _ = self.tx.send(WorkerMsg::CopyClips { clips });
    }

    pub fn paste_at(&self, tick: i64) {
        let _ = self.tx.send(WorkerMsg::PasteAt { tick });
    }

    pub fn duplicate_clips(&self, clips: Vec<String>) {
        let _ = self.tx.send(WorkerMsg::DuplicateClips { clips });
    }

    pub fn unlink_clips(&self, clips: Vec<String>) {
        let _ = self.tx.send(WorkerMsg::UnlinkClips { clips });
    }

    pub fn set_main_magnet(&self, enabled: bool) {
        let _ = self.tx.send(WorkerMsg::SetMainMagnet(enabled));
    }

    pub fn set_linkage(&self, enabled: bool) {
        let _ = self.tx.send(WorkerMsg::SetLinkage(enabled));
    }

    pub fn set_track_flag(&self, track: String, flag: TrackFlag, value: bool) {
        let _ = self.tx.send(WorkerMsg::SetTrackFlag { track, flag, value });
    }
}
