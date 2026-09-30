use super::*;

/// Shared flags between the worker loop and the export thread. `active`
/// gates one-job-at-a-time (the export thread clears it when it exits);
/// `cancel` is reset at job start and set by [`WorkerMsg::CancelExport`].
#[derive(Default)]
pub(super) struct ExportJobState {
    active: Arc<AtomicBool>,
    pub(super) cancel: Arc<AtomicBool>,
}

/// One snapshot of the export job for the Slint `ExportBackend` global.
#[derive(Default)]
pub(super) struct ExportUiState {
    running: bool,
    done: u64,
    total: u64,
    completed: bool,
    failed: bool,
    status: String,
}

pub(super) fn publish_export_state(
    weak: &slint::Weak<ExportBackend<'static>>,
    state: ExportUiState,
) {
    let weak = weak.clone();
    if let Err(e) = slint::invoke_from_event_loop(move || {
        if let Some(backend) = weak.upgrade() {
            backend.set_running(state.running);
            backend.set_frames_done(state.done.min(i32::MAX as u64) as i32);
            backend.set_frames_total(state.total.min(i32::MAX as u64) as i32);
            backend.set_progress(if state.total > 0 {
                (state.done as f32 / state.total as f32).clamp(0.0, 1.0)
            } else {
                0.0
            });
            backend.set_completed(state.completed);
            backend.set_failed(state.failed);
            backend.set_status(state.status.into());
        }
    }) {
        error!("failed to publish export state to UI: {e}");
    }
}

/// The output settings for one export request: the dialog's height preset
/// scales the composite canvas (aspect preserved), the fps preset overrides
/// the sampling rate, and everything is evened for H.264.
pub(super) fn export_settings_for(
    project: &cutlass_models::Project,
    request: &ExportRequest,
) -> ExportSettings {
    let mut settings = ExportSettings::for_project(project);
    if let Some(target_h) = request.target_height.filter(|&h| h > 0) {
        let (w, h) = settings.size;
        if h > 0 {
            let scaled_w =
                ((u64::from(w) * u64::from(target_h) + u64::from(h) / 2) / u64::from(h)) as u32;
            settings.size = (scaled_w.max(2), target_h);
        }
    }
    if let Some(num) = request.fps_num.filter(|&n| n > 0) {
        settings.frame_rate = Rational::new(num, 1);
    }
    settings.evened()
}

/// Write the project's caption cues beside the finished video, so players and
/// upload pipelines get toggleable subtitles alongside the burned-in captions.
///
/// Every group contributes its lines; groups that overlap in time interleave by
/// start, which is all a subtitle file can express.
pub(super) fn write_subtitle_sidecar(
    project: &Project,
    video: &Path,
    format: CaptionFileFormat,
) -> std::io::Result<PathBuf> {
    let timeline = project.timeline();
    let cues = cutlass_captions::subtitles_from_clips(
        timeline
            .caption_groups_ordered()
            .iter()
            .flat_map(|group| timeline.caption_cues(group.id))
            .map(|clip| {
                (
                    clip.timeline,
                    clip.text_content().unwrap_or_default().to_string(),
                )
            }),
        cutlass_captions::Placement::at_rate(timeline.frame_rate),
    );
    let text = match format {
        CaptionFileFormat::Srt => cutlass_captions::write_srt(&cues),
        CaptionFileFormat::Vtt => cutlass_captions::write_vtt(&cues),
    };
    let path = video.with_extension(format.id());
    std::fs::write(&path, text)?;
    Ok(path)
}

/// Snapshot the project and run the export on a dedicated thread: decode +
/// GPU composite + encode would otherwise freeze preview and edits for the
/// whole render. The thread brings up its own headless [`Renderer`] (own GPU
/// queue + decoder cache — the mobile `export_job.rs` pattern), publishes
/// progress to the UI at most ~10×/sec, and tears the `active` gate down
/// when it exits — whatever the outcome.
pub(super) fn start_export(
    engine: &Engine,
    ui: &UiSink,
    state: &ExportJobState,
    request: ExportRequest,
) {
    if state.active.swap(true, Ordering::SeqCst) {
        warn!("export refused: a job is already running");
        return;
    }
    state.cancel.store(false, Ordering::SeqCst);

    let project = engine.project().clone();
    let settings = export_settings_for(&project, &request);
    let export_weak = ui.export.clone();
    let active = state.active.clone();
    let cancel = state.cancel.clone();
    let path = request.path;
    let subtitles = request.subtitles;

    publish_export_state(
        &export_weak,
        ExportUiState {
            running: true,
            ..Default::default()
        },
    );

    let spawned = std::thread::Builder::new()
        .name("cutlass-export".into())
        .spawn(move || {
            info!(path = %path.display(), size = ?settings.size, "export job started");
            let weak = export_weak.clone();
            let mut last_publish = Instant::now();
            let mut published_once = false;
            let result = Renderer::new_headless().and_then(|mut renderer| {
                cutlass_render::export_to_file_observed(
                    &mut renderer,
                    &project,
                    &path,
                    settings,
                    &mut |done, total| {
                        if cancel.load(Ordering::Relaxed) {
                            return false;
                        }
                        // Throttle event-loop traffic, but always deliver the
                        // first call (the dialog learns the total) and the last.
                        if !published_once
                            || done == total
                            || last_publish.elapsed() >= Duration::from_millis(100)
                        {
                            published_once = true;
                            last_publish = Instant::now();
                            publish_export_state(
                                &weak,
                                ExportUiState {
                                    running: true,
                                    done,
                                    total,
                                    ..Default::default()
                                },
                            );
                        }
                        true
                    },
                )
            });

            let outcome = match result {
                Ok(frames) => {
                    info!(
                        frames,
                        width = settings.size.0,
                        height = settings.size.1,
                        path = %path.display(),
                        "export job finished"
                    );
                    let mut status = format!(
                        "Saved {}×{}, {} frames to {}",
                        settings.size.0,
                        settings.size.1,
                        frames,
                        path.display()
                    );
                    // The video is already on disk, so a sidecar failure is
                    // reported alongside the success rather than as one.
                    if let Some(format) = subtitles {
                        match write_subtitle_sidecar(&project, &path, format) {
                            Ok(sidecar) => {
                                info!(path = %sidecar.display(), "wrote subtitle sidecar");
                                status.push_str(&format!(
                                    " · subtitles: {}",
                                    sidecar
                                        .file_name()
                                        .unwrap_or(sidecar.as_os_str())
                                        .to_string_lossy()
                                ));
                            }
                            Err(e) => {
                                error!("failed to write subtitle sidecar: {e}");
                                status.push_str(&format!(" · subtitles failed: {e}"));
                            }
                        }
                    }
                    ExportUiState {
                        done: frames,
                        total: frames,
                        completed: true,
                        status,
                        ..Default::default()
                    }
                }
                Err(RenderError::Cancelled) => {
                    info!(path = %path.display(), "export job cancelled");
                    ExportUiState {
                        failed: true,
                        status: "Export cancelled".into(),
                        ..Default::default()
                    }
                }
                Err(e) => {
                    error!(path = %path.display(), "export job failed: {e}");
                    ExportUiState {
                        failed: true,
                        status: format!("Export failed: {e}"),
                        ..Default::default()
                    }
                }
            };
            publish_export_state(&weak, outcome);
            active.store(false, Ordering::SeqCst);
        });

    if let Err(e) = spawned {
        error!("failed to spawn export thread: {e}");
        state.active.store(false, Ordering::SeqCst);
        publish_export_state(
            &ui.export,
            ExportUiState {
                failed: true,
                status: format!("Export failed to start: {e}"),
                ..Default::default()
            },
        );
    }
}
