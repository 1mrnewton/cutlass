use super::*;

/// Persist the session to its draft file. `path` is the draft's
/// `project.cutlass` (binding a freshly created draft); `None` reuses the
/// engine's current path — the debounced auto-save and the flush before a
/// session swap / close. Success refreshes the draft's name sidecar and
/// republishes the projection; failure surfaces through `session-error`. A
/// `None` flush with no bound draft (e.g. New from the launch screen over the
/// empty boot session) has nothing to persist and is a quiet no-op.
pub(super) fn save_project_and_publish(engine: &mut Engine, path: Option<PathBuf>, ui: &UiSink) {
    let _ = save_project_rpc_and_publish(engine, path, Some(ui));
}

/// Shared save implementation for fire-and-forget UI work and acknowledged
/// RPCs. A missing implicit path remains a quiet UI no-op, but is an explicit
/// RPC error. `ui` is `None` only in engine-level unit tests.
pub(super) fn save_project_rpc_and_publish(
    engine: &mut Engine,
    path: Option<PathBuf>,
    ui: Option<&UiSink>,
) -> Result<SaveProjectRpcResult, String> {
    let Some(path) = path.or_else(|| engine.project_path().cloned()) else {
        return Err("save project failed: no current project path is bound".into());
    };
    match engine.apply(Command::Project(ProjectCommand::Save {
        path: path.clone(),
    })) {
        Ok(ApplyOutcome::Saved) => {
            crate::drafts::write_meta(&path, &engine.project().name);
            if let Some(ui) = ui {
                publish_projection(engine, ui);
            }
            let actual_path = engine.project_path().cloned().ok_or_else(|| {
                format!(
                    "save reported success for {} but the engine has no current project path",
                    path.display()
                )
            })?;
            if engine.is_dirty() {
                return Err(format!(
                    "save reported success for {} but the engine remains dirty",
                    actual_path.display()
                ));
            }
            Ok(SaveProjectRpcResult {
                path: actual_path,
                dirty: false,
            })
        }
        Ok(other) => {
            let message = format!("unexpected save outcome for {}: {other:?}", path.display());
            error!("{message}");
            Err(message)
        }
        Err(e) => {
            let message = format!("save failed for {}: {e}", path.display());
            error!("{message}");
            if let Some(ui) = ui {
                publish_session_error(
                    ui,
                    format!("Couldn't save the project to {}: {e}", path.display()),
                );
            }
            Err(message)
        }
    }
}

/// Internal error contract for a whole-session operation. The worker transport
/// gets `rpc_message`; the fire-and-forget UI transport gets `ui_message`.
/// `session_replaced_in_memory` distinguishes a rejected atomic operation from
/// a post-replacement persistence/binding failure that still needs projection
/// publication and an epoch bump.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SessionReplacementError {
    pub(super) rpc_message: String,
    pub(super) ui_message: String,
    pub(super) session_replaced_in_memory: bool,
}

impl SessionReplacementError {
    fn unchanged(rpc_message: String, ui_message: String) -> Self {
        Self {
            rpc_message,
            ui_message,
            session_replaced_in_memory: false,
        }
    }

    fn after_replacement(rpc_message: String, ui_message: String) -> Self {
        Self {
            rpc_message,
            ui_message,
            session_replaced_in_memory: true,
        }
    }
}

/// An `Ok` session operation always replaced the session. An error only did
/// so when it occurred after the atomic engine replacement (currently draft
/// creation/binding after a template apply).
pub(super) fn session_was_replaced<T>(result: &Result<T, SessionReplacementError>) -> bool {
    result
        .as_ref()
        .map(|_| true)
        .unwrap_or_else(|error| error.session_replaced_in_memory)
}

pub(super) fn count_missing_media(engine: &Engine) -> usize {
    engine
        .project()
        .media_iter()
        .filter(|media| !media.path().exists())
        .count()
}

/// Finish one transport-neutral session operation. UI fire-and-forget callers
/// opt into one `session-error`; acknowledged RPC callers receive the explicit
/// error instead. A replaced in-memory session is published and bumps the
/// epoch exactly once even when its subsequent template binding failed.
pub(super) fn complete_session_replacement<T>(
    engine: &mut Engine,
    ui: Option<&UiSink>,
    result: Result<T, SessionReplacementError>,
    publish_ui_error: bool,
) -> Result<T, String> {
    if publish_ui_error && let (Some(ui), Err(error)) = (ui, &result) {
        publish_session_error(ui, error.ui_message.clone());
    }

    if session_was_replaced(&result)
        && let Some(ui) = ui
    {
        for media in engine.project().media_iter() {
            if media.path().exists() {
                register_media_with_workers(media, ui);
            }
        }
        // Engine already cleared its override lanes on replace; drop the
        // audio mixer's mirrored volume/pan map before the snapshot reopen
        // re-applies it onto the new project's mixer.
        ui.audio.clear_all_param_overrides();
        publish_projection(engine, ui);
        bump_session_epoch(ui);
    }

    result.map_err(|error| error.rpc_message)
}

/// Transport-neutral tolerant project load. Missing media is retained for the
/// relink flow, and the result reads the actual binding back from the engine.
pub(super) fn open_project_core(
    engine: &mut Engine,
    path: PathBuf,
) -> Result<OpenProjectRpcResult, SessionReplacementError> {
    match engine.apply(Command::Project(ProjectCommand::Load {
        path: path.clone(),
    })) {
        Ok(ApplyOutcome::Loaded) => {
            let Some(actual_path) = engine.project_path().cloned() else {
                let message = format!(
                    "open project outcome uncertain/partially committed: {} replaced the \
                     in-memory session, but the engine reports no bound project path",
                    path.display()
                );
                error!("{message}");
                return Err(SessionReplacementError::after_replacement(
                    message,
                    "The project was opened, but its file binding couldn't be confirmed.".into(),
                ));
            };
            info!(
                path = %actual_path.display(),
                pool = engine.project().media_count(),
                "opened project"
            );
            Ok(OpenProjectRpcResult {
                path: actual_path,
                project_name: engine.project().name.clone(),
                missing_media_count: count_missing_media(engine),
            })
        }
        Ok(other) => {
            let message = format!("unexpected open outcome for {}: {other:?}", path.display());
            error!("{message}");
            Err(SessionReplacementError::unchanged(
                message,
                format!(
                    "Couldn't open {}: unexpected engine outcome {other:?}",
                    path.display()
                ),
            ))
        }
        Err(e) => {
            let message = format!("open project failed for {}: {e}", path.display());
            error!("{message}");
            Err(SessionReplacementError::unchanged(
                message,
                format!("Couldn't open {}: {e}", path.display()),
            ))
        }
    }
}

/// Fire-and-forget UI wrapper: errors are published exactly once.
pub(super) fn open_project_and_publish(engine: &mut Engine, path: PathBuf, ui: &UiSink) {
    let result = open_project_core(engine, path);
    let _ = complete_session_replacement(engine, Some(ui), result, true);
}

/// Acknowledged wrapper: success still updates the live UI, while errors are
/// returned to the RPC caller and never duplicated through `session-error`.
pub(super) fn open_project_rpc_and_publish(
    engine: &mut Engine,
    path: PathBuf,
    ui: Option<&UiSink>,
) -> Result<OpenProjectRpcResult, String> {
    let result = open_project_core(engine, path);
    complete_session_replacement(engine, ui, result, false)
}

/// Transport-neutral fresh-session replacement. It intentionally remains
/// unbound: the host owns app-draft creation and the subsequent queue-ordered
/// acknowledged save.
pub(super) fn new_project_core(
    engine: &mut Engine,
) -> Result<NewProjectRpcResult, SessionReplacementError> {
    engine.new_session();
    info!("new session");
    let path = engine.project_path().cloned();
    Ok(NewProjectRpcResult {
        requires_save_binding: path.is_none(),
        path,
        project_name: engine.project().name.clone(),
        missing_media_count: count_missing_media(engine),
    })
}

pub(super) fn new_project_and_publish(engine: &mut Engine, ui: &UiSink) {
    let result = new_project_core(engine);
    let _ = complete_session_replacement(engine, Some(ui), result, true);
}

pub(super) fn new_project_rpc_and_publish(
    engine: &mut Engine,
    ui: Option<&UiSink>,
) -> Result<NewProjectRpcResult, String> {
    let result = new_project_core(engine);
    complete_session_replacement(engine, ui, result, false)
}

/// Transport-neutral template mutation and app-draft binding. The injected
/// creator is the production draft allocator and a filesystem-failure seam for
/// tests; it runs only after the engine atomically applied the template.
pub(super) fn apply_template_core(
    engine: &mut Engine,
    path: PathBuf,
    picks: Vec<TemplatePick>,
    create_draft: impl FnOnce() -> std::io::Result<PathBuf>,
) -> Result<ApplyTemplateRpcResult, SessionReplacementError> {
    match engine.apply(Command::Project(ProjectCommand::ApplyTemplate {
        path: path.clone(),
        picks,
    })) {
        Ok(ApplyOutcome::AppliedTemplate) => {
            info!(
                template = %path.display(),
                pool = engine.project().media_count(),
                "applied template"
            );
        }
        Ok(other) => {
            let message = format!(
                "unexpected apply-template outcome for {}: {other:?}",
                path.display()
            );
            error!("{message}");
            return Err(SessionReplacementError::unchanged(
                message,
                format!("Couldn't use the template: unexpected engine outcome {other:?}"),
            ));
        }
        Err(e) => {
            let message = format!("apply template failed for {}: {e}", path.display());
            error!("{message}");
            return Err(SessionReplacementError::unchanged(
                message,
                format!("Couldn't use the template: {e}"),
            ));
        }
    }

    let draft = create_draft().map_err(|error| {
        let message = format!(
            "apply template outcome uncertain/partially committed: {} replaced the in-memory \
             session, but creating an app-owned draft failed: {error}; the current session is \
             unbound and has not been persisted",
            path.display()
        );
        error!("{message}");
        SessionReplacementError::after_replacement(
            message,
            format!("The template was applied but a project draft couldn't be created: {error}"),
        )
    })?;

    let saved =
        save_project_rpc_and_publish(engine, Some(draft.clone()), None).map_err(|error| {
            let binding = engine
                .project_path()
                .map(|bound| bound.display().to_string())
                .unwrap_or_else(|| "<unbound>".into());
            let message = format!(
                "apply template outcome uncertain/partially committed: {} replaced the in-memory \
             session, but binding/persisting app-owned draft {} failed or could not be verified: \
             {error}; current engine binding: {binding}",
                path.display(),
                draft.display()
            );
            error!("{message}");
            SessionReplacementError::after_replacement(
                message,
                format!(
                    "The template was applied but its project draft couldn't be saved: {error}"
                ),
            )
        })?;

    Ok(ApplyTemplateRpcResult {
        path: saved.path,
        project_name: engine.project().name.clone(),
        missing_media_count: count_missing_media(engine),
    })
}

pub(super) fn apply_template_and_publish(
    engine: &mut Engine,
    path: PathBuf,
    picks: Vec<TemplatePick>,
    ui: &UiSink,
) {
    let result = apply_template_core(engine, path, picks, crate::drafts::create);
    let _ = complete_session_replacement(engine, Some(ui), result, true);
}

pub(super) fn apply_template_rpc_and_publish(
    engine: &mut Engine,
    path: PathBuf,
    picks: Vec<TemplatePick>,
    ui: Option<&UiSink>,
) -> Result<ApplyTemplateRpcResult, String> {
    let result = apply_template_core(engine, path, picks, crate::drafts::create);
    complete_session_replacement(engine, ui, result, false)
}

/// Re-point a pool entry at a user-picked file (missing-media relink, M0).
/// The engine re-probes and swaps the entry's path/metadata in place (same
/// id — clips recover without being touched); the tile workers re-register
/// so the thumbnail and filmstrips regenerate from the new file; the
/// projection republish clears the entry's missing badge and decrements
/// the dialog's count. Failures (unreadable file, probe error) surface
/// through `session-error` and leave the entry untouched.
pub(super) fn relink_media_and_publish(engine: &mut Engine, media: &str, path: &Path, ui: &UiSink) {
    let _ = relink_media_rpc_and_publish(engine, media, path, Some(ui));
}

/// Shared single-media relink implementation for UI work and acknowledged
/// RPCs. `ui` is `None` only in engine-level unit tests.
pub(super) fn relink_media_rpc_and_publish(
    engine: &mut Engine,
    media: &str,
    path: &Path,
    ui: Option<&UiSink>,
) -> Result<RelinkMediaRpcResult, String> {
    let Some(media_id) = parse_raw_id(media).map(MediaId::from_raw) else {
        let message = format!("relink failed: unparsable media id `{media}`");
        error!("{message}");
        return Err(message);
    };
    match engine.apply(Command::Project(ProjectCommand::RelinkMedia {
        media: media_id,
        path: path.to_path_buf(),
    })) {
        Ok(ApplyOutcome::Relinked { media: relinked }) => {
            if relinked != media_id {
                let message = format!(
                    "relink for media {} returned mismatched media {}",
                    media_id.raw(),
                    relinked.raw()
                );
                error!("{message}");
                return Err(message);
            }
            info!(?relinked, path = %path.display(), "relinked media");
            let current_path = engine
                .project()
                .media(relinked)
                .map(|source| {
                    if let Some(ui) = ui {
                        register_media_with_workers(source, ui);
                    }
                    source.path().to_path_buf()
                })
                .ok_or_else(|| {
                    format!(
                        "relink succeeded but media {} is missing from the pool",
                        relinked.raw()
                    )
                })?;
            if let Some(ui) = ui {
                publish_projection(engine, ui);
            }
            Ok(RelinkMediaRpcResult {
                media_id: relinked.raw(),
                path: current_path,
            })
        }
        Ok(other) => {
            let message = format!(
                "unexpected relink outcome for media {} to {}: {other:?}",
                media_id.raw(),
                path.display()
            );
            error!("{message}");
            Err(message)
        }
        Err(e) => {
            let message = format!(
                "relink failed for media {} to {}: {e}",
                media_id.raw(),
                path.display()
            );
            error!("{message}");
            if let Some(ui) = ui {
                publish_session_error(ui, format!("Couldn't relink to {}: {e}", path.display()));
            }
            Err(message)
        }
    }
}

/// Try `folder/<filename>` for every missing pool entry; relink each match.
pub(super) fn relink_folder_and_publish(engine: &mut Engine, folder: PathBuf, ui: &UiSink) {
    let result = relink_folder_rpc_and_publish(engine, folder, Some(ui));
    report_relink_folder_error(&result, |message| publish_session_error(ui, message));
}

/// Invoke `report` exactly once for a failed UI folder relink and never for
/// success. Keeping this decision outside the shared operation gives errors a
/// single owner: fire-and-forget UI calls publish `session-error`, while RPC
/// calls return the same string to their caller without a duplicate dialog.
pub(super) fn report_relink_folder_error(
    result: &Result<RelinkFolderRpcResult, String>,
    report: impl FnOnce(String),
) {
    if let Err(message) = result {
        report(message.clone());
    }
}

/// Shared folder-relink operation for UI work and acknowledged RPCs.
///
/// `RelinkMedia` is intentionally non-undoable, so an engine history group
/// cannot make this operation atomic. Every candidate is canonicalized and
/// probed before the first mutation. The engine then re-probes while applying
/// each relink; a filesystem race can still make an individual apply fail.
/// In that case all remaining candidates are attempted, successful relinks
/// stay applied, the UI is published once, and the RPC returns an explicit
/// partial-failure error naming the retained successes. Error reporting is
/// deliberately transport-neutral: this function logs and returns errors but
/// never publishes `session-error`. `ui` supplies success-side worker
/// registration/projection effects and is `None` only in engine-level tests.
pub(super) fn relink_folder_rpc_and_publish(
    engine: &mut Engine,
    folder: PathBuf,
    ui: Option<&UiSink>,
) -> Result<RelinkFolderRpcResult, String> {
    let mut candidates: Vec<(MediaId, PathBuf)> = engine
        .project()
        .media_iter()
        .filter(|media| !media.path().exists())
        .filter_map(|media| {
            media
                .path()
                .file_name()
                .map(|name| (media.id, folder.join(name)))
        })
        .filter(|(_, candidate)| candidate.exists())
        .collect();
    candidates.sort_by_key(|(media, _)| media.raw());

    if candidates.is_empty() {
        let message = format!(
            "No missing media files were found in {}. \
             Pick individual files or choose a folder that contains them.",
            folder.display()
        );
        return Err(message);
    }

    // Validate every candidate before the first non-undoable mutation. This
    // is the same native probe RelinkMedia performs after canonicalization.
    for (media, path) in &mut candidates {
        let canonical = path.canonicalize().map_err(|e| {
            let message = format!(
                "folder relink preflight failed for media {} at {}: {e}; no media was relinked",
                media.raw(),
                path.display()
            );
            error!("{message}");
            message
        })?;
        cutlass_decoder::probe(&canonical).map_err(|e| {
            let message = format!(
                "folder relink preflight failed for media {} at {}: {e}; no media was relinked",
                media.raw(),
                canonical.display()
            );
            error!("{message}");
            message
        })?;
        *path = canonical;
    }

    let mut relinked = Vec::with_capacity(candidates.len());
    let mut failures = Vec::new();
    for (media_id, path) in candidates {
        match engine.apply(Command::Project(ProjectCommand::RelinkMedia {
            media: media_id,
            path: path.clone(),
        })) {
            Ok(ApplyOutcome::Relinked { media }) => match engine.project().media(media) {
                Some(source) => {
                    if let Some(ui) = ui {
                        register_media_with_workers(source, ui);
                    }
                    relinked.push(RelinkMediaRpcResult {
                        media_id: media.raw(),
                        path: source.path().to_path_buf(),
                    });
                }
                None => {
                    let message = format!(
                        "media {} disappeared after a successful relink to {}",
                        media.raw(),
                        path.display()
                    );
                    error!("{message}");
                    failures.push(message);
                }
            },
            Ok(other) => {
                let message = format!(
                    "unexpected folder relink outcome for media {} at {}: {other:?}",
                    media_id.raw(),
                    path.display()
                );
                error!("{message}");
                failures.push(message);
            }
            Err(e) => {
                let message = format!(
                    "folder relink failed for media {} at {}: {e}",
                    media_id.raw(),
                    path.display()
                );
                error!("{message}");
                failures.push(message);
            }
        }
    }

    relinked.sort_by_key(|entry| entry.media_id);
    if !relinked.is_empty() {
        info!(count = relinked.len(), folder = %folder.display(), "relinked media from folder");
        if let Some(ui) = ui {
            publish_projection(engine, ui);
        }
    }

    if !failures.is_empty() {
        let retained = relinked
            .iter()
            .map(|entry| entry.media_id.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let partial = if relinked.is_empty() {
            "no relinks succeeded".to_string()
        } else {
            format!("non-undoable successful relinks remain applied for media ids [{retained}]")
        };
        return Err(format!(
            "folder relink completed with individual failures; {partial}; {}",
            failures.join("; ")
        ));
    }

    Ok(RelinkFolderRpcResult { relinked })
}

/// Delete a source from the media pool (Library bin). `force` false removes
/// only unreferenced media — the engine rejects a referenced source, which the
/// UI prevents by gating on the tile's usage count. `force` true first deletes
/// every clip referencing the source and then the source, all in one history
/// group, so a single undo restores both. The thumbnail cache entry is evicted
/// on success. Lanes the cascade empties are pruned, matching the clip-delete
/// flow (`remove_clips_and_publish`).
pub(super) fn remove_media_and_publish(engine: &mut Engine, media: &str, force: bool, ui: &UiSink) {
    let Some(media_id) = parse_raw_id(media).map(MediaId::from_raw) else {
        error!(media, "delete-media ignored: unparsable media id");
        return;
    };
    if engine.project().media(media_id).is_none() {
        error!(%media_id, "delete-media ignored: not in the pool");
        return;
    }

    if !force {
        match engine.apply(Command::Project(ProjectCommand::RemoveMedia {
            media: media_id,
        })) {
            Ok(ApplyOutcome::RemovedMedia { media }) => {
                info!(?media, "removed media from pool");
                crate::thumbnails::forget(media.raw());
                publish_projection(engine, ui);
            }
            Ok(other) => error!(%media_id, "unexpected remove-media outcome: {other:?}"),
            // The UI only sends the unforced delete for an unreferenced tile,
            // so a rejection here is a race (a clip landed on it between the
            // projection and the click) — surface it instead of dropping it.
            Err(e) => {
                error!(%media_id, "remove media failed: {e}");
                publish_session_error(ui, format!("Couldn't remove the media: {e}"));
            }
        }
        return;
    }

    // Cascade: gather every clip that references the source up front (a Library
    // delete leaves gaps where the clips sat — it isn't a timeline-timing
    // edit), then remove them and the source as one undoable group.
    let mut doomed: Vec<(ClipId, TrackId)> = Vec::new();
    for track in engine.project().timeline().tracks_ordered() {
        for clip in track.clips() {
            if clip.media() == Some(media_id) {
                doomed.push((clip.id, track.id));
            }
        }
    }

    engine.begin_group();
    for &(clip_id, _) in &doomed {
        if let Err(e) = apply_edit(engine, EditCommand::RemoveClip { clip: clip_id }) {
            error!(%clip_id, "remove referencing clip failed: {e}");
            engine.rollback_group();
            publish_projection(engine, ui);
            return;
        }
    }
    if let Err(e) = engine.apply(Command::Project(ProjectCommand::RemoveMedia {
        media: media_id,
    })) {
        error!(%media_id, "remove media failed after clearing clips: {e}");
        engine.rollback_group();
        publish_projection(engine, ui);
        return;
    }
    // Prune lanes the removals emptied (CapCut drops emptied overlay tracks).
    let mut lanes: Vec<TrackId> = doomed.iter().map(|&(_, track)| track).collect();
    lanes.sort();
    lanes.dedup();
    for lane in lanes {
        remove_track_if_empty(engine, lane);
    }
    engine.commit_group();
    info!(%media_id, clips = doomed.len(), "removed media and its referencing clips");
    crate::thumbnails::forget(media_id.raw());
    publish_projection(engine, ui);
}

/// Register one pool media with the off-thread tile workers: a library
/// thumbnail render and the strip worker's id → path record (filmstrips /
/// waveforms resolve by media id alone). Shared by import, open, and relink.
pub(super) fn register_media_with_workers(media: &cutlass_models::MediaSource, ui: &UiSink) {
    let kind = match media.kind() {
        cutlass_models::MediaKind::Audio => ThumbKind::Audio,
        cutlass_models::MediaKind::Image => ThumbKind::Image,
        cutlass_models::MediaKind::Video => ThumbKind::Video,
    };
    ui.thumbs
        .request(media.id.raw(), media.path().to_path_buf(), kind);
    // Stills register too: the strip sampler repeats the one picture across
    // the clip's filmstrip tiles.
    ui.strips
        .register_media(media.id.raw(), media.path().to_path_buf());
    // Large video sources get a preview proxy encoded in the background
    // (or re-bound instantly when one is already on disk); the worker
    // skips sources small enough to decode comfortably.
    if media.kind() == cutlass_models::MediaKind::Video {
        ui.proxy.request(
            media.id.raw(),
            media.path().to_path_buf(),
            media.width,
            media.height,
        );
    }
}
