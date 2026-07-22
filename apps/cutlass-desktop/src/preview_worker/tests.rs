use super::*;
use cutlass_models::AudioRole;
use cutlass_models::{Project, Template, TemplateMeta};

use std::sync::atomic::Ordering;
fn cache_key(tick: i64) -> FrameKey {
    FrameKey {
        tick,
        revision: 1,
        bound: Some((320, 180)),
    }
}

fn cache_buffer(width: u32, height: u32) -> SharedPixelBuffer<Rgba8Pixel> {
    SharedPixelBuffer::new(width, height)
}

fn image_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/cutlass-decoder/tests/fixtures/halves.jpg")
}

fn copy_image_fixture(path: &Path) {
    std::fs::copy(image_fixture(), path).expect("copy image fixture");
}

fn write_template_fixture(path: &Path, name: &str) {
    let mut project = Project::new(name, Rational::FPS_24);
    project.add_track(TrackKind::Video, "Main");
    Template::from_project(project, TemplateMeta::new(name))
        .save_to_file(path)
        .expect("write template fixture");
}

#[test]
fn frame_cache_reports_exact_insert_and_replacement_accounting() {
    let cache = FrameCache::default();
    let first = cache_key(1);
    let second = cache_key(2);

    cache.insert(first, cache_buffer(2, 3));
    assert_eq!(
        cache.stats(),
        PreviewCacheStats {
            entries: 1,
            bytes: 24,
        }
    );

    cache.insert(second, cache_buffer(1, 4));
    assert_eq!(
        cache.stats(),
        PreviewCacheStats {
            entries: 2,
            bytes: 40,
        }
    );

    // Replacing a key subtracts the old allocation before adding the new
    // one; cache hits only update LRU order.
    cache.insert(first, cache_buffer(3, 3));
    assert!(cache.get(&second).is_some());
    assert_eq!(
        cache.stats(),
        PreviewCacheStats {
            entries: 2,
            bytes: 52,
        }
    );
}

#[test]
fn frame_cache_clear_returns_removed_usage_and_resets_all_state() {
    let cache = FrameCache::default();
    let key = cache_key(1);
    cache.insert(key, cache_buffer(4, 2));
    assert!(cache.get(&key).is_some());
    assert!(cache.clock.get() > 1);

    assert_eq!(
        cache.clear(),
        PreviewCacheStats {
            entries: 1,
            bytes: 32,
        }
    );
    assert_eq!(cache.stats(), PreviewCacheStats::default());
    assert!(cache.entries.borrow().is_empty());
    assert_eq!(cache.bytes.get(), 0);
    assert_eq!(cache.clock.get(), 0);

    cache.insert(cache_key(2), cache_buffer(1, 1));
    assert_eq!(cache.clock.get(), 1, "LRU order restarts after clear");
}

#[test]
fn project_rpc_reports_a_disconnected_worker_as_not_started() {
    let (tx, rx) = unbounded();
    drop(rx);
    let handle = WorkerHandle { tx };

    assert_eq!(
        handle
            .import_media_rpc(PathBuf::from("/unused/disconnected.jpg"))
            .unwrap_err(),
        "import media request failed: preview worker is not running; not started"
    );
}

#[test]
fn project_rpc_pre_cancel_does_not_enqueue() {
    let (tx, rx) = unbounded();
    let handle = WorkerHandle { tx };
    let cancel = AtomicBool::new(true);

    assert_eq!(
        handle
            .save_project_rpc_with_cancel(None, &cancel)
            .unwrap_err(),
        "save project request cancelled before worker claim; not started"
    );
    assert!(
        matches!(rx.try_recv(), Err(TryRecvError::Empty)),
        "pre-cancelled RPC must not enqueue a worker message"
    );
}

#[test]
fn session_replacement_rpc_pre_cancel_does_not_enqueue() {
    let (tx, rx) = unbounded();
    let handle = WorkerHandle { tx };
    let cancel = AtomicBool::new(true);

    assert_eq!(
        handle
            .open_project_rpc_with_cancel(PathBuf::from("/unused/project.cutlass"), &cancel)
            .unwrap_err(),
        "open project request cancelled before worker claim; not started"
    );
    assert_eq!(
        handle.new_project_rpc_with_cancel(&cancel).unwrap_err(),
        "new project request cancelled before worker claim; not started"
    );
    assert_eq!(
        handle
            .apply_template_rpc_with_cancel(
                PathBuf::from("/unused/template.cutlasst"),
                Vec::new(),
                &cancel,
            )
            .unwrap_err(),
        "apply template request cancelled before worker claim; not started"
    );
    assert!(
        matches!(rx.try_recv(), Err(TryRecvError::Empty)),
        "pre-cancelled session RPCs must not enqueue worker messages"
    );
}

#[test]
fn project_rpc_cancellation_abandons_an_enqueued_pending_request() {
    let (tx, rx) = unbounded();
    let handle = WorkerHandle { tx };
    let cancel = Arc::new(AtomicBool::new(false));
    let caller_cancel = Arc::clone(&cancel);
    let (done_tx, done_rx) = bounded(1);
    let caller = std::thread::spawn(move || {
        let result = handle.import_media_rpc_with_cancel(
            PathBuf::from("/unused/pending.jpg"),
            caller_cancel.as_ref(),
        );
        done_tx.send(result).unwrap();
    });

    let WorkerMsg::ImportMediaRpc {
        reply, operation, ..
    } = rx.recv().unwrap()
    else {
        panic!("expected enqueued import RPC");
    };
    cancel.store(true, Ordering::Release);
    assert_eq!(
        done_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap_err(),
        "import media request cancelled before worker claim; not started"
    );
    assert!(
        !operation.claim(),
        "cancellation must atomically abandon the pending operation"
    );

    let mut mutated = false;
    serve_worker_rpc(reply, operation, || {
        mutated = true;
        Ok(ImportMediaRpcResult {
            media_id: 1,
            path: PathBuf::from("/unused/should-not-run.jpg"),
        })
    });
    assert!(!mutated, "an abandoned handler must remain a no-op");
    caller.join().unwrap();
}

#[test]
fn project_rpc_queue_timeout_prevents_worker_claim() {
    let (tx, rx) = unbounded();
    let handle = WorkerHandle { tx };
    let cancel = AtomicBool::new(false);

    let result: Result<ImportMediaRpcResult, String> = handle.project_rpc(
        "import media",
        Duration::ZERO,
        &cancel,
        |reply, operation| WorkerMsg::ImportMediaRpc {
            path: PathBuf::from("/unused/timeout.jpg"),
            reply,
            operation,
        },
    );
    assert_eq!(
        result.unwrap_err(),
        "import media request timed out before worker claim; not started after 0 ms"
    );
    let WorkerMsg::ImportMediaRpc { operation, .. } = rx.try_recv().unwrap() else {
        panic!("expected queued import RPC");
    };
    assert!(
        !operation.claim(),
        "timed-out queued RPC must remain abandoned"
    );
}

#[test]
fn project_rpc_post_claim_timeout_reports_outcome_unknown() {
    let (tx, rx) = unbounded();
    let handle = WorkerHandle { tx };
    let cancel = AtomicBool::new(false);

    let result: Result<ImportMediaRpcResult, String> = handle.project_rpc(
        "import media",
        Duration::ZERO,
        &cancel,
        |reply, operation| {
            assert!(operation.claim(), "simulate a worker claim before waiting");
            WorkerMsg::ImportMediaRpc {
                path: PathBuf::from("/unused/claimed-timeout.jpg"),
                reply,
                operation,
            }
        },
    );
    assert_eq!(
        result.unwrap_err(),
        "import media request timed out after worker claim; outcome unknown after 0 ms"
    );

    let WorkerMsg::ImportMediaRpc { operation, .. } = rx.try_recv().unwrap() else {
        panic!("expected claimed import RPC");
    };
    assert!(!operation.abandon());
}

#[test]
fn project_rpc_claim_ignores_late_cancellation_and_delivers_result() {
    let (tx, rx) = unbounded();
    let handle = WorkerHandle { tx };
    let cancel = Arc::new(AtomicBool::new(false));
    let caller_cancel = Arc::clone(&cancel);
    let (done_tx, done_rx) = bounded(1);
    let caller = std::thread::spawn(move || {
        let result = handle.import_media_rpc_with_cancel(
            PathBuf::from("/unused/claimed.jpg"),
            caller_cancel.as_ref(),
        );
        done_tx.send(result).unwrap();
    });

    let WorkerMsg::ImportMediaRpc {
        reply, operation, ..
    } = rx.recv().unwrap()
    else {
        panic!("expected import RPC");
    };
    assert!(operation.claim());
    cancel.store(true, Ordering::Release);
    assert!(matches!(done_rx.try_recv(), Err(TryRecvError::Empty)));

    let expected = ImportMediaRpcResult {
        media_id: 41,
        path: PathBuf::from("/canonical/claimed.jpg"),
    };
    reply.send(Ok(expected.clone())).unwrap();
    assert_eq!(
        done_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap(),
        expected
    );
    caller.join().unwrap();
}

#[test]
fn abandoned_project_rpc_handler_never_runs_mutation() {
    let operation = Arc::new(WorkerRpcOperation::pending());
    assert!(operation.abandon());
    let (reply, response) = bounded(1);
    let mut mutated = false;

    serve_worker_rpc(reply, operation, || {
        mutated = true;
        Ok::<_, String>(())
    });

    assert!(!mutated);
    assert!(matches!(
        response.try_recv(),
        Err(TryRecvError::Disconnected)
    ));
}

#[test]
fn project_rpc_handler_delivers_helper_success_and_error_dtos() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source.jpg");
    copy_image_fixture(&source);
    let mut engine = Engine::new(EngineConfig::default()).expect("engine");

    let (reply, response) = bounded(1);
    serve_worker_rpc(reply, Arc::new(WorkerRpcOperation::pending()), || {
        import_media_rpc_and_publish(&mut engine, &source, None)
    });
    let imported = response.recv().unwrap().expect("import DTO");
    assert_eq!(imported.path, source.canonicalize().unwrap());

    let (reply, response) = bounded(1);
    serve_worker_rpc(reply, Arc::new(WorkerRpcOperation::pending()), || {
        import_media_rpc_and_publish(&mut engine, &dir.path().join("missing.jpg"), None)
    });
    assert!(
        response
            .recv()
            .unwrap()
            .unwrap_err()
            .contains("import failed")
    );
}

#[test]
fn import_rpc_helper_returns_canonical_dto_and_preserves_errors() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source.jpg");
    copy_image_fixture(&source);
    let mut engine = Engine::new(EngineConfig::default()).expect("engine");

    let result = import_media_rpc_and_publish(&mut engine, &source, None).expect("import result");
    assert_eq!(result.path, source.canonicalize().unwrap());
    assert_eq!(
        engine
            .project()
            .media(MediaId::from_raw(result.media_id))
            .unwrap()
            .path(),
        result.path
    );

    let count = engine.project().media_count();
    let error = import_media_rpc_and_publish(&mut engine, &dir.path().join("missing.jpg"), None)
        .unwrap_err();
    assert!(error.contains("import failed"));
    assert_eq!(engine.project().media_count(), count);
}

#[test]
fn save_rpc_helper_returns_actual_clean_path_and_missing_path_error() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source.jpg");
    copy_image_fixture(&source);
    let mut engine = Engine::new(EngineConfig::default()).expect("engine");
    import_media_rpc_and_publish(&mut engine, &source, None).expect("dirty engine");
    assert!(engine.is_dirty());

    let project_path = dir.path().join("project.cutlass");
    let result = save_project_rpc_and_publish(&mut engine, Some(project_path.clone()), None)
        .expect("save result");
    assert_eq!(result.path, project_path);
    assert!(!result.dirty);
    assert!(!engine.is_dirty());

    let mut unbound = Engine::new(EngineConfig::default()).expect("unbound engine");
    assert_eq!(
        save_project_rpc_and_publish(&mut unbound, None, None).unwrap_err(),
        "save project failed: no current project path is bound"
    );
}

#[test]
fn open_project_core_returns_bound_path_name_and_missing_count() {
    let dir = tempfile::tempdir().unwrap();
    let present = dir.path().join("present.jpg");
    copy_image_fixture(&present);
    let missing = dir.path().join("missing.jpg");
    let project_path = dir.path().join("opened.cutlass");

    let mut project = Project::new("Acknowledged open", Rational::FPS_30);
    project.add_media(cutlass_models::MediaSource::image(present, 32, 32));
    project.add_media(cutlass_models::MediaSource::image(missing, 32, 32));
    project.save_to_file(&project_path).expect("save fixture");

    let mut engine = Engine::new(EngineConfig::default()).expect("engine");
    let outcome = open_project_core(&mut engine, project_path.clone());
    assert!(
        session_was_replaced(&outcome),
        "successful open must publish and bump the session epoch"
    );
    let result = outcome.expect("open result");
    assert_eq!(result.path, project_path);
    assert_eq!(result.project_name, "Acknowledged open");
    assert_eq!(result.missing_media_count, 1);
    assert_eq!(engine.project_path(), Some(&result.path));
}

#[test]
fn rejected_open_and_template_do_not_request_epoch_bump() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = Engine::with_project(
        EngineConfig::default(),
        Project::new("keep me", Rational::FPS_30),
    )
    .expect("engine");
    let revision = engine.revision();

    let open = open_project_core(&mut engine, dir.path().join("missing.cutlass"));
    assert!(!session_was_replaced(&open));
    assert!(
        open.unwrap_err()
            .rpc_message
            .contains("open project failed")
    );
    assert_eq!(engine.project().name, "keep me");
    assert_eq!(engine.revision(), revision);

    let template = apply_template_core(
        &mut engine,
        dir.path().join("missing.cutlasst"),
        Vec::new(),
        || panic!("draft creation must not run after a rejected template"),
    );
    assert!(!session_was_replaced(&template));
    assert!(
        template
            .unwrap_err()
            .rpc_message
            .contains("apply template failed")
    );
    assert_eq!(engine.project().name, "keep me");
    assert_eq!(engine.revision(), revision);
}

#[test]
fn new_project_core_reports_unbound_followup_save_state() {
    let dir = tempfile::tempdir().unwrap();
    let old_path = dir.path().join("old.cutlass");
    let mut engine = Engine::with_project(
        EngineConfig::default(),
        Project::new("old", Rational::FPS_30),
    )
    .expect("engine");
    engine
        .apply(Command::Project(ProjectCommand::Save {
            path: old_path.clone(),
        }))
        .expect("bind old session");
    assert_eq!(engine.project_path(), Some(&old_path));

    let outcome = new_project_core(&mut engine);
    assert!(session_was_replaced(&outcome));
    let result = outcome.expect("new result");
    assert_eq!(result.project_name, "untitled");
    assert_eq!(result.path, None);
    assert_eq!(result.missing_media_count, 0);
    assert!(result.requires_save_binding);
    assert!(engine.project_path().is_none());
    assert!(engine.project().timeline().main_track().is_some());
}

#[test]
fn apply_template_core_returns_verified_bound_draft() {
    let dir = tempfile::tempdir().unwrap();
    let template_path = dir.path().join("simple.cutlasst");
    write_template_fixture(&template_path, "Template result");
    let draft_dir = dir.path().join("draft");
    std::fs::create_dir(&draft_dir).unwrap();
    let draft_path = draft_dir.join("project.cutlass");
    let mut engine = Engine::new(EngineConfig::default()).expect("engine");

    let outcome = apply_template_core(&mut engine, template_path, Vec::new(), || {
        Ok(draft_path.clone())
    });
    assert!(session_was_replaced(&outcome));
    let result = outcome.expect("template result");
    assert_eq!(result.path, draft_path);
    assert_eq!(result.project_name, "Template result");
    assert_eq!(result.missing_media_count, 0);
    assert_eq!(engine.project_path(), Some(&result.path));
    assert!(!engine.is_dirty(), "acknowledged binding must be persisted");
    assert!(result.path.is_file());
}

#[test]
fn template_binding_failure_reports_partially_committed_memory_session() {
    let dir = tempfile::tempdir().unwrap();
    let template_path = dir.path().join("simple.cutlasst");
    write_template_fixture(&template_path, "Applied in memory");
    let invalid_draft_file = dir.path().join("directory-not-file");
    std::fs::create_dir(&invalid_draft_file).unwrap();
    let mut engine = Engine::with_project(
        EngineConfig::default(),
        Project::new("outgoing", Rational::FPS_30),
    )
    .expect("engine");

    let outcome = apply_template_core(&mut engine, template_path, Vec::new(), || {
        Ok(invalid_draft_file)
    });
    assert!(
        session_was_replaced(&outcome),
        "the applied in-memory session still needs publication and an epoch bump"
    );
    let error = outcome.unwrap_err();
    assert!(error.session_replaced_in_memory);
    assert!(error.rpc_message.contains("uncertain/partially committed"));
    assert!(error.rpc_message.contains("binding/persisting"));
    assert!(error.ui_message.contains("template was applied"));
    assert_eq!(engine.project().name, "Applied in memory");
    assert!(engine.project_path().is_none());
    assert!(engine.is_dirty());
}

#[test]
fn relink_media_rpc_helper_returns_current_path_and_validation_errors() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("missing.jpg");
    let target = dir.path().join("replacement.jpg");
    copy_image_fixture(&target);
    let mut project = Project::new("relink", Rational::FPS_30);
    let media = project.add_media(cutlass_models::MediaSource::image(missing, 32, 32));
    let mut engine = Engine::with_project(EngineConfig::default(), project).expect("relink engine");

    let result = relink_media_rpc_and_publish(&mut engine, &media.raw().to_string(), &target, None)
        .expect("relink result");
    assert_eq!(result.media_id, media.raw());
    assert_eq!(result.path, target.canonicalize().unwrap());
    assert_eq!(
        engine.project().media(media).unwrap().path(),
        result.path.as_path()
    );

    assert!(
        relink_media_rpc_and_publish(&mut engine, "not-an-id", &target, None)
            .unwrap_err()
            .contains("unparsable media id")
    );
    assert!(
        relink_media_rpc_and_publish(&mut engine, "999999", &target, None)
            .unwrap_err()
            .contains("relink failed")
    );
}

#[test]
fn relink_folder_rpc_helper_returns_sorted_results_and_no_candidate_error() {
    let dir = tempfile::tempdir().unwrap();
    let originals = dir.path().join("originals");
    let replacements = dir.path().join("replacements");
    std::fs::create_dir(&replacements).unwrap();
    copy_image_fixture(&replacements.join("z.jpg"));
    copy_image_fixture(&replacements.join("a.jpg"));

    let mut project = Project::new("folder relink", Rational::FPS_30);
    let z = project.add_media(cutlass_models::MediaSource::image(
        originals.join("z.jpg"),
        32,
        32,
    ));
    let a = project.add_media(cutlass_models::MediaSource::image(
        originals.join("a.jpg"),
        32,
        32,
    ));
    let mut engine = Engine::with_project(EngineConfig::default(), project).expect("folder engine");

    let result = relink_folder_rpc_and_publish(&mut engine, replacements.clone(), None)
        .expect("folder relink result");
    let ids: Vec<u64> = result.relinked.iter().map(|entry| entry.media_id).collect();
    assert_eq!(ids, {
        let mut expected = vec![z.raw(), a.raw()];
        expected.sort_unstable();
        expected
    });
    assert!(
        result
            .relinked
            .iter()
            .all(|entry| entry.path.is_absolute() && entry.path.exists())
    );

    let error = relink_folder_rpc_and_publish(&mut engine, replacements, None).unwrap_err();
    assert!(error.contains("No missing media files were found"));
}

#[test]
fn relink_folder_error_reporter_emits_each_error_once_without_rewriting() {
    let messages = [
        "No missing media files were found in /missing.",
        "folder relink preflight failed; no media was relinked",
        "folder relink completed with individual failures; non-undoable successful relinks \
         remain applied for media ids [7]; media 8 failed",
    ];
    for expected in messages {
        let result: Result<RelinkFolderRpcResult, String> = Err(expected.into());
        let mut reported = Vec::new();
        report_relink_folder_error(&result, |message| reported.push(message));
        assert_eq!(reported.len(), 1);
        assert_eq!(reported[0], expected);
    }

    let result = Ok(RelinkFolderRpcResult {
        relinked: Vec::new(),
    });
    let mut reported = Vec::new();
    report_relink_folder_error(&result, |message| reported.push(message));
    assert!(reported.is_empty());
}

#[test]
fn relink_folder_rpc_preflight_failure_mutates_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let originals = dir.path().join("originals");
    let replacements = dir.path().join("replacements");
    std::fs::create_dir(&replacements).unwrap();
    copy_image_fixture(&replacements.join("good.jpg"));
    std::fs::write(replacements.join("bad.jpg"), b"not media").unwrap();

    let mut project = Project::new("folder preflight", Rational::FPS_30);
    let good_path = originals.join("good.jpg");
    let bad_path = originals.join("bad.jpg");
    let good = project.add_media(cutlass_models::MediaSource::image(
        good_path.clone(),
        32,
        32,
    ));
    let bad = project.add_media(cutlass_models::MediaSource::image(bad_path.clone(), 32, 32));
    let mut engine =
        Engine::with_project(EngineConfig::default(), project).expect("preflight engine");

    let error = relink_folder_rpc_and_publish(&mut engine, replacements, None).unwrap_err();
    assert!(error.contains("preflight failed"));
    assert!(error.contains("no media was relinked"));
    assert_eq!(engine.project().media(good).unwrap().path(), good_path);
    assert_eq!(engine.project().media(bad).unwrap().path(), bad_path);
}

#[test]
fn acknowledged_relinks_invalidate_preview_like_ui_relinks() {
    let (media_reply, _) = bounded(1);
    let (folder_reply, _) = bounded(1);
    assert!(message_invalidates_preview(&WorkerMsg::RelinkMediaRpc {
        media: "1".into(),
        path: PathBuf::from("/unused/relink.jpg"),
        reply: media_reply,
        operation: Arc::new(WorkerRpcOperation::pending()),
    }));
    assert!(message_invalidates_preview(&WorkerMsg::RelinkFolderRpc {
        folder: PathBuf::from("/unused"),
        reply: folder_reply,
        operation: Arc::new(WorkerRpcOperation::pending()),
    }));
}

#[test]
fn look_edits_invalidate_preview() {
    assert!(message_invalidates_preview(&WorkerMsg::SetClipLut {
        clip: "1".into(),
        path: "/unused/look.cube".into(),
        intensity: 0.8,
    }));
    assert!(message_invalidates_preview(&WorkerMsg::SetClipAnimation {
        clip: "1".into(),
        slot: "in".into(),
        animation_id: "fade".into(),
        speed: 1.0,
        intensity: 1.0,
        stagger: 1.0,
    }));
    assert!(message_invalidates_preview(&WorkerMsg::SetBlendMode {
        clip: "1".into(),
        mode: "multiply".into(),
    }));
    assert!(message_invalidates_preview(&WorkerMsg::SetLayerStyles {
        clip: "1".into(),
        styles: LayerStyles {
            shadow: Some(cutlass_models::LayerShadow::default()),
            ..Default::default()
        },
    }));
}

#[test]
fn set_blend_mode_enqueues_worker_msg() {
    let (tx, rx) = unbounded();
    let handle = WorkerHandle { tx };
    handle.set_blend_mode("42".into(), "screen".into());
    let WorkerMsg::SetBlendMode { clip, mode } = rx.try_recv().unwrap() else {
        panic!("expected SetBlendMode");
    };
    assert_eq!(clip, "42");
    assert_eq!(mode, "screen");
}

#[test]
fn set_layer_styles_enqueues_worker_msg() {
    let (tx, rx) = unbounded();
    let handle = WorkerHandle { tx };
    let styles = LayerStyles {
        shadow: Some(cutlass_models::LayerShadow::default()),
        ..Default::default()
    };
    handle.set_layer_styles("42".into(), styles.clone());
    let WorkerMsg::SetLayerStyles { clip, styles: got } = rx.try_recv().unwrap() else {
        panic!("expected SetLayerStyles");
    };
    assert_eq!(clip, "42");
    assert_eq!(got, styles);
}

#[test]
fn preview_clip_styles_enqueues_worker_msg() {
    let (tx, rx) = unbounded();
    let handle = WorkerHandle { tx };
    let styles = LayerStyles {
        shadow: Some(cutlass_models::LayerShadow {
            blur: cutlass_models::Param::Constant(24.0),
            ..cutlass_models::LayerShadow::default()
        }),
        ..Default::default()
    };
    handle.preview_clip_styles("42".into(), styles.clone(), 12);
    let WorkerMsg::PreviewClipStyles {
        clip,
        styles: got,
        tick,
    } = rx.try_recv().unwrap()
    else {
        panic!("expected PreviewClipStyles");
    };
    assert_eq!(clip, "42");
    assert_eq!(got, styles);
    assert_eq!(tick, 12);

    handle.clear_styles_override(7);
    let WorkerMsg::ClearStylesOverride { tick } = rx.try_recv().unwrap() else {
        panic!("expected ClearStylesOverride");
    };
    assert_eq!(tick, 7);
}

#[test]
fn styles_override_previews_then_clears_on_commit() {
    use cutlass_models::{LayerShadow, Param};
    use cutlass_render::{ResolveOverrides, resolve, resolve_with};

    let r = Rational::FPS_24;
    let mut project = Project::new("styles-preview", r);
    let media = project.add_media(cutlass_models::MediaSource::new(
        "/tmp/styles-preview.mp4",
        1920,
        1080,
        r,
        1000,
        true,
    ));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip = project
        .add_clip(
            track,
            media,
            TimeRange::at_rate(0, 48, r),
            RationalTime::new(0, r),
        )
        .expect("clip");
    project
        .set_layer_styles(
            clip,
            LayerStyles {
                shadow: Some(LayerShadow {
                    blur: Param::Constant(8.0),
                    ..LayerShadow::default()
                }),
                ..Default::default()
            },
        )
        .expect("enable shadow");
    let mut engine = Engine::with_project(EngineConfig::default(), project).expect("engine");

    let live = LayerStyles {
        shadow: Some(LayerShadow {
            blur: Param::Constant(24.0),
            ..LayerShadow::default()
        }),
        ..Default::default()
    };
    // Same path the PreviewClipStyles worker arm uses.
    apply_styles_override(&mut engine, &clip.raw().to_string(), live.clone());
    assert!(engine.has_live_overrides());

    let overrides = ResolveOverrides {
        transform: None,
        generator: None,
        look: None,
        styles: Some((clip, &live)),
    };
    let scene = resolve_with(engine.project(), RationalTime::new(0, r), overrides)
        .expect("resolve with override");
    assert!(
        (scene.layers[0]
            .styles
            .as_ref()
            .unwrap()
            .shadow
            .as_ref()
            .unwrap()
            .blur
            - 24.0)
            .abs()
            < f32::EPSILON
    );

    // Commit clears the session override (mirrors look filter/adjust).
    engine.set_styles_override(None);
    engine
        .apply(Command::Edit(EditCommand::SetClipLayerStyles {
            clip,
            styles: live,
        }))
        .expect("commit styles");
    assert!(!engine.has_live_overrides());
    let scene = resolve(engine.project(), RationalTime::new(0, r)).expect("committed");
    assert!(
        (scene.layers[0]
            .styles
            .as_ref()
            .unwrap()
            .shadow
            .as_ref()
            .unwrap()
            .blur
            - 24.0)
            .abs()
            < f32::EPSILON
    );
}

#[test]
fn set_blend_mode_updates_projected_clip() {
    use crate::projection::project_to_slint;
    use cutlass_models::BlendMode;
    use slint::Model;
    use std::collections::{HashMap, HashSet};

    let r = Rational::FPS_24;
    let mut project = Project::new("blend", r);
    let media = project.add_media(cutlass_models::MediaSource::new(
        "/tmp/blend.mp4",
        1920,
        1080,
        r,
        1000,
        true,
    ));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip = project
        .add_clip(
            track,
            media,
            TimeRange::at_rate(0, 48, r),
            RationalTime::new(0, r),
        )
        .expect("clip");
    let mut engine = Engine::with_project(EngineConfig::default(), project).expect("engine");

    // Same EditCommand the worker's set_blend_mode path applies.
    engine
        .apply(Command::Edit(EditCommand::SetClipBlendMode {
            clip,
            mode: BlendMode::Multiply,
        }))
        .expect("set blend");

    let projected = project_to_slint(engine.project(), &HashMap::new(), &HashSet::new());
    let track = projected.sequence.tracks.row_data(0).unwrap();
    let c = track.clips.row_data(0).unwrap();
    assert_eq!(c.blend_mode.as_str(), "multiply");
    assert_eq!(c.blend_label.as_str(), "Multiply");
}

#[test]
fn set_layer_styles_updates_projected_clip() {
    use crate::projection::project_to_slint;
    use cutlass_models::LayerShadow;
    use slint::Model;
    use std::collections::{HashMap, HashSet};

    let r = Rational::FPS_24;
    let mut project = Project::new("styles", r);
    let media = project.add_media(cutlass_models::MediaSource::new(
        "/tmp/styles.mp4",
        1920,
        1080,
        r,
        1000,
        true,
    ));
    let track = project.add_track(TrackKind::Video, "V1");
    let clip = project
        .add_clip(
            track,
            media,
            TimeRange::at_rate(0, 48, r),
            RationalTime::new(0, r),
        )
        .expect("clip");
    let mut engine = Engine::with_project(EngineConfig::default(), project).expect("engine");

    engine
        .apply(Command::Edit(EditCommand::SetClipLayerStyles {
            clip,
            styles: LayerStyles {
                shadow: Some(LayerShadow::default()),
                ..Default::default()
            },
        }))
        .expect("enable shadow");

    let projected = project_to_slint(engine.project(), &HashMap::new(), &HashSet::new());
    let track = projected.sequence.tracks.row_data(0).unwrap();
    let c = track.clips.row_data(0).unwrap();
    assert!(c.style_shadow_enabled);
    assert!((c.style_shadow_blur - 8.0).abs() < f32::EPSILON);

    engine
        .apply(Command::Edit(EditCommand::SetClipLayerStyles {
            clip,
            styles: LayerStyles::default(),
        }))
        .expect("clear styles");

    let projected = project_to_slint(engine.project(), &HashMap::new(), &HashSet::new());
    let track = projected.sequence.tracks.row_data(0).unwrap();
    let c = track.clips.row_data(0).unwrap();
    assert!(!c.style_shadow_enabled);
}

#[test]
fn acknowledged_session_replacements_invalidate_preview() {
    let (open_reply, _) = bounded(1);
    let (new_reply, _) = bounded(1);
    let (template_reply, _) = bounded(1);
    assert!(message_invalidates_preview(&WorkerMsg::OpenProjectRpc {
        path: PathBuf::from("/unused/project.cutlass"),
        reply: open_reply,
        operation: Arc::new(WorkerRpcOperation::pending()),
    }));
    assert!(message_invalidates_preview(&WorkerMsg::NewProjectRpc {
        reply: new_reply,
        operation: Arc::new(WorkerRpcOperation::pending()),
    }));
    assert!(message_invalidates_preview(&WorkerMsg::ApplyTemplateRpc {
        path: PathBuf::from("/unused/template.cutlasst"),
        picks: Vec::new(),
        reply: template_reply,
        operation: Arc::new(WorkerRpcOperation::pending()),
    }));
}

#[test]
fn preview_cache_rpc_reports_a_disconnected_worker() {
    let (tx, rx) = unbounded();
    drop(rx);
    let handle = WorkerHandle { tx };

    assert_eq!(
        handle.preview_cache_stats().unwrap_err(),
        "preview cache stats request failed: preview worker is not running"
    );
}

#[test]
fn preview_cache_rpc_times_out_when_worker_does_not_reply() {
    let (tx, rx) = unbounded();
    let handle = WorkerHandle { tx };
    let cancel = AtomicBool::new(false);

    let error = handle
        .preview_cache_rpc(
            "clear",
            Duration::from_millis(1),
            &cancel,
            |reply, operation| WorkerMsg::ClearPreviewCache { reply, operation },
        )
        .unwrap_err();
    assert_eq!(
        error,
        "preview cache clear request timed out while still queued after 1 ms"
    );
    let WorkerMsg::ClearPreviewCache { operation, .. } = rx.try_recv().unwrap() else {
        panic!("expected queued clear request");
    };
    assert!(
        !operation.claim(),
        "a timed-out queued clear must remain a no-op"
    );
}

#[test]
fn preview_cache_rpc_pre_cancel_does_not_enqueue() {
    let (tx, rx) = unbounded();
    let handle = WorkerHandle { tx };
    let cancel = AtomicBool::new(true);

    assert_eq!(
        handle.clear_preview_cache_with_cancel(&cancel).unwrap_err(),
        "preview cache clear request was cancelled"
    );
    assert!(
        matches!(rx.try_recv(), Err(TryRecvError::Empty)),
        "pre-cancelled cache RPC must not enqueue a worker message"
    );
}

#[test]
fn project_maintenance_pre_cancel_does_not_enqueue() {
    let (tx, rx) = unbounded();
    let handle = WorkerHandle { tx };
    let cancel = AtomicBool::new(true);

    let error = match handle.begin_project_maintenance_with_timeout(&cancel, Duration::from_secs(1))
    {
        Ok(_) => panic!("cancelled maintenance request was granted"),
        Err(error) => error,
    };
    assert_eq!(
        error,
        "project maintenance request was cancelled before worker claim"
    );
    assert!(
        matches!(rx.try_recv(), Err(TryRecvError::Empty)),
        "pre-cancelled maintenance must not enqueue a worker message"
    );
}

#[test]
fn project_maintenance_claim_wins_a_cancellation_delivery_race() {
    let (tx, rx) = unbounded();
    let handle = WorkerHandle { tx };
    let cancel = Arc::new(AtomicBool::new(false));
    let worker_cancel = Arc::clone(&cancel);
    let worker = std::thread::spawn(move || {
        let WorkerMsg::BeginProjectMaintenance {
            reply,
            resume,
            operation,
        } = rx.recv().unwrap()
        else {
            panic!("expected maintenance request");
        };
        assert!(operation.claim());
        worker_cancel.store(true, Ordering::Release);
        reply
            .send(Ok(Project::new("claimed", Rational::FPS_30)))
            .unwrap();
        assert_eq!(
            resume.recv_timeout(Duration::from_secs(1)).unwrap(),
            ProjectMaintenanceResumeAction::Resume
        );
    });

    let guard = handle
        .begin_project_maintenance_with_timeout(cancel.as_ref(), Duration::from_secs(1))
        .expect("a claim that beat cancellation must deliver its guard");
    assert_eq!(guard.project().name, "claimed");
    drop(guard);
    worker.join().unwrap();
}

#[test]
fn project_maintenance_guard_ordinary_drop_resumes_queued_work_and_is_send() {
    let (tx, rx) = unbounded();
    let handle = WorkerHandle { tx };
    let (resumed_tx, resumed_rx) = bounded(1);
    let (processed_tx, processed_rx) = bounded(1);
    let worker = std::thread::spawn(move || {
        let WorkerMsg::BeginProjectMaintenance {
            reply,
            resume,
            operation,
        } = rx.recv().unwrap()
        else {
            panic!("expected maintenance request");
        };
        let project = Project::new("live snapshot", Rational::FPS_30);
        let action = serve_project_maintenance(&project, reply, resume, operation);
        resumed_tx.send(action).unwrap();

        let WorkerMsg::RenameProject { name } = rx.recv().unwrap() else {
            panic!("expected work queued behind maintenance");
        };
        processed_tx.send(name).unwrap();
    });

    let guard = handle
        .begin_project_maintenance_with_timeout(&AtomicBool::new(false), Duration::from_secs(1))
        .expect("maintenance guard");
    assert_eq!(guard.project().name, "live snapshot");
    handle.rename_project("after maintenance".into());
    assert!(matches!(processed_rx.try_recv(), Err(TryRecvError::Empty)));

    // Moving the guard into this background holder is also a compile-time
    // assertion that ProjectMaintenanceGuard is Send.
    std::thread::spawn(move || drop(guard)).join().unwrap();
    assert_eq!(
        resumed_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("guard drop must resume worker"),
        ProjectMaintenanceResumeAction::Resume
    );
    assert_eq!(
        processed_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
        "after maintenance"
    );
    worker.join().unwrap();
}

#[test]
fn project_maintenance_guard_can_request_proxy_refresh_on_drop() {
    let (resume, wait_for_resume) = bounded(1);
    let mut guard = ProjectMaintenanceGuard {
        project: Project::new("refresh", Rational::FPS_30),
        resume: Some(resume),
        resume_action: ProjectMaintenanceResumeAction::Resume,
    };

    guard.refresh_proxies_on_resume();
    drop(guard);

    assert_eq!(
        wait_for_resume
            .recv_timeout(Duration::from_secs(1))
            .unwrap(),
        ProjectMaintenanceResumeAction::RefreshProxies
    );
}

#[test]
fn disconnected_maintenance_resume_defaults_to_ordinary_action() {
    let project = Project::new("disconnected resume", Rational::FPS_30);
    let (reply, response) = bounded(1);
    let (resume, wait_for_resume) = bounded(1);
    drop(resume);

    assert_eq!(
        serve_project_maintenance(
            &project,
            reply,
            wait_for_resume,
            Arc::new(WorkerRpcOperation::pending()),
        ),
        ProjectMaintenanceResumeAction::Resume
    );
    assert_eq!(
        response.recv().unwrap().unwrap().name,
        "disconnected resume"
    );
}

#[test]
fn proxy_refresh_plan_captures_all_media_and_requests_only_videos() {
    let mut project = Project::new("proxy refresh", Rational::FPS_30);
    let video = project.add_media(cutlass_models::MediaSource::new(
        "/media/video.mov",
        3840,
        2160,
        Rational::FPS_30,
        300,
        true,
    ));
    let audio = project.add_media(cutlass_models::MediaSource::new(
        "/media/audio.wav",
        0,
        0,
        Rational::FPS_30,
        300,
        true,
    ));
    let image = project.add_media(cutlass_models::MediaSource::image(
        "/media/still.png",
        1600,
        900,
    ));

    let plan = plan_proxy_refresh(&project);
    assert_eq!(plan.len(), 3);
    assert!(
        plan.windows(2)
            .all(|pair| pair[0].id.raw() < pair[1].id.raw())
    );

    let video = plan.iter().find(|source| source.id == video).unwrap();
    assert_eq!(video.path, PathBuf::from("/media/video.mov"));
    assert_eq!((video.width, video.height), (3840, 2160));
    assert!(video.is_video);
    assert!(
        !plan
            .iter()
            .find(|source| source.id == audio)
            .unwrap()
            .is_video
    );
    assert!(
        !plan
            .iter()
            .find(|source| source.id == image)
            .unwrap()
            .is_video
    );
    assert_eq!(plan.iter().filter(|source| source.is_video).count(), 1);
}

#[test]
fn project_maintenance_acquisition_timeout_is_finite() {
    let (tx, rx) = unbounded();
    let handle = WorkerHandle { tx };
    let (done_tx, done_rx) = bounded(1);
    let caller = std::thread::spawn(move || {
        let cancel = AtomicBool::new(false);
        let result =
            handle.begin_project_maintenance_with_timeout(&cancel, Duration::from_millis(5));
        assert!(done_tx.send(result).is_ok());
    });

    let result = done_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("maintenance acquisition must have a finite timeout");
    let error = match result {
        Ok(_) => panic!("unserviced maintenance request was granted"),
        Err(error) => error,
    };
    assert_eq!(
        error,
        "project maintenance request timed out while still queued after 5 ms"
    );
    caller.join().unwrap();

    let WorkerMsg::BeginProjectMaintenance { operation, .. } = rx.try_recv().unwrap() else {
        panic!("expected timed-out maintenance request");
    };
    assert!(!operation.claim());
}

#[test]
fn abandoned_project_maintenance_request_cannot_freeze_worker_later() {
    let (tx, rx) = unbounded();
    let handle = WorkerHandle { tx };
    let cancel = AtomicBool::new(false);
    let result = handle.begin_project_maintenance_with_timeout(&cancel, Duration::from_millis(1));
    assert!(result.is_err());
    handle.rename_project("still runs".into());

    let (done_tx, done_rx) = bounded(1);
    let worker = std::thread::spawn(move || {
        let WorkerMsg::BeginProjectMaintenance {
            reply,
            resume,
            operation,
        } = rx.recv().unwrap()
        else {
            panic!("expected abandoned maintenance request");
        };
        let project = Project::new("late", Rational::FPS_30);
        serve_project_maintenance(&project, reply, resume, operation);

        let WorkerMsg::RenameProject { name } = rx.recv().unwrap() else {
            panic!("abandoned maintenance froze later queue work");
        };
        done_tx.send(name).unwrap();
    });

    assert_eq!(
        done_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
        "still runs"
    );
    worker.join().unwrap();
}

#[test]
fn project_maintenance_reports_worker_refusal_separately() {
    let (tx, rx) = unbounded();
    let handle = WorkerHandle { tx };
    let worker = std::thread::spawn(move || {
        let WorkerMsg::BeginProjectMaintenance {
            reply, operation, ..
        } = rx.recv().unwrap()
        else {
            panic!("expected maintenance request");
        };
        assert!(operation.claim());
        reply.send(Err(())).unwrap();
    });

    let error = match handle
        .begin_project_maintenance_with_timeout(&AtomicBool::new(false), Duration::from_secs(1))
    {
        Ok(_) => panic!("refused maintenance request was granted"),
        Err(error) => error,
    };
    assert_eq!(
        error,
        "project maintenance request was refused by preview worker"
    );
    worker.join().unwrap();
}

#[test]
fn project_maintenance_reports_a_disconnected_queue() {
    let (tx, rx) = unbounded();
    drop(rx);
    let handle = WorkerHandle { tx };

    let error = match handle.begin_project_maintenance_with_cancel(&AtomicBool::new(false)) {
        Ok(_) => panic!("disconnected worker granted maintenance"),
        Err(error) => error,
    };
    assert_eq!(
        error,
        "project maintenance request failed: preview worker is not running"
    );
}

/// The ladder drops on one slow render but climbs back only on
/// sustained, uniformly fast evidence: dwell elapsed, enough samples,
/// and no spike in the window. Guards against the tier-thrash seen in
/// field logs (a lone fast sequential-decode frame re-raising the tier
/// right before a seek slams it back down). Costs fed here are the
/// *scaled* (resolution-dependent) share of each render.
#[test]
fn quality_ladder_raises_only_on_sustained_fast_evidence() {
    let bottom = QUALITY_LADDER.len() - 1;
    let fit = FrameFit::default();

    // One catastrophically slow composite floors the ladder immediately.
    fit.note_render_cost(500.0);
    assert_eq!(fit.tier.get(), bottom);

    // A burst of fast renders alone doesn't raise: dwell hasn't passed.
    for _ in 0..RAISE_MIN_SAMPLES {
        fit.note_render_cost(5.0);
    }
    assert_eq!(fit.tier.get(), bottom);

    // With the dwell behind it, the same sustained-fast evidence raises.
    fit.changed_at.set(Instant::now() - RAISE_MIN_DWELL);
    for _ in 0..RAISE_MIN_SAMPLES {
        fit.note_render_cost(5.0);
    }
    assert_eq!(fit.tier.get(), bottom - 1);

    // A single mid-window spike (not slow enough to move the EMA past
    // the drop bound) vetoes raising even after the EMA looks fast
    // again…
    for _ in 0..3 {
        fit.note_render_cost(5.0);
    }
    fit.note_render_cost(100.0);
    assert_eq!(fit.tier.get(), bottom - 1, "spike must not drop the tier");
    fit.changed_at.set(Instant::now() - RAISE_MIN_DWELL);
    for _ in 0..RAISE_MIN_SAMPLES {
        fit.note_render_cost(5.0);
    }
    assert_eq!(fit.tier.get(), bottom - 1);

    // …until the evidence window turns over: a fresh uniformly fast
    // window raises again (and the raise re-arms the dwell, so exactly
    // one step happens here).
    for _ in 0..(RAISE_WINDOW_SAMPLES + RAISE_MIN_SAMPLES) {
        fit.note_render_cost(5.0);
    }
    assert_eq!(fit.tier.get(), bottom - 2);
}

/// Before the panel reports a viewport the fit bound is the conservative
/// default, never `None` (which meant "composite the full canvas" — an
/// 8.3 MP readback on a 4K project's launch frame).
#[test]
fn fit_bound_defaults_before_viewport_reports() {
    let fit = FrameFit::default();
    assert_eq!(fit.fit_bound(), Some(UNREPORTED_VIEWPORT_BOUND));

    fit.set_viewport(800, 600);
    assert_ne!(fit.fit_bound(), Some(UNREPORTED_VIEWPORT_BOUND));
}

/// `keyframes_at` slices one merged timeline diamond: only the
/// properties keyframed exactly at the tick, each with its own value
/// and easing, position as vec2.
#[test]
fn keyframes_at_collects_per_property_hits() {
    let mut t = AnimatedTransform::identity();
    t.set_param_keyframe(
        ClipParam::Scale,
        10,
        ParamValue::Scalar(2.0),
        Easing::EaseIn,
    )
    .unwrap();
    t.set_param_keyframe(
        ClipParam::Scale,
        20,
        ParamValue::Scalar(3.0),
        Easing::Linear,
    )
    .unwrap();
    t.set_param_keyframe(
        ClipParam::Position,
        10,
        ParamValue::Vec2([0.1, -0.2]),
        Easing::Linear,
    )
    .unwrap();
    t.set_param_keyframe(
        ClipParam::Opacity,
        30,
        ParamValue::Scalar(0.5),
        Easing::Linear,
    )
    .unwrap();

    let hits = keyframes_at(&t, 10);
    assert_eq!(
        hits,
        vec![
            (
                ClipParam::Position,
                ParamValue::Vec2([0.1, -0.2]),
                Easing::Linear
            ),
            (ClipParam::Scale, ParamValue::Scalar(2.0), Easing::EaseIn),
        ]
    );

    assert!(keyframes_at(&t, 15).is_empty());
    assert_eq!(keyframes_at(&t, 30).len(), 1);
}

// --- export settings mapping ---------------------------------------

#[test]
fn export_settings_default_to_the_project_native_output() {
    let project = Project::new("t", Rational::FPS_30);
    let request = ExportRequest {
        path: PathBuf::from("/tmp/out.mp4"),
        target_height: None,
        fps_num: None,
    };
    let settings = export_settings_for(&project, &request);
    let native = ExportSettings::for_project(&project).evened();
    assert_eq!(settings, native);
}

#[test]
fn export_settings_scale_to_the_target_height_preserving_aspect() {
    let project = Project::new("t", Rational::FPS_30);
    let (cw, ch) = ExportSettings::for_project(&project).size;
    let request = ExportRequest {
        path: PathBuf::from("/tmp/out.mp4"),
        target_height: Some(720),
        fps_num: None,
    };
    let settings = export_settings_for(&project, &request);
    assert_eq!(settings.size.1, 720);
    // Aspect preserved (within the even rounding).
    let expected_w = (f64::from(cw) * 720.0 / f64::from(ch)).round();
    assert!((f64::from(settings.size.0) - expected_w).abs() <= 1.0);
    // Evened for H.264.
    assert_eq!(settings.size.0 % 2, 0);
    assert_eq!(settings.size.1 % 2, 0);
}

#[test]
fn export_settings_override_the_frame_rate() {
    let project = Project::new("t", Rational::FPS_30);
    let request = ExportRequest {
        path: PathBuf::from("/tmp/out.mp4"),
        target_height: None,
        fps_num: Some(24),
    };
    let settings = export_settings_for(&project, &request);
    assert_eq!(settings.frame_rate, Rational::new(24, 1));
    // Zero/negative presets keep the timeline rate.
    let request = ExportRequest {
        path: PathBuf::from("/tmp/out.mp4"),
        target_height: None,
        fps_num: Some(0),
    };
    let settings = export_settings_for(&project, &request);
    assert_eq!(settings.frame_rate, Rational::FPS_30);
}

// --- magnet ripple trim (commit path) -----------------------------------

/// Engine over an empty project that carries one 1000-tick media entry
/// (no real file needed — nothing here decodes).
fn trim_test_engine() -> (Engine, MediaId) {
    let r = Rational::FPS_24;
    let mut project = Project::new("trim-fixture", r);
    let media = project.add_media(cutlass_models::MediaSource::new(
        "/tmp/trim-fixture.mp4",
        1920,
        1080,
        r,
        1000,
        false,
    ));
    let engine = Engine::with_project(EngineConfig::default(), project).expect("engine");
    (engine, media)
}

fn add_video_track(engine: &mut Engine, name: &str) -> TrackId {
    match engine
        .apply(Command::Edit(EditCommand::AddTrack {
            kind: TrackKind::Video,
            name: name.into(),
            index: None,
            pinned: false,
        }))
        .expect("add track")
    {
        ApplyOutcome::Edited(EditOutcome::CreatedTrack(id)) => id,
        other => panic!("expected CreatedTrack, got {other:?}"),
    }
}

/// Media-backed clip at `[start, start+duration)`; the source window is
/// offset 100 ticks into the fixture media so both edges keep headroom.
fn add_media_clip(
    engine: &mut Engine,
    track: TrackId,
    media: MediaId,
    start: i64,
    duration: i64,
) -> ClipId {
    match engine
        .apply(Command::Edit(EditCommand::AddClip {
            track,
            media,
            source: TimeRange::at_rate(100 + start, duration, Rational::FPS_24),
            start: RationalTime::new(start, Rational::FPS_24),
        }))
        .expect("add clip")
    {
        ApplyOutcome::Edited(EditOutcome::Created(id)) => id,
        other => panic!("expected Created, got {other:?}"),
    }
}

fn clip_starts(engine: &Engine, clips: &[ClipId]) -> Vec<i64> {
    clips
        .iter()
        .map(|id| engine.project().clip(*id).expect("clip").start().value)
        .collect()
}

#[test]
fn ripple_tail_shrink_shifts_downstream_on_main_lane() {
    let (mut engine, media) = trim_test_engine();
    let track = add_video_track(&mut engine, "V1");
    let a = add_media_clip(&mut engine, track, media, 0, 50);
    let b = add_media_clip(&mut engine, track, media, 50, 30);
    let c = add_media_clip(&mut engine, track, media, 80, 40);

    commit_trims(
        &mut engine,
        &[(b, TimeRange::at_rate(50, 20, Rational::FPS_24))],
        true,
    )
    .expect("ripple tail shrink");

    assert_eq!(clip_starts(&engine, &[a, b, c]), [0, 50, 70]);
    assert!(engine.undo());
    assert_eq!(clip_starts(&engine, &[a, b, c]), [0, 50, 80]);
}

#[test]
fn ripple_tail_grow_opens_room_before_extending() {
    let (mut engine, media) = trim_test_engine();
    let track = add_video_track(&mut engine, "V1");
    let a = add_media_clip(&mut engine, track, media, 0, 50);
    let b = add_media_clip(&mut engine, track, media, 50, 30);

    commit_trims(
        &mut engine,
        &[(a, TimeRange::at_rate(0, 60, Rational::FPS_24))],
        true,
    )
    .expect("ripple tail grow");

    assert_eq!(clip_starts(&engine, &[a, b]), [0, 60]);
}

#[test]
fn ripple_head_shrink_reanchors_at_old_start() {
    let (mut engine, media) = trim_test_engine();
    let track = add_video_track(&mut engine, "V1");
    let a = add_media_clip(&mut engine, track, media, 0, 50);
    let b = add_media_clip(&mut engine, track, media, 50, 30);

    commit_trims(
        &mut engine,
        &[(a, TimeRange::at_rate(10, 40, Rational::FPS_24))],
        true,
    )
    .expect("ripple head shrink");

    assert_eq!(clip_starts(&engine, &[a, b]), [0, 40]);
}

#[test]
fn plain_trim_off_magnet_leaves_gap() {
    let (mut engine, media) = trim_test_engine();
    let track = add_video_track(&mut engine, "V1");
    let a = add_media_clip(&mut engine, track, media, 0, 50);
    let b = add_media_clip(&mut engine, track, media, 50, 30);
    let c = add_media_clip(&mut engine, track, media, 80, 40);

    commit_trims(
        &mut engine,
        &[(b, TimeRange::at_rate(50, 20, Rational::FPS_24))],
        false,
    )
    .expect("plain trim");

    assert_eq!(clip_starts(&engine, &[a, b, c]), [0, 50, 80]);
}

// --- magnet ripple trim: media-backed clips (source derivation, links,
// --- rollback) -----------------------------------------------------------

/// Main lane `V1` packed gapless — A [0,100) B [100,200) C [200,300) —
/// each clip cut from the middle of a 1000-tick media, so both edges
/// have plenty of source headroom: A source [100,200), B [300,400),
/// C [500,600).
fn ripple_fixture() -> (Engine, [ClipId; 3], TrackId) {
    let r = Rational::FPS_24;
    let mut project = Project::new("ripple-fixture", r);
    let media = project.add_media(cutlass_models::MediaSource::new(
        "/tmp/ripple-fixture.mp4",
        1920,
        1080,
        r,
        1000,
        false,
    ));
    let track = project.add_track(TrackKind::Video, "V1");
    let a = project
        .add_clip(
            track,
            media,
            TimeRange::at_rate(100, 100, r),
            RationalTime::new(0, r),
        )
        .expect("clip A");
    let b = project
        .add_clip(
            track,
            media,
            TimeRange::at_rate(300, 100, r),
            RationalTime::new(100, r),
        )
        .expect("clip B");
    let c = project
        .add_clip(
            track,
            media,
            TimeRange::at_rate(500, 100, r),
            RationalTime::new(200, r),
        )
        .expect("clip C");

    let engine = Engine::with_project(EngineConfig::default(), project).expect("engine");
    (engine, [a, b, c], track)
}

fn extent(engine: &Engine, clip: ClipId) -> (i64, i64) {
    let placed = engine.project().clip(clip).expect("clip exists").timeline;
    (placed.start.value, placed.duration.value)
}

fn source_start(engine: &Engine, clip: ClipId) -> i64 {
    engine
        .project()
        .clip(clip)
        .expect("clip exists")
        .source_range()
        .expect("media clip has a source range")
        .start
        .value
}

fn tr24(start: i64, duration: i64) -> TimeRange {
    TimeRange::at_rate(start, duration, Rational::FPS_24)
}

/// Leading-edge shrink: the resolved extent moves the start right (that
/// delta advances the source in-point), the commit re-anchors at the old
/// start, and downstream follows — the lane stays gapless.
#[test]
fn ripple_head_shrink_advances_source_and_stays_anchored() {
    let (mut engine, [a, b, c], _track) = ripple_fixture();

    let rippled =
        commit_trims(&mut engine, &[(b, tr24(120, 80))], true).expect("ripple head shrink");
    assert!(rippled);

    assert_eq!(extent(&engine, b), (100, 80));
    assert_eq!(source_start(&engine, b), 320);
    assert_eq!(extent(&engine, c), (180, 100));
    assert_eq!(extent(&engine, a), (0, 100));

    // One undo restores the trim and the shift together.
    assert!(engine.undo());
    assert_eq!(extent(&engine, b), (100, 100));
    assert_eq!(source_start(&engine, b), 300);
    assert_eq!(extent(&engine, c), (200, 100));
}

/// Leading-edge grow: earlier source is revealed (in-point retreats),
/// the clip stays anchored, downstream moves right by the delta.
#[test]
fn ripple_head_grow_reveals_earlier_source() {
    let (mut engine, [a, b, c], _track) = ripple_fixture();

    let rippled = commit_trims(&mut engine, &[(b, tr24(50, 150))], true).expect("ripple head grow");
    assert!(rippled);

    assert_eq!(extent(&engine, b), (100, 150));
    assert_eq!(source_start(&engine, b), 250);
    assert_eq!(extent(&engine, c), (250, 100));
    assert_eq!(extent(&engine, a), (0, 100));
}

/// Trailing-edge trims keep the source in-point and move the out-point;
/// the shift only touches clips after the old end.
#[test]
fn ripple_tail_trims_keep_source_in_point() {
    let (mut engine, [a, b, c], _track) = ripple_fixture();

    commit_trims(&mut engine, &[(b, tr24(100, 140))], true).expect("ripple tail grow");
    assert_eq!(extent(&engine, b), (100, 140));
    assert_eq!(source_start(&engine, b), 300);
    assert_eq!(extent(&engine, c), (240, 100));
    assert_eq!(extent(&engine, a), (0, 100));
}

/// The last clip on the lane has nothing downstream — the ripple is a
/// plain trim (the shift selects an empty set and stays a no-op).
#[test]
fn ripple_trim_of_last_clip_has_no_downstream() {
    let (mut engine, [a, b, c], _track) = ripple_fixture();

    commit_trims(&mut engine, &[(c, tr24(200, 40))], true).expect("ripple last clip");
    assert_eq!(extent(&engine, c), (200, 40));
    assert_eq!(extent(&engine, a), (0, 100));
    assert_eq!(extent(&engine, b), (100, 100));
}

/// The magnet only governs the main lane (bottom video track): an
/// overlay-lane trim stays plain even with the magnet on.
#[test]
fn overlay_lane_trim_does_not_ripple() {
    let (mut engine, [a, _, _], _track) = ripple_fixture();
    let media = engine
        .project()
        .clip(a)
        .expect("clip")
        .media()
        .expect("media");
    let overlay = add_video_track(&mut engine, "V2");
    let d = add_media_clip(&mut engine, overlay, media, 0, 100);
    let e = add_media_clip(&mut engine, overlay, media, 100, 100);

    let rippled = commit_trims(&mut engine, &[(d, tr24(0, 60))], true).expect("overlay trim");
    assert!(!rippled);
    assert_eq!(extent(&engine, d), (0, 60));
    assert_eq!(extent(&engine, e), (100, 100));
}

/// A trim the engine rejects (source bounds) rolls the whole group back:
/// the shift that already opened room is undone, history stays untouched.
#[test]
fn rejected_ripple_rolls_back_whole_group() {
    let (mut engine, [a, b, c], _track) = ripple_fixture();

    // B's source is [300,400) of a 1000-tick media: growing the tail by
    // 700 ticks would need source up to 1100 — rejected after the shift.
    let result = commit_trims(&mut engine, &[(b, tr24(100, 800))], true);
    assert!(result.is_err());

    assert_eq!(extent(&engine, a), (0, 100));
    assert_eq!(extent(&engine, b), (100, 100));
    assert_eq!(extent(&engine, c), (200, 100));
    assert!(
        !engine.can_undo(),
        "rolled-back group must leave no history"
    );
}

/// Linked-pair trim (video on the main lane, audio partner elsewhere):
/// both members ripple on their own lanes, so the pair stays in sync and
/// downstream clips on both lanes shift by the same delta.
#[test]
fn linked_pair_ripples_on_both_lanes() {
    let (mut engine, [_, b, c], _track) = ripple_fixture();
    let r = Rational::FPS_24;
    let audio = match engine
        .apply(Command::Edit(EditCommand::AddTrack {
            kind: TrackKind::Audio,
            name: "A1".into(),
            index: None,
            pinned: false,
        }))
        .expect("add audio track")
    {
        ApplyOutcome::Edited(EditOutcome::CreatedTrack(id)) => id,
        other => panic!("expected CreatedTrack, got {other:?}"),
    };
    let media = engine
        .project()
        .clip(b)
        .expect("clip B")
        .media()
        .expect("media");
    let add_audio_clip = |engine: &mut Engine, source: TimeRange, start: i64| match engine
        .apply(Command::Edit(EditCommand::AddClip {
            track: audio,
            media,
            source,
            start: RationalTime::new(start, r),
        }))
        .expect("add audio clip")
    {
        ApplyOutcome::Edited(EditOutcome::Created(id)) => id,
        other => panic!("expected Created, got {other:?}"),
    };
    // P mirrors B; Q sits downstream on the audio lane, aligned with C.
    let p = add_audio_clip(&mut engine, tr24(300, 100), 100);
    let q = add_audio_clip(&mut engine, tr24(500, 100), 200);

    // Head-shrink both members by 20 (the worker's trim path hands the
    // same edge delta to every link-group member).
    commit_trims(&mut engine, &[(b, tr24(120, 80)), (p, tr24(120, 80))], true)
        .expect("linked ripple trim");

    assert_eq!(extent(&engine, b), (100, 80));
    assert_eq!(extent(&engine, p), (100, 80));
    assert_eq!(source_start(&engine, b), 320);
    assert_eq!(source_start(&engine, p), 320);
    // Downstream on both lanes shifted left by 20, staying aligned.
    assert_eq!(extent(&engine, c), (180, 100));
    assert_eq!(extent(&engine, q), (180, 100));
}

// --- OS file drop → timeline placement -----------------------------------

/// A pool entry wrapped the way `drop_files_and_publish` hands it to the
/// placement: full source range + duration resampled to timeline ticks.
fn dropped(engine: &Engine, media: MediaId) -> DroppedMedia {
    let entry = engine.project().media(media).expect("pool entry");
    let source = entry.full_range();
    let tl_rate = engine.project().timeline().frame_rate;
    DroppedMedia {
        media,
        source,
        duration_ticks: resample(source.duration, tl_rate).value.max(1),
    }
}

/// Engine over an empty timeline whose pool holds a 1000-tick base video
/// (for pre-placed clips), dropped videos of `video_ticks`, and dropped
/// audio-only entries of `audio_ms` — no real files, nothing decodes.
fn drop_fixture(
    video_ticks: &[i64],
    audio_ms: &[i64],
) -> (Engine, MediaId, Vec<MediaId>, Vec<MediaId>) {
    let r = Rational::FPS_24;
    let mut project = Project::new("drop-fixture", r);
    let base = project.add_media(cutlass_models::MediaSource::new(
        "/tmp/base.mp4",
        1920,
        1080,
        r,
        1000,
        false,
    ));
    let videos = video_ticks
        .iter()
        .enumerate()
        .map(|(i, &ticks)| {
            project.add_media(cutlass_models::MediaSource::new(
                format!("/tmp/drop-v{i}.mp4"),
                1920,
                1080,
                r,
                ticks,
                false,
            ))
        })
        .collect();
    let audios = audio_ms
        .iter()
        .enumerate()
        .map(|(i, &ms)| {
            project.add_media(cutlass_models::MediaSource::new(
                format!("/tmp/drop-a{i}.wav"),
                0,
                0,
                Rational::new(1000, 1),
                ms,
                true,
            ))
        })
        .collect();
    let engine = Engine::with_project(EngineConfig::default(), project).expect("engine");
    (engine, base, videos, audios)
}

fn lane_clip_extents(engine: &Engine, track: TrackId) -> Vec<(i64, i64)> {
    occupied_spans(engine.project().timeline().track(track).expect("track"))
}

/// Dropped videos land on the targeted video lane end-to-end from the
/// drop tick, first-fit sliding past a clip that blocks the anchor.
#[test]
fn os_drop_places_videos_end_to_end_on_the_target_lane() {
    let (mut engine, base, videos, _) = drop_fixture(&[50, 30], &[]);
    let track = add_video_track(&mut engine, "V1");
    add_media_clip(&mut engine, track, base, 80, 100); // occupies [80, 180)
    let items = [dropped(&engine, videos[0]), dropped(&engine, videos[1])];

    // Row 0 is the only lane (single track). Anchor 100 is inside the
    // existing clip → slide to 180, then chain: 180+50 = 230.
    place_drop_group(&mut engine, &items, TrackKind::Video, 0, 100, false);

    assert_eq!(
        lane_clip_extents(&engine, track),
        vec![(80, 180), (180, 230), (230, 260)]
    );
}

/// Audio-only files never land on a video lane: a drop anywhere creates
/// (or reuses) an audio lane, and the model's lane zones keep it below
/// the main track.
#[test]
fn os_drop_routes_audio_to_an_audio_lane_below_main() {
    // 4000 ms of audio ≙ 96 ticks at the FPS_24 timeline rate.
    let (mut engine, base, _, audios) = drop_fixture(&[], &[4000]);
    let track = add_video_track(&mut engine, "V1");
    add_media_clip(&mut engine, track, base, 0, 100);
    let items = [dropped(&engine, audios[0])];

    // Dropped on the main video row: wrong kind → new audio lane.
    place_drop_group(&mut engine, &items, TrackKind::Audio, 0, 240, false);

    let timeline = engine.project().timeline();
    let bottom = timeline.order()[0];
    let lane = timeline.track(bottom).expect("bottom lane");
    assert_eq!(lane.kind, TrackKind::Audio, "audio sits at the stack floor");
    assert_eq!(lane_clip_extents(&engine, bottom), vec![(240, 336)]);
    assert_eq!(
        timeline.main_track(),
        Some(track),
        "main track unchanged above the audio floor"
    );
}

/// Videos dropped on empty space (no video lane at the row) fall back to
/// the *empty* main track instead of spawning an overlay lane.
#[test]
fn os_drop_falls_back_to_the_empty_main_lane() {
    let (mut engine, _, videos, _) = drop_fixture(&[50, 40], &[]);
    let main = add_video_track(&mut engine, "V1"); // empty main lane
    let items = [dropped(&engine, videos[0]), dropped(&engine, videos[1])];

    // Row 7 hits no lane at all; the empty main catches the drop.
    place_drop_group(&mut engine, &items, TrackKind::Video, 7, 60, false);

    assert_eq!(
        lane_clip_extents(&engine, main),
        vec![(60, 110), (110, 150)]
    );
    assert_eq!(
        engine.project().timeline().order().len(),
        1,
        "no overlay lane spawned"
    );
}

/// With the magnet on, a drop on the main lane ripple-inserts the whole
/// set at the caret boundary: existing downstream clips shift right by
/// the sum of the inserted durations.
#[test]
fn os_drop_magnet_inserts_ripple_on_the_main_lane() {
    let (mut engine, base, videos, _) = drop_fixture(&[60, 40], &[]);
    let track = add_video_track(&mut engine, "V1");
    add_media_clip(&mut engine, track, base, 0, 100); // A [0, 100)
    add_media_clip(&mut engine, track, base, 100, 100); // B [100, 200)
    let items = [dropped(&engine, videos[0]), dropped(&engine, videos[1])];

    // Tick 120: past A's midpoint, before B's → boundary at B's start.
    place_drop_group(&mut engine, &items, TrackKind::Video, 0, 120, true);

    assert_eq!(
        lane_clip_extents(&engine, track),
        vec![(0, 100), (100, 160), (160, 200), (200, 300)],
        "inserted end-to-end at the boundary, B shifted right by 100"
    );
}

// --- extract audio -------------------------------------------------------

/// Video-with-audio on a video lane at timeline start `start`, duration
/// `duration` (source window starts at 0).
fn extract_fixture(has_audio: bool, start: i64, duration: i64) -> (Engine, ClipId, MediaId) {
    let r = Rational::FPS_24;
    let mut project = Project::new("extract-audio", r);
    let media = project.add_media(cutlass_models::MediaSource::new(
        "/tmp/extract-audio.mp4",
        1920,
        1080,
        r,
        1000,
        has_audio,
    ));
    let track = project.add_track(TrackKind::Video, "V1");
    let video = project
        .add_clip(
            track,
            media,
            TimeRange::at_rate(0, duration, r),
            RationalTime::new(start, r),
        )
        .expect("video clip");
    let engine = Engine::with_project(EngineConfig::default(), project).expect("engine");
    (engine, video, media)
}

fn audio_lane_count(engine: &Engine) -> usize {
    engine
        .project()
        .timeline()
        .tracks_ordered()
        .filter(|t| t.kind == TrackKind::Audio)
        .count()
}

fn audio_clip_count(engine: &Engine) -> usize {
    engine
        .project()
        .timeline()
        .tracks_ordered()
        .filter(|t| t.kind == TrackKind::Audio)
        .map(|t| t.clips_ordered().len())
        .sum()
}

#[test]
fn extract_audio_lands_linked_extracted_companion() {
    let (mut engine, video, media) = extract_fixture(true, 24, 96);
    assert!(engine.project().timeline().carries_own_audio(video));
    assert_eq!(audio_lane_count(&engine), 0);

    let audio = extract_audio(&mut engine, video).expect("extract");
    let audio_track = engine
        .project()
        .timeline()
        .track_of(audio)
        .expect("audio track");
    assert_eq!(
        engine
            .project()
            .timeline()
            .track(audio_track)
            .expect("track")
            .kind,
        TrackKind::Audio
    );
    assert_eq!(audio_lane_count(&engine), 1);

    let video_clip = engine.project().clip(video).expect("video").clone();
    let audio_clip = engine.project().clip(audio).expect("audio").clone();
    assert_eq!(video_clip.link, audio_clip.link);
    assert!(video_clip.link.is_some());
    assert_eq!(audio_clip.audio_role, Some(AudioRole::Extracted));
    assert_eq!(audio_clip.timeline.start.value, 24);
    assert_eq!(audio_clip.timeline.duration.value, 96);
    match (&video_clip.content, &audio_clip.content) {
        (
            ClipSource::Media {
                media: vm,
                source: vs,
            },
            ClipSource::Media {
                media: am,
                source: asrc,
            },
        ) => {
            assert_eq!(*vm, media);
            assert_eq!(vm, am);
            assert_eq!(vs, asrc);
        }
        _ => panic!("both halves must be media-backed"),
    }
    assert!(!engine.project().timeline().carries_own_audio(video));
    assert!(engine.project().timeline().carries_own_audio(audio));
    assert!(engine.project().timeline().detached_to_audio_lane(video));
    assert_eq!(
        engine.project().media_count(),
        1,
        "extract must not create a new media pool entry"
    );

    // New audio lane sinks below the video lane (audio-floor invariant).
    let order: Vec<_> = engine.project().timeline().order().to_vec();
    let video_track = engine.project().timeline().track_of(video).unwrap();
    let video_idx = order.iter().position(|t| *t == video_track).unwrap();
    let audio_idx = order.iter().position(|t| *t == audio_track).unwrap();
    assert!(
        audio_idx < video_idx,
        "audio lane must sit below video in stack order"
    );
}

#[test]
fn extract_audio_is_idempotent_when_already_detached() {
    let (mut engine, video, _) = extract_fixture(true, 0, 48);
    extract_audio(&mut engine, video).expect("first extract");
    assert_eq!(audio_clip_count(&engine), 1);

    let err = extract_audio(&mut engine, video).expect_err("already detached");
    assert!(err.starts_with("ignored:"));
    assert_eq!(audio_clip_count(&engine), 1);
    assert_eq!(audio_lane_count(&engine), 1);
}

#[test]
fn extract_audio_rejects_silent_video() {
    let (mut engine, video, _) = extract_fixture(false, 0, 48);
    let err = extract_audio(&mut engine, video).expect_err("no audio stream");
    assert!(err.starts_with("ignored:"));
    assert_eq!(audio_lane_count(&engine), 0);
    assert!(engine.project().timeline().carries_own_audio(video));
}

#[test]
fn extract_audio_rejects_audio_lane_clip() {
    let r = Rational::FPS_24;
    let mut project = Project::new("extract-audio-lane", r);
    let media = project.add_media(cutlass_models::MediaSource::new(
        "/tmp/extract-audio-lane.mp4",
        1920,
        1080,
        r,
        480,
        true,
    ));
    let track = project.add_track(TrackKind::Audio, "A1");
    let clip = project
        .add_clip(
            track,
            media,
            TimeRange::at_rate(0, 48, r),
            RationalTime::new(0, r),
        )
        .expect("audio clip");
    let mut engine = Engine::with_project(EngineConfig::default(), project).expect("engine");

    let err = extract_audio(&mut engine, clip).expect_err("audio lane");
    assert!(err.starts_with("ignored:"));
    assert_eq!(audio_clip_count(&engine), 1);
}

#[test]
fn extract_audio_rejects_generated_clip() {
    let r = Rational::FPS_24;
    let mut project = Project::new("extract-generated", r);
    let track = project.add_track(TrackKind::Text, "T1");
    let mut engine = Engine::with_project(EngineConfig::default(), project).expect("engine");
    let clip = match engine
        .apply(Command::Edit(EditCommand::AddGenerated {
            track,
            generator: Generator::text("Title"),
            timeline: TimeRange::at_rate(0, 48, r),
        }))
        .expect("add generated")
    {
        ApplyOutcome::Edited(EditOutcome::Created(id)) => id,
        other => panic!("expected Created, got {other:?}"),
    };

    let err = extract_audio(&mut engine, clip).expect_err("generated");
    assert!(err.starts_with("ignored:"));
    assert_eq!(audio_lane_count(&engine), 0);
}

#[test]
fn extract_audio_copies_speed_and_reverse() {
    let (mut engine, video, _) = extract_fixture(true, 0, 96);
    apply_edit(
        &mut engine,
        EditCommand::SetClipSpeed {
            clip: video,
            speed: Rational::new(2, 1),
            reversed: true,
        },
    )
    .expect("retime video");

    let audio = extract_audio(&mut engine, video).expect("extract");
    let audio_clip = engine.project().clip(audio).expect("audio");
    assert_eq!(audio_clip.speed, Rational::new(2, 1));
    assert!(audio_clip.reversed);
    // Timeline duration follows the retimed video half.
    let video_dur = engine
        .project()
        .clip(video)
        .unwrap()
        .timeline
        .duration
        .value;
    assert_eq!(audio_clip.timeline.duration.value, video_dur);
}

#[test]
fn extract_audio_reuses_free_audio_lane() {
    let (mut engine, video, _) = extract_fixture(true, 0, 48);
    // Pre-create an empty audio lane; extract should land there instead of
    // spawning a second one.
    let existing = match engine
        .apply(Command::Edit(EditCommand::AddTrack {
            kind: TrackKind::Audio,
            name: "A1".into(),
            index: None,
            pinned: false,
        }))
        .expect("add audio lane")
    {
        ApplyOutcome::Edited(EditOutcome::CreatedTrack(id)) => id,
        other => panic!("expected CreatedTrack, got {other:?}"),
    };
    assert_eq!(audio_lane_count(&engine), 1);

    let audio = extract_audio(&mut engine, video).expect("extract");
    assert_eq!(audio_lane_count(&engine), 1);
    assert_eq!(engine.project().timeline().track_of(audio), Some(existing));
}

#[test]
fn extract_audio_creates_lane_when_existing_overlaps() {
    let r = Rational::FPS_24;
    let mut project = Project::new("extract-overlap", r);
    let media = project.add_media(cutlass_models::MediaSource::new(
        "/tmp/extract-overlap.mp4",
        1920,
        1080,
        r,
        1000,
        true,
    ));
    let video_track = project.add_track(TrackKind::Video, "V1");
    let audio_track = project.add_track(TrackKind::Audio, "A1");
    // Occupant covers [0, 96) on the only audio lane.
    project
        .add_clip(
            audio_track,
            media,
            TimeRange::at_rate(0, 96, r),
            RationalTime::new(0, r),
        )
        .expect("occupant");
    let video = project
        .add_clip(
            video_track,
            media,
            TimeRange::at_rate(0, 48, r),
            RationalTime::new(0, r),
        )
        .expect("video");
    let mut engine = Engine::with_project(EngineConfig::default(), project).expect("engine");
    assert_eq!(audio_lane_count(&engine), 1);

    let audio = extract_audio(&mut engine, video).expect("extract");
    assert_eq!(audio_lane_count(&engine), 2);
    assert_ne!(
        engine.project().timeline().track_of(audio),
        Some(audio_track),
        "extracted clip must not land on the occupied lane"
    );
}

#[test]
fn extract_audio_undo_restores_pre_extract_state() {
    let (mut engine, video, _) = extract_fixture(true, 12, 60);
    assert!(engine.project().timeline().carries_own_audio(video));
    assert_eq!(engine.project().timeline().clip_count(), 1);

    extract_audio(&mut engine, video).expect("extract");
    assert_eq!(engine.project().timeline().clip_count(), 2);
    assert!(!engine.project().timeline().carries_own_audio(video));
    assert_eq!(audio_lane_count(&engine), 1);

    // One undo group → one undo clears the companion, link, role, and lane.
    assert!(engine.undo());
    assert_eq!(engine.project().timeline().clip_count(), 1);
    assert!(engine.project().timeline().carries_own_audio(video));
    assert!(!engine.project().timeline().detached_to_audio_lane(video));
    // Empty non-main audio lane may remain or be gone depending on policy;
    // the video half must be fully restored either way.
    assert!(engine.project().clip(video).unwrap().link.is_none());

    assert!(engine.redo());
    assert_eq!(engine.project().timeline().clip_count(), 2);
    assert!(!engine.project().timeline().carries_own_audio(video));
    assert_eq!(audio_clip_count(&engine), 1);
}
