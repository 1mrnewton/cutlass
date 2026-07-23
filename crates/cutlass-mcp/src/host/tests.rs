use super::*;

#[tokio::test]
async fn new_project_reports_meta() {
    let host = EngineHost::spawn();
    let meta = host
        .new_project("untitled".into(), 30.0)
        .await
        .expect("new_project");
    assert_eq!(meta.name, "untitled");
    assert_eq!(meta.revision, 0);
    assert!(!meta.dirty);
    assert!(!meta.can_undo);
    assert!(!meta.can_redo);
    assert!(meta.path.is_none());
}

#[tokio::test]
async fn get_project_before_open_errors() {
    let host = EngineHost::spawn();
    let err = host.get_project().await.expect_err("no project");
    assert_eq!(err, NO_PROJECT);
}

#[tokio::test]
async fn save_and_reopen_roundtrip() {
    let host = EngineHost::spawn();
    host.new_project("roundtrip".into(), 24.0)
        .await
        .expect("new");

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("roundtrip.cutlass");
    let saved = host.save_project(Some(path.clone())).await.expect("save");
    assert_eq!(saved, path);
    assert!(path.is_file());

    let host2 = EngineHost::spawn();
    let meta = host2.open_project(path).await.expect("open");
    assert_eq!(meta.name, "roundtrip");
}

#[tokio::test]
async fn unsupported_fps_is_readable() {
    let host = EngineHost::spawn();
    let err = host
        .new_project("x".into(), 31.5)
        .await
        .expect_err("bad fps");
    assert!(
        err.contains("unsupported frame rate"),
        "unexpected error: {err}"
    );
    assert!(err.contains("23.976"), "should list supported rates: {err}");
}

fn clip_count(doc: &ProjectDoc) -> usize {
    doc.project["tracks"]
        .as_array()
        .map(|tracks| {
            tracks
                .iter()
                .map(|t| t["clips"].as_array().map(|c| c.len()).unwrap_or(0))
                .sum()
        })
        .unwrap_or(0)
}

fn outcome_id(batch: &AppliedBatch, kind: &str, field: &str) -> u64 {
    batch
        .applied
        .iter()
        .find_map(|row| {
            let o = &row.outcome;
            if o["kind"] == kind {
                o[field].as_u64()
            } else {
                None
            }
        })
        .unwrap_or_else(|| panic!("no {kind}/{field} in {batch:?}"))
}

/// Solids need a sticker lane; create one and return its track id.
async fn ensure_sticker_track(host: &EngineHost) -> u64 {
    let batch = host
        .apply_edits(vec![serde_json::json!({
            "command": "add_track",
            "kind": "sticker",
            "name": "Overlays",
        })])
        .await
        .expect("add_track");
    outcome_id(&batch, "created_track", "track")
}

#[tokio::test]
async fn apply_edits_split_and_undo_batches() {
    let host = EngineHost::spawn();
    host.new_project("edits".into(), 30.0)
        .await
        .expect("new_project");
    let track = ensure_sticker_track(&host).await;

    // Batch 1: place a solid (no media file required).
    let batch1 = host
        .apply_edits(vec![serde_json::json!({
            "command": "add_generated",
            "track": track,
            "generator": { "type": "solid", "rgba": [255, 0, 0, 255] },
            "start": 0.0,
            "duration": 10.0,
        })])
        .await
        .expect("add_generated");
    let clip = outcome_id(&batch1, "created", "clip");
    assert_eq!(clip_count(&host.get_project().await.expect("get")), 1);

    // Batch 2: split using the id returned from batch 1.
    host.apply_edits(vec![serde_json::json!({
        "command": "split_clip",
        "clip": clip,
        "at": 5.0,
    })])
    .await
    .expect("split_clip");
    assert_eq!(clip_count(&host.get_project().await.expect("get")), 2);

    // Each apply_edits batch is one undo step.
    let u1 = host.undo().await.expect("undo split");
    assert!(u1.changed);
    assert_eq!(clip_count(&host.get_project().await.expect("get")), 1);

    let u2 = host.undo().await.expect("undo add");
    assert!(u2.changed);
    assert_eq!(clip_count(&host.get_project().await.expect("get")), 0);
}

#[tokio::test]
async fn apply_edits_rejection_rolls_back_whole_batch() {
    let host = EngineHost::spawn();
    host.new_project("rollback".into(), 30.0)
        .await
        .expect("new_project");
    let track = ensure_sticker_track(&host).await;
    let meta_before = host.get_project().await.expect("get").meta;

    let err = host
        .apply_edits(vec![
            serde_json::json!({
                "command": "add_generated",
                "track": track,
                "generator": { "type": "solid", "rgba": [0, 255, 0, 255] },
                "start": 0.0,
                "duration": 4.0,
            }),
            serde_json::json!({
                "command": "split_clip",
                "clip": 999,
                "at": 1.0,
            }),
        ])
        .await
        .expect_err("batch should fail");
    assert!(err.contains("edit[1]"), "names failing index: {err}");
    assert!(err.contains("rolled back"), "{err}");
    assert!(
        err.contains("999") || err.contains("does not exist") || err.contains("clip"),
        "rejection message: {err}"
    );

    let doc = host.get_project().await.expect("get");
    assert_eq!(clip_count(&doc), 0, "valid prefix must not linger");
    // Rolled-back group leaves no history entry — undo stack unchanged
    // (setup add_track is still the only undoable step).
    assert_eq!(doc.meta.can_undo, meta_before.can_undo);
    assert_eq!(doc.meta.can_redo, meta_before.can_redo);
}

#[tokio::test]
async fn rolled_back_batch_alone_leaves_no_undo() {
    // When the failing batch is the only mutation, rollback leaves can_undo false.
    let host = EngineHost::spawn();
    host.new_project("clean-rollback".into(), 30.0)
        .await
        .expect("new_project");

    let err = host
        .apply_edits(vec![
            serde_json::json!({
                "command": "add_track",
                "kind": "sticker",
                "name": "Temp",
            }),
            serde_json::json!({
                "command": "split_clip",
                "clip": 999,
                "at": 1.0,
            }),
        ])
        .await
        .expect_err("batch should fail");
    assert!(err.contains("edit[1]"), "{err}");

    let doc = host.get_project().await.expect("get");
    assert!(
        !doc.meta.can_undo,
        "rolled-back group must not push history"
    );
    assert!(!doc.meta.can_redo);
}

#[tokio::test]
async fn apply_edits_unknown_command_is_readable() {
    let host = EngineHost::spawn();
    host.new_project("parse".into(), 30.0)
        .await
        .expect("new_project");

    let err = host
        .apply_edits(vec![serde_json::json!({
            "command": "not_a_real_tool",
            "clip": 1,
        })])
        .await
        .expect_err("unknown");
    assert!(err.contains("not_a_real_tool"), "{err}");
    assert!(
        err.contains("unknown tool") || err.contains("available"),
        "{err}"
    );
}
