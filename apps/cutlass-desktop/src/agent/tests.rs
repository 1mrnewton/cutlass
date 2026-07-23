use super::*;
use crate::AgentEntry;
use crate::agent_senses::AgentSenses;
use crate::agent_session::ChatMeta;
use crate::preview_worker::agent_replay;
use crossbeam_channel::unbounded;
use cutlass_ai::wire;
use cutlass_ai::{EngineBridge, ToolHost, ToolOutput, ToolTier, WireCommand, summarize};
use cutlass_commands::EditOutcome;
use cutlass_engine::{Engine, EngineConfig};
use cutlass_jobs::JobManager;
use cutlass_models::{
    Generator, MediaSource, Project, Rational, RationalTime, TimeRange, TrackKind,
};
use cutlass_settings::Autonomy;
use slint::{Model, VecModel};
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

const TEST_APPROVAL_WAIT: Duration = Duration::from_millis(10);

fn decision(request_id: u64, choice: ApprovalChoice) -> ApprovalDecision {
    ApprovalDecision { request_id, choice }
}

fn test_tool_handles(job_manager: JobManager) -> DesktopToolHandles {
    DesktopToolHandles {
        store: slint::Weak::default(),
        app: slint::Weak::default(),
        cache_registry: None,
        job_manager,
        worker: None,
    }
}

#[test]
fn transcript_text_coalesces_only_adjacent_rows_of_the_same_kind() {
    let model = VecModel::<AgentEntry>::default();
    append_transcript_text(&model, "reasoning", "Inspecting ".into());
    append_transcript_text(&model, "reasoning", "the timeline.".into());
    append_transcript_text(&model, "assistant", "Done.".into());
    append_transcript_text(&model, "reasoning", "Verifying.".into());

    assert_eq!(model.row_count(), 3);
    assert_eq!(model.row_data(0).unwrap().kind, "reasoning");
    assert_eq!(model.row_data(0).unwrap().text, "Inspecting the timeline.");
    assert_eq!(model.row_data(1).unwrap().kind, "assistant");
    assert_eq!(model.row_data(2).unwrap().kind, "reasoning");
    assert_eq!(model.row_data(2).unwrap().text, "Verifying.");
}

#[test]
fn format_usage_line_matches_product_examples() {
    use cutlass_ai::TokenUsage;

    assert_eq!(format_usage_line(&TokenUsage::default()), None);
    assert_eq!(
        format_usage_line(&TokenUsage {
            input_tokens: 12_300,
            cached_input_tokens: 0,
            output_tokens: 456,
            cost: Some(0.68),
        })
        .as_deref(),
        Some("12.3k tokens in (0% cached) · 456 out · $0.68")
    );
}

#[test]
fn chat_choices_keep_duplicate_titles_unique_and_include_an_unsaved_active_chat() {
    let choices = chat_choices(
        vec![
            ChatMeta {
                id: "chat-3".into(),
                title: "Trim the clip".into(),
                updated_millis: 3,
            },
            ChatMeta {
                id: "chat-2".into(),
                title: "Trim the clip".into(),
                updated_millis: 2,
            },
            ChatMeta {
                id: "chat-1".into(),
                title: "Add captions".into(),
                updated_millis: 1,
            },
        ],
        Some("chat-4"),
    );

    assert_eq!(
        choices,
        vec![
            ChatChoice {
                id: "chat-4".into(),
                label: "New chat".into(),
            },
            ChatChoice {
                id: "chat-3".into(),
                label: "Trim the clip".into(),
            },
            ChatChoice {
                id: "chat-2".into(),
                label: "Trim the clip · 2".into(),
            },
            ChatChoice {
                id: "chat-1".into(),
                label: "Add captions".into(),
            },
        ]
    );
}

#[test]
fn ask_approval_accepts_the_matching_run_decision() {
    let (tx, rx) = unbounded();
    tx.send(decision(7, ApprovalChoice::Approve)).unwrap();
    let cancel = AtomicBool::new(false);

    assert_eq!(
        wait_for_system_tool_approval(&rx, 7, &cancel, TEST_APPROVAL_WAIT),
        ApprovalWaitOutcome::Approved
    );
}

#[test]
fn ask_approval_returns_the_matching_decline_decision() {
    let (tx, rx) = unbounded();
    tx.send(decision(11, ApprovalChoice::Deny)).unwrap();
    let cancel = AtomicBool::new(false);

    assert_eq!(
        wait_for_system_tool_approval(&rx, 11, &cancel, TEST_APPROVAL_WAIT),
        ApprovalWaitOutcome::Declined
    );
}

#[test]
fn ask_approval_cancellation_wins_over_a_queued_run_decision() {
    let (tx, rx) = unbounded();
    tx.send(decision(19, ApprovalChoice::Approve)).unwrap();
    let cancel = AtomicBool::new(true);

    assert_eq!(
        wait_for_system_tool_approval(&rx, 19, &cancel, TEST_APPROVAL_WAIT),
        ApprovalWaitOutcome::Cancelled
    );
}

#[test]
fn stale_run_decision_cannot_approve_a_later_request() {
    let (tx, rx) = unbounded();
    // Request 23 has already finished. Its delayed Run click must be
    // consumed but ignored while request 24 waits for its own answer.
    tx.send(decision(23, ApprovalChoice::Approve)).unwrap();
    tx.send(decision(24, ApprovalChoice::Deny)).unwrap();
    let cancel = AtomicBool::new(false);

    assert_eq!(
        wait_for_system_tool_approval(&rx, 24, &cancel, TEST_APPROVAL_WAIT),
        ApprovalWaitOutcome::Declined
    );
}

#[test]
fn approval_wait_reports_channel_closure() {
    let (tx, rx) = unbounded();
    drop(tx);
    let cancel = AtomicBool::new(false);

    assert_eq!(
        wait_for_system_tool_approval(&rx, 1, &cancel, TEST_APPROVAL_WAIT),
        ApprovalWaitOutcome::ChannelClosed
    );
}

#[test]
fn approval_request_ids_are_monotonic_and_never_zero() {
    let allocator = AtomicU64::new(0);

    assert_eq!(allocate_approval_request_id(&allocator).unwrap(), 1);
    assert_eq!(allocate_approval_request_id(&allocator).unwrap(), 2);

    let exhausted = AtomicU64::new(u64::MAX);
    assert!(allocate_approval_request_id(&exhausted).is_err());
    assert_eq!(exhausted.load(Ordering::Relaxed), u64::MAX);
}

#[test]
fn approval_detail_is_bounded_and_handles_empty_arguments() {
    assert_eq!(approval_title("project_open"), "Open this project draft?");
    assert_eq!(
        approval_title("project_import_media"),
        "Import this media file?"
    );
    assert_eq!(approval_title("system_cache_clear"), "Clear this cache?");
    assert_eq!(approval_title("system_cache_relocate"), "Move this cache?");
    assert_eq!(approval_title("future_tool"), "Run future_tool?");
    assert_eq!(
        approval_detail("system_cache_list", &serde_json::json!({}), None, None),
        "No arguments."
    );

    let detail = approval_detail(
        "python_run",
        &serde_json::json!({
            "script": "x".repeat(APPROVAL_DETAIL_MAX_CHARS + 100)
        }),
        None,
        None,
    );
    assert_eq!(detail.chars().count(), APPROVAL_DETAIL_MAX_CHARS + 1);
    assert!(detail.ends_with('…'));
    assert!(detail.starts_with("{\n  \"script\": \""));

    let project_detail = approval_detail(
        "project_open",
        &serde_json::json!({"draft_id": "abcdef-12"}),
        None,
        None,
    );
    assert_eq!(
        project_detail,
        "Draft ID: abcdef-12\n\nOpening this draft replaces the current session and may discard unsaved work."
    );
    assert!(!project_detail.contains("project.cutlass"));
    assert!(
        project_detail.chars().count() <= APPROVAL_DETAIL_MAX_CHARS,
        "{project_detail}"
    );

    let unsafe_project_detail = approval_detail(
        "project_open",
        &serde_json::json!({
            "draft_id": "/private/agent-secret/project.cutlass"
        }),
        None,
        None,
    );
    assert!(unsafe_project_detail.contains("<invalid draft ID>"));
    assert!(!unsafe_project_detail.contains("/private"));
    assert!(!unsafe_project_detail.contains("agent-secret"));

    let temp = tempfile::tempdir().expect("tempdir");
    let media = temp.path().join("approval clip.mp4");
    std::fs::write(&media, b"media").expect("write media");
    let import_arguments = serde_json::json!({"path": media});
    let validated = crate::agent_project::validated_import_media(&import_arguments)
        .expect("validated approval media");
    let canonical = validated.canonical_path().to_path_buf();
    let import_detail = approval_detail(
        "project_import_media",
        &import_arguments,
        None,
        Some(&validated),
    );
    assert_eq!(
        import_detail,
        format!(
            "Canonical file: {}\n\nCutlass adds a reference to this file rather than copying the source. Moving or deleting it can make the media missing.",
            canonical.display()
        )
    );
    assert!(import_detail.contains("rather than copying the source"));
    assert!(import_detail.contains("Moving or deleting it"));
    assert!(import_detail.chars().count() <= APPROVAL_DETAIL_MAX_CHARS);

    let hostile_import_detail = approval_detail(
        "project_import_media",
        &serde_json::json!({"path": "../../agent-secret\nclip.mp4"}),
        None,
        None,
    );
    assert!(hostile_import_detail.contains("<invalid media path>"));
    assert!(!hostile_import_detail.contains("agent-secret"));
    assert!(!hostile_import_detail.contains("clip.mp4"));

    let relocation_detail = format_cache_relocation_approval_detail(
        cutlass_storage::CacheId::Download,
        Path::new("/current/download-cache"),
        Path::new("/requested/download-cache"),
    );
    assert!(relocation_detail.contains("Cache: Downloads"));
    assert!(relocation_detail.contains("Current path: /current/download-cache"));
    assert!(relocation_detail.contains("Requested destination: /requested/download-cache"));
    assert!(relocation_detail.contains("projects reference cache-owned files"));
}

#[test]
fn full_autonomy_bypasses_the_approval_channel() {
    let (tx, rx) = unbounded();
    drop(tx);
    let pending = Arc::new(AtomicU64::new(0));
    let allocator = Arc::new(AtomicU64::new(0));
    let mut host = DesktopToolHost::new(
        Autonomy::Full,
        test_tool_handles(JobManager::new()),
        rx,
        pending.clone(),
        allocator.clone(),
    );
    let cancel = AtomicBool::new(false);

    assert_eq!(
        host.authorize(
            "system_cache_clear",
            &serde_json::json!({ "cache_id": "download" }),
            ToolTier::System,
            &cancel,
        ),
        Ok(())
    );
    assert_eq!(pending.load(Ordering::Acquire), 0);
    assert_eq!(allocator.load(Ordering::Relaxed), 0);

    let temp = tempfile::tempdir().expect("tempdir");
    assert_eq!(
        host.authorize(
            "system_cache_relocate",
            &serde_json::json!({
                "cache_id": "download",
                "destination": temp.path().join("new-download-cache")
            }),
            ToolTier::System,
            &cancel,
        ),
        Ok(())
    );
    assert_eq!(pending.load(Ordering::Acquire), 0);
    assert_eq!(allocator.load(Ordering::Relaxed), 0);

    let private_path = "/private/agent-secret/project.cutlass";
    assert_eq!(
        host.authorize(
            "project_open",
            &serde_json::json!({ "draft_id": private_path }),
            ToolTier::System,
            &cancel,
        ),
        Ok(())
    );
    assert_eq!(pending.load(Ordering::Acquire), 0);
    assert_eq!(allocator.load(Ordering::Relaxed), 0);
    let dispatch_error = host
        .call(
            "project_open",
            &serde_json::json!({ "draft_id": private_path }),
            &cancel,
        )
        .expect_err("dispatch must still validate under full autonomy");
    assert!(dispatch_error.contains("canonical app-owned draft ID"));
    assert!(!dispatch_error.contains("/private"));
    assert!(!dispatch_error.contains("agent-secret"));

    assert_eq!(
        host.authorize(
            "project_import_media",
            &serde_json::json!({ "path": "relative/clip.mp4" }),
            ToolTier::System,
            &cancel,
        ),
        Ok(())
    );
    let import_dispatch_error = host
        .call(
            "project_import_media",
            &serde_json::json!({ "path": "relative/clip.mp4" }),
            &cancel,
        )
        .expect_err("full-autonomy dispatch must still validate import paths");
    assert_eq!(
        import_dispatch_error,
        "project_import_media argument 'path' must be absolute"
    );

    assert_eq!(
        host.authorize(
            "media_preview_frame",
            &serde_json::json!({}),
            ToolTier::ReadOnly,
            &cancel,
        ),
        Ok(())
    );
}

#[test]
fn malformed_relocation_is_rejected_before_approval_side_effects() {
    let (_tx, rx) = unbounded();
    let pending = Arc::new(AtomicU64::new(0));
    let allocator = Arc::new(AtomicU64::new(0));
    let mut host = DesktopToolHost::new(
        Autonomy::Ask,
        test_tool_handles(JobManager::new()),
        rx,
        pending.clone(),
        allocator.clone(),
    );
    let cancel = AtomicBool::new(false);

    let error = host
        .authorize(
            "system_cache_relocate",
            &serde_json::json!({
                "cache_id": "download",
                "destination": "relative/cache"
            }),
            ToolTier::System,
            &cancel,
        )
        .expect_err("relative relocation must fail before approval");
    assert!(error.contains("must be absolute"));
    assert_eq!(pending.load(Ordering::Acquire), 0);
    assert_eq!(allocator.load(Ordering::Relaxed), 0);
}

#[test]
fn project_open_preflight_rejects_invalid_ids_before_approval_side_effects() {
    let (_tx, rx) = unbounded();
    let pending = Arc::new(AtomicU64::new(0));
    let allocator = Arc::new(AtomicU64::new(0));
    let mut host = DesktopToolHost::new(
        Autonomy::Ask,
        test_tool_handles(JobManager::new()),
        rx,
        pending.clone(),
        allocator.clone(),
    );
    let cancel = AtomicBool::new(false);

    let private_path = "/private/agent-secret/project.cutlass";
    let malformed = host
        .authorize(
            "project_open",
            &serde_json::json!({ "draft_id": private_path }),
            ToolTier::System,
            &cancel,
        )
        .expect_err("filesystem path must fail before approval");
    assert!(malformed.contains("canonical app-owned draft ID"));
    assert!(!malformed.contains("/private"));
    assert!(!malformed.contains("agent-secret"));
    assert_eq!(pending.load(Ordering::Acquire), 0);
    assert_eq!(allocator.load(Ordering::Relaxed), 0);

    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("current time after epoch")
        .as_nanos();
    let missing_id = format!("{nanos:x}-ffffffffffffffff");
    let missing = host
        .authorize(
            "project_open",
            &serde_json::json!({ "draft_id": missing_id }),
            ToolTier::System,
            &cancel,
        )
        .expect_err("missing draft must fail before approval");
    assert!(missing.starts_with("project_open failed:"), "{missing}");
    assert!(
        !missing.contains(crate::drafts::root_dir().to_string_lossy().as_ref()),
        "{missing}"
    );
    assert!(!missing.contains("project.cutlass"), "{missing}");
    assert_eq!(pending.load(Ordering::Acquire), 0);
    assert_eq!(allocator.load(Ordering::Relaxed), 0);
}

#[test]
fn project_import_preflight_rejects_paths_before_approval_side_effects() {
    let (_tx, rx) = unbounded();
    let pending = Arc::new(AtomicU64::new(0));
    let allocator = Arc::new(AtomicU64::new(0));
    let mut host = DesktopToolHost::new(
        Autonomy::Ask,
        test_tool_handles(JobManager::new()),
        rx,
        pending.clone(),
        allocator.clone(),
    );
    let cancel = AtomicBool::new(false);
    let temp = tempfile::tempdir().expect("tempdir");
    let missing = temp.path().join("agent-secret-missing.mp4");

    for arguments in [
        serde_json::json!({"path": "relative/clip.mp4"}),
        serde_json::json!({"path": missing}),
        serde_json::json!({"path": format!("{}\0clip.mp4", temp.path().display())}),
        serde_json::json!({"path": temp.path().join(".").join("clip.mp4")}),
    ] {
        let error = host
            .authorize(
                "project_import_media",
                &arguments,
                ToolTier::System,
                &cancel,
            )
            .expect_err("unsafe import path must fail before approval");
        assert!(!error.contains("agent-secret"), "{error}");
        assert!(
            !error.contains(temp.path().to_string_lossy().as_ref()),
            "{error}"
        );
        assert_eq!(pending.load(Ordering::Acquire), 0);
        assert_eq!(allocator.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn desktop_host_registers_app_project_job_and_system_tools_by_tier() {
    let (_tx, rx) = unbounded();
    let host = DesktopToolHost::new(
        Autonomy::Ask,
        test_tool_handles(JobManager::new()),
        rx,
        Arc::new(AtomicU64::new(0)),
        Arc::new(AtomicU64::new(0)),
    );
    let specs = host.tools();
    assert_eq!(specs.len(), 24);
    assert_eq!(
        specs
            .iter()
            .find(|spec| spec.name == "app_state")
            .map(|spec| spec.tier),
        Some(ToolTier::ReadOnly)
    );
    assert_eq!(
        specs
            .iter()
            .find(|spec| spec.name == "app_close")
            .map(|spec| spec.tier),
        Some(ToolTier::System)
    );
    assert_eq!(
        specs
            .iter()
            .filter(|spec| spec.name.starts_with("project_"))
            .map(|spec| (spec.name.as_str(), spec.tier))
            .collect::<Vec<_>>(),
        vec![
            ("project_list_drafts", ToolTier::ReadOnly),
            ("project_save", ToolTier::Workspace),
            ("project_open", ToolTier::System),
            ("project_import_media", ToolTier::System),
        ]
    );
    assert_eq!(
        specs
            .iter()
            .filter(|spec| spec.name.starts_with("job_"))
            .map(|spec| (spec.name.as_str(), spec.tier))
            .collect::<Vec<_>>(),
        vec![
            ("job_list", ToolTier::ReadOnly),
            ("job_status", ToolTier::ReadOnly),
            ("job_cancel", ToolTier::Workspace),
        ]
    );
    assert!(
        specs
            .iter()
            .filter(|spec| spec.name.starts_with("system_"))
            .all(|spec| spec.tier == ToolTier::System)
    );
    assert_eq!(
        specs
            .iter()
            .filter(|spec| spec.tier == ToolTier::System)
            .map(|spec| spec.name.as_str())
            .collect::<Vec<_>>(),
        vec![
            "app_close",
            "project_open",
            "project_import_media",
            "system_reveal",
            "system_open_external",
            "system_cache_list",
            "system_cache_clear",
            "system_cache_relocate",
        ]
    );
}

#[test]
fn non_system_project_tools_never_enter_the_system_approval_flow() {
    let (tx, rx) = unbounded();
    drop(tx);
    let pending = Arc::new(AtomicU64::new(0));
    let allocator = Arc::new(AtomicU64::new(0));
    let mut host = DesktopToolHost::new(
        Autonomy::Ask,
        test_tool_handles(JobManager::new()),
        rx,
        pending.clone(),
        allocator.clone(),
    );
    let cancel = AtomicBool::new(false);

    for (name, tier) in [
        ("project_list_drafts", ToolTier::ReadOnly),
        ("project_save", ToolTier::Workspace),
    ] {
        host.authorize(name, &serde_json::json!({}), tier, &cancel)
            .expect("project tools do not require approval");
    }
    assert_eq!(pending.load(Ordering::Acquire), 0);
    assert_eq!(allocator.load(Ordering::Relaxed), 0);
}

#[test]
fn desktop_host_dispatches_jobs_and_tracks_cancel_but_not_reads() {
    let (_tx, rx) = unbounded();
    let jobs = JobManager::new();
    let mut host = DesktopToolHost::new(
        Autonomy::Full,
        test_tool_handles(jobs),
        rx,
        Arc::new(AtomicU64::new(0)),
        Arc::new(AtomicU64::new(0)),
    );
    let cancel = AtomicBool::new(false);

    assert!(!host.ordinary_host_call_attempted());
    let list = host
        .call("job_list", &serde_json::json!({}), &cancel)
        .expect("job namespace must dispatch");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&list.text).unwrap()["status"],
        "ok"
    );
    assert!(!host.ordinary_host_call_attempted());

    let status_error = host
        .call("job_status", &serde_json::json!({ "job_id": 1 }), &cancel)
        .expect_err("unknown job status");
    assert!(status_error.contains("unknown or has been pruned"));
    assert!(!host.ordinary_host_call_attempted());

    let cancel_error = host
        .call("job_cancel", &serde_json::json!({ "job_id": 1 }), &cancel)
        .expect_err("unknown job cancellation");
    assert!(cancel_error.contains("unknown or has been pruned"));
    assert!(host.ordinary_host_call_attempted());
}

#[test]
fn desktop_host_dispatches_project_tools_and_tracks_mutations_as_effects() {
    let (_tx, rx) = unbounded();
    let mut host = DesktopToolHost::new(
        Autonomy::Full,
        test_tool_handles(JobManager::new()),
        rx,
        Arc::new(AtomicU64::new(0)),
        Arc::new(AtomicU64::new(0)),
    );
    let cancel = AtomicBool::new(false);

    assert!(!host.ordinary_host_call_attempted());
    let list_error = host
        .call(
            "project_list_drafts",
            &serde_json::json!({"limit": 0}),
            &cancel,
        )
        .expect_err("malformed draft listing must dispatch to its strict parser");
    assert!(list_error.contains("integer from 1 through 100"));
    assert!(
        !host.ordinary_host_call_attempted(),
        "a read-only project query is not an ordinary host effect"
    );

    let private_path = "/private/agent-secret/project.cutlass";
    let open_error = host
        .call(
            "project_open",
            &serde_json::json!({ "draft_id": private_path }),
            &cancel,
        )
        .expect_err("malformed project open must dispatch to strict validation");
    assert!(open_error.contains("canonical app-owned draft ID"));
    assert!(!open_error.contains("/private"));
    assert!(host.ordinary_host_call_attempted());

    let error = host
        .call("project_save", &serde_json::json!({}), &cancel)
        .expect_err("test fixture has no worker");
    assert!(error.contains("editor worker is unavailable"));
    assert!(host.ordinary_host_call_attempted());
}

#[test]
fn abort_status_distinguishes_sandbox_only_from_host_effects() {
    assert_eq!(
        abort_status_message("cancelled", false),
        "Stopped — nothing was applied."
    );
    assert_eq!(
        abort_status_message("provider failed", false),
        "provider failed — nothing was applied."
    );

    let cancelled_after_host = abort_status_message("cancelled", true);
    assert!(cancelled_after_host.starts_with("Stopped —"));
    assert!(
        cancelled_after_host.contains("Timeline edits staged by this prompt were rolled back"),
        "{cancelled_after_host}"
    );
    assert!(
        cancelled_after_host.contains("host actions that already completed remain in effect"),
        "{cancelled_after_host}"
    );
    assert!(
        !cancelled_after_host.contains("nothing was applied"),
        "{cancelled_after_host}"
    );

    let failed_after_host = abort_status_message("HTTP 500", true);
    assert!(failed_after_host.contains("HTTP 500"));
    assert!(failed_after_host.contains("Timeline edits staged by this prompt were rolled back"));
    assert!(failed_after_host.contains("remain in effect"));
}

#[test]
fn transcript_images_decode_through_the_bounded_rgba_boundary() {
    let expected = cutlass_render::RgbaImage::new(2, 1, vec![1, 2, 3, 255, 4, 5, 6, 128]);
    let image = cutlass_ai::ImagePart::png(
        cutlass_render::encode_png(&expected).expect("encode fixture"),
        "fixture",
    );
    assert_eq!(decode_transcript_image(&image).expect("decode"), expected);

    let unsupported = cutlass_ai::ImagePart {
        media_type: "image/gif".into(),
        data: Arc::new(vec![1, 2, 3]),
        label: "animated".into(),
    };
    assert!(
        decode_transcript_image(&unsupported)
            .expect_err("unsupported type")
            .contains("unsupported")
    );
}

#[test]
fn transcript_image_labels_are_safe_and_bounded() {
    assert_eq!(transcript_image_label(""), "Agent image");
    assert_eq!(transcript_image_label("bad\nlabel"), "bad\u{fffd}label");
    let long = "x".repeat(200);
    let label = transcript_image_label(&long);
    assert_eq!(label.chars().count(), 161);
    assert!(label.ends_with('…'));
}

fn fixture_project() -> (Project, u64) {
    let mut project = Project::new("agent-ui-fixture", Rational::FPS_24);
    let media = project
        .add_media(MediaSource::new(
            "/tmp/agent-ui-fixture.mp4",
            1920,
            1080,
            Rational::FPS_24,
            60 * 24,
            false,
        ))
        .raw();
    (project, media)
}

fn temp_engine(project: Project) -> Engine {
    Engine::with_project(EngineConfig::default(), project).expect("engine")
}

struct UnexpectedProjectSnapshot;

impl ProjectSnapshotSource for UnexpectedProjectSnapshot {
    fn snapshot_project(&self) -> Option<Project> {
        panic!("this test must not request a live project snapshot")
    }
}

static UNEXPECTED_PROJECT_SNAPSHOT: UnexpectedProjectSnapshot = UnexpectedProjectSnapshot;

struct ScriptedProjectSnapshots {
    snapshots: RefCell<VecDeque<Option<Project>>>,
    calls: Cell<usize>,
}

impl ScriptedProjectSnapshots {
    fn new(snapshots: impl IntoIterator<Item = Option<Project>>) -> Self {
        Self {
            snapshots: RefCell::new(snapshots.into_iter().collect()),
            calls: Cell::new(0),
        }
    }
}

impl ProjectSnapshotSource for ScriptedProjectSnapshots {
    fn snapshot_project(&self) -> Option<Project> {
        self.calls.set(self.calls.get() + 1);
        self.snapshots
            .borrow_mut()
            .pop_front()
            .expect("scripted project snapshot")
    }
}

#[test]
fn sandbox_bridge_exposes_read_only_senses_of_its_project() {
    let (project, _) = fixture_project();
    let mut sandbox = temp_engine(project);
    let mut plan = Vec::new();
    let mut senses = AgentSenses::new();
    let cancel = AtomicBool::new(false);
    let output = {
        let mut bridge = SandboxBridge {
            worker: &UNEXPECTED_PROJECT_SNAPSHOT,
            engine: &mut sandbox,
            plan: &mut plan,
            senses: &mut senses,
            default_playhead_seconds: 1.25,
        };
        assert_eq!(bridge.sense_tools().len(), AgentSenses::specs().len());
        bridge
            .sense(
                "media_timeline_map",
                &serde_json::json!({"playhead_seconds": 1.25}),
                &cancel,
            )
            .expect("timeline sense")
    };

    assert!(plan.is_empty(), "a sense never adds an edit step");
    assert_eq!(output.images.len(), 1);
    assert_eq!(output.images[0].media_type, "image/png");
    assert_eq!(&output.images[0].data[..8], b"\x89PNG\r\n\x1a\n");
    assert!(output.text.contains("playhead 1.25s"));
}

#[test]
fn project_host_pre_hook_rejects_an_existing_staged_plan() {
    let (project, _) = fixture_project();
    let mut sandbox = temp_engine(project);
    let mut plan = vec![AgentPlanStep {
        command: WireCommand::AddMarker(wire::AddMarker {
            at: 1.0,
            name: Some("pending".into()),
            color: None,
        }),
        created: None,
    }];
    let mut senses = AgentSenses::new();
    let mut bridge = SandboxBridge {
        worker: &UNEXPECTED_PROJECT_SNAPSHOT,
        engine: &mut sandbox,
        plan: &mut plan,
        senses: &mut senses,
        default_playhead_seconds: 0.0,
    };

    let error = bridge
        .before_host_call("project_save", &serde_json::json!({}))
        .expect_err("project mutation must not invalidate a staged plan");
    assert!(error.contains("before staged edits"), "{error}");
    assert!(
        error.contains("applies or discards the pending plan"),
        "{error}"
    );
    assert_eq!(
        bridge.before_host_call("project_list_drafts", &serde_json::json!({ "limit": 10 })),
        Ok(()),
        "read-only project tools remain available with staged edits"
    );
    assert_eq!(
        bridge.before_host_call("app_state", &serde_json::json!({})),
        Ok(()),
        "non-project host tools are unchanged"
    );
}

#[test]
fn project_post_hook_refreshes_after_host_success_and_failure_and_reopens_group() {
    fn live_snapshot(name: &str) -> Project {
        let mut project = Project::new(name, Rational::FPS_24);
        project.add_track(TrackKind::Video, "Live Main");
        project
    }

    let success_snapshot = live_snapshot("after-success");
    let failure_snapshot = live_snapshot("after-failure");
    let snapshots = ScriptedProjectSnapshots::new([
        Some(success_snapshot.clone()),
        Some(failure_snapshot.clone()),
    ]);

    let mut success_sandbox = temp_engine(live_snapshot("stale-success"));
    let mut success_plan = Vec::new();
    let mut success_senses = AgentSenses::new();
    {
        let mut bridge = SandboxBridge {
            worker: &snapshots,
            engine: &mut success_sandbox,
            plan: &mut success_plan,
            senses: &mut success_senses,
            default_playhead_seconds: 0.0,
        };
        bridge.begin_group();
        let output = ToolOutput::text(r#"{"media_id":42}"#);
        bridge
            .after_host_call("project_save", &serde_json::json!({}), Ok(&output))
            .expect("successful project call reconciliation");
        assert_eq!(bridge.engine.project().name, "after-success");
        assert!(bridge.plan.is_empty());

        bridge
            .apply(&WireCommand::AddMarker(wire::AddMarker {
                at: 1.0,
                name: Some("later edit".into()),
                color: None,
            }))
            .expect("edit after reconciliation");
        assert_eq!(bridge.engine.project().timeline().marker_count(), 1);
        bridge.rollback_group();
    }
    assert_eq!(success_sandbox.project().name, "after-success");
    assert_eq!(
        success_sandbox.project().timeline().marker_count(),
        success_snapshot.timeline().marker_count(),
        "abort rollback restores the reconciled live snapshot"
    );
    assert!(
        !success_sandbox.undo(),
        "the reopened group did not leak an undo entry"
    );

    let mut failure_sandbox = temp_engine(live_snapshot("stale-failure"));
    let mut failure_plan = Vec::new();
    let mut failure_senses = AgentSenses::new();
    {
        let mut bridge = SandboxBridge {
            worker: &snapshots,
            engine: &mut failure_sandbox,
            plan: &mut failure_plan,
            senses: &mut failure_senses,
            default_playhead_seconds: 0.0,
        };
        bridge.begin_group();
        bridge
            .after_host_call(
                "project_save",
                &serde_json::json!({}),
                Err("import failed after dispatch"),
            )
            .expect("failed host result still reconciles");
        assert_eq!(bridge.engine.project().name, "after-failure");
        assert!(bridge.plan.is_empty());
        bridge.rollback_group();
    }
    assert_eq!(failure_sandbox.project().name, "after-failure");
    assert_eq!(
        snapshots.calls.get(),
        2,
        "one ordered snapshot per dispatch"
    );
    assert!(
        snapshots.snapshots.borrow().is_empty(),
        "snapshots were consumed in queue order"
    );
}

#[test]
fn project_post_hook_fails_hard_when_the_worker_cannot_reply() {
    let snapshots = ScriptedProjectSnapshots::new([None]);
    let (project, _) = fixture_project();
    let mut sandbox = temp_engine(project);
    let mut plan = Vec::new();
    let mut senses = AgentSenses::new();
    let mut bridge = SandboxBridge {
        worker: &snapshots,
        engine: &mut sandbox,
        plan: &mut plan,
        senses: &mut senses,
        default_playhead_seconds: 0.0,
    };
    bridge.begin_group();

    let error = bridge
        .after_host_call(
            "project_save",
            &serde_json::json!({}),
            Err("host result is immaterial"),
        )
        .expect_err("a missing live snapshot must abort reconciliation");
    assert!(error.contains("could not reconcile"), "{error}");
    assert!(error.contains("did not reply"), "{error}");
    bridge.rollback_group();
}

#[test]
fn read_only_project_and_non_project_hooks_do_not_snapshot_or_reset_the_sandbox() {
    let snapshots = ScriptedProjectSnapshots::new([Some(Project::new("unused", Rational::FPS_24))]);
    let (project, _) = fixture_project();
    let mut sandbox = temp_engine(project);
    let revision = sandbox.revision();
    let mut plan = vec![AgentPlanStep {
        command: WireCommand::AddMarker(wire::AddMarker {
            at: 1.0,
            name: None,
            color: None,
        }),
        created: None,
    }];
    let mut senses = AgentSenses::new();
    let mut bridge = SandboxBridge {
        worker: &snapshots,
        engine: &mut sandbox,
        plan: &mut plan,
        senses: &mut senses,
        default_playhead_seconds: 0.0,
    };
    let output = ToolOutput::text("ok");

    assert_eq!(
        bridge.before_host_call("project_list_drafts", &serde_json::json!({ "limit": 5 })),
        Ok(())
    );
    assert_eq!(
        bridge.after_host_call(
            "project_list_drafts",
            &serde_json::json!({ "limit": 5 }),
            Ok(&output)
        ),
        Ok(())
    );
    assert_eq!(
        bridge.before_host_call("app_state", &serde_json::json!({})),
        Ok(())
    );
    assert_eq!(
        bridge.after_host_call("app_state", &serde_json::json!({}), Ok(&output)),
        Ok(())
    );
    assert_eq!(snapshots.calls.get(), 0);
    assert_eq!(bridge.engine.revision(), revision);
    assert_eq!(bridge.engine.project().name, "agent-ui-fixture");
    assert_eq!(bridge.plan.len(), 1);
}

#[test]
fn rehearsed_plan_replays_with_id_remapping_and_single_undo() {
    let (project, media) = fixture_project();
    let mut sandbox = temp_engine(project.clone());
    let mut live = temp_engine(project);

    let mut plan: Vec<AgentPlanStep> = Vec::new();
    let mut senses = AgentSenses::new();
    let mut bridge = SandboxBridge {
        worker: &UNEXPECTED_PROJECT_SNAPSHOT,
        engine: &mut sandbox,
        plan: &mut plan,
        senses: &mut senses,
        default_playhead_seconds: 0.0,
    };
    bridge.begin_group();
    let track = match bridge
        .apply(&WireCommand::AddTrack(wire::AddTrack {
            kind: wire::WireTrackKind::Video,
            name: "V1".into(),
            index: None,
        }))
        .expect("add track")
    {
        EditOutcome::CreatedTrack(id) => id.raw(),
        other => panic!("expected created track, got {other:?}"),
    };
    let head = match bridge
        .apply(&WireCommand::AddClip(wire::AddClip {
            track,
            media,
            source_start: 0.0,
            source_duration: 10.0,
            start: 0.0,
        }))
        .expect("add clip")
    {
        EditOutcome::Created(id) => id.raw(),
        other => panic!("expected created clip, got {other:?}"),
    };
    let right = match bridge
        .apply(&WireCommand::SplitClip(wire::SplitClip {
            clip: head,
            at: 4.0,
        }))
        .expect("split clip")
    {
        EditOutcome::Created(id) => id.raw(),
        other => panic!("expected created clip, got {other:?}"),
    };
    bridge
        .apply(&WireCommand::TrimClip(wire::TrimClip {
            clip: right,
            start: 4.0,
            duration: 2.0,
        }))
        .expect("trim clip");
    bridge.end_group();
    assert_eq!(plan.len(), 4);

    agent_replay(&mut live, vec![plan], |_| {}).expect("replay");

    let timeline = live.project().timeline();
    assert_eq!(timeline.track_count(), 1);
    assert_eq!(timeline.clip_count(), 2);

    assert!(live.undo(), "the plan is one undo entry");
    assert_eq!(live.project().timeline().track_count(), 0);
    assert!(!live.undo(), "nothing left to undo");
}

#[test]
fn stale_plan_rolls_back_and_reports() {
    let (project, _media) = fixture_project();
    let mut live = temp_engine(project);

    let plan = vec![AgentPlanStep {
        command: WireCommand::TrimClip(wire::TrimClip {
            clip: 999_999,
            start: 0.0,
            duration: 1.0,
        }),
        created: None,
    }];
    let err = agent_replay(&mut live, vec![plan], |_| {}).expect_err("stale plan must fail");
    assert!(err.contains("step 1/1"), "names the failing step: {err}");
    assert!(err.contains("nothing was applied"), "{err}");
    assert!(!live.undo(), "rollback leaves no history entry");
}

#[test]
fn removing_last_sticker_clip_also_removes_its_lane() {
    let mut project = Project::new("agent-sticker-removal", Rational::FPS_24);
    let main = project.add_track(TrackKind::Video, "V1");
    let stickers = project.add_track(TrackKind::Sticker, "Stickers");
    let sticker = project
        .add_generated(
            stickers,
            Generator::sticker(""),
            TimeRange::at_rate(0, 48, Rational::FPS_24),
        )
        .expect("sticker");
    let mut live = temp_engine(project);

    let plan = vec![AgentPlanStep {
        command: WireCommand::RemoveClip(wire::RemoveClip {
            clip: sticker.raw(),
        }),
        created: None,
    }];
    agent_replay(&mut live, vec![plan], |_| {}).expect("replay");

    let timeline = live.project().timeline();
    assert!(timeline.track(main).is_some(), "main lane remains");
    assert!(
        timeline.track(stickers).is_none(),
        "empty sticker lane is removed"
    );
    assert!(timeline.clip(sticker).is_none());

    assert!(live.undo(), "clip removal and lane cleanup share one undo");
    let timeline = live.project().timeline();
    assert!(timeline.track(stickers).is_some());
    assert!(timeline.clip(sticker).is_some());
}

#[test]
fn split_plan_phases_keeps_breaks_and_drops_an_empty_tail() {
    let step = || AgentPlanStep {
        command: WireCommand::SplitClip(wire::SplitClip { clip: 1, at: 1.0 }),
        created: None,
    };
    let plan: Vec<AgentPlanStep> = (0..4).map(|_| step()).collect();
    let phases = split_plan_phases(plan, &[2]);
    assert_eq!(phases.iter().map(Vec::len).collect::<Vec<_>>(), vec![2, 2]);

    // A commit flush with the plan's end must not leave an empty phase.
    let plan: Vec<AgentPlanStep> = (0..3).map(|_| step()).collect();
    let phases = split_plan_phases(plan, &[3]);
    assert_eq!(phases.iter().map(Vec::len).collect::<Vec<_>>(), vec![3]);

    let plan: Vec<AgentPlanStep> = (0..3).map(|_| step()).collect();
    let phases = split_plan_phases(plan, &[]);
    assert_eq!(phases.iter().map(Vec::len).collect::<Vec<_>>(), vec![3]);
}

#[test]
fn phased_plan_replays_as_separate_undo_steps_with_remapping() {
    let mut project = Project::new("agent-phase-fixture", Rational::FPS_24);
    let media = project.add_media(MediaSource::new(
        "/tmp/agent-phase-fixture.mp4",
        1920,
        1080,
        Rational::FPS_24,
        60 * 24,
        false,
    ));
    // The sandbox rehearses against a snapshot taken before the live
    // project grew an extra lane and clip: live allocations diverge
    // from sandbox ids, so the remap must do real work — including
    // across the phase boundary.
    let sandbox_project = project.clone();
    let existing = project.add_track(TrackKind::Video, "Existing");
    let seed_clip = project
        .add_clip(
            existing,
            media,
            TimeRange::at_rate(0, 24, Rational::FPS_24),
            RationalTime::new(0, Rational::FPS_24),
        )
        .expect("seed clip");
    let mut sandbox = temp_engine(sandbox_project);
    let mut live = temp_engine(project);

    let mut plan: Vec<AgentPlanStep> = Vec::new();
    let mut senses = AgentSenses::new();
    let mut bridge = SandboxBridge {
        worker: &UNEXPECTED_PROJECT_SNAPSHOT,
        engine: &mut sandbox,
        plan: &mut plan,
        senses: &mut senses,
        default_playhead_seconds: 0.0,
    };
    bridge.begin_group();
    let track = match bridge
        .apply(&WireCommand::AddTrack(wire::AddTrack {
            kind: wire::WireTrackKind::Video,
            name: "V1".into(),
            index: None,
        }))
        .expect("add track")
    {
        EditOutcome::CreatedTrack(id) => id.raw(),
        other => panic!("expected created track, got {other:?}"),
    };
    let head = match bridge
        .apply(&WireCommand::AddClip(wire::AddClip {
            track,
            media: media.raw(),
            source_start: 0.0,
            source_duration: 10.0,
            start: 0.0,
        }))
        .expect("add clip")
    {
        EditOutcome::Created(id) => id.raw(),
        other => panic!("expected created clip, got {other:?}"),
    };
    // Phase 2 splits the clip phase 1 created — the cross-phase remap.
    let right = match bridge
        .apply(&WireCommand::SplitClip(wire::SplitClip {
            clip: head,
            at: 4.0,
        }))
        .expect("split clip")
    {
        EditOutcome::Created(id) => id.raw(),
        other => panic!("expected created clip, got {other:?}"),
    };
    bridge
        .apply(&WireCommand::TrimClip(wire::TrimClip {
            clip: right,
            start: 4.0,
            duration: 2.0,
        }))
        .expect("trim clip");
    bridge.end_group();
    assert_eq!(plan.len(), 4);

    let phases = split_plan_phases(plan, &[2]);
    agent_replay(&mut live, phases, |_| {}).expect("replay");

    let summary = summarize(live.project());
    let v1 = summary
        .tracks
        .iter()
        .find(|t| t.name == "V1")
        .expect("replayed lane");
    assert_ne!(v1.id, track, "live allocated fresh ids — the remap is real");
    assert_eq!(v1.clips.len(), 2);
    assert_eq!(
        (v1.clips[0].start_seconds, v1.clips[0].duration_seconds),
        (0.0, 4.0)
    );
    assert_eq!(
        (v1.clips[1].start_seconds, v1.clips[1].duration_seconds),
        (4.0, 2.0)
    );
    // Without the remap, the split would have hit the seed clip
    // (which reuses the sandbox's head id on the live engine).
    let seed = live.project().timeline().clip(seed_clip).expect("seed");
    assert_eq!(seed.timeline, TimeRange::at_rate(0, 24, Rational::FPS_24));

    // Two phases ⇒ two undo steps: first undo removes only phase 2.
    assert!(live.undo(), "undo phase 2");
    let summary = summarize(live.project());
    let v1 = summary
        .tracks
        .iter()
        .find(|t| t.name == "V1")
        .expect("phase 1 remains");
    assert_eq!(v1.clips.len(), 1, "the split and trim are undone");
    assert_eq!(v1.clips[0].duration_seconds, 10.0);

    assert!(live.undo(), "undo phase 1");
    let summary = summarize(live.project());
    assert!(summary.tracks.iter().all(|t| t.name != "V1"));
    assert!(
        live.project().timeline().clip(seed_clip).is_some(),
        "the pre-existing timeline is untouched"
    );
    assert!(!live.undo(), "nothing left to undo");
}

#[test]
fn mid_phase_failure_keeps_earlier_phases_and_names_the_boundary() {
    let (project, media) = fixture_project();
    let mut sandbox = temp_engine(project.clone());
    let mut live = temp_engine(project);

    let mut plan: Vec<AgentPlanStep> = Vec::new();
    let mut senses = AgentSenses::new();
    let mut bridge = SandboxBridge {
        worker: &UNEXPECTED_PROJECT_SNAPSHOT,
        engine: &mut sandbox,
        plan: &mut plan,
        senses: &mut senses,
        default_playhead_seconds: 0.0,
    };
    bridge.begin_group();
    let track = match bridge
        .apply(&WireCommand::AddTrack(wire::AddTrack {
            kind: wire::WireTrackKind::Video,
            name: "V1".into(),
            index: None,
        }))
        .expect("add track")
    {
        EditOutcome::CreatedTrack(id) => id.raw(),
        other => panic!("expected created track, got {other:?}"),
    };
    bridge
        .apply(&WireCommand::AddClip(wire::AddClip {
            track,
            media,
            source_start: 0.0,
            source_duration: 10.0,
            start: 0.0,
        }))
        .expect("add clip");
    bridge.end_group();
    // Phase 2 goes stale before replay (as if the user deleted the
    // clip it targets mid-prompt).
    plan.push(AgentPlanStep {
        command: WireCommand::TrimClip(wire::TrimClip {
            clip: 999_999,
            start: 0.0,
            duration: 1.0,
        }),
        created: None,
    });

    let phases = split_plan_phases(plan, &[2]);
    let err = agent_replay(&mut live, phases, |_| {}).expect_err("phase 2 must fail");
    assert!(err.contains("phase 2/2"), "names the boundary: {err}");
    assert!(err.contains("step 1/1"), "{err}");
    assert!(
        err.contains("phase 1 of 2 was applied and stays undoable"),
        "{err}"
    );

    // Phase 1 landed; the failing phase 2 left no trace.
    let timeline = live.project().timeline();
    assert_eq!(timeline.track_count(), 1);
    assert_eq!(timeline.clip_count(), 1);

    assert!(live.undo(), "phase 1 is its own undo step");
    assert_eq!(live.project().timeline().track_count(), 0);
    assert!(!live.undo(), "the rolled-back phase 2 left no history");
}

#[test]
fn committed_phase_enforces_empty_lane_cleanup_before_a_later_failure() {
    let mut project = Project::new("agent-phase-cleanup", Rational::FPS_24);
    let main = project.add_track(TrackKind::Video, "V1");
    let stickers = project.add_track(TrackKind::Sticker, "Stickers");
    let sticker = project
        .add_generated(
            stickers,
            Generator::sticker(""),
            TimeRange::at_rate(0, 48, Rational::FPS_24),
        )
        .expect("sticker");
    let mut live = temp_engine(project);

    let phases = vec![
        vec![AgentPlanStep {
            command: WireCommand::RemoveClip(wire::RemoveClip {
                clip: sticker.raw(),
            }),
            created: None,
        }],
        vec![AgentPlanStep {
            command: WireCommand::TrimClip(wire::TrimClip {
                clip: 999_999,
                start: 0.0,
                duration: 1.0,
            }),
            created: None,
        }],
    ];
    agent_replay(&mut live, phases, |_| {}).expect_err("phase 2 must fail");

    let timeline = live.project().timeline();
    assert!(timeline.track(main).is_some(), "main lane remains");
    assert!(
        timeline.track(stickers).is_none(),
        "phase 1 commits a coherent desktop timeline"
    );
    assert!(timeline.clip(sticker).is_none());

    assert!(live.undo(), "phase 1 cleanup is in its undo step");
    let timeline = live.project().timeline();
    assert!(timeline.track(stickers).is_some());
    assert!(timeline.clip(sticker).is_some());
}
