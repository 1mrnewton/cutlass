//! In-process MCP client ↔ CutlassMcp server over a duplex pipe.
//!
//! Proves the wire contract (tool catalog, annotations, content types,
//! edit/error shapes) rather than host internals alone.

use std::collections::BTreeSet;
use std::sync::LazyLock;

use base64::{Engine as _, prelude::BASE64_STANDARD};
use cutlass_mcp::server::CutlassMcp;
use rmcp::{
    ServiceExt,
    model::{CallToolRequestParams, CallToolResult, ContentBlock},
};
use serde_json::{Value, json};
use tokio::sync::Mutex;

/// 1 MiB — large enough for PNG frame payloads in `frame_get`.
const DUPLEX_BUF: usize = 1 << 20;

/// Serialize tests that allocate `TrackId`s so a single-batch
/// `add_track` + `add_generated` can predict the new track id.
static ENGINE_ID_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

struct Session {
    client: rmcp::service::RunningService<rmcp::RoleClient, ()>,
    server: tokio::task::JoinHandle<()>,
}

impl Session {
    async fn connect() -> Self {
        let (client_io, server_io) = tokio::io::duplex(DUPLEX_BUF);
        let server = tokio::spawn(async move {
            let service = match CutlassMcp::new().serve(server_io).await {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("MCP server failed to start: {e}");
                    return;
                }
            };
            let _ = service.waiting().await;
        });
        let client = ().serve(client_io).await.expect("MCP client handshake");
        Self { client, server }
    }

    async fn call(&self, name: &str, arguments: Value) -> CallToolResult {
        let params = match arguments {
            Value::Null => CallToolRequestParams::new(name.to_string()),
            Value::Object(map) => CallToolRequestParams::new(name.to_string()).with_arguments(map),
            other => panic!("tool arguments must be a JSON object or null, got {other}"),
        };
        self.client
            .call_tool(params)
            .await
            .unwrap_or_else(|e| panic!("call_tool({name}): {e}"))
    }

    async fn call_ok_text(&self, name: &str, arguments: Value) -> String {
        let result = self.call(name, arguments).await;
        assert_ne!(
            result.is_error,
            Some(true),
            "{name} should succeed: {}",
            content_text(&result)
        );
        content_text(&result)
    }

    async fn shutdown(self) {
        let _ = self.client.cancel().await;
        self.server.abort();
    }
}

fn content_text(result: &CallToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|b| b.as_text().map(|t| t.text.clone()))
        .collect::<Vec<_>>()
        .join("\n")
}

fn is_read_only(tool: &rmcp::model::Tool) -> bool {
    tool.annotations
        .as_ref()
        .and_then(|a| a.read_only_hint)
        .unwrap_or(false)
}

/// Predict the next track id for chaining `add_track` + `add_generated` in
/// one batch. Ids are process-global atomics, so a fresh server usually yields
/// Main=1 / sticker=2; parallel tests may allocate higher ids.
fn next_track_id(project_doc: &Value) -> u64 {
    project_doc["project"]["tracks"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|t| t["id"].as_u64())
        .max()
        .unwrap_or(0)
        + 1
}

fn sticker_clip_count(project_doc: &Value) -> usize {
    project_doc["project"]["tracks"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|t| t["kind"] == "sticker")
        .map(|t| t["clips"].as_array().map(|c| c.len()).unwrap_or(0))
        .sum()
}

#[tokio::test]
async fn tool_catalog_and_annotations() {
    let session = Session::connect().await;

    let tools = session
        .client
        .list_all_tools()
        .await
        .expect("list_all_tools");
    let names: BTreeSet<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
    let expected: BTreeSet<&str> = [
        "version",
        "project_new",
        "project_open",
        "project_save",
        "project_get",
        "media_import",
        "edit_commands_list",
        "edit_schema_get",
        "edit_apply",
        "edit_undo",
        "edit_redo",
        "frame_get",
        "export_video",
    ]
    .into_iter()
    .collect();
    assert_eq!(names, expected, "tool catalog mismatch");

    for name in [
        "project_get",
        "frame_get",
        "edit_commands_list",
        "edit_schema_get",
        "version",
    ] {
        let tool = tools.iter().find(|t| t.name == name).expect(name);
        assert!(
            is_read_only(tool),
            "{name} must advertise readOnlyHint=true; annotations={:?}",
            tool.annotations
        );
    }

    for name in ["edit_apply", "export_video"] {
        let tool = tools.iter().find(|t| t.name == name).expect(name);
        assert!(
            !is_read_only(tool),
            "{name} must not advertise readOnlyHint=true; annotations={:?}",
            tool.annotations
        );
    }

    let edit_apply = tools.iter().find(|t| t.name == "edit_apply").unwrap();
    let schema_text = Value::Object((*edit_apply.input_schema).clone()).to_string();
    assert!(
        schema_text.contains("edits"),
        "edit_apply input schema should mention edits: {schema_text}"
    );

    session.shutdown().await;
}

#[tokio::test]
async fn full_edit_flow() {
    let _id_guard = ENGINE_ID_LOCK.lock().await;
    let session = Session::connect().await;

    let meta_text = session
        .call_ok_text("project_new", json!({ "fps": 30.0 }))
        .await;
    let meta: Value = serde_json::from_str(&meta_text).expect("project_new JSON");
    let project_name = meta["name"].as_str().expect("name").to_string();

    let get_text = session.call_ok_text("project_get", Value::Null).await;
    let doc: Value = serde_json::from_str(&get_text).expect("project_get JSON");
    // Fresh process: Main=1 → sticker=2. Under the id lock, max(existing)+1
    // is the next TrackId::next() even if earlier tests already allocated.
    let track = next_track_id(&doc);

    let apply = session
        .call(
            "edit_apply",
            json!({
                "edits": [
                    {
                        "command": "add_track",
                        "kind": "sticker",
                        "name": "Overlays"
                    },
                    {
                        "command": "add_generated",
                        "track": track,
                        "generator": { "type": "solid", "rgba": [0, 0, 255, 255] },
                        "start": 0.0,
                        "duration": 4.0
                    }
                ]
            }),
        )
        .await;
    assert_ne!(apply.is_error, Some(true), "{}", content_text(&apply));
    let batch: Value = serde_json::from_str(&content_text(&apply)).expect("edit_apply JSON");
    let clip = batch["applied"][1]["outcome"]["clip"]
        .as_u64()
        .expect("applied[1].outcome.clip");
    assert!(clip > 0, "clip id should be positive");

    let get_text = session.call_ok_text("project_get", Value::Null).await;
    let doc: Value = serde_json::from_str(&get_text).expect("project_get JSON");
    assert_eq!(
        sticker_clip_count(&doc),
        1,
        "one clip on sticker track: {doc}"
    );

    let frame = session
        .call("frame_get", json!({ "time": 1.0, "max_dim": 128 }))
        .await;
    assert_ne!(frame.is_error, Some(true), "{}", content_text(&frame));
    let images: Vec<&ContentBlock> = frame
        .content
        .iter()
        .filter(|b| b.as_image().is_some())
        .collect();
    let texts: Vec<&ContentBlock> = frame
        .content
        .iter()
        .filter(|b| b.as_text().is_some())
        .collect();
    assert_eq!(
        images.len(),
        1,
        "expected one image block: {:?}",
        frame.content
    );
    assert_eq!(
        texts.len(),
        1,
        "expected one text block: {:?}",
        frame.content
    );

    let image = images[0].as_image().unwrap();
    assert_eq!(image.mime_type, "image/png");
    let png = BASE64_STANDARD
        .decode(&image.data)
        .expect("frame_get image data must be base64");
    assert!(
        png.starts_with(b"\x89PNG\r\n\x1a\n"),
        "PNG magic missing: {:?}",
        &png[..png.len().min(8)]
    );

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("flow.cutlass");
    let path_str = path.to_string_lossy().to_string();
    let saved = session
        .call_ok_text("project_save", json!({ "path": path_str }))
        .await;
    assert!(
        path.is_file(),
        "project_save should create {path_str}; tool said {saved}"
    );

    let opened = session
        .call_ok_text("project_open", json!({ "path": path_str }))
        .await;
    let opened_meta: Value = serde_json::from_str(&opened).expect("project_open JSON");
    assert_eq!(
        opened_meta["name"].as_str(),
        Some(project_name.as_str()),
        "re-opened meta name"
    );

    session.shutdown().await;
}

#[tokio::test]
async fn edit_errors_are_tool_results() {
    let _id_guard = ENGINE_ID_LOCK.lock().await;
    let session = Session::connect().await;
    session
        .call_ok_text("project_new", json!({ "fps": 30.0 }))
        .await;

    let missing = session
        .call(
            "edit_apply",
            json!({
                "edits": [{ "command": "split_clip", "clip": 999, "at": 1.0 }]
            }),
        )
        .await;
    assert_eq!(
        missing.is_error,
        Some(true),
        "missing clip should be isError"
    );
    let missing_text = content_text(&missing);
    assert!(
        missing_text.contains("999") || missing_text.to_ascii_lowercase().contains("clip"),
        "readable rejection: {missing_text}"
    );

    let bogus = session
        .call(
            "edit_apply",
            json!({
                "edits": [{ "command": "not_a_real_tool", "clip": 1 }]
            }),
        )
        .await;
    assert_eq!(
        bogus.is_error,
        Some(true),
        "bogus command should be isError"
    );
    let bogus_text = content_text(&bogus);
    assert!(
        bogus_text.contains("not_a_real_tool"),
        "error should name the bogus command: {bogus_text}"
    );

    // Protocol stays healthy after tool-level errors.
    let version = session.call_ok_text("version", Value::Null).await;
    assert!(version.contains("cutlass-mcp"), "{version}");

    session.shutdown().await;
}
