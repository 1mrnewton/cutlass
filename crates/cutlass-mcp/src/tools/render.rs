//! Render tools — composited frame grab (PNG) and whole-timeline video export.

use std::path::PathBuf;

use base64::{Engine as _, prelude::BASE64_STANDARD};
use rmcp::{
    handler::server::wrapper::Parameters,
    model::{CallToolResult, ContentBlock},
    tool, tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::server::CutlassMcp;

/// Default max edge for `frame_get` when the agent omits `max_dim`.
const DEFAULT_MAX_DIM: u32 = 1024;
const MIN_MAX_DIM: u32 = 64;
const MAX_MAX_DIM: u32 = 4096;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct FrameGetParams {
    /// Timeline time in seconds (frame-snapped to the project rate).
    pub time: f64,
    /// Longest edge in pixels after fit-scale (default 1024, clamped to
    /// 64..=4096). Aspect is preserved; never upscaled past the canvas.
    pub max_dim: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExportVideoParams {
    /// Absolute destination path for the H.264/AAC MP4.
    pub path: String,
}

fn clamp_max_dim(max_dim: Option<u32>) -> u32 {
    max_dim
        .unwrap_or(DEFAULT_MAX_DIM)
        .clamp(MIN_MAX_DIM, MAX_MAX_DIM)
}

#[tool_router(router = render_router, vis = "pub(crate)")]
impl CutlassMcp {
    /// Render the composited frame at `time` (what export would contain),
    /// scaled to fit `max_dim`. Returns an image content block plus a short
    /// text caption so agents can look at the frame and log dimensions.
    #[tool(
        description = "Render the composited frame at time (seconds) as a PNG scaled to fit max_dim — for agents to visually verify edits. Returns image + caption text.",
        annotations(read_only_hint = true)
    )]
    async fn frame_get(&self, Parameters(params): Parameters<FrameGetParams>) -> CallToolResult {
        let max_dim = clamp_max_dim(params.max_dim);
        match self.host.get_frame(params.time, max_dim).await {
            Ok(grab) => {
                let b64 = BASE64_STANDARD.encode(&grab.png);
                CallToolResult::success(vec![
                    ContentBlock::image(b64, "image/png"),
                    ContentBlock::text(format!(
                        "frame at {}s — {}x{} PNG",
                        grab.seconds, grab.width, grab.height
                    )),
                ])
            }
            Err(e) => CallToolResult::error(vec![ContentBlock::text(e)]),
        }
    }

    /// Render the whole timeline to an H.264/AAC MP4. Blocks until done
    /// (can take minutes); other tools queue behind it on the engine thread.
    #[tool(
        description = "Export the whole timeline to an H.264/AAC MP4 at an absolute path. Synchronous — blocks until done (can take minutes); other tools queue behind it. Platform encoders (macOS/Windows; not supported on Linux)."
    )]
    async fn export_video(
        &self,
        Parameters(params): Parameters<ExportVideoParams>,
    ) -> Result<String, String> {
        let done = self.host.export_video(PathBuf::from(params.path)).await?;
        Ok(format!("exported {} ({} frames)", done.path, done.frames))
    }
}
