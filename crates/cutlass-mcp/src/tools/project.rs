//! Project lifecycle and media-pool tools.

use std::path::PathBuf;

use rmcp::{handler::server::wrapper::Parameters, tool, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::server::CutlassMcp;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ProjectNewParams {
    /// Project display name. Defaults to `"untitled"`.
    pub name: Option<String>,
    /// Timeline frame rate in fps. Defaults to `30`. Must match a supported
    /// named rate (24, 23.976, 25, 30, 29.97, 50, 60, 59.94).
    pub fps: Option<f64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ProjectOpenParams {
    /// Absolute path to a `.cutlass` project file on this machine.
    pub path: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ProjectSaveParams {
    /// Destination `.cutlass` path. Omit to save back to the path the
    /// project was opened from or last saved to.
    pub path: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct MediaImportParams {
    /// Absolute paths to media files on this machine (video, audio, or
    /// still image). Each file is probed and registered in the media pool.
    pub paths: Vec<String>,
}

#[tool_router(router = project_router, vis = "pub(crate)")]
impl CutlassMcp {
    /// Create a new empty project with a magnetic Main video track.
    ///
    /// Replaces any currently open project. Frame rate must be a supported
    /// named rate — arbitrary fps values are rejected.
    #[tool(description = "Create a new empty .cutlass project with a Main video track")]
    async fn project_new(
        &self,
        Parameters(params): Parameters<ProjectNewParams>,
    ) -> Result<String, String> {
        let name = params.name.unwrap_or_else(|| "untitled".into());
        let fps = params.fps.unwrap_or(30.0);
        let meta = self.host.new_project(name, fps).await?;
        serde_json::to_string_pretty(&meta).map_err(|e| e.to_string())
    }

    /// Open a `.cutlass` project file.
    ///
    /// Uses Load semantics: missing media paths are tolerated so a moved
    /// project still opens; offline media is visible in `project_get` for
    /// later relink.
    #[tool(description = "Open a .cutlass project file (tolerates missing media; relink later)")]
    async fn project_open(
        &self,
        Parameters(params): Parameters<ProjectOpenParams>,
    ) -> Result<String, String> {
        let meta = self.host.open_project(PathBuf::from(params.path)).await?;
        serde_json::to_string_pretty(&meta).map_err(|e| e.to_string())
    }

    /// Save the open project to disk.
    ///
    /// When `path` is omitted, writes to the path the project was opened
    /// from or last saved to. Errors if the project has never been given a
    /// file path.
    #[tool(description = "Save the open project to a .cutlass file")]
    async fn project_save(
        &self,
        Parameters(params): Parameters<ProjectSaveParams>,
    ) -> Result<String, String> {
        let path = params.path.map(PathBuf::from);
        let saved = self.host.save_project(path).await?;
        Ok(saved.display().to_string())
    }

    /// Return session meta plus the compact project summary the in-app
    /// agent uses (tracks, media pool, canvas, duration).
    #[tool(
        description = "Read the open project: session meta + compact timeline/media summary",
        annotations(read_only_hint = true)
    )]
    async fn project_get(&self) -> Result<String, String> {
        let doc = self.host.get_project().await?;
        serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())
    }

    /// Import media files into the open project's media pool.
    ///
    /// Paths must be absolute paths on this machine. Imported media is
    /// referenced by clips via media id; this tool only registers pool
    /// entries (it does not place clips on the timeline). Per-path failures
    /// are reported individually without aborting the batch.
    #[tool(
        description = "Import absolute-path media files (video/audio/image) into the project media pool"
    )]
    async fn media_import(
        &self,
        Parameters(params): Parameters<MediaImportParams>,
    ) -> Result<String, String> {
        let paths = params.paths.into_iter().map(PathBuf::from).collect();
        let results = self.host.import_media(paths).await?;
        serde_json::to_string_pretty(&results).map_err(|e| e.to_string())
    }
}
