//! MCP server handler and tool router.

use cutlass_ai::TOOL_SCHEMA_VERSION;
use rmcp::{
    ServerHandler,
    handler::server::router::tool::ToolRouter,
    model::{Implementation, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
};

use crate::host::EngineHost;

/// Headless Cutlass MCP service.
///
/// Owns an [`EngineHost`] (engine on a dedicated OS thread) and the combined
/// rmcp tool router (probe + project lifecycle + validated edits).
#[derive(Clone)]
pub struct CutlassMcp {
    pub(crate) host: EngineHost,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl CutlassMcp {
    pub fn new() -> Self {
        Self {
            host: EngineHost::spawn(),
            tool_router: Self::tool_router() + Self::project_router() + Self::edits_router(),
        }
    }

    /// Trivial read-only probe so hosts can confirm the server is alive.
    #[tool(
        description = "Cutlass MCP server version and wire tool-schema version",
        annotations(read_only_hint = true)
    )]
    fn version(&self) -> String {
        format!(
            "cutlass-mcp {} (tool schema v{TOOL_SCHEMA_VERSION})",
            env!("CARGO_PKG_VERSION")
        )
    }
}

impl Default for CutlassMcp {
    fn default() -> Self {
        Self::new()
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for CutlassMcp {
    fn get_info(&self) -> ServerInfo {
        // ServerInfo is non-exhaustive in rmcp 2.x — use the builder helpers.
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("cutlass", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Cutlass video-editor project control over MCP. \
                 Lifecycle: project_new / project_open → project_save → project_get; \
                 media_import registers pool entries (does not place clips). \
                 Edits: edit_commands_list → edit_schema_get for argument shapes → \
                 edit_apply (batch of {\"command\":\"<name>\", ...args}, one undo group, \
                 all-or-nothing, validated against live project state) → verify with \
                 project_get. edit_undo / edit_redo reverse one edit_apply batch. \
                 Never invent raw engine mutations — only the wire vocabulary.",
            )
    }
}
