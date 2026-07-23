//! MCP server handler and tool router.

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
/// rmcp tool router. Edit tools will attach here in later milestones.
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
            // Base probe + project lifecycle routers combined with `+`.
            tool_router: Self::tool_router() + Self::project_router(),
        }
    }

    /// Trivial read-only probe so hosts can confirm the server is alive.
    #[tool(
        description = "Cutlass MCP server version and schema info",
        annotations(read_only_hint = true)
    )]
    fn version(&self) -> String {
        format!("cutlass-mcp {}", env!("CARGO_PKG_VERSION"))
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
                "Cutlass video-editor project control over MCP. Create or open \
                 .cutlass projects (project_new / project_open), save them \
                 (project_save), inspect the compact timeline/media summary \
                 (project_get), and import media into the pool (media_import). \
                 Validated edit tools arrive in a later milestone — do not \
                 invent raw engine mutations.",
            )
    }
}
