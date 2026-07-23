//! MCP server handler and tool router.

use rmcp::{
    ServerHandler,
    handler::server::router::tool::ToolRouter,
    model::{Implementation, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
};

/// Headless Cutlass MCP service.
///
/// Holds the rmcp tool router; later milestones will attach project state
/// and validated edit tooling here without changing the transport.
#[derive(Clone)]
pub struct CutlassMcp {
    // Accessed by `#[tool_handler]` generated code; rustc can't see that use.
    #[expect(dead_code)]
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl CutlassMcp {
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    /// Trivial read-only probe so the router wiring is exercised end-to-end
    /// before real project/edit tools land.
    #[tool(description = "Cutlass MCP server version and schema info")]
    fn version(&self) -> String {
        format!("cutlass-mcp {}", env!("CARGO_PKG_VERSION"))
    }
}

impl Default for CutlassMcp {
    fn default() -> Self {
        Self::new()
    }
}

#[tool_handler]
impl ServerHandler for CutlassMcp {
    fn get_info(&self) -> ServerInfo {
        // ServerInfo is non-exhaustive in rmcp 2.x — use the builder helpers.
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("cutlass", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Cutlass video-editor project control over MCP. Open or create \
                 .cutlass projects, inspect timelines, and apply validated edits. \
                 Edit tools arrive in later milestones; this scaffold only exposes \
                 a version probe.",
            )
    }
}
