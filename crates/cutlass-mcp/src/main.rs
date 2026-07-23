//! Binary entry for the Cutlass MCP server (stdio JSON-RPC transport).

use cutlass_mcp::server::CutlassMcp;
use rmcp::{ServiceExt, transport::stdio};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Logging must stay on stderr — stdout is the MCP JSON-RPC transport.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let service = CutlassMcp::new().serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}
