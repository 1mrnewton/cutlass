use tracing::info;
use tracing_subscriber::EnvFilter;

fn setup_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
}

fn main() {
    setup_tracing();
    cutlass_decoder::init();
    cutlass_encoder::init();
    cutlass_models::init();
    cutlass_compositor::init();
    cutlass_engine::init();
    info!("cutlass-app ready");
}
