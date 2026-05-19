slint::include_modules!();

use slint::BackendSelector;
use slint::wgpu_28::WGPUConfiguration;
use tracing_subscriber::EnvFilter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    BackendSelector::new()
        .require_wgpu_28(WGPUConfiguration::default())
        .select()?;

    let app = AppWindow::new()?;

    app.on_preview_playback_changed(|playing| {
        tracing::info!(playing, "preview playback");
    });

    app.run()?;
    Ok(())
}

fn create_project() -> UiProject {
    UiProject {
        id: "1".into(),
        name: "Project 1".into(),
        file_path: "project.cutlass".into(),
        schema: todo!(),
        sequences: todo!(),
        media_bin: todo!(),
        active_sequence_id: todo!(),
        is_dirty: todo!(),
    }
}
