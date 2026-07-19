use slint::BackendSelector;
use slint::wgpu_28::WGPUConfiguration;

slint::include_modules!();

fn main() -> Result<(), slint::PlatformError> {
    BackendSelector::new()
        .require_wgpu_28(WGPUConfiguration::default())
        .select()?;

    let app = GalleryWindow::new()?;
    app.run()
}
