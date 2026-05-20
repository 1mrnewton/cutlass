//! Cutlass shell entry point.
//!
//! Boot sequence:
//!   1. Init tracing + WGPU-backed Slint backend.
//!   2. Build the in-memory `models::Project` (domain types).
//!   3. Convert it into the Slint `ui::Project` DTO and seed `AppState`.
//!   4. Run the event loop.

mod convert;
mod demo;

pub mod ui {
    //! Slint-generated types live here so they don't collide with the
    //! domain types from `models` (both expose `Project`, `Clip`, etc.).
    //! Outside this module use `ui::Project` for the DTO, `Project` for
    //! the domain.
    slint::include_modules!();
}

use models::Project;
use slint::BackendSelector;
use slint::ComponentHandle;
use slint::wgpu_28::WGPUConfiguration;
use tracing_subscriber::EnvFilter;

use crate::ui::{AppState, AppWindow, TimelineState};

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
    seed_demo_project(&app);
    app.run()?;
    Ok(())
}

fn seed_demo_project(app: &AppWindow) {
    let project: Project = demo::build_demo_project();

    // Push the source frame rate into the timeline ruler so its frame-mode
    // labels stay correct. Everything else the ruler needs is computed in
    // Slint from this + `zoom`.
    let fps = project.sequence.fps.as_f32().max(1.0);
    app.global::<TimelineState>().set_fps(fps);

    let dto: ui::Project = (&project).into();
    app.global::<AppState>().set_project(dto);

    // Wire pre-multiplied pixel geometry. Clip x/width and the sequence
    // viewport-width are computed in Rust at f64 precision (see
    // `convert::recompute_pixel_geometry`) — Slint just casts the result
    // to `length`. Two trigger points:
    //
    //   1. Initial pass right after `set_project`, so clips render at the
    //      correct position on the very first frame.
    //   2. The `geometry-dirty` callback fired by anything that changes
    //      zoom (today: the toolbar slider).
    //
    // The callback uses a Weak handle so we don't keep the AppWindow
    // alive past its natural drop point on shutdown.
    let zoom = app.global::<TimelineState>().get_zoom() as f64;
    convert::recompute_pixel_geometry(app, zoom);

    let weak = app.as_weak();
    app.global::<TimelineState>().on_geometry_dirty(move || {
        let Some(app) = weak.upgrade() else { return };
        let zoom = app.global::<TimelineState>().get_zoom() as f64;
        convert::recompute_pixel_geometry(&app, zoom);
    });
}
