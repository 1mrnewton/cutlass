//! WGPU frame compositor for Cutlass preview and export.
//!
//! Layers are composited **bottom-to-top** with src-over alpha blending.
//! [`GpuContext::new_headless_blocking`] is the default entry point for engine
//! and tests. Future Slint UI should create one shared [`GpuContext`] and pass
//! it to both Slint (`WGPUConfiguration::Manual`) and [`Compositor::new`].

mod compositor;
mod error;
mod gpu;
mod image;
mod layer;

pub use compositor::Compositor;
pub use error::CompositorError;
pub use gpu::GpuContext;
pub use image::RgbaImage;
pub use layer::{CompositeLayer, CompositorConfig};

use tracing::info;

pub fn init() {
    info!("cutlass-compositor ready");
}
