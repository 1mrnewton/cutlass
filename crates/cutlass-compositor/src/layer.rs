/// Canvas dimensions for a composite pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompositorConfig {
    pub width: u32,
    pub height: u32,
}

impl CompositorConfig {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

/// One layer in bottom-to-top stacking order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompositeLayer {
    /// Full-canvas RGBA8 (width×height×4), row-major.
    Rgba { bytes: Vec<u8> },
    /// Full-canvas solid fill (RGBA 0–255).
    Solid { rgba: [u8; 4] },
}
