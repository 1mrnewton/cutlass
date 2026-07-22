use super::RT_FORMAT;

/// Reused canvas-sized RGBA textures for effect ping-pong and transitions.
pub(crate) struct OffscreenPool {
    pub(crate) width: u32,
    pub(crate) height: u32,
    // The views keep the underlying textures alive; no need to store them.
    views: [wgpu::TextureView; 3],
}

impl OffscreenPool {
    pub fn ensure(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let make = |label: &str| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: RT_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        };
        let (_t0, v0) = make("cutlass.offscreen.a");
        let (_t1, v1) = make("cutlass.offscreen.b");
        let (_t2, v2) = make("cutlass.offscreen.c");
        Self {
            width,
            height,
            views: [v0, v1, v2],
        }
    }

    pub fn view(&self, index: usize) -> &wgpu::TextureView {
        &self.views[index % 3]
    }
}
