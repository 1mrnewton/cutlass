use super::RT_FORMAT;

/// Reused canvas-sized RGBA textures for effect ping-pong, transitions, and
/// blend-mode canvas snapshots.
///
/// Slot invariant for non-`Normal` blend composite:
/// - Layer base draws into slot 0; effects/LUT ping-pong and may finish on
///   any of slots 0..=2 (gaussian blur ends on 2).
/// - After the layer chain finishes on slot `S`, the canvas is
///   `copy_texture_to_texture`'d into `(S + 1) % SLOTS` as the dst snapshot.
/// - `blend_composite` then samples snapshot + `S` and writes the canvas.
/// - Slot `(S + 2) % SLOTS` is unused for that step. Three slots therefore
///   suffice (canvas is a separate texture); the pool does not need a 4th.
pub(crate) struct OffscreenPool {
    pub(crate) width: u32,
    pub(crate) height: u32,
    textures: [wgpu::Texture; Self::SLOTS],
    views: [wgpu::TextureView; Self::SLOTS],
}

impl OffscreenPool {
    pub const SLOTS: usize = 3;

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
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        };
        let (t0, v0) = make("cutlass.offscreen.a");
        let (t1, v1) = make("cutlass.offscreen.b");
        let (t2, v2) = make("cutlass.offscreen.c");
        Self {
            width,
            height,
            textures: [t0, t1, t2],
            views: [v0, v1, v2],
        }
    }

    pub fn view(&self, index: usize) -> &wgpu::TextureView {
        &self.views[index % Self::SLOTS]
    }

    pub fn texture(&self, index: usize) -> &wgpu::Texture {
        &self.textures[index % Self::SLOTS]
    }

    /// Pool slot holding `view`, if it is one of this pool's views.
    pub fn index_of(&self, view: &wgpu::TextureView) -> Option<usize> {
        self.views.iter().position(|v| std::ptr::eq(v, view))
    }
}
