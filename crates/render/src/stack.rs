//! What the mask passes share: a clip's masks as one uniform block, and the
//! one-byte-a-pixel pictures they read (a selection, or the drawn masks'
//! atlas) as textures kept from frame to frame.

use cutlass_effects::{CanvasMasks, STACK_FLOATS};

use crate::pool::{Pool, TextureKey};

/// Bytes in a mask stack's uniform block.
pub(crate) const STACK_BYTES: u64 = (STACK_FLOATS * 4) as u64;

/// `masks`' uniform block as bytes.
pub(crate) fn stack_bytes(masks: &CanvasMasks) -> Vec<u8> {
    masks
        .uniform()
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// `masks`' drawn pictures as one texture from `pool`.
pub(crate) fn atlas_view(
    pool: &Pool,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    masks: &CanvasMasks,
) -> Result<wgpu::TextureView, crate::RenderError> {
    let (width, height, bytes) = masks.atlas();
    alpha_view(pool, device, queue, width, height, &bytes)
}

/// A `width` by `height` picture of one byte a pixel as a texture from
/// `pool`. A picture past the device's textures is refused.
pub(crate) fn alpha_view(
    pool: &Pool,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    bytes: &[u8],
) -> Result<wgpu::TextureView, crate::RenderError> {
    let limit = device.limits().max_texture_dimension_2d;
    if width == 0 || height == 0 || width > limit || height > limit {
        return Err(crate::RenderError::Size { width, height });
    }
    let expected = width as usize * height as usize;
    if bytes.len() != expected {
        return Err(crate::RenderError::Bytes {
            expected,
            actual: bytes.len(),
        });
    }
    let (texture, view) = pool.texture(
        device,
        "cutlass-render mask picture",
        TextureKey {
            size: (width, height),
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        },
    );
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    Ok(view)
}
