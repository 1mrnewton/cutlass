use cutlass_core::{CpuImage, MatrixCoefficients, PixelFormat, Plane, VideoFrame};

use crate::error::CompositorError;
use crate::gpu::GpuContext;
use crate::grade::ColorGrade;
use crate::layer::{CompositeLayer, CompositorConfig, LayerPlacement};

/// Upload a CPU frame's planes to fresh textures per its pixel layout, honoring
/// each plane's byte stride. Returns the plane textures and whether the layout
/// is NV12 (biplanar) — which selects the YUV shader's chroma path.
pub(super) fn upload_cpu_planes(
    gpu: &GpuContext,
    frame: &VideoFrame,
    image: &CpuImage,
    cw: u32,
    ch: u32,
) -> Result<(Vec<(wgpu::Texture, wgpu::TextureView)>, bool), CompositorError> {
    match frame.format {
        PixelFormat::Nv12 => {
            expect_planes(image.planes.len(), 2, frame.format)?;
            let y = upload_plane(
                gpu,
                "y",
                wgpu::TextureFormat::R8Unorm,
                cw,
                ch,
                &image.planes[0],
            )?;
            let uv = upload_plane(
                gpu,
                "uv",
                wgpu::TextureFormat::Rg8Unorm,
                cw.div_ceil(2),
                ch.div_ceil(2),
                &image.planes[1],
            )?;
            Ok((vec![y, uv], true))
        }
        PixelFormat::Yuv420p => {
            expect_planes(image.planes.len(), 3, frame.format)?;
            let y = upload_plane(
                gpu,
                "y",
                wgpu::TextureFormat::R8Unorm,
                cw,
                ch,
                &image.planes[0],
            )?;
            let u = upload_plane(
                gpu,
                "u",
                wgpu::TextureFormat::R8Unorm,
                cw.div_ceil(2),
                ch.div_ceil(2),
                &image.planes[1],
            )?;
            let v = upload_plane(
                gpu,
                "v",
                wgpu::TextureFormat::R8Unorm,
                cw.div_ceil(2),
                ch.div_ceil(2),
                &image.planes[2],
            )?;
            Ok((vec![y, u, v], false))
        }
        other => Err(CompositorError::UnsupportedFormat(format!(
            "{other:?} is not handled by the compositor yet (8-bit NV12 / I420 only)"
        ))),
    }
}

/// Upload one plane to a fresh texture, honoring its byte stride.
fn upload_plane(
    gpu: &GpuContext,
    label: &str,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    plane: &Plane,
) -> Result<(wgpu::Texture, wgpu::TextureView), CompositorError> {
    let need = plane
        .stride
        .checked_mul(height as usize)
        .ok_or_else(|| CompositorError::MalformedFrame("plane size overflow".into()))?;
    if plane.data.len() < need {
        return Err(CompositorError::MalformedFrame(format!(
            "{label} plane: {} bytes < stride {} × {} rows",
            plane.data.len(),
            plane.stride,
            height
        )));
    }

    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &plane.data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(plane.stride as u32),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    Ok((texture, view))
}

fn expect_planes(got: usize, want: usize, format: PixelFormat) -> Result<(), CompositorError> {
    if got == want {
        Ok(())
    } else {
        Err(CompositorError::MalformedFrame(format!(
            "{format:?} needs {want} planes, got {got}"
        )))
    }
}

/// Luma coefficients (Kr, Kb) for the YUV→RGB matrix. `resolved()` upstream maps
/// `Unspecified` to BT.709, so only concrete matrices reach here; `Identity` is
/// degenerate on YUV and falls back to BT.709.
pub(super) fn luma_coeffs(matrix: MatrixCoefficients) -> (f32, f32) {
    match matrix {
        MatrixCoefficients::Bt601 => (0.299, 0.114),
        MatrixCoefficients::Bt2020Ncl => (0.2627, 0.0593),
        _ => (0.2126, 0.0722),
    }
}

/// Effective clockwise rotation (radians): the placement plus the frame's
/// container rotation, so sideways-recorded phone video plays upright.
pub(super) fn placement_rotation(layer: &CompositeLayer<'_>, frame: &VideoFrame) -> f32 {
    layer.placement.rotation + (frame.rotation.degrees() as f32).to_radians()
}

/// Map the layer's UV rect (in visible-picture space) into the coded texture,
/// since the visible picture may be a sub-rect of a padded decode buffer.
pub(super) fn visible_uv_rect(frame: &VideoFrame, uv: [f32; 4]) -> [f32; 4] {
    let (cw, ch) = (frame.coded_size.0 as f32, frame.coded_size.1 as f32);
    let vx = frame.visible.x as f32 / cw;
    let vy = frame.visible.y as f32 / ch;
    let vw = frame.visible.width as f32 / cw;
    let vh = frame.visible.height as f32 / ch;
    [
        vx + uv[0] * vw,
        vy + uv[1] * vh,
        vx + uv[2] * vw,
        vy + uv[3] * vh,
    ]
}

pub(super) fn placement_affine(
    config: &CompositorConfig,
    p: &LayerPlacement,
    extra_rotation: f32,
) -> ([f32; 4], [f32; 4]) {
    placement_affine_rot(config, p, p.rotation + extra_rotation)
}

/// Build the unit-quad ([-0.5, 0.5]², +y down) → clip-space affine for a
/// placement, using `rotation` (radians, clockwise) instead of `p.rotation`.
pub(super) fn placement_affine_rot(
    config: &CompositorConfig,
    p: &LayerPlacement,
    rotation: f32,
) -> ([f32; 4], [f32; 4]) {
    let (cw, ch) = (config.width as f32, config.height as f32);
    let (cos, sin) = (rotation.cos(), rotation.sin());
    // Canvas space (+y down): pos = center + R·(corner ⊙ size), R the clockwise
    // rotation [cos, -sin; sin, cos] in screen coords.
    let a = cos * p.size[0];
    let b = sin * p.size[0];
    let c = -sin * p.size[1];
    let d = cos * p.size[1];
    // Canvas px → clip space: x' = 2x/cw − 1, y' = 1 − 2y/ch (flip y).
    let linear = [2.0 * a / cw, -2.0 * b / ch, 2.0 * c / cw, -2.0 * d / ch];
    let trans = [
        2.0 * p.center[0] / cw - 1.0,
        1.0 - 2.0 * p.center[1] / ch,
        p.opacity,
        0.0,
    ];
    (linear, trans)
}

pub(super) fn pack_grade(grade: Option<ColorGrade>) -> ([f32; 4], [f32; 4], [f32; 4]) {
    match grade.filter(|g| !g.is_identity()) {
        Some(g) => (
            [g.brightness, g.contrast, g.saturation, 1.0],
            [g.exposure, g.temperature, g.tint, g.hue],
            [g.highlights, g.shadows, g.sharpness, g.vignette],
        ),
        None => ([0.0; 4], [0.0; 4], [0.0; 4]),
    }
}
