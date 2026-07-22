//! Layer style passes: background plate, shadow, glow, and outline.
//!
//! Canvas order: background → shadow → glow → content → outline.
//!
//! Outline uses silhouette → iterated box blur → harden threshold. Blur +
//! threshold approximates morphological dilation; the effective width follows
//! [`run_style_blur`]'s radius cap ([`STYLE_BLUR_RADIUS_CAP`] px).

use bytemuck::{Pod, Zeroable};
use cutlass_shapes::{SDF_AA, SdfParams};
use wgpu::util::DeviceExt;

use crate::layer::{
    CompositeLayer, LayerBackground, LayerGlow, LayerOutline, LayerPlacement, LayerShadow, SdfLayer,
};

use super::OffscreenPool;
use super::draw_effect_pass;
use super::pipelines::PassRegistry;

/// Soft radius above this is clamped; with 3×16-tap iterations the effective
/// max is 48 px of iterated box blur (≈ gaussian).
const STYLE_BLUR_RADIUS_CAP: f32 = 48.0;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SilhouetteUniforms {
    rgba: [f32; 4],
    gain: f32,
    _pad: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OffsetUniforms {
    offset_uv: [f32; 2],
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HardenUniforms {
    rgba: [f32; 4],
}

/// Separable box blur for style silhouettes.
///
/// The catalog blur shaders cap taps at 16. Larger radii iterate H+V up to
/// three times (`ceil(radius/16)`, clamped 1..=3) with `per_pass = radius/iters`.
/// Repeated box blur approximates a gaussian. Radius is clamped to
/// [`STYLE_BLUR_RADIUS_CAP`] (48 px). Radii below 0.5 skip blurring.
///
/// Ping-pongs between `slot_a` and `slot_b`. Input must start in `slot_a`;
/// the blurred result ends in `slot_a`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_style_blur(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    registry: &PassRegistry,
    pool: &OffscreenPool,
    slot_a: usize,
    slot_b: usize,
    radius: f32,
    width: u32,
    height: u32,
) {
    let radius = radius.clamp(0.0, STYLE_BLUR_RADIUS_CAP);
    if radius < 0.5 {
        return;
    }
    let iters = ((radius / 16.0).ceil() as u32).clamp(1, 3);
    let per_pass = radius / iters as f32;
    for _ in 0..iters {
        draw_effect_pass(
            device,
            encoder,
            registry,
            &registry.blur_h,
            pool.view(slot_a),
            pool.view(slot_b),
            width,
            height,
            per_pass,
        );
        draw_effect_pass(
            device,
            encoder,
            registry,
            &registry.blur_v,
            pool.view(slot_b),
            pool.view(slot_a),
            width,
            height,
            per_pass,
        );
    }
}

/// Build a synthetic SDF rounded-rect layer for the background plate.
///
/// Placement is rebuilt from the content's center/rotation/opacity with a
/// larger size (`content + 2·padding` plus the SDF AA pad on each side) so the
/// plate shares the content's rotation without scaling the content's linear
/// matrix (which already folds size×rotation together).
pub(crate) fn background_plate_layer(
    content: &LayerPlacement,
    bg: LayerBackground,
) -> CompositeLayer<'static> {
    let plate_w = content.size[0] + 2.0 * bg.padding;
    let plate_h = content.size[1] + 2.0 * bg.padding;
    let half = [plate_w * 0.5, plate_h * 0.5];
    let radius = bg.radius.min(half[0].min(half[1]));
    // Same AA pad convention as shape resolve / CPU raster (no stroke).
    let aa_pad = 2.0 * SDF_AA;
    let placement = LayerPlacement {
        center: content.center,
        size: [plate_w + 2.0 * aa_pad, plate_h + 2.0 * aa_pad],
        rotation: content.rotation,
        opacity: content.opacity,
    };
    CompositeLayer::sdf(
        SdfLayer {
            shape: SdfParams::RoundedRect { radius }.with_half(half),
            fill: bg.rgba,
            stroke: None,
        },
        placement,
    )
}

/// Tint `src` alpha into a premultiplied silhouette in `output`.
fn run_silhouette(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    registry: &PassRegistry,
    src: &wgpu::TextureView,
    output: &wgpu::TextureView,
    rgba: [u8; 4],
    gain: f32,
) {
    let uniforms = SilhouetteUniforms {
        rgba: [
            f32::from(rgba[0]) / 255.0,
            f32::from(rgba[1]) / 255.0,
            f32::from(rgba[2]) / 255.0,
            f32::from(rgba[3]) / 255.0,
        ],
        gain,
        _pad: [0.0; 3],
    };
    let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cutlass.style.silhouette.uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cutlass.style.silhouette.bg"),
        layout: &registry.effect_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(src),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&registry.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: ubuf.as_entire_binding(),
            },
        ],
    });
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("cutlass.style.silhouette.pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: output,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(&registry.silhouette_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.draw(0..6, 0..1);
}

/// Premultiplied-over blit of `source` onto the canvas with a UV offset.
fn blit_offset_premultiplied_to_canvas(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    registry: &PassRegistry,
    source: &wgpu::TextureView,
    canvas_view: &wgpu::TextureView,
    offset_uv: [f32; 2],
) {
    let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cutlass.style.offset.uniforms"),
        contents: bytemuck::bytes_of(&OffsetUniforms {
            offset_uv,
            _pad: [0.0; 2],
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cutlass.style.offset.bg"),
        layout: &registry.offset_blit_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(source),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&registry.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: ubuf.as_entire_binding(),
            },
        ],
    });
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("cutlass.style.offset.pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: canvas_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(&registry.offset_blit_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.draw(0..6, 0..1);
}

/// Additive blit of premultiplied `source` onto the canvas (glow).
fn blit_additive_to_canvas(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    registry: &PassRegistry,
    source: &wgpu::TextureView,
    canvas_view: &wgpu::TextureView,
) {
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cutlass.style.additive.bg"),
        layout: &registry.blit_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(source),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&registry.sampler),
            },
        ],
    });
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("cutlass.style.additive.pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: canvas_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(&registry.additive_blit_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.draw(0..6, 0..1);
}

/// Harden the blurred silhouette into an outside stroke on the canvas.
fn blit_outline_harden_to_canvas(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    registry: &PassRegistry,
    blurred: &wgpu::TextureView,
    content: &wgpu::TextureView,
    canvas_view: &wgpu::TextureView,
    rgba: [u8; 4],
) {
    let uniforms = HardenUniforms {
        rgba: [
            f32::from(rgba[0]) / 255.0,
            f32::from(rgba[1]) / 255.0,
            f32::from(rgba[2]) / 255.0,
            f32::from(rgba[3]) / 255.0,
        ],
    };
    let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cutlass.style.harden.uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cutlass.style.harden.bg"),
        layout: &registry.harden_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(blurred),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(content),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&registry.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: ubuf.as_entire_binding(),
            },
        ],
    });
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("cutlass.style.harden.pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: canvas_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(&registry.harden_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.draw(0..6, 0..1);
}

/// Draw shadow then glow under the layer content.
///
/// Slot lifetimes (content already in `src_slot` = S):
/// - A = (S+1)%3, B = (S+2)%3 hold silhouette + blur ping-pong.
/// - Both are free again before the caller composites content, so a
///   non-`Normal` blend may still snapshot the canvas into (S+1)%3.
#[allow(clippy::too_many_arguments)]
pub(crate) fn composite_layer_styles(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    registry: &PassRegistry,
    pool: &OffscreenPool,
    content: &wgpu::TextureView,
    src_slot: usize,
    canvas_view: &wgpu::TextureView,
    shadow: Option<LayerShadow>,
    glow: Option<LayerGlow>,
    width: u32,
    height: u32,
) {
    let slot_a = (src_slot + 1) % OffscreenPool::SLOTS;
    let slot_b = (src_slot + 2) % OffscreenPool::SLOTS;
    let w = width.max(1) as f32;
    let h = height.max(1) as f32;

    if let Some(shadow) = shadow {
        run_silhouette(
            device,
            encoder,
            registry,
            content,
            pool.view(slot_a),
            shadow.rgba,
            1.0,
        );
        run_style_blur(
            device,
            encoder,
            registry,
            pool,
            slot_a,
            slot_b,
            shadow.blur,
            width,
            height,
        );
        blit_offset_premultiplied_to_canvas(
            device,
            encoder,
            registry,
            pool.view(slot_a),
            canvas_view,
            [shadow.offset[0] / w, shadow.offset[1] / h],
        );
    }

    if let Some(glow) = glow {
        run_silhouette(
            device,
            encoder,
            registry,
            content,
            pool.view(slot_a),
            glow.rgba,
            glow.intensity,
        );
        run_style_blur(
            device,
            encoder,
            registry,
            pool,
            slot_a,
            slot_b,
            glow.radius,
            width,
            height,
        );
        blit_additive_to_canvas(device, encoder, registry, pool.view(slot_a), canvas_view);
    }
}

/// Draw an outside outline after the layer content has been composited.
///
/// Skips when `width` or alpha is 0. Reuses A/B slots after content (and any
/// blend snapshot) so content in `src_slot` is left intact.
#[allow(clippy::too_many_arguments)]
pub(crate) fn composite_layer_outline(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    registry: &PassRegistry,
    pool: &OffscreenPool,
    content: &wgpu::TextureView,
    src_slot: usize,
    canvas_view: &wgpu::TextureView,
    outline: LayerOutline,
    width: u32,
    height: u32,
) {
    if outline.width <= 0.0 || outline.rgba[3] == 0 {
        return;
    }
    let slot_a = (src_slot + 1) % OffscreenPool::SLOTS;
    let slot_b = (src_slot + 2) % OffscreenPool::SLOTS;
    run_silhouette(
        device,
        encoder,
        registry,
        content,
        pool.view(slot_a),
        outline.rgba,
        1.0,
    );
    run_style_blur(
        device,
        encoder,
        registry,
        pool,
        slot_a,
        slot_b,
        outline.width,
        width,
        height,
    );
    blit_outline_harden_to_canvas(
        device,
        encoder,
        registry,
        pool.view(slot_a),
        content,
        canvas_view,
        outline.rgba,
    );
}
