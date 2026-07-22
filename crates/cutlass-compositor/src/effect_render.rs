//! Offscreen targets, pass pipelines, and effect/transition execution.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::error::CompositorError;
use crate::grade::ColorGrade;
use crate::passes::{PassInstance, effect_is_noop};

mod blend;
mod blit;
mod lut;
mod pipelines;
mod pool;
mod styles;

pub(crate) use blend::run_blend_composite;
pub(crate) use blit::{blit_premultiplied_to_canvas, blit_replace};
pub(crate) use lut::run_lut_pass;
pub(crate) use pipelines::PassRegistry;
pub(crate) use pool::OffscreenPool;
pub(crate) use styles::{background_plate_layer, composite_layer_outline, composite_layer_styles};

const RT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TexelUniforms {
    texel_size: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EffectUniforms {
    texel_size: [f32; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TransitionUniforms {
    texel_size: [f32; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GradeUniforms {
    grade0: [f32; 4],
    grade1: [f32; 4],
}

/// Run an effect chain on `input` view, ping-ponging through `pool`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_effect_chain<'a>(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    registry: &PassRegistry,
    pool: &'a OffscreenPool,
    input_view: &'a wgpu::TextureView,
    effects: &[PassInstance<'_>],
    width: u32,
    height: u32,
) -> &'a wgpu::TextureView {
    let mut read_idx = 0usize;
    let mut current_input = input_view;

    for effect in effects {
        if effect_is_noop(effect.id, effect.params) {
            continue;
        }
        let write_idx = read_idx ^ 1;
        let write_view = pool.view(write_idx);

        match effect.id {
            "gaussian_blur" => {
                let radius = effect.params.first().copied().unwrap_or(0.0);
                let mid_idx = 1usize;
                let out_idx = 2usize;
                let mid_view = pool.view(mid_idx);
                let out_view = pool.view(out_idx);
                draw_effect_pass(
                    device,
                    encoder,
                    registry,
                    &registry.blur_h,
                    current_input,
                    mid_view,
                    width,
                    height,
                    radius,
                );
                draw_effect_pass(
                    device,
                    encoder,
                    registry,
                    &registry.blur_v,
                    mid_view,
                    out_view,
                    width,
                    height,
                    radius,
                );
                read_idx = out_idx;
                current_input = out_view;
                continue;
            }
            _ => {
                let pipeline = registry.effect_pipeline(effect.id);
                let param0 = effect.params.first().copied().unwrap_or(0.0);
                draw_effect_pass(
                    device,
                    encoder,
                    registry,
                    pipeline,
                    current_input,
                    write_view,
                    width,
                    height,
                    param0,
                );
            }
        }
        read_idx = write_idx;
        current_input = pool.view(read_idx);
    }

    current_input
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_transition_pass(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    registry: &PassRegistry,
    outgoing: &wgpu::TextureView,
    incoming: &wgpu::TextureView,
    output: &wgpu::TextureView,
    transition_id: &str,
    progress: f32,
    width: u32,
    height: u32,
) {
    let pipeline = registry.transition_pipeline(transition_id);
    let uniforms = TransitionUniforms {
        texel_size: texel_size(width, height),
        params: [progress, 0.0, 0.0, 0.0],
    };
    let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cutlass.transition.uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cutlass.transition.bg"),
        layout: &registry.transition_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(outgoing),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(incoming),
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
        label: Some("cutlass.transition.pass"),
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
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.draw(0..6, 0..1);
}

pub(crate) fn run_grade_pass<'a>(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    registry: &PassRegistry,
    input: &'a wgpu::TextureView,
    output: &'a wgpu::TextureView,
    grade: ColorGrade,
) -> &'a wgpu::TextureView {
    let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cutlass.canvas_grade.uniforms"),
        contents: bytemuck::bytes_of(&GradeUniforms {
            grade0: [
                grade.exposure,
                grade.brightness,
                grade.contrast,
                grade.saturation,
            ],
            grade1: [grade.temperature, grade.tint, 0.0, 0.0],
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cutlass.canvas_grade.bg"),
        layout: &registry.effect_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(input),
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
        label: Some("cutlass.canvas_grade.pass"),
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
    pass.set_pipeline(&registry.grade);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.draw(0..6, 0..1);
    output
}

#[allow(clippy::too_many_arguments)]
pub(super) fn draw_effect_pass(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    registry: &PassRegistry,
    pipeline: &wgpu::RenderPipeline,
    input: &wgpu::TextureView,
    output: &wgpu::TextureView,
    width: u32,
    height: u32,
    param0: f32,
) {
    let uniforms = EffectUniforms {
        texel_size: texel_size(width, height),
        params: [param0, 0.0, 0.0, 0.0],
    };
    let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cutlass.effect.uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cutlass.effect.bg"),
        layout: &registry.effect_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(input),
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
        label: Some("cutlass.effect.pass"),
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
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.draw(0..6, 0..1);
}

fn texel_size(width: u32, height: u32) -> [f32; 4] {
    [
        1.0 / width.max(1) as f32,
        1.0 / height.max(1) as f32,
        0.0,
        0.0,
    ]
}

/// Draw one layer to an offscreen (or canvas) target.
///
/// `instances` is `Some((buffer, count))` for instanced glyph draws; `None`
/// draws a single unit-quad like the rest of the layer pipelines.
pub(crate) fn draw_layer_to_offscreen(
    pass: &mut wgpu::RenderPass<'_>,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    instances: Option<(&wgpu::Buffer, u32)>,
) {
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    if let Some((buffer, count)) = instances {
        pass.set_vertex_buffer(0, buffer.slice(..));
        pass.draw(0..6, 0..count);
    } else {
        pass.draw(0..6, 0..1);
    }
}

/// Returns `true` when the effect chain is empty or all passes are no-ops.
pub(crate) fn effects_need_offscreen(effects: &[PassInstance<'_>]) -> bool {
    !effects.is_empty() && effects.iter().any(|e| !effect_is_noop(e.id, e.params))
}

#[allow(dead_code)]
pub(crate) fn check_dimensions(width: u32, height: u32) -> Result<(), CompositorError> {
    if width == 0 || height == 0 {
        return Err(CompositorError::InvalidDimensions { width, height });
    }
    Ok(())
}
