use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::effects::EffectRegistry;
use crate::error::CompositorError;
use crate::gpu::GpuContext;
use crate::image::RgbaImage;
use crate::layer::{CompositeLayer, CompositorConfig, LayerContent, LayerEffect, LayerPlacement};
use crate::transitions::TransitionRegistry;
use crate::yuv::{Yuv420pImage, Yuv420pLayer};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const R8: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// Shared preamble prepended to every effect fragment shader.
const EFFECT_HEADER: &str = include_str!("../shaders/effect_header.wgsl");

/// Per-layer quad placement, precomputed to clip space. Mirrors the
/// `Placement` struct in the WGSL shaders.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PlacementUniforms {
    /// Columns of the 2×2 linear part mapping unit-quad corners
    /// ([-0.5, 0.5]², +y down) to clip space: (m00, m10, m01, m11).
    linear: [f32; 4],
    /// Clip-space translation (x, y), opacity, pad.
    trans_opacity: [f32; 4],
    /// Content UV rect (u0, v0, u1, v1) interpolated across the quad:
    /// sub-rects crop, reversed axes mirror. Unused by the solid shader.
    uv_rect: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SolidUniforms {
    color: [f32; 4],
    placement: PlacementUniforms,
}

/// Build the unit-quad → clip-space affine for a placement on this canvas.
fn placement_uniforms(
    config: &CompositorConfig,
    p: &LayerPlacement,
    uv: [f32; 4],
) -> PlacementUniforms {
    let (cw, ch) = (config.width as f32, config.height as f32);
    let (cos, sin) = (p.rotation.cos(), p.rotation.sin());
    // Canvas space (+y down): pos = center + R·(corner ⊙ size), with R the
    // clockwise rotation [cos, -sin; sin, cos] in screen coordinates.
    let a = cos * p.size[0]; // ∂pos.x/∂corner.x
    let b = sin * p.size[0]; // ∂pos.y/∂corner.x
    let c = -sin * p.size[1]; // ∂pos.x/∂corner.y
    let d = cos * p.size[1]; // ∂pos.y/∂corner.y
    // Canvas px → clip space: x' = 2x/cw − 1, y' = 1 − 2y/ch (flip y).
    PlacementUniforms {
        linear: [2.0 * a / cw, -2.0 * b / ch, 2.0 * c / cw, -2.0 * d / ch],
        trans_opacity: [
            2.0 * p.center[0] / cw - 1.0,
            1.0 - 2.0 * p.center[1] / ch,
            p.opacity,
            0.0,
        ],
        uv_rect: uv,
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RgbaToYuvParams {
    width: u32,
    height: u32,
    y_stride: u32,
    uv_stride: u32,
}

/// Per-pass uniform for effect fragment shaders. Mirrors `EffectParams` in
/// `effect_header.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EffectUniforms {
    /// (width, height, 1/width, 1/height).
    resolution: [f32; 4],
    /// Parameter slots 0..3.
    p0: [f32; 4],
    /// Parameter slots 4..7.
    p1: [f32; 4],
    /// (pass_index, pass_count, 0, 0).
    pass_info: [f32; 4],
}

/// WGPU alpha-over compositor. Layers are composited bottom-to-top with
/// standard src-over blending onto a single offscreen target.
pub struct Compositor {
    solid_pipeline: wgpu::RenderPipeline,
    blit_pipeline: wgpu::RenderPipeline,
    yuv_pipeline: wgpu::RenderPipeline,
    rgba_to_yuv_pipeline: wgpu::ComputePipeline,
    /// Blits a finished effect scratch onto the canvas (premultiplied blend).
    composite_pipeline: wgpu::RenderPipeline,
    solid_bind_layout: wgpu::BindGroupLayout,
    blit_bind_layout: wgpu::BindGroupLayout,
    yuv_bind_layout: wgpu::BindGroupLayout,
    rgba_to_yuv_bind_layout: wgpu::BindGroupLayout,
    effect_bind_layout: wgpu::BindGroupLayout,
    effects: EffectRegistry,
    transitions: TransitionRegistry,
    sampler: wgpu::Sampler,
    rgba_to_yuv_params: wgpu::Buffer,
    /// Reused each composite when canvas size matches.
    target: Option<CachedTarget>,
    /// Canvas-sized ping-pong targets for effect chains; built on first use
    /// and reused while the canvas size holds (no allocation when no layer
    /// carries effects).
    scratch: Option<ScratchTargets>,
    /// Per-layer upload textures reused across frames (see [`UploadPool`]).
    /// `RefCell` because layer encoding runs behind `&self`.
    upload_pool: RefCell<UploadPool>,
}

struct CachedTarget {
    width: u32,
    height: u32,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    readback: wgpu::Buffer,
    readback_stride: u32,
}

/// Three canvas-sized RGBA targets for effect chains: `orig` holds the placed
/// layer (kept available to combine-style passes), `a`/`b` ping-pong between
/// passes. All hold premultiplied alpha.
struct ScratchTargets {
    width: u32,
    height: u32,
    orig: wgpu::TextureView,
    a: wgpu::TextureView,
    b: wgpu::TextureView,
    _textures: [wgpu::Texture; 3],
}

/// Reusable per-layer upload textures, bucketed by `(format, width, height)`.
///
/// Compositing a media layer used to allocate a fresh GPU texture per plane
/// *every frame* — at 4K that's three ~R8 textures (~12MB) created and dropped
/// per preview frame, which dominated the per-frame cost on the playback hot
/// path. Instead we keep textures alive and hand them back out on later frames.
///
/// Safety of cross-frame reuse: [`Compositor::composite`] blocks on the readback
/// map (`device.poll(Wait)`) before returning, so the GPU is idle by the time a
/// texture is reacquired — no in-flight submit still references it. Within a
/// single frame, `cursor` hands out a *distinct* texture per `acquire` so two
/// same-sized layers (e.g. the equal-sized U and V planes) never collide.
#[derive(Default)]
struct UploadPool {
    buckets: HashMap<(wgpu::TextureFormat, u32, u32), Vec<wgpu::Texture>>,
    cursor: HashMap<(wgpu::TextureFormat, u32, u32), usize>,
}

impl UploadPool {
    /// Mark every pooled texture available again for a new frame.
    fn begin_frame(&mut self) {
        self.cursor.clear();
    }

    /// Hand out a texture of the requested shape, creating one only when the
    /// bucket can't satisfy this frame's demand.
    fn acquire(
        &mut self,
        gpu: &GpuContext,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> wgpu::Texture {
        let key = (format, width, height);
        let bucket = self.buckets.entry(key).or_default();
        let idx = self.cursor.entry(key).or_insert(0);
        if *idx >= bucket.len() {
            bucket.push(create_upload_texture(gpu, format, width, height));
        }
        let texture = bucket[*idx].clone();
        *idx += 1;
        texture
    }
}

impl Compositor {
    pub fn new(gpu: &GpuContext) -> Result<Self, CompositorError> {
        let blend = src_over_blend();

        let solid_shader = shader(gpu, "solid", include_str!("../shaders/solid.wgsl"));
        let blit_shader = shader(gpu, "blit", include_str!("../shaders/blit.wgsl"));
        let yuv_shader = shader(gpu, "yuv_blit", include_str!("../shaders/yuv_blit.wgsl"));
        let rgba_to_yuv_shader = shader(
            gpu,
            "rgba_to_yuv",
            include_str!("../shaders/rgba_to_yuv.wgsl"),
        );

        // Placement matrices live in the vertex stage; opacity in the fragment.
        let uniform_stages = wgpu::ShaderStages::VERTEX_FRAGMENT;

        let solid_bind_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("solid_bind"),
                    entries: &[uniform_binding(0, uniform_stages)],
                });

        let blit_bind_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("blit_bind"),
                    entries: &[
                        texture_binding(0, wgpu::ShaderStages::FRAGMENT),
                        sampler_binding(1, wgpu::ShaderStages::FRAGMENT),
                        uniform_binding(2, uniform_stages),
                    ],
                });

        let yuv_bind_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("yuv_bind"),
                    entries: &[
                        texture_binding(0, wgpu::ShaderStages::FRAGMENT),
                        texture_binding(1, wgpu::ShaderStages::FRAGMENT),
                        texture_binding(2, wgpu::ShaderStages::FRAGMENT),
                        sampler_binding(3, wgpu::ShaderStages::FRAGMENT),
                        uniform_binding(4, uniform_stages),
                    ],
                });

        let rgba_to_yuv_bind_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("rgba_to_yuv_bind"),
                    entries: &[
                        texture_binding(0, wgpu::ShaderStages::COMPUTE),
                        storage_binding(1, true),
                        storage_binding(2, true),
                        storage_binding(3, true),
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let solid_pipeline = render_pipeline(
            gpu,
            "solid_pipeline",
            &solid_shader,
            &pipeline_layout(gpu, "solid_layout", &solid_bind_layout),
            blend,
        );
        let blit_pipeline = render_pipeline(
            gpu,
            "blit_pipeline",
            &blit_shader,
            &pipeline_layout(gpu, "blit_layout", &blit_bind_layout),
            blend,
        );
        let yuv_pipeline = render_pipeline(
            gpu,
            "yuv_pipeline",
            &yuv_shader,
            &pipeline_layout(gpu, "yuv_layout", &yuv_bind_layout),
            blend,
        );

        // Effect passes: src texture + the untouched placed layer + sampler +
        // per-pass uniform. The composite step reuses the blit layout.
        let effect_bind_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("effect_bind"),
                    entries: &[
                        texture_binding(0, wgpu::ShaderStages::FRAGMENT),
                        texture_binding(1, wgpu::ShaderStages::FRAGMENT),
                        sampler_binding(2, wgpu::ShaderStages::FRAGMENT),
                        uniform_binding(3, wgpu::ShaderStages::FRAGMENT),
                    ],
                });

        let effects = EffectRegistry::build(
            &gpu.device,
            &pipeline_layout(gpu, "effect_layout", &effect_bind_layout),
            FORMAT,
            EFFECT_HEADER,
        );

        // Transitions share the effect bind layout and header (orig_tex =
        // outgoing frame, src_tex = incoming, progress in fx.p0.x).
        let transitions = TransitionRegistry::build(
            &gpu.device,
            &pipeline_layout(gpu, "transition_layout", &effect_bind_layout),
            FORMAT,
            EFFECT_HEADER,
        );

        let composite_shader = shader(gpu, "composite", include_str!("../shaders/composite.wgsl"));
        let composite_pipeline = render_pipeline(
            gpu,
            "composite_pipeline",
            &composite_shader,
            &pipeline_layout(gpu, "composite_layout", &blit_bind_layout),
            premultiplied_blend(),
        );

        let rgba_to_yuv_pipeline =
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("rgba_to_yuv_pipeline"),
                    layout: Some(&gpu.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: Some("rgba_to_yuv_layout"),
                            bind_group_layouts: &[&rgba_to_yuv_bind_layout],
                            push_constant_ranges: &[],
                        },
                    )),
                    module: &rgba_to_yuv_shader,
                    entry_point: Some("cs"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("layer_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let rgba_to_yuv_params = uniform_buffer(
            gpu,
            "rgba_to_yuv_params",
            &RgbaToYuvParams {
                width: 0,
                height: 0,
                y_stride: 0,
                uv_stride: 0,
            },
        );

        Ok(Self {
            solid_pipeline,
            blit_pipeline,
            yuv_pipeline,
            rgba_to_yuv_pipeline,
            composite_pipeline,
            solid_bind_layout,
            blit_bind_layout,
            yuv_bind_layout,
            rgba_to_yuv_bind_layout,
            effect_bind_layout,
            effects,
            transitions,
            sampler,
            rgba_to_yuv_params,
            target: None,
            scratch: None,
            upload_pool: RefCell::new(UploadPool::default()),
        })
    }

    /// Composite layers bottom-to-top and read back RGBA8 bytes.
    ///
    /// Render and texture→buffer copy share one command encoder and one submit;
    /// the CPU then blocks on the readback map.
    pub fn composite(
        &mut self,
        gpu: &GpuContext,
        config: &CompositorConfig,
        layers: &[CompositeLayer],
    ) -> Result<RgbaImage, CompositorError> {
        validate_layers(config, layers)?;
        self.ensure_target(gpu, config)?;
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("composite_encoder"),
            });
        self.render_layers_into(&mut encoder, gpu, config, layers)?;
        self.encode_readback_copy(&mut encoder, config);
        gpu.queue.submit(Some(encoder.finish()));
        self.map_readback(gpu, config)
    }

    /// Composite layers, convert the canvas to YUV420P on GPU, and read back.
    pub fn composite_yuv420p(
        &mut self,
        gpu: &GpuContext,
        config: &CompositorConfig,
        layers: &[CompositeLayer],
    ) -> Result<Yuv420pImage, CompositorError> {
        validate_layers(config, layers)?;
        self.ensure_target(gpu, config)?;

        let width = config.width;
        let height = config.height;
        let y_stride = width;
        let uv_stride = width / 2;
        let y_count = (y_stride * height) as u64;
        let uv_count = (uv_stride * (height / 2)) as u64;
        let y_buf = storage_buffer(gpu, "y_out", y_count * 4);
        let u_buf = storage_buffer(gpu, "u_out", uv_count * 4);
        let v_buf = storage_buffer(gpu, "v_out", uv_count * 4);

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("composite_yuv_encoder"),
            });
        self.render_layers_into(&mut encoder, gpu, config, layers)?;
        self.encode_yuv_into(&mut encoder, gpu, config, &y_buf, &u_buf, &v_buf)?;
        gpu.queue.submit(Some(encoder.finish()));

        let y = read_storage_u8(gpu, &y_buf, y_count as usize)?;
        let u = read_storage_u8(gpu, &u_buf, uv_count as usize)?;
        let v = read_storage_u8(gpu, &v_buf, uv_count as usize)?;
        Ok(Yuv420pImage {
            width,
            height,
            y,
            u,
            v,
        })
    }

    fn render_layers_into(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        gpu: &GpuContext,
        config: &CompositorConfig,
        layers: &[CompositeLayer],
    ) -> Result<(), CompositorError> {
        // Reclaim last frame's upload textures for reuse on this one.
        self.upload_pool.borrow_mut().begin_frame();

        // Effect and transition layers need offscreen ping-pong targets;
        // allocate them only when something actually needs them.
        if layers
            .iter()
            .any(|l| !l.effects.is_empty() || matches!(l.content, LayerContent::Transition { .. }))
        {
            self.ensure_scratch(gpu, config)?;
        }

        // Consecutive plain layers share one render pass onto the target (the
        // pre-M4 single-pass behavior). An effect layer renders offscreen and
        // composites in its own pass, breaking the run. `first` tracks whether
        // the target still needs its background clear.
        let mut first = true;
        let mut i = 0;
        while i < layers.len() {
            let layer = &layers[i];
            if matches!(layer.content, LayerContent::Adjustment) {
                // An adjustment filters everything below it. Ensure there is a
                // cleared canvas to work on even when it is the first layer.
                if first {
                    self.clear_target(encoder, config);
                    first = false;
                }
                if !layer.effects.is_empty() {
                    self.apply_adjustment(encoder, gpu, config, layer);
                }
                i += 1;
            } else if matches!(layer.content, LayerContent::Transition { .. }) {
                self.draw_transition_layer(encoder, gpu, config, layer, first);
                first = false;
                i += 1;
            } else if layer.effects.is_empty() {
                let start = i;
                while i < layers.len()
                    && layers[i].effects.is_empty()
                    && !matches!(
                        layers[i].content,
                        LayerContent::Adjustment | LayerContent::Transition { .. }
                    )
                {
                    i += 1;
                }
                self.draw_plain_run(encoder, gpu, config, &layers[start..i], first);
                first = false;
            } else {
                self.draw_effect_layer(encoder, gpu, config, &layers[i], first);
                first = false;
                i += 1;
            }
        }

        // No layers at all: still clear the canvas to the background once
        // (empty timeline / fully-disabled tracks show the background color).
        if first {
            self.clear_target(encoder, config);
        }

        Ok(())
    }

    /// Background clear color for the canvas target.
    fn clear_color(config: &CompositorConfig) -> wgpu::Color {
        // The target is Rgba8Unorm (no sRGB encode), so byte/255 lands the
        // exact color, same as the solid-layer shader.
        wgpu::Color {
            r: f64::from(config.background[0]) / 255.0,
            g: f64::from(config.background[1]) / 255.0,
            b: f64::from(config.background[2]) / 255.0,
            a: 1.0,
        }
    }

    /// Open a pass that only clears the target to the background (used when
    /// there are no layers to draw).
    fn clear_target(&self, encoder: &mut wgpu::CommandEncoder, config: &CompositorConfig) {
        let target = self.target.as_ref().expect("target initialized");
        let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("composite_clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(Self::clear_color(config)),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
    }

    /// Draw a run of plain (no-effect) layers in one render pass onto the
    /// target. Clears to the background when `first`, otherwise loads.
    fn draw_plain_run(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gpu: &GpuContext,
        config: &CompositorConfig,
        layers: &[CompositeLayer],
        first: bool,
    ) {
        let target = self.target.as_ref().expect("target initialized");
        let load = if first {
            wgpu::LoadOp::Clear(Self::clear_color(config))
        } else {
            wgpu::LoadOp::Load
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("composite_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        for layer in layers {
            self.encode_layer(&mut pass, gpu, config, layer, layer.placement.opacity);
        }
    }

    /// Record one layer's draw into an already-open render pass, with an
    /// explicit opacity (plain layers pass their own; the effect path renders
    /// the placed layer at full opacity into scratch and applies opacity at
    /// the final composite).
    fn encode_layer(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        gpu: &GpuContext,
        config: &CompositorConfig,
        layer: &CompositeLayer,
        opacity: f32,
    ) {
        // Per-layer uniform buffer: each draw needs its own values alive at
        // submit time (a single reused buffer would be clobbered by later
        // `write_buffer` calls in the same pass).
        let mut placement_in = layer.placement;
        placement_in.opacity = opacity;
        let placement = placement_uniforms(config, &placement_in, layer.uv);
        match &layer.content {
            LayerContent::Solid { rgba } => {
                let uniforms = SolidUniforms {
                    color: [
                        rgba[0] as f32 / 255.0,
                        rgba[1] as f32 / 255.0,
                        rgba[2] as f32 / 255.0,
                        rgba[3] as f32 / 255.0,
                    ],
                    placement,
                };
                let buffer = uniform_buffer(gpu, "solid_uniform", &uniforms);
                let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("solid_bind"),
                    layout: &self.solid_bind_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    }],
                });
                pass.set_pipeline(&self.solid_pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.draw(0..6, 0..1);
            }
            LayerContent::Rgba {
                bytes,
                width,
                height,
            } => {
                let texture = self
                    .upload_pool
                    .borrow_mut()
                    .acquire(gpu, FORMAT, *width, *height);
                write_rgba_texture(gpu, &texture, bytes, *width, *height);
                let view = texture.create_view(&Default::default());
                let buffer = uniform_buffer(gpu, "blit_placement", &placement);
                let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("blit_bind"),
                    layout: &self.blit_bind_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: buffer.as_entire_binding(),
                        },
                    ],
                });
                pass.set_pipeline(&self.blit_pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.draw(0..6, 0..1);
            }
            LayerContent::Yuv420p(yuv) => {
                // Signal full-range YUV to the shader through the placement
                // pad slot (unused by the limited-range default path).
                let mut placement = placement;
                placement.trans_opacity[3] = if yuv.full_range { 1.0 } else { 0.0 };
                let uv_w = yuv.width / 2;
                let uv_h = yuv.height / 2;
                // Acquire all three planes first: the U and V planes share a
                // shape, so the pool must hand back distinct textures for them.
                let (y_tex, u_tex, v_tex) = {
                    let mut pool = self.upload_pool.borrow_mut();
                    (
                        pool.acquire(gpu, R8, yuv.width, yuv.height),
                        pool.acquire(gpu, R8, uv_w, uv_h),
                        pool.acquire(gpu, R8, uv_w, uv_h),
                    )
                };
                write_r8_texture(gpu, &y_tex, &yuv.tight_y(), yuv.width, yuv.height);
                write_r8_texture(gpu, &u_tex, &yuv.tight_u(), uv_w, uv_h);
                write_r8_texture(gpu, &v_tex, &yuv.tight_v(), uv_w, uv_h);
                let buffer = uniform_buffer(gpu, "yuv_placement", &placement);
                let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("yuv_bind"),
                    layout: &self.yuv_bind_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &y_tex.create_view(&Default::default()),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &u_tex.create_view(&Default::default()),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(
                                &v_tex.create_view(&Default::default()),
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: buffer.as_entire_binding(),
                        },
                    ],
                });
                pass.set_pipeline(&self.yuv_pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.draw(0..6, 0..1);
            }
            // Adjustment and transition layers carry no directly-encodable
            // pixels of their own; they are handled out of band in
            // render_layers_into and never reach a plain run.
            LayerContent::Adjustment | LayerContent::Transition { .. } => {}
        }
    }

    /// Render a layer that carries an effect chain: draw it into the `orig`
    /// scratch (transparent clear, full opacity → premultiplied content), run
    /// each effect pass ping-ponging `a`/`b`, then composite the result onto
    /// the target with the layer's opacity (premultiplied blend).
    fn draw_effect_layer(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gpu: &GpuContext,
        config: &CompositorConfig,
        layer: &CompositeLayer,
        first: bool,
    ) {
        let scratch = self.scratch.as_ref().expect("scratch initialized");

        // 1. Placed layer → scratch.orig, over a transparent clear.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("effect_orig"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &scratch.orig,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            self.encode_layer(&mut pass, gpu, config, layer, 1.0);
        }

        // 2. Run the effect chain over the placed scratch.
        let input = self.run_effect_chain(encoder, gpu, config, &layer.effects, scratch);

        // 3. Composite the finished scratch onto the target with the layer
        // opacity (premultiplied src-over).
        self.composite_view_to_target(encoder, gpu, config, input, layer.placement.opacity, first);
    }

    /// Apply an adjustment layer: filter the accumulated canvas through the
    /// layer's effect chain and write it back (CapCut adjustment-layer
    /// semantics). The canvas below is copied into `scratch.orig`, run through
    /// the chain, then composited back over itself — at opacity 1 the adjusted
    /// result fully replaces the canvas; below 1 it blends for a partial
    /// strength. Called only with a non-empty chain and after the target has
    /// been cleared, so there is always content below to adjust.
    fn apply_adjustment(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gpu: &GpuContext,
        config: &CompositorConfig,
        layer: &CompositeLayer,
    ) {
        let scratch = self.scratch.as_ref().expect("scratch initialized");

        // 1. Snapshot the accumulated canvas into scratch.orig (premultiplied
        // copy over a transparent clear).
        let target = self.target.as_ref().expect("target initialized");
        let full = LayerPlacement::full_canvas(config);
        let placement = placement_uniforms(config, &full, crate::layer::FULL_UV);
        let buffer = uniform_buffer(gpu, "adjustment_copy", &placement);
        let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("adjustment_copy_bind"),
            layout: &self.blit_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&target.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("adjustment_copy_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &scratch.orig,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..6, 0..1);
        }

        // 2. Run the chain, then 3. composite the result back over the canvas.
        let input = self.run_effect_chain(encoder, gpu, config, &layer.effects, scratch);
        self.composite_view_to_target(encoder, gpu, config, input, layer.placement.opacity, false);
    }

    /// Draw a transition layer: render the outgoing sub-layer into `scratch.orig`
    /// and the incoming one into `scratch.a`, blend them by progress through the
    /// transition shader into `scratch.b`, then composite the blend over the
    /// canvas. An unknown transition id falls back to a hard cut to the
    /// incoming clip.
    fn draw_transition_layer(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gpu: &GpuContext,
        config: &CompositorConfig,
        layer: &CompositeLayer,
        first: bool,
    ) {
        let LayerContent::Transition {
            from,
            to,
            transition_id,
            progress,
        } = &layer.content
        else {
            return;
        };
        let scratch = self.scratch.as_ref().expect("scratch initialized");

        // 1. Outgoing → orig, incoming → a (each over a transparent clear, so
        // the scratch holds premultiplied content).
        for (target_view, sub) in [(&scratch.orig, from.as_ref()), (&scratch.a, to.as_ref())] {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("transition_side"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            self.encode_layer(&mut pass, gpu, config, sub, 1.0);
        }

        let Some(pipeline_idx) = self.transitions.pipeline(transition_id) else {
            self.composite_view_to_target(encoder, gpu, config, &scratch.a, 1.0, first);
            return;
        };

        // 2. Blend orig (from) and a (to) by progress into b.
        let resolution = [
            config.width as f32,
            config.height as f32,
            1.0 / config.width as f32,
            1.0 / config.height as f32,
        ];
        let uniforms = EffectUniforms {
            resolution,
            p0: [*progress, 0.0, 0.0, 0.0],
            p1: [0.0; 4],
            pass_info: [0.0, 1.0, 0.0, 0.0],
        };
        let buffer = uniform_buffer(gpu, "transition_uniform", &uniforms);
        let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transition_bind"),
            layout: &self.effect_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scratch.a),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&scratch.orig),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffer.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("transition_blend"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &scratch.b,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.transitions.pipelines[pipeline_idx]);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }

        // 3. Composite the blend over the canvas below.
        self.composite_view_to_target(encoder, gpu, config, &scratch.b, 1.0, first);
    }

    /// Run an effect chain starting from `scratch.orig`, ping-ponging through
    /// `scratch.a`/`scratch.b`; returns the view holding the final result
    /// (`scratch.orig` if the chain contributed no pass). `scratch.orig` stays
    /// bound as `orig_tex` for combine-style passes (e.g. glow).
    fn run_effect_chain<'s>(
        &'s self,
        encoder: &mut wgpu::CommandEncoder,
        gpu: &GpuContext,
        config: &CompositorConfig,
        effects: &[LayerEffect],
        scratch: &'s ScratchTargets,
    ) -> &'s wgpu::TextureView {
        let resolution = [
            config.width as f32,
            config.height as f32,
            1.0 / config.width as f32,
            1.0 / config.height as f32,
        ];
        let mut input = &scratch.orig;
        let mut use_a = true;
        for fx in effects {
            let Some(passes) = self.effects.passes(&fx.effect_id) else {
                continue;
            };
            let pass_count = passes.len();
            for (pass_index, &pipeline_idx) in passes.iter().enumerate() {
                let output = if use_a { &scratch.a } else { &scratch.b };
                let uniforms = EffectUniforms {
                    resolution,
                    p0: [fx.params[0], fx.params[1], fx.params[2], fx.params[3]],
                    p1: [fx.params[4], fx.params[5], fx.params[6], fx.params[7]],
                    pass_info: [pass_index as f32, pass_count as f32, 0.0, 0.0],
                };
                let buffer = uniform_buffer(gpu, "effect_uniform", &uniforms);
                let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("effect_bind"),
                    layout: &self.effect_bind_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(input),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&scratch.orig),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: buffer.as_entire_binding(),
                        },
                    ],
                });
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("effect_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: output,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.effects.pipelines[pipeline_idx]);
                pass.set_bind_group(0, &bind, &[]);
                pass.draw(0..3, 0..1);
                drop(pass);
                input = output;
                use_a = !use_a;
            }
        }
        input
    }

    /// Composite a full-canvas scratch view onto the target with `opacity`
    /// (premultiplied src-over). Clears to the background when `first`.
    fn composite_view_to_target(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gpu: &GpuContext,
        config: &CompositorConfig,
        input: &wgpu::TextureView,
        opacity: f32,
        first: bool,
    ) {
        let target = self.target.as_ref().expect("target initialized");
        let mut full = LayerPlacement::full_canvas(config);
        full.opacity = opacity;
        let placement = placement_uniforms(config, &full, crate::layer::FULL_UV);
        let buffer = uniform_buffer(gpu, "composite_placement", &placement);
        let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite_bind"),
            layout: &self.blit_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer.as_entire_binding(),
                },
            ],
        });
        let load = if first {
            wgpu::LoadOp::Clear(Self::clear_color(config))
        } else {
            wgpu::LoadOp::Load
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("effect_composite"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.composite_pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.draw(0..6, 0..1);
    }

    fn ensure_scratch(
        &mut self,
        gpu: &GpuContext,
        config: &CompositorConfig,
    ) -> Result<(), CompositorError> {
        let needs_new = self
            .scratch
            .as_ref()
            .is_none_or(|s| s.width != config.width || s.height != config.height);
        if needs_new {
            let make = |label: &str| {
                gpu.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: config.width,
                        height: config.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
            };
            let orig_t = make("scratch_orig");
            let a_t = make("scratch_a");
            let b_t = make("scratch_b");
            let orig = orig_t.create_view(&Default::default());
            let a = a_t.create_view(&Default::default());
            let b = b_t.create_view(&Default::default());
            self.scratch = Some(ScratchTargets {
                width: config.width,
                height: config.height,
                orig,
                a,
                b,
                _textures: [orig_t, a_t, b_t],
            });
        }
        Ok(())
    }

    fn encode_yuv_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gpu: &GpuContext,
        config: &CompositorConfig,
        y_buf: &wgpu::Buffer,
        u_buf: &wgpu::Buffer,
        v_buf: &wgpu::Buffer,
    ) -> Result<(), CompositorError> {
        let target = self.target.as_ref().expect("target initialized");
        let width = config.width;
        let height = config.height;
        let params = RgbaToYuvParams {
            width,
            height,
            y_stride: width,
            uv_stride: width / 2,
        };
        gpu.queue
            .write_buffer(&self.rgba_to_yuv_params, 0, bytemuck::bytes_of(&params));

        let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rgba_to_yuv_bind"),
            layout: &self.rgba_to_yuv_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&target.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: y_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: v_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.rgba_to_yuv_params.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rgba_to_yuv_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.rgba_to_yuv_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        Ok(())
    }

    /// Record the canvas → readback-buffer copy into the caller's encoder so
    /// it rides the same submit as the render passes (no second submit).
    fn encode_readback_copy(&self, encoder: &mut wgpu::CommandEncoder, config: &CompositorConfig) {
        let target = self.target.as_ref().expect("target initialized");
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &target.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(target.readback_stride),
                    rows_per_image: Some(config.height),
                },
            },
            wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Map the already-copied readback buffer (blocks on the GPU) and de-pad
    /// the aligned rows into a tight RGBA image.
    fn map_readback(
        &mut self,
        gpu: &GpuContext,
        config: &CompositorConfig,
    ) -> Result<RgbaImage, CompositorError> {
        let target = self.target.as_ref().expect("target initialized");
        let row_bytes = usize::try_from(config.width * 4).expect("width");
        let padded_row = usize::try_from(target.readback_stride).expect("stride");
        let height = usize::try_from(config.height).expect("height");
        let mut tight = vec![0u8; row_bytes * height];
        map_read_buffer(gpu, &target.readback, |mapped| {
            for row in 0..height {
                let src = row * padded_row;
                let dst = row * row_bytes;
                tight[dst..dst + row_bytes].copy_from_slice(&mapped[src..src + row_bytes]);
            }
        })?;
        RgbaImage::new(config.width, config.height, tight)
    }

    fn ensure_target(
        &mut self,
        gpu: &GpuContext,
        config: &CompositorConfig,
    ) -> Result<(), CompositorError> {
        let needs_new = self
            .target
            .as_ref()
            .is_none_or(|t| t.width != config.width || t.height != config.height);
        if needs_new {
            let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("composite_target"),
                size: wgpu::Extent3d {
                    width: config.width,
                    height: config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            let readback_stride = align_row_bytes(config.width * 4);
            let readback_size = u64::from(readback_stride) * u64::from(config.height);
            let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: readback_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            self.target = Some(CachedTarget {
                width: config.width,
                height: config.height,
                texture,
                view,
                readback,
                readback_stride,
            });
        }
        Ok(())
    }
}

fn validate_layers(
    config: &CompositorConfig,
    layers: &[CompositeLayer],
) -> Result<(), CompositorError> {
    validate_config(config)?;
    for layer in layers {
        match &layer.content {
            LayerContent::Rgba {
                bytes,
                width,
                height,
            } => {
                let expected = (*width as usize)
                    .saturating_mul(*height as usize)
                    .saturating_mul(4);
                if *width == 0 || *height == 0 || bytes.len() != expected {
                    return Err(CompositorError::LayerSizeMismatch {
                        got: bytes.len(),
                        expected,
                    });
                }
            }
            LayerContent::Yuv420p(yuv) => validate_yuv_layer(yuv)?,
            LayerContent::Transition { from, to, .. } => {
                validate_layers(config, std::slice::from_ref(from.as_ref()))?;
                validate_layers(config, std::slice::from_ref(to.as_ref()))?;
            }
            LayerContent::Solid { .. } | LayerContent::Adjustment => {}
        }
    }
    Ok(())
}

fn validate_yuv_layer(layer: &Yuv420pLayer) -> Result<(), CompositorError> {
    if layer.width == 0
        || layer.height == 0
        || !layer.width.is_multiple_of(2)
        || !layer.height.is_multiple_of(2)
    {
        return Err(CompositorError::InvalidYuvDimensions {
            width: layer.width,
            height: layer.height,
        });
    }
    Ok(())
}

fn validate_config(config: &CompositorConfig) -> Result<(), CompositorError> {
    if config.width == 0 || config.height == 0 {
        return Err(CompositorError::InvalidDimensions {
            width: config.width,
            height: config.height,
        });
    }
    Ok(())
}

fn align_row_bytes(bytes_per_row: u32) -> u32 {
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    bytes_per_row.div_ceil(align) * align
}

fn src_over_blend() -> wgpu::BlendState {
    wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::SrcAlpha,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
    }
}

/// Src-over for sources that are already premultiplied by alpha (the effect
/// scratch). The exact straight-alpha over-operator without re-multiplying.
fn premultiplied_blend() -> wgpu::BlendState {
    wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
    }
}

fn shader(gpu: &GpuContext, label: &str, source: &str) -> wgpu::ShaderModule {
    gpu.device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        })
}

fn pipeline_layout(
    gpu: &GpuContext,
    label: &str,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::PipelineLayout {
    gpu.device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[layout],
            push_constant_ranges: &[],
        })
}

fn render_pipeline(
    gpu: &GpuContext,
    label: &str,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    blend: wgpu::BlendState,
) -> wgpu::RenderPipeline {
    gpu.device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: FORMAT,
                    blend: Some(blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
}

fn texture_binding(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_binding(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn uniform_binding(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_binding(binding: u32, read_write: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: if read_write {
                wgpu::BufferBindingType::Storage { read_only: false }
            } else {
                wgpu::BufferBindingType::Storage { read_only: true }
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_buffer<T: Pod>(gpu: &GpuContext, label: &str, value: &T) -> wgpu::Buffer {
    gpu.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(value),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
}

fn storage_buffer(gpu: &GpuContext, label: &str, size: u64) -> wgpu::Buffer {
    gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Create a bare upload texture (no contents). Pooled by [`UploadPool`] and
/// filled per frame by [`write_rgba_texture`] / [`write_r8_texture`].
fn create_upload_texture(
    gpu: &GpuContext,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("layer_upload"),
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
    })
}

fn write_rgba_texture(
    gpu: &GpuContext,
    texture: &wgpu::Texture,
    bytes: &[u8],
    width: u32,
    height: u32,
) {
    write_texture_rows(gpu, texture, bytes, width * 4, width, height);
}

fn write_r8_texture(
    gpu: &GpuContext,
    texture: &wgpu::Texture,
    bytes: &[u8],
    width: u32,
    height: u32,
) {
    write_texture_rows(gpu, texture, bytes, width, width, height);
}

fn write_texture_rows(
    gpu: &GpuContext,
    texture: &wgpu::Texture,
    bytes: &[u8],
    bytes_per_row: u32,
    width: u32,
    height: u32,
) {
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

fn map_read_buffer<F>(gpu: &GpuContext, buffer: &wgpu::Buffer, f: F) -> Result<(), CompositorError>
where
    F: FnOnce(&[u8]),
{
    let slice = buffer.slice(..);
    let (tx, rx) = mpsc::sync_channel(1);
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = gpu.device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|_| CompositorError::MapFailed)?
        .map_err(|_| CompositorError::MapFailed)?;
    let mapped = slice.get_mapped_range();
    f(&mapped);
    drop(mapped);
    buffer.unmap();
    Ok(())
}

fn read_storage_u8(
    gpu: &GpuContext,
    buffer: &wgpu::Buffer,
    count: usize,
) -> Result<Vec<u8>, CompositorError> {
    let byte_len = count * 4;
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("storage_readback"),
        size: byte_len as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("storage_copy"),
        });
    encoder.copy_buffer_to_buffer(buffer, 0, &readback, 0, byte_len as u64);
    gpu.queue.submit(Some(encoder.finish()));

    let mut out = vec![0u8; count];
    map_read_buffer(gpu, &readback, |mapped| {
        for (i, chunk) in mapped.chunks_exact(4).take(count).enumerate() {
            out[i] = chunk[0];
        }
    })?;
    Ok(out)
}
