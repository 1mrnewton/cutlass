use crate::effect_render::{
    OffscreenPool, blit_premultiplied_to_canvas, blit_replace, draw_layer_to_offscreen,
    effects_need_offscreen, run_effect_chain, run_grade_pass, run_lut_pass, run_transition_pass,
};
use crate::error::CompositorError;
use crate::gpu::GpuContext;
use crate::layer::{CompositeLayer, CompositorConfig, CompositorLayer, LayerLut};
use crate::lut::CubeLut;

use super::{
    CANVAS_FORMAT, Compositor, FrameSink, ImageSink, LUT_CACHE_MAX, LayerPipeline, LutTexture,
    TargetCache,
};
use cutlass_core::RgbaImage;

impl Compositor {
    /// Make `self.target` hold a canvas + readback buffer for `width`×`height`,
    /// reusing the cached pair when the size already matches.
    fn ensure_target(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self
            .target
            .as_ref()
            .is_some_and(|t| t.width == width && t.height == height)
        {
            return;
        }

        let unpadded = width * 4;
        let padded = unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cutlass.canvas"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: CANVAS_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cutlass.readback"),
            size: u64::from(padded) * u64::from(height),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.target = Some(TargetCache {
            width,
            height,
            texture,
            view,
            readback,
            padded_bytes_per_row: padded,
        });
    }

    /// Upload `lut` as a 3D texture under `key`, reusing the cached copy when
    /// it's already there. Called for every LUT in the scene before the pass
    /// encodes, so lookups inside the render loop are immutable.
    fn ensure_lut(&mut self, gpu: &GpuContext, key: &str, lut: &CubeLut) {
        if self.lut_cache.contains_key(key) {
            return;
        }
        if self.lut_cache.len() >= LUT_CACHE_MAX {
            self.lut_cache.clear();
        }
        let n = lut.size();
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cutlass.lut3d"),
            size: wgpu::Extent3d {
                width: n,
                height: n,
                depth_or_array_layers: n,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8Unorm,
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
            &lut.rgba8_texels(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(n * 4),
                rows_per_image: Some(n),
            },
            wgpu::Extent3d {
                width: n,
                height: n,
                depth_or_array_layers: n,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.lut_cache.insert(
            key.to_owned(),
            LutTexture {
                _texture: texture,
                view,
                size: n,
                domain_min: lut.domain_min(),
                domain_max: lut.domain_max(),
            },
        );
    }

    fn ensure_offscreen(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self
            .offscreen
            .as_ref()
            .is_some_and(|o| o.width == width && o.height == height)
        {
            return;
        }
        self.offscreen = Some(OffscreenPool::ensure(device, width, height));
    }

    /// Composite `layers` (bottom-to-top) onto a cleared canvas and read it
    /// back. The canvas texture and readback buffer are cached on `self` and
    /// reused while the render size stays the same.
    pub fn render(
        &mut self,
        gpu: &GpuContext,
        config: &CompositorConfig,
        layers: &[CompositeLayer<'_>],
    ) -> Result<RgbaImage, CompositorError> {
        let mut sink = ImageSink::default();
        self.render_into(gpu, config, layers, &mut sink)?;
        Ok(sink
            .into_image()
            .expect("render_into fills the sink on success"))
    }

    /// [`render`](Self::render) writing the readback rows directly into sink-provided storage.
    pub fn render_into(
        &mut self,
        gpu: &GpuContext,
        config: &CompositorConfig,
        layers: &[CompositeLayer<'_>],
        sink: &mut dyn FrameSink,
    ) -> Result<(), CompositorError> {
        let items: Vec<CompositorLayer<'_>> = layers.iter().map(CompositorLayer::layer).collect();
        self.render_compositor_layers_into(gpu, config, &items, sink)
    }

    /// Composite with effect chains, canvas passes, and transitions.
    pub fn render_compositor_layers(
        &mut self,
        gpu: &GpuContext,
        config: &CompositorConfig,
        layers: &[CompositorLayer<'_>],
    ) -> Result<RgbaImage, CompositorError> {
        let mut sink = ImageSink::default();
        self.render_compositor_layers_into(gpu, config, layers, &mut sink)?;
        Ok(sink
            .into_image()
            .expect("render_compositor_layers_into fills the sink on success"))
    }

    /// [`render_compositor_layers`](Self::render_compositor_layers) writing rows directly into sink-provided storage.
    pub fn render_compositor_layers_into(
        &mut self,
        gpu: &GpuContext,
        config: &CompositorConfig,
        layers: &[CompositorLayer<'_>],
        sink: &mut dyn FrameSink,
    ) -> Result<(), CompositorError> {
        if config.width == 0 || config.height == 0 {
            return Err(CompositorError::InvalidDimensions {
                width: config.width,
                height: config.height,
            });
        }

        let device = &gpu.device;
        self.ensure_target(device, config.width, config.height);
        self.ensure_offscreen(device, config.width, config.height);
        // Upload every LUT the scene references before target/offscreen are
        // borrowed; the loop below only reads the cache.
        for item in layers {
            let lut = match item {
                CompositorLayer::Layer(layer) => layer.lut,
                CompositorLayer::CanvasPass { lut, .. } => *lut,
                // Transition sides skip LUTs (like effect chains).
                CompositorLayer::Transition { .. } => None,
            };
            if let Some(LayerLut { key, lut, .. }) = lut {
                self.ensure_lut(gpu, key, lut);
            }
        }
        let target = self
            .target
            .as_ref()
            .expect("ensure_target populates the cache");
        let offscreen = self.offscreen.as_ref().expect("offscreen pool");

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cutlass.encoder"),
        });

        // Clear canvas.
        {
            let bg = config.background;
            let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("cutlass.clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: f64::from(bg[0]) / 255.0,
                            g: f64::from(bg[1]) / 255.0,
                            b: f64::from(bg[2]) / 255.0,
                            a: f64::from(bg[3]) / 255.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }

        for item in layers {
            match item {
                CompositorLayer::Layer(layer)
                    if !effects_need_offscreen(layer.effects) && layer.lut.is_none() =>
                {
                    let built = self.build_layer(gpu, config, layer)?;
                    let pipeline = pipeline_for(&built.pipeline, self);
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("cutlass.pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &target.view,
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
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &built.bind_group, &[]);
                    pass.draw(0..6, 0..1);
                }
                CompositorLayer::Layer(layer) => {
                    let built = self.build_layer(gpu, config, layer)?;
                    let pipeline = pipeline_for(&built.pipeline, self);
                    {
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("cutlass.offscreen.base"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: offscreen.view(0),
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
                        draw_layer_to_offscreen(&mut pass, pipeline, &built.bind_group);
                    }
                    let mut result = run_effect_chain(
                        device,
                        &mut encoder,
                        &self.pass_registry,
                        offscreen,
                        offscreen.view(0),
                        layer.effects,
                        config.width,
                        config.height,
                    );
                    if let Some(lut) = &layer.lut {
                        let uploaded = &self.lut_cache[lut.key];
                        let lut_output = if std::ptr::eq(result, offscreen.view(1)) {
                            offscreen.view(2)
                        } else {
                            offscreen.view(1)
                        };
                        result = run_lut_pass(
                            device,
                            &mut encoder,
                            &self.pass_registry,
                            result,
                            lut_output,
                            &uploaded.view,
                            uploaded.size,
                            uploaded.domain_min,
                            uploaded.domain_max,
                            lut.intensity,
                        );
                    }
                    blit_premultiplied_to_canvas(
                        device,
                        &mut encoder,
                        &self.pass_registry,
                        result,
                        &target.view,
                    );
                }
                CompositorLayer::CanvasPass {
                    effects,
                    grade,
                    lut,
                } => {
                    if !effects_need_offscreen(effects) && grade.is_none() && lut.is_none() {
                        continue;
                    }

                    // Snapshot the current canvas, process it as a texture,
                    // then replace the canvas. Alpha-over blits would double
                    // the already-drawn stack.
                    blit_replace(
                        device,
                        &mut encoder,
                        &self.pass_registry,
                        &target.view,
                        offscreen.view(0),
                    );
                    let mut result = if effects_need_offscreen(effects) {
                        run_effect_chain(
                            device,
                            &mut encoder,
                            &self.pass_registry,
                            offscreen,
                            offscreen.view(0),
                            effects,
                            config.width,
                            config.height,
                        )
                    } else {
                        offscreen.view(0)
                    };
                    if let Some(grade) = grade {
                        let grade_output = if std::ptr::eq(result, offscreen.view(0)) {
                            offscreen.view(1)
                        } else {
                            offscreen.view(0)
                        };
                        result = run_grade_pass(
                            device,
                            &mut encoder,
                            &self.pass_registry,
                            result,
                            grade_output,
                            *grade,
                        );
                    }
                    if let Some(lut) = lut {
                        let uploaded = &self.lut_cache[lut.key];
                        let lut_output = if std::ptr::eq(result, offscreen.view(1)) {
                            offscreen.view(2)
                        } else {
                            offscreen.view(1)
                        };
                        result = run_lut_pass(
                            device,
                            &mut encoder,
                            &self.pass_registry,
                            result,
                            lut_output,
                            &uploaded.view,
                            uploaded.size,
                            uploaded.domain_min,
                            uploaded.domain_max,
                            lut.intensity,
                        );
                    }
                    blit_replace(
                        device,
                        &mut encoder,
                        &self.pass_registry,
                        result,
                        &target.view,
                    );
                }
                CompositorLayer::Transition {
                    outgoing,
                    incoming,
                    transition_id,
                    progress,
                } => {
                    let out_built = self.build_layer(gpu, config, outgoing)?;
                    let in_built = self.build_layer(gpu, config, incoming)?;
                    let out_pipe = pipeline_for(&out_built.pipeline, self);
                    let in_pipe = pipeline_for(&in_built.pipeline, self);
                    {
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("cutlass.transition.out"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: offscreen.view(0),
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
                        draw_layer_to_offscreen(&mut pass, out_pipe, &out_built.bind_group);
                    }
                    {
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("cutlass.transition.in"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: offscreen.view(1),
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
                        draw_layer_to_offscreen(&mut pass, in_pipe, &in_built.bind_group);
                    }
                    run_transition_pass(
                        device,
                        &mut encoder,
                        &self.pass_registry,
                        offscreen.view(0),
                        offscreen.view(1),
                        offscreen.view(2),
                        transition_id,
                        *progress,
                        config.width,
                        config.height,
                    );
                    blit_premultiplied_to_canvas(
                        device,
                        &mut encoder,
                        &self.pass_registry,
                        offscreen.view(2),
                        &target.view,
                    );
                }
            }
        }

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
                    bytes_per_row: Some(target.padded_bytes_per_row),
                    rows_per_image: Some(config.height),
                },
            },
            wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit(std::iter::once(encoder.finish()));

        let result = map_readback(
            gpu,
            &target.readback,
            target.padded_bytes_per_row,
            config.width,
            config.height,
            sink,
        );
        if result.is_err() {
            self.target = None;
        }
        result
    }
}

/// Map the (already-submitted) readback buffer and copy it into a tight
/// [`RgbaImage`], stripping the per-row copy padding.
fn pipeline_for<'a>(kind: &LayerPipeline, comp: &'a Compositor) -> &'a wgpu::RenderPipeline {
    match kind {
        LayerPipeline::Yuv => &comp.yuv_pipeline,
        LayerPipeline::YuvFx => &comp.yuv_fx_pipeline,
        LayerPipeline::Rgba => &comp.rgba_pipeline,
        LayerPipeline::RgbaFx => &comp.rgba_fx_pipeline,
        LayerPipeline::Solid => &comp.solid_pipeline,
        LayerPipeline::Sdf => &comp.sdf_pipeline,
    }
}

/// Map the (already-submitted) readback buffer and copy its rows — stripped
/// of the per-row copy padding — into `sink`-provided storage. The sink is
/// only consulted once the map succeeded.
fn map_readback(
    gpu: &GpuContext,
    buffer: &wgpu::Buffer,
    padded: u32,
    width: u32,
    height: u32,
    sink: &mut dyn FrameSink,
) -> Result<(), CompositorError> {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|_| CompositorError::MapFailed)?;
    rx.recv()
        .map_err(|_| CompositorError::MapFailed)?
        .map_err(|_| CompositorError::MapFailed)?;

    let unpadded = (width * 4) as usize;
    let mapped = slice.get_mapped_range();
    let pixels = sink.pixels(width, height);
    debug_assert_eq!(pixels.len(), unpadded * height as usize);
    for row in 0..height as usize {
        let src = row * padded as usize;
        let dst = row * unpadded;
        pixels[dst..dst + unpadded].copy_from_slice(&mapped[src..src + unpadded]);
    }
    drop(mapped);
    buffer.unmap();

    Ok(())
}
