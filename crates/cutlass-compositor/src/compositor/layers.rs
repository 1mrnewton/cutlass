use wgpu::util::DeviceExt;

#[cfg(any(target_vendor = "apple", target_os = "windows"))]
use cutlass_core::PixelFormat;
use cutlass_core::{FrameData, RgbaImage, VideoFrame};

use crate::error::CompositorError;
use crate::gpu::GpuContext;
use crate::grade::ColorGrade;
use crate::layer::{CompositeLayer, CompositorConfig, LayerContent, LayerEffects, LayerPlacement};

use super::upload::{
    luma_coeffs, pack_grade, placement_affine, placement_affine_rot, placement_rotation,
    upload_cpu_planes, visible_uv_rect,
};
use super::{
    Compositor, FxEffectsUniform, LayerGpu, LayerPipeline, RgbaUniforms, SdfUniforms,
    SolidUniforms, YuvUniforms,
};

impl Compositor {
    pub(super) fn build_layer(
        &self,
        gpu: &GpuContext,
        config: &CompositorConfig,
        layer: &CompositeLayer<'_>,
    ) -> Result<LayerGpu, CompositorError> {
        let grade = layer.color_grade;
        match &layer.content {
            LayerContent::Solid(rgba) => self.build_solid(gpu, config, layer, *rgba, grade),
            LayerContent::Frame(frame) => self.build_frame(gpu, config, layer, frame, grade),
            LayerContent::Rgba(image) => self.build_rgba(gpu, config, layer, image, grade),
            LayerContent::Sdf(shape) => self.build_sdf(gpu, config, layer, shape, grade),
        }
    }

    fn build_sdf(
        &self,
        gpu: &GpuContext,
        config: &CompositorConfig,
        layer: &CompositeLayer<'_>,
        shape: &crate::layer::SdfLayer,
        color_grade: Option<ColorGrade>,
    ) -> Result<LayerGpu, CompositorError> {
        use cutlass_shapes::SdfParams;

        let placement = &layer.placement;
        let (linear, mut trans) = placement_affine(config, &layer.placement, 0.0);
        let stroke = shape.stroke.filter(|s| s.width > 0.0);
        trans[3] = stroke.map_or(0.0, |s| s.width);

        let h = shape.shape.half;
        // Shape kind ids and parameter packing per shape.wgsl's `Sdf` block.
        let (kind, radius, star) = match shape.shape.params {
            SdfParams::RoundedRect { radius } => (0.0, radius, [0.0, 0.0]),
            SdfParams::Ellipse => (1.0, 0.0, [0.0, 0.0]),
            SdfParams::Star {
                points,
                inner,
                round,
            } => (2.0, round, [points as f32, inner]),
            SdfParams::Line => (3.0, 0.0, [0.0, 0.0]),
            SdfParams::Arrow => (4.0, 0.0, [0.0, 0.0]),
            SdfParams::Heart => (5.0, 0.0, [0.0, 0.0]),
        };

        let color = |rgba: [u8; 4]| {
            [
                f32::from(rgba[0]) / 255.0,
                f32::from(rgba[1]) / 255.0,
                f32::from(rgba[2]) / 255.0,
                f32::from(rgba[3]) / 255.0,
            ]
        };
        let (grade_adj0, grade_adj1) = pack_grade(color_grade);
        let uniforms = SdfUniforms {
            fill: color(shape.fill),
            stroke_color: stroke.map_or([0.0; 4], |s| color(s.rgba)),
            linear,
            trans_opacity: trans,
            geo: [h[0], h[1], kind, radius],
            star: [
                star[0],
                star[1],
                placement.size[0] * 0.5,
                placement.size[1] * 0.5,
            ],
            grade_adj0,
            grade_adj1,
        };
        let buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cutlass.sdf.uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cutlass.sdf.bg"),
            layout: &self.sdf_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Ok(LayerGpu {
            pipeline: LayerPipeline::Sdf,
            bind_group,
            _textures: Vec::new(),
            _uniform: buffer,
            _keep_alive: None,
        })
    }

    fn build_solid(
        &self,
        gpu: &GpuContext,
        config: &CompositorConfig,
        layer: &CompositeLayer<'_>,
        rgba: [u8; 4],
        color_grade: Option<ColorGrade>,
    ) -> Result<LayerGpu, CompositorError> {
        let (linear, trans) = placement_affine(config, &layer.placement, 0.0);
        let (grade_adj0, grade_adj1) = pack_grade(color_grade);
        let uniforms = SolidUniforms {
            color: [
                f32::from(rgba[0]) / 255.0,
                f32::from(rgba[1]) / 255.0,
                f32::from(rgba[2]) / 255.0,
                f32::from(rgba[3]) / 255.0,
            ],
            linear,
            trans_opacity: trans,
            grade_adj0,
            grade_adj1,
        };
        let buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cutlass.solid.uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cutlass.solid.bg"),
            layout: &self.solid_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Ok(LayerGpu {
            pipeline: LayerPipeline::Solid,
            bind_group,
            _textures: Vec::new(),
            _uniform: buffer,
            _keep_alive: None,
        })
    }

    fn build_frame(
        &self,
        gpu: &GpuContext,
        config: &CompositorConfig,
        layer: &CompositeLayer<'_>,
        frame: &VideoFrame,
        color_grade: Option<ColorGrade>,
    ) -> Result<LayerGpu, CompositorError> {
        let (cw, ch) = frame.coded_size;
        if cw == 0 || ch == 0 {
            return Err(CompositorError::MalformedFrame(format!(
                "coded size is {cw}x{ch}"
            )));
        }

        // Acquire the plane textures: CPU frames upload their planes; GPU frames
        // map their native surface straight into textures with no pixel copy.
        // `nv12` selects the shader's chroma path; `keep_alive` parks any native
        // wrappers (e.g. CVMetalTexture) that must outlive the GPU read.
        let (textures, nv12, keep_alive) = match &frame.data {
            FrameData::Cpu(image) => {
                let (textures, nv12) = upload_cpu_planes(gpu, frame, image, cw, ch)?;
                (textures, nv12, None)
            }
            FrameData::Gpu(surface) => self.import_gpu_planes(gpu, frame, surface, cw, ch)?,
        };

        // For NV12 the V slot is bound to the same CbCr texture (unsampled).
        let y_view = &textures[0].1;
        let u_view = &textures[1].1;
        let v_view = if nv12 { &textures[1].1 } else { &textures[2].1 };

        let resolved = frame.color.resolved();
        let (kr, kb) = luma_coeffs(resolved.matrix);
        let full = if resolved.range.is_full() { 1.0 } else { 0.0 };
        let plane_mode = if nv12 { 1.0 } else { 0.0 };

        // Container rotation rides on top of the placement rotation.
        let rotation = placement_rotation(layer, frame);
        let (linear, trans) = placement_affine_rot(config, &layer.placement, rotation);
        let uv_rect = visible_uv_rect(frame, layer.uv);

        let (grade_adj0, grade_adj1) = pack_grade(color_grade);
        let uniforms = YuvUniforms {
            linear,
            trans_opacity: trans,
            uv_rect,
            coeffs: [kr, kb, full, plane_mode],
            grade_adj0,
            grade_adj1,
        };

        let plane_entries = |layout: &wgpu::BindGroupLayout,
                             placement_buf: &wgpu::Buffer,
                             fx_buf: Option<&wgpu::Buffer>| {
            let mut entries = vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(y_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(u_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(v_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: placement_buf.as_entire_binding(),
                },
            ];
            if let Some(fx) = fx_buf {
                entries.push(wgpu::BindGroupEntry {
                    binding: 5,
                    resource: fx.as_entire_binding(),
                });
            }
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cutlass.yuv.bg"),
                layout,
                entries: &entries,
            })
        };

        if layer.fx.is_identity() {
            let buffer = gpu
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("cutlass.yuv.uniforms"),
                    contents: bytemuck::bytes_of(&uniforms),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bind_group = plane_entries(&self.yuv_layout, &buffer, None);
            return Ok(LayerGpu {
                pipeline: LayerPipeline::Yuv,
                bind_group,
                _textures: textures.into_iter().map(|(tex, _)| tex).collect(),
                _uniform: buffer,
                _keep_alive: keep_alive,
            });
        }

        let fx_uniforms = pack_fx_uniforms(&layer.fx, &layer.placement);
        let placement_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cutlass.yuv_fx.placement"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let fx_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cutlass.yuv_fx.effects"),
                contents: bytemuck::bytes_of(&fx_uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = plane_entries(&self.yuv_fx_layout, &placement_buffer, Some(&fx_buffer));

        Ok(LayerGpu {
            pipeline: LayerPipeline::YuvFx,
            bind_group,
            _textures: textures.into_iter().map(|(tex, _)| tex).collect(),
            _uniform: placement_buffer,
            _keep_alive: Some(Box::new((fx_buffer, keep_alive))),
        })
    }

    /// Map a zero-copy [`FrameData::Gpu`] surface into plane textures the YUV
    /// pipeline can sample. Returns the textures, the NV12 flag, and a native
    /// keep-alive bundle. On platforms/formats without an import path this is a
    /// [`CompositorError::UnsupportedFormat`] so the caller can fall back to a
    /// CPU-mode decode.
    #[allow(clippy::type_complexity)]
    fn import_gpu_planes(
        &self,
        gpu: &GpuContext,
        frame: &VideoFrame,
        surface: &cutlass_core::GpuSurface,
        cw: u32,
        ch: u32,
    ) -> Result<
        (
            Vec<(wgpu::Texture, wgpu::TextureView)>,
            bool,
            Option<Box<dyn core::any::Any + Send>>,
        ),
        CompositorError,
    > {
        #[cfg(target_vendor = "apple")]
        {
            use cutlass_core::GpuSurfaceKind;
            match surface.kind() {
                GpuSurfaceKind::CoreVideoPixelBuffer => {
                    if frame.format != PixelFormat::Nv12 {
                        return Err(CompositorError::UnsupportedFormat(format!(
                            "zero-copy import supports NV12 CoreVideo surfaces; got {:?}",
                            frame.format
                        )));
                    }
                    let importer = self.metal_import.as_ref().ok_or_else(|| {
                        CompositorError::UnsupportedFormat(
                            "Metal surface import is unavailable on this device".into(),
                        )
                    })?;
                    let (textures, keep) = importer.import_nv12(gpu, surface, (cw, ch))?;
                    Ok((textures, true, Some(Box::new(keep))))
                }
                other => Err(CompositorError::UnsupportedFormat(format!(
                    "GPU surface kind {other:?} import is not implemented yet"
                ))),
            }
        }
        #[cfg(target_os = "windows")]
        {
            use cutlass_core::GpuSurfaceKind;
            match surface.kind() {
                GpuSurfaceKind::D3D11Texture2D => {
                    if frame.format != PixelFormat::Nv12 {
                        return Err(CompositorError::UnsupportedFormat(format!(
                            "zero-copy import supports NV12 D3D11 surfaces; got {:?}",
                            frame.format
                        )));
                    }
                    let importer = self.dx12_import.as_ref().ok_or_else(|| {
                        CompositorError::UnsupportedFormat(
                            "D3D12 surface import is unavailable on this device".into(),
                        )
                    })?;
                    let textures = importer.import_nv12(gpu, surface, (cw, ch))?;
                    // No compositor-side keep-alive needed: the wgpu texture
                    // holds the opened ID3D12Resource, and the decoder-side
                    // pool slot rides in the *frame's* keep-alive, which the
                    // renderer holds until the pass completes.
                    Ok((textures, true, None))
                }
                other => Err(CompositorError::UnsupportedFormat(format!(
                    "GPU surface kind {other:?} import is not implemented yet"
                ))),
            }
        }
        // Android (`AHardwareBuffer`) import is not wired yet; the decoder
        // delivers CPU planes there, so this branch is currently unreachable
        // in practice.
        //
        // Android plan: decode into a `Surface`/`ImageReader` to obtain an
        // `AHardwareBuffer`, then import on the Vulkan backend via
        // `VK_ANDROID_external_memory_android_hardware_buffer`: create a
        // `VkImage` with external memory, bind the imported allocation, and
        // adopt it with `wgpu::hal::vulkan::Device::texture_from_raw`.
        //
        // Blocker (researched against wgpu 29, gfx-rs/wgpu#2320 / #3145 / #7324):
        // robust YCbCr sampling needs a `VkSamplerYcbcrConversion` bound as an
        // *immutable* sampler, and real devices hand back vendor formats
        // (`VK_FORMAT_UNDEFINED` + `VkExternalFormatANDROID`, e.g. UBWC) that
        // *only* resolve through that conversion. wgpu-hal hardcodes
        // `p_immutable_samplers = null` and its format map rejects
        // `VK_FORMAT_UNDEFINED`, so neither the immutable-sampler binding nor the
        // external format is expressible through wgpu's safe API today. The
        // Metal trick (split planes into R8/RG8 and convert in our shader) does
        // not carry over: per-plane VkImage aliasing only works for the rare
        // concrete-format AHBs, not the vendor-tiled ones that dominate.
        //
        // So the viable route is a raw-Vulkan YCbCr->RGBA conversion pass on a
        // device shared with wgpu (own `VkInstance`/`VkDevice` with the AHB +
        // ycbcr extensions, handed to wgpu per #7324/#7829), writing into an
        // RGBA `VkImage` that wgpu then samples — GPU-resident throughout, no CPU
        // copy, but not a single-sample import. Until that lands, Android stays
        // on the hardware-decode + CPU-plane path (one plane copy/frame), which
        // is already wired and correct.
        #[cfg(not(any(target_vendor = "apple", target_os = "windows")))]
        {
            let _ = (gpu, frame, surface, cw, ch);
            Err(CompositorError::UnsupportedFormat(
                "zero-copy GPU surface import is not implemented on this platform yet".into(),
            ))
        }
    }

    fn build_rgba(
        &self,
        gpu: &GpuContext,
        config: &CompositorConfig,
        layer: &CompositeLayer<'_>,
        image: &RgbaImage,
        color_grade: Option<ColorGrade>,
    ) -> Result<LayerGpu, CompositorError> {
        if image.width == 0 || image.height == 0 {
            return Err(CompositorError::MalformedFrame(format!(
                "rgba bitmap is {}x{}",
                image.width, image.height
            )));
        }
        let expected = (image.width as usize)
            .checked_mul(image.height as usize)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| CompositorError::MalformedFrame("rgba bitmap size overflow".into()))?;
        if image.pixels.len() < expected {
            return Err(CompositorError::MalformedFrame(format!(
                "rgba bitmap: {} bytes < {}×{}×4",
                image.pixels.len(),
                image.width,
                image.height
            )));
        }

        // Premultiply straight-alpha input so bilinear sampling and src-over
        // blending don't leak color from transparent texels into glyph edges.
        // Unwrapped text can be wider than the adapter's texture limit. Fit
        // only the upload bitmap while retaining the original layer placement
        // below, so the title keeps its authored on-canvas dimensions.
        let max_dimension = gpu.device.limits().max_texture_dimension_2d;
        let (premul, upload_width, upload_height) =
            rgba_upload_pixels(image, expected, max_dimension)?;

        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cutlass.rgba.tex"),
            size: wgpu::Extent3d {
                width: upload_width,
                height: upload_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
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
            &premul,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(upload_width * 4),
                rows_per_image: Some(upload_height),
            },
            wgpu::Extent3d {
                width: upload_width,
                height: upload_height,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let (linear, trans) = placement_affine(config, &layer.placement, 0.0);
        let (grade_adj0, grade_adj1) = pack_grade(color_grade);
        let placement_uniforms = RgbaUniforms {
            linear,
            trans_opacity: trans,
            uv_rect: layer.uv,
            grade_adj0,
            grade_adj1,
        };

        if layer.fx.is_identity() {
            let buffer = gpu
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("cutlass.rgba.uniforms"),
                    contents: bytemuck::bytes_of(&placement_uniforms),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cutlass.rgba.bg"),
                layout: &self.rgba_layout,
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

            return Ok(LayerGpu {
                pipeline: LayerPipeline::Rgba,
                bind_group,
                _textures: vec![texture],
                _uniform: buffer,
                _keep_alive: None,
            });
        }

        let fx_uniforms = pack_fx_uniforms(&layer.fx, &layer.placement);
        let placement_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cutlass.rgba_fx.placement"),
                contents: bytemuck::bytes_of(&placement_uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let fx_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cutlass.rgba_fx.effects"),
                contents: bytemuck::bytes_of(&fx_uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cutlass.rgba_fx.bg"),
            layout: &self.rgba_fx_layout,
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
                    resource: placement_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: fx_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(LayerGpu {
            pipeline: LayerPipeline::RgbaFx,
            bind_group,
            _textures: vec![texture],
            _uniform: placement_buffer,
            _keep_alive: Some(Box::new(fx_buffer)),
        })
    }
}

/// Convert straight-alpha RGBA to the compositor's premultiplied upload
/// format, reducing oversized bitmaps to the device's 2D texture limit.
fn rgba_upload_pixels(
    image: &RgbaImage,
    expected: usize,
    max_dimension: u32,
) -> Result<(Vec<u8>, u32, u32), CompositorError> {
    let mut premul = vec![0u8; expected];
    for (dst, src) in premul.chunks_exact_mut(4).zip(image.pixels.chunks_exact(4)) {
        let a = u16::from(src[3]);
        dst[0] = ((u16::from(src[0]) * a + 127) / 255) as u8;
        dst[1] = ((u16::from(src[1]) * a + 127) / 255) as u8;
        dst[2] = ((u16::from(src[2]) * a + 127) / 255) as u8;
        dst[3] = src[3];
    }

    if image.width <= max_dimension && image.height <= max_dimension {
        return Ok((premul, image.width, image.height));
    }
    if max_dimension == 0 {
        return Err(CompositorError::MalformedFrame(
            "GPU reports a zero 2D texture limit".into(),
        ));
    }

    let scale = (max_dimension as f64 / f64::from(image.width))
        .min(max_dimension as f64 / f64::from(image.height));
    let width = ((f64::from(image.width) * scale).round() as u32).clamp(1, max_dimension);
    let height = ((f64::from(image.height) * scale).round() as u32).clamp(1, max_dimension);
    let source = image::RgbaImage::from_raw(image.width, image.height, premul)
        .ok_or_else(|| CompositorError::MalformedFrame("invalid RGBA upload buffer".into()))?;
    let resized = image::imageops::resize(
        &source,
        width,
        height,
        image::imageops::FilterType::Lanczos3,
    );
    Ok((resized.into_raw(), width, height))
}

fn pack_fx_uniforms(effects: &LayerEffects, placement: &LayerPlacement) -> FxEffectsUniform {
    let mask = effects
        .mask
        .map(|m| [m.kind as f32, m.feather, m.invert as f32, 1.0]);
    let chroma = effects
        .chroma_key
        .map(|c| [c.rgb[0], c.rgb[1], c.rgb[2], 1.0]);
    let chroma_params = effects.chroma_key.map(|c| [c.strength, c.shadow, 0.0, 0.0]);
    FxEffectsUniform {
        mask: mask.unwrap_or([0.0, 0.0, 0.0, 0.0]),
        chroma: chroma.unwrap_or([0.0, 0.0, 0.0, 0.0]),
        chroma_params: chroma_params.unwrap_or([0.0, 0.0, 0.0, 0.0]),
        half: [placement.size[0] * 0.5, placement.size[1] * 0.5, 0.0, 0.0],
    }
}

#[cfg(test)]
mod tests {
    use super::rgba_upload_pixels;
    use crate::{CompositeLayer, Compositor, CompositorConfig, GpuContext, LayerPlacement};
    use cutlass_core::RgbaImage;

    #[test]
    fn rgba_upload_premultiplies_without_resizing_small_images() {
        let image = RgbaImage::new(1, 1, vec![200, 100, 50, 128]);
        let (pixels, width, height) = rgba_upload_pixels(&image, 4, 16_384).unwrap();
        assert_eq!((width, height), (1, 1));
        assert_eq!(pixels, [100, 50, 25, 128]);
    }

    #[test]
    fn rgba_upload_fits_unwrapped_text_to_gpu_texture_limit() {
        let image = RgbaImage::new(19_285, 4, vec![255; 19_285 * 4 * 4]);
        let (pixels, width, height) =
            rgba_upload_pixels(&image, image.pixels.len(), 16_384).unwrap();
        assert_eq!(width, 16_384);
        assert!(height <= 16_384);
        assert_eq!(pixels.len(), width as usize * height as usize * 4);
    }

    #[test]
    fn compositor_accepts_bitmap_wider_than_device_texture_limit() {
        let Ok(gpu) = GpuContext::new_headless_blocking() else {
            return;
        };
        let width = gpu.device.limits().max_texture_dimension_2d + 1;
        let bitmap = RgbaImage::new(width, 2, vec![255; width as usize * 2 * 4]);
        let config = CompositorConfig::new(8, 8);
        let layer = CompositeLayer::rgba(&bitmap, LayerPlacement::full_canvas(&config));
        let mut compositor = Compositor::new(&gpu);
        let rendered = compositor
            .render(&gpu, &config, &[layer])
            .expect("oversized RGBA layer should be fitted before upload");
        assert_eq!(rendered.pixel(4, 4), [255, 255, 255, 255]);
    }
}
