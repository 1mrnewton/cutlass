//! Glyph atlas packing and instanced GPU upload for per-character text.
//!
//! Cluster bitmaps (from `cutlass-text::TextRenderer::shape`) are packed into
//! a shared RGBA atlas once per [`GlyphsLayer::atlas_key`]. Subsequent frames
//! with the same key only rewrite the instance buffer. Eviction is
//! **byte-budgeted** ([`RASTER_MEMO_BUDGET_BYTES`]) by atlas RGBA payload —
//! same soft cap as the CPU text/path memos — so supersampled glyph runs
//! cannot retain multiple GiB of GPU atlases.

use bytemuck::{Pod, Zeroable};
use cutlass_core::RgbaImage;
use wgpu::util::DeviceExt;

use crate::error::CompositorError;
use crate::gpu::GpuContext;
use crate::grade::ColorGrade;
use crate::layer::{CompositeLayer, CompositorConfig, GlyphInstance, GlyphsLayer};

use super::upload::pack_grade;
use super::{Compositor, InstanceDraw, LayerGpu, LayerPipeline};

/// Default atlas edge length in pixels. Grows only by rebuilding a larger
/// texture when a pack fails (rare for title-card text).
const ATLAS_BASE: u32 = 1024;

/// One packed atlas texture keyed by [`GlyphsLayer::atlas_key`].
#[derive(Clone)]
pub(super) struct CachedAtlas {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    /// Atlas UV rect `[u0, v0, u1, v1]` per glyph index. Zero-area for empty
    /// (whitespace) glyphs that were not packed.
    uvs: Vec<[f32; 4]>,
    /// Number of glyphs the key was packed with — a mismatch forces rebuild.
    glyph_count: usize,
    /// RGBA payload bytes used for the byte-budget LRU (`edge² × 4`).
    bytes: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct GlyphsUniforms {
    grade_adj0: [f32; 4],
    grade_adj1: [f32; 4],
    grade_adj2: [f32; 4],
    /// Canvas width/height (x, y), layer opacity (z), pad (w).
    canvas: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct GlyphInstanceGpu {
    center_size: [f32; 4],
    rot_opacity: [f32; 4],
    uv_rect: [f32; 4],
}

impl GlyphInstanceGpu {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<GlyphInstanceGpu>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: 16,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: 32,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x4,
            },
        ],
    };
}

impl Compositor {
    pub(super) fn build_glyphs(
        &self,
        gpu: &GpuContext,
        config: &CompositorConfig,
        layer: &CompositeLayer<'_>,
        glyphs: &GlyphsLayer<'_>,
        color_grade: Option<ColorGrade>,
    ) -> Result<LayerGpu, CompositorError> {
        if glyphs.glyphs.is_empty() || glyphs.instances.is_empty() {
            return Err(CompositorError::MalformedFrame(
                "glyphs layer has no drawable instances".into(),
            ));
        }

        let CachedAtlasHandles {
            view: atlas_view,
            texture: atlas_tex,
            uvs,
        } = self.ensure_atlas(gpu, glyphs)?;
        let (grade_adj0, grade_adj1, grade_adj2) = pack_grade(color_grade);
        let uniforms = GlyphsUniforms {
            grade_adj0,
            grade_adj1,
            grade_adj2,
            canvas: [
                config.width as f32,
                config.height as f32,
                layer.placement.opacity.clamp(0.0, 1.0),
                0.0,
            ],
        };
        let uniform = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cutlass.glyphs.uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let mut instances = Vec::with_capacity(glyphs.instances.len());
        for inst in glyphs.instances {
            let Some(uv) = uvs.get(inst.glyph as usize).copied() else {
                continue;
            };
            // Skip whitespace / empty clusters (zero UV area).
            if uv[2] <= uv[0] || uv[3] <= uv[1] {
                continue;
            }
            if inst.size[0] <= 0.0 || inst.size[1] <= 0.0 || inst.opacity <= 0.0 {
                continue;
            }
            instances.push(GlyphInstanceGpu {
                center_size: [inst.center[0], inst.center[1], inst.size[0], inst.size[1]],
                rot_opacity: [inst.rotation, inst.opacity.clamp(0.0, 1.0), 0.0, 0.0],
                uv_rect: uv,
            });
        }
        if instances.is_empty() {
            return Err(CompositorError::MalformedFrame(
                "glyphs layer produced no visible instances".into(),
            ));
        }

        let instance_buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cutlass.glyphs.instances"),
                contents: bytemuck::cast_slice(&instances),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cutlass.glyphs.bg"),
            layout: &self.glyphs_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });

        Ok(LayerGpu {
            pipeline: LayerPipeline::Glyphs,
            bind_group,
            _textures: Vec::new(),
            _uniform: uniform,
            _keep_alive: Some(Box::new((atlas_tex, atlas_view))),
            instances: Some(InstanceDraw {
                buffer: instance_buf,
                count: instances.len() as u32,
            }),
        })
    }

    fn ensure_atlas(
        &self,
        gpu: &GpuContext,
        glyphs: &GlyphsLayer<'_>,
    ) -> Result<CachedAtlasHandles, CompositorError> {
        let key = glyphs.atlas_key;
        let mut cache = self.glyph_atlases.borrow_mut();
        let cached = cache.get_cloned(&key);
        let needs_build = match &cached {
            Some(entry) => entry.glyph_count != glyphs.glyphs.len(),
            None => true,
        };
        let atlas = if needs_build {
            let packed = pack_atlas(gpu, glyphs.glyphs)?;
            let cost = packed.bytes;
            cache.insert(key, packed.clone(), cost);
            packed
        } else {
            cached.expect("atlas present when needs_build is false")
        };
        Ok(CachedAtlasHandles {
            view: atlas.view,
            texture: atlas.texture,
            uvs: atlas.uvs,
        })
    }
}

/// Cloned handles from a cache entry so the `RefCell` borrow can end.
struct CachedAtlasHandles {
    view: wgpu::TextureView,
    texture: wgpu::Texture,
    uvs: Vec<[f32; 4]>,
}

/// Pack straight-alpha glyph bitmaps into a premultiplied atlas texture.
fn pack_atlas(gpu: &GpuContext, glyphs: &[RgbaImage]) -> Result<CachedAtlas, CompositorError> {
    let mut size = ATLAS_BASE;
    let max_dim = gpu.device.limits().max_texture_dimension_2d.min(4096);
    loop {
        match try_pack(glyphs, size) {
            Ok((pixels, uvs)) => {
                let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("cutlass.glyphs.atlas"),
                    size: wgpu::Extent3d {
                        width: size,
                        height: size,
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
                    &pixels,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(size * 4),
                        rows_per_image: Some(size),
                    },
                    wgpu::Extent3d {
                        width: size,
                        height: size,
                        depth_or_array_layers: 1,
                    },
                );
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                let bytes = (size as usize) * (size as usize) * 4;
                return Ok(CachedAtlas {
                    texture,
                    view,
                    uvs,
                    glyph_count: glyphs.len(),
                    bytes,
                });
            }
            Err(PackError::DoesNotFit) if size < max_dim => {
                size = (size * 2).min(max_dim);
            }
            Err(PackError::DoesNotFit) => {
                return Err(CompositorError::MalformedFrame(format!(
                    "glyph atlas does not fit in {max_dim}×{max_dim}"
                )));
            }
            Err(PackError::BadBitmap(msg)) => {
                return Err(CompositorError::MalformedFrame(msg));
            }
        }
    }
}

enum PackError {
    DoesNotFit,
    BadBitmap(String),
}

/// Shelf-pack glyphs into an `atlas_size`² premultiplied RGBA buffer.
fn try_pack(glyphs: &[RgbaImage], atlas_size: u32) -> Result<(Vec<u8>, Vec<[f32; 4]>), PackError> {
    let mut pixels = vec![0u8; (atlas_size as usize) * (atlas_size as usize) * 4];
    let mut uvs = Vec::with_capacity(glyphs.len());
    // Shelf state: current row y, row height, next x.
    let mut shelf_y = 1u32; // 1px border
    let mut shelf_h = 0u32;
    let mut cursor_x = 1u32;

    for image in glyphs {
        if image.width == 0 || image.height == 0 {
            uvs.push([0.0, 0.0, 0.0, 0.0]);
            continue;
        }
        let expected = (image.width as usize)
            .checked_mul(image.height as usize)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| PackError::BadBitmap("glyph bitmap size overflow".into()))?;
        if image.pixels.len() < expected {
            return Err(PackError::BadBitmap(format!(
                "glyph bitmap: {} bytes < {}×{}×4",
                image.pixels.len(),
                image.width,
                image.height
            )));
        }

        // 1px padding around each glyph to avoid bilinear bleed.
        let need_w = image.width + 2;
        let need_h = image.height + 2;
        if need_w + 1 > atlas_size || need_h + 1 > atlas_size {
            return Err(PackError::DoesNotFit);
        }
        if cursor_x + need_w + 1 > atlas_size {
            shelf_y += shelf_h;
            cursor_x = 1;
            shelf_h = 0;
        }
        if shelf_y + need_h + 1 > atlas_size {
            return Err(PackError::DoesNotFit);
        }

        let dest_x = cursor_x + 1;
        let dest_y = shelf_y + 1;
        blit_premul(&mut pixels, atlas_size, dest_x, dest_y, image, expected);
        let inv = 1.0 / atlas_size as f32;
        uvs.push([
            dest_x as f32 * inv,
            dest_y as f32 * inv,
            (dest_x + image.width) as f32 * inv,
            (dest_y + image.height) as f32 * inv,
        ]);

        cursor_x += need_w;
        shelf_h = shelf_h.max(need_h);
    }

    Ok((pixels, uvs))
}

/// Premultiply and copy one glyph into the atlas at `(dx, dy)`.
fn blit_premul(
    atlas: &mut [u8],
    atlas_w: u32,
    dx: u32,
    dy: u32,
    image: &RgbaImage,
    expected: usize,
) {
    for row in 0..image.height {
        for col in 0..image.width {
            let src_i = ((row * image.width + col) * 4) as usize;
            if src_i + 3 >= expected {
                break;
            }
            let (r, g, b, a) = (
                image.pixels[src_i],
                image.pixels[src_i + 1],
                image.pixels[src_i + 2],
                image.pixels[src_i + 3],
            );
            let dst_i = (((dy + row) * atlas_w + (dx + col)) * 4) as usize;
            if a == 255 {
                atlas[dst_i] = r;
                atlas[dst_i + 1] = g;
                atlas[dst_i + 2] = b;
                atlas[dst_i + 3] = a;
            } else if a == 0 {
                atlas[dst_i] = 0;
                atlas[dst_i + 1] = 0;
                atlas[dst_i + 2] = 0;
                atlas[dst_i + 3] = 0;
            } else {
                let af = f32::from(a) / 255.0;
                atlas[dst_i] = (f32::from(r) * af).round() as u8;
                atlas[dst_i + 1] = (f32::from(g) * af).round() as u8;
                atlas[dst_i + 2] = (f32::from(b) * af).round() as u8;
                atlas[dst_i + 3] = a;
            }
        }
    }
}

/// Helper used by tests: place each glyph at its natural size with identity
/// transform so GPU compositing of clusters matches CPU `rasterize`.
pub fn identity_instances(
    glyphs: &[RgbaImage],
    offsets: &[[f32; 2]],
    origin: [f32; 2],
    scale: f32,
) -> Vec<GlyphInstance> {
    glyphs
        .iter()
        .zip(offsets.iter())
        .enumerate()
        .filter(|(_, (img, _))| img.width > 0 && img.height > 0)
        .map(|(i, (img, offset))| {
            let size = [img.width as f32 * scale, img.height as f32 * scale];
            GlyphInstance {
                glyph: i as u32,
                center: [
                    origin[0] + offset[0] * scale + size[0] * 0.5,
                    origin[1] + offset[1] * scale + size[1] * 0.5,
                ],
                size,
                rotation: 0.0,
                opacity: 1.0,
            }
        })
        .collect()
}
