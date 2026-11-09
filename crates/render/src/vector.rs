//! The vector pass: a Lottie frame drawn from its edges, as
//! `cutlass-lottie`'s CPU rasterizer draws it.
//!
//! The scene is packed into one RGBA32F texture. Each fill is one quad over
//! its pixel bounds whose fragments count the samples its edges enclose and
//! read its paint; a run of fills is one instanced draw. A group's items
//! are drawn on a texture of their own first, its matte on another, and a
//! quad over the whole output lands them through the group's opacity,
//! masks and matte. Every draw blends with premultiplied "over".

use cutlass_lottie::{DATA_WIDTH, PackedItem, Scene, WGSL, pack};

use crate::{CANVAS_FORMAT, RenderError, blur};

pub(crate) struct Vector {
    fill: wgpu::RenderPipeline,
    group: wgpu::RenderPipeline,
    data_layout: wgpu::BindGroupLayout,
    pictures_layout: wgpu::BindGroupLayout,
}

/// What one scene's draws share.
struct Shared {
    data: wgpu::BindGroup,
    instances: wgpu::Buffer,
    empty: wgpu::TextureView,
    size: (u32, u32),
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

impl Vector {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cutlass-render vector"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
        let data_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cutlass-render vector data"),
            entries: &[texture_entry(0)],
        });
        let pictures_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cutlass-render vector pictures"),
            entries: &[texture_entry(0), texture_entry(1)],
        });
        let over = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        };
        let instance = [wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Uint32,
            offset: 0,
            shader_location: 0,
        }];
        let pipeline = |label: &str, layouts: &[Option<&wgpu::BindGroupLayout>], entry: &str| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: layouts,
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: 4,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &instance,
                    })],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: CANVAS_FORMAT,
                        blend: Some(wgpu::BlendState {
                            color: over,
                            alpha: over,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let fill = pipeline(
            "cutlass-render vector fill",
            &[Some(&data_layout)],
            "fs_fill",
        );
        let group = pipeline(
            "cutlass-render vector group",
            &[Some(&data_layout), Some(&pictures_layout)],
            "fs_group",
        );
        Self {
            fill,
            group,
            data_layout,
            pictures_layout,
        }
    }

    /// Clear `target`, a texture the scene's size, and draw `scene` into it.
    pub(crate) fn draw(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        scene: &Scene,
        target: &wgpu::TextureView,
    ) -> Result<(), RenderError> {
        let packed = pack(scene);
        let rows = packed.rows();
        let limit = device.limits().max_texture_dimension_2d;
        if rows == 0 || rows > limit || DATA_WIDTH > limit {
            return Err(RenderError::Size {
                width: DATA_WIDTH,
                height: rows,
            });
        }
        let extent = wgpu::Extent3d {
            width: DATA_WIDTH,
            height: rows,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cutlass-render vector data"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let bytes: Vec<u8> = packed
            .texels
            .iter()
            .flatten()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(DATA_WIDTH * 16),
                rows_per_image: Some(rows),
            },
            extent,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let data = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cutlass-render vector data"),
            layout: &self.data_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });
        // At least one slot, so the buffer is never empty.
        let mut records: Vec<u8> = packed
            .records
            .iter()
            .flat_map(|record| record.to_le_bytes())
            .collect();
        if records.is_empty() {
            records.extend([0; 4]);
        }
        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cutlass-render vector instances"),
            size: records.len() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&instances, 0, &records);
        let shared = Shared {
            data,
            instances,
            empty: blur::scratch(device, (1, 1)),
            size: (scene.width, scene.height),
        };
        self.paint(device, encoder, &shared, &packed.items, target);
        Ok(())
    }

    /// Clear `target` and draw `items` into it, each group's own textures
    /// first.
    fn paint(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        shared: &Shared,
        items: &[PackedItem],
        target: &wgpu::TextureView,
    ) {
        let mut pictures = Vec::new();
        for item in items {
            if let PackedItem::Group { items, matte, .. } = item {
                let child = blur::scratch(device, shared.size);
                self.paint(device, encoder, shared, items, &child);
                let matte = match matte {
                    Some(matte) => {
                        let view = blur::scratch(device, shared.size);
                        self.paint(device, encoder, shared, matte, &view);
                        view
                    }
                    None => shared.empty.clone(),
                };
                pictures.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("cutlass-render vector pictures"),
                    layout: &self.pictures_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&child),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&matte),
                        },
                    ],
                }));
            }
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("cutlass-render vector"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
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
        pass.set_bind_group(0, &shared.data, &[]);
        let mut pictures = pictures.iter();
        let mut at = 0;
        while at < items.len() {
            match &items[at] {
                PackedItem::Fill { slot } => {
                    // Fills in a row have slots in a row: one instanced draw.
                    let mut run = 1;
                    while matches!(items.get(at + run), Some(PackedItem::Fill { .. })) {
                        run += 1;
                    }
                    pass.set_pipeline(&self.fill);
                    pass.set_vertex_buffer(0, shared.instances.slice(u64::from(*slot) * 4..));
                    pass.draw(0..6, 0..run as u32);
                    at += run;
                }
                PackedItem::Group { slot, .. } => {
                    if let Some(group) = pictures.next() {
                        pass.set_pipeline(&self.group);
                        pass.set_bind_group(1, group, &[]);
                        pass.set_vertex_buffer(0, shared.instances.slice(u64::from(*slot) * 4..));
                        pass.draw(0..6, 0..1);
                    }
                    at += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use cutlass_gpu::Placement;
    use cutlass_lottie::{Animation, Scene, rasterize};

    use crate::{Layer, Renderer};

    const STICKER: &str = include_str!("../../lottie/tests/fixtures/sticker.json");
    const MASKS: &str = include_str!("../../lottie/tests/fixtures/masks.json");
    const MATTE: &str = include_str!("../../lottie/tests/fixtures/matte.json");
    const PRECOMP: &str = include_str!("../../lottie/tests/fixtures/precomp.json");
    const TRIM: &str = include_str!("../../lottie/tests/fixtures/trim.json");

    fn renderer() -> Option<Renderer> {
        let renderer = Renderer::for_test().expect("a GPU is required");
        if renderer.is_none() {
            eprintln!("skipped: no GPU");
        }
        renderer
    }

    fn gpu(renderer: &Renderer, scene: &Scene) -> Vec<u8> {
        let canvas = renderer.canvas(scene.width, scene.height).expect("canvas");
        renderer
            .draw(
                &canvas,
                &[Layer::Vector {
                    scene,
                    placement: Placement::canvas(scene.width, scene.height),
                }],
            )
            .expect("draw");
        renderer.read(&canvas).expect("read")
    }

    /// GPU and CPU agree to a rounding step. A device that rounds an edge
    /// crossing differently may move one sample of a pixel (1/16, 16 steps)
    /// on a few pixels; no more.
    fn agree(label: &str, gpu: &[u8], cpu: &[u8]) {
        assert_eq!(gpu.len(), cpu.len());
        let mut worst = 0;
        let mut off = 0;
        let mut total = 0_u64;
        for (a, b) in gpu.iter().zip(cpu) {
            let difference = a.abs_diff(*b);
            worst = worst.max(difference);
            total += u64::from(difference);
            if difference > 2 {
                off += 1;
            }
        }
        let mean = total as f64 / gpu.len() as f64;
        assert!(worst <= 17, "{label}: worst {worst}");
        assert!(
            off * 1_000 <= gpu.len(),
            "{label}: {off} bytes differ by more than 2"
        );
        assert!(mean < 0.05, "{label}: mean {mean}");
        assert!(cpu.iter().any(|byte| *byte != 0), "{label}: nothing drawn");
    }

    #[test]
    fn every_fixture_frame_on_the_gpu_matches_the_cpu() {
        let Some(renderer) = renderer() else { return };
        for (label, text, frames) in [
            ("sticker", STICKER, &[0.0, 17.0, 30.0, 44.5][..]),
            ("masks", MASKS, &[0.0][..]),
            ("matte", MATTE, &[0.0][..]),
            ("precomp", PRECOMP, &[10.0, 50.0][..]),
            ("trim", TRIM, &[3.0, 7.0][..]),
        ] {
            let animation = Animation::parse(text.as_bytes()).expect("parse");
            for &at in frames {
                for (width, height) in [(128, 128), (97, 61)] {
                    let scene = animation.scene(at, width, height).expect("scene");
                    agree(
                        &format!("{label} frame {at} at {width}x{height}"),
                        &gpu(&renderer, &scene),
                        &rasterize(&scene),
                    );
                }
            }
        }
    }

    #[test]
    fn a_placed_lottie_sticker_draws_the_same_frame_on_both_paths() {
        use cutlass_eval::{Micros, OpenClips, ReaderFactory, Snapshot};
        use cutlass_model::{Draft, MaterialVideo, Segment, TimeRange, Track, TrackType};
        use cutlass_renderer_software::SoftwareDevice;

        let Some(renderer) = renderer() else { return };
        let folder =
            std::env::temp_dir().join(format!("cutlass-render-lottie-{}", std::process::id()));
        std::fs::create_dir_all(&folder).expect("folder");
        let path = folder.join("sticker.json");
        std::fs::write(&path, STICKER).expect("write");
        let mut draft = Draft::empty("draft");
        draft.canvas_config.width = 160;
        draft.canvas_config.height = 90;
        let mut video = MaterialVideo::new("sticker");
        video.kind = "sticker".to_owned();
        video.path = path.display().to_string();
        draft.materials.videos.push(video);
        let mut row = Track::main("stickers");
        row.kind = TrackType::Sticker;
        let mut segment = Segment::new("sticker", "sticker", TimeRange::new(0, 4_000_000));
        segment.source_range = Some(TimeRange::new(0, 4_000_000));
        segment.clip.scale = cutlass_model::Point::new(0.6, 0.6);
        segment.clip.transform = cutlass_model::Point::new(0.3, -0.2);
        segment.clip.alpha = 0.8;
        row.segments.push(segment);
        draft.tracks.push(row);
        let media = ReaderFactory::new();
        for time in [0, 700_000, 2_500_000] {
            let mut device = SoftwareDevice::new();
            let mut clips = OpenClips::new();
            let snapshot = Snapshot::new(&draft);
            let cpu =
                cutlass_eval::frame(&snapshot, Micros(time), &mut device, &media, &mut clips, 0)
                    .expect("cpu frame");
            let layers = cutlass_eval::layers_in(
                &snapshot,
                Micros(time),
                &mut device,
                &media,
                &mut clips,
                0,
                None,
            )
            .expect("layers");
            assert!(
                layers
                    .layers
                    .iter()
                    .any(|layer| matches!(layer.source, cutlass_eval::GpuSource::Vector(_))),
                "the sticker is not a vector layer"
            );
            let canvas = renderer.canvas(160, 90).expect("canvas");
            renderer.draw_frame(&canvas, &layers).expect("draw");
            agree(
                &format!("placed sticker at {time}"),
                &renderer.read(&canvas).expect("read"),
                &cpu.rgba,
            );
        }
        let _ = std::fs::remove_dir_all(folder);
    }

    #[test]
    fn an_empty_scene_clears_the_canvas() {
        let Some(renderer) = renderer() else { return };
        let scene = Scene {
            width: 4,
            height: 3,
            items: Vec::new(),
        };
        assert_eq!(gpu(&renderer, &scene), vec![0; 4 * 3 * 4]);
    }

    #[test]
    fn a_placed_scene_lands_where_its_placement_says() {
        let Some(renderer) = renderer() else { return };
        let animation = Animation::parse(MATTE.as_bytes()).expect("parse");
        let scene = animation.scene(0.0, 10, 10).expect("scene");
        // The 10 by 10 frame at the canvas's top left, pixel for pixel.
        let canvas = renderer.canvas(20, 10).expect("canvas");
        let placement = Placement {
            picture: (10, 10),
            oriented: (10, 10),
            ..Placement::canvas(10, 10)
        };
        renderer
            .draw(
                &canvas,
                &[Layer::Vector {
                    scene: &scene,
                    placement,
                }],
            )
            .expect("draw");
        let read = renderer.read(&canvas).expect("read");
        let cpu = rasterize(&scene);
        for y in 0..10 {
            for x in 0..20 {
                let at = (y * 20 + x) * 4;
                let expected = if x < 10 {
                    cpu[(y * 10 + x) * 4..(y * 10 + x) * 4 + 4].to_vec()
                } else {
                    vec![0; 4]
                };
                assert_eq!(read[at..at + 4].to_vec(), expected, "pixel {x}, {y}");
            }
        }
    }
}
