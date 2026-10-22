//! A GPU that uploads NV12 or P010 and draws it as RGBA8.

use std::borrow::Cow;

use cutlass_color::{Coefficients, YuvPlanes};
use cutlass_core::PixelFormat;
use wgpu::util::DeviceExt;

use crate::Error;

const SHADER: &str = include_str!("shader.wgsl");

/// One GPU device and the YUV pipeline.
pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
}

/// An RGBA8 picture on the GPU. [`Gpu::read_rgba`] copies it back.
///
/// On Apple the wrapped source planes stay alive with the picture, so a
/// queued draw can finish after the pixel buffer is released.
pub struct RgbImage {
    texture: wgpu::Texture,
    width: u32,
    height: u32,
    /// Keeps the wrapped planes alive until this picture is dropped.
    #[cfg(target_vendor = "apple")]
    _source: Option<crate::import::Imported>,
}

impl RgbImage {
    /// The RGBA texture. A later pass on the same queue can sample it.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }
}

impl Gpu {
    /// Opens the default adapter.
    pub fn new() -> Result<Self, Error> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .map_err(|_| Error::NoAdapter)?;
        let mut required_features = wgpu::Features::empty();
        if adapter
            .features()
            .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
        {
            required_features |= wgpu::Features::TEXTURE_FORMAT_16BIT_NORM;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features,
            ..Default::default()
        }))
        .map_err(|error| Error::Device(error.to_string()))?;
        Ok(Self::with_device(device, queue))
    }

    /// Builds the YUV pipeline on a device the caller already opened.
    ///
    /// The desktop passes the window's device so the preview and the UI share it.
    pub fn with_device(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("yuv"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("yuv"),
            entries: &[
                texture_entry(0),
                texture_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("yuv"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("yuv"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            device,
            queue,
            pipeline,
            layout,
        }
    }

    /// The device this compositor samples on.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The queue that submits the conversion.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Uploads `planes` and draws them into a new RGBA texture.
    pub fn render(&self, planes: &YuvPlanes<'_>) -> Result<RgbImage, Error> {
        if matches!(planes.format, PixelFormat::P010)
            && !self
                .device
                .features()
                .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
        {
            return Err(Error::NoP010);
        }
        let (y_format, chroma_format, sample_bytes, chroma_bytes) = formats(planes.format);
        let width = planes.width;
        let height = planes.height;
        check_size(width, height)?;
        let y_row = sample_bytes
            .checked_mul(width)
            .ok_or(cutlass_color::Error::TooLarge)?;
        let chroma_row = chroma_bytes
            .checked_mul(width / 2)
            .ok_or(cutlass_color::Error::TooLarge)?;
        let y = upload_texture(
            &self.device,
            &self.queue,
            PlaneUpload {
                data: planes.y,
                stride: planes.y_stride,
                row_bytes: y_row,
                width,
                rows: height,
                format: y_format,
                name: "luma",
            },
        )?;
        let chroma = upload_texture(
            &self.device,
            &self.queue,
            PlaneUpload {
                data: planes.chroma,
                stride: planes.chroma_stride,
                row_bytes: chroma_row,
                width: width / 2,
                rows: height / 2,
                format: chroma_format,
                name: "chroma",
            },
        )?;
        let coefficients = Coefficients::new(planes.format, planes.range, planes.matrix);
        let image = self.encode(&y, &chroma, width, height, coefficients)?;
        self.wait()?;
        Ok(image)
    }

    /// Samples an IOSurface-backed frame. The GPU reads the surface, so the
    /// pixels are not copied on the CPU. This returns once the GPU has
    /// finished, and the caller may then reuse the buffer.
    #[cfg(target_vendor = "apple")]
    pub fn render_pixel_buffer(
        &self,
        buffer: &cutlass_videotoolbox::PixelBuffer,
        matrix: cutlass_color::Matrix,
    ) -> Result<RgbImage, Error> {
        let image = self.paint_pixel_buffer(buffer, matrix)?;
        self.wait()?;
        Ok(image)
    }

    /// Queues the same draw and returns before the GPU finishes.
    ///
    /// A later pass on this queue can sample the texture. The picture keeps
    /// the wrapped surface until it is dropped, so drop it after that pass
    /// has finished. Finished work already on the device is reaped here. This
    /// draw is not waited on, and the pixels are not read back.
    #[cfg(target_vendor = "apple")]
    pub fn queue_pixel_buffer(
        &self,
        buffer: &cutlass_videotoolbox::PixelBuffer,
        matrix: cutlass_color::Matrix,
    ) -> Result<RgbImage, Error> {
        let image = self.paint_pixel_buffer(buffer, matrix)?;
        self.device
            .poll(wgpu::PollType::Poll)
            .map_err(|error| Error::Wait(error.to_string()))?;
        Ok(image)
    }

    #[cfg(target_vendor = "apple")]
    fn paint_pixel_buffer(
        &self,
        buffer: &cutlass_videotoolbox::PixelBuffer,
        matrix: cutlass_color::Matrix,
    ) -> Result<RgbImage, Error> {
        let Some((format, range)) = buffer.format() else {
            return Err(Error::UnsupportedFormat);
        };
        if matches!(format, PixelFormat::P010)
            && !self
                .device
                .features()
                .contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM)
        {
            return Err(Error::NoP010);
        }
        let width = u32::try_from(buffer.width()).map_err(|_| cutlass_color::Error::TooLarge)?;
        let height = u32::try_from(buffer.height()).map_err(|_| cutlass_color::Error::TooLarge)?;
        check_size(width, height)?;
        let (y_format, chroma_format, _, _) = formats(format);
        let imported =
            crate::import::import(&self.device, buffer, y_format, chroma_format, width, height)?;
        let coefficients = Coefficients::new(format, range, matrix);
        let mut image = self.encode(&imported.y, &imported.chroma, width, height, coefficients)?;
        image._source = Some(imported);
        Ok(image)
    }

    fn wait(&self) -> Result<(), Error> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| Error::Wait(error.to_string()))?;
        Ok(())
    }

    fn encode(
        &self,
        y: &wgpu::Texture,
        chroma: &wgpu::Texture,
        width: u32,
        height: u32,
        coefficients: Coefficients,
    ) -> Result<RgbImage, Error> {
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("yuv-coefficients"),
                contents: &uniform_bytes(coefficients),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let y_view = y.create_view(&wgpu::TextureViewDescriptor::default());
        let chroma_view = chroma.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("yuv"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&y_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&chroma_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rgba"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("yuv") });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("yuv"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        Ok(RgbImage {
            texture,
            width,
            height,
            #[cfg(target_vendor = "apple")]
            _source: None,
        })
    }

    /// Copies an RGBA texture back to tightly packed bytes.
    pub fn read_rgba(&self, image: &RgbImage) -> Result<Vec<u8>, Error> {
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let row = image
            .width
            .checked_mul(4)
            .ok_or(cutlass_color::Error::TooLarge)?;
        let padded = row.div_ceil(align) * align;
        let rows = image.height;
        let size = u64::from(padded)
            .checked_mul(u64::from(rows))
            .ok_or(cutlass_color::Error::TooLarge)?;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rgba-readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &image.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(rows),
                },
            },
            wgpu::Extent3d {
                width: image.width,
                height: rows,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);
        let slice = buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| Error::Wait(error.to_string()))?;
        receiver
            .recv()
            .map_err(|_| Error::Readback("the map callback did not run".to_owned()))?
            .map_err(|error| Error::Readback(error.to_string()))?;
        let mapped = slice
            .get_mapped_range()
            .map_err(|error| Error::Readback(error.to_string()))?;
        let mut pixels = Vec::with_capacity(
            usize::try_from(row)
                .map_err(|_| cutlass_color::Error::TooLarge)?
                .checked_mul(usize::try_from(rows).map_err(|_| cutlass_color::Error::TooLarge)?)
                .ok_or(cutlass_color::Error::TooLarge)?,
        );
        let row_usize = usize::try_from(row).map_err(|_| cutlass_color::Error::TooLarge)?;
        let padded_usize = usize::try_from(padded).map_err(|_| cutlass_color::Error::TooLarge)?;
        for row_index in 0..usize::try_from(rows).map_err(|_| cutlass_color::Error::TooLarge)? {
            let start = row_index * padded_usize;
            pixels.extend_from_slice(&mapped[start..start + row_usize]);
        }
        drop(mapped);
        buffer.unmap();
        Ok(pixels)
    }
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn formats(format: PixelFormat) -> (wgpu::TextureFormat, wgpu::TextureFormat, u32, u32) {
    match format {
        PixelFormat::Nv12 => (
            wgpu::TextureFormat::R8Unorm,
            wgpu::TextureFormat::Rg8Unorm,
            1,
            2,
        ),
        PixelFormat::P010 => (
            wgpu::TextureFormat::R16Unorm,
            wgpu::TextureFormat::Rg16Unorm,
            2,
            4,
        ),
    }
}

fn check_size(width: u32, height: u32) -> Result<(), Error> {
    if width == 0 || height == 0 {
        return Err(cutlass_color::Error::ZeroDimension.into());
    }
    if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
        return Err(cutlass_color::Error::OddDimension.into());
    }
    Ok(())
}

struct PlaneUpload<'a> {
    data: &'a [u8],
    stride: usize,
    row_bytes: u32,
    width: u32,
    rows: u32,
    format: wgpu::TextureFormat,
    name: &'static str,
}

fn upload_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plane: PlaneUpload<'_>,
) -> Result<wgpu::Texture, Error> {
    let row_bytes_usize =
        usize::try_from(plane.row_bytes).map_err(|_| cutlass_color::Error::TooLarge)?;
    let rows_usize = usize::try_from(plane.rows).map_err(|_| cutlass_color::Error::TooLarge)?;
    if plane.stride < row_bytes_usize
        || rows_usize == 0
        || plane
            .stride
            .checked_mul(rows_usize - 1)
            .and_then(|offset| offset.checked_add(row_bytes_usize - 1))
            .is_none_or(|last| last >= plane.data.len())
    {
        return Err(cutlass_color::Error::ShortPlane(plane.name).into());
    }
    let (bytes, bytes_per_row) =
        aligned_rows(plane.data, plane.stride, row_bytes_usize, rows_usize)?;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(plane.name),
        size: wgpu::Extent3d {
            width: plane.width,
            height: plane.rows,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: plane.format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
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
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: Some(plane.rows),
        },
        wgpu::Extent3d {
            width: plane.width,
            height: plane.rows,
            depth_or_array_layers: 1,
        },
    );
    Ok(texture)
}

fn aligned_rows(
    data: &[u8],
    stride: usize,
    row_bytes: usize,
    rows: usize,
) -> Result<(Cow<'_, [u8]>, u32), Error> {
    let align = usize::try_from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        .map_err(|_| cutlass_color::Error::TooLarge)?;
    let padded = row_bytes.div_ceil(align) * align;
    let padded_u32 = u32::try_from(padded).map_err(|_| cutlass_color::Error::TooLarge)?;
    if stride == padded {
        let end = stride
            .checked_mul(rows)
            .ok_or(cutlass_color::Error::TooLarge)?;
        return Ok((Cow::Borrowed(&data[..end]), padded_u32));
    }
    let mut packed = vec![
        0;
        padded
            .checked_mul(rows)
            .ok_or(cutlass_color::Error::TooLarge)?
    ];
    for row in 0..rows {
        let source = row * stride;
        let dest = row * padded;
        packed[dest..dest + row_bytes].copy_from_slice(&data[source..source + row_bytes]);
    }
    Ok((Cow::Owned(packed), padded_u32))
}

fn uniform_bytes(coefficients: Coefficients) -> [u8; 32] {
    let mut bytes = [0; 32];
    for (index, value) in coefficients.gpu_values().into_iter().enumerate() {
        let start = index * 4;
        bytes[start..start + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}
