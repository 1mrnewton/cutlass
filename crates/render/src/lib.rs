//! The wgpu compositor.
//!
//! [`Renderer`] draws layers into an RGBA canvas texture, back to front,
//! with premultiplied "over": `out = src + dst * (1 - src.a)`. A layer is a
//! texture and a mapping from canvas pixels to that texture's pixels. Video
//! arrives as 4:2:0 planes and is converted in the shader, so the CPU never
//! touches a decoded picture's pixels.
//!
//! The result matches `cutlass-renderer-software` and the CPU placement in
//! `cutlass-eval` to within rounding: texels are fetched, not filtered, and
//! each stage of the CPU path floors to a whole pixel before the next.
//!
//! [`Renderer::draw_frame`] draws eval's frame layers, so the desktop and
//! the phones draw a frame the same way.
//!
//! The desktop preview shares the device eframe already opened, so the canvas can be
//! shown without a copy. Export and the tests open their own device with
//! [`Renderer::headless`] and read the canvas back.

#![forbid(unsafe_code)]

mod blur;
mod effect;
mod frame;
mod layer;
mod mix;
mod target;

pub use cutlass_decode::Surface;
pub use cutlass_effects::{Block, Recipe, Step};
pub use cutlass_gpu::{Affine, Placement};
pub use frame::frame_layers;
pub use layer::{Layer, Plane, PlaneSource, Planes, Rgba};
pub use target::Canvas;
pub use wgpu;

use std::fmt;

/// Set in CI so a test that needs a GPU fails instead of skipping.
pub const REQUIRE_GPU: &str = "CUTLASS_REQUIRE_GPU";

/// The canvas pixel format. Bytes are premultiplied RGBA.
pub const CANVAS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Why the renderer could not start or finish a frame.
#[derive(Debug)]
pub enum RenderError {
    /// No adapter or device is available.
    NoDevice(String),
    /// A canvas or layer edge is zero or past the device's limit.
    Size { width: u32, height: u32 },
    /// Layer bytes do not match the size they claim.
    Bytes { expected: usize, actual: usize },
    /// The canvas could not be read back.
    Readback(String),
}

impl fmt::Display for RenderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoDevice(reason) => write!(formatter, "no GPU device: {reason}"),
            Self::Size { width, height } => {
                write!(formatter, "{width}x{height} is not a size the GPU can draw")
            }
            Self::Bytes { expected, actual } => {
                write!(formatter, "layer has {actual} bytes, expected {expected}")
            }
            Self::Readback(reason) => write!(formatter, "could not read the canvas: {reason}"),
        }
    }
}

impl std::error::Error for RenderError {}

/// Pipelines and a sampler on one device.
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layers: layer::Pipelines,
    blurs: blur::Blurs,
    effects: effect::Effects,
    mix: mix::Mix,
    import: Option<SurfaceImport>,
}

/// Imports a platform surface on `device` as two textures the layer shader
/// samples in place: luma as [`wgpu::TextureFormat::R8Unorm`] and chroma as
/// [`wgpu::TextureFormat::Rg8Unorm`], Cb in red and Cr in green, at half
/// size. `None` when the surface is not one this import knows; the renderer
/// then draws the bytes the surface copies out.
pub type SurfaceImport =
    std::sync::Arc<dyn Fn(&wgpu::Device, &Surface) -> Option<[wgpu::Texture; 2]> + Send + Sync>;

impl Renderer {
    /// A renderer on a device someone else opened, such as eframe's.
    #[must_use]
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let layers = layer::Pipelines::new(&device);
        let blurs = blur::Blurs::new(&device);
        let effects = effect::Effects::new(&device);
        let mix = mix::Mix::new(&device);
        Self {
            device,
            queue,
            layers,
            blurs,
            effects,
            mix,
            import: None,
        }
    }

    /// Sample platform surfaces in place through `import` from now on.
    pub fn set_surface_import(&mut self, import: SurfaceImport) {
        self.import = Some(import);
    }

    /// A renderer on its own device, for export and tests.
    pub fn headless() -> Result<Self, RenderError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..wgpu::RequestAdapterOptions::default()
        }))
        .map_err(|error| RenderError::NoDevice(error.to_string()))?;
        // Ask for what this adapter has, not wgpu's desktop defaults: the iOS
        // simulator, for one, falls short of them and its device would not open.
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("cutlass-render"),
            required_limits: adapter.limits(),
            ..wgpu::DeviceDescriptor::default()
        }))
        .map_err(|error| RenderError::NoDevice(error.to_string()))?;
        Ok(Self::new(device, queue))
    }

    /// A headless renderer for a test. `Ok(None)` means this machine has no
    /// GPU and the test may skip; with [`REQUIRE_GPU`] set that is an error.
    pub fn for_test() -> Result<Option<Self>, RenderError> {
        match Self::headless() {
            Ok(renderer) => Ok(Some(renderer)),
            Err(error) if std::env::var_os(REQUIRE_GPU).is_some() => Err(error),
            Err(_) => Ok(None),
        }
    }

    #[must_use]
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    #[must_use]
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// A canvas of `width` by `height` pixels.
    pub fn canvas(&self, width: u32, height: u32) -> Result<Canvas, RenderError> {
        Canvas::new(&self.device, width, height)
    }

    /// Clear `canvas` to transparent and draw `layers` back to front.
    pub fn draw(&self, canvas: &Canvas, layers: &[Layer<'_>]) -> Result<(), RenderError> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cutlass-render frame"),
            });
        self.draw_into(&mut encoder, canvas.view(), canvas.size(), layers)?;
        self.queue.submit([encoder.finish()]);
        Ok(())
    }

    /// Clear `target` and draw `layers` into it. A nested layer is drawn
    /// into its own texture first, on the same encoder.
    fn draw_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        size: (u32, u32),
        layers: &[Layer<'_>],
    ) -> Result<(), RenderError> {
        let (device, queue) = (&self.device, &self.queue);
        let mut bound = Vec::with_capacity(layers.len());
        for layer in layers {
            match layer {
                Layer::Rgba(rgba, placement) => {
                    let uploaded = layer::Pipelines::upload_rgba(device, queue, rgba)?;
                    bound.push(self.layers.bind(device, queue, &uploaded, *placement));
                }
                Layer::Yuv(planes, placement) => {
                    let uploaded = layer::Pipelines::upload_planes(
                        device,
                        queue,
                        planes,
                        self.import.as_ref(),
                    )?;
                    bound.push(self.layers.bind(device, queue, &uploaded, *placement));
                }
                Layer::YuvOverBlur {
                    planes,
                    placement,
                    cover,
                    radius,
                } => {
                    let uploaded = layer::Pipelines::upload_planes(
                        device,
                        queue,
                        planes,
                        self.import.as_ref(),
                    )?;
                    let backdrop = self.blurred(encoder, &uploaded, *cover, *radius, size);
                    bound.push(self.layers.bind(
                        device,
                        queue,
                        &backdrop,
                        Placement::canvas(size.0, size.1),
                    ));
                    bound.push(self.layers.bind(device, queue, &uploaded, *placement));
                }
                Layer::Nested {
                    layers,
                    size: inner,
                    placement,
                } => {
                    let limit = device.limits().max_texture_dimension_2d;
                    if inner.0 == 0 || inner.1 == 0 || inner.0 > limit || inner.1 > limit {
                        return Err(RenderError::Size {
                            width: inner.0,
                            height: inner.1,
                        });
                    }
                    let view = blur::scratch(device, *inner);
                    self.draw_into(encoder, &view, *inner, layers)?;
                    let uploaded = layer::Uploaded {
                        views: [view.clone(), view.clone(), view],
                        size: *inner,
                        yuv: false,
                        tone: cutlass_gpu::YuvTone::NONE,
                        interleaved: false,
                    };
                    bound.push(self.layers.bind(device, queue, &uploaded, *placement));
                }
                Layer::Blend {
                    mode,
                    layers: inner,
                } => {
                    // The clip alone, what is below it, then the two mixed;
                    // the mix is the bottom layer from here on.
                    let top = blur::scratch(device, size);
                    self.draw_into(encoder, &top, size, inner)?;
                    let below = blur::scratch(device, size);
                    Self::paint(&self.layers, encoder, &below, &bound);
                    let mixed = blur::scratch(device, size);
                    self.mix
                        .run(device, queue, encoder, &top, &below, &mixed, *mode);
                    let uploaded = layer::Uploaded {
                        views: [mixed.clone(), mixed.clone(), mixed],
                        size,
                        yuv: false,
                        tone: cutlass_gpu::YuvTone::NONE,
                        interleaved: false,
                    };
                    bound.clear();
                    bound.push(self.layers.bind(
                        device,
                        queue,
                        &uploaded,
                        Placement::canvas(size.0, size.1),
                    ));
                }
                Layer::Effect(steps) => {
                    if steps.is_empty() {
                        continue;
                    }
                    // What is drawn so far becomes one picture the steps
                    // run over, and that result is the bottom layer again.
                    let first = blur::scratch(device, size);
                    let second = blur::scratch(device, size);
                    Self::paint(&self.layers, encoder, &first, &bound);
                    let result = self
                        .effects
                        .run(device, queue, encoder, &first, &second, size, steps)
                        .clone();
                    let uploaded = layer::Uploaded {
                        views: [result.clone(), result.clone(), result],
                        size,
                        yuv: false,
                        tone: cutlass_gpu::YuvTone::NONE,
                        interleaved: false,
                    };
                    bound.clear();
                    bound.push(self.layers.bind(
                        device,
                        queue,
                        &uploaded,
                        Placement::canvas(size.0, size.1),
                    ));
                }
            }
        }
        Self::paint(&self.layers, encoder, target, &bound);
        Ok(())
    }

    /// Clear `target` and draw `bound` into it in one pass.
    fn paint(
        layers: &layer::Pipelines,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        bound: &[layer::Bound],
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("cutlass-render layers"),
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
        for layer in bound {
            layers.draw(&mut pass, layer);
        }
    }

    /// Draw `uploaded` through `cover` into a canvas-sized texture and blur
    /// it, ahead of the frame's own pass.
    fn blurred(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        uploaded: &layer::Uploaded,
        cover: Placement,
        radius: u32,
        size: (u32, u32),
    ) -> layer::Uploaded {
        let (device, queue) = (&self.device, &self.queue);
        let first = blur::scratch(device, size);
        let second = blur::scratch(device, size);
        let covered = self.layers.bind(device, queue, uploaded, cover);
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("cutlass-render cover"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &first,
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
            self.layers.draw(&mut pass, &covered);
        }
        self.blurs
            .run(device, queue, encoder, &first, &second, size, radius);
        layer::Uploaded {
            views: [first.clone(), first.clone(), first],
            size,
            yuv: false,
            tone: cutlass_gpu::YuvTone::NONE,
            interleaved: false,
        }
    }

    /// The canvas as premultiplied RGBA bytes, row by row with no padding.
    pub fn read(&self, canvas: &Canvas) -> Result<Vec<u8>, RenderError> {
        canvas.read(&self.device, &self.queue)
    }
}

#[cfg(test)]
mod tests {
    use super::{Affine, Layer, Placement, Planes, Renderer, Rgba};

    fn renderer() -> Option<Renderer> {
        let renderer = Renderer::for_test().expect("a GPU is required");
        if renderer.is_none() {
            eprintln!("skipped: no GPU");
        }
        renderer
    }

    fn close(got: &[u8], expected: &[u8], tolerance: u8) {
        assert_eq!(got.len(), expected.len());
        for (index, (a, b)) in got.iter().zip(expected).enumerate() {
            assert!(a.abs_diff(*b) <= tolerance, "byte {index}: {a} vs {b}");
        }
    }

    #[test]
    fn a_canvas_sized_layer_comes_back_unchanged() {
        let Some(renderer) = renderer() else { return };
        let bytes: Vec<u8> = (0..3 * 2 * 4).map(|value| (value * 10) as u8).collect();
        let premultiplied: Vec<u8> = bytes
            .chunks(4)
            .flat_map(|px| [px[0].min(px[3]), px[1].min(px[3]), px[2].min(px[3]), px[3]])
            .collect();
        let canvas = renderer.canvas(3, 2).unwrap();
        let layer = Rgba {
            width: 3,
            height: 2,
            bytes: &premultiplied,
        };
        renderer
            .draw(&canvas, &[Layer::Rgba(layer, Placement::canvas(3, 2))])
            .unwrap();
        assert_eq!(renderer.read(&canvas).unwrap(), premultiplied);
    }

    #[test]
    fn layers_blend_with_premultiplied_over() {
        let Some(renderer) = renderer() else { return };
        let below = [200, 100, 50, 255];
        let above = [60, 0, 30, 128];
        let canvas = renderer.canvas(1, 1).unwrap();
        let layer = |bytes| {
            Layer::Rgba(
                Rgba {
                    width: 1,
                    height: 1,
                    bytes,
                },
                Placement::canvas(1, 1),
            )
        };
        renderer
            .draw(&canvas, &[layer(&below[..]), layer(&above[..])])
            .unwrap();
        let expected: Vec<u8> = below
            .iter()
            .zip(above)
            .map(|(d, s)| (u32::from(s) + u32::from(*d) * (255 - 128) / 255) as u8)
            .collect();
        close(&renderer.read(&canvas).unwrap(), &expected, 1);
    }

    #[test]
    fn video_range_bt709_red_comes_back_red() {
        let Some(renderer) = renderer() else { return };
        // Limited-range BT.709 red: Y 63, Cb 102, Cr 240.
        let mut bytes = vec![63; 4];
        bytes.push(102);
        bytes.push(240);
        let planes = Planes::packed(2, 2, &bytes).unwrap();
        let canvas = renderer.canvas(2, 2).unwrap();
        renderer
            .draw(&canvas, &[Layer::Yuv(planes, Placement::canvas(2, 2))])
            .unwrap();
        for pixel in renderer.read(&canvas).unwrap().chunks(4) {
            assert!(pixel[0] > 250 && pixel[1] < 6 && pixel[2] < 6, "{pixel:?}");
            assert_eq!(pixel[3], 255);
        }
    }

    #[test]
    fn a_quarter_turn_and_half_opacity_land_where_the_maps_say() {
        let Some(renderer) = renderer() else { return };
        // A 2x1 texture, red then green, turned clockwise into 1x2.
        let bytes = [255, 0, 0, 255, 0, 255, 0, 255];
        let placement = Placement {
            to_picture: Affine::IDENTITY,
            picture: (1, 2),
            to_oriented: Affine::IDENTITY,
            oriented: (1, 2),
            // Clockwise90: stored = (y, stored_h - x).
            to_texture: Affine([0.0, 1.0, 0.0, -1.0, 0.0, 1.0]),
            opacity: 0.5,
        };
        let canvas = renderer.canvas(1, 2).unwrap();
        let layer = Rgba {
            width: 2,
            height: 1,
            bytes: &bytes,
        };
        renderer
            .draw(&canvas, &[Layer::Rgba(layer, placement)])
            .unwrap();
        close(
            &renderer.read(&canvas).unwrap(),
            &[128, 0, 0, 128, 0, 128, 0, 128],
            1,
        );
    }
}
