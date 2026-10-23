//! The pictures under the playhead.
//!
//! The readers live in `cutlass-engine`. This module uploads each frame the
//! engine delivers and keeps the texture. A failure is written to the app log.
//! Where no video decoder is linked, there is no picture.

use cutlass_compose::Gpu;
use cutlass_project::Project;

#[cfg(target_vendor = "apple")]
use cutlass_color::{Matrix, YuvPlanes};
#[cfg(target_vendor = "apple")]
use cutlass_core::Timestamp;
#[cfg(target_vendor = "apple")]
use cutlass_engine::{Ask, Body, LayerFrame, Pictures, ShowLayer, ShowNext, shows_frame};
#[cfg(target_vendor = "apple")]
use cutlass_project::{AssetId, TrackId};
#[cfg(target_vendor = "apple")]
use cutlass_videotoolbox::PixelBuffer;

/// The open readers and the GPU the window already created.
pub struct Reel {
    gpu: Option<Gpu>,
    #[cfg(target_vendor = "apple")]
    pictures: Option<Pictures>,
    /// One held picture per track, bottom to top as last asked.
    #[cfg(target_vendor = "apple")]
    held: Vec<Held>,
    #[cfg(target_vendor = "apple")]
    pending: Option<Ask>,
    #[cfg(target_vendor = "apple")]
    waiting: bool,
    #[cfg(target_vendor = "apple")]
    generation: u64,
    /// The GPU import already failed for this run. The next frames stay quiet.
    #[cfg(target_vendor = "apple")]
    copied: bool,
    /// A dropped frame was already written. The next drops stay quiet.
    #[cfg(target_vendor = "apple")]
    dropped: bool,
}

#[cfg(target_vendor = "apple")]
struct Held {
    track: TrackId,
    asset: AssetId,
    source: String,
    generation: u64,
    /// GPU picture the preview samples. The texture stays until the next frame.
    plate: Option<cutlass_compose::RgbImage>,
    /// egui id for `plate`. The renderer does not own the wgpu texture.
    native: Option<eframe::egui::TextureId>,
    /// CPU fallback, used when the frame has no surface the GPU can sample.
    texture: Option<eframe::egui::TextureHandle>,
    shown: Option<Shown>,
}

impl Reel {
    /// `gpu` is the window's device. Without one, the plate stays the background.
    pub fn new(gpu: Option<Gpu>) -> Self {
        Self {
            #[cfg(target_vendor = "apple")]
            pictures: gpu.is_some().then(|| Pictures::new(proxy())),
            gpu,
            #[cfg(target_vendor = "apple")]
            held: Vec::new(),
            #[cfg(target_vendor = "apple")]
            pending: None,
            #[cfg(target_vendor = "apple")]
            waiting: false,
            #[cfg(target_vendor = "apple")]
            generation: 0,
            #[cfg(target_vendor = "apple")]
            copied: false,
            #[cfg(target_vendor = "apple")]
            dropped: false,
        }
    }

    /// Drops the open clip. The GPU stays.
    pub fn clear(&mut self) {
        #[cfg(target_vendor = "apple")]
        {
            self.pending = None;
            self.held.clear();
            self.waiting = false;
            self.copied = false;
            self.dropped = false;
            self.generation = self.generation.wrapping_add(1);
            if let Some(pictures) = &self.pictures {
                pictures.drop_all();
            }
        }
    }

    /// A frame was asked for and has not arrived.
    pub fn waiting(&self) -> bool {
        #[cfg(target_vendor = "apple")]
        {
            self.waiting
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            false
        }
    }

    /// Textures for the pictures at `playhead_us`, bottom track first.
    pub fn images(
        &mut self,
        ctx: &eframe::egui::Context,
        state: Option<&eframe::egui_wgpu::RenderState>,
        project: &Project,
        playhead_us: i64,
        hidden: &[cutlass_project::TrackId],
        playing: bool,
    ) -> Vec<crate::preview::DrawnPicture> {
        if self.gpu.is_none() {
            return Vec::new();
        }
        #[cfg(target_vendor = "apple")]
        {
            self.images_apple(ctx, state, project, playhead_us, hidden, playing)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (ctx, state, project, playhead_us, hidden, playing);
            Vec::new()
        }
    }

    #[cfg(target_vendor = "apple")]
    fn images_apple(
        &mut self,
        ctx: &eframe::egui::Context,
        state: Option<&eframe::egui_wgpu::RenderState>,
        project: &Project,
        playhead_us: i64,
        hidden: &[cutlass_project::TrackId],
        playing: bool,
    ) -> Vec<crate::preview::DrawnPicture> {
        let layers = crate::preview::picture_layers(project, playhead_us, hidden);
        let next = crate::preview::upcoming_layers(project, playhead_us, hidden);
        self.retain(&layers, state);
        self.ask(project, playhead_us, &layers, &next, playing);
        self.deliver(ctx, state);
        let mut drawn = Vec::new();
        for (track, _) in layers {
            let Some(held) = self.held.iter().find(|held| held.track == track) else {
                continue;
            };
            let Some(texture) = held.texture_id() else {
                continue;
            };
            drawn.push(crate::preview::DrawnPicture { track, texture });
        }
        drawn
    }

    /// Drops pictures whose tracks are no longer under the playhead.
    #[cfg(target_vendor = "apple")]
    fn retain(
        &mut self,
        layers: &[(TrackId, &cutlass_project::Segment)],
        state: Option<&eframe::egui_wgpu::RenderState>,
    ) {
        let mut kept = Vec::new();
        for held in self.held.drain(..) {
            if layers.iter().any(|(track, _)| *track == held.track) {
                kept.push(held);
            } else {
                release(held, state);
            }
        }
        self.held = kept;
    }

    #[cfg(target_vendor = "apple")]
    fn ask(
        &mut self,
        project: &Project,
        playhead_us: i64,
        layers: &[(TrackId, &cutlass_project::Segment)],
        next: &[(TrackId, AssetId, &str)],
        playing: bool,
    ) {
        let mut show = Vec::new();
        for (track, segment) in layers {
            let Some(asset) = segment.asset_id() else {
                continue;
            };
            let Some(source) = project.asset(asset).map(|asset| asset.source()) else {
                continue;
            };
            let Some(time) = crate::preview::media_time(segment, playhead_us) else {
                continue;
            };
            let generation = self.prepare(*track, asset, source);
            show.push(ShowLayer {
                track: *track,
                generation,
                asset,
                source: source.to_owned(),
                time,
            });
        }
        let next = next
            .iter()
            .map(|(track, asset, source)| ShowNext {
                track: *track,
                asset: *asset,
                source: (*source).to_owned(),
            })
            .collect();
        if self.pending.as_ref().is_some_and(|pending| {
            pending.layers == show && pending.next == next && pending.playing == playing
        }) {
            return;
        }
        let ask = Ask {
            layers: show,
            next,
            playing,
        };
        let posted = self
            .pictures
            .as_ref()
            .is_some_and(|pictures| pictures.show(ask.clone()));
        self.pending = Some(ask);
        if posted {
            self.refresh_waiting();
        } else {
            self.waiting = false;
        }
    }

    /// The generation for this track. A new file keeps the last picture.
    #[cfg(target_vendor = "apple")]
    fn prepare(&mut self, track: TrackId, asset: AssetId, source: &str) -> u64 {
        if let Some(held) = self.held.iter_mut().find(|held| held.track == track) {
            if held.asset != asset || held.source != source {
                held.asset = asset;
                held.source = source.to_owned();
                self.generation = self.generation.wrapping_add(1);
                held.generation = self.generation;
                held.shown = None;
            }
            return held.generation;
        }
        self.generation = self.generation.wrapping_add(1);
        let generation = self.generation;
        self.held.push(Held {
            track,
            asset,
            source: source.to_owned(),
            generation,
            plate: None,
            native: None,
            texture: None,
            shown: None,
        });
        generation
    }

    #[cfg(target_vendor = "apple")]
    fn refresh_waiting(&mut self) {
        let Some(pending) = &self.pending else {
            self.waiting = false;
            return;
        };
        self.waiting = pending.layers.iter().any(|layer| {
            self.held
                .iter()
                .find(|held| held.track == layer.track)
                .is_none_or(|held| {
                    held.shown
                        .as_ref()
                        .is_none_or(|shown| shown.requested != layer.time)
                })
        });
    }

    #[cfg(target_vendor = "apple")]
    fn deliver(
        &mut self,
        ctx: &eframe::egui::Context,
        state: Option<&eframe::egui_wgpu::RenderState>,
    ) {
        let Some(state) = state else {
            return;
        };
        let Some(pictures) = &self.pictures else {
            return;
        };
        for frame in pictures.take() {
            self.import(ctx, state, frame);
        }
        self.refresh_waiting();
    }

    #[cfg(target_vendor = "apple")]
    fn import(
        &mut self,
        ctx: &eframe::egui::Context,
        state: &eframe::egui_wgpu::RenderState,
        frame: LayerFrame,
    ) {
        let Some((asked, playing)) = self.pending.as_ref().and_then(|pending| {
            pending
                .layers
                .iter()
                .find(|layer| layer.track == frame.track)
                .map(|layer| (layer.time, pending.playing))
        }) else {
            return;
        };
        let Some(index) = self
            .held
            .iter()
            .position(|held| held.track == frame.track && held.generation == frame.generation)
        else {
            return;
        };
        let Body::Frame {
            buffer,
            matrix,
            pts,
        } = frame.body
        else {
            if frame.requested == asked {
                blank(&mut self.held[index], Some(state));
            }
            return;
        };
        // The playhead moved. A frame that still covers it is shown. While
        // scrubbing backward, a frame that is only closer than the one on
        // screen is shown too, so a fast drag does not sit still. Playback
        // still ignores a frame that is ahead of the clock.
        let shown = self.held[index].shown.as_ref().map(|shown| shown.pts);
        if !shows_frame(playing, asked, pts, frame.requested, shown) {
            return;
        }
        if self.held[index].has_picture()
            && self.held[index]
                .shown
                .as_ref()
                .is_some_and(|shown| shown.pts == pts)
        {
            // A frame still ahead of a backward scrub is not the one the
            // playhead asked for, so the window keeps asking.
            if pts <= asked
                && let Some(shown) = &mut self.held[index].shown
            {
                shown.requested = asked;
            }
            return;
        }
        let imported = {
            let Some(gpu) = &self.gpu else {
                return;
            };
            gpu.queue_pixel_buffer(&buffer, matrix)
        };
        let drawn = match imported {
            Ok(image) => {
                self.copied = false;
                self.dropped = false;
                Drawn::Gpu(Box::new(image))
            }
            Err(error) => {
                if !self.copied {
                    log::warn!(
                        "The preview copied a frame on the CPU. The GPU import failed: {error}."
                    );
                    self.copied = true;
                }
                let copied = {
                    let Some(gpu) = &self.gpu else {
                        return;
                    };
                    rgba_of(gpu, &buffer, matrix)
                };
                match copied {
                    Some((pixels, width, height)) => Drawn::Cpu(pixels, width, height),
                    None => {
                        if !self.dropped {
                            log::warn!("The preview dropped a frame. The CPU copy failed.");
                            self.dropped = true;
                        }
                        Drawn::Failed
                    }
                }
            }
        };
        match drawn {
            Drawn::Gpu(image) => {
                bind(&mut self.held[index].native, state, &image);
                self.held[index].texture = None;
                self.held[index].plate = Some(*image);
            }
            Drawn::Cpu(pixels, width, height) => {
                if store(&mut self.held[index].texture, ctx, &pixels, width, height).is_none() {
                    if !self.dropped {
                        log::warn!("The preview dropped a frame. The CPU copy could not be shown.");
                        self.dropped = true;
                    }
                    return;
                }
                self.held[index].plate = None;
                free_native(&mut self.held[index].native, Some(state));
            }
            Drawn::Failed => return,
        }
        self.held[index].shown = Some(Shown {
            pts,
            requested: if pts <= asked { asked } else { frame.requested },
        });
    }
}

#[cfg(target_vendor = "apple")]
impl Held {
    fn has_picture(&self) -> bool {
        self.plate.is_some() || self.texture.is_some()
    }

    fn texture_id(&self) -> Option<eframe::egui::TextureId> {
        if self.plate.is_some() {
            return self.native;
        }
        self.texture.as_ref().map(|texture| texture.id())
    }
}

/// Points the preview at `image`. The wgpu texture stays owned by the reel.
#[cfg(target_vendor = "apple")]
fn bind(
    native: &mut Option<eframe::egui::TextureId>,
    state: &eframe::egui_wgpu::RenderState,
    image: &cutlass_compose::RgbImage,
) {
    let view = image
        .texture()
        .create_view(&eframe::wgpu::TextureViewDescriptor::default());
    let mut renderer = state.renderer.write();
    if let Some(id) = *native {
        renderer.update_egui_texture_from_wgpu_texture(
            &state.device,
            &view,
            eframe::wgpu::FilterMode::Linear,
            id,
        );
    } else {
        *native = Some(renderer.register_native_texture(
            &state.device,
            &view,
            eframe::wgpu::FilterMode::Linear,
        ));
    }
}

#[cfg(target_vendor = "apple")]
fn free_native(
    native: &mut Option<eframe::egui::TextureId>,
    state: Option<&eframe::egui_wgpu::RenderState>,
) {
    let Some(id) = native.take() else {
        return;
    };
    let Some(state) = state else {
        *native = Some(id);
        return;
    };
    state.renderer.write().free_texture(&id);
}

#[cfg(target_vendor = "apple")]
fn blank(held: &mut Held, state: Option<&eframe::egui_wgpu::RenderState>) {
    held.plate = None;
    held.texture = None;
    held.shown = None;
    free_native(&mut held.native, state);
}

#[cfg(target_vendor = "apple")]
fn release(mut held: Held, state: Option<&eframe::egui_wgpu::RenderState>) {
    blank(&mut held, state);
}

#[cfg(target_vendor = "apple")]
fn store(
    texture: &mut Option<eframe::egui::TextureHandle>,
    ctx: &eframe::egui::Context,
    pixels: &[u8],
    width: u32,
    height: u32,
) -> Option<eframe::egui::TextureId> {
    let width = usize::try_from(width).ok()?;
    let height = usize::try_from(height).ok()?;
    let needed = width.checked_mul(height)?.checked_mul(4)?;
    if width == 0 || height == 0 || pixels.len() != needed {
        return None;
    }
    let image = eframe::egui::ColorImage::from_rgba_unmultiplied([width, height], pixels);
    let options = eframe::egui::TextureOptions::LINEAR;
    if let Some(texture) = texture {
        texture.set(image, options);
    } else {
        *texture = Some(ctx.load_texture("preview-frame", image, options));
    }
    texture.as_ref().map(|texture| texture.id())
}

/// A source string the shell can open. `file://` drops that prefix.
/// `content:` is not a file.
pub(crate) fn file_path(source: &str) -> Option<&str> {
    cutlass_engine::file_path(source)
}

/// The prepared picture, when this system writes one.
#[cfg(target_vendor = "apple")]
fn proxy() -> cutlass_engine::Proxy {
    cutlass_engine::Proxy {
        ensure: crate::preview_file::ensure,
        ready: crate::preview_file::ready,
        discard: crate::preview_file::discard,
    }
}

#[cfg(target_vendor = "apple")]
struct Shown {
    pts: Timestamp,
    requested: Timestamp,
}

#[cfg(target_vendor = "apple")]
enum Drawn {
    Gpu(Box<cutlass_compose::RgbImage>),
    Cpu(Vec<u8>, u32, u32),
    Failed,
}

#[cfg(target_vendor = "apple")]
fn rgba_of(gpu: &Gpu, buffer: &PixelBuffer, matrix: Matrix) -> Option<(Vec<u8>, u32, u32)> {
    let width = u32::try_from(buffer.width()).ok()?;
    let height = u32::try_from(buffer.height()).ok()?;
    if width == 0 || height == 0 {
        return None;
    }
    let image = match gpu.render_pixel_buffer(buffer, matrix) {
        Ok(image) => image,
        Err(_) => cpu_image(gpu, buffer, matrix)?,
    };
    let pixels = gpu.read_rgba(&image).ok()?;
    Some((pixels, width, height))
}

#[cfg(target_vendor = "apple")]
fn cpu_image(gpu: &Gpu, buffer: &PixelBuffer, matrix: Matrix) -> Option<cutlass_compose::RgbImage> {
    let (format, range) = buffer.format()?;
    buffer
        .read_planes(|planes| {
            let [y, chroma, ..] = planes else {
                return None;
            };
            let width = u32::try_from(y.width).ok()?;
            let height = u32::try_from(y.height).ok()?;
            gpu.render(&YuvPlanes {
                format,
                range,
                matrix,
                width,
                height,
                y: y.data,
                y_stride: y.bytes_per_row,
                chroma: chroma.data,
                chroma_stride: chroma.bytes_per_row,
            })
            .ok()
        })
        .ok()
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::file_path;

    #[test]
    fn a_file_url_is_a_path_and_a_content_url_is_not() {
        assert_eq!(file_path("file:///Users/a.mp4"), Some("/Users/a.mp4"));
        assert_eq!(
            file_path("file://localhost/Users/a.mp4"),
            Some("/Users/a.mp4")
        );
        assert_eq!(file_path("/Users/a.mp4"), Some("/Users/a.mp4"));
        assert_eq!(file_path("content://media/1"), None);
        assert_eq!(file_path(""), None);
        assert_eq!(file_path("file://"), None);
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn a_fixture_frame_becomes_packed_rgba() {
        use std::num::NonZeroUsize;

        use cutlass_color::Matrix;
        use cutlass_compose::{Error, Gpu};
        use cutlass_core::{Packet, Rotation};
        use cutlass_media::VideoSession;
        use cutlass_playback::{Ahead, Decode};
        use cutlass_videotoolbox::PixelBuffer;

        struct Session(VideoSession);

        impl Decode for Session {
            type Image = PixelBuffer;
            type Error = cutlass_media::Error;

            fn send(&mut self, packet: &Packet) -> Result<(), Self::Error> {
                self.0.send_packet(packet)
            }

            fn receive(
                &mut self,
            ) -> Result<Option<cutlass_core::VideoFrame<PixelBuffer>>, Self::Error> {
                self.0.receive_frame()
            }

            fn flush(&mut self) -> Result<(), Self::Error> {
                self.0.flush()
            }

            fn reset(&mut self) -> Result<(), Self::Error> {
                self.0.reset()
            }
        }

        let gpu = match Gpu::new() {
            Ok(gpu) => gpu,
            Err(Error::NoAdapter) => {
                eprintln!("skipping: no GPU adapter");
                return;
            }
            Err(error) => panic!("opening a GPU failed: {error}"),
        };
        let fixture = cutlass_test_fixtures::video("h264-high-bframes-128x72");
        let mut session = Session(
            VideoSession::open(
                &fixture.config,
                None,
                None,
                None,
                Rotation::Zero,
                &fixture.packets,
            )
            .expect("decoder"),
        );
        let mut ahead = Ahead::new(NonZeroUsize::new(2).expect("depth"));
        let time = fixture.frames[0].pts;
        let matrix = Matrix::for_size(fixture.config.width, fixture.config.height);
        ahead
            .seek_to(&mut session, &fixture.packets, time)
            .expect("seek");
        let frame = ahead.at(time).expect("frame");
        let (pixels, width, height) = super::rgba_of(&gpu, &frame.image, matrix).expect("rgba");
        assert!(width > 0 && height > 0);
        let pixels_len =
            usize::try_from(width).expect("width") * usize::try_from(height).expect("height") * 4;
        assert_eq!(pixels.len(), pixels_len);
    }
}
