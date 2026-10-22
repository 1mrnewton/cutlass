//! The picture under the playhead.
//!
//! On Apple the top picture is decoded and converted on the window's GPU, then
//! painted in its box. A resting playhead keeps that picture. A new time
//! decodes on the UI thread, so moving the playhead can wait. Where no video
//! decoder is linked, there is no picture. `docs/preview.md`.

use std::path::Path;

use cutlass_compose::Gpu;
use cutlass_project::{AssetId, Project};

#[cfg(target_vendor = "apple")]
use cutlass_color::{Matrix, YuvPlanes};
#[cfg(target_vendor = "apple")]
use cutlass_core::{Packet, Timestamp};
#[cfg(target_vendor = "apple")]
use cutlass_media::{Content, Probe, VideoSession};
#[cfg(target_vendor = "apple")]
use cutlass_playback::{Ahead, Decode};
#[cfg(target_vendor = "apple")]
use cutlass_videotoolbox::PixelBuffer;

/// The open clip and the GPU the window already created.
pub struct Reel {
    gpu: Option<Gpu>,
    #[cfg(target_vendor = "apple")]
    clip: Option<Clip>,
    #[cfg(target_vendor = "apple")]
    texture: Option<eframe::egui::TextureHandle>,
}

impl Reel {
    /// `gpu` is the window's device. Without one, the plate stays the background.
    pub fn new(gpu: Option<Gpu>) -> Self {
        Self {
            gpu,
            #[cfg(target_vendor = "apple")]
            clip: None,
            #[cfg(target_vendor = "apple")]
            texture: None,
        }
    }

    /// Drops the open clip. The GPU stays.
    pub fn clear(&mut self) {
        #[cfg(target_vendor = "apple")]
        {
            self.clip = None;
            self.texture = None;
        }
    }

    /// The texture for the top picture at `playhead_us`, when one can be drawn.
    pub fn image(
        &mut self,
        ctx: &eframe::egui::Context,
        project: &Project,
        playhead_us: i64,
        hidden: &[cutlass_project::TrackId],
    ) -> Option<eframe::egui::TextureId> {
        self.gpu.as_ref()?;
        #[cfg(target_vendor = "apple")]
        {
            self.image_apple(ctx, project, playhead_us, hidden)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (ctx, project, playhead_us, hidden);
            None
        }
    }

    #[cfg(target_vendor = "apple")]
    fn image_apple(
        &mut self,
        ctx: &eframe::egui::Context,
        project: &Project,
        playhead_us: i64,
        hidden: &[cutlass_project::TrackId],
    ) -> Option<eframe::egui::TextureId> {
        let segment = crate::preview::picture_segment(project, playhead_us, hidden)?;
        let asset_id = segment.asset_id()?;
        let time = crate::preview::media_time(segment, playhead_us)?;
        let source = project.asset(asset_id)?.source();
        if !self.same(asset_id, source) {
            let owned = source.to_owned();
            self.prepare(asset_id, &owned);
        }
        if !self.ready() {
            return None;
        }
        self.frame(ctx, time)
    }

    #[cfg(target_vendor = "apple")]
    fn same(&self, asset: AssetId, source: &str) -> bool {
        self.clip
            .as_ref()
            .is_some_and(|clip| clip.asset() == asset && clip.source() == source)
    }

    #[cfg(target_vendor = "apple")]
    fn ready(&self) -> bool {
        self.clip.as_ref().is_some_and(Clip::is_ready)
    }

    #[cfg(target_vendor = "apple")]
    fn prepare(&mut self, asset: AssetId, source: &str) {
        self.texture = None;
        self.clip = Some(open_clip(asset, source));
    }

    #[cfg(target_vendor = "apple")]
    fn frame(
        &mut self,
        ctx: &eframe::egui::Context,
        time: Timestamp,
    ) -> Option<eframe::egui::TextureId> {
        let key = (time.value, time.timescale.get());
        if self.cached(key) {
            return self.texture.as_ref().map(|texture| texture.id());
        }
        let (pixels, width, height) = self.decode(time)?;
        self.store(ctx, key, &pixels, width, height)
    }

    #[cfg(target_vendor = "apple")]
    fn cached(&self, key: (i64, u32)) -> bool {
        let Some(Clip::Ready(open)) = &self.clip else {
            return false;
        };
        let Some(shown) = &open.shown else {
            return false;
        };
        shown.value == key.0 && shown.scale == key.1 && self.texture.is_some()
    }

    #[cfg(target_vendor = "apple")]
    fn decode(&mut self, time: Timestamp) -> Option<(Vec<u8>, u32, u32)> {
        let key = (time.value, time.timescale.get());
        if self.failed_at(key) {
            return None;
        }
        let decoded = {
            let gpu = self.gpu.as_ref()?;
            let Clip::Ready(open) = self.clip.as_mut()? else {
                return None;
            };
            frame_rgba(
                gpu,
                &mut open.session,
                &mut open.ahead,
                &open.packets,
                time,
                open.matrix,
            )
        };
        if decoded.is_none() {
            self.note_failure(key);
        }
        decoded
    }

    #[cfg(target_vendor = "apple")]
    fn failed_at(&self, key: (i64, u32)) -> bool {
        matches!(&self.clip, Some(Clip::Ready(open)) if open.failed == Some(key))
    }

    #[cfg(target_vendor = "apple")]
    fn note_failure(&mut self, key: (i64, u32)) {
        if let Some(Clip::Ready(open)) = &mut self.clip {
            open.failed = Some(key);
        }
    }

    #[cfg(target_vendor = "apple")]
    fn store(
        &mut self,
        ctx: &eframe::egui::Context,
        key: (i64, u32),
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
        if let Some(texture) = &mut self.texture {
            texture.set(image, options);
        } else {
            self.texture = Some(ctx.load_texture("preview-frame", image, options));
        }
        if let Some(Clip::Ready(open)) = &mut self.clip {
            open.shown = Some(Shown {
                value: key.0,
                scale: key.1,
            });
            open.failed = None;
        }
        self.texture.as_ref().map(|texture| texture.id())
    }
}

/// A source string the shell can open. `file://` drops that prefix.
/// `content:` is not a file.
pub(crate) fn file_path(source: &str) -> Option<&str> {
    if let Some(rest) = source.strip_prefix("file://") {
        let rest = rest.strip_prefix("localhost").unwrap_or(rest);
        return (!rest.is_empty()).then_some(rest);
    }
    if source.starts_with("content:") || source.is_empty() {
        return None;
    }
    Some(source)
}

#[cfg(target_vendor = "apple")]
struct Shown {
    value: i64,
    scale: u32,
}

#[cfg(target_vendor = "apple")]
struct Open {
    asset: AssetId,
    source: String,
    packets: Vec<Packet>,
    matrix: Matrix,
    session: Session,
    ahead: Ahead<PixelBuffer>,
    shown: Option<Shown>,
    failed: Option<(i64, u32)>,
}

#[cfg(target_vendor = "apple")]
enum Clip {
    Missing { asset: AssetId, source: String },
    Ready(Box<Open>),
}

#[cfg(target_vendor = "apple")]
impl Clip {
    fn asset(&self) -> AssetId {
        match self {
            Self::Missing { asset, .. } => *asset,
            Self::Ready(open) => open.asset,
        }
    }

    fn source(&self) -> &str {
        match self {
            Self::Missing { source, .. } => source,
            Self::Ready(open) => &open.source,
        }
    }

    fn is_ready(&self) -> bool {
        matches!(self, Self::Ready(_))
    }
}

#[cfg(target_vendor = "apple")]
struct Session(VideoSession);

#[cfg(target_vendor = "apple")]
impl Decode for Session {
    type Image = PixelBuffer;
    type Error = cutlass_media::Error;

    fn send(&mut self, packet: &Packet) -> Result<(), Self::Error> {
        self.0.send_packet(packet)
    }

    fn receive(&mut self) -> Result<Option<cutlass_core::VideoFrame<PixelBuffer>>, Self::Error> {
        self.0.receive_frame()
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        self.0.flush()
    }

    fn reset(&mut self) -> Result<(), Self::Error> {
        self.0.reset()
    }
}

#[cfg(target_vendor = "apple")]
fn open_clip(asset: AssetId, source: &str) -> Clip {
    let Some(path) = file_path(source) else {
        return missing(asset, source);
    };
    let Ok(bytes) = std::fs::read(Path::new(path)) else {
        return missing(asset, source);
    };
    let Ok(probe) = Probe::parse(&bytes) else {
        return missing(asset, source);
    };
    let Some(index) = probe
        .tracks()
        .iter()
        .position(|track| track.enabled && matches!(track.content, Content::Video { .. }))
    else {
        return missing(asset, source);
    };
    let Content::Video { config, color } = &probe.tracks()[index].content else {
        return missing(asset, source);
    };
    let Ok(matrix) = Matrix::resolve(
        color.as_ref().map(|tags| tags.matrix),
        config.width,
        config.height,
    ) else {
        return missing(asset, source);
    };
    let Ok(session) = VideoSession::open(config) else {
        return missing(asset, source);
    };
    let Some(packets) = probe.samples(index).map(<[Packet]>::to_vec) else {
        return missing(asset, source);
    };
    let depth = std::num::NonZeroUsize::new(2).expect("two frames fit in the queue");
    Clip::Ready(Box::new(Open {
        asset,
        source: source.to_owned(),
        packets,
        matrix,
        session: Session(session),
        ahead: Ahead::new(depth),
        shown: None,
        failed: None,
    }))
}

#[cfg(target_vendor = "apple")]
fn missing(asset: AssetId, source: &str) -> Clip {
    Clip::Missing {
        asset,
        source: source.to_owned(),
    }
}

#[cfg(target_vendor = "apple")]
fn frame_rgba(
    gpu: &Gpu,
    session: &mut Session,
    ahead: &mut Ahead<PixelBuffer>,
    packets: &[Packet],
    time: Timestamp,
    matrix: Matrix,
) -> Option<(Vec<u8>, u32, u32)> {
    ahead.seek_to(session, packets, time).ok()?;
    let frame = ahead.at(time)?;
    rgba_of(gpu, &frame.image, matrix)
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
        use cutlass_media::VideoSession;
        use cutlass_playback::Ahead;

        use super::{Session, frame_rgba};

        let gpu = match Gpu::new() {
            Ok(gpu) => gpu,
            Err(Error::NoAdapter) => {
                eprintln!("skipping: no GPU adapter");
                return;
            }
            Err(error) => panic!("opening a GPU failed: {error}"),
        };
        let fixture = cutlass_test_fixtures::video("h264-high-bframes-128x72");
        let mut session = Session(VideoSession::open(&fixture.config).expect("decoder"));
        let mut ahead = Ahead::new(NonZeroUsize::new(2).expect("depth"));
        let time = fixture.frames[0].pts;
        let (pixels, width, height) = frame_rgba(
            &gpu,
            &mut session,
            &mut ahead,
            &fixture.packets,
            time,
            Matrix::for_size(fixture.config.width, fixture.config.height),
        )
        .expect("rgba");
        assert!(width > 0 && height > 0);
        let pixels_len =
            usize::try_from(width).expect("width") * usize::try_from(height).expect("height") * 4;
        assert_eq!(pixels.len(), pixels_len);
    }
}
