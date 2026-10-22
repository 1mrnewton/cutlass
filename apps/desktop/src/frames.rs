//! The picture under the playhead.
//!
//! On Apple a worker decodes VideoToolbox frames ahead of the playhead. The
//! window samples the GPU texture for the frame that is due and does not copy
//! it back. A seek keeps the last picture until the new one is ready. Where no
//! video decoder is linked, there is no picture.

use cutlass_compose::Gpu;
use cutlass_project::Project;

#[cfg(any(test, target_vendor = "apple"))]
use cutlass_core::Timestamp;
#[cfg(target_vendor = "apple")]
use cutlass_project::AssetId;
#[cfg(target_vendor = "apple")]
use std::path::Path;

#[cfg(target_vendor = "apple")]
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
#[cfg(target_vendor = "apple")]
use std::thread::JoinHandle;

#[cfg(target_vendor = "apple")]
use cutlass_color::{Matrix, YuvPlanes};
#[cfg(target_vendor = "apple")]
use cutlass_core::Packet;
#[cfg(target_vendor = "apple")]
use cutlass_media::{Content, Probe, VideoSession};
#[cfg(target_vendor = "apple")]
use cutlass_playback::{Ahead, Decode};
#[cfg(target_vendor = "apple")]
use cutlass_videotoolbox::PixelBuffer;

/// How many decoded frames the worker keeps, including the one on screen.
#[cfg(target_vendor = "apple")]
const QUEUE_DEPTH: usize = 8;

/// A forward jump longer than this seeks from a keyframe instead of decoding through.
#[cfg(any(test, target_vendor = "apple"))]
const SEEK_GAP_SECS: f64 = 1.0;

/// The open clip and the GPU the window already created.
pub struct Reel {
    gpu: Option<Gpu>,
    #[cfg(target_vendor = "apple")]
    bridge: Option<Bridge>,
    /// GPU picture the preview samples. The texture stays here until the next frame.
    #[cfg(target_vendor = "apple")]
    plate: Option<cutlass_compose::RgbImage>,
    /// egui id for `plate`. The renderer does not own the wgpu texture.
    #[cfg(target_vendor = "apple")]
    native: Option<eframe::egui::TextureId>,
    /// CPU fallback, used when the frame has no surface the GPU can sample.
    #[cfg(target_vendor = "apple")]
    texture: Option<eframe::egui::TextureHandle>,
    #[cfg(target_vendor = "apple")]
    pending: Option<Ask>,
    #[cfg(target_vendor = "apple")]
    shown: Option<Shown>,
    #[cfg(target_vendor = "apple")]
    waiting: bool,
    #[cfg(target_vendor = "apple")]
    generation: u64,
}

impl Reel {
    /// `gpu` is the window's device. Without one, the plate stays the background.
    pub fn new(gpu: Option<Gpu>) -> Self {
        Self {
            #[cfg(target_vendor = "apple")]
            bridge: gpu.is_some().then(Bridge::start),
            gpu,
            #[cfg(target_vendor = "apple")]
            plate: None,
            #[cfg(target_vendor = "apple")]
            native: None,
            #[cfg(target_vendor = "apple")]
            texture: None,
            #[cfg(target_vendor = "apple")]
            pending: None,
            #[cfg(target_vendor = "apple")]
            shown: None,
            #[cfg(target_vendor = "apple")]
            waiting: false,
            #[cfg(target_vendor = "apple")]
            generation: 0,
        }
    }

    /// Drops the open clip. The GPU stays.
    pub fn clear(&mut self) {
        #[cfg(target_vendor = "apple")]
        {
            self.pending = None;
            self.shown = None;
            self.plate = None;
            self.texture = None;
            self.waiting = false;
            self.generation = self.generation.wrapping_add(1);
            if let Some(bridge) = &self.bridge {
                bridge.post(Message::Drop);
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

    /// The texture for the top picture at `playhead_us`, when one can be drawn.
    pub fn image(
        &mut self,
        ctx: &eframe::egui::Context,
        state: Option<&eframe::egui_wgpu::RenderState>,
        project: &Project,
        playhead_us: i64,
        hidden: &[cutlass_project::TrackId],
    ) -> Option<eframe::egui::TextureId> {
        self.gpu.as_ref()?;
        #[cfg(target_vendor = "apple")]
        {
            self.image_apple(ctx, state, project, playhead_us, hidden)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (ctx, state, project, playhead_us, hidden);
            None
        }
    }

    #[cfg(target_vendor = "apple")]
    fn image_apple(
        &mut self,
        ctx: &eframe::egui::Context,
        state: Option<&eframe::egui_wgpu::RenderState>,
        project: &Project,
        playhead_us: i64,
        hidden: &[cutlass_project::TrackId],
    ) -> Option<eframe::egui::TextureId> {
        let Some(segment) = crate::preview::picture_segment(project, playhead_us, hidden) else {
            self.hide(state);
            return None;
        };
        let Some(asset) = segment.asset_id() else {
            self.hide(state);
            return None;
        };
        let Some(time) = crate::preview::media_time(segment, playhead_us) else {
            self.hide(state);
            return None;
        };
        let Some(source) = project.asset(asset).map(|asset| asset.source()) else {
            self.hide(state);
            return None;
        };
        self.ask(asset, source, time);
        self.deliver(ctx, state);
        if self.plate.is_some() {
            return self.native;
        }
        self.texture.as_ref().map(|texture| texture.id())
    }

    #[cfg(target_vendor = "apple")]
    fn hide(&mut self, state: Option<&eframe::egui_wgpu::RenderState>) {
        self.pending = None;
        self.shown = None;
        self.plate = None;
        self.texture = None;
        self.free_native(state);
        self.waiting = false;
    }

    #[cfg(target_vendor = "apple")]
    fn ask(&mut self, asset: AssetId, source: &str, time: Timestamp) {
        if self.pending.as_ref().is_some_and(|pending| {
            pending.asset == asset && pending.source == source && pending.time == time
        }) {
            return;
        }
        let same_clip = self
            .pending
            .as_ref()
            .is_some_and(|pending| pending.asset == asset && pending.source == source);
        if !same_clip {
            self.generation = self.generation.wrapping_add(1);
            self.plate = None;
            self.texture = None;
            self.shown = None;
        }
        let ask = Ask {
            generation: self.generation,
            asset,
            source: source.to_owned(),
            time,
        };
        let posted = self
            .bridge
            .as_ref()
            .is_some_and(|bridge| bridge.post(Message::Show(ask.clone())));
        self.pending = Some(ask);
        self.waiting = posted
            && self
                .shown
                .as_ref()
                .is_none_or(|shown| shown.requested != time);
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
        let Some(picture) = self.bridge.as_ref().and_then(Bridge::take) else {
            return;
        };
        let Some(pending) = self.pending.clone() else {
            return;
        };
        if picture.generation != pending.generation || picture.requested != pending.time {
            return;
        }
        let Body::Frame {
            buffer,
            matrix,
            pts,
        } = picture.body
        else {
            self.waiting = false;
            return;
        };
        if self.has_picture() && self.shown.as_ref().is_some_and(|shown| shown.pts == pts) {
            if let Some(shown) = &mut self.shown {
                shown.requested = pending.time;
            }
            self.waiting = false;
            return;
        }
        let drawn = {
            let Some(gpu) = &self.gpu else {
                return;
            };
            match gpu.queue_pixel_buffer(&buffer, matrix) {
                Ok(image) => Drawn::Gpu(Box::new(image)),
                Err(_) => match rgba_of(gpu, &buffer, matrix) {
                    Some((pixels, width, height)) => Drawn::Cpu(pixels, width, height),
                    None => Drawn::Failed,
                },
            }
        };
        match drawn {
            Drawn::Gpu(image) => {
                self.bind(state, &image);
                self.texture = None;
                self.plate = Some(*image);
            }
            Drawn::Cpu(pixels, width, height) => {
                if self.store(ctx, &pixels, width, height).is_none() {
                    self.pending = None;
                    self.waiting = true;
                    return;
                }
                self.plate = None;
                self.free_native(Some(state));
            }
            Drawn::Failed => {
                self.pending = None;
                self.waiting = true;
                return;
            }
        }
        self.shown = Some(Shown {
            pts,
            requested: pending.time,
        });
        self.waiting = false;
    }

    #[cfg(target_vendor = "apple")]
    fn has_picture(&self) -> bool {
        self.plate.is_some() || self.texture.is_some()
    }

    /// Points the preview at `image`. The wgpu texture stays owned by the reel.
    #[cfg(target_vendor = "apple")]
    fn bind(&mut self, state: &eframe::egui_wgpu::RenderState, image: &cutlass_compose::RgbImage) {
        let view = image
            .texture()
            .create_view(&eframe::wgpu::TextureViewDescriptor::default());
        let mut renderer = state.renderer.write();
        if let Some(id) = self.native {
            renderer.update_egui_texture_from_wgpu_texture(
                &state.device,
                &view,
                eframe::wgpu::FilterMode::Linear,
                id,
            );
        } else {
            self.native = Some(renderer.register_native_texture(
                &state.device,
                &view,
                eframe::wgpu::FilterMode::Linear,
            ));
        }
    }

    #[cfg(target_vendor = "apple")]
    fn free_native(&mut self, state: Option<&eframe::egui_wgpu::RenderState>) {
        let Some(id) = self.native.take() else {
            return;
        };
        let Some(state) = state else {
            self.native = Some(id);
            return;
        };
        state.renderer.write().free_texture(&id);
    }

    #[cfg(target_vendor = "apple")]
    fn store(
        &mut self,
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
        if let Some(texture) = &mut self.texture {
            texture.set(image, options);
        } else {
            self.texture = Some(ctx.load_texture("preview-frame", image, options));
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
struct Open {
    asset: AssetId,
    source: String,
    packets: Vec<Packet>,
    matrix: Matrix,
    session: Session,
    ahead: Ahead<PixelBuffer>,
    cursor: usize,
    followed: Option<Timestamp>,
    failed: Option<Timestamp>,
}

#[cfg(target_vendor = "apple")]
enum Clip {
    Missing,
    Ready(Box<Open>),
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
        return Clip::Missing;
    };
    let Ok(bytes) = std::fs::read(Path::new(path)) else {
        return Clip::Missing;
    };
    let Ok(probe) = Probe::parse(&bytes) else {
        return Clip::Missing;
    };
    let Some(index) = probe
        .tracks()
        .iter()
        .position(|track| track.enabled && matches!(track.content, Content::Video { .. }))
    else {
        return Clip::Missing;
    };
    let Content::Video {
        config,
        color,
        mastering,
        content_light,
    } = &probe.tracks()[index].content
    else {
        return Clip::Missing;
    };
    let Ok(matrix) = Matrix::resolve(
        color.as_ref().map(|tags| tags.matrix),
        config.width,
        config.height,
    ) else {
        return Clip::Missing;
    };
    let rotation = probe.tracks()[index].rotation;
    let Ok(session) = VideoSession::open(config, *color, *mastering, *content_light, rotation)
    else {
        return Clip::Missing;
    };
    let Some(packets) = probe.samples(index).map(<[Packet]>::to_vec) else {
        return Clip::Missing;
    };
    let depth = std::num::NonZeroUsize::new(QUEUE_DEPTH).expect("the queue holds a frame");
    Clip::Ready(Box::new(Open {
        asset,
        source: source.to_owned(),
        packets,
        matrix,
        session: Session(session),
        ahead: Ahead::new(depth),
        cursor: 0,
        followed: None,
        failed: None,
    }))
}

#[cfg(target_vendor = "apple")]
struct Bridge {
    ask: Arc<Note<Message>>,
    picture: Arc<Slot<Picture>>,
    join: Option<JoinHandle<()>>,
}

#[cfg(target_vendor = "apple")]
impl Bridge {
    fn start() -> Self {
        let ask = Arc::new(Note::new());
        let picture = Arc::new(Slot::new());
        let ask_thread = Arc::clone(&ask);
        let picture_thread = Arc::clone(&picture);
        let join = std::thread::Builder::new()
            .name("cutlass-playback".to_owned())
            .spawn(move || play_loop(ask_thread, picture_thread))
            .ok();
        Self { ask, picture, join }
    }

    fn post(&self, message: Message) -> bool {
        if self.join.is_none() {
            return false;
        }
        self.ask.post(message);
        true
    }

    fn take(&self) -> Option<Picture> {
        self.picture.take()
    }
}

#[cfg(target_vendor = "apple")]
impl Drop for Bridge {
    fn drop(&mut self) {
        self.ask.close();
        if let Some(handle) = self.join.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(target_vendor = "apple")]
struct Note<T> {
    state: Mutex<NoteState<T>>,
    wake: Condvar,
}

#[cfg(target_vendor = "apple")]
struct NoteState<T> {
    value: Option<T>,
    closed: bool,
}

#[cfg(target_vendor = "apple")]
impl<T> Note<T> {
    fn new() -> Self {
        Self {
            state: Mutex::new(NoteState {
                value: None,
                closed: false,
            }),
            wake: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, NoteState<T>> {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn post(&self, value: T) {
        let mut state = self.lock();
        if state.closed {
            return;
        }
        state.value = Some(value);
        self.wake.notify_one();
    }

    fn close(&self) {
        let mut state = self.lock();
        state.closed = true;
        state.value = None;
        self.wake.notify_one();
    }

    fn wait(&self) -> Option<T> {
        let mut state = self.lock();
        loop {
            if let Some(value) = state.value.take() {
                return Some(value);
            }
            if state.closed {
                return None;
            }
            state = self
                .wake
                .wait(state)
                .unwrap_or_else(|poison| poison.into_inner());
        }
    }
}

#[cfg(target_vendor = "apple")]
struct Slot<T> {
    inner: Mutex<Option<T>>,
}

#[cfg(target_vendor = "apple")]
impl<T> Slot<T> {
    fn new() -> Self {
        Self {
            inner: Mutex::new(None),
        }
    }

    fn publish(&self, value: T) {
        *self.lock() = Some(value);
    }

    fn take(&self) -> Option<T> {
        self.lock().take()
    }

    fn lock(&self) -> MutexGuard<'_, Option<T>> {
        self.inner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }
}

#[cfg(target_vendor = "apple")]
enum Message {
    Show(Ask),
    Drop,
}

#[cfg(target_vendor = "apple")]
#[derive(Clone)]
struct Ask {
    generation: u64,
    asset: AssetId,
    source: String,
    time: Timestamp,
}

#[cfg(target_vendor = "apple")]
struct Picture {
    generation: u64,
    requested: Timestamp,
    body: Body,
}

#[cfg(target_vendor = "apple")]
enum Body {
    Frame {
        buffer: PixelBuffer,
        matrix: Matrix,
        pts: Timestamp,
    },
    Unavailable,
}

#[cfg(target_vendor = "apple")]
fn play_loop(ask: Arc<Note<Message>>, picture: Arc<Slot<Picture>>) {
    let mut open = None;
    while let Some(message) = ask.wait() {
        match message {
            Message::Drop => open = None,
            Message::Show(show) => serve(&mut open, &picture, show),
        }
    }
}

#[cfg(target_vendor = "apple")]
fn serve(open: &mut Option<Open>, picture: &Slot<Picture>, show: Ask) {
    if open
        .as_ref()
        .is_none_or(|clip| clip.asset != show.asset || clip.source != show.source)
    {
        *open = ready(show.asset, &show.source);
    }
    let Some(clip) = open.as_mut() else {
        picture.publish(unavailable(&show));
        return;
    };
    if clip.failed == Some(show.time) {
        picture.publish(unavailable(&show));
        return;
    }
    if pump(clip, show.time).is_err() {
        clip.failed = Some(show.time);
        picture.publish(unavailable(&show));
        return;
    }
    let taken = clip
        .ahead
        .at(show.time)
        .map(|frame| (frame.pts, frame.image.clone()));
    let Some((pts, buffer)) = taken else {
        clip.failed = Some(show.time);
        picture.publish(unavailable(&show));
        return;
    };
    clip.failed = None;
    picture.publish(Picture {
        generation: show.generation,
        requested: show.time,
        body: Body::Frame {
            buffer,
            matrix: clip.matrix,
            pts,
        },
    });
}

#[cfg(target_vendor = "apple")]
fn unavailable(show: &Ask) -> Picture {
    Picture {
        generation: show.generation,
        requested: show.time,
        body: Body::Unavailable,
    }
}

#[cfg(target_vendor = "apple")]
fn ready(asset: AssetId, source: &str) -> Option<Open> {
    match open_clip(asset, source) {
        Clip::Ready(open) => Some(*open),
        Clip::Missing => None,
    }
}

#[cfg(target_vendor = "apple")]
fn pump(clip: &mut Open, time: Timestamp) -> Result<(), ()> {
    let jump = jumps(clip.followed, time) || clip.ahead.is_empty();
    if jump {
        let rest = clip
            .ahead
            .seek_to(&mut clip.session, &clip.packets, time)
            .map_err(|_| ())?;
        clip.cursor = clip.packets.len() - rest.len();
    } else {
        let _ = clip.ahead.at(time);
        let start = clip.cursor.min(clip.packets.len());
        let mut rest = &clip.packets[start..];
        clip.ahead
            .fill(&mut clip.session, &mut rest)
            .map_err(|_| ())?;
        clip.cursor = clip.packets.len() - rest.len();
    }
    clip.followed = Some(time);
    Ok(())
}

#[cfg(test)]
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

/// `true` when `time` is behind the last request, or more than a second ahead of it.
#[cfg(any(test, target_vendor = "apple"))]
fn jumps(followed: Option<Timestamp>, time: Timestamp) -> bool {
    let Some(prev) = followed else {
        return true;
    };
    if time < prev {
        return true;
    }
    time.as_secs_f64() - prev.as_secs_f64() > SEEK_GAP_SECS
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

    #[test]
    fn a_backward_or_long_jump_seeks() {
        use std::num::NonZeroU32;

        let scale = NonZeroU32::new(1_000).expect("scale");
        let at = |ms| super::Timestamp::new(ms, scale);
        assert!(super::jumps(None, at(0)));
        assert!(super::jumps(Some(at(1_000)), at(0)));
        assert!(!super::jumps(Some(at(0)), at(500)));
        assert!(super::jumps(Some(at(0)), at(2_000)));
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn a_fixture_frame_becomes_packed_rgba() {
        use std::num::NonZeroUsize;

        use cutlass_color::Matrix;
        use cutlass_compose::{Error, Gpu};
        use cutlass_core::Rotation;
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
        let mut session = Session(
            VideoSession::open(&fixture.config, None, None, None, Rotation::Zero).expect("decoder"),
        );
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
