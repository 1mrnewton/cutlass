//! The picture under the playhead.
//!
//! On Apple a worker decodes VideoToolbox frames ahead of the playhead. It
//! keeps the decoder inside the current group of pictures, skips to a keyframe
//! the clock has reached, and drops frames that are already late. The next
//! clip opens on a side thread during the last second before the cut. The
//! window samples the GPU texture for the frame that is due and does not copy
//! it back. A failure is written to the app log. A seek, and a cut, keep the
//! last picture until the new one is ready. Where no video decoder is linked,
//! there is no picture.

use cutlass_compose::Gpu;
use cutlass_project::Project;

#[cfg(target_vendor = "apple")]
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
use cutlass_playback::{Ahead, Decode, Reach};
#[cfg(target_vendor = "apple")]
use cutlass_videotoolbox::PixelBuffer;

/// How many decoded frames the worker keeps, including the one on screen.
#[cfg(target_vendor = "apple")]
const QUEUE_DEPTH: usize = 8;

/// How many new frames one pass decodes before it looks at the playhead again.
#[cfg(target_vendor = "apple")]
const DECODE_BUDGET: usize = QUEUE_DEPTH;

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
    /// The GPU import already failed for this run. The next frames stay quiet.
    #[cfg(target_vendor = "apple")]
    copied: bool,
    /// A dropped frame was already written. The next drops stay quiet.
    #[cfg(target_vendor = "apple")]
    dropped: bool,
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
            self.shown = None;
            self.plate = None;
            self.texture = None;
            self.waiting = false;
            self.copied = false;
            self.dropped = false;
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
        let next = crate::preview::upcoming_picture(project, playhead_us, hidden).map(
            |(asset, source)| Next {
                asset,
                source: source.to_owned(),
            },
        );
        self.ask(asset, source, time, next);
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
    fn ask(&mut self, asset: AssetId, source: &str, time: Timestamp, next: Option<Next>) {
        if self.pending.as_ref().is_some_and(|pending| {
            pending.asset == asset
                && pending.source == source
                && pending.time == time
                && pending.next == next
        }) {
            return;
        }
        let same_clip = self
            .pending
            .as_ref()
            .is_some_and(|pending| pending.asset == asset && pending.source == source);
        if !same_clip {
            self.generation = self.generation.wrapping_add(1);
            // The last frame stays until this clip's picture is imported.
            // Dropping it here paints the empty plate across the cut.
            self.shown = None;
        }
        let ask = Ask {
            generation: self.generation,
            asset,
            source: source.to_owned(),
            time,
            next,
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
        if picture.generation != pending.generation {
            return;
        }
        let Body::Frame {
            buffer,
            matrix,
            pts,
        } = picture.body
        else {
            if picture.requested == pending.time {
                self.plate = None;
                self.texture = None;
                self.shown = None;
                self.free_native(Some(state));
                self.waiting = false;
            }
            return;
        };
        // The playhead moved forward. This frame still covers it. A backward
        // seek is waiting for a different frame, so this one stays off the plate.
        if picture.requested != pending.time && pts > pending.time {
            return;
        }
        if self.has_picture() && self.shown.as_ref().is_some_and(|shown| shown.pts == pts) {
            if let Some(shown) = &mut self.shown {
                shown.requested = pending.time;
            }
            self.waiting = false;
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
                self.bind(state, &image);
                self.texture = None;
                self.plate = Some(*image);
            }
            Drawn::Cpu(pixels, width, height) => {
                if self.store(ctx, &pixels, width, height).is_none() {
                    if !self.dropped {
                        log::warn!("The preview dropped a frame. The CPU copy could not be shown.");
                        self.dropped = true;
                    }
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
        log::warn!("The preview skipped {source}: it is not a file.");
        return Clip::Missing;
    };
    let bytes = match std::fs::read(Path::new(path)) {
        Ok(bytes) => bytes,
        Err(error) => {
            log::warn!("The preview could not read {path}: {error}.");
            return Clip::Missing;
        }
    };
    let probe = match Probe::parse(&bytes) {
        Ok(probe) => probe,
        Err(error) => {
            log::warn!("The preview could not open {path}: {error}.");
            return Clip::Missing;
        }
    };
    let Some(index) = probe
        .tracks()
        .iter()
        .position(|track| track.enabled && matches!(track.content, Content::Video { .. }))
    else {
        log::warn!("The preview found no picture track in {path}.");
        return Clip::Missing;
    };
    let Content::Video {
        config,
        color,
        mastering,
        content_light,
    } = &probe.tracks()[index].content
    else {
        log::warn!("The preview found no picture track in {path}.");
        return Clip::Missing;
    };
    let matrix = match Matrix::resolve(
        color.as_ref().map(|tags| tags.matrix),
        config.width,
        config.height,
    ) {
        Ok(matrix) => matrix,
        Err(error) => {
            log::warn!("The preview could not resolve the color of {path}: {error}.");
            return Clip::Missing;
        }
    };
    let rotation = probe.tracks()[index].rotation;
    let session = match VideoSession::open(config, *color, *mastering, *content_light, rotation) {
        Ok(session) => session,
        Err(error) => {
            log::warn!("The preview could not decode {path}: {error}.");
            return Clip::Missing;
        }
    };
    let Some(packets) = probe.samples(index).map(<[Packet]>::to_vec) else {
        log::warn!("The preview found no picture samples in {path}.");
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

    /// The newest message, when one is waiting. Does not block.
    fn poll(&self) -> Inbox<T> {
        let mut state = self.lock();
        if let Some(value) = state.value.take() {
            return Inbox::Message(value);
        }
        if state.closed {
            Inbox::Closed
        } else {
            Inbox::Empty
        }
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
enum Inbox<T> {
    Message(T),
    Empty,
    Closed,
}

/// What one decode pass did for the requested time.
#[cfg(target_vendor = "apple")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum Chunk {
    /// A frame at or before the request is queued.
    Covers,
    /// More packets can still produce that frame.
    Short,
    /// Decoding will not produce a frame at or before the request.
    Settled,
}

#[cfg(target_vendor = "apple")]
#[derive(Clone)]
struct Ask {
    generation: u64,
    asset: AssetId,
    source: String,
    time: Timestamp,
    next: Option<Next>,
}

/// The picture that starts within a second of the playhead.
#[cfg(target_vendor = "apple")]
#[derive(Clone, PartialEq, Eq)]
struct Next {
    asset: AssetId,
    source: String,
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
    let mut missed = None;
    let mut warm = None;
    while let Some(message) = ask.wait() {
        match message {
            Message::Drop => release(&mut open, &mut missed, &mut warm),
            Message::Show(show) => {
                if !follow_show(&ask, &picture, &mut open, &mut missed, &mut warm, show) {
                    break;
                }
            }
        }
    }
    join_warm(&mut warm);
}

#[cfg(target_vendor = "apple")]
fn follow_show(
    ask: &Note<Message>,
    picture: &Slot<Picture>,
    open: &mut Option<Open>,
    missed: &mut Option<String>,
    warm: &mut Option<Warm>,
    mut show: Ask,
) -> bool {
    loop {
        let chunk = serve(open, missed, warm, picture, &show);
        if chunk == Chunk::Covers {
            warm_next(warm, open, show.next.as_ref());
        }
        match ask.poll() {
            Inbox::Closed => return false,
            Inbox::Message(Message::Drop) => {
                release(open, missed, warm);
                return true;
            }
            Inbox::Message(Message::Show(next)) => show = next,
            Inbox::Empty => {
                if chunk == Chunk::Short {
                    continue;
                }
                return true;
            }
        }
    }
}

#[cfg(target_vendor = "apple")]
fn release(open: &mut Option<Open>, missed: &mut Option<String>, warm: &mut Option<Warm>) {
    *open = None;
    *missed = None;
    join_warm(warm);
}

#[cfg(target_vendor = "apple")]
fn serve(
    open: &mut Option<Open>,
    missed: &mut Option<String>,
    warm: &mut Option<Warm>,
    picture: &Slot<Picture>,
    show: &Ask,
) -> Chunk {
    let switched = open
        .as_ref()
        .is_none_or(|clip| clip.asset != show.asset || clip.source != show.source);
    if switched {
        match claim(warm, show.asset, &show.source) {
            Claimed::Ready(clip) => {
                *open = Some(*clip);
                missed.take();
            }
            Claimed::Missing => {
                *open = None;
                *missed = Some(show.source.clone());
            }
            Claimed::Elsewhere => {
                if missed.as_deref() == Some(show.source.as_str()) {
                    *open = None;
                } else {
                    *open = ready(show.asset, &show.source);
                    if open.is_none() {
                        *missed = Some(show.source.clone());
                    } else {
                        missed.take();
                    }
                }
            }
        }
    }
    let Some(clip) = open.as_mut() else {
        picture.publish(unavailable(show));
        return Chunk::Settled;
    };
    if clip.failed == Some(show.time) {
        picture.publish(unavailable(show));
        return Chunk::Settled;
    }
    let readable = match pump(clip, show.time) {
        Ok(readable) => readable,
        Err(error) => {
            if clip.failed.is_none() {
                log::warn!("The preview could not decode {}: {error}.", clip.source);
            }
            clip.failed = Some(show.time);
            picture.publish(unavailable(show));
            return Chunk::Settled;
        }
    };
    if !readable {
        if clip.failed.is_none() {
            log::warn!("The preview has no frame at this time in {}.", clip.source);
        }
        clip.failed = Some(show.time);
        picture.publish(unavailable(show));
        return Chunk::Settled;
    }
    let taken = clip
        .ahead
        .at(show.time)
        .map(|frame| (frame.pts, frame.image.clone()));
    let Some((pts, buffer)) = taken else {
        let finished =
            clip.ahead.is_full() || (clip.cursor >= clip.packets.len() && clip.ahead.drained());
        if !finished {
            return Chunk::Short;
        }
        if clip.ahead.is_empty() {
            if clip.failed.is_none() {
                log::warn!("The preview has no frame at this time in {}.", clip.source);
            }
            clip.failed = Some(show.time);
            picture.publish(unavailable(show));
        }
        return Chunk::Settled;
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
    Chunk::Covers
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
fn pump(clip: &mut Open, time: Timestamp) -> Result<bool, String> {
    let (start, restart) =
        match cutlass_playback::reach(&clip.packets, clip.cursor, clip.followed, time) {
            None => {
                clip.followed = Some(time);
                return Ok(false);
            }
            Some(Reach::Keyframe(index)) => (index, true),
            Some(Reach::Forward) => (clip.cursor.min(clip.packets.len()), false),
        };
    let mut pending = &clip.packets[start..];
    clip.ahead
        .follow(
            &mut clip.session,
            &mut pending,
            time,
            DECODE_BUDGET,
            restart,
        )
        .map_err(|error| error.to_string())?;
    clip.cursor = clip.packets.len() - pending.len();
    clip.followed = Some(time);
    Ok(true)
}

/// A clip opened before the playhead reaches it.
#[cfg(target_vendor = "apple")]
struct Warm {
    asset: AssetId,
    source: String,
    join: JoinHandle<Clip>,
}

#[cfg(target_vendor = "apple")]
enum Claimed {
    Elsewhere,
    Missing,
    Ready(Box<Open>),
}

#[cfg(target_vendor = "apple")]
fn claim(warm: &mut Option<Warm>, asset: AssetId, source: &str) -> Claimed {
    let ready = warm
        .as_ref()
        .is_some_and(|warm| warm.asset == asset && warm.source == source);
    if !ready {
        return Claimed::Elsewhere;
    }
    let Some(warm) = warm.take() else {
        return Claimed::Elsewhere;
    };
    match warm.join.join() {
        Ok(Clip::Ready(open)) => Claimed::Ready(open),
        _ => Claimed::Missing,
    }
}

#[cfg(target_vendor = "apple")]
fn warm_next(warm: &mut Option<Warm>, open: &Option<Open>, next: Option<&Next>) {
    let Some(next) = next else {
        return;
    };
    if open
        .as_ref()
        .is_some_and(|clip| clip.asset == next.asset && clip.source == next.source)
    {
        return;
    }
    if let Some(current) = warm.as_ref() {
        if current.asset == next.asset && current.source == next.source {
            return;
        }
        if !current.join.is_finished() {
            return;
        }
    }
    if let Some(done) = warm.take() {
        let _ = done.join.join();
    }
    let asset = next.asset;
    let source = next.source.clone();
    let spawned = std::thread::Builder::new()
        .name("cutlass-preload".to_owned())
        .spawn(move || open_clip(asset, &source));
    match spawned {
        Ok(join) => {
            *warm = Some(Warm {
                asset: next.asset,
                source: next.source.clone(),
                join,
            });
        }
        Err(error) => log::warn!("The preview could not prepare the next clip: {error}."),
    }
}

#[cfg(target_vendor = "apple")]
fn join_warm(warm: &mut Option<Warm>) {
    if let Some(warm) = warm.take() {
        let _ = warm.join.join();
    }
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
