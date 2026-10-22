//! The pictures under the playhead.
//!
//! On Apple each video track has its own VideoToolbox reader and frame queue.
//! The reader stays inside the current group of pictures, skips to a keyframe
//! the clock has reached, and drops frames that are already late. The next
//! clip on a track opens while the hardware reader count is under the cap.
//! The window draws those frames from the bottom track upward and does not
//! copy them back. A failure is written to the app log. A seek, and a cut,
//! keep the last picture until the new one is ready. Where no video decoder
//! is linked, there is no picture.

use cutlass_compose::Gpu;
use cutlass_project::Project;

#[cfg(target_vendor = "apple")]
use cutlass_core::Timestamp;
#[cfg(target_vendor = "apple")]
use cutlass_project::{AssetId, TrackId};
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

/// CapCut's preload scheduler stops at `hw reader count max`. The number was
/// not recovered. A picture that is already on screen still gets a reader.
#[cfg(target_vendor = "apple")]
const HW_READER_CAP: usize = 8;

/// The open readers and the GPU the window already created.
pub struct Reel {
    gpu: Option<Gpu>,
    #[cfg(target_vendor = "apple")]
    bridge: Option<Bridge>,
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
            bridge: gpu.is_some().then(Bridge::start),
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

    /// Textures for the pictures at `playhead_us`, bottom track first.
    pub fn images(
        &mut self,
        ctx: &eframe::egui::Context,
        state: Option<&eframe::egui_wgpu::RenderState>,
        project: &Project,
        playhead_us: i64,
        hidden: &[cutlass_project::TrackId],
    ) -> Vec<crate::preview::DrawnPicture> {
        if self.gpu.is_none() {
            return Vec::new();
        }
        #[cfg(target_vendor = "apple")]
        {
            self.images_apple(ctx, state, project, playhead_us, hidden)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (ctx, state, project, playhead_us, hidden);
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
    ) -> Vec<crate::preview::DrawnPicture> {
        let layers = crate::preview::picture_layers(project, playhead_us, hidden);
        let next = crate::preview::upcoming_layers(project, playhead_us, hidden);
        self.retain(&layers, state);
        self.ask(project, playhead_us, &layers, &next);
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
        if self
            .pending
            .as_ref()
            .is_some_and(|pending| pending.layers == show && pending.next == next)
        {
            return;
        }
        let ask = Ask { layers: show, next };
        let posted = self
            .bridge
            .as_ref()
            .is_some_and(|bridge| bridge.post(Message::Show(ask.clone())));
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
        let Some(bridge) = &self.bridge else {
            return;
        };
        for frame in bridge.take_frames() {
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
        let Some(asked) = self.pending.as_ref().and_then(|pending| {
            pending
                .layers
                .iter()
                .find(|layer| layer.track == frame.track)
                .map(|layer| layer.time)
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
        // The playhead moved forward. This frame still covers it. A backward
        // seek is waiting for a different frame, so this one stays off the plate.
        if frame.requested != asked && pts > asked {
            return;
        }
        if self.held[index].has_picture()
            && self.held[index]
                .shown
                .as_ref()
                .is_some_and(|shown| shown.pts == pts)
        {
            if let Some(shown) = &mut self.held[index].shown {
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
            requested: asked,
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
fn open_clip(source: &str) -> Clip {
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
    board: Arc<Mutex<Vec<LayerFrame>>>,
    join: Option<JoinHandle<()>>,
}

#[cfg(target_vendor = "apple")]
impl Bridge {
    fn start() -> Self {
        let ask = Arc::new(Note::new());
        let board = Arc::new(Mutex::new(Vec::new()));
        let ask_thread = Arc::clone(&ask);
        let board_thread = Arc::clone(&board);
        let join = std::thread::Builder::new()
            .name("cutlass-playback".to_owned())
            .spawn(move || direct_loop(ask_thread, board_thread))
            .ok();
        Self { ask, board, join }
    }

    fn post(&self, message: Message) -> bool {
        if self.join.is_none() {
            return false;
        }
        self.ask.post(message);
        true
    }

    fn take_frames(&self) -> Vec<LayerFrame> {
        let mut board = self
            .board
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        std::mem::take(&mut *board)
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
#[derive(Clone, PartialEq, Eq)]
struct Ask {
    layers: Vec<ShowLayer>,
    next: Vec<ShowNext>,
}

#[cfg(target_vendor = "apple")]
#[derive(Clone, PartialEq, Eq)]
struct ShowLayer {
    track: TrackId,
    generation: u64,
    asset: AssetId,
    source: String,
    time: Timestamp,
}

/// A clip that starts within a second of the playhead.
#[cfg(target_vendor = "apple")]
#[derive(Clone, PartialEq, Eq)]
struct ShowNext {
    track: TrackId,
    asset: AssetId,
    source: String,
}

#[cfg(target_vendor = "apple")]
struct LayerFrame {
    track: TrackId,
    generation: u64,
    requested: Timestamp,
    body: Body,
}

#[cfg(target_vendor = "apple")]
enum ReaderMsg {
    At { generation: u64, time: Timestamp },
    Stop,
}

#[cfg(target_vendor = "apple")]
struct Reader {
    track: TrackId,
    asset: AssetId,
    source: String,
    note: Arc<Note<ReaderMsg>>,
    join: JoinHandle<()>,
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
fn direct_loop(ask: Arc<Note<Message>>, board: Arc<Mutex<Vec<LayerFrame>>>) {
    let mut readers = Vec::new();
    let mut open = 0;
    let mut delayed = false;
    while let Some(message) = ask.wait() {
        match message {
            Message::Drop => {
                stop_readers(&mut readers);
                note_readers(&mut open, 0);
                delayed = false;
            }
            Message::Show(show) => {
                reconcile(&board, &mut readers, &mut open, &mut delayed, &show);
            }
        }
    }
    stop_readers(&mut readers);
}

#[cfg(target_vendor = "apple")]
fn reconcile(
    board: &Arc<Mutex<Vec<LayerFrame>>>,
    readers: &mut Vec<Reader>,
    open: &mut usize,
    delayed: &mut bool,
    show: &Ask,
) {
    let mut index = 0;
    while index < readers.len() {
        let reader = &readers[index];
        let kept = show
            .layers
            .iter()
            .any(|layer| matches_reader(reader, layer.track, layer.asset, &layer.source))
            || show
                .next
                .iter()
                .any(|next| matches_reader(reader, next.track, next.asset, &next.source));
        if kept {
            index += 1;
        } else {
            let reader = readers.swap_remove(index);
            reader.note.post(ReaderMsg::Stop);
        }
    }
    for layer in &show.layers {
        if let Some(reader) = readers
            .iter()
            .find(|reader| matches_reader(reader, layer.track, layer.asset, &layer.source))
        {
            reader.note.post(ReaderMsg::At {
                generation: layer.generation,
                time: layer.time,
            });
            continue;
        }
        spawn_reader(
            readers,
            Arc::clone(board),
            layer.track,
            layer.asset,
            layer.source.clone(),
            Some((layer.generation, layer.time)),
        );
    }
    let mut skipped = false;
    for next in &show.next {
        if readers
            .iter()
            .any(|reader| matches_reader(reader, next.track, next.asset, &next.source))
        {
            continue;
        }
        if readers.len() >= HW_READER_CAP {
            skipped = true;
            continue;
        }
        spawn_reader(
            readers,
            Arc::clone(board),
            next.track,
            next.asset,
            next.source.clone(),
            None,
        );
    }
    if skipped && !*delayed {
        log::info!("The next clip stays closed. Hardware readers are at the cap.");
        *delayed = true;
    }
    if !skipped {
        *delayed = false;
    }
    note_readers(open, readers.len());
}

#[cfg(target_vendor = "apple")]
fn matches_reader(reader: &Reader, track: TrackId, asset: AssetId, source: &str) -> bool {
    reader.track == track && reader.asset == asset && reader.source == source
}

#[cfg(target_vendor = "apple")]
fn note_readers(open: &mut usize, count: usize) {
    if *open == count {
        return;
    }
    *open = count;
    log::info!("The preview is decoding {count} pictures.");
}

#[cfg(target_vendor = "apple")]
fn spawn_reader(
    readers: &mut Vec<Reader>,
    board: Arc<Mutex<Vec<LayerFrame>>>,
    track: TrackId,
    asset: AssetId,
    source: String,
    first: Option<(u64, Timestamp)>,
) {
    let note = Arc::new(Note::new());
    if let Some((generation, time)) = first {
        note.post(ReaderMsg::At { generation, time });
    }
    let note_thread = Arc::clone(&note);
    let source_thread = source.clone();
    let spawned = std::thread::Builder::new()
        .name("cutlass-decode".to_owned())
        .spawn(move || decode_loop(note_thread, board, track, source_thread));
    match spawned {
        Ok(join) => readers.push(Reader {
            track,
            asset,
            source,
            note,
            join,
        }),
        Err(error) => log::warn!("The preview could not open a picture reader: {error}."),
    }
}

#[cfg(target_vendor = "apple")]
fn stop_readers(readers: &mut Vec<Reader>) {
    for reader in readers.iter() {
        reader.note.post(ReaderMsg::Stop);
    }
    for reader in readers.drain(..) {
        let _ = reader.join.join();
    }
}

#[cfg(target_vendor = "apple")]
fn decode_loop(
    note: Arc<Note<ReaderMsg>>,
    board: Arc<Mutex<Vec<LayerFrame>>>,
    track: TrackId,
    source: String,
) {
    let mut open = match open_clip(&source) {
        Clip::Ready(open) => Some(*open),
        Clip::Missing => None,
    };
    let mut target = None;
    loop {
        if target.is_none() {
            match note.wait() {
                None | Some(ReaderMsg::Stop) => return,
                Some(ReaderMsg::At { generation, time }) => target = Some((generation, time)),
            }
            continue;
        }
        let Some((generation, time)) = target else {
            continue;
        };
        let Some(clip) = open.as_mut() else {
            publish(
                &board,
                LayerFrame {
                    track,
                    generation,
                    requested: time,
                    body: Body::Unavailable,
                },
            );
            target = None;
            continue;
        };
        let chunk = step(clip, &board, track, generation, time);
        match note.poll() {
            Inbox::Closed | Inbox::Message(ReaderMsg::Stop) => return,
            Inbox::Message(ReaderMsg::At { generation, time }) => {
                target = Some((generation, time));
            }
            Inbox::Empty => {
                if chunk != Chunk::Short {
                    target = None;
                }
            }
        }
    }
}

#[cfg(target_vendor = "apple")]
fn step(
    clip: &mut Open,
    board: &Mutex<Vec<LayerFrame>>,
    track: TrackId,
    generation: u64,
    time: Timestamp,
) -> Chunk {
    if clip.failed == Some(time) {
        publish(board, missing(track, generation, time));
        return Chunk::Settled;
    }
    let readable = match pump(clip, time) {
        Ok(readable) => readable,
        Err(error) => {
            if clip.failed.is_none() {
                log::warn!("The preview could not decode {}: {error}.", clip.source);
            }
            clip.failed = Some(time);
            publish(board, missing(track, generation, time));
            return Chunk::Settled;
        }
    };
    if !readable {
        if clip.failed.is_none() {
            log::warn!("The preview has no frame at this time in {}.", clip.source);
        }
        clip.failed = Some(time);
        publish(board, missing(track, generation, time));
        return Chunk::Settled;
    }
    let matrix = clip.matrix;
    let taken = clip
        .ahead
        .at(time)
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
            clip.failed = Some(time);
            publish(board, missing(track, generation, time));
        }
        return Chunk::Settled;
    };
    clip.failed = None;
    publish(
        board,
        LayerFrame {
            track,
            generation,
            requested: time,
            body: Body::Frame {
                buffer,
                matrix,
                pts,
            },
        },
    );
    Chunk::Covers
}

#[cfg(target_vendor = "apple")]
fn missing(track: TrackId, generation: u64, time: Timestamp) -> LayerFrame {
    LayerFrame {
        track,
        generation,
        requested: time,
        body: Body::Unavailable,
    }
}

#[cfg(target_vendor = "apple")]
fn publish(board: &Mutex<Vec<LayerFrame>>, frame: LayerFrame) {
    let mut board = board.lock().unwrap_or_else(|poison| poison.into_inner());
    if let Some(existing) = board.iter_mut().find(|item| item.track == frame.track) {
        *existing = frame;
    } else {
        board.push(frame);
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
