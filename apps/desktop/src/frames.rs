//! The pictures under the playhead.
//!
//! On Apple each video track has its own VideoToolbox reader and frame queue.
//! The reader stays inside the current group of pictures, skips to a keyframe
//! the clock has reached, and drops frames that are already late. Pictures it
//! has already decoded stay in a per-track cache, and the ones that cache
//! drops stay in a shared pool. A file that was opened before is parsed once.
//! A source clip is also prepared as an H.264 file, and a later open plays
//! that file. The sound stays on the source. The next clip on a track opens
//! while the hardware reader count is under the cap.
//! The window draws those frames from the bottom track upward and does not
//! copy them back. A failure is written to the app log. A seek, and a cut,
//! keep the last picture until the new one is ready. Where no video decoder
//! is linked, there is no picture.

use cutlass_compose::Gpu;
use cutlass_project::Project;

#[cfg(target_vendor = "apple")]
use cutlass_core::{
    ColorTags, ContentLight, MasteringDisplay, Packet, Rotation, Timestamp, VideoFrame,
    VideoStreamConfig,
};
#[cfg(target_vendor = "apple")]
use cutlass_project::{AssetId, TrackId};
#[cfg(target_vendor = "apple")]
use std::path::Path;

#[cfg(target_vendor = "apple")]
use std::collections::VecDeque;
#[cfg(target_vendor = "apple")]
use std::num::NonZeroUsize;
#[cfg(target_vendor = "apple")]
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
#[cfg(target_vendor = "apple")]
use std::thread::JoinHandle;

#[cfg(target_vendor = "apple")]
use cutlass_color::{Matrix, YuvPlanes};
#[cfg(target_vendor = "apple")]
use cutlass_media::{Content, Probe, VideoSession};
#[cfg(target_vendor = "apple")]
use cutlass_playback::{Ahead, Decode, FrameCache, FramePool, Reach};
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

/// Pictures one track keeps. CapCut's frame-cache maximum was not recovered.
#[cfg(target_vendor = "apple")]
const FRAME_CACHE_MAX: usize = 8;

/// Pictures playback keeps behind the playhead.
#[cfg(target_vendor = "apple")]
const FRAME_CACHE_MIN: usize = 2;

/// Pictures kept after a track cache drops them. CapCut's pool size was not recovered.
#[cfg(target_vendor = "apple")]
const FRAME_POOL_MAX: usize = 64;

/// Parsed files kept so the next reader does not read them again.
#[cfg(target_vendor = "apple")]
const DEMUX_CACHE_MAX: usize = 8;

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
    /// Keeps this file in the demuxer cache for as long as the reader is open.
    parsed: Arc<ParsedClip>,
    session: Session,
    ahead: Ahead<PixelBuffer>,
    pictures: FrameCache<PixelBuffer>,
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
struct ParsedClip {
    packets: Arc<[Packet]>,
    matrix: Matrix,
    config: VideoStreamConfig,
    color: Option<ColorTags>,
    mastering: Option<MasteringDisplay>,
    content_light: Option<ContentLight>,
    rotation: Rotation,
}

#[cfg(target_vendor = "apple")]
struct Caches {
    demux: Mutex<DemuxCache>,
    pool: Mutex<PicturePool>,
}

#[cfg(target_vendor = "apple")]
impl Caches {
    fn new() -> Self {
        Self {
            demux: Mutex::new(DemuxCache::new(DEMUX_CACHE_MAX)),
            pool: Mutex::new(PicturePool::new(FRAME_POOL_MAX)),
        }
    }
}

#[cfg(target_vendor = "apple")]
struct PicturePool {
    frames: FramePool<PixelBuffer>,
    noted: usize,
}

#[cfg(target_vendor = "apple")]
impl PicturePool {
    fn new(maximum: usize) -> Self {
        let maximum = NonZeroUsize::new(maximum).unwrap_or(NonZeroUsize::MIN);
        Self {
            frames: FramePool::new(maximum),
            noted: 0,
        }
    }

    fn find(&self, key: &str, time: Timestamp) -> Option<VideoFrame<PixelBuffer>> {
        self.frames
            .find(key, time, Timestamp::new(0, time.timescale))
            .cloned()
    }

    fn put_all(&mut self, key: &str, around: Timestamp, frames: Vec<VideoFrame<PixelBuffer>>) {
        if frames.is_empty() {
            return;
        }
        for frame in frames {
            self.frames.put(key, frame, around);
        }
        self.note();
    }

    fn note(&mut self) {
        let len = self.frames.len();
        if self.noted == len {
            return;
        }
        self.noted = len;
        log::info!("The preview is holding {len} cached pictures.");
    }
}

/// Parsed files, oldest first. A file a reader still holds is not dropped.
#[cfg(target_vendor = "apple")]
struct DemuxCache {
    maximum: usize,
    entries: VecDeque<(String, Arc<ParsedClip>)>,
}

#[cfg(target_vendor = "apple")]
#[derive(Debug)]
enum DemuxStore {
    Stored,
    Evicted(String),
    Full,
}

#[cfg(target_vendor = "apple")]
impl DemuxCache {
    fn new(maximum: usize) -> Self {
        Self {
            maximum,
            entries: VecDeque::new(),
        }
    }

    fn fetch(&mut self, source: &str) -> Option<Arc<ParsedClip>> {
        let index = self.entries.iter().position(|(key, _)| key == source)?;
        let entry = self.entries.remove(index)?;
        let parsed = Arc::clone(&entry.1);
        self.entries.push_back(entry);
        Some(parsed)
    }

    fn store(&mut self, source: String, parsed: Arc<ParsedClip>) -> DemuxStore {
        if let Some(index) = self.entries.iter().position(|(key, _)| key == &source) {
            if let Some(entry) = self.entries.remove(index) {
                self.entries.push_back(entry);
            }
            return DemuxStore::Stored;
        }
        if self.entries.len() >= self.maximum {
            let unused = self
                .entries
                .iter()
                .position(|(_, parsed)| Arc::strong_count(parsed) == 1);
            let Some(index) = unused else {
                return DemuxStore::Full;
            };
            let Some((previous, _)) = self.entries.remove(index) else {
                return DemuxStore::Full;
            };
            self.entries.push_back((source, parsed));
            return DemuxStore::Evicted(previous);
        }
        self.entries.push_back((source, parsed));
        DemuxStore::Stored
    }
}

#[cfg(target_vendor = "apple")]
fn open_clip(source: &str, caches: &Caches, prepared: bool) -> Clip {
    let Some(path) = file_path(source) else {
        log::warn!("The preview skipped {source}: it is not a file.");
        return Clip::Missing;
    };
    // The prepared file uses its own demuxer key, so a cached source parse is
    // not reused as the proxy. `prepared` is the decision from spawn, so this
    // open and the reader flag agree.
    crate::preview_file::ensure(source);
    if prepared && let Some(file) = crate::preview_file::ready(source) {
        if let Some(clip) = try_open(source, &file, &file, caches) {
            return clip;
        }
        crate::preview_file::discard(source);
    }
    try_open(source, source, path, caches).unwrap_or(Clip::Missing)
}

#[cfg(target_vendor = "apple")]
fn try_open(source: &str, key: &str, path: &str, caches: &Caches) -> Option<Clip> {
    let parsed = load_parsed(caches, key, path)?;
    let session = match VideoSession::open(
        &parsed.config,
        parsed.color,
        parsed.mastering,
        parsed.content_light,
        parsed.rotation,
    ) {
        Ok(session) => session,
        Err(error) => {
            log::warn!("The preview could not decode {path}: {error}.");
            return None;
        }
    };
    let depth = NonZeroUsize::new(QUEUE_DEPTH).expect("the queue holds a frame");
    let pictures = FrameCache::new(
        NonZeroUsize::new(FRAME_CACHE_MAX).expect("the cache holds a frame"),
        FRAME_CACHE_MIN,
    );
    Some(Clip::Ready(Box::new(Open {
        source: source.to_owned(),
        parsed,
        session: Session(session),
        ahead: Ahead::new(depth),
        pictures,
        cursor: 0,
        followed: None,
        failed: None,
    })))
}

#[cfg(target_vendor = "apple")]
/// The lookup lock ends before `read`. Holding it across the disk read, then
/// locking again to store, waits forever: a mutex does not lock twice on the
/// same thread.
fn load_parsed(caches: &Caches, source: &str, path: &str) -> Option<Arc<ParsedClip>> {
    load_parsed_with(caches, source, path, read_clip)
}

fn load_parsed_with(
    caches: &Caches,
    source: &str,
    path: &str,
    read: impl FnOnce(&str) -> Option<Arc<ParsedClip>>,
) -> Option<Arc<ParsedClip>> {
    if let Some(parsed) = fetch_parsed(caches, source) {
        log::info!("The preview reused a cached demuxer for {path}.");
        return Some(parsed);
    }
    let parsed = read(path)?;
    match lock_demux(caches).store(source.to_owned(), Arc::clone(&parsed)) {
        DemuxStore::Evicted(previous) => {
            log::info!(
                "The preview dropped the cached demuxer for {previous}. The demuxer cache is full."
            );
        }
        DemuxStore::Full => {
            log::info!("The preview opened {path} and the demuxer cache is full.");
        }
        DemuxStore::Stored => {}
    }
    Some(parsed)
}

fn fetch_parsed(caches: &Caches, source: &str) -> Option<Arc<ParsedClip>> {
    lock_demux(caches).fetch(source)
}

fn lock_demux(caches: &Caches) -> MutexGuard<'_, DemuxCache> {
    caches
        .demux
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
}

#[cfg(target_vendor = "apple")]
fn read_clip(path: &str) -> Option<Arc<ParsedClip>> {
    let bytes = match std::fs::read(Path::new(path)) {
        Ok(bytes) => bytes,
        Err(error) => {
            log::warn!("The preview could not read {path}: {error}.");
            return None;
        }
    };
    let probe = match Probe::parse(&bytes) {
        Ok(probe) => probe,
        Err(error) => {
            log::warn!("The preview could not open {path}: {error}.");
            return None;
        }
    };
    let Some(index) = probe
        .tracks()
        .iter()
        .position(|track| track.enabled && matches!(track.content, Content::Video { .. }))
    else {
        log::warn!("The preview found no picture track in {path}.");
        return None;
    };
    let Content::Video {
        config,
        color,
        mastering,
        content_light,
    } = &probe.tracks()[index].content
    else {
        log::warn!("The preview found no picture track in {path}.");
        return None;
    };
    let matrix = match Matrix::resolve(
        color.as_ref().map(|tags| tags.matrix),
        config.width,
        config.height,
    ) {
        Ok(matrix) => matrix,
        Err(error) => {
            log::warn!("The preview could not resolve the color of {path}: {error}.");
            return None;
        }
    };
    let Some(packets) = probe.samples(index).map(<[Packet]>::to_vec) else {
        log::warn!("The preview found no picture samples in {path}.");
        return None;
    };
    Some(Arc::new(ParsedClip {
        packets: Arc::from(packets),
        matrix,
        config: config.clone(),
        color: *color,
        mastering: *mastering,
        content_light: *content_light,
        rotation: probe.tracks()[index].rotation,
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
    playing: bool,
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
    At {
        generation: u64,
        time: Timestamp,
        playing: bool,
    },
    Stop,
}

#[cfg(target_vendor = "apple")]
struct Reader {
    track: TrackId,
    asset: AssetId,
    source: String,
    /// This reader was opened on the prepared file. A play that is already
    /// running stays on the file it started with.
    prepared: bool,
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
    let caches = Arc::new(Caches::new());
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
                reconcile(
                    &board,
                    &caches,
                    &mut readers,
                    &mut open,
                    &mut delayed,
                    &show,
                );
            }
        }
    }
    stop_readers(&mut readers);
}

#[cfg(target_vendor = "apple")]
fn reconcile(
    board: &Arc<Mutex<Vec<LayerFrame>>>,
    caches: &Arc<Caches>,
    readers: &mut Vec<Reader>,
    open: &mut usize,
    delayed: &mut bool,
    show: &Ask,
) {
    let mut index = 0;
    while index < readers.len() {
        let reader = &readers[index];
        let kept = show.layers.iter().any(|layer| {
            matches_reader(
                reader,
                layer.track,
                layer.asset,
                &layer.source,
                show.playing,
            )
        }) || show
            .next
            .iter()
            .any(|next| matches_reader(reader, next.track, next.asset, &next.source, show.playing));
        if kept {
            index += 1;
        } else {
            let reader = readers.swap_remove(index);
            reader.note.post(ReaderMsg::Stop);
        }
    }
    for layer in &show.layers {
        if let Some(reader) = readers.iter().find(|reader| {
            matches_reader(
                reader,
                layer.track,
                layer.asset,
                &layer.source,
                show.playing,
            )
        }) {
            reader.note.post(ReaderMsg::At {
                generation: layer.generation,
                time: layer.time,
                playing: show.playing,
            });
            continue;
        }
        spawn_reader(
            readers,
            Arc::clone(board),
            Arc::clone(caches),
            layer.track,
            layer.asset,
            layer.source.clone(),
            Some((layer.generation, layer.time, show.playing)),
        );
    }
    let mut skipped = false;
    for next in &show.next {
        if readers.iter().any(|reader| {
            matches_reader(reader, next.track, next.asset, &next.source, show.playing)
        }) {
            continue;
        }
        if readers.len() >= HW_READER_CAP {
            skipped = true;
            continue;
        }
        spawn_reader(
            readers,
            Arc::clone(board),
            Arc::clone(caches),
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
fn matches_reader(
    reader: &Reader,
    track: TrackId,
    asset: AssetId,
    source: &str,
    playing: bool,
) -> bool {
    reader.track == track
        && reader.asset == asset
        && reader.source == source
        && (reader.prepared == crate::preview_file::ready(source).is_some() || playing)
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
    caches: Arc<Caches>,
    track: TrackId,
    asset: AssetId,
    source: String,
    first: Option<(u64, Timestamp, bool)>,
) {
    let note = Arc::new(Note::new());
    if let Some((generation, time, playing)) = first {
        note.post(ReaderMsg::At {
            generation,
            time,
            playing,
        });
    }
    let note_thread = Arc::clone(&note);
    let source_thread = source.clone();
    let prepared = crate::preview_file::ready(&source).is_some();
    let spawned = std::thread::Builder::new()
        .name("cutlass-decode".to_owned())
        .spawn(move || decode_loop(note_thread, board, caches, track, source_thread, prepared));
    match spawned {
        Ok(join) => {
            readers.push(Reader {
                track,
                asset,
                source,
                prepared,
                note,
                join,
            });
        }
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
    caches: Arc<Caches>,
    track: TrackId,
    source: String,
    prepared: bool,
) {
    let mut open = match open_clip(&source, &caches, prepared) {
        Clip::Ready(open) => Some(*open),
        Clip::Missing => None,
    };
    let parked = ParkPictures {
        open: &mut open,
        pool: &caches.pool,
    };
    let mut target = None;
    loop {
        if target.is_none() {
            match note.wait() {
                None | Some(ReaderMsg::Stop) => return,
                Some(ReaderMsg::At {
                    generation,
                    time,
                    playing,
                }) => target = Some((generation, time, playing)),
            }
            continue;
        }
        let Some((generation, time, playing)) = target else {
            continue;
        };
        let Some(clip) = parked.open.as_mut() else {
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
        let chunk = step(clip, &board, &caches.pool, track, generation, time, playing);
        match note.poll() {
            Inbox::Closed | Inbox::Message(ReaderMsg::Stop) => return,
            Inbox::Message(ReaderMsg::At {
                generation,
                time,
                playing,
            }) => {
                target = Some((generation, time, playing));
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
struct ParkPictures<'a> {
    open: &'a mut Option<Open>,
    pool: &'a Mutex<PicturePool>,
}

#[cfg(target_vendor = "apple")]
impl Drop for ParkPictures<'_> {
    fn drop(&mut self) {
        let Some(clip) = self.open.as_mut() else {
            return;
        };
        let frames = clip.pictures.drain();
        let Some(around) = frames.last().map(|frame| frame.pts) else {
            return;
        };
        let mut pool = self
            .pool
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        pool.put_all(&clip.source, around, frames);
    }
}

#[cfg(target_vendor = "apple")]
fn step(
    clip: &mut Open,
    board: &Mutex<Vec<LayerFrame>>,
    pool: &Mutex<PicturePool>,
    track: TrackId,
    generation: u64,
    time: Timestamp,
    playing: bool,
) -> Chunk {
    let hit = cached_frame(clip, pool, time, playing);
    if let Some(frame) = &hit {
        clip.failed = None;
        show_frame(clip, board, track, generation, time, frame);
        let behind = clip.followed.is_some_and(|followed| time < followed);
        if !playing || behind {
            return Chunk::Covers;
        }
    }
    if hit.is_none() && clip.failed == Some(time) {
        publish(board, missing(track, generation, time));
        return Chunk::Settled;
    }
    let (readable, passed) = match pump(clip, time) {
        Ok(read) => read,
        Err(error) => {
            if hit.is_some() {
                return Chunk::Covers;
            }
            if clip.failed.is_none() {
                log::warn!("The preview could not decode {}: {error}.", clip.source);
            }
            clip.failed = Some(time);
            publish(board, missing(track, generation, time));
            return Chunk::Settled;
        }
    };
    if !readable {
        if hit.is_some() {
            return Chunk::Covers;
        }
        if clip.failed.is_none() {
            log::warn!("The preview has no frame at this time in {}.", clip.source);
        }
        clip.failed = Some(time);
        publish(board, missing(track, generation, time));
        return Chunk::Settled;
    }
    remember(clip, pool, time, playing, passed);
    let matrix = clip.parsed.matrix;
    let taken = clip
        .ahead
        .at(time)
        .map(|frame| (frame.pts, frame.image.clone()));
    let Some((pts, buffer)) = taken else {
        if hit.is_some() {
            return Chunk::Covers;
        }
        let finished = clip.ahead.is_full()
            || (clip.cursor >= clip.parsed.packets.len() && clip.ahead.drained());
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
fn cached_frame(
    clip: &mut Open,
    pool: &Mutex<PicturePool>,
    time: Timestamp,
    playing: bool,
) -> Option<VideoFrame<PixelBuffer>> {
    let tolerance = Timestamp::new(0, time.timescale);
    if let Some(frame) = clip.pictures.find(time, tolerance).cloned() {
        return Some(frame);
    }
    let found = pool
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .find(&clip.source, time);
    let frame = found?;
    let remembered = clip.pictures.insert(frame.clone(), time, playing);
    let mut extra = remembered.evicted;
    if let Some(refused) = remembered.refused {
        extra.push(refused);
    }
    if !extra.is_empty() {
        pool.lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .put_all(&clip.source, time, extra);
    }
    Some(frame)
}

#[cfg(target_vendor = "apple")]
fn remember(
    clip: &mut Open,
    pool: &Mutex<PicturePool>,
    time: Timestamp,
    playing: bool,
    passed: Vec<VideoFrame<PixelBuffer>>,
) {
    let mut extra = Vec::new();
    let queued: Vec<_> = clip.ahead.frames().cloned().collect();
    for frame in passed.into_iter().chain(queued) {
        keep(clip, time, playing, frame, &mut extra);
    }
    if extra.is_empty() {
        return;
    }
    pool.lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .put_all(&clip.source, time, extra);
}

fn keep(
    clip: &mut Open,
    time: Timestamp,
    playing: bool,
    frame: VideoFrame<PixelBuffer>,
    extra: &mut Vec<VideoFrame<PixelBuffer>>,
) {
    let remembered = clip.pictures.insert(frame, time, playing);
    extra.extend(remembered.evicted);
    if let Some(refused) = remembered.refused {
        extra.push(refused);
    }
}

#[cfg(target_vendor = "apple")]
fn show_frame(
    clip: &Open,
    board: &Mutex<Vec<LayerFrame>>,
    track: TrackId,
    generation: u64,
    time: Timestamp,
    frame: &VideoFrame<PixelBuffer>,
) {
    publish(
        board,
        LayerFrame {
            track,
            generation,
            requested: time,
            body: Body::Frame {
                buffer: frame.image.clone(),
                matrix: clip.parsed.matrix,
                pts: frame.pts,
            },
        },
    );
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
fn pump(clip: &mut Open, time: Timestamp) -> Result<(bool, Vec<VideoFrame<PixelBuffer>>), String> {
    let (start, restart) = match cutlass_playback::reach(
        &clip.parsed.packets,
        clip.cursor,
        clip.followed,
        clip.ahead.earliest(),
        clip.ahead.drained(),
        time,
    ) {
        None => {
            clip.followed = Some(time);
            return Ok((false, Vec::new()));
        }
        Some(Reach::Keyframe(index)) => (index, true),
        Some(Reach::Forward) => (clip.cursor.min(clip.parsed.packets.len()), false),
    };
    let mut pending = &clip.parsed.packets[start..];
    let passed = clip
        .ahead
        .follow(
            &mut clip.session,
            &mut pending,
            time,
            DECODE_BUDGET,
            restart,
        )
        .map_err(|error| error.to_string())?;
    clip.cursor = clip.parsed.packets.len() - pending.len();
    clip.followed = Some(time);
    Ok((true, passed))
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
    fn a_miss_reads_the_file_without_holding_the_demuxer_cache() {
        let caches = super::Caches::new();
        let parsed = super::load_parsed_with(&caches, "file:///cuts/a.mp4", "/cuts/a.mp4", |_| {
            assert!(
                caches.demux.try_lock().is_ok(),
                "the demuxer cache stayed locked while the file was read"
            );
            None
        });
        assert!(parsed.is_none());
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn a_demuxer_cache_reuses_a_parsed_file_and_keeps_one_still_open() {
        use std::sync::Arc;

        use cutlass_color::Matrix;
        use cutlass_core::{Rotation, VideoCodec, VideoStreamConfig};

        use super::{DemuxCache, DemuxStore, ParsedClip};

        let parsed = || {
            Arc::new(ParsedClip {
                packets: Arc::from(Vec::new()),
                matrix: Matrix::Bt709,
                config: VideoStreamConfig {
                    codec: VideoCodec::H264,
                    width: 16,
                    height: 16,
                    config_record: Vec::new(),
                },
                color: None,
                mastering: None,
                content_light: None,
                rotation: Rotation::Zero,
            })
        };
        let mut cache = DemuxCache::new(1);
        let first = parsed();
        assert!(matches!(
            cache.store("a".to_owned(), Arc::clone(&first)),
            DemuxStore::Stored
        ));
        drop(first);
        let held = parsed();
        match cache.store("b".to_owned(), Arc::clone(&held)) {
            DemuxStore::Evicted(previous) => assert_eq!(previous, "a"),
            other => panic!("expected the idle demuxer to be dropped, got {other:?}"),
        }
        assert!(matches!(
            cache.store("c".to_owned(), parsed()),
            DemuxStore::Full
        ));
        assert!(cache.fetch("b").is_some());
        assert!(cache.fetch("a").is_none());
        drop(held);
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
