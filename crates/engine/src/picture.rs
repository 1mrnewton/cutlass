//! The pictures under the playhead.
//!
//! On Apple each video track has its own reader and frame queue. The shell
//! uploads the frame. A prepared file, when the shell has one, is what the
//! reader opens. This module does not draw.

use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;

use cutlass_color::Matrix;
use cutlass_core::{
    ColorTags, ContentLight, MasteringDisplay, Packet, Rotation, Timestamp, VideoFrame,
    VideoStreamConfig,
};
use cutlass_media::{Content, Probe, VideoSession};
use cutlass_playback::{Ahead, Decode, FrameCache, FramePool, Reach};
use cutlass_project::{AssetId, TrackId};
use cutlass_videotoolbox::PixelBuffer;

use crate::file_path;

/// How many decoded frames the worker keeps, including the one on screen.
const QUEUE_DEPTH: usize = 8;

/// How many new frames one pass decodes before it looks at the playhead again.
const DECODE_BUDGET: usize = QUEUE_DEPTH;

/// CapCut's preload scheduler stops at `hw reader count max`. The number was
/// not recovered. A picture that is already on screen still gets a reader.
const HW_READER_CAP: usize = 8;

/// Pictures one track keeps. CapCut's frame-cache maximum was not recovered.
const FRAME_CACHE_MAX: usize = 8;

/// Pictures playback keeps behind the playhead.
const FRAME_CACHE_MIN: usize = 2;

/// Pictures kept after a track cache drops them. CapCut's pool size was not recovered.
const FRAME_POOL_MAX: usize = 64;

/// Parsed files kept so the next reader does not read them again.
const DEMUX_CACHE_MAX: usize = 8;

/// How the shell names a prepared picture, when it has one.
#[derive(Clone, Copy)]
pub struct Proxy {
    /// Start preparing `source`, if the shell does that.
    pub ensure: fn(&str),
    /// The prepared file, when it is valid.
    pub ready: fn(&str) -> Option<String>,
    /// The prepared file could not be opened. The shell drops it.
    pub discard: fn(&str),
}

impl Proxy {
    /// No prepared file. The readers play the source.
    pub const fn off() -> Self {
        Self {
            ensure: ignore_source,
            ready: no_file,
            discard: ignore_source,
        }
    }
}

fn ignore_source(_source: &str) {}

fn no_file(_source: &str) -> Option<String> {
    None
}

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
    proxy: Proxy,
}

#[cfg(target_vendor = "apple")]
impl Caches {
    fn new(proxy: Proxy) -> Self {
        Self {
            demux: Mutex::new(DemuxCache::new(DEMUX_CACHE_MAX)),
            pool: Mutex::new(PicturePool::new(FRAME_POOL_MAX)),
            proxy,
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
    (caches.proxy.ensure)(source);
    if prepared {
        if let Some(file) = (caches.proxy.ready)(source) {
            if let Some(clip) = try_open(source, &file, &file, caches) {
                return clip;
            }
            (caches.proxy.discard)(source);
        }
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
        &parsed.packets,
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
    fn start(proxy: Proxy) -> Self {
        let ask = Arc::new(Note::new());
        let board = Arc::new(Mutex::new(Vec::new()));
        let ask_thread = Arc::clone(&ask);
        let board_thread = Arc::clone(&board);
        let join = std::thread::Builder::new()
            .name("cutlass-playback".to_owned())
            .spawn(move || direct_loop(ask_thread, board_thread, proxy))
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
pub struct Ask {
    pub layers: Vec<ShowLayer>,
    pub next: Vec<ShowNext>,
    pub playing: bool,
}

#[cfg(target_vendor = "apple")]
#[derive(Clone, PartialEq, Eq)]
pub struct ShowLayer {
    pub track: TrackId,
    pub generation: u64,
    pub asset: AssetId,
    pub source: String,
    pub time: Timestamp,
}

/// A clip that starts within a second of the playhead.
#[cfg(target_vendor = "apple")]
#[derive(Clone, PartialEq, Eq)]
pub struct ShowNext {
    pub track: TrackId,
    pub asset: AssetId,
    pub source: String,
}

#[cfg(target_vendor = "apple")]
pub struct LayerFrame {
    pub track: TrackId,
    pub generation: u64,
    pub requested: Timestamp,
    pub body: Body,
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
pub enum Body {
    Frame {
        buffer: PixelBuffer,
        matrix: Matrix,
        pts: Timestamp,
    },
    Unavailable,
}

#[cfg(target_vendor = "apple")]
fn direct_loop(ask: Arc<Note<Message>>, board: Arc<Mutex<Vec<LayerFrame>>>, proxy: Proxy) {
    let caches = Arc::new(Caches::new(proxy));
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
                caches.proxy,
            )
        }) || show.next.iter().any(|next| {
            matches_reader(
                reader,
                next.track,
                next.asset,
                &next.source,
                show.playing,
                caches.proxy,
            )
        });
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
                caches.proxy,
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
            matches_reader(
                reader,
                next.track,
                next.asset,
                &next.source,
                show.playing,
                caches.proxy,
            )
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
    proxy: Proxy,
) -> bool {
    reader.track == track
        && reader.asset == asset
        && reader.source == source
        && (reader.prepared == (proxy.ready)(source).is_some() || playing)
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
    let prepared = (caches.proxy.ready)(&source).is_some();
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
        let finished = chase_finished(clip);
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

/// Whether a frame decoded for `requested` still belongs on the plate.
///
/// A frame at or before `asked` covers a playhead that moved forward. While
/// scrubbing, a frame still ahead of the playhead is shown when it is closer
/// than the picture already there. Playback does not show that frame.
#[cfg(target_vendor = "apple")]
pub fn shows_frame(
    playing: bool,
    asked: Timestamp,
    pts: Timestamp,
    requested: Timestamp,
    shown: Option<Timestamp>,
) -> bool {
    if requested == asked || pts <= asked {
        return true;
    }
    if playing {
        return false;
    }
    shown.is_none_or(|shown| apart(pts, asked) < apart(shown, asked))
}

#[cfg(target_vendor = "apple")]
fn apart(left: Timestamp, right: Timestamp) -> i128 {
    let left_ticks = i128::from(left.value) * i128::from(right.timescale.get());
    let right_ticks = i128::from(right.value) * i128::from(left.timescale.get());
    (left_ticks - right_ticks).abs()
}

#[cfg(target_vendor = "apple")]
fn chase_finished(clip: &Open) -> bool {
    clip.ahead.is_full() || (clip.cursor >= clip.parsed.packets.len() && clip.ahead.drained())
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

/// The readers for the pictures under the playhead.
pub struct Pictures {
    bridge: Bridge,
}

impl Pictures {
    /// Starts the reader thread. `proxy` names a prepared file when the shell has one.
    pub fn new(proxy: Proxy) -> Self {
        Self {
            bridge: Bridge::start(proxy),
        }
    }

    /// Points the readers at `ask`. False when the thread did not start.
    pub fn show(&self, ask: Ask) -> bool {
        self.bridge.post(Message::Show(ask))
    }

    /// Closes every reader.
    pub fn drop_all(&self) {
        let _ = self.bridge.post(Message::Drop);
    }

    /// Frames decoded since the last take. The shell uploads them.
    pub fn take(&self) -> Vec<LayerFrame> {
        self.bridge.take_frames()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cutlass_color::Matrix;
    use cutlass_core::{Rotation, VideoCodec, VideoStreamConfig};

    use super::{Caches, DemuxCache, DemuxStore, ParsedClip, Proxy, load_parsed_with, shows_frame};

    #[test]
    fn a_miss_reads_the_file_without_holding_the_demuxer_cache() {
        let caches = Caches::new(Proxy::off());
        let parsed = load_parsed_with(&caches, "file:///cuts/a.mp4", "/cuts/a.mp4", |_| {
            assert!(
                caches.demux.try_lock().is_ok(),
                "the demuxer cache stayed locked while the file was read"
            );
            None
        });
        assert!(parsed.is_none());
    }

    #[test]
    fn a_demuxer_cache_reuses_a_parsed_file_and_keeps_one_still_open() {
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

    #[test]
    fn a_fast_backward_scrub_keeps_a_closer_frame() {
        use std::num::NonZeroU32;

        let scale = NonZeroU32::new(1_000).expect("scale");
        let at = |ms| cutlass_core::Timestamp::new(ms, scale);
        let asked = at(1_000);
        let on_screen = Some(at(4_000));
        assert!(shows_frame(false, asked, at(2_000), at(4_000), on_screen));
        assert!(!shows_frame(false, asked, at(7_000), at(4_000), on_screen));
        assert!(!shows_frame(true, asked, at(2_000), at(4_000), on_screen));
        assert!(!shows_frame(
            false,
            asked,
            at(2_000),
            at(4_000),
            Some(at(0)),
        ));
        assert!(shows_frame(false, asked, at(1_000), at(4_000), on_screen));
        assert!(shows_frame(true, asked, at(900), at(4_000), on_screen));
        assert!(shows_frame(true, asked, at(2_000), asked, on_screen));
        assert!(shows_frame(false, asked, at(2_000), at(4_000), None));
        assert!(!shows_frame(true, asked, at(2_000), at(4_000), None));
    }
}
