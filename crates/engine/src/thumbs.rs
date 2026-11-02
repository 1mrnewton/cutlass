//! Timeline thumbnails, decoded on their own thread.
//!
//! The timeline asks for the picture of a file at a source time. A lookup
//! never decodes and never waits. A miss is queued for the worker, which
//! opens the file with its own readers, decodes that frame, shrinks it to at
//! most [`THUMB_HEIGHT`] rows, and stores it. The newest ask replaces the
//! queue, so tiles that scrolled out of view are dropped. While paused the
//! worker stops between frames, so playback keeps the decoder to itself.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;

use cutlass_codec::access::PixelFormat;
use cutlass_model::{Micros, TimeRange};
use cutlass_picture::{Picture, rgba_len};

use crate::error::EngineError;
use crate::reader::{DecodedFrame, Media, ReaderFactory};
use crate::source::decoded_picture;
use cutlass_container::DisplayQuarter;

/// Rows in a stored thumbnail. A shorter picture keeps its own height.
pub const THUMB_HEIGHT: u32 = 96;

/// Thumbnails kept at once. The oldest is dropped past this.
pub const MAX_THUMBS: usize = 1200;

/// Files the worker keeps open between asks.
const OPEN_FILES: usize = 4;

/// The picture of `path` at source time `time`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ThumbKey {
    pub path: PathBuf,
    pub time: Micros,
}

/// A small RGBA picture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Thumb {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// The thumbnail store and the thread that fills it.
pub struct Thumbnails {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
}

#[derive(Default)]
struct State {
    stored: HashMap<ThumbKey, Arc<Thumb>>,
    order: VecDeque<ThumbKey>,
    wanted: VecDeque<ThumbKey>,
    failed: HashSet<ThumbKey>,
    /// Each opened file's picture size, upright.
    shapes: HashMap<PathBuf, (u32, u32)>,
    paused: bool,
    closed: bool,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Thumbnails {
    /// Start the worker. `ready` runs on the worker after each thumbnail is
    /// stored, so the caller can draw again.
    pub fn start(factory: ReaderFactory, ready: impl Fn() + Send + 'static) -> Self {
        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            wake: Condvar::new(),
        });
        let worker = {
            let shared = Arc::clone(&shared);
            std::thread::Builder::new()
                .name("cutlass-thumbnails".to_owned())
                .spawn(move || work(&shared, &factory, &ready))
                .ok()
        };
        Self { shared, worker }
    }

    /// The stored thumbnail for `key`. This never decodes.
    #[must_use]
    pub fn get(&self, key: &ThumbKey) -> Option<Arc<Thumb>> {
        self.shared.lock().stored.get(key).cloned()
    }

    /// Ask for `keys`, most wanted first. The asks replace the earlier ones;
    /// a key already stored, or one that failed, is not asked again.
    pub fn want(&self, keys: impl IntoIterator<Item = ThumbKey>) {
        let mut state = self.shared.lock();
        let mut seen = HashSet::new();
        let wanted: VecDeque<ThumbKey> = keys
            .into_iter()
            .filter(|key| !state.stored.contains_key(key) && !state.failed.contains(key))
            .filter(|key| seen.insert(key.clone()))
            .collect();
        state.wanted = wanted;
        drop(state);
        self.shared.wake.notify_all();
    }

    /// Stop decoding between frames, or carry on.
    pub fn set_paused(&self, paused: bool) {
        self.shared.lock().paused = paused;
        self.shared.wake.notify_all();
    }

    /// The upright picture size of `path`, once the worker has opened it.
    #[must_use]
    pub fn shape(&self, path: &std::path::Path) -> Option<(u32, u32)> {
        self.shared.lock().shapes.get(path).copied()
    }

    /// Asks still waiting for the worker.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.shared.lock().wanted.len()
    }
}

impl Drop for Thumbnails {
    fn drop(&mut self) {
        self.shared.lock().closed = true;
        self.shared.wake.notify_all();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn work(shared: &Shared, factory: &ReaderFactory, ready: &dyn Fn()) {
    let mut open: VecDeque<(PathBuf, Media)> = VecDeque::new();
    loop {
        let key = {
            let mut state = shared.lock();
            loop {
                if state.closed {
                    return;
                }
                if !state.paused
                    && let Some(key) = state.wanted.pop_front()
                {
                    break key;
                }
                state = shared
                    .wake
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner);
            }
        };
        let made = thumb_for(factory, &mut open, &key);
        let mut state = shared.lock();
        if let Some((path, media)) = open.front() {
            let info = media.info();
            if let (Some(width), Some(height)) = (info.width, info.height) {
                state
                    .shapes
                    .insert(path.clone(), upright(width, height, info.display));
            }
        }
        match made {
            Ok(thumb) => {
                state.stored.insert(key.clone(), Arc::new(thumb));
                state.order.push_back(key);
                while state.order.len() > MAX_THUMBS {
                    if let Some(oldest) = state.order.pop_front() {
                        state.stored.remove(&oldest);
                    }
                }
            }
            Err(_) => {
                state.failed.insert(key);
            }
        }
        drop(state);
        ready();
    }
}

fn thumb_for(
    factory: &ReaderFactory,
    open: &mut VecDeque<(PathBuf, Media)>,
    key: &ThumbKey,
) -> Result<Thumb, EngineError> {
    let index = match open.iter().position(|(path, _)| path == &key.path) {
        Some(index) => index,
        None => {
            if open.len() == OPEN_FILES {
                open.pop_back();
            }
            open.push_front((key.path.clone(), factory.open(&key.path)?));
            0
        }
    };
    let Some(entry) = open.remove(index) else {
        return Err(EngineError::NotReady);
    };
    open.push_front(entry);
    let Some((_, media)) = open.front_mut() else {
        return Err(EngineError::NotReady);
    };
    let length = media
        .info()
        .duration
        .map_or(i64::MAX / 2, |end| end.0.max(1));
    let display = media.info().display;
    let held = media.decode_at(key.time, &TimeRange::new(0, length))?;
    shrink(&held.frame, display)
}

/// `width` by `height` after a display quarter turn.
fn upright(width: u32, height: u32, display: DisplayQuarter) -> (u32, u32) {
    match display {
        DisplayQuarter::Clockwise90 | DisplayQuarter::Clockwise270 => (height, width),
        DisplayQuarter::Zero | DisplayQuarter::Half => (width, height),
    }
}

/// The rows a picture `height` tall is shrunk to, and the width that keeps
/// its shape.
fn thumb_size(width: u32, height: u32) -> Option<(u32, u32)> {
    if width == 0 || height == 0 {
        return None;
    }
    let rows = height.min(THUMB_HEIGHT);
    let columns = (u64::from(width) * u64::from(rows) + u64::from(height) / 2) / u64::from(height);
    Some((u32::try_from(columns.max(1)).ok()?, rows))
}

/// `frame` shrunk to a thumbnail and turned upright. The thumbnail height
/// is counted upright. A planar 4:2:0 picture is averaged in YUV and only
/// the small result is converted.
fn shrink(frame: &DecodedFrame, display: DisplayQuarter) -> Result<Thumb, EngineError> {
    let (coded_w, coded_h) = (frame.width(), frame.height());
    let (up_w, up_h) = upright(coded_w, coded_h, display);
    let (out_w, out_h) = thumb_size(up_w, up_h).ok_or(EngineError::Window)?;
    let size = upright(out_w, out_h, display);
    if let DecodedFrame::Coded {
        width,
        height,
        pixel: PixelFormat::Yuv420p,
        bytes,
        ..
    } = frame
        && let Some(thumb) = shrink_yuv(*width, *height, bytes, size)
    {
        return Ok(turn(thumb, display));
    }
    let Picture::Cpu(picture) = decoded_picture(frame)? else {
        return Err(EngineError::NotReady);
    };
    let (width, height) = size;
    let mut rgba = Vec::with_capacity(rgba_len(width, height).ok_or(EngineError::Window)?);
    for y in 0..height {
        for x in 0..width {
            for channel in 0..4 {
                rgba.push(average(
                    picture.rgba(),
                    (picture.width(), picture.height()),
                    4,
                    channel,
                    (x, y),
                    (width, height),
                ));
            }
        }
    }
    Ok(turn(
        Thumb {
            width,
            height,
            rgba,
        },
        display,
    ))
}

/// `thumb` turned clockwise by `display`.
fn turn(thumb: Thumb, display: DisplayQuarter) -> Thumb {
    if display == DisplayQuarter::Zero {
        return thumb;
    }
    let (src_w, src_h) = (thumb.width as usize, thumb.height as usize);
    let (width, height) = upright(thumb.width, thumb.height, display);
    let mut rgba = Vec::with_capacity(thumb.rgba.len());
    for y in 0..height as usize {
        for x in 0..width as usize {
            let (sx, sy) = match display {
                DisplayQuarter::Zero => (x, y),
                DisplayQuarter::Clockwise90 => (y, src_h - 1 - x),
                DisplayQuarter::Half => (src_w - 1 - x, src_h - 1 - y),
                DisplayQuarter::Clockwise270 => (src_w - 1 - y, x),
            };
            let at = (sy * src_w + sx) * 4;
            rgba.extend_from_slice(&thumb.rgba[at..at + 4]);
        }
    }
    Thumb {
        width,
        height,
        rgba,
    }
}

fn shrink_yuv(width: u32, height: u32, bytes: &[u8], (out_w, out_h): (u32, u32)) -> Option<Thumb> {
    let luma = usize::try_from(width).ok()? * usize::try_from(height).ok()?;
    let (chroma_w, chroma_h) = (width / 2, height / 2);
    let chroma = usize::try_from(chroma_w).ok()? * usize::try_from(chroma_h).ok()?;
    if chroma == 0 || bytes.len() != luma + 2 * chroma {
        return None;
    }
    let (y_plane, rest) = bytes.split_at(luma);
    let (u_plane, v_plane) = rest.split_at(chroma);
    let mut rgba = Vec::with_capacity(rgba_len(out_w, out_h)?);
    for y in 0..out_h {
        for x in 0..out_w {
            let cell = (x, y);
            let size = (out_w, out_h);
            let luma = average(y_plane, (width, height), 1, 0, cell, size);
            let cb = average(u_plane, (chroma_w, chroma_h), 1, 0, cell, size);
            let cr = average(v_plane, (chroma_w, chroma_h), 1, 0, cell, size);
            rgba.extend_from_slice(&bt709(luma, cb, cr));
        }
    }
    Some(Thumb {
        width: out_w,
        height: out_h,
        rgba,
    })
}

/// The mean of `channel` over the source block under output pixel `cell`.
/// A wide block is sampled on a grid of at most 4 by 4.
fn average(
    plane: &[u8],
    (src_w, src_h): (u32, u32),
    stride: usize,
    channel: usize,
    (x, y): (u32, u32),
    (out_w, out_h): (u32, u32),
) -> u8 {
    let span = |at: u32, out: u32, src: u32| {
        let from = u64::from(at) * u64::from(src) / u64::from(out);
        let to = (u64::from(at + 1) * u64::from(src) / u64::from(out)).max(from + 1);
        (from as usize, to.min(u64::from(src)) as usize)
    };
    let (x0, x1) = span(x, out_w, src_w);
    let (y0, y1) = span(y, out_h, src_h);
    let step_x = ((x1 - x0) / 4).max(1);
    let step_y = ((y1 - y0) / 4).max(1);
    let row_len = src_w as usize * stride;
    let (mut sum, mut count) = (0_u32, 0_u32);
    for row in (y0..y1).step_by(step_y) {
        for column in (x0..x1).step_by(step_x) {
            if let Some(value) = plane.get(row * row_len + column * stride + channel) {
                sum += u32::from(*value);
                count += 1;
            }
        }
    }
    if count == 0 {
        return 0;
    }
    u8::try_from((sum + count / 2) / count).unwrap_or(u8::MAX)
}

/// Limited-range BT.709 to RGBA, as the compositor converts.
fn bt709(luma: u8, cb: u8, cr: u8) -> [u8; 4] {
    let y = (f32::from(luma) - 16.0) / 219.0;
    let cb = (f32::from(cb) - 128.0) / 224.0;
    let cr = (f32::from(cr) - 128.0) / 224.0;
    let (kr, kb) = (0.2126_f32, 0.0722_f32);
    let kg = 1.0 - kr - kb;
    let red = y + 2.0 * (1.0 - kr) * cr;
    let green = y - 2.0 * kb * (1.0 - kb) / kg * cb - 2.0 * kr * (1.0 - kr) / kg * cr;
    let blue = y + 2.0 * (1.0 - kb) * cb;
    let byte = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u8;
    [byte(red), byte(green), byte(blue), 255]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tall_picture_shrinks_to_the_thumb_height_and_keeps_its_shape() {
        assert_eq!(thumb_size(1920, 1080), Some((171, 96)));
        assert_eq!(thumb_size(720, 1280), Some((54, 96)));
        assert_eq!(thumb_size(40, 20), Some((40, 20)));
        assert_eq!(thumb_size(0, 20), None);
    }

    #[test]
    fn a_planar_picture_is_averaged_then_converted() {
        // 4x2, left half white, right half black, no color.
        let (width, height) = (4, 2);
        let mut bytes = vec![235, 235, 16, 16, 235, 235, 16, 16];
        bytes.extend([128, 128, 128, 128]);
        let thumb = shrink_yuv(width, height, &bytes, (4, 2)).expect("thumb");
        assert_eq!((thumb.width, thumb.height), (4, 2));
        assert_eq!(&thumb.rgba[..4], &[255, 255, 255, 255]);
        assert_eq!(&thumb.rgba[8..12], &[0, 0, 0, 255]);
        assert!(shrink_yuv(width, height, &bytes[..10], (4, 2)).is_none());
    }

    #[test]
    fn a_quarter_turn_stands_the_thumb_up() {
        // 2x1: red then blue. Turned clockwise, red is on top.
        let flat = Thumb {
            width: 2,
            height: 1,
            rgba: vec![255, 0, 0, 255, 0, 0, 255, 255],
        };
        let up = turn(flat.clone(), DisplayQuarter::Clockwise90);
        assert_eq!((up.width, up.height), (1, 2));
        assert_eq!(up.rgba, [255, 0, 0, 255, 0, 0, 255, 255]);
        let down = turn(flat.clone(), DisplayQuarter::Clockwise270);
        assert_eq!(down.rgba, [0, 0, 255, 255, 255, 0, 0, 255]);
        assert_eq!(turn(flat.clone(), DisplayQuarter::Zero), flat);
        // A sideways 1280x720 file stands 96 rows tall.
        let (up_w, up_h) = upright(1280, 720, DisplayQuarter::Clockwise90);
        assert_eq!(thumb_size(up_w, up_h), Some((54, 96)));
    }

    #[test]
    fn an_average_covers_the_whole_block() {
        let plane = [0, 100, 200, 60];
        assert_eq!(average(&plane, (4, 1), 1, 0, (0, 0), (2, 1)), 50);
        assert_eq!(average(&plane, (4, 1), 1, 0, (1, 0), (2, 1)), 130);
    }

    #[test]
    fn a_missing_file_fails_once_and_is_not_asked_again() {
        let (sender, receiver) = std::sync::mpsc::channel();
        let thumbs = Thumbnails::start(ReaderFactory::new(), move || {
            let _ = sender.send(());
        });
        let key = ThumbKey {
            path: PathBuf::from("/nowhere/missing.mp4"),
            time: Micros(0),
        };
        thumbs.want([key.clone(), key.clone()]);
        receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("the worker answered");
        assert!(thumbs.get(&key).is_none());
        thumbs.want([key]);
        assert_eq!(thumbs.pending(), 0);
    }

    #[test]
    fn a_paused_store_keeps_its_asks() {
        let thumbs = Thumbnails::start(ReaderFactory::new(), || {});
        thumbs.set_paused(true);
        let key = ThumbKey {
            path: PathBuf::from("/nowhere/missing.mp4"),
            time: Micros(0),
        };
        thumbs.want([key]);
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert_eq!(thumbs.pending(), 1);
    }
}
