//! Library tile thumbnails: poster frames for video, waveform images for
//! audio, the picture itself for stills.
//!
//! Generation runs on a dedicated worker thread so neither the UI nor the
//! preview/engine thread stalls after an import. Finished images land in a
//! UI-thread registry that the projection reads on every publish, and the
//! live `EditorStore` media model is patched in place so tiles update
//! without waiting for the next engine publish.
//!
//! Visual thumbnails ride the mobile thumbnailer pattern instead of main's
//! bespoke FFmpeg poster path: probe → scratch single-clip project → private
//! headless [`Renderer`] → `render_frame_fit`. Frames take the same
//! resolve/render road as the editor, so orientation/aspect handling can
//! never drift from what the preview shows. The worker keeps one renderer
//! (own GPU queue + decoder cache) across requests; each media decodes once
//! for its poster and the pool size stays library-bounded.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use crossbeam_channel::{RecvTimeoutError, Sender, unbounded};
use cutlass_models::{MediaSource, Project, Rational, RationalTime, TimeRange, TrackKind};
use cutlass_render::Renderer;
use slint::{Image, Model, Rgba8Pixel, SharedPixelBuffer};
use tracing::{error, info};

use crate::EditorStore;
use crate::interaction::InteractionGate;

/// While the interaction gate is closed, poll for new requests at this
/// cadence instead of generating — keeps the queue draining without decode
/// or GPU work competing with the preview.
const GATE_POLL: Duration = Duration::from_millis(100);

/// Box for video poster frames: 2× the 100px library tile for hidpi.
const VIDEO_THUMB_MAX: u32 = 256;
/// Waveform images are square like the tile (drawn 2× for hidpi).
const WAVEFORM_SIZE: u32 = 200;
/// One vertical bar per bucket across the waveform image.
const WAVEFORM_BARS: usize = 40;

/// CapCut-style waveform palette: blue card, lighter bars.
const WAVEFORM_BG: [u8; 4] = [0x2A, 0x46, 0xC8, 0xFF];
const WAVEFORM_BAR: [u8; 4] = [0xA9, 0xC0, 0xFF, 0xFF];

#[derive(Debug, Clone, Copy)]
pub enum ThumbKind {
    Video,
    Audio,
    /// Still image: the tile shows the picture itself (no poster-frame seek).
    Image,
}

struct ThumbRequest {
    media_id: u64,
    path: PathBuf,
    kind: ThumbKind,
}

/// Cheap, cloneable sender to the thumbnail thread.
#[derive(Clone)]
pub struct ThumbnailHandle {
    tx: Sender<ThumbRequest>,
}

impl ThumbnailHandle {
    pub fn request(&self, media_id: u64, path: PathBuf, kind: ThumbKind) {
        let _ = self.tx.send(ThumbRequest {
            media_id,
            path,
            kind,
        });
    }
}

pub struct ThumbnailWorker {
    handle: ThumbnailHandle,
    _join: JoinHandle<()>,
}

impl ThumbnailWorker {
    pub fn spawn(
        editor_weak: slint::Weak<EditorStore<'static>>,
        gate: Arc<InteractionGate>,
    ) -> Result<Self, String> {
        let (tx, rx) = unbounded::<ThumbRequest>();
        let join = std::thread::Builder::new()
            .name("cutlass-thumbs".into())
            .spawn(move || {
                // One renderer for the thread's lifetime: GPU bring-up is paid
                // once, not per import. Lazy so a machine whose GPU can't spare
                // another queue still runs (video tiles keep placeholders).
                let mut renderer: Option<Renderer> = None;
                // Requests deferred while the user scrubs/plays (see
                // `InteractionGate`): decode + composite here would steal
                // exactly the hardware the preview is using. FIFO — deferred,
                // never dropped.
                let mut pending: VecDeque<ThumbRequest> = VecDeque::new();
                loop {
                    if pending.is_empty() {
                        match rx.recv() {
                            Ok(req) => pending.push_back(req),
                            Err(_) => return,
                        }
                    }
                    while let Ok(req) = rx.try_recv() {
                        pending.push_back(req);
                    }
                    if gate.busy() {
                        match rx.recv_timeout(GATE_POLL) {
                            Ok(req) => pending.push_back(req),
                            Err(RecvTimeoutError::Timeout) => {}
                            Err(RecvTimeoutError::Disconnected) => return,
                        }
                        continue;
                    }
                    let Some(req) = pending.pop_front() else {
                        continue;
                    };
                    match generate(&req, &mut renderer) {
                        Ok((width, height, rgba)) => {
                            info!(media_id = req.media_id, width, height, kind = ?req.kind, "thumbnail ready");
                            deliver(req.media_id, width, height, rgba, &editor_weak);
                        }
                        Err(e) => {
                            error!(media_id = req.media_id, path = %req.path.display(), "thumbnail failed: {e}")
                        }
                    }
                }
            })
            .map_err(|e| e.to_string())?;

        Ok(Self {
            handle: ThumbnailHandle { tx },
            _join: join,
        })
    }

    pub fn handle(&self) -> ThumbnailHandle {
        self.handle.clone()
    }
}

fn generate(
    req: &ThumbRequest,
    renderer: &mut Option<Renderer>,
) -> Result<(u32, u32, Vec<u8>), String> {
    match req.kind {
        ThumbKind::Video | ThumbKind::Image => {
            let renderer = match renderer {
                Some(renderer) => renderer,
                None => renderer.insert(Renderer::new_headless().map_err(|e| e.to_string())?),
            };
            let image = render_poster(renderer, &req.path, VIDEO_THUMB_MAX, VIDEO_THUMB_MAX)
                .map_err(|e| e.to_string())?;
            Ok((image.width, image.height, image.pixels))
        }
        ThumbKind::Audio => {
            let peaks = cutlass_decoder::audio_peaks(&req.path, WAVEFORM_BARS)
                .map_err(|e| e.to_string())?;
            Ok(render_waveform(&peaks, WAVEFORM_SIZE, WAVEFORM_SIZE))
        }
    }
}

/// Fit-render one poster frame of the media at `path` through a scratch
/// single-clip project — the mobile thumbnailer pattern. The target is ~10%
/// into the file (capped at 3s) to skip black/fade-in openings; stills have
/// one cached frame at any tick.
fn render_poster(
    renderer: &mut Renderer,
    path: &std::path::Path,
    max_w: u32,
    max_h: u32,
) -> Result<cutlass_render::RgbaImage, String> {
    let probe = cutlass_decoder::probe(path).map_err(|e| e.to_string())?;
    let source = if probe.is_image {
        MediaSource::image(path, probe.width, probe.height)
    } else {
        MediaSource::new(
            path,
            probe.width,
            probe.height,
            probe.frame_rate,
            probe.frame_count.max(1),
            probe.has_audio,
        )
    };
    let frame_rate = source.frame_rate;
    let frames = source.duration.value.max(1);

    let mut project = Project::new("thumbnail", frame_rate);
    let media = project.add_media(source);
    let track = project.add_track(TrackKind::Video, "Media");
    project
        .add_clip(
            track,
            media,
            TimeRange::at_rate(0, frames, frame_rate),
            RationalTime::new(0, frame_rate),
        )
        .map_err(|e| e.to_string())?;

    let poster = poster_tick(frames, frame_rate);
    renderer
        .render_frame_fit(
            &project,
            RationalTime::new(poster, frame_rate),
            max_w,
            max_h,
        )
        .map_err(|e| e.to_string())
}

/// Poster-frame tick: ~10% into the source, capped at 3 seconds.
fn poster_tick(duration_frames: i64, rate: Rational) -> i64 {
    if rate.num <= 0 || rate.den <= 0 {
        return 0;
    }
    let three_seconds = 3 * i64::from(rate.num) / i64::from(rate.den);
    (duration_frames / 10)
        .min(three_seconds)
        .clamp(0, (duration_frames - 1).max(0))
}

/// Hand a finished RGBA thumbnail to the UI thread: build the `slint::Image`
/// there (images are `!Send`), record it for future projection publishes, and
/// patch the currently displayed media model in place.
fn deliver(
    media_id: u64,
    width: u32,
    height: u32,
    rgba: Vec<u8>,
    editor_weak: &slint::Weak<EditorStore<'static>>,
) {
    let editor_weak = editor_weak.clone();
    if let Err(e) = slint::invoke_from_event_loop(move || {
        let buffer = SharedPixelBuffer::<Rgba8Pixel>::clone_from_slice(&rgba, width, height);
        let image = Image::from_rgba8(buffer);
        THUMBS.with(|thumbs| thumbs.borrow_mut().insert(media_id, image.clone()));

        let Some(store) = editor_weak.upgrade() else {
            return;
        };
        let media_model = store.get_project().media;
        let id = slint::SharedString::from(media_id.to_string());
        for row in 0..media_model.row_count() {
            if let Some(mut media) = media_model.row_data(row)
                && media.id == id
            {
                media.thumbnail = image.clone();
                media_model.set_row_data(row, media);
            }
        }
    }) {
        error!(media_id, "failed to deliver thumbnail to UI: {e}");
    }
}

thread_local! {
    /// UI-thread registry of finished thumbnails, keyed by raw media id.
    /// Projection publishes rebuild the Slint media model from the engine
    /// snapshot; this keeps already-generated images across those rebuilds.
    static THUMBS: RefCell<HashMap<u64, Image>> = RefCell::new(HashMap::new());
}

/// The generated thumbnail for `media_id`, if it's ready (UI thread only).
pub fn thumbnail_for(media_id: u64) -> Option<Image> {
    THUMBS.with(|thumbs| thumbs.borrow().get(&media_id).cloned())
}

/// Drop a deleted source's cached thumbnail so its texture isn't held for the
/// rest of the session. Safe to call from any thread — the eviction hops onto
/// the UI thread, where `THUMBS` lives. A no-op id (never imported, or already
/// forgotten) just clears nothing.
pub fn forget(media_id: u64) {
    if let Err(e) = slint::invoke_from_event_loop(move || {
        THUMBS.with(|thumbs| thumbs.borrow_mut().remove(&media_id));
    }) {
        error!(media_id, "failed to evict thumbnail on delete: {e}");
    }
}

/// Draw mirrored peak bars onto a solid card, CapCut-library style.
fn render_waveform(peaks: &[f32], width: u32, height: u32) -> (u32, u32, Vec<u8>) {
    let w = width as usize;
    let h = height as usize;
    let mut rgba = WAVEFORM_BG.repeat(w * h);

    if !peaks.is_empty() {
        let slot = (w / peaks.len()).max(1);
        let bar_w = (slot * 3 / 5).max(1);
        let mid = h / 2;
        // Loudest bar reaches ~90% of the half-height; quiet ones stay visible.
        let max_half = (h / 2).saturating_sub(h / 10).max(1);
        let min_half = (h / 50).max(1);

        for (i, peak) in peaks.iter().enumerate() {
            let x0 = i * slot + (slot - bar_w) / 2;
            if x0 + bar_w > w {
                break;
            }
            let half = ((f64::from(peak.clamp(0.0, 1.0)) * max_half as f64) as usize)
                .clamp(min_half, max_half);
            for y in mid.saturating_sub(half)..(mid + half).min(h) {
                let row = y * w * 4;
                for x in x0..x0 + bar_w {
                    let px = row + x * 4;
                    rgba[px..px + 4].copy_from_slice(&WAVEFORM_BAR);
                }
            }
        }
    }

    (width, height, rgba)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn waveform_paints_bars_on_background() {
        let (w, h, rgba) = render_waveform(&[0.0, 0.5, 1.0], 50, 50);
        assert_eq!((w, h), (50, 50));
        assert_eq!(rgba.len(), 50 * 50 * 4);

        let count = |color: [u8; 4]| rgba.chunks_exact(4).filter(|px| *px == color).count();
        let bar = count(WAVEFORM_BAR);
        let bg = count(WAVEFORM_BG);
        assert!(bar > 0, "bars should be painted");
        assert!(bg > 0, "background should remain visible");
        assert_eq!(bar + bg, 50 * 50, "only palette colors are painted");
    }

    #[test]
    fn waveform_with_no_peaks_is_solid_card() {
        let (_, _, rgba) = render_waveform(&[], 10, 10);
        assert!(rgba.chunks_exact(4).all(|px| px == WAVEFORM_BG));
    }

    #[test]
    fn louder_peaks_paint_taller_bars() {
        let quiet = render_waveform(&[0.1], 20, 100).2;
        let loud = render_waveform(&[1.0], 20, 100).2;
        let bars = |buf: &[u8]| buf.chunks_exact(4).filter(|px| *px == WAVEFORM_BAR).count();
        assert!(bars(&loud) > bars(&quiet));
    }

    #[test]
    fn poster_tick_skips_the_opening_but_stays_in_range() {
        let r24 = Rational::FPS_24;
        // 10% of a 100-frame clip.
        assert_eq!(poster_tick(100, r24), 10);
        // Long clip: capped at 3s (72 ticks at 24fps).
        assert_eq!(poster_tick(24_000, r24), 72);
        // Single frame / empty: stays at 0.
        assert_eq!(poster_tick(1, r24), 0);
        assert_eq!(poster_tick(0, r24), 0);
    }
}
