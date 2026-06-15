//! Library tile thumbnails: poster frames for video, waveform images for
//! audio, the picture itself for stills.
//!
//! Generation (decode + scale, or full audio decode for peaks) runs on a
//! dedicated worker thread so neither the UI nor the preview/engine thread
//! stalls after an import. Finished images land in a UI-thread registry that
//! the projection reads on every publish, and the live `EditorStore` media
//! model is patched in place so tiles update without waiting for the next
//! engine publish.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::thread::JoinHandle;

use crossbeam_channel::{Sender, unbounded};
use cutlass_decoder::{audio_peaks, decode_image, video_thumbnail};
use slint::{Image, Model, Rgba8Pixel, SharedPixelBuffer};
use tracing::{error, info};

use crate::EditorStore;

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
    pub fn spawn(editor_weak: slint::Weak<EditorStore<'static>>) -> Result<Self, String> {
        let (tx, rx) = unbounded::<ThumbRequest>();
        let join = std::thread::Builder::new()
            .name("cutlass-thumbs".into())
            .spawn(move || {
                while let Ok(req) = rx.recv() {
                    match generate(&req) {
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

fn generate(req: &ThumbRequest) -> Result<(u32, u32, Vec<u8>), String> {
    match req.kind {
        ThumbKind::Video => {
            let thumb = video_thumbnail(&req.path, VIDEO_THUMB_MAX, VIDEO_THUMB_MAX)
                .map_err(|e| e.to_string())?;
            Ok((thumb.width, thumb.height, thumb.rgba))
        }
        ThumbKind::Audio => {
            let peaks = audio_peaks(&req.path, WAVEFORM_BARS).map_err(|e| e.to_string())?;
            Ok(render_waveform(&peaks, WAVEFORM_SIZE, WAVEFORM_SIZE))
        }
        ThumbKind::Image => {
            let thumb = decode_image(&req.path, VIDEO_THUMB_MAX, VIDEO_THUMB_MAX)
                .map_err(|e| e.to_string())?;
            Ok((thumb.width, thumb.height, thumb.rgba))
        }
    }
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
}
