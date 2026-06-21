//! Per-clip decoder reuse for preview and export.
//!
//! Decoders are keyed by [`ClipId`], not media: two clips of the *same* source
//! sit at different timeline offsets, so each frame needs two different read
//! positions in the file. A single shared decoder would seek backward on every
//! frame — each seek flushes and re-decodes a whole GOP prefix, turning a
//! linear export into O(GOP) work per layer per frame. A decoder per clip keeps
//! every read cursor rolling forward. The keyframe index *is* shared per media
//! (built once — it's a full-file demux scan).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cutlass_decoder::{DecodeOptions, Decoder, HwAccel, KeyframeIndex, STILL_MAX_DIM};
use cutlass_models::{ClipId, MediaId};

use crate::error::EngineError;

/// Hard cap on simultaneously open per-clip decoders. Reconciling against the
/// live timeline ([`DecoderPool::retain_clips`]) bounds the pool to the clip
/// count in normal use; this is the backstop for sessions that touch more
/// distinct clips between edits than that — e.g. scrubbing a very long
/// timeline, where every clip passed under the playhead opens a decoder with
/// no edit in between to trigger reconciliation. Each software decoder holds
/// tens of MB of FFmpeg buffers, so the cap trades a re-open on the rare
/// scroll-back for a firm memory ceiling.
const MAX_DECODERS: usize = 64;

struct Entry {
    path: PathBuf,
    decoder: Decoder,
    index: Arc<KeyframeIndex>,
    /// Monotonic touch stamp for LRU eviction (see [`DecoderPool::use_tick`]).
    last_used: u64,
}

/// One decoded still image, shared by every composite that shows it.
/// The `Arc` is what `CompositeLayer::rgba` wants, so re-showing a still
/// is a refcount bump — no copy, no re-decode.
struct StillEntry {
    path: PathBuf,
    bytes: Arc<Vec<u8>>,
    width: u32,
    height: u32,
}

pub struct DecoderPool {
    entries: HashMap<ClipId, Entry>,
    indices: HashMap<MediaId, Arc<KeyframeIndex>>,
    stills: HashMap<MediaId, StillEntry>,
    options: DecodeOptions,
    /// Monotonic counter stamped onto an entry's `last_used` on every access,
    /// so the smallest stamp is the least-recently-used decoder to evict.
    use_tick: u64,
}

impl DecoderPool {
    pub fn new() -> Self {
        Self::with_hw_accel(HwAccel::None)
    }

    /// Pool that decodes through `hw_accel` (e.g. [`HwAccel::Auto`] to probe the
    /// platform media engine — VideoToolbox / NVDEC / VA-API — and fall back to
    /// software if none is usable). Hardware-decoded frames come back as NV12,
    /// which the engine's `decoded_to_yuv_layer` deinterleaves to YUV420P.
    ///
    /// Used by export so big 4K renders offload decode off the CPU. Preview
    /// stays on [`HwAccel::None`] ([`Self::new`]): its YUV scrub cache packs
    /// YUV420P planes and isn't fed NV12.
    pub fn with_hw_accel(hw_accel: HwAccel) -> Self {
        Self {
            entries: HashMap::new(),
            indices: HashMap::new(),
            stills: HashMap::new(),
            options: DecodeOptions::default().hw_accel(hw_accel),
            use_tick: 0,
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.indices.clear();
        self.stills.clear();
    }

    /// Drop the decoder of every clip no longer on the timeline. Decoders are
    /// keyed by [`ClipId`] and otherwise live for the whole session (only a
    /// New/Open [`clear`](Self::clear)s them), so a deleted, split, trimmed
    /// (split mints fresh ids), or undone clip would otherwise leak its
    /// decoder — and the megabytes of FFmpeg decode/reference buffers behind
    /// it — until the next project swap. The engine calls this after every
    /// edit that can remove a clip. Keyframe indices are keyed by media (cheap,
    /// bounded by distinct sources) and left untouched.
    pub fn retain_clips(&mut self, live: &HashSet<ClipId>) {
        self.entries.retain(|clip_id, _| live.contains(clip_id));
    }

    /// The decoder dedicated to `clip` and the keyframe index for its backing
    /// `media`. The decoder is per clip (so overlapping clips of one source
    /// don't fight over a single read cursor); the index is built once per
    /// media and shared.
    pub fn decoder_and_index(
        &mut self,
        clip_id: ClipId,
        media_id: MediaId,
        path: &Path,
    ) -> Result<(&mut Decoder, &KeyframeIndex), EngineError> {
        // Build the keyframe index once per media (a full-file demux scan) and
        // share it across every clip that reads this source.
        if let std::collections::hash_map::Entry::Vacant(slot) = self.indices.entry(media_id) {
            slot.insert(Arc::new(KeyframeIndex::build(path)?));
        }

        let stale = self.entries.get(&clip_id).is_none_or(|e| e.path != path);
        if stale {
            let decoder = Decoder::open_with(path, self.options)?;
            let index = Arc::clone(&self.indices[&media_id]);
            self.entries.insert(
                clip_id,
                Entry {
                    path: path.to_path_buf(),
                    decoder,
                    index,
                    last_used: 0,
                },
            );
        }

        // Touch first (so the requested clip is the most-recently-used and can
        // never be the eviction victim), then trim back to the cap.
        self.use_tick += 1;
        let tick = self.use_tick;
        self.entries
            .get_mut(&clip_id)
            .expect("just inserted")
            .last_used = tick;
        self.evict_over_cap(clip_id);

        let entry = self.entries.get_mut(&clip_id).expect("just inserted");
        Ok((&mut entry.decoder, &*entry.index))
    }

    /// Drop least-recently-used decoders until the pool is back at the cap,
    /// never evicting `keep` (the clip whose decoder the caller is about to
    /// use). O(n) per drop with n ≤ [`MAX_DECODERS`]; only runs when the cap is
    /// already exceeded, so it's off every steady-state hot path.
    fn evict_over_cap(&mut self, keep: ClipId) {
        while self.entries.len() > MAX_DECODERS {
            let victim = self
                .entries
                .iter()
                .filter(|(id, _)| **id != keep)
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(id, _)| *id);
            match victim {
                Some(id) => {
                    self.entries.remove(&id);
                }
                None => break,
            }
        }
    }

    /// The decoded RGBA for a still-image media, decoding on first use
    /// (capped to [`STILL_MAX_DIM`] per side; the GPU scales into place).
    /// Returns `(bytes, width, height)`.
    pub fn still(
        &mut self,
        media_id: MediaId,
        path: &Path,
    ) -> Result<(Arc<Vec<u8>>, u32, u32), EngineError> {
        let stale = self.stills.get(&media_id).is_none_or(|e| e.path != path);

        if stale {
            let image = cutlass_decoder::decode_image(path, STILL_MAX_DIM, STILL_MAX_DIM)?;
            self.stills.insert(
                media_id,
                StillEntry {
                    path: path.to_path_buf(),
                    bytes: Arc::new(image.rgba),
                    width: image.width,
                    height: image.height,
                },
            );
        }

        let entry = self.stills.get(&media_id).expect("just inserted");
        Ok((Arc::clone(&entry.bytes), entry.width, entry.height))
    }

    /// Number of open video decoders (one per distinct clip seen).
    #[cfg(test)]
    pub(crate) fn decoder_count(&self) -> usize {
        self.entries.len()
    }

    /// Number of keyframe indices held (one per distinct media seen).
    #[cfg(test)]
    pub(crate) fn index_count(&self) -> usize {
        self.indices.len()
    }
}

impl Default for DecoderPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_models::MediaId;
    use std::path::PathBuf;

    fn sample_video() -> Option<PathBuf> {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../local-assets/assets");
        std::fs::read_dir(dir)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|ext| ext == "mp4"))
    }

    #[test]
    fn distinct_clips_of_same_media_get_distinct_decoders_but_share_index() {
        let Some(path) = sample_video() else {
            return; // media scratch not present (CI without assets)
        };
        let mut pool = DecoderPool::new();
        let media = MediaId::from_raw(1);
        let clip_a = ClipId::from_raw(10);
        let clip_b = ClipId::from_raw(20);

        pool.decoder_and_index(clip_a, media, &path).unwrap();
        pool.decoder_and_index(clip_b, media, &path).unwrap();
        // Re-request clip A: must reuse, not reopen.
        pool.decoder_and_index(clip_a, media, &path).unwrap();

        assert_eq!(
            pool.decoder_count(),
            2,
            "each clip should own a decoder (no shared read cursor)"
        );
        assert_eq!(
            pool.index_count(),
            1,
            "the keyframe index is built once per media and shared"
        );
    }

    #[test]
    fn retain_clips_evicts_decoders_for_clips_no_longer_on_the_timeline() {
        let Some(path) = sample_video() else {
            return;
        };
        let mut pool = DecoderPool::new();
        let media = MediaId::from_raw(1);
        let keep = ClipId::from_raw(1);
        let gone = ClipId::from_raw(2);

        pool.decoder_and_index(keep, media, &path).unwrap();
        pool.decoder_and_index(gone, media, &path).unwrap();
        assert_eq!(pool.decoder_count(), 2);

        let live: HashSet<ClipId> = std::iter::once(keep).collect();
        pool.retain_clips(&live);
        assert_eq!(
            pool.decoder_count(),
            1,
            "the deleted clip's decoder is dropped, the live one kept"
        );

        // An empty timeline frees every decoder (the reported leak: GBs held
        // with nothing on the timeline).
        pool.retain_clips(&HashSet::new());
        assert_eq!(pool.decoder_count(), 0);
    }

    #[test]
    fn open_decoders_are_capped_keeping_the_most_recent() {
        let Some(path) = sample_video() else {
            return;
        };
        let mut pool = DecoderPool::new();
        let media = MediaId::from_raw(1);

        // Touch far more distinct clips than the cap, oldest-id first.
        let total = MAX_DECODERS as u64 + 16;
        for i in 0..total {
            pool.decoder_and_index(ClipId::from_raw(i + 1), media, &path)
                .unwrap();
        }
        assert_eq!(
            pool.decoder_count(),
            MAX_DECODERS,
            "the pool never holds more than the cap"
        );

        // Re-touching the newest clip is a cache hit (no decoder churn).
        let newest = ClipId::from_raw(total);
        pool.decoder_and_index(newest, media, &path).unwrap();
        assert_eq!(pool.decoder_count(), MAX_DECODERS);
        assert!(
            pool.entries.contains_key(&newest),
            "the most-recently-used clip survives eviction"
        );
        assert!(
            !pool.entries.contains_key(&ClipId::from_raw(1)),
            "the oldest clip was evicted"
        );
    }

    #[test]
    fn clear_drops_decoders_and_indices() {
        let Some(path) = sample_video() else {
            return;
        };
        let mut pool = DecoderPool::new();
        pool.decoder_and_index(ClipId::from_raw(1), MediaId::from_raw(1), &path)
            .unwrap();
        pool.clear();
        assert_eq!(pool.decoder_count(), 0);
        assert_eq!(pool.index_count(), 0);
    }
}
