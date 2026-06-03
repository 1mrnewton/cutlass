//! Decoded-frame cache: the engine's primary defense against re-decoding.
//!
//! Scrubbing and playback repeatedly ask for the same source frames (a back-and-
//! forth scrub, a paused playhead, two clips referencing one media file). Decode
//! — especially a seek into a long-GOP stream — is orders of magnitude more
//! expensive than a hash lookup, so caching decoded frames is the single biggest
//! lever for a responsive timeline.
//!
//! # Why byte-bounded, not count-bounded
//!
//! A decoded frame's size depends on resolution and pixel format: a 4K YUV420P
//! frame is ~12 MB while an SD frame is a few hundred KB — a ~50× spread. A
//! fixed *count* would either blow the memory budget on 4K media or waste it on
//! SD. So the cache is bounded by **total bytes** and evicts the least-recently-
//! used entries until it fits.
//!
//! # Invalidation
//!
//! Keys are `(MediaId, source_frame)`. Source media is immutable, so timeline
//! edits (cut, trim, move) never invalidate a cached frame — only *removing* a
//! media file from the pool does (see [`FrameCache::invalidate_media`]).
//!
//! # Complexity
//!
//! `get`/`insert` are `O(log n)` in the number of cached frames (a `BTreeMap`
//! tracks recency order). For a cache holding hundreds-to-thousands of frames
//! that `log n` (~10) is negligible next to a single decode, and it buys a
//! simple, allocation-light implementation with no `unsafe`.
//!
//! # Why not `moka` (yet)
//!
//! [`moka`](https://crates.io/crates/moka) is a strong concurrent cache, but the
//! hand-rolled cache fits this workload better *today*:
//!
//! - **Recency beats frequency here.** A scrub buffer wants the frames near the
//!   playhead; moka's TinyLFU admission is frequency-biased and can retain a
//!   long-ago frame over the current one. Plain LRU matches the access pattern.
//! - **Exact, synchronous accounting.** moka evicts via background maintenance,
//!   so `used_bytes` is approximate and may briefly exceed the budget; ours is
//!   exact, which also keeps eviction unit-testable.
//! - **No concurrency need yet.** The engine is single-threaded. When prefetch
//!   moves to a background thread, a `Mutex<FrameCache>` is the cheap first step
//!   (the lock only wraps a hash lookup; decode — the costly part — stays
//!   outside it).
//!
//! Switch to `moka` when any of these becomes real: (1) multi-threaded prefetch
//! where a `Mutex<FrameCache>` shows measurable contention, (2) we want TTL /
//! time-to-idle expiry to release memory between edits, or (3) profiling shows
//! LRU hit rates are poor and a smarter admission policy would help. The public
//! API here (`get`, `get_or_try_insert_with`, `invalidate_media`, `stats`) maps
//! almost 1:1 onto moka, so the swap stays cheap.
//!
//! # Preview resolution must not enter the cache key (planned)
//!
//! The preview/viewport size is a *continuously varying* input (a resize drag
//! changes it every frame). The cache key must never depend on it, or every
//! resize would discard expensive decode work. Two rules for the future preview
//! path (which lives in the playback session + compositor, not here):
//!
//! - **Cache a fixed canonical resolution and scale to the viewport at draw
//!   time** (on the GPU). Resizing then only changes a scale factor — it never
//!   touches the cache. A ~1080p canonical tier is a good default: cheap to
//!   downscale for normal previews, only upscaled in the rare >1080p-fullscreen
//!   case. Export uses a separate full-res path.
//! - **If multiple resolutions are ever needed, make the tier part of the key**
//!   (quantized: 360/540/720/1080/full), so tiers coexist and stale ones **age
//!   out via LRU** instead of being explicitly invalidated. Additive keys + LRU,
//!   never destructive invalidation on a volatile UI input.

use std::collections::BTreeMap;
use std::sync::Arc;

use cutlass_decode::DecodedFrame;
use cutlass_models::{Map, MediaId};

/// Default cache budget: 256 MiB of decoded frames.
pub const DEFAULT_CAPACITY_BYTES: usize = 256 * 1024 * 1024;

/// Identifies one decoded frame: a source frame index within a media file.
///
/// `source_frame` is in the media's *native* frames, not timeline frames — the
/// engine converts timeline → source before touching the cache, so the same
/// decoded frame is shared across clips/projects regardless of timeline rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FrameKey {
    pub media: MediaId,
    pub source_frame: i64,
}

impl FrameKey {
    pub fn new(media: MediaId, source_frame: i64) -> Self {
        Self {
            media,
            source_frame,
        }
    }
}

/// Hit/miss counters for tuning cache size and prefetch behavior.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    /// Frames dropped to stay within the byte budget.
    pub evictions: u64,
}

impl CacheStats {
    /// Fraction of lookups served from cache, in `[0, 1]` (0 if no lookups yet).
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

struct Entry {
    frame: Arc<DecodedFrame>,
    bytes: usize,
    /// Recency tag; also the key into `lru`. Larger == more recently used.
    seq: u64,
}

/// A byte-bounded LRU cache of decoded frames.
///
/// Frames are stored behind [`Arc`] so a caller (e.g. the compositor) can hold a
/// frame cheaply while the engine keeps mutating the cache; eviction only drops
/// the cache's own reference, so an in-use frame stays alive until released.
pub struct FrameCache {
    capacity_bytes: usize,
    used_bytes: usize,
    next_seq: u64,
    entries: Map<FrameKey, Entry>,
    /// Recency index: `seq -> key`, ascending, so the front is the LRU victim.
    lru: BTreeMap<u64, FrameKey>,
    stats: CacheStats,
}

impl FrameCache {
    /// Create a cache that holds up to `capacity_bytes` of decoded frames.
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity_bytes,
            used_bytes: 0,
            next_seq: 0,
            entries: Map::default(),
            lru: BTreeMap::new(),
            stats: CacheStats::default(),
        }
    }

    /// Fetch a frame, marking it most-recently-used. Records a hit or miss.
    pub fn get(&mut self, key: FrameKey) -> Option<Arc<DecodedFrame>> {
        match self.entries.get_mut(&key) {
            Some(entry) => {
                // Re-tag as most-recently-used: move it to the back of `lru`.
                let old_seq = entry.seq;
                let seq = self.next_seq;
                self.next_seq += 1;
                entry.seq = seq;
                let frame = Arc::clone(&entry.frame);
                self.lru.remove(&old_seq);
                self.lru.insert(seq, key);
                self.stats.hits += 1;
                Some(frame)
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    /// Insert (or replace) a frame and return the stored [`Arc`].
    ///
    /// Evicts least-recently-used frames as needed to stay within the budget.
    /// A single frame larger than the whole budget is still kept (it cannot be
    /// served otherwise) but will be the first victim once anything else lands.
    pub fn insert(&mut self, key: FrameKey, frame: DecodedFrame) -> Arc<DecodedFrame> {
        let bytes = frame_bytes(&frame);
        let frame = Arc::new(frame);
        let seq = self.next_seq;
        self.next_seq += 1;

        if let Some(old) = self.entries.insert(
            key,
            Entry {
                frame: Arc::clone(&frame),
                bytes,
                seq,
            },
        ) {
            // Replacing an existing key: drop its byte accounting and stale tag.
            self.used_bytes -= old.bytes;
            self.lru.remove(&old.seq);
        }
        self.used_bytes += bytes;
        self.lru.insert(seq, key);

        self.evict_to_fit();
        frame
    }

    /// Return the cached frame, or produce, cache, and return it on a miss.
    ///
    /// The common decode path: `cache.get_or_try_insert_with(key, || decode(..))`.
    /// `produce` runs only on a miss, so decode work is never duplicated.
    pub fn get_or_try_insert_with<F, E>(
        &mut self,
        key: FrameKey,
        produce: F,
    ) -> Result<Arc<DecodedFrame>, E>
    where
        F: FnOnce() -> Result<DecodedFrame, E>,
    {
        if let Some(frame) = self.get(key) {
            return Ok(frame);
        }
        let frame = produce()?;
        Ok(self.insert(key, frame))
    }

    /// Whether `key` is currently cached (does not affect recency or stats).
    pub fn contains(&self, key: FrameKey) -> bool {
        self.entries.contains_key(&key)
    }

    /// Drop every frame belonging to `media` (e.g. when it leaves the pool).
    ///
    /// `O(n)` over the cache; intended for the cold media-management path.
    pub fn invalidate_media(&mut self, media: MediaId) {
        let victims: Vec<FrameKey> = self
            .entries
            .keys()
            .filter(|k| k.media == media)
            .copied()
            .collect();
        for key in victims {
            if let Some(entry) = self.entries.remove(&key) {
                self.used_bytes -= entry.bytes;
                self.lru.remove(&entry.seq);
            }
        }
    }

    /// Drop all cached frames (recency tags and stats are reset too).
    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru.clear();
        self.used_bytes = 0;
        self.next_seq = 0;
        self.stats = CacheStats::default();
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    pub fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    /// Evict LRU frames until within budget, keeping at least one frame so an
    /// over-budget frame remains usable for the caller that just inserted it.
    fn evict_to_fit(&mut self) {
        while self.used_bytes > self.capacity_bytes && self.entries.len() > 1 {
            // Front of `lru` is the smallest seq == least recently used.
            let Some((&seq, &key)) = self.lru.iter().next() else {
                break;
            };
            self.lru.remove(&seq);
            if let Some(entry) = self.entries.remove(&key) {
                self.used_bytes -= entry.bytes;
                self.stats.evictions += 1;
            }
        }
    }
}

impl Default for FrameCache {
    fn default() -> Self {
        Self::new(DEFAULT_CAPACITY_BYTES)
    }
}

/// Approximate heap footprint of a decoded frame: the pixel planes dominate;
/// the struct/`Vec` overhead is added so tiny frames aren't accounted as free.
fn frame_bytes(frame: &DecodedFrame) -> usize {
    let planes: usize = frame.planes.iter().map(|p| p.data.len()).sum();
    planes + std::mem::size_of::<DecodedFrame>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_decode::{PixelFormat, Plane};

    /// A frame whose payload is exactly `bytes` (single plane) for budget math.
    fn frame_of(bytes: usize, pts: i64) -> DecodedFrame {
        DecodedFrame {
            width: 2,
            height: 2,
            pts_ticks: pts,
            format: PixelFormat::Rgba8,
            planes: vec![Plane {
                data: vec![0u8; bytes],
                stride: bytes,
            }],
        }
    }

    fn key(frame: i64) -> FrameKey {
        FrameKey::new(MediaId::from_raw(1), frame)
    }

    fn frame_cost(bytes: usize) -> usize {
        bytes + std::mem::size_of::<DecodedFrame>()
    }

    #[test]
    fn insert_then_get_hits() {
        let mut cache = FrameCache::new(1 << 20);
        cache.insert(key(0), frame_of(64, 0));
        let got = cache.get(key(0)).expect("present");
        assert_eq!(got.pts_ticks, 0);
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 0);
    }

    #[test]
    fn miss_is_recorded() {
        let mut cache = FrameCache::new(1 << 20);
        assert!(cache.get(key(42)).is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn evicts_least_recently_used_to_fit_budget() {
        // Budget holds two frames but not three.
        let cap = frame_cost(100) * 2 + 1;
        let mut cache = FrameCache::new(cap);

        cache.insert(key(0), frame_of(100, 0));
        cache.insert(key(1), frame_of(100, 1));
        // Touch frame 0 so frame 1 becomes the least recently used.
        assert!(cache.get(key(0)).is_some());

        cache.insert(key(2), frame_of(100, 2));

        assert!(cache.contains(key(0)), "recently used survives");
        assert!(!cache.contains(key(1)), "LRU victim evicted");
        assert!(cache.contains(key(2)), "newest survives");
        assert_eq!(cache.stats().evictions, 1);
        assert!(cache.used_bytes() <= cap);
    }

    #[test]
    fn replacing_key_updates_byte_accounting() {
        let mut cache = FrameCache::new(1 << 20);
        cache.insert(key(0), frame_of(100, 0));
        let after_first = cache.used_bytes();
        cache.insert(key(0), frame_of(300, 0));
        assert_eq!(cache.len(), 1, "same key, no duplicate");
        assert_eq!(cache.used_bytes(), after_first + 200);
        assert_eq!(cache.get(key(0)).unwrap().planes[0].data.len(), 300);
    }

    #[test]
    fn oversized_frame_is_kept_until_displaced() {
        let mut cache = FrameCache::new(10); // smaller than any frame
        cache.insert(key(0), frame_of(100, 0));
        assert!(cache.contains(key(0)), "single over-budget frame is usable");
        // Next insert displaces the over-budget frame rather than both staying.
        cache.insert(key(1), frame_of(100, 1));
        assert_eq!(cache.len(), 1);
        assert!(cache.contains(key(1)));
        assert!(!cache.contains(key(0)));
    }

    #[test]
    fn get_or_try_insert_with_runs_producer_once() {
        let mut cache = FrameCache::new(1 << 20);
        let mut calls = 0;
        let mut produce = |frame: i64| {
            calls += 1;
            Ok::<_, ()>(frame_of(64, frame))
        };

        let a = cache
            .get_or_try_insert_with(key(0), || produce(0))
            .unwrap();
        let b = cache
            .get_or_try_insert_with(key(0), || produce(0))
            .unwrap();

        assert_eq!(calls, 1, "producer skipped on hit");
        assert!(Arc::ptr_eq(&a, &b), "same cached Arc returned");
    }

    #[test]
    fn invalidate_media_drops_only_that_media() {
        let mut cache = FrameCache::new(1 << 20);
        let m1 = MediaId::from_raw(1);
        let m2 = MediaId::from_raw(2);
        cache.insert(FrameKey::new(m1, 0), frame_of(100, 0));
        cache.insert(FrameKey::new(m1, 1), frame_of(100, 1));
        cache.insert(FrameKey::new(m2, 0), frame_of(100, 0));

        cache.invalidate_media(m1);

        assert_eq!(cache.len(), 1);
        assert!(cache.contains(FrameKey::new(m2, 0)));
        assert_eq!(cache.used_bytes(), frame_cost(100));
    }

    #[test]
    fn hit_rate_reflects_lookups() {
        let mut cache = FrameCache::new(1 << 20);
        cache.insert(key(0), frame_of(64, 0));
        cache.get(key(0)); // hit
        cache.get(key(9)); // miss
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate() - 0.5).abs() < f64::EPSILON);
    }
}
