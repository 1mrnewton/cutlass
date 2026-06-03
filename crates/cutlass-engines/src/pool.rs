//! The media pool: the engine's source-frame provider.
//!
//! Owns one [`FrameReader`] per registered media plus the single shared
//! [`FrameCache`]. Every frame request goes through the cache first; only a miss
//! reaches the reader (and thus the decoder). This is the seam the rest of the
//! engine — timeline resolution, the compositor, export — pulls source frames
//! from.

use std::sync::Arc;

use cutlass_decode::DecodedFrame;
use cutlass_models::{Map, MediaId, MediaSource};

use crate::cache::{CacheStats, FrameCache, FrameKey};
use crate::error::EngineError;
use crate::media::{FrameReader, MediaReader};

/// A registry of decodable media backed by a shared decoded-frame cache.
pub struct MediaPool {
    readers: Map<MediaId, Box<dyn FrameReader>>,
    cache: FrameCache,
}

impl MediaPool {
    /// Create an empty pool with the default cache budget.
    pub fn new() -> Self {
        Self::with_cache(FrameCache::default())
    }

    /// Create an empty pool backed by a caller-configured cache.
    pub fn with_cache(cache: FrameCache) -> Self {
        Self {
            readers: Map::default(),
            cache,
        }
    }

    /// Open `media`'s file and register it for decoding.
    ///
    /// Returns the [`MediaId`] now served by the pool. Opening touches the
    /// filesystem and probes the stream, so it is comparatively expensive — do
    /// it at import time, not per frame.
    pub fn open(&mut self, media: &MediaSource) -> Result<MediaId, EngineError> {
        let reader = MediaReader::open(media)?;
        self.register(media.id, Box::new(reader));
        Ok(media.id)
    }

    /// Register a pre-built reader under `media`. Replaces any existing reader
    /// for that id and drops its now-stale cached frames.
    pub fn register(&mut self, media: MediaId, reader: Box<dyn FrameReader>) {
        if self.readers.insert(media, reader).is_some() {
            self.cache.invalidate_media(media);
        }
    }

    /// Remove `media` from the pool and drop its cached frames.
    pub fn remove(&mut self, media: MediaId) {
        if self.readers.remove(&media).is_some() {
            self.cache.invalidate_media(media);
        }
    }

    pub fn contains(&self, media: MediaId) -> bool {
        self.readers.contains_key(&media)
    }

    /// Fetch source frame `source_frame` of `media`, decoding on a cache miss.
    ///
    /// The returned [`Arc`] is shared with the cache, so holding it (e.g. while
    /// compositing) costs no copy and keeps the frame alive even if it is later
    /// evicted.
    pub fn frame(
        &mut self,
        media: MediaId,
        source_frame: i64,
    ) -> Result<Arc<DecodedFrame>, EngineError> {
        let reader = self
            .readers
            .get_mut(&media)
            .ok_or(EngineError::UnknownMedia(media))?;
        let key = FrameKey::new(media, source_frame);
        // Disjoint borrows: `reader` borrows `self.readers`, the cache call
        // borrows `self.cache`. Only a miss runs the (decoding) closure.
        self.cache
            .get_or_try_insert_with(key, || reader.read(source_frame))
    }

    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    pub fn cache(&self) -> &FrameCache {
        &self.cache
    }
}

impl Default for MediaPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_decode::{PixelFormat, Plane};

    /// A reader that fabricates frames and counts how often it actually runs,
    /// so we can prove the cache prevents re-decoding.
    struct CountingReader {
        reads: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl FrameReader for CountingReader {
        fn read(&mut self, source_frame: i64) -> Result<DecodedFrame, EngineError> {
            self.reads.set(self.reads.get() + 1);
            Ok(DecodedFrame {
                width: 2,
                height: 2,
                pts_ticks: source_frame,
                format: PixelFormat::Rgba8,
                planes: vec![Plane {
                    data: vec![0u8; 16],
                    stride: 16,
                }],
            })
        }
    }

    fn counting() -> (Box<dyn FrameReader>, std::rc::Rc<std::cell::Cell<usize>>) {
        let reads = std::rc::Rc::new(std::cell::Cell::new(0));
        (
            Box::new(CountingReader {
                reads: reads.clone(),
            }),
            reads,
        )
    }

    #[test]
    fn unknown_media_errors() {
        let mut pool = MediaPool::new();
        let err = pool.frame(MediaId::from_raw(1), 0).unwrap_err();
        assert!(matches!(err, EngineError::UnknownMedia(_)));
    }

    #[test]
    fn second_request_is_served_from_cache() {
        let mut pool = MediaPool::new();
        let (reader, reads) = counting();
        let media = MediaId::from_raw(1);
        pool.register(media, reader);

        let a = pool.frame(media, 10).unwrap();
        let b = pool.frame(media, 10).unwrap();

        assert_eq!(reads.get(), 1, "decode happens once");
        assert!(Arc::ptr_eq(&a, &b), "same cached frame returned");
        assert_eq!(pool.cache_stats().hits, 1);
        assert_eq!(pool.cache_stats().misses, 1);
    }

    #[test]
    fn distinct_frames_each_decode_once() {
        let mut pool = MediaPool::new();
        let (reader, reads) = counting();
        let media = MediaId::from_raw(1);
        pool.register(media, reader);

        pool.frame(media, 0).unwrap();
        pool.frame(media, 1).unwrap();
        pool.frame(media, 0).unwrap();

        assert_eq!(reads.get(), 2, "frame 0 reused, frame 1 decoded once");
    }

    #[test]
    fn reregistering_media_drops_cached_frames() {
        let mut pool = MediaPool::new();
        let media = MediaId::from_raw(1);

        let (reader, _) = counting();
        pool.register(media, reader);
        pool.frame(media, 5).unwrap();
        assert!(pool.cache().contains(FrameKey::new(media, 5)));

        let (reader2, reads2) = counting();
        pool.register(media, reader2);
        assert!(!pool.cache().contains(FrameKey::new(media, 5)), "cache purged");

        pool.frame(media, 5).unwrap();
        assert_eq!(reads2.get(), 1, "served by the new reader, not stale cache");
    }

    #[test]
    fn remove_purges_media_from_cache() {
        let mut pool = MediaPool::new();
        let media = MediaId::from_raw(1);
        let (reader, _) = counting();
        pool.register(media, reader);
        pool.frame(media, 3).unwrap();

        pool.remove(media);

        assert!(!pool.contains(media));
        assert!(!pool.cache().contains(FrameKey::new(media, 3)));
        assert!(matches!(
            pool.frame(media, 3).unwrap_err(),
            EngineError::UnknownMedia(_)
        ));
    }
}
