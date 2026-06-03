//! The engine: the headless editing session a front-end drives.
//!
//! [`Engine`] owns the [`Project`] (the edit state) and the [`MediaPool`] (the
//! decoders + shared frame cache), and keeps them in sync. Its headline call is
//! [`frame_at`](Engine::frame_at): given a timeline frame, it resolves the layer
//! stack and decodes (or cache-hits) each media frame, returning an owned,
//! back-to-front [`RenderedLayer`] list ready for the compositor.

use std::sync::Arc;

use cutlass_decode::DecodedFrame;
use cutlass_models::{ClipId, Generator, MediaId, MediaSource, Project, Rational, TrackId};

use crate::cache::{CacheStats, FrameCache};
use crate::error::EngineError;
use crate::media::FrameReader;
use crate::pool::MediaPool;
use crate::resolve::{resolve_frame, LayerContent};

/// One fully-resolved layer of a rendered frame: its source content is in hand
/// (a decoded frame or generator parameters), tagged with origin for the UI.
#[derive(Debug, Clone)]
pub struct RenderedLayer {
    pub track: TrackId,
    pub clip: ClipId,
    pub content: RenderedContent,
}

/// The drawable content of a [`RenderedLayer`].
#[derive(Debug, Clone)]
pub enum RenderedContent {
    /// A decoded media frame, shared with the cache (cheap to clone/hold).
    Media(Arc<DecodedFrame>),
    /// Engine-generated content to be drawn by the compositor.
    Generated(Generator),
}

/// A headless editing session: edit state plus the decode/cache machinery.
pub struct Engine {
    project: Project,
    pool: MediaPool,
}

impl Engine {
    /// Create an engine with an empty project whose timeline runs at `frame_rate`.
    pub fn new(name: impl Into<String>, frame_rate: Rational) -> Self {
        Self {
            project: Project::new(name, frame_rate),
            pool: MediaPool::new(),
        }
    }

    /// Create an engine backed by a caller-configured frame cache.
    pub fn with_cache(name: impl Into<String>, frame_rate: Rational, cache: FrameCache) -> Self {
        Self {
            project: Project::new(name, frame_rate),
            pool: MediaPool::with_cache(cache),
        }
    }

    pub fn project(&self) -> &Project {
        &self.project
    }

    /// Mutable access to the project for edits/commands.
    ///
    /// Prefer [`import_media`](Engine::import_media) to add media so the decoder
    /// pool stays in sync; adding media directly here leaves it without a reader
    /// until one is registered.
    pub fn project_mut(&mut self) -> &mut Project {
        &mut self.project
    }

    /// Import a source: open its decoder in the pool, then add it to the project.
    ///
    /// Opening first means a file that fails to decode never enters the project.
    pub fn import_media(&mut self, media: MediaSource) -> Result<MediaId, EngineError> {
        self.pool.open(&media)?;
        Ok(self.project.add_media(media))
    }

    /// Register a pre-built reader for `media` (proxies, tests, synthetic input).
    /// The media must already exist in the project for resolution to find it.
    pub fn register_reader(&mut self, media: MediaId, reader: Box<dyn FrameReader>) {
        self.pool.register(media, reader);
    }

    /// Resolve and decode the layer stack at `timeline_frame`, back-to-front.
    ///
    /// Index 0 is the bottommost layer. Media layers are fetched through the
    /// cache, so repeated/adjacent requests are cheap. Returns an owned list so
    /// the caller can hold it without borrowing the engine.
    pub fn frame_at(&mut self, timeline_frame: i64) -> Result<Vec<RenderedLayer>, EngineError> {
        // `layers` borrows `self.project`; the decode below borrows `self.pool`
        // — disjoint fields, so both live at once.
        let layers = resolve_frame(&self.project, timeline_frame);
        let mut rendered = Vec::with_capacity(layers.len());
        for layer in &layers {
            let content = match layer.content {
                LayerContent::Media {
                    media,
                    source_frame,
                } => RenderedContent::Media(self.pool.frame(media, source_frame)?),
                LayerContent::Generated(generator) => RenderedContent::Generated(generator.clone()),
            };
            rendered.push(RenderedLayer {
                track: layer.track,
                clip: layer.clip,
                content,
            });
        }
        Ok(rendered)
    }

    /// Total timeline length in timeline frames.
    pub fn duration(&self) -> i64 {
        self.project.timeline().duration()
    }

    pub fn cache_stats(&self) -> CacheStats {
        self.pool.cache_stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_decode::{PixelFormat, Plane};
    use cutlass_models::{TimeRange, TrackKind};

    /// A reader that fabricates a frame whose `pts_ticks` encodes the requested
    /// source frame, so tests can assert which frame was fetched.
    struct StubReader;

    impl FrameReader for StubReader {
        fn read(&mut self, source_frame: i64) -> Result<DecodedFrame, EngineError> {
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

    /// 30fps media on a 24fps timeline, with a stub reader registered.
    fn engine_with_media() -> (Engine, MediaId, TrackId) {
        let mut engine = Engine::new("test", Rational::FPS_24);
        let media = MediaSource::new("/tmp/a.mp4", 1920, 1080, Rational::FPS_30, 3000, false);
        let media_id = engine.project_mut().add_media(media);
        let track = engine.project_mut().add_track(TrackKind::Video, "V1");
        engine.register_reader(media_id, Box::new(StubReader));
        (engine, media_id, track)
    }

    #[test]
    fn empty_timeline_renders_no_layers() {
        let mut engine = Engine::new("test", Rational::FPS_24);
        assert!(engine.frame_at(0).unwrap().is_empty());
    }

    #[test]
    fn frame_at_decodes_mapped_source_frame() {
        let (mut engine, _media, track) = engine_with_media();
        engine
            .project_mut()
            .add_clip(track, _media, TimeRange::new(300, 300), 0)
            .unwrap();

        // 24 timeline frames in == source frame 330 (1s @ 30fps from in-point 300).
        let layers = engine.frame_at(24).unwrap();
        assert_eq!(layers.len(), 1);
        match &layers[0].content {
            RenderedContent::Media(frame) => assert_eq!(frame.pts_ticks, 330),
            other => panic!("expected media, got {other:?}"),
        }
    }

    #[test]
    fn repeated_frame_at_hits_cache() {
        let (mut engine, _media, track) = engine_with_media();
        engine
            .project_mut()
            .add_clip(track, _media, TimeRange::new(0, 300), 0)
            .unwrap();

        let a = engine.frame_at(10).unwrap();
        let b = engine.frame_at(10).unwrap();

        let (RenderedContent::Media(fa), RenderedContent::Media(fb)) =
            (&a[0].content, &b[0].content)
        else {
            panic!("expected media layers");
        };
        assert!(Arc::ptr_eq(fa, fb), "second resolve served from cache");
        assert_eq!(engine.cache_stats().hits, 1);
        assert_eq!(engine.cache_stats().misses, 1);
    }

    #[test]
    fn generated_layer_passes_through() {
        let mut engine = Engine::new("test", Rational::FPS_24);
        let track = engine.project_mut().add_track(TrackKind::Video, "V1");
        engine
            .project_mut()
            .add_generated(
                track,
                Generator::Text {
                    content: "hello".into(),
                },
                TimeRange::new(0, 48),
            )
            .unwrap();

        let layers = engine.frame_at(5).unwrap();
        assert_eq!(layers.len(), 1);
        match &layers[0].content {
            RenderedContent::Generated(Generator::Text { content }) => assert_eq!(content, "hello"),
            other => panic!("expected text generator, got {other:?}"),
        }
    }

    #[test]
    fn import_media_missing_file_errors_and_does_not_register() {
        let mut engine = Engine::new("test", Rational::FPS_24);
        let media = MediaSource::new(
            "/nonexistent/path.mp4",
            1920,
            1080,
            Rational::FPS_30,
            100,
            false,
        );
        let id = media.id;
        assert!(engine.import_media(media).is_err());
        assert_eq!(engine.project().media_count(), 0);
        assert!(engine.project().media(id).is_none());
    }
}
