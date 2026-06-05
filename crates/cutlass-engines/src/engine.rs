//! The engine: the headless editing session a front-end drives.
//!
//! [`Engine`] owns the [`Project`] (the edit state) and the [`MediaPool`] (the
//! decoders + shared frame cache), and keeps them in sync. Its headline call is
//! [`frame_at`](Engine::frame_at): given a timeline frame, it resolves the layer
//! stack and decodes (or cache-hits) each media frame, returning an owned,
//! back-to-front [`RenderedLayer`] list ready for the compositor.

use std::sync::Arc;

use cutlass_decode::DecodedFrame;
use cutlass_models::{ClipId, Generator, MediaId, MediaSource, ModelError, Project, Rational, TrackId};

use crate::cache::{CacheStats, FrameCache};
use crate::command::{EditCommand, EditHistory, EditOutcome};
use crate::error::EngineError;
use crate::media::FrameReader;
use crate::pool::MediaPool;
use crate::proxy::ProxyStatus;
use crate::resolve::{resolve_frame, LayerContent};

/// How many undo snapshots the engine retains by default.
const DEFAULT_HISTORY_LIMIT: usize = 128;

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
    history: EditHistory,
}

impl Engine {
    /// Create an engine with an empty project whose timeline runs at `frame_rate`.
    pub fn new(name: impl Into<String>, frame_rate: Rational) -> Self {
        Self {
            project: Project::new(name, frame_rate),
            pool: MediaPool::new(),
            history: EditHistory::new(DEFAULT_HISTORY_LIMIT),
        }
    }

    /// Create an engine backed by a caller-configured frame cache.
    pub fn with_cache(name: impl Into<String>, frame_rate: Rational, cache: FrameCache) -> Self {
        Self {
            project: Project::new(name, frame_rate),
            pool: MediaPool::with_cache(cache),
            history: EditHistory::new(DEFAULT_HISTORY_LIMIT),
        }
    }

    /// Build a standalone, source-only engine that owns `project`, opening a
    /// decoder for each of its media on the **current thread**.
    ///
    /// Unlike [`new`](Engine::new) + [`import_media`](Engine::import_media), this
    /// starts no background proxy builds and spawns no threads, so the whole
    /// engine lives on one thread — which is exactly what lets an export run on a
    /// worker thread (decoders aren't `Send`, but a freshly-opened one never
    /// crosses threads). Frames come straight from the source files at full
    /// resolution, the right input for a final render.
    pub fn for_export(project: Project) -> Result<Self, EngineError> {
        let mut pool = MediaPool::new();
        for media in project.media_iter() {
            pool.open(media)?;
        }
        Ok(Self {
            project,
            pool,
            history: EditHistory::new(0),
        })
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
        // Kick off a background proxy build so scrubbing becomes smooth shortly
        // after import; the source reader serves frames in the meantime.
        self.pool.request_proxy(&media);
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
        // Install any proxies that finished building since the last call, so
        // subsequent decodes take the fast disk-proxy path.
        self.pool.poll_proxies();
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

    /// Install finished proxy builds without resolving a frame (e.g. on an idle
    /// tick, so proxies get adopted even while playback is paused).
    pub fn poll_proxies(&mut self) {
        self.pool.poll_proxies();
    }

    /// Proxy build status for `media`, for surfacing progress in the UI.
    pub fn proxy_status(&self, media: MediaId) -> Option<&ProxyStatus> {
        self.pool.proxy_status(media)
    }

    /// Tell the build scheduler where the playhead is: every clip visible at
    /// `timeline_frame` is bumped up the proxy queue (topmost highest), so the
    /// footage the user is looking at becomes smooth first.
    pub fn set_playhead(&mut self, timeline_frame: i64) {
        // Back-to-front, so the topmost layer ends with the highest priority.
        for layer in resolve_frame(&self.project, timeline_frame) {
            if let LayerContent::Media { media, .. } = layer.content {
                self.pool.prioritize_proxy(media);
            }
        }
    }

    /// Bump a single media to the front of the proxy build queue.
    pub fn prioritize_proxy(&mut self, media: MediaId) {
        self.pool.prioritize_proxy(media);
    }

    /// Pause/resume background proxy builds. Pause during active scrub/playback
    /// so transcodes don't compete with the live decode path; resume when idle.
    pub fn set_background_paused(&mut self, paused: bool) {
        self.pool.set_background_paused(paused);
    }

    // --- edit commands ----------------------------------------------------

    /// Apply one structured [`EditCommand`] to the timeline.
    ///
    /// On success the pre-edit timeline is pushed onto the undo stack and the
    /// outcome (e.g. the id of a created/affected clip) is returned. If the
    /// command violates a model invariant it returns an error and the project —
    /// and the undo history — are left untouched, so a rejected edit is a no-op.
    pub fn apply(&mut self, command: EditCommand) -> Result<EditOutcome, EngineError> {
        // Snapshot before mutating so a successful edit is undoable; on failure
        // the model rejects the change without partial mutation, so we simply
        // drop the snapshot and record nothing.
        let snapshot = self.project.timeline().clone();
        let outcome = self.execute(command)?;
        self.history.record(snapshot);
        Ok(outcome)
    }

    /// Run a command against the project. Kept separate from [`apply`] so the
    /// history is only touched once the edit has provably succeeded.
    fn execute(&mut self, command: EditCommand) -> Result<EditOutcome, EngineError> {
        Ok(match command {
            EditCommand::AddClip {
                track,
                media,
                source,
                start,
            } => EditOutcome::Created(self.project.add_clip(track, media, source, start)?),
            EditCommand::AddGenerated {
                track,
                generator,
                timeline,
            } => EditOutcome::Created(self.project.add_generated(track, generator, timeline)?),
            EditCommand::SplitClip { clip, at } => {
                EditOutcome::Created(self.project.split_clip(clip, at)?)
            }
            EditCommand::TrimClip { clip, timeline } => {
                self.project.trim_clip(clip, timeline)?;
                EditOutcome::Updated(clip)
            }
            EditCommand::MoveClip {
                clip,
                to_track,
                start,
            } => {
                self.project.move_clip(clip, to_track, start)?;
                EditOutcome::Updated(clip)
            }
            EditCommand::RemoveClip { clip } => {
                self.project
                    .remove_clip(clip)
                    .ok_or(ModelError::UnknownClip(clip))?;
                EditOutcome::Removed(clip)
            }
            EditCommand::RippleDelete { clip } => {
                self.project.ripple_delete(clip)?;
                EditOutcome::Removed(clip)
            }
        })
    }

    /// Undo the most recent applied command, restoring the prior timeline.
    /// Returns `false` if there was nothing to undo.
    pub fn undo(&mut self) -> bool {
        if !self.history.can_undo() {
            return false;
        }
        let current = self.project.timeline().clone();
        let previous = self.history.undo(current).expect("undo stack non-empty");
        *self.project.timeline_mut() = previous;
        true
    }

    /// Re-apply the most recently undone command. Returns `false` if there is
    /// nothing to redo (including after a fresh edit invalidates the redo stack).
    pub fn redo(&mut self) -> bool {
        if !self.history.can_redo() {
            return false;
        }
        let current = self.project.timeline().clone();
        let next = self.history.redo(current).expect("redo stack non-empty");
        *self.project.timeline_mut() = next;
        true
    }

    pub fn can_undo(&self) -> bool {
        self.history.can_undo()
    }

    pub fn can_redo(&self) -> bool {
        self.history.can_redo()
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

    /// Engine on a 24fps timeline with one 24fps media source registered, so
    /// source and timeline frames map 1:1 in the edit-command tests.
    fn engine_for_edits() -> (Engine, MediaId, TrackId) {
        let mut engine = Engine::new("edit", Rational::FPS_24);
        let media = MediaSource::new("/tmp/edit.mp4", 1920, 1080, Rational::FPS_24, 1000, false);
        let media_id = engine.project_mut().add_media(media);
        let track = engine.project_mut().add_track(TrackKind::Video, "V1");
        engine.register_reader(media_id, Box::new(StubReader));
        (engine, media_id, track)
    }

    #[test]
    fn apply_add_then_split_and_undo_redo() {
        let (mut engine, media, track) = engine_for_edits();

        let added = engine
            .apply(EditCommand::AddClip {
                track,
                media,
                source: cutlass_models::TimeRange::new(0, 100),
                start: 0,
            })
            .unwrap();
        let EditOutcome::Created(clip) = added else {
            panic!("expected a created clip");
        };
        assert_eq!(engine.project().timeline().clip_count(), 1);

        // Split into two clips.
        let split = engine.apply(EditCommand::SplitClip { clip, at: 40 }).unwrap();
        assert!(matches!(split, EditOutcome::Created(_)));
        assert_eq!(engine.project().timeline().clip_count(), 2);

        // Undo the split: back to one clip. Undo the add: back to empty.
        assert!(engine.undo());
        assert_eq!(engine.project().timeline().clip_count(), 1);
        assert!(engine.undo());
        assert_eq!(engine.project().timeline().clip_count(), 0);
        assert!(!engine.can_undo());

        // Redo both, restoring the split state.
        assert!(engine.redo());
        assert_eq!(engine.project().timeline().clip_count(), 1);
        assert!(engine.redo());
        assert_eq!(engine.project().timeline().clip_count(), 2);
        assert!(!engine.can_redo());
    }

    #[test]
    fn rejected_command_is_a_noop_and_records_no_history() {
        let (mut engine, media, track) = engine_for_edits();
        engine
            .apply(EditCommand::AddClip {
                track,
                media,
                source: cutlass_models::TimeRange::new(0, 100),
                start: 0,
            })
            .unwrap();

        // Drop the undo history so we can assert the failed edit adds nothing.
        assert!(engine.undo());
        assert!(engine.redo());
        let undo_before = engine.can_undo();

        // Overlapping placement is rejected by the model.
        let err = engine.apply(EditCommand::AddClip {
            track,
            media,
            source: cutlass_models::TimeRange::new(0, 100),
            start: 50,
        });
        assert!(matches!(err, Err(EngineError::Model(_))));
        // No new undo entry, and the timeline still holds exactly the one clip.
        assert_eq!(engine.can_undo(), undo_before);
        assert_eq!(engine.project().timeline().clip_count(), 1);
    }

    #[test]
    fn fresh_edit_invalidates_redo() {
        let (mut engine, media, track) = engine_for_edits();
        engine
            .apply(EditCommand::AddClip {
                track,
                media,
                source: cutlass_models::TimeRange::new(0, 100),
                start: 0,
            })
            .unwrap();
        assert!(engine.undo());
        assert!(engine.can_redo());

        // A new command after an undo clears the redo stack.
        engine
            .apply(EditCommand::AddGenerated {
                track,
                generator: Generator::SolidColor { rgba: [0, 0, 0, 255] },
                timeline: cutlass_models::TimeRange::new(0, 48),
            })
            .unwrap();
        assert!(!engine.can_redo());
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
