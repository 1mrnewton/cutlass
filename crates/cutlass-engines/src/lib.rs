//! Cutlass editing engine: the headless core that drives the model, decoders,
//! and compositor on behalf of a front-end (UI, export, or the AI agent).
//!
//! The frame-resolution path:
//! - [`Engine`] — owns the project + pool; `frame_at(n)` is the headline call.
//! - [`resolve_frame`] — timeline frame N -> ordered layer list to composite.
//! - [`FrameCache`] — byte-bounded LRU of decoded frames (defense vs re-decode).
//! - [`MediaReader`] — sequential frame reads over one file (seek-vs-step).
//! - [`MediaPool`] — owns the readers + shared cache; the `frame(media, src)`
//!   entry point the rest of the engine pulls source frames from.

mod cache;
mod engine;
mod error;
mod media;
mod pool;
mod resolve;

pub use cache::{CacheStats, FrameCache, FrameKey, DEFAULT_CAPACITY_BYTES};
pub use engine::{Engine, RenderedContent, RenderedLayer};
pub use error::EngineError;
pub use media::{frame_to_time, time_to_frame, FrameReader, MediaReader};
pub use pool::MediaPool;
pub use resolve::{resolve_frame, LayerContent, ResolvedLayer};
