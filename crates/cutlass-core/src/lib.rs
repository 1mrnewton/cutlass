//! Cutlass core: the contract every other crate signs.
//!
//! Cutlass decodes with the platform's native codec, hands the resulting frame
//! to a wgpu compositor, and presents into a native surface. Three crates have
//! to agree on the *shape* of the data flowing across that path — the decoder
//! (platform-specific), the compositor (wgpu), and the engine that drives them.
//! This crate is that shared vocabulary, and nothing more:
//!
//! - [`time`] — exact, OTIO-style [`RationalTime`] / [`Rational`] / [`TimeRange`].
//!   The timeline clock and stream PTS both live here, so a frame's presentation
//!   time is exact (NTSC `30000/1001` never drifts).
//! - [`color`] — [`ColorSpace`] (primaries, transfer, matrix, range). Carried
//!   end-to-end so the compositor can do YUV→RGB and tone-mapping correctly,
//!   including 10-bit / HDR (PQ, HLG).
//! - [`pixel`] — [`PixelFormat`], the plane layouts a hardware decoder emits
//!   (NV12, P010, planar YUV) plus packed RGB for stills and CPU fallback.
//! - [`frame`] — [`VideoFrame`], the decode→compositor handoff. It is either a
//!   zero-copy, opaque **GPU surface** (the fast path: VideoToolbox/MediaCodec
//!   output mapped straight into wgpu) or **CPU planes** (the portable
//!   fallback). Geometry carries coded size, visible rect, and rotation — the
//!   last is essential for phone video.
//! - [`image`] — [`RgbaImage`], the canonical CPU RGBA8 buffer: compositor
//!   readback, generator (text/shape) output, and the `get_frame` return type.
//! - [`decode`] / [`encode`] — the [`VideoDecoder`] / [`VideoEncoder`] traits.
//!   The codec call is native; *control* (seek, frame selection, the decode
//!   pipeline) stays in Rust behind these traits, which is what lets desktop,
//!   mobile, and the Python bindings share one path.
//!
//! ## Deliberately dependency-light
//!
//! Core does **not** depend on `wgpu`, `objc2`, or any platform SDK. The frame's
//! GPU surface is an opaque, type-erased handle ([`frame::GpuSurface`]); the
//! renderer interprets it. Keeping wgpu out of core matters in practice: Slint
//! pins specific wgpu versions, and we don't want the universal contract crate
//! dragged onto anyone's GPU-library upgrade cadence. The only dependencies are
//! `thiserror` and (optional) `serde`.

pub mod audio;
pub mod byte_budget;
pub mod color;
pub mod decode;
pub mod encode;
pub mod frame;
pub mod geometry;
pub mod image;
pub mod pixel;
pub mod time;

pub use audio::{AudioEncoderConfig, AudioReader};
pub use byte_budget::{ByteBudgetLru, RASTER_MEMO_BUDGET_BYTES};
pub use color::{ColorPrimaries, ColorRange, ColorSpace, MatrixCoefficients, TransferFunction};
pub use decode::{DecodeError, SourceInfo, VideoDecoder};
pub use encode::{EncodeError, EncoderConfig, VideoEncoder};
pub use frame::{
    CpuImage, D3d11SharedSurface, FrameData, GpuSurface, GpuSurfaceKind, Plane, VideoFrame,
};
pub use geometry::{Rect, Rotation};
pub use image::RgbaImage;
pub use pixel::{ChromaSubsampling, PixelFormat};
pub use time::{
    Rational, RationalTime, TimeError, TimeRange, check_same_rate, rate_eq, resample, time_add,
    time_sub,
};
