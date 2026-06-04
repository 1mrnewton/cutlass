//! Video demux + decode via ffmpeg-next.
//!
//! Hardware acceleration follows the same FFmpeg device-context model as
//! [ff-decode](https://docs.rs/ff-decode): optional GPU decode, CPU transfer via
//! `av_hwframe_transfer_data` until the compositor can consume GPU surfaces.
//!
//! See `cutlass-main/docs/decoder/research.md` for seek/threading design.

mod decoder;
mod encode;
mod error;
mod export;
mod frame;
mod hwaccel;
mod index;

pub use decoder::{Decoder, SourceInfo, ffmpeg_version, hw_accel_from_env};
pub use encode::{ProxyBuildOptions, ProxyConfig, ProxyStats, build_proxy, build_proxy_with};
pub use error::DecodeError;
pub use export::{ExportConfig, ExportStats, VideoExport};
pub use frame::{DecodedFrame, PixelFormat, Plane};
pub use hwaccel::{DecodeOptions, HwAccel};
pub use index::{KeyframeIndex, duration_to_ticks, ticks_to_duration};
