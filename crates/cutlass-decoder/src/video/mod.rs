//! Video demux + decode.

mod decoder;
mod frame;
mod hwaccel;
mod keyframe_indexer;
mod strip;
mod thumbnail;

pub use decoder::{Decoder, SourceInfo, ffmpeg_version, hw_accel_from_env};
pub(crate) use decoder::ensure_ffmpeg_init;
pub use frame::{DecodedFrame, PixelFormat, Plane};
pub use strip::video_strip;
pub use thumbnail::{ThumbnailImage, video_thumbnail};
pub use hwaccel::{
    DecodeOptions, HwAccel, attach as attach_hwaccel, is_hardware_pixel_format,
    transfer_to_cpu as transfer_hw_frame_to_cpu,
};
pub use keyframe_indexer::{KeyframeIndex, duration_to_ticks, ticks_to_duration};
