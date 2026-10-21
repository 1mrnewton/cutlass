//! H.264 profile and level settings.

use cutlass_core::{PixelFormat, VideoCodec};
use objc2_core_media::kCMVideoCodecType_H264;
use objc2_video_toolbox::kVTProfileLevel_H264_High_AutoLevel;

use super::Settings;
use crate::error::Error;

/// High profile with the level picked from size and frame rate. Apple's H.264
/// encoders are 8-bit only.
pub(super) fn settings(format: PixelFormat) -> Result<Settings, Error> {
    match format {
        PixelFormat::Nv12 => Ok(Settings {
            codec_type: kCMVideoCodecType_H264,
            // SAFETY: an immutable CFString constant exported by VideoToolbox.
            profile: unsafe { kVTProfileLevel_H264_High_AutoLevel },
            atom: "avcC",
        }),
        PixelFormat::P010 => Err(Error::EncodeUnsupported(VideoCodec::H264)),
    }
}
