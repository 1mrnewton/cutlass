//! HEVC profile and bit depth settings.

use cutlass_core::PixelFormat;
use objc2_core_media::kCMVideoCodecType_HEVC;
use objc2_video_toolbox::{
    kVTProfileLevel_HEVC_Main_AutoLevel, kVTProfileLevel_HEVC_Main10_AutoLevel,
};

use super::Settings;
use crate::error::Error;

/// Main for 8-bit input, Main 10 for 10-bit, with the level picked from size
/// and frame rate.
pub(super) fn settings(format: PixelFormat) -> Result<Settings, Error> {
    // SAFETY: immutable CFString constants exported by VideoToolbox.
    let profile = unsafe {
        match format {
            PixelFormat::Nv12 => kVTProfileLevel_HEVC_Main_AutoLevel,
            PixelFormat::P010 => kVTProfileLevel_HEVC_Main10_AutoLevel,
        }
    };
    Ok(Settings {
        codec_type: kCMVideoCodecType_HEVC,
        profile,
        atom: "hvcC",
    })
}
