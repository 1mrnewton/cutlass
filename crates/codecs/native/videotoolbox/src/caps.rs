//! What this machine can decode and encode, checked once at startup.

use std::ptr::{self, NonNull};

use cutlass_core::VideoCodec;
use objc2_core_foundation::{
    CFArray, CFBoolean, CFDictionary, CFNumber, CFRetained, CFString, CFType,
};
use objc2_core_media::{
    CMVideoCodecType, kCMVideoCodecType_AV1, kCMVideoCodecType_AppleProRes422,
    kCMVideoCodecType_AppleProRes422HQ, kCMVideoCodecType_AppleProRes422LT,
    kCMVideoCodecType_AppleProRes422Proxy, kCMVideoCodecType_AppleProRes4444,
    kCMVideoCodecType_AppleProRes4444XQ, kCMVideoCodecType_H264, kCMVideoCodecType_HEVC,
    kCMVideoCodecType_JPEG, kCMVideoCodecType_MPEG2Video, kCMVideoCodecType_MPEG4Video,
    kCMVideoCodecType_VP9,
};
use objc2_video_toolbox::{
    VTCopyVideoEncoderList, VTIsHardwareDecodeSupported,
    VTRegisterSupplementalVideoDecoderIfAvailable, kVTVideoEncoderList_CodecType,
    kVTVideoEncoderList_DisplayName, kVTVideoEncoderList_EncoderID,
    kVTVideoEncoderList_IsHardwareAccelerated,
};

use crate::cf::owned;
use crate::error::{Error, check};

const VIDEO_CODECS: [VideoCodec; 9] = [
    VideoCodec::H264,
    VideoCodec::Hevc,
    VideoCodec::Vp8,
    VideoCodec::Vp9,
    VideoCodec::Av1,
    VideoCodec::Mpeg2,
    VideoCodec::Mpeg4Part2,
    VideoCodec::Mjpeg,
    VideoCodec::ProRes,
];

const PRORES_TYPES: [CMVideoCodecType; 6] = [
    kCMVideoCodecType_AppleProRes422Proxy,
    kCMVideoCodecType_AppleProRes422LT,
    kCMVideoCodecType_AppleProRes422,
    kCMVideoCodecType_AppleProRes422HQ,
    kCMVideoCodecType_AppleProRes4444,
    kCMVideoCodecType_AppleProRes4444XQ,
];

/// What VideoToolbox can do on this device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capabilities {
    hardware_decode: Vec<VideoCodec>,
    encoders: Vec<EncoderInfo>,
}

/// One entry from `VTCopyVideoEncoderList`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncoderInfo {
    pub codec: VideoCodec,
    pub hardware: bool,
    /// Apple's encoder ID, the same on every run.
    pub id: String,
    pub name: String,
}

impl Capabilities {
    /// Asks the device. Query once at startup and pass the result around.
    pub fn query() -> Result<Self, Error> {
        let hardware_decode = VIDEO_CODECS
            .into_iter()
            .filter(|&codec| decode_type(codec).is_some_and(hardware_decode_supported))
            .collect();
        Ok(Self {
            hardware_decode,
            encoders: encoders()?,
        })
    }

    /// VideoToolbox also decodes H.264, HEVC, MPEG-2, MPEG-4 Part 2, JPEG, and
    /// ProRes in software where this is false.
    pub fn hardware_decode(&self, codec: VideoCodec) -> bool {
        self.hardware_decode.contains(&codec)
    }

    pub fn encoders(&self) -> &[EncoderInfo] {
        &self.encoders
    }

    pub fn can_encode(&self, codec: VideoCodec) -> bool {
        self.encoders.iter().any(|encoder| encoder.codec == codec)
    }

    pub fn hardware_encode(&self, codec: VideoCodec) -> bool {
        self.encoders
            .iter()
            .any(|encoder| encoder.codec == codec && encoder.hardware)
    }
}

/// The codec type a decode session uses. ProRes stands in with 422; each
/// flavor has its own type.
pub(crate) fn decode_type(codec: VideoCodec) -> Option<CMVideoCodecType> {
    match codec {
        VideoCodec::H264 => Some(kCMVideoCodecType_H264),
        VideoCodec::Hevc => Some(kCMVideoCodecType_HEVC),
        VideoCodec::Vp8 => None,
        VideoCodec::Vp9 => Some(kCMVideoCodecType_VP9),
        VideoCodec::Av1 => Some(kCMVideoCodecType_AV1),
        VideoCodec::Mpeg2 => Some(kCMVideoCodecType_MPEG2Video),
        VideoCodec::Mpeg4Part2 => Some(kCMVideoCodecType_MPEG4Video),
        VideoCodec::Mjpeg => Some(kCMVideoCodecType_JPEG),
        VideoCodec::ProRes => Some(kCMVideoCodecType_AppleProRes422),
    }
}

fn codec_for_type(codec_type: CMVideoCodecType) -> Option<VideoCodec> {
    if PRORES_TYPES.contains(&codec_type) {
        return Some(VideoCodec::ProRes);
    }
    [
        (kCMVideoCodecType_H264, VideoCodec::H264),
        (kCMVideoCodecType_HEVC, VideoCodec::Hevc),
        (kCMVideoCodecType_JPEG, VideoCodec::Mjpeg),
    ]
    .into_iter()
    .find_map(|(known, codec)| (known == codec_type).then_some(codec))
}

/// VP9 and AV1 decoders ship with macOS but load only on request.
pub(crate) fn register_supplemental_decoder(codec_type: CMVideoCodecType) {
    if codec_type == kCMVideoCodecType_VP9 || codec_type == kCMVideoCodecType_AV1 {
        // SAFETY: takes a plain codec type and has no other preconditions. It
        // does nothing when the decoder is missing or already registered.
        unsafe { VTRegisterSupplementalVideoDecoderIfAvailable(codec_type) };
    }
}

fn hardware_decode_supported(codec_type: CMVideoCodecType) -> bool {
    register_supplemental_decoder(codec_type);
    // SAFETY: takes a plain codec type and has no other preconditions.
    unsafe { VTIsHardwareDecodeSupported(codec_type) }
}

fn encoders() -> Result<Vec<EncoderInfo>, Error> {
    let mut list: *const CFArray = ptr::null();
    // SAFETY: no options, and `list` is a valid place for the result.
    let status = unsafe { VTCopyVideoEncoderList(None, NonNull::from(&mut list)) };
    check("VTCopyVideoEncoderList", status)?;
    // SAFETY: Copy functions return an owned reference.
    let list = unsafe { owned(list.cast_mut(), "VTCopyVideoEncoderList") }?;
    // SAFETY: Apple documents the list as an array of dictionaries keyed by
    // CFString.
    let list =
        unsafe { CFRetained::cast_unchecked::<CFArray<CFDictionary<CFString, CFType>>>(list) };

    // SAFETY: these are immutable CFString constants exported by VideoToolbox.
    let (codec_key, id_key, name_key, hardware_key) = unsafe {
        (
            kVTVideoEncoderList_CodecType,
            kVTVideoEncoderList_EncoderID,
            kVTVideoEncoderList_DisplayName,
            kVTVideoEncoderList_IsHardwareAccelerated,
        )
    };
    let string = |entry: &CFDictionary<CFString, CFType>, key: &CFString| {
        entry
            .get(key)
            .and_then(|value| value.downcast::<CFString>().ok())
            .map(|value| value.to_string())
    };

    let mut encoders = Vec::new();
    for entry in list.iter() {
        let codec = entry
            .get(codec_key)
            .and_then(|value| value.downcast::<CFNumber>().ok())
            .and_then(|number| number.as_i64())
            .and_then(|number| CMVideoCodecType::try_from(number).ok())
            .and_then(codec_for_type);
        let (Some(codec), Some(id)) = (codec, string(&entry, id_key)) else {
            continue;
        };
        let hardware = entry
            .get(hardware_key)
            .and_then(|value| value.downcast::<CFBoolean>().ok())
            .is_some_and(|value| value.as_bool());
        let name = string(&entry, name_key).unwrap_or_else(|| id.clone());
        encoders.push(EncoderInfo {
            codec,
            hardware,
            id,
            name,
        });
    }
    Ok(encoders)
}
