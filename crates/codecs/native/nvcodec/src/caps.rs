//! Codec ids from the NVIDIA decode and encode APIs.

use cutlass_core::VideoCodec;

/// Decode and encode codecs the NVIDIA driver listed.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Capabilities {
    decode: Vec<VideoCodec>,
    encode: Vec<VideoCodec>,
}

impl Capabilities {
    /// Asks NVDEC and NVENC. A missing library is an empty list. Windows
    /// returns an empty list.
    pub fn query() -> Self {
        #[cfg(target_os = "linux")]
        {
            crate::nvidia::query()
        }
        #[cfg(not(target_os = "linux"))]
        {
            Self::default()
        }
    }

    /// True when NVDEC listed `codec`.
    pub fn decodes(&self, codec: VideoCodec) -> bool {
        self.decode.contains(&codec)
    }

    /// True when NVENC listed `codec`.
    pub fn encodes(&self, codec: VideoCodec) -> bool {
        self.encode.contains(&codec)
    }

    #[cfg(target_os = "linux")]
    pub(crate) fn from_lists(decode: Vec<VideoCodec>, encode: Vec<VideoCodec>) -> Self {
        Self { decode, encode }
    }
}

/// `id` is a `cudaVideoCodec`. MPEG-1 and VC-1 are not video codecs here.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn video_codec(id: i32) -> Option<VideoCodec> {
    match id {
        1 => Some(VideoCodec::Mpeg2),
        2 => Some(VideoCodec::Mpeg4Part2),
        4 | 6 | 7 => Some(VideoCodec::H264),
        5 => Some(VideoCodec::Mjpeg),
        8 => Some(VideoCodec::Hevc),
        9 => Some(VideoCodec::Vp8),
        10 => Some(VideoCodec::Vp9),
        11 => Some(VideoCodec::Av1),
        _ => None,
    }
}

/// One NVIDIA GUID. The layout matches the encode API on every OS.
#[cfg(any(target_os = "linux", test))]
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Guid {
    pub data1: u32,
    pub data2: u16,
    pub data3: u16,
    pub data4: [u8; 8],
}

/// H.264, HEVC, and AV1 are the encode GUIDs. Anything else is skipped.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn guid_codec(guid: &Guid) -> Option<VideoCodec> {
    const H264: Guid = Guid {
        data1: 0x6bc8_2762,
        data2: 0x4e63,
        data3: 0x4ca4,
        data4: [0xaa, 0x85, 0x1e, 0x50, 0xf3, 0x21, 0xf6, 0xbf],
    };
    const HEVC: Guid = Guid {
        data1: 0x790c_dc88,
        data2: 0x4522,
        data3: 0x4d7b,
        data4: [0x94, 0x25, 0xbd, 0xa9, 0x97, 0x5f, 0x76, 0x03],
    };
    const AV1: Guid = Guid {
        data1: 0x0a35_2289,
        data2: 0x0aa7,
        data3: 0x4759,
        data4: [0x86, 0x2d, 0x5d, 0x15, 0xcd, 0x16, 0xd2, 0x54],
    };
    if guid == &H264 {
        Some(VideoCodec::H264)
    } else if guid == &HEVC {
        Some(VideoCodec::Hevc)
    } else if guid == &AV1 {
        Some(VideoCodec::Av1)
    } else {
        None
    }
}

/// The driver packs the minor version in the low 4 bits.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn driver_version(packed: u32) -> Option<(u32, u32)> {
    let major = packed >> 4;
    let minor = packed & 0xF;
    if major < 8 {
        None
    } else {
        Some((major, minor))
    }
}

#[cfg(any(target_os = "linux", test))]
pub(crate) fn api_version(major: u32, minor: u32) -> u32 {
    major | (minor << 24)
}

#[cfg(any(target_os = "linux", test))]
pub(crate) fn struct_version(api: u32, field: u32) -> u32 {
    api | (field << 16) | (0x7 << 28)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_codec_ids_map_to_video_codecs() {
        assert_eq!(video_codec(1), Some(VideoCodec::Mpeg2));
        assert_eq!(video_codec(2), Some(VideoCodec::Mpeg4Part2));
        assert_eq!(video_codec(4), Some(VideoCodec::H264));
        assert_eq!(video_codec(6), Some(VideoCodec::H264));
        assert_eq!(video_codec(5), Some(VideoCodec::Mjpeg));
        assert_eq!(video_codec(8), Some(VideoCodec::Hevc));
        assert_eq!(video_codec(9), Some(VideoCodec::Vp8));
        assert_eq!(video_codec(10), Some(VideoCodec::Vp9));
        assert_eq!(video_codec(11), Some(VideoCodec::Av1));
        assert_eq!(video_codec(0), None);
        assert_eq!(video_codec(3), None);
    }

    #[test]
    fn encode_guids_are_h264_hevc_and_av1() {
        let h264 = Guid {
            data1: 0x6bc8_2762,
            data2: 0x4e63,
            data3: 0x4ca4,
            data4: [0xaa, 0x85, 0x1e, 0x50, 0xf3, 0x21, 0xf6, 0xbf],
        };
        assert_eq!(guid_codec(&h264), Some(VideoCodec::H264));
        let hevc = Guid {
            data1: 0x790c_dc88,
            data2: 0x4522,
            data3: 0x4d7b,
            data4: [0x94, 0x25, 0xbd, 0xa9, 0x97, 0x5f, 0x76, 0x03],
        };
        assert_eq!(guid_codec(&hevc), Some(VideoCodec::Hevc));
        let av1 = Guid {
            data1: 0x0a35_2289,
            data2: 0x0aa7,
            data3: 0x4759,
            data4: [0x86, 0x2d, 0x5d, 0x15, 0xcd, 0x16, 0xd2, 0x54],
        };
        assert_eq!(guid_codec(&av1), Some(VideoCodec::Av1));
        let mut other = h264;
        other.data1 = 0;
        assert_eq!(guid_codec(&other), None);
    }

    #[test]
    fn the_driver_version_splits_into_major_and_minor() {
        assert_eq!(driver_version((12 << 4) | 1), Some((12, 1)));
        assert_eq!(driver_version(7 << 4), None);
        assert_eq!(api_version(12, 1), 0x0100_000C);
        assert_eq!(struct_version(0x0100_000C, 2), 0x7102_000C);
    }

    #[test]
    fn prores_is_not_an_nvidia_codec() {
        for id in 0..16 {
            assert_ne!(video_codec(id), Some(VideoCodec::ProRes));
        }
    }
}
