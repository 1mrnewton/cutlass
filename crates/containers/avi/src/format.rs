//! `strf` contents: BITMAPINFOHEADER for video, WAVEFORMATEX for audio.
//!
//! The bytes after the fixed part of `strf` are the codec configuration,
//! as FFmpeg and GStreamer read them: everything past the 40-byte
//! BITMAPINFOHEADER, or the `cbSize` bytes after WAVEFORMATEX. Two
//! streams are reshaped so a Cutlass decoder can take them:
//!
//! - PCM gets the sixteen-byte wave description `cutlass-riff` hands a PCM
//!   decoder (tag, width, block align, channel mask, integer or float).
//! - Motion JPEG keeps its extradata only when it is a JPEG table stream
//!   (`FF D8`). Any other bytes there are dropped; the frames carry their
//!   own tables.

use cutlass_codec::{CodecId, FrameSize, StreamCodec, StreamParams, UnknownCodec};
use cutlass_riff::WavFormat;

use crate::bytes::{id_at, u16_at};
use cutlass_container::OpenError;

use crate::header::{MainHeader, bad, signed_at};

/// What `strh.fccType` says the stream is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamKind {
    Video,
    Audio,
    Text,
    /// `dats`, `mids`, DV `iavs`, and any other type.
    Other,
}

impl StreamKind {
    pub(crate) fn of(kind: [u8; 4]) -> Self {
        match &kind {
            b"vids" => Self::Video,
            b"auds" => Self::Audio,
            b"txts" => Self::Text,
            _ => Self::Other,
        }
    }
}

/// The parts of a BITMAPINFOHEADER this crate reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoFormat {
    pub compression: [u8; 4],
    pub width: i32,
    /// Negative for a top-down raw picture.
    pub height: i32,
    pub bit_count: u16,
    pub extra: Vec<u8>,
}

pub(crate) fn parse_video(strf: &[u8]) -> Option<VideoFormat> {
    if strf.len() < 40 {
        return None;
    }
    Some(VideoFormat {
        width: signed_at(strf, 4).ok()?,
        height: signed_at(strf, 8).ok()?,
        bit_count: u16_at(strf, 14).ok()?,
        compression: id_at(strf, 16).ok()?,
        extra: strf.get(40..)?.to_vec(),
    })
}

/// Codec ids for the video FourCCs a Cutlass decoder reads, compared
/// without case. The list follows FFmpeg's `ff_codec_bmp_tags` for those
/// codecs. DivX 3 (`DIV3`, `MP43`) is Microsoft MPEG-4 v3, not Part 2, and
/// stays unknown.
pub(crate) fn video_codec(compression: [u8; 4]) -> Option<CodecId> {
    let tag = compression.map(|byte| byte.to_ascii_uppercase());
    let id = match &tag {
        b"H264" | b"X264" | b"AVC1" | b"DAVC" | b"SMV2" | b"VSSH" | b"Q264" | b"V264" | b"GAVC"
        | b"UMSV" | b"TSHD" | b"INMC" => CodecId::H264,
        b"HEVC" | b"H265" | b"X265" | b"HVC1" | b"HEV1" => CodecId::Hevc,
        b"FMP4" | b"DIVX" | b"DX50" | b"XVID" | b"MP4S" | b"M4S2" | b"MP4V" | b"3IV2" | b"BLZ0"
        | b"RMP4" | b"DXGM" | b"XVIX" | b"UMP4" | b"HDX4" | b"SEDG" | b"DIGI" | b"EPHV"
        | b"WV1F" | b"SN40" | b"SIPP" | b"FVFW" | b"DM4V" | b"GEOV" | b"GEOX" | b"DMK2"
        | b"PVMM" | b"M4CC" | b"EM4A" | b"MP4 " => CodecId::Mpeg4,
        b"MPG2" | b"MPEG" | b"MPG1" | b"PIM1" | b"PIM2" | b"VCR2" | b"MMES" | b"MPGI" | b"MX5P"
        | b"DVR " | b"SLIF" | b"LMP2" | b"HDV2" => CodecId::Mpeg2,
        b"MJPG" | b"JPEG" | b"IJPG" | b"AVRN" | b"AVI1" | b"DMB1" | b"CJPG" | b"SLMJ" | b"QIVG"
        | b"ACDV" | b"AVDJ" | b"TVMJ" | b"TJPG" | b"JPGL" | b"MJPA" => CodecId::Mjpeg,
        b"VP80" => CodecId::Vp8,
        b"VP90" => CodecId::Vp9,
        b"AV01" => CodecId::Av1,
        _ => return None,
    };
    Some(id)
}

/// Uncompressed pictures: `BI_RGB`, `BI_BITFIELDS`, and the common raw FourCCs.
pub(crate) fn is_raw(compression: [u8; 4]) -> bool {
    matches!(
        &compression,
        [0, 0, 0, 0]
            | [3, 0, 0, 0]
            | b"RGB "
            | b"RAW "
            | b"DIB "
            | b"I420"
            | b"IYUV"
            | b"YV12"
            | b"YUY2"
            | b"UYVY"
            | b"YVYU"
            | b"Y800"
            | b"NV12"
            | b"NV21"
            | b"Y41P"
            | b"YV16"
            | b"Y422"
            | b"v210"
    )
}

fn raw_tag(compression: [u8; 4]) -> Vec<u8> {
    match compression {
        [0, 0, 0, 0] => b"BI_RGB".to_vec(),
        [3, 0, 0, 0] => b"BI_BITFIELDS".to_vec(),
        other => other.to_vec(),
    }
}

fn unknown(mut tag: Vec<u8>) -> Result<StreamCodec, OpenError> {
    tag.truncate(UnknownCodec::MAX_TAG);
    UnknownCodec::new(tag)
        .map(StreamCodec::Unknown)
        .map_err(|_| bad("codec tag does not fit"))
}

/// What a stream describes, plus whether its frames all stand alone.
pub(crate) struct Described {
    pub params: StreamParams,
    pub codec: Option<CodecId>,
    /// Every chunk is a random-access point.
    pub intra: bool,
    /// Packets carry a presentation time equal to the decode time.
    pub presentation: bool,
    /// `nBlockAlign` for audio, used for the DirectShow chunk duration.
    pub block_align: u32,
}

pub(crate) fn describe_video(strf: &[u8], main: &MainHeader) -> Result<Described, OpenError> {
    let Some(format) = parse_video(strf) else {
        return Ok(Described {
            params: StreamParams::opaque(unknown(b"avi:vids".to_vec())?, Vec::new()),
            codec: None,
            intra: false,
            presentation: false,
            block_align: 0,
        });
    };
    let codec = video_codec(format.compression);
    let raw = codec.is_none() && is_raw(format.compression);
    let stream_codec = match codec {
        Some(id) => StreamCodec::Known(id),
        None if raw => unknown(raw_tag(format.compression))?,
        None => unknown(format.compression.to_vec())?,
    };
    let extra = match codec {
        Some(CodecId::Mjpeg) if !format.extra.starts_with(&[0xFF, 0xD8]) => Vec::new(),
        _ => format.extra,
    };
    let size = picture_size(format.width, format.height)
        .or_else(|| picture_size_u(main.width, main.height));
    let params = match size {
        Some(size) => StreamParams::video(stream_codec, size, extra),
        None => StreamParams::opaque(stream_codec, extra),
    };
    let intra = raw || codec == Some(CodecId::Mjpeg);
    let in_order = matches!(codec, Some(CodecId::Vp8 | CodecId::Vp9 | CodecId::Av1));
    Ok(Described {
        params,
        codec,
        intra,
        presentation: intra || in_order,
        block_align: 0,
    })
}

fn picture_size(width: i32, height: i32) -> Option<FrameSize> {
    FrameSize::try_new(width.unsigned_abs(), height.unsigned_abs()).ok()
}

fn picture_size_u(width: u32, height: u32) -> Option<FrameSize> {
    FrameSize::try_new(width, height).ok()
}

pub(crate) fn describe_audio(strf: &[u8]) -> Result<Described, OpenError> {
    let Ok(format) = cutlass_riff::parse_wave_format(strf) else {
        let tag = u16_at(strf, 0).map_or_else(
            |_| b"avi:auds".to_vec(),
            |tag| format!("wav:{tag:04x}").into_bytes(),
        );
        return Ok(Described {
            params: StreamParams::opaque(unknown(tag)?, Vec::new()),
            codec: None,
            intra: true,
            presentation: true,
            block_align: u32::from(u16_at(strf, 12).unwrap_or(0)),
        });
    };
    Ok(audio_from(&format))
}

fn audio_from(format: &WavFormat) -> Described {
    let codec = format.codec().clone().known();
    let extra = if codec == Some(CodecId::Pcm) {
        format.extradata()
    } else {
        format.extra().to_vec()
    };
    let params = StreamParams::try_audio(
        format.codec().clone(),
        format.sample_rate(),
        format.channels(),
        extra.clone(),
    )
    .unwrap_or_else(|_| StreamParams::opaque(format.codec().clone(), extra));
    Described {
        params,
        codec,
        intra: true,
        presentation: true,
        block_align: u32::from(format.block_align()),
    }
}

pub(crate) fn describe_other(kind: [u8; 4], handler: [u8; 4]) -> Result<Described, OpenError> {
    let mut tag = b"avi:".to_vec();
    tag.extend_from_slice(&kind);
    if handler != [0; 4] {
        tag.push(b':');
        tag.extend_from_slice(&handler);
    }
    Ok(Described {
        params: StreamParams::opaque(unknown(tag)?, Vec::new()),
        codec: None,
        intra: true,
        presentation: true,
        block_align: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::{describe_audio, describe_video, is_raw, video_codec};
    use crate::header::MainHeader;
    use cutlass_codec::{CodecId, StreamCodec};

    fn bih(compression: &[u8; 4], width: i32, height: i32, extra: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(40 + u32::try_from(extra.len()).unwrap()).to_le_bytes());
        out.extend_from_slice(&width.to_le_bytes());
        out.extend_from_slice(&height.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&24u16.to_le_bytes());
        out.extend_from_slice(compression);
        out.extend_from_slice(&[0; 20]);
        out.extend_from_slice(extra);
        out
    }

    #[test]
    fn the_cutlass_video_fourccs_map_without_case() {
        assert_eq!(video_codec(*b"MJPG"), Some(CodecId::Mjpeg));
        assert_eq!(video_codec(*b"h264"), Some(CodecId::H264));
        assert_eq!(video_codec(*b"xvid"), Some(CodecId::Mpeg4));
        assert_eq!(video_codec(*b"DX50"), Some(CodecId::Mpeg4));
        assert_eq!(video_codec(*b"FMP4"), Some(CodecId::Mpeg4));
        assert_eq!(video_codec(*b"HEVC"), Some(CodecId::Hevc));
        assert_eq!(video_codec(*b"MPG2"), Some(CodecId::Mpeg2));
        assert_eq!(video_codec(*b"VP80"), Some(CodecId::Vp8));
        assert_eq!(video_codec(*b"DIV3"), None);
        assert!(is_raw([0; 4]) && is_raw(*b"YUY2") && !is_raw(*b"MJPG"));
    }

    #[test]
    fn the_bytes_after_the_bitmap_header_are_the_extradata() {
        let main = MainHeader::default();
        let video = describe_video(&bih(b"H264", 64, 48, &[0, 0, 0, 1, 0x67]), &main).unwrap();
        assert_eq!(video.params.codec(), &StreamCodec::Known(CodecId::H264));
        assert_eq!(video.params.extradata(), &[0, 0, 0, 1, 0x67]);
        assert_eq!(video.params.size().unwrap().height(), 48);
        assert!(!video.presentation);
        let raw = describe_video(&bih(&[0; 4], 8, -8, &[]), &main).unwrap();
        assert!(raw.params.codec().is_unknown() && raw.intra && raw.presentation);
        assert_eq!(raw.params.size().unwrap().height(), 8);
        let jpeg = describe_video(&bih(b"MJPG", 8, 8, b"junk"), &main).unwrap();
        assert!(jpeg.params.extradata().is_empty());
        let short = describe_video(&[0; 12], &main).unwrap();
        assert!(short.params.codec().is_unknown());
    }

    #[test]
    fn pcm_gets_the_wave_description_and_aac_its_config() {
        let mut pcm = Vec::new();
        for value in [1u16, 2] {
            pcm.extend_from_slice(&value.to_le_bytes());
        }
        pcm.extend_from_slice(&44_100u32.to_le_bytes());
        pcm.extend_from_slice(&176_400u32.to_le_bytes());
        pcm.extend_from_slice(&4u16.to_le_bytes());
        pcm.extend_from_slice(&16u16.to_le_bytes());
        let audio = describe_audio(&pcm).unwrap();
        assert_eq!(audio.codec, Some(CodecId::Pcm));
        assert_eq!(audio.params.extradata().len(), 16);
        assert_eq!(audio.block_align, 4);
        let mut aac = pcm.clone();
        aac[0] = 0xFF;
        aac.extend_from_slice(&2u16.to_le_bytes());
        aac.extend_from_slice(&[0x12, 0x10]);
        let audio = describe_audio(&aac).unwrap();
        assert_eq!(audio.codec, Some(CodecId::Aac));
        assert_eq!(audio.params.extradata(), &[0x12, 0x10]);
        let broken = describe_audio(&[0x55, 0x00, 0, 0]).unwrap();
        assert!(broken.params.codec().is_unknown());
    }
}
