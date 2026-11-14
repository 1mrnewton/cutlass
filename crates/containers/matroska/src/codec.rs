//! Matroska CodecID strings to codec ids and extradata, and back.
//!
//! Source: the Matroska codec mappings
//! (<https://www.matroska.org/technical/codec_specs.html>), checked against
//! FFmpeg's `matroska.c` table and its `matroskadec.c` and `matroskaenc.c`
//! handling of AAC, ALAC, PCM, and ProRes.
//!
//! The container decides which bytes are extradata. Three CodecIDs need the
//! bytes shaped for the codec crates:
//!
//! - `A_AAC/MPEG2/*` and `A_AAC/MPEG4/*` have no CodecPrivate; the profile is
//!   in the CodecID. An AudioSpecificConfig is built from it, as FFmpeg does.
//! - `A_PCM/*` names the coding in the CodecID and BitDepth. The extradata
//!   is the three layout bytes `cutlass-pcm` reads: coding (1 s16, 2 s24,
//!   3 s32, 4 f32, 5 f64, 6 u8), byte order (0 little, 1 big), arrangement (0).
//! - `V_PRORES` frames are stored without their first eight bytes (the frame
//!   size and `icpf`); [`FrameFix::ProRes`] puts them back on read and the
//!   muxer takes them off on write.

use cutlass_codec::{CodecId, FrameSize, StreamCodec, StreamParams, UnknownCodec};

use crate::error::{Error, Result};
use crate::limits::MAX_CODEC_ID;
use crate::meta::{Encoding, TrackEntry, TrackKind};

/// A change the demuxer makes to each frame of a track.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FrameFix {
    None,
    /// Header stripping: these bytes go back in front of each frame.
    Prepend(Vec<u8>),
    /// ProRes: the frame size and `icpf` go back in front unless the frame
    /// already starts with them.
    ProRes,
}

/// The codec side of one track.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mapped {
    pub params: StreamParams,
    pub fix: FrameFix,
    /// The OpusHead pre-skip, in 48 kHz samples.
    pub pre_skip: Option<u16>,
}

/// The Matroska CodecID and the extradata for one track.
pub fn map_track(track: &TrackEntry) -> Result<Mapped> {
    let id = track.codec_id.as_slice();
    let mut private = track.codec_private.clone();
    let mut fix = FrameFix::None;
    match &track.encoding {
        Encoding::None => {}
        Encoding::HeaderStripping(bytes) => {
            if track.private_encoded {
                let mut whole = bytes.clone();
                whole.extend_from_slice(&private);
                private = whole;
            }
            if !bytes.is_empty() {
                fix = FrameFix::Prepend(bytes.clone());
            }
        }
        Encoding::Opaque(what) => {
            let mut tag = id.to_vec();
            tag.push(b'+');
            tag.extend_from_slice(what.as_bytes());
            return Ok(Mapped {
                params: params_for(track, unknown(&tag)?, private),
                fix,
                pre_skip: None,
            });
        }
    }
    let (codec, extradata) = match id {
        b"V_MPEG4/ISO/AVC" => (CodecId::H264, private),
        b"V_MPEGH/ISO/HEVC" => (CodecId::Hevc, private),
        b"V_AV1" => (CodecId::Av1, private),
        b"V_VP8" => (CodecId::Vp8, Vec::new()),
        // VP9 CodecPrivate is the WebM feature list, not a `vpcC` record, and
        // a decoder rebuilds it from the first frame. FFmpeg drops it too.
        b"V_VP9" => (CodecId::Vp9, Vec::new()),
        b"V_MPEG1" | b"V_MPEG2" => (CodecId::Mpeg2, private),
        b"V_MPEG4/ISO/SP" | b"V_MPEG4/ISO/ASP" | b"V_MPEG4/ISO/AP" => (CodecId::Mpeg4, private),
        b"V_MJPEG" => (CodecId::Mjpeg, Vec::new()),
        b"V_PRORES" => {
            if fix == FrameFix::None {
                fix = FrameFix::ProRes;
            }
            (CodecId::Prores, private)
        }
        b"V_MS/VFW/FOURCC" => return vfw(track, &private, fix),
        b"A_AAC" => {
            if private.is_empty() {
                (CodecId::Aac, legacy_aac(track, b"A_AAC/MPEG4/LC"))
            } else {
                (CodecId::Aac, private)
            }
        }
        _ if id.starts_with(b"A_AAC/") => (CodecId::Aac, legacy_aac(track, id)),
        b"A_OPUS" | b"A_OPUS/EXPERIMENTAL" => (CodecId::Opus, private),
        b"A_VORBIS" => (CodecId::Vorbis, private),
        b"A_FLAC" => (CodecId::Flac, private),
        b"A_MPEG/L3" => (CodecId::Mp3, Vec::new()),
        b"A_AC3" | b"A_AC3/BSID9" | b"A_AC3/BSID10" => (CodecId::Ac3, Vec::new()),
        b"A_EAC3" => (CodecId::Eac3, Vec::new()),
        b"A_ALAC" => (CodecId::Alac, private),
        b"A_PCM/INT/LIT" | b"A_PCM/INT/BIG" | b"A_PCM/FLOAT/IEEE" => {
            match pcm_layout(id, track.bit_depth) {
                Some(layout) => (CodecId::Pcm, layout.to_vec()),
                None => {
                    return Ok(Mapped {
                        params: params_for(track, unknown(id)?, private),
                        fix,
                        pre_skip: None,
                    });
                }
            }
        }
        b"A_MS/ACM" => return acm(track, &private, fix),
        _ => {
            return Ok(Mapped {
                params: params_for(track, unknown(id)?, private),
                fix,
                pre_skip: None,
            });
        }
    };
    let pre_skip = if codec == CodecId::Opus {
        opus_pre_skip(&extradata)
    } else {
        None
    };
    Ok(Mapped {
        params: params_for(track, StreamCodec::Known(codec), extradata),
        fix,
        pre_skip,
    })
}

fn unknown(tag: &[u8]) -> Result<StreamCodec> {
    let cut = tag.get(..MAX_CODEC_ID.min(tag.len())).unwrap_or(tag);
    UnknownCodec::new(cut.to_vec())
        .map(StreamCodec::Unknown)
        .map_err(|error| Error::malformed(error.to_string()))
}

/// Stream parameters from the track's Video or Audio element. A picture
/// with no size, or sound with no rate or channels, is kept as an opaque
/// stream rather than dropped.
fn params_for(track: &TrackEntry, codec: StreamCodec, extradata: Vec<u8>) -> StreamParams {
    match track.kind {
        TrackKind::Video => {
            let width = u32::try_from(track.pixel_width).unwrap_or(0);
            let height = u32::try_from(track.pixel_height).unwrap_or(0);
            match FrameSize::try_new(width, height) {
                Ok(size) => StreamParams::video(codec, size, extradata),
                Err(_) => StreamParams::opaque(codec, extradata),
            }
        }
        TrackKind::Audio => {
            let rate = audio_rate(track, &codec);
            let channels = u16::try_from(track.channels).unwrap_or(0);
            match rate {
                Some(rate) => StreamParams::try_audio(codec.clone(), rate, channels, extradata)
                    .unwrap_or_else(|_| StreamParams::opaque(codec, Vec::new())),
                None => StreamParams::opaque(codec, extradata),
            }
        }
        TrackKind::Other(_) => StreamParams::opaque(codec, extradata),
    }
}

/// The rate the decoder plays at: 48 kHz for Opus (RFC 7845, 5.1: the
/// stored rate is the encoder's input), the OutputSamplingFrequency when
/// the file gives one (AAC with SBR), and the SamplingFrequency otherwise.
fn audio_rate(track: &TrackEntry, codec: &StreamCodec) -> Option<u32> {
    if *codec == StreamCodec::Known(CodecId::Opus) {
        return Some(48_000);
    }
    let rate = track
        .output_sampling_frequency
        .filter(|rate| rate.is_finite() && *rate > 0.0)
        .unwrap_or(track.sampling_frequency);
    rate_u32(rate)
}

fn rate_u32(rate: f64) -> Option<u32> {
    if !rate.is_finite() || rate < 1.0 || rate > f64::from(u32::MAX) {
        return None;
    }
    // In range and integral after rounding, so the cast is exact.
    Some(rate.round() as u32)
}

/// The three PCM layout bytes for an `A_PCM/*` CodecID and BitDepth.
/// FFmpeg reads a missing BitDepth as 16 for integers; floats default to 32.
/// 8-bit integer PCM is unsigned (coding 11) in either byte order, as
/// FFmpeg's `matroskadec.c` reads it (`AV_CODEC_ID_PCM_U8`).
#[must_use]
pub fn pcm_layout(id: &[u8], bits: Option<u64>) -> Option<[u8; 3]> {
    match id {
        b"A_PCM/INT/LIT" | b"A_PCM/INT/BIG" => {
            let big = u8::from(id == b"A_PCM/INT/BIG");
            let coding = match bits.unwrap_or(16) {
                8 => 11,
                16 => 1,
                24 => 2,
                32 => 3,
                _ => return None,
            };
            Some([coding, big, 0])
        }
        b"A_PCM/FLOAT/IEEE" => match bits.unwrap_or(32) {
            32 => Some([4, 0, 0]),
            64 => Some([5, 0, 0]),
            _ => None,
        },
        _ => None,
    }
}

/// AAC sampling frequency index table (ISO/IEC 14496-3, 1.6.3.4).
const AAC_RATES: [u32; 13] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
    7_350,
];

/// An AudioSpecificConfig for a legacy `A_AAC/MPEG*/<profile>` CodecID.
///
/// The object type is MAIN 1, LC 2, SSR 3, LTP 4. `/SBR` adds the explicit
/// SBR signal (sync extension 0x2B7, object type 5) with the output rate,
/// as FFmpeg writes it.
#[must_use]
pub fn legacy_aac(track: &TrackEntry, id: &[u8]) -> Vec<u8> {
    let contains = |needle: &[u8]| id.windows(needle.len()).any(|window| window == needle);
    let object = if contains(b"MAIN") {
        1
    } else if contains(b"LC") {
        2
    } else if contains(b"SSR") {
        3
    } else {
        4
    };
    let rate = rate_u32(track.sampling_frequency).unwrap_or(0);
    let channels = u32::try_from(track.channels.min(15)).unwrap_or(0);
    let mut bits = Bits::default();
    bits.put(object, 5);
    put_rate(&mut bits, rate);
    bits.put(channels, 4);
    bits.put(0, 3);
    if contains(b"SBR") {
        let output = track
            .output_sampling_frequency
            .and_then(rate_u32)
            .unwrap_or(rate.saturating_mul(2));
        bits.put(0x2B7, 11);
        bits.put(5, 5);
        bits.put(1, 1);
        put_rate(&mut bits, output);
    }
    bits.finish()
}

fn put_rate(bits: &mut Bits, rate: u32) {
    match AAC_RATES.iter().position(|known| *known == rate) {
        Some(index) => bits.put(u32::try_from(index).unwrap_or(15), 4),
        None => {
            bits.put(15, 4);
            bits.put(rate & 0x00FF_FFFF, 24);
        }
    }
}

#[derive(Default)]
struct Bits {
    out: Vec<u8>,
    acc: u64,
    used: u32,
}

impl Bits {
    fn put(&mut self, value: u32, width: u32) {
        let mask = if width >= 32 {
            u32::MAX
        } else {
            (1 << width) - 1
        };
        self.acc = (self.acc << width) | u64::from(value & mask);
        self.used += width;
        while self.used >= 8 {
            self.used -= 8;
            self.out
                .push(u8::try_from((self.acc >> self.used) & 0xFF).unwrap_or(0));
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.used > 0 {
            let pad = 8 - self.used;
            self.put(0, pad);
        }
        self.out
    }
}

pub(crate) fn opus_pre_skip(head: &[u8]) -> Option<u16> {
    if head.len() < 19 || head.get(..8) != Some(b"OpusHead") {
        return None;
    }
    Some(u16::from_le_bytes([*head.get(10)?, *head.get(11)?]))
}

/// Samples at 48 kHz an Opus packet decodes to, from its TOC byte and frame
/// count (RFC 6716, 3.1). `None` for an empty packet, a code-3 packet with
/// no count byte, or one longer than the 120 ms the format allows.
#[must_use]
pub fn opus_packet_samples(packet: &[u8]) -> Option<u32> {
    let toc = *packet.first()?;
    let config = toc >> 3;
    let frame = match config {
        // SILK: 10, 20, 40, 60 ms.
        0..=11 => [480, 960, 1920, 2880][usize::from(config % 4)],
        // Hybrid: 10, 20 ms.
        12..=15 => [480, 960][usize::from(config % 2)],
        // CELT: 2.5, 5, 10, 20 ms.
        _ => [120, 240, 480, 960][usize::from(config % 4)],
    };
    let frames = match toc & 3 {
        0 => 1,
        1 | 2 => 2,
        _ => u32::from(*packet.get(1)? & 0x3F),
    };
    let samples = frame * frames;
    (samples > 0 && samples <= 5760).then_some(samples)
}

/// `V_MS/VFW/FOURCC`: a BITMAPINFOHEADER. The compression FourCC names the
/// codec; bytes after the 40-byte header are its extradata.
fn vfw(track: &TrackEntry, private: &[u8], fix: FrameFix) -> Result<Mapped> {
    let fourcc = private.get(16..20).unwrap_or(&[]);
    let extra = private.get(40..).unwrap_or(&[]).to_vec();
    let codec = match fourcc {
        b"MJPG" | b"mjpg" | b"AVRn" | b"mjpa" => Some((CodecId::Mjpeg, Vec::new())),
        b"FMP4" | b"fmp4" | b"XVID" | b"xvid" | b"DIVX" | b"divx" | b"DX50" | b"MP4V" | b"mp4v" => {
            Some((CodecId::Mpeg4, extra))
        }
        b"H264" | b"h264" | b"X264" | b"x264" | b"AVC1" | b"avc1" => Some((CodecId::H264, extra)),
        b"MPG2" | b"mpg2" | b"MPEG" => Some((CodecId::Mpeg2, extra)),
        _ => None,
    };
    let mut sized = track.clone();
    if sized.pixel_width == 0 || sized.pixel_height == 0 {
        let word = |at: usize| {
            private.get(at..at + 4).map_or(0, |raw| {
                i32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])
            })
        };
        sized.pixel_width = u64::from(word(4).unsigned_abs());
        sized.pixel_height = u64::from(word(8).unsigned_abs());
    }
    Ok(match codec {
        Some((codec, extradata)) => Mapped {
            params: params_for(&sized, StreamCodec::Known(codec), extradata),
            fix,
            pre_skip: None,
        },
        None => {
            let mut tag = b"V_MS/VFW/FOURCC:".to_vec();
            tag.extend_from_slice(fourcc);
            Mapped {
                params: params_for(&sized, unknown(&tag)?, private.to_vec()),
                fix,
                pre_skip: None,
            }
        }
    })
}

/// `A_MS/ACM`: a WAVEFORMATEX. PCM, float, MP3, and AC-3 tags map to codec
/// ids; WAVE_FORMAT_EXTENSIBLE takes the tag from its SubFormat GUID.
fn acm(track: &TrackEntry, private: &[u8], fix: FrameFix) -> Result<Mapped> {
    let half = |at: usize| {
        private
            .get(at..at + 2)
            .map_or(0, |raw| u16::from_le_bytes([raw[0], raw[1]]))
    };
    let mut tag = half(0);
    let bits = half(14);
    if tag == 0xFFFE && half(16) >= 22 {
        tag = half(24);
    }
    let mut sized = track.clone();
    if private.len() >= 16 {
        sized.channels = u64::from(half(2));
        let rate = private.get(4..8).map_or(0, |raw| {
            u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])
        });
        sized.sampling_frequency = f64::from(rate);
    }
    let codec = match tag {
        0x0001 => pcm_layout(b"A_PCM/INT/LIT", Some(u64::from(bits)))
            .map(|layout| (CodecId::Pcm, layout.to_vec())),
        0x0003 => pcm_layout(b"A_PCM/FLOAT/IEEE", Some(u64::from(bits)))
            .map(|layout| (CodecId::Pcm, layout.to_vec())),
        0x0055 => Some((CodecId::Mp3, Vec::new())),
        0x2000 => Some((CodecId::Ac3, Vec::new())),
        _ => None,
    };
    Ok(match codec {
        Some((codec, extradata)) => Mapped {
            params: params_for(&sized, StreamCodec::Known(codec), extradata),
            fix,
            pre_skip: None,
        },
        None => {
            let label = format!("A_MS/ACM:{tag:04x}");
            Mapped {
                params: params_for(&sized, unknown(label.as_bytes())?, private.to_vec()),
                fix,
                pre_skip: None,
            }
        }
    })
}

// Writing.

/// What the muxer writes for one track's codec.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Written {
    pub codec_id: &'static str,
    pub private: Vec<u8>,
    pub bit_depth: Option<u64>,
    /// Frames lose their first eight bytes (ProRes).
    pub strip_prores: bool,
    pub video: bool,
}

/// Whether a muxer stores `codec` at all. `webm` limits the codecs to VP8,
/// VP9, AV1, Opus, and Vorbis; still image codecs are never a track.
pub fn allowed(codec: CodecId, webm: bool) -> Result<()> {
    if matches!(
        codec,
        CodecId::Jpeg | CodecId::Png | CodecId::Webp | CodecId::Gif
    ) {
        return Err(Error::unsupported(format!(
            "{codec:?} is a still image codec"
        )));
    }
    let allowed_in_webm = matches!(
        codec,
        CodecId::Vp8 | CodecId::Vp9 | CodecId::Av1 | CodecId::Opus | CodecId::Vorbis
    );
    if webm && !allowed_in_webm {
        return Err(Error::unsupported(format!(
            "{codec:?} is not a WebM codec (VP8, VP9, AV1, Opus, Vorbis)"
        )));
    }
    Ok(())
}

/// The CodecID and CodecPrivate for a stream, or why the muxer refuses it.
/// `webm` limits the codecs to VP8, VP9, AV1, Opus, and Vorbis.
pub fn written(codec: CodecId, extradata: &[u8], webm: bool) -> Result<Written> {
    allowed(codec, webm)?;
    let keep = extradata.to_vec();
    let video = |codec_id, private| Written {
        codec_id,
        private,
        bit_depth: None,
        strip_prores: false,
        video: true,
    };
    let audio = |codec_id, private| Written {
        codec_id,
        private,
        bit_depth: None,
        strip_prores: false,
        video: false,
    };
    let written = match codec {
        CodecId::H264 => video("V_MPEG4/ISO/AVC", keep),
        CodecId::Hevc => video("V_MPEGH/ISO/HEVC", keep),
        CodecId::Av1 => video("V_AV1", keep),
        CodecId::Vp8 => video("V_VP8", Vec::new()),
        CodecId::Vp9 => video("V_VP9", Vec::new()),
        CodecId::Mpeg4 => video("V_MPEG4/ISO/ASP", keep),
        CodecId::Mpeg2 => video("V_MPEG2", keep),
        CodecId::Mjpeg => video("V_MJPEG", Vec::new()),
        CodecId::Prores => {
            let private = if keep.len() == 4 {
                keep
            } else {
                b"apcn".to_vec()
            };
            Written {
                strip_prores: true,
                ..video("V_PRORES", private)
            }
        }
        CodecId::Aac => audio("A_AAC", keep),
        CodecId::Opus => audio("A_OPUS", keep),
        CodecId::Vorbis => audio("A_VORBIS", keep),
        CodecId::Flac => audio("A_FLAC", flac_private(&keep)),
        CodecId::Mp3 => audio("A_MPEG/L3", Vec::new()),
        CodecId::Ac3 => audio("A_AC3", Vec::new()),
        CodecId::Eac3 => audio("A_EAC3", Vec::new()),
        CodecId::Alac => audio("A_ALAC", keep),
        CodecId::Pcm => {
            let layout = match extradata {
                [] => [1, 0, 0],
                [coding, order, 0] => [*coding, *order, 0],
                _ => return Err(Error::unsupported("PCM layout other than three bytes")),
            };
            let (codec_id, bits) = match layout {
                [1, 0, _] => ("A_PCM/INT/LIT", 16),
                [2, 0, _] => ("A_PCM/INT/LIT", 24),
                [3, 0, _] => ("A_PCM/INT/LIT", 32),
                [11, _, _] => ("A_PCM/INT/LIT", 8),
                [1, 1, _] => ("A_PCM/INT/BIG", 16),
                [2, 1, _] => ("A_PCM/INT/BIG", 24),
                [3, 1, _] => ("A_PCM/INT/BIG", 32),
                [4, 0, _] => ("A_PCM/FLOAT/IEEE", 32),
                [5, 0, _] => ("A_PCM/FLOAT/IEEE", 64),
                _ => {
                    return Err(Error::unsupported(
                        "this PCM layout has no Matroska CodecID",
                    ));
                }
            };
            Written {
                bit_depth: Some(bits),
                ..audio(codec_id, Vec::new())
            }
        }
        CodecId::Jpeg | CodecId::Png | CodecId::Webp | CodecId::Gif => {
            return Err(Error::unsupported("a still image codec is not a track"));
        }
        CodecId::RawVideo => {
            // V_UNCOMPRESSED needs a pixel layout the stream parameters do
            // not carry yet.
            return Err(Error::unsupported("raw video is not written to Matroska"));
        }
    };
    let needs_private = matches!(
        codec,
        CodecId::H264
            | CodecId::Hevc
            | CodecId::Av1
            | CodecId::Aac
            | CodecId::Opus
            | CodecId::Vorbis
            | CodecId::Flac
            | CodecId::Alac
    );
    if needs_private && written.private.is_empty() {
        return Err(Error::unsupported(format!(
            "{codec:?} needs its codec configuration before the first packet"
        )));
    }
    Ok(written)
}

/// The bare AudioSpecificConfig `A_AAC` stores. A configuration wrapped in
/// an MPEG-4 ES descriptor (ISO/IEC 14496-1, 7.2.6.5), as some encoders hand
/// it over, is unwrapped to the DecoderSpecificInfo inside.
pub fn aac_config(config: &[u8]) -> Result<Vec<u8>> {
    if config.first() != Some(&0x03) {
        return Ok(config.to_vec());
    }
    let short = || Error::malformed("short MPEG-4 ES descriptor");
    let mut at = 0;
    while at < config.len() {
        let tag = config[at];
        at += 1;
        let mut length = 0_usize;
        for _ in 0..4 {
            let byte = *config.get(at).ok_or_else(short)?;
            at += 1;
            length = (length << 7) | usize::from(byte & 0x7F);
            if byte & 0x80 == 0 {
                break;
            }
        }
        match tag {
            // ES_Descriptor: ES_ID, then flags that say which fields follow.
            0x03 => {
                let flags = *config.get(at + 2).ok_or_else(short)?;
                at += 3;
                if flags & 0x80 != 0 {
                    at += 2;
                }
                if flags & 0x40 != 0 {
                    let url = usize::from(*config.get(at).ok_or_else(short)?);
                    at += 1 + url;
                }
                if flags & 0x20 != 0 {
                    at += 2;
                }
            }
            // DecoderConfigDescriptor: 13 fixed bytes, then its children.
            0x04 => at += 13,
            0x05 => {
                let end = at.checked_add(length).ok_or_else(short)?;
                return config.get(at..end).map(<[u8]>::to_vec).ok_or_else(short);
            }
            _ => at = at.checked_add(length).ok_or_else(short)?,
        }
    }
    Err(Error::malformed("ES descriptor has no DecoderSpecificInfo"))
}

/// The `OpusHead` `A_OPUS` stores. An `OpusHead` stays as it is; a `dOps`
/// body (ISO BMFF Opus mapping, big-endian fields) becomes an `OpusHead`.
pub fn opus_head(config: &[u8]) -> Result<Vec<u8>> {
    if config.starts_with(b"OpusHead") {
        return Ok(config.to_vec());
    }
    let [
        0,
        channels,
        skip0,
        skip1,
        rate0,
        rate1,
        rate2,
        rate3,
        gain0,
        gain1,
        family,
        rest @ ..,
    ] = config
    else {
        return Err(Error::malformed(
            "Opus configuration is not an OpusHead or dOps",
        ));
    };
    let mut head = b"OpusHead".to_vec();
    head.extend_from_slice(&[1, *channels, *skip1, *skip0]);
    head.extend_from_slice(&[*rate3, *rate2, *rate1, *rate0]);
    head.extend_from_slice(&[*gain1, *gain0, *family]);
    head.extend_from_slice(rest);
    Ok(head)
}

/// `A_FLAC` CodecPrivate is the `fLaC` marker and the metadata blocks. A
/// bare 34-byte STREAMINFO gets the marker and a last-block header.
fn flac_private(extra: &[u8]) -> Vec<u8> {
    if extra.len() == 34 {
        let mut out = b"fLaC".to_vec();
        out.extend_from_slice(&[0x80, 0, 0, 34]);
        out.extend_from_slice(extra);
        return out;
    }
    extra.to_vec()
}

#[cfg(test)]
mod tests {
    use super::{
        FrameFix, legacy_aac, map_track, opus_packet_samples, pcm_layout, rate_u32, written,
    };
    use crate::meta::{Encoding, TrackEntry, TrackKind};
    use cutlass_codec::{CodecId, StreamCodec};

    #[test]
    fn opus_packets_count_their_samples_from_the_toc() {
        // SILK 60 ms, one frame.
        assert_eq!(opus_packet_samples(&[3 << 3]), Some(2880));
        // Hybrid 20 ms, two frames.
        assert_eq!(opus_packet_samples(&[(13 << 3) | 1, 0]), Some(1920));
        // CELT 2.5 ms, code 3 with six frames.
        assert_eq!(opus_packet_samples(&[(28 << 3) | 3, 6]), Some(720));
        // CELT 20 ms, code 3 with seven frames: 140 ms, more than allowed.
        assert_eq!(opus_packet_samples(&[(31 << 3) | 3, 7]), None);
        // Code 3 with no count byte, a zero count, and nothing at all.
        assert_eq!(opus_packet_samples(&[(31 << 3) | 3]), None);
        assert_eq!(opus_packet_samples(&[(31 << 3) | 3, 0]), None);
        assert_eq!(opus_packet_samples(&[]), None);
    }

    fn track(id: &[u8], kind: TrackKind) -> TrackEntry {
        TrackEntry {
            number: 1,
            uid: 1,
            kind,
            codec_id: id.to_vec(),
            codec_private: Vec::new(),
            default_duration: None,
            codec_delay: 0,
            seek_pre_roll: 0,
            language: None,
            lacing: true,
            pixel_width: 64,
            pixel_height: 48,
            display_width: None,
            display_height: None,
            sampling_frequency: 44_100.0,
            output_sampling_frequency: None,
            channels: 2,
            bit_depth: None,
            encoding: Encoding::None,
            private_encoded: false,
            track_timestamp_scale: 1.0,
        }
    }

    #[test]
    fn video_ids_map_with_their_private_data() {
        let mut avc = track(b"V_MPEG4/ISO/AVC", TrackKind::Video);
        avc.codec_private = vec![1, 0x64, 0, 0x1F];
        let mapped = map_track(&avc).expect("avc");
        assert_eq!(mapped.params.codec(), &StreamCodec::Known(CodecId::H264));
        assert_eq!(mapped.params.extradata(), &[1, 0x64, 0, 0x1F]);
        assert_eq!(mapped.params.size().expect("size").width(), 64);

        let mut vp9 = track(b"V_VP9", TrackKind::Video);
        vp9.codec_private = vec![1, 1, 0];
        assert!(map_track(&vp9).expect("vp9").params.extradata().is_empty());

        let prores = map_track(&track(b"V_PRORES", TrackKind::Video)).expect("prores");
        assert_eq!(prores.fix, FrameFix::ProRes);
    }

    #[test]
    fn an_unknown_id_stays_on_the_stream() {
        let mapped = map_track(&track(b"S_TEXT/UTF8", TrackKind::Other(17))).expect("text");
        assert!(mapped.params.codec().is_unknown());
        let long = vec![b'X'; 200];
        let mapped = map_track(&track(&long, TrackKind::Video)).expect("long");
        let StreamCodec::Unknown(tag) = mapped.params.codec() else {
            panic!("known");
        };
        assert_eq!(tag.tag().len(), 64);
    }

    #[test]
    fn an_opaque_encoding_is_reported_as_unknown() {
        let mut zlib = track(b"A_AAC", TrackKind::Audio);
        zlib.encoding = Encoding::Opaque("zlib");
        let mapped = map_track(&zlib).expect("zlib");
        let StreamCodec::Unknown(tag) = mapped.params.codec() else {
            panic!("known");
        };
        assert_eq!(tag.tag(), b"A_AAC+zlib");
    }

    #[test]
    fn header_stripping_also_restores_an_encoded_codec_private() {
        let mut mp3 = track(b"A_MPEG/L3", TrackKind::Audio);
        mp3.encoding = Encoding::HeaderStripping(vec![0xFF, 0xFB]);
        let mapped = map_track(&mp3).expect("mp3");
        assert_eq!(mapped.fix, FrameFix::Prepend(vec![0xFF, 0xFB]));
        let mut aac = track(b"A_AAC", TrackKind::Audio);
        aac.codec_private = vec![0x90];
        aac.encoding = Encoding::HeaderStripping(vec![0x12]);
        aac.private_encoded = true;
        assert_eq!(
            map_track(&aac).expect("aac").params.extradata(),
            &[0x12, 0x90]
        );
    }

    #[test]
    fn legacy_aac_ids_build_an_audio_specific_config() {
        let lc = track(b"A_AAC/MPEG4/LC", TrackKind::Audio);
        // LC (2), 44.1 kHz (index 4), stereo: 00010 0100 0010 000.
        assert_eq!(legacy_aac(&lc, b"A_AAC/MPEG4/LC"), vec![0x12, 0x10]);
        let mut sbr = track(b"A_AAC/MPEG4/LC/SBR", TrackKind::Audio);
        sbr.sampling_frequency = 24_000.0;
        sbr.output_sampling_frequency = Some(48_000.0);
        // The same bytes FFmpeg builds: 0x13 0x10 then 0x56 0xE5 0x98.
        assert_eq!(
            legacy_aac(&sbr, b"A_AAC/MPEG4/LC/SBR"),
            vec![0x13, 0x10, 0x56, 0xE5, 0x98]
        );
        let mapped = map_track(&lc).expect("lc");
        assert_eq!(mapped.params.extradata(), &[0x12, 0x10]);
    }

    #[test]
    fn pcm_layouts_follow_the_id_and_bit_depth() {
        assert_eq!(pcm_layout(b"A_PCM/INT/LIT", None), Some([1, 0, 0]));
        assert_eq!(pcm_layout(b"A_PCM/INT/BIG", Some(24)), Some([2, 1, 0]));
        assert_eq!(pcm_layout(b"A_PCM/INT/LIT", Some(8)), Some([11, 0, 0]));
        assert_eq!(pcm_layout(b"A_PCM/INT/BIG", Some(8)), Some([11, 1, 0]));
        assert_eq!(pcm_layout(b"A_PCM/FLOAT/IEEE", Some(64)), Some([5, 0, 0]));
        assert_eq!(pcm_layout(b"A_PCM/INT/LIT", Some(20)), None);
        let mut odd = track(b"A_PCM/INT/LIT", TrackKind::Audio);
        odd.bit_depth = Some(20);
        assert!(map_track(&odd).expect("odd").params.codec().is_unknown());
    }

    #[test]
    fn eight_bit_pcm_is_written_unsigned_and_signed_is_refused() {
        let unsigned = written(CodecId::Pcm, &[11, 0, 0], false).expect("u8");
        assert_eq!(unsigned.codec_id, "A_PCM/INT/LIT");
        assert_eq!(unsigned.bit_depth, Some(8));
        // Matroska has no signed 8-bit PCM, and a container does not
        // convert samples.
        assert!(written(CodecId::Pcm, &[6, 0, 0], false).is_err());
    }

    #[test]
    fn opus_plays_at_48_khz_and_names_its_pre_skip() {
        let mut opus = track(b"A_OPUS", TrackKind::Audio);
        opus.sampling_frequency = 44_100.0;
        let mut head = b"OpusHead".to_vec();
        head.extend_from_slice(&[1, 2, 0x38, 0x01, 0x44, 0xAC, 0, 0, 0, 0, 0]);
        opus.codec_private = head;
        let mapped = map_track(&opus).expect("opus");
        assert_eq!(mapped.params.sample_rate(), Some(48_000));
        assert_eq!(mapped.pre_skip, Some(312));
    }

    #[test]
    fn a_sound_track_without_a_rate_is_opaque_and_kept() {
        let mut bad = track(b"A_MPEG/L3", TrackKind::Audio);
        bad.sampling_frequency = f64::NAN;
        let mapped = map_track(&bad).expect("bad");
        assert_eq!(mapped.params.sample_rate(), None);
        assert_eq!(mapped.params.codec(), &StreamCodec::Known(CodecId::Mp3));
        assert_eq!(rate_u32(48_000.4), Some(48_000));
        assert_eq!(rate_u32(-1.0), None);
        assert_eq!(rate_u32(f64::INFINITY), None);
    }

    #[test]
    fn the_writer_refuses_codecs_outside_webm_and_missing_configuration() {
        assert!(written(CodecId::H264, &[1], true).is_err());
        assert!(written(CodecId::H264, &[], false).is_err());
        assert_eq!(
            written(CodecId::Vp9, &[1, 2], true).expect("vp9").codec_id,
            "V_VP9"
        );
        let pcm = written(CodecId::Pcm, &[2, 1, 0], false).expect("pcm");
        assert_eq!(pcm.codec_id, "A_PCM/INT/BIG");
        assert_eq!(pcm.bit_depth, Some(24));
        let flac = written(CodecId::Flac, &[0; 34], false).expect("flac");
        assert_eq!(&flac.private[..8], b"fLaC\x80\0\0\x22");
        assert!(written(CodecId::Png, &[], false).is_err());
        assert!(
            written(CodecId::Prores, b"apch", false)
                .expect("prores")
                .strip_prores
        );
    }
}
