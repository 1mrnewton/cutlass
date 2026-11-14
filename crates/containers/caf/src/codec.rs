//! `mFormatID` to a codec id and the extradata its codec crate reads.
//!
//! Linear PCM extradata is the three-byte layout `cutlass-pcm` reads:
//! coding, endian (1 is big), arrangement (0, interleaved). Coding 1 to 5
//! are s16, s24, s32, f32 and f64; 6 is signed 8-bit, 7 a-law, 8 µ-law,
//! 9 s64. CAF integer PCM is signed. A sample takes `mBytesPerPacket` over
//! the channel count bytes and `mFramesPerPacket` is 1.
//!
//! When `mBitsPerChannel` is less than that word, the sample is unpacked.
//! The CAF `desc` has no alignment flag; AudioToolbox reports such a file as
//! high-aligned and decodes the whole word, low bits included (checked for
//! 8 in 2, 12 in 2, 16 in 4, 20 in 3, 20 in 4 and 24 in 4 bytes against
//! `afconvert -d LEF32`). So an unpacked sample maps to the integer coding
//! of its word: 2 bytes s16, 3 bytes s24, 4 bytes s32. A word of 8 bytes is
//! s64 only when all 64 bits are used, since AudioToolbox refuses to decode
//! narrower samples in 8 bytes and there is no reference for them. Any
//! other layout, including unpacked float, is an unknown `lpcm` stream.

use cutlass_codec::{CodecId, StreamCodec, UnknownCodec};
use cutlass_container::OpenError;

use crate::cookie;
use crate::desc::{Description, FLAG_FLOAT, FLAG_LITTLE_ENDIAN};
use crate::error::malformed;

/// The codec and extradata of one stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Mapped {
    pub codec: StreamCodec,
    pub extradata: Vec<u8>,
}

fn unknown(tag: &[u8]) -> Result<StreamCodec, OpenError> {
    UnknownCodec::new(tag.to_vec())
        .map(StreamCodec::Unknown)
        .map_err(|error| malformed(error.to_string()))
}

fn known(codec: CodecId, extradata: Vec<u8>) -> Mapped {
    Mapped {
        codec: StreamCodec::Known(codec),
        extradata,
    }
}

/// The PCM coding byte for a linear PCM description, or `None` when the
/// layout is not one `cutlass-pcm` names.
pub(crate) fn pcm_coding(desc: &Description) -> Option<u8> {
    if desc.frames_per_packet != 1
        || desc.channels == 0
        || !desc.bytes_per_packet.is_multiple_of(desc.channels)
    {
        return None;
    }
    let word = desc.bytes_per_packet / desc.channels;
    let bits = desc.bits_per_channel;
    if bits == 0 || bits > word.checked_mul(8)? {
        return None;
    }
    if desc.flags & FLAG_FLOAT != 0 {
        return match (bits, word) {
            (32, 4) => Some(4),
            (64, 8) => Some(5),
            _ => None,
        };
    }
    match (word, bits) {
        (1, _) => Some(6),
        (2, _) => Some(1),
        (3, _) => Some(2),
        (4, _) => Some(3),
        (8, 64) => Some(9),
        _ => None,
    }
}

/// Maps a description and its cookie.
///
/// `priming` is the packet table's priming frames, which an Opus header
/// carries as its pre-skip at 48 kHz.
pub(crate) fn map(
    desc: &Description,
    cookie: Option<&[u8]>,
    priming: u32,
) -> Result<Mapped, OpenError> {
    let little = desc.flags & FLAG_LITTLE_ENDIAN != 0;
    match &desc.format {
        b"lpcm" => Ok(match pcm_coding(desc) {
            Some(coding) => known(CodecId::Pcm, vec![coding, u8::from(!little), 0]),
            None => Mapped {
                codec: unknown(b"lpcm")?,
                extradata: Vec::new(),
            },
        }),
        b"alaw" | b"ulaw" => {
            let coding = if desc.format == *b"alaw" { 7 } else { 8 };
            if desc.frames_per_packet != 1 || desc.bytes_per_packet != desc.channels {
                return Ok(Mapped {
                    codec: unknown(&desc.format)?,
                    extradata: Vec::new(),
                });
            }
            Ok(known(CodecId::Pcm, vec![coding, 1, 0]))
        }
        b"aac " | b"aach" | b"aacp" | b"aacl" | b"aace" | b"aacf" | b"aacg" => {
            let cookie = cookie.ok_or_else(|| malformed("AAC needs a kuki chunk"))?;
            Ok(known(CodecId::Aac, cookie::aac_config(cookie)?))
        }
        b"alac" => {
            let cookie = cookie.ok_or_else(|| malformed("ALAC needs a kuki chunk"))?;
            Ok(known(CodecId::Alac, cookie::alac_config(cookie)?))
        }
        b"flac" => {
            // libavformat writes FLAC without a cookie; the decoder then
            // reads the frame headers.
            let extradata = match cookie {
                Some(cookie) => cookie::flac_streaminfo(cookie)?,
                None => Vec::new(),
            };
            Ok(known(CodecId::Flac, extradata))
        }
        b"opus" => {
            let Ok(channels @ 1..=2) = u8::try_from(desc.channels) else {
                // Several Opus streams need a mapping table this file does
                // not carry in a published form.
                return Ok(Mapped {
                    codec: unknown(b"opus")?,
                    extradata: Vec::new(),
                });
            };
            let pre_skip = u64::from(priming) * 48_000 / u64::from(desc.sample_rate.max(1));
            let pre_skip = u16::try_from(pre_skip).unwrap_or(u16::MAX);
            Ok(known(
                CodecId::Opus,
                cookie::opus_head(channels, pre_skip, desc.sample_rate),
            ))
        }
        // `ms\0U` is the WAVE format tag 0x55 inside CAF, as libavformat maps it.
        b".mp3" | [b'm', b's', 0, 0x55] => Ok(known(CodecId::Mp3, Vec::new())),
        b"ac-3" => Ok(known(CodecId::Ac3, Vec::new())),
        b"ec-3" => Ok(known(CodecId::Eac3, Vec::new())),
        other => Ok(Mapped {
            codec: unknown(other)?,
            extradata: cookie.map(<[u8]>::to_vec).unwrap_or_default(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::{map, pcm_coding};
    use crate::desc::Description;
    use cutlass_codec::{CodecId, StreamCodec};

    fn pcm(bits: u32, flags: u32, channels: u32, bytes: u32) -> Description {
        Description {
            sample_rate: 48_000,
            format: *b"lpcm",
            flags,
            bytes_per_packet: bytes,
            frames_per_packet: 1,
            channels,
            bits_per_channel: bits,
        }
    }

    #[test]
    fn every_pcm_layout_has_a_coding_byte() {
        assert_eq!(pcm_coding(&pcm(16, 2, 2, 4)), Some(1));
        assert_eq!(pcm_coding(&pcm(24, 0, 2, 6)), Some(2));
        assert_eq!(pcm_coding(&pcm(32, 0, 1, 4)), Some(3));
        assert_eq!(pcm_coding(&pcm(32, 1, 1, 4)), Some(4));
        assert_eq!(pcm_coding(&pcm(64, 3, 1, 8)), Some(5));
        assert_eq!(pcm_coding(&pcm(8, 2, 2, 2)), Some(6));
        assert_eq!(pcm_coding(&pcm(64, 0, 1, 8)), Some(9));
        // Unpacked integers take the coding of their word.
        assert_eq!(pcm_coding(&pcm(24, 0, 2, 8)), Some(3));
        assert_eq!(pcm_coding(&pcm(20, 2, 1, 4)), Some(3));
        assert_eq!(pcm_coding(&pcm(20, 2, 1, 3)), Some(2));
        assert_eq!(pcm_coding(&pcm(12, 0, 1, 2)), Some(1));
        assert_eq!(pcm_coding(&pcm(8, 0, 1, 2)), Some(1));
        assert_eq!(pcm_coding(&pcm(4, 0, 1, 1)), Some(6));
        // Not readable: narrower than 64 bits in 8 bytes, wider than the
        // word, zero bits, a word that is not a whole share of the packet,
        // several frames a packet, and float that is not 32 or 64 bits.
        assert_eq!(pcm_coding(&pcm(24, 0, 1, 8)), None);
        assert_eq!(pcm_coding(&pcm(32, 0, 1, 2)), None);
        assert_eq!(pcm_coding(&pcm(0, 0, 1, 2)), None);
        assert_eq!(pcm_coding(&pcm(16, 0, 2, 5)), None);
        assert_eq!(pcm_coding(&pcm(16, 0, 0, 0)), None);
        assert_eq!(pcm_coding(&pcm(16, 1, 1, 2)), None);
        assert_eq!(pcm_coding(&pcm(32, 1, 1, 8)), None);
        assert_eq!(pcm_coding(&pcm(24, 0, 1, 5)), None);
        let mut several = pcm(16, 0, 1, 4);
        several.frames_per_packet = 2;
        assert_eq!(pcm_coding(&several), None);
        let mapped = map(&pcm(16, 2, 2, 4), None, 0).expect("map");
        assert_eq!(mapped.codec, StreamCodec::Known(CodecId::Pcm));
        assert_eq!(mapped.extradata, vec![1, 0, 0]);
        let mapped = map(&pcm(24, 0, 2, 6), None, 0).expect("map");
        assert_eq!(mapped.extradata, vec![2, 1, 0]);
        let mapped = map(&pcm(24, 2, 2, 8), None, 0).expect("map");
        assert_eq!(mapped.extradata, vec![3, 0, 0]);
        assert!(
            map(&pcm(24, 0, 1, 8), None, 0)
                .expect("map")
                .codec
                .is_unknown()
        );
    }

    #[test]
    fn compressed_formats_map_to_their_codec() {
        let mut desc = pcm(0, 0, 2, 0);
        desc.frames_per_packet = 1024;
        desc.format = *b"aac ";
        assert!(map(&desc, None, 0).is_err());
        desc.format = *b"ac-3";
        assert_eq!(
            map(&desc, None, 0).expect("ac3").codec,
            StreamCodec::Known(CodecId::Ac3)
        );
        desc.format = *b".mp3";
        assert_eq!(
            map(&desc, None, 0).expect("mp3").codec,
            StreamCodec::Known(CodecId::Mp3)
        );
        desc.format = *b"opus";
        desc.sample_rate = 24_000;
        let opus = map(&desc, None, 156).expect("opus");
        assert_eq!(opus.codec, StreamCodec::Known(CodecId::Opus));
        assert_eq!(crate::cookie::opus_pre_skip(&opus.extradata), Some(312));
        desc.channels = 3;
        assert!(map(&desc, None, 0).expect("opus").codec.is_unknown());
        desc.format = *b"ima4";
        let ima = map(&desc, Some(&[1, 2]), 0).expect("ima4");
        assert!(ima.codec.is_unknown());
        assert_eq!(ima.extradata, vec![1, 2]);
    }
}
