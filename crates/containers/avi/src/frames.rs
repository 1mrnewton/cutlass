//! Frame lengths of the audio codecs whose chunks may hold several frames.
//!
//! An AVI audio chunk is whatever the writer flushed: a constant-bitrate
//! MP3 or AC-3 stream is often stored in chunks of a fixed byte count, so a
//! chunk can hold several frames and end inside one. The demuxer returns
//! one frame per packet, as FFmpeg's demuxer does through its parsers, and
//! only needs each frame's length and sample count for that.
//!
//! MPEG audio headers are ISO/IEC 11172-3 and 13818-3 (MPEG-2.5 included);
//! AC-3 and E-AC-3 sync frames are ATSC A/52 (Annex E for E-AC-3). Free
//! format MPEG audio has no length in its header and is not split.

use cutlass_codec::{CodecId, StreamCodec};

/// Bytes a frame header needs before its length is known.
pub(crate) const HEADER_BYTES: usize = 6;

/// Longest frame either syntax allows: an E-AC-3 frame is at most 2048
/// words; an MPEG-1 Layer II frame at 384 kbit/s and 32 kHz is 1728 bytes.
pub(crate) const MAX_FRAME_BYTES: usize = 4_096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Framing {
    /// MPEG-1, MPEG-2 and MPEG-2.5 audio, layers I to III.
    Mpeg,
    /// AC-3 and E-AC-3 sync frames.
    Ac3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Frame {
    pub len: usize,
    pub samples: u32,
    pub sample_rate: u32,
}

impl Framing {
    /// The framing of an audio stream's codec, if its chunks are split.
    /// `wav:0050` is MPEG-1 Layer I or II, which has no codec id here.
    pub(crate) fn of(codec: &StreamCodec) -> Option<Self> {
        match codec {
            StreamCodec::Known(CodecId::Mp3) => Some(Self::Mpeg),
            StreamCodec::Known(CodecId::Ac3 | CodecId::Eac3) => Some(Self::Ac3),
            StreamCodec::Unknown(unknown) if unknown.tag() == b"wav:0050" => Some(Self::Mpeg),
            _ => None,
        }
    }

    /// The frame that starts at `bytes[0]`, if a valid header is there.
    /// `bytes` may be shorter than the frame.
    pub(crate) fn frame(self, bytes: &[u8]) -> Option<Frame> {
        match self {
            Self::Mpeg => mpeg(bytes),
            Self::Ac3 => ac3(bytes),
        }
    }
}

const MPEG_KBPS: [[[u16; 15]; 3]; 2] = [
    [
        [
            0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448,
        ],
        [
            0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384,
        ],
        [
            0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
        ],
    ],
    [
        [
            0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256,
        ],
        [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
        [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
    ],
];

fn mpeg(bytes: &[u8]) -> Option<Frame> {
    let head: [u8; 4] = bytes.get(..4)?.try_into().ok()?;
    let bits = u32::from_be_bytes(head);
    if bits & 0xFFE0_0000 != 0xFFE0_0000 {
        return None;
    }
    let version = (bits >> 19) & 3;
    let layer = 4 - ((bits >> 17) & 3);
    let bitrate_index = usize::try_from((bits >> 12) & 0xF).ok()?;
    let rate_index = usize::try_from((bits >> 10) & 3).ok()?;
    let padding = (bits >> 9) & 1;
    if version == 1 || layer == 4 || bitrate_index == 0 || bitrate_index == 15 || rate_index == 3 {
        return None;
    }
    let base = [44_100, 48_000, 32_000][rate_index];
    let sample_rate = match version {
        3 => base,
        2 => base / 2,
        _ => base / 4,
    };
    let lsf = usize::from(version != 3);
    let layer_row = usize::try_from(layer - 1).ok()?;
    let bitrate = u32::from(MPEG_KBPS[lsf][layer_row][bitrate_index]) * 1_000;
    let (len, samples) = match layer {
        1 => ((12 * bitrate / sample_rate + padding) * 4, 384),
        2 => (144 * bitrate / sample_rate + padding, 1_152),
        _ if version == 3 => (144 * bitrate / sample_rate + padding, 1_152),
        _ => (72 * bitrate / sample_rate + padding, 576),
    };
    let len = usize::try_from(len).ok()?;
    (len >= 4).then_some(Frame {
        len,
        samples,
        sample_rate,
    })
}

/// Nominal AC-3 bit rates in kbit/s, by `frmsizecod / 2` (A/52 Table 5.18).
const AC3_KBPS: [u32; 19] = [
    32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 576, 640,
];

fn ac3(bytes: &[u8]) -> Option<Frame> {
    let head = bytes.get(..HEADER_BYTES)?;
    if head[0] != 0x0B || head[1] != 0x77 {
        return None;
    }
    let fscod = head[4] >> 6;
    if fscod == 3 {
        return None;
    }
    let sample_rate = [48_000, 44_100, 32_000][usize::from(fscod)];
    match head[5] >> 3 {
        0..=8 => {
            let code = head[4] & 0x3F;
            let kbps = *AC3_KBPS.get(usize::from(code / 2))?;
            // Sixteen-bit words in one frame (A/52 Table 5.18): exact at 48
            // and 32 kHz; at 44.1 kHz rounded down, and one more word for
            // an odd frmsizecod.
            let words = match fscod {
                0 => kbps * 2,
                1 => kbps * 320 / 147 + u32::from(code & 1),
                _ => kbps * 3,
            };
            Some(Frame {
                len: usize::try_from(words * 2).ok()?,
                samples: 1_536,
                sample_rate,
            })
        }
        11..=16 => {
            let words = (usize::from(head[2] & 7) << 8 | usize::from(head[3])) + 1;
            let blocks = [1, 2, 3, 6][usize::from((head[4] >> 4) & 3)];
            Some(Frame {
                len: words * 2,
                samples: blocks * 256,
                sample_rate,
            })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{Framing, ac3, mpeg};

    #[test]
    fn mpeg_lengths_follow_the_iso_formula() {
        // MPEG-1 Layer III, 128 kbit/s, 44.1 kHz: 417 bytes, 418 padded.
        let frame = mpeg(&[0xFF, 0xFB, 0x90, 0x00]).expect("frame");
        assert_eq!(
            (frame.len, frame.samples, frame.sample_rate),
            (417, 1_152, 44_100)
        );
        assert_eq!(mpeg(&[0xFF, 0xFB, 0x92, 0x00]).expect("padded").len, 418);
        // MPEG-2 Layer III, 64 kbit/s, 24 kHz: 72 * 64000 / 24000 = 192.
        let low = mpeg(&[0xFF, 0xF3, 0x84, 0x00]).expect("lsf");
        assert_eq!((low.len, low.samples, low.sample_rate), (192, 576, 24_000));
        // MPEG-1 Layer II, 192 kbit/s, 48 kHz: 576 bytes.
        assert_eq!(mpeg(&[0xFF, 0xFD, 0xA4, 0x00]).expect("layer 2").len, 576);
        // Free format, a reserved rate, and no sync.
        assert_eq!(mpeg(&[0xFF, 0xFB, 0x00, 0x00]), None);
        assert_eq!(mpeg(&[0xFF, 0xFB, 0x9C, 0x00]), None);
        assert_eq!(mpeg(&[0x00, 0xFB, 0x90, 0x00]), None);
    }

    #[test]
    fn ac3_lengths_match_table_5_18() {
        // frmsizecod 30 (448 kbit/s): 896, 975 and 1344 words.
        for (fscod, words) in [(0u8, 896usize), (1, 975), (2, 1_344)] {
            let head = [0x0B, 0x77, 0, 0, fscod << 6 | 30, 8 << 3];
            assert_eq!(ac3(&head).expect("ac3").len, words * 2);
        }
        // An odd code at 44.1 kHz carries one more word: 32 kbit/s, 70 words.
        assert_eq!(
            ac3(&[0x0B, 0x77, 0, 0, 1 << 6 | 1, 8 << 3])
                .expect("odd")
                .len,
            140
        );
        // The last code at 44.1 kHz: 640 kbit/s, 1394 words.
        assert_eq!(
            ac3(&[0x0B, 0x77, 0, 0, 1 << 6 | 37, 8 << 3])
                .expect("last")
                .len,
            2_788
        );
        // E-AC-3: frmsiz + 1 words, numblkscod 3 is six blocks.
        let eac3 = ac3(&[0x0B, 0x77, 0x00, 0xFF, 0x30, 16 << 3]).expect("eac3");
        assert_eq!((eac3.len, eac3.samples), (512, 1_536));
        assert_eq!(ac3(&[0x0B, 0x77, 0, 0, 3 << 6, 8 << 3]), None);
        assert_eq!(ac3(&[0x0B, 0x77, 0, 0, 0, 9 << 3]), None);
        assert_eq!(Framing::Ac3.frame(&[0x0B, 0x77, 0, 0]), None);
    }
}
