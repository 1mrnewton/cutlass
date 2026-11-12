//! MPEG-1 and MPEG-2 audio frames, layers I to III (ISO/IEC 11172-3
//! clause 2.4.2.3, ISO/IEC 13818-3 clause 2.4.2.3, and the MPEG 2.5
//! extension). Layer III is [`CodecId::Mp3`]; layers I and II are an
//! unknown codec this workspace does not decode. Free-format frames have
//! no length in their header and are not framed.

use cutlass_codec::{CodecId, StreamCodec, UnknownCodec};

use super::Facts;
use super::audio::{Found, Frame, Syntax};

/// kbit/s by layer for MPEG-1 (rows: layer I, II, III).
const MPEG1: [[u16; 15]; 3] = [
    [
        0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448,
    ],
    [
        0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384,
    ],
    [
        0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
    ],
];

/// kbit/s for MPEG-2 and 2.5 (rows: layer I, layers II and III).
const MPEG2: [[u16; 15]; 2] = [
    [
        0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256,
    ],
    [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
];

const RATES: [u32; 3] = [44_100, 48_000, 32_000];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    /// 1, 2, or 3.
    pub layer: u8,
    pub len: usize,
    pub rate: u32,
    pub samples: u32,
    pub channels: u16,
}

/// Reads a four-byte frame header.
#[must_use]
pub fn header(bytes: &[u8]) -> Option<Header> {
    let [b0, b1, b2, b3, ..] = *bytes else {
        return None;
    };
    if b0 != 0xFF || b1 & 0xE0 != 0xE0 {
        return None;
    }
    // 0: MPEG 2.5, 2: MPEG-2, 3: MPEG-1. 1 is reserved.
    let version = (b1 >> 3) & 3;
    let layer = match (b1 >> 1) & 3 {
        3 => 1,
        2 => 2,
        1 => 3,
        _ => return None,
    };
    if version == 1 {
        return None;
    }
    let bit_index = usize::from(b2 >> 4);
    let rate_index = usize::from((b2 >> 2) & 3);
    if bit_index == 0 || bit_index == 15 || rate_index == 3 {
        return None;
    }
    let mpeg1 = version == 3;
    let kbps = u32::from(if mpeg1 {
        MPEG1[usize::from(layer - 1)][bit_index]
    } else {
        MPEG2[usize::from(layer > 1)][bit_index]
    });
    let rate = match version {
        3 => RATES[rate_index],
        2 => RATES[rate_index] / 2,
        _ => RATES[rate_index] / 4,
    };
    let padding = u32::from((b2 >> 1) & 1);
    let (len, samples) = match layer {
        1 => ((12 * kbps * 1000 / rate + padding) * 4, 384),
        2 => (144 * kbps * 1000 / rate + padding, 1152),
        _ if mpeg1 => (144 * kbps * 1000 / rate + padding, 1152),
        _ => (72 * kbps * 1000 / rate + padding, 576),
    };
    Some(Header {
        layer,
        len: usize::try_from(len).ok()?,
        rate,
        samples,
        channels: if b3 >> 6 == 3 { 1 } else { 2 },
    })
}

#[derive(Debug, Default)]
pub struct MpegAudio {
    facts: Option<Facts>,
}

impl Syntax for MpegAudio {
    const HEADER: usize = 4;
    const MAX_FRAME: usize = 2881;
    const SYNC_BYTE: u8 = 0xFF;

    fn frame(&mut self, bytes: &[u8]) -> Found {
        if bytes.len() < Self::HEADER {
            return Found::NeedMore;
        }
        match header(bytes) {
            Some(header) if header.len >= Self::HEADER => Found::Frame(Frame {
                len: header.len,
                payload: (0, header.len),
                samples: header.samples,
                rate: header.rate,
                continues: false,
            }),
            _ => Found::NotSync,
        }
    }

    fn learn(&mut self, frame: &[u8]) {
        let Some(header) = header(frame) else {
            return;
        };
        let codec = match header.layer {
            3 => StreamCodec::Known(CodecId::Mp3),
            2 => unknown(b"mpegts:mp2"),
            _ => unknown(b"mpegts:mp1"),
        };
        self.facts = Some(Facts::Audio {
            codec,
            rate: header.rate,
            channels: header.channels,
            extradata: Vec::new(),
        });
    }

    fn facts(&self) -> Option<Facts> {
        self.facts.clone()
    }
}

fn unknown(tag: &[u8]) -> StreamCodec {
    match UnknownCodec::new(tag.to_vec()) {
        Ok(codec) => StreamCodec::Unknown(codec),
        Err(_) => StreamCodec::Known(CodecId::Mp3),
    }
}

#[cfg(test)]
mod tests {
    use super::header;

    #[test]
    fn layer_three_frames_have_their_lengths() {
        // MPEG-1 layer III, 128 kbit/s, 44.1 kHz, no padding, stereo: 417.
        let parsed = header(&[0xFF, 0xFB, 0x90, 0x00]).expect("mp3");
        assert_eq!((parsed.layer, parsed.len, parsed.rate), (3, 417, 44_100));
        assert_eq!(parsed.samples, 1152);
        assert_eq!(parsed.channels, 2);
        // Padding adds a byte; mono is channel mode 3.
        let padded = header(&[0xFF, 0xFB, 0x92, 0xC0]).expect("padded");
        assert_eq!((padded.len, padded.channels), (418, 1));
        // MPEG-2 layer III, 64 kbit/s, 24 kHz: 72 * 64000 / 24000 = 192.
        let lsf = header(&[0xFF, 0xF3, 0x84, 0x00]).expect("lsf");
        assert_eq!((lsf.len, lsf.rate, lsf.samples), (192, 24_000, 576));
    }

    #[test]
    fn layer_two_and_reserved_values() {
        // MPEG-1 layer II, 192 kbit/s, 48 kHz: 144 * 192000 / 48000 = 576.
        let mp2 = header(&[0xFF, 0xFD, 0xA4, 0x00]).expect("mp2");
        assert_eq!((mp2.layer, mp2.len, mp2.rate), (2, 576, 48_000));
        assert_eq!(header(&[0xFF, 0xFB, 0xF0, 0]), None);
        assert_eq!(header(&[0xFF, 0xFB, 0x0C, 0]), None);
        assert_eq!(header(&[0xFF, 0xEB, 0x90, 0]), None);
        assert_eq!(header(&[0xFF, 0xF9, 0x90, 0]), None);
    }
}
