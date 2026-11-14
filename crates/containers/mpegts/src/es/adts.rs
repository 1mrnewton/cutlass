//! AAC in ADTS frames (ISO/IEC 13818-7 clause 6.2, ISO/IEC 14496-3
//! clause 1.A.2). The packet is the raw data block without the header and
//! its CRC; the stream's extradata is the two-byte AudioSpecificConfig the
//! header implies, the bytes an MP4 `esds` would hold.

use cutlass_codec::{CodecId, StreamCodec};

use super::Facts;
use super::audio::{Found, Frame, Syntax};

/// Sampling frequencies by `sampling_frequency_index` (14496-3 table 1.18).
pub const RATES: [u32; 13] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
    7_350,
];

/// One ADTS header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub profile: u8,
    pub rate_index: u8,
    pub channel_config: u8,
    pub frame_len: usize,
    pub blocks: u8,
    pub protection_absent: bool,
}

impl Header {
    /// Bytes before the first raw data block.
    #[must_use]
    pub const fn header_len(&self) -> usize {
        if self.protection_absent {
            7
        } else {
            9 + 2 * self.blocks as usize
        }
    }
}

/// Reads the fixed and variable header at the start of `bytes`.
#[must_use]
pub fn header(bytes: &[u8]) -> Option<Header> {
    let [b0, b1, b2, b3, b4, b5, _b6, ..] = *bytes else {
        return None;
    };
    if b0 != 0xFF || b1 & 0xF6 != 0xF0 {
        return None;
    }
    let rate_index = (b2 >> 2) & 0x0F;
    if usize::from(rate_index) >= RATES.len() {
        return None;
    }
    let header = Header {
        profile: b2 >> 6,
        rate_index,
        channel_config: ((b2 & 1) << 2) | (b3 >> 6),
        frame_len: (usize::from(b3 & 3) << 11) | (usize::from(b4) << 3) | usize::from(b5 >> 5),
        blocks: bytes[6] & 3,
        protection_absent: b1 & 1 == 1,
    };
    if header.frame_len < header.header_len() {
        return None;
    }
    Some(header)
}

/// The AudioSpecificConfig for an ADTS header: object type, rate index,
/// channel configuration.
#[must_use]
pub fn audio_specific_config(header: &Header) -> [u8; 2] {
    let object = header.profile + 1;
    [
        (object << 3) | (header.rate_index >> 1),
        ((header.rate_index & 1) << 7) | (header.channel_config << 3),
    ]
}

/// The channel count of `channel_configuration`, or `None` when the
/// layout is in a program config element (0) or reserved.
#[must_use]
pub const fn channels(config: u8) -> Option<u16> {
    match config {
        1..=6 => Some(config as u16),
        7 => Some(8),
        _ => None,
    }
}

#[derive(Debug, Default)]
pub struct Adts {
    facts: Option<Facts>,
}

impl Syntax for Adts {
    const HEADER: usize = 7;
    const MAX_FRAME: usize = 8191;
    const SYNC_BYTE: u8 = 0xFF;

    fn frame(&mut self, bytes: &[u8]) -> Found {
        if bytes.len() < Self::HEADER {
            return Found::NeedMore;
        }
        let Some(header) = header(bytes) else {
            return Found::NotSync;
        };
        let rate = RATES[usize::from(header.rate_index)];
        let blocks = u32::from(header.blocks) + 1;
        // A frame of several raw data blocks stays one packet: without a
        // CRC their positions are not in the header.
        let start = header.header_len();
        Found::Frame(Frame {
            len: header.frame_len,
            payload: (start, header.frame_len),
            samples: 1024 * blocks,
            rate,
            continues: false,
        })
    }

    fn learn(&mut self, frame: &[u8]) {
        let Some(header) = header(frame) else {
            return;
        };
        let rate = RATES[usize::from(header.rate_index)];
        self.facts = Some(Facts::Audio {
            codec: StreamCodec::Known(CodecId::Aac),
            rate,
            channels: channels(header.channel_config).unwrap_or(0),
            extradata: audio_specific_config(&header).to_vec(),
        });
    }

    fn facts(&self) -> Option<Facts> {
        self.facts.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::{Adts, audio_specific_config, header};
    use crate::es::audio::{AudioFramer, Found, Syntax};
    use crate::es::{Facts, Pes};

    /// An ADTS header for an AAC-LC frame of `body` bytes.
    pub fn adts(body: &[u8], rate_index: u8, channels: u8) -> Vec<u8> {
        let len = body.len() + 7;
        let mut out = vec![
            0xFF,
            0xF1,
            (1 << 6) | (rate_index << 2) | (channels >> 2),
            ((channels & 3) << 6) | ((len >> 11) as u8 & 3),
            (len >> 3) as u8,
            ((len & 7) << 5) as u8 | 0x1F,
            0xFC,
        ];
        out.extend_from_slice(body);
        out
    }

    #[test]
    fn the_header_gives_the_config_and_frame_length() {
        let frame = adts(&[1, 2, 3], 4, 2);
        let parsed = header(&frame).expect("header");
        assert_eq!(parsed.frame_len, 10);
        assert_eq!(parsed.profile, 1);
        // AAC-LC, 44.1 kHz, stereo: 0x12 0x10, as FFmpeg's encoder writes.
        assert_eq!(audio_specific_config(&parsed), [0x12, 0x10]);
        assert_eq!(header(&[0xFF, 0xF1, 0x3C, 0, 0, 0, 0]), None);
        let mut short = frame.clone();
        short[4] = 0;
        short[5] = 0x1F;
        assert_eq!(header(&short), None);
    }

    #[test]
    fn frames_split_across_pes_packets_are_joined_and_timed() {
        let mut framer = AudioFramer::new(Adts::default());
        let mut stream = Vec::new();
        for index in 0..3_u8 {
            stream.extend_from_slice(&adts(&[index; 20], 3, 2));
        }
        let mut out = Vec::new();
        framer
            .push(
                Pes {
                    payload: stream[..40].to_vec(),
                    pts: Some(9000),
                    dts: None,
                    position: 0,
                },
                &mut out,
            )
            .expect("push");
        framer
            .push(
                Pes {
                    payload: stream[40..].to_vec(),
                    pts: None,
                    dts: None,
                    position: 188,
                },
                &mut out,
            )
            .expect("push");
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].bytes, vec![0; 20]);
        assert_eq!(out[2].bytes, vec![2; 20]);
        let times: Vec<_> = out.iter().map(|unit| unit.pts).collect();
        assert_eq!(
            times,
            vec![Some(9000), Some(9000 + 1920), Some(9000 + 3840)]
        );
        assert_eq!(out[1].duration, Some(1920));
        assert!(matches!(
            framer.facts(),
            Some(Facts::Audio {
                rate: 48_000,
                channels: 2,
                ..
            })
        ));
    }

    #[test]
    fn garbage_before_a_frame_is_skipped() {
        let mut framer = AudioFramer::new(Adts::default());
        let mut payload = vec![0xFF, 0x00, 0x12, 0xFF];
        payload.extend_from_slice(&adts(&[5; 8], 4, 1));
        let mut out = Vec::new();
        framer
            .push(
                Pes {
                    payload,
                    pts: Some(0),
                    dts: None,
                    position: 0,
                },
                &mut out,
            )
            .expect("push");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].bytes, vec![5; 8]);
        assert_eq!(framer.skipped, 4);
        assert_eq!(Adts::default().frame(&[0xFF]), Found::NeedMore);
    }
}
