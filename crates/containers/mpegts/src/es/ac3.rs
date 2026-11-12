//! AC-3 and E-AC-3 sync frames (ATSC A/52 clause 5.4 and annex E).
//!
//! One packet is one AC-3 frame, or an independent E-AC-3 frame of
//! substream 0 with the dependent and other substream frames that follow
//! it, the way FFmpeg's parser joins them.

use cutlass_codec::{CodecId, StreamCodec};

use super::Facts;
use super::audio::{Found, Frame, Syntax};
use super::bits::Bits;

/// Nominal bit rates in kbit/s by `frmsizecod / 2` (A/52 table 5.18).
const BIT_RATES: [u32; 19] = [
    32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 576, 640,
];

/// Full-bandwidth channels by `acmod` (A/52 table 5.8).
const CHANNELS: [u16; 8] = [2, 1, 2, 3, 3, 4, 4, 5];

/// One AC-3 or E-AC-3 sync frame header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub eac3: bool,
    pub len: usize,
    pub rate: u32,
    pub samples: u32,
    pub channels: u16,
    /// E-AC-3 `strmtyp` 0 and `substreamid` 0, or any AC-3 frame.
    pub starts_unit: bool,
}

/// Reads the header at the start of `bytes`, which must hold eight bytes.
#[must_use]
pub fn header(bytes: &[u8]) -> Option<Header> {
    if bytes.len() < 8 || bytes[0] != 0x0B || bytes[1] != 0x77 {
        return None;
    }
    let bsid = bytes[5] >> 3;
    if bsid <= 10 {
        ac3(bytes)
    } else if (11..=16).contains(&bsid) {
        eac3(bytes)
    } else {
        None
    }
}

fn ac3(bytes: &[u8]) -> Option<Header> {
    let fscod = bytes[4] >> 6;
    let frmsizecod = bytes[4] & 0x3F;
    let rate = match fscod {
        0 => 48_000,
        1 => 44_100,
        2 => 32_000,
        _ => return None,
    };
    let bit_rate = *BIT_RATES.get(usize::from(frmsizecod / 2))?;
    let words = match fscod {
        0 => bit_rate * 2,
        1 => bit_rate * 320 / 147 + u32::from(frmsizecod & 1),
        _ => bit_rate * 3,
    };
    let mut bits = Bits::new(&bytes[6..]);
    let acmod = bits.bits(3)?;
    if acmod & 1 != 0 && acmod != 1 {
        bits.skip(2)?;
    }
    if acmod & 4 != 0 {
        bits.skip(2)?;
    }
    if acmod == 2 {
        bits.skip(2)?;
    }
    let lfe = u16::from(bits.bit()?);
    Some(Header {
        eac3: false,
        len: usize::try_from(words).ok()? * 2,
        rate,
        samples: 1536,
        channels: CHANNELS[acmod as usize] + lfe,
        starts_unit: true,
    })
}

fn eac3(bytes: &[u8]) -> Option<Header> {
    let mut bits = Bits::new(&bytes[2..]);
    let stream_type = bits.bits(2)?;
    let substream = bits.bits(3)?;
    let frame_size = bits.bits(11)?;
    let fscod = bits.bits(2)?;
    let (rate, blocks) = if fscod == 3 {
        let rate = match bits.bits(2)? {
            0 => 24_000,
            1 => 22_050,
            2 => 16_000,
            _ => return None,
        };
        (rate, 6)
    } else {
        let rate = [48_000, 44_100, 32_000][fscod as usize];
        (rate, [1, 2, 3, 6][bits.bits(2)? as usize])
    };
    let acmod = bits.bits(3)?;
    let lfe = u16::from(bits.bit()?);
    if stream_type == 3 {
        return None;
    }
    Some(Header {
        eac3: true,
        len: (usize::try_from(frame_size).ok()? + 1) * 2,
        rate,
        samples: 256 * blocks,
        channels: CHANNELS[acmod as usize] + lfe,
        starts_unit: stream_type != 1 && substream == 0,
    })
}

#[derive(Debug, Default)]
pub struct Ac3 {
    facts: Option<Facts>,
}

impl Syntax for Ac3 {
    const HEADER: usize = 8;
    const MAX_FRAME: usize = 4096;
    const SYNC_BYTE: u8 = 0x0B;

    fn frame(&mut self, bytes: &[u8]) -> Found {
        if bytes.len() < Self::HEADER {
            return Found::NeedMore;
        }
        let Some(header) = header(bytes) else {
            return Found::NotSync;
        };
        if header.len < Self::HEADER {
            return Found::NotSync;
        }
        Found::Frame(Frame {
            len: header.len,
            payload: (0, header.len),
            // A frame that joins the unit before adds no time of its own.
            samples: if header.starts_unit {
                header.samples
            } else {
                0
            },
            rate: header.rate,
            continues: !header.starts_unit,
        })
    }

    fn learn(&mut self, frame: &[u8]) {
        if let Some(header) = header(frame) {
            self.facts = Some(Facts::Audio {
                codec: StreamCodec::Known(if header.eac3 {
                    CodecId::Eac3
                } else {
                    CodecId::Ac3
                }),
                rate: header.rate,
                channels: header.channels,
                extradata: Vec::new(),
            });
        }
    }

    fn facts(&self) -> Option<Facts> {
        self.facts.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::header;

    #[test]
    fn an_ac3_header_gives_size_rate_and_channels() {
        // 48 kHz, 448 kbit/s (frmsizecod 30), bsid 8; acmod 7 (3/2) with
        // cmixlev and surmixlev, then lfeon set.
        let bytes = [0x0B, 0x77, 0, 0, 0x1E, 0x40, 0b1110_0001, 0];
        let parsed = header(&bytes).expect("ac3");
        assert_eq!(parsed.len, 1792);
        assert_eq!(parsed.rate, 48_000);
        assert_eq!(parsed.channels, 6);
        assert_eq!(parsed.samples, 1536);
        assert!(!parsed.eac3);
    }

    #[test]
    fn a_44100_odd_code_adds_a_word() {
        let mut bytes = [0x0B, 0x77, 0, 0, 0x40 | 31, 0x40, 0x40, 0];
        let even = {
            let mut copy = bytes;
            copy[4] = 0x40 | 30;
            header(&copy).expect("even").len
        };
        assert_eq!(even, 975 * 2);
        assert_eq!(header(&bytes).expect("odd").len, 976 * 2);
        bytes[4] = 0xC0;
        assert_eq!(header(&bytes), None);
    }

    #[test]
    fn an_eac3_header_names_dependent_frames() {
        // strmtyp 0, substream 0, frmsiz 383 (768 bytes); fscod 0 and
        // numblkscod 3 (six blocks), acmod 2, no LFE; bsid 16.
        let bytes = [0x0B, 0x77, 0x01, 0x7F, 0b0011_0100, 0x80, 0, 0];
        let parsed = header(&bytes).expect("eac3");
        assert!(parsed.eac3);
        assert_eq!(parsed.len, 768);
        assert_eq!(parsed.samples, 1536);
        assert_eq!(parsed.channels, 2);
        assert!(parsed.starts_unit);
        let mut dependent = bytes;
        dependent[2] |= 0x40;
        assert!(!header(&dependent).expect("dependent").starts_unit);
    }
}
