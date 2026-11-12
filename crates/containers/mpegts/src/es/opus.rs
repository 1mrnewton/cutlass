//! Opus access units behind the TS control header: an 11-bit `0x3FF`
//! prefix, trim and extension flags, the size in `0xFF`-continued bytes,
//! the optional trims and extension, then the Opus packet. This is the
//! layout FFmpeg (`libavcodec/opus/parser.c`, `mpegtsenc.c`) and
//! GStreamer read and write. The `OpusHead` is built from the
//! `channel_config_code` of the DVB extension descriptor, as FFmpeg does.

use cutlass_codec::{CodecId, StreamCodec};

use super::audio::Clock;
use super::{Facts, FrameFault, MAX_UNITS_PER_PES, Pes, Unit};

/// Samples at 48 kHz in one Opus packet, from its TOC byte and frame count
/// (RFC 6716 clause 3.1). `None` for an empty or malformed packet.
#[must_use]
pub fn packet_samples(packet: &[u8]) -> Option<u32> {
    let toc = *packet.first()?;
    let config = toc >> 3;
    let frame = match config {
        0..=11 => [480, 960, 1920, 2880][usize::from(config % 4)],
        12..=15 => [480, 960][usize::from(config % 2)],
        _ => [120, 240, 480, 960][usize::from(config % 4)],
    };
    let frames = match toc & 3 {
        0 => 1,
        1 | 2 => 2,
        _ => u32::from(*packet.get(1)? & 0x3F),
    };
    let samples = frame * frames;
    (frames > 0 && samples <= 5760).then_some(samples)
}

/// One access unit's header: where the packet starts and how long it is.
#[must_use]
pub fn control_header(bytes: &[u8]) -> Option<(usize, usize)> {
    let [b0, b1, ..] = *bytes else {
        return None;
    };
    if b0 != 0x7F || b1 & 0xE0 != 0xE0 {
        return None;
    }
    let start_trim = b1 & 0x10 != 0;
    let end_trim = b1 & 0x08 != 0;
    let extension = b1 & 0x04 != 0;
    let mut at = 2;
    let mut size = 0_usize;
    loop {
        let byte = *bytes.get(at)?;
        at += 1;
        size = size.checked_add(usize::from(byte))?;
        if byte != 0xFF {
            break;
        }
    }
    if start_trim {
        at += 2;
    }
    if end_trim {
        at += 2;
    }
    if extension {
        let length = usize::from(*bytes.get(at)?);
        at += 1 + length;
    }
    let end = at.checked_add(size)?;
    (end <= bytes.len()).then_some((at, size))
}

/// The `OpusHead` FFmpeg builds for `channel_config_code` (0 to 8): 0 is
/// dual mono, 1 and 2 are family 0, 3 to 8 are family 1 in Vorbis order.
#[must_use]
pub fn opus_head(code: u8) -> Option<(Vec<u8>, u16)> {
    const STREAMS: [u8; 9] = [1, 1, 1, 2, 2, 3, 4, 4, 5];
    const COUPLED: [u8; 9] = [1, 0, 1, 1, 2, 2, 2, 3, 3];
    const MAPS: [&[u8]; 8] = [
        &[0],
        &[0, 1],
        &[0, 2, 1],
        &[0, 1, 2, 3],
        &[0, 4, 1, 2, 3],
        &[0, 4, 1, 2, 3, 5],
        &[0, 4, 1, 2, 3, 5, 6],
        &[0, 6, 1, 2, 3, 4, 5, 7],
    ];
    if code > 8 {
        return None;
    }
    let channels = if code == 0 { 2 } else { code };
    let family = match code {
        0 => 255,
        1 | 2 => 0,
        _ => 1,
    };
    let mut head = b"OpusHead".to_vec();
    head.push(1);
    head.push(channels);
    head.extend_from_slice(&0_u16.to_le_bytes());
    head.extend_from_slice(&48_000_u32.to_le_bytes());
    head.extend_from_slice(&0_i16.to_le_bytes());
    head.push(family);
    if family != 0 {
        head.push(STREAMS[usize::from(code)]);
        head.push(COUPLED[usize::from(code)]);
        head.extend_from_slice(MAPS[usize::from(channels - 1)]);
    }
    Some((head, u16::from(channels)))
}

pub struct OpusFramer {
    clock: Clock,
    facts: Option<Facts>,
    /// Bytes that were not an access unit.
    pub skipped: u64,
}

impl OpusFramer {
    #[must_use]
    pub fn new(channel_config: Option<u8>) -> Self {
        let facts = channel_config
            .and_then(opus_head)
            .map(|(head, channels)| Facts::Audio {
                codec: StreamCodec::Known(CodecId::Opus),
                rate: 48_000,
                channels,
                extradata: head,
            });
        Self {
            clock: Clock::default(),
            facts,
            skipped: 0,
        }
    }

    pub fn facts(&self) -> Option<Facts> {
        self.facts.clone()
    }

    pub fn push(&mut self, pes: Pes, out: &mut Vec<Unit>) -> Result<(), FrameFault> {
        if let Some(pts) = pes.pts {
            self.clock.rebase(pts);
        }
        let data = pes.payload;
        let mut at = 0;
        let mut units = 0;
        while at < data.len() {
            let Some((start, size)) = control_header(&data[at..]) else {
                self.skipped = self.skipped.saturating_add((data.len() - at) as u64);
                break;
            };
            let packet = &data[at + start..at + start + size];
            let samples = packet_samples(packet).unwrap_or(0);
            let (pts, duration) = self.clock.advance(samples, 48_000);
            units += 1;
            if units > MAX_UNITS_PER_PES {
                return Err(FrameFault::TooManyUnits);
            }
            out.push(Unit {
                bytes: packet.to_vec(),
                pts,
                dts: pts,
                duration,
                keyframe: true,
                position: pes.position,
            });
            at += start + size;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{OpusFramer, control_header, opus_head, packet_samples};
    use crate::es::Pes;

    #[test]
    fn the_toc_gives_the_samples() {
        assert_eq!(packet_samples(&[0xFC]), Some(960));
        assert_eq!(packet_samples(&[0x70]), Some(480));
        assert_eq!(packet_samples(&[0xFD]), Some(1920));
        assert_eq!(packet_samples(&[0xFB, 0x03]), Some(960 * 3));
        assert_eq!(packet_samples(&[0xFB, 0x00]), None);
        assert_eq!(packet_samples(&[]), None);
    }

    #[test]
    fn control_headers_frame_access_units() {
        // Size 300 = 0xFF + 45, a start trim and an extension of one byte.
        let mut unit = vec![0x7F, 0xE0 | 0x10 | 0x04, 0xFF, 45, 0, 0, 1, 9];
        unit.extend(std::iter::repeat_n(0xFC_u8, 300));
        assert_eq!(control_header(&unit), Some((8, 300)));
        assert_eq!(control_header(&unit[..100]), None);
        assert_eq!(control_header(&[0x7F, 0x00, 1, 0]), None);
    }

    #[test]
    fn a_pes_of_two_units_gives_two_timed_packets() {
        let mut payload = vec![0x7F, 0xE0, 2, 0xFC, 0xAA];
        payload.extend_from_slice(&[0x7F, 0xE0, 1, 0xFC]);
        let mut framer = OpusFramer::new(Some(2));
        let mut out = Vec::new();
        framer
            .push(
                Pes {
                    payload,
                    pts: Some(3600),
                    dts: None,
                    position: 0,
                },
                &mut out,
            )
            .expect("push");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].bytes, vec![0xFC, 0xAA]);
        assert_eq!(out[1].pts, Some(3600 + 1800));
        assert_eq!(out[0].duration, Some(1800));
    }

    #[test]
    fn the_opus_head_follows_the_channel_code() {
        let (stereo, channels) = opus_head(2).expect("stereo");
        assert_eq!((stereo.len(), channels), (19, 2));
        let (surround, channels) = opus_head(6).expect("5.1");
        assert_eq!((surround.len(), channels), (27, 6));
        assert_eq!(&surround[21..], &[0, 4, 1, 2, 3, 5]);
        assert_eq!(opus_head(9), None);
    }
}
