//! AAC in LOAS frames carrying LATM (ISO/IEC 14496-3 clause 1.7), the
//! framing of `stream_type` 0x11.
//!
//! This crate finds each `AudioSyncStream` frame by its 11-bit sync word
//! and 13-bit length. The `AudioMuxElement` inside is read by
//! `cutlass-adts`, which owns LATM: it gives the payload and the
//! AudioSpecificConfig from `StreamMuxConfig`.

use cutlass_adts::{Demuxer, LatmDemuxer};
use cutlass_codec::{CodecId, StreamCodec};

use super::Facts;
use super::audio::{Found, Frame, Syntax};

#[derive(Default)]
pub struct Loas {
    reader: Option<LatmDemuxer>,
    payload: Option<Vec<u8>>,
    facts: Option<Facts>,
}

impl std::fmt::Debug for Loas {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Loas")
            .field("facts", &self.facts)
            .finish_non_exhaustive()
    }
}

/// The length of the LOAS frame at the start of `bytes`, sync word and
/// length field included.
#[must_use]
pub fn frame_len(bytes: &[u8]) -> Option<usize> {
    let [0x56, b1, b2, ..] = *bytes else {
        return None;
    };
    if b1 & 0xE0 != 0xE0 {
        return None;
    }
    let length = (usize::from(b1 & 0x1F) << 8) | usize::from(b2);
    (length > 0).then_some(3 + length)
}

impl Loas {
    /// Runs one whole LOAS frame through the LATM reader.
    fn read(&mut self, frame: &[u8]) -> Option<(u32, u32)> {
        let reader = self.reader.get_or_insert_with(LatmDemuxer::new);
        reader.push_bytes(frame).ok()?;
        let packet = reader.next_packet().ok()??;
        let samples = u32::try_from(packet.duration).ok()?;
        let rate = packet.timescale;
        if self.facts.is_none()
            && let Some(stream) = reader.streams().first()
        {
            let channels = stream
                .params
                .layout
                .as_ref()
                .and_then(|layout| u16::try_from(layout.channels().len()).ok())
                .unwrap_or(0);
            self.facts = Some(Facts::Audio {
                codec: StreamCodec::Known(CodecId::Aac),
                rate: stream.params.sample_rate,
                channels,
                extradata: stream.params.extra.clone(),
            });
        }
        self.payload = Some(packet.data);
        Some((samples, rate))
    }
}

impl Syntax for Loas {
    const HEADER: usize = 3;
    const MAX_FRAME: usize = 3 + 0x1FFF;
    const SYNC_BYTE: u8 = 0x56;

    fn frame(&mut self, bytes: &[u8]) -> Found {
        if bytes.len() < Self::HEADER {
            return Found::NeedMore;
        }
        let Some(len) = frame_len(bytes) else {
            return Found::NotSync;
        };
        let Some(frame) = bytes.get(..len) else {
            // Not all here yet: the framer keeps the bytes and asks again.
            return Found::Frame(Frame {
                len,
                payload: (0, len),
                samples: 0,
                rate: 0,
                continues: false,
            });
        };
        self.payload = None;
        let (samples, rate) = self.read(frame).unwrap_or((0, 0));
        Found::Frame(Frame {
            len,
            payload: (0, len),
            samples,
            rate,
            continues: false,
        })
    }

    fn learn(&mut self, _frame: &[u8]) {}

    fn facts(&self) -> Option<Facts> {
        self.facts.clone()
    }

    fn payload(&mut self, _frame: &[u8], _payload: (usize, usize)) -> Option<Vec<u8>> {
        self.payload.take()
    }
}

#[cfg(test)]
mod tests {
    use super::frame_len;

    #[test]
    fn the_sync_word_and_length_frame_a_loas_packet() {
        assert_eq!(frame_len(&[0x56, 0xE0, 0x10]), Some(19));
        assert_eq!(frame_len(&[0x56, 0xFF, 0xFF]), Some(3 + 0x1FFF));
        assert_eq!(frame_len(&[0x56, 0xC0, 0x10]), None);
        assert_eq!(frame_len(&[0x56, 0xE0, 0x00]), None);
        assert_eq!(frame_len(&[0x57, 0xE0, 0x10]), None);
    }
}
