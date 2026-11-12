//! PES packet assembly from transport packet payloads (ITU-T H.222.0
//! clause 2.4.3.6).
//!
//! The header is read as its bytes arrive, so a header split across
//! transport packets is joined. The payload is copied once, into a buffer
//! that becomes the access unit when one PES packet carries one. A packet
//! ends at the next `payload_unit_start_indicator` or, when
//! `PES_packet_length` is not 0, as soon as that many bytes arrived.
//! Nothing above the packet cap is buffered.

/// Stream ids whose PES packets have no optional header (Table 2-21).
const fn has_no_header(stream_id: u8) -> bool {
    matches!(
        stream_id,
        0xBC | 0xBE | 0xBF | 0xF0 | 0xF1 | 0xF2 | 0xF8 | 0xFF
    )
}

/// A PES packet with its timestamps as read, before unwrapping.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RawPes {
    pub stream_id: u8,
    pub payload: Vec<u8>,
    pub pts: Option<u64>,
    pub dts: Option<u64>,
    /// File offset of the transport packet that started it.
    pub position: u64,
    /// `data_alignment_indicator`.
    pub aligned: bool,
    /// It ended before `PES_packet_length` was reached.
    pub short: bool,
}

/// Why a PES packet was dropped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PesFault {
    /// The payload does not start with `00 00 01`.
    StartCode,
    /// `PES_header_data_length` runs past `PES_packet_length`.
    Header,
    /// The payload grew past the packet cap.
    Oversize { declared: u64, limit: u64 },
}

/// Reads a 33-bit timestamp from its five bytes; marker bits are not
/// checked, as decoders commonly do not.
#[must_use]
pub fn timestamp(bytes: &[u8]) -> Option<u64> {
    let [a, b, c, d, e] = *bytes.get(..5)? else {
        return None;
    };
    Some(
        (u64::from(a >> 1) & 7) << 30
            | u64::from(b) << 22
            | u64::from(c >> 1) << 15
            | u64::from(d) << 7
            | u64::from(e >> 1),
    )
}

/// Writes a timestamp's five bytes with `prefix` in the top four bits.
#[must_use]
pub const fn timestamp_bytes(prefix: u8, value: u64) -> [u8; 5] {
    [
        (prefix << 4) | (((value >> 29) & 0x0E) as u8) | 1,
        (value >> 22) as u8,
        (((value >> 14) & 0xFE) as u8) | 1,
        (value >> 7) as u8,
        (((value << 1) & 0xFE) as u8) | 1,
    ]
}

#[derive(Debug, Default)]
pub struct Assembler {
    head: Vec<u8>,
    /// Header bytes before the payload, once known.
    header_len: Option<usize>,
    payload: Vec<u8>,
    /// Payload bytes `PES_packet_length` declares, when not 0.
    expected: Option<usize>,
    pes: RawPes,
    open: bool,
    /// The last payload's length, to size the next buffer.
    hint: usize,
}

impl Assembler {
    /// Forgets the packet in progress, after a lost transport packet.
    pub fn reset(&mut self) {
        self.open = false;
        self.head.clear();
        self.header_len = None;
        self.payload = Vec::new();
        self.expected = None;
    }

    #[must_use]
    pub const fn open(&self) -> bool {
        self.open
    }

    /// Bytes held for the packet in progress.
    #[must_use]
    pub const fn buffered(&self) -> usize {
        self.payload.len()
    }

    /// Feeds one transport packet payload. A finished PES packet, or why
    /// one was dropped, goes to `sink`.
    pub fn push(
        &mut self,
        data: &[u8],
        start: bool,
        position: u64,
        limit: usize,
        sink: &mut dyn FnMut(Result<RawPes, PesFault>),
    ) {
        if start {
            if self.open {
                self.finish(true, sink);
            }
            self.open = true;
            self.head.clear();
            self.header_len = None;
            self.expected = None;
            self.payload = Vec::new();
            self.pes = RawPes {
                position,
                ..RawPes::default()
            };
        } else if !self.open {
            return;
        }
        let mut data = data;
        if self.header_len.is_none() {
            let taken = self.take_header(data, limit);
            match taken {
                Err(fault) => {
                    self.reset();
                    sink(Err(fault));
                    return;
                }
                Ok(used) => data = &data[used..],
            }
            if self.header_len.is_none() {
                return;
            }
        }
        let room = self.expected.map_or(data.len(), |expected| {
            expected.saturating_sub(self.payload.len()).min(data.len())
        });
        if self.payload.len() + room > limit {
            let declared = (self.payload.len() + room) as u64;
            self.reset();
            sink(Err(PesFault::Oversize {
                declared,
                limit: limit as u64,
            }));
            return;
        }
        self.payload.extend_from_slice(&data[..room]);
        if self
            .expected
            .is_some_and(|expected| self.payload.len() >= expected)
        {
            self.finish(false, sink);
        }
    }

    /// The stream ended: the packet in progress is finished as it is.
    pub fn flush(&mut self, sink: &mut dyn FnMut(Result<RawPes, PesFault>)) {
        if self.open {
            self.finish(true, sink);
        }
    }

    /// Takes header bytes from `data`; returns how many were used.
    fn take_header(&mut self, data: &[u8], limit: usize) -> Result<usize, PesFault> {
        let mut used = 0;
        loop {
            let need = match self.head.len() {
                0..6 => 6,
                _ if has_no_header(self.head[3]) => 6,
                6..9 => 9,
                _ => 9 + usize::from(self.head[8]),
            };
            if self.head.len() >= need {
                break;
            }
            let take = (need - self.head.len()).min(data.len() - used);
            if take == 0 {
                return Ok(used);
            }
            self.head.extend_from_slice(&data[used..used + take]);
            used += take;
            if self.head.len() >= 3 && self.head[..3] != [0, 0, 1] {
                return Err(PesFault::StartCode);
            }
        }
        let header_len = self.head.len();
        let stream_id = self.head[3];
        let length = usize::from(u16::from_be_bytes([self.head[4], self.head[5]]));
        if length != 0 {
            let Some(expected) = (length + 6).checked_sub(header_len) else {
                return Err(PesFault::Header);
            };
            self.expected = Some(expected);
            self.payload.reserve_exact(expected.min(limit));
        } else {
            self.payload.reserve(self.hint.min(limit));
        }
        self.pes.stream_id = stream_id;
        if !has_no_header(stream_id) {
            let flags = self.head[7] >> 6;
            self.pes.aligned = self.head[6] & 0x04 != 0;
            let fields = &self.head[9..];
            if flags & 2 != 0 {
                self.pes.pts = timestamp(fields);
                if flags == 3 {
                    self.pes.dts = fields.get(5..).and_then(timestamp);
                }
            }
        }
        self.header_len = Some(header_len);
        Ok(used)
    }

    fn finish(&mut self, interrupted: bool, sink: &mut dyn FnMut(Result<RawPes, PesFault>)) {
        self.open = false;
        if self.header_len.is_none() {
            self.head.clear();
            return;
        }
        let mut pes = std::mem::take(&mut self.pes);
        pes.payload = std::mem::take(&mut self.payload);
        pes.short = interrupted
            && self
                .expected
                .is_some_and(|expected| pes.payload.len() < expected);
        self.hint = pes.payload.len();
        self.header_len = None;
        self.expected = None;
        self.head.clear();
        sink(Ok(pes));
    }
}

#[cfg(test)]
mod tests {
    use super::{Assembler, PesFault, RawPes, timestamp, timestamp_bytes};

    fn header(stream_id: u8, length: u16, pts: Option<u64>, dts: Option<u64>) -> Vec<u8> {
        let mut out = vec![0, 0, 1, stream_id];
        out.extend_from_slice(&length.to_be_bytes());
        let flags = match (pts, dts) {
            (Some(_), Some(_)) => 0xC0,
            (Some(_), None) => 0x80,
            _ => 0,
        };
        let mut fields = Vec::new();
        if let Some(pts) = pts {
            fields.extend_from_slice(&timestamp_bytes(if dts.is_some() { 3 } else { 2 }, pts));
        }
        if let Some(dts) = dts {
            fields.extend_from_slice(&timestamp_bytes(1, dts));
        }
        out.extend_from_slice(&[0x84, flags, fields.len() as u8]);
        out.extend_from_slice(&fields);
        out
    }

    fn collect(
        assembler: &mut Assembler,
        data: &[u8],
        start: bool,
    ) -> Vec<Result<RawPes, PesFault>> {
        let mut out = Vec::new();
        assembler.push(data, start, 0, 1 << 20, &mut |pes| out.push(pes));
        out
    }

    #[test]
    fn timestamps_read_back_through_all_33_bits() {
        for value in [0, 1, 90_000, (1 << 32) + 12_345, (1 << 33) - 1] {
            assert_eq!(timestamp(&timestamp_bytes(2, value)), Some(value));
        }
        // FFmpeg's first PTS with its default 1.4 s delay, as it writes it.
        assert_eq!(timestamp(&[0x21, 0x00, 0x07, 0xD8, 0x61]), Some(126_000));
    }

    #[test]
    fn a_header_split_across_packets_is_joined() {
        let mut bytes = header(0xE0, 0, Some(1000), Some(900));
        bytes.extend_from_slice(&[9, 8, 7]);
        let mut assembler = Assembler::default();
        assert!(collect(&mut assembler, &bytes[..5], true).is_empty());
        assert!(collect(&mut assembler, &bytes[5..12], false).is_empty());
        assert!(collect(&mut assembler, &bytes[12..], false).is_empty());
        let done = collect(&mut assembler, &header(0xE0, 0, None, None), true);
        let pes = done[0].clone().unwrap();
        assert_eq!(pes.pts, Some(1000));
        assert_eq!(pes.dts, Some(900));
        assert_eq!(pes.payload, vec![9, 8, 7]);
        assert!(pes.aligned);
    }

    #[test]
    fn a_declared_length_ends_the_packet_early_and_drops_the_rest() {
        let mut bytes = header(0xC0, 3 + 5 + 4, Some(5), None);
        bytes.extend_from_slice(&[1, 2, 3, 4, 0xFF, 0xFF]);
        let mut assembler = Assembler::default();
        let done = collect(&mut assembler, &bytes, true);
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].clone().unwrap().payload, vec![1, 2, 3, 4]);
        assert!(!assembler.open());
        // A continuation of a finished packet is ignored.
        assert!(collect(&mut assembler, &[5; 10], false).is_empty());
    }

    #[test]
    fn hostile_headers_and_endless_payloads_are_dropped() {
        let mut assembler = Assembler::default();
        assert_eq!(
            collect(&mut assembler, &[0, 0, 2, 0xE0, 0, 0], true),
            vec![Err(PesFault::StartCode)]
        );
        // PES_packet_length 2 cannot hold a 3-byte header.
        assert_eq!(
            collect(&mut assembler, &[0, 0, 1, 0xE0, 0, 2, 0x80, 0, 0], true),
            vec![Err(PesFault::Header)]
        );
        let mut out = Vec::new();
        assembler.push(&header(0xE0, 0, None, None), true, 0, 1000, &mut |pes| {
            out.push(pes)
        });
        for _ in 0..10 {
            assembler.push(&[0; 184], false, 0, 1000, &mut |pes| out.push(pes));
        }
        assert!(matches!(out[..], [Err(PesFault::Oversize { .. })]));
        assert_eq!(assembler.buffered(), 0);
    }

    #[test]
    fn padding_has_no_optional_header() {
        let mut assembler = Assembler::default();
        let done = collect(
            &mut assembler,
            &[0, 0, 1, 0xBE, 0, 3, 0xFF, 0xFF, 0xFF],
            true,
        );
        let pes = done[0].clone().unwrap();
        assert_eq!(pes.stream_id, 0xBE);
        assert_eq!(pes.payload, vec![0xFF; 3]);
    }
}
