//! PSI section reassembly across packets (ITU-T H.222.0 clause 2.4.4).
//!
//! A packet that starts a section carries a `pointer_field`: the bytes
//! before it finish the section already open, and the bytes after it start
//! new sections back to back until a `0xFF` stuffing byte. A section is
//! handed on only when its whole `section_length` has arrived; the CRC is
//! the caller's check. Nothing above [`MAX_SECTION_BYTES`] is buffered.

use crate::limits::MAX_SECTION_BYTES;

/// Why a section was dropped while it was being assembled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SectionFault {
    /// `pointer_field` points past the payload.
    Pointer,
    /// `section_length` is above the cap.
    Length,
}

/// The section bytes of one PID.
#[derive(Debug, Default)]
pub struct SectionBuffer {
    buf: Vec<u8>,
    /// A section is open and later packets continue it.
    open: bool,
}

impl SectionBuffer {
    /// Forgets a partial section. A lost packet calls this.
    pub fn reset(&mut self) {
        self.buf.clear();
        self.open = false;
    }

    /// Feeds one packet payload. Each complete section goes to `sink`.
    ///
    /// The work is linear in the payload; at most one section is held.
    pub fn push(
        &mut self,
        payload: &[u8],
        start: bool,
        sink: &mut dyn FnMut(&[u8]),
    ) -> Result<(), SectionFault> {
        if !start {
            if self.open {
                self.buf.extend_from_slice(payload);
                return self.drain(sink);
            }
            return Ok(());
        }
        let Some((&pointer, rest)) = payload.split_first() else {
            self.reset();
            return Err(SectionFault::Pointer);
        };
        let pointer = usize::from(pointer);
        if pointer > rest.len() {
            self.reset();
            return Err(SectionFault::Pointer);
        }
        let (tail, fresh) = rest.split_at(pointer);
        let mut fault = Ok(());
        if self.open && !tail.is_empty() {
            self.buf.extend_from_slice(tail);
            fault = self.drain(sink);
        }
        // Whatever the old section still lacked is lost: a new one starts here.
        self.buf.clear();
        self.buf.extend_from_slice(fresh);
        self.open = true;
        let next = self.drain(sink);
        fault.and(next)
    }

    fn drain(&mut self, sink: &mut dyn FnMut(&[u8])) -> Result<(), SectionFault> {
        let mut at = 0;
        let result = loop {
            let rest = &self.buf[at..];
            if rest.first() == Some(&0xFF) {
                self.open = false;
                at = self.buf.len();
                break Ok(());
            }
            if rest.len() < 3 {
                break Ok(());
            }
            let length = ((usize::from(rest[1]) & 0x0F) << 8 | usize::from(rest[2])) + 3;
            if length > MAX_SECTION_BYTES {
                self.open = false;
                at = self.buf.len();
                break Err(SectionFault::Length);
            }
            if rest.len() < length {
                break Ok(());
            }
            sink(&rest[..length]);
            at += length;
        };
        self.buf.drain(..at);
        if self.buf.len() > MAX_SECTION_BYTES {
            self.reset();
        }
        result
    }
}

/// The fields of the long section header (clause 2.4.4.10 and 2.4.4.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LongHeader {
    pub table_id: u8,
    /// `transport_stream_id` in a PAT, `program_number` in a PMT.
    pub id: u16,
    pub version: u8,
    pub current: bool,
    pub number: u8,
    pub last: u8,
}

/// Why a section was not used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SectionError {
    /// Shorter than the long header plus the CRC.
    Short,
    /// The section syntax indicator is 0 where a long header is required.
    Syntax,
    /// `CRC_32` does not check.
    Crc,
}

/// Checks the CRC and reads the long header. The body is the bytes after
/// the header and before the CRC.
pub fn long_section(section: &[u8]) -> Result<(LongHeader, &[u8]), SectionError> {
    if section.len() < 12 {
        return Err(SectionError::Short);
    }
    if section[1] & 0x80 == 0 {
        return Err(SectionError::Syntax);
    }
    if crate::crc::crc32(section) != 0 {
        return Err(SectionError::Crc);
    }
    let header = LongHeader {
        table_id: section[0],
        id: u16::from_be_bytes([section[3], section[4]]),
        version: (section[5] >> 1) & 0x1F,
        current: section[5] & 1 != 0,
        number: section[6],
        last: section[7],
    };
    Ok((header, &section[8..section.len() - 4]))
}

#[cfg(test)]
mod tests {
    use super::{SectionBuffer, SectionError, SectionFault, long_section};
    use crate::crc::crc32;
    use crate::limits::MAX_SECTION_BYTES;

    fn section(table: u8, body: &[u8]) -> Vec<u8> {
        let length = body.len() + 5 + 4;
        let mut out = vec![
            table,
            0xB0 | (length >> 8) as u8,
            length as u8,
            0,
            1,
            0xC1,
            0,
            0,
        ];
        out.extend_from_slice(body);
        let crc = crc32(&out);
        out.extend_from_slice(&crc.to_be_bytes());
        out
    }

    fn collect(buffer: &mut SectionBuffer, payload: &[u8], start: bool) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        let _ = buffer.push(payload, start, &mut |s| out.push(s.to_vec()));
        out
    }

    #[test]
    fn a_section_across_three_packets_is_joined() {
        let whole = section(2, &[7; 400]);
        let mut buffer = SectionBuffer::default();
        let mut first = vec![0];
        first.extend_from_slice(&whole[..183]);
        assert!(collect(&mut buffer, &first, true).is_empty());
        assert!(collect(&mut buffer, &whole[183..367], false).is_empty());
        let mut last = whole[367..].to_vec();
        last.resize(184, 0xFF);
        assert_eq!(collect(&mut buffer, &last, false), vec![whole]);
    }

    #[test]
    fn the_pointer_field_finishes_one_section_and_starts_two() {
        let a = section(0, &[1, 2, 3, 4]);
        let b = section(2, &[5; 10]);
        let c = section(2, &[6; 3]);
        let mut buffer = SectionBuffer::default();
        let mut first = vec![0];
        first.extend_from_slice(&a[..5]);
        assert!(collect(&mut buffer, &first, true).is_empty());
        let mut second = vec![u8::try_from(a.len() - 5).unwrap()];
        second.extend_from_slice(&a[5..]);
        second.extend_from_slice(&b);
        second.extend_from_slice(&c);
        second.push(0xFF);
        assert_eq!(collect(&mut buffer, &second, true), vec![a, b, c]);
    }

    #[test]
    fn a_continuation_without_a_start_is_ignored() {
        let mut buffer = SectionBuffer::default();
        assert!(collect(&mut buffer, &[0; 184], false).is_empty());
    }

    #[test]
    fn a_hostile_pointer_or_length_is_dropped() {
        let mut buffer = SectionBuffer::default();
        assert_eq!(
            buffer.push(&[200, 1, 2], true, &mut |_| {}),
            Err(SectionFault::Pointer)
        );
        // A section_length of 0xFFF is 4098 bytes with its header: above the cap.
        let mut huge = vec![0, 0x02, 0xBF, 0xFF];
        huge.resize(184, 0);
        assert_eq!(
            buffer.push(&huge, true, &mut |_| {}),
            Err(SectionFault::Length)
        );
        let mut long = vec![0, 0x02, 0xBF, 0xF0];
        long.resize(184, 0);
        assert!(collect(&mut buffer, &long, true).is_empty());
        for _ in 0..20 {
            assert!(collect(&mut buffer, &[0; 184], false).is_empty());
        }
        assert!(buffer.buf.len() <= MAX_SECTION_BYTES);
    }

    #[test]
    fn the_long_header_checks_its_crc() {
        let good = section(2, &[0xE1, 0x00, 0xF0, 0x00]);
        let (header, body) = long_section(&good).expect("section");
        assert_eq!(header.table_id, 2);
        assert_eq!(header.id, 1);
        assert!(header.current);
        assert_eq!(body, &[0xE1, 0x00, 0xF0, 0x00]);
        let mut bad = good.clone();
        bad[9] ^= 1;
        assert_eq!(long_section(&bad), Err(SectionError::Crc));
        assert_eq!(long_section(&good[..8]), Err(SectionError::Short));
    }
}
