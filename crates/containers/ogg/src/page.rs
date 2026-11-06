//! One Ogg page: header fields, lacing, and body (RFC 3533 section 6).
//!
//! The page layer knows nothing about codecs. A page is split into packet
//! pieces by its lacing values: a run of 255s ends at the first value below
//! 255, and a run still open at the end of the table continues on the next
//! page of that logical stream.

use crate::crc::page_crc;
use crate::limits::PAGE_HEADER_BYTES;

/// Header type flag: the first packet piece continues one from an earlier page.
pub const FLAG_CONTINUED: u8 = 0x01;
/// Header type flag: first page of a logical stream.
pub const FLAG_BOS: u8 = 0x02;
/// Header type flag: last page of a logical stream.
pub const FLAG_EOS: u8 = 0x04;

/// The capture pattern every page starts with.
pub const CAPTURE: &[u8; 4] = b"OggS";

/// One page as read from the file. Its checksum has already been checked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Page {
    /// Byte offset of the capture pattern in the file.
    pub offset: u64,
    pub flags: u8,
    /// Granule position. -1 means no packet completes on this page.
    pub granule: i64,
    pub serial: u32,
    pub sequence: u32,
    pub lacing: Vec<u8>,
    pub body: Vec<u8>,
}

/// One packet piece on a page: a byte range of the body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Piece {
    pub start: usize,
    pub end: usize,
    /// The packet ends on this page. A piece that is not complete is the
    /// last on its page and continues on the next page of the stream.
    pub complete: bool,
}

impl Page {
    #[must_use]
    pub const fn continued(&self) -> bool {
        self.flags & FLAG_CONTINUED != 0
    }

    #[must_use]
    pub const fn bos(&self) -> bool {
        self.flags & FLAG_BOS != 0
    }

    #[must_use]
    pub const fn eos(&self) -> bool {
        self.flags & FLAG_EOS != 0
    }

    /// Bytes this page takes in the file.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        PAGE_HEADER_BYTES + self.lacing.len() + self.body.len()
    }

    /// The packet pieces of this page in order.
    #[must_use]
    pub fn pieces(&self) -> Vec<Piece> {
        let mut pieces = Vec::new();
        let mut start = 0_usize;
        let mut end = 0_usize;
        let mut open = false;
        for &value in &self.lacing {
            end += usize::from(value);
            open = true;
            if value < 255 {
                pieces.push(Piece {
                    start,
                    end,
                    complete: true,
                });
                start = end;
                open = false;
            }
        }
        if open {
            pieces.push(Piece {
                start,
                end,
                complete: false,
            });
        }
        pieces
    }

    /// Packets that end on this page.
    #[must_use]
    pub fn completed(&self) -> usize {
        self.lacing.iter().filter(|&&value| value < 255).count()
    }
}

/// The fixed header fields of a candidate page, before its body is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub version: u8,
    pub flags: u8,
    pub granule: i64,
    pub serial: u32,
    pub sequence: u32,
    pub checksum: u32,
    pub segments: u8,
}

/// Reads the 27 fixed bytes. `None` when the slice is short or the capture
/// pattern is missing.
#[must_use]
pub fn parse_header(bytes: &[u8]) -> Option<Header> {
    let fixed = bytes.get(..PAGE_HEADER_BYTES)?;
    if &fixed[..4] != CAPTURE {
        return None;
    }
    let mut granule = [0_u8; 8];
    granule.copy_from_slice(&fixed[6..14]);
    Some(Header {
        version: fixed[4],
        flags: fixed[5],
        granule: i64::from_le_bytes(granule),
        serial: u32::from_le_bytes([fixed[14], fixed[15], fixed[16], fixed[17]]),
        sequence: u32::from_le_bytes([fixed[18], fixed[19], fixed[20], fixed[21]]),
        checksum: u32::from_le_bytes([fixed[22], fixed[23], fixed[24], fixed[25]]),
        segments: fixed[26],
    })
}

/// Checks a whole candidate page held in `bytes`: header, lacing, body.
///
/// False when `bytes` is shorter than the page or the checksum differs.
#[must_use]
pub fn verify(bytes: &[u8], header: &Header) -> bool {
    let table_end = PAGE_HEADER_BYTES + usize::from(header.segments);
    let Some(table) = bytes.get(PAGE_HEADER_BYTES..table_end) else {
        return false;
    };
    let body_len: usize = table.iter().map(|&value| usize::from(value)).sum();
    let Some(body) = bytes.get(table_end..table_end + body_len) else {
        return false;
    };
    page_crc(&bytes[..table_end], body) == header.checksum
}

/// Lacing values for one packet of `len` bytes: `len / 255` values of 255,
/// then the remainder, which is 0 for an empty packet or a multiple of 255.
#[must_use]
pub fn lacing_for(len: usize) -> Vec<u8> {
    let mut values = vec![255_u8; len / 255];
    // The remainder is below 255.
    values.push(u8::try_from(len % 255).unwrap_or(0));
    values
}

/// Serializes one page with its checksum.
///
/// `lacing` has at most 255 values and `body` is their sum; the mux keeps
/// both true.
#[must_use]
pub fn encode(
    flags: u8,
    granule: i64,
    serial: u32,
    sequence: u32,
    lacing: &[u8],
    body: &[u8],
) -> Vec<u8> {
    let mut page = Vec::with_capacity(PAGE_HEADER_BYTES + lacing.len() + body.len());
    page.extend_from_slice(CAPTURE);
    page.push(0);
    page.push(flags);
    page.extend_from_slice(&granule.to_le_bytes());
    page.extend_from_slice(&serial.to_le_bytes());
    page.extend_from_slice(&sequence.to_le_bytes());
    page.extend_from_slice(&[0, 0, 0, 0]);
    page.push(u8::try_from(lacing.len()).unwrap_or(u8::MAX));
    page.extend_from_slice(lacing);
    let checksum = page_crc(&page, body);
    page[22..26].copy_from_slice(&checksum.to_le_bytes());
    page.extend_from_slice(body);
    page
}

#[cfg(test)]
mod tests {
    use super::{FLAG_BOS, FLAG_EOS, Page, Piece, encode, lacing_for, parse_header, verify};

    fn page(lacing: Vec<u8>) -> Page {
        let len = lacing.iter().map(|&value| usize::from(value)).sum();
        Page {
            offset: 0,
            flags: 0,
            granule: -1,
            serial: 1,
            sequence: 0,
            lacing,
            body: vec![0; len],
        }
    }

    #[test]
    fn a_packet_of_whole_255_blocks_ends_with_a_zero() {
        assert_eq!(lacing_for(0), vec![0]);
        assert_eq!(lacing_for(254), vec![254]);
        assert_eq!(lacing_for(255), vec![255, 0]);
        assert_eq!(lacing_for(510), vec![255, 255, 0]);
        assert_eq!(lacing_for(511), vec![255, 255, 1]);
    }

    #[test]
    fn lacing_splits_pieces_and_leaves_an_open_run_incomplete() {
        let pieces = page(vec![10, 255, 0, 0, 255, 255]).pieces();
        assert_eq!(
            pieces,
            vec![
                Piece {
                    start: 0,
                    end: 10,
                    complete: true
                },
                Piece {
                    start: 10,
                    end: 265,
                    complete: true
                },
                Piece {
                    start: 265,
                    end: 265,
                    complete: true
                },
                Piece {
                    start: 265,
                    end: 775,
                    complete: false
                },
            ]
        );
        assert_eq!(page(vec![10, 255, 0, 0, 255, 255]).completed(), 3);
        assert!(page(Vec::new()).pieces().is_empty());
    }

    #[test]
    fn an_encoded_page_parses_and_verifies() {
        let body = b"hello".to_vec();
        let bytes = encode(FLAG_BOS | FLAG_EOS, 960, 0xABCD, 3, &[5], &body);
        let header = parse_header(&bytes).expect("header");
        assert_eq!(header.version, 0);
        assert_eq!(header.flags, FLAG_BOS | FLAG_EOS);
        assert_eq!(header.granule, 960);
        assert_eq!(header.serial, 0xABCD);
        assert_eq!(header.sequence, 3);
        assert_eq!(header.segments, 1);
        assert!(verify(&bytes, &header));
        let mut broken = bytes.clone();
        let last = broken.len() - 1;
        broken[last] ^= 1;
        assert!(!verify(&broken, &header));
        assert!(!verify(&bytes[..bytes.len() - 1], &header));
        assert!(parse_header(b"OggS").is_none());
        assert!(parse_header(&[0; 27]).is_none());
    }

    #[test]
    fn a_minus_one_granule_round_trips() {
        let bytes = encode(0, -1, 1, 0, &[255], &[7; 255]);
        assert_eq!(parse_header(&bytes).expect("header").granule, -1);
    }
}
