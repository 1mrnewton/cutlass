//! A buffered byte source that reads whole pages and resyncs past garbage.
//!
//! The demuxer owns one [`Input`]. This layer keeps a small read-ahead
//! buffer so a page header, its lacing table, and its body cost one read in
//! the common case, and so a capture-pattern search does not read byte by
//! byte from the reader.
//!
//! A candidate page is accepted only when its version is 0 and its checksum
//! matches. A checksum failure is not an error here: the candidate is
//! counted on [`Source::crc_failures`] and the search continues one byte
//! later, the same recovery FFmpeg and libogg use. The search gives up after
//! [`MAX_RESYNC_BYTES`] bytes without a valid page.

use std::io::{Read, Seek};

use cutlass_container::{Input, SourceError};

use crate::error::{Fault, OggError};
use crate::limits::{MAX_RESYNC_BYTES, PAGE_HEADER_BYTES};
use crate::page::{CAPTURE, Page, parse_header, verify};

/// Bytes asked of the reader per refill.
const CHUNK: usize = 16 * 1024;

/// What the next page read found.
#[derive(Debug)]
pub enum PageRead {
    Page(Page),
    /// The input ended. Bytes after the last page that held no page are
    /// counted on [`Source::skipped`].
    End,
    /// A capture pattern was found and the input ended inside that page.
    Truncated {
        expected: u64,
        got: u64,
    },
}

pub struct Source<R> {
    input: Input<R>,
    buf: Vec<u8>,
    /// File offset of `buf[0]`.
    start: u64,
    /// Read cursor inside `buf`.
    at: usize,
    /// The reader returned zero bytes at the end of `buf`.
    eof: bool,
    crc_failures: u64,
    skipped: u64,
}

impl<R: Read> Source<R> {
    pub fn new(input: Input<R>) -> Self {
        Self {
            input,
            buf: Vec::new(),
            start: 0,
            at: 0,
            eof: false,
            crc_failures: 0,
            skipped: 0,
        }
    }

    /// File offset of the next byte this source returns.
    pub fn position(&self) -> u64 {
        self.start
            .saturating_add(u64::try_from(self.at).unwrap_or(u64::MAX))
    }

    /// Bytes read from the reader so far.
    pub fn consumed(&self) -> u64 {
        self.input.consumed()
    }

    pub fn length(&self) -> Option<u64> {
        self.input.length()
    }

    pub const fn crc_failures(&self) -> u64 {
        self.crc_failures
    }

    pub const fn skipped(&self) -> u64 {
        self.skipped
    }

    /// Makes at least `need` bytes available after the cursor when the
    /// input has them. Returns how many are available.
    fn fill(&mut self, need: usize) -> Result<usize, SourceError> {
        let mut have = self.buf.len() - self.at;
        if have >= need || self.eof {
            return Ok(have);
        }
        if self.at > 0 {
            self.buf.drain(..self.at);
            self.start = self
                .start
                .saturating_add(u64::try_from(self.at).unwrap_or(0));
            self.at = 0;
        }
        while have < need && !self.eof {
            let old = self.buf.len();
            let want = (need - have).max(CHUNK);
            self.buf.resize(old + want, 0);
            let got = match self.input.read(&mut self.buf[old..]) {
                Ok(got) => got,
                Err(error) => {
                    self.buf.truncate(old);
                    return Err(error);
                }
            };
            self.buf.truncate(old + got);
            if got == 0 {
                self.eof = true;
            }
            have = self.buf.len();
        }
        Ok(have)
    }

    fn advance(&mut self, count: usize) {
        self.at = (self.at + count).min(self.buf.len());
    }

    /// Reads the next valid page, scanning past garbage and bad checksums.
    pub fn read_page(&mut self) -> Result<PageRead, Fault> {
        let mut scanned = 0_u64;
        let mut truncated = None;
        loop {
            if scanned > MAX_RESYNC_BYTES {
                if let Some((expected, got)) = truncated {
                    return Ok(PageRead::Truncated { expected, got });
                }
                return Err(OggError::NoSync { scanned }.into());
            }
            let have = self.fill(PAGE_HEADER_BYTES)?;
            if have == 0 {
                self.skipped = self.skipped.saturating_add(scanned);
                if let Some((expected, got)) = truncated {
                    return Ok(PageRead::Truncated { expected, got });
                }
                return Ok(PageRead::End);
            }
            let window = &self.buf[self.at..];
            if window.len() < CAPTURE.len() || &window[..4] != CAPTURE {
                let step = next_capture(window).unwrap_or_else(|| {
                    if self.eof {
                        window.len()
                    } else {
                        window.len().saturating_sub(CAPTURE.len() - 1).max(1)
                    }
                });
                self.advance(step);
                scanned = scanned.saturating_add(u64::try_from(step).unwrap_or(u64::MAX));
                continue;
            }
            match self.candidate()? {
                Candidate::Page(page) => {
                    self.skipped = self.skipped.saturating_add(scanned);
                    return Ok(PageRead::Page(page));
                }
                Candidate::Truncated { expected, got } => {
                    truncated.get_or_insert((expected, got));
                }
                Candidate::Invalid => {}
                Candidate::BadChecksum => {
                    self.crc_failures = self.crc_failures.saturating_add(1);
                }
            }
            self.advance(1);
            scanned = scanned.saturating_add(1);
        }
    }

    /// Checks the candidate at the cursor. On success the cursor moves past it.
    fn candidate(&mut self) -> Result<Candidate, SourceError> {
        let have = self.fill(PAGE_HEADER_BYTES)?;
        let Some(header) = parse_header(&self.buf[self.at..]) else {
            return Ok(truncated_or_invalid(have, PAGE_HEADER_BYTES, self.eof));
        };
        if header.version != 0 {
            return Ok(Candidate::Invalid);
        }
        let table_end = PAGE_HEADER_BYTES + usize::from(header.segments);
        let have = self.fill(table_end)?;
        if have < table_end {
            return Ok(truncated_or_invalid(have, table_end, self.eof));
        }
        let body_len: usize = self.buf[self.at + PAGE_HEADER_BYTES..self.at + table_end]
            .iter()
            .map(|&value| usize::from(value))
            .sum();
        let total = table_end + body_len;
        let have = self.fill(total)?;
        if have < total {
            return Ok(truncated_or_invalid(have, total, self.eof));
        }
        let bytes = &self.buf[self.at..self.at + total];
        if !verify(bytes, &header) {
            return Ok(Candidate::BadChecksum);
        }
        let page = Page {
            offset: self.position(),
            flags: header.flags,
            granule: header.granule,
            serial: header.serial,
            sequence: header.sequence,
            lacing: bytes[PAGE_HEADER_BYTES..table_end].to_vec(),
            body: bytes[table_end..].to_vec(),
        };
        self.advance(total);
        Ok(Candidate::Page(page))
    }
}

impl<R: Read + Seek> Source<R> {
    /// Moves the cursor to `position`. Inside the buffer this costs no read.
    pub fn seek(&mut self, position: u64) -> Result<(), SourceError> {
        let end = self
            .start
            .saturating_add(u64::try_from(self.buf.len()).unwrap_or(u64::MAX));
        if position >= self.start && position <= end {
            self.at = usize::try_from(position - self.start).unwrap_or(self.buf.len());
            return Ok(());
        }
        self.input.seek(position)?;
        self.buf.clear();
        self.start = position;
        self.at = 0;
        self.eof = false;
        Ok(())
    }
}

enum Candidate {
    Page(Page),
    Truncated {
        expected: u64,
        got: u64,
    },
    /// Not a page: wrong version.
    Invalid,
    BadChecksum,
}

fn truncated_or_invalid(have: usize, need: usize, eof: bool) -> Candidate {
    if eof {
        Candidate::Truncated {
            expected: u64::try_from(need).unwrap_or(u64::MAX),
            got: u64::try_from(have).unwrap_or(u64::MAX),
        }
    } else {
        Candidate::Invalid
    }
}

/// Offset of the next capture pattern after the first byte of `window`.
fn next_capture(window: &[u8]) -> Option<usize> {
    window
        .windows(CAPTURE.len())
        .skip(1)
        .position(|candidate| candidate == CAPTURE)
        .map(|at| at + 1)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use cutlass_container::{Input, Limits};

    use super::{PageRead, Source};
    use crate::error::{Fault, OggError};
    use crate::page::{FLAG_BOS, encode};

    fn source(bytes: Vec<u8>) -> Source<Cursor<Vec<u8>>> {
        let len = u64::try_from(bytes.len()).expect("len");
        Source::new(Input::random(Cursor::new(bytes), len, Limits::standard()))
    }

    fn page(sequence: u32, body: &[u8]) -> Vec<u8> {
        let len = u8::try_from(body.len()).expect("short body");
        encode(FLAG_BOS, 0, 9, sequence, &[len], body)
    }

    fn next(source: &mut Source<Cursor<Vec<u8>>>) -> PageRead {
        source.read_page().expect("read")
    }

    #[test]
    fn garbage_before_a_page_is_skipped_and_counted() {
        let mut bytes = b"garbage OggX more garbage".to_vec();
        bytes.extend(page(0, b"abc"));
        let mut source = source(bytes);
        match next(&mut source) {
            PageRead::Page(page) => {
                assert_eq!(page.offset, 25);
                assert_eq!(page.body, b"abc");
            }
            other => panic!("unexpected {other:?}"),
        }
        assert_eq!(source.skipped(), 25);
        assert!(matches!(next(&mut source), PageRead::End));
    }

    #[test]
    fn a_bad_checksum_is_counted_and_the_next_page_is_found() {
        let mut first = page(0, b"one");
        let last = first.len() - 1;
        first[last] ^= 0x40;
        let mut bytes = first;
        bytes.extend(page(1, b"two"));
        let mut source = source(bytes);
        match next(&mut source) {
            PageRead::Page(page) => assert_eq!(page.body, b"two"),
            other => panic!("unexpected {other:?}"),
        }
        assert_eq!(source.crc_failures(), 1);
    }

    #[test]
    fn a_page_cut_by_the_end_of_the_file_is_truncated() {
        let whole = page(0, b"payload");
        let cut = whole[..whole.len() - 3].to_vec();
        let mut source = source(cut);
        match next(&mut source) {
            PageRead::Truncated { expected, got } => {
                assert_eq!(expected, u64::try_from(whole.len()).expect("len"));
                assert_eq!(got, expected - 3);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn a_lacing_table_that_claims_more_than_the_file_is_truncated() {
        let mut bytes = page(0, b"x");
        bytes[26] = 200;
        let mut source = source(bytes);
        assert!(matches!(next(&mut source), PageRead::Truncated { .. }));
    }

    #[test]
    fn a_long_run_of_garbage_stops_the_search() {
        let mut source = source(vec![b'x'; 600 * 1024]);
        match source.read_page() {
            Err(Fault::Ogg(OggError::NoSync { scanned })) => {
                assert!(scanned > 256 * 1024);
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(source.consumed() < 400 * 1024);
    }

    #[test]
    fn a_seek_inside_the_buffer_reads_nothing_new() {
        let mut bytes = page(0, b"a");
        bytes.extend(page(1, b"b"));
        let mut source = source(bytes);
        let _ = next(&mut source);
        let consumed = source.consumed();
        source.seek(0).expect("seek");
        match next(&mut source) {
            PageRead::Page(page) => assert_eq!(page.sequence, 0),
            other => panic!("unexpected {other:?}"),
        }
        assert_eq!(source.consumed(), consumed);
    }

    #[test]
    fn an_empty_input_is_the_end() {
        let mut source = source(Vec::new());
        assert!(matches!(next(&mut source), PageRead::End));
    }
}
