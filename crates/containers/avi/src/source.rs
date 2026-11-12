//! Buffered random-access reads over the reader a demuxer owns.
//!
//! Chunk headers are eight bytes and most AVI chunks are small, so reads go
//! through one read-ahead buffer. A seek inside the buffer costs nothing. A
//! read at least as long as the buffer goes straight into its destination.
//! `consumed` counts the bytes taken from the reader, read-ahead included,
//! so a seek that skips a payload does not count that payload.

use std::io::{Read, Seek, SeekFrom};

use cutlass_container::SourceError;

/// Read-ahead size. A chunk header and a small packet share one fill.
pub(crate) const BUFFER: usize = 64 * 1024;

pub(crate) struct Source<R> {
    reader: R,
    length: u64,
    buf: Vec<u8>,
    /// File offset of `buf[0]`.
    buf_start: u64,
    /// Valid bytes in `buf`.
    buf_len: usize,
    /// Where the next read starts.
    pos: u64,
    /// Where the reader itself stands, when known.
    io_pos: Option<u64>,
    consumed: u64,
}

impl<R: Read + Seek> Source<R> {
    pub(crate) fn new(reader: R, length: u64) -> Self {
        Self {
            reader,
            length,
            buf: Vec::new(),
            buf_start: 0,
            buf_len: 0,
            pos: 0,
            io_pos: None,
            consumed: 0,
        }
    }

    pub(crate) const fn length(&self) -> u64 {
        self.length
    }

    #[cfg(test)]
    pub(crate) const fn position(&self) -> u64 {
        self.pos
    }

    pub(crate) const fn consumed(&self) -> u64 {
        self.consumed
    }

    /// Moves the next read. Nothing is read, and a position past the end is refused.
    pub(crate) fn seek(&mut self, position: u64) -> Result<(), SourceError> {
        if position > self.length {
            return Err(SourceError::OutOfRange {
                position,
                length: self.length,
            });
        }
        self.pos = position;
        Ok(())
    }

    pub(crate) fn remaining(&self) -> u64 {
        self.length.saturating_sub(self.pos)
    }

    /// Fills `out` from the current position.
    pub(crate) fn read_exact(&mut self, out: &mut [u8]) -> Result<(), SourceError> {
        let want = u64::try_from(out.len()).unwrap_or(u64::MAX);
        if want > self.remaining() {
            return Err(SourceError::Truncated {
                expected: want,
                got: self.remaining(),
            });
        }
        let mut filled = 0;
        while filled < out.len() {
            if let Some(at) = self.buffered_offset() {
                let have = self.buf_len - at;
                let take = have.min(out.len() - filled);
                out[filled..filled + take].copy_from_slice(&self.buf[at..at + take]);
                filled += take;
                self.advance(take);
                continue;
            }
            let rest = out.len() - filled;
            if rest >= BUFFER {
                self.direct(&mut out[filled..])?;
                return Ok(());
            }
            self.fill()?;
        }
        Ok(())
    }

    /// Reads `len` bytes into a new buffer after checking `len` against
    /// `limit` and the bytes left, so a hostile size allocates nothing.
    pub(crate) fn read_vec(&mut self, len: u64, limit: u64) -> Result<Vec<u8>, SourceError> {
        if len > limit {
            return Err(SourceError::HostileSize {
                declared: len,
                limit,
            });
        }
        if len > self.remaining() {
            return Err(SourceError::Truncated {
                expected: len,
                got: self.remaining(),
            });
        }
        let size = usize::try_from(len).map_err(|_| SourceError::HostileSize {
            declared: len,
            limit,
        })?;
        let mut out = Vec::new();
        out.try_reserve_exact(size)
            .map_err(|_| SourceError::Failed {
                message: "out of memory".to_owned(),
            })?;
        out.resize(size, 0);
        self.read_exact(&mut out)?;
        Ok(out)
    }

    pub(crate) fn read_array<const N: usize>(&mut self) -> Result<[u8; N], SourceError> {
        let mut out = [0; N];
        self.read_exact(&mut out)?;
        Ok(out)
    }

    fn buffered_offset(&self) -> Option<usize> {
        let end = self.buf_start.checked_add(u64::try_from(self.buf_len).ok()?)?;
        if self.pos >= self.buf_start && self.pos < end {
            usize::try_from(self.pos - self.buf_start).ok()
        } else {
            None
        }
    }

    fn advance(&mut self, count: usize) {
        self.pos = self
            .pos
            .saturating_add(u64::try_from(count).unwrap_or(u64::MAX));
    }

    fn place(&mut self) -> Result<(), SourceError> {
        if self.io_pos == Some(self.pos) {
            return Ok(());
        }
        let landed = self
            .reader
            .seek(SeekFrom::Start(self.pos))
            .map_err(failed)?;
        if landed != self.pos {
            self.io_pos = None;
            return Err(SourceError::Failed {
                message: "seek landed at a different position".to_owned(),
            });
        }
        self.io_pos = Some(self.pos);
        Ok(())
    }

    fn fill(&mut self) -> Result<(), SourceError> {
        self.place()?;
        if self.buf.len() < BUFFER {
            self.buf.resize(BUFFER, 0);
        }
        let room = usize::try_from(self.remaining())
            .unwrap_or(usize::MAX)
            .min(BUFFER);
        let mut got = 0;
        while got < room {
            let amount = self.reader.read(&mut self.buf[got..room]).map_err(failed)?;
            if amount == 0 {
                break;
            }
            got += amount;
        }
        self.count(got);
        self.buf_start = self.pos;
        self.buf_len = got;
        if got == 0 {
            return Err(SourceError::Truncated {
                expected: u64::try_from(room).unwrap_or(u64::MAX),
                got: 0,
            });
        }
        Ok(())
    }

    fn direct(&mut self, out: &mut [u8]) -> Result<(), SourceError> {
        self.place()?;
        let mut got = 0;
        while got < out.len() {
            let amount = self.reader.read(&mut out[got..]).map_err(failed)?;
            if amount == 0 {
                self.count(got);
                return Err(SourceError::Truncated {
                    expected: u64::try_from(out.len()).unwrap_or(u64::MAX),
                    got: u64::try_from(got).unwrap_or(u64::MAX),
                });
            }
            got += amount;
        }
        self.count(got);
        self.advance(got);
        Ok(())
    }

    fn count(&mut self, amount: usize) {
        let amount = u64::try_from(amount).unwrap_or(u64::MAX);
        self.consumed = self.consumed.saturating_add(amount);
        self.io_pos = self.io_pos.map(|at| at.saturating_add(amount));
    }
}

fn failed(error: std::io::Error) -> SourceError {
    SourceError::Failed {
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::{BUFFER, Source};
    use cutlass_container::SourceError;
    use std::io::Cursor;

    fn source(len: usize) -> Source<Cursor<Vec<u8>>> {
        let bytes: Vec<u8> = (0..len).map(|i| u8::try_from(i % 251).unwrap()).collect();
        Source::new(Cursor::new(bytes), u64::try_from(len).unwrap())
    }

    #[test]
    fn small_reads_share_one_fill_and_a_seek_inside_it_reads_nothing() {
        let mut src = source(1000);
        assert_eq!(src.read_array::<4>().unwrap(), [0, 1, 2, 3]);
        assert_eq!(src.consumed(), 1000);
        src.seek(500).unwrap();
        assert_eq!(src.read_array::<2>().unwrap(), [500 % 251, 501 % 251].map(|v| v as u8));
        assert_eq!(src.consumed(), 1000);
    }

    #[test]
    fn a_long_read_goes_straight_to_its_destination() {
        let len = BUFFER * 3;
        let mut src = source(len);
        src.seek(10).unwrap();
        let out = src.read_vec(u64::try_from(BUFFER * 2).unwrap(), u64::MAX).unwrap();
        assert_eq!(out[0], 10);
        assert_eq!(src.consumed(), u64::try_from(BUFFER * 2).unwrap());
        assert_eq!(src.position(), u64::try_from(BUFFER * 2 + 10).unwrap());
    }

    #[test]
    fn a_hostile_or_truncated_size_allocates_nothing() {
        let mut src = source(16);
        assert!(matches!(
            src.read_vec(u64::MAX, 1 << 20),
            Err(SourceError::HostileSize { .. })
        ));
        assert!(matches!(
            src.read_vec(17, 1 << 20),
            Err(SourceError::Truncated { expected: 17, got: 16 })
        ));
        assert_eq!(src.consumed(), 0);
        assert!(matches!(src.seek(17), Err(SourceError::OutOfRange { .. })));
    }

    #[test]
    fn a_reader_shorter_than_its_declared_length_is_truncated() {
        let mut src = Source::new(Cursor::new(vec![1u8; 8]), 64);
        let mut out = [0; 16];
        assert!(matches!(
            src.read_exact(&mut out),
            Err(SourceError::Truncated { .. })
        ));
    }
}
