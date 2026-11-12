//! Element headers read from the owned input, through a read-ahead buffer.
//!
//! The demuxer reads one element header at a time and either descends into
//! it (Segment, Cluster, BlockGroup), reads its data (bounded), or skips it.
//! Headers are a few octets each, so they come out of a buffer; frame data
//! larger than what is buffered is read straight into the packet's vector.
//! A skip or seek that stays inside the buffer moves a cursor and makes no
//! call on the reader, which keeps a seek's walk over block headers cheap.
//! A random-access input skips further with a seek; a forward input reads
//! and drops.

use std::io::{Read, Seek};

use cutlass_container::{Access, Input, SourceError};

use crate::ebml::{self, Size};
use crate::error::{Error, Result};

/// Read-ahead size. Element headers and small frames come out of it.
const BUFFER: usize = 64 * 1024;

/// The first read after a seek outside the buffer. A seek's walk reads one
/// block header and skips the frame, so a full buffer there would mostly
/// be frame bytes nobody looks at. Each refill after it doubles, back up to
/// [`BUFFER`] for a sequential read.
const FIRST_READ: usize = 4 * 1024;

/// One element header and where it sits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub id: u32,
    pub size: Size,
    /// Offset of the ID.
    pub start: u64,
    /// Offset of the first data byte.
    pub data: u64,
}

impl Header {
    /// Offset just past the data, when the size is known.
    #[must_use]
    pub fn end(&self) -> Option<u64> {
        match self.size {
            Size::Known(size) => self.data.checked_add(size),
            Size::Unknown => None,
        }
    }
}

pub struct Source<R> {
    input: Input<R>,
    buffer: Box<[u8]>,
    /// Next unread byte in `buffer`.
    at: usize,
    /// Bytes of `buffer` that hold data.
    filled: usize,
    /// Size of the next refill.
    window: usize,
}

impl<R: Read + Seek> Source<R> {
    pub fn new(input: Input<R>) -> Self {
        Self {
            input,
            buffer: vec![0; BUFFER].into_boxed_slice(),
            at: 0,
            filled: 0,
            window: FIRST_READ,
        }
    }

    /// Offset of the next byte the demuxer reads.
    pub fn position(&self) -> u64 {
        let ahead = u64::try_from(self.filled - self.at).unwrap_or(0);
        self.input.position().saturating_sub(ahead)
    }

    pub const fn length(&self) -> Option<u64> {
        self.input.length()
    }

    /// Bytes taken from the reader, read-ahead included.
    pub const fn consumed(&self) -> u64 {
        self.input.consumed()
    }

    pub fn random(&self) -> bool {
        self.input.access() == Access::Random
    }

    const fn buffered(&self) -> usize {
        self.filled - self.at
    }

    /// Refills the buffer when it is empty. Returns the bytes now buffered;
    /// zero at the end of the input.
    fn refill(&mut self) -> Result<usize> {
        if self.at < self.filled {
            return Ok(self.filled - self.at);
        }
        self.at = 0;
        self.filled = 0;
        let window = self.window;
        self.window = (window * 2).min(BUFFER);
        let want = match self.input.remaining() {
            Some(left) => usize::try_from(left).unwrap_or(window).min(window),
            None => window,
        };
        if want == 0 {
            return Ok(0);
        }
        let got = self
            .input
            .read(&mut self.buffer[..want])
            .map_err(Error::Source)?;
        self.filled = got;
        Ok(got)
    }

    /// The next element header, or `None` when the input ends exactly here.
    pub fn header(&mut self) -> Result<Option<Header>> {
        let start = self.position();
        if self.refill()? == 0 {
            return Ok(None);
        }
        let first = self.buffer[self.at];
        let id_len = ebml::id_len(first).ok_or(Error::Ebml(ebml::EbmlError::BadId { first }))?;
        let mut raw = [0_u8; 8];
        self.fill(&mut raw[..id_len])?;
        let (id, _) = ebml::read_id(&raw[..id_len])?;
        self.fill(&mut raw[..1])?;
        let size_len = ebml::size_len(raw[0]).ok_or(Error::Ebml(ebml::EbmlError::BadSize))?;
        self.fill(&mut raw[1..size_len])?;
        let size = ebml::decode_size(&raw[..size_len], size_len);
        Ok(Some(Header {
            id,
            size,
            start,
            data: self.position(),
        }))
    }

    /// Fills `buf`, mapping an early end to a truncation.
    pub fn fill(&mut self, buf: &mut [u8]) -> Result<()> {
        let from_buffer = self.buffered().min(buf.len());
        buf[..from_buffer].copy_from_slice(&self.buffer[self.at..self.at + from_buffer]);
        self.at += from_buffer;
        let total = buf.len();
        let rest = &mut buf[from_buffer..];
        if rest.is_empty() {
            return Ok(());
        }
        if rest.len() >= BUFFER {
            return self.input.read_exact(rest).map_err(Error::Source);
        }
        let mut done = 0;
        while done < rest.len() {
            if self.refill()? == 0 {
                let expected = u64::try_from(total).unwrap_or(u64::MAX);
                let got = u64::try_from(from_buffer + done).unwrap_or(u64::MAX);
                return Err(Error::Source(if self.length().is_some() {
                    SourceError::Truncated { expected, got }
                } else {
                    SourceError::ShortRead { expected, got }
                }));
            }
            let take = self.buffered().min(rest.len() - done);
            rest[done..done + take].copy_from_slice(&self.buffer[self.at..self.at + take]);
            self.at += take;
            done += take;
        }
        Ok(())
    }

    fn check(&self, len: u64, limit: u64) -> Result<usize> {
        if len > limit {
            return Err(Error::Limit {
                what: "element bytes",
                value: len,
                limit,
            });
        }
        if let Some(length) = self.length() {
            let available = length.saturating_sub(self.position());
            if len > available {
                return Err(Error::Source(SourceError::Truncated {
                    expected: len,
                    got: available,
                }));
            }
        }
        usize::try_from(len).map_err(|_| Error::Limit {
            what: "element bytes",
            value: len,
            limit,
        })
    }

    /// Reads `len` bytes, refused before allocation when above `limit` or
    /// past the end of a known-length input.
    pub fn bytes(&mut self, len: u64, limit: u64) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        self.append(&mut out, len, limit)?;
        Ok(out)
    }

    /// Reads `len` bytes into the end of `out`, bounded like [`Self::bytes`].
    pub fn append(&mut self, out: &mut Vec<u8>, len: u64, limit: u64) -> Result<()> {
        let len = self.check(len, limit)?;
        let start = out.len();
        out.try_reserve_exact(len).map_err(|_| Error::Limit {
            what: "element bytes",
            value: u64::try_from(len).unwrap_or(u64::MAX),
            limit,
        })?;
        out.resize(start + len, 0);
        self.fill(&mut out[start..])
    }

    /// Moves past `len` bytes without keeping them.
    pub fn skip(&mut self, len: u64) -> Result<()> {
        let target = self
            .position()
            .checked_add(len)
            .ok_or(Error::malformed("element size overflows"))?;
        if self.random() {
            return self.seek(target);
        }
        let mut rest = len;
        while rest > 0 {
            if self.refill()? == 0 {
                return Err(Error::Source(SourceError::ShortRead {
                    expected: len,
                    got: len - rest,
                }));
            }
            let take = usize::try_from(rest)
                .unwrap_or(usize::MAX)
                .min(self.buffered());
            self.at += take;
            rest -= u64::try_from(take).unwrap_or(rest);
        }
        Ok(())
    }

    /// Repositions a random-access input. A target past the known end is a
    /// truncation: the element that pointed there runs past the file. A
    /// target inside the buffer makes no call on the reader.
    pub fn seek(&mut self, position: u64) -> Result<()> {
        if let Some(length) = self.length()
            && position > length
        {
            return Err(Error::Source(SourceError::Truncated {
                expected: position,
                got: length,
            }));
        }
        let end = self.input.position();
        let begin = end.saturating_sub(u64::try_from(self.filled).unwrap_or(0));
        if (begin..=end).contains(&position) {
            self.at = usize::try_from(position - begin).unwrap_or(self.filled);
            return Ok(());
        }
        self.at = 0;
        self.filled = 0;
        self.window = FIRST_READ;
        self.input.seek(position).map_err(Error::Source)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use cutlass_container::{Input, Limits};

    use super::{BUFFER, Source};
    use crate::ebml::Size;

    #[test]
    fn reads_skips_and_seeks_across_the_buffer() {
        let mut data: Vec<u8> = (0..BUFFER * 3)
            .map(|i| u8::try_from(i % 251).expect("byte"))
            .collect();
        // An element header at 0: ID 0xEC (Void), size 2.
        data[0] = 0xEC;
        data[1] = 0x82;
        let len = u64::try_from(data.len()).expect("len");
        let buffer = u64::try_from(BUFFER).expect("buffer");
        let input = Input::random(Cursor::new(data.clone()), len, Limits::standard());
        let mut source = Source::new(input);
        let header = source.header().expect("header").expect("some");
        assert_eq!(header.id, 0xEC);
        assert_eq!(header.size, Size::Known(2));
        assert_eq!(header.data, 2);
        source.skip(2).expect("skip");
        assert_eq!(source.position(), 4);
        let big = source.bytes(buffer * 2, u64::MAX).expect("big");
        assert_eq!(big, data[4..4 + BUFFER * 2]);
        source.seek(10).expect("back");
        assert_eq!(source.bytes(3, 8).expect("three"), data[10..13]);
        source.seek(len).expect("end");
        assert_eq!(source.header().expect("eof"), None);
        assert!(source.seek(len + 1).is_err());
        source.seek(len - 2).expect("near end");
        assert!(source.bytes(3, 8).is_err());
        assert!(source.bytes(9, 8).is_err());

        let input = Input::forward(Cursor::new(data.clone()), Limits::standard());
        let mut forward = Source::new(input);
        forward.skip(buffer + 5).expect("skip forward");
        assert_eq!(forward.position(), buffer + 5);
        assert_eq!(
            forward.bytes(2, 8).expect("two"),
            data[BUFFER + 5..BUFFER + 7]
        );
        assert!(forward.skip(len).is_err());
        let mut short = [0_u8; 4];
        assert!(forward.fill(&mut short).is_err());
    }
}
