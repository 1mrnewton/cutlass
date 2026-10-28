//! An owned byte source.
//!
//! The demuxer owns the reader. A forward source is consumed in order.
//! A random-access source can reposition. `consumed` counts bytes read.
//! A seek does not count as a read. A declared length above the limit is
//! refused before any buffer of that length is allocated.

use crate::error::SourceError;
use crate::limits::Limits;
use std::io::{Read, Seek, SeekFrom};

/// Whether the demuxer may reposition this input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    /// Seek may use an index.
    Random,
    /// Bytes arrive in order. Seek is refused.
    Forward,
}

/// The reader a demuxer owns, plus the bytes it has taken.
#[derive(Debug)]
pub struct Input<R> {
    reader: R,
    access: Access,
    limits: Limits,
    consumed: u64,
    position: u64,
    length: Option<u64>,
}

impl<R: Read> Input<R> {
    /// A forward source of unknown length.
    #[must_use]
    pub fn forward(reader: R, limits: Limits) -> Self {
        Self {
            reader,
            access: Access::Forward,
            limits,
            consumed: 0,
            position: 0,
            length: None,
        }
    }

    #[must_use]
    pub const fn access(&self) -> Access {
        self.access
    }

    #[must_use]
    pub const fn limits(&self) -> Limits {
        self.limits
    }

    /// Bytes successfully read. A refused size and a seek leave this unchanged.
    #[must_use]
    pub const fn consumed(&self) -> u64 {
        self.consumed
    }

    #[must_use]
    pub const fn position(&self) -> u64 {
        self.position
    }

    #[must_use]
    pub const fn length(&self) -> Option<u64> {
        self.length
    }

    /// Bytes left when the length is known.
    #[must_use]
    pub const fn remaining(&self) -> Option<u64> {
        match self.length {
            Some(length) => Some(length.saturating_sub(self.position)),
            None => None,
        }
    }

    /// Reads into `buf`. A short source returns the bytes it had, including zero.
    pub fn read(&mut self, buf: &mut [u8]) -> Result<usize, SourceError> {
        let amount = self.reader.read(buf).map_err(failed)?;
        let amount_u64 = u64::try_from(amount).map_err(|_| SourceError::Failed {
            message: "read length does not fit".to_owned(),
        })?;
        self.consumed = self.consumed.saturating_add(amount_u64);
        self.position = self.position.saturating_add(amount_u64);
        Ok(amount)
    }

    /// Fills `buf`.
    ///
    /// A known-length input that ends early is [`SourceError::Truncated`].
    /// A forward input that ends early is [`SourceError::ShortRead`].
    pub fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), SourceError> {
        let mut filled = 0;
        while filled < buf.len() {
            let amount = self.read(&mut buf[filled..])?;
            if amount == 0 {
                let expected = u64::try_from(buf.len()).unwrap_or(u64::MAX);
                let got = u64::try_from(filled).unwrap_or(u64::MAX);
                if self.length.is_some() {
                    return Err(SourceError::Truncated { expected, got });
                }
                return Err(SourceError::ShortRead { expected, got });
            }
            filled += amount;
        }
        Ok(())
    }

    /// Reads `declared` bytes, refusing the read before allocating past `limit`.
    pub fn read_limited(&mut self, declared: u64, limit: u64) -> Result<Vec<u8>, SourceError> {
        if declared > limit {
            return Err(SourceError::HostileSize { declared, limit });
        }
        if let Some(length) = self.length {
            let available = length.saturating_sub(self.position);
            if declared > available {
                return Err(SourceError::HostileSize {
                    declared,
                    limit: available,
                });
            }
        }
        let len =
            usize::try_from(declared).map_err(|_| SourceError::HostileSize { declared, limit })?;
        if len == 0 {
            return Ok(Vec::new());
        }
        let mut buf = Vec::new();
        buf.try_reserve_exact(len)
            .map_err(|_| SourceError::Failed {
                message: "out of memory".to_owned(),
            })?;
        buf.resize(len, 0);
        self.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Reads one access unit, bounded by [`Limits::max_packet_bytes`].
    pub fn read_declared(&mut self, declared: u64) -> Result<Vec<u8>, SourceError> {
        self.read_limited(declared, self.limits.max_packet_bytes)
    }
}

impl<R: Read + Seek> Input<R> {
    /// A random-access source whose length is known.
    ///
    /// The length is not allocated. It is the bound a declared size is
    /// checked against.
    #[must_use]
    pub fn random(reader: R, length: u64, limits: Limits) -> Self {
        Self {
            reader,
            access: Access::Random,
            limits,
            consumed: 0,
            position: 0,
            length: Some(length),
        }
    }

    /// Repositions a random-access input.
    ///
    /// A forward input is refused and stays where it is. A position past
    /// the known length is refused and stays where it is. Consumed bytes
    /// do not change.
    pub fn seek(&mut self, position: u64) -> Result<(), SourceError> {
        if self.access == Access::Forward {
            return Err(SourceError::ForwardOnly);
        }
        let Some(length) = self.length else {
            return Err(SourceError::ForwardOnly);
        };
        if position > length {
            return Err(SourceError::OutOfRange { position, length });
        }
        let landed = self
            .reader
            .seek(SeekFrom::Start(position))
            .map_err(failed)?;
        if landed != position {
            return Err(SourceError::Failed {
                message: "seek landed at a different position".to_owned(),
            });
        }
        self.position = landed;
        Ok(())
    }
}

fn failed(error: std::io::Error) -> SourceError {
    SourceError::Failed {
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::{Access, Input};
    use crate::error::SourceError;
    use crate::limits::Limits;
    use std::io::{self, Cursor, Read, Seek, SeekFrom};

    fn random(bytes: &[u8]) -> Input<Cursor<Vec<u8>>> {
        let length = u64::try_from(bytes.len()).expect("len");
        Input::random(Cursor::new(bytes.to_vec()), length, Limits::standard())
    }

    #[test]
    fn the_input_owns_the_reader_and_counts_consumed_bytes() {
        let mut input = random(b"abcdef");
        let mut buf = [0; 1];
        assert_eq!(input.read(&mut buf).expect("byte"), 1);
        assert_eq!(buf, [b'a']);
        assert_eq!(input.consumed(), 1);
        assert_eq!(input.position(), 1);
        let rest = input.read_declared(2).expect("two");
        assert_eq!(rest, b"bc");
        assert_eq!(input.consumed(), 3);
    }

    #[test]
    fn a_seek_does_not_consume_and_a_forward_input_cannot_seek() {
        let mut input = random(b"abcdef");
        input.seek(3).expect("seek");
        assert_eq!(input.position(), 3);
        assert_eq!(input.consumed(), 0);
        let mut forward = Input::forward(Cursor::new(b"abcdef".to_vec()), Limits::standard());
        assert_eq!(forward.access(), Access::Forward);
        let before = forward.position();
        assert_eq!(forward.seek(1), Err(SourceError::ForwardOnly));
        assert_eq!(forward.position(), before);
        assert_eq!(forward.consumed(), 0);
    }

    #[test]
    fn a_seek_past_the_end_leaves_the_position() {
        let mut input = random(b"ab");
        assert_eq!(
            input.seek(3),
            Err(SourceError::OutOfRange {
                position: 3,
                length: 2
            })
        );
        assert_eq!(input.position(), 0);
    }

    #[test]
    fn a_declared_size_past_the_limit_is_not_allocated() {
        let mut input = random(&[0; 8]);
        let error = input.read_declared(u64::MAX).expect_err("limit");
        assert_eq!(
            error,
            SourceError::HostileSize {
                declared: u64::MAX,
                limit: Limits::standard().max_packet_bytes
            }
        );
        assert_eq!(input.consumed(), 0);
    }

    #[test]
    fn a_declared_size_past_the_remaining_input_is_not_allocated() {
        let mut input = random(b"abcd");
        let error = input.read_declared(8).expect_err("remaining");
        assert_eq!(
            error,
            SourceError::HostileSize {
                declared: 8,
                limit: 4
            }
        );
        assert_eq!(input.consumed(), 0);
        assert_eq!(input.position(), 0);
    }

    #[test]
    fn a_known_length_that_ends_early_is_truncated() {
        let mut input = Input::random(Cursor::new(b"ab".to_vec()), 8, Limits::standard());
        let mut buf = [0; 4];
        assert_eq!(
            input.read_exact(&mut buf),
            Err(SourceError::Truncated {
                expected: 4,
                got: 2
            })
        );
    }

    #[test]
    fn a_forward_input_that_ends_early_is_a_short_read() {
        let mut input = Input::forward(Cursor::new(b"ab".to_vec()), Limits::standard());
        let mut buf = [0; 4];
        assert_eq!(
            input.read_exact(&mut buf),
            Err(SourceError::ShortRead {
                expected: 4,
                got: 2
            })
        );
        assert_eq!(input.consumed(), 2);
    }

    #[test]
    fn input_is_send_when_the_reader_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Input<Cursor<Vec<u8>>>>();
    }

    /// A reader whose `read` always fails.
    struct Broken;

    impl Read for Broken {
        fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::other("disk"))
        }
    }

    impl Seek for Broken {
        fn seek(&mut self, _pos: SeekFrom) -> io::Result<u64> {
            Err(io::Error::other("disk"))
        }
    }

    #[test]
    fn a_source_failure_is_not_a_short_read() {
        let mut input = Input::forward(Broken, Limits::standard());
        let mut buf = [0; 1];
        let error = input.read(&mut buf).expect_err("disk");
        assert!(matches!(error, SourceError::Failed { .. }));
        assert_eq!(input.consumed(), 0);
    }
}
