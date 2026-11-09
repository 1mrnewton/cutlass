//! Marker and entropy reader.
//!
//! Entropy bytes treat `FF 00` as a data `FF`. Any other byte after `FF`
//! is a marker and stops the entropy read. The reader keeps up to 64 bits
//! buffered. Past a marker or the end of the data it buffers zero bits and
//! counts them, so a block that consumes one of them is truncated.

use cutlass_decode::CodecError;

pub(crate) struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
    /// Buffered bits, most significant first.
    acc: u64,
    nbits: u32,
    /// Zero bits buffered past a marker or the end of the data.
    fake: u32,
    pending_marker: Option<u8>,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            acc: 0,
            nbits: 0,
            fake: 0,
            pending_marker: None,
        }
    }

    pub(crate) fn at(&self) -> u64 {
        u64::try_from(self.pos)
            .unwrap_or(u64::MAX)
            .saturating_mul(8)
    }

    /// Bytes read so far. After the end-of-image marker this is the image length.
    pub(crate) fn offset(&self) -> usize {
        self.pos
    }

    pub(crate) fn take_marker(&mut self) -> Result<u8, CodecError> {
        self.acc = 0;
        self.nbits = 0;
        self.fake = 0;
        if let Some(marker) = self.pending_marker.take() {
            return Ok(marker);
        }
        self.next_marker()
    }

    fn next_marker(&mut self) -> Result<u8, CodecError> {
        let mut byte = self.read_u8()?;
        if byte != 0xff {
            return Err(self.malformed("marker"));
        }
        loop {
            byte = self.read_u8()?;
            if byte != 0xff {
                return Ok(byte);
            }
        }
    }

    pub(crate) fn segment(&mut self) -> Result<&'a [u8], CodecError> {
        let len = usize::from(self.read_u16()?);
        if len < 2 {
            return Err(self.malformed("length"));
        }
        let end = self
            .pos
            .checked_add(len - 2)
            .ok_or(CodecError::Limit { what: "segment" })?;
        let Some(body) = self.data.get(self.pos..end) else {
            return Err(self.truncated("segment"));
        };
        self.pos = end;
        Ok(body)
    }

    /// Fill the buffer to at least 57 bits.
    #[inline]
    fn refill(&mut self) {
        while self.nbits <= 56 {
            let byte = self.entropy_byte();
            self.acc |= u64::from(byte) << (56 - self.nbits);
            self.nbits += 8;
        }
    }

    #[inline]
    fn entropy_byte(&mut self) -> u8 {
        if self.pending_marker.is_none() {
            match self.data.get(self.pos) {
                Some(&0xff) => return self.after_ff(),
                Some(&byte) => {
                    self.pos += 1;
                    return byte;
                }
                None => {}
            }
        }
        self.fake += 8;
        0
    }

    fn after_ff(&mut self) -> u8 {
        let mut next = self.pos + 1;
        while self.data.get(next) == Some(&0xff) {
            next += 1;
        }
        match self.data.get(next) {
            Some(0) => {
                self.pos = next + 1;
                0xff
            }
            Some(&marker) => {
                self.pos = next + 1;
                self.pending_marker = Some(marker);
                self.fake += 8;
                0
            }
            None => {
                self.pos = self.data.len();
                self.fake += 8;
                0
            }
        }
    }

    /// The next `n` bits, `n` in 1..=16, without consuming them.
    #[inline]
    pub(crate) fn peek(&mut self, n: u32) -> u32 {
        if self.nbits < n {
            self.refill();
        }
        u32::try_from(self.acc >> (64 - n)).unwrap_or(0)
    }

    #[inline]
    pub(crate) fn consume(&mut self, n: u32) {
        self.acc <<= n;
        self.nbits -= n;
    }

    pub(crate) fn read_bit(&mut self) -> Result<u32, CodecError> {
        let bit = self.peek(1);
        self.consume(1);
        self.check()?;
        Ok(bit)
    }

    /// `n` bits, `n` in 0..=16.
    #[inline]
    pub(crate) fn read_bits(&mut self, n: u32) -> Result<u32, CodecError> {
        if n == 0 {
            return Ok(0);
        }
        if n > 16 {
            return Err(self.malformed("magnitude"));
        }
        let value = self.peek(n);
        self.consume(n);
        Ok(value)
    }

    /// Fails once a read has used a bit that is not in the data.
    #[inline]
    pub(crate) fn check(&self) -> Result<(), CodecError> {
        if self.nbits < self.fake {
            return Err(self.truncated("entropy"));
        }
        Ok(())
    }

    /// End an entropy segment. Only the pad bits of the last byte may be left.
    pub(crate) fn end_entropy(&mut self) -> Result<(), CodecError> {
        self.check()?;
        if self.nbits - self.fake >= 8 {
            return Err(self.malformed("entropy"));
        }
        Ok(())
    }

    fn read_u8(&mut self) -> Result<u8, CodecError> {
        let Some(byte) = self.data.get(self.pos).copied() else {
            return Err(self.truncated("marker"));
        };
        self.pos += 1;
        Ok(byte)
    }

    fn read_u16(&mut self) -> Result<u16, CodecError> {
        let hi = u16::from(self.read_u8()?);
        let lo = u16::from(self.read_u8()?);
        Ok((hi << 8) | lo)
    }

    pub(crate) fn malformed(&self, what: &'static str) -> CodecError {
        CodecError::Malformed {
            at_bit: self.at(),
            what,
        }
    }

    pub(crate) fn truncated(&self, what: &'static str) -> CodecError {
        CodecError::Truncated {
            at_bit: self.at(),
            what,
        }
    }
}
