//! Marker and entropy cursor.
//!
//! Entropy bytes treat `FF 00` as a data `FF`. Any other byte after `FF`
//! is a marker and stops the entropy read. Restart alignment discards the
//! pad bits the encoder filled with ones.

use cutlass_decode::CodecError;

pub(crate) struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
    acc: u64,
    nbits: u32,
    pending_marker: Option<u8>,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            acc: 0,
            nbits: 0,
            pending_marker: None,
        }
    }

    pub(crate) fn at(&self) -> u64 {
        u64::try_from(self.pos)
            .unwrap_or(u64::MAX)
            .saturating_mul(8)
    }

    pub(crate) fn take_marker(&mut self) -> Result<u8, CodecError> {
        self.nbits = 0;
        self.acc = 0;
        if let Some(marker) = self.pending_marker.take() {
            return Ok(marker);
        }
        self.next_marker()
    }

    pub(crate) fn next_marker(&mut self) -> Result<u8, CodecError> {
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
        let n = len - 2;
        let end = self
            .pos
            .checked_add(n)
            .ok_or(CodecError::Limit { what: "segment" })?;
        if end > self.data.len() {
            return Err(self.truncated("segment"));
        }
        let start = self.pos;
        self.pos = end;
        Ok(&self.data[start..end])
    }

    pub(crate) fn read_bit(&mut self) -> Result<u8, CodecError> {
        if self.nbits == 0 {
            let byte = self.entropy_byte()?;
            self.acc = u64::from(byte);
            self.nbits = 8;
        }
        self.nbits -= 1;
        Ok(u8::try_from((self.acc >> self.nbits) & 1).unwrap_or(0))
    }

    pub(crate) fn read_bits(&mut self, n: u32) -> Result<u32, CodecError> {
        if n > 16 {
            return Err(self.malformed("magnitude"));
        }
        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | u32::from(self.read_bit()?);
        }
        Ok(value)
    }

    fn entropy_byte(&mut self) -> Result<u8, CodecError> {
        if self.pending_marker.is_some() {
            return Err(self.truncated("entropy"));
        }
        let byte = self.read_u8()?;
        if byte != 0xff {
            return Ok(byte);
        }
        let mut marker = self.read_u8()?;
        while marker == 0xff {
            marker = self.read_u8()?;
        }
        if marker == 0x00 {
            return Ok(0xff);
        }
        self.pending_marker = Some(marker);
        Err(self.truncated("entropy"))
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
