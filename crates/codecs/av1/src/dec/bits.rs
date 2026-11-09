//! MSB-first bit reader for OBU headers (spec 4.10 and 8.1).

use cutlass_decode::CodecError;

pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize, // bit position
}

pub type R<T> = Result<T, CodecError>;

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub fn position(&self) -> usize {
        self.pos
    }

    pub fn byte_pos(&self) -> usize {
        self.pos.div_ceil(8)
    }

    fn truncated(&self) -> CodecError {
        CodecError::Truncated {
            at_bit: self.pos as u64,
            what: "obu bits",
        }
    }

    #[inline]
    pub fn bit(&mut self) -> R<u32> {
        let byte = *self
            .data
            .get(self.pos >> 3)
            .ok_or_else(|| self.truncated())?;
        let b = (byte >> (7 - (self.pos & 7))) & 1;
        self.pos += 1;
        Ok(u32::from(b))
    }

    pub fn flag(&mut self) -> R<bool> {
        Ok(self.bit()? != 0)
    }

    /// f(n), n <= 32.
    pub fn f(&mut self, n: u32) -> R<u32> {
        let mut v: u64 = 0;
        for _ in 0..n {
            v = (v << 1) | u64::from(self.bit()?);
        }
        Ok(v as u32)
    }

    /// su(n)
    pub fn su(&mut self, n: u32) -> R<i32> {
        let v = self.f(n)? as i64;
        let sign = 1i64 << (n - 1);
        Ok(if v & sign != 0 {
            (v - 2 * sign) as i32
        } else {
            v as i32
        })
    }

    /// ns(n)
    pub fn ns(&mut self, n: u32) -> R<u32> {
        if n <= 1 {
            return Ok(0);
        }
        let w = 32 - n.leading_zeros(); // FloorLog2(n) + 1
        let m = (1u32 << w) - n;
        let v = self.f(w - 1)?;
        if v < m {
            return Ok(v);
        }
        let extra = self.bit()?;
        Ok((v << 1) - m + extra)
    }

    /// le(n) bytes little endian
    pub fn le(&mut self, n: u32) -> R<u32> {
        let mut t = 0u32;
        for i in 0..n {
            t = t.wrapping_add(self.f(8)? << (i * 8));
        }
        Ok(t)
    }

    pub fn uvlc(&mut self) -> R<u32> {
        let mut leading = 0u32;
        while self.bit()? == 0 {
            leading += 1;
            if leading >= 32 {
                return Ok(u32::MAX);
            }
        }
        let v = self.f(leading)?;
        Ok(v.wrapping_add((1u32 << leading).wrapping_sub(1)))
    }

    pub fn byte_alignment(&mut self) {
        self.pos = (self.pos + 7) & !7;
    }
}

/// leb128() from a byte slice. Returns (value, bytes used).
pub fn leb128(data: &[u8]) -> R<(u64, usize)> {
    let mut value: u64 = 0;
    for i in 0..8 {
        let b = *data.get(i).ok_or(CodecError::Truncated {
            at_bit: (i * 8) as u64,
            what: "leb128",
        })?;
        value |= u64::from(b & 0x7f) << (i * 7);
        if b & 0x80 == 0 {
            return Ok((value, i + 1));
        }
    }
    Err(CodecError::Malformed {
        at_bit: 64,
        what: "leb128 too long",
    })
}
