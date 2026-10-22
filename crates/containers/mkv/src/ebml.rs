//! EBML element header. An id or a size is a variable-width integer: the
//! leading zeros of the first byte, then a one, choose the width. For a size,
//! that marker bit is not part of the value. All ones in the value bits means
//! the size is unknown.

use crate::Error;

const MAX_DEPTH: u32 = 16;

/// How wide an id or a size may be. The EBML header sets these for the rest of
/// the file. The header itself uses the defaults.
#[derive(Clone, Copy)]
pub(crate) struct Limits {
    pub(crate) max_id: u8,
    pub(crate) max_size: u8,
}

impl Limits {
    pub(crate) const DEFAULT: Self = Self {
        max_id: 4,
        max_size: 8,
    };
}

#[derive(Debug)]
pub(crate) struct Element<'a> {
    pub(crate) id: u64,
    pub(crate) body: &'a [u8],
    /// File offset of `body`.
    pub(crate) body_at: u64,
}

pub(crate) struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
    base: u64,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(data: &'a [u8], base: u64) -> Self {
        Self { data, pos: 0, base }
    }

    pub(crate) fn next(&mut self, limits: Limits) -> Result<Option<Element<'a>>, Error> {
        if self.pos >= self.data.len() {
            return Ok(None);
        }
        let id = self.vint(limits.max_id, false)?.value;
        let size = self.vint(limits.max_size, true)?;
        let body_at = self.locate()?;
        let body = if size.unknown {
            self.rest()
        } else {
            let need = usize::try_from(size.value).map_err(|_| Error::TooLarge)?;
            self.read(need)?
        };
        Ok(Some(Element { id, body, body_at }))
    }

    fn rest(&mut self) -> &'a [u8] {
        let out = &self.data[self.pos..];
        self.pos = self.data.len();
        out
    }

    fn read(&mut self, need: usize) -> Result<&'a [u8], Error> {
        let Some(end) = self.pos.checked_add(need) else {
            return Err(Error::TooLarge);
        };
        if end > self.data.len() {
            return Err(Error::Truncated {
                at: self.locate()?,
                need: u64::try_from(need).map_err(|_| Error::TooLarge)?,
            });
        }
        let out = &self.data[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn locate(&self) -> Result<u64, Error> {
        let pos = u64::try_from(self.pos).map_err(|_| Error::TooLarge)?;
        self.base.checked_add(pos).ok_or(Error::TooLarge)
    }

    /// `strip` clears the width marker. That is how a size is stored. An id
    /// keeps the marker bits, because they are part of the id.
    fn vint(&mut self, max_width: u8, strip: bool) -> Result<RawVint, Error> {
        let first = self.read(1)?[0];
        if first == 0 {
            return Err(if strip { Error::BadSize } else { Error::BadId });
        }
        let width = usize::try_from(first.leading_zeros()).map_err(|_| Error::TooLarge)? + 1;
        if width > 8 || width > usize::from(max_width) {
            return Err(if strip { Error::BadSize } else { Error::BadId });
        }
        let mut raw = u64::from(first);
        if width > 1 {
            for byte in self.read(width - 1)? {
                raw = (raw << 8) | u64::from(*byte);
            }
        }
        if !strip {
            return Ok(RawVint {
                value: raw,
                unknown: false,
            });
        }
        let shift = width * 7;
        if shift >= 64 {
            return Err(Error::BadSize);
        }
        let marker = 1u64 << shift;
        let value = raw & !marker;
        Ok(RawVint {
            value,
            unknown: value == marker - 1,
        })
    }
}

struct RawVint {
    value: u64,
    unknown: bool,
}

pub(crate) fn for_each<'a>(
    body: &'a [u8],
    body_at: u64,
    limits: Limits,
    depth: u32,
    mut visit: impl FnMut(Element<'a>) -> Result<(), Error>,
) -> Result<(), Error> {
    if depth > MAX_DEPTH {
        return Err(Error::Inconsistent("the element is nested too deep"));
    }
    let mut cursor = Cursor::new(body, body_at);
    while let Some(element) = cursor.next(limits)? {
        visit(element)?;
    }
    Ok(())
}

/// Big-endian integer of at most eight bytes. An empty body is zero.
pub(crate) fn uint(bytes: &[u8]) -> Result<u64, Error> {
    if bytes.len() > 8 {
        return Err(Error::TooLarge);
    }
    let mut value = 0u64;
    for byte in bytes {
        value = (value << 8) | u64::from(*byte);
    }
    Ok(value)
}

/// Four-byte or eight-byte IEEE float.
pub(crate) fn float(bytes: &[u8]) -> Result<f64, Error> {
    match bytes.len() {
        4 => {
            let bytes: [u8; 4] = match bytes.try_into() {
                Ok(bytes) => bytes,
                Err(_) => return Err(Error::Inconsistent("a float is the wrong size")),
            };
            Ok(f64::from(f32::from_be_bytes(bytes)))
        }
        8 => {
            let bytes: [u8; 8] = match bytes.try_into() {
                Ok(bytes) => bytes,
                Err(_) => return Err(Error::Inconsistent("a float is the wrong size")),
            };
            Ok(f64::from_be_bytes(bytes))
        }
        _ => Err(Error::Inconsistent("a float is the wrong size")),
    }
}

/// UTF-8 with trailing NUL bytes removed.
pub(crate) fn text(bytes: &[u8]) -> Result<&str, Error> {
    let end = match bytes.iter().rposition(|byte| *byte != 0) {
        Some(index) => index + 1,
        None => 0,
    };
    std::str::from_utf8(&bytes[..end]).map_err(|_| Error::Inconsistent("a string is not UTF-8"))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::Error;

    #[test]
    fn an_element_keeps_its_id_and_body() {
        let bytes = [0x81, 0x82, 0x10, 0x20];
        let mut cursor = Cursor::new(&bytes, 0);
        let element = cursor
            .next(Limits::DEFAULT)
            .expect("header")
            .expect("element");
        assert_eq!(element.id, 0x81);
        assert_eq!(element.body, &[0x10, 0x20]);
        assert_eq!(element.body_at, 2);
        assert!(cursor.next(Limits::DEFAULT).expect("end").is_none());
    }

    #[test]
    fn an_unknown_size_runs_to_the_end() {
        let bytes = [0x81, 0xFF, 0x01, 0x02];
        let mut cursor = Cursor::new(&bytes, 10);
        let element = cursor
            .next(Limits::DEFAULT)
            .expect("header")
            .expect("element");
        assert_eq!(element.body, &[0x01, 0x02]);
        assert_eq!(element.body_at, 12);
        assert!(cursor.next(Limits::DEFAULT).expect("end").is_none());
    }

    #[test]
    fn a_short_body_is_truncated() {
        let bytes = [0x81, 0x82, 0x01];
        let mut cursor = Cursor::new(&bytes, 0);
        match cursor.next(Limits::DEFAULT) {
            Err(Error::Truncated { at: 2, need: 2 }) => {}
            other => panic!("expected a short body, got {other:?}"),
        }
    }

    #[test]
    fn a_zero_width_marker_is_a_bad_id() {
        let mut cursor = Cursor::new(&[0x00], 0);
        match cursor.next(Limits::DEFAULT) {
            Err(Error::BadId) => {}
            other => panic!("expected a bad id, got {other:?}"),
        }
    }
}
