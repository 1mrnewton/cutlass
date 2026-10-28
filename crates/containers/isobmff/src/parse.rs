//! Box headers. Size 0 extends to the parent. Size 1 is a 64-bit size.
//! A `uuid` box carries 16 extra type bytes and is skipped by its size.

use crate::legacy::Error;
use crate::limits::MAX_ATOMS;

pub struct Reader<'a> {
    data: &'a [u8],
    index: usize,
    base: u64,
}

impl<'a> Reader<'a> {
    #[must_use]
    pub const fn new(data: &'a [u8], base: u64) -> Self {
        Self {
            data,
            index: 0,
            base,
        }
    }

    #[must_use]
    pub fn remaining(&self) -> &'a [u8] {
        self.data.get(self.index..).unwrap_or(&[])
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.index >= self.data.len()
    }

    fn position(&self) -> u64 {
        self.base
            .saturating_add(u64::try_from(self.index).unwrap_or(u64::MAX))
    }

    pub fn u8(&mut self) -> Result<u8, Error> {
        let Some(value) = self.data.get(self.index).copied() else {
            return Err(self.short(1));
        };
        self.index += 1;
        Ok(value)
    }

    pub fn u16(&mut self) -> Result<u16, Error> {
        Ok(u16::from_be_bytes(self.array()?))
    }

    pub fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_be_bytes(self.array()?))
    }

    pub fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_be_bytes(self.array()?))
    }

    pub fn i32(&mut self) -> Result<i32, Error> {
        Ok(i32::from_be_bytes(self.array()?))
    }

    pub fn i64(&mut self) -> Result<i64, Error> {
        Ok(i64::from_be_bytes(self.array()?))
    }

    pub fn skip(&mut self, len: usize) -> Result<(), Error> {
        let Some(next) = self.index.checked_add(len) else {
            return Err(Error::HostileSize {
                at: self.position(),
                claimed: u64::MAX,
            });
        };
        if next > self.data.len() {
            return Err(self.short(len));
        }
        self.index = next;
        Ok(())
    }

    pub fn take(&mut self, len: usize) -> Result<&'a [u8], Error> {
        let start = self.index;
        self.skip(len)?;
        self.data
            .get(start..self.index)
            .ok_or_else(|| self.short(len))
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        let bytes = self.take(N)?;
        let mut out = [0u8; N];
        out.copy_from_slice(bytes);
        Ok(out)
    }

    fn short(&self, needed: usize) -> Error {
        Error::Truncated {
            at: self.position(),
            needed: self
                .position()
                .saturating_add(u64::try_from(needed).unwrap_or(u64::MAX)),
        }
    }
}

#[derive(Debug)]
pub struct Atom<'a> {
    pub kind: [u8; 4],
    pub payload: &'a [u8],
    /// File offset of `payload`.
    pub payload_at: u64,
}

/// Walk the boxes in `data`. `base` is the file offset of `data[0]`.
pub fn atoms<'a>(data: &'a [u8], base: u64, budget: &mut u32) -> Result<Vec<Atom<'a>>, Error> {
    let mut out = Vec::new();
    let mut index = 0usize;
    while index + 8 <= data.len() {
        *budget = budget.saturating_add(1);
        if *budget > MAX_ATOMS {
            return Err(Error::Limit {
                what: "boxes",
                value: u64::from(*budget),
            });
        }
        let at = base.saturating_add(u64::try_from(index).unwrap_or(u64::MAX));
        let size32 = u32::from_be_bytes([
            data[index],
            data[index + 1],
            data[index + 2],
            data[index + 3],
        ]);
        let kind = [
            data[index + 4],
            data[index + 5],
            data[index + 6],
            data[index + 7],
        ];
        let mut header = 8usize;
        let size = if size32 == 1 {
            if index + 16 > data.len() {
                return Err(Error::Truncated {
                    at,
                    needed: at.saturating_add(16),
                });
            }
            header = 16;
            u64::from_be_bytes([
                data[index + 8],
                data[index + 9],
                data[index + 10],
                data[index + 11],
                data[index + 12],
                data[index + 13],
                data[index + 14],
                data[index + 15],
            ])
        } else if size32 == 0 {
            u64::try_from(data.len() - index).unwrap_or(u64::MAX)
        } else {
            u64::from(size32)
        };
        if kind == *b"uuid" {
            header = header.saturating_add(16);
        }
        let header_len = u64::try_from(header).unwrap_or(u64::MAX);
        if size < header_len {
            return Err(Error::HostileSize { at, claimed: size });
        }
        let Some(end) = u64::try_from(index).unwrap_or(u64::MAX).checked_add(size) else {
            return Err(Error::HostileSize { at, claimed: size });
        };
        if end > u64::try_from(data.len()).unwrap_or(u64::MAX) {
            return Err(Error::HostileSize { at, claimed: size });
        }
        let payload_start = index.saturating_add(header);
        let end_usize = usize::try_from(end).unwrap_or(data.len());
        if payload_start > end_usize {
            return Err(Error::HostileSize { at, claimed: size });
        }
        let Some(payload) = data.get(payload_start..end_usize) else {
            return Err(Error::HostileSize { at, claimed: size });
        };
        out.push(Atom {
            kind,
            payload,
            payload_at: at.saturating_add(u64::try_from(header).unwrap_or(0)),
        });
        index = end_usize;
    }
    Ok(out)
}

pub struct FileHeader {
    pub kind: [u8; 4],
    pub start: u64,
    pub end: u64,
    pub payload_start: u64,
}

/// Read one box header at `at` without reading its payload.
pub fn file_header(
    source: &mut dyn crate::legacy::Source,
    at: u64,
    file_len: u64,
) -> Result<Option<FileHeader>, Error> {
    if at >= file_len {
        return Ok(None);
    }
    if at.saturating_add(8) > file_len {
        return Err(Error::Truncated {
            at,
            needed: at.saturating_add(8),
        });
    }
    let mut head = [0u8; 16];
    source.read_exact_at(at, &mut head[..8])?;
    let size32 = u32::from_be_bytes([head[0], head[1], head[2], head[3]]);
    let kind = [head[4], head[5], head[6], head[7]];
    let mut header_len = 8u64;
    let size = if size32 == 1 {
        if at.saturating_add(16) > file_len {
            return Err(Error::Truncated {
                at,
                needed: at.saturating_add(16),
            });
        }
        source.read_exact_at(at + 8, &mut head[8..16])?;
        header_len = 16;
        u64::from_be_bytes([
            head[8], head[9], head[10], head[11], head[12], head[13], head[14], head[15],
        ])
    } else if size32 == 0 {
        file_len.saturating_sub(at)
    } else {
        u64::from(size32)
    };
    if kind == *b"uuid" {
        header_len = header_len.saturating_add(16);
        if at.saturating_add(header_len) > file_len {
            return Err(Error::Truncated {
                at,
                needed: at.saturating_add(header_len),
            });
        }
    }
    if size < header_len {
        return Err(Error::HostileSize { at, claimed: size });
    }
    let Some(end) = at.checked_add(size) else {
        return Err(Error::HostileSize { at, claimed: size });
    };
    if end > file_len {
        return Err(Error::HostileSize { at, claimed: size });
    }
    Ok(Some(FileHeader {
        kind,
        start: at,
        end,
        payload_start: at.saturating_add(header_len),
    }))
}

pub fn load_payload(
    source: &mut dyn crate::legacy::Source,
    header: &FileHeader,
    cap: u64,
) -> Result<Vec<u8>, Error> {
    let len = header.end.saturating_sub(header.payload_start);
    if len > cap {
        return Err(Error::Limit {
            what: "metadata",
            value: len,
        });
    }
    let n = usize::try_from(len).map_err(|_| Error::Limit {
        what: "metadata",
        value: len,
    })?;
    let mut buf = vec![0u8; n];
    if n > 0 {
        source.read_exact_at(header.payload_start, &mut buf)?;
    }
    Ok(buf)
}

pub fn enter(depth: u32) -> Result<u32, Error> {
    let next = depth.saturating_add(1);
    if next > crate::limits::MAX_DEPTH {
        Err(Error::Nesting { depth: next })
    } else {
        Ok(next)
    }
}

#[cfg(test)]
mod tests {
    use super::atoms;
    use crate::legacy::Error;

    #[test]
    fn largesize_and_a_size_past_the_parent_are_distinct() {
        let mut big = vec![0, 0, 0, 1, b'f', b'r', b'e', b'e'];
        big.extend_from_slice(&24u64.to_be_bytes());
        big.extend_from_slice(&[0u8; 8]);
        let parsed = atoms(&big, 0, &mut 0).expect("largesize");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].payload.len(), 8);

        let hostile = [0, 0, 0, 32, b'f', b'r', b'e', b'e'];
        let err = atoms(&hostile, 10, &mut 0).expect_err("past parent");
        assert_eq!(
            err,
            Error::HostileSize {
                at: 10,
                claimed: 32
            }
        );
    }

    #[test]
    fn a_zero_size_box_fills_the_parent() {
        let bytes = [0, 0, 0, 0, b'm', b'd', b'a', b't', 1, 2, 3, 4];
        let parsed = atoms(&bytes, 0, &mut 0).expect("to end");
        assert_eq!(parsed[0].payload, &[1, 2, 3, 4]);
    }

    #[test]
    fn a_uuid_box_skips_the_user_type() {
        let mut bytes = vec![0, 0, 0, 32, b'u', b'u', b'i', b'd'];
        bytes.extend_from_slice(&[0x11; 16]);
        bytes.extend_from_slice(b"payload!");
        bytes.extend_from_slice(&[0, 0, 0, 8, b'f', b'r', b'e', b'e']);
        let parsed = atoms(&bytes, 0, &mut 0).expect("uuid");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].kind, *b"uuid");
        assert_eq!(parsed[0].payload, b"payload!");
        assert_eq!(parsed[1].kind, *b"free");
    }
}
