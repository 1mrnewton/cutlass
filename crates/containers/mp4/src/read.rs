//! Bytes and ISO boxes.

use crate::{Error, Fourcc};

pub(crate) struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
    base: u64,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(data: &'a [u8], base: u64) -> Self {
        Self { data, pos: 0, base }
    }

    pub(crate) fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    pub(crate) fn rest(&self) -> &'a [u8] {
        &self.data[self.pos..]
    }

    pub(crate) fn file_pos(&self) -> Result<u64, Error> {
        self.base
            .checked_add(u64_from_usize(self.pos)?)
            .ok_or(Error::TooLarge)
    }

    pub(crate) fn u8(&mut self) -> Result<u8, Error> {
        let bytes = self.take(1)?;
        Ok(bytes[0])
    }

    pub(crate) fn u16(&mut self) -> Result<u16, Error> {
        let bytes = self.take(2)?;
        Ok(u16::from_be_bytes([bytes[0], bytes[1]]))
    }

    pub(crate) fn u32(&mut self) -> Result<u32, Error> {
        let bytes = self.take(4)?;
        Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    pub(crate) fn u64(&mut self) -> Result<u64, Error> {
        let bytes = self.take(8)?;
        let mut buf = [0u8; 8];
        buf.copy_from_slice(bytes);
        Ok(u64::from_be_bytes(buf))
    }

    pub(crate) fn i32(&mut self) -> Result<i32, Error> {
        let bytes = self.take(4)?;
        Ok(i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    pub(crate) fn i64(&mut self) -> Result<i64, Error> {
        let bytes = self.take(8)?;
        let mut buf = [0u8; 8];
        buf.copy_from_slice(bytes);
        Ok(i64::from_be_bytes(buf))
    }

    pub(crate) fn fourcc(&mut self) -> Result<[u8; 4], Error> {
        let bytes = self.take(4)?;
        Ok([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    pub(crate) fn skip(&mut self, len: usize) -> Result<(), Error> {
        self.take(len)?;
        Ok(())
    }

    pub(crate) fn take(&mut self, len: usize) -> Result<&'a [u8], Error> {
        let Some(end) = self.pos.checked_add(len) else {
            return Err(Error::TooLarge);
        };
        if end > self.data.len() {
            return Err(Error::Truncated {
                at: self.file_pos()?,
                need: u64_from_usize(len)?,
            });
        }
        let start = self.pos;
        self.pos = end;
        Ok(&self.data[start..end])
    }
}

pub(crate) struct Atom<'a> {
    pub(crate) kind: [u8; 4],
    pub(crate) payload: &'a [u8],
    start: u64,
    header: u64,
}

impl<'a> Atom<'a> {
    /// File offset of the first byte of this box.
    pub(crate) fn start(&self) -> u64 {
        self.start
    }

    pub(crate) fn payload_base(&self) -> Result<u64, Error> {
        self.start.checked_add(self.header).ok_or(Error::TooLarge)
    }
}

pub(crate) fn atoms(data: &[u8], base: u64) -> Result<Vec<Atom<'_>>, Error> {
    let mut reader = Reader::new(data, base);
    let mut found = Vec::new();
    // Fewer than eight bytes cannot start a box. ISO/IEC 14496-12 leaves that
    // tail as padding. QuickTime puts four zero bytes after a sample entry's
    // children, and treating them as a truncated box refused the file.
    while reader.remaining() >= 8 {
        let start = reader.file_pos()?;
        let size32 = reader.u32()?;
        let kind = reader.fourcc()?;
        let (header, size) = if size32 == 1 {
            (16, reader.u64()?)
        } else if size32 == 0 {
            let payload = u64_from_usize(reader.rest().len())?;
            (8, 8u64.checked_add(payload).ok_or(Error::TooLarge)?)
        } else {
            (8, u64::from(size32))
        };
        if size < header {
            return Err(Error::BadBoxSize {
                kind: Fourcc::new(kind),
                size,
            });
        }
        let payload_len = usize_from_u64(size - header)?;
        let payload = reader.take(payload_len)?;
        found.push(Atom {
            kind,
            payload,
            start,
            header,
        });
    }
    Ok(found)
}

pub(crate) fn only<'payload, 'list>(
    atoms: &'list [Atom<'payload>],
    kind: [u8; 4],
) -> Result<Option<&'list Atom<'payload>>, Error> {
    let mut found = None;
    for atom in atoms {
        if atom.kind == kind {
            if found.is_some() {
                return Err(Error::DuplicateBox(Fourcc::new(kind)));
            }
            found = Some(atom);
        }
    }
    Ok(found)
}

pub(crate) fn required<'payload, 'list>(
    atoms: &'list [Atom<'payload>],
    kind: [u8; 4],
) -> Result<&'list Atom<'payload>, Error> {
    only(atoms, kind)?.ok_or(Error::MissingBox(Fourcc::new(kind)))
}

pub(crate) fn full_box<'a>(atom: &Atom<'a>) -> Result<(u8, u32, &'a [u8]), Error> {
    if atom.payload.len() < 4 {
        return Err(Error::Truncated {
            at: atom.payload_base()?,
            need: 4,
        });
    }
    let version = atom.payload[0];
    let flags = u32::from_be_bytes([0, atom.payload[1], atom.payload[2], atom.payload[3]]);
    Ok((version, flags, &atom.payload[4..]))
}

pub(crate) fn u64_from_usize(value: usize) -> Result<u64, Error> {
    u64::try_from(value).map_err(|_| Error::TooLarge)
}

pub(crate) fn usize_from_u64(value: u64) -> Result<usize, Error> {
    usize::try_from(value).map_err(|_| Error::TooLarge)
}
