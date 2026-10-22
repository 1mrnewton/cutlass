//! RIFF chunks. A chunk is an id, a little-endian size, and that many data
//! bytes. An odd size is followed by a pad byte when the parent still has one.

use crate::Error;

pub(crate) struct Chunk<'a> {
    pub(crate) id: [u8; 4],
    pub(crate) data: &'a [u8],
    /// File offset of `data`.
    pub(crate) data_at: u64,
}

pub(crate) struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
    end: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            end: data.len(),
        }
    }

    /// `start` is the file offset of the first child. `end` is the parent's end.
    pub(crate) fn range(data: &'a [u8], start: usize, end: usize) -> Result<Self, Error> {
        if start > end || end > data.len() {
            let at = u64::try_from(start).map_err(|_| Error::TooLarge)?;
            let need = u64::try_from(end.saturating_sub(start)).map_err(|_| Error::TooLarge)?;
            return Err(Error::Truncated { at, need });
        }
        Ok(Self {
            data,
            pos: start,
            end,
        })
    }

    pub(crate) fn next_chunk(&mut self) -> Result<Option<Chunk<'a>>, Error> {
        if self.pos == self.end {
            return Ok(None);
        }
        if self.end - self.pos < 8 {
            return Err(Error::Truncated {
                at: self.locate()?,
                need: 8,
            });
        }
        let id = self.read_array::<4>()?;
        let size = u32::from_le_bytes(self.read_array::<4>()?);
        let len = usize::try_from(size).map_err(|_| Error::TooLarge)?;
        let data_at = self.locate()?;
        let Some(data_end) = self.pos.checked_add(len) else {
            return Err(Error::TooLarge);
        };
        if data_end > self.end {
            return Err(Error::Truncated {
                at: data_at,
                need: u64::from(size),
            });
        }
        let data = &self.data[self.pos..data_end];
        self.pos = data_end;
        if size % 2 == 1 && self.pos < self.end {
            self.pos += 1;
        }
        Ok(Some(Chunk { id, data, data_at }))
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        let Some(end) = self.pos.checked_add(N) else {
            return Err(Error::TooLarge);
        };
        if end > self.end {
            return Err(Error::Truncated {
                at: self.locate()?,
                need: u64::try_from(N).map_err(|_| Error::TooLarge)?,
            });
        }
        let bytes = &self.data[self.pos..end];
        self.pos = end;
        match bytes.try_into() {
            Ok(bytes) => Ok(bytes),
            Err(_) => Err(Error::Truncated {
                at: self.locate()?,
                need: u64::try_from(N).map_err(|_| Error::TooLarge)?,
            }),
        }
    }

    fn locate(&self) -> Result<u64, Error> {
        u64::try_from(self.pos).map_err(|_| Error::TooLarge)
    }
}

pub(crate) fn list_parts(chunk: &Chunk<'_>) -> Result<([u8; 4], u64), Error> {
    if chunk.data.len() < 4 {
        return Err(Error::Inconsistent("the list has no type"));
    }
    let kind: [u8; 4] = match chunk.data[..4].try_into() {
        Ok(kind) => kind,
        Err(_) => return Err(Error::Inconsistent("the list has no type")),
    };
    let children_at = chunk.data_at.checked_add(4).ok_or(Error::TooLarge)?;
    Ok((kind, children_at))
}

pub(crate) fn children<'a>(file: &'a [u8], chunk: &Chunk<'a>) -> Result<Cursor<'a>, Error> {
    let (_, children_at) = list_parts(chunk)?;
    let start = usize::try_from(children_at).map_err(|_| Error::TooLarge)?;
    let end = start
        .checked_add(chunk.data.len() - 4)
        .ok_or(Error::TooLarge)?;
    Cursor::range(file, start, end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_odd_chunk_skips_its_pad() {
        let bytes = [
            b'J', b'U', b'N', b'K', 1, 0, 0, 0, 0xAB, 0x00, b'a', b'v', b'i', b'h', 0, 0, 0, 0,
        ];
        let mut cursor = Cursor::new(&bytes);
        let junk = cursor.next_chunk().expect("junk").expect("chunk");
        assert_eq!(junk.data, &[0xAB]);
        let avih = cursor.next_chunk().expect("avih").expect("chunk");
        assert_eq!(&avih.id, b"avih");
        assert!(cursor.next_chunk().expect("end").is_none());
    }
}
