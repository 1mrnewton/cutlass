//! IFF chunks. A chunk is an id, a big-endian size, and that many data bytes.
//! An odd size is followed by a pad byte when the parent still has one.

use crate::Error;

pub(crate) struct Chunk<'a> {
    pub(crate) id: [u8; 4],
    pub(crate) data: &'a [u8],
    /// File offset of `data`.
    data_at: u64,
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
        let size = u32::from_be_bytes(self.read_array::<4>()?);
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

pub(crate) fn children<'a>(file: &'a [u8], chunk: &Chunk<'a>) -> Result<Cursor<'a>, Error> {
    if chunk.data.len() < 4 {
        return Err(Error::Inconsistent("the file has no form"));
    }
    let start = usize::try_from(chunk.data_at).map_err(|_| Error::TooLarge)?;
    let body = start.checked_add(4).ok_or(Error::TooLarge)?;
    let end = start.checked_add(chunk.data.len()).ok_or(Error::TooLarge)?;
    if end > file.len() {
        return Err(Error::Truncated {
            at: chunk.data_at,
            need: u64::try_from(chunk.data.len()).map_err(|_| Error::TooLarge)?,
        });
    }
    Ok(Cursor {
        data: file,
        pos: body,
        end,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_odd_chunk_skips_its_pad() {
        let bytes = [
            b'N', b'A', b'M', b'E', 0, 0, 0, 1, b'A', 0, b'C', b'O', b'M', b'M', 0, 0, 0, 0,
        ];
        let mut cursor = Cursor::new(&bytes);
        let name = cursor.next_chunk().expect("name").expect("chunk");
        assert_eq!(name.data, b"A");
        let comm = cursor.next_chunk().expect("comm").expect("chunk");
        assert_eq!(&comm.id, b"COMM");
        assert!(cursor.next_chunk().expect("end").is_none());
    }
}
