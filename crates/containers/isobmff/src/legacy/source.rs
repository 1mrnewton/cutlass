use super::Error;

/// Random access to a container file.
///
/// Reads are absolute. A demuxer seeks by offset instead of scanning a payload.
pub trait Source {
    fn len(&self) -> u64;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn read_exact_at(&mut self, at: u64, buf: &mut [u8]) -> Result<(), Error>;
}

/// An in-memory file.
pub struct SliceSource<'a> {
    data: &'a [u8],
}

impl<'a> SliceSource<'a> {
    #[must_use]
    pub const fn new(data: &'a [u8]) -> Self {
        Self { data }
    }
}

impl Source for SliceSource<'_> {
    fn len(&self) -> u64 {
        u64::try_from(self.data.len()).unwrap_or(u64::MAX)
    }

    fn read_exact_at(&mut self, at: u64, buf: &mut [u8]) -> Result<(), Error> {
        let start = usize::try_from(at).map_err(|_| Error::HostileSize { at, claimed: at })?;
        let Some(end) = start.checked_add(buf.len()) else {
            return Err(Error::HostileSize {
                at,
                claimed: u64::MAX,
            });
        };
        let Some(bytes) = self.data.get(start..end) else {
            return Err(Error::Truncated {
                at,
                needed: at.saturating_add(u64::try_from(buf.len()).unwrap_or(u64::MAX)),
            });
        };
        buf.copy_from_slice(bytes);
        Ok(())
    }
}

/// A file on disk. The read position is independent of other readers.
pub struct FileSource {
    file: std::fs::File,
    len: u64,
}

impl FileSource {
    pub fn open(path: &std::path::Path) -> Result<Self, Error> {
        let file = std::fs::File::open(path).map_err(|err| Error::Io {
            at: 0,
            kind: err.kind(),
        })?;
        let len = file.metadata().map_err(|err| Error::Io {
            at: 0,
            kind: err.kind(),
        })?;
        Ok(Self {
            file,
            len: len.len(),
        })
    }
}

impl Source for FileSource {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_exact_at(&mut self, at: u64, buf: &mut [u8]) -> Result<(), Error> {
        use std::io::{Read, Seek, SeekFrom};
        self.file
            .seek(SeekFrom::Start(at))
            .map_err(|err| Error::Io {
                at,
                kind: err.kind(),
            })?;
        self.file.read_exact(buf).map_err(|err| {
            if err.kind() == std::io::ErrorKind::UnexpectedEof {
                Error::Truncated {
                    at,
                    needed: at.saturating_add(u64::try_from(buf.len()).unwrap_or(u64::MAX)),
                }
            } else {
                Error::Io {
                    at,
                    kind: err.kind(),
                }
            }
        })
    }
}
