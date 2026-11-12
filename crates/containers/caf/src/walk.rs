//! The chunk walk: file header, `desc`, then every chunk in file order.
//!
//! `desc` must be the first chunk. `data` may sit anywhere after it; with
//! a size of -1 it runs to the end of the file and must be last. A random
//! access input walks every chunk, seeking over `data`. A forward input
//! stops at `data`, so a packet table after the sound cannot be read there.
//!
//! A chunk whose declared size runs past the file is
//! [`OpenError::HostileSize`], with one tolerance: once `data` has been
//! found, a chunk that is not `desc`, `kuki`, `pakt` or `chan` and that
//! claims more bytes than remain, or a negative size, ends the walk. Some
//! of macOS's own system sounds end with such a chunk after a one-byte
//! slip in the chunk before it; libavformat and AudioToolbox both play
//! them. The offset where the walk stopped is reported, not hidden.

use std::io::{Read, Seek};

use cutlass_container::{Access, Input, OpenError};

use crate::bytes::{be_i64, be_u32, fourcc, len_u64};
use crate::chan::ChannelLayout;
use crate::desc::{DESC_BYTES, Description};
use crate::error::{hostile, malformed, truncated, unrecognized};
use crate::info::{self, InfoEntry};
use crate::limits::{MAX_CHAN_BYTES, MAX_CHUNKS, MAX_INFO_BYTES};
use crate::pakt::{PAKT_HEADER_BYTES, PacketTableHeader, Table, skip};

/// Bytes of a chunk header: type and `SInt64` size.
pub const CHUNK_HEADER_BYTES: u64 = 12;
/// Bytes of the file header: `caff`, version, flags.
pub const FILE_HEADER_BYTES: u64 = 8;

/// One chunk as it sits in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Chunk {
    pub kind: [u8; 4],
    /// Offset of the chunk header.
    pub offset: u64,
    /// Body bytes, or `None` for a `data` chunk of size -1.
    pub size: Option<u64>,
}

/// Where the sound bytes are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DataChunk {
    /// Offset of the first sound byte, after the edit count.
    pub start: u64,
    /// Sound bytes, or `None` when the size was -1 and the input's length is unknown.
    pub len: Option<u64>,
    pub edit_count: u32,
    pub open_ended: bool,
}

#[derive(Debug)]
pub(crate) struct Scanned {
    pub desc: Description,
    pub cookie: Option<Vec<u8>>,
    pub pakt: Option<PacketTableHeader>,
    pub table: Option<Table>,
    pub chan: Option<ChannelLayout>,
    pub info: Vec<InfoEntry>,
    pub data: DataChunk,
    pub chunks: Vec<Chunk>,
    pub stopped_at: Option<u64>,
}

struct Walk {
    desc: Description,
    cookie: Option<Vec<u8>>,
    pakt: Option<PacketTableHeader>,
    table: Option<Table>,
    chan: Option<ChannelLayout>,
    info: Vec<InfoEntry>,
    data: Option<DataChunk>,
    chunks: Vec<Chunk>,
    stopped_at: Option<u64>,
}

/// Reads a header field or a chunk body already checked against the file.
/// Bytes the file does not have are [`OpenError::Truncated`].
fn read_bytes<R: Read>(input: &mut Input<R>, len: u64, limit: u64) -> Result<Vec<u8>, OpenError> {
    if let Some(left) = input.remaining()
        && len > left
        && len <= limit
    {
        return Err(truncated(len, left));
    }
    input
        .read_limited(len, limit)
        .map_err(OpenError::from_header)
}

/// Walks the header chunks and leaves the input somewhere after them.
pub(crate) fn scan<R: Read + Seek>(input: &mut Input<R>) -> Result<Scanned, OpenError> {
    let limits = input.limits();
    let file = read_bytes(input, FILE_HEADER_BYTES, limits.max_header_bytes)?;
    if fourcc(&file, 0) != Some(*b"caff") {
        return Err(unrecognized());
    }
    if file.get(4..6) != Some(&[0, 1]) {
        return Err(OpenError::UnsupportedBrand {
            family: cutlass_codec::ContainerId::Caf,
        });
    }
    let head = read_bytes(input, CHUNK_HEADER_BYTES, limits.max_header_bytes)?;
    if fourcc(&head, 0) != Some(*b"desc") {
        return Err(malformed("desc is not the first chunk"));
    }
    let size = be_i64(&head, 4).ok_or_else(|| truncated(12, len_u64(head.len())))?;
    if size != 32 {
        return Err(malformed(format!("desc is {size} bytes, not 32")));
    }
    let body = read_bytes(input, DESC_BYTES, limits.max_header_bytes)?;
    let desc = Description::parse(&body, &limits)?;
    let mut walk = Walk {
        desc,
        cookie: None,
        pakt: None,
        table: None,
        chan: None,
        info: Vec::new(),
        data: None,
        chunks: vec![Chunk {
            kind: *b"desc",
            offset: FILE_HEADER_BYTES,
            size: Some(DESC_BYTES),
        }],
        stopped_at: None,
    };
    walk.chunks_after_desc(input)?;
    let Some(data) = walk.data else {
        return Err(malformed("the file has no data chunk"));
    };
    Ok(Scanned {
        desc: walk.desc,
        cookie: walk.cookie,
        pakt: walk.pakt,
        table: walk.table,
        chan: walk.chan,
        info: walk.info,
        data,
        chunks: walk.chunks,
        stopped_at: walk.stopped_at,
    })
}

impl Walk {
    fn chunks_after_desc<R: Read + Seek>(&mut self, input: &mut Input<R>) -> Result<(), OpenError> {
        let random = input.access() == Access::Random;
        let mut count = 1_u32;
        loop {
            let offset = input.position();
            let remaining = input.remaining();
            if remaining == Some(0) {
                return Ok(());
            }
            if remaining.is_some_and(|left| left < CHUNK_HEADER_BYTES) {
                if self.data.is_some() {
                    self.stopped_at = Some(offset);
                    return Ok(());
                }
                return Err(truncated(CHUNK_HEADER_BYTES, remaining.unwrap_or(0)));
            }
            count += 1;
            if count > MAX_CHUNKS {
                return Err(hostile(u64::from(count), u64::from(MAX_CHUNKS)));
            }
            let head = read_bytes(input, CHUNK_HEADER_BYTES, CHUNK_HEADER_BYTES)?;
            let kind = fourcc(&head, 0).ok_or_else(|| truncated(12, 0))?;
            let size = be_i64(&head, 4).ok_or_else(|| truncated(12, 0))?;
            let left = input.remaining();
            if kind == *b"data" {
                if self.data.is_some() {
                    return Err(malformed("the file has two data chunks"));
                }
                let stop = self.data_chunk(input, offset, size)?;
                if stop || !random {
                    return Ok(());
                }
                continue;
            }
            let essential = matches!(&kind, b"desc" | b"kuki" | b"pakt" | b"chan");
            let Ok(size) = u64::try_from(size) else {
                if self.data.is_some() && !essential {
                    self.stopped_at = Some(offset);
                    return Ok(());
                }
                return Err(malformed(format!(
                    "chunk {} has a negative size",
                    crate::bytes::fourcc_text(kind)
                )));
            };
            if let Some(left) = left
                && size > left
            {
                if self.data.is_some() && !essential {
                    self.stopped_at = Some(offset);
                    return Ok(());
                }
                return Err(hostile(size, left));
            }
            self.chunks.push(Chunk {
                kind,
                offset,
                size: Some(size),
            });
            let end = offset
                .checked_add(CHUNK_HEADER_BYTES)
                .and_then(|at| at.checked_add(size))
                .ok_or_else(|| hostile(size, u64::MAX))?;
            self.body(input, kind, size)?;
            if input.position() != end {
                if random {
                    input.seek(end).map_err(OpenError::from_header)?;
                } else {
                    let consumed = input.position().saturating_sub(offset + CHUNK_HEADER_BYTES);
                    skip(input, size.saturating_sub(consumed))?;
                }
            }
        }
    }

    /// Records `data`. Returns true when the walk must stop here.
    fn data_chunk<R: Read + Seek>(
        &mut self,
        input: &mut Input<R>,
        offset: u64,
        size: i64,
    ) -> Result<bool, OpenError> {
        let limits = input.limits();
        let edit = read_bytes(input, 4, limits.max_header_bytes)?;
        let edit_count = be_u32(&edit, 0).ok_or_else(|| truncated(4, 0))?;
        let start = input.position();
        if size == -1 {
            self.chunks.push(Chunk {
                kind: *b"data",
                offset,
                size: None,
            });
            self.data = Some(DataChunk {
                start,
                len: input.remaining(),
                edit_count,
                open_ended: true,
            });
            return Ok(true);
        }
        let size = u64::try_from(size)
            .map_err(|_| malformed(format!("data size {size} is negative and not -1")))?;
        if size < 4 {
            return Err(malformed("data is shorter than its edit count"));
        }
        let len = size - 4;
        if let Some(left) = input.remaining()
            && len > left
        {
            return Err(hostile(len, left));
        }
        self.chunks.push(Chunk {
            kind: *b"data",
            offset,
            size: Some(size),
        });
        self.data = Some(DataChunk {
            start,
            len: Some(len),
            edit_count,
            open_ended: false,
        });
        if input.access() == Access::Random {
            let end = start
                .checked_add(len)
                .ok_or_else(|| hostile(len, u64::MAX))?;
            input.seek(end).map_err(OpenError::from_header)?;
        }
        Ok(false)
    }

    fn body<R: Read + Seek>(
        &mut self,
        input: &mut Input<R>,
        kind: [u8; 4],
        size: u64,
    ) -> Result<(), OpenError> {
        let limits = input.limits();
        match &kind {
            b"desc" => Err(malformed("the file has two desc chunks")),
            b"kuki" => {
                if self.cookie.is_some() {
                    return Err(malformed("the file has two kuki chunks"));
                }
                self.cookie = Some(read_bytes(input, size, limits.max_extradata_bytes)?);
                Ok(())
            }
            b"chan" => {
                if self.chan.is_some() {
                    return Err(malformed("the file has two chan chunks"));
                }
                let body = read_bytes(input, size, MAX_CHAN_BYTES)?;
                self.chan = Some(ChannelLayout::parse(&body)?);
                Ok(())
            }
            b"info" if size <= MAX_INFO_BYTES => {
                let body = read_bytes(input, size, MAX_INFO_BYTES)?;
                self.info.extend(info::parse(&body));
                Ok(())
            }
            b"pakt" => self.packet_table(input, size),
            _ => Ok(()),
        }
    }

    fn packet_table<R: Read + Seek>(
        &mut self,
        input: &mut Input<R>,
        size: u64,
    ) -> Result<(), OpenError> {
        if self.pakt.is_some() {
            return Err(malformed("the file has two pakt chunks"));
        }
        if size < PAKT_HEADER_BYTES {
            return Err(truncated(PAKT_HEADER_BYTES, size));
        }
        let limits = input.limits();
        let head = read_bytes(input, PAKT_HEADER_BYTES, PAKT_HEADER_BYTES)?;
        let header = PacketTableHeader::parse(&head)?;
        if !self.desc.is_constant() {
            self.table = Some(Table::read(
                input,
                &header,
                size - PAKT_HEADER_BYTES,
                self.desc.bytes_per_packet,
                self.desc.frames_per_packet,
                &limits,
            )?);
        }
        self.pakt = Some(header);
        Ok(())
    }
}
