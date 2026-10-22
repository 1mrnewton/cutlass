//! One Ogg page. RFC 3533. The checksum is not checked.

use crate::Error;

const HEADER: usize = 27;
const CAPTURE: &[u8] = b"OggS";

pub(crate) struct Page<'a> {
    pub(crate) flags: u8,
    pub(crate) granule: i64,
    pub(crate) serial: u32,
    pub(crate) sequence: u32,
    pub(crate) table: &'a [u8],
    pub(crate) payload: &'a [u8],
    pub(crate) payload_at: usize,
}

pub(crate) fn read_page(data: &[u8], at: usize) -> Result<(Page<'_>, usize), Error> {
    let rest = data.len().saturating_sub(at);
    if rest < 4 {
        return Err(truncated(at, HEADER));
    }
    if &data[at..at + 4] != CAPTURE {
        return Err(if at == 0 {
            Error::Form
        } else {
            Error::Inconsistent("a page is not ogg")
        });
    }
    if rest < HEADER {
        return Err(truncated(at, HEADER));
    }
    let head = &data[at..at + HEADER];
    if head[4] != 0 {
        return Err(Error::Inconsistent("the ogg version is not supported"));
    }
    let nseg = usize::from(head[26]);
    let table_at = at + HEADER;
    let table_end = table_at.checked_add(nseg).ok_or(Error::TooLarge)?;
    if table_end > data.len() {
        return Err(truncated(table_at, nseg));
    }
    let table = &data[table_at..table_end];
    let mut payload_len = 0usize;
    for &size in table {
        payload_len = payload_len
            .checked_add(usize::from(size))
            .ok_or(Error::TooLarge)?;
    }
    let payload_end = table_end.checked_add(payload_len).ok_or(Error::TooLarge)?;
    if payload_end > data.len() {
        return Err(truncated(table_end, payload_len));
    }
    let page = Page {
        flags: head[5],
        granule: le_i64(&head[6..14]),
        serial: le_u32(&head[14..18]),
        sequence: le_u32(&head[18..22]),
        table,
        payload: &data[table_end..payload_end],
        payload_at: table_end,
    };
    Ok((page, payload_end))
}

pub(crate) fn truncated(at: usize, need: usize) -> Error {
    Error::Truncated {
        at: u64::try_from(at).unwrap_or(u64::MAX),
        need: u64::try_from(need).unwrap_or(u64::MAX),
    }
}

fn le_u32(bytes: &[u8]) -> u32 {
    let mut buf = [0; 4];
    buf.copy_from_slice(bytes);
    u32::from_le_bytes(buf)
}

fn le_i64(bytes: &[u8]) -> i64 {
    let mut buf = [0; 8];
    buf.copy_from_slice(bytes);
    i64::from_le_bytes(buf)
}
