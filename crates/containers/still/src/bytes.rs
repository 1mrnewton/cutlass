//! Checked reads over a buffer that is already in memory.
//!
//! The buffer is the file, and it is at most the file cap. A size field is
//! compared with what remains before any further slice is taken.

use crate::fail::{Fail, as_u64, hostile, truncated};

pub(crate) fn at(data: &[u8], index: usize, len: usize) -> Result<&[u8], Fail> {
    let Some(end) = index.checked_add(len) else {
        return Err(hostile(
            as_u64(len),
            as_u64(data.len().saturating_sub(index)),
        ));
    };
    if end > data.len() {
        return Err(truncated(as_u64(end), as_u64(data.len())));
    }
    Ok(&data[index..end])
}

pub(crate) fn u16_le(data: &[u8], index: usize) -> Result<u16, Fail> {
    let bytes = at(data, index, 2)?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

pub(crate) fn u16_be(data: &[u8], index: usize) -> Result<u16, Fail> {
    let bytes = at(data, index, 2)?;
    Ok(u16::from_be_bytes([bytes[0], bytes[1]]))
}

pub(crate) fn u32_le(data: &[u8], index: usize) -> Result<u32, Fail> {
    let bytes = at(data, index, 4)?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

pub(crate) fn u32_be(data: &[u8], index: usize) -> Result<u32, Fail> {
    let bytes = at(data, index, 4)?;
    Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

pub(crate) fn u24_le(data: &[u8], index: usize) -> Result<u32, Fail> {
    let bytes = at(data, index, 3)?;
    Ok(u32::from(bytes[0]) | (u32::from(bytes[1]) << 8) | (u32::from(bytes[2]) << 16))
}
