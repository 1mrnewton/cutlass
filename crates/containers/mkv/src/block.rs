//! A SimpleBlock or Block, including lacing.
//!
//! The first laced frame has the block time. Each later frame is one default
//! duration later. That step is applied by the caller. This module only splits
//! the bytes.

use crate::Error;
use crate::ebml::{Cursor, Limits};

pub(crate) struct Block<'a> {
    pub(crate) track: u64,
    pub(crate) relative: i16,
    /// SimpleBlock keyframe bit. A Block in a group does not use it.
    pub(crate) keyframe_flag: bool,
    pub(crate) frames: Vec<&'a [u8]>,
}

pub(crate) fn parse<'a>(data: &'a [u8], at: u64, limits: Limits) -> Result<Block<'a>, Error> {
    let mut cursor = Cursor::new(data, at);
    let track = cursor.data_vint(limits.max_size)?;
    if track.value == 0 {
        return Err(Error::Inconsistent("a block names an unknown track"));
    }
    let relative = read_i16(&mut cursor)?;
    let flags = cursor.read(1)?[0];
    let rest = cursor.remaining();
    let frames = split(rest, flags >> 1 & 0x03, limits)?;
    Ok(Block {
        track: track.value,
        relative,
        keyframe_flag: flags & 0x80 != 0,
        frames,
    })
}

fn read_i16(cursor: &mut Cursor<'_>) -> Result<i16, Error> {
    let bytes = cursor.read(2)?;
    let bytes: [u8; 2] = match bytes.try_into() {
        Ok(bytes) => bytes,
        Err(_) => return Err(Error::Inconsistent("a block is too short")),
    };
    Ok(i16::from_be_bytes(bytes))
}

fn split(data: &[u8], lacing: u8, limits: Limits) -> Result<Vec<&[u8]>, Error> {
    match lacing {
        0 => Ok(vec![data]),
        1 => xiph(data),
        2 => fixed(data),
        3 => ebml(data, limits),
        _ => Err(Error::Inconsistent("the lace type is invalid")),
    }
}

fn xiph(data: &[u8]) -> Result<Vec<&[u8]>, Error> {
    let (count, mut pos) = frame_count(data)?;
    let mut sizes = Vec::with_capacity(count - 1);
    for _ in 1..count {
        let mut size = 0u64;
        loop {
            if pos >= data.len() {
                return Err(Error::Inconsistent("the lace sizes do not add up"));
            }
            let byte = data[pos];
            pos += 1;
            size = size.checked_add(u64::from(byte)).ok_or(Error::TooLarge)?;
            if byte != 0xFF {
                break;
            }
        }
        sizes.push(size);
    }
    take_frames(&data[pos..], &sizes)
}

fn fixed(data: &[u8]) -> Result<Vec<&[u8]>, Error> {
    let (count, pos) = frame_count(data)?;
    let body = &data[pos..];
    if !body.len().is_multiple_of(count) {
        return Err(Error::Inconsistent("the lace sizes do not add up"));
    }
    let size = body.len() / count;
    let mut frames = Vec::with_capacity(count);
    for index in 0..count {
        let start = index * size;
        frames.push(&body[start..start + size]);
    }
    Ok(frames)
}

fn ebml(data: &[u8], limits: Limits) -> Result<Vec<&[u8]>, Error> {
    let (count, pos) = frame_count(data)?;
    let mut cursor = Cursor::new(&data[pos..], 0);
    if count == 1 {
        return Ok(vec![cursor.remaining()]);
    }
    let first = cursor.data_vint(limits.max_size)?;
    let mut sizes = Vec::with_capacity(count - 1);
    sizes.push(first.value);
    let mut prev = first.value;
    for _ in 0..count - 2 {
        let raw = cursor.data_vint(limits.max_size)?;
        let diff = signed_vint(raw.value, raw.width)?;
        prev = add_size(prev, diff)?;
        sizes.push(prev);
    }
    take_frames(cursor.remaining(), &sizes)
}

fn frame_count(data: &[u8]) -> Result<(usize, usize), Error> {
    let Some(count) = data.first() else {
        return Err(Error::Inconsistent("a block is too short"));
    };
    Ok((usize::from(*count) + 1, 1))
}

fn take_frames<'a>(data: &'a [u8], sizes: &[u64]) -> Result<Vec<&'a [u8]>, Error> {
    let mut pos = 0usize;
    let mut frames = Vec::with_capacity(sizes.len() + 1);
    for size in sizes {
        let size = usize::try_from(*size).map_err(|_| Error::TooLarge)?;
        let end = pos.checked_add(size).ok_or(Error::TooLarge)?;
        if end > data.len() {
            return Err(Error::Inconsistent("the lace sizes do not add up"));
        }
        frames.push(&data[pos..end]);
        pos = end;
    }
    frames.push(&data[pos..]);
    Ok(frames)
}

/// EBML lace stores a signed difference. The bias for a `width`-byte vint is
/// `2^(7*width - 1) - 1`, so a stored zero is the midpoint.
fn signed_vint(value: u64, width: usize) -> Result<i64, Error> {
    let bits = width.checked_mul(7).ok_or(Error::TooLarge)?;
    if bits == 0 || bits >= 63 {
        return Err(Error::TooLarge);
    }
    let bias = (1i64 << (bits - 1)) - 1;
    let value = i64::try_from(value).map_err(|_| Error::TooLarge)?;
    value.checked_sub(bias).ok_or(Error::TooLarge)
}

fn add_size(prev: u64, diff: i64) -> Result<u64, Error> {
    if diff >= 0 {
        let diff = u64::try_from(diff).map_err(|_| Error::TooLarge)?;
        return prev.checked_add(diff).ok_or(Error::TooLarge);
    }
    let magnitude = diff.checked_neg().ok_or(Error::TooLarge)?;
    let magnitude = u64::try_from(magnitude).map_err(|_| Error::TooLarge)?;
    prev.checked_sub(magnitude)
        .ok_or(Error::Inconsistent("the lace sizes do not add up"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_one_byte_signed_vint_is_biased_by_63() {
        assert_eq!(signed_vint(63, 1).expect("zero"), 0);
        assert_eq!(signed_vint(64, 1).expect("plus"), 1);
        assert_eq!(signed_vint(62, 1).expect("minus"), -1);
    }
}
