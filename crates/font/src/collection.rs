//! A TrueType or OpenType collection (`.ttc`).
//!
//! The header is `ttcf`, then a face count and an offset to each face's
//! table directory. Those offsets are from the start of the file.

use crate::{FontError, u16_at, u32_at};

/// Byte offset of face `index` in a collection.
pub(crate) fn face_offset(bytes: &[u8], index: u32) -> Result<usize, FontError> {
    if bytes.len() < 12 {
        return Err(FontError::Truncated);
    }
    if &bytes[..4] != b"ttcf" {
        return Err(FontError::BadTable);
    }
    let major = u16_at(bytes, 4)?;
    if major != 1 && major != 2 {
        return Err(FontError::Unsupported);
    }
    let count = u32_at(bytes, 8)?;
    if count == 0 {
        return Err(FontError::EmptyTable);
    }
    if index >= count {
        return Err(FontError::CollectionIndex { index, count });
    }
    let faces = usize::try_from(count).map_err(|_| FontError::BadTable)?;
    let array = faces.checked_mul(4).ok_or(FontError::BadTable)?;
    let array_end = array.checked_add(12).ok_or(FontError::BadTable)?;
    if bytes.len() < array_end {
        return Err(FontError::Truncated);
    }
    if major == 2 {
        let signed = array_end.checked_add(12).ok_or(FontError::BadTable)?;
        if bytes.len() < signed {
            return Err(FontError::Truncated);
        }
    }
    let at = usize::try_from(index)
        .ok()
        .and_then(|face| face.checked_mul(4))
        .and_then(|face| face.checked_add(12))
        .ok_or(FontError::BadTable)?;
    let offset = u32_at(bytes, at)?;
    if offset == 0 {
        return Err(FontError::BadTable);
    }
    let offset = usize::try_from(offset).map_err(|_| FontError::BadTable)?;
    if bytes.get(offset..).is_none() {
        return Err(FontError::Truncated);
    }
    Ok(offset)
}
