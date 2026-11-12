//! Big-endian fields read out of a slice that was already bounds-checked.

/// A big-endian `u32` at `at`, or `None` when the slice is too short.
pub(crate) fn be_u32(bytes: &[u8], at: usize) -> Option<u32> {
    let end = at.checked_add(4)?;
    let field: [u8; 4] = bytes.get(at..end)?.try_into().ok()?;
    Some(u32::from_be_bytes(field))
}

/// A big-endian `u64` at `at`, or `None` when the slice is too short.
pub(crate) fn be_u64(bytes: &[u8], at: usize) -> Option<u64> {
    let end = at.checked_add(8)?;
    let field: [u8; 8] = bytes.get(at..end)?.try_into().ok()?;
    Some(u64::from_be_bytes(field))
}

/// A big-endian `i64` at `at`.
pub(crate) fn be_i64(bytes: &[u8], at: usize) -> Option<i64> {
    be_u64(bytes, at).map(|value| i64::from_be_bytes(value.to_be_bytes()))
}

/// A big-endian `i32` at `at`.
pub(crate) fn be_i32(bytes: &[u8], at: usize) -> Option<i32> {
    be_u32(bytes, at).map(|value| i32::from_be_bytes(value.to_be_bytes()))
}

/// Four bytes at `at`.
pub(crate) fn fourcc(bytes: &[u8], at: usize) -> Option<[u8; 4]> {
    let end = at.checked_add(4)?;
    bytes.get(at..end)?.try_into().ok()
}

/// A four-character code as text, or hex when it is not printable.
pub(crate) fn fourcc_text(tag: [u8; 4]) -> String {
    if tag
        .iter()
        .all(|byte| byte.is_ascii_graphic() || *byte == b' ')
    {
        tag.iter().map(|byte| char::from(*byte)).collect()
    } else {
        format!("{:02x}{:02x}{:02x}{:02x}", tag[0], tag[1], tag[2], tag[3])
    }
}

/// `usize` for a length that already fits in memory.
pub(crate) fn len_u64(len: usize) -> u64 {
    u64::try_from(len).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::{be_i32, be_i64, be_u32, be_u64, fourcc, fourcc_text};

    #[test]
    fn fields_are_big_endian_and_a_short_slice_is_none() {
        let bytes = [0, 0, 1, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        assert_eq!(be_u32(&bytes, 0), Some(256));
        assert_eq!(be_i32(&bytes, 4), Some(-1));
        assert_eq!(be_u64(&bytes, 0), Some(0x0000_0100_FFFF_FFFF));
        assert_eq!(be_i64(&bytes, 1), Some(0x0001_00FF_FFFF_FFFF));
        assert_eq!(be_u32(&bytes, 6), None);
        assert_eq!(be_u64(&bytes, 2), None);
        assert_eq!(be_u32(&bytes, usize::MAX), None);
        assert_eq!(fourcc(b"caff", 0), Some(*b"caff"));
        assert_eq!(fourcc_text(*b"aac "), "aac ");
        assert_eq!(fourcc_text([b'm', b's', 0, 0x11]), "6d730011");
    }
}
