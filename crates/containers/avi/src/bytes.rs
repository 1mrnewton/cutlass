//! Little-endian fields read from a slice that is already in memory.

/// A slice ended before a field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Short {
    pub need: usize,
    pub have: usize,
}

pub(crate) fn u16_at(bytes: &[u8], at: usize) -> Result<u16, Short> {
    array_at::<2>(bytes, at).map(u16::from_le_bytes)
}

pub(crate) fn u32_at(bytes: &[u8], at: usize) -> Result<u32, Short> {
    array_at::<4>(bytes, at).map(u32::from_le_bytes)
}

pub(crate) fn i32_at(bytes: &[u8], at: usize) -> Result<i32, Short> {
    array_at::<4>(bytes, at).map(i32::from_le_bytes)
}

pub(crate) fn u64_at(bytes: &[u8], at: usize) -> Result<u64, Short> {
    array_at::<8>(bytes, at).map(u64::from_le_bytes)
}

pub(crate) fn id_at(bytes: &[u8], at: usize) -> Result<[u8; 4], Short> {
    array_at::<4>(bytes, at)
}

pub(crate) fn array_at<const N: usize>(bytes: &[u8], at: usize) -> Result<[u8; N], Short> {
    let end = at.checked_add(N).ok_or(Short {
        need: usize::MAX,
        have: bytes.len(),
    })?;
    let slice = bytes.get(at..end).ok_or(Short {
        need: end,
        have: bytes.len(),
    })?;
    let mut out = [0; N];
    out.copy_from_slice(slice);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{Short, u16_at, u32_at, u64_at};

    #[test]
    fn a_field_past_the_slice_is_short_and_not_a_panic() {
        assert_eq!(u16_at(&[1, 2], 0), Ok(0x0201));
        assert_eq!(u32_at(&[1, 2, 3], 0), Err(Short { need: 4, have: 3 }));
        assert!(u64_at(&[0; 8], usize::MAX).is_err());
    }
}
