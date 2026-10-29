//! Little-endian reads that fail instead of panicking.

use cutlass_container::OpenError;

pub(crate) fn u16_at(bytes: &[u8], at: usize) -> Result<u16, OpenError> {
    let slice = slice_at(bytes, at, 2)?;
    let mut buf = [0; 2];
    buf.copy_from_slice(slice);
    Ok(u16::from_le_bytes(buf))
}

pub(crate) fn u32_at(bytes: &[u8], at: usize) -> Result<u32, OpenError> {
    let slice = slice_at(bytes, at, 4)?;
    let mut buf = [0; 4];
    buf.copy_from_slice(slice);
    Ok(u32::from_le_bytes(buf))
}

pub(crate) fn u64_at(bytes: &[u8], at: usize) -> Result<u64, OpenError> {
    let slice = slice_at(bytes, at, 8)?;
    let mut buf = [0; 8];
    buf.copy_from_slice(slice);
    Ok(u64::from_le_bytes(buf))
}

pub(crate) fn id_at(bytes: &[u8], at: usize) -> Result<[u8; 4], OpenError> {
    let slice = slice_at(bytes, at, 4)?;
    let mut buf = [0; 4];
    buf.copy_from_slice(slice);
    Ok(buf)
}

pub(crate) fn guid_at(bytes: &[u8], at: usize) -> Result<[u8; 16], OpenError> {
    let slice = slice_at(bytes, at, 16)?;
    let mut buf = [0; 16];
    buf.copy_from_slice(slice);
    Ok(buf)
}

fn slice_at(bytes: &[u8], at: usize, len: usize) -> Result<&[u8], OpenError> {
    let Some(end) = at.checked_add(len) else {
        return Err(OpenError::Truncated {
            expected: u64::try_from(len).unwrap_or(u64::MAX),
            got: 0,
        });
    };
    bytes.get(at..end).ok_or(OpenError::Truncated {
        expected: u64::try_from(end).unwrap_or(u64::MAX),
        got: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
    })
}
