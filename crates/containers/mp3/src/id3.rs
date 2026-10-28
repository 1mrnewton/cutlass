//! ID3v2, ID3v1, and APEv2 envelopes.
//!
//! The tag body is skipped. Title, artist, and attached pictures are not
//! part of the stream contract, so they are not decoded.
//!
//! ID3v2 size is the synchsafe integer in the 10-byte header (id3.org
//! ID3v2.4.0 structure). A footer flag adds another 10 bytes. ID3v1 is the
//! 128-byte `TAG` record at the end of the file.

use crate::limits::{MAX_APE_BYTES, MAX_ID3_BYTES};
use cutlass_container::ContainerError;

/// Byte length of a leading ID3v2 tag, including its header and optional footer.
///
/// `None` when `header` is not an ID3v2 header. A size above
/// [`MAX_ID3_BYTES`](crate::limits::MAX_ID3_BYTES) is an error, not a length.
pub fn id3v2_tag_bytes(header: &[u8]) -> Result<Option<u64>, ContainerError> {
    if header.len() < 10 || !is_id3v2(header) {
        return Ok(None);
    }
    if header[6] & 0x80 != 0
        || header[7] & 0x80 != 0
        || header[8] & 0x80 != 0
        || header[9] & 0x80 != 0
    {
        return Err(ContainerError::Malformed { what: "id3 size" });
    }
    let size = (u32::from(header[6]) << 21)
        | (u32::from(header[7]) << 14)
        | (u32::from(header[8]) << 7)
        | u32::from(header[9]);
    let mut total = u64::from(size).saturating_add(10);
    // ID3v2.4 footer flag. The same bit is tested for earlier revisions so a
    // set bit cannot hide ten extra bytes.
    if header[5] & 0x10 != 0 {
        total = total.saturating_add(10);
    }
    if total > MAX_ID3_BYTES {
        return Err(ContainerError::Limit { what: "id3" });
    }
    Ok(Some(total))
}

#[must_use]
pub fn is_id3v2(header: &[u8]) -> bool {
    header.len() >= 10
        && header[0] == b'I'
        && header[1] == b'D'
        && header[2] == b'3'
        && header[3] != 0xFF
        && header[4] != 0xFF
        && header[6] & 0x80 == 0
        && header[7] & 0x80 == 0
        && header[8] & 0x80 == 0
        && header[9] & 0x80 == 0
}

#[must_use]
pub fn is_id3v1(tail: &[u8]) -> bool {
    tail.len() == 128 && tail[0] == b'T' && tail[1] == b'A' && tail[2] == b'G'
}

/// Bytes occupied by an APEv2 tag that ends at `footer`.
///
/// The span includes the footer and, when the header flag is set, the header.
/// `Ok(None)` when those 32 bytes are not a footer.
pub fn ape_tag_bytes(footer: &[u8]) -> Result<Option<u64>, ContainerError> {
    if footer.len() < 32 || &footer[..8] != b"APETAGEX" {
        return Ok(None);
    }
    let version = u32::from_le_bytes([footer[8], footer[9], footer[10], footer[11]]);
    if version != 1000 && version != 2000 {
        return Ok(None);
    }
    let size = u32::from_le_bytes([footer[12], footer[13], footer[14], footer[15]]);
    let flags = u32::from_le_bytes([footer[20], footer[21], footer[22], footer[23]]);
    let mut total = u64::from(size);
    if flags & 0x8000_0000 != 0 {
        total = total.saturating_add(32);
    }
    if total < 32 {
        return Err(ContainerError::Malformed { what: "ape" });
    }
    if total > MAX_APE_BYTES {
        return Err(ContainerError::Limit { what: "ape" });
    }
    Ok(Some(total))
}

#[cfg(test)]
mod tests {
    use super::{ape_tag_bytes, id3v2_tag_bytes, is_id3v1};

    #[test]
    fn synchsafe_size_includes_the_ten_byte_header() {
        let mut header = [0u8; 10];
        header[0] = b'I';
        header[1] = b'D';
        header[2] = b'3';
        header[3] = 4;
        header[9] = 1;
        assert_eq!(id3v2_tag_bytes(&header).expect("id3"), Some(11));
    }

    #[test]
    fn footer_flag_adds_ten_bytes_and_a_huge_size_is_a_limit() {
        let mut header = [0u8; 10];
        header[0] = b'I';
        header[1] = b'D';
        header[2] = b'3';
        header[3] = 4;
        header[5] = 0x10;
        header[9] = 1;
        assert_eq!(id3v2_tag_bytes(&header).expect("id3"), Some(21));

        header[5] = 0;
        header[6] = 0x0F;
        header[7] = 0x7F;
        header[8] = 0x7F;
        header[9] = 0x7F;
        assert_eq!(
            id3v2_tag_bytes(&header),
            Err(cutlass_container::ContainerError::Limit { what: "id3" })
        );
    }

    #[test]
    fn id3v1_is_tag_at_128_bytes() {
        let mut tail = [0u8; 128];
        tail[0] = b'T';
        tail[1] = b'A';
        tail[2] = b'G';
        assert!(is_id3v1(&tail));
        tail[2] = b'X';
        assert!(!is_id3v1(&tail));
    }

    #[test]
    fn ape_footer_span_includes_a_header_when_flagged() {
        let mut footer = [0u8; 32];
        footer[..8].copy_from_slice(b"APETAGEX");
        footer[8..12].copy_from_slice(&2000u32.to_le_bytes());
        footer[12..16].copy_from_slice(&64u32.to_le_bytes());
        footer[20..24].copy_from_slice(&0x8000_0000u32.to_le_bytes());
        assert_eq!(ape_tag_bytes(&footer).expect("ape"), Some(96));
    }
}
