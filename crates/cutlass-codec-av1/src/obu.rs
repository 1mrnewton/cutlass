//! Low-overhead OBU splitting.
//!
//! Payloads are copied into owned buffers so a later tile decode does not
//! borrow the caller's packet.

use crate::bits::BitReader;
use crate::error::DecodeError;
use crate::limits::DecoderLimits;

pub const OBU_SEQUENCE_HEADER: u8 = 1;
pub const OBU_TEMPORAL_DELIMITER: u8 = 2;
pub const OBU_FRAME_HEADER: u8 = 3;
pub const OBU_TILE_GROUP: u8 = 4;
pub const OBU_METADATA: u8 = 5;
pub const OBU_FRAME: u8 = 6;
pub const OBU_REDUNDANT_FRAME_HEADER: u8 = 7;
pub const OBU_TILE_LIST: u8 = 8;
pub const OBU_PADDING: u8 = 15;

/// One OBU with its payload (bytes after the header and optional size).
#[derive(Debug, Clone)]
pub struct RawObu {
    pub obu_type: u8,
    pub extension: bool,
    pub temporal_id: u8,
    pub spatial_id: u8,
    /// Payload bytes. Length is `obu_size`.
    pub payload: Vec<u8>,
}

/// Split a low-overhead buffer into complete OBUs. `has_size_field` must be 1.
pub fn split_low_overhead(data: &[u8], limits: &DecoderLimits) -> Result<Vec<RawObu>, DecodeError> {
    if data.len() > limits.max_packet_bytes {
        return Err(DecodeError::limit("packet exceeds max_packet_bytes"));
    }
    let mut out = Vec::new();
    let mut rest = data;
    if rest.is_empty() {
        return Err(DecodeError::malformed("empty AV1 packet"));
    }
    while !rest.is_empty() {
        let (obu, consumed) = read_obu(rest, None, limits)?;
        if obu.obu_type == OBU_TILE_LIST {
            return Err(DecodeError::unsupported(
                "large-scale tile list OBU (specialized mode is not enabled)",
            ));
        }
        rest = &rest[consumed..];
        out.push(obu);
        if out.len() > limits.max_pending_obus {
            return Err(DecodeError::limit("too many OBUs in one packet"));
        }
    }
    Ok(out)
}

/// Parse one OBU. `enclosing` is the full OBU length including the header when
/// the caller already knows it (Annex B). `None` means the size field is
/// required and the buffer is only a prefix bound (the remaining slice).
pub fn read_obu(
    data: &[u8],
    enclosing: Option<usize>,
    limits: &DecoderLimits,
) -> Result<(RawObu, usize), DecodeError> {
    if data.is_empty() {
        return Err(DecodeError::truncated("OBU header"));
    }
    let mut bits = BitReader::new(data);
    let forbidden = bits.f(1)?;
    if forbidden != 0 {
        return Err(DecodeError::malformed("obu_forbidden_bit"));
    }
    let obu_type = bits.f(4)? as u8;
    let extension = bits.f(1)? == 1;
    let has_size = bits.f(1)? == 1;
    let reserved = bits.f(1)?;
    if reserved != 0 {
        return Err(DecodeError::malformed("obu_reserved_1bit"));
    }
    let (temporal_id, spatial_id) = if extension {
        let temporal_id = bits.f(3)? as u8;
        let spatial_id = bits.f(2)? as u8;
        let ext_reserved = bits.f(3)?;
        if ext_reserved != 0 {
            return Err(DecodeError::malformed("extension_header_reserved_3bits"));
        }
        (temporal_id, spatial_id)
    } else {
        (0, 0)
    };
    let header_bytes = if extension { 2 } else { 1 };
    let (payload_size, size_field_bytes) = if has_size {
        let (sz, n) = bits.leb128()?;
        (sz as usize, n as usize)
    } else if let Some(total) = enclosing {
        if total < header_bytes {
            return Err(DecodeError::malformed(
                "enclosing OBU length smaller than the header",
            ));
        }
        (total - header_bytes, 0)
    } else {
        return Err(DecodeError::malformed(
            "obu_has_size_field is 0 without an enclosing length",
        ));
    };
    if payload_size > limits.max_obu_bytes {
        return Err(DecodeError::limit("OBU payload exceeds max_obu_bytes"));
    }
    let total = header_bytes
        .checked_add(size_field_bytes)
        .and_then(|n| n.checked_add(payload_size))
        .ok_or_else(|| DecodeError::malformed("OBU size overflow"))?;
    if enclosing.is_some_and(|enclosing| total != enclosing) {
        return Err(DecodeError::malformed(
            "obu_size does not match the enclosing length",
        ));
    }
    if data.len() < total {
        return Err(DecodeError::truncated("OBU payload"));
    }
    if obu_type == OBU_TILE_LIST {
        // Bounded rejection: the payload is not interpreted as tile groups.
        return Err(DecodeError::unsupported(
            "large-scale tile list OBU (specialized mode is not enabled)",
        ));
    }
    let payload_at = header_bytes + size_field_bytes;
    let payload = data[payload_at..total].to_vec();
    Ok((
        RawObu {
            obu_type,
            extension,
            temporal_id,
            spatial_id,
            payload,
        },
        total,
    ))
}

/// True when this OBU is part of the selected operating point.
///
/// Sequence headers and temporal delimiters are never dropped. `idc == 0`
/// means every layer.
pub fn in_operating_point(obu: &RawObu, idc: u16) -> bool {
    if obu.obu_type == OBU_SEQUENCE_HEADER || obu.obu_type == OBU_TEMPORAL_DELIMITER {
        return true;
    }
    if idc == 0 || !obu.extension {
        return true;
    }
    let temporal = ((idc >> obu.temporal_id) & 1) == 1;
    let spatial = ((idc >> (obu.spatial_id + 8)) & 1) == 1;
    temporal && spatial
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> DecoderLimits {
        DecoderLimits::default()
    }

    #[test]
    fn temporal_delimiter_round_trip() {
        // forbidden 0, type 2, extension 0, has_size 1, reserved 0, leb128 size 0.
        let data = [0b0001_0010, 0x00];
        let obus = split_low_overhead(&data, &limits()).unwrap();
        assert_eq!(obus.len(), 1);
        assert_eq!(obus[0].obu_type, OBU_TEMPORAL_DELIMITER);
        assert!(obus[0].payload.is_empty());
    }

    #[test]
    fn truncation_does_not_borrow_following_bytes() {
        // Size claims 5 payload bytes but only 1 follows.
        let data = [0b0001_0010, 0x05, 0xff];
        let err = split_low_overhead(&data, &limits()).unwrap_err();
        assert!(matches!(err, DecodeError::Truncated(_)));
    }

    #[test]
    fn reserved_header_bit() {
        let data = [0b0001_0011, 0x00];
        assert!(split_low_overhead(&data, &limits()).is_err());
    }

    #[test]
    fn tile_list_is_unsupported() {
        // type 8, has_size 1, one payload byte. Rejected before it can be a tile group.
        let data = [0b0100_0010, 0x01, 0x00];
        let err = split_low_overhead(&data, &limits()).unwrap_err();
        assert!(matches!(err, DecodeError::Unsupported(_)));
    }

    #[test]
    fn unsized_without_enclosing_is_malformed() {
        // has_size = 0, type temporal delimiter. Low-overhead input has no enclosing length.
        let data = [0b0001_0000];
        assert!(matches!(
            split_low_overhead(&data, &limits()),
            Err(DecodeError::Malformed(_))
        ));
    }
}
