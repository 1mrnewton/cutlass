//! Annex B length-delimited adapter.
//!
//! Temporal-unit, frame-unit, and OBU lengths are distinct from the low-overhead
//! `obu_size`. When an OBU also carries `obu_size`, the two lengths must
//! describe the same bytes.

use crate::bits::BitReader;
use crate::error::DecodeError;
use crate::limits::DecoderLimits;
use crate::obu::{self, OBU_TEMPORAL_DELIMITER, RawObu};

/// One Annex B temporal unit: the OBUs of its frame units, in order.
pub struct TemporalUnit {
    pub obus: Vec<RawObu>,
}

pub fn split_temporal_units(
    data: &[u8],
    limits: &DecoderLimits,
) -> Result<Vec<TemporalUnit>, DecodeError> {
    if data.len() > limits.max_packet_bytes {
        return Err(DecodeError::limit("packet exceeds max_packet_bytes"));
    }
    if data.is_empty() {
        return Err(DecodeError::malformed("empty Annex B packet"));
    }
    let mut bits = BitReader::new(data);
    let mut units = Vec::new();
    while bits.bits_left() > 0 {
        if !bits.byte_aligned() {
            return Err(DecodeError::malformed("Annex B length is not byte aligned"));
        }
        let (tu_size, _) = bits.leb128()?;
        let tu_size = tu_size as usize;
        if bits.bits_left() < tu_size * 8 {
            return Err(DecodeError::truncated("Annex B temporal unit"));
        }
        let start = bits.position() / 8;
        let tu_bytes = &data[start..start + tu_size];
        bits.skip_bits(tu_size * 8)?;
        units.push(parse_temporal_unit(tu_bytes, limits)?);
    }
    Ok(units)
}

fn parse_temporal_unit(data: &[u8], limits: &DecoderLimits) -> Result<TemporalUnit, DecodeError> {
    let mut bits = BitReader::new(data);
    let mut obus = Vec::new();
    let mut first = true;
    let mut sz = data.len();
    while sz > 0 {
        let pos = bits.position() / 8;
        let (frame_size, leb_bytes) = bits.leb128()?;
        let leb_bytes = leb_bytes as usize;
        if leb_bytes > sz {
            return Err(DecodeError::malformed("frame unit length overflow"));
        }
        sz -= leb_bytes;
        let frame_size = frame_size as usize;
        if frame_size > sz {
            return Err(DecodeError::truncated("Annex B frame unit"));
        }
        let frame_at = pos + leb_bytes;
        let frame = &data[frame_at..frame_at + frame_size];
        bits.skip_bits(frame_size * 8)?;
        sz -= frame_size;
        parse_frame_unit(frame, limits, &mut obus, &mut first)?;
    }
    if obus.is_empty() {
        return Err(DecodeError::malformed("empty Annex B temporal unit"));
    }
    Ok(TemporalUnit { obus })
}

fn parse_frame_unit(
    data: &[u8],
    limits: &DecoderLimits,
    out: &mut Vec<RawObu>,
    first_in_tu: &mut bool,
) -> Result<(), DecodeError> {
    let mut bits = BitReader::new(data);
    let mut sz = data.len();
    let mut saw_frame_header = false;
    while sz > 0 {
        let pos = bits.position() / 8;
        let (obu_length, leb_bytes) = bits.leb128()?;
        let leb_bytes = leb_bytes as usize;
        if leb_bytes > sz {
            return Err(DecodeError::malformed("OBU length overflow"));
        }
        sz -= leb_bytes;
        let obu_length = obu_length as usize;
        if obu_length > sz {
            return Err(DecodeError::truncated("Annex B OBU"));
        }
        let at = pos + leb_bytes;
        let bytes = &data[at..at + obu_length];
        bits.skip_bits(obu_length * 8)?;
        sz -= obu_length;
        let (obu, consumed) = obu::read_obu(bytes, Some(obu_length), limits)?;
        if consumed != obu_length {
            return Err(DecodeError::malformed(
                "Annex B OBU did not consume obu_length",
            ));
        }
        if *first_in_tu {
            if obu.obu_type != OBU_TEMPORAL_DELIMITER {
                return Err(DecodeError::malformed(
                    "Annex B temporal unit must start with a temporal delimiter",
                ));
            }
            *first_in_tu = false;
        } else if obu.obu_type == OBU_TEMPORAL_DELIMITER {
            return Err(DecodeError::malformed(
                "temporal delimiter is only legal at the start of a temporal unit",
            ));
        }
        if matches!(obu.obu_type, 3 | 6 | 7) {
            if saw_frame_header {
                return Err(DecodeError::malformed(
                    "Annex B frame unit contains more than one frame header",
                ));
            }
            saw_frame_header = true;
        }
        out.push(obu);
        if out.len() > limits.max_pending_obus {
            return Err(DecodeError::limit("too many OBUs in one packet"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packet_with_obus(obus: &[&[u8]]) -> Vec<u8> {
        let mut frame = Vec::new();
        for obu in obus {
            frame.push(obu.len() as u8);
            frame.extend_from_slice(obu);
        }
        let mut tu = vec![frame.len() as u8];
        tu.extend_from_slice(&frame);
        let mut packet = vec![tu.len() as u8];
        packet.extend_from_slice(&tu);
        packet
    }

    #[test]
    fn temporal_unit_with_delimiter() {
        // Temporal-unit length, frame-unit length, OBU length, then a
        // temporal delimiter whose own obu_size is 0.
        let td = [0b0001_0010, 0x00];
        let units =
            split_temporal_units(&packet_with_obus(&[&td]), &DecoderLimits::default()).unwrap();
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].obus.len(), 1);
        assert_eq!(units[0].obus[0].obu_type, OBU_TEMPORAL_DELIMITER);
    }

    #[test]
    fn missing_delimiter_is_malformed() {
        // A padding OBU is not a temporal delimiter.
        let obu = [0b0111_1010, 0x00];
        assert!(
            split_temporal_units(&packet_with_obus(&[&obu]), &DecoderLimits::default()).is_err()
        );
    }

    #[test]
    fn truncated_length() {
        let packet = [0x05]; // temporal unit claims 5 bytes
        assert!(matches!(
            split_temporal_units(&packet, &DecoderLimits::default()),
            Err(DecodeError::Truncated(_))
        ));
    }
}
