//! HEVC format description from the `hvcC` record.

use cutlass_core::PixelFormat;
use objc2_core_media::kCMVideoCodecType_HEVC;

use super::{Description, skip_length_prefixed};
use crate::error::Error;

const HEADER_LEN: usize = 23;

/// ISO/IEC 14496-15 8.3.3.1: a 23-byte header, then arrays of parameter set
/// NAL units.
pub(super) fn describe(hvcc: &[u8]) -> Result<Description<'_>, Error> {
    let (header, mut rest) = hvcc
        .split_first_chunk::<HEADER_LEN>()
        .ok_or(Error::InvalidConfig(
            "hvcC is shorter than its 23-byte header",
        ))?;
    if header[0] != 1 {
        return Err(Error::InvalidConfig("hvcC version is not 1"));
    }
    if header[21] & 0b11 == 2 {
        return Err(Error::InvalidConfig(
            "hvcC NAL length size is 3 bytes, which is not allowed",
        ));
    }
    for _ in 0..header[22] {
        let &[_nal_type, count_hi, count_lo, ref tail @ ..] = rest else {
            return Err(Error::InvalidConfig("hvcC array header runs past the end"));
        };
        let count = u16::from_be_bytes([count_hi, count_lo]);
        rest = skip_length_prefixed(tail, count.into(), "hvcC NAL unit runs past the end")?;
    }

    let bit_depth = 8 + (header[17] & 0b111);
    let output = if bit_depth > 8 {
        PixelFormat::P010
    } else {
        PixelFormat::Nv12
    };
    Ok(Description {
        codec_type: kCMVideoCodecType_HEVC,
        atom: Some(("hvcC", hvcc)),
        output,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hvcc(bit_depth_minus_8: u8) -> Vec<u8> {
        let mut record = vec![0; HEADER_LEN];
        record[0] = 1;
        record[17] = 0xf8 | bit_depth_minus_8;
        record[21] = 0x0f;
        record[22] = 1;
        record.extend_from_slice(&[0x20, 0, 1, 0, 2, 0x40, 0x01]);
        record
    }

    #[test]
    fn main_asks_for_8_bit_output() {
        assert_eq!(describe(&hvcc(0)).unwrap().output, PixelFormat::Nv12);
    }

    #[test]
    fn main_10_asks_for_10_bit_output() {
        assert_eq!(describe(&hvcc(2)).unwrap().output, PixelFormat::P010);
    }

    #[test]
    fn rejects_short_header() {
        assert!(describe(&hvcc(0)[..HEADER_LEN - 1]).is_err());
    }

    #[test]
    fn rejects_nal_unit_running_past_the_end() {
        let record = hvcc(0);
        assert!(describe(&record[..record.len() - 1]).is_err());
    }

    #[test]
    fn rejects_array_count_without_arrays() {
        let mut record = hvcc(0);
        record[22] = 2;
        assert!(describe(&record).is_err());
    }
}
