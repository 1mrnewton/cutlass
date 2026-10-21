//! H.264 format description from the `avcC` record.

use cutlass_core::PixelFormat;
use objc2_core_media::kCMVideoCodecType_H264;

use super::{Description, skip_length_prefixed};
use crate::error::Error;

/// `profile_idc` values that allow more than 8 bits per sample: CAVLC 4:4:4
/// Intra, High 10, High 4:2:2, and High 4:4:4 Predictive.
const HIGH_BIT_DEPTH_PROFILES: [u8; 4] = [44, 110, 122, 244];

/// ISO/IEC 14496-15 5.3.3.1: version, profile, compatibility, level, NAL
/// length size, then the SPS and PPS lists.
pub(super) fn describe(avcc: &[u8]) -> Result<Description<'_>, Error> {
    let &[
        version,
        profile,
        _,
        _,
        length_size,
        sps_count,
        ref rest @ ..,
    ] = avcc
    else {
        return Err(Error::InvalidConfig(
            "avcC is shorter than its 6-byte header",
        ));
    };
    if version != 1 {
        return Err(Error::InvalidConfig("avcC version is not 1"));
    }
    if length_size & 0b11 == 2 {
        return Err(Error::InvalidConfig(
            "avcC NAL length size is 3 bytes, which is not allowed",
        ));
    }
    let sps_count = sps_count & 0b1_1111;
    if sps_count == 0 {
        return Err(Error::InvalidConfig("avcC has no SPS"));
    }
    let rest = skip_length_prefixed(rest, sps_count.into(), "avcC SPS runs past the end")?;
    let (&pps_count, rest) = rest
        .split_first()
        .ok_or(Error::InvalidConfig("avcC ends before the PPS count"))?;
    skip_length_prefixed(rest, pps_count.into(), "avcC PPS runs past the end")?;

    let output = if HIGH_BIT_DEPTH_PROFILES.contains(&profile) {
        PixelFormat::P010
    } else {
        PixelFormat::Nv12
    };
    Ok(Description {
        codec_type: kCMVideoCodecType_H264,
        atom: Some(("avcC", avcc)),
        output,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const AVCC_HIGH: [u8; 15] = [
        1, 100, 0, 30, 0xff, 0xe1, 0, 3, 0x67, 0x64, 0x00, 1, 0, 1, 0x68,
    ];

    #[test]
    fn accepts_a_minimal_record() {
        let description = describe(&AVCC_HIGH).unwrap();
        assert_eq!(description.output, PixelFormat::Nv12);
    }

    #[test]
    fn high_10_asks_for_10_bit_output() {
        let mut avcc = AVCC_HIGH;
        avcc[1] = 110;
        assert_eq!(describe(&avcc).unwrap().output, PixelFormat::P010);
    }

    #[test]
    fn rejects_sps_running_past_the_end() {
        assert!(describe(&AVCC_HIGH[..9]).is_err());
    }

    #[test]
    fn rejects_missing_pps_count() {
        assert!(describe(&AVCC_HIGH[..11]).is_err());
    }

    #[test]
    fn rejects_record_without_sps() {
        let mut avcc = AVCC_HIGH;
        avcc[5] = 0xe0;
        assert!(describe(&avcc).is_err());
    }

    #[test]
    fn rejects_three_byte_nal_lengths() {
        let mut avcc = AVCC_HIGH;
        avcc[4] = 0xfe;
        assert!(describe(&avcc).is_err());
    }
}
