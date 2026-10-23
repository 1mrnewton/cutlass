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

/// An HEVC sample, as far as a seek needs to know.
pub(super) enum Picture {
    /// CRA or a BLA that can be followed by RASL pictures.
    Clean,
    /// A leading picture that still refers to the group before the clean point.
    Rasl,
    /// A picture a decoder can keep after a clean random access.
    Coded,
    /// No picture NAL.
    Empty,
}

/// Classifies the VCL NALs in one length-prefixed sample.
pub(super) fn picture(data: &[u8], length: usize) -> Picture {
    if length == 0 || length > 4 {
        return Picture::Empty;
    }
    let mut clean = false;
    let mut rasl = false;
    let mut coded = false;
    let mut rest = data;
    while rest.len() >= length {
        let mut size = 0usize;
        for byte in &rest[..length] {
            size = (size << 8) | usize::from(*byte);
        }
        rest = &rest[length..];
        if size == 0 || size > rest.len() {
            break;
        }
        match (rest[0] >> 1) & 0x3f {
            16 | 21 => clean = true,
            8 | 9 => rasl = true,
            0..=31 => coded = true,
            _ => {}
        }
        rest = &rest[size..];
    }
    if clean && !coded {
        Picture::Clean
    } else if rasl && !coded && !clean {
        Picture::Rasl
    } else if coded || clean {
        Picture::Coded
    } else {
        Picture::Empty
    }
}

/// `true` when this sample must not be decoded.
///
/// `open` is set once this span has sent a picture. A seek starts closed, so
/// the clean random access it starts on drops the RASL pictures after it.
/// Those pictures still need the group the seek did not decode. A clean
/// random access reached during playback keeps them, because that group is
/// already decoded.
pub(super) fn skip_leading(open: &mut bool, drop_rasl: &mut bool, picture: Picture) -> bool {
    match picture {
        Picture::Rasl if *drop_rasl => true,
        Picture::Clean if !*open => {
            *open = true;
            *drop_rasl = true;
            false
        }
        Picture::Clean | Picture::Coded | Picture::Rasl => {
            *open = true;
            *drop_rasl = false;
            false
        }
        Picture::Empty => false,
    }
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

    #[test]
    fn rasl_after_a_clean_random_access_is_skipped() {
        let cra = sample(4, 21);
        let rasl = sample(4, 8);
        let trail = sample(4, 1);
        let mut open = false;
        let mut drop = false;
        assert!(!skip_leading(&mut open, &mut drop, picture(&cra, 4)));
        assert!(drop);
        assert!(skip_leading(&mut open, &mut drop, picture(&rasl, 4)));
        assert!(!skip_leading(&mut open, &mut drop, picture(&trail, 4)));
        assert!(!drop);
        assert!(!skip_leading(&mut open, &mut drop, picture(&cra, 4)));
        assert!(!drop);
        assert!(!skip_leading(&mut open, &mut drop, picture(&rasl, 4)));
    }

    fn sample(length: usize, nal_type: u8) -> Vec<u8> {
        let mut bytes = vec![0; length + 2];
        bytes[length - 1] = 2;
        bytes[length] = nal_type << 1;
        bytes
    }
}
