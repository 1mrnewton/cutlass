//! Annex B and AVC/HEVC configuration records.
//!
//! Media Foundation wants Annex B. A file hands length-prefixed units and
//! an `avcC`, and the muxer wants them back. This module
//! is the conversion. It does not decode the slice.

// The HEVC records are here for the HEVC transform, which is not wired yet.
#![allow(dead_code)]

use cutlass_decode::CodecError;

/// One NAL unit without its start code or length prefix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Nal {
    pub bytes: Vec<u8>,
}

/// Parameter sets taken from an AVC decoder configuration record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AvcConfig {
    pub length_size: u8,
    pub profile: u8,
    pub compatibility: u8,
    pub level: u8,
    pub sps: Vec<Vec<u8>>,
    pub pps: Vec<Vec<u8>>,
}

/// Parameter sets taken from an HEVC decoder configuration record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HevcConfig {
    pub length_size: u8,
    pub nals: Vec<Vec<u8>>,
}

/// Private envelope for HEVC parameter sets this crate writes.
///
/// A real `hvcC` still parses through [`hevc_config`]. This envelope is what
/// the Apple encoder attaches when it has the NAL units and not the profile
/// fields a full configuration record needs.
pub const HEVC_ENVELOPE: &[u8; 4] = b"HVCC";

/// Split a buffer on 3-byte and 4-byte start codes.
pub fn annex_b(data: &[u8]) -> Result<Vec<Nal>, CodecError> {
    let starts = start_codes(data);
    if starts.is_empty() {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "annex b",
        });
    }
    let mut nals = Vec::with_capacity(starts.len());
    for (index, &(at, prefix)) in starts.iter().enumerate() {
        let begin = at + prefix;
        let end = starts
            .get(index + 1)
            .map(|(next, _)| *next)
            .unwrap_or(data.len());
        if begin > end || begin > data.len() {
            return Err(CodecError::Malformed {
                at_bit: bit_at(at),
                what: "annex b",
            });
        }
        let bytes = data[begin..end].to_vec();
        if bytes.is_empty() {
            return Err(CodecError::Malformed {
                at_bit: bit_at(begin),
                what: "empty nal",
            });
        }
        nals.push(Nal { bytes });
    }
    Ok(nals)
}

/// Write start codes in front of each NAL.
#[must_use]
pub fn to_annex_b(nals: &[Nal]) -> Vec<u8> {
    let mut out = Vec::new();
    for nal in nals {
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&nal.bytes);
    }
    out
}

/// Replace start codes with a 4-byte big-endian length.
pub fn annex_b_to_length(data: &[u8]) -> Result<Vec<u8>, CodecError> {
    let nals = annex_b(data)?;
    length_prefixed(&nals, 4)
}

/// Split one access unit whose NAL lengths are `length_size` bytes.
pub fn split_lengths(data: &[u8], length_size: u8) -> Result<Vec<Nal>, CodecError> {
    if !matches!(length_size, 1 | 2 | 4) {
        return Err(CodecError::Unsupported {
            what: "nal length size",
        });
    }
    let width = usize::from(length_size);
    let mut nals = Vec::new();
    let mut at = 0usize;
    while at < data.len() {
        if data.len() - at < width {
            return Err(CodecError::Truncated {
                at_bit: bit_at(at),
                what: "nal length",
            });
        }
        let mut raw = [0u8; 4];
        raw[4 - width..].copy_from_slice(&data[at..at + width]);
        let len = usize::try_from(u32::from_be_bytes(raw))
            .map_err(|_| CodecError::Limit { what: "nal length" })?;
        at += width;
        if len == 0 || data.len() - at < len {
            return Err(CodecError::Truncated {
                at_bit: bit_at(at),
                what: "nal",
            });
        }
        nals.push(Nal {
            bytes: data[at..at + len].to_vec(),
        });
        at += len;
    }
    if nals.is_empty() {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "access unit",
        });
    }
    Ok(nals)
}

/// Prefix each NAL with a big-endian length of `length_size` bytes.
pub fn length_prefixed(nals: &[Nal], length_size: u8) -> Result<Vec<u8>, CodecError> {
    if !matches!(length_size, 1 | 2 | 4) {
        return Err(CodecError::Unsupported {
            what: "nal length size",
        });
    }
    let width = usize::from(length_size);
    let max = match length_size {
        1 => u64::from(u8::MAX),
        2 => u64::from(u16::MAX),
        _ => u64::from(u32::MAX),
    };
    let mut out = Vec::new();
    for nal in nals {
        let len =
            u64::try_from(nal.bytes.len()).map_err(|_| CodecError::Limit { what: "nal length" })?;
        if len == 0 || len > max {
            return Err(CodecError::Limit { what: "nal length" });
        }
        let narrow = u32::try_from(len).map_err(|_| CodecError::Limit { what: "nal length" })?;
        let bytes = narrow.to_be_bytes();
        out.extend_from_slice(&bytes[4 - width..]);
        out.extend_from_slice(&nal.bytes);
    }
    Ok(out)
}

/// H.264 NAL type in the low five bits of the header.
#[must_use]
pub fn h264_type(nal: &[u8]) -> Option<u8> {
    nal.first().map(|header| header & 0x1f)
}

/// HEVC NAL type in bits 1 through 6 of the header.
#[must_use]
pub fn hevc_type(nal: &[u8]) -> Option<u8> {
    nal.first().map(|header| (header >> 1) & 0x3f)
}

/// Parse an AVC decoder configuration record.
pub fn avc_config(data: &[u8]) -> Result<AvcConfig, CodecError> {
    if data.len() < 7 || data[0] != 1 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "avcC",
        });
    }
    let length_size = (data[4] & 0x03) + 1;
    if !matches!(length_size, 1 | 2 | 4) {
        return Err(CodecError::Malformed {
            at_bit: 4 * 8,
            what: "avcC",
        });
    }
    let mut at = 5usize;
    let sps_count = usize::from(data[at] & 0x1f);
    at += 1;
    let sps = read_sets(data, &mut at, sps_count, "avcC sps")?;
    if at >= data.len() {
        return Err(CodecError::Truncated {
            at_bit: bit_at(at),
            what: "avcC pps",
        });
    }
    let pps_count = usize::from(data[at]);
    at += 1;
    let pps = read_sets(data, &mut at, pps_count, "avcC pps")?;
    if sps.is_empty() || pps.is_empty() {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "avcC sets",
        });
    }
    Ok(AvcConfig {
        length_size,
        profile: data[1],
        compatibility: data[2],
        level: data[3],
        sps,
        pps,
    })
}

/// Build an AVC decoder configuration record from parameter sets.
pub fn write_avc_config(config: &AvcConfig) -> Result<Vec<u8>, CodecError> {
    if config.sps.is_empty() || config.pps.is_empty() || config.sps.len() > 31 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "avcC sets",
        });
    }
    if !matches!(config.length_size, 1 | 2 | 4) {
        return Err(CodecError::Unsupported {
            what: "nal length size",
        });
    }
    let sps_count =
        u8::try_from(config.sps.len()).map_err(|_| CodecError::Limit { what: "avcC sps" })?;
    let mut out = vec![
        1,
        config.profile,
        config.compatibility,
        config.level,
        0xfc | (config.length_size - 1),
        0xe0 | sps_count,
    ];
    write_sets(&mut out, &config.sps)?;
    out.push(u8::try_from(config.pps.len()).map_err(|_| CodecError::Limit { what: "avcC pps" })?);
    write_sets(&mut out, &config.pps)?;
    Ok(out)
}

/// Parse the NAL arrays of an HEVC decoder configuration record.
///
/// Profile fields are skipped. The NAL units are what a decoder needs.
pub fn hevc_config(data: &[u8]) -> Result<HevcConfig, CodecError> {
    if data.len() < 23 || data[0] != 1 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "hvcC",
        });
    }
    let length_size = (data[21] & 0x03) + 1;
    if !matches!(length_size, 1 | 2 | 4) {
        return Err(CodecError::Malformed {
            at_bit: 21 * 8,
            what: "hvcC",
        });
    }
    let arrays = usize::from(data[22]);
    let mut at = 23usize;
    let mut nals = Vec::new();
    for _ in 0..arrays {
        if at >= data.len() {
            return Err(CodecError::Truncated {
                at_bit: bit_at(at),
                what: "hvcC array",
            });
        }
        at += 1;
        if data.len() - at < 2 {
            return Err(CodecError::Truncated {
                at_bit: bit_at(at),
                what: "hvcC count",
            });
        }
        let count = usize::from(u16::from_be_bytes([data[at], data[at + 1]]));
        at += 2;
        nals.extend(read_sets(data, &mut at, count, "hvcC nal")?);
    }
    if nals.is_empty() {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "hvcC sets",
        });
    }
    Ok(HevcConfig { length_size, nals })
}

/// Write the private HEVC envelope.
pub fn write_hevc_envelope(nals: &[Vec<u8>]) -> Result<Vec<u8>, CodecError> {
    if nals.is_empty() || nals.len() > usize::from(u16::MAX) {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "hvcC sets",
        });
    }
    let mut out = Vec::from(*HEVC_ENVELOPE);
    out.push(1);
    out.push(4);
    let count = u16::try_from(nals.len()).map_err(|_| CodecError::Limit { what: "hvcC sets" })?;
    out.extend_from_slice(&count.to_be_bytes());
    write_sets(&mut out, nals)?;
    Ok(out)
}

/// Parse either a private HEVC envelope or an `hvcC`.
pub fn read_hevc_sets(data: &[u8]) -> Result<HevcConfig, CodecError> {
    if data.starts_with(HEVC_ENVELOPE) {
        if data.len() < 8 {
            return Err(CodecError::Truncated {
                at_bit: 0,
                what: "hvcC",
            });
        }
        let length_size = data[5];
        if !matches!(length_size, 1 | 2 | 4) {
            return Err(CodecError::Malformed {
                at_bit: 5 * 8,
                what: "hvcC",
            });
        }
        let count = usize::from(u16::from_be_bytes([data[6], data[7]]));
        let mut at = 8usize;
        let nals = read_sets(data, &mut at, count, "hvcC nal")?;
        if nals.is_empty() {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "hvcC sets",
            });
        }
        return Ok(HevcConfig { length_size, nals });
    }
    hevc_config(data)
}

/// Build an `hvcC` from VPS, SPS, and PPS NAL units, with 4-byte lengths.
///
/// The profile, tier, and level are copied from the SPS. Chroma format and
/// bit depth are not parsed from it: Main is 4:2:0 at 8 bits, Main 10 at 10,
/// which are the profiles an encoder here is asked for.
pub fn write_hevc_config(nals: &[Vec<u8>]) -> Result<Vec<u8>, CodecError> {
    let of = |kind: u8| -> Vec<Vec<u8>> {
        nals.iter()
            .filter(|nal| hevc_type(nal) == Some(kind))
            .cloned()
            .collect()
    };
    let (vps, sps, pps) = (of(32), of(33), of(34));
    let Some(first) = sps.first() else {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "hvcC sps",
        });
    };
    if vps.is_empty() || pps.is_empty() {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "hvcC sets",
        });
    }
    // Two header bytes, one byte of ids and sub-layer count, then the
    // 12-byte general profile, tier, and level.
    let rbsp = unescape(first);
    if rbsp.len() < 15 {
        return Err(CodecError::Truncated {
            at_bit: bit_at(rbsp.len()),
            what: "hevc sps",
        });
    }
    let sub_layers = (rbsp[2] >> 1) & 0x07;
    let nested = rbsp[2] & 0x01;
    let general = &rbsp[3..15];
    let profile = general[0] & 0x1f;
    let depth = if profile == 2 { 2 } else { 0 };
    let mut out = Vec::with_capacity(64);
    out.push(1);
    out.extend_from_slice(general);
    out.extend_from_slice(&[0xF0, 0x00, 0xFC, 0xFD, 0xF8 | depth, 0xF8 | depth, 0, 0]);
    out.push(((sub_layers + 1) << 3) | (nested << 2) | 3);
    out.push(3);
    for (kind, sets) in [(32_u8, &vps), (33, &sps), (34, &pps)] {
        out.push(0x80 | kind);
        let count =
            u16::try_from(sets.len()).map_err(|_| CodecError::Limit { what: "hvcC sets" })?;
        out.extend_from_slice(&count.to_be_bytes());
        write_sets(&mut out, sets)?;
    }
    Ok(out)
}

/// Drop emulation prevention bytes: `00 00 03` reads as `00 00`.
fn unescape(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut zeros = 0;
    for &byte in data {
        if zeros >= 2 && byte == 3 {
            zeros = 0;
            continue;
        }
        zeros = if byte == 0 { zeros + 1 } else { 0 };
        out.push(byte);
    }
    out
}

fn read_sets(
    data: &[u8],
    at: &mut usize,
    count: usize,
    what: &'static str,
) -> Result<Vec<Vec<u8>>, CodecError> {
    let mut sets = Vec::with_capacity(count);
    for _ in 0..count {
        if data.len() - *at < 2 {
            return Err(CodecError::Truncated {
                at_bit: bit_at(*at),
                what,
            });
        }
        let len = usize::from(u16::from_be_bytes([data[*at], data[*at + 1]]));
        *at += 2;
        if len == 0 || data.len() - *at < len {
            return Err(CodecError::Truncated {
                at_bit: bit_at(*at),
                what,
            });
        }
        sets.push(data[*at..*at + len].to_vec());
        *at += len;
    }
    Ok(sets)
}

fn write_sets(out: &mut Vec<u8>, sets: &[Vec<u8>]) -> Result<(), CodecError> {
    for set in sets {
        let len = u16::try_from(set.len()).map_err(|_| CodecError::Limit { what: "nal length" })?;
        if len == 0 {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "empty nal",
            });
        }
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(set);
    }
    Ok(())
}

fn start_codes(data: &[u8]) -> Vec<(usize, usize)> {
    let mut found = Vec::new();
    let mut index = 0usize;
    while index + 3 <= data.len() {
        if data[index] == 0 && data[index + 1] == 0 {
            if data[index + 2] == 1 {
                found.push((index, 3));
                index += 3;
                continue;
            }
            if index + 4 <= data.len() && data[index + 2] == 0 && data[index + 3] == 1 {
                found.push((index, 4));
                index += 4;
                continue;
            }
        }
        index += 1;
    }
    found
}

fn bit_at(byte: usize) -> u64 {
    u64::try_from(byte).unwrap_or(u64::MAX).saturating_mul(8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn annex_b_keeps_both_start_code_lengths() {
        let data = [0, 0, 1, 0x67, 0xaa, 0, 0, 0, 1, 0x68, 0xbb, 0xcc];
        let nals = annex_b(&data).expect("split");
        assert_eq!(nals.len(), 2);
        assert_eq!(nals[0].bytes, vec![0x67, 0xaa]);
        assert_eq!(nals[1].bytes, vec![0x68, 0xbb, 0xcc]);
        assert_eq!(h264_type(&nals[0].bytes), Some(7));
        assert_eq!(h264_type(&nals[1].bytes), Some(8));
    }

    #[test]
    fn annex_b_without_a_start_code_is_malformed() {
        assert!(matches!(
            annex_b(&[0, 0, 2, 1]),
            Err(CodecError::Malformed {
                what: "annex b",
                ..
            })
        ));
    }

    #[test]
    fn an_empty_nal_between_start_codes_is_malformed() {
        assert!(matches!(
            annex_b(&[0, 0, 1, 0, 0, 1, 0x65]),
            Err(CodecError::Malformed {
                what: "empty nal",
                ..
            })
        ));
    }

    #[test]
    fn length_prefixes_round_trip_through_annex_b() {
        let nals = vec![
            Nal {
                bytes: vec![0x65, 1, 2, 3],
            },
            Nal {
                bytes: vec![0x41, 9],
            },
        ];
        let prefixed = length_prefixed(&nals, 4).expect("prefix");
        assert_eq!(split_lengths(&prefixed, 4).expect("split"), nals);
        let annex = to_annex_b(&nals);
        assert_eq!(annex_b_to_length(&annex).expect("convert"), prefixed);
    }

    #[test]
    fn a_short_length_prefix_is_truncated() {
        assert!(matches!(
            split_lengths(&[0, 0, 0], 4),
            Err(CodecError::Truncated {
                what: "nal length",
                ..
            })
        ));
        assert!(matches!(
            split_lengths(&[0, 5, 1, 2], 2),
            Err(CodecError::Truncated { what: "nal", .. })
        ));
    }

    #[test]
    fn a_one_byte_length_rejects_a_nal_past_255() {
        let nal = Nal {
            bytes: vec![0; 256],
        };
        assert_eq!(
            length_prefixed(&[nal], 1),
            Err(CodecError::Limit { what: "nal length" })
        );
    }

    #[test]
    fn avc_config_round_trips_one_sps_and_one_pps() {
        let config = AvcConfig {
            length_size: 4,
            profile: 0x42,
            compatibility: 0,
            level: 0x1f,
            sps: vec![vec![0x67, 0x42, 0x00, 0x1f, 0xaa]],
            pps: vec![vec![0x68, 0xce, 0x38, 0x80]],
        };
        let bytes = write_avc_config(&config).expect("write");
        assert_eq!(avc_config(&bytes).expect("read"), config);
    }

    #[test]
    fn a_truncated_avc_config_names_the_missing_set() {
        let err = avc_config(&[1, 0x42, 0, 0x1f, 0xff, 0xe1, 0x00]).expect_err("short");
        assert!(matches!(
            err,
            CodecError::Truncated {
                what: "avcC sps",
                ..
            }
        ));
    }

    #[test]
    fn an_avc_config_without_a_pps_is_malformed() {
        let bytes = [1, 0x42, 0, 0x1f, 0xff, 0xe1, 0x00, 0x02, 0x67, 0x42, 0x00];
        assert!(matches!(
            avc_config(&bytes),
            Err(CodecError::Malformed {
                what: "avcC sets",
                ..
            })
        ));
    }

    #[test]
    fn hevc_config_reads_one_array_of_parameter_sets() {
        let vps = [0x40, 0x01, 0x0c];
        let mut data = vec![1u8];
        data.extend(std::iter::repeat_n(0, 20));
        data.push(0xff);
        data.push(1);
        data.push(0x20);
        data.extend_from_slice(&1u16.to_be_bytes());
        data.extend_from_slice(&u16::try_from(vps.len()).unwrap().to_be_bytes());
        data.extend_from_slice(&vps);
        let config = hevc_config(&data).expect("hvcC");
        assert_eq!(config.length_size, 4);
        assert_eq!(config.nals, vec![vps.to_vec()]);
        assert_eq!(hevc_type(&config.nals[0]), Some(32));
    }

    #[test]
    fn a_short_hevc_header_is_malformed() {
        assert!(matches!(
            hevc_config(&[1, 2, 3]),
            Err(CodecError::Malformed { what: "hvcC", .. })
        ));
    }

    #[test]
    fn a_hevc_envelope_round_trips_the_parameter_sets() {
        let nals = vec![vec![0x40, 0x01], vec![0x42, 0x01], vec![0x44, 0x01]];
        let bytes = write_hevc_envelope(&nals).expect("write");
        let config = read_hevc_sets(&bytes).expect("read");
        assert_eq!(config.length_size, 4);
        assert_eq!(config.nals, nals);
    }

    #[test]
    fn an_hvcc_carries_the_sps_profile_and_reads_back() {
        let vps = vec![0x40, 0x01, 0x0c];
        // Main profile, compatibility bit 1, level 3.1 (93), one sub-layer.
        let mut sps = vec![
            0x42, 0x01, 0x01, 0x01, 0x60, 0, 0, 0, 0x90, 0, 0, 0, 0, 0, 93,
        ];
        sps.extend_from_slice(&[0xa0, 0x03]);
        let pps = vec![0x44, 0x01, 0xc1];
        let bytes = write_hevc_config(&[vps.clone(), sps.clone(), pps.clone()]).expect("write");
        assert_eq!(bytes[1], 0x01, "profile space, tier, and Main");
        assert_eq!(bytes[12], 93, "level");
        assert_eq!(bytes[21] & 0x03, 3, "4-byte lengths");
        let config = hevc_config(&bytes).expect("read");
        assert_eq!(config.length_size, 4);
        assert_eq!(config.nals, vec![vps, sps, pps]);
        assert!(write_hevc_config(&[vec![0x42, 0x01]]).is_err());
    }
}
