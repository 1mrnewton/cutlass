//! Annex B start codes and length-prefixed NAL units.
//!
//! `avcC` is the decoder configuration record. Its NAL units have no start code.

use cutlass_decode::CodecError;

pub const NAL_LIMIT: usize = 8 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct Nal {
    pub ref_idc: u8,
    pub kind: u8,
    pub rbsp: Vec<u8>,
}

pub fn rbsp(ebsp: &[u8]) -> Result<Vec<u8>, CodecError> {
    if ebsp.len() > NAL_LIMIT {
        return Err(CodecError::Limit { what: "nal unit" });
    }
    let mut out = Vec::with_capacity(ebsp.len());
    let mut zeros = 0u8;
    for &byte in ebsp {
        if zeros >= 2 && byte == 0x03 {
            zeros = 0;
            continue;
        }
        out.push(byte);
        if byte == 0 {
            zeros = zeros.saturating_add(1);
        } else {
            zeros = 0;
        }
    }
    Ok(out)
}

fn nal_from_ebsp(ebsp: &[u8]) -> Result<Nal, CodecError> {
    if ebsp.is_empty() {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "nal header",
        });
    }
    let header = ebsp[0];
    if header & 0x80 != 0 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "forbidden_zero_bit",
        });
    }
    let kind = header & 0x1f;
    if matches!(kind, 2 | 3 | 4 | 14 | 15 | 20 | 21) {
        return Err(CodecError::Unsupported {
            what: "nal unit type",
        });
    }
    Ok(Nal {
        ref_idc: (header >> 5) & 0x3,
        kind,
        rbsp: rbsp(&ebsp[1..])?,
    })
}

/// Annex B access unit, or one length-prefixed access unit.
pub fn access_unit(data: &[u8], length_size: u8) -> Result<Vec<Nal>, CodecError> {
    if data.is_empty() {
        return Err(CodecError::InvalidApi {
            what: "empty packet",
        });
    }
    if starts_annex_b(data) {
        return annex_b(data);
    }
    if length_size == 0 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "access unit framing",
        });
    }
    length_prefixed(data, length_size)
}

fn starts_annex_b(data: &[u8]) -> bool {
    data.starts_with(&[0, 0, 1]) || data.starts_with(&[0, 0, 0, 1])
}

fn annex_b(data: &[u8]) -> Result<Vec<Nal>, CodecError> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 2 < data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            starts.push(i);
            i += 3;
        } else {
            i += 1;
        }
    }
    if starts.is_empty() {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "annex b",
        });
    }
    let mut nals = Vec::new();
    for (n, &at) in starts.iter().enumerate() {
        let begin = at + 3;
        let mut end = starts.get(n + 1).copied().unwrap_or(data.len());
        // The next start code may be prefixed by a zero_byte. That byte is not
        // part of this NAL. Alignment zeros inside the NAL stay when this is
        // the last unit, and the bit reader stops on the rbsp stop bit.
        if n + 1 < starts.len() {
            while end > begin && data[end - 1] == 0 {
                end -= 1;
            }
        }
        if begin < end {
            nals.push(nal_from_ebsp(&data[begin..end])?);
        }
    }
    if nals.is_empty() {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "annex b",
        });
    }
    Ok(nals)
}

fn length_prefixed(data: &[u8], length_size: u8) -> Result<Vec<Nal>, CodecError> {
    if !matches!(length_size, 1 | 2 | 4) {
        return Err(CodecError::Unsupported {
            what: "nal length size",
        });
    }
    let width = length_size as usize;
    let mut nals = Vec::new();
    let mut i = 0;
    while i < data.len() {
        if i + width > data.len() {
            return Err(CodecError::Truncated {
                at_bit: (i as u64) * 8,
                what: "nal length",
            });
        }
        let mut len = 0usize;
        for byte in &data[i..i + width] {
            len = (len << 8) | usize::from(*byte);
        }
        i += width;
        if len > NAL_LIMIT || i + len > data.len() {
            return Err(CodecError::Truncated {
                at_bit: (i as u64) * 8,
                what: "length-prefixed nal",
            });
        }
        nals.push(nal_from_ebsp(&data[i..i + len])?);
        i += len;
    }
    Ok(nals)
}

/// Decoder configuration record. Returns the NAL length size and the parameter sets.
pub fn avcc(extra: &[u8]) -> Result<(u8, Vec<Nal>), CodecError> {
    if extra.len() < 7 || extra[0] != 1 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "avcC",
        });
    }
    let length_size = (extra[4] & 0x03) + 1;
    let mut nals = Vec::new();
    let mut i = 5;
    let sps_count = usize::from(extra[i] & 0x1f);
    i += 1;
    for _ in 0..sps_count {
        i = push_avcc_nal(extra, i, &mut nals)?;
    }
    if i >= extra.len() {
        return Err(CodecError::Truncated {
            at_bit: (i as u64) * 8,
            what: "avcC pps count",
        });
    }
    let pps_count = usize::from(extra[i]);
    i += 1;
    for _ in 0..pps_count {
        i = push_avcc_nal(extra, i, &mut nals)?;
    }
    Ok((length_size, nals))
}

fn push_avcc_nal(extra: &[u8], i: usize, nals: &mut Vec<Nal>) -> Result<usize, CodecError> {
    if i + 2 > extra.len() {
        return Err(CodecError::Truncated {
            at_bit: (i as u64) * 8,
            what: "avcC nal length",
        });
    }
    let len = usize::from(u16::from_be_bytes([extra[i], extra[i + 1]]));
    let start = i + 2;
    if start + len > extra.len() {
        return Err(CodecError::Truncated {
            at_bit: (start as u64) * 8,
            what: "avcC nal",
        });
    }
    nals.push(nal_from_ebsp(&extra[start..start + len])?);
    Ok(start + len)
}

pub fn annex_b_nal(ref_idc: u8, kind: u8, rbsp: &[u8]) -> Vec<u8> {
    let mut ebsp = emulation(rbsp);
    let mut out = vec![0, 0, 0, 1, (ref_idc << 5) | kind];
    out.append(&mut ebsp);
    out
}

pub fn length_nal(length_size: u8, ref_idc: u8, kind: u8, rbsp: &[u8]) -> Vec<u8> {
    let ebsp = emulation(rbsp);
    let len = 1 + ebsp.len();
    let mut out = Vec::with_capacity(usize::from(length_size) + len);
    for shift in (0..length_size).rev() {
        out.push((len >> (8 * shift)) as u8);
    }
    out.push((ref_idc << 5) | kind);
    out.extend_from_slice(&ebsp);
    out
}

fn emulation(rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len() + 8);
    let mut zeros = 0u8;
    for &byte in rbsp {
        if zeros >= 2 && byte <= 0x03 {
            out.push(0x03);
            zeros = 0;
        }
        out.push(byte);
        if byte == 0 {
            zeros = zeros.saturating_add(1);
        } else {
            zeros = 0;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{access_unit, rbsp};

    #[test]
    fn emulation_prevention_is_removed() {
        assert_eq!(
            rbsp(&[0x00, 0x00, 0x03, 0x01]).unwrap(),
            vec![0x00, 0x00, 0x01]
        );
    }

    #[test]
    fn annex_b_and_length_prefix_name_the_same_nal() {
        let annex = access_unit(&[0, 0, 0, 1, 0x67, 0x42], 0).unwrap();
        let length = access_unit(&[0, 0, 0, 2, 0x67, 0x42], 4).unwrap();
        assert_eq!(annex.len(), 1);
        assert_eq!(annex[0].kind, 7);
        assert_eq!(length[0].rbsp, annex[0].rbsp);
    }
}
