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
    /// Header extension for NAL types 14, 20, and 21. Type 15 has none.
    pub extension: Option<Extension>,
}

/// NAL unit header extension. Clause 7.3.1. The three-byte form is Annex G
/// when `svc_extension_flag` is 1 and Annex H when it is 0. NAL type 21 with
/// `avc_3d_extension_flag` 0 is Annex I depth and is three bytes. The flag
/// equal to 1 is the two-byte 3D-AVC header.
#[derive(Clone, Copy, Debug)]
pub enum Extension {
    Svc(SvcHeader),
    /// Annex H. `svc_extension_flag` is 0.
    Mvc(MvcHeader),
    /// NAL type 21 with `avc_3d_extension_flag` 1.
    ThreeD(ThreeDHeader),
}

/// Annex H `nal_unit_header_mvc_extension`. Fields the reconstruction reads.
#[derive(Clone, Copy, Debug)]
pub struct MvcHeader {
    /// `IdrPicFlag`, the inverse of `non_idr_flag`.
    pub idr: bool,
    pub view_id: u16,
    /// `anchor_pic_flag`. Selects the anchor interview list.
    pub anchor: bool,
    /// `inter_view_flag`. This view can be an interview reference.
    pub inter_view: bool,
}

/// `nal_unit_header_3davc_extension`. `view_idx` is the view order index.
#[derive(Clone, Copy, Debug)]
pub struct ThreeDHeader {
    pub view_idx: u8,
    /// `depth_flag`. The NAL belongs to a depth view.
    pub depth: bool,
    /// `IdrPicFlag`, the inverse of `non_idr_flag`.
    pub idr: bool,
    /// `anchor_pic_flag`.
    pub anchor: bool,
    /// `inter_view_flag`. This view can be an interview reference.
    pub inter_view: bool,
}

/// `nal_unit_header_svc_extension`. Annex G. Fields the reconstruction reads.
#[derive(Clone, Copy, Debug)]
pub struct SvcHeader {
    pub idr: bool,
    pub no_inter_layer_pred: bool,
    /// `dependency_id`. With `quality_id` this is the layer's DQId.
    pub dependency_id: u8,
    pub quality_id: u8,
    pub use_ref_base_pic: bool,
    pub output: bool,
}

pub fn rbsp(ebsp: &[u8]) -> Result<Vec<u8>, CodecError> {
    if ebsp.len() > NAL_LIMIT {
        return Err(CodecError::Limit { what: "nal unit" });
    }
    // An emulation prevention byte is a 3 after two zeros. Everything
    // between two of them is copied in one piece.
    let mut out = Vec::with_capacity(ebsp.len());
    let mut kept = 0usize;
    let mut from = 0usize;
    while let Some(found) = find_three(&ebsp[from..]) {
        let at = from + found;
        if at >= 2 && ebsp[at - 1] == 0 && ebsp[at - 2] == 0 {
            out.extend_from_slice(&ebsp[kept..at]);
            kept = at + 1;
        }
        from = at + 1;
    }
    out.extend_from_slice(&ebsp[kept..]);
    Ok(out)
}

/// Where the first 3 in `bytes` is. Eight bytes are looked at a time, and
/// only a word that may hold one is looked at byte by byte: in a slice's
/// data a 3 is rare.
fn find_three(bytes: &[u8]) -> Option<usize> {
    const ONES: u64 = 0x0101_0101_0101_0101;
    const HIGHS: u64 = 0x8080_8080_8080_8080;
    let (words, rest) = bytes.as_chunks::<8>();
    for (index, word) in words.iter().enumerate() {
        // A byte of the word that is 3 is a zero byte here, which sets its
        // high bit below; other bytes may too, so they are looked at again.
        let moved = u64::from_le_bytes(*word) ^ (3 * ONES);
        if moved.wrapping_sub(ONES) & !moved & HIGHS != 0
            && let Some(found) = word.iter().position(|&byte| byte == 3)
        {
            return Some(index * 8 + found);
        }
    }
    rest.iter()
        .position(|&byte| byte == 3)
        .map(|found| words.len() * 8 + found)
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
    let (extension, header_bytes) = if matches!(kind, 14 | 20 | 21) {
        let (extension, len) = extension_header(kind, &ebsp[1..])?;
        (Some(extension), len)
    } else {
        (None, 0)
    };
    Ok(Nal {
        ref_idc: (header >> 5) & 0x3,
        kind,
        rbsp: rbsp(&ebsp[1 + header_bytes..])?,
        extension,
    })
}

/// The extension bytes are part of the NAL header. Emulation prevention
/// starts after them, so a `00 00 03` inside the header stays.
fn extension_header(kind: u8, bytes: &[u8]) -> Result<(Extension, usize), CodecError> {
    let Some(first) = bytes.first().copied() else {
        return Err(CodecError::Truncated {
            at_bit: 8,
            what: "extension header",
        });
    };
    let flag = first & 0x80 != 0;
    let three_d = kind == 21 && flag;
    let len = if three_d { 2 } else { 3 };
    if bytes.len() < len {
        return Err(CodecError::Truncated {
            at_bit: 8,
            what: "extension header",
        });
    }
    let header = &bytes[..len];
    if three_d {
        return Ok((Extension::ThreeD(three_d_header(header)), len));
    }
    if kind != 21 && flag {
        Ok((Extension::Svc(svc_header(header)?), len))
    } else {
        Ok((Extension::Mvc(mvc_header(header)?), len))
    }
}

fn three_d_header(bytes: &[u8]) -> ThreeDHeader {
    let bits = (u16::from(bytes[0]) << 8) | u16::from(bytes[1]);
    ThreeDHeader {
        view_idx: ((bits >> 7) & 0xff) as u8,
        depth: bits & 0x40 != 0,
        idr: bits & 0x20 == 0,
        anchor: bits & 0x02 != 0,
        inter_view: bits & 0x01 != 0,
    }
}

fn svc_header(bytes: &[u8]) -> Result<SvcHeader, CodecError> {
    let b0 = bytes[0];
    let b1 = bytes[1];
    let b2 = bytes[2];
    if b2 & 0x03 != 0x03 {
        return Err(CodecError::Malformed {
            at_bit: 8,
            what: "extension header",
        });
    }
    Ok(SvcHeader {
        idr: b0 & 0x40 != 0,
        no_inter_layer_pred: b1 & 0x80 != 0,
        dependency_id: (b1 >> 4) & 7,
        quality_id: b1 & 0x0f,
        use_ref_base_pic: b2 & 0x10 != 0,
        output: b2 & 0x04 != 0,
    })
}

fn mvc_header(bytes: &[u8]) -> Result<MvcHeader, CodecError> {
    let b0 = bytes[0];
    let b1 = bytes[1];
    let b2 = bytes[2];
    if b2 & 0x01 != 0x01 {
        return Err(CodecError::Malformed {
            at_bit: 8,
            what: "extension header",
        });
    }
    Ok(MvcHeader {
        idr: b0 & 0x40 == 0,
        view_id: (u16::from(b1) << 2) | u16::from(b2 >> 6),
        anchor: b2 & 0x04 != 0,
        inter_view: b2 & 0x02 != 0,
    })
}

/// Annex B access unit, or one length-prefixed access unit.
pub fn access_unit(data: &[u8], length_size: u8) -> Result<Vec<Nal>, CodecError> {
    if data.is_empty() {
        return Err(CodecError::InvalidApi {
            what: "empty packet",
        });
    }
    // A 4-byte length in 256..=511 begins with 00 00 01, the same bytes as an
    // Annex B start code. A length that accounts for the whole buffer wins.
    // Annex B remains the fallback when that length does not fit.
    if length_size != 0 {
        match length_prefixed(data, length_size) {
            Ok(nals) => return Ok(nals),
            Err(error) if !starts_annex_b(data) => return Err(error),
            Err(_) => {}
        }
    }
    if !starts_annex_b(data) {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "access unit framing",
        });
    }
    annex_b(data)
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

/// Annex B NAL whose header is longer than one byte. `header` is not
/// escaped. Emulation prevention applies to `rbsp` only.
#[cfg(test)]
pub(crate) fn annex_b_extension(ref_idc: u8, kind: u8, header: &[u8], rbsp: &[u8]) -> Vec<u8> {
    let mut out = vec![0, 0, 0, 1, (ref_idc << 5) | (kind & 0x1f)];
    out.extend_from_slice(header);
    out.extend(emulation(rbsp));
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
    #[test]
    fn find_three_finds_the_first_three_anywhere() {
        let mut seed = 7u32;
        for round in 0..5_000 {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let len = (seed >> 8) as usize % 40;
            let bytes: Vec<u8> = (0..len)
                .map(|i| {
                    let mix = seed.wrapping_mul(i as u32 + 1).rotate_left(i as u32);
                    [0, 3, 2, 4, 0x83, 0x03 ^ 0x80][(mix >> 3) as usize % 6]
                })
                .collect();
            let wanted = bytes.iter().position(|&byte| byte == 3);
            assert_eq!(
                super::find_three(&bytes),
                wanted,
                "round {round}: {bytes:?}"
            );
        }
    }

    use super::{Extension, access_unit, rbsp};

    #[test]
    fn emulation_prevention_is_removed() {
        assert_eq!(
            rbsp(&[0x00, 0x00, 0x03, 0x01]).unwrap(),
            vec![0x00, 0x00, 0x01]
        );
    }

    #[test]
    fn rbsp_drops_exactly_the_bytes_the_rule_names() {
        // Clause 7.4.1: a 3 that follows two zero bytes is dropped, and the
        // count of zeros starts again after it.
        fn by_the_rule(ebsp: &[u8]) -> Vec<u8> {
            let mut out = Vec::new();
            let mut zeros = 0u8;
            for &byte in ebsp {
                if zeros >= 2 && byte == 0x03 {
                    zeros = 0;
                    continue;
                }
                out.push(byte);
                zeros = if byte == 0 {
                    zeros.saturating_add(1)
                } else {
                    0
                };
            }
            out
        }
        // Every string of up to seven bytes over 0, 1 and 3, then long ones.
        for len in 0..=7u32 {
            for code in 0..3u32.pow(len) {
                let bytes: Vec<u8> = (0..len)
                    .map(|i| [0, 1, 3][(code / 3u32.pow(i) % 3) as usize])
                    .collect();
                assert_eq!(rbsp(&bytes).unwrap(), by_the_rule(&bytes), "{bytes:?}");
            }
        }
        let mut state = 9u32;
        for _ in 0..200 {
            let bytes: Vec<u8> = (0..300)
                .map(|_| {
                    state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    [0, 0, 0, 3, 3, 1, 255][(state >> 16) as usize % 7]
                })
                .collect();
            assert_eq!(rbsp(&bytes).unwrap(), by_the_rule(&bytes));
        }
    }

    #[test]
    fn an_extension_header_is_not_part_of_the_rbsp() {
        let svc = access_unit(&[0, 0, 0, 1, 0x74, 0xc0, 0x80, 0x07, 0x80], 0).unwrap();
        let Extension::Svc(header) = svc[0].extension.unwrap() else {
            panic!("svc");
        };
        assert!(header.idr && header.no_inter_layer_pred && header.output);
        assert_eq!(header.quality_id, 0);
        assert_eq!(svc[0].rbsp, vec![0x80]);

        // `00 00 03` is a legal MVC header. It must not be stripped as emulation.
        let mvc = access_unit(&[0, 0, 0, 1, 0x74, 0x00, 0x00, 0x03, 0x80], 0).unwrap();
        assert!(matches!(mvc[0].extension, Some(Extension::Mvc(_))));
        assert_eq!(mvc[0].rbsp, vec![0x80]);
    }

    #[test]
    fn a_short_extension_header_is_truncated() {
        let err = access_unit(&[0, 0, 0, 1, 0x74], 0).unwrap_err();
        assert!(matches!(
            err,
            cutlass_decode::CodecError::Truncated {
                what: "extension header",
                ..
            }
        ));
    }

    #[test]
    fn annex_b_and_length_prefix_name_the_same_nal() {
        let annex = access_unit(&[0, 0, 0, 1, 0x67, 0x42], 0).unwrap();
        let length = access_unit(&[0, 0, 0, 2, 0x67, 0x42], 4).unwrap();
        assert_eq!(annex.len(), 1);
        assert_eq!(annex[0].kind, 7);
        assert_eq!(length[0].rbsp, annex[0].rbsp);
    }

    #[test]
    fn a_length_of_384_is_not_an_annex_b_start() {
        let mut payload = vec![0x21];
        payload.resize(0x180, 0x11);
        let mut data = 0x180_u32.to_be_bytes().to_vec();
        data.extend(payload);
        let nals = access_unit(&data, 4).expect("length prefix");
        assert_eq!(nals.len(), 1);
        assert_eq!(nals[0].kind, 1);
        assert_eq!(nals[0].rbsp, vec![0x11; 0x17f]);
    }
}
