//! NAL units (clause 7.3.1), Annex B byte streams, length-prefixed samples,
//! and the `hvcC` configuration record (ISO/IEC 14496-15 clause 8.3.3).

use cutlass_decode::CodecError;

/// Most NAL units one access unit may hold. A real picture has a few
/// parameter sets, SEI, and at most a few hundred slice segments.
pub const MAX_NALS: usize = 4096;

pub const RASL_R: u8 = 9;
pub const BLA_W_LP: u8 = 16;
pub const BLA_N_LP: u8 = 18;
pub const IDR_W_RADL: u8 = 19;
pub const IDR_N_LP: u8 = 20;
pub const CRA: u8 = 21;
pub const SPS: u8 = 33;
pub const PPS: u8 = 34;
pub const EOS: u8 = 36;
pub const EOB: u8 = 37;

#[derive(Clone, Debug)]
pub struct Nal {
    pub kind: u8,
    pub layer: u8,
    pub temporal_id: u8,
    /// Payload after the two-byte header, with emulation prevention removed.
    pub rbsp: Vec<u8>,
    /// For each removed `0x03`, the number of payload bytes kept before it.
    pub epb: Vec<usize>,
}

impl Nal {
    pub fn is_slice(&self) -> bool {
        self.kind <= 21 && !(10..=15).contains(&self.kind)
    }

    pub fn is_irap(&self) -> bool {
        (16..=23).contains(&self.kind)
    }

    pub fn is_idr(&self) -> bool {
        self.kind == IDR_W_RADL || self.kind == IDR_N_LP
    }

    pub fn is_bla(&self) -> bool {
        (BLA_W_LP..=BLA_N_LP).contains(&self.kind)
    }

    pub fn is_rasl(&self) -> bool {
        self.kind == 8 || self.kind == RASL_R
    }

    pub fn is_radl(&self) -> bool {
        self.kind == 6 || self.kind == 7
    }

    /// A sub-layer non-reference picture (clause 3.132): an even type below 16.
    pub fn is_sub_layer_non_ref(&self) -> bool {
        self.kind <= 14 && self.kind.is_multiple_of(2)
    }

    /// Payload byte position in the coded NAL (emulation bytes included)
    /// of payload byte `rbsp` of the RBSP.
    pub fn raw_of(&self, rbsp: usize) -> usize {
        rbsp + self.epb.partition_point(|&at| at <= rbsp)
    }

    /// RBSP byte position of coded payload byte `raw`.
    pub fn rbsp_of(&self, raw: usize) -> usize {
        // Emulation byte i sits at coded position epb[i] + i.
        let before = self
            .epb
            .iter()
            .enumerate()
            .take_while(|(i, at)| *at + *i < raw)
            .count();
        raw - before
    }
}

/// Parse one NAL unit, header included.
pub fn parse(data: &[u8]) -> Result<Nal, CodecError> {
    if data.len() < 2 {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "nal header",
        });
    }
    if data[0] & 0x80 != 0 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "forbidden_zero_bit",
        });
    }
    let kind = (data[0] >> 1) & 0x3f;
    let layer = ((data[0] & 1) << 5) | (data[1] >> 3);
    let tid_plus1 = data[1] & 7;
    if tid_plus1 == 0 {
        return Err(CodecError::Malformed {
            at_bit: 13,
            what: "nuh_temporal_id_plus1",
        });
    }
    let payload = &data[2..];
    let mut rbsp = Vec::with_capacity(payload.len());
    let mut epb = Vec::new();
    let mut zeros = 0usize;
    for &byte in payload {
        if zeros >= 2 && byte == 3 {
            epb.push(rbsp.len());
            zeros = 0;
            continue;
        }
        if byte == 0 {
            zeros += 1;
        } else {
            zeros = 0;
        }
        rbsp.push(byte);
    }
    Ok(Nal {
        kind,
        layer,
        temporal_id: tid_plus1 - 1,
        rbsp,
        epb,
    })
}

/// Every NAL unit of one packet. `length_size` 0 is Annex B. A length
/// prefix that does not fit falls back to Annex B when the packet starts
/// with a start code, because a 4-byte length of 256..=511 reads the same.
pub fn access_unit(data: &[u8], length_size: u8) -> Result<Vec<Nal>, CodecError> {
    if data.is_empty() {
        return Err(CodecError::InvalidApi {
            what: "empty packet",
        });
    }
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

/// The byte ranges of the NAL units in an Annex B stream.
pub fn annex_b_ranges(data: &[u8]) -> Vec<(usize, usize)> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 2 < data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut ranges = Vec::with_capacity(starts.len());
    for (n, &start) in starts.iter().enumerate() {
        let mut end = starts.get(n + 1).map_or(data.len(), |&next| next - 3);
        while end > start && data[end - 1] == 0 {
            end -= 1;
        }
        if end > start {
            ranges.push((start, end));
        }
    }
    ranges
}

fn annex_b(data: &[u8]) -> Result<Vec<Nal>, CodecError> {
    let ranges = annex_b_ranges(data);
    if ranges.len() > MAX_NALS {
        return Err(CodecError::Limit {
            what: "nal units in one access unit",
        });
    }
    ranges
        .into_iter()
        .map(|(start, end)| parse(&data[start..end]))
        .collect()
}

fn length_prefixed(data: &[u8], length_size: u8) -> Result<Vec<Nal>, CodecError> {
    if !(1..=4).contains(&length_size) {
        return Err(CodecError::Unsupported {
            what: "nal length size",
        });
    }
    let size = usize::from(length_size);
    let mut nals = Vec::new();
    let mut at = 0usize;
    while at < data.len() {
        let Some(prefix) = data.get(at..at + size) else {
            return Err(CodecError::Truncated {
                at_bit: (at as u64) * 8,
                what: "nal length",
            });
        };
        let len = prefix
            .iter()
            .fold(0usize, |acc, &byte| (acc << 8) | usize::from(byte));
        at += size;
        let Some(body) = data.get(at..at.saturating_add(len)) else {
            return Err(CodecError::Truncated {
                at_bit: (at as u64) * 8,
                what: "nal unit",
            });
        };
        at += len;
        if len == 0 {
            continue;
        }
        if nals.len() >= MAX_NALS {
            return Err(CodecError::Limit {
                what: "nal units in one access unit",
            });
        }
        nals.push(parse(body)?);
    }
    Ok(nals)
}

/// Whether `extra` is an `hvcC` record rather than Annex B parameter sets.
pub fn is_hvcc(extra: &[u8]) -> bool {
    extra.len() >= 23 && !starts_annex_b(extra)
}

/// The length size and parameter set NAL units of an `hvcC` record.
pub fn hvcc(extra: &[u8]) -> Result<(u8, Vec<Nal>), CodecError> {
    let truncated = |at: usize| CodecError::Truncated {
        at_bit: (at as u64) * 8,
        what: "hvcC",
    };
    if extra.len() < 23 {
        return Err(truncated(extra.len()));
    }
    let length_size = (extra[21] & 3) + 1;
    let arrays = extra[22];
    let mut at = 23usize;
    let mut nals = Vec::new();
    for _ in 0..arrays {
        let Some(header) = extra.get(at..at + 3) else {
            return Err(truncated(at));
        };
        let count = usize::from(u16::from_be_bytes([header[1], header[2]]));
        at += 3;
        for _ in 0..count {
            let Some(len) = extra.get(at..at + 2) else {
                return Err(truncated(at));
            };
            let len = usize::from(u16::from_be_bytes([len[0], len[1]]));
            at += 2;
            let Some(body) = extra.get(at..at + len) else {
                return Err(truncated(at));
            };
            at += len;
            if len == 0 {
                continue;
            }
            if nals.len() >= MAX_NALS {
                return Err(CodecError::Limit {
                    what: "hvcC nal units",
                });
            }
            nals.push(parse(body)?);
        }
    }
    Ok((length_size, nals))
}

#[cfg(test)]
mod tests {
    use super::{access_unit, hvcc, parse};

    #[test]
    fn emulation_prevention_is_removed_and_remembered() {
        let nal = parse(&[0x40, 0x01, 0x00, 0x00, 0x03, 0x01, 0x00, 0x00, 0x03, 0x00]).unwrap();
        assert_eq!(nal.kind, 32);
        assert_eq!(nal.temporal_id, 0);
        assert_eq!(nal.rbsp, vec![0, 0, 1, 0, 0, 0]);
        assert_eq!(nal.epb, vec![2, 5]);
        assert_eq!(nal.raw_of(2), 3);
        assert_eq!(nal.raw_of(5), 7);
        assert_eq!(nal.rbsp_of(3), 2);
        assert_eq!(nal.rbsp_of(7), 5);
    }

    #[test]
    fn annex_b_and_length_prefixes_split_the_same_units() {
        let annex = [0, 0, 0, 1, 0x40, 0x01, 0xaa, 0, 0, 1, 0x42, 0x01, 0xbb];
        let nals = access_unit(&annex, 0).unwrap();
        assert_eq!(nals.len(), 2);
        assert_eq!(nals[1].kind, 33);
        let prefixed = [0, 3, 0x40, 0x01, 0xaa, 0, 3, 0x42, 0x01, 0xbb];
        let nals = access_unit(&prefixed, 2).unwrap();
        assert_eq!(nals.len(), 2);
        assert_eq!(nals[1].rbsp, vec![0xbb]);
    }

    #[test]
    fn a_short_length_prefix_is_truncated() {
        assert!(access_unit(&[0, 0, 0, 9, 0x40, 0x01], 4).is_err());
        assert!(access_unit(&[0x80, 0x01], 0).is_err());
        assert!(parse(&[0x40, 0x00]).is_err());
    }

    #[test]
    fn hvcc_lists_its_parameter_sets() {
        let mut record = vec![1u8; 21];
        record.push(0xff);
        record.push(1);
        record.extend([0x20, 0, 1, 0, 3, 0x40, 0x01, 0xcc]);
        let (size, nals) = hvcc(&record).unwrap();
        assert_eq!(size, 4);
        assert_eq!(nals.len(), 1);
        assert_eq!(nals[0].kind, 32);
        record.truncate(record.len() - 1);
        assert!(hvcc(&record).is_err());
    }
}
