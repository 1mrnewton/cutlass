//! The identification and comment headers (Vorbis I sections 4.2.1 to
//! 4.2.3, 5) and the codec configuration: the three header packets joined
//! with Xiph lacing, as Matroska `CodecPrivate` and FFmpeg's extradata
//! carry them.

use crate::bits::BitReader;
#[cfg(test)]
use crate::bits::BitWriter;
use cutlass_decode::CodecError;

/// The largest configuration this crate takes. libvorbis setups are under
/// 8 KiB; a comment header with cover art can be larger.
pub const MAX_CONFIG_BYTES: usize = 16 << 20;
/// Channels the decoder outputs (the `ChannelLayout` limit).
pub const MAX_CHANNELS: usize = 8;

pub(crate) const ID_HEADER: u8 = 1;
pub(crate) const COMMENT_HEADER: u8 = 3;
pub(crate) const SETUP_HEADER: u8 = 5;
const SIGNATURE: &[u8; 6] = b"vorbis";
const ID_HEADER_BYTES: usize = 30;

/// The identification header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Identification {
    pub channels: u8,
    pub sample_rate: u32,
    pub bitrate_maximum: i32,
    pub bitrate_nominal: i32,
    pub bitrate_minimum: i32,
    /// Short and long block sizes in samples, 64 to 8192.
    pub blocksizes: [usize; 2],
}

impl Identification {
    /// Parses an identification header packet.
    ///
    /// # Errors
    ///
    /// `Malformed` for a wrong packet type, signature, version, zero
    /// channels or rate, a block size outside 64 to 8192 or a short block
    /// longer than the long one, or a missing framing bit. `Truncated` when
    /// the packet ends early.
    pub fn parse(packet: &[u8]) -> Result<Self, CodecError> {
        common_header(packet, ID_HEADER, "identification header")?;
        if packet.len() < ID_HEADER_BYTES {
            return Err(CodecError::Truncated {
                at_bit: packet.len() as u64 * 8,
                what: "identification header",
            });
        }
        let word = |at: usize| {
            u32::from_le_bytes([packet[at], packet[at + 1], packet[at + 2], packet[at + 3]])
        };
        if word(7) != 0 {
            return Err(malformed(56, "Vorbis version"));
        }
        let channels = packet[11];
        if channels == 0 {
            return Err(malformed(88, "zero channels"));
        }
        let sample_rate = word(12);
        if sample_rate == 0 {
            return Err(malformed(96, "zero sample rate"));
        }
        let sizes = packet[28];
        let (short, long) = (u32::from(sizes & 0x0f), u32::from(sizes >> 4));
        if !(6..=13).contains(&short) || !(6..=13).contains(&long) || short > long {
            return Err(malformed(224, "block sizes"));
        }
        if packet[29] & 1 == 0 {
            return Err(malformed(232, "identification framing bit"));
        }
        Ok(Self {
            channels,
            sample_rate,
            bitrate_maximum: word(16) as i32,
            bitrate_nominal: word(20) as i32,
            bitrate_minimum: word(24) as i32,
            blocksizes: [1 << short, 1 << long],
        })
    }

    /// The identification header packet.
    #[must_use]
    pub fn write(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(ID_HEADER_BYTES);
        out.push(ID_HEADER);
        out.extend_from_slice(SIGNATURE);
        out.extend_from_slice(&0_u32.to_le_bytes());
        out.push(self.channels);
        out.extend_from_slice(&self.sample_rate.to_le_bytes());
        out.extend_from_slice(&self.bitrate_maximum.to_le_bytes());
        out.extend_from_slice(&self.bitrate_nominal.to_le_bytes());
        out.extend_from_slice(&self.bitrate_minimum.to_le_bytes());
        let log = |size: usize| size.trailing_zeros() as u8;
        out.push(log(self.blocksizes[0]) | (log(self.blocksizes[1]) << 4));
        out.push(1);
        out
    }
}

/// The comment header: a vendor string and `NAME=value` fields.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Comments {
    pub vendor: String,
    pub fields: Vec<String>,
}

impl Comments {
    /// Parses a comment header packet. Strings that are not UTF-8 are
    /// taken lossily.
    ///
    /// # Errors
    ///
    /// `Malformed` for a wrong type or signature or a missing framing bit,
    /// `Truncated` when a length runs past the packet.
    pub fn parse(packet: &[u8]) -> Result<Self, CodecError> {
        common_header(packet, COMMENT_HEADER, "comment header")?;
        let mut at = 7;
        let mut take = |what: &'static str| -> Result<&[u8], CodecError> {
            let len = read_u32(packet, at, what)? as usize;
            let start = at + 4;
            let end = start
                .checked_add(len)
                .filter(|end| *end <= packet.len())
                .ok_or(CodecError::Truncated {
                    at_bit: start as u64 * 8,
                    what,
                })?;
            at = end;
            Ok(&packet[start..end])
        };
        let vendor = String::from_utf8_lossy(take("comment vendor")?).into_owned();
        let count = read_u32(packet, at, "comment count")?;
        at += 4;
        // Each field takes at least its four length bytes.
        if u64::from(count) * 4 > (packet.len() - at) as u64 {
            return Err(CodecError::Truncated {
                at_bit: at as u64 * 8,
                what: "comment fields",
            });
        }
        let mut fields = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let len = read_u32(packet, at, "comment field")? as usize;
            let start = at + 4;
            let end = start
                .checked_add(len)
                .filter(|end| *end <= packet.len())
                .ok_or(CodecError::Truncated {
                    at_bit: start as u64 * 8,
                    what: "comment field",
                })?;
            fields.push(String::from_utf8_lossy(&packet[start..end]).into_owned());
            at = end;
        }
        match packet.get(at) {
            Some(byte) if byte & 1 == 1 => Ok(Self { vendor, fields }),
            Some(_) => Err(malformed(at as u64 * 8, "comment framing bit")),
            None => Err(CodecError::Truncated {
                at_bit: at as u64 * 8,
                what: "comment framing bit",
            }),
        }
    }

    /// The comment header packet.
    #[must_use]
    pub fn write(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(COMMENT_HEADER);
        out.extend_from_slice(SIGNATURE);
        push_string(&mut out, self.vendor.as_bytes());
        out.extend_from_slice(&(self.fields.len() as u32).to_le_bytes());
        for field in &self.fields {
            push_string(&mut out, field.as_bytes());
        }
        out.push(1);
        out
    }
}

fn push_string(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}

fn read_u32(packet: &[u8], at: usize, what: &'static str) -> Result<u32, CodecError> {
    let bytes = packet.get(at..at + 4).ok_or(CodecError::Truncated {
        at_bit: at as u64 * 8,
        what,
    })?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

/// Checks the packet type byte and the `vorbis` signature.
pub(crate) fn common_header(packet: &[u8], kind: u8, what: &'static str) -> Result<(), CodecError> {
    if packet.len() < 7 {
        return Err(CodecError::Truncated {
            at_bit: packet.len() as u64 * 8,
            what,
        });
    }
    if packet[0] != kind {
        return Err(malformed(0, what));
    }
    if &packet[1..7] != SIGNATURE {
        return Err(malformed(8, "Vorbis signature"));
    }
    Ok(())
}

pub(crate) fn malformed(at_bit: u64, what: &'static str) -> CodecError {
    CodecError::Malformed { at_bit, what }
}

/// The three header packets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Headers {
    pub identification: Vec<u8>,
    pub comment: Vec<u8>,
    pub setup: Vec<u8>,
}

impl Headers {
    /// Splits a configuration blob into its three header packets.
    ///
    /// Takes Xiph lacing (`0x02`, two laced lengths, then the packets: the
    /// Matroska and FFmpeg form), three 16-bit big-endian lengths each
    /// before its packet (the form some MP4 writers use), and the three
    /// packets back to back with no lengths (split by reading the comment
    /// header's own lengths).
    ///
    /// # Errors
    ///
    /// `Limit` above [`MAX_CONFIG_BYTES`], `Malformed` or `Truncated` when
    /// none of the forms fits.
    pub fn split(config: &[u8]) -> Result<Self, CodecError> {
        if config.len() > MAX_CONFIG_BYTES {
            return Err(CodecError::Limit {
                what: "Vorbis configuration",
            });
        }
        if config.first() == Some(&2)
            && let Some(headers) = Self::split_xiph(config)
        {
            return Ok(headers);
        }
        if config.len() >= 6
            && config[0] == 0
            && usize::from(config[1]) == ID_HEADER_BYTES
            && let Some(headers) = Self::split_sixteen(config)
        {
            return Ok(headers);
        }
        if config.first() == Some(&ID_HEADER) {
            return Self::split_plain(config);
        }
        Err(malformed(0, "Vorbis configuration"))
    }

    fn split_xiph(config: &[u8]) -> Option<Self> {
        let mut at = 1;
        let mut lengths = [0_usize; 2];
        for length in &mut lengths {
            loop {
                let byte = *config.get(at)?;
                at += 1;
                *length = length.checked_add(usize::from(byte))?;
                if byte != 255 {
                    break;
                }
            }
        }
        let first_end = at.checked_add(lengths[0])?;
        let second_end = first_end.checked_add(lengths[1])?;
        if second_end > config.len() {
            return None;
        }
        let headers = Self {
            identification: config[at..first_end].to_vec(),
            comment: config[first_end..second_end].to_vec(),
            setup: config[second_end..].to_vec(),
        };
        headers.has_types().then_some(headers)
    }

    fn split_sixteen(config: &[u8]) -> Option<Self> {
        let mut at = 0;
        let mut parts = Vec::with_capacity(3);
        for _ in 0..3 {
            let len = usize::from(u16::from_be_bytes([*config.get(at)?, *config.get(at + 1)?]));
            let start = at + 2;
            let end = start.checked_add(len)?;
            parts.push(config.get(start..end)?.to_vec());
            at = end;
        }
        let setup = parts.pop()?;
        let comment = parts.pop()?;
        let identification = parts.pop()?;
        let headers = Self {
            identification,
            comment,
            setup,
        };
        headers.has_types().then_some(headers)
    }

    fn split_plain(config: &[u8]) -> Result<Self, CodecError> {
        let id_end = ID_HEADER_BYTES.min(config.len());
        let rest = &config[id_end..];
        let comment_len = comment_length(rest)?;
        let headers = Self {
            identification: config[..id_end].to_vec(),
            comment: rest[..comment_len].to_vec(),
            setup: rest[comment_len..].to_vec(),
        };
        if headers.has_types() {
            Ok(headers)
        } else {
            Err(malformed(0, "Vorbis configuration"))
        }
    }

    fn has_types(&self) -> bool {
        self.identification.first() == Some(&ID_HEADER)
            && self.comment.first() == Some(&COMMENT_HEADER)
            && self.setup.first() == Some(&SETUP_HEADER)
    }

    /// The Xiph-laced configuration blob.
    #[must_use]
    pub fn join(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            3 + self.identification.len() + self.comment.len() + self.setup.len() + 8,
        );
        out.push(2);
        for len in [self.identification.len(), self.comment.len()] {
            let mut left = len;
            while left >= 255 {
                out.push(255);
                left -= 255;
            }
            out.push(left as u8);
        }
        out.extend_from_slice(&self.identification);
        out.extend_from_slice(&self.comment);
        out.extend_from_slice(&self.setup);
        out
    }
}

/// Bytes of the comment header at the start of `bytes`, framing byte
/// included.
fn comment_length(bytes: &[u8]) -> Result<usize, CodecError> {
    common_header(bytes, COMMENT_HEADER, "comment header")?;
    let mut at = 7;
    let vendor = read_u32(bytes, at, "comment vendor")? as usize;
    at = at
        .checked_add(4 + vendor)
        .ok_or(malformed(56, "comment vendor"))?;
    let count = read_u32(bytes, at, "comment count")?;
    at += 4;
    for _ in 0..count {
        let len = read_u32(bytes, at, "comment field")? as usize;
        at = at
            .checked_add(4 + len)
            .filter(|end| *end <= bytes.len())
            .ok_or(CodecError::Truncated {
                at_bit: at as u64 * 8,
                what: "comment field",
            })?;
    }
    if at >= bytes.len() {
        return Err(CodecError::Truncated {
            at_bit: at as u64 * 8,
            what: "comment framing bit",
        });
    }
    Ok(at + 1)
}

/// Reads the bit after the common header of a setup packet onward.
pub(crate) fn setup_reader(packet: &[u8]) -> Result<BitReader<'_>, CodecError> {
    common_header(packet, SETUP_HEADER, "setup header")?;
    Ok(BitReader::new(&packet[7..]))
}

#[cfg(test)]
/// Starts a setup packet: the type byte and the signature.
pub(crate) fn setup_writer() -> BitWriter {
    let mut writer = BitWriter::new();
    writer.write(u32::from(SETUP_HEADER), 8);
    writer.write_bytes(SIGNATURE);
    writer
}

#[cfg(test)]
mod tests {
    use super::{Comments, Headers, Identification};
    use cutlass_decode::CodecError;

    fn id() -> Identification {
        Identification {
            channels: 2,
            sample_rate: 44_100,
            bitrate_maximum: 0,
            bitrate_nominal: 128_000,
            bitrate_minimum: 0,
            blocksizes: [256, 2048],
        }
    }

    #[test]
    fn the_identification_header_round_trips() {
        let packet = id().write();
        assert_eq!(packet.len(), 30);
        assert_eq!(packet[28], 0xb8);
        assert_eq!(Identification::parse(&packet).unwrap(), id());
    }

    #[test]
    fn bad_identification_fields_are_refused() {
        let good = id().write();
        let cases: [(usize, u8); 6] = [(0, 3), (7, 1), (11, 0), (28, 0x85), (28, 0xe6), (29, 0)];
        for (at, value) in cases {
            let mut packet = good.clone();
            packet[at] = value;
            assert!(
                matches!(
                    Identification::parse(&packet),
                    Err(CodecError::Malformed { .. })
                ),
                "byte {at} = {value}"
            );
        }
        let mut zero_rate = good.clone();
        zero_rate[12..16].fill(0);
        assert!(Identification::parse(&zero_rate).is_err());
        assert!(matches!(
            Identification::parse(&good[..29]),
            Err(CodecError::Truncated { .. })
        ));
    }

    #[test]
    fn comments_round_trip_and_bounds_are_checked() {
        let comments = Comments {
            vendor: "Cutlass".into(),
            fields: vec!["TITLE=Rain".into(), "ARTIST=".into()],
        };
        let packet = comments.write();
        assert_eq!(Comments::parse(&packet).unwrap(), comments);
        // A field count far past the packet fails before allocating.
        let mut hostile = Comments::default().write();
        let at = hostile.len() - 5;
        hostile[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            Comments::parse(&hostile),
            Err(CodecError::Truncated { .. })
        ));
        assert!(matches!(
            Comments::parse(&packet[..packet.len() - 1]),
            Err(CodecError::Truncated { .. })
        ));
    }

    fn headers() -> Headers {
        let mut setup = vec![5];
        setup.extend_from_slice(b"vorbis");
        setup.extend(std::iter::repeat_n(0xa5, 600));
        Headers {
            identification: id().write(),
            comment: Comments {
                vendor: "v".repeat(300),
                fields: Vec::new(),
            }
            .write(),
            setup,
        }
    }

    #[test]
    fn every_configuration_form_splits() {
        let headers = headers();
        let xiph = headers.join();
        assert_eq!(xiph[0], 2);
        assert_eq!(Headers::split(&xiph).unwrap(), headers);
        let mut sixteen = Vec::new();
        for part in [&headers.identification, &headers.comment, &headers.setup] {
            sixteen.extend_from_slice(&(part.len() as u16).to_be_bytes());
            sixteen.extend_from_slice(part);
        }
        assert_eq!(Headers::split(&sixteen).unwrap(), headers);
        let plain = [
            headers.identification.clone(),
            headers.comment.clone(),
            headers.setup.clone(),
        ]
        .concat();
        assert_eq!(Headers::split(&plain).unwrap(), headers);
    }

    #[test]
    fn a_broken_configuration_is_refused() {
        let xiph = headers().join();
        assert!(Headers::split(&xiph[..20]).is_err());
        assert!(Headers::split(&[]).is_err());
        assert!(Headers::split(&[2, 255, 255, 255]).is_err());
        let mut wrong = xiph.clone();
        wrong[3] = 3;
        assert!(Headers::split(&wrong).is_err());
    }
}
