//! The Opus mapping's header packets and packet durations (RFC 7845).
//!
//! The container needs three things from Opus: the identification header
//! (channel count, pre-skip, and stream count for the packet size bound),
//! the comment header (kept for the caller, never returned as audio), and
//! each audio packet's sample count from its TOC byte (RFC 6716 section 3.1)
//! for the granule arithmetic. Decoding stays in the Opus codec crate.

use crate::error::OggError;
use crate::limits::MAX_COMMENTS;

/// Magic of the identification header.
pub const HEAD_MAGIC: &[u8; 8] = b"OpusHead";
/// Magic of the comment header.
pub const TAGS_MAGIC: &[u8; 8] = b"OpusTags";

/// Fields of `OpusHead` (RFC 7845 section 5.1), little-endian in the file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpusHead {
    /// The version byte. The upper four bits are the major version, 0 here.
    pub version: u8,
    pub channels: u8,
    /// Samples at 48 kHz to decode and discard at the start of the stream.
    pub pre_skip: u16,
    /// The rate the encoder was fed. Informational only.
    pub input_rate: u32,
    /// Q7.8 dB.
    pub output_gain: i16,
    pub mapping_family: u8,
    /// Opus streams in each Ogg packet. 1 for mapping family 0.
    pub streams: u8,
    pub coupled: u8,
    /// Output channel to decoded channel, or 255 for silence.
    pub mapping: Vec<u8>,
}

impl OpusHead {
    /// Reads and validates `OpusHead`. `serial` names the stream in the error.
    pub fn parse(bytes: &[u8], serial: u32) -> Result<Self, OggError> {
        let bad = |reason| OggError::BadOpusHead { serial, reason };
        if !bytes.starts_with(HEAD_MAGIC) {
            return Err(bad("first packet is not OpusHead"));
        }
        if bytes.len() < 19 {
            return Err(bad("shorter than 19 bytes"));
        }
        let version = bytes[8];
        if version & 0xF0 != 0 {
            return Err(OggError::UnsupportedOpusVersion { serial, version });
        }
        let channels = bytes[9];
        if channels == 0 {
            return Err(bad("zero output channels"));
        }
        let pre_skip = u16::from_le_bytes([bytes[10], bytes[11]]);
        let input_rate = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        let output_gain = i16::from_le_bytes([bytes[16], bytes[17]]);
        let mapping_family = bytes[18];
        let (streams, coupled, mapping) = if mapping_family == 0 {
            if channels > 2 {
                return Err(bad("mapping family 0 with more than two channels"));
            }
            let mapping = if channels == 2 { vec![0, 1] } else { vec![0] };
            (1, channels - 1, mapping)
        } else {
            let table_end = 21 + usize::from(channels);
            let Some(table) = bytes.get(21..table_end) else {
                return Err(bad("channel mapping table is cut short"));
            };
            let streams = bytes[19];
            let coupled = bytes[20];
            if streams == 0 {
                return Err(bad("zero streams"));
            }
            if coupled > streams {
                return Err(bad("more coupled streams than streams"));
            }
            let decoded = u16::from(streams) + u16::from(coupled);
            if decoded > 255 {
                return Err(bad("more than 255 decoded channels"));
            }
            if table
                .iter()
                .any(|&entry| entry != 255 && u16::from(entry) >= decoded)
            {
                return Err(bad("channel mapping names a missing channel"));
            }
            (streams, coupled, table.to_vec())
        };
        Ok(Self {
            version,
            channels,
            pre_skip,
            input_rate,
            output_gain,
            mapping_family,
            streams,
            coupled,
            mapping,
        })
    }
}

/// Fields of `OpusTags` (RFC 7845 section 5.2).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct OpusTags {
    /// Vendor string. Bytes that are not UTF-8 are replaced.
    pub vendor: String,
    /// `NAME=value` comments in file order. Bytes that are not UTF-8 are replaced.
    pub comments: Vec<String>,
    /// The packet was larger than the header bound. Comments past the bound
    /// were ignored, as RFC 7845 section 5.2 allows.
    pub cut: bool,
}

impl OpusTags {
    /// Reads `OpusTags`. `cut` says the packet was cut at the header bound,
    /// in which case a comment that runs past the kept bytes ends the list
    /// instead of failing it.
    pub fn parse(bytes: &[u8], serial: u32, cut: bool) -> Result<Self, OggError> {
        let bad = |reason| OggError::BadOpusTags { serial, reason };
        if !bytes.starts_with(TAGS_MAGIC) {
            return Err(bad("second packet is not OpusTags"));
        }
        let (vendor, comments, _) = read_comment_list(bytes, TAGS_MAGIC.len(), cut, bad)?;
        Ok(Self {
            vendor,
            comments,
            cut,
        })
    }

    /// Serializes the comment header, with no trailing padding.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = TAGS_MAGIC.to_vec();
        push_string(&mut out, self.vendor.as_bytes());
        out.extend_from_slice(
            &u32::try_from(self.comments.len())
                .unwrap_or(u32::MAX)
                .to_le_bytes(),
        );
        for comment in &self.comments {
            push_string(&mut out, comment.as_bytes());
        }
        out
    }
}

/// The vendor string and comments that follow a comment header's magic at
/// `at`, and the offset after the last comment. The list is the same in
/// `OpusTags` (RFC 7845 section 5.2) and the Vorbis comment header (Vorbis
/// I section 5). With `cut`, a comment that runs past the bytes ends the
/// list.
pub(crate) fn read_comment_list(
    bytes: &[u8],
    mut at: usize,
    cut: bool,
    bad: impl Fn(&'static str) -> OggError,
) -> Result<(String, Vec<String>, usize), OggError> {
    let vendor_len = read_len(bytes, at).ok_or_else(|| bad("no vendor length"))?;
    at += 4;
    let Some(vendor) = take(bytes, at, vendor_len) else {
        return Err(bad("vendor string runs past the packet"));
    };
    at += vendor.len();
    let vendor = String::from_utf8_lossy(vendor).into_owned();
    let count = read_len(bytes, at).ok_or_else(|| bad("no comment count"))?;
    at += 4;
    let remaining = bytes.len() - at;
    // Each comment needs at least its 4-byte length.
    if !cut && count > remaining / 4 {
        return Err(bad("comment count needs more bytes than the packet has"));
    }
    if count > usize::try_from(MAX_COMMENTS).unwrap_or(usize::MAX) {
        return Err(OggError::TooLarge {
            declared: u64::try_from(count).unwrap_or(u64::MAX),
            limit: u64::from(MAX_COMMENTS),
        });
    }
    let mut comments = Vec::with_capacity(count.min(remaining / 4));
    for _ in 0..count {
        let Some(len) = read_len(bytes, at) else {
            if cut {
                break;
            }
            return Err(bad("comment length runs past the packet"));
        };
        let Some(text) = take(bytes, at + 4, len) else {
            if cut {
                break;
            }
            return Err(bad("comment runs past the packet"));
        };
        at += 4 + text.len();
        comments.push(String::from_utf8_lossy(text).into_owned());
    }
    Ok((vendor, comments, at))
}

pub(crate) fn push_string(out: &mut Vec<u8>, text: &[u8]) {
    out.extend_from_slice(&u32::try_from(text.len()).unwrap_or(u32::MAX).to_le_bytes());
    out.extend_from_slice(text);
}

fn read_len(bytes: &[u8], at: usize) -> Option<usize> {
    let field = bytes.get(at..at.checked_add(4)?)?;
    usize::try_from(u32::from_le_bytes([field[0], field[1], field[2], field[3]])).ok()
}

fn take(bytes: &[u8], at: usize, len: usize) -> Option<&[u8]> {
    bytes.get(at..at.checked_add(len)?)
}

/// Samples at 48 kHz in one Opus packet, from its TOC byte and, for code 3,
/// its frame count byte (RFC 6716 section 3.1 and 3.2.5).
///
/// `None` for a packet with no TOC byte, a code 3 packet with no count byte
/// or a zero count, or a total above 120 ms: RFC 6716 section 3.4 calls
/// those malformed. In a multistream Ogg packet the first Opus packet is
/// self-delimited, and its TOC and count bytes sit at the same offsets.
#[must_use]
pub fn packet_samples(packet: &[u8]) -> Option<u32> {
    let toc = *packet.first()?;
    let config = usize::from(toc >> 3);
    let frame: u32 = match config {
        0..=11 => [480, 960, 1920, 2880][config & 3],
        12..=15 => [480, 960][config & 1],
        _ => [120, 240, 480, 960][config & 3],
    };
    let count = match toc & 3 {
        0 => 1,
        1 | 2 => 2,
        _ => {
            let count = u32::from(*packet.get(1)? & 0x3F);
            if count == 0 {
                return None;
            }
            count
        }
    };
    let total = frame * count;
    if total > 5_760 {
        return None;
    }
    Some(total)
}

/// A short tag naming a stream mapping this crate does not read, from the
/// first bytes of its beginning-of-stream packet.
#[must_use]
pub fn mapping_tag(bos: &[u8]) -> Vec<u8> {
    const KNOWN: &[(&[u8], &str)] = &[
        (b"\x7fFLAC", "ogg:flac"),
        (b"fLaC", "ogg:flac"),
        (b"\x80theora", "ogg:theora"),
        (b"Speex   ", "ogg:speex"),
        (b"fishead\0", "ogg:skeleton"),
        (b"fisbone\0", "ogg:skeleton"),
        (b"OVP80", "ogg:vp8"),
        (b"\x80kate\0\0\0", "ogg:kate"),
        (b"CELT    ", "ogg:celt"),
        (b"PCM     ", "ogg:pcm"),
        (b"BBCD\0", "ogg:dirac"),
        (b"\x01video\0\0\0", "ogg:ogm-video"),
        (b"\x01audio\0\0\0", "ogg:ogm-audio"),
        (b"\x01text\0\0\0", "ogg:ogm-text"),
    ];
    for (magic, tag) in KNOWN {
        if bos.starts_with(magic) {
            return tag.as_bytes().to_vec();
        }
    }
    let mut tag = b"ogg:".to_vec();
    for byte in bos.iter().take(8) {
        tag.extend_from_slice(format!("{byte:02x}").as_bytes());
    }
    tag
}

#[cfg(test)]
mod tests {
    use super::{OpusHead, OpusTags, mapping_tag, packet_samples};
    use crate::error::OggError;

    fn head(channels: u8, family: u8, tail: &[u8]) -> Vec<u8> {
        let mut bytes = b"OpusHead".to_vec();
        bytes.push(1);
        bytes.push(channels);
        bytes.extend_from_slice(&312_u16.to_le_bytes());
        bytes.extend_from_slice(&44_100_u32.to_le_bytes());
        bytes.extend_from_slice(&(-256_i16).to_le_bytes());
        bytes.push(family);
        bytes.extend_from_slice(tail);
        bytes
    }

    #[test]
    fn a_stereo_head_reads_its_fields() {
        let parsed = OpusHead::parse(&head(2, 0, &[]), 1).expect("head");
        assert_eq!(parsed.channels, 2);
        assert_eq!(parsed.pre_skip, 312);
        assert_eq!(parsed.input_rate, 44_100);
        assert_eq!(parsed.output_gain, -256);
        assert_eq!((parsed.streams, parsed.coupled), (1, 1));
        assert_eq!(parsed.mapping, vec![0, 1]);
    }

    #[test]
    fn a_surround_head_reads_its_table() {
        let parsed = OpusHead::parse(&head(6, 1, &[4, 2, 0, 4, 1, 2, 3, 5]), 1).expect("5.1 head");
        assert_eq!((parsed.streams, parsed.coupled), (4, 2));
        assert_eq!(parsed.mapping, vec![0, 4, 1, 2, 3, 5]);
    }

    #[test]
    fn a_broken_head_is_refused_with_a_reason() {
        let cases: Vec<Vec<u8>> = vec![
            b"OpusTags".to_vec(),
            b"OpusHead\x01\x02".to_vec(),
            head(0, 0, &[]),
            head(3, 0, &[]),
            head(6, 1, &[4, 2, 0, 4]),
            head(2, 1, &[0, 0, 0, 1]),
            head(2, 1, &[1, 2, 0, 1]),
            head(2, 1, &[1, 0, 0, 1]),
        ];
        for bytes in cases {
            assert!(
                matches!(
                    OpusHead::parse(&bytes, 5),
                    Err(OggError::BadOpusHead { serial: 5, .. })
                ),
                "{bytes:?}"
            );
        }
        let mut future = head(2, 0, &[]);
        future[8] = 0x10;
        assert_eq!(
            OpusHead::parse(&future, 5),
            Err(OggError::UnsupportedOpusVersion {
                serial: 5,
                version: 0x10
            })
        );
    }

    #[test]
    fn tags_round_trip_and_hostile_lengths_fail_closed() {
        let tags = OpusTags {
            vendor: "cutlass".to_owned(),
            comments: vec!["TITLE=rain".to_owned(), "ARTIST=sky".to_owned()],
            cut: false,
        };
        let bytes = tags.encode();
        assert_eq!(OpusTags::parse(&bytes, 1, false).expect("tags"), tags);

        let mut long_vendor = bytes.clone();
        long_vendor[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            OpusTags::parse(&long_vendor, 1, false),
            Err(OggError::BadOpusTags { .. })
        ));
        let mut many = b"OpusTags".to_vec();
        many.extend_from_slice(&0_u32.to_le_bytes());
        many.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            OpusTags::parse(&many, 1, false),
            Err(OggError::BadOpusTags { .. })
        ));
        assert!(matches!(
            OpusTags::parse(b"OpusTags\x01", 1, false),
            Err(OggError::BadOpusTags { .. })
        ));
        assert!(matches!(
            OpusTags::parse(b"OpusHead", 1, false),
            Err(OggError::BadOpusTags { .. })
        ));
        // A cut packet keeps the comments that fit.
        let cut = &bytes[..bytes.len() - 3];
        let kept = OpusTags::parse(cut, 1, true).expect("cut tags");
        assert_eq!(kept.comments, vec!["TITLE=rain".to_owned()]);
        assert!(kept.cut);
    }

    #[test]
    fn toc_durations_follow_rfc_6716() {
        // SILK NB 10, 20, 40, 60 ms, code 0.
        assert_eq!(packet_samples(&[0]), Some(480));
        assert_eq!(packet_samples(&[1 << 3]), Some(960));
        assert_eq!(packet_samples(&[2 << 3]), Some(1920));
        assert_eq!(packet_samples(&[3 << 3]), Some(2880));
        // Hybrid SWB/FB 10 and 20 ms.
        assert_eq!(packet_samples(&[12 << 3]), Some(480));
        assert_eq!(packet_samples(&[15 << 3]), Some(960));
        // CELT 2.5, 5, 10, 20 ms.
        assert_eq!(packet_samples(&[16 << 3]), Some(120));
        assert_eq!(packet_samples(&[29 << 3]), Some(240));
        assert_eq!(packet_samples(&[30 << 3]), Some(480));
        assert_eq!(packet_samples(&[31 << 3]), Some(960));
        // Codes 1 and 2 hold two frames.
        assert_eq!(packet_samples(&[(31 << 3) | 1]), Some(1920));
        assert_eq!(packet_samples(&[(31 << 3) | 2]), Some(1920));
        // Code 3 reads its count; 120 ms is the most.
        assert_eq!(packet_samples(&[(31 << 3) | 3, 6]), Some(5760));
        assert_eq!(packet_samples(&[(16 << 3) | 3, 48]), Some(5760));
        assert_eq!(packet_samples(&[(31 << 3) | 3, 0x80 | 3]), Some(2880));
        assert_eq!(packet_samples(&[(31 << 3) | 3, 7]), None);
        assert_eq!(packet_samples(&[(31 << 3) | 3, 0]), None);
        assert_eq!(packet_samples(&[(31 << 3) | 3]), None);
        assert_eq!(packet_samples(&[]), None);
    }

    #[test]
    fn unknown_mappings_get_a_tag_from_their_magic() {
        assert_eq!(mapping_tag(b"\x7fFLAC\x01\x00"), b"ogg:flac");
        assert_eq!(mapping_tag(b"\x80theora"), b"ogg:theora");
        assert_eq!(mapping_tag(b"Speex   1.2"), b"ogg:speex");
        assert_eq!(mapping_tag(b"\x00\x01\x02"), b"ogg:000102");
    }
}
