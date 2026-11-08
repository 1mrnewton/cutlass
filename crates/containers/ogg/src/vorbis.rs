//! The Vorbis mapping's header packets and packet durations (Vorbis I
//! specification, sections 4.2 and A).
//!
//! The container needs three things from Vorbis: the identification header
//! (channels, rate, and the two block sizes), the comment header (kept for
//! the caller), and each audio packet's sample count for the granule
//! arithmetic. A packet returns `previous / 4 + current / 4` samples, and
//! the first packet of a stream none (section 4.3.8 and A.2), so a packet's
//! block size comes from its mode, and the modes are the last fields of the
//! setup header. The setup header is walked forward, field by field, to
//! reach them; nothing in it is decoded or stored. Decoding stays in the
//! Vorbis codec crate.

use crate::error::OggError;
use crate::opus::read_comment_list;

/// The first seven bytes of the identification header.
pub const ID_MAGIC: &[u8; 7] = b"\x01vorbis";
/// The first seven bytes of the comment header.
pub const COMMENT_MAGIC: &[u8; 7] = b"\x03vorbis";
/// The first seven bytes of the setup header.
pub const SETUP_MAGIC: &[u8; 7] = b"\x05vorbis";

/// Bytes of a valid identification header (section 4.2.2).
pub const ID_BYTES: usize = 30;

/// Fields of the identification header (section 4.2.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VorbisId {
    pub channels: u8,
    pub sample_rate: u32,
    pub bitrate_maximum: i32,
    pub bitrate_nominal: i32,
    pub bitrate_minimum: i32,
    /// Short and long block sizes, powers of two from 64 to 8192.
    pub blocksizes: [u32; 2],
}

impl VorbisId {
    /// Reads and validates the identification header. `serial` names the
    /// stream in the error.
    pub fn parse(bytes: &[u8], serial: u32) -> Result<Self, OggError> {
        let bad = |reason| OggError::BadVorbisHeader { serial, reason };
        if !bytes.starts_with(ID_MAGIC) {
            return Err(bad("first packet is not the identification header"));
        }
        if bytes.len() < ID_BYTES {
            return Err(bad("identification header is shorter than 30 bytes"));
        }
        let word = |at: usize| {
            u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
        };
        if word(7) != 0 {
            return Err(OggError::UnsupportedVorbisVersion {
                serial,
                version: word(7),
            });
        }
        let channels = bytes[11];
        if channels == 0 {
            return Err(bad("zero channels"));
        }
        let sample_rate = word(12);
        if sample_rate == 0 {
            return Err(bad("zero sample rate"));
        }
        let signed = |at: usize| {
            i32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
        };
        let short = bytes[28] & 0x0F;
        let long = bytes[28] >> 4;
        if !(6..=13).contains(&short) || !(6..=13).contains(&long) || short > long {
            return Err(bad("block sizes out of range"));
        }
        if bytes[29] & 1 != 1 {
            return Err(bad("identification header framing bit is not set"));
        }
        Ok(Self {
            channels,
            sample_rate,
            bitrate_maximum: signed(16),
            bitrate_nominal: signed(20),
            bitrate_minimum: signed(24),
            blocksizes: [1 << short, 1 << long],
        })
    }
}

/// Fields of the comment header (section 5).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct VorbisComment {
    /// Vendor string. Bytes that are not UTF-8 are replaced.
    pub vendor: String,
    /// `NAME=value` comments in file order. Bytes that are not UTF-8 are replaced.
    pub comments: Vec<String>,
    /// The packet was larger than the header bound. Comments past the bound
    /// were ignored, and the framing bit was not checked.
    pub cut: bool,
}

impl VorbisComment {
    /// Reads the comment header. `cut` says the packet was cut at the header
    /// bound, in which case a comment that runs past the kept bytes ends the
    /// list instead of failing it.
    pub fn parse(bytes: &[u8], serial: u32, cut: bool) -> Result<Self, OggError> {
        let bad = |reason| OggError::BadVorbisHeader { serial, reason };
        if !bytes.starts_with(COMMENT_MAGIC) {
            return Err(bad("second packet is not the comment header"));
        }
        let (vendor, comments, end) = read_comment_list(bytes, COMMENT_MAGIC.len(), cut, bad)?;
        if !cut && bytes.get(end).is_none_or(|&framing| framing & 1 != 1) {
            return Err(bad("comment header framing bit is not set"));
        }
        Ok(Self {
            vendor,
            comments,
            cut,
        })
    }

    /// Serializes the comment header, with its framing bit.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = COMMENT_MAGIC.to_vec();
        crate::opus::push_string(&mut out, self.vendor.as_bytes());
        out.extend_from_slice(
            &u32::try_from(self.comments.len())
                .unwrap_or(u32::MAX)
                .to_le_bytes(),
        );
        for comment in &self.comments {
            crate::opus::push_string(&mut out, comment.as_bytes());
        }
        out.push(1);
        out
    }
}

/// What the container keeps of a setup header: each mode's block flag.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VorbisModes {
    /// Long-block flag per mode, 1 to 64 modes.
    pub long: Vec<bool>,
    /// Width of the mode number in an audio packet: `ilog(modes - 1)`.
    pub bits: u32,
    pub blocksizes: [u32; 2],
}

impl VorbisModes {
    /// Walks the setup header (section 4.2.4) to its modes. `channels`
    /// sizes the mapping fields. Every count is checked against the bits
    /// left before it is walked.
    pub fn parse(bytes: &[u8], id: &VorbisId, serial: u32) -> Result<Self, OggError> {
        let bad = |reason| OggError::BadVorbisSetup { serial, reason };
        if !bytes.starts_with(SETUP_MAGIC) {
            return Err(bad("third packet is not the setup header"));
        }
        let mut bits = Bits::new(&bytes[SETUP_MAGIC.len()..]);
        let channels = u32::from(id.channels);
        let books = bits
            .take(8)
            .ok_or_else(|| bad("cut inside the codebooks"))?
            + 1;
        for _ in 0..books {
            skip_codebook(&mut bits).map_err(bad)?;
        }
        let truncated = || bad("cut inside the setup header");
        let times = bits.take(6).ok_or_else(truncated)? + 1;
        for _ in 0..times {
            if bits.take(16).ok_or_else(truncated)? != 0 {
                return Err(bad("time domain transform is not 0"));
            }
        }
        let floors = bits.take(6).ok_or_else(truncated)? + 1;
        for _ in 0..floors {
            skip_floor(&mut bits, books).map_err(bad)?;
        }
        let residues = bits.take(6).ok_or_else(truncated)? + 1;
        for _ in 0..residues {
            skip_residue(&mut bits, books).map_err(bad)?;
        }
        let mappings = bits.take(6).ok_or_else(truncated)? + 1;
        for _ in 0..mappings {
            skip_mapping(&mut bits, channels, floors, residues).map_err(bad)?;
        }
        let modes = bits.take(6).ok_or_else(truncated)? + 1;
        let mut long = Vec::with_capacity(modes as usize);
        for _ in 0..modes {
            let flag = bits.take(1).ok_or_else(truncated)? == 1;
            let window = bits.take(16).ok_or_else(truncated)?;
            let transform = bits.take(16).ok_or_else(truncated)?;
            let mapping = bits.take(8).ok_or_else(truncated)?;
            if window != 0 || transform != 0 {
                return Err(bad("mode window or transform type is not 0"));
            }
            if mapping >= mappings {
                return Err(bad("mode names a missing mapping"));
            }
            long.push(flag);
        }
        if bits.take(1).ok_or_else(truncated)? != 1 {
            return Err(bad("setup header framing bit is not set"));
        }
        Ok(Self {
            long,
            bits: ilog(modes - 1),
            blocksizes: id.blocksizes,
        })
    }

    /// Block size of an audio packet, from its first byte: the packet type
    /// bit, then the mode number. `None` for a header packet, an empty
    /// packet, or a mode the setup does not have.
    #[must_use]
    pub fn block(&self, packet: &[u8]) -> Option<u32> {
        let &first = packet.first()?;
        if first & 1 == 1 {
            return None;
        }
        let mode = usize::from(first >> 1) & ((1 << self.bits) - 1);
        let long = *self.long.get(mode)?;
        Some(self.blocksizes[usize::from(long)])
    }
}

/// Samples a packet of block `current` returns after one of block
/// `previous`; the first packet returns none.
#[must_use]
pub fn packet_samples(previous: Option<u32>, current: u32) -> i64 {
    previous.map_or(0, |previous| i64::from(previous / 4 + current / 4))
}

/// The three header packets as Xiph lacing: `2`, the first two lengths in
/// 255-runs, then the packets. This is the extradata the Vorbis decoder
/// takes.
#[must_use]
pub fn xiph_lace(packets: [&[u8]; 3]) -> Vec<u8> {
    let mut out = vec![2];
    for packet in &packets[..2] {
        let mut len = packet.len();
        while len >= 255 {
            out.push(255);
            len -= 255;
        }
        out.push(u8::try_from(len).unwrap_or(0));
    }
    for packet in packets {
        out.extend_from_slice(packet);
    }
    out
}

/// Bits used to write `value` (section 9.2.1).
const fn ilog(value: u32) -> u32 {
    u32::BITS - value.leading_zeros()
}

/// Reads LSb-first values (section 2.1.4).
struct Bits<'a> {
    data: &'a [u8],
    at: u64,
}

impl<'a> Bits<'a> {
    const fn new(data: &'a [u8]) -> Self {
        Self { data, at: 0 }
    }

    fn remaining(&self) -> u64 {
        (self.data.len() as u64 * 8).saturating_sub(self.at)
    }

    /// The next `width` bits (0 to 32), or `None` past the end.
    fn take(&mut self, width: u32) -> Option<u32> {
        if u64::from(width) > self.remaining() {
            self.at = self.data.len() as u64 * 8;
            return None;
        }
        let mut value = 0_u64;
        for bit in 0..u64::from(width) {
            let at = self.at + bit;
            let byte = self.data[usize::try_from(at / 8).ok()?];
            value |= u64::from((byte >> (at % 8)) & 1) << bit;
        }
        self.at += u64::from(width);
        u32::try_from(value).ok()
    }

    fn skip(&mut self, count: u64) -> Option<()> {
        if count > self.remaining() {
            return None;
        }
        self.at += count;
        Some(())
    }
}

fn skip_codebook(bits: &mut Bits<'_>) -> Result<(), &'static str> {
    let cut = "cut inside a codebook";
    if bits.take(24).ok_or(cut)? != 0x56_4342 {
        return Err("codebook sync pattern is wrong");
    }
    let dimensions = bits.take(16).ok_or(cut)?;
    let entries = bits.take(24).ok_or(cut)?;
    if dimensions == 0 || entries == 0 {
        return Err("codebook with no dimensions or no entries");
    }
    if bits.take(1).ok_or(cut)? == 1 {
        // Ordered: runs of entries with lengths 1 more each time.
        let mut length = bits.take(5).ok_or(cut)? + 1;
        let mut entry = 0;
        while entry < entries {
            let number = bits.take(ilog(entries - entry)).ok_or(cut)?;
            if number > entries - entry {
                return Err("codebook length runs past its entries");
            }
            entry += number;
            if entry < entries {
                length += 1;
                if length > 32 {
                    return Err("codebook length above 32");
                }
            }
        }
    } else if bits.take(1).ok_or(cut)? == 1 {
        // Sparse: a used flag per entry, then a length for each used one.
        for _ in 0..entries {
            if bits.take(1).ok_or(cut)? == 1 {
                bits.take(5).ok_or(cut)?;
            }
        }
    } else {
        bits.skip(u64::from(entries) * 5).ok_or(cut)?;
    }
    match bits.take(4).ok_or(cut)? {
        0 => Ok(()),
        kind @ (1 | 2) => {
            bits.skip(64).ok_or(cut)?;
            let value_bits = bits.take(4).ok_or(cut)? + 1;
            bits.take(1).ok_or(cut)?;
            let values = if kind == 1 {
                lookup1_values(entries, dimensions)
            } else {
                u64::from(entries) * u64::from(dimensions)
            };
            bits.skip(values.saturating_mul(u64::from(value_bits)))
                .ok_or(cut)
        }
        _ => Err("codebook lookup type above 2"),
    }
}

/// The largest `r` with `r ^ dimensions <= entries` (section 9.2.3).
fn lookup1_values(entries: u32, dimensions: u32) -> u64 {
    let entries = u64::from(entries);
    let fits = |root: u64| {
        root.checked_pow(dimensions)
            .is_some_and(|value| value <= entries)
    };
    let guess = (entries as f64).powf(1.0 / f64::from(dimensions)).floor() as u64;
    let mut root = guess.max(1);
    while root > 1 && !fits(root) {
        root -= 1;
    }
    while fits(root + 1) {
        root += 1;
    }
    root
}

fn skip_floor(bits: &mut Bits<'_>, books: u32) -> Result<(), &'static str> {
    let cut = "cut inside a floor";
    match bits.take(16).ok_or(cut)? {
        0 => {
            bits.skip(8 + 16 + 16 + 6 + 8).ok_or(cut)?;
            let count = bits.take(4).ok_or(cut)? + 1;
            for _ in 0..count {
                if bits.take(8).ok_or(cut)? >= books {
                    return Err("floor 0 names a missing codebook");
                }
            }
            Ok(())
        }
        1 => {
            let partitions = bits.take(5).ok_or(cut)?;
            let mut classes = Vec::with_capacity(partitions as usize);
            for _ in 0..partitions {
                classes.push(bits.take(4).ok_or(cut)?);
            }
            let class_count = classes.iter().max().map_or(0, |&max| max + 1);
            let mut dimensions = Vec::with_capacity(class_count as usize);
            for _ in 0..class_count {
                dimensions.push(bits.take(3).ok_or(cut)? + 1);
                let subclass = bits.take(2).ok_or(cut)?;
                if subclass > 0 && bits.take(8).ok_or(cut)? >= books {
                    return Err("floor 1 names a missing codebook");
                }
                for _ in 0..(1_u32 << subclass) {
                    let book = bits.take(8).ok_or(cut)?;
                    if book > books {
                        return Err("floor 1 names a missing codebook");
                    }
                }
            }
            bits.take(2).ok_or(cut)?;
            let range_bits = bits.take(4).ok_or(cut)?;
            let mut values = 2_u32;
            for class in classes {
                let count = dimensions[class as usize];
                values += count;
                if values > 65 {
                    return Err("floor 1 has more than 65 points");
                }
                bits.skip(u64::from(count) * u64::from(range_bits))
                    .ok_or(cut)?;
            }
            Ok(())
        }
        _ => Err("floor type above 1"),
    }
}

fn skip_residue(bits: &mut Bits<'_>, books: u32) -> Result<(), &'static str> {
    let cut = "cut inside a residue";
    if bits.take(16).ok_or(cut)? > 2 {
        return Err("residue type above 2");
    }
    bits.skip(24 * 3).ok_or(cut)?;
    let classifications = bits.take(6).ok_or(cut)? + 1;
    if bits.take(8).ok_or(cut)? >= books {
        return Err("residue names a missing codebook");
    }
    let mut cascades = Vec::with_capacity(classifications as usize);
    for _ in 0..classifications {
        let low = bits.take(3).ok_or(cut)?;
        let high = if bits.take(1).ok_or(cut)? == 1 {
            bits.take(5).ok_or(cut)?
        } else {
            0
        };
        cascades.push(high << 3 | low);
    }
    for cascade in cascades {
        for _ in 0..cascade.count_ones() {
            if bits.take(8).ok_or(cut)? >= books {
                return Err("residue names a missing codebook");
            }
        }
    }
    Ok(())
}

fn skip_mapping(
    bits: &mut Bits<'_>,
    channels: u32,
    floors: u32,
    residues: u32,
) -> Result<(), &'static str> {
    let cut = "cut inside a mapping";
    if bits.take(16).ok_or(cut)? != 0 {
        return Err("mapping type is not 0");
    }
    let submaps = if bits.take(1).ok_or(cut)? == 1 {
        bits.take(4).ok_or(cut)? + 1
    } else {
        1
    };
    if bits.take(1).ok_or(cut)? == 1 {
        let steps = bits.take(8).ok_or(cut)? + 1;
        let width = ilog(channels - 1);
        for _ in 0..steps {
            let magnitude = bits.take(width).ok_or(cut)?;
            let angle = bits.take(width).ok_or(cut)?;
            if magnitude == angle || magnitude >= channels || angle >= channels {
                return Err("mapping couples a missing channel or a channel with itself");
            }
        }
    }
    if bits.take(2).ok_or(cut)? != 0 {
        return Err("mapping reserved bits are not 0");
    }
    if submaps > 1 {
        for _ in 0..channels {
            if bits.take(4).ok_or(cut)? >= submaps {
                return Err("mapping sends a channel to a missing submap");
            }
        }
    }
    for _ in 0..submaps {
        bits.take(8).ok_or(cut)?;
        if bits.take(8).ok_or(cut)? >= floors {
            return Err("mapping names a missing floor");
        }
        if bits.take(8).ok_or(cut)? >= residues {
            return Err("mapping names a missing residue");
        }
    }
    Ok(())
}

/// Vorbis headers built in memory for tests.
#[cfg(test)]
pub(crate) mod build {
    use super::{COMMENT_MAGIC, ID_MAGIC, SETUP_MAGIC};

    /// Writes LSb-first values.
    #[derive(Default)]
    pub struct Writer {
        pub bytes: Vec<u8>,
        bits: u64,
    }

    impl Writer {
        pub fn put(&mut self, value: u32, width: u32) {
            for bit in 0..width {
                if self.bits.is_multiple_of(8) {
                    self.bytes.push(0);
                }
                let last = self.bytes.len() - 1;
                self.bytes[last] |=
                    u8::try_from((value >> bit) & 1).expect("bit") << (self.bits % 8);
                self.bits += 1;
            }
        }
    }

    /// An identification header with block exponents `short` and `long`.
    pub fn id(channels: u8, rate: u32, short: u8, long: u8) -> Vec<u8> {
        let mut out = ID_MAGIC.to_vec();
        out.extend_from_slice(&0_u32.to_le_bytes());
        out.push(channels);
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&0_i32.to_le_bytes());
        out.extend_from_slice(&128_000_i32.to_le_bytes());
        out.extend_from_slice(&0_i32.to_le_bytes());
        out.push(long << 4 | short);
        out.push(1);
        out
    }

    /// A comment header with `comments`.
    pub fn comment(vendor: &str, comments: &[&str]) -> Vec<u8> {
        let mut out = COMMENT_MAGIC.to_vec();
        out.extend_from_slice(&u32::try_from(vendor.len()).expect("len").to_le_bytes());
        out.extend_from_slice(vendor.as_bytes());
        out.extend_from_slice(&u32::try_from(comments.len()).expect("len").to_le_bytes());
        for comment in comments {
            out.extend_from_slice(&u32::try_from(comment.len()).expect("len").to_le_bytes());
            out.extend_from_slice(comment.as_bytes());
        }
        out.push(1);
        out
    }

    /// The fields of a setup header up to its modes: one two-entry
    /// codebook, one floor 1 with no partitions, one residue 2, and one
    /// mapping, coupling channels 0 and 1 when there are two or more.
    pub fn setup_head(writer: &mut Writer, channels: u32) {
        writer.put(0, 8);
        writer.put(0x56_4342, 24);
        writer.put(1, 16);
        writer.put(2, 24);
        writer.put(0, 1);
        writer.put(0, 1);
        writer.put(0, 5);
        writer.put(0, 5);
        writer.put(0, 4);
        // Time domain transforms.
        writer.put(0, 6);
        writer.put(0, 16);
        // Floor 1: no partitions, multiplier 2, range bits 8.
        writer.put(0, 6);
        writer.put(1, 16);
        writer.put(0, 5);
        writer.put(1, 2);
        writer.put(8, 4);
        // Residue 2 with one classification and no books.
        writer.put(0, 6);
        writer.put(2, 16);
        writer.put(0, 24);
        writer.put(0, 24);
        writer.put(0, 24);
        writer.put(0, 6);
        writer.put(0, 8);
        writer.put(0, 3);
        writer.put(0, 1);
        // Mapping 0, one submap.
        writer.put(0, 6);
        writer.put(0, 16);
        writer.put(0, 1);
        if channels >= 2 {
            let width = 32 - (channels - 1).leading_zeros();
            writer.put(1, 1);
            writer.put(0, 8);
            writer.put(0, width);
            writer.put(1, width);
        } else {
            writer.put(0, 1);
        }
        writer.put(0, 2);
        writer.put(0, 8);
        writer.put(0, 8);
        writer.put(0, 8);
    }

    /// A setup header whose modes have block flags `long`.
    pub fn setup(channels: u32, long: &[bool]) -> Vec<u8> {
        let mut writer = Writer::default();
        setup_head(&mut writer, channels);
        writer.put(u32::try_from(long.len() - 1).expect("modes"), 6);
        for &flag in long {
            writer.put(u32::from(flag), 1);
            writer.put(0, 16);
            writer.put(0, 16);
            writer.put(0, 8);
        }
        writer.put(1, 1);
        let mut out = SETUP_MAGIC.to_vec();
        out.extend_from_slice(&writer.bytes);
        out
    }

    /// An audio packet in `mode`, below 64. Bit 7 is set: it is past the
    /// mode number of a setup with 64 modes or fewer.
    pub fn audio(mode: u8, len: usize) -> Vec<u8> {
        let mut out = vec![0x5A; len.max(1)];
        out[0] = (mode << 1) | 0x80;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::build::{Writer, audio, comment, id, setup, setup_head};
    use super::{VorbisComment, VorbisId, VorbisModes, packet_samples, xiph_lace};
    use crate::error::OggError;

    fn stereo() -> VorbisId {
        VorbisId::parse(&id(2, 44_100, 8, 11), 1).expect("id")
    }

    #[test]
    fn the_identification_header_gives_rate_channels_and_blocks() {
        let parsed = stereo();
        assert_eq!(parsed.channels, 2);
        assert_eq!(parsed.sample_rate, 44_100);
        assert_eq!(parsed.bitrate_nominal, 128_000);
        assert_eq!(parsed.blocksizes, [256, 2048]);
    }

    #[test]
    fn a_bad_identification_header_is_refused() {
        let good = id(2, 44_100, 8, 11);
        let reason = |bytes: &[u8]| match VorbisId::parse(bytes, 9) {
            Err(OggError::BadVorbisHeader { serial: 9, reason }) => reason,
            other => panic!("{other:?}"),
        };
        assert!(reason(&good[..29]).contains("shorter"));
        assert!(reason(b"\x03vorbis").contains("identification"));
        let mut zero = good.clone();
        zero[11] = 0;
        assert!(reason(&zero).contains("channels"));
        let mut rate = good.clone();
        rate[12..16].fill(0);
        assert!(reason(&rate).contains("rate"));
        for blocks in [0x55, 0xE8, 0x8B, 0x0B] {
            let mut sizes = good.clone();
            sizes[28] = blocks;
            assert!(reason(&sizes).contains("block"), "{blocks:#x}");
        }
        let mut framing = good.clone();
        framing[29] = 0;
        assert!(reason(&framing).contains("framing"));
        let mut version = good;
        version[7] = 1;
        assert_eq!(
            VorbisId::parse(&version, 9),
            Err(OggError::UnsupportedVorbisVersion {
                serial: 9,
                version: 1
            })
        );
    }

    #[test]
    fn the_comment_header_reads_and_writes_back() {
        let bytes = comment(
            "Xiph.Org libVorbis I 20200704",
            &["TITLE=One", "ARTIST=Two"],
        );
        let parsed = VorbisComment::parse(&bytes, 1, false).expect("comment");
        assert_eq!(parsed.vendor, "Xiph.Org libVorbis I 20200704");
        assert_eq!(parsed.comments, ["TITLE=One", "ARTIST=Two"]);
        assert_eq!(parsed.encode(), bytes);
    }

    #[test]
    fn a_bad_comment_header_is_refused_unless_cut() {
        let bytes = comment("v", &["A=1", "B=2"]);
        let mut unframed = bytes.clone();
        unframed.pop();
        assert!(matches!(
            VorbisComment::parse(&unframed, 1, false),
            Err(OggError::BadVorbisHeader { reason, .. }) if reason.contains("framing")
        ));
        let short = &bytes[..bytes.len() - 3];
        assert!(VorbisComment::parse(short, 1, false).is_err());
        let cut = VorbisComment::parse(short, 1, true).expect("cut");
        assert_eq!(cut.comments, ["A=1"]);
        assert!(cut.cut);
        let mut huge = comment("v", &[]);
        let count = 12;
        huge[count..count + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(VorbisComment::parse(&huge, 1, false).is_err());
        assert!(VorbisComment::parse(b"\x01vorbis", 1, false).is_err());
    }

    #[test]
    fn the_setup_header_gives_each_mode_its_block() {
        let modes = VorbisModes::parse(&setup(2, &[false, true]), &stereo(), 1).expect("setup");
        assert_eq!(modes.long, [false, true]);
        assert_eq!(modes.bits, 1);
        assert_eq!(modes.block(&audio(0, 10)), Some(256));
        assert_eq!(modes.block(&audio(1, 10)), Some(2048));
        assert_eq!(modes.block(&[1, 2, 3]), None, "a header packet");
        assert_eq!(modes.block(&[]), None);

        let three =
            VorbisModes::parse(&setup(2, &[false, true, true]), &stereo(), 1).expect("setup");
        assert_eq!(three.bits, 2);
        assert_eq!(three.block(&audio(2, 1)), Some(2048));
        assert_eq!(three.block(&audio(3, 1)), None, "mode 3 is missing");

        let one = VorbisModes::parse(
            &setup(1, &[true]),
            &VorbisId::parse(&id(1, 8_000, 6, 6), 1).expect("id"),
            1,
        )
        .expect("setup");
        assert_eq!(one.bits, 0);
        assert_eq!(one.block(&[0xFE]), Some(64), "no mode bits");

        let many: Vec<bool> = (0..64).map(|mode| mode % 3 == 0).collect();
        let modes = VorbisModes::parse(&setup(2, &many), &stereo(), 1).expect("setup");
        assert_eq!(modes.bits, 6);
        assert_eq!(modes.block(&[63 << 1]), Some(2048));
        assert_eq!(modes.block(&[62 << 1]), Some(256));
    }

    #[test]
    fn every_cut_of_the_setup_header_is_refused() {
        let bytes = setup(2, &[false, true]);
        for end in 0..bytes.len() {
            assert!(
                matches!(
                    VorbisModes::parse(&bytes[..end], &stereo(), 4),
                    Err(OggError::BadVorbisSetup { serial: 4, .. })
                ),
                "cut at {end}"
            );
        }
    }

    #[test]
    fn a_setup_naming_what_it_lacks_is_refused() {
        let reason = |bytes: &[u8], id: &VorbisId| match VorbisModes::parse(bytes, id, 1) {
            Err(OggError::BadVorbisSetup { reason, .. }) => reason,
            other => panic!("{other:?}"),
        };
        // Coupling in a mono stream names a missing channel.
        let mono = VorbisId::parse(&id(1, 8_000, 8, 11), 1).expect("id");
        assert!(reason(&setup(2, &[false]), &mono).contains("channel"));
        // A mode that names mapping 1 when there is one mapping.
        let mut writer = Writer::default();
        setup_head(&mut writer, 2);
        writer.put(0, 6);
        writer.put(0, 1);
        writer.put(0, 16);
        writer.put(0, 16);
        writer.put(1, 8);
        writer.put(1, 1);
        let mut bytes = b"\x05vorbis".to_vec();
        bytes.extend_from_slice(&writer.bytes);
        assert!(reason(&bytes, &stereo()).contains("mapping"));
        // A broken codebook sync.
        let mut sync = setup(2, &[false]);
        sync[8] ^= 0xFF;
        assert!(reason(&sync, &stereo()).contains("sync"));
        assert!(reason(b"\x03vorbis", &stereo()).contains("setup"));
    }

    #[test]
    fn a_hostile_codebook_count_ends_at_the_bytes() {
        // 2^24 entries with flat lengths need 10 MB the packet does not have.
        let mut writer = Writer::default();
        writer.put(0, 8);
        writer.put(0x56_4342, 24);
        writer.put(1, 16);
        writer.put(0xFF_FFFF, 24);
        writer.put(0, 2);
        let mut bytes = b"\x05vorbis".to_vec();
        bytes.extend_from_slice(&writer.bytes);
        assert!(matches!(
            VorbisModes::parse(&bytes, &stereo(), 1),
            Err(OggError::BadVorbisSetup { reason, .. }) if reason.contains("cut")
        ));
    }

    #[test]
    fn packet_samples_follow_the_two_blocks() {
        assert_eq!(packet_samples(None, 2048), 0);
        assert_eq!(packet_samples(Some(256), 256), 128);
        assert_eq!(packet_samples(Some(256), 2048), 576);
        assert_eq!(packet_samples(Some(2048), 256), 576);
        assert_eq!(packet_samples(Some(2048), 2048), 1024);
    }

    #[test]
    fn xiph_lacing_writes_two_lengths_in_runs_of_255() {
        let long = vec![7_u8; 300];
        let laced = xiph_lace([b"ab", &long, b"xyz"]);
        assert_eq!(&laced[..4], &[2, 2, 255, 45]);
        assert_eq!(laced.len(), 4 + 2 + 300 + 3);
        assert!(laced.ends_with(b"xyz"));
    }
}
