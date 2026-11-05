//! Opus configuration in MP4 (the Opus-in-ISOBMFF encapsulation, version
//! 0.6.8): the `dOps` box body and the Ogg `OpusHead` packet (RFC 7845)
//! carry the same fields, `dOps` big-endian with a version of 0 and
//! `OpusHead` little-endian behind its magic with a version of 1.
//!
//! The demuxer hands out `OpusHead`, the form every Opus decoder and the
//! Ogg mux read, and the muxer writes `dOps` from either form.

const MAGIC: &[u8; 8] = b"OpusHead";

/// The fields both forms share.
struct Fields<'a> {
    channels: u8,
    pre_skip: u16,
    input_rate: u32,
    gain: i16,
    family: u8,
    /// Stream count, coupled count, and one mapping byte per channel; empty
    /// for family 0.
    table: &'a [u8],
}

fn table(rest: &[u8], channels: u8, family: u8) -> Option<&[u8]> {
    if family == 0 {
        return (channels == 1 || channels == 2).then_some(&[]);
    }
    rest.get(..2 + usize::from(channels))
}

fn from_head(head: &[u8]) -> Option<Fields<'_>> {
    if head.len() < 19 || &head[..8] != MAGIC || head[8] >> 4 != 0 {
        return None;
    }
    let channels = head[9];
    let family = head[18];
    Some(Fields {
        channels,
        pre_skip: u16::from_le_bytes([head[10], head[11]]),
        input_rate: u32::from_le_bytes([head[12], head[13], head[14], head[15]]),
        gain: i16::from_le_bytes([head[16], head[17]]),
        family,
        table: table(&head[19..], channels, family)?,
    })
}

fn from_dops(dops: &[u8]) -> Option<Fields<'_>> {
    if dops.len() < 11 || dops[0] != 0 {
        return None;
    }
    let channels = dops[1];
    let family = dops[10];
    Some(Fields {
        channels,
        pre_skip: u16::from_be_bytes([dops[2], dops[3]]),
        input_rate: u32::from_be_bytes([dops[4], dops[5], dops[6], dops[7]]),
        gain: i16::from_be_bytes([dops[8], dops[9]]),
        family,
        table: table(&dops[11..], channels, family)?,
    })
}

fn fields(config: &[u8]) -> Option<Fields<'_>> {
    from_head(config).or_else(|| from_dops(config))
}

/// The `OpusHead` for a `dOps` body or an `OpusHead`; `None` when `config`
/// is neither.
pub(crate) fn head(config: &[u8]) -> Option<Vec<u8>> {
    let fields = fields(config)?;
    let mut out = MAGIC.to_vec();
    out.extend([1, fields.channels]);
    out.extend(fields.pre_skip.to_le_bytes());
    out.extend(fields.input_rate.to_le_bytes());
    out.extend(fields.gain.to_le_bytes());
    out.push(fields.family);
    out.extend(fields.table);
    Some(out)
}

/// The `dOps` body for an `OpusHead` or a `dOps` body; `None` when `config`
/// is neither.
pub(crate) fn dops(config: &[u8]) -> Option<Vec<u8>> {
    let fields = fields(config)?;
    let mut out = vec![0, fields.channels];
    out.extend(fields.pre_skip.to_be_bytes());
    out.extend(fields.input_rate.to_be_bytes());
    out.extend(fields.gain.to_be_bytes());
    out.push(fields.family);
    out.extend(fields.table);
    Some(out)
}

/// The samples at 48 kHz a decoder drops before the first one played.
pub(crate) fn pre_skip(config: &[u8]) -> Option<u16> {
    fields(config).map(|fields| fields.pre_skip)
}

#[cfg(test)]
mod tests {
    use super::{dops, head, pre_skip};

    /// The body ffmpeg writes for a stereo libopus track: version 0, two
    /// channels, 312 samples of pre-skip, 48 kHz, no gain, family 0.
    const FFMPEG_DOPS: [u8; 11] = [0, 2, 0x01, 0x38, 0, 0, 0xBB, 0x80, 0, 0, 0];

    #[test]
    fn dops_and_opus_head_round_trip() {
        let opus_head = head(&FFMPEG_DOPS).unwrap();
        assert_eq!(&opus_head[..8], b"OpusHead");
        assert_eq!(opus_head.len(), 19);
        assert_eq!(&opus_head[8..12], &[1, 2, 0x38, 0x01]);
        assert_eq!(dops(&opus_head).unwrap(), FFMPEG_DOPS);
        assert_eq!(dops(&FFMPEG_DOPS).unwrap(), FFMPEG_DOPS);
        assert_eq!(pre_skip(&opus_head), Some(312));
        assert_eq!(pre_skip(&FFMPEG_DOPS), Some(312));
    }

    #[test]
    fn a_channel_mapping_table_survives_and_a_short_one_is_refused() {
        // 5.1, family 1: four streams, two coupled, Vorbis order.
        let mut surround = vec![0, 6, 0, 120, 0, 0, 0xBB, 0x80, 0xFF, 0x00, 1, 4, 2];
        surround.extend([0, 4, 1, 2, 3, 5]);
        let opus_head = head(&surround).unwrap();
        assert_eq!(opus_head.len(), 19 + 2 + 6);
        assert_eq!(i16::from_le_bytes([opus_head[16], opus_head[17]]), -256);
        assert_eq!(dops(&opus_head).unwrap(), surround);
        assert!(dops(&surround[..surround.len() - 1]).is_none());
        // Family 0 is mono or stereo only, and the version must be 0 or 1.
        assert!(dops(&[0, 3, 0, 0, 0, 0, 0xBB, 0x80, 0, 0, 0]).is_none());
        assert!(dops(&[1, 2, 0, 0, 0, 0, 0xBB, 0x80, 0, 0, 0]).is_none());
        assert!(head(b"OpusHea").is_none());
    }
}
