//! FLAC configuration in MP4 (Xiph's "Encapsulation of FLAC in ISO Base
//! Media File Format", version 0.0.4). A `fLaC` sample entry holds one
//! `dfLa` box: a FullBox of version 0 and flags 0 whose body is native FLAC
//! metadata blocks (RFC 9639 section 8), STREAMINFO first, with the
//! last-metadata-block flag on the last one.
//!
//! A FLAC track's extradata comes in three forms: the bare 34-byte
//! STREAMINFO body a native file's demuxer hands out, the `fLaC` marker and
//! metadata blocks of a Matroska CodecPrivate, and the `dfLa` body this
//! crate's demuxer hands out. The muxer writes `dfLa` from any of them. It
//! leaves out SEEKTABLE and PADDING, which describe the native file the
//! blocks came from, and keeps every other block, as the encapsulation asks
//! a remux to do.

/// The STREAMINFO body length.
const STREAMINFO_LEN: usize = 34;
const STREAMINFO: u8 = 0;
const PADDING: u8 = 1;
const SEEKTABLE: u8 = 3;
/// The block type RFC 9639 forbids.
const FORBIDDEN: u8 = 127;
/// The last-metadata-block flag in a block header's first byte.
const LAST: u8 = 0x80;

/// One metadata block: its type and its body.
struct Block<'a> {
    kind: u8,
    body: &'a [u8],
}

/// The metadata blocks in `config`, STREAMINFO first. `None` when `config`
/// is none of the three forms: not 34 bytes and starting with neither the
/// marker nor version 0, a first block other than a 34-byte STREAMINFO, a
/// second STREAMINFO, the forbidden type, or a block that runs past the
/// end. Bytes after a block flagged last are not configuration and are
/// ignored; blocks that end with the data need no flag.
fn blocks(config: &[u8]) -> Option<Vec<Block<'_>>> {
    if config.len() == STREAMINFO_LEN {
        return Some(vec![Block {
            kind: STREAMINFO,
            body: config,
        }]);
    }
    // The marker and the FullBox version and flags are both four bytes.
    let mut rest = match config {
        [b'f', b'L', b'a', b'C', rest @ ..] | [0, _, _, _, rest @ ..] => rest,
        _ => return None,
    };
    let mut blocks = Vec::new();
    loop {
        let [head, l0, l1, l2, tail @ ..] = rest else {
            return None;
        };
        let kind = head & !LAST;
        let len = (usize::from(*l0) << 16) | (usize::from(*l1) << 8) | usize::from(*l2);
        let body = tail.get(..len)?;
        let first = blocks.is_empty();
        if kind == FORBIDDEN || (kind == STREAMINFO) != first || (first && len != STREAMINFO_LEN) {
            return None;
        }
        blocks.push(Block { kind, body });
        rest = &tail[len..];
        if head & LAST != 0 || rest.is_empty() {
            return Some(blocks);
        }
    }
}

/// The `dfLa` body for FLAC configuration in any of the three forms;
/// `None` when `config` is not FLAC configuration.
pub(crate) fn dfla(config: &[u8]) -> Option<Vec<u8>> {
    let kept: Vec<Block<'_>> = blocks(config)?
        .into_iter()
        .filter(|block| !matches!(block.kind, PADDING | SEEKTABLE))
        .collect();
    let mut out = vec![0, 0, 0, 0];
    for (index, block) in kept.iter().enumerate() {
        let len = u32::try_from(block.body.len()).ok()?.to_be_bytes();
        let last = if index + 1 == kept.len() { LAST } else { 0 };
        out.extend_from_slice(&[last | block.kind, len[1], len[2], len[3]]);
        out.extend_from_slice(block.body);
    }
    Some(out)
}

/// The STREAMINFO fields a `fLaC` sample entry repeats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Facts {
    pub(crate) rate: u32,
    pub(crate) channels: u16,
    pub(crate) bits: u16,
}

/// Rate, channel count, and depth from FLAC configuration in any of the
/// three forms. `None` when it is not FLAC configuration or states a rate
/// of zero.
pub(crate) fn facts(config: &[u8]) -> Option<Facts> {
    let blocks = blocks(config)?;
    let body = blocks.first()?.body;
    let packed = u64::from_be_bytes(body.get(10..18)?.try_into().ok()?);
    let rate = u32::try_from(packed >> 44).ok()?;
    (rate != 0).then_some(Facts {
        rate,
        channels: u16::try_from((packed >> 41) & 0x7).ok()? + 1,
        bits: u16::try_from((packed >> 36) & 0x1F).ok()? + 1,
    })
}

/// The 16.16 sample entry rate's integer part for a FLAC stream of `rate`
/// Hz. Above 65535 it is the largest whole halving of the rate that fits
/// (48000 for 96 and 192 kHz), else 65535, as the encapsulation says. The
/// STREAMINFO rate is the stream's true rate.
pub(crate) fn entry_rate(rate: u32) -> u16 {
    let mut divided = rate;
    while divided > u32::from(u16::MAX) && divided.is_multiple_of(2) {
        divided /= 2;
    }
    u16::try_from(divided).unwrap_or(u16::MAX)
}

#[cfg(test)]
mod tests {
    use super::{Facts, dfla, entry_rate, facts};

    /// The `dfLa` body FFmpeg 9.0.2 writes for
    /// `-f lavfi -i sine=duration=1 -c:a flac sine.mp4`: version 0, flags 0,
    /// then STREAMINFO flagged last (44.1 kHz, mono, 16-bit, 44100 samples).
    const FFMPEG_DFLA: [u8; 42] = [
        0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x22, 0x04, 0x00, 0x04, 0x00, 0x00, 0x00, 0x55,
        0x00, 0x01, 0x87, 0x0a, 0xc4, 0x40, 0xf0, 0x00, 0x00, 0xac, 0x44, 0xa4, 0x84, 0xb6, 0x60,
        0xd9, 0xe0, 0x33, 0xb0, 0x9b, 0xed, 0x9e, 0x68, 0x6a, 0x74, 0x73, 0xec,
    ];

    fn streaminfo() -> &'static [u8] {
        &FFMPEG_DFLA[8..]
    }

    #[test]
    fn each_form_becomes_the_same_dfla_body() {
        assert_eq!(dfla(&FFMPEG_DFLA).unwrap(), FFMPEG_DFLA);
        assert_eq!(dfla(streaminfo()).unwrap(), FFMPEG_DFLA);
        let mut native = b"fLaC\x80\x00\x00\x22".to_vec();
        native.extend_from_slice(streaminfo());
        assert_eq!(dfla(&native).unwrap(), FFMPEG_DFLA);
        assert_eq!(
            facts(&native),
            Some(Facts {
                rate: 44_100,
                channels: 1,
                bits: 16
            })
        );
    }

    #[test]
    fn comments_stay_and_the_old_file_layout_goes() {
        // A Matroska CodecPrivate as mkvmerge copies it from a native file:
        // STREAMINFO, SEEKTABLE, VORBIS_COMMENT, PADDING, none flagged last.
        let mut native = b"fLaC\x00\x00\x00\x22".to_vec();
        native.extend_from_slice(streaminfo());
        native.extend_from_slice(&[0x03, 0, 0, 18]);
        native.extend_from_slice(&[0xFF; 18]);
        native.extend_from_slice(&[0x04, 0, 0, 8, 4, 0, 0, 0, b'L', b'a', b'v', b'f']);
        native.extend_from_slice(&[0x01, 0, 0, 3, 0, 0, 0]);
        let body = dfla(&native).unwrap();
        let mut expected = FFMPEG_DFLA.to_vec();
        expected[4] = 0x00;
        expected.extend_from_slice(&[0x84, 0, 0, 8, 4, 0, 0, 0, b'L', b'a', b'v', b'f']);
        assert_eq!(body, expected);
        // The body the muxer wrote reads back to itself.
        assert_eq!(dfla(&body).unwrap(), body);
    }

    #[test]
    fn what_is_not_flac_configuration_is_refused() {
        let mut native = b"fLaC\x80\x00\x00\x22".to_vec();
        native.extend_from_slice(streaminfo());
        // Empty, a 33-byte body, another version, another marker.
        assert_eq!(dfla(&[]), None);
        assert_eq!(dfla(&streaminfo()[1..]), None);
        let mut other = FFMPEG_DFLA;
        other[0] = 1;
        assert_eq!(dfla(&other), None);
        assert_eq!(dfla(b"OpusHead\x01\x02\x38\x01\x80\xbb\0\0\0\0\0"), None);
        // A first block that is not STREAMINFO, or not 34 bytes long.
        let mut first = native.clone();
        first[4] = 0x84;
        assert_eq!(dfla(&first), None);
        let mut long = native.clone();
        long[7] = 0x23;
        assert_eq!(dfla(&long), None);
        // A block running past the end, a cut header, a second STREAMINFO,
        // and the forbidden type.
        assert_eq!(dfla(&native[..41]), None);
        let mut more = native.clone();
        more[4] = 0x00;
        for tail in [
            &[0x84, 0, 0, 9, 0][..],
            &[0x84, 0],
            &[0x80, 0, 0, 0],
            &[0xFF, 0, 0, 0],
        ] {
            let mut bytes = more.clone();
            bytes.extend_from_slice(tail);
            assert_eq!(dfla(&bytes), None, "{tail:?}");
        }
        // A rate of zero has no sample entry facts.
        // The rate is the top 20 bits of the 8 bytes at STREAMINFO byte 10.
        let mut silent = FFMPEG_DFLA;
        silent[18..20].fill(0);
        silent[20] &= 0x0F;
        assert_eq!(facts(&silent), None);
    }

    #[test]
    fn a_high_rate_is_halved_into_the_entry() {
        assert_eq!(entry_rate(44_100), 44_100);
        assert_eq!(entry_rate(65_535), 65_535);
        assert_eq!(entry_rate(96_000), 48_000);
        assert_eq!(entry_rate(192_000), 48_000);
        assert_eq!(entry_rate(176_400), 44_100);
        assert_eq!(entry_rate(65_537), 65_535);
        assert_eq!(entry_rate(655_350), 65_535);
    }
}
