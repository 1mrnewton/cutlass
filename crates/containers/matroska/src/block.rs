//! Block and SimpleBlock headers and lacing (RFC 9559, sections 10.1 to 10.3).
//!
//! A block starts with the track number (a VINT), a signed 16-bit timestamp
//! relative to its Cluster, and a flag octet. Lacing packs up to 256 frames
//! into one block: the first octet after the header is the frame count less
//! one, then the sizes of all frames but the last, coded as Xiph (runs of
//! 255), EBML (a VINT, then signed differences), or fixed (equal shares).
//! The last frame takes what is left. A size that runs past the block is an
//! error, as in FFmpeg and GStreamer.

use std::ops::Range;

use crate::ebml::{self, EbmlError};
use crate::error::{Error, Result};
use crate::limits::MAX_LACED_FRAMES;

/// SimpleBlock keyframe flag (10.2).
pub const KEYFRAME: u8 = 0x80;
/// Lacing bits of the flag octet: 00 none, 01 Xiph, 11 EBML, 10 fixed.
pub const LACING: u8 = 0x06;

/// The fixed part of a block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockHeader {
    pub track: u64,
    pub relative: i16,
    pub flags: u8,
    /// Octets before the frame data (or before the lace header).
    pub len: usize,
}

impl BlockHeader {
    #[must_use]
    pub const fn lacing(self) -> u8 {
        (self.flags & LACING) >> 1
    }
}

/// Longest block header: an eight-octet track number, the timestamp, and the flags.
pub const MAX_HEADER: usize = 11;

/// Parses the block header at the start of `bytes`.
pub fn parse_header(bytes: &[u8]) -> Result<BlockHeader> {
    let (track, len) = ebml::read_vint(bytes)?;
    let rest = bytes
        .get(len..len + 3)
        .ok_or(Error::Ebml(EbmlError::Short))?;
    let relative = i16::from_be_bytes([rest[0], rest[1]]);
    Ok(BlockHeader {
        track,
        relative,
        flags: rest[2],
        len: len + 3,
    })
}

/// Octets of a block header that starts with `first`.
#[must_use]
pub fn header_len(first: u8) -> Option<usize> {
    ebml::size_len(first).map(|len| len + 3)
}

/// The byte ranges of each frame in `body`, which is the block after its
/// header. `lacing` is the two lacing bits.
pub fn frames(body: &[u8], lacing: u8) -> Result<Vec<Range<usize>>> {
    if lacing == 0 {
        return Ok(std::iter::once(0..body.len()).collect());
    }
    let count = usize::from(
        *body
            .first()
            .ok_or(Error::malformed("laced block is empty"))?,
    ) + 1;
    debug_assert!(count <= MAX_LACED_FRAMES);
    let mut at = 1;
    let mut sizes = Vec::with_capacity(count);
    match lacing {
        // Xiph.
        1 => {
            for _ in 0..count - 1 {
                let mut size = 0_usize;
                loop {
                    let byte = *body
                        .get(at)
                        .ok_or(Error::malformed("Xiph lace sizes run past the block"))?;
                    at += 1;
                    size += usize::from(byte);
                    if byte != 0xFF {
                        break;
                    }
                }
                sizes.push(size);
            }
        }
        // Fixed size.
        2 => {
            let rest = body.len() - at;
            if !rest.is_multiple_of(count) {
                return Err(Error::malformed(
                    "fixed-size lacing does not divide the block evenly",
                ));
            }
            sizes.extend(std::iter::repeat_n(rest / count, count - 1));
        }
        // EBML.
        _ if count > 1 => {
            let (first, len) = ebml::read_vint(body.get(at..).unwrap_or(&[]))?;
            at += len;
            let mut size = i64::try_from(first)
                .map_err(|_| Error::malformed("EBML lace size does not fit"))?;
            sizes.push(usize::try_from(size).unwrap_or(usize::MAX));
            for _ in 1..count - 1 {
                let (delta, len) = ebml::read_signed_vint(body.get(at..).unwrap_or(&[]))?;
                at += len;
                size = size
                    .checked_add(delta)
                    .filter(|value| *value >= 0)
                    .ok_or(Error::malformed("EBML lace size is negative"))?;
                sizes.push(usize::try_from(size).unwrap_or(usize::MAX));
            }
        }
        _ => {}
    }
    let mut ranges = Vec::with_capacity(count);
    let mut start = at;
    for size in sizes {
        let end = start
            .checked_add(size)
            .filter(|end| *end <= body.len())
            .ok_or(Error::malformed("lace sizes run past the block"))?;
        ranges.push(start..end);
        start = end;
    }
    ranges.push(start..body.len());
    Ok(ranges)
}

/// The frame count a laced block declares, from the octet after its header.
#[must_use]
pub fn lace_count(first_body_byte: u8) -> u32 {
    u32::from(first_body_byte) + 1
}

/// Lace sizes as Xiph lacing codes them, for the muxer's tests and fuzzing.
#[must_use]
pub fn xiph_sizes(sizes: &[usize]) -> Vec<u8> {
    let mut out = Vec::new();
    for size in sizes {
        let mut rest = *size;
        while rest >= 255 {
            out.push(255);
            rest -= 255;
        }
        out.push(u8::try_from(rest).unwrap_or(0));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{frames, parse_header, xiph_sizes};
    use crate::ebml::put_size_width;

    #[test]
    fn a_header_reads_track_time_and_flags() {
        let header = parse_header(&[0x81, 0xFF, 0xFE, 0x80, 9]).expect("header");
        assert_eq!(header.track, 1);
        assert_eq!(header.relative, -2);
        assert_eq!(header.flags, 0x80);
        assert_eq!(header.len, 4);
        let two = parse_header(&[0x40, 0x02, 0, 5, 0x06]).expect("two-octet track");
        assert_eq!(two.track, 2);
        assert_eq!(two.lacing(), 3);
        assert!(parse_header(&[0x81, 0]).is_err());
        assert!(parse_header(&[0x00, 0, 0, 0]).is_err());
    }

    fn laced(kind: u8, sizes: &[usize]) -> (Vec<u8>, Vec<u8>) {
        let mut body = vec![u8::try_from(sizes.len() - 1).expect("count")];
        let mut payload = Vec::new();
        for (index, size) in sizes.iter().enumerate() {
            payload.extend(std::iter::repeat_n(u8::try_from(index).expect("i"), *size));
        }
        match kind {
            1 => body.extend(xiph_sizes(&sizes[..sizes.len() - 1])),
            3 => {
                put_size_width(&mut body, sizes[0] as u64, 2);
                for pair in sizes[..sizes.len() - 1].windows(2) {
                    let delta = pair[1] as i64 - pair[0] as i64;
                    // Two octets: bias 8191.
                    put_size_width(&mut body, (delta + 8191) as u64, 2);
                }
            }
            _ => {}
        }
        body.extend_from_slice(&payload);
        (body, payload)
    }

    #[test]
    fn xiph_ebml_and_fixed_lacing_split_the_same_frames() {
        let sizes = [300_usize, 255, 10, 0, 77];
        for kind in [1_u8, 3] {
            let (body, payload) = laced(kind, &sizes);
            let ranges = frames(&body, kind).expect("lacing");
            let lens: Vec<usize> = ranges.iter().map(|range| range.len()).collect();
            assert_eq!(lens, sizes, "kind {kind}");
            let joined: Vec<u8> = ranges
                .iter()
                .flat_map(|range| body[range.clone()].to_vec())
                .collect();
            assert_eq!(joined, payload);
        }
        let (body, _) = laced(2, &[4, 4, 4]);
        let lens: Vec<usize> = frames(&body, 2)
            .expect("fixed")
            .iter()
            .map(|range| range.len())
            .collect();
        assert_eq!(lens, vec![4, 4, 4]);
    }

    #[test]
    fn lace_sizes_past_the_block_are_errors() {
        // Two frames, the first claims 300 bytes of a 10-byte block.
        let mut body = vec![1];
        body.extend(xiph_sizes(&[300]));
        body.extend_from_slice(&[0; 10]);
        assert!(frames(&body, 1).is_err());
        // Fixed lacing that does not divide.
        assert!(frames(&[2, 1, 2, 3, 4], 2).is_err());
        // An EBML difference that makes a size negative.
        let mut ebml = vec![2];
        put_size_width(&mut ebml, 1, 1);
        put_size_width(&mut ebml, 0, 1);
        ebml.extend_from_slice(&[0; 4]);
        assert!(frames(&ebml, 3).is_err());
        // Xiph runs that never end.
        assert!(frames(&[1, 255, 255], 1).is_err());
        assert!(frames(&[], 1).is_err());
        // 256 frames of zero bytes is the most a block holds.
        let mut many = vec![255];
        many.extend(std::iter::repeat_n(0, 255));
        assert_eq!(frames(&many, 1).expect("256").len(), 256);
    }
}
