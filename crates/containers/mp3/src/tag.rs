//! Xing, Info, and VBRI headers.
//!
//! The tag sits in the first Layer III frame and is not an audio packet.
//! Xing and Info share a layout: a four-character id, a flag word, then the
//! optional frame count, byte count, 100-byte TOC, and quality. A LAME
//! extension may follow. VBRI is the Fraunhofer header 32 bytes after the
//! MPEG header (36 bytes from the start of the frame) and is version 1.
//!
//! Encoder delay and trailing padding are the 12-bit fields in that LAME
//! extension. They are samples at the frame's sample rate. This module does
//! not add the decoder's own filterbank delay and does not drop samples.

use crate::header::{Header, side_info_len};
use crate::limits::MAX_VBRI_ENTRIES;

const XING: &[u8; 4] = b"Xing";
const INFO: &[u8; 4] = b"Info";
const VBRI: &[u8; 4] = b"VBRI";

const FLAG_FRAMES: u32 = 0x01;
const FLAG_BYTES: u32 = 0x02;
const FLAG_TOC: u32 = 0x04;
const FLAG_QUALITY: u32 = 0x08;

/// A Xing or Info tag found in a frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Xing {
    pub cbr: bool,
    pub frames: Option<u32>,
    pub bytes: Option<u32>,
    pub toc: Option<[u8; 100]>,
    pub encoder_delay: Option<u32>,
    pub trailing_padding: Option<u32>,
}

/// A VBRI tag. `starts[i]` is the byte offset of table entry `i` from the
/// first byte of the VBRI frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vbri {
    pub frames: u32,
    pub bytes: u32,
    pub frames_per_entry: u32,
    pub starts: Vec<u64>,
}

/// Read a Xing or Info tag. The bytes before the id must be the header and
/// a zero side-information block, which is how the tag is written.
#[must_use]
pub fn read_xing(frame: &[u8], header: &Header) -> Option<Xing> {
    if header.layer != 3 {
        return None;
    }
    let base = 4 + side_info_len(header);
    read_xing_at(frame, base).or_else(|| {
        if header.crc {
            read_xing_at(frame, base + 2)
        } else {
            None
        }
    })
}

#[must_use]
pub fn read_vbri(frame: &[u8], header: &Header) -> Option<Vbri> {
    if header.layer != 3 {
        return None;
    }
    // 32 bytes after the 4-byte MPEG header.
    read_vbri_at(frame, 36)
}

#[must_use]
pub fn is_tag_frame(frame: &[u8], header: &Header) -> bool {
    read_xing(frame, header).is_some() || read_vbri(frame, header).is_some()
}

fn read_xing_at(frame: &[u8], offset: usize) -> Option<Xing> {
    if offset < 4 || frame.len() < offset + 8 {
        return None;
    }
    if frame[4..offset].iter().any(|byte| *byte != 0) {
        return None;
    }
    let id = &frame[offset..offset + 4];
    let cbr = if id == INFO {
        true
    } else if id == XING {
        false
    } else {
        return None;
    };
    let flags = read_u32(frame, offset + 4)?;
    let mut cursor = offset + 8;
    let frames = if flags & FLAG_FRAMES != 0 {
        let value = read_u32(frame, cursor)?;
        cursor += 4;
        if value == 0 { None } else { Some(value) }
    } else {
        None
    };
    let bytes = if flags & FLAG_BYTES != 0 {
        let value = read_u32(frame, cursor)?;
        cursor += 4;
        if value == 0 { None } else { Some(value) }
    } else {
        None
    };
    let toc = if flags & FLAG_TOC != 0 {
        if frame.len() < cursor + 100 {
            return None;
        }
        let mut toc = [0u8; 100];
        toc.copy_from_slice(&frame[cursor..cursor + 100]);
        cursor += 100;
        if toc_monotonic(&toc) { Some(toc) } else { None }
    } else {
        None
    };
    if flags & FLAG_QUALITY != 0 {
        if frame.len() < cursor + 4 {
            return None;
        }
        cursor += 4;
    }
    let (encoder_delay, trailing_padding) = read_lame_trim(frame, cursor);
    Some(Xing {
        cbr,
        frames,
        bytes,
        toc,
        encoder_delay,
        trailing_padding,
    })
}

fn toc_monotonic(toc: &[u8; 100]) -> bool {
    let mut prev = toc[0];
    for value in &toc[1..] {
        if *value < prev {
            return false;
        }
        prev = *value;
    }
    true
}

/// LAME writes the delay 21 bytes into the extension that follows the Xing
/// fields: 9 bytes of encoder name, then revision, lowpass, replay gain,
/// flags, and the ABR byte.
fn read_lame_trim(frame: &[u8], cursor: usize) -> (Option<u32>, Option<u32>) {
    if frame.len() < cursor + 24 {
        return (None, None);
    }
    let name = &frame[cursor..cursor + 4];
    if name != b"LAME" && name != b"Lavf" && name != b"Lavc" && name != b"Cutl" {
        return (None, None);
    }
    let delay_at = cursor + 21;
    let packed = (u32::from(frame[delay_at]) << 16)
        | (u32::from(frame[delay_at + 1]) << 8)
        | u32::from(frame[delay_at + 2]);
    let delay = packed >> 12;
    let padding = packed & 0x0FFF;
    // LAME rejects a delay or pad above 3000 samples as a false tag.
    if delay > 3000 || padding > 3000 {
        return (None, None);
    }
    (Some(delay), Some(padding))
}

fn read_vbri_at(frame: &[u8], offset: usize) -> Option<Vbri> {
    if frame.len() < offset + 26 || &frame[offset..offset + 4] != VBRI {
        return None;
    }
    let version = read_u16(frame, offset + 4)?;
    if version != 1 {
        return None;
    }
    let bytes = read_u32(frame, offset + 10)?;
    let frames_count = read_u32(frame, offset + 14)?;
    if bytes == 0 || frames_count == 0 {
        return None;
    }
    let entries = usize::from(read_u16(frame, offset + 18)?);
    let scale = u64::from(read_u16(frame, offset + 20)?);
    let entry_size = usize::from(read_u16(frame, offset + 22)?);
    let frames_per_entry = u32::from(read_u16(frame, offset + 24)?);
    if entries == 0
        || entries > MAX_VBRI_ENTRIES
        || scale == 0
        || !(1..=4).contains(&entry_size)
        || frames_per_entry == 0
    {
        return Some(Vbri {
            frames: frames_count,
            bytes,
            frames_per_entry: 0,
            starts: Vec::new(),
        });
    }
    let table_at = offset + 26;
    let table_bytes = entries.checked_mul(entry_size)?;
    if frame.len() < table_at + table_bytes {
        return Some(Vbri {
            frames: frames_count,
            bytes,
            frames_per_entry: 0,
            starts: Vec::new(),
        });
    }
    let mut starts = Vec::with_capacity(entries);
    let mut acc = 0u64;
    for index in 0..entries {
        let at = table_at + index * entry_size;
        let raw = read_entry(&frame[at..at + entry_size])?;
        starts.push(acc);
        acc = acc.saturating_add(raw.saturating_mul(scale));
    }
    Some(Vbri {
        frames: frames_count,
        bytes,
        frames_per_entry,
        starts,
    })
}

fn read_entry(bytes: &[u8]) -> Option<u64> {
    match bytes.len() {
        1 => Some(u64::from(bytes[0])),
        2 => Some(u64::from(read_u16(bytes, 0)?)),
        3 => Some((u64::from(bytes[0]) << 16) | (u64::from(bytes[1]) << 8) | u64::from(bytes[2])),
        4 => Some(u64::from(read_u32(bytes, 0)?)),
        _ => None,
    }
}

fn read_u32(buf: &[u8], at: usize) -> Option<u32> {
    let bytes: [u8; 4] = buf.get(at..at + 4)?.try_into().ok()?;
    Some(u32::from_be_bytes(bytes))
}

fn read_u16(buf: &[u8], at: usize) -> Option<u16> {
    let bytes: [u8; 2] = buf.get(at..at + 2)?.try_into().ok()?;
    Some(u16::from_be_bytes(bytes))
}

/// Byte offset from the start of the Xing frame for a presentation time.
///
/// `toc[i] / 256` is the fraction of `total_bytes` at `i` percent of `duration`.
#[must_use]
pub fn xing_offset(toc: &[u8; 100], total_bytes: u64, time: i64, duration: i64) -> u64 {
    if duration <= 0 || time <= 0 || total_bytes == 0 {
        return 0;
    }
    let duration = u128::from(duration as u64);
    let time = u128::from(time as u64);
    let index = ((time * 100) / duration).min(99) as usize;
    let base = u128::from(toc[0]);
    let fa = u128::from(toc[index]).saturating_sub(base);
    let fb = if index == 99 {
        256u128.saturating_sub(base)
    } else {
        u128::from(toc[index + 1]).saturating_sub(base)
    };
    let frac = (time * 100) % duration;
    let span = fb.saturating_sub(fa);
    let numer = fa * duration + span * frac;
    let pos = u128::from(total_bytes) * numer / (duration * 256);
    u64::try_from(pos).unwrap_or(total_bytes)
}

#[cfg(test)]
mod tests {
    use super::{read_vbri, read_xing};
    use crate::header::{MpegVersion, encode_layer3, parse_header, side_info_len};

    fn stereo_frame() -> (Vec<u8>, crate::header::Header) {
        let (raw, size) = encode_layer3(MpegVersion::Mpeg1, 128, 48_000, false, 2).expect("hdr");
        let mut frame = vec![0u8; usize::try_from(size).expect("size")];
        frame[..4].copy_from_slice(&raw.to_be_bytes());
        let header = parse_header(raw).expect("parse");
        (frame, header)
    }

    #[test]
    fn xing_toc_and_lame_delay_are_read_and_the_frame_is_not_audio() {
        let (mut frame, header) = stereo_frame();
        let at = 4 + side_info_len(&header);
        frame[at..at + 4].copy_from_slice(b"Xing");
        frame[at + 4..at + 8].copy_from_slice(&0x0Fu32.to_be_bytes());
        frame[at + 8..at + 12].copy_from_slice(&10u32.to_be_bytes());
        frame[at + 12..at + 16].copy_from_slice(&5000u32.to_be_bytes());
        for (index, value) in frame[at + 16..at + 116].iter_mut().enumerate() {
            *value = u8::try_from(index).unwrap_or(0);
        }
        frame[at + 120..at + 129].copy_from_slice(b"LAME3.100");
        let packed: u32 = (576 << 12) | 100;
        frame[at + 141..at + 144].copy_from_slice(&packed.to_be_bytes()[1..]);
        let tag = read_xing(&frame, &header).expect("xing");
        assert!(!tag.cbr);
        assert_eq!(tag.frames, Some(10));
        assert_eq!(tag.bytes, Some(5000));
        assert_eq!(tag.toc.expect("toc")[50], 50);
        assert_eq!(tag.encoder_delay, Some(576));
        assert_eq!(tag.trailing_padding, Some(100));
    }

    #[test]
    fn vbri_table_is_a_prefix_sum_of_scaled_entries() {
        let (mut frame, header) = stereo_frame();
        frame[36..40].copy_from_slice(b"VBRI");
        frame[40..42].copy_from_slice(&1u16.to_be_bytes());
        frame[46..50].copy_from_slice(&1000u32.to_be_bytes());
        frame[50..54].copy_from_slice(&4u32.to_be_bytes());
        frame[54..56].copy_from_slice(&2u16.to_be_bytes());
        frame[56..58].copy_from_slice(&1u16.to_be_bytes());
        frame[58..60].copy_from_slice(&2u16.to_be_bytes());
        frame[60..62].copy_from_slice(&2u16.to_be_bytes());
        frame[62..64].copy_from_slice(&10u16.to_be_bytes());
        frame[64..66].copy_from_slice(&20u16.to_be_bytes());
        let tag = read_vbri(&frame, &header).expect("vbri");
        assert_eq!(tag.frames, 4);
        assert_eq!(tag.frames_per_entry, 2);
        assert_eq!(tag.starts, vec![0, 10]);
    }
}
