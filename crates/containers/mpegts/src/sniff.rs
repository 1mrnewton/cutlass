//! Naming private streams by their first bytes.
//!
//! A PMT may list audio as stream type `0x06` (private data) with nothing
//! that names the codec: FFmpeg's M2TS mode writes AAC that way, with only
//! a language descriptor. Such a stream is read as ADTS AAC, AC-3 or
//! E-AC-3 when its first bytes hold two frame headers of that syntax back
//! to back, the second where the first frame's length puts it. Anything
//! else stays [`Kind::Other`].

use std::io::Read;

use cutlass_container::SourceError;

use crate::es::{ac3, adts};
use crate::limits::{MAX_OPEN_SCAN_BYTES, TS_PACKET_BYTES};
use crate::packet;
use crate::psi::{Elementary, Kind};
use crate::source::{Next, Source};

/// Payload bytes gathered per stream: two of the largest ADTS frames.
const SNIFF_BYTES: usize = 2 * 8191 + 8;

/// Whether `elementary` is private data with nothing naming its codec.
const fn unnamed(elementary: &Elementary) -> bool {
    elementary.stream_type == 0x06
        && matches!(elementary.kind, Kind::Other)
        && elementary.registration.is_none()
}

/// Names the unnamed private streams of `elementaries` from the packets
/// `source` gives from `data_start`, until each is decided or has
/// [`SNIFF_BYTES`] of payload, or [`MAX_OPEN_SCAN_BYTES`] are read. The caller rewinds the
/// source after.
pub fn name_private_streams<R: Read>(
    source: &mut Source<R>,
    data_start: u64,
    elementaries: &mut [Elementary],
) -> Result<(), SourceError> {
    let mut gathered: Vec<(usize, Vec<u8>)> = elementaries
        .iter()
        .enumerate()
        .filter(|(_, elementary)| unnamed(elementary))
        .map(|(index, _)| (index, Vec::new()))
        .collect();
    while !gathered.is_empty() {
        let Next::Packet(position, index) = source.next()? else {
            break;
        };
        if position.saturating_sub(data_start) > MAX_OPEN_SCAN_BYTES {
            break;
        }
        let bytes = source.packet(index);
        let Ok(parsed) = packet::parse(bytes) else {
            continue;
        };
        let pid = parsed.header.pid;
        let Some(slot) = gathered
            .iter()
            .position(|(at, _)| elementaries[*at].pid == pid)
        else {
            continue;
        };
        let payload = &bytes[parsed.payload_at.min(TS_PACKET_BYTES)..];
        let data = &mut gathered[slot].1;
        if parsed.header.start {
            if let Some(body) = pes_body(payload) {
                data.extend_from_slice(body);
            }
        } else if !data.is_empty() {
            data.extend_from_slice(payload);
        }
        let decided = if data.len() >= SNIFF_BYTES {
            Some(kind_of_bytes(data))
        } else {
            decide(data)
        };
        if let Some(kind) = decided {
            let (at, _) = gathered.swap_remove(slot);
            elementaries[at].kind = kind;
        }
    }
    for (at, data) in gathered {
        elementaries[at].kind = kind_of_bytes(&data);
    }
    Ok(())
}

/// The bytes after the PES header at the start of `payload`.
fn pes_body(payload: &[u8]) -> Option<&[u8]> {
    let [0, 0, 1, stream_id, _, _, ..] = *payload else {
        return None;
    };
    // Private stream 1 and audio streams carry the optional header.
    if stream_id != 0xBD && stream_id & 0xE0 != 0xC0 {
        return None;
    }
    let length = usize::from(*payload.get(8)?);
    payload.get(9 + length..)
}

/// ADTS AAC, AC-3 or E-AC-3 when `data` starts with two headers of that
/// syntax a frame apart, [`Kind::Other`] when it cannot, and `None` while
/// the second header is not in `data` yet.
fn decide(data: &[u8]) -> Option<Kind> {
    if data.len() < 8 {
        return None;
    }
    if let Some(first) = adts::header(data) {
        let next = data.get(first.frame_len..)?;
        if next.len() < 8 {
            return None;
        }
        let same = adts::header(next).is_some_and(|second| {
            (second.rate_index, second.channel_config) == (first.rate_index, first.channel_config)
        });
        return Some(if same { Kind::Adts } else { Kind::Other });
    }
    if let Some(first) = ac3::header(data) {
        let next = data.get(first.len..)?;
        if next.len() < 8 {
            return None;
        }
        let same = ac3::header(next).is_some_and(|second| second.eac3 == first.eac3);
        return Some(match (same, first.eac3) {
            (false, _) => Kind::Other,
            (true, false) => Kind::Ac3,
            (true, true) => Kind::Eac3,
        });
    }
    Some(Kind::Other)
}

/// [`decide`], taking a stream that ends before the second header as
/// [`Kind::Other`].
fn kind_of_bytes(data: &[u8]) -> Kind {
    decide(data).unwrap_or(Kind::Other)
}

#[cfg(test)]
mod tests {
    use super::{Kind, decide, kind_of_bytes, pes_body};

    fn adts_frame(len: usize) -> Vec<u8> {
        let mut frame = vec![0_u8; len];
        frame[..7].copy_from_slice(&[
            0xFF,
            0xF1,
            0x50,
            0x80 | ((len >> 11) as u8 & 3),
            (len >> 3) as u8,
            ((len & 7) as u8) << 5 | 0x1F,
            0xFC,
        ]);
        frame
    }

    #[test]
    fn two_adts_frames_in_a_row_are_aac() {
        let mut data = adts_frame(100);
        data.extend(adts_frame(120));
        assert_eq!(kind_of_bytes(&data), Kind::Adts);
    }

    #[test]
    fn the_answer_waits_for_the_second_header() {
        let mut data = adts_frame(100);
        assert_eq!(decide(&data), None);
        data.extend(&adts_frame(120)[..7]);
        assert_eq!(decide(&data), None);
        data.push(0);
        assert_eq!(decide(&data), Some(Kind::Adts));
        assert_eq!(decide(&[0x47; 8]), Some(Kind::Other));
    }

    #[test]
    fn one_adts_header_alone_is_not_enough() {
        let mut data = adts_frame(100);
        data.extend([0x20, 0x00, 0x0F, 0x10]);
        assert_eq!(kind_of_bytes(&data), Kind::Other);
        assert_eq!(kind_of_bytes(&adts_frame(100)), Kind::Other);
        // DVB subtitles start with data identifier 0x20.
        assert_eq!(kind_of_bytes(&[0x20, 0x00, 0x0F, 0x10, 0, 0]), Kind::Other);
    }

    #[test]
    fn the_body_starts_after_the_optional_header() {
        let pes = [0, 0, 1, 0xBD, 0, 0, 0x80, 0x80, 5, 1, 2, 3, 4, 5, 0xAA];
        assert_eq!(pes_body(&pes), Some(&[0xAA][..]));
        // Padding and other streams without the optional header are skipped.
        assert_eq!(pes_body(&[0, 0, 1, 0xBE, 0, 0, 0, 0, 0]), None);
        assert_eq!(pes_body(&[0, 0, 1, 0xBD, 0, 0, 0x80, 0x80, 9]), None);
    }
}
