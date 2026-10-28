//! Family probe.
//!
//! Ten bytes of a valid ID3v2 header match this family. A Layer III frame
//! matches when a second compatible header confirms it. Four bytes of sync
//! alone do not match: that pattern occurs inside other media.

use crate::api::{ContainerError, Probe};
use crate::header::{compatible, parse_header};
use crate::id3::{id3v2_tag_bytes, is_id3v2};
use crate::limits::{MAX_FRAME_BYTES, MAX_RESYNC_BYTES};

#[must_use]
pub fn probe(prefix: &[u8]) -> Probe {
    if prefix.len() >= 10 && is_id3v2(prefix) {
        return match id3v2_tag_bytes(prefix) {
            Ok(Some(len)) => {
                let have = u64::try_from(prefix.len()).unwrap_or(u64::MAX);
                if have < len {
                    Probe::NeedMore {
                        minimum: len.saturating_add(4),
                    }
                } else {
                    Probe::Matched
                }
            }
            // A hostile or broken ID3 header is still this family's problem.
            Ok(None) | Err(ContainerError::Limit { .. } | ContainerError::Malformed { .. }) => {
                Probe::Matched
            }
            Err(_) => Probe::Reject,
        };
    }
    let hunt_len = usize::try_from(MAX_RESYNC_BYTES).unwrap_or(prefix.len());
    let hunt = prefix.len().min(hunt_len);
    if let Some(hit) = confirmed_layer3(&prefix[..hunt], prefix) {
        let _ = hit;
        return Probe::Matched;
    }
    if let Some(minimum) = unconfirmed_layer3(prefix) {
        return Probe::NeedMore { minimum };
    }
    if prefix.len() < 4 {
        return Probe::NeedMore { minimum: 4 };
    }
    Probe::Reject
}

/// A Layer III header whose following header is not in `prefix` yet.
fn unconfirmed_layer3(prefix: &[u8]) -> Option<u64> {
    let hunt_len = usize::try_from(MAX_RESYNC_BYTES).unwrap_or(prefix.len());
    let hunt = prefix.len().min(hunt_len);
    let mut index = 0usize;
    while index + 4 <= hunt {
        let Some(header) = read_raw(&prefix[index..]) else {
            index += 1;
            continue;
        };
        if header.layer != 3 {
            index += 1;
            continue;
        }
        let size = header.frame_bytes.unwrap_or(MAX_FRAME_BYTES);
        let Ok(size_us) = usize::try_from(size) else {
            index += 1;
            continue;
        };
        let Some(next) = index.checked_add(size_us) else {
            index += 1;
            continue;
        };
        if next.saturating_add(4) > prefix.len() {
            return u64::try_from(next.saturating_add(4)).ok();
        }
        index += 1;
    }
    None
}

fn confirmed_layer3(hunt: &[u8], full: &[u8]) -> Option<usize> {
    let mut index = 0usize;
    while index + 4 <= hunt.len() {
        let Some(header) = read_raw(&full[index..]) else {
            index += 1;
            continue;
        };
        if header.layer != 3 {
            index += 1;
            continue;
        }
        let Some(size) = header.frame_bytes else {
            index += 1;
            continue;
        };
        let Ok(size_us) = usize::try_from(size) else {
            index += 1;
            continue;
        };
        let Some(next) = index.checked_add(size_us) else {
            index += 1;
            continue;
        };
        if next + 4 <= full.len()
            && read_raw(&full[next..]).is_some_and(|n| compatible(&header, &n))
        {
            return Some(index);
        }
        index += 1;
    }
    None
}

fn read_raw(buf: &[u8]) -> Option<crate::header::Header> {
    if buf.len() < 4 {
        return None;
    }
    let raw = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
    parse_header(raw)
}

#[cfg(test)]
mod tests {
    use super::probe;
    use crate::api::Probe;
    use crate::header::{MpegVersion, synthetic_frame};

    #[test]
    fn id3_header_matches_and_a_movie_box_does_not() {
        let mut id3 = [0u8; 10];
        id3[0] = b'I';
        id3[1] = b'D';
        id3[2] = b'3';
        id3[3] = 4;
        id3[9] = 20;
        assert_eq!(probe(&id3), Probe::NeedMore { minimum: 34 });
        id3[9] = 0;
        assert_eq!(probe(&id3), Probe::Matched);
        assert_eq!(probe(b"ftypmp42"), Probe::Reject);
        assert_eq!(
            probe(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]),
            Probe::Reject
        );
    }

    #[test]
    fn two_layer3_frames_match_and_one_frame_asks_for_more() {
        let frame = synthetic_frame(MpegVersion::Mpeg1, 128, 48_000, 2, 0);
        let mut two = frame.clone();
        two.extend_from_slice(&frame);
        assert_eq!(probe(&two), Probe::Matched);
        assert!(matches!(probe(&frame), Probe::NeedMore { .. }));
    }
}
