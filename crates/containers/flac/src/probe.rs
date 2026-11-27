//! Recognition of the `fLaC` marker, at the start or behind ID3v2 tags.

use cutlass_codec::ContainerId;
use cutlass_container::{HeaderVerdict, Probe, ProbeResult};

use crate::tag::marker_in;

/// The native FLAC family.
#[derive(Clone, Copy, Debug)]
pub struct FlacFamily;

impl Probe for FlacFamily {
    const HEADER_BYTES: usize = 4;

    fn family(&self) -> ContainerId {
        ContainerId::Flac
    }

    fn matches(&self, header: &[u8]) -> HeaderVerdict {
        if marker_in(header).is_some() {
            HeaderVerdict::Match
        } else {
            HeaderVerdict::Unrecognized
        }
    }
}

/// Accepts the `fLaC` marker, at the start of `header` or behind ID3v2
/// tags that end inside it. A tag that runs past the header is refused: the
/// MP3 family claims those bytes, and [`starts_with_flac`] tells such a FLAC
/// file apart.
///
/// [`starts_with_flac`]: crate::starts_with_flac
#[must_use]
pub fn probe(header: &[u8]) -> ProbeResult {
    match marker_in(header) {
        Some(at) if at > 0 => ProbeResult::Accept {
            family: ContainerId::Flac,
            examined: at + 4,
        },
        _ => FlacFamily.probe(header),
    }
}

#[cfg(test)]
mod tests {
    use super::probe;
    use cutlass_codec::ContainerId;
    use cutlass_container::ProbeResult;

    #[test]
    fn the_marker_matches_and_other_bytes_do_not() {
        assert_eq!(
            probe(b"fLaC\x00\x00\x00\x22"),
            ProbeResult::Accept {
                family: ContainerId::Flac,
                examined: 4
            }
        );
        assert_eq!(probe(b"fLa"), ProbeResult::NeedBytes { have: 3, need: 4 });
        assert_eq!(probe(b"RIFF"), ProbeResult::Refuse { examined: 4 });
        assert_eq!(probe(b"ID3\x04"), ProbeResult::Refuse { examined: 4 });
    }

    #[test]
    fn the_marker_behind_a_tag_matches_when_the_header_holds_both() {
        let mut header = b"ID3\x03\x00\x00\x00\x00\x01\x00".to_vec();
        header.resize(10 + 128, 0);
        header.extend(b"fLaC\x00\x00\x00\x22");
        assert_eq!(
            probe(&header),
            ProbeResult::Accept {
                family: ContainerId::Flac,
                examined: 10 + 128 + 4
            }
        );
        // The tag ends past the header, or MPEG audio follows it.
        assert_eq!(probe(&header[..100]), ProbeResult::Refuse { examined: 4 });
        header.splice(138..142, [0xFF, 0xFB, 0x90, 0x64]);
        assert_eq!(probe(&header), ProbeResult::Refuse { examined: 4 });
    }
}
