//! Four-byte recognition of the `fLaC` marker.

use cutlass_codec::ContainerId;
use cutlass_container::{HeaderVerdict, Probe, ProbeResult};

/// The native FLAC family.
#[derive(Clone, Copy, Debug)]
pub struct FlacFamily;

impl Probe for FlacFamily {
    const HEADER_BYTES: usize = 4;

    fn family(&self) -> ContainerId {
        ContainerId::Flac
    }

    fn matches(&self, header: &[u8]) -> HeaderVerdict {
        if header.get(..4) == Some(b"fLaC") {
            HeaderVerdict::Match
        } else {
            HeaderVerdict::Unrecognized
        }
    }
}

/// Accepts the `fLaC` marker. A file that starts with an ID3v2 tag is
/// refused: the MP3 family claims those bytes.
#[must_use]
pub fn probe(header: &[u8]) -> ProbeResult {
    FlacFamily.probe(header)
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
}
