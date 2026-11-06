//! Five-byte recognition of an Ogg file: the capture pattern and version.

use cutlass_codec::ContainerId;
use cutlass_container::{HeaderVerdict, Probe, ProbeResult};

use crate::limits::PROBE_BYTES;
use crate::page::CAPTURE;

/// The Ogg family. A capture pattern with a version other than 0 is this
/// family and is not opened.
#[derive(Clone, Copy, Debug)]
pub struct OggFamily;

impl Probe for OggFamily {
    const HEADER_BYTES: usize = PROBE_BYTES;

    fn family(&self) -> ContainerId {
        ContainerId::Ogg
    }

    fn matches(&self, header: &[u8]) -> HeaderVerdict {
        match header.get(..PROBE_BYTES) {
            Some([b'O', b'g', b'g', b'S', 0]) => HeaderVerdict::Match,
            Some(window) if window.starts_with(CAPTURE) => HeaderVerdict::UnsupportedBrand,
            _ => HeaderVerdict::Unrecognized,
        }
    }
}

/// Accepts a header that starts with an Ogg page.
#[must_use]
pub fn probe(header: &[u8]) -> ProbeResult {
    OggFamily.probe(header)
}

#[cfg(test)]
mod tests {
    use super::probe;
    use cutlass_codec::ContainerId;
    use cutlass_container::ProbeResult;

    #[test]
    fn the_capture_pattern_and_version_decide() {
        assert_eq!(probe(b"Ogg"), ProbeResult::NeedBytes { have: 3, need: 5 });
        assert_eq!(
            probe(b"OggS\0\x02"),
            ProbeResult::Accept {
                family: ContainerId::Ogg,
                examined: 5
            }
        );
        assert_eq!(
            probe(b"OggS\x01"),
            ProbeResult::Accept {
                family: ContainerId::Ogg,
                examined: 5
            }
        );
        assert_eq!(probe(b"RIFF\0"), ProbeResult::Refuse { examined: 5 });
        assert_eq!(probe(b"fLaC\0"), ProbeResult::Refuse { examined: 5 });
    }
}
