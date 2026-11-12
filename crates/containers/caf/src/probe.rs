//! Twelve-byte recognition: `caff`, `mFileVersion`, `mFileFlags`, then the
//! type of the first chunk, which the specification requires to be `desc`.

use cutlass_codec::ContainerId;
use cutlass_container::{HeaderVerdict, Probe, ProbeResult};

use crate::limits::PROBE_BYTES;

/// The CAF family.
#[derive(Clone, Copy, Debug)]
pub struct CafFamily;

impl Probe for CafFamily {
    const HEADER_BYTES: usize = PROBE_BYTES;

    fn family(&self) -> ContainerId {
        ContainerId::Caf
    }

    fn matches(&self, header: &[u8]) -> HeaderVerdict {
        verdict(header)
    }
}

/// Accepts `caff` version 1 followed by `desc`. A later version is the
/// family with an unsupported brand: open refuses it.
pub fn probe(header: &[u8]) -> ProbeResult {
    CafFamily.probe(header)
}

fn verdict(header: &[u8]) -> HeaderVerdict {
    if header.get(..4) != Some(b"caff") || header.get(8..12) != Some(b"desc") {
        return HeaderVerdict::Unrecognized;
    }
    if header.get(4..6) == Some(&[0, 1]) {
        HeaderVerdict::Match
    } else {
        HeaderVerdict::UnsupportedBrand
    }
}

#[cfg(test)]
mod tests {
    use super::probe;
    use cutlass_codec::ContainerId;
    use cutlass_container::ProbeResult;

    #[test]
    fn a_short_window_is_not_a_decision() {
        assert_eq!(probe(b""), ProbeResult::NeedBytes { have: 0, need: 12 });
        assert_eq!(
            probe(b"caff\x00\x01"),
            ProbeResult::NeedBytes { have: 6, need: 12 }
        );
    }

    #[test]
    fn caf_matches_and_other_files_do_not() {
        let accept = ProbeResult::Accept {
            family: ContainerId::Caf,
            examined: 12,
        };
        assert_eq!(probe(b"caff\x00\x01\x00\x00desc"), accept);
        // A later version is still this family; open refuses the brand.
        assert_eq!(probe(b"caff\x00\x02\x00\x00desc"), accept);
        assert_eq!(
            probe(b"caff\x00\x01\x00\x00data"),
            ProbeResult::Refuse { examined: 12 }
        );
        assert_eq!(
            probe(b"FORM\x00\x00\x00\x00AIFF"),
            ProbeResult::Refuse { examined: 12 }
        );
        assert_eq!(
            probe(b"RIFF\x00\x00\x00\x00WAVE"),
            ProbeResult::Refuse { examined: 12 }
        );
    }
}
