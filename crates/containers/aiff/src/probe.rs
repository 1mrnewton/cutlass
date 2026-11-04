//! Twelve-byte recognition of `FORM` / `AIFF` and `FORM` / `AIFC`.

use cutlass_codec::ContainerId;
use cutlass_container::{HeaderVerdict, Probe, ProbeResult};

use crate::limits::PROBE_BYTES;

/// The AIFF family.
#[derive(Clone, Copy, Debug)]
pub struct AiffFamily;

impl Probe for AiffFamily {
    const HEADER_BYTES: usize = PROBE_BYTES;

    fn family(&self) -> ContainerId {
        ContainerId::Aiff
    }

    fn matches(&self, header: &[u8]) -> HeaderVerdict {
        verdict(header)
    }
}

pub fn probe(header: &[u8]) -> ProbeResult {
    AiffFamily.probe(header)
}

fn verdict(header: &[u8]) -> HeaderVerdict {
    if &header[..4] != b"FORM" {
        return HeaderVerdict::Unrecognized;
    }
    if &header[8..12] == b"AIFF" || &header[8..12] == b"AIFC" {
        HeaderVerdict::Match
    } else {
        HeaderVerdict::Unrecognized
    }
}

#[cfg(test)]
mod tests {
    use super::probe;
    use cutlass_codec::ContainerId;
    use cutlass_container::ProbeResult;

    #[test]
    fn a_short_window_is_not_a_decision() {
        assert_eq!(probe(b"FORM"), ProbeResult::NeedBytes { have: 4, need: 12 });
        assert_eq!(probe(b""), ProbeResult::NeedBytes { have: 0, need: 12 });
    }

    #[test]
    fn aiff_and_aifc_match_and_other_forms_do_not() {
        assert_eq!(
            probe(b"FORM\x00\x00\x00\x00AIFF"),
            ProbeResult::Accept {
                family: ContainerId::Aiff,
                examined: 12
            }
        );
        assert_eq!(
            probe(b"FORM\xff\xff\xff\xffAIFC"),
            ProbeResult::Accept {
                family: ContainerId::Aiff,
                examined: 12
            }
        );
        assert_eq!(
            probe(b"FORM\x00\x00\x00\x048SVX"),
            ProbeResult::Refuse { examined: 12 }
        );
        assert_eq!(
            probe(b"RIFF\x00\x00\x00\x00WAVE"),
            ProbeResult::Refuse { examined: 12 }
        );
    }
}
