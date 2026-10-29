//! Twelve-byte recognition of RIFF WAVE and RF64.

use cutlass_codec::ContainerId;
use cutlass_container::{HeaderVerdict, Probe, ProbeResult};

use crate::limits::PROBE_BYTES;

/// The WAV family. Big-endian `RIFX` and `BW64` are this family and are not opened.
#[derive(Clone, Copy, Debug)]
pub struct WavFamily;

impl Probe for WavFamily {
    const HEADER_BYTES: usize = PROBE_BYTES;

    fn family(&self) -> ContainerId {
        ContainerId::Riff
    }

    fn matches(&self, header: &[u8]) -> HeaderVerdict {
        verdict(header)
    }
}

pub fn probe(header: &[u8]) -> ProbeResult {
    WavFamily.probe(header)
}

pub(crate) fn verdict(header: &[u8]) -> HeaderVerdict {
    let Some(form) = header.get(8..12) else {
        return HeaderVerdict::Unrecognized;
    };
    if form != b"WAVE" {
        return HeaderVerdict::Unrecognized;
    }
    match header.get(..4) {
        Some(b"RIFF" | b"RF64") => HeaderVerdict::Match,
        Some(b"RIFX" | b"BW64") => HeaderVerdict::UnsupportedBrand,
        _ => HeaderVerdict::Unrecognized,
    }
}

#[cfg(test)]
mod tests {
    use super::probe;
    use cutlass_codec::ContainerId;
    use cutlass_container::ProbeResult;

    #[test]
    fn a_short_window_is_not_a_decision() {
        assert_eq!(probe(b"RIFF"), ProbeResult::NeedBytes { have: 4, need: 12 });
        assert_eq!(probe(b""), ProbeResult::NeedBytes { have: 0, need: 12 });
    }

    #[test]
    fn riff_and_rf64_match_and_other_forms_do_not() {
        assert_eq!(
            probe(b"RIFF\x00\x00\x00\x00WAVE"),
            ProbeResult::Accept {
                family: ContainerId::Riff,
                examined: 12
            }
        );
        assert_eq!(
            probe(b"RF64\xff\xff\xff\xffWAVE"),
            ProbeResult::Accept {
                family: ContainerId::Riff,
                examined: 12
            }
        );
        assert_eq!(
            probe(b"RIFX\x00\x00\x00\x00WAVE"),
            ProbeResult::Accept {
                family: ContainerId::Riff,
                examined: 12
            }
        );
        assert_eq!(
            probe(b"FORM\x00\x00\x00\x00AIFF"),
            ProbeResult::Refuse { examined: 12 }
        );
        assert_eq!(
            probe(b"RIFF\x00\x00\x00\x00AVI "),
            ProbeResult::Refuse { examined: 12 }
        );
    }
}
