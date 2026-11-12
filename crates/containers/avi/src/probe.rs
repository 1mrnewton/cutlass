//! Twelve-byte recognition of an AVI file.

use cutlass_codec::ContainerId;
use cutlass_container::{HeaderVerdict, Probe, ProbeResult};

use crate::limits::PROBE_BYTES;

/// The AVI family. A file that starts with an OpenDML `AVIX` segment, the
/// broken `AVI\x19` form, or big-endian `RIFX` is this family and does not open.
#[derive(Clone, Copy, Debug)]
pub struct AviFamily;

impl Probe for AviFamily {
    const HEADER_BYTES: usize = PROBE_BYTES;

    fn family(&self) -> ContainerId {
        ContainerId::Avi
    }

    fn matches(&self, header: &[u8]) -> HeaderVerdict {
        verdict(header)
    }
}

/// Accepts, refuses, or asks for more bytes.
#[must_use]
pub fn probe(header: &[u8]) -> ProbeResult {
    AviFamily.probe(header)
}

pub(crate) fn verdict(header: &[u8]) -> HeaderVerdict {
    let (Some(riff), Some(form)) = (header.get(..4), header.get(8..12)) else {
        return HeaderVerdict::Unrecognized;
    };
    match (riff, form) {
        (b"RIFF", b"AVI ") => HeaderVerdict::Match,
        (b"RIFF", b"AVIX" | b"AVI\x19") | (b"RIFX", b"AVI ") => HeaderVerdict::UnsupportedBrand,
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
    fn avi_matches_and_wave_and_other_forms_do_not() {
        let accept = ProbeResult::Accept {
            family: ContainerId::Avi,
            examined: 12,
        };
        assert_eq!(probe(b"RIFF\0\0\0\0AVI more"), accept);
        assert_eq!(probe(b"RIFF\0\0\0\0AVIX"), accept);
        assert_eq!(probe(b"RIFX\0\0\0\0AVI "), accept);
        assert_eq!(
            probe(b"RIFF\0\0\0\0WAVE"),
            ProbeResult::Refuse { examined: 12 }
        );
        assert_eq!(
            probe(b"RIFF\0\0\0\0AMV "),
            ProbeResult::Refuse { examined: 12 }
        );
        assert_eq!(
            cutlass_riff::probe(b"RIFF\0\0\0\0AVI "),
            ProbeResult::Refuse { examined: 12 }
        );
    }
}
