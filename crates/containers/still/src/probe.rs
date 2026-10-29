//! Signature probe for JPEG, PNG, WebP, and GIF.
//!
//! The window grows until the signature is complete. A prefix that cannot
//! become one of those four is refused. A movie box, a RIFF `AVI ` form,
//! and a Matroska header are refused. They are not opened as stills.

use cutlass_codec::ContainerId;
use cutlass_container::{HeaderVerdict, Probe, ProbeResult};

/// Twelve bytes is enough to tell a WebP form from any other RIFF form.
pub const PROBE_BYTES: usize = 12;

const PNG: [u8; 8] = [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1A, b'\n'];

/// Header recognition for this family.
///
/// [`probe`] is the function. This type exists so the shared [`Probe`]
/// trait can be used by a reader that holds one probe per family.
#[derive(Clone, Copy, Debug, Default)]
pub struct StillProbe;

impl Probe for StillProbe {
    const HEADER_BYTES: usize = PROBE_BYTES;

    fn family(&self) -> ContainerId {
        ContainerId::Still
    }

    fn matches(&self, header: &[u8]) -> HeaderVerdict {
        match probe(header) {
            ProbeResult::Accept { .. } => HeaderVerdict::Match,
            ProbeResult::Refuse { .. } | ProbeResult::NeedBytes { .. } => {
                HeaderVerdict::Unrecognized
            }
        }
    }

    fn probe(&self, header: &[u8]) -> ProbeResult {
        probe(header)
    }
}

/// Accepts a finished signature and refuses anything else.
///
/// A short prefix of JPEG, PNG, GIF, or `RIFF` is [`ProbeResult::NeedBytes`].
/// It is not a refusal and not an acceptance.
#[must_use]
pub fn probe(header: &[u8]) -> ProbeResult {
    if header.is_empty() {
        return ProbeResult::NeedBytes { have: 0, need: 2 };
    }
    match header[0] {
        0xFF => probe_jpeg(header),
        0x89 => probe_png(header),
        b'G' => probe_gif(header),
        b'R' => probe_riff(header),
        _ => ProbeResult::Refuse { examined: 1 },
    }
}

fn probe_jpeg(header: &[u8]) -> ProbeResult {
    if header.len() < 2 {
        return ProbeResult::NeedBytes { have: 1, need: 2 };
    }
    if header[1] == 0xD8 {
        ProbeResult::Accept {
            family: ContainerId::Still,
            examined: 2,
        }
    } else {
        ProbeResult::Refuse { examined: 2 }
    }
}

fn probe_png(header: &[u8]) -> ProbeResult {
    if header.len() < PNG.len() {
        if PNG.starts_with(header) {
            return ProbeResult::NeedBytes {
                have: header.len(),
                need: PNG.len(),
            };
        }
        return ProbeResult::Refuse {
            examined: header.len(),
        };
    }
    if header.starts_with(&PNG) {
        ProbeResult::Accept {
            family: ContainerId::Still,
            examined: PNG.len(),
        }
    } else {
        ProbeResult::Refuse {
            examined: PNG.len(),
        }
    }
}

fn probe_gif(header: &[u8]) -> ProbeResult {
    const NEED: usize = 6;
    if header.len() < 3 {
        if b"GIF".starts_with(header) {
            return ProbeResult::NeedBytes {
                have: header.len(),
                need: NEED,
            };
        }
        return ProbeResult::Refuse {
            examined: header.len(),
        };
    }
    if &header[..3] != b"GIF" {
        return ProbeResult::Refuse { examined: 3 };
    }
    if header.len() < NEED {
        return ProbeResult::NeedBytes {
            have: header.len(),
            need: NEED,
        };
    }
    if header.starts_with(b"GIF87a") || header.starts_with(b"GIF89a") {
        ProbeResult::Accept {
            family: ContainerId::Still,
            examined: NEED,
        }
    } else {
        ProbeResult::Refuse { examined: NEED }
    }
}

fn probe_riff(header: &[u8]) -> ProbeResult {
    const NEED: usize = 12;
    if header.len() < 4 {
        if b"RIFF".starts_with(header) {
            return ProbeResult::NeedBytes {
                have: header.len(),
                need: NEED,
            };
        }
        return ProbeResult::Refuse {
            examined: header.len(),
        };
    }
    if &header[..4] != b"RIFF" {
        return ProbeResult::Refuse { examined: 4 };
    }
    if header.len() < NEED {
        return ProbeResult::NeedBytes {
            have: header.len(),
            need: NEED,
        };
    }
    if &header[8..12] == b"WEBP" {
        ProbeResult::Accept {
            family: ContainerId::Still,
            examined: NEED,
        }
    } else {
        ProbeResult::Refuse { examined: NEED }
    }
}
