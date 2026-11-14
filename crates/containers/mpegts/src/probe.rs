//! Recognition of a transport stream from its first kilobyte.
//!
//! Five sync bytes in a row at one of the three strides (188, 192, 204),
//! the first anywhere in the first 204 bytes, name the family. A random
//! window has about one chance in 10^9 of doing that.

use cutlass_codec::ContainerId;
use cutlass_container::{HeaderVerdict, Probe, ProbeResult};

use crate::limits::PROBE_BYTES;
use crate::packet::probe_grid;

/// The MPEG transport stream family: `.ts`, `.m2ts`, `.mts`.
#[derive(Clone, Copy, Debug)]
pub struct TsFamily;

impl Probe for TsFamily {
    const HEADER_BYTES: usize = PROBE_BYTES;

    fn family(&self) -> ContainerId {
        ContainerId::MpegTs
    }

    fn matches(&self, header: &[u8]) -> HeaderVerdict {
        let window = header.get(..PROBE_BYTES).unwrap_or(header);
        if probe_grid(window).is_some() {
            HeaderVerdict::Match
        } else {
            HeaderVerdict::Unrecognized
        }
    }
}

/// Accepts a header that holds five packets at one stride.
#[must_use]
pub fn probe(header: &[u8]) -> ProbeResult {
    TsFamily.probe(header)
}

#[cfg(test)]
mod tests {
    use super::probe;
    use crate::limits::PROBE_BYTES;
    use crate::packet::PacketSize;
    use cutlass_codec::ContainerId;
    use cutlass_container::ProbeResult;

    fn stream(size: PacketSize, lead: usize) -> Vec<u8> {
        let mut bytes = vec![0x11; lead];
        while bytes.len() < 4096 {
            let mut unit = vec![0_u8; size.stride()];
            unit[size.sync_offset()] = 0x47;
            bytes.extend_from_slice(&unit);
        }
        bytes
    }

    #[test]
    fn every_packet_size_is_accepted() {
        for size in PacketSize::ALL {
            assert_eq!(
                probe(&stream(size, 0)),
                ProbeResult::Accept {
                    family: ContainerId::MpegTs,
                    examined: PROBE_BYTES
                },
                "{size:?}"
            );
        }
        assert!(matches!(
            probe(&stream(PacketSize::Ts, 77)),
            ProbeResult::Accept { .. }
        ));
    }

    #[test]
    fn a_short_window_asks_for_more_and_other_files_are_refused() {
        assert_eq!(
            probe(&[0x47; 10]),
            ProbeResult::NeedBytes {
                have: 10,
                need: PROBE_BYTES
            }
        );
        let mut mp4 = vec![0, 0, 0, 0x20];
        mp4.extend_from_slice(b"ftypisom");
        mp4.resize(PROBE_BYTES, 0);
        assert_eq!(
            probe(&mp4),
            ProbeResult::Refuse {
                examined: PROBE_BYTES
            }
        );
        let mut ogg = b"OggS\0".to_vec();
        ogg.resize(PROBE_BYTES, 0x47);
        // A run of 0x47 bytes is five syncs at every stride: still a match,
        // and open decides. Ogg's own probe runs first in every registry.
        assert!(matches!(probe(&ogg), ProbeResult::Accept { .. }));
    }
}
