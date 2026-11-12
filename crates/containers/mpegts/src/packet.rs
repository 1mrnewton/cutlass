//! One transport packet: the 4-byte header, the adaptation field, and the
//! payload span (ITU-T H.222.0 clauses 2.4.3.2 to 2.4.3.5), plus the three
//! packet strides a file can use.

use crate::limits::{SYNC_BYTE, SYNC_RUN, TS_PACKET_BYTES};

/// The null packet PID. Its packets carry no data and no continuity.
pub const NULL_PID: u16 = 0x1FFF;

/// The PAT PID.
pub const PAT_PID: u16 = 0x0000;

/// How a file lays its 188-byte packets out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PacketSize {
    /// Plain transport stream: 188 bytes per packet.
    Ts,
    /// BDAV / AVCHD source packets (`.m2ts`, `.mts`): a 4-byte
    /// `TP_extra_header` (2-bit copy permission, 30-bit arrival time stamp)
    /// before each 188-byte packet.
    M2ts,
    /// 188 bytes followed by 16 bytes of Reed-Solomon parity.
    Fec,
}

impl PacketSize {
    pub const ALL: [Self; 3] = [Self::Ts, Self::M2ts, Self::Fec];

    /// Bytes from one packet unit to the next.
    #[must_use]
    pub const fn stride(self) -> usize {
        match self {
            Self::Ts => TS_PACKET_BYTES,
            Self::M2ts => TS_PACKET_BYTES + 4,
            Self::Fec => TS_PACKET_BYTES + 16,
        }
    }

    /// Where the sync byte sits inside a unit.
    #[must_use]
    pub const fn sync_offset(self) -> usize {
        match self {
            Self::Ts | Self::Fec => 0,
            Self::M2ts => 4,
        }
    }
}

/// Where packets start in a window of bytes, and at which stride.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    pub size: PacketSize,
    /// Offset of the first sync byte in the window.
    pub sync_at: usize,
    /// Sync bytes found in a row at that stride.
    pub run: usize,
}

/// Finds the stride and phase with the longest run of sync bytes whose
/// first sync lies in `[0, search)`. A run must reach `need` syncs, or the
/// end of the window after at least two, to count.
///
/// The work is bounded by `search * 3 * (window / 188)` comparisons.
#[must_use]
pub fn find_grid(window: &[u8], search: usize, need: usize) -> Option<Grid> {
    let mut best: Option<Grid> = None;
    let limit = search.min(window.len());
    for at in 0..limit {
        if window[at] != SYNC_BYTE {
            continue;
        }
        for size in PacketSize::ALL {
            let stride = size.stride();
            let mut run = 0;
            let mut probe = at;
            while probe < window.len() && window[probe] == SYNC_BYTE && run < need {
                run += 1;
                probe += stride;
            }
            let reached_end = probe >= window.len();
            let enough = run >= need || (reached_end && run >= 2);
            if !enough {
                continue;
            }
            if best.is_none_or(|grid| run > grid.run) {
                best = Some(Grid {
                    size,
                    sync_at: at,
                    run,
                });
            }
        }
        if best.is_some_and(|grid| grid.run >= need) {
            break;
        }
    }
    best
}

/// The probe's grid: [`SYNC_RUN`] sync bytes at one stride, the first in
/// the first 204 bytes.
#[must_use]
pub fn probe_grid(window: &[u8]) -> Option<Grid> {
    find_grid(window, PacketSize::Fec.stride(), SYNC_RUN).filter(|grid| grid.run >= SYNC_RUN)
}

/// The fixed fields of a packet header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub error: bool,
    pub start: bool,
    pub pid: u16,
    pub scrambling: u8,
    pub adaptation: bool,
    pub payload: bool,
    pub continuity: u8,
}

/// Reads the header of a 188-byte packet that starts with the sync byte.
#[must_use]
pub fn header(packet: &[u8; TS_PACKET_BYTES]) -> Header {
    let control = packet[3];
    Header {
        error: packet[1] & 0x80 != 0,
        start: packet[1] & 0x40 != 0,
        pid: (u16::from(packet[1] & 0x1F) << 8) | u16::from(packet[2]),
        scrambling: control >> 6,
        adaptation: control & 0x20 != 0,
        payload: control & 0x10 != 0,
        continuity: control & 0x0F,
    }
}

/// The adaptation field flags this crate acts on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Adaptation {
    pub discontinuity: bool,
    pub random_access: bool,
    /// Program clock reference in 27 MHz ticks: `base * 300 + extension`.
    /// The base is 33 bits, so the value is below `2^33 * 300`.
    pub pcr: Option<u64>,
}

/// A packet whose adaptation field cannot be read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PacketFault {
    /// `adaptation_field_length` is above 183 and runs past the packet.
    AdaptationOverrun { length: u8 },
    /// `adaptation_field_control` is `00`, which is reserved.
    ReservedControl,
}

/// What a packet holds once the adaptation field is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Parsed {
    pub header: Header,
    pub adaptation: Adaptation,
    /// Offset of the payload in the packet. Equal to 188 when there is none.
    pub payload_at: usize,
}

/// Splits a packet into header, adaptation field, and payload.
///
/// An adaptation field longer than the packet is a fault, not a payload
/// read past the end. A length that leaves no payload bytes behind is
/// accepted and gives an empty payload.
pub fn parse(packet: &[u8; TS_PACKET_BYTES]) -> Result<Parsed, PacketFault> {
    let header = header(packet);
    if !header.adaptation && !header.payload {
        return Err(PacketFault::ReservedControl);
    }
    let mut adaptation = Adaptation::default();
    let mut payload_at = 4;
    if header.adaptation {
        let length = packet[4];
        let end = 5 + usize::from(length);
        if end > TS_PACKET_BYTES {
            return Err(PacketFault::AdaptationOverrun { length });
        }
        if length > 0 {
            let flags = packet[5];
            adaptation.discontinuity = flags & 0x80 != 0;
            adaptation.random_access = flags & 0x40 != 0;
            if flags & 0x10 != 0 && length >= 7 {
                adaptation.pcr = Some(pcr(&packet[6..12]));
            }
        }
        payload_at = end;
    }
    if !header.payload {
        payload_at = TS_PACKET_BYTES;
    }
    Ok(Parsed {
        header,
        adaptation,
        payload_at,
    })
}

/// A 6-byte PCR field: 33-bit base, 6 reserved bits, 9-bit extension.
fn pcr(bytes: &[u8]) -> u64 {
    let base = (u64::from(bytes[0]) << 25)
        | (u64::from(bytes[1]) << 17)
        | (u64::from(bytes[2]) << 9)
        | (u64::from(bytes[3]) << 1)
        | (u64::from(bytes[4]) >> 7);
    let extension = (u64::from(bytes[4] & 1) << 8) | u64::from(bytes[5]);
    base * 300 + extension
}

/// Writes a PCR field for `value` 27 MHz ticks, reduced to the 33-bit base.
#[must_use]
pub fn pcr_bytes(value: u64) -> [u8; 6] {
    let base = (value / 300) & ((1 << 33) - 1);
    let extension = value % 300;
    [
        (base >> 25) as u8,
        (base >> 17) as u8,
        (base >> 9) as u8,
        (base >> 1) as u8,
        (((base & 1) << 7) as u8) | 0x7E | ((extension >> 8) as u8),
        extension as u8,
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        Adaptation, PacketFault, PacketSize, find_grid, header, parse, pcr, pcr_bytes, probe_grid,
    };
    use crate::limits::TS_PACKET_BYTES;

    fn packet(pid: u16, control: u8, body: &[u8]) -> [u8; TS_PACKET_BYTES] {
        let mut out = [0xFF; TS_PACKET_BYTES];
        out[0] = 0x47;
        out[1] = 0x40 | (pid >> 8) as u8;
        out[2] = pid as u8;
        out[3] = control;
        out[4..4 + body.len()].copy_from_slice(body);
        out
    }

    #[test]
    fn the_header_fields_come_from_their_bits() {
        let bytes = packet(0x1ABC & 0x1FFF, 0xB7, &[]);
        let fields = header(&bytes);
        assert!(fields.start);
        assert!(!fields.error);
        assert_eq!(fields.pid, 0x1ABC);
        assert_eq!(fields.scrambling, 2);
        assert!(fields.adaptation);
        assert!(fields.payload);
        assert_eq!(fields.continuity, 7);
    }

    #[test]
    fn a_pcr_round_trips_through_its_six_bytes() {
        for value in [0, 1, 299, 300, 27_000_000, ((1_u64 << 33) - 1) * 300 + 299] {
            assert_eq!(pcr(&pcr_bytes(value)), value, "{value}");
        }
    }

    #[test]
    fn the_adaptation_field_gives_flags_pcr_and_payload_start() {
        let mut body = vec![7, 0x50];
        body.extend_from_slice(&pcr_bytes(27_000_000 * 3 + 5));
        let bytes = packet(0x100, 0x30, &body);
        let parsed = parse(&bytes).expect("packet");
        assert_eq!(
            parsed.adaptation,
            Adaptation {
                discontinuity: false,
                random_access: true,
                pcr: Some(27_000_000 * 3 + 5)
            }
        );
        assert_eq!(parsed.payload_at, 12);
    }

    #[test]
    fn an_adaptation_field_past_the_packet_is_a_fault() {
        let bytes = packet(0x100, 0x30, &[184]);
        assert_eq!(
            parse(&bytes),
            Err(PacketFault::AdaptationOverrun { length: 184 })
        );
        let full = packet(0x100, 0x30, &[183]);
        assert_eq!(parse(&full).expect("full").payload_at, TS_PACKET_BYTES);
        let reserved = packet(0x100, 0x00, &[]);
        assert_eq!(parse(&reserved), Err(PacketFault::ReservedControl));
    }

    #[test]
    fn a_short_adaptation_field_carries_no_pcr() {
        // The PCR flag with too few bytes behind it is not read.
        let bytes = packet(0x100, 0x30, &[3, 0x10, 1, 2]);
        let parsed = parse(&bytes).expect("packet");
        assert_eq!(parsed.adaptation.pcr, None);
        assert_eq!(parsed.payload_at, 8);
    }

    fn units(size: PacketSize, count: usize, lead: usize) -> Vec<u8> {
        let mut bytes = vec![0_u8; lead];
        for _ in 0..count {
            let mut unit = vec![0_u8; size.stride()];
            unit[size.sync_offset()] = 0x47;
            bytes.extend_from_slice(&unit);
        }
        bytes
    }

    #[test]
    fn every_stride_and_phase_is_found() {
        for size in PacketSize::ALL {
            for lead in [0, 3, 100] {
                let bytes = units(size, 8, lead);
                let grid = probe_grid(&bytes).expect("grid");
                assert_eq!(grid.size, size, "{size:?} {lead}");
                assert_eq!(grid.sync_at, lead + size.sync_offset());
            }
        }
    }

    #[test]
    fn four_syncs_are_not_a_transport_stream() {
        let mut bytes = units(PacketSize::Ts, 4, 0);
        bytes.extend_from_slice(&[0; 400]);
        assert_eq!(probe_grid(&bytes), None);
        assert_eq!(find_grid(&[0; 2000], 204, 5), None);
    }
}
