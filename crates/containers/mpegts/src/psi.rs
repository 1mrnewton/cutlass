//! The program association and program map tables (ITU-T H.222.0 clauses
//! 2.4.4.3 and 2.4.4.8), the descriptors this crate reads (clause 2.6,
//! ETSI EN 300 468 annex D, ATSC A/52 annex A), and the map from
//! `stream_type` to how an elementary stream is framed.

use crate::limits::{MAX_DESCRIPTORS, MAX_PROGRAMS};
use crate::section::{SectionError, long_section};

/// `table_id` of a PAT.
pub const PAT_TABLE: u8 = 0x00;
/// `table_id` of a PMT.
pub const PMT_TABLE: u8 = 0x02;

/// One row of the PAT. Program 0 names the network PID and is not kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PatEntry {
    pub program: u16,
    pub pmt_pid: u16,
}

/// A PAT section that checked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pat {
    pub transport_stream_id: u16,
    pub version: u8,
    pub section: u8,
    pub last_section: u8,
    pub programs: Vec<PatEntry>,
}

/// Why a table was not used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableError {
    Section(SectionError),
    /// Another table id than the one this PID carries.
    WrongTable {
        table_id: u8,
    },
    /// `current_next_indicator` is 0: the table is not in force yet.
    NotCurrent,
    /// A length inside the section runs past its body.
    Overrun,
}

impl From<SectionError> for TableError {
    fn from(error: SectionError) -> Self {
        Self::Section(error)
    }
}

/// Reads a PAT section.
pub fn parse_pat(section: &[u8]) -> Result<Pat, TableError> {
    let (header, body) = long_section(section)?;
    if header.table_id != PAT_TABLE {
        return Err(TableError::WrongTable {
            table_id: header.table_id,
        });
    }
    if !header.current {
        return Err(TableError::NotCurrent);
    }
    let mut programs = Vec::new();
    for entry in body.as_chunks::<4>().0.iter().take(MAX_PROGRAMS) {
        let program = u16::from_be_bytes([entry[0], entry[1]]);
        let pid = u16::from_be_bytes([entry[2] & 0x1F, entry[3]]);
        if program != 0 {
            programs.push(PatEntry {
                program,
                pmt_pid: pid,
            });
        }
    }
    Ok(Pat {
        transport_stream_id: header.id,
        version: header.version,
        section: header.number,
        last_section: header.last,
        programs,
    })
}

/// How an elementary stream's bytes split into access units.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    H264,
    Hevc,
    /// MPEG-1 or MPEG-2 video.
    Mpeg2Video,
    /// MPEG-4 Part 2 visual: one PES packet is one access unit.
    Mpeg4Video,
    /// AAC in ADTS frames.
    Adts,
    /// AAC in LOAS frames carrying LATM.
    Latm,
    Ac3,
    Eac3,
    /// MPEG-1 or MPEG-2 audio, layer I, II, or III.
    MpegAudio,
    /// Opus, each packet behind the `0x7FE0` control header that FFmpeg
    /// and GStreamer write and read.
    Opus,
    /// Anything else carried in PES packets: one PES packet is one packet.
    Other,
}

/// One elementary stream of a PMT.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Elementary {
    pub stream_type: u8,
    pub pid: u16,
    pub kind: Kind,
    /// The first ISO 639 language code.
    pub language: Option<[u8; 3]>,
    /// The registration descriptor's `format_identifier`.
    pub registration: Option<[u8; 4]>,
    /// The Opus `channel_config_code` from the DVB extension descriptor.
    pub opus_channels: Option<u8>,
}

/// A PMT section that checked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pmt {
    pub program: u16,
    pub version: u8,
    /// `None` when the PMT says `0x1FFF`: the program has no PCR.
    pub pcr_pid: Option<u16>,
    pub registration: Option<[u8; 4]>,
    pub streams: Vec<Elementary>,
}

#[derive(Default)]
struct Descriptors {
    registration: Option<[u8; 4]>,
    language: Option<[u8; 3]>,
    ac3: bool,
    eac3: bool,
    opus_channels: Option<u8>,
}

fn descriptors(mut bytes: &[u8]) -> Result<Descriptors, TableError> {
    let mut found = Descriptors::default();
    for _ in 0..MAX_DESCRIPTORS {
        let [tag, length, rest @ ..] = bytes else {
            break;
        };
        let length = usize::from(*length);
        let Some(body) = rest.get(..length) else {
            return Err(TableError::Overrun);
        };
        match (*tag, body) {
            (0x05, [a, b, c, d, ..]) => found.registration = Some([*a, *b, *c, *d]),
            (0x0A, [a, b, c, ..]) if found.language.is_none() => {
                if [*a, *b, *c].iter().all(u8::is_ascii_alphabetic) {
                    found.language = Some([*a, *b, *c]);
                }
            }
            (0x6A, _) | (0x81, _) => found.ac3 = true,
            (0x7A, _) | (0xCC, _) => found.eac3 = true,
            (0x7F, [0x80, code, ..]) => found.opus_channels = Some(*code),
            _ => {}
        }
        bytes = &rest[length..];
    }
    Ok(found)
}

/// Reads a PMT section.
pub fn parse_pmt(section: &[u8]) -> Result<Pmt, TableError> {
    let (header, body) = long_section(section)?;
    if header.table_id != PMT_TABLE {
        return Err(TableError::WrongTable {
            table_id: header.table_id,
        });
    }
    if !header.current {
        return Err(TableError::NotCurrent);
    }
    let [p0, p1, i0, i1, rest @ ..] = body else {
        return Err(TableError::Overrun);
    };
    let pcr = u16::from_be_bytes([p0 & 0x1F, *p1]);
    let info_length = usize::from(u16::from_be_bytes([i0 & 0x0F, *i1]));
    let Some(info) = rest.get(..info_length) else {
        return Err(TableError::Overrun);
    };
    let program = descriptors(info)?;
    let mut bytes = &rest[info_length..];
    let mut streams = Vec::new();
    while let [stream_type, q0, q1, l0, l1, after @ ..] = bytes {
        let pid = u16::from_be_bytes([q0 & 0x1F, *q1]);
        let length = usize::from(u16::from_be_bytes([l0 & 0x0F, *l1]));
        let Some(info) = after.get(..length) else {
            return Err(TableError::Overrun);
        };
        let found = descriptors(info)?;
        let registration = found.registration.or(program.registration);
        let hdmv = program.registration == Some(*b"HDMV");
        streams.push(Elementary {
            stream_type: *stream_type,
            pid,
            kind: kind_of(*stream_type, registration, hdmv, &found),
            language: found.language,
            registration: found.registration,
            opus_channels: found.opus_channels,
        });
        bytes = &after[length..];
    }
    Ok(Pmt {
        program: header.id,
        version: header.version,
        pcr_pid: (pcr != crate::packet::NULL_PID).then_some(pcr),
        registration: program.registration,
        streams,
    })
}

/// The framing of `stream_type`, from H.222.0 table 2-34, the ATSC and
/// Blu-ray (`HDMV`) private assignments, and the descriptors that name a
/// codec carried as private data (`0x06`).
/// `hdmv` is a Blu-ray registration on the program, which a stream's own
/// registration (FFmpeg writes `EAC3` on an HDMV E-AC-3 stream) does not
/// undo.
fn kind_of(
    stream_type: u8,
    registration: Option<[u8; 4]>,
    hdmv: bool,
    found: &Descriptors,
) -> Kind {
    match stream_type {
        0x01 | 0x02 => Kind::Mpeg2Video,
        0x03 | 0x04 => Kind::MpegAudio,
        0x0F => Kind::Adts,
        0x10 => Kind::Mpeg4Video,
        0x11 => Kind::Latm,
        0x1B => Kind::H264,
        0x24 => Kind::Hevc,
        0x81 => Kind::Ac3,
        0x87 => Kind::Eac3,
        // Blu-ray and AVCHD: E-AC-3, and E-AC-3 secondary audio.
        0x84 | 0xA1 if hdmv || matches!(registration.as_ref(), Some(b"HDMV" | b"EAC3")) => {
            Kind::Eac3
        }
        0x06 => private_kind(registration, found),
        _ => Kind::Other,
    }
}

fn private_kind(registration: Option<[u8; 4]>, found: &Descriptors) -> Kind {
    if found.eac3 {
        return Kind::Eac3;
    }
    if found.ac3 {
        return Kind::Ac3;
    }
    match registration.as_ref() {
        Some(b"AC-3") => Kind::Ac3,
        Some(b"EAC3") => Kind::Eac3,
        Some(b"Opus") => Kind::Opus,
        Some(b"HEVC") => Kind::Hevc,
        _ => Kind::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::{Kind, PatEntry, TableError, parse_pat, parse_pmt};
    use crate::crc::crc32;
    use crate::section::SectionError;

    pub fn long(table: u8, id: u16, body: &[u8]) -> Vec<u8> {
        let length = body.len() + 9;
        let mut out = vec![
            table,
            0xB0 | (length >> 8) as u8,
            length as u8,
            (id >> 8) as u8,
            id as u8,
            0xC1,
            0,
            0,
        ];
        out.extend_from_slice(body);
        let crc = crc32(&out);
        out.extend_from_slice(&crc.to_be_bytes());
        out
    }

    #[test]
    fn the_pat_lists_programs_and_skips_the_network_pid() {
        let pat = long(0, 7, &[0, 0, 0xE0, 0x10, 0, 1, 0xF0, 0, 0, 2, 0xE1, 0]);
        let parsed = parse_pat(&pat).expect("pat");
        assert_eq!(parsed.transport_stream_id, 7);
        assert_eq!(
            parsed.programs,
            vec![
                PatEntry {
                    program: 1,
                    pmt_pid: 0x1000
                },
                PatEntry {
                    program: 2,
                    pmt_pid: 0x100
                }
            ]
        );
        assert_eq!(parse_pmt(&pat), Err(TableError::WrongTable { table_id: 0 }));
    }

    #[test]
    fn the_pmt_maps_streams_and_descriptors() {
        let mut body = vec![0xE1, 0x00, 0xF0, 0x06, 0x05, 0x04];
        body.extend_from_slice(b"HDMV");
        // H.264 on 0x100.
        body.extend_from_slice(&[0x1B, 0xE1, 0x00, 0xF0, 0x00]);
        // AC-3 by descriptor on 0x101, in English.
        body.extend_from_slice(&[0x06, 0xE1, 0x01, 0xF0, 0x09, 0x6A, 0x01, 0x00]);
        body.extend_from_slice(&[0x0A, 0x04, b'e', b'n', b'g', 0]);
        // Opus on 0x102.
        body.extend_from_slice(&[0x06, 0xE1, 0x02, 0xF0, 0x0A, 0x05, 0x04]);
        body.extend_from_slice(b"Opus");
        body.extend_from_slice(&[0x7F, 0x02, 0x80, 0x02]);
        // HDMV E-AC-3 on 0x103 with its own EAC3 registration, as FFmpeg
        // writes it, and an unknown type on 0x104.
        body.extend_from_slice(&[0x84, 0xE1, 0x03, 0xF0, 0x06, 0x05, 0x04]);
        body.extend_from_slice(b"EAC3");
        body.extend_from_slice(&[0x90, 0xE1, 0x04, 0xF0, 0x00]);
        let pmt = parse_pmt(&long(2, 1, &body)).expect("pmt");
        assert_eq!(pmt.program, 1);
        assert_eq!(pmt.pcr_pid, Some(0x100));
        assert_eq!(pmt.registration, Some(*b"HDMV"));
        let kinds: Vec<(u16, Kind)> = pmt.streams.iter().map(|s| (s.pid, s.kind)).collect();
        assert_eq!(
            kinds,
            vec![
                (0x100, Kind::H264),
                (0x101, Kind::Ac3),
                (0x102, Kind::Opus),
                (0x103, Kind::Eac3),
                (0x104, Kind::Other)
            ]
        );
        assert_eq!(pmt.streams[1].language, Some(*b"eng"));
        assert_eq!(pmt.streams[2].opus_channels, Some(2));
    }

    #[test]
    fn a_length_past_the_section_or_a_bad_crc_is_refused() {
        let body = [0xE1, 0x00, 0xF0, 0x40, 0x05, 0x04];
        assert_eq!(parse_pmt(&long(2, 1, &body)), Err(TableError::Overrun));
        let stream = [0xE1, 0x00, 0xF0, 0x00, 0x1B, 0xE1, 0x00, 0xF0, 0x09, 0x0A];
        assert_eq!(parse_pmt(&long(2, 1, &stream)), Err(TableError::Overrun));
        let mut bad = long(2, 1, &[0xE1, 0x00, 0xF0, 0x00]);
        bad[4] ^= 0x10;
        assert_eq!(parse_pmt(&bad), Err(TableError::Section(SectionError::Crc)));
        let truncated_descriptor = [0xE1, 0x00, 0xF0, 0x03, 0x0A, 0x09, b'e'];
        assert_eq!(
            parse_pmt(&long(2, 1, &truncated_descriptor)),
            Err(TableError::Overrun)
        );
    }

    #[test]
    fn a_pmt_with_no_pcr_pid_says_so() {
        let pmt = parse_pmt(&long(2, 3, &[0xFF, 0xFF, 0xF0, 0x00])).expect("pmt");
        assert_eq!(pmt.pcr_pid, None);
        assert!(pmt.streams.is_empty());
    }
}
