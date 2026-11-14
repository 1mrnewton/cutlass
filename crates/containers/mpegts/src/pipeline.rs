//! From transport packets to access units, for the elementary streams of
//! one program.
//!
//! Each PID has its continuity counter checked (clause 2.4.3.3): a lost
//! packet drops the PES packet in progress, a duplicate is skipped, and a
//! `discontinuity_indicator` excuses a jump. A packet with
//! `transport_error_indicator` set is dropped like a lost one. Scrambled
//! payloads are skipped and counted.

use std::collections::VecDeque;

use crate::es::{Framer, Pes, Unit};
use crate::limits::TS_PACKET_BYTES;
use crate::packet::{self, NULL_PID};
use crate::pes::{Assembler, PesFault, RawPes};
use crate::psi::{Elementary, Kind, parse_pmt};
use crate::section::SectionBuffer;
use crate::time::{Unwrapper, unwrap};

/// What went wrong while reading, counted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    /// Continuity counter jumps: packets lost on a PID.
    pub cc_errors: u64,
    /// Repeated packets skipped.
    pub duplicates: u64,
    /// Packets with `transport_error_indicator` set, dropped.
    pub transport_errors: u64,
    /// Packets whose adaptation field runs past the packet.
    pub bad_packets: u64,
    /// Scrambled payloads skipped.
    pub scrambled: u64,
    /// PES packets dropped: a lost packet, a bad header, or past the cap.
    pub dropped_pes: u64,
    /// Access units dropped for their size, or frames over the per-PES cap.
    pub dropped_units: u64,
    /// PES packets that ended before their declared length.
    pub short_pes: u64,
    /// PMT sections of the open program with a new `version_number`. The
    /// streams stay the ones the opening PMT lists: a PID the new version
    /// adds is not read, and one it drops reads nothing more.
    pub pmt_changes: u64,
}

/// The open program's PMT PID, watched for a new version.
struct PmtWatch {
    pid: u16,
    program: u16,
    version: u8,
    buffer: SectionBuffer,
}

pub struct Track {
    pub pid: u16,
    pub kind: Kind,
    pub stream_type: u8,
    assembler: Assembler,
    pub framer: Framer,
    cc: Option<u8>,
}

pub struct Pipeline {
    pub tracks: Vec<Track>,
    /// Track index per PID, `u8::MAX` for none.
    slots: Vec<u8>,
    pub clock: Unwrapper,
    pub counters: Counters,
    limit: usize,
    done: Vec<Result<RawPes, PesFault>>,
    units: Vec<Unit>,
    pmt: Option<PmtWatch>,
}

impl Pipeline {
    /// At most 255 tracks; the caller caps them lower.
    pub fn new(streams: &[Elementary], limit: usize) -> Self {
        let mut slots = vec![u8::MAX; 8192];
        let mut tracks = Vec::new();
        for elementary in streams.iter().take(usize::from(u8::MAX)) {
            let pid = usize::from(elementary.pid & 0x1FFF);
            if slots[pid] != u8::MAX || elementary.pid == NULL_PID {
                continue;
            }
            slots[pid] = tracks.len() as u8;
            tracks.push(Track {
                pid: elementary.pid,
                kind: elementary.kind,
                stream_type: elementary.stream_type,
                assembler: Assembler::default(),
                framer: Framer::new(elementary.kind, elementary.opus_channels, limit),
                cc: None,
            });
        }
        Self {
            tracks,
            slots,
            clock: Unwrapper::default(),
            counters: Counters::default(),
            limit,
            done: Vec::new(),
            units: Vec::new(),
            pmt: None,
        }
    }

    /// Counts the PMT versions after `version` that `pid` carries for
    /// `program`.
    pub fn watch_pmt(&mut self, pid: u16, program: u16, version: u8) {
        self.pmt = Some(PmtWatch {
            pid,
            program,
            version,
            buffer: SectionBuffer::default(),
        });
    }

    fn pmt_packet(&mut self, bytes: &[u8; TS_PACKET_BYTES]) {
        let Some(PmtWatch {
            program,
            version,
            buffer,
            ..
        }) = &mut self.pmt
        else {
            return;
        };
        let header = packet::header(bytes);
        if header.error || !header.payload {
            return;
        }
        let Ok(parsed) = packet::parse(bytes) else {
            return;
        };
        let payload = &bytes[parsed.payload_at.min(bytes.len())..];
        let counters = &mut self.counters;
        let _ = buffer.push(payload, header.start, &mut |section| {
            if let Ok(pmt) = parse_pmt(section)
                && pmt.program == *program
                && pmt.version != *version
            {
                *version = pmt.version;
                counters.pmt_changes += 1;
            }
        });
    }

    /// The track index of `pid`.
    #[must_use]
    pub fn track_of(&self, pid: u16) -> Option<usize> {
        let slot = *self.slots.get(usize::from(pid))?;
        (slot != u8::MAX).then_some(usize::from(slot))
    }

    /// Forgets every partial unit and continuity count, keeping what the
    /// streams taught the framers.
    pub fn reset(&mut self) {
        for track in &mut self.tracks {
            track.assembler.reset();
            track.framer.reset();
            track.cc = None;
        }
        if let Some(watch) = &mut self.pmt {
            watch.buffer.reset();
        }
    }

    /// Bytes held in partial PES packets and access units.
    #[must_use]
    pub fn buffered(&self) -> usize {
        self.tracks
            .iter()
            .map(|track| track.assembler.buffered() + track.framer.buffered())
            .sum()
    }

    /// Drops the partial units of the track holding the most bytes.
    pub fn shed(&mut self) {
        if let Some(track) = self
            .tracks
            .iter_mut()
            .max_by_key(|track| track.assembler.buffered() + track.framer.buffered())
        {
            track.assembler.reset();
            track.framer.reset();
            self.counters.dropped_pes += 1;
        }
    }

    /// Reads one transport packet. Finished access units go to `out` with
    /// their track index.
    pub fn packet(
        &mut self,
        position: u64,
        bytes: &[u8; TS_PACKET_BYTES],
        out: &mut VecDeque<(usize, Unit)>,
    ) {
        let header = packet::header(bytes);
        if self
            .pmt
            .as_ref()
            .is_some_and(|watch| watch.pid == header.pid)
        {
            self.pmt_packet(bytes);
            return;
        }
        let Some(index) = self.track_of(header.pid) else {
            return;
        };
        let parsed = match packet::parse(bytes) {
            Ok(parsed) => parsed,
            Err(_) => {
                self.counters.bad_packets += 1;
                self.lose(index, out);
                return;
            }
        };
        if header.error {
            self.counters.transport_errors += 1;
            self.lose(index, out);
            return;
        }
        let discontinuity = parsed.adaptation.discontinuity;
        let track = &mut self.tracks[index];
        if let Some(last) = track.cc
            && !discontinuity
        {
            if header.payload {
                if header.continuity == last {
                    self.counters.duplicates += 1;
                    return;
                }
                if header.continuity != (last + 1) & 0x0F {
                    self.counters.cc_errors += 1;
                    track.cc = Some(header.continuity);
                    self.lose(index, out);
                }
            } else if header.continuity != last {
                self.counters.cc_errors += 1;
            }
        }
        let track = &mut self.tracks[index];
        track.cc = Some(header.continuity);
        if !header.payload || parsed.payload_at >= TS_PACKET_BYTES {
            return;
        }
        if header.scrambling != 0 {
            self.counters.scrambled += 1;
            return;
        }
        let data = &bytes[parsed.payload_at..];
        let done = &mut self.done;
        track
            .assembler
            .push(data, header.start, position, self.limit, &mut |pes| {
                done.push(pes);
            });
        self.frame(index, out);
    }

    /// The input ended: every partial PES packet and access unit is
    /// finished as it is.
    pub fn flush(&mut self, out: &mut VecDeque<(usize, Unit)>) {
        for index in 0..self.tracks.len() {
            let done = &mut self.done;
            self.tracks[index]
                .assembler
                .flush(&mut |pes| done.push(pes));
            self.frame(index, out);
            self.tracks[index].framer.flush(&mut self.units);
            out.extend(self.units.drain(..).map(|unit| (index, unit)));
        }
    }

    /// A packet of `index` was lost: the PES packet in progress goes, and
    /// the access units already whole come out.
    fn lose(&mut self, index: usize, out: &mut VecDeque<(usize, Unit)>) {
        let track = &mut self.tracks[index];
        if track.assembler.open() {
            self.counters.dropped_pes += 1;
        }
        track.assembler.reset();
        track.framer.flush(&mut self.units);
        out.extend(self.units.drain(..).map(|unit| (index, unit)));
    }

    /// Frames the PES packets the assembler of `index` finished.
    fn frame(&mut self, index: usize, out: &mut VecDeque<(usize, Unit)>) {
        for result in std::mem::take(&mut self.done) {
            let raw = match result {
                Ok(raw) => raw,
                Err(_) => {
                    self.counters.dropped_pes += 1;
                    continue;
                }
            };
            if raw.short {
                self.counters.short_pes += 1;
            }
            let dts = raw.dts.map(|dts| self.clock.place(dts));
            let pts = raw.pts.map(|pts| match dts {
                Some(dts) => unwrap(pts, dts),
                None => self.clock.place(pts),
            });
            let pes = Pes {
                payload: raw.payload,
                pts,
                dts,
                position: raw.position,
            };
            if self.tracks[index]
                .framer
                .push(pes, &mut self.units)
                .is_err()
            {
                self.counters.dropped_units += 1;
            }
            out.extend(self.units.drain(..).map(|unit| (index, unit)));
        }
    }
}
