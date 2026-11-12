//! Reading one program of a transport stream as packets.
//!
//! **Open.** The first [`MAX_OPEN_SCAN_BYTES`] are read once and kept. The
//! PAT and every PMT it lists are read from them, and one program is
//! chosen: the one asked for, or else the first in PAT order with a stream
//! this crate frames. Then the kept bytes are read again through that
//! program's streams until each has given its parameters (picture size,
//! sample rate) and a few access units, which also give the program's
//! start: the lowest timestamp seen. Reading then starts over from the
//! first packet, from the kept bytes, so a forward input works too.
//!
//! **Duration.** A random-access input has its last megabyte read, then
//! two, four, up to [`MAX_TAIL_SCAN_BYTES`], until every stream that had
//! data shows a timestamp there. The program ends at the latest timestamp
//! plus that unit's length (audio) or the shortest gap between timestamps
//! seen there (video). Every stream gets the same edit: from the start, for
//! the program's duration, so the streams stay in step.
//!
//! **Seek.** There is no index. The seek bisects the file on the
//! timestamps of the asked-for stream's PES headers, at most
//! [`MAX_SEEK_PROBES`] probes of up to [`MAX_PROBE_WINDOW_BYTES`] each, to a
//! byte position whose first timestamp is not past the target.
//!
//! For a picture stream (H.264, HEVC, MPEG-2, MPEG-4 Part 2) it then reads
//! transport headers only, back from there in doubling windows (at most
//! [`MAX_RANDOM_ACCESS_SCAN_BYTES`]), for the last PES packet of the stream
//! that a muxer marked with the random access indicator and that is shown
//! by the target. It frames that one access unit and, when it is a
//! keyframe, walks the stream's PES packets from there to the target,
//! reading each one's header and only as much payload as names its first
//! picture (at most 64 KiB): a later keyframe the muxer left unmarked (an
//! MP4 sync-sample table may list fewer than the bitstream holds) moves
//! the landing to it. The preroll is the PES packets from the landing
//! shown before the target.
//!
//! Otherwise (an audio stream, no mark, or a mark on a picture that is not
//! a keyframe: the indicator is only a hint) it frames the stream forward
//! from the bisection's position and lands on the last keyframe whose
//! presentation time is not past the target, starting further back while
//! no keyframe comes before the target, doubling the distance, at most
//! [`MAX_SEEK_WIDENINGS`] times.
//!
//! Reading resumes at the transport packet that started the landing's PES
//! packet; packets of other streams framed before the landing are held and
//! returned right after it.
//! A failed seek leaves the read position where it was.

use std::collections::VecDeque;
use std::io::{Read, Seek};

use cutlass_codec::{
    CodecId, ContainerId, Packet, StreamCodec, StreamParams, Timescale, Timestamp,
};
use cutlass_container::{
    Demux, Edit, Gap, Input, Landing, Language, Limits, OpenError, ReadError, SeekError,
    SeekLanding, SeekTo, Stream, TrackId, Trim,
};

use crate::es::{Unit, unknown};
use crate::limits::{
    MAX_BUFFERED_BYTES, MAX_HELD_PACKETS, MAX_OPEN_SCAN_BYTES, MAX_PROBE_WINDOW_BYTES,
    MAX_RANDOM_ACCESS_SCAN_BYTES, MAX_SEEK_PROBES, MAX_SEEK_WIDENINGS, MAX_TAIL_SCAN_BYTES,
};
use crate::packet::{self, PAT_PID, PacketSize};
use crate::pes::pes_times;
use crate::pipeline::{Counters, Pipeline};
use crate::psi::{Elementary, Kind, Pat, Pmt, SDT_PID, Service, parse_pat, parse_pmt, parse_sdt};
use crate::section::SectionBuffer;
use crate::sniff;
use crate::source::{Next, Source};
use crate::time::unwrap;

/// The clock of every stream: PTS and DTS ticks.
pub const TIMESCALE: u32 = 90_000;

/// Bytes read on after every stream gave its parameters, for a better
/// start time when B pictures come before the first picture shown.
const START_SETTLE_BYTES: u64 = 1 << 20;

/// Access units per stream after which the start is taken as known.
const START_UNITS: u32 = 16;

/// A bisection stops when its bracket is this small.
const SEEK_CLOSE_BYTES: u64 = 256 * 1024;

/// The first step back when no keyframe came before the target.
const WIDEN_BYTES: u64 = 1 << 20;

/// One program the PAT lists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub number: u16,
    pub pmt_pid: u16,
    /// `None` when its PMT was not found in the opening bytes.
    pub pmt: Option<Pmt>,
    /// The provider and service name a DVB service description table (PID
    /// 0x0011) gives this program, when one is in the bytes the open reads.
    pub service: Option<Service>,
}

/// Collects the first service description table on PID 0x0011.
#[derive(Default)]
struct SdtReader {
    buffer: SectionBuffer,
    sections: Vec<Vec<u8>>,
    services: Option<Vec<Service>>,
}

impl SdtReader {
    fn packet(&mut self, bytes: &[u8; 188]) {
        if self.services.is_some() {
            return;
        }
        let header = packet::header(bytes);
        if header.pid != SDT_PID || header.error || !header.payload {
            return;
        }
        let Ok(parsed) = packet::parse(bytes) else {
            return;
        };
        let payload = &bytes[parsed.payload_at.min(bytes.len())..];
        let sections = &mut self.sections;
        let _ = self.buffer.push(payload, header.start, &mut |section| {
            sections.push(section.to_vec());
        });
        for section in self.sections.drain(..) {
            // Section 0 of a one-section table is the whole of it; a longer
            // table is rare and its first section names the main services.
            if let Ok(services) = parse_sdt(&section) {
                self.services = Some(services);
                break;
            }
        }
    }

    fn name(self, programs: &mut [Program]) {
        for service in self.services.into_iter().flatten() {
            if let Some(program) = programs
                .iter_mut()
                .find(|program| program.number == service.program && program.service.is_none())
            {
                program.service = Some(service);
            }
        }
    }
}

/// What the last seek cost.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SeekCost {
    /// Bytes read from the input.
    pub bytes: u64,
    /// Bisection probes.
    pub probes: u32,
    /// Forward runs to find the keyframe, widening included.
    pub runs: u32,
    /// Access units of the sought stream framed by those runs.
    pub units: u64,
    /// Transport packets whose header alone was read looking for a random
    /// access point.
    pub headers: u64,
}

/// What a packet read skips after a seek: the sought stream's units before
/// the landing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Skip {
    track: usize,
    pts: Option<i64>,
    dts: Option<i64>,
    position: u64,
}

/// What [`TsDemux::walk_to`] found between a landing and the target.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Walk {
    /// PES packets after the landing shown before the target.
    shown: u32,
    /// The last later PES packet whose first picture is a keyframe shown by
    /// the target, and the PES packets after it shown before the target.
    later_key: Option<(u64, u32)>,
    /// Payload bytes read to classify picture heads.
    head_bytes: u64,
}

impl Walk {
    fn mark_key(&mut self, position: u64) {
        self.later_key = Some((position, 0));
    }
}

/// Payload bytes read at the start of a PES packet to find its first
/// picture header before giving up on it.
const MAX_HEAD_BYTES: usize = 64 * 1024;

/// A keyframe a forward run found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Candidate {
    pts: i64,
    dts: Option<i64>,
    position: u64,
    preroll: u32,
    /// Units of the stream before it in the run.
    before: u64,
}

/// One MPEG transport stream, one program read as packets.
pub struct TsDemux<R> {
    source: Source<R>,
    random: bool,
    length: Option<u64>,
    limits: Limits,
    data_start: u64,
    programs: Vec<Program>,
    program: u16,
    elementaries: Vec<Elementary>,
    pipeline: Pipeline,
    streams: Vec<Stream>,
    ready: VecDeque<(usize, Unit)>,
    held: VecDeque<(usize, Unit)>,
    skip: Option<Skip>,
    start: i64,
    end: Option<i64>,
    finished: bool,
    cancelled: bool,
    seek_read: u64,
    last_seek: SeekCost,
    held_dropped: u64,
    shed: u64,
}

impl<R: Read + Seek> TsDemux<R> {
    /// Opens a random-access input of `length` bytes with
    /// [`Limits::standard`], choosing the program by the policy above.
    pub fn open(reader: R, length: u64) -> Result<Self, OpenError> {
        Self::open_limited(reader, length, Limits::standard())
    }

    /// Opens a random-access input with explicit limits.
    pub fn open_limited(reader: R, length: u64, limits: Limits) -> Result<Self, OpenError> {
        let source = Source::new(Input::random(reader, length, limits));
        Self::build(source, limits, Some(length), None)
    }

    /// Opens a random-access input and reads the program numbered
    /// `program`.
    pub fn open_program(reader: R, length: u64, program: u16) -> Result<Self, OpenError> {
        let limits = Limits::standard();
        let source = Source::new(Input::random(reader, length, limits));
        Self::build(source, limits, Some(length), Some(program))
    }

    /// Opens a forward input. The duration is not known: every edit has a
    /// duration of 0. Seek returns [`SeekError::ForwardOnly`].
    pub fn open_forward(reader: R) -> Result<Self, OpenError> {
        let limits = Limits::standard();
        let source = Source::new(Input::forward(reader, limits));
        Self::build(source, limits, None, None)
    }

    fn build(
        mut source: Source<R>,
        limits: Limits,
        length: Option<u64>,
        wanted: Option<u16>,
    ) -> Result<Self, OpenError> {
        source.keep_from_here();
        let mut sdt = SdtReader::default();
        let (data_start, mut programs, leading) = read_tables(&mut source, wanted, &mut sdt)?;
        let Some(data_start) = data_start else {
            return Err(OpenError::Unrecognized {
                family: ContainerId::MpegTs,
            });
        };
        if programs.is_empty() {
            return Err(OpenError::Source {
                message: "no program association table in the opening bytes".to_owned(),
            });
        }
        let pmt = choose(&programs, wanted)?.clone();
        let max_tracks = usize::try_from(limits.max_tracks)
            .unwrap_or(usize::MAX)
            .min(255);
        let mut elementaries: Vec<Elementary> =
            pmt.streams.iter().take(max_tracks).cloned().collect();
        let limit = usize::try_from(limits.max_packet_bytes).unwrap_or(usize::MAX);

        source.rewind(data_start);
        sniff::name_private_streams(&mut source, data_start, &mut elementaries)
            .map_err(OpenError::from_header)?;
        // Read the kept bytes again through the program's streams.
        source.rewind(data_start);
        let mut pipeline = Pipeline::new(&elementaries, limit);
        let start = learn(&mut source, &mut pipeline, data_start, &mut sdt)?;
        sdt.name(&mut programs);

        let random = length.is_some();
        let mut end = None;
        if let Some(length) = length {
            source.release();
            end = tail_end(&mut source, &elementaries, limit, start, data_start, length)
                .map_err(OpenError::from_header)?;
            source
                .seek_packet(data_start)
                .map_err(OpenError::from_header)?;
        } else {
            source.rewind(data_start);
            source.release();
        }
        let duration = end.map_or(0, |end: i64| end.saturating_sub(start).max(0));
        let edit =
            Edit::try_new(Some(start), duration, 1, 1).map_err(|error| OpenError::Source {
                message: error.to_string(),
            })?;
        let streams = pipeline
            .tracks
            .iter()
            .zip(&elementaries)
            .map(|(track, elementary)| {
                let params = track
                    .framer
                    .facts()
                    .map_or_else(|| fallback_params(elementary), |facts| facts.params());
                let params = cap_extradata(params, limits.max_extradata_bytes);
                let language = elementary
                    .language
                    .and_then(|code| std::str::from_utf8(&code).ok().map(str::to_owned))
                    .and_then(|code| Language::new(code).ok());
                Stream::new(
                    TrackId::new(u32::from(track.pid)),
                    params,
                    Timescale::new(TIMESCALE),
                    language,
                    vec![edit],
                    Trim::new(None, None),
                )
            })
            .collect();
        pipeline.reset();
        pipeline.clock.set(None);
        // The passes over the opening bytes and the tail are not reads:
        // only what was skipped before the first packet stays counted.
        pipeline.counters = Counters::default();
        // A PMT PID that is also a stream's stays the stream's.
        if let Some(program) = programs
            .iter()
            .find(|program| program.number == pmt.program)
            && pipeline.track_of(program.pmt_pid).is_none()
        {
            pipeline.watch_pmt(program.pmt_pid, pmt.program, pmt.version);
        }
        (source.resyncs, source.skipped) = leading;
        source.tail = 0;
        Ok(Self {
            source,
            random,
            length,
            limits,
            data_start,
            programs,
            program: pmt.program,
            elementaries,
            pipeline,
            streams,
            ready: VecDeque::new(),
            held: VecDeque::new(),
            skip: None,
            start,
            end,
            finished: false,
            cancelled: false,
            seek_read: 0,
            last_seek: SeekCost::default(),
            held_dropped: 0,
            shed: 0,
        })
    }

    /// Bisects on the timestamps of `pid` toward `target`: a byte position
    /// whose first timestamp is not past it.
    fn bisect(&mut self, pid: u16, target: i64, cost: &mut SeekCost) -> Result<u64, ReadError> {
        let length = self.length.unwrap_or(0);
        let stride = self.source.size().stride() as u64;
        let (mut lo, mut lo_time) = (self.data_start, self.start);
        let (mut hi, mut hi_time) = (length, self.end.unwrap_or(self.start));
        let mut best = self.data_start;
        for _ in 0..MAX_SEEK_PROBES {
            if hi.saturating_sub(lo) < SEEK_CLOSE_BYTES {
                break;
            }
            let middle = lo + (hi - lo) / 2;
            let middle = self.data_start + (middle - self.data_start) / stride * stride;
            let reference = interpolate((lo, lo_time), (hi, hi_time), middle);
            cost.probes += 1;
            if self.cancelled {
                return Err(ReadError::Cancelled);
            }
            match self.probe(middle, pid, reference)? {
                Some((at, time)) if time <= target => {
                    lo = middle;
                    lo_time = time;
                    best = at;
                }
                Some((_, time)) => {
                    hi = middle;
                    hi_time = time;
                }
                None => hi = middle,
            }
        }
        Ok(best)
    }

    /// The first PES header of `pid` from `from` on, within
    /// [`MAX_PROBE_WINDOW_BYTES`]: its packet's position and its DTS, or
    /// PTS, unwrapped near `reference`.
    fn probe(
        &mut self,
        from: u64,
        pid: u16,
        reference: i64,
    ) -> Result<Option<(u64, i64)>, ReadError> {
        self.source.seek(from).map_err(ReadError::from_packet)?;
        loop {
            let Next::Packet(position, index) =
                self.source.next().map_err(ReadError::from_packet)?
            else {
                return Ok(None);
            };
            if position.saturating_sub(from) > MAX_PROBE_WINDOW_BYTES {
                return Ok(None);
            }
            let bytes = self.source.packet(index);
            let header = packet::header(bytes);
            if header.pid != pid || !header.start || header.error {
                continue;
            }
            let Ok(parsed) = packet::parse(bytes) else {
                continue;
            };
            let (pts, dts) = pes_times(&bytes[parsed.payload_at.min(bytes.len())..]);
            if let Some(raw) = dts.or(pts) {
                return Ok(Some((position, unwrap(raw, reference))));
            }
        }
    }

    /// Reads transport headers only, back from `best` in doubling windows,
    /// for the last packet of `pid` that starts a PES packet, carries the
    /// random access indicator, and is shown by `target`. Each window ends
    /// where the one before began; the first ends at the first PES header
    /// decoded after the target.
    fn random_access_before(
        &mut self,
        pid: u16,
        target: i64,
        best: u64,
        cost: &mut SeekCost,
    ) -> Result<Option<u64>, ReadError> {
        let stride = self.source.size().stride() as u64;
        let align = |at: u64, start: u64| start + at.saturating_sub(start) / stride * stride;
        let line = (
            (self.data_start, self.start),
            (self.length.unwrap_or(0), self.end.unwrap_or(self.start)),
        );
        let (mut lo, mut hi) = (align(best, self.data_start), u64::MAX);
        let mut step = WIDEN_BYTES;
        let mut scanned = 0_u64;
        loop {
            self.source.seek(lo).map_err(ReadError::from_packet)?;
            let mut found = None;
            loop {
                if self.cancelled {
                    return Err(ReadError::Cancelled);
                }
                let Next::Packet(position, index) =
                    self.source.next().map_err(ReadError::from_packet)?
                else {
                    break;
                };
                if position >= hi {
                    break;
                }
                cost.headers += 1;
                scanned = scanned.saturating_add(stride);
                let bytes = self.source.packet(index);
                let header = packet::header(bytes);
                if header.pid != pid || !header.start || header.error {
                    continue;
                }
                let Ok(parsed) = packet::parse(bytes) else {
                    continue;
                };
                let (pts, dts) = pes_times(&bytes[parsed.payload_at.min(bytes.len())..]);
                let reference = interpolate(line.0, line.1, position);
                let Some(decode) = dts.or(pts).map(|raw| unwrap(raw, reference)) else {
                    continue;
                };
                if decode > target {
                    break;
                }
                let shown = pts.map_or(decode, |raw| unwrap(raw, decode));
                if parsed.adaptation.random_access && shown <= target {
                    found = Some(position);
                }
            }
            if found.is_some() {
                return Ok(found);
            }
            if lo <= self.data_start || scanned >= MAX_RANDOM_ACCESS_SCAN_BYTES {
                return Ok(None);
            }
            hi = lo;
            lo = align(
                lo.saturating_sub(step).max(self.data_start),
                self.data_start,
            );
            step = step.saturating_mul(2);
        }
    }

    /// Walks the PES packets of `track` after the one at `landing`, up to
    /// the first decoded after `target`, reading their headers and only as
    /// much payload as tells whether their first picture is a keyframe.
    fn walk_to(
        &mut self,
        track: usize,
        landing: u64,
        target: i64,
        cost: &mut SeekCost,
    ) -> Result<Walk, ReadError> {
        let pid = self.pipeline.tracks[track].pid;
        let kind = self.pipeline.tracks[track].kind;
        let line = (
            (self.data_start, self.start),
            (self.length.unwrap_or(0), self.end.unwrap_or(self.start)),
        );
        self.source.seek(landing).map_err(ReadError::from_packet)?;
        let mut walk = Walk::default();
        // The PES packet whose start is being read: its position, and the
        // payload so far.
        let mut head: Option<(u64, Vec<u8>)> = None;
        loop {
            if self.cancelled {
                return Err(ReadError::Cancelled);
            }
            let Next::Packet(position, index) =
                self.source.next().map_err(ReadError::from_packet)?
            else {
                return Ok(walk);
            };
            cost.headers += 1;
            let bytes = self.source.packet(index);
            let header = packet::header(bytes);
            if position == landing || header.pid != pid || header.error {
                continue;
            }
            let Ok(parsed) = packet::parse(bytes) else {
                continue;
            };
            let payload = &bytes[parsed.payload_at.min(bytes.len())..];
            if !header.start {
                if let Some((at, data)) = head.as_mut() {
                    data.extend_from_slice(payload);
                    let at = *at;
                    if Self::read_head(kind, &mut head, &mut walk) {
                        walk.mark_key(at);
                    }
                }
                continue;
            }
            head = None;
            let (pts, dts) = pes_times(payload);
            let reference = interpolate(line.0, line.1, position);
            let Some(decode) = dts.or(pts).map(|raw| unwrap(raw, reference)) else {
                continue;
            };
            if decode > target {
                return Ok(walk);
            }
            let shown = pts.map_or(decode, |raw| unwrap(raw, decode));
            if shown > target {
                continue;
            }
            if shown < target {
                walk.shown = walk.shown.saturating_add(1);
                if let Some((_, after)) = walk.later_key.as_mut() {
                    *after = after.saturating_add(1);
                }
            }
            let body = crate::pes::pes_body(payload).unwrap_or_default();
            head = Some((position, body.to_vec()));
            if Self::read_head(kind, &mut head, &mut walk) {
                walk.mark_key(position);
            }
        }
    }

    /// Classifies the PES start in `head`. True when it is a keyframe; the
    /// head is dropped once decided or too long to be worth reading on.
    fn read_head(kind: Kind, head: &mut Option<(u64, Vec<u8>)>, walk: &mut Walk) -> bool {
        let Some((_, data)) = head.as_ref() else {
            return false;
        };
        walk.head_bytes = walk.head_bytes.saturating_add(data.len() as u64);
        match crate::es::head_keyframe(kind, data) {
            Some(key) => {
                *head = None;
                key
            }
            None => {
                if data.len() > MAX_HEAD_BYTES {
                    *head = None;
                }
                false
            }
        }
    }

    /// Frames the stream of `track` from `from` up to `target`. Returns the
    /// last keyframe not shown after the target, and the first keyframe.
    /// With `first_only` the run stops at the stream's first unit.
    fn run(
        &mut self,
        from: u64,
        track: usize,
        target: i64,
        reference: i64,
        first_only: bool,
        cost: &mut SeekCost,
    ) -> Result<(Option<Candidate>, Option<Candidate>), ReadError> {
        self.source.seek(from).map_err(ReadError::from_packet)?;
        let limit = usize::try_from(self.limits.max_packet_bytes).unwrap_or(usize::MAX);
        let mut pipeline = Pipeline::new(&self.elementaries, limit);
        pipeline.clock.set(Some(reference));
        let mut out = VecDeque::new();
        let mut landing: Option<Candidate> = None;
        let mut first: Option<Candidate> = None;
        let mut seen = 0_u64;
        let mut ended = false;
        while !ended {
            if self.cancelled {
                return Err(ReadError::Cancelled);
            }
            match self.source.next().map_err(ReadError::from_packet)? {
                Next::End => {
                    pipeline.flush(&mut out);
                    ended = true;
                }
                Next::Packet(position, index) => {
                    pipeline.packet(position, self.source.packet(index), &mut out);
                    if pipeline.buffered() > MAX_BUFFERED_BYTES {
                        pipeline.shed();
                    }
                }
            }
            let mut stop = false;
            for (index, unit) in out.drain(..) {
                if index != track {
                    continue;
                }
                cost.units += 1;
                let Some(pts) = unit.pts else {
                    seen += 1;
                    continue;
                };
                let candidate = Candidate {
                    pts,
                    dts: unit.dts,
                    position: unit.position,
                    preroll: u32::from(pts < target),
                    before: seen,
                };
                seen += 1;
                if unit.keyframe && first.is_none() {
                    first = Some(Candidate {
                        preroll: 0,
                        ..candidate
                    });
                }
                // Decode times only rise: nothing after this is shown
                // before the target. The rest of the batch is only looked
                // at for the first keyframe.
                if stop || unit.dts.unwrap_or(pts) > target {
                    stop = true;
                    continue;
                }
                if unit.keyframe && pts <= target {
                    landing = Some(candidate);
                } else if let Some(landing) = landing.as_mut()
                    && pts < target
                {
                    landing.preroll = landing.preroll.saturating_add(1);
                }
                stop |= first_only;
            }
            ended |= stop;
        }
        // Keep looking for the first keyframe when the run began at the
        // start and none came before the target.
        if first.is_none() && from == self.data_start && !first_only {
            first = self.first_keyframe(&mut pipeline, track, cost)?;
        }
        Ok((landing, first))
    }

    /// Reads on until `track` gives a keyframe with a timestamp.
    fn first_keyframe(
        &mut self,
        pipeline: &mut Pipeline,
        track: usize,
        cost: &mut SeekCost,
    ) -> Result<Option<Candidate>, ReadError> {
        let mut out = VecDeque::new();
        let mut seen = 0_u64;
        loop {
            if self.cancelled {
                return Err(ReadError::Cancelled);
            }
            let ended = match self.source.next().map_err(ReadError::from_packet)? {
                Next::End => {
                    pipeline.flush(&mut out);
                    true
                }
                Next::Packet(position, index) => {
                    pipeline.packet(position, self.source.packet(index), &mut out);
                    false
                }
            };
            for (index, unit) in out.drain(..) {
                if index != track {
                    continue;
                }
                cost.units += 1;
                if unit.keyframe
                    && let Some(pts) = unit.pts
                {
                    return Ok(Some(Candidate {
                        pts,
                        dts: unit.dts,
                        position: unit.position,
                        preroll: 0,
                        before: seen,
                    }));
                }
                seen += 1;
            }
            if ended {
                return Ok(None);
            }
        }
    }

    fn seek_inner(&mut self, to: SeekTo, cost: &mut SeekCost) -> Result<SeekLanding, ReadError> {
        let track = self
            .pipeline
            .tracks
            .iter()
            .position(|track| u32::from(track.pid) == to.track)
            .ok_or(ReadError::Seek(SeekError::UnknownTrack))?;
        let pid = self.pipeline.tracks[track].pid;
        if to.end.or(self.end).is_some_and(|end| to.time >= end) {
            let length = self.length.unwrap_or(0);
            self.source.seek(length).map_err(ReadError::from_packet)?;
            self.commit_end();
            return Ok(SeekLanding {
                landing: Landing::End,
                index_used: false,
                payload_scans: 0,
                entries_visited: 0,
            });
        }
        let best = self.bisect(pid, to.time, cost)?;
        let mut chosen = None;
        // Audio PES packets may hold many frames, each a keyframe, so a
        // mark there is not the frame to land on; audio is framed instead.
        let pictures = matches!(
            self.pipeline.tracks[track].kind,
            Kind::H264 | Kind::Hevc | Kind::Mpeg2Video | Kind::Mpeg4Video
        );
        let mark = if pictures {
            self.random_access_before(pid, to.time, best, cost)?
        } else {
            None
        };
        if let Some(mark) = mark {
            cost.runs += 1;
            let reference = interpolate(
                (self.data_start, self.start),
                (self.length.unwrap_or(0), self.end.unwrap_or(self.start)),
                mark,
            );
            // The marked unit alone is framed. The PES packets after it, up
            // to the target, are read from their headers and the start of
            // their payload: a keyframe the muxer did not mark is found
            // there, and the preroll counted.
            let (landing, _) = self.run(mark, track, to.time, reference, true, cost)?;
            if let Some(mut landing) = landing {
                let walk = self.walk_to(track, landing.position, to.time, cost)?;
                landing.preroll = landing.preroll.saturating_add(walk.shown);
                chosen = Some((landing, mark));
                if let Some((later, shown_after)) = walk.later_key {
                    cost.runs += 1;
                    let reference = interpolate(
                        (self.data_start, self.start),
                        (self.length.unwrap_or(0), self.end.unwrap_or(self.start)),
                        later,
                    );
                    if let (Some(mut key), _) =
                        self.run(later, track, to.time, reference, true, cost)?
                    {
                        key.preroll = key.preroll.saturating_add(shown_after);
                        chosen = Some((key, later));
                    }
                }
            }
        }
        let mut from = best;
        let mut step = WIDEN_BYTES;
        for _ in 0..=MAX_SEEK_WIDENINGS {
            if chosen.is_some() {
                break;
            }
            cost.runs += 1;
            let reference = interpolate(
                (self.data_start, self.start),
                (self.length.unwrap_or(0), self.end.unwrap_or(self.start)),
                from,
            );
            let (landing, first) = self.run(from, track, to.time, reference, false, cost)?;
            if let Some(landing) = landing {
                chosen = Some((landing, from));
                break;
            }
            if from == self.data_start {
                chosen = first.map(|first| (first, from));
                break;
            }
            from = from.saturating_sub(step).max(self.data_start);
            step = step.saturating_mul(2);
        }
        let Some((landing, from)) = chosen else {
            return Err(ReadError::Seek(SeekError::Empty));
        };
        self.source
            .seek_packet(landing.position)
            .map_err(ReadError::from_packet)?;
        self.pipeline.reset();
        self.pipeline
            .clock
            .set(Some(landing.dts.unwrap_or(landing.pts)));
        self.ready.clear();
        self.held.clear();
        self.finished = false;
        self.skip = Some(Skip {
            track,
            pts: Some(landing.pts),
            dts: landing.dts,
            position: landing.position,
        });
        let ordinal = if from == self.data_start && landing.before == 0 {
            0
        } else {
            u32::MAX
        };
        Ok(SeekLanding {
            landing: Landing::Packet {
                ordinal,
                time: landing.pts,
                keyframe: true,
                preroll: landing.preroll,
                byte_position: landing.position,
            },
            index_used: false,
            payload_scans: u32::try_from(cost.units + u64::from(cost.probes)).unwrap_or(u32::MAX),
            entries_visited: 0,
        })
    }
}

impl<R: Read> TsDemux<R> {
    /// The programs the PAT lists, in its order.
    #[must_use]
    pub fn programs(&self) -> &[Program] {
        &self.programs
    }

    /// The program being read.
    #[must_use]
    pub const fn program(&self) -> u16 {
        self.program
    }

    /// The stream types and descriptors of the streams, in track order.
    #[must_use]
    pub fn elementary_streams(&self) -> &[Elementary] {
        &self.elementaries
    }

    /// 188-byte TS, 192-byte M2TS, or 204-byte TS with a FEC trailer.
    #[must_use]
    pub const fn packet_size(&self) -> PacketSize {
        self.source.size()
    }

    /// The lowest timestamp of the opening bytes: time zero of every edit.
    #[must_use]
    pub const fn start(&self) -> i64 {
        self.start
    }

    /// The end of the program, when the input's length was known.
    #[must_use]
    pub const fn end(&self) -> Option<i64> {
        self.end
    }

    /// Faults met while reading packets since open.
    #[must_use]
    pub const fn counters(&self) -> Counters {
        self.pipeline.counters
    }

    /// Times the packet grid was lost and found again.
    #[must_use]
    pub const fn resyncs(&self) -> u64 {
        self.source.resyncs
    }

    /// Bytes skipped finding the packet grid, at open and since.
    #[must_use]
    pub const fn skipped_bytes(&self) -> u64 {
        self.source.skipped
    }

    /// Packets of other streams dropped while a seek waited for its landing.
    #[must_use]
    pub const fn held_dropped(&self) -> u64 {
        self.held_dropped
    }

    /// Times buffered partial units passed [`MAX_BUFFERED_BYTES`] and the
    /// largest was dropped.
    #[must_use]
    pub const fn shed(&self) -> u64 {
        self.shed
    }

    /// What the last seek cost.
    #[must_use]
    pub const fn last_seek(&self) -> SeekCost {
        self.last_seek
    }

    /// Bytes read from the input by open, reads and seeks.
    #[must_use]
    pub fn total_read(&self) -> u64 {
        self.source.consumed()
    }

    fn commit_end(&mut self) {
        self.pipeline.reset();
        self.ready.clear();
        self.held.clear();
        self.skip = None;
        self.finished = true;
    }

    fn next_unit(&mut self) -> Result<Option<(usize, Unit)>, ReadError> {
        loop {
            if self.skip.is_none()
                && let Some(held) = self.held.pop_front()
            {
                return Ok(Some(held));
            }
            if let Some((index, unit)) = self.ready.pop_front() {
                let Some(skip) = self.skip else {
                    return Ok(Some((index, unit)));
                };
                if index != skip.track {
                    if self.held.len() >= MAX_HELD_PACKETS {
                        self.held.pop_front();
                        self.held_dropped += 1;
                    }
                    self.held.push_back((index, unit));
                    continue;
                }
                let landed =
                    unit.position == skip.position && unit.pts == skip.pts && unit.dts == skip.dts;
                let passed = match (unit.dts.or(unit.pts), skip.dts.or(skip.pts)) {
                    (Some(time), Some(landing)) => time > landing,
                    _ => false,
                };
                if landed || passed {
                    self.skip = None;
                    return Ok(Some((index, unit)));
                }
                continue;
            }
            if self.finished {
                if self.skip.take().is_some() {
                    continue;
                }
                return Ok(None);
            }
            match self.source.next().map_err(ReadError::from_packet)? {
                Next::End => {
                    self.pipeline.flush(&mut self.ready);
                    self.finished = true;
                }
                Next::Packet(position, index) => {
                    self.pipeline
                        .packet(position, self.source.packet(index), &mut self.ready);
                    if self.pipeline.buffered() > MAX_BUFFERED_BYTES {
                        self.pipeline.shed();
                        self.shed += 1;
                    }
                }
            }
        }
    }
}

impl<R: Read + Seek + Send> Demux for TsDemux<R> {
    fn family(&self) -> ContainerId {
        ContainerId::MpegTs
    }

    fn streams(&self) -> &[Stream] {
        &self.streams
    }

    fn gaps(&self) -> &[Gap] {
        &[]
    }

    fn read_packet(&mut self) -> Result<Option<Packet>, ReadError> {
        if self.cancelled {
            return Err(ReadError::Cancelled);
        }
        let Some((index, unit)) = self.next_unit()? else {
            return Ok(None);
        };
        let pid = self.pipeline.tracks[index].pid;
        let timestamp = Timestamp::try_new(unit.dts, unit.pts, unit.duration).map_err(|error| {
            ReadError::Source {
                message: error.to_string(),
            }
        })?;
        Packet::try_new(
            u32::from(pid),
            unit.bytes,
            timestamp,
            Timescale::new(TIMESCALE),
            unit.keyframe,
        )
        .map(Some)
        .map_err(|error| ReadError::Source {
            message: error.to_string(),
        })
    }

    fn seek(&mut self, to: SeekTo) -> Result<SeekLanding, ReadError> {
        if self.cancelled {
            return Err(ReadError::Cancelled);
        }
        if !self.random {
            return Err(ReadError::Seek(SeekError::ForwardOnly));
        }
        let before = self.source.consumed();
        let saved = self.source.save();
        let mut cost = SeekCost::default();
        let result = self.seek_inner(to, &mut cost);
        if result.is_err() {
            // The pipeline was not touched; only the input moved.
            self.source.restore(saved).map_err(ReadError::from_packet)?;
        }
        cost.bytes = self.source.consumed().saturating_sub(before);
        self.seek_read = self.seek_read.saturating_add(cost.bytes);
        self.last_seek = cost;
        result
    }

    fn cancel(&mut self) {
        self.cancelled = true;
    }

    /// Bytes read by open and by packet reads. Bytes a seek reads are on
    /// [`TsDemux::last_seek`] and [`TsDemux::total_read`] instead.
    fn consumed(&self) -> u64 {
        self.source.consumed().saturating_sub(self.seek_read)
    }
}

/// The position on a straight line between two (position, time) points.
fn interpolate(lo: (u64, i64), hi: (u64, i64), at: u64) -> i64 {
    let span = i128::from(hi.0.saturating_sub(lo.0));
    if span == 0 {
        return lo.1;
    }
    let along = i128::from(at.saturating_sub(lo.0)).min(span);
    let time = i128::from(lo.1) + (i128::from(hi.1) - i128::from(lo.1)) * along / span;
    i64::try_from(time).unwrap_or(lo.1)
}

/// The first packet's position, the programs, and the resyncs and bytes
/// skipped before the first packet.
type Tables = (Option<u64>, Vec<Program>, (u64, u64));

/// Reads the PAT and the PMTs from the opening bytes. Returns the first
/// packet's position, the programs, and the resyncs and bytes skipped
/// before the first packet.
fn read_tables<R: Read>(
    source: &mut Source<R>,
    wanted: Option<u16>,
    sdt: &mut SdtReader,
) -> Result<Tables, OpenError> {
    let mut leading = (0, 0);
    let mut first = None;
    let mut pat: Option<Pat> = None;
    let mut pat_buffer = SectionBuffer::default();
    let mut pmt_buffers: Vec<(u16, SectionBuffer)> = Vec::new();
    let mut pmts: Vec<Pmt> = Vec::new();
    let mut sections: Vec<Vec<u8>> = Vec::new();
    while let Next::Packet(position, index) = source.next().map_err(OpenError::from_header)? {
        if first.is_none() {
            leading = (source.resyncs, source.skipped);
        }
        let first = *first.get_or_insert(position);
        if position.saturating_sub(first) > MAX_OPEN_SCAN_BYTES {
            break;
        }
        let bytes = source.packet(index);
        sdt.packet(bytes);
        let header = packet::header(bytes);
        if header.error || !header.payload {
            continue;
        }
        let Ok(parsed) = packet::parse(bytes) else {
            continue;
        };
        let payload = &bytes[parsed.payload_at.min(bytes.len())..];
        if header.pid == PAT_PID {
            let _ = pat_buffer.push(payload, header.start, &mut |section| {
                sections.push(section.to_vec());
            });
            for section in sections.drain(..) {
                if let Ok(table) = parse_pat(&section)
                    && pat.as_ref().is_none_or(|old| old.programs.is_empty())
                {
                    pmt_buffers = table
                        .programs
                        .iter()
                        .map(|entry| (entry.pmt_pid, SectionBuffer::default()))
                        .collect();
                    pmt_buffers.dedup_by_key(|(pid, _)| *pid);
                    pat = Some(table);
                }
            }
        } else if let Some((_, buffer)) = pmt_buffers.iter_mut().find(|(pid, _)| *pid == header.pid)
        {
            let _ = buffer.push(payload, header.start, &mut |section| {
                sections.push(section.to_vec());
            });
            for section in sections.drain(..) {
                if let Ok(pmt) = parse_pmt(&section)
                    && !pmts.iter().any(|old| old.program == pmt.program)
                    && pat.as_ref().is_some_and(|pat| {
                        pat.programs.iter().any(|entry| {
                            entry.program == pmt.program && entry.pmt_pid == header.pid
                        })
                    })
                {
                    pmts.push(pmt);
                }
            }
        }
        if let Some(pat) = &pat {
            let complete = match wanted {
                Some(number) => pmts.iter().any(|pmt| pmt.program == number),
                None => pat
                    .programs
                    .iter()
                    .all(|entry| pmts.iter().any(|pmt| pmt.program == entry.program)),
            };
            if complete {
                break;
            }
        }
    }
    let programs = pat.map_or_else(Vec::new, |pat| {
        pat.programs
            .iter()
            .map(|entry| Program {
                number: entry.program,
                pmt_pid: entry.pmt_pid,
                pmt: pmts
                    .iter()
                    .find(|pmt| pmt.program == entry.program)
                    .cloned(),
                service: None,
            })
            .collect()
    });
    Ok((first, programs, leading))
}

/// The program asked for, or the first with a stream this crate frames,
/// or the first with any stream.
fn choose(programs: &[Program], wanted: Option<u16>) -> Result<&Pmt, OpenError> {
    let found = match wanted {
        Some(number) => programs
            .iter()
            .find(|program| program.number == number)
            .and_then(|program| program.pmt.as_ref()),
        None => programs
            .iter()
            .filter_map(|program| program.pmt.as_ref())
            .find(|pmt| pmt.streams.iter().any(|stream| stream.kind != Kind::Other))
            .or_else(|| {
                programs
                    .iter()
                    .filter_map(|program| program.pmt.as_ref())
                    .find(|pmt| !pmt.streams.is_empty())
            }),
    };
    found.ok_or_else(|| OpenError::Source {
        message: match wanted {
            Some(number) => format!("program {number} has no PMT in the opening bytes"),
            None => "no program with streams in the opening bytes".to_owned(),
        },
    })
}

/// Reads the kept bytes through the program's streams until each gave its
/// parameters and a few units. Returns the lowest timestamp seen.
fn learn<R: Read>(
    source: &mut Source<R>,
    pipeline: &mut Pipeline,
    data_start: u64,
    sdt: &mut SdtReader,
) -> Result<i64, OpenError> {
    let mut out = VecDeque::new();
    let mut units = vec![0_u32; pipeline.tracks.len()];
    let mut start: Option<i64> = None;
    let mut settled_at: Option<u64> = None;
    loop {
        let next = source.next().map_err(OpenError::from_header)?;
        let position = match next {
            Next::End => {
                pipeline.flush(&mut out);
                None
            }
            Next::Packet(position, index) => {
                sdt.packet(source.packet(index));
                pipeline.packet(position, source.packet(index), &mut out);
                if pipeline.buffered() > MAX_BUFFERED_BYTES {
                    pipeline.shed();
                }
                Some(position)
            }
        };
        for (index, unit) in out.drain(..) {
            units[index] = units[index].saturating_add(1);
            if let Some(pts) = unit.pts {
                start = Some(start.map_or(pts, |start| start.min(pts)));
            }
        }
        let Some(position) = position else {
            break;
        };
        if position.saturating_sub(data_start) > MAX_OPEN_SCAN_BYTES {
            break;
        }
        let learned = pipeline
            .tracks
            .iter()
            .all(|track| !track.framer.learns_facts() || track.framer.facts().is_some());
        if learned && start.is_some() {
            let settled = *settled_at.get_or_insert(position);
            let enough = units
                .iter()
                .zip(&pipeline.tracks)
                .all(|(count, track)| *count >= START_UNITS || !track.framer.learns_facts());
            if enough || position.saturating_sub(settled) > START_SETTLE_BYTES {
                break;
            }
        }
    }
    Ok(start.unwrap_or(0))
}

/// The program's end from the last bytes of the input, or `None` when they
/// hold no timestamp.
fn tail_end<R: Read + Seek>(
    source: &mut Source<R>,
    elementaries: &[Elementary],
    limit: usize,
    start: i64,
    data_start: u64,
    length: u64,
) -> Result<Option<i64>, cutlass_container::SourceError> {
    let mut window = 1_u64 << 20;
    loop {
        let from = length.saturating_sub(window).max(data_start);
        source.seek(from)?;
        let mut pipeline = Pipeline::new(elementaries, limit);
        pipeline.clock.set(Some(start));
        let tracks = pipeline.tracks.len();
        let mut ends: Vec<Option<i64>> = vec![None; tracks];
        let mut times: Vec<Vec<i64>> = vec![Vec::new(); tracks];
        let mut out = VecDeque::new();
        loop {
            let ended = match source.next()? {
                Next::End => {
                    pipeline.flush(&mut out);
                    true
                }
                Next::Packet(position, index) => {
                    pipeline.packet(position, source.packet(index), &mut out);
                    if pipeline.buffered() > MAX_BUFFERED_BYTES {
                        pipeline.shed();
                    }
                    false
                }
            };
            for (index, unit) in out.drain(..) {
                let Some(pts) = unit.pts else {
                    continue;
                };
                let end = pts.saturating_add(unit.duration.unwrap_or(0));
                ends[index] = Some(ends[index].map_or(end, |old| old.max(end)));
                if unit.duration.is_none() {
                    let list = &mut times[index];
                    if list.len() >= 256 {
                        list.remove(0);
                    }
                    list.push(pts);
                }
            }
            if ended {
                break;
            }
        }
        let seen = ends.iter().filter(|end| end.is_some()).count();
        let whole = from == data_start || window >= MAX_TAIL_SCAN_BYTES;
        if seen == tracks || whole {
            let mut best: Option<i64> = None;
            for (end, list) in ends.iter().zip(&mut times) {
                let Some(mut end) = *end else {
                    continue;
                };
                list.sort_unstable();
                let step = list
                    .windows(2)
                    .map(|pair| pair[1] - pair[0])
                    .filter(|step| *step > 0)
                    .min();
                if let Some(step) = step {
                    end = end.saturating_add(step);
                }
                best = Some(best.map_or(end, |best| best.max(end)));
            }
            if best.is_some() || whole {
                return Ok(best);
            }
        }
        window = window.saturating_mul(2);
    }
}

/// Parameters for a stream whose bitstream gave none in the opening bytes.
fn fallback_params(elementary: &Elementary) -> StreamParams {
    let known = match elementary.kind {
        Kind::H264 => Some(CodecId::H264),
        Kind::Hevc => Some(CodecId::Hevc),
        Kind::Mpeg2Video => Some(CodecId::Mpeg2),
        Kind::Mpeg4Video => Some(CodecId::Mpeg4),
        Kind::Adts | Kind::Latm => Some(CodecId::Aac),
        Kind::Ac3 => Some(CodecId::Ac3),
        Kind::Eac3 => Some(CodecId::Eac3),
        Kind::Opus => Some(CodecId::Opus),
        Kind::MpegAudio | Kind::Other => None,
    };
    let codec = known.map_or_else(
        || unknown(format!("mpegts:0x{:02x}", elementary.stream_type).as_bytes()),
        StreamCodec::Known,
    );
    StreamParams::opaque(codec, Vec::new())
}

/// Drops extradata above the limit rather than publishing it.
fn cap_extradata(params: StreamParams, limit: u64) -> StreamParams {
    if params.extradata().len() as u64 <= limit {
        return params;
    }
    match (params.size(), params.sample_rate(), params.channels()) {
        (Some(size), _, _) => StreamParams::video(params.codec().clone(), size, Vec::new()),
        (None, Some(rate), Some(channels)) => {
            StreamParams::try_audio(params.codec().clone(), rate, channels, Vec::new())
                .unwrap_or_else(|_| StreamParams::opaque(params.codec().clone(), Vec::new()))
        }
        _ => StreamParams::opaque(params.codec().clone(), Vec::new()),
    }
}
