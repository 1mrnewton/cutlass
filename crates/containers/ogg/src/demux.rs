//! The Ogg demuxer.
//!
//! Open reads the beginning-of-stream pages, then each stream's other
//! header packets (`OpusTags`, or the Vorbis comment and setup headers),
//! then the first audio page of each stream (to learn where its timeline
//! starts). On a random-access input it also reads the end of the first
//! link to learn each stream's final granule position and the end trimming
//! on its last page. Reads then return one Opus or Vorbis packet per call
//! in file order. Seek bisects on page granule positions.

use std::collections::VecDeque;
use std::io::{Read, Seek};

use cutlass_codec::{
    CodecId, ContainerId, Packet, StreamCodec, StreamParams, Timescale, Timestamp, UnknownCodec,
};
use cutlass_container::{
    Demux, Edit, Gap, Input, Landing, Limits, OpenError, ReadError, SeekError, SeekLanding, SeekTo,
    Stream, TrackId, Trim,
};

use crate::error::{Fault, OggError};
use crate::limits::{
    MAX_LINK_PROBES, MAX_OPUS_PACKET_PER_STREAM, MAX_SEEK_PROBES, MAX_START_SCAN_BYTES,
    MAX_TAIL_SCAN_BYTES, OPUS_PREROLL, OPUS_RATE,
};
use crate::logical::{Counters, Logical, Ready, State, Timing};
use crate::opus::{HEAD_MAGIC, OpusHead, OpusTags, mapping_tag};
use crate::page::Page;
use crate::source::{PageRead, Source};
use crate::vorbis::{ID_MAGIC, VorbisComment, VorbisId, VorbisModes, xiph_lace};

/// Bytes read per step when scanning backwards from the end of the file.
const TAIL_CHUNK: u64 = 64 * 1024;

/// A bisection stops and scans linearly once its byte window is this small.
const BISECT_WINDOW: u64 = 64 * 1024;

/// A logical stream of the first link.
#[derive(Clone, Debug)]
enum Kind {
    Logical(Logical),
    /// A mapping this crate does not read. Its pages are skipped.
    Unknown {
        serial: u32,
    },
}

impl Kind {
    const fn serial(&self) -> u32 {
        match self {
            Self::Logical(logical) => logical.serial,
            Self::Unknown { serial } => *serial,
        }
    }
}

/// What open learned about one Opus stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpusInfo {
    pub serial: u32,
    pub head: OpusHead,
    pub tags: OpusTags,
    /// Granule-domain time of the first sample of the first packet, when
    /// the stream has an audio page.
    pub start: Option<i64>,
    /// Granule position of the stream's last page, when it was found.
    pub end: Option<i64>,
    /// Samples decoded past `end`: the end trimming on the last page.
    pub trailing: Option<i64>,
}

/// What open learned about one Vorbis stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VorbisInfo {
    pub serial: u32,
    pub id: VorbisId,
    pub comment: VorbisComment,
    /// Long-block flag of each mode in the setup header.
    pub modes: Vec<bool>,
    /// Granule-domain time of the first sample of the first packet, when
    /// the stream has an audio page. Below zero when the first page
    /// completes more samples than its granule.
    pub start: Option<i64>,
    /// Granule position of the stream's last page, when it was found.
    pub end: Option<i64>,
    /// Samples decoded past `end`: the end trimming on the last page.
    pub trailing: Option<i64>,
}

/// What open learned about one stream of a mapping this crate reads.
#[derive(Clone, Debug)]
enum Info {
    Opus(OpusInfo),
    Vorbis(VorbisInfo),
}

impl Info {
    const fn serial(&self) -> u32 {
        match self {
            Self::Opus(info) => info.serial,
            Self::Vorbis(info) => info.serial,
        }
    }

    const fn start(&self) -> Option<i64> {
        match self {
            Self::Opus(info) => info.start,
            Self::Vorbis(info) => info.start,
        }
    }

    const fn end(&self) -> Option<i64> {
        match self {
            Self::Opus(info) => info.end,
            Self::Vorbis(info) => info.end,
        }
    }

    /// Samples a decoder needs before a target (RFC 7845 section 4.6; for
    /// Vorbis, the packet before the one that holds the target, at most
    /// half a long block).
    fn preroll(&self) -> i64 {
        match self {
            Self::Opus(_) => OPUS_PREROLL,
            Self::Vorbis(info) => i64::from(info.id.blocksizes[1] / 2),
        }
    }
}

/// A header packet that begins a stream, by mapping.
enum Head {
    Opus(OpusHead),
    Vorbis(VorbisId),
    Unknown,
}

/// Header packets after the first, by mapping.
enum Headers {
    Opus(OpusTags),
    Vorbis {
        comment: VorbisComment,
        /// The comment packet for the extradata: as stored, or written again
        /// from the comments kept when it was cut.
        comment_bytes: Vec<u8>,
        setup: Vec<u8>,
    },
}

/// Bytes and probes the last seek used.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SeekCost {
    /// Bytes read from the input, including the read-ahead for the landing.
    pub bytes: u64,
    /// Bisection probes.
    pub probes: u32,
    /// Pages looked at while searching.
    pub pages: u32,
}

/// One Ogg file's first link, read as packets.
pub struct OggDemux<R> {
    source: Source<R>,
    random: bool,
    streams: Vec<Stream>,
    kinds: Vec<Kind>,
    infos: Vec<Option<Info>>,
    gaps: Vec<Gap>,
    ready: VecDeque<Ready>,
    replay: VecDeque<Page>,
    audio_start: u64,
    link_end: Option<u64>,
    chained: bool,
    finished: bool,
    cancelled: bool,
    last_error: Option<OggError>,
    counters: Counters,
    ignored_pages: u64,
    /// Checksum failures and skipped bytes met by packet reads.
    crc_failures: u64,
    skipped: u64,
    seek_read: u64,
    last_seek: SeekCost,
}

/// Reading state a failed seek puts back.
struct Snapshot {
    position: u64,
    states: Vec<Option<State>>,
    ready: VecDeque<Ready>,
    finished: bool,
    gaps: usize,
    counters: Counters,
    ignored_pages: u64,
    crc_failures: u64,
    skipped: u64,
}

impl<R: Read + Seek> OggDemux<R> {
    /// Opens a random-access input of `length` bytes with [`Limits::standard`].
    pub fn open(reader: R, length: u64) -> Result<Self, OpenError> {
        Self::open_limited(reader, length, Limits::standard())
    }

    /// Opens a random-access input with explicit limits.
    pub fn open_limited(reader: R, length: u64, limits: Limits) -> Result<Self, OpenError> {
        let source = Source::new(Input::random(reader, length, limits));
        Self::build(source, limits, true).map_err(open_error)
    }

    /// Opens a forward input. The end of the stream is not known until it
    /// is read: the edit duration is 0 and the trailing trim is absent.
    /// Seek returns [`SeekError::ForwardOnly`].
    pub fn open_forward(reader: R) -> Result<Self, OpenError> {
        let limits = Limits::standard();
        let source = Source::new(Input::forward(reader, limits));
        Self::build(source, limits, false).map_err(open_error)
    }

    fn build(mut source: Source<R>, limits: Limits, random: bool) -> Result<Self, Fault> {
        let first = match source.read_page()? {
            PageRead::Page(page) => page,
            PageRead::End => return Err(OggError::NoPage.into()),
            PageRead::Truncated { expected, got } => {
                return Err(OggError::TruncatedPage { expected, got }.into());
            }
        };
        if !first.bos() {
            return Err(OggError::NotBos.into());
        }
        let leading = source.skipped();

        // Beginning-of-stream pages come first, one per logical stream.
        let mut kinds = Vec::new();
        let mut heads: Vec<Head> = Vec::new();
        let mut extradata = Vec::new();
        let mut bos_sequences = Vec::new();
        let mut current = Some(first);
        while let Some(page) = current.take() {
            if !page.bos() {
                current = Some(page);
                break;
            }
            if kinds.iter().any(|kind: &Kind| kind.serial() == page.serial) {
                return Err(OggError::DuplicateSerial {
                    serial: page.serial,
                }
                .into());
            }
            let count = u32::try_from(kinds.len()).unwrap_or(u32::MAX);
            if count >= limits.max_tracks {
                return Err(OggError::TooManyStreams {
                    limit: limits.max_tracks,
                }
                .into());
            }
            bos_sequences.push(page.sequence);
            let pieces = page.pieces();
            let first_piece = pieces
                .first()
                .map_or(&[][..], |piece| &page.body[piece.start..piece.end]);
            // "Opus" begins both Opus header magics, so a stream that starts
            // with it and has no valid `OpusHead` is a broken Opus stream,
            // not another mapping.
            if first_piece.starts_with(&HEAD_MAGIC[..4]) {
                if pieces.len() != 1 || !pieces[0].complete {
                    return Err(OggError::BadBosPage {
                        serial: page.serial,
                    }
                    .into());
                }
                let size = u64::try_from(first_piece.len()).unwrap_or(u64::MAX);
                if size > limits.max_extradata_bytes {
                    return Err(OggError::TooLarge {
                        declared: size,
                        limit: limits.max_extradata_bytes,
                    }
                    .into());
                }
                let head = OpusHead::parse(first_piece, page.serial)?;
                let cap = MAX_OPUS_PACKET_PER_STREAM
                    .saturating_mul(u64::from(head.streams))
                    .min(limits.max_packet_bytes);
                kinds.push(Kind::Logical(Logical {
                    serial: page.serial,
                    track: count,
                    timing: Timing::Opus {
                        pre_skip: i64::from(head.pre_skip),
                    },
                    first_audio_sequence: u32::MAX,
                    packet_cap: cap,
                    state: State::fresh(),
                }));
                heads.push(Head::Opus(head));
                extradata.push(first_piece.to_vec());
            } else if first_piece.starts_with(ID_MAGIC) {
                // Vorbis I section A.2: the identification header is alone
                // on the stream's first page.
                if pieces.len() != 1 || !pieces[0].complete {
                    return Err(OggError::BadBosPage {
                        serial: page.serial,
                    }
                    .into());
                }
                let id = VorbisId::parse(first_piece, page.serial)?;
                kinds.push(Kind::Logical(Logical {
                    serial: page.serial,
                    track: count,
                    // The modes come with the setup header.
                    timing: Timing::Vorbis(VorbisModes {
                        long: Vec::new(),
                        bits: 0,
                        blocksizes: id.blocksizes,
                    }),
                    first_audio_sequence: u32::MAX,
                    packet_cap: limits.max_packet_bytes,
                    state: State::fresh(),
                }));
                heads.push(Head::Vorbis(id));
                extradata.push(first_piece.to_vec());
            } else {
                let keep = usize::try_from(limits.max_extradata_bytes).unwrap_or(usize::MAX);
                kinds.push(Kind::Unknown {
                    serial: page.serial,
                });
                heads.push(Head::Unknown);
                extradata.push(first_piece[..first_piece.len().min(keep)].to_vec());
            }
            current = match source.read_page()? {
                PageRead::Page(page) => Some(page),
                PageRead::End => None,
                PageRead::Truncated { expected, got } => {
                    return Err(OggError::TruncatedPage { expected, got }.into());
                }
            };
        }

        let mut replay = VecDeque::new();
        if !random && let Some(page) = &current {
            replay.push_back(page.clone());
        }
        let mut audio_start = current
            .as_ref()
            .map_or_else(|| source.position(), |page| page.offset);

        // Each stream's other header packets, which may span pages.
        let (mut headers, headers_end) = read_headers(
            &mut source,
            &mut kinds,
            &heads,
            &bos_sequences,
            &mut current,
            (!random).then_some(&mut replay),
            limits,
        )?;
        if let Some(end) = headers_end {
            audio_start = end;
        }

        let mut demux = Self {
            source,
            random,
            streams: Vec::new(),
            kinds,
            infos: Vec::new(),
            gaps: Vec::new(),
            ready: VecDeque::new(),
            replay,
            audio_start,
            link_end: None,
            chained: false,
            finished: false,
            cancelled: false,
            last_error: None,
            counters: Counters::default(),
            ignored_pages: 0,
            crc_failures: 0,
            skipped: leading,
            seek_read: 0,
            last_seek: SeekCost::default(),
        };
        let starts = demux.scan_starts()?;
        let mut tails = vec![(None, None); demux.kinds.len()];
        if random {
            let (link_end, chained) = demux.find_link_end()?;
            demux.link_end = Some(link_end);
            demux.chained = chained;
            tails = demux.scan_tails();
        }

        let mut infos = Vec::with_capacity(demux.kinds.len());
        let mut streams = Vec::with_capacity(demux.kinds.len());
        for (index, kind) in demux.kinds.iter().enumerate() {
            let id = TrackId::new(u32::try_from(index).unwrap_or(u32::MAX));
            let bytes = extradata.get(index).cloned().unwrap_or_default();
            match kind {
                Kind::Logical(logical) => {
                    let (end, trailing) = tails.get(index).copied().unwrap_or((None, None));
                    let span = Span {
                        start: starts.get(index).copied().flatten(),
                        end,
                        trailing,
                    };
                    let header = headers.get_mut(index).and_then(Option::take);
                    let (stream, info) = match (heads.get(index), header) {
                        (Some(Head::Opus(head)), Some(Headers::Opus(tags))) => {
                            opus_stream(id, logical, head.clone(), tags, bytes, span)?
                        }
                        (
                            Some(Head::Vorbis(head)),
                            Some(Headers::Vorbis {
                                comment,
                                comment_bytes,
                                setup,
                            }),
                        ) => {
                            let bytes = vorbis_extradata(
                                [&bytes, &comment_bytes, &setup],
                                &comment,
                                limits,
                            )?;
                            vorbis_stream(id, logical, *head, comment, bytes, span)?
                        }
                        _ => return Err(OggError::NoPage.into()),
                    };
                    streams.push(stream);
                    infos.push(Some(info));
                }
                Kind::Unknown { .. } => {
                    let tag =
                        UnknownCodec::new(mapping_tag(&bytes)).map_err(|_| OggError::TooLarge {
                            declared: 64,
                            limit: 64,
                        })?;
                    streams.push(Stream::new(
                        id,
                        StreamParams::opaque(StreamCodec::Unknown(tag), bytes),
                        Timescale::new(0),
                        None,
                        Vec::new(),
                        Trim::default(),
                    ));
                    infos.push(None);
                }
            }
        }
        demux.streams = streams;
        demux.infos = infos;
        demux.rewind()?;
        Ok(demux)
    }

    /// Back to the first audio page with every stream fresh.
    fn rewind(&mut self) -> Result<(), Fault> {
        if self.random {
            self.source.seek(self.audio_start)?;
        }
        for kind in &mut self.kinds {
            if let Kind::Logical(opus) = kind {
                opus.state = State::fresh();
            }
        }
        self.ready.clear();
        self.gaps.clear();
        self.counters = Counters::default();
        self.ignored_pages = 0;
        self.finished = false;
        Ok(())
    }

    /// Reads from the audio start until each Opus stream's first packet is
    /// timed. Returns each stream's first packet time.
    fn scan_starts(&mut self) -> Result<Vec<Option<i64>>, Fault> {
        let mut starts = vec![None; self.kinds.len()];
        if self.random {
            self.source.seek(self.audio_start)?;
        }
        let mut replayed = 0;
        let mut scanned = 0_u64;
        let mut scratch_gaps = Vec::new();
        let mut scratch = Counters::default();
        loop {
            let waiting = self.kinds.iter().enumerate().any(|(index, kind)| {
                matches!(kind, Kind::Logical(opus) if starts[index].is_none() && !opus.state.ended)
            });
            if !waiting || scanned > MAX_START_SCAN_BYTES {
                break;
            }
            let page = if !self.random && replayed < self.replay.len() {
                replayed += 1;
                self.replay[replayed - 1].clone()
            } else {
                match self.source.read_page()? {
                    PageRead::Page(page) => {
                        if !self.random {
                            self.replay.push_back(page.clone());
                            replayed += 1;
                        }
                        page
                    }
                    PageRead::End | PageRead::Truncated { .. } => break,
                }
            };
            scanned = scanned.saturating_add(u64::try_from(page.byte_len()).unwrap_or(0));
            let Some(index) = self.index_of(page.serial) else {
                if page.bos() {
                    break;
                }
                continue;
            };
            if page.bos() && page.offset >= self.audio_start {
                break;
            }
            if let Kind::Logical(opus) = &mut self.kinds[index] {
                let mut out = Vec::new();
                opus.take_page(&page, &mut out, &mut scratch_gaps, &mut scratch)?;
                if starts[index].is_none()
                    && let Some(first) = out.first()
                {
                    starts[index] = Some(first.pts);
                }
            }
        }
        Ok(starts)
    }

    fn index_of(&self, serial: u32) -> Option<usize> {
        self.kinds.iter().position(|kind| kind.serial() == serial)
    }

    /// The page belongs to the first link.
    fn in_first_link(&self, page: &Page) -> bool {
        self.index_of(page.serial).is_some() && !(page.bos() && page.offset >= self.audio_start)
    }

    /// Pages that start in `[from, to)`, read forward from `from`.
    fn pages_between(&mut self, from: u64, to: u64) -> Result<Vec<Page>, Fault> {
        self.source.seek(from)?;
        let mut pages = Vec::new();
        loop {
            match self.source.read_page() {
                Ok(PageRead::Page(page)) => {
                    if page.offset >= to {
                        break;
                    }
                    pages.push(page);
                }
                Ok(PageRead::End | PageRead::Truncated { .. })
                | Err(Fault::Ogg(_) | Fault::Seek(_)) => break,
                Err(Fault::Source(error)) => return Err(error.into()),
            }
        }
        Ok(pages)
    }

    /// The offset where the first link ends, and whether another link with
    /// its own beginning-of-stream pages follows.
    fn find_link_end(&mut self) -> Result<(u64, bool), Fault> {
        let length = self.source.length().unwrap_or(0);
        // The last page of the file.
        let mut to = length;
        let mut last = None;
        while to > self.audio_start && length - to <= MAX_TAIL_SCAN_BYTES {
            let from = to.saturating_sub(TAIL_CHUNK).max(self.audio_start);
            if let Some(page) = self.pages_between(from, to)?.pop() {
                last = Some(page);
                break;
            }
            to = from;
        }
        let Some(last) = last else {
            return Ok((length, false));
        };
        if self.in_first_link(&last) {
            return Ok((length, false));
        }
        // Bisect for the first page that is not in the first link.
        let mut candidate = last.offset;
        let mut lo = self.audio_start;
        let mut hi = candidate;
        let mut probes = 0;
        while hi.saturating_sub(lo) > BISECT_WINDOW && probes < MAX_LINK_PROBES {
            probes += 1;
            let mid = lo + (hi - lo) / 2;
            self.source.seek(mid)?;
            match self.source.read_page() {
                Ok(PageRead::Page(page)) if page.offset < hi => {
                    if self.in_first_link(&page) {
                        lo = page.offset + u64::try_from(page.byte_len()).unwrap_or(0);
                    } else {
                        candidate = candidate.min(page.offset);
                        hi = page.offset;
                    }
                }
                Ok(_) | Err(Fault::Ogg(_) | Fault::Seek(_)) => hi = mid,
                Err(Fault::Source(error)) => return Err(error.into()),
            }
        }
        let mut boundary = None;
        for page in self.pages_between(lo, candidate)? {
            if !self.in_first_link(&page) {
                boundary = Some(page);
                break;
            }
        }
        let (end, bos) = match boundary {
            Some(page) => (page.offset, page.bos()),
            None => (
                candidate,
                last.bos() || self.index_of(last.serial).is_none(),
            ),
        };
        Ok((end, bos))
    }

    /// Each Opus stream's final granule and trailing trim, read from the
    /// end of the first link.
    fn scan_tails(&mut self) -> Vec<(Option<i64>, Option<i64>)> {
        let mut tails = vec![(None, None); self.kinds.len()];
        let Some(link_end) = self.link_end else {
            return tails;
        };
        // The last two pages with a granule position, per stream.
        let mut marks: Vec<Vec<(u64, i64)>> = vec![Vec::new(); self.kinds.len()];
        let mut to = link_end;
        let mut reached_start = false;
        while !reached_start && link_end - to <= MAX_TAIL_SCAN_BYTES {
            let from = to.saturating_sub(TAIL_CHUNK).max(self.audio_start);
            reached_start = from <= self.audio_start;
            let Ok(pages) = self.pages_between(from, to) else {
                break;
            };
            let mut found: Vec<Vec<(u64, i64)>> = vec![Vec::new(); self.kinds.len()];
            for page in pages {
                if page.granule == -1 {
                    continue;
                }
                if let Some(index) = self.index_of(page.serial)
                    && let Kind::Logical(opus) = &self.kinds[index]
                    && page.sequence >= opus.first_audio_sequence
                {
                    found[index].push((page.offset, page.granule));
                }
            }
            for (index, mut chunk) in found.into_iter().enumerate() {
                chunk.append(&mut marks[index]);
                let keep = chunk.len().saturating_sub(2);
                marks[index] = chunk.split_off(keep);
            }
            let done =
                self.kinds.iter().enumerate().all(|(index, kind)| {
                    !matches!(kind, Kind::Logical(_)) || marks[index].len() >= 2
                });
            if done {
                break;
            }
            to = from;
        }
        for index in 0..self.kinds.len() {
            let Some(&(last_offset, last_granule)) = marks[index].last() else {
                continue;
            };
            let previous = if marks[index].len() >= 2 {
                Some(marks[index][0].0)
            } else if reached_start {
                None
            } else {
                tails[index] = (Some(last_granule), None);
                continue;
            };
            let trailing = self.trailing(index, previous, last_offset);
            tails[index] = (Some(last_granule), trailing);
        }
        tails
    }

    /// Samples the last page of stream `index` trims, decoded from the page
    /// before it (or from the stream start).
    fn trailing(&mut self, index: usize, previous: Option<u64>, last: u64) -> Option<i64> {
        let Kind::Logical(original) = self.kinds.get(index)?.clone() else {
            return None;
        };
        let mut opus = original;
        opus.state = match previous {
            Some(_) => State::resume(true),
            None => State::fresh(),
        };
        self.source
            .seek(previous.unwrap_or(self.audio_start))
            .ok()?;
        let mut scratch_gaps = Vec::new();
        let mut scratch = Counters::default();
        let mut scanned = 0_u64;
        loop {
            let PageRead::Page(page) = self.source.read_page().ok()? else {
                return None;
            };
            scanned = scanned.saturating_add(u64::try_from(page.byte_len()).unwrap_or(0));
            if page.offset > last || scanned > MAX_TAIL_SCAN_BYTES {
                return None;
            }
            if page.serial != opus.serial {
                continue;
            }
            let mut out = Vec::new();
            opus.take_page(&page, &mut out, &mut scratch_gaps, &mut scratch)
                .ok()?;
            if page.offset == last {
                if !page.eos() {
                    return Some(0);
                }
                return Some(out.iter().map(|ready| ready.samples - ready.duration).sum());
            }
        }
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot {
            position: self.source.position(),
            states: self
                .kinds
                .iter()
                .map(|kind| match kind {
                    Kind::Logical(opus) => Some(opus.state.clone()),
                    Kind::Unknown { .. } => None,
                })
                .collect(),
            ready: self.ready.clone(),
            finished: self.finished,
            gaps: self.gaps.len(),
            counters: self.counters,
            ignored_pages: self.ignored_pages,
            crc_failures: self.crc_failures,
            skipped: self.skipped,
        }
    }

    fn restore(&mut self, snapshot: Snapshot) {
        let _ = self.source.seek(snapshot.position);
        for (kind, state) in self.kinds.iter_mut().zip(snapshot.states) {
            if let (Kind::Logical(opus), Some(state)) = (kind, state) {
                opus.state = state;
            }
        }
        self.ready = snapshot.ready;
        self.finished = snapshot.finished;
        self.gaps.truncate(snapshot.gaps);
        self.counters = snapshot.counters;
        self.ignored_pages = snapshot.ignored_pages;
        self.crc_failures = snapshot.crc_failures;
        self.skipped = snapshot.skipped;
    }

    fn seek_inner(&mut self, to: SeekTo, cost: &mut SeekCost) -> Result<SeekLanding, Fault> {
        let Some(index) = usize::try_from(to.track)
            .ok()
            .filter(|&index| index < self.kinds.len())
        else {
            return Err(Fault::Seek(SeekError::UnknownTrack));
        };
        let Some(info) = self.infos.get(index).cloned().flatten() else {
            return Err(Fault::Seek(SeekError::Empty));
        };
        let serial = info.serial();
        let Some(start) = info.start() else {
            self.end_landing();
            return Ok(landing(Landing::End, 0, 0));
        };
        if info.end().is_some_and(|end| to.time >= end) || to.end.is_some_and(|end| to.time >= end)
        {
            self.end_landing();
            return Ok(landing(Landing::End, 0, 0));
        }
        let aim = to.time.saturating_sub(info.preroll()).max(start);
        let first_audio = match &self.kinds[index] {
            Kind::Logical(opus) => opus.first_audio_sequence,
            Kind::Unknown { .. } => return Err(Fault::Seek(SeekError::Empty)),
        };

        // Bisection for the last page of this stream whose granule is at or
        // before `aim`.
        let link_end = self.link_end.unwrap_or(self.audio_start);
        let mut lo = self.audio_start;
        let mut lo_granule = start;
        let mut hi = link_end;
        let mut hi_granule = info.end();
        let mut best: Option<(u64, i64)> = None;
        // Each probe interpolates between the bounds, a quarter window
        // toward the bound that has not moved yet, so two probes bracket
        // the target; each further probe on the same side doubles that
        // step. Three in a row on one side fall back to a midpoint.
        let mut last_low: Option<bool> = None;
        let mut same_side = 0_u32;
        while hi.saturating_sub(lo) > BISECT_WINDOW && cost.probes < MAX_SEEK_PROBES {
            cost.probes += 1;
            let mid = match hi_granule {
                Some(hi_g) if same_side < 3 && hi_g > lo_granule => {
                    let span = i128::from(hi - lo);
                    let along = i128::from(aim - lo_granule) * span / i128::from(hi_g - lo_granule);
                    let guess = lo.saturating_add(u64::try_from(along).unwrap_or(0));
                    let step = (BISECT_WINDOW / 4) << same_side.saturating_sub(1);
                    let biased = if last_low == Some(true) {
                        guess.saturating_add(step)
                    } else {
                        guess.saturating_sub(step)
                    };
                    biased.clamp(lo, hi - 1)
                }
                _ => lo + (hi - lo) / 2,
            };
            self.source.seek(mid)?;
            let mut hit = None;
            loop {
                match self.source.read_page() {
                    Ok(PageRead::Page(page)) => {
                        cost.pages = cost.pages.saturating_add(1);
                        if page.offset >= hi {
                            break;
                        }
                        if page.serial == serial
                            && page.granule != -1
                            && page.sequence >= first_audio
                            && !page.bos()
                        {
                            hit = Some(page);
                            break;
                        }
                    }
                    Ok(_) | Err(Fault::Ogg(_)) => break,
                    Err(error) => return Err(error),
                }
            }
            let low = hit.as_ref().is_some_and(|page| page.granule <= aim);
            same_side = if last_low == Some(low) {
                same_side + 1
            } else {
                1
            };
            last_low = Some(low);
            match hit {
                Some(page) if page.granule <= aim => {
                    best = Some((page.offset, page.granule));
                    lo = page.offset + u64::try_from(page.byte_len()).unwrap_or(0);
                    lo_granule = page.granule;
                }
                Some(page) => {
                    hi = mid;
                    hi_granule = Some(page.granule);
                }
                None => hi = mid,
            }
        }
        // Linear scan of the remaining window.
        self.source.seek(lo)?;
        loop {
            match self.source.read_page() {
                Ok(PageRead::Page(page)) => {
                    cost.pages = cost.pages.saturating_add(1);
                    if page.offset >= link_end {
                        break;
                    }
                    if page.serial != serial
                        || page.granule == -1
                        || page.sequence < first_audio
                        || page.bos()
                    {
                        if page.offset >= hi && best.is_some() {
                            break;
                        }
                        continue;
                    }
                    if page.granule > aim {
                        break;
                    }
                    best = Some((page.offset, page.granule));
                }
                Ok(_) | Err(Fault::Ogg(_)) => break,
                Err(error) => return Err(error),
            }
        }

        // Position at the landing page.
        let resume_at = match best {
            Some((offset, _)) => {
                for (position, kind) in self.kinds.iter_mut().enumerate() {
                    if let Kind::Logical(opus) = kind {
                        opus.state = State::resume(position == index);
                    }
                }
                offset
            }
            None => {
                for kind in &mut self.kinds {
                    if let Kind::Logical(opus) = kind {
                        opus.state = State::fresh();
                    }
                }
                self.audio_start
            }
        };
        self.source.seek(resume_at)?;
        self.replay.clear();
        self.ready.clear();
        self.finished = false;

        // Drop packets until the one that holds `aim`.
        let mut scans = 0_u32;
        let mut first_packet = best.is_none();
        let target = loop {
            let Some(ready) = self.next_ready()? else {
                self.end_landing();
                return Ok(landing(Landing::End, scans, cost.pages));
            };
            scans = scans.saturating_add(1);
            if ready.track != to.track {
                continue;
            }
            if ready.pts.saturating_add(ready.samples.max(1)) > aim {
                break ready;
            }
            first_packet = false;
        };

        // Count the packets before the requested time, then put them back.
        let mut preroll = u32::from(target.pts < to.time);
        if target.pts < to.time {
            let saved = self.snapshot();
            while let Ok(Some(ready)) = self.next_ready() {
                scans = scans.saturating_add(1);
                if ready.track != to.track {
                    continue;
                }
                if ready.pts >= to.time {
                    break;
                }
                preroll = preroll.saturating_add(1);
            }
            self.restore(saved);
        }
        let time = target.pts;
        self.ready.push_front(target);
        Ok(landing(
            Landing::Packet {
                ordinal: if first_packet { 0 } else { u32::MAX },
                time,
                keyframe: true,
                preroll,
                byte_position: resume_at,
            },
            scans,
            cost.pages,
        ))
    }

    fn end_landing(&mut self) {
        self.ready.clear();
        self.replay.clear();
        self.finished = true;
    }
}

fn landing(landing: Landing, scans: u32, pages: u32) -> SeekLanding {
    SeekLanding {
        landing,
        index_used: false,
        payload_scans: scans,
        entries_visited: pages,
    }
}

impl<R: Read> OggDemux<R> {
    /// The parsed headers and timing facts of an Opus track.
    #[must_use]
    pub fn opus(&self, track: u32) -> Option<&OpusInfo> {
        match self.infos.get(usize::try_from(track).ok()?)? {
            Some(Info::Opus(info)) => Some(info),
            _ => None,
        }
    }

    /// The parsed headers and timing facts of a Vorbis track.
    #[must_use]
    pub fn vorbis(&self, track: u32) -> Option<&VorbisInfo> {
        match self.infos.get(usize::try_from(track).ok()?)? {
            Some(Info::Vorbis(info)) => Some(info),
            _ => None,
        }
    }

    /// `OpusTags` of an Opus track: vendor and comments.
    #[must_use]
    pub fn tags(&self, track: u32) -> Option<&OpusTags> {
        self.opus(track).map(|info| &info.tags)
    }

    /// Another link with its own beginning-of-stream pages follows the
    /// first. Its packets are not returned.
    #[must_use]
    pub const fn chained(&self) -> bool {
        self.chained
    }

    /// Offset where the first link's pages end, on a random-access input.
    #[must_use]
    pub const fn link_end(&self) -> Option<u64> {
        self.link_end
    }

    /// Offset of the first page after the header packets.
    #[must_use]
    pub const fn audio_start(&self) -> u64 {
        self.audio_start
    }

    /// Candidate pages packet reads skipped because their checksum failed.
    /// Open and seek re-read pages; those reads are not counted here.
    #[must_use]
    pub const fn crc_failures(&self) -> u64 {
        self.crc_failures
    }

    /// Bytes skipped before the first page and by packet reads while
    /// searching for a capture pattern.
    #[must_use]
    pub const fn skipped_bytes(&self) -> u64 {
        self.skipped
    }

    /// Pages missing from sequence numbers since open or the last seek.
    #[must_use]
    pub const fn lost_pages(&self) -> u64 {
        self.counters.lost_pages
    }

    /// Packet pieces dropped because a neighbouring piece was lost.
    #[must_use]
    pub const fn dropped_pieces(&self) -> u64 {
        self.counters.dropped_pieces
    }

    /// Pages since open or the last seek whose granule position disagreed
    /// with the packet durations by less than the shortest packet, and was
    /// taken as rounding. A remux that rebuilds granules from millisecond
    /// timestamps writes such pages.
    #[must_use]
    pub const fn granule_drift(&self) -> u64 {
        self.counters.granule_drift
    }

    /// Pages of a serial the first link did not begin.
    #[must_use]
    pub const fn ignored_pages(&self) -> u64 {
        self.ignored_pages
    }

    /// The typed reason behind the last failed read or seek.
    #[must_use]
    pub const fn last_error(&self) -> Option<&OggError> {
        self.last_error.as_ref()
    }

    /// Bytes, probes, and pages the last seek used.
    #[must_use]
    pub const fn last_seek(&self) -> SeekCost {
        self.last_seek
    }

    /// Every byte read from the input, including seeks.
    #[must_use]
    pub fn total_read(&self) -> u64 {
        self.source.consumed()
    }

    fn next_page(&mut self) -> Result<PageRead, Fault> {
        if let Some(page) = self.replay.pop_front() {
            return Ok(PageRead::Page(page));
        }
        let crc = self.source.crc_failures();
        let skipped = self.source.skipped();
        let read = self.source.read_page();
        self.crc_failures = self
            .crc_failures
            .saturating_add(self.source.crc_failures().saturating_sub(crc));
        self.skipped = self
            .skipped
            .saturating_add(self.source.skipped().saturating_sub(skipped));
        read
    }

    /// Reads one page into the ready queue. False at the end of the link.
    fn pump(&mut self) -> Result<bool, Fault> {
        if self.finished {
            return Ok(false);
        }
        let page = match self.next_page()? {
            PageRead::Page(page) => page,
            PageRead::End => {
                self.finished = true;
                return Ok(false);
            }
            PageRead::Truncated { expected, got } => {
                self.finished = true;
                return Err(OggError::TruncatedPage { expected, got }.into());
            }
        };
        if self.link_end.is_some_and(|end| page.offset >= end) {
            self.finished = true;
            return Ok(false);
        }
        let Some(index) = self
            .kinds
            .iter()
            .position(|kind| kind.serial() == page.serial)
        else {
            if page.bos() {
                self.chained = true;
                self.finished = true;
                return Ok(false);
            }
            self.ignored_pages = self.ignored_pages.saturating_add(1);
            return Ok(true);
        };
        if page.bos() && page.offset >= self.audio_start {
            self.chained = true;
            self.finished = true;
            return Ok(false);
        }
        if let Kind::Logical(opus) = &mut self.kinds[index] {
            let mut out = Vec::new();
            let mut gaps = Vec::new();
            let result = opus.take_page(&page, &mut out, &mut gaps, &mut self.counters);
            for gap in gaps {
                if !self.gaps.contains(&gap) {
                    self.gaps.push(gap);
                }
            }
            result?;
            self.ready.extend(out);
        }
        Ok(true)
    }

    fn next_ready(&mut self) -> Result<Option<Ready>, Fault> {
        loop {
            if let Some(ready) = self.ready.pop_front() {
                return Ok(Some(ready));
            }
            if !self.pump()? {
                return Ok(None);
            }
        }
    }

    fn read_error(&mut self, fault: Fault) -> ReadError {
        match fault {
            Fault::Source(error) => ReadError::from_packet(error),
            Fault::Seek(error) => ReadError::Seek(error),
            Fault::Ogg(error) => {
                let mapped = error.to_read();
                self.last_error = Some(error);
                mapped
            }
        }
    }
}

/// Reads `OpusTags` for every Opus stream. Updates each stream's first audio
/// sequence number and the audio start offset.
/// Where a stream's samples begin and end in the granule domain.
#[derive(Clone, Copy)]
struct Span {
    start: Option<i64>,
    end: Option<i64>,
    trailing: Option<i64>,
}

/// The track of an Opus stream (RFC 7845 section 4): the edit starts after
/// the pre-skip, which the decoder drops.
fn opus_stream(
    id: TrackId,
    opus: &Logical,
    head: OpusHead,
    tags: OpusTags,
    extradata: Vec<u8>,
    span: Span,
) -> Result<(Stream, Info), OggError> {
    let Span {
        start,
        end,
        trailing,
    } = span;
    let pre_skip = i64::from(head.pre_skip);
    let begin = start.unwrap_or(0);
    let playable = match end {
        Some(end) => {
            let playable = end - begin - pre_skip;
            if playable < 0 && start.is_some() {
                return Err(OggError::PreSkipPastEnd {
                    serial: opus.serial,
                    end,
                    pre_skip,
                });
            }
            playable.max(0)
        }
        None => 0,
    };
    let edit = Edit::try_new(Some(begin + pre_skip), playable, 1, 1).map_err(|_| {
        OggError::BadOpusHead {
            serial: opus.serial,
            reason: "edit does not fit",
        }
    })?;
    let trim = Trim::new(
        Some(u64::try_from(pre_skip).unwrap_or(0)),
        trailing.and_then(|value| u64::try_from(value).ok()),
    );
    let params = StreamParams::try_audio(
        StreamCodec::Known(CodecId::Opus),
        OPUS_RATE,
        u16::from(head.channels),
        extradata,
    )
    .map_err(|_| OggError::BadOpusHead {
        serial: opus.serial,
        reason: "zero output channels",
    })?;
    let stream = Stream::new(
        id,
        params,
        Timescale::new(OPUS_RATE),
        None,
        vec![edit],
        trim,
    );
    let info = Info::Opus(OpusInfo {
        serial: opus.serial,
        head,
        tags,
        start,
        end,
        trailing,
    });
    Ok((stream, info))
}

/// The three Vorbis headers, Xiph-laced. When they exceed
/// `max_extradata_bytes`, the comment packet is written again with only its
/// vendor string: a decoder reads nothing else from it.
fn vorbis_extradata(
    packets: [&[u8]; 3],
    comment: &VorbisComment,
    limits: Limits,
) -> Result<Vec<u8>, OggError> {
    let fits =
        |bytes: &[u8]| u64::try_from(bytes.len()).unwrap_or(u64::MAX) <= limits.max_extradata_bytes;
    let laced = xiph_lace(packets);
    if fits(&laced) {
        return Ok(laced);
    }
    let vendor_only = VorbisComment {
        vendor: comment.vendor.clone(),
        comments: Vec::new(),
        cut: true,
    }
    .encode();
    let laced = xiph_lace([packets[0], &vendor_only, packets[2]]);
    if fits(&laced) {
        return Ok(laced);
    }
    Err(OggError::TooLarge {
        declared: u64::try_from(laced.len()).unwrap_or(u64::MAX),
        limit: limits.max_extradata_bytes,
    })
}

/// The track of a Vorbis stream (Vorbis I section A.2). The timescale is
/// the sample rate. Samples before granule zero are leading trim, and the
/// edit starts at zero or at the first sample, whichever is later.
fn vorbis_stream(
    id: TrackId,
    vorbis: &Logical,
    head: VorbisId,
    comment: VorbisComment,
    extradata: Vec<u8>,
    span: Span,
) -> Result<(Stream, Info), OggError> {
    let Span {
        start,
        end,
        trailing,
    } = span;
    let bad = |reason| OggError::BadVorbisHeader {
        serial: vorbis.serial,
        reason,
    };
    let begin = start.unwrap_or(0).max(0);
    let playable = end.map_or(0, |end| (end - begin).max(0));
    let leading = start.map_or(0, |start| start.min(0).unsigned_abs());
    let edit = Edit::try_new(Some(begin), playable, 1, 1).map_err(|_| bad("edit does not fit"))?;
    let trim = Trim::new(
        Some(leading),
        trailing.and_then(|value| u64::try_from(value).ok()),
    );
    let params = StreamParams::try_audio(
        StreamCodec::Known(CodecId::Vorbis),
        head.sample_rate,
        u16::from(head.channels),
        extradata,
    )
    .map_err(|_| bad("audio parameters out of range"))?;
    let stream = Stream::new(
        id,
        params,
        Timescale::new(head.sample_rate),
        None,
        vec![edit],
        trim,
    );
    let modes = match &vorbis.timing {
        Timing::Vorbis(modes) => modes.long.clone(),
        Timing::Opus { .. } => Vec::new(),
    };
    let info = Info::Vorbis(VorbisInfo {
        serial: vorbis.serial,
        id: head,
        comment,
        modes,
        start,
        end,
        trailing,
    });
    Ok((stream, info))
}

/// Reads each stream's header packets after the first: `OpusTags` (RFC 7845
/// section 5.2), or the Vorbis comment and setup headers (Vorbis I section
/// A.2). The last header packet finishes its page. A comment longer than
/// `max_header_bytes` is cut; a longer setup header is refused, because the
/// stream cannot be decoded without all of it. Sets each Vorbis stream's
/// modes, and each stream's first audio page sequence. Returns the headers
/// and the offset after the first page where a stream's headers end.
fn read_headers<R: Read>(
    source: &mut Source<R>,
    kinds: &mut [Kind],
    heads: &[Head],
    bos_sequences: &[u32],
    current: &mut Option<Page>,
    mut replay: Option<&mut VecDeque<Page>>,
    limits: Limits,
) -> Result<(Vec<Option<Headers>>, Option<u64>), Fault> {
    struct Pending {
        /// Header packets completed so far.
        done: Vec<(Vec<u8>, bool)>,
        /// Header packets the mapping has after the first.
        wanted: usize,
        bytes: Vec<u8>,
        len: u64,
        /// A packet is partly read: the next page continues it.
        open: bool,
        next_sequence: u32,
    }
    let mut headers: Vec<Option<Headers>> = Vec::with_capacity(kinds.len());
    headers.resize_with(kinds.len(), || None);
    let mut pending: Vec<Option<Pending>> = heads
        .iter()
        .zip(bos_sequences)
        .map(|(head, &bos)| {
            let wanted = match head {
                Head::Opus(_) => 1,
                Head::Vorbis(_) => 2,
                Head::Unknown => return None,
            };
            Some(Pending {
                done: Vec::new(),
                wanted,
                bytes: Vec::new(),
                len: 0,
                open: false,
                next_sequence: bos.wrapping_add(1),
            })
        })
        .collect();
    let missing = |head: &Head, serial| match head {
        Head::Vorbis(_) => OggError::MissingVorbisHeaders { serial },
        Head::Opus(_) | Head::Unknown => OggError::MissingOpusTags { serial },
    };
    let cap = limits.max_header_bytes;
    let wanted = pending.iter().flatten().map(|state| state.wanted).max();
    let scan_limit = cap
        .saturating_mul(u64::try_from(wanted.unwrap_or(1)).unwrap_or(u64::MAX))
        .saturating_add(MAX_START_SCAN_BYTES);
    let mut scanned = 0_u64;
    let mut audio_start: Option<u64> = None;
    while let Some(waiting) = pending.iter().position(Option::is_some) {
        let page = match current.take() {
            Some(page) => page,
            None => match source.read_page()? {
                PageRead::Page(page) => {
                    if let Some(replay) = replay.as_mut() {
                        replay.push_back(page.clone());
                    }
                    page
                }
                PageRead::End => {
                    return Err(missing(&heads[waiting], kinds[waiting].serial()).into());
                }
                PageRead::Truncated { expected, got } => {
                    return Err(OggError::TruncatedPage { expected, got }.into());
                }
            },
        };
        scanned = scanned.saturating_add(u64::try_from(page.byte_len()).unwrap_or(0));
        if scanned > scan_limit {
            return Err(OggError::TooLarge {
                declared: scanned,
                limit: scan_limit,
            }
            .into());
        }
        let Some(index) = kinds.iter().position(|kind| kind.serial() == page.serial) else {
            if page.bos() {
                return Err(missing(&heads[waiting], kinds[waiting].serial()).into());
            }
            continue;
        };
        let Some(state) = pending[index].as_mut() else {
            continue;
        };
        let serial = page.serial;
        let vorbis = matches!(heads[index], Head::Vorbis(_));
        let bad = |reason| {
            Fault::Ogg(if vorbis {
                OggError::BadVorbisHeader { serial, reason }
            } else {
                OggError::BadOpusTags { serial, reason }
            })
        };
        if page.bos() {
            return Err(bad("a stream begins twice"));
        }
        if page.sequence != state.next_sequence {
            return Err(bad("a header page is missing"));
        }
        state.next_sequence = page.sequence.wrapping_add(1);
        for (position, piece) in page.pieces().iter().enumerate() {
            if state.done.len() == state.wanted {
                return Err(if vorbis {
                    OggError::SetupPageNotFinished { serial }
                } else {
                    OggError::TagsPageNotFinished { serial }
                }
                .into());
            }
            if position == 0 && page.continued() != state.open {
                return Err(bad("header page continuation does not match"));
            }
            let data = &page.body[piece.start..piece.end];
            let room = usize::try_from(cap.saturating_sub(state.len)).unwrap_or(usize::MAX);
            state.bytes.extend_from_slice(&data[..data.len().min(room)]);
            state.len = state
                .len
                .saturating_add(u64::try_from(data.len()).unwrap_or(u64::MAX));
            state.open = !piece.complete;
            if piece.complete {
                let cut = state.len > cap;
                // Vorbis I section A.2: the setup header is the last header.
                if vorbis && cut && state.done.len() + 1 == state.wanted {
                    return Err(OggError::TooLarge {
                        declared: state.len,
                        limit: cap,
                    }
                    .into());
                }
                state.done.push((std::mem::take(&mut state.bytes), cut));
                state.len = 0;
            }
        }
        if state.done.len() < state.wanted {
            continue;
        }
        let done = std::mem::take(&mut state.done);
        pending[index] = None;
        let parsed = match (&heads[index], &mut kinds[index], done.as_slice()) {
            (Head::Opus(_), _, [(tags, cut)]) => {
                Headers::Opus(OpusTags::parse(tags, serial, *cut)?)
            }
            (Head::Vorbis(id), Kind::Logical(logical), [(comment, cut), (setup, _)]) => {
                let parsed = VorbisComment::parse(comment, serial, *cut)?;
                logical.timing = Timing::Vorbis(VorbisModes::parse(setup, id, serial)?);
                let comment_bytes = if *cut {
                    parsed.encode()
                } else {
                    comment.clone()
                };
                Headers::Vorbis {
                    comment: parsed,
                    comment_bytes,
                    setup: setup.clone(),
                }
            }
            _ => return Err(missing(&heads[index], serial).into()),
        };
        headers[index] = Some(parsed);
        if let Kind::Logical(logical) = &mut kinds[index] {
            logical.first_audio_sequence = page.sequence.wrapping_add(1);
        }
        let after = page
            .offset
            .saturating_add(u64::try_from(page.byte_len()).unwrap_or(0));
        audio_start = Some(audio_start.map_or(after, |start| start.min(after)));
    }
    Ok((headers, audio_start))
}

fn open_error(fault: Fault) -> OpenError {
    match fault {
        Fault::Source(error) => OpenError::from_header(error),
        Fault::Ogg(error) => error.to_open(),
        Fault::Seek(_) => OpenError::Source {
            message: "seek during open".to_owned(),
        },
    }
}

impl<R: Read + Seek + Send> Demux for OggDemux<R> {
    fn family(&self) -> ContainerId {
        ContainerId::Ogg
    }

    fn streams(&self) -> &[Stream] {
        &self.streams
    }

    /// Timeline jumps found so far: a page whose granule position starts
    /// later than the previous page ended, after a lost page or a muxer gap.
    fn gaps(&self) -> &[Gap] {
        &self.gaps
    }

    fn read_packet(&mut self) -> Result<Option<Packet>, ReadError> {
        if self.cancelled {
            return Err(ReadError::Cancelled);
        }
        let ready = match self.next_ready() {
            Ok(Some(ready)) => ready,
            Ok(None) => return Ok(None),
            Err(fault) => return Err(self.read_error(fault)),
        };
        let bytes = match ready.data {
            Ok(bytes) => bytes,
            Err((declared, limit)) => {
                self.last_error = Some(OggError::TooLarge { declared, limit });
                return Err(ReadError::HostileSize { declared, limit });
            }
        };
        let timestamp = Timestamp::try_new(Some(ready.pts), Some(ready.pts), Some(ready.duration))
            .map_err(|_| ReadError::Source {
                message: "negative Ogg packet duration".to_owned(),
            })?;
        let timescale = usize::try_from(ready.track)
            .ok()
            .and_then(|index| self.streams.get(index))
            .map_or(Timescale::new(OPUS_RATE), Stream::timescale);
        Packet::try_new(ready.track, bytes, timestamp, timescale, true)
            .map_err(|error| ReadError::Source {
                message: error.to_string(),
            })
            .map(Some)
    }

    fn seek(&mut self, to: SeekTo) -> Result<SeekLanding, ReadError> {
        if self.cancelled {
            return Err(ReadError::Cancelled);
        }
        if !self.random {
            self.last_error = Some(OggError::ForwardOnly);
            return Err(ReadError::Seek(SeekError::ForwardOnly));
        }
        let before = self.source.consumed();
        let saved = self.snapshot();
        let mut cost = SeekCost::default();
        let result = self.seek_inner(to, &mut cost);
        if result.is_err() {
            self.restore(saved);
        }
        cost.bytes = self.source.consumed().saturating_sub(before);
        self.seek_read = self.seek_read.saturating_add(cost.bytes);
        self.last_seek = cost;
        result.map_err(|fault| self.read_error(fault))
    }

    fn cancel(&mut self) {
        self.cancelled = true;
    }

    /// Bytes read by open and by packet reads. Bytes a seek reads are on
    /// [`OggDemux::last_seek`] and [`OggDemux::total_read`] instead.
    fn consumed(&self) -> u64 {
        self.source.consumed().saturating_sub(self.seek_read)
    }
}
