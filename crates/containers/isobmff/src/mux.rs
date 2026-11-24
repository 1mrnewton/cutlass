//! Writes an ISO base media file.
//!
//! There are four layouts.
//!
//! - Progressive, the default: `ftyp`, `mdat`, then `moov`. Packets go to
//!   the output as they arrive. Until `finish` the file has no index.
//! - Fast start (`moov_at_end: false`): the same file with `moov` before
//!   `mdat`. On an output that can be read back ([`IsoBmffMux::with_reader`])
//!   the payload is written first and moved up at `finish` in fixed-size
//!   blocks, so memory does not grow with the file. On an output that cannot
//!   be read back the payload waits in memory, up to
//!   `MAX_BUFFERED_MUX_BYTES`.
//! - Fragmented (`fragmented: true`): `ftyp`, `moov` with `mvex`, then
//!   `sidx`/`moof`/`mdat` groups, then `mfra`.
//! - Recoverable (`recoverable: true`): fragments are written while packets
//!   arrive, and `finish` turns the file into a progressive or fast-start
//!   movie. The fragments' payload becomes the movie's `mdat`, and the
//!   `moov` and `moof` boxes written on the way are left inside it as
//!   unreferenced bytes. A process that dies before `finish` leaves a
//!   fragmented file that plays up to its last whole fragment.
//!
//! A fragment ends before a video keyframe once it holds
//! `fragment_millis` of video, or once any track holds that much when there
//! is no video. It also ends when it would pass `fragment_bytes` or
//! `samples_per_fragment`, and when one track has waited ten times the
//! target with no keyframe to cut at: twice the target in a recoverable
//! file, whose fragments are only ever read after a crash, so that little
//! is lost when the keyframes are far apart. Each track has one `traf` in a
//! fragment. Fragmented output keeps the caller's interleave, with a `trun`
//! for each run of a track's samples; a recoverable file lays each track's
//! samples together under one `trun`.
//!
//! Packet decode times on a track start at 0 and advance by each sample's
//! duration. Presentation time may differ; that difference is the composition
//! offset. Edit durations use `MuxOptions::movie_timescale` when the caller
//! sets it. Otherwise the movie timescale is the least common multiple of
//! the track timescales when that is at most `MAX_MOVIE_TIMESCALE`, the
//! first video track's timescale when not, and 1000 as the last resort.
//! Durations that do not divide round up. `mvhd`, `tkhd`, `mdhd` and `elst`
//! are version 1, with 64-bit times, when a value does not fit 32 bits.

use crate::alac;
use crate::legacy::{
    CodecId, ColorTags, Edit, Error, MediaKind, Mux, Packet, SampleFormat, StreamParams,
};
use crate::limits::{MAX_BUFFERED_MUX_BYTES, MAX_COVER_BYTES, MAX_PACKET_BYTES, MAX_TRACKS};
use crate::movie::pack_language;
use std::io::{Read, Seek, SeekFrom, Write};

/// The largest movie timescale chosen from the track timescales. A 32-bit
/// duration at this scale lasts 71 minutes; longer movies use 64-bit boxes.
pub const MAX_MOVIE_TIMESCALE: u32 = 1_000_000;

/// The fragment length aimed for, in milliseconds.
pub const DEFAULT_FRAGMENT_MILLIS: u32 = 1_000;

/// The payload bytes one fragment may hold before it is cut early.
pub const DEFAULT_FRAGMENT_BYTES: u64 = 16 * 1024 * 1024;

/// Consecutive samples of a track share a chunk up to this many bytes, the
/// bound FFmpeg's movenc.c uses.
const CHUNK_BYTES: u64 = 1024 * 1024;

/// The block moved at a time when `moov` is put before the payload.
const SHIFT_BLOCK: usize = 4 * 1024 * 1024;

/// Seconds from 1904-01-01, the epoch of the header times, to 1970-01-01.
const EPOCH_1904: u64 = 2_082_844_800;

/// A track waits at most this many fragment lengths for a video keyframe
/// before a cut is forced: in fragmented output, and in a recoverable file.
const FORCED_CUT_LENGTHS: u64 = 10;
const FORCED_CUT_LENGTHS_RECOVERABLE: u64 = 2;

const SAMPLE_KEY: u32 = 0x0200_0000;
const SAMPLE_NOT_KEY: u32 = 0x0101_0000;

/// Brand written in `ftyp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FileBrand {
    #[default]
    Mp4,
    Mov,
    M4a,
    M4v,
    ThreeGp,
}

/// Where the index goes, and whether fragments are written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MuxOptions {
    pub brand: FileBrand,
    /// When true, `mdat` is written first and `moov` last. When false,
    /// `moov` comes first. Ignored for fragments.
    pub moov_at_end: bool,
    pub fragmented: bool,
    /// The most samples, over all tracks, in one fragment. 0 is no bound.
    pub samples_per_fragment: u32,
    /// Movie-header timescale. Edit durations are in this scale.
    /// `None` picks one from the track timescales, at most
    /// [`MAX_MOVIE_TIMESCALE`].
    pub movie_timescale: Option<u32>,
    /// Write fragments while packets arrive and make the file a normal
    /// movie at `finish`. Ignored when `fragmented` is set.
    pub recoverable: bool,
    /// The fragment length aimed for, in milliseconds.
    pub fragment_millis: u32,
    /// The payload bytes one fragment may hold before it is cut early.
    /// This bounds the memory a fragment takes.
    pub fragment_bytes: u64,
    /// Creation time in seconds since 1970-01-01 UTC, written as the
    /// creation and modification time of the movie and each track. `None`
    /// writes 0, "unknown".
    pub creation_time: Option<u64>,
}

impl Default for MuxOptions {
    fn default() -> Self {
        Self {
            brand: FileBrand::Mp4,
            moov_at_end: true,
            fragmented: false,
            samples_per_fragment: 0,
            movie_timescale: None,
            recoverable: false,
            fragment_millis: DEFAULT_FRAGMENT_MILLIS,
            fragment_bytes: DEFAULT_FRAGMENT_BYTES,
            creation_time: None,
        }
    }
}

/// How the output is laid out while packets arrive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layout {
    /// Payload straight to the output inside one `mdat`.
    Progressive,
    /// Payload in memory until `finish`, for `moov` first on an output that
    /// cannot be read back.
    Buffered,
    Fragmented,
    Recoverable,
}

/// One sample in the movie index.
#[derive(Clone, Copy)]
struct Sample {
    /// Byte offset in the output, or in the buffered payload.
    offset: u64,
    duration: u32,
    size: u32,
    /// Presentation time less decode time.
    cts: i32,
    keyframe: bool,
}

/// One sample waiting in the current fragment.
#[derive(Clone, Copy)]
struct FragSample {
    duration: u32,
    size: u32,
    cts: i32,
    keyframe: bool,
}

/// Samples of one track that lie one after another in the fragment buffer.
#[derive(Clone, Copy)]
struct Run {
    /// Index of the run's first sample among the track's pending samples.
    first: usize,
    /// Where the run starts in the fragment buffer, and its length.
    at: usize,
    len: usize,
}

struct TfraRow {
    time: u64,
    moof_offset: u64,
    traf_number: u8,
    trun_number: u32,
    sample_number: u32,
}

struct TrackOut {
    params: StreamParams,
    edits: Vec<Edit>,
    next_dts: i64,
    /// Encoder delay and trailing padding, in the track timescale.
    /// `None` leaves the track as it was. AAC, AC-3 and E-AC-3 keep both
    /// inside the frame durations, so the edit has to name them.
    gapless: Option<(u64, u64)>,
    /// Every sample written so far. Empty in fragmented output, whose
    /// index is in the fragments.
    samples: Vec<Sample>,
    /// The current fragment's samples, and where their payload is in the
    /// fragment buffer.
    pending: Vec<FragSample>,
    runs: Vec<Run>,
    /// Decode time of the first pending sample.
    pending_dts: i64,
    /// Sum of the pending durations.
    pending_ticks: u64,
    tfra: Vec<TfraRow>,
}

impl TrackOut {
    fn is_video(&self) -> bool {
        matches!(self.params.kind, MediaKind::Video(_))
    }
}

/// One sample handed to the muxer, with its payload borrowed.
pub(crate) struct SampleIn<'a> {
    pub track_id: u32,
    pub payload: &'a [u8],
    pub dts: i64,
    pub pts: i64,
    pub duration: u32,
    pub timescale: u32,
    pub keyframe: bool,
}

/// Reads from the output at its current position.
type ReadBack<W> = fn(&mut W, &mut [u8]) -> std::io::Result<usize>;

/// Muxes packets into an ISOBMFF file.
pub struct IsoBmffMux<W: Write + Seek> {
    writer: Option<W>,
    /// Reads the output back, for moving the payload behind `moov`.
    read: Option<ReadBack<W>>,
    options: MuxOptions,
    tracks: Vec<TrackOut>,
    buffered: Vec<u8>,
    cursor: u64,
    /// Where the 16 bytes that become the `mdat` header are.
    mdat_at: Option<u64>,
    started: bool,
    finished: bool,
    sequence: u32,
    /// The current fragment's payload, in the order it arrived.
    fragment: Vec<u8>,
    /// Samples waiting in the current fragment.
    pending_samples: u64,
    /// A JPEG written as the movie's cover art in `moov/udta`.
    cover: Option<Vec<u8>>,
}

impl<W: Read + Write + Seek> IsoBmffMux<W> {
    /// A muxer whose output can be read back. With `moov_at_end: false` the
    /// payload is written as it arrives and moved behind `moov` at `finish`,
    /// a block at a time, however large the file is.
    #[must_use]
    pub fn with_reader(writer: W, options: MuxOptions) -> Self {
        let mut mux = Self::new(writer, options);
        mux.read = Some(<W as Read>::read);
        mux
    }
}

impl<W: Write + Seek> IsoBmffMux<W> {
    #[must_use]
    pub fn new(writer: W, options: MuxOptions) -> Self {
        Self {
            writer: Some(writer),
            read: None,
            options,
            tracks: Vec::new(),
            buffered: Vec::new(),
            cursor: 0,
            mdat_at: None,
            started: false,
            finished: false,
            sequence: 1,
            fragment: Vec::new(),
            pending_samples: 0,
            cover: None,
        }
    }

    /// Sets the movie's cover art, a JPEG picture. It is written as iTunes
    /// metadata, `moov/udta/meta/ilst/covr`, with a `data` atom of type 13
    /// (JPEG), as FFmpeg's movenc.c writes it. A fragmented file
    /// writes `moov` with its first fragment, so the cover must come
    /// before that. Bytes that do not start with a JPEG start-of-image
    /// marker, or more than `MAX_COVER_BYTES`, are refused and the file is
    /// unchanged.
    pub fn set_cover_art(&mut self, jpeg: Vec<u8>) -> Result<(), Error> {
        if self.finished || (self.options.fragmented && self.started) {
            return Err(Error::Incomplete);
        }
        if !jpeg.starts_with(&[0xFF, 0xD8, 0xFF]) {
            return Err(Error::Unsupported {
                what: "cover art that is not a JPEG",
            });
        }
        let size = u64::try_from(jpeg.len()).unwrap_or(u64::MAX);
        if size > MAX_COVER_BYTES {
            return Err(Error::Limit {
                what: "cover art",
                value: size,
            });
        }
        self.cover = Some(jpeg);
        Ok(())
    }

    /// Remember the encoder delay and trailing padding of `track_id`.
    ///
    /// Both are samples in the track timescale. A track that already has
    /// edits keeps those. Otherwise finish writes one edit (ISO/IEC 14496-12
    /// `elst`): `media_time` is the first sample to present, and the segment
    /// lasts for the samples that remain after the padding. Packet times are
    /// unchanged and still include the priming.
    pub(crate) fn set_gapless(
        &mut self,
        track_id: u32,
        delay: u64,
        padding: u64,
    ) -> Result<(), Error> {
        if self.finished {
            return Err(Error::Incomplete);
        }
        let Some(track) = self
            .tracks
            .iter_mut()
            .find(|track| track.params.track_id == track_id)
        else {
            return Err(Error::Missing { what: "track" });
        };
        track.gapless = Some((delay, padding));
        Ok(())
    }

    pub fn set_edits(&mut self, track_id: u32, edits: Vec<Edit>) -> Result<(), Error> {
        let Some(track) = self
            .tracks
            .iter_mut()
            .find(|track| track.params.track_id == track_id)
        else {
            return Err(Error::Missing { what: "track" });
        };
        track.edits = edits;
        Ok(())
    }

    /// Replace a track's codec configuration record. The sample
    /// description is written at finish, so this holds until then. A
    /// fragmented file writes it with the first fragment; a recoverable
    /// file writes it there too and again, as it then stands, at finish.
    pub fn set_extradata(&mut self, track_id: u32, extradata: Vec<u8>) -> Result<(), Error> {
        if self.finished {
            return Err(Error::Incomplete);
        }
        let Some(track) = self
            .tracks
            .iter_mut()
            .find(|track| track.params.track_id == track_id)
        else {
            return Err(Error::Missing { what: "track" });
        };
        track.params.extradata = extradata;
        Ok(())
    }

    /// Sets a track's color code points, written as a `colr` box in its
    /// sample description at finish.
    pub fn set_color(&mut self, track_id: u32, color: ColorTags) -> Result<(), Error> {
        if self.finished {
            return Err(Error::Incomplete);
        }
        let Some(track) = self
            .tracks
            .iter_mut()
            .find(|track| track.params.track_id == track_id)
        else {
            return Err(Error::Missing { what: "track" });
        };
        track.params.color = Some(color);
        Ok(())
    }

    pub fn finish_into(mut self) -> Result<W, Error> {
        self.finish()?;
        self.writer.take().ok_or(Error::Incomplete)
    }

    fn layout(&self) -> Result<Layout, Error> {
        if self.options.fragmented {
            return Ok(Layout::Fragmented);
        }
        let first = !self.options.moov_at_end;
        if self.options.recoverable {
            if first && self.read.is_none() {
                return Err(Error::Unsupported {
                    what: "a recoverable file with moov first on an output that cannot be read back",
                });
            }
            return Ok(Layout::Recoverable);
        }
        if first && self.read.is_none() {
            return Ok(Layout::Buffered);
        }
        Ok(Layout::Progressive)
    }

    fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), Error> {
        let Some(writer) = self.writer.as_mut() else {
            return Err(Error::Incomplete);
        };
        writer.write_all(bytes).map_err(|err| Error::Io {
            at: self.cursor,
            kind: err.kind(),
        })?;
        self.cursor = self
            .cursor
            .checked_add(u64::try_from(bytes.len()).unwrap_or(u64::MAX))
            .ok_or(Error::Limit {
                what: "output",
                value: self.cursor,
            })?;
        Ok(())
    }

    /// Hands what is written so far to the system, so it outlives this
    /// process.
    fn flush_output(&mut self) -> Result<(), Error> {
        let Some(writer) = self.writer.as_mut() else {
            return Err(Error::Incomplete);
        };
        writer.flush().map_err(|err| Error::Io {
            at: self.cursor,
            kind: err.kind(),
        })
    }

    fn patch(&mut self, at: u64, bytes: &[u8]) -> Result<(), Error> {
        let Some(writer) = self.writer.as_mut() else {
            return Err(Error::Incomplete);
        };
        writer.seek(SeekFrom::Start(at)).map_err(|err| Error::Io {
            at,
            kind: err.kind(),
        })?;
        writer.write_all(bytes).map_err(|err| Error::Io {
            at,
            kind: err.kind(),
        })?;
        writer
            .seek(SeekFrom::Start(self.cursor))
            .map_err(|err| Error::Io {
                at: self.cursor,
                kind: err.kind(),
            })?;
        Ok(())
    }

    /// Moves the bytes from `from` to the end of the output `by` bytes
    /// later, last block first, so no byte is overwritten before it is read.
    fn shift_later(&mut self, from: u64, by: u64) -> Result<(), Error> {
        let (Some(writer), Some(read)) = (self.writer.as_mut(), self.read) else {
            return Err(Error::Incomplete);
        };
        let io = |at: u64| {
            move |err: std::io::Error| Error::Io {
                at,
                kind: err.kind(),
            }
        };
        let mut block = vec![0u8; SHIFT_BLOCK];
        let mut end = self.cursor;
        while end > from {
            let len = usize::try_from(end - from)
                .unwrap_or(SHIFT_BLOCK)
                .min(SHIFT_BLOCK);
            let start = end - u64::try_from(len).unwrap_or(0);
            writer.seek(SeekFrom::Start(start)).map_err(io(start))?;
            let mut filled = 0;
            while filled < len {
                let got = read(writer, &mut block[filled..len]).map_err(io(start))?;
                if got == 0 {
                    return Err(Error::Io {
                        at: start,
                        kind: std::io::ErrorKind::UnexpectedEof,
                    });
                }
                filled += got;
            }
            let to = start.checked_add(by).ok_or(Error::Limit {
                what: "output",
                value: start,
            })?;
            writer.seek(SeekFrom::Start(to)).map_err(io(to))?;
            writer.write_all(&block[..len]).map_err(io(to))?;
            end = start;
        }
        self.cursor = self.cursor.checked_add(by).ok_or(Error::Limit {
            what: "output",
            value: self.cursor,
        })?;
        writer
            .seek(SeekFrom::Start(self.cursor))
            .map_err(io(self.cursor))?;
        Ok(())
    }

    /// Writes what comes before the first payload byte.
    fn start(&mut self) -> Result<(), Error> {
        if self.started {
            return Ok(());
        }
        let layout = self.layout()?;
        let ftyp = ftyp_bytes(self.options.brand, layout, &self.tracks);
        let times = header_time(&self.options);
        match layout {
            Layout::Buffered => {}
            Layout::Fragmented => {
                let cover = self.cover.as_deref();
                let moov = init_moov(&self.tracks, self.movie_scale()?, cover, times)?;
                self.write_bytes(&ftyp)?;
                self.write_bytes(&moov)?;
            }
            Layout::Recoverable => {
                // The cover goes in the movie written at finish.
                let moov = init_moov(&self.tracks, self.movie_scale()?, None, times)?;
                self.write_bytes(&ftyp)?;
                self.mdat_at = Some(self.cursor);
                // Sixteen free bytes that finish turns into the `mdat`
                // header over everything written after them.
                self.write_bytes(&[0, 0, 0, 16, b'f', b'r', b'e', b'e', 0, 0, 0, 0, 0, 0, 0, 0])?;
                self.write_bytes(&moov)?;
            }
            Layout::Progressive => {
                self.write_bytes(&ftyp)?;
                self.mdat_at = Some(self.cursor);
                // An `mdat` with a 64-bit size that finish fills in.
                self.write_bytes(&[0, 0, 0, 1, b'm', b'd', b'a', b't'])?;
                self.write_bytes(&[0, 0, 0, 0, 0, 0, 0, 0])?;
            }
        }
        self.started = true;
        Ok(())
    }

    /// Writes one sample. `write_packet` of the mux trait does the same
    /// from an owned packet.
    pub(crate) fn write_sample(&mut self, sample: &SampleIn<'_>) -> Result<(), Error> {
        if self.finished {
            return Err(Error::Incomplete);
        }
        let Some(track) = self
            .tracks
            .iter()
            .position(|track| track.params.track_id == sample.track_id)
        else {
            return Err(Error::Missing { what: "track" });
        };
        // An `av01` sample is the temporal unit without its delimiter,
        // padding and redundant headers, each OBU with its size (AV1 Codec
        // ISO Media File Format Binding 2.4).
        let unit;
        let payload = if self.tracks[track].params.sample_entry == *b"av01" {
            unit = cutlass_container::av1_sample(sample.payload).ok_or(Error::Unsupported {
                what: "AV1 packet that is not whole OBUs",
            })?;
            unit.as_slice()
        } else {
            sample.payload
        };
        if sample.timescale != self.tracks[track].params.timescale.get() {
            return Err(Error::Unsupported {
                what: "packet timescale",
            });
        }
        let size = u32::try_from(payload.len()).map_err(|_| Error::Limit {
            what: "packet",
            value: u64::try_from(payload.len()).unwrap_or(u64::MAX),
        })?;
        if u64::from(size) > MAX_PACKET_BYTES {
            return Err(Error::Limit {
                what: "packet",
                value: u64::from(size),
            });
        }
        if sample.dts != self.tracks[track].next_dts {
            return Err(Error::Unsupported {
                what: "decode time must start at 0 and follow durations",
            });
        }
        let next_dts = sample
            .dts
            .checked_add(i64::from(sample.duration))
            .ok_or(Error::Limit {
                what: "decode time",
                value: sample.dts.unsigned_abs(),
            })?;
        let offset = sample.pts.checked_sub(sample.dts).ok_or(Error::Limit {
            what: "composition offset",
            value: 0,
        })?;
        let cts = i32::try_from(offset).map_err(|_| Error::Limit {
            what: "composition offset",
            value: offset.unsigned_abs(),
        })?;
        let mut entry = Sample {
            offset: 0,
            duration: sample.duration,
            size,
            cts,
            keyframe: sample.keyframe,
        };
        match self.layout()? {
            Layout::Progressive => {
                self.start()?;
                entry.offset = self.cursor;
                self.write_bytes(payload)?;
                self.tracks[track].samples.push(entry);
            }
            Layout::Buffered => {
                let next = self
                    .buffered
                    .len()
                    .checked_add(payload.len())
                    .ok_or(Error::Limit {
                        what: "buffered mux",
                        value: MAX_BUFFERED_MUX_BYTES,
                    })?;
                if u64::try_from(next).unwrap_or(u64::MAX) > MAX_BUFFERED_MUX_BYTES {
                    return Err(Error::Limit {
                        what: "buffered mux",
                        value: u64::try_from(next).unwrap_or(u64::MAX),
                    });
                }
                entry.offset = u64::try_from(self.buffered.len()).unwrap_or(0);
                self.buffered.extend_from_slice(payload);
                self.tracks[track].samples.push(entry);
            }
            Layout::Fragmented | Layout::Recoverable => {
                if self.cut_before(track, &entry) {
                    self.flush_fragment()?;
                }
                let held = &mut self.tracks[track];
                if held.pending.is_empty() {
                    held.pending_dts = sample.dts;
                }
                held.pending.push(FragSample {
                    duration: entry.duration,
                    size,
                    cts,
                    keyframe: entry.keyframe,
                });
                let at = self.fragment.len();
                match held.runs.last_mut() {
                    Some(run) if run.at + run.len == at => run.len += payload.len(),
                    _ => held.runs.push(Run {
                        first: held.pending.len() - 1,
                        at,
                        len: payload.len(),
                    }),
                }
                held.pending_ticks = held.pending_ticks.saturating_add(u64::from(entry.duration));
                self.fragment.extend_from_slice(payload);
                self.pending_samples += 1;
            }
        }
        self.tracks[track].next_dts = next_dts;
        Ok(())
    }

    /// Whether the current fragment ends before this sample of `track`.
    fn cut_before(&self, track: usize, sample: &Sample) -> bool {
        if self.pending_samples == 0 {
            return false;
        }
        let cap = u64::from(self.options.samples_per_fragment);
        if cap > 0 && self.pending_samples >= cap {
            return true;
        }
        let waiting = u64::try_from(self.fragment.len()).unwrap_or(u64::MAX);
        if waiting.saturating_add(u64::from(sample.size)) > self.options.fragment_bytes {
            return true;
        }
        let held = &self.tracks[track];
        let scale = u64::from(held.params.timescale.get());
        // Compared as ticks * 1000 against milliseconds * timescale.
        let waited = u128::from(held.pending_ticks) * 1000;
        let target = u128::from(self.options.fragment_millis) * u128::from(scale);
        let lengths = if self.options.fragmented {
            FORCED_CUT_LENGTHS
        } else {
            FORCED_CUT_LENGTHS_RECOVERABLE
        };
        if waited >= target * u128::from(lengths) {
            return true;
        }
        // The first fragment carries the sample descriptions, so it waits
        // for every track's first sample, which may bring its configuration.
        if !self.started && self.tracks.iter().any(|other| other.next_dts == 0) {
            return false;
        }
        let video = self.tracks.iter().position(TrackOut::is_video);
        match video {
            Some(video) => track == video && sample.keyframe && waited >= target,
            None => waited >= target,
        }
    }

    /// Writes the pending samples as one `moof` and `mdat`.
    ///
    /// Fragmented output keeps the payload in the order it arrived, with a
    /// `trun` for each run of a track's samples. A recoverable file writes
    /// each track's payload together under one `trun`, so the movie made of
    /// it at finish has one chunk for a track in a fragment.
    fn flush_fragment(&mut self) -> Result<(), Error> {
        if self.pending_samples == 0 {
            return Ok(());
        }
        self.start()?;
        let fragmented = self.options.fragmented;
        let grouped = !fragmented;
        let draft = moof_bytes(&self.tracks, self.sequence, 0, grouped)?;
        let moof_len = u64::try_from(draft.len()).unwrap_or(u64::MAX);
        let payload_len = u64::try_from(self.fragment.len()).unwrap_or(u64::MAX);
        let mut mdat_header = Vec::with_capacity(16);
        if payload_len.saturating_add(8) > u64::from(u32::MAX) {
            mdat_header.extend_from_slice(&[0, 0, 0, 1, b'm', b'd', b'a', b't']);
            mdat_header.extend_from_slice(&payload_len.saturating_add(16).to_be_bytes());
        } else {
            let size = u32::try_from(payload_len + 8).unwrap_or(u32::MAX);
            mdat_header.extend_from_slice(&size.to_be_bytes());
            mdat_header.extend_from_slice(b"mdat");
        }
        let header_len = u64::try_from(mdat_header.len()).unwrap_or(16);
        let data_start = moof_len.saturating_add(header_len);
        let moof = moof_bytes(&self.tracks, self.sequence, data_start, grouped)?;
        let sidx = if fragmented {
            sidx_bytes(&self.tracks, data_start.saturating_add(payload_len))
        } else {
            Vec::new()
        };
        // Nothing is on the output yet, so an error above leaves the
        // pending samples as they were.
        self.write_bytes(&sidx)?;
        let moof_at = self.cursor;
        self.write_bytes(&moof)?;
        self.write_bytes(&mdat_header)?;
        let payload = std::mem::take(&mut self.fragment);
        let written = self.write_payload(&payload, moof_at, grouped);
        self.fragment = payload;
        self.fragment.clear();
        for track in &mut self.tracks {
            track.pending.clear();
            track.runs.clear();
            track.pending_ticks = 0;
        }
        self.pending_samples = 0;
        written?;
        self.sequence = self.sequence.saturating_add(1);
        self.flush_output()
    }

    /// Writes a fragment's payload after its `mdat` header, and records
    /// where it went: the sample table of a recoverable file, the random
    /// access rows of a fragmented one.
    fn write_payload(&mut self, payload: &[u8], moof_at: u64, grouped: bool) -> Result<(), Error> {
        if !grouped {
            self.write_bytes(payload)?;
        }
        let mut traf_number = 0u8;
        for index in 0..self.tracks.len() {
            if self.tracks[index].pending.is_empty() {
                continue;
            }
            traf_number = traf_number.saturating_add(1);
            if grouped {
                let mut at = self.cursor;
                for run in 0..self.tracks[index].runs.len() {
                    let run = self.tracks[index].runs[run];
                    let bytes = payload
                        .get(run.at..run.at + run.len)
                        .ok_or(Error::Incomplete)?;
                    self.write_bytes(bytes)?;
                }
                let track = &mut self.tracks[index];
                track.samples.reserve(track.pending.len());
                for sample in &track.pending {
                    track.samples.push(Sample {
                        offset: at,
                        duration: sample.duration,
                        size: sample.size,
                        cts: sample.cts,
                        keyframe: sample.keyframe,
                    });
                    at += u64::from(sample.size);
                }
                continue;
            }
            // The first sync sample of the track in this fragment.
            let track = &mut self.tracks[index];
            let mut dts = track.pending_dts;
            for (number, sample) in track.pending.iter().enumerate() {
                if sample.keyframe {
                    let run = track
                        .runs
                        .iter()
                        .rposition(|run| run.first <= number)
                        .unwrap_or(0);
                    let pts = dts.saturating_add(i64::from(sample.cts));
                    track.tfra.push(TfraRow {
                        time: u64::try_from(pts).unwrap_or(0),
                        moof_offset: moof_at,
                        traf_number,
                        trun_number: u32::try_from(run + 1).unwrap_or(u32::MAX),
                        sample_number: u32::try_from(number - track.runs[run].first + 1)
                            .unwrap_or(u32::MAX),
                    });
                    break;
                }
                dts = dts.saturating_add(i64::from(sample.duration));
            }
        }
        Ok(())
    }

    /// Writes `moov` for a file whose payload is on the output after the
    /// sixteen bytes at `mdat_at`, and makes those bytes the `mdat` header.
    fn finish_movie(&mut self) -> Result<(), Error> {
        let Some(mdat_at) = self.mdat_at else {
            return Err(Error::Incomplete);
        };
        let scale = self.movie_scale()?;
        let times = header_time(&self.options);
        let mut header = vec![0, 0, 0, 1, b'm', b'd', b'a', b't'];
        header.extend_from_slice(&self.cursor.saturating_sub(mdat_at).to_be_bytes());
        if self.options.moov_at_end {
            let moov = movie_bytes(&self.tracks, scale, self.cover.as_deref(), 0, times)?;
            // `moov` goes down before the header changes, so a recoverable
            // file is a readable fragmented one until the last write.
            self.write_bytes(&moov)?;
            self.flush_output()?;
            return self.patch(mdat_at, &header);
        }
        // The size of `moov` depends on its place only through the width of
        // the chunk offsets, which can only grow.
        let mut shift = 0u64;
        let mut moov = movie_bytes(&self.tracks, scale, self.cover.as_deref(), shift, times)?;
        for _ in 0..=self.tracks.len() {
            let len = u64::try_from(moov.len()).unwrap_or(u64::MAX);
            if len == shift {
                break;
            }
            shift = len;
            moov = movie_bytes(&self.tracks, scale, self.cover.as_deref(), shift, times)?;
        }
        if u64::try_from(moov.len()).unwrap_or(u64::MAX) != shift {
            return Err(Error::Unsupported {
                what: "moov size changed while placing it first",
            });
        }
        self.patch(mdat_at, &header)?;
        self.shift_later(mdat_at, shift)?;
        self.patch(mdat_at, &moov)?;
        self.flush_output()
    }
}

impl<W: Write + Seek> Mux for IsoBmffMux<W> {
    fn add_stream(&mut self, params: &StreamParams) -> Result<u32, Error> {
        if self.finished || self.started {
            return Err(Error::Incomplete);
        }
        if self.tracks.len() >= MAX_TRACKS {
            return Err(Error::Limit {
                what: "tracks",
                value: u64::try_from(self.tracks.len()).unwrap_or(u64::MAX),
            });
        }
        if params.timescale.get() == 0 {
            return Err(Error::Unsupported {
                what: "media timescale 0",
            });
        }
        let sample_entry = entry_code(params)?;
        let mut stored = params.clone();
        if stored.track_id == 0 {
            stored.track_id = u32::try_from(self.tracks.len() + 1).unwrap_or(1);
        }
        if self
            .tracks
            .iter()
            .any(|track| track.params.track_id == stored.track_id)
        {
            return Err(Error::Unsupported {
                what: "duplicate track id",
            });
        }
        stored.sample_entry = sample_entry;
        let id = stored.track_id;
        self.tracks.push(TrackOut {
            params: stored,
            edits: Vec::new(),
            next_dts: 0,
            gapless: None,
            samples: Vec::new(),
            pending: Vec::new(),
            runs: Vec::new(),
            pending_dts: 0,
            pending_ticks: 0,
            tfra: Vec::new(),
        });
        Ok(id)
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<(), Error> {
        self.write_sample(&SampleIn {
            track_id: packet.track_id,
            payload: &packet.payload,
            dts: packet.dts,
            pts: packet.pts,
            duration: packet.duration,
            timescale: packet.timescale.get(),
            keyframe: packet.keyframe,
        })
    }

    fn finish(&mut self) -> Result<(), Error> {
        if self.finished {
            return Ok(());
        }
        match self.layout()? {
            Layout::Fragmented => {
                self.start()?;
                self.flush_fragment()?;
                let mfra = mfra_bytes(&self.tracks);
                self.write_bytes(&mfra)?;
            }
            Layout::Recoverable => {
                self.start()?;
                self.flush_fragment()?;
                self.finish_movie()?;
            }
            Layout::Progressive => {
                self.start()?;
                self.finish_movie()?;
            }
            Layout::Buffered => {
                let ftyp = ftyp_bytes(self.options.brand, Layout::Buffered, &self.tracks);
                let scale = self.movie_scale()?;
                let times = header_time(&self.options);
                let cover = self.cover.as_deref();
                // The payload follows `ftyp`, `moov` and the `mdat` header.
                let before = u64::try_from(ftyp.len()).unwrap_or(0) + 16;
                let mut shift = before;
                let mut moov = movie_bytes(&self.tracks, scale, cover, shift, times)?;
                for _ in 0..=self.tracks.len() {
                    let place = before + u64::try_from(moov.len()).unwrap_or(u64::MAX);
                    if place == shift {
                        break;
                    }
                    shift = place;
                    moov = movie_bytes(&self.tracks, scale, cover, shift, times)?;
                }
                if before + u64::try_from(moov.len()).unwrap_or(u64::MAX) != shift {
                    return Err(Error::Unsupported {
                        what: "moov size changed while placing it first",
                    });
                }
                self.write_bytes(&ftyp)?;
                self.write_bytes(&moov)?;
                let mut header = vec![0, 0, 0, 1, b'm', b'd', b'a', b't'];
                let mdat_size =
                    16u64.saturating_add(u64::try_from(self.buffered.len()).unwrap_or(0));
                header.extend_from_slice(&mdat_size.to_be_bytes());
                self.write_bytes(&header)?;
                let buffered = std::mem::take(&mut self.buffered);
                self.write_bytes(&buffered)?;
            }
        }
        self.finished = true;
        Ok(())
    }
}

/// The creation and modification time for the headers, in seconds since
/// 1904-01-01.
fn header_time(options: &MuxOptions) -> u64 {
    options
        .creation_time
        .map_or(0, |unix| unix.saturating_add(EPOCH_1904))
}

fn entry_code(params: &StreamParams) -> Result<[u8; 4], Error> {
    if params.sample_entry != [0; 4] {
        return Ok(params.sample_entry);
    }
    let code = match params.codec {
        CodecId::H264 => *b"avc1",
        CodecId::Hevc => *b"hvc1",
        CodecId::Av1 => *b"av01",
        CodecId::Vp8 => *b"vp08",
        CodecId::Vp9 => *b"vp09",
        CodecId::Mpeg4 | CodecId::Mpeg2 => *b"mp4v",
        CodecId::Mjpeg => *b"jpeg",
        CodecId::ProRes => *b"apcn",
        CodecId::Aac | CodecId::Mp3 => *b"mp4a",
        CodecId::Opus => *b"Opus",
        CodecId::Flac => *b"fLaC",
        CodecId::Alac => *b"alac",
        CodecId::Ac3 => *b"ac-3",
        CodecId::Eac3 => *b"ec-3",
        CodecId::Pcm => pcm_entry_code(params)?,
        CodecId::Unknown(code) => code,
        _ => {
            return Err(Error::CodecRejected {
                sample_entry: [0, 0, 0, 0],
            });
        }
    };
    Ok(code)
}

/// The QuickTime entry for a PCM track. Mono or stereo 8- and 16-bit sound
/// at up to 65535 Hz keeps the classic version 0 codes FFmpeg also writes:
/// `raw ` (unsigned 8-bit), `twos` (signed 8-bit, s16be) and `sowt`
/// (s16le). Anything else is `lpcm` with a version 2 description, whose
/// flags state float, byte order and sign exactly.
fn pcm_entry_code(params: &StreamParams) -> Result<[u8; 4], Error> {
    let MediaKind::Audio(audio) = params.kind else {
        return Err(Error::CodecRejected {
            sample_entry: [0, 0, 0, 0],
        });
    };
    if lpcm_flags(audio.format).is_none() {
        return Err(Error::Unsupported {
            what: "PCM track without a sample format",
        });
    }
    let classic = audio.channels <= 2 && audio.sample_rate <= u32::from(u16::MAX);
    Ok(match audio.format {
        SampleFormat::U8 if classic => *b"raw ",
        SampleFormat::S8 | SampleFormat::S16Be if classic => *b"twos",
        SampleFormat::S16Le if classic => *b"sowt",
        _ => *b"lpcm",
    })
}

/// `formatSpecificFlags` of a version 2 `lpcm` description, as Apple's
/// `AudioStreamBasicDescription` defines them: 1 float, 2 big-endian,
/// 4 signed integer, 8 packed. The values match FFmpeg's movenc.c.
fn lpcm_flags(format: SampleFormat) -> Option<u32> {
    Some(match format {
        SampleFormat::F32Be | SampleFormat::F64Be => 1 | 2 | 8,
        SampleFormat::F32Le | SampleFormat::F64Le => 1 | 8,
        SampleFormat::U8 => 2 | 8,
        SampleFormat::S16Be | SampleFormat::S24Be | SampleFormat::S32Be => 2 | 4 | 8,
        SampleFormat::S8 | SampleFormat::S16Le | SampleFormat::S24Le | SampleFormat::S32Le => 4 | 8,
        SampleFormat::Unknown => return None,
    })
}

/// A QuickTime sound description version 2 (QTFF, "Sound Sample Description
/// (Version 2)"): the fields after the eight-byte sample entry header.
fn lpcm_description(audio: &crate::legacy::AudioParams) -> Result<Vec<u8>, Error> {
    let flags = lpcm_flags(audio.format).ok_or(Error::Unsupported {
        what: "PCM track without a sample format",
    })?;
    let bits = u32::from(audio.bits_per_sample);
    let channels = u32::from(audio.channels);
    let mut entry = Vec::with_capacity(56);
    entry.extend_from_slice(&2u16.to_be_bytes()); // version
    entry.extend_from_slice(&[0, 0, 0, 0, 0, 0]); // revision, vendor
    entry.extend_from_slice(&3u16.to_be_bytes()); // always 3
    entry.extend_from_slice(&16u16.to_be_bytes()); // always 16
    entry.extend_from_slice(&(-2i16).to_be_bytes()); // always -2
    entry.extend_from_slice(&0u16.to_be_bytes()); // always 0
    entry.extend_from_slice(&0x0001_0000u32.to_be_bytes()); // always 65536
    entry.extend_from_slice(&72u32.to_be_bytes()); // size of the struct
    entry.extend_from_slice(&f64::from(audio.sample_rate).to_bits().to_be_bytes());
    entry.extend_from_slice(&channels.to_be_bytes());
    entry.extend_from_slice(&0x7F00_0000u32.to_be_bytes()); // always 0x7F000000
    entry.extend_from_slice(&bits.to_be_bytes());
    entry.extend_from_slice(&flags.to_be_bytes());
    entry.extend_from_slice(&(channels * bits / 8).to_be_bytes()); // bytes per packet
    entry.extend_from_slice(&1u32.to_be_bytes()); // frames per packet
    Ok(entry)
}

/// The file type box. A file with an AV1 track that is not a QuickTime
/// movie lists `av01` among its compatible brands, as the AV1 binding
/// requires (2.1).
fn ftyp_bytes(brand: FileBrand, layout: Layout, tracks: &[TrackOut]) -> Vec<u8> {
    let av1 = brand != FileBrand::Mov
        && tracks
            .iter()
            .any(|track| track.params.sample_entry == *b"av01");
    let (major, compatible): ([u8; 4], &[&[u8; 4]]) = if layout == Layout::Fragmented {
        (*b"iso5", &[b"iso5", b"iso6", b"mp41"])
    } else {
        match brand {
            FileBrand::Mp4 => (*b"isom", &[b"isom", b"iso2", b"mp41"]),
            FileBrand::Mov => (*b"qt  ", &[b"qt  "]),
            FileBrand::M4a => (*b"M4A ", &[b"M4A ", b"mp42", b"isom"]),
            FileBrand::M4v => (*b"M4V ", &[b"M4V ", b"mp42", b"isom"]),
            FileBrand::ThreeGp => (*b"3gp4", &[b"3gp4", b"isom"]),
        }
    };
    // Until finish a recoverable file is read through its fragments, whose
    // offsets count from their `moof`: that is brand `iso5`.
    let fragments = layout == Layout::Recoverable && brand != FileBrand::Mov;
    let mut payload = Vec::new();
    payload.extend_from_slice(&major);
    payload.extend_from_slice(&0u32.to_be_bytes());
    for brand in compatible {
        payload.extend_from_slice(*brand);
    }
    if fragments {
        payload.extend_from_slice(b"iso5");
    }
    if av1 {
        payload.extend_from_slice(b"av01");
    }
    box_bytes(b"ftyp", &payload)
}

fn movie_bytes(
    tracks: &[TrackOut],
    movie_scale: u32,
    cover: Option<&[u8]>,
    shift: u64,
    times: u64,
) -> Result<Vec<u8>, Error> {
    let mut body = Vec::new();
    let mut duration = 0u64;
    let mut traks = Vec::new();
    for track in tracks {
        let (bytes, presented) = trak(track, &track.samples, movie_scale, shift, times)?;
        duration = duration.max(presented);
        traks.extend(bytes);
    }
    body.extend(mvhd(movie_scale, duration, tracks.len(), times));
    body.extend(traks);
    if let Some(jpeg) = cover {
        body.extend(cover_udta(jpeg));
    }
    sized_box(b"moov", &body)
}

fn init_moov(
    tracks: &[TrackOut],
    movie_scale: u32,
    cover: Option<&[u8]>,
    times: u64,
) -> Result<Vec<u8>, Error> {
    let mut body = Vec::new();
    body.extend(mvhd(movie_scale, 0, tracks.len(), times));
    for track in tracks {
        body.extend(trak(track, &[], movie_scale, 0, times)?.0);
    }
    let mut mvex = Vec::new();
    for track in tracks {
        mvex.extend(trex(track));
    }
    body.extend(box_bytes(b"mvex", &mvex));
    if let Some(jpeg) = cover {
        body.extend(cover_udta(jpeg));
    }
    sized_box(b"moov", &body)
}

/// The `data` atom type of a JPEG picture, from Apple's well-known types.
const JPEG_DATA_TYPE: u32 = 13;

/// `udta` holding iTunes metadata with one cover picture, laid out as
/// FFmpeg's movenc.c writes it: `meta` is a full box whose `hdlr` names
/// handler `mdir` and manufacturer `appl`, and `ilst/covr/data` holds the
/// type, a zero locale, then the picture.
fn cover_udta(jpeg: &[u8]) -> Vec<u8> {
    let mut hdlr = Vec::with_capacity(25);
    hdlr.extend_from_slice(&[0; 4]); // version, flags
    hdlr.extend_from_slice(&[0; 4]); // pre_defined
    hdlr.extend_from_slice(b"mdir");
    hdlr.extend_from_slice(b"appl");
    hdlr.extend_from_slice(&[0; 8]); // reserved
    hdlr.push(0); // empty name
    let mut data = Vec::with_capacity(jpeg.len() + 8);
    data.extend_from_slice(&JPEG_DATA_TYPE.to_be_bytes());
    data.extend_from_slice(&[0; 4]); // locale
    data.extend_from_slice(jpeg);
    let covr = box_bytes(b"covr", &box_bytes(b"data", &data));
    let mut meta = vec![0; 4]; // version, flags
    meta.extend(box_bytes(b"hdlr", &hdlr));
    meta.extend(box_bytes(b"ilst", &covr));
    box_bytes(b"udta", &box_bytes(b"meta", &meta))
}

impl<W: Write + Seek> IsoBmffMux<W> {
    fn movie_scale(&self) -> Result<u32, Error> {
        if let Some(scale) = self.options.movie_timescale {
            if scale == 0 {
                return Err(Error::Unsupported {
                    what: "movie timescale 0",
                });
            }
            return Ok(scale);
        }
        Ok(movie_timescale(&self.tracks))
    }
}

/// The least common multiple of the track timescales when it is at most
/// [`MAX_MOVIE_TIMESCALE`], so every track duration is a whole number of
/// movie ticks. Otherwise the first video track's timescale, and 1000 when
/// that is too large as well or there is no video.
fn movie_timescale(tracks: &[TrackOut]) -> u32 {
    let bound = u64::from(MAX_MOVIE_TIMESCALE);
    let mut scale = 1u64;
    for track in tracks {
        let right = u64::from(track.params.timescale.get());
        if scale <= bound && right != 0 {
            scale = scale / gcd(scale, right) * right;
        }
    }
    if !tracks.is_empty()
        && scale <= bound
        && let Ok(scale) = u32::try_from(scale)
    {
        return scale;
    }
    tracks
        .iter()
        .find(|track| track.is_video())
        .map(|track| track.params.timescale.get())
        .filter(|scale| *scale <= MAX_MOVIE_TIMESCALE)
        .unwrap_or(1000)
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let next = left % right;
        left = right;
        right = next;
    }
    left
}

/// `ticks` of timescale `from` in timescale `to`, rounded up so a duration
/// never ends before its last sample does.
fn scale_u(ticks: u64, from: u32, to: u32) -> u64 {
    if from == 0 {
        return 0;
    }
    let value = (u128::from(ticks) * u128::from(to)).div_ceil(u128::from(from));
    u64::try_from(value).unwrap_or(u64::MAX)
}

/// One `trak`, and the time it presents in the movie timescale: the sum of
/// its edits, or its media duration when it has none.
fn trak(
    track: &TrackOut,
    samples: &[Sample],
    movie_scale: u32,
    shift: u64,
    times: u64,
) -> Result<(Vec<u8>, u64), Error> {
    let media_duration = samples.iter().fold(0u64, |sum, sample| {
        sum.saturating_add(u64::from(sample.duration))
    });
    let edits = present_edits(track, media_duration, movie_scale);
    let end_cut = end_cut(track, samples, media_duration);
    let media_duration = media_duration - u64::from(end_cut);
    let presented = if edits.is_empty() {
        scale_u(media_duration, track.params.timescale.get(), movie_scale)
    } else {
        edits
            .iter()
            .fold(0u64, |sum, edit| sum.saturating_add(edit.segment_duration))
    };
    let mut body = tkhd(track, presented, times);
    if !edits.is_empty() {
        body.extend(elst(&edits));
    }
    let mut mdia = Vec::new();
    mdia.extend(mdhd(track, media_duration, times));
    mdia.extend(hdlr(track));
    mdia.extend(minf(track, samples, shift, end_cut)?);
    body.extend(sized_box(b"mdia", &mdia)?);
    Ok((sized_box(b"trak", &body)?, presented))
}

/// The version, creation time, modification time, timescale when the box
/// has one, and duration that open `mvhd`, `tkhd` and `mdhd`. Version 1
/// holds each time in 64 bits; it is written only when one needs it.
fn timed_header(flags: u32, times: u64, middle: &[u8], duration: u64) -> Vec<u8> {
    let wide = duration > u64::from(u32::MAX) || times > u64::from(u32::MAX);
    let mut payload = (u32::from(wide) << 24 | flags).to_be_bytes().to_vec();
    if wide {
        payload.extend_from_slice(&times.to_be_bytes());
        payload.extend_from_slice(&times.to_be_bytes());
        payload.extend_from_slice(middle);
        payload.extend_from_slice(&duration.to_be_bytes());
    } else {
        let times = u32::try_from(times).unwrap_or(0);
        payload.extend_from_slice(&times.to_be_bytes());
        payload.extend_from_slice(&times.to_be_bytes());
        payload.extend_from_slice(middle);
        payload.extend_from_slice(&u32::try_from(duration).unwrap_or(u32::MAX).to_be_bytes());
    }
    payload
}

fn tkhd(track: &TrackOut, duration: u64, times: u64) -> Vec<u8> {
    let mut middle = track.params.track_id.to_be_bytes().to_vec();
    middle.extend_from_slice(&[0, 0, 0, 0]);
    let mut payload = timed_header(3, times, &middle, duration);
    payload.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]);
    payload.extend_from_slice(&[0, 0]);
    payload.extend_from_slice(&[0, 0]);
    let volume: u16 = if matches!(track.params.kind, MediaKind::Audio(_)) {
        0x0100
    } else {
        0
    };
    payload.extend_from_slice(&volume.to_be_bytes());
    payload.extend_from_slice(&[0, 0]);
    payload.extend_from_slice(&matrix());
    let (width, height) = match track.params.kind {
        MediaKind::Video(video) => (video.width, video.height),
        _ => (0, 0),
    };
    payload.extend_from_slice(&fixed_16_16(width).to_be_bytes());
    payload.extend_from_slice(&fixed_16_16(height).to_be_bytes());
    box_bytes(b"tkhd", &payload)
}

/// A zero segment duration with a media time does not describe a dwell.
/// Fragmented files leave that duration unknown. Use the media length so a
/// player does not treat the edit as empty.
///
/// An Opus track without edits gets the one its encapsulation asks for:
/// play from the pre-skip to the end of the sample durations, whose last
/// one is already cut to the end trim.
///
/// A track with gapless trim and no edits of its own plays from the delay
/// through the last sample that is not padding. That is the AAC, AC-3 and
/// E-AC-3 case: the frame duration still holds both.
fn present_edits(track: &TrackOut, media_duration: u64, movie_scale: u32) -> Vec<Edit> {
    if track.edits.is_empty()
        && let Some((delay, padding)) = track.gapless
        && (delay > 0 || padding > 0)
        && let Ok(media_time) = i64::try_from(delay)
        && media_duration > delay.saturating_add(padding)
    {
        let played = media_duration - delay - padding;
        return vec![Edit {
            segment_duration: scale_u(played, track.params.timescale.get(), movie_scale).max(1),
            media_time: Some(media_time),
            media_rate: 0x0001_0000,
        }];
    }
    if track.edits.is_empty()
        && track.params.codec == CodecId::Opus
        && let Some(pre_skip) = crate::opus::pre_skip(&track.params.extradata)
        && media_duration > u64::from(pre_skip)
    {
        let played = media_duration - u64::from(pre_skip);
        return vec![Edit {
            segment_duration: scale_u(played, track.params.timescale.get(), movie_scale).max(1),
            media_time: Some(i64::from(pre_skip)),
            media_rate: 0x0001_0000,
        }];
    }
    track
        .edits
        .iter()
        .map(|edit| {
            if edit.segment_duration == 0 && edit.media_time.is_some() && media_duration > 0 {
                Edit {
                    segment_duration: scale_u(
                        media_duration,
                        track.params.timescale.get(),
                        movie_scale,
                    )
                    .max(1),
                    media_time: edit.media_time,
                    media_rate: edit.media_rate,
                }
            } else {
                *edit
            }
        })
        .collect()
}

/// Version 1, with 64-bit durations and media times, when an edit needs it.
/// The ticks taken off the last sample's duration: the trailing padding of
/// a track whose gapless trim becomes its edit, when that padding ends
/// inside the last sample. The edit says where play stops, and QuickTime
/// follows it; FFmpeg and the players built on it stop a sound track where
/// its last sample's duration does, as FFmpeg's own muxer writes it.
fn end_cut(track: &TrackOut, samples: &[Sample], media_duration: u64) -> u32 {
    let Some((delay, padding)) = track.gapless else {
        return 0;
    };
    if !track.edits.is_empty() || media_duration <= delay.saturating_add(padding) {
        return 0;
    }
    match (samples.last(), u32::try_from(padding)) {
        (Some(last), Ok(padding)) if padding < last.duration => padding,
        _ => 0,
    }
}

fn elst(edits: &[Edit]) -> Vec<u8> {
    let wide = edits.iter().any(|edit| {
        u32::try_from(edit.segment_duration).is_err()
            || i32::try_from(edit.media_time.unwrap_or(-1)).is_err()
    });
    let mut payload = vec![u8::from(wide), 0, 0, 0];
    payload.extend_from_slice(&u32::try_from(edits.len()).unwrap_or(0).to_be_bytes());
    for edit in edits {
        let media = edit.media_time.unwrap_or(-1);
        if wide {
            payload.extend_from_slice(&edit.segment_duration.to_be_bytes());
            payload.extend_from_slice(&media.to_be_bytes());
        } else {
            let duration = u32::try_from(edit.segment_duration).unwrap_or(u32::MAX);
            payload.extend_from_slice(&duration.to_be_bytes());
            payload.extend_from_slice(&i32::try_from(media).unwrap_or(-1).to_be_bytes());
        }
        payload.extend_from_slice(&edit.media_rate.to_be_bytes());
    }
    box_bytes(b"edts", &box_bytes(b"elst", &payload))
}

fn mdhd(track: &TrackOut, duration: u64, times: u64) -> Vec<u8> {
    let scale = track.params.timescale.get().to_be_bytes();
    let mut payload = timed_header(0, times, &scale, duration);
    payload.extend_from_slice(&pack_language(track.params.language).to_be_bytes());
    payload.extend_from_slice(&[0, 0]);
    box_bytes(b"mdhd", &payload)
}

fn hdlr(track: &TrackOut) -> Vec<u8> {
    let (kind, name): ([u8; 4], &[u8]) = match track.params.kind {
        MediaKind::Video(_) => (*b"vide", b"Video\0"),
        MediaKind::Audio(_) => (*b"soun", b"Sound\0"),
        MediaKind::Data => (*b"meta", b"Data\0"),
    };
    let mut payload = vec![0, 0, 0, 0, 0, 0, 0, 0];
    payload.extend_from_slice(&kind);
    payload.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    payload.extend_from_slice(name);
    box_bytes(b"hdlr", &payload)
}

fn minf(track: &TrackOut, samples: &[Sample], shift: u64, end_cut: u32) -> Result<Vec<u8>, Error> {
    let mut body = Vec::new();
    body.extend(match track.params.kind {
        MediaKind::Video(_) => box_bytes(b"vmhd", &[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        MediaKind::Audio(_) => box_bytes(b"smhd", &[0, 0, 0, 0, 0, 0, 0, 0]),
        MediaKind::Data => box_bytes(b"nmhd", &[0, 0, 0, 0]),
    });
    let url = box_bytes(b"url ", &[0, 0, 0, 1]);
    let mut dref = vec![0, 0, 0, 0, 0, 0, 0, 1];
    dref.extend(url);
    body.extend(box_bytes(b"dinf", &box_bytes(b"dref", &dref)));
    body.extend(stbl(track, samples, shift, end_cut)?);
    sized_box(b"minf", &body)
}

fn stbl(track: &TrackOut, samples: &[Sample], shift: u64, end_cut: u32) -> Result<Vec<u8>, Error> {
    let mut body = Vec::new();
    body.extend(stsd(track)?);
    if let Some(frame) = pcm_frame_bytes(track) {
        body.extend(pcm_tables(samples, frame, shift)?);
        return sized_box(b"stbl", &body);
    }
    body.extend(stts(samples, end_cut)?);
    if samples.iter().any(|sample| sample.cts != 0) {
        body.extend(ctts(samples)?);
        if samples.iter().any(|sample| sample.cts < 0) {
            body.extend(cslg(samples));
        }
    }
    let chunks = chunks(samples, CHUNK_BYTES, |_| 1);
    body.extend(stsc(&chunks)?);
    body.extend(chunk_offsets(&chunks, shift)?);
    body.extend(stsz(samples)?);
    if samples.iter().any(|sample| !sample.keyframe) {
        body.extend(stss(samples)?);
    }
    sized_box(b"stbl", &body)
}

/// The bytes of one audio frame of a raw PCM track, or `None` for any
/// other track.
fn pcm_frame_bytes(track: &TrackOut) -> Option<u32> {
    if track.params.codec != CodecId::Pcm {
        return None;
    }
    let MediaKind::Audio(audio) = track.params.kind else {
        return None;
    };
    let bytes = u32::from(audio.channels) * u32::from(audio.bits_per_sample / 8);
    (bytes > 0).then_some(bytes)
}

/// Sample tables for raw PCM, as QuickTime reads them: each audio frame is
/// one sample of one tick, and each written packet is one chunk, so a
/// reader gets the packets back as they were written.
fn pcm_tables(samples: &[Sample], frame: u32, shift: u64) -> Result<Vec<u8>, Error> {
    let chunks = chunks(samples, 0, |sample| sample.size / frame);
    let total: u32 = chunks
        .iter()
        .fold(0, |sum, chunk| sum.saturating_add(chunk.samples));
    let mut stts = vec![0, 0, 0, 0];
    if total == 0 {
        stts.extend_from_slice(&0u32.to_be_bytes());
    } else {
        stts.extend_from_slice(&1u32.to_be_bytes());
        stts.extend_from_slice(&total.to_be_bytes());
        stts.extend_from_slice(&1u32.to_be_bytes());
    }
    let mut stsz = vec![0, 0, 0, 0];
    stsz.extend_from_slice(&frame.to_be_bytes());
    stsz.extend_from_slice(&total.to_be_bytes());
    let mut out = box_bytes(b"stts", &stts);
    out.extend(stsc(&chunks)?);
    out.extend(chunk_offsets(&chunks, shift)?);
    out.extend(box_bytes(b"stsz", &stsz));
    Ok(out)
}

fn stsd(track: &TrackOut) -> Result<Vec<u8>, Error> {
    let mut entry = Vec::new();
    let code = track.params.sample_entry;
    match track.params.kind {
        MediaKind::Video(video) => {
            entry.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 1]);
            entry.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            entry.extend_from_slice(&u16::try_from(video.width).unwrap_or(u16::MAX).to_be_bytes());
            entry.extend_from_slice(
                &u16::try_from(video.height)
                    .unwrap_or(u16::MAX)
                    .to_be_bytes(),
            );
            entry.extend_from_slice(&0x00480000u32.to_be_bytes());
            entry.extend_from_slice(&0x00480000u32.to_be_bytes());
            entry.extend_from_slice(&[0, 0, 0, 0, 0, 1]);
            entry.extend_from_slice(&[0; 32]);
            entry.extend_from_slice(&0x0018u16.to_be_bytes());
            entry.extend_from_slice(&(-1i16).to_be_bytes());
            entry.extend(config_boxes(track)?);
        }
        MediaKind::Audio(mut audio) => {
            if &code == b"alac"
                && let Some(facts) = alac::facts(&track.params.extradata)
            {
                audio.channels = facts.channels;
                audio.bits_per_sample = facts.bits;
            }
            entry.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 1]);
            if &code == b"lpcm" {
                entry.extend(lpcm_description(&audio)?);
                entry.extend(config_boxes(track)?);
                let mut payload = vec![0, 0, 0, 0, 0, 0, 0, 1];
                payload.extend(box_bytes(&code, &entry));
                return Ok(box_bytes(b"stsd", &payload));
            }
            entry.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]);
            entry.extend_from_slice(&audio.channels.to_be_bytes());
            entry.extend_from_slice(&audio.bits_per_sample.to_be_bytes());
            entry.extend_from_slice(&[0, 0, 0, 0]);
            let rate = u32::from(u16::try_from(audio.sample_rate).unwrap_or(u16::MAX)) << 16;
            entry.extend_from_slice(&rate.to_be_bytes());
            entry.extend(config_boxes(track)?);
        }
        MediaKind::Data => {
            entry.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 1]);
            entry.extend(&track.params.extradata);
        }
    }
    let mut payload = vec![0, 0, 0, 0, 0, 0, 0, 1];
    payload.extend(box_bytes(&code, &entry));
    Ok(box_bytes(b"stsd", &payload))
}

fn config_boxes(track: &TrackOut) -> Result<Vec<u8>, Error> {
    let mut out = Vec::new();
    let code = track.params.sample_entry;
    if matches!(&code, b"avc1" | b"avc3") {
        out.extend(box_bytes(b"avcC", &track.params.extradata));
    } else if matches!(&code, b"hvc1" | b"hev1" | b"dvh1" | b"dvhe") {
        out.extend(box_bytes(b"hvcC", &track.params.extradata));
    } else if &code == b"av01" {
        out.extend(box_bytes(b"av1C", &track.params.extradata));
    } else if matches!(&code, b"vp08" | b"vp09") {
        out.extend(box_bytes(b"vpcC", &track.params.extradata));
    } else if &code == b"Opus" {
        let dops = crate::opus::dops(&track.params.extradata).ok_or(Error::Unsupported {
            what: "Opus configuration that is neither dOps nor OpusHead",
        })?;
        out.extend(box_bytes(b"dOps", &dops));
    } else if &code == b"alac" {
        // Extradata is the bare cookie; the entry holds its config in an
        // `alac` box and its layout in a sibling `chan` box, as Apple writes.
        let (config, chan) = alac::split(&track.params.extradata).ok_or(Error::Missing {
            what: "alac cookie",
        })?;
        out.extend(box_bytes(b"alac", &config));
        if let Some(chan) = chan {
            out.extend(box_bytes(b"chan", &chan));
        }
    } else if matches!(&code, b"ac-3" | b"sac3") {
        out.extend(box_bytes(b"dac3", &track.params.extradata));
    } else if &code == b"ec-3" {
        out.extend(box_bytes(b"dec3", &track.params.extradata));
    } else if &code == b"fLaC" {
        out.extend(box_bytes(b"dfLa", &track.params.extradata));
    } else if matches!(&code, b"mp4a" | b"mp4v")
        && (!track.params.extradata.is_empty() || track.params.codec == CodecId::Mpeg2)
    {
        // An MPEG-2 track is named only by the descriptor's object type.
        out.extend(box_bytes(b"esds", &esds(&track.params)));
    }
    if let Some(color) = track.params.color {
        let mut colr = b"nclx".to_vec();
        colr.extend_from_slice(&color.primaries.to_be_bytes());
        colr.extend_from_slice(&color.transfer.to_be_bytes());
        colr.extend_from_slice(&color.matrix.to_be_bytes());
        colr.push(if color.full_range { 0x80 } else { 0 });
        out.extend(box_bytes(b"colr", &colr));
    }
    Ok(out)
}

fn esds(params: &StreamParams) -> Vec<u8> {
    let (oti, stream_type): (u8, u8) = match params.codec {
        CodecId::Mp3 => (0x6B, 0x15),
        CodecId::Mpeg4 => (0x20, 0x11),
        CodecId::Mpeg2 => (mpeg2_object_type(&params.extradata), 0x11),
        _ => (0x40, 0x15),
    };
    let mut decoder = Vec::new();
    decoder.push(oti);
    decoder.push(stream_type);
    decoder.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    if !params.extradata.is_empty() {
        decoder.extend(descriptor(0x05, &params.extradata));
    }
    let mut es = Vec::new();
    es.extend_from_slice(&[0, 0, 0]);
    es.extend(descriptor(0x04, &decoder));
    es.extend(descriptor(0x06, &[0x02]));
    let mut payload = vec![0, 0, 0, 0];
    payload.extend(descriptor(0x03, &es));
    payload
}

/// The ISO/IEC 14496-1 object type for an H.262 sequence header and its
/// extension: the profile in `profile_and_level_indication`, 4:2:2 for
/// that escaped profile, and MPEG-1 when no sequence extension follows.
/// Main when the headers are not there yet.
fn mpeg2_object_type(headers: &[u8]) -> u8 {
    const MAIN: u8 = 0x61;
    if headers.is_empty() {
        return MAIN;
    }
    let extension = headers
        .windows(6)
        .find(|bytes| bytes[..4] == [0, 0, 1, 0xB5] && bytes[4] >> 4 == 1);
    let Some(bytes) = extension else {
        return 0x6A;
    };
    let indication = (bytes[4] << 4) | (bytes[5] >> 4);
    if indication & 0x80 != 0 {
        return match indication & 0x0F {
            0x02 | 0x05 => 0x65,
            _ => MAIN,
        };
    }
    match (indication >> 4) & 0x07 {
        1 => 0x64,
        2 => 0x63,
        3 => 0x62,
        5 => 0x60,
        _ => MAIN,
    }
}

fn descriptor(tag: u8, body: &[u8]) -> Vec<u8> {
    let mut out = vec![tag];
    let mut size = body.len();
    let mut bytes = Vec::new();
    for _ in 0..4 {
        let mut piece = u8::try_from(size & 0x7f).unwrap_or(0);
        size >>= 7;
        if size > 0 {
            piece |= 0x80;
        }
        bytes.push(piece);
        if size == 0 {
            break;
        }
    }
    out.extend(bytes);
    out.extend_from_slice(body);
    out
}

/// A table's entry count, refused when it does not fit the 32-bit field.
fn count(rows: usize, what: &'static str) -> Result<[u8; 4], Error> {
    u32::try_from(rows)
        .map(u32::to_be_bytes)
        .map_err(|_| Error::Limit {
            what,
            value: u64::try_from(rows).unwrap_or(u64::MAX),
        })
}

/// `stts`, with the last sample `end_cut` ticks shorter than it was
/// written.
fn stts(samples: &[Sample], end_cut: u32) -> Result<Vec<u8>, Error> {
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for (index, sample) in samples.iter().enumerate() {
        let duration = if index + 1 == samples.len() {
            sample.duration.saturating_sub(end_cut)
        } else {
            sample.duration
        };
        if let Some(last) = runs.last_mut()
            && last.1 == duration
            && last.0 < u32::MAX
        {
            last.0 += 1;
            continue;
        }
        runs.push((1, duration));
    }
    let mut payload = vec![0, 0, 0, 0];
    payload.extend_from_slice(&count(runs.len(), "stts rows")?);
    for (count, delta) in runs {
        payload.extend_from_slice(&count.to_be_bytes());
        payload.extend_from_slice(&delta.to_be_bytes());
    }
    sized_box(b"stts", &payload)
}

fn ctts(samples: &[Sample]) -> Result<Vec<u8>, Error> {
    let negative = samples.iter().any(|sample| sample.cts < 0);
    let mut runs: Vec<(u32, i32)> = Vec::new();
    for sample in samples {
        if let Some(last) = runs.last_mut()
            && last.1 == sample.cts
            && last.0 < u32::MAX
        {
            last.0 += 1;
            continue;
        }
        runs.push((1, sample.cts));
    }
    // Version 1 reads the offset as signed. Without a negative offset the
    // bytes are the same either way.
    let mut payload = vec![u8::from(negative), 0, 0, 0];
    payload.extend_from_slice(&count(runs.len(), "ctts rows")?);
    for (count, offset) in runs {
        payload.extend_from_slice(&count.to_be_bytes());
        payload.extend_from_slice(&offset.to_be_bytes());
    }
    sized_box(b"ctts", &payload)
}

/// `cslg` (ISO/IEC 14496-12 8.6.1.4) for a `ctts` with negative offsets,
/// as QuickTime and AVFoundation write it: the shift that puts every
/// composition time at or after its decode time, the least and greatest
/// offsets, and the first and last composition times. Version 1 holds a
/// value that does not fit 32 bits.
fn cslg(samples: &[Sample]) -> Vec<u8> {
    let offsets = samples.iter().map(|sample| i64::from(sample.cts));
    let least = offsets.clone().min().unwrap_or(0);
    let greatest = offsets.max().unwrap_or(0);
    let mut start = i64::MAX;
    let mut end = i64::MIN;
    let mut dts = 0i64;
    for sample in samples {
        let pts = dts.saturating_add(i64::from(sample.cts));
        start = start.min(pts);
        end = end.max(pts.saturating_add(i64::from(sample.duration)));
        dts = dts.saturating_add(i64::from(sample.duration));
    }
    if samples.is_empty() {
        start = 0;
        end = 0;
    }
    let fields = [least.saturating_neg().max(0), least, greatest, start, end];
    let narrow = fields.iter().all(|&value| i32::try_from(value).is_ok());
    let mut payload = vec![u8::from(!narrow), 0, 0, 0];
    for value in fields {
        if narrow {
            payload.extend_from_slice(&i32::try_from(value).unwrap_or(0).to_be_bytes());
        } else {
            payload.extend_from_slice(&value.to_be_bytes());
        }
    }
    box_bytes(b"cslg", &payload)
}

/// A run of one track's samples that lie one after another in the file.
struct Chunk {
    offset: u64,
    /// Entries of the sample size table in this chunk.
    samples: u32,
}

/// Groups samples into chunks. A sample joins the chunk before it when it
/// starts where that chunk ends and the chunk stays within `max_bytes`;
/// `max_bytes` of 0 gives every sample its own chunk. `units` is how many
/// entries of the sample size table one written sample stands for.
fn chunks(samples: &[Sample], max_bytes: u64, units: impl Fn(&Sample) -> u32) -> Vec<Chunk> {
    let mut out: Vec<Chunk> = Vec::new();
    let mut bytes = 0u64;
    for sample in samples {
        let size = u64::from(sample.size);
        if let Some(last) = out.last_mut()
            && last.offset.checked_add(bytes) == Some(sample.offset)
            && bytes.saturating_add(size) <= max_bytes
            && let Some(joined) = last.samples.checked_add(units(sample))
        {
            last.samples = joined;
            bytes += size;
            continue;
        }
        out.push(Chunk {
            offset: sample.offset,
            samples: units(sample),
        });
        bytes = size;
    }
    out
}

/// `stsc`: one row where the samples in a chunk change.
fn stsc(chunks: &[Chunk]) -> Result<Vec<u8>, Error> {
    let mut rows = Vec::new();
    let mut last = None;
    for (index, chunk) in chunks.iter().enumerate() {
        if last == Some(chunk.samples) {
            continue;
        }
        last = Some(chunk.samples);
        let first = u32::try_from(index + 1).map_err(|_| Error::Limit {
            what: "chunks",
            value: u64::try_from(index).unwrap_or(u64::MAX),
        })?;
        rows.extend_from_slice(&first.to_be_bytes());
        rows.extend_from_slice(&chunk.samples.to_be_bytes());
        rows.extend_from_slice(&1u32.to_be_bytes());
    }
    let mut payload = vec![0, 0, 0, 0];
    payload.extend_from_slice(&count(rows.len() / 12, "stsc rows")?);
    payload.extend(rows);
    sized_box(b"stsc", &payload)
}

/// `stco` when every chunk offset, moved `shift` bytes later, fits 32
/// bits, and `co64` otherwise.
fn chunk_offsets(chunks: &[Chunk], shift: u64) -> Result<Vec<u8>, Error> {
    let place = |chunk: &Chunk| {
        chunk.offset.checked_add(shift).ok_or(Error::Limit {
            what: "chunk offset",
            value: chunk.offset,
        })
    };
    let mut narrow = true;
    for chunk in chunks {
        narrow &= place(chunk)? <= u64::from(u32::MAX);
    }
    let mut payload = vec![0, 0, 0, 0];
    payload.extend_from_slice(&count(chunks.len(), "chunks")?);
    for chunk in chunks {
        let at = place(chunk)?;
        if narrow {
            payload.extend_from_slice(&u32::try_from(at).unwrap_or(u32::MAX).to_be_bytes());
        } else {
            payload.extend_from_slice(&at.to_be_bytes());
        }
    }
    sized_box(if narrow { b"stco" } else { b"co64" }, &payload)
}

fn stsz(samples: &[Sample]) -> Result<Vec<u8>, Error> {
    let constant = samples
        .first()
        .is_some_and(|first| samples.iter().all(|sample| sample.size == first.size));
    let mut payload = vec![0, 0, 0, 0];
    if constant && !samples.is_empty() {
        payload.extend_from_slice(&samples[0].size.to_be_bytes());
        payload.extend_from_slice(&count(samples.len(), "samples")?);
    } else {
        payload.extend_from_slice(&0u32.to_be_bytes());
        payload.extend_from_slice(&count(samples.len(), "samples")?);
        for sample in samples {
            payload.extend_from_slice(&sample.size.to_be_bytes());
        }
    }
    sized_box(b"stsz", &payload)
}

fn stss(samples: &[Sample]) -> Result<Vec<u8>, Error> {
    let keys: Vec<u32> = samples
        .iter()
        .enumerate()
        .filter(|(_, sample)| sample.keyframe)
        .filter_map(|(index, _)| u32::try_from(index + 1).ok())
        .collect();
    let mut payload = vec![0, 0, 0, 0];
    payload.extend_from_slice(&count(keys.len(), "sync samples")?);
    for key in keys {
        payload.extend_from_slice(&key.to_be_bytes());
    }
    sized_box(b"stss", &payload)
}

fn trex(track: &TrackOut) -> Vec<u8> {
    let mut payload = vec![0, 0, 0, 0];
    payload.extend_from_slice(&track.params.track_id.to_be_bytes());
    payload.extend_from_slice(&1u32.to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes());
    box_bytes(b"trex", &payload)
}

/// One `moof` for the pending samples: a `traf` for each track that has
/// some, in track order. `data_start` is the distance from the start of the
/// `moof` to the first payload byte. When `grouped`, each track's payload
/// lies together, in track order, and its `traf` has one `trun`. Otherwise
/// the payload is in arrival order and each run of a track's samples has
/// its own `trun`.
fn moof_bytes(
    tracks: &[TrackOut],
    sequence: u32,
    data_start: u64,
    grouped: bool,
) -> Result<Vec<u8>, Error> {
    let mut body = Vec::new();
    let mut mfhd = vec![0, 0, 0, 0];
    mfhd.extend_from_slice(&sequence.to_be_bytes());
    body.extend(box_bytes(b"mfhd", &mfhd));
    let mut track_at = data_start;
    for track in tracks {
        if track.pending.is_empty() {
            continue;
        }
        let place = |at: u64| {
            i32::try_from(at).map_err(|_| Error::Limit {
                what: "fragment offset",
                value: at,
            })
        };
        let mut runs = Vec::new();
        if grouped {
            runs.push((track.pending.as_slice(), place(track_at)?));
            let bytes = track.runs.iter().fold(0u64, |sum, run| {
                sum.saturating_add(u64::try_from(run.len).unwrap_or(u64::MAX))
            });
            track_at = track_at.saturating_add(bytes);
        } else {
            for (index, run) in track.runs.iter().enumerate() {
                let end = track
                    .runs
                    .get(index + 1)
                    .map_or(track.pending.len(), |next| next.first);
                let samples = track.pending.get(run.first..end).ok_or(Error::Incomplete)?;
                let at = data_start.saturating_add(u64::try_from(run.at).unwrap_or(u64::MAX));
                runs.push((samples, place(at)?));
            }
        }
        body.extend(traf(track, &runs)?);
    }
    sized_box(b"moof", &body)
}

fn sample_flags(sample: &FragSample) -> u32 {
    if sample.keyframe {
        SAMPLE_KEY
    } else {
        SAMPLE_NOT_KEY
    }
}

/// One track's part of a fragment: `tfhd`, `tfdt`, and a `trun` for each
/// of `runs`, which pair a run's samples with its data offset.
///
/// A duration or size every sample of the track shares is stated once in
/// `tfhd` and left out of the runs. `tfhd` also states the flags of a
/// sample that is not a keyframe, or of a keyframe when all are. A run
/// whose first sample alone differs from that, the usual keyframe then its
/// dependents, names it in `first_sample_flags`; a run that differs
/// elsewhere lists every sample's flags. Composition offsets are left out
/// when all are 0.
fn traf(track: &TrackOut, runs: &[(&[FragSample], i32)]) -> Result<Vec<u8>, Error> {
    let samples = &track.pending;
    let Some(first) = samples.first() else {
        return Ok(Vec::new());
    };
    let one_duration = samples.iter().all(|s| s.duration == first.duration);
    let one_size = samples.iter().all(|s| s.size == first.size);
    let usual = if samples.iter().all(|s| s.keyframe) {
        SAMPLE_KEY
    } else {
        SAMPLE_NOT_KEY
    };
    let offsets = samples.iter().any(|s| s.cts != 0);

    // default-base-is-moof and default sample flags, then what is shared.
    let mut tf_flags = 0x02_0020u32;
    let mut tfhd_fields = Vec::new();
    if one_duration {
        tf_flags |= 0x08;
        tfhd_fields.extend_from_slice(&first.duration.to_be_bytes());
    }
    if one_size {
        tf_flags |= 0x10;
        tfhd_fields.extend_from_slice(&first.size.to_be_bytes());
    }
    tfhd_fields.extend_from_slice(&usual.to_be_bytes());
    let mut tfhd = tf_flags.to_be_bytes().to_vec();
    tfhd.extend_from_slice(&track.params.track_id.to_be_bytes());
    tfhd.extend(tfhd_fields);

    let mut tfdt = vec![1, 0, 0, 0];
    tfdt.extend_from_slice(&u64::try_from(track.pending_dts).unwrap_or(0).to_be_bytes());

    let mut body = box_bytes(b"tfhd", &tfhd);
    body.extend(box_bytes(b"tfdt", &tfdt));
    for (samples, data_offset) in runs {
        let Some(first) = samples.first() else {
            continue;
        };
        let rest_usual = samples.iter().skip(1).all(|s| sample_flags(s) == usual);
        let first_differs = sample_flags(first) != usual;
        // Version 1 so the composition offset is signed. Data offset always.
        let mut run_flags = 0x0100_0001u32;
        if rest_usual && first_differs {
            run_flags |= 0x004;
        }
        if !one_duration {
            run_flags |= 0x100;
        }
        if !one_size {
            run_flags |= 0x200;
        }
        if !rest_usual {
            run_flags |= 0x400;
        }
        if offsets {
            run_flags |= 0x800;
        }
        let mut trun = run_flags.to_be_bytes().to_vec();
        trun.extend_from_slice(&count(samples.len(), "fragment samples")?);
        trun.extend_from_slice(&data_offset.to_be_bytes());
        if rest_usual && first_differs {
            trun.extend_from_slice(&sample_flags(first).to_be_bytes());
        }
        for sample in *samples {
            if !one_duration {
                trun.extend_from_slice(&sample.duration.to_be_bytes());
            }
            if !one_size {
                trun.extend_from_slice(&sample.size.to_be_bytes());
            }
            if !rest_usual {
                trun.extend_from_slice(&sample_flags(sample).to_be_bytes());
            }
            if offsets {
                trun.extend_from_slice(&sample.cts.to_be_bytes());
            }
        }
        body.extend(sized_box(b"trun", &trun)?);
    }
    sized_box(b"traf", &body)
}

/// A `sidx` (ISO/IEC 14496-12 8.16.3) with one reference: the `moof` and
/// `mdat` that follow it, `group` bytes together. It is timed on the first
/// video track, or the first track of a file without video. Empty when
/// that track has no sample in the fragment or a field does not fit.
fn sidx_bytes(tracks: &[TrackOut], group: u64) -> Vec<u8> {
    let Some(track) = tracks.iter().find(|t| t.is_video()).or(tracks.first()) else {
        return Vec::new();
    };
    let Some(first) = track.pending.first() else {
        return Vec::new();
    };
    let mut earliest = i64::MAX;
    let mut dts = track.pending_dts;
    for sample in &track.pending {
        earliest = earliest.min(dts.saturating_add(i64::from(sample.cts)));
        dts = dts.saturating_add(i64::from(sample.duration));
    }
    let (Ok(size), Ok(duration)) = (u32::try_from(group), u32::try_from(track.pending_ticks))
    else {
        return Vec::new();
    };
    if size > 0x7fff_ffff {
        return Vec::new();
    }
    let mut payload = vec![1, 0, 0, 0];
    payload.extend_from_slice(&track.params.track_id.to_be_bytes());
    payload.extend_from_slice(&track.params.timescale.get().to_be_bytes());
    payload.extend_from_slice(&u64::try_from(earliest).unwrap_or(0).to_be_bytes());
    payload.extend_from_slice(&0u64.to_be_bytes()); // first_offset
    payload.extend_from_slice(&[0, 0, 0, 1]); // reserved, reference_count
    payload.extend_from_slice(&size.to_be_bytes()); // a media reference
    payload.extend_from_slice(&duration.to_be_bytes());
    // starts_with_SAP and SAP_type 1 when the fragment opens on a keyframe.
    let sap: u32 = if first.keyframe { 0x9000_0000 } else { 0 };
    payload.extend_from_slice(&sap.to_be_bytes());
    box_bytes(b"sidx", &payload)
}

/// `mfra`: for each track, the first sync sample of each fragment, by its
/// presentation time, with the `traf` and `trun` it is in and its place in
/// that run, all counted from 1.
fn mfra_bytes(tracks: &[TrackOut]) -> Vec<u8> {
    let mut body = Vec::new();
    for track in tracks {
        if track.tfra.is_empty() {
            continue;
        }
        let mut tfra = vec![1, 0, 0, 0];
        tfra.extend_from_slice(&track.params.track_id.to_be_bytes());
        // One byte for the traf number, four each for the trun and sample.
        tfra.extend_from_slice(&0x0Fu32.to_be_bytes());
        tfra.extend_from_slice(&u32::try_from(track.tfra.len()).unwrap_or(0).to_be_bytes());
        for row in &track.tfra {
            tfra.extend_from_slice(&row.time.to_be_bytes());
            tfra.extend_from_slice(&row.moof_offset.to_be_bytes());
            tfra.push(row.traf_number);
            tfra.extend_from_slice(&row.trun_number.to_be_bytes());
            tfra.extend_from_slice(&row.sample_number.to_be_bytes());
        }
        body.extend(box_bytes(b"tfra", &tfra));
    }
    let mfra_size = u32::try_from(body.len() + 8 + 16).unwrap_or(u32::MAX);
    let mut mfro = vec![0, 0, 0, 0];
    mfro.extend_from_slice(&mfra_size.to_be_bytes());
    body.extend(box_bytes(b"mfro", &mfro));
    box_bytes(b"mfra", &body)
}

fn mvhd(timescale: u32, duration: u64, tracks: usize, times: u64) -> Vec<u8> {
    let mut payload = timed_header(0, times, &timescale.to_be_bytes(), duration);
    payload.extend_from_slice(&0x00010000u32.to_be_bytes());
    payload.extend_from_slice(&0x0100u16.to_be_bytes());
    payload.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    payload.extend_from_slice(&matrix());
    payload.extend_from_slice(&[0; 24]);
    let next = u32::try_from(tracks + 1).unwrap_or(1);
    payload.extend_from_slice(&next.to_be_bytes());
    box_bytes(b"mvhd", &payload)
}

fn matrix() -> Vec<u8> {
    let mut out = Vec::new();
    for value in [0x00010000u32, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000] {
        out.extend_from_slice(&value.to_be_bytes());
    }
    out
}

fn fixed_16_16(value: u32) -> u32 {
    u32::from(u16::try_from(value).unwrap_or(u16::MAX)) << 16
}

fn box_bytes(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let size = 8u32.saturating_add(u32::try_from(payload.len()).unwrap_or(0));
    let mut out = Vec::with_capacity(payload.len() + 8);
    out.extend_from_slice(&size.to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(payload);
    out
}

/// A box whose payload can be large: refused when its size does not fit
/// the 32-bit size field, where `box_bytes` would write a wrong size.
fn sized_box(kind: &[u8; 4], payload: &[u8]) -> Result<Vec<u8>, Error> {
    let size = u64::try_from(payload.len()).unwrap_or(u64::MAX);
    if size > u64::from(u32::MAX) - 8 {
        return Err(Error::Limit {
            what: "index box",
            value: size,
        });
    }
    Ok(box_bytes(kind, payload))
}

#[cfg(test)]
mod tests {
    use super::mpeg2_object_type;

    fn headers(indication: Option<u8>) -> Vec<u8> {
        let mut bytes = vec![
            0, 0, 1, 0xB3, 0x2D, 0x02, 0x40, 0x13, 0xFF, 0xFF, 0xE0, 0x00,
        ];
        if let Some(indication) = indication {
            bytes.extend([0, 0, 1, 0xB5, 0x10 | (indication >> 4), indication << 4]);
            bytes.extend([0, 1, 0, 0]);
        }
        bytes
    }

    #[test]
    fn the_mpeg2_object_type_follows_the_sequence_profile() {
        assert_eq!(mpeg2_object_type(&headers(Some(0x48))), 0x61);
        assert_eq!(mpeg2_object_type(&headers(Some(0x58))), 0x60);
        assert_eq!(mpeg2_object_type(&headers(Some(0x14))), 0x64);
        assert_eq!(mpeg2_object_type(&headers(Some(0x38))), 0x62);
        assert_eq!(mpeg2_object_type(&headers(Some(0x26))), 0x63);
        assert_eq!(mpeg2_object_type(&headers(Some(0x85))), 0x65);
        assert_eq!(mpeg2_object_type(&headers(Some(0x82))), 0x65);
        assert_eq!(mpeg2_object_type(&headers(None)), 0x6A);
        assert_eq!(mpeg2_object_type(&[]), 0x61);
    }
}
