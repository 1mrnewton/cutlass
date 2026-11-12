//! The Matroska and WebM muxer.
//!
//! Layout, in file order: the EBML header; a Segment whose size is patched
//! at finish; room for a SeekHead, filled at finish; Info (TimestampScale,
//! the writing apps, and a Duration patched at finish); Tracks; Clusters;
//! and Cues. The SeekHead points at Info, Tracks and Cues, so a reader that
//! wants the Cues finds them without scanning (RFC 9559, 25.1).
//!
//! # Packets
//!
//! Packets of different tracks are held per track and written in time
//! order, the earliest head first, once every track has a packet waiting.
//! A track that stops sending does not stall the file for long: past
//! [`MAX_HELD_BEFORE_HEADER`] bytes held, the earliest packet goes out
//! anyway. Finish writes everything still held. The Tracks element is
//! written with the first packet out, by which time each track's encoder
//! has handed over its configuration.
//!
//! Frames are written straight from the packet to the writer; only element
//! headers are built in memory. A Cluster is opened with an eight-octet
//! size that is patched when it closes. A new Cluster starts at a video
//! keyframe once the open one holds more than 4 KiB, after five seconds or
//! five megabytes (RFC 9559, 25.1), and whenever a block's time would not
//! fit the signed 16-bit offset from the Cluster's Timestamp.
//!
//! # Timestamps
//!
//! The Segment clock is TimestampScale nanoseconds per tick, one
//! millisecond unless [`MatroskaMux::with_timestamp_scale`] says otherwise.
//! A packet's presentation time is converted from its track's timescale to
//! that clock, rounded to the nearest tick, as the format requires: a file
//! at the default scale keeps times to the millisecond. A track's first
//! packet duration (when not zero) becomes its DefaultDuration in
//! nanoseconds; a packet with another duration is written in a BlockGroup
//! with its BlockDuration.
//!
//! # Codec delay and padding
//!
//! An Opus track's CodecDelay is its pre-skip and its SeekPreRoll 80 ms
//! (RFC 9559, 5.1.4.1.30; the Matroska Opus mapping). Any other audio
//! track whose encoder delay is stated ([`Mux::set_trim`]) before its first
//! packet goes out gets that as CodecDelay. Packet times are not shifted:
//! the stored timestamps include the delay, which a player subtracts.
//! Trailing padding stated for those tracks is the number of samples to
//! drop from the track's end: the packets that come after it are held to
//! the end, and the padding comes off their BlockDurations, the last packet
//! first, and is written as their DiscardPadding. An
//! Opus packet whose duration falls short of the samples its TOC byte says
//! it holds (an encoder's last packet) has the difference written as
//! DiscardPadding, so a player drops those samples.

use std::collections::VecDeque;
use std::io::{Seek, SeekFrom, Write};

use cutlass_codec::{CodecId, ContainerId, Packet, StreamCodec, StreamParams, Timescale};
use cutlass_container::{
    FailedMux, FinishedMux, IndexPlace, Language, Limits, Mux, MuxError, MuxMode, MuxStatus,
    NewStream, TrackDuration, TrackId, Trim,
};

use crate::codec::{
    Written, aac_config, allowed, opus_head, opus_packet_samples, opus_pre_skip, written,
};
use crate::ebml::{
    UNKNOWN_SIZE, put_bytes, put_float, put_id, put_master, put_size, put_size_width, put_uint,
    put_uint8, put_void, size_width,
};
use crate::error::Error;
use crate::ids;
use crate::limits::{CLUSTER_BYTES, CLUSTER_SPAN_NS, MAX_HELD_BEFORE_HEADER};
use crate::meta::DocType;

const SECOND_NS: i128 = 1_000_000_000;

/// Bytes kept for the SeekHead: three Seek entries with eight-octet
/// positions take 68, and a Void fills the rest.
const SEEK_HEAD_ROOM: usize = 96;

/// A video keyframe starts a new Cluster once the open one holds this much.
const KEYFRAME_CLUSTER_BYTES: u64 = 4 * 1024;

/// SeekPreRoll for Opus, from the Matroska Opus mapping.
const OPUS_SEEK_PRE_ROLL_NS: u64 = 80_000_000;

const APP: &[u8] = b"Cutlass";

/// One packet waiting for its turn. `tail` marks a packet that came after
/// the encoder stated trailing padding: it is held to the end, when
/// `padding` (samples) is shared out over the track's last packets.
struct Held {
    packet: Packet,
    padding: u64,
    tail: bool,
}

struct Track {
    id: TrackId,
    codec: CodecId,
    params: StreamParams,
    timescale: Timescale,
    language: Option<Language>,
    config: Option<Vec<u8>>,
    trim: Option<Trim>,
    /// Samples to drop from the end of the track, as last stated.
    end_padding: u64,
    queue: VecDeque<Held>,
    /// Sum of the durations of the packets taken, in track ticks.
    next: i64,
    /// The first packet's duration in track ticks, written as
    /// DefaultDuration when it is not zero.
    default_duration: Option<i64>,
    written: Option<Written>,
    /// Where the CodecPrivate data sits, for a configuration that comes
    /// again later with the same length (FLAC's final STREAMINFO).
    private_at: Option<u64>,
    late_private: Option<Vec<u8>>,
    /// Time of this track's previous block, Segment ticks.
    previous: Option<i64>,
    /// A block of this track is in the open Cluster.
    in_cluster: bool,
}

impl Track {
    fn is_video(&self) -> bool {
        self.params.size().is_some()
    }

    /// Configuration as the file stores it.
    fn private_source(&self) -> &[u8] {
        self.config
            .as_deref()
            .unwrap_or_else(|| self.params.extradata())
    }
}

#[derive(Clone, Copy, Debug)]
struct Layout {
    segment_size_at: u64,
    segment_data: u64,
    seek_head_at: u64,
    info_at: u64,
    duration_at: u64,
    tracks_at: u64,
}

#[derive(Clone, Copy, Debug)]
struct OpenCluster {
    start: u64,
    data: u64,
    /// Segment ticks.
    timestamp: i64,
    bytes: u64,
}

/// Track index and the offset of its CodecPrivate data inside Tracks.
type PrivateOffsets = Vec<(usize, u64)>;

/// A block just written.
#[derive(Clone, Copy, Debug)]
struct BlockDone {
    element_start: u64,
    /// Segment ticks.
    time: i64,
    /// End of the packet in track ticks.
    end: i64,
    frame_len: u64,
    keyframe: bool,
}

#[derive(Clone, Copy, Debug)]
struct CueEntry {
    time: u64,
    track: u32,
    cluster: u64,
    relative: u64,
}

/// A Matroska or WebM file written as packets arrive, with Cues at the end.
pub struct MatroskaMux<W: Write + Seek> {
    writer: W,
    doc_type: DocType,
    scale: u64,
    limits: Limits,
    tracks: Vec<Track>,
    status: MuxStatus,
    failure: Option<MuxError>,
    payload: u64,
    /// Offset the writer is at.
    position: u64,
    layout: Option<Layout>,
    held: u64,
    cluster: Option<OpenCluster>,
    cues: Vec<CueEntry>,
    /// Latest packet end, in nanoseconds.
    end_ns: i128,
}

impl<W: Write + Seek> MatroskaMux<W> {
    /// A Matroska (`.mkv`) file.
    #[must_use]
    pub fn new(writer: W) -> Self {
        Self::with_doc_type(writer, DocType::Matroska)
    }

    /// A WebM file: VP8, VP9, AV1, Opus and Vorbis only.
    #[must_use]
    pub fn webm(writer: W) -> Self {
        Self::with_doc_type(writer, DocType::Webm)
    }

    #[must_use]
    pub fn with_doc_type(writer: W, doc_type: DocType) -> Self {
        Self {
            writer,
            doc_type,
            scale: 1_000_000,
            limits: Limits::standard(),
            tracks: Vec::new(),
            status: MuxStatus::Building,
            failure: None,
            payload: 0,
            position: 0,
            layout: None,
            held: 0,
            cluster: None,
            cues: Vec::new(),
            end_ns: 0,
        }
    }

    /// Nanoseconds per Segment tick. One millisecond by default; a finer
    /// scale keeps more precision and makes Clusters shorter, since a block
    /// sits at most 32767 ticks from its Cluster's Timestamp. Zero is
    /// refused at the first packet.
    #[must_use]
    pub const fn with_timestamp_scale(mut self, nanoseconds: u64) -> Self {
        self.scale = nanoseconds;
        self
    }

    /// The writer, once the file is finished.
    pub fn into_writer(self) -> Result<W, MuxError> {
        if self.status != MuxStatus::Finished {
            return Err(MuxError::NotBuilding);
        }
        Ok(self.writer)
    }

    fn webm_only(&self) -> bool {
        self.doc_type == DocType::Webm
    }

    fn fail(&mut self, error: MuxError) -> MuxError {
        self.status = MuxStatus::Failed;
        self.failure = Some(error.clone());
        error
    }

    fn building(&self) -> Result<(), MuxError> {
        match self.status {
            MuxStatus::Building => Ok(()),
            MuxStatus::Failed => Err(self.failure.clone().unwrap_or(MuxError::NotBuilding)),
            MuxStatus::Finished => Err(MuxError::AlreadyFinished),
        }
    }

    fn track(&self, id: u32) -> Result<usize, MuxError> {
        self.tracks
            .iter()
            .position(|track| track.id.get() == id)
            .ok_or(MuxError::UnknownTrack { id })
    }

    // Writing bytes.

    fn emit(&mut self, bytes: &[u8]) -> Result<(), MuxError> {
        self.writer.write_all(bytes).map_err(io)?;
        self.position = self
            .position
            .saturating_add(u64::try_from(bytes.len()).unwrap_or(u64::MAX));
        Ok(())
    }

    /// Writes `bytes` at `at` and comes back to the end.
    fn patch(&mut self, at: u64, bytes: &[u8]) -> Result<(), MuxError> {
        self.writer.seek(SeekFrom::Start(at)).map_err(io)?;
        self.writer.write_all(bytes).map_err(io)?;
        self.writer
            .seek(SeekFrom::Start(self.position))
            .map_err(io)?;
        Ok(())
    }

    /// Segment ticks for `ticks` of `timescale`, rounded to the nearest.
    fn to_segment(&self, ticks: i64, timescale: Timescale) -> Option<i64> {
        let denominator = i128::from(timescale.ticks_per_second()) * i128::from(self.scale);
        if denominator == 0 {
            return None;
        }
        let numerator = i128::from(ticks) * SECOND_NS;
        let rounded = (2 * numerator + denominator).div_euclid(2 * denominator);
        i64::try_from(rounded).ok()
    }

    // The header.

    fn write_header(&mut self) -> Result<(), MuxError> {
        if self.scale == 0 {
            return Err(rejected("TimestampScale is zero"));
        }
        let webm = self.webm_only();
        for track in &mut self.tracks {
            let source = track.private_source().to_vec();
            let private = match track.codec {
                CodecId::Aac if !source.is_empty() => aac_config(&source).map_err(from_crate)?,
                CodecId::Opus if !source.is_empty() => opus_head(&source).map_err(from_crate)?,
                _ => source,
            };
            track.written = Some(written(track.codec, &private, webm).map_err(from_crate)?);
            track.default_duration = track
                .queue
                .front()
                .and_then(|held| held.packet.timestamp().duration())
                .filter(|duration| *duration > 0);
        }
        self.position = self.writer.stream_position().map_err(io)?;
        let mut out = Vec::new();
        let mut header = Vec::new();
        put_uint(&mut header, ids::EBML_VERSION, 1);
        put_uint(&mut header, ids::EBML_READ_VERSION, 1);
        put_uint(&mut header, ids::EBML_MAX_ID_LENGTH, 4);
        put_uint(&mut header, ids::EBML_MAX_SIZE_LENGTH, 8);
        put_bytes(
            &mut header,
            ids::DOC_TYPE,
            self.doc_type.as_str().as_bytes(),
        );
        put_uint(&mut header, ids::DOC_TYPE_VERSION, 4);
        put_uint(&mut header, ids::DOC_TYPE_READ_VERSION, 2);
        put_master(&mut out, ids::EBML, &header);
        put_id(&mut out, ids::SEGMENT);
        let segment_size_at = self.position + len(&out);
        out.extend_from_slice(&UNKNOWN_SIZE);
        let segment_data = self.position + len(&out);
        let seek_head_at = segment_data;
        put_void(&mut out, SEEK_HEAD_ROOM);
        let info_at = self.position + len(&out);
        let mut info = Vec::new();
        put_uint(&mut info, ids::TIMESTAMP_SCALE, self.scale);
        put_bytes(&mut info, ids::MUXING_APP, APP);
        put_bytes(&mut info, ids::WRITING_APP, APP);
        let duration_in_info = len(&info) + 3;
        put_float(&mut info, ids::DURATION, 0.0);
        put_id(&mut out, ids::INFO);
        put_size(&mut out, len(&info));
        let duration_at = self.position + len(&out) + duration_in_info;
        out.extend_from_slice(&info);
        let tracks_at = self.position + len(&out);
        let (entries, private_offsets) = self.track_entries()?;
        put_id(&mut out, ids::TRACKS);
        put_size(&mut out, len(&entries));
        let entries_at = self.position + len(&out);
        out.extend_from_slice(&entries);
        for (index, offset) in private_offsets {
            self.tracks[index].private_at = Some(entries_at + offset);
        }
        self.emit(&out)?;
        self.layout = Some(Layout {
            segment_size_at,
            segment_data,
            seek_head_at,
            info_at,
            duration_at,
            tracks_at,
        });
        Ok(())
    }

    /// The TrackEntry elements, and where each CodecPrivate's data sits
    /// inside them.
    fn track_entries(&self) -> Result<(Vec<u8>, PrivateOffsets), MuxError> {
        let mut out = Vec::new();
        let mut offsets = Vec::new();
        for (index, track) in self.tracks.iter().enumerate() {
            let Some(written) = &track.written else {
                return Err(rejected("track header is not ready"));
            };
            let number = u64::from(track.id.get());
            let mut entry = Vec::new();
            put_uint(&mut entry, ids::TRACK_NUMBER, number);
            put_uint(&mut entry, ids::TRACK_UID, number);
            put_uint(
                &mut entry,
                ids::TRACK_TYPE,
                if written.video { 1 } else { 2 },
            );
            put_uint(&mut entry, ids::FLAG_LACING, 0);
            if let Some(language) = &track.language {
                let text = language.as_str();
                if text.len() == 3 && text.bytes().all(|byte| byte.is_ascii_lowercase()) {
                    put_bytes(&mut entry, ids::LANGUAGE, text.as_bytes());
                } else if !self.webm_only() {
                    put_bytes(&mut entry, ids::LANGUAGE_BCP47, text.as_bytes());
                }
            }
            put_bytes(&mut entry, ids::CODEC_ID, written.codec_id.as_bytes());
            if !written.private.is_empty() {
                put_id(&mut entry, ids::CODEC_PRIVATE);
                put_size(&mut entry, len(&written.private));
                offsets.push((index, len(&entry)));
                entry.extend_from_slice(&written.private);
            }
            if let Some(duration) = track.default_duration {
                let ns = ns_of(duration, track.timescale);
                if let Ok(ns) = u64::try_from(ns)
                    && ns > 0
                {
                    put_uint(&mut entry, ids::DEFAULT_DURATION, ns);
                }
            }
            let delay = self.codec_delay_ns(track, written);
            if delay > 0 {
                put_uint(&mut entry, ids::CODEC_DELAY, delay);
            }
            if track.codec == CodecId::Opus {
                put_uint(&mut entry, ids::SEEK_PRE_ROLL, OPUS_SEEK_PRE_ROLL_NS);
            }
            if let Some(size) = track.params.size() {
                let mut video = Vec::new();
                put_uint(&mut video, ids::PIXEL_WIDTH, u64::from(size.width()));
                put_uint(&mut video, ids::PIXEL_HEIGHT, u64::from(size.height()));
                put_master(&mut entry, ids::VIDEO, &video);
            } else if let (Some(rate), Some(channels)) =
                (track.params.sample_rate(), track.params.channels())
            {
                let mut audio = Vec::new();
                put_float(&mut audio, ids::SAMPLING_FREQUENCY, f64::from(rate));
                put_uint(&mut audio, ids::CHANNELS, u64::from(channels));
                if let Some(bits) = written.bit_depth {
                    put_uint(&mut audio, ids::BIT_DEPTH, bits);
                }
                put_master(&mut entry, ids::AUDIO, &audio);
            }
            put_id(&mut out, ids::TRACK_ENTRY);
            put_size(&mut out, len(&entry));
            let entry_data = len(&out);
            for (_, offset) in offsets.iter_mut().filter(|(at, _)| *at == index) {
                *offset += entry_data;
            }
            out.extend_from_slice(&entry);
        }
        Ok((out, offsets))
    }

    /// CodecDelay in nanoseconds: Opus's pre-skip, or the encoder delay
    /// stated for any other audio track.
    fn codec_delay_ns(&self, track: &Track, written: &Written) -> u64 {
        let samples = match track.codec {
            CodecId::Opus => opus_pre_skip(&written.private).map(u64::from),
            _ if !track.is_video() => track.trim.and_then(Trim::encoder_delay),
            _ => None,
        };
        let rate = match track.codec {
            CodecId::Opus => 48_000,
            _ => track.params.sample_rate().unwrap_or(0),
        };
        match (samples, rate) {
            (Some(samples), rate) if rate > 0 => {
                let ns =
                    (u128::from(samples) * 1_000_000_000 + u128::from(rate) / 2) / u128::from(rate);
                u64::try_from(ns).unwrap_or(0)
            }
            _ => 0,
        }
    }

    // Packets.

    /// Writes held packets in time order while every track has one, or
    /// all of them when `force` is set.
    fn drain(&mut self, force: bool) -> Result<(), MuxError> {
        loop {
            let waiting = self.tracks.iter().any(|track| track.queue.is_empty());
            if !force && waiting && self.held < MAX_HELD_BEFORE_HEADER {
                return Ok(());
            }
            if self.layout.is_none() {
                self.write_header()?;
            }
            let Some(index) = self.earliest() else {
                return Ok(());
            };
            // The end of a padded track waits for finish, unless holding it
            // would keep too much.
            let tail = self.tracks[index]
                .queue
                .front()
                .is_some_and(|held| held.tail);
            if !force && tail && self.held < MAX_HELD_BEFORE_HEADER {
                return Ok(());
            }
            let Some(held) = self.tracks[index].queue.pop_front() else {
                return Ok(());
            };
            self.held = self
                .held
                .saturating_sub(u64::try_from(held.packet.bytes().len()).unwrap_or(0));
            self.write_block(index, &held)?;
        }
    }

    /// The track whose head packet comes first.
    fn earliest(&self) -> Option<usize> {
        let mut best: Option<(usize, i64, Timescale)> = None;
        for (index, track) in self.tracks.iter().enumerate() {
            let Some(head) = track.queue.front() else {
                continue;
            };
            let time = head.packet.timestamp();
            let ticks = time.decode().or(time.presentation()).unwrap_or(0);
            let earlier = match best {
                None => true,
                Some((_, best_ticks, best_scale)) => {
                    i128::from(ticks) * i128::from(best_scale.ticks_per_second())
                        < i128::from(best_ticks) * i128::from(track.timescale.ticks_per_second())
                }
            };
            if earlier {
                best = Some((index, ticks, track.timescale));
            }
        }
        best.map(|(index, _, _)| index)
    }

    fn write_block(&mut self, index: usize, held: &Held) -> Result<(), MuxError> {
        let packet = &held.packet;
        let timestamp = packet.timestamp();
        let track = &self.tracks[index];
        let timescale = track.timescale;
        let pts = timestamp
            .presentation_or_decode()
            .ok_or(MuxError::MissingTimestamp)?;
        let duration = timestamp.duration().ok_or(MuxError::MissingTimestamp)?;
        let time = self
            .to_segment(pts, timescale)
            .ok_or_else(|| rejected("timestamp does not fit"))?;
        let (played, discard_ns) = trailing(track, held, duration);
        let video = track.is_video();
        let keyframe = packet.keyframe();
        let needs_group = track.default_duration != Some(duration) || discard_ns > 0;
        let strip = track
            .written
            .as_ref()
            .is_some_and(|written| written.strip_prores);
        let frame = match packet.bytes() {
            bytes if strip && bytes.get(4..8) == Some(b"icpf") => &bytes[8..],
            bytes => bytes,
        };
        let frame_len = len(frame);

        // The Cluster.
        let span_ticks = i64::try_from(CLUSTER_SPAN_NS / self.scale.max(1)).unwrap_or(i64::MAX);
        let reopen = match self.cluster {
            None => true,
            Some(open) => {
                let relative = time - open.timestamp;
                !(i64::from(i16::MIN)..=i64::from(i16::MAX)).contains(&relative)
                    || open.bytes > u64::try_from(CLUSTER_BYTES).unwrap_or(u64::MAX)
                    || relative > span_ticks
                    || (video && keyframe && open.bytes > KEYFRAME_CLUSTER_BYTES)
            }
        };
        if reopen {
            self.close_cluster()?;
            self.open_cluster(time.max(0))?;
        }
        let open = self.cluster.ok_or_else(|| rejected("no open Cluster"))?;
        let relative = i16::try_from(time - open.timestamp)
            .map_err(|_| rejected("a timestamp is more than 32767 ticks before zero"))?;

        // The block header: track, time, flags.
        let mut head = Vec::with_capacity(32);
        put_size(&mut head, u64::from(self.tracks[index].id.get()));
        head.extend_from_slice(&relative.to_be_bytes());
        let mut element = Vec::with_capacity(48);
        if needs_group {
            head.push(0);
            let block_len = len(&head) + frame_len;
            let mut tail = Vec::new();
            let block_duration = self
                .to_segment(played, timescale)
                .and_then(|value| u64::try_from(value).ok())
                .unwrap_or(0);
            put_uint(&mut tail, ids::BLOCK_DURATION, block_duration);
            if !keyframe {
                let back = self.tracks[index]
                    .previous
                    .map_or(-1, |previous| previous - time)
                    .clamp(i64::from(i16::MIN), i64::from(i16::MAX));
                let back = if back == 0 { -1 } else { back };
                put_id(&mut tail, ids::REFERENCE_BLOCK);
                let bytes = back.to_be_bytes();
                put_size(&mut tail, 2);
                tail.extend_from_slice(&bytes[6..]);
            }
            if discard_ns > 0 {
                put_id(&mut tail, ids::DISCARD_PADDING);
                put_size(&mut tail, 8);
                tail.extend_from_slice(&discard_ns.to_be_bytes());
            }
            let group_len = 1 + size_width(block_len) as u64 + block_len + len(&tail);
            put_id(&mut element, ids::BLOCK_GROUP);
            put_size(&mut element, group_len);
            put_id(&mut element, ids::BLOCK);
            put_size(&mut element, block_len);
            element.extend_from_slice(&head);
            let element_start = self.position;
            self.emit(&element)?;
            self.emit(frame)?;
            self.emit(&tail)?;
            self.after_block(
                index,
                open,
                BlockDone {
                    element_start,
                    time,
                    end: pts.saturating_add(duration),
                    frame_len,
                    keyframe,
                },
            )?;
        } else {
            head.push(if keyframe { 0x80 } else { 0 });
            let block_len = len(&head) + frame_len;
            put_id(&mut element, ids::SIMPLE_BLOCK);
            put_size(&mut element, block_len);
            element.extend_from_slice(&head);
            let element_start = self.position;
            self.emit(&element)?;
            self.emit(frame)?;
            self.after_block(
                index,
                open,
                BlockDone {
                    element_start,
                    time,
                    end: pts.saturating_add(duration),
                    frame_len,
                    keyframe,
                },
            )?;
        }
        Ok(())
    }

    /// Cue point, Cluster size, payload and end time after a block.
    fn after_block(
        &mut self,
        index: usize,
        open: OpenCluster,
        block: BlockDone,
    ) -> Result<(), MuxError> {
        let layout = self.layout.ok_or_else(|| rejected("no header"))?;
        let has_video = self.tracks.iter().any(Track::is_video);
        let track = &mut self.tracks[index];
        let first_in_cluster = !track.in_cluster;
        track.in_cluster = true;
        track.previous = Some(block.time);
        let id = track.id.get();
        let timescale = track.timescale;
        // Video keyframes when there is video; otherwise each track's first
        // block in each Cluster.
        let wants_cue = if has_video {
            track.is_video() && block.keyframe
        } else {
            first_in_cluster
        };
        if wants_cue {
            let count = u32::try_from(self.cues.len()).unwrap_or(u32::MAX);
            if count >= self.limits.max_index_entries {
                return Err(MuxError::HostileSize {
                    declared: u64::from(count) + 1,
                    limit: u64::from(self.limits.max_index_entries),
                });
            }
            self.cues.push(CueEntry {
                time: u64::try_from(block.time.max(0)).unwrap_or(0),
                track: id,
                cluster: open.start - layout.segment_data,
                relative: block.element_start - open.data,
            });
        }
        self.end_ns = self.end_ns.max(ns_of(block.end, timescale));
        if let Some(cluster) = self.cluster.as_mut() {
            cluster.bytes = cluster
                .bytes
                .saturating_add(self.position.saturating_sub(block.element_start));
        }
        self.payload = self.payload.saturating_add(block.frame_len);
        Ok(())
    }

    fn open_cluster(&mut self, timestamp: i64) -> Result<(), MuxError> {
        let start = self.position;
        let mut out = Vec::with_capacity(24);
        put_id(&mut out, ids::CLUSTER);
        out.extend_from_slice(&UNKNOWN_SIZE);
        let data = start + len(&out);
        put_uint(
            &mut out,
            ids::TIMESTAMP,
            u64::try_from(timestamp).unwrap_or(0),
        );
        self.emit(&out)?;
        for track in &mut self.tracks {
            track.in_cluster = false;
        }
        self.cluster = Some(OpenCluster {
            start,
            data,
            timestamp,
            bytes: self.position - data,
        });
        Ok(())
    }

    /// Writes the open Cluster's size over its unknown-size marker.
    fn close_cluster(&mut self) -> Result<(), MuxError> {
        let Some(open) = self.cluster.take() else {
            return Ok(());
        };
        let mut size = Vec::with_capacity(8);
        put_size_width(&mut size, self.position - open.data, 8);
        self.patch(open.start + 4, &size)
    }

    // Finish.

    fn finish_file(&mut self) -> Result<(), MuxError> {
        for track in &mut self.tracks {
            share_padding(track);
        }
        self.drain(true)?;
        if self.layout.is_none() {
            self.write_header()?;
        }
        self.close_cluster()?;
        let layout = self.layout.ok_or_else(|| rejected("no header"))?;
        let cues_at = self.position;
        if !self.cues.is_empty() {
            let cues = self.cues_element();
            self.emit(&cues)?;
        }
        // SeekHead.
        let mut entries = vec![
            (ids::INFO, layout.info_at - layout.segment_data),
            (ids::TRACKS, layout.tracks_at - layout.segment_data),
        ];
        if !self.cues.is_empty() {
            entries.push((ids::CUES, cues_at - layout.segment_data));
        }
        let mut body = Vec::new();
        for (id, position) in entries {
            let mut seek = Vec::new();
            let id_bytes = id.to_be_bytes();
            put_bytes(&mut seek, ids::SEEK_ID, &id_bytes);
            put_uint8(&mut seek, ids::SEEK_POSITION, position);
            put_master(&mut body, ids::SEEK, &seek);
        }
        let mut head = Vec::new();
        put_master(&mut head, ids::SEEK_HEAD, &body);
        let room = SEEK_HEAD_ROOM - head.len();
        put_void(&mut head, room);
        self.patch(layout.seek_head_at, &head)?;
        // Duration in Segment ticks.
        #[allow(clippy::cast_precision_loss)]
        let duration = self.end_ns.max(0) as f64 / self.scale as f64;
        self.patch(layout.duration_at, &duration.to_be_bytes())?;
        // A configuration that came again with the same length.
        for index in 0..self.tracks.len() {
            if let (Some(at), Some(private)) = (
                self.tracks[index].private_at,
                self.tracks[index].late_private.take(),
            ) {
                self.patch(at, &private)?;
            }
        }
        let mut size = Vec::with_capacity(8);
        put_size_width(&mut size, self.position - layout.segment_data, 8);
        self.patch(layout.segment_size_at, &size)?;
        self.writer.flush().map_err(io)
    }

    fn cues_element(&self) -> Vec<u8> {
        let mut body = Vec::with_capacity(self.cues.len() * 24);
        for cue in &self.cues {
            let mut positions = Vec::new();
            put_uint(&mut positions, ids::CUE_TRACK, u64::from(cue.track));
            put_uint(&mut positions, ids::CUE_CLUSTER_POSITION, cue.cluster);
            put_uint(&mut positions, ids::CUE_RELATIVE_POSITION, cue.relative);
            let mut point = Vec::new();
            put_uint(&mut point, ids::CUE_TIME, cue.time);
            put_master(&mut point, ids::CUE_TRACK_POSITIONS, &positions);
            put_master(&mut body, ids::CUE_POINT, &point);
        }
        let mut out = Vec::new();
        put_master(&mut out, ids::CUES, &body);
        out
    }
}

/// Shares a track's end padding out over its held last packets, the last
/// one first: a packet wholly padding is all discarded, and the rest comes
/// off the one before it.
fn share_padding(track: &mut Track) {
    let rate = i128::from(track.params.sample_rate().unwrap_or(0));
    let timescale = i128::from(track.timescale.ticks_per_second().max(1));
    let mut left = i128::from(track.end_padding);
    if rate == 0 {
        return;
    }
    if track.codec == CodecId::Opus {
        // The duration already leaves the padding out: the last packet
        // takes all of it, and `trailing` checks it against the TOC.
        if let Some(held) = track.queue.back_mut().filter(|held| held.tail) {
            held.padding = track.end_padding;
        }
        return;
    }
    for held in track.queue.iter_mut().rev() {
        if left <= 0 || !held.tail {
            break;
        }
        let duration = i128::from(held.packet.timestamp().duration().unwrap_or(0).max(0));
        // A packet of unknown length, as a Vorbis packet copied from a file
        // that states none, takes what is left.
        let samples = if duration == 0 {
            left
        } else {
            duration * rate / timescale
        };
        let take = left.min(samples);
        held.padding = u64::try_from(take).unwrap_or(0);
        left -= take;
    }
}

/// A packet's BlockDuration (track ticks) and DiscardPadding (ns).
///
/// Trailing padding the encoder stated, in samples, comes off the packet's
/// duration. An Opus packet whose duration is shorter than what its TOC
/// says it decodes to has the rest discarded, as an Opus encoder's last
/// packet is; a shortfall under one track tick is taken for rounding, so a
/// file remuxed from millisecond times does not discard inside the stream.
/// A stated padding within that tick of the shortfall is written exactly.
fn trailing(track: &Track, held: &Held, duration: i64) -> (i64, u64) {
    let timescale = i128::from(track.timescale.ticks_per_second().max(1));
    if track.codec == CodecId::Opus {
        let Some(samples) = opus_packet_samples(held.packet.bytes()) else {
            return (duration, 0);
        };
        let have = i128::from(duration) * 48_000 / timescale;
        let slack = (48_000 + timescale - 1) / timescale - 1;
        let extra = i128::from(samples) - have;
        let stated = i128::from(held.padding);
        if stated > 0 && stated < i128::from(samples) && (extra - stated).abs() <= slack.max(0) {
            // The stated padding, which the rounded duration only nears.
            let ns = u64::try_from(stated * SECOND_NS / 48_000).unwrap_or(0);
            return (duration, ns);
        }
        if extra <= slack.max(0) {
            return (duration, 0);
        }
        let ns = u64::try_from(extra * SECOND_NS / 48_000).unwrap_or(0);
        return (duration, ns);
    }
    let rate = i128::from(track.params.sample_rate().unwrap_or(0));
    if held.padding == 0 || rate == 0 {
        return (duration, 0);
    }
    let padding = i128::from(held.padding);
    let ticks = (padding * timescale + rate / 2) / rate;
    let ticks = i64::try_from(ticks)
        .unwrap_or(i64::MAX)
        .clamp(0, duration.max(0));
    let ns = u64::try_from(padding * SECOND_NS / rate).unwrap_or(0);
    (duration - ticks, ns)
}

fn len(bytes: &[u8]) -> u64 {
    u64::try_from(bytes.len()).unwrap_or(u64::MAX)
}

/// `ticks` of `timescale` in nanoseconds, rounded down.
fn ns_of(ticks: i64, timescale: Timescale) -> i128 {
    let rate = i128::from(timescale.ticks_per_second());
    if rate == 0 {
        return 0;
    }
    (i128::from(ticks) * SECOND_NS).div_euclid(rate)
}

fn io(error: std::io::Error) -> MuxError {
    MuxError::Rejected {
        message: format!("matroska write failed: {error}"),
    }
}

fn rejected(message: &str) -> MuxError {
    MuxError::Rejected {
        message: format!("matroska: {message}"),
    }
}

fn from_crate(error: Error) -> MuxError {
    MuxError::Rejected {
        message: error.to_string(),
    }
}

impl<W: Write + Seek + Send> Mux for MatroskaMux<W> {
    fn family(&self) -> ContainerId {
        ContainerId::Matroska
    }

    fn add_stream(&mut self, stream: NewStream) -> Result<TrackId, MuxError> {
        self.building()?;
        if self.layout.is_some() {
            return Err(self.fail(rejected("streams are added before the first packet")));
        }
        let count = u32::try_from(self.tracks.len()).unwrap_or(u32::MAX);
        if count >= self.limits.max_tracks {
            return Err(self.fail(MuxError::TooManyTracks {
                count: count.saturating_add(1),
                limit: self.limits.max_tracks,
            }));
        }
        if !stream.timescale.is_clock() {
            return Err(self.fail(MuxError::Clockless));
        }
        let codec = match stream.params.codec() {
            StreamCodec::Known(codec) => *codec,
            StreamCodec::Unknown(tag) => {
                let tag = tag.clone();
                return Err(self.fail(MuxError::UnknownCodec(tag)));
            }
        };
        if let Err(error) = allowed(codec, self.webm_only()) {
            return Err(self.fail(from_crate(error)));
        }
        let audio = stream.params.sample_rate().is_some() && stream.params.channels().is_some();
        if stream.params.size().is_none() && !audio {
            return Err(self.fail(rejected(
                "a track needs a picture size or a sample rate and channels",
            )));
        }
        let id = TrackId::new(count + 1);
        self.tracks.push(Track {
            id,
            codec,
            params: stream.params,
            timescale: stream.timescale,
            language: stream.language,
            config: None,
            trim: None,
            end_padding: 0,
            queue: VecDeque::new(),
            next: 0,
            default_duration: None,
            written: None,
            private_at: None,
            late_private: None,
            previous: None,
            in_cluster: false,
        });
        Ok(id)
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<(), MuxError> {
        self.building()?;
        let index = self
            .track(packet.stream())
            .map_err(|error| self.fail(error))?;
        let track = &self.tracks[index];
        if packet.timescale() != track.timescale {
            let error = MuxError::TimescaleMismatch {
                packet: packet.timescale().ticks_per_second(),
                stream: track.timescale.ticks_per_second(),
            };
            return Err(self.fail(error));
        }
        let timestamp = packet.timestamp();
        let (Some(_), Some(duration)) = (timestamp.presentation_or_decode(), timestamp.duration())
        else {
            return Err(self.fail(MuxError::MissingTimestamp));
        };
        let bytes = len(packet.bytes());
        if bytes > self.limits.max_packet_bytes {
            return Err(self.fail(MuxError::HostileSize {
                declared: bytes,
                limit: self.limits.max_packet_bytes,
            }));
        }
        let track = &mut self.tracks[index];
        track.next = track.next.saturating_add(duration);
        track.queue.push_back(Held {
            packet: packet.clone(),
            padding: 0,
            tail: track.end_padding > 0,
        });
        self.held = self.held.saturating_add(bytes);
        self.drain(false).map_err(|error| self.fail(error))
    }

    fn interleave(&mut self) -> Result<(), MuxError> {
        self.building()
    }

    fn cancel(&mut self) {
        if self.status == MuxStatus::Building {
            self.fail(MuxError::Cancelled);
        }
    }

    fn finish(&mut self, durations: &[TrackDuration]) -> Result<FinishedMux, FailedMux> {
        if let Err(error) = self.building() {
            return Err(FailedMux::incomplete(self.payload, error));
        }
        for index in 0..self.tracks.len() {
            let track = &self.tracks[index];
            let Some(duration) = durations.iter().find(|duration| duration.track == track.id)
            else {
                let error = self.fail(MuxError::Incomplete);
                return Err(FailedMux::incomplete(self.payload, error));
            };
            if duration.ticks != track.next {
                let error = self.fail(MuxError::DurationMismatch {
                    track: track.id.get(),
                });
                return Err(FailedMux::incomplete(self.payload, error));
            }
        }
        if let Err(error) = self.finish_file() {
            let error = self.fail(error);
            return Err(FailedMux::incomplete(self.payload, error));
        }
        self.status = MuxStatus::Finished;
        Ok(FinishedMux::complete(
            durations.to_vec(),
            IndexPlace::End,
            MuxMode::Progressive,
            true,
            Vec::new(),
            self.payload,
        ))
    }

    fn status(&self) -> MuxStatus {
        self.status
    }

    fn set_config(&mut self, track: TrackId, config: &[u8]) -> Result<(), MuxError> {
        self.building()?;
        let index = self.track(track.get()).map_err(|error| self.fail(error))?;
        let header_written = self.layout.is_some();
        let codec = self.tracks[index].codec;
        if !header_written {
            self.tracks[index].config = Some(config.to_vec());
            return Ok(());
        }
        // After the Tracks element: the same configuration is fine, and one
        // of the same length (FLAC's final STREAMINFO) is patched at finish.
        let private = match codec {
            CodecId::Aac => aac_config(config),
            CodecId::Opus => opus_head(config),
            _ => Ok(config.to_vec()),
        }
        .and_then(|private| written(codec, &private, self.webm_only()))
        .map_err(|error| self.fail(from_crate(error)))?
        .private;
        let current = self.tracks[index]
            .written
            .as_ref()
            .map(|written| written.private.clone())
            .unwrap_or_default();
        if private == current {
            return Ok(());
        }
        if private.len() == current.len() && self.tracks[index].private_at.is_some() {
            self.tracks[index].late_private = Some(private);
            return Ok(());
        }
        Err(self.fail(rejected(
            "a codec configuration changed after the Tracks element was written",
        )))
    }

    fn set_trim(&mut self, track: TrackId, trim: Trim) -> Result<(), MuxError> {
        self.building()?;
        let index = self.track(track.get()).map_err(|error| self.fail(error))?;
        let entry = &mut self.tracks[index];
        // An Opus packet's own TOC says what of it is padding; a stated
        // padding only makes the last packet's exact where its millisecond
        // duration rounds.
        if !entry.is_video() {
            entry.end_padding = trim.trailing_padding().unwrap_or(0);
        }
        entry.trim = Some(trim);
        Ok(())
    }
}
