//! The CAF demuxer.
//!
//! Open walks the chunks and decodes the packet table; it does not read
//! the sound. Each read returns one packet; linear PCM and other formats
//! of one frame per packet are grouped into whole frames up to
//! [`PCM_PACKET_BYTES`]. Timestamps are
//! frames at the `desc` sample rate, counted from the first frame in the
//! file: priming frames are not subtracted. They are exposed as the
//! stream's [`Trim`] and as one edit. A seek is arithmetic for constant
//! packets and uses the table's checkpoints for variable ones; it reads no
//! sound bytes.
//!
//! A file that stops inside `data` plays the whole packets it holds: the
//! frames that are there for constant packets, and the rows of the packet
//! table whose bytes are there for variable ones. [`CafDemux::is_cut`]
//! says so.

use std::io::{BufReader, Read, Seek};

use cutlass_codec::{ContainerId, Packet, StreamParams, Timescale, Timestamp};
use cutlass_container::{
    Access, Demux, Edit, Gap, Input, Landing, Limits, OpenError, ReadError, SeekError, SeekLanding,
    SeekTo, Stream, TrackId, Trim,
};

use crate::chan::ChannelLayout;
use crate::codec;
use crate::desc::Description;
use crate::error::malformed;
use crate::info::InfoEntry;
use crate::kept::KeptChunk;
use crate::limits::PCM_PACKET_BYTES;
use crate::pakt::{PacketTableHeader, Packets};
use crate::walk::{self, Chunk, DataChunk, Scanned};

/// Bytes the demuxer buffers ahead of the parser: libavformat's own
/// buffer size. Each seek refills it, so 64 KiB made a seek and one packet
/// read cost 1.03 to 1.23 times libavformat's instructions; at 32 KiB it is
/// 0.73 to 0.91, and a whole read goes from 0.20-0.67 to 0.21-0.77.
const READ_BUFFER: usize = 32 * 1024;

/// One opened CAF file.
pub struct CafDemux<R> {
    input: Input<BufReader<R>>,
    streams: Vec<Stream>,
    desc: Description,
    cookie: Option<Vec<u8>>,
    pakt: Option<PacketTableHeader>,
    chan: Option<ChannelLayout>,
    info: Vec<InfoEntry>,
    kept: Vec<KeptChunk>,
    chunks: Vec<Chunk>,
    stopped_at: Option<u64>,
    data: DataChunk,
    packets: Packets,
    /// Packets returned together when each is one frame.
    group: u64,
    /// A forward input whose `data` runs to an unknown end.
    unbounded: bool,
    cut: bool,
    next: u64,
    offset: u64,
    time: u64,
    ended: bool,
    cancelled: bool,
}

impl<R: Read + Seek> CafDemux<R> {
    /// Opens a random-access file. `length` is the file size, not an allocation.
    pub fn open(reader: R, length: u64) -> Result<Self, OpenError> {
        Self::open_limited(reader, length, Limits::standard())
    }

    /// [`Self::open`] with caller bounds.
    pub fn open_limited(reader: R, length: u64, limits: Limits) -> Result<Self, OpenError> {
        let input = Input::random(
            BufReader::with_capacity(READ_BUFFER, reader),
            length,
            limits,
        );
        Self::from_input(input)
    }

    /// Opens a reader that is read once, in order. Chunks after `data` are
    /// not read: a variable-packet file whose `pakt` follows `data` is
    /// refused. A `data` size of -1 reads to the end of the stream. Seek is
    /// refused.
    pub fn open_forward(reader: R) -> Result<Self, OpenError> {
        Self::open_forward_limited(reader, Limits::standard())
    }

    /// [`Self::open_forward`] with caller bounds.
    pub fn open_forward_limited(reader: R, limits: Limits) -> Result<Self, OpenError> {
        let input = Input::forward(BufReader::with_capacity(READ_BUFFER, reader), limits);
        Self::from_input(input)
    }

    fn from_input(mut input: Input<BufReader<R>>) -> Result<Self, OpenError> {
        let mut scanned = walk::scan(&mut input)?;
        let unbounded = scanned.data.len.is_none() && scanned.desc.is_constant();
        let (packets, cut) = plan(&mut scanned, &input)?;
        let stream = stream_of(&scanned, &packets, unbounded)?;
        let group = if scanned.desc.is_constant() && scanned.desc.frames_per_packet == 1 {
            let limit = PCM_PACKET_BYTES.min(input.limits().max_packet_bytes);
            (limit / u64::from(scanned.desc.bytes_per_packet)).max(1)
        } else {
            1
        };
        Ok(Self {
            input,
            streams: vec![stream],
            desc: scanned.desc,
            cookie: scanned.cookie,
            pakt: scanned.pakt,
            chan: scanned.chan,
            info: scanned.info,
            kept: scanned.kept,
            chunks: scanned.chunks,
            stopped_at: scanned.stopped_at,
            data: scanned.data,
            packets,
            group,
            unbounded,
            cut,
            next: 0,
            offset: 0,
            time: 0,
            ended: false,
            cancelled: false,
        })
    }

    /// The `desc` chunk.
    #[must_use]
    pub const fn description(&self) -> &Description {
        &self.desc
    }

    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        self.desc.sample_rate
    }

    /// The `kuki` chunk as stored. The stream extradata is what the codec reads.
    #[must_use]
    pub fn cookie(&self) -> Option<&[u8]> {
        self.cookie.as_deref()
    }

    /// The `pakt` header, when the file has one.
    #[must_use]
    pub const fn packet_table(&self) -> Option<&PacketTableHeader> {
        self.pakt.as_ref()
    }

    #[must_use]
    pub const fn channel_layout(&self) -> Option<&ChannelLayout> {
        self.chan.as_ref()
    }

    /// The `info` strings, in file order.
    #[must_use]
    pub fn info(&self) -> &[InfoEntry] {
        &self.info
    }

    /// The `mark`, `regn`, `strg`, `uuid` and `midi` chunks, in file order,
    /// as stored, for a remux to hand to [`crate::CafMux::add_chunk`]. One
    /// with less than its fixed head, or past [`crate::MAX_KEPT_BYTES`] in
    /// all, is left out. A forward input stops at `data` and keeps only
    /// those before it.
    #[must_use]
    pub fn kept_chunks(&self) -> &[KeptChunk] {
        &self.kept
    }

    /// Every chunk the walk read a header of, in file order.
    #[must_use]
    pub fn chunks(&self) -> &[Chunk] {
        &self.chunks
    }

    /// Where the walk stopped at a chunk header after `data` that claimed
    /// more bytes than the file has. `None` when every chunk fit.
    #[must_use]
    pub const fn stopped_at(&self) -> Option<u64> {
        self.stopped_at
    }

    /// File offset of the first sound byte.
    #[must_use]
    pub const fn data_offset(&self) -> u64 {
        self.data.start
    }

    /// `mEditCount` of the `data` chunk.
    #[must_use]
    pub const fn edit_count(&self) -> u32 {
        self.data.edit_count
    }

    /// True when the `data` size was -1.
    #[must_use]
    pub const fn is_open_ended(&self) -> bool {
        self.data.open_ended
    }

    /// True when the file holds less sound than its header states: `data`
    /// runs past the end of the file, or the packet table counts packets
    /// whose bytes are not there. What is there plays. A forward reader
    /// learns this only when the input ends.
    #[must_use]
    pub const fn is_cut(&self) -> bool {
        self.cut
    }

    /// CAF packets in the file, or `None` for a forward input whose `data`
    /// runs to an unknown end.
    #[must_use]
    pub fn packet_count(&self) -> Option<u64> {
        (!self.unbounded).then(|| self.packets.count())
    }

    /// Frames in every packet, priming and remainder included.
    #[must_use]
    pub fn total_frames(&self) -> Option<u64> {
        (!self.unbounded).then(|| self.packets.total_frames())
    }

    fn place(&mut self, at: u64) -> Result<(), ReadError> {
        if self.input.position() == at {
            return Ok(());
        }
        self.input.seek(at).map_err(ReadError::from_packet)
    }

    fn read_next(&mut self) -> Result<Option<Packet>, ReadError> {
        if self.cancelled {
            return Err(ReadError::Cancelled);
        }
        if self.unbounded {
            return self.read_unbounded();
        }
        let count = self.packets.count();
        // An entry of no bytes and no frames carries nothing: ffmpeg ends
        // its FLAC tables with one, which libavformat stops at and
        // AudioToolbox counts. It is skipped, not handed to a decoder.
        while self.next < count && self.packets.entry(self.next) == (0, 0) {
            self.next += 1;
        }
        if self.next >= count {
            return Ok(None);
        }
        let (bytes, frames, taken) = if self.group > 1 {
            let taken = self.group.min(count - self.next);
            let (bytes, frames) = self.packets.entry(self.next);
            (u64::from(bytes) * taken, u64::from(frames) * taken, taken)
        } else {
            let (bytes, frames) = self.packets.entry(self.next);
            (u64::from(bytes), u64::from(frames), 1)
        };
        let at = self
            .data
            .start
            .checked_add(self.offset)
            .ok_or_else(|| source("packet offset does not fit"))?;
        self.place(at)?;
        if self.input.access() == Access::Forward && self.group > 1 {
            // A forward input may end before the size it stated.
            return self.read_run(bytes, taken);
        }
        let payload = self
            .input
            .read_limited(bytes, self.input.limits().max_packet_bytes)
            .map_err(ReadError::from_packet)?;
        let packet = self.packet(payload, frames)?;
        self.next += taken;
        self.offset += bytes;
        self.time += frames;
        Ok(Some(packet))
    }

    /// `asked` one-frame packets of a forward input, or the whole ones
    /// that arrive before it ends.
    fn read_run(&mut self, bytes: u64, asked: u64) -> Result<Option<Packet>, ReadError> {
        let limit = self.input.limits().max_packet_bytes;
        if bytes > limit {
            return Err(ReadError::HostileSize {
                declared: bytes,
                limit,
            });
        }
        let want = usize::try_from(bytes).map_err(|_| source("packet size does not fit"))?;
        let payload = self.fill(want)?;
        let (each, frames) = self.packets.entry(0);
        let each = u64::from(each).max(1);
        let taken = crate::bytes::len_u64(payload.len()) / each;
        if taken < asked {
            self.cut = true;
            self.next = self.packets.count();
        }
        if taken == 0 {
            return Ok(None);
        }
        let mut payload = payload;
        payload.truncate(usize::try_from(taken * each).unwrap_or(0));
        let frames = u64::from(frames) * taken;
        let packet = self.packet(payload, frames)?;
        if taken == asked {
            self.next += taken;
        }
        self.offset += taken * each;
        self.time += frames;
        Ok(Some(packet))
    }

    /// Up to `want` bytes: fewer when the input ends first.
    fn fill(&mut self, want: usize) -> Result<Vec<u8>, ReadError> {
        let mut payload = Vec::new();
        payload
            .try_reserve_exact(want)
            .map_err(|_| source("out of memory"))?;
        payload.resize(want, 0);
        let mut filled = 0;
        while filled < want {
            let got = self
                .input
                .read(&mut payload[filled..])
                .map_err(ReadError::from_packet)?;
            if got == 0 {
                break;
            }
            filled += got;
        }
        payload.truncate(filled);
        Ok(payload)
    }

    /// A forward read of constant packets up to the end of the stream.
    fn read_unbounded(&mut self) -> Result<Option<Packet>, ReadError> {
        if self.ended {
            return Ok(None);
        }
        let (bytes, frames) = self.packets.entry(0);
        let packet_bytes =
            usize::try_from(bytes).map_err(|_| source("packet size does not fit"))?;
        let want = packet_bytes.saturating_mul(usize::try_from(self.group).unwrap_or(1));
        let mut payload = self.fill(want)?;
        let filled = payload.len();
        if filled < want {
            self.ended = true;
            // The input may end anywhere, but not inside a frame.
            self.cut |= !filled.is_multiple_of(packet_bytes.max(1));
        }
        let whole = filled / packet_bytes.max(1);
        if whole == 0 {
            self.ended = true;
            return Ok(None);
        }
        payload.truncate(whole * packet_bytes);
        let taken = crate::bytes::len_u64(whole);
        let frames = u64::from(frames) * taken;
        let packet = self.packet(payload, frames)?;
        self.next += taken;
        self.offset += crate::bytes::len_u64(whole * packet_bytes);
        self.time += frames;
        Ok(Some(packet))
    }

    fn packet(&self, payload: Vec<u8>, frames: u64) -> Result<Packet, ReadError> {
        let pts = i64::try_from(self.time).map_err(|_| source("packet time does not fit"))?;
        let duration = i64::try_from(frames).map_err(|_| source("packet duration does not fit"))?;
        let timestamp = Timestamp::try_new(Some(pts), Some(pts), Some(duration))
            .map_err(|error| source(error.to_string()))?;
        Packet::try_new(
            0,
            payload,
            timestamp,
            Timescale::new(self.desc.sample_rate),
            true,
        )
        .map_err(|error| source(error.to_string()))
    }

    fn seek_to(&mut self, to: SeekTo) -> Result<SeekLanding, ReadError> {
        if self.cancelled {
            return Err(ReadError::Cancelled);
        }
        if to.track != 0 {
            return Err(ReadError::Seek(SeekError::UnknownTrack));
        }
        if self.input.access() == Access::Forward || self.unbounded {
            return Err(ReadError::Seek(SeekError::ForwardOnly));
        }
        let count = self.packets.count();
        if count == 0 {
            return Err(ReadError::Seek(SeekError::Empty));
        }
        let total = self.packets.total_frames();
        let past_end = to.end.is_some_and(|end| to.time >= end)
            || u64::try_from(to.time).is_ok_and(|time| time >= total);
        if past_end {
            let end_at = self.data.start.saturating_add(self.packets.total_bytes());
            self.place(end_at)?;
            self.next = count;
            self.offset = self.packets.total_bytes();
            self.time = total;
            return Ok(SeekLanding {
                landing: Landing::End,
                index_used: true,
                payload_scans: 0,
                entries_visited: 0,
            });
        }
        let time = u64::try_from(to.time.max(0)).unwrap_or(0);
        // With one frame per packet the landing is the frame itself.
        let located = self.packets.locate(time);
        let at = self.data.start.saturating_add(located.offset);
        self.place(at)?;
        self.next = located.index;
        self.offset = located.offset;
        self.time = located.time;
        let landed = i64::try_from(located.time).map_err(|_| source("seek time does not fit"))?;
        Ok(SeekLanding {
            landing: Landing::Packet {
                ordinal: u32::try_from(located.index).unwrap_or(u32::MAX),
                time: landed,
                keyframe: true,
                preroll: u32::from(landed < to.time),
                byte_position: at,
            },
            index_used: true,
            payload_scans: 0,
            entries_visited: located.visited,
        })
    }
}

fn source(message: impl Into<String>) -> ReadError {
    ReadError::Source {
        message: message.into(),
    }
}

/// The packet layout, fitted to the bytes in `data`, and whether the file
/// holds less than its header states. The table moves out of `scanned`
/// rather than being copied: it is the one large thing an open holds.
fn plan<R: Read>(scanned: &mut Scanned, input: &Input<R>) -> Result<(Packets, bool), OpenError> {
    let desc = &scanned.desc;
    let data_len = scanned.data.len;
    let mut cut = scanned.data.cut;
    if desc.is_constant() {
        let bytes = u64::from(desc.bytes_per_packet);
        let declared = scanned.pakt.map_or(0, |pakt| pakt.packets);
        let count = match data_len {
            Some(len) if declared > 0 => {
                // A table that counts more packets than the bytes hold is
                // a file that stopped: the packets that are there play.
                let held = len / bytes;
                cut |= declared > held;
                declared.min(held)
            }
            Some(len) => len / bytes,
            None => 0,
        };
        let packets = Packets::Constant {
            bytes: desc.bytes_per_packet,
            frames: desc.frames_per_packet,
            count,
        };
        if i64::try_from(packets.total_frames()).is_err()
            || count
                .checked_mul(u64::from(desc.frames_per_packet))
                .is_none()
        {
            return Err(malformed("the frame count does not fit"));
        }
        return Ok((packets, cut));
    }
    let Some(mut table) = scanned.table.take() else {
        if input.access() == Access::Forward {
            return Err(malformed(
                "variable packets need a pakt chunk before data on a forward input",
            ));
        }
        return Err(malformed("variable packets need a pakt chunk"));
    };
    // Rows whose bytes the file does not hold are dropped, not refused.
    if let Some(len) = data_len {
        cut |= table.keep_within(len);
    }
    Ok((Packets::Table(table), cut))
}

fn stream_of(scanned: &Scanned, packets: &Packets, unbounded: bool) -> Result<Stream, OpenError> {
    let desc = &scanned.desc;
    let priming = scanned.pakt.map_or(0, |pakt| pakt.priming_frames);
    let mapped = codec::map(desc, scanned.cookie.as_deref(), priming)?;
    let channels =
        u16::try_from(desc.channels).map_err(|_| malformed("channel count does not fit"))?;
    let params =
        StreamParams::try_audio(mapped.codec, desc.sample_rate, channels, mapped.extradata)
            .map_err(|error| malformed(error.to_string()))?;
    let (trim, edits) = match scanned.pakt {
        Some(pakt) => {
            let trim = Trim::new(
                Some(u64::from(pakt.priming_frames)),
                Some(u64::from(pakt.remainder_frames)),
            );
            let mut edits = Vec::new();
            if pakt.priming_frames > 0 || pakt.remainder_frames > 0 {
                let mut valid = pakt.valid_frames;
                if !unbounded {
                    let after = packets
                        .total_frames()
                        .saturating_sub(u64::from(pakt.priming_frames));
                    valid = valid.min(after);
                }
                let duration =
                    i64::try_from(valid).map_err(|_| malformed("valid frames do not fit"))?;
                let edit = Edit::try_new(Some(i64::from(pakt.priming_frames)), duration, 1, 1)
                    .map_err(|error| malformed(error.to_string()))?;
                edits.push(edit);
            }
            (trim, edits)
        }
        None => (Trim::default(), Vec::new()),
    };
    Ok(Stream::new(
        TrackId::new(0),
        params,
        Timescale::new(desc.sample_rate),
        None,
        edits,
        trim,
    ))
}

impl<R: Read + Seek + Send> Demux for CafDemux<R> {
    fn family(&self) -> ContainerId {
        ContainerId::Caf
    }

    fn streams(&self) -> &[Stream] {
        &self.streams
    }

    fn gaps(&self) -> &[Gap] {
        &[]
    }

    fn read_packet(&mut self) -> Result<Option<Packet>, ReadError> {
        self.read_next()
    }

    fn seek(&mut self, to: SeekTo) -> Result<SeekLanding, ReadError> {
        self.seek_to(to)
    }

    fn cancel(&mut self) {
        self.cancelled = true;
    }

    fn consumed(&self) -> u64 {
        self.input.consumed()
    }

    fn indexed_end(&self, track: TrackId) -> Option<i64> {
        if track.get() != 0 || self.unbounded {
            return None;
        }
        let frames = self.packets.total_frames();
        (frames != 0).then(|| i64::try_from(frames).ok()).flatten()
    }
}
