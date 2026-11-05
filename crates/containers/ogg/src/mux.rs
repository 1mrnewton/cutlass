//! The Ogg Opus muxer.
//!
//! Pages go to the writer as they fill, so a long export does not hold its
//! packets in memory. The first packet of any stream writes every stream's
//! beginning-of-stream page (`OpusHead` alone, granule 0), then every
//! stream's `OpusTags` page (granule 0), as RFC 3533 grouping and RFC 7845
//! section 3 require. Streams are added and configured before that.
//!
//! Page policy: an audio page is closed before the next packet once its
//! body holds [`MUX_PAGE_BYTES`] bytes or its completed packets span
//! [`MUX_PAGE_SAMPLES`] samples (one second), and whenever its 255 lacing
//! values are used. A packet that does not fit continues on the next page.
//! A page's granule position is the end time (pts plus duration) of the
//! last packet completed on it, or -1 when none completes. The last page of
//! each stream is written by [`Mux::finish`] with the end-of-stream flag, so
//! a final packet shorter than its TOC duration becomes RFC 7845 end
//! trimming. A stream whose packets last less than its pre-skip is refused
//! at finish, because no reader can play it.
//!
//! Serial numbers are deterministic: the first stream takes
//! [`FIRST_SERIAL`] (or the value given to [`OggMux::with_first_serial`])
//! and each later stream the next number. Page sequence numbers start at 0
//! per stream.

use std::io::Write;

use cutlass_codec::{CodecId, ContainerId, Packet, StreamCodec};
use cutlass_container::{
    FailedMux, FinishedMux, IndexPlace, Limits, Mux, MuxError, MuxMode, MuxStatus, NewStream,
    TrackDuration, TrackId,
};

use crate::limits::{MAX_OPUS_PACKET_PER_STREAM, MUX_PAGE_BYTES, MUX_PAGE_SAMPLES, OPUS_RATE};
use crate::opus::{HEAD_MAGIC, OpusHead, OpusTags, packet_samples};
use crate::page::{FLAG_BOS, FLAG_CONTINUED, FLAG_EOS, encode};

/// Serial number of the first stream: the ASCII bytes `cutl`.
pub const FIRST_SERIAL: u32 = 0x6375_746C;

/// Vendor string written in `OpusTags`.
pub const VENDOR: &str = "cutlass";

/// One stream being written.
struct Track {
    id: TrackId,
    serial: u32,
    channels: u16,
    head: Option<Vec<u8>>,
    streams: u8,
    pre_skip: u16,
    sequence: u32,
    /// Lacing values and body of the page being filled.
    lacing: Vec<u8>,
    body: Vec<u8>,
    /// The page's first piece continues a packet from the previous page.
    continued: bool,
    /// End of the last packet completed on this page, or -1.
    granule: i64,
    /// Samples of packets completed on this page.
    page_samples: i64,
    /// End of the last packet written.
    next: Option<i64>,
    /// The last packet was shorter than its TOC: only the final packet may be.
    trimmed: bool,
    /// Sum of packet durations.
    sum: i64,
    /// Start of the first packet.
    first: Option<i64>,
    /// Audio pages written with a completed packet.
    completed_pages: u32,
    /// Where the last packet started: page sequence, lacing index, body
    /// offset, and that page's granule and samples before it.
    last_packet: Option<(u32, usize, usize, i64, i64)>,
}

impl Track {
    fn page_full(&self) -> bool {
        self.lacing.len() >= 255
            || (self.granule != -1
                && (self.body.len() >= MUX_PAGE_BYTES || self.page_samples >= MUX_PAGE_SAMPLES))
    }
}

/// Writes Ogg Opus.
pub struct OggMux<W: Write> {
    writer: W,
    limits: Limits,
    first_serial: u32,
    tracks: Vec<Track>,
    status: MuxStatus,
    failure: Option<MuxError>,
    headers: bool,
    payload: u64,
    /// Every packet so far started at or after the previous one, across tracks.
    in_order: bool,
    last_time: Option<i64>,
}

impl<W: Write> OggMux<W> {
    #[must_use]
    pub fn new(writer: W) -> Self {
        Self::with_first_serial(writer, FIRST_SERIAL)
    }

    /// A muxer whose first stream takes `serial`. Chained files need a
    /// different serial in each link.
    #[must_use]
    pub fn with_first_serial(writer: W, serial: u32) -> Self {
        Self {
            writer,
            limits: Limits::standard(),
            first_serial: serial,
            tracks: Vec::new(),
            status: MuxStatus::Building,
            failure: None,
            headers: false,
            payload: 0,
            in_order: true,
            last_time: None,
        }
    }

    /// The writer. A mux that did not finish leaves a stream with no
    /// end-of-stream page.
    pub fn into_inner(self) -> W {
        self.writer
    }

    fn fail(&mut self, error: MuxError) -> MuxError {
        if self.status == MuxStatus::Building {
            self.status = MuxStatus::Failed;
            self.failure = Some(error.clone());
        }
        error
    }

    fn building(&self) -> Result<(), MuxError> {
        match self.status {
            MuxStatus::Building => Ok(()),
            MuxStatus::Failed => Err(self.failure.clone().unwrap_or(MuxError::NotBuilding)),
            MuxStatus::Finished => Err(MuxError::AlreadyFinished),
        }
    }

    fn track_index(&self, id: u32) -> Result<usize, MuxError> {
        self.tracks
            .iter()
            .position(|track| track.id.get() == id)
            .ok_or(MuxError::UnknownTrack { id })
    }

    fn emit(&mut self, bytes: &[u8]) -> Result<(), MuxError> {
        self.writer
            .write_all(bytes)
            .map_err(|error| rejected(format!("ogg write failed: {error}")))
    }

    /// Writes every stream's beginning-of-stream page, then every stream's
    /// comment header page.
    fn write_headers(&mut self) -> Result<(), MuxError> {
        if self.headers {
            return Ok(());
        }
        let mut pages = Vec::new();
        for track in &mut self.tracks {
            let Some(head) = &track.head else {
                return Err(rejected(format!(
                    "track {} has no OpusHead: give it in the stream's extradata or set_config",
                    track.id.get()
                )));
            };
            let lacing = crate::page::lacing_for(head.len());
            pages.push(encode(FLAG_BOS, 0, track.serial, 0, &lacing, head));
            track.sequence = 1;
        }
        let tags = OpusTags {
            vendor: VENDOR.to_owned(),
            comments: Vec::new(),
            cut: false,
        }
        .encode();
        for track in &mut self.tracks {
            let lacing = crate::page::lacing_for(tags.len());
            pages.push(encode(0, 0, track.serial, track.sequence, &lacing, &tags));
            track.sequence += 1;
        }
        for page in pages {
            self.emit(&page)?;
        }
        self.headers = true;
        Ok(())
    }

    /// Closes the page being filled on track `index`.
    fn flush_page(&mut self, index: usize, eos: bool) -> Result<(), MuxError> {
        let track = &mut self.tracks[index];
        let mut flags = 0;
        if track.continued {
            flags |= FLAG_CONTINUED;
        }
        if eos {
            flags |= FLAG_EOS;
        }
        let granule = if eos && track.granule == -1 {
            track.next.unwrap_or(0)
        } else {
            track.granule
        };
        let page = encode(
            flags,
            granule,
            track.serial,
            track.sequence,
            &track.lacing,
            &track.body,
        );
        track.sequence = track.sequence.wrapping_add(1);
        if track.granule != -1 {
            track.completed_pages = track.completed_pages.saturating_add(1);
        }
        track.continued = track.lacing.last() == Some(&255);
        track.lacing.clear();
        track.body.clear();
        track.granule = -1;
        track.page_samples = 0;
        self.emit(&page)
    }

    /// RFC 7845 section 4.5 reads the first audio page with a completed
    /// packet, when it is also the last page, as starting at its granule
    /// minus its samples, or at 0 when the granule is smaller. A stream
    /// that starts after 0 and trims its end cannot say both on that one
    /// page, so the packets before the trimmed one go on a page of their
    /// own. One trimmed packet alone, starting after 0, cannot be written.
    fn separate_trim_from_start(&mut self, index: usize) -> Result<(), MuxError> {
        let track = &mut self.tracks[index];
        if track.completed_pages > 0 || !track.trimmed || track.first.is_none_or(|first| first == 0)
        {
            return Ok(());
        }
        let Some((sequence, lacing_at, body_at, granule, samples)) = track.last_packet else {
            return Ok(());
        };
        // Only the last packet completes here when it began on an earlier
        // page or opens this one.
        if sequence != track.sequence || lacing_at == 0 {
            return Err(rejected(format!(
                "Opus stream {} is one trimmed packet starting at {}; Ogg reads that as starting at 0",
                track.id.get(),
                track.first.unwrap_or(0)
            )));
        }
        let tail_lacing = track.lacing.split_off(lacing_at);
        let tail_body = track.body.split_off(body_at);
        let end = track.granule;
        let tail_samples = track.page_samples.saturating_sub(samples);
        track.granule = granule;
        track.page_samples = samples;
        self.flush_page(index, false)?;
        let track = &mut self.tracks[index];
        track.lacing = tail_lacing;
        track.body = tail_body;
        track.granule = end;
        track.page_samples = tail_samples;
        Ok(())
    }

    /// Adds one packet's lacing and bytes, closing pages as they fill.
    fn lay(&mut self, index: usize, data: &[u8], end: i64, samples: i64) -> Result<(), MuxError> {
        if self.tracks[index].page_full() {
            self.flush_page(index, false)?;
        }
        let lacing = crate::page::lacing_for(data.len());
        let mut offset = 0;
        let mut starting = true;
        let mut values = lacing.iter().peekable();
        while let Some(&value) = values.next() {
            if self.tracks[index].lacing.len() >= 255 {
                self.flush_page(index, false)?;
            }
            let track = &mut self.tracks[index];
            if starting {
                track.last_packet = Some((
                    track.sequence,
                    track.lacing.len(),
                    track.body.len(),
                    track.granule,
                    track.page_samples,
                ));
                starting = false;
            }
            let len = usize::from(value);
            track.lacing.push(value);
            track.body.extend_from_slice(&data[offset..offset + len]);
            offset += len;
            if values.peek().is_none() {
                track.granule = end;
                track.page_samples = track.page_samples.saturating_add(samples);
            }
        }
        Ok(())
    }

    fn check_packet(&self, index: usize, packet: &Packet) -> Result<(i64, i64, i64), MuxError> {
        let track = &self.tracks[index];
        if packet.timescale().ticks_per_second() != OPUS_RATE {
            return Err(MuxError::TimescaleMismatch {
                packet: packet.timescale().ticks_per_second(),
                stream: OPUS_RATE,
            });
        }
        let timestamp = packet.timestamp();
        let (Some(pts), Some(duration)) =
            (timestamp.presentation_or_decode(), timestamp.duration())
        else {
            return Err(MuxError::MissingTimestamp);
        };
        let size = u64::try_from(packet.bytes().len()).unwrap_or(u64::MAX);
        let cap = MAX_OPUS_PACKET_PER_STREAM
            .saturating_mul(u64::from(track.streams))
            .min(self.limits.max_packet_bytes);
        if size > cap {
            return Err(MuxError::HostileSize {
                declared: size,
                limit: cap,
            });
        }
        let Some(samples) = packet_samples(packet.bytes()) else {
            return Err(rejected("an Opus packet has a malformed TOC".to_owned()));
        };
        let samples = i64::from(samples);
        if duration > samples {
            return Err(rejected(format!(
                "packet duration {duration} is longer than its TOC's {samples} samples"
            )));
        }
        if track.trimmed {
            return Err(rejected(
                "only the last packet of a stream may be shorter than its TOC".to_owned(),
            ));
        }
        match track.next {
            Some(next) if pts < next => {
                return Err(rejected(format!(
                    "packet at {pts} starts before the previous one ends at {next}"
                )));
            }
            Some(next) if pts > next => {
                return Err(rejected(format!(
                    "packet at {pts} leaves a gap after {next}; Ogg Opus needs packets that fill it"
                )));
            }
            None if pts < 0 => {
                return Err(rejected(format!("first packet starts at {pts}")));
            }
            _ => {}
        }
        Ok((pts, duration, samples))
    }

    fn store_config(&mut self, index: usize, config: &[u8]) -> Result<(), MuxError> {
        let size = u64::try_from(config.len()).unwrap_or(u64::MAX);
        if size > self.limits.max_extradata_bytes {
            return Err(MuxError::HostileSize {
                declared: size,
                limit: self.limits.max_extradata_bytes,
            });
        }
        let head = opus_head(config)?;
        let parsed = OpusHead::parse(&head, self.tracks[index].serial)
            .map_err(|error| rejected(error.to_string()))?;
        if u16::from(parsed.channels) != self.tracks[index].channels {
            return Err(rejected(format!(
                "OpusHead has {} channels and the stream has {}",
                parsed.channels, self.tracks[index].channels
            )));
        }
        self.tracks[index].streams = parsed.streams;
        self.tracks[index].pre_skip = parsed.pre_skip;
        self.tracks[index].head = Some(head);
        Ok(())
    }

    fn finish_inner(&mut self, durations: &[TrackDuration]) -> Result<(), MuxError> {
        for track in &self.tracks {
            let Some(duration) = durations.iter().find(|item| item.track == track.id) else {
                return Err(MuxError::Incomplete);
            };
            if duration.ticks < 0 {
                return Err(MuxError::NegativeDuration);
            }
            if duration.timescale.ticks_per_second() != OPUS_RATE {
                return Err(MuxError::TimescaleMismatch {
                    packet: duration.timescale.ticks_per_second(),
                    stream: OPUS_RATE,
                });
            }
            if duration.ticks != track.sum {
                return Err(MuxError::DurationMismatch {
                    track: track.id.get(),
                });
            }
            // RFC 7845 section 4.5: the pre-skip is decoded and dropped, so a
            // stream must last at least that long.
            if track.next.is_some() && track.sum < i64::from(track.pre_skip) {
                return Err(rejected(format!(
                    "Opus stream {} lasts {} samples, less than its pre-skip of {}",
                    track.id.get(),
                    track.sum,
                    track.pre_skip
                )));
            }
        }
        if durations.len() != self.tracks.len() {
            return Err(MuxError::Incomplete);
        }
        self.write_headers()?;
        for index in 0..self.tracks.len() {
            self.separate_trim_from_start(index)?;
            self.flush_page(index, true)?;
        }
        self.writer
            .flush()
            .map_err(|error| rejected(format!("ogg flush failed: {error}")))
    }
}

/// `OpusHead` from a configuration: an `OpusHead` packet as stored in Ogg,
/// or an ISOBMFF `dOps` body (big-endian), rewritten as `OpusHead`.
fn opus_head(config: &[u8]) -> Result<Vec<u8>, MuxError> {
    if config.starts_with(HEAD_MAGIC) {
        return Ok(config.to_vec());
    }
    // dOps: Version, OutputChannelCount, PreSkip, InputSampleRate,
    // OutputGain, ChannelMappingFamily, then the mapping table.
    if config.len() < 11 || config[0] != 0 {
        return Err(rejected(
            "codec configuration is neither OpusHead nor dOps".to_owned(),
        ));
    }
    let mut head = HEAD_MAGIC.to_vec();
    head.push(1);
    head.push(config[1]);
    head.extend_from_slice(&u16::from_be_bytes([config[2], config[3]]).to_le_bytes());
    head.extend_from_slice(
        &u32::from_be_bytes([config[4], config[5], config[6], config[7]]).to_le_bytes(),
    );
    head.extend_from_slice(&i16::from_be_bytes([config[8], config[9]]).to_le_bytes());
    head.push(config[10]);
    head.extend_from_slice(&config[11..]);
    Ok(head)
}

fn rejected(message: String) -> MuxError {
    MuxError::Rejected { message }
}

impl<W: Write> Mux for OggMux<W> {
    fn family(&self) -> ContainerId {
        ContainerId::Ogg
    }

    fn add_stream(&mut self, stream: NewStream) -> Result<TrackId, MuxError> {
        self.building()?;
        if self.headers {
            return Err(self.fail(rejected(
                "Ogg writes every beginning-of-stream page first; add streams before packets"
                    .to_owned(),
            )));
        }
        match stream.params.codec() {
            StreamCodec::Known(CodecId::Opus) => {}
            StreamCodec::Unknown(unknown) => {
                return Err(self.fail(MuxError::UnknownCodec(unknown.clone())));
            }
            StreamCodec::Known(other) => {
                return Err(self.fail(rejected(format!(
                    "this Ogg muxer writes the Opus mapping only, not {other:?}"
                ))));
            }
        }
        if stream.timescale.ticks_per_second() != OPUS_RATE {
            return Err(self.fail(MuxError::TimescaleMismatch {
                packet: stream.timescale.ticks_per_second(),
                stream: OPUS_RATE,
            }));
        }
        let Some(channels) = stream.params.channels() else {
            return Err(self.fail(rejected("an Opus stream needs a channel count".to_owned())));
        };
        let count = u32::try_from(self.tracks.len()).unwrap_or(u32::MAX);
        if count >= self.limits.max_tracks {
            return Err(self.fail(MuxError::TooManyTracks {
                count,
                limit: self.limits.max_tracks,
            }));
        }
        let id = TrackId::new(count);
        self.tracks.push(Track {
            id,
            serial: self.first_serial.wrapping_add(count),
            channels,
            head: None,
            streams: 1,
            pre_skip: 0,
            sequence: 0,
            lacing: Vec::new(),
            body: Vec::new(),
            continued: false,
            granule: -1,
            page_samples: 0,
            next: None,
            trimmed: false,
            sum: 0,
            first: None,
            completed_pages: 0,
            last_packet: None,
        });
        let index = self.tracks.len() - 1;
        if !stream.params.extradata().is_empty()
            && let Err(error) = self.store_config(index, stream.params.extradata())
        {
            self.tracks.pop();
            return Err(self.fail(error));
        }
        Ok(id)
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<(), MuxError> {
        self.building()?;
        let index = self
            .track_index(packet.stream())
            .map_err(|error| self.fail(error))?;
        let (pts, duration, samples) = self
            .check_packet(index, packet)
            .map_err(|error| self.fail(error))?;
        if let Err(error) = self.write_headers() {
            return Err(self.fail(error));
        }
        let end = pts.saturating_add(duration);
        if let Err(error) = self.lay(index, packet.bytes(), end, samples) {
            return Err(self.fail(error));
        }
        let track = &mut self.tracks[index];
        if track.next.is_none() {
            track.first = Some(pts);
        }
        track.next = Some(end);
        track.trimmed = duration < samples;
        track.sum = track.sum.saturating_add(duration);
        if self.last_time.is_some_and(|last| pts < last) {
            self.in_order = false;
        }
        self.last_time = Some(self.last_time.map_or(pts, |last| last.max(pts)));
        self.payload = self
            .payload
            .saturating_add(u64::try_from(packet.bytes().len()).unwrap_or(u64::MAX));
        Ok(())
    }

    /// Pages are written as packets arrive, in arrival order. This checks
    /// the mux is still building and moves nothing.
    fn interleave(&mut self) -> Result<(), MuxError> {
        self.building()
    }

    fn cancel(&mut self) {
        self.fail(MuxError::Cancelled);
    }

    fn finish(&mut self, durations: &[TrackDuration]) -> Result<FinishedMux, FailedMux> {
        if let Err(error) = self.building() {
            return Err(FailedMux::incomplete(self.payload, error));
        }
        if let Err(error) = self.finish_inner(durations) {
            let error = self.fail(error);
            return Err(FailedMux::incomplete(self.payload, error));
        }
        self.status = MuxStatus::Finished;
        Ok(FinishedMux::complete(
            durations.to_vec(),
            IndexPlace::Absent,
            MuxMode::Progressive,
            self.in_order,
            Vec::new(),
            self.payload,
        ))
    }

    fn status(&self) -> MuxStatus {
        self.status
    }

    /// Stores the `OpusHead` (or an MP4 `dOps` body) an encoder hands over.
    /// It must arrive before the first packet of any stream.
    fn set_config(&mut self, track: TrackId, config: &[u8]) -> Result<(), MuxError> {
        self.building()?;
        let index = self
            .track_index(track.get())
            .map_err(|error| self.fail(error))?;
        if self.headers {
            return Err(self.fail(rejected(
                "the OpusHead page is already written; set the configuration before packets"
                    .to_owned(),
            )));
        }
        self.store_config(index, config)
            .map_err(|error| self.fail(error))
    }
}
