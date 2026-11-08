//! The Ogg Opus and Ogg Vorbis muxer.
//!
//! Pages go to the writer as they fill, so a long export does not hold its
//! packets in memory. The first packet of any stream writes every stream's
//! beginning-of-stream page (`OpusHead` or the Vorbis identification
//! header, alone, granule 0), then every stream's other headers, as RFC
//! 3533 grouping, RFC 7845 section 3, and Vorbis I section A.2 require.
//! `OpusTags` takes one page. The Vorbis comment and setup headers may span
//! pages, and the setup header ends its page; a header page that completes
//! a packet has granule 0, and one that completes none -1. Streams are
//! added and configured before that.
//!
//! Page policy: an audio page is closed before the next packet once its
//! body holds [`MUX_PAGE_BYTES`] bytes or its completed packets span one
//! second of samples, and whenever its 255 lacing values are used. A packet
//! that does not fit continues on the next page. A page's granule position
//! is the end time (pts plus duration) of the last packet completed on it,
//! or -1 when none completes. The last page of each stream is written by
//! [`Mux::finish`] with the end-of-stream flag, so a final packet shorter
//! than its natural duration (an Opus packet's TOC, or a quarter of each of
//! a Vorbis packet's block and the block before) becomes end trimming. An
//! Opus stream whose packets last less than its pre-skip is refused at
//! finish, because no reader can play it.
//!
//! A Vorbis stream's timescale is its sample rate. Its first packet has
//! duration 0, as a Vorbis decoder returns nothing for it. A stream may
//! start before 0, as a stream copy cut mid-file does: its first audio page
//! stays open until a packet ends at 0 or later, and section A.2 reads the
//! samples before 0 from that page's granule. A stream whose first page
//! would close before 0, at 255 lacing values, is refused.
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

use crate::limits::{MAX_OPUS_PACKET_PER_STREAM, MUX_PAGE_BYTES, OPUS_RATE};
use crate::opus::{HEAD_MAGIC, OpusHead, OpusTags, packet_samples};
use crate::page::{FLAG_BOS, FLAG_CONTINUED, FLAG_EOS, encode};
use crate::vorbis::{self, VorbisComment, VorbisId, VorbisModes, split_config};

/// Serial number of the first stream: the ASCII bytes `cutl`.
pub const FIRST_SERIAL: u32 = 0x6375_746C;

/// Vendor string written in `OpusTags`.
pub const VENDOR: &str = "cutlass";

/// The three Vorbis headers and the modes of the setup header.
struct VorbisHeaders {
    id: Vec<u8>,
    comment: Vec<u8>,
    setup: Vec<u8>,
    modes: VorbisModes,
}

/// What a stream's mapping needs to write it.
enum Mapping {
    Opus {
        head: Option<Vec<u8>>,
        streams: u8,
        pre_skip: u16,
    },
    Vorbis {
        headers: Option<VorbisHeaders>,
        /// Block size of the last packet written.
        previous: Option<u32>,
    },
}

/// One stream being written.
struct Track {
    id: TrackId,
    serial: u32,
    channels: u16,
    /// Samples per second: the stream's timescale.
    rate: u32,
    mapping: Mapping,
    sequence: u32,
    /// Lacing values and body of the page being filled.
    lacing: Vec<u8>,
    body: Vec<u8>,
    /// The page's first piece continues a packet from the previous page.
    continued: bool,
    /// End of the last packet completed on this page, or -1.
    granule: i64,
    /// A packet completes on this page.
    completes: bool,
    /// Samples of packets completed on this page.
    page_samples: i64,
    /// End of the last packet written.
    next: Option<i64>,
    /// The last packet was shorter than its natural duration: only the
    /// final packet may be.
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
            || (self.completes
                && self.granule >= 0
                && (self.body.len() >= MUX_PAGE_BYTES || self.page_samples >= i64::from(self.rate)))
    }

    const fn mapping_name(&self) -> &'static str {
        match self.mapping {
            Mapping::Opus { .. } => "Opus",
            Mapping::Vorbis { .. } => "Vorbis",
        }
    }
}

/// Writes Ogg Opus and Ogg Vorbis.
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
    /// other header pages.
    fn write_headers(&mut self) -> Result<(), MuxError> {
        if self.headers {
            return Ok(());
        }
        let mut pages = Vec::new();
        for track in &mut self.tracks {
            let first = match &track.mapping {
                Mapping::Opus {
                    head: Some(head), ..
                } => head,
                Mapping::Vorbis {
                    headers: Some(headers),
                    ..
                } => &headers.id,
                Mapping::Opus { head: None, .. } | Mapping::Vorbis { headers: None, .. } => {
                    return Err(missing_headers(track));
                }
            };
            let lacing = crate::page::lacing_for(first.len());
            pages.push(encode(FLAG_BOS, 0, track.serial, 0, &lacing, first));
            track.sequence = 1;
        }
        for page in pages {
            self.emit(&page)?;
        }
        let tags = OpusTags {
            vendor: VENDOR.to_owned(),
            comments: Vec::new(),
            cut: false,
        }
        .encode();
        for index in 0..self.tracks.len() {
            let packets = match &self.tracks[index].mapping {
                Mapping::Opus { .. } => vec![tags.clone()],
                Mapping::Vorbis {
                    headers: Some(headers),
                    ..
                } => vec![headers.comment.clone(), headers.setup.clone()],
                Mapping::Vorbis { headers: None, .. } => Vec::new(),
            };
            for packet in &packets {
                self.lay(index, packet, 0, 0)?;
            }
            self.flush_page(index, false)?;
            let track = &mut self.tracks[index];
            track.completed_pages = 0;
            track.last_packet = None;
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
        let granule = if track.completes {
            if track.granule < 0 {
                return Err(rejected(format!(
                    "a page of Vorbis stream {} would end at {}, before 0; the stream starts at {}",
                    track.id.get(),
                    track.granule,
                    track.first.unwrap_or(0)
                )));
            }
            track.granule
        } else if eos {
            track.next.unwrap_or(0).max(0)
        } else {
            -1
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
        if track.completes {
            track.completed_pages = track.completed_pages.saturating_add(1);
        }
        track.continued = track.lacing.last() == Some(&255);
        track.lacing.clear();
        track.body.clear();
        track.granule = -1;
        track.completes = false;
        track.page_samples = 0;
        self.emit(&page)
    }

    /// RFC 7845 section 4.5 and Vorbis I section A.2 read the first audio
    /// page with a completed packet, when it is also the last page, as
    /// starting at its granule minus its samples, or at 0 when the granule
    /// is smaller. A stream that starts anywhere but 0 and trims its end
    /// cannot say both on that one page, so the packets before the trimmed
    /// one go on a page of their own. One trimmed packet alone, starting
    /// after 0, cannot be written. A Vorbis stream that starts before 0 and
    /// trims nothing ends on an empty page of its own.
    fn separate_trim_from_start(&mut self, index: usize) -> Result<(), MuxError> {
        let track = &mut self.tracks[index];
        if track.completed_pages > 0 || track.first.is_none_or(|first| first == 0) {
            return Ok(());
        }
        if !track.trimmed {
            if track.first.is_some_and(|first| first < 0) && track.completes {
                self.flush_page(index, false)?;
            }
            return Ok(());
        }
        let Some((sequence, lacing_at, body_at, granule, samples)) = track.last_packet else {
            return Ok(());
        };
        // Only the last packet completes here when it began on an earlier
        // page or opens this one.
        if sequence != track.sequence || lacing_at == 0 {
            return Err(rejected(format!(
                "{} stream {} is one trimmed packet starting at {}; Ogg reads that as starting at 0",
                track.mapping_name(),
                track.id.get(),
                track.first.unwrap_or(0)
            )));
        }
        let tail_lacing = track.lacing.split_off(lacing_at);
        let tail_body = track.body.split_off(body_at);
        let end = track.granule;
        let tail_samples = track.page_samples.saturating_sub(samples);
        track.granule = granule;
        track.completes = true;
        track.page_samples = samples;
        self.flush_page(index, false)?;
        let track = &mut self.tracks[index];
        track.lacing = tail_lacing;
        track.body = tail_body;
        track.granule = end;
        track.completes = true;
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
                track.completes = true;
                track.page_samples = track.page_samples.saturating_add(samples);
            }
        }
        Ok(())
    }

    /// The packet's start, duration, natural duration, and, for Vorbis,
    /// block size.
    fn check_packet(&self, index: usize, packet: &Packet) -> Result<Checked, MuxError> {
        let track = &self.tracks[index];
        if packet.timescale().ticks_per_second() != track.rate {
            return Err(MuxError::TimescaleMismatch {
                packet: packet.timescale().ticks_per_second(),
                stream: track.rate,
            });
        }
        let timestamp = packet.timestamp();
        let (Some(pts), Some(duration)) =
            (timestamp.presentation_or_decode(), timestamp.duration())
        else {
            return Err(MuxError::MissingTimestamp);
        };
        let size = u64::try_from(packet.bytes().len()).unwrap_or(u64::MAX);
        let cap = match &track.mapping {
            Mapping::Opus { streams, .. } => MAX_OPUS_PACKET_PER_STREAM
                .saturating_mul(u64::from(*streams))
                .min(self.limits.max_packet_bytes),
            Mapping::Vorbis { .. } => self.limits.max_packet_bytes,
        };
        if size > cap {
            return Err(MuxError::HostileSize {
                declared: size,
                limit: cap,
            });
        }
        let (samples, block) = match &track.mapping {
            Mapping::Opus { .. } => {
                let Some(samples) = packet_samples(packet.bytes()) else {
                    return Err(rejected("an Opus packet has a malformed TOC".to_owned()));
                };
                (i64::from(samples), None)
            }
            Mapping::Vorbis { headers, previous } => {
                let Some(headers) = headers else {
                    return Err(missing_headers(track));
                };
                let Some(block) = headers.modes.block(packet.bytes()) else {
                    return Err(rejected(
                        "a Vorbis packet is empty, a header, or names a mode the setup lacks"
                            .to_owned(),
                    ));
                };
                (vorbis::packet_samples(*previous, block), Some(block))
            }
        };
        if duration > samples {
            return Err(rejected(format!(
                "packet duration {duration} is longer than its natural {samples} samples"
            )));
        }
        if track.trimmed {
            return Err(rejected(
                "only the last packet of a stream may be shorter than its natural duration"
                    .to_owned(),
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
                    "packet at {pts} leaves a gap after {next}; Ogg needs packets that fill it"
                )));
            }
            None if pts < 0 && matches!(track.mapping, Mapping::Opus { .. }) => {
                return Err(rejected(format!("first packet starts at {pts}")));
            }
            _ => {}
        }
        Ok(Checked {
            pts,
            duration,
            samples,
            block,
        })
    }

    fn store_config(&mut self, index: usize, config: &[u8]) -> Result<(), MuxError> {
        let size = u64::try_from(config.len()).unwrap_or(u64::MAX);
        if size > self.limits.max_extradata_bytes {
            return Err(MuxError::HostileSize {
                declared: size,
                limit: self.limits.max_extradata_bytes,
            });
        }
        let track = &mut self.tracks[index];
        match &mut track.mapping {
            Mapping::Opus {
                head,
                streams,
                pre_skip,
            } => {
                let bytes = opus_head(config)?;
                let parsed = OpusHead::parse(&bytes, track.serial)
                    .map_err(|error| rejected(error.to_string()))?;
                if u16::from(parsed.channels) != track.channels {
                    return Err(rejected(format!(
                        "OpusHead has {} channels and the stream has {}",
                        parsed.channels, track.channels
                    )));
                }
                *streams = parsed.streams;
                *pre_skip = parsed.pre_skip;
                *head = Some(bytes);
            }
            Mapping::Vorbis { headers, .. } => {
                *headers = Some(vorbis_headers(
                    config,
                    track.serial,
                    track.channels,
                    track.rate,
                )?);
            }
        }
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
            if duration.timescale.ticks_per_second() != track.rate {
                return Err(MuxError::TimescaleMismatch {
                    packet: duration.timescale.ticks_per_second(),
                    stream: track.rate,
                });
            }
            if duration.ticks != track.sum {
                return Err(MuxError::DurationMismatch {
                    track: track.id.get(),
                });
            }
            // RFC 7845 section 4.5: the pre-skip is decoded and dropped, so a
            // stream must last at least that long.
            if let Mapping::Opus { pre_skip, .. } = track.mapping
                && track.next.is_some()
                && track.sum < i64::from(pre_skip)
            {
                return Err(rejected(format!(
                    "Opus stream {} lasts {} samples, less than its pre-skip of {}",
                    track.id.get(),
                    track.sum,
                    pre_skip
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

fn missing_headers(track: &Track) -> MuxError {
    let what = match track.mapping {
        Mapping::Opus { .. } => "OpusHead",
        Mapping::Vorbis { .. } => "Vorbis id, comment, and setup headers",
    };
    rejected(format!(
        "track {} has no {what}: give them in the stream's extradata or set_config",
        track.id.get()
    ))
}

/// A packet that passed [`OggMux::check_packet`].
struct Checked {
    pts: i64,
    duration: i64,
    /// The packet's natural duration.
    samples: i64,
    /// Vorbis: the packet's block size.
    block: Option<u32>,
}

/// The three Vorbis headers of a configuration, checked against the
/// stream: Xiph lacing (Matroska, and `cutlass-ogg`'s demuxer), three
/// 16-bit lengths, or the packets back to back. The comment packet is kept
/// as given.
fn vorbis_headers(
    config: &[u8],
    serial: u32,
    channels: u16,
    rate: u32,
) -> Result<VorbisHeaders, MuxError> {
    let Some([id, comment, setup]) = split_config(config) else {
        return Err(rejected(
            "codec configuration is not three Vorbis headers".to_owned(),
        ));
    };
    let parsed = VorbisId::parse(&id, serial).map_err(|error| rejected(error.to_string()))?;
    if u16::from(parsed.channels) != channels || parsed.sample_rate != rate {
        return Err(rejected(format!(
            "the Vorbis identification header has {} channels at {} Hz and the stream {channels} at {rate} Hz",
            parsed.channels, parsed.sample_rate
        )));
    }
    VorbisComment::parse(&comment, serial, false).map_err(|error| rejected(error.to_string()))?;
    let modes =
        VorbisModes::parse(&setup, &parsed, serial).map_err(|error| rejected(error.to_string()))?;
    Ok(VorbisHeaders {
        id,
        comment,
        setup,
        modes,
    })
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
        let (mapping, rate) = match stream.params.codec() {
            StreamCodec::Known(CodecId::Opus) => (
                Mapping::Opus {
                    head: None,
                    streams: 1,
                    pre_skip: 0,
                },
                OPUS_RATE,
            ),
            StreamCodec::Known(CodecId::Vorbis) => {
                let Some(rate) = stream.params.sample_rate().filter(|&rate| rate > 0) else {
                    return Err(
                        self.fail(rejected("a Vorbis stream needs a sample rate".to_owned()))
                    );
                };
                (
                    Mapping::Vorbis {
                        headers: None,
                        previous: None,
                    },
                    rate,
                )
            }
            StreamCodec::Unknown(unknown) => {
                return Err(self.fail(MuxError::UnknownCodec(unknown.clone())));
            }
            StreamCodec::Known(other) => {
                return Err(self.fail(rejected(format!(
                    "this Ogg muxer writes the Opus and Vorbis mappings only, not {other:?}"
                ))));
            }
        };
        if stream.timescale.ticks_per_second() != rate {
            return Err(self.fail(MuxError::TimescaleMismatch {
                packet: stream.timescale.ticks_per_second(),
                stream: rate,
            }));
        }
        let Some(channels) = stream.params.channels() else {
            return Err(self.fail(rejected(
                "an Ogg audio stream needs a channel count".to_owned(),
            )));
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
            rate,
            mapping,
            sequence: 0,
            lacing: Vec::new(),
            body: Vec::new(),
            continued: false,
            granule: -1,
            completes: false,
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
        let Checked {
            pts,
            duration,
            samples,
            block,
        } = self
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
        if let Mapping::Vorbis { previous, .. } = &mut track.mapping {
            *previous = block;
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

    /// Stores the `OpusHead` (or an MP4 `dOps` body), or the three Vorbis
    /// headers, an encoder hands over. It must arrive before the first
    /// packet of any stream.
    fn set_config(&mut self, track: TrackId, config: &[u8]) -> Result<(), MuxError> {
        self.building()?;
        let index = self
            .track_index(track.get())
            .map_err(|error| self.fail(error))?;
        if self.headers {
            return Err(self.fail(rejected(
                "the header pages are already written; set the configuration before packets"
                    .to_owned(),
            )));
        }
        self.store_config(index, config)
            .map_err(|error| self.fail(error))
    }
}
