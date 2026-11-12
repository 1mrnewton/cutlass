//! The CAF muxer.
//!
//! The header is written when the first packet arrives, so a codec
//! configuration handed over with that packet still reaches the `kuki`
//! chunk. The `data` chunk is written with a size of -1, as the
//! specification allows while recording, and each packet's bytes go
//! straight to the writer: nothing but the packet table is held in memory.
//! Finish appends the `pakt` chunk after `data`, then fills in the `data`
//! size, `mFramesPerPacket`, and a cookie whose bitrates are only known at
//! the end. A mux that stops before finish leaves a `data` size of -1 and
//! no packet table, which no reader takes for a complete variable-packet file.
//!
//! Layout: `caff`, `desc`, `chan` (mono and stereo get the matching tag;
//! other counts only when [`CafMux::set_channel_layout`] names one),
//! `kuki` when the codec has one, `info` when set, `data`, then `pakt` when
//! packets vary or the stream has priming or remainder frames.
//!
//! Compressed packets: when every packet but the last has the same
//! duration `F` and the last is not longer, `mFramesPerPacket` is `F` and
//! the frames the last packet lacks are remainder frames, as AudioToolbox
//! writes a short final ALAC or AAC packet. Otherwise `mFramesPerPacket` is
//! zero and the table carries each packet's frames. When the last packet is
//! short and [`Mux::set_trim`] also names trailing padding, the two describe
//! the same tail (the Opus encoder hands over both); the larger is written.

use std::io::{Seek, SeekFrom, Write};

use cutlass_codec::{CodecId, ContainerId, Packet, StreamCodec};
use cutlass_container::{
    FailedMux, FinishedMux, IndexPlace, Limits, Mux, MuxError, MuxMode, MuxStatus, NewStream,
    TrackDuration, TrackId, Trim,
};

use crate::bytes::len_u64;
use crate::chan::ChannelLayout;
use crate::cookie::{self, AacRates};
use crate::desc::{Description, FLAG_FLOAT, FLAG_LITTLE_ENDIAN};
use crate::error::rejected;
use crate::info::{self, InfoEntry};
use crate::kept::KeptChunk;
use crate::limits::{MAX_CHANNELS, MAX_FRAMES_PER_PACKET, MAX_KEPT_BYTES};
use crate::pakt::PacketTableHeader;
use crate::varint;
use crate::walk::{CHUNK_HEADER_BYTES, FILE_HEADER_BYTES};

/// How packets of the stream are sized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// One frame per CAF packet; a written packet holds whole frames.
    Constant { frame_bytes: u32 },
    /// One codec packet per CAF packet, sizes in the table.
    Variable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cookie {
    None,
    Aac,
    Alac,
    Flac,
}

struct Track {
    codec: CodecId,
    rate: u32,
    channels: u16,
    format: [u8; 4],
    flags: u32,
    bits: u32,
    kind: Kind,
    cookie: Cookie,
    /// The codec configuration the cookie is built from.
    config: Option<Vec<u8>>,
    /// An `OpusHead` pre-skip, used when no trim names the priming.
    pre_skip: Option<u16>,
}

/// Offsets of the fields finish fills in.
#[derive(Clone, Copy, Debug, Default)]
struct Written {
    desc_body: u64,
    kuki_body: Option<(u64, u64)>,
    data_size: u64,
    data_start: u64,
}

/// Writes one CAF stream.
pub struct CafMux<W: Write + Seek> {
    out: W,
    limits: Limits,
    status: MuxStatus,
    failure: Option<MuxError>,
    track: Option<Track>,
    layout: Option<ChannelLayout>,
    info: Vec<InfoEntry>,
    kept: Vec<KeptChunk>,
    written: Option<Written>,
    sizes: Vec<u32>,
    frames: Vec<u32>,
    packets: u64,
    payload: u64,
    total_frames: u64,
    largest: u32,
    next_pts: Option<i64>,
    trim: Option<Trim>,
}

impl<W: Write + Seek> CafMux<W> {
    #[must_use]
    pub fn new(out: W) -> Self {
        Self::with_limits(out, Limits::standard())
    }

    /// [`Self::new`] with caller bounds. `max_tracks` is always one.
    #[must_use]
    pub fn with_limits(out: W, limits: Limits) -> Self {
        Self {
            out,
            limits,
            status: MuxStatus::Building,
            failure: None,
            track: None,
            layout: None,
            info: Vec::new(),
            kept: Vec::new(),
            written: None,
            sizes: Vec::new(),
            frames: Vec::new(),
            packets: 0,
            payload: 0,
            total_frames: 0,
            largest: 0,
            next_pts: None,
            trim: None,
        }
    }

    #[must_use]
    pub fn into_inner(self) -> W {
        self.out
    }

    #[must_use]
    pub const fn writer(&self) -> &W {
        &self.out
    }

    /// Names the channel layout written in `chan`. Call before the first packet.
    pub fn set_channel_layout(&mut self, layout: ChannelLayout) -> Result<(), MuxError> {
        self.building()?;
        if self.written.is_some() {
            return Err(rejected("the channel layout is written with the header"));
        }
        self.layout = Some(layout);
        Ok(())
    }

    /// Strings written in `info`. Call before the first packet.
    pub fn set_info(&mut self, entries: Vec<InfoEntry>) -> Result<(), MuxError> {
        self.building()?;
        if self.written.is_some() {
            return Err(rejected("info is written with the header"));
        }
        self.info = entries;
        Ok(())
    }

    /// Adds a kept chunk (a demuxer's [`crate::CafDemux::kept_chunks`]),
    /// written after `info`, before `data`, in the order added. Call before
    /// the first packet. Marker and region positions are frames, so carry
    /// them only with the same sound. Refused once all kept chunks together
    /// would pass [`MAX_KEPT_BYTES`].
    pub fn add_chunk(&mut self, chunk: KeptChunk) -> Result<(), MuxError> {
        self.building()?;
        if self.written.is_some() {
            return Err(rejected("kept chunks are written with the header"));
        }
        let total = self
            .kept
            .iter()
            .map(KeptChunk::len)
            .fold(chunk.len(), u64::saturating_add);
        if total > MAX_KEPT_BYTES {
            return Err(MuxError::HostileSize {
                declared: total,
                limit: MAX_KEPT_BYTES,
            });
        }
        self.kept.push(chunk);
        Ok(())
    }

    fn building(&self) -> Result<(), MuxError> {
        match self.status {
            MuxStatus::Building => Ok(()),
            MuxStatus::Failed => Err(self.failure.clone().unwrap_or(MuxError::NotBuilding)),
            MuxStatus::Finished => Err(MuxError::AlreadyFinished),
        }
    }

    fn fail(&mut self, error: MuxError) -> MuxError {
        if self.status == MuxStatus::Building {
            self.status = MuxStatus::Failed;
            self.failure = Some(error.clone());
        }
        error
    }

    fn io(&mut self, error: &std::io::Error) -> MuxError {
        self.fail(rejected(format!("caf write failed: {error}")))
    }

    fn write_all(&mut self, bytes: &[u8]) -> Result<(), MuxError> {
        if let Err(error) = self.out.write_all(bytes) {
            return Err(self.io(&error));
        }
        Ok(())
    }

    fn position(&mut self) -> Result<u64, MuxError> {
        match self.out.stream_position() {
            Ok(at) => Ok(at),
            Err(error) => Err(self.io(&error)),
        }
    }

    fn write_at(&mut self, at: u64, bytes: &[u8]) -> Result<(), MuxError> {
        if let Err(error) = self.out.seek(SeekFrom::Start(at)) {
            return Err(self.io(&error));
        }
        self.write_all(bytes)
    }

    fn description(&self, track: &Track, frames_per_packet: u32) -> Description {
        let bytes_per_packet = match track.kind {
            Kind::Constant { frame_bytes } => frame_bytes,
            Kind::Variable => 0,
        };
        Description {
            sample_rate: track.rate,
            format: track.format,
            flags: track.flags,
            bytes_per_packet,
            frames_per_packet,
            channels: u32::from(track.channels),
            bits_per_channel: track.bits,
        }
    }

    /// The cookie for the current configuration.
    fn cookie_bytes(&self, track: &Track) -> Result<Option<Vec<u8>>, MuxError> {
        let config = || {
            track.config.clone().ok_or_else(|| {
                rejected("the codec configuration did not arrive before the first packet")
            })
        };
        Ok(match track.cookie {
            Cookie::None => None,
            Cookie::Aac => Some(cookie::aac_cookie(&config()?, self.aac_rates(track))),
            Cookie::Alac | Cookie::Flac => Some(match track.cookie {
                Cookie::Flac => cookie::flac_cookie(&config()?),
                _ => config()?,
            }),
        })
    }

    fn aac_rates(&self, track: &Track) -> AacRates {
        let rate = u64::from(track.rate);
        let frames = self.total_frames.max(1);
        let avg = self.payload.saturating_mul(8).saturating_mul(rate) / frames;
        let longest = self.frames.iter().copied().max().unwrap_or(1).max(1);
        let max = u64::from(self.largest)
            .saturating_mul(8)
            .saturating_mul(rate)
            / u64::from(longest);
        AacRates {
            buffer_bytes: self.largest,
            max_bitrate: u32::try_from(max).unwrap_or(u32::MAX),
            avg_bitrate: u32::try_from(avg).unwrap_or(u32::MAX),
        }
    }

    fn write_header(&mut self) -> Result<(), MuxError> {
        let Some(track) = self.track.take() else {
            return Err(rejected("add a stream before a packet"));
        };
        let result = self.header_bytes(&track);
        self.track = Some(track);
        let (bytes, written) = match result {
            Ok(done) => done,
            Err(error) => return Err(self.fail(error)),
        };
        let start = self.position()?;
        self.write_all(&bytes)?;
        self.written = Some(Written {
            desc_body: start + written.desc_body,
            kuki_body: written.kuki_body.map(|(at, len)| (start + at, len)),
            data_size: start + written.data_size,
            data_start: start + written.data_start,
        });
        Ok(())
    }

    fn header_bytes(&self, track: &Track) -> Result<(Vec<u8>, Written), MuxError> {
        let mut out = Vec::new();
        out.extend_from_slice(b"caff");
        out.extend_from_slice(&1_u16.to_be_bytes());
        out.extend_from_slice(&0_u16.to_be_bytes());
        let desc = self.description(track, 0);
        chunk(&mut out, *b"desc", &desc.to_bytes());
        let mut written = Written {
            desc_body: FILE_HEADER_BYTES + CHUNK_HEADER_BYTES,
            ..Written::default()
        };
        let layout = self
            .layout
            .clone()
            .or_else(|| ChannelLayout::default_for(track.channels));
        if let Some(layout) = layout {
            chunk(&mut out, *b"chan", &layout.to_bytes());
        }
        if let Some(cookie) = self.cookie_bytes(track)? {
            let at = len_u64(out.len()) + CHUNK_HEADER_BYTES;
            written.kuki_body = Some((at, len_u64(cookie.len())));
            chunk(&mut out, *b"kuki", &cookie);
        }
        if !self.info.is_empty() {
            chunk(&mut out, *b"info", &info::to_bytes(&self.info));
        }
        for kept in &self.kept {
            chunk(&mut out, kept.kind(), kept.body());
        }
        out.extend_from_slice(b"data");
        written.data_size = len_u64(out.len());
        out.extend_from_slice(&(-1_i64).to_be_bytes());
        out.extend_from_slice(&0_u32.to_be_bytes());
        written.data_start = len_u64(out.len());
        Ok((out, written))
    }

    fn check_packet(&self, packet: &Packet) -> Result<(u32, u32), MuxError> {
        let Some(track) = &self.track else {
            return Err(rejected("add a stream before a packet"));
        };
        if packet.stream() != 0 {
            return Err(MuxError::UnknownTrack {
                id: packet.stream(),
            });
        }
        if packet.timescale().ticks_per_second() != track.rate {
            return Err(MuxError::TimescaleMismatch {
                packet: packet.timescale().ticks_per_second(),
                stream: track.rate,
            });
        }
        let stamp = packet.timestamp();
        let (Some(start), Some(duration)) = (stamp.presentation_or_decode(), stamp.duration())
        else {
            return Err(MuxError::MissingTimestamp);
        };
        if let Some(expect) = self.next_pts
            && start != expect
        {
            return Err(rejected(format!(
                "packet starts at {start}, not {expect}: CAF cannot store a gap or an overlap"
            )));
        }
        let size = len_u64(packet.bytes().len());
        if size > self.limits.max_packet_bytes {
            return Err(MuxError::HostileSize {
                declared: size,
                limit: self.limits.max_packet_bytes,
            });
        }
        let size = u32::try_from(size).map_err(|_| MuxError::HostileSize {
            declared: size,
            limit: u64::from(u32::MAX),
        })?;
        let frames = u64::try_from(duration).map_err(|_| MuxError::NegativeDuration)?;
        match track.kind {
            Kind::Constant { frame_bytes } => {
                if !size.is_multiple_of(frame_bytes) || u64::from(size / frame_bytes) != frames {
                    return Err(rejected(
                        "a PCM packet must hold whole frames, as many as its duration",
                    ));
                }
                Ok((size, u32::try_from(frames).unwrap_or(u32::MAX)))
            }
            Kind::Variable => {
                if frames == 0 || frames > u64::from(MAX_FRAMES_PER_PACKET) {
                    return Err(rejected(format!(
                        "a packet of {frames} frames cannot be stored"
                    )));
                }
                let count = len_u64(self.sizes.len()) + 1;
                if count > u64::from(self.limits.max_index_entries) {
                    return Err(MuxError::HostileSize {
                        declared: count,
                        limit: u64::from(self.limits.max_index_entries),
                    });
                }
                Ok((size, u32::try_from(frames).unwrap_or(u32::MAX)))
            }
        }
    }

    fn accept(&mut self, packet: &Packet) -> Result<(), MuxError> {
        let (size, frames) = self.check_packet(packet)?;
        if self.written.is_none() {
            self.write_header()?;
        }
        let variable = matches!(
            self.track.as_ref().map(|track| track.kind),
            Some(Kind::Variable)
        );
        if variable {
            if self.sizes.try_reserve(1).is_err() || self.frames.try_reserve(1).is_err() {
                return Err(self.fail(rejected("out of memory for the packet table")));
            }
            self.sizes.push(size);
            self.frames.push(frames);
        }
        self.write_all(packet.bytes())?;
        self.packets += 1;
        self.payload += u64::from(size);
        self.total_frames += u64::from(frames);
        self.largest = self.largest.max(size);
        let start = packet.timestamp().presentation_or_decode().unwrap_or(0);
        self.next_pts = start.checked_add(i64::from(frames));
        Ok(())
    }

    fn complete(&mut self, durations: &[TrackDuration]) -> Result<IndexPlace, MuxError> {
        let Some(rate) = self.track.as_ref().map(|track| track.rate) else {
            return Err(rejected("CAF needs one audio stream"));
        };
        let [duration] = durations else {
            return Err(MuxError::Incomplete);
        };
        if duration.track != TrackId::new(0) {
            return Err(MuxError::Incomplete);
        }
        if duration.timescale.ticks_per_second() != rate {
            return Err(MuxError::TimescaleMismatch {
                packet: duration.timescale.ticks_per_second(),
                stream: rate,
            });
        }
        if duration.ticks < 0 {
            return Err(MuxError::NegativeDuration);
        }
        if u64::try_from(duration.ticks).ok() != Some(self.total_frames) {
            return Err(MuxError::DurationMismatch { track: 0 });
        }
        if self.written.is_none() {
            self.write_header()?;
        }
        let Some(track) = self.track.take() else {
            return Err(rejected("CAF needs one audio stream"));
        };
        let result = self.finish_track(&track);
        self.track = Some(track);
        result
    }

    fn finish_track(&mut self, track: &Track) -> Result<IndexPlace, MuxError> {
        let Some(written) = self.written else {
            return Err(MuxError::Incomplete);
        };
        let (frames_per_packet, short) = self.frames_per_packet(track);
        let coded = match track.kind {
            Kind::Constant { .. } => self.total_frames,
            Kind::Variable if frames_per_packet > 0 => {
                self.packets.saturating_mul(u64::from(frames_per_packet))
            }
            Kind::Variable => self.total_frames,
        };
        let trim = self.trim.unwrap_or_default();
        let priming = match trim.encoder_delay() {
            Some(delay) => delay,
            None => track
                .pre_skip
                .map_or(0, |skip| u64::from(skip) * u64::from(track.rate) / 48_000),
        };
        let remainder = trim.trailing_padding().unwrap_or(0).max(short);
        let (Ok(priming32), Ok(remainder32)) = (
            u32::try_from(priming).map(|value| value.min(i32::MAX as u32)),
            u32::try_from(remainder).map(|value| value.min(i32::MAX as u32)),
        ) else {
            return Err(rejected(
                "priming or remainder frames do not fit in 32 bits",
            ));
        };
        if u64::from(priming32) != priming || u64::from(remainder32) != remainder {
            return Err(rejected(
                "priming or remainder frames do not fit in 32 bits",
            ));
        }
        let Some(valid) = coded.checked_sub(priming + remainder) else {
            return Err(rejected("the trim is longer than the sound"));
        };
        let end = self.position()?;
        let variable = track.kind == Kind::Variable;
        let table = variable || priming > 0 || remainder > 0;
        if table {
            let header = PacketTableHeader {
                packets: if variable { self.packets } else { 0 },
                valid_frames: valid,
                priming_frames: priming32,
                remainder_frames: remainder32,
            };
            let mut body = header.to_bytes().to_vec();
            if variable {
                for (index, size) in self.sizes.iter().enumerate() {
                    varint::encode(*size, &mut body);
                    if frames_per_packet == 0 {
                        varint::encode(self.frames.get(index).copied().unwrap_or(0), &mut body);
                    }
                }
            }
            let mut bytes = Vec::with_capacity(body.len() + 12);
            chunk(&mut bytes, *b"pakt", &body);
            self.write_all(&bytes)?;
        }
        let file_end = self.position()?;
        let data_size = end.saturating_sub(written.data_start) + 4;
        let data_size = i64::try_from(data_size).map_err(|_| rejected("data does not fit"))?;
        self.write_at(written.data_size, &data_size.to_be_bytes())?;
        let desc = self.description(track, frames_per_packet);
        self.write_at(written.desc_body, &desc.to_bytes())?;
        if let Some((at, len)) = written.kuki_body
            && let Some(cookie) = self.cookie_bytes(track)?
            && len_u64(cookie.len()) == len
        {
            self.write_at(at, &cookie)?;
        }
        if let Err(error) = self.out.seek(SeekFrom::Start(file_end)) {
            return Err(self.io(&error));
        }
        if let Err(error) = self.out.flush() {
            return Err(self.io(&error));
        }
        Ok(if table {
            IndexPlace::End
        } else {
            IndexPlace::Absent
        })
    }

    /// `mFramesPerPacket`, and the frames a short last packet lacks.
    fn frames_per_packet(&self, track: &Track) -> (u32, u64) {
        match track.kind {
            Kind::Constant { .. } => (1, 0),
            Kind::Variable => {
                let Some((last, rest)) = self.frames.split_last() else {
                    return (0, 0);
                };
                let first = rest.first().copied().unwrap_or(*last);
                if rest.iter().all(|frames| *frames == first) && *last <= first {
                    (first, u64::from(first - last))
                } else {
                    (0, 0)
                }
            }
        }
    }

    fn take_config(&mut self, config: &[u8]) -> Result<(), MuxError> {
        let Some(track) = self.track.as_mut() else {
            return Err(rejected("add a stream before a config"));
        };
        let normalized = match track.cookie {
            Cookie::None => {
                if track.codec == CodecId::Opus {
                    track.pre_skip = cookie::opus_pre_skip(config).or(track.pre_skip);
                }
                // AC-3, E-AC-3 and MP3 have no cookie in CAF.
                return Ok(());
            }
            Cookie::Aac => aac_config(config)?,
            Cookie::Alac => alac_config(config)?,
            Cookie::Flac => flac_config(config)?,
        };
        if self.written.is_some() {
            let same = track.config.as_ref().map(Vec::len) == Some(normalized.len());
            if !same {
                return Err(rejected(
                    "a codec configuration of another size arrived after the header",
                ));
            }
        }
        if self.written.is_none() {
            track.flags = source_depth(track.cookie, Some(&normalized));
        }
        track.config = Some(normalized);
        Ok(())
    }
}

/// The `desc` flags of ALAC and FLAC: the bit depth of the source, from
/// the codec configuration.
fn source_depth(cookie: Cookie, config: Option<&[u8]>) -> u32 {
    let Some(config) = config else { return 0 };
    match cookie {
        Cookie::Alac => config.get(5).map_or(0, |bits| depth_flag(u32::from(*bits))),
        Cookie::Flac => {
            // STREAMINFO: bits per sample less one in the 5 bits that
            // straddle bytes 12 and 13.
            let high = config.get(12).copied().unwrap_or(0) & 1;
            let low = config.get(13).copied().unwrap_or(0) >> 4;
            depth_flag(u32::from((high << 4) | low) + 1)
        }
        Cookie::Aac | Cookie::None => 0,
    }
}

fn flac_config(config: &[u8]) -> Result<Vec<u8>, MuxError> {
    cookie::streaminfo_from_extradata(config)
        .ok_or_else(|| rejected("the FLAC config is not a STREAMINFO"))
}

fn chunk(out: &mut Vec<u8>, kind: [u8; 4], body: &[u8]) {
    out.extend_from_slice(&kind);
    out.extend_from_slice(&i64::try_from(body.len()).unwrap_or(0).to_be_bytes());
    out.extend_from_slice(body);
}

fn aac_config(config: &[u8]) -> Result<Vec<u8>, MuxError> {
    if config.is_empty() || config.len() > cookie::MAX_ASC_BYTES {
        return Err(rejected(
            "an AAC config is a 1 to 64 byte AudioSpecificConfig",
        ));
    }
    Ok(config.to_vec())
}

fn alac_config(config: &[u8]) -> Result<Vec<u8>, MuxError> {
    cookie::alac_config(config).map_err(|error| rejected(error.to_string()))
}

/// The bit depth flag AudioToolbox writes for ALAC and FLAC: 1, 2, 3, 4 for
/// 16, 20, 24, 32-bit source.
const fn depth_flag(bits: u32) -> u32 {
    match bits {
        16 => 1,
        20 => 2,
        24 => 3,
        32 => 4,
        _ => 0,
    }
}

fn track_of(stream: &NewStream) -> Result<Track, MuxError> {
    let params = &stream.params;
    let codec = match params.codec() {
        StreamCodec::Unknown(unknown) => return Err(MuxError::UnknownCodec(unknown.clone())),
        StreamCodec::Known(codec) => *codec,
    };
    let (Some(rate), Some(channels)) = (params.sample_rate(), params.channels()) else {
        return Err(rejected(
            "CAF carries sound: the stream needs a rate and channels",
        ));
    };
    if stream.timescale.ticks_per_second() != rate {
        return Err(MuxError::TimescaleMismatch {
            packet: stream.timescale.ticks_per_second(),
            stream: rate,
        });
    }
    if u32::from(channels) > MAX_CHANNELS {
        return Err(MuxError::HostileSize {
            declared: u64::from(channels),
            limit: u64::from(MAX_CHANNELS),
        });
    }
    let extra = params.extradata();
    let config = |normalize: fn(&[u8]) -> Result<Vec<u8>, MuxError>| {
        if extra.is_empty() {
            Ok(None)
        } else {
            normalize(extra).map(Some)
        }
    };
    let variable = |format: [u8; 4], cookie: Cookie, config: Option<Vec<u8>>, flags: u32| Track {
        codec,
        rate,
        channels,
        format,
        flags,
        bits: 0,
        kind: Kind::Variable,
        cookie,
        config,
        pre_skip: None,
    };
    Ok(match codec {
        CodecId::Pcm => pcm_track(extra, rate, channels)?,
        CodecId::Aac => variable(*b"aac ", Cookie::Aac, config(aac_config)?, 0),
        CodecId::Alac => {
            let config = config(alac_config)?;
            let flags = source_depth(Cookie::Alac, config.as_deref());
            variable(*b"alac", Cookie::Alac, config, flags)
        }
        CodecId::Flac => {
            let config = config(flac_config)?;
            let flags = source_depth(Cookie::Flac, config.as_deref());
            variable(*b"flac", Cookie::Flac, config, flags)
        }
        CodecId::Opus => {
            if channels > 2 {
                return Err(rejected("CAF Opus is written for one or two channels"));
            }
            let mut track = variable(*b"opus", Cookie::None, None, 0);
            track.pre_skip = cookie::opus_pre_skip(extra);
            track
        }
        CodecId::Mp3 => variable(*b".mp3", Cookie::None, None, 0),
        CodecId::Ac3 => variable(*b"ac-3", Cookie::None, None, 0),
        CodecId::Eac3 => variable(*b"ec-3", Cookie::None, None, 0),
        other => {
            return Err(rejected(format!("CAF does not carry {other:?}")));
        }
    })
}

/// PCM with no extradata is 16-bit little-endian, as the PCM encoder makes
/// it for 16-bit input and as the WAV muxer takes it.
const DEFAULT_PCM: [u8; 3] = [1, 0, 0];

fn pcm_track(extra: &[u8], rate: u32, channels: u16) -> Result<Track, MuxError> {
    let extra = if extra.is_empty() {
        &DEFAULT_PCM[..]
    } else {
        extra
    };
    let [coding, endian, arrangement] = extra else {
        return Err(rejected("PCM extradata is the three-byte layout"));
    };
    if *arrangement != 0 {
        return Err(rejected("CAF stores interleaved frames"));
    }
    let little = match endian {
        0 => FLAG_LITTLE_ENDIAN,
        1 => 0,
        _ => return Err(rejected("PCM endian is not 0 or 1")),
    };
    let (format, bits, flags) = match coding {
        1 => (*b"lpcm", 16, little),
        2 => (*b"lpcm", 24, little),
        3 => (*b"lpcm", 32, little),
        4 => (*b"lpcm", 32, little | FLAG_FLOAT),
        5 => (*b"lpcm", 64, little | FLAG_FLOAT),
        6 => (*b"lpcm", 8, little),
        9 => (*b"lpcm", 64, little),
        7 => (*b"alaw", 8, 0),
        8 => (*b"ulaw", 8, 0),
        _ => return Err(rejected("this PCM coding has no CAF form")),
    };
    let frame_bytes = u32::from(channels) * (bits / 8);
    Ok(Track {
        codec: CodecId::Pcm,
        rate,
        channels,
        format,
        flags,
        bits,
        kind: Kind::Constant { frame_bytes },
        cookie: Cookie::None,
        config: None,
        pre_skip: None,
    })
}

impl<W: Write + Seek> Mux for CafMux<W> {
    fn family(&self) -> ContainerId {
        ContainerId::Caf
    }

    fn add_stream(&mut self, stream: NewStream) -> Result<TrackId, MuxError> {
        self.building()?;
        if self.track.is_some() {
            return Err(self.fail(MuxError::TooManyTracks { count: 1, limit: 1 }));
        }
        let track = match track_of(&stream) {
            Ok(track) => track,
            Err(error) => return Err(self.fail(error)),
        };
        self.track = Some(track);
        Ok(TrackId::new(0))
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<(), MuxError> {
        self.building()?;
        match self.accept(packet) {
            Ok(()) => Ok(()),
            Err(error) => Err(self.fail(error)),
        }
    }

    /// One stream is already in order.
    fn interleave(&mut self) -> Result<(), MuxError> {
        self.building()
    }

    fn cancel(&mut self) {
        let _ = self.fail(MuxError::Cancelled);
        let _ = self.out.flush();
    }

    fn finish(&mut self, durations: &[TrackDuration]) -> Result<FinishedMux, FailedMux> {
        if let Err(error) = self.building() {
            return Err(FailedMux::incomplete(self.payload, error));
        }
        match self.complete(durations) {
            Ok(index) => {
                self.status = MuxStatus::Finished;
                Ok(FinishedMux::complete(
                    durations.to_vec(),
                    index,
                    MuxMode::Progressive,
                    true,
                    Vec::new(),
                    self.payload,
                ))
            }
            Err(error) => {
                let error = self.fail(error);
                Err(FailedMux::incomplete(self.payload, error))
            }
        }
    }

    fn status(&self) -> MuxStatus {
        self.status
    }

    fn set_config(&mut self, track: TrackId, config: &[u8]) -> Result<(), MuxError> {
        self.building()?;
        if track != TrackId::new(0) {
            return Err(MuxError::UnknownTrack { id: track.get() });
        }
        match self.take_config(config) {
            Ok(()) => Ok(()),
            Err(error) => Err(self.fail(error)),
        }
    }

    fn set_trim(&mut self, track: TrackId, trim: Trim) -> Result<(), MuxError> {
        self.building()?;
        if track != TrackId::new(0) {
            return Err(MuxError::UnknownTrack { id: track.get() });
        }
        self.trim = Some(trim);
        Ok(())
    }
}
