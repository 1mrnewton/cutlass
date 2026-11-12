//! Writing a transport stream: one program, one PID per stream.
//!
//! Each access unit becomes one PES packet. Timestamps are moved to the
//! 90 kHz clock and offset by [`MuxOptions::delay`] (1.4 s by default, as
//! FFmpeg does), so a stream whose first DTS is below its first PTS still
//! starts at a positive time and the PCR can run ahead of every DTS. The
//! PCR rides on the first video stream, or the first stream, at least every
//! 20 ms of decode time; a PCR-only packet is sent when another stream's
//! data would leave a longer gap. The PAT and PMT go out first, every
//! 100 ms of decode time, and before each video keyframe.
//!
//! H.264 and HEVC are written as Annex B byte streams with an access unit
//! delimiter; packets in length-prefixed form (with an `avcC` or `hvcC`
//! record) are converted, and the parameter sets are put in front of every
//! keyframe that does not carry its own. Raw AAC gets an ADTS header from
//! the AudioSpecificConfig. Opus gets the control header of the Opus
//! mapping for MPEG-TS and the descriptors FFmpeg writes for it: a
//! registration descriptor `Opus` and a DVB extension descriptor with the
//! channel configuration code.

use std::io::Write;

use cutlass_codec::{CodecId, ContainerId, Packet, StreamCodec, Timescale};
use cutlass_container::{
    FailedMux, FinishedMux, IndexPlace, Language, Limits, Mux, MuxError, MuxMode, MuxStatus,
    NewStream, TrackDuration, TrackId,
};

use crate::crc::crc32;
use crate::es::bits::{Bits, find_start_code, starts_with_start_code};
use crate::limits::{SYNC_BYTE, TS_PACKET_BYTES};
use crate::packet::{PAT_PID, PacketSize, pcr_bytes};
use crate::pes::timestamp_bytes;
use crate::time::WRAP;

/// The 90 kHz clock every timestamp is written on.
const CLOCK: i64 = 90_000;
/// The first stream's PID, as FFmpeg numbers them.
const FIRST_PID: u16 = 0x100;
const PMT_PID: u16 = 0x1000;
const PROGRAM: u16 = 1;
const TRANSPORT_STREAM_ID: u16 = 1;
/// Decode time between PAT and PMT repeats: 100 ms.
const PSI_PERIOD: i64 = 9_000;
/// Decode time between PCRs: 20 ms, in 27 MHz ticks.
const PCR_PERIOD: u64 = 540_000;
/// The PCR runs this far behind the DTS: 0.7 s.
const PCR_LEAD: i64 = 63_000;
/// Output is written once this much is buffered.
const FLUSH_BYTES: usize = 256 * 1024;
/// The largest ADTS frame.
const MAX_ADTS_FRAME: usize = 8191;

/// How the stream is written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MuxOptions {
    /// 188-byte TS or 192-byte M2TS. A FEC stride is written as TS.
    pub packet_size: PacketSize,
    /// Ticks of the 90 kHz clock added to every timestamp.
    pub delay: i64,
    pub limits: Limits,
}

impl Default for MuxOptions {
    fn default() -> Self {
        Self {
            packet_size: PacketSize::Ts,
            delay: 126_000,
            limits: Limits::standard(),
        }
    }
}

/// How one stream's access units are written.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Shape {
    /// Annex B with an AUD; `length` is the NAL length size of
    /// length-prefixed input, when known.
    H264 {
        params: Vec<u8>,
        length: Option<u8>,
    },
    Hevc {
        params: Vec<u8>,
        length: Option<u8>,
    },
    /// The sequence header to put before keyframes that lack one.
    Mpeg2 {
        sequence: Vec<u8>,
    },
    /// ADTS fields from the AudioSpecificConfig, when the input is raw.
    Aac {
        header: Option<(u8, u8, u8)>,
    },
    /// Written as they come.
    Plain,
    Opus,
}

struct Track {
    id: TrackId,
    pid: u16,
    stream_id: u8,
    stream_type: u8,
    descriptors: Vec<u8>,
    timescale: Timescale,
    shape: Shape,
    video: bool,
    continuity: u8,
    last_dts: Option<i64>,
}

/// Writes one program of MPEG-TS or M2TS.
pub struct TsMux<W: Write> {
    writer: W,
    options: MuxOptions,
    tracks: Vec<Track>,
    status: MuxStatus,
    failure: Option<MuxError>,
    started: bool,
    out: Vec<u8>,
    pat_continuity: u8,
    pmt_continuity: u8,
    last_psi: Option<i64>,
    last_pcr: Option<u64>,
    /// The PCR of the last packet, for the M2TS arrival time.
    clock: u64,
    last_time: Option<i64>,
    in_order: bool,
    payload: u64,
}

impl<W: Write> TsMux<W> {
    /// A 188-byte transport stream with the default options.
    pub fn new(writer: W) -> Self {
        Self::with_options(writer, MuxOptions::default())
    }

    /// A 192-byte M2TS stream: each packet carries a 4-byte arrival time.
    pub fn m2ts(writer: W) -> Self {
        Self::with_options(
            writer,
            MuxOptions {
                packet_size: PacketSize::M2ts,
                ..MuxOptions::default()
            },
        )
    }

    pub fn with_options(writer: W, options: MuxOptions) -> Self {
        Self {
            writer,
            options,
            tracks: Vec::new(),
            status: MuxStatus::Building,
            failure: None,
            started: false,
            out: Vec::new(),
            pat_continuity: 0,
            pmt_continuity: 0,
            last_psi: None,
            last_pcr: None,
            clock: 0,
            last_time: None,
            in_order: true,
            payload: 0,
        }
    }

    /// The writer, after [`Mux::finish`].
    pub fn into_inner(self) -> W {
        self.writer
    }

    /// The PID each track was given, in track order.
    #[must_use]
    pub fn pids(&self) -> Vec<u16> {
        self.tracks.iter().map(|track| track.pid).collect()
    }

    fn fail(&mut self, error: MuxError) -> MuxError {
        self.status = MuxStatus::Failed;
        self.failure.get_or_insert_with(|| error.clone());
        error
    }

    fn building(&self) -> Result<(), MuxError> {
        match self.status {
            MuxStatus::Building => Ok(()),
            MuxStatus::Finished => Err(MuxError::AlreadyFinished),
            MuxStatus::Failed => Err(self.failure.clone().unwrap_or(MuxError::NotBuilding)),
        }
    }

    fn track_index(&self, id: u32) -> Result<usize, MuxError> {
        self.tracks
            .iter()
            .position(|track| track.id.get() == id)
            .ok_or(MuxError::UnknownTrack { id })
    }

    fn flush_out(&mut self) -> Result<(), MuxError> {
        if self.out.is_empty() {
            return Ok(());
        }
        let result = self.writer.write_all(&self.out);
        self.out.clear();
        result.map_err(|error| rejected(format!("write failed: {error}")))
    }

    /// The 90 kHz time of `ticks` at `scale`, rounded, plus the delay.
    fn clock_time(&self, ticks: i64, scale: Timescale) -> Result<i64, MuxError> {
        let rate = i128::from(scale.ticks_per_second());
        if rate == 0 {
            return Err(MuxError::Clockless);
        }
        let scaled = (i128::from(ticks) * i128::from(CLOCK) * 2 + rate).div_euclid(2 * rate);
        let time = scaled + i128::from(self.options.delay);
        if time < 0 {
            return Err(rejected(format!(
                "timestamp {ticks} at {} Hz is before the start of the stream",
                scale.ticks_per_second()
            )));
        }
        i64::try_from(time).map_err(|_| rejected("timestamp out of range".to_owned()))
    }

    fn pcr_track(&self) -> usize {
        self.tracks
            .iter()
            .position(|track| track.video)
            .unwrap_or(0)
    }

    /// Starts one transport packet: the M2TS arrival time, then the header.
    fn packet_header(&mut self, pid: u16, start: bool, control: u8, continuity: u8) {
        if self.options.packet_size == PacketSize::M2ts {
            let arrival = (self.clock % (1 << 30)) as u32;
            self.out.extend_from_slice(&arrival.to_be_bytes());
        }
        self.out.extend_from_slice(&[
            SYNC_BYTE,
            (u8::from(start) << 6) | ((pid >> 8) as u8 & 0x1F),
            pid as u8,
            (control << 4) | (continuity & 0x0F),
        ]);
    }

    /// Writes one PSI section on `pid`, across as many packets as it needs.
    fn write_section(&mut self, pid: u16, section: &[u8]) {
        let mut data = Vec::with_capacity(section.len() + 1);
        data.push(0);
        data.extend_from_slice(section);
        for (index, chunk) in data.chunks(TS_PACKET_BYTES - 4).enumerate() {
            let continuity = if pid == PAT_PID {
                let value = self.pat_continuity;
                self.pat_continuity = (value + 1) & 0x0F;
                value
            } else {
                let value = self.pmt_continuity;
                self.pmt_continuity = (value + 1) & 0x0F;
                value
            };
            self.packet_header(pid, index == 0, 0b01, continuity);
            self.out.extend_from_slice(chunk);
            self.out
                .resize(self.out.len() + TS_PACKET_BYTES - 4 - chunk.len(), 0xFF);
        }
    }

    fn write_tables(&mut self) {
        let pat = section(
            0x00,
            TRANSPORT_STREAM_ID,
            &[
                (PROGRAM >> 8) as u8,
                PROGRAM as u8,
                0xE0 | (PMT_PID >> 8) as u8,
                PMT_PID as u8,
            ],
        );
        self.write_section(PAT_PID, &pat);
        let pcr_pid = self
            .tracks
            .get(self.pcr_track())
            .map_or(0x1FFF, |track| track.pid);
        let mut body = vec![0xE0 | (pcr_pid >> 8) as u8, pcr_pid as u8, 0xF0, 0x00];
        for track in &self.tracks {
            let length = track.descriptors.len();
            body.extend_from_slice(&[
                track.stream_type,
                0xE0 | (track.pid >> 8) as u8,
                track.pid as u8,
                0xF0 | (length >> 8) as u8,
                length as u8,
            ]);
            body.extend_from_slice(&track.descriptors);
        }
        let pmt = section(0x02, PROGRAM, &body);
        self.write_section(PMT_PID, &pmt);
    }

    /// A packet with only an adaptation field carrying a PCR.
    fn write_pcr_only(&mut self, pid: u16, continuity: u8, pcr: u64) {
        self.clock = pcr;
        self.packet_header(pid, false, 0b10, continuity);
        self.out.push(183);
        self.out.push(0x10);
        self.out.extend_from_slice(&pcr_bytes(pcr));
        self.out.resize(self.out.len() + 183 - 7, 0xFF);
    }

    /// Writes one PES packet across transport packets.
    fn write_pes(&mut self, index: usize, pes: &[u8], pcr: Option<u64>, random_access: bool) {
        let pid = self.tracks[index].pid;
        if let Some(pcr) = pcr {
            self.clock = pcr;
        }
        let mut at = 0;
        let mut first = true;
        while at < pes.len() || first {
            let mut fields = Vec::new();
            if first && (pcr.is_some() || random_access) {
                let mut flags = 0_u8;
                if random_access {
                    flags |= 0x40;
                }
                if pcr.is_some() {
                    flags |= 0x10;
                }
                fields.push(flags);
                if let Some(pcr) = pcr {
                    fields.extend_from_slice(&pcr_bytes(pcr));
                }
            }
            let remaining = pes.len() - at;
            // Room for payload when the adaptation field holds `fields`.
            let header_room = if fields.is_empty() {
                0
            } else {
                1 + fields.len()
            };
            let room = TS_PACKET_BYTES - 4 - header_room;
            let take = remaining.min(room);
            let stuffing = room - take;
            let adaptation = if fields.is_empty() && stuffing == 0 {
                None
            } else if fields.is_empty() {
                // The adaptation field itself takes the stuffing.
                let length = stuffing - 1;
                let mut field = vec![length as u8];
                if length > 0 {
                    field.push(0x00);
                    field.resize(1 + length, 0xFF);
                }
                Some(field)
            } else {
                let length = fields.len() + stuffing;
                let mut field = vec![length as u8];
                field.extend_from_slice(&fields);
                field.resize(1 + length, 0xFF);
                Some(field)
            };
            let continuity = self.tracks[index].continuity;
            self.tracks[index].continuity = (continuity + 1) & 0x0F;
            let control = if adaptation.is_some() { 0b11 } else { 0b01 };
            self.packet_header(pid, first, control, continuity);
            if let Some(field) = adaptation {
                self.out.extend_from_slice(&field);
            }
            self.out.extend_from_slice(&pes[at..at + take]);
            at += take;
            first = false;
        }
    }

    /// The access unit's bytes as this stream writes them.
    fn shape(&self, index: usize, bytes: &[u8], keyframe: bool) -> Result<Vec<u8>, MuxError> {
        match &self.tracks[index].shape {
            Shape::H264 { params, length } => {
                Ok(annex_b(bytes, *length, params, keyframe, &H264_UNITS)?)
            }
            Shape::Hevc { params, length } => {
                Ok(annex_b(bytes, *length, params, keyframe, &HEVC_UNITS)?)
            }
            Shape::Mpeg2 { sequence } => {
                let has_sequence = find_start_code(bytes, 0)
                    .and_then(|at| bytes.get(at + 3))
                    .is_some_and(|&code| code == 0xB3);
                let mut out = Vec::with_capacity(sequence.len() + bytes.len());
                if keyframe && !has_sequence {
                    out.extend_from_slice(sequence);
                }
                out.extend_from_slice(bytes);
                Ok(out)
            }
            Shape::Aac { header } => {
                if bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] & 0xF6 == 0xF0 {
                    return Ok(bytes.to_vec());
                }
                let Some((profile, rate, channels)) = *header else {
                    return Err(rejected(
                        "raw AAC needs an AudioSpecificConfig to write ADTS".to_owned(),
                    ));
                };
                let length = bytes.len() + 7;
                if length > MAX_ADTS_FRAME {
                    return Err(MuxError::HostileSize {
                        declared: length as u64,
                        limit: MAX_ADTS_FRAME as u64,
                    });
                }
                let mut out = Vec::with_capacity(length);
                out.extend_from_slice(&adts_header(profile, rate, channels, length));
                out.extend_from_slice(bytes);
                Ok(out)
            }
            Shape::Opus => {
                let mut out = Vec::with_capacity(bytes.len() + 2 + bytes.len() / 255 + 1);
                out.extend_from_slice(&[0x7F, 0xE0]);
                let mut size = bytes.len();
                while size >= 255 {
                    out.push(0xFF);
                    size -= 255;
                }
                out.push(size as u8);
                out.extend_from_slice(bytes);
                Ok(out)
            }
            Shape::Plain => Ok(bytes.to_vec()),
        }
    }

    fn store_config(&mut self, index: usize, config: &[u8]) -> Result<(), MuxError> {
        let track = &mut self.tracks[index];
        match &mut track.shape {
            Shape::H264 { params, length } => {
                let (annex, size) = parse_config(config, ConfigKind::Avc)?;
                *params = annex;
                *length = size.or(*length);
            }
            Shape::Hevc { params, length } => {
                let (annex, size) = parse_config(config, ConfigKind::Hevc)?;
                *params = annex;
                *length = size.or(*length);
            }
            Shape::Mpeg2 { sequence } => *sequence = config.to_vec(),
            Shape::Aac { header } => {
                *header = Some(parse_asc(config).ok_or_else(|| {
                    rejected("the AudioSpecificConfig cannot be written as ADTS".to_owned())
                })?);
            }
            Shape::Opus => {
                let code = opus_code(config, None)?;
                track.descriptors = opus_descriptors(code, &track.descriptors);
            }
            Shape::Plain => {}
        }
        Ok(())
    }

    fn write_unit(&mut self, index: usize, packet: &Packet) -> Result<(), MuxError> {
        let scale = self.tracks[index].timescale;
        if packet.timescale() != scale {
            return Err(MuxError::TimescaleMismatch {
                packet: packet.timescale().ticks_per_second(),
                stream: scale.ticks_per_second(),
            });
        }
        let time = packet.timestamp();
        let (Some(dts), Some(pts)) = (
            time.decode().or(time.presentation()),
            time.presentation().or(time.decode()),
        ) else {
            return Err(MuxError::MissingTimestamp);
        };
        let dts = self.clock_time(dts, scale)?;
        let pts = self.clock_time(pts, scale)?;
        if pts < dts {
            return Err(rejected(format!("PTS {pts} is before DTS {dts}")));
        }
        if self.tracks[index].last_dts.is_some_and(|last| dts < last) {
            return Err(rejected(
                "decode times of a stream must not go back".to_owned(),
            ));
        }
        let declared = packet.bytes().len() as u64;
        if declared > self.options.limits.max_packet_bytes {
            return Err(MuxError::HostileSize {
                declared,
                limit: self.options.limits.max_packet_bytes,
            });
        }
        let keyframe = packet.keyframe();
        let bytes = self.shape(index, packet.bytes(), keyframe)?;

        let video = self.tracks[index].video;
        // PES header data: 3 bytes of flags and up to 10 of timestamps.
        if !video && bytes.len() + 13 > 0xFFFF {
            return Err(MuxError::HostileSize {
                declared: bytes.len() as u64,
                limit: 0xFFFF - 13,
            });
        }
        let psi_due = !self.started
            || self.last_psi.is_none_or(|last| dts - last >= PSI_PERIOD)
            || video && keyframe;
        if psi_due {
            self.write_tables();
            self.last_psi = Some(dts);
            self.started = true;
        }
        let pcr_index = self.pcr_track();
        let pcr_now = ((dts - PCR_LEAD.min(self.options.delay)).max(0) as u64) * 300;
        let pcr_now = self.last_pcr.map_or(pcr_now, |last| pcr_now.max(last));
        let pcr_due = self
            .last_pcr
            .is_none_or(|last| pcr_now.saturating_sub(last) >= PCR_PERIOD);
        let mut pcr = None;
        if pcr_due {
            if index == pcr_index {
                pcr = Some(pcr_now);
            } else {
                let track = &self.tracks[pcr_index];
                // A PCR-only packet carries no payload: the continuity
                // counter does not move.
                let continuity = track.continuity.wrapping_sub(1) & 0x0F;
                let pid = track.pid;
                self.write_pcr_only(pid, continuity, pcr_now);
            }
            self.last_pcr = Some(pcr_now);
        }

        let header = pes_header(
            self.tracks[index].stream_id,
            pts,
            (dts != pts).then_some(dts),
            bytes.len(),
            video,
        );
        let mut pes = header;
        pes.extend_from_slice(&bytes);
        self.write_pes(index, &pes, pcr, keyframe && (video || index == pcr_index));
        self.tracks[index].last_dts = Some(dts);
        if self.last_time.is_some_and(|last| dts < last) {
            self.in_order = false;
        }
        self.last_time = Some(self.last_time.map_or(dts, |last| last.max(dts)));
        self.payload = self.payload.saturating_add(declared);
        if self.out.len() >= FLUSH_BYTES {
            self.flush_out()?;
        }
        Ok(())
    }
}

impl<W: Write> Mux for TsMux<W> {
    fn family(&self) -> ContainerId {
        ContainerId::MpegTs
    }

    fn add_stream(&mut self, stream: NewStream) -> Result<TrackId, MuxError> {
        self.building()?;
        if self.started {
            return Err(self.fail(rejected(
                "the PMT is already written; add streams before packets".to_owned(),
            )));
        }
        if !stream.timescale.is_clock() {
            return Err(self.fail(MuxError::Clockless));
        }
        let count = u32::try_from(self.tracks.len()).unwrap_or(u32::MAX);
        let limit = self.options.limits.max_tracks.min(0x0F00);
        if count >= limit {
            return Err(self.fail(MuxError::TooManyTracks { count, limit }));
        }
        let codec = match stream.params.codec() {
            StreamCodec::Known(codec) => *codec,
            StreamCodec::Unknown(unknown) => {
                return Err(self.fail(MuxError::UnknownCodec(unknown.clone())));
            }
        };
        let videos = self.tracks.iter().filter(|track| track.video).count() as u8;
        let audios = self.tracks.iter().filter(|track| !track.video).count() as u8;
        let video_id = 0xE0 | (videos & 0x0F);
        let audio_id = 0xC0 | (audios & 0x1F);
        let (stream_type, stream_id, shape, mut descriptors, video) = match codec {
            CodecId::H264 => (
                0x1B,
                video_id,
                Shape::H264 {
                    params: Vec::new(),
                    length: None,
                },
                Vec::new(),
                true,
            ),
            CodecId::Hevc => (
                0x24,
                video_id,
                Shape::Hevc {
                    params: Vec::new(),
                    length: None,
                },
                registration(*b"HEVC"),
                true,
            ),
            CodecId::Mpeg2 => (
                0x02,
                video_id,
                Shape::Mpeg2 {
                    sequence: Vec::new(),
                },
                Vec::new(),
                true,
            ),
            CodecId::Aac => (
                0x0F,
                audio_id,
                Shape::Aac { header: None },
                Vec::new(),
                false,
            ),
            CodecId::Mp3 => {
                let low = stream
                    .params
                    .sample_rate()
                    .is_some_and(|rate| rate < 32_000);
                (
                    if low { 0x04 } else { 0x03 },
                    audio_id,
                    Shape::Plain,
                    Vec::new(),
                    false,
                )
            }
            CodecId::Ac3 => (0x81, 0xBD, Shape::Plain, registration(*b"AC-3"), false),
            CodecId::Eac3 => (0x87, 0xBD, Shape::Plain, registration(*b"EAC3"), false),
            CodecId::Opus => {
                let code = opus_code(stream.params.extradata(), stream.params.channels())?;
                (0x06, 0xBD, Shape::Opus, opus_descriptors(code, &[]), false)
            }
            other => {
                return Err(self.fail(rejected(format!(
                    "this MPEG-TS muxer does not carry {other:?}"
                ))));
            }
        };
        if let Some(language) = stream.language.as_ref().and_then(language_code) {
            descriptors.extend_from_slice(&[0x0A, 4]);
            descriptors.extend_from_slice(&language);
            descriptors.push(0);
        }
        let pid = FIRST_PID + self.tracks.len() as u16;
        let id = TrackId::new(count);
        self.tracks.push(Track {
            id,
            pid,
            stream_id,
            stream_type,
            descriptors,
            timescale: stream.timescale,
            shape,
            video,
            continuity: 0,
            last_dts: None,
        });
        let index = self.tracks.len() - 1;
        if !stream.params.extradata().is_empty()
            && !matches!(codec, CodecId::Opus)
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
        self.write_unit(index, packet)
            .map_err(|error| self.fail(error))
    }

    /// Packets are written as they arrive. This checks the mux is still
    /// building and moves nothing.
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
        for duration in durations {
            if self.track_index(duration.track.get()).is_err() {
                let error = self.fail(MuxError::UnknownTrack {
                    id: duration.track.get(),
                });
                return Err(FailedMux::incomplete(self.payload, error));
            }
        }
        if !self.started && !self.tracks.is_empty() {
            self.write_tables();
            self.started = true;
        }
        let flushed = self.flush_out().and_then(|()| {
            self.writer
                .flush()
                .map_err(|error| rejected(format!("flush failed: {error}")))
        });
        if let Err(error) = flushed {
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

    /// Stores an `avcC` or `hvcC` record (or Annex B parameter sets), an
    /// MPEG-2 sequence header, an AAC AudioSpecificConfig, or an
    /// `OpusHead`. A video configuration may change between packets.
    fn set_config(&mut self, track: TrackId, config: &[u8]) -> Result<(), MuxError> {
        self.building()?;
        let index = self
            .track_index(track.get())
            .map_err(|error| self.fail(error))?;
        if self.started && matches!(self.tracks[index].shape, Shape::Opus) {
            return Err(self.fail(rejected(
                "the PMT is already written; set the Opus configuration before packets".to_owned(),
            )));
        }
        self.store_config(index, config)
            .map_err(|error| self.fail(error))
    }
}

fn rejected(message: String) -> MuxError {
    MuxError::Rejected { message }
}

fn registration(tag: [u8; 4]) -> Vec<u8> {
    vec![0x05, 4, tag[0], tag[1], tag[2], tag[3]]
}

fn language_code(language: &Language) -> Option<[u8; 3]> {
    let bytes = language.as_str().as_bytes();
    match bytes {
        [a, b, c] if bytes.iter().all(u8::is_ascii_alphabetic) => Some([
            a.to_ascii_lowercase(),
            b.to_ascii_lowercase(),
            c.to_ascii_lowercase(),
        ]),
        _ => None,
    }
}

/// A long-form PSI section with version 0, current, one section.
fn section(table: u8, id: u16, body: &[u8]) -> Vec<u8> {
    let length = body.len() + 5 + 4;
    let mut out = vec![
        table,
        0xB0 | (length >> 8) as u8 & 0x0F,
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

/// The PES header of one access unit. Video is written with a
/// `PES_packet_length` of 0, which H.222.0 allows for video in a transport
/// stream and FFmpeg writes by default; other streams state their length,
/// which the caller checked fits.
fn pes_header(stream_id: u8, pts: i64, dts: Option<i64>, payload: usize, video: bool) -> Vec<u8> {
    let fields = if dts.is_some() { 10 } else { 5 };
    let length = 3 + fields + payload;
    let declared = if video { 0 } else { length.min(0xFFFF) as u16 };
    let mut out = Vec::with_capacity(9 + fields);
    out.extend_from_slice(&[0, 0, 1, stream_id]);
    out.extend_from_slice(&declared.to_be_bytes());
    out.push(0x84);
    out.push(if dts.is_some() { 0xC0 } else { 0x80 });
    out.push(fields as u8);
    let wrap = |time: i64| (time.rem_euclid(WRAP)) as u64;
    out.extend_from_slice(&timestamp_bytes(
        if dts.is_some() { 3 } else { 2 },
        wrap(pts),
    ));
    if let Some(dts) = dts {
        out.extend_from_slice(&timestamp_bytes(1, wrap(dts)));
    }
    out
}

fn adts_header(profile: u8, rate: u8, channels: u8, length: usize) -> [u8; 7] {
    [
        0xFF,
        0xF1,
        (profile << 6) | (rate << 2) | ((channels >> 2) & 1),
        ((channels & 3) << 6) | ((length >> 11) as u8 & 3),
        (length >> 3) as u8,
        (((length & 7) as u8) << 5) | 0x1F,
        0xFC,
    ]
}

/// The ADTS profile, rate index and channel configuration of an
/// AudioSpecificConfig. SBR and PS signal their AAC core.
fn parse_asc(asc: &[u8]) -> Option<(u8, u8, u8)> {
    let mut bits = Bits::new(asc);
    let mut object = bits.bits(5)?;
    if object == 31 {
        object = 32 + bits.bits(6)?;
    }
    let rate = bits.bits(4)?;
    if rate == 15 {
        return None;
    }
    let channels = bits.bits(4)?;
    if object == 5 || object == 29 {
        let extension = bits.bits(4)?;
        if extension == 15 {
            bits.skip(24)?;
        }
        object = bits.bits(5)?;
    }
    if !(1..=4).contains(&object) || channels == 0 || channels > 7 {
        return None;
    }
    Some(((object - 1) as u8, rate as u8, channels as u8))
}

#[derive(Clone, Copy)]
enum ConfigKind {
    Avc,
    Hevc,
}

/// Annex B parameter sets and the NAL length size from an `avcC` or `hvcC`
/// record, or Annex B parameter sets as they are.
fn parse_config(config: &[u8], kind: ConfigKind) -> Result<(Vec<u8>, Option<u8>), MuxError> {
    if starts_with_start_code(config) {
        return Ok((config.to_vec(), None));
    }
    let bad = || rejected("the codec configuration record is malformed".to_owned());
    let mut out = Vec::new();
    let mut push = |nal: &[u8]| {
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(nal);
    };
    let take = |at: &mut usize, count: usize| -> Option<&[u8]> {
        let slice = config.get(*at..at.checked_add(count)?)?;
        *at += count;
        Some(slice)
    };
    match kind {
        ConfigKind::Avc => {
            if config.first() != Some(&1) || config.len() < 7 {
                return Err(bad());
            }
            let length = (config[4] & 3) + 1;
            let mut at = 5;
            let sets = take(&mut at, 1).ok_or_else(bad)?[0] & 0x1F;
            for _ in 0..sets {
                let size = take(&mut at, 2).ok_or_else(bad)?;
                let size = usize::from(u16::from_be_bytes([size[0], size[1]]));
                push(take(&mut at, size).ok_or_else(bad)?);
            }
            let sets = take(&mut at, 1).ok_or_else(bad)?[0];
            for _ in 0..sets {
                let size = take(&mut at, 2).ok_or_else(bad)?;
                let size = usize::from(u16::from_be_bytes([size[0], size[1]]));
                push(take(&mut at, size).ok_or_else(bad)?);
            }
            Ok((out, Some(length)))
        }
        ConfigKind::Hevc => {
            if config.len() < 23 {
                return Err(bad());
            }
            let length = (config[21] & 3) + 1;
            let mut at = 22;
            let arrays = take(&mut at, 1).ok_or_else(bad)?[0];
            for _ in 0..arrays {
                take(&mut at, 1).ok_or_else(bad)?;
                let count = take(&mut at, 2).ok_or_else(bad)?;
                let count = u16::from_be_bytes([count[0], count[1]]);
                for _ in 0..count {
                    let size = take(&mut at, 2).ok_or_else(bad)?;
                    let size = usize::from(u16::from_be_bytes([size[0], size[1]]));
                    push(take(&mut at, size).ok_or_else(bad)?);
                }
            }
            Ok((out, Some(length)))
        }
    }
}

/// The unit types an H.264 or HEVC access unit is checked for.
struct Units {
    /// The unit type of a NAL header's first byte.
    kind: fn(u8) -> u8,
    delimiter: &'static [u8],
    delimiter_type: u8,
    /// A sequence parameter set.
    sequence: u8,
}

const H264_UNITS: Units = Units {
    kind: |byte| byte & 0x1F,
    delimiter: &[0, 0, 0, 1, 0x09, 0xF0],
    delimiter_type: 9,
    sequence: 7,
};

const HEVC_UNITS: Units = Units {
    kind: |byte| (byte >> 1) & 0x3F,
    delimiter: &[0, 0, 0, 1, 0x46, 0x01, 0x50],
    delimiter_type: 35,
    sequence: 33,
};

/// An access unit as Annex B: from length-prefixed units when `length` is
/// set or the bytes do not start with a start code, with an access unit
/// delimiter first and `params` before a keyframe that has no SPS.
fn annex_b(
    bytes: &[u8],
    length: Option<u8>,
    params: &[u8],
    keyframe: bool,
    units: &Units,
) -> Result<Vec<u8>, MuxError> {
    let mut nals: Vec<&[u8]> = Vec::new();
    // With a configuration record the packets are length-prefixed: a NAL of
    // 256 to 511 bytes has a length field that reads as `00 00 01`.
    if length.is_none() && starts_with_start_code(bytes) {
        let mut at = find_start_code(bytes, 0);
        while let Some(start) = at {
            let next = find_start_code(bytes, start + 3);
            let mut end = next.unwrap_or(bytes.len());
            // A zero byte before the next start code belongs to it.
            while end > start + 3 && next.is_some() && bytes[end - 1] == 0 {
                end -= 1;
            }
            nals.push(&bytes[start + 3..end]);
            at = next;
        }
    } else {
        let size = usize::from(length.unwrap_or(4));
        let mut at = 0;
        while at < bytes.len() {
            let Some(field) = bytes.get(at..at + size) else {
                return Err(rejected("a NAL length field is cut short".to_owned()));
            };
            let nal_length = field
                .iter()
                .fold(0_usize, |value, &byte| (value << 8) | usize::from(byte));
            at += size;
            let Some(nal) = bytes.get(at..at.saturating_add(nal_length)) else {
                return Err(rejected("a NAL unit runs past the packet".to_owned()));
            };
            nals.push(nal);
            at += nal_length;
        }
    }
    let kinds: Vec<u8> = nals
        .iter()
        .map(|nal| nal.first().map_or(0xFF, |&byte| (units.kind)(byte)))
        .collect();
    let mut out = Vec::with_capacity(bytes.len() + params.len() + 64);
    if kinds.first() != Some(&units.delimiter_type) {
        out.extend_from_slice(units.delimiter);
    }
    let has_sequence = kinds.contains(&units.sequence);
    let mut params_written = !keyframe || has_sequence || params.is_empty();
    for (nal, kind) in nals.iter().zip(&kinds) {
        if !params_written && *kind != units.delimiter_type {
            out.extend_from_slice(params);
            params_written = true;
        }
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(nal);
    }
    if !params_written {
        out.extend_from_slice(params);
    }
    Ok(out)
}

/// The Opus `channel_config_code` for an `OpusHead`, or for a channel
/// count when there is none (FFmpeg's rules).
fn opus_code(head: &[u8], channels: Option<u16>) -> Result<u8, MuxError> {
    const COUPLED: [u8; 9] = [1, 0, 1, 1, 2, 2, 2, 3, 3];
    const MAPS: [&[u8]; 8] = [
        &[0],
        &[0, 1],
        &[0, 2, 1],
        &[0, 1, 2, 3],
        &[0, 4, 1, 2, 3],
        &[0, 4, 1, 2, 3, 5],
        &[0, 4, 1, 2, 3, 5, 6],
        &[0, 6, 1, 2, 3, 4, 5, 7],
    ];
    let unsupported = || rejected("this Opus channel mapping has no MPEG-TS code".to_owned());
    if head.len() >= 19 && head.starts_with(b"OpusHead") {
        let count = head[9];
        match head[18] {
            0 if (1..=2).contains(&count) => return Ok(count),
            1 if (1..=8).contains(&count) && head.len() >= 21 + usize::from(count) => {
                let coupled = COUPLED[usize::from(count)];
                if head[19] == count - coupled
                    && head[20] == coupled
                    && head[21..21 + usize::from(count)] == *MAPS[usize::from(count) - 1]
                {
                    return Ok(count);
                }
                return Err(unsupported());
            }
            _ => return Err(unsupported()),
        }
    }
    match channels {
        Some(count @ 1..=2) => Ok(count as u8),
        _ => Err(unsupported()),
    }
}

/// `existing` without its Opus descriptors, then the registration and
/// extension descriptors for `code`.
fn opus_descriptors(code: u8, existing: &[u8]) -> Vec<u8> {
    let mut out = registration(*b"Opus");
    out.extend_from_slice(&[0x7F, 2, 0x80, code]);
    // Keep a language descriptor added after the Opus ones.
    let mut at = 0;
    while at + 2 <= existing.len() {
        let tag = existing[at];
        let length = usize::from(existing[at + 1]);
        let end = (at + 2 + length).min(existing.len());
        if tag == 0x0A {
            out.extend_from_slice(&existing[at..end]);
        }
        at = end;
    }
    out
}
