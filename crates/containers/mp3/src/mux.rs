//! MP3 muxer.
//!
//! Packet bytes are written unchanged. A Xing or Info frame is placed first
//! so a demuxer can read the duration and seek without scanning. That frame
//! is not one of the caller's packets. `finish` fills the frame count, the
//! byte count, and the TOC. Until then the count in the file is zero, so a
//! failed finish does not advertise a complete duration.
//!
//! A stream that has a trim when its first packet arrives also gets an
//! ID3v2.3 tag ahead of that frame with an `iTunSMPB` comment, the gapless
//! record Apple's decoders read; the LAME tag in the Info frame is the one
//! LAME and mpg123 read. Both are filled at finish.

use std::io::{BufWriter, Seek, SeekFrom, Write};

use cutlass_container::{
    FailedMux, FinishedMux, IndexPlace, Limits, MuxDraft, MuxError, MuxMode, MuxStatus, NewStream,
    TrackDuration, TrackId, Trim,
};

use crate::api::{CodecId, CodecParams, ContainerId, Packet};
use crate::api::{ContainerError, Mux};

use crate::header::{
    Header, MpegVersion, classify_rate, compatible, encode_layer3, layer3_bitrate_index,
    parse_header, side_info_len,
};

const BAGS: usize = 400;
const TOC: usize = 100;
/// Bytes from the Xing id through the LAME tag CRC.
const XING_TAG_BYTES: usize = 156;

/// Writes one Layer III stream.
pub struct Mp3Mux<W: Write + Seek> {
    writer: BufWriter<W>,
    params: Option<CodecParams>,
    stream: Option<StreamLock>,
    xing_pos: u64,
    /// Where the `iTunSMPB` text starts, when the tag was written.
    smpb_pos: Option<u64>,
    xing_frame: Vec<u8>,
    xing_at: usize,
    frames: u32,
    size: u64,
    bag: Vec<u64>,
    bag_pos: usize,
    want: u32,
    seen: u32,
    bitrate_kbps: Option<u32>,
    variable: bool,
    next_pts: i64,
    finished: bool,
    cancelled: bool,
    /// The bookkeeping of the shared [`cutlass_container::Mux`] contract.
    draft: MuxDraft,
}

struct StreamLock {
    header: Header,
}

impl<W: Write + Seek> Mp3Mux<W> {
    #[must_use]
    pub fn new(writer: W) -> Self {
        Self {
            writer: BufWriter::with_capacity(64 * 1024, writer),
            params: None,
            stream: None,
            xing_pos: 0,
            smpb_pos: None,
            xing_frame: Vec::new(),
            xing_at: 0,
            frames: 0,
            size: 0,
            bag: vec![0; BAGS],
            bag_pos: 0,
            want: 1,
            seen: 0,
            bitrate_kbps: None,
            variable: false,
            next_pts: 0,
            finished: false,
            cancelled: false,
            draft: MuxDraft::new(
                ContainerId::Mp3,
                MuxMode::Progressive,
                IndexPlace::Start,
                single_track(),
            ),
        }
    }

    /// Set the encoder delay and trailing padding the LAME tag carries,
    /// replacing what `add_stream` gave. Both fit in 12 bits.
    pub fn set_trim(
        &mut self,
        encoder_delay: u32,
        trailing_padding: u32,
    ) -> Result<(), ContainerError> {
        if self.finished {
            return Err(ContainerError::InvalidApi {
                what: "trim after finish",
            });
        }
        if encoder_delay > 4095 || trailing_padding > 4095 {
            return Err(ContainerError::Limit {
                what: "encoder delay",
            });
        }
        let params = self.params.as_mut().ok_or(ContainerError::InvalidApi {
            what: "trim before add_stream",
        })?;
        params.encoder_delay = encoder_delay;
        params.trailing_padding = trailing_padding;
        Ok(())
    }

    pub fn into_inner(self) -> Result<W, ContainerError> {
        self.writer.into_inner().map_err(|err| ContainerError::Io {
            kind: err.error().kind(),
        })
    }

    fn write_all(&mut self, buf: &[u8]) -> Result<(), ContainerError> {
        self.writer.write_all(buf)?;
        Ok(())
    }

    fn note_frame(&mut self, len: u64) {
        self.frames = self.frames.saturating_add(1);
        self.seen = self.seen.saturating_add(1);
        self.size = self.size.saturating_add(len);
        if self.want != self.seen {
            return;
        }
        if self.bag_pos < BAGS {
            self.bag[self.bag_pos] = self.size;
            self.bag_pos += 1;
        }
        self.seen = 0;
        if self.bag_pos == BAGS {
            let mut index = 1;
            while index < BAGS {
                self.bag[index / 2] = self.bag[index];
                index += 2;
            }
            self.want = self.want.saturating_mul(2);
            self.bag_pos = BAGS / 2;
        }
    }

    fn patch_xing(&mut self) -> Result<(), ContainerError> {
        if self.xing_frame.is_empty() {
            return Ok(());
        }
        let tag = if self.variable { b"Xing" } else { b"Info" };
        let at = self.xing_at;
        self.xing_frame[at..at + 4].copy_from_slice(tag);
        self.xing_frame[at + 8..at + 12].copy_from_slice(&self.frames.to_be_bytes());
        let bytes = u32::try_from(self.size).unwrap_or(u32::MAX);
        self.xing_frame[at + 12..at + 16].copy_from_slice(&bytes.to_be_bytes());
        self.xing_frame[at + 16] = 0;
        if self.bag_pos > 0 && self.size > 0 {
            for percent in 1..TOC {
                let slot = percent * self.bag_pos / TOC;
                let slot = slot.min(self.bag_pos - 1);
                let point = (256 * self.bag[slot] / self.size).min(255);
                self.xing_frame[at + 16 + percent] = u8::try_from(point).unwrap_or(255);
            }
        }
        let params = self
            .params
            .as_ref()
            .ok_or(ContainerError::InvalidApi { what: "finish" })?;
        if params.encoder_delay > 4095 || params.trailing_padding > 4095 {
            return Err(ContainerError::Limit {
                what: "encoder delay",
            });
        }
        let packed = (params.encoder_delay << 12) | params.trailing_padding;
        let be = packed.to_be_bytes();
        self.xing_frame[at + 141] = be[1];
        self.xing_frame[at + 142] = be[2];
        self.xing_frame[at + 143] = be[3];
        let end = self.writer.stream_position()?;
        if let Some(at) = self.smpb_pos {
            let total = u64::try_from(self.next_pts).unwrap_or(0);
            let trim = u64::from(params.encoder_delay) + u64::from(params.trailing_padding);
            let text = smpb_text(
                params.encoder_delay,
                params.trailing_padding,
                total.saturating_sub(trim),
            );
            self.writer.seek(SeekFrom::Start(at))?;
            self.writer.write_all(text.as_bytes())?;
        }
        self.writer.seek(SeekFrom::Start(self.xing_pos))?;
        self.writer.write_all(&self.xing_frame)?;
        self.writer.seek(SeekFrom::Start(end))?;
        self.writer.flush()?;
        Ok(())
    }
}

impl<W: Write + Seek> Mux for Mp3Mux<W> {
    fn container_id(&self) -> ContainerId {
        ContainerId::Mp3
    }

    fn add_stream(&mut self, params: CodecParams) -> Result<u32, ContainerError> {
        if self.finished {
            return Err(ContainerError::InvalidApi {
                what: "add_stream after finish",
            });
        }
        if params.codec != CodecId::Mp3 {
            return Err(ContainerError::Unsupported { what: "codec" });
        }
        if self.params.is_some() {
            return Err(ContainerError::Unsupported { what: "streams" });
        }
        if params.channels != 1 && params.channels != 2 {
            return Err(ContainerError::Unsupported { what: "channels" });
        }
        if classify_rate(params.sample_rate).is_none() {
            return Err(ContainerError::Unsupported {
                what: "sample rate",
            });
        }
        if !params.extra.is_empty() {
            return Err(ContainerError::Unsupported { what: "extradata" });
        }
        if params.encoder_delay > 4095 || params.trailing_padding > 4095 {
            return Err(ContainerError::Limit {
                what: "encoder delay",
            });
        }
        self.params = Some(params);
        Ok(0)
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<(), ContainerError> {
        if self.cancelled {
            return Err(ContainerError::Cancelled);
        }
        if self.finished {
            return Err(ContainerError::InvalidApi {
                what: "write after finish",
            });
        }
        let params = self.params.as_ref().ok_or(ContainerError::InvalidApi {
            what: "write before add_stream",
        })?;
        if packet.stream != 0 {
            return Err(ContainerError::InvalidApi { what: "stream" });
        }
        if packet.timescale != params.sample_rate {
            return Err(ContainerError::InvalidApi { what: "timescale" });
        }
        if packet.data.len() < 4 {
            return Err(ContainerError::Malformed { what: "frame" });
        }
        let raw = u32::from_be_bytes([
            packet.data[0],
            packet.data[1],
            packet.data[2],
            packet.data[3],
        ]);
        let header = parse_header(raw).ok_or(ContainerError::Malformed { what: "frame" })?;
        if header.layer != 3 {
            return Err(ContainerError::Unsupported { what: "layer" });
        }
        if header.sample_rate != params.sample_rate || header.channels != params.channels {
            return Err(ContainerError::Malformed {
                what: "sample rate",
            });
        }
        let size = match header.frame_bytes {
            Some(size) => size,
            None => u32::try_from(packet.data.len())
                .map_err(|_| ContainerError::Limit { what: "frame" })?,
        };
        if usize::try_from(size).ok() != Some(packet.data.len())
            || size > crate::limits::MAX_FRAME_BYTES
        {
            return Err(ContainerError::Malformed { what: "frame" });
        }
        if header.bitrate_kbps == 0 {
            let unpadded = size.saturating_sub(header.padding);
            if unpadded < 4 {
                return Err(ContainerError::Unsupported {
                    what: "free format",
                });
            }
        }
        if crate::tag::is_tag_frame(&packet.data, &header) {
            return Err(ContainerError::Malformed { what: "vbr header" });
        }
        let duration = i64::from(header.samples_per_frame);
        if packet.duration != duration || packet.pts != self.next_pts {
            return Err(ContainerError::InvalidApi { what: "timestamp" });
        }

        if self.stream.is_none() {
            if params.encoder_delay > 0 || params.trailing_padding > 0 {
                let (tag, text_at) = smpb_tag();
                let start = self.writer.stream_position()?;
                self.write_all(&tag)?;
                self.smpb_pos = Some(start + text_at);
            }
            let params = self.params.as_ref().ok_or(ContainerError::InvalidApi {
                what: "write before add_stream",
            })?;
            let frame = build_xing_frame(&header, params)?;
            self.xing_at = 4 + side_info_len(&header);
            self.xing_pos = self.writer.stream_position()?;
            self.write_all(&frame)?;
            self.size = u64::try_from(frame.len()).unwrap_or(0);
            self.xing_frame = frame;
            self.stream = Some(StreamLock { header });
        } else if let Some(stream) = &self.stream
            && !compatible(&stream.header, &header)
        {
            return Err(ContainerError::Malformed {
                what: "sample rate",
            });
        }

        match self.bitrate_kbps {
            None => self.bitrate_kbps = Some(header.bitrate_kbps),
            Some(prev) if prev != header.bitrate_kbps => self.variable = true,
            Some(_) => {}
        }
        if header.bitrate_kbps == 0 {
            self.variable = true;
        }

        let len = u64::try_from(packet.data.len()).unwrap_or(0);
        self.note_frame(len);
        self.write_all(&packet.data)?;
        self.next_pts = self
            .next_pts
            .checked_add(duration)
            .ok_or(ContainerError::Limit { what: "timestamp" })?;
        Ok(())
    }

    fn finish(&mut self) -> Result<(), ContainerError> {
        if self.cancelled {
            return Err(ContainerError::Cancelled);
        }
        if self.finished {
            return Err(ContainerError::InvalidApi {
                what: "second finish",
            });
        }
        if self.params.is_none() {
            return Err(ContainerError::InvalidApi { what: "finish" });
        }
        if self.frames == 0 {
            return Err(ContainerError::NoFrames);
        }
        self.patch_xing()?;
        self.finished = true;
        Ok(())
    }

    fn cancel(&mut self) {
        self.cancelled = true;
    }
}

/// Apple's gapless comment: priming, padding and the sample count in hex,
/// fixed width so finish can rewrite it in place.
fn smpb_text(delay: u32, padding: u32, samples: u64) -> String {
    format!(
        " 00000000 {delay:08X} {padding:08X} {samples:016X} 00000000 00000000 00000000 \
         00000000 00000000 00000000 00000000 00000000"
    )
}

/// An ID3v2.3 tag holding one `COMM` frame, description `iTunSMPB`, and
/// where in the tag its text starts.
fn smpb_tag() -> (Vec<u8>, u64) {
    let text = smpb_text(0, 0, 0);
    let mut body = vec![0u8]; // ISO-8859-1
    body.extend_from_slice(b"eng");
    body.extend_from_slice(b"iTunSMPB\0");
    let text_in_body = body.len();
    body.extend_from_slice(text.as_bytes());
    let body_len = u32::try_from(body.len()).unwrap_or(0);
    let mut frame = b"COMM".to_vec();
    frame.extend_from_slice(&body_len.to_be_bytes());
    frame.extend_from_slice(&[0, 0]);
    let text_in_frame = frame.len() + text_in_body;
    frame.extend_from_slice(&body);
    let size = u32::try_from(frame.len()).unwrap_or(0);
    let mut tag = b"ID3\x03\x00\x00".to_vec();
    // Syncsafe: seven bits a byte.
    tag.extend_from_slice(&[
        ((size >> 21) & 0x7F) as u8,
        ((size >> 14) & 0x7F) as u8,
        ((size >> 7) & 0x7F) as u8,
        (size & 0x7F) as u8,
    ]);
    let text_at = (tag.len() + text_in_frame) as u64;
    tag.extend_from_slice(&frame);
    (tag, text_at)
}

fn single_track() -> Limits {
    let mut limits = Limits::standard();
    limits.max_tracks = 1;
    limits
}

fn rejected(error: ContainerError) -> MuxError {
    match error {
        ContainerError::Cancelled => MuxError::Cancelled,
        other => MuxError::Rejected {
            message: format!("mp3 {other}"),
        },
    }
}

/// The workspace mux contract: packets are written as they arrive, and the
/// Info or Xing frame, with the trim [`cutlass_container::Mux::set_trim`]
/// gave, is filled at finish.
impl<W: Write + Seek> cutlass_container::Mux for Mp3Mux<W> {
    fn family(&self) -> ContainerId {
        ContainerId::Mp3
    }

    fn add_stream(&mut self, stream: NewStream) -> Result<TrackId, MuxError> {
        let (Some(rate), Some(channels)) = (stream.params.sample_rate(), stream.params.channels())
        else {
            return Err(MuxError::Rejected {
                message: "mp3 stores one audio stream".to_owned(),
            });
        };
        let Some(codec) = stream.params.codec().clone().known() else {
            return Err(MuxError::Rejected {
                message: "mp3 stores Layer III only".to_owned(),
            });
        };
        if stream.timescale.ticks_per_second() != rate {
            return Err(MuxError::TimescaleMismatch {
                packet: stream.timescale.ticks_per_second(),
                stream: rate,
            });
        }
        let mut params = CodecParams::audio(codec, rate, channels);
        params.extra = stream.params.extradata().to_vec();
        if let Err(error) = Mux::add_stream(self, params) {
            return Err(rejected(error));
        }
        self.draft.add_stream(stream)
    }

    fn write_packet(&mut self, packet: &cutlass_codec::Packet) -> Result<(), MuxError> {
        let (Some(pts), Some(duration)) = (
            packet.timestamp().presentation_or_decode(),
            packet.timestamp().duration(),
        ) else {
            let error = MuxError::MissingTimestamp;
            self.draft.abort(error.clone());
            return Err(error);
        };
        let mut frame = Packet::audio(
            packet.bytes().to_vec(),
            pts,
            duration,
            packet.timescale().ticks_per_second(),
        );
        frame.stream = packet.stream();
        if let Err(error) = Mux::write_packet(self, &frame) {
            let error = rejected(error);
            self.draft.abort(error.clone());
            return Err(error);
        }
        self.draft.write_packet(packet)
    }

    fn interleave(&mut self) -> Result<(), MuxError> {
        self.draft.interleave()
    }

    fn cancel(&mut self) {
        Mux::cancel(self);
        self.draft.cancel();
    }

    fn finish(&mut self, durations: &[TrackDuration]) -> Result<FinishedMux, FailedMux> {
        if self.draft.status() == MuxStatus::Building
            && let Err(error) = Mux::finish(self)
        {
            self.draft.abort(rejected(error));
        }
        self.draft.finish(durations)
    }

    fn status(&self) -> MuxStatus {
        self.draft.status()
    }

    fn set_trim(&mut self, track: TrackId, trim: Trim) -> Result<(), MuxError> {
        if track != TrackId::new(0) {
            return Err(MuxError::UnknownTrack { id: track.get() });
        }
        let delay = u32::try_from(trim.encoder_delay().unwrap_or(0)).unwrap_or(u32::MAX);
        let padding = u32::try_from(trim.trailing_padding().unwrap_or(0)).unwrap_or(u32::MAX);
        Mp3Mux::set_trim(self, delay, padding).map_err(rejected)
    }
}

fn build_xing_frame(audio: &Header, params: &CodecParams) -> Result<Vec<u8>, ContainerError> {
    let (version, _) = classify_rate(audio.sample_rate).ok_or(ContainerError::Unsupported {
        what: "sample rate",
    })?;
    let preferred = if params.bitrate >= 1000 {
        params.bitrate / 1000
    } else if audio.bitrate_kbps > 0 {
        audio.bitrate_kbps
    } else {
        128
    };
    let mut index = layer3_bitrate_index(version, preferred).unwrap_or(1);
    if index == 0 {
        index = 1;
    }
    let (raw, size) = loop {
        let Some(kbps) = bitrate_of(version, index) else {
            return Err(ContainerError::Unsupported { what: "xing frame" });
        };
        let Some((raw, size)) =
            encode_layer3(version, kbps, audio.sample_rate, false, audio.channels)
        else {
            return Err(ContainerError::Unsupported { what: "xing frame" });
        };
        let tag_at = 4 + side_info_len(audio);
        if tag_at + XING_TAG_BYTES <= size as usize {
            break (raw, size);
        }
        if index >= 14 {
            return Err(ContainerError::Unsupported { what: "xing frame" });
        }
        index += 1;
    };
    let mut frame = vec![0u8; size as usize];
    frame[..4].copy_from_slice(&raw.to_be_bytes());
    let at = 4 + side_info_len(audio);
    frame[at..at + 4].copy_from_slice(b"Xing");
    frame[at + 4..at + 8].copy_from_slice(&0x0Fu32.to_be_bytes());
    frame[at + 120..at + 127].copy_from_slice(b"Cutlass");
    Ok(frame)
}

fn bitrate_of(version: MpegVersion, index: u8) -> Option<u32> {
    // Probe the public encoder table by asking for each legal step.
    const MPEG1: [u32; 15] = [
        0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
    ];
    const LSF: [u32; 15] = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160];
    let row = if version == MpegVersion::Mpeg1 {
        &MPEG1
    } else {
        &LSF
    };
    let index = usize::from(index);
    let kbps = *row.get(index)?;
    if kbps == 0 { None } else { Some(kbps) }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::Mp3Mux;
    use crate::api::{CodecId, CodecParams, Packet};
    use crate::api::{ContainerError, Demux, Mux};
    use crate::demux::Mp3Demux;
    use crate::header::{MpegVersion, frame_index, synthetic_frame};

    fn params(rate: u32, channels: u16) -> CodecParams {
        CodecParams::audio(CodecId::Mp3, rate, channels)
    }

    fn packet(frame: Vec<u8>, pts: i64, duration: i64, rate: u32) -> Packet {
        Packet::audio(frame, pts, duration, rate)
    }

    #[test]
    fn the_shared_contract_writes_the_frames_and_the_trim_set_last() {
        use cutlass_codec::{
            Packet as SharedPacket, StreamCodec, StreamParams, Timescale, Timestamp,
        };
        use cutlass_container::{MuxStatus, NewStream, TrackDuration, TrackId, Trim};

        let mut mux = Mp3Mux::new(Cursor::new(Vec::new()));
        let shared: &mut dyn cutlass_container::Mux = &mut mux;
        let params =
            StreamParams::try_audio(StreamCodec::Known(CodecId::Mp3), 48_000, 2, Vec::new())
                .expect("params");
        let track = shared
            .add_stream(NewStream {
                params,
                timescale: Timescale::new(48_000),
                language: None,
            })
            .expect("stream");
        assert_eq!(track, TrackId::new(0));
        for index in 0..4u16 {
            let frame = synthetic_frame(MpegVersion::Mpeg1, 128, 48_000, 2, index);
            let pts = i64::from(index) * 1152;
            let timestamp =
                Timestamp::try_new(Some(pts), Some(pts), Some(1152)).expect("timestamp");
            let packet = SharedPacket::try_new(0, frame, timestamp, Timescale::new(48_000), true)
                .expect("packet");
            // As export hands it over, ahead of each packet: padding on the
            // last packet only.
            let padding = if index == 3 { 1000 } else { 0 };
            shared
                .set_trim(track, Trim::new(Some(576), Some(padding)))
                .expect("trim");
            shared.write_packet(&packet).expect("write");
        }
        assert!(shared.set_trim(TrackId::new(1), Trim::default()).is_err());
        let done = shared
            .finish(&[TrackDuration {
                track,
                timescale: Timescale::new(48_000),
                ticks: 4 * 1152,
            }])
            .expect("finish");
        assert_eq!(done.durations()[0].ticks, 4 * 1152);
        assert_eq!(shared.status(), MuxStatus::Finished);
        let bytes = mux.into_inner().expect("inner").into_inner();
        assert_eq!(&bytes[..4], b"ID3\x03");
        let at = bytes
            .windows(9)
            .position(|w| w == b"iTunSMPB\0")
            .expect("iTunSMPB")
            + 9;
        let text = std::str::from_utf8(&bytes[at..at + 44]).expect("text");
        assert_eq!(
            text,
            format!(" 00000000 00000240 000003E8 {:016X}", 4 * 1152 - 576 - 1000)
        );
        let demux = Mp3Demux::open(Cursor::new(bytes)).expect("open");
        let trim = demux.streams()[0].trim.expect("trim");
        assert_eq!((trim.encoder_delay, trim.trailing_padding), (576, 1000));
    }

    #[test]
    fn the_shared_contract_fails_on_a_wrong_duration_or_a_trim_too_long() {
        use cutlass_codec::{
            Packet as SharedPacket, StreamCodec, StreamParams, Timescale, Timestamp,
        };
        use cutlass_container::{NewStream, TrackDuration, Trim};

        let mut mux = Mp3Mux::new(Cursor::new(Vec::new()));
        let shared: &mut dyn cutlass_container::Mux = &mut mux;
        let params =
            StreamParams::try_audio(StreamCodec::Known(CodecId::Mp3), 44_100, 1, Vec::new())
                .expect("params");
        let track = shared
            .add_stream(NewStream {
                params,
                timescale: Timescale::new(44_100),
                language: None,
            })
            .expect("stream");
        assert!(
            shared
                .set_trim(track, Trim::new(Some(5000), Some(0)))
                .is_err()
        );
        let frame = synthetic_frame(MpegVersion::Mpeg1, 64, 44_100, 1, 0);
        let timestamp = Timestamp::try_new(Some(0), Some(0), Some(1152)).expect("timestamp");
        let packet = SharedPacket::try_new(0, frame, timestamp, Timescale::new(44_100), true)
            .expect("packet");
        shared.write_packet(&packet).expect("write");
        assert!(
            shared
                .finish(&[TrackDuration {
                    track,
                    timescale: Timescale::new(44_100),
                    ticks: 2 * 1152,
                }])
                .is_err()
        );
    }

    #[test]
    fn round_trip_preserves_frame_bytes_and_the_sample_clock() {
        let mut mux = Mp3Mux::new(Cursor::new(Vec::new()));
        mux.add_stream(params(48_000, 2)).expect("stream");
        for index in 0..5u16 {
            let frame = synthetic_frame(MpegVersion::Mpeg1, 128, 48_000, 2, index);
            let pts = i64::from(index) * 1152;
            mux.write_packet(&packet(frame, pts, 1152, 48_000))
                .expect("write");
        }
        mux.finish().expect("finish");
        assert_eq!(
            mux.finish(),
            Err(ContainerError::InvalidApi {
                what: "second finish"
            })
        );
        let cursor = mux.into_inner().expect("inner");
        let bytes = cursor.into_inner();
        assert!(bytes.windows(4).any(|w| w == b"Info"));
        let mut demux = Mp3Demux::open(Cursor::new(bytes)).expect("open");
        assert!(demux.duration_exact());
        assert_eq!(demux.duration(), Some(5 * 1152));
        for index in 0..5u16 {
            let packet = demux.next_packet().expect("pkt").expect("some");
            assert_eq!(frame_index(&packet.data), index);
            assert_eq!(packet.pts, i64::from(index) * 1152);
            assert_eq!(packet.duration, 1152);
            assert_eq!(packet.timescale, 48_000);
        }
        assert!(demux.next_packet().expect("end").is_none());
        let landing = demux.seek(3 * 1152).expect("seek");
        assert_eq!(landing.pts, 3 * 1152);
        assert_eq!(
            frame_index(&demux.next_packet().expect("seek pkt").expect("some").data),
            3
        );
    }

    #[test]
    fn vbr_writes_xing_and_keeps_each_frame() {
        let mut mux = Mp3Mux::new(Cursor::new(Vec::new()));
        mux.add_stream(params(48_000, 2)).expect("stream");
        let rates = [128u32, 96, 160, 112];
        for (index, kbps) in rates.iter().copied().enumerate() {
            let frame = synthetic_frame(MpegVersion::Mpeg1, kbps, 48_000, 2, index as u16);
            let pts = index as i64 * 1152;
            mux.write_packet(&packet(frame, pts, 1152, 48_000))
                .expect("write");
        }
        mux.finish().expect("finish");
        let bytes = mux.into_inner().expect("inner").into_inner();
        assert!(bytes.windows(4).any(|w| w == b"Xing"));
        let mut demux = Mp3Demux::open(Cursor::new(bytes)).expect("open");
        assert_eq!(demux.duration(), Some(4 * 1152));
        for index in 0..4u16 {
            let packet = demux.next_packet().expect("pkt").expect("some");
            assert_eq!(frame_index(&packet.data), index);
            assert_eq!(packet.pts, i64::from(index) * 1152);
        }
    }

    #[test]
    fn rejects_a_foreign_codec_a_second_stream_and_a_gap() {
        let mut mux = Mp3Mux::new(Cursor::new(Vec::new()));
        assert_eq!(mux.add_stream(params(48_000, 2).clone()), Ok(0));
        let mut aac = params(48_000, 2);
        aac.codec = CodecId::Aac;
        assert_eq!(
            mux.add_stream(aac),
            Err(ContainerError::Unsupported { what: "codec" })
        );
        assert_eq!(
            mux.add_stream(params(48_000, 2)),
            Err(ContainerError::Unsupported { what: "streams" })
        );
        let frame = synthetic_frame(MpegVersion::Mpeg1, 128, 48_000, 2, 0);
        mux.write_packet(&packet(frame, 1152, 1152, 48_000))
            .expect_err("gap");
        let empty = Mp3Mux::new(Cursor::new(Vec::new()));
        let mut empty = empty;
        empty.add_stream(params(48_000, 2)).expect("stream");
        assert_eq!(empty.finish(), Err(ContainerError::NoFrames));
    }

    #[test]
    fn gapless_metadata_round_trips_without_dropping_frames() {
        let mut mux = Mp3Mux::new(Cursor::new(Vec::new()));
        let mut stream = params(48_000, 2);
        stream.encoder_delay = 576;
        stream.trailing_padding = 100;
        mux.add_stream(stream).expect("stream");
        let frame = synthetic_frame(MpegVersion::Mpeg1, 128, 48_000, 2, 9);
        mux.write_packet(&packet(frame, 0, 1152, 48_000))
            .expect("write");
        mux.finish().expect("finish");
        let bytes = mux.into_inner().expect("inner").into_inner();
        let demux = Mp3Demux::open(Cursor::new(bytes)).expect("open");
        let trim = demux.streams()[0].trim.expect("trim");
        assert_eq!(trim.encoder_delay, 576);
        assert_eq!(trim.trailing_padding, 100);
        assert_eq!(demux.duration(), Some(1152));
    }

    #[test]
    fn cancel_leaves_a_zero_frame_count() {
        let mut mux = Mp3Mux::new(Cursor::new(Vec::new()));
        mux.add_stream(params(48_000, 2)).expect("stream");
        let frame = synthetic_frame(MpegVersion::Mpeg1, 128, 48_000, 2, 0);
        mux.write_packet(&packet(frame, 0, 1152, 48_000))
            .expect("write");
        mux.cancel();
        assert_eq!(mux.finish(), Err(ContainerError::Cancelled));
        let bytes = mux.into_inner().expect("inner").into_inner();
        let mut demux = Mp3Demux::open(Cursor::new(bytes)).expect("open");
        let packet = demux.next_packet().expect("pkt").expect("some");
        assert_eq!(frame_index(&packet.data), 0);
        assert!(demux.next_packet().expect("end").is_none());
        assert!(!demux.duration_exact());
    }

    #[test]
    fn free_format_packets_round_trip() {
        let mut mux = Mp3Mux::new(Cursor::new(Vec::new()));
        mux.add_stream(params(48_000, 2)).expect("stream");
        for index in 0..3u16 {
            let mut frame = synthetic_frame(MpegVersion::Mpeg1, 128, 48_000, 2, index);
            let raw = u32::from_be_bytes([frame[0], frame[1], frame[2], frame[3]]);
            frame[..4].copy_from_slice(&(raw & !0x0000_F000).to_be_bytes());
            let pts = i64::from(index) * 1152;
            mux.write_packet(&packet(frame, pts, 1152, 48_000))
                .expect("write");
        }
        mux.finish().expect("finish");
        let bytes = mux.into_inner().expect("inner").into_inner();
        let mut demux = Mp3Demux::open(Cursor::new(bytes)).expect("open");
        assert_eq!(demux.duration(), Some(3 * 1152));
        for index in 0..3u16 {
            let packet = demux.next_packet().expect("pkt").expect("some");
            assert_eq!(frame_index(&packet.data), index);
            assert_eq!(packet.pts, i64::from(index) * 1152);
            assert_eq!(packet.timescale, 48_000);
        }
    }

    #[test]
    fn a_short_packet_a_foreign_rate_and_a_late_write_are_refused() {
        let mut mux = Mp3Mux::new(Cursor::new(Vec::new()));
        mux.add_stream(params(48_000, 2)).expect("stream");
        assert_eq!(
            mux.write_packet(&packet(vec![0, 1, 2], 0, 1152, 48_000)),
            Err(ContainerError::Malformed { what: "frame" })
        );
        let foreign = synthetic_frame(MpegVersion::Mpeg1, 128, 44_100, 2, 0);
        assert_eq!(
            mux.write_packet(&packet(foreign, 0, 1152, 48_000)),
            Err(ContainerError::Malformed {
                what: "sample rate"
            })
        );
        let mut long = synthetic_frame(MpegVersion::Mpeg1, 128, 48_000, 2, 0);
        long.push(0);
        assert_eq!(
            mux.write_packet(&packet(long, 0, 1152, 48_000)),
            Err(ContainerError::Malformed { what: "frame" })
        );
        let frame = synthetic_frame(MpegVersion::Mpeg1, 128, 48_000, 2, 1);
        mux.write_packet(&packet(frame, 0, 1152, 48_000))
            .expect("write");
        mux.finish().expect("finish");
        let again = synthetic_frame(MpegVersion::Mpeg1, 128, 48_000, 2, 2);
        assert_eq!(
            mux.write_packet(&packet(again, 1152, 1152, 48_000)),
            Err(ContainerError::InvalidApi {
                what: "write after finish"
            })
        );
    }
}
