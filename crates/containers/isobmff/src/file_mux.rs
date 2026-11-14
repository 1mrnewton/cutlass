//! An MP4 written through the workspace mux trait, for export.
//!
//! Packets go straight to the file after `mdat` and the index follows at
//! finish, so a long export does not hold its packets in memory. Each
//! track's codec configuration may arrive with its first packet, as encoders
//! hand it over; it is written in the index at the end. An AAC
//! configuration wrapped in an MPEG-4 ES descriptor, as Apple's encoder
//! gives it, is unwrapped to the bare AudioSpecificConfig MP4 stores. An
//! Opus configuration may be an `OpusHead` or a `dOps` body; the file
//! stores `dOps`, and an edit that starts play after the pre-skip.

use std::io::{Seek, Write};

use cutlass_codec::{CodecId as Id, ContainerId, Packet, StreamCodec, Timescale};
use cutlass_container::{
    FailedMux, FinishedMux, IndexPlace, Mux, MuxError, MuxMode, MuxStatus, NewStream,
    TrackDuration, TrackId, Trim,
};

use crate::legacy::{
    self, CodecId, ColorTags, Error, SampleFormat, StreamParams, Timescale as BoxScale,
};
use crate::mux::{FileBrand, IsoBmffMux, MuxOptions};

/// One track: the id handed out, the box writer's id, and the next decode
/// time in the track's timescale.
#[derive(Clone, Copy, Debug)]
struct Track {
    id: TrackId,
    boxed: u32,
    timescale: Timescale,
    codec: CodecId,
    next: i64,
    /// A ProRes track whose `colr` box still waits for its first frame.
    color_from_frame: bool,
    /// The last gapless trim an AC-3 or E-AC-3 encoder handed over.
    trim: Option<Trim>,
}

/// An MP4 file written as packets arrive, with its index at the end.
pub struct Mp4Mux<W: Write + Seek> {
    inner: IsoBmffMux<W>,
    brand: FileBrand,
    tracks: Vec<Track>,
    status: MuxStatus,
    payload: u64,
    failure: Option<MuxError>,
}

impl<W: Write + Seek> Mp4Mux<W> {
    #[must_use]
    pub fn new(writer: W) -> Self {
        Self::with_brand(writer, FileBrand::Mp4)
    }

    /// A file of brand `brand`: `Mov` writes a QuickTime movie, `M4a` a
    /// sound-only MPEG-4 file.
    #[must_use]
    pub fn with_brand(writer: W, brand: FileBrand) -> Self {
        let options = MuxOptions {
            brand,
            ..MuxOptions::default()
        };
        Self {
            inner: IsoBmffMux::new(writer, options),
            brand,
            tracks: Vec::new(),
            status: MuxStatus::Building,
            payload: 0,
            failure: None,
        }
    }

    /// Sets the file's cover art, a JPEG picture, written as iTunes
    /// metadata in `moov/udta` at finish. A refused picture (not a JPEG, or
    /// too large) leaves the file as it was and the muxer building.
    pub fn set_cover_art(&mut self, jpeg: Vec<u8>) -> Result<(), MuxError> {
        self.building()?;
        self.inner.set_cover_art(jpeg).map_err(boxed)
    }

    /// The writer, once the file is finished.
    pub fn into_writer(self) -> Result<W, MuxError> {
        if self.status != MuxStatus::Finished {
            return Err(MuxError::NotBuilding);
        }
        self.inner.finish_into().map_err(boxed)
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
}

fn boxed(error: Error) -> MuxError {
    MuxError::Rejected {
        message: error.to_string(),
    }
}

fn rejected(message: &str) -> MuxError {
    MuxError::Rejected {
        message: message.to_owned(),
    }
}

/// The color code points a ProRes frame header carries (SMPTE RDD 36,
/// bytes 14 to 16 of the header), for the track's `colr` box. A frame
/// that names none of them gives `None`; 0, "unknown", becomes 2,
/// "unspecified".
fn prores_color(frame: &[u8]) -> Option<ColorTags> {
    if frame.get(4..8) != Some(b"icpf".as_slice()) {
        return None;
    }
    let codes = frame.get(22..25)?;
    if codes.iter().all(|&code| code == 0 || code == 2) {
        return None;
    }
    let code = |byte: u8| if byte == 0 { 2 } else { u16::from(byte) };
    Some(ColorTags {
        primaries: code(codes[0]),
        transfer: code(codes[1]),
        matrix: code(codes[2]),
        full_range: false,
    })
}

/// ProRes profiles a picture track may name in its extradata: Proxy, LT,
/// 422, HQ, 4444 and 4444 XQ, by their sample entry.
const PRORES_ENTRIES: [[u8; 4]; 6] = [*b"apco", *b"apcs", *b"apcn", *b"apch", *b"ap4h", *b"ap4x"];

/// The box writer's parameters for `stream`: H.264, HEVC, MPEG-4 Visual,
/// ProRes, Motion JPEG, VP8, AV1 or MPEG-2 pictures, or AAC, Opus, or PCM sound.
/// PCM keeps the layout its three layout bytes name (empty is s16le). A VP8 track's extradata is its `vpcC` payload,
/// and an AV1 track's is its `av1C` record. A ProRes track whose extradata is
/// a profile's four-character code is written with that sample entry.
/// Motion JPEG is a QuickTime `jpeg` entry, so only a movie takes it.
fn box_params(stream: &NewStream, brand: FileBrand) -> Result<StreamParams, MuxError> {
    let StreamCodec::Known(codec) = stream.params.codec() else {
        return Err(rejected(
            "mp4 export writes H.264, HEVC, MPEG-4 Visual, ProRes, Motion JPEG, VP8, AV1, MPEG-2, AAC, ALAC, Opus, AC-3, E-AC-3, and PCM",
        ));
    };
    let scale = BoxScale(stream.timescale.ticks_per_second());
    let mut params = match codec {
        Id::Mjpeg if brand != FileBrand::Mov => {
            return Err(rejected("Motion JPEG is written to a .mov movie"));
        }
        Id::H264
        | Id::Hevc
        | Id::Mpeg4
        | Id::Prores
        | Id::Mjpeg
        | Id::Vp8
        | Id::Av1
        | Id::Mpeg2 => {
            let size = stream
                .params
                .size()
                .ok_or_else(|| rejected("a picture track needs a size"))?;
            let codec = match codec {
                Id::H264 => CodecId::H264,
                Id::Hevc => CodecId::Hevc,
                Id::Mpeg4 => CodecId::Mpeg4,
                Id::Vp8 => CodecId::Vp8,
                Id::Av1 => CodecId::Av1,
                Id::Mjpeg => CodecId::Mjpeg,
                Id::Mpeg2 => CodecId::Mpeg2,
                _ => CodecId::ProRes,
            };
            StreamParams::video(codec, scale, size.width(), size.height())
        }
        Id::Pcm => {
            let rate = stream
                .params
                .sample_rate()
                .ok_or_else(|| rejected("a sound track needs a rate"))?;
            let channels = stream
                .params
                .channels()
                .ok_or_else(|| rejected("a sound track needs channels"))?;
            let (bits, format) = pcm_entry(stream.params.extradata())?;
            // The entry states the layout; the PCM track carries no extradata.
            return Ok(StreamParams::audio(
                CodecId::Pcm,
                scale,
                rate,
                channels,
                bits,
                format,
            ));
        }
        Id::Aac | Id::Opus | Id::Ac3 | Id::Eac3 => {
            let rate = stream
                .params
                .sample_rate()
                .ok_or_else(|| rejected("a sound track needs a rate"))?;
            let channels = stream
                .params
                .channels()
                .ok_or_else(|| rejected("a sound track needs channels"))?;
            let codec = match codec {
                Id::Aac => CodecId::Aac,
                Id::Opus => CodecId::Opus,
                Id::Ac3 => CodecId::Ac3,
                _ => CodecId::Eac3,
            };
            StreamParams::audio(codec, scale, rate, channels, 16, SampleFormat::S16Le)
        }
        Id::Alac => {
            let rate = stream
                .params
                .sample_rate()
                .ok_or_else(|| rejected("a sound track needs a rate"))?;
            let channels = stream
                .params
                .channels()
                .ok_or_else(|| rejected("a sound track needs channels"))?;
            // The entry's depth and count come from the cookie at finish;
            // this is the depth until it arrives.
            StreamParams::audio(
                CodecId::Alac,
                scale,
                rate,
                channels,
                16,
                SampleFormat::Unknown,
            )
        }
        _ => {
            return Err(rejected(
                "mp4 export writes H.264, HEVC, MPEG-4 Visual, ProRes, Motion JPEG, VP8, AV1, MPEG-2, AAC, ALAC, Opus, AC-3, E-AC-3, and PCM",
            ));
        }
    };
    let extradata = stream.params.extradata();
    if *codec == Id::Mjpeg {
        // Each frame is a whole JPEG image; the entry holds no configuration.
        if !extradata.is_empty() && extradata != b"jpeg" {
            return Err(rejected("Motion JPEG extradata other than `jpeg`"));
        }
        params.sample_entry = *b"jpeg";
    } else if *codec == Id::Prores
        && let Some(entry) = PRORES_ENTRIES
            .iter()
            .find(|entry| extradata == entry.as_slice())
    {
        params.sample_entry = *entry;
    } else {
        params.extradata = extradata.to_vec();
    }
    Ok(params)
}

/// The bit depth and sample format for PCM layout bytes (coding, byte order,
/// arrangement), as `cutlass-pcm` names them. Empty is interleaved s16le.
/// A-law, mu-law, s64, planar samples and any other extradata are refused
/// rather than written under a label that would misread them.
fn pcm_entry(extradata: &[u8]) -> Result<(u16, SampleFormat), MuxError> {
    let (coding, big) = match extradata {
        [] => (1, false),
        [coding, order @ (0 | 1), 0] => (*coding, *order == 1),
        _ => {
            return Err(rejected(
                "PCM extradata other than three interleaved layout bytes",
            ));
        }
    };
    let pick =
        |little: SampleFormat, big_endian: SampleFormat| if big { big_endian } else { little };
    Ok(match coding {
        1 => (16, pick(SampleFormat::S16Le, SampleFormat::S16Be)),
        2 => (24, pick(SampleFormat::S24Le, SampleFormat::S24Be)),
        3 => (32, pick(SampleFormat::S32Le, SampleFormat::S32Be)),
        4 => (32, pick(SampleFormat::F32Le, SampleFormat::F32Be)),
        5 => (64, pick(SampleFormat::F64Le, SampleFormat::F64Be)),
        6 => (8, SampleFormat::S8),
        11 => (8, SampleFormat::U8),
        _ => {
            return Err(rejected(
                "PCM coding without an MP4 sample entry (A-law, mu-law or s64)",
            ));
        }
    })
}

/// The AudioSpecificConfig inside an MPEG-4 ES descriptor, or `config` as
/// it is when it is already bare. A bare config cannot start with the ES
/// descriptor tag 3: that would be an audio object type of 0.
fn bare_asc(config: &[u8]) -> Result<Vec<u8>, MuxError> {
    if config.first() != Some(&0x03) {
        return Ok(config.to_vec());
    }
    let mut at = 0;
    while at < config.len() {
        let tag = config[at];
        at += 1;
        let mut length = 0_usize;
        for _ in 0..4 {
            let byte = *config
                .get(at)
                .ok_or_else(|| rejected("short ES descriptor"))?;
            at += 1;
            length = (length << 7) | usize::from(byte & 0x7F);
            if byte & 0x80 == 0 {
                break;
            }
        }
        match tag {
            // ES descriptor: id and flags, then the descriptors inside.
            0x03 => {
                let flags = *config
                    .get(at + 2)
                    .ok_or_else(|| rejected("short ES descriptor"))?;
                at += 3;
                if flags & 0x80 != 0 {
                    at += 2;
                }
                if flags & 0x40 != 0 {
                    let url = *config
                        .get(at)
                        .ok_or_else(|| rejected("short ES descriptor"))?;
                    at += 1 + usize::from(url);
                }
                if flags & 0x20 != 0 {
                    at += 2;
                }
            }
            // Decoder config: thirteen fixed bytes, then the specific info.
            0x04 => at += 13,
            0x05 => {
                return config
                    .get(at..at + length)
                    .map(<[u8]>::to_vec)
                    .ok_or_else(|| rejected("short ES descriptor"));
            }
            _ => at += length,
        }
    }
    Err(rejected("the ES descriptor has no AudioSpecificConfig"))
}

impl<W: Write + Seek> Mux for Mp4Mux<W> {
    fn family(&self) -> ContainerId {
        ContainerId::Isobmff
    }

    fn add_stream(&mut self, stream: NewStream) -> Result<TrackId, MuxError> {
        self.building()?;
        let params = box_params(&stream, self.brand).map_err(|error| self.fail(error))?;
        let codec = params.codec;
        let boxed_id = legacy::Mux::add_stream(&mut self.inner, &params)
            .map_err(|error| self.fail(boxed(error)))?;
        let id = TrackId::new(boxed_id);
        self.tracks.push(Track {
            id,
            boxed: boxed_id,
            timescale: stream.timescale,
            codec,
            next: 0,
            color_from_frame: codec == CodecId::ProRes,
            trim: None,
        });
        Ok(id)
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<(), MuxError> {
        self.building()?;
        let index = self
            .track(packet.stream())
            .map_err(|error| self.fail(error))?;
        let track = self.tracks[index];
        if packet.timescale() != track.timescale {
            return Err(self.fail(MuxError::TimescaleMismatch {
                packet: packet.timescale().ticks_per_second(),
                stream: track.timescale.ticks_per_second(),
            }));
        }
        let timestamp = packet.timestamp();
        let (Some(pts), Some(duration)) =
            (timestamp.presentation_or_decode(), timestamp.duration())
        else {
            return Err(self.fail(MuxError::MissingTimestamp));
        };
        let Ok(span) = u32::try_from(duration) else {
            return Err(self.fail(MuxError::NegativeDuration));
        };
        if track.color_from_frame {
            self.tracks[index].color_from_frame = false;
            if let Some(color) = prores_color(packet.bytes()) {
                self.inner
                    .set_color(track.boxed, color)
                    .map_err(|error| self.fail(boxed(error)))?;
            }
        }
        legacy::Mux::write_packet(
            &mut self.inner,
            &legacy::Packet {
                track_id: track.boxed,
                payload: packet.bytes().to_vec(),
                dts: track.next,
                pts,
                duration: span,
                timescale: BoxScale(track.timescale.ticks_per_second()),
                keyframe: packet.keyframe(),
            },
        )
        .map_err(|error| self.fail(boxed(error)))?;
        self.tracks[index].next = track.next.saturating_add(duration);
        self.payload = self
            .payload
            .saturating_add(u64::try_from(packet.bytes().len()).unwrap_or(u64::MAX));
        Ok(())
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
        for track in &self.tracks {
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
        for track in &self.tracks {
            let Some(trim) = track.trim else {
                continue;
            };
            if let Err(error) = self.inner.set_gapless(
                track.boxed,
                trim.encoder_delay().unwrap_or(0),
                trim.trailing_padding().unwrap_or(0),
            ) {
                let error = self.fail(boxed(error));
                return Err(FailedMux::incomplete(self.payload, error));
            }
        }
        if let Err(error) = legacy::Mux::finish(&mut self.inner) {
            let error = self.fail(boxed(error));
            return Err(FailedMux::incomplete(self.payload, error));
        }
        self.status = MuxStatus::Finished;
        Ok(FinishedMux::complete(
            durations.to_vec(),
            IndexPlace::End,
            MuxMode::Progressive,
            false,
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
        let stored = match self.tracks[index].codec {
            CodecId::Aac => bare_asc(config).map_err(|error| self.fail(error))?,
            CodecId::Opus => crate::opus::dops(config)
                .ok_or_else(|| self.fail(rejected("Opus configuration is not an OpusHead")))?,
            _ => config.to_vec(),
        };
        self.inner
            .set_extradata(self.tracks[index].boxed, stored)
            .map_err(|error| self.fail(boxed(error)))
    }

    /// AC-3 and E-AC-3 keep the encoder delay and the trailing padding
    /// inside the frame durations. The edit written at finish skips the
    /// delay and ends before the padding. Opus already shortens its last
    /// packet and writes its pre-skip from the head, so this trim is not
    /// stored again for it.
    fn set_trim(&mut self, track: TrackId, trim: Trim) -> Result<(), MuxError> {
        self.building()?;
        let index = self.track(track.get()).map_err(|error| self.fail(error))?;
        if matches!(self.tracks[index].codec, CodecId::Ac3 | CodecId::Eac3) {
            self.tracks[index].trim = Some(trim);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use cutlass_codec::{
        CodecId as Id, FrameSize, Packet, StreamCodec, StreamParams, Timescale, Timestamp,
    };
    use cutlass_container::{Mux, NewStream, TrackDuration, Trim};

    use super::{Mp4Mux, bare_asc};
    use crate::mux::FileBrand;

    fn packet(stream: u32, data: Vec<u8>, pts: i64, duration: i64, scale: u32) -> Packet {
        let time = Timestamp::try_new(Some(pts), Some(pts), Some(duration)).unwrap();
        Packet::try_new(stream, data, time, Timescale::new(scale), true).unwrap()
    }

    #[test]
    fn a_movie_names_its_prores_profile_and_carries_pcm() {
        let mut mux = Mp4Mux::with_brand(Cursor::new(Vec::new()), FileBrand::Mov);
        let picture = mux
            .add_stream(NewStream {
                params: StreamParams::video(
                    StreamCodec::Known(Id::Prores),
                    FrameSize::try_new(64, 64).unwrap(),
                    b"apch".to_vec(),
                ),
                timescale: Timescale::new(30),
                language: None,
            })
            .unwrap();
        let sound = mux
            .add_stream(NewStream {
                params: StreamParams::try_audio(StreamCodec::Known(Id::Pcm), 48_000, 2, Vec::new())
                    .unwrap(),
                timescale: Timescale::new(48_000),
                language: None,
            })
            .unwrap();
        mux.write_packet(&packet(picture.get(), vec![1; 64], 0, 1, 30))
            .unwrap();
        mux.write_packet(&packet(sound.get(), vec![0; 1600 * 4], 0, 1600, 48_000))
            .unwrap();
        mux.write_packet(&packet(sound.get(), vec![7; 800 * 4], 1600, 800, 48_000))
            .unwrap();
        mux.finish(&[
            TrackDuration {
                track: picture,
                timescale: Timescale::new(30),
                ticks: 1,
            },
            TrackDuration {
                track: sound,
                timescale: Timescale::new(48_000),
                ticks: 2400,
            },
        ])
        .unwrap();
        let bytes = mux.into_writer().unwrap().into_inner();
        assert!(bytes.windows(4).any(|w| w == b"apch"));
        assert!(!bytes.windows(4).any(|w| w == b"apcn"));
        assert!(bytes.windows(4).any(|w| w == b"sowt"));

        // Read back, the sound is every byte that went in, at its time.
        let mut file = crate::IsoBmffFile::open(Cursor::new(bytes)).unwrap();
        let mut sound_bytes = Vec::new();
        let mut last_end = 0;
        while let Some(packet) = cutlass_container::Demux::read_packet(&mut file).unwrap() {
            if packet.stream() == sound.get() {
                let time = packet.timestamp();
                assert_eq!(time.presentation_or_decode(), Some(last_end));
                last_end += time.duration().unwrap();
                sound_bytes.extend_from_slice(packet.bytes());
            }
        }
        assert_eq!(last_end, 2400);
        assert_eq!(sound_bytes.len(), 2400 * 4);
        assert_eq!(sound_bytes[1600 * 4], 7);
    }

    /// Layout, bytes per sample, rate, channels, sample entry, read-back layout.
    type WriteCase<'a> = (&'a [u8], usize, u32, u16, &'a [u8; 4], [u8; 3]);

    /// Layout, bytes per sample, rate, channels, FFmpeg codec, afinfo format.
    type OracleCase<'a> = (&'a [u8], usize, u32, u16, &'a str, &'a str);

    /// A PCM track with these layout bytes, `frames` frames long, written to
    /// a movie: the file's bytes, or the muxer's refusal.
    pub(crate) fn pcm_movie(
        layout: &[u8],
        rate: u32,
        channels: u16,
        payload: &[u8],
        frames: i64,
    ) -> Result<Vec<u8>, String> {
        let mut mux = Mp4Mux::with_brand(Cursor::new(Vec::new()), FileBrand::Mov);
        let sound = mux
            .add_stream(NewStream {
                params: StreamParams::try_audio(
                    StreamCodec::Known(Id::Pcm),
                    rate,
                    channels,
                    layout.to_vec(),
                )
                .unwrap(),
                timescale: Timescale::new(rate),
                language: None,
            })
            .map_err(|error| format!("{error:?}"))?;
        mux.write_packet(&packet(sound.get(), payload.to_vec(), 0, frames, rate))
            .map_err(|error| format!("{error:?}"))?;
        mux.finish(&[TrackDuration {
            track: sound,
            timescale: Timescale::new(rate),
            ticks: frames,
        }])
        .map_err(|error| format!("{error:?}"))?;
        Ok(mux
            .into_writer()
            .map_err(|error| format!("{error:?}"))?
            .into_inner())
    }

    /// Every PCM layout the muxer takes is written under an entry that says
    /// it, and reads back with the same layout bytes, rate, channels and
    /// samples. Before, every track was declared 16-bit little-endian.
    #[test]
    fn every_pcm_layout_reads_back_as_written() {
        // (layout, bytes per sample, rate, channels, entry, read-back layout)
        let cases: [WriteCase<'_>; 15] = [
            (&[], 2, 48_000, 2, b"sowt", [1, 0, 0]),
            (&[1, 0, 0], 2, 48_000, 2, b"sowt", [1, 0, 0]),
            (&[1, 1, 0], 2, 44_100, 1, b"twos", [1, 1, 0]),
            (&[11, 0, 0], 1, 8_000, 1, b"raw ", [11, 0, 0]),
            (&[6, 0, 0], 1, 8_000, 2, b"twos", [6, 0, 0]),
            (&[2, 0, 0], 3, 48_000, 2, b"lpcm", [2, 0, 0]),
            (&[2, 1, 0], 3, 48_000, 2, b"lpcm", [2, 1, 0]),
            (&[3, 0, 0], 4, 48_000, 2, b"lpcm", [3, 0, 0]),
            (&[3, 1, 0], 4, 48_000, 1, b"lpcm", [3, 1, 0]),
            (&[4, 0, 0], 4, 48_000, 2, b"lpcm", [4, 0, 0]),
            (&[4, 1, 0], 4, 48_000, 2, b"lpcm", [4, 1, 0]),
            (&[5, 0, 0], 8, 48_000, 1, b"lpcm", [5, 0, 0]),
            (&[5, 1, 0], 8, 48_000, 1, b"lpcm", [5, 1, 0]),
            // 16-bit past the version 0 limits: 96 kHz, and six channels.
            (&[1, 0, 0], 2, 96_000, 2, b"lpcm", [1, 0, 0]),
            (&[1, 0, 0], 2, 48_000, 6, b"lpcm", [1, 0, 0]),
        ];
        for (layout, width, rate, channels, entry, read) in cases {
            let frames = 300_i64;
            let size = usize::try_from(frames).unwrap() * width * usize::from(channels);
            let payload: Vec<u8> = (0..size).map(|i| (i * 7 % 251) as u8).collect();
            let bytes = pcm_movie(layout, rate, channels, &payload, frames).unwrap();
            assert!(
                bytes.windows(4).any(|w| w == entry),
                "{layout:?}: no {entry:?}"
            );
            let mut file = crate::IsoBmffFile::open(Cursor::new(bytes)).unwrap();
            let stream = &cutlass_container::Demux::streams(&file)[0];
            assert_eq!(stream.params().extradata(), read, "{layout:?}");
            assert_eq!(stream.params().sample_rate(), Some(rate), "{layout:?}");
            assert_eq!(stream.params().channels(), Some(channels), "{layout:?}");
            let mut back = Vec::new();
            while let Some(packet) = cutlass_container::Demux::read_packet(&mut file).unwrap() {
                back.extend_from_slice(packet.bytes());
            }
            assert_eq!(back, payload, "{layout:?}");
        }
    }

    /// FFmpeg and, where present, Apple's `afinfo` read each layout the
    /// muxer writes as the format it is: FFmpeg names the codec and gives
    /// back every byte; `afinfo` states the channels, rate and format.
    #[test]
    fn ffmpeg_and_afinfo_read_each_pcm_layout_as_written() {
        let which = |name: &str| {
            let out = std::process::Command::new("which")
                .arg(name)
                .output()
                .ok()?;
            let path = String::from_utf8(out.stdout).ok()?;
            let path = std::path::PathBuf::from(path.trim());
            path.is_file().then_some(path)
        };
        let (Some(ffmpeg), Some(ffprobe)) = (which("ffmpeg"), which("ffprobe")) else {
            eprintln!("not available: ffmpeg or ffprobe");
            return;
        };
        let afinfo = which("afinfo");
        if afinfo.is_none() {
            eprintln!("not available: afinfo");
        }
        let dir = std::env::temp_dir().join(format!("isobmff-pcm-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // (layout, bytes per sample, rate, channels, FFmpeg codec, afinfo text)
        // afinfo's short names (Int16, Float32) are native little-endian.
        let cases: [OracleCase<'_>; 14] = [
            (&[1, 0, 0], 2, 48_000, 2, "pcm_s16le", "Int16"),
            (
                &[1, 1, 0],
                2,
                44_100,
                1,
                "pcm_s16be",
                "16-bit big-endian signed",
            ),
            (&[11, 0, 0], 1, 8_000, 1, "pcm_u8", "8-bit unsigned integer"),
            (&[6, 0, 0], 1, 8_000, 2, "pcm_s8", "8-bit signed integer"),
            (
                &[2, 0, 0],
                3,
                48_000,
                2,
                "pcm_s24le",
                "24-bit little-endian signed",
            ),
            (
                &[2, 1, 0],
                3,
                48_000,
                2,
                "pcm_s24be",
                "24-bit big-endian signed",
            ),
            (&[3, 0, 0], 4, 48_000, 2, "pcm_s32le", "Int32"),
            (
                &[3, 1, 0],
                4,
                48_000,
                1,
                "pcm_s32be",
                "32-bit big-endian signed",
            ),
            (&[4, 0, 0], 4, 48_000, 2, "pcm_f32le", "Float32"),
            (
                &[4, 1, 0],
                4,
                48_000,
                2,
                "pcm_f32be",
                "32-bit big-endian float",
            ),
            (&[5, 0, 0], 8, 48_000, 1, "pcm_f64le", "Float64"),
            (
                &[5, 1, 0],
                8,
                48_000,
                1,
                "pcm_f64be",
                "64-bit big-endian float",
            ),
            (&[1, 0, 0], 2, 96_000, 2, "pcm_s16le", "Int16"),
            (&[1, 0, 0], 2, 48_000, 6, "pcm_s16le", "Int16"),
        ];
        for (index, (layout, width, rate, channels, codec, apple)) in cases.into_iter().enumerate()
        {
            let frames = 480_i64;
            let size = usize::try_from(frames).unwrap() * width * usize::from(channels);
            let payload: Vec<u8> = (0..size).map(|i| (i * 13 % 241) as u8).collect();
            let path = dir.join(format!("pcm-{index}.mov"));
            std::fs::write(
                &path,
                pcm_movie(layout, rate, channels, &payload, frames).unwrap(),
            )
            .unwrap();
            let probe = std::process::Command::new(&ffprobe)
                .args([
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=codec_name,sample_rate,channels",
                ])
                .args(["-of", "csv=p=0"])
                .arg(&path)
                .output()
                .unwrap();
            let probed = String::from_utf8_lossy(&probe.stdout).trim().to_owned();
            assert_eq!(probed, format!("{codec},{rate},{channels}"), "{layout:?}");
            let copy = std::process::Command::new(&ffmpeg)
                .args(["-v", "error", "-nostdin", "-i"])
                .arg(&path)
                .args(["-map", "0:a", "-c", "copy", "-f", "data", "-"])
                .output()
                .unwrap();
            assert!(copy.status.success(), "{layout:?}");
            assert_eq!(copy.stdout, payload, "{layout:?}: FFmpeg's packets");
            if let Some(afinfo) = &afinfo {
                let out = std::process::Command::new(afinfo)
                    .arg(&path)
                    .output()
                    .unwrap();
                let text = String::from_utf8_lossy(&out.stdout);
                assert!(out.status.success(), "{layout:?}: {text}");
                let line = text
                    .lines()
                    .find(|line| line.contains("Data format"))
                    .unwrap_or("");
                assert!(
                    line.contains(&format!("{channels} ch")),
                    "{layout:?}: {line}"
                );
                assert!(line.contains(&format!("{rate} Hz")), "{layout:?}: {line}");
                assert!(line.contains(apple), "{layout:?} wants {apple}: {line}");
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// PCM movies FFmpeg writes read with the layout they hold: `in24` and
    /// `fl32` with an `enda` atom are little-endian, 8-bit `twos`/`sowt` is
    /// signed and `raw ` unsigned. Before, every one handed the decoder no
    /// layout and only float decoded.
    #[test]
    fn ffmpeg_pcm_movies_read_with_their_layout() {
        let Some(ffmpeg) = std::process::Command::new("which")
            .arg("ffmpeg")
            .output()
            .ok()
            .and_then(|out| String::from_utf8(out.stdout).ok())
            .map(|path| std::path::PathBuf::from(path.trim()))
            .filter(|path| path.is_file())
        else {
            eprintln!("not available: ffmpeg");
            return;
        };
        let dir = std::env::temp_dir().join(format!("isobmff-ffpcm-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cases: [(&str, [u8; 3]); 11] = [
            ("pcm_s16le", [1, 0, 0]),
            ("pcm_s16be", [1, 1, 0]),
            ("pcm_u8", [11, 0, 0]),
            ("pcm_s8", [6, 0, 0]),
            ("pcm_s24le", [2, 0, 0]),
            ("pcm_s24be", [2, 1, 0]),
            ("pcm_s32le", [3, 0, 0]),
            ("pcm_s32be", [3, 1, 0]),
            ("pcm_f32le", [4, 0, 0]),
            ("pcm_f32be", [4, 1, 0]),
            ("pcm_f64le", [5, 0, 0]),
        ];
        for (codec, layout) in cases {
            let path = dir.join(format!("{codec}.mov"));
            let made = std::process::Command::new(&ffmpeg)
                .args(["-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i"])
                .arg("sine=frequency=440:sample_rate=8000:duration=0.1")
                .args(["-c:a", codec])
                .arg(&path)
                .output()
                .unwrap();
            assert!(made.status.success(), "{codec}");
            let file = crate::IsoBmffFile::open(std::fs::File::open(&path).unwrap()).unwrap();
            let stream = &cutlass_container::Demux::streams(&file)[0];
            assert_eq!(stream.params().extradata(), layout, "{codec}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A-law, mu-law, s64, planar samples and foreign extradata have no
    /// entry here; they are refused rather than written under a wrong one.
    #[test]
    fn pcm_without_an_entry_is_refused() {
        for layout in [
            &[7, 0, 0][..],
            &[8, 0, 0],
            &[9, 0, 0],
            &[1, 0, 1],
            &[1, 2, 0],
            &[0; 18],
        ] {
            assert!(
                pcm_movie(layout, 8_000, 1, &[0; 64], 8).is_err(),
                "{layout:?} was written"
            );
        }
    }

    fn opus_track(mux: &mut Mp4Mux<Cursor<Vec<u8>>>) -> cutlass_container::TrackId {
        mux.add_stream(NewStream {
            params: StreamParams::try_audio(StreamCodec::Known(Id::Opus), 48_000, 2, Vec::new())
                .unwrap(),
            timescale: Timescale::new(48_000),
            language: None,
        })
        .unwrap()
    }

    /// An Opus track whose `OpusHead` arrives with the first packet is
    /// stored as `dOps`, gets an edit that starts after the pre-skip and
    /// ends with the samples, and reads back as the same `OpusHead`.
    #[test]
    fn an_opus_head_is_stored_as_dops_with_a_pre_skip_edit() {
        let mut head = b"OpusHead".to_vec();
        head.extend([1, 2, 0x38, 0x01, 0x44, 0xAC, 0, 0, 0, 0, 0]);
        let mut mux = Mp4Mux::new(Cursor::new(Vec::new()));
        let sound = opus_track(&mut mux);
        mux.set_config(sound, &head).unwrap();
        // 312 samples of pre-skip, then 2000 played: two whole packets and
        // one cut to 392.
        for (pts, duration) in [(0, 960), (960, 960), (1920, 392)] {
            let data = vec![0xFC, 0xFF, 0xFE];
            mux.write_packet(&packet(sound.get(), data, pts, duration, 48_000))
                .unwrap();
        }
        mux.finish(&[TrackDuration {
            track: sound,
            timescale: Timescale::new(48_000),
            ticks: 2312,
        }])
        .unwrap();
        let bytes = mux.into_writer().unwrap().into_inner();
        let at = bytes.windows(4).position(|w| w == b"dOps").unwrap();
        assert_eq!(
            &bytes[at + 4..at + 15],
            &[0, 2, 0x01, 0x38, 0, 0, 0xAC, 0x44, 0, 0, 0]
        );

        let file = crate::IsoBmffFile::open(Cursor::new(bytes)).unwrap();
        let stream = &cutlass_container::Demux::streams(&file)[0];
        assert_eq!(stream.params().extradata(), head.as_slice());
        assert_eq!(stream.trim().encoder_delay(), Some(312));
        let edit = stream.edits()[0];
        assert_eq!(edit.media_time(), Some(312));
        assert_eq!(edit.duration(), 2000);
    }

    /// AC-3 and E-AC-3 frame durations include the encoder delay and the
    /// trailing padding. The edit starts after the delay and ends before
    /// the padding. A later trim replaces an earlier one.
    #[test]
    fn ac3_and_eac3_trim_is_an_edit_around_the_played_samples() {
        for codec in [Id::Ac3, Id::Eac3] {
            let mut mux = Mp4Mux::with_brand(Cursor::new(Vec::new()), FileBrand::M4a);
            let sound = mux
                .add_stream(NewStream {
                    params: StreamParams::try_audio(
                        StreamCodec::Known(codec),
                        48_000,
                        2,
                        Vec::new(),
                    )
                    .unwrap(),
                    timescale: Timescale::new(48_000),
                    language: None,
                })
                .unwrap();
            mux.set_trim(sound, Trim::new(Some(1), Some(1))).unwrap();
            mux.set_trim(sound, Trim::new(Some(256), Some(128)))
                .unwrap();
            for (pts, duration) in [(0, 1536), (1536, 1536)] {
                mux.write_packet(&packet(
                    sound.get(),
                    vec![0x0B, 0x77],
                    pts,
                    duration,
                    48_000,
                ))
                .unwrap();
            }
            mux.finish(&[TrackDuration {
                track: sound,
                timescale: Timescale::new(48_000),
                ticks: 3072,
            }])
            .unwrap();
            let bytes = mux.into_writer().unwrap().into_inner();
            let file = crate::IsoBmffFile::open(Cursor::new(bytes)).unwrap();
            let stream = &cutlass_container::Demux::streams(&file)[0];
            let edit = stream.edits()[0];
            assert_eq!(edit.media_time(), Some(256), "{codec:?}");
            assert_eq!(edit.duration(), 3072 - 256 - 128, "{codec:?}");
            assert!(stream.trim().encoder_delay().is_none(), "{codec:?}");
        }
    }

    /// An MPEG-4 Visual track whose layer headers arrive with the first
    /// packet is an `mp4v` entry with object type 0x20 and those headers,
    /// its reordered samples get a `cslg`, and they read back in decode
    /// order with their times.
    #[test]
    fn an_mpeg4_visual_track_is_an_mp4v_entry_with_its_headers() {
        for brand in [FileBrand::Mp4, FileBrand::Mov] {
            let mut mux = Mp4Mux::with_brand(Cursor::new(Vec::new()), brand);
            let picture = mux
                .add_stream(NewStream {
                    params: StreamParams::video(
                        StreamCodec::Known(Id::Mpeg4),
                        FrameSize::try_new(64, 48).unwrap(),
                        Vec::new(),
                    ),
                    timescale: Timescale::new(30),
                    language: None,
                })
                .unwrap();
            let headers = [0, 0, 1, 0xB0, 0xF1, 0, 0, 1, 0x20, 0x08];
            mux.set_config(picture, &headers).unwrap();
            for pts in [0, 2, 1] {
                let vop = vec![0, 0, 1, 0xB6, u8::try_from(pts).unwrap()];
                mux.write_packet(&packet(picture.get(), vop, pts, 1, 30))
                    .unwrap();
            }
            mux.finish(&[TrackDuration {
                track: picture,
                timescale: Timescale::new(30),
                ticks: 3,
            }])
            .unwrap();
            let bytes = mux.into_writer().unwrap().into_inner();
            assert!(bytes.windows(4).any(|w| w == b"mp4v"));
            // The B sample shows before it decodes: a version 1 `ctts`, and
            // `cslg` with shift 1, offsets -1 to 1, and times 0 to 3.
            let ctts = bytes.windows(4).position(|w| w == b"ctts").unwrap();
            assert_eq!(bytes[ctts + 4], 1);
            let cslg = bytes.windows(4).position(|w| w == b"cslg").unwrap();
            assert_eq!(
                u32::from_be_bytes(bytes[cslg - 4..cslg].try_into().unwrap()),
                32
            );
            let field = |n: usize| {
                let at = cslg + 8 + n * 4;
                i32::from_be_bytes(bytes[at..at + 4].try_into().unwrap())
            };
            assert_eq!(bytes[cslg + 4], 0, "version 0");
            assert_eq!((0..5).map(field).collect::<Vec<_>>(), [1, -1, 1, 0, 3]);
            let mut file = crate::IsoBmffFile::open(Cursor::new(bytes)).unwrap();
            let params = cutlass_container::Demux::streams(&file)[0].params().clone();
            assert_eq!(params.codec(), &StreamCodec::Known(Id::Mpeg4));
            assert_eq!(params.extradata(), headers.as_slice());
            let mut times = Vec::new();
            let mut marks = Vec::new();
            while let Some(packet) = cutlass_container::Demux::read_packet(&mut file).unwrap() {
                let time = packet.timestamp();
                times.push((time.decode().unwrap(), time.presentation().unwrap()));
                marks.push(packet.bytes()[4]);
            }
            assert_eq!(times, [(0, 0), (1, 2), (2, 1)], "{brand:?}");
            assert_eq!(marks, [0, 2, 1], "{brand:?}");
        }
    }

    #[test]
    fn a_bad_opus_configuration_is_refused() {
        let mut mux = Mp4Mux::new(Cursor::new(Vec::new()));
        let sound = opus_track(&mut mux);
        assert!(mux.set_config(sound, b"OpusTags").is_err());
    }

    #[test]
    fn an_alac_cookie_set_late_comes_back_whole_with_its_layout() {
        // 6 channels, 24 bits, 96 kHz, then the 5.1 `chan` atom.
        let mut cookie = vec![
            0x00, 0x00, 0x10, 0x00, 0x00, 0x18, 0x28, 0x0a, 0x0e, 0x06, 0x00, 0xff, 0x00, 0x00,
            0x47, 0x0d, 0x00, 0x10, 0xc1, 0x2d, 0x00, 0x01, 0x77, 0x00,
        ];
        cookie.extend([0, 0, 0, 24, b'c', b'h', b'a', b'n', 0, 0, 0, 0]);
        cookie.extend([0x00, 0x7c, 0x00, 0x06, 0, 0, 0, 0, 0, 0, 0, 0]);
        let mut mux = Mp4Mux::new(Cursor::new(Vec::new()));
        let sound = mux
            .add_stream(NewStream {
                params: StreamParams::try_audio(
                    StreamCodec::Known(Id::Alac),
                    96_000,
                    6,
                    Vec::new(),
                )
                .unwrap(),
                timescale: Timescale::new(96_000),
                language: None,
            })
            .unwrap();
        mux.set_config(sound, &cookie).unwrap();
        mux.write_packet(&packet(sound.get(), vec![1; 40], 0, 4096, 96_000))
            .unwrap();
        mux.finish(&[TrackDuration {
            track: sound,
            timescale: Timescale::new(96_000),
            ticks: 4096,
        }])
        .unwrap();
        let bytes = mux.into_writer().unwrap().into_inner();
        assert!(bytes.windows(4).any(|w| w == b"chan"));
        let file = crate::IsoBmffFile::open(Cursor::new(bytes)).unwrap();
        let params = cutlass_container::Demux::streams(&file)[0].params().clone();
        assert_eq!(params.extradata(), cookie.as_slice());
        assert_eq!(params.channels(), Some(6));
        assert_eq!(params.sample_rate(), Some(96_000));
    }

    #[test]
    fn an_alac_track_without_a_cookie_fails_at_finish() {
        let mut mux = Mp4Mux::new(Cursor::new(Vec::new()));
        let sound = mux
            .add_stream(NewStream {
                params: StreamParams::try_audio(
                    StreamCodec::Known(Id::Alac),
                    44_100,
                    2,
                    Vec::new(),
                )
                .unwrap(),
                timescale: Timescale::new(44_100),
                language: None,
            })
            .unwrap();
        mux.write_packet(&packet(sound.get(), vec![1; 40], 0, 4096, 44_100))
            .unwrap();
        assert!(
            mux.finish(&[TrackDuration {
                track: sound,
                timescale: Timescale::new(44_100),
                ticks: 4096,
            }])
            .is_err()
        );
    }

    #[test]
    fn every_prores_profile_survives_a_movie_round_trip() {
        for code in [b"apco", b"apcs", b"apcn", b"apch", b"ap4h", b"ap4x"] {
            let mut mux = Mp4Mux::with_brand(Cursor::new(Vec::new()), FileBrand::Mov);
            let picture = mux
                .add_stream(NewStream {
                    params: StreamParams::video(
                        StreamCodec::Known(Id::Prores),
                        FrameSize::try_new(32, 16).unwrap(),
                        code.to_vec(),
                    ),
                    timescale: Timescale::new(25),
                    language: None,
                })
                .unwrap();
            mux.write_packet(&packet(picture.get(), vec![3; 40], 0, 1, 25))
                .unwrap();
            mux.finish(&[TrackDuration {
                track: picture,
                timescale: Timescale::new(25),
                ticks: 1,
            }])
            .unwrap();
            let bytes = mux.into_writer().unwrap().into_inner();
            assert!(bytes.windows(4).any(|w| w == code.as_slice()));
            let file = crate::IsoBmffFile::open(Cursor::new(bytes)).unwrap();
            let streams = cutlass_container::Demux::streams(&file);
            assert_eq!(streams.len(), 1);
            assert_eq!(streams[0].params().codec(), &StreamCodec::Known(Id::Prores));
            assert_eq!(streams[0].params().extradata(), code.as_slice());
        }
    }

    fn mjpeg_stream(extradata: &[u8]) -> NewStream {
        NewStream {
            params: StreamParams::video(
                StreamCodec::Known(Id::Mjpeg),
                FrameSize::try_new(48, 32).unwrap(),
                extradata.to_vec(),
            ),
            timescale: Timescale::new(25),
            language: None,
        }
    }

    #[test]
    fn a_motion_jpeg_movie_is_a_jpeg_entry_with_every_frame() {
        let mut mux = Mp4Mux::with_brand(Cursor::new(Vec::new()), FileBrand::Mov);
        let picture = mux.add_stream(mjpeg_stream(b"jpeg")).unwrap();
        let frames: Vec<Vec<u8>> = (0u8..3)
            .map(|index| [&[0xFF, 0xD8][..], &[index; 20], &[0xFF, 0xD9]].concat())
            .collect();
        for (pts, frame) in frames.iter().enumerate() {
            let pts = i64::try_from(pts).unwrap();
            mux.write_packet(&packet(picture.get(), frame.clone(), pts, 1, 25))
                .unwrap();
        }
        mux.finish(&[TrackDuration {
            track: picture,
            timescale: Timescale::new(25),
            ticks: 3,
        }])
        .unwrap();
        let bytes = mux.into_writer().unwrap().into_inner();
        let mut file = crate::IsoBmffFile::open(Cursor::new(bytes)).unwrap();
        let params = cutlass_container::Demux::streams(&file)[0].params().clone();
        assert_eq!(params.codec(), &StreamCodec::Known(Id::Mjpeg));
        assert_eq!(params.extradata(), b"jpeg");
        assert_eq!(params.size(), Some(FrameSize::try_new(48, 32).unwrap()));
        let mut read = Vec::new();
        while let Some(packet) = cutlass_container::Demux::read_packet(&mut file).unwrap() {
            assert!(packet.keyframe());
            read.push((packet.timestamp().presentation(), packet.into_bytes()));
        }
        let want: Vec<_> = frames
            .into_iter()
            .enumerate()
            .map(|(pts, frame)| (Some(i64::try_from(pts).unwrap()), frame))
            .collect();
        assert_eq!(read, want);
    }

    #[test]
    fn motion_jpeg_is_refused_outside_a_movie_or_with_foreign_extradata() {
        for brand in [FileBrand::Mp4, FileBrand::M4v] {
            let mut mux = Mp4Mux::with_brand(Cursor::new(Vec::new()), brand);
            assert!(mux.add_stream(mjpeg_stream(&[])).is_err());
        }
        let mut mux = Mp4Mux::with_brand(Cursor::new(Vec::new()), FileBrand::Mov);
        assert!(mux.add_stream(mjpeg_stream(b"mjpb")).is_err());
        let mut mux = Mp4Mux::with_brand(Cursor::new(Vec::new()), FileBrand::Mov);
        assert!(mux.add_stream(mjpeg_stream(&[])).is_ok());
    }

    #[test]
    fn a_prores_frame_header_names_the_movie_color() {
        let movie = |codes: [u8; 3]| {
            let mut frame = vec![0u8; 40];
            frame[4..8].copy_from_slice(b"icpf");
            frame[22..25].copy_from_slice(&codes);
            let mut mux = Mp4Mux::with_brand(Cursor::new(Vec::new()), FileBrand::Mov);
            let picture = mux
                .add_stream(NewStream {
                    params: StreamParams::video(
                        StreamCodec::Known(Id::Prores),
                        FrameSize::try_new(32, 16).unwrap(),
                        b"apch".to_vec(),
                    ),
                    timescale: Timescale::new(25),
                    language: None,
                })
                .unwrap();
            mux.write_packet(&packet(picture.get(), frame, 0, 1, 25))
                .unwrap();
            mux.finish(&[TrackDuration {
                track: picture,
                timescale: Timescale::new(25),
                ticks: 1,
            }])
            .unwrap();
            mux.into_writer().unwrap().into_inner()
        };
        // BT.2020 primaries, HLG, BT.2020 non-constant luminance.
        let tagged = movie([9, 18, 9]);
        let colr = [
            b'c', b'o', b'l', b'r', b'n', b'c', b'l', b'x', 0, 9, 0, 18, 0, 9, 0,
        ];
        assert!(tagged.windows(colr.len()).any(|w| w == colr));
        // Unknown and unspecified codes write no box.
        let untagged = movie([0, 2, 0]);
        assert!(!untagged.windows(4).any(|w| w == b"colr"));
    }

    /// A VP8 track is a `vp08` sample entry whose `vpcC` box holds the
    /// configuration the first packet carries; it reads back as VP8 with
    /// that configuration and every sample.
    #[test]
    fn a_vp8_track_is_vp08_with_its_vpcc() {
        let vpcc = vec![1, 0, 0, 0, 0, 0, 0x82, 1, 1, 1, 0, 0];
        let mut mux = Mp4Mux::new(Cursor::new(Vec::new()));
        let picture = mux
            .add_stream(NewStream {
                params: StreamParams::video(
                    StreamCodec::Known(Id::Vp8),
                    FrameSize::try_new(64, 32).unwrap(),
                    Vec::new(),
                ),
                timescale: Timescale::new(30),
                language: None,
            })
            .unwrap();
        mux.set_config(picture, &vpcc).unwrap();
        for index in 0..3 {
            mux.write_packet(&packet(picture.get(), vec![index as u8; 40], index, 1, 30))
                .unwrap();
        }
        mux.finish(&[TrackDuration {
            track: picture,
            timescale: Timescale::new(30),
            ticks: 3,
        }])
        .unwrap();
        let bytes = mux.into_writer().unwrap().into_inner();
        assert!(bytes.windows(4).any(|w| w == b"vp08"));
        let at = bytes.windows(4).position(|w| w == b"vpcC").unwrap();
        assert_eq!(&bytes[at + 4..at + 4 + vpcc.len()], vpcc.as_slice());

        let mut file = crate::IsoBmffFile::open(Cursor::new(bytes)).unwrap();
        let stream = &cutlass_container::Demux::streams(&file)[0];
        assert_eq!(stream.params().codec(), &StreamCodec::Known(Id::Vp8));
        assert_eq!(stream.params().extradata(), vpcc.as_slice());
        let mut samples = 0;
        while let Some(packet) = cutlass_container::Demux::read_packet(&mut file).unwrap() {
            assert_eq!(packet.bytes(), vec![samples as u8; 40].as_slice());
            samples += 1;
        }
        assert_eq!(samples, 3);
    }

    /// An AV1 track is an `av01` sample entry whose `av1C` box holds the
    /// record the first packet carries, in a file that lists the `av01`
    /// brand. Each sample is its temporal unit without the delimiter and
    /// padding, every OBU with a size, and reads back as that.
    #[test]
    fn an_av1_track_is_av01_with_its_av1c_brand_and_bare_samples() {
        // The record of a 64x32 8-bit 4:2:0 stream: marker and version,
        // profile and level, the chroma bits, no delay, a sequence header.
        let av1c = vec![0x81, 0x00, 0x0c, 0x00, 0x0a, 0x03, 1, 2, 3];
        let obu = |obu_type: u8, payload: &[u8]| {
            let mut out = vec![(obu_type << 3) | 2, payload.len() as u8];
            out.extend_from_slice(payload);
            out
        };
        // A delimiter, then a frame; the second unit also pads.
        let unit = |index: u8| {
            let mut out = obu(2, &[]);
            if index == 1 {
                out.extend(obu(15, &[0, 0]));
            }
            out.extend(obu(6, &[index; 30]));
            out
        };
        let mut mux = Mp4Mux::new(Cursor::new(Vec::new()));
        let picture = mux
            .add_stream(NewStream {
                params: StreamParams::video(
                    StreamCodec::Known(Id::Av1),
                    FrameSize::try_new(64, 32).unwrap(),
                    Vec::new(),
                ),
                timescale: Timescale::new(30),
                language: None,
            })
            .unwrap();
        mux.set_config(picture, &av1c).unwrap();
        for index in 0..3u8 {
            mux.write_packet(&packet(picture.get(), unit(index), i64::from(index), 1, 30))
                .unwrap();
        }
        mux.finish(&[TrackDuration {
            track: picture,
            timescale: Timescale::new(30),
            ticks: 3,
        }])
        .unwrap();
        let bytes = mux.into_writer().unwrap().into_inner();
        assert!(bytes.windows(4).any(|w| w == b"av01"));
        let at = bytes.windows(4).position(|w| w == b"av1C").unwrap();
        assert_eq!(&bytes[at + 4..at + 4 + av1c.len()], av1c.as_slice());
        // The file type box lists the brand.
        assert_eq!(&bytes[4..8], b"ftyp");
        let ftyp = u32::from_be_bytes(bytes[..4].try_into().unwrap()) as usize;
        assert!(bytes[16..ftyp].chunks(4).any(|brand| brand == b"av01"));

        let mut file = crate::IsoBmffFile::open(Cursor::new(bytes)).unwrap();
        let stream = &cutlass_container::Demux::streams(&file)[0];
        assert_eq!(stream.params().codec(), &StreamCodec::Known(Id::Av1));
        assert_eq!(stream.params().extradata(), av1c.as_slice());
        let mut samples = 0u8;
        while let Some(packet) = cutlass_container::Demux::read_packet(&mut file).unwrap() {
            assert_eq!(packet.bytes(), obu(6, &[samples; 30]).as_slice());
            samples += 1;
        }
        assert_eq!(samples, 3);

        // A packet that is not whole OBUs is refused, and a movie without
        // an AV1 track does not list the brand.
        let mut mux = Mp4Mux::new(Cursor::new(Vec::new()));
        let picture = mux
            .add_stream(NewStream {
                params: StreamParams::video(
                    StreamCodec::Known(Id::Av1),
                    FrameSize::try_new(64, 32).unwrap(),
                    Vec::new(),
                ),
                timescale: Timescale::new(30),
                language: None,
            })
            .unwrap();
        assert!(
            mux.write_packet(&packet(picture.get(), vec![0x32, 9, 1], 0, 1, 30))
                .is_err()
        );
        let mut mux = Mp4Mux::new(Cursor::new(Vec::new()));
        let picture = mux
            .add_stream(NewStream {
                params: StreamParams::video(
                    StreamCodec::Known(Id::Vp8),
                    FrameSize::try_new(64, 32).unwrap(),
                    Vec::new(),
                ),
                timescale: Timescale::new(30),
                language: None,
            })
            .unwrap();
        mux.write_packet(&packet(picture.get(), vec![0; 8], 0, 1, 30))
            .unwrap();
        mux.finish(&[TrackDuration {
            track: picture,
            timescale: Timescale::new(30),
            ticks: 1,
        }])
        .unwrap();
        let bytes = mux.into_writer().unwrap().into_inner();
        let ftyp = u32::from_be_bytes(bytes[..4].try_into().unwrap()) as usize;
        assert!(!bytes[16..ftyp].chunks(4).any(|brand| brand == b"av01"));
    }

    /// An MPEG-2 track is an `mp4v` entry whose ES descriptor names Main
    /// profile and carries the sequence headers. It reads back as MPEG-2
    /// with those headers, every sample in decode order, and the reordered
    /// presentation times of its B pictures.
    #[test]
    fn an_mpeg2_track_is_mp4v_with_its_object_type() {
        let headers = vec![
            0, 0, 1, 0xB3, 0x2D, 0x02, 0x40, 0x13, 0xFF, 0xFF, 0xE0, 0x18, 0, 0, 1, 0xB5, 0x14,
            0x8A, 0, 1, 0, 0,
        ];
        let movie = |config: Option<&[u8]>| {
            let mut mux = Mp4Mux::new(Cursor::new(Vec::new()));
            let picture = mux
                .add_stream(NewStream {
                    params: StreamParams::video(
                        StreamCodec::Known(Id::Mpeg2),
                        FrameSize::try_new(720, 576).unwrap(),
                        Vec::new(),
                    ),
                    timescale: Timescale::new(25),
                    language: None,
                })
                .unwrap();
            if let Some(config) = config {
                mux.set_config(picture, config).unwrap();
            }
            for (index, pts) in [0, 3, 1, 2].into_iter().enumerate() {
                let data = vec![index as u8; 40];
                mux.write_packet(&packet(picture.get(), data, pts, 1, 25))
                    .unwrap();
            }
            mux.finish(&[TrackDuration {
                track: picture,
                timescale: Timescale::new(25),
                ticks: 4,
            }])
            .unwrap();
            mux.into_writer().unwrap().into_inner()
        };

        let bytes = movie(Some(&headers));
        assert!(bytes.windows(4).any(|w| w == b"mp4v"));
        assert!(!bytes.windows(4).any(|w| w == b"mp2v"));
        let at = bytes.windows(4).position(|w| w == b"esds").unwrap();
        let oti = bytes[at..].windows(2).position(|w| w == [0x61, 0x11]);
        assert!(oti.is_some());
        let mut file = crate::IsoBmffFile::open(Cursor::new(bytes)).unwrap();
        let stream = &cutlass_container::Demux::streams(&file)[0];
        assert_eq!(stream.params().codec(), &StreamCodec::Known(Id::Mpeg2));
        assert_eq!(stream.params().extradata(), headers.as_slice());
        let mut times = Vec::new();
        while let Some(packet) = cutlass_container::Demux::read_packet(&mut file).unwrap() {
            assert_eq!(packet.bytes(), vec![times.len() as u8; 40].as_slice());
            let time = packet.timestamp();
            times.push((time.decode().unwrap(), time.presentation().unwrap()));
        }
        assert_eq!(times, [(0, 0), (1, 3), (2, 1), (3, 2)]);

        // Without the headers the descriptor still names MPEG-2.
        let bare = movie(None);
        let file = crate::IsoBmffFile::open(Cursor::new(bare)).unwrap();
        let stream = &cutlass_container::Demux::streams(&file)[0];
        assert_eq!(stream.params().codec(), &StreamCodec::Known(Id::Mpeg2));
        assert!(stream.params().extradata().is_empty());
    }

    #[test]
    fn an_es_descriptor_gives_up_its_audio_specific_config() {
        // AAC LC, 48 kHz, stereo.
        let asc = [0x11, 0x90];
        assert_eq!(bare_asc(&asc).unwrap(), asc);
        let mut decoder = vec![0x40, 0x15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        decoder.extend([0x05, 0x02, 0x11, 0x90]);
        let mut es = vec![0x03, 0x80, 0x80, 0x80, 0x19, 0x00, 0x00, 0x00];
        es.extend([0x04, u8::try_from(decoder.len()).unwrap()]);
        es.extend(&decoder);
        es.extend([0x06, 0x01, 0x02]);
        assert_eq!(bare_asc(&es).unwrap(), asc);
        assert!(bare_asc(&[0x03, 0x05]).is_err());
        assert!(bare_asc(&[0x03, 0x02, 0x00, 0x00, 0x00]).is_err());
    }

    /// The payload of the first `kind` box among the boxes in `bytes`.
    fn child<'a>(bytes: &'a [u8], kind: &[u8; 4]) -> Option<&'a [u8]> {
        let mut at = 0;
        while at + 8 <= bytes.len() {
            let mut size = u32::from_be_bytes(bytes[at..at + 4].try_into().ok()?) as usize;
            let mut header = 8;
            if size == 1 {
                let large = bytes.get(at + 8..at + 16)?;
                size = usize::try_from(u64::from_be_bytes(large.try_into().ok()?)).ok()?;
                header = 16;
            }
            if size < header || at + size > bytes.len() {
                return None;
            }
            if &bytes[at + 4..at + 8] == kind {
                return Some(&bytes[at + header..at + size]);
            }
            at += size;
        }
        None
    }

    /// A short JPEG-looking picture: the start-of-image marker, then bytes.
    fn fake_jpeg() -> Vec<u8> {
        let mut jpeg = vec![0xFF, 0xD8, 0xFF, 0xE0];
        jpeg.extend((0..300).map(|i| (i * 7 % 251) as u8));
        jpeg.extend([0xFF, 0xD9]);
        jpeg
    }

    /// A sound-only file of `brand`, with `cover` set when given.
    fn covered(brand: FileBrand, cover: Option<Vec<u8>>) -> Vec<u8> {
        let mut mux = Mp4Mux::with_brand(Cursor::new(Vec::new()), brand);
        let sound = mux
            .add_stream(NewStream {
                params: StreamParams::try_audio(StreamCodec::Known(Id::Pcm), 48_000, 2, Vec::new())
                    .unwrap(),
                timescale: Timescale::new(48_000),
                language: None,
            })
            .unwrap();
        if let Some(cover) = cover {
            mux.set_cover_art(cover).unwrap();
        }
        let payload: Vec<u8> = (0..4800 * 4).map(|i| (i % 253) as u8).collect();
        mux.write_packet(&packet(sound.get(), payload, 0, 4800, 48_000))
            .unwrap();
        mux.finish(&[TrackDuration {
            track: sound,
            timescale: Timescale::new(48_000),
            ticks: 4800,
        }])
        .unwrap();
        mux.into_writer().unwrap().into_inner()
    }

    /// Every packet of a file, as the crate's demuxer reads it.
    fn packets(bytes: Vec<u8>) -> Vec<(u32, Option<i64>, Vec<u8>)> {
        let mut file = crate::IsoBmffFile::open(Cursor::new(bytes)).unwrap();
        let mut out = Vec::new();
        while let Some(packet) = cutlass_container::Demux::read_packet(&mut file).unwrap() {
            let time = packet.timestamp().presentation_or_decode();
            out.push((packet.stream(), time, packet.bytes().to_vec()));
        }
        out
    }

    /// The cover is `moov/udta/meta/ilst/covr/data` as FFmpeg's movenc.c
    /// writes it: `meta` a full box, an `hdlr` of 33 bytes naming `mdir`
    /// and `appl`, and a `data` atom of type 13 (JPEG), locale 0, then the
    /// picture. The demuxer reads the same packets as without it.
    #[test]
    fn a_cover_is_itunes_metadata_and_the_movie_reads_the_same() {
        let jpeg = fake_jpeg();
        for brand in [FileBrand::Mp4, FileBrand::Mov, FileBrand::M4a] {
            let plain = covered(brand, None);
            let bytes = covered(brand, Some(jpeg.clone()));
            assert!(child(child(&plain, b"moov").unwrap(), b"udta").is_none());

            let moov = child(&bytes, b"moov").unwrap();
            let udta = child(moov, b"udta").unwrap();
            let meta = child(udta, b"meta").unwrap();
            assert_eq!(&meta[..4], &[0, 0, 0, 0], "{brand:?}: version and flags");
            let mut hdlr = 33u32.to_be_bytes().to_vec();
            hdlr.extend_from_slice(b"hdlr");
            hdlr.extend_from_slice(&[0; 8]);
            hdlr.extend_from_slice(b"mdirappl");
            hdlr.extend_from_slice(&[0; 9]);
            assert_eq!(&meta[4..37], hdlr.as_slice(), "{brand:?}");
            let ilst = child(&meta[4..], b"ilst").unwrap();
            let covr = child(ilst, b"covr").unwrap();
            assert_eq!(covr.len(), 8 + 8 + jpeg.len());
            let data = child(covr, b"data").unwrap();
            assert_eq!(&data[..8], &[0, 0, 0, 13, 0, 0, 0, 0], "{brand:?}");
            assert_eq!(&data[8..], jpeg.as_slice(), "{brand:?}");
            // `udta` is the last child of `moov`, after the tracks.
            assert!(moov.ends_with(udta));

            assert_eq!(packets(bytes), packets(plain), "{brand:?}");
        }
    }

    /// With `moov` written first, and in a fragmented file, the cover is
    /// in `moov` too and the files open with the same packets.
    #[test]
    fn a_cover_goes_in_moov_first_and_fragmented_files() {
        use crate::legacy;
        use crate::mux::{IsoBmffMux, MuxOptions};

        let jpeg = fake_jpeg();
        for (moov_at_end, fragmented) in [(false, false), (true, true)] {
            let write = |cover: bool| {
                let mut mux = IsoBmffMux::new(
                    Cursor::new(Vec::new()),
                    MuxOptions {
                        moov_at_end,
                        fragmented,
                        samples_per_fragment: 1,
                        ..MuxOptions::default()
                    },
                );
                let params = legacy::StreamParams::audio(
                    legacy::CodecId::Pcm,
                    legacy::Timescale(8_000),
                    8_000,
                    1,
                    16,
                    legacy::SampleFormat::S16Le,
                );
                let track = legacy::Mux::add_stream(&mut mux, &params).unwrap();
                if cover {
                    mux.set_cover_art(jpeg.clone()).unwrap();
                }
                for index in 0..3 {
                    legacy::Mux::write_packet(
                        &mut mux,
                        &legacy::Packet {
                            track_id: track,
                            payload: vec![index; 320],
                            dts: i64::from(index) * 160,
                            pts: i64::from(index) * 160,
                            duration: 160,
                            timescale: legacy::Timescale(8_000),
                            keyframe: true,
                        },
                    )
                    .unwrap();
                }
                // A fragmented file has its `moov` out already.
                if fragmented {
                    assert!(mux.set_cover_art(jpeg.clone()).is_err());
                }
                mux.finish_into().unwrap().into_inner()
            };
            let bytes = write(true);
            let data = child(&bytes, b"moov")
                .and_then(|moov| child(moov, b"udta"))
                .and_then(|udta| child(udta, b"meta"))
                .and_then(|meta| child(&meta[4..], b"ilst"))
                .and_then(|ilst| child(ilst, b"covr"))
                .and_then(|covr| child(covr, b"data"))
                .unwrap();
            assert_eq!(&data[8..], jpeg.as_slice());
            assert_eq!(packets(bytes), packets(write(false)), "{fragmented}");
        }
    }

    /// Bytes that are not a JPEG, a picture past the limit, and a cover
    /// after finish are refused; a refused cover leaves the file plain and
    /// the muxer building.
    #[test]
    fn a_cover_that_is_not_a_jpeg_is_refused_and_the_file_stays_plain() {
        let mut mux = Mp4Mux::new(Cursor::new(Vec::new()));
        assert!(mux.set_cover_art(Vec::new()).is_err());
        assert!(mux.set_cover_art(b"\x89PNG\r\n\x1a\n".to_vec()).is_err());
        let mut large = vec![0xFF, 0xD8, 0xFF];
        large.resize(
            usize::try_from(crate::limits::MAX_COVER_BYTES).unwrap() + 1,
            0,
        );
        assert!(mux.set_cover_art(large).is_err());
        assert_eq!(mux.status(), cutlass_container::MuxStatus::Building);
        let sound = mux
            .add_stream(NewStream {
                params: StreamParams::try_audio(StreamCodec::Known(Id::Pcm), 8_000, 1, Vec::new())
                    .unwrap(),
                timescale: Timescale::new(8_000),
                language: None,
            })
            .unwrap();
        mux.write_packet(&packet(sound.get(), vec![0; 320], 0, 160, 8_000))
            .unwrap();
        mux.finish(&[TrackDuration {
            track: sound,
            timescale: Timescale::new(8_000),
            ticks: 160,
        }])
        .unwrap();
        assert!(mux.set_cover_art(fake_jpeg()).is_err());
        let bytes = mux.into_writer().unwrap().into_inner();
        assert!(child(child(&bytes, b"moov").unwrap(), b"udta").is_none());
    }

    /// FFmpeg reads the cover of an MP4, a movie and an M4A as an attached
    /// picture: one more stream, Motion JPEG with `attached_pic` set, whose
    /// packet is the picture. The sound stream and its packets are the same
    /// as in the file without a cover.
    #[test]
    fn ffmpeg_reads_the_cover_as_an_attached_picture() {
        let which = |name: &str| {
            let out = std::process::Command::new("which")
                .arg(name)
                .output()
                .ok()?;
            let path = String::from_utf8(out.stdout).ok()?;
            let path = std::path::PathBuf::from(path.trim());
            path.is_file().then_some(path)
        };
        let (Some(ffmpeg), Some(ffprobe)) = (which("ffmpeg"), which("ffprobe")) else {
            eprintln!("not available: ffmpeg or ffprobe");
            return;
        };
        let dir = std::env::temp_dir().join(format!("isobmff-cover-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let picture = dir.join("cover.jpg");
        let made = std::process::Command::new(&ffmpeg)
            .args(["-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i"])
            .arg("color=c=red:s=16x16")
            .args(["-frames:v", "1", "-c:v", "mjpeg", "-f", "image2"])
            .arg(&picture)
            .output()
            .unwrap();
        assert!(made.status.success());
        let jpeg = std::fs::read(&picture).unwrap();
        let streams = |path: &std::path::Path| {
            let out = std::process::Command::new(&ffprobe)
                .args(["-v", "error", "-show_entries"])
                .arg("stream=index,codec_type,codec_name,sample_rate,channels,width,height:stream_disposition=attached_pic")
                .args(["-of", "csv=p=0"])
                .arg(path)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "{}",
                String::from_utf8_lossy(&out.stderr)
            );
            assert!(
                out.stderr.is_empty(),
                "{}",
                String::from_utf8_lossy(&out.stderr)
            );
            String::from_utf8(out.stdout).unwrap()
        };
        let stream_bytes = |path: &std::path::Path, map: &str| {
            let out = std::process::Command::new(&ffmpeg)
                .args(["-v", "error", "-nostdin", "-i"])
                .arg(path)
                .args(["-map", map, "-c", "copy", "-f", "data", "-"])
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "{}",
                String::from_utf8_lossy(&out.stderr)
            );
            out.stdout
        };
        for (brand, name) in [
            (FileBrand::Mp4, "cover.mp4"),
            (FileBrand::Mov, "cover.mov"),
            (FileBrand::M4a, "cover.m4a"),
        ] {
            let plain = dir.join(format!("plain-{name}"));
            let path = dir.join(name);
            std::fs::write(&plain, covered(brand, None)).unwrap();
            std::fs::write(&path, covered(brand, Some(jpeg.clone()))).unwrap();
            let before = streams(&plain);
            let after = streams(&path);
            assert_eq!(before.lines().count(), 1, "{name}: {before}");
            let lines: Vec<&str> = after.lines().collect();
            assert_eq!(lines.len(), 2, "{name}: {after}");
            assert_eq!(lines[0], before.trim(), "{name}");
            assert_eq!(lines[1], "1,mjpeg,video,16,16,1", "{name}");
            assert_eq!(stream_bytes(&path, "0:0"), stream_bytes(&plain, "0:0"));
            assert_eq!(stream_bytes(&path, "0:1"), jpeg, "{name}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
