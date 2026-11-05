//! An MP4 written through the workspace mux trait, for export.
//!
//! Packets go straight to the file after `mdat` and the index follows at
//! finish, so a long export does not hold its packets in memory. Each
//! track's codec configuration may arrive with its first packet, as encoders
//! hand it over; it is written in the index at the end. An AAC
//! configuration wrapped in an MPEG-4 ES descriptor, as Apple's encoder
//! gives it, is unwrapped to the bare AudioSpecificConfig MP4 stores.

use std::io::{Seek, Write};

use cutlass_codec::{CodecId as Id, ContainerId, Packet, StreamCodec, Timescale};
use cutlass_container::{
    FailedMux, FinishedMux, IndexPlace, Mux, MuxError, MuxMode, MuxStatus, NewStream,
    TrackDuration, TrackId,
};

use crate::legacy::{self, CodecId, Error, SampleFormat, StreamParams, Timescale as BoxScale};
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
}

/// An MP4 file written as packets arrive, with its index at the end.
pub struct Mp4Mux<W: Write + Seek> {
    inner: IsoBmffMux<W>,
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
            tracks: Vec::new(),
            status: MuxStatus::Building,
            payload: 0,
            failure: None,
        }
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

/// ProRes profiles a picture track may name in its extradata: Proxy, LT,
/// 422 and HQ, by their sample entry.
const PRORES_ENTRIES: [[u8; 4]; 4] = [*b"apco", *b"apcs", *b"apcn", *b"apch"];

/// The box writer's parameters for `stream`: H.264, HEVC or ProRes
/// pictures, or AAC or 16-bit PCM sound. A ProRes track whose extradata is
/// a profile's four-character code is written with that sample entry.
fn box_params(stream: &NewStream) -> Result<StreamParams, MuxError> {
    let StreamCodec::Known(codec) = stream.params.codec() else {
        return Err(rejected(
            "mp4 export writes H.264, HEVC, ProRes, AAC, ALAC, and PCM",
        ));
    };
    let scale = BoxScale(stream.timescale.ticks_per_second());
    let mut params = match codec {
        Id::H264 | Id::Hevc | Id::Prores => {
            let size = stream
                .params
                .size()
                .ok_or_else(|| rejected("a picture track needs a size"))?;
            let codec = match codec {
                Id::H264 => CodecId::H264,
                Id::Hevc => CodecId::Hevc,
                _ => CodecId::ProRes,
            };
            StreamParams::video(codec, scale, size.width(), size.height())
        }
        Id::Aac | Id::Pcm => {
            let rate = stream
                .params
                .sample_rate()
                .ok_or_else(|| rejected("a sound track needs a rate"))?;
            let channels = stream
                .params
                .channels()
                .ok_or_else(|| rejected("a sound track needs channels"))?;
            let codec = if *codec == Id::Aac {
                CodecId::Aac
            } else {
                CodecId::Pcm
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
                "mp4 export writes H.264, HEVC, ProRes, AAC, ALAC, and PCM",
            ));
        }
    };
    let extradata = stream.params.extradata();
    if *codec == Id::Prores
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
        let params = box_params(&stream).map_err(|error| self.fail(error))?;
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
            _ => config.to_vec(),
        };
        self.inner
            .set_extradata(self.tracks[index].boxed, stored)
            .map_err(|error| self.fail(boxed(error)))
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use cutlass_codec::{
        CodecId as Id, FrameSize, Packet, StreamCodec, StreamParams, Timescale, Timestamp,
    };
    use cutlass_container::{Mux, NewStream, TrackDuration};

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
}
