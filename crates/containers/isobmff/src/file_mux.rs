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
use crate::mux::{IsoBmffMux, MuxOptions};

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
        Self {
            inner: IsoBmffMux::new(writer, MuxOptions::default()),
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

/// The box writer's parameters for `stream`: H.264 pictures or AAC sound.
fn box_params(stream: &NewStream) -> Result<StreamParams, MuxError> {
    let StreamCodec::Known(codec) = stream.params.codec() else {
        return Err(rejected("mp4 export writes H.264 and AAC"));
    };
    let scale = BoxScale(stream.timescale.ticks_per_second());
    let mut params = match codec {
        Id::H264 => {
            let size = stream
                .params
                .size()
                .ok_or_else(|| rejected("a picture track needs a size"))?;
            StreamParams::video(CodecId::H264, scale, size.width(), size.height())
        }
        Id::Aac => {
            let rate = stream
                .params
                .sample_rate()
                .ok_or_else(|| rejected("a sound track needs a rate"))?;
            let channels = stream
                .params
                .channels()
                .ok_or_else(|| rejected("a sound track needs channels"))?;
            StreamParams::audio(CodecId::Aac, scale, rate, channels, 16, SampleFormat::S16Le)
        }
        _ => return Err(rejected("mp4 export writes H.264 and AAC")),
    };
    params.extradata = stream.params.extradata().to_vec();
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
    use super::bare_asc;

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
