//! Maps this parser's tracks and samples onto the shared demux contract.
//!
//! The box parser keeps its own packet and stream types. This module is the
//! boundary `engine` can hold as `Box<dyn Demux>`.

use crate::legacy::{
    CodecId as LegacyCodec, Edit as LegacyEdit, Error, MediaKind, Packet as LegacyPacket, Source,
    Stream as LegacyStream,
};
use cutlass_codec::{
    CodecId, FrameSize, Packet, PacketError, StreamCodec, StreamParams, Timescale, Timestamp,
    UnknownCodec,
};
use cutlass_container::{
    Edit, Input, Language, OpenError, ReadError, SeekError, Stream, TrackId, Trim,
};
use std::io::{Read, Seek};

pub(crate) struct RandomSource<R> {
    input: Input<R>,
    length: u64,
}

impl<R> RandomSource<R> {
    pub(crate) fn new(input: Input<R>, length: u64) -> Self {
        Self { input, length }
    }
}

impl<R: Read> RandomSource<R> {
    pub(crate) fn consumed(&self) -> u64 {
        self.input.consumed()
    }
}

impl<R: Read + Seek> Source for RandomSource<R> {
    fn len(&self) -> u64 {
        self.length
    }

    fn read_exact_at(&mut self, at: u64, buf: &mut [u8]) -> Result<(), Error> {
        self.input.seek(at).map_err(source_error)?;
        self.input.read_exact(buf).map_err(source_error)?;
        Ok(())
    }
}

pub(crate) fn map_open(error: Error) -> OpenError {
    match error {
        Error::NotThisFamily => OpenError::Unrecognized {
            family: cutlass_codec::ContainerId::Isobmff,
        },
        Error::UnsupportedBrand { .. } => OpenError::UnsupportedBrand {
            family: cutlass_codec::ContainerId::Isobmff,
        },
        other => OpenError::Source {
            message: other.to_string(),
        },
    }
}

pub(crate) fn map_read(error: Error) -> ReadError {
    match error {
        Error::Missing { what: "track" } => ReadError::Seek(SeekError::UnknownTrack),
        Error::SeekLimited {
            reason: "track has no sample index",
        } => ReadError::Seek(SeekError::NoIndex),
        Error::SeekLimited {
            reason: "track has no sync samples",
        } => ReadError::Seek(SeekError::NoIndex),
        other => ReadError::Source {
            message: other.to_string(),
        },
    }
}

pub(crate) fn map_streams(
    movie_timescale: u32,
    streams: &[LegacyStream],
) -> Result<Vec<Stream>, OpenError> {
    let mut shared = Vec::with_capacity(streams.len());
    for stream in streams {
        shared.push(map_stream(movie_timescale, stream)?);
    }
    Ok(shared)
}

pub(crate) fn to_shared(packet: LegacyPacket) -> Result<Packet, ReadError> {
    let timestamp = Timestamp::try_new(
        Some(packet.dts),
        Some(packet.pts),
        Some(i64::from(packet.duration)),
    )
    .map_err(packet_error)?;
    let timescale = Timescale::new(packet.timescale.get());
    Packet::try_new(
        packet.track_id,
        packet.payload,
        timestamp,
        timescale,
        packet.keyframe,
    )
    .map_err(packet_error)
}

fn map_stream(movie_timescale: u32, stream: &LegacyStream) -> Result<Stream, OpenError> {
    let params = &stream.params;
    let timescale = Timescale::new(params.timescale.get());
    if !timescale.is_clock() {
        return Err(OpenError::Source {
            message: "track timescale is zero".to_owned(),
        });
    }
    let codec = map_codec(params.codec)?;
    let described = match params.kind {
        MediaKind::Video(video) => {
            let size =
                FrameSize::try_new(video.width, video.height).map_err(|_| OpenError::Source {
                    message: "frame size is empty".to_owned(),
                })?;
            StreamParams::video(codec, size, params.extradata.clone())
        }
        MediaKind::Audio(audio) => StreamParams::try_audio(
            codec,
            audio.sample_rate,
            audio.channels,
            params.extradata.clone(),
        )
        .map_err(|error| OpenError::Source {
            message: error.to_string(),
        })?,
        MediaKind::Data => StreamParams::opaque(codec, params.extradata.clone()),
    };
    let mut edits = Vec::with_capacity(stream.edits.len());
    for edit in &stream.edits {
        edits.push(map_edit(edit, movie_timescale, params.timescale.get())?);
    }
    Ok(Stream::new(
        TrackId::new(params.track_id),
        described,
        timescale,
        language(params.language),
        edits,
        Trim::default(),
    ))
}

fn map_edit(
    edit: &LegacyEdit,
    movie_timescale: u32,
    media_timescale: u32,
) -> Result<Edit, OpenError> {
    let duration = scale_duration(edit.segment_duration, movie_timescale, media_timescale)?;
    Edit::try_new(edit.media_time, duration, edit.media_rate, 65_536).map_err(|error| {
        OpenError::Source {
            message: error.to_string(),
        }
    })
}

fn scale_duration(value: u64, from: u32, to: u32) -> Result<i64, OpenError> {
    if from == 0 || to == 0 {
        return Err(OpenError::Source {
            message: "edit timescale is zero".to_owned(),
        });
    }
    let scaled = u128::from(value) * u128::from(to) / u128::from(from);
    i64::try_from(scaled).map_err(|_| OpenError::Source {
        message: "edit duration does not fit".to_owned(),
    })
}

fn language(bytes: [u8; 3]) -> Option<Language> {
    let text = std::str::from_utf8(&bytes).ok()?;
    Language::new(text).ok()
}

fn map_codec(codec: LegacyCodec) -> Result<StreamCodec, OpenError> {
    let known = match codec {
        LegacyCodec::H264 => CodecId::H264,
        LegacyCodec::Hevc => CodecId::Hevc,
        LegacyCodec::Av1 => CodecId::Av1,
        LegacyCodec::Vp8 => CodecId::Vp8,
        LegacyCodec::Vp9 => CodecId::Vp9,
        LegacyCodec::Mpeg4 => CodecId::Mpeg4,
        LegacyCodec::Mpeg2 => CodecId::Mpeg2,
        LegacyCodec::Mjpeg => CodecId::Mjpeg,
        LegacyCodec::ProRes => CodecId::Prores,
        LegacyCodec::Aac => CodecId::Aac,
        LegacyCodec::Opus => CodecId::Opus,
        LegacyCodec::Mp3 => CodecId::Mp3,
        LegacyCodec::Flac => CodecId::Flac,
        LegacyCodec::Pcm => CodecId::Pcm,
        LegacyCodec::Vorbis => CodecId::Vorbis,
        LegacyCodec::Alac => CodecId::Alac,
        LegacyCodec::Ac3 => CodecId::Ac3,
        LegacyCodec::Jpeg => CodecId::Jpeg,
        LegacyCodec::Png => CodecId::Png,
        LegacyCodec::Webp => CodecId::Webp,
        LegacyCodec::Gif => CodecId::Gif,
        LegacyCodec::Unknown(tag) => {
            let unknown = UnknownCodec::new(tag.to_vec()).map_err(|error| OpenError::Source {
                message: error.to_string(),
            })?;
            return Ok(StreamCodec::Unknown(unknown));
        }
    };
    Ok(StreamCodec::Known(known))
}

fn source_error(error: cutlass_container::SourceError) -> Error {
    Error::Source {
        message: error.to_string(),
    }
}

fn packet_error(error: PacketError) -> ReadError {
    ReadError::Source {
        message: error.to_string(),
    }
}
