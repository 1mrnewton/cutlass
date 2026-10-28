//! Adapt framed MP3 values onto the shared container types.

use cutlass_codec::{Packet as SharedPacket, StreamCodec, StreamParams, Timescale, Timestamp};
use cutlass_container::{
    ReadError, SeekError, Stream as SharedStream, TrackId, Trim as SharedTrim,
};

use crate::api::{ContainerError, Packet, Stream};

pub(crate) fn shared_streams(streams: &[Stream]) -> Result<Vec<SharedStream>, ContainerError> {
    let mut shared = Vec::with_capacity(streams.len());
    for stream in streams {
        let params = StreamParams::try_audio(
            StreamCodec::Known(stream.params.codec),
            stream.params.sample_rate,
            stream.params.channels,
            stream.params.extra.clone(),
        )
        .map_err(|_| ContainerError::Unsupported {
            what: "audio params",
        })?;
        let trim = match stream.trim {
            Some(trim) => SharedTrim::new(
                Some(u64::from(trim.encoder_delay)),
                Some(u64::from(trim.trailing_padding)),
            ),
            None => SharedTrim::default(),
        };
        shared.push(SharedStream::new(
            TrackId::new(stream.index),
            params,
            Timescale::new(stream.timescale),
            None,
            Vec::new(),
            trim,
        ));
    }
    Ok(shared)
}

pub(crate) fn map_read(error: ContainerError) -> ReadError {
    match error {
        ContainerError::Cancelled => ReadError::Cancelled,
        ContainerError::SeekUnavailable => ReadError::Seek(SeekError::NoIndex),
        ContainerError::Limit { what } => ReadError::Source {
            message: format!("limit exceeded: {what}"),
        },
        ContainerError::Truncated { what } => ReadError::Source {
            message: format!("truncated {what}"),
        },
        ContainerError::Malformed { what } => ReadError::Source {
            message: format!("malformed {what}"),
        },
        ContainerError::Unsupported { what } => ReadError::Source {
            message: format!("unsupported {what}"),
        },
        ContainerError::InvalidApi { what } => ReadError::Source {
            message: format!("invalid call: {what}"),
        },
        ContainerError::Io { kind } => ReadError::Source {
            message: format!("io error: {kind:?}"),
        },
        ContainerError::NoFrames => ReadError::Source {
            message: "no frames".to_owned(),
        },
    }
}

pub(crate) fn to_shared(packet: Packet) -> Result<SharedPacket, ReadError> {
    let timestamp =
        Timestamp::try_new(None, Some(packet.pts), Some(packet.duration)).map_err(|_| {
            ReadError::Source {
                message: "packet duration is negative".to_owned(),
            }
        })?;
    SharedPacket::try_new(
        packet.stream,
        packet.data,
        timestamp,
        Timescale::new(packet.timescale),
        packet.keyframe,
    )
    .map_err(|_| ReadError::Source {
        message: "packet timescale does not match its timestamps".to_owned(),
    })
}
