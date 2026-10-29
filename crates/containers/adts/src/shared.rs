//! Owned-reader adapters from ADTS and LOAS onto the shared demux trait.
//!
//! The parsers still accept bytes incrementally. This wrapper owns the
//! reader, pulls only what the next frame needs, and leaves timestamps in
//! the sample-rate timescale.

use crate::{AdtsDemuxer, Demuxer, LatmDemuxer, StreamInfo};
use cutlass_codec::access::{CodecError, Packet as AccessUnit};
use cutlass_codec::{ContainerId, Packet, StreamCodec, StreamParams, Timescale, Timestamp};
use cutlass_container::{
    Input, Limits, OpenError, ReadError, SeekError, SeekLanding, SeekTo, Stream, TrackId, Trim,
};
use std::io::Read;

trait ByteFeed {
    fn push_bytes(&mut self, data: &[u8]) -> Result<(), CodecError>;
    fn finish_bytes(&mut self) -> Result<(), CodecError>;
    fn streams(&self) -> &[StreamInfo];
    fn next_packet(&mut self) -> Result<Option<AccessUnit>, CodecError>;
}

impl ByteFeed for AdtsDemuxer {
    fn push_bytes(&mut self, data: &[u8]) -> Result<(), CodecError> {
        Self::push_bytes(self, data)
    }

    fn finish_bytes(&mut self) -> Result<(), CodecError> {
        Self::finish_bytes(self)
    }

    fn streams(&self) -> &[StreamInfo] {
        Demuxer::streams(self)
    }

    fn next_packet(&mut self) -> Result<Option<AccessUnit>, CodecError> {
        Demuxer::next_packet(self)
    }
}

impl ByteFeed for LatmDemuxer {
    fn push_bytes(&mut self, data: &[u8]) -> Result<(), CodecError> {
        Self::push_bytes(self, data)
    }

    fn finish_bytes(&mut self) -> Result<(), CodecError> {
        Self::finish_bytes(self)
    }

    fn streams(&self) -> &[StreamInfo] {
        Demuxer::streams(self)
    }

    fn next_packet(&mut self) -> Result<Option<AccessUnit>, CodecError> {
        Demuxer::next_packet(self)
    }
}

struct FramedFile<D, R> {
    family: ContainerId,
    demux: D,
    input: Input<R>,
    shared: Vec<Stream>,
    cancelled: bool,
    ended: bool,
}

pub struct AdtsFile<R>(FramedFile<AdtsDemuxer, R>);
pub struct LatmFile<R>(FramedFile<LatmDemuxer, R>);

impl<R: Read> AdtsFile<R> {
    pub fn open(reader: R) -> Result<Self, OpenError> {
        Ok(Self(FramedFile::open_with(
            ContainerId::Adts,
            AdtsDemuxer::new(),
            reader,
        )?))
    }
}

impl<R: Read> LatmFile<R> {
    pub fn open(reader: R) -> Result<Self, OpenError> {
        Ok(Self(FramedFile::open_with(
            ContainerId::Latm,
            LatmDemuxer::new(),
            reader,
        )?))
    }
}

impl<D: ByteFeed, R: Read> FramedFile<D, R> {
    fn open_with(family: ContainerId, demux: D, reader: R) -> Result<Self, OpenError> {
        let mut file = Self {
            family,
            demux,
            input: Input::forward(reader, Limits::standard()),
            shared: Vec::new(),
            cancelled: false,
            ended: false,
        };
        while file.demux.streams().is_empty() {
            if !file.pull()? {
                return Err(OpenError::Unrecognized { family });
            }
        }
        file.shared = map_streams(file.demux.streams())?;
        Ok(file)
    }

    fn pull(&mut self) -> Result<bool, OpenError> {
        if self.ended {
            return Ok(false);
        }
        let mut buf = [0u8; 4096];
        let n = self.input.read(&mut buf).map_err(open_source)?;
        if n == 0 {
            self.ended = true;
            self.demux.finish_bytes().map_err(open_codec)?;
            return Ok(false);
        }
        self.demux.push_bytes(&buf[..n]).map_err(open_codec)?;
        Ok(true)
    }

    fn read_access_unit(&mut self) -> Result<Option<AccessUnit>, ReadError> {
        loop {
            match self.demux.next_packet() {
                Ok(Some(packet)) => return Ok(Some(packet)),
                Ok(None) => return Ok(None),
                Err(CodecError::NeedMoreData) if !self.ended => {
                    let mut buf = [0u8; 4096];
                    let n = self.input.read(&mut buf).map_err(read_source)?;
                    if n == 0 {
                        self.ended = true;
                        self.demux.finish_bytes().map_err(read_codec)?;
                    } else {
                        self.demux.push_bytes(&buf[..n]).map_err(read_codec)?;
                    }
                }
                Err(error) => return Err(read_codec(error)),
            }
        }
    }
}

impl<D: ByteFeed, R: Read> cutlass_container::Demux for FramedFile<D, R> {
    fn family(&self) -> ContainerId {
        self.family
    }

    fn streams(&self) -> &[Stream] {
        &self.shared
    }

    fn gaps(&self) -> &[cutlass_container::Gap] {
        &[]
    }

    fn read_packet(&mut self) -> Result<Option<Packet>, ReadError> {
        if self.cancelled {
            return Err(ReadError::Cancelled);
        }
        self.read_access_unit()?.map(to_shared).transpose()
    }

    fn seek(&mut self, _to: SeekTo) -> Result<SeekLanding, ReadError> {
        if self.cancelled {
            return Err(ReadError::Cancelled);
        }
        Err(ReadError::Seek(SeekError::ForwardOnly))
    }

    fn cancel(&mut self) {
        self.cancelled = true;
    }

    fn consumed(&self) -> u64 {
        self.input.consumed()
    }
}

macro_rules! shared_demux {
    ($name:ident) => {
        impl<R: Read> cutlass_container::Demux for $name<R> {
            fn family(&self) -> ContainerId {
                cutlass_container::Demux::family(&self.0)
            }

            fn streams(&self) -> &[Stream] {
                cutlass_container::Demux::streams(&self.0)
            }

            fn gaps(&self) -> &[cutlass_container::Gap] {
                cutlass_container::Demux::gaps(&self.0)
            }

            fn read_packet(&mut self) -> Result<Option<Packet>, ReadError> {
                cutlass_container::Demux::read_packet(&mut self.0)
            }

            fn seek(&mut self, to: SeekTo) -> Result<SeekLanding, ReadError> {
                cutlass_container::Demux::seek(&mut self.0, to)
            }

            fn cancel(&mut self) {
                cutlass_container::Demux::cancel(&mut self.0);
            }

            fn consumed(&self) -> u64 {
                cutlass_container::Demux::consumed(&self.0)
            }
        }
    };
}

shared_demux!(AdtsFile);
shared_demux!(LatmFile);

fn map_streams(streams: &[StreamInfo]) -> Result<Vec<Stream>, OpenError> {
    let mut shared = Vec::with_capacity(streams.len());
    for stream in streams {
        let Some(layout) = stream.params.layout.as_ref() else {
            return Err(OpenError::Source {
                message: "audio stream has no channel layout".to_owned(),
            });
        };
        let channels = u16::try_from(layout.channels().len()).map_err(|_| OpenError::Source {
            message: "channel count does not fit".to_owned(),
        })?;
        let params = StreamParams::try_audio(
            StreamCodec::Known(stream.params.codec),
            stream.params.sample_rate,
            channels,
            stream.params.extra.clone(),
        )
        .map_err(|error| OpenError::Source {
            message: error.to_string(),
        })?;
        let timescale = Timescale::new(stream.timescale);
        if !timescale.is_clock() {
            return Err(OpenError::Source {
                message: "track timescale is zero".to_owned(),
            });
        }
        shared.push(Stream::new(
            TrackId::new(stream.index),
            params,
            timescale,
            None,
            Vec::new(),
            Trim::default(),
        ));
    }
    Ok(shared)
}

fn to_shared(packet: AccessUnit) -> Result<Packet, ReadError> {
    if packet.duration < 0 {
        return Err(ReadError::Source {
            message: "packet duration is negative".to_owned(),
        });
    }
    let timestamp =
        Timestamp::try_new(None, Some(packet.pts), Some(packet.duration)).map_err(|error| {
            ReadError::Source {
                message: error.to_string(),
            }
        })?;
    Packet::try_new(
        packet.stream,
        packet.data,
        timestamp,
        Timescale::new(packet.timescale),
        packet.keyframe,
    )
    .map_err(|error| ReadError::Source {
        message: error.to_string(),
    })
}

fn open_codec(error: CodecError) -> OpenError {
    OpenError::Source {
        message: error.to_string(),
    }
}

fn open_source(error: cutlass_container::SourceError) -> OpenError {
    OpenError::from_header(error)
}

fn read_codec(error: CodecError) -> ReadError {
    if matches!(error, CodecError::Cancelled) {
        ReadError::Cancelled
    } else {
        ReadError::Source {
            message: error.to_string(),
        }
    }
}

fn read_source(error: cutlass_container::SourceError) -> ReadError {
    ReadError::from_packet(error)
}
