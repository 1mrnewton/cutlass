//! Demux and mux traits.
//!
//! Timestamps inside a container stay in the file timescale. A demuxer
//! yields one [`Packet`] at a time. A muxer takes those packets back.
//! Incremental byte streams that are not yet a whole packet return
//! [`CodecError::NeedMoreData`] from the container that owns the framing.

#![forbid(unsafe_code)]

use cutlass_codec::{CodecError, CodecParams, ContainerId, Packet};

#[derive(Clone, Debug, PartialEq)]
pub struct StreamInfo {
    pub index: u32,
    pub params: CodecParams,
    pub timescale: u32,
}

pub trait Demuxer {
    fn id(&self) -> ContainerId;

    fn streams(&self) -> &[StreamInfo];

    fn next_packet(&mut self) -> Result<Option<Packet>, CodecError>;
}

pub trait Muxer {
    fn id(&self) -> ContainerId;

    fn add_stream(&mut self, params: CodecParams) -> Result<u32, CodecError>;

    fn write_packet(&mut self, packet: &Packet) -> Result<(), CodecError>;

    /// Finishes the file. A second call returns [`CodecError::InvalidApi`].
    fn finish(&mut self) -> Result<Vec<u8>, CodecError>;
}

#[cfg(test)]
mod tests {
    use super::{Demuxer, Muxer, StreamInfo};
    use cutlass_codec::{ChannelLayout, CodecError, CodecId, CodecParams, ContainerId, Packet};

    struct Mem {
        streams: Vec<StreamInfo>,
        packets: Vec<Packet>,
        cursor: usize,
        done: bool,
    }

    impl Demuxer for Mem {
        fn id(&self) -> ContainerId {
            ContainerId::Adts
        }

        fn streams(&self) -> &[StreamInfo] {
            &self.streams
        }

        fn next_packet(&mut self) -> Result<Option<Packet>, CodecError> {
            if self.cursor >= self.packets.len() {
                return Ok(None);
            }
            let packet = self.packets[self.cursor].clone();
            self.cursor += 1;
            Ok(Some(packet))
        }
    }

    impl Muxer for Mem {
        fn id(&self) -> ContainerId {
            ContainerId::Adts
        }

        fn add_stream(&mut self, params: CodecParams) -> Result<u32, CodecError> {
            let index = u32::try_from(self.streams.len())
                .map_err(|_| CodecError::Limit { what: "streams" })?;
            self.streams.push(StreamInfo {
                index,
                timescale: params.sample_rate,
                params,
            });
            Ok(index)
        }

        fn write_packet(&mut self, packet: &Packet) -> Result<(), CodecError> {
            if self.done {
                return Err(CodecError::InvalidApi {
                    what: "write after finish",
                });
            }
            self.packets.push(packet.clone());
            Ok(())
        }

        fn finish(&mut self) -> Result<Vec<u8>, CodecError> {
            if self.done {
                return Err(CodecError::InvalidApi {
                    what: "second finish",
                });
            }
            self.done = true;
            Ok(self.packets.iter().flat_map(|p| p.data.clone()).collect())
        }
    }

    #[test]
    fn mux_then_demux_returns_the_same_packet_bytes() {
        let mut mux = Mem {
            streams: Vec::new(),
            packets: Vec::new(),
            cursor: 0,
            done: false,
        };
        let params = CodecParams::audio(CodecId::Aac, 48_000, ChannelLayout::mono());
        mux.add_stream(params).expect("stream");
        mux.write_packet(&Packet::audio(vec![9, 8, 7], 0, 1024, 48_000))
            .expect("write");
        let bytes = mux.finish().expect("finish");
        assert_eq!(bytes, vec![9, 8, 7]);
        assert!(mux.finish().is_err());
        let mut demux = Mem {
            streams: mux.streams.clone(),
            packets: mux.packets.clone(),
            cursor: 0,
            done: false,
        };
        let packet = demux.next_packet().expect("packet").expect("some");
        assert_eq!(packet.data, vec![9, 8, 7]);
        assert!(demux.next_packet().expect("end").is_none());
    }
}
