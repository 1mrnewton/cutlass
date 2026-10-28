//! The framing traits AAC transport was written against.
//!
//! ADTS, LOAS, and the M4A writer use these. The shared demux trait is
//! implemented beside them and does not replace them.

use cutlass_codec::access::{CodecError, CodecParams, Packet};
use cutlass_codec::ContainerId;

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

    fn finish(&mut self) -> Result<Vec<u8>, CodecError>;
}
