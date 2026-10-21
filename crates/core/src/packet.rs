//! Compressed data for one frame.

use crate::Timestamp;

/// One compressed frame, as demuxed from a container or produced by an encoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Packet {
    /// The compressed bytes in the container's layout. For H.264 and HEVC that
    /// is length-prefixed NAL units, as MP4 and MKV store them. The config
    /// record gives the prefix size.
    pub data: Vec<u8>,
    pub pts: Timestamp,
    /// Decode time, never later than `pts`, and rising from packet to packet.
    /// `None` when the container doesn't record one, as in MKV. Decoders use
    /// it to put frames back in presentation order with the least delay.
    pub dts: Option<Timestamp>,
    pub duration: Option<Timestamp>,
    /// The frame decodes without any earlier packet.
    pub keyframe: bool,
}
