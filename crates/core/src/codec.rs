//! Codec names and the stream config a decoder starts from.

/// A video codec Cutlass decodes. `docs/codecs.md` lists which backend handles
/// each one on each OS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VideoCodec {
    H264,
    Hevc,
    Vp8,
    Vp9,
    Av1,
    Mpeg2,
    /// MPEG-4 Part 2: DivX, Xvid, and many older cameras.
    Mpeg4Part2,
    Mjpeg,
    ProRes,
}

/// What a video decoder needs from the container before the first packet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoStreamConfig {
    pub codec: VideoCodec,
    /// Size from the container's sample entry, in pixels.
    pub width: u32,
    pub height: u32,
    /// The codec configuration record exactly as the container stores it:
    /// `avcC` for H.264, `hvcC` for HEVC, `vpcC` for VP9, `av1C` for AV1, and
    /// the `esds` decoder config for MPEG-4 Part 2. Empty for codecs without one.
    pub config_record: Vec<u8>,
}
