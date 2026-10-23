//! A video decoder for one stream.
//!
//! On Apple this is VideoToolbox. Elsewhere [`VideoSession::open`] returns
//! [`Error::NoDecoder`] until that platform's codec crate is connected.

use cutlass_core::{ColorTags, ContentLight, MasteringDisplay, Rotation, VideoStreamConfig};

use crate::Error;

#[cfg(target_vendor = "apple")]
use crate::error::from_videotoolbox;
#[cfg(target_vendor = "apple")]
use cutlass_core::{Packet, VideoFrame};
#[cfg(target_vendor = "apple")]
use cutlass_videotoolbox::PixelBuffer;

/// One open video decoder.
pub struct VideoSession {
    #[cfg(target_vendor = "apple")]
    decoder: cutlass_videotoolbox::Decoder,
    /// No system decoder is linked on this target, so a session cannot exist.
    #[cfg(not(target_vendor = "apple"))]
    _private: (),
}

impl VideoSession {
    /// Opens the system decoder for `config`.
    ///
    /// `packets` supplies the presentation and decode times. A B-frame can be
    /// shown before its decode time, and the session keeps that delay so the
    /// pictures come out in presentation order. `color`, `mastering`,
    /// `content_light`, and `rotation` are copied onto every frame. `None`
    /// leaves that colour part unset. [`Rotation::Zero`] leaves the picture
    /// upright. The caller has already chosen [`crate::DecoderKind::System`].
    /// A software session is not opened here, because no Rust video decoder
    /// is connected.
    #[cfg(target_vendor = "apple")]
    pub fn open(
        config: &VideoStreamConfig,
        color: Option<ColorTags>,
        mastering: Option<MasteringDisplay>,
        content_light: Option<ContentLight>,
        rotation: Rotation,
        packets: &[Packet],
    ) -> Result<Self, Error> {
        let decoder = cutlass_videotoolbox::Decoder::new(
            config,
            cutlass_videotoolbox::DecoderOptions {
                color,
                mastering,
                content_light,
                rotation,
                composition_offset: cutlass_videotoolbox::composition_offset(packets),
                ..cutlass_videotoolbox::DecoderOptions::default()
            },
        )
        .map_err(from_videotoolbox)?;
        Ok(Self { decoder })
    }

    /// Opens the system decoder for `config`.
    ///
    /// No system decoder is linked here, so the colour and rotation arguments
    /// are ignored.
    #[cfg(not(target_vendor = "apple"))]
    pub fn open(
        config: &VideoStreamConfig,
        color: Option<ColorTags>,
        mastering: Option<MasteringDisplay>,
        content_light: Option<ContentLight>,
        rotation: Rotation,
        packets: &[cutlass_core::Packet],
    ) -> Result<Self, Error> {
        let _ = (color, mastering, content_light, rotation, packets);
        Err(Error::NoDecoder(config.codec))
    }
}

#[cfg(target_vendor = "apple")]
impl VideoSession {
    /// Sends one compressed packet, in decode order.
    pub fn send_packet(&mut self, packet: &Packet) -> Result<(), Error> {
        self.decoder.send_packet(packet).map_err(from_videotoolbox)
    }

    /// The next frame, in presentation order, when the decoder has one.
    pub fn receive_frame(&mut self) -> Result<Option<VideoFrame<PixelBuffer>>, Error> {
        match self.decoder.receive_frame() {
            None => Ok(None),
            Some(Ok(frame)) => Ok(Some(frame)),
            Some(Err(error)) => Err(from_videotoolbox(error)),
        }
    }

    /// Tells the decoder no more packets are coming.
    pub fn flush(&mut self) -> Result<(), Error> {
        self.decoder.flush().map_err(from_videotoolbox)
    }

    /// Drops buffered frames so the same session can start again.
    pub fn reset(&mut self) -> Result<(), Error> {
        self.decoder.reset().map_err(from_videotoolbox)
    }

    /// Whether the operating system is decoding this stream in hardware.
    pub fn is_hardware(&self) -> Result<bool, Error> {
        self.decoder.is_hardware().map_err(from_videotoolbox)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_session_can_move_to_another_thread() {
        fn assert_send<T: Send>() {}
        assert_send::<VideoSession>();
    }

    #[cfg(not(target_vendor = "apple"))]
    #[test]
    fn opening_fails_where_no_system_decoder_is_linked() {
        let config = VideoStreamConfig {
            codec: cutlass_core::VideoCodec::H264,
            width: 16,
            height: 16,
            config_record: vec![1],
        };
        assert!(matches!(
            VideoSession::open(&config, None, None, None, Rotation::Zero, &[]),
            Err(Error::NoDecoder(cutlass_core::VideoCodec::H264))
        ));
    }
}
