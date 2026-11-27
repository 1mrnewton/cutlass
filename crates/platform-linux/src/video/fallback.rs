//! Hand a hardware decoder's stream to the Rust decoder.
//!
//! A hardware decoder can open and still refuse the stream once it reads
//! the parameter sets: a profile, bit depth, or size the driver does not
//! decode. Until the first picture comes out, [`Fallback`] keeps the
//! packets pushed since open or reset. If the hardware fails before then,
//! the Rust decoder for the codec opens and those packets are pushed into
//! it again, so the caller sees pictures, not an error. After the first
//! picture an error is the caller's, as with any decoder.

use cutlass_decode::{CodecError, CodecId, CodecParams, DecodeFactory, Decoder, Output, Packet};

/// Packets kept while waiting for the first picture.
const MAX_HELD: usize = 256;

/// Bytes kept while waiting for the first picture.
const MAX_HELD_BYTES: usize = 64 * 1024 * 1024;

/// Opens the decoder that takes over.
pub type Software = fn(&CodecParams) -> Result<Box<dyn Decoder>, CodecError>;

/// The Rust decoder for `params`, through a factory that holds only the
/// Rust openers.
pub fn rust_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    let mut factory = DecodeFactory::new();
    cutlass_h264::register_decoders(&mut factory);
    cutlass_hevc::register_decoders(&mut factory);
    cutlass_av1::register_decoders(&mut factory);
    factory.open(params)
}

/// A hardware decoder that hands over to software if it fails before its
/// first picture.
pub struct Fallback {
    params: CodecParams,
    inner: Box<dyn Decoder>,
    backend: &'static str,
    software: Software,
    hardware: bool,
    /// Packets since open or reset, while no picture has come out.
    held: Vec<Packet>,
    held_bytes: usize,
    holding: bool,
}

impl Fallback {
    pub fn new(params: &CodecParams, inner: Box<dyn Decoder>, backend: &'static str) -> Self {
        Self::with_software(params, inner, backend, rust_decoder)
    }

    pub fn with_software(
        params: &CodecParams,
        inner: Box<dyn Decoder>,
        backend: &'static str,
        software: Software,
    ) -> Self {
        Self {
            params: params.clone(),
            inner,
            backend,
            software,
            hardware: true,
            held: Vec::new(),
            held_bytes: 0,
            holding: true,
        }
    }

    /// Whether the hardware is still decoding.
    #[cfg(test)]
    pub fn hardware(&self) -> bool {
        self.hardware
    }

    fn hold(&mut self, packet: &Packet) {
        if !self.holding {
            return;
        }
        let bytes = self.held_bytes.saturating_add(packet.data.len());
        if self.held.len() >= MAX_HELD || bytes > MAX_HELD_BYTES {
            // Too long without a picture to replay: stop keeping them.
            self.holding = false;
            self.held.clear();
            self.held_bytes = 0;
            return;
        }
        self.held.push(packet.clone());
        self.held_bytes = bytes;
    }

    fn produced(&mut self) {
        self.holding = false;
        self.held.clear();
        self.held_bytes = 0;
    }

    /// Open the software decoder and push the held packets into it.
    fn hand_over(&mut self, error: &CodecError) -> Result<Vec<Output>, CodecError> {
        let mut software = (self.software)(&self.params)?;
        cutlass_logger::warn!(
            "{} could not decode this {:?} stream ({error}); the Rust decoder takes it",
            self.backend,
            self.params.codec
        );
        let mut out = Vec::new();
        for packet in std::mem::take(&mut self.held) {
            out.extend(software.push(&packet)?);
        }
        self.inner = software;
        self.hardware = false;
        self.holding = false;
        self.held_bytes = 0;
        Ok(out)
    }
}

impl Decoder for Fallback {
    fn codec(&self) -> CodecId {
        self.inner.codec()
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        if !self.hardware || !self.holding {
            return self.inner.push(packet);
        }
        match self.inner.push(packet) {
            Ok(out) => {
                if out.is_empty() {
                    self.hold(packet);
                } else {
                    self.produced();
                }
                Ok(out)
            }
            Err(error) => {
                let mut out = self.hand_over(&error)?;
                out.extend(self.inner.push(packet)?);
                Ok(out)
            }
        }
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        if !self.hardware || !self.holding {
            return self.inner.drain();
        }
        match self.inner.drain() {
            Ok(out) => {
                if !out.is_empty() {
                    self.produced();
                }
                Ok(out)
            }
            Err(error) => {
                let mut out = self.hand_over(&error)?;
                out.extend(self.inner.drain()?);
                Ok(out)
            }
        }
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.held.clear();
        self.held_bytes = 0;
        self.holding = self.hardware;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_decode::{ColorPrimaries, PictureInfo, PictureOutput, PixelFormat, Transfer};

    fn picture(pts: i64) -> Output {
        let info = PictureInfo::new(
            2,
            2,
            PixelFormat::Yuv420p,
            ColorPrimaries::Bt709,
            Transfer::Bt709,
        );
        Output::Picture(PictureOutput::cpu(info, vec![0; 6], pts, 30).expect("picture"))
    }

    /// Fails on the packet whose first byte is 0xff; gives a picture for
    /// one whose first byte is 1.
    struct Hardware;

    impl Decoder for Hardware {
        fn codec(&self) -> CodecId {
            CodecId::H264
        }
        fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
            match packet.data.first() {
                Some(0xff) => Err(CodecError::Unsupported {
                    what: "test profile",
                }),
                Some(1) => Ok(vec![picture(packet.pts)]),
                _ => Ok(Vec::new()),
            }
        }
        fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
            Ok(Vec::new())
        }
        fn reset(&mut self) {}
    }

    /// One picture for every packet.
    struct Software;

    impl Decoder for Software {
        fn codec(&self) -> CodecId {
            CodecId::H264
        }
        fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
            Ok(vec![picture(packet.pts)])
        }
        fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
            Ok(Vec::new())
        }
        fn reset(&mut self) {}
    }

    fn open_software(_params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
        Ok(Box::new(Software))
    }

    fn packet(first: u8, pts: i64) -> Packet {
        Packet::video(vec![first, 0], pts, 1, 30, pts == 0)
    }

    fn fallback() -> Fallback {
        let params = CodecParams::with_format(CodecId::H264, 64, 48, PixelFormat::Yuv420p);
        Fallback::with_software(&params, Box::new(Hardware), "test", open_software)
    }

    #[test]
    fn a_failure_before_the_first_picture_replays_into_software() {
        let mut decoder = fallback();
        assert!(decoder.push(&packet(0, 0)).expect("held").is_empty());
        assert!(decoder.push(&packet(0, 1)).expect("held").is_empty());
        // The software decoder gives a picture a packet: the two held and
        // the one that failed.
        let out = decoder.push(&packet(0xff, 2)).expect("handed over");
        let times: Vec<i64> = out
            .iter()
            .map(|output| match output {
                Output::Picture(picture) => picture.pts(),
                Output::Audio(_) => panic!("picture"),
            })
            .collect();
        assert_eq!(times, [0, 1, 2]);
        assert!(!decoder.hardware());
    }

    #[test]
    fn after_the_first_picture_an_error_is_the_callers() {
        let mut decoder = fallback();
        assert_eq!(decoder.push(&packet(1, 0)).expect("picture").len(), 1);
        assert!(decoder.push(&packet(0xff, 1)).is_err());
        assert!(decoder.hardware());
    }

    #[test]
    fn a_reset_holds_again() {
        let mut decoder = fallback();
        assert_eq!(decoder.push(&packet(1, 0)).expect("picture").len(), 1);
        decoder.reset();
        assert!(decoder.push(&packet(0, 5)).expect("held").is_empty());
        let out = decoder.push(&packet(0xff, 6)).expect("handed over");
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn the_rust_decoders_open_for_every_hardware_codec() {
        for codec in [CodecId::H264, CodecId::Hevc, CodecId::Av1] {
            let params = CodecParams::with_format(codec, 64, 48, PixelFormat::Yuv420p);
            assert!(rust_decoder(&params).is_ok(), "{codec:?}");
        }
    }
}
