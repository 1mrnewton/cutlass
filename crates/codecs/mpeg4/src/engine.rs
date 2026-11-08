//! The MPEG-4 Visual decoder on the engine's codec traits.
//!
//! Playback opens a [`Decoder`] through [`DecodeFactory::open`] and takes
//! packed 4:2:0 pictures in display order. The decoder configuration is
//! the `esds` DecoderSpecificInfo: the VOS, visual object and VOL headers.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use cutlass_decode::{
    CodecError, CodecId, CodecParams, ColorMatrix, ColorPrimaries, DecodeFactory, Decoder, Output,
    Packet, PictureInfo, PictureOutput, PixelFormat, SideData, Transfer,
};

use crate::dec::{self, Image};
use crate::header::Colour;

/// The largest picture playback opens: the 13-bit VOL edges allow 8191.
const MAX_PIXELS: u64 = 8192 * 8192;

/// Registers the MPEG-4 Visual decoder that playback opens.
pub fn register_engine_decoders(factory: &mut DecodeFactory) {
    factory.register(CodecId::Mpeg4, open_engine_decoder);
}

fn open_engine_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(EngineDecoder::open(params)?))
}

/// MPEG-4 Visual for playback: one packed 4:2:0 picture per displayed VOP.
///
/// A not-coded VOP gives no picture, so the previous one stays on screen.
/// Pictures leave in display order; each takes the smallest packet time
/// not yet given out, which is its presentation time when the container
/// stores them in decode order. The engine's 4:2:0 pictures have even
/// edges, so an odd-sized stream drops its last column or row.
pub struct EngineDecoder {
    inner: dec::Decoder,
    pts: BinaryHeap<Reverse<i64>>,
    timescale: u32,
}

impl EngineDecoder {
    /// Opens with the container's parameters; `params.extra` holds the
    /// stream headers when the container carries them out of band.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        Self::open_limited(params, MAX_PIXELS)
    }

    /// [`Self::open`] refusing layers over `max_pixels`.
    pub(crate) fn open_limited(params: &CodecParams, max_pixels: u64) -> Result<Self, CodecError> {
        if params.codec != CodecId::Mpeg4 {
            return Err(CodecError::Unsupported { what: "codec" });
        }
        let mut inner = dec::Decoder::with_max_pixels(max_pixels);
        inner.configure(&params.extra)?;
        Ok(Self {
            inner,
            pts: BinaryHeap::new(),
            timescale: 0,
        })
    }

    fn output(&mut self, image: Image, fallback: i64) -> Result<Output, CodecError> {
        let pts = self.pts.pop().map_or(fallback, |Reverse(pts)| pts);
        let colour = self.inner.colour();
        Ok(Output::Picture(PictureOutput::cpu(
            info(&image, colour),
            image.bytes,
            pts,
            self.timescale,
        )?))
    }
}

/// The picture's description with the visual object's colour.
fn info(image: &Image, colour: Colour) -> PictureInfo {
    let primaries = colour
        .primaries
        .map_or(ColorPrimaries::Unspecified, primaries);
    let transfer = colour.transfer.map_or(Transfer::Unspecified, transfer);
    let mut info = PictureInfo::new(
        image.width,
        image.height,
        PixelFormat::Yuv420p,
        primaries,
        transfer,
    );
    if let Some(code) = colour.matrix {
        info.matrix = matrix(code);
        info.cicp_matrix = code;
    }
    info
}

/// ISO/IEC 14496-2 Table 6-7, the ISO/IEC 23091-2 code points.
fn primaries(code: u8) -> ColorPrimaries {
    match code {
        1 => ColorPrimaries::Bt709,
        4 => ColorPrimaries::Bt470m,
        5 | 6 => ColorPrimaries::Bt601,
        7 => ColorPrimaries::Smpte240m,
        8 => ColorPrimaries::Film,
        9 => ColorPrimaries::Bt2020,
        _ => ColorPrimaries::Unspecified,
    }
}

/// ISO/IEC 14496-2 Table 6-8.
fn transfer(code: u8) -> Transfer {
    match code {
        1 => Transfer::Bt709,
        4..=6 => Transfer::Bt601,
        7 => Transfer::Smpte240m,
        8 => Transfer::Linear,
        9 => Transfer::Log100,
        10 => Transfer::LogSqrt,
        _ => Transfer::Unspecified,
    }
}

/// ISO/IEC 14496-2 Table 6-9.
fn matrix(code: u8) -> ColorMatrix {
    match code {
        1 => ColorMatrix::Bt709,
        4 => ColorMatrix::Fcc,
        5 | 6 => ColorMatrix::Bt601,
        7 => ColorMatrix::Smpte240m,
        _ => ColorMatrix::Unspecified,
    }
}

impl Decoder for EngineDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Mpeg4
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        for side in &packet.side {
            if let SideData::Config(config) = side {
                self.inner.configure(config)?;
            }
        }
        if packet.data.is_empty() {
            return Ok(Vec::new());
        }
        self.timescale = packet.timescale;
        let decoded = self.inner.decode(&packet.data)?;
        if decoded.timed {
            self.pts.push(Reverse(packet.pts));
        }
        decoded
            .pictures
            .into_iter()
            .map(|image| self.output(image, packet.pts))
            .collect()
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        let fallback = self.pts.peek().map_or(0, |Reverse(pts)| *pts);
        let out = self
            .inner
            .drain()
            .into_iter()
            .map(|image| self.output(image, fallback))
            .collect();
        self.pts.clear();
        out
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.pts.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ASP: &[u8] = include_bytes!("../tests/fixtures/asp.m4v");

    fn packet(data: &[u8]) -> Packet {
        Packet {
            stream: 0,
            data: data.to_vec(),
            pts: 0,
            duration: 1,
            timescale: 25,
            keyframe: true,
            side: Vec::new(),
        }
    }

    fn first_vop() -> usize {
        ASP.windows(4)
            .position(|w| w == [0, 0, 1, 0xB6])
            .expect("a VOP")
    }

    #[test]
    fn a_layer_over_the_pixel_limit_is_refused() {
        let params = CodecParams::video(CodecId::Mpeg4, 0, 0);
        let mut decoder = EngineDecoder::open_limited(&params, 48 * 32 - 1).expect("open");
        let refused = decoder.push(&packet(ASP));
        assert!(
            matches!(refused, Err(CodecError::Limit { .. })),
            "{refused:?}"
        );

        let mut configured = params.clone();
        configured.extra = ASP[..first_vop()].to_vec();
        let refused = EngineDecoder::open_limited(&configured, 48 * 32 - 1)
            .and_then(|mut d| d.push(&packet(&ASP[first_vop()..])));
        assert!(
            matches!(refused, Err(CodecError::Limit { .. })),
            "{refused:?}"
        );

        let mut decoder = EngineDecoder::open_limited(&params, 48 * 32).expect("open");
        assert!(!decoder.push(&packet(ASP)).expect("push").is_empty());
    }

    #[test]
    fn reserved_colour_codes_are_unspecified() {
        let colour = Colour {
            full_range: false,
            primaries: Some(3),
            transfer: Some(0),
            matrix: Some(200),
        };
        let image = Image {
            width: 2,
            height: 2,
            bytes: vec![0; 6],
        };
        let reserved = info(&image, colour);
        assert_eq!(reserved.primaries, ColorPrimaries::Unspecified);
        assert_eq!(reserved.transfer, Transfer::Unspecified);
        assert_eq!(reserved.matrix, ColorMatrix::Unspecified);
        assert_eq!(reserved.cicp_matrix, 200);

        let bt709 = Colour {
            full_range: false,
            primaries: Some(1),
            transfer: Some(1),
            matrix: Some(1),
        };
        let standard = info(&image, bt709);
        assert_eq!(standard.primaries, ColorPrimaries::Bt709);
        assert_eq!(standard.transfer, Transfer::Bt709);
        assert_eq!(standard.matrix, ColorMatrix::Bt709);
    }
}
