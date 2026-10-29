//! Public AV1 decoder. One access unit is one temporal unit.

use cutlass_decode::{
    CodecError, CodecId, CodecParams, Decoded, Decoder, Packet, PixelFormat, Plane, VideoOutput,
    Yuv420,
};

use crate::header::{FrameInfo, Sequence, header_is_addressable, parse_unit};
use crate::tile::{DecodedPicture, decode_tile};

/// Rust AV1 decoder. Reference slots stay inside this type.
pub struct Av1Decoder {
    sequence: Option<Sequence>,
    refs: [Option<Yuv420>; 8],
}

impl Av1Decoder {
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Av1 {
            return Err(CodecError::Unsupported { what: "codec" });
        }
        if params.pixel != PixelFormat::None && params.pixel != PixelFormat::Yuv420p {
            return Err(CodecError::Unsupported {
                what: "pixel format",
            });
        }
        if !crate::bits::signed_and_little_endian_hold()
            || !crate::symbol::literal_holds()
            || !crate::tables::default_tables_match_their_shapes()
            || !crate::dims::rectangular_transforms_are_numbered()
        {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "default tables",
            });
        }
        Ok(Self {
            sequence: None,
            refs: std::array::from_fn(|_| None),
        })
    }
}

impl Decoder for Av1Decoder {
    fn codec(&self) -> CodecId {
        CodecId::Av1
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Decoded>, CodecError> {
        if packet.data.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "empty packet",
            });
        }
        let (sequence, frame) = parse_unit(&packet.data, self.sequence.as_ref())?;
        header_is_addressable(&sequence, &frame)?;
        let picture = if frame.show_existing {
            let Some(stored) = self.refs[usize::from(frame.frame_to_show)].clone() else {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "missing reference",
                });
            };
            stored
        } else {
            owned(decode_tile(&sequence, &frame)?)?
        };
        self.store(&frame, &picture);
        self.sequence = Some(sequence);
        if !frame.show_frame {
            return Ok(Vec::new());
        }
        Ok(vec![Decoded::Video(VideoOutput {
            picture,
            pts: packet.pts,
            duration: packet.duration,
            timescale: packet.timescale,
            keyframe: frame.intra && !frame.show_existing,
        })])
    }

    fn drain(&mut self) -> Result<Vec<Decoded>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        self.refs = std::array::from_fn(|_| None);
    }
}

impl Av1Decoder {
    fn store(&mut self, frame: &FrameInfo, picture: &Yuv420) {
        for slot in 0..8 {
            if frame.refresh_flags & (1 << slot) != 0 {
                self.refs[slot] = Some(picture.clone());
            }
        }
    }
}

fn owned(picture: DecodedPicture) -> Result<Yuv420, CodecError> {
    let width =
        u32::try_from(picture.width).map_err(|_| CodecError::Limit { what: "frame size" })?;
    let height =
        u32::try_from(picture.height).map_err(|_| CodecError::Limit { what: "frame size" })?;
    Yuv420::new(
        Plane::new(width, height, picture.y)?,
        Plane::new(width.div_ceil(2), height.div_ceil(2), picture.u)?,
        Plane::new(width.div_ceil(2), height.div_ceil(2), picture.v)?,
    )
}

#[cfg(test)]
mod tests {
    use super::Av1Decoder;
    use crate::encode::Av1Encoder;
    use crate::register_decoders;
    use cutlass_decode::{CodecId, CodecParams, DecodeFactory, Decoder};
    use cutlass_encode::{Encoder, Plane, VideoInput, Yuv420};

    fn solid(width: u32, height: u32, sample: u8) -> Yuv420 {
        let y = vec![sample; (width * height) as usize];
        let cw = width.div_ceil(2);
        let ch = height.div_ceil(2);
        let chroma = vec![128u8; (cw * ch) as usize];
        Yuv420::new(
            Plane::new(width, height, y).expect("y"),
            Plane::new(cw, ch, chroma.clone()).expect("u"),
            Plane::new(cw, ch, chroma).expect("v"),
        )
        .expect("yuv")
    }

    fn input(picture: Yuv420, pts: i64) -> VideoInput {
        VideoInput {
            picture,
            pts,
            duration: 1,
            timescale: 30,
        }
    }

    #[test]
    fn the_factory_opens_the_rust_decoder() {
        let mut factory = DecodeFactory::new();
        register_decoders(&mut factory);
        let params = CodecParams::video(CodecId::Av1, 16, 16);
        let decoder = factory.open(&params).expect("open");
        assert_eq!(decoder.codec(), CodecId::Av1);
    }

    #[test]
    fn an_empty_packet_is_refused() {
        let params = CodecParams::video(CodecId::Av1, 16, 16);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let packet = cutlass_decode::Packet {
            stream: 0,
            data: Vec::new(),
            pts: 0,
            duration: 1,
            timescale: 30,
            keyframe: true,
            side: Vec::new(),
        };
        assert!(matches!(
            decoder.push(&packet),
            Err(cutlass_decode::CodecError::InvalidApi { .. })
        ));
    }

    #[test]
    fn a_truncated_packet_leaves_no_picture() {
        let params = CodecParams::video(CodecId::Av1, 16, 16);
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let packet = cutlass_decode::Packet {
            stream: 0,
            data: vec![0x12, 0x00],
            pts: 0,
            duration: 1,
            timescale: 30,
            keyframe: true,
            side: Vec::new(),
        };
        assert!(decoder.push(&packet).is_err());
        assert!(decoder.drain().expect("drain").is_empty());
        assert!(decoder.drain().expect("drain again").is_empty());
    }

    #[test]
    fn a_repeated_picture_shows_the_stored_reference() {
        let params = CodecParams::video(CodecId::Av1, 32, 32);
        let mut encoder = Av1Encoder::open(&params).expect("encoder");
        encoder.set_quality(40);
        let picture = solid(32, 32, 40);
        let first = encoder
            .push_picture(&input(picture.clone(), 0))
            .expect("first");
        let second = encoder.push_picture(&input(picture, 1)).expect("second");
        assert!(second[0].data.len() < first[0].data.len());
        let mut decoder = Av1Decoder::open(&params).expect("decoder");
        let a = decoder.push(&first[0]).expect("decode first");
        let b = decoder.push(&second[0]).expect("decode second");
        assert_eq!(
            a[0].as_video().expect("video").picture.y.samples(),
            b[0].as_video().expect("video").picture.y.samples()
        );
    }

    #[test]
    fn a_bad_packet_keeps_the_reference() {
        let params = CodecParams::video(CodecId::Av1, 32, 32);
        let mut encoder = Av1Encoder::open(&params).expect("encoder");
        let picture = solid(32, 32, 90);
        let first = encoder
            .push_picture(&input(picture.clone(), 0))
            .expect("first");
        let second = encoder.push_picture(&input(picture, 1)).expect("second");
        let mut decoder = Av1Decoder::open(&params).expect("decoder");
        let shown = decoder.push(&first[0]).expect("decode");
        let mut broken = second[0].clone();
        broken.data = vec![0xff, 0x00, 0x01];
        assert!(decoder.push(&broken).is_err());
        let again = decoder.push(&second[0]).expect("reference still there");
        assert_eq!(
            shown[0].as_video().expect("video").picture.y.samples(),
            again[0].as_video().expect("video").picture.y.samples()
        );
    }

    #[test]
    fn reset_drops_the_reference() {
        let params = CodecParams::video(CodecId::Av1, 32, 32);
        let mut encoder = Av1Encoder::open(&params).expect("encoder");
        let picture = solid(32, 32, 70);
        let first = encoder
            .push_picture(&input(picture.clone(), 0))
            .expect("first");
        let second = encoder.push_picture(&input(picture, 1)).expect("second");
        let mut decoder = Av1Decoder::open(&params).expect("decoder");
        decoder.push(&first[0]).expect("decode");
        decoder.reset();
        assert!(decoder.push(&second[0]).is_err());
        assert!(decoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn hostile_bytes_leave_the_reference() {
        let params = CodecParams::video(CodecId::Av1, 32, 32);
        let mut encoder = Av1Encoder::open(&params).expect("encoder");
        let picture = solid(32, 32, 50);
        let first = encoder
            .push_picture(&input(picture.clone(), 0))
            .expect("first");
        let repeat = encoder.push_picture(&input(picture, 1)).expect("repeat");
        let mut decoder = Av1Decoder::open(&params).expect("open");
        let shown = decoder.push(&first[0]).expect("decode");
        let mut state = 0x1234_5678u32;
        for _ in 0..200 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let len = (state % 80) as usize;
            if len == 0 {
                continue;
            }
            let mut data = vec![0u8; len];
            for byte in &mut data {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *byte = (state >> 16) as u8;
            }
            let packet = cutlass_decode::Packet {
                stream: 0,
                data,
                pts: 0,
                duration: 1,
                timescale: 30,
                keyframe: false,
                side: Vec::new(),
            };
            assert!(decoder.push(&packet).is_err());
        }
        let again = decoder.push(&repeat[0]).expect("reference still there");
        assert_eq!(
            shown[0].as_video().expect("video").picture.y.samples(),
            again[0].as_video().expect("video").picture.y.samples()
        );
    }
}
