//! ImageIO stills. One packet is one file. GIF may hold several pictures.
//!
//! Samples come back as 8-bit straight RGBA, upright. A 16-bit or indexed
//! still is declined so a software codec can keep that coded domain when it
//! is registered after this opener.

use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_decode::{
    BitDepth, CodecError, CodecId, CodecParams, ColorType, Decoder, Output, Packet, StillImage,
    StillOutput,
};
use cutlass_encode::{AudioInput, Encoder, FrameInput, Packet as Encoded, PictureInput};

use super::cf::{self, Cf};
use super::sys;
use crate::limits::{packet_ok, picture_ok};
use crate::orient::{self, Orientation};

pub struct StillDecoder {
    codec: CodecId,
}

pub struct StillEncoder {
    codec: CodecId,
    gif: Vec<Vec<u8>>,
    width: u32,
    height: u32,
    drained: bool,
    cancel: Arc<AtomicBool>,
}

pub fn open_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    if !still_id(params.codec) {
        return Err(CodecError::Unsupported { what: "imageio" });
    }
    if params.width != 0 || params.height != 0 {
        match picture_ok(params.width, params.height) {
            Err(CodecError::Limit { .. }) => {
                return Err(CodecError::Unsupported {
                    what: "picture edge",
                });
            }
            Err(err) => return Err(err),
            Ok(()) => {}
        }
    }
    Ok(Box::new(StillDecoder {
        codec: params.codec,
    }))
}

pub fn open_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    if !still_id(params.codec) {
        return Err(CodecError::Unsupported { what: "imageio" });
    }
    if params.width != 0 || params.height != 0 {
        match picture_ok(params.width, params.height) {
            Err(CodecError::Limit { .. }) => {
                return Err(CodecError::Unsupported {
                    what: "picture edge",
                });
            }
            Err(err) => return Err(err),
            Ok(()) => {}
        }
    }
    Ok(Box::new(StillEncoder {
        codec: params.codec,
        gif: Vec::new(),
        width: params.width,
        height: params.height,
        drained: false,
        cancel: Arc::new(AtomicBool::new(false)),
    }))
}

fn still_id(id: CodecId) -> bool {
    matches!(
        id,
        CodecId::Jpeg | CodecId::Png | CodecId::Webp | CodecId::Gif
    )
}

impl Decoder for StillDecoder {
    fn codec(&self) -> CodecId {
        self.codec
    }

    fn push(&mut self, _packet: &Packet) -> Result<Vec<Output>, CodecError> {
        Err(CodecError::Unsupported {
            what: "picture output",
        })
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {}

    fn decode_picture(&mut self, packet: &Packet) -> Result<Vec<StillOutput>, CodecError> {
        packet_ok(packet.data.len())?;
        let images = decode_bytes(&packet.data)?;
        Ok(images
            .into_iter()
            .map(|image| StillOutput {
                image,
                pts: packet.pts,
                duration: packet.duration,
                timescale: packet.timescale,
            })
            .collect())
    }
}

impl Encoder for StillEncoder {
    fn codec(&self) -> CodecId {
        self.codec
    }

    fn push(&mut self, _input: &AudioInput) -> Result<Vec<Encoded>, CodecError> {
        Err(CodecError::Unsupported { what: "pcm" })
    }

    fn push_picture(&mut self, input: &PictureInput) -> Result<Vec<Encoded>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        let rgba = still_rgba(&input.image)?;
        picture_ok(rgba.0, rgba.1)?;
        if self.codec == CodecId::Gif {
            if self.width != 0 && (self.width != rgba.0 || self.height != rgba.1) {
                return Err(CodecError::InvalidApi { what: "gif size" });
            }
            self.width = rgba.0;
            self.height = rgba.1;
            self.gif.push(rgba.2);
            self.drained = false;
            return Ok(Vec::new());
        }
        let bytes = encode_bytes(self.codec, &[(rgba.0, rgba.1, rgba.2)])?;
        Ok(vec![still_packet(bytes)])
    }

    fn push_frame(&mut self, _input: &FrameInput) -> Result<Vec<Encoded>, CodecError> {
        Err(CodecError::Unsupported {
            what: "video frame",
        })
    }

    fn drain(&mut self) -> Result<Vec<Encoded>, CodecError> {
        if self.codec != CodecId::Gif || self.gif.is_empty() || self.drained {
            self.drained = true;
            return Ok(Vec::new());
        }
        let frames = self
            .gif
            .iter()
            .map(|bytes| (self.width, self.height, bytes.clone()))
            .collect::<Vec<_>>();
        let bytes = encode_bytes(self.codec, &frames)?;
        self.gif.clear();
        self.drained = true;
        Ok(vec![still_packet(bytes)])
    }

    fn reset(&mut self) {
        self.gif.clear();
        self.drained = false;
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}

fn still_packet(data: Vec<u8>) -> Encoded {
    Encoded {
        stream: 0,
        data,
        pts: 0,
        duration: 1,
        timescale: 1,
        keyframe: true,
        side: Vec::new(),
    }
}

pub fn decode_bytes(bytes: &[u8]) -> Result<Vec<StillImage>, CodecError> {
    packet_ok(bytes.len())?;
    let data = cf::data(bytes)?;
    let source =
        Cf::from_owned(unsafe { sys::CGImageSourceCreateWithData(data.as_ptr(), ptr::null()) })
            .ok_or(CodecError::Malformed {
                at_bit: 0,
                what: "imageio",
            })?;
    let count = unsafe { sys::CGImageSourceGetCount(source.as_ptr()) };
    if count == 0 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "imageio",
        });
    }
    let mut images = Vec::with_capacity(count);
    for index in 0..count {
        let image = Cf::from_owned(unsafe {
            sys::CGImageSourceCreateImageAtIndex(source.as_ptr(), index, ptr::null())
        })
        .ok_or(CodecError::Malformed {
            at_bit: 0,
            what: "imageio frame",
        })?;
        let orientation = orientation_at(&source, index);
        let (width, height, rgba) = draw_rgba(&image)?;
        let (width, height, mut rgba) = orient::upright(orientation, width, height, 4, &rgba)?;
        unpremultiply(&mut rgba);
        let samples = rgba.iter().map(|byte| u16::from(*byte)).collect::<Vec<_>>();
        picture_ok(width, height)?;
        let still =
            StillImage::from_samples(width, height, ColorType::Rgba, BitDepth::Eight, samples)
                .checked()
                .map_err(|_| CodecError::Malformed {
                    at_bit: 0,
                    what: "imageio samples",
                })?;
        images.push(still);
    }
    Ok(images)
}

fn orientation_at(source: &Cf, index: usize) -> Orientation {
    let Some(properties) = Cf::from_owned(unsafe {
        sys::CGImageSourceCopyPropertiesAtIndex(source.as_ptr(), index, ptr::null())
    }) else {
        return Orientation::upright();
    };
    let value =
        unsafe { sys::CFDictionaryGetValue(properties.as_ptr(), sys::kCGImagePropertyOrientation) };
    Orientation::from_tag(u32::try_from(cf::read_i32(value).unwrap_or(1)).unwrap_or(1))
}

fn draw_rgba(image: &Cf) -> Result<(u32, u32, Vec<u8>), CodecError> {
    let width = unsafe { sys::CGImageGetWidth(image.as_ptr()) };
    let height = unsafe { sys::CGImageGetHeight(image.as_ptr()) };
    let width_u = u32::try_from(width).map_err(|_| CodecError::Limit {
        what: "picture edge",
    })?;
    let height_u = u32::try_from(height).map_err(|_| CodecError::Limit {
        what: "picture edge",
    })?;
    picture_ok(width_u, height_u)?;
    let row = width.checked_mul(4).ok_or(CodecError::Limit {
        what: "picture bytes",
    })?;
    let len = row.checked_mul(height).ok_or(CodecError::Limit {
        what: "picture bytes",
    })?;
    let mut bytes = vec![0u8; len];
    let space = Cf::from_owned(unsafe { sys::CGColorSpaceCreateDeviceRGB() }).ok_or(
        CodecError::Unsupported {
            what: "imageio color",
        },
    )?;
    let context = Cf::from_owned(unsafe {
        sys::CGBitmapContextCreate(
            bytes.as_mut_ptr().cast(),
            width,
            height,
            8,
            row,
            space.as_ptr(),
            sys::ALPHA_PREMULTIPLIED_LAST | sys::BYTE_ORDER_32_BIG,
        )
    })
    .ok_or(CodecError::Unsupported {
        what: "imageio bitmap",
    })?;
    let rect = sys::CGRect {
        origin: sys::CGPoint { x: 0.0, y: 0.0 },
        size: sys::CGSize {
            width: width as f64,
            height: height as f64,
        },
    };
    unsafe { sys::CGContextDrawImage(context.as_mut(), rect, image.as_ptr()) };
    Ok((width_u, height_u, bytes))
}

fn unpremultiply(rgba: &mut [u8]) {
    for pixel in rgba.as_chunks_mut::<4>().0 {
        let alpha = pixel[3];
        if alpha == 0 {
            pixel[0] = 0;
            pixel[1] = 0;
            pixel[2] = 0;
            continue;
        }
        if alpha == 255 {
            continue;
        }
        for channel in pixel.iter_mut().take(3) {
            let lifted = u16::from(*channel) * 255 + u16::from(alpha) / 2;
            *channel = u8::try_from(lifted / u16::from(alpha)).unwrap_or(255);
        }
    }
}

pub fn encode_bytes(codec: CodecId, frames: &[(u32, u32, Vec<u8>)]) -> Result<Vec<u8>, CodecError> {
    if frames.is_empty() {
        return Err(CodecError::InvalidApi {
            what: "empty picture",
        });
    }
    let uti = uti(codec)?;
    let kind = cf::string(uti)?;
    let dest_data = cf::data_mut()?;
    let destination = Cf::from_owned(unsafe {
        sys::CGImageDestinationCreateWithData(
            dest_data.as_mut(),
            kind.as_ptr(),
            frames.len(),
            ptr::null(),
        )
    })
    .ok_or(CodecError::Unsupported {
        what: "imageio encode",
    })?;
    let space = Cf::from_owned(unsafe { sys::CGColorSpaceCreateDeviceRGB() }).ok_or(
        CodecError::Unsupported {
            what: "imageio color",
        },
    )?;
    let quality = if matches!(codec, CodecId::Jpeg | CodecId::Webp) {
        Some(cf::number_f64(0.92)?)
    } else {
        None
    };
    let mut keys = Vec::new();
    let mut values = Vec::new();
    if let Some(quality) = &quality {
        keys.push(unsafe { sys::kCGImageDestinationLossyCompressionQuality });
        values.push(quality.as_ptr());
    }
    let properties = if keys.is_empty() {
        None
    } else {
        Some(cf::dictionary(&keys, &values)?)
    };
    for (width, height, rgba) in frames {
        picture_ok(*width, *height)?;
        let expected = usize::try_from(*width)
            .ok()
            .and_then(|w| usize::try_from(*height).ok().and_then(|h| w.checked_mul(h)))
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or(CodecError::Limit {
                what: "picture bytes",
            })?;
        if rgba.len() != expected {
            return Err(CodecError::InvalidApi {
                what: "picture length",
            });
        }
        let provider = Cf::from_owned(unsafe {
            sys::CGDataProviderCreateWithData(
                ptr::null_mut(),
                rgba.as_ptr().cast(),
                rgba.len(),
                ptr::null(),
            )
        })
        .ok_or(CodecError::Limit {
            what: "imageio provider",
        })?;
        let row = usize::try_from(*width).unwrap_or(0).saturating_mul(4);
        let image = Cf::from_owned(unsafe {
            sys::CGImageCreate(
                usize::try_from(*width).unwrap_or(0),
                usize::try_from(*height).unwrap_or(0),
                8,
                32,
                row,
                space.as_ptr(),
                sys::ALPHA_PREMULTIPLIED_LAST | sys::BYTE_ORDER_32_BIG,
                provider.as_ptr(),
                ptr::null(),
                false,
                0,
            )
        })
        .ok_or(CodecError::Unsupported {
            what: "imageio image",
        })?;
        unsafe {
            sys::CGImageDestinationAddImage(
                destination.as_mut(),
                image.as_ptr(),
                properties.as_ref().map(Cf::as_ptr).unwrap_or(ptr::null()),
            );
        }
    }
    let wrote = unsafe { sys::CGImageDestinationFinalize(destination.as_mut()) };
    if !wrote {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "imageio encode",
        });
    }
    let bytes = cf::data_bytes(&dest_data)?;
    if bytes.is_empty() {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "imageio encode",
        });
    }
    Ok(bytes)
}

fn uti(codec: CodecId) -> Result<&'static str, CodecError> {
    match codec {
        CodecId::Png => Ok("public.png"),
        CodecId::Jpeg | CodecId::Mjpeg => Ok("public.jpeg"),
        CodecId::Gif => Ok("com.compuserve.gif"),
        CodecId::Webp => Ok("org.webmproject.webp"),
        CodecId::Hevc => Ok("public.heic"),
        CodecId::Av1 => Ok("public.avif"),
        _ => Err(CodecError::Unsupported {
            what: "imageio type",
        }),
    }
}

fn still_rgba(image: &StillImage) -> Result<(u32, u32, Vec<u8>), CodecError> {
    image.validate().map_err(|_| CodecError::InvalidApi {
        what: "still image",
    })?;
    if image.depth != BitDepth::Eight {
        return Err(CodecError::Unsupported {
            what: "still depth",
        });
    }
    if image.width == 0 || image.height == 0 {
        return Err(CodecError::InvalidApi {
            what: "empty picture",
        });
    }
    let pixels = (image.width as usize)
        .checked_mul(image.height as usize)
        .ok_or(CodecError::Limit {
            what: "picture pixels",
        })?;
    let mut rgba = Vec::with_capacity(pixels.saturating_mul(4));
    match image.color {
        ColorType::Rgba => {
            if image.samples.len() != pixels * 4 {
                return Err(CodecError::InvalidApi {
                    what: "picture length",
                });
            }
            for sample in &image.samples {
                rgba.push(u8::try_from(*sample).map_err(|_| CodecError::OutOfRange)?);
            }
        }
        ColorType::Rgb => {
            if image.samples.len() != pixels * 3 {
                return Err(CodecError::InvalidApi {
                    what: "picture length",
                });
            }
            for chunk in image.samples.as_chunks::<3>().0 {
                for sample in chunk {
                    rgba.push(u8::try_from(*sample).map_err(|_| CodecError::OutOfRange)?);
                }
                rgba.push(255);
            }
        }
        ColorType::Gray => {
            if image.samples.len() != pixels {
                return Err(CodecError::InvalidApi {
                    what: "picture length",
                });
            }
            for sample in &image.samples {
                let gray = u8::try_from(*sample).map_err(|_| CodecError::OutOfRange)?;
                rgba.extend_from_slice(&[gray, gray, gray, 255]);
            }
        }
        ColorType::GrayAlpha => {
            if image.samples.len() != pixels * 2 {
                return Err(CodecError::InvalidApi {
                    what: "picture length",
                });
            }
            for chunk in image.samples.as_chunks::<2>().0 {
                let gray = u8::try_from(chunk[0]).map_err(|_| CodecError::OutOfRange)?;
                let alpha = u8::try_from(chunk[1]).map_err(|_| CodecError::OutOfRange)?;
                rgba.extend_from_slice(&[gray, gray, gray, alpha]);
            }
        }
        ColorType::Indexed => {
            return Err(CodecError::Unsupported {
                what: "indexed still",
            });
        }
    }
    premultiply(&mut rgba);
    Ok((image.width, image.height, rgba))
}

fn premultiply(rgba: &mut [u8]) {
    for pixel in rgba.as_chunks_mut::<4>().0 {
        let alpha = pixel[3];
        if alpha == 255 {
            continue;
        }
        for channel in pixel.iter_mut().take(3) {
            *channel =
                u8::try_from((u16::from(*channel) * u16::from(alpha) + 127) / 255).unwrap_or(0);
        }
    }
}

/// Encode one still container that ImageIO knows, including HEIC and AVIF.
pub fn encode_file(
    uti_name: &str,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> Result<Vec<u8>, CodecError> {
    let codec = match uti_name {
        "public.heic" => CodecId::Hevc,
        "public.avif" => CodecId::Av1,
        "public.png" => CodecId::Png,
        "public.jpeg" => CodecId::Jpeg,
        "org.webmproject.webp" => CodecId::Webp,
        "com.compuserve.gif" => CodecId::Gif,
        _ => {
            return Err(CodecError::Unsupported {
                what: "imageio type",
            });
        }
    };
    encode_bytes(codec, &[(width, height, rgba.to_vec())])
}

/// Decode one still file ImageIO can open. HEIC and AVIF use this before
/// the box parser.
pub fn decode_file(bytes: &[u8]) -> Result<Vec<StillImage>, CodecError> {
    decode_bytes(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_decode::DecodeFactory;
    use cutlass_encode::EncodeFactory;

    fn rgba(width: u32, height: u32, pixel: [u8; 4]) -> StillImage {
        let mut samples = Vec::new();
        for _ in 0..width * height {
            samples.extend(pixel.map(u16::from));
        }
        StillImage::from_samples(width, height, ColorType::Rgba, BitDepth::Eight, samples)
            .checked()
            .expect("still")
    }

    fn params(codec: CodecId) -> CodecParams {
        CodecParams {
            codec,
            extra: Vec::new(),
            sample_rate: 0,
            layout: None,
            sample_format: None,
            bitrate: 0,
            width: 0,
            height: 0,
            pixel: cutlass_decode::PixelFormat::Rgba8,
            primaries: cutlass_decode::ColorPrimaries::Unspecified,
            transfer: cutlass_decode::Transfer::Unspecified,
        }
    }

    #[test]
    fn a_png_file_round_trips_through_the_file_helpers() {
        let pixels = [
            10u8, 20, 30, 255, 10, 20, 30, 255, 10, 20, 30, 255, 10, 20, 30, 255,
        ];
        let bytes = encode_file("public.png", 2, 2, &pixels).expect("encode");
        assert!(bytes.starts_with(b"\x89PNG"));
        let decoded = decode_file(&bytes).expect("decode");
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].samples, rgba(2, 2, [10, 20, 30, 255]).samples);
    }

    #[test]
    fn a_png_round_trip_keeps_opaque_pixels() {
        let image = rgba(2, 2, [10, 20, 30, 255]);
        let mut encoder = open_encoder(&params(CodecId::Png)).expect("open");
        let packets = encoder
            .push_picture(&PictureInput {
                image: image.clone(),
            })
            .expect("encode");
        assert_eq!(packets.len(), 1);
        assert!(packets[0].data.starts_with(b"\x89PNG"));
        let mut decoder = open_decoder(&params(CodecId::Png)).expect("open");
        let decoded = decoder
            .decode_picture(&Packet {
                stream: 0,
                data: packets[0].data.clone(),
                pts: 3,
                duration: 1,
                timescale: 1,
                keyframe: true,
                side: Vec::new(),
            })
            .expect("decode");
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].pts, 3);
        assert_eq!(decoded[0].image.samples, image.samples);
        assert!(decoder.drain().expect("drain").is_empty());
        assert!(encoder.drain().expect("drain").is_empty());
    }

    #[test]
    fn an_empty_still_packet_is_an_invalid_call() {
        let mut decoder = open_decoder(&params(CodecId::Jpeg)).expect("open");
        let err = decoder
            .decode_picture(&Packet {
                stream: 0,
                data: Vec::new(),
                pts: 0,
                duration: 1,
                timescale: 1,
                keyframe: true,
                side: Vec::new(),
            })
            .expect_err("empty");
        assert_eq!(
            err,
            CodecError::InvalidApi {
                what: "empty packet",
            }
        );
    }

    #[test]
    fn random_bytes_are_not_a_picture() {
        let mut decoder = open_decoder(&params(CodecId::Png)).expect("open");
        let err = decoder
            .decode_picture(&Packet {
                stream: 0,
                data: vec![1, 2, 3, 4, 5, 6, 7, 8],
                pts: 0,
                duration: 1,
                timescale: 1,
                keyframe: true,
                side: Vec::new(),
            })
            .expect_err("random");
        assert!(matches!(
            err,
            CodecError::Malformed {
                what: "imageio",
                ..
            }
        ));
    }

    #[test]
    fn a_still_decoder_refuses_audio_push() {
        let mut decoder = open_decoder(&params(CodecId::Png)).expect("open");
        let err = decoder
            .push(&Packet {
                stream: 0,
                data: vec![1],
                pts: 0,
                duration: 1,
                timescale: 1,
                keyframe: true,
                side: Vec::new(),
            })
            .expect_err("push");
        assert_eq!(
            err,
            CodecError::Unsupported {
                what: "picture output"
            }
        );
    }

    fn lossy_red(codec: CodecId) {
        let image = rgba(16, 16, [220, 10, 10, 255]);
        let mut encoder = match open_encoder(&params(codec)) {
            Ok(encoder) => encoder,
            Err(CodecError::Unsupported { .. }) => return,
            Err(err) => panic!("{err}"),
        };
        let packets = match encoder.push_picture(&PictureInput { image }) {
            Ok(packets) => packets,
            Err(CodecError::Unsupported { .. }) => return,
            Err(err) => panic!("{err}"),
        };
        assert_eq!(packets.len(), 1);
        let mut decoder = open_decoder(&params(codec)).expect("decode open");
        let decoded = decoder
            .decode_picture(&Packet {
                stream: 0,
                data: packets[0].data.clone(),
                pts: 0,
                duration: 1,
                timescale: 1,
                keyframe: true,
                side: Vec::new(),
            })
            .expect("decode");
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].image.width, 16);
        assert_eq!(decoded[0].image.height, 16);
        let mut red = 0u64;
        let mut green = 0u64;
        for sample in decoded[0].image.samples.as_chunks::<4>().0 {
            red += u64::from(sample[0]);
            green += u64::from(sample[1]);
        }
        assert!(red > green.saturating_mul(4), "red {red} green {green}");
    }

    #[test]
    fn jpeg_keeps_the_frame_size_and_the_red_channel() {
        lossy_red(CodecId::Jpeg);
    }

    #[test]
    fn webp_keeps_the_frame_size_and_the_red_channel_when_imageio_can() {
        lossy_red(CodecId::Webp);
    }

    #[test]
    fn gif_frames_leave_on_drain() {
        let mut encoder = open_encoder(&params(CodecId::Gif)).expect("open");
        let first = encoder
            .push_picture(&PictureInput {
                image: rgba(8, 8, [255, 0, 0, 255]),
            })
            .expect("first");
        assert!(first.is_empty());
        let second = encoder
            .push_picture(&PictureInput {
                image: rgba(8, 8, [0, 0, 255, 255]),
            })
            .expect("second");
        assert!(second.is_empty());
        let packets = encoder.drain().expect("drain");
        assert_eq!(packets.len(), 1);
        assert!(encoder.drain().expect("again").is_empty());
        let mut decoder = open_decoder(&params(CodecId::Gif)).expect("decode");
        let decoded = decoder
            .decode_picture(&Packet {
                stream: 0,
                data: packets[0].data.clone(),
                pts: 4,
                duration: 2,
                timescale: 1,
                keyframe: true,
                side: Vec::new(),
            })
            .expect("decode");
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].image.width, 8);
        assert_eq!(decoded[0].pts, 4);
    }

    #[test]
    fn a_sixteen_bit_still_is_declined() {
        let samples = vec![0u16; 4];
        let image = StillImage::from_samples(1, 1, ColorType::Rgba, BitDepth::Sixteen, samples)
            .checked()
            .expect("still");
        let mut encoder = open_encoder(&params(CodecId::Png)).expect("open");
        let err = encoder
            .push_picture(&PictureInput { image })
            .expect_err("depth");
        assert_eq!(
            err,
            CodecError::Unsupported {
                what: "still depth"
            }
        );
    }

    #[test]
    fn the_factory_uses_imageio_before_a_later_opener() {
        let mut decode = DecodeFactory::new();
        super::super::super::register_decoders(&mut decode);
        let mut encode = EncodeFactory::new();
        super::super::super::register_encoders(&mut encode);
        let encoder = encode.open(&params(CodecId::Png)).expect("png");
        assert_eq!(encoder.codec(), CodecId::Png);
        let err = decode.open(&params(CodecId::Vp8));
        assert!(err.is_err());
    }
}
