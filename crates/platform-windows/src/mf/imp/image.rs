//! Windows Imaging Component stills: JPEG and PNG. One packet is one file.
//!
//! Samples come back straight RGBA, upright: a JPEG's EXIF orientation is
//! applied. An 8-bit file is 8-bit RGBA. A 16-bit PNG stays 16 bits, read
//! as `64bppRGBA`, so registering WIC ahead of `cutlass-png` does not lose
//! that depth the way ImageIO does on Apple. An indexed PNG is expanded to
//! RGBA. Encode takes 8-bit or 16-bit gray, gray-alpha, RGB, or RGBA, and an
//! indexed still is [`CodecError::Unsupported`].

use std::ffi::c_void;
use std::mem::transmute;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_decode::orient::{self, Orientation};
use cutlass_decode::{
    BitDepth, CodecError, CodecId, CodecParams, ColorType, Decoder, Output, Packet, StillImage,
    StillOutput,
};
use cutlass_encode::{AudioInput, Encoder, FrameInput, Packet as Encoded, PictureInput};

use super::com::{self, Com, Hresult, com, vslot};
use crate::format::Guid;
use crate::limits::{packet_ok, picture_ok};

const CLSID_WIC_FACTORY: Guid = Guid {
    data1: 0xCACA_F262,
    data2: 0x9370,
    data3: 0x4615,
    data4: [0xA1, 0x3B, 0x9F, 0x55, 0x39, 0xDA, 0x4C, 0x0A],
};
const IID_WIC_FACTORY: Guid = Guid {
    data1: 0xEC5E_C8A9,
    data2: 0xC395,
    data3: 0x4314,
    data4: [0x9C, 0x77, 0x54, 0xD7, 0xA9, 0x35, 0xFF, 0x70],
};
const CONTAINER_PNG: Guid = Guid {
    data1: 0x1B7C_FAF4,
    data2: 0x713F,
    data3: 0x473C,
    data4: [0xBB, 0xCD, 0x61, 0x37, 0x42, 0x5F, 0xAE, 0xAF],
};
const CONTAINER_JPEG: Guid = Guid {
    data1: 0x19E4_A5AA,
    data2: 0x5662,
    data3: 0x4FC5,
    data4: [0xA0, 0xC0, 0x17, 0x58, 0x02, 0x8E, 0x10, 0x57],
};
const RGBA8: Guid = Guid {
    data1: 0xF5C7_AD2D,
    data2: 0x6A8D,
    data3: 0x43DD,
    data4: [0xA7, 0xA8, 0xA2, 0x99, 0x35, 0x26, 0x1A, 0xE9],
};
const RGBA16: Guid = Guid {
    data1: 0x6FDD_C324,
    data2: 0x4E03,
    data3: 0x4BFE,
    data4: [0xB1, 0x85, 0x3D, 0x77, 0x76, 0x8D, 0xC9, 0x16],
};

/// `WICBitmapEncoderNoCache`.
const NO_CACHE: u32 = 2;
/// `VT_UI2`, the type EXIF orientation is read as.
const VT_UI2: u16 = 18;

#[link(name = "shlwapi")]
unsafe extern "system" {
    fn SHCreateMemStream(data: *const u8, size: u32) -> *mut c_void;
}

#[link(name = "ole32")]
unsafe extern "system" {
    fn PropVariantClear(value: *mut PropVariant) -> Hresult;
}

/// `PROPVARIANT` as it is laid out on 64-bit Windows.
#[repr(C)]
struct PropVariant {
    kind: u16,
    reserved: [u16; 3],
    value: u64,
    record: u64,
}

pub struct StillDecoder {
    codec: CodecId,
}

pub struct StillEncoder {
    codec: CodecId,
    cancel: Arc<AtomicBool>,
}

pub fn open_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    still_size(params)?;
    factory()?;
    Ok(Box::new(StillDecoder {
        codec: params.codec,
    }))
}

pub fn open_encoder(params: &CodecParams) -> Result<Box<dyn Encoder>, CodecError> {
    still_size(params)?;
    factory()?;
    Ok(Box::new(StillEncoder {
        codec: params.codec,
        cancel: Arc::new(AtomicBool::new(false)),
    }))
}

fn still_size(params: &CodecParams) -> Result<(), CodecError> {
    if !matches!(params.codec, CodecId::Jpeg | CodecId::Png) {
        return Err(CodecError::Unsupported { what: "wic codec" });
    }
    if params.width == 0 && params.height == 0 {
        return Ok(());
    }
    match picture_ok(params.width, params.height) {
        Err(CodecError::Limit { .. }) => Err(CodecError::Unsupported {
            what: "picture edge",
        }),
        other => other,
    }
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
        let image = decode_bytes(&packet.data)?;
        Ok(vec![StillOutput {
            image,
            pts: packet.pts,
            duration: packet.duration,
            timescale: packet.timescale,
        }])
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
        let data = encode_image(self.codec, &input.image)?;
        Ok(vec![Encoded {
            stream: 0,
            data,
            pts: 0,
            duration: 1,
            timescale: 1,
            keyframe: true,
            side: Vec::new(),
        }])
    }

    fn push_frame(&mut self, _input: &FrameInput) -> Result<Vec<Encoded>, CodecError> {
        Err(CodecError::Unsupported {
            what: "video frame",
        })
    }

    fn drain(&mut self) -> Result<Vec<Encoded>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {}

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}

/// One file to one upright still.
pub fn decode_bytes(bytes: &[u8]) -> Result<StillImage, CodecError> {
    let factory = factory()?;
    let stream = memory_stream(bytes)?;
    let mut raw = ptr::null_mut();
    let code = com!(
        factory.raw(),
        4,
        fn(*mut c_void, *const Guid, u32, *mut *mut c_void),
        stream.raw(),
        ptr::null(),
        0,
        &mut raw
    );
    let decoder = owned(code, raw).map_err(|_| not_a_picture())?;
    let mut raw = ptr::null_mut();
    let code = com!(decoder.raw(), 13, fn(u32, *mut *mut c_void), 0, &mut raw);
    let frame = owned(code, raw).map_err(|_| not_a_picture())?;
    let (mut width, mut height) = (0_u32, 0_u32);
    check(com!(
        frame.raw(),
        3,
        fn(*mut u32, *mut u32),
        &mut width,
        &mut height
    ))
    .map_err(|_| not_a_picture())?;
    picture_ok(width, height)?;
    let wide = png_depth(bytes) == Some(16);
    let format = if wide { RGBA16 } else { RGBA8 };
    let converter = convert(&factory, &frame, &format)?;
    let row = width as usize * if wide { 8 } else { 4 };
    let mut pixels = vec![0_u8; row * height as usize];
    check(com!(
        converter.raw(),
        7,
        fn(*const c_void, u32, u32, *mut u8),
        ptr::null(),
        u32::try_from(row).map_err(|_| too_large())?,
        u32::try_from(pixels.len()).map_err(|_| too_large())?,
        pixels.as_mut_ptr()
    ))
    .map_err(|_| not_a_picture())?;
    let samples: Vec<u16> = if wide {
        // `64bppRGBA` is little-endian 16-bit samples.
        pixels
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_le_bytes(*pair))
            .collect()
    } else {
        pixels.iter().map(|byte| u16::from(*byte)).collect()
    };
    let (width, height, samples) =
        orient::upright(orientation(&frame), width, height, 4, &samples)?;
    let depth = if wide {
        BitDepth::Sixteen
    } else {
        BitDepth::Eight
    };
    StillImage::from_samples(width, height, ColorType::Rgba, depth, samples)
        .checked()
        .map_err(|_| CodecError::Malformed {
            at_bit: 0,
            what: "wic samples",
        })
}

/// One still to one file.
pub fn encode_image(codec: CodecId, image: &StillImage) -> Result<Vec<u8>, CodecError> {
    let (width, height, wide, rgba) = still_rgba(image)?;
    picture_ok(width, height)?;
    let container = match codec {
        CodecId::Png => CONTAINER_PNG,
        CodecId::Jpeg => CONTAINER_JPEG,
        _ => return Err(CodecError::Unsupported { what: "wic codec" }),
    };
    let factory = factory()?;
    let format = if wide { RGBA16 } else { RGBA8 };
    let row = width as usize * if wide { 8 } else { 4 };
    let mut raw = ptr::null_mut();
    let code = com!(
        factory.raw(),
        20,
        fn(u32, u32, *const Guid, u32, u32, *const u8, *mut *mut c_void),
        width,
        height,
        &format,
        u32::try_from(row).map_err(|_| too_large())?,
        u32::try_from(rgba.len()).map_err(|_| too_large())?,
        rgba.as_ptr(),
        &mut raw
    );
    let bitmap = owned(code, raw).map_err(|_| refused())?;
    let mut raw = ptr::null_mut();
    let code = com!(
        factory.raw(),
        8,
        fn(*const Guid, *const Guid, *mut *mut c_void),
        &container,
        ptr::null(),
        &mut raw
    );
    let encoder = owned(code, raw).map_err(|_| CodecError::Unsupported { what: "wic codec" })?;
    let stream = memory_stream(&[])?;
    check(com!(
        encoder.raw(),
        3,
        fn(*mut c_void, u32),
        stream.raw(),
        NO_CACHE
    ))
    .map_err(|_| refused())?;
    let (mut frame, mut bag) = (ptr::null_mut(), ptr::null_mut());
    let code = com!(
        encoder.raw(),
        10,
        fn(*mut *mut c_void, *mut *mut c_void),
        &mut frame,
        &mut bag
    );
    let bag = Com::from_raw(bag);
    let frame = owned(code, frame).map_err(|_| refused())?;
    check(com!(
        frame.raw(),
        3,
        fn(*mut c_void),
        bag.as_ref().map_or(ptr::null_mut(), Com::raw)
    ))
    .map_err(|_| refused())?;
    check(com!(frame.raw(), 4, fn(u32, u32), width, height)).map_err(|_| refused())?;
    // The encoder names the format it will take; JPEG has no alpha.
    let mut accepted = format;
    check(com!(frame.raw(), 6, fn(*mut Guid), &mut accepted)).map_err(|_| refused())?;
    let source = if accepted == format {
        bitmap
    } else {
        convert(&factory, &bitmap, &accepted)?
    };
    check(com!(
        frame.raw(),
        11,
        fn(*mut c_void, *const c_void),
        source.raw(),
        ptr::null()
    ))
    .map_err(|_| refused())?;
    check(com!(frame.raw(), 12, fn(),)).map_err(|_| refused())?;
    check(com!(encoder.raw(), 11, fn(),)).map_err(|_| refused())?;
    read_stream(&stream)
}

/// The WIC factory, with COM entered on this thread first: the free
/// functions are called from threads that never opened a codec.
fn factory() -> Result<Com, CodecError> {
    com::enter()?;
    let mut raw = ptr::null_mut();
    let code = unsafe {
        CoCreateInstance(
            &CLSID_WIC_FACTORY,
            ptr::null_mut(),
            1,
            &IID_WIC_FACTORY,
            &mut raw,
        )
    };
    owned(code, raw).map_err(|_| CodecError::Unsupported { what: "wic" })
}

#[link(name = "ole32")]
unsafe extern "system" {
    fn CoCreateInstance(
        clsid: *const Guid,
        outer: *mut c_void,
        context: u32,
        iid: *const Guid,
        out: *mut *mut c_void,
    ) -> Hresult;
}

/// `source` converted to `format` by an `IWICFormatConverter`.
fn convert(factory: &Com, source: &Com, format: &Guid) -> Result<Com, CodecError> {
    let mut raw = ptr::null_mut();
    let code = com!(factory.raw(), 10, fn(*mut *mut c_void), &mut raw);
    let converter = owned(code, raw).map_err(|_| refused())?;
    check(com!(
        converter.raw(),
        8,
        fn(*mut c_void, *const Guid, u32, *mut c_void, f64, u32),
        source.raw(),
        format,
        0,
        ptr::null_mut(),
        0.0,
        0
    ))
    .map_err(|_| CodecError::Unsupported {
        what: "wic pixel format",
    })?;
    Ok(converter)
}

/// The EXIF orientation of a frame, or upright when it has none.
fn orientation(frame: &Com) -> Orientation {
    let mut raw = ptr::null_mut();
    let code = com!(frame.raw(), 8, fn(*mut *mut c_void), &mut raw);
    let Ok(reader) = owned(code, raw) else {
        return Orientation::upright();
    };
    let mut tag = 1_u32;
    for query in ["/app1/ifd/{ushort=274}", "/ifd/{ushort=274}"] {
        let name: Vec<u16> = query.encode_utf16().chain([0]).collect();
        let mut value = PropVariant {
            kind: 0,
            reserved: [0; 3],
            value: 0,
            record: 0,
        };
        let code = com!(
            reader.raw(),
            5,
            fn(*const u16, *mut PropVariant),
            name.as_ptr(),
            &mut value
        );
        let found = code >= 0 && value.kind == VT_UI2;
        if found {
            tag = (value.value & 0xFFFF) as u32;
        }
        unsafe {
            let _ = PropVariantClear(&mut value);
        }
        if found {
            break;
        }
    }
    Orientation::from_tag(tag)
}

/// A PNG's bit depth from its header, or `None` for another format.
fn png_depth(bytes: &[u8]) -> Option<u8> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") && bytes.get(12..16) == Some(b"IHDR") {
        bytes.get(24).copied()
    } else {
        None
    }
}

/// A read-write memory `IStream` holding a copy of `bytes`.
fn memory_stream(bytes: &[u8]) -> Result<Com, CodecError> {
    let size = u32::try_from(bytes.len()).map_err(|_| too_large())?;
    let data = if bytes.is_empty() {
        ptr::null()
    } else {
        bytes.as_ptr()
    };
    Com::from_raw(unsafe { SHCreateMemStream(data, size) }).ok_or(CodecError::Unsupported {
        what: "memory stream",
    })
}

/// Everything written to a memory stream.
fn read_stream(stream: &Com) -> Result<Vec<u8>, CodecError> {
    const SET: u32 = 0;
    const END: u32 = 2;
    let mut size = 0_u64;
    check(com!(
        stream.raw(),
        5,
        fn(i64, u32, *mut u64),
        0,
        END,
        &mut size
    ))
    .map_err(|_| refused())?;
    check(com!(
        stream.raw(),
        5,
        fn(i64, u32, *mut u64),
        0,
        SET,
        ptr::null_mut()
    ))
    .map_err(|_| refused())?;
    let mut out = vec![0_u8; usize::try_from(size).map_err(|_| too_large())?];
    let mut read = 0_u32;
    check(com!(
        stream.raw(),
        3,
        fn(*mut u8, u32, *mut u32),
        out.as_mut_ptr(),
        u32::try_from(out.len()).map_err(|_| too_large())?,
        &mut read
    ))
    .map_err(|_| refused())?;
    out.truncate(read as usize);
    Ok(out)
}

/// Straight RGBA bytes for WIC, and whether they are 16 bits a sample.
fn still_rgba(image: &StillImage) -> Result<(u32, u32, bool, Vec<u8>), CodecError> {
    image.validate().map_err(|_| CodecError::InvalidApi {
        what: "still image",
    })?;
    if image.width == 0 || image.height == 0 {
        return Err(CodecError::InvalidApi {
            what: "empty picture",
        });
    }
    let wide = match image.depth {
        BitDepth::Eight => false,
        BitDepth::Sixteen => true,
        _ => {
            return Err(CodecError::Unsupported {
                what: "still depth",
            });
        }
    };
    let opaque = if wide { u16::MAX } else { 255 };
    let channels = match image.color {
        ColorType::Rgba => 4,
        ColorType::Rgb => 3,
        ColorType::GrayAlpha => 2,
        ColorType::Gray => 1,
        ColorType::Indexed => {
            return Err(CodecError::Unsupported {
                what: "indexed still",
            });
        }
    };
    let pixels = (image.width as usize)
        .checked_mul(image.height as usize)
        .ok_or(too_large())?;
    if image.samples.len() != pixels.saturating_mul(channels) {
        return Err(CodecError::InvalidApi {
            what: "picture length",
        });
    }
    let mut rgba = Vec::with_capacity(pixels * 4);
    for pixel in image.samples.chunks_exact(channels) {
        let [r, g, b, a] = match pixel {
            [r, g, b, a] => [*r, *g, *b, *a],
            [r, g, b] => [*r, *g, *b, opaque],
            [gray, a] => [*gray, *gray, *gray, *a],
            [gray] => [*gray, *gray, *gray, opaque],
            _ => [0, 0, 0, opaque],
        };
        rgba.extend([r, g, b, a]);
    }
    let bytes = if wide {
        rgba.iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect()
    } else {
        rgba.iter()
            .map(|sample| u8::try_from(*sample).map_err(|_| CodecError::OutOfRange))
            .collect::<Result<Vec<u8>, _>>()?
    };
    Ok((image.width, image.height, wide, bytes))
}

fn owned(code: Hresult, raw: *mut c_void) -> Result<Com, Hresult> {
    let object = Com::from_raw(raw);
    if code < 0 {
        return Err(code);
    }
    object.ok_or(code)
}

fn check(code: Hresult) -> Result<(), Hresult> {
    if code < 0 { Err(code) } else { Ok(()) }
}

fn not_a_picture() -> CodecError {
    CodecError::Malformed {
        at_bit: 0,
        what: "wic",
    }
}

fn refused() -> CodecError {
    CodecError::Malformed {
        at_bit: 0,
        what: "wic encode",
    }
}

fn too_large() -> CodecError {
    CodecError::Limit {
        what: "picture bytes",
    }
}
