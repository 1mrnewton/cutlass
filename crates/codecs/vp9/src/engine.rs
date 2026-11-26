//! The VP9 decoder on the engine's codec trait.
//!
//! Playback opens a [`Decoder`] through [`DecodeFactory::open`] and takes
//! packed pictures: profiles 0 and 1 as 8-bit planes, profiles 2 and 3 as
//! little-endian 16-bit ones, in the stream's own chroma layout. Export
//! opens the encoder [`crate::register_engine_encoders`] registers.
//!
//! The decoding is [`Vp9Decoder`]'s: a packet is one frame or a
//! superframe, a hidden frame gives no picture, and frames are
//! reconstructed on worker threads while the caller parses the next, so a
//! picture may come out of a later push or of [`Decoder::drain`].

use cutlass_decode::video::{VideoDecoder as _, VideoOutput};
use cutlass_decode::{
    CodecError, CodecId, CodecParams, ColorMatrix, ColorPrimaries, ColorRange, DecodeFactory,
    Decoder, MatrixCoefficients, Output, Packet, PictureInfo, PictureOutput, PixelFormat, Plane,
    Transfer, picture_byte_len,
};

use crate::decode::Vp9Decoder;

/// Registers the VP9 decoder that playback opens.
pub fn register_engine_decoders(factory: &mut DecodeFactory) {
    factory.register(CodecId::Vp9, open_engine_decoder);
}

fn open_engine_decoder(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    Ok(Box::new(EngineDecoder::open(params)?))
}

/// VP9 for playback: one packed picture per shown frame.
///
/// The picture's format, matrix and range are the frame header's. A VP9
/// header names one colour space and no transfer curve, so the primaries
/// and transfer are the container's when it names them (the parameters'
/// own fields, or a `vpcC` payload in `params.extra`) and follow the
/// header's colour space otherwise. The engine's 4:2:0 pictures have even
/// edges, so an odd-sized 4:2:0 stream repeats its last column or row; the
/// render size stays the header's.
///
/// The engine's picture carries no alpha plane: a WebM alpha stream
/// (`SideData::BlockAdditional` 1) is decoded by [`Vp9Decoder`] and not
/// passed on.
pub struct EngineDecoder {
    inner: Vp9Decoder,
    container: ContainerColour,
}

/// The colour a container names, each with its ISO/IEC 23091-2 code.
#[derive(Clone, Copy, Default)]
struct ContainerColour {
    primaries: Option<(ColorPrimaries, u8)>,
    transfer: Option<(Transfer, u8)>,
    matrix: Option<(ColorMatrix, u8)>,
}

impl EngineDecoder {
    /// Opens with the container's parameters, on as many threads as
    /// [`Vp9Decoder::open`] takes (one per core, at most 16).
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        let stream = stream_params(params)?;
        Ok(Self {
            inner: Vp9Decoder::open(&stream)?,
            container: ContainerColour::of(params),
        })
    }

    /// [`Self::open`] on `threads` worker threads (0 or 1: everything on
    /// the caller, and each shown frame's picture out of its own push).
    pub fn open_with_threads(params: &CodecParams, threads: usize) -> Result<Self, CodecError> {
        let stream = stream_params(params)?;
        Ok(Self {
            inner: Vp9Decoder::open_with_threads(&stream, threads)?,
            container: ContainerColour::of(params),
        })
    }

    fn outputs(&self, frames: Vec<VideoOutput>) -> Result<Vec<Output>, CodecError> {
        frames
            .iter()
            .map(|frame| Ok(Output::Picture(self.picture(frame)?)))
            .collect()
    }

    fn picture(&self, output: &VideoOutput) -> Result<PictureOutput, CodecError> {
        let frame = &output.frame;
        let pixel = frame.format;
        let sample = sample_bytes(pixel)?;
        let (mut width, mut height) = (frame.width, frame.height);
        if matches!(
            pixel,
            PixelFormat::Yuv420p | PixelFormat::Yuv420p10 | PixelFormat::Yuv420p12
        ) {
            width = width.checked_add(width & 1).ok_or(EDGE)?;
            height = height.checked_add(height & 1).ok_or(EDGE)?;
        }
        let (chroma_w, chroma_h) = pixel.chroma_wh(width, height).ok_or(FORMAT)?;
        let mut bytes = Vec::with_capacity(picture_byte_len(width, height, pixel)?);
        append_plane(&mut bytes, &frame.y, sample, width, height)?;
        append_plane(&mut bytes, &frame.u, sample, chroma_w, chroma_h)?;
        append_plane(&mut bytes, &frame.v, sample, chroma_w, chroma_h)?;

        let (primaries, cicp_primaries) = self
            .container
            .primaries
            .unwrap_or((frame.primaries, frame.primaries.cicp()));
        let (transfer, cicp_transfer) = self
            .container
            .transfer
            .unwrap_or((frame.transfer, frame.transfer.cicp()));
        // The header's matrix, and the container's where the header says
        // the colour space is unknown.
        let (matrix, cicp_matrix) = match (frame.matrix, self.container.matrix) {
            (MatrixCoefficients::Unspecified, Some(named)) => named,
            (header, _) => (header_matrix(header), header.cicp()),
        };
        let info = PictureInfo {
            matrix,
            render_width: frame.render_width,
            render_height: frame.render_height,
            cicp_primaries,
            cicp_transfer,
            cicp_matrix,
            range: match frame.range {
                ColorRange::Full => ColorRange::Full,
                ColorRange::Unspecified => ColorRange::Unspecified,
                ColorRange::Limited | ColorRange::Studio => ColorRange::Limited,
            },
            ..PictureInfo::new(width, height, pixel, primaries, transfer)
        };
        PictureOutput::cpu(info, bytes, output.pts, output.timescale)
    }
}

const EDGE: CodecError = CodecError::Limit {
    what: "picture edge",
};
const FORMAT: CodecError = CodecError::Unsupported {
    what: "vp9 pixel format",
};
const PLANES: CodecError = CodecError::InvalidApi {
    what: "picture planes",
};

/// What [`Vp9Decoder`] opens with. The frame headers carry the size and
/// the format, so a container that names neither (or a format this codec
/// has none of) still opens: a zero edge passes as one sample, and the
/// format as 4:2:0. An edge past the largest VP9 level is refused here,
/// as it is on each frame.
fn stream_params(params: &CodecParams) -> Result<CodecParams, CodecError> {
    if params.codec != CodecId::Vp9 {
        return Err(CodecError::Unsupported { what: "codec" });
    }
    Ok(CodecParams::video(
        CodecId::Vp9,
        params.width.max(1),
        params.height.max(1),
    ))
}

impl ContainerColour {
    fn of(params: &CodecParams) -> Self {
        let mut colour = vpcc_colour(&params.extra);
        if params.primaries != ColorPrimaries::Unspecified {
            colour.primaries = Some((params.primaries, params.primaries.cicp()));
        }
        if params.transfer != Transfer::Unspecified {
            colour.transfer = Some((params.transfer, params.transfer.cicp()));
        }
        if params.matrix != MatrixCoefficients::Unspecified {
            colour.matrix = Some((header_matrix(params.matrix), params.matrix.cicp()));
        }
        colour
    }
}

/// The colour of a `vpcC` payload (version 1, with its four bytes of
/// version and flags): byte 7 is the ISO/IEC 23091-2 colour primaries
/// code, byte 8 the transfer code, byte 9 the matrix code. Code 2, and a
/// code this crate has no name for, is no colour.
fn vpcc_colour(extra: &[u8]) -> ContainerColour {
    let (Some(&[1, 0, 0, 0]), Some(&[primaries, transfer, matrix])) =
        (extra.first_chunk::<4>(), extra.get(7..10))
    else {
        return ContainerColour::default();
    };
    ContainerColour {
        primaries: primaries_of(primaries).map(|named| (named, primaries)),
        transfer: transfer_of(transfer).map(|named| (named, transfer)),
        matrix: matrix_of(matrix).map(|named| (named, matrix)),
    }
}

fn primaries_of(code: u8) -> Option<ColorPrimaries> {
    Some(match code {
        1 => ColorPrimaries::Bt709,
        4 => ColorPrimaries::Bt470m,
        5 | 6 => ColorPrimaries::Bt601,
        7 => ColorPrimaries::Smpte240m,
        8 => ColorPrimaries::Film,
        9 => ColorPrimaries::Bt2020,
        10 => ColorPrimaries::Smpte428,
        11 => ColorPrimaries::Smpte431,
        12 => ColorPrimaries::Smpte432,
        22 => ColorPrimaries::Ebu3213,
        _ => return None,
    })
}

fn transfer_of(code: u8) -> Option<Transfer> {
    Some(match code {
        1 => Transfer::Bt709,
        6 => Transfer::Bt601,
        7 => Transfer::Smpte240m,
        8 => Transfer::Linear,
        9 => Transfer::Log100,
        10 => Transfer::LogSqrt,
        11 => Transfer::Iec61966_2_4,
        12 => Transfer::Bt1361,
        13 => Transfer::Srgb,
        14 | 15 => Transfer::Bt2020,
        16 => Transfer::Pq,
        17 => Transfer::Smpte428,
        18 => Transfer::Hlg,
        _ => return None,
    })
}

fn matrix_of(code: u8) -> Option<ColorMatrix> {
    Some(match code {
        0 => ColorMatrix::Identity,
        1 => ColorMatrix::Bt709,
        4 => ColorMatrix::Fcc,
        5 | 6 => ColorMatrix::Bt601,
        7 => ColorMatrix::Smpte240m,
        8 => ColorMatrix::Ycgco,
        9 | 10 => ColorMatrix::Bt2020,
        _ => return None,
    })
}

fn header_matrix(matrix: MatrixCoefficients) -> ColorMatrix {
    match matrix {
        MatrixCoefficients::Rgb => ColorMatrix::Identity,
        MatrixCoefficients::Bt709 => ColorMatrix::Bt709,
        MatrixCoefficients::Bt601 => ColorMatrix::Bt601,
        MatrixCoefficients::Bt2020 => ColorMatrix::Bt2020,
        MatrixCoefficients::Unspecified => ColorMatrix::Unspecified,
    }
}

/// Bytes a sample of a format this decoder gives out.
fn sample_bytes(pixel: PixelFormat) -> Result<usize, CodecError> {
    match pixel {
        PixelFormat::Yuv420p
        | PixelFormat::Yuv422p
        | PixelFormat::Yuv440p
        | PixelFormat::Yuv444p
        | PixelFormat::Gbrp => Ok(1),
        PixelFormat::Yuv420p10
        | PixelFormat::Yuv420p12
        | PixelFormat::Yuv422p10
        | PixelFormat::Yuv422p12
        | PixelFormat::Yuv440p10
        | PixelFormat::Yuv440p12
        | PixelFormat::Yuv444p10
        | PixelFormat::Yuv444p12
        | PixelFormat::Gbrp10
        | PixelFormat::Gbrp12 => Ok(2),
        PixelFormat::None | PixelFormat::Rgba8 | PixelFormat::Gray8 | PixelFormat::Gray10 => {
            Err(FORMAT)
        }
    }
}

/// Appends `plane` tightly packed at `width` by `height`, each at least
/// the plane's own, repeating its last column and row past them.
fn append_plane(
    out: &mut Vec<u8>,
    plane: &Plane,
    sample: usize,
    width: u32,
    height: u32,
) -> Result<(), CodecError> {
    let (width, height) = (width as usize, height as usize);
    let (own_w, own_h) = (plane.width as usize, plane.height as usize);
    if own_w == 0 || own_h == 0 || own_w > width || own_h > height {
        return Err(PLANES);
    }
    let row_bytes = own_w * sample;
    if own_w == width && own_h == height && plane.stride == row_bytes {
        out.extend_from_slice(plane.samples.get(..row_bytes * height).ok_or(PLANES)?);
        return Ok(());
    }
    for row in 0..height {
        let at = row.min(own_h - 1) * plane.stride;
        let samples = plane.samples.get(at..at + row_bytes).ok_or(PLANES)?;
        out.extend_from_slice(samples);
        if let Some(last) = samples.get(row_bytes - sample..) {
            for _ in own_w..width {
                out.extend_from_slice(last);
            }
        }
    }
    Ok(())
}

impl Decoder for EngineDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Vp9
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        if packet.data.is_empty() {
            return Ok(Vec::new());
        }
        let frames = self.inner.push(packet)?;
        self.outputs(frames)
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        let frames = self.inner.drain()?;
        self.outputs(frames)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

#[cfg(test)]
#[path = "engine_tests.rs"]
mod tests;
