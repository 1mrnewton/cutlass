//! One decoded bitstream as straight 8-bit RGBA.

use cutlass_decode::CodecError;

use crate::lossless;
use crate::lossy;
use crate::riff::Bitstream;

/// A decoded still or frame, straight (not premultiplied) RGBA.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Picture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    /// The bitstream carries alpha: a `VP8L` alpha hint or a non-opaque
    /// pixel, or an `ALPH` chunk.
    pub alpha: bool,
}

/// Decode one coded picture.
pub(crate) fn decode(bitstream: Bitstream<'_>) -> Result<Picture, CodecError> {
    match bitstream {
        Bitstream::Lossy { vp8, alpha } => {
            if alpha.is_some() {
                return Err(CodecError::Unsupported { what: "webp alpha" });
            }
            let planes = lossy::decode(vp8)?;
            Ok(Picture {
                width: planes.width as u32,
                height: planes.height as u32,
                rgba: lossy::yuv::to_rgba(&planes),
                alpha: false,
            })
        }
        Bitstream::Lossless(data) => {
            let (header, argb) = lossless::decode::decode(data)?;
            let mut rgba = Vec::with_capacity(argb.len() * 4);
            let mut opaque = true;
            for pixel in argb {
                let [blue, green, red, alpha] = pixel.to_le_bytes();
                opaque &= alpha == 0xff;
                rgba.extend_from_slice(&[red, green, blue, alpha]);
            }
            Ok(Picture {
                width: header.width,
                height: header.height,
                rgba,
                alpha: header.alpha || !opaque,
            })
        }
    }
}
