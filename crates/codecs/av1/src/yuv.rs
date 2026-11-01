//! Tightly packed 8-bit 4:2:0 used inside this codec.
//!
//! Chroma edges are the luma edges rounded up, then halved, so an odd luma
//! size stays representable. The shared picture output is even 4:2:0, or one
//! gray plane when the sequence is monochrome.
//! Conversion happens at the trait boundary.

use cutlass_decode::{
    CodecError, ColorPrimaries, Output, PictureInfo, PictureOutput, PictureSamples, PixelFormat,
    Transfer,
};

/// One tightly packed sample plane. `samples.len()` is `width * height`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plane {
    width: u32,
    height: u32,
    samples: Vec<u8>,
}

impl Plane {
    /// Rejects an empty edge and a buffer whose length is not the pixel count.
    pub fn new(width: u32, height: u32, samples: Vec<u8>) -> Result<Self, CodecError> {
        if width == 0 || height == 0 {
            return Err(CodecError::InvalidApi {
                what: "plane edge is zero",
            });
        }
        let len = (width as usize)
            .checked_mul(height as usize)
            .ok_or(CodecError::Limit {
                what: "plane byte length",
            })?;
        if samples.len() != len {
            return Err(CodecError::InvalidApi {
                what: "plane byte length",
            });
        }
        Ok(Self {
            width,
            height,
            samples,
        })
    }

    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[must_use]
    pub fn samples(&self) -> &[u8] {
        &self.samples
    }

    #[must_use]
    pub fn samples_mut(&mut self) -> &mut [u8] {
        &mut self.samples
    }
}

/// 8-bit 4:2:0. Chroma edges are the luma edges rounded up to a multiple of two, then halved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Yuv420 {
    pub y: Plane,
    pub u: Plane,
    pub v: Plane,
}

impl Yuv420 {
    pub fn new(y: Plane, u: Plane, v: Plane) -> Result<Self, CodecError> {
        let chroma_w = y.width().div_ceil(2);
        let chroma_h = y.height().div_ceil(2);
        if u.width() != chroma_w
            || v.width() != chroma_w
            || u.height() != chroma_h
            || v.height() != chroma_h
        {
            return Err(CodecError::InvalidApi {
                what: "4:2:0 chroma size",
            });
        }
        Ok(Self { y, u, v })
    }

    /// Chroma edges follow the sequence subsampling flags. Both flags set is 4:2:0.
    /// Both clear is 4:4:4.
    pub fn subsampled(
        y: Plane,
        u: Plane,
        v: Plane,
        sub_x: bool,
        sub_y: bool,
    ) -> Result<Self, CodecError> {
        let chroma_w = if sub_x {
            y.width().div_ceil(2)
        } else {
            y.width()
        };
        let chroma_h = if sub_y {
            y.height().div_ceil(2)
        } else {
            y.height()
        };
        if u.width() != chroma_w
            || v.width() != chroma_w
            || u.height() != chroma_h
            || v.height() != chroma_h
        {
            return Err(CodecError::InvalidApi {
                what: "chroma size",
            });
        }
        Ok(Self { y, u, v })
    }
}

/// One picture presented to this encoder, beside the shared still and frame inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoInput {
    pub picture: Yuv420,
    pub pts: i64,
    pub duration: i64,
    pub timescale: u32,
}

/// A decoded picture split back into planes for a caller that still indexes them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct View {
    pub picture: Yuv420,
}

/// Read a shared picture output as this codec's planes.
pub trait AsView {
    fn as_video(&self) -> Option<View>;
}

impl AsView for Output {
    fn as_video(&self) -> Option<View> {
        let Output::Picture(picture) = self else {
            return None;
        };
        let info = picture.info();
        if info.pixel != PixelFormat::Yuv420p {
            return None;
        }
        let PictureSamples::Cpu(bytes) = picture.samples() else {
            return None;
        };
        let width = info.width;
        let height = info.height;
        if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return None;
        }
        let y_len = usize::try_from(width)
            .ok()?
            .checked_mul(usize::try_from(height).ok()?)?;
        let chroma_w = width / 2;
        let chroma_h = height / 2;
        let chroma_len = usize::try_from(chroma_w)
            .ok()?
            .checked_mul(usize::try_from(chroma_h).ok()?)?;
        if bytes.len() != y_len.checked_add(chroma_len)?.checked_add(chroma_len)? {
            return None;
        }
        let y = bytes[..y_len].to_vec();
        let u = bytes[y_len..y_len + chroma_len].to_vec();
        let v = bytes[y_len + chroma_len..].to_vec();
        let picture = Yuv420::new(
            Plane::new(width, height, y).ok()?,
            Plane::new(chroma_w, chroma_h, u).ok()?,
            Plane::new(chroma_w, chroma_h, v).ok()?,
        )
        .ok()?;
        Some(View { picture })
    }
}

/// Pack the luma plane. A monochrome sequence has no chroma samples to publish.
pub fn gray_output(
    picture: &Yuv420,
    pts: i64,
    timescale: u32,
) -> Result<PictureOutput, CodecError> {
    PictureOutput::cpu(
        PictureInfo {
            width: picture.y.width(),
            height: picture.y.height(),
            pixel: PixelFormat::Gray8,
            primaries: ColorPrimaries::Unspecified,
            transfer: Transfer::Unspecified,
        },
        picture.y.samples().to_vec(),
        pts,
        timescale,
    )
}

/// Pack one 10-bit luma plane. A monochrome sequence has no chroma samples.
pub fn gray_wide(
    y: &[u16],
    width: u32,
    height: u32,
    pts: i64,
    timescale: u32,
) -> Result<PictureOutput, CodecError> {
    let mut bytes = Vec::new();
    pack_le(y, &mut bytes);
    PictureOutput::cpu(
        PictureInfo {
            width,
            height,
            pixel: PixelFormat::Gray10,
            primaries: ColorPrimaries::Unspecified,
            transfer: Transfer::Unspecified,
        },
        bytes,
        pts,
        timescale,
    )
}

/// Pack 10-bit 4:2:0. Each sample is a little-endian `u16` in the low 10 bits.
pub fn wide_output(
    y: &[u16],
    u: &[u16],
    v: &[u16],
    width: u32,
    height: u32,
    pts: i64,
    timescale: u32,
) -> Result<PictureOutput, CodecError> {
    let mut bytes = Vec::new();
    pack_le(y, &mut bytes);
    pack_le(u, &mut bytes);
    pack_le(v, &mut bytes);
    let luma = usize::try_from(width)
        .unwrap_or(0)
        .saturating_mul(usize::try_from(height).unwrap_or(0));
    let pixel = if u.len() == luma && v.len() == luma {
        PixelFormat::Yuv444p10
    } else {
        PixelFormat::Yuv420p10
    };
    PictureOutput::cpu(
        PictureInfo {
            width,
            height,
            pixel,
            primaries: ColorPrimaries::Unspecified,
            transfer: Transfer::Unspecified,
        },
        bytes,
        pts,
        timescale,
    )
}

fn pack_le(samples: &[u16], out: &mut Vec<u8>) {
    out.reserve(samples.len().saturating_mul(2));
    for sample in samples {
        out.extend_from_slice(&sample.to_le_bytes());
    }
}

/// Pack even 4:2:0 into the shared picture output.
pub fn picture_output(
    picture: &Yuv420,
    pts: i64,
    timescale: u32,
) -> Result<PictureOutput, CodecError> {
    let width = picture.y.width();
    let height = picture.y.height();
    let mut bytes = Vec::with_capacity(
        picture.y.samples().len() + picture.u.samples().len() + picture.v.samples().len(),
    );
    bytes.extend_from_slice(picture.y.samples());
    bytes.extend_from_slice(picture.u.samples());
    bytes.extend_from_slice(picture.v.samples());
    PictureOutput::cpu(
        PictureInfo {
            width,
            height,
            pixel: if picture.u.width() == width && picture.u.height() == height {
                PixelFormat::Yuv444p
            } else {
                PixelFormat::Yuv420p
            },
            primaries: ColorPrimaries::Unspecified,
            transfer: Transfer::Unspecified,
        },
        bytes,
        pts,
        timescale,
    )
}

#[cfg(test)]
mod tests {
    use super::{Plane, Yuv420};
    use cutlass_decode::CodecError;

    #[test]
    fn a_plane_keeps_a_matching_buffer() {
        let plane = Plane::new(2, 2, vec![1, 2, 3, 4]).expect("plane");
        assert_eq!(plane.samples(), &[1, 2, 3, 4]);
    }

    #[test]
    fn a_short_plane_is_refused() {
        let err = Plane::new(2, 2, vec![1, 2, 3]).expect_err("short");
        assert_eq!(
            err,
            CodecError::InvalidApi {
                what: "plane byte length",
            }
        );
    }

    #[test]
    fn yuv420_requires_half_resolution_chroma() {
        let y = Plane::new(4, 2, vec![0; 8]).expect("y");
        let u = Plane::new(2, 1, vec![128, 128]).expect("u");
        let v = Plane::new(2, 1, vec![128, 128]).expect("v");
        assert!(Yuv420::new(y, u, v).is_ok());
    }
}
