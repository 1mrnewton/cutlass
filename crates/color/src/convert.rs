//! NV12 and P010 to opaque RGBA8.

use cutlass_core::{ColorRange, PixelFormat};

use crate::{Error, Matrix};

/// One YUV picture. Chroma is interleaved Cb, Cr, one pair per 2×2 luma block.
pub struct YuvPlanes<'a> {
    pub format: PixelFormat,
    pub range: ColorRange,
    pub matrix: Matrix,
    pub width: u32,
    pub height: u32,
    pub y: &'a [u8],
    pub y_stride: usize,
    pub chroma: &'a [u8],
    pub chroma_stride: usize,
}

/// The weights shared by the CPU conversion and the GPU shader.
///
/// A GPU sample is an unorm. The code is `sample * code_scale / code_divisor`.
/// For P010 the 10-bit code sits in the top 10 bits, so the divisor is 64.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coefficients {
    pub y_offset: f32,
    pub y_scale: f32,
    pub chroma_offset: f32,
    pub chroma_scale: f32,
    pub kr: f32,
    pub kb: f32,
    pub code_scale: f32,
    pub code_divisor: f32,
}

impl Coefficients {
    pub fn new(format: PixelFormat, range: ColorRange, matrix: Matrix) -> Self {
        let (y_offset, y_scale, chroma_offset, chroma_scale, code_scale, code_divisor) =
            match (format, range) {
                (PixelFormat::Nv12, ColorRange::Limited) => {
                    (16.0, 1.0 / 219.0, 128.0, 1.0 / 224.0, 255.0, 1.0)
                }
                (PixelFormat::Nv12, ColorRange::Full) => {
                    (0.0, 1.0 / 255.0, 128.0, 1.0 / 255.0, 255.0, 1.0)
                }
                (PixelFormat::P010, ColorRange::Limited) => {
                    (64.0, 1.0 / 876.0, 512.0, 1.0 / 896.0, 65535.0, 64.0)
                }
                (PixelFormat::P010, ColorRange::Full) => {
                    (0.0, 1.0 / 1023.0, 512.0, 1.0 / 1023.0, 65535.0, 64.0)
                }
            };
        Self {
            y_offset,
            y_scale,
            chroma_offset,
            chroma_scale,
            kr: matrix.kr(),
            kb: matrix.kb(),
            code_scale,
            code_divisor,
        }
    }

    /// Eight floats in the order the shader reads them.
    pub fn gpu_values(self) -> [f32; 8] {
        [
            self.y_offset,
            self.y_scale,
            self.chroma_offset,
            self.chroma_scale,
            self.kr,
            self.kb,
            self.code_scale,
            self.code_divisor,
        ]
    }
}

/// Converts the picture to opaque RGBA8, one byte per channel, top row first.
///
/// Each chroma pair covers the 2×2 luma block whose top-left pixel is even.
pub fn to_rgba(planes: &YuvPlanes<'_>) -> Result<Vec<u8>, Error> {
    let width = usize::try_from(planes.width).map_err(|_| Error::TooLarge)?;
    let height = usize::try_from(planes.height).map_err(|_| Error::TooLarge)?;
    if width == 0 || height == 0 {
        return Err(Error::ZeroDimension);
    }
    if width % 2 != 0 || height % 2 != 0 {
        return Err(Error::OddDimension);
    }
    let (sample_bytes, chroma_bytes) = match planes.format {
        PixelFormat::Nv12 => (1, 2),
        PixelFormat::P010 => (2, 4),
    };
    let y_row = width.checked_mul(sample_bytes).ok_or(Error::TooLarge)?;
    let chroma_row = (width / 2)
        .checked_mul(chroma_bytes)
        .ok_or(Error::TooLarge)?;
    if !covers(planes.y, planes.y_stride, y_row, height) {
        return Err(Error::ShortPlane("luma"));
    }
    if !covers(planes.chroma, planes.chroma_stride, chroma_row, height / 2) {
        return Err(Error::ShortPlane("chroma"));
    }
    let pixels = width.checked_mul(height).ok_or(Error::TooLarge)?;
    let mut out = vec![0; pixels.checked_mul(4).ok_or(Error::TooLarge)?];
    let coefficients = Coefficients::new(planes.format, planes.range, planes.matrix);
    for row in 0..height {
        for column in 0..width {
            let y_code = luma_code(planes, column, row);
            let (cb, cr) = chroma_code(planes, column / 2, row / 2);
            let (red, green, blue) = rgb(coefficients, y_code, cb, cr);
            let index = (row * width + column) * 4;
            out[index] = byte(red);
            out[index + 1] = byte(green);
            out[index + 2] = byte(blue);
            out[index + 3] = 255;
        }
    }
    Ok(out)
}

fn covers(data: &[u8], stride: usize, row_bytes: usize, rows: usize) -> bool {
    if stride < row_bytes || rows == 0 || row_bytes == 0 {
        return false;
    }
    let Some(last_row) = stride.checked_mul(rows - 1) else {
        return false;
    };
    let Some(last) = last_row.checked_add(row_bytes - 1) else {
        return false;
    };
    last < data.len()
}

fn luma_code(planes: &YuvPlanes<'_>, x: usize, y: usize) -> f32 {
    match planes.format {
        PixelFormat::Nv12 => f32::from(planes.y[y * planes.y_stride + x]),
        PixelFormat::P010 => {
            let index = y * planes.y_stride + x * 2;
            f32::from(u16::from_le_bytes([planes.y[index], planes.y[index + 1]]) >> 6)
        }
    }
}

fn chroma_code(planes: &YuvPlanes<'_>, x: usize, y: usize) -> (f32, f32) {
    match planes.format {
        PixelFormat::Nv12 => {
            let index = y * planes.chroma_stride + x * 2;
            (
                f32::from(planes.chroma[index]),
                f32::from(planes.chroma[index + 1]),
            )
        }
        PixelFormat::P010 => {
            let index = y * planes.chroma_stride + x * 4;
            let cb = u16::from_le_bytes([planes.chroma[index], planes.chroma[index + 1]]) >> 6;
            let cr = u16::from_le_bytes([planes.chroma[index + 2], planes.chroma[index + 3]]) >> 6;
            (f32::from(cb), f32::from(cr))
        }
    }
}

fn rgb(coefficients: Coefficients, y_code: f32, cb_code: f32, cr_code: f32) -> (f32, f32, f32) {
    let y = (y_code - coefficients.y_offset) * coefficients.y_scale;
    let cb = (cb_code - coefficients.chroma_offset) * coefficients.chroma_scale;
    let cr = (cr_code - coefficients.chroma_offset) * coefficients.chroma_scale;
    let kg = 1.0 - coefficients.kr - coefficients.kb;
    let red = y + 2.0 * (1.0 - coefficients.kr) * cr;
    let green = y
        - 2.0 * (1.0 - coefficients.kb) * coefficients.kb / kg * cb
        - 2.0 * (1.0 - coefficients.kr) * coefficients.kr / kg * cr;
    let blue = y + 2.0 * (1.0 - coefficients.kb) * cb;
    (red, green, blue)
}

fn byte(channel: f32) -> u8 {
    let scaled = channel.clamp(0.0, 1.0) * 255.0;
    let rounded = scaled.round();
    // `round` of a value in `0.0..=255.0` is an integer in that range.
    u8::try_from(rounded as u16).expect("a rounded channel fits in a byte")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nv12(matrix: Matrix, range: ColorRange, y: u8, cb: u8, cr: u8) -> Vec<u8> {
        let luma = [y, y, y, y];
        let chroma = [cb, cr];
        to_rgba(&YuvPlanes {
            format: PixelFormat::Nv12,
            range,
            matrix,
            width: 2,
            height: 2,
            y: &luma,
            y_stride: 2,
            chroma: &chroma,
            chroma_stride: 2,
        })
        .expect("2x2 frame")
    }

    #[test]
    fn limited_range_black_and_white_ignore_the_matrix() {
        for matrix in [Matrix::Bt601, Matrix::Bt709, Matrix::Bt2020] {
            let black = nv12(matrix, ColorRange::Limited, 16, 128, 128);
            let white = nv12(matrix, ColorRange::Limited, 235, 128, 128);
            assert_eq!(&black[..4], &[0, 0, 0, 255]);
            assert_eq!(&white[..4], &[255, 255, 255, 255]);
        }
    }

    #[test]
    fn full_range_uses_the_whole_byte() {
        let black = nv12(Matrix::Bt709, ColorRange::Full, 0, 128, 128);
        let white = nv12(Matrix::Bt709, ColorRange::Full, 255, 128, 128);
        assert_eq!(&black[..4], &[0, 0, 0, 255]);
        assert_eq!(&white[..4], &[255, 255, 255, 255]);
    }

    #[test]
    fn the_matrix_changes_a_saturated_colour() {
        let red_709 = nv12(Matrix::Bt709, ColorRange::Limited, 81, 90, 240);
        let red_601 = nv12(Matrix::Bt601, ColorRange::Limited, 81, 90, 240);
        let red_2020 = nv12(Matrix::Bt2020, ColorRange::Limited, 81, 90, 240);
        assert_eq!(&red_709[..4], &[255, 24, 0, 255]);
        assert_eq!(&red_601[..4], &[254, 0, 0, 255]);
        assert_eq!(&red_2020[..4], &[255, 10, 0, 255]);
    }

    #[test]
    fn each_luma_sample_keeps_its_place() {
        let y = [16u8, 235, 235, 16];
        let chroma = [128u8, 128];
        let rgba = to_rgba(&YuvPlanes {
            format: PixelFormat::Nv12,
            range: ColorRange::Limited,
            matrix: Matrix::Bt709,
            width: 2,
            height: 2,
            y: &y,
            y_stride: 2,
            chroma: &chroma,
            chroma_stride: 2,
        })
        .expect("placed samples");
        assert_eq!(&rgba[0..4], &[0, 0, 0, 255]);
        assert_eq!(&rgba[4..8], &[255, 255, 255, 255]);
        assert_eq!(&rgba[8..12], &[255, 255, 255, 255]);
        assert_eq!(&rgba[12..16], &[0, 0, 0, 255]);
    }

    #[test]
    fn padding_past_the_visible_row_is_ignored() {
        let y = [16u8, 16, 0xff, 16, 16, 0xff];
        let chroma = [128u8, 128, 0xff];
        let rgba = to_rgba(&YuvPlanes {
            format: PixelFormat::Nv12,
            range: ColorRange::Limited,
            matrix: Matrix::Bt601,
            width: 2,
            height: 2,
            y: &y,
            y_stride: 3,
            chroma: &chroma,
            chroma_stride: 3,
        })
        .expect("padded frame");
        assert!(rgba.chunks(4).all(|pixel| pixel == [0, 0, 0, 255]));
    }

    #[test]
    fn p010_limited_black_is_black() {
        let y_sample = (64u16 << 6).to_le_bytes();
        let c_sample = (512u16 << 6).to_le_bytes();
        let mut y = Vec::new();
        let mut chroma = Vec::new();
        for _ in 0..4 {
            y.extend(y_sample);
        }
        chroma.extend(c_sample);
        chroma.extend(c_sample);
        let rgba = to_rgba(&YuvPlanes {
            format: PixelFormat::P010,
            range: ColorRange::Limited,
            matrix: Matrix::Bt709,
            width: 2,
            height: 2,
            y: &y,
            y_stride: 4,
            chroma: &chroma,
            chroma_stride: 4,
        })
        .expect("p010 frame");
        assert_eq!(&rgba[..4], &[0, 0, 0, 255]);
    }

    #[test]
    fn a_short_or_odd_frame_is_rejected() {
        let err = to_rgba(&YuvPlanes {
            format: PixelFormat::Nv12,
            range: ColorRange::Limited,
            matrix: Matrix::Bt709,
            width: 3,
            height: 2,
            y: &[0; 6],
            y_stride: 3,
            chroma: &[128, 128],
            chroma_stride: 2,
        });
        assert_eq!(err, Err(Error::OddDimension));
        let err = to_rgba(&YuvPlanes {
            format: PixelFormat::Nv12,
            range: ColorRange::Limited,
            matrix: Matrix::Bt709,
            width: 2,
            height: 2,
            y: &[16, 16],
            y_stride: 2,
            chroma: &[128, 128],
            chroma_stride: 2,
        });
        assert_eq!(err, Err(Error::ShortPlane("luma")));
    }
}
