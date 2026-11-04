//! NV12 in and out of the pictures this workspace passes around.
//!
//! The Media Foundation H.264 transforms take and give NV12: a full-size Y
//! plane, then one plane of interleaved U and V at half size. The caller
//! has tightly packed RGBA or planar 4:2:0. The colour math is the BT.709
//! limited-range fixed point that `cutlass-encode` uses for its Rust path.

#![cfg_attr(not(target_os = "windows"), allow(dead_code))]

use cutlass_decode::CodecError;

/// Where one NV12 picture sits inside a transform's buffer.
///
/// `stride` is the byte step between rows of both planes. `rows` is the
/// height of the Y plane in the buffer, which a decoder pads to 16. The
/// picture is the top-left `width` by `height`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub width: u32,
    pub height: u32,
    pub stride: usize,
    pub rows: usize,
}

impl Layout {
    /// A tightly packed picture.
    pub fn packed(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            stride: width as usize,
            rows: height as usize,
        }
    }

    /// Bytes the buffer must hold for this layout.
    pub fn len(&self) -> Result<usize, CodecError> {
        self.check()?;
        let luma = self.stride.checked_mul(self.rows).ok_or(LIMIT)?;
        let chroma = self
            .stride
            .checked_mul(self.rows.div_ceil(2))
            .ok_or(LIMIT)?;
        luma.checked_add(chroma).ok_or(LIMIT)
    }

    fn check(&self) -> Result<(), CodecError> {
        if self.width == 0
            || self.height == 0
            || !self.width.is_multiple_of(2)
            || !self.height.is_multiple_of(2)
        {
            return Err(CodecError::InvalidApi { what: "nv12 size" });
        }
        if self.stride < self.width as usize || self.rows < self.height as usize {
            return Err(CodecError::InvalidApi {
                what: "nv12 stride",
            });
        }
        Ok(())
    }

    fn chroma_at(&self) -> usize {
        self.stride * self.rows
    }
}

const LIMIT: CodecError = CodecError::Limit {
    what: "picture bytes",
};

/// Tightly packed RGBA to packed NV12. The alpha byte is ignored.
pub fn from_rgba(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>, CodecError> {
    let layout = Layout::packed(width, height);
    let mut out = vec![0_u8; layout.len()?];
    let (w, h) = (width as usize, height as usize);
    if rgba.len() != w * h * 4 {
        return Err(CodecError::InvalidApi {
            what: "picture length",
        });
    }
    let (y, uv) = out.split_at_mut(w * h);
    for row in (0..h).step_by(2) {
        for col in (0..w).step_by(2) {
            let mut sum = [0_i32; 3];
            for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let at = ((row + dy) * w + col + dx) * 4;
                let rgb = [
                    i32::from(rgba[at]),
                    i32::from(rgba[at + 1]),
                    i32::from(rgba[at + 2]),
                ];
                y[(row + dy) * w + col + dx] = luma(rgb);
                for (total, value) in sum.iter_mut().zip(rgb) {
                    *total += value;
                }
            }
            let (cb, cr) = chroma(sum.map(|total| (total + 2) / 4));
            let at = (row / 2) * w + col;
            uv[at] = cb;
            uv[at + 1] = cr;
        }
    }
    Ok(out)
}

/// Planar Y, U, V to packed NV12.
pub fn from_planar(width: u32, height: u32, yuv: &[u8]) -> Result<Vec<u8>, CodecError> {
    let layout = Layout::packed(width, height);
    let len = layout.len()?;
    if yuv.len() != len {
        return Err(CodecError::InvalidApi {
            what: "picture length",
        });
    }
    let luma = width as usize * height as usize;
    let quarter = luma / 4;
    let mut out = Vec::with_capacity(len);
    out.extend_from_slice(&yuv[..luma]);
    let (u, v) = yuv[luma..].split_at(quarter);
    for (cb, cr) in u.iter().zip(v) {
        out.push(*cb);
        out.push(*cr);
    }
    Ok(out)
}

/// One NV12 picture in a transform buffer to tightly packed planar 4:2:0.
pub fn to_planar(layout: Layout, buffer: &[u8]) -> Result<Vec<u8>, CodecError> {
    if buffer.len() < layout.len()? {
        return Err(CodecError::Truncated {
            at_bit: (buffer.len() as u64).saturating_mul(8),
            what: "nv12 buffer",
        });
    }
    let (w, h) = (layout.width as usize, layout.height as usize);
    let mut out = Vec::with_capacity(w * h * 3 / 2);
    for row in 0..h {
        let at = row * layout.stride;
        out.extend_from_slice(&buffer[at..at + w]);
    }
    let chroma = layout.chroma_at();
    let mut v = Vec::with_capacity(w * h / 4);
    for row in 0..h / 2 {
        let at = chroma + row * layout.stride;
        for [cb, cr] in buffer[at..at + w].as_chunks::<2>().0 {
            out.push(*cb);
            v.push(*cr);
        }
    }
    out.extend_from_slice(&v);
    Ok(out)
}

/// One NV12 picture in a transform buffer to tightly packed opaque RGBA.
pub fn to_rgba(layout: Layout, buffer: &[u8]) -> Result<Vec<u8>, CodecError> {
    if buffer.len() < layout.len()? {
        return Err(CodecError::Truncated {
            at_bit: (buffer.len() as u64).saturating_mul(8),
            what: "nv12 buffer",
        });
    }
    let (w, h) = (layout.width as usize, layout.height as usize);
    let chroma = layout.chroma_at();
    let mut out = vec![0_u8; w * h * 4];
    for row in 0..h {
        let luma_row = row * layout.stride;
        let chroma_row = chroma + (row / 2) * layout.stride;
        for col in 0..w {
            let pair = chroma_row + (col & !1);
            let rgb = rgb(buffer[luma_row + col], buffer[pair], buffer[pair + 1]);
            let at = (row * w + col) * 4;
            out[at..at + 3].copy_from_slice(&rgb);
            out[at + 3] = 255;
        }
    }
    Ok(out)
}

/// BT.709 luma in 16..=235, in 8.8 fixed point.
fn luma([r, g, b]: [i32; 3]) -> u8 {
    clamp(16 + ((47 * r + 157 * g + 16 * b + 128) >> 8))
}

/// BT.709 Cb and Cr in 16..=240, in 8.8 fixed point.
fn chroma([r, g, b]: [i32; 3]) -> (u8, u8) {
    let cb = 128 + ((-26 * r - 87 * g + 113 * b + 128) >> 8);
    let cr = 128 + ((112 * r - 102 * g - 10 * b + 128) >> 8);
    (clamp(cb), clamp(cr))
}

/// Limited-range BT.709 back to full-range RGB, in 8.8 fixed point.
fn rgb(y: u8, cb: u8, cr: u8) -> [u8; 3] {
    let y = 298 * (i32::from(y) - 16);
    let cb = i32::from(cb) - 128;
    let cr = i32::from(cr) - 128;
    [
        clamp((y + 459 * cr + 128) >> 8),
        clamp((y - 55 * cb - 136 * cr + 128) >> 8),
        clamp((y + 541 * cb + 128) >> 8),
    ]
}

fn clamp(value: i32) -> u8 {
    u8::try_from(value.clamp(0, 255)).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_and_black_survive_the_round_trip() {
        for value in [0_u8, 255] {
            let rgba = [value, value, value, 255].repeat(4 * 2);
            let nv12 = from_rgba(4, 2, &rgba).unwrap();
            assert_eq!(to_rgba(Layout::packed(4, 2), &nv12).unwrap(), rgba);
        }
    }

    #[test]
    fn primaries_come_back_within_a_few_steps() {
        for colour in [[255_u8, 0, 0], [0, 255, 0], [0, 0, 255], [200, 120, 40]] {
            let rgba = [colour[0], colour[1], colour[2], 255].repeat(4);
            let back = to_rgba(Layout::packed(2, 2), &from_rgba(2, 2, &rgba).unwrap()).unwrap();
            for (got, want) in back.iter().zip(&rgba) {
                assert!(got.abs_diff(*want) <= 3, "{colour:?} came back as {back:?}");
            }
        }
    }

    #[test]
    fn planar_and_nv12_carry_the_same_samples() {
        let planar: Vec<u8> = (0..4 * 2 + 2 + 2).collect();
        let nv12 = from_planar(4, 2, &planar).unwrap();
        assert_eq!(nv12, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 9, 11]);
        assert_eq!(to_planar(Layout::packed(4, 2), &nv12).unwrap(), planar);
    }

    #[test]
    fn a_padded_buffer_is_cropped_to_the_picture() {
        let layout = Layout {
            width: 2,
            height: 2,
            stride: 4,
            rows: 4,
        };
        let mut buffer = vec![0xEE_u8; layout.len().unwrap()];
        buffer[0..2].copy_from_slice(&[1, 2]);
        buffer[4..6].copy_from_slice(&[3, 4]);
        buffer[16..18].copy_from_slice(&[5, 6]);
        assert_eq!(to_planar(layout, &buffer).unwrap(), [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn a_short_buffer_or_odd_size_is_refused() {
        assert!(to_planar(Layout::packed(2, 2), &[0; 5]).is_err());
        assert!(from_rgba(3, 2, &[0; 24]).is_err());
        assert!(from_planar(2, 2, &[0; 5]).is_err());
    }
}
