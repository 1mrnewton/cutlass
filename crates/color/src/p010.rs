//! P010 and P016 out of a hardware decoder, as 8-bit planar 4:2:0.
//!
//! Both are NV12 with 16-bit little-endian samples: a full-size Y plane,
//! then interleaved U and V at half size, each row padded to a stride. The
//! significant bits sit at the top of each sample, so the high byte is the
//! sample shifted down to 8 bits, as the Rust HEVC decoder does for 10-bit
//! streams.

use cutlass_codec::access::CodecError;

/// Where one P010 picture sits inside a decoder's buffer.
///
/// `stride` is the byte step between rows of both planes, at least twice
/// `width`. `rows` is the height of the Y plane in the buffer. `chroma` is
/// the byte offset of the UV plane. The picture is the top-left `width` by
/// `height`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub width: u32,
    pub height: u32,
    pub stride: usize,
    pub chroma: usize,
}

impl Layout {
    /// A tightly packed picture: rows two bytes a sample, chroma right
    /// after the luma plane.
    pub fn packed(width: u32, height: u32) -> Self {
        let stride = width as usize * 2;
        Self {
            width,
            height,
            stride,
            chroma: stride * height as usize,
        }
    }

    /// Bytes the buffer must hold for this layout.
    pub fn byte_len(&self) -> Result<usize, CodecError> {
        self.check()?;
        let luma = self.stride.checked_mul(self.height as usize).ok_or(LIMIT)?;
        let chroma = self
            .stride
            .checked_mul(self.height.div_ceil(2) as usize)
            .ok_or(LIMIT)?;
        let end = self.chroma.checked_add(chroma).ok_or(LIMIT)?;
        Ok(end.max(luma))
    }

    fn check(&self) -> Result<(), CodecError> {
        if self.width == 0
            || self.height == 0
            || !self.width.is_multiple_of(2)
            || !self.height.is_multiple_of(2)
        {
            return Err(CodecError::InvalidApi { what: "p010 size" });
        }
        let row = (self.width as usize).checked_mul(2).ok_or(LIMIT)?;
        let luma = self.stride.checked_mul(self.height as usize).ok_or(LIMIT)?;
        if self.stride < row || self.chroma < luma {
            return Err(CodecError::InvalidApi {
                what: "p010 stride",
            });
        }
        Ok(())
    }
}

const LIMIT: CodecError = CodecError::Limit {
    what: "picture bytes",
};

/// One P010 picture to tightly packed 8-bit planar 4:2:0.
pub fn to_planar8(layout: Layout, buffer: &[u8]) -> Result<Vec<u8>, CodecError> {
    if buffer.len() < layout.byte_len()? {
        return Err(CodecError::Truncated {
            at_bit: (buffer.len() as u64).saturating_mul(8),
            what: "p010 buffer",
        });
    }
    let (w, h) = (layout.width as usize, layout.height as usize);
    let mut out = Vec::with_capacity(w * h * 3 / 2);
    for row in 0..h {
        let at = row * layout.stride;
        for sample in buffer[at..at + w * 2].as_chunks::<2>().0 {
            out.push(sample[1]);
        }
    }
    let mut v = Vec::with_capacity(w * h / 4);
    for row in 0..h / 2 {
        let at = layout.chroma + row * layout.stride;
        for [_, cb, _, cr] in buffer[at..at + w * 2].as_chunks::<4>().0 {
            out.push(*cb);
            v.push(*cr);
        }
    }
    out.extend_from_slice(&v);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(value: u16) -> [u8; 2] {
        (value << 6).to_le_bytes()
    }

    #[test]
    fn the_top_eight_bits_of_each_sample_come_out_planar() {
        // 2x2 picture, stride 6 (4 bytes of samples and 2 of padding).
        let layout = Layout {
            width: 2,
            height: 2,
            stride: 6,
            chroma: 12,
        };
        let mut buffer = Vec::new();
        for row in [[400_u16, 404], [408, 1023]] {
            for value in row {
                buffer.extend_from_slice(&sample(value));
            }
            buffer.extend_from_slice(&[9, 9]);
        }
        buffer.extend_from_slice(&sample(512));
        buffer.extend_from_slice(&sample(256));
        buffer.extend_from_slice(&[9, 9]);
        let planar = to_planar8(layout, &buffer).expect("planar");
        assert_eq!(planar, vec![100, 101, 102, 255, 128, 64]);
    }

    #[test]
    fn a_short_buffer_is_truncated_and_a_narrow_stride_refused() {
        let layout = Layout::packed(2, 2);
        assert!(matches!(
            to_planar8(layout, &[0; 4]),
            Err(CodecError::Truncated { .. })
        ));
        let narrow = Layout {
            stride: 3,
            ..layout
        };
        assert!(matches!(
            to_planar8(narrow, &[0; 64]),
            Err(CodecError::InvalidApi { .. })
        ));
    }
}
