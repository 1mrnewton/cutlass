//! Frame header, picture header, and slice table (RDD 36 sections 5.1 and 5.2).

use cutlass_decode::CodecError;

/// Longest edge this crate decodes or encodes.
pub(crate) const MAX_EDGE: u32 = 16_384;
/// Most pixels in one frame: 8192 by 8192.
pub(crate) const MAX_PIXELS: u64 = 67_108_864;
/// Bytes before the frame header: `frame_size` and `'icpf'`.
pub(crate) const FRAME_PREFIX: usize = 8;
/// Frame header bytes before the optional matrices.
pub(crate) const FRAME_HEADER_FIXED: usize = 20;
/// The frame identifier.
pub(crate) const ICPF: [u8; 4] = *b"icpf";

/// Chroma sampling of a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chroma {
    /// 4:2:2, the 422 family.
    Yuv422,
    /// 4:4:4, the 4444 family.
    Yuv444,
}

/// How a frame is split into pictures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interlace {
    Progressive,
    /// Two field pictures, the top field first.
    TopFirst,
    /// Two field pictures, the bottom field first.
    BottomFirst,
}

/// The coded alpha channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlphaBits {
    None,
    Eight,
    Sixteen,
}

impl AlphaBits {
    /// Bits per coded alpha sample, 0 when there is no alpha.
    #[must_use]
    pub const fn bits(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Eight => 8,
            Self::Sixteen => 16,
        }
    }
}

/// Everything the frame header says.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameHeader {
    /// Bytes in the frame, `frame_size` itself and `'icpf'` included.
    pub frame_size: usize,
    /// Bytes in the frame header.
    pub header_size: usize,
    pub version: u8,
    pub encoder: [u8; 4],
    pub width: u32,
    pub height: u32,
    pub chroma: Chroma,
    pub interlace: Interlace,
    pub aspect_ratio: u8,
    pub frame_rate: u8,
    /// H.273 code points.
    pub primaries: u8,
    pub transfer: u8,
    pub matrix: u8,
    pub alpha: AlphaBits,
    /// Quantization weights in raster order, `[v * 8 + u]`.
    pub luma_matrix: [u8; 64],
    pub chroma_matrix: [u8; 64],
}

impl FrameHeader {
    /// Rows in picture `index`: the whole frame, or one field.
    #[must_use]
    pub fn picture_height(&self, index: usize) -> u32 {
        match self.interlace {
            Interlace::Progressive => self.height,
            Interlace::TopFirst | Interlace::BottomFirst => {
                if self.field_parity(index) == 0 {
                    self.height.div_ceil(2)
                } else {
                    self.height / 2
                }
            }
        }
    }

    /// Pictures in the frame: 1, or 2 for interlaced.
    #[must_use]
    pub fn pictures(&self) -> usize {
        if self.interlace == Interlace::Progressive {
            1
        } else {
            2
        }
    }

    /// 0 when picture `index` holds the top field (or the whole frame), 1 for the bottom field.
    #[must_use]
    pub fn field_parity(&self, index: usize) -> usize {
        match self.interlace {
            Interlace::Progressive | Interlace::TopFirst => index & 1,
            Interlace::BottomFirst => (index & 1) ^ 1,
        }
    }

    pub(crate) fn width_in_mb(&self) -> u32 {
        self.width.div_ceil(16)
    }
}

fn truncated(at_byte: usize, what: &'static str) -> CodecError {
    CodecError::Truncated {
        at_bit: (at_byte as u64).saturating_mul(8),
        what,
    }
}

fn malformed(at_byte: usize, what: &'static str) -> CodecError {
    CodecError::Malformed {
        at_bit: (at_byte as u64).saturating_mul(8),
        what,
    }
}

fn be16(data: &[u8], at: usize) -> Option<u16> {
    let bytes = data.get(at..at.checked_add(2)?)?;
    Some(u16::from_be_bytes([bytes[0], bytes[1]]))
}

fn be32(data: &[u8], at: usize) -> Option<u32> {
    let bytes = data.get(at..at.checked_add(4)?)?;
    Some(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

/// Parses the frame header at the start of `data`.
pub fn parse_frame_header(data: &[u8]) -> Result<FrameHeader, CodecError> {
    let frame_size = be32(data, 0).ok_or(truncated(0, "frame size"))? as usize;
    if data.get(4..8) != Some(&ICPF[..]) {
        if data.len() < FRAME_PREFIX {
            return Err(truncated(4, "frame identifier"));
        }
        return Err(malformed(4, "frame identifier"));
    }
    if frame_size < FRAME_PREFIX + FRAME_HEADER_FIXED {
        return Err(malformed(0, "frame size"));
    }
    if frame_size > data.len() {
        return Err(truncated(data.len(), "frame"));
    }
    let frame = &data[..frame_size];
    let h = FRAME_PREFIX;
    let header_size = usize::from(be16(frame, h).ok_or(truncated(h, "frame header"))?);
    if header_size < FRAME_HEADER_FIXED {
        return Err(malformed(h, "frame header size"));
    }
    if h + header_size > frame.len() {
        return Err(truncated(frame.len(), "frame header"));
    }
    let header = &frame[h..h + header_size];
    let version = header[3];
    if version > 1 {
        return Err(CodecError::Unsupported {
            what: "bitstream version",
        });
    }
    let encoder = [header[4], header[5], header[6], header[7]];
    let width = u32::from(u16::from_be_bytes([header[8], header[9]]));
    let height = u32::from(u16::from_be_bytes([header[10], header[11]]));
    if width == 0 || height == 0 {
        return Err(malformed(h + 8, "frame dimensions"));
    }
    if width > MAX_EDGE || height > MAX_EDGE || u64::from(width) * u64::from(height) > MAX_PIXELS {
        return Err(CodecError::Limit {
            what: "frame dimensions",
        });
    }
    let chroma = match header[12] >> 6 {
        2 => Chroma::Yuv422,
        3 => Chroma::Yuv444,
        _ => return Err(malformed(h + 12, "chroma format")),
    };
    let interlace = match (header[12] >> 2) & 3 {
        0 => Interlace::Progressive,
        1 => Interlace::TopFirst,
        2 => Interlace::BottomFirst,
        _ => return Err(malformed(h + 12, "interlace mode")),
    };
    let alpha = match header[17] & 0x0f {
        0 => AlphaBits::None,
        1 => AlphaBits::Eight,
        2 => AlphaBits::Sixteen,
        _ => return Err(malformed(h + 17, "alpha channel type")),
    };
    let load_luma = header[19] & 2 != 0;
    let load_chroma = header[19] & 1 != 0;
    let mut at = FRAME_HEADER_FIXED;
    let mut luma_matrix = [4u8; 64];
    if load_luma {
        luma_matrix = read_matrix(header, at, h)?;
        at += 64;
    }
    let chroma_matrix = if load_chroma {
        read_matrix(header, at, h)?
    } else {
        luma_matrix
    };
    Ok(FrameHeader {
        frame_size,
        header_size,
        version,
        encoder,
        width,
        height,
        chroma,
        interlace,
        aspect_ratio: header[13] >> 4,
        frame_rate: header[13] & 0x0f,
        primaries: header[14],
        transfer: header[15],
        matrix: header[16],
        alpha,
        luma_matrix,
        chroma_matrix,
    })
}

fn read_matrix(header: &[u8], at: usize, base: usize) -> Result<[u8; 64], CodecError> {
    let bytes = header
        .get(at..at + 64)
        .ok_or(malformed(base, "frame header size"))?;
    let mut matrix = [0u8; 64];
    matrix.copy_from_slice(bytes);
    if matrix.contains(&0) {
        return Err(malformed(base + at, "quantization weight"));
    }
    Ok(matrix)
}

/// One slice: where its bytes are and which macroblocks it covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SliceRef {
    /// Byte offset in the frame.
    pub offset: usize,
    pub size: usize,
    /// Macroblock column of the slice's first macroblock.
    pub mb_x: u32,
    /// Macroblocks in the slice: 1, 2, 4 or 8.
    pub mbs: u32,
}

/// One picture's slices, by macroblock row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Picture {
    /// Rows of the picture: the frame, or a field.
    pub height: u32,
    pub height_in_mb: u32,
    pub slices_per_row: usize,
    /// `height_in_mb * slices_per_row` slices, row by row.
    pub slices: Vec<SliceRef>,
    /// Frame row of picture row 0, and the step between picture rows.
    pub first_row: u32,
    pub row_step: u32,
}

/// The slice widths of one macroblock row: full slices of `1 << log2`
/// macroblocks, then the rest split into falling powers of two.
pub(crate) fn slice_widths(width_in_mb: u32, log2: u32) -> Vec<u32> {
    let full = 1u32 << log2;
    let mut widths = vec![full; (width_in_mb >> log2) as usize];
    let mut rest = width_in_mb & (full - 1);
    let mut size = full >> 1;
    while rest > 0 {
        if rest >= size {
            widths.push(size);
            rest -= size;
        }
        size >>= 1;
    }
    widths
}

/// Parses picture `index` starting at byte `at` of the frame.
/// Returns the picture and the byte after it.
pub(crate) fn parse_picture(
    frame: &[u8],
    header: &FrameHeader,
    index: usize,
    at: usize,
) -> Result<(Picture, usize), CodecError> {
    let first = *frame.get(at).ok_or(truncated(at, "picture header"))?;
    let header_size = usize::from(first >> 3);
    if header_size < 8 {
        return Err(malformed(at, "picture header size"));
    }
    if at + header_size > frame.len() {
        return Err(truncated(frame.len(), "picture header"));
    }
    let picture_size = be32(frame, at + 1).ok_or(truncated(at + 1, "picture header"))? as usize;
    if picture_size < header_size {
        return Err(malformed(at + 1, "picture size"));
    }
    let end = at
        .checked_add(picture_size)
        .ok_or(malformed(at + 1, "picture size"))?;
    if end > frame.len() {
        return Err(truncated(frame.len(), "picture"));
    }
    let log2 = u32::from((frame[at + 7] >> 4) & 3);
    let height = header.picture_height(index);
    if height == 0 {
        return Err(malformed(at, "picture height"));
    }
    let height_in_mb = height.div_ceil(16);
    let widths = slice_widths(header.width_in_mb(), log2);
    let slices_per_row = widths.len();
    let count = slices_per_row * height_in_mb as usize;
    let table = at + header_size;
    let mut offset = table + 2 * count;
    if offset > end {
        return Err(truncated(end, "slice table"));
    }
    let mut slices = Vec::with_capacity(count);
    for _ in 0..height_in_mb {
        let mut mb_x = 0;
        for &mbs in &widths {
            let entry = table + 2 * slices.len();
            let size = usize::from(be16(frame, entry).ok_or(truncated(entry, "slice table"))?);
            if size < 6 {
                return Err(malformed(entry, "slice size"));
            }
            if offset + size > end {
                return Err(malformed(entry, "slice sizes overrun the picture"));
            }
            slices.push(SliceRef {
                offset,
                size,
                mb_x,
                mbs,
            });
            offset += size;
            mb_x += mbs;
        }
    }
    let parity = header.field_parity(index) as u32;
    let (first_row, row_step) = if header.pictures() == 1 {
        (0, 1)
    } else {
        (parity, 2)
    };
    Ok((
        Picture {
            height,
            height_in_mb,
            slices_per_row,
            slices,
            first_row,
            row_step,
        },
        end,
    ))
}

#[cfg(test)]
mod tests {
    use super::slice_widths;

    #[test]
    fn a_row_splits_into_falling_powers_of_two() {
        assert_eq!(slice_widths(45, 3), vec![8, 8, 8, 8, 8, 4, 1]);
        assert_eq!(slice_widths(120, 3), vec![8; 15]);
        assert_eq!(slice_widths(3, 3), vec![2, 1]);
        assert_eq!(slice_widths(7, 1), vec![2, 2, 2, 1]);
        assert_eq!(slice_widths(5, 0), vec![1; 5]);
        assert_eq!(slice_widths(1, 2), vec![1]);
    }
}
