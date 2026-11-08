//! Reconstructed frames and the field and frame views into them.

use crate::headers::{CHROMA_420, CHROMA_422};

/// Coded layout of one sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Geometry {
    pub mb_width: usize,
    /// Macroblock rows of a frame. A field has half as many.
    pub mb_height: usize,
    pub chroma_format: u8,
}

impl Geometry {
    /// `(log2 horizontal, log2 vertical)` chroma subsampling.
    pub(crate) fn chroma_shift(&self) -> (u32, u32) {
        match self.chroma_format {
            CHROMA_420 => (1, 1),
            CHROMA_422 => (1, 0),
            _ => (0, 0),
        }
    }

    /// Row stride and line count of plane `c` of a frame.
    pub(crate) fn plane(&self, c: usize) -> (usize, usize) {
        let (w, h) = (self.mb_width * 16, self.mb_height * 16);
        if c == 0 {
            (w, h)
        } else {
            let (sx, sy) = self.chroma_shift();
            (w >> sx, h >> sy)
        }
    }

    /// Width and height of plane `c` inside one macroblock.
    pub(crate) fn mb_size(&self, c: usize) -> (usize, usize) {
        if c == 0 {
            (16, 16)
        } else {
            let (sx, sy) = self.chroma_shift();
            (16 >> sx, 16 >> sy)
        }
    }

    pub(crate) fn blocks_per_mb(&self) -> usize {
        match self.chroma_format {
            CHROMA_420 => 6,
            CHROMA_422 => 8,
            _ => 12,
        }
    }

    pub(crate) fn frame_bytes(&self) -> usize {
        (0..3).map(|c| self.plane(c).0 * self.plane(c).1).sum()
    }
}

/// Three planes at the coded size, rows packed at the plane stride.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Frame {
    pub planes: [Vec<u8>; 3],
}

impl Frame {
    pub(crate) fn new(geo: &Geometry, fill: u8) -> Self {
        let plane = |c| {
            let (w, h) = geo.plane(c);
            vec![fill; w * h]
        };
        Self {
            planes: [plane(0), plane(1), plane(2)],
        }
    }
}

/// A read-only window onto one plane: a whole frame or one field of it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct View<'a> {
    pub data: &'a [u8],
    pub offset: usize,
    pub stride: usize,
    pub width: usize,
    pub height: usize,
}

impl<'a> View<'a> {
    pub(crate) fn frame(frame: &'a Frame, geo: &Geometry, c: usize) -> Self {
        let (w, h) = geo.plane(c);
        Self {
            data: &frame.planes[c],
            offset: 0,
            stride: w,
            width: w,
            height: h,
        }
    }

    /// Field `parity` (0 top, 1 bottom) of plane `c`.
    pub(crate) fn field(frame: &'a Frame, geo: &Geometry, c: usize, parity: usize) -> Self {
        let (w, h) = geo.plane(c);
        Self {
            data: &frame.planes[c],
            offset: parity * w,
            stride: 2 * w,
            width: w,
            height: h / 2,
        }
    }
}
