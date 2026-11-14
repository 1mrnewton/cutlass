//! Decoded pictures: three planes padded to whole superblocks, and the
//! motion of every 8×8 block for the next frame's vector prediction.

use crate::dsp::pixel::Pixel;

/// One plane. Rows are `stride` apart; `w`×`h` is the visible size and
/// `aw`×`ah` the size aligned to 8 luma samples, which intra edges use.
/// The buffer covers whole superblocks, so a block never writes past it.
#[derive(Clone)]
pub struct Plane<P> {
    pub data: Vec<P>,
    pub stride: usize,
    pub w: usize,
    pub h: usize,
    pub aw: usize,
    pub ah: usize,
}

/// A motion vector in eighths of a luma sample.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Mv {
    pub row: i16,
    pub col: i16,
}

impl Mv {
    pub const ZERO: Mv = Mv { row: 0, col: 0 };
}

/// The vectors and references of one 8×8 block. `refs[k]` is 0 when the
/// block has no `k`-th reference (and `refs[0]` is 0 for intra blocks).
#[derive(Clone, Copy, Default, Debug)]
pub struct MvRef {
    pub mv: [Mv; 2],
    pub refs: [u8; 2],
}

#[derive(Clone)]
pub struct Frame<P> {
    pub planes: [Plane<P>; 3],
    pub width: usize,
    pub height: usize,
    pub ss_x: usize,
    pub ss_y: usize,
    pub bit_depth: u8,
    pub color_space: u8,
    pub color_range: u8,
    pub render_width: u32,
    pub render_height: u32,
}

impl<P: Pixel> Frame<P> {
    pub fn new(width: usize, height: usize, ss_x: usize, ss_y: usize, bit_depth: u8) -> Self {
        let w64 = width.div_ceil(64) * 64;
        let h64 = height.div_ceil(64) * 64;
        let aw = width.div_ceil(8) * 8;
        let ah = height.div_ceil(8) * 8;
        let luma = Plane {
            data: vec![P::default(); w64 * h64],
            stride: w64,
            w: width,
            h: height,
            aw,
            ah,
        };
        let chroma = Plane {
            data: vec![P::default(); (w64 >> ss_x) * (h64 >> ss_y)],
            stride: w64 >> ss_x,
            w: (width + ss_x) >> ss_x,
            h: (height + ss_y) >> ss_y,
            aw: aw >> ss_x,
            ah: ah >> ss_y,
        };
        Self {
            planes: [luma, chroma.clone(), chroma],
            width,
            height,
            ss_x,
            ss_y,
            bit_depth,
            color_space: 0,
            color_range: 0,
            render_width: width as u32,
            render_height: height as u32,
        }
    }

    /// Whether the buffers can be filled again for a frame of this shape.
    pub fn fits(&self, width: usize, height: usize, ss_x: usize, ss_y: usize, bd: u8) -> bool {
        self.width == width
            && self.height == height
            && self.ss_x == ss_x
            && self.ss_y == ss_y
            && self.bit_depth == bd
    }
}
