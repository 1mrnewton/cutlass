//! Decoded pictures: three planes padded to whole superblocks, shared
//! between the thread that reconstructs them and the frames that predict
//! from their finished rows, and the motion of every 8×8 block for the
//! next frame's vector prediction.

use std::sync::Arc;

use cutlass_simd::vp9::frame::{FrameWriter, SharedFrame};

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

/// The shape of a picture's planes: visible size `w`×`h`, the size
/// aligned to 8 luma samples that intra edges use (`aw`×`ah`), and the
/// buffers' stride and rows (whole superblocks).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    pub width: usize,
    pub height: usize,
    pub ss_x: usize,
    pub ss_y: usize,
    pub bit_depth: u8,
    pub w: [usize; 3],
    pub h: [usize; 3],
    pub aw: [usize; 3],
    pub ah: [usize; 3],
    pub stride: [usize; 3],
    pub rows: [usize; 3],
}

impl Dims {
    pub fn new(width: usize, height: usize, ss_x: usize, ss_y: usize, bit_depth: u8) -> Self {
        let w64 = width.div_ceil(64) * 64;
        let h64 = height.div_ceil(64) * 64;
        let aw = width.div_ceil(8) * 8;
        let ah = height.div_ceil(8) * 8;
        let cw = (width + ss_x) >> ss_x;
        let ch = (height + ss_y) >> ss_y;
        Self {
            width,
            height,
            ss_x,
            ss_y,
            bit_depth,
            w: [width, cw, cw],
            h: [height, ch, ch],
            aw: [aw, aw >> ss_x, aw >> ss_x],
            ah: [ah, ah >> ss_y, ah >> ss_y],
            stride: [w64, w64 >> ss_x, w64 >> ss_x],
            rows: [h64, h64 >> ss_y, h64 >> ss_y],
        }
    }

    pub fn shifts(&self) -> [u32; 3] {
        [0, self.ss_y as u32, self.ss_y as u32]
    }
}

/// A picture: its samples, published row by row as they become final.
pub struct Pic<P> {
    pub buf: Arc<SharedFrame<P>>,
    pub dims: Dims,
    pub color_space: u8,
    pub color_range: u8,
    pub render_width: u32,
    pub render_height: u32,
}

/// A buffer for a new picture: a picture's whose readers are gone when
/// one of `spare` fits, else a new one.
pub fn writer<P: Copy + Default>(dims: &Dims, spare: &mut Vec<(Dims, Arc<SharedFrame<P>>)>) -> FrameWriter<P> {
    while let Some(i) = spare.iter().position(|(d, _)| d.stride == dims.stride && d.rows == dims.rows) {
        let (_, buf) = spare.swap_remove(i);
        if let Ok(w) = FrameWriter::restart(buf) {
            return w;
        }
    }
    FrameWriter::new(dims.stride, dims.rows, dims.shifts())
}
