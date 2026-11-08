//! Decoded pictures and the macroblock record a later B-VOP reads.

use crate::mc::RefPlane;

/// How a macroblock of a reference VOP was predicted, as direct mode in
/// the B-VOPs between it and the previous reference reads it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Kind {
    #[default]
    Intra,
    /// Not coded in a P-VOP: B-VOPs skip the colocated macroblock too.
    Skipped,
    /// One vector for the macroblock, including GMC.
    Frame,
    /// Four 8x8 vectors.
    Four,
    /// Two field vectors.
    Field,
}

/// One macroblock of a reference VOP.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct MbRecord {
    pub(crate) kind: Kind,
    /// Vector of each 8x8 block, in the VOP's half or quarter samples.
    pub(crate) mv: [(i32, i32); 4],
    /// Field vectors (vertical in field units) and their reference fields.
    pub(crate) field_mv: [(i32, i32); 2],
    pub(crate) field_select: [u8; 2],
}

/// A decoded picture over whole macroblocks.
#[derive(Clone)]
pub(crate) struct Frame {
    pub(crate) mbw: usize,
    pub(crate) mbh: usize,
    pub(crate) y: Vec<u8>,
    pub(crate) u: Vec<u8>,
    pub(crate) v: Vec<u8>,
    /// Per macroblock, for direct mode; empty for B-VOPs.
    pub(crate) mbs: Vec<MbRecord>,
    /// Visible size a four-vector macroblock clips to. Whole macroblocks
    /// when the picture fills them.
    pub(crate) clip_width: i32,
    pub(crate) clip_height: i32,
}

impl Frame {
    pub(crate) fn new(mbw: usize, mbh: usize) -> Self {
        let luma = mbw * 16 * mbh * 16;
        Self {
            mbw,
            mbh,
            y: vec![0; luma],
            u: vec![0; luma / 4],
            v: vec![0; luma / 4],
            mbs: Vec::new(),
            clip_width: (mbw * 16) as i32,
            clip_height: (mbh * 16) as i32,
        }
    }

    pub(crate) fn luma_stride(&self) -> usize {
        self.mbw * 16
    }

    pub(crate) fn chroma_stride(&self) -> usize {
        self.mbw * 8
    }

    /// The planes as motion compensation references, padded from
    /// `edge` (luma columns and rows of real samples).
    pub(crate) fn refs(&self, edge: (u32, u32)) -> [RefPlane<'_>; 3] {
        let (ew, eh) = (edge.0 as i32, edge.1 as i32);
        [
            RefPlane {
                data: &self.y,
                stride: self.luma_stride(),
                edge_w: ew,
                edge_h: eh,
            },
            RefPlane {
                data: &self.u,
                stride: self.chroma_stride(),
                edge_w: ew >> 1,
                edge_h: eh >> 1,
            },
            RefPlane {
                data: &self.v,
                stride: self.chroma_stride(),
                edge_w: ew >> 1,
                edge_h: eh >> 1,
            },
        ]
    }

    /// Copies the `width` by `height` (both even) top-left of the picture
    /// as packed 4:2:0.
    pub(crate) fn packed(&self, width: usize, height: usize) -> Vec<u8> {
        let (cw, ch) = (width / 2, height / 2);
        let mut out = Vec::with_capacity(width * height + 2 * cw * ch);
        let ys = self.luma_stride();
        let cs = self.chroma_stride();
        for r in 0..height {
            if let Some(row) = self.y.get(r * ys..r * ys + width) {
                out.extend_from_slice(row);
            }
        }
        for plane in [&self.u, &self.v] {
            for r in 0..ch {
                if let Some(row) = plane.get(r * cs..r * cs + cw) {
                    out.extend_from_slice(row);
                }
            }
        }
        out
    }
}
