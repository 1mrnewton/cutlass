//! Closed-loop partition choice: a node is coded once with each candidate
//! partition, and the cheaper coding (squared error of the reconstruction
//! plus lambda times the bits written) is kept. Intra frames try NONE and
//! SPLIT at every node the open-loop estimate does not rule out; inter
//! frames (optionally) check the splits the estimate chooses.

use crate::dec::cdf::CdfContext;
use crate::dec::coder::Coder;
use crate::dec::tile::{Block, TileDec};
use crate::dec::types::{BLOCK_8X8, MiInfo, bw4};
use crate::dsp::Pixel;

use super::msac_enc::CoderMark;

const PARTITION_NONE: usize = 0;
const PARTITION_SPLIT: usize = 3;

/// What coding a node changes, saved to code it again another way.
pub struct NodeState<P> {
    cdf: Box<CdfContext>,
    msac: CoderMark,
    mi: Vec<MiInfo>,
    bytes: [Vec<u8>; 4],
    lf_tx: [Vec<u8>; 3],
    cdef: i8,
    pix: [Vec<P>; 3],
    above: [Vec<u8>; 7],
    left: [[u8; 32]; 7],
    above_palette: [Vec<[u16; 8]>; 2],
    left_palette: [[[u16; 8]; 32]; 2],
    block_decoded: [[[bool; 35]; 35]; 3],
    delta_lf: [i8; 4],
    current_q_index: i32,
    read_deltas: bool,
    b: Block,
}

/// A node: position and size in 4x4 units.
#[derive(Clone, Copy)]
struct Node {
    r: usize,
    c: usize,
    n4: usize,
}

fn save_rect<T: Copy>(v: &[T], stride: usize, r: usize, c: usize, w: usize, h: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(w * h);
    for i in 0..h {
        let at = (r + i) * stride + c;
        out.extend_from_slice(&v[at..at + w]);
    }
    out
}

fn load_rect<T: Copy>(v: &mut [T], stride: usize, r: usize, c: usize, w: usize, h: usize, s: &[T]) {
    for i in 0..h {
        let at = (r + i) * stride + c;
        v[at..at + w].copy_from_slice(&s[i * w..(i + 1) * w]);
    }
}

impl<P: Pixel, C: Coder> TileDec<'_, P, C> {
    /// Plane `p`'s area of a node: (row, col, width, height) in 4x4 units.
    fn node_area4(&self, nd: Node, p: usize) -> (usize, usize, usize, usize) {
        let (sx, sy) = self.plane_sub(p);
        let w = (nd.n4 >> sx).max(1);
        let h = (nd.n4 >> sy).max(1);
        (nd.r >> sy, nd.c >> sx, w, h)
    }

    fn node_save(&self, nd: Node, base: Option<&CoderMark>) -> Option<NodeState<P>> {
        let msac = self.msac.checkpoint(base)?;
        let st = self.fs.mi_stride;
        let n = nd.n4;
        let mut lf_tx: [Vec<u8>; 3] = Default::default();
        let mut pix: [Vec<P>; 3] = Default::default();
        let mut above: [Vec<u8>; 7] = Default::default();
        let mut left = [[0u8; 32]; 7];
        for p in 0..self.num_planes {
            let (r4, c4, w4, h4) = self.node_area4(nd, p);
            lf_tx[p] = save_rect(&self.fs.lf_tx[p], st, r4, c4, w4, h4);
            let pl = &self.cur.planes[p];
            pix[p] = save_rect(&pl.data, pl.stride, r4 * 4, c4 * 4, w4 * 4, h4 * 4);
            above[p] = self.above_level[p][c4..c4 + w4].to_vec();
            above[3 + p] = self.above_dc[p][c4..c4 + w4].to_vec();
            left[p] = self.left_level[p];
            left[3 + p] = self.left_dc[p];
        }
        above[6] = self.above_seg_pred[nd.c..nd.c + n].to_vec();
        left[6] = self.left_seg_pred;
        let ap = |k: usize| {
            let v = &self.above_palette[k];
            v[nd.c.min(v.len())..(nd.c + n).min(v.len())].to_vec()
        };
        let cdef_at = (nd.r >> 4) * self.fs.cdef_stride + (nd.c >> 4);
        Some(NodeState {
            cdf: Box::new(self.cdf.clone()),
            msac,
            mi: save_rect(&self.fs.mi, st, nd.r, nd.c, n, n),
            bytes: [
                save_rect(&self.fs.uv_modes, st, nd.r, nd.c, n, n),
                save_rect(&self.fs.tx_types, st, nd.r, nd.c, n, n),
                save_rect(&self.fs.inter_tx_sizes, st, nd.r, nd.c, n, n),
                save_rect(&self.fs.segment_ids, st, nd.r, nd.c, n, n),
            ],
            lf_tx,
            cdef: self.fs.cdef_idx.get(cdef_at).copied().unwrap_or(-1),
            pix,
            above,
            left,
            above_palette: [ap(0), ap(1)],
            left_palette: self.left_palette,
            block_decoded: self.block_decoded,
            delta_lf: self.delta_lf,
            current_q_index: self.current_q_index,
            read_deltas: self.read_deltas,
            b: self.b.clone(),
        })
    }

    fn node_restore(&mut self, nd: Node, s: &NodeState<P>) {
        self.msac.restore(&s.msac);
        self.cdf.clone_from(&s.cdf);
        let st = self.fs.mi_stride;
        let n = nd.n4;
        load_rect(&mut self.fs.mi, st, nd.r, nd.c, n, n, &s.mi);
        load_rect(&mut self.fs.uv_modes, st, nd.r, nd.c, n, n, &s.bytes[0]);
        load_rect(&mut self.fs.tx_types, st, nd.r, nd.c, n, n, &s.bytes[1]);
        load_rect(
            &mut self.fs.inter_tx_sizes,
            st,
            nd.r,
            nd.c,
            n,
            n,
            &s.bytes[2],
        );
        load_rect(&mut self.fs.segment_ids, st, nd.r, nd.c, n, n, &s.bytes[3]);
        for p in 0..self.num_planes {
            let (r4, c4, w4, h4) = self.node_area4(nd, p);
            load_rect(&mut self.fs.lf_tx[p], st, r4, c4, w4, h4, &s.lf_tx[p]);
            let pl = &mut self.cur.planes[p];
            let stride = pl.stride;
            load_rect(
                &mut pl.data,
                stride,
                r4 * 4,
                c4 * 4,
                w4 * 4,
                h4 * 4,
                &s.pix[p],
            );
            self.above_level[p][c4..c4 + w4].copy_from_slice(&s.above[p]);
            self.above_dc[p][c4..c4 + w4].copy_from_slice(&s.above[3 + p]);
            self.left_level[p] = s.left[p];
            self.left_dc[p] = s.left[3 + p];
        }
        self.above_seg_pred[nd.c..nd.c + n].copy_from_slice(&s.above[6]);
        self.left_seg_pred = s.left[6];
        for k in 0..2 {
            let v = &mut self.above_palette[k];
            let (a, b) = (nd.c.min(v.len()), (nd.c + n).min(v.len()));
            v[a..b].copy_from_slice(&s.above_palette[k]);
        }
        let cdef_at = (nd.r >> 4) * self.fs.cdef_stride + (nd.c >> 4);
        if let Some(v) = self.fs.cdef_idx.get_mut(cdef_at) {
            *v = s.cdef;
        }
        self.left_palette = s.left_palette;
        self.block_decoded = s.block_decoded;
        self.delta_lf = s.delta_lf;
        self.current_q_index = s.current_q_index;
        self.read_deltas = s.read_deltas;
        self.b.clone_from(&s.b);
        self.me_scratch.pred_cache.clear();
        self.me_scratch.tx_memo = None;
    }

    /// Squared error of the node's reconstruction (all planes, visible part).
    fn node_sse(&self, nd: Node) -> f64 {
        let Some(src) = self.src else { return 0.0 };
        let mut d = 0u64;
        for p in 0..self.num_planes {
            let (r4, c4, w4, h4) = self.node_area4(nd, p);
            let a = &self.cur.planes[p];
            let s = &src.planes[p];
            let (x0, y0) = (c4 * 4, r4 * 4);
            let x1 = (x0 + w4 * 4).min(a.w);
            let y1 = (y0 + h4 * 4).min(a.h);
            for y in y0..y1 {
                let ra = &a.data[y * a.stride..];
                let rs = &s.data[y * s.stride..];
                for x in x0..x1 {
                    let e = i64::from(ra[x].to_i32() - rs[x].to_i32());
                    d += (e * e) as u64;
                }
            }
        }
        d as f64
    }

    /// Codes the node with `p` (forced on the coming `enc_partition`) and
    /// returns its cost.
    fn node_try(&mut self, nd: Node, bsize: u8, p: usize) -> crate::dec::tile::R<f64> {
        let b0 = self.msac.bits_written();
        self.me_scratch.force_part = Some((nd.r, nd.c, bsize, p));
        self.decode_partition(nd.r, nd.c, bsize)?;
        let bits = self.msac.bits_written() - b0;
        Ok(self.node_sse(nd) + self.enc.lambda * bits)
    }

    /// Codes a partition node by trying NONE and SPLIT closed loop. Returns
    /// whether it did (otherwise the caller picks the partition).
    pub fn enc_partition_trials(
        &mut self,
        r: usize,
        c: usize,
        bsize: u8,
        inside: bool,
    ) -> crate::dec::tile::R<bool> {
        if !C::WRITE || !self.mode.part_rd || !inside || bsize <= BLOCK_8X8 {
            return Ok(false);
        }
        if self
            .me_scratch
            .force_part
            .is_some_and(|f| f.0 == r && f.1 == c && f.2 == bsize)
        {
            return Ok(false);
        }
        let n = bw4(bsize) * 4;
        if n > self.mode.max_block || n / 2 < self.mode.min_block {
            return Ok(false);
        }
        if n >= self.sb_size4 * 4 {
            self.me_scratch.est_memo.clear();
        }
        let (x, y) = (c * 4, r * 4);
        let nd = Node {
            r,
            c,
            n4: bw4(bsize),
        };
        let lvl = (n.trailing_zeros() as usize).saturating_sub(4).min(2);
        let (mut try_none, mut try_split) = (true, true);
        let mut none_first = false;
        if self.mode.inter_frame {
            // Inter frames: a split the estimate chooses is checked against
            // coding the node whole.
            if n > self.mode.merge_check
                || self.enc_partition_pick(r, c, bsize, true, true) != PARTITION_SPLIT
            {
                return Ok(false);
            }
            none_first = self.mode.merge_none_first;
        } else if self.mode.intra_none_skip[lvl] <= 0.0 {
            try_none = false;
        } else if self.mode.intra_split_skip[lvl] > 0.0 || self.mode.intra_none_skip[lvl] < f64::MAX
        {
            // The estimate per sample (in bits) tells flat areas, where the
            // whole block wins, from busy ones, where the split does.
            let bpp = self.est_block(x, y, n, n) / (self.enc.lambda * (n * n) as f64);
            if bpp < self.mode.intra_split_skip[lvl] {
                try_split = false;
            } else if bpp > self.mode.intra_none_skip[lvl] {
                try_none = false;
            }
            none_first = true;
        }
        if !try_none {
            self.node_try(nd, bsize, PARTITION_SPLIT)?;
            return Ok(true);
        }
        if !try_split {
            self.node_try(nd, bsize, PARTITION_NONE)?;
            return Ok(true);
        }
        let Some(s0) = self.node_save(nd, None) else {
            return Ok(false);
        };
        let (pa, pb) = if none_first {
            (PARTITION_NONE, PARTITION_SPLIT)
        } else {
            (PARTITION_SPLIT, PARTITION_NONE)
        };
        let saved_q = self.enc;
        if self.mode.inter_frame && pa == PARTITION_NONE {
            // The whole block's trial searches fewer transform types.
            self.enc.tx_type_search = self.enc.tx_type_search.min(self.mode.merge_tx_search);
        }
        let first = self.node_try(nd, bsize, pa);
        self.enc = saved_q;
        let first = first?;
        if pa == PARTITION_NONE
            && self.mode.inter_frame
            && (self.fs.mi[nd.r * self.fs.mi_stride + nd.c].skip()
                || (self.mode.merge_accept > 0.0 && {
                    let h = n / 2;
                    let es = self.est_block(x, y, h, h)
                        + self.est_block(x + h, y, h, h)
                        + self.est_block(x, y + h, h, h)
                        + self.est_block(x + h, y + h, h, h);
                    first < self.mode.merge_accept * es
                }))
        {
            // The whole block codes no residual, or costs well below the
            // split's estimate: keep it.
            return Ok(true);
        }
        let Some(s_first) = self.node_save(nd, Some(&s0.msac)) else {
            return Ok(true);
        };
        self.node_restore(nd, &s0);
        let second = self.node_try(nd, bsize, pb)?;
        let first_wins = if pa == PARTITION_NONE {
            first <= second
        } else {
            first < second
        };
        if first_wins {
            self.node_restore(nd, &s_first);
        }
        Ok(true)
    }
}
