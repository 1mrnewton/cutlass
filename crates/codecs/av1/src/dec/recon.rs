//! Prediction and reconstruction calls for one block (spec 7.11, 7.12).

use crate::dsp::Pixel;

use super::coder::Coder;
use crate::dsp::ipred::{self, EDGE_LEN, EDGE_OFF};
use crate::spec::tables::{AC_QLOOKUP, DC_QLOOKUP, QM_OFFSET, QUANTIZER_MATRIX};

use super::obu::SEG_LVL_ALT_Q;
use super::tile::{DequantInfo, TileDec};
use super::types::{
    INTRA_FRAME, SMOOTH_PRED, TX_16X32, TX_16X64, TX_32X16, TX_32X32, TX_64X16, TX_64X64, tx_h,
    tx_h_log2, tx_w, tx_w_log2,
};

/// Reusable buffers for one tile.
pub struct Scratch<P> {
    pub levels: Vec<u16>,
    /// Scan indices of the nonzero levels of the transform block being read,
    /// from the end of block back.
    pub nz_scan: Vec<u16>,
    /// Planned levels of the transform block being written (encoder).
    pub want: Vec<i32>,
    pub coefs: Vec<i32>,
    pub nz_rows: usize,
    pub nz_cols: usize,
    pub tx_type: u8,
    pub above: [P; EDGE_LEN],
    pub left: [P; EDGE_LEN],
    pub ac: Vec<i32>,
    pub tmp0: Vec<i16>,
    pub tmp1: Vec<i16>,
    pub mask: Vec<u8>,
    pub emu: Vec<P>,
    pub obmc: Vec<P>,
    /// Encoder work buffers (empty until the encoder first uses them).
    pub enc_i32: [Vec<i32>; 4],
    pub enc_px: [Vec<P>; 4],
    pub enc_q: Option<Box<crate::enc::quant::QuantScratch>>,
}

impl<P: Pixel> Scratch<P> {
    pub fn new() -> Self {
        Self {
            levels: vec![0; 36 * 36],
            nz_scan: vec![0; 1024],
            want: vec![0; 1024],
            coefs: vec![0; 64 * 64],
            nz_rows: 0,
            nz_cols: 0,
            tx_type: 0,
            above: [P::default(); EDGE_LEN],
            left: [P::default(); EDGE_LEN],
            ac: vec![0; 32 * 32],
            tmp0: vec![0; 128 * 128],
            tmp1: vec![0; 128 * 128],
            mask: vec![0; 128 * 128],
            emu: vec![P::default(); (128 * 2 + 16) * (128 * 2 + 16)],
            obmc: vec![P::default(); 128 * 128],
            enc_i32: Default::default(),
            enc_px: Default::default(),
            enc_q: None,
        }
    }
}

impl<P: Pixel> Scratch<P> {
    /// A placeholder that allocates nothing.
    pub fn empty() -> Self {
        Self {
            levels: Vec::new(),
            nz_scan: Vec::new(),
            want: Vec::new(),
            coefs: Vec::new(),
            nz_rows: 0,
            nz_cols: 0,
            tx_type: 0,
            above: [P::default(); EDGE_LEN],
            left: [P::default(); EDGE_LEN],
            ac: Vec::new(),
            tmp0: Vec::new(),
            tmp1: Vec::new(),
            mask: Vec::new(),
            emu: Vec::new(),
            obmc: Vec::new(),
            enc_i32: Default::default(),
            enc_px: Default::default(),
            enc_q: None,
        }
    }
}

impl<P: Pixel> Default for Scratch<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: Pixel, C: Coder> TileDec<'_, P, C> {
    /// get_qindex( 0, segment_id )
    fn block_qindex(&self) -> i32 {
        let seg = self.b.segment_id as usize;
        let h = self.hdr;
        if h.seg_feature_active(seg, SEG_LVL_ALT_Q) {
            let data = i32::from(h.segmentation.feature_data[seg][SEG_LVL_ALT_Q]);
            let q = if h.delta_q_present {
                self.current_q_index + data
            } else {
                i32::from(h.quant.base_q_idx) + data
            };
            q.clamp(0, 255)
        } else if h.delta_q_present {
            self.current_q_index
        } else {
            i32::from(h.quant.base_q_idx)
        }
    }

    pub fn dequant_info(&self, plane: usize, tx_type: u8, tx: u8) -> DequantInfo {
        let q = self.block_qindex();
        let bdi = ((self.bd - 8) >> 1) as usize;
        let qp = &self.hdr.quant;
        let (dc_delta, ac_delta) = match plane {
            0 => (qp.delta_q_y_dc, 0),
            1 => (qp.delta_q_u_dc, qp.delta_q_u_ac),
            _ => (qp.delta_q_v_dc, qp.delta_q_v_ac),
        };
        let dc = DC_QLOOKUP[bdi][(q + i32::from(dc_delta)).clamp(0, 255) as usize] as i32;
        let ac = AC_QLOOKUP[bdi][(q + i32::from(ac_delta)).clamp(0, 255) as usize] as i32;
        let denom = match tx {
            TX_32X32 | TX_16X32 | TX_32X16 | TX_16X64 | TX_64X16 => 2,
            TX_64X64 | 11 | 12 => 4,
            _ => 1,
        };
        let level = self.hdr.seg_qm_level[plane][self.b.segment_id as usize];
        let qm = if qp.using_qmatrix && tx_type < 9 && level < 15 {
            let off = QM_OFFSET[tx as usize] as usize;
            Some(&QUANTIZER_MATRIX[level as usize][usize::from(plane > 0)][off..])
        } else {
            None
        };
        DequantInfo { dc, ac, denom, qm }
    }

    fn filter_type(&self, plane: usize) -> bool {
        let b = &self.b;
        let is_smooth = |r: usize, c: usize| {
            let m = self.mi(r, c);
            let mode = if plane == 0 {
                m.y_mode
            } else {
                if m.ref_frame[0] > INTRA_FRAME {
                    return false;
                }
                self.fs.uv_modes[r * self.fs.mi_stride + c]
            };
            (SMOOTH_PRED..=SMOOTH_PRED + 2).contains(&mode)
        };
        let mut above = false;
        let mut left = false;
        if if plane == 0 {
            b.avail_u
        } else {
            b.avail_u_chroma
        } {
            let mut r = b.mi_row - 1;
            let mut c = b.mi_col;
            if plane > 0 {
                if self.sub_x && (b.mi_col & 1) == 0 {
                    c += 1;
                }
                if self.sub_y && (b.mi_row & 1) == 1 {
                    r -= 1;
                }
            }
            above = is_smooth(r, c);
        }
        if if plane == 0 {
            b.avail_l
        } else {
            b.avail_l_chroma
        } {
            let mut r = b.mi_row;
            let mut c = b.mi_col - 1;
            if plane > 0 {
                if self.sub_x && (b.mi_col & 1) == 1 {
                    c -= 1;
                }
                if self.sub_y && (b.mi_row & 1) == 0 {
                    r += 1;
                }
            }
            left = is_smooth(r, c);
        }
        above || left
    }

    /// Intra prediction process (7.11.2) into the current frame.
    pub fn predict_intra(
        &mut self,
        plane: usize,
        x: usize,
        y: usize,
        have_left: bool,
        have_above: bool,
        have_above_right: bool,
        have_below_left: bool,
        mode: u8,
        log2w: u32,
        log2h: u32,
    ) {
        let w = 1usize << log2w;
        let h = 1usize << log2h;
        let (sx, sy) = self.plane_sub(plane);
        let max_x = ((self.fs.mi_cols * 4) >> sx) - 1;
        let max_y = ((self.fs.mi_rows * 4) >> sy) - 1;
        let bd = self.bd;
        let filter_type = if self.seq.enable_intra_edge_filter {
            self.filter_type(plane)
        } else {
            false
        };
        let pl = &mut self.cur.planes[plane];
        let stride = pl.stride;
        let buf = &mut pl.data;
        let above = &mut self.scratch.above;
        let left = &mut self.scratch.left;
        let base = 1i32 << (bd - 1);
        let n = w + h;
        if !have_above && have_left {
            let v = buf[y * stride + x - 1];
            for i in 0..n {
                above[EDGE_OFF + i] = v;
            }
        } else if !have_above && !have_left {
            let v = P::from_i32(base - 1);
            for i in 0..n {
                above[EDGE_OFF + i] = v;
            }
        } else {
            let lim = max_x.min(x + if have_above_right { 2 * w } else { w } - 1);
            let row = &buf[(y - 1) * stride..];
            let avail = (lim + 1).saturating_sub(x).min(n);
            above[EDGE_OFF..EDGE_OFF + avail].copy_from_slice(&row[x..x + avail]);
            let last = row[lim];
            for i in avail..n {
                above[EDGE_OFF + i] = last;
            }
        }
        if !have_left && have_above {
            let v = buf[(y - 1) * stride + x];
            for i in 0..n {
                left[EDGE_OFF + i] = v;
            }
        } else if !have_left && !have_above {
            let v = P::from_i32(base + 1);
            for i in 0..n {
                left[EDGE_OFF + i] = v;
            }
        } else {
            let lim = max_y.min(y + if have_below_left { 2 * h } else { h } - 1);
            for i in 0..n {
                left[EDGE_OFF + i] = buf[(lim.min(y + i)) * stride + x - 1];
            }
        }
        let corner = if have_above && have_left {
            buf[(y - 1) * stride + x - 1]
        } else if have_above {
            buf[(y - 1) * stride + x]
        } else if have_left {
            buf[y * stride + x - 1]
        } else {
            P::from_i32(base)
        };
        above[EDGE_OFF - 1] = corner;
        left[EDGE_OFF - 1] = corner;
        let dst = &mut buf[y * stride + x..];
        if plane == 0 && self.b.use_filter_intra {
            ipred::filter_intra(dst, stride, w, h, self.b.filter_intra_mode, above, left, bd);
        } else {
            let angle = if plane == 0 {
                self.b.angle_delta_y
            } else {
                self.b.angle_delta_uv
            };
            let edge_w = w.min(max_x + 1 - x);
            let edge_h = h.min(max_y + 1 - y);
            ipred::predict_clipped(
                dst,
                stride,
                w,
                h,
                mode,
                angle,
                above,
                left,
                have_left,
                have_above,
                filter_type,
                self.seq.enable_intra_edge_filter,
                edge_w,
                edge_h,
                bd,
            );
        }
    }

    pub fn predict_cfl(&mut self, plane: usize, sx0: usize, sy0: usize, tx: u8) {
        let w = tx_w(tx);
        let h = tx_h(tx);
        let sub_x = self.sub_x;
        let sub_y = self.sub_y;
        let (sxs, sys) = (u32::from(sub_x), u32::from(sub_y));
        // Number of columns/rows whose luma position is not clamped by MaxLumaW/H.
        let max_w = self.b.max_luma_w;
        let max_h = self.b.max_luma_h;
        let valid_w = ((max_w.saturating_sub(sx0 << sxs) + (1 << sxs) - 1) >> sxs).clamp(1, w);
        let valid_h = ((max_h.saturating_sub(sy0 << sys) + (1 << sys) - 1) >> sys).clamp(1, h);
        let alpha = i32::from(if plane == 1 {
            self.b.cfl_alpha_u
        } else {
            self.b.cfl_alpha_v
        });
        let (luma, rest) = self.cur.planes.split_at_mut(1);
        let lp = &luma[0];
        let off = (sy0 << sys) * lp.stride + (sx0 << sxs);
        ipred::cfl_ac(
            &mut self.scratch.ac,
            &lp.data[off..],
            lp.stride,
            w,
            h,
            sub_x,
            sub_y,
            valid_w,
            valid_h,
        );
        let cp = &mut rest[plane - 1];
        let stride = cp.stride;
        ipred::cfl_pred(
            &mut cp.data[sy0 * stride + sx0..],
            stride,
            w,
            h,
            &self.scratch.ac,
            alpha,
            self.bd,
        );
    }

    pub fn predict_palette(
        &mut self,
        plane: usize,
        sx0: usize,
        sy0: usize,
        x: usize,
        y: usize,
        tx: u8,
    ) {
        let w = tx_w(tx);
        let h = tx_h(tx);
        let palette = match plane {
            0 => self.b.palette_y,
            1 => self.b.palette_u,
            _ => self.b.palette_v,
        };
        let (map, mstride) = if plane == 0 {
            (&self.color_map_y, super::types::block_width(self.b.bsize))
        } else {
            let mut bw = super::types::block_width(self.b.bsize) >> u32::from(self.sub_x);
            if bw < 4 {
                bw += 2;
            }
            (&self.color_map_uv, bw)
        };
        let pl = &mut self.cur.planes[plane];
        let stride = pl.stride;
        for i in 0..h {
            let row = &mut pl.data[(sy0 + i) * stride + sx0..][..w];
            let mrow = &map[(y * 4 + i) * mstride + x * 4..];
            for j in 0..w {
                row[j] = P::from_i32(i32::from(palette[mrow[j] as usize]));
            }
        }
    }

    /// Reconstruct process (7.12.3): inverse transform of the dequantized
    /// coefficients, added to the prediction.
    pub fn reconstruct(&mut self, plane: usize, x: usize, y: usize, tx: u8) {
        let pl = &mut self.cur.planes[plane];
        let stride = pl.stride;
        let _ = (tx_w_log2(tx), tx_h_log2(tx));
        crate::dsp::itx::inv_txfm_add(
            &mut pl.data[y * stride + x..],
            stride,
            &mut self.scratch.coefs,
            tx as usize,
            self.scratch.tx_type as usize,
            self.scratch.nz_rows,
            self.scratch.nz_cols,
            self.b.lossless,
            self.bd,
        );
    }
}
