//! One profile-0 picture: partitions, DC intra, integer motion, and residuals.
//!
//! Frames are error resilient, so probabilities stay at the defaults. The
//! partition walk matches `decode_partition`. A block is coded whole only
//! when it sits inside the frame; everything else splits.

use cutlass_decode::CodecError;

use super::coef::{TxTokens, write_tokens};
use crate::bits::BoolWriter;
use crate::fwd::{fdct4x4, fdct8x8, fwht4x4, quantize};
use crate::geom::{
    self, BLOCK_64X64, DC, NEW, PARTITION_NONE, PARTITION_SPLIT, TX_4X4, TX_8X8, ZERO, max_tx,
    mi_width_log2, partition_context_byte, subsize, uv_tx_420,
};
use crate::intra::{self, Site};
use crate::prob::{self, FrameContext, write_tree};
use crate::recon::{Job, RefPlane, Reference, render};
use crate::tables::{
    AC_QLOOKUP, DC_QLOOKUP, KF_UV_MODE_PROB, KF_Y_MODE_PROB, NEIGHBORS_DEFAULT_4X4,
    NEIGHBORS_DEFAULT_8X8, SCAN_DEFAULT_4X4, SCAN_DEFAULT_8X8,
};
use crate::txfm::{iht_add, iwht4x4_add};

const MODE_COUNTER: [i32; 14] = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 3, 1];
const COUNTER_CTX: [u8; 19] = [2, 3, 4, 1, 3, 9, 0, 9, 9, 5, 5, 9, 5, 9, 9, 9, 9, 9, 6];

const MV_REF: [[[i8; 2]; 8]; 13] = [
    near8(),
    near8(),
    near8(),
    near8(),
    [
        [0, -1],
        [-1, 0],
        [1, -1],
        [-1, -1],
        [0, -2],
        [-2, 0],
        [-2, -1],
        [-1, -2],
    ],
    [
        [-1, 0],
        [0, -1],
        [-1, 1],
        [-1, -1],
        [-2, 0],
        [0, -2],
        [-1, -2],
        [-2, -1],
    ],
    [
        [-1, 0],
        [0, -1],
        [-1, 1],
        [1, -1],
        [-1, -1],
        [-3, 0],
        [0, -3],
        [-3, -3],
    ],
    [
        [0, -1],
        [-1, 0],
        [2, -1],
        [-1, -1],
        [-1, 1],
        [0, -3],
        [-3, 0],
        [-3, -3],
    ],
    [
        [-1, 0],
        [0, -1],
        [-1, 2],
        [-1, -1],
        [1, -1],
        [-3, 0],
        [0, -3],
        [-3, -3],
    ],
    [
        [-1, 1],
        [1, -1],
        [-1, 2],
        [2, -1],
        [-1, -1],
        [-3, 0],
        [0, -3],
        [-3, -3],
    ],
    [
        [0, -1],
        [-1, 0],
        [4, -1],
        [-1, 2],
        [-1, -1],
        [0, -3],
        [-3, 0],
        [2, -1],
    ],
    [
        [-1, 0],
        [0, -1],
        [-1, 4],
        [2, -1],
        [-1, -1],
        [-3, 0],
        [0, -3],
        [-1, 2],
    ],
    [
        [-1, 3],
        [3, -1],
        [-1, 4],
        [4, -1],
        [-1, -1],
        [-1, 0],
        [0, -1],
        [-1, 6],
    ],
];

const fn near8() -> [[i8; 2]; 8] {
    [
        [-1, 0],
        [0, -1],
        [-1, -1],
        [-2, 0],
        [0, -2],
        [-2, -1],
        [-1, -2],
        [-2, -2],
    ]
}

#[derive(Clone, Copy)]
struct Cell {
    skip: bool,
    intra: bool,
    mode: u8,
    mv_row: i16,
    mv_col: i16,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            skip: false,
            intra: true,
            mode: DC,
            mv_row: 0,
            mv_col: 0,
        }
    }
}

struct Plane {
    samples: Vec<u8>,
    stride: usize,
    width: usize,
    height: usize,
}

pub struct Coded {
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
    pub tile: Vec<u8>,
}

pub struct Source<'a> {
    pub y: &'a [u8],
    pub u: &'a [u8],
    pub v: &'a [u8],
    pub y_stride: usize,
    pub u_stride: usize,
    pub v_stride: usize,
    pub width: u32,
    pub height: u32,
}

struct JobState<'a> {
    source: Source<'a>,
    reference: Option<&'a Reference>,
    lossless: bool,
    qindex: usize,
    key: bool,
    fc: FrameContext,
    tile: BoolWriter,
    pending: Vec<TxTokens>,
    y: Plane,
    u: Plane,
    v: Plane,
    grid: Vec<Cell>,
    mi_cols: usize,
    mi_rows: usize,
    above_part: Vec<u8>,
    left_part: [u8; 8],
    above_y: Vec<u8>,
    above_u: Vec<u8>,
    above_v: Vec<u8>,
    left_y: [u8; 16],
    left_u: [u8; 8],
    left_v: [u8; 8],
}

pub fn code_frame(
    source: Source<'_>,
    reference: Option<&Reference>,
    lossless: bool,
    qindex: u8,
) -> Result<Coded, CodecError> {
    let width = usize::try_from(source.width).unwrap_or(0);
    let height = usize::try_from(source.height).unwrap_or(0);
    let mi_cols = geom::mi_count(source.width);
    let mi_rows = geom::mi_count(source.height);
    let aligned = mi_cols.div_ceil(8) * 8;
    let chroma_w = width / 2;
    let chroma_h = height / 2;
    let mut job = JobState {
        source,
        reference,
        lossless,
        qindex: usize::from(qindex),
        key: reference.is_none(),
        fc: FrameContext::defaults(),
        tile: BoolWriter::new(),
        pending: Vec::new(),
        y: Plane::blank(width, height)?,
        u: Plane::blank(chroma_w, chroma_h)?,
        v: Plane::blank(chroma_w, chroma_h)?,
        grid: vec![Cell::default(); mi_cols.saturating_mul(mi_rows)],
        mi_cols,
        mi_rows,
        above_part: vec![0; aligned],
        left_part: [0; 8],
        above_y: vec![0; aligned * 2],
        above_u: vec![0; aligned],
        above_v: vec![0; aligned],
        left_y: [0; 16],
        left_u: [0; 8],
        left_v: [0; 8],
    };
    job.walk()?;
    let tile = job.tile.finish()?;
    Ok(Coded {
        y: job.y.samples,
        u: job.u.samples,
        v: job.v.samples,
        tile,
    })
}

impl Plane {
    fn blank(width: usize, height: usize) -> Result<Self, CodecError> {
        let samples = vec![0u8; width.checked_mul(height).ok_or(limit())?];
        Ok(Self {
            samples,
            stride: width,
            width,
            height,
        })
    }
}

impl JobState<'_> {
    fn walk(&mut self) -> Result<(), CodecError> {
        let mut row = 0usize;
        while row < self.mi_rows {
            self.left_part = [0; 8];
            self.left_y = [0; 16];
            self.left_u = [0; 8];
            self.left_v = [0; 8];
            let mut col = 0usize;
            while col < self.mi_cols {
                self.partition(row, col, BLOCK_64X64, 4)?;
                col += 8;
            }
            row += 8;
        }
        Ok(())
    }

    fn partition(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        bsize: u8,
        n4: u32,
    ) -> Result<(), CodecError> {
        if mi_row >= self.mi_rows || mi_col >= self.mi_cols {
            return Ok(());
        }
        let n8 = n4 - 1;
        let hbs = (1usize << n8) >> 1;
        let has_rows = mi_row + hbs < self.mi_rows;
        let has_cols = mi_col + hbs < self.mi_cols;
        let partition = self.choose(mi_row, mi_col, has_rows, has_cols, n4);
        self.write_partition(mi_row, mi_col, has_rows, has_cols, n8, partition)?;
        let sub = subsize(partition, bsize);
        if hbs == 0 {
            self.encode_block(mi_row, mi_col, sub, 1, 1)?;
        } else {
            self.descend(Descent {
                mi_row,
                mi_col,
                partition,
                sub,
                n4,
                n8,
                hbs,
                has_rows,
                has_cols,
            })?;
        }
        if bsize >= geom::BLOCK_8X8 && (bsize == geom::BLOCK_8X8 || partition != PARTITION_SPLIT) {
            self.update_partition(mi_row, mi_col, sub, bsize)?;
        }
        Ok(())
    }

    fn choose(&self, mi_row: usize, mi_col: usize, has_rows: bool, has_cols: bool, n4: u32) -> u8 {
        let block = 1usize << (n4 - 1);
        let fits = mi_row + block <= self.mi_rows && mi_col + block <= self.mi_cols;
        if has_rows && has_cols && fits {
            PARTITION_NONE
        } else {
            PARTITION_SPLIT
        }
    }

    fn descend(&mut self, split: Descent) -> Result<(), CodecError> {
        let Descent {
            mi_row,
            mi_col,
            partition,
            sub,
            n4,
            n8,
            hbs,
            has_rows,
            has_cols,
        } = split;
        match partition {
            PARTITION_NONE => self.encode_block(mi_row, mi_col, sub, n4, n4),
            geom::PARTITION_HORZ => {
                self.encode_block(mi_row, mi_col, sub, n4, n8)?;
                if has_rows {
                    self.encode_block(mi_row + hbs, mi_col, sub, n4, n8)?;
                }
                Ok(())
            }
            geom::PARTITION_VERT => {
                self.encode_block(mi_row, mi_col, sub, n8, n4)?;
                if has_cols {
                    self.encode_block(mi_row, mi_col + hbs, sub, n8, n4)?;
                }
                Ok(())
            }
            _ => {
                self.partition(mi_row, mi_col, sub, n8)?;
                self.partition(mi_row, mi_col + hbs, sub, n8)?;
                self.partition(mi_row + hbs, mi_col, sub, n8)?;
                self.partition(mi_row + hbs, mi_col + hbs, sub, n8)
            }
        }
    }

    fn write_partition(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        has_rows: bool,
        has_cols: bool,
        bsl: u32,
        partition: u8,
    ) -> Result<(), CodecError> {
        if !has_rows && !has_cols {
            return Ok(());
        }
        let ctx = self.partition_ctx(mi_row, mi_col, bsl);
        let probs = if self.key {
            prob::kf_partition()[ctx]
        } else {
            self.fc.partition[ctx]
        };
        if has_rows && has_cols {
            write_tree(&mut self.tile, &prob::PARTITION_TREE, &probs, partition);
        } else if !has_rows {
            let bit = u32::from(partition == PARTITION_SPLIT);
            self.tile.write(bit, u32::from(probs[1]));
        } else {
            let bit = u32::from(partition == PARTITION_SPLIT);
            self.tile.write(bit, u32::from(probs[2]));
        }
        Ok(())
    }

    fn partition_ctx(&self, mi_row: usize, mi_col: usize, bsl: u32) -> usize {
        let above = usize::from(self.above_part.get(mi_col).copied().unwrap_or(0) >> bsl) & 1;
        let left = usize::from(self.left_part[mi_row & 7] >> bsl) & 1;
        left * 2 + above + (bsl as usize) * 4
    }

    fn update_partition(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        subsize: u8,
        parent: u8,
    ) -> Result<(), CodecError> {
        let n = 1usize << mi_width_log2(parent);
        let (above, left) = partition_context_byte(subsize);
        let above_end = mi_col + n;
        let row = self
            .above_part
            .get_mut(mi_col..above_end)
            .ok_or(bad("partition context"))?;
        row.fill(above);
        let left_at = mi_row & 7;
        let column = self
            .left_part
            .get_mut(left_at..left_at + n)
            .ok_or(bad("partition context"))?;
        column.fill(left);
        Ok(())
    }

    fn encode_block(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        bsize: u8,
        bwl: u32,
        bhl: u32,
    ) -> Result<(), CodecError> {
        let bw = 1usize << (bwl - 1);
        let bh = 1usize << (bhl - 1);
        let motion = if self.key {
            None
        } else {
            Some(self.search(mi_row, mi_col, bsize, bw, bh))
        };
        let any = self.reconstruct(mi_row, mi_col, bsize, bw, bh, motion)?;
        self.write_mode(mi_row, mi_col, bsize, !any, motion)?;
        if any {
            self.write_pending()?;
        } else {
            self.pending.clear();
        }
        self.store_cell(mi_row, mi_col, bw, bh, !any, motion);
        Ok(())
    }

    fn write_mode(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        bsize: u8,
        skip: bool,
        motion: Option<MotionPick>,
    ) -> Result<(), CodecError> {
        let ctx = self.skip_ctx(mi_row, mi_col);
        self.tile
            .write(u32::from(skip), u32::from(self.fc.skip[ctx]));
        if self.key {
            let above = self.edge_mode(mi_row.wrapping_sub(1), mi_col, mi_row > 0);
            let left = self.edge_mode(mi_row, mi_col.wrapping_sub(1), mi_col > 0);
            write_tree(
                &mut self.tile,
                &prob::INTRA_TREE,
                &KF_Y_MODE_PROB[usize::from(above)][usize::from(left)],
                DC,
            );
            write_tree(
                &mut self.tile,
                &prob::INTRA_TREE,
                &KF_UV_MODE_PROB[usize::from(DC)],
                DC,
            );
            return Ok(());
        }
        let inter_ctx = self.inter_ctx(mi_row, mi_col);
        self.tile
            .write(1, u32::from(self.fc.intra_inter[inter_ctx]));
        let ref_ctx = self.last_ctx(mi_row, mi_col);
        self.tile
            .write(0, u32::from(self.fc.single_ref[ref_ctx][0]));
        let mode_ctx = self.mode_ctx(bsize, mi_row, mi_col)?;
        let pick = motion.unwrap_or(MotionPick {
            new_mv: false,
            row: 0,
            col: 0,
        });
        let symbol = if pick.new_mv {
            NEW - geom::NEAREST
        } else {
            ZERO - geom::NEAREST
        };
        write_tree(
            &mut self.tile,
            &prob::INTER_TREE,
            &self.fc.inter_mode[mode_ctx],
            symbol,
        );
        if pick.new_mv {
            let predictor = self.predictor(bsize, mi_row, mi_col);
            write_mv(&mut self.tile, &self.fc, predictor, pick.row, pick.col);
        }
        Ok(())
    }

    fn write_pending(&mut self) -> Result<(), CodecError> {
        let pending = std::mem::take(&mut self.pending);
        for tx in &pending {
            let (scan, neighbors) = scans(tx.tx);
            write_tokens(&mut self.tile, &self.fc, tx, scan, neighbors);
        }
        Ok(())
    }

    fn reconstruct(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        bsize: u8,
        bw: usize,
        bh: usize,
        motion: Option<MotionPick>,
    ) -> Result<bool, CodecError> {
        self.pending.clear();
        if let Some(pick) = motion {
            self.compensate(mi_row, mi_col, bw, bh, pick)?;
        }
        let luma_tx = if self.lossless {
            TX_4X4
        } else {
            max_tx(bsize).min(TX_8X8)
        };
        let mut any = false;
        for plane in 0..3 {
            let tx = if plane == 0 {
                luma_tx
            } else {
                uv_tx_420(bsize, luma_tx)
            };
            if self.plane_residual(PlanePass {
                plane,
                mi_row,
                mi_col,
                bw,
                bh,
                tx,
                inter: motion.is_some(),
            })? {
                any = true;
            }
        }
        Ok(any)
    }

    fn plane_residual(&mut self, pass: PlanePass) -> Result<bool, CodecError> {
        let ss = u32::from(pass.plane > 0);
        let n4_w = (pass.bw << 1) >> ss;
        let n4_h = (pass.bh << 1) >> ss;
        let step = 1usize << pass.tx;
        let mut any = false;
        let mut row = 0usize;
        while row < n4_h {
            let mut col = 0usize;
            while col < n4_w {
                if self.one_tx(TxSite {
                    plane: pass.plane,
                    mi_row: pass.mi_row,
                    mi_col: pass.mi_col,
                    tx: pass.tx,
                    row,
                    col,
                    n4_w,
                    inter: pass.inter,
                })? {
                    any = true;
                }
                col += step;
            }
            row += step;
        }
        Ok(any)
    }

    fn one_tx(&mut self, site: TxSite) -> Result<bool, CodecError> {
        let ss = u32::from(site.plane > 0);
        let x = ((site.mi_col << 3) >> ss) + site.col * 4;
        let y = ((site.mi_row << 3) >> ss) + site.row * 4;
        let n = 4usize << site.tx;
        if !site.inter {
            self.predict_intra(&site, x, y)?;
        }
        let ctx = self.coef_ctx(&site)?;
        let tokens = self.tokens(site.plane, x, y, n)?;
        let nonzero = tokens.iter().any(|token| *token != 0);
        self.add_inverse(site.plane, x, y, n, &tokens)?;
        self.paint(&site, u8::from(nonzero))?;
        self.pending.push(TxTokens {
            tx: site.tx,
            plane: site.plane,
            inter: site.inter,
            ctx,
            tokens,
        });
        Ok(nonzero)
    }

    fn predict_intra(&mut self, site: &TxSite, x: usize, y: usize) -> Result<(), CodecError> {
        let (crop_w, crop_h, stride, samples) = self.plane_parts(site.plane);
        intra::predict(
            DC,
            samples,
            Site {
                tx: site.tx,
                stride,
                crop_w,
                crop_h,
                x,
                y,
                have_left: site.col > 0 || site.mi_col > 0,
                have_above: site.row > 0 || site.mi_row > 0,
                have_right: site.col + (1usize << site.tx) < site.n4_w,
            },
        );
        Ok(())
    }

    fn tokens(&self, plane: usize, x: usize, y: usize, n: usize) -> Result<Vec<i32>, CodecError> {
        let mut residual = vec![0i16; n * n];
        for row in 0..n {
            for col in 0..n {
                let src = self.source_at(plane, x + col, y + row)?;
                let pred = self.recon_at(plane, x + col, y + row)?;
                let diff = i32::from(src) - i32::from(pred);
                residual[row * n + col] = i16::try_from(diff).unwrap_or(0);
            }
        }
        let mut coef = vec![0i32; n * n];
        if self.lossless {
            fwht4x4(&residual, n, &mut coef);
            return Ok(coef.iter().map(|value| value / 4).collect());
        }
        if n == 4 {
            fdct4x4(&residual, n, &mut coef);
        } else {
            fdct8x8(&residual, n, &mut coef);
        }
        let (dc, ac) = self.quant_steps();
        let mut tokens = vec![0i32; n * n];
        for (index, (token, value)) in tokens.iter_mut().zip(&coef).enumerate() {
            let step = if index == 0 { dc } else { ac };
            *token = quantize(*value, step);
        }
        Ok(tokens)
    }

    fn add_inverse(
        &mut self,
        plane: usize,
        x: usize,
        y: usize,
        n: usize,
        tokens: &[i32],
    ) -> Result<(), CodecError> {
        let (dc, ac) = if self.lossless {
            (4i32, 4i32)
        } else {
            let (dc, ac) = self.quant_steps();
            (dc, ac)
        };
        let mut coef = vec![0i32; n * n];
        for (index, token) in tokens.iter().enumerate() {
            let step = if index == 0 { dc } else { ac };
            coef[index] = token.saturating_mul(step);
        }
        let lossless = self.lossless;
        let stride = self.stride(plane);
        let start = y * stride + x;
        let dest = self.plane_samples(plane);
        let slot = dest.get_mut(start..).ok_or(bad("inverse plane"))?;
        if lossless {
            iwht4x4_add(&coef, slot, stride);
        } else {
            iht_add(&coef, slot, stride, n, 0);
        }
        Ok(())
    }

    fn quant_steps(&self) -> (i32, i32) {
        (
            i32::from(DC_QLOOKUP[self.qindex]),
            i32::from(AC_QLOOKUP[self.qindex]),
        )
    }

    fn compensate(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        bw: usize,
        bh: usize,
        pick: MotionPick,
    ) -> Result<(), CodecError> {
        let reference = self.reference.ok_or(bad("reference"))?;
        for plane in 0..3 {
            let ss = u32::from(plane > 0);
            let x = (mi_col << 3) >> ss;
            let y = (mi_row << 3) >> ss;
            let w = (bw << 3) >> ss;
            let h = (bh << 3) >> ss;
            let scale = if plane == 0 { 2 } else { 1 };
            let stride = self.stride(plane);
            let start = y * stride + x;
            let pred = self
                .plane_samples(plane)
                .get_mut(start..)
                .ok_or(bad("motion sample"))?;
            let ref_plane = match plane {
                0 => &reference.y,
                1 => &reference.u,
                _ => &reference.v,
            };
            render(Job {
                pred,
                pred_stride: stride,
                w,
                h,
                origin_x: i32::try_from(x).unwrap_or(0),
                origin_y: i32::try_from(y).unwrap_or(0),
                mv_row: i32::from(pick.row) * scale,
                mv_col: i32::from(pick.col) * scale,
                filter: usize::from(geom::EIGHTTAP),
                reference: ref_plane,
            })?;
        }
        Ok(())
    }

    fn search(&self, mi_row: usize, mi_col: usize, _bsize: u8, bw: usize, bh: usize) -> MotionPick {
        let x = mi_col * 8;
        let y = mi_row * 8;
        let w = bw * 8;
        let h = bh * 8;
        let zero = self.sad(x, y, w, h, 0, 0);
        let mut best_sad = zero;
        let mut best_row = 0i16;
        let mut best_col = 0i16;
        for dy in -8..=8 {
            for dx in -8..=8 {
                let row = i16::try_from(dy * 8).unwrap_or(0);
                let col = i16::try_from(dx * 8).unwrap_or(0);
                let score = self.sad(x, y, w, h, row, col);
                if score < best_sad {
                    best_sad = score;
                    best_row = row;
                    best_col = col;
                }
            }
        }
        MotionPick {
            new_mv: best_row != 0 || best_col != 0,
            row: best_row,
            col: best_col,
        }
    }

    fn sad(&self, x: usize, y: usize, w: usize, h: usize, mv_row: i16, mv_col: i16) -> u64 {
        let Some(reference) = self.reference else {
            return u64::MAX;
        };
        let dx = i32::from(mv_col) / 8;
        let dy = i32::from(mv_row) / 8;
        let mut sum = 0u64;
        for row in 0..h {
            for col in 0..w {
                let src = self.source_at(0, x + col, y + row).unwrap_or(0);
                let sample = sample_ref(
                    &reference.y,
                    x as i32 + dx + col as i32,
                    y as i32 + dy + row as i32,
                );
                let diff = i32::from(src) - i32::from(sample);
                sum += u64::from(diff.unsigned_abs());
            }
        }
        sum
    }

    fn paint(&mut self, site: &TxSite, on: u8) -> Result<(), CodecError> {
        let (above_at, left_at, n) = coef_span(site);
        let (above, left) = self.context_pair(site.plane);
        fill(above, above_at, n, on)?;
        fill(left, left_at, n, on)
    }

    fn coef_ctx(&self, site: &TxSite) -> Result<usize, CodecError> {
        let (above_at, left_at, n) = coef_span(site);
        let above_nz = region_nz(self.above(site.plane), above_at, n)?;
        let left_nz = region_nz(self.left(site.plane), left_at, n)?;
        Ok(usize::from(above_nz) + usize::from(left_nz))
    }

    fn context_pair(&mut self, plane: usize) -> (&mut [u8], &mut [u8]) {
        match plane {
            0 => (&mut self.above_y, &mut self.left_y),
            1 => (&mut self.above_u, &mut self.left_u),
            _ => (&mut self.above_v, &mut self.left_v),
        }
    }

    fn above(&self, plane: usize) -> &[u8] {
        match plane {
            0 => &self.above_y,
            1 => &self.above_u,
            _ => &self.above_v,
        }
    }

    fn left(&self, plane: usize) -> &[u8] {
        match plane {
            0 => &self.left_y,
            1 => &self.left_u,
            _ => &self.left_v,
        }
    }

    fn store_cell(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        bw: usize,
        bh: usize,
        skip: bool,
        motion: Option<MotionPick>,
    ) {
        let pick = motion.unwrap_or(MotionPick {
            new_mv: false,
            row: 0,
            col: 0,
        });
        let cell = Cell {
            skip,
            intra: motion.is_none(),
            mode: if motion.is_none() {
                DC
            } else if pick.new_mv {
                NEW
            } else {
                ZERO
            },
            mv_row: if pick.new_mv { pick.row } else { 0 },
            mv_col: if pick.new_mv { pick.col } else { 0 },
        };
        for y in 0..bh {
            for x in 0..bw {
                let index = (mi_row + y) * self.mi_cols + mi_col + x;
                if let Some(slot) = self.grid.get_mut(index) {
                    *slot = cell;
                }
            }
        }
    }

    fn skip_ctx(&self, mi_row: usize, mi_col: usize) -> usize {
        let above = usize::from(mi_row > 0 && self.cell(mi_row - 1, mi_col).skip);
        let left = usize::from(mi_col > 0 && self.cell(mi_row, mi_col - 1).skip);
        above + left
    }

    fn edge_mode(&self, mi_row: usize, mi_col: usize, present: bool) -> u8 {
        if !present {
            return DC;
        }
        let cell = self.cell(mi_row, mi_col);
        if cell.intra { cell.mode } else { DC }
    }

    fn inter_ctx(&self, mi_row: usize, mi_col: usize) -> usize {
        let above = (mi_row > 0).then(|| self.cell(mi_row - 1, mi_col));
        let left = (mi_col > 0).then(|| self.cell(mi_row, mi_col - 1));
        match (above, left) {
            (Some(above), Some(left)) if above.intra && left.intra => 3,
            (Some(above), Some(left)) if above.intra || left.intra => 1,
            (Some(edge), None) | (None, Some(edge)) if edge.intra => 2,
            _ => 0,
        }
    }

    fn last_ctx(&self, mi_row: usize, mi_col: usize) -> usize {
        let above_last = mi_row > 0 && !self.cell(mi_row - 1, mi_col).intra;
        let left_last = mi_col > 0 && !self.cell(mi_row, mi_col - 1).intra;
        if above_last || left_last { 4 } else { 2 }
    }

    fn mode_ctx(&self, bsize: u8, mi_row: usize, mi_col: usize) -> Result<usize, CodecError> {
        let mut sum = 0i32;
        for pos in MV_REF[usize::from(bsize)].iter().take(2) {
            if let Some(cell) = self.neighbor(mi_row, mi_col, *pos) {
                sum += MODE_COUNTER[usize::from(cell.mode)];
            }
        }
        let mapped = COUNTER_CTX[usize::try_from(sum).unwrap_or(0)];
        if mapped > 6 {
            return Err(bad("mode context"));
        }
        Ok(usize::from(mapped))
    }

    fn predictor(&self, bsize: u8, mi_row: usize, mi_col: usize) -> (i16, i16) {
        let mut found: Option<(i16, i16)> = None;
        for pos in MV_REF[usize::from(bsize)] {
            let Some(cell) = self.neighbor(mi_row, mi_col, pos) else {
                continue;
            };
            if cell.intra {
                continue;
            }
            found = Some((cell.mv_row, cell.mv_col));
            break;
        }
        let (row, col) = found.unwrap_or((0, 0));
        let (bw, bh) = (bw_of(bsize), bh_of(bsize));
        clamp_mv(
            BlockMv {
                row,
                col,
                mi_row,
                mi_col,
                bh,
                bw,
            },
            self.mi_rows,
            self.mi_cols,
        )
    }

    fn neighbor(&self, mi_row: usize, mi_col: usize, pos: [i8; 2]) -> Option<&Cell> {
        let row = mi_row as isize + isize::from(pos[0]);
        let col = mi_col as isize + isize::from(pos[1]);
        if row < 0 || col < 0 || row >= self.mi_rows as isize || col >= self.mi_cols as isize {
            return None;
        }
        let row = usize::try_from(row).ok()?;
        let col = usize::try_from(col).ok()?;
        Some(self.cell(row, col))
    }

    fn cell(&self, mi_row: usize, mi_col: usize) -> &Cell {
        &self.grid[mi_row * self.mi_cols + mi_col]
    }

    fn source_at(&self, plane: usize, x: usize, y: usize) -> Result<u8, CodecError> {
        let (samples, stride) = match plane {
            0 => (self.source.y, self.source.y_stride),
            1 => (self.source.u, self.source.u_stride),
            _ => (self.source.v, self.source.v_stride),
        };
        samples
            .get(y * stride + x)
            .copied()
            .ok_or(bad("source sample"))
    }

    fn recon_at(&self, plane: usize, x: usize, y: usize) -> Result<u8, CodecError> {
        let plane = match plane {
            0 => &self.y,
            1 => &self.u,
            _ => &self.v,
        };
        plane
            .samples
            .get(y * plane.stride + x)
            .copied()
            .ok_or(bad("recon sample"))
    }

    fn plane_parts(&mut self, plane: usize) -> (usize, usize, usize, &mut [u8]) {
        let plane = match plane {
            0 => &mut self.y,
            1 => &mut self.u,
            _ => &mut self.v,
        };
        (plane.width, plane.height, plane.stride, &mut plane.samples)
    }

    fn plane_samples(&mut self, plane: usize) -> &mut [u8] {
        match plane {
            0 => &mut self.y.samples,
            1 => &mut self.u.samples,
            _ => &mut self.v.samples,
        }
    }

    fn stride(&self, plane: usize) -> usize {
        match plane {
            0 => self.y.stride,
            1 => self.u.stride,
            _ => self.v.stride,
        }
    }
}

struct Descent {
    mi_row: usize,
    mi_col: usize,
    partition: u8,
    sub: u8,
    n4: u32,
    n8: u32,
    hbs: usize,
    has_rows: bool,
    has_cols: bool,
}

struct PlanePass {
    plane: usize,
    mi_row: usize,
    mi_col: usize,
    bw: usize,
    bh: usize,
    tx: u8,
    inter: bool,
}

#[derive(Clone, Copy)]
struct TxSite {
    plane: usize,
    mi_row: usize,
    mi_col: usize,
    tx: u8,
    row: usize,
    col: usize,
    n4_w: usize,
    inter: bool,
}

struct BlockMv {
    row: i16,
    col: i16,
    mi_row: usize,
    mi_col: usize,
    bh: usize,
    bw: usize,
}

#[derive(Clone, Copy)]
struct MotionPick {
    new_mv: bool,
    row: i16,
    col: i16,
}

fn write_mv(w: &mut BoolWriter, fc: &FrameContext, predictor: (i16, i16), row: i16, col: i16) {
    let diff_row = i32::from(row) - i32::from(predictor.0);
    let diff_col = i32::from(col) - i32::from(predictor.1);
    let joint = if diff_row == 0 {
        u8::from(diff_col != 0)
    } else if diff_col == 0 {
        2
    } else {
        3
    };
    write_tree(w, &prob::JOINT_TREE, &fc.mv.joints, joint);
    let use_hp = predictor.0.unsigned_abs() < 64 && predictor.1.unsigned_abs() < 64;
    if joint == 2 || joint == 3 {
        write_component(w, &fc.mv.comp[0], diff_row, use_hp);
    }
    if joint == 1 || joint == 3 {
        write_component(w, &fc.mv.comp[1], diff_col, use_hp);
    }
}

fn write_component(w: &mut BoolWriter, comp: &prob::MvComp, value: i32, use_hp: bool) {
    w.write(u32::from(value < 0), u32::from(comp.sign));
    let z = value.unsigned_abs() - 1;
    let class = mv_class(z);
    write_tree(w, &prob::CLASS_TREE, &comp.classes, class);
    let base = if class == 0 {
        0
    } else {
        2u32 << (u32::from(class) + 2)
    };
    let offset = z - base;
    let integer = offset >> 3;
    let frac = u8::try_from((offset >> 1) & 3).unwrap_or(0);
    let hp = offset & 1;
    if class == 0 {
        w.write(integer, u32::from(comp.class0));
        let row = usize::try_from(integer).unwrap_or(0);
        write_tree(w, &prob::FP_TREE, &comp.class0_fp[row], frac);
    } else {
        let bits = usize::from(class);
        for (index, prob) in comp.bits.iter().take(bits).enumerate() {
            let bit = (integer >> index) & 1;
            w.write(bit, u32::from(*prob));
        }
        write_tree(w, &prob::FP_TREE, &comp.fp, frac);
    }
    if use_hp {
        let prob = if class == 0 { comp.class0_hp } else { comp.hp };
        w.write(hp, u32::from(prob));
    }
}

fn mv_class(z: u32) -> u8 {
    if z >= 8_192 {
        return 10;
    }
    let index = z >> 3;
    if index <= 1 {
        0
    } else {
        u8::try_from(31 - index.leading_zeros()).unwrap_or(10)
    }
}

fn clamp_mv(block: BlockMv, mi_rows: usize, mi_cols: usize) -> (i16, i16) {
    let mi_row = i32::try_from(block.mi_row).unwrap_or(i32::MAX);
    let mi_col = i32::try_from(block.mi_col).unwrap_or(i32::MAX);
    let bh = i32::try_from(block.bh).unwrap_or(i32::MAX);
    let bw = i32::try_from(block.bw).unwrap_or(i32::MAX);
    let rows = i32::try_from(mi_rows).unwrap_or(i32::MAX);
    let cols = i32::try_from(mi_cols).unwrap_or(i32::MAX);
    let top = -(mi_row.saturating_mul(64)) - 128;
    let bottom = (rows - bh - mi_row).saturating_mul(64) + 128;
    let left = -(mi_col.saturating_mul(64)) - 128;
    let right = (cols - bw - mi_col).saturating_mul(64) + 128;
    (
        fit(i32::from(block.row), top, bottom),
        fit(i32::from(block.col), left, right),
    )
}

fn coef_span(site: &TxSite) -> (usize, usize, usize) {
    let ss = usize::from(site.plane > 0);
    let n = 1usize << site.tx;
    let above_at = ((site.mi_col * 2) >> ss) + site.col;
    let left_at = (((site.mi_row * 2) & 15) >> ss) + site.row;
    (above_at, left_at, n)
}

fn fit(value: i32, min: i32, max: i32) -> i16 {
    let (min, max) = if min <= max { (min, max) } else { (max, min) };
    let clamped = value.clamp(min, max);
    i16::try_from(clamped.clamp(i32::from(i16::MIN), i32::from(i16::MAX))).unwrap_or(0)
}

fn sample_ref(plane: &RefPlane, x: i32, y: i32) -> u8 {
    if plane.width == 0 || plane.height == 0 {
        return 0;
    }
    let max_x = i32::try_from(plane.width - 1).unwrap_or(0);
    let max_y = i32::try_from(plane.height - 1).unwrap_or(0);
    let x = usize::try_from(x.clamp(0, max_x)).unwrap_or(0);
    let y = usize::try_from(y.clamp(0, max_y)).unwrap_or(0);
    plane
        .samples
        .get(y * plane.stride + x)
        .copied()
        .unwrap_or(0)
}

fn scans(tx: u8) -> (&'static [u16], &'static [u16]) {
    if tx == TX_4X4 {
        (&SCAN_DEFAULT_4X4, &NEIGHBORS_DEFAULT_4X4)
    } else {
        (&SCAN_DEFAULT_8X8, &NEIGHBORS_DEFAULT_8X8)
    }
}

fn fill(buf: &mut [u8], at: usize, n: usize, value: u8) -> Result<(), CodecError> {
    let slot = buf.get_mut(at..at + n).ok_or(bad("entropy context"))?;
    slot.fill(value);
    Ok(())
}

fn region_nz(buf: &[u8], at: usize, n: usize) -> Result<bool, CodecError> {
    let slot = buf.get(at..at + n).ok_or(bad("entropy context"))?;
    Ok(slot.iter().any(|value| *value != 0))
}

fn bw_of(bsize: u8) -> usize {
    geom::mi_wh(bsize).0
}

fn bh_of(bsize: u8) -> usize {
    geom::mi_wh(bsize).1
}

fn bad(what: &'static str) -> CodecError {
    CodecError::Malformed { at_bit: 0, what }
}

fn limit() -> CodecError {
    CodecError::Limit { what: "vp9 frame" }
}
