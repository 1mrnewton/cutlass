//! The macroblock layer of one VOP or short-header picture: syntax
//! (Sections 6.2.6 to 6.2.8), motion vector and intra prediction (7.4,
//! 7.6), video packets and data partitioning (E.1), and reconstruction.

use cutlass_decode::CodecError;

use crate::bits::BitReader;
use crate::block::{Coding, Quant, dequant_inter, dequant_intra, read_coefs, read_dc};
use crate::frame::{Frame, Kind, MbRecord};
use crate::header::{Encoder, VopType, read_trajectory};
use crate::idct;
use crate::mc::{self, Pel, Rows, Warp};
use crate::tables::{
    ALT_HORIZONTAL, ALT_VERTICAL, MCBPC_INTER_STUFFING, MCBPC_INTRA_STUFFING, ZIGZAG,
    dc_scaler_chroma, dc_scaler_luma, tables,
};

/// `dc_marker`, 19 bits, after partition A of an I-VOP.
const DC_MARKER: u32 = 0x6B001;
/// `motion_marker`, 17 bits, after partition A of a P- or S-VOP.
const MOTION_MARKER: u32 = 0x1F001;
/// `dquant` (Table 6-24).
const DQUANT: [i32; 4] = [-1, -2, 1, 2];

/// Decoding differences that streams of some encoder versions need, as
/// libavcodec detects them from user data.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Fixes {
    /// Quarter-sample chroma rounded toward the half position.
    pub(crate) qpel_chroma: bool,
    /// Quarter-sample chroma with the DivX 5.02+ rounding table.
    pub(crate) qpel_chroma2: bool,
    /// Field chroma vectors truncated vertically.
    pub(crate) hpel_chroma: bool,
    /// Padding from the coded size instead of whole macroblocks.
    pub(crate) edge: bool,
    /// Intra DC predictors not clipped above 2047.
    pub(crate) dc_clip: bool,
    /// The quarter-sample filter of libavcodec builds before 4653.
    pub(crate) legacy_qpel: bool,
    /// DivX 5.00 build 413's sprite trajectory.
    pub(crate) divx_413: bool,
    /// Quarter-sample field luma as two 8x8 halves, which libavcodec's
    /// encoder reconstructs with, instead of one 16x8 block.
    pub(crate) qpel_field_halves: bool,
    /// `dct_type` in every interlaced P macroblock, coded blocks or not,
    /// as old Xvid builds write it. No header names those builds.
    pub(crate) xvid_ilace: bool,
}

impl Fixes {
    pub(crate) fn from_encoder(enc: &Encoder) -> Self {
        let divx = enc.divx_version;
        let build = enc.divx_build;
        let below = |v: Option<u32>, limit: u32| v.is_some_and(|v| v < limit);
        let at_most = |v: Option<u32>, limit: u32| v.is_some_and(|v| v <= limit);
        let divx_build_below_1814 = build.is_none_or(|b| b < 1814);
        Self {
            qpel_chroma: (divx.is_some_and(|v| v >= 500) && divx_build_below_1814)
                || at_most(enc.xvid_build, 1),
            qpel_chroma2: divx.is_some_and(|v| v > 502) && divx_build_below_1814,
            hpel_chroma: divx.is_some(),
            edge: at_most(enc.xvid_build, 12) || below(enc.lavc_build, 4670) || below(divx, 500),
            dc_clip: at_most(enc.xvid_build, 32) || at_most(enc.lavc_build, 4712),
            legacy_qpel: below(enc.lavc_build, 4653),
            divx_413: divx == Some(500) && build == Some(413),
            qpel_field_halves: enc.lavc_build.is_some(),
            xvid_ilace: false,
        }
    }
}

/// The coding tools of a VOL, or of short-header pictures.
#[derive(Clone, Copy)]
pub(crate) struct Layer<'a> {
    pub(crate) mbw: usize,
    pub(crate) mbh: usize,
    /// Luma columns and rows motion compensation pads from.
    pub(crate) edge: (u32, u32),
    /// The picture's luma width and height.
    pub(crate) size: (u32, u32),
    pub(crate) interlaced: bool,
    pub(crate) quarter: bool,
    pub(crate) quant: Quant<'a>,
    pub(crate) partitioned: bool,
    pub(crate) rvlc: bool,
    pub(crate) gmc: bool,
    pub(crate) warping_points: u8,
    pub(crate) time_bits: u32,
    pub(crate) short: bool,
    pub(crate) fixes: Fixes,
}

/// Distances between a B-VOP and its references, in VOL ticks and in
/// fields (Section 7.6.9.5).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Times {
    pub(crate) pp: i32,
    pub(crate) pb: i32,
    pub(crate) pp_field: i32,
    pub(crate) pb_field: i32,
}

/// The header fields of one picture the macroblock layer reads.
pub(crate) struct Picture {
    pub(crate) kind: VopType,
    pub(crate) quant: u32,
    pub(crate) rounding: bool,
    pub(crate) fcode_forward: u32,
    pub(crate) fcode_backward: u32,
    pub(crate) dc_threshold: u32,
    pub(crate) alternate_scan: bool,
    pub(crate) top_field_first: bool,
    pub(crate) warp: Option<Warp>,
    pub(crate) times: Times,
    pub(crate) data_start: usize,
    /// Short header: macroblock rows in a group of blocks.
    pub(crate) gob_rows: usize,
}

/// What partitions A and B of a data partitioned packet leave for C.
#[derive(Clone, Copy, Default)]
struct PartMb {
    intra: bool,
    skipped: bool,
    gmc: bool,
    four: bool,
    dquant: bool,
    cbp: u8,
    ac_pred: bool,
    use_dc_vlc: bool,
    /// Quantised DC and prediction direction of each block.
    dc: [(i32, u8); 6],
}

/// Prediction state that persists over the macroblocks of a picture.
pub(crate) struct Grid {
    mbw: usize,
    mbh: usize,
    /// Reconstructed intra DC of each block: luma, Cb, Cr.
    dc: [Vec<i32>; 3],
    /// Quantised first column (`[1..8]`) and row (`[9..16]`) of each block.
    ac: [Vec<[i32; 16]>; 3],
    qscale: Vec<i32>,
    /// Vector of each 8x8 luma block of the current picture.
    mv: Vec<(i32, i32)>,
    parts: Vec<PartMb>,
}

impl Grid {
    pub(crate) fn new(mbw: usize, mbh: usize) -> Self {
        let luma = 4 * mbw * mbh;
        let chroma = mbw * mbh;
        Self {
            mbw,
            mbh,
            dc: [vec![1024; luma], vec![1024; chroma], vec![1024; chroma]],
            ac: [
                vec![[0; 16]; luma],
                vec![[0; 16]; chroma],
                vec![[0; 16]; chroma],
            ],
            qscale: vec![1; chroma],
            mv: vec![(0, 0); luma],
            parts: vec![PartMb::default(); chroma],
        }
    }

    pub(crate) fn fits(&self, mbw: usize, mbh: usize) -> bool {
        self.mbw == mbw && self.mbh == mbh
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Motion {
    Frame((i32, i32)),
    Four([(i32, i32); 4]),
    Field {
        mv: [(i32, i32); 2],
        select: [u8; 2],
    },
    Gmc,
}

/// Coefficients of one macroblock, dequantised, and which blocks carry
/// a residual.
struct Blocks {
    coef: [[i16; 64]; 6],
    coded: u8,
}

impl Blocks {
    fn new() -> Self {
        Self {
            coef: [[0; 64]; 6],
            coded: 0,
        }
    }
}

/// Decodes the macroblocks of one picture from `data` into `out`.
/// `past` is the forward reference of a P-, S- or B-VOP and `future`
/// the backward reference of a B-VOP.
pub(crate) fn decode(
    data: &[u8],
    layer: &Layer<'_>,
    pic: &Picture,
    grid: &mut Grid,
    past: Option<&Frame>,
    future: Option<&Frame>,
    out: &mut Frame,
) -> Result<(), CodecError> {
    let mb_num = layer.mbw * layer.mbh;
    if !grid.fits(layer.mbw, layer.mbh) || out.mbw != layer.mbw || out.mbh != layer.mbh {
        return Err(CodecError::InvalidApi {
            what: "picture buffers do not match the layer",
        });
    }
    let needs_past = pic.kind != VopType::I;
    let needs_future = pic.kind == VopType::B;
    let fits = |f: Option<&Frame>| f.is_some_and(|f| f.mbw == layer.mbw && f.mbh == layer.mbh);
    if (needs_past && !fits(past))
        || (needs_future && !(fits(future) && future.is_some_and(|f| f.mbs.len() == mb_num)))
    {
        return Err(CodecError::InvalidApi {
            what: "reference picture missing",
        });
    }
    if pic.kind == VopType::B {
        out.mbs.clear();
    } else {
        out.mbs.clear();
        out.mbs.resize(mb_num, MbRecord::default());
    }
    let prefix = match pic.kind {
        VopType::I => 16,
        VopType::P | VopType::S => pic.fcode_forward + 15,
        VopType::B => pic.fcode_forward.max(pic.fcode_backward).max(2) + 15,
    };
    let mut r = BitReader::new(data);
    r.seek(pic.data_start);
    let mut vop = Vop {
        layer,
        pic,
        grid,
        past,
        future,
        out,
        mb_num,
        start: 0,
        q: pic.quant.clamp(1, 31) as i32,
        last_mv: [[(0, 0); 2]; 2],
        prefix,
    };
    if layer.short {
        vop.short_picture(&mut r)
    } else if layer.partitioned && pic.kind != VopType::B {
        vop.partitioned(&mut r)
    } else {
        vop.combined(&mut r)
    }
}

struct Vop<'a> {
    layer: &'a Layer<'a>,
    pic: &'a Picture,
    grid: &'a mut Grid,
    past: Option<&'a Frame>,
    future: Option<&'a Frame>,
    out: &'a mut Frame,
    mb_num: usize,
    /// First macroblock of the current video packet.
    start: usize,
    /// Running quantiser.
    q: i32,
    /// B-VOP vector predictors, `[direction][field]`.
    last_mv: [[(i32, i32); 2]; 2],
    /// Zeros of a resync marker.
    prefix: u32,
}

fn read_mv(r: &mut BitReader<'_>, pred: i32, fcode: u32) -> Result<i32, CodecError> {
    let Some(code) = tables().mvd.read(r) else {
        return Err(r.malformed("motion vector"));
    };
    if code == 0 {
        return Ok(pred);
    }
    let sign = r.bit();
    let shift = fcode.saturating_sub(1);
    let mut v = i32::from(code);
    if shift > 0 {
        v = (((v - 1) << shift) | r.read(shift) as i32) + 1;
    }
    if sign {
        v = -v;
    }
    let v = v + pred;
    // The vector range wraps (Section 7.6.3).
    let bits = 32 - (5 + fcode);
    Ok((v << bits) >> bits)
}

fn rounded_div(a: i32, b: i32) -> i32 {
    if a >= 0 {
        (a + (b >> 1)) / b
    } else {
        (a - (b >> 1)) / b
    }
}

fn median(a: i32, b: i32, c: i32) -> i32 {
    a.max(b).min(a.min(b).max(c))
}

/// One axis of an 8x8 block vector, `shift` fraction bits, moved so the
/// block's integer position stays in `low..=limit`, with no fraction at
/// `limit`. FFmpeg places 8x8 blocks this way, so a vector past the
/// picture reads the rows between the picture and the padded edge.
fn clip_block(origin: i32, m: i32, shift: u32, low: i32, limit: i32) -> i32 {
    let pos = (origin + (m >> shift)).clamp(low, limit);
    let fraction = if pos == limit {
        0
    } else {
        m & ((1 << shift) - 1)
    };
    ((pos - origin) << shift) | fraction
}

impl Vop<'_> {
    fn kind(&self) -> VopType {
        self.pic.kind
    }

    fn set_q(&mut self, delta: i32) {
        self.q = (self.q + delta).clamp(1, 31);
    }

    fn mb_xy(&self, mb: usize) -> (usize, usize) {
        (mb % self.layer.mbw, mb / self.layer.mbw)
    }

    // ---- intra prediction ----

    /// Plane, block column and row, and blocks per row of block `k`.
    fn geometry(&self, mbx: usize, mby: usize, k: usize) -> (usize, isize, isize, usize) {
        if k < 4 {
            (
                0,
                (2 * mbx + (k & 1)) as isize,
                (2 * mby + (k >> 1)) as isize,
                2 * self.layer.mbw,
            )
        } else {
            (k - 3, mbx as isize, mby as isize, self.layer.mbw)
        }
    }

    /// Index of block `(bx, by)` of `plane` if it lies in the picture and
    /// the current video packet.
    fn neighbour(&self, plane: usize, bx: isize, by: isize, w: usize) -> Option<usize> {
        if bx < 0 || by < 0 || bx as usize >= w {
            return None;
        }
        let (bx, by) = (bx as usize, by as usize);
        let owner = if plane == 0 {
            (by >> 1) * self.layer.mbw + (bx >> 1)
        } else {
            by * self.layer.mbw + bx
        };
        (owner >= self.start).then_some(by * w + bx)
    }

    /// The DC predictor of block `k` (Section 7.4.3.1) and its direction:
    /// 0 from the left, 1 from above.
    fn dc_pred(&self, mbx: usize, mby: usize, k: usize) -> (i32, u8) {
        let (plane, bx, by, w) = self.geometry(mbx, mby, k);
        let dc = &self.grid.dc[plane];
        let get = |dx: isize, dy: isize| {
            self.neighbour(plane, bx + dx, by + dy, w)
                .and_then(|i| dc.get(i).copied())
                .unwrap_or(1024)
        };
        let (a, b, c) = (get(-1, 0), get(-1, -1), get(0, -1));
        if (a - b).abs() < (b - c).abs() {
            (c, 1)
        } else {
            (a, 0)
        }
    }

    /// Completes the DC of block `k` from its differential, stores the
    /// predictor for later blocks, and returns the quantised DC.
    fn dc_complete(&mut self, mbx: usize, mby: usize, k: usize, diff: i32, pred: i32) -> i32 {
        let scale = self.dc_scale(k);
        let qf = diff + (pred + (scale >> 1)) / scale;
        let mut stored = qf * scale;
        if stored < 0 {
            stored = 0;
        } else if stored > 2047 && !self.layer.fixes.dc_clip {
            stored = 2047;
        }
        let (plane, bx, by, w) = self.geometry(mbx, mby, k);
        if let Some(slot) = self.grid.dc[plane].get_mut(by as usize * w + bx as usize) {
            *slot = stored;
        }
        qf
    }

    fn dc_scale(&self, k: usize) -> i32 {
        let q = self.q as u32;
        if k < 4 {
            dc_scaler_luma(q)
        } else {
            dc_scaler_chroma(q)
        }
    }

    /// Adds the AC prediction of block `k` from direction `dir` when
    /// `ac_pred` is set, and stores the block's first row and column.
    fn ac_complete(
        &mut self,
        mbx: usize,
        mby: usize,
        k: usize,
        dir: u8,
        ac_pred: bool,
        levels: &mut [i32; 64],
    ) {
        let (plane, bx, by, w) = self.geometry(mbx, mby, k);
        if ac_pred {
            let (nx, ny) = if dir == 0 { (bx - 1, by) } else { (bx, by - 1) };
            if let Some(i) = self.neighbour(plane, nx, ny, w)
                && let Some(nb) = self.grid.ac[plane].get(i)
            {
                let owner = if plane == 0 {
                    (ny as usize >> 1) * self.layer.mbw + (nx as usize >> 1)
                } else {
                    ny as usize * self.layer.mbw + nx as usize
                };
                let here = mby * self.layer.mbw + mbx;
                let qn = self.grid.qscale.get(owner).copied().unwrap_or(self.q);
                let rescale = owner != here && qn != self.q;
                for j in 1..8 {
                    let v = if dir == 0 { nb[j] } else { nb[8 + j] };
                    let v = if rescale {
                        rounded_div(v * qn, self.q)
                    } else {
                        v
                    };
                    if dir == 0 {
                        levels[j * 8] += v;
                    } else {
                        levels[j] += v;
                    }
                }
            }
        }
        if let Some(slot) = self.grid.ac[plane].get_mut(by as usize * w + bx as usize) {
            for j in 1..8 {
                slot[j] = levels[j * 8];
                slot[8 + j] = levels[j];
            }
        }
    }

    /// Forgets the intra predictors of a macroblock that is not intra.
    fn clear_intra(&mut self, mbx: usize, mby: usize) {
        for k in 0..6 {
            let (plane, bx, by, w) = self.geometry(mbx, mby, k);
            let i = by as usize * w + bx as usize;
            if let Some(d) = self.grid.dc[plane].get_mut(i) {
                *d = 1024;
            }
            if let Some(a) = self.grid.ac[plane].get_mut(i) {
                *a = [0; 16];
            }
        }
    }

    fn set_qscale(&mut self, mbx: usize, mby: usize) {
        let q = self.q;
        if let Some(s) = self.grid.qscale.get_mut(mby * self.layer.mbw + mbx) {
            *s = q;
        }
    }

    /// Reads the six DCs of an intra macroblock coded with the DC VLCs
    /// in partition A or B.
    fn read_dcs(
        &mut self,
        r: &mut BitReader<'_>,
        mbx: usize,
        mby: usize,
    ) -> Result<[(i32, u8); 6], CodecError> {
        let mut out = [(0, 0); 6];
        for (k, slot) in out.iter_mut().enumerate() {
            let diff = read_dc(r, k >= 4)?;
            let (pred, dir) = self.dc_pred(mbx, mby, k);
            *slot = (self.dc_complete(mbx, mby, k, diff, pred), dir);
        }
        Ok(out)
    }

    /// Reads and dequantises the six blocks of an intra macroblock.
    /// `dcs` holds DCs partition A or B already read.
    #[allow(clippy::too_many_arguments)]
    fn intra_blocks(
        &mut self,
        r: &mut BitReader<'_>,
        mbx: usize,
        mby: usize,
        cbp: u8,
        ac_pred: bool,
        use_dc_vlc: bool,
        dcs: Option<&[(i32, u8); 6]>,
        coding: Coding,
        blocks: &mut Blocks,
    ) -> Result<(), CodecError> {
        let t = tables();
        let table = if coding == Coding::Reversible {
            &t.intra_rvlc
        } else {
            &t.intra
        };
        let alt = self.pic.alternate_scan;
        for k in 0..6 {
            let coded = cbp & (32 >> k) != 0;
            let mut levels = [0i32; 64];
            let scan = |dir: u8| {
                if alt {
                    &ALT_VERTICAL
                } else if !ac_pred {
                    &ZIGZAG
                } else if dir == 0 {
                    &ALT_VERTICAL
                } else {
                    &ALT_HORIZONTAL
                }
            };
            let (qf, dir) = if use_dc_vlc {
                let (qf, dir) = match dcs.and_then(|d| d.get(k)) {
                    Some(&d) => d,
                    None => {
                        let diff = read_dc(r, k >= 4)?;
                        let (pred, dir) = self.dc_pred(mbx, mby, k);
                        (self.dc_complete(mbx, mby, k, diff, pred), dir)
                    }
                };
                if coded {
                    read_coefs(r, table, coding, scan(dir), 1, &mut levels)?;
                }
                (qf, dir)
            } else {
                let (pred, dir) = self.dc_pred(mbx, mby, k);
                if coded {
                    read_coefs(r, table, coding, scan(dir), 0, &mut levels)?;
                }
                let diff = levels[0];
                levels[0] = 0;
                (self.dc_complete(mbx, mby, k, diff, pred), dir)
            };
            self.ac_complete(mbx, mby, k, dir, ac_pred, &mut levels);
            let scale = self.dc_scale(k);
            blocks.coef[k] = dequant_intra(&levels, qf * scale, self.q, self.layer.quant);
        }
        blocks.coded = 0x3F;
        Ok(())
    }

    fn inter_blocks(
        &self,
        r: &mut BitReader<'_>,
        cbp: u8,
        coding: Coding,
        blocks: &mut Blocks,
    ) -> Result<(), CodecError> {
        let t = tables();
        let table = if coding == Coding::Reversible {
            &t.inter_rvlc
        } else {
            &t.inter
        };
        let scan = if self.pic.alternate_scan {
            &ALT_VERTICAL
        } else {
            &ZIGZAG
        };
        let quant = if self.layer.short {
            Quant::H263
        } else {
            self.layer.quant
        };
        blocks.coded = 0;
        for k in 0..6 {
            if cbp & (32 >> k) == 0 {
                continue;
            }
            let mut levels = [0i32; 64];
            read_coefs(r, table, coding, scan, 0, &mut levels)?;
            blocks.coef[k] = dequant_inter(&levels, self.q, quant);
            blocks.coded |= 1 << k;
        }
        Ok(())
    }

    // ---- motion vectors ----

    /// The median predictor of 8x8 block `k` (16x16 uses block 0) from
    /// the current picture's vectors (Section 7.6.5).
    fn mv_pred(&self, mbx: usize, mby: usize, k: usize) -> (i32, i32) {
        let w = 2 * self.layer.mbw;
        let bx = (2 * mbx + (k & 1)) as isize;
        let by = (2 * mby + (k >> 1)) as isize;
        // Left, above, above right (above left for block 3).
        let cands = [
            (-1, 0),
            (0, -1),
            (
                if k == 3 {
                    -1
                } else if k == 0 {
                    2
                } else {
                    1
                },
                -1,
            ),
        ];
        let mut valid = [None; 3];
        for (slot, (dx, dy)) in valid.iter_mut().zip(cands) {
            if let Some(i) = self.neighbour(0, bx + dx, by + dy, w) {
                *slot = self.grid.mv.get(i).copied();
            }
        }
        match valid {
            [Some(a), Some(b), Some(c)] => (median(a.0, b.0, c.0), median(a.1, b.1, c.1)),
            [Some(a), Some(b), None] | [Some(a), None, Some(b)] | [None, Some(a), Some(b)] => {
                (median(a.0, b.0, 0), median(a.1, b.1, 0))
            }
            [Some(a), None, None] | [None, Some(a), None] | [None, None, Some(a)] => a,
            [None, None, None] => (0, 0),
        }
    }

    fn set_mvs(&mut self, mbx: usize, mby: usize, mvs: [(i32, i32); 4]) {
        let w = 2 * self.layer.mbw;
        for (k, mv) in mvs.into_iter().enumerate() {
            let i = (2 * mby + (k >> 1)) * w + 2 * mbx + (k & 1);
            if let Some(slot) = self.grid.mv.get_mut(i) {
                *slot = mv;
            }
        }
    }

    fn amv(&self, mbx: usize, mby: usize) -> (i32, i32) {
        self.pic.warp.map_or((0, 0), |w| {
            w.average_vector(
                mbx as u32,
                mby as u32,
                self.layer.quarter,
                self.pic.fcode_forward,
                self.layer.fixes.divx_413,
            )
        })
    }

    // ---- resync ----

    /// When a resync marker follows, a reader positioned at it, past the
    /// stuffing. Macroblock stuffing before it is consumed from `r`.
    fn at_resync<'d>(&self, r: &mut BitReader<'d>) -> Option<BitReader<'d>> {
        if !self.layer.partitioned && self.kind() != VopType::B {
            let n = if self.kind() == VopType::I { 9 } else { 10 };
            while r.left() > n as usize && r.peek(n) == 1 {
                r.skip(n);
            }
        }
        let mut t = r.clone();
        let n = t.bits_to_align();
        if t.read(n) != (1 << (n - 1)) - 1 {
            return None;
        }
        let at = t.clone();
        let mut zeros = 0;
        while zeros < 32 && !t.bit() {
            zeros += 1;
        }
        (!t.overrun() && zeros >= self.prefix).then_some(at)
    }

    /// Starts the video packet a header read.
    fn begin_packet(&mut self, mb: usize, q: u32) {
        if q != 0 {
            self.q = q as i32;
        }
        self.start = mb;
        self.last_mv = [[(0, 0); 2]; 2];
    }

    /// Reads `video_packet_header()` after the stuffing: its first
    /// macroblock and quantiser.
    fn packet_header(&self, r: &mut BitReader<'_>) -> Result<(usize, u32), CodecError> {
        for _ in 0..self.prefix {
            if r.bit() {
                return Err(r.malformed("resync marker length"));
            }
        }
        r.marker("resync marker")?;
        let bits = (usize::BITS - (self.mb_num - 1).leading_zeros()).max(1);
        let mb = r.read(bits) as usize;
        if mb == 0 || mb >= self.mb_num {
            return Err(r.malformed("macroblock_number"));
        }
        let q = r.read(5);
        if r.bit() {
            // header_extension_code: a copy of the VOP header.
            let mut n = 0;
            while r.bit() {
                n += 1;
                if n > 3600 || r.overrun() {
                    return Err(r.malformed("modulo_time_base"));
                }
            }
            r.marker("header extension")?;
            r.skip(self.layer.time_bits);
            r.marker("header extension")?;
            r.skip(2 + 3);
            if self.kind() == VopType::S && self.layer.gmc {
                read_trajectory(r, self.layer.warping_points, self.layer.fixes.divx_413)?;
            }
            if self.kind() != VopType::I {
                r.skip(3);
            }
            if self.kind() == VopType::B {
                r.skip(3);
            }
        }
        r.check("video packet header")?;
        Ok((mb, q))
    }

    // ---- reconstruction ----

    fn predict(
        &self,
        mbx: usize,
        mby: usize,
        motion: Motion,
        reference: &Frame,
        rounding: bool,
        y: &mut [u8; 256],
        u: &mut [u8; 64],
        v: &mut [u8; 64],
    ) {
        let [ry, ru, rv] = reference.refs(self.layer.edge);
        let quarter = self.layer.quarter;
        let pel = match (quarter, self.layer.fixes.legacy_qpel) {
            (false, _) => Pel::Half,
            (true, false) => Pel::Quarter,
            (true, true) => Pel::QuarterLegacy,
        };
        let fixes = self.layer.fixes;
        let (x, yy) = ((mbx * 16) as i32, (mby * 16) as i32);
        let (cx0, cy0) = ((mbx * 8) as i32, (mby * 8) as i32);
        let (width, height) = (self.layer.size.0 as i32, self.layer.size.1 as i32);
        match motion {
            Motion::Frame((mx, my)) => {
                mc::luma_block(&ry, x, yy, 16, 16, mx, my, pel, Rows::Frame, rounding, y);
                let chroma = |m: i32| {
                    if !quarter {
                        return mc::chroma_hpel(m);
                    }
                    let h = if fixes.qpel_chroma2 {
                        const RTAB: [i32; 8] = [0, 0, 1, 1, 0, 0, 0, 1];
                        (m >> 1) + RTAB[(m & 7) as usize]
                    } else if fixes.qpel_chroma {
                        (m >> 1) | (m & 1)
                    } else {
                        m / 2
                    };
                    mc::chroma_hpel(h)
                };
                let (cx, cy) = (chroma(mx), chroma(my));
                mc::chroma_block(&ru, cx0, cy0, 8, 8, cx, cy, Rows::Frame, rounding, u);
                mc::chroma_block(&rv, cx0, cy0, 8, 8, cx, cy, Rows::Frame, rounding, v);
            }
            Motion::Four(mvs) => {
                let mut tmp = [0u8; 64];
                let (mut sx, mut sy) = (0, 0);
                let shift = if quarter { 2 } else { 1 };
                for (k, &(mx, my)) in mvs.iter().enumerate() {
                    let (bx, by) = ((k & 1) * 8, (k >> 1) * 8);
                    let (bx0, by0) = (x + bx as i32, yy + by as i32);
                    mc::luma_block(
                        &ry,
                        bx0,
                        by0,
                        8,
                        8,
                        clip_block(bx0, mx, shift, -16, width),
                        clip_block(by0, my, shift, -16, height),
                        pel,
                        Rows::Frame,
                        rounding,
                        &mut tmp,
                    );
                    for row in 0..8 {
                        let d = (by + row) * 16 + bx;
                        y[d..d + 8].copy_from_slice(&tmp[row * 8..row * 8 + 8]);
                    }
                    if quarter {
                        sx += mx / 2;
                        sy += my / 2;
                    } else {
                        sx += mx;
                        sy += my;
                    }
                }
                let cx = clip_block(cx0, mc::chroma_4mv(sx), 1, -8, width >> 1);
                let cy = clip_block(cy0, mc::chroma_4mv(sy), 1, -8, height >> 1);
                mc::chroma_block(&ru, cx0, cy0, 8, 8, cx, cy, Rows::Frame, rounding, u);
                mc::chroma_block(&rv, cx0, cy0, 8, 8, cx, cy, Rows::Frame, rounding, v);
            }
            Motion::Field { mv, select } => {
                let mut luma = [0u8; 128];
                let mut chroma = [0u8; 32];
                for f in 0..2 {
                    let (mx, my) = mv[f];
                    let rows = Rows::Field(select[f] & 1);
                    let fy = (mby * 8) as i32;
                    if quarter && fixes.qpel_field_halves {
                        let mut half = [0u8; 64];
                        for side in 0..2 {
                            mc::luma_block(
                                &ry,
                                x + 8 * side as i32,
                                fy,
                                8,
                                8,
                                mx,
                                my,
                                pel,
                                rows,
                                rounding,
                                &mut half,
                            );
                            for row in 0..8 {
                                let d = row * 16 + 8 * side;
                                luma[d..d + 8].copy_from_slice(&half[row * 8..row * 8 + 8]);
                            }
                        }
                    } else {
                        mc::luma_block(&ry, x, fy, 16, 8, mx, my, pel, rows, rounding, &mut luma);
                    }
                    for row in 0..8 {
                        let d = (2 * row + f) * 16;
                        y[d..d + 16].copy_from_slice(&luma[row * 16..row * 16 + 16]);
                    }
                    let (cx, cy) = if quarter {
                        (mc::chroma_hpel(mx / 2), mc::chroma_hpel(my >> 1))
                    } else if fixes.hpel_chroma {
                        (mc::chroma_hpel(mx), my >> 1)
                    } else {
                        (mc::chroma_hpel(mx), mc::chroma_hpel(my))
                    };
                    for (plane, out) in [(&ru, &mut *u), (&rv, &mut *v)] {
                        mc::chroma_block(
                            plane,
                            cx0,
                            (mby * 4) as i32,
                            8,
                            4,
                            cx,
                            cy,
                            rows,
                            rounding,
                            &mut chroma,
                        );
                        for row in 0..4 {
                            let d = (2 * row + f) * 8;
                            out[d..d + 8].copy_from_slice(&chroma[row * 8..row * 8 + 8]);
                        }
                    }
                }
            }
            Motion::Gmc => {
                if let Some(w) = &self.pic.warp {
                    mc::gmc_macroblock(w, &ry, &ru, &rv, mbx as u32, mby as u32, rounding, y, u, v);
                }
            }
        }
    }

    /// Writes the prediction and residual of a macroblock into the
    /// picture.
    fn reconstruct(
        &mut self,
        mbx: usize,
        mby: usize,
        fwd: Option<Motion>,
        bwd: Option<Motion>,
        field_dct: bool,
        blocks: &mut Blocks,
    ) {
        let ls = self.out.luma_stride();
        let cs = self.out.chroma_stride();
        let intra = fwd.is_none() && bwd.is_none();
        if !intra {
            let mut py = [0u8; 256];
            let mut pu = [0u8; 64];
            let mut pv = [0u8; 64];
            let b_vop = self.kind() == VopType::B;
            let rounding = !b_vop && self.pic.rounding;
            let (fwd_ref, bwd_ref) = if b_vop {
                (self.past, self.future)
            } else {
                (self.past, None)
            };
            if let (Some(m), Some(f)) = (fwd, fwd_ref) {
                self.predict(mbx, mby, m, f, rounding, &mut py, &mut pu, &mut pv);
            }
            if let (Some(m), Some(b)) = (bwd, bwd_ref) {
                if fwd.is_some() {
                    let mut by = [0u8; 256];
                    let mut bu = [0u8; 64];
                    let mut bv = [0u8; 64];
                    self.predict(mbx, mby, m, b, rounding, &mut by, &mut bu, &mut bv);
                    mc::average(&mut py, &by);
                    mc::average(&mut pu, &bu);
                    mc::average(&mut pv, &bv);
                } else {
                    self.predict(mbx, mby, m, b, rounding, &mut py, &mut pu, &mut pv);
                }
            }
            let base = mby * 16 * ls + mbx * 16;
            for row in 0..16 {
                if let Some(d) = self.out.y.get_mut(base + row * ls..base + row * ls + 16) {
                    d.copy_from_slice(&py[row * 16..row * 16 + 16]);
                }
            }
            let base = mby * 8 * cs + mbx * 8;
            for row in 0..8 {
                if let Some(d) = self.out.u.get_mut(base + row * cs..base + row * cs + 8) {
                    d.copy_from_slice(&pu[row * 8..row * 8 + 8]);
                }
                if let Some(d) = self.out.v.get_mut(base + row * cs..base + row * cs + 8) {
                    d.copy_from_slice(&pv[row * 8..row * 8 + 8]);
                }
            }
        }
        for k in 0..6 {
            if !intra && blocks.coded & (1 << k) == 0 {
                continue;
            }
            let (plane, offset, stride) = if k < 4 {
                let (row, step) = if field_dct {
                    (k >> 1, 2 * ls)
                } else {
                    ((k >> 1) * 8, ls)
                };
                (
                    &mut self.out.y,
                    (mby * 16 + row) * ls + mbx * 16 + (k & 1) * 8,
                    step,
                )
            } else {
                let plane = if k == 4 {
                    &mut self.out.u
                } else {
                    &mut self.out.v
                };
                (plane, mby * 8 * cs + mbx * 8, cs)
            };
            if intra {
                idct::put(&mut blocks.coef[k], plane, offset, stride);
            } else {
                idct::add(&mut blocks.coef[k], plane, offset, stride);
            }
        }
    }

    fn record(&mut self, mb: usize, record: MbRecord) {
        if let Some(slot) = self.out.mbs.get_mut(mb) {
            *slot = record;
        }
    }

    // ---- combined (not partitioned) VOPs ----

    fn combined(&mut self, r: &mut BitReader<'_>) -> Result<(), CodecError> {
        let mut mb = 0;
        loop {
            if mb % self.layer.mbw == 0 && self.kind() == VopType::B {
                self.last_mv = [[(0, 0); 2]; 2];
            }
            match self.kind() {
                VopType::I => self.i_macroblock(r, mb)?,
                VopType::P | VopType::S => self.p_macroblock(r, mb)?,
                VopType::B => self.b_macroblock(r, mb)?,
            }
            r.check("macroblock")?;
            mb += 1;
            if mb == self.mb_num {
                return Ok(());
            }
            if let Some(mut probe) = self.at_resync(r) {
                let (n, q) = self.packet_header(&mut probe)?;
                if n == mb {
                    *r = probe;
                    self.begin_packet(n, q);
                } else if !(n > mb && self.kind() == VopType::B && self.colocated_skip(mb)) {
                    // A B-VOP's skipped macroblocks hold no bits, so the
                    // marker can belong to a later one; anything else
                    // means lost macroblocks.
                    return Err(r.malformed("video packet starts out of order"));
                }
            }
        }
    }

    fn colocated_skip(&self, mb: usize) -> bool {
        self.future
            .and_then(|f| f.mbs.get(mb))
            .is_some_and(|m| m.kind == Kind::Skipped)
    }

    fn i_macroblock(&mut self, r: &mut BitReader<'_>, mb: usize) -> Result<(), CodecError> {
        let t = tables();
        let sym = loop {
            let Some(sym) = t.mcbpc_intra.read(r) else {
                return Err(r.malformed("mcbpc"));
            };
            if sym != MCBPC_INTRA_STUFFING {
                break sym;
            }
            r.check("mcbpc stuffing")?;
        };
        self.intra_macroblock(r, mb, (sym & 3) as u8, sym & 4 != 0)
    }

    fn intra_macroblock(
        &mut self,
        r: &mut BitReader<'_>,
        mb: usize,
        cbpc: u8,
        dquant: bool,
    ) -> Result<(), CodecError> {
        let (mbx, mby) = self.mb_xy(mb);
        let ac_pred = r.bit();
        let Some(cbpy) = tables().cbpy.read(r) else {
            return Err(r.malformed("cbpy"));
        };
        let cbp = cbpc | ((cbpy as u8) << 2);
        let use_dc_vlc = (self.q as u32) < self.pic.dc_threshold;
        if dquant {
            self.set_q(DQUANT[r.read(2) as usize]);
        }
        let field_dct = self.layer.interlaced && r.bit();
        self.set_qscale(mbx, mby);
        let mut blocks = Blocks::new();
        self.intra_blocks(
            r,
            mbx,
            mby,
            cbp,
            ac_pred,
            use_dc_vlc,
            None,
            Coding::Vlc,
            &mut blocks,
        )?;
        self.set_mvs(mbx, mby, [(0, 0); 4]);
        self.reconstruct(mbx, mby, None, None, field_dct, &mut blocks);
        self.record(mb, MbRecord::default());
        Ok(())
    }

    fn p_macroblock(&mut self, r: &mut BitReader<'_>, mb: usize) -> Result<(), CodecError> {
        let (mbx, mby) = self.mb_xy(mb);
        let t = tables();
        let gmc = self.kind() == VopType::S && self.layer.gmc;
        let sym = loop {
            if r.bit() {
                return self.p_skip(mb, gmc);
            }
            let Some(sym) = t.mcbpc_inter.read(r) else {
                return Err(r.malformed("mcbpc"));
            };
            if sym != MCBPC_INTER_STUFFING {
                break sym;
            }
            r.check("mcbpc stuffing")?;
        };
        let cbpc = (sym & 3) as u8;
        if sym & 4 != 0 {
            return self.intra_macroblock(r, mb, cbpc, sym & 8 != 0);
        }
        let four = sym & 16 != 0;
        let mcsel = gmc && !four && r.bit();
        let Some(cbpy) = t.cbpy.read(r) else {
            return Err(r.malformed("cbpy"));
        };
        let cbp = cbpc | (((cbpy as u8) ^ 0xF) << 2);
        if sym & 8 != 0 {
            self.set_q(DQUANT[r.read(2) as usize]);
        }
        let field_dct =
            self.layer.interlaced && (cbp != 0 || self.layer.fixes.xvid_ilace) && r.bit();
        let f = self.pic.fcode_forward;
        let (motion, record) = if four {
            let mut mvs = [(0, 0); 4];
            for (k, slot) in mvs.iter_mut().enumerate() {
                let (px, py) = self.mv_pred(mbx, mby, k);
                let mv = (read_mv(r, px, f)?, read_mv(r, py, f)?);
                *slot = mv;
                let w = 2 * self.layer.mbw;
                if let Some(s) = self
                    .grid
                    .mv
                    .get_mut((2 * mby + (k >> 1)) * w + 2 * mbx + (k & 1))
                {
                    *s = mv;
                }
            }
            (
                Motion::Four(mvs),
                MbRecord {
                    kind: Kind::Four,
                    mv: mvs,
                    ..MbRecord::default()
                },
            )
        } else if mcsel {
            let amv = self.amv(mbx, mby);
            self.set_mvs(mbx, mby, [amv; 4]);
            (Motion::Gmc, frame_record(amv))
        } else if self.layer.interlaced && r.bit() {
            let select = [u8::from(r.bit()), u8::from(r.bit())];
            let (px, py) = self.mv_pred(mbx, mby, 0);
            let mut mv = [(0, 0); 2];
            for slot in &mut mv {
                *slot = (read_mv(r, px, f)?, read_mv(r, py / 2, f)?);
            }
            let sx = mv[0].0 + mv[1].0;
            let combined = ((sx >> 1) | (sx & 1), mv[0].1 + mv[1].1);
            self.set_mvs(mbx, mby, [combined; 4]);
            (
                Motion::Field { mv, select },
                MbRecord {
                    kind: Kind::Field,
                    mv: [combined; 4],
                    field_mv: mv,
                    field_select: select,
                },
            )
        } else {
            let (px, py) = self.mv_pred(mbx, mby, 0);
            let mv = (read_mv(r, px, f)?, read_mv(r, py, f)?);
            self.set_mvs(mbx, mby, [mv; 4]);
            (Motion::Frame(mv), frame_record(mv))
        };
        self.clear_intra(mbx, mby);
        self.set_qscale(mbx, mby);
        let mut blocks = Blocks::new();
        self.inter_blocks(r, cbp, Coding::Vlc, &mut blocks)?;
        self.reconstruct(mbx, mby, Some(motion), None, field_dct, &mut blocks);
        self.record(mb, record);
        Ok(())
    }

    /// A macroblock a P- or S-VOP does not code.
    fn p_skip(&mut self, mb: usize, gmc: bool) -> Result<(), CodecError> {
        let (mbx, mby) = self.mb_xy(mb);
        let (motion, record) = if gmc {
            let amv = self.amv(mbx, mby);
            (Motion::Gmc, frame_record(amv))
        } else {
            (
                Motion::Frame((0, 0)),
                MbRecord {
                    kind: Kind::Skipped,
                    ..MbRecord::default()
                },
            )
        };
        self.set_mvs(mbx, mby, record.mv);
        self.clear_intra(mbx, mby);
        self.set_qscale(mbx, mby);
        let mut blocks = Blocks::new();
        self.reconstruct(mbx, mby, Some(motion), None, false, &mut blocks);
        self.record(mb, record);
        Ok(())
    }

    fn b_macroblock(&mut self, r: &mut BitReader<'_>, mb: usize) -> Result<(), CodecError> {
        let (mbx, mby) = self.mb_xy(mb);
        let mut blocks = Blocks::new();
        if self.colocated_skip(mb) {
            self.reconstruct(
                mbx,
                mby,
                Some(Motion::Frame((0, 0))),
                None,
                false,
                &mut blocks,
            );
            return Ok(());
        }
        let t = tables();
        // modb '1': direct with no vector delta and no texture.
        let (delta_coded, kind, cbp) = if r.bit() {
            (false, 0, 0)
        } else {
            let no_cbp = r.bit();
            let Some(kind) = t.mb_type_b.read(r) else {
                return Err(r.malformed("mb_type"));
            };
            let cbp = if no_cbp { 0 } else { r.read(6) as u8 };
            (true, kind, cbp)
        };
        let direct = kind == 0;
        if !direct && cbp != 0 && r.bit() {
            self.set_q(if r.bit() { 2 } else { -2 });
        }
        let (has_fwd, has_bwd) = match kind {
            1 => (true, true),
            2 => (false, true),
            3 => (true, false),
            _ => (false, false),
        };
        let mut field_dct = false;
        let mut field = false;
        let mut select = [[0u8; 2]; 2];
        if self.layer.interlaced {
            if cbp != 0 {
                field_dct = r.bit();
            }
            if !direct && r.bit() {
                field = true;
                if has_fwd {
                    select[0] = [u8::from(r.bit()), u8::from(r.bit())];
                }
                if has_bwd {
                    select[1] = [u8::from(r.bit()), u8::from(r.bit())];
                }
            }
        }
        let (fwd, bwd) = if direct {
            let delta = if delta_coded {
                (read_mv(r, 0, 1)?, read_mv(r, 0, 1)?)
            } else {
                (0, 0)
            };
            let (f, b) = self.direct(mb, delta);
            (Some(f), Some(b))
        } else {
            let mut motions = [None, None];
            for (d, used) in [(0, has_fwd), (1, has_bwd)] {
                if !used {
                    continue;
                }
                let f = if d == 0 {
                    self.pic.fcode_forward
                } else {
                    self.pic.fcode_backward
                };
                let m = if field {
                    let mut mv = [(0, 0); 2];
                    for (i, slot) in mv.iter_mut().enumerate() {
                        let (lx, ly) = self.last_mv[d][i];
                        let v = (read_mv(r, lx, f)?, read_mv(r, ly / 2, f)?);
                        self.last_mv[d][i] = (v.0, v.1 * 2);
                        *slot = v;
                    }
                    Motion::Field {
                        mv,
                        select: select[d],
                    }
                } else {
                    let (lx, ly) = self.last_mv[d][0];
                    let v = (read_mv(r, lx, f)?, read_mv(r, ly, f)?);
                    self.last_mv[d] = [v; 2];
                    Motion::Frame(v)
                };
                motions[d] = Some(m);
            }
            (motions[0], motions[1])
        };
        self.inter_blocks(r, cbp, Coding::Vlc, &mut blocks)?;
        self.reconstruct(mbx, mby, fwd, bwd, field_dct, &mut blocks);
        Ok(())
    }

    /// Direct mode vectors from the colocated macroblock of the backward
    /// reference plus `delta` (Section 7.6.9.5).
    fn direct(&self, mb: usize, delta: (i32, i32)) -> (Motion, Motion) {
        let rec = self
            .future
            .and_then(|f| f.mbs.get(mb))
            .copied()
            .unwrap_or_default();
        let t = self.pic.times;
        let scale = |p: i32, d: i32, pp: i32, pb: i32| {
            let pp = pp.max(1);
            let f = p * pb / pp + d;
            let b = if d != 0 { f - p } else { p * (pb - pp) / pp };
            (f, b)
        };
        let pair = |p: (i32, i32), pp: i32, pb: i32| {
            let (fx, bx) = scale(p.0, delta.0, pp, pb);
            let (fy, by) = scale(p.1, delta.1, pp, pb);
            ((fx, fy), (bx, by))
        };
        match rec.kind {
            Kind::Four => {
                let mut f = [(0, 0); 4];
                let mut b = [(0, 0); 4];
                for k in 0..4 {
                    (f[k], b[k]) = pair(rec.mv[k], t.pp, t.pb);
                }
                (Motion::Four(f), Motion::Four(b))
            }
            Kind::Field => {
                let mut f = [(0, 0); 2];
                let mut b = [(0, 0); 2];
                for i in 0..2 {
                    let fs = i32::from(rec.field_select[i]);
                    let ii = i as i32;
                    let (pp, pb) = if self.pic.top_field_first {
                        (t.pp_field - fs + ii, t.pb_field - fs + ii)
                    } else {
                        (t.pp_field + fs - ii, t.pb_field + fs - ii)
                    };
                    (f[i], b[i]) = pair(rec.field_mv[i], pp, pb);
                }
                (
                    Motion::Field {
                        mv: f,
                        select: rec.field_select,
                    },
                    Motion::Field {
                        mv: b,
                        select: [0, 1],
                    },
                )
            }
            _ => {
                let (f, b) = pair(rec.mv[0], t.pp, t.pb);
                // Quarter-sample direct mode filters each 8x8 block on its
                // own (Section 7.6.9.5). DivX 5 and libavcodec before build
                // 4655 encode as if 16x16; FFmpeg detects them but decodes
                // 8x8 anyway, and so does this decoder.
                if self.layer.quarter {
                    (Motion::Four([f; 4]), Motion::Four([b; 4]))
                } else {
                    (Motion::Frame(f), Motion::Frame(b))
                }
            }
        }
    }

    // ---- data partitioned VOPs (Annex E.1.2) ----

    fn partitioned(&mut self, r: &mut BitReader<'_>) -> Result<(), CodecError> {
        let intra_vop = self.kind() == VopType::I;
        let mut mb = 0;
        loop {
            let start = mb;
            let end = self.partition_a(r, start)?;
            if end == start {
                return Err(r.malformed("empty video packet"));
            }
            let (stuffing, marker, bits, what) = if intra_vop {
                (9, DC_MARKER, 19, "dc_marker")
            } else {
                (10, MOTION_MARKER, 17, "motion_marker")
            };
            while r.left() > stuffing && r.peek(stuffing as u32) == 1 {
                r.skip(stuffing as u32);
            }
            if r.read(bits) != marker {
                return Err(r.malformed(what));
            }
            self.partition_b(r, start, end)?;
            for m in start..end {
                self.partition_c(r, m)?;
                r.check("macroblock")?;
            }
            mb = end;
            if mb == self.mb_num {
                return Ok(());
            }
            let Some(mut probe) = self.at_resync(r) else {
                return Err(r.malformed("resync marker"));
            };
            let (next, q) = self.packet_header(&mut probe)?;
            if next != mb {
                return Err(r.malformed("video packet starts out of order"));
            }
            *r = probe;
            self.begin_packet(next, q);
        }
    }

    /// Macroblock types, I-VOP DCs and motion vectors; returns the
    /// macroblock after the packet's last.
    fn partition_a(&mut self, r: &mut BitReader<'_>, start: usize) -> Result<usize, CodecError> {
        let t = tables();
        let gmc = self.kind() == VopType::S && self.layer.gmc;
        let f = self.pic.fcode_forward;
        let mut mb = start;
        'mbs: while mb < self.mb_num {
            let (mbx, mby) = self.mb_xy(mb);
            let mut part = PartMb::default();
            if self.kind() == VopType::I {
                let sym = loop {
                    if r.peek(19) == DC_MARKER {
                        break 'mbs;
                    }
                    let Some(sym) = t.mcbpc_intra.read(r) else {
                        return Err(r.malformed("mcbpc"));
                    };
                    if sym != MCBPC_INTRA_STUFFING {
                        break sym;
                    }
                    r.check("mcbpc stuffing")?;
                };
                part.intra = true;
                part.cbp = (sym & 3) as u8;
                part.use_dc_vlc = (self.q as u32) < self.pic.dc_threshold;
                if sym & 4 != 0 {
                    self.set_q(DQUANT[r.read(2) as usize]);
                }
                self.set_qscale(mbx, mby);
                if part.use_dc_vlc {
                    part.dc = self.read_dcs(r, mbx, mby)?;
                }
            } else {
                let sym = loop {
                    if r.peek(17) == MOTION_MARKER {
                        break 'mbs;
                    }
                    if r.bit() {
                        break None;
                    }
                    let Some(sym) = t.mcbpc_inter.read(r) else {
                        return Err(r.malformed("mcbpc"));
                    };
                    if sym != MCBPC_INTER_STUFFING {
                        break Some(sym);
                    }
                    r.check("mcbpc stuffing")?;
                };
                match sym {
                    None => {
                        part.skipped = true;
                        part.gmc = gmc;
                        let mv = if gmc { self.amv(mbx, mby) } else { (0, 0) };
                        self.set_mvs(mbx, mby, [mv; 4]);
                        self.clear_intra(mbx, mby);
                    }
                    Some(sym) => {
                        part.cbp = (sym & 3) as u8;
                        part.dquant = sym & 8 != 0;
                        if sym & 4 != 0 {
                            part.intra = true;
                            self.set_mvs(mbx, mby, [(0, 0); 4]);
                        } else {
                            self.clear_intra(mbx, mby);
                            part.four = sym & 16 != 0;
                            part.gmc = gmc && !part.four && r.bit();
                            if part.four {
                                let w = 2 * self.layer.mbw;
                                for k in 0..4 {
                                    let (px, py) = self.mv_pred(mbx, mby, k);
                                    let mv = (read_mv(r, px, f)?, read_mv(r, py, f)?);
                                    let i = (2 * mby + (k >> 1)) * w + 2 * mbx + (k & 1);
                                    if let Some(s) = self.grid.mv.get_mut(i) {
                                        *s = mv;
                                    }
                                }
                            } else if part.gmc {
                                let amv = self.amv(mbx, mby);
                                self.set_mvs(mbx, mby, [amv; 4]);
                            } else {
                                let (px, py) = self.mv_pred(mbx, mby, 0);
                                let mv = (read_mv(r, px, f)?, read_mv(r, py, f)?);
                                self.set_mvs(mbx, mby, [mv; 4]);
                            }
                        }
                    }
                }
            }
            r.check("partition A")?;
            if let Some(slot) = self.grid.parts.get_mut(mb) {
                *slot = part;
            }
            mb += 1;
        }
        Ok(mb)
    }

    /// `ac_pred_flag`, `cbpy`, `dquant` and P-VOP intra DCs.
    fn partition_b(
        &mut self,
        r: &mut BitReader<'_>,
        start: usize,
        end: usize,
    ) -> Result<(), CodecError> {
        let t = tables();
        for mb in start..end {
            let (mbx, mby) = self.mb_xy(mb);
            let Some(mut part) = self.grid.parts.get(mb).copied() else {
                return Err(r.malformed("video packet"));
            };
            if part.intra {
                part.ac_pred = r.bit();
                let Some(cbpy) = t.cbpy.read(r) else {
                    return Err(r.malformed("cbpy"));
                };
                part.cbp |= (cbpy as u8) << 2;
                if self.kind() != VopType::I {
                    part.use_dc_vlc = (self.q as u32) < self.pic.dc_threshold;
                    if part.dquant {
                        self.set_q(DQUANT[r.read(2) as usize]);
                    }
                    self.set_qscale(mbx, mby);
                    if part.use_dc_vlc {
                        part.dc = self.read_dcs(r, mbx, mby)?;
                    }
                }
            } else if part.skipped {
                self.set_qscale(mbx, mby);
            } else {
                let Some(cbpy) = t.cbpy.read(r) else {
                    return Err(r.malformed("cbpy"));
                };
                part.cbp |= ((cbpy as u8) ^ 0xF) << 2;
                if part.dquant {
                    self.set_q(DQUANT[r.read(2) as usize]);
                }
                self.set_qscale(mbx, mby);
            }
            r.check("partition B")?;
            if let Some(slot) = self.grid.parts.get_mut(mb) {
                *slot = part;
            }
        }
        Ok(())
    }

    /// The texture of one macroblock, and its reconstruction.
    fn partition_c(&mut self, r: &mut BitReader<'_>, mb: usize) -> Result<(), CodecError> {
        let (mbx, mby) = self.mb_xy(mb);
        let part = self.grid.parts.get(mb).copied().unwrap_or_default();
        self.q = self.grid.qscale.get(mb).copied().unwrap_or(self.q);
        let coding = if self.layer.rvlc {
            Coding::Reversible
        } else {
            Coding::Vlc
        };
        let mut blocks = Blocks::new();
        if part.intra {
            let dcs = part.use_dc_vlc.then_some(&part.dc);
            self.intra_blocks(
                r,
                mbx,
                mby,
                part.cbp,
                part.ac_pred,
                part.use_dc_vlc,
                dcs,
                coding,
                &mut blocks,
            )?;
            self.reconstruct(mbx, mby, None, None, false, &mut blocks);
            self.record(mb, MbRecord::default());
            return Ok(());
        }
        let w = 2 * self.layer.mbw;
        let mut mvs = [(0, 0); 4];
        for (k, slot) in mvs.iter_mut().enumerate() {
            let i = (2 * mby + (k >> 1)) * w + 2 * mbx + (k & 1);
            *slot = self.grid.mv.get(i).copied().unwrap_or_default();
        }
        let (motion, record) = if part.gmc {
            (Motion::Gmc, frame_record(mvs[0]))
        } else if part.skipped {
            (
                Motion::Frame((0, 0)),
                MbRecord {
                    kind: Kind::Skipped,
                    ..MbRecord::default()
                },
            )
        } else if part.four {
            (
                Motion::Four(mvs),
                MbRecord {
                    kind: Kind::Four,
                    mv: mvs,
                    ..MbRecord::default()
                },
            )
        } else {
            (Motion::Frame(mvs[0]), frame_record(mvs[0]))
        };
        if !part.skipped {
            self.inter_blocks(r, part.cbp, coding, &mut blocks)?;
        }
        self.reconstruct(mbx, mby, Some(motion), None, false, &mut blocks);
        self.record(mb, record);
        Ok(())
    }

    // ---- short-header pictures (Section 6.2.5.2) ----

    fn short_picture(&mut self, r: &mut BitReader<'_>) -> Result<(), CodecError> {
        let rows = self.pic.gob_rows.max(1);
        for mb in 0..self.mb_num {
            let (mbx, mby) = self.mb_xy(mb);
            if mbx == 0 && mby > 0 && mby % rows == 0 {
                if let Some(q) = self.gob_header(r, (mby / rows) as u32)? {
                    self.begin_packet(mb, q);
                }
            }
            self.short_macroblock(r, mb)?;
            r.check("macroblock")?;
        }
        Ok(())
    }

    /// An optional `gob_layer()` header: its quantiser.
    fn gob_header(&self, r: &mut BitReader<'_>, gob: u32) -> Result<Option<u32>, CodecError> {
        // Macroblock data never holds 16 zeros in a row.
        if r.peek(16) != 0 {
            return Ok(None);
        }
        let mut zeros = 0;
        while zeros < 24 && !r.bit() {
            zeros += 1;
        }
        if zeros >= 24 {
            return Err(r.malformed("gob_resync_marker"));
        }
        if r.read(5) != gob {
            return Err(r.malformed("gob_number"));
        }
        r.skip(2); // gob_frame_id
        let q = r.read(5);
        if q == 0 {
            return Err(r.malformed("quant_scale"));
        }
        r.check("gob header")?;
        Ok(Some(q))
    }

    fn short_macroblock(&mut self, r: &mut BitReader<'_>, mb: usize) -> Result<(), CodecError> {
        let (mbx, mby) = self.mb_xy(mb);
        let t = tables();
        let inter_vop = self.kind() != VopType::I;
        let sym = loop {
            if inter_vop {
                if r.bit() {
                    self.set_mvs(mbx, mby, [(0, 0); 4]);
                    let mut blocks = Blocks::new();
                    self.reconstruct(
                        mbx,
                        mby,
                        Some(Motion::Frame((0, 0))),
                        None,
                        false,
                        &mut blocks,
                    );
                    self.record(
                        mb,
                        MbRecord {
                            kind: Kind::Skipped,
                            ..MbRecord::default()
                        },
                    );
                    return Ok(());
                }
                let Some(sym) = t.mcbpc_inter.read(r) else {
                    return Err(r.malformed("mcbpc"));
                };
                if sym != MCBPC_INTER_STUFFING {
                    break sym;
                }
            } else {
                let Some(sym) = t.mcbpc_intra.read(r) else {
                    return Err(r.malformed("mcbpc"));
                };
                // Intra symbols hold intra+q as 4..8; map to the inter
                // layout (dquant in bit 3, intra in bit 2).
                if sym != MCBPC_INTRA_STUFFING {
                    break 4 | (sym & 3) | ((sym & 4) << 1);
                }
            }
            r.check("mcbpc stuffing")?;
        };
        if sym & 16 != 0 {
            return Err(CodecError::Unsupported {
                what: "short header four vector macroblock",
            });
        }
        let intra = sym & 4 != 0;
        let Some(cbpy) = t.cbpy.read(r) else {
            return Err(r.malformed("cbpy"));
        };
        let cbpy = if intra {
            cbpy as u8
        } else {
            (cbpy as u8) ^ 0xF
        };
        let cbp = (sym & 3) as u8 | (cbpy << 2);
        if sym & 8 != 0 {
            self.set_q(DQUANT[r.read(2) as usize]);
        }
        let mut blocks = Blocks::new();
        if intra {
            for k in 0..6 {
                let v = r.read(8) as i32;
                if v == 0 || v == 128 {
                    return Err(r.malformed("intra_dc_coefficient"));
                }
                let dc = if v == 255 { 1024 } else { v * 8 };
                let mut levels = [0i32; 64];
                if cbp & (32 >> k) != 0 {
                    read_coefs(r, &t.inter, Coding::Short, &ZIGZAG, 1, &mut levels)?;
                }
                blocks.coef[k] = dequant_intra(&levels, dc, self.q, Quant::H263);
            }
            blocks.coded = 0x3F;
            self.set_mvs(mbx, mby, [(0, 0); 4]);
            self.reconstruct(mbx, mby, None, None, false, &mut blocks);
            self.record(mb, MbRecord::default());
            return Ok(());
        }
        let (px, py) = self.mv_pred(mbx, mby, 0);
        let mv = (read_mv(r, px, 1)?, read_mv(r, py, 1)?);
        self.set_mvs(mbx, mby, [mv; 4]);
        self.inter_blocks(r, cbp, Coding::Short, &mut blocks)?;
        self.reconstruct(mbx, mby, Some(Motion::Frame(mv)), None, false, &mut blocks);
        self.record(mb, frame_record(mv));
        Ok(())
    }
}

fn frame_record(mv: (i32, i32)) -> MbRecord {
    MbRecord {
        kind: Kind::Frame,
        mv: [mv; 4],
        ..MbRecord::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_vectors_stop_at_the_picture() {
        // Block at row 296 of a 300-row picture, ten rows down: the block
        // starts at row 300 and keeps no half-pel step there.
        assert_eq!(clip_block(296, 21, 1, -16, 300), 8);
        assert_eq!(clip_block(296, 20, 1, -16, 300), 8);
        // Inside the picture the vector is unchanged, fraction included.
        assert_eq!(clip_block(296, 7, 1, -16, 300), 7);
        assert_eq!(clip_block(0, -31, 1, -16, 300), -31);
        // Far above, the block stops sixteen rows out and keeps its fraction.
        assert_eq!(clip_block(0, -41, 1, -16, 300), -31);
        // Quarter-pel: two fraction bits, both dropped at the limit.
        assert_eq!(clip_block(392, 47, 2, -16, 400), 32);
        assert_eq!(clip_block(392, 13, 2, -16, 400), 13);
        // Chroma of a 4MV macroblock stops at half the picture.
        assert_eq!(clip_block(144, 21, 1, -8, 150), 12);
    }
}
