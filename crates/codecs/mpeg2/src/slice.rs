//! Slices, macroblocks and blocks (H.262 6.2.4 to 6.2.6 and clause 7), and
//! the ISO/IEC 11172-2 variants of them.

use cutlass_decode::CodecError;

use crate::bits::BitReader;
use crate::dct::{idct, idct_dc};
use crate::frame::{Frame, Geometry, View};
use crate::headers::{CHROMA_420, CHROMA_422, Matrix, PICTURE_B, PICTURE_D, PICTURE_P};
use crate::mc::{Target, predict};
use crate::tables::{
    CBP, CBP_BITS, COEF_TABLE_ONE, COEF_TABLE_ZERO, CoefTable, DC_BITS, DC_CHROMA, DC_LUMA, DMV,
    MB_ADDR, MB_ADDR_BITS, MB_BACKWARD, MB_ESCAPE, MB_FORWARD, MB_INTRA, MB_PATTERN, MB_QUANT,
    MB_STUFFING, MB_TYPE_B, MB_TYPE_BITS, MB_TYPE_I, MB_TYPE_P, MOTION, MOTION_BITS,
    NON_LINEAR_SCALE, RUN_EOB, RUN_ESCAPE, SCAN, Vlc,
};

/// The picture-level fields slices depend on.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PictureParams {
    pub mpeg1: bool,
    pub coding_type: u8,
    /// `None` for a frame picture, else the parity of the field (0 top).
    pub field: Option<usize>,
    /// This field is the second field of its frame.
    pub second_field: bool,
    pub f_code: [[u8; 2]; 2],
    pub full_pel: [bool; 2],
    pub intra_dc_precision: u8,
    pub frame_pred_frame_dct: bool,
    pub concealment: bool,
    pub q_scale_type: bool,
    pub intra_vlc: bool,
    pub alternate_scan: bool,
    pub top_field_first: bool,
    pub vertical_size: usize,
}

/// One picture to decode and the frames it predicts from.
pub(crate) struct Picture<'a> {
    pub geo: Geometry,
    pub params: PictureParams,
    pub matrices: &'a [Matrix; 4],
    pub fwd: Option<&'a Frame>,
    pub bwd: Option<&'a Frame>,
}

/// One `slice()`: the start code value and the bytes after the start code.
pub(crate) struct SliceData<'a> {
    pub vertical_position: u8,
    pub data: &'a [u8],
    pub at_bit: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Motion {
    Frame,
    Field,
    Dual,
    /// 16×8 prediction, field pictures only.
    Half,
}

#[derive(Clone, Copy, Debug)]
struct Prediction {
    /// Forward, backward.
    dir: [bool; 2],
    motion: Motion,
    /// `vector[r][s][t]` in half samples, as motion compensation uses it.
    mv: [[[i32; 2]; 2]; 2],
    /// `motion_vertical_field_select[r][s]`.
    select: [[usize; 2]; 2],
    dmv: [i32; 2],
}

impl Prediction {
    fn new(motion: Motion) -> Self {
        Self {
            dir: [false; 2],
            motion,
            mv: [[[0; 2]; 2]; 2],
            select: [[0; 2]; 2],
            dmv: [0; 2],
        }
    }
}

/// Decodes every slice of one picture into `cur`. Macroblocks that no slice
/// covers keep the co-located samples of `fallback`, or mid-grey.
pub(crate) fn decode_picture(
    pic: &Picture<'_>,
    cur: &mut Frame,
    slices: &[SliceData<'_>],
    fallback: Option<&Frame>,
) -> Result<(), CodecError> {
    let mut dec = SliceDecoder::new(pic);
    for slice in slices {
        dec.slice(cur, slice)?;
    }
    // Every non-scalable MPEG-2 profile uses the restricted slice structure
    // (6.1.2.2, Table 8-1), so a gap is a damaged or truncated picture.
    // MPEG-1 allows gaps; those macroblocks keep the previous picture.
    if !pic.params.mpeg1 && dec.covered.iter().any(|&hit| !hit) {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "slices do not cover the picture",
        });
    }
    dec.fill_uncovered(cur, fallback);
    Ok(())
}

struct Blocks {
    coef: [[i32; 64]; 12],
    /// Only `F[0][0]` is non-zero.
    dc_only: [bool; 12],
    out: [[u8; 256]; 3],
}

struct SliceDecoder<'a> {
    pic: &'a Picture<'a>,
    mb_w: usize,
    mb_h: usize,
    quantiser_scale_code: u32,
    /// `W[w][pos] · quantiser_scale` for the four matrices.
    qw: [[i32; 64]; 4],
    dc_pred: [i32; 3],
    pmv: [[[i32; 2]; 2]; 2],
    /// Direction and vectors of the previous macroblock, for skips in B
    /// pictures. `None` after an intra macroblock.
    prev: Option<Prediction>,
    blocks: Box<Blocks>,
    covered: Vec<bool>,
    /// Macroblocks a picture may still decode or skip, against a hostile
    /// stream that repeats slices.
    budget: usize,
}

impl<'a> SliceDecoder<'a> {
    fn new(pic: &'a Picture<'a>) -> Self {
        let mb_w = pic.geo.mb_width;
        let mb_h = if pic.params.field.is_some() {
            pic.geo.mb_height / 2
        } else {
            pic.geo.mb_height
        };
        Self {
            pic,
            mb_w,
            mb_h,
            quantiser_scale_code: 0,
            qw: [[0; 64]; 4],
            dc_pred: [0; 3],
            pmv: [[[0; 2]; 2]; 2],
            prev: None,
            blocks: Box::new(Blocks {
                coef: [[0; 64]; 12],
                dc_only: [false; 12],
                out: [[0; 256]; 3],
            }),
            covered: vec![false; mb_w * mb_h],
            budget: 4 * mb_w * mb_h,
        }
    }

    fn params(&self) -> &PictureParams {
        &self.pic.params
    }

    fn set_quantiser(&mut self, code: u32) {
        if code == self.quantiser_scale_code {
            return;
        }
        self.quantiser_scale_code = code;
        let params = &self.pic.params;
        let scale = if params.mpeg1 {
            code as i32
        } else if params.q_scale_type {
            i32::from(NON_LINEAR_SCALE[code as usize])
        } else {
            2 * code as i32
        };
        for (table, matrix) in self.qw.iter_mut().zip(self.pic.matrices) {
            for (q, &w) in table.iter_mut().zip(matrix) {
                *q = i32::from(w) * scale;
            }
        }
    }

    fn reset_dc(&mut self) {
        let reset = 1 << (7 + self.params().intra_dc_precision);
        self.dc_pred = [reset; 3];
    }

    fn slice(&mut self, cur: &mut Frame, slice: &SliceData<'_>) -> Result<(), CodecError> {
        let mut r = BitReader::at(slice.data, slice.at_bit);
        let mut row = usize::from(slice.vertical_position) - 1;
        if !self.params().mpeg1 && self.params().vertical_size > 2800 {
            row += (r.read(3) as usize) << 7;
        }
        let at = r.at_bit();
        let code = r.read(5);
        if code == 0 {
            return Err(CodecError::Malformed {
                at_bit: at,
                what: "zero quantiser_scale_code",
            });
        }
        self.set_quantiser(code);
        // intra_slice_flag with its 8 bits, then extra_information_slice.
        while r.read_bit() {
            r.skip(8);
        }
        r.check("slice header")?;
        if row >= self.mb_h {
            return Err(r.malformed("slice_vertical_position"));
        }
        self.reset_dc();
        self.pmv = [[[0; 2]; 2]; 2];
        self.prev = None;
        let total = self.mb_w * self.mb_h;
        let mut addr: Option<usize> = None;
        loop {
            let increment = self.address_increment(&mut r)?;
            let next = match addr {
                None => row * self.mb_w + increment - 1,
                Some(prev) => prev + increment,
            };
            if next >= total {
                return Err(r.malformed("macroblock address past the picture"));
            }
            if let Some(prev) = addr {
                for skipped in prev + 1..next {
                    self.skipped(cur, skipped, &r)?;
                }
            }
            self.macroblock(&mut r, cur, next)?;
            addr = Some(next);
            r.check("macroblock")?;
            if r.bits_left() <= 0 || r.peek(23) == 0 {
                break;
            }
        }
        Ok(())
    }

    fn address_increment(&self, r: &mut BitReader<'_>) -> Result<usize, CodecError> {
        let mut increment = 0usize;
        loop {
            let entry = MB_ADDR[r.peek(MB_ADDR_BITS) as usize];
            if entry.len == 0 {
                return Err(r.malformed("macroblock_address_increment"));
            }
            r.skip(u32::from(entry.len));
            match entry.value {
                MB_ESCAPE => increment += 33,
                MB_STUFFING if self.params().mpeg1 => {}
                MB_STUFFING => return Err(r.malformed("macroblock stuffing in MPEG-2")),
                value => return Ok(increment + value as usize),
            }
            if r.overrun() {
                return Err(r.truncated("macroblock_address_increment"));
            }
        }
    }

    fn spend(&mut self) -> Result<(), CodecError> {
        if self.budget == 0 {
            return Err(CodecError::Limit {
                what: "macroblocks in one picture",
            });
        }
        self.budget -= 1;
        Ok(())
    }

    fn skipped(
        &mut self,
        cur: &mut Frame,
        addr: usize,
        r: &BitReader<'_>,
    ) -> Result<(), CodecError> {
        self.spend()?;
        let params = *self.params();
        let frame_pic = params.field.is_none();
        let prediction = match params.coding_type {
            PICTURE_P => {
                self.pmv = [[[0; 2]; 2]; 2];
                let mut p = Prediction::new(if frame_pic {
                    Motion::Frame
                } else {
                    Motion::Field
                });
                p.dir = [true, false];
                p.select[0][0] = params.field.unwrap_or(0);
                p
            }
            PICTURE_B => {
                let Some(prev) = self.prev else {
                    return Err(r.malformed("skipped macroblock after an intra macroblock"));
                };
                let mut p = Prediction::new(if frame_pic {
                    Motion::Frame
                } else {
                    Motion::Field
                });
                p.dir = prev.dir;
                for s in 0..2 {
                    for t in 0..2 {
                        p.mv[0][s][t] = self.pmv[0][s][t] << u32::from(params.full_pel[s]);
                    }
                    p.select[0][s] = params.field.unwrap_or(0);
                }
                p
            }
            _ => return Err(r.malformed("skipped macroblock in an intra picture")),
        };
        self.reset_dc();
        self.reconstruct(cur, addr, Some(&prediction), 0, false, false);
        Ok(())
    }

    fn macroblock(
        &mut self,
        r: &mut BitReader<'_>,
        cur: &mut Frame,
        addr: usize,
    ) -> Result<(), CodecError> {
        self.spend()?;
        let params = *self.params();
        let frame_pic = params.field.is_none();
        let table: &[Vlc] = match params.coding_type {
            PICTURE_P => &MB_TYPE_P,
            PICTURE_B => &MB_TYPE_B,
            _ => &MB_TYPE_I,
        };
        let entry = table[r.peek(MB_TYPE_BITS) as usize];
        if entry.len == 0 {
            return Err(r.malformed("macroblock_type"));
        }
        r.skip(u32::from(entry.len));
        let flags = entry.value;
        let intra = flags & MB_INTRA != 0;
        let pattern = flags & MB_PATTERN != 0;
        let forward = flags & MB_FORWARD != 0;
        let backward = flags & MB_BACKWARD != 0;
        let mut motion = if frame_pic {
            Motion::Frame
        } else {
            Motion::Field
        };
        if forward || backward {
            if frame_pic {
                if !params.frame_pred_frame_dct {
                    motion = match r.read(2) {
                        1 => Motion::Field,
                        2 => Motion::Frame,
                        3 => Motion::Dual,
                        _ => return Err(r.malformed("frame_motion_type")),
                    };
                }
            } else {
                motion = match r.read(2) {
                    1 => Motion::Field,
                    2 => Motion::Half,
                    3 => Motion::Dual,
                    _ => return Err(r.malformed("field_motion_type")),
                };
            }
            if motion == Motion::Dual && (backward || params.coding_type != PICTURE_P) {
                return Err(r.malformed("dual prime outside a P picture"));
            }
        }
        let dct_field =
            frame_pic && !params.frame_pred_frame_dct && (intra || pattern) && r.read_bit();
        if flags & MB_QUANT != 0 {
            let at = r.at_bit();
            let code = r.read(5);
            if code == 0 {
                return Err(CodecError::Malformed {
                    at_bit: at,
                    what: "zero quantiser_scale_code",
                });
            }
            self.set_quantiser(code);
        }
        let mut p = Prediction::new(motion);
        p.dir = [forward, backward];
        if forward || (intra && params.concealment) {
            self.motion_vectors(r, 0, &mut p)?;
        }
        if backward {
            self.motion_vectors(r, 1, &mut p)?;
        }
        if intra && params.concealment {
            r.skip(1);
        }
        let count = self.pic.geo.blocks_per_mb();
        let coded: u32 = if intra {
            (1 << count) - 1
        } else if pattern {
            let entry = CBP[r.peek(CBP_BITS) as usize];
            if entry.len == 0 {
                return Err(r.malformed("coded_block_pattern"));
            }
            r.skip(u32::from(entry.len));
            let mut coded = (entry.value as u32) << (count - 6);
            if count > 6 {
                coded |= r.read(count as u32 - 6);
            }
            coded
        } else {
            0
        };
        // Blocks.
        if params.coding_type == PICTURE_D {
            for i in 0..6 {
                self.dc_block(r, i)?;
            }
            if !r.read_bit() {
                return Err(r.malformed("end_of_macroblock"));
            }
        } else {
            for i in 0..count {
                if coded >> (count - 1 - i) & 1 == 1 {
                    self.block(r, i, intra)?;
                }
            }
        }
        // Predictor upkeep (7.2.1, 7.6.3.3, 7.6.3.4).
        if intra {
            if params.concealment {
                self.pmv[1][0] = self.pmv[0][0];
            } else {
                self.pmv = [[[0; 2]; 2]; 2];
            }
            self.prev = None;
        } else {
            self.reset_dc();
            match (motion, frame_pic) {
                (Motion::Frame, true) | (Motion::Field, false) => {
                    if forward {
                        self.pmv[1][0] = self.pmv[0][0];
                    }
                    if backward {
                        self.pmv[1][1] = self.pmv[0][1];
                    }
                }
                (Motion::Dual, _) => self.pmv[1][0] = self.pmv[0][0],
                _ => {}
            }
            if params.coding_type == PICTURE_P && !forward {
                self.pmv = [[[0; 2]; 2]; 2];
                p.dir = [true, false];
                p.select[0][0] = params.field.unwrap_or(0);
            }
            self.prev = Some(p);
        }
        let prediction = (!intra).then_some(&p);
        self.reconstruct(cur, addr, prediction, coded, dct_field, intra);
        Ok(())
    }

    fn motion_vectors(
        &mut self,
        r: &mut BitReader<'_>,
        s: usize,
        p: &mut Prediction,
    ) -> Result<(), CodecError> {
        let frame_pic = self.params().field.is_none();
        let (count, field_format, dmv) = match (frame_pic, p.motion) {
            (true, Motion::Frame) => (1, false, false),
            (true, Motion::Field) | (_, Motion::Half) => (2, true, false),
            (_, Motion::Dual) => (1, true, true),
            (false, _) => (1, true, false),
        };
        if count == 1 {
            if field_format && !dmv {
                p.select[0][s] = r.read(1) as usize;
            }
            self.motion_vector(r, 0, s, field_format, dmv, p)
        } else {
            for rr in 0..2 {
                p.select[rr][s] = r.read(1) as usize;
                self.motion_vector(r, rr, s, true, false, p)?;
            }
            Ok(())
        }
    }

    fn motion_vector(
        &mut self,
        r: &mut BitReader<'_>,
        rr: usize,
        s: usize,
        field_format: bool,
        dmv: bool,
        p: &mut Prediction,
    ) -> Result<(), CodecError> {
        let params = *self.params();
        let vertical_field = field_format && params.field.is_none();
        for t in 0..2 {
            let entry = MOTION[r.peek(MOTION_BITS) as usize];
            if entry.len == 0 {
                return Err(r.malformed("motion_code"));
            }
            r.skip(u32::from(entry.len));
            let f_code = params.f_code[s][t];
            if f_code == 0 || f_code > 9 {
                return Err(r.malformed("f_code"));
            }
            let r_size = u32::from(f_code - 1);
            let code = i32::from(entry.value);
            let delta = if r_size == 0 || code == 0 {
                code
            } else {
                let residual = r.read(r_size) as i32;
                let magnitude = ((code.abs() - 1) << r_size) + residual + 1;
                if code < 0 { -magnitude } else { magnitude }
            };
            if dmv {
                let entry = DMV[r.peek(2) as usize];
                r.skip(u32::from(entry.len));
                p.dmv[t] = i32::from(entry.value);
            }
            let halve = vertical_field && t == 1;
            let pmv = self.pmv[rr][s][t];
            let prediction = if halve { pmv >> 1 } else { pmv };
            let f = 1i32 << r_size;
            let mut vector = prediction + delta;
            if vector < -16 * f {
                vector += 32 * f;
            } else if vector > 16 * f - 1 {
                vector -= 32 * f;
            }
            self.pmv[rr][s][t] = if halve { vector * 2 } else { vector };
            p.mv[rr][s][t] = vector << u32::from(params.full_pel[s]);
        }
        Ok(())
    }

    fn coef_table(&self, intra: bool) -> &'static CoefTable {
        if intra && self.params().intra_vlc {
            &COEF_TABLE_ONE
        } else {
            &COEF_TABLE_ZERO
        }
    }

    fn matrix_index(&self, intra: bool, cc: usize) -> usize {
        let chroma = cc > 0 && self.pic.geo.chroma_format != CHROMA_420;
        usize::from(!intra) + if chroma { 2 } else { 0 }
    }

    /// `dct_dc_size` and `dct_dc_differential`, added to the predictor.
    fn intra_dc(&mut self, r: &mut BitReader<'_>, cc: usize) -> Result<i32, CodecError> {
        let table = if cc == 0 { &DC_LUMA } else { &DC_CHROMA };
        let entry = table[r.peek(DC_BITS) as usize];
        if entry.len == 0 {
            return Err(r.malformed("dct_dc_size"));
        }
        r.skip(u32::from(entry.len));
        let size = entry.value as u32;
        let diff = if size == 0 {
            0
        } else {
            let value = r.read(size) as i32;
            if value >= 1 << (size - 1) {
                value
            } else {
                value + 1 - (1 << size)
            }
        };
        let dc = self.dc_pred[cc] + diff;
        let precision = u32::from(self.params().intra_dc_precision);
        if !(0..1 << (8 + precision)).contains(&dc) {
            return Err(r.malformed("intra DC out of range"));
        }
        self.dc_pred[cc] = dc;
        Ok(dc << (3 - precision))
    }

    /// An 11172-2 D-picture block: the DC term alone.
    fn dc_block(&mut self, r: &mut BitReader<'_>, i: usize) -> Result<(), CodecError> {
        let cc = component(i);
        let dc = self.intra_dc(r, cc)?;
        self.blocks.coef[i][0] = dc;
        self.blocks.dc_only[i] = true;
        Ok(())
    }

    fn block(&mut self, r: &mut BitReader<'_>, i: usize, intra: bool) -> Result<(), CodecError> {
        let params = *self.params();
        let cc = component(i);
        let mut parity = 0i32;
        let mut n = 0usize;
        let mut ac = false;
        if intra {
            let dc = self.intra_dc(r, cc)?;
            self.blocks.coef[i][0] = dc;
            parity ^= dc;
            n = 1;
        }
        let table = self.coef_table(intra);
        let scan = &SCAN[usize::from(params.alternate_scan)];
        let qw = &self.qw[self.matrix_index(intra, cc)];
        let block = &mut self.blocks.coef[i];
        let mut first = !intra;
        loop {
            let (run, level) = if first && r.peek(1) == 1 {
                r.skip(1);
                (0, if r.read_bit() { -1 } else { 1 })
            } else {
                let window = r.peek(16);
                let mut entry = table.short[(window >> 8) as usize];
                if entry.len == 0 {
                    entry = table.long[(window & 0x3ff) as usize];
                    if entry.len == 0 {
                        return Err(r.malformed("DCT coefficient code"));
                    }
                }
                r.skip(u32::from(entry.len));
                match entry.run {
                    RUN_EOB if !first => break,
                    RUN_EOB => return Err(r.malformed("End of Block first")),
                    RUN_ESCAPE => escape(r, params.mpeg1)?,
                    run => {
                        let level = i32::from(entry.level);
                        (u32::from(run), if r.read_bit() { -level } else { level })
                    }
                }
            };
            first = false;
            n += run as usize;
            if n > 63 {
                return Err(r.malformed("DCT coefficients past the block"));
            }
            let pos = usize::from(scan[n]);
            let value = dequantise(level, qw[pos], intra, params.mpeg1);
            block[pos] = value;
            parity ^= value;
            ac |= pos != 0;
            n += 1;
        }
        if !params.mpeg1 && parity & 1 == 0 {
            block[63] ^= 1;
            ac |= block[63] != 0;
        }
        self.blocks.dc_only[i] = !ac;
        r.check("block")
    }

    /// Prediction, residual, and the write into `cur`.
    fn reconstruct(
        &mut self,
        cur: &mut Frame,
        addr: usize,
        prediction: Option<&Prediction>,
        coded: u32,
        dct_field: bool,
        intra: bool,
    ) {
        let geo = self.pic.geo;
        let (mbx, mby) = (addr % self.mb_w, addr / self.mb_w);
        if let Some(p) = prediction {
            predict_mb(self.pic, cur, &mut self.blocks.out, mbx, mby, p);
        }
        let count = geo.blocks_per_mb();
        for i in 0..count {
            if coded >> (count - 1 - i) & 1 == 0 {
                continue;
            }
            let (c, x0, y0, step) = place(&geo, i, dct_field);
            let stride = geo.mb_size(c).0;
            let blocks = &mut *self.blocks;
            let coef = &mut blocks.coef[i];
            let out = &mut blocks.out[c];
            if blocks.dc_only[i] {
                let v = idct_dc(coef[0]);
                coef[0] = 0;
                for j in 0..8 {
                    let row = &mut out[(y0 + j * step) * stride + x0..][..8];
                    for px in row {
                        *px = if intra { v } else { i32::from(*px) + v }.clamp(0, 255) as u8;
                    }
                }
            } else {
                idct(coef);
                for j in 0..8 {
                    let row = &mut out[(y0 + j * step) * stride + x0..][..8];
                    for (px, &v) in row.iter_mut().zip(&coef[j * 8..j * 8 + 8]) {
                        *px = if intra { v } else { i32::from(*px) + v }.clamp(0, 255) as u8;
                    }
                }
                coef.fill(0);
            }
        }
        // Store.
        for c in 0..3 {
            let (bw, bh) = geo.mb_size(c);
            let (stride, _) = geo.plane(c);
            let (line, base) = match self.pic.params.field {
                None => (stride, 0),
                Some(parity) => (2 * stride, parity * stride),
            };
            let plane = &mut cur.planes[c];
            let out = &self.blocks.out[c];
            for j in 0..bh {
                let at = base + (mby * bh + j) * line + mbx * bw;
                plane[at..at + bw].copy_from_slice(&out[j * bw..j * bw + bw]);
            }
        }
        self.covered[addr] = true;
    }

    fn fill_uncovered(&self, cur: &mut Frame, fallback: Option<&Frame>) {
        let geo = self.pic.geo;
        for (addr, _) in self.covered.iter().enumerate().filter(|&(_, hit)| !hit) {
            let (mbx, mby) = (addr % self.mb_w, addr / self.mb_w);
            for c in 0..3 {
                let (bw, bh) = geo.mb_size(c);
                let (stride, _) = geo.plane(c);
                let (line, base) = match self.pic.params.field {
                    None => (stride, 0),
                    Some(parity) => (2 * stride, parity * stride),
                };
                for j in 0..bh {
                    let at = base + (mby * bh + j) * line + mbx * bw;
                    match fallback {
                        Some(frame) => {
                            cur.planes[c][at..at + bw]
                                .copy_from_slice(&frame.planes[c][at..at + bw]);
                        }
                        None => cur.planes[c][at..at + bw].fill(128),
                    }
                }
            }
        }
    }
}

/// Colour component of block `i` (Table 7-1).
fn component(i: usize) -> usize {
    if i < 4 { 0 } else { 1 + (i - 4) % 2 }
}

/// Plane, left column, first row, and row step of block `i` inside the
/// macroblock buffer, for frame or field DCT (6.1.3).
fn place(geo: &Geometry, i: usize, dct_field: bool) -> (usize, usize, usize, usize) {
    let (k, c) = if i < 4 {
        (i, 0)
    } else {
        ((i - 4) / 2, component(i))
    };
    let (column, vertical) = if c == 0 {
        (k & 1, k >> 1)
    } else {
        match geo.chroma_format {
            CHROMA_420 => return (c, 0, 0, 1),
            CHROMA_422 => (0, k),
            // Figure 6-12. Cb blocks 4, 6, 8, 10 are the top-left,
            // bottom-left, top-right and bottom-right; Cr is 5, 7, 9, 11.
            _ => (k >> 1, k & 1),
        }
    };
    if dct_field {
        (c, column * 8, vertical, 2)
    } else {
        (c, column * 8, vertical * 8, 1)
    }
}

/// Escape-coded run and signed level: 6 + 12 bits in H.262, 6 + 8 or 16
/// bits in 11172-2.
fn escape(r: &mut BitReader<'_>, mpeg1: bool) -> Result<(u32, i32), CodecError> {
    let run = r.read(6);
    let level = if mpeg1 {
        match r.read(8) {
            0x80 => r.read(8) as i32 - 256,
            0 => r.read(8) as i32,
            byte => i32::from(byte as u8 as i8),
        }
    } else {
        ((r.read(12) << 20) as i32) >> 20
    };
    if level == 0 {
        return Err(r.malformed("zero escape level"));
    }
    Ok((run, level))
}

/// Inverse quantisation and saturation of one coefficient (7.4, and the
/// odd-ified 11172-2 rule).
#[inline(always)]
fn dequantise(level: i32, qw: i32, intra: bool, mpeg1: bool) -> i32 {
    let abs = level.abs();
    let magnitude = if mpeg1 {
        let v = if intra {
            (abs * qw) >> 3
        } else {
            ((2 * abs + 1) * qw) >> 4
        };
        if v & 1 == 0 && v != 0 { v - 1 } else { v }
    } else if intra {
        (abs * qw) >> 4
    } else {
        ((2 * abs + 1) * qw) >> 5
    };
    if level < 0 {
        -magnitude.min(2048)
    } else {
        magnitude.min(2047)
    }
}

fn chroma_vector(geo: &Geometry, c: usize, mv: [i32; 2]) -> [i32; 2] {
    if c == 0 {
        return mv;
    }
    match geo.chroma_format {
        CHROMA_420 => [mv[0] / 2, mv[1] / 2],
        CHROMA_422 => [mv[0] / 2, mv[1]],
        _ => mv,
    }
}

/// `((vector · m) // 2) + e + dmvector` of 7.6.3.6, `//` rounding half away from zero.
fn dual_prime(mv: [i32; 2], dmv: [i32; 2], m: i32, e: i32) -> [i32; 2] {
    let scale = |v: i32| (v * m + i32::from(v > 0)) >> 1;
    [scale(mv[0]) + dmv[0], scale(mv[1]) + e + dmv[1]]
}

/// The frame that field `parity` of the `s` reference comes from. The second
/// field of a P frame predicts the opposite parity from its own first field.
fn reference<'a>(pic: &Picture<'a>, cur: &'a Frame, s: usize, parity: usize) -> &'a Frame {
    if s == 1 {
        return pic.bwd.unwrap_or(cur);
    }
    if let Some(current) = pic.params.field
        && pic.params.coding_type == PICTURE_P
        && pic.params.second_field
        && parity != current
    {
        return cur;
    }
    pic.fwd.unwrap_or(cur)
}

fn predict_mb(
    pic: &Picture<'_>,
    cur: &Frame,
    out: &mut [[u8; 256]; 3],
    mbx: usize,
    mby: usize,
    p: &Prediction,
) {
    let geo = &pic.geo;
    let mut average = false;
    for s in 0..2 {
        if !p.dir[s] {
            continue;
        }
        for (c, plane) in out.iter_mut().enumerate() {
            let (bw, bh) = geo.mb_size(c);
            let (x, y) = ((mbx * bw) as i32, (mby * bh) as i32);
            match (p.motion, pic.params.field) {
                (Motion::Frame, _) => {
                    let frame = reference(pic, cur, s, 0);
                    let mut dst = Target {
                        buf: &mut plane[..],
                        offset: 0,
                        stride: bw,
                    };
                    let mv = chroma_vector(geo, c, p.mv[0][s]);
                    predict(
                        &mut dst,
                        View::frame(frame, geo, c),
                        [x, y],
                        mv,
                        [bw, bh],
                        average,
                    );
                }
                (Motion::Field, None) => {
                    for r in 0..2 {
                        let select = p.select[r][s];
                        let frame = reference(pic, cur, s, select);
                        let mut dst = Target {
                            buf: &mut plane[..],
                            offset: r * bw,
                            stride: 2 * bw,
                        };
                        let mv = chroma_vector(geo, c, p.mv[r][s]);
                        let src = View::field(frame, geo, c, select);
                        predict(&mut dst, src, [x, y / 2], mv, [bw, bh / 2], average);
                    }
                }
                (Motion::Field, Some(_)) => {
                    let select = p.select[0][s];
                    let frame = reference(pic, cur, s, select);
                    let mut dst = Target {
                        buf: &mut plane[..],
                        offset: 0,
                        stride: bw,
                    };
                    let mv = chroma_vector(geo, c, p.mv[0][s]);
                    let src = View::field(frame, geo, c, select);
                    predict(&mut dst, src, [x, y], mv, [bw, bh], average);
                }
                (Motion::Half, _) => {
                    for r in 0..2 {
                        let select = p.select[r][s];
                        let frame = reference(pic, cur, s, select);
                        let mut dst = Target {
                            buf: &mut plane[..],
                            offset: r * (bh / 2) * bw,
                            stride: bw,
                        };
                        let mv = chroma_vector(geo, c, p.mv[r][s]);
                        let src = View::field(frame, geo, c, select);
                        let top = y + (r * bh / 2) as i32;
                        predict(&mut dst, src, [x, top], mv, [bw, bh / 2], average);
                    }
                }
                (Motion::Dual, None) => {
                    let frame = reference(pic, cur, 0, 0);
                    let (m_top, m_bottom) = if pic.params.top_field_first {
                        (1, 3)
                    } else {
                        (3, 1)
                    };
                    let derived = [
                        dual_prime(p.mv[0][0], p.dmv, m_top, -1),
                        dual_prime(p.mv[0][0], p.dmv, m_bottom, 1),
                    ];
                    for (r, &vector) in derived.iter().enumerate() {
                        let mut dst = Target {
                            buf: &mut plane[..],
                            offset: r * bw,
                            stride: 2 * bw,
                        };
                        let same = chroma_vector(geo, c, p.mv[0][0]);
                        let src = View::field(frame, geo, c, r);
                        predict(&mut dst, src, [x, y / 2], same, [bw, bh / 2], false);
                        let opposite = chroma_vector(geo, c, vector);
                        let src = View::field(frame, geo, c, 1 - r);
                        predict(&mut dst, src, [x, y / 2], opposite, [bw, bh / 2], true);
                    }
                }
                (Motion::Dual, Some(parity)) => {
                    let e = if parity == 0 { -1 } else { 1 };
                    let derived = dual_prime(p.mv[0][0], p.dmv, 1, e);
                    let mut dst = Target {
                        buf: &mut plane[..],
                        offset: 0,
                        stride: bw,
                    };
                    let same = reference(pic, cur, 0, parity);
                    let mv = chroma_vector(geo, c, p.mv[0][0]);
                    predict(
                        &mut dst,
                        View::field(same, geo, c, parity),
                        [x, y],
                        mv,
                        [bw, bh],
                        false,
                    );
                    let opposite = reference(pic, cur, 0, 1 - parity);
                    let mv = chroma_vector(geo, c, derived);
                    let src = View::field(opposite, geo, c, 1 - parity);
                    predict(&mut dst, src, [x, y], mv, [bw, bh], true);
                }
            }
        }
        average = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequantisation_follows_both_standards() {
        // H.262 intra: (QF · W · q · 2) / 32.
        assert_eq!(dequantise(3, 16 * 8, true, false), 24);
        assert_eq!(dequantise(-3, 16 * 8, true, false), -24);
        // H.262 non-intra: ((2 · QF + 1) · W · q) / 32, toward zero.
        assert_eq!(dequantise(1, 16 * 2, false, false), 3);
        assert_eq!(dequantise(-1, 16 * 2, false, false), -3);
        // Saturation.
        assert_eq!(dequantise(2047, 255 * 112, false, false), 2047);
        assert_eq!(dequantise(-2048, 255 * 112, false, false), -2048);
        // 11172-2 intra: (2 · QF · q · W) / 16, then odd toward zero.
        assert_eq!(dequantise(1, 16 * 2, true, true), 3);
        assert_eq!(dequantise(-2, 16, true, true), -3);
        // A zero stays zero.
        assert_eq!(dequantise(1, 1, true, true), 0);
        // 11172-2 non-intra: ((2 · QF + 1) · q · W) / 16, odd.
        assert_eq!(dequantise(1, 16, false, true), 3);
        assert_eq!(dequantise(2, 16, false, true), 5);
    }

    #[test]
    fn dual_prime_rounds_half_away_from_zero() {
        assert_eq!(dual_prime([3, 3], [0, 0], 1, 0), [2, 2]);
        assert_eq!(dual_prime([-3, -3], [0, 0], 1, 0), [-2, -2]);
        assert_eq!(dual_prime([-1, 1], [1, -1], 3, 1), [-1, 2]);
        assert_eq!(dual_prime([4, -4], [0, 0], 3, -1), [6, -7]);
    }

    #[test]
    fn chroma_vectors_truncate_toward_zero() {
        let geo = Geometry {
            mb_width: 1,
            mb_height: 1,
            chroma_format: CHROMA_420,
        };
        assert_eq!(chroma_vector(&geo, 1, [-3, 3]), [-1, 1]);
        let geo = Geometry {
            chroma_format: CHROMA_422,
            ..geo
        };
        assert_eq!(chroma_vector(&geo, 2, [-3, -3]), [-1, -3]);
        let geo = Geometry {
            chroma_format: crate::headers::CHROMA_444,
            ..geo
        };
        assert_eq!(chroma_vector(&geo, 1, [-3, 5]), [-3, 5]);
    }

    #[test]
    fn block_places_follow_figures_6_10_to_6_14() {
        let geo420 = Geometry {
            mb_width: 1,
            mb_height: 1,
            chroma_format: CHROMA_420,
        };
        assert_eq!(place(&geo420, 3, false), (0, 8, 8, 1));
        assert_eq!(place(&geo420, 2, true), (0, 0, 1, 2));
        assert_eq!(place(&geo420, 5, true), (2, 0, 0, 1));
        let geo422 = Geometry {
            chroma_format: CHROMA_422,
            ..geo420
        };
        assert_eq!(place(&geo422, 6, false), (1, 0, 8, 1));
        assert_eq!(place(&geo422, 7, true), (2, 0, 1, 2));
        let geo444 = Geometry {
            chroma_format: crate::headers::CHROMA_444,
            ..geo420
        };
        assert_eq!(place(&geo444, 6, false), (1, 0, 8, 1));
        assert_eq!(place(&geo444, 8, false), (1, 8, 0, 1));
        assert_eq!(place(&geo444, 11, false), (2, 8, 8, 1));
        assert_eq!(place(&geo444, 6, true), (1, 0, 1, 2));
    }
}
