//! Keyframe encoder for 8-bit 4:2:0.
//!
//! Blocks are DC predicted and transformed with the 2D DCT. The loop filter,
//! CDEF, restoration, super resolution, and film grain stay off. A repeated
//! picture is emitted as a frame that shows an existing reference.

use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use cutlass_encode::{
    AudioInput, CodecError, CodecId, CodecParams, Encoder, FrameInput, Packet, PixelFormat,
};

use crate::yuv::VideoInput;

use crate::bits::{BitWriter, floor_log2, write_leb128};
use crate::dims::{
    BLOCK_8X8, DC_PRED, DCT_DCT, MAX_TX_SIZE_RECT, MI_HEIGHT_LOG2, MI_WIDTH_LOG2, NUM_4X4_HIGH,
    NUM_4X4_WIDE, PARTITION_NONE, PARTITION_SPLIT, TRANSFORM_ROW_SHIFT, TX_4X4, TX_8X8, TX_16X16,
    TX_32X32, TX_HEIGHT, TX_HEIGHT_LOG2, TX_SIZE_SQR, TX_SIZE_SQR_UP, TX_WIDTH, TX_WIDTH_LOG2,
    block_height, block_width, subsampled_size,
};
use crate::idct::inverse_2d;
use crate::predict::{IntraPred, predict_intra};
use crate::symbol::SymbolEncoder;
use crate::tables::{
    AC_QLOOKUP, DC_QLOOKUP, DEFAULT_COEFF_BASE_CDF, DEFAULT_COEFF_BASE_EOB_CDF,
    DEFAULT_COEFF_BR_CDF, DEFAULT_DC_SIGN_CDF, DEFAULT_EOB_EXTRA_CDF, DEFAULT_EOB_PT_16_CDF,
    DEFAULT_EOB_PT_32_CDF, DEFAULT_EOB_PT_64_CDF, DEFAULT_EOB_PT_128_CDF, DEFAULT_EOB_PT_256_CDF,
    DEFAULT_EOB_PT_512_CDF, DEFAULT_EOB_PT_1024_CDF, DEFAULT_INTRA_FRAME_Y_MODE_CDF,
    DEFAULT_INTRA_TX_TYPE_SET1_CDF, DEFAULT_INTRA_TX_TYPE_SET2_CDF, DEFAULT_PARTITION_W8_CDF,
    DEFAULT_PARTITION_W16_CDF, DEFAULT_PARTITION_W32_CDF, DEFAULT_PARTITION_W64_CDF,
    DEFAULT_PARTITION_W128_CDF, DEFAULT_SKIP_CDF, DEFAULT_TXB_SKIP_CDF,
    DEFAULT_UV_MODE_CFL_ALLOWED_CDF, DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF,
};
use crate::tile::{
    MODE_CTX, Neighbor, br_ctx, chroma_coeff_ctx, coeff_base_ctx, edge_partition_cdf,
    luma_coeff_ctx, partition_shape, row_range, scan_for, slice_q,
};

const SB_MI: usize = 16;
const ROOT: usize = 12;

/// Rust AV1 encoder. `bitrate` of zero keeps the quality index.
pub struct Av1Encoder {
    width: u32,
    height: u32,
    bitrate: u32,
    start_q: u8,
    q: u8,
    previous: Option<Vec<u8>>,
    bits_out: u64,
    ticks: u64,
    cancel: Arc<AtomicBool>,
}

impl Av1Encoder {
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        if params.codec != CodecId::Av1 {
            return Err(CodecError::Unsupported { what: "codec" });
        }
        if params.pixel != PixelFormat::Yuv420p {
            return Err(CodecError::Unsupported {
                what: "pixel format",
            });
        }
        if params.width == 0 || params.height == 0 || params.width > 4096 || params.height > 4096 {
            return Err(CodecError::Limit { what: "frame size" });
        }
        let q = if params.bitrate == 0 { 40 } else { 220 };
        Ok(Self {
            width: params.width,
            height: params.height,
            bitrate: params.bitrate,
            start_q: q,
            q,
            previous: None,
            bits_out: 0,
            ticks: 0,
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Quantizer index used when `bitrate` is zero. Zero is raised to one.
    pub fn set_quality(&mut self, qindex: u8) {
        let q = qindex.clamp(1, 255);
        self.start_q = q;
        self.q = q;
    }

    /// Encode one 4:2:0 picture. Odd luma edges stay inside this type.
    pub fn push_picture(&mut self, input: &VideoInput) -> Result<Vec<Packet>, CodecError> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(CodecError::Cancelled);
        }
        if input.picture.y.width() != self.width || input.picture.y.height() != self.height {
            return Err(CodecError::InvalidApi {
                what: "picture size",
            });
        }
        let packed = pack(input);
        let repeat = self.previous.as_deref() == Some(packed.as_slice());
        let data = if repeat {
            show_existing_packet()?
        } else {
            let q = self.q;
            let tile = FrameCode::new(input, q)?.encode()?;
            let mut payload = key_header(self.width, self.height, q)?;
            payload.extend(tile);
            let mut unit = obu(2, &[])?;
            unit.extend(obu(1, &sequence_header(self.width, self.height)?)?);
            unit.extend(obu(6, &payload)?);
            self.previous = Some(packed);
            unit
        };
        self.note_rate(data.len(), input.duration, input.timescale);
        Ok(vec![packet(data, input, repeat)])
    }
}

impl Encoder for Av1Encoder {
    fn codec(&self) -> CodecId {
        CodecId::Av1
    }

    fn push(&mut self, _input: &AudioInput) -> Result<Vec<Packet>, CodecError> {
        Err(CodecError::Unsupported { what: "audio" })
    }

    fn push_frame(&mut self, input: &FrameInput) -> Result<Vec<Packet>, CodecError> {
        let FrameInput::Cpu(frame) = input else {
            return Err(CodecError::Unsupported {
                what: "platform frame",
            });
        };
        if frame.pixel() != PixelFormat::Yuv420p {
            return Err(CodecError::Unsupported {
                what: "pixel format",
            });
        }
        let picture = yuv_from_packed(frame.width(), frame.height(), frame.bytes())?;
        let time = frame.time();
        self.push_picture(&VideoInput {
            picture,
            pts: time.pts(),
            duration: time.duration(),
            timescale: time.timescale(),
        })
    }

    fn drain(&mut self) -> Result<Vec<Packet>, CodecError> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        self.previous = None;
        self.bits_out = 0;
        self.ticks = 0;
        self.q = self.start_q;
    }

    fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }
}

impl Av1Encoder {
    fn note_rate(&mut self, nbytes: usize, duration: i64, timescale: u32) {
        if self.bitrate == 0 || timescale == 0 || duration <= 0 {
            return;
        }
        self.bits_out = self
            .bits_out
            .saturating_add((nbytes as u64).saturating_mul(8));
        self.ticks = self.ticks.saturating_add(duration as u64);
        let expected = u64::from(self.bitrate).saturating_mul(self.ticks) / u64::from(timescale);
        if expected == 0 {
            return;
        }
        if self.bits_out > expected.saturating_mul(14) / 10 {
            self.q = self.q.saturating_add(6);
        } else if self.bits_out.saturating_mul(10) < expected.saturating_mul(6) {
            self.q = self.q.saturating_sub(6).max(1);
        }
    }
}

fn yuv_from_packed(
    width: u32,
    height: u32,
    bytes: &[u8],
) -> Result<crate::yuv::Yuv420, CodecError> {
    if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
        return Err(CodecError::InvalidApi { what: "4:2:0 size" });
    }
    let y_len = usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or(CodecError::Limit { what: "frame size" })?;
    let chroma_w = width / 2;
    let chroma_h = height / 2;
    let chroma_len = usize::try_from(chroma_w)
        .ok()
        .and_then(|width| {
            usize::try_from(chroma_h)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or(CodecError::Limit { what: "frame size" })?;
    if bytes.len() != y_len + chroma_len + chroma_len {
        return Err(CodecError::InvalidApi {
            what: "picture length",
        });
    }
    crate::yuv::Yuv420::new(
        crate::yuv::Plane::new(width, height, bytes[..y_len].to_vec())?,
        crate::yuv::Plane::new(
            chroma_w,
            chroma_h,
            bytes[y_len..y_len + chroma_len].to_vec(),
        )?,
        crate::yuv::Plane::new(chroma_w, chroma_h, bytes[y_len + chroma_len..].to_vec())?,
    )
}

fn packet(data: Vec<u8>, input: &VideoInput, repeat: bool) -> Packet {
    Packet {
        stream: 0,
        data,
        pts: input.pts,
        duration: input.duration,
        timescale: input.timescale,
        keyframe: !repeat,
        side: Vec::new(),
    }
}

fn pack(input: &VideoInput) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(input.picture.y.samples());
    out.extend_from_slice(input.picture.u.samples());
    out.extend_from_slice(input.picture.v.samples());
    out
}

fn show_existing_packet() -> Result<Vec<u8>, CodecError> {
    let mut bits = BitWriter::new();
    bits.write_bool(true)?;
    bits.write(0, 3)?;
    bits.trailing_bits()?;
    let mut unit = obu(2, &[])?;
    unit.extend(obu(6, &bits.finish())?);
    Ok(unit)
}

struct CoeffTables {
    txb_skip: Vec<u16>,
    eob: [Vec<u16>; 7],
    eob_extra: Vec<u16>,
    dc_sign: Vec<u16>,
    coeff_eob: Vec<u16>,
    coeff_base: Vec<u16>,
    coeff_br: Vec<u16>,
}

struct Job {
    plane: usize,
    x: usize,
    y: usize,
    tx: usize,
    levels: Vec<i32>,
}

struct FrameCode {
    symbols: SymbolEncoder,
    tables: CoeffTables,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    src_y: Vec<u8>,
    src_u: Vec<u8>,
    src_v: Vec<u8>,
    visible_w: usize,
    visible_h: usize,
    buf_w: usize,
    buf_h: usize,
    chroma_w: usize,
    chroma_h: usize,
    mi_cols: usize,
    mi_rows: usize,
    modes: Vec<u8>,
    skips: Vec<u8>,
    sizes: Vec<u8>,
    above_level: [Vec<u8>; 3],
    left_level: [Vec<u8>; 3],
    above_dc: [Vec<u8>; 3],
    left_dc: [Vec<u8>; 3],
    dc_q: i32,
    ac_q: i32,
}

impl FrameCode {
    fn new(input: &VideoInput, q: u8) -> Result<Self, CodecError> {
        let visible_w = input.picture.y.width() as usize;
        let visible_h = input.picture.y.height() as usize;
        let mi_cols = 2 * ((visible_w + 7) >> 3);
        let mi_rows = 2 * ((visible_h + 7) >> 3);
        let cells = mi_cols.saturating_mul(mi_rows);
        if cells == 0 || mi_cols > 2048 || mi_rows > 2048 {
            return Err(CodecError::Limit { what: "frame size" });
        }
        let buf_w = mi_cols * 4;
        let buf_h = mi_rows * 4;
        let chroma_w = buf_w.div_ceil(2);
        let chroma_h = buf_h.div_ceil(2);
        Ok(Self {
            symbols: SymbolEncoder::new(),
            tables: load_tables(q),
            y: vec![0; buf_w * buf_h],
            u: vec![128; chroma_w * chroma_h],
            v: vec![128; chroma_w * chroma_h],
            src_y: input.picture.y.samples().to_vec(),
            src_u: input.picture.u.samples().to_vec(),
            src_v: input.picture.v.samples().to_vec(),
            visible_w,
            visible_h,
            buf_w,
            buf_h,
            chroma_w,
            chroma_h,
            mi_cols,
            mi_rows,
            modes: vec![DC_PRED; cells],
            skips: vec![0; cells],
            sizes: vec![0; cells],
            above_level: [vec![0; mi_cols], vec![0; mi_cols], vec![0; mi_cols]],
            left_level: [vec![0; mi_rows], vec![0; mi_rows], vec![0; mi_rows]],
            above_dc: [vec![0; mi_cols], vec![0; mi_cols], vec![0; mi_cols]],
            left_dc: [vec![0; mi_rows], vec![0; mi_rows], vec![0; mi_rows]],
            dc_q: quantizer(q, true),
            ac_q: quantizer(q, false),
        })
    }

    fn encode(mut self) -> Result<Vec<u8>, CodecError> {
        self.walk()?;
        self.symbols.finish()
    }

    fn walk(&mut self) -> Result<(), CodecError> {
        let mut row = 0;
        while row < self.mi_rows {
            self.clear_left();
            let mut col = 0;
            while col < self.mi_cols {
                self.encode_partition(row, col, ROOT)?;
                col += SB_MI;
            }
            row += SB_MI;
        }
        Ok(())
    }

    fn clear_left(&mut self) {
        for plane in 0..3 {
            self.left_level[plane].fill(0);
            self.left_dc[plane].fill(0);
        }
    }

    fn encode_partition(&mut self, row: usize, col: usize, bsize: usize) -> Result<(), CodecError> {
        if row >= self.mi_rows || col >= self.mi_cols {
            return Ok(());
        }
        let half = NUM_4X4_WIDE[bsize] >> 1;
        let has_rows = row + half < self.mi_rows;
        let has_cols = col + half < self.mi_cols;
        let partition = self.write_partition(row, col, bsize, has_rows, has_cols)?;
        let sub = crate::dims::partition_subsize(partition, bsize)
            .ok_or(CodecError::Unsupported { what: "partition" })?;
        if partition == PARTITION_NONE {
            self.code_block(row, col, sub)?;
        } else if partition == PARTITION_SPLIT {
            if half == 0 {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "partition",
                });
            }
            self.encode_partition(row, col, sub)?;
            self.encode_partition(row, col + half, sub)?;
            self.encode_partition(row + half, col, sub)?;
            self.encode_partition(row + half, col + half, sub)?;
        }
        Ok(())
    }

    fn write_partition(
        &mut self,
        row: usize,
        col: usize,
        bsize: usize,
        has_rows: bool,
        has_cols: bool,
    ) -> Result<u32, CodecError> {
        if bsize < BLOCK_8X8 {
            return Ok(PARTITION_NONE);
        }
        let bsl = MI_WIDTH_LOG2[bsize];
        let above =
            row > 0 && MI_WIDTH_LOG2[usize::from(self.sizes[(row - 1) * self.mi_cols + col])] < bsl;
        let left =
            col > 0 && MI_HEIGHT_LOG2[usize::from(self.sizes[row * self.mi_cols + col - 1])] < bsl;
        let ctx = usize::from(left) * 2 + usize::from(above);
        let shape = partition_shape(bsl);
        let table = partition_cdf(bsl.saturating_sub(1).min(4));
        if has_rows && has_cols {
            let symbol = usize::from(bsize != BLOCK_8X8) * 3;
            self.write_symbol(table, shape, &[ctx], symbol)?;
            return Ok(if symbol == 0 {
                PARTITION_NONE
            } else {
                PARTITION_SPLIT
            });
        }
        if !has_rows && !has_cols {
            return Ok(PARTITION_SPLIT);
        }
        let (start, len) = row_range(table.len(), shape, &[ctx])?;
        let built = edge_partition_cdf(&table[start..start + len], bsize, has_cols);
        self.symbols.symbol(1, &built)?;
        Ok(PARTITION_SPLIT)
    }

    fn code_block(&mut self, row: usize, col: usize, bsize: usize) -> Result<(), CodecError> {
        let bw4 = NUM_4X4_WIDE[bsize];
        let bh4 = NUM_4X4_HIGH[bsize];
        let has_chroma = self.has_chroma(row, col, bw4, bh4);
        let skip_ctx = self.skip_ctx(row, col);
        let mode_index = self.mode_index(row, col);
        let jobs = self.analyze(row, col, bsize, has_chroma)?;
        let skip = jobs
            .iter()
            .all(|job| job.levels.iter().all(|level| *level == 0));
        self.write_symbol(&DEFAULT_SKIP_CDF, &[3, 3], &[skip_ctx], usize::from(skip))?;
        self.write_symbol(&DEFAULT_INTRA_FRAME_Y_MODE_CDF, &[5, 5, 14], &mode_index, 0)?;
        if has_chroma {
            self.write_uv_mode(bsize)?;
        }
        self.fill(row, col, bw4, bh4, skip, bsize);
        if skip {
            self.reset_context(row, col, bw4, bh4, has_chroma);
            return Ok(());
        }
        for job in &jobs {
            self.write_job(bsize, job)?;
        }
        Ok(())
    }

    fn write_uv_mode(&mut self, bsize: usize) -> Result<(), CodecError> {
        if block_width(bsize).max(block_height(bsize)) > 32 {
            self.write_symbol(&DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF, &[13, 14], &[0], 0)
        } else {
            self.write_symbol(&DEFAULT_UV_MODE_CFL_ALLOWED_CDF, &[13, 15], &[0], 0)
        }
    }

    fn analyze(
        &mut self,
        row: usize,
        col: usize,
        bsize: usize,
        has_chroma: bool,
    ) -> Result<Vec<Job>, CodecError> {
        let mut jobs = Vec::new();
        let count = if has_chroma { 3 } else { 1 };
        for plane in 0..count {
            jobs.push(self.predict_plane(plane, row, col, bsize)?);
        }
        Ok(jobs)
    }

    fn predict_plane(
        &mut self,
        plane: usize,
        row: usize,
        col: usize,
        bsize: usize,
    ) -> Result<Job, CodecError> {
        let tx = if plane == 0 {
            MAX_TX_SIZE_RECT[bsize]
        } else {
            chroma_tx(bsize)
        };
        let n = TX_WIDTH[tx];
        let sub = usize::from(plane > 0);
        let start_x = (col >> sub) * 4;
        let start_y = (row >> sub) * 4;
        let window = self.source_window(plane, start_x, start_y, n);
        let stride = if plane == 0 {
            self.buf_w
        } else {
            self.chroma_w
        };
        let rows = if plane == 0 {
            self.buf_h
        } else {
            self.chroma_h
        };
        let empty = Job {
            plane,
            x: start_x,
            y: start_y,
            tx,
            levels: vec![0; n * n],
        };
        if start_x >= stride || start_y >= rows || start_x + n > stride || start_y + n > rows {
            return Ok(empty);
        }
        let buf = match plane {
            0 => &mut self.y,
            1 => &mut self.u,
            _ => &mut self.v,
        };
        predict_dc(
            buf,
            stride,
            rows,
            DcAt {
                x: start_x,
                y: start_y,
                n,
                col,
                row,
            },
        )?;
        let mut residual = vec![0i32; n * n];
        for y in 0..n {
            for x in 0..n {
                let at = (start_y + y) * stride + start_x + x;
                let predicted = i32::from(buf[at]);
                let source = window[y * n + x].map_or(predicted, i32::from);
                residual[y * n + x] = source - predicted;
            }
        }
        let dc_q = self.dc_q;
        let ac_q = self.ac_q;
        Ok(Job {
            plane,
            x: start_x,
            y: start_y,
            tx,
            levels: choose_levels(&residual, tx, dc_q, ac_q),
        })
    }

    fn source_window(&self, plane: usize, x0: usize, y0: usize, n: usize) -> Vec<Option<u8>> {
        let (src, width, height) = match plane {
            0 => (self.src_y.as_slice(), self.visible_w, self.visible_h),
            1 => (
                self.src_u.as_slice(),
                self.visible_w.div_ceil(2),
                self.visible_h.div_ceil(2),
            ),
            _ => (
                self.src_v.as_slice(),
                self.visible_w.div_ceil(2),
                self.visible_h.div_ceil(2),
            ),
        };
        let mut out = vec![None; n * n];
        for y in 0..n {
            for x in 0..n {
                let sx = x0 + x;
                let sy = y0 + y;
                if sx < width && sy < height {
                    out[y * n + x] = Some(src[sy * width + sx]);
                }
            }
        }
        out
    }

    fn write_job(&mut self, bsize: usize, job: &Job) -> Result<(), CodecError> {
        let geom = geom_for(job, bsize, self.mi_cols, self.mi_rows);
        let tx_ctx = (TX_SIZE_SQR[job.tx] + TX_SIZE_SQR_UP[job.tx] + 1) >> 1;
        let coeff_ctx = if job.plane == 0 {
            luma_coeff_ctx(&self.above_level[0], &self.left_level[0], &geom)
        } else {
            chroma_coeff_ctx(
                &self.above_level[job.plane],
                &self.above_dc[job.plane],
                &self.left_level[job.plane],
                &self.left_dc[job.plane],
                &geom,
            )
        };
        let nonzero = job.levels.iter().any(|level| *level != 0);
        let row = cdf_row(&self.tables.txb_skip, &[5, 13, 3], &[tx_ctx, coeff_ctx])?;
        self.symbols.symbol(usize::from(!nonzero), &row)?;
        if !nonzero {
            self.put_context(job.plane, &geom, 0, 0);
            return Ok(());
        }
        self.write_tx_type(job.plane, job.tx)?;
        let scan = scan_for(job.tx, DCT_DCT);
        let eob = eob_of(&job.levels, scan);
        let ptype = usize::from(job.plane > 0);
        self.write_eob(eob, job.tx, tx_ctx, ptype)?;
        self.write_abs(&job.levels, scan, eob, job.tx, tx_ctx, ptype)?;
        self.write_signs(&job.levels, scan, eob, job.plane, &geom)?;
        let n = TX_WIDTH[job.tx];
        let residual = inverse_levels(&job.levels, n, job.tx, self.dc_q, self.ac_q);
        self.add(job.plane, job.x, job.y, &residual);
        Ok(())
    }

    fn write_tx_type(&mut self, plane: usize, tx: usize) -> Result<(), CodecError> {
        if plane != 0 || TX_SIZE_SQR_UP[tx] >= TX_32X32 {
            return Ok(());
        }
        let sqr = TX_SIZE_SQR[tx];
        if TX_SIZE_SQR[tx] == TX_16X16 {
            self.write_symbol(&DEFAULT_INTRA_TX_TYPE_SET2_CDF, &[3, 13, 6], &[sqr, 0], 1)
        } else {
            self.write_symbol(&DEFAULT_INTRA_TX_TYPE_SET1_CDF, &[2, 13, 8], &[sqr, 0], 1)
        }
    }

    fn write_eob(
        &mut self,
        eob: usize,
        tx: usize,
        tx_ctx: usize,
        ptype: usize,
    ) -> Result<(), CodecError> {
        let eob_pt = if eob <= 2 {
            eob as u32
        } else {
            floor_log2((eob - 1) as u32) + 2
        };
        let (which, len) = eob_shape(tx);
        let symbol = usize::try_from(eob_pt.saturating_sub(1)).unwrap_or(0);
        if which <= 4 {
            let row = cdf_row(&self.tables.eob[which], &[2, 2, len], &[ptype, 0])?;
            self.symbols.symbol(symbol, &row)?;
        } else {
            let row = cdf_row(&self.tables.eob[which], &[2, len], &[ptype])?;
            self.symbols.symbol(symbol, &row)?;
        }
        let shift = eob_pt as i32 - 3;
        if shift < 0 {
            return Ok(());
        }
        let low = (1usize << (eob_pt as usize - 2)) + 1;
        let offset = eob.saturating_sub(low);
        for bit in (0..=shift).rev() {
            let on = (offset >> u32::try_from(bit).unwrap_or(0)) & 1 == 1;
            if bit == shift {
                let index = usize::try_from(eob_pt).unwrap_or(3).saturating_sub(3);
                let row = cdf_row(
                    &self.tables.eob_extra,
                    &[5, 2, 9, 3],
                    &[tx_ctx, ptype, index],
                )?;
                self.symbols.symbol(usize::from(on), &row)?;
            } else {
                self.symbols.bool(on)?;
            }
        }
        Ok(())
    }

    fn write_abs(
        &mut self,
        levels: &[i32],
        scan: &[u16],
        eob: usize,
        tx: usize,
        tx_ctx: usize,
        ptype: usize,
    ) -> Result<(), CodecError> {
        let mut quant = vec![0i32; levels.len()];
        let mut c = eob;
        while c > 0 {
            c -= 1;
            let pos = usize::from(scan[c]);
            let abs_level = levels.get(pos).copied().unwrap_or(0).unsigned_abs();
            let eob_coeff = c + 1 == eob;
            let ctx = coeff_base_ctx(tx, DCT_DCT, pos, c, eob_coeff, &quant)?;
            if eob_coeff {
                let base = abs_level.clamp(1, 3);
                let row = cdf_row(&self.tables.coeff_eob, &[5, 2, 4, 4], &[tx_ctx, ptype, ctx])?;
                self.symbols
                    .symbol(usize::try_from(base - 1).unwrap_or(0), &row)?;
            } else {
                let row = cdf_row(
                    &self.tables.coeff_base,
                    &[5, 2, 42, 5],
                    &[tx_ctx, ptype, ctx],
                )?;
                self.symbols
                    .symbol(usize::try_from(abs_level.min(3)).unwrap_or(0), &row)?;
            }
            if abs_level > 2 {
                self.write_br(tx, pos, &quant, tx_ctx, ptype, abs_level - 3)?;
            }
            quant[pos] = if abs_level > 14 {
                15
            } else {
                i32::try_from(abs_level).unwrap_or(0)
            };
        }
        Ok(())
    }

    fn write_br(
        &mut self,
        tx: usize,
        pos: usize,
        quant: &[i32],
        tx_ctx: usize,
        ptype: usize,
        extra: u32,
    ) -> Result<(), CodecError> {
        let mut left = extra;
        for _ in 0..4 {
            let br = if left >= 3 { 3 } else { left };
            let ctx = br_ctx(tx, DCT_DCT, pos, quant);
            let row = cdf_row(
                &self.tables.coeff_br,
                &[5, 2, 21, 5],
                &[tx_ctx.min(3), ptype, ctx],
            )?;
            self.symbols
                .symbol(usize::try_from(br).unwrap_or(0), &row)?;
            if br < 3 {
                break;
            }
            left -= 3;
        }
        Ok(())
    }

    fn write_signs(
        &mut self,
        levels: &[i32],
        scan: &[u16],
        eob: usize,
        plane: usize,
        geom: &Neighbor,
    ) -> Result<(), CodecError> {
        let ptype = usize::from(plane > 0);
        let mut cul = 0i32;
        let mut dc_cat = 0u8;
        for (c, scanned) in scan.iter().enumerate().take(eob) {
            let pos = usize::from(*scanned);
            let level = levels.get(pos).copied().unwrap_or(0);
            if level == 0 {
                continue;
            }
            let negative = level < 0;
            if c == 0 {
                let ctx = dc_context(&self.above_dc[plane], &self.left_dc[plane], geom);
                let row = cdf_row(&self.tables.dc_sign, &[2, 3, 3], &[ptype, ctx])?;
                self.symbols.symbol(usize::from(negative), &row)?;
            } else {
                self.symbols.bool(negative)?;
            }
            let abs_level = level.unsigned_abs();
            if abs_level > 14 {
                write_golomb(&mut self.symbols, abs_level)?;
            }
            if pos == 0 {
                dc_cat = if negative { 1 } else { 2 };
            }
            cul = cul.saturating_add(i32::try_from(abs_level).unwrap_or(i32::MAX));
        }
        let stored = u8::try_from(cul.min(63)).unwrap_or(63);
        self.put_context(plane, geom, stored, dc_cat);
        Ok(())
    }

    fn write_symbol(
        &mut self,
        table: &[u16],
        shape: &[usize],
        index: &[usize],
        symbol: usize,
    ) -> Result<(), CodecError> {
        let row = cdf_row(table, shape, index)?;
        self.symbols.symbol(symbol, &row)
    }

    fn skip_ctx(&self, row: usize, col: usize) -> usize {
        let mut ctx = 0;
        if row > 0 {
            ctx += usize::from(self.skips[(row - 1) * self.mi_cols + col]);
        }
        if col > 0 {
            ctx += usize::from(self.skips[row * self.mi_cols + col - 1]);
        }
        ctx
    }

    fn mode_index(&self, row: usize, col: usize) -> [usize; 2] {
        let above = if row > 0 {
            self.modes[(row - 1) * self.mi_cols + col]
        } else {
            DC_PRED
        };
        let left = if col > 0 {
            self.modes[row * self.mi_cols + col - 1]
        } else {
            DC_PRED
        };
        [MODE_CTX[usize::from(above)], MODE_CTX[usize::from(left)]]
    }

    fn has_chroma(&self, row: usize, col: usize, bw4: usize, bh4: usize) -> bool {
        if bh4 == 1 && row.is_multiple_of(2) {
            return false;
        }
        if bw4 == 1 && col.is_multiple_of(2) {
            return false;
        }
        true
    }

    fn fill(&mut self, row: usize, col: usize, bw4: usize, bh4: usize, skip: bool, bsize: usize) {
        for y in 0..bh4 {
            for x in 0..bw4 {
                if row + y >= self.mi_rows || col + x >= self.mi_cols {
                    continue;
                }
                let at = (row + y) * self.mi_cols + col + x;
                self.modes[at] = DC_PRED;
                self.skips[at] = u8::from(skip);
                self.sizes[at] = u8::try_from(bsize).unwrap_or(0);
            }
        }
    }

    fn reset_context(&mut self, row: usize, col: usize, bw4: usize, bh4: usize, has_chroma: bool) {
        let planes = if has_chroma { 3 } else { 1 };
        for plane in 0..planes {
            let sub = usize::from(plane > 0);
            self.clear_range(
                plane,
                col >> sub,
                row >> sub,
                (col + bw4) >> sub,
                (row + bh4) >> sub,
            );
        }
    }

    fn clear_range(&mut self, plane: usize, x0: usize, y0: usize, x1: usize, y1: usize) {
        for x in x0..x1.min(self.above_level[plane].len()) {
            self.above_level[plane][x] = 0;
            self.above_dc[plane][x] = 0;
        }
        for y in y0..y1.min(self.left_level[plane].len()) {
            self.left_level[plane][y] = 0;
            self.left_dc[plane][y] = 0;
        }
    }

    fn put_context(&mut self, plane: usize, geom: &Neighbor, level: u8, dc: u8) {
        for i in 0..geom.w4 {
            let x = geom.x4 + i;
            if x < self.above_level[plane].len() {
                self.above_level[plane][x] = level;
                self.above_dc[plane][x] = dc;
            }
        }
        for i in 0..geom.h4 {
            let y = geom.y4 + i;
            if y < self.left_level[plane].len() {
                self.left_level[plane][y] = level;
                self.left_dc[plane][y] = dc;
            }
        }
    }

    fn add(&mut self, plane: usize, x: usize, y: usize, residual: &[i32]) {
        let stride = if plane == 0 {
            self.buf_w
        } else {
            self.chroma_w
        };
        let buf = match plane {
            0 => &mut self.y,
            1 => &mut self.u,
            _ => &mut self.v,
        };
        let n = TX_WIDTH[if residual.len() == 16 { TX_4X4 } else { TX_8X8 }];
        for row in 0..n {
            for col in 0..n {
                let at = (y + row) * stride + x + col;
                if at >= buf.len() || row * n + col >= residual.len() {
                    continue;
                }
                let sum = i32::from(buf[at]) + residual[row * n + col];
                buf[at] = u8::try_from(sum.clamp(0, 255)).unwrap_or(0);
            }
        }
    }
}

fn geom_for(job: &Job, bsize: usize, mi_cols: usize, mi_rows: usize) -> Neighbor {
    let sub = usize::from(job.plane > 0);
    let plane_size = if job.plane == 0 {
        bsize
    } else {
        subsampled_size(bsize, true, true)
    };
    Neighbor {
        x4: job.x >> 2,
        y4: job.y >> 2,
        w4: TX_WIDTH[job.tx] >> 2,
        h4: TX_HEIGHT[job.tx] >> 2,
        max_x: mi_cols >> sub,
        max_y: mi_rows >> sub,
        bw: block_width(plane_size),
        bh: block_height(plane_size),
        tw: TX_WIDTH[job.tx],
        th: TX_HEIGHT[job.tx],
    }
}

fn chroma_tx(bsize: usize) -> usize {
    MAX_TX_SIZE_RECT[subsampled_size(bsize, true, true)]
}

struct DcAt {
    x: usize,
    y: usize,
    n: usize,
    col: usize,
    row: usize,
}

fn predict_dc(plane: &mut [u8], stride: usize, rows: usize, at: DcAt) -> Result<(), CodecError> {
    let block = IntraPred {
        x: at.x,
        y: at.y,
        w: at.n,
        h: at.n,
        have_left: at.col > 0 && at.x > 0,
        have_above: at.row > 0 && at.y > 0,
        have_above_right: false,
        have_below_left: false,
        mode: DC_PRED,
        angle_delta: 0,
        edge_filter: true,
        filter_type: false,
        filter: None,
        max_x: stride.saturating_sub(1),
        max_y: rows.saturating_sub(1),
    };
    let mut scratch = crate::predict::PredictScratch::new();
    predict_intra(plane, stride, &block, &mut scratch)
}

fn dc_context(above: &[u8], left: &[u8], geom: &Neighbor) -> usize {
    let mut sign = 0i32;
    for k in 0..geom.w4 {
        if geom.x4 + k < geom.max_x {
            match above.get(geom.x4 + k).copied().unwrap_or(0) {
                1 => sign -= 1,
                2 => sign += 1,
                _ => {}
            }
        }
    }
    for k in 0..geom.h4 {
        if geom.y4 + k < geom.max_y {
            match left.get(geom.y4 + k).copied().unwrap_or(0) {
                1 => sign -= 1,
                2 => sign += 1,
                _ => {}
            }
        }
    }
    if sign < 0 {
        1
    } else if sign > 0 {
        2
    } else {
        0
    }
}

fn eob_of(levels: &[i32], scan: &[u16]) -> usize {
    let mut eob = 0;
    for (i, pos) in scan.iter().enumerate() {
        if levels.get(usize::from(*pos)).copied().unwrap_or(0) != 0 {
            eob = i + 1;
        }
    }
    eob
}

fn eob_shape(tx: usize) -> (usize, usize) {
    let multi = TX_WIDTH_LOG2[tx].min(5) + TX_HEIGHT_LOG2[tx].min(5) - 4;
    match multi {
        0 => (0, 6),
        1 => (1, 7),
        2 => (2, 8),
        3 => (3, 9),
        4 => (4, 10),
        5 => (5, 11),
        _ => (6, 12),
    }
}

fn write_golomb(symbols: &mut SymbolEncoder, abs_level: u32) -> Result<(), CodecError> {
    let x = abs_level.saturating_sub(14).clamp(1, (1 << 20) - 1);
    let length = floor_log2(x) + 1;
    for _ in 1..length {
        symbols.bool(false)?;
    }
    symbols.bool(true)?;
    if length > 1 {
        let bits = length - 1;
        symbols.literal(bits, x & ((1 << bits) - 1))?;
    }
    Ok(())
}

fn cdf_row(table: &[u16], shape: &[usize], index: &[usize]) -> Result<Vec<u16>, CodecError> {
    let (start, len) = row_range(table.len(), shape, index)?;
    if len < 3 || start + len > table.len() {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "cdf",
        });
    }
    Ok(table[start..start + len].to_vec())
}

fn load_tables(q: u8) -> CoeffTables {
    let ctx = q_context(q);
    CoeffTables {
        txb_skip: slice_q(&DEFAULT_TXB_SKIP_CDF, 4, ctx),
        eob: [
            slice_q(&DEFAULT_EOB_PT_16_CDF, 4, ctx),
            slice_q(&DEFAULT_EOB_PT_32_CDF, 4, ctx),
            slice_q(&DEFAULT_EOB_PT_64_CDF, 4, ctx),
            slice_q(&DEFAULT_EOB_PT_128_CDF, 4, ctx),
            slice_q(&DEFAULT_EOB_PT_256_CDF, 4, ctx),
            slice_q(&DEFAULT_EOB_PT_512_CDF, 4, ctx),
            slice_q(&DEFAULT_EOB_PT_1024_CDF, 4, ctx),
        ],
        eob_extra: slice_q(&DEFAULT_EOB_EXTRA_CDF, 4, ctx),
        dc_sign: slice_q(&DEFAULT_DC_SIGN_CDF, 4, ctx),
        coeff_eob: slice_q(&DEFAULT_COEFF_BASE_EOB_CDF, 4, ctx),
        coeff_base: slice_q(&DEFAULT_COEFF_BASE_CDF, 4, ctx),
        coeff_br: slice_q(&DEFAULT_COEFF_BR_CDF, 4, ctx),
    }
}

fn partition_cdf(slot: usize) -> &'static [u16] {
    match slot {
        0 => &DEFAULT_PARTITION_W8_CDF,
        1 => &DEFAULT_PARTITION_W16_CDF,
        2 => &DEFAULT_PARTITION_W32_CDF,
        3 => &DEFAULT_PARTITION_W64_CDF,
        _ => &DEFAULT_PARTITION_W128_CDF,
    }
}

struct Basis {
    n: usize,
    scale: i32,
    vectors: Vec<i32>,
    norms: Vec<i64>,
}

fn basis_for(tx: usize) -> &'static Basis {
    static SMALL: OnceLock<Basis> = OnceLock::new();
    static LARGE: OnceLock<Basis> = OnceLock::new();
    if tx == TX_4X4 {
        SMALL.get_or_init(|| build_basis(4, TX_4X4))
    } else {
        LARGE.get_or_init(|| build_basis(8, TX_8X8))
    }
}

fn build_basis(n: usize, tx: usize) -> Basis {
    let scale = 256i32;
    let count = n * n;
    let mut vectors = vec![0i32; count * count];
    let mut norms = vec![0i64; count];
    for k in 0..count {
        let mut coeff = vec![0i32; count];
        coeff[k] = scale;
        let response = inverse_2d(&coeff, n, n, DCT_DCT, TRANSFORM_ROW_SHIFT[tx]);
        let mut norm = 0i64;
        for (i, sample) in response.iter().enumerate().take(count) {
            vectors[k * count + i] = *sample;
            norm += i64::from(*sample) * i64::from(*sample);
        }
        norms[k] = norm;
    }
    Basis {
        n,
        scale,
        vectors,
        norms,
    }
}

fn choose_levels(residual: &[i32], tx: usize, dc_q: i32, ac_q: i32) -> Vec<i32> {
    let n = TX_WIDTH[tx];
    let basis = basis_for(tx);
    if residual.len() != n * n || basis.n != n {
        return vec![0; residual.len()];
    }
    let projected = project(residual, basis, dc_q, ac_q);
    let mut levels = projected.clone();
    for level in levels.iter_mut().skip(1) {
        *level = 0;
    }
    refine_dc(residual, &mut levels, n, tx, dc_q, ac_q);
    let dc_error = sse(residual, &inverse_levels(&levels, n, tx, dc_q, ac_q));
    let zero_error = sse(residual, &vec![0; residual.len()]);
    if dc_error >= zero_error {
        levels[0] = 0;
    }
    let mut with_ac = projected;
    with_ac[0] = levels[0];
    let ac_error = sse(residual, &inverse_levels(&with_ac, n, tx, dc_q, ac_q));
    let kept = if levels[0] == 0 {
        zero_error
    } else {
        dc_error.min(zero_error)
    };
    if ac_error < kept {
        with_ac
    } else if levels[0] == 0 {
        vec![0; residual.len()]
    } else {
        levels
    }
}

fn project(residual: &[i32], basis: &Basis, dc_q: i32, ac_q: i32) -> Vec<i32> {
    let count = basis.n * basis.n;
    let mut levels = vec![0i32; count];
    for (k, level_slot) in levels.iter_mut().enumerate() {
        if basis.norms[k] == 0 {
            continue;
        }
        let mut dot = 0i64;
        let row = &basis.vectors[k * count..(k + 1) * count];
        for (sample, value) in residual.iter().zip(row.iter()) {
            dot += i64::from(*sample) * i64::from(*value);
        }
        let coeff = div_round(dot * i64::from(basis.scale), basis.norms[k]);
        let q = if k == 0 { dc_q } else { ac_q };
        if q == 0 {
            continue;
        }
        let mut level = div_round(coeff, i64::from(q));
        if k != 0 && level.abs() <= 1 {
            level = 0;
        }
        *level_slot = i32::try_from(level.clamp(-((1 << 20) + 13), (1 << 20) + 13)).unwrap_or(0);
    }
    levels
}

fn refine_dc(residual: &[i32], levels: &mut [i32], n: usize, tx: usize, dc_q: i32, ac_q: i32) {
    let guess = levels.first().copied().unwrap_or(0);
    let mut best = guess;
    let mut best_error = i64::MAX;
    for delta in -8..=8 {
        levels[0] = guess + delta;
        let error = sse(residual, &inverse_levels(levels, n, tx, dc_q, ac_q));
        if error < best_error {
            best_error = error;
            best = levels[0];
        }
    }
    levels[0] = best;
}

fn inverse_levels(levels: &[i32], n: usize, tx: usize, dc_q: i32, ac_q: i32) -> Vec<i32> {
    let mut coeff = vec![0i32; n * n];
    for row in 0..n {
        for col in 0..n {
            let q = if row == 0 && col == 0 { dc_q } else { ac_q };
            let Some(level) = levels.get(row * n + col) else {
                continue;
            };
            let dq = i64::from(*level) * i64::from(q);
            let sign = if dq < 0 { -1i64 } else { 1 };
            let value = sign * (dq.abs() & 0x00ff_ffff);
            coeff[row * n + col] = i32::try_from(value.clamp(-32768, 32767)).unwrap_or(0);
        }
    }
    inverse_2d(&coeff, n, n, DCT_DCT, TRANSFORM_ROW_SHIFT[tx])
}

fn sse(left: &[i32], right: &[i32]) -> i64 {
    left.iter().zip(right.iter()).fold(0i64, |total, (a, b)| {
        let delta = i64::from(*a) - i64::from(*b);
        total.saturating_add(delta * delta)
    })
}

fn div_round(num: i64, den: i64) -> i64 {
    if den == 0 {
        return 0;
    }
    if num >= 0 {
        (num + den / 2) / den
    } else {
        -((-num + den / 2) / den)
    }
}

fn quantizer(q: u8, dc: bool) -> i32 {
    let index = usize::from(q);
    i32::from(if dc {
        DC_QLOOKUP[index]
    } else {
        AC_QLOOKUP[index]
    })
}

fn q_context(q: u8) -> usize {
    match q {
        0..=20 => 0,
        21..=60 => 1,
        61..=120 => 2,
        _ => 3,
    }
}

fn sequence_header(width: u32, height: u32) -> Result<Vec<u8>, CodecError> {
    let mut bits = BitWriter::new();
    bits.write(0, 3)?;
    bits.write_bool(false)?;
    bits.write_bool(false)?;
    bits.write_bool(false)?;
    bits.write_bool(false)?;
    bits.write(0, 5)?;
    bits.write(0, 12)?;
    bits.write(0, 5)?;
    let width_bits = size_bits(width);
    let height_bits = size_bits(height);
    bits.write(width_bits - 1, 4)?;
    bits.write(height_bits - 1, 4)?;
    bits.write(width - 1, width_bits)?;
    bits.write(height - 1, height_bits)?;
    bits.write_bool(false)?;
    bits.write_bool(false)?;
    bits.write_bool(false)?;
    bits.write_bool(true)?;
    bits.write_bool(false)?;
    bits.write_bool(false)?;
    bits.write_bool(false)?;
    bits.write_bool(false)?;
    bits.write_bool(false)?;
    bits.write_bool(false)?;
    bits.write(0, 1)?;
    bits.write_bool(false)?;
    bits.write_bool(false)?;
    bits.write_bool(false)?;
    bits.write_bool(false)?;
    bits.write_bool(false)?;
    bits.write_bool(false)?;
    bits.write_bool(false)?;
    bits.write(0, 2)?;
    bits.write_bool(false)?;
    bits.write_bool(false)?;
    bits.trailing_bits()?;
    Ok(bits.finish())
}

fn key_header(width: u32, height: u32, q: u8) -> Result<Vec<u8>, CodecError> {
    let mut bits = BitWriter::new();
    bits.write_bool(false)?;
    bits.write(0, 2)?;
    bits.write_bool(true)?;
    bits.write_bool(true)?;
    bits.write_bool(false)?;
    bits.write_bool(false)?;
    write_tiles(&mut bits, width, height)?;
    bits.write(u32::from(q), 8)?;
    bits.write_bool(false)?;
    bits.write_bool(false)?;
    bits.write_bool(false)?;
    bits.write_bool(false)?;
    bits.write_bool(false)?;
    if q > 0 {
        bits.write_bool(false)?;
    }
    bits.write(0, 6)?;
    bits.write(0, 6)?;
    bits.write(0, 3)?;
    bits.write_bool(false)?;
    bits.write_bool(false)?;
    bits.write_bool(false)?;
    Ok(bits.finish())
}

fn write_tiles(bits: &mut BitWriter, width: u32, height: u32) -> Result<(), CodecError> {
    bits.write(1, 1)?;
    let sb_cols = width.div_ceil(64);
    let sb_rows = height.div_ceil(64);
    let min_cols = tile_log2(64, sb_cols);
    let max_cols = tile_log2(1, sb_cols.min(64));
    if min_cols < max_cols {
        bits.write_bool(false)?;
    }
    let min_tiles = tile_log2(2304, sb_rows.saturating_mul(sb_cols)).max(min_cols);
    let max_rows = tile_log2(1, sb_rows.min(64));
    if min_tiles.saturating_sub(min_cols) < max_rows {
        bits.write_bool(false)?;
    }
    Ok(())
}

fn tile_log2(block: u32, target: u32) -> u32 {
    let mut k = 0u32;
    let mut size = block.max(1);
    while size < target && k < 16 {
        size <<= 1;
        k += 1;
    }
    k
}

fn size_bits(value: u32) -> u32 {
    let minus = value.saturating_sub(1);
    if minus == 0 { 1 } else { floor_log2(minus) + 1 }
}

fn obu(kind: u32, payload: &[u8]) -> Result<Vec<u8>, CodecError> {
    let len = u32::try_from(payload.len()).map_err(|_| CodecError::Limit { what: "obu" })?;
    let mut out = vec![u8::try_from((kind << 3) | 0x02).unwrap_or(0)];
    write_leb128(&mut out, len);
    out.extend_from_slice(payload);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::Av1Encoder;
    use crate::decode::Av1Decoder;
    use crate::header::parse_unit;
    use crate::tile::decode_tile;
    use cutlass_decode::{CodecId, CodecParams, Decoder};

    use crate::yuv::{AsView, Plane, VideoInput, Yuv420};

    fn picture(width: u32, height: u32, fill: impl Fn(u32, u32) -> u8) -> Yuv420 {
        let mut y = vec![0u8; (width * height) as usize];
        for row in 0..height {
            for col in 0..width {
                y[(row * width + col) as usize] = fill(col, row);
            }
        }
        let cw = width.div_ceil(2);
        let ch = height.div_ceil(2);
        let chroma = vec![128u8; (cw * ch) as usize];
        Yuv420::new(
            Plane::new(width, height, y).expect("y"),
            Plane::new(cw, ch, chroma.clone()).expect("u"),
            Plane::new(cw, ch, chroma).expect("v"),
        )
        .expect("yuv")
    }

    fn encode(width: u32, height: u32, q: u8, fill: impl Fn(u32, u32) -> u8) -> Vec<u8> {
        let params = CodecParams::video(CodecId::Av1, width, height);
        let mut encoder = Av1Encoder::open(&params).expect("open");
        encoder.set_quality(q);
        let frame = VideoInput {
            picture: picture(width, height, fill),
            pts: 0,
            duration: 1,
            timescale: 30,
        };
        encoder.push_picture(&frame).expect("encode").remove(0).data
    }

    #[test]
    fn a_flat_picture_round_trips() {
        let bytes = encode(64, 64, 40, |_, _| 128);
        let (sequence, frame) = parse_unit(&bytes, None).expect("parse");
        assert_eq!(frame.width, 64);
        assert!(frame.intra);
        assert_eq!(frame.base_q, 40);
        let picture = decode_tile(&sequence, &frame).expect("decode");
        assert!(
            picture.y.iter().all(|sample| *sample == 128),
            "luma {}",
            picture.y[0]
        );
        assert!(picture.u.iter().all(|sample| *sample == 128));
        assert!(picture.v.iter().all(|sample| *sample == 128));
    }

    #[test]
    fn a_step_at_quality_40_keeps_luma_detail() {
        let bytes = encode(64, 64, 40, |col, _| if col < 32 { 40 } else { 180 });
        let (sequence, frame) = parse_unit(&bytes, None).expect("parse");
        let picture = decode_tile(&sequence, &frame).expect("decode");
        let mut sse = 0u64;
        for row in 0..64 {
            for col in 0..64 {
                let expected = if col < 32 { 40 } else { 180 };
                let delta = i32::from(picture.y[row * 64 + col]) - expected;
                sse += u64::from(delta.unsigned_abs()).pow(2);
            }
        }
        let mse = sse as f64 / 4096.0;
        let psnr = if mse == 0.0 {
            100.0
        } else {
            10.0 * (255.0 * 255.0 / mse).log10()
        };
        assert!(psnr >= 28.0, "luma psnr {psnr}");
    }

    #[test]
    fn thirty_frames_land_near_the_target_rate() {
        let mut params = CodecParams::video(CodecId::Av1, 320, 180);
        params.bitrate = 500_000;
        let mut encoder = Av1Encoder::open(&params).expect("open");
        let mut decoder = Av1Decoder::open(&params).expect("decoder");
        let mut total = 0usize;
        for frame in 0..30u32 {
            let picture = picture(320, 180, |x, y| {
                u8::try_from(
                    (x.wrapping_mul(3)
                        .wrapping_add(y)
                        .wrapping_add(frame.wrapping_mul(17)))
                        & 255,
                )
                .unwrap_or(0)
            });
            let packet = encoder
                .push_picture(&VideoInput {
                    picture,
                    pts: i64::from(frame),
                    duration: 1,
                    timescale: 30,
                })
                .expect("encode")
                .remove(0);
            total += packet.data.len();
            decoder.push(&packet).expect("decode");
        }
        let rate = total as u64 * 8;
        assert!((300_000..=700_000).contains(&rate), "rate {rate}");
    }

    #[test]
    fn one_sd_frame_decodes() {
        let bytes = encode(720, 480, 40, |x, y| {
            u8::try_from((x.wrapping_add(y)) & 255).unwrap_or(0)
        });
        let params = CodecParams::video(CodecId::Av1, 720, 480);
        let mut decoder = Av1Decoder::open(&params).expect("decoder");
        let decoded = decoder
            .push(&cutlass_encode::Packet {
                stream: 0,
                data: bytes,
                pts: 0,
                duration: 1,
                timescale: 30,
                keyframe: true,
                side: Vec::new(),
            })
            .expect("decode");
        let video = decoded[0].as_video().expect("video");
        assert_eq!(video.picture.y.width(), 720);
        assert_eq!(video.picture.y.height(), 480);
    }

    #[test]
    fn dav1d_matches_this_decoder_when_present() {
        if std::process::Command::new("dav1d")
            .arg("-v")
            .output()
            .ok()
            .filter(|out| out.status.success())
            .is_none()
        {
            return;
        }
        let bytes = encode(64, 64, 40, |col, _| if col < 32 { 40 } else { 180 });
        let ivf_path = std::env::temp_dir().join(format!("cutlass-av1-{}.ivf", std::process::id()));
        let yuv_path = std::env::temp_dir().join(format!("cutlass-av1-{}.yuv", std::process::id()));
        std::fs::write(&ivf_path, ivf(64, 64, &bytes)).expect("ivf");
        let output = std::process::Command::new("dav1d")
            .arg("-i")
            .arg(&ivf_path)
            .arg("-o")
            .arg(&yuv_path)
            .args(["--muxer", "yuv", "--threads", "1", "-q"])
            .output()
            .expect("dav1d");
        let _ = std::fs::remove_file(&ivf_path);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let decoded = std::fs::read(&yuv_path).expect("yuv");
        let _ = std::fs::remove_file(&yuv_path);
        let (sequence, frame) = parse_unit(&bytes, None).expect("parse");
        let picture = decode_tile(&sequence, &frame).expect("decode");
        let mut ours = picture.y.clone();
        ours.extend_from_slice(&picture.u);
        ours.extend_from_slice(&picture.v);
        assert_eq!(decoded, ours);
    }

    #[test]
    fn a_real_frame_round_trips_when_local_media_is_present() {
        let Some(root) = local_media() else { return };
        if std::process::Command::new("ffmpeg")
            .arg("-version")
            .output()
            .ok()
            .filter(|out| out.status.success())
            .is_none()
        {
            return;
        }
        let Some(input) = first_video(&root, false) else {
            return;
        };
        let yuv_path =
            std::env::temp_dir().join(format!("cutlass-av1-src-{}.yuv", std::process::id()));
        let extracted = std::process::Command::new("ffmpeg")
            .args(["-y", "-i"])
            .arg(&input)
            .args([
                "-frames:v",
                "1",
                "-vf",
                "scale=64:64",
                "-pix_fmt",
                "yuv420p",
                "-f",
                "rawvideo",
            ])
            .arg(&yuv_path)
            .output();
        let Ok(extracted) = extracted else { return };
        if !extracted.status.success() {
            let _ = std::fs::remove_file(&yuv_path);
            return;
        }
        let raw = std::fs::read(&yuv_path).expect("yuv");
        let _ = std::fs::remove_file(&yuv_path);
        if raw.len() < 64 * 64 * 3 / 2 {
            return;
        }
        let picture = Yuv420::new(
            Plane::new(64, 64, raw[..4096].to_vec()).expect("y"),
            Plane::new(32, 32, raw[4096..5120].to_vec()).expect("u"),
            Plane::new(32, 32, raw[5120..6144].to_vec()).expect("v"),
        )
        .expect("yuv");
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut encoder = Av1Encoder::open(&params).expect("encoder");
        encoder.set_quality(40);
        let packet = encoder
            .push_picture(&VideoInput {
                picture,
                pts: 0,
                duration: 1,
                timescale: 30,
            })
            .expect("encode")
            .remove(0);
        let mut decoder = Av1Decoder::open(&params).expect("decoder");
        let decoded = decoder.push(&packet).expect("decode");
        let video = decoded[0].as_video().expect("video");
        assert_eq!(video.picture.y.width(), 64u32);
        assert_eq!(video.picture.y.height(), 64u32);
    }

    #[test]
    fn a_production_av1_keyframe_matches_dav1d_when_present() {
        let Some(root) = local_media() else { return };
        if std::process::Command::new("ffmpeg")
            .arg("-version")
            .output()
            .ok()
            .filter(|out| out.status.success())
            .is_none()
        {
            return;
        }
        let Some(input) = first_video(&root, true) else {
            return;
        };
        let ivf_path =
            std::env::temp_dir().join(format!("cutlass-av1-file-{}.ivf", std::process::id()));
        let extracted = std::process::Command::new("ffmpeg")
            .args(["-y", "-i"])
            .arg(&input)
            .args(["-frames:v", "1", "-c:v", "copy", "-f", "ivf"])
            .arg(&ivf_path)
            .output();
        let Ok(extracted) = extracted else { return };
        if !extracted.status.success() {
            let _ = std::fs::remove_file(&ivf_path);
            return;
        }
        let file = std::fs::read(&ivf_path).expect("ivf");
        if file.len() < 44 || &file[..4] != b"DKIF" {
            let _ = std::fs::remove_file(&ivf_path);
            return;
        }
        let size = u32::from_le_bytes([file[32], file[33], file[34], file[35]]) as usize;
        let start = 44;
        if file.len() < start + size {
            return;
        }
        let params = CodecParams::video(CodecId::Av1, 64, 64);
        let mut decoder = Av1Decoder::open(&params).expect("decoder");
        let packet = cutlass_encode::Packet {
            stream: 0,
            data: file[start..start + size].to_vec(),
            pts: 0,
            duration: 1,
            timescale: 30,
            keyframe: true,
            side: Vec::new(),
        };
        let yuv_path =
            std::env::temp_dir().join(format!("cutlass-av1-file-{}.yuv", std::process::id()));
        match decoder.push(&packet) {
            Ok(frames) => {
                let video = frames
                    .first()
                    .and_then(|frame| frame.as_video())
                    .expect("picture");
                let compared = std::process::Command::new("dav1d")
                    .arg("-i")
                    .arg(&ivf_path)
                    .arg("-o")
                    .arg(&yuv_path)
                    .output()
                    .ok()
                    .filter(|out| out.status.success())
                    .is_some();
                let _ = std::fs::remove_file(&ivf_path);
                if !compared {
                    let _ = std::fs::remove_file(&yuv_path);
                    assert!(video.picture.y.width() > 0);
                    return;
                }
                let raw = std::fs::read(&yuv_path).expect("yuv");
                let _ = std::fs::remove_file(&yuv_path);
                let mut packed = Vec::new();
                packed.extend_from_slice(video.picture.y.samples());
                packed.extend_from_slice(video.picture.u.samples());
                packed.extend_from_slice(video.picture.v.samples());
                assert_eq!(packed, raw);
            }
            Err(error) => {
                let _ = std::fs::remove_file(&ivf_path);
                let text = format!("{error:?}");
                assert!(
                    text.contains("Unsupported")
                        || text.contains("Malformed")
                        || text.contains("Truncated")
                        || text.contains("Limit"),
                    "{text}"
                );
            }
        }
    }

    fn local_media() -> Option<std::path::PathBuf> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../local-files");
        path.is_dir().then_some(path)
    }

    fn first_video(root: &std::path::Path, av1_name: bool) -> Option<std::path::PathBuf> {
        let mut found = std::fs::read_dir(root).ok()?;
        found.find_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?.to_ascii_lowercase();
            let video = name.ends_with(".mp4") || name.ends_with(".webm") || name.ends_with(".mov");
            let named = name.contains("av1");
            if video && named == av1_name {
                Some(path)
            } else {
                None
            }
        })
    }

    fn ivf(width: u16, height: u16, frame: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"DKIF");
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&32u16.to_le_bytes());
        out.extend_from_slice(b"AV01");
        out.extend_from_slice(&width.to_le_bytes());
        out.extend_from_slice(&height.to_le_bytes());
        out.extend_from_slice(&30u32.to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&u32::try_from(frame.len()).unwrap_or(0).to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(frame);
        out
    }
}
