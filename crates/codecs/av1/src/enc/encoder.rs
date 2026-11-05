//! Frame-level encoding: headers, tiles, packets and references.
//!
//! Every frame is coded by the decoder's own tile code running with a writing
//! [`Coder`]. The finished temporal unit is then decoded by an internal
//! [`Av1Core`], which produces the reference frames (with the in-loop filters)
//! exactly as any decoder will and checks every packet.

use std::sync::Arc;

use cutlass_decode::CodecError;

use crate::dec::cdf::CdfContext;
use crate::dec::coder::Coder;
use crate::dec::decoder::{AnyPicture, Av1Core, PicturePixel};
use crate::dec::frame::{FrameState, Picture};
use crate::dec::obu::{
    Cdef, FrameHeader, HeaderContext, INTER_FRAME, KEY_FRAME, LoopFilter, OBU_FRAME,
    OBU_SEQUENCE_HEADER, OBU_TEMPORAL_DELIMITER, OperatingPoint, PRIMARY_REF_NONE, Quant,
    SequenceHeader, TX_MODE_SELECT, parse_frame_header,
};
use crate::dec::tile::{RefView, TileDec};
use crate::dsp::Pixel;
use crate::spec::tables::{AC_QLOOKUP, DC_QLOOKUP};

use super::bitwriter::write_obu;
use super::msac_enc::MsacEnc;
use super::obu_write::{
    FrameHeaderWriteOptions, uniform_tile_info, write_frame_header, write_sequence_header,
    write_tile_group,
};
use super::quant::QuantParams;
use super::rdo::ModeParams;

type R<T> = Result<T, CodecError>;

impl Coder for MsacEnc {
    const WRITE: bool = true;
    #[inline(always)]
    fn bool_adapt(&mut self, cdf: &mut [u16; 2], v: bool) -> bool {
        MsacEnc::bool_adapt(self, cdf, v);
        v
    }
    #[inline(always)]
    fn bool_prob(&mut self, f: u32, v: bool) -> bool {
        MsacEnc::bool_prob(self, f, v);
        v
    }
    #[inline(always)]
    fn symbol<const N: usize>(&mut self, cdf: &mut [u16; N], v: usize) -> usize {
        let v = v.min(N - 1);
        MsacEnc::symbol(self, cdf, v);
        v
    }
    #[inline(always)]
    fn literal(&mut self, n: u32, v: u32) -> u32 {
        let v = if n >= 32 { v } else { v & ((1u32 << n) - 1) };
        MsacEnc::literal(self, n, v);
        v
    }
    #[inline(always)]
    fn golomb(&mut self, v: u32) -> u32 {
        MsacEnc::golomb(self, v);
        v
    }
    #[inline(always)]
    fn ns(&mut self, n: u32, v: u32) -> u32 {
        let v = if n == 0 { 0 } else { v.min(n - 1) };
        MsacEnc::ns(self, n, v);
        v
    }
}

/// How the encoder chooses quantizers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RateControl {
    /// A fixed base_q_idx (0..=255) for every frame.
    Qindex(u8),
    /// Target bits per second at the given frame rate.
    Bitrate { bps: u64, fps: f64 },
}

#[derive(Clone, Debug)]
pub struct EncConfig {
    pub width: usize,
    pub height: usize,
    pub bit_depth: u32,
    pub sub_x: bool,
    pub sub_y: bool,
    pub mono: bool,
    /// 0 is slowest and best, 10 fastest.
    pub speed: u8,
    pub rc: RateControl,
    /// Distance between key frames.
    pub keyint: u32,
    pub tile_cols_log2: u32,
    pub tile_rows_log2: u32,
    pub color_primaries: u8,
    pub transfer: u8,
    pub matrix: u8,
    pub full_range: bool,
}

impl EncConfig {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            bit_depth: 8,
            sub_x: true,
            sub_y: true,
            mono: false,
            speed: 6,
            rc: RateControl::Qindex(120),
            keyint: 120,
            tile_cols_log2: 0,
            tile_rows_log2: 0,
            color_primaries: 2,
            transfer: 2,
            matrix: 2,
            full_range: false,
        }
    }
}

/// One input picture, as plane slices with strides.
pub struct InputFrame<'a, P> {
    pub planes: [&'a [P]; 3],
    pub strides: [usize; 3],
}

/// The AV1 encoder.
pub struct Av1Enc {
    cfg: EncConfig,
    seq: SequenceHeader,
    core: Av1Core,
    frame_num: u64,
    rc_q: f64,
    /// Bits produced so far and frames coded, for rate control.
    bits_total: u64,
    /// Last reconstructed frame (shown output of the internal decoder).
    last_recon: Option<Arc<AnyPicture>>,
}

fn bits_for(v: usize) -> u32 {
    let mut n = 1;
    while (1usize << n) < v {
        n += 1;
    }
    n
}

impl Av1Enc {
    pub fn new(cfg: EncConfig) -> R<Self> {
        if cfg.width == 0 || cfg.height == 0 || cfg.width > 16384 || cfg.height > 16384 {
            return Err(CodecError::Unsupported { what: "frame size" });
        }
        if !matches!(cfg.bit_depth, 8 | 10 | 12) {
            return Err(CodecError::Unsupported { what: "bit depth" });
        }
        let profile = if cfg.bit_depth == 12 || (cfg.sub_x && !cfg.sub_y && !cfg.mono) {
            2
        } else if !cfg.sub_x && !cfg.sub_y && !cfg.mono {
            1
        } else {
            0
        };
        let (sub_x, sub_y) = if cfg.mono {
            (true, true)
        } else {
            (cfg.sub_x, cfg.sub_y)
        };
        let seq = SequenceHeader {
            profile,
            operating_points: vec![OperatingPoint {
                idc: 0,
                seq_level_idx: 31,
                ..OperatingPoint::default()
            }],
            frame_width_bits: bits_for(cfg.width),
            frame_height_bits: bits_for(cfg.height),
            max_frame_width: cfg.width as u32,
            max_frame_height: cfg.height as u32,
            enable_filter_intra: true,
            enable_intra_edge_filter: true,
            enable_order_hint: true,
            order_hint_bits: 7,
            seq_force_screen_content_tools: 0,
            seq_force_integer_mv: 2,
            enable_cdef: true,
            bit_depth: cfg.bit_depth,
            mono_chrome: cfg.mono,
            color_primaries: cfg.color_primaries,
            transfer_characteristics: cfg.transfer,
            matrix_coefficients: cfg.matrix,
            color_range: cfg.full_range,
            subsampling_x: sub_x,
            subsampling_y: sub_y,
            ..SequenceHeader::default()
        };
        let q0 = match cfg.rc {
            RateControl::Qindex(q) => f64::from(q),
            RateControl::Bitrate { .. } => 120.0,
        };
        Ok(Self {
            cfg,
            seq,
            core: Av1Core::new(),
            frame_num: 0,
            rc_q: q0,
            bits_total: 0,
            last_recon: None,
        })
    }

    pub fn sequence_header(&self) -> &SequenceHeader {
        &self.seq
    }

    /// The reconstruction of the last coded frame.
    pub fn last_recon(&self) -> Option<&Arc<AnyPicture>> {
        self.last_recon.as_ref()
    }

    /// Encodes one picture and returns one temporal unit.
    pub fn encode<P: PicturePixel>(&mut self, input: &InputFrame<'_, P>) -> R<Vec<u8>> {
        if P::HBD != (self.cfg.bit_depth > 8) {
            return Err(CodecError::InvalidApi {
                what: "sample size",
            });
        }
        let key = self
            .frame_num
            .is_multiple_of(u64::from(self.cfg.keyint.max(1)));
        let q = self.rc_q.round().clamp(1.0, 255.0) as u8;
        let mut out = Vec::new();
        write_obu(&mut out, OBU_TEMPORAL_DELIMITER, 0, 0, false, &[]);
        if key {
            let sh = write_sequence_header(&self.seq);
            write_obu(&mut out, OBU_SEQUENCE_HEADER, 0, 0, false, &sh);
        }
        let frame = self.encode_frame(input, key, q)?;
        write_obu(&mut out, OBU_FRAME, 0, 0, false, &frame);
        // Reconstruct exactly as a decoder does; this also checks the packet.
        let shown = self.core.decode(&out)?;
        if let Some(f) = shown.into_iter().last() {
            self.last_recon = Some(f.picture);
        }
        self.frame_num += 1;
        self.bits_total += out.len() as u64 * 8;
        self.update_rate_control(out.len() as u64 * 8, key);
        Ok(out)
    }

    fn update_rate_control(&mut self, bits: u64, key: bool) {
        let RateControl::Bitrate { bps, fps } = self.cfg.rc else {
            return;
        };
        let target = bps as f64 / fps.max(1.0);
        // Key frames are expected to cost more.
        let expect = if key { target * 4.0 } else { target };
        let ratio = (bits as f64 / expect.max(1.0)).max(0.05);
        // About 6 q-index steps per doubling of the rate near the middle range.
        self.rc_q = (self.rc_q + 12.0 * ratio.log2()).clamp(10.0, 250.0);
        let avg = self.bits_total as f64 / self.frame_num.max(1) as f64;
        if avg > target * 1.1 {
            self.rc_q = (self.rc_q + 1.0).min(250.0);
        } else if avg < target * 0.9 {
            self.rc_q = (self.rc_q - 1.0).max(10.0);
        }
    }

    fn frame_header(&self, key: bool, q: u8) -> FrameHeader {
        let bd = self.cfg.bit_depth;
        let bdi = ((bd - 8) >> 1) as usize;
        let ac = i32::from(AC_QLOOKUP[bdi][q as usize]) >> (bd - 8);
        // Loop filter strength from the quantizer (libaom-style fit).
        let lvl = if key {
            (ac * 20723 + 1_015_158) >> 18
        } else {
            (ac * 20723 + 1_015_158 - 300_000) >> 18
        }
        .clamp(0, 63) as u8;
        let ev = |k: &str, d: f64| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        let lvl = (f64::from(lvl) * ev("AV1_LFS", 1.0)).clamp(0.0, 63.0) as u8;
        let pri = ((f64::from(q) / 24.0) * ev("AV1_CDS", 1.0)).clamp(0.0, 15.0) as u8;
        let mut h = FrameHeader {
            frame_type: if key { KEY_FRAME } else { INTER_FRAME },
            frame_is_intra: key,
            show_frame: true,
            showable_frame: !key,
            error_resilient_mode: key,
            order_hint: (self.frame_num & 127) as u32,
            primary_ref_frame: if key { PRIMARY_REF_NONE } else { 0 },
            refresh_frame_flags: if key { 0xff } else { 0x01 },
            frame_width: self.cfg.width as u32,
            frame_height: self.cfg.height as u32,
            upscaled_width: self.cfg.width as u32,
            render_width: self.cfg.width as u32,
            render_height: self.cfg.height as u32,
            superres_denom: 8,
            ref_frame_idx: [0; 7],
            allow_high_precision_mv: !key && q < 128,
            interpolation_filter: 0,
            tile_info: {
                let mi_cols = 2 * ((self.cfg.width as u32 + 7) >> 3);
                let mi_rows = 2 * ((self.cfg.height as u32 + 7) >> 3);
                let mut ti = uniform_tile_info(
                    &self.seq,
                    mi_cols,
                    mi_rows,
                    self.cfg.tile_cols_log2,
                    self.cfg.tile_rows_log2,
                );
                if ti.cols * ti.rows > 1 {
                    ti.tile_size_bytes = 4;
                }
                ti
            },
            quant: Quant {
                base_q_idx: q,
                ..Quant::default()
            },
            loop_filter: LoopFilter {
                level: [lvl, lvl, lvl, lvl],
                ref_deltas: crate::dec::obu::DEFAULT_LF_REF_DELTAS,
                ..LoopFilter::default()
            },
            cdef: Cdef {
                damping: 3 + u8::from(q > 120),
                bits: 0,
                y_pri: [pri; 8],
                y_sec: [u8::from(q > 80) * 2; 8],
                uv_pri: [pri / 2; 8],
                uv_sec: [u8::from(q > 80); 8],
            },
            tx_mode: TX_MODE_SELECT,
            ..FrameHeader::default()
        };
        if key {
            h.ref_order_hint = [0; 8];
        }
        h
    }

    fn encode_frame<P: PicturePixel>(
        &mut self,
        input: &InputFrame<'_, P>,
        key: bool,
        q: u8,
    ) -> R<Vec<u8>> {
        let seq = self.seq.clone();
        let intent = self.frame_header(key, q);
        let refs_hdr = self.core.ref_headers().clone();
        let ctx = HeaderContext {
            seq: &seq,
            refs: &refs_hdr,
            temporal_id: 0,
            spatial_id: 0,
        };
        let opts = FrameHeaderWriteOptions::default();
        let hdr_bytes = write_frame_header(&intent, &ctx, &opts);
        // The parser derives every dependent field exactly as a decoder will.
        let hdr = parse_frame_header(&hdr_bytes, &ctx)?;
        let src = source_picture(&self.cfg, &seq, input);
        // Initial CDFs.
        let cdf: Box<CdfContext> = if hdr.primary_ref_frame == PRIMARY_REF_NONE {
            CdfContext::new(hdr.quant.base_q_idx)
        } else {
            let slot = hdr.ref_frame_idx[hdr.primary_ref_frame as usize] as usize;
            let Some(c) = self.core.slot_cdf(slot) else {
                return Err(CodecError::InvalidApi {
                    what: "missing reference",
                });
            };
            let mut c = Box::new((*c).clone());
            c.clear_counts();
            c
        };
        let mut fs = FrameState::new(&seq, &hdr);
        let mut cur = Picture::<P>::new(
            hdr.frame_width as usize,
            hdr.frame_height as usize,
            seq.num_planes(),
            seq.subsampling_x,
            seq.subsampling_y,
            seq.bit_depth,
        );
        let ref_pics: Vec<Option<Arc<AnyPicture>>> =
            (0..8).map(|s| self.core.slot_picture(s)).collect();
        let bd = seq.bit_depth;
        let bdi = ((bd - 8) >> 1) as usize;
        let ac = f64::from(AC_QLOOKUP[bdi][q as usize]);
        let dcq = f64::from(DC_QLOOKUP[bdi][q as usize]);
        let qstep = ac / 8.0;
        let speed = self.cfg.speed;
        let ev = |k: &str, d: f64| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        let mode = ModeParams {
            lambda_satd: ev("AV1_LS", 0.9) * qstep,
            min_block: ev("AV1_MINB", if speed >= 8 { 16.0 } else { 8.0 }) as usize,
            max_block: ev("AV1_MAXB", 32.0) as usize,
            skip_sse_per_px: ev("AV1_SK", 0.25) * qstep * qstep,
            inter_frame: !key,
            me_range: if speed >= 8 { 16 } else { 48 },
            subpel: if speed >= 9 { 1 } else { 2 },
            intra_modes: if speed >= 8 { 10 } else { 13 },
            tx_depth: if speed >= 9 {
                0
            } else if speed >= 5 {
                1
            } else {
                2
            },
            cfl: speed < 9,
            angle_deltas: speed < 7,
            filter_intra: speed < 7,
            tx64: ev("AV1_TX64", 0.0) > 0.0,
            rd_partition: ev("AV1_RDP", 1.0) > 0.0,
        };
        let qp = QuantParams {
            // Per bit; inter frames trade residual for rate more readily.
            lambda: ev("AV1_QL", if key { 0.15 } else { 0.9 }) * qstep * qstep,
            round_intra: ev("AV1_RI", 56.0) as i32,
            round_inter: ev("AV1_RN", 44.0) as i32,
            tx_type_search: u8::from(speed < 9),
            rdoq: if speed >= 9 {
                0
            } else if speed >= 7 {
                1
            } else {
                2
            },
        };
        let _ = dcq;
        let ti = hdr.tile_info.clone();
        let n_above = fs.mi_stride + 64;
        let mut tile_bytes: Vec<Vec<u8>> = Vec::with_capacity(ti.cols * ti.rows);
        for tile in 0..ti.cols * ti.rows {
            let tile_row = tile / ti.cols;
            let tile_col = tile % ti.cols;
            let mut tile_cdf = Box::new((*cdf).clone());
            let ref_views: [Option<RefView<'_, P>>; 8] = std::array::from_fn(|i| {
                if i == 0 || hdr.frame_is_intra {
                    return None;
                }
                let slot = hdr.ref_frame_idx[i - 1] as usize;
                let pic = ref_pics[slot].as_deref()?;
                let p = P::view(pic)?;
                Some(RefView {
                    pic: p,
                    upscaled_width: refs_hdr[slot].upscaled_width as usize,
                    frame_height: refs_hdr[slot].frame_height as usize,
                    frame_width: refs_hdr[slot].frame_width as usize,
                })
            });
            let mut td = TileDec {
                seq: &seq,
                hdr: &hdr,
                fs: &mut fs,
                cur: &mut cur,
                refs: ref_views,
                msac: MsacEnc::new(!hdr.disable_cdf_update),
                pb: Default::default(),
                plan: Default::default(),
                plan_q: 0,
                cfl_choice: (0, 0),
                intra_choice: (0, None),
                plan_delta_lf: [0; 4],
                src: Some(&src),
                enc: qp,
                mode,
                me_scratch: Default::default(),
                cdf: &mut tile_cdf,
                mi_row_start: ti.mi_row_starts[tile_row] as usize,
                mi_row_end: ti.mi_row_starts[tile_row + 1] as usize,
                mi_col_start: ti.mi_col_starts[tile_col] as usize,
                mi_col_end: ti.mi_col_starts[tile_col + 1] as usize,
                sb_size: if seq.use_128x128_superblock { 15 } else { 12 },
                sb_size4: if seq.use_128x128_superblock { 32 } else { 16 },
                bd: seq.bit_depth,
                sub_x: seq.subsampling_x,
                sub_y: seq.subsampling_y,
                num_planes: seq.num_planes(),
                above_level: std::array::from_fn(|_| vec![0u8; n_above]),
                above_dc: std::array::from_fn(|_| vec![0u8; n_above]),
                above_seg_pred: vec![0u8; n_above],
                left_level: [[0; 32]; 3],
                left_dc: [[0; 32]; 3],
                left_seg_pred: [0; 32],
                above_palette: std::array::from_fn(|_| vec![[0u16; 8]; n_above]),
                left_palette: [[[0; 8]; 32]; 2],
                delta_lf: [0; 4],
                current_q_index: i32::from(hdr.quant.base_q_idx),
                read_deltas: false,
                ref_lr_wiener: [[[0; 3]; 2]; 3],
                ref_sgr_xqd: [[0; 2]; 3],
                block_decoded: [[[false; 35]; 35]; 3],
                b: Default::default(),
                color_map_y: vec![0u8; 128 * 128],
                color_map_uv: vec![0u8; 128 * 128],
                scratch: Default::default(),
            };
            td.decode_tile()?;
            let TileDec { msac, .. } = td;
            tile_bytes.push(msac.finish());
        }
        drop(cdf);
        // Frame OBU: header, then one tile group with every tile.
        let mut payload = hdr_bytes;
        payload.extend_from_slice(&write_tile_group(
            &ti,
            0,
            ti.cols * ti.rows - 1,
            &tile_bytes,
        ));
        Ok(payload)
    }
}

/// Copies the input into a picture padded by edge replication to whole superblocks.
fn source_picture<P: Pixel>(
    cfg: &EncConfig,
    seq: &SequenceHeader,
    input: &InputFrame<'_, P>,
) -> Picture<P> {
    let mut pic = Picture::<P>::new(
        cfg.width,
        cfg.height,
        seq.num_planes(),
        seq.subsampling_x,
        seq.subsampling_y,
        seq.bit_depth,
    );
    for (p, pl) in pic.planes.iter_mut().enumerate() {
        let src = input.planes[p];
        let ss = input.strides[p];
        let (w, h) = (pl.w, pl.h);
        let stride = pl.stride;
        for y in 0..pl.alloc_h {
            let sy = y.min(h - 1);
            let row = &mut pl.data[y * stride..(y + 1) * stride];
            row[..w].copy_from_slice(&src[sy * ss..sy * ss + w]);
            let last = row[w - 1];
            for v in &mut row[w..] {
                *v = last;
            }
        }
    }
    pic
}
