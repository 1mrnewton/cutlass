//! The encoder: a frame's headers, its tiles (see [`tile`]), and the
//! state it leaves for the next frame.
//!
//! The encoder keeps a [`Decoder`] beside it and commits every frame it
//! makes through it, as a decoder would after reading the frame: the
//! probabilities adapted from the frame's symbol counts, the reference
//! slots, the motion of the previous frame. Its headers go through the
//! decoder's parser before its tiles are coded, so that the tiles are coded
//! against exactly the header, probabilities and references a decoder
//! sees, and the reconstruction is the decoder's own (`dec::recon`, the
//! loop filter of `dec::lf`).

pub mod api;
pub mod tile;
pub mod tx;

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;

use cutlass_decode::CodecError;

use crate::bits::{BitWriter, BoolWriter};
use crate::dec::frame::{self, Dims, Mv, Pic};
use crate::dec::lf;
use crate::dec::parse::State;
use crate::dec::recon::{Ctx, Recon, Scratch, View};
use crate::dec::ref_pics;
use crate::decode::Decoder;
use crate::geom;
use crate::header::{self, Parsed};
use crate::lpf;
use crate::prob::{self, FrameContext};

use self::tile::{Shared, Source, Tile};

/// How a frame is coded.
#[derive(Clone, Copy, Debug)]
pub struct FrameParams {
    pub key: bool,
    /// Whether the frame also refreshes GOLDEN (slot 1).
    pub golden: bool,
    pub qindex: u8,
    /// Motion search effort: higher is faster.
    pub speed: u8,
}

/// One coded frame.
pub struct Coded {
    pub data: Vec<u8>,
    pub key: bool,
}

/// The frame-level encoder of 8-bit 4:2:0 pictures.
pub struct FrameEncoder {
    width: u32,
    height: u32,
    dec: Decoder,
    /// The source, padded to whole superblocks by repeating its edges.
    src: [Vec<u8>; 3],
    strides: [usize; 3],
    /// Tile column threads' buffers.
    work: Vec<TileWork>,
    /// Tile columns: at most this many threads (log2 of the count).
    log2_tiles: u8,
}

/// What one tile column's encoder keeps from frame to frame.
struct TileWork {
    /// Its own copy of the picture; only its columns are meaningful.
    pix: [Vec<u8>; 3],
    seeds: Vec<Mv>,
    state: State,
    scratch: Scratch<u8>,
    data: Vec<u8>,
}

impl TileWork {
    fn new(dims: &Dims) -> Self {
        Self {
            pix: std::array::from_fn(|p| vec![0u8; dims.stride[p] * dims.rows[p]]),
            seeds: Vec::new(),
            state: State::empty(),
            scratch: Scratch::new(),
            data: Vec::new(),
        }
    }
}

/// The first 8×8 column of tile `i` of `1 << log2` (libvpx
/// `get_tile_offsets`).
fn tile_edge(i: usize, mi_cols: usize, log2: u8) -> usize {
    let sb_cols = mi_cols.div_ceil(8);
    (((i * sb_cols) >> log2) * 8).min(mi_cols)
}

impl FrameEncoder {
    /// An encoder of `width`×`height` pictures on up to `threads` tile
    /// columns (rounded down to a power of two the width allows).
    pub fn new(width: u32, height: u32, threads: usize) -> Self {
        let dims = Dims::new(width as usize, height as usize, 1, 1, 8);
        let src = std::array::from_fn(|p| vec![0u8; dims.stride[p] * dims.rows[p]]);
        Self {
            width,
            height,
            dec: Decoder::new(),
            src,
            strides: dims.stride,
            work: Vec::new(),
            log2_tiles: threads.max(1).ilog2().min(6) as u8,
        }
    }

    /// The picture the last frame reconstructed (what a decoder shows).
    pub fn last(&self) -> Option<&Arc<Pic<u8>>> {
        self.dec.ref8(0)
    }

    /// Copies a source picture in, repeating its last column and row out
    /// to whole superblocks.
    pub fn load(&mut self, planes: [(&[u8], usize); 3]) {
        let dims = Dims::new(self.width as usize, self.height as usize, 1, 1, 8);
        for (p, (data, stride)) in planes.into_iter().enumerate() {
            let (w, h) = (dims.w[p], dims.h[p]);
            let ds = self.strides[p];
            let dst = &mut self.src[p];
            for y in 0..dims.rows[p] {
                let sy = y.min(h - 1);
                let row = &data[sy * stride..sy * stride + w];
                let out = &mut dst[y * ds..(y + 1) * ds];
                out[..w].copy_from_slice(row);
                let last = row[w - 1];
                out[w..].fill(last);
            }
        }
    }

    /// Codes the loaded source.
    pub fn encode(&mut self, fp: FrameParams) -> Result<Coded, CodecError> {
        let key = fp.key || self.dec.ref8(0).is_none();
        let head = self.headers(key, fp.golden, fp.qindex)?;
        let parsed = header::parse(self.dec.persist(), &head)?;
        let h = &parsed.header;
        let tile_data = self.tiles(&parsed, fp)?;
        let mut data = head.clone();
        data.extend_from_slice(&tile_data.0);
        let (counts, mvs, segmap, pic) = (tile_data.1, tile_data.2, tile_data.3, tile_data.4);
        let _ = h;
        let _shown = self.dec.commit(&parsed, &counts, mvs, segmap, pic);
        Ok(Coded { data, key })
    }

    /// The uncompressed and compressed headers.
    fn headers(&self, key: bool, golden: bool, qindex: u8) -> Result<Vec<u8>, CodecError> {
        let lossless = qindex == 0;
        let compressed = compressed_header(lossless, !key)?;
        let mut b = BitWriter::new();
        b.write(2, 2); // frame marker
        b.write(0, 2); // profile 0
        b.write_bit(false); // show_existing_frame
        b.write_bit(!key); // frame_type
        b.write_bit(true); // show_frame
        b.write_bit(false); // error_resilient_mode
        if key {
            b.write(0x49, 8);
            b.write(0x83, 8);
            b.write(0x42, 8);
            b.write(1, 3); // BT.601
            b.write_bit(false); // studio range
            b.write(self.width - 1, 16);
            b.write(self.height - 1, 16);
            b.write_bit(false); // render size = frame size
        } else {
            b.write(0, 2); // reset_frame_context
            // LAST, and GOLDEN on a golden frame.
            b.write(if golden { 0x03 } else { 0x01 }, 8);
            for slot in 0..3 {
                b.write(slot, 3); // LAST 0, GOLDEN 1, ALTREF 2
                b.write_bit(false); // sign bias
            }
            b.write_bit(true); // size from the first reference
            b.write_bit(false); // render size = frame size
            b.write_bit(true); // allow_high_precision_mv
            b.write_bit(false); // not switchable
            b.write(1, 2); // EIGHTTAP
        }
        b.write_bit(true); // refresh_frame_context
        b.write_bit(false); // frame_parallel_decoding_mode
        b.write(0, 2); // frame_context_idx
        // Loop filter: libvpx's level from the quantizer.
        let level = if lossless || std::env::var("NOLF").is_ok() { 0 } else { filter_level(qindex, key) };
        b.write(u32::from(level), 6);
        b.write(0, 3); // sharpness
        b.write_bit(true); // mode_ref_delta_enabled (the defaults)
        b.write_bit(false); // no update
        b.write(u32::from(qindex), 8);
        b.write(0, 3); // no dc / uv deltas
        b.write_bit(false); // no segmentation
        let (min_log2, max_log2) = header::tile_limits(geom::mi_count(self.width));
        let log2 = u32::from(self.log2_tiles).clamp(min_log2, max_log2);
        for _ in min_log2..log2 {
            b.write_bit(true);
        }
        if log2 < max_log2 {
            b.write_bit(false);
        }
        b.write_bit(false); // one tile row
        let len = u32::try_from(compressed.len()).unwrap_or(u32::MAX);
        if len > u32::from(u16::MAX) {
            return Err(CodecError::Limit { what: "compressed header" });
        }
        b.write(len, 16);
        let mut out = b.finish();
        out.extend_from_slice(&compressed);
        Ok(out)
    }

    /// Codes the tiles; returns their bytes, the frame's symbol counts,
    /// motion, segmentation map and reconstructed picture.
    #[allow(clippy::type_complexity)]
    fn tiles(
        &mut self,
        parsed: &Parsed<'_>,
        fp: FrameParams,
    ) -> Result<(Vec<u8>, prob::Counts, Vec<frame::MvRef>, Vec<u8>, Arc<Pic<u8>>), CodecError> {
        let h = &parsed.header;
        let dims = Dims::new(h.width as usize, h.height as usize, 1, 1, 8);
        let mi_cols = dims.width.div_ceil(8);
        let mi_rows = dims.height.div_ceil(8);
        let sb_cols = mi_cols.div_ceil(8);
        let sb_rows = mi_rows.div_ceil(8);
        let slot = |i: usize| self.dec.ref8(usize::from(h.ref_index[i])).map(|p| &**p);
        let refs = if h.is_intra() { [None, None, None] } else { [slot(0), slot(1), slot(2)] };
        // A reference worth searching holds a picture no earlier one does.
        let same = |a: Option<&Pic<u8>>, b: Option<&Pic<u8>>| match (a, b) {
            (Some(a), Some(b)) => std::ptr::eq(a, b),
            _ => false,
        };
        let refs_live = [
            refs[0].is_some(),
            refs[1].is_some() && !same(refs[1], refs[0]),
            refs[2].is_some() && !same(refs[2], refs[0]) && !same(refs[2], refs[1]),
        ];
        let (pics, _) = ref_pics(parsed, refs);
        let ctx = Ctx {
            dims,
            refs: pics,
            mi_cols,
            mi_rows,
            lossless: h.lossless,
        };
        let levels = lpf::levels(h);
        let geo = lf::Geo {
            mi_rows,
            mi_cols,
            sb_cols,
            ss_x: 1,
            ss_y: 1,
            bd: 8,
        };
        let filter_on = h.filter_level != 0;
        let mut writer = frame::writer(&dims, self.dec.spare8());
        let prev_mvs = if h.use_prev_mvs { self.dec.prev_mvs() } else { None };
        let ac = i32::from(h.y_dq[0][1]);
        let shared = Shared {
            h,
            fc: &parsed.fc,
            prev_mvs,
            levels: &levels,
            mi_cols,
            mi_rows,
            intra_frame: h.is_intra(),
            part_probs: if h.is_intra() { prob::kf_partition() } else { parsed.fc.partition },
            filter_on,
            lambda: sad_per_bit(ac),
            rd_lambda: rd_lambda(i32::from(h.y_dq[0][0])),
            src: Source {
                planes: [&self.src[0], &self.src[1], &self.src[2]],
                strides: self.strides,
            },
            dq: [
                [i32::from(h.y_dq[0][0]), ac],
                [i32::from(h.uv_dq[0][0]), i32::from(h.uv_dq[0][1])],
            ],
            speed: fp.speed,
            refs_live,
        };
        // Each tile column on its own thread, into its own copy of the
        // picture (intra prediction reads nothing across tile columns).
        let ntiles = 1usize << h.log2_tile_cols;
        let edges: Vec<usize> = (0..=ntiles).map(|i| tile_edge(i, mi_cols, h.log2_tile_cols)).collect();
        while self.work.len() < ntiles {
            self.work.push(TileWork::new(&dims));
        }
        let plane_len: [usize; 3] = std::array::from_fn(|p| dims.stride[p] * dims.rows[p]);
        let run = |t: &mut TileWork, col0: usize, col1: usize| -> Result<(), CodecError> {
            for (p, buf) in t.pix.iter_mut().enumerate() {
                buf.resize(plane_len[p], 0);
            }
            let tsb = (col1 - col0).div_ceil(8);
            t.state.reset(col0, col1 - col0, mi_rows, sb_rows * tsb, false);
            let mut w = BoolWriter::new();
            let [y, u, v] = &mut t.pix;
            let mut views = [
                View { data: &mut y[..], base: 0, stride: dims.stride[0] },
                View { data: &mut u[..], base: 0, stride: dims.stride[1] },
                View { data: &mut v[..], base: 0, stride: dims.stride[2] },
            ];
            let mut seeds = std::mem::take(&mut t.seeds);
            for r in 0..sb_rows {
                let mut rec = Recon::new(&ctx, &mut views, &mut t.scratch);
                let mut tile = Tile::new(&shared, &mut t.state, &mut w, &mut rec, (col0, col1), seeds);
                tile.sb_row(r * 8);
                seeds = tile.finish();
            }
            t.seeds = seeds;
            t.data = w.finish()?;
            Ok(())
        };
        let work = &mut self.work[..ntiles];
        if ntiles == 1 {
            run(&mut work[0], 0, mi_cols)?;
        } else {
            let results: Vec<Result<(), CodecError>> = std::thread::scope(|sc| {
                let handles: Vec<_> = work
                    .iter_mut()
                    .enumerate()
                    .map(|(i, t)| {
                        let (c0, c1) = (edges[i], edges[i + 1]);
                        let run = &run;
                        sc.spawn(move || run(t, c0, c1))
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap_or(Err(CodecError::Malformed { at_bit: 0, what: "vp9 encoder thread" })))
                    .collect()
            });
            for r in results {
                r?;
            }
        }
        // The tiles' columns into the picture, their state into the
        // frame's; then the loop filter, row by row.
        let buf = Arc::clone(writer.shared());
        let mut masks = vec![lpf::Mask::default(); sb_rows * sb_cols];
        let mut mvs = vec![frame::MvRef::default(); mi_cols * mi_rows];
        let mut counts = prob::Counts::new();
        let mut data = Vec::new();
        {
            let mut views = writer.tails().map(View::of);
            for (i, t) in work.iter().enumerate() {
                let (c0, c1) = (edges[i], edges[i + 1]);
                for (p, v) in views.iter_mut().enumerate() {
                    let ss = usize::from(p > 0);
                    let (x0, x1) = ((c0 * 8) >> ss, ((c1 * 8) >> ss).min(dims.stride[p]));
                    let stride = v.stride;
                    for row in 0..dims.rows[p] {
                        v.data[row * stride + x0..row * stride + x1].copy_from_slice(&t.pix[p][row * stride + x0..row * stride + x1]);
                    }
                }
                let st = &t.state;
                let w = c1 - c0;
                let tsb = w.div_ceil(8);
                for r in 0..sb_rows {
                    masks[r * sb_cols + c0 / 8..r * sb_cols + c0 / 8 + tsb].copy_from_slice(&st.masks[r * tsb..r * tsb + tsb]);
                }
                for r in 0..mi_rows {
                    mvs[r * mi_cols + c0..r * mi_cols + c1].copy_from_slice(&st.mvs[r * w..r * w + w]);
                }
                counts.add(&st.counts);
                let d = &t.data;
                if i + 1 < ntiles {
                    data.extend_from_slice(&(d.len() as u32).to_be_bytes());
                }
                data.extend_from_slice(d);
            }
            if filter_on {
                for r in 0..sb_rows {
                    lf::sb_row(&mut views, &masks, &[], &levels, &geo, r);
                }
            }
        }
        writer.publish(usize::MAX);
        drop(writer);
        let pic = Arc::new(Pic {
            buf,
            dims,
            color_space: h.color_space,
            color_range: h.color_range,
            render_width: h.render_width,
            render_height: h.render_height,
            out: Mutex::new(None),
            made: AtomicBool::new(false),
        });
        let segmap = vec![0u8; mi_cols * mi_rows];
        Ok((data, counts, mvs, segmap, pic))
    }
}

/// The SAD one bit of rate is worth: libvpx's `sad_per_bit16` from the
/// AC quantizer step.
fn sad_per_bit(ac: i32) -> u32 {
    let q = f64::from(ac) / 4.0;
    (0.0418 * q + 2.4107).round().max(1.0) as u32
}

/// The sample SSE a bit is worth: libvpx's `rdmult` (88·q²/24, on
/// distortions 16 times the SSE, rates in 1/512 bits over a 128 scale).
fn rd_lambda(dc: i32) -> f32 {
    let k = std::env::var("LK").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(1.0);
    0.00179 * (dc * dc) as f32 * k
}

/// libvpx's loop filter level from the quantizer (`LPF_PICK_FROM_Q`,
/// 8-bit).
pub fn filter_level(qindex: u8, key: bool) -> u8 {
    let q = i64::from(crate::tables::AC_QLOOKUP[usize::from(qindex)]);
    let mut guess = (q * 20723 + 1_015_158 + (1 << 17)) >> 18;
    if key {
        guess -= 4;
    }
    guess.clamp(0, 63) as u8
}

fn compressed_header(lossless: bool, inter: bool) -> Result<Vec<u8>, CodecError> {
    let mut w = BoolWriter::new();
    if !lossless {
        // TX_MODE_SELECT, its probabilities as they are.
        w.write_literal(u32::from(geom::ALLOW_32X32), 2);
        w.write(1, 128);
        for _ in 0..2 * (1 + 2 + 3) {
            w.write(0, 252);
        }
    }
    let max_tx = if lossless { 0 } else { 3 };
    for _ in 0..=max_tx {
        w.write(0, 128); // no coefficient updates
    }
    for _ in 0..3 {
        w.write(0, 252); // skip
    }
    if inter {
        let fc = FrameContext::defaults();
        let none = |w: &mut BoolWriter, n: usize| {
            for _ in 0..n {
                w.write(0, 252);
            }
        };
        none(&mut w, fc.inter_mode.len() * fc.inter_mode[0].len());
        none(&mut w, fc.intra_inter.len());
        none(&mut w, fc.single_ref.len() * 2);
        none(&mut w, fc.y_mode.len() * fc.y_mode[0].len());
        none(&mut w, fc.partition.len() * 3);
        none(&mut w, fc.mv.joints.len());
        for c in &fc.mv.comp {
            none(&mut w, 1 + c.classes.len() + 1 + c.bits.len());
        }
        for c in &fc.mv.comp {
            none(&mut w, c.class0_fp.len() * c.class0_fp[0].len() + c.fp.len());
        }
        for _ in &fc.mv.comp {
            none(&mut w, 2); // class0_hp, hp
        }
    }
    w.finish()
}

#[cfg(test)]
mod tests;
