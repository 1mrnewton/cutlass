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
pub mod aq;
pub mod arf;
pub mod probs;
pub mod tile;
pub mod tx;

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;

use cutlass_decode::CodecError;

use crate::bits::{BitWriter, BoolWriter};
use crate::dec::frame::{self, Dims, Mv, Pic};
use crate::dec::lf;
use crate::dec::parse::{LfMi, State};
use crate::dec::recon::{Ctx, Recon, Scratch, View};
use crate::dec::ref_pics;
use crate::decode::{Decoder, Store};
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
    /// A hidden frame (an alt-ref): not shown, refreshes ALTREF (slot 2)
    /// only.
    pub hidden: bool,
    pub qindex: u8,
    /// Motion search effort: higher is faster.
    pub speed: u8,
    /// Whether ALTREF holds a picture still to come (an alt-ref before its
    /// overlay): its sign bias is set, and blocks may predict from it and
    /// a past picture at once.
    pub altref_future: bool,
    /// Adaptive quantization by segment ([`aq`]).
    pub aq: bool,
    /// A second pass's effort: at speed 6 a block that codes a residual
    /// of some size also tries the next smaller transform, as speeds 0 to
    /// 5 do (see [`tile::TX_TRIAL_BITS`]), and the best inter candidates
    /// are ranked with their chroma too (see [`tile::UV_RANK`]).
    pub second_pass: bool,
    /// The slot GOLDEN reads (1 but in two passes, see
    /// [`super::api`]'s reference pool).
    pub golden_slot: u8,
    /// Slots this frame also refreshes, besides its type's.
    pub also_refresh: u8,
    /// Two passes: the error a luma sample of the picture keeps from its
    /// best reference picture (its noise), 0 when unknown.
    pub noise: f32,
}

/// What every frame of the stream signals, set once for the stream.
#[derive(Clone, Copy, Debug)]
pub struct Stream {
    /// The loop filter's sharpness, 0 to 7 (libvpx's `sharpness`).
    pub sharpness: u8,
    /// The colour space code of the key frames' colour config (0
    /// unknown, 1 BT.601, 2 BT.709, 3 SMPTE-170, 4 SMPTE-240, 5 BT.2020,
    /// 7 sRGB, in profiles 1 and 3 only).
    pub color_space: u8,
    /// Full (`true`) or studio range.
    pub full_range: bool,
    /// The size the picture is meant to be shown at, when not its own.
    pub render_size: Option<(u32, u32)>,
    /// Every frame decodable without the probabilities, motion and
    /// segmentation of the frames before it (`error_resilient_mode`).
    pub error_resilient: bool,
    /// No backward adaptation of the probabilities
    /// (`frame_parallel_decoding_mode`), so that a decoder can start a
    /// frame's parse before the one before it is parsed through.
    pub frame_parallel: bool,
    /// libvpx's `static-thresh`: an inter block whose prediction error is
    /// under this threshold (scaled to its size) codes no residual.
    pub static_thresh: u32,
}

impl Default for Stream {
    fn default() -> Self {
        Self {
            sharpness: 0,
            color_space: 1,
            full_range: false,
            render_size: None,
            error_resilient: false,
            frame_parallel: false,
            static_thresh: 0,
        }
    }
}

/// Speeds at and below which a frame with a future ALTREF lets each block
/// choose compound prediction.
pub const COMPOUND_SPEED: u8 = 6;

/// From this speed a reference that predicted almost nothing of the last
/// inter frame is not searched (see [`FrameEncoder::tiles`]).
pub const REF_PRUNE_SPEED: u8 = 5;

/// From this speed each tile column's rows are coded on one thread (as
/// many as the threads allow beyond one a column): a superblock is too
/// little work for the wavefront's waits. Speed 10 on 720p at four
/// threads (four tile columns): the tiles in 0.7 of the time with one
/// lane a column against two. The frames are the same either way.
pub const ONE_LANE_SPEED: u8 = 9;

/// At this speed each tile column's thread makes its own columns' part of
/// the fast path's search ([`tile::prepass_sb`]) row by row as it codes
/// them, instead of the frame's search coming first on its own threads.
/// (Not at speed 10: it is also the first pass of two passes and the probe
/// of one, whose bits the rate control of the other speeds reads.)
pub const FUSED_SPEED: u8 = 9;

/// One coded frame.
pub struct Coded {
    pub data: Vec<u8>,
    pub key: bool,
}

/// The samples the encoder codes: 8-bit (`u8`) or 10- and 12-bit
/// (`u16`), with the distortion measures of its searches.
pub trait EncPixel: Store {
    fn sad(a: &[Self], sa: usize, b: &[Self], sb: usize, w: usize, h: usize) -> u32;
    fn sad4(a: &[Self], sa: usize, b: [&[Self]; 4], sb: usize, w: usize, h: usize) -> [u32; 4];
    fn sse(a: &[Self], sa: usize, b: &[Self], sb: usize, w: usize, h: usize) -> u64;
    #[allow(clippy::too_many_arguments)]
    fn sad_bilinear(
        a: &[Self],
        sa: usize,
        b: &[Self],
        sb: usize,
        w: usize,
        h: usize,
        fy: u32,
        fx: u32,
    ) -> u32;
}

impl EncPixel for u8 {
    #[inline(always)]
    fn sad(a: &[u8], sa: usize, b: &[u8], sb: usize, w: usize, h: usize) -> u32 {
        crate::dsp::sad::sad(a, sa, b, sb, w, h)
    }
    #[inline(always)]
    fn sad4(a: &[u8], sa: usize, b: [&[u8]; 4], sb: usize, w: usize, h: usize) -> [u32; 4] {
        crate::dsp::sad::sad4(a, sa, b, sb, w, h)
    }
    #[inline(always)]
    fn sse(a: &[u8], sa: usize, b: &[u8], sb: usize, w: usize, h: usize) -> u64 {
        crate::dsp::sad::sse(a, sa, b, sb, w, h)
    }
    #[inline(always)]
    fn sad_bilinear(
        a: &[u8],
        sa: usize,
        b: &[u8],
        sb: usize,
        w: usize,
        h: usize,
        fy: u32,
        fx: u32,
    ) -> u32 {
        crate::dsp::sad::sad_bilinear(a, sa, b, sb, w, h, fy, fx)
    }
}

impl EncPixel for u16 {
    #[inline(always)]
    fn sad(a: &[u16], sa: usize, b: &[u16], sb: usize, w: usize, h: usize) -> u32 {
        crate::dsp::sad::high::sad(a, sa, b, sb, w, h)
    }
    #[inline(always)]
    fn sad4(a: &[u16], sa: usize, b: [&[u16]; 4], sb: usize, w: usize, h: usize) -> [u32; 4] {
        crate::dsp::sad::high::sad4(a, sa, b, sb, w, h)
    }
    #[inline(always)]
    fn sse(a: &[u16], sa: usize, b: &[u16], sb: usize, w: usize, h: usize) -> u64 {
        crate::dsp::sad::high::sse(a, sa, b, sb, w, h)
    }
    #[inline(always)]
    fn sad_bilinear(
        a: &[u16],
        sa: usize,
        b: &[u16],
        sb: usize,
        w: usize,
        h: usize,
        fy: u32,
        fx: u32,
    ) -> u32 {
        crate::dsp::sad::high::sad_bilinear(a, sa, b, sb, w, h, fy, fx)
    }
}

/// The frame-level encoder of pictures of `P` samples (`u8`: 8-bit,
/// `u16`: 10- or 12-bit).
pub struct FrameEncoder<P: EncPixel = u8> {
    width: u32,
    height: u32,
    /// Bits a sample: 8, 10 or 12.
    bd: u8,
    dec: Decoder,
    /// The source, padded to whole superblocks by repeating its edges.
    src: [Vec<P>; 3],
    strides: [usize; 3],
    /// Tile column threads' buffers.
    work: Vec<TileWork<P>>,
    /// Tile columns: at most this many threads (log2 of the count).
    log2_tiles: u8,
    /// Tile rows (log2 of the count, 0 to 2).
    log2_tile_rows: u8,
    /// Chroma subsampling across and down: (1, 1) 4:2:0 (profile 0),
    /// (0, 0) 4:4:4 (profile 1).
    ss: (usize, usize),
    /// The fast search's vectors of the last frame, by 8×8 block.
    pre_seeds: Vec<Mv>,
    /// The fast path's searches made row by row (see [`Lazy`]): each tile
    /// column's seeds and searches, kept from frame to frame.
    pre_bufs: Vec<(Vec<Mv>, Vec<tile::Pre>)>,
    /// The frame's searches, made ahead on threads.
    pre_frame: Vec<tile::Pre>,
    /// The segmentation of the frame being coded, with adaptive
    /// quantization.
    aq: Option<aq::Plan>,
    /// Each reference's picture in the last inter frame, the share of its
    /// area that frame predicted from it (in 1/1024), and how many frames
    /// in a row it has been left out of the search.
    ref_use: [(usize, u32, u8); 3],
    /// What every frame signals.
    stream: Stream,
    /// The threads the encoder was made for.
    threads: usize,
    /// Threads coding each tile column's rows (see [`tile::Lanes`]); 0:
    /// from the threads.
    row_lanes: usize,
    /// The next frame's whole-picture motion from LAST, if far.
    global: Option<Mv>,
}

/// What one tile column's encoder keeps from frame to frame.
struct TileWork<P: EncPixel> {
    /// Its own copy of the picture; only its columns are meaningful.
    pix: [Vec<P>; 3],
    seeds: Vec<Mv>,
    state: State,
    scratch: Scratch<P>,
    /// The tile's symbols, written once the frame's probabilities are.
    out: tile::Out,
    /// How many of them each superblock row ends at.
    row_ends: Vec<usize>,
    bufs: tile::TileBufs<P>,
}

impl<P: EncPixel> TileWork<P> {
    fn new(dims: &Dims) -> Self {
        Self {
            pix: std::array::from_fn(|p| vec![P::default(); dims.stride[p] * dims.rows[p]]),
            seeds: Vec::new(),
            state: State::empty(),
            scratch: Scratch::new(),
            out: Vec::new(),
            row_ends: Vec::new(),
            bufs: tile::TileBufs::default(),
        }
    }
}

/// The first 8×8 column of tile `i` of `1 << log2` (libvpx
/// `get_tile_offsets`).
fn tile_edge(i: usize, mi_cols: usize, log2: u8) -> usize {
    let sb_cols = mi_cols.div_ceil(8);
    (((i * sb_cols) >> log2) * 8).min(mi_cols)
}

impl FrameEncoder<u8> {
    /// An encoder of `width`×`height` pictures on up to `threads` tile
    /// columns (rounded down to a power of two the width allows).
    #[cfg(test)]
    pub fn new(width: u32, height: u32, threads: usize) -> Self {
        Self::with_depth(width, height, threads, (1, 1), 8)
    }
}

impl<P: EncPixel> FrameEncoder<P> {
    /// An encoder of `width`×`height` pictures of `bd` bits a sample (8
    /// for `u8`; 10 or 12 for `u16`), chroma subsampled `ss` across and
    /// down ((1, 1) for 4:2:0, (0, 0) for 4:4:4), on up to `threads` tile
    /// columns (rounded down to a power of two the width allows).
    pub fn with_depth(width: u32, height: u32, threads: usize, ss: (usize, usize), bd: u8) -> Self {
        let dims = Dims::new(width as usize, height as usize, ss.0, ss.1, bd);
        let src = std::array::from_fn(|p| vec![P::default(); dims.stride[p] * dims.rows[p]]);
        Self {
            width,
            height,
            bd,
            dec: Decoder::new(),
            src,
            strides: dims.stride,
            work: Vec::new(),
            log2_tiles: threads.max(1).ilog2().min(6) as u8,
            log2_tile_rows: 0,
            ss,
            pre_seeds: Vec::new(),
            pre_bufs: Vec::new(),
            pre_frame: Vec::new(),
            aq: None,
            ref_use: [(0, 1024, 0); 3],
            stream: Stream::default(),
            threads: threads.max(1),
            row_lanes: 0,
            global: None,
        }
    }

    /// Codes `1 << log2` tile rows (0 to 2; libvpx's `tile-rows`). Tile
    /// rows only restart the arithmetic coder, so they cost a little
    /// size and let a decoder start each row's coder at once.
    pub fn set_tile_rows(&mut self, log2: u8) {
        self.log2_tile_rows = log2.min(2);
    }

    /// Codes `1 << log2` tile columns (as the width allows; libvpx's
    /// `tile-columns`), each on its own thread, instead of one per
    /// thread.
    pub fn set_tile_columns(&mut self, log2: u8) {
        self.log2_tiles = log2.min(6);
    }

    /// What every frame signals from now on.
    pub fn set_stream(&mut self, stream: Stream) {
        self.stream = stream;
    }

    /// The next frame's picture moved by `(x, y)` whole samples from
    /// LAST's, as a whole (a jump of the camera that the searches, which
    /// reach a few dozen samples, would take for a cut): every block's
    /// search on LAST starts there too.
    pub fn set_global(&mut self, (x, y): (isize, isize)) {
        self.global = Some(Mv {
            row: (y * 8).clamp(-8 * 1023, 8 * 1023) as i16,
            col: (x * 8).clamp(-8 * 1023, 8 * 1023) as i16,
        });
    }

    /// Codes each tile column's superblock rows on `n` threads at once
    /// (0: two or more when the encoder has threads; 1: one after
    /// another). The frames are the same either way.
    #[cfg(test)]
    pub fn set_row_lanes(&mut self, n: usize) {
        self.row_lanes = n;
    }

    /// The picture the last frame reconstructed (what a decoder shows).
    #[cfg(test)]
    pub fn last(&self) -> Option<&Arc<Pic<P>>> {
        self.dec.ref_pic(0)
    }

    /// Copies a source picture in, repeating its last column and row out
    /// to whole superblocks.
    pub fn load(&mut self, planes: [(&[P], usize); 3]) {
        let dims = Dims::new(
            self.width as usize,
            self.height as usize,
            self.ss.0,
            self.ss.1,
            self.bd,
        );
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
        let key = fp.key || self.dec.ref_pic::<P>(0).is_none();
        let lossless = fp.qindex == 0;
        self.aq = (fp.aq && !lossless).then(|| {
            let (w, h) = (self.width as usize, self.height as usize);
            aq::plan(
                &self.src[0],
                self.strides[0],
                (w, h),
                (w.div_ceil(8), h.div_ceil(8)),
                fp.qindex,
            )
        });
        // The tiles are coded against the probabilities as they stand; their
        // symbols are kept, the coefficient probabilities that pay for
        // their update are updated, and the symbols written with them.
        let head = self.headers(key, fp, None, None)?;
        let parsed = header::parse(self.dec.persist(), &head)?;
        let (counts, mvs, segmap, pic, coded) = self.tiles(&parsed, fp, key)?;
        let Written {
            old,
            new,
            old_nc,
            new_nc,
            tiles,
        } = coded;
        let head = self.headers(key, fp, Some((&old, &new)), Some((&old_nc, &new_nc)))?;
        let mut data = head;
        // Tiles go row by row; every one but the last carries its size.
        let cols = tiles.into_iter().collect::<Result<Vec<_>, _>>()?;
        let rows = cols.first().map_or(0, Vec::len);
        for tr in 0..rows {
            for (tc, col) in cols.iter().enumerate() {
                let d = &col[tr];
                if tr + 1 < rows || tc + 1 < cols.len() {
                    data.extend_from_slice(&(d.len() as u32).to_be_bytes());
                }
                data.extend_from_slice(d);
            }
        }
        let parsed = header::parse(self.dec.persist(), &data)?;
        let _shown = self.dec.commit(&parsed, &counts, mvs, segmap, pic);
        Ok(Coded { data, key })
    }

    /// What the headers of a frame code (and so which probabilities it
    /// can update).
    fn coded(key: bool, fp: FrameParams) -> probs::Coded {
        // The sign bias goes with compound prediction: alone it only makes
        // the vectors of neighbours on the other side worse candidates.
        let future = !key && !fp.hidden && fp.altref_future && fp.speed <= COMPOUND_SPEED;
        probs::Coded {
            tx_select: fp.qindex != 0,
            inter: !key,
            switchable: !key && fp.speed <= SWITCHABLE_SPEED,
            allow_hp: true,
            compound_allowed: future,
            compound: future.then_some(true),
        }
    }

    /// The uncompressed and compressed headers.
    fn headers(
        &self,
        key: bool,
        fp: FrameParams,
        coef: Option<(&[u8; 1728], &[u8; 1728])>,
        nc: Option<(&probs::Flat, &probs::Flat)>,
    ) -> Result<Vec<u8>, CodecError> {
        let qindex = fp.qindex;
        let hidden = fp.hidden && !key;
        let lossless = qindex == 0;
        let switchable = !key && fp.speed <= SWITCHABLE_SPEED;
        let compressed = compressed_header(lossless, &Self::coded(key, fp), coef, nc)?;
        let mut b = BitWriter::new();
        b.write(2, 2); // frame marker
        // Profile 0 for 8-bit 4:2:0, 1 for 8-bit 4:2:2, 4:4:0 and 4:4:4;
        // 2 and 3 for the same at 10 and 12 bits. Its low bit goes first,
        // and profile 3 has a reserved bit after it.
        let profile1 = self.ss != (1, 1);
        let profile = u32::from(profile1) | if self.bd > 8 { 2 } else { 0 };
        b.write(((profile & 1) << 1) | (profile >> 1), 2);
        if profile == 3 {
            b.write_bit(false);
        }
        b.write_bit(false); // show_existing_frame
        b.write_bit(!key); // frame_type
        b.write_bit(!hidden); // show_frame
        let st = self.stream;
        b.write_bit(st.error_resilient);
        let render = |b: &mut BitWriter| match st.render_size {
            Some((w, h)) => {
                b.write_bit(true);
                b.write(w - 1, 16);
                b.write(h - 1, 16);
            }
            None => b.write_bit(false), // render size = frame size
        };
        if key {
            b.write(0x49, 8);
            b.write(0x83, 8);
            b.write(0x42, 8);
            if self.bd > 8 {
                b.write_bit(self.bd == 12); // ten_or_twelve_bit
            }
            b.write(u32::from(st.color_space), 3);
            if st.color_space != 7 {
                b.write_bit(st.full_range);
                if profile1 {
                    b.write_bit(self.ss.0 == 1);
                    b.write_bit(self.ss.1 == 1);
                    b.write_bit(false); // reserved
                }
            } else if profile1 {
                b.write_bit(false); // sRGB: 4:4:4, reserved
            }
            b.write(self.width - 1, 16);
            b.write(self.height - 1, 16);
            render(&mut b);
        } else {
            if hidden {
                b.write_bit(false); // intra_only
            }
            if !st.error_resilient {
                b.write(0, 2); // reset_frame_context
            }
            // LAST, and GOLDEN on a golden frame; an alt-ref only ALTREF.
            let refresh = if hidden {
                0x04
            } else if fp.golden {
                0x03
            } else {
                0x01
            } | u32::from(fp.also_refresh);
            b.write(refresh, 8);
            let future = Self::coded(key, fp).compound_allowed;
            for (i, slot) in [0, u32::from(fp.golden_slot), 2].into_iter().enumerate() {
                b.write(slot, 3); // LAST 0, GOLDEN (1), ALTREF 2
                b.write_bit(i == 2 && future); // sign bias
            }
            b.write_bit(true); // size from the first reference
            render(&mut b);
            b.write_bit(true); // allow_high_precision_mv
            if switchable {
                b.write_bit(true); // a filter per block
            } else {
                b.write_bit(false);
                b.write(1, 2); // EIGHTTAP
            }
        }
        if !st.error_resilient {
            b.write_bit(true); // refresh_frame_context
            b.write_bit(st.frame_parallel);
        }
        b.write(0, 2); // frame_context_idx
        // Loop filter: libvpx's level from the quantizer.
        let level = if lossless {
            0
        } else {
            filter_level(qindex, key, self.bd)
        };
        b.write(u32::from(level), 6);
        b.write(u32::from(st.sharpness.min(7)), 3);
        b.write_bit(true); // mode_ref_delta_enabled (the defaults)
        b.write_bit(false); // no update
        b.write(u32::from(qindex), 8);
        b.write(0, 3); // no dc / uv deltas
        match &self.aq {
            Some(plan) => {
                b.write_bit(true); // segmentation
                b.write_bit(true); // update_map
                for p in plan.tree_probs {
                    b.write_bit(p != 255);
                    if p != 255 {
                        b.write(u32::from(p), 8);
                    }
                }
                b.write_bit(false); // no temporal prediction
                b.write_bit(true); // update_data
                b.write_bit(false); // deltas
                for d in plan.deltas {
                    // Feature 0, the quantizer; no others.
                    b.write_bit(d != 0);
                    if d != 0 {
                        b.write(u32::from(d.unsigned_abs()), 8);
                        b.write_bit(d < 0);
                    }
                    for _ in 1..4 {
                        b.write_bit(false);
                    }
                }
            }
            None => b.write_bit(false), // no segmentation
        }
        let (min_log2, max_log2) = header::tile_limits(geom::mi_count(self.width));
        let log2 = u32::from(self.log2_tiles).clamp(min_log2, max_log2);
        for _ in min_log2..log2 {
            b.write_bit(true);
        }
        if log2 < max_log2 {
            b.write_bit(false);
        }
        // Tile rows: each starts its own coder and nothing else.
        b.write_bit(self.log2_tile_rows > 0);
        if self.log2_tile_rows > 0 {
            b.write_bit(self.log2_tile_rows > 1);
        }
        let len = u32::try_from(compressed.len()).unwrap_or(u32::MAX);
        if len > u32::from(u16::MAX) {
            return Err(CodecError::Limit {
                what: "compressed header",
            });
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
        key: bool,
    ) -> Result<
        (
            prob::Counts,
            Vec<frame::MvRef>,
            Vec<u8>,
            Arc<Pic<P>>,
            Written,
        ),
        CodecError,
    > {
        let h = &parsed.header;
        let dims = Dims::new(
            h.width as usize,
            h.height as usize,
            h.ss_x as usize,
            h.ss_y as usize,
            self.bd,
        );
        let mi_cols = dims.width.div_ceil(8);
        let mi_rows = dims.height.div_ceil(8);
        let sb_cols = mi_cols.div_ceil(8);
        let sb_rows = mi_rows.div_ceil(8);
        let slot = |i: usize| {
            self.dec
                .ref_pic::<P>(usize::from(h.ref_index[i]))
                .map(|p| &**p)
        };
        let refs = if h.is_intra() {
            [None, None, None]
        } else {
            [slot(0), slot(1), slot(2)]
        };
        // A reference worth searching holds a picture no earlier one does.
        let same = |a: Option<&Pic<P>>, b: Option<&Pic<P>>| match (a, b) {
            (Some(a), Some(b)) => std::ptr::eq(a, b),
            _ => false,
        };
        let mut refs_live = [
            refs[0].is_some(),
            refs[1].is_some() && !same(refs[1], refs[0]),
            refs[2].is_some() && !same(refs[2], refs[0]) && !same(refs[2], refs[1]),
        ];
        // From REF_PRUNE_SPEED, GOLDEN and ALTREF are left out of the
        // search while the same picture predicted under 1% of the last
        // inter frame that searched it, but tried again every fourth frame.
        let ids = refs.map(|r| r.map_or(0, |p| std::ptr::from_ref(p) as usize));
        let searched_before = refs_live;
        if !h.is_intra() && fp.speed >= REF_PRUNE_SPEED && !fp.altref_future {
            for r in 1..3 {
                let (id, share, out) = &mut self.ref_use[r];
                if refs_live[r] && *id == ids[r] && *share < 10 && *out < 3 {
                    refs_live[r] = false;
                    *out += 1;
                }
            }
        }
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
            ss_x: h.ss_x as usize,
            ss_y: h.ss_y as usize,
            bd: u32::from(self.bd),
        };
        let filter_on = h.filter_level != 0;
        // Chroma subsampled one way only filters from a grid of the
        // blocks' sizes and transforms, as the decoder's.
        let needs_grid = filter_on && h.ss_x != h.ss_y;
        let mut writer = frame::writer(&dims, self.dec.spare::<P>());
        let prev_mvs = if h.use_prev_mvs {
            self.dec.prev_mvs()
        } else {
            None
        };
        let ac = i32::from(h.y_dq[0][1]);
        let tok = tile::token_costs(&parsed.fc);
        let shared = Shared {
            h,
            fc: &parsed.fc,
            prev_mvs,
            levels: &levels,
            mi_cols,
            mi_rows,
            intra_frame: h.is_intra(),
            part_probs: if h.is_intra() {
                prob::kf_partition()
            } else {
                parsed.fc.partition
            },
            filter_on,
            lambda: sad_per_bit(ac, self.bd),
            rd_lambda: rd_lambda(i32::from(h.y_dq[0][0])),
            src: Source {
                planes: [&self.src[0], &self.src[1], &self.src[2]],
                strides: self.strides,
            },
            dq: std::array::from_fn(|g| {
                [
                    [i32::from(h.y_dq[g][0]), i32::from(h.y_dq[g][1])],
                    [i32::from(h.uv_dq[g][0]), i32::from(h.uv_dq[g][1])],
                ]
            }),
            seg_map: self.aq.as_ref().map_or(&[][..], |p| &p.map[..]),
            speed: fp.speed,
            refs_live,
            switchable: h.interp == geom::SWITCHABLE,
            prepass: None,
            tok: &tok,
            second_pass: fp.second_pass,
            noise: fp.noise,
            static_thresh: self.stream.static_thresh,
            global: self.global.take(),
        };
        // Each tile column on its own thread, into its own copy of the
        // picture (intra prediction reads nothing across tile columns).
        let ntiles = 1usize << h.log2_tile_cols;
        let edges: Vec<usize> = (0..=ntiles)
            .map(|i| tile_edge(i, mi_cols, h.log2_tile_cols))
            .collect();
        // Each tile column's superblock rows on `lanes` threads at once,
        // in a wavefront (see [`tile::Lanes`]): tile columns alone keep
        // threads waiting for the busiest column.
        let lanes = if self.row_lanes > 0 {
            self.row_lanes
        } else if self.threads > 1 && fp.speed >= ONE_LANE_SPEED {
            self.threads.div_ceil(ntiles)
        } else if self.threads > 1 {
            (2 * self.threads).div_ceil(ntiles).clamp(2, 8)
        } else {
            1
        }
        .min(sb_rows.max(1));
        // Fast inter frames: the motion search of every superblock first,
        // its rows shared out evenly among the threads (tile columns carry
        // uneven work), each thread with its own copy of the seed vectors.
        let mut pre_buf: Vec<tile::Pre> = std::mem::take(&mut self.pre_frame);
        let rp = ctx.refs[0]
            .as_ref()
            .map(|r| (r.buf.plane(0), (r.dims.w[0] as i32, r.dims.h[0] as i32)));
        // On one thread, each superblock row's search just before the row
        // is coded, while its samples are in the cache: the same search, in
        // the same order. At FUSED_SPEED, with one lane a tile column,
        // each tile column's thread searches its own columns so, with its
        // own copy of the seeds (those across its left edge are the last
        // frame's), and no thread waits for the frame's search.
        let single = self.log2_tiles == 0 && ntiles == 1 && lanes == 1 && self.threads <= 1;
        let lazy = fp.speed >= 7
            && !h.is_intra()
            && rp.is_some()
            && (single || (fp.speed == FUSED_SPEED && lanes == 1));
        if lazy {
            self.pre_seeds.resize(mi_cols * mi_rows, Mv::ZERO);
        }
        let shared = if fp.speed >= 7 && !h.is_intra() && !lazy {
            match rp {
                Some(rp) => {
                    let threads = (1usize << self.log2_tiles).max(1);
                    self.pre_seeds.resize(mi_cols * mi_rows, Mv::ZERO);
                    // (Each thread's seeds and searches, kept from frame to
                    // frame.)
                    let mut bufs = std::mem::take(&mut self.pre_bufs);
                    bufs.resize_with(threads, Default::default);
                    for (seeds, out) in &mut bufs {
                        seeds.clone_from(&self.pre_seeds);
                        out.clear();
                    }
                    let sh = &shared;
                    let done: Vec<bool> = std::thread::scope(|sc| {
                        let hs: Vec<_> = bufs
                            .iter_mut()
                            .enumerate()
                            .map(|(t, (seeds, out))| {
                                sc.spawn(move || {
                                    let mut r = t;
                                    while r < sb_rows {
                                        for c in 0..sb_cols {
                                            out.push(tile::prepass_sb(sh, rp, seeds, r * 8, c * 8));
                                        }
                                        r += threads;
                                    }
                                })
                            })
                            .collect();
                        hs.into_iter().map(|h| h.join().is_ok()).collect()
                    });
                    let mut pre = std::mem::take(&mut pre_buf);
                    pre.clear();
                    pre.resize(sb_rows * sb_cols, tile::Pre::default());
                    for (t, (seeds, out)) in bufs.iter().enumerate() {
                        if !done[t] {
                            continue;
                        }
                        let mut it = out.iter().copied();
                        let mut r = t;
                        while r < sb_rows {
                            for c in 0..sb_cols {
                                if let Some(p) = it.next() {
                                    pre[r * sb_cols + c] = p;
                                }
                            }
                            let (lo, hi) = (r * 8 * mi_cols, ((r * 8 + 8).min(mi_rows)) * mi_cols);
                            self.pre_seeds[lo..hi].copy_from_slice(&seeds[lo..hi]);
                            r += threads;
                        }
                    }
                    self.pre_bufs = bufs;
                    pre_buf = pre;
                    Shared {
                        prepass: Some(&pre_buf),
                        ..shared
                    }
                }
                None => shared,
            }
        } else {
            shared
        };
        while self.work.len() < ntiles * lanes {
            self.work.push(TileWork::new(&dims));
        }
        let plane_len: [usize; 3] = std::array::from_fn(|p| dims.stride[p] * dims.rows[p]);
        // With one lane a tile column on threads, the loop filter follows
        // the tiles row by row ([`lf_rows`]) instead of filtering the frame
        // after them: on four threads at speed 10 it was a tenth of the
        // time. (Not with a grid: 4:2:2 and 4:4:0 filter chroma from it.)
        let piped = lanes == 1 && ntiles > 1 && !needs_grid;
        let run = |t: &mut TileWork<P>,
                   col0: usize,
                   col1: usize,
                   lane: usize,
                   sync: Option<&tile::Lanes<P>>,
                   pipe: Option<&[mpsc::Sender<Band<P>>]>,
                   mut lazy: Option<Lazy<'_, '_, P>>|
         -> Result<(), CodecError> {
            for (p, buf) in t.pix.iter_mut().enumerate() {
                buf.resize(plane_len[p], P::default());
            }
            let tsb = (col1 - col0).div_ceil(8);
            t.state
                .reset(col0, col1 - col0, mi_rows, sb_rows * tsb, needs_grid);
            t.out.clear();
            t.row_ends.clear();
            let [y, u, v] = &mut t.pix;
            let mut views = [
                View {
                    data: &mut y[..],
                    base: 0,
                    stride: dims.stride[0],
                },
                View {
                    data: &mut u[..],
                    base: 0,
                    stride: dims.stride[1],
                },
                View {
                    data: &mut v[..],
                    base: 0,
                    stride: dims.stride[2],
                },
            ];
            let mut seeds = std::mem::take(&mut t.seeds);
            for r in (lane..sb_rows).step_by(lanes) {
                let row_shared;
                let sh = match lazy.as_mut() {
                    Some(l) => {
                        for c in col0 / 8..col1.div_ceil(8) {
                            l.pre[r * sb_cols + c] =
                                tile::prepass_sb(&shared, l.rp, l.seeds, r * 8, c * 8);
                        }
                        row_shared = Shared {
                            prepass: Some(&l.pre[..]),
                            ..shared
                        };
                        &row_shared
                    }
                    None => &shared,
                };
                let mut rec = Recon::new(&ctx, &mut views, &mut t.scratch);
                let mut tile = Tile::new(
                    sh,
                    &mut t.state,
                    &mut t.out,
                    &mut rec,
                    (col0, col1),
                    seeds,
                    std::mem::take(&mut t.bufs),
                    sync,
                );
                tile.sb_row(r * 8);
                (seeds, t.bufs) = tile.finish();
                t.row_ends.push(t.out.len());
                if let Some(pipe) = pipe {
                    // The row's samples and filter masks, for the loop
                    // filter (a receiver gone has failed: the frame does).
                    for (p, tx) in pipe.iter().enumerate() {
                        let (sx, sy) = if p == 0 {
                            (0, 0)
                        } else {
                            (dims.ss_x, dims.ss_y)
                        };
                        let stride = dims.stride[p];
                        let (x0, x1) = ((col0 * 8) >> sx, ((col1 * 8) >> sx).min(stride));
                        let y0 = (r * 64) >> sy;
                        let y1 = if r + 1 == sb_rows {
                            dims.rows[p]
                        } else {
                            (((r + 1) * 64) >> sy).min(dims.rows[p])
                        };
                        let mut pix = Vec::with_capacity((x1 - x0) * (y1 - y0));
                        for y in y0..y1 {
                            pix.extend_from_slice(&views[p].data[y * stride + x0..y * stride + x1]);
                        }
                        let band = Band {
                            cols: (col0, col1),
                            row: r,
                            pix,
                            masks: t.state.masks[r * tsb..(r + 1) * tsb].to_vec(),
                        };
                        if tx.send(band).is_err() {
                            return Err(CodecError::Malformed {
                                at_bit: 0,
                                what: "vp9 encoder loop filter",
                            });
                        }
                    }
                }
            }
            t.seeds = seeds;
            Ok(())
        };
        let work = &mut self.work[..ntiles * lanes];
        if ntiles * lanes == 1 {
            // (Each entry is written before it is read.)
            let mut bufs = std::mem::take(&mut self.pre_bufs);
            bufs.resize_with(1, Default::default);
            let pre = &mut bufs[0].1;
            let lazy = match rp {
                Some(rp) if lazy => {
                    pre.resize(sb_rows * sb_cols, tile::Pre::default());
                    Some(Lazy {
                        rp,
                        seeds: &mut self.pre_seeds,
                        pre,
                    })
                }
                _ => None,
            };
            let r = run(&mut work[0], 0, mi_cols, 0, None, None, lazy);
            self.pre_bufs = bufs;
            r?;
        } else {
            let syncs: Vec<tile::Lanes<P>> = (0..ntiles)
                .map(|i| tile::Lanes::new(sb_rows, (edges[i + 1] - edges[i]).div_ceil(8)))
                .collect();
            let writer = &mut writer;
            // (Fused: each tile column's seeds and searches.)
            let mut fused = std::mem::take(&mut self.pre_bufs);
            if lazy {
                fused.resize_with(ntiles, Default::default);
                for (seeds, pre) in &mut fused {
                    seeds.clone_from(&self.pre_seeds);
                    pre.resize(sb_rows * sb_cols, tile::Pre::default());
                }
            } else {
                fused.clear();
            }
            let results: Vec<Result<(), CodecError>> = std::thread::scope(|sc| {
                // Piped: each plane's rows go into the picture and through
                // the loop filter as every tile column finishes them.
                let mut lf_threads = Vec::new();
                let mut txs = Vec::new();
                if piped {
                    for (p, v) in writer.tails().map(View::of).into_iter().enumerate() {
                        let (tx, rx) = mpsc::channel();
                        txs.push(tx);
                        let (l, g) = (&levels, &geo);
                        lf_threads.push(
                            sc.spawn(move || lf_rows(v, p, rx, (ntiles, sb_rows), filter_on, l, g)),
                        );
                    }
                }
                let mut fused_it = fused.iter_mut();
                let handles: Vec<_> = work
                    .iter_mut()
                    .enumerate()
                    .map(|(k, t)| {
                        let lazy = match (rp, fused_it.next()) {
                            (Some(rp), Some((seeds, pre))) => Some(Lazy { rp, seeds, pre }),
                            _ => None,
                        };
                        let (i, lane) = (k / lanes, k % lanes);
                        let (c0, c1) = (edges[i], edges[i + 1]);
                        let sync = (lanes > 1).then(|| &syncs[i]);
                        let run = &run;
                        let pipe = piped.then(|| txs.clone());
                        sc.spawn(move || {
                            // (A lane that stops lets the others stop
                            // waiting for it.)
                            let _guard = Release(sync);
                            let r = run(t, c0, c1, lane, sync, pipe.as_deref(), lazy);
                            if r.is_err()
                                && let Some(l) = sync
                            {
                                l.fail();
                            }
                            r
                        })
                    })
                    .collect();
                drop(txs);
                let thread_err = || {
                    Err(CodecError::Malformed {
                        at_bit: 0,
                        what: "vp9 encoder thread",
                    })
                };
                let mut out: Vec<Result<(), CodecError>> = handles
                    .into_iter()
                    .map(|h| h.join().unwrap_or_else(|_| thread_err()))
                    .collect();
                for h in lf_threads {
                    if h.join().unwrap_or(false) {
                        continue;
                    }
                    out.push(thread_err());
                }
                out
            });
            for (i, (seeds, _)) in fused.iter().enumerate() {
                let (c0, c1) = (edges[i], edges[i + 1]);
                for r in 0..mi_rows {
                    let at = r * mi_cols;
                    self.pre_seeds[at + c0..at + c1].copy_from_slice(&seeds[at + c0..at + c1]);
                }
            }
            self.pre_bufs = fused;
            for r in results {
                r?;
            }
            if lanes > 1 {
                for (i, chunk) in work.chunks_mut(lanes).enumerate() {
                    merge_lanes(chunk, (edges[i], edges[i + 1]), &dims, sb_rows, needs_grid);
                }
            }
        }
        let work: Vec<&TileWork<P>> = self.work[..ntiles * lanes].iter().step_by(lanes).collect();
        // The tiles' state into the frame's, and the probability updates
        // it pays for. Then, at once: each plane's columns into the
        // picture and its loop filter, and each tile's symbols into its
        // own coder.
        let buf = Arc::clone(writer.shared());
        let mut masks = vec![lpf::Mask::default(); sb_rows * sb_cols];
        let mut mvs = vec![frame::MvRef::default(); mi_cols * mi_rows];
        let mut grid = if needs_grid {
            vec![LfMi::default(); mi_cols * mi_rows]
        } else {
            Vec::new()
        };
        let mut counts = prob::Counts::new();
        for (i, t) in work.iter().enumerate() {
            let (c0, c1) = (edges[i], edges[i + 1]);
            let st = &t.state;
            let w = c1 - c0;
            let tsb = w.div_ceil(8);
            for r in 0..sb_rows {
                masks[r * sb_cols + c0 / 8..r * sb_cols + c0 / 8 + tsb]
                    .copy_from_slice(&st.masks[r * tsb..r * tsb + tsb]);
            }
            for r in 0..mi_rows {
                mvs[r * mi_cols + c0..r * mi_cols + c1].copy_from_slice(&st.mvs[r * w..r * w + w]);
                if needs_grid {
                    grid[r * mi_cols + c0..r * mi_cols + c1]
                        .copy_from_slice(&st.grid[r * w..r * w + w]);
                }
            }
            counts.add(&st.counts);
        }
        let lossless = h.lossless;
        let old = flat_coef(&parsed.fc);
        let new = coef_updates(&old, &counts, if lossless { 0 } else { 3 });
        let old_nc = probs::flatten(&parsed.fc);
        let new_nc = probs::updates(&old_nc, &counts, &Self::coded(key, fp));
        // A tile column's symbols, one coder for each tile row.
        let log2_rows = h.log2_tile_rows;
        let probs = tile::prob_table(&new, &new_nc);
        let write = |t: &TileWork<P>| -> Result<Vec<Vec<u8>>, CodecError> {
            (0..1usize << log2_rows)
                .map(|tr| {
                    let symbols = |sb_row: usize| {
                        sb_row
                            .checked_sub(1)
                            .and_then(|r| t.row_ends.get(r))
                            .copied()
                            .unwrap_or(0)
                    };
                    // In superblock rows, as libvpx's `get_tile_offsets`.
                    let (r0, r1) = (
                        (tr * sb_rows) >> log2_rows,
                        ((tr + 1) * sb_rows) >> log2_rows,
                    );
                    let mut w = BoolWriter::new();
                    w.write_all(
                        t.out[symbols(r0)..symbols(r1)]
                            .iter()
                            .map(|&(bit, id)| (u32::from(bit), u32::from(probs[usize::from(id)]))),
                    );
                    w.finish()
                })
                .collect()
        };
        let work = &work[..];
        let (m, l, g, ed, gr) = (&masks, &levels, &geo, &edges, &grid);
        let tiles = std::thread::scope(|sc| {
            for (p, mut v) in writer.tails().map(View::of).into_iter().enumerate() {
                if piped {
                    // (Filtered already.)
                    break;
                }
                sc.spawn(move || {
                    let ss = if p == 0 { 0 } else { dims.ss_x };
                    // (Rows are whole: tile columns split only across.)
                    let stride = v.stride;
                    for (i, t) in work.iter().enumerate() {
                        let (x0, x1) = ((ed[i] * 8) >> ss, ((ed[i + 1] * 8) >> ss).min(stride));
                        for row in 0..dims.rows[p] {
                            v.data[row * stride + x0..row * stride + x1]
                                .copy_from_slice(&t.pix[p][row * stride + x0..row * stride + x1]);
                        }
                    }
                    if filter_on {
                        for r in 0..sb_rows {
                            lf::sb_row_plane(&mut v, p, m, gr, l, g, r);
                        }
                    }
                });
            }
            let write = &write;
            let hs: Vec<_> = work.iter().map(|&t| sc.spawn(move || write(t))).collect();
            hs.into_iter()
                .map(|h| {
                    h.join().unwrap_or(Err(CodecError::Malformed {
                        at_bit: 0,
                        what: "vp9 encoder thread",
                    }))
                })
                .collect::<Vec<_>>()
        });
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
            made_cv: std::sync::Condvar::new(),
            made: AtomicBool::new(false),
        });
        let mut segmap = vec![0u8; mi_cols * mi_rows];
        if self.aq.is_some() {
            for (i, t) in work.iter().enumerate() {
                let (c0, c1) = (edges[i], edges[i + 1]);
                let (st, w) = (&t.state, c1 - c0);
                for r in 0..mi_rows {
                    segmap[r * mi_cols + c0..r * mi_cols + c1]
                        .copy_from_slice(&st.segmap[r * w..r * w + w]);
                }
            }
        }
        if !h.is_intra() {
            let mut area = [0u32; 3];
            for m in &mvs {
                for &r in &m.refs {
                    if (1..=3).contains(&r) {
                        area[usize::from(r) - 1] += 1;
                    }
                }
            }
            let total = (mvs.len() as u32).max(1);
            for r in 0..3 {
                let u = &mut self.ref_use[r];
                if refs_live[r] {
                    // Searched: what it won is known.
                    *u = (ids[r], area[r] * 1024 / total, 0);
                } else if !searched_before[r] {
                    // The same picture as another reference: unknown.
                    *u = (ids[r], 1024, 0);
                }
                // Left out: its share stands, and its count goes on.
            }
        }
        self.pre_frame = pre_buf;
        Ok((
            counts,
            mvs,
            segmap,
            pic,
            Written {
                old,
                new,
                old_nc,
                new_nc,
                tiles,
            },
        ))
    }
}

/// The one-thread search of each superblock row before it is coded (see
/// [`FrameEncoder::tiles`]): LAST's luma, the frame's seeds, and where
/// the rows' searches go.
#[allow(clippy::type_complexity)]
struct Lazy<'a, 'b, P> {
    rp: ((&'a [P], usize), (i32, i32)),
    seeds: &'b mut [Mv],
    pre: &'b mut [tile::Pre],
}

/// A superblock row of one tile column, coded: each plane's samples (its
/// columns, row by row) and the row's filter masks.
struct Band<P> {
    /// Its tile column's first and last mode info columns.
    cols: (usize, usize),
    row: usize,
    pix: Vec<P>,
    masks: Vec<lpf::Mask>,
}

/// Puts the rows of plane `p` that the tile columns send into the picture
/// `v`, and filters each superblock row once every tile column has sent
/// it: a row's filter reads and writes no sample under it, and the row
/// above is filtered first, so the picture is the one filtering the whole
/// frame after coding it makes. Whether every row came.
fn lf_rows<P: EncPixel>(
    mut v: View<'_, P>,
    p: usize,
    rx: mpsc::Receiver<Band<P>>,
    (ntiles, sb_rows): (usize, usize),
    filter_on: bool,
    levels: &lpf::Levels,
    g: &lf::Geo,
) -> bool {
    let (sx, sy) = if p == 0 { (0, 0) } else { (g.ss_x, g.ss_y) };
    let stride = v.stride;
    let mut masks = vec![lpf::Mask::default(); sb_rows * g.sb_cols];
    let mut have = vec![0usize; sb_rows];
    let mut next = 0;
    for b in rx {
        let (x0, x1) = ((b.cols.0 * 8) >> sx, ((b.cols.1 * 8) >> sx).min(stride));
        let w = x1 - x0;
        let y0 = (b.row * 64) >> sy;
        for (k, line) in b.pix.chunks_exact(w.max(1)).enumerate() {
            let at = (y0 + k) * stride - v.base + x0;
            v.data[at..at + w].copy_from_slice(line);
        }
        let at = b.row * g.sb_cols + b.cols.0 / 8;
        masks[at..at + b.masks.len()].copy_from_slice(&b.masks);
        have[b.row] += 1;
        while next < sb_rows && have[next] == ntiles {
            if filter_on {
                lf::sb_row_plane(&mut v, p, &masks, &[], levels, g, next);
            }
            next += 1;
        }
    }
    next == sb_rows
}

/// Releases a tile column's other lanes when its thread unwinds.
struct Release<'a, P: Copy>(Option<&'a tile::Lanes<P>>);

impl<P: Copy> Drop for Release<'_, P> {
    fn drop(&mut self) {
        if std::thread::panicking()
            && let Some(l) = self.0
        {
            l.fail();
        }
    }
}

/// Puts the rows a tile column's other lanes coded (see [`tile::Lanes`])
/// into its first lane's work: their motion, filter masks and grid,
/// segments, samples and symbol counts, and every row's symbols in
/// order.
fn merge_lanes<P: EncPixel>(
    chunk: &mut [TileWork<P>],
    (c0, c1): (usize, usize),
    dims: &Dims,
    sb_rows: usize,
    grid: bool,
) {
    let lanes = chunk.len();
    let Some((first, rest)) = chunk.split_first_mut() else {
        return;
    };
    let (w, tsb) = (c1 - c0, (c1 - c0).div_ceil(8));
    let mi_rows = dims.height.div_ceil(8);
    for (j, t) in rest.iter().enumerate() {
        for r in (j + 1..sb_rows).step_by(lanes) {
            let (lo, hi) = (r * 8 * w, ((r * 8 + 8).min(mi_rows)) * w);
            first.state.mvs[lo..hi].copy_from_slice(&t.state.mvs[lo..hi]);
            first.state.segmap[lo..hi].copy_from_slice(&t.state.segmap[lo..hi]);
            if grid {
                first.state.grid[lo..hi].copy_from_slice(&t.state.grid[lo..hi]);
            }
            first.state.masks[r * tsb..(r + 1) * tsb]
                .copy_from_slice(&t.state.masks[r * tsb..(r + 1) * tsb]);
            for p in 0..3 {
                let (sx, sy) = if p == 0 {
                    (0, 0)
                } else {
                    (dims.ss_x, dims.ss_y)
                };
                let stride = dims.stride[p];
                let (x0, x1) = ((c0 * 8) >> sx, ((c1 * 8) >> sx).min(stride));
                let (y0, y1) = ((r * 64) >> sy, (((r + 1) * 64) >> sy).min(dims.rows[p]));
                for y in y0..y1 {
                    let at = y * stride;
                    first.pix[p][at + x0..at + x1].copy_from_slice(&t.pix[p][at + x0..at + x1]);
                }
            }
        }
        first.state.counts.add(&t.state.counts);
    }
    let mut out = Vec::with_capacity(chunk_symbols(first, rest));
    let mut ends = Vec::with_capacity(sb_rows);
    for r in 0..sb_rows {
        let (j, k) = (r % lanes, r / lanes);
        let t = if j == 0 { &*first } else { &rest[j - 1] };
        let start = k.checked_sub(1).map_or(0, |k| t.row_ends[k]);
        out.extend_from_slice(&t.out[start..t.row_ends[k]]);
        ends.push(out.len());
    }
    first.out = out;
    first.row_ends = ends;
}

/// The symbols of a tile column's lanes.
fn chunk_symbols<P: EncPixel>(first: &TileWork<P>, rest: &[TileWork<P>]) -> usize {
    first.out.len() + rest.iter().map(|t| t.out.len()).sum::<usize>()
}

/// A frame's probability updates and its tiles, written with them.
struct Written {
    old: [u8; 1728],
    new: [u8; 1728],
    old_nc: probs::Flat,
    new_nc: probs::Flat,
    /// By tile column, each column's tiles from top to bottom.
    tiles: Vec<Result<Vec<Vec<u8>>, CodecError>>,
}

/// The SAD one bit of rate is worth: libvpx's `sad_per_bit16` from the
/// AC quantizer step. Above 8 bits libvpx takes it from the step scaled
/// down to 8 bits, and shifts its SADs down to 8 bits too; here SADs stay
/// at the samples' scale, so the lambda is scaled up as they are.
fn sad_per_bit(ac: i32, bd: u8) -> u32 {
    let scale = f64::from(1u32 << (bd - 8));
    let q = f64::from(ac) / 4.0 / scale;
    ((0.0418 * q + 2.4107) * scale).round().max(1.0) as u32
}

/// The sample SSE a bit is worth: libvpx's `rdmult` (88·q²/24, on
/// distortions 16 times the SSE, rates in 1/512 bits over a 128 scale),
/// doubled: measured against libvpx's good quality on grainy and clean
/// 720p, twice the weight on rate takes 8 to 17% off the bitrate at the
/// same PSNR, Y or averaged over the planes, in the same time.
fn rd_lambda(dc: i32) -> f32 {
    0.00358 * (dc * dc) as f32
}

/// libvpx's loop filter level from the quantizer (`LPF_PICK_FROM_Q`), on
/// the AC step of the bit depth: at 10 and 12 bits the step is about 4
/// and 16 times larger, and so are the offset and the shift.
pub fn filter_level(qindex: u8, key: bool, bd: u8) -> u8 {
    let i = usize::from(qindex);
    let mut guess = match bd {
        10 => {
            let q = i64::from(crate::tables::AC_QLOOKUP_10[i]);
            (q * 20723 + 4_060_632 + (1 << 19)) >> 20
        }
        12 => {
            let q = i64::from(crate::tables::AC_QLOOKUP_12[i]);
            (q * 20723 + 16_242_526 + (1 << 21)) >> 22
        }
        _ => {
            let q = i64::from(crate::tables::AC_QLOOKUP[i]);
            (q * 20723 + 1_015_158 + (1 << 17)) >> 18
        }
    };
    if key {
        guess -= 4;
    }
    guess.clamp(0, 63) as u8
}

/// Speeds at and below which each inter block picks its filter.
pub const SWITCHABLE_SPEED: u8 = 2;

fn compressed_header(
    lossless: bool,
    coded: &probs::Coded,
    coef: Option<(&[u8; 1728], &[u8; 1728])>,
    nc: Option<(&probs::Flat, &probs::Flat)>,
) -> Result<Vec<u8>, CodecError> {
    let none = [0u8; probs::LEN];
    let (old, new) = nc.unwrap_or((&none, &none));
    let mut w = BoolWriter::new();
    if !lossless {
        // TX_MODE_SELECT.
        w.write_literal(u32::from(geom::ALLOW_32X32), 2);
        w.write(1, 128);
        probs::write(&mut w, old, new, coded, true);
    }
    let max_tx = if lossless { 0 } else { 3 };
    for tx in 0..=max_tx {
        write_coef_updates(&mut w, tx, coef);
    }
    probs::write(&mut w, old, new, coded, false);
    w.finish()
}

#[cfg(test)]
mod tests;

/// The coefficient probabilities in the order of the compressed header:
/// size, plane, reference, band, context, node.
fn flat_coef(fc: &FrameContext) -> [u8; 1728] {
    let mut out = [0u8; 1728];
    let mut i = 0;
    for tx in &fc.coef {
        for plane in tx {
            for r in plane {
                for band in r {
                    for ctx in band {
                        for &p in ctx {
                            out[i] = p;
                            i += 1;
                        }
                    }
                }
            }
        }
    }
    out
}

/// Whether band `band` codes context `ctx`.
fn coded(band: usize, ctx: usize) -> bool {
    band > 0 || ctx < 3
}

/// The bits of `v` in the header's sub-exponential code.
fn subexp_bits(v: u8) -> u32 {
    match v {
        0..=15 => 5,
        16..=31 => 6,
        32..=63 => 8,
        _ => 3 + if v - 64 < 65 { 7 } else { 8 },
    }
}

/// The code of probability `new` against `old` (`prob::inv_remap`'s
/// inverse).
fn remap(new: u8, old: u8) -> Option<u8> {
    (0..=254u8).find(|&v| prob::inv_remap(v, old) == new)
}

/// The coefficient probabilities after the updates that save bits on this
/// frame's counts (libvpx's savings test, a size at a time).
fn coef_updates(old: &[u8; 1728], counts: &prob::Counts, max_tx: usize) -> [u8; 1728] {
    let mut new = *old;
    let flag = |b: bool| u64::from(tile::bit_cost(b, 252));
    for tx in 0..=max_tx {
        let mut gain = 0i64;
        let mut coded_probs = 0u64;
        let mut changed = Vec::new();
        for plane in 0..2 {
            for r in 0..2 {
                for band in 0..6 {
                    for ctx in 0..6 {
                        if !coded(band, ctx) {
                            continue;
                        }
                        let c = &counts.coef[tx][plane][r][band][ctx];
                        let branch = counts.eob[tx][plane][r][band][ctx];
                        let pairs = [
                            (c[3], branch.saturating_sub(c[3])),
                            (c[0], c[1] + c[2]),
                            (c[1], c[2]),
                        ];
                        for (node, (c0, c1)) in pairs.into_iter().enumerate() {
                            coded_probs += 1;
                            let i = ((((tx * 2 + plane) * 2 + r) * 6 + band) * 6 + ctx) * 3 + node;
                            let total = u64::from(c0) + u64::from(c1);
                            if total == 0 {
                                continue;
                            }
                            let p = ((256 * u64::from(c0) + total / 2) / total).clamp(1, 255) as u8;
                            if p == old[i] {
                                continue;
                            }
                            let Some(v) = remap(p, old[i]) else { continue };
                            let cost = |q: u8| {
                                u64::from(c0) * u64::from(tile::bit_cost(false, q))
                                    + u64::from(c1) * u64::from(tile::bit_cost(true, q))
                            };
                            let save = cost(old[i]) as i64
                                - cost(p) as i64
                                - (flag(true) as i64 - flag(false) as i64)
                                - i64::from(subexp_bits(v)) * 256;
                            if save > 0 {
                                gain += save;
                                changed.push((i, p));
                            }
                        }
                    }
                }
            }
        }
        // Updating a size costs a flag for every probability in it.
        if gain > (coded_probs * flag(false) + 256) as i64 {
            for (i, p) in changed {
                new[i] = p;
            }
        }
    }
    new
}

/// The coefficient updates of size `tx` in the compressed header.
fn write_coef_updates(w: &mut BoolWriter, tx: usize, coef: Option<(&[u8; 1728], &[u8; 1728])>) {
    let base = tx * 2 * 2 * 6 * 6 * 3;
    let Some((old, new)) = coef.filter(|(o, n)| o[base..base + 432] != n[base..base + 432]) else {
        w.write(0, 128);
        return;
    };
    w.write(1, 128);
    for plane in 0..2 {
        for r in 0..2 {
            for band in 0..6 {
                for ctx in 0..6 {
                    if !coded(band, ctx) {
                        continue;
                    }
                    for node in 0..3 {
                        let i = ((((tx * 2 + plane) * 2 + r) * 6 + band) * 6 + ctx) * 3 + node;
                        match (new[i] != old[i]).then(|| remap(new[i], old[i])).flatten() {
                            Some(v) => {
                                w.write(1, 252);
                                write_subexp(w, v);
                            }
                            None => w.write(0, 252),
                        }
                    }
                }
            }
        }
    }
}

/// libvpx `encode_term_subexp`.
fn write_subexp(w: &mut BoolWriter, v: u8) {
    let v = u32::from(v);
    if v < 16 {
        w.write(0, 128);
        w.write_literal(v, 4);
    } else if v < 32 {
        w.write(1, 128);
        w.write(0, 128);
        w.write_literal(v - 16, 4);
    } else if v < 64 {
        w.write(1, 128);
        w.write(1, 128);
        w.write(0, 128);
        w.write_literal(v - 32, 5);
    } else {
        w.write(1, 128);
        w.write(1, 128);
        w.write(1, 128);
        let u = v - 64;
        let m = 65;
        if u < m {
            w.write_literal(u, 7);
        } else {
            w.write_literal((u + m) >> 1, 7);
            w.write((u + m) & 1, 128);
        }
    }
}
