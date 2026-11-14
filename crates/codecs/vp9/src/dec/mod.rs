//! The decoder of one frame's tile data, built for speed.
//!
//! A frame is decoded in two parts that can run on different threads.
//! [`FrameParse`] reads the tiles' syntax, a superblock row at a time,
//! into compact per-tile records; it also builds the frame's motion and
//! segmentation maps, the loop filter masks and the symbol counts that
//! adapt the probabilities. [`FrameRecon`] turns the records into samples
//! a superblock row at a time, with the loop filter one row behind, as
//! libvpx does: intra prediction reads the row above before it is
//! filtered, and filtering a row never touches the row below it. As rows
//! become final it publishes them, so that a later frame predicting from
//! this one can start before it is done.

pub mod bool;
pub mod ctx;
pub mod frame;
pub mod lf;
pub mod parse;
pub mod recon;
pub mod syntax;

use std::sync::Arc;

use cutlass_decode::CodecError;
use cutlass_simd::vp9::frame::FrameWriter;

use crate::dsp::pixel::Pixel;
use crate::geom;
use crate::header::Parsed;
use crate::lpf;
use crate::prob::{self, Counts};

use self::bool::Bool;
use self::ctx::Nb;
use self::frame::{Dims, MvRef, Pic};
use self::parse::{LfMi, Shared, State, Tile};
use self::recon::{Ctx, Recon, RefPic, Scale, View};
use self::syntax::TileSyntax;

fn bad(what: &'static str) -> CodecError {
    CodecError::Malformed { at_bit: 0, what }
}

/// The tile column and row edges in mode-info units (libvpx
/// `get_tile_offset`).
fn tile_edge(index: usize, mis: usize, log2: u8) -> usize {
    let sb = mis.div_ceil(8);
    (((index * sb) >> log2) << 3).min(mis)
}

/// Buffers a frame's parse keeps for the next frame.
#[derive(Default)]
pub struct ParseBufs {
    pub tiles: Vec<TileSyntax>,
    pub mvs: Vec<MvRef>,
    pub above: Vec<Nb>,
    pub above_part: Vec<u8>,
    pub above_nz: [Vec<u8>; 3],
    pub masks: Vec<lpf::Mask>,
    pub grid: Vec<LfMi>,
}

/// What a frame's parse leaves: its syntax for reconstruction, and what
/// the next frame and the probability adaptation need.
pub struct ParseOut {
    pub tiles: Vec<TileSyntax>,
    pub mvs: Vec<MvRef>,
    pub segmap: Vec<u8>,
    pub counts: Counts,
    pub masks: Vec<lpf::Mask>,
    pub grid: Vec<LfMi>,
    pub spare: ParseBufs,
}

/// The parse of one frame's tiles, a superblock row at a time.
pub struct FrameParse<'p> {
    parsed: &'p Parsed<'p>,
    levels: lpf::Levels,
    prev_mvs: Option<&'p [MvRef]>,
    last_seg: &'p [u8],
    state: State,
    tiles: Vec<TileSyntax>,
    data: Vec<&'p [u8]>,
    coders: Vec<Bool<'p>>,
    tile_row: usize,
    row_end: usize,
    ref_status: [u8; 3],
    mi_cols: usize,
    mi_rows: usize,
    sb_rows: usize,
    spare: ParseBufs,
}

impl<'p> FrameParse<'p> {
    pub fn new(
        parsed: &'p Parsed<'p>,
        prev_mvs: Option<&'p [MvRef]>,
        last_seg: &'p [u8],
        ref_status: [u8; 3],
        mut bufs: ParseBufs,
    ) -> Result<Self, CodecError> {
        let h = &parsed.header;
        if !geom::level_picture(h.width, h.height) {
            return Err(CodecError::Limit {
                what: "vp9 frame edge",
            });
        }
        let mi_cols = (h.width as usize).div_ceil(8);
        let mi_rows = (h.height as usize).div_ceil(8);
        let sb_cols = mi_cols.div_ceil(8);
        let sb_rows = mi_rows.div_ceil(8);
        let cells = mi_cols * mi_rows;
        if !h.is_intra() && h.use_prev_mvs && prev_mvs.is_none_or(|m| m.len() != cells) {
            return Err(bad("previous motion"));
        }
        let filter_on = h.filter_level != 0;
        let needs_grid = filter_on && h.ss_x != h.ss_y;
        let fill = |v: &mut Vec<u8>, n: usize| {
            let mut v = std::mem::take(v);
            v.clear();
            v.resize(n, 0);
            v
        };
        let mut mvs = std::mem::take(&mut bufs.mvs);
        mvs.clear();
        mvs.resize(cells, MvRef::default());
        let state = State {
            mvs,
            segmap: vec![0u8; cells],
            above: {
                let mut v = std::mem::take(&mut bufs.above);
                v.clear();
                v.resize(sb_cols * 8, Nb::default());
                v
            },
            above_part: fill(&mut bufs.above_part, sb_cols * 8),
            above_nz: [
                fill(&mut bufs.above_nz[0], sb_cols * 16),
                fill(&mut bufs.above_nz[1], sb_cols * 16),
                fill(&mut bufs.above_nz[2], sb_cols * 16),
            ],
            masks: {
                let mut v = std::mem::take(&mut bufs.masks);
                v.clear();
                v.resize(if filter_on { sb_cols * sb_rows } else { 0 }, lpf::Mask::default());
                v
            },
            grid: {
                let mut v = std::mem::take(&mut bufs.grid);
                v.clear();
                v.resize(if needs_grid { cells } else { 0 }, LfMi::default());
                v
            },
            counts: Counts::new(),
        };
        // The tile data, in tile order.
        let tile_cols = 1usize << h.log2_tile_cols;
        let tile_rows = 1usize << h.log2_tile_rows;
        let mut rest = parsed.tiles;
        let mut data: Vec<&[u8]> = Vec::with_capacity(tile_cols * tile_rows);
        for i in 0..tile_cols * tile_rows {
            if i + 1 == tile_cols * tile_rows {
                data.push(rest);
                break;
            }
            let size = rest
                .first_chunk::<4>()
                .map(|b| u32::from_be_bytes(*b) as usize)
                .ok_or(CodecError::Truncated {
                    at_bit: 0,
                    what: "tile size",
                })?;
            let after = &rest[4..];
            if after.len() < size {
                return Err(CodecError::Truncated {
                    at_bit: 0,
                    what: "tile",
                });
            }
            data.push(&after[..size]);
            rest = &after[size..];
        }
        let mut tiles = std::mem::take(&mut bufs.tiles);
        tiles.resize_with(tile_cols, TileSyntax::default);
        tiles.truncate(tile_cols);
        for (tc, t) in tiles.iter_mut().enumerate() {
            let col0 = tile_edge(tc, mi_cols, h.log2_tile_cols);
            let col1 = tile_edge(tc + 1, mi_cols, h.log2_tile_cols);
            t.reset(col0, col1, sb_rows);
        }
        Ok(Self {
            parsed,
            levels: lpf::levels(h),
            prev_mvs: if h.use_prev_mvs { prev_mvs } else { None },
            last_seg,
            state,
            tiles,
            data,
            coders: Vec::new(),
            tile_row: 0,
            row_end: 0,
            ref_status,
            mi_cols,
            mi_rows,
            sb_rows,
            spare: bufs,
        })
    }

    pub fn sb_rows(&self) -> usize {
        self.sb_rows
    }

    /// Parses superblock row `sb_row` of every tile column; rows go in
    /// order.
    pub fn row(&mut self, sb_row: usize) -> Result<(), CodecError> {
        let h = &self.parsed.header;
        let mi_row = sb_row * 8;
        let tile_cols = 1usize << h.log2_tile_cols;
        while mi_row >= self.row_end {
            // The next tile row's coders.
            let tr = if self.coders.is_empty() { 0 } else { self.tile_row + 1 };
            self.tile_row = tr;
            self.row_end = tile_edge(tr + 1, self.mi_rows, h.log2_tile_rows);
            self.coders.clear();
            for tc in 0..tile_cols {
                let b = Bool::new(self.data[tr * tile_cols + tc]).ok_or(bad("tile data"))?;
                self.coders.push(b);
            }
            if tr + 1 >= 1usize << h.log2_tile_rows {
                self.row_end = usize::MAX;
            }
        }
        let shared = Shared {
            h,
            fc: &self.parsed.fc,
            prev_mvs: self.prev_mvs,
            last_seg: self.last_seg,
            levels: &self.levels,
            mi_cols: self.mi_cols,
            mi_rows: self.mi_rows,
            sb_cols: self.mi_cols.div_ceil(8),
            collect: !h.error_resilient && !h.frame_parallel,
            intra_frame: h.is_intra(),
            part_probs: if h.is_intra() {
                prob::kf_partition()
            } else {
                self.parsed.fc.partition
            },
            bd: u32::from(h.bit_depth),
            filter_on: h.filter_level != 0,
            needs_grid: h.filter_level != 0 && h.ss_x != h.ss_y,
            ref_status: self.ref_status,
        };
        let mut corrupt = false;
        let mut unsupported = false;
        for (t, b) in self.tiles.iter_mut().zip(self.coders.iter_mut()) {
            if t.col0 >= t.col1 {
                t.end_row(sb_row);
                continue;
            }
            let (col0, col1) = (t.col0, t.col1);
            let mut p = Tile::new(&shared, &mut self.state, t, b, col0, col1);
            p.sb_row(mi_row);
            corrupt |= p.corrupt;
            unsupported |= p.unsupported;
        }
        if unsupported {
            return Err(CodecError::Unsupported {
                what: "vp9 reference scale",
            });
        }
        if corrupt {
            return Err(bad("tile data"));
        }
        Ok(())
    }

    /// The syntax parsed so far, and the loop filter masks.
    pub fn syntax(&self) -> (&[TileSyntax], &[lpf::Mask], &[LfMi]) {
        (&self.tiles, &self.state.masks, &self.state.grid)
    }

    pub fn finish(self) -> ParseOut {
        let State {
            mvs,
            segmap,
            above,
            above_part,
            above_nz,
            masks,
            grid,
            counts,
        } = self.state;
        let mut spare = self.spare;
        spare.above = above;
        spare.above_part = above_part;
        spare.above_nz = above_nz;
        ParseOut {
            tiles: self.tiles,
            mvs,
            segmap,
            counts,
            masks,
            grid,
            spare,
        }
    }
}

/// The references of a frame, by the header's LAST, GOLDEN and ALTREF,
/// and their status for the parse (0 usable, 1 missing or of another
/// format, 2 of a size the frame cannot predict from).
pub fn ref_pics<P>(parsed: &Parsed<'_>, refs: [Option<&Pic<P>>; 3]) -> ([Option<RefPic<P>>; 3], [u8; 3]) {
    let h = &parsed.header;
    let mut out: [Option<RefPic<P>>; 3] = [None, None, None];
    let mut status = [1u8; 3];
    if h.is_intra() {
        return (out, status);
    }
    for (i, slot) in out.iter_mut().enumerate() {
        let Some(f) = refs[i] else { continue };
        let d = &f.dims;
        if d.ss_x != h.ss_x as usize || d.ss_y != h.ss_y as usize || d.bit_depth != h.bit_depth {
            continue;
        }
        match Scale::between(d.width, d.height, h.width as usize, h.height as usize) {
            Ok(scale) => {
                *slot = Some(RefPic {
                    buf: Arc::clone(&f.buf),
                    dims: *d,
                    scale,
                });
                status[i] = 0;
            }
            Err(()) => status[i] = 2,
        }
    }
    (out, status)
}

/// The reconstruction of one frame, a superblock row at a time.
pub struct FrameRecon<P> {
    ctx: Ctx<P>,
    writer: FrameWriter<P>,
    levels: lpf::Levels,
    geo: lf::Geo,
    filter_on: bool,
    sb_rows: usize,
}

impl<P: Pixel> FrameRecon<P> {
    pub fn new(parsed: &Parsed<'_>, dims: Dims, refs: [Option<RefPic<P>>; 3], writer: FrameWriter<P>) -> Self {
        let h = &parsed.header;
        let mi_cols = dims.width.div_ceil(8);
        let mi_rows = dims.height.div_ceil(8);
        Self {
            ctx: Ctx {
                dims,
                refs,
                mi_cols,
                mi_rows,
                lossless: h.lossless,
            },
            writer,
            levels: lpf::levels(h),
            geo: lf::Geo {
                mi_rows,
                mi_cols,
                sb_cols: mi_cols.div_ceil(8),
                ss_x: dims.ss_x,
                ss_y: dims.ss_y,
                bd: u32::from(h.bit_depth),
            },
            filter_on: h.filter_level != 0,
            sb_rows: mi_rows.div_ceil(8),
        }
    }

    /// Reconstructs superblock row `sb_row`, filters the one above it and
    /// publishes the rows that are then final. Rows go in order.
    pub fn row(&mut self, tiles: &[TileSyntax], masks: &[lpf::Mask], grid: &[LfMi], sb_row: usize) {
        {
            let mut views = self.writer.tails().map(View::of);
            Recon::new(&self.ctx, &mut views).sb_row(tiles, sb_row);
            if self.filter_on && sb_row > 0 {
                lf::sb_row(&mut views, masks, grid, &self.levels, &self.geo, sb_row - 1);
            }
            if sb_row + 1 == self.sb_rows && self.filter_on {
                lf::sb_row(&mut views, masks, grid, &self.levels, &self.geo, sb_row);
            }
        }
        if sb_row + 1 == self.sb_rows {
            self.writer.publish(usize::MAX);
        } else {
            // The filter of the next row changes the last 8 chroma rows
            // (16 luma rows for 4:2:0) above it.
            self.writer.publish((sb_row * 64).saturating_sub(16));
        }
    }
}

/// Decodes a frame on this thread: each superblock row parsed, then
/// reconstructed.
pub fn decode_frame<P: Pixel>(
    parsed: &Parsed<'_>,
    refs: [Option<&Pic<P>>; 3],
    prev_mvs: Option<&[MvRef]>,
    last_seg: &[u8],
    bufs: ParseBufs,
    spare: &mut Vec<(Dims, Arc<cutlass_simd::vp9::frame::SharedFrame<P>>)>,
) -> Result<(Pic<P>, ParseOut), CodecError> {
    let h = &parsed.header;
    let dims = Dims::new(h.width as usize, h.height as usize, h.ss_x as usize, h.ss_y as usize, h.bit_depth);
    let (pics, status) = ref_pics(parsed, refs);
    let mut fp = FrameParse::new(parsed, prev_mvs, last_seg, status, bufs)?;
    let writer = frame::writer(&dims, spare);
    let buf = Arc::clone(writer.shared());
    let mut fr = FrameRecon::new(parsed, dims, pics, writer);
    for r in 0..fp.sb_rows() {
        fp.row(r)?;
        let (tiles, masks, grid) = fp.syntax();
        fr.row(tiles, masks, grid, r);
    }
    drop(fr);
    let out = fp.finish();
    Ok((
        Pic {
            buf,
            dims,
            color_space: h.color_space,
            color_range: h.color_range,
            render_width: h.render_width,
            render_height: h.render_height,
        },
        out,
    ))
}
