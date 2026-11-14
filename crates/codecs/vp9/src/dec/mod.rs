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
pub mod threads;

use std::sync::Arc;

use cutlass_decode::CodecError;
use cutlass_simd::vp9::frame::FrameWriter;

use crate::dsp::pixel::Pixel;
use crate::geom;
use crate::header::Parsed;
use crate::lpf;
use crate::prob::{self, Counts};

use self::bool::Bool;
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
    pub masks: Vec<lpf::Mask>,
    pub grid: Vec<LfMi>,
    pub states: Vec<State>,
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

/// The parse of one tile column, over every tile row.
struct TileCol<'p> {
    state: State,
    coder: Option<Bool<'p>>,
    tile_row: usize,
    row_end: usize,
}

/// The parse of one frame's tiles: a superblock row at a time on this
/// thread, or each tile column on its own.
pub struct FrameParse<'p> {
    parsed: &'p Parsed<'p>,
    levels: lpf::Levels,
    prev_mvs: Option<&'p [MvRef]>,
    last_seg: &'p [u8],
    cols: Vec<TileCol<'p>>,
    tiles: Vec<TileSyntax>,
    data: Vec<&'p [u8]>,
    ref_status: [u8; 3],
    mi_cols: usize,
    mi_rows: usize,
    sb_rows: usize,
    /// The frame's masks (and per-sample grid), merged row by row.
    masks: Vec<lpf::Mask>,
    grid: Vec<LfMi>,
    mvs: Vec<MvRef>,
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
        let mut spare_states = std::mem::take(&mut bufs.states);
        let mut cols = Vec::with_capacity(tile_cols);
        for (tc, t) in tiles.iter_mut().enumerate() {
            let col0 = tile_edge(tc, mi_cols, h.log2_tile_cols);
            let col1 = tile_edge(tc + 1, mi_cols, h.log2_tile_cols);
            t.reset(col0, col1, sb_rows);
            let width = col1 - col0;
            let tsb = width.div_ceil(8);
            let mut st = spare_states.pop().unwrap_or_else(State::empty);
            st.reset(col0, width, mi_rows, if filter_on { tsb * sb_rows } else { 0 }, needs_grid);
            cols.push(TileCol {
                state: st,
                coder: None,
                tile_row: 0,
                row_end: 0,
            });
        }
        let mut masks = std::mem::take(&mut bufs.masks);
        masks.clear();
        masks.resize(if filter_on { sb_cols * sb_rows } else { 0 }, lpf::Mask::default());
        let mut grid = std::mem::take(&mut bufs.grid);
        grid.clear();
        grid.resize(if needs_grid { cells } else { 0 }, LfMi::default());
        let mut mvs = std::mem::take(&mut bufs.mvs);
        mvs.clear();
        mvs.resize(cells, MvRef::default());
        Ok(Self {
            parsed,
            levels: lpf::levels(h),
            prev_mvs: if h.use_prev_mvs { prev_mvs } else { None },
            last_seg,
            cols,
            tiles,
            data,
            ref_status,
            mi_cols,
            mi_rows,
            sb_rows,
            masks,
            grid,
            mvs,
        })
    }

    pub fn sb_rows(&self) -> usize {
        self.sb_rows
    }

    /// What every tile reads, from the fields it borrows.
    fn shared<'s>(
        parsed: &'s Parsed<'s>,
        levels: &'s lpf::Levels,
        prev_mvs: Option<&'s [MvRef]>,
        last_seg: &'s [u8],
        ref_status: [u8; 3],
        (mi_cols, mi_rows): (usize, usize),
    ) -> Shared<'s> {
        let h = &parsed.header;
        Shared {
            h,
            fc: &parsed.fc,
            prev_mvs,
            last_seg,
            levels,
            mi_cols,
            mi_rows,
            sb_cols: mi_cols.div_ceil(8),
            collect: !h.error_resilient && !h.frame_parallel,
            intra_frame: h.is_intra(),
            part_probs: if h.is_intra() {
                prob::kf_partition()
            } else {
                parsed.fc.partition
            },
            bd: u32::from(h.bit_depth),
            filter_on: h.filter_level != 0,
            needs_grid: h.filter_level != 0 && h.ss_x != h.ss_y,
            ref_status,
        }
    }

    /// Parses superblock row `sb_row` of every tile column, on this
    /// thread, and merges its loop filter masks; rows go in order.
    pub fn row(&mut self, sb_row: usize) -> Result<(), CodecError> {
        let shared = Self::shared(self.parsed, &self.levels, self.prev_mvs, self.last_seg, self.ref_status, (self.mi_cols, self.mi_rows));
        let tile_cols = self.cols.len();
        for (tc, (col, syn)) in self.cols.iter_mut().zip(self.tiles.iter_mut()).enumerate() {
            parse_row(col, syn, &shared, &self.data, tc, tile_cols, sb_row)?;
        }
        merge_row(&self.cols, &shared, &mut self.masks, &mut self.grid, sb_row);
        Ok(())
    }

    /// Parses every row of every tile column, each column on its own
    /// thread when there are several.
    pub fn all(&mut self) -> Result<(), CodecError> {
        let shared = Self::shared(self.parsed, &self.levels, self.prev_mvs, self.last_seg, self.ref_status, (self.mi_cols, self.mi_rows));
        let tile_cols = self.cols.len();
        let rows = self.sb_rows;
        let data = &self.data;
        let run = |tc: usize, col: &mut TileCol<'p>, syn: &mut TileSyntax| -> Result<(), CodecError> {
            for r in 0..rows {
                parse_row(col, syn, &shared, data, tc, tile_cols, r)?;
            }
            Ok(())
        };
        if tile_cols == 1 {
            if let (Some(col), Some(syn)) = (self.cols.first_mut(), self.tiles.first_mut()) {
                run(0, col, syn)?;
            }
        } else {
            let results: Vec<Result<(), CodecError>> = std::thread::scope(|scope| {
                let handles: Vec<_> = self
                    .cols
                    .iter_mut()
                    .zip(self.tiles.iter_mut())
                    .enumerate()
                    .map(|(tc, (col, syn))| scope.spawn(move || run(tc, col, syn)))
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap_or(Err(bad("tile thread"))))
                    .collect()
            });
            for r in results {
                r?;
            }
        }
        for r in 0..rows {
            merge_row(&self.cols, &shared, &mut self.masks, &mut self.grid, r);
        }
        Ok(())
    }

    /// The syntax parsed so far, and the loop filter masks.
    pub fn syntax(&self) -> (&[TileSyntax], &[lpf::Mask], &[LfMi]) {
        (&self.tiles, &self.masks, &self.grid)
    }

    pub fn finish(self) -> ParseOut {
        let mi_cols = self.mi_cols;
        let mut mvs = self.mvs;
        let mut segmap = vec![0u8; mi_cols * self.mi_rows];
        let mut counts = Counts::new();
        let mut states = Vec::with_capacity(self.cols.len());
        for col in self.cols {
            let st = col.state;
            let (c0, w) = (st.col0, st.width);
            if w > 0 {
                for r in 0..self.mi_rows {
                    mvs[r * mi_cols + c0..r * mi_cols + c0 + w].copy_from_slice(&st.mvs[r * w..r * w + w]);
                    segmap[r * mi_cols + c0..r * mi_cols + c0 + w].copy_from_slice(&st.segmap[r * w..r * w + w]);
                }
            }
            counts.add(&st.counts);
            states.push(st);
        }
        ParseOut {
            tiles: self.tiles,
            mvs,
            segmap,
            counts,
            masks: self.masks,
            grid: self.grid,
            spare: ParseBufs {
                states,
                ..ParseBufs::default()
            },
        }
    }
}

/// Parses superblock row `sb_row` of tile column `tc`, starting the next
/// tile row's coder when the row is its first.
fn parse_row<'p>(
    col: &mut TileCol<'p>,
    syn: &mut TileSyntax,
    shared: &Shared<'_>,
    data: &[&'p [u8]],
    tc: usize,
    tile_cols: usize,
    sb_row: usize,
) -> Result<(), CodecError> {
    let h = shared.h;
    let mi_row = sb_row * 8;
    while mi_row >= col.row_end {
        let tr = if col.coder.is_none() { 0 } else { col.tile_row + 1 };
        col.tile_row = tr;
        col.row_end = if tr + 1 >= 1usize << h.log2_tile_rows {
            usize::MAX
        } else {
            tile_edge(tr + 1, shared.mi_rows, h.log2_tile_rows)
        };
        let d = data.get(tr * tile_cols + tc).copied().unwrap_or(&[]);
        col.coder = Some(Bool::new(d).ok_or(bad("tile data"))?);
    }
    if syn.col0 >= syn.col1 {
        syn.end_row(sb_row);
        return Ok(());
    }
    let (col0, col1) = (syn.col0, syn.col1);
    let Some(b) = col.coder.as_mut() else {
        return Err(bad("tile data"));
    };
    let mut p = Tile::new(shared, &mut col.state, syn, b, col0, col1);
    p.sb_row(mi_row);
    if p.unsupported {
        return Err(CodecError::Unsupported {
            what: "vp9 reference scale",
        });
    }
    if p.corrupt {
        return Err(bad("tile data"));
    }
    Ok(())
}

/// Copies superblock row `sb_row`'s loop filter masks (and per-sample
/// grid) from the tile columns into the frame's.
fn merge_row(cols: &[TileCol<'_>], shared: &Shared<'_>, masks: &mut [lpf::Mask], grid: &mut [LfMi], sb_row: usize) {
    if !shared.filter_on {
        return;
    }
    let sb_cols = shared.sb_cols;
    for col in cols {
        let st = &col.state;
        let tsb = st.width.div_ceil(8);
        let sb0 = st.col0 / 8;
        if tsb == 0 {
            continue;
        }
        masks[sb_row * sb_cols + sb0..sb_row * sb_cols + sb0 + tsb]
            .copy_from_slice(&st.masks[sb_row * tsb..sb_row * tsb + tsb]);
        if shared.needs_grid {
            let w = st.width;
            for r in sb_row * 8..(sb_row * 8 + 8).min(shared.mi_rows) {
                let at = r * shared.mi_cols + st.col0;
                grid[at..at + w].copy_from_slice(&st.grid[r * w..r * w + w]);
            }
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
    scratch: recon::Scratch<P>,
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
            scratch: recon::Scratch::new(),
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
            Recon::new(&self.ctx, &mut views, &mut self.scratch).sb_row(tiles, sb_row);
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

/// A frame's syntax buffers, back from the worker that reconstructed it.
pub struct Reuse {
    pub tiles: Vec<TileSyntax>,
    pub masks: Vec<lpf::Mask>,
    pub grid: Vec<LfMi>,
}

/// Decodes a frame. Without `workers`, on this thread: each superblock
/// row parsed, then reconstructed. With them, the whole frame is parsed
/// here and reconstructed on a worker, which sends its syntax buffers
/// back on the channel; the picture returned is then still being made,
/// and its readers wait for its rows.
pub fn decode_frame<P: Pixel + Send + Sync>(
    parsed: &Parsed<'_>,
    refs: [Option<&Pic<P>>; 3],
    prev_mvs: Option<&[MvRef]>,
    last_seg: &[u8],
    bufs: ParseBufs,
    spare: &mut Vec<(Dims, Arc<cutlass_simd::vp9::frame::SharedFrame<P>>)>,
    workers: Option<(&threads::Workers, &std::sync::mpsc::Sender<Reuse>)>,
) -> Result<(Pic<P>, ParseOut), CodecError> {
    let h = &parsed.header;
    let dims = Dims::new(h.width as usize, h.height as usize, h.ss_x as usize, h.ss_y as usize, h.bit_depth);
    let (pics, status) = ref_pics(parsed, refs);
    let mut fp = FrameParse::new(parsed, prev_mvs, last_seg, status, bufs)?;
    let rows = fp.sb_rows();
    let pic_of = |buf| Pic {
        buf,
        dims,
        color_space: h.color_space,
        color_range: h.color_range,
        render_width: h.render_width,
        render_height: h.render_height,
    };
    if let Some((w, ret)) = workers {
        fp.all()?;
        let mut out = fp.finish();
        let writer = frame::writer(&dims, spare);
        let pic = pic_of(Arc::clone(writer.shared()));
        let mut fr = FrameRecon::new(parsed, dims, pics, writer);
        let reuse = Reuse {
            tiles: std::mem::take(&mut out.tiles),
            masks: std::mem::take(&mut out.masks),
            grid: std::mem::take(&mut out.grid),
        };
        let ret = ret.clone();
        w.run(move || {
            for r in 0..rows {
                fr.row(&reuse.tiles, &reuse.masks, &reuse.grid, r);
            }
            drop(fr);
            let _ = ret.send(reuse);
        });
        return Ok((pic, out));
    }
    let writer = frame::writer(&dims, spare);
    let pic = pic_of(Arc::clone(writer.shared()));
    let mut fr = FrameRecon::new(parsed, dims, pics, writer);
    for r in 0..rows {
        fp.row(r)?;
        let (tiles, masks, grid) = fp.syntax();
        fr.row(tiles, masks, grid, r);
    }
    drop(fr);
    Ok((pic, fp.finish()))
}
