//! The decoder of one frame's tile data, built for speed.
//!
//! Superblock rows are decoded across every tile column, and the loop
//! filter follows one superblock row behind, as libvpx does: intra
//! prediction reads the row above before it is filtered, and filtering a
//! row never touches the row below it.

pub mod bool;
pub mod ctx;
pub mod frame;
pub mod lf;
pub mod tile;

use cutlass_decode::CodecError;

use crate::dsp::pixel::Pixel;
use crate::geom;
use crate::header::Parsed;
use crate::lpf;
use crate::prob::{self, Counts};

use self::bool::Bool;
use self::ctx::Nb;
use self::frame::{Frame, MvRef};
use self::tile::{LfMi, RefPic, Scale, Shared, State, Tile};

/// A decoded frame and what the next frame needs from it.
pub struct Decoded<P> {
    pub frame: Frame<P>,
    pub mvs: Vec<MvRef>,
    pub segmap: Vec<u8>,
    pub counts: Counts,
}

/// Buffers kept from frame to frame so that a frame allocates nothing.
pub struct Spare<P> {
    pub frames: Vec<Frame<P>>,
    pub mvs: Vec<Vec<MvRef>>,
    pub emu: Vec<P>,
    pub above: Vec<Nb>,
    pub above_part: Vec<u8>,
    pub above_nz: [Vec<u8>; 3],
    pub masks: Vec<lpf::Mask>,
    pub grid: Vec<LfMi>,
}

impl<P> Default for Spare<P> {
    fn default() -> Self {
        Self {
            frames: Vec::new(),
            mvs: Vec::new(),
            emu: Vec::new(),
            above: Vec::new(),
            above_part: Vec::new(),
            above_nz: [Vec::new(), Vec::new(), Vec::new()],
            masks: Vec::new(),
            grid: Vec::new(),
        }
    }
}

fn bad(what: &'static str) -> CodecError {
    CodecError::Malformed { at_bit: 0, what }
}

/// The tile column and row edges in mode-info units (libvpx
/// `get_tile_offset`).
fn tile_edge(index: usize, mis: usize, log2: u8) -> usize {
    let sb = mis.div_ceil(8);
    (((index * sb) >> log2) << 3).min(mis)
}

/// Decodes the tiles of a parsed frame. `refs` are the frame's LAST,
/// GOLDEN and ALTREF pictures.
pub fn decode_frame<P: Pixel>(
    parsed: &Parsed<'_>,
    refs: [Option<&Frame<P>>; 3],
    prev_mvs: Option<&[MvRef]>,
    last_seg: &[u8],
    spare: &mut Spare<P>,
) -> Result<Decoded<P>, CodecError> {
    let h = &parsed.header;
    let width = h.width as usize;
    let height = h.height as usize;
    if !geom::level_picture(h.width, h.height) {
        return Err(CodecError::Limit {
            what: "vp9 frame edge",
        });
    }
    let (ss_x, ss_y) = (h.ss_x as usize, h.ss_y as usize);
    let mi_cols = width.div_ceil(8);
    let mi_rows = height.div_ceil(8);
    let sb_cols = mi_cols.div_ceil(8);
    let sb_rows = mi_rows.div_ceil(8);
    let intra_frame = h.is_intra();
    let mut pics: [Option<RefPic<'_, P>>; 3] = [None, None, None];
    let mut bad_scale = false;
    if !intra_frame {
        for (i, slot) in pics.iter_mut().enumerate() {
            let Some(f) = refs[i] else { continue };
            if f.ss_x != ss_x || f.ss_y != ss_y || f.bit_depth != h.bit_depth {
                continue;
            }
            match Scale::between(f.width, f.height, width, height) {
                Ok(scale) => *slot = Some(RefPic { frame: f, scale }),
                Err(()) => bad_scale = true,
            }
        }
    }
    let levels = lpf::levels(h);
    let filter_on = h.filter_level != 0;
    let needs_grid = filter_on && !(ss_x == ss_y);
    let shared = Shared {
        h,
        fc: &parsed.fc,
        refs: pics,
        prev_mvs: if h.use_prev_mvs { prev_mvs } else { None },
        last_seg,
        levels: &levels,
        mi_cols,
        mi_rows,
        sb_cols,
        collect: !h.error_resilient && !h.frame_parallel,
        intra_frame,
        part_probs: if intra_frame {
            prob::kf_partition()
        } else {
            parsed.fc.partition
        },
        bd: u32::from(h.bit_depth),
        max: (1i32 << h.bit_depth) - 1,
        filter_on,
        needs_grid,
    };
    if !intra_frame && h.use_prev_mvs && prev_mvs.is_none_or(|m| m.len() != mi_cols * mi_rows) {
        return Err(bad("previous motion"));
    }
    // Buffers, reused from earlier frames where they fit.
    let cur = match spare
        .frames
        .iter()
        .position(|f| f.fits(width, height, ss_x, ss_y, h.bit_depth))
    {
        Some(i) => spare.frames.swap_remove(i),
        None => Frame::new(width, height, ss_x, ss_y, h.bit_depth),
    };
    let cells = mi_cols * mi_rows;
    let mut mvs = spare.mvs.pop().unwrap_or_default();
    mvs.clear();
    mvs.resize(cells, MvRef::default());
    let take = |v: &mut Vec<u8>, n: usize| {
        let mut v = std::mem::take(v);
        v.clear();
        v.resize(n, 0);
        v
    };
    let mut state = State {
        cur,
        mvs,
        segmap: vec![0u8; cells],
        above: {
            let mut v = std::mem::take(&mut spare.above);
            v.clear();
            v.resize(sb_cols * 8, Nb::default());
            v
        },
        above_part: take(&mut spare.above_part, sb_cols * 8),
        above_nz: [
            take(&mut spare.above_nz[0], sb_cols * 16),
            take(&mut spare.above_nz[1], sb_cols * 16),
            take(&mut spare.above_nz[2], sb_cols * 16),
        ],
        masks: {
            let mut v = std::mem::take(&mut spare.masks);
            v.clear();
            v.resize(if filter_on { sb_cols * sb_rows } else { 0 }, lpf::Mask::default());
            v
        },
        grid: {
            let mut v = std::mem::take(&mut spare.grid);
            v.clear();
            v.resize(if needs_grid { cells } else { 0 }, LfMi::default());
            v
        },
        counts: Counts::new(),
    };
    // Split the tile data.
    let tile_cols = 1usize << h.log2_tile_cols;
    let tile_rows = 1usize << h.log2_tile_rows;
    let mut data = parsed.tiles;
    let mut tiles: Vec<&[u8]> = Vec::with_capacity(tile_cols * tile_rows);
    for i in 0..tile_cols * tile_rows {
        if i + 1 == tile_cols * tile_rows {
            tiles.push(data);
            break;
        }
        let size = data
            .first_chunk::<4>()
            .map(|b| u32::from_be_bytes(*b) as usize)
            .ok_or(CodecError::Truncated {
                at_bit: 0,
                what: "tile size",
            })?;
        let rest = &data[4..];
        if rest.len() < size {
            return Err(CodecError::Truncated {
                at_bit: 0,
                what: "tile",
            });
        }
        tiles.push(&rest[..size]);
        data = &rest[size..];
    }
    let geo = lf::Geo {
        mi_rows,
        mi_cols,
        sb_cols,
        ss_x,
        ss_y,
        bd: u32::from(h.bit_depth),
    };
    let mut emu = std::mem::take(&mut spare.emu);
    let mut corrupt = false;
    for tr in 0..tile_rows {
        let row0 = tile_edge(tr, mi_rows, h.log2_tile_rows);
        let row1 = tile_edge(tr + 1, mi_rows, h.log2_tile_rows);
        let mut coders: Vec<(Bool<'_>, usize, usize)> = Vec::with_capacity(tile_cols);
        for tc in 0..tile_cols {
            let col0 = tile_edge(tc, mi_cols, h.log2_tile_cols);
            let col1 = tile_edge(tc + 1, mi_cols, h.log2_tile_cols);
            let b = Bool::new(tiles[tr * tile_cols + tc]).ok_or(bad("tile data"))?;
            coders.push((b, col0, col1));
        }
        let mut mi_row = row0;
        while mi_row < row1 {
            for (b, col0, col1) in &mut coders {
                if *col0 >= *col1 {
                    continue;
                }
                let mut t = Tile::new(&shared, &mut state, b, &mut emu, *col0, *col1);
                t.sb_row(mi_row);
                corrupt |= t.corrupt;
            }
            if corrupt {
                break;
            }
            if filter_on && mi_row >= 8 {
                lf::sb_row(&mut state.cur.planes, &state.masks, &state.grid, &levels, &geo, mi_row / 8 - 1);
            }
            mi_row += 8;
        }
        if corrupt {
            break;
        }
    }
    spare.emu = emu;
    if corrupt && bad_scale {
        return Err(CodecError::Unsupported {
            what: "vp9 reference scale",
        });
    }
    if corrupt {
        return Err(bad("tile data"));
    }
    if filter_on && sb_rows > 0 {
        lf::sb_row(&mut state.cur.planes, &state.masks, &state.grid, &levels, &geo, sb_rows - 1);
    }
    let State {
        mut cur,
        mvs,
        segmap,
        above,
        above_part,
        above_nz,
        masks,
        grid,
        counts,
    } = state;
    spare.above = above;
    spare.above_part = above_part;
    spare.above_nz = above_nz;
    spare.masks = masks;
    spare.grid = grid;
    cur.color_space = h.color_space;
    cur.color_range = h.color_range;
    cur.render_width = h.render_width;
    cur.render_height = h.render_height;
    Ok(Decoded {
        frame: cur,
        mvs,
        segmap,
        counts,
    })
}
