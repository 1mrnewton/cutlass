//! Decoding the tiles of a frame on several threads.
//!
//! Tiles are independent until the in-loop filters: entropy coding, the
//! mode info contexts, motion vector candidates and intra edges all stop
//! at the tile edge (spec 7.3, `is_inside`, `clear_block_decoded_flags`),
//! and inter prediction reads only the reference frames. So the tiles of a
//! frame can be decoded at the same time. This thread decodes tiles into
//! the frame itself. Each helper decodes its tiles into a [`TileShell`] of
//! its own (a frame state and picture with the frame's size), and what a
//! helper's tile wrote is copied into the frame once every tile is done.

use std::ops::Range;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use cutlass_decode::CodecError;

use crate::dsp::Pixel;

use super::super::cdf::CdfContext;
use super::super::frame::{FrameState, Picture};
use super::super::obu::{FrameHeader, RESTORE_NONE, SequenceHeader};
use super::super::recon::Scratch;
use super::super::tile::{ReconLog, lr_unit_span};
use super::{
    PicturePixel, R, RefHeaderState, Spares, TileBufs, decode_tile, decode_tiles, lock, malformed,
};

/// A frame state, a picture and tile buffers a helper decodes tiles into.
/// Only the tiles' areas are ever read back, and a tile writes all of its
/// area before reading it, so a shell is reused without clearing.
pub(super) struct TileShell<P> {
    fs: FrameState,
    pic: Picture<P>,
    bufs: TileBufs<P>,
}

impl<P: Pixel> TileShell<P> {
    fn new(seq: &SequenceHeader, hdr: &FrameHeader, cur: &Picture<P>) -> Self {
        let fs = FrameState::new(seq, hdr);
        let bufs = TileBufs::new(&fs, Scratch::default());
        let pic = Picture::new(
            cur.planes[0].w,
            cur.planes[0].h,
            cur.planes.len(),
            cur.sub_x,
            cur.sub_y,
            cur.bd,
        );
        Self { fs, pic, bufs }
    }

    /// Whether this shell has the size and format of the frame.
    fn fits(&self, fs: &FrameState, cur: &Picture<P>) -> bool {
        self.fs.mi_cols == fs.mi_cols
            && self.fs.mi_rows == fs.mi_rows
            && self.pic.bd == cur.bd
            && self.pic.sub_x == cur.sub_x
            && self.pic.sub_y == cur.sub_y
            && self.pic.planes.len() == cur.planes.len()
            && self.pic.planes.iter().zip(&cur.planes).all(|(a, b)| {
                a.w == b.w && a.h == b.h && a.stride == b.stride && a.alloc_h == b.alloc_h
            })
    }

    /// Readies the shell for the tiles of the frame of `fs`: what the
    /// frame starts with (CDEF indices unset, fresh loop restoration
    /// units) and what tiles read of the frame and never write (the motion
    /// field and the segment ids of the primary reference frame).
    fn prepare(self, seq: &SequenceHeader, hdr: &FrameHeader, fs: &FrameState) -> Self {
        let Self { fs: old, pic, bufs } = self;
        let mut sfs = FrameState::reuse(Some(old), seq, hdr);
        sfs.motion_field.extend_from_slice(&fs.motion_field);
        sfs.prev_segment_ids.clone_from(&fs.prev_segment_ids);
        Self { fs: sfs, pic, bufs }
    }

    /// Copies what tile `tile_num` wrote of this shell into the frame.
    fn copy_tile(
        &self,
        seq: &SequenceHeader,
        hdr: &FrameHeader,
        tile_num: usize,
        fs: &mut FrameState,
        cur: &mut Picture<P>,
    ) {
        let ti = &hdr.tile_info;
        let (tr, tc) = (tile_num / ti.cols, tile_num % ti.cols);
        let rows = ti.mi_row_starts[tr] as usize..ti.mi_row_starts[tr + 1] as usize;
        let cols = ti.mi_col_starts[tc] as usize..ti.mi_col_starts[tc + 1] as usize;
        // A tile on the bottom or right edge of the frame also writes past
        // it (blocks and transforms reaching over the edge): its area runs
        // to the end of the arrays there.
        let (last_row, last_col) = (tr + 1 == ti.rows, tc + 1 == ti.cols);
        // `rows` and `cols` shifted down by `sy` and `sx`, in an array of
        // `width` columns and `height` rows.
        let span = |r: &Range<usize>, c: &Range<usize>, (sx, sy): (u32, u32), width, height| {
            let r = (r.start >> sy)..if last_row { height } else { r.end >> sy };
            let c = (c.start >> sx)..if last_col { width } else { c.end >> sx };
            (r, c)
        };
        let st = fs.mi_stride;
        let mi_height = fs.mi.len() / st;
        let (r, c) = span(&rows, &cols, (0, 0), st, mi_height);
        let from = &self.fs;
        copy_rect(&mut fs.mi, &from.mi, st, &r, &c);
        copy_rect(&mut fs.uv_modes, &from.uv_modes, st, &r, &c);
        copy_rect(&mut fs.tx_types, &from.tx_types, st, &r, &c);
        copy_rect(&mut fs.inter_tx_sizes, &from.inter_tx_sizes, st, &r, &c);
        copy_rect(&mut fs.segment_ids, &from.segment_ids, st, &r, &c);
        let num_planes = seq.num_planes();
        for p in 0..num_planes {
            let sub = plane_sub(seq, p);
            let (r, c) = span(&rows, &cols, sub, st, mi_height);
            copy_rect(&mut fs.lf_tx[p], &from.lf_tx[p], st, &r, &c);
            if hdr.lr.frame_type[p] != RESTORE_NONE {
                let (ur, uc) = lr_unit_span(seq, hdr, from, p, rows.clone(), cols.clone());
                copy_rect(&mut fs.lr[p], &from.lr[p], fs.lr_cols[p], &ur, &uc);
            }
        }
        // CDEF indices are per 64x64 luma: 16 4x4s.
        let cst = fs.cdef_stride;
        let (r, c) = span(&rows, &cols, (4, 4), cst, fs.cdef_idx.len() / cst);
        copy_rect(&mut fs.cdef_idx, &from.cdef_idx, cst, &r, &c);
        for (p, (to, src)) in cur.planes.iter_mut().zip(&self.pic.planes).enumerate() {
            let (sx, sy) = plane_sub(seq, p);
            let px_rows = 4 * rows.start..4 * rows.end;
            let px_cols = 4 * cols.start..4 * cols.end;
            let (r, c) = span(
                &px_rows,
                &px_cols,
                (sx, sy),
                to.stride,
                to.data.len() / to.stride,
            );
            copy_rect(&mut to.data, &src.data, to.stride, &r, &c);
        }
    }
}

fn plane_sub(seq: &SequenceHeader, plane: usize) -> (u32, u32) {
    if plane > 0 {
        (u32::from(seq.subsampling_x), u32::from(seq.subsampling_y))
    } else {
        (0, 0)
    }
}

/// Copies rows `rows`, columns `cols` of a `stride`-wide array.
fn copy_rect<T: Copy>(
    dst: &mut [T],
    src: &[T],
    stride: usize,
    rows: &Range<usize>,
    cols: &Range<usize>,
) {
    if cols.is_empty() {
        return;
    }
    for r in rows.clone() {
        let at = r * stride;
        dst[at + cols.start..at + cols.end].copy_from_slice(&src[at + cols.start..at + cols.end]);
    }
}

/// [`decode_tiles`] for a frame of more than one tile, on up to `threads`
/// threads (this one among them). The helpers' shells come from the
/// decoder's pool, and at most `threads - 1` are lent at once to all the
/// frames decoding together: a frame finding none left decodes its tiles
/// alone. The result is the one decoding the tiles in order gives: the
/// error of the first tile that fails, else the CDFs of the context tile.
pub(super) fn decode_tiles_threaded<P: PicturePixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &mut FrameState,
    tiles: &[(usize, Vec<u8>)],
    cdf: &mut CdfContext,
    refs: &[Option<&Picture<P>>; 8],
    ref_hdr: &[RefHeaderState; 8],
    cur: &mut Picture<P>,
    spares: &Mutex<Spares<P>>,
    threads: usize,
) -> R<Box<CdfContext>> {
    let want = threads.min(tiles.len()).saturating_sub(1);
    let shells = take_shells(seq, hdr, fs, cur, spares, want, threads);
    if shells.is_empty() {
        return decode_tiles(
            seq,
            hdr,
            fs,
            tiles,
            cdf,
            refs,
            ref_hdr,
            cur,
            None,
            None,
            &mut |_: &FrameState,
                  _: &Picture<P>,
                  _: Option<&mut ReconLog>,
                  _: usize,
                  _: usize,
                  _: bool| {},
            Some(spares),
        );
    }
    let mut shells: Vec<_> = shells
        .into_iter()
        .map(|sh| sh.prepare(seq, hdr, fs))
        .collect();
    let scratch = lock(spares).scratch.pop().unwrap_or_default();
    let mut own = TileBufs::new(fs, scratch);
    let cdf = &*cdf;
    // Tiles are taken in order, so when tile `i` fails every tile before it
    // has been taken, and the tiles after it need not be decoded.
    let next = AtomicUsize::new(0);
    let failed = AtomicUsize::new(usize::MAX);
    let work = |fs: &mut FrameState, pic: &mut Picture<P>, bufs: &mut TileBufs<P>| {
        let mut done = Vec::new();
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= tiles.len() || i > failed.load(Ordering::Relaxed) {
                return done;
            }
            let (tile_num, data) = &tiles[i];
            let res = decode_tile(
                seq,
                hdr,
                fs,
                pic,
                cdf,
                refs,
                ref_hdr,
                *tile_num,
                data,
                bufs,
                None,
                None,
                &mut |_: &FrameState,
                      _: &Picture<P>,
                      _: Option<&mut ReconLog>,
                      _: usize,
                      _: usize,
                      _: bool| {},
            );
            if res.is_err() {
                failed.fetch_min(i, Ordering::Relaxed);
            }
            done.push((i, res));
        }
    };
    let (mine, theirs) = std::thread::scope(|s| {
        let work = &work;
        let handles: Vec<_> = shells
            .iter_mut()
            .map(|sh| s.spawn(move || work(&mut sh.fs, &mut sh.pic, &mut sh.bufs)))
            .collect();
        let mine = work(fs, cur, &mut own);
        let theirs: Vec<_> = handles.into_iter().map(|h| h.join().ok()).collect();
        (mine, theirs)
    });
    let ctx = hdr.tile_info.context_update_tile_id as usize;
    let mut first_err: Option<(usize, CodecError)> = None;
    let mut saved = None;
    let mut copies = Vec::new();
    let all = std::iter::once((None, Some(mine)))
        .chain(theirs.into_iter().enumerate().map(|(k, t)| (Some(k), t)));
    for (shell, done) in all {
        let Some(done) = done else {
            // A helper that panicked: the frame fails as a tile would.
            first_err = Some((0, malformed("tile thread")));
            continue;
        };
        for (i, res) in done {
            match res {
                Err(e) => {
                    if first_err.as_ref().is_none_or(|(j, _)| i < *j) {
                        first_err = Some((i, e));
                    }
                }
                Ok(tile_cdf) => {
                    if tiles[i].0 == ctx {
                        saved = Some(tile_cdf);
                    }
                    if let Some(k) = shell {
                        copies.push((k, tiles[i].0));
                    }
                }
            }
        }
    }
    if first_err.is_none() {
        for (k, tile_num) in copies {
            shells[k].copy_tile(seq, hdr, tile_num, fs, cur);
        }
    }
    {
        let mut sp = lock(spares);
        sp.scratch.push(own.scratch);
        sp.tile_shells_out -= shells.len();
        sp.tile_shells.append(&mut shells);
    }
    if let Some((_, e)) = first_err {
        return Err(e);
    }
    Ok(saved.unwrap_or_else(|| Box::new(cdf.clone())))
}

/// Up to `want` shells for the frame of `fs`, within the decoder's
/// budget of `threads - 1` lent at once.
fn take_shells<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &FrameState,
    cur: &Picture<P>,
    spares: &Mutex<Spares<P>>,
    want: usize,
    threads: usize,
) -> Vec<TileShell<P>> {
    let mut pooled = Vec::new();
    let n = {
        let mut sp = lock(spares);
        let n = want.min(threads.saturating_sub(1).saturating_sub(sp.tile_shells_out));
        sp.tile_shells_out += n;
        while pooled.len() < n {
            let Some(sh) = sp.tile_shells.pop() else {
                break;
            };
            pooled.push(sh);
        }
        n
    };
    // New shells are made with the pool unlocked.
    let mut out = Vec::with_capacity(n);
    for sh in pooled {
        if sh.fits(fs, cur) {
            out.push(sh);
        } else {
            out.push(TileShell::new(seq, hdr, cur));
        }
    }
    while out.len() < n {
        out.push(TileShell::new(seq, hdr, cur));
    }
    out
}
