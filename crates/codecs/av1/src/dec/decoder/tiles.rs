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
use super::super::types::MiInfo;
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
}

fn plane_sub(seq: &SequenceHeader, plane: usize) -> (u32, u32) {
    if plane > 0 {
        (u32::from(seq.subsampling_x), u32::from(seq.subsampling_y))
    } else {
        (0, 0)
    }
}

/// The area of tile `tile_num` in luma 4x4 units. A tile on the bottom or
/// right edge of the frame also writes past it (blocks and transforms
/// reaching over the edge), so its area runs to the end of the arrays
/// there.
struct TileArea {
    rows: Range<usize>,
    cols: Range<usize>,
    last_row: bool,
    last_col: bool,
}

impl TileArea {
    fn of(hdr: &FrameHeader, tile_num: usize) -> Self {
        let ti = &hdr.tile_info;
        let (tr, tc) = (tile_num / ti.cols, tile_num % ti.cols);
        Self {
            rows: ti.mi_row_starts[tr] as usize..ti.mi_row_starts[tr + 1] as usize,
            cols: ti.mi_col_starts[tc] as usize..ti.mi_col_starts[tc + 1] as usize,
            last_row: tr + 1 == ti.rows,
            last_col: tc + 1 == ti.cols,
        }
    }

    /// The area in an array of `width` columns and `height` rows whose
    /// units are `1 << scale` 4x4s wide and high (`scale` = (x, y)),
    /// or with `px`, `4 >> scale` samples.
    fn span(
        &self,
        (sx, sy): (u32, u32),
        px: bool,
        width: usize,
        height: usize,
    ) -> (Range<usize>, Range<usize>) {
        let m = if px { 4 } else { 1 };
        let rows = ((m * self.rows.start) >> sy)..if self.last_row {
            height
        } else {
            (m * self.rows.end) >> sy
        };
        let cols = ((m * self.cols.start) >> sx)..if self.last_col {
            width
        } else {
            (m * self.cols.end) >> sx
        };
        (rows, cols)
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

/// Rows `rows` of a `stride`-wide array, as the slice of those rows.
struct Band<'a, T> {
    rows: Range<usize>,
    stride: usize,
    data: &'a mut [T],
}

impl<T: Copy> Band<'_, T> {
    /// Copies the rows of `rows` in this band, columns `cols`, from the
    /// whole array `src`.
    fn copy(&mut self, src: &[T], rows: &Range<usize>, cols: &Range<usize>) {
        let r = rows.start.max(self.rows.start)..rows.end.min(self.rows.end);
        if cols.is_empty() {
            return;
        }
        let st = self.stride;
        for row in r {
            let s = row * st;
            let d = (row - self.rows.start) * st;
            self.data[d + cols.start..d + cols.end]
                .copy_from_slice(&src[s + cols.start..s + cols.end]);
        }
    }
}

/// `a`, `stride` wide, cut into bands at the rows of `cuts` (ascending,
/// from 0 to the height of the array).
fn bands<'a, T>(a: &'a mut [T], stride: usize, cuts: &[usize]) -> Vec<Band<'a, T>> {
    let mut out = Vec::with_capacity(cuts.len().saturating_sub(1));
    let mut rest = a;
    for w in cuts.windows(2) {
        let n = ((w[1] - w[0]) * stride).min(rest.len());
        let (band, tail) = std::mem::take(&mut rest).split_at_mut(n);
        out.push(Band {
            rows: w[0]..w[1],
            stride,
            data: band,
        });
        rest = tail;
    }
    out
}

/// One band of rows of the frame: of every mode info array, and of the
/// planes.
struct FrameBand<'a, P> {
    mi: Band<'a, MiInfo>,
    uv_modes: Band<'a, u8>,
    tx_types: Band<'a, u8>,
    inter_tx_sizes: Band<'a, u8>,
    segment_ids: Band<'a, u8>,
    lf_tx: Vec<Band<'a, u8>>,
    planes: Vec<Band<'a, P>>,
}

/// Copies what the tiles `copies` (a shell and a tile number each) wrote of
/// their shells into the frame. The rows are cut into `threads` bands,
/// copied at the same time; the CDEF indices and loop restoration units
/// are few and copied here.
fn copy_tiles<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    shells: &[TileShell<P>],
    copies: &[(usize, usize)],
    fs: &mut FrameState,
    cur: &mut Picture<P>,
    threads: usize,
) {
    if copies.is_empty() {
        return;
    }
    let num_planes = seq.num_planes();
    for &(k, tile_num) in copies {
        let a = TileArea::of(hdr, tile_num);
        let from = &shells[k].fs;
        for p in 0..num_planes {
            if hdr.lr.frame_type[p] != RESTORE_NONE {
                let (ur, uc) = lr_unit_span(seq, hdr, from, p, a.rows.clone(), a.cols.clone());
                copy_rect(&mut fs.lr[p], &from.lr[p], fs.lr_cols[p], &ur, &uc);
            }
        }
        // CDEF indices are per 64x64 luma: 16 4x4s.
        let cst = fs.cdef_stride;
        let (r, c) = a.span((4, 4), false, cst, fs.cdef_idx.len() / cst);
        copy_rect(&mut fs.cdef_idx, &from.cdef_idx, cst, &r, &c);
    }
    let st = fs.mi_stride;
    let mi_height = fs.mi.len() / st;
    let nb = threads.clamp(1, mi_height.max(1));
    let cuts: Vec<usize> = (0..=nb).map(|i| mi_height * i / nb).collect();
    let scaled = |sy: u32, px: bool, height: usize| -> Vec<usize> {
        let m = if px { 4 } else { 1 };
        let mut c: Vec<usize> = cuts.iter().map(|&r| ((m * r) >> sy).min(height)).collect();
        if let Some(l) = c.last_mut() {
            *l = height;
        }
        c
    };
    let mut lf: Vec<std::vec::IntoIter<Band<'_, u8>>> = fs.lf_tx[..num_planes]
        .iter_mut()
        .enumerate()
        .map(|(p, a)| bands(a, st, &scaled(plane_sub(seq, p).1, false, mi_height)).into_iter())
        .collect();
    let mut px: Vec<std::vec::IntoIter<Band<'_, P>>> = cur
        .planes
        .iter_mut()
        .enumerate()
        .map(|(p, pl)| {
            let h = pl.data.len() / pl.stride;
            bands(
                &mut pl.data,
                pl.stride,
                &scaled(plane_sub(seq, p).1, true, h),
            )
            .into_iter()
        })
        .collect();
    let mut mi = bands(&mut fs.mi, st, &cuts).into_iter();
    let mut uv = bands(&mut fs.uv_modes, st, &cuts).into_iter();
    let mut tx = bands(&mut fs.tx_types, st, &cuts).into_iter();
    let mut itx = bands(&mut fs.inter_tx_sizes, st, &cuts).into_iter();
    let mut seg = bands(&mut fs.segment_ids, st, &cuts).into_iter();
    let mut parts = Vec::with_capacity(nb);
    for _ in 0..nb {
        let (Some(mi), Some(uv_modes), Some(tx_types), Some(inter_tx_sizes), Some(segment_ids)) =
            (mi.next(), uv.next(), tx.next(), itx.next(), seg.next())
        else {
            break;
        };
        parts.push(FrameBand {
            mi,
            uv_modes,
            tx_types,
            inter_tx_sizes,
            segment_ids,
            lf_tx: lf.iter_mut().filter_map(Iterator::next).collect(),
            planes: px.iter_mut().filter_map(Iterator::next).collect(),
        });
    }
    let copy = |mut b: FrameBand<'_, P>| {
        for &(k, tile_num) in copies {
            let a = TileArea::of(hdr, tile_num);
            let from = &shells[k];
            let (r, c) = a.span((0, 0), false, st, mi_height);
            b.mi.copy(&from.fs.mi, &r, &c);
            b.uv_modes.copy(&from.fs.uv_modes, &r, &c);
            b.tx_types.copy(&from.fs.tx_types, &r, &c);
            b.inter_tx_sizes.copy(&from.fs.inter_tx_sizes, &r, &c);
            b.segment_ids.copy(&from.fs.segment_ids, &r, &c);
            for (p, band) in b.lf_tx.iter_mut().enumerate() {
                let (r, c) = a.span(plane_sub(seq, p), false, st, mi_height);
                band.copy(&from.fs.lf_tx[p], &r, &c);
            }
            for (p, (band, src)) in b.planes.iter_mut().zip(&from.pic.planes).enumerate() {
                let h = src.data.len() / src.stride;
                let (r, c) = a.span(plane_sub(seq, p), true, src.stride, h);
                band.copy(&src.data, &r, &c);
            }
        }
    };
    let mut parts = parts.into_iter();
    let first = parts.next();
    std::thread::scope(|s| {
        for b in parts {
            s.spawn(|| copy(b));
        }
        if let Some(b) = first {
            copy(b);
        }
    });
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
    let copying = std::time::Instant::now();
    if first_err.is_none() {
        copy_tiles(seq, hdr, &shells, &copies, fs, cur, threads);
    }
    super::prof::add(super::prof::TILE_COPY, copying);
    {
        let mut sp = lock(spares);
        sp.scratch.extend(own.scratch);
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

#[cfg(test)]
mod tests;
