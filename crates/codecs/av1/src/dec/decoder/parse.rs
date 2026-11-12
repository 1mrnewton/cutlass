//! Parsing the tiles of a frame on several threads, for the frames whose
//! reconstruction and filters run behind the parse (see
//! [`super::Backend`]).
//!
//! Tiles parse independently (see [`super::tiles`]), each thread into a
//! frame state of its own. Nothing is copied back: at the end of each
//! superblock row of its tile, a thread hands over the tile's part of
//! what the rest of the decoder reads, and a [`Sink`] sends whole rows on
//! in order.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

use cutlass_decode::CodecError;

use super::super::cdf::CdfContext;
use super::super::frame::{FrameState, MF_BAND_ROWS, MfBand, Picture, RefData};
use super::super::obu::{FrameHeader, SequenceHeader};
use super::super::recon::Scratch;
use super::super::tile::ReconLog;
use super::{
    FrameOut, LOG_RESERVE_BLOCKS, MfFeed, MfShared, PicturePixel, R, Rect, RefHeaderState, RowMsg,
    RowPools, Spares, TileBufs, decode_tile, lock, malformed, mf_band, prof,
};

/// What ends one tile's part of a superblock row.
struct RowEnd {
    msg: RowMsg,
    /// The saved motion of the row's bands, in the tile's columns.
    motion: Vec<MfBand>,
    /// The segment ids of the row, in the tile's columns.
    seg: Rect<u8>,
}

/// One tile column's part of a superblock row: the logs of its blocks,
/// and its end once the tile has parsed the whole row.
#[derive(Default)]
struct Held {
    logs: Vec<ReconLog>,
    end: Option<RowEnd>,
}

/// Collects the parts of superblock rows the tile parsers finish, in any
/// order, and sends each row on once every tile column has ended it and
/// the rows above have gone: its logs to reconstruction column by column,
/// its saved motion to the frames that project it.
struct Sink<'a> {
    tx: &'a Sender<(Option<RowMsg>, ReconLog)>,
    cols: usize,
    /// First luma 4x4 row not sent yet.
    next: usize,
    held: BTreeMap<usize, Vec<Held>>,
    data: Option<&'a RefData>,
    /// The frame's segment ids, gathered for `data`.
    seg: Vec<u8>,
    seg_stride: usize,
}

impl Sink<'_> {
    /// Takes logs of tile column `col` in the row starting at luma 4x4
    /// row `r0`, and the row's end when they are the tile's last there.
    fn push(&mut self, col: usize, r0: usize, logs: &mut Vec<ReconLog>, end: Option<RowEnd>) {
        if self.cols == 1 && end.is_none() && r0 == self.next {
            // Nothing to wait for: reconstruction starts on the row
            // while its tile is still parsing it.
            for log in logs.drain(..) {
                let _ = self.tx.send((None, log));
            }
            return;
        }
        let cols = self.cols;
        let row = self
            .held
            .entry(r0)
            .or_insert_with(|| (0..cols).map(|_| Held::default()).collect());
        let Some(h) = row.get_mut(col) else {
            return;
        };
        h.logs.append(logs);
        if end.is_some() {
            h.end = end;
            self.flush();
        }
    }

    /// Sends every whole row from `next` on.
    fn flush(&mut self) {
        loop {
            let whole = self
                .held
                .get(&self.next)
                .is_some_and(|row| row.iter().all(|h| h.end.is_some()));
            if !whole {
                return;
            }
            let Some(row) = self.held.remove(&self.next) else {
                return;
            };
            let r0 = self.next;
            let mut motion: Vec<Vec<MfBand>> = Vec::with_capacity(self.cols);
            let last_col = row.len().saturating_sub(1);
            for (c, h) in row.into_iter().enumerate() {
                let Some(RowEnd {
                    mut msg,
                    motion: m,
                    seg,
                }) = h.end
                else {
                    return;
                };
                self.next = msg.r1;
                msg.row_end = c == last_col;
                seg.put(&mut self.seg, self.seg_stride);
                motion.push(m);
                // A send only fails once reconstruction stopped (it
                // failed): the job reports that.
                let mut msg = Some(msg);
                let n = h.logs.len();
                for (i, log) in h.logs.into_iter().enumerate() {
                    let m = if i + 1 == n { msg.take() } else { None };
                    let _ = self.tx.send((m, log));
                }
            }
            if let Some(rd) = self.data {
                let k0 = r0 / (2 * MF_BAND_ROWS);
                let n = motion.iter().map(Vec::len).min().unwrap_or(0);
                for b in 0..n {
                    let rows = rd.band_rows(k0 + b);
                    let mut band = MfBand::default();
                    let n: usize = motion.iter().map(|part| part[b].refs.len()).sum();
                    band.refs.reserve_exact(n);
                    band.mvs
                        .reserve_exact(motion.iter().map(|part| part[b].mvs.len()).sum());
                    for y in 0..rows {
                        for part in &motion {
                            let w = part[b].refs.len() / rows.max(1);
                            band.refs
                                .extend_from_slice(&part[b].refs[y * w..(y + 1) * w]);
                            band.mvs.extend_from_slice(&part[b].mvs[y * w..(y + 1) * w]);
                        }
                    }
                    rd.set_band(k0 + b, band);
                }
            }
        }
    }
}

/// A tile's place among the tile columns: its luma 4x4 columns, and
/// whether it is the last column (which also writes past the frame edge).
struct Column {
    index: usize,
    cols: Range<usize>,
    last: bool,
}

impl Column {
    fn of(hdr: &FrameHeader, tile_num: usize) -> Self {
        let ti = &hdr.tile_info;
        let index = tile_num % ti.cols.max(1);
        Self {
            index,
            cols: ti.mi_col_starts[index] as usize..ti.mi_col_starts[index + 1] as usize,
            last: index + 1 == ti.cols,
        }
    }
}

/// The end of the row `r0..r1` of the tile in `col`, read from the frame
/// state `fs` its parser wrote, in the message `msg`.
fn row_end(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &FrameState,
    col: &Column,
    r0: usize,
    r1: usize,
    mut msg: RowMsg,
    saves: bool,
) -> RowEnd {
    msg.fill_meta(seq, hdr, fs, r0, r1, col.cols.clone(), col.last);
    let mut motion = Vec::new();
    let mut seg = Rect::default();
    if saves {
        let w8 = fs.mi_cols >> 1;
        let x8 = (col.cols.start >> 1).min(w8)..if col.last {
            w8
        } else {
            (col.cols.end >> 1).min(w8)
        };
        let per = 2 * MF_BAND_ROWS;
        for k in r0 / per..r1.min(fs.mi_rows).div_ceil(per) {
            motion.push(mf_band(seq, hdr, fs, k, x8.clone()));
        }
        seg.take(
            &fs.segment_ids,
            fs.mi_stride,
            r0..r1.min(fs.mi_rows),
            col.cols.start..col.cols.end.min(fs.mi_cols),
        );
    }
    RowEnd { msg, motion, seg }
}

/// Most threads that parse the tiles of one frame. A frame's parse then
/// takes a quarter of its time, which is as fast as the frames behind it
/// can use the threads.
const PARSE_THREADS: usize = 4;

/// Spare logs and row messages a tile parser takes from the pool at once:
/// the pool is shared by every frame being decoded, and a lock for each
/// log made the parsers wait on each other.
const POOL_BATCH: usize = 32;

/// The logs and messages a tile parser holds: spares from the pool, and
/// the logs of the superblock row it is parsing.
#[derive(Default)]
struct Stash {
    logs: Vec<ReconLog>,
    msgs: Vec<RowMsg>,
    pending: Vec<ReconLog>,
}

impl Stash {
    fn log(&mut self, pool: &Mutex<RowPools>) -> ReconLog {
        if self.logs.is_empty() {
            let mut pool = lock(pool);
            let keep = pool.logs.len().saturating_sub(POOL_BATCH);
            self.logs.extend(pool.logs.drain(keep..));
        }
        self.logs.pop().unwrap_or_else(|| {
            prof::count(prof::NEW_LOG);
            ReconLog::default()
        })
    }

    fn msg(&mut self, pool: &Mutex<RowPools>) -> RowMsg {
        if self.msgs.is_empty() {
            let mut pool = lock(pool);
            let keep = pool.msgs.len().saturating_sub(POOL_BATCH);
            self.msgs.extend(pool.msgs.drain(keep..));
        }
        self.msgs.pop().unwrap_or_default()
    }

    /// Gives the unused spares back.
    fn give_back(&mut self, pool: &Mutex<RowPools>) {
        let mut pool = lock(pool);
        pool.logs.append(&mut self.logs);
        pool.msgs.append(&mut self.msgs);
    }
}

/// Parses the tiles of a frame on up to `threads` threads, handing each
/// superblock row to `tx` once all its tiles have parsed it (see
/// [`Sink`]), and to `out` the CDFs of the context tile as soon as that
/// tile ends and the frame's saved motion row by row.
///
/// The next frame's parse waits for the context tile, so this thread
/// starts on it at once, into `fs`, while another starts the rest. The
/// other threads parse into frame states from the pool of `spares`.
/// Returns the error of the first tile that fails, else the CDFs of the
/// context tile.
pub(super) fn parse_tiles<P: PicturePixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &mut FrameState,
    tiles: &[(usize, Vec<u8>)],
    cdf: &CdfContext,
    refs: &[Option<&Picture<P>>; 8],
    ref_hdr: &[RefHeaderState; 8],
    rows: &Mutex<RowPools>,
    spares: &Mutex<Spares<P>>,
    field: Option<&Arc<MfShared>>,
    threads: usize,
    tx: &Sender<(Option<RowMsg>, ReconLog)>,
    out: &FrameOut<'_>,
) -> R<Box<CdfContext>> {
    let ctx = hdr.tile_info.context_update_tile_id as usize;
    // The parsers read the motion field where its thread puts it.
    fs.motion_field.clear();
    fs.motion_shared = field.cloned();
    let prev_ids = fs.prev_segment_ids.clone();
    let (mi_cols, mi_rows) = (fs.mi_cols, fs.mi_rows);
    // A pooled frame state for every thread but this one, each made
    // ready by its thread.
    let helpers = threads
        .min(tiles.len())
        .min(PARSE_THREADS)
        .saturating_sub(1);
    let pooled: Vec<Option<FrameState>> = {
        let mut sp = lock(spares);
        (0..helpers)
            .map(|_| {
                let i = sp
                    .states
                    .iter()
                    .position(|f| f.mi_cols == mi_cols && f.mi_rows == mi_rows);
                i.map(|i| sp.states.swap_remove(i))
            })
            .collect()
    };
    let sink = Mutex::new(Sink {
        tx,
        cols: hdr.tile_info.cols.max(1),
        next: 0,
        held: BTreeMap::new(),
        data: out.data,
        seg: vec![
            0;
            if out.data.is_some() {
                fs.segment_ids.len()
            } else {
                0
            }
        ],
        seg_stride: fs.mi_stride,
    });
    // The tiles in the order they are taken: the context tile first.
    let mut order: Vec<usize> = (0..tiles.len()).collect();
    if let Some(at) = tiles.iter().position(|(t, _)| *t == ctx) {
        order[..=at].rotate_right(1);
    }
    let order = &order;
    let next = AtomicUsize::new(0);
    // The first tile to fail, by its place in the frame: the tiles after
    // it need not be parsed, the ones before it may fail before it.
    let failed = AtomicUsize::new(usize::MAX);
    let one_column = hdr.tile_info.cols <= 1;
    let work = |fs: &mut FrameState| {
        // A tile's rows are parsed in order and each is done with once
        // the row below it is (see `row_end`): the mode info goes round a
        // ring that stays in the cache.
        fs.mi_ring = super::MI_RING - 1;
        let mut bufs = TileBufs::new(fs, Scratch::<P>::parsing());
        let mut no_pixels = Picture::<P> {
            planes: Vec::new(),
            bd: seq.bit_depth,
            sub_x: seq.subsampling_x,
            sub_y: seq.subsampling_y,
        };
        let mut feed = field.map(|f| MfFeed {
            shared: f.clone(),
            have: 0,
        });
        let mut stash = Stash::default();
        let mut done = Vec::new();
        loop {
            let Some(&i) = order.get(next.fetch_add(1, Ordering::Relaxed)) else {
                stash.give_back(rows);
                return done;
            };
            if i > failed.load(Ordering::Relaxed) {
                continue;
            }
            let (tile_num, data) = &tiles[i];
            let col = Column::of(hdr, *tile_num);
            let res = decode_tile(
                seq,
                hdr,
                fs,
                &mut no_pixels,
                cdf,
                refs,
                ref_hdr,
                *tile_num,
                data,
                &mut bufs,
                Some(rows),
                feed.as_mut(),
                &mut |fs: &FrameState,
                      _: &Picture<P>,
                      log: Option<&mut ReconLog>,
                      r0: usize,
                      r1: usize,
                      done: bool| {
                    // With other tile columns the row waits for them
                    // anyway: the tile keeps one log for the row.
                    if !done && !one_column {
                        return;
                    }
                    let mut spare = stash.log(rows);
                    // Room for a typical chunk, so that logs rarely grow
                    // while the parser fills them.
                    spare.blocks.reserve(LOG_RESERVE_BLOCKS);
                    spare.ops.reserve(4 * LOG_RESERVE_BLOCKS);
                    spare.coefs.reserve(32 * LOG_RESERVE_BLOCKS);
                    stash
                        .pending
                        .push(log.map_or_else(ReconLog::default, |l| std::mem::replace(l, spare)));
                    let end = done.then(|| {
                        let msg = stash.msg(rows);
                        row_end(seq, hdr, fs, &col, r0, r1, msg, out.data.is_some())
                    });
                    lock(&sink).push(col.index, r0, &mut stash.pending, end);
                },
            );
            match &res {
                // The next frame's parse starts from these.
                Ok(c) if *tile_num == ctx => out.save_cdf(c),
                Ok(_) => {}
                Err(_) => {
                    failed.fetch_min(i, Ordering::Relaxed);
                }
            }
            done.push((i, res));
        }
    };
    // A helper readies its frame state, parses, and gives the state back.
    let help = |old: Option<FrameState>| {
        if old.is_none() {
            prof::count(prof::NEW_STATE);
        }
        let mut st = FrameState::reuse(old, seq, hdr);
        st.motion_shared = field.cloned();
        st.prev_segment_ids.clone_from(&prev_ids);
        let done = work(&mut st);
        st.motion_shared = None;
        lock(spares).states.push(st);
        done
    };
    let (mine, theirs) = std::thread::scope(|s| {
        let (work, help) = (&work, &help);
        // One spawn before this thread starts on the context tile; the
        // first helper starts the others.
        let mut pooled = pooled.into_iter();
        let first = pooled.next().map(|old| {
            s.spawn(move || {
                let others: Vec<_> = pooled.map(|old| s.spawn(move || help(old))).collect();
                let mut all = vec![Some(help(old))];
                all.extend(others.into_iter().map(|h| h.join().ok()));
                all
            })
        });
        let mine = work(fs);
        fs.motion_shared = None;
        let theirs = match first.map(|h| h.join()) {
            None => Vec::new(),
            Some(Ok(all)) => all,
            Some(Err(_)) => vec![None],
        };
        (mine, theirs)
    });
    let seg = std::mem::take(&mut lock(&sink).seg);
    let mut first_err: Option<(usize, CodecError)> = None;
    let mut saved = None;
    for done in std::iter::once(Some(mine)).chain(theirs) {
        let Some(done) = done else {
            // A thread that panicked: the frame fails as a tile would.
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
                }
            }
        }
    }
    if let Some((_, e)) = first_err {
        return Err(e);
    }
    if let Some(rd) = out.data {
        rd.set_segment_ids(seg);
    }
    Ok(saved.unwrap_or_else(|| Box::new(cdf.clone())))
}

#[cfg(test)]
mod tests;
