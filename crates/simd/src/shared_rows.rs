//! The final rows of a picture still being made, shared with the threads
//! that read them as they become final.
//!
//! One writer ([`RowsWriter`]) copies rows in, top to bottom, and then
//! publishes them: from then on they never change. Readers
//! ([`SharedRows::rows`]) see only published rows. The rows a reader may
//! see and the rows the writer may still write never overlap, so the
//! writer needs no lock against the readers: the published count, stored
//! with release ordering after the copy and loaded with acquire ordering
//! before a read, orders the two.
//!
//! The planes lie one after another in one buffer, rows end to end, which
//! for 8-bit 4:2:0 is the packed picture itself: once every row is
//! published, [`SharedRows::whole`] hands the buffer out to be read as it
//! is, and a writer fills it again only once no one else holds it.
//!
//! This is the one module of the crate whose `unsafe` is not a kernel
//! call: it makes slices of the buffer by hand, the published rows for
//! readers and the rows past them for the writer.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The buffer of the planes: plane 0's rows, then plane 1's, then plane
/// 2's, each row `width` samples.
pub type PlaneBuf<P> = Arc<Vec<P>>;

/// Rows of up to three planes, published in luma rows; a plane's rows are
/// the luma rows shifted down by its vertical subsampling.
///
/// The count of published rows, which the writer stores as it goes, lies
/// in other cache lines than the rest, which every reader reads.
#[repr(C)]
pub struct SharedRows<P> {
    /// Holds the samples; never read or written through, but by `base`.
    buf: PlaneBuf<P>,
    /// The first sample of `buf`, taken while the writer held it alone.
    base: *mut P,
    /// Where each plane starts in `buf`.
    starts: [usize; 3],
    widths: [usize; 3],
    heights: [usize; 3],
    shifts: [u32; 3],
    _apart: [Apart; 0],
    /// Luma rows `0..published` are final (`usize::MAX`: all).
    published: AtomicUsize,
}

/// Starts a new pair of cache lines in a `#[repr(C)]` struct.
#[repr(align(128))]
struct Apart;

// SAFETY: a `SharedRows` is only written through its one `RowsWriter`, and
// only in rows no reader can see yet (see the module docs); readers only
// read. Sharing or sending it is then sharing `&[P]` and moving the
// `Arc<Vec<P>>` that owns the samples, which needs `P: Send + Sync`.
#[allow(unsafe_code)]
unsafe impl<P: Send + Sync> Sync for SharedRows<P> {}
// SAFETY: as for `Sync`.
#[allow(unsafe_code)]
unsafe impl<P: Send + Sync> Send for SharedRows<P> {}

impl<P> SharedRows<P> {
    /// Rows of plane `p` that luma rows `0..luma` make.
    #[inline]
    fn plane_rows(&self, p: usize, luma: usize) -> usize {
        match luma {
            usize::MAX => self.heights[p],
            n => (n >> self.shifts[p]).min(self.heights[p]),
        }
    }

    /// Published rows of plane `p`.
    #[inline]
    fn have(&self, p: usize) -> usize {
        self.plane_rows(p, self.published.load(Ordering::Acquire))
    }

    /// Published rows `y0..` of plane `p`, from row `y0` to the last
    /// published one: `rows - y0` rows of `width` samples.
    #[inline]
    fn published_from(&self, p: usize, y0: usize, rows: usize) -> &[P] {
        let w = self.widths[p];
        let (a, b) = (y0 * w, rows * w);
        debug_assert!(a <= b && b <= w * self.heights[p]);
        let b = b.min(w * self.heights[p]);
        let a = a.min(b);
        // SAFETY: rows `0..rows` are published (`rows` is at most what
        // `have` loaded), so the writer never writes them again; `a..b`
        // lies in plane `p`, which lies in the buffer `base` points into
        // (see `RowsWriter::new`), and `buf` keeps that buffer alive.
        #[allow(unsafe_code)]
        unsafe {
            std::slice::from_raw_parts(self.base.add(self.starts[p] + a).cast_const(), b - a)
        }
    }

    /// Rows `y0..y1` of plane `p` when they are published: the samples
    /// from row `y0` on, and their stride.
    #[inline]
    pub fn rows(&self, p: usize, y0: usize, y1: usize) -> Option<(&[P], usize)> {
        let p = p.min(2);
        let have = self.have(p);
        if y1 > have || y1 <= y0 || self.widths[p] == 0 {
            return None;
        }
        Some((self.published_from(p, y0, have), self.widths[p]))
    }

    /// The published rows of plane `p` from row `y` on: their samples,
    /// their stride, and the row past the last.
    #[inline]
    pub fn rows_from(&self, p: usize, y: usize) -> Option<(&[P], usize, usize)> {
        let p = p.min(2);
        let have = self.have(p);
        if y >= have || self.widths[p] == 0 {
            return None;
        }
        Some((self.published_from(p, y, have), self.widths[p], have))
    }

    /// Luma rows published so far.
    pub fn published(&self) -> usize {
        self.published.load(Ordering::Acquire)
    }

    /// Every sample, planes one after another, once every row is
    /// published: the buffer itself, which no one writes again until
    /// every holder but this lets it go.
    pub fn whole(&self) -> Option<PlaneBuf<P>> {
        (self.published() == usize::MAX).then(|| self.buf.clone())
    }

    /// The buffer, for reuse.
    pub fn into_buffers(self) -> PlaneBuf<P> {
        self.buf
    }

    /// Rows of no samples, all published: a stand-in to move a real one
    /// out of a place that must hold one.
    pub fn empty() -> Self {
        let mut buf = Arc::new(Vec::new());
        let base = Arc::get_mut(&mut buf).map_or(std::ptr::null_mut(), Vec::as_mut_ptr);
        Self {
            buf,
            base,
            starts: [0; 3],
            widths: [0; 3],
            heights: [0; 3],
            shifts: [0; 3],
            _apart: [],
            published: AtomicUsize::new(usize::MAX),
        }
    }
}

/// The one writer of a [`SharedRows`].
pub struct RowsWriter<P> {
    shared: Arc<SharedRows<P>>,
    done: usize,
}

impl<P> RowsWriter<P> {
    /// What readers read.
    pub fn shared(&self) -> &Arc<SharedRows<P>> {
        &self.shared
    }

    /// Luma rows published so far.
    pub fn done(&self) -> usize {
        self.done
    }

    /// The rows that luma rows `done..upto` make in each plane, to write
    /// in place; [`Self::publish`] then hands them to the readers. Empty
    /// when `upto` is not past `done`.
    pub fn pending_mut(&mut self, upto: usize) -> [&mut [P]; 3] {
        let sh = &*self.shared;
        let done = self.done;
        core::array::from_fn(|p| {
            let w = sh.widths[p];
            let (a, b) = (sh.plane_rows(p, done), sh.plane_rows(p, upto));
            if w == 0 || a >= b {
                return &mut [][..];
            }
            let (lo, hi) = (a * w, (b * w).min(w * sh.heights[p]));
            // SAFETY: as in `write`: rows `a..` are past the published
            // ones, so no reader makes a slice of them, and `&mut self`
            // keeps this writer from handing them out twice. The planes
            // lie apart in the buffer, so the three slices never overlap.
            #[allow(unsafe_code)]
            unsafe {
                std::slice::from_raw_parts_mut(sh.base.add(sh.starts[p] + lo), hi - lo)
            }
        })
    }

    /// Publishes luma rows `done..upto`, written in place through
    /// [`Self::pending_mut`] (`usize::MAX`: every row).
    pub fn publish(&mut self, upto: usize) {
        if upto <= self.done {
            return;
        }
        self.done = upto;
        self.shared.published.store(upto, Ordering::Release);
    }
}

impl<P: Copy + Default> RowsWriter<P> {
    /// Rows of planes of `widths` x `heights` samples (subsampled down by
    /// `shifts` from luma), none published, in `buf` when it is large
    /// enough and no one else holds it.
    pub fn new(
        widths: [usize; 3],
        heights: [usize; 3],
        shifts: [u32; 3],
        buf: Option<PlaneBuf<P>>,
    ) -> Self {
        let sizes = [0, 1, 2].map(|p| widths[p] * heights[p]);
        let starts = [0, sizes[0], sizes[0] + sizes[1]];
        let total = starts[2] + sizes[2];
        let fresh = || Arc::new(vec![P::default(); total]);
        let mut buf = buf.filter(|b| b.len() >= total).unwrap_or_else(fresh);
        if Arc::get_mut(&mut buf).is_none() {
            buf = fresh();
        }
        // `buf` was made here or no one else holds it: this writer alone
        // writes it, through `base`, until every row is published.
        let base = Arc::get_mut(&mut buf).map_or(std::ptr::null_mut(), |v| {
            v.truncate(total);
            v.as_mut_ptr()
        });
        Self {
            shared: Arc::new(SharedRows {
                buf,
                base,
                starts,
                widths,
                heights,
                shifts,
                _apart: [],
                published: AtomicUsize::new(0),
            }),
            done: 0,
        }
    }

    /// Copies the rows that luma rows `done..upto` make in each plane
    /// from `src(p, row)` (a row of at least the plane's width), then
    /// publishes them.
    pub fn write<'s>(&mut self, upto: usize, mut src: impl FnMut(usize, usize) -> &'s [P])
    where
        P: 's,
    {
        if upto <= self.done {
            return;
        }
        let sh = &*self.shared;
        for p in 0..3 {
            let w = sh.widths[p];
            let (a, b) = (sh.plane_rows(p, self.done), sh.plane_rows(p, upto));
            if w == 0 || a >= b {
                continue;
            }
            let (lo, hi) = (a * w, (b * w).min(w * sh.heights[p]));
            // SAFETY: rows `a..` are past the published ones (`a` is what
            // `done`, the published count, makes), so no reader makes a
            // slice of them; this writer is the only one, and `&mut self`
            // keeps it from overlapping itself. `lo..hi` lies in plane
            // `p`, inside the buffer `base` points into.
            #[allow(unsafe_code)]
            let dst =
                unsafe { std::slice::from_raw_parts_mut(sh.base.add(sh.starts[p] + lo), hi - lo) };
            for (r, d) in (a..b).zip(dst.chunks_exact_mut(w)) {
                d.copy_from_slice(&src(p, r)[..w]);
            }
        }
        self.done = upto;
        sh.published.store(upto, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::RowsWriter;

    #[test]
    fn rows_written_in_place_are_seen_once_published() {
        let mut w = RowsWriter::<u16>::new([4, 2, 2], [8, 4, 4], [0, 1, 1], None);
        let s = w.shared().clone();
        {
            let [y, u, v] = w.pending_mut(4);
            assert_eq!((y.len(), u.len(), v.len()), (16, 4, 4));
            y.fill(7);
            u.fill(8);
            v.fill(9);
        }
        assert!(s.rows(0, 0, 1).is_none());
        w.publish(4);
        assert_eq!(s.rows(0, 3, 4).map(|(r, _)| r[0]), Some(7));
        assert_eq!(s.rows(2, 1, 2).map(|(r, _)| r[1]), Some(9));
        let [y, u, _] = w.pending_mut(8);
        assert_eq!((y.len(), u.len()), (16, 4));
    }

    #[test]
    fn readers_see_only_published_rows() {
        let rows: Vec<Vec<u8>> = (0..64).map(|r| vec![r as u8; 8]).collect();
        let crow: Vec<Vec<u8>> = (0..32).map(|r| vec![100 + r as u8; 4]).collect();
        let mut w = RowsWriter::<u8>::new([8, 4, 4], [64, 32, 32], [0, 1, 1], None);
        let s = w.shared().clone();
        assert!(s.rows(0, 0, 1).is_none());
        w.write(16, |p, r| if p == 0 { &rows[r] } else { &crow[r] });
        let (l, st) = s.rows(0, 4, 16).unwrap_or_default();
        assert_eq!((st, l.len(), l[0], l[l.len() - 1]), (8, 12 * 8, 4, 15));
        assert!(s.rows(0, 4, 17).is_none());
        assert_eq!(
            s.rows(1, 0, 8).map(|(c, _)| (c.len(), c[0], c[31])),
            Some((32, 100, 107))
        );
        assert!(s.rows(2, 0, 9).is_none());
        w.write(usize::MAX, |p, r| if p == 0 { &rows[r] } else { &crow[r] });
        assert!(w.pending_mut(usize::MAX).iter().all(|p| p.is_empty()));
        assert_eq!(
            s.rows_from(0, 60).map(|(l, _, end)| (l[0], end)),
            Some((60, 64))
        );
        assert_eq!(
            s.rows_from(2, 31).map(|(c, _, end)| (c[0], end)),
            Some((131, 32))
        );
    }

    #[test]
    fn the_whole_buffer_is_shared_once_published_and_not_refilled_while_held() {
        let mut w = RowsWriter::<u8>::new([4, 2, 2], [4, 2, 2], [0, 1, 1], None);
        assert!(w.shared().whole().is_none());
        for (p, plane) in w.pending_mut(usize::MAX).into_iter().enumerate() {
            plane.fill(p as u8 + 1);
        }
        w.publish(usize::MAX);
        let whole = w.shared().whole().unwrap_or_default();
        assert_eq!(
            &whole[..],
            &[1; 16]
                .iter()
                .chain(&[2; 4])
                .chain(&[3; 4])
                .copied()
                .collect::<Vec<u8>>()[..]
        );
        let shared = w.shared().clone();
        drop(w);
        let buf = std::sync::Arc::try_unwrap(shared)
            .ok()
            .map(super::SharedRows::into_buffers);
        // Still held by `whole`: a new writer takes other memory.
        let again = RowsWriter::<u8>::new([4, 2, 2], [4, 2, 2], [0, 1, 1], buf.clone());
        assert!(!std::sync::Arc::ptr_eq(
            &again.shared().buf,
            buf.as_ref().unwrap_or(&whole)
        ));
        assert_eq!(whole[0], 1);
        drop((again, whole));
        // Let go of: the next writer fills the same memory.
        let at = buf.as_ref().map(|b| b.as_ptr());
        let reused = RowsWriter::<u8>::new([4, 2, 2], [4, 2, 2], [0, 1, 1], buf);
        assert_eq!(Some(reused.shared().buf.as_ptr()), at);
    }
}
