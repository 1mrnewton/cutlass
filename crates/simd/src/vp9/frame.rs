//! A picture that one thread is still writing while others read the rows
//! it has finished, without copying them.
//!
//! The writer ([`FrameWriter`]) holds the only mutable view of the rows
//! past the published ones, and works on them in place: reconstruction,
//! intra prediction from the rows above, the loop filter. When rows are
//! final it publishes them ([`FrameWriter::publish`]) and they never change
//! again. Readers ([`SharedFrame::plane`]) see only published rows, and
//! [`SharedFrame::wait`] blocks until enough of them are. The rows a reader
//! may see and the rows the writer may still change never overlap, so no
//! lock guards the samples: the published count, stored with release
//! ordering after the writes and loaded with acquire ordering before a
//! read, orders the two.
//!
//! This module's `unsafe` is not a kernel call: it makes slices of the
//! buffers by hand, the published rows for readers and the rows past them
//! for the writer, like [`crate::shared_rows`].

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// Three planes of rows `stride` samples apart, published in luma rows; a
/// plane's rows are the luma rows shifted down by its vertical
/// subsampling.
pub struct SharedFrame<P> {
    planes: [Box<[UnsafeCell<P>]>; 3],
    /// Where each plane starts in its buffer: its first sample on a
    /// 64-byte boundary, so that the rows of an aligned block are whole
    /// cache lines (an allocator's large blocks start 16 bytes into a
    /// page, which split every other 32-byte access in two).
    starts: [usize; 3],
    strides: [usize; 3],
    rows: [usize; 3],
    shifts: [u32; 3],
    /// Luma rows `0..published` are final (`usize::MAX`: all).
    published: AtomicUsize,
    /// Per plane, the rows of the picture (0: none) when every row below
    /// them repeats their last one ([`FrameWriter::hold_edge`]).
    held: [AtomicUsize; 3],
    lock: Mutex<()>,
    ready: Condvar,
}

// SAFETY: a `SharedFrame` is only written through its one `FrameWriter`,
// and only in rows no reader can see (see the module docs); readers only
// read. Sharing it is then sharing `&[P]`, which needs `P: Sync`.
#[allow(unsafe_code)]
unsafe impl<P: Send + Sync> Sync for SharedFrame<P> {}

impl<P> SharedFrame<P> {
    /// Rows of plane `p` that luma rows `0..luma` make.
    #[inline]
    fn plane_rows(&self, p: usize, luma: usize) -> usize {
        match luma {
            usize::MAX => self.rows[p],
            n => (n >> self.shifts[p]).min(self.rows[p]),
        }
    }

    /// Luma rows published so far (`usize::MAX`: all of them).
    #[inline]
    pub fn published(&self) -> usize {
        self.published.load(Ordering::Acquire)
    }

    /// Blocks until luma rows `0..luma` are published (or every row is).
    pub fn wait(&self, luma: usize) {
        if self.published() >= luma {
            return;
        }
        let mut guard = match self.lock.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        while self.published() < luma {
            guard = match self.ready.wait(guard) {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
        }
    }

    /// The published rows of plane `p`, from its first row, and the
    /// plane's stride.
    #[inline]
    pub fn plane(&self, p: usize) -> (&[P], usize) {
        let p = p.min(2);
        let rows = self.plane_rows(p, self.published());
        let (buf, start) = (&self.planes[p], self.starts[p]);
        let n = (rows * self.strides[p]).min(buf.len() - start);
        // SAFETY: rows `0..rows` are published (`rows` is at most what
        // the acquire load saw), so the writer never writes them again;
        // `start..start + n` lies in the buffer, and `UnsafeCell<P>` has the layout of
        // `P`.
        #[allow(unsafe_code)]
        let s = unsafe {
            std::slice::from_raw_parts(UnsafeCell::raw_get(buf.as_ptr().add(start)).cast_const(), n)
        };
        (s, self.strides[p])
    }

    /// The rows of plane `p` a reader may take as the picture's `visible`
    /// rows held at their last one: every row of the buffer once the
    /// writer has repeated that row below them all
    /// ([`FrameWriter::hold_edge`]) and published every row, else
    /// `visible`.
    #[inline]
    pub fn held_rows(&self, p: usize, visible: usize) -> usize {
        let p = p.min(2);
        if visible > 0
            && self.published() == usize::MAX
            && self.held[p].load(Ordering::Relaxed) == visible
        {
            self.rows[p].max(visible)
        } else {
            visible
        }
    }

    /// The stride and allocated rows of plane `p`.
    pub fn shape(&self, p: usize) -> (usize, usize) {
        let p = p.min(2);
        (self.strides[p], self.rows[p])
    }
}

impl SharedFrame<u16> {
    /// The published rows of plane `p` as little-endian bytes, two a
    /// sample, and the plane's stride in bytes: the samples' own memory,
    /// which is little-endian on a little-endian target (`None` on any
    /// other).
    #[inline]
    pub fn plane_le_bytes(&self, p: usize) -> Option<(&[u8], usize)> {
        if cfg!(target_endian = "little") {
            let (s, stride) = self.plane(p);
            // SAFETY: `s` is a live `&[u16]` (see `plane`); its memory is
            // `2 * s.len()` initialized bytes, any byte is a valid `u8`,
            // `u8` has alignment 1, and the view borrows `self` as `s`
            // does, so the samples cannot change while it lives.
            #[allow(unsafe_code)]
            let b = unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<u8>(), 2 * s.len()) };
            Some((b, 2 * stride))
        } else {
            None
        }
    }
}

/// The one writer of a [`SharedFrame`].
pub struct FrameWriter<P> {
    frame: Arc<SharedFrame<P>>,
}

/// The writer's view of one plane: the rows from `first_row` on.
pub struct Tail<'a, P> {
    pub data: &'a mut [P],
    pub first_row: usize,
    pub stride: usize,
}

impl<P: Copy + Default> FrameWriter<P> {
    /// A frame of planes `strides[p]` wide and `rows[p]` high (`shifts`:
    /// vertical subsampling from luma), none of it published.
    pub fn new(strides: [usize; 3], rows: [usize; 3], shifts: [u32; 3]) -> Self {
        // Room to move the start up to a 64-byte boundary.
        let slack = 64 / size_of::<P>().max(1);
        let planes: [Box<[UnsafeCell<P>]>; 3] = core::array::from_fn(|p| {
            (0..strides[p] * rows[p] + slack)
                .map(|_| UnsafeCell::new(P::default()))
                .collect()
        });
        let starts = core::array::from_fn(|p| {
            // In samples; one that cannot be aligned keeps its start.
            match planes[p].as_ptr().align_offset(64) {
                skip if skip <= slack => skip,
                _ => 0,
            }
        });
        Self {
            frame: Arc::new(SharedFrame {
                planes,
                starts,
                strides,
                rows,
                shifts,
                published: AtomicUsize::new(0),
                held: [const { AtomicUsize::new(0) }; 3],
                lock: Mutex::new(()),
                ready: Condvar::new(),
            }),
        }
    }
}

impl<P> FrameWriter<P> {
    /// Writes a frame again whose last reader has gone: `Err` gives the
    /// frame back when someone still holds it.
    pub fn restart(mut frame: Arc<SharedFrame<P>>) -> Result<Self, Arc<SharedFrame<P>>> {
        match Arc::get_mut(&mut frame) {
            Some(f) => {
                *f.published.get_mut() = 0;
                for h in &mut f.held {
                    *h.get_mut() = 0;
                }
                Ok(Self { frame })
            }
            None => Err(frame),
        }
    }

    /// What readers read.
    pub fn shared(&self) -> &Arc<SharedFrame<P>> {
        &self.frame
    }

    /// The rows not yet published, of each plane.
    pub fn tails(&mut self) -> [Tail<'_, P>; 3] {
        let f = &*self.frame;
        let published = f.published.load(Ordering::Relaxed);
        let mut out = [None, None, None];
        for (p, slot) in out.iter_mut().enumerate() {
            let first = f.plane_rows(p, published);
            let (buf, start) = (&f.planes[p], f.starts[p]);
            let end = start + f.strides[p] * f.rows[p];
            let lo = start + (first * f.strides[p]).min(end - start);
            // SAFETY: rows from `first` on are past the published ones,
            // so no reader makes a slice of them; this writer is the only
            // one, and `&mut self` keeps its views from overlapping one
            // another or a later publish. `lo..end` lies in the buffer,
            // and `UnsafeCell<P>` has the layout of `P`.
            #[allow(unsafe_code)]
            let data = unsafe {
                std::slice::from_raw_parts_mut(UnsafeCell::raw_get(buf.as_ptr().add(lo)), end - lo)
            };
            *slot = Some(Tail {
                data,
                first_row: first,
                stride: f.strides[p],
            });
        }
        out.map(|t| {
            t.unwrap_or(Tail {
                data: &mut [],
                first_row: 0,
                stride: 0,
            })
        })
    }

    /// Repeats row `visible[p] - 1` of each plane into every row below it,
    /// so that readers can take the rows past the picture as its edge held
    /// ([`SharedFrame::held_rows`], once every row is published). Rows it
    /// cannot write (published ones, or a plane without such a row) are
    /// left, and that plane is not marked.
    pub fn hold_edge(&mut self, visible: [usize; 3])
    where
        P: Copy,
    {
        let all = self.frame.rows;
        let tails = self.tails();
        let mut held = [0; 3];
        for (p, t) in tails.into_iter().enumerate() {
            let v = visible[p];
            let (stride, rows) = (t.stride, all[p]);
            if v == 0 || v >= rows || v - 1 < t.first_row || stride == 0 {
                continue;
            }
            let from = (v - 1 - t.first_row) * stride;
            let end = (rows - t.first_row) * stride;
            if end > t.data.len() {
                continue;
            }
            let mut at = from + stride;
            while at < end {
                t.data.copy_within(from..from + stride, at);
                at += stride;
            }
            held[p] = v;
        }
        for (h, v) in self.frame.held.iter().zip(held) {
            // Ordered before readers by the release of the next publish.
            h.store(v, Ordering::Relaxed);
        }
    }

    /// Publishes luma rows `0..luma` (`usize::MAX`: every row). The count
    /// never goes back.
    pub fn publish(&mut self, luma: usize) {
        let f = &*self.frame;
        if luma <= f.published.load(Ordering::Relaxed) {
            return;
        }
        {
            let _guard = match f.lock.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            f.published.store(luma, Ordering::Release);
        }
        f.ready.notify_all();
    }
}

impl<P> Drop for FrameWriter<P> {
    /// A writer that stops (a failed or abandoned frame) publishes every
    /// row, so that no reader waits for ever.
    fn drop(&mut self) {
        self.publish(usize::MAX);
    }
}

#[cfg(test)]
mod tests {
    use super::FrameWriter;

    #[test]
    fn readers_see_published_rows_while_the_writer_works_below() {
        let mut w = FrameWriter::<u8>::new([4, 2, 2], [8, 4, 4], [0, 1, 1]);
        let shared = w.shared().clone();
        {
            let [y, u, _] = w.tails();
            y.data.fill(7);
            u.data.fill(9);
        }
        assert_eq!(shared.plane(0).0.len(), 0);
        w.publish(4);
        assert_eq!(shared.plane(0).0, &[7u8; 16][..]);
        assert_eq!(shared.plane(1).0, &[9u8; 4][..]);
        {
            let [y, _, _] = w.tails();
            assert_eq!((y.first_row, y.data.len()), (4, 16));
            y.data.fill(1);
        }
        let reader = std::thread::spawn(move || {
            shared.wait(8);
            shared.plane(0).0.to_vec()
        });
        drop(w);
        let all = reader.join().unwrap_or_default();
        assert_eq!(&all[..16], &[7u8; 16][..]);
        assert_eq!(&all[16..], &[1u8; 16][..]);
    }

    #[test]
    fn a_held_edge_repeats_the_last_row_below() {
        let mut w = FrameWriter::<u8>::new([4, 2, 2], [8, 4, 4], [0, 1, 1]);
        let shared = w.shared().clone();
        {
            let [y, u, v] = w.tails();
            for (i, s) in y.data.iter_mut().enumerate() {
                *s = i as u8;
            }
            u.data.fill(9);
            v.data.fill(3);
        }
        w.publish(4);
        w.hold_edge([6, 3, 4]);
        assert_eq!(shared.held_rows(0, 6), 6, "not yet published");
        w.publish(usize::MAX);
        assert_eq!(shared.held_rows(0, 6), 8);
        assert_eq!(shared.held_rows(0, 5), 5, "held for another height");
        assert_eq!(shared.held_rows(1, 3), 4);
        assert_eq!(shared.held_rows(2, 4), 4, "no row below");
        let (y, _) = shared.plane(0);
        assert_eq!(&y[20..24], &[20, 21, 22, 23]);
        assert_eq!(&y[24..32], &[20, 21, 22, 23, 20, 21, 22, 23]);
    }

    #[test]
    fn planes_start_on_a_cache_line() {
        let mut w = FrameWriter::<u8>::new([1920, 960, 960], [1088, 544, 544], [0, 1, 1]);
        for (p, t) in w.tails().iter().enumerate() {
            assert_eq!(t.data.as_ptr() as usize % 64, 0, "plane {p}");
            assert_eq!(t.data.len(), [1920 * 1088, 960 * 544, 960 * 544][p]);
        }
        let mut w = FrameWriter::<u16>::new([64, 32, 32], [64, 32, 32], [0, 1, 1]);
        for t in w.tails() {
            assert_eq!(t.data.as_ptr() as usize % 64, 0);
        }
        w.publish(usize::MAX);
        let shared = w.shared().clone();
        assert_eq!(shared.plane(0).0.as_ptr() as usize % 64, 0);
        assert_eq!(shared.plane(0).0.len(), 64 * 64);
    }

    #[test]
    fn sixteen_bit_planes_read_as_little_endian_bytes() {
        let mut w = FrameWriter::<u16>::new([3, 2, 2], [2, 1, 1], [0, 1, 1]);
        let shared = w.shared().clone();
        {
            let [y, _, _] = w.tails();
            for (i, s) in y.data.iter_mut().enumerate() {
                *s = 0x0300 + i as u16;
            }
        }
        w.publish(usize::MAX);
        let (b, stride) = shared.plane_le_bytes(0).expect("little-endian");
        assert_eq!(stride, 6);
        let want: Vec<u8> = (0..6u16).flat_map(|i| (0x0300 + i).to_le_bytes()).collect();
        assert_eq!(b, &want[..]);
    }
}
