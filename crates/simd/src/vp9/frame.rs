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
    strides: [usize; 3],
    rows: [usize; 3],
    shifts: [u32; 3],
    /// Luma rows `0..published` are final (`usize::MAX`: all).
    published: AtomicUsize,
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
        let buf = &self.planes[p];
        let n = (rows * self.strides[p]).min(buf.len());
        // SAFETY: rows `0..rows` are published (`rows` is at most what
        // the acquire load saw), so the writer never writes them again;
        // `0..n` lies in the buffer, and `UnsafeCell<P>` has the layout of
        // `P`.
        #[allow(unsafe_code)]
        let s = unsafe {
            std::slice::from_raw_parts(UnsafeCell::raw_get(buf.as_ptr()).cast_const(), n)
        };
        (s, self.strides[p])
    }

    /// The stride and allocated rows of plane `p`.
    pub fn shape(&self, p: usize) -> (usize, usize) {
        let p = p.min(2);
        (self.strides[p], self.rows[p])
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
        let planes = core::array::from_fn(|p| {
            (0..strides[p] * rows[p])
                .map(|_| UnsafeCell::new(P::default()))
                .collect()
        });
        Self {
            frame: Arc::new(SharedFrame {
                planes,
                strides,
                rows,
                shifts,
                published: AtomicUsize::new(0),
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
            let buf = &f.planes[p];
            let lo = (first * f.strides[p]).min(buf.len());
            // SAFETY: rows from `first` on are past the published ones,
            // so no reader makes a slice of them; this writer is the only
            // one, and `&mut self` keeps its views from overlapping one
            // another or a later publish. `lo..len` lies in the buffer,
            // and `UnsafeCell<P>` has the layout of `P`.
            #[allow(unsafe_code)]
            let data = unsafe {
                std::slice::from_raw_parts_mut(
                    UnsafeCell::raw_get(buf.as_ptr().add(lo)),
                    buf.len() - lo,
                )
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
}
