//! A buffer one writer fills from the front while other threads read the
//! part already filled: how a picture still being decoded on one thread
//! hands its final rows to the threads decoding later pictures, which read
//! them as one plain slice, as FFmpeg's frame threads read a reference's
//! rows straight from its buffer once its progress counter has passed
//! them.
//!
//! # Memory safety
//!
//! The buffer's first `filled` items are written before `filled` is raised
//! past them, with a release store, and are never written again. A reader
//! loads `filled` with an acquire and is given only those items, so it
//! sees them whole. The writer only ever writes past `filled`, and a lock
//! keeps two writers apart. Items are `Copy`: nothing is dropped, and an
//! item never written is never read.

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A buffer of a fixed number of items, filled from the front.
pub struct Growing<T> {
    cells: Box<[UnsafeCell<MaybeUninit<T>>]>,
    /// How many items from the front are written, and final.
    filled: AtomicUsize,
    /// Held while items are written.
    writer: Mutex<()>,
}

// SAFETY: readers share only the filled items, which no one writes again;
// the one writer writes past them, under the lock. See the module's notes.
#[allow(unsafe_code)]
unsafe impl<T: Copy + Send + Sync> Sync for Growing<T> {}

impl<T: Copy> Growing<T> {
    /// Room for `len` items, none filled.
    pub fn new(len: usize) -> Self {
        let uninit: Box<[MaybeUninit<T>]> = Box::new_uninit_slice(len);
        let raw = Box::into_raw(uninit) as *mut [UnsafeCell<MaybeUninit<T>>];
        // SAFETY: `UnsafeCell` has the layout of what it holds, so this is
        // the same allocation seen with the same layout.
        #[allow(unsafe_code)]
        let cells = unsafe { Box::from_raw(raw) };
        Self {
            cells,
            filled: AtomicUsize::new(0),
            writer: Mutex::new(()),
        }
    }

    /// How many items the buffer holds when it is full.
    pub fn capacity(&self) -> usize {
        self.cells.len()
    }

    /// The items filled so far, from the front.
    #[inline]
    pub fn filled(&self) -> &[T] {
        let filled = self.filled.load(Ordering::Acquire);
        // SAFETY: the first `filled` items were written before the release
        // store this load saw, and are never written again.
        #[allow(unsafe_code)]
        unsafe {
            core::slice::from_raw_parts(self.cells.as_ptr().cast::<T>(), filled)
        }
    }

    /// Writes `items` after the filled ones and counts them in, as many as
    /// there is room for. Gives how many went in.
    pub fn push(&self, items: &[T]) -> usize {
        // The lock is only held across a copy, which cannot panic.
        let _writing = self
            .writer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let start = self.filled.load(Ordering::Relaxed);
        let count = items.len().min(self.cells.len() - start);
        if count == 0 {
            return 0;
        }
        let first = UnsafeCell::raw_get(&raw const self.cells[start]).cast::<T>();
        // SAFETY: items `start..start + count` are inside the buffer and
        // past the filled ones, which is all any reader is given; the lock
        // keeps any other writer out. `items` is a different allocation.
        #[allow(unsafe_code)]
        unsafe {
            core::ptr::copy_nonoverlapping(items.as_ptr(), first, count);
        }
        self.filled.store(start + count, Ordering::Release);
        count
    }
}

#[cfg(test)]
mod tests {
    use super::Growing;

    #[test]
    fn fills_from_the_front_and_stops_when_full() {
        let growing = Growing::<u8>::new(10);
        assert_eq!(growing.capacity(), 10);
        assert!(growing.filled().is_empty());
        assert_eq!(growing.push(&[1, 2, 3]), 3);
        assert_eq!(growing.filled(), &[1, 2, 3]);
        assert_eq!(growing.push(&[]), 0);
        assert_eq!(growing.push(&[4, 5, 6, 7, 8, 9, 10, 11, 12]), 7);
        assert_eq!(growing.filled(), &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        assert_eq!(growing.push(&[13]), 0);
        assert!(Growing::<u16>::new(0).filled().is_empty());
    }

    #[test]
    fn readers_on_other_threads_see_whole_items() {
        // A reader may look at any moment; whatever it is given must be
        // exactly what was written, in order.
        let growing = std::sync::Arc::new(Growing::<u32>::new(1 << 16));
        let readers: Vec<_> = (0..3)
            .map(|_| {
                let growing = growing.clone();
                std::thread::spawn(move || {
                    let mut seen = 0;
                    while seen < growing.capacity() {
                        let filled = growing.filled();
                        assert!(filled.len() >= seen);
                        for (index, &item) in filled.iter().enumerate().skip(seen) {
                            assert_eq!(item, (index as u32).wrapping_mul(2_654_435_761));
                        }
                        seen = filled.len();
                    }
                })
            })
            .collect();
        let items: Vec<u32> = (0..1u32 << 16)
            .map(|index| index.wrapping_mul(2_654_435_761))
            .collect();
        for chunk in items.chunks(37) {
            growing.push(chunk);
        }
        for reader in readers {
            assert!(reader.join().is_ok());
        }
    }
}
