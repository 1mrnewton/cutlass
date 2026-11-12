//! A picture's finished rows, handed out while the rest is still being
//! decoded.
//!
//! One thread decodes a picture into planes only it holds. As rows become
//! final it copies them here, once each, into a buffer per plane that grows
//! from the top, and threads decoding later pictures read that buffer for
//! inter prediction as they would a whole picture: one slice, rows a
//! picture's width apart. A reader waits until every row its block reads
//! is in. This is how FFmpeg's frame threads share a reference, which they
//! read from its own buffer once its progress has passed their rows.

use std::sync::{Condvar, Mutex, OnceLock};

use cutlass_h264_kernels::growing::Growing;

use cutlass_decode::CodecError;

use super::{Colocated, Frame, Mb, Shared, Spares};
use crate::recon::Sample;

/// Deblocking macroblock row `e` changes up to three luma rows above it,
/// so with `e` rows done the luma rows before `16e − 3` are final.
const LUMA_UNSETTLED: usize = 3;
/// Deblocking changes one chroma row above a macroblock row, so with `e`
/// rows done the chroma rows before `8e − 1` are final.
const CHROMA_UNSETTLED: usize = 1;

/// A finished picture's planes and macroblocks.
pub(crate) struct Planes<S: Sample> {
    pub y: Shared<S>,
    pub u: Shared<S>,
    pub v: Shared<S>,
    pub mb: Shared<Mb>,
    /// The planes went out as the picture's output, and are empty here. A
    /// reference is then read through the rows it published, all in.
    pub handed_out: bool,
    /// The picture in its output form, made by the thread that decoded
    /// it, until it is handed over.
    pub output: Mutex<Option<cutlass_decode::VideoOutput>>,
}

/// How many threads may be decoding at once. A thread takes a permit to
/// decode and gives it up while it waits for another picture's rows, so a
/// waiting picture costs no core and another can use it.
pub(crate) struct Permits {
    free: Mutex<usize>,
    freed: Condvar,
}

impl Permits {
    pub(crate) fn new(count: usize) -> Self {
        Self {
            free: Mutex::new(count.max(1)),
            freed: Condvar::new(),
        }
    }

    pub(crate) fn take(&self) {
        let Ok(mut free) = self.free.lock() else {
            return;
        };
        while *free == 0 {
            match self.freed.wait(free) {
                Ok(woken) => free = woken,
                Err(_) => return,
            }
        }
        *free -= 1;
    }

    pub(crate) fn give(&self) {
        if let Ok(mut free) = self.free.lock() {
            *free += 1;
        }
        self.freed.notify_one();
    }
}

pub(crate) struct Published<S: Sample> {
    /// Luma width and height.
    w: usize,
    h: usize,
    mb_w: usize,
    /// The planes and the macroblocks, row by row. Once every row is in, a
    /// finished reference is read from these alone: see
    /// [`Planes::handed_out`].
    luma: Growing<S>,
    cb: Growing<S>,
    cr: Growing<S>,
    mbs: Growing<Colocated>,
    /// The whole picture, or why it has none.
    done: OnceLock<Result<Planes<S>, CodecError>>,
    gate: Mutex<()>,
    changed: Condvar,
    /// The decoding threads' permits, when they have them. A reader that
    /// has to wait here is one of those threads.
    permits: Option<std::sync::Arc<Permits>>,
    /// Where the planes' buffers go once every row is in, for a later
    /// picture's planes.
    home: Option<std::sync::Arc<Spares>>,
}

impl<S: Sample> Drop for Published<S> {
    fn drop(&mut self) {
        let Some(home) = self.home.take() else {
            return;
        };
        fn give<T: Copy + Send + 'static>(home: &Spares, growing: &mut Growing<T>) {
            let full = growing.filled().len() == growing.capacity();
            let held = std::mem::replace(growing, Growing::new(0)).into_vec();
            if full && !held.is_empty() {
                home.keep(held);
            }
        }
        give(&home, &mut self.luma);
        give(&home, &mut self.cb);
        give(&home, &mut self.cr);
    }
}

/// A block of a published plane: the rows of the plane that are in, the
/// distance between rows, the plane's size, and the block's place.
pub(crate) struct Window<'a, S> {
    pub samples: &'a [S],
    pub stride: usize,
    pub rows: usize,
    pub x: i32,
    pub y: i32,
}

fn broken() -> CodecError {
    CodecError::Malformed {
        at_bit: 0,
        what: "reference picture",
    }
}

impl<S: Sample> Published<S> {
    /// For a 4:2:0 frame of `w`×`h` luma samples, `h` a multiple of 16.
    /// Its buffers are made anew, which costs nothing as nothing is written
    /// in them first, and are given to `home` once every row is in and no
    /// one reads them: a picture's planes, which are written in full first,
    /// take them from there.
    pub(crate) fn new(
        w: usize,
        h: usize,
        mb_w: usize,
        permits: Option<std::sync::Arc<Permits>>,
        home: Option<std::sync::Arc<Spares>>,
    ) -> Self {
        let chroma = (w / 2).saturating_mul(h / 2);
        Self {
            w,
            h,
            mb_w,
            luma: Growing::new(w.saturating_mul(h)),
            cb: Growing::new(chroma),
            cr: Growing::new(chroma),
            mbs: Growing::new(mb_w.saturating_mul(h / 16)),
            done: OnceLock::new(),
            gate: Mutex::new(()),
            changed: Condvar::new(),
            permits,
            home,
        }
    }

    fn wake(&self) {
        // Taking the gate first means a reader that has just looked and
        // found nothing is already waiting when this wakes it.
        drop(self.gate.lock());
        self.changed.notify_all();
    }

    /// The rows of `plane` that are in, once there are `rows` of `width`.
    /// Inline, as it is asked for every block a later picture predicts,
    /// and nearly always the rows are in.
    #[inline(always)]
    fn wait_rows<'a, T: Copy>(
        &'a self,
        plane: &'a Growing<T>,
        width: usize,
        rows: usize,
    ) -> Result<&'a [T], CodecError> {
        let wanted = rows.saturating_mul(width);
        let filled = plane.filled();
        if filled.len() >= wanted {
            return Ok(filled);
        }
        self.wait_rows_slow(plane, wanted)
    }

    #[cold]
    #[inline(never)]
    fn wait_rows_slow<'a, T: Copy>(
        &'a self,
        plane: &'a Growing<T>,
        wanted: usize,
    ) -> Result<&'a [T], CodecError> {
        if let Some(permits) = &self.permits {
            permits.give();
        }
        let found = self.wait_rows_parked(plane, wanted);
        if let Some(permits) = &self.permits {
            permits.take();
        }
        found
    }

    fn wait_rows_parked<'a, T: Copy>(
        &'a self,
        plane: &'a Growing<T>,
        wanted: usize,
    ) -> Result<&'a [T], CodecError> {
        let mut guard = self.gate.lock().map_err(|_| broken())?;
        loop {
            let filled = plane.filled();
            if filled.len() >= wanted {
                return Ok(filled);
            }
            match self.done.get() {
                Some(Err(error)) => return Err(*error),
                // A finished picture has every row in.
                Some(Ok(_)) => return Err(broken()),
                None => {}
            }
            guard = self.changed.wait(guard).map_err(|_| broken())?;
        }
    }

    /// Publishes what `deblocked` finished macroblock rows of `pic` make
    /// final, or everything when `last` is set. `from` is how many rows
    /// were published before; the new count comes back.
    pub(crate) fn publish(
        &self,
        pic: &Frame<S>,
        from: usize,
        deblocked: usize,
        last: bool,
    ) -> usize {
        let rows = self.h / 16;
        let done = if last { rows } else { deblocked.min(rows) };
        let (first, end) = (from.min(done) * self.mb_w, done * self.mb_w);
        if let Some(mbs) = pic.mb.get(first..end.min(pic.mb.len())) {
            let cols: Vec<Colocated> = mbs.iter().map(|mb| mb.col).collect();
            self.mbs.push(&cols);
        }
        let (luma, chroma) = if last {
            (self.h, self.h / 2)
        } else {
            (
                (16 * done).saturating_sub(LUMA_UNSETTLED),
                (8 * done).saturating_sub(CHROMA_UNSETTLED),
            )
        };
        self.fill_rows((&pic.y, &pic.u, &pic.v), luma, chroma);
        self.wake();
        done
    }

    /// Puts in the rows of the three planes before luma row `luma` and
    /// chroma row `chroma` that are not in yet.
    fn fill_rows(&self, planes: (&[S], &[S], &[S]), luma: usize, chroma: usize) {
        let (y, u, v) = planes;
        let cw = self.w / 2;
        for (growing, plane, end) in [
            (&self.luma, y, luma.saturating_mul(self.w)),
            (&self.cb, u, chroma.saturating_mul(cw)),
            (&self.cr, v, chroma.saturating_mul(cw)),
        ] {
            let start = growing.filled().len();
            if let Some(rows) = plane.get(start..end.min(plane.len())) {
                growing.push(rows);
            }
        }
    }

    /// Ends the picture: its planes, or the error that stopped it. Every
    /// reader still waiting is let go.
    pub(crate) fn finish(&self, result: Result<Planes<S>, CodecError>) {
        let _ = self.done.set(result);
        self.wake();
    }

    pub(crate) fn finished(&self) -> bool {
        self.done.get().is_some()
    }

    /// The error that ended the picture, if one did.
    pub(crate) fn failure(&self) -> Option<CodecError> {
        match self.done.get() {
            Some(Err(error)) => Some(*error),
            _ => None,
        }
    }

    /// Waits for the whole picture.
    pub(crate) fn planes(&self) -> Result<&Planes<S>, CodecError> {
        let mut guard = self.gate.lock().map_err(|_| broken())?;
        loop {
            match self.done.get() {
                Some(Ok(planes)) => return Ok(planes),
                Some(Err(error)) => return Err(*error),
                None => guard = self.changed.wait(guard).map_err(|_| broken())?,
            }
        }
    }

    /// The luma samples for a block at most 16 wide and `height` rows
    /// tall, at most 16, whose first full sample is at (`x`, `y`), which
    /// may lie outside the picture. Waits until the last row the block can
    /// read, three below its own, is in.
    pub(crate) fn luma(&self, x: i32, y: i32, height: usize) -> Result<Window<'_, S>, CodecError> {
        let rows = needed(y, height.clamp(1, 16) + 3, self.h);
        Ok(Window {
            samples: self.wait_rows(&self.luma, self.w, rows)?,
            stride: self.w,
            rows: self.h,
            x,
            y,
        })
    }

    /// The chroma samples for a block at most 8 wide and `height` rows
    /// tall, at most 8, at (`x`, `y`) in the chroma plane, Cb when `cb` is
    /// set. Waits until the row below the block is in.
    pub(crate) fn chroma(
        &self,
        cb: bool,
        x: i32,
        y: i32,
        height: usize,
    ) -> Result<Window<'_, S>, CodecError> {
        let (cw, ch) = (self.w / 2, self.h / 2);
        let rows = needed(y, height.clamp(1, 8) + 1, ch);
        let plane = if cb { &self.cb } else { &self.cr };
        Ok(Window {
            samples: self.wait_rows(plane, cw, rows)?,
            stride: cw,
            rows: ch,
            x,
            y,
        })
    }

    /// What direct prediction reads of the macroblock at address `addr`,
    /// once its row is decoded. `None` when the picture failed or has no
    /// such macroblock.
    pub(crate) fn col(&self, addr: usize) -> Option<&Colocated> {
        let width = self.mb_w.max(1);
        let rows = (addr / width + 1).min(self.h / 16);
        self.wait_rows(&self.mbs, width, rows).ok()?.get(addr)
    }

    /// Whether every row of the planes and every macroblock is in.
    pub(crate) fn complete(&self) -> bool {
        let full = |filled: usize, capacity: usize| filled == capacity;
        full(self.luma.filled().len(), self.luma.capacity())
            && full(self.cb.filled().len(), self.cb.capacity())
            && full(self.cr.filled().len(), self.cr.capacity())
            && full(self.mbs.filled().len(), self.mbs.capacity())
    }

    /// Copies of the planes and the macroblocks, for code that reads a
    /// finished picture directly. `None` until every row is in.
    pub(crate) fn copy_whole(&self) -> Option<[Shared<S>; 3]> {
        self.complete().then(|| {
            [&self.luma, &self.cb, &self.cr].map(|plane| Shared::from(plane.filled().to_vec()))
        })
    }

    /// [`Self::copy_whole`] for the macroblocks: decoded ones, with what
    /// direct prediction reads of each, which is all that is read of a
    /// reference's macroblocks.
    pub(crate) fn copy_mbs(&self) -> Option<Shared<Mb>> {
        self.complete().then(|| {
            let mbs = self.mbs.filled().iter().map(|&col| {
                let mut mb = Mb::new(0);
                mb.col = col;
                mb
            });
            Shared::from(mbs.collect::<Vec<_>>())
        })
    }

    /// The rows of each plane that are in so far, without waiting for
    /// more: luma, Cb and Cr.
    pub(crate) fn filled_planes(&self) -> (&[S], &[S], &[S]) {
        (self.luma.filled(), self.cb.filled(), self.cr.filled())
    }

    /// Asks for [`Self::col`]'s answer for `addr` to be brought into the
    /// cache when its row is in; nothing else changes.
    pub(crate) fn prefetch_col(&self, addr: usize) {
        cutlass_h264_kernels::prefetch_item(self.mbs.filled(), addr);
    }
}

/// How many rows from the top of a plane `rows` tall a block whose reads
/// start at row `first` and run `reach` rows down needs: rows outside the
/// plane read its edge row.
fn needed(first: i32, reach: usize, rows: usize) -> usize {
    let end = i64::from(first).saturating_add(i64::try_from(reach).unwrap_or(i64::MAX));
    usize::try_from(end.max(1)).unwrap_or(usize::MAX).min(rows)
}

#[cfg(test)]
mod tests {
    use super::super::{Colocated, Mb};
    use super::Published;
    use crate::recon::sample;

    #[test]
    fn a_block_is_read_once_every_row_it_reads_is_final() {
        // Rows go in as deblocking makes them final. Any block anywhere,
        // inside the picture or far outside it, may be read as soon as the
        // rows it needs are in, without waiting, and then finds in its
        // window every sample it reads, the edges repeated.
        let (w, h) = (48usize, 64usize);
        let y: Vec<u8> = (0..w * h).map(|i| (i * 7 % 251) as u8).collect();
        let u: Vec<u8> = (0..w * h / 4).map(|i| (i * 13 % 241) as u8).collect();
        let v: Vec<u8> = (0..w * h / 4).map(|i| (i * 29 % 239) as u8).collect();
        let published = Published::<u8>::new(w, h, w / 16, None, None);
        let (pic_w, pic_h) = (w as i32, h as i32);
        let mut read = 0;
        for done in 0..=h / 16 {
            let (luma, chroma) = if done == h / 16 {
                (h, h / 2)
            } else {
                ((16 * done).saturating_sub(3), (8 * done).saturating_sub(1))
            };
            published.fill_rows((&y, &u, &v), luma, chroma);
            assert_eq!(published.luma.filled().len(), luma * w);
            for by in -40..pic_h + 40 {
                for bx in [-30, -3, 0, 5, 30, 44, 70] {
                    for height in [4usize, 8, 16] {
                        if super::needed(by, height + 3, h) > luma {
                            continue;
                        }
                        let window = published.luma(bx, by, height).unwrap();
                        assert_eq!((window.x, window.y, window.stride), (bx, by, w));
                        // Two rows before the block and three after it.
                        for dy in -2..height as i32 + 3 {
                            for dx in [-2, 0, 7, 18] {
                                let (at_x, at_y) = (bx + dx, by + dy);
                                let got = sample(window.samples, w, pic_w, pic_h, at_x, at_y);
                                let want = sample(&y, w, pic_w, pic_h, at_x, at_y);
                                assert_eq!(got, want, "luma {bx},{by} +{dx},{dy} height {height}");
                            }
                        }
                        read += 1;
                    }
                }
            }
            for by in -20..pic_h / 2 + 20 {
                for bx in [-12, 0, 3, 20, 40] {
                    for height in [2usize, 4, 8] {
                        if super::needed(by, height + 1, h / 2) > chroma {
                            continue;
                        }
                        for (cb, plane) in [(true, &u), (false, &v)] {
                            let window = published.chroma(cb, bx, by, height).unwrap();
                            // The block and one sample after it each way.
                            for dy in 0..height as i32 + 1 {
                                for dx in [0, 4, 8] {
                                    let (at_x, at_y) = (bx + dx, by + dy);
                                    let size = (w / 2, pic_w / 2, pic_h / 2);
                                    let got =
                                        sample(window.samples, size.0, size.1, size.2, at_x, at_y);
                                    let want = sample(plane, size.0, size.1, size.2, at_x, at_y);
                                    assert_eq!(got, want, "chroma {bx},{by} height {height}");
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(read > 1000);
    }

    #[test]
    fn a_whole_picture_gives_its_buffers_back_and_a_broken_one_does_not() {
        let home = std::sync::Arc::new(super::Spares::default());
        let (w, h) = (32usize, 32usize);
        let y = vec![9u8; w * h];
        let c = vec![5u8; w * h / 4];
        let whole = Published::<u8>::new(w, h, 2, None, Some(home.clone()));
        whole.fill_rows((&y, &c, &c), h, h / 2);
        whole.mbs.push(&[Mb::new(0).col; 4]);
        let at = whole.luma.filled().as_ptr();
        drop(whole);
        let luma = home.take::<u8>(w * h).unwrap();
        assert_eq!((luma.as_ptr(), &luma), (at, &y));
        assert_eq!(home.take::<u8>(w * h / 4), Some(c.clone()));
        assert_eq!(home.take::<u8>(w * h / 4), Some(c.clone()));
        assert!(home.take::<Colocated>(4).is_none());
        let broken = Published::<u8>::new(w, h, 2, None, Some(home.clone()));
        broken.fill_rows((&y, &c, &c), h - 3, h / 2 - 1);
        drop(broken);
        assert!(home.take::<u8>(w * h).is_none());
    }

    #[test]
    fn a_failed_picture_lets_its_readers_go() {
        let published = std::sync::Arc::new(Published::<u8>::new(32, 32, 2, None, None));
        let reader = {
            let published = published.clone();
            std::thread::spawn(move || {
                (
                    published.luma(0, 0, 16).is_err(),
                    published.col(0).is_none(),
                    published.planes().is_err(),
                )
            })
        };
        let error = cutlass_decode::CodecError::Malformed {
            at_bit: 7,
            what: "test",
        };
        published.finish(Err(error));
        assert_eq!(reader.join().unwrap(), (true, true, true));
        assert_eq!(published.failure(), Some(error));
    }
}
