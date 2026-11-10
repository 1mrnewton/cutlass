//! A picture's finished rows, handed out while the rest is still being
//! decoded.
//!
//! One thread decodes a picture into planes only it holds. As rows become
//! final it copies them here in bands, and threads decoding later pictures
//! read those copies for inter prediction. Nothing is ever read while it is
//! written: a band is filled once, by its writer, before any reader can see
//! it, and readers wait for the band they need.
//!
//! A band is a run of rows with the picture's edge samples repeated around
//! it. Band `e` ends at the last row that is final once `e` macroblock rows
//! are deblocked, and is tall enough that any block whose last row read
//! falls in the band's last sixteen rows finds every row it reads inside.
//! So a band is handed out the moment its rows are final, a reader waits
//! for exactly the rows it needs and no more, and it gets one unbroken
//! slice with no edge handling to do.

use std::sync::{Condvar, Mutex, OnceLock};

use cutlass_decode::CodecError;

use super::{Frame, Mb, Shared};
use crate::recon::Sample;

/// Samples repeated left and right of a luma row. A block origin is held
/// within 19 samples of the picture, and the filter reads 2 before a block
/// and 3 after one of at most 16.
const LUMA_PAD: usize = 24;
/// Rows in a luma band. A block's last row read is one of the band's last
/// 16, and its first is at most 16 + 2 + 3 − 1 rows before that.
const LUMA_ROWS: usize = 36;
/// Deblocking macroblock row `e` changes up to three luma rows above it,
/// so with `e` rows done the last final luma row is `16e − 4`.
const LUMA_UNSETTLED: i32 = 3;
/// Samples repeated left and right of a chroma row.
const CHROMA_PAD: usize = 12;
/// Rows in a chroma band: a last row read among its last 8, and up to 8
/// rows before that.
const CHROMA_ROWS: usize = 16;
/// Deblocking changes one chroma row above a macroblock row, so with `e`
/// rows done the last final chroma row is `8e − 2`.
const CHROMA_UNSETTLED: i32 = 1;

/// The first row of luma band `band`.
fn luma_first(band: i32) -> i32 {
    band * 16 - LUMA_UNSETTLED - LUMA_ROWS as i32
}

/// The first row of chroma band `band`.
fn chroma_first(band: i32) -> i32 {
    band * 8 - CHROMA_UNSETTLED - CHROMA_ROWS as i32
}

/// A finished picture's planes and macroblocks.
pub(crate) struct Planes<S: Sample> {
    pub y: Shared<S>,
    pub u: Shared<S>,
    pub v: Shared<S>,
    pub mb: Shared<Mb>,
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

/// One band or row, set once.
type Slot<T> = OnceLock<Box<[T]>>;

pub(crate) struct Published<S: Sample> {
    /// Luma width and height.
    w: usize,
    h: usize,
    mb_w: usize,
    luma: Vec<Slot<S>>,
    cb: Vec<Slot<S>>,
    cr: Vec<Slot<S>>,
    /// One slot per macroblock row.
    mbs: Vec<Slot<Mb>>,
    /// The whole picture, or why it has none.
    done: OnceLock<Result<Planes<S>, CodecError>>,
    gate: Mutex<()>,
    changed: Condvar,
    /// The decoding threads' permits, when they have them. A reader that
    /// has to wait here is one of those threads.
    permits: Option<std::sync::Arc<Permits>>,
}

/// A block of a published plane: the samples to read, the distance between
/// their rows, their size as a picture, and the block's place in them.
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

fn slots<T>(count: usize) -> Vec<Slot<T>> {
    (0..count).map(|_| OnceLock::new()).collect()
}

impl<S: Sample> Published<S> {
    /// For a 4:2:0 frame of `w`×`h` luma samples, `h` a multiple of 16.
    pub(crate) fn new(
        w: usize,
        h: usize,
        mb_w: usize,
        permits: Option<std::sync::Arc<Permits>>,
    ) -> Self {
        // A block held to within 19 rows of the picture and 16 rows tall
        // reads at most 19 rows past the bottom.
        let bands = h / 16 + 3;
        Self {
            w,
            h,
            mb_w,
            luma: slots(bands),
            cb: slots(bands),
            cr: slots(bands),
            mbs: slots(h / 16),
            done: OnceLock::new(),
            gate: Mutex::new(()),
            changed: Condvar::new(),
            permits,
        }
    }

    fn wake(&self) {
        // Taking the gate first means a reader that has just looked and
        // found nothing is already waiting when this wakes it.
        drop(self.gate.lock());
        self.changed.notify_all();
    }

    fn wait<'a, T>(&'a self, slot: &'a Slot<T>) -> Result<&'a [T], CodecError> {
        if let Some(held) = slot.get() {
            return Ok(held);
        }
        // This thread's core goes to another picture while it waits.
        if let Some(permits) = &self.permits {
            permits.give();
        }
        let found = self.wait_parked(slot);
        if let Some(permits) = &self.permits {
            permits.take();
        }
        found
    }

    fn wait_parked<'a, T>(&'a self, slot: &'a Slot<T>) -> Result<&'a [T], CodecError> {
        let mut guard = self.gate.lock().map_err(|_| broken())?;
        loop {
            if let Some(held) = slot.get() {
                return Ok(held);
            }
            match self.done.get() {
                Some(Err(error)) => return Err(*error),
                // A finished picture has every slot filled.
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
        for row in from.min(done)..done {
            let start = row * self.mb_w;
            if let Some(mbs) = pic.mb.get(start..start + self.mb_w) {
                let _ = self.mbs[row].set(mbs.into());
            }
        }
        // With `done` rows deblocked, bands 0 to `done` are final. Band 0
        // is the picture's first row repeated, final with the first row.
        let count = |rows: usize| if rows == 0 { 0 } else { rows + 1 };
        let bands = if last { self.luma.len() } else { count(done) };
        self.fill_bands((&pic.y, &pic.u, &pic.v), count(from).min(bands)..bands);
        self.wake();
        done
    }

    /// Copies bands `bands` out of the three planes.
    fn fill_bands(&self, planes: (&[S], &[S], &[S]), bands: std::ops::Range<usize>) {
        let (y, u, v) = planes;
        let (cw, ch) = (self.w / 2, self.h / 2);
        for index in bands.start..bands.end.min(self.luma.len()) {
            let band = index as i32;
            let luma = band_of(y, (self.w, self.h), luma_first(band), LUMA_ROWS, LUMA_PAD);
            let _ = self.luma[index].set(luma);
            let first = chroma_first(band);
            let _ = self.cb[index].set(band_of(u, (cw, ch), first, CHROMA_ROWS, CHROMA_PAD));
            let _ = self.cr[index].set(band_of(v, (cw, ch), first, CHROMA_ROWS, CHROMA_PAD));
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
    /// read, three below its own, is final.
    pub(crate) fn luma(&self, x: i32, y: i32, height: usize) -> Result<Window<'_, S>, CodecError> {
        // Past these, every sample the block reads is the same edge sample.
        let x = x.clamp(-19, self.w as i32 + 2);
        let y = y.clamp(-19, self.h as i32 + 1);
        let last = y + height.clamp(1, 16) as i32 + 2;
        // The first band whose final rows reach `last`.
        let band = (last + LUMA_UNSETTLED + 16).div_euclid(16);
        let slot = self.luma.get(band as usize).ok_or_else(broken)?;
        Ok(Window {
            samples: self.wait(slot)?,
            stride: self.w + 2 * LUMA_PAD,
            rows: LUMA_ROWS,
            x: x + LUMA_PAD as i32,
            y: y - luma_first(band),
        })
    }

    /// The chroma samples for a block at most 8 wide and `height` rows
    /// tall, at most 8, at (`x`, `y`) in the chroma plane, Cb when `cb` is
    /// set. Waits until the row below the block is final.
    pub(crate) fn chroma(
        &self,
        cb: bool,
        x: i32,
        y: i32,
        height: usize,
    ) -> Result<Window<'_, S>, CodecError> {
        let x = x.clamp(-9, self.w as i32 / 2);
        let y = y.clamp(-9, self.h as i32 / 2);
        let last = y + height.clamp(1, 8) as i32;
        let band = (last + CHROMA_UNSETTLED + 8).div_euclid(8);
        let planes = if cb { &self.cb } else { &self.cr };
        let slot = planes.get(band as usize).ok_or_else(broken)?;
        Ok(Window {
            samples: self.wait(slot)?,
            stride: self.w / 2 + 2 * CHROMA_PAD,
            rows: CHROMA_ROWS,
            x: x + CHROMA_PAD as i32,
            y: y - chroma_first(band),
        })
    }

    /// The macroblock at address `addr`, once its row is decoded. `None`
    /// when the picture failed or has no such macroblock.
    pub(crate) fn mb(&self, addr: usize) -> Option<&Mb> {
        let row = self.mbs.get(addr / self.mb_w.max(1))?;
        self.wait(row).ok()?.get(addr % self.mb_w.max(1))
    }
}

/// `rows` rows of a plane from row `first`, each with `pad` samples added
/// on both sides. Rows and samples outside the plane repeat its edge.
fn band_of<S: Sample>(
    plane: &[S],
    size: (usize, usize),
    first: i32,
    rows: usize,
    pad: usize,
) -> Box<[S]> {
    let (w, h) = size;
    let mut out = Vec::with_capacity((w + 2 * pad) * rows);
    if w == 0 || h == 0 {
        return out.into_boxed_slice();
    }
    for row in 0..rows {
        let source = (first + row as i32).clamp(0, h as i32 - 1) as usize;
        let line = &plane[source * w..source * w + w];
        out.extend(std::iter::repeat_n(line[0], pad));
        out.extend_from_slice(line);
        out.extend(std::iter::repeat_n(line[w - 1], pad));
    }
    out.into_boxed_slice()
}

#[cfg(test)]
mod tests {
    use super::Published;
    use crate::recon::sample;

    #[test]
    fn a_window_holds_every_sample_a_block_reads() {
        // A block anywhere, inside the picture or far outside it, must find
        // in its window exactly the samples it would read from the picture
        // with the edges repeated.
        let (w, h) = (48usize, 32usize);
        let y: Vec<u8> = (0..w * h).map(|i| (i * 7 % 251) as u8).collect();
        let u: Vec<u8> = (0..w * h / 4).map(|i| (i * 13 % 241) as u8).collect();
        let v: Vec<u8> = (0..w * h / 4).map(|i| (i * 29 % 239) as u8).collect();
        let published = Published::<u8>::new(w, h, w / 16, None);
        published.fill_bands((&y, &u, &v), 0..published.luma.len());
        let (pic_w, pic_h) = (w as i32, h as i32);
        for by in -80..pic_h + 80 {
            for bx in (-80..pic_w + 80).step_by(3) {
                for height in [4usize, 8, 16] {
                    let window = published.luma(bx, by, height).unwrap();
                    // Two samples before the block and three after it.
                    for dy in -2..height as i32 + 3 {
                        for dx in -2..19 {
                            let at =
                                (window.y + dy) as usize * window.stride + (window.x + dx) as usize;
                            let want = sample(&y, w, pic_w, pic_h, bx + dx, by + dy);
                            let got = i32::from(window.samples[at]);
                            assert_eq!(got, want, "luma {bx},{by} +{dx},{dy} height {height}");
                        }
                    }
                    assert!(window.y >= 2 && window.y + height as i32 + 3 <= window.rows as i32);
                }
            }
        }
        for by in -40..pic_h / 2 + 40 {
            for bx in -40..pic_w / 2 + 40 {
                for (cb, plane) in [(true, &u), (false, &v)] {
                    for height in [2usize, 4, 8] {
                        let window = published.chroma(cb, bx, by, height).unwrap();
                        // The block and one sample after it each way.
                        for dy in 0..height as i32 + 1 {
                            for dx in 0..9 {
                                let at = (window.y + dy) as usize * window.stride
                                    + (window.x + dx) as usize;
                                let want =
                                    sample(plane, w / 2, pic_w / 2, pic_h / 2, bx + dx, by + dy);
                                let got = i32::from(window.samples[at]);
                                assert_eq!(got, want, "chroma {bx},{by} height {height}");
                            }
                        }
                        assert!(window.y >= 0 && window.y + (height as i32) < window.rows as i32);
                    }
                }
            }
        }
    }

    #[test]
    fn a_band_is_out_as_soon_as_its_rows_are_final() {
        // With `e` macroblock rows deblocked, a 16-row block two rows up
        // can be read: its last row read, three below its own, is final.
        let (w, h) = (32usize, 96usize);
        let published = Published::<u8>::new(w, h, 2, None);
        let (y, c) = (vec![0u8; w * h], vec![0u8; w * h / 4]);
        published.fill_bands((&y, &c, &c), 0..4);
        // Three rows done: bands 0 to 3. A block at row 16 reads to row 34.
        assert!(published.luma[3].get().is_some() && published.luma[4].get().is_none());
        assert_eq!((34 + super::LUMA_UNSETTLED + 16).div_euclid(16), 3);
        // The same block one row lower needs the fourth row done.
        assert_eq!((50 + super::LUMA_UNSETTLED + 16).div_euclid(16), 4);
    }

    #[test]
    fn a_failed_picture_lets_its_readers_go() {
        let published = std::sync::Arc::new(Published::<u8>::new(32, 32, 2, None));
        let reader = {
            let published = published.clone();
            std::thread::spawn(move || {
                (
                    published.luma(0, 0, 16).is_err(),
                    published.mb(0).is_none(),
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
