//! A picture other pictures reference: its final samples and the motion
//! kept for temporal prediction, both published row by row while the
//! picture is still being decoded, and what its decode job left.

use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::SeqCst;
use std::sync::{Arc, Condvar, Mutex};

use cutlass_decode::CodecError;
use cutlass_simd::shared_rows::{PlaneBuf, RowsWriter, SharedRows};

use crate::picture::{ColMv, lock};
use crate::pixel::{AnyRows, Pixel};

/// What a finished decode job hands back with its picture.
pub struct Done {
    pub result: Result<(), CodecError>,
    /// The cropped output samples, when the picture is output.
    pub out: Option<OutSamples>,
}

/// The output picture's samples.
pub enum OutSamples {
    /// Luma, then Cb, then Cr, 8 bits each, one buffer.
    Packed(Vec<u8>),
    /// Each plane on its own, 8 bits, and above 8 bits the full samples.
    Planes {
        eight: [Vec<u8>; 3],
        deep: Option<[Vec<u16>; 3]>,
    },
}

/// Buffers of finished pictures for the next ones.
#[derive(Default)]
pub struct Spares {
    pub eight: Mutex<Vec<[PlaneBuf<u8>; 3]>>,
    pub deep: Mutex<Vec<[PlaneBuf<u16>; 3]>>,
    pub col: Mutex<Vec<[PlaneBuf<ColMv>; 3]>>,
}

/// A decoded picture, or one still being decoded, as references see it.
pub struct Frame {
    pub poc: i32,
    pub width: usize,
    pub height: usize,
    /// Final samples, published in luma rows.
    pub rows: AnyRows,
    /// Motion of each 16×16 block (clause 8.5.3.2.8), published in luma rows.
    pub col: Arc<SharedRows<ColMv>>,
    gate: Mutex<()>,
    moved: Condvar,
    /// The job stopped without publishing: waiters take what is there.
    abandoned: AtomicBool,
    done: Mutex<Option<Done>>,
    finished: Condvar,
    spares: Option<Arc<Spares>>,
}

impl Frame {
    /// A frame to be decoded: no row published yet. Returns the frame
    /// and the writers of its samples and its motion.
    pub fn new<P: Pixel>(
        (width, height): (usize, usize),
        poc: i32,
        spares: &Arc<Spares>,
        wrap: impl FnOnce(Arc<SharedRows<P>>) -> AnyRows,
        bufs: Option<[PlaneBuf<P>; 3]>,
    ) -> (Arc<Self>, RowsWriter<P>, RowsWriter<ColMv>) {
        let (cw, ch) = (width / 2, height / 2);
        let writer = RowsWriter::new([width, cw, cw], [height, ch, ch], [0, 1, 1], bufs);
        let col_stride = width.div_ceil(16);
        let col_rows = height.div_ceil(16);
        let col_bufs = lock(&spares.col).pop();
        let col = RowsWriter::new([col_stride, 0, 0], [col_rows, 0, 0], [4, 0, 0], col_bufs);
        let frame = Arc::new(Self {
            poc,
            width,
            height,
            rows: wrap(writer.shared().clone()),
            col: col.shared().clone(),
            gate: Mutex::new(()),
            moved: Condvar::new(),
            abandoned: AtomicBool::new(false),
            done: Mutex::new(None),
            finished: Condvar::new(),
            spares: Some(spares.clone()),
        });
        (frame, writer, col)
    }

    /// A picture the stream named and did not send (clause 8.3.3):
    /// mid-grey, with no motion, every row published.
    pub fn missing(width: usize, height: usize, bit_depth: u8, poc: i32) -> Arc<Self> {
        let grey = 1i32 << (bit_depth - 1);
        fn grey_rows<P: Pixel>(w: usize, h: usize, v: i32) -> RowsWriter<P> {
            let (cw, ch) = (w / 2, h / 2);
            let mut writer = RowsWriter::new([w, cw, cw], [h, ch, ch], [0, 1, 1], None);
            for plane in writer.pending_mut(usize::MAX) {
                plane.fill(P::of(v));
            }
            writer.publish(usize::MAX);
            writer
        }
        let rows = if bit_depth == 8 {
            AnyRows::Eight(grey_rows::<u8>(width, height, grey).shared().clone())
        } else {
            AnyRows::Deep(grey_rows::<u16>(width, height, grey).shared().clone())
        };
        let col_stride = width.div_ceil(16);
        let mut col = RowsWriter::new(
            [col_stride, 0, 0],
            [height.div_ceil(16), 0, 0],
            [4, 0, 0],
            None,
        );
        col.publish(usize::MAX);
        Arc::new(Self {
            poc,
            width,
            height,
            rows,
            col: col.shared().clone(),
            gate: Mutex::new(()),
            moved: Condvar::new(),
            abandoned: AtomicBool::new(false),
            done: Mutex::new(Some(Done {
                result: Ok(()),
                out: None,
            })),
            finished: Condvar::new(),
            spares: None,
        })
    }

    /// Wake the threads waiting for rows of this frame.
    pub fn notify(&self) {
        drop(lock(&self.gate));
        self.moved.notify_all();
    }

    /// Block until luma rows `0..rows` (at most the picture's) are final.
    #[inline]
    pub fn wait_rows(&self, rows: usize) {
        let need = rows.min(self.height);
        if self.rows.published() >= need {
            return;
        }
        let mut gate = lock(&self.gate);
        while self.rows.published() < need && !self.abandoned.load(SeqCst) {
            gate = self
                .moved
                .wait(gate)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// Block until the motion of luma rows `0..rows` is final.
    #[inline]
    pub fn wait_motion(&self, rows: usize) {
        let need = rows.min(self.height);
        if self.col.published() >= need {
            return;
        }
        let mut gate = lock(&self.gate);
        while self.col.published() < need && !self.abandoned.load(SeqCst) {
            gate = self
                .moved
                .wait(gate)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// The motion of the 16×16 block holding luma (`x`, `y`). Its row must
    /// be published ([`Self::wait_motion`]).
    #[inline]
    pub fn col_at(&self, x: usize, y: usize) -> ColMv {
        self.col
            .rows(0, y >> 4, (y >> 4) + 1)
            .and_then(|(r, _)| r.get(x >> 4).copied())
            .unwrap_or_default()
    }

    /// The decode job ended.
    pub fn finish(&self, done: Done) {
        *lock(&self.done) = Some(done);
        self.finished.notify_all();
    }

    /// The job stopped without finishing (it panicked): release every
    /// waiter and record a failure.
    pub fn abandon(&self) {
        self.abandoned.store(true, SeqCst);
        self.notify();
        let mut done = lock(&self.done);
        if done.is_none() {
            *done = Some(Done {
                result: Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "picture decode stopped",
                }),
                out: None,
            });
        }
        drop(done);
        self.finished.notify_all();
    }

    /// Whether the decode job ended with an error.
    pub fn failed(&self) -> bool {
        lock(&self.done).as_ref().is_some_and(|d| d.result.is_err())
    }

    /// Block until the decode job ends.
    pub fn wait_finished(&self) {
        let mut done = lock(&self.done);
        while done.is_none() {
            done = self
                .finished
                .wait(done)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// Whether the decode job has ended.
    pub fn is_done(&self) -> bool {
        lock(&self.done).is_some()
    }

    /// Block until the decode job ends, and take what it left: the result
    /// stays (as `Ok` for later takers), the output samples move out.
    pub fn take_done(&self) -> Done {
        let mut done = lock(&self.done);
        loop {
            if let Some(d) = done.as_mut() {
                return Done {
                    result: std::mem::replace(&mut d.result, Ok(())),
                    out: d.out.take(),
                };
            }
            done = self
                .finished
                .wait(done)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        let Some(spares) = self.spares.take() else {
            return;
        };
        // The writers are gone once the job has ended: the buffers come
        // back whole.
        let rows = std::mem::replace(
            &mut self.rows,
            AnyRows::Eight(Arc::new(SharedRows::empty())),
        );
        match rows {
            AnyRows::Eight(r) => {
                if let Ok(r) = Arc::try_unwrap(r) {
                    let mut s = lock(&spares.eight);
                    if s.len() < 24 {
                        s.push(r.into_buffers());
                    }
                }
            }
            AnyRows::Deep(r) => {
                if let Ok(r) = Arc::try_unwrap(r) {
                    let mut s = lock(&spares.deep);
                    if s.len() < 24 {
                        s.push(r.into_buffers());
                    }
                }
            }
        }
        let col = std::mem::replace(&mut self.col, Arc::new(SharedRows::empty()));
        if let Ok(c) = Arc::try_unwrap(col) {
            let mut s = lock(&spares.col);
            if s.len() < 24 {
                s.push(c.into_buffers());
            }
        }
    }
}
