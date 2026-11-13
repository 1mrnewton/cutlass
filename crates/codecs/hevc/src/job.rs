//! The decode of one picture: its slice segments, the loop filters run one
//! CTB row behind reconstruction, and the final rows and motion published
//! to the pictures that reference this one as they become final.

use std::sync::atomic::Ordering::SeqCst;
use std::sync::atomic::{AtomicBool, AtomicU32};
use std::sync::{Arc, Mutex};

use cutlass_decode::CodecError;
use cutlass_pool::Rows;
use cutlass_simd::shared_rows::RowsWriter;

use crate::cabac::Cabac;
use crate::ctu::{Cu, SliceDecoder, Wave};
use crate::frame::{Done, Frame, OutSamples};
use crate::nal::Nal;
use crate::picture::{ColMv, Picture, Work};
use crate::pixel::Pixel;
use crate::ps::Sps;
use crate::slice::SliceType;

/// One slice segment of the picture, ready to decode.
pub struct SliceData {
    pub nal: Arc<Nal>,
    /// Index into the picture's slices.
    pub index: usize,
    /// Where the slice data starts in the RBSP.
    pub start: usize,
    /// Substream starts, from `start`.
    pub entries: Vec<usize>,
    pub qp: i32,
    pub slice_type: SliceType,
    pub segment: usize,
}

/// How the output picture is handed back.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OutForm {
    Packed,
    Planes,
}

/// A picture to decode.
pub struct Job<P> {
    pub frame: Arc<Frame>,
    pub pic: Picture,
    pub work: Work<P>,
    pub writer: RowsWriter<P>,
    pub col: RowsWriter<ColMv>,
    pub slices: Vec<SliceData>,
    /// The output form, when the picture is output.
    pub output: Option<OutForm>,
    /// Threads for the rows of a wavefront slice segment.
    pub row_threads: usize,
}

/// What a finished job gives back for the next pictures.
pub struct Leftovers<P> {
    pub pic: Picture,
    pub work: Work<P>,
}

impl<P: Pixel> Job<P> {
    /// Decode the picture, publish every row, and record the result on the
    /// frame. Returns the buffers to reuse.
    pub fn run(self) -> Leftovers<P> {
        let Self {
            frame,
            pic,
            work,
            writer,
            col,
            slices,
            output,
            row_threads,
        } = self;
        let out = output.map(|form| OutRows::new(&pic.sps, form));
        let pipe = Pipeline::new(&pic, &work, &frame, writer, col, out);
        let mut result = Ok(());
        for slice in &slices {
            if let Err(e) = decode_slice(&pic, &work, &pipe, slice, row_threads) {
                result = Err(e);
                break;
            }
        }
        let (writer, out) = pipe.finish();
        let out = out.map(OutRows::finish);
        drop(writer);
        frame.finish(Done { result, out });
        Leftovers { pic, work }
    }
}

/// The cropped output samples of a picture, taken a band of rows at a
/// time as its final rows are published, while they are still in cache.
struct OutRows {
    form: OutForm,
    /// Bits above 8 per sample.
    shift: u32,
    /// Left column, top row, width and height of each cropped plane.
    crop: [(usize, usize, usize, usize); 3],
    /// Rows of each cropped plane taken.
    next: [usize; 3],
    eight: [Vec<u8>; 3],
    deep: Option<[Vec<u16>; 3]>,
}

impl OutRows {
    fn new(sps: &Sps, form: OutForm) -> Self {
        let width = sps.width as usize;
        let height = sps.height as usize;
        let [left, right, top, bottom] = sps.crop.map(|v| v as usize);
        let w = width.saturating_sub(left + right);
        let h = height.saturating_sub(top + bottom);
        let crop = [
            (left, top, w, h),
            (left / 2, top / 2, w / 2, h / 2),
            (left / 2, top / 2, w / 2, h / 2),
        ];
        let shift = u32::from(sps.bit_depth) - 8;
        let size = |c: usize| crop[c].2 * crop[c].3;
        // Packed output gathers the chroma planes after luma: the luma
        // buffer is made large enough for all three.
        let eight = match form {
            OutForm::Packed => [
                Vec::with_capacity(size(0) + size(1) + size(2)),
                Vec::with_capacity(size(1)),
                Vec::with_capacity(size(2)),
            ],
            OutForm::Planes => std::array::from_fn(|c| Vec::with_capacity(size(c))),
        };
        let deep = (form == OutForm::Planes && shift > 0)
            .then(|| std::array::from_fn(|c| Vec::with_capacity(size(c))));
        Self {
            form,
            shift,
            crop,
            next: [0; 3],
            eight,
            deep,
        }
    }

    /// Take the cropped rows published since the last call.
    fn take<P: Pixel>(&mut self, writer: &RowsWriter<P>) {
        let shared = writer.shared();
        for c in 0..3 {
            let (x0, y0, w, h) = self.crop[c];
            let y = y0 + self.next[c];
            if self.next[c] >= h {
                continue;
            }
            let Some((rows, stride, have)) = shared.rows_from(c, y) else {
                continue;
            };
            let n = have.saturating_sub(y).min(h - self.next[c]);
            for i in 0..n {
                let row = &rows[i * stride + x0..i * stride + x0 + w];
                let out = &mut self.eight[c];
                match P::bytes(row) {
                    Some(bytes) => out.extend_from_slice(bytes),
                    None => out.extend(row.iter().map(|s| (s.i() >> self.shift) as u8)),
                }
                if let Some(deep) = &mut self.deep {
                    deep[c].extend(row.iter().map(|s| s.i() as u16));
                }
            }
            self.next[c] += n;
        }
    }

    /// The samples; rows never published are zeros.
    fn finish(mut self) -> OutSamples {
        for c in 0..3 {
            let (_, _, w, h) = self.crop[c];
            self.eight[c].resize(w * h, 0);
            if let Some(deep) = &mut self.deep {
                deep[c].resize(w * h, 0);
            }
        }
        match self.form {
            OutForm::Packed => {
                let [mut bytes, cb, cr] = self.eight;
                bytes.extend_from_slice(&cb);
                bytes.extend_from_slice(&cr);
                OutSamples::Packed(bytes)
            }
            OutForm::Planes => OutSamples::Planes {
                eight: self.eight,
                deep: self.deep,
            },
        }
    }
}

/// Decode one slice segment: on this thread, or a worker for each CTB row
/// of a wavefront segment.
fn decode_slice<P: Pixel>(
    pic: &Picture,
    work: &Work<P>,
    pipe: &Pipeline<'_, P>,
    slice: &SliceData,
    threads: usize,
) -> Result<(), CodecError> {
    let Some(data) = slice.nal.rbsp.get(slice.start..) else {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "slice data",
        });
    };
    let job = SliceJob {
        index: slice.index,
        data,
        entries: &slice.entries,
        qp: slice.qp,
        slice_type: slice.slice_type,
    };
    let layout = &pic.layout;
    let ctb_w = layout.ctb_width as usize;
    let segment = slice.segment;
    let first_row = segment / ctb_w;
    let rows = slice.entries.len() + 1;
    let wavefront = threads > 1
        && pic.pps.entropy_sync
        && !pic.pps.tiles
        && segment.is_multiple_of(ctb_w)
        && rows > 1
        && first_row + rows <= layout.ctb_height as usize;
    if !wavefront {
        let result = job
            .decoder(pic, work, pipe, first_row, None, 0, None)
            .decode();
        pipe.nudge(None);
        return result;
    }
    // One worker for each CTB row of the segment. The last decodes every
    // row below too, where the segment may go on without entry points.
    let gates: Vec<Rows> = (0..rows).map(|_| Rows::new(ctb_w as u32)).collect();
    let results = in_parallel(rows, threads, |r| {
        let _done = RowDone(&gates[r]);
        let wave = Wave {
            row: first_row + r,
            above: r.checked_sub(1).map(|above| &gates[above]),
            mine: &gates[r],
            row_only: r + 1 < rows,
        };
        let start = (r > 0).then_some((first_row + r) * ctb_w);
        let result = job
            .decoder(
                pic,
                work,
                pipe,
                first_row + r,
                start,
                r.saturating_sub(1),
                Some(wave),
            )
            .decode();
        pipe.nudge(None);
        result
    });
    results.into_iter().collect()
}

/// Run `count` pieces on up to `threads` threads, this one among them,
/// each thread taking the next piece in order. Results in index order.
fn in_parallel<T: Send>(count: usize, threads: usize, work: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let threads = threads.min(count);
    if threads <= 1 {
        return (0..count).map(&work).collect();
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let run = || {
        let mut done = Vec::new();
        loop {
            let index = next.fetch_add(1, SeqCst);
            if index >= count {
                break;
            }
            done.push((index, work(index)));
        }
        done
    };
    let mut paired = std::thread::scope(|scope| {
        let joins: Vec<_> = (1..threads).map(|_| scope.spawn(run)).collect();
        let mut paired = run();
        for join in joins {
            match join.join() {
                Ok(done) => paired.extend(done),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        paired
    });
    paired.sort_by_key(|(index, _)| *index);
    paired.into_iter().map(|(_, value)| value).collect()
}

/// Marks a row worker's row finished however it stops, so the row below
/// never waits for a worker that has returned.
struct RowDone<'a>(&'a Rows);

impl Drop for RowDone<'_> {
    fn drop(&mut self) {
        self.0.report_through(u32::MAX);
    }
}

/// What every worker of one slice segment starts from.
struct SliceJob<'a> {
    index: usize,
    data: &'a [u8],
    entries: &'a [usize],
    qp: i32,
    slice_type: SliceType,
}

impl<'a> SliceJob<'a> {
    #[allow(clippy::too_many_arguments)]
    fn decoder<'p, P: Pixel>(
        &self,
        pic: &'p Picture,
        work: &'p Work<P>,
        pipe: &'p Pipeline<'p, P>,
        row: usize,
        start_rs: Option<usize>,
        next_entry: usize,
        wave: Option<Wave<'p>>,
    ) -> SliceDecoder<'p, 'p, P>
    where
        'a: 'p,
    {
        let qp = self.qp;
        let row = row.min(work.count.saturating_sub(1));
        SliceDecoder {
            pic,
            work,
            out: work.band(row),
            pipe,
            start_rs,
            wave,
            slice: self.index,
            sps: &pic.sps,
            pps: &pic.pps,
            cabac: Cabac::new(self.data, [0; 256]),
            data: self.data,
            entries: self.entries.to_vec(),
            next_entry,
            qp_y: qp,
            qp_pred: qp,
            last_qp: qp,
            first_qg: true,
            qp_delta_coded: false,
            qp_delta: 0,
            cu: Cu::default(),
            coeffs: [0; 32 * 32],
            res: [0; 32 * 32],
            pred: [[0; 64 * 64]; 2],
            ctb_x: 0,
            ctb_y: 0,
            ctb_rs: 0,
            merge_2nx2n: false,
            slice_type: self.slice_type,
            mc: crate::mc::Scratch::default(),
        }
    }
}

/// The loop filters and the publication of one picture, run behind its
/// reconstruction. Every CTB reconstructed is counted; a CTB row whose
/// CTBs are all in has its motion published; once the row below is in
/// too it is deblocked, and once that row is deblocked it gets SAO and
/// its final samples are published.
///
/// Reconstruction calls [`Self::ctb_done`] while it holds the lock of its
/// own CTB row, so the filters never wait for the pipeline's lock there:
/// whoever holds it does the work any other thread made ready.
pub struct Pipeline<'p, P> {
    pic: &'p Picture,
    work: &'p Work<P>,
    frame: &'p Frame,
    /// CTBs reconstructed in each CTB row.
    counts: Vec<AtomicU32>,
    dirty: AtomicBool,
    state: Mutex<PipeState<P>>,
}

struct PipeState<P> {
    writer: RowsWriter<P>,
    col: RowsWriter<ColMv>,
    /// CTB rows whose motion is published.
    motion: usize,
    /// CTB rows deblocked.
    deblocked: usize,
    /// CTB rows with their final samples published.
    finals: usize,
    /// The output samples taken so far, when the picture is output.
    out: Option<OutRows>,
}

impl<'p, P: Pixel> Pipeline<'p, P> {
    fn new(
        pic: &'p Picture,
        work: &'p Work<P>,
        frame: &'p Frame,
        writer: RowsWriter<P>,
        col: RowsWriter<ColMv>,
        out: Option<OutRows>,
    ) -> Self {
        Self {
            pic,
            work,
            frame,
            counts: (0..work.count).map(|_| AtomicU32::new(0)).collect(),
            dirty: AtomicBool::new(false),
            state: Mutex::new(PipeState {
                writer,
                col,
                motion: 0,
                deblocked: 0,
                finals: 0,
                out,
            }),
        }
    }

    fn complete(&self, row: usize) -> bool {
        self.counts
            .get(row)
            .is_some_and(|c| c.load(SeqCst) >= self.pic.layout.ctb_width)
    }

    /// CTB `rs` is reconstructed, by a thread that holds the lock of its
    /// CTB row.
    pub fn ctb_done(&self, rs: usize) {
        let row = rs / self.pic.layout.ctb_width as usize;
        let Some(count) = self.counts.get(row) else {
            return;
        };
        if count.fetch_add(1, SeqCst) + 1 < self.pic.layout.ctb_width {
            return;
        }
        self.nudge(Some(row));
    }

    /// Run what is ready, except what touches CTB row `held`, whose lock
    /// the calling thread holds. A thread that lets a row go nudges again.
    pub fn nudge(&self, held: Option<usize>) {
        self.dirty.store(true, SeqCst);
        while let Ok(mut state) = self.state.try_lock() {
            self.dirty.store(false, SeqCst);
            self.advance(&mut state, false, held);
            drop(state);
            if !self.dirty.load(SeqCst) {
                break;
            }
        }
    }

    /// Run what is ready. `end`: reconstruction has stopped, so every row
    /// is filtered whether or not the row below came in.
    fn advance(&self, st: &mut PipeState<P>, end: bool, held: Option<usize>) {
        let touches = |lo: usize, hi: usize| held.is_some_and(|h| h + 1 >= lo && h <= hi);
        let rows = self.work.count;
        let ctb = 1usize << self.work.log2_ctb;
        let height = self.pic.height;
        let mut moved = false;
        while st.motion < rows && (end || self.complete(st.motion)) {
            self.motion_row(st, st.motion);
            st.motion += 1;
            let upto = if st.motion == rows {
                usize::MAX
            } else {
                st.motion * ctb
            };
            st.col.publish(upto);
            moved = true;
        }
        loop {
            let d = st.deblocked;
            // Row `d` is deblocked once nothing reconstructs it any more:
            // the row below is in, or reconstruction has stopped.
            if d < rows && (end || (self.complete(d) && self.complete(d + 1))) && !touches(d, d) {
                self.pic.deblock_row(self.work, d);
                st.deblocked += 1;
                continue;
            }
            let f = st.finals;
            if f < rows && (st.deblocked > f + 1 || st.deblocked == rows) && !touches(f, f + 1) {
                let upto = ((f + 1) * ctb).min(height);
                let dst = st.writer.pending_mut(upto);
                self.pic.sao_row(self.work, f, dst);
                st.finals += 1;
                st.writer
                    .publish(if st.finals == rows { usize::MAX } else { upto });
                if let Some(out) = &mut st.out {
                    out.take(&st.writer);
                }
                moved = true;
                continue;
            }
            break;
        }
        if moved {
            self.frame.notify();
        }
    }

    /// The motion of CTB row `row` kept for temporal prediction: each
    /// 16×16 block takes the motion of its top-left 4×4.
    fn motion_row(&self, st: &mut PipeState<P>, row: usize) {
        let pic = self.pic;
        let ctb = 1usize << self.work.log2_ctb;
        let y0 = row * ctb;
        let y1 = ((row + 1) * ctb).min(pic.height);
        let upto = if row + 1 == self.work.count {
            usize::MAX
        } else {
            y1
        };
        let [cells, _, _] = st.col.pending_mut(upto);
        let stride = pic.width.div_ceil(16);
        for (i, cell) in cells.iter_mut().enumerate() {
            let (bx, by) = (i % stride, y0 / 16 + i / stride);
            let (x, y) = (bx * 16, by * 16);
            *cell = ColMv::default();
            if y >= y1 {
                continue;
            }
            let field = pic.mv_at(x, y);
            if field.pred == 0 {
                continue;
            }
            let Some(slice) = pic.slice_of(x, y) else {
                continue;
            };
            cell.pred = field.pred;
            for list in 0..2 {
                if !field.uses(list) {
                    continue;
                }
                let Some(r) = usize::try_from(field.ref_idx[list])
                    .ok()
                    .and_then(|i| slice.refs[list].get(i))
                else {
                    continue;
                };
                cell.mv[list] = field.mv[list];
                cell.poc[list] = r.poc;
                cell.long_term[list] = r.long_term;
            }
        }
    }

    /// Filter and publish every row left, once reconstruction has stopped.
    fn finish(self) -> (RowsWriter<P>, Option<OutRows>) {
        let mut st = self
            .state
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let this = Pipeline {
            pic: self.pic,
            work: self.work,
            frame: self.frame,
            counts: self.counts,
            dirty: self.dirty,
            state: Mutex::new(PipeState {
                writer: RowsWriter::new([0; 3], [0; 3], [0; 3], None),
                col: RowsWriter::new([0; 3], [0; 3], [0; 3], None),
                motion: 0,
                deblocked: 0,
                finals: 0,
                out: None,
            }),
        };
        this.advance(&mut st, true, None);
        st.writer.publish(usize::MAX);
        st.col.publish(usize::MAX);
        this.frame.notify();
        let mut out = st.out.take();
        if let Some(out) = &mut out {
            out.take(&st.writer);
        }
        (st.writer, out)
    }
}
