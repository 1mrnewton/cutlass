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
use crate::pixel::{AnyRows, Pixel};
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

/// Whether a picture of samples above 8 bits is decoded with no samples
/// in its frame: no later picture references it, and its output is the
/// packed picture, uncropped, which its final rows are shifted into as
/// they come. The frame's rows are then published empty, in step.
pub fn staged(reference: bool, form: Option<OutForm>, sps: &Sps) -> bool {
    !reference && form == Some(OutForm::Packed) && sps.crop == [0; 4] && sps.bit_depth > 8
}

/// A picture to decode.
pub struct Job<P> {
    pub frame: Arc<Frame>,
    pub pic: Picture,
    pub work: Work<P>,
    pub writer: RowsWriter<P>,
    pub col: RowsWriter<ColMv>,
    pub slices: Vec<SliceData>,
    /// The output, when the picture is output. It is made on the thread
    /// that takes the pictures and frees them: the allocator then hands
    /// the same memory from picture to picture instead of giving it back
    /// to the system and taking it again, page by page.
    pub out: Option<OutRows>,
    /// The picture is output as its frame's own samples
    /// ([`OutSamples::Shared`]), and `out` is `None`.
    pub share: bool,
    /// Threads for the rows of a wavefront slice segment.
    pub row_threads: usize,
    /// Whether a later picture may reference this one. When not, its final
    /// samples need only reach the output.
    pub reference: bool,
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
            out,
            share,
            row_threads,
            reference,
        } = self;
        // A picture no other references, of 8-bit samples and not cropped,
        // has its final rows written straight to the output.
        let direct = staged(reference, out.as_ref().map(|o| o.form), &pic.sps)
            || (!reference
                && out.is_some()
                && pic.sps.crop == [0; 4]
                && P::of_bytes_mut(&mut []).is_some());
        let pipe = Pipeline::new(&pic, &work, &frame, writer, col, (out, direct));
        let mut result = Ok(());
        for (i, slice) in slices.iter().enumerate() {
            let last = i + 1 == slices.len();
            if let Err(e) = decode_slice(&pic, &work, &pipe, slice, (row_threads, last)) {
                result = Err(e);
                break;
            }
        }
        let (writer, out) = pipe.finish();
        let out = match &frame.rows {
            AnyRows::Eight(rows) if share => rows.whole().map(OutSamples::Shared),
            _ => out.map(OutRows::finish),
        };
        drop(writer);
        frame.finish(Done { result, out });
        Leftovers { pic, work }
    }
}

/// The cropped output samples of a picture, taken a band of rows at a
/// time as its final rows are published, while they are still in cache.
/// Packed output is one buffer of the picture's full size, its three planes
/// one after another, each row written in its place; it is a buffer of an
/// earlier picture when its holders have let it go (see
/// [`crate::dec::HevcDecoder`]), and every byte of it is written again.
/// Planes grow a band at a time.
pub struct OutRows {
    form: OutForm,
    /// Bits above 8 per sample.
    shift: u32,
    /// Left column, top row, width and height of each cropped plane.
    crop: [(usize, usize, usize, usize); 3],
    /// Rows of each cropped plane taken.
    next: [usize; 3],
    /// Packed: the three planes in the first buffer. Planes: one each.
    eight: [Vec<u8>; 3],
    deep: Option<[Vec<u16>; 3]>,
}

impl OutRows {
    /// `spare`: a buffer of an earlier packed picture, to fill again.
    pub fn new(sps: &Sps, form: OutForm, spare: Option<Vec<u8>>) -> Self {
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
        let total = size(0) + size(1) + size(2);
        let eight = match form {
            OutForm::Packed => [
                spare
                    .filter(|b| b.len() == total)
                    .unwrap_or_else(|| vec![0; total]),
                Vec::new(),
                Vec::new(),
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

    /// Rows `from..from + n` of plane `c`, to write: in place for packed
    /// output, appended for planes.
    fn rows_mut(&mut self, c: usize, from: usize, n: usize) -> &mut [u8] {
        let w = self.crop[c].2;
        match self.form {
            OutForm::Packed => {
                let base: usize = self.crop[..c].iter().map(|&(_, _, w, h)| w * h).sum();
                &mut self.eight[0][base + from * w..base + (from + n) * w]
            }
            OutForm::Planes => {
                let plane = &mut self.eight[c];
                plane.resize((from + n) * w, 0);
                &mut plane[from * w..]
            }
        }
    }

    /// Take the cropped rows published since the last call.
    fn take<P: Pixel>(&mut self, writer: &RowsWriter<P>) {
        let shared = writer.shared();
        for c in 0..3 {
            let (x0, y0, w, h) = self.crop[c];
            let next = self.next[c];
            if next >= h || w == 0 {
                continue;
            }
            let Some((rows, stride, have)) = shared.rows_from(c, y0 + next) else {
                continue;
            };
            let n = have.saturating_sub(y0 + next).min(h - next);
            let shift = self.shift;
            let dst = self.rows_mut(c, next, n);
            let row = |i: usize| &rows[i * stride + x0..i * stride + x0 + w];
            if x0 == 0
                && w == stride
                && let Some(bytes) = P::bytes(rows.get(..n * w).unwrap_or_default())
                && bytes.len() == dst.len()
            {
                // Uncropped rows lie end to end on both sides: one copy.
                dst.copy_from_slice(bytes);
            } else if P::bytes(&[]).is_some() {
                for (i, out) in dst.chunks_exact_mut(w).enumerate() {
                    if let Some(bytes) = P::bytes(row(i)) {
                        out.copy_from_slice(bytes);
                    }
                }
            } else {
                // Deeper samples, shifted down to 8 bits: compiled for
                // AVX2 where the CPU has it.
                cutlass_simd::wide::wide(
                    #[inline(always)]
                    || {
                        for (i, out) in dst.chunks_exact_mut(w).enumerate() {
                            if let Some(words) = P::words(row(i)) {
                                for (o, &s) in out.iter_mut().zip(words) {
                                    *o = (s >> shift) as u8;
                                }
                            }
                        }
                    },
                );
            }
            if let Some(deep) = &mut self.deep {
                for i in 0..n {
                    if let Some(words) = P::words(row(i)) {
                        deep[c].extend_from_slice(words);
                    }
                }
            }
            self.next[c] += n;
        }
    }

    /// The next `lines` rows of each plane, to write in place (the
    /// picture uncropped, its samples 8-bit): the rows a band of final
    /// samples fills.
    fn band_mut<P: Pixel>(&mut self, lines: [usize; 3]) -> Option<[&mut [P]; 3]> {
        let [y, u, v] = self.band_bytes(lines);
        Some([
            P::of_bytes_mut(y)?,
            P::of_bytes_mut(u)?,
            P::of_bytes_mut(v)?,
        ])
    }

    /// [`Self::band_mut`] as bytes, for samples of any depth shifted down.
    fn band_bytes(&mut self, lines: [usize; 3]) -> [&mut [u8]; 3] {
        let (crop, next) = (self.crop, self.next);
        let mut taken = [0; 3];
        for c in 0..3 {
            taken[c] = lines[c].min(crop[c].3 - next[c]);
            self.next[c] += taken[c];
            if self.form == OutForm::Planes {
                self.eight[c].resize((next[c] + taken[c]) * crop[c].2, 0);
            }
        }
        let rows = |c: usize| next[c] * crop[c].2..(next[c] + taken[c]) * crop[c].2;
        // Clearing the band first brings its lines into cache without
        // reading them, before the filters store into every sample.
        let [y, u, v] = match self.form {
            OutForm::Packed => {
                let (y, rest) = self.eight[0].split_at_mut(crop[0].2 * crop[0].3);
                let (u, v) = rest.split_at_mut(crop[1].2 * crop[1].3);
                [y, u, v]
            }
            OutForm::Planes => {
                let [y, u, v] = &mut self.eight;
                [&mut y[..], &mut u[..], &mut v[..]]
            }
        };
        let (y, u, v) = (&mut y[rows(0)], &mut u[rows(1)], &mut v[rows(2)]);
        if self.form == OutForm::Packed {
            for band in [&mut *y, &mut *u, &mut *v] {
                band.fill(0);
            }
        }
        [y, u, v]
    }

    /// The samples; for planes, rows never published are zeros.
    fn finish(mut self) -> OutSamples {
        match self.form {
            OutForm::Packed => {
                let [bytes, _, _] = self.eight;
                OutSamples::Packed(bytes)
            }
            OutForm::Planes => {
                for c in 0..3 {
                    let (_, _, w, h) = self.crop[c];
                    self.eight[c].resize(w * h, 0);
                    if let Some(deep) = &mut self.deep {
                        deep[c].resize(w * h, 0);
                    }
                }
                OutSamples::Planes {
                    eight: self.eight,
                    deep: self.deep,
                }
            }
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
    (threads, last): (usize, bool),
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
    // In the picture's last segment, a worker without a row left frees
    // its thread for the next picture.
    let idle = || {
        if last {
            pipe.frame.release();
        }
    };
    let results = in_parallel(rows, threads, idle, |r| {
        let _done = RowDone(&gates[r]);
        let wave = Wave {
            row: first_row + r,
            above: r.checked_sub(1).map(|above| &gates[above]),
            mine: &gates[r],
            row_only: r + 1 < rows,
        };
        let start = (r > 0).then_some((first_row + r) * ctb_w);
        pipe.wait_slot(first_row + r);
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
        if result.is_err() {
            pipe.row_failed();
        }
        pipe.nudge(None);
        result
    });
    results.into_iter().collect()
}

/// Run `count` pieces on up to `threads` threads, this one among them,
/// each thread taking the next piece in order. Results in index order.
/// `idle` runs once, when the first thread finds no piece left.
fn in_parallel<T: Send>(
    count: usize,
    threads: usize,
    idle: impl Fn() + Sync,
    work: impl Fn(usize) -> T + Sync,
) -> Vec<T> {
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
                if index == count {
                    idle();
                }
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
            ctb_near: [false; 3],
            merge_2nx2n: false,
            slice_type: self.slice_type,
            mc: crate::mc::Scratch::default(),
            seen_rows: [[0; 16]; 2],
            seen_motion: std::cell::Cell::new(0),
        }
    }
}

/// The loop filters and the publication of one picture, run behind its
/// reconstruction. Every CTB reconstructed is counted; a CTB row whose
/// CTBs are all in has its motion published and is deblocked, and once
/// the row below is deblocked it gets SAO and its final samples are
/// published.
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
    /// A row worker failed: rows below it may never be filtered, so
    /// nothing waits for them.
    failed: AtomicBool,
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
    /// SAO writes the final rows to `out` instead of the frame, whose rows
    /// no picture reads (they are published all the same, in step).
    direct: bool,
    /// For samples above 8 bits written to `out`: a band of them, shifted
    /// down from here (the frame then holds no samples).
    stage: [Vec<P>; 3],
}

impl<'p, P: Pixel> Pipeline<'p, P> {
    fn new(
        pic: &'p Picture,
        work: &'p Work<P>,
        frame: &'p Frame,
        writer: RowsWriter<P>,
        col: RowsWriter<ColMv>,
        (out, direct): (Option<OutRows>, bool),
    ) -> Self {
        Self {
            pic,
            work,
            frame,
            counts: (0..work.count).map(|_| AtomicU32::new(0)).collect(),
            dirty: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            state: Mutex::new(PipeState {
                writer,
                col,
                motion: 0,
                deblocked: 0,
                finals: 0,
                out,
                direct,
                stage: Default::default(),
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

    /// Wait until the row buffer of CTB row `row` is free: the final
    /// samples of the row that used it last, and of the row below that
    /// one, whose filters read it, are out. With the rows on as many
    /// threads as the ring has room for, they are by the time `row`
    /// starts; this holds the order if a thread is late.
    pub fn wait_slot(&self, row: usize) {
        let slots = self.work.rows.len();
        if slots >= self.work.count || row < slots {
            return;
        }
        let ctb = 1usize << self.work.log2_ctb;
        self.nudge(None);
        // The band of the row below the one that used the buffer last
        // reads it; that band ends `HOLD` lines short of its row.
        let upto = ((row + 2 - slots) * ctb).saturating_sub(crate::filter::HOLD);
        self.frame.wait_rows_or(upto, &self.failed);
    }

    /// A row worker failed: release whoever waits for a row buffer.
    pub fn row_failed(&self) {
        self.failed.store(true, SeqCst);
        self.frame.notify();
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
            // Row `d` is deblocked once it is reconstructed and its motion
            // kept: the row below reads the line above it from the copy
            // saved before the filters (`Samples::above`), never the row
            // itself, so the filters follow reconstruction by one row and
            // find its samples still in cache. The motion goes first
            // because publishing the row frees its place in the 4×4 maps,
            // and a row that came in after the motion loop above has not
            // been kept yet. (`motion` passes only complete rows.)
            if d < st.motion && !touches(d, d) {
                self.pic.deblock_row(self.work, d);
                st.deblocked += 1;
                continue;
            }
            // The final lines of row `f` (see `Picture::sao_row`) follow its
            // deblocking: all but its last few, and those of the row above.
            let f = st.finals;
            if f < rows && st.deblocked > f && !touches(f, f) {
                let (from, upto) = self.pic.sao_lines(self.work, f);
                let lines = upto - from;
                // Packed output takes luma in place and leaves chroma in the
                // frame's buffers.
                let band = [lines, lines / 2, lines / 2];
                let direct = match &mut st.out {
                    Some(out) if st.direct && P::bytes(&[]).is_some() => {
                        out.band_mut::<P>(band).is_some_and(|dst| {
                            self.pic.sao_row(self.work, f, dst);
                            true
                        })
                    }
                    Some(out) if st.direct => {
                        // Deeper samples: the band in 16 bits, then shifted
                        // down into the output.
                        let width = self.pic.width;
                        let sizes = [lines * width, band[1] * (width / 2), band[2] * (width / 2)];
                        // SAO writes every sample of the band, and the
                        // buffer stays in cache from band to band: sized,
                        // not cleared.
                        for (s, n) in st.stage.iter_mut().zip(sizes) {
                            s.resize(n, P::default());
                        }
                        let [a, b, c] = &mut st.stage;
                        self.pic
                            .sao_row(self.work, f, [&mut a[..], &mut b[..], &mut c[..]]);
                        let shift = out.shift;
                        let dst = out.band_bytes(band);
                        cutlass_simd::wide::wide(
                            #[inline(always)]
                            || {
                                for (d, s) in dst.into_iter().zip(&st.stage) {
                                    if let Some(s) = P::words(s) {
                                        for (o, &v) in d.iter_mut().zip(s) {
                                            *o = (v >> shift) as u8;
                                        }
                                    }
                                }
                            },
                        );
                        true
                    }
                    _ => false,
                };
                if !direct {
                    let mut dst = st.writer.pending_mut(upto);
                    // As for the output: clearing the band first brings its
                    // lines into cache without reading them.
                    for band in &mut dst {
                        band.fill(P::default());
                    }
                    self.pic.sao_row(self.work, f, dst);
                }
                st.finals += 1;
                st.writer
                    .publish(if st.finals == rows { usize::MAX } else { upto });
                if !direct && let Some(out) = &mut st.out {
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
            failed: self.failed,
            state: Mutex::new(PipeState {
                writer: RowsWriter::new([0; 3], [0; 3], [0; 3], None),
                col: RowsWriter::new([0; 3], [0; 3], [0; 3], None),
                motion: 0,
                deblocked: 0,
                finals: 0,
                out: None,
                direct: false,
                stage: Default::default(),
            }),
        };
        this.advance(&mut st, true, None);
        st.writer.publish(usize::MAX);
        st.col.publish(usize::MAX);
        this.frame.notify();
        let mut out = st.out.take();
        if let Some(out) = &mut out
            && !st.direct
        {
            out.take(&st.writer);
        }
        (st.writer, out)
    }
}
