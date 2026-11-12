//! Frame-level decoding: OBU dispatch, reference frames, tiles, filters and output.

use std::collections::VecDeque;
use std::ops::Range;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError};

use cutlass_decode::CodecError;
use cutlass_simd::shared_rows::{PlaneBuf, RowsWriter, SharedRows};

use crate::dsp::Pixel;
use crate::dsp::fg::{FilmGrainParams, apply_film_grain};
use crate::dsp::lr::LrScratch;

use super::cdf::CdfContext;
use super::frame::{FrameState, LrUnitParams, MF_BAND_ROWS, MfBand, Picture, Plane, RefData};
use super::msac::Msac;
use super::mvpred::{MfPlan, MfShared, motion_field_plan};
use super::obu::{
    FrameHeader, GrainParams, HeaderContext, KEY_FRAME, OBU_FRAME, OBU_FRAME_HEADER,
    OBU_REDUNDANT_FRAME_HEADER, OBU_SEQUENCE_HEADER, OBU_TEMPORAL_DELIMITER, OBU_TILE_GROUP,
    OBU_TILE_LIST, PRIMARY_REF_NONE, RESTORE_NONE, RefHeaderState, SequenceHeader,
    parse_frame_header, parse_sequence_header, relative_dist, split_obus,
};
use super::postfilter;
use super::recon::Scratch;
use super::tile::{Block, MiFill, ReconLog, RefView, RowHook, TileDec, lr_unit_span, malformed};
use super::types::{INTRA_FRAME, MiInfo, Mv, bw4};

type R<T> = Result<T, CodecError>;

/// A decoded picture of either sample size.
#[derive(Clone)]
pub enum AnyPicture {
    U8(Picture<u8>),
    U16(Picture<u16>),
}

/// The final rows of a frame still being decoded, of either sample size
/// (see [`Mirror`]).
#[derive(Clone)]
pub enum AnyShared {
    U8(Arc<SharedRows<u8>>),
    U16(Arc<SharedRows<u16>>),
}

/// A sample type that can be viewed inside an [`AnyPicture`].
pub trait PicturePixel: Pixel {
    fn view(p: &AnyPicture) -> Option<&Picture<Self>>;
    fn wrap(p: Picture<Self>) -> AnyPicture;
    fn view_shared(r: &AnyShared) -> Option<&Arc<SharedRows<Self>>>;
    fn wrap_shared(r: Arc<SharedRows<Self>>) -> AnyShared;
}

impl PicturePixel for u8 {
    fn view(p: &AnyPicture) -> Option<&Picture<u8>> {
        match p {
            AnyPicture::U8(x) => Some(x),
            AnyPicture::U16(_) => None,
        }
    }
    fn wrap(p: Picture<u8>) -> AnyPicture {
        AnyPicture::U8(p)
    }
    fn view_shared(r: &AnyShared) -> Option<&Arc<SharedRows<u8>>> {
        match r {
            AnyShared::U8(x) => Some(x),
            AnyShared::U16(_) => None,
        }
    }
    fn wrap_shared(r: Arc<SharedRows<u8>>) -> AnyShared {
        AnyShared::U8(r)
    }
}

impl PicturePixel for u16 {
    fn view(p: &AnyPicture) -> Option<&Picture<u16>> {
        match p {
            AnyPicture::U16(x) => Some(x),
            AnyPicture::U8(_) => None,
        }
    }
    fn wrap(p: Picture<u16>) -> AnyPicture {
        AnyPicture::U16(p)
    }
    fn view_shared(r: &AnyShared) -> Option<&Arc<SharedRows<u16>>> {
        match r {
            AnyShared::U16(x) => Some(x),
            AnyShared::U8(_) => None,
        }
    }
    fn wrap_shared(r: Arc<SharedRows<u16>>) -> AnyShared {
        AnyShared::U16(r)
    }
}

/// One shown frame.
#[derive(Clone)]
pub struct DecodedFrame {
    pub picture: Arc<AnyPicture>,
    pub width: usize,
    pub height: usize,
    pub render_width: usize,
    pub render_height: usize,
    pub bit_depth: u32,
    pub mono: bool,
    pub sub_x: bool,
    pub sub_y: bool,
    pub color_primaries: u8,
    pub transfer: u8,
    pub matrix: u8,
    pub full_range: bool,
    /// `chroma_sample_position` (0 unknown, 1 vertical, 2 colocated).
    pub chroma_sample_position: u8,
    /// Sequence timing info; zero when the sequence has none.
    pub num_units_in_display_tick: u32,
    pub time_scale: u32,
    pub equal_picture_interval: bool,
    /// Present when the picture interval is equal.
    pub num_ticks_per_picture_minus_1: Option<u32>,
    /// Zero when the sequence has no decoder model info.
    pub num_units_in_decoding_tick: u32,
    /// `seq_level_idx` of the operating point decoded.
    pub seq_level_idx: u8,
}

/// What a frame's parse leaves for later frames (its CDFs, its motion
/// field and segmentation data), once that parse gets there.
struct Promise<T> {
    v: Mutex<Option<R<T>>>,
    cv: Condvar,
}

impl<T: Clone> Promise<T> {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            v: Mutex::new(None),
            cv: Condvar::new(),
        })
    }

    fn ready(r: R<T>) -> Arc<Self> {
        Arc::new(Self {
            v: Mutex::new(Some(r)),
            cv: Condvar::new(),
        })
    }

    /// Sets the value (only the first call counts).
    fn set(&self, r: R<T>) {
        let mut v = lock(&self.v);
        if v.is_none() {
            *v = Some(r);
            self.cv.notify_all();
        }
    }

    fn wait(&self) -> R<T> {
        let mut v = lock(&self.v);
        loop {
            if let Some(r) = &*v {
                return r.clone();
            }
            v = self.cv.wait(v).unwrap_or_else(PoisonError::into_inner);
        }
    }
}

type CdfPromise = Arc<Promise<Arc<CdfContext>>>;
type DataPromise = Arc<Promise<Arc<RefData>>>;

#[derive(Clone)]
struct Slot {
    pic: Arc<FrameJob>,
    data: DataPromise,
    cdf: CdfPromise,
}

mod parse;
mod tiles;

/// Time spent in the stages of the threaded decoder (summed over threads),
/// gathered with the `prof` feature.
pub mod prof {
    use std::sync::atomic::{AtomicU64, Ordering};
    /// The calling thread waiting for pictures.
    pub const WAIT_OUT: usize = 0;
    /// Reconstruction waiting for its references.
    pub const WAIT_REFS: usize = 1;
    /// Parsing on the calling thread (threaded pipeline only).
    pub const PARSE: usize = 2;
    /// Reconstruction of parsed blocks.
    pub const RECON: usize = 3;
    /// In-loop filters.
    pub const FILTER: usize = 4;
    /// Copying final rows for the frames referencing one still being
    /// decoded (part of `filter`).
    pub const PUBLISH: usize = 5;
    /// Copying them into the copies of the references.
    pub const MIRROR: usize = 6;
    /// Decoding the tiles of a frame the pipeline does not take, from the
    /// first tile to the last (the frame's wall time, on any number of
    /// threads).
    pub const TILES: usize = 7;
    /// Copying the tiles decoded into shells into the frame (part of
    /// `tiles`).
    pub const TILE_COPY: usize = 8;
    const NAMES: [&str; 9] = [
        "wait_out",
        "wait_refs",
        "parse",
        "recon",
        "filter",
        "publish",
        "mirror",
        "tiles",
        "tile_copy",
    ];
    static T: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];

    /// Adds its lifetime to a stage.
    pub struct Span(pub usize, pub std::time::Instant);

    impl Drop for Span {
        fn drop(&mut self) {
            add(self.0, self.1);
        }
    }

    /// Adds the time since `t` to stage `k`.
    #[inline]
    pub fn add(k: usize, t: std::time::Instant) {
        if cfg!(feature = "prof") {
            T[k].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            mark(k, t);
        }
    }

    /// Spans on the timeline only (see [`mark`]), past the stages: a
    /// frame's backend from start to end, a parse waiting for the CDFs of
    /// its primary reference frame, the moment a frame's CDFs are ready,
    /// and a parse waiting for a band of the motion field.
    pub const BACKEND: usize = 9;
    pub const CDF_WAIT: usize = 10;
    pub const CDF_SET: usize = 11;
    pub const MF_WAIT: usize = 12;
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    static LINE: std::sync::Mutex<Vec<(usize, u64, u64, u64)>> = std::sync::Mutex::new(Vec::new());
    static NEXT_THREAD: AtomicU64 = AtomicU64::new(0);
    thread_local! {
        static THREAD: u64 = NEXT_THREAD.fetch_add(1, Ordering::Relaxed);
    }

    /// Records the span of kind `k` from `t` to now, and the thread, on
    /// the timeline [`report`] writes to the file `AV1_TIMELINE` names, as
    /// lines of the kind, start and end in nanoseconds, and thread.
    pub fn mark(k: usize, t: std::time::Instant) {
        if !cfg!(feature = "prof") {
            return;
        }
        let s = *START.get_or_init(std::time::Instant::now);
        let a = t.saturating_duration_since(s).as_nanos() as u64;
        let b = s.elapsed().as_nanos() as u64;
        let th = THREAD.with(|t| *t);
        if let Ok(mut l) = LINE.lock() {
            l.push((k, a, b, th));
        }
    }

    /// Threads whose processor time [`cpu`] sums.
    pub const CPU_MAIN: usize = 0;
    pub const CPU_TASK: usize = 1;
    pub const CPU_BACKEND: usize = 2;
    pub const CPU_MF: usize = 3;
    /// Tasks of frames no later frame references.
    pub const CPU_TASK_UNREF: usize = 4;
    const CPU_NAMES: [&str; 5] = ["main", "task", "backend", "mf", "task_unref"];
    static CPU: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
    static FLT: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

    /// The processor time of the calling thread so far, in nanoseconds,
    /// and its minor page faults (0 without the `prof` feature).
    pub fn thread_cpu() -> (u64, u64) {
        if !cfg!(feature = "prof") {
            return (0, 0);
        }
        let t = std::fs::read_to_string("/proc/thread-self/schedstat")
            .ok()
            .and_then(|s| s.split(' ').next().and_then(|v| v.parse::<u64>().ok()))
            .unwrap_or(0);
        // Field 10 of the thread's stat.
        let f = std::fs::read_to_string("/proc/thread-self/stat")
            .ok()
            .and_then(|st| {
                st.rsplit(')')
                    .next()
                    .and_then(|r| r.split(' ').nth(8).and_then(|v| v.parse::<u64>().ok()))
            })
            .unwrap_or(0);
        (t, f)
    }

    /// Adds the processor time and page faults of the calling thread since
    /// `t0` (a [`thread_cpu`]) to kind `k`.
    pub fn cpu_since(k: usize, t0: (u64, u64)) {
        if cfg!(feature = "prof") {
            let (t, f) = thread_cpu();
            CPU[k].fetch_add(t.saturating_sub(t0.0), Ordering::Relaxed);
            FLT[k].fetch_add(f.saturating_sub(t0.1), Ordering::Relaxed);
        }
    }

    /// Adds the processor time of the calling thread so far to kind `k`
    /// (call it as the thread ends).
    pub fn cpu(k: usize) {
        cpu_since(k, (0, 0));
    }

    /// Allocations of the pooled buffers (pool misses).
    pub const NEW_PIC: usize = 0;
    pub const NEW_STATE: usize = 1;
    pub const NEW_SHELL: usize = 2;
    pub const NEW_LOG: usize = 3;
    pub const NEW_BAND: usize = 4;
    pub const NEW_MIRROR: usize = 5;
    const NEW_NAMES: [&str; 6] = ["pic", "state", "shell", "log", "band", "mirror"];
    static NEW: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

    /// Counts an allocation of kind `k`.
    #[inline]
    pub fn count(k: usize) {
        if cfg!(feature = "prof") {
            NEW[k].fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn report() -> String {
        if let Some(path) = std::env::var_os("AV1_TIMELINE")
            && let Ok(l) = LINE.lock()
        {
            let text: String = l
                .iter()
                .map(|(k, a, b, t)| format!("{k} {a} {b} {t}\n"))
                .collect();
            let _ = std::fs::write(path, text);
        }
        cpu(CPU_MAIN);
        let mut s: String = NAMES
            .iter()
            .zip(&T)
            .map(|(n, v)| format!("{n}={:.3}s ", v.load(Ordering::Relaxed) as f64 * 1e-9))
            .collect();
        // Minor page faults of the process (field 10 of its stat).
        if let Ok(st) = std::fs::read_to_string("/proc/self/stat")
            && let Some(rest) = st.rsplit(')').next()
            && let Some(f) = rest.split(' ').nth(8)
        {
            s.push_str(&format!("faults={f} "));
        }
        // Peak resident memory.
        if let Ok(st) = std::fs::read_to_string("/proc/self/status")
            && let Some(l) = st.lines().find(|l| l.starts_with("VmHWM:"))
        {
            s.push_str(&format!("hwm={}kB ", l[6..].trim().trim_end_matches(" kB")));
        }
        s.push_str("| new ");
        for (n, v) in NEW_NAMES.iter().zip(&NEW) {
            s.push_str(&format!("{n}={} ", v.load(Ordering::Relaxed)));
        }
        s.push_str("| cpu ");
        for ((n, v), f) in CPU_NAMES.iter().zip(&CPU).zip(&FLT) {
            s.push_str(&format!(
                "{n}={:.3}s/{} ",
                v.load(Ordering::Relaxed) as f64 * 1e-9,
                f.load(Ordering::Relaxed)
            ));
        }
        s
    }
}

/// Locks `m`, also after a thread holding it panicked (every user leaves
/// the data consistent between statements).
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The rows of a reference still being decoded that reconstruction reads
/// ([`RefView::mirror`]): its final rows so far, read in place.
pub struct Mirror<P> {
    shared: Option<Arc<SharedRows<P>>>,
}

impl<P> Default for Mirror<P> {
    fn default() -> Self {
        Self { shared: None }
    }
}

impl<P> Mirror<P> {
    /// Rows `y0..y1` of plane `p` when they are final: the samples from
    /// row `y0` on, and their stride.
    #[inline]
    pub fn rows(&self, p: usize, y0: usize, y1: usize) -> Option<(&[P], usize)> {
        self.shared.as_ref()?.rows(p, y0, y1)
    }

    /// The final rows of plane `p` from row `y` on: their samples, their
    /// stride, and the row past the last of them.
    #[inline]
    pub fn rows_from(&self, p: usize, y: usize) -> Option<(&[P], usize, usize)> {
        self.shared.as_ref()?.rows_from(p, y)
    }
}

impl<P: PicturePixel> Mirror<P> {
    /// Drops the rows, handing their buffers back to `spares` when no
    /// other thread holds them.
    fn release(&mut self, spares: &Mutex<Spares<P>>) {
        if let Some(sh) = self.shared.take() {
            lock(spares).give_shared(sh);
        }
    }
}

struct JobState {
    result: Option<R<Arc<AnyPicture>>>,
    /// While decoding: the final rows so far, luma rows `0..rows`.
    shared: Option<AnyShared>,
    rows: usize,
    /// Whether a frame reads the rows as they become final (else they
    /// are not published).
    watched: bool,
    /// Luma rows a reader asked for so far, with some to spare: rows
    /// below are published only once one is asked for.
    want: usize,
    /// Told when the job ends (see [`Av1Core::limit_in_flight`]).
    ended: Option<Arc<Ended>>,
}

/// How many of the frame jobs that tell it have ended: the decoder waits
/// on it for any frame decoding to end.
#[derive(Default)]
struct Ended {
    n: Mutex<u64>,
    cv: Condvar,
}

/// The picture of a frame whose reconstruction and in-loop filters may
/// still be running on other threads, and its rows as they become final.
struct FrameJob {
    state: Mutex<JobState>,
    cv: Condvar,
}

impl FrameJob {
    fn with(result: Option<R<Arc<AnyPicture>>>) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(JobState {
                result,
                shared: None,
                rows: 0,
                watched: false,
                want: 0,
                ended: None,
            }),
            cv: Condvar::new(),
        })
    }

    fn running() -> Arc<Self> {
        Self::with(None)
    }

    fn ready(r: R<Arc<AnyPicture>>) -> Arc<Self> {
        Self::with(Some(r))
    }

    /// Publishes the result (only the first call counts).
    fn finish(&self, r: R<Arc<AnyPicture>>) {
        drop(self.finish_take(r));
    }

    /// [`Self::finish`], returning the rows published before.
    fn finish_take(&self, r: R<Arc<AnyPicture>>) -> Option<AnyShared> {
        let mut s = lock(&self.state);
        if s.result.is_none() {
            s.result = Some(r);
            self.cv.notify_all();
            if let Some(e) = &s.ended {
                *lock(&e.n) += 1;
                e.cv.notify_all();
            }
            return s.shared.take();
        }
        None
    }

    /// Publishes the final rows `0..rows` of `shared`.
    fn publish(&self, shared: &AnyShared, rows: usize) {
        let mut s = lock(&self.state);
        if s.result.is_none() && rows > s.rows {
            if s.shared.is_none() {
                s.shared = Some(shared.clone());
            }
            s.rows = rows;
            self.cv.notify_all();
        }
    }

    /// Whether the rows are to be published.
    fn watched(&self, published: usize) -> bool {
        let s = lock(&self.state);
        s.watched && s.want > published
    }

    /// The result, or `None` after asking for the rows to be published.
    fn poll_or_watch(&self) -> Option<R<Arc<AnyPicture>>> {
        let mut s = lock(&self.state);
        if s.result.is_none() {
            s.watched = true;
        }
        s.result.clone()
    }

    fn poll(&self) -> Option<R<Arc<AnyPicture>>> {
        lock(&self.state).result.clone()
    }

    fn is_done(&self) -> bool {
        lock(&self.state).result.is_some()
    }

    /// Waits for the result.
    fn wait(&self) -> R<Arc<AnyPicture>> {
        let t = std::time::Instant::now();
        let mut s = lock(&self.state);
        loop {
            if let Some(r) = &s.result {
                prof::add(prof::WAIT_OUT, t);
                return r.clone();
            }
            s = self.cv.wait(s).unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// Whether the job is done and nothing else holds its picture.
    fn pic_unshared(&self) -> bool {
        matches!(&lock(&self.state).result, Some(Ok(p)) if Arc::strong_count(p) == 1)
    }

    fn into_result(self) -> Option<R<Arc<AnyPicture>>> {
        self.state
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner)
            .result
    }
}

/// The writer of the final rows of `pic` (its visible samples), in the
/// buffers `bufs` when they fit.
fn shared_rows_of<P: PicturePixel>(
    pic: &Picture<P>,
    bufs: Option<[PlaneBuf<P>; 3]>,
) -> RowsWriter<P> {
    let mut widths = [0; 3];
    let mut heights = [0; 3];
    let mut shifts = [0; 3];
    for (p, pl) in pic.planes.iter().enumerate().take(3) {
        widths[p] = pl.w.min(pl.stride);
        heights[p] = pl.h.min(pl.alloc_h);
        shifts[p] = u32::from(p > 0 && pic.sub_y);
    }
    RowsWriter::new(widths, heights, shifts, bufs)
}

/// The rows of a reference still being decoded ([`RefView::mirror`]),
/// brought up to the rows reconstruction reads.
#[derive(Clone)]
struct RefFill {
    job: Arc<FrameJob>,
    /// Reference frame (1..=7) of the view; other views of the same job
    /// share the mirror.
    ref_frame: usize,
    /// Index of the bands in [`TileDec::mirrors`].
    mirror: usize,
    /// Luma rows of the job known to be final.
    have: usize,
    /// Every read waits for the whole picture (scaled reference).
    whole: bool,
}

/// Luma rows past those it needs that a reader asks a reference for, so
/// that they are published before it waits for them.
const WANT_AHEAD: usize = 512;

impl RefFill {
    /// Waits until the reference has its luma rows `0..need` final (all
    /// of it for `usize::MAX`) and adds their bands to `m`. Once the
    /// reference is done, returns its picture instead (to be read in
    /// place) and drops the bands.
    fn fill<P: PicturePixel>(
        &mut self,
        m: &mut Mirror<P>,
        need: usize,
        spares: &Mutex<Spares<P>>,
    ) -> R<Option<Arc<AnyPicture>>> {
        if self.have >= need {
            return Ok(None);
        }
        let t = std::time::Instant::now();
        let (shared, rows, fin) = {
            let mut s = lock(&self.job.state);
            s.want = s.want.max(need.saturating_add(WANT_AHEAD));
            while s.result.is_none() && s.rows < need {
                s = self.job.cv.wait(s).unwrap_or_else(PoisonError::into_inner);
            }
            let shared = match (&s.result, &m.shared) {
                (None, None) => s.shared.clone(),
                _ => None,
            };
            (shared, s.rows, s.result.clone())
        };
        prof::add(prof::WAIT_REFS, t);
        if let Some(r) = fin {
            m.release(spares);
            return self.done(r);
        }
        if let Some(sh) = shared {
            match P::view_shared(&sh) {
                Some(sh) => m.shared = Some(sh.clone()),
                None => {
                    // Not rows this reader can use: take the whole picture.
                    m.release(spares);
                    return self.done(self.job.wait());
                }
            }
        }
        self.have = rows;
        Ok(None)
    }

    fn done(&mut self, r: R<Arc<AnyPicture>>) -> R<Option<Arc<AnyPicture>>> {
        self.have = usize::MAX;
        r.map(Some)
    }

    /// [`Self::fill`] into the mirror of this reference; once the
    /// reference is done, the views of it in `refs` read its picture in
    /// place (kept in `finals`) instead.
    fn bring<'r, P: PicturePixel>(
        &mut self,
        need: usize,
        mirrors: &mut [Mirror<P>],
        refs: &mut [Option<RefView<'r, P>>; 8],
        finals: &'r [OnceLock<Arc<AnyPicture>>],
        spares: &Mutex<Spares<P>>,
    ) -> R<()> {
        let Some(m) = mirrors.get_mut(self.mirror) else {
            return Ok(());
        };
        let Some(pic) = self.fill(m, need, spares)? else {
            return Ok(());
        };
        let Some(fin) = finals.get(self.mirror) else {
            return Ok(());
        };
        let fin = fin.get_or_init(|| pic);
        let Some(view) = P::view(fin) else {
            return Err(malformed("reference sample size"));
        };
        for rv in refs.iter_mut().flatten() {
            if rv.mirror == Some(self.mirror) {
                rv.pic = view;
                rv.mirror = None;
            }
        }
        Ok(())
    }
}

/// Finishes a job with an error if its thread ends without a result.
struct FinishGuard(Arc<FrameJob>);

impl Drop for FinishGuard {
    fn drop(&mut self) {
        self.0.finish(Err(malformed("decoding thread")));
    }
}

/// A shown frame waiting for its picture.
struct Queued {
    job: Arc<FrameJob>,
    seq: Arc<SequenceHeader>,
    rh: RefHeaderState,
    grain: GrainParams,
    tag: u64,
}

type Job = Box<dyn FnOnce() + Send>;

#[derive(Default)]
struct WorkQueue {
    jobs: VecDeque<Job>,
    idle: usize,
    closed: bool,
}

/// The threads frame tasks and their backends run on, kept between frames
/// (their stacks and allocator caches stay warm). A job goes to an idle
/// thread, or to a new one when none is idle, so jobs that block waiting
/// for each other never wait for a thread.
#[derive(Default)]
struct Workers {
    q: Mutex<WorkQueue>,
    cv: Condvar,
}

impl Workers {
    fn run(self: &Arc<Self>, job: Job) {
        let mut q = lock(&self.q);
        q.jobs.push_back(job);
        if q.idle >= q.jobs.len() {
            self.cv.notify_one();
            return;
        }
        drop(q);
        let w = self.clone();
        std::thread::spawn(move || w.serve());
    }

    fn serve(&self) {
        loop {
            let job = {
                let mut q = lock(&self.q);
                loop {
                    if let Some(j) = q.jobs.pop_front() {
                        break j;
                    }
                    if q.closed {
                        return;
                    }
                    q.idle += 1;
                    q = self.cv.wait(q).unwrap_or_else(PoisonError::into_inner);
                    q.idle -= 1;
                }
            };
            job();
        }
    }

    /// Lets the threads end once the queued jobs are done.
    fn close(&self) {
        lock(&self.q).closed = true;
        self.cv.notify_all();
    }
}

/// What finished frame jobs hand back for reuse.
struct Spares<P> {
    /// Pictures nothing holds any more (see [`recycle`]).
    pics: Vec<Picture<P>>,
    /// Frame states of reconstruction ([`FrameState::recon_shell`]) and
    /// of the filters ([`FrameState::filter_shell`]).
    shells: Vec<FrameState>,
    filter_shells: Vec<FrameState>,
    lr: Vec<LrScratch>,
    /// Frame states of parsed frames.
    states: Vec<FrameState>,
    /// Buffers of rows sent to the filters ([`copy_rows_out`]).
    bands: Vec<[Vec<P>; 3]>,
    /// Buffers of the final rows of frames ([`SharedRows`]).
    shared: Vec<[PlaneBuf<P>; 3]>,
    /// Pictures of a superblock row (see [`take_sb_row`]).
    sb_rows: Vec<Picture<P>>,
    /// Reconstruction scratch of the backends.
    scratch: Vec<Scratch<P>>,
    /// The saved boundary rows of the in-loop filters of the backends.
    filters: Vec<FilterRows<P>>,
    /// Shells of the threads decoding tiles (see [`tiles`]), and how many
    /// are lent out.
    tile_shells: Vec<tiles::TileShell<P>>,
    tile_shells_out: usize,
    /// Whether reconstruction waits for the rows of the references a
    /// chunk of blocks reads before it starts on the chunk (tests only).
    /// Without it every block stops at the first rows that are not
    /// published and is taken up again once they are, as a warped block
    /// does when it reads rows its motion vector does not name: the first
    /// blocks of a chunk then go on while the reference makes the rows the
    /// last ones read, and the chunk waits for no row of the vector of
    /// another block (at 18 threads 3% faster on a 4K clip with CDEF).
    rows_ahead: bool,
    /// Times a block stopped for rows that were not published yet.
    row_waits: usize,
}

/// The CDEF scratch and loop restoration boundaries of an [`InlineFilter`].
type FilterRows<P> = (
    postfilter::CdefScratch<P>,
    postfilter::LrBounds<P>,
    postfilter::FilterInfo,
);

/// Band buffers kept for reuse, at most.
const MAX_BANDS: usize = 96;

/// Buffers of final rows ([`Spares::shared`]) kept for reuse, at most.
const MAX_SHARED: usize = 16;

/// Superblock rows ([`Spares::sb_rows`]) kept for reuse, at most.
const MAX_SB_ROWS: usize = 64;

impl<P: PicturePixel> Spares<P> {
    /// Keeps the buffers of `b` for reuse when this is its last holder
    /// (the publishing frame and each frame reading it drop theirs).
    fn give_shared(&mut self, b: Arc<SharedRows<P>>) {
        if let Some(b) = Arc::into_inner(b)
            && self.shared.len() < MAX_SHARED
        {
            self.shared.push(b.into_buffers());
        }
    }
}

/// Per sample type: the row message pools of the threaded pipeline and
/// the spares (among them the picture pool), shared with the threads.
struct PixState<P> {
    rows: Arc<Mutex<RowPools>>,
    spares: Arc<Mutex<Spares<P>>>,
}

impl<P> Default for PixState<P> {
    fn default() -> Self {
        Self {
            rows: Arc::new(Mutex::new(RowPools::default())),
            spares: Arc::new(Mutex::new(Spares {
                pics: Vec::new(),
                shells: Vec::new(),
                filter_shells: Vec::new(),
                lr: Vec::new(),
                states: Vec::new(),
                bands: Vec::new(),
                shared: Vec::new(),
                sb_rows: Vec::new(),
                scratch: Vec::new(),
                filters: Vec::new(),
                tile_shells: Vec::new(),
                tile_shells_out: 0,
                rows_ahead: false,
                row_waits: 0,
            })),
        }
    }
}

impl<P: Pixel> PixState<P> {
    /// Trims the spares.
    fn collect(&mut self) {
        let mut sp = lock(&self.spares);
        sp.shells.truncate(MAX_IN_FLIGHT);
        sp.filter_shells.truncate(MAX_IN_FLIGHT);
        sp.lr.truncate(4);
        sp.states.truncate(16);
        sp.scratch.truncate(8);
        sp.filters.truncate(8);
        sp.sb_rows.truncate(MAX_SB_ROWS);
    }
}

/// Frames that may be decoding at once (parsing, reconstruction or
/// filters), at most. Each keeps up to two pictures, its mode info twice,
/// its block logs and the bands of rows it publishes (on 18 threads and
/// 4K, eight to twelve takes one clip from 0.46 s to 0.41 s for 0.3 GB
/// more). See also [`IN_FLIGHT_WEIGHT`].
const MAX_IN_FLIGHT: usize = 16;

/// The most the weights of the frames decoding at once come to: a frame
/// whose filters run on a thread of their own (see [`banded`]) weighs 4,
/// another 3. So at most twelve of the first and sixteen of the others:
/// past those the first mostly wait for the rows of their references,
/// and more of them make a 4K clip with CDEF slower on 18 threads (0.40 s
/// rather than 0.38 s at sixteen), while sixteen of the others take a
/// deblock-only one from 0.34 s to 0.32 s.
const IN_FLIGHT_WEIGHT: usize = 48;

/// The decoder state that persists between temporal units.
pub struct Av1Core {
    seq: Option<SequenceHeader>,
    ref_hdr: [RefHeaderState; 8],
    slots: [Option<Slot>; 8],
    /// Frame in progress (header seen, tiles pending).
    pending: Option<Pending>,
    px8: PixState<u8>,
    px16: PixState<u16>,
    /// Apply film grain to shown pictures.
    pub apply_grain: bool,
    /// Pictures handed out (shown or still referenced) that return to the
    /// pools once nothing else holds them.
    outstanding: Vec<Arc<FrameJob>>,
    /// Shown frames not returned yet, in output order.
    queue: VecDeque<Queued>,
    /// Frames that may still be decoding.
    /// With their weights (see [`Self::limit_in_flight`]).
    in_flight: Vec<(Arc<FrameJob>, usize)>,
    /// Told by each of them when it ends.
    ended: Arc<Ended>,
    /// The weight of the frame being started.
    weight: usize,
    /// Frames started by the current call.
    unit_jobs: Vec<Arc<FrameJob>>,
    /// Largest picture (upscaled width x height) accepted; larger is `Limit`.
    /// The default is the largest picture any AV1 level allows.
    pub max_pixels: u64,
    /// Threads decoding may use; 1 decodes on the calling thread. Defaults
    /// to `std::thread::available_parallelism()`. With 2 or more, the
    /// calling thread only reads the frame headers: each frame is parsed
    /// on a thread of its own as soon as the frames it depends on are
    /// parsed far enough, and reconstructed and filtered on another, so
    /// with [`Self::decode_delayed`] up to as many frames as threads
    /// decode at once.
    pub threads: usize,
    /// Whether frames decode beside each other when there are threads
    /// for it (the default). Off, a frame waits for its references and
    /// only its tiles decode at once: slower, with fewer frames and
    /// their buffers alive at a time.
    pub frames_at_once: bool,
    /// Shown frames [`Self::decode_delayed`] may keep back (it returns
    /// earlier ones as soon as they are done); `None` (the default) keeps
    /// back as many as there are [`Self::threads`].
    pub frame_delay: Option<usize>,
    /// Operating point to decode (`choose_operating_point()`); 0 by default.
    /// Set it with [`Self::select_operating_point`].
    operating_point: usize,
    workers: Arc<Workers>,
}

impl Drop for Av1Core {
    fn drop(&mut self) {
        self.workers.close();
    }
}

struct Pending {
    hdr: FrameHeader,
    tiles: Vec<(usize, Vec<u8>)>,
}

impl Default for Av1Core {
    fn default() -> Self {
        Self::new()
    }
}

impl Av1Core {
    pub fn new() -> Self {
        Self {
            seq: None,
            ref_hdr: Default::default(),
            slots: Default::default(),
            pending: None,
            px8: PixState::default(),
            px16: PixState::default(),
            apply_grain: true,
            outstanding: Vec::new(),
            queue: VecDeque::new(),
            in_flight: Vec::new(),
            ended: Arc::default(),
            weight: 0,
            unit_jobs: Vec::new(),
            max_pixels: 35_651_584,
            threads: std::thread::available_parallelism().map_or(1, usize::from),
            frames_at_once: true,
            frame_delay: None,
            operating_point: 0,
            workers: Arc::default(),
        }
    }

    /// Selects the operating point to decode (7.2.1); the default is 0. OBUs
    /// of layers outside it are dropped. An index past the last point of the
    /// current sequence, or of a later sequence header, is refused with
    /// `Unsupported`.
    pub fn select_operating_point(&mut self, index: usize) -> R<()> {
        if index > 31 {
            return Err(CodecError::Unsupported {
                what: "operating point",
            });
        }
        if let Some(seq) = &mut self.seq {
            let mut chosen = seq.clone();
            chosen.choose_operating_point(index)?;
            *seq = chosen;
        }
        self.operating_point = index;
        Ok(())
    }

    /// Drops every reference and every frame not returned yet (frames
    /// still decoding finish on their threads and are discarded).
    pub fn reset(&mut self) {
        let seq = self.seq.take();
        let (grain, max, threads, at_once, delay, op) = (
            self.apply_grain,
            self.max_pixels,
            self.threads,
            self.frames_at_once,
            self.frame_delay,
            self.operating_point,
        );
        *self = Self::new();
        self.seq = seq;
        self.apply_grain = grain;
        self.max_pixels = max;
        self.threads = threads;
        self.frames_at_once = at_once;
        self.frame_delay = delay;
        self.operating_point = op;
    }

    /// The header state of every reference slot.
    pub fn ref_headers(&self) -> &[RefHeaderState; 8] {
        &self.ref_hdr
    }

    /// The picture held by a reference slot (waiting for it to be decoded).
    pub fn slot_picture(&self, slot: usize) -> Option<Arc<AnyPicture>> {
        self.slots.get(slot)?.as_ref()?.pic.wait().ok()
    }

    /// The CDFs saved with a reference slot.
    pub fn slot_cdf(&self, slot: usize) -> Option<Arc<CdfContext>> {
        self.slots.get(slot)?.as_ref()?.cdf.wait().ok()
    }

    /// Motion field and segmentation data saved with a reference slot.
    pub fn slot_data(&self, slot: usize) -> Option<Arc<RefData>> {
        self.slots.get(slot)?.as_ref()?.data.wait().ok()
    }

    pub fn sequence(&self) -> Option<&SequenceHeader> {
        self.seq.as_ref()
    }

    /// Decodes one Annex B (length-delimited) temporal unit.
    pub fn decode_annexb(&mut self, data: &[u8]) -> R<Vec<DecodedFrame>> {
        let low = super::obu::annexb_to_low_overhead(data)?;
        self.decode(&low)
    }

    /// Decodes one temporal unit and returns the frames it shows, once they
    /// are fully decoded. A unit that fails leaves the reference state as
    /// it was before the call.
    pub fn decode(&mut self, data: &[u8]) -> R<Vec<DecodedFrame>> {
        let r = self.decode_checked(data, 0, true);
        let r = r.and_then(|()| self.take_outputs(0));
        r.map(|v| v.into_iter().map(|(_, f)| f).collect())
    }

    /// Decodes one temporal unit and returns the shown frames that are
    /// ready, oldest first, each with the `tag` of the call that decoded
    /// it. With 2 or more [`Self::threads`] the call only reads the frame
    /// headers: the frames are parsed, reconstructed and filtered on other
    /// threads, beside the next units; at most [`Self::frame_delay`] shown
    /// frames are kept back. [`Self::drain`] returns the rest.
    ///
    /// A unit whose headers fail leaves the reference state as it was
    /// before the call. A failure found later (in the tile data, when the
    /// frame is decoded on other threads) is returned in place of the
    /// frame's picture, and the frames that depend on it fail too.
    pub fn decode_delayed(&mut self, data: &[u8], tag: u64) -> R<Vec<(u64, DecodedFrame)>> {
        self.decode_checked(data, tag, false)?;
        let delay = self.frame_delay.unwrap_or(self.threads.max(1));
        self.take_outputs(delay)
    }

    /// Waits for every frame [`Self::decode_delayed`] kept back and returns
    /// them in order.
    pub fn drain(&mut self) -> R<Vec<(u64, DecodedFrame)>> {
        self.take_outputs(0)
    }

    /// Decodes a unit, restoring the reference state when it fails (when
    /// `wait`, also after waiting for the frames it started).
    fn decode_checked(&mut self, data: &[u8], tag: u64, wait: bool) -> R<()> {
        let saved = (self.seq.clone(), self.ref_hdr.clone(), self.slots.clone());
        let q0 = self.queue.len();
        self.unit_jobs.clear();
        let mut r = self.decode_unit(data, tag);
        if r.is_ok() && wait {
            for j in &self.unit_jobs {
                if let Err(e) = j.wait() {
                    r = Err(e);
                    break;
                }
            }
        }
        self.unit_jobs.clear();
        if r.is_err() {
            self.seq = saved.0;
            self.ref_hdr = saved.1;
            self.slots = saved.2;
            self.pending = None;
            self.queue.truncate(q0);
        }
        r
    }

    /// Returns the queued frames in order: all that are done, waiting for
    /// those past the first `keep`.
    fn take_outputs(&mut self, keep: usize) -> R<Vec<(u64, DecodedFrame)>> {
        let mut out = Vec::new();
        while let Some(q) = self.queue.front() {
            let r = if self.queue.len() > keep {
                Some(q.job.wait())
            } else {
                q.job.poll()
            };
            match r {
                None => break,
                Some(Ok(pic)) => {
                    let f = self.output(&q.seq, &pic, &q.rh, &q.grain);
                    out.push((q.tag, f));
                    self.queue.pop_front();
                }
                Some(Err(e)) => {
                    // Frames before the failed one go out first.
                    if out.is_empty() {
                        self.queue.pop_front();
                        return Err(e);
                    }
                    break;
                }
            }
        }
        Ok(out)
    }

    fn decode_unit(&mut self, data: &[u8], tag: u64) -> R<()> {
        for obu in split_obus(data)? {
            // 7.5: an OBU of a layer outside the operating point is dropped.
            if obu.obu_type != OBU_SEQUENCE_HEADER
                && obu.obu_type != OBU_TEMPORAL_DELIMITER
                && obu.has_extension
                && let Some(seq) = &self.seq
                && seq.operating_point_idc != 0
            {
                let idc = seq.operating_point_idc;
                let in_t = (idc >> obu.temporal_id) & 1;
                let in_s = (idc >> (obu.spatial_id + 8)) & 1;
                if in_t == 0 || in_s == 0 {
                    continue;
                }
            }
            match obu.obu_type {
                OBU_SEQUENCE_HEADER => {
                    let mut s = parse_sequence_header(obu.payload)?;
                    if s.max_frame_width > 16384 || s.max_frame_height > 16384 {
                        return Err(CodecError::Limit { what: "frame size" });
                    }
                    s.choose_operating_point(self.operating_point)?;
                    self.seq = Some(s);
                }
                OBU_FRAME_HEADER | OBU_REDUNDANT_FRAME_HEADER | OBU_FRAME => {
                    if obu.obu_type == OBU_REDUNDANT_FRAME_HEADER && self.pending.is_some() {
                        continue;
                    }
                    if obu.obu_type == OBU_FRAME_HEADER && self.pending.is_some() {
                        // frame_header_copy
                        continue;
                    }
                    let Some(seq) = self.seq.clone() else {
                        return Err(malformed("frame before sequence header"));
                    };
                    let ctx = HeaderContext {
                        seq: &seq,
                        refs: &self.ref_hdr,
                        temporal_id: obu.temporal_id,
                        spatial_id: obu.spatial_id,
                    };
                    let hdr = parse_frame_header(obu.payload, &ctx)?;
                    #[cfg(feature = "stats")]
                    if !hdr.show_existing_frame {
                        super::stats::frame(format!(
                            "{}{}:q{}:{}b:cdef{}:lr{}{}{}:dq{},{},{},{},{}:qm{}",
                            hdr.order_hint,
                            if hdr.show_frame { "" } else { "h" },
                            hdr.quant.base_q_idx,
                            obu.payload.len(),
                            hdr.cdef.bits,
                            hdr.lr.frame_type[0],
                            hdr.lr.frame_type[1],
                            hdr.lr.frame_type[2],
                            hdr.quant.delta_q_y_dc,
                            hdr.quant.delta_q_u_dc,
                            hdr.quant.delta_q_u_ac,
                            hdr.quant.delta_q_v_dc,
                            hdr.quant.delta_q_v_ac,
                            u8::from(hdr.quant.using_qmatrix)
                        ));
                    }
                    if hdr.show_existing_frame {
                        self.show_existing(&seq, &hdr, tag)?;
                        continue;
                    }
                    if u64::from(hdr.upscaled_width) * u64::from(hdr.frame_height) > self.max_pixels
                    {
                        return Err(CodecError::Limit { what: "frame size" });
                    }
                    let mut p = Pending {
                        hdr,
                        tiles: Vec::new(),
                    };
                    if obu.obu_type == OBU_FRAME {
                        let rest = &obu.payload[p.hdr.header_bytes..];
                        self.add_tile_group(&mut p, rest)?;
                    }
                    let done = p.tiles.len() == p.hdr.tile_info.cols * p.hdr.tile_info.rows;
                    if done {
                        self.finish_frame(&seq, p, tag)?;
                    } else {
                        self.pending = Some(p);
                    }
                }
                OBU_TILE_GROUP => {
                    let Some(mut p) = self.pending.take() else {
                        return Err(malformed("tile group without frame header"));
                    };
                    let Some(seq) = self.seq.clone() else {
                        return Err(malformed("frame before sequence header"));
                    };
                    self.add_tile_group(&mut p, obu.payload)?;
                    if p.tiles.len() == p.hdr.tile_info.cols * p.hdr.tile_info.rows {
                        self.finish_frame(&seq, p, tag)?;
                    } else {
                        self.pending = Some(p);
                    }
                }
                OBU_TILE_LIST => {
                    return Err(CodecError::Unsupported {
                        what: "large scale tile list",
                    });
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn add_tile_group(&mut self, p: &mut Pending, data: &[u8]) -> R<()> {
        let ti = &p.hdr.tile_info;
        let num_tiles = ti.cols * ti.rows;
        let mut r = super::bits::BitReader::new(data);
        let mut start_end_present = false;
        if num_tiles > 1 {
            start_end_present = r.flag()?;
        }
        let (tg_start, tg_end) = if num_tiles == 1 || !start_end_present {
            (0, num_tiles - 1)
        } else {
            let bits = ti.cols_log2 + ti.rows_log2;
            (r.f(bits)? as usize, r.f(bits)? as usize)
        };
        r.byte_alignment();
        let mut pos = r.position() / 8;
        if tg_end >= num_tiles || tg_start > tg_end || tg_start != p.tiles.len() {
            return Err(malformed("tile group range"));
        }
        for tile in tg_start..=tg_end {
            let size = if tile == tg_end {
                data.len() - pos
            } else {
                let n = ti.tile_size_bytes as usize;
                if pos + n > data.len() {
                    return Err(CodecError::Truncated {
                        at_bit: (pos * 8) as u64,
                        what: "tile size",
                    });
                }
                let mut v = 0usize;
                for i in 0..n {
                    v |= usize::from(data[pos + i]) << (8 * i);
                }
                pos += n;
                v + 1
            };
            if pos + size > data.len() {
                return Err(CodecError::Truncated {
                    at_bit: (pos * 8) as u64,
                    what: "tile data",
                });
            }
            p.tiles.push((tile, data[pos..pos + size].to_vec()));
            pos += size;
        }
        Ok(())
    }

    fn show_existing(&mut self, seq: &SequenceHeader, hdr: &FrameHeader, tag: u64) -> R<()> {
        let idx = hdr.frame_to_show_map_idx as usize;
        let Some(slot) = self.slots[idx].clone() else {
            return Err(malformed("show_existing_frame of empty slot"));
        };
        let rh = self.ref_hdr[idx].clone();
        let mut grain = rh.grain.clone();
        grain.grain_seed = hdr.grain.grain_seed;
        if seq.film_grain_params_present {
            grain = hdr.grain.clone();
        }
        self.queue.push_back(Queued {
            job: slot.pic.clone(),
            seq: Arc::new(seq.clone()),
            rh: rh.clone(),
            grain,
            tag,
        });
        if rh.frame_type == KEY_FRAME {
            // Reference frame loading process (7.21) then update all slots.
            let mut new_hdr = rh.clone();
            new_hdr.showable = false;
            for i in 0..8 {
                self.ref_hdr[i] = new_hdr.clone();
                self.slots[i] = Some(slot.clone());
            }
        }
        Ok(())
    }

    fn output(
        &self,
        seq: &SequenceHeader,
        pic: &Arc<AnyPicture>,
        rh: &RefHeaderState,
        grain: &GrainParams,
    ) -> DecodedFrame {
        let w = rh.upscaled_width as usize;
        let h = rh.frame_height as usize;
        let mono = seq.mono_chrome;
        let picture = if self.apply_grain && grain.apply_grain {
            let p = to_fg_params(grain);
            Arc::new(match &**pic {
                AnyPicture::U8(src) => {
                    AnyPicture::U8(grain_apply(src, &p, w, h, mono, seq.matrix_coefficients))
                }
                AnyPicture::U16(src) => {
                    AnyPicture::U16(grain_apply(src, &p, w, h, mono, seq.matrix_coefficients))
                }
            })
        } else {
            Arc::clone(pic)
        };
        DecodedFrame {
            picture,
            width: w,
            height: h,
            render_width: rh.render_width as usize,
            render_height: rh.render_height as usize,
            bit_depth: rh.bit_depth,
            mono,
            sub_x: rh.subsampling_x,
            sub_y: rh.subsampling_y,
            color_primaries: seq.color_primaries,
            transfer: seq.transfer_characteristics,
            matrix: seq.matrix_coefficients,
            full_range: seq.color_range,
            chroma_sample_position: seq.chroma_sample_position,
            num_units_in_display_tick: seq.num_units_in_display_tick,
            time_scale: seq.time_scale,
            equal_picture_interval: seq.equal_picture_interval,
            num_ticks_per_picture_minus_1: seq
                .equal_picture_interval
                .then_some(seq.num_ticks_per_picture_minus_1),
            num_units_in_decoding_tick: seq.num_units_in_decoding_tick,
            seq_level_idx: seq.seq_level_idx(),
        }
    }

    fn finish_frame(&mut self, seq: &SequenceHeader, p: Pending, tag: u64) -> R<()> {
        let hdr = p.hdr;
        // The CDFs the frame starts from.
        let primary_slot = (hdr.primary_ref_frame != PRIMARY_REF_NONE)
            .then(|| hdr.ref_frame_idx[hdr.primary_ref_frame as usize] as usize);
        let primary_cdf = match primary_slot {
            None => None,
            Some(s) => match &self.slots[s] {
                Some(slot) => Some(slot.cdf.clone()),
                None => return Err(malformed("primary reference frame")),
            },
        };
        if !hdr.frame_is_intra {
            // Every reference must share the sequence format and obey the
            // scaling limits of 7.11.3.3.
            for i in 0..7 {
                let s = hdr.ref_frame_idx[i] as usize;
                let rh = &self.ref_hdr[s];
                let fw = hdr.frame_width;
                let fh = hdr.frame_height;
                if self.slots[s].is_none()
                    || rh.bit_depth != seq.bit_depth
                    || rh.subsampling_x != seq.subsampling_x
                    || rh.subsampling_y != seq.subsampling_y
                    || 2 * fw < rh.upscaled_width
                    || 2 * fh < rh.frame_height
                    || fw > 16 * rh.upscaled_width
                    || fh > 16 * rh.frame_height
                {
                    return Err(malformed("reference frame format"));
                }
            }
        }
        self.limit_in_flight(if banded(seq, &hdr) { 4 } else { 3 });
        self.reclaim_pictures();
        // The parse waits for the motion fields of the references it
        // projects and the segment ids of the primary reference.
        let mut deps: [Option<DataPromise>; 8] = Default::default();
        let mut need = |s: usize| {
            deps[s] = self.slots[s].as_ref().map(|sl| sl.data.clone());
        };
        if hdr.use_ref_frame_mvs {
            for &s in &hdr.ref_frame_idx[..7] {
                need(s as usize);
            }
        }
        if let Some(s) = primary_slot {
            need(s);
        }
        // The pictures of the references (by reference frame 1..=7), which
        // may still be decoding.
        let ref_jobs: [Option<Arc<FrameJob>>; 8] = std::array::from_fn(|i| {
            if i == 0 || hdr.frame_is_intra {
                return None;
            }
            let slot = hdr.ref_frame_idx[i - 1] as usize;
            self.slots[slot].as_ref().map(|s| s.pic.clone())
        });
        let job = FrameJob::running();
        lock(&job.state).ended = Some(self.ended.clone());
        let cdf_out = Promise::new();
        let data_out = Promise::new();
        let t = TaskParts {
            seq: seq.clone(),
            hdr: hdr.clone(),
            tiles: p.tiles,
            primary_cdf,
            deps,
            ref_jobs,
            ref_hdr: self.ref_hdr.clone(),
            threads: self.threads,
            at_once: self.frames_at_once,
            workers: self.workers.clone(),
            job: job.clone(),
            cdf_out: cdf_out.clone(),
            data_out: data_out.clone(),
        };
        let inline = self.threads <= 1;
        if seq.bit_depth == 8 {
            launch(&mut self.px8, t, inline)?;
        } else {
            launch(&mut self.px16, t, inline)?;
        }
        self.unit_jobs.push(job.clone());
        if !job.is_done() {
            self.in_flight.push((job.clone(), self.weight));
        }
        self.store_frame(seq, &hdr, job, cdf_out, data_out, tag);
        Ok(())
    }

    /// Takes in a frame reconstructed outside the decoder, as decoding its
    /// tiles would have left it: an encoder's own reconstruction of a
    /// frame it coded, with `hdr` the frame's header (parsed against this
    /// decoder's state), `fs` the frame state its tiles left, `pic` the
    /// final picture (after every in-loop filter) and `cdf` the CDFs to
    /// save with it (those of the context tile, or the frame's initial
    /// ones when the frame end update is disabled). Returns the frame if
    /// it is shown.
    pub fn install_frame(
        &mut self,
        hdr: &FrameHeader,
        fs: &FrameState,
        pic: AnyPicture,
        cdf: Box<CdfContext>,
    ) -> R<Vec<DecodedFrame>> {
        let Some(seq) = self.seq.clone() else {
            return Err(malformed("frame before sequence header"));
        };
        if hdr.show_existing_frame || fs.mi_rows != hdr.mi_rows as usize {
            return Err(CodecError::InvalidApi {
                what: "installed frame",
            });
        }
        let job = FrameJob::ready(Ok(Arc::new(pic)));
        let mut cdf = cdf;
        cdf.clear_counts();
        let data = Promise::ready(Ok(Arc::new(ref_data(&seq, hdr, fs))));
        let cdf = Promise::ready(Ok(Arc::from(cdf)));
        self.store_frame(&seq, hdr, job, cdf, data, 0);
        let out = self.take_outputs(0)?;
        Ok(out.into_iter().map(|(_, f)| f).collect())
    }

    /// Queues a frame for output if shown and runs the reference frame
    /// update process (7.20) with its picture `job`, CDFs `cdf` and
    /// motion field and segmentation data `data`.
    fn store_frame(
        &mut self,
        seq: &SequenceHeader,
        hdr: &FrameHeader,
        job: Arc<FrameJob>,
        cdf: CdfPromise,
        data: DataPromise,
        tag: u64,
    ) {
        let rh = RefHeaderState {
            valid: true,
            frame_id: hdr.current_frame_id,
            upscaled_width: hdr.upscaled_width,
            frame_width: hdr.frame_width,
            frame_height: hdr.frame_height,
            render_width: hdr.render_width,
            render_height: hdr.render_height,
            mi_cols: hdr.mi_cols,
            mi_rows: hdr.mi_rows,
            frame_type: hdr.frame_type,
            order_hint: hdr.order_hint,
            saved_order_hints: hdr.order_hints,
            gm_params: hdr.gm_params,
            loop_filter_ref_deltas: hdr.loop_filter.ref_deltas,
            loop_filter_mode_deltas: hdr.loop_filter.mode_deltas,
            feature_enabled: hdr.segmentation.feature_enabled,
            feature_data: hdr.segmentation.feature_data,
            grain: hdr.grain.clone(),
            bit_depth: seq.bit_depth,
            subsampling_x: seq.subsampling_x,
            subsampling_y: seq.subsampling_y,
            showable: hdr.showable_frame,
        };
        if hdr.show_frame {
            self.queue.push_back(Queued {
                job: job.clone(),
                seq: Arc::new(seq.clone()),
                rh: rh.clone(),
                grain: hdr.grain.clone(),
                tag,
            });
        }
        if hdr.refresh_frame_flags != 0 {
            let slot = Slot {
                pic: job,
                data,
                cdf,
            };
            let mut old = Vec::new();
            for i in 0..8 {
                if (hdr.refresh_frame_flags >> i) & 1 != 0 {
                    self.ref_hdr[i] = rh.clone();
                    if let Some(o) = self.slots[i].replace(slot.clone()) {
                        old.push(o.pic);
                    }
                }
            }
            drop(slot);
            for p in old {
                self.release_picture(p);
            }
        } else {
            self.release_picture(job);
        }
    }

    /// Recycles the picture of `j` if nothing else holds it, else remembers
    /// it until the holders (the caller's frames, other slots, frames still
    /// decoding) let go.
    fn release_picture(&mut self, j: Arc<FrameJob>) {
        let keep = |this: &mut Self, j: Arc<FrameJob>| {
            if !this.outstanding.iter().any(|o| Arc::ptr_eq(o, &j)) {
                if this.outstanding.len() >= 16 {
                    this.outstanding.remove(0);
                }
                this.outstanding.push(j);
            }
        };
        if !j.is_done() {
            return keep(self, j);
        }
        let p = match Arc::try_unwrap(j) {
            Ok(job) => match job.into_result() {
                Some(Ok(p)) => p,
                _ => return,
            },
            Err(j) => return keep(self, j),
        };
        match Arc::try_unwrap(p) {
            Ok(AnyPicture::U8(p)) => recycle(&mut lock(&self.px8.spares).pics, p),
            Ok(AnyPicture::U16(p)) => recycle(&mut lock(&self.px16.spares).pics, p),
            Err(p) => keep(self, FrameJob::ready(Ok(p))),
        }
    }

    /// Moves outstanding pictures nobody else holds back to the pools, and
    /// the spares of finished frames.
    fn reclaim_pictures(&mut self) {
        let mut i = 0;
        while i < self.outstanding.len() {
            let o = &self.outstanding[i];
            if Arc::strong_count(o) == 1 && o.pic_unshared() {
                let p = self.outstanding.swap_remove(i);
                self.release_picture(p);
            } else {
                i += 1;
            }
        }
        self.px8.collect();
        self.px16.collect();
    }

    /// Waits until a frame of weight `w` (see [`IN_FLIGHT_WEIGHT`]) may
    /// start: until fewer frames are decoding than the threads allow (as
    /// many as the threads, at least 2, at most [`MAX_IN_FLIGHT`]), and
    /// the weights of those decoding come to at most
    /// `IN_FLIGHT_WEIGHT - w`.
    fn limit_in_flight(&mut self, w: usize) {
        let full = |f: &[(Arc<FrameJob>, usize)]| {
            f.len() >= self.threads.clamp(2, MAX_IN_FLIGHT)
                || f.iter().map(|&(_, w)| w).sum::<usize>() + w > IN_FLIGHT_WEIGHT
        };
        // Whichever frame ends first makes room, not only the oldest: a
        // later one often ends before it.
        loop {
            // Read before looking at the jobs, so that one ending after
            // the look moves the count past it (a job counts once its
            // result is in).
            let seen = *lock(&self.ended.n);
            self.in_flight.retain(|(j, _)| !j.is_done());
            if !full(&self.in_flight) {
                break;
            }
            let t = std::time::Instant::now();
            let mut n = lock(&self.ended.n);
            while *n == seen {
                n = self
                    .ended
                    .cv
                    .wait(n)
                    .unwrap_or_else(PoisonError::into_inner);
            }
            prof::add(prof::WAIT_OUT, t);
        }
        self.weight = w;
    }
}

/// Whether CDEF filters the frame of `hdr`.
fn cdef_on(seq: &SequenceHeader, hdr: &FrameHeader) -> bool {
    seq.enable_cdef
        && !hdr.coded_lossless
        && !hdr.allow_intrabc
        && (hdr.cdef.bits > 0
            || hdr.cdef.y_pri[0] != 0
            || hdr.cdef.y_sec[0] != 0
            || hdr.cdef.uv_pri[0] != 0
            || hdr.cdef.uv_sec[0] != 0)
}

/// Whether the threaded pipeline reconstructs the frame of `hdr` a
/// superblock row at a time and runs its filters on a thread of their own
/// (see [`decode_frame_async`]).
fn banded(seq: &SequenceHeader, hdr: &FrameHeader) -> bool {
    hdr.refresh_frame_flags != 0
        && (cdef_on(seq, hdr) || hdr.lr.uses_lr)
        && !hdr.allow_intrabc
        && !hdr.use_superres
}

/// What the parse of a frame needs besides its picture and the pools.
struct TaskParts {
    seq: SequenceHeader,
    hdr: FrameHeader,
    tiles: Vec<(usize, Vec<u8>)>,
    /// The CDFs of the primary reference frame.
    primary_cdf: Option<CdfPromise>,
    /// By slot: the data of the references the parse reads.
    deps: [Option<DataPromise>; 8],
    ref_jobs: [Option<Arc<FrameJob>>; 8],
    ref_hdr: [RefHeaderState; 8],
    threads: usize,
    /// [`Av1Core::frames_at_once`].
    at_once: bool,
    workers: Arc<Workers>,
    /// The final picture, the CDFs saved for later frames and the motion
    /// field and segmentation data.
    job: Arc<FrameJob>,
    cdf_out: CdfPromise,
    data_out: DataPromise,
}

/// The decoding of one frame whose header is read: it waits for what its
/// parse depends on (the CDFs of the primary reference frame, the motion
/// fields and segment ids of its references, each available once that
/// frame's parse ends), parses the tiles and hands reconstruction and the
/// in-loop filters to a [`Backend`] (see [`decode_frame`]). With 2 or more
/// threads each task runs on a thread of its own, so frames that do not
/// depend on each other's parse (such as those never referenced) parse at
/// once.
struct FrameTask<P> {
    t: TaskParts,
    cur: Picture<P>,
    rows: Arc<Mutex<RowPools>>,
    spares: Arc<Mutex<Spares<P>>>,
}

/// Fails what a task did not deliver when it ends without a result (its
/// thread panicked).
struct TaskGuard {
    open: Option<(Arc<FrameJob>, CdfPromise, DataPromise)>,
    /// The frame's data for later frames, which they may already read.
    data: Option<Arc<RefData>>,
}

impl TaskGuard {
    fn fail(&mut self, e: CodecError) {
        if let Some((job, cdf, data)) = self.open.take() {
            job.finish(Err(e));
            cdf.set(Err(e));
            data.set(Err(e));
        }
    }
}

impl Drop for TaskGuard {
    fn drop(&mut self) {
        self.fail(malformed("decoding thread"));
        // Whatever happened, no reader waits for motion that never comes.
        if let Some(d) = &self.data {
            d.close();
        }
    }
}

/// Starts the task of a frame: on the calling thread when `inline` (and
/// returns its result), else on a thread of its own.
fn launch<P: PicturePixel>(px: &mut PixState<P>, t: TaskParts, inline: bool) -> R<()> {
    let cur = take_picture::<P>(
        &mut lock(&px.spares).pics,
        t.hdr.frame_width as usize,
        t.hdr.frame_height as usize,
        t.seq.num_planes(),
        t.seq.subsampling_x,
        t.seq.subsampling_y,
        t.seq.bit_depth,
    );
    let task = FrameTask {
        t,
        cur,
        rows: px.rows.clone(),
        spares: px.spares.clone(),
    };
    if inline {
        task.run()
    } else {
        let unref = task.t.hdr.refresh_frame_flags == 0;
        let workers = task.t.workers.clone();
        workers.run(Box::new(move || {
            let t = prof::thread_cpu();
            let _ = task.run();
            prof::cpu_since(
                if unref {
                    prof::CPU_TASK_UNREF
                } else {
                    prof::CPU_TASK
                },
                t,
            );
        }));
        Ok(())
    }
}

impl<P: PicturePixel> FrameTask<P> {
    fn run(self) -> R<()> {
        // Later frames get this frame's data now and wait for the rows
        // they read, not for the end of its parse.
        let hdr = &self.t.hdr;
        let data = (hdr.refresh_frame_flags != 0).then(|| {
            Arc::new(RefData::new(
                hdr.mi_cols as usize,
                hdr.mi_rows as usize,
                hdr.frame_type,
                hdr.order_hints,
            ))
        });
        self.t.data_out.set(match &data {
            Some(d) => Ok(d.clone()),
            None => Err(malformed("frame not referenced")),
        });
        let mut guard = TaskGuard {
            open: Some((
                self.t.job.clone(),
                self.t.cdf_out.clone(),
                self.t.data_out.clone(),
            )),
            data: data.clone(),
        };
        let r = self.parse(data.as_deref());
        match r {
            Err(e) => guard.fail(e),
            // The reconstruction thread finishes the job.
            Ok(()) => guard.open = None,
        }
        r
    }

    fn parse(self, data: Option<&RefData>) -> R<()> {
        let Self {
            t,
            cur,
            rows,
            spares,
        } = self;
        let (seq, hdr) = (&t.seq, &t.hdr);
        let mut refs_data: [Option<Arc<RefData>>; 8] = Default::default();
        for (d, p) in refs_data.iter_mut().zip(&t.deps) {
            if let Some(p) = p {
                *d = Some(p.wait()?);
            }
        }
        let spare = {
            let mut sp = lock(&spares);
            let i = sp.states.iter().position(|f| {
                f.mi_cols == hdr.mi_cols as usize && f.mi_rows == hdr.mi_rows as usize
            });
            match i {
                Some(i) => Some(sp.states.swap_remove(i)),
                None => sp.states.pop(),
            }
        };
        if spare.is_none() {
            prof::count(prof::NEW_STATE);
        }
        let mut fs = FrameState::reuse(spare, seq, hdr);
        let mf_plan = hdr
            .use_ref_frame_mvs
            .then(|| Arc::new(motion_field_plan(seq, hdr, &refs_data)));
        // A pipelined frame's motion field is computed band by band on a
        // thread of its own while the tiles are parsed (see MfFeed), from
        // now on: it reads only the references, and the parse, which
        // first waits for the CDFs of the frame before, reads it from
        // its first block (decode_frame computes the others').
        let field = mf_plan
            .as_ref()
            .filter(|_| pipelined(hdr, t.threads, t.at_once))
            .and_then(|plan| start_motion_field(&t.workers, plan));
        let mut cdf: Box<CdfContext> = match &t.primary_cdf {
            None => CdfContext::new(hdr.quant.base_q_idx),
            Some(p) => {
                let tw = std::time::Instant::now();
                let w = p.wait();
                prof::mark(prof::CDF_WAIT, tw);
                let mut c = Box::new((*w?).clone());
                c.clear_counts();
                c
            }
        };
        if hdr.disable_frame_end_update_cdf {
            t.cdf_out.set(Ok(Arc::from(cdf.clone())));
        }
        if hdr.primary_ref_frame != PRIMARY_REF_NONE {
            let slot_idx = hdr.ref_frame_idx[hdr.primary_ref_frame as usize] as usize;
            if let Some(rd) = &refs_data[slot_idx]
                && rd.mi_rows == fs.mi_rows
                && rd.mi_cols == fs.mi_cols
            {
                fs.prev_segment_ids = Some(rd.clone());
            }
        }
        let saved = decode_frame::<P>(
            seq,
            hdr,
            &mut fs,
            &t.tiles,
            &mut cdf,
            &t.ref_jobs,
            &t.ref_hdr,
            cur,
            &t.job,
            &rows,
            &spares,
            &t.workers,
            t.threads,
            t.at_once,
            mf_plan.as_ref(),
            field,
            &FrameOut {
                data,
                cdf: (!hdr.disable_frame_end_update_cdf).then_some(&t.cdf_out),
            },
        )?;
        drop(mf_plan);
        if !hdr.disable_frame_end_update_cdf {
            let mut saved = saved;
            saved.clear_counts();
            t.cdf_out.set(Ok(Arc::from(saved)));
        }
        if let Some(rd) = data {
            // The bands the parse did not save on its way.
            save_motion(seq, hdr, &fs, rd, 0, fs.mi_rows);
            rd.set_segment_ids(segment_map(hdr, &fs));
        }
        lock(&spares).states.push(fs);
        Ok(())
    }
}

/// MfRefFrames and MfMvs (7.19) of band `k` of the frame of `fs`, whose
/// rows are parsed, in the columns `cols` of 8x8 positions.
fn mf_band(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &FrameState,
    k: usize,
    cols: Range<usize>,
) -> MfBand {
    let w8 = cols.len();
    let h8 = fs.mi_rows >> 1;
    let rows = h8.saturating_sub(k * MF_BAND_ROWS).min(MF_BAND_ROWS);
    let mut band = MfBand {
        refs: vec![-1i8; w8 * rows],
        mvs: vec![Mv::ZERO; w8 * rows],
    };
    if hdr.frame_is_intra {
        return band;
    }
    // Whether each reference frame lies before this one.
    let past: [bool; 8] = std::array::from_fn(|r| {
        r > INTRA_FRAME as usize && {
            let ref_idx = hdr.ref_frame_idx[r - 1] as usize;
            relative_dist(seq, hdr.ref_order_hint[ref_idx], hdr.order_hint) < 0
        }
    });
    let ok = |r: i8, mv: Mv| {
        r > INTRA_FRAME
            && past[r as usize & 7]
            && i32::from(mv.row).abs() < (1 << 12)
            && i32::from(mv.col).abs() < (1 << 12)
    };
    // Block by block: a block is as far from the left as a whole number
    // of its widths, and its 8x8 positions save the same. The next block
    // of a row is known only once its block size is read, so the rows go
    // round, a block of each at a time: their reads wait together.
    let mut mis: [&[MiInfo]; MF_BAND_ROWS] = [&[]; MF_BAND_ROWS];
    let mut x8s = [cols.end; MF_BAND_ROWS];
    for (y, (m, x8)) in mis.iter_mut().zip(&mut x8s).enumerate().take(rows) {
        let at = fs.mi_row(2 * (k * MF_BAND_ROWS + y) + 1);
        if let Some(mi) = fs.mi.get(at..at + 2 * cols.end) {
            (*m, *x8) = (mi, cols.start);
        }
    }
    let (refs, mvs) = (&mut band.refs, &mut band.mvs);
    let mut busy = true;
    while busy {
        busy = false;
        for (y, (mi, x8)) in mis.iter().zip(&mut x8s).enumerate() {
            if *x8 >= cols.end {
                continue;
            }
            busy = true;
            let c4 = 2 * *x8 + 1;
            let m = &mi[c4];
            let bw = bw4(m.bsize).max(1);
            let end = (((c4 & !(bw - 1)) + bw) / 2).clamp(*x8 + 1, cols.end);
            // The second list wins when both qualify.
            if let Some(list) = [1, 0].into_iter().find(|&l| ok(m.ref_frame[l], m.mv[l])) {
                let run = y * w8 + *x8 - cols.start..y * w8 + end - cols.start;
                fill_run(&mut refs[run.clone()], m.ref_frame[list]);
                fill_run(&mut mvs[run], m.mv[list]);
            }
            *x8 = end;
        }
    }
    band
}

/// `d.fill(v)` for the lengths a block's run of 8x8 positions has (a
/// power of two up to 16), in stores of a fixed size rather than a
/// library call.
#[inline(always)]
fn fill_run<T: Copy>(d: &mut [T], v: T) {
    match d {
        [a] => *a = v,
        [_, _] => d.copy_from_slice(&[v; 2]),
        [_, _, _, _] => d.copy_from_slice(&[v; 4]),
        [_, _, _, _, _, _, _, _] => d.copy_from_slice(&[v; 8]),
        _ if d.len() == 16 => d.copy_from_slice(&[v; 16]),
        _ => d.fill(v),
    }
}

/// The segment ids of the parsed frame of `fs` that later frames read:
/// none without segmentation, where every id is 0.
fn segment_map(hdr: &FrameHeader, fs: &FrameState) -> Vec<u8> {
    if hdr.segmentation.enabled {
        fs.segment_ids.clone()
    } else {
        Vec::new()
    }
}

/// The data of the parsed frame of `fs`, whole.
fn ref_data(seq: &SequenceHeader, hdr: &FrameHeader, fs: &FrameState) -> RefData {
    let rd = RefData::new(fs.mi_cols, fs.mi_rows, hdr.frame_type, hdr.order_hints);
    save_motion(seq, hdr, fs, &rd, 0, fs.mi_rows);
    rd.set_segment_ids(segment_map(hdr, fs));
    rd
}

/// Saves the motion of the bands of luma 4x4 rows `r0..r1` of the frame
/// of `fs` in `rd`, for the frames that project it.
fn save_motion(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &FrameState,
    rd: &RefData,
    r0: usize,
    r1: usize,
) {
    let per = 2 * MF_BAND_ROWS;
    for k in r0 / per..r1.min(fs.mi_rows).div_ceil(per).min(rd.bands()) {
        if !rd.has_band(k) {
            rd.set_band(k, mf_band(seq, hdr, fs, k, 0..fs.mi_cols >> 1));
        }
    }
}

fn to_fg_params(g: &GrainParams) -> FilmGrainParams {
    let mut p = FilmGrainParams {
        apply_grain: g.apply_grain,
        grain_seed: g.grain_seed,
        update_grain: g.update_grain,
        num_y_points: g.num_y_points,
        chroma_scaling_from_luma: g.chroma_scaling_from_luma,
        num_cb_points: g.num_cb_points,
        num_cr_points: g.num_cr_points,
        grain_scaling_minus_8: g.grain_scaling_minus_8,
        ar_coeff_lag: g.ar_coeff_lag,
        ar_coeffs_y_plus_128: g.ar_coeffs_y_plus_128,
        ar_coeffs_cb_plus_128: g.ar_coeffs_cb_plus_128,
        ar_coeffs_cr_plus_128: g.ar_coeffs_cr_plus_128,
        ar_coeff_shift_minus_6: g.ar_coeff_shift_minus_6,
        grain_scale_shift: g.grain_scale_shift,
        cb_mult: g.cb_mult,
        cb_luma_mult: g.cb_luma_mult,
        cb_offset: g.cb_offset,
        cr_mult: g.cr_mult,
        cr_luma_mult: g.cr_luma_mult,
        cr_offset: g.cr_offset,
        overlap_flag: g.overlap_flag,
        clip_to_restricted_range: g.clip_to_restricted_range,
        ..FilmGrainParams::default()
    };
    p.point_y_value.copy_from_slice(&g.point_y_value[..14]);
    p.point_y_scaling.copy_from_slice(&g.point_y_scaling[..14]);
    p.point_cb_value.copy_from_slice(&g.point_cb_value[..10]);
    p.point_cb_scaling
        .copy_from_slice(&g.point_cb_scaling[..10]);
    p.point_cr_value.copy_from_slice(&g.point_cr_value[..10]);
    p.point_cr_scaling
        .copy_from_slice(&g.point_cr_scaling[..10]);
    p
}

fn grain_apply<P: Pixel>(
    src: &Picture<P>,
    p: &FilmGrainParams,
    w: usize,
    h: usize,
    mono: bool,
    matrix: u8,
) -> Picture<P> {
    let mut dst = src.clone();
    let empty: [P; 0] = [];
    let (su, sv, suv_stride) = if mono {
        (&empty[..], &empty[..], 0)
    } else {
        (
            &src.planes[1].data[..],
            &src.planes[2].data[..],
            src.planes[1].stride,
        )
    };
    let (dy, rest) = dst.planes.split_at_mut(1);
    let ys = dy[0].stride;
    if mono {
        let mut e1: [P; 0] = [];
        let mut e2: [P; 0] = [];
        apply_film_grain(
            p,
            &src.planes[0].data,
            su,
            sv,
            src.planes[0].stride,
            suv_stride,
            &mut dy[0].data,
            &mut e1,
            &mut e2,
            ys,
            0,
            w,
            h,
            src.sub_x,
            src.sub_y,
            true,
            matrix,
            src.bd,
        );
    } else {
        let (du, dv) = rest.split_at_mut(1);
        let uvs = du[0].stride;
        apply_film_grain(
            p,
            &src.planes[0].data,
            su,
            sv,
            src.planes[0].stride,
            suv_stride,
            &mut dy[0].data,
            &mut du[0].data,
            &mut dv[0].data,
            ys,
            uvs,
            w,
            h,
            src.sub_x,
            src.sub_y,
            false,
            matrix,
            src.bd,
        );
    }
    dst
}

/// Takes a picture of the given format from the pool, or allocates one.
/// Pooled pictures keep old samples; the decoder writes every sample it reads.
fn take_picture<P: Pixel>(
    pool: &mut Vec<Picture<P>>,
    w: usize,
    h: usize,
    num_planes: usize,
    sub_x: bool,
    sub_y: bool,
    bd: u32,
) -> Picture<P> {
    let (aw, ah) = Picture::<P>::geometry(w, h);
    if let Some(i) = pool.iter().position(|p| {
        p.planes.len() == num_planes
            && p.sub_x == sub_x
            && p.sub_y == sub_y
            && p.bd == bd
            && p.planes[0].stride == aw
            && p.planes[0].alloc_h == ah
    }) {
        let mut p = pool.swap_remove(i);
        for (k, pl) in p.planes.iter_mut().enumerate() {
            let (sx, sy) = if k > 0 {
                (sub_x, sub_y)
            } else {
                (false, false)
            };
            pl.w = (w + usize::from(sx)) >> u32::from(sx);
            pl.h = (h + usize::from(sy)) >> u32::from(sy);
        }
        return p;
    }
    prof::count(prof::NEW_PIC);
    Picture::new(w, h, num_planes, sub_x, sub_y, bd)
}

const POOL_MAX: usize = 16;

fn recycle<P: Pixel>(pool: &mut Vec<Picture<P>>, p: Picture<P>) {
    if pool.len() < POOL_MAX {
        pool.push(p);
    }
}

/// What a tile decode reuses from one tile to the next: the above
/// contexts, the palette color maps and the prediction scratch.
struct TileBufs<P> {
    above_level: [Vec<u8>; 3],
    above_dc: [Vec<u8>; 3],
    above_seg_pred: Vec<u8>,
    above_palette: [Vec<[u16; 8]>; 2],
    color_map_y: Vec<u8>,
    color_map_uv: Vec<u8>,
    /// Always there between tile decodes: a decode takes it and gives it
    /// back (a move, rather than building a placeholder in its place).
    scratch: Option<Scratch<P>>,
    /// The BlockDecoded flags, taken and given back the same way.
    block_decoded: Option<Box<super::tile::BlockDecoded>>,
}

impl<P: Pixel> TileBufs<P> {
    /// Buffers for the tiles of frames with the size of `fs`.
    fn new(fs: &FrameState, scratch: Scratch<P>) -> Self {
        let n_above = fs.mi_stride + 64;
        Self {
            above_level: std::array::from_fn(|_| vec![0u8; n_above]),
            above_dc: std::array::from_fn(|_| vec![0u8; n_above]),
            above_seg_pred: vec![0u8; n_above],
            above_palette: std::array::from_fn(|_| vec![[0u16; 8]; n_above]),
            color_map_y: vec![0u8; 128 * 128],
            color_map_uv: vec![0u8; 128 * 128],
            scratch: Some(scratch),
            block_decoded: None,
        }
    }
}

/// What a tile carries from one superblock row to the next: its symbol
/// decoder and CDFs, the delta q and delta loop filter values, the loop
/// restoration references, and the last block.
struct TileCarry<'d> {
    tile_num: usize,
    msac: Msac<'d>,
    cdf: Box<CdfContext>,
    delta_lf: [i8; 4],
    current_q_index: i32,
    read_deltas: bool,
    ref_lr_wiener: [[[i8; 3]; 2]; 3],
    ref_sgr_xqd: [[i8; 2]; 3],
    b: Block,
    /// Whether the tile's first row is still to come.
    fresh: bool,
}

impl<'d> TileCarry<'d> {
    fn new(hdr: &FrameHeader, cdf: &CdfContext, tile_num: usize, data: &'d [u8]) -> Self {
        Self {
            tile_num,
            msac: Msac::new(data, hdr.disable_cdf_update),
            cdf: Box::new(cdf.clone()),
            delta_lf: [0; 4],
            current_q_index: i32::from(hdr.quant.base_q_idx),
            read_deltas: false,
            ref_lr_wiener: [[[0; 3]; 2]; 3],
            ref_sgr_xqd: [[0; 2]; 3],
            b: Block::default(),
            fresh: true,
        }
    }
}

/// Decodes the superblock rows of tile `carry` starting in luma 4x4 rows
/// `rows` into `fs` and `cur` (see [`decode_tiles`]), the tile's first
/// row starting it.
fn decode_tile_part<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &mut FrameState,
    cur: &mut Picture<P>,
    refs: &[Option<&Picture<P>>; 8],
    ref_hdr: &[RefHeaderState; 8],
    carry: &mut TileCarry<'_>,
    rows: Range<usize>,
    bufs: &mut TileBufs<P>,
    logs: Option<&Mutex<RowPools>>,
    mut feed: Option<&mut MfFeed>,
    on_row: &mut RowHook<'_, P>,
) -> R<()> {
    let num_planes = seq.num_planes();
    let sb128 = seq.use_128x128_superblock;
    let ti = &hdr.tile_info;
    let tile_row = carry.tile_num / ti.cols;
    let tile_col = carry.tile_num % ti.cols;
    let ref_views: [Option<RefView<'_, P>>; 8] = std::array::from_fn(|i| {
        if i == 0 {
            return None;
        }
        let slot = hdr.ref_frame_idx[i - 1] as usize;
        refs[i].map(|pic| RefView {
            pic,
            mirror: None,
            upscaled_width: ref_hdr[slot].upscaled_width as usize,
            frame_height: ref_hdr[slot].frame_height as usize,
            frame_width: ref_hdr[slot].frame_width as usize,
        })
    });
    let msac = std::mem::replace(&mut carry.msac, Msac::new(&[], true));
    let mut td = TileDec {
        seq,
        hdr,
        fs: &mut *fs,
        cur: &mut *cur,
        refs: ref_views,
        mirrors: Vec::new(),
        msac,
        cdf: &mut carry.cdf,
        mi_row_start: ti.mi_row_starts[tile_row] as usize,
        mi_row_end: ti.mi_row_starts[tile_row + 1] as usize,
        mi_col_start: ti.mi_col_starts[tile_col] as usize,
        mi_col_end: ti.mi_col_starts[tile_col + 1] as usize,
        sb_size: if sb128 { 15 } else { 12 },
        sb_size4: if sb128 { 32 } else { 16 },
        bd: seq.bit_depth,
        sub_x: seq.subsampling_x,
        sub_y: seq.subsampling_y,
        num_planes,
        above_level: std::mem::take(&mut bufs.above_level),
        above_dc: std::mem::take(&mut bufs.above_dc),
        above_seg_pred: std::mem::take(&mut bufs.above_seg_pred),
        left_level: [[0; 32]; 3],
        left_dc: [[0; 32]; 3],
        left_seg_pred: [0; 32],
        above_palette: std::mem::take(&mut bufs.above_palette),
        left_palette: [[[0; 8]; 32]; 2],
        delta_lf: carry.delta_lf,
        current_q_index: carry.current_q_index,
        read_deltas: carry.read_deltas,
        ref_lr_wiener: carry.ref_lr_wiener,
        ref_sgr_xqd: carry.ref_sgr_xqd,
        block_decoded: bufs.block_decoded.take().unwrap_or_default(),
        b: std::mem::take(&mut carry.b),
        pb: Default::default(),
        plan: Default::default(),
        src: None,
        enc: Default::default(),
        mode: Default::default(),
        me_scratch: crate::enc::me::MeScratch::empty(),
        plan_q: 0,
        cfl_choice: (0, 0),
        intra_choice: (0, None),
        est_hint: (0, 0),
        cur_filters: [0, 0],
        plan_delta_lf: [0; 4],
        color_map_y: std::mem::take(&mut bufs.color_map_y),
        color_map_uv: std::mem::take(&mut bufs.color_map_uv),
        scratch: bufs.scratch.take().unwrap_or_default(),
        rec: logs.map(|l| Box::new(lock(l).logs.pop().unwrap_or_default())),
        y0: 0,
        mi_r0: 0,
        mi_fill: MiFill::OddRows,
    };
    if carry.fresh {
        td.begin_tile();
        carry.fresh = false;
    }
    let res = td.decode_rows(rows, &mut *on_row, &mut |fs: &mut FrameState, r1: usize| {
        if let Some(f) = feed.as_deref_mut() {
            f.need(fs, r1);
        }
    });
    let TileDec {
        msac,
        delta_lf,
        current_q_index,
        read_deltas,
        ref_lr_wiener,
        ref_sgr_xqd,
        b,
        above_level,
        above_dc,
        above_seg_pred,
        above_palette,
        color_map_y,
        color_map_uv,
        scratch,
        rec,
        block_decoded,
        ..
    } = td;
    carry.msac = msac;
    carry.delta_lf = delta_lf;
    carry.current_q_index = current_q_index;
    carry.read_deltas = read_deltas;
    carry.ref_lr_wiener = ref_lr_wiener;
    carry.ref_sgr_xqd = ref_sgr_xqd;
    carry.b = b;
    bufs.above_level = above_level;
    bufs.above_dc = above_dc;
    bufs.above_seg_pred = above_seg_pred;
    bufs.above_palette = above_palette;
    bufs.color_map_y = color_map_y;
    bufs.color_map_uv = color_map_uv;
    bufs.scratch = Some(scratch);
    bufs.block_decoded = Some(block_decoded);
    // The log left holds the buffers of a spare: back to the pool.
    if let (Some(l), Some(mut log)) = (logs, rec) {
        log.clear();
        lock(l).logs.push(*log);
    }
    res
}

/// Decodes tile `tile_num` of a frame into `fs` and `cur` (see
/// [`decode_tiles`]). Returns the CDFs at the end of the tile.
fn decode_tile<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &mut FrameState,
    cur: &mut Picture<P>,
    cdf: &CdfContext,
    refs: &[Option<&Picture<P>>; 8],
    ref_hdr: &[RefHeaderState; 8],
    tile_num: usize,
    data: &[u8],
    bufs: &mut TileBufs<P>,
    logs: Option<&Mutex<RowPools>>,
    feed: Option<&mut MfFeed>,
    on_row: &mut RowHook<'_, P>,
) -> R<Box<CdfContext>> {
    let mut carry = TileCarry::new(hdr, cdf, tile_num, data);
    decode_tile_part(
        seq,
        hdr,
        fs,
        cur,
        refs,
        ref_hdr,
        &mut carry,
        0..usize::MAX,
        bufs,
        logs,
        feed,
        on_row,
    )?;
    Ok(carry.cdf)
}

/// Decodes the tiles of a frame into `cur`, calling `on_row(fs, cur, r0,
/// r1)` after each superblock row of each tile (luma 4x4 rows `r0..r1`).
/// With `logs`, the tiles record what reconstruction needs (see
/// [`ReconLog`]) in logs taken from and given back to that pool.
/// Returns the CDFs saved from the context tile.
fn decode_tiles<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &mut FrameState,
    tiles: &[(usize, Vec<u8>)],
    cdf: &mut CdfContext,
    refs: &[Option<&Picture<P>>; 8],
    ref_hdr: &[RefHeaderState; 8],
    cur: &mut Picture<P>,
    logs: Option<&Mutex<RowPools>>,
    mut feed: Option<&mut MfFeed>,
    on_row: &mut RowHook<'_, P>,
    spares: Option<&Mutex<Spares<P>>>,
) -> R<Box<CdfContext>> {
    let ti = &hdr.tile_info;
    // Recording tiles only parse: the prediction buffers stay empty.
    let scratch = if logs.is_some() {
        Scratch::<P>::parsing()
    } else {
        spares
            .and_then(|s| lock(s).scratch.pop())
            .unwrap_or_default()
    };
    let mut bufs = TileBufs::new(fs, scratch);
    let mut saved: Option<Box<CdfContext>> = None;
    for (tile_num, data) in tiles {
        let tile_cdf = decode_tile(
            seq,
            hdr,
            fs,
            cur,
            cdf,
            refs,
            ref_hdr,
            *tile_num,
            data,
            &mut bufs,
            logs,
            feed.as_deref_mut(),
            on_row,
        )?;
        if *tile_num == ti.context_update_tile_id as usize {
            saved = Some(tile_cdf);
        }
    }
    if let (Some(s), None) = (spares, logs) {
        lock(s).scratch.extend(bufs.scratch);
    }
    Ok(saved.unwrap_or_else(|| Box::new(cdf.clone())))
}

/// What [`decode_tiles_by_rows`] calls after each superblock row.
type AfterRow<'h, P> = dyn FnMut(&FrameState, &mut Picture<P>, usize, usize, bool) + 'h;

/// [`decode_tiles`] one superblock row of the frame at a time: each row
/// of every tile of a tile row in turn, then `after_row(fs, cur, r0, r1,
/// last)` for the luma 4x4 rows `r0..r1` the row covers (`last`: the
/// bottom of the frame). The tiles decode to the same result, since a
/// tile reads only its own earlier rows and the reference frames. `None`
/// when `tiles` is not every tile of the frame once.
fn decode_tiles_by_rows<P: Pixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &mut FrameState,
    tiles: &[(usize, Vec<u8>)],
    cdf: &CdfContext,
    refs: &[Option<&Picture<P>>; 8],
    ref_hdr: &[RefHeaderState; 8],
    cur: &mut Picture<P>,
    spares: &Mutex<Spares<P>>,
    after_row: &mut AfterRow<'_, P>,
) -> Option<R<Box<CdfContext>>> {
    let ti = &hdr.tile_info;
    let n = ti.rows * ti.cols;
    let mut data: Vec<Option<&[u8]>> = vec![None; n];
    for (num, d) in tiles {
        match data.get_mut(*num) {
            Some(slot @ None) => *slot = Some(d),
            _ => return None,
        }
    }
    if data.iter().any(Option::is_none) {
        return None;
    }
    let mut scratch = lock(spares).scratch.pop().unwrap_or_default();
    scratch.edge_y = usize::MAX;
    let mut bufs = TileBufs::new(fs, scratch);
    let sb4 = if seq.use_128x128_superblock { 32 } else { 16 };
    let ctx = ti.context_update_tile_id as usize;
    let mut saved = None;
    for tr in 0..ti.rows {
        let mut carries: Vec<TileCarry<'_>> = (0..ti.cols)
            .map(|tc| {
                let num = tr * ti.cols + tc;
                TileCarry::new(hdr, cdf, num, data[num].unwrap_or_default())
            })
            .collect();
        let (r0, r1) = (
            ti.mi_row_starts[tr] as usize,
            ti.mi_row_starts[tr + 1] as usize,
        );
        let mut r = r0;
        while r < r1 {
            let end = (r + sb4).min(r1);
            for carry in &mut carries {
                let res = decode_tile_part(
                    seq,
                    hdr,
                    fs,
                    cur,
                    refs,
                    ref_hdr,
                    carry,
                    r..end,
                    &mut bufs,
                    None,
                    None,
                    &mut |_: &FrameState,
                          _: &Picture<P>,
                          _: Option<&mut ReconLog>,
                          _: usize,
                          _: usize,
                          _: bool| {},
                );
                if let Err(e) = res {
                    return Some(Err(e));
                }
            }
            // The row's filters may run now: intra prediction below reads
            // its last lines as they are.
            if let Some(s) = &mut bufs.scratch {
                s.save_edge(cur, 4 * end, 0);
            }
            after_row(fs, cur, r, end, tr + 1 == ti.rows && end == r1);
            r = end;
        }
        if let Some(c) = carries.into_iter().find(|c| c.tile_num == ctx) {
            saved = Some(c.cdf);
        }
    }
    if let Some(mut s) = bufs.scratch {
        s.edge_y = usize::MAX;
        lock(spares).scratch.push(s);
    }
    Some(Ok(saved.unwrap_or_else(|| Box::new(cdf.clone()))))
}

/// Rows `rows` and columns `cols` of a row-major array, copied out.
#[derive(Default)]
struct Rect<T> {
    rows: Range<usize>,
    cols: Range<usize>,
    data: Vec<T>,
}

impl<T: Copy> Rect<T> {
    /// Copies the area out of `src`, `stride` wide, clipped to it.
    fn take(&mut self, src: &[T], stride: usize, rows: Range<usize>, cols: Range<usize>) {
        self.take_ring(src, stride, usize::MAX, rows, cols);
    }

    /// [`Self::take`] from an array whose rows go round a ring: row `r` is
    /// at row `r & ring` (see [`FrameState::mi_ring`]).
    fn take_ring(
        &mut self,
        src: &[T],
        stride: usize,
        ring: usize,
        rows: Range<usize>,
        cols: Range<usize>,
    ) {
        self.data.clear();
        let height = src.len() / stride.max(1);
        let rows = rows.start.min(height)..rows.end.min(height);
        let cols = cols.start.min(stride)..cols.end.min(stride);
        // Room for the area at once: grown row by row, the buffer is
        // copied each time it doubles.
        self.data.reserve(rows.len() * cols.len());
        for r in rows.clone() {
            let at = (r & ring) * stride;
            self.data
                .extend_from_slice(&src[at + cols.start..at + cols.end]);
        }
        self.rows = rows;
        self.cols = cols;
    }

    /// Writes the area into `dst`, `stride` wide.
    fn put(&self, dst: &mut [T], stride: usize) {
        self.put_ring(dst, stride, usize::MAX);
    }

    /// [`Self::put`] into an array whose rows go round a ring, as in
    /// [`Self::take_ring`].
    fn put_ring(&self, dst: &mut [T], stride: usize, ring: usize) {
        let w = self.cols.len();
        if w == 0 {
            return;
        }
        for (r, row) in self.rows.clone().zip(self.data.chunks_exact(w)) {
            let at = (r & ring) * stride + self.cols.start;
            if let Some(d) = dst.get_mut(at..at + w) {
                d.copy_from_slice(row);
            }
        }
    }
}

/// What the in-loop filters need of one tile's part of a parsed superblock
/// row besides the mode info reconstruction stores: luma 4x4 rows
/// `r0..r1`, and in the tile's columns their loop filter transform sizes,
/// CDEF indices and loop restoration units.
#[derive(Default)]
struct RowMsg {
    r0: usize,
    r1: usize,
    last: bool,
    /// Whether this is the last tile column's part: the row is whole.
    row_end: bool,
    lf_tx: [Rect<u8>; 3],
    cdef: Rect<i8>,
    lr: [Rect<LrUnitParams>; 3],
}

/// Ends a motion field when its thread does: the bands it did not get to
/// stay unset, and no parser waits for them.
struct MfClose(Arc<MfShared>);

impl Drop for MfClose {
    fn drop(&mut self) {
        for (k, b) in self.0.bands.iter().enumerate() {
            let _ = b.set(self.0.blank(k));
        }
    }
}

/// One tile parser's wait for the motion field, which its frame state
/// reads where the thread computing it puts it (see
/// [`FrameState::motion_shared`]).
struct MfFeed {
    shared: Arc<MfShared>,
    /// Bands known to be there.
    have: usize,
}

impl MfFeed {
    /// Waits for the bands luma 4x4 rows up to `r1` use.
    fn need(&mut self, _fs: &mut FrameState, r1: usize) {
        let need = r1.div_ceil(2).div_ceil(8).min(self.shared.bands.len());
        while self.have < need {
            let tw = std::time::Instant::now();
            self.shared.bands[self.have].wait();
            prof::mark(prof::MF_WAIT, tw);
            self.have += 1;
        }
    }
}

/// Blocks a reconstruction log has room for when the parser takes it.
const LOG_RESERVE_BLOCKS: usize = 256;

/// Spare [`RowMsg`]s and [`ReconLog`]s, with their buffers.
#[derive(Default)]
pub(crate) struct RowPools {
    msgs: Vec<RowMsg>,
    logs: Vec<ReconLog>,
}

impl RowMsg {
    /// Copies the transform sizes, CDEF indices and loop restoration units
    /// of luma 4x4 rows `r0..r1` in the tile columns `cols` (to the end
    /// of the arrays for the last column, `last_col`, and the last row).
    fn fill_meta(
        &mut self,
        seq: &SequenceHeader,
        hdr: &FrameHeader,
        fs: &FrameState,
        r0: usize,
        r1: usize,
        cols: Range<usize>,
        last_col: bool,
    ) {
        let last = r1 >= fs.mi_rows;
        self.r0 = r0;
        self.r1 = r1;
        self.last = last;
        self.row_end = false;
        let st = fs.mi_stride;
        // The transform size maps are indexed in each plane's 4x4 units.
        for (p, (d, s)) in self.lf_tx.iter_mut().zip(&fs.lf_tx).enumerate() {
            let sx = usize::from(p > 0 && seq.subsampling_x);
            let sy = usize::from(p > 0 && seq.subsampling_y);
            // Past the last row of the plane, nothing reads them.
            let rows = (r0 >> sy)..if last {
                fs.mi_rows.div_ceil(1 << sy)
            } else {
                r1 >> sy
            };
            let c = (cols.start >> sx)..if last_col { st } else { cols.end >> sx };
            d.take_ring(s, st, fs.mi_ring, rows, c);
        }
        // CDEF indices are per 64x64 luma: 16 4x4s.
        let cs = fs.cdef_stride;
        let rows = (r0 >> 4)..if last { usize::MAX } else { r1.div_ceil(16) };
        let c = (cols.start >> 4)..if last_col { cs } else { cols.end >> 4 };
        self.cdef.take(&fs.cdef_idx, cs, rows, c);
        for (p, (d, s)) in self.lr.iter_mut().zip(&fs.lr).enumerate() {
            if hdr.lr.frame_type[p] == RESTORE_NONE || s.is_empty() {
                d.take(s, 1, 0..0, 0..0);
                continue;
            }
            let (ur, uc) = lr_unit_span(seq, hdr, fs, p, r0..r1, cols.clone());
            d.take(s, fs.lr_cols[p], ur, uc);
        }
    }

    /// Writes the transform sizes, CDEF indices and loop restoration units
    /// into `fs` (whose mode info reconstruction wrote).
    fn apply_meta(&self, fs: &mut FrameState) {
        let (st, ring) = (fs.mi_stride, fs.mi_ring);
        for (d, s) in fs.lf_tx.iter_mut().zip(&self.lf_tx) {
            s.put_ring(d, st, ring);
        }
        self.cdef.put(&mut fs.cdef_idx, fs.cdef_stride);
        for (p, s) in self.lr.iter().enumerate() {
            s.put(&mut fs.lr[p], fs.lr_cols[p]);
        }
    }
}

/// The shells of reconstruction and of the filters for the frame of
/// `fs`, from the pools when one fits.
fn take_shells<P>(
    sp: &mut Spares<P>,
    fs: &FrameState,
    band: Option<usize>,
) -> (FrameState, FrameState) {
    let recon = match band {
        // Reconstruction keeps the mode info of `rows` rows (see
        // TileDec::mi_r0).
        Some(rows) => match sp.shells.iter().position(|f| f.fits_band_shell(fs, rows)) {
            Some(i) => sp.shells.swap_remove(i).reshape_shell(fs),
            None => fs.band_recon_shell(rows),
        },
        None => match sp.shells.iter().position(|f| f.fits_recon_shell(fs)) {
            Some(i) => sp.shells.swap_remove(i).reshape_shell(fs),
            None => {
                prof::count(prof::NEW_SHELL);
                fs.recon_shell()
            }
        },
    };
    let filter = match sp
        .filter_shells
        .iter()
        .position(|f| f.fits_filter_shell(fs))
    {
        Some(i) => sp.filter_shells.swap_remove(i).reshape_shell(fs),
        None => fs.filter_shell(),
    };
    (recon, filter)
}

/// What reconstruction hands the in-loop filters of its frame: a log of
/// blocks it has reconstructed (the filters read their mode info), with
/// its tile's part of the row's filter data when it is the tile's last of
/// a superblock row, and with the row's unfiltered samples when it is
/// the row's last.
type Filtered<P> = (ReconLog, Option<RowMsg>, Option<Samples<P>>);

/// The unfiltered samples of a superblock row, for its filters.
enum Samples<P> {
    /// Its whole rows (see [`copy_rows_out`]).
    Rows([Vec<P>; 3]),
    /// A picture of the row alone (see [`take_sb_row`]).
    Row(Picture<P>),
}

/// A picture of one superblock row of a frame, which reconstruction
/// writes (see [`TileDec::y0`]) and hands to the filters as it is, from
/// the pool when one fits.
fn take_sb_row<P: Pixel>(
    sp: &mut Spares<P>,
    seq: &SequenceHeader,
    hdr: &FrameHeader,
) -> Picture<P> {
    let (w, h) = (hdr.frame_width as usize, hdr.frame_height as usize);
    let (aw, _) = Picture::<P>::geometry(w, h);
    let sb = if seq.use_128x128_superblock { 128 } else { 64 };
    let (sub_x, sub_y, bd) = (seq.subsampling_x, seq.subsampling_y, seq.bit_depth);
    let np = seq.num_planes();
    let i = sp.sb_rows.iter().position(|p| {
        p.planes.len() == np
            && p.sub_x == sub_x
            && p.sub_y == sub_y
            && p.planes[0].stride == aw
            && p.planes[0].alloc_h == sb
    });
    let mut p = match i {
        Some(i) => sp.sb_rows.swap_remove(i),
        None => {
            let planes = (0..np)
                .map(|k| {
                    let (sx, sy) = if k > 0 {
                        (usize::from(sub_x), usize::from(sub_y))
                    } else {
                        (0, 0)
                    };
                    Plane::new((w + sx) >> sx, sb >> sy, aw >> sx, sb >> sy)
                })
                .collect();
            Picture {
                planes,
                bd,
                sub_x,
                sub_y,
            }
        }
    };
    p.bd = bd;
    for (k, pl) in p.planes.iter_mut().enumerate() {
        let sx = usize::from(k > 0 && sub_x);
        pl.w = (w + sx) >> sx;
    }
    p
}

/// Mode-info rows of reconstruction's shell when it writes a superblock
/// row at a time (see [`take_sb_row`]): the row's, and the last of the row
/// above.
fn band_rows(seq: &SequenceHeader) -> usize {
    if seq.use_128x128_superblock { 33 } else { 17 }
}

/// Keeps `p`, a picture of a superblock row, for reuse.
fn give_sb_row<P>(sp: &mut Spares<P>, p: Picture<P>) {
    if sp.sb_rows.len() < MAX_SB_ROWS {
        sp.sb_rows.push(p);
    }
}

/// Writes `src`, a superblock row from luma 4x4 row `r0` (see
/// [`take_sb_row`]), into `pic`, a picture of the same stride.
fn copy_row_in<P: Pixel>(pic: &mut Picture<P>, r0: usize, src: &Picture<P>) {
    let sub_y = pic.sub_y;
    for (p, (d, s)) in pic.planes.iter_mut().zip(&src.planes).enumerate() {
        let at = ((4 * r0) >> usize::from(p > 0 && sub_y)) * d.stride;
        let n = s.data.len().min(d.data.len().saturating_sub(at));
        if d.stride == s.stride {
            d.data[at..at + n].copy_from_slice(&s.data[..n]);
        }
    }
}

/// The samples of luma 4x4 rows `r0..r1` of `pic` (to the end of the
/// planes when `last`), whole rows of each plane, copied into `bufs`.
fn copy_rows_out<P: Pixel>(
    pic: &Picture<P>,
    r0: usize,
    r1: usize,
    last: bool,
    mut bufs: [Vec<P>; 3],
) -> [Vec<P>; 3] {
    for (p, buf) in bufs.iter_mut().enumerate() {
        buf.clear();
        let Some(pl) = pic.planes.get(p) else {
            continue;
        };
        let sy = usize::from(p > 0 && pic.sub_y);
        let height = pl.data.len() / pl.stride.max(1);
        let y0 = ((4 * r0) >> sy).min(height);
        let y1 = if last {
            height
        } else {
            ((4 * r1) >> sy).clamp(y0, height)
        };
        buf.extend_from_slice(&pl.data[y0 * pl.stride..y1 * pl.stride]);
    }
    bufs
}

/// Writes rows copied by [`copy_rows_out`] from luma 4x4 row `r0` into
/// `pic`, a picture of the same layout.
fn copy_rows_in<P: Pixel>(pic: &mut Picture<P>, r0: usize, bufs: &[Vec<P>; 3]) {
    let sub_y = pic.sub_y;
    for (p, (pl, buf)) in pic.planes.iter_mut().zip(bufs).enumerate() {
        let at = ((4 * r0) >> usize::from(p > 0 && sub_y)) * pl.stride;
        if let Some(d) = pl.data.get_mut(at..at + buf.len()) {
            d.copy_from_slice(buf);
        }
    }
}

/// The in-loop filters of a frame run by its reconstruction, in place in
/// the picture it writes (as on one thread, see [`decode_frame_sync`]): a
/// row is deblocked once reconstructed, and intra prediction in the row
/// below reads a copy of its last lines (see [`Scratch::edge`]).
struct InPlace<'a, P> {
    filt: InlineFilter<'a, P>,
    /// The filters' shell of the frame state.
    fs: FrameState,
    info: postfilter::FilterInfo,
    levels: postfilter::LfLevels,
    rows: &'a Mutex<RowPools>,
    back: (Vec<ReconLog>, Vec<RowMsg>),
}

impl<P: PicturePixel> InPlace<'_, P> {
    /// Takes a reconstructed chunk of blocks, and the tile's part of the
    /// row's filter data when the chunk ends it.
    fn take(
        &mut self,
        seq: &SequenceHeader,
        hdr: &FrameHeader,
        cur: &mut Picture<P>,
        mut log: ReconLog,
        m: Option<RowMsg>,
    ) {
        let size = (self.fs.mi_rows, self.fs.mi_cols);
        for rec in &log.blocks {
            let b = &rec.b;
            let of = self.levels.info(
                hdr,
                b.ref_frame[0],
                b.segment_id,
                b.y_mode,
                b.skip,
                b.bsize,
                b.delta_lf,
            );
            self.info
                .store((b.mi_row as usize, b.mi_col as usize), size, of);
        }
        log.clear();
        self.back.0.push(log);
        if let Some(m) = m {
            m.apply_meta(&mut self.fs);
            if m.row_end {
                let tf = std::time::Instant::now();
                self.filt.deblock(
                    seq, hdr, &self.fs, &self.info, cur, m.r0, m.r1, m.last, None,
                );
                prof::add(prof::FILTER, tf);
            }
            self.back.1.push(m);
        }
        if self.back.0.len() >= 32 {
            self.give_back();
        }
    }

    fn give_back(&mut self) {
        let mut pool = lock(self.rows);
        pool.logs.append(&mut self.back.0);
        pool.msgs.append(&mut self.back.1);
    }
}

/// Predicts and reconstructs the superblock rows parsed on another thread
/// (arriving on `rx` in chunks of blocks) into `cur`, which stays
/// unfiltered, and hands each chunk on to the frame's filters (`filters`),
/// with a copy of the samples of every row it completes.
///
/// References still being decoded are read from the bands in `mirrors` (see
/// [`RefView::mirror`]), which `fills` bring up to the rows each chunk of
/// blocks may read before it is reconstructed. Returns the mirrors.
///
/// With `inplace`, the filters run here instead, in `cur` (see [`InPlace`]),
/// and nothing goes to `filters`.
///
/// With `banded`, `cur` is a picture of one superblock row: each row done
/// goes to the filters as it is, and the next is reconstructed into a new
/// one. With `band_mi`, `fs` holds the mode info of one superblock row and
/// the row above it.
fn recon_worker<'r, P: PicturePixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &mut FrameState,
    cur: &mut Picture<P>,
    refs: [Option<RefView<'r, P>>; 8],
    mirrors: Vec<Mirror<P>>,
    finals: &'r [OnceLock<Arc<AnyPicture>>],
    scratch: Scratch<P>,
    mut fills: Vec<RefFill>,
    rx: Receiver<(Option<RowMsg>, ReconLog)>,
    filters: &Sender<Filtered<P>>,
    mut inplace: Option<&mut InPlace<'_, P>>,
    spares: &Mutex<Spares<P>>,
    banded: bool,
    band_mi: bool,
) -> R<(Vec<Mirror<P>>, Scratch<P>)> {
    // A reference read through a scale may need any row.
    let scaled = std::array::from_fn::<bool, 8, _>(|i| {
        refs[i].as_ref().is_some_and(|rv| {
            rv.upscaled_width != hdr.frame_width as usize
                || rv.frame_height != hdr.frame_height as usize
        })
    });
    for f in &mut fills {
        if scaled[f.ref_frame] {
            f.whole = true;
        }
    }
    // The largest downward motion into each reference frame, in eighth
    // samples, in the superblock row above and in the row being
    // reconstructed: blocks also predict with the motion of the
    // neighbours above and left of them (OBMC, small chroma). `UNREAD`
    // for a reference frame no block of the row predicts from.
    const UNREAD: i32 = i32::MIN;
    let mut down = [[UNREAD; 8]; 2];
    // The copy each reference frame is read through, if any.
    let mirror_of: [Option<usize>; 8] =
        std::array::from_fn(|i| refs[i].as_ref().and_then(|rv| rv.mirror));
    let mut cdf = CdfContext::new(hdr.quant.base_q_idx);
    let sb128 = seq.use_128x128_superblock;
    let mut td = TileDec {
        seq,
        hdr,
        fs,
        cur,
        refs,
        mirrors,
        msac: Msac::new(&[], true),
        cdf: &mut cdf,
        mi_row_start: 0,
        mi_row_end: 0,
        mi_col_start: 0,
        mi_col_end: 0,
        sb_size: if sb128 { 15 } else { 12 },
        sb_size4: if sb128 { 32 } else { 16 },
        bd: seq.bit_depth,
        sub_x: seq.subsampling_x,
        sub_y: seq.subsampling_y,
        num_planes: seq.num_planes(),
        above_level: Default::default(),
        above_dc: Default::default(),
        above_seg_pred: Vec::new(),
        left_level: [[0; 32]; 3],
        left_dc: [[0; 32]; 3],
        left_seg_pred: [0; 32],
        above_palette: Default::default(),
        left_palette: [[[0; 8]; 32]; 2],
        delta_lf: [0; 4],
        current_q_index: i32::from(hdr.quant.base_q_idx),
        read_deltas: false,
        ref_lr_wiener: [[[0; 3]; 2]; 3],
        ref_sgr_xqd: [[0; 2]; 3],
        block_decoded: Box::default(),
        b: Default::default(),
        pb: Default::default(),
        plan: Default::default(),
        src: None,
        enc: Default::default(),
        mode: Default::default(),
        me_scratch: crate::enc::me::MeScratch::empty(),
        plan_q: 0,
        cfl_choice: (0, 0),
        intra_choice: (0, None),
        est_hint: (0, 0),
        cur_filters: [0, 0],
        plan_delta_lf: [0; 4],
        color_map_y: vec![0u8; 128 * 128],
        color_map_uv: vec![0u8; 128 * 128],
        scratch,
        rec: None,
        y0: 0,
        mi_r0: 0,
        // Reconstruction reads the mode info of neighbours only.
        mi_fill: MiFill::Edges,
    };
    let ahead = lock(spares).rows_ahead;
    td.scratch.edge_y = usize::MAX;
    for (m, log) in rx {
        if ahead && !fills.is_empty() {
            let mut bottom = 0;
            for rec in &log.blocks {
                let b = &rec.b;
                if !b.is_inter {
                    continue;
                }
                bottom = bottom.max(4 * (b.mi_row as usize + super::types::bh4(b.bsize)));
                for l in 0..2 {
                    if b.ref_frame[l] > INTRA_FRAME {
                        let d = &mut down[1][b.ref_frame[l] as usize & 7];
                        *d = (*d).max(i32::from(b.mv[l].row));
                    }
                }
            }
            if bottom > 0 {
                // The 8-tap filters read 4 rows below a sample; chroma
                // positions round, and 16 luma rows cover both.
                for f in &mut fills {
                    let reach = (1..8)
                        .filter(|&r| mirror_of[r] == Some(f.mirror))
                        .map(|r| down[0][r].max(down[1][r]))
                        .max()
                        .unwrap_or(UNREAD);
                    if reach == UNREAD {
                        // Nothing here reads this reference: no need to
                        // wait for it.
                        continue;
                    }
                    let need = if f.whole {
                        usize::MAX
                    } else {
                        bottom + (reach.max(0) as usize).div_ceil(8) + 16
                    };
                    f.bring(need, &mut td.mirrors, &mut td.refs, finals, spares)?;
                }
            }
        }
        let tr = std::time::Instant::now();
        let mut at = (0, 0);
        while let Err(e) = td.replay(&log, &mut at) {
            // A warped block reads rows its motion vector does not name:
            // it stops at the first that are not published yet, and is
            // predicted again once they are.
            let Some((k, need)) = td.scratch.want_rows.take() else {
                return Err(e);
            };
            let Some(f) = fills.iter_mut().find(|f| f.mirror == k) else {
                return Err(e);
            };
            let mut need = if f.whole { usize::MAX } else { need };
            if f.have >= need {
                // Rows that are published and still cannot be read: only
                // the finished picture will do, and that was tried if all
                // of it is here.
                if f.have == usize::MAX {
                    return Err(e);
                }
                need = usize::MAX;
            }
            lock(spares).row_waits += 1;
            f.bring(need, &mut td.mirrors, &mut td.refs, finals, spares)?;
        }
        prof::add(prof::RECON, tr);
        let done = m.as_ref().filter(|m| m.row_end);
        if let Some(m) = done {
            down = [down[1], [UNREAD; 8]];
            td.scratch.save_edge(td.cur, 4 * m.r1, td.y0);
            if band_mi {
                // The next row's blocks read the mode info of the last row
                // of this one.
                let st = td.fs.mi_stride;
                let from = (m.r1 - 1 - td.mi_r0) * st;
                if td.fs.mi.len() >= from + st {
                    td.fs.mi.copy_within(from..from + st, 0);
                }
                if td.fs.uv_modes.len() >= from + st {
                    td.fs.uv_modes.copy_within(from..from + st, 0);
                }
                td.mi_r0 = m.r1 - 1;
            }
        }
        if let Some(ip) = inplace.as_deref_mut() {
            ip.take(seq, hdr, td.cur, log, m);
            continue;
        }
        // The samples of a whole row go to the filters, which work on a
        // picture of their own: this one stays unfiltered for the intra
        // prediction of the rows below.
        let samples = match done {
            None => None,
            Some(m) if !banded => {
                let bufs = lock(spares).bands.pop().unwrap_or_default();
                Some(Samples::Rows(copy_rows_out(
                    td.cur, m.r0, m.r1, m.last, bufs,
                )))
            }
            Some(m) => {
                let row = std::mem::replace(td.cur, take_sb_row(&mut lock(spares), seq, hdr));
                td.y0 = 4 * m.r1;
                Some(Samples::Row(row))
            }
        };
        if filters.send((log, m, samples)).is_err() {
            // The filters stopped: their thread failed.
            return Err(malformed("filter thread"));
        }
    }
    td.scratch.edge_y = usize::MAX;
    Ok((
        std::mem::take(&mut td.mirrors),
        std::mem::replace(&mut td.scratch, Scratch::empty()),
    ))
}

/// The in-loop filters of a frame, run on a thread of their own on the
/// superblock rows reconstruction has finished (see [`FilterStage`]): a
/// row is deblocked in place once it arrives, and gets CDEF and loop
/// restoration, also in place,
/// once the row below is deblocked (they read 2 and 3 rows past it). Final
/// rows are published to `publish` for the frames that reference this one.
struct InlineFilter<'a, P> {
    cdef_on: bool,
    lr_on: bool,
    lr_scratch: &'a mut LrScratch,
    /// CDEF scratch, with the pre-CDEF rows next to the boundaries.
    cdef: postfilter::CdefScratch<P>,
    /// The deblocked rows next to the loop restoration stripe boundaries.
    lr_bounds: postfilter::LrBounds<P>,
    lf_on: bool,
    /// No job: nothing waits on the rows (the frame is done on one
    /// thread).
    publish: Option<&'a FrameJob>,
    /// Where the buffers of the final rows come from.
    spares: &'a Mutex<Spares<P>>,
    /// Deblocked luma 4x4 rows waiting for CDEF and loop restoration.
    to_filter: Option<(usize, usize)>,
    published: usize,
    /// The final rows published, once a frame reads them.
    rows_out: Option<RowsWriter<P>>,
}

impl<P: PicturePixel> InlineFilter<'_, P> {
    /// Hands the buffers of the published rows back once no other thread
    /// holds them.
    fn release_rows(&mut self) {
        if let Some(w) = self.rows_out.take() {
            let sh = w.shared().clone();
            drop(w);
            lock(self.spares).give_shared(sh);
        }
    }
}
impl<P: PicturePixel> InlineFilter<'_, P> {
    /// Takes the reconstructed superblock row of `m`.
    fn row(
        &mut self,
        seq: &SequenceHeader,
        hdr: &FrameHeader,
        fs: &mut FrameState,
        info: &postfilter::FilterInfo,
        cur: &mut Picture<P>,
        m: &RowMsg,
        src: Option<&Picture<P>>,
    ) {
        m.apply_meta(fs);
        self.deblock(seq, hdr, fs, info, cur, m.r0, m.r1, m.last, src);
    }

    fn deblock(
        &mut self,
        seq: &SequenceHeader,
        hdr: &FrameHeader,
        fs: &FrameState,
        info: &postfilter::FilterInfo,
        cur: &mut Picture<P>,
        a: usize,
        b: usize,
        last: bool,
        src: Option<&Picture<P>>,
    ) {
        // `src`, the samples of the superblock row from `a` on, are copied
        // in first (by deblocking, as it goes).
        if self.lf_on {
            postfilter::loop_filter_rows_with(seq, hdr, fs, info, cur, a, b, last, src);
        } else if let Some(src) = src {
            copy_row_in(cur, a, src);
        }
        // Deblocking the next row changes up to 6 luma rows above it (2
        // chroma rows), and CDEF reads 2 rows past what it filters: the
        // rows 8 above it get CDEF (in 64-row areas moved up as many rows)
        // and loop restoration (whose stripes lie the same way) now.
        let start = self.to_filter.take().map_or(a, |(c, _)| c);
        if last {
            self.filter_ready_rows(seq, hdr, fs, info, cur, start, b, true);
            return;
        }
        let ready = b.saturating_sub(postfilter::CDEF_ROW_OFFSET).max(start);
        if ready > start {
            self.filter_ready_rows(seq, hdr, fs, info, cur, start, ready, false);
            // Luma rows above 4 * ready are final: deblocking below
            // changes none of them.
            let done = 4 * ready;
            if let Some(job) = self.publish
                && done > self.published
                && job.watched(self.published)
            {
                let tc = std::time::Instant::now();
                let w = self.rows_out.get_or_insert_with(|| {
                    let bufs = lock(self.spares).shared.pop();
                    if bufs.is_none() {
                        prof::count(prof::NEW_BAND);
                    }
                    shared_rows_of(cur, bufs)
                });
                w.write(done, |p, r| {
                    let pl = &cur.planes[p];
                    &pl.data[r * pl.stride..]
                });
                job.publish(&P::wrap_shared(w.shared().clone()), done);
                prof::add(prof::PUBLISH, tc);
                self.published = done;
            }
        }
        self.to_filter = Some((ready, b));
    }

    /// Runs CDEF and loop restoration on luma 4x4 rows `a..b` (to the
    /// bottom of the frame when `last`), whose deblocking and that of the
    /// rows below them that these stages read is done.
    fn filter_ready_rows(
        &mut self,
        seq: &SequenceHeader,
        hdr: &FrameHeader,
        fs: &FrameState,
        info: &postfilter::FilterInfo,
        cur: &mut Picture<P>,
        a: usize,
        b: usize,
        last: bool,
    ) {
        // Stripe k ends above luma row 64 k + 56.
        let off = postfilter::CDEF_ROW_OFFSET;
        let stripes = (a + off) / 16..if last { usize::MAX } else { (b + off) / 16 };
        if self.lr_on {
            postfilter::lr_save_frame_bounds(seq, hdr, cur, &mut self.lr_bounds, stripes.clone());
        }
        if self.cdef_on {
            postfilter::cdef_sb_rows_in_place(seq, hdr, fs, Some(info), cur, a, b, &mut self.cdef);
        }
        if self.lr_on {
            let units = postfilter::lr_units(seq, fs);
            postfilter::loop_restoration_stripes(
                seq,
                hdr,
                fs,
                cur,
                &self.lr_bounds,
                &units,
                self.lr_scratch,
                stripes,
            );
        }
    }
}

/// Decodes all tiles of a frame and runs the in-loop filters. Returns the
/// job of the final picture and the CDFs saved from the context tile.
///
/// With 2 or more threads and a frame the threaded pipeline takes, only
/// parsing happens before the return: reconstruction and the filters run
/// on threads of their own (see [`Backend`]), and the job finishes when
/// they are done. Otherwise the references are waited for first and the
/// job is done on return.
fn decode_frame<P: PicturePixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &mut FrameState,
    tiles: &[(usize, Vec<u8>)],
    cdf: &mut CdfContext,
    ref_jobs: &[Option<Arc<FrameJob>>; 8],
    ref_hdr: &[RefHeaderState; 8],
    cur: Picture<P>,
    job: &Arc<FrameJob>,
    rows: &Arc<Mutex<RowPools>>,
    spares: &Arc<Mutex<Spares<P>>>,
    workers: &Arc<Workers>,
    threads: usize,
    at_once: bool,
    mf: Option<&Arc<MfPlan>>,
    field: Option<Arc<MfShared>>,
    out: &FrameOut<'_>,
) -> R<Box<CdfContext>> {
    let lf_on = hdr.loop_filter.level[0] != 0 || hdr.loop_filter.level[1] != 0;
    let cdef_on = cdef_on(seq, hdr);
    // The tiles are parsed here, on several threads, and their superblock
    // rows are reconstructed and filtered in order on threads of their
    // own, a superblock row behind parsing. Frames without in-loop filters
    // too: the other way lends each of its threads a frame state and a
    // picture of its own, which the pool then keeps.
    if pipelined(hdr, threads, at_once) {
        return decode_frame_async(
            seq, hdr, fs, tiles, cdf, ref_jobs, ref_hdr, cur, job, rows, spares, workers, field,
            lf_on, cdef_on, threads, out,
        );
    }
    let mut held: [Option<Arc<AnyPicture>>; 8] = Default::default();
    let waited = std::time::Instant::now();
    for (h, j) in held.iter_mut().zip(ref_jobs) {
        if let Some(j) = j {
            *h = Some(j.wait()?);
        }
    }
    prof::add(prof::WAIT_REFS, waited);
    let refs: [Option<&Picture<P>>; 8] =
        std::array::from_fn(|i| held[i].as_deref().and_then(P::view));
    let mut lr_scratch = lock(spares).lr.pop().unwrap_or_default();
    let r = decode_frame_sync(
        seq,
        hdr,
        fs,
        tiles,
        cdf,
        refs,
        ref_hdr,
        &mut lr_scratch,
        cur,
        spares,
        threads,
        mf.map(|p| &**p),
        cdef_on,
        out.data,
    );
    lock(spares).lr.push(lr_scratch);
    let (pic, saved) = r?;
    job.finish(Ok(Arc::new(P::wrap(pic))));
    Ok(saved)
}

/// Whether the frame of `hdr` goes through the threaded pipeline
/// ([`decode_frame_async`]): its tiles parsed here and its rows
/// reconstructed and filtered on threads of their own.
fn pipelined(hdr: &FrameHeader, threads: usize, at_once: bool) -> bool {
    at_once && threads > 1 && !hdr.use_superres
}

/// Starts computing the motion field of `plan` band by band on a thread
/// of its own; `None` when it has no bands.
fn start_motion_field(workers: &Arc<Workers>, plan: &Arc<MfPlan>) -> Option<Arc<MfShared>> {
    if plan.bands() == 0 {
        return None;
    }
    let field = Arc::new(MfShared::new(plan));
    let (plan, to) = (plan.clone(), MfClose(field.clone()));
    workers.run(Box::new(move || {
        let t = prof::thread_cpu();
        for (k, b) in to.0.bands.iter().enumerate() {
            let mut band = to.0.blank(k);
            plan.run_band(k, &mut band);
            let _ = b.set(band);
        }
        prof::cpu_since(prof::CPU_MF, t);
    }));
    Some(field)
}

/// Sets the visible size of the planes of a final picture.
fn set_visible_size<P>(seq: &SequenceHeader, hdr: &FrameHeader, pic: &mut Picture<P>) {
    for (p, pl) in pic.planes.iter_mut().enumerate() {
        let (sx, sy) = if p > 0 {
            (
                usize::from(seq.subsampling_x),
                usize::from(seq.subsampling_y),
            )
        } else {
            (0, 0)
        };
        pl.w = (hdr.upscaled_width as usize + sx) >> sx;
        pl.h = (hdr.frame_height as usize + sy) >> sy;
    }
}

/// Starts the threads that reconstruct and filter a frame (a [`Backend`])
/// and parses its tiles on this thread, recording what reconstruction
/// needs and handing it on chunk by chunk.
fn decode_frame_async<P: PicturePixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &mut FrameState,
    tiles: &[(usize, Vec<u8>)],
    cdf: &mut CdfContext,
    ref_jobs: &[Option<Arc<FrameJob>>; 8],
    ref_hdr: &[RefHeaderState; 8],
    cur: Picture<P>,
    job: &Arc<FrameJob>,
    rows: &Arc<Mutex<RowPools>>,
    spares: &Arc<Mutex<Spares<P>>>,
    workers: &Arc<Workers>,
    field: Option<Arc<MfShared>>,
    lf_on: bool,
    cdef_on: bool,
    threads: usize,
    out: &FrameOut<'_>,
) -> R<Box<CdfContext>> {
    // Reconstruction and the filters each keep the mode info of the frame
    // and a picture: the unfiltered one, from the pool, and the final one.
    // The filters run in place behind reconstruction, which then writes
    // the final picture, when no frame waits for this one's rows, or when
    // deblocking is all there is: rows are then final 8 rows behind it
    // (see InlineFilter::deblock). CDEF and loop restoration keep rows
    // from being final for a superblock row more: then they run on a
    // thread of their own, on a copy of the rows, and the rows other
    // frames wait for are final sooner.
    //
    // Such rows are reconstructed each into a picture of its own (see
    // take_sb_row) that then goes to the filters as it is, to be copied
    // once into the final picture; intra prediction reads the last lines
    // of the row above from a copy (Scratch::edge). Intra block copy reads
    // any row above, and superres frames take the whole picture.
    let banded = banded(seq, hdr);
    let in_place = hdr.refresh_frame_flags == 0 || !(cdef_on || hdr.lr.uses_lr);
    // Whatever the picture, reconstruction keeps the mode info of one
    // superblock row and the row above it: its blocks read no other, and
    // the filters take theirs from the log.
    let band_mi = !hdr.allow_intrabc && !hdr.use_superres;
    let (rc_fs, mut ft_fs, lr_scratch, rec, cur) = {
        let mut sp = lock(spares);
        let (rc_fs, mut ft_fs) = take_shells(&mut sp, fs, band_mi.then(|| band_rows(seq)));
        // The filters take each row's data as it comes, in order, and are
        // done with it by the end of the superblock row below: their
        // transform sizes and block info go round a ring that stays in
        // the cache (as on one thread).
        ft_fs.mi_ring = if hdr.use_superres {
            usize::MAX
        } else {
            MI_RING - 1
        };
        let lr = sp.lr.pop().unwrap_or_default();
        let none = || Picture::<P> {
            planes: Vec::new(),
            bd: seq.bit_depth,
            sub_x: seq.subsampling_x,
            sub_y: seq.subsampling_y,
        };
        let (rec, cur) = if banded {
            (take_sb_row(&mut sp, seq, hdr), cur)
        } else if in_place {
            (cur, none())
        } else {
            let rec = take_picture::<P>(
                &mut sp.pics,
                hdr.frame_width as usize,
                hdr.frame_height as usize,
                seq.num_planes(),
                seq.subsampling_x,
                seq.subsampling_y,
                seq.bit_depth,
            );
            (rec, cur)
        };
        (rc_fs, ft_fs, lr, rec, cur)
    };
    // The loop restoration units arrive part by part: all unset first.
    ft_fs.lr.clone_from(&fs.lr);
    let (rtx, rrx) = std::sync::mpsc::channel::<(Option<RowMsg>, ReconLog)>();
    let backend = Backend {
        seq: seq.clone(),
        hdr: hdr.clone(),
        ref_jobs: ref_jobs.clone(),
        ref_hdr: ref_hdr.clone(),
        rec,
        rc_fs,
        filters: FilterStage {
            seq: seq.clone(),
            hdr: hdr.clone(),
            cur,
            fs: ft_fs,
            lr_scratch,
            cdef_on,
            lf_on,
            rows: rows.clone(),
            spares: spares.clone(),
            job: job.clone(),
        },
        workers: workers.clone(),
        spares: spares.clone(),
        job: job.clone(),
        rx: rrx,
        banded,
        band_mi,
        in_place,
    };
    workers.run(Box::new(move || {
        let t = prof::thread_cpu();
        let t0 = std::time::Instant::now();
        backend.run();
        prof::mark(prof::BACKEND, t0);
        prof::cpu_since(prof::CPU_BACKEND, t);
    }));
    // Parsing reads only the presence and size of the references.
    let placeholder = Picture::<P> {
        planes: Vec::new(),
        bd: seq.bit_depth,
        sub_x: seq.subsampling_x,
        sub_y: seq.subsampling_y,
    };
    let parse_refs: [Option<&Picture<P>>; 8] =
        std::array::from_fn(|i| ref_jobs[i].as_ref().map(|_| &placeholder));
    let t_parse = std::time::Instant::now();
    let saved = parse::parse_tiles(
        seq,
        hdr,
        fs,
        tiles,
        cdf,
        &parse_refs,
        ref_hdr,
        rows,
        spares,
        field.as_ref(),
        threads,
        &rtx,
        out,
    );
    // A failed parse fails the job before reconstruction sees the end of
    // its input.
    if let Err(e) = &saved {
        job.finish(Err(*e));
    }
    drop(rtx);
    prof::add(prof::PARSE, t_parse);
    saved
}

/// What a frame's parse gives later frames before it ends.
struct FrameOut<'a> {
    /// The frame's saved motion and segment ids, when it is referenced.
    data: Option<&'a RefData>,
    /// The CDFs of the context tile, when the frame saves them.
    cdf: Option<&'a CdfPromise>,
}

impl FrameOut<'_> {
    /// Hands on the CDFs the context tile ended with.
    fn save_cdf(&self, c: &CdfContext) {
        if let Some(p) = self.cdf {
            let mut c = Box::new(c.clone());
            c.clear_counts();
            p.set(Ok(Arc::from(c)));
            prof::mark(prof::CDF_SET, std::time::Instant::now());
        }
    }
}

/// The reconstruction of one frame, run on a thread of its own while
/// another parses the frame and the next ones, with the frame's in-loop
/// filters on a third (a [`FilterStage`]).
struct Backend<P> {
    seq: SequenceHeader,
    hdr: FrameHeader,
    ref_jobs: [Option<Arc<FrameJob>>; 8],
    ref_hdr: [RefHeaderState; 8],
    /// The picture reconstruction writes, which stays unfiltered (a
    /// superblock row's with `banded`).
    rec: Picture<P>,
    rc_fs: FrameState,
    filters: FilterStage<P>,
    workers: Arc<Workers>,
    spares: Arc<Mutex<Spares<P>>>,
    job: Arc<FrameJob>,
    /// The parsed chunks.
    rx: Receiver<(Option<RowMsg>, ReconLog)>,
    /// Whether `rec` is a picture of a superblock row (see
    /// [`take_sb_row`]).
    banded: bool,
    /// Whether `rc_fs` is a shell of the mode info of a superblock row and
    /// the row above it (see [`TileDec::mi_r0`]).
    band_mi: bool,
    /// Whether the filters run in place in `rec`, the final picture.
    in_place: bool,
}

/// The in-loop filters of one frame, a superblock row behind its
/// reconstruction: the rows arrive as reconstruction completes them and
/// are filtered in the final picture, which reconstruction never reads.
struct FilterStage<P> {
    seq: SequenceHeader,
    hdr: FrameHeader,
    /// The final picture.
    cur: Picture<P>,
    fs: FrameState,
    lr_scratch: LrScratch,
    cdef_on: bool,
    lf_on: bool,
    rows: Arc<Mutex<RowPools>>,
    spares: Arc<Mutex<Spares<P>>>,
    job: Arc<FrameJob>,
}

/// What a [`FilterStage`] gives back when its rows end.
type FilterEnd<P> = (Picture<P>, FrameState, LrScratch);

impl<P: PicturePixel> FilterStage<P> {
    /// Filters the rows arriving on `rx` until it closes.
    fn run(self, rx: Receiver<Filtered<P>>) -> FilterEnd<P> {
        let Self {
            seq,
            hdr,
            mut cur,
            mut fs,
            mut lr_scratch,
            cdef_on,
            lf_on,
            rows,
            spares,
            job,
        } = self;
        let (cdef, lr_bounds, mut info) = lock(&spares).filters.pop().unwrap_or_else(|| {
            (
                postfilter::cdef_pads(),
                postfilter::LrBounds::default(),
                postfilter::FilterInfo::default(),
            )
        });
        // What the filters read of the blocks, stored from their logs.
        info.size_for(&fs);
        let levels = postfilter::LfLevels::new(&hdr);
        let size = (fs.mi_rows, fs.mi_cols);
        let mut filt = InlineFilter {
            cdef_on,
            lr_on: hdr.lr.uses_lr,
            lr_scratch: &mut lr_scratch,
            cdef,
            lr_bounds,
            lf_on,
            publish: Some(&job),
            spares: &spares,
            to_filter: None,
            published: 0,
            rows_out: None,
        };
        let mut back: (Vec<ReconLog>, Vec<RowMsg>) = Default::default();
        for (mut log, m, samples) in rx {
            for rec in &log.blocks {
                let b = &rec.b;
                let at = (b.mi_row as usize, b.mi_col as usize);
                let of = levels.info(
                    &hdr,
                    b.ref_frame[0],
                    b.segment_id,
                    b.y_mode,
                    b.skip,
                    b.bsize,
                    b.delta_lf,
                );
                info.store(at, size, of);
            }
            log.clear();
            if let Some(m) = m {
                match samples {
                    // One tile's part of the row: wait for the rest.
                    None => m.apply_meta(&mut fs),
                    Some(Samples::Row(pic)) => {
                        let tf = std::time::Instant::now();
                        // Deblocking copies the rows in as it goes.
                        filt.row(&seq, &hdr, &mut fs, &info, &mut cur, &m, Some(&pic));
                        give_sb_row(&mut lock(&spares), pic);
                        prof::add(prof::FILTER, tf);
                    }
                    Some(Samples::Rows(bufs)) => {
                        let tf = std::time::Instant::now();
                        copy_rows_in(&mut cur, m.r0, &bufs);
                        {
                            let mut sp = lock(&spares);
                            if sp.bands.len() < MAX_BANDS {
                                sp.bands.push(bufs);
                            }
                        }
                        filt.row(&seq, &hdr, &mut fs, &info, &mut cur, &m, None);
                        prof::add(prof::FILTER, tf);
                    }
                }
                back.1.push(m);
            }
            back.0.push(log);
            // The pool is shared by every frame: give back in batches.
            if back.0.len() >= 32 {
                let mut pool = lock(&rows);
                pool.logs.append(&mut back.0);
                pool.msgs.append(&mut back.1);
            }
        }
        {
            let mut pool = lock(&rows);
            pool.logs.append(&mut back.0);
            pool.msgs.append(&mut back.1);
        }
        filt.release_rows();
        let filter_rows = (filt.cdef, filt.lr_bounds, info);
        lock(&spares).filters.push(filter_rows);
        (cur, fs, lr_scratch)
    }
}

impl<P: PicturePixel> Backend<P> {
    fn run(self) {
        let Self {
            seq,
            hdr,
            ref_jobs,
            ref_hdr,
            mut rec,
            mut rc_fs,
            filters,
            workers,
            spares,
            job,
            rx,
            banded,
            band_mi,
            in_place: filter_here,
        } = self;
        let _guard = FinishGuard(job.clone());
        // The filters run here, in place (see decode_frame_async), or take
        // the rows on a thread of their own and give the final picture
        // back when the rows end.
        let (ftx, frx) = std::sync::mpsc::channel::<Filtered<P>>();
        let (dtx, drx) = std::sync::mpsc::channel::<FilterEnd<P>>();
        // The final picture, when the filters ran here.
        let mut filtered_here = None;
        let parked = if filter_here {
            Some(filters)
        } else {
            workers.run(Box::new(move || {
                let _ = dtx.send(filters.run(frx));
            }));
            None
        };
        // References already decoded are read in place; those still
        // decoding through the bands they publish (one reader per job).
        let mut held: [Option<Arc<AnyPicture>>; 8] = Default::default();
        let mut fills: Vec<RefFill> = Vec::new();
        let mut mirror_of = [None; 8];
        let mut mirrors = Vec::new();
        for i in 1..8 {
            let Some(j) = &ref_jobs[i] else {
                continue;
            };
            match j.poll_or_watch() {
                Some(Ok(p)) => held[i] = Some(p),
                Some(Err(e)) => return job.finish(Err(e)),
                None => {
                    if let Some(f) = fills.iter().find(|f| Arc::ptr_eq(&f.job, j)) {
                        mirror_of[i] = Some(f.mirror);
                        continue;
                    }
                    mirror_of[i] = Some(mirrors.len());
                    fills.push(RefFill {
                        job: j.clone(),
                        ref_frame: i,
                        mirror: mirrors.len(),
                        have: 0,
                        whole: false,
                    });
                    mirrors.push(Mirror::default());
                }
            }
        }
        drop(ref_jobs);
        // The pictures of the references read through a copy, once done.
        let finals: Vec<OnceLock<Arc<AnyPicture>>> =
            fills.iter().map(|_| OnceLock::new()).collect();
        let placeholder = Picture::<P> {
            planes: Vec::new(),
            bd: seq.bit_depth,
            sub_x: seq.subsampling_x,
            sub_y: seq.subsampling_y,
        };
        let res = {
            let refs: [Option<RefView<'_, P>>; 8] = std::array::from_fn(|i| {
                if i == 0 {
                    return None;
                }
                let pic = match mirror_of[i] {
                    Some(_) => &placeholder,
                    None => held[i].as_deref().and_then(P::view)?,
                };
                let slot = hdr.ref_frame_idx[i - 1] as usize;
                Some(RefView {
                    pic,
                    mirror: mirror_of[i],
                    upscaled_width: ref_hdr[slot].upscaled_width as usize,
                    frame_height: ref_hdr[slot].frame_height as usize,
                    frame_width: ref_hdr[slot].frame_width as usize,
                })
            });
            let scratch = lock(&spares).scratch.pop().unwrap_or_else(Scratch::new);
            match parked {
                None => recon_worker(
                    &seq, &hdr, &mut rc_fs, &mut rec, refs, mirrors, &finals, scratch, fills, rx,
                    &ftx, None, &spares, banded, band_mi,
                ),
                Some(stage) => {
                    let FilterStage {
                        fs: ft_fs,
                        mut lr_scratch,
                        cdef_on,
                        lf_on,
                        rows,
                        ..
                    } = stage;
                    let (cdef, lr_bounds, mut info) =
                        lock(&spares).filters.pop().unwrap_or_else(|| {
                            (
                                postfilter::cdef_pads(),
                                postfilter::LrBounds::default(),
                                postfilter::FilterInfo::default(),
                            )
                        });
                    info.size_for(&ft_fs);
                    let mut ip = InPlace {
                        filt: InlineFilter {
                            cdef_on,
                            lr_on: hdr.lr.uses_lr,
                            lr_scratch: &mut lr_scratch,
                            cdef,
                            lr_bounds,
                            lf_on,
                            publish: Some(&job),
                            spares: &spares,
                            to_filter: None,
                            published: 0,
                            rows_out: None,
                        },
                        fs: ft_fs,
                        info,
                        levels: postfilter::LfLevels::new(&hdr),
                        rows: &rows,
                        back: Default::default(),
                    };
                    let r = recon_worker(
                        &seq,
                        &hdr,
                        &mut rc_fs,
                        &mut rec,
                        refs,
                        mirrors,
                        &finals,
                        scratch,
                        fills,
                        rx,
                        &ftx,
                        Some(&mut ip),
                        &spares,
                        false,
                        band_mi,
                    );
                    ip.give_back();
                    let InPlace {
                        filt,
                        fs: ft_fs,
                        info,
                        ..
                    } = ip;
                    let InlineFilter {
                        cdef, lr_bounds, ..
                    } = filt;
                    let mut sp = lock(&spares);
                    sp.filters.push((cdef, lr_bounds, info));
                    sp.filter_shells.push(ft_fs);
                    sp.lr.push(lr_scratch);
                    // The picture reconstruction wrote is the final one.
                    let blank = Picture::<P> {
                        planes: Vec::new(),
                        bd: seq.bit_depth,
                        sub_x: seq.subsampling_x,
                        sub_y: seq.subsampling_y,
                    };
                    filtered_here = Some(std::mem::replace(&mut rec, blank));
                    r
                }
            }
        };
        drop(held);
        // The filters end once they have every row sent.
        drop(ftx);
        let filtered = match filtered_here {
            Some(fin) => Ok(fin),
            None => drx.recv().map(|(fin, ft_fs, lr_scratch)| {
                let mut sp = lock(&spares);
                sp.filter_shells.push(ft_fs);
                sp.lr.push(lr_scratch);
                fin
            }),
        };
        let res = res.map(|(mirrors, scratch)| {
            for mut m in mirrors {
                m.release(&spares);
            }
            lock(&spares).scratch.push(scratch);
        });
        {
            let mut sp = lock(&spares);
            sp.shells.push(rc_fs);
            if banded {
                give_sb_row(&mut sp, rec);
            } else if !rec.planes.is_empty() {
                recycle(&mut sp.pics, rec);
            }
        }
        let Ok(mut fin) = filtered else {
            return job.finish(Err(malformed("filter thread")));
        };
        set_visible_size(&seq, &hdr, &mut fin);
        let shared = job.finish_take(res.map(|()| Arc::new(P::wrap(fin))));
        // The buffers of the final rows no other thread holds go back.
        if let Some(sh) = shared.as_ref().and_then(P::view_shared) {
            lock(&spares).give_shared(sh.clone());
        }
    }
}

/// The mode-info rows a single thread decodes a frame in (see
/// [`FrameState::mi_ring`]): two rows of 128x128 superblocks, so the row
/// above one keeps its last rows while that one is decoded and filtered.
const MI_RING: usize = 64;

/// [`decode_frame`] with the references at hand: parsing and
/// reconstruction on this thread, then the in-loop filters (on `threads`).
fn decode_frame_sync<P: PicturePixel>(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    fs: &mut FrameState,
    tiles: &[(usize, Vec<u8>)],
    cdf: &mut CdfContext,
    refs: [Option<&Picture<P>>; 8],
    ref_hdr: &[RefHeaderState; 8],
    lr_scratch: &mut LrScratch,
    mut cur: Picture<P>,
    spares: &Mutex<Spares<P>>,
    threads: usize,
    mf: Option<&MfPlan>,
    cdef_on: bool,
    data: Option<&RefData>,
) -> R<(Picture<P>, Box<CdfContext>)> {
    if let Some(p) = mf {
        fs.motion_field = p.run(threads);
    }
    let (mut cdef_sc, mut bounds, mut info) = lock(spares).filters.pop().unwrap_or_else(|| {
        (
            postfilter::cdef_pads(),
            postfilter::LrBounds::default(),
            postfilter::FilterInfo::default(),
        )
    });
    // On one thread, each superblock row is filtered as soon as it is
    // decoded, while its samples are still in the cache.
    let lf_on = hdr.loop_filter.level[0] != 0 || hdr.loop_filter.level[1] != 0;
    if threads <= 1 && !hdr.use_superres && (lf_on || cdef_on || hdr.lr.uses_lr) {
        // Everything that reads the mode info of a row does so by the end
        // of the superblock row below it: the rows go round a ring that
        // stays in the cache rather than through the frame's.
        fs.mi_ring = MI_RING - 1;
        info.size_for(fs);
        let levels = postfilter::LfLevels::new(hdr);
        let mut filt = InlineFilter {
            cdef_on,
            lr_on: hdr.lr.uses_lr,
            lr_scratch,
            cdef: cdef_sc,
            lr_bounds: bounds,
            lf_on,
            publish: None,
            spares,
            to_filter: None,
            published: 0,
            rows_out: None,
        };
        let res = decode_tiles_by_rows(
            seq,
            hdr,
            fs,
            tiles,
            cdf,
            &refs,
            ref_hdr,
            &mut cur,
            spares,
            &mut |fs: &FrameState, cur: &mut Picture<P>, a: usize, b: usize, last: bool| {
                // The row's motion, before the ring reuses its mode info.
                if let Some(rd) = data {
                    save_motion(seq, hdr, fs, rd, a, b);
                }
                // Deblocking reads the blocks' levels, CDEF their skips.
                if lf_on || cdef_on {
                    info.fill_rows(hdr, &levels, fs, a..b);
                }
                // Intra prediction in the row below reads this row's last
                // lines as they were before the filters, from a copy.
                filt.deblock(seq, hdr, fs, &info, cur, a, b, last, None);
            },
        );
        filt.release_rows();
        cdef_sc = filt.cdef;
        bounds = filt.lr_bounds;
        if res.is_none() {
            fs.mi_ring = usize::MAX;
        }
        if let Some(res) = res {
            lock(spares).filters.push((cdef_sc, bounds, info));
            let saved = res?;
            set_visible_size(seq, hdr, &mut cur);
            return Ok((cur, saved));
        }
    }
    let saved = {
        let tiles_span = prof::Span(prof::TILES, std::time::Instant::now());
        // Intra block copy reads the frame being decoded: its tiles stay on
        // this thread.
        let saved = if threads > 1 && tiles.len() > 1 && !hdr.allow_intrabc {
            tiles::decode_tiles_threaded(
                seq, hdr, fs, tiles, cdf, &refs, ref_hdr, &mut cur, spares, threads,
            )?
        } else {
            decode_tiles(
                seq,
                hdr,
                fs,
                tiles,
                cdf,
                &refs,
                ref_hdr,
                &mut cur,
                None,
                None,
                &mut |_: &FrameState,
                      _: &Picture<P>,
                      _: Option<&mut ReconLog>,
                      _: usize,
                      _: usize,
                      _: bool| {},
                Some(spares),
            )?
        };
        drop(tiles_span);
        // decode_frame_wrapup: in-loop filters, CDEF and loop restoration
        // in place.
        let _filter_span = prof::Span(prof::FILTER, std::time::Instant::now());
        postfilter::loop_filter_scratch(seq, hdr, fs, &mut cur, threads, &mut info);
        saved
    };
    let _filter_span = prof::Span(prof::FILTER, std::time::Instant::now());
    let lr = hdr.lr.uses_lr;

    let mut final_pic = if hdr.use_superres {
        if lr {
            let d = postfilter::superres(seq, hdr, &cur);
            postfilter::lr_save_frame_bounds(seq, hdr, &d, &mut bounds, 0..usize::MAX);
        }
        if cdef_on {
            postfilter::cdef_in_place(seq, hdr, fs, &mut cur, threads, &mut cdef_sc);
        }
        let up = postfilter::superres(seq, hdr, &cur);
        recycle(&mut lock(spares).pics, cur);
        up
    } else {
        if lr {
            postfilter::lr_save_frame_bounds(seq, hdr, &cur, &mut bounds, 0..usize::MAX);
        }
        if cdef_on {
            postfilter::cdef_in_place(seq, hdr, fs, &mut cur, threads, &mut cdef_sc);
        }
        cur
    };
    if lr {
        postfilter::loop_restoration_in_place(
            seq,
            hdr,
            fs,
            &mut final_pic,
            &bounds,
            lr_scratch,
            threads,
        );
    }
    lock(spares).filters.push((cdef_sc, bounds, info));
    set_visible_size(seq, hdr, &mut final_pic);
    Ok((final_pic, saved))
}
