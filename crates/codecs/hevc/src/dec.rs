//! The decoder: parameter sets, picture order counts, reference picture
//! sets and lists (clause 8.3), the decoded picture buffer and output order
//! (clause C.5.2), the jobs that decode pictures, and the engine's decoder
//! interfaces.
//!
//! The caller's thread reads every header, keeps the picture buffer and
//! starts a job for each picture. With more than one thread the jobs run on
//! worker threads, several pictures at once: a picture waits only for the
//! rows of its references it reads, and those references publish their
//! rows as they become final. A picture comes out of a later `push` once
//! its job and every job before it in output order have ended.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, PoisonError};

use cutlass_decode::{
    CodecError, CodecId, CodecParams, DecodeFactory, Decoder, Output, Packet, PictureInfo,
    PictureOutput, PixelFormat, SequenceTiming, VideoDecoder, VideoOutput, Yuv420High, Yuv420p,
    Yuv420p10,
};

use crate::frame::{Frame, OutSamples, Spares};
use crate::job::{Job, Leftovers, OutForm, SliceData};
use crate::nal::{self, Nal};
use crate::picture::{Picture, RefPic, SliceCtx, Work, lock};
use crate::pixel::{AnyRows, Pixel};
use crate::ps::{self, Layout, Pps, Sps};
use crate::slice::{self, SliceHeader, SliceType};

const MAX_SPS: usize = 16;
const MAX_PPS: usize = 64;
/// Most threads a decoder uses.
const MAX_THREADS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Marking {
    Unused,
    Short,
    Long,
}

/// What an output picture carries besides its samples.
#[derive(Clone, Debug)]
struct Meta {
    pts: i64,
    duration: i64,
    timescale: u32,
    keyframe: bool,
    sps: Arc<Sps>,
}

#[derive(Clone)]
struct DpbPic {
    frame: Arc<Frame>,
    poc: i32,
    marking: Marking,
    /// Waiting for output.
    output: bool,
    uid: u64,
    meta: Option<Arc<Meta>>,
}

/// Everything the headers of a push may change. A failed push leaves the
/// last good copy.
#[derive(Clone)]
struct State {
    sps: Vec<Option<Arc<Sps>>>,
    pps: Vec<Option<Arc<Pps>>>,
    dpb: Vec<DpbPic>,
    /// `PicOrderCntVal` of the previous `TemporalId` 0 picture.
    prev_tid0_poc: i32,
    /// An IRAP picture has started a coded video sequence.
    started: bool,
    /// `NoRaslOutputFlag` of the last IRAP picture.
    no_rasl_output: bool,
    /// An end of sequence came before the next picture.
    after_eos: bool,
    next_uid: u64,
    /// The layout of the last parameter sets a picture used.
    layout: Option<(Arc<Sps>, Arc<Pps>, Arc<Layout>)>,
}

/// A picture chosen for output, in output order.
type Bumped = (Arc<Frame>, Arc<Meta>);

impl State {
    fn new() -> Self {
        Self {
            sps: vec![None; MAX_SPS],
            pps: vec![None; MAX_PPS],
            dpb: Vec::new(),
            prev_tid0_poc: 0,
            started: false,
            no_rasl_output: true,
            after_eos: false,
            next_uid: 1,
            layout: None,
        }
    }

    fn uid(&mut self) -> u64 {
        let uid = self.next_uid;
        self.next_uid += 1;
        uid
    }

    fn store_param(&mut self, nal: &Nal) -> Result<(), CodecError> {
        match nal.kind {
            nal::SPS => {
                let sps = ps::parse_sps(&nal.rbsp)?;
                let id = usize::from(sps.id);
                self.sps[id] = Some(Arc::new(sps));
            }
            nal::PPS => {
                let pps = ps::parse_pps(&nal.rbsp)?;
                let id = usize::from(pps.id);
                self.pps[id] = Some(Arc::new(pps));
            }
            _ => {}
        }
        Ok(())
    }

    fn sets(&self, pps_id: u8) -> Option<(Arc<Sps>, Arc<Pps>)> {
        let pps = self.pps.get(usize::from(pps_id))?.clone()?;
        let sps = self.sps.get(usize::from(pps.sps_id))?.clone()?;
        Some((sps, pps))
    }

    /// The bumping process: output the picture with the smallest order
    /// count while too many wait (clause C.5.2.4, as the reference decoder
    /// counts them).
    fn bump(&mut self, max_output: usize, max_dpb: usize, out: &mut Vec<Bumped>) {
        loop {
            let waiting = self.dpb.iter().filter(|p| p.output).count();
            if waiting == 0 || (waiting <= max_output && self.dpb.len() <= max_dpb) {
                break;
            }
            let Some(index) = self
                .dpb
                .iter()
                .enumerate()
                .filter(|(_, p)| p.output)
                .min_by_key(|(_, p)| p.poc)
                .map(|(i, _)| i)
            else {
                break;
            };
            self.dpb[index].output = false;
            if let Some(meta) = &self.dpb[index].meta {
                out.push((self.dpb[index].frame.clone(), meta.clone()));
            }
            self.dpb
                .retain(|p| p.output || p.marking != Marking::Unused);
        }
    }

    fn flush(&mut self, out: &mut Vec<Bumped>) {
        self.bump(0, usize::MAX, out);
        self.dpb
            .retain(|p| p.output || p.marking != Marking::Unused);
    }
}

/// The picture whose slice segments are being read.
struct Pending {
    pic: Picture,
    sps: Arc<Sps>,
    output: bool,
    meta: Arc<Meta>,
    /// The last independent slice segment header, for dependent ones.
    independent: Option<SliceHeader>,
    slice_addr: u32,
    st_before: Vec<RefPic>,
    st_after: Vec<RefPic>,
    lt_curr: Vec<RefPic>,
    uid: u64,
    slices: Vec<SliceData>,
}

/// A picture whose headers are read, to decode.
enum Ready {
    Eight(Job<u8>),
    Deep(Job<u16>),
}

impl Ready {
    fn frame(&self) -> &Arc<Frame> {
        match self {
            Self::Eight(j) => &j.frame,
            Self::Deep(j) => &j.frame,
        }
    }
}

/// Buffers of finished jobs for the next ones.
#[derive(Default)]
struct Shared {
    frames: Arc<Spares>,
    pics: Mutex<Vec<Picture>>,
    work8: Mutex<Vec<Work<u8>>>,
    work16: Mutex<Vec<Work<u16>>>,
}

impl Shared {
    fn give8(&self, left: Leftovers<u8>) {
        put(&self.pics, left.pic);
        put(&self.work8, left.work);
    }

    fn give16(&self, left: Leftovers<u16>) {
        put(&self.pics, left.pic);
        put(&self.work16, left.work);
    }
}

fn put<T>(spares: &Mutex<Vec<T>>, item: T) {
    let mut s = lock(spares);
    if s.len() < MAX_THREADS + 2 {
        s.push(item);
    }
}

/// Threads the decode jobs run on, kept between pictures. A job goes to an
/// idle thread, or to a new one when none is idle, so jobs that wait for
/// each other never wait for a thread.
#[derive(Default)]
struct Workers {
    queue: Mutex<WorkQueue>,
    ready: Condvar,
}

#[derive(Default)]
struct WorkQueue {
    jobs: VecDeque<Box<dyn FnOnce() + Send>>,
    idle: usize,
    closed: bool,
}

impl Workers {
    fn run(self: &Arc<Self>, job: Box<dyn FnOnce() + Send>) {
        let mut q = lock(&self.queue);
        q.jobs.push_back(job);
        if q.idle >= q.jobs.len() {
            drop(q);
            self.ready.notify_one();
            return;
        }
        drop(q);
        let workers = self.clone();
        let spawned = std::thread::Builder::new()
            .name(String::from("cutlass-hevc"))
            .spawn(move || workers.serve());
        if spawned.is_err() {
            // No thread to run it: run what is queued here.
            while let Some(job) = lock(&self.queue).jobs.pop_front() {
                job();
            }
        }
    }

    fn serve(&self) {
        loop {
            let job = {
                let mut q = lock(&self.queue);
                loop {
                    if let Some(job) = q.jobs.pop_front() {
                        break job;
                    }
                    if q.closed {
                        return;
                    }
                    q.idle += 1;
                    q = self.ready.wait(q).unwrap_or_else(PoisonError::into_inner);
                    q.idle -= 1;
                }
            };
            job();
        }
    }

    fn close(&self) {
        lock(&self.queue).closed = true;
        self.ready.notify_all();
    }
}

/// Native HEVC (H.265) decoder: Main, Main 10 and Main 12, 4:2:0.
///
/// With more than one thread, several pictures decode at once on worker
/// threads, and the CTB rows of a wavefront slice segment on more. The
/// output does not depend on the number of threads.
pub struct HevcDecoder {
    length_size: u8,
    extra: Vec<u8>,
    state: State,
    drained: bool,
    /// Pictures decoding at once (1: each on the caller's thread).
    threads: usize,
    /// Threads for the CTB rows of a wavefront slice segment.
    row_threads: usize,
    form: OutForm,
    shared: Arc<Shared>,
    workers: Arc<Workers>,
    /// Pictures whose jobs were started, oldest first, until they end.
    running: VecDeque<Arc<Frame>>,
    /// Pictures chosen for output whose samples are not handed out yet.
    queue: VecDeque<Bumped>,
}

impl HevcDecoder {
    /// A decoder with as many threads as the machine runs at once, at most 16.
    pub fn open(params: &CodecParams) -> Result<Self, CodecError> {
        let threads = std::thread::available_parallelism().map_or(1, usize::from);
        Self::open_with_threads(params, threads)
    }

    /// A decoder with `threads` threads, at most 16. Zero or one decodes on
    /// the caller.
    pub fn open_with_threads(params: &CodecParams, threads: usize) -> Result<Self, CodecError> {
        if params.codec != CodecId::Hevc {
            return Err(CodecError::Unsupported { what: "codec" });
        }
        let threads = threads.clamp(1, MAX_THREADS);
        let mut dec = Self {
            length_size: params.nal_length_size,
            extra: params.extra.clone(),
            state: State::new(),
            drained: false,
            threads,
            row_threads: threads,
            form: OutForm::Planes,
            shared: Arc::new(Shared::default()),
            workers: Arc::new(Workers::default()),
            running: VecDeque::new(),
            queue: VecDeque::new(),
        };
        dec.apply_extra()?;
        if let Some(v) = std::env::var("HEVC_FRAMES")
            .ok()
            .and_then(|v| v.parse().ok())
        {
            dec.threads = v;
        }
        if let Some(v) = std::env::var("HEVC_ROWS").ok().and_then(|v| v.parse().ok()) {
            dec.row_threads = v;
        }
        Ok(dec)
    }

    /// Pictures decoding at once, and threads for the rows of a wavefront
    /// slice segment (each at least 1).
    pub fn set_threads(&mut self, frames: usize, rows: usize) {
        self.threads = frames.clamp(1, MAX_THREADS);
        self.row_threads = rows.clamp(1, MAX_THREADS);
    }

    fn apply_extra(&mut self) -> Result<(), CodecError> {
        if self.extra.is_empty() {
            return Ok(());
        }
        let nals = if nal::is_hvcc(&self.extra) {
            let (size, nals) = nal::hvcc(&self.extra)?;
            self.length_size = size;
            nals
        } else {
            nal::access_unit(&self.extra, 0)?
        };
        for nal in &nals {
            self.state.store_param(nal)?;
        }
        Ok(())
    }

    /// Read the headers of one access unit: the pictures to decode and
    /// the pictures chosen for output.
    fn parse_unit(
        &self,
        state: &mut State,
        packet: &Packet,
    ) -> Result<(Vec<Ready>, Vec<Bumped>), CodecError> {
        let nals = nal::access_unit(&packet.data, self.length_size)?;
        let mut ready = Vec::new();
        let mut out = Vec::new();
        let mut current: Option<Pending> = None;
        let mut skipping = false;
        for nal in nals {
            if nal.layer != 0 {
                continue;
            }
            match nal.kind {
                nal::SPS | nal::PPS => state.store_param(&nal)?,
                nal::EOS | nal::EOB => {
                    if let Some(done) = current.take() {
                        ready.push(self.finish(state, done, &mut out));
                    }
                    state.after_eos = true;
                }
                _ if nal.is_slice() => {
                    let first = nal.rbsp.first().is_some_and(|b| b & 0x80 != 0);
                    if first {
                        if let Some(done) = current.take() {
                            ready.push(self.finish(state, done, &mut out));
                        }
                        let header = parse(state, &nal, None)?;
                        current = self.start_picture(state, &nal, &header, packet, &mut out)?;
                        skipping = current.is_none();
                        if let Some(pending) = current.as_mut() {
                            add_slice(pending, Arc::new(nal), header)?;
                        }
                    } else if !skipping && let Some(pending) = current.as_mut() {
                        let header = parse(state, &nal, pending.independent.as_ref())?;
                        add_slice(pending, Arc::new(nal), header)?;
                    }
                }
                _ => {}
            }
        }
        if let Some(done) = current.take() {
            ready.push(self.finish(state, done, &mut out));
        }
        Ok((ready, out))
    }

    /// Start a picture at its first slice segment: order count, reference
    /// picture set, missing references, and the output of prior pictures.
    /// `None` skips the picture.
    fn start_picture(
        &self,
        state: &mut State,
        nal: &Nal,
        header: &SliceHeader,
        packet: &Packet,
        out: &mut Vec<Bumped>,
    ) -> Result<Option<Pending>, CodecError> {
        let Some((sps, pps)) = state.sets(header.pps_id) else {
            return Err(CodecError::Malformed {
                at_bit: 0,
                what: "slice references a missing parameter set",
            });
        };
        let irap = nal.is_irap();
        if irap {
            state.no_rasl_output =
                nal.is_idr() || nal.is_bla() || !state.started || state.after_eos;
            state.started = true;
        } else if !state.started {
            return Ok(None);
        }
        if nal.is_rasl() && state.no_rasl_output {
            return Ok(None);
        }
        let new_sequence = irap && state.no_rasl_output;
        let flushed_by_eos = state.after_eos;
        state.after_eos = false;

        // Picture order count (clause 8.3.1).
        let max_lsb = 1i32 << sps.log2_max_poc_lsb;
        let lsb = header.poc_lsb as i32;
        let msb = if new_sequence {
            0
        } else {
            let prev_lsb = state.prev_tid0_poc & (max_lsb - 1);
            let prev_msb = state.prev_tid0_poc - prev_lsb;
            if lsb < prev_lsb && prev_lsb - lsb >= max_lsb / 2 {
                prev_msb + max_lsb
            } else if lsb > prev_lsb && lsb - prev_lsb > max_lsb / 2 {
                prev_msb - max_lsb
            } else {
                prev_msb
            }
        };
        let poc = msb + lsb;
        if nal.temporal_id == 0 && !nal.is_rasl() && !nal.is_radl() && !nal.is_sub_layer_non_ref() {
            state.prev_tid0_poc = poc;
        }

        // Output of prior pictures at a new sequence (clause C.5.2.2).
        if new_sequence {
            let discard =
                header.no_output_of_prior_pics || (nal.kind == nal::CRA && flushed_by_eos);
            if discard {
                state.dpb.clear();
            } else {
                for p in &mut state.dpb {
                    p.marking = Marking::Unused;
                }
                state.flush(out);
            }
            state.dpb.clear();
        }

        // Reference picture set (clause 8.3.2).
        let mut keep = vec![Marking::Unused; state.dpb.len()];
        let mut lt_curr = Vec::new();
        for lt in &header.long_term {
            let (value, exact) = if lt.msb {
                (poc + lt.poc, true)
            } else {
                (lt.poc, false)
            };
            let found = state.dpb.iter().position(|p| {
                p.marking != Marking::Unused
                    && if exact {
                        p.poc == value
                    } else {
                        p.poc & (max_lsb - 1) == value
                    }
            });
            if let Some(i) = found {
                keep[i] = Marking::Long;
            }
            if lt.used {
                lt_curr.push((value, found));
            }
        }
        let mut st_lists = [Vec::new(), Vec::new()];
        let rps = &header.st_rps;
        let deltas = [
            (&rps.negative, &rps.negative_used, 0usize),
            (&rps.positive, &rps.positive_used, 1usize),
        ];
        for (values, used, list) in deltas {
            for (&delta, &used) in values.iter().zip(used.iter()) {
                let target = poc + delta;
                let found = state.dpb.iter().enumerate().position(|(i, p)| {
                    p.marking == Marking::Short && keep[i] != Marking::Long && p.poc == target
                });
                if let Some(i) = found {
                    keep[i] = Marking::Short;
                }
                if used {
                    st_lists[list].push((target, found));
                }
            }
        }
        for (p, mark) in state.dpb.iter_mut().zip(&keep) {
            p.marking = *mark;
        }
        // Generate what is missing (clause 8.3.3), mid-grey and not output.
        let (width, height) = (sps.width as usize, sps.height as usize);
        let resolve =
            |state: &mut State, wanted: Vec<(i32, Option<usize>)>, long: bool| -> Vec<RefPic> {
                wanted
                    .into_iter()
                    .map(|(value, found)| {
                        let index = found.unwrap_or_else(|| {
                            let uid = state.uid();
                            state.dpb.push(DpbPic {
                                frame: Frame::missing(width, height, sps.bit_depth, value),
                                poc: value,
                                marking: if long { Marking::Long } else { Marking::Short },
                                output: false,
                                uid,
                                meta: None,
                            });
                            state.dpb.len() - 1
                        });
                        let p = &state.dpb[index];
                        RefPic {
                            frame: p.frame.clone(),
                            poc: p.poc,
                            long_term: long,
                            uid: p.uid,
                        }
                    })
                    .collect()
            };
        let [before, after] = st_lists;
        let st_before = resolve(state, before, false);
        let st_after = resolve(state, after, false);
        let lt_curr = resolve(state, lt_curr, true);
        state
            .dpb
            .retain(|p| p.output || p.marking != Marking::Unused);

        let layout = match &state.layout {
            Some((s, p, l)) if Arc::ptr_eq(s, &sps) && Arc::ptr_eq(p, &pps) => l.clone(),
            _ => {
                let l = Arc::new(Layout::new(&sps, &pps)?);
                state.layout = Some((sps.clone(), pps.clone(), l.clone()));
                l
            }
        };
        let old = lock(&self.shared.pics).pop();
        let pic = Picture::reuse(old, sps.clone(), pps.clone(), layout, poc)?;
        let output = header.pic_output && !(nal.is_rasl() && state.no_rasl_output);
        let uid = state.uid();
        Ok(Some(Pending {
            pic,
            sps: sps.clone(),
            output,
            meta: Arc::new(Meta {
                pts: packet.pts,
                duration: packet.duration,
                timescale: packet.timescale,
                keyframe: irap,
                sps,
            }),
            independent: None,
            slice_addr: 0,
            st_before,
            st_after,
            lt_curr,
            uid,
            slices: Vec::new(),
        }))
    }

    /// The picture's slice segments are all read: its frame enters the
    /// picture buffer (clause C.5.2.3) and its job is made.
    fn finish(&self, state: &mut State, pending: Pending, out: &mut Vec<Bumped>) -> Ready {
        let Pending {
            pic,
            sps,
            output,
            meta,
            uid,
            slices,
            ..
        } = pending;
        let (width, height) = (pic.width, pic.height);
        let poc = pic.poc;
        let log2_ctb = u32::from(pic.layout.log2_ctb);
        let form = output.then_some(self.form);
        let row_threads = self.row_threads;
        let ready = if sps.bit_depth == 8 {
            Ready::Eight(make_job(
                &self.shared,
                &self.shared.work8,
                &self.shared.frames.eight,
                AnyRows::Eight,
                (pic, slices, form, row_threads),
                (width, height, log2_ctb, poc),
            ))
        } else {
            Ready::Deep(make_job(
                &self.shared,
                &self.shared.work16,
                &self.shared.frames.deep,
                AnyRows::Deep,
                (pic, slices, form, row_threads),
                (width, height, log2_ctb, poc),
            ))
        };
        state.dpb.push(DpbPic {
            frame: ready.frame().clone(),
            poc,
            marking: Marking::Short,
            output,
            uid,
            meta: Some(meta),
        });
        let top = usize::from(sps.max_sub_layers.saturating_sub(1)).min(7);
        let reorder = usize::from(sps.max_num_reorder[top]);
        let max_dpb = usize::from(sps.max_dec_pic_buffering[top]);
        state.bump(reorder, max_dpb, out);
        ready
    }

    /// Start the decode of a picture, on a worker or (one thread) here.
    fn start(&mut self, ready: Ready) {
        // Keep at most `threads` pictures decoding.
        while self.running.len() >= self.threads {
            if let Some(oldest) = self.running.pop_front() {
                oldest.wait_finished();
            }
        }
        self.running.retain(|f| !f.is_done());
        let shared = self.shared.clone();
        let frame = ready.frame().clone();
        let job: Box<dyn FnOnce() + Send> = match ready {
            Ready::Eight(job) => Box::new(move || {
                if let Some(left) = guarded(job) {
                    shared.give8(left);
                }
            }),
            Ready::Deep(job) => Box::new(move || {
                if let Some(left) = guarded(job) {
                    shared.give16(left);
                }
            }),
        };
        if self.threads <= 1 {
            job();
        } else {
            self.running.push_back(frame);
            self.workers.run(job);
        }
    }

    /// The pictures chosen for output whose jobs have ended, in output
    /// order; with `wait`, every one. A failed picture ends the run: it is
    /// reported alone, by this call when nothing came before it.
    fn collect(&mut self, wait: bool) -> Result<Vec<(OutSamples, Arc<Meta>)>, CodecError> {
        let mut out = Vec::new();
        while let Some((frame, _)) = self.queue.front() {
            if !wait && !frame.is_done() {
                break;
            }
            if !out.is_empty() && frame.failed() {
                break;
            }
            let Some((frame, meta)) = self.queue.pop_front() else {
                break;
            };
            let done = frame.take_done();
            done.result?;
            if let Some(samples) = done.out {
                out.push((samples, meta));
            }
        }
        Ok(out)
    }

    fn push_unit(&mut self, packet: &Packet) -> Result<Vec<(OutSamples, Arc<Meta>)>, CodecError> {
        if packet.data.is_empty() {
            return Err(CodecError::InvalidApi {
                what: "empty packet",
            });
        }
        let mut state = self.state.clone();
        let (ready, bumped) = self.parse_unit(&mut state, packet)?;
        if self.threads <= 1 {
            // One thread: the push fails as a whole, the state unchanged.
            let frames: Vec<Arc<Frame>> = ready.iter().map(|r| r.frame().clone()).collect();
            for r in ready {
                self.start(r);
            }
            for f in &frames {
                if f.failed() {
                    f.take_done().result?;
                }
            }
            self.state = state;
        } else {
            self.state = state;
            for r in ready {
                self.start(r);
            }
        }
        self.queue.extend(bumped);
        self.drained = false;
        self.collect(self.threads <= 1)
    }

    fn drain_all(&mut self) -> Result<Vec<(OutSamples, Arc<Meta>)>, CodecError> {
        if self.drained && self.queue.is_empty() {
            return Ok(Vec::new());
        }
        self.drained = true;
        let mut bumped = Vec::new();
        self.state.flush(&mut bumped);
        self.queue.extend(bumped);
        let out = self.collect(true)?;
        self.running.retain(|f| !f.is_done());
        Ok(out)
    }

    fn reset_all(&mut self) {
        self.state = State::new();
        self.drained = false;
        self.queue.clear();
        self.running.clear();
        // The extradata parsed when the decoder opened, so it parses again.
        let _ = self.apply_extra();
    }
}

impl Drop for HevcDecoder {
    fn drop(&mut self) {
        // Jobs still running end on their own; the threads then stop.
        self.workers.close();
    }
}

/// Run a job; a panic inside it still ends the picture, so nothing waits
/// for its rows forever.
fn guarded<P: Pixel>(job: Job<P>) -> Option<Leftovers<P>> {
    let frame = job.frame.clone();
    let left = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job.run())).ok();
    if left.is_none() {
        frame.abandon();
    }
    left
}

#[allow(clippy::type_complexity)]
fn make_job<P: Pixel>(
    shared: &Shared,
    works: &Mutex<Vec<Work<P>>>,
    frame_bufs: &Mutex<Vec<[cutlass_simd::shared_rows::PlaneBuf<P>; 3]>>,
    wrap: fn(Arc<cutlass_simd::shared_rows::SharedRows<P>>) -> AnyRows,
    (pic, slices, output, row_threads): (Picture, Vec<SliceData>, Option<OutForm>, usize),
    (width, height, log2_ctb, poc): (usize, usize, u32, i32),
) -> Job<P> {
    let work = {
        let mut works = lock(works);
        match works.iter().position(|w| w.fits(width, height, log2_ctb)) {
            Some(i) => works.swap_remove(i),
            None => Work::new(width, height, log2_ctb),
        }
    };
    let bufs = lock(frame_bufs).pop();
    let (frame, writer, col) = Frame::new((width, height), poc, &shared.frames, wrap, bufs);
    Job {
        frame,
        pic,
        work,
        writer,
        col,
        slices,
        output,
        row_threads,
    }
}

fn parse(state: &State, nal: &Nal, prev: Option<&SliceHeader>) -> Result<SliceHeader, CodecError> {
    let lookup = |id: u8| state.sets(id).map(|(s, p)| ((*s).clone(), (*p).clone()));
    let prev = prev.cloned();
    slice::parse_header(nal, &lookup, prev.as_ref()).map(|(header, _, _)| header)
}

/// Reference picture lists of one slice (clause 8.3.4).
fn ref_lists(pending: &Pending, header: &SliceHeader) -> Result<[Vec<RefPic>; 2], CodecError> {
    if header.slice_type == SliceType::I {
        return Ok([Vec::new(), Vec::new()]);
    }
    let total = pending.st_before.len() + pending.st_after.len() + pending.lt_curr.len();
    if total == 0 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "inter slice without references",
        });
    }
    let mut lists = [Vec::new(), Vec::new()];
    let count = if header.slice_type == SliceType::B {
        2
    } else {
        1
    };
    for (list, out) in lists.iter_mut().enumerate().take(count) {
        let wanted = header.num_ref_idx[list];
        let size = wanted.max(total);
        let order: [&Vec<RefPic>; 3] = if list == 0 {
            [&pending.st_before, &pending.st_after, &pending.lt_curr]
        } else {
            [&pending.st_after, &pending.st_before, &pending.lt_curr]
        };
        let mut temp = Vec::with_capacity(size);
        while temp.len() < size {
            for part in order {
                for r in part {
                    if temp.len() < size {
                        temp.push(r.clone());
                    }
                }
            }
        }
        for i in 0..wanted {
            let index = match &header.list_entry[list] {
                Some(entries) => *entries.get(i).ok_or(CodecError::Malformed {
                    at_bit: 0,
                    what: "list_entry",
                })? as usize,
                None => i,
            };
            let Some(r) = temp.get(index) else {
                return Err(CodecError::Malformed {
                    at_bit: 0,
                    what: "list_entry",
                });
            };
            out.push(r.clone());
        }
    }
    Ok(lists)
}

/// Record one slice segment of the pending picture.
fn add_slice(pending: &mut Pending, nal: Arc<Nal>, header: SliceHeader) -> Result<(), CodecError> {
    if !header.dependent {
        pending.slice_addr = header.segment_address;
        pending.independent = Some(header.clone());
    } else if pending.independent.is_none() {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "dependent slice without a slice",
        });
    }
    if pending.pic.pps.id != header.pps_id {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "pps changed inside a picture",
        });
    }
    let refs = ref_lists(pending, &header)?;
    let poc = pending.pic.poc;
    let no_backward_pred = refs.iter().flatten().all(|r| r.poc <= poc);
    let qp = pending.pic.pps.init_qp + header.qp_delta;
    let slice_type = header.slice_type;
    // Substream starts, from coded offsets to RBSP offsets in the data.
    let start = header.data_offset;
    let Some(data) = nal.rbsp.get(start..) else {
        return Err(CodecError::Truncated {
            at_bit: 0,
            what: "slice data",
        });
    };
    let raw_start = nal.raw_of(start);
    let mut entries = Vec::with_capacity(header.entry_points.len());
    let mut raw = raw_start;
    for &offset in &header.entry_points {
        raw = raw.saturating_add(offset as usize);
        let at = nal.rbsp_of(raw).saturating_sub(start);
        if at > data.len() {
            return Err(CodecError::Truncated {
                at_bit: 0,
                what: "entry point",
            });
        }
        entries.push(at);
    }
    let segment = header.segment_address as usize;
    if segment >= pending.pic.layout.ctb_count() as usize {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "slice runs past the picture",
        });
    }
    pending.pic.slices.push(SliceCtx {
        header,
        refs,
        slice_addr: pending.slice_addr,
        qp,
        no_backward_pred,
    });
    let index = pending.pic.slices.len() - 1;
    if index >= usize::from(u16::MAX) {
        return Err(CodecError::Limit { what: "slices" });
    }
    pending.slices.push(SliceData {
        nal,
        index,
        start,
        entries,
        qp,
        slice_type,
        segment,
    });
    Ok(())
}

/// The output picture, 8-bit and, above 8 bits, the full samples.
fn video_output(samples: OutSamples, meta: &Meta) -> Option<VideoOutput> {
    let sps = &meta.sps;
    let [left, right, top, bottom] = sps.crop.map(|v| v as usize);
    let w = (sps.width as usize).checked_sub(left + right)?;
    let h = (sps.height as usize).checked_sub(top + bottom)?;
    let bits = u32::from(sps.bit_depth);
    let OutSamples::Planes { eight, deep } = samples else {
        return None;
    };
    let [y, u, v] = eight;
    let picture = Yuv420p::try_new(w as u32, h as u32, y, u, v).ok()?;
    let (yuv420p10, yuv420_high) = match (bits, deep) {
        (10, Some([y, u, v])) => (Yuv420p10::try_new(w as u32, h as u32, y, u, v).ok(), None),
        (8, _) | (_, None) => (None, None),
        (_, Some([y, u, v])) => (
            None,
            Yuv420High::try_new(bits as u8, w as u32, h as u32, y, u, v).ok(),
        ),
    };
    let vui = &sps.vui;
    Some(VideoOutput {
        picture,
        yuv420p10,
        yuv420_high,
        yuv422: None,
        yuv422_high: None,
        yuv444: None,
        yuv444_high: None,
        pts: meta.pts,
        duration: meta.duration,
        timescale: meta.timescale,
        keyframe: meta.keyframe,
        range: vui.range,
        primaries: vui.primaries,
        transfer: vui.transfer,
        matrix: vui.matrix,
        sample_aspect: vui.sample_aspect,
        timing: SequenceTiming {
            num_units: vui.num_units,
            time_scale: vui.time_scale,
            fixed: vui.fixed_rate,
        },
    })
}

/// The engine's packed picture.
fn packed_output(samples: OutSamples, meta: &Meta) -> Result<Option<Output>, CodecError> {
    let OutSamples::Packed(bytes) = samples else {
        return Ok(None);
    };
    let sps = &meta.sps;
    let [left, right, top, bottom] = sps.crop.map(|v| v as usize);
    let w = (sps.width as usize).saturating_sub(left + right);
    let h = (sps.height as usize).saturating_sub(top + bottom);
    let vui = &sps.vui;
    let picture = PictureOutput::cpu(
        PictureInfo::new(
            w as u32,
            h as u32,
            PixelFormat::Yuv420p,
            vui.primaries,
            vui.transfer,
        ),
        bytes,
        meta.pts,
        meta.timescale,
    )?;
    Ok(Some(Output::Picture(picture)))
}

impl VideoDecoder for HevcDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Hevc
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<VideoOutput>, CodecError> {
        self.form = OutForm::Planes;
        let out = self.push_unit(packet)?;
        Ok(out
            .into_iter()
            .filter_map(|(s, m)| video_output(s, &m))
            .collect())
    }

    fn drain(&mut self) -> Result<Vec<VideoOutput>, CodecError> {
        let out = self.drain_all()?;
        Ok(out
            .into_iter()
            .filter_map(|(s, m)| video_output(s, &m))
            .collect())
    }

    fn reset(&mut self) {
        self.reset_all();
    }
}

pub fn register_decoders(factory: &mut DecodeFactory) {
    factory.register(CodecId::Hevc, open_registered);
    factory.register_video(CodecId::Hevc, open_decoder);
}

fn open_decoder(params: &CodecParams) -> Result<Box<dyn VideoDecoder>, CodecError> {
    Ok(Box::new(HevcDecoder::open(params)?))
}

fn open_registered(params: &CodecParams) -> Result<Box<dyn Decoder>, CodecError> {
    let mut inner = HevcDecoder::open(params)?;
    inner.form = OutForm::Packed;
    Ok(Box::new(AccessDecoder { inner }))
}

/// The engine's decoder. It returns one packed picture per displayed frame.
struct AccessDecoder {
    inner: HevcDecoder,
}

impl Decoder for AccessDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Hevc
    }

    fn push(&mut self, packet: &Packet) -> Result<Vec<Output>, CodecError> {
        let out = self.inner.push_unit(packet)?;
        let mut pictures = Vec::with_capacity(out.len());
        for (s, m) in out {
            if let Some(p) = packed_output(s, &m)? {
                pictures.push(p);
            }
        }
        Ok(pictures)
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        let out = self.inner.drain_all()?;
        let mut pictures = Vec::with_capacity(out.len());
        for (s, m) in out {
            if let Some(p) = packed_output(s, &m)? {
                pictures.push(p);
            }
        }
        Ok(pictures)
    }

    fn reset(&mut self) {
        self.inner.reset_all();
    }
}

#[allow(dead_code)]
fn _assert_send() {
    fn send<T: Send>() {}
    send::<Job<u8>>();
    send::<Job<u16>>();
}
