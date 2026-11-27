//! The software decoder: the jobs that reconstruct pictures, on the
//! caller's thread or on workers, and the engine's decoder interfaces. What
//! comes before reconstruction (parameter sets, picture order counts,
//! reference picture sets and lists, the decoded picture buffer and output
//! order) is the front end of [`crate::front`], shared with the stateless
//! decoder.
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
use crate::front::{Backend, Front, Meta, SliceIn, Start};
use crate::job::{Job, Leftovers, OutForm, OutRows, SliceData};
use crate::picture::{Picture, RefPic, SliceCtx, Work, lock};
use crate::pixel::{AnyRows, Pixel};
use crate::ps::{Layout, Pps, Sps};

/// Most threads a decoder uses.
const MAX_THREADS: usize = 16;

/// Everything the headers of a push may change. A failed push leaves the
/// last good copy.
#[derive(Clone)]
struct State {
    front: Front<Arc<Frame>>,
    /// The layout of the last parameter sets a picture used.
    layout: Option<(Arc<Sps>, Arc<Pps>, Arc<Layout>)>,
}

impl State {
    fn new() -> Self {
        Self {
            front: Front::new(),
            layout: None,
        }
    }
}

/// A picture chosen for output, in output order.
type Bumped = crate::front::Bumped<Arc<Frame>>;

/// The picture whose slice segments are being read.
struct Pending {
    pic: Picture,
    sps: Arc<Sps>,
    output: bool,
    /// Whether a later picture may reference this one: not when it is a
    /// sub-layer non-reference picture of the highest sub-layer.
    reference: bool,
    /// The references the front end handed over, in its order.
    refs: Vec<RefPic>,
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

    fn pic(&self) -> &Picture {
        match self {
            Self::Eight(j) => &j.pic,
            Self::Deep(j) => &j.pic,
        }
    }

    /// Whether the picture's slice segments decode as wavefronts.
    fn wavefront(&self) -> bool {
        self.pic().pps.entropy_coding_sync_enabled_flag
    }
}

/// Buffers of finished jobs for the next ones.
#[derive(Default)]
struct Shared {
    frames: Arc<Spares>,
    pics: Mutex<Vec<Picture>>,
    work8: Mutex<Vec<Work<u8>>>,
    work16: Mutex<Vec<Work<u16>>>,
    /// The packed pictures handed out, to fill again once every other
    /// holder has let one go.
    packed: Mutex<Vec<Arc<Vec<u8>>>>,
}

impl Shared {
    /// A packed picture handed out earlier that no one else holds now.
    fn spare_packed(&self) -> Option<Vec<u8>> {
        let mut packed = lock(&self.packed);
        let free = packed.iter().position(|p| Arc::strong_count(p) == 1)?;
        Arc::try_unwrap(packed.swap_remove(free)).ok()
    }

    /// Keep a packed picture handed out, to fill again later.
    fn keep_packed(&self, picture: &Arc<Vec<u8>>) {
        let mut packed = lock(&self.packed);
        if packed.len() < 2 * MAX_THREADS {
            packed.push(picture.clone());
        }
    }

    fn give8(&self, mut left: Leftovers<u8>) {
        // The slices hold the picture's references: let them go now, not
        // when the picture is filled again, so that a spare picture keeps
        // no reference frame alive.
        left.pic.slices.clear();
        put(&self.pics, left.pic);
        put(&self.work8, left.work);
    }

    fn give16(&self, mut left: Leftovers<u16>) {
        left.pic.slices.clear();
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
    /// Threads the decoder may use (1: each picture on the caller's
    /// thread).
    threads: usize,
    /// Pictures decoding at once and threads for the CTB rows of a
    /// wavefront slice segment, when set by [`Self::set_threads`];
    /// otherwise chosen per picture by [`Self::split`].
    split: Option<(usize, usize)>,
    form: OutForm,
    shared: Arc<Shared>,
    workers: Arc<Workers>,
    /// Pictures whose jobs were started, oldest first, until they end.
    running: VecDeque<Arc<Frame>>,
    /// Pictures chosen for output whose samples are not handed out yet.
    queue: VecDeque<Bumped>,
    /// For tests: each picture decoded, in decoding order, as its order
    /// count and, for each slice segment, the order counts of its
    /// reference picture lists as reconstruction reads them.
    #[cfg(test)]
    pub(crate) ref_trace: Vec<(i32, Vec<[Vec<i32>; 2]>)>,
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
            split: None,
            form: OutForm::Planes,
            shared: Arc::new(Shared::default()),
            workers: Arc::new(Workers::default()),
            running: VecDeque::new(),
            queue: VecDeque::new(),
            #[cfg(test)]
            ref_trace: Vec::new(),
        };
        dec.apply_extra()?;
        Ok(dec)
    }

    /// Pictures decoding at once, and threads for the rows of a wavefront
    /// slice segment (each at least 1), in place of the split the decoder
    /// chooses itself.
    pub fn set_threads(&mut self, frames: usize, rows: usize) {
        let (frames, rows) = (frames.clamp(1, MAX_THREADS), rows.clamp(1, MAX_THREADS));
        self.threads = frames;
        self.split = Some((frames, rows));
    }

    /// Pictures decoding at once and threads for the rows of a wavefront
    /// slice segment, for a picture with wavefronts (`wavefront`) or
    /// without. With four threads or more a wavefront picture takes two
    /// threads, one row each, and half as many pictures decode at once: a
    /// picture waiting for the rows of its references then holds less of
    /// the machine idle.
    fn split(&self, wavefront: bool) -> (usize, usize) {
        match self.split {
            Some(split) => split,
            // Pictures decode at once up to eight, each on a thread of its
            // own: their data stays on one core, and a picture waits only
            // for the rows of its references it reads. Past eight, the rows
            // of a wavefront picture share the threads.
            None if wavefront && self.threads > 8 => (8, self.threads / 8),
            None => (self.threads, 1),
        }
    }

    fn apply_extra(&mut self) -> Result<(), CodecError> {
        self.state
            .front
            .apply_extra(&self.extra, &mut self.length_size)
    }

    /// Read the headers of one access unit: the pictures to decode and
    /// the pictures chosen for output.
    fn parse_unit(
        &self,
        state: &mut State,
        packet: &Packet,
    ) -> Result<(Vec<Ready>, Vec<Bumped>), CodecError> {
        let State { front, layout } = state;
        let mut soft = Soft {
            dec: self,
            layout,
            ready: Vec::new(),
        };
        let mut out = Vec::new();
        front.access_unit(&mut soft, packet, self.length_size, &mut out)?;
        Ok((soft.ready, out))
    }

    /// The job of a picture whose slice segments are all read.
    fn ready(&self, pending: Pending) -> Ready {
        let Pending {
            pic,
            sps,
            output,
            reference,
            slices,
            ..
        } = pending;
        let (width, height) = (pic.width, pic.height);
        let poc = pic.poc;
        let log2_ctb = u32::from(pic.layout.log2_ctb);
        let form = output.then_some(self.form);
        let row_threads = self.split(pic.pps.entropy_coding_sync_enabled_flag).1;
        if sps.bit_depth() == 8 {
            Ready::Eight(make_job(
                &self.shared,
                &self.shared.work8,
                &self.shared.frames.eight,
                AnyRows::Eight,
                (pic, slices, form, row_threads),
                (width, height, log2_ctb, poc, reference),
            ))
        } else {
            Ready::Deep(make_job(
                &self.shared,
                &self.shared.work16,
                &self.shared.frames.deep,
                AnyRows::Deep,
                (pic, slices, form, row_threads),
                (width, height, log2_ctb, poc, reference),
            ))
        }
    }

    /// Start the decode of a picture, on a worker or (one thread) here.
    fn start(&mut self, ready: Ready) {
        // Keep at most so many pictures decoding. The next starts as soon
        // as any of them is released: done, or down to its last rows with
        // threads to spare.
        let (frames, rows) = self.split(ready.wavefront());
        // With a picture to each thread, one more thread than that decodes
        // at once, so that a core is not left idle while a thread that
        // waited for rows wakes, and up to half as many pictures again wait
        // ready: a picture that waits for the rows of a reference lets one
        // of them run in its place (see `Frame::run`). Rows of a picture on
        // several threads keep one picture to a thread.
        let slots = if rows == 1 && frames > 1 {
            frames + 1
        } else {
            0
        };
        self.shared.frames.set_slots(slots);
        let frames = if slots > 0 {
            frames + frames / 2
        } else {
            frames
        };
        loop {
            let seen = self.shared.frames.released();
            self.running.retain(|f| !f.is_released());
            if self.running.len() < frames {
                break;
            }
            self.shared.frames.wait_released(seen);
        }
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
            match (done.out, &frame.rows) {
                (Some(OutSamples::Later), AnyRows::Deep(rows)) => {
                    let mut o =
                        OutRows::new(&meta.sps, OutForm::Packed, self.shared.spare_packed());
                    o.take_rows(rows);
                    out.push((o.finish(), meta));
                }
                (Some(OutSamples::Later), AnyRows::Eight(_)) | (None, _) => {}
                (Some(samples), _) => out.push((samples, meta)),
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
        #[cfg(test)]
        let trace: Vec<(i32, Vec<[Vec<i32>; 2]>)> = ready
            .iter()
            .map(|r| {
                let pic = r.pic();
                let slices = pic.slices.iter().map(|s| {
                    s.refs
                        .each_ref()
                        .map(|list| list.iter().map(|r| r.poc).collect())
                });
                (pic.poc, slices.collect())
            })
            .collect();
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
        #[cfg(test)]
        self.ref_trace.extend(trace);
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
        self.state.front.flush(&mut bumped);
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
    let left = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let frame = job.frame.clone();
        frame.run(|| job.run())
    }))
    .ok();
    if left.is_none() {
        frame.abandon();
    }
    left
}

#[allow(clippy::type_complexity)]
fn make_job<P: Pixel>(
    shared: &Shared,
    works: &Mutex<Vec<Work<P>>>,
    frame_bufs: &Mutex<Vec<cutlass_simd::shared_rows::PlaneBuf<P>>>,
    wrap: fn(Arc<cutlass_simd::shared_rows::SharedRows<P>>) -> AnyRows,
    (pic, slices, output, row_threads): (Picture, Vec<SliceData>, Option<OutForm>, usize),
    (width, height, log2_ctb, poc, reference): (usize, usize, u32, i32, bool),
) -> Job<P> {
    // A ring of row buffers, unless tiles reorder the CTB rows.
    let slots = if pic.pps.tiles_enabled_flag {
        usize::MAX
    } else if pic.pps.entropy_coding_sync_enabled_flag {
        row_threads + crate::picture::RING
    } else {
        crate::picture::RING
    };
    let work = {
        let mut works = lock(works);
        match works
            .iter()
            .position(|w| w.fits(width, height, log2_ctb, slots))
        {
            Some(i) => works.swap_remove(i),
            None => Work::new(width, height, log2_ctb, slots),
        }
    };
    let staged = crate::job::staged(reference, output, &pic.sps);
    let bufs = (!staged).then(|| crate::frame::free(frame_bufs)).flatten();
    let (frame, writer, col) =
        Frame::new((width, height), poc, &shared.frames, wrap, bufs, !staged);
    // The frame of an 8-bit picture that is not cropped holds its planes
    // as the packed picture does: it is the output itself.
    let share = output == Some(OutForm::Packed)
        && if P::bytes(&[]).is_some() {
            pic.sps.crop() == [0; 4]
        } else {
            !staged
        };
    let spare = (output == Some(OutForm::Packed) && !share)
        .then(|| shared.spare_packed())
        .flatten();
    let out = output
        .filter(|_| !share)
        .map(|form| OutRows::new(&pic.sps, form, spare));
    Job {
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
    }
}

/// The software decoder as the front end's backend: a job for each
/// picture.
struct Soft<'a> {
    dec: &'a HevcDecoder,
    layout: &'a mut Option<(Arc<Sps>, Arc<Pps>, Arc<Layout>)>,
    /// The jobs of the pictures read, in decoding order.
    ready: Vec<Ready>,
}

impl Backend for Soft<'_> {
    type Pic = Arc<Frame>;
    type Building = Pending;

    fn sequence(&mut self, _sps: &Arc<Sps>) -> Result<(), CodecError> {
        Ok(())
    }

    /// Mid-grey.
    fn missing(&mut self, sps: &Arc<Sps>, poc: i32) -> Result<Arc<Frame>, CodecError> {
        let width = sps.pic_width_in_luma_samples as usize;
        let height = sps.pic_height_in_luma_samples as usize;
        Ok(Frame::missing(width, height, sps.bit_depth(), poc))
    }

    fn start(&mut self, start: Start<'_, Arc<Frame>>) -> Result<Pending, CodecError> {
        let Start {
            sps,
            pps,
            poc,
            output,
            reference,
            refs,
            ..
        } = start;
        let layout = match &*self.layout {
            Some((s, p, l)) if Arc::ptr_eq(s, sps) && Arc::ptr_eq(p, pps) => l.clone(),
            _ => {
                let l = Arc::new(Layout::new(sps, pps)?);
                *self.layout = Some((sps.clone(), pps.clone(), l.clone()));
                l
            }
        };
        let old = lock(&self.dec.shared.pics).pop();
        // The 4×4 maps are a ring of CTB rows unless tiles reorder them.
        let rows = if pps.entropy_coding_sync_enabled_flag {
            self.dec.split(true).1
        } else {
            1
        };
        let mask4 = Picture::map_mask(sps, (!pps.tiles_enabled_flag).then_some(rows));
        let pic = Picture::reuse(old, sps.clone(), pps.clone(), layout, (poc, mask4))?;
        let refs = refs
            .iter()
            .map(|r| RefPic {
                frame: r.pic.clone(),
                poc: r.poc,
                long_term: r.long_term,
                uid: r.uid,
            })
            .collect();
        Ok(Pending {
            pic,
            sps: sps.clone(),
            output,
            reference,
            refs,
            slices: Vec::new(),
        })
    }

    fn slice(&mut self, pending: &mut Pending, slice: SliceIn) -> Result<(), CodecError> {
        let SliceIn {
            nal,
            header,
            data_offset,
            entries,
            slice_addr,
            ref_lists,
            ..
        } = slice;
        let mut refs: [Vec<RefPic>; 2] = [Vec::new(), Vec::new()];
        for (list, indices) in refs.iter_mut().zip(&ref_lists) {
            for &i in indices {
                let Some(r) = pending.refs.get(i) else {
                    return Err(CodecError::Malformed {
                        at_bit: 0,
                        what: "reference picture list",
                    });
                };
                list.push(r.clone());
            }
        }
        let poc = pending.pic.poc;
        let no_backward_pred = refs.iter().flatten().all(|r| r.poc <= poc);
        let qp = 26 + pending.pic.pps.init_qp_minus26 + header.slice_qp_delta;
        let slice_type = header.slice_type;
        let segment = header.slice_segment_address as usize;
        pending.pic.slices.push(SliceCtx {
            header,
            refs,
            slice_addr,
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
            start: data_offset,
            entries,
            qp,
            slice_type,
            segment,
        });
        Ok(())
    }

    fn finish(&mut self, pending: Pending, _data: &[u8]) -> Result<Arc<Frame>, CodecError> {
        let ready = self.dec.ready(pending);
        let frame = ready.frame().clone();
        self.ready.push(ready);
        Ok(frame)
    }
}

/// The output picture, 8-bit and, above 8 bits, the full samples.
fn video_output(samples: OutSamples, meta: &Meta) -> Option<VideoOutput> {
    let sps = &meta.sps;
    let [left, right, top, bottom] = sps.crop().map(|v| v as usize);
    let w = (sps.pic_width_in_luma_samples as usize).checked_sub(left + right)?;
    let h = (sps.pic_height_in_luma_samples as usize).checked_sub(top + bottom)?;
    let bits = u32::from(sps.bit_depth());
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
        range: vui.range(),
        primaries: vui.primaries(),
        transfer: vui.transfer(),
        matrix: vui.matrix(),
        sample_aspect: vui.sample_aspect(),
        timing: SequenceTiming {
            num_units: vui.vui_num_units_in_tick,
            time_scale: vui.vui_time_scale,
            fixed: vui.vui_timing_info_present_flag,
        },
    })
}

/// The engine's packed picture.
fn packed_output(
    samples: OutSamples,
    meta: &Meta,
    shared: &Shared,
) -> Result<Option<Output>, CodecError> {
    let bytes = match samples {
        OutSamples::Packed(bytes) => {
            let bytes = Arc::new(bytes);
            shared.keep_packed(&bytes);
            bytes
        }
        OutSamples::Shared(bytes) => bytes,
        OutSamples::Planes { .. } | OutSamples::Later => return Ok(None),
    };
    let sps = &meta.sps;
    let [left, right, top, bottom] = sps.crop().map(|v| v as usize);
    let w = (sps.pic_width_in_luma_samples as usize).saturating_sub(left + right);
    let h = (sps.pic_height_in_luma_samples as usize).saturating_sub(top + bottom);
    let vui = &sps.vui;
    let picture = PictureOutput::cpu_shared(
        PictureInfo::new(
            w as u32,
            h as u32,
            PixelFormat::Yuv420p,
            vui.primaries(),
            vui.transfer(),
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
            if let Some(p) = packed_output(s, &m, &self.inner.shared)? {
                pictures.push(p);
            }
        }
        Ok(pictures)
    }

    fn drain(&mut self) -> Result<Vec<Output>, CodecError> {
        let out = self.inner.drain_all()?;
        let mut pictures = Vec::with_capacity(out.len());
        for (s, m) in out {
            if let Some(p) = packed_output(s, &m, &self.inner.shared)? {
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
