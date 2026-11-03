//! Decode a clip on its own thread, ahead of the time being asked for.
//!
//! The worker owns a second [`Media`] on the same file. A caller posts the
//! source time it needs and waits for the picture that covers it. Between
//! requests the worker decodes forward from the last picture, so a caller
//! moving forward one frame at a time finds the next picture already there.
//! A jump decodes from the keyframe before it, on the worker.
//!
//! The worker keeps at most `ahead` pictures past the asked time and
//! [`BEHIND`] before it. It stops reading ahead at the end of the stream.
//!
//! Moving backward, a long-GOP file can only be decoded forward from a
//! keyframe. So while the asked time moves back the worker keeps every
//! picture it decodes behind it, up to [`BEHIND_BYTES`], and once it has
//! answered it fills in the pictures before those, a keyframe run at a time,
//! so stepping back finds them already decoded.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use cutlass_model::{Micros, TimeRange};

use crate::error::EngineError;
use crate::reader::{HeldFrame, Media, Steer};

/// Pictures kept before the asked time, for a short step back.
pub const BEHIND: usize = 2;
/// The memory kept in pictures before the asked time while it moves back.
pub const BEHIND_BYTES: usize = 384 << 20;
/// Pictures this close are one run with no picture missing between them.
const TOUCHING: i64 = 1_000;
/// The longest a caller waits for one picture.
const WAIT: Duration = Duration::from_secs(2);
/// How often a long decode stores a picture on its way, so a caller that
/// does not wait sees it move.
const PUBLISH: Duration = Duration::from_millis(50);

/// One clip's decode thread and the pictures it has ready.
pub(crate) struct ReadAhead {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
}

struct State {
    /// The newest time a caller is waiting on, in source time.
    want: Option<Micros>,
    /// Decoded pictures in display order.
    frames: VecDeque<HeldFrame>,
    /// The picture the worker chose for `want`, by start, once decoded.
    /// A time before the first picture or past the last one is answered by a
    /// picture that does not cover it.
    answer: Option<(Micros, Micros)>,
    /// Why the last asked time has no picture.
    failed: Option<(Micros, EngineError)>,
    /// Decoding forward reached the last picture.
    ended: bool,
    /// The asked time last moved earlier.
    backward: bool,
    /// A fill behind this picture start found nothing earlier: the start of
    /// the stream, or a run that would not decode.
    floor: Option<Micros>,
    stop: bool,
}

impl ReadAhead {
    /// Start a worker over `media`. `ahead` is at least one.
    pub(crate) fn start(media: Media, cover: TimeRange, ahead: usize) -> Result<Self, EngineError> {
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                want: None,
                frames: VecDeque::new(),
                answer: None,
                failed: None,
                ended: false,
                backward: false,
                floor: None,
                stop: false,
            }),
            wake: Condvar::new(),
        });
        let worker_shared = Arc::clone(&shared);
        let ahead = ahead.max(1);
        let worker = std::thread::Builder::new()
            .name("cutlass-read-ahead".to_owned())
            .spawn(move || run(&worker_shared, media, &cover, ahead))
            .map_err(|_| EngineError::NotReady)?;
        Ok(Self {
            shared,
            worker: Some(worker),
        })
    }

    /// The picture that covers `source` if it is decoded, without waiting.
    /// A miss asks the worker for it. A time the worker could not decode is
    /// its error.
    pub(crate) fn try_picture(&self, source: Micros) -> Result<Option<HeldFrame>, EngineError> {
        let mut state = lock(&self.shared.state);
        if let Some(held) = find(&state, source) {
            let held = held.clone();
            if state.want != Some(source) {
                aim(&mut state, source);
                self.shared.wake.notify_all();
            }
            return Ok(Some(held));
        }
        if let Some((at, error)) = &state.failed
            && *at == source
        {
            return Err(error.clone());
        }
        if state.want != Some(source) {
            aim(&mut state, source);
            state.answer = None;
            state.failed = None;
            self.shared.wake.notify_all();
        }
        Ok(None)
    }

    /// The stored picture nearest `source`, covering it or not, for a caller
    /// that shows something while the asked picture decodes.
    pub(crate) fn nearest(&self, source: Micros) -> Option<HeldFrame> {
        lock(&self.shared.state)
            .frames
            .iter()
            .min_by_key(|held| held.start.0.abs_diff(source.0))
            .cloned()
    }

    /// Ask for the picture that covers `source` without waiting for it, so
    /// it is decoded by the time [`Self::picture`] asks.
    pub(crate) fn prefetch(&self, source: Micros) {
        let mut state = lock(&self.shared.state);
        if state.want == Some(source) || find(&state, source).is_some() {
            return;
        }
        aim(&mut state, source);
        state.answer = None;
        state.failed = None;
        self.shared.wake.notify_all();
    }

    /// The picture that covers `source`, waiting for the worker when it is
    /// not decoded yet.
    pub(crate) fn picture(&self, source: Micros) -> Result<HeldFrame, EngineError> {
        let mut state = lock(&self.shared.state);
        if let Some(held) = find(&state, source) {
            let held = held.clone();
            if state.want != Some(source) {
                aim(&mut state, source);
                self.shared.wake.notify_all();
            }
            return Ok(held);
        }
        aim(&mut state, source);
        state.answer = None;
        state.failed = None;
        self.shared.wake.notify_all();
        let (mut state, _) = self
            .shared
            .wake
            .wait_timeout_while(state, WAIT, |state| {
                find(state, source).is_none()
                    && state.failed.as_ref().is_none_or(|(at, _)| *at != source)
            })
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(held) = find(&state, source) {
            return Ok(held.clone());
        }
        match state.failed.take() {
            Some((at, error)) if at == source => Err(error),
            _ => Err(EngineError::NotReady),
        }
    }
}

impl Drop for ReadAhead {
    fn drop(&mut self) {
        lock(&self.shared.state).stop = true;
        self.shared.wake.notify_all();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Ask for `source`, noting whether the asked time moved back.
fn aim(state: &mut State, source: Micros) {
    if let Some(want) = state.want
        && want != source
    {
        state.backward = source.0 < want.0;
    }
    state.want = Some(source);
}

fn lock(state: &Mutex<State>) -> MutexGuard<'_, State> {
    state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The stored picture for `source`: one that covers it, or the answer the
/// worker gave for exactly this time.
fn find(state: &State, source: Micros) -> Option<&HeldFrame> {
    if let Some(held) = state
        .frames
        .iter()
        .find(|held| held.start.0 <= source.0 && source.0 < held.end.0)
    {
        return Some(held);
    }
    let (want, start) = state.answer?;
    if want != source {
        return None;
    }
    state.frames.iter().find(|held| held.start == start)
}

/// What the worker decodes next.
enum Job {
    /// A caller waits on this time.
    Answer(Micros),
    /// Read ahead from the end of the last picture.
    Ahead(Micros),
    /// Moving back, decode the pictures before this picture start.
    Behind(Micros),
}

fn next_job(state: &State, ahead: usize) -> Option<Job> {
    let want = state.want?;
    if find(state, want).is_none() {
        if state.failed.as_ref().is_some_and(|(at, _)| *at == want) {
            return None;
        }
        return Some(Job::Answer(want));
    }
    if state.backward {
        return behind_job(state, want);
    }
    if state.ended {
        return None;
    }
    let past = state
        .frames
        .iter()
        .filter(|held| held.start.0 > want.0)
        .count();
    if past >= ahead {
        return None;
    }
    state.frames.back().map(|last| Job::Ahead(last.end))
}

/// Fill in before the run of pictures that ends at `want`, once it holds
/// less than three quarters of [`BEHIND_BYTES`]. Waiting for the run to run
/// low keeps each step back from decoding a keyframe run for one picture.
fn behind_job(state: &State, want: Micros) -> Option<Job> {
    let found = find(state, want)?.start;
    let mut index = state.frames.iter().position(|held| held.start == found)?;
    let mut bytes = 0;
    while index > 0 {
        let (earlier, later) = (&state.frames[index - 1], &state.frames[index]);
        if earlier.end.0.saturating_add(TOUCHING) < later.start.0 {
            break;
        }
        bytes += earlier.frame.byte_size();
        index -= 1;
    }
    let edge = state.frames[index].start;
    if bytes >= BEHIND_BYTES / 4 * 3 || state.floor == Some(edge) {
        return None;
    }
    Some(Job::Behind(edge))
}

fn run(shared: &Shared, mut media: Media, cover: &TimeRange, ahead: usize) {
    let mut state = lock(&shared.state);
    loop {
        if state.stop {
            return;
        }
        let Some(job) = next_job(&state, ahead) else {
            state = shared
                .wake
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            continue;
        };
        drop(state);
        let (mut time, answering) = match job {
            Job::Answer(time) => (time, true),
            Job::Ahead(time) => (time, false),
            Job::Behind(edge) => {
                let filled = fill(shared, &mut media, cover, ahead, edge);
                state = lock(&shared.state);
                let interrupted = matches!(filled, Err(EngineError::Interrupted));
                match filled {
                    Ok(held) => store(&mut state, held, ahead),
                    Err(EngineError::Interrupted) => {}
                    Err(_) => state.floor = Some(edge),
                }
                if !interrupted && !state.frames.iter().any(|held| held.start.0 < edge.0) {
                    state.floor = Some(edge);
                }
                shared.wake.notify_all();
                continue;
            }
        };
        let decoded = if answering {
            answer(shared, &mut media, cover, ahead, &mut time)
        } else {
            media.decode_at(time, cover)
        };
        state = lock(&shared.state);
        match decoded {
            Err(EngineError::Interrupted) => {}
            Ok(held) => {
                if answering {
                    state.answer = Some((time, held.start));
                    state.ended = false;
                } else if state
                    .frames
                    .back()
                    .is_some_and(|last| held.start.0 <= last.start.0)
                {
                    // Decoding forward returned the last picture again.
                    state.ended = true;
                }
                store(&mut state, held, ahead);
            }
            Err(error) => {
                if answering {
                    state.failed = Some((time, error));
                } else {
                    state.ended = true;
                }
            }
        }
        shared.wake.notify_all();
    }
}

/// Decode toward `target`, following the newest asked time while it is still
/// ahead, so dragging the playhead never waits on a time it has left.
///
/// A time behind the decode stops it; the next job seeks for it. On a long
/// decode a picture is stored every [`PUBLISH`], so the caller can show
/// progress toward the asked time instead of holding one picture.
fn answer(
    shared: &Shared,
    media: &mut Media,
    cover: &TimeRange,
    ahead: usize,
    target: &mut Micros,
) -> Result<HeldFrame, EngineError> {
    let mut published = Instant::now();
    let start = *target;
    media.decode_steered(start, cover, &mut |held| {
        let mut state = lock(&shared.state);
        if state.stop {
            return Steer::Stop;
        }
        // Moving back, every picture on the way is one a later step wants.
        if state.backward || published.elapsed() >= PUBLISH {
            published = Instant::now();
            store(&mut state, held.clone(), ahead);
            shared.wake.notify_all();
        }
        match state.want {
            Some(want) if want == *target => Steer::Go,
            Some(want) => {
                *target = want;
                Steer::Retarget(want)
            }
            None => Steer::Go,
        }
    })
}

/// Decode the pictures before `edge`, from the keyframe before it, and
/// store each one. A turn forward, or an asked time this run will not reach,
/// stops it.
fn fill(
    shared: &Shared,
    media: &mut Media,
    cover: &TimeRange,
    ahead: usize,
    edge: Micros,
) -> Result<HeldFrame, EngineError> {
    media.decode_steered(Micros(edge.0.saturating_sub(1)), cover, &mut |held| {
        let mut state = lock(&shared.state);
        if state.stop || !state.backward {
            return Steer::Stop;
        }
        store(&mut state, held.clone(), ahead);
        shared.wake.notify_all();
        match state.want {
            Some(want)
                if find(&state, want).is_none() && !(held.start.0 <= want.0 && want.0 < edge.0) =>
            {
                Steer::Stop
            }
            _ => Steer::Go,
        }
    })
}

/// Insert `held` in display order and drop pictures outside the window
/// around the asked time. Moving back, the pictures nearest it are kept up to
/// [`BEHIND_BYTES`]; otherwise [`BEHIND`] of them.
fn store(state: &mut State, held: HeldFrame, ahead: usize) {
    if !state.frames.iter().any(|stored| stored.start == held.start) {
        let at = state
            .frames
            .iter()
            .position(|stored| stored.start.0 > held.start.0)
            .unwrap_or(state.frames.len());
        state.frames.insert(at, held);
    }
    let Some(want) = state.want else {
        return;
    };
    let answer = state
        .answer
        .filter(|(at, _)| *at == want)
        .map(|(_, start)| start);
    let keep = |held: &HeldFrame| {
        answer == Some(held.start) || held.end.0 > want.0 || held.start.0 > want.0
    };
    let budget = if state.backward { BEHIND_BYTES } else { 0 };
    let (mut behind, mut bytes) = (0, 0);
    for index in (0..state.frames.len()).rev() {
        if !keep(&state.frames[index]) {
            behind += 1;
            bytes += state.frames[index].frame.byte_size();
            if behind > BEHIND && bytes > budget {
                state.frames.remove(index);
            }
        }
    }
    let mut past = 0;
    state.frames.retain(|held| {
        if held.start.0 > want.0 {
            past += 1;
            past <= ahead
        } else {
            true
        }
    });
}

#[cfg(test)]
mod tests {
    use super::{BEHIND, BEHIND_BYTES, Job, State, aim, next_job, store};
    use crate::reader::{DecodedFrame, HeldFrame};
    use cutlass_model::Micros;
    use std::collections::VecDeque;

    fn held(start: i64) -> HeldFrame {
        HeldFrame {
            start: Micros(start),
            end: Micros(start + 10),
            frame: DecodedFrame::Platform {
                width: 2,
                height: 2,
                token: 0,
            },
        }
    }

    fn state(want: i64) -> State {
        State {
            want: Some(Micros(want)),
            frames: VecDeque::new(),
            answer: None,
            failed: None,
            ended: false,
            backward: false,
            floor: None,
            stop: false,
        }
    }

    #[test]
    fn the_window_keeps_a_few_behind_and_the_lookahead_past() {
        let mut state = state(50);
        for start in [70, 0, 10, 20, 30, 40, 50, 60, 80, 90, 100] {
            store(&mut state, held(start), 3);
        }
        let starts: Vec<i64> = state.frames.iter().map(|held| held.start.0).collect();
        assert_eq!(starts.len(), BEHIND + 1 + 3);
        assert_eq!(starts, [30, 40, 50, 60, 70, 80]);
    }

    fn sized(start: i64, bytes: usize) -> HeldFrame {
        HeldFrame {
            start: Micros(start),
            end: Micros(start + 10),
            frame: DecodedFrame::Coded {
                width: 1,
                height: 1,
                pixel: cutlass_decode::PixelFormat::Yuv420p,
                primaries: cutlass_decode::ColorPrimaries::Bt709,
                transfer: cutlass_decode::Transfer::Bt709,
                bytes: vec![0; bytes],
            },
        }
    }

    #[test]
    fn moving_back_keeps_the_pictures_behind_up_to_the_memory_budget() {
        let mut state = state(50);
        aim(&mut state, Micros(40));
        assert!(state.backward);
        let each = BEHIND_BYTES / 3;
        for start in [0, 10, 20, 30, 40] {
            store(&mut state, sized(start, each), 3);
        }
        let starts: Vec<i64> = state.frames.iter().map(|held| held.start.0).collect();
        // The picture at 40 covers the asked time. Three behind it fill the
        // budget, so the oldest goes.
        assert_eq!(starts, [10, 20, 30, 40]);
        aim(&mut state, Micros(45));
        assert!(!state.backward);
        store(&mut state, sized(50, each), 3);
        let starts: Vec<i64> = state.frames.iter().map(|held| held.start.0).collect();
        assert_eq!(starts.len(), BEHIND + 2, "{starts:?}");
    }

    #[test]
    fn moving_back_fills_before_the_run_once_it_runs_low() {
        let mut state = state(50);
        aim(&mut state, Micros(40));
        let each = BEHIND_BYTES / 8;
        for start in [20, 30, 40] {
            store(&mut state, sized(start, each), 3);
        }
        assert!(matches!(next_job(&state, 3), Some(Job::Behind(Micros(20)))));
        // A fill that found nothing earlier is not tried again.
        state.floor = Some(Micros(20));
        assert!(next_job(&state, 3).is_none());
        // A full run waits.
        let mut state = state_at(40, true);
        for start in [0, 10, 20, 30, 40] {
            store(&mut state, sized(start, BEHIND_BYTES / 4), 3);
        }
        assert!(next_job(&state, 3).is_none());
        // Moving forward reads ahead instead.
        let mut state = state_at(40, false);
        store(&mut state, sized(40, 1), 3);
        assert!(matches!(next_job(&state, 3), Some(Job::Ahead(Micros(50)))));
    }

    fn state_at(want: i64, backward: bool) -> State {
        let mut state = state(want);
        state.backward = backward;
        state
    }

    #[test]
    fn a_picture_is_stored_once() {
        let mut state = state(0);
        store(&mut state, held(0), 4);
        store(&mut state, held(0), 4);
        assert_eq!(state.frames.len(), 1);
    }
}
