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

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;

use cutlass_model::{Micros, TimeRange};

use crate::error::EngineError;
use crate::reader::{HeldFrame, Media};

/// Pictures kept before the asked time, for a short step back.
pub const BEHIND: usize = 2;
/// The longest a caller waits for one picture.
const WAIT: Duration = Duration::from_secs(2);

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

    /// The picture that covers `source`, waiting for the worker when it is
    /// not decoded yet.
    pub(crate) fn picture(&self, source: Micros) -> Result<HeldFrame, EngineError> {
        let mut state = lock(&self.shared.state);
        if let Some(held) = find(&state, source) {
            let held = held.clone();
            if state.want != Some(source) {
                state.want = Some(source);
                self.shared.wake.notify_all();
            }
            return Ok(held);
        }
        state.want = Some(source);
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
}

fn next_job(state: &State, ahead: usize) -> Option<Job> {
    let want = state.want?;
    if find(state, want).is_none() {
        if state.failed.as_ref().is_some_and(|(at, _)| *at == want) {
            return None;
        }
        return Some(Job::Answer(want));
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
        let (time, answering) = match job {
            Job::Answer(time) => (time, true),
            Job::Ahead(time) => (time, false),
        };
        let decoded = media.decode_at(time, cover);
        state = lock(&shared.state);
        match decoded {
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

/// Insert `held` in display order and drop pictures outside the window
/// around the asked time.
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
    let mut behind = 0;
    for index in (0..state.frames.len()).rev() {
        if !keep(&state.frames[index]) {
            behind += 1;
            if behind > BEHIND {
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
    use super::{BEHIND, State, store};
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

    #[test]
    fn a_picture_is_stored_once() {
        let mut state = state(0);
        store(&mut state, held(0), 4);
        store(&mut state, held(0), 4);
        assert_eq!(state.frames.len(), 1);
    }
}
