//! Long work on its own thread, so the window keeps drawing. A job says
//! what it is doing, how far along it is when it knows, and stops when
//! asked between its steps.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError, mpsc};
use std::time::{Duration, Instant};

/// What a running job shows in the window.
#[derive(Clone, Debug, PartialEq)]
pub struct JobView {
    pub label: String,
    pub stage: String,
    /// 0 to 1, or `None` while the job cannot tell.
    pub fraction: Option<f32>,
    /// About how long the stage has to go, once it has run long enough to
    /// tell.
    pub left: Option<Duration>,
}

/// What a worker last reported: its stage, when the stage began, and how
/// far along it is.
struct State {
    stage: String,
    began: Instant,
    fraction: Option<f32>,
}

impl Default for State {
    fn default() -> Self {
        Self {
            stage: String::new(),
            began: Instant::now(),
            fraction: None,
        }
    }
}

/// The time a stage `fraction` done after `elapsed` has left, at the pace
/// so far. Nothing until a second has passed and a fiftieth is done.
#[must_use]
pub fn time_left(elapsed: Duration, fraction: f32) -> Option<Duration> {
    if elapsed < Duration::from_secs(1) || !(0.02..1.0).contains(&fraction) {
        return None;
    }
    Some(elapsed.mul_f32((1.0 - fraction) / fraction))
}

/// The worker's side: report where it is, and look for a cancel.
#[derive(Default)]
pub struct Progress {
    state: Mutex<State>,
    cancel: AtomicBool,
}

impl Progress {
    /// A new step, with no fraction until the step reports one.
    pub fn stage(&self, text: &str) {
        *self.state.lock().unwrap_or_else(PoisonError::into_inner) = State {
            stage: text.to_owned(),
            began: Instant::now(),
            fraction: None,
        };
    }

    pub fn fraction(&self, done: f32) {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .fraction = Some(done.clamp(0.0, 1.0));
    }

    /// The window asked the job to stop.
    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

/// A job started on its own thread, handing back an `L` when it ends.
pub struct Job<L> {
    label: String,
    progress: Arc<Progress>,
    done: mpsc::Receiver<L>,
}

/// How a poll found the job.
pub enum Poll<L> {
    Running,
    Finished(L),
    /// The thread ended without an answer.
    Lost,
}

impl<L: Send + 'static> Job<L> {
    pub fn start(
        label: &str,
        work: impl FnOnce(&Progress) -> L + Send + 'static,
    ) -> std::io::Result<Self> {
        let progress = Arc::new(Progress::default());
        let (sender, done) = mpsc::channel();
        let shared = Arc::clone(&progress);
        std::thread::Builder::new()
            .name(format!("cutlass job: {label}"))
            .spawn(move || {
                let landing = work(&shared);
                // A cancelled job has no one waiting.
                let _ = sender.send(landing);
            })?;
        Ok(Self {
            label: label.to_owned(),
            progress,
            done,
        })
    }

    pub fn view(&self) -> JobView {
        let state = self
            .progress
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        JobView {
            label: self.label.clone(),
            stage: state.stage.clone(),
            fraction: state.fraction,
            left: state
                .fraction
                .and_then(|fraction| time_left(state.began.elapsed(), fraction)),
        }
    }

    /// Ask the worker to stop at its next step. Its answer is dropped.
    pub fn cancel(&self) {
        self.progress.cancel.store(true, Ordering::Relaxed);
    }

    pub fn poll(&self) -> Poll<L> {
        match self.done.try_recv() {
            Ok(landing) => Poll::Finished(landing),
            Err(mpsc::TryRecvError::Empty) => Poll::Running,
            Err(mpsc::TryRecvError::Disconnected) => Poll::Lost,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Job, Poll, time_left};
    use std::time::{Duration, Instant};

    #[test]
    fn time_left_follows_the_pace_so_far_once_it_can_tell() {
        let ten = Duration::from_secs(10);
        assert_eq!(time_left(ten, 0.25), Some(Duration::from_secs(30)));
        assert_eq!(time_left(ten, 0.5), Some(ten));
        assert_eq!(time_left(Duration::from_millis(500), 0.5), None);
        assert_eq!(time_left(ten, 0.01), None);
        assert_eq!(time_left(ten, 1.0), None);
    }

    fn wait<L: Send + 'static>(job: &Job<L>) -> Poll<L> {
        let until = Instant::now() + Duration::from_secs(10);
        loop {
            match job.poll() {
                Poll::Running if Instant::now() < until => {
                    std::thread::sleep(Duration::from_millis(1));
                }
                other => return other,
            }
        }
    }

    #[test]
    fn a_job_reports_its_stage_and_hands_back_its_answer() {
        let (go, gate) = std::sync::mpsc::channel::<()>();
        let job = Job::start("Count", move |progress| {
            progress.stage("Counting");
            progress.fraction(0.5);
            let _ = gate.recv();
            7
        })
        .unwrap();
        let until = Instant::now() + Duration::from_secs(10);
        while job.view().fraction.is_none() && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(1));
        }
        let view = job.view();
        assert_eq!(
            (view.label.as_str(), view.stage.as_str(), view.fraction),
            ("Count", "Counting", Some(0.5))
        );
        go.send(()).unwrap();
        assert!(matches!(wait(&job), Poll::Finished(7)));
    }

    #[test]
    fn a_cancelled_job_sees_the_cancel_between_steps() {
        let job = Job::start("Spin", |progress| {
            let until = Instant::now() + Duration::from_secs(10);
            while !progress.cancelled() && Instant::now() < until {
                std::thread::sleep(Duration::from_millis(1));
            }
            progress.cancelled()
        })
        .unwrap();
        job.cancel();
        assert!(matches!(wait(&job), Poll::Finished(true)));
    }

    #[test]
    fn a_job_whose_thread_ends_without_an_answer_is_lost() {
        // The thread unwinds, so no answer is sent.
        let job: Job<u8> = Job::start("Fail", |_| std::panic::resume_unwind(Box::new(()))).unwrap();
        assert!(matches!(wait(&job), Poll::Lost));
    }
}
