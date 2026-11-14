//! Worker threads that reconstruct frames while the caller parses the
//! next one.
//!
//! Jobs start in the order they are sent, so a frame's reconstruction is
//! always running before a later frame's, which may wait on its rows; a
//! worker that waits never waits on a job no worker has started.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

type Job = Box<dyn FnOnce() + Send + 'static>;

pub struct Workers {
    tx: Option<Sender<Job>>,
    joins: Vec<JoinHandle<()>>,
    count: usize,
    /// Set when a job panicked: its picture is not to be trusted.
    failed: Arc<AtomicBool>,
}

impl Workers {
    /// `count` threads; `None` when none could start.
    pub fn new(count: usize) -> Option<Self> {
        let (tx, rx) = channel::<Job>();
        let rx = Arc::new(Mutex::new(rx));
        let failed = Arc::new(AtomicBool::new(false));
        let mut joins = Vec::with_capacity(count);
        for _ in 0..count {
            let rx: Arc<Mutex<Receiver<Job>>> = Arc::clone(&rx);
            let failed = Arc::clone(&failed);
            let spawned = std::thread::Builder::new()
                .name(String::from("cutlass-vp9"))
                .spawn(move || {
                    loop {
                        let job = {
                            let guard = match rx.lock() {
                                Ok(g) => g,
                                Err(p) => p.into_inner(),
                            };
                            guard.recv()
                        };
                        match job {
                            Ok(job) => {
                                if catch_unwind(AssertUnwindSafe(job)).is_err() {
                                    failed.store(true, Ordering::Release);
                                }
                            }
                            Err(_) => break,
                        }
                    }
                });
            match spawned {
                Ok(j) => joins.push(j),
                Err(_) => break,
            }
        }
        if joins.is_empty() {
            return None;
        }
        Some(Self {
            tx: Some(tx),
            count: joins.len(),
            joins,
            failed,
        })
    }

    pub fn count(&self) -> usize {
        self.count
    }

    /// Whether a job has panicked since the last call.
    pub fn take_failure(&self) -> bool {
        self.failed.swap(false, Ordering::AcqRel)
    }

    /// Runs `job` on a worker, or here if the workers are gone.
    pub fn run(&self, job: impl FnOnce() + Send + 'static) {
        let job: Job = Box::new(job);
        match &self.tx {
            Some(tx) => {
                if let Err(back) = tx.send(job) {
                    (back.0)();
                }
            }
            None => job(),
        }
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        self.tx.take();
        for j in self.joins.drain(..) {
            let _ = j.join();
        }
    }
}
