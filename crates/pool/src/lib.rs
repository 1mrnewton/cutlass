//! Shared workers for software decode.
//!
//! Frame jobs each reconstruct one picture. Slice jobs split one picture.
//! [`Rows`] is how a picture waits until another worker has written a row it
//! still needs. One worker runs the job on the caller, which is the
//! single-thread picture check. The codec calls this crate. The engine does not.

#![forbid(unsafe_code)]

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread::JoinHandle;

/// Upper bound for [`Pool::shared`]. Matches the software decoder cap of 16.
const MAX_WORKERS: usize = 16;

type Task = Box<dyn FnOnce() + Send + 'static>;

/// The pool stopped before the job finished.
#[derive(Debug)]
pub struct PoolStopped;

impl std::fmt::Display for PoolStopped {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("pool stopped")
    }
}

impl std::error::Error for PoolStopped {}

/// Frame jobs, slice jobs, and the worker budget they share.
pub struct Pool {
    limit: usize,
    tx: Option<SyncSender<Task>>,
    joins: Vec<JoinHandle<()>>,
}

/// A frame job that has been handed to the pool.
#[must_use]
pub struct Join<T> {
    rx: Option<Receiver<std::thread::Result<T>>>,
}

/// How far a picture has been written, so another frame can wait for a row.
#[derive(Clone)]
pub struct Rows {
    inner: Arc<RowGate>,
}

struct RowGate {
    height: u32,
    through: Mutex<Option<u32>>,
    ready: Condvar,
}

impl Pool {
    /// `workers` of 0 or 1 keeps every job on the caller. A larger count is
    /// capped at [`MAX_WORKERS`].
    pub fn new(workers: usize) -> Self {
        let limit = workers.clamp(1, MAX_WORKERS);
        if limit == 1 {
            return Self {
                limit,
                tx: None,
                joins: Vec::new(),
            };
        }
        let (tx, rx) = sync_channel(0);
        let rx = Arc::new(Mutex::new(rx));
        let mut joins = Vec::with_capacity(limit);
        for _ in 0..limit {
            let rx = Arc::clone(&rx);
            match std::thread::Builder::new()
                .name(String::from("cutlass-pool"))
                .spawn(move || serve(rx))
            {
                Ok(handle) => joins.push(handle),
                Err(_) => {
                    drop(tx);
                    for handle in joins {
                        let _ = handle.join();
                    }
                    return Self {
                        limit: 1,
                        tx: None,
                        joins: Vec::new(),
                    };
                }
            }
        }
        Self {
            limit,
            tx: Some(tx),
            joins,
        }
    }

    /// The process pool. Its size is the machine's parallelism, at most 16.
    pub fn shared() -> &'static Self {
        static SHARED: OnceLock<Pool> = OnceLock::new();
        SHARED.get_or_init(|| Self::new(default_workers()))
    }

    /// Workers this pool will run at once. One means the caller.
    pub fn workers(&self) -> usize {
        self.limit
    }

    /// Reconstruct one picture on a frame worker.
    ///
    /// The job owns its packet and its decoder. A later job waits on [`Rows`]
    /// before it reads a row this job is still writing. With one worker the
    /// job runs before this call returns. A frame job does not start another
    /// frame job. The workers are already inside frame jobs, so the new one
    /// would wait forever.
    pub fn spawn_frame<T>(&self, job: impl FnOnce() -> T + Send + 'static) -> Join<T>
    where
        T: Send + 'static,
    {
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        let task: Task = Box::new(move || {
            let result = catch_unwind(AssertUnwindSafe(job));
            let _ = result_tx.send(result);
        });
        if let Some(tx) = &self.tx {
            if tx.send(task).is_err() {
                return Join { rx: None };
            }
            return Join {
                rx: Some(result_rx),
            };
        }
        task();
        Join {
            rx: Some(result_rx),
        }
    }

    /// Run `count` pieces of one picture.
    ///
    /// Results come back in index order. The closure may borrow the picture
    /// for this call. With one worker the pieces run in order on the caller.
    /// More workers start threads for this call and join them before it returns.
    pub fn slices<T, F>(&self, count: usize, work: F) -> Vec<T>
    where
        T: Send,
        F: Fn(usize) -> T + Sync,
    {
        if count == 0 {
            return Vec::new();
        }
        let workers = self.limit.min(count);
        if workers == 1 {
            return (0..count).map(&work).collect();
        }
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            let mut joins = Vec::with_capacity(workers);
            for _ in 0..workers {
                joins.push(scope.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        if index >= count {
                            break;
                        }
                        done.push((index, work(index)));
                    }
                    done
                }));
            }
            let mut paired = Vec::with_capacity(count);
            for join in joins {
                match join.join() {
                    Ok(done) => paired.extend(done),
                    Err(payload) => std::panic::resume_unwind(payload),
                }
            }
            paired.sort_by_key(|(index, _)| *index);
            paired.into_iter().map(|(_, value)| value).collect()
        })
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.tx.take();
        for handle in self.joins.drain(..) {
            let _ = handle.join();
        }
    }
}

impl<T> Join<T> {
    /// Block until the frame job returns.
    ///
    /// A panic inside the job continues on this call.
    pub fn wait(self) -> Result<T, PoolStopped> {
        let Some(rx) = self.rx else {
            return Err(PoolStopped);
        };
        match rx.recv() {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(payload)) => std::panic::resume_unwind(payload),
            Err(_) => Err(PoolStopped),
        }
    }
}

impl Rows {
    /// `height` is the number of rows the picture will report. Zero rows are
    /// already finished.
    pub fn new(height: u32) -> Self {
        Self {
            inner: Arc::new(RowGate {
                height,
                through: Mutex::new(None),
                ready: Condvar::new(),
            }),
        }
    }

    /// Rows `0..=row` are written. A later report does not move the mark backward.
    pub fn report_through(&self, row: u32) {
        let Some(target) = self.inner.target(row) else {
            return;
        };
        let mut through = lock(&self.inner.through);
        let advanced = match *through {
            Some(done) if done >= target => false,
            _ => {
                *through = Some(target);
                true
            }
        };
        drop(through);
        if advanced {
            self.inner.ready.notify_all();
        }
    }

    /// Block until `row` has been reported.
    ///
    /// The worker that reports must not be the worker that waits. A row past
    /// the last one waits for the picture to finish.
    pub fn wait(&self, row: u32) {
        let Some(target) = self.inner.target(row) else {
            return;
        };
        let mut through = lock(&self.inner.through);
        while through.is_none_or(|done| done < target) {
            through = match self.inner.ready.wait(through) {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
        }
    }
}

impl RowGate {
    fn target(&self, row: u32) -> Option<u32> {
        if self.height == 0 {
            None
        } else {
            Some(row.min(self.height - 1))
        }
    }
}

fn serve(rx: Arc<Mutex<Receiver<Task>>>) {
    loop {
        let job = {
            let guard = lock(&rx);
            guard.recv()
        };
        match job {
            Ok(job) => job(),
            Err(_) => break,
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn default_workers() -> usize {
    match std::thread::available_parallelism() {
        Ok(count) => count.get().min(MAX_WORKERS),
        Err(_) => 1,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};

    use super::{Pool, Rows};

    #[test]
    fn one_worker_runs_slices_in_order_on_the_caller() {
        let pool = Pool::new(1);
        assert_eq!(pool.workers(), 1);
        let current = AtomicUsize::new(0);
        let max_in_flight = AtomicUsize::new(0);
        let values = pool.slices(4, |index| {
            let now = current.fetch_add(1, Ordering::SeqCst) + 1;
            max_in_flight.fetch_max(now, Ordering::SeqCst);
            current.fetch_sub(1, Ordering::SeqCst);
            index * 2
        });
        assert_eq!(values, vec![0, 2, 4, 6]);
        assert_eq!(max_in_flight.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn slice_results_stay_in_index_order_when_workers_overlap() {
        let pool = Pool::new(4);
        if pool.workers() < 2 {
            return;
        }
        let values = pool.slices(8, |index| index);
        assert_eq!(values, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn slices_overlap_across_the_worker_budget() {
        let pool = Pool::new(4);
        if pool.workers() < 4 {
            return;
        }
        let current = AtomicUsize::new(0);
        let max_in_flight = AtomicUsize::new(0);
        let barrier = Barrier::new(4);
        let _ = pool.slices(4, |_| {
            let now = current.fetch_add(1, Ordering::SeqCst) + 1;
            max_in_flight.fetch_max(now, Ordering::SeqCst);
            let _ = barrier.wait();
            current.fetch_sub(1, Ordering::SeqCst);
        });
        assert!(max_in_flight.load(Ordering::SeqCst) >= 2);
    }

    #[test]
    fn an_empty_slice_list_does_no_work() {
        let pool = Pool::new(4);
        let values = pool.slices(0, |_| -> i32 { 1 });
        assert!(values.is_empty());
    }

    #[test]
    fn one_worker_runs_a_frame_job_before_spawn_returns() {
        let pool = Pool::new(1);
        let join = pool.spawn_frame(|| 7);
        assert_eq!(join.wait().ok(), Some(7));
    }

    #[test]
    fn frame_jobs_overlap() {
        let pool = Pool::new(2);
        if pool.workers() < 2 {
            return;
        }
        let barrier = Arc::new(Barrier::new(2));
        let first_barrier = Arc::clone(&barrier);
        let second_barrier = Arc::clone(&barrier);
        let first = pool.spawn_frame(move || {
            let _ = first_barrier.wait();
            1
        });
        let second = pool.spawn_frame(move || {
            let _ = second_barrier.wait();
            2
        });
        let mut values = [first.wait().ok(), second.wait().ok()];
        values.sort_unstable();
        assert_eq!(values, [Some(1), Some(2)]);
    }

    #[test]
    fn a_row_wait_blocks_until_that_row_is_reported() {
        let rows = Rows::new(8);
        let waiting = rows.clone();
        let started = Arc::new(AtomicUsize::new(0));
        let entered = Arc::clone(&started);
        let finished = Arc::new(AtomicUsize::new(0));
        let done = Arc::clone(&finished);
        std::thread::scope(|scope| {
            scope.spawn(move || {
                entered.store(1, Ordering::SeqCst);
                waiting.wait(3);
                done.store(1, Ordering::SeqCst);
            });
            while started.load(Ordering::SeqCst) == 0 {
                std::thread::yield_now();
            }
            for _ in 0..64 {
                std::thread::yield_now();
            }
            rows.report_through(2);
            assert_eq!(finished.load(Ordering::SeqCst), 0);
            rows.report_through(3);
        });
        assert_eq!(finished.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_row_that_is_already_written_does_not_block() {
        let rows = Rows::new(4);
        rows.report_through(1);
        rows.wait(0);
        rows.wait(1);
        rows.report_through(0);
        rows.wait(1);
    }

    #[test]
    fn a_picture_with_no_rows_is_already_finished() {
        let rows = Rows::new(0);
        rows.wait(0);
        rows.report_through(3);
        rows.wait(3);
    }

    #[test]
    fn the_shared_pool_stays_inside_the_cap() {
        let pool = Pool::shared();
        assert!(pool.workers() >= 1);
        assert!(pool.workers() <= 16);
    }
}
