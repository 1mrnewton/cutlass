//! Shared workers for software decode.
//!
//! Frame jobs each reconstruct one picture. Slice jobs split one picture.
//! [`Rows`] is how a picture waits until another worker has written a row it
//! still needs. One worker runs the job on the caller, which is the
//! single-thread picture check. The codec calls this crate. The engine does not.
//!
//! The workers are started once and kept. Frame jobs and
//! [`Pool::slices_owned`] run on them. [`Pool::slices`] lends the caller's
//! borrows to its pieces, which safe Rust allows only on threads that end
//! before the call returns, so it starts scoped threads for the call.

#![forbid(unsafe_code)]

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::mpsc::Receiver;
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
    shared: Option<Arc<Shared>>,
    joins: Vec<JoinHandle<()>>,
}

/// What the kept workers wait on.
struct Shared {
    state: Mutex<State>,
    /// A task was queued, or the pool is stopping.
    work: Condvar,
    /// A worker took a task off the queue.
    taken: Condvar,
}

struct State {
    queue: VecDeque<Task>,
    /// Tasks ever queued and ever taken. A frame job's sender waits until
    /// the count taken reaches its own place in the queue.
    pushed: u64,
    popped: u64,
    stop: bool,
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
    /// One past the last row reported, 0 before the first.
    through: AtomicU32,
    /// Workers blocked in [`Rows::wait`] or about to be. A report takes
    /// the lock and wakes them only when there are some, so the usual
    /// report and the usual wait touch no lock.
    waiters: AtomicU32,
    lock: Mutex<()>,
    ready: Condvar,
}

/// One [`Pool::slices_owned`] call, shared by the caller and its helpers.
struct Pieces<T, F> {
    work: F,
    count: usize,
    next: AtomicUsize,
    done: std::sync::mpsc::Sender<(usize, std::thread::Result<T>)>,
}

impl<T, F> Pieces<T, F>
where
    F: Fn(usize) -> T,
{
    /// Takes pieces until none are left. Every piece taken sends one result.
    fn run(&self) {
        loop {
            let index = self.next.fetch_add(1, Ordering::Relaxed);
            if index >= self.count {
                break;
            }
            let result = catch_unwind(AssertUnwindSafe(|| (self.work)(index)));
            let failed = result.is_err();
            let _ = self.done.send((index, result));
            if failed {
                // The caller unwinds on this result; leave the rest undone.
                self.next.store(self.count, Ordering::Relaxed);
                break;
            }
        }
    }
}

impl Pool {
    /// `workers` of 0 or 1 keeps every job on the caller. A larger count is
    /// capped at [`MAX_WORKERS`].
    pub fn new(workers: usize) -> Self {
        let limit = workers.clamp(1, MAX_WORKERS);
        let alone = Self {
            limit: 1,
            shared: None,
            joins: Vec::new(),
        };
        if limit == 1 {
            return alone;
        }
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                queue: VecDeque::new(),
                pushed: 0,
                popped: 0,
                stop: false,
            }),
            work: Condvar::new(),
            taken: Condvar::new(),
        });
        let mut joins = Vec::with_capacity(limit);
        for _ in 0..limit {
            let served = Arc::clone(&shared);
            match std::thread::Builder::new()
                .name(String::from("cutlass-pool"))
                .spawn(move || serve(&served))
            {
                Ok(handle) => joins.push(handle),
                Err(_) => {
                    stop(&shared, joins);
                    return alone;
                }
            }
        }
        Self {
            limit,
            shared: Some(shared),
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
    /// job runs before this call returns. With more, this call returns once
    /// a worker has taken the job. A frame job does not start another
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
        let Some(shared) = &self.shared else {
            task();
            return Join {
                rx: Some(result_rx),
            };
        };
        let mut state = lock(&shared.state);
        if state.stop {
            return Join { rx: None };
        }
        state.queue.push_back(task);
        state.pushed += 1;
        let ticket = state.pushed;
        shared.work.notify_one();
        while state.popped < ticket {
            state = wait(&shared.taken, state);
        }
        Join {
            rx: Some(result_rx),
        }
    }

    /// Run `count` pieces of one picture.
    ///
    /// Results come back in index order. The closure may borrow the picture
    /// for this call. With one worker the pieces run in order on the caller.
    /// More workers start threads for this call and join them before it
    /// returns; the caller takes pieces too, so one thread fewer than the
    /// worker count is started. A piece that panics continues on this call.
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
        let take = || {
            let mut done = Vec::new();
            loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= count {
                    break;
                }
                done.push((index, work(index)));
            }
            done
        };
        std::thread::scope(|scope| {
            let mut joins = Vec::with_capacity(workers - 1);
            for _ in 1..workers {
                joins.push(scope.spawn(take));
            }
            let mut paired = Vec::with_capacity(count);
            // A panic here unwinds through the scope, which joins the rest.
            paired.extend(take());
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

    /// Run `count` pieces of one picture on the kept workers.
    ///
    /// The same contract as [`Self::slices`], for a closure that owns what
    /// it reads, so no thread is started. The caller takes pieces too, and
    /// workers that are free join in; workers inside frame jobs do not, so
    /// this may be called from a frame job. It returns when every piece has
    /// finished. A piece that panics continues on this call.
    pub fn slices_owned<T, F>(&self, count: usize, work: F) -> Vec<T>
    where
        T: Send + 'static,
        F: Fn(usize) -> T + Send + Sync + 'static,
    {
        if count == 0 {
            return Vec::new();
        }
        let helpers = self.limit.min(count) - 1;
        let Some(shared) = self.shared.as_ref().filter(|_| helpers > 0) else {
            return (0..count).map(&work).collect();
        };
        let (done, results) = std::sync::mpsc::channel();
        let pieces = Arc::new(Pieces {
            work,
            count,
            next: AtomicUsize::new(0),
            done,
        });
        {
            let mut state = lock(&shared.state);
            // A helper that starts after the pieces are gone returns at
            // once. Helpers left waiting behind busy workers are capped so
            // repeated calls cannot grow the queue.
            let room = self.limit.saturating_sub(state.queue.len());
            for _ in 0..helpers.min(room) {
                let helper = Arc::clone(&pieces);
                state.queue.push_back(Box::new(move || helper.run()));
                state.pushed += 1;
                shared.work.notify_one();
            }
        }
        pieces.run();
        let mut slots: Vec<Option<T>> = (0..count).map(|_| None).collect();
        let mut left = count;
        while left > 0 {
            match results.recv() {
                Ok((index, Ok(value))) => {
                    if let Some(slot) = slots.get_mut(index) {
                        *slot = Some(value);
                    }
                    left -= 1;
                }
                Ok((_, Err(payload))) => std::panic::resume_unwind(payload),
                // Every sender is gone with pieces missing: a worker died.
                Err(_) => break,
            }
        }
        slots.into_iter().flatten().collect()
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        if let Some(shared) = self.shared.take() {
            stop(&shared, std::mem::take(&mut self.joins));
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
                through: AtomicU32::new(0),
                waiters: AtomicU32::new(0),
                lock: Mutex::new(()),
                ready: Condvar::new(),
            }),
        }
    }

    /// Rows `0..=row` are written. A later report does not move the mark backward.
    pub fn report_through(&self, row: u32) {
        let Some(target) = self.inner.target(row) else {
            return;
        };
        let gate = &*self.inner;
        // The mark goes up before the waiters are counted, and a waiter is
        // counted before it looks at the mark again under the lock: one of
        // the two sees the other.
        if gate.through.fetch_max(target + 1, Ordering::SeqCst) > target {
            return;
        }
        if gate.waiters.load(Ordering::SeqCst) > 0 {
            drop(lock(&gate.lock));
            gate.ready.notify_all();
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
        let gate = &*self.inner;
        if gate.through.load(Ordering::Acquire) > target {
            return;
        }
        gate.waiters.fetch_add(1, Ordering::SeqCst);
        let mut held = lock(&gate.lock);
        while gate.through.load(Ordering::SeqCst) <= target {
            held = match gate.ready.wait(held) {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
        }
        drop(held);
        gate.waiters.fetch_sub(1, Ordering::SeqCst);
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

fn serve(shared: &Shared) {
    let mut state = lock(&shared.state);
    loop {
        if let Some(task) = state.queue.pop_front() {
            state.popped += 1;
            drop(state);
            shared.taken.notify_all();
            task();
            state = lock(&shared.state);
        } else if state.stop {
            break;
        } else {
            state = wait(&shared.work, state);
        }
    }
}

fn stop(shared: &Shared, joins: Vec<JoinHandle<()>>) {
    lock(&shared.state).stop = true;
    shared.work.notify_all();
    for handle in joins {
        let _ = handle.join();
    }
}

fn wait<'a, T>(ready: &Condvar, held: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
    match ready.wait(held) {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
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
    fn a_panic_in_a_slice_continues_on_the_caller() {
        let pool = Pool::new(3);
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.slices(6, |index| assert!(index != 4, "piece four"))
        }));
        assert!(caught.is_err());
        assert_eq!(pool.slices(3, |index| index), vec![0, 1, 2]);
    }

    #[test]
    fn owned_slices_come_back_in_index_order() {
        for workers in [1, 2, 4, 16] {
            let pool = Pool::new(workers);
            for count in [0usize, 1, 2, 7, 100] {
                let values = pool.slices_owned(count, |index| index * 3);
                assert_eq!(values, (0..count).map(|i| i * 3).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn one_worker_runs_owned_slices_in_order_on_the_caller() {
        let pool = Pool::new(1);
        let caller = std::thread::current().id();
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = Arc::clone(&order);
        let _ = pool.slices_owned(5, move |index| {
            assert_eq!(std::thread::current().id(), caller);
            seen.lock().expect("order").push(index);
        });
        assert_eq!(*order.lock().expect("order"), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn owned_slices_run_on_the_kept_workers() {
        let pool = Pool::new(4);
        if pool.workers() < 4 {
            return;
        }
        // Four pieces that each wait for the other three: the caller and
        // three kept workers must all be inside one at once.
        let barrier = Arc::new(Barrier::new(4));
        let names = pool.slices_owned(4, move |_| {
            let _ = barrier.wait();
            std::thread::current().name().map(str::to_owned)
        });
        let kept = names
            .iter()
            .filter(|name| name.as_deref() == Some("cutlass-pool"))
            .count();
        assert_eq!(kept, 3);
    }

    #[test]
    fn owned_slices_repeat_without_starting_threads() {
        let pool = Pool::new(4);
        let ids = Arc::new(std::sync::Mutex::new(std::collections::HashSet::new()));
        for _ in 0..200 {
            let seen = Arc::clone(&ids);
            let _ = pool.slices_owned(16, move |_| {
                seen.lock()
                    .expect("ids")
                    .insert(std::thread::current().id());
            });
        }
        // The caller and at most the pool's own workers, however many calls.
        assert!(ids.lock().expect("ids").len() <= pool.workers() + 1);
    }

    #[test]
    fn a_panic_in_an_owned_slice_continues_on_the_caller() {
        let pool = Pool::new(3);
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.slices_owned(6, |index| assert!(index != 2, "piece two"))
        }));
        assert!(caught.is_err());
        assert_eq!(pool.slices_owned(3, |index| index), vec![0, 1, 2]);
    }

    #[test]
    fn owned_slices_finish_inside_a_frame_job_with_every_worker_busy() {
        let pool = Arc::new(Pool::new(2));
        if pool.workers() < 2 {
            return;
        }
        // Both workers are inside frame jobs; each splits its picture. No
        // worker is free to help, so each frame job runs its own pieces.
        let barrier = Arc::new(Barrier::new(2));
        let joins: Vec<_> = (0..2)
            .map(|_| {
                let inner = Arc::clone(&pool);
                let barrier = Arc::clone(&barrier);
                pool.spawn_frame(move || {
                    let _ = barrier.wait();
                    inner.slices_owned(8, |index| index).len()
                })
            })
            .collect();
        for join in joins {
            assert_eq!(join.wait().ok(), Some(8));
        }
        // Helpers left queued behind the frame jobs are harmless afterwards.
        assert_eq!(pool.spawn_frame(|| 5).wait().ok(), Some(5));
    }

    #[test]
    fn spawn_returns_only_when_a_worker_has_the_job() {
        let pool = Pool::new(2);
        if pool.workers() < 2 {
            return;
        }
        let running = Arc::new(AtomicUsize::new(0));
        let release = Arc::new(Barrier::new(3));
        let joins: Vec<_> = (0..2)
            .map(|_| {
                let running = Arc::clone(&running);
                let release = Arc::clone(&release);
                pool.spawn_frame(move || {
                    running.fetch_add(1, Ordering::SeqCst);
                    let _ = release.wait();
                })
            })
            .collect();
        // Both jobs were taken by the time the second spawn returned, so a
        // third thread at the barrier lets them through.
        let _ = release.wait();
        for join in joins {
            assert!(join.wait().is_ok());
        }
        assert_eq!(running.load(Ordering::SeqCst), 2);
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
    fn waiters_wake_for_every_report() {
        // Rows reported one at a time while three threads wait for each in
        // turn: a lost wake-up hangs here.
        for _ in 0..20 {
            let rows = Rows::new(2000);
            std::thread::scope(|scope| {
                for _ in 0..3 {
                    let rows = rows.clone();
                    scope.spawn(move || {
                        for row in 0..2000 {
                            rows.wait(row);
                        }
                    });
                }
                for row in 0..2000 {
                    rows.report_through(row);
                    if row % 64 == 0 {
                        std::thread::yield_now();
                    }
                }
            });
        }
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
