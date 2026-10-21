//! Output handoff from VideoToolbox callback threads to the caller.

use std::collections::VecDeque;
use std::sync::{Mutex, MutexGuard, PoisonError};

/// A FIFO the output callback pushes to and the caller pops from. The lock is
/// held only for a push or a pop.
pub(crate) struct Queue<T>(Mutex<VecDeque<T>>);

impl<T> Queue<T> {
    pub(crate) fn new() -> Self {
        Self(Mutex::new(VecDeque::new()))
    }

    pub(crate) fn push(&self, item: T) {
        self.lock().push_back(item);
    }

    pub(crate) fn pop(&self) -> Option<T> {
        self.lock().pop_front()
    }

    pub(crate) fn clear(&self) {
        self.lock().clear();
    }

    // Poisoning can't leave the deque half-updated: no code that holds the
    // lock can panic partway through a push or pop.
    fn lock(&self) -> MutexGuard<'_, VecDeque<T>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}
