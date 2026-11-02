//! Shared-mode AAudio output.
//!
//! The data callback walks the stored slice on AAudio's thread. It does not
//! wait on the slice lock: a period that arrives while the slice is replaced
//! is silence. Past the end of the slice, the rest of the buffer is silence.
//! An error callback only sets a flag. Stopping and closing happen on the
//! thread that owns the queue.

#![allow(unsafe_code)]

use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::slice::{QueueError, Slice, layout};

const OK: i32 = 0;
const DIRECTION_OUTPUT: i32 = 0;
const SHARING_SHARED: i32 = 1;
const FORMAT_FLOAT: i32 = 2;
const CONTINUE: i32 = 0;
const STOP: i32 = 1;
/// `AAUDIO_STREAM_STATE_STOPPED`, from the header's enumerator order.
const STATE_STOPPED: i32 = 10;
/// `AAUDIO_STREAM_STATE_CLOSED`.
const STATE_CLOSED: i32 = 12;
/// `AAUDIO_STREAM_STATE_DISCONNECTED`.
const STATE_DISCONNECTED: i32 = 13;
const STOP_WAIT_NS: i64 = 50_000_000;

#[link(name = "aaudio")]
unsafe extern "C" {
    fn AAudio_createStreamBuilder(builder: *mut *mut c_void) -> i32;
    fn AAudioStreamBuilder_setDirection(builder: *mut c_void, direction: i32);
    fn AAudioStreamBuilder_setSharingMode(builder: *mut c_void, mode: i32);
    fn AAudioStreamBuilder_setSampleRate(builder: *mut c_void, rate: i32);
    fn AAudioStreamBuilder_setChannelCount(builder: *mut c_void, channels: i32);
    fn AAudioStreamBuilder_setFormat(builder: *mut c_void, format: i32);
    fn AAudioStreamBuilder_setDataCallback(
        builder: *mut c_void,
        callback: DataCallback,
        user: *mut c_void,
    );
    fn AAudioStreamBuilder_setErrorCallback(
        builder: *mut c_void,
        callback: ErrorCallback,
        user: *mut c_void,
    );
    fn AAudioStreamBuilder_openStream(builder: *mut c_void, stream: *mut *mut c_void) -> i32;
    fn AAudioStreamBuilder_delete(builder: *mut c_void) -> i32;
    fn AAudioStream_requestStart(stream: *mut c_void) -> i32;
    fn AAudioStream_requestStop(stream: *mut c_void) -> i32;
    fn AAudioStream_close(stream: *mut c_void) -> i32;
    fn AAudioStream_getState(stream: *mut c_void) -> i32;
    fn AAudioStream_waitForStateChange(
        stream: *mut c_void,
        input: i32,
        next: *mut i32,
        timeout_ns: i64,
    ) -> i32;
    fn AAudioStream_getSampleRate(stream: *mut c_void) -> i32;
    fn AAudioStream_getChannelCount(stream: *mut c_void) -> i32;
    fn AAudioStream_getFormat(stream: *mut c_void) -> i32;
}

type DataCallback = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, i32) -> i32;
type ErrorCallback = unsafe extern "C" fn(*mut c_void, *mut c_void, i32);

struct Shared {
    slice: Slice,
    failed: AtomicBool,
    inside: AtomicUsize,
}

/// A shared-mode AAudio output. The callback walks the stored slice.
pub struct OutputQueue {
    stream: *mut c_void,
    shared: Arc<Shared>,
}

impl OutputQueue {
    /// Open the default shared-mode output at `rate` hertz and `channels`.
    pub fn open(rate: u32, channels: u16) -> Result<Self, QueueError> {
        let frame = layout(rate, channels)?;
        let slice = Slice::new(frame)?;
        if slice.stride() != frame {
            return Err(QueueError::Format);
        }
        let shared = Arc::new(Shared {
            slice,
            failed: AtomicBool::new(false),
            inside: AtomicUsize::new(0),
        });
        let user = Arc::as_ptr(&shared).cast_mut().cast::<c_void>();
        let stream = open_stream(user)?;
        Ok(Self { stream, shared })
    }

    /// Replace the slice the callback walks. An empty slice is silence.
    pub fn set_sample(&self, sample: &[u8]) -> Result<(), QueueError> {
        self.shared.slice.set_sample(sample)
    }
}

impl Drop for OutputQueue {
    fn drop(&mut self) {
        if self.stream.is_null() {
            return;
        }
        let stopped = shutdown(self.stream);
        self.stream = ptr::null_mut();
        if !stopped || self.shared.inside.load(Ordering::Acquire) != 0 {
            keep_alive(Arc::clone(&self.shared));
        }
    }
}

fn open_stream(user: *mut c_void) -> Result<*mut c_void, QueueError> {
    let mut builder = ptr::null_mut();
    let created = unsafe { AAudio_createStreamBuilder(&mut builder) };
    if created != OK || builder.is_null() {
        return Err(QueueError::NoDevice);
    }
    unsafe {
        AAudioStreamBuilder_setDirection(builder, DIRECTION_OUTPUT);
        AAudioStreamBuilder_setSharingMode(builder, SHARING_SHARED);
        AAudioStreamBuilder_setSampleRate(builder, 48_000);
        AAudioStreamBuilder_setChannelCount(builder, 2);
        AAudioStreamBuilder_setFormat(builder, FORMAT_FLOAT);
        AAudioStreamBuilder_setDataCallback(builder, data_callback, user);
        AAudioStreamBuilder_setErrorCallback(builder, error_callback, user);
    }
    let mut stream = ptr::null_mut();
    let opened = unsafe { AAudioStreamBuilder_openStream(builder, &mut stream) };
    unsafe {
        let _ = AAudioStreamBuilder_delete(builder);
    }
    if opened != OK || stream.is_null() {
        return Err(QueueError::NoDevice);
    }
    if !preview_format(stream) {
        shutdown(stream);
        return Err(QueueError::NoDevice);
    }
    if unsafe { AAudioStream_requestStart(stream) } != OK {
        shutdown(stream);
        return Err(QueueError::NoDevice);
    }
    Ok(stream)
}

fn preview_format(stream: *mut c_void) -> bool {
    unsafe {
        AAudioStream_getSampleRate(stream) == 48_000
            && AAudioStream_getChannelCount(stream) == 2
            && AAudioStream_getFormat(stream) == FORMAT_FLOAT
    }
}

/// Stop, wait until the callback should be finished, then close.
///
/// Returns whether the stream reported a terminal state before the wait ended.
fn shutdown(stream: *mut c_void) -> bool {
    if stream.is_null() {
        return true;
    }
    unsafe {
        let _ = AAudioStream_requestStop(stream);
        let mut state = AAudioStream_getState(stream);
        for _ in 0..4 {
            if terminal(state) {
                let _ = AAudioStream_close(stream);
                return true;
            }
            let mut next = state;
            let result = AAudioStream_waitForStateChange(stream, state, &mut next, STOP_WAIT_NS);
            if result != OK {
                break;
            }
            state = next;
        }
        let stopped = terminal(state);
        let _ = AAudioStream_close(stream);
        stopped
    }
}

fn terminal(state: i32) -> bool {
    state == STATE_STOPPED || state == STATE_CLOSED || state == STATE_DISCONNECTED
}

/// A callback that outlives `close` still has a live slice.
fn keep_alive(shared: Arc<Shared>) {
    static KEPT: Mutex<Vec<Arc<Shared>>> = Mutex::new(Vec::new());
    match KEPT.lock() {
        Ok(mut kept) => kept.push(shared),
        Err(poisoned) => poisoned.into_inner().push(shared),
    }
}

unsafe extern "C" fn data_callback(
    _stream: *mut c_void,
    user: *mut c_void,
    audio: *mut c_void,
    frames: i32,
) -> i32 {
    if user.is_null() || audio.is_null() || frames <= 0 {
        return CONTINUE;
    }
    let shared = unsafe { &*user.cast::<Shared>() };
    shared.inside.fetch_add(1, Ordering::AcqRel);
    let result = write_frames(shared, audio, frames);
    shared.inside.fetch_sub(1, Ordering::AcqRel);
    result
}

fn write_frames(shared: &Shared, audio: *mut c_void, frames: i32) -> i32 {
    let Ok(count) = usize::try_from(frames) else {
        return STOP;
    };
    let Some(bytes) = count.checked_mul(shared.slice.stride()) else {
        return STOP;
    };
    let out = unsafe { std::slice::from_raw_parts_mut(audio.cast::<u8>(), bytes) };
    if shared.failed.load(Ordering::Acquire) {
        out.fill(0);
        return STOP;
    }
    shared.slice.pull_now(out);
    CONTINUE
}

unsafe extern "C" fn error_callback(_stream: *mut c_void, user: *mut c_void, _error: i32) {
    if user.is_null() {
        return;
    }
    let shared = unsafe { &*user.cast::<Shared>() };
    shared.failed.store(true, Ordering::Release);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "opens the default AAudio output"]
    fn a_queue_opens_on_the_default_device() {
        let queue = OutputQueue::open(48_000, 2).expect("device");
        queue.set_sample(&[0_u8; 8]).expect("sample");
    }
}
