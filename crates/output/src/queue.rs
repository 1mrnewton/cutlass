//! An AudioQueue that plays interleaved `f32`.
//!
//! The callback runs on the queue's thread. It copies one buffer and returns.
//! `play` is called from the owner thread.

use std::ffi::c_void;
use std::ptr::{self, NonNull};
use std::sync::{Arc, Mutex, MutexGuard};

use cutlass_core::AudioSamples;
use objc2_audio_toolbox::{
    AudioQueueAllocateBuffer, AudioQueueBufferRef, AudioQueueDispose, AudioQueueEnqueueBuffer,
    AudioQueueNewOutput, AudioQueueRef, AudioQueueStart, AudioQueueStop,
    kAudioQueueErr_InvalidRunState,
};
use objc2_core_audio_types::{
    AudioStreamBasicDescription, kAudioFormatFlagIsFloat, kAudioFormatFlagIsPacked,
    kAudioFormatLinearPCM,
};

use crate::error::{self, Error};

const BUFFERS: usize = 3;
const BUFFER_FRAMES: u32 = 1024;

struct Shared {
    samples: Vec<f32>,
    frame: usize,
    frames: usize,
    channels: u16,
    /// Frames copied into a device buffer.
    played: usize,
    /// When set, a returned buffer is not filled again.
    closed: bool,
}

/// Plays planar PCM on the default output device.
pub struct Output {
    queue: AudioQueueRef,
    buffers: Vec<AudioQueueBufferRef>,
    shared: Arc<Mutex<Shared>>,
    sample_rate: u32,
    channels: u16,
}

// SAFETY: the queue is used from the owner thread and from its own callback.
// The callback only touches `shared`. Every owner call takes `&mut self` or
// `&self`, and the type is not `Sync`.
unsafe impl Send for Output {}

impl Output {
    /// Opens the default output device at `sample_rate` for `channels`.
    ///
    /// Stereo is left, then right. `channels` is at most 8.
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self, Error> {
        check_layout(sample_rate, channels)?;
        let shared = Arc::new(Mutex::new(Shared {
            samples: Vec::new(),
            frame: 0,
            frames: 0,
            channels,
            played: 0,
            closed: true,
        }));
        let mut format = pcm(sample_rate, channels);
        let mut queue = ptr::null_mut();
        // SAFETY: `format` is live for the call, `shared` outlives the queue
        // because `Output` drops the queue first, and `queue` is a place for
        // the created queue. A null run loop asks AudioQueue for its own thread.
        let status = unsafe {
            AudioQueueNewOutput(
                NonNull::from(&mut format),
                Some(completed),
                Arc::as_ptr(&shared).cast::<c_void>().cast_mut(),
                None,
                None,
                0,
                NonNull::from(&mut queue),
            )
        };
        error::check("AudioQueueNewOutput", status)?;
        if queue.is_null() {
            return Err(Error::Os {
                call: "AudioQueueNewOutput",
                status: 0,
            });
        }
        let mut output = Self {
            queue,
            buffers: Vec::with_capacity(BUFFERS),
            shared,
            sample_rate,
            channels,
        };
        let bytes = BUFFER_FRAMES
            .checked_mul(u32::from(channels))
            .and_then(|frames| frames.checked_mul(4))
            .ok_or(Error::InvalidInput("audio buffer"))?;
        for _ in 0..BUFFERS {
            let mut buffer = ptr::null_mut();
            // SAFETY: `queue` is the queue just created, and `buffer` is a
            // place for the allocated buffer.
            let status = unsafe {
                AudioQueueAllocateBuffer(output.queue, bytes, NonNull::from(&mut buffer))
            };
            error::check("AudioQueueAllocateBuffer", status)?;
            if buffer.is_null() {
                return Err(Error::Os {
                    call: "AudioQueueAllocateBuffer",
                    status: 0,
                });
            }
            output.buffers.push(buffer);
        }
        Ok(output)
    }

    /// Replaces anything still queued and starts playback.
    ///
    /// An empty buffer stops the device. Frames still inside the previous
    /// buffers are dropped.
    pub fn play(&mut self, samples: &AudioSamples) -> Result<(), Error> {
        if samples.sample_rate() != self.sample_rate || samples.channels() != self.channels {
            return Err(Error::InvalidInput("audio samples do not match the output"));
        }
        {
            let mut shared = self.lock()?;
            shared.closed = true;
            shared.samples = samples.interleaved();
            shared.frame = 0;
            shared.frames = samples.frames();
            shared.played = 0;
        }
        self.stop_queue()?;
        {
            let mut shared = self.lock()?;
            shared.closed = false;
        }
        if samples.frames() == 0 {
            return Ok(());
        }
        let buffers = self.buffers.clone();
        for buffer in buffers {
            self.enqueue(buffer)?;
        }
        self.start()
    }

    /// Frames copied into a device buffer.
    ///
    /// This is the handoff, not how far the speaker has gotten.
    pub fn enqueued_frames(&self) -> usize {
        self.lock().map(|shared| shared.played).unwrap_or(0)
    }

    fn enqueue(&mut self, buffer: AudioQueueBufferRef) -> Result<(), Error> {
        let give = {
            let mut shared = self.lock()?;
            if shared.closed {
                return Ok(());
            }
            copy_into(buffer, &mut shared)
        };
        if give == 0 {
            return Ok(());
        }
        // SAFETY: `buffer` was allocated by this queue and `copy_into` just
        // filled it. The sample lock is not held, so a callback can run.
        let status = unsafe { AudioQueueEnqueueBuffer(self.queue, buffer, 0, ptr::null()) };
        error::check("AudioQueueEnqueueBuffer", status)
    }

    fn start(&self) -> Result<(), Error> {
        // SAFETY: `queue` is the queue this value owns. A null start time
        // means start now.
        let status = unsafe { AudioQueueStart(self.queue, ptr::null()) };
        error::check("AudioQueueStart", status)
    }

    fn stop_queue(&self) -> Result<(), Error> {
        // SAFETY: `queue` is the queue this value owns. Stopping immediately
        // returns after the current callback, which only copies one buffer.
        let status = unsafe { AudioQueueStop(self.queue, true) };
        if status == kAudioQueueErr_InvalidRunState {
            return Ok(());
        }
        error::check("AudioQueueStop", status)
    }

    fn lock(&self) -> Result<MutexGuard<'_, Shared>, Error> {
        self.shared.lock().map_err(|_| Error::Poisoned)
    }
}

impl Drop for Output {
    fn drop(&mut self) {
        if let Ok(mut shared) = self.shared.lock() {
            shared.closed = true;
        }
        // SAFETY: `queue` is the queue this value created. Stop finishes the
        // current callback. Dispose then guarantees no further callbacks, so
        // the sample buffer can drop with this value.
        unsafe {
            let _ = AudioQueueStop(self.queue, true);
            let _ = AudioQueueDispose(self.queue, true);
        }
    }
}

/// # Safety
///
/// `user` points at the `Mutex<Shared>` inside the `Arc` that `Output` holds
/// for this queue. `buffer` is a buffer that queue just finished playing.
unsafe extern "C-unwind" fn completed(
    user: *mut c_void,
    queue: AudioQueueRef,
    buffer: AudioQueueBufferRef,
) {
    // SAFETY: `Output` created the queue with this pointer and disposes the
    // queue before dropping the `Arc`.
    let shared = unsafe { &*user.cast::<Mutex<Shared>>() };
    let Ok(mut shared) = shared.lock() else {
        return;
    };
    if shared.closed {
        return;
    }
    let give = copy_into(buffer, &mut shared);
    drop(shared);
    if give == 0 {
        return;
    }
    // SAFETY: `buffer` belongs to `queue` and was just returned by it.
    unsafe {
        let _ = AudioQueueEnqueueBuffer(queue, buffer, 0, ptr::null());
    }
}

fn copy_into(buffer: AudioQueueBufferRef, shared: &mut Shared) -> usize {
    if buffer.is_null() {
        return 0;
    }
    // SAFETY: the queue allocated `buffer`, and the caller holds it: either
    // the owner before the first enqueue, or the callback after the device
    // returned it.
    let buffer = unsafe { &mut *buffer };
    let channels = usize::from(shared.channels);
    let frame_bytes = channels.saturating_mul(4);
    if channels == 0 || frame_bytes == 0 {
        return 0;
    }
    let capacity = usize::try_from(buffer.mAudioDataBytesCapacity).unwrap_or(0);
    let room = capacity / frame_bytes;
    let left = shared.frames.saturating_sub(shared.frame);
    let give = room.min(left);
    if give == 0 {
        buffer.mAudioDataByteSize = 0;
        return 0;
    }
    let start = shared.frame.saturating_mul(channels);
    let floats = give.saturating_mul(channels);
    let Some(end) = start.checked_add(floats) else {
        return 0;
    };
    if end > shared.samples.len() {
        return 0;
    }
    let Some(bytes) = u32::try_from(floats.saturating_mul(4)).ok() else {
        return 0;
    };
    // SAFETY: `mAudioData` has `mAudioDataBytesCapacity` bytes. `give` was
    // capped by that capacity, so `floats` fits.
    unsafe {
        let dest = std::slice::from_raw_parts_mut(buffer.mAudioData.as_ptr().cast::<f32>(), floats);
        dest.copy_from_slice(&shared.samples[start..end]);
    }
    buffer.mAudioDataByteSize = bytes;
    shared.frame += give;
    shared.played += give;
    give
}

fn check_layout(sample_rate: u32, channels: u16) -> Result<(), Error> {
    if sample_rate == 0 {
        return Err(Error::InvalidInput("audio sample rate is zero"));
    }
    if channels == 0 || channels > 8 {
        return Err(Error::InvalidInput("audio channel count must be 1 to 8"));
    }
    Ok(())
}

fn pcm(sample_rate: u32, channels: u16) -> AudioStreamBasicDescription {
    let channels = u32::from(channels);
    AudioStreamBasicDescription {
        mSampleRate: f64::from(sample_rate),
        mFormatID: kAudioFormatLinearPCM,
        mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
        mBytesPerPacket: 4 * channels,
        mFramesPerPacket: 1,
        mBytesPerFrame: 4 * channels,
        mChannelsPerFrame: channels,
        mBitsPerChannel: 32,
        mReserved: 0,
    }
}
