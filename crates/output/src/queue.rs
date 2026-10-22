//! An AudioQueue that plays interleaved `f32`.
//!
//! The callback runs on the queue's thread. It copies one buffer and returns.
//! `play` is called from the owner thread.

use std::ffi::c_void;
use std::num::NonZeroU32;
use std::ptr::{self, NonNull};
use std::sync::{Arc, Mutex, MutexGuard};

use cutlass_core::{AudioSamples, Timestamp};
use objc2_audio_toolbox::{
    AudioQueueAllocateBuffer, AudioQueueBufferRef, AudioQueueDispose, AudioQueueEnqueueBuffer,
    AudioQueueGetCurrentTime, AudioQueueNewOutput, AudioQueueRef, AudioQueueStart, AudioQueueStop,
    AudioQueueTimelineRef, kAudioQueueErr_InvalidRunState,
};
use objc2_core_audio_types::{
    AudioStreamBasicDescription, AudioTimeStamp, AudioTimeStampFlags, SMPTETime, SMPTETimeFlags,
    SMPTETimeType, kAudioFormatFlagIsFloat, kAudioFormatFlagIsPacked, kAudioFormatLinearPCM,
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
    /// Buffers the callback could not fill. The owner enqueues them after
    /// more samples arrive. A raw pointer is stored as `usize` so the sample
    /// lock stays `Send`. The owner is the only thread that enqueues one.
    idle: Vec<usize>,
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
            idle: Vec::new(),
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
            shared.idle.clear();
        }
        self.stop_queue()?;
        {
            let mut shared = self.lock()?;
            shared.closed = false;
            shared.idle.clear();
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

    /// Appends samples without moving the clock back to zero.
    ///
    /// [`Self::play`] has to be called first. A buffer the device returned
    /// empty is filled from the new samples and queued again. Played samples
    /// already copied into a device buffer are dropped, so a long cut does
    /// not keep the whole mix.
    pub fn extend(&mut self, samples: &AudioSamples) -> Result<(), Error> {
        if samples.sample_rate() != self.sample_rate || samples.channels() != self.channels {
            return Err(Error::InvalidInput("audio samples do not match the output"));
        }
        if samples.frames() == 0 {
            return Ok(());
        }
        let idle = {
            let mut shared = self.lock()?;
            if shared.closed {
                return Err(Error::InvalidInput("audio output is stopped"));
            }
            discard_played(&mut shared);
            let extra = samples.interleaved();
            shared.frames = shared.frames.saturating_add(samples.frames());
            shared.samples.extend(extra);
            std::mem::take(&mut shared.idle)
        };
        let mut resumed = false;
        for buffer in idle {
            let buffer = buffer as AudioQueueBufferRef;
            if buffer.is_null() {
                continue;
            }
            self.enqueue(buffer)?;
            resumed = true;
        }
        if resumed {
            self.resume()?;
        }
        Ok(())
    }

    /// Stops the device and drops anything still queued.
    pub fn stop(&mut self) -> Result<(), Error> {
        {
            let mut shared = self.lock()?;
            shared.closed = true;
            shared.samples.clear();
            shared.frame = 0;
            shared.frames = 0;
            shared.played = 0;
            shared.idle.clear();
        }
        self.stop_queue()?;
        let mut shared = self.lock()?;
        shared.idle.clear();
        Ok(())
    }

    /// Frames copied into a device buffer.
    ///
    /// This is the handoff, not how far the speaker has gotten.
    pub fn enqueued_frames(&self) -> usize {
        self.lock().map(|shared| shared.played).unwrap_or(0)
    }

    /// Sample time since [`Self::play`] started, at this device's rate.
    ///
    /// `None` when the queue is stopped or the device has not published a
    /// sample time. The value is a whole sample frame, rounded down.
    pub fn clock(&self) -> Result<Option<Timestamp>, Error> {
        let mut stamp = blank_stamp();
        // SAFETY: `queue` is the queue this value owns. A null timeline asks
        // for the current time without discontinuity reporting.
        let status = unsafe {
            AudioQueueGetCurrentTime(self.queue, no_timeline(), &mut stamp, ptr::null_mut())
        };
        if status == kAudioQueueErr_InvalidRunState {
            return Ok(None);
        }
        error::check("AudioQueueGetCurrentTime", status)?;
        if !stamp.mFlags.contains(AudioTimeStampFlags::SampleTimeValid) {
            return Ok(None);
        }
        sample_time(stamp.mSampleTime, self.sample_rate)
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

    /// Starts a queue that ran out of buffers. A queue that is already
    /// running reports an invalid state, and that is not a failure.
    fn resume(&self) -> Result<(), Error> {
        // SAFETY: `queue` is the queue this value owns. A null start time
        // means start now.
        let status = unsafe { AudioQueueStart(self.queue, ptr::null()) };
        if status == 0 || status == kAudioQueueErr_InvalidRunState {
            return Ok(());
        }
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
    if give == 0 {
        shared.idle.push(buffer as usize);
        return;
    }
    drop(shared);
    // SAFETY: `buffer` belongs to `queue` and was just returned by it.
    unsafe {
        let _ = AudioQueueEnqueueBuffer(queue, buffer, 0, ptr::null());
    }
}

fn discard_played(shared: &mut Shared) {
    let channels = usize::from(shared.channels);
    if channels == 0 || shared.frame == 0 {
        return;
    }
    let floats = shared.frame.saturating_mul(channels);
    if floats > shared.samples.len() {
        return;
    }
    shared.samples.drain(..floats);
    shared.frames = shared.frames.saturating_sub(shared.frame);
    shared.frame = 0;
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

fn no_timeline() -> AudioQueueTimelineRef {
    ptr::null_mut()
}

fn blank_stamp() -> AudioTimeStamp {
    AudioTimeStamp {
        mSampleTime: 0.0,
        mHostTime: 0,
        mRateScalar: 0.0,
        mWordClockTime: 0,
        mSMPTETime: SMPTETime {
            mSubframes: 0,
            mSubframeDivisor: 0,
            mCounter: 0,
            mType: SMPTETimeType(0),
            mFlags: SMPTETimeFlags(0),
            mHours: 0,
            mMinutes: 0,
            mSeconds: 0,
            mFrames: 0,
        },
        mFlags: AudioTimeStampFlags(0),
        mReserved: 0,
    }
}

fn sample_time(frames: f64, sample_rate: u32) -> Result<Option<Timestamp>, Error> {
    let frames = frames.floor();
    // 2^53 is the last integer f64 stores exactly, and it fits in i64.
    if !(0.0..9_007_199_254_740_992.0).contains(&frames) {
        return Ok(None);
    }
    let value = frames as i64;
    let Some(timescale) = NonZeroU32::new(sample_rate) else {
        return Err(Error::InvalidInput("audio sample rate is zero"));
    };
    Ok(Some(Timestamp::new(value, timescale)))
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

#[cfg(test)]
mod tests {
    use super::sample_time;

    #[test]
    fn a_fractional_sample_time_rounds_down() {
        let time = sample_time(10.9, 48_000).expect("time").expect("frames");
        assert_eq!(time.value, 10);
        assert_eq!(time.timescale.get(), 48_000);
    }

    #[test]
    fn a_negative_sample_time_is_absent() {
        assert!(sample_time(-0.1, 48_000).expect("time").is_none());
    }
}
