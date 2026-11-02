//! The preview output queue.
//!
//! The render callback walks the stored slice. Past the end it writes
//! silence, so the device keeps running. It does not decode or mix.

use std::ffi::c_void;
use std::ptr;
use std::sync::{Arc, Mutex};

use super::sys::{self, AudioQueueBuffer, AudioQueueRef, AudioStreamBasicDescription};
use crate::limits::audio_ok;

const LINEAR_PCM: u32 = u32::from_be_bytes(*b"lpcm");
const FLOAT_PACKED: u32 = 1 | 8;
const BUFFERS: usize = 2;
const BUFFER_BYTES: u32 = 4_096;
/// Eight channels of `f32`, the largest frame this queue will copy.
const MAX_FRAME_BYTES: usize = 32;
const MAX_SLICE_BYTES: usize = 64 * 1024;

/// The output queue could not take the sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueError {
    /// Core Audio did not open an output device.
    NoDevice,
    /// The rate or channel count is not a format this queue will open.
    Format,
    /// The sample ended before one frame.
    Truncated,
    /// The sample is larger than this queue will copy.
    Hostile,
}

struct Held {
    bytes: Vec<u8>,
    cursor: usize,
}

struct Shared {
    held: Mutex<Held>,
    frame_bytes: usize,
}

/// A Core Audio output queue. The callback copies one stored sample.
pub struct OutputQueue {
    queue: AudioQueueRef,
    shared: Arc<Shared>,
}

impl OutputQueue {
    /// Open the default output at `rate` hertz and `channels` of interleaved `f32`.
    pub fn open(rate: u32, channels: u16) -> Result<Self, QueueError> {
        let channels_usize = usize::from(channels);
        if audio_ok(rate, channels_usize).is_err() {
            return Err(QueueError::Format);
        }
        let frame_bytes = channels_usize
            .checked_mul(4)
            .filter(|bytes| *bytes > 0 && *bytes <= MAX_FRAME_BYTES)
            .ok_or(QueueError::Format)?;
        let shared = Arc::new(Shared {
            held: Mutex::new(Held {
                bytes: Vec::new(),
                cursor: 0,
            }),
            frame_bytes,
        });
        let format = AudioStreamBasicDescription {
            sample_rate: f64::from(rate),
            format_id: LINEAR_PCM,
            format_flags: FLOAT_PACKED,
            bytes_per_packet: u32::try_from(frame_bytes).map_err(|_| QueueError::Format)?,
            frames_per_packet: 1,
            bytes_per_frame: u32::try_from(frame_bytes).map_err(|_| QueueError::Format)?,
            channels: u32::from(channels),
            bits: 32,
            reserved: 0,
        };
        let mut queue = ptr::null_mut();
        let status = unsafe {
            sys::AudioQueueNewOutput(
                &format,
                render,
                Arc::as_ptr(&shared).cast_mut().cast(),
                ptr::null_mut(),
                ptr::null(),
                0,
                &mut queue,
            )
        };
        if status != 0 || queue.is_null() {
            return Err(QueueError::NoDevice);
        }
        let output = Self { queue, shared };
        output.prime()?;
        let started = unsafe { sys::AudioQueueStart(output.queue, ptr::null()) };
        if started != 0 {
            return Err(QueueError::NoDevice);
        }
        Ok(output)
    }

    /// Replace the slice the callback walks. An empty slice is silence.
    ///
    /// The cursor returns to the start, so the next callback plays this slice
    /// from its first frame.
    pub fn set_sample(&self, sample: &[u8]) -> Result<(), QueueError> {
        if sample.len() > MAX_SLICE_BYTES {
            return Err(QueueError::Hostile);
        }
        let frame = self.shared.frame_bytes;
        if !sample.is_empty() && (sample.len() < frame || !sample.len().is_multiple_of(frame)) {
            return Err(QueueError::Truncated);
        }
        let mut guard = lock(&self.shared.held);
        guard.bytes.clear();
        guard.bytes.extend_from_slice(sample);
        guard.cursor = 0;
        Ok(())
    }

    /// Copy one sample the way the callback does.
    pub fn copy_one_sample(
        sample: &[u8],
        frame_bytes: usize,
        buffer: &mut [u8],
    ) -> Result<usize, QueueError> {
        copy_one_sample(sample, frame_bytes, buffer)
    }

    fn prime(&self) -> Result<(), QueueError> {
        for _ in 0..BUFFERS {
            let mut buffer = ptr::null_mut();
            let status =
                unsafe { sys::AudioQueueAllocateBuffer(self.queue, BUFFER_BYTES, &mut buffer) };
            if status != 0 || buffer.is_null() {
                return Err(QueueError::NoDevice);
            }
            unsafe {
                let header = &mut *buffer;
                let silence = header
                    .audio_data_bytes_capacity
                    .min(u32::try_from(self.shared.frame_bytes).unwrap_or(0));
                if !header.audio_data.is_null() && silence > 0 {
                    ptr::write_bytes(header.audio_data, 0, usize::try_from(silence).unwrap_or(0));
                }
                header.audio_data_byte_size = silence;
                let queued = sys::AudioQueueEnqueueBuffer(self.queue, buffer, 0, ptr::null());
                if queued != 0 {
                    return Err(QueueError::NoDevice);
                }
            }
        }
        Ok(())
    }
}

impl Drop for OutputQueue {
    fn drop(&mut self) {
        if self.queue.is_null() {
            return;
        }
        unsafe {
            let _ = sys::AudioQueueStop(self.queue, 1);
            let _ = sys::AudioQueueDispose(self.queue, 1);
        }
        self.queue = ptr::null_mut();
    }
}

fn lock(held: &Mutex<Held>) -> std::sync::MutexGuard<'_, Held> {
    match held.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Copy the next frames from `cursor`. Past the end of `bytes` the rest of
/// `out` is silence and the cursor stays at the end.
pub(crate) fn pull_frames(bytes: &[u8], cursor: &mut usize, frame: usize, out: &mut [u8]) -> usize {
    if frame == 0 || out.len() < frame {
        return 0;
    }
    let room = out.len() - (out.len() % frame);
    let mut filled = 0;
    while filled + frame <= room {
        let start = *cursor;
        let Some(end) = start.checked_add(frame) else {
            break;
        };
        if end <= bytes.len() {
            out[filled..filled + frame].copy_from_slice(&bytes[start..end]);
            *cursor = end;
        } else {
            out[filled..filled + frame].fill(0);
        }
        filled += frame;
    }
    filled
}

fn copy_one_sample(
    sample: &[u8],
    frame_bytes: usize,
    buffer: &mut [u8],
) -> Result<usize, QueueError> {
    if frame_bytes == 0 || frame_bytes > MAX_FRAME_BYTES {
        return Err(QueueError::Hostile);
    }
    if sample.len() > MAX_SLICE_BYTES {
        return Err(QueueError::Hostile);
    }
    if buffer.len() < frame_bytes {
        return Err(QueueError::Truncated);
    }
    if !sample.is_empty()
        && (sample.len() < frame_bytes || !sample.len().is_multiple_of(frame_bytes))
    {
        return Err(QueueError::Truncated);
    }
    if sample.is_empty() {
        buffer[..frame_bytes].fill(0);
    } else {
        buffer[..frame_bytes].copy_from_slice(&sample[..frame_bytes]);
    }
    Ok(frame_bytes)
}

unsafe extern "C" fn render(
    user: *mut c_void,
    queue: AudioQueueRef,
    buffer: *mut AudioQueueBuffer,
) {
    if user.is_null() || buffer.is_null() || queue.is_null() {
        return;
    }
    let shared = unsafe { &*(user.cast::<Shared>()) };
    let frame = shared.frame_bytes;
    let mut block = [0_u8; 4_096];
    let copied = {
        let mut guard = lock(&shared.held);
        let Held { bytes, cursor } = &mut *guard;
        pull_frames(bytes, cursor, frame, &mut block)
    };
    unsafe {
        let header = &mut *buffer;
        let capacity = usize::try_from(header.audio_data_bytes_capacity).unwrap_or(0);
        let bytes = copied.min(capacity).min(block.len());
        if !header.audio_data.is_null() && bytes > 0 {
            ptr::copy_nonoverlapping(block.as_ptr(), header.audio_data.cast(), bytes);
        }
        header.audio_data_byte_size = u32::try_from(bytes).unwrap_or(0);
        let _ = sys::AudioQueueEnqueueBuffer(queue, buffer, 0, ptr::null());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_callback_helper_copies_one_sample() {
        let sample = [1_u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let mut buffer = [0_u8; 32];
        let copied = copy_one_sample(&sample, 8, &mut buffer).expect("copy");
        assert_eq!(copied, 8);
        assert_eq!(&buffer[..8], &sample[..8]);
        assert_eq!(&buffer[8..], &[0_u8; 24]);
        assert_eq!(
            copy_one_sample(&[1, 2, 3], 8, &mut buffer),
            Err(QueueError::Truncated)
        );
        let hostile = vec![0_u8; MAX_SLICE_BYTES + 1];
        assert_eq!(
            copy_one_sample(&hostile, 8, &mut buffer),
            Err(QueueError::Hostile)
        );
        assert_eq!(
            copy_one_sample(&[], 0, &mut buffer),
            Err(QueueError::Hostile)
        );
    }

    #[test]
    fn the_callback_walks_the_slice_and_then_plays_silence() {
        let bytes = [1_u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let mut cursor = 0;
        let mut out = [0_u8; 24];
        let filled = pull_frames(&bytes, &mut cursor, 8, &mut out);
        assert_eq!(filled, 24);
        assert_eq!(&out[..16], &bytes);
        assert_eq!(&out[16..], &[0_u8; 8]);
        assert_eq!(cursor, 16);
        let filled = pull_frames(&bytes, &mut cursor, 8, &mut out);
        assert_eq!(filled, 24);
        assert_eq!(out, [0_u8; 24]);
        assert_eq!(cursor, 16);
        assert_eq!(pull_frames(&bytes, &mut cursor, 0, &mut out), 0);
    }

    #[test]
    #[ignore = "opens the default Core Audio output"]
    fn a_queue_opens_on_the_default_device() {
        let queue = OutputQueue::open(48_000, 2).expect("device");
        queue.set_sample(&[0_u8; 8]).expect("sample");
    }
}
