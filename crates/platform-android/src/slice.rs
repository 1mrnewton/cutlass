//! The bytes the preview callback walks.
//!
//! Replacing the slice starts at the first frame. Past the end, the rest of
//! the callback buffer is silence and the cursor stays at the end.

use std::sync::{Mutex, TryLockError};

/// Largest slice the callback will look at.
pub(crate) const MAX_SLICE_BYTES: usize = 64 * 1024;

/// Eight channels of `f32`, the largest frame this queue will copy.
const MAX_FRAME_BYTES: usize = 32;

/// The preview queue only opens 48 kHz stereo `f32`.
const PREVIEW_RATE: u32 = 48_000;
const PREVIEW_CHANNELS: u16 = 2;

/// The output queue could not take the sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueError {
    /// The host did not open an output device.
    NoDevice,
    /// The rate or channel count is not the preview format.
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

/// One interleaved slice and the cursor the callback advances.
pub(crate) struct Slice {
    held: Mutex<Held>,
    frame_bytes: usize,
}

impl Slice {
    pub(crate) fn new(frame_bytes: usize) -> Result<Self, QueueError> {
        if frame_bytes == 0 || frame_bytes > MAX_FRAME_BYTES {
            return Err(QueueError::Format);
        }
        Ok(Self {
            held: Mutex::new(Held {
                bytes: Vec::new(),
                cursor: 0,
            }),
            frame_bytes,
        })
    }

    /// Replace the slice. An empty slice is silence, and the cursor returns
    /// to the first frame.
    pub(crate) fn set_sample(&self, sample: &[u8]) -> Result<(), QueueError> {
        if sample.len() > MAX_SLICE_BYTES {
            return Err(QueueError::Hostile);
        }
        let frame = self.frame_bytes;
        if !sample.is_empty() && (sample.len() < frame || !sample.len().is_multiple_of(frame)) {
            return Err(QueueError::Truncated);
        }
        let mut guard = lock(&self.held);
        guard.bytes.clear();
        guard.bytes.extend_from_slice(sample);
        guard.cursor = 0;
        Ok(())
    }

    pub(crate) fn stride(&self) -> usize {
        self.frame_bytes
    }

    /// Copy the next frames. When the slice is being replaced, write silence
    /// for this call and leave the cursor where it is.
    pub(crate) fn pull_now(&self, out: &mut [u8]) -> usize {
        match self.held.try_lock() {
            Ok(mut guard) => {
                let Held { bytes, cursor } = &mut *guard;
                pull_frames(bytes, cursor, self.frame_bytes, out)
            }
            Err(TryLockError::Poisoned(poisoned)) => {
                let mut guard = poisoned.into_inner();
                let Held { bytes, cursor } = &mut *guard;
                pull_frames(bytes, cursor, self.frame_bytes, out)
            }
            Err(TryLockError::WouldBlock) => silence(out, self.frame_bytes),
        }
    }
}

/// Frame size for the preview format. Any other rate or channel count is refused.
pub(crate) fn layout(rate: u32, channels: u16) -> Result<usize, QueueError> {
    if rate != PREVIEW_RATE || channels != PREVIEW_CHANNELS {
        return Err(QueueError::Format);
    }
    Ok(usize::from(PREVIEW_CHANNELS) * size_of::<f32>())
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

fn silence(out: &mut [u8], frame: usize) -> usize {
    if frame == 0 || out.len() < frame {
        return 0;
    }
    let room = out.len() - (out.len() % frame);
    out[..room].fill(0);
    room
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, mpsc};
    use std::thread;

    use super::*;

    #[test]
    fn the_preview_layout_is_stereo_at_48_khz() {
        assert_eq!(layout(48_000, 2), Ok(8));
        assert_eq!(layout(0, 2), Err(QueueError::Format));
        assert_eq!(layout(44_100, 2), Err(QueueError::Format));
        assert_eq!(layout(48_000, 1), Err(QueueError::Format));
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
    fn replacing_the_slice_starts_at_the_first_frame() {
        let slice = Slice::new(8).expect("slice");
        slice
            .set_sample(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
            .expect("sample");
        let mut out = [0_u8; 8];
        assert_eq!(slice.pull_now(&mut out), 8);
        assert_eq!(out, [1, 2, 3, 4, 5, 6, 7, 8]);
        slice.set_sample(&[4, 4, 4, 4, 4, 4, 4, 4]).expect("reset");
        assert_eq!(slice.pull_now(&mut out), 8);
        assert_eq!(out, [4, 4, 4, 4, 4, 4, 4, 4]);
        slice.set_sample(&[]).expect("silence");
        out = [9_u8; 8];
        assert_eq!(slice.pull_now(&mut out), 8);
        assert_eq!(out, [0_u8; 8]);
    }

    #[test]
    fn a_short_or_hostile_slice_is_refused() {
        let slice = Slice::new(8).expect("slice");
        assert_eq!(slice.set_sample(&[1, 2, 3]), Err(QueueError::Truncated));
        let hostile = vec![0_u8; MAX_SLICE_BYTES + 1];
        assert_eq!(slice.set_sample(&hostile), Err(QueueError::Hostile));
        assert_eq!(Slice::new(0).err(), Some(QueueError::Format));
        let mut out = [0_u8; 4];
        assert_eq!(slice.pull_now(&mut out), 0);
    }

    #[test]
    fn a_busy_slice_plays_silence_without_moving_the_cursor() {
        let slice = Arc::new(Slice::new(8).expect("slice"));
        slice.set_sample(&[1, 2, 3, 4, 5, 6, 7, 8]).expect("sample");
        let held = Arc::clone(&slice);
        let (ready_tx, ready_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let worker = thread::spawn(move || {
            let _guard = lock(&held.held);
            ready_tx.send(()).expect("ready");
            release_rx.recv().expect("release");
        });
        ready_rx.recv().expect("held");
        let mut out = [9_u8; 8];
        assert_eq!(slice.pull_now(&mut out), 8);
        assert_eq!(out, [0_u8; 8]);
        release_tx.send(()).expect("unlock");
        worker.join().expect("worker");
        assert_eq!(slice.pull_now(&mut out), 8);
        assert_eq!(out, [1, 2, 3, 4, 5, 6, 7, 8]);
    }
}
