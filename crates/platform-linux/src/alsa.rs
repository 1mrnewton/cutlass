//! Default-device ALSA output.
//!
//! One thread owns the PCM. It pulls a period from the stored slice and
//! writes it with `snd_pcm_writei`, which blocks while the device buffer is
//! full, so the device sets the pace. Past the end of the slice it writes
//! silence. An underrun is recovered and playing goes on; a device that
//! cannot recover ends the thread, and the queue stays silent.
//!
//! `libasound.so.2` is opened with `dlopen`, so neither the build nor a
//! system without ALSA needs it until a queue opens.

#![allow(unsafe_code)]

use std::ffi::{CStr, c_char, c_int, c_long, c_uint, c_ulong, c_void};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};

use crate::slice::{QueueError, Slice, layout};

/// `RTLD_NOW`.
const LOAD_NOW: c_int = 2;
/// `SND_PCM_STREAM_PLAYBACK`.
const STREAM_PLAYBACK: c_int = 0;
/// `SND_PCM_FORMAT_FLOAT_LE`.
const FORMAT_FLOAT_LE: c_int = 14;
/// `SND_PCM_ACCESS_RW_INTERLEAVED`.
const ACCESS_RW_INTERLEAVED: c_int = 3;
/// Ask ALSA to resample when the device runs at another rate.
const SOFT_RESAMPLE: c_int = 1;
/// The device buffer, in microseconds: short enough to follow the playhead.
const LATENCY_US: c_uint = 40_000;
/// Frames written at a time: 10 ms at 48 kHz.
const PERIOD_FRAMES: usize = 480;

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(file: *const c_char, mode: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}

type Open = unsafe extern "C" fn(*mut *mut c_void, *const c_char, c_int, c_int) -> c_int;
type SetParams =
    unsafe extern "C" fn(*mut c_void, c_int, c_int, c_uint, c_uint, c_int, c_uint) -> c_int;
type WriteI = unsafe extern "C" fn(*mut c_void, *const c_void, c_ulong) -> c_long;
type Recover = unsafe extern "C" fn(*mut c_void, c_int, c_int) -> c_int;
type Handle = unsafe extern "C" fn(*mut c_void) -> c_int;

/// The library and the few calls the queue makes.
struct Alsa {
    library: *mut c_void,
    open: Open,
    set_params: SetParams,
    writei: WriteI,
    recover: Recover,
    drop: Handle,
    close: Handle,
}

impl Alsa {
    fn load() -> Option<Self> {
        // SAFETY: the name is a valid C string, and dlopen has no other
        // precondition.
        let library = unsafe { dlopen(c"libasound.so.2".as_ptr(), LOAD_NOW) };
        if library.is_null() {
            return None;
        }
        let found = (|| {
            Some(Self {
                library,
                // SAFETY: each symbol has this signature in alsa/pcm.h.
                open: unsafe { symbol::<Open>(library, c"snd_pcm_open")? },
                set_params: unsafe { symbol::<SetParams>(library, c"snd_pcm_set_params")? },
                writei: unsafe { symbol::<WriteI>(library, c"snd_pcm_writei")? },
                recover: unsafe { symbol::<Recover>(library, c"snd_pcm_recover")? },
                drop: unsafe { symbol::<Handle>(library, c"snd_pcm_drop")? },
                close: unsafe { symbol::<Handle>(library, c"snd_pcm_close")? },
            })
        })();
        if found.is_none() {
            // SAFETY: the handle came from dlopen and nothing from it is kept.
            unsafe { dlclose(library) };
        }
        found
    }
}

impl Drop for Alsa {
    fn drop(&mut self) {
        // SAFETY: the handle came from dlopen, and the PCM using it was
        // closed before the library is dropped.
        unsafe { dlclose(self.library) };
    }
}

/// Look `name` up in `library` as a function of type `F`.
///
/// # Safety
///
/// `F` must be the function pointer type of the symbol.
unsafe fn symbol<F: Copy>(library: *mut c_void, name: &CStr) -> Option<F> {
    // SAFETY: the handle is live and the name is a valid C string.
    let found = unsafe { dlsym(library, name.as_ptr()) };
    if found.is_null() || size_of::<F>() != size_of::<*mut c_void>() {
        return None;
    }
    // SAFETY: the caller names the symbol's type, which is a function
    // pointer the size of the address.
    Some(unsafe { std::mem::transmute_copy::<*mut c_void, F>(&found) })
}

/// An open PCM on the default device. Closed on drop.
struct Pcm {
    alsa: Alsa,
    handle: *mut c_void,
}

impl Pcm {
    fn open(device: &CStr, rate: u32, channels: u16) -> Option<Self> {
        let alsa = Alsa::load()?;
        let mut handle = ptr::null_mut();
        // SAFETY: the out pointer is valid and the device name is a C string.
        let opened = unsafe { (alsa.open)(&raw mut handle, device.as_ptr(), STREAM_PLAYBACK, 0) };
        if opened < 0 || handle.is_null() {
            return None;
        }
        let pcm = Self { alsa, handle };
        // SAFETY: the handle is an open PCM.
        let set = unsafe {
            (pcm.alsa.set_params)(
                pcm.handle,
                FORMAT_FLOAT_LE,
                ACCESS_RW_INTERLEAVED,
                c_uint::from(channels),
                rate,
                SOFT_RESAMPLE,
                LATENCY_US,
            )
        };
        (set >= 0).then_some(pcm)
    }

    /// Write whole frames from `bytes`. Recovers an underrun once; false
    /// when the device is gone.
    fn write(&self, bytes: &[u8], frame: usize) -> bool {
        let mut start = 0;
        while start + frame <= bytes.len() {
            let rest = &bytes[start..];
            let frames = c_ulong::try_from(rest.len() / frame).unwrap_or(0);
            // SAFETY: the handle is open, and `rest` holds `frames` whole
            // interleaved frames in the format the PCM was set to.
            let wrote = unsafe { (self.alsa.writei)(self.handle, rest.as_ptr().cast(), frames) };
            if wrote < 0 {
                let error = c_int::try_from(wrote).unwrap_or(c_int::MIN);
                // SAFETY: the handle is open; recover takes the error writei gave.
                if unsafe { (self.alsa.recover)(self.handle, error, 1) } < 0 {
                    return false;
                }
                continue;
            }
            let Ok(wrote) = usize::try_from(wrote) else {
                return false;
            };
            start += wrote * frame;
        }
        true
    }
}

impl Drop for Pcm {
    fn drop(&mut self) {
        // SAFETY: the handle is open, and nothing uses it after close.
        unsafe {
            (self.alsa.drop)(self.handle);
            (self.alsa.close)(self.handle);
        }
    }
}

/// An ALSA output. The writer thread walks the stored slice.
pub struct OutputQueue {
    slice: Arc<Slice>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl OutputQueue {
    /// Open the default ALSA output at `rate` hertz and `channels`.
    pub fn open(rate: u32, channels: u16) -> Result<Self, QueueError> {
        let frame = layout(rate, channels)?;
        let slice = Arc::new(Slice::new(frame)?);
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        let thread_slice = Arc::clone(&slice);
        let thread_stop = Arc::clone(&stop);
        let spawned = thread::Builder::new()
            .name("cutlass-alsa".to_owned())
            .spawn(move || {
                // The PCM stays on this thread from open to close.
                let Some(pcm) = Pcm::open(c"default", rate, channels) else {
                    let _ = tx.send(false);
                    return;
                };
                let _ = tx.send(true);
                let mut period = vec![0_u8; PERIOD_FRAMES * frame];
                while !thread_stop.load(Ordering::Acquire) {
                    let filled = thread_slice.pull(&mut period);
                    if !pcm.write(&period[..filled], frame) {
                        break;
                    }
                }
            });
        let Ok(thread) = spawned else {
            return Err(QueueError::NoDevice);
        };
        if rx.recv() == Ok(true) {
            Ok(Self {
                slice,
                stop,
                thread: Some(thread),
            })
        } else {
            let _ = thread.join();
            Err(QueueError::NoDevice)
        }
    }

    /// Replace the slice the writer thread walks. An empty slice is silence.
    pub fn set_sample(&self, sample: &[u8]) -> Result<(), QueueError> {
        self.slice.set_sample(sample)
    }
}

impl Drop for OutputQueue {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_refuses_a_bad_format() {
        assert!(matches!(OutputQueue::open(0, 2), Err(QueueError::Format)));
        assert!(matches!(
            OutputQueue::open(48_000, 1),
            Err(QueueError::Format)
        ));
    }

    #[test]
    fn the_null_device_takes_whole_periods_when_alsa_is_installed() {
        // ALSA's `null` plugin needs no hardware, so CI writes through the
        // same calls a real device gets.
        if Alsa::load().is_none() {
            return;
        }
        let pcm = Pcm::open(c"null", 48_000, 2).expect("null pcm");
        let period = vec![0_u8; PERIOD_FRAMES * 8];
        for _ in 0..5 {
            assert!(pcm.write(&period, 8));
        }
        // A trailing partial frame is not written.
        assert!(pcm.write(&period[..12], 8));
        drop(pcm);
    }

    #[test]
    #[ignore = "opens the default ALSA output"]
    fn the_default_device_opens_plays_and_closes() {
        let queue = match OutputQueue::open(48_000, 2) {
            Ok(queue) => queue,
            Err(error) => {
                assert_eq!(error, QueueError::NoDevice);
                return;
            }
        };
        let tone: Vec<u8> = (0..4_800_u16)
            .flat_map(|n| {
                let value = (f32::from(n) * 0.06).sin() * 0.1;
                [value, value]
            })
            .flat_map(f32::to_le_bytes)
            .collect();
        queue.set_sample(&tone).expect("tone");
        thread::sleep(std::time::Duration::from_millis(150));
        drop(queue);
    }
}
