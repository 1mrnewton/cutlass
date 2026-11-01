//! Shared-mode WASAPI output.
//!
//! One thread owns the COM client. It waits on the buffer event and copies
//! the stored slice into the render buffer. Past the end of the slice it
//! writes silence. `AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM` lets the engine accept
//! 48 kHz stereo `f32` when the device mix format is different.

#![allow(unsafe_code)]

use std::ffi::c_void;
use std::mem::{size_of_val, transmute};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};

use crate::format::{Guid, WaveFormatExtensible, float_stereo};
use crate::slice::{QueueError, Slice, layout};

const CLSCTX_ALL: u32 = 23;
const SHARE_SHARED: u32 = 0;
const STREAM_FLAGS: u32 = 0x0004_0000 | 0x0800_0000 | 0x8000_0000;
const BUFFER_SILENT: u32 = 0x2;
const WAIT_SLICE: u32 = 200;
const WAIT_TIMEOUT: u32 = 0x0000_0102;

const CLSID_MM_DEVICE_ENUMERATOR: Guid = Guid {
    data1: 0xBCDE_0395,
    data2: 0xE52F,
    data3: 0x467C,
    data4: [0x8E, 0x3D, 0xC4, 0x57, 0x92, 0x91, 0x69, 0x2E],
};
const IID_IMM_DEVICE_ENUMERATOR: Guid = Guid {
    data1: 0xA956_64D2,
    data2: 0x9614,
    data3: 0x4F35,
    data4: [0xA7, 0x46, 0xDE, 0x8D, 0xB6, 0x36, 0x17, 0xE6],
};
const IID_IAUDIO_CLIENT: Guid = Guid {
    data1: 0x1CB9_AD4C,
    data2: 0xDBFA,
    data3: 0x4C32,
    data4: [0xB1, 0x78, 0xC2, 0xF5, 0x68, 0xA7, 0x03, 0xB2],
};
const IID_IAUDIO_RENDER_CLIENT: Guid = Guid {
    data1: 0xF294_ACFC,
    data2: 0x3146,
    data3: 0x4483,
    data4: [0xA7, 0xBF, 0xAD, 0xDC, 0xA7, 0xC2, 0x60, 0xE2],
};

type Slot = unsafe extern "system" fn();

#[link(name = "ole32")]
unsafe extern "system" {
    fn CoInitializeEx(reserved: *mut c_void, model: u32) -> i32;
    fn CoUninitialize();
    fn CoCreateInstance(
        clsid: *const Guid,
        outer: *mut c_void,
        context: u32,
        iid: *const Guid,
        out: *mut *mut c_void,
    ) -> i32;
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn CreateEventW(attrs: *mut c_void, manual: i32, initial: i32, name: *const u16)
    -> *mut c_void;
    fn SetEvent(handle: *mut c_void) -> i32;
    fn WaitForSingleObject(handle: *mut c_void, millis: u32) -> u32;
    fn CloseHandle(handle: *mut c_void) -> i32;
}

macro_rules! com {
    ($object:expr, $index:expr, fn($($arg:ty),*) -> $ret:ty, $($value:expr),* $(,)?) => {{
        type Method = unsafe extern "system" fn(*mut c_void, $($arg),*) -> $ret;
        let method: Method = transmute(vslot($object, $index));
        method($object, $($value),*)
    }};
}

unsafe fn vslot(object: *mut c_void, index: usize) -> Slot {
    unsafe {
        let table = *object.cast::<*const Slot>();
        *table.add(index)
    }
}

struct Shared {
    slice: Slice,
}

struct EventHandle(*mut c_void);

unsafe impl Send for EventHandle {}
unsafe impl Sync for EventHandle {}

impl EventHandle {
    fn create() -> Result<Self, QueueError> {
        let handle = unsafe { CreateEventW(ptr::null_mut(), 0, 0, ptr::null()) };
        if handle.is_null() {
            Err(QueueError::NoDevice)
        } else {
            Ok(Self(handle))
        }
    }

    fn signal(&self) {
        unsafe {
            let _ = SetEvent(self.0);
        }
    }
}

impl Drop for EventHandle {
    fn drop(&mut self) {
        if self.0.is_null() {
            return;
        }
        unsafe {
            let _ = CloseHandle(self.0);
        }
        self.0 = ptr::null_mut();
    }
}

struct Client {
    enumerator: *mut c_void,
    device: *mut c_void,
    audio: *mut c_void,
    render: *mut c_void,
    buffer_frames: u32,
    started: bool,
    com: bool,
}

impl Client {
    fn open(event: *mut c_void, format: &WaveFormatExtensible) -> Result<Self, ()> {
        if unsafe { CoInitializeEx(ptr::null_mut(), 0) } < 0 {
            return Err(());
        }
        let mut client = Self {
            enumerator: ptr::null_mut(),
            device: ptr::null_mut(),
            audio: ptr::null_mut(),
            render: ptr::null_mut(),
            buffer_frames: 0,
            started: false,
            com: true,
        };
        if client.bind(event, format).is_err() {
            return Err(());
        }
        Ok(client)
    }

    fn bind(&mut self, event: *mut c_void, format: &WaveFormatExtensible) -> Result<(), ()> {
        self.enumerator = enumerator()?;
        self.device = default_render(self.enumerator)?;
        self.audio = activate(self.device)?;
        initialize(self.audio, format)?;
        self.buffer_frames = buffer_frames(self.audio)?;
        set_event(self.audio, event)?;
        self.render = render_client(self.audio)?;
        start(self.audio)?;
        self.started = true;
        Ok(())
    }

    fn pump(&self, event: *mut c_void, shared: &Shared, stop: &AtomicBool) {
        while !stop.load(Ordering::Acquire) {
            let waited = unsafe { WaitForSingleObject(event, WAIT_SLICE) };
            if stop.load(Ordering::Acquire) {
                break;
            }
            if waited != 0 && waited != WAIT_TIMEOUT {
                break;
            }
            if self.fill(shared).is_err() {
                break;
            }
        }
    }

    fn fill(&self, shared: &Shared) -> Result<(), ()> {
        let mut padding = 0_u32;
        if invoke(self.audio, 6, &mut padding) < 0 {
            return Err(());
        }
        if padding > self.buffer_frames {
            return Err(());
        }
        let available = self.buffer_frames - padding;
        if available == 0 {
            return Ok(());
        }
        let mut data = ptr::null_mut();
        if render_get(self.render, available, &mut data) < 0 {
            return Err(());
        }
        let stride = shared.slice.stride();
        let Ok(frames) = usize::try_from(available) else {
            let _ = render_release(self.render, available, BUFFER_SILENT);
            return Err(());
        };
        let Some(bytes) = frames.checked_mul(stride) else {
            let _ = render_release(self.render, available, BUFFER_SILENT);
            return Err(());
        };
        if data.is_null() {
            let _ = render_release(self.render, available, BUFFER_SILENT);
            return Ok(());
        }
        let out = unsafe { std::slice::from_raw_parts_mut(data, bytes) };
        shared.slice.pull(out);
        if render_release(self.render, available, 0) < 0 {
            Err(())
        } else {
            Ok(())
        }
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        unsafe {
            if self.started && !self.audio.is_null() {
                let _ = invoke_hr(self.audio, 11);
            }
            release(self.render);
            release(self.audio);
            release(self.device);
            release(self.enumerator);
            if self.com {
                CoUninitialize();
            }
        }
        self.render = ptr::null_mut();
        self.audio = ptr::null_mut();
        self.device = ptr::null_mut();
        self.enumerator = ptr::null_mut();
        self.com = false;
        self.started = false;
    }
}

/// A shared-mode WASAPI output. The render thread walks the stored slice.
pub struct OutputQueue {
    shared: Arc<Shared>,
    stop: Arc<AtomicBool>,
    event: Arc<EventHandle>,
    thread: Option<JoinHandle<()>>,
}

impl OutputQueue {
    /// Open the default shared-mode output at `rate` hertz and `channels`.
    pub fn open(rate: u32, channels: u16) -> Result<Self, QueueError> {
        let frame = layout(rate, channels)?;
        let format = float_stereo(rate);
        if size_of_format(&format) != 40 {
            return Err(QueueError::Format);
        }
        let slice = Slice::new(frame)?;
        if slice.stride() != frame {
            return Err(QueueError::Format);
        }
        let shared = Arc::new(Shared { slice });
        let stop = Arc::new(AtomicBool::new(false));
        let event = Arc::new(EventHandle::create()?);
        let (tx, rx) = mpsc::channel();
        let thread_shared = Arc::clone(&shared);
        let thread_stop = Arc::clone(&stop);
        let thread_event = Arc::clone(&event);
        let spawned = thread::Builder::new()
            .name("cutlass-wasapi".to_owned())
            .spawn(move || {
                let handle = thread_event.0;
                match Client::open(handle, &format) {
                    Ok(client) => {
                        let _ = tx.send(true);
                        client.pump(handle, &thread_shared, &thread_stop);
                    }
                    Err(()) => {
                        let _ = tx.send(false);
                    }
                }
            });
        let Ok(thread) = spawned else {
            return Err(QueueError::NoDevice);
        };
        match rx.recv() {
            Ok(true) => Ok(Self {
                shared,
                stop,
                event,
                thread: Some(thread),
            }),
            _ => {
                stop.store(true, Ordering::Release);
                event.signal();
                let _ = thread.join();
                Err(QueueError::NoDevice)
            }
        }
    }

    /// Replace the slice the render thread walks. An empty slice is silence.
    pub fn set_sample(&self, sample: &[u8]) -> Result<(), QueueError> {
        self.shared.slice.set_sample(sample)
    }
}

impl Drop for OutputQueue {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.event.signal();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn size_of_format(format: &WaveFormatExtensible) -> usize {
    size_of_val(format)
}

fn enumerator() -> Result<*mut c_void, ()> {
    let mut out = ptr::null_mut();
    let hr = unsafe {
        CoCreateInstance(
            &CLSID_MM_DEVICE_ENUMERATOR,
            ptr::null_mut(),
            CLSCTX_ALL,
            &IID_IMM_DEVICE_ENUMERATOR,
            &mut out,
        )
    };
    if hr < 0 || out.is_null() {
        Err(())
    } else {
        Ok(out)
    }
}

fn default_render(enumerator: *mut c_void) -> Result<*mut c_void, ()> {
    let mut device = ptr::null_mut();
    let hr = unsafe {
        com!(
            enumerator,
            4,
            fn(u32, u32, *mut *mut c_void) -> i32,
            0,
            0,
            &mut device,
        )
    };
    if hr < 0 || device.is_null() {
        Err(())
    } else {
        Ok(device)
    }
}

fn activate(device: *mut c_void) -> Result<*mut c_void, ()> {
    let mut audio = ptr::null_mut();
    let hr = unsafe {
        com!(
            device,
            3,
            fn(*const Guid, u32, *mut c_void, *mut *mut c_void) -> i32,
            &IID_IAUDIO_CLIENT,
            CLSCTX_ALL,
            ptr::null_mut(),
            &mut audio,
        )
    };
    if hr < 0 || audio.is_null() {
        Err(())
    } else {
        Ok(audio)
    }
}

fn initialize(audio: *mut c_void, format: &WaveFormatExtensible) -> Result<(), ()> {
    let hr = unsafe {
        com!(
            audio,
            3,
            fn(u32, u32, i64, i64, *const WaveFormatExtensible, *const Guid) -> i32,
            SHARE_SHARED,
            STREAM_FLAGS,
            0,
            0,
            format,
            ptr::null(),
        )
    };
    if hr < 0 { Err(()) } else { Ok(()) }
}

fn buffer_frames(audio: *mut c_void) -> Result<u32, ()> {
    let mut frames = 0_u32;
    let hr = unsafe { com!(audio, 4, fn(*mut u32) -> i32, &mut frames) };
    if hr < 0 || frames == 0 {
        Err(())
    } else {
        Ok(frames)
    }
}

fn set_event(audio: *mut c_void, event: *mut c_void) -> Result<(), ()> {
    let hr = unsafe { com!(audio, 13, fn(*mut c_void) -> i32, event) };
    if hr < 0 { Err(()) } else { Ok(()) }
}

fn render_client(audio: *mut c_void) -> Result<*mut c_void, ()> {
    let mut render = ptr::null_mut();
    let hr = unsafe {
        com!(
            audio,
            14,
            fn(*const Guid, *mut *mut c_void) -> i32,
            &IID_IAUDIO_RENDER_CLIENT,
            &mut render,
        )
    };
    if hr < 0 || render.is_null() {
        Err(())
    } else {
        Ok(render)
    }
}

fn start(audio: *mut c_void) -> Result<(), ()> {
    if unsafe { invoke_hr(audio, 10) } < 0 {
        Err(())
    } else {
        Ok(())
    }
}

fn invoke(object: *mut c_void, index: usize, padding: *mut u32) -> i32 {
    unsafe { com!(object, index, fn(*mut u32) -> i32, padding) }
}

fn render_get(render: *mut c_void, frames: u32, data: *mut *mut u8) -> i32 {
    unsafe { com!(render, 3, fn(u32, *mut *mut u8) -> i32, frames, data) }
}

fn render_release(render: *mut c_void, frames: u32, flags: u32) -> i32 {
    unsafe { com!(render, 4, fn(u32, u32) -> i32, frames, flags) }
}

unsafe fn invoke_hr(object: *mut c_void, index: usize) -> i32 {
    unsafe { com!(object, index, fn() -> i32,) }
}

unsafe fn release(object: *mut c_void) {
    if object.is_null() {
        return;
    }
    unsafe {
        let method: unsafe extern "system" fn(*mut c_void) -> u32 = transmute(vslot(object, 2));
        let _ = method(object);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "opens the default WASAPI output"]
    fn a_queue_opens_on_the_default_device() {
        let queue = OutputQueue::open(48_000, 2).expect("device");
        queue.set_sample(&[0_u8; 8]).expect("sample");
    }
}
