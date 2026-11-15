//! Retained CoreFoundation values. Drop releases one retain.

use std::ffi::c_void;
use std::ptr;

use cutlass_decode::CodecError;

use super::sys;

pub struct Cf {
    ptr: *mut c_void,
}

impl Cf {
    pub fn from_owned(ptr: *mut c_void) -> Option<Self> {
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    pub fn as_ptr(&self) -> *const c_void {
        self.ptr
    }

    pub fn as_mut(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for Cf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `from_owned` took one retain on a non-null CF object, and
            // this releases that retain once, when the owner goes away.
            unsafe { sys::CFRelease(self.ptr) }
        }
    }
}

// SAFETY: a `Cf` owns one retain. CoreFoundation retain and release are
// thread safe, and `Cf` is not `Sync`, so the object is used from one thread
// at a time. A mutable object (a `CFMutableData`, a bitmap context) is only
// written through the one owner.
unsafe impl Send for Cf {}

pub fn string(text: &str) -> Result<Cf, CodecError> {
    let mut bytes = Vec::with_capacity(text.len() + 1);
    bytes.extend_from_slice(text.as_bytes());
    bytes.push(0);
    // SAFETY: `bytes` is the UTF-8 text with a trailing NUL and lives until the
    // call returns; CoreFoundation copies it. A null allocator is the default.
    let ptr = unsafe { sys::CFStringCreateWithCString(ptr::null(), bytes.as_ptr(), sys::UTF8) };
    Cf::from_owned(ptr).ok_or(CodecError::Limit { what: "cf string" })
}

pub fn data(bytes: &[u8]) -> Result<Cf, CodecError> {
    let len =
        sys::CFIndex::try_from(bytes.len()).map_err(|_| CodecError::Limit { what: "cf data" })?;
    // SAFETY: `bytes` is valid for `len` bytes during the call, and
    // `CFDataCreate` copies them.
    let ptr = unsafe { sys::CFDataCreate(ptr::null(), bytes.as_ptr(), len) };
    Cf::from_owned(ptr).ok_or(CodecError::Limit { what: "cf data" })
}

pub fn data_mut() -> Result<Cf, CodecError> {
    // SAFETY: the call takes only the default allocator and a capacity of 0,
    // which means no limit.
    let ptr = unsafe { sys::CFDataCreateMutable(ptr::null(), 0) };
    Cf::from_owned(ptr).ok_or(CodecError::Limit { what: "cf data" })
}

pub fn data_bytes(data: &Cf) -> Result<Vec<u8>, CodecError> {
    // SAFETY: `data` owns a retain on a non-null object. Every `Cf` passed here
    // is a `CFData` made by `data` or `data_mut`.
    let len = unsafe { sys::CFDataGetLength(data.as_ptr()) };
    if len < 0 {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "cf data",
        });
    }
    let len = usize::try_from(len).map_err(|_| CodecError::Limit { what: "cf data" })?;
    if len == 0 {
        return Ok(Vec::new());
    }
    // SAFETY: as above; `data` is a live `CFData`.
    let ptr = unsafe { sys::CFDataGetBytePtr(data.as_ptr()) };
    if ptr.is_null() {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "cf data",
        });
    }
    // SAFETY: `ptr` is the non-null start of the data's `len` bytes. `data` is
    // borrowed and not written for the life of the slice, which is copied out
    // before return.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    Ok(bytes.to_vec())
}

pub fn number_i32(value: i32) -> Result<Cf, CodecError> {
    // SAFETY: the pointer is to a live `i32`, which is the type
    // `kCFNumberSInt32Type` reads. The value is copied.
    let ptr = unsafe {
        sys::CFNumberCreate(
            ptr::null(),
            sys::CF_NUMBER_SINT32,
            (&raw const value).cast(),
        )
    };
    Cf::from_owned(ptr).ok_or(CodecError::Limit { what: "cf number" })
}

pub fn number_f64(value: f64) -> Result<Cf, CodecError> {
    // SAFETY: the pointer is to a live `f64`, which is the type
    // `kCFNumberFloat64Type` reads. The value is copied.
    let ptr = unsafe {
        sys::CFNumberCreate(
            ptr::null(),
            sys::CF_NUMBER_FLOAT64,
            (&raw const value).cast(),
        )
    };
    Cf::from_owned(ptr).ok_or(CodecError::Limit { what: "cf number" })
}

pub fn dictionary(keys: &[*const c_void], values: &[*const c_void]) -> Result<Cf, CodecError> {
    if keys.len() != values.len() {
        return Err(CodecError::InvalidApi {
            what: "cf dictionary",
        });
    }
    let count = sys::CFIndex::try_from(keys.len()).map_err(|_| CodecError::Limit {
        what: "cf dictionary",
    })?;
    // SAFETY: `keys` and `values` both hold `count` pointers and live through
    // the call. Callers pass live CF objects, which the CFType callbacks
    // retain, so the dictionary does not borrow from the slices. The callback
    // statics are the framework's own structs; only their address is passed.
    let ptr = unsafe {
        sys::CFDictionaryCreate(
            ptr::null(),
            keys.as_ptr(),
            values.as_ptr(),
            count,
            (&raw const sys::kCFTypeDictionaryKeyCallBacks).cast(),
            (&raw const sys::kCFTypeDictionaryValueCallBacks).cast(),
        )
    };
    Cf::from_owned(ptr).ok_or(CodecError::Limit {
        what: "cf dictionary",
    })
}

pub fn array(values: &[*const c_void]) -> Result<Cf, CodecError> {
    let count =
        sys::CFIndex::try_from(values.len()).map_err(|_| CodecError::Limit { what: "cf array" })?;
    // SAFETY: `values` holds `count` pointers to live CF objects for the call,
    // and the CFType callbacks retain each one. Only the callback static's
    // address is passed.
    let ptr = unsafe {
        sys::CFArrayCreate(
            ptr::null(),
            values.as_ptr(),
            count,
            (&raw const sys::kCFTypeArrayCallBacks).cast(),
        )
    };
    Cf::from_owned(ptr).ok_or(CodecError::Limit { what: "cf array" })
}

pub fn boolean(value: bool) -> *const c_void {
    // SAFETY: the two statics are CoreFoundation's immortal boolean
    // constants, set before any code here runs and never written.
    unsafe {
        if value {
            sys::kCFBooleanTrue
        } else {
            sys::kCFBooleanFalse
        }
    }
}

pub fn read_i32(number: *const c_void) -> Option<i32> {
    if number.is_null() {
        return None;
    }
    let mut value = 0i32;
    // SAFETY: `number` is non-null and callers pass a `CFNumber` they hold for
    // the call. The out pointer is a live `i32`, the type
    // `kCFNumberSInt32Type` writes.
    let ok =
        unsafe { sys::CFNumberGetValue(number, sys::CF_NUMBER_SINT32, (&raw mut value).cast()) };
    if ok == 0 { None } else { Some(value) }
}

pub fn time(value: i64, timescale: i32) -> sys::CMTime {
    sys::CMTime {
        value,
        timescale,
        flags: sys::TIME_VALID,
        epoch: 0,
    }
}

pub fn time_invalid() -> sys::CMTime {
    sys::CMTime {
        value: 0,
        timescale: 0,
        flags: 0,
        epoch: 0,
    }
}
