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
            unsafe { sys::CFRelease(self.ptr) }
        }
    }
}

unsafe impl Send for Cf {}

pub fn string(text: &str) -> Result<Cf, CodecError> {
    let mut bytes = Vec::with_capacity(text.len() + 1);
    bytes.extend_from_slice(text.as_bytes());
    bytes.push(0);
    let ptr = unsafe { sys::CFStringCreateWithCString(ptr::null(), bytes.as_ptr(), sys::UTF8) };
    Cf::from_owned(ptr).ok_or(CodecError::Limit { what: "cf string" })
}

pub fn data(bytes: &[u8]) -> Result<Cf, CodecError> {
    let len =
        sys::CFIndex::try_from(bytes.len()).map_err(|_| CodecError::Limit { what: "cf data" })?;
    let ptr = unsafe { sys::CFDataCreate(ptr::null(), bytes.as_ptr(), len) };
    Cf::from_owned(ptr).ok_or(CodecError::Limit { what: "cf data" })
}

pub fn data_mut() -> Result<Cf, CodecError> {
    let ptr = unsafe { sys::CFDataCreateMutable(ptr::null(), 0) };
    Cf::from_owned(ptr).ok_or(CodecError::Limit { what: "cf data" })
}

pub fn data_bytes(data: &Cf) -> Result<Vec<u8>, CodecError> {
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
    let ptr = unsafe { sys::CFDataGetBytePtr(data.as_ptr()) };
    if ptr.is_null() {
        return Err(CodecError::Malformed {
            at_bit: 0,
            what: "cf data",
        });
    }
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    Ok(bytes.to_vec())
}

pub fn number_i32(value: i32) -> Result<Cf, CodecError> {
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

pub fn boolean(value: bool) -> *const c_void {
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
