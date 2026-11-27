//! Shared libraries opened at run time.
//!
//! ALSA, libva, CUDA, NVDEC, and NVENC are all opened with `dlopen`, so the
//! build needs none of their headers and a system without one of them still
//! starts. A library that is missing, or lacks a symbol, is `None`, and the
//! code that wanted it declines.

#![allow(unsafe_code)]

use std::ffi::{CStr, c_char, c_int, c_void};

/// `RTLD_NOW`.
const LOAD_NOW: c_int = 2;

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(file: *const c_char, mode: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}

/// One loaded library. Closed on drop.
pub(crate) struct Library {
    handle: *mut c_void,
}

// SAFETY: a `dlopen` handle is a process-wide token. `dlsym` and `dlclose`
// may be called on it from any thread.
unsafe impl Send for Library {}
// SAFETY: as above; the handle is never written after open.
unsafe impl Sync for Library {}

impl Library {
    /// The first of `names` that loads, by soname.
    pub(crate) fn open(names: &[&CStr]) -> Option<Self> {
        names.iter().find_map(|name| {
            // SAFETY: the name is a valid C string, and dlopen has no other
            // precondition.
            let handle = unsafe { dlopen(name.as_ptr(), LOAD_NOW) };
            // Not `then_some`: that builds the value first, and dropping a
            // null handle would pass it to dlclose, which faults on it.
            if handle.is_null() {
                None
            } else {
                Some(Self { handle })
            }
        })
    }

    /// Look `name` up as a function of type `F`.
    ///
    /// # Safety
    ///
    /// `F` must be the function pointer type of the symbol, and the
    /// pointer must not be called after this library is dropped.
    pub(crate) unsafe fn symbol<F: Copy>(&self, name: &CStr) -> Option<F> {
        // SAFETY: the handle is live and the name is a valid C string.
        let found = unsafe { dlsym(self.handle, name.as_ptr()) };
        if found.is_null() || size_of::<F>() != size_of::<*mut c_void>() {
            return None;
        }
        // SAFETY: the caller names the symbol's type, which is a function
        // pointer the size of the address.
        Some(unsafe { std::mem::transmute_copy::<*mut c_void, F>(&found) })
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        // SAFETY: the handle came from dlopen, and the owner keeps no
        // pointer from it past this drop.
        unsafe { dlclose(self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_library_is_none() {
        assert!(Library::open(&[c"libcutlass-does-not-exist.so.9"]).is_none());
    }

    #[test]
    fn libc_loads_and_a_missing_symbol_is_none() {
        let Some(libc) = Library::open(&[c"libc.so.6"]) else {
            return;
        };
        type Getpid = unsafe extern "C" fn() -> c_int;
        // SAFETY: `getpid` has this signature in unistd.h.
        let getpid = unsafe { libc.symbol::<Getpid>(c"getpid") };
        assert!(getpid.is_some());
        // SAFETY: the type is never called.
        let missing = unsafe { libc.symbol::<Getpid>(c"cutlass_no_such_symbol") };
        assert!(missing.is_none());
    }
}
