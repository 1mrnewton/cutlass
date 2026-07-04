//! JSON-over-C-strings marshalling shared by the session FFI.
//!
//! Commands, intents, UI state, and outcomes cross the boundary as UTF-8 JSON:
//! inputs arrive as `(ptr, len)` byte slices (no NUL needed), outputs leave as
//! NUL-terminated heap strings the caller must release with
//! [`cutlass_string_free`]. Responses share one envelope so every shell parses
//! a single shape:
//!
//! ```json
//! {"ok": <payload>, "revision": 41}
//! {"err": {"kind": "model", "message": "clip 7 not found"}}
//! ```

use std::ffi::{CString, c_char};

use cutlass_engine::EngineError;

/// Borrow a `(ptr, len)` FFI argument as `&str`. `None` for null/empty
/// pointers or invalid UTF-8.
///
/// # Safety
/// `ptr` must point to `len` initialized bytes when non-null.
pub unsafe fn str_arg<'a>(ptr: *const u8, len: usize) -> Option<&'a str> {
    if ptr.is_null() || len == 0 {
        return None;
    }
    // SAFETY: caller guarantees `len` initialized bytes at `ptr`.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    std::str::from_utf8(bytes).ok()
}

/// Hand a Rust string to C as a NUL-terminated heap allocation. The caller
/// owns it until [`cutlass_string_free`]. JSON never contains NUL bytes, but
/// if one sneaks in the string is truncated there rather than dropped.
pub fn to_c_string(s: String) -> *mut c_char {
    let cstring = match CString::new(s) {
        Ok(cstring) => cstring,
        Err(err) => {
            let mut bytes = err.into_vec();
            let nul = bytes.iter().position(|b| *b == 0).unwrap_or(0);
            bytes.truncate(nul);
            CString::new(bytes).expect("truncated at first NUL")
        }
    };
    cstring.into_raw()
}

/// A success envelope: `{"ok": <payload>, "revision": n}`.
pub fn ok_response(payload: serde_json::Value, revision: u64) -> String {
    serde_json::json!({ "ok": payload, "revision": revision }).to_string()
}

/// An error envelope: `{"err": {"kind": …, "message": …}}`. `kind` values are
/// [`EngineError::kind`] — a stable part of the protocol.
pub fn err_response(error: &EngineError) -> String {
    serde_json::json!({
        "err": { "kind": error.kind(), "message": error.to_string() }
    })
    .to_string()
}

/// An error envelope for failures that happen before a command reaches the
/// engine (unparseable JSON, bad handle, invalid UTF-8).
pub fn protocol_err_response(message: impl AsRef<str>) -> String {
    serde_json::json!({
        "err": { "kind": "protocol", "message": message.as_ref() }
    })
    .to_string()
}

/// Free a string returned by any `cutlass_*` call that documents it. Null is
/// a no-op.
///
/// # Safety
/// `ptr` must be null or a pointer previously returned by this crate's string
/// APIs, not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cutlass_string_free(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: `ptr` came from `CString::into_raw` in `to_c_string`.
    drop(unsafe { CString::from_raw(ptr) });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn str_arg_rejects_null_empty_and_bad_utf8() {
        // SAFETY: null/zero-len are the guarded cases under test.
        unsafe {
            assert_eq!(str_arg(std::ptr::null(), 4), None);
            assert_eq!(str_arg(b"abc".as_ptr(), 0), None);
            assert_eq!(str_arg([0xffu8, 0xfe].as_ptr(), 2), None);
            assert_eq!(str_arg(b"abc".as_ptr(), 3), Some("abc"));
        }
    }

    #[test]
    fn c_string_roundtrips_and_truncates_at_nul() {
        let ptr = to_c_string("hello".into());
        // SAFETY: `ptr` came from `to_c_string` above.
        let back = unsafe { CString::from_raw(ptr) };
        assert_eq!(back.to_str().unwrap(), "hello");

        let ptr = to_c_string("cut\0lass".into());
        // SAFETY: same.
        let back = unsafe { CString::from_raw(ptr) };
        assert_eq!(back.to_str().unwrap(), "cut");
    }

    #[test]
    fn envelopes_have_the_documented_shape() {
        let ok: serde_json::Value =
            serde_json::from_str(&ok_response(serde_json::json!({"clip": 3}), 7)).unwrap();
        assert_eq!(ok["ok"]["clip"], 3);
        assert_eq!(ok["revision"], 7);

        let err: serde_json::Value =
            serde_json::from_str(&protocol_err_response("bad json")).unwrap();
        assert_eq!(err["err"]["kind"], "protocol");
        assert_eq!(err["err"]["message"], "bad json");
    }
}
