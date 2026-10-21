//! Core Foundation ownership, dictionaries, and session properties.

use std::ptr::{self, NonNull};

use objc2_core_foundation::{CFDictionary, CFRetained, CFString, CFType, Type};
use objc2_video_toolbox::{
    VTSessionCopyProperty, VTSessionSetProperty, kVTPropertyNotSupportedErr,
};

use crate::error::{Error, check};

/// Takes ownership of the result of a Core Foundation Create or Copy call.
///
/// # Safety
///
/// `object` must be null or a reference the caller owns, as returned by a
/// function with Create or Copy in its name.
pub(crate) unsafe fn owned<T: Type>(
    object: *mut T,
    call: &'static str,
) -> Result<CFRetained<T>, Error> {
    let object = NonNull::new(object).ok_or(Error::NullOutput { call })?;
    // SAFETY: the caller guarantees `object` is an owned reference.
    Ok(unsafe { CFRetained::from_raw(object) })
}

pub(crate) fn dictionary(
    entries: &[(&CFString, &CFType)],
) -> CFRetained<CFDictionary<CFString, CFType>> {
    let (keys, values): (Vec<&CFString>, Vec<&CFType>) = entries.iter().copied().unzip();
    CFDictionary::from_slices(&keys, &values)
}

/// Sets a property on a VideoToolbox session. `name` goes into the error.
pub(crate) fn set_property(
    session: &CFType,
    name: &'static str,
    key: &CFString,
    value: &CFType,
) -> Result<(), Error> {
    // SAFETY: `session` is a live VideoToolbox session, and the value type
    // matches the key, as each caller documents.
    let status = unsafe { VTSessionSetProperty(session, key, Some(value)) };
    check(name, status)
}

/// Reads a property from a VideoToolbox session. `None` when the session
/// doesn't have it.
pub(crate) fn copy_property(
    session: &CFType,
    key: &CFString,
) -> Result<Option<CFRetained<CFType>>, Error> {
    let mut value: *const CFType = ptr::null();
    // SAFETY: `session` is a live VideoToolbox session, and `value` is a
    // valid place for one CF object reference.
    let status =
        unsafe { VTSessionCopyProperty(session, key, None, ptr::from_mut(&mut value).cast()) };
    if status == kVTPropertyNotSupportedErr {
        return Ok(None);
    }
    check("VTSessionCopyProperty", status)?;
    // SAFETY: Copy functions return an owned reference or null.
    Ok(NonNull::new(value.cast_mut()).map(|value| unsafe { CFRetained::from_raw(value) }))
}
