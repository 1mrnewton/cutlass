//! The libva calls this crate makes, from `libva.so.2` and
//! `libva-drm.so.2`, opened once.

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::sync::OnceLock;

use crate::dl::Library;

pub type Display = *mut c_void;
pub type Status = c_int;
pub type Id = c_uint;

pub type MessageCallback = unsafe extern "C" fn(*mut c_void, *const c_char);

type GetDisplayDrm = unsafe extern "C" fn(c_int) -> Display;
type Initialize = unsafe extern "C" fn(Display, *mut c_int, *mut c_int) -> Status;
type Terminate = unsafe extern "C" fn(Display) -> Status;
type SetCallback =
    unsafe extern "C" fn(Display, Option<MessageCallback>, *mut c_void) -> Option<MessageCallback>;
type QueryVendorString = unsafe extern "C" fn(Display) -> *const c_char;
type MaxNum = unsafe extern "C" fn(Display) -> c_int;
type QueryProfiles = unsafe extern "C" fn(Display, *mut c_int, *mut c_int) -> Status;
type QueryEntrypoints = unsafe extern "C" fn(Display, c_int, *mut c_int, *mut c_int) -> Status;
type GetConfigAttributes =
    unsafe extern "C" fn(Display, c_int, c_int, *mut c_void, c_int) -> Status;
type CreateConfig =
    unsafe extern "C" fn(Display, c_int, c_int, *mut c_void, c_int, *mut Id) -> Status;
type DestroyId = unsafe extern "C" fn(Display, Id) -> Status;
type CreateSurfaces = unsafe extern "C" fn(
    Display,
    c_uint,
    c_uint,
    c_uint,
    *mut Id,
    c_uint,
    *mut c_void,
    c_uint,
) -> Status;
type DestroySurfaces = unsafe extern "C" fn(Display, *mut Id, c_int) -> Status;
type CreateContext =
    unsafe extern "C" fn(Display, Id, c_int, c_int, c_int, *mut Id, c_int, *mut Id) -> Status;
type CreateBuffer =
    unsafe extern "C" fn(Display, Id, c_int, c_uint, c_uint, *mut c_void, *mut Id) -> Status;
type MapBuffer = unsafe extern "C" fn(Display, Id, *mut *mut c_void) -> Status;
type BeginPicture = unsafe extern "C" fn(Display, Id, Id) -> Status;
type RenderPicture = unsafe extern "C" fn(Display, Id, *mut Id, c_int) -> Status;
type DeriveImage = unsafe extern "C" fn(Display, Id, *mut c_void) -> Status;
type CreateImage = unsafe extern "C" fn(Display, *mut c_void, c_int, c_int, *mut c_void) -> Status;
type GetImage = unsafe extern "C" fn(Display, Id, c_int, c_int, c_uint, c_uint, Id) -> Status;
type PutImage = unsafe extern "C" fn(
    Display,
    Id,
    Id,
    c_int,
    c_int,
    c_uint,
    c_uint,
    c_int,
    c_int,
    c_uint,
    c_uint,
) -> Status;
type QueryImageFormats = unsafe extern "C" fn(Display, *mut c_void, *mut c_int) -> Status;
type ErrorStr = unsafe extern "C" fn(Status) -> *const c_char;

/// libva's entry points.
pub struct Va {
    pub get_display_drm: GetDisplayDrm,
    pub initialize: Initialize,
    pub terminate: Terminate,
    pub set_error_callback: SetCallback,
    pub set_info_callback: SetCallback,
    pub query_vendor_string: QueryVendorString,
    pub max_num_profiles: MaxNum,
    pub max_num_entrypoints: MaxNum,
    pub max_num_image_formats: MaxNum,
    pub query_config_profiles: QueryProfiles,
    pub query_config_entrypoints: QueryEntrypoints,
    pub get_config_attributes: GetConfigAttributes,
    pub create_config: CreateConfig,
    pub destroy_config: DestroyId,
    pub create_surfaces: CreateSurfaces,
    pub destroy_surfaces: DestroySurfaces,
    pub create_context: CreateContext,
    pub destroy_context: DestroyId,
    pub create_buffer: CreateBuffer,
    pub destroy_buffer: DestroyId,
    pub map_buffer: MapBuffer,
    pub unmap_buffer: DestroyId,
    pub begin_picture: BeginPicture,
    pub render_picture: RenderPicture,
    pub end_picture: DestroyId,
    pub sync_surface: DestroyId,
    pub derive_image: DeriveImage,
    pub create_image: CreateImage,
    pub get_image: GetImage,
    pub put_image: PutImage,
    pub destroy_image: DestroyId,
    pub query_image_formats: QueryImageFormats,
    error_str: ErrorStr,
    _drm: Library,
    _va: Library,
}

/// libva, or `None` when it is not installed.
pub fn va() -> Option<&'static Va> {
    static VA: OnceLock<Option<Va>> = OnceLock::new();
    VA.get_or_init(Va::load).as_ref()
}

impl Va {
    fn load() -> Option<Self> {
        let va = Library::open(&[c"libva.so.2"])?;
        let drm = Library::open(&[c"libva-drm.so.2"])?;
        // SAFETY: each symbol has this signature in va/va.h or
        // va/va_drm.h (libva 2.x); the pointers live in the same struct as
        // both libraries.
        unsafe {
            Some(Self {
                get_display_drm: drm.symbol(c"vaGetDisplayDRM")?,
                initialize: va.symbol(c"vaInitialize")?,
                terminate: va.symbol(c"vaTerminate")?,
                set_error_callback: va.symbol(c"vaSetErrorCallback")?,
                set_info_callback: va.symbol(c"vaSetInfoCallback")?,
                query_vendor_string: va.symbol(c"vaQueryVendorString")?,
                max_num_profiles: va.symbol(c"vaMaxNumProfiles")?,
                max_num_entrypoints: va.symbol(c"vaMaxNumEntrypoints")?,
                max_num_image_formats: va.symbol(c"vaMaxNumImageFormats")?,
                query_config_profiles: va.symbol(c"vaQueryConfigProfiles")?,
                query_config_entrypoints: va.symbol(c"vaQueryConfigEntrypoints")?,
                get_config_attributes: va.symbol(c"vaGetConfigAttributes")?,
                create_config: va.symbol(c"vaCreateConfig")?,
                destroy_config: va.symbol(c"vaDestroyConfig")?,
                create_surfaces: va.symbol(c"vaCreateSurfaces")?,
                destroy_surfaces: va.symbol(c"vaDestroySurfaces")?,
                create_context: va.symbol(c"vaCreateContext")?,
                destroy_context: va.symbol(c"vaDestroyContext")?,
                create_buffer: va.symbol(c"vaCreateBuffer")?,
                destroy_buffer: va.symbol(c"vaDestroyBuffer")?,
                map_buffer: va.symbol(c"vaMapBuffer")?,
                unmap_buffer: va.symbol(c"vaUnmapBuffer")?,
                begin_picture: va.symbol(c"vaBeginPicture")?,
                render_picture: va.symbol(c"vaRenderPicture")?,
                end_picture: va.symbol(c"vaEndPicture")?,
                sync_surface: va.symbol(c"vaSyncSurface")?,
                derive_image: va.symbol(c"vaDeriveImage")?,
                create_image: va.symbol(c"vaCreateImage")?,
                get_image: va.symbol(c"vaGetImage")?,
                put_image: va.symbol(c"vaPutImage")?,
                destroy_image: va.symbol(c"vaDestroyImage")?,
                query_image_formats: va.symbol(c"vaQueryImageFormats")?,
                error_str: va.symbol(c"vaErrorStr")?,
                _drm: drm,
                _va: va,
            })
        }
    }

    /// libva's words for a status.
    pub fn error(&self, status: Status) -> String {
        // SAFETY: vaErrorStr takes any status and returns a static string.
        let text = unsafe { (self.error_str)(status) };
        if text.is_null() {
            return format!("status {status}");
        }
        // SAFETY: the string is NUL-terminated and static.
        let text = unsafe { CStr::from_ptr(text) }.to_string_lossy();
        format!("{text} ({status})")
    }
}

/// `VAConfigAttrib`: a type and its value.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ConfigAttrib {
    pub kind: c_int,
    pub value: u32,
}

#[cfg(test)]
mod tests {
    use super::super::abi;
    use super::*;

    #[test]
    fn the_config_attribute_has_the_c_layout() {
        assert_eq!(size_of::<ConfigAttrib>(), abi::config_attrib::SIZE);
        assert_eq!(
            std::mem::offset_of!(ConfigAttrib, kind),
            abi::config_attrib::TYPE
        );
        assert_eq!(
            std::mem::offset_of!(ConfigAttrib, value),
            abi::config_attrib::VALUE
        );
    }
}
