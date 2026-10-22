//! Codecs a VA-API render node lists. Profile numbers are the libva enum.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::fs::OpenOptions;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

use cutlass_core::VideoCodec;

const RTLD_NOW: c_int = 2;
const RTLD_GLOBAL: c_int = 0x00100;
const MAX_PROFILES: c_int = 64;
const MAX_ENTRYPOINTS: c_int = 32;
const MAX_NODES: usize = 8;

/// Decode and encode codecs one or more render nodes listed.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Capabilities {
    decode: Vec<VideoCodec>,
    encode: Vec<VideoCodec>,
}

impl Capabilities {
    /// Asks every render node. A missing library or a node that will not open
    /// is skipped.
    pub fn query() -> Self {
        let Some(va) = Library::open(c"libva.so.2", RTLD_NOW | RTLD_GLOBAL) else {
            return Self::default();
        };
        let Some(drm) = Library::open(c"libva-drm.so.2", RTLD_NOW) else {
            return Self::default();
        };
        let Some(get_display) = load(&drm, c"vaGetDisplayDRM") else {
            return Self::default();
        };
        let Some(initialize) = load(&va, c"vaInitialize") else {
            return Self::default();
        };
        let Some(terminate) = load(&va, c"vaTerminate") else {
            return Self::default();
        };
        let Some(max_profiles) = load(&va, c"vaMaxNumProfiles") else {
            return Self::default();
        };
        let Some(max_entrypoints) = load(&va, c"vaMaxNumEntrypoints") else {
            return Self::default();
        };
        let Some(query_profiles) = load(&va, c"vaQueryConfigProfiles") else {
            return Self::default();
        };
        let Some(query_entrypoints) = load(&va, c"vaQueryConfigEntrypoints") else {
            return Self::default();
        };

        let api = Api {
            get_display,
            initialize,
            terminate,
            max_profiles,
            max_entrypoints,
            query_profiles,
            query_entrypoints,
        };
        let mut caps = Self::default();
        for node in render_nodes() {
            query_node(&node, &api, &mut caps);
        }
        caps
    }

    /// True when a node listed a decode entrypoint for `codec`.
    pub fn decodes(&self, codec: VideoCodec) -> bool {
        self.decode.contains(&codec)
    }

    /// True when a node listed an encode entrypoint for `codec`.
    pub fn encodes(&self, codec: VideoCodec) -> bool {
        self.encode.contains(&codec)
    }
}

#[derive(Clone, Copy)]
enum Direction {
    Decode,
    Encode,
}

/// `profile` is a `VAProfile`. Unknown and non-video profiles are skipped.
fn video_codec(profile: i32) -> Option<VideoCodec> {
    match profile {
        0..=1 => Some(VideoCodec::Mpeg2),
        2..=4 => Some(VideoCodec::Mpeg4Part2),
        5 | 6 | 7 | 13 | 15 | 16 | 36 => Some(VideoCodec::H264),
        12 => Some(VideoCodec::Mjpeg),
        14 => Some(VideoCodec::Vp8),
        17 | 18 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 34 => Some(VideoCodec::Hevc),
        19..=22 => Some(VideoCodec::Vp9),
        32..=33 => Some(VideoCodec::Av1),
        _ => None,
    }
}

/// `entrypoint` is a `VAEntrypoint`. Processing and statistics are neither.
fn direction(entrypoint: i32) -> Option<Direction> {
    match entrypoint {
        1 | 3 | 4 => Some(Direction::Decode),
        6..=8 => Some(Direction::Encode),
        _ => None,
    }
}

fn push_codec(list: &mut Vec<VideoCodec>, codec: VideoCodec) {
    if !list.contains(&codec) {
        list.push(codec);
    }
}

fn render_nodes() -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir("/dev/dri") else {
        return Vec::new();
    };
    let mut nodes = Vec::new();
    for entry in entries {
        let Ok(entry) = entry else {
            continue;
        };
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if is_render_node(name) {
            nodes.push(entry.path());
        }
    }
    nodes.sort();
    nodes.truncate(MAX_NODES);
    nodes
}

fn is_render_node(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("renderD") else {
        return false;
    };
    !rest.is_empty() && rest.bytes().all(|byte| byte.is_ascii_digit())
}

struct Api {
    get_display: GetDisplay,
    initialize: Initialize,
    terminate: Terminate,
    max_profiles: MaxCount,
    max_entrypoints: MaxCount,
    query_profiles: QueryProfiles,
    query_entrypoints: QueryEntrypoints,
}

fn query_node(path: &Path, api: &Api, caps: &mut Capabilities) {
    let Ok(file) = OpenOptions::new().read(true).write(true).open(path) else {
        return;
    };
    // SAFETY: `fd` stays open until this function returns, which is after
    // `vaTerminate`. `vaGetDisplayDRM` returns null when the node is not a
    // VA-API device.
    let display = unsafe { (api.get_display)(file.as_raw_fd()) };
    if display.is_null() {
        return;
    }
    let mut major = 0;
    let mut minor = 0;
    // SAFETY: `display` came from `vaGetDisplayDRM` and the version pointers
    // are local.
    let status = unsafe { (api.initialize)(display, &mut major, &mut minor) };
    if status != 0 {
        return;
    }
    let guard = DisplayGuard {
        display,
        terminate: api.terminate,
    };
    read_profiles(guard.display, api, caps);
}

fn read_profiles(display: *mut c_void, api: &Api, caps: &mut Capabilities) {
    // SAFETY: `display` was initialized and is not yet terminated.
    let profiles = unsafe { (api.max_profiles)(display) };
    if profiles <= 0 || profiles > MAX_PROFILES {
        return;
    }
    let Ok(count) = usize::try_from(profiles) else {
        return;
    };
    let mut listed = vec![0i32; count];
    let mut found = 0;
    // SAFETY: `listed` holds `profiles` entries, which is what VA-API asks for.
    let status = unsafe { (api.query_profiles)(display, listed.as_mut_ptr(), &mut found) };
    if status != 0 || found < 0 || found > profiles {
        return;
    }
    let found = usize::try_from(found).expect("the profile count is non-negative");
    for profile in listed.into_iter().take(found) {
        let Some(codec) = video_codec(profile) else {
            continue;
        };
        read_entrypoints(display, profile, codec, api, caps);
    }
}

fn read_entrypoints(
    display: *mut c_void,
    profile: i32,
    codec: VideoCodec,
    api: &Api,
    caps: &mut Capabilities,
) {
    // SAFETY: `display` was initialized and is not yet terminated.
    let entrypoints = unsafe { (api.max_entrypoints)(display) };
    if entrypoints <= 0 || entrypoints > MAX_ENTRYPOINTS {
        return;
    }
    let Ok(count) = usize::try_from(entrypoints) else {
        return;
    };
    let mut listed = vec![0i32; count];
    let mut found = 0;
    // SAFETY: `listed` holds `entrypoints` entries, which is what VA-API asks for.
    let status =
        unsafe { (api.query_entrypoints)(display, profile, listed.as_mut_ptr(), &mut found) };
    if status != 0 || found < 0 || found > entrypoints {
        return;
    }
    let found = usize::try_from(found).expect("the entrypoint count is non-negative");
    for entrypoint in listed.into_iter().take(found) {
        match direction(entrypoint) {
            Some(Direction::Decode) => push_codec(&mut caps.decode, codec),
            Some(Direction::Encode) => push_codec(&mut caps.encode, codec),
            None => {}
        }
    }
}

struct DisplayGuard {
    display: *mut c_void,
    terminate: Terminate,
}

impl Drop for DisplayGuard {
    fn drop(&mut self) {
        if self.display.is_null() {
            return;
        }
        // SAFETY: `display` was initialized and this guard is the only owner.
        unsafe { (self.terminate)(self.display) };
    }
}

struct Library {
    handle: *mut c_void,
}

impl Library {
    fn open(name: &CStr, flags: c_int) -> Option<Self> {
        // SAFETY: `name` is a nul-terminated literal. A null handle means the
        // library is not installed.
        let handle = unsafe { dlopen(name.as_ptr(), flags) };
        if handle.is_null() {
            None
        } else {
            Some(Self { handle })
        }
    }

    fn symbol(&self, name: &CStr) -> Option<*mut c_void> {
        // SAFETY: `handle` came from `dlopen` and is still open. A null symbol
        // means the library does not export `name`.
        let symbol = unsafe { dlsym(self.handle, name.as_ptr()) };
        if symbol.is_null() { None } else { Some(symbol) }
    }
}

fn load<T>(library: &Library, name: &CStr) -> Option<T> {
    const {
        assert!(
            std::mem::size_of::<T>() == std::mem::size_of::<*mut c_void>(),
            "a loaded symbol is one pointer"
        );
    }
    let symbol = library.symbol(name)?;
    // SAFETY: `T` is the C function type of `name`, and the assert makes it
    // the same size as the symbol pointer.
    Some(unsafe { std::mem::transmute_copy(&symbol) })
}

impl Drop for Library {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        // SAFETY: `handle` came from `dlopen` and has not been closed.
        unsafe { dlclose(self.handle) };
    }
}

type GetDisplay = unsafe extern "C" fn(c_int) -> *mut c_void;
type Initialize = unsafe extern "C" fn(*mut c_void, *mut c_int, *mut c_int) -> c_int;
type Terminate = unsafe extern "C" fn(*mut c_void) -> c_int;
type MaxCount = unsafe extern "C" fn(*mut c_void) -> c_int;
type QueryProfiles = unsafe extern "C" fn(*mut c_void, *mut i32, *mut c_int) -> c_int;
type QueryEntrypoints = unsafe extern "C" fn(*mut c_void, i32, *mut i32, *mut c_int) -> c_int;

// SAFETY: these are the POSIX dynamic-loader functions. They have no
// preconditions beyond a nul-terminated name and a handle from `dlopen`.
#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_map_to_the_video_codecs() {
        assert_eq!(video_codec(0), Some(VideoCodec::Mpeg2));
        assert_eq!(video_codec(1), Some(VideoCodec::Mpeg2));
        assert_eq!(video_codec(4), Some(VideoCodec::Mpeg4Part2));
        assert_eq!(video_codec(6), Some(VideoCodec::H264));
        assert_eq!(video_codec(13), Some(VideoCodec::H264));
        assert_eq!(video_codec(36), Some(VideoCodec::H264));
        assert_eq!(video_codec(12), Some(VideoCodec::Mjpeg));
        assert_eq!(video_codec(14), Some(VideoCodec::Vp8));
        assert_eq!(video_codec(17), Some(VideoCodec::Hevc));
        assert_eq!(video_codec(34), Some(VideoCodec::Hevc));
        assert_eq!(video_codec(19), Some(VideoCodec::Vp9));
        assert_eq!(video_codec(32), Some(VideoCodec::Av1));
        assert_eq!(video_codec(33), Some(VideoCodec::Av1));
        assert_eq!(video_codec(-1), None);
        assert_eq!(video_codec(35), None);
        assert_eq!(video_codec(37), None);
        assert_eq!(video_codec(99), None);
    }

    #[test]
    fn entrypoints_are_decode_or_encode() {
        assert!(matches!(direction(1), Some(Direction::Decode)));
        assert!(matches!(direction(3), Some(Direction::Decode)));
        assert!(matches!(direction(6), Some(Direction::Encode)));
        assert!(matches!(direction(7), Some(Direction::Encode)));
        assert!(matches!(direction(8), Some(Direction::Encode)));
        assert!(direction(10).is_none());
        assert!(direction(0).is_none());
    }

    #[test]
    fn a_render_node_name_is_render_d_and_digits() {
        assert!(is_render_node("renderD128"));
        assert!(!is_render_node("renderD"));
        assert!(!is_render_node("card0"));
        assert!(!is_render_node("renderD128x"));
    }

    #[test]
    fn prores_is_not_a_va_profile() {
        for profile in -1..40 {
            assert_ne!(video_codec(profile), Some(VideoCodec::ProRes));
        }
    }
}
