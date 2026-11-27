//! VA-API displays: one per DRM render node, opened once for the process.
//!
//! Every `/dev/dri/renderD128` to `renderD191` that opens and initialises is
//! kept with the profiles and entrypoints its driver offers. An opener takes
//! the first node that offers what it needs. `CUTLASS_VAAPI_DEVICE` names
//! one node to use instead, for a machine with two GPUs.
//!
//! libva is thread-safe across contexts on one display, so decoders and
//! encoders share a node's display, each with its own config and context.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::fs::File;
use std::os::fd::AsRawFd;
use std::panic::catch_unwind;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::OnceLock;

use super::abi;
use super::api::{self, ConfigAttrib, Display, Va};

/// A render node to use instead of probing them all.
pub const DEVICE_VAR: &str = "CUTLASS_VAAPI_DEVICE";

/// The first and last render node minors.
const NODES: std::ops::RangeInclusive<u32> = 128..=191;

/// One initialised display.
pub struct Node {
    pub va: &'static Va,
    pub display: Display,
    pub path: PathBuf,
    /// The driver's own name for itself.
    pub vendor: String,
    pub version: (i32, i32),
    /// Each profile with its entrypoints.
    pub profiles: Vec<(i32, Vec<i32>)>,
    /// Kept open for as long as the display, which uses its descriptor.
    _file: File,
}

// SAFETY: the display is never terminated, and libva takes calls on one
// display from any thread, for different contexts at once.
unsafe impl Send for Node {}
// SAFETY: as above; the fields are not written after the probe.
unsafe impl Sync for Node {}

impl Node {
    pub fn offers(&self, profile: i32, entrypoint: i32) -> bool {
        self.profiles
            .iter()
            .any(|(offered, entrypoints)| *offered == profile && entrypoints.contains(&entrypoint))
    }

    /// The values of `kinds` for a profile and entrypoint, with
    /// [`abi::VA_ATTRIB_NOT_SUPPORTED`] for each the driver does not know.
    pub fn attributes(&self, profile: i32, entrypoint: i32, kinds: &[c_int]) -> Vec<u32> {
        let mut attributes: Vec<ConfigAttrib> = kinds
            .iter()
            .map(|kind| ConfigAttrib {
                kind: *kind,
                value: 0,
            })
            .collect();
        let count = c_int::try_from(attributes.len()).unwrap_or(0);
        // SAFETY: the array holds `count` VAConfigAttribs.
        let status = unsafe {
            (self.va.get_config_attributes)(
                self.display,
                profile,
                entrypoint,
                attributes.as_mut_ptr().cast(),
                count,
            )
        };
        if status != abi::VA_STATUS_SUCCESS {
            return vec![abi::VA_ATTRIB_NOT_SUPPORTED; kinds.len()];
        }
        attributes.iter().map(|attribute| attribute.value).collect()
    }
}

/// Every display that initialised, or none when libva is missing.
pub fn nodes() -> &'static [Node] {
    static NODES_FOUND: OnceLock<Vec<Node>> = OnceLock::new();
    NODES_FOUND.get_or_init(probe)
}

/// The first node that offers `profile` at `entrypoint`.
pub fn find(profile: i32, entrypoint: i32) -> Option<&'static Node> {
    nodes().iter().find(|node| node.offers(profile, entrypoint))
}

fn probe() -> Vec<Node> {
    let Some(va) = api::va() else {
        return Vec::new();
    };
    if let Some(path) = std::env::var_os(DEVICE_VAR) {
        return open(va, Path::new(&path)).into_iter().collect();
    }
    NODES
        .filter_map(|minor| open(va, &PathBuf::from(format!("/dev/dri/renderD{minor}"))))
        .collect()
}

fn open(va: &'static Va, path: &Path) -> Option<Node> {
    let file = File::options().read(true).write(true).open(path).ok()?;
    // SAFETY: the descriptor is open and stays open in the node.
    let display = unsafe { (va.get_display_drm)(file.as_raw_fd()) };
    if display.is_null() {
        return None;
    }
    // SAFETY: the display is valid; the callbacks take any message and a
    // null context.
    unsafe {
        (va.set_error_callback)(display, Some(on_error), ptr::null_mut());
        (va.set_info_callback)(display, Some(on_info), ptr::null_mut());
    }
    let (mut major, mut minor) = (0, 0);
    // SAFETY: the display is valid and the out pointers are too.
    let status = unsafe { (va.initialize)(display, &raw mut major, &raw mut minor) };
    if status != abi::VA_STATUS_SUCCESS {
        cutlass_logger::debug!(
            "VA-API: {} does not initialise: {}",
            path.display(),
            va.error(status)
        );
        // SAFETY: a display that failed to initialise is still terminated.
        unsafe { (va.terminate)(display) };
        return None;
    }
    // SAFETY: the display is initialised; the string belongs to it.
    let vendor = unsafe { (va.query_vendor_string)(display) };
    let vendor = if vendor.is_null() {
        String::new()
    } else {
        // SAFETY: NUL-terminated and owned by the display.
        unsafe { CStr::from_ptr(vendor) }
            .to_string_lossy()
            .into_owned()
    };
    let profiles = profiles(va, display);
    Some(Node {
        va,
        display,
        path: path.to_path_buf(),
        vendor,
        version: (major, minor),
        profiles,
        _file: file,
    })
}

fn profiles(va: &Va, display: Display) -> Vec<(i32, Vec<i32>)> {
    // SAFETY: the display is initialised.
    let most = unsafe { (va.max_num_profiles)(display) }.clamp(0, 256);
    let mut list = vec![0; usize::try_from(most).unwrap_or(0)];
    let mut count = 0;
    // SAFETY: the list holds the most the driver says it returns.
    let status = unsafe { (va.query_config_profiles)(display, list.as_mut_ptr(), &raw mut count) };
    if status != abi::VA_STATUS_SUCCESS {
        return Vec::new();
    }
    list.truncate(usize::try_from(count).unwrap_or(0));
    // SAFETY: the display is initialised.
    let most = unsafe { (va.max_num_entrypoints)(display) }.clamp(0, 64);
    list.into_iter()
        .filter(|profile| *profile != abi::VAProfileNone)
        .map(|profile| {
            let mut entrypoints = vec![0; usize::try_from(most).unwrap_or(0)];
            let mut count = 0;
            // SAFETY: the list holds the most the driver says it returns.
            let status = unsafe {
                (va.query_config_entrypoints)(
                    display,
                    profile,
                    entrypoints.as_mut_ptr(),
                    &raw mut count,
                )
            };
            if status == abi::VA_STATUS_SUCCESS {
                entrypoints.truncate(usize::try_from(count).unwrap_or(0));
            } else {
                entrypoints.clear();
            }
            (profile, entrypoints)
        })
        .collect()
}

fn message(text: *const c_char) -> String {
    if text.is_null() {
        return String::new();
    }
    // SAFETY: libva passes a NUL-terminated message.
    unsafe { CStr::from_ptr(text) }
        .to_string_lossy()
        .trim_end()
        .to_owned()
}

/// libva's errors, in the log instead of on stderr.
unsafe extern "C" fn on_error(_context: *mut c_void, text: *const c_char) {
    let _ = catch_unwind(|| cutlass_logger::warn!("libva: {}", message(text)));
}

/// libva's notes, at debug level.
unsafe extern "C" fn on_info(_context: *mut c_void, text: *const c_char) {
    let _ = catch_unwind(|| cutlass_logger::debug!("libva: {}", message(text)));
}

/// A profile's name, for the probe.
pub fn profile_name(profile: i32) -> String {
    match profile {
        abi::VAProfileH264ConstrainedBaseline => "H.264 Constrained Baseline".into(),
        abi::VAProfileH264Main => "H.264 Main".into(),
        abi::VAProfileH264High => "H.264 High".into(),
        abi::VAProfileHEVCMain => "HEVC Main".into(),
        abi::VAProfileHEVCMain10 => "HEVC Main 10".into(),
        abi::VAProfileAV1Profile0 => "AV1 Main".into(),
        other => format!("profile {other}"),
    }
}

/// An entrypoint's name, for the probe.
pub fn entrypoint_name(entrypoint: i32) -> String {
    match entrypoint {
        abi::VAEntrypointVLD => "decode".into(),
        abi::VAEntrypointEncSlice => "encode".into(),
        abi::VAEntrypointEncSliceLP => "encode (low power)".into(),
        other => format!("entrypoint {other}"),
    }
}

/// Lines for the probe: each node, then the profiles this crate uses.
pub fn report() -> Vec<String> {
    if api::va().is_none() {
        return vec![String::from(
            "VA-API: libva.so.2 or libva-drm.so.2 is missing",
        )];
    }
    let nodes = nodes();
    if nodes.is_empty() {
        return vec![String::from("VA-API: no render node initialises")];
    }
    let wanted = [
        abi::VAProfileH264ConstrainedBaseline,
        abi::VAProfileH264Main,
        abi::VAProfileH264High,
        abi::VAProfileHEVCMain,
        abi::VAProfileHEVCMain10,
        abi::VAProfileAV1Profile0,
    ];
    let mut lines = Vec::new();
    for node in nodes {
        lines.push(format!(
            "VA-API {}: {} (VA-API {}.{})",
            node.path.display(),
            node.vendor,
            node.version.0,
            node.version.1
        ));
        for (profile, entrypoints) in &node.profiles {
            if !wanted.contains(profile) {
                continue;
            }
            let names: Vec<String> = entrypoints.iter().map(|e| entrypoint_name(*e)).collect();
            lines.push(format!(
                "  {}: {}",
                profile_name(*profile),
                names.join(", ")
            ));
        }
    }
    lines
}
