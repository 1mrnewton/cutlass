//! Which hardware backends may open, and a note of which one did.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock, PoisonError};

use cutlass_decode::CodecId;

/// Names the backends that may open, comma-separated: `nvidia`, `vaapi`,
/// or `none`. Unset, both may. For trying one backend on a machine that
/// has both.
pub const BACKEND_VAR: &str = "CUTLASS_VIDEO_BACKEND";

/// The backends that may open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Backends {
    pub nvidia: bool,
    pub vaapi: bool,
}

impl Backends {
    /// From [`BACKEND_VAR`].
    pub fn from_env() -> Self {
        Self::parse(std::env::var(BACKEND_VAR).ok().as_deref())
    }

    fn parse(value: Option<&str>) -> Self {
        let all = Self {
            nvidia: true,
            vaapi: true,
        };
        let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
            return all;
        };
        let mut chosen = Self {
            nvidia: false,
            vaapi: false,
        };
        for name in value
            .split(',')
            .map(|name| name.trim().to_ascii_lowercase())
        {
            match name.as_str() {
                "nvidia" | "nvdec" | "nvenc" | "cuda" => chosen.nvidia = true,
                "vaapi" | "va-api" | "va" => chosen.vaapi = true,
                "none" | "off" => {}
                _ => {
                    cutlass_logger::warn!("{BACKEND_VAR} names an unknown backend: {name}");
                    return all;
                }
            }
        }
        chosen
    }
}

type Seen = (&'static str, CodecId, bool, &'static str);

/// Whether this is the first time `key` is seen in the process.
fn first(key: Seen) -> bool {
    static SEEN: OnceLock<Mutex<HashSet<Seen>>> = OnceLock::new();
    SEEN.get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(key)
}

/// Log, once per backend, codec, and direction, that a hardware session
/// opened. Opening is not a per-frame path, but the engine opens a decoder
/// for every clip, so the line is written only the first time.
pub fn note_open(backend: &'static str, codec: CodecId, encode: bool) {
    if first((backend, codec, encode, "")) {
        let what = if encode { "encodes" } else { "decodes" };
        cutlass_logger::info!("{backend} {what} {codec:?}");
    }
}

/// Log, once per backend, codec, direction, and reason, why a backend that
/// is present declined, so a machine that falls back to the Rust codec says
/// why. A missing library or device is not noted: that is most machines.
pub fn note_decline(backend: &'static str, codec: CodecId, encode: bool, why: &'static str) {
    if first((backend, codec, encode, why)) {
        let what = if encode { "encode" } else { "decode" };
        cutlass_logger::info!("{backend} does not {what} {codec:?}: {why}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_or_unknown_allows_both_and_a_list_narrows() {
        let both = Backends {
            nvidia: true,
            vaapi: true,
        };
        assert_eq!(Backends::parse(None), both);
        assert_eq!(Backends::parse(Some("  ")), both);
        assert_eq!(Backends::parse(Some("quicksync")), both);
        assert_eq!(
            Backends::parse(Some("VAAPI")),
            Backends {
                nvidia: false,
                vaapi: true
            }
        );
        assert_eq!(Backends::parse(Some("nvidia, vaapi")), both);
        assert_eq!(
            Backends::parse(Some("none")),
            Backends {
                nvidia: false,
                vaapi: false
            }
        );
    }
}
