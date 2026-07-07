//! GPU pass descriptors for clip effects and track transitions.
//!
//! The model catalog (`cutlass-models`) and these descriptors are drift-checked
//! from `cutlass-engine` tests — ids and parameter slot order must agree.
//!
//! ## Effect coverage
//!
//! | Effect id | Renders |
//! |-----------|---------|
//! | `gaussian_blur`, `vignette`, `pixelate` | Yes |
//! | `sharpen`, `glitch`, `chromatic_aberration`, `grain`, `glow`, `zoom_blur`, `mirror` | Passthrough (no visual change) |
//!
//! ## Transition coverage
//!
//! | Transition id | Renders |
//! |---------------|---------|
//! | `crossfade`, `wipe_left` | Yes |
//! | `dip_to_black`, `dip_to_white`, `wipe_right`, `wipe_up`, `wipe_down`, `slide` | Crossfade fallback |

/// Static descriptor for one catalog effect or transition pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PassDescriptor {
    pub id: &'static str,
    /// Uniform slot order (names only; values come from [`PassInstance`]).
    pub params: &'static [&'static str],
}

/// One effect instance at render time: catalog id + sampled parameter values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PassInstance<'a> {
    pub id: &'static str,
    pub params: &'a [f32],
}

/// Whether a catalog id has a real WGSL implementation or a safe stand-in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassCoverage {
    /// Full GPU implementation.
    Implemented,
    /// Identity blit (effects) or crossfade fallback (transitions).
    Passthrough,
}

const EFFECT_DESCRIPTORS: &[PassDescriptor] = &[
    PassDescriptor {
        id: "gaussian_blur",
        params: &["radius"],
    },
    PassDescriptor {
        id: "vignette",
        params: &["amount"],
    },
    PassDescriptor {
        id: "sharpen",
        params: &["amount"],
    },
    PassDescriptor {
        id: "pixelate",
        params: &["size"],
    },
    PassDescriptor {
        id: "glitch",
        params: &["amount", "seed"],
    },
    PassDescriptor {
        id: "chromatic_aberration",
        params: &["amount"],
    },
    PassDescriptor {
        id: "grain",
        params: &["amount", "seed"],
    },
    PassDescriptor {
        id: "glow",
        params: &["threshold", "intensity"],
    },
    PassDescriptor {
        id: "zoom_blur",
        params: &["amount"],
    },
    PassDescriptor {
        id: "mirror",
        params: &["mode"],
    },
];

const TRANSITION_IDS: &[&str] = &[
    "crossfade",
    "dip_to_black",
    "dip_to_white",
    "wipe_left",
    "wipe_right",
    "wipe_up",
    "wipe_down",
    "slide",
];

/// Every effect the compositor can dispatch (matches `cutlass_models::effect_catalog`).
pub fn effect_descriptors() -> &'static [PassDescriptor] {
    EFFECT_DESCRIPTORS
}

/// Every transition id the compositor can dispatch (matches `cutlass_models::transition_catalog`).
pub fn transition_ids() -> &'static [&'static str] {
    TRANSITION_IDS
}

/// Implementation status for an effect catalog id.
pub fn effect_coverage(id: &str) -> PassCoverage {
    match id {
        "gaussian_blur" | "vignette" | "pixelate" => PassCoverage::Implemented,
        "sharpen"
        | "glitch"
        | "chromatic_aberration"
        | "grain"
        | "glow"
        | "zoom_blur"
        | "mirror" => PassCoverage::Passthrough,
        _ => PassCoverage::Passthrough,
    }
}

/// Implementation status for a transition catalog id.
pub fn transition_coverage(id: &str) -> PassCoverage {
    match id {
        "crossfade" | "wipe_left" => PassCoverage::Implemented,
        _ => PassCoverage::Passthrough,
    }
}

/// Resolve a transition id to the WGSL pass id actually dispatched.
/// Unimplemented transitions fall back to `crossfade`.
pub fn resolve_transition_pass(id: &str) -> &'static str {
    match transition_coverage(id) {
        PassCoverage::Implemented => {
            if id == "wipe_left" {
                "wipe_left"
            } else {
                "crossfade"
            }
        }
        PassCoverage::Passthrough => "crossfade",
    }
}

/// Return `true` when an effect pass can be skipped (identity).
pub fn effect_is_noop(id: &str, params: &[f32]) -> bool {
    match id {
        "gaussian_blur" => params.first().is_none_or(|&r| r <= 0.0),
        "vignette" => params.first().is_none_or(|&a| a <= 0.0),
        "pixelate" => params.first().is_none_or(|&s| s <= 1.0),
        _ => effect_coverage(id) == PassCoverage::Passthrough,
    }
}
