// --- Chroma key ---------------------------------------------------------------

use serde::{Deserialize, Serialize};

use crate::error::ModelError;
use crate::param::Param;

use super::{default_zero_param, is_zero_param, validate_unit_param};

/// Green-screen keying (CapCut chroma key): pixels near `rgb` turn
/// transparent. `None` on the clip ⇔ keying off.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChromaKey {
    /// Key color, opaque `[r, g, b]`.
    pub rgb: [u8; 3],
    /// Keying strength (tolerance), `0` … `1`.
    #[serde(default = "default_zero_param", skip_serializing_if = "is_zero_param")]
    pub strength: Param<f32>,
    /// Shadow retention, `0` … `1`.
    #[serde(default = "default_zero_param", skip_serializing_if = "is_zero_param")]
    pub shadow: Param<f32>,
}

impl ChromaKey {
    pub fn validate(&self) -> Result<(), ModelError> {
        validate_unit_param("chroma strength", &self.strength)?;
        validate_unit_param("chroma shadow", &self.shadow)
    }
}

// --- Stabilization ------------------------------------------------------------

/// Stabilization strength (CapCut stabilize panel). `None` on the clip ⇔ off.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StabilizeLevel {
    Recommended,
    Smooth,
    MaxSmooth,
}

impl StabilizeLevel {
    /// Stable wire/catalog id (the serde name).
    pub const fn id(self) -> &'static str {
        match self {
            StabilizeLevel::Recommended => "recommended",
            StabilizeLevel::Smooth => "smooth",
            StabilizeLevel::MaxSmooth => "max_smooth",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            StabilizeLevel::Recommended => "Recommended",
            StabilizeLevel::Smooth => "Smooth",
            StabilizeLevel::MaxSmooth => "Max smooth",
        }
    }

    /// Every level (UI browsing order).
    pub const ALL: [StabilizeLevel; 3] = [
        StabilizeLevel::Recommended,
        StabilizeLevel::Smooth,
        StabilizeLevel::MaxSmooth,
    ];
}
