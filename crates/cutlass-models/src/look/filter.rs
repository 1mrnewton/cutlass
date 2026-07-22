// --- Filter presets -------------------------------------------------------------

use serde::{Deserialize, Serialize};

use crate::error::ModelError;
use crate::param::Param;

use super::validate_unit_param;

/// A color-grade filter applied to a clip (CapCut filters). `None` on the
/// clip ⇔ no filter. Also the payload persisted on `Generator::Filter` lane
/// bars, which grade everything beneath them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Filter {
    /// Catalog id (see [`filter_catalog`]).
    pub id: String,
    /// Blend of the graded result over the original, `0` … `1`.
    #[serde(
        default = "default_filter_intensity_param",
        skip_serializing_if = "is_default_filter_intensity"
    )]
    pub intensity: Param<f32>,
}

impl Filter {
    /// A filter at the default intensity.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            intensity: Param::Constant(default_filter_intensity()),
        }
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        if filter_spec(&self.id).is_none() {
            return Err(ModelError::InvalidParam(format!(
                "unknown filter '{}'",
                self.id
            )));
        }
        validate_unit_param("filter intensity", &self.intensity)
    }
}

fn default_filter_intensity() -> f32 {
    0.8
}

fn default_filter_intensity_param() -> Param<f32> {
    Param::Constant(default_filter_intensity())
}

fn is_default_filter_intensity(v: &Param<f32>) -> bool {
    v.constant() == Some(default_filter_intensity())
}

/// One filter catalog entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilterSpec {
    pub id: &'static str,
    pub label: &'static str,
}

const FILTERS: &[FilterSpec] = &[
    FilterSpec {
        id: "vivid",
        label: "Vivid",
    },
    FilterSpec {
        id: "warm",
        label: "Warm",
    },
    FilterSpec {
        id: "cool",
        label: "Cool",
    },
    FilterSpec {
        id: "mono",
        label: "Mono",
    },
    FilterSpec {
        id: "fade",
        label: "Fade",
    },
    FilterSpec {
        id: "chrome",
        label: "Chrome",
    },
    FilterSpec {
        id: "noir",
        label: "Noir",
    },
    FilterSpec {
        id: "sunset",
        label: "Sunset",
    },
    FilterSpec {
        id: "forest",
        label: "Forest",
    },
    FilterSpec {
        id: "berry",
        label: "Berry",
    },
];

/// Every filter preset (UI browsing order).
pub fn filter_catalog() -> &'static [FilterSpec] {
    FILTERS
}

/// The catalog entry for `id`, or `None`.
pub fn filter_spec(id: &str) -> Option<&'static FilterSpec> {
    FILTERS.iter().find(|s| s.id == id)
}

// --- 3D LUTs ----------------------------------------------------------------

/// A `.cube` 3D LUT applied to a clip after its filter/adjust grade. `None`
/// on the clip ⇔ no LUT. File-backed like Lottie animations: `path` points at
/// a `.cube` file on disk (downloaded from the asset catalog or supplied by
/// the user); the renderer parses and uploads it lazily and skips missing
/// files gracefully. Also valid on `Generator::Filter` lane bars, which apply
/// the LUT to everything composited beneath them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Lut {
    /// Absolute path to the `.cube` file.
    pub path: String,
    /// Blend of the looked-up result over the original, `0` … `1`.
    #[serde(
        default = "default_filter_intensity_param",
        skip_serializing_if = "is_default_filter_intensity"
    )]
    pub intensity: Param<f32>,
}

impl Lut {
    /// A LUT at the default intensity.
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            intensity: Param::Constant(default_filter_intensity()),
        }
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        if self.path.trim().is_empty() {
            return Err(ModelError::InvalidParam("empty LUT path".into()));
        }
        validate_unit_param("LUT intensity", &self.intensity)
    }
}
