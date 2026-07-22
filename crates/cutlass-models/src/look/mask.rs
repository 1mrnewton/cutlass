// --- Mask -------------------------------------------------------------------

use serde::{Deserialize, Serialize};

use crate::error::ModelError;
use crate::param::Param;

use super::{default_zero_param, is_false, is_zero_param, validate_unit_param};

/// Mask shapes (CapCut mask panel). Serialized by snake_case id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaskKind {
    Linear,
    Mirror,
    Circle,
    Rectangle,
    Heart,
    Star,
}

impl MaskKind {
    /// Stable wire/catalog id (the serde name).
    pub const fn id(self) -> &'static str {
        match self {
            MaskKind::Linear => "linear",
            MaskKind::Mirror => "mirror",
            MaskKind::Circle => "circle",
            MaskKind::Rectangle => "rectangle",
            MaskKind::Heart => "heart",
            MaskKind::Star => "star",
        }
    }
}

/// A shaped alpha mask over a clip's content. `None` on the clip ⇔ no mask.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mask {
    pub kind: MaskKind,
    /// Edge softness, `0` (hard) … `1` (fully feathered).
    #[serde(default = "default_zero_param", skip_serializing_if = "is_zero_param")]
    pub feather: Param<f32>,
    /// Keep the outside instead of the inside.
    #[serde(default, skip_serializing_if = "is_false")]
    pub invert: bool,
}

impl Mask {
    /// A hard, non-inverted mask of `kind`.
    pub fn new(kind: MaskKind) -> Self {
        Self {
            kind,
            feather: Param::Constant(0.0),
            invert: false,
        }
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        validate_unit_param("mask feather", &self.feather)
    }
}

/// One mask catalog entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskSpec {
    pub kind: MaskKind,
    pub label: &'static str,
}

const MASKS: &[MaskSpec] = &[
    MaskSpec {
        kind: MaskKind::Linear,
        label: "Linear",
    },
    MaskSpec {
        kind: MaskKind::Mirror,
        label: "Mirror",
    },
    MaskSpec {
        kind: MaskKind::Circle,
        label: "Circle",
    },
    MaskSpec {
        kind: MaskKind::Rectangle,
        label: "Rectangle",
    },
    MaskSpec {
        kind: MaskKind::Heart,
        label: "Heart",
    },
    MaskSpec {
        kind: MaskKind::Star,
        label: "Star",
    },
];

/// Every mask shape (UI browsing order).
pub fn mask_catalog() -> &'static [MaskSpec] {
    MASKS
}
