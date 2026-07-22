use serde::{Deserialize, Serialize};

use crate::error::ModelError;

/// Per-clip motion blur (temporal supersampling of the clip's own animated
/// transform). Off (and absent from saves) by default.
///
/// Params are plain values — not animatable. Motion blur settings are rarely
/// keyframed in practice; exporting a changing shutter mid-clip is out of
/// scope.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MotionBlur {
    /// When false, the renderer skips supersampling entirely.
    #[serde(default)]
    pub enabled: bool,
    /// Shutter angle in degrees: 360 = full frame interval. 0 disables.
    #[serde(default = "default_shutter_deg")]
    pub shutter_deg: f32,
    /// Sub-frame samples (quality). Clamped `2..=16` at render time;
    /// validated `2..=32` on write.
    #[serde(default = "default_samples")]
    pub samples: u32,
}

fn default_shutter_deg() -> f32 {
    180.0
}

fn default_samples() -> u32 {
    8
}

impl Default for MotionBlur {
    fn default() -> Self {
        Self {
            enabled: false,
            shutter_deg: default_shutter_deg(),
            samples: default_samples(),
        }
    }
}

impl MotionBlur {
    /// True iff this is the off/default state — the serde skip predicate on
    /// [`crate::Clip::motion_blur`].
    pub fn is_default(&self) -> bool {
        !self.enabled
            && self.shutter_deg == default_shutter_deg()
            && self.samples == default_samples()
    }

    /// Validate shutter (`0..=720` finite) and samples (`2..=32`).
    pub fn validate(&self) -> Result<(), ModelError> {
        if !self.shutter_deg.is_finite() || !(0.0..=720.0).contains(&self.shutter_deg) {
            return Err(ModelError::InvalidParam(format!(
                "motion blur shutter_deg = {} out of range [0, 720]",
                self.shutter_deg
            )));
        }
        if !(2..=32).contains(&self.samples) {
            return Err(ModelError::InvalidParam(format!(
                "motion blur samples = {} out of range [2, 32]",
                self.samples
            )));
        }
        Ok(())
    }
}
