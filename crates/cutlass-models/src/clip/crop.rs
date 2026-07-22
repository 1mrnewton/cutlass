use serde::{Deserialize, Serialize};

use crate::error::ModelError;
use crate::param::{Lerp, SegmentSample};

/// Normalized crop window into a clip's content (CapCut crop, M1).
///
/// Fractions of the uncropped frame: `(x, y)` is the kept region's top-left
/// corner, `(w, h)` its extent — `{0, 0, 1, 1}` keeps everything. Crop
/// happens in content space *before* placement: the kept region aspect-fits
/// the canvas and transforms exactly like the full frame did, so cropping
/// never moves the layer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CropRect {
    /// Left edge of the kept region, `0.0..1.0` of content width.
    pub x: f32,
    /// Top edge of the kept region, `0.0..1.0` of content height.
    pub y: f32,
    /// Kept width fraction, `(0.0..=1.0]`.
    pub w: f32,
    /// Kept height fraction, `(0.0..=1.0]`.
    pub h: f32,
}

/// Component-wise lerp. Intermediate values are not re-clamped: keyframe
/// values are validated at set time, and the valid-rect set
/// (`w`/`h` ≥ [`MIN_CROP_FRACTION`], `x`/`y` ≥ 0, `x+w`/`y+h` ≤ 1) is
/// convex, so a lerp of two valid rects stays valid.
impl Lerp for CropRect {
    fn lerp(a: Self, b: Self, t: f32) -> Self {
        Self {
            x: Lerp::lerp(a.x, b.x, t),
            y: Lerp::lerp(a.y, b.y, t),
            w: Lerp::lerp(a.w, b.w, t),
            h: Lerp::lerp(a.h, b.h, t),
        }
    }
}

impl SegmentSample for CropRect {}

/// Smallest croppable extent per axis (1% of the content) — keeps the kept
/// region non-degenerate so placement math and UV rects never collapse.
pub const MIN_CROP_FRACTION: f32 = 0.01;

impl CropRect {
    /// Keep the whole frame (the default; absent from saves).
    pub const FULL: Self = Self {
        x: 0.0,
        y: 0.0,
        w: 1.0,
        h: 1.0,
    };

    /// True iff the crop keeps the whole frame.
    pub fn is_full(&self) -> bool {
        *self == Self::FULL
    }

    /// `Ok` iff the kept region is non-degenerate and inside the frame:
    /// finite fields, `w`/`h` at least [`MIN_CROP_FRACTION`], edges within
    /// `0..=1`.
    pub fn validate(&self) -> Result<(), ModelError> {
        let finite = [self.x, self.y, self.w, self.h]
            .iter()
            .all(|v| v.is_finite());
        if !finite {
            return Err(ModelError::InvalidParam(
                "crop: non-finite component".into(),
            ));
        }
        if self.w < MIN_CROP_FRACTION || self.h < MIN_CROP_FRACTION {
            return Err(ModelError::InvalidParam(format!(
                "crop: kept region must be at least {MIN_CROP_FRACTION} per axis"
            )));
        }
        if self.x < 0.0 || self.y < 0.0 || self.x + self.w > 1.0 || self.y + self.h > 1.0 {
            return Err(ModelError::InvalidParam(
                "crop: kept region must lie inside the frame".into(),
            ));
        }
        Ok(())
    }
}

impl Default for CropRect {
    fn default() -> Self {
        Self::FULL
    }
}
