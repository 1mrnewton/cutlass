//! Move, scale, and turn a picture clip on the preview.
//!
//! The geometry is `cutlass-editor`'s. This keeps the one gesture in flight
//! and folds its changes into one undo step.

use cutlass_editor::{Pose, StageBox, StageGesture, StageGuides};

/// A picture clip's box on the canvas, in canvas pixels with y down.
#[derive(Clone, Debug, PartialEq)]
pub struct PhoneStage {
    pub canvas_width: f64,
    pub canvas_height: f64,
    pub center_x: f64,
    pub center_y: f64,
    pub width: f64,
    pub height: f64,
    /// Degrees clockwise on screen.
    pub turn: f64,
    /// The clip's centre sits on the canvas's upright centre line.
    pub vertical_guide: bool,
    /// The clip's centre sits on the canvas's level centre line.
    pub horizontal_guide: bool,
}

impl PhoneStage {
    pub(crate) fn new(canvas: (f64, f64), stage: &StageBox, guides: StageGuides) -> Self {
        Self {
            canvas_width: canvas.0,
            canvas_height: canvas.1,
            center_x: stage.center_x,
            center_y: stage.center_y,
            width: stage.width,
            height: stage.height,
            turn: stage.turn,
            vertical_guide: guides.vertical,
            horizontal_guide: guides.horizontal,
        }
    }
}

/// The gesture in flight, and the revision and placement of its last change.
pub(crate) struct Live {
    pub(crate) gesture: StageGesture,
    pub(crate) applied: Option<(u64, Pose)>,
}
