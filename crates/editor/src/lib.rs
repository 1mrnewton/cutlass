//! Timeline geometry. A gesture becomes one session request.
//!
//! A lane is the row drawn for a track. The draft has no lane record.
//! Zoom is a pixel size. Snap reports a time. This crate does not rewrite tracks.

#![forbid(unsafe_code)]

mod append;
mod controls;
mod filmstrip;
mod fling;
mod gesture;
mod hit;
mod lane;
mod reorder;
mod ruler;
mod snap;
mod stage;
mod tools;
mod zoom;

pub use append::{Append, AppendLane, append_placement, material_span};
pub use controls::{MAX_VOLUME, SPEED_MARKS, speed_at, speed_position, volume_at, volume_position};
pub use filmstrip::{FilmTile, MAX_TILES, film_tiles};
pub use fling::{
    FLING_DECAY, FLING_LIMIT, FLING_REST, FLING_START, FlingStep, fling_start, fling_step,
};
pub use gesture::{
    DropGesture, DropOutcome, GestureError, GestureOutcome, MoveGesture, Placement, TrimGesture,
    TrimPlan, drop_gesture, move_gesture, placement, trim_gesture,
};
pub use hit::{BadScale, Hit, Pointer, TimelineScale, hit_test};
pub use lane::{Lane, lane_category, lanes};
pub use reorder::{place_of, reorder_start};
pub use ruler::{RulerStep, Tick, clock, ruler_step, ruler_ticks, zoom_ceiling};
pub use snap::{SnapKind, SnapTarget, snap, snap_guides, transition_middle};
pub use stage::{
    MAX_SCALE, MIN_SCALE, NoText, Pose, SCALE_REACH, SNAP_REACH, StageAlign, StageBox,
    StageGesture, StageGuides, StageStep, TURN_REACH, TextSizes, canvas_size, stage_align,
    stage_box, stage_gesture, stage_pick,
};
pub use tools::{Bar, Tool, ToolId, clip_bar, tools};
pub use zoom::{
    Zoom, ZoomBounds, fit_pixel_size, last_accuracy_step, minimum_pixel_size, quarter_lane,
};
