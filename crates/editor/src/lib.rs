//! Timeline geometry. A gesture becomes one session request.
//!
//! A lane is the row drawn for a track. The draft has no lane record.
//! Zoom is a pixel size. Snap reports a time. This crate does not rewrite tracks.

#![forbid(unsafe_code)]

mod hit;
mod lane;
mod zoom;

pub use hit::{BadScale, Hit, Pointer, TimelineScale, hit_test};
pub use lane::{Lane, lane_category, lanes};
pub use zoom::{
    Zoom, ZoomBounds, fit_pixel_size, last_accuracy_step, minimum_pixel_size, quarter_lane,
};
