//! Timeline geometry. A gesture becomes one session request.
//!
//! A lane is the row drawn for a track. The draft has no lane record.
//! Zoom is a pixel size. Snap reports a time. This crate does not rewrite tracks.

#![forbid(unsafe_code)]

mod lane;

pub use lane::{Lane, lane_category, lanes};
