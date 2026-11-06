//! The open cut and its history.
//!
//! One apply is a list of requests and one undo step. A failed request
//! leaves the draft as it was. Magnet is not applied here.

#![forbid(unsafe_code)]

mod edit;
mod history;
mod request;
mod session;

pub use cutlass_model::{
    Draft, LinkageSettings, Micros, SegmentId, TrackAttribute, TrackId, linkage_from_biz,
};
pub use edit::{
    CANVAS_SHAPES, CanvasShape, EditError, MAX_CAPTION_STAGGER_MS, NEW_TEXT_SIZE, canvas_for,
    follow_first_picture, lane_fits, ratio_request,
};
pub use history::UNDO_LIMIT;
pub use request::{Arrange, ImportedMedium, Request, TrimEdge};
pub use session::Session;
