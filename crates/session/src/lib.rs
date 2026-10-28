//! The open cut and its history.
//!
//! One apply is a list of requests and one undo step. A failed request
//! leaves the draft as it was. Magnet, linkage, transitions, and speed
//! curves are not applied here.

#![forbid(unsafe_code)]

mod edit;
mod history;
mod request;
mod session;

pub use edit::EditError;
pub use request::{Request, TrimEdge};
pub use session::Session;
