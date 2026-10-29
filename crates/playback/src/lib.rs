//! Preview.
//!
//! `present` asks `eval` for the frame at the playhead. Export uses that same
//! call. Dropping a late picture is not done here.

#![forbid(unsafe_code)]

mod present;

pub use present::present;
