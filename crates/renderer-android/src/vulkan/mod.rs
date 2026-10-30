//! Android Vulkan images.
//!
//! The session owns the instance, the device, and at most eight images.
//! Tokens are image handles. A zero handle is never issued.

mod backend;
mod sys;

pub(crate) use backend::Session;
pub(crate) use backend::{open, sample_platform};
