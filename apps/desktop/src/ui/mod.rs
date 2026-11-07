//! The desktop's own components.
//!
//! A component reads its colors and sizes from [`crate::tokens`], holds no
//! app state, and hands back what the user did. Panels build from these
//! instead of painting their own chrome.

mod button;
mod field;
mod scroll;
mod surface;

pub use button::{Button, ButtonKind};
pub use field::{Field, FieldResponse};
pub use scroll::Scroll;
pub use surface::Surface;
