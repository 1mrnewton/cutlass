//! The desktop's own components.
//!
//! A component reads its colors and sizes from [`crate::tokens`], holds no
//! app state, and hands back what the user did. Panels build from these
//! instead of painting their own chrome.

mod button;
mod choice;
mod field;
mod scroll;
mod surface;
mod toggle;

pub use button::{Button, ButtonKind};
pub use choice::{Choice, Segmented};
pub use field::{Field, FieldResponse};
pub use scroll::Scroll;
pub use surface::Surface;
pub use toggle::{Check, Switch, switch_row};
