//! Marks that are not from the Hugeicons free set.

use crate::{IconData, Mark, Shape};

/// The split tool's blade. Filled, not stroked.
pub const BLADE: IconData = IconData {
    name: "blade",
    source: None,
    marks: &[Mark {
        shape: Shape::Path("M4.5 19.5 L12 3 H15 L7.8 19.5 Z"),
        fill: true,
        stroke: None,
    }],
};
