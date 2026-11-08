//! Marks that are not from the Hugeicons free set.

use crate::{Cap, IconData, Join, Mark, Shape, Stroke};

/// Every custom mark. `icons.txt` names one with `custom`.
pub static CUSTOM: &[IconData] = &[BLADE, ZOOMFIT];

/// The split tool's blade. Filled, not stroked.
pub const BLADE: IconData = IconData {
    name: "blade",
    source: None,
    marks: &[Mark {
        shape: Shape::Path("M4.5 19.5 L12 3 H15 L7.8 19.5 Z"),
        fill: true,
        even_odd: false,
        stroke: None,
    }],
};

const LINE: Option<Stroke> = Some(Stroke {
    width: 1.5,
    cap: Cap::Round,
    join: Join::Round,
});

/// Zoom to fit: the zoom in and out frame and lens, with four corner
/// brackets in the lens. Drawn with imagent to sit beside them.
pub const ZOOMFIT: IconData = IconData {
    name: "zoomFit",
    source: None,
    marks: &[
        Mark {
            shape: Shape::Path(
                "M18.5016 18.5L21 21M20 14.5C20 11.4624 17.5376 9 14.5 9C11.4624 9 9 11.4624 9 14.5C9 17.5376 11.4624 20 14.5 20C17.5376 20 20 17.5376 20 14.5Z",
            ),
            fill: false,
            even_odd: false,
            stroke: LINE,
        },
        Mark {
            shape: Shape::Path("M12 13.5V12H13.5M15.5 12H17V13.5M17 15.5V17H15.5M13.5 17H12V15.5"),
            fill: false,
            even_odd: false,
            stroke: LINE,
        },
        Mark {
            shape: Shape::Path(
                "M10 3H14M3 10V14M6.5 21C4.567 21 3 19.433 3 17.5M17.5 3C19.433 3 21 4.567 21 6.5M3 6.5C3 4.567 4.567 3 6.5 3",
            ),
            fill: false,
            even_odd: false,
            stroke: LINE,
        },
    ],
};
