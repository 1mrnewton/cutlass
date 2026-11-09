//! Marks that are not from the Hugeicons free set.

use crate::{Cap, IconData, Join, Mark, Shape, Stroke};

/// Every custom mark. `icons.txt` names one with `custom`.
pub static CUSTOM: &[IconData] = &[BLADE, ZOOMFIT, KEYBACKSPACE, SPLITAT, TRIMLEFT, TRIMRIGHT];

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

/// The backspace key: a key pointing left with a cross in it, stroked like
/// the Hugeicons key marks beside it. The free set has none.
pub const KEYBACKSPACE: IconData = IconData {
    name: "keyBackspace",
    source: None,
    marks: &[
        Mark {
            shape: Shape::Path(
                "M8.6 4H18C19.6569 4 21 5.34315 21 7V17C21 18.6569 19.6569 20 18 20H8.6C8.0034 20 7.4374 19.7336 7.0574 19.274L3.3074 13.274C2.7309 12.5232 2.7309 11.4768 3.3074 10.726L7.0574 4.726C7.4374 4.2664 8.0034 4 8.6 4Z",
            ),
            fill: false,
            even_odd: false,
            stroke: LINE,
        },
        Mark {
            shape: Shape::Path("M11.75 9.25L17.25 14.75M17.25 9.25L11.75 14.75"),
            fill: false,
            even_odd: false,
            stroke: LINE,
        },
    ],
};

/// A closing bracket on the left half: the clip before the playhead.
const BEFORE: &str = "M4 3H9V21H4";
/// An opening bracket on the right half: the clip after the playhead.
const AFTER: &str = "M20 3H15V21H20";

const fn stroked(path: &'static str) -> Mark {
    Mark {
        shape: Shape::Path(path),
        fill: false,
        even_odd: false,
        stroke: LINE,
    }
}

/// Split at the playhead: the clip's two halves, parted.
pub const SPLITAT: IconData = IconData {
    name: "splitAt",
    source: None,
    marks: &[stroked(BEFORE), stroked(AFTER)],
};

/// Delete left of the playhead: the half before it dashed, as gone.
pub const TRIMLEFT: IconData = IconData {
    name: "trimLeft",
    source: None,
    marks: &[
        stroked("M4 3H5.5M7.5 3H9V4.5M9 7V9M9 11V13M9 15V17M9 19.5V21H7.5M5.5 21H4"),
        stroked(AFTER),
    ],
};

/// Delete right of the playhead: the half after it dashed, as gone.
pub const TRIMRIGHT: IconData = IconData {
    name: "trimRight",
    source: None,
    marks: &[
        stroked(BEFORE),
        stroked("M20 3H18.5M16.5 3H15V4.5M15 7V9M15 11V13M15 15V17M15 19.5V21H16.5M18.5 21H20"),
    ],
};
