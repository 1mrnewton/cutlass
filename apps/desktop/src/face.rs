//! The faces the desktop trusts for text clips.
//!
//! The table itself is shared with the phones; this list says where a
//! desktop keeps a usable face.

pub use cutlass_playback::FaceTable;

const FACE_CANDIDATES: &[&str] = &[
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Geneva.ttf",
    "/System/Library/Fonts/NewYork.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
];

/// Every candidate that is a single readable face, in that order.
#[must_use]
pub fn installed() -> FaceTable {
    FaceTable::load_from(FACE_CANDIDATES)
}
