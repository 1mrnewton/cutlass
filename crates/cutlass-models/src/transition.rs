//! Transitions (v1 roadmap M4): a timed blend at the junction between two
//! abutting clips on a track.
//!
//! A [`Transition`] is **data**, like effects: the model stores
//! `{left, right, transition_id, duration}` and the compositor owns the WGSL
//! that blends the two frames by progress. The [`transition_catalog`] is the
//! validation + UI source of truth (ids, display names); it is drift-checked
//! against the compositor's renderable set from `cutlass-engine`.
//!
//! A transition lives only while its pair of clips still abuts (left's end
//! tick equals right's start tick). Structural edits that break the abutment
//! prune the dead junction; undo restores it.

use serde::{Deserialize, Serialize};

use crate::ids::ClipId;

/// A renderable transition: its stable id and a human label.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransitionSpec {
    pub id: &'static str,
    pub label: &'static str,
}

/// The starter set (M4). Ids must stay in lockstep with
/// `cutlass_compositor::transition_descriptors`.
const CATALOG: &[TransitionSpec] = &[
    TransitionSpec { id: "crossfade", label: "Cross Fade" },
    TransitionSpec { id: "dip_to_black", label: "Dip to Black" },
    TransitionSpec { id: "dip_to_white", label: "Dip to White" },
    TransitionSpec { id: "wipe_left", label: "Wipe Left" },
    TransitionSpec { id: "wipe_right", label: "Wipe Right" },
    TransitionSpec { id: "wipe_up", label: "Wipe Up" },
    TransitionSpec { id: "wipe_down", label: "Wipe Down" },
    TransitionSpec { id: "slide", label: "Slide" },
];

/// Default transition window length, in timeline ticks (centered on the cut).
pub const DEFAULT_TRANSITION_TICKS: i64 = 24;

/// Every transition the model knows about (validation + UI browsing).
pub fn transition_catalog() -> &'static [TransitionSpec] {
    CATALOG
}

/// The catalog entry for `id`, or `None`.
pub fn transition_spec(id: &str) -> Option<&'static TransitionSpec> {
    CATALOG.iter().find(|s| s.id == id)
}

/// A transition at the junction between `left` and the clip immediately to its
/// right on the same track. Centered on the cut; `duration` is the total
/// window length in timeline ticks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Transition {
    /// The clip on the outgoing (left) side of the cut.
    pub left: ClipId,
    /// The clip on the incoming (right) side of the cut.
    pub right: ClipId,
    /// Catalog id of the blend to render.
    pub transition_id: String,
    /// Total window length in timeline ticks, centered on the junction.
    pub duration: i64,
}

impl Transition {
    pub fn new(left: ClipId, right: ClipId, transition_id: impl Into<String>, duration: i64) -> Self {
        Self {
            left,
            right,
            transition_id: transition_id.into(),
            duration: duration.max(1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_ids_are_unique() {
        let mut ids: Vec<&str> = CATALOG.iter().map(|s| s.id).collect();
        let count = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), count, "transition ids are unique");
        assert_eq!(count, 8, "the M4 starter set is eight transitions");
    }

    #[test]
    fn spec_lookup_resolves_known_and_rejects_unknown() {
        assert_eq!(transition_spec("crossfade").unwrap().label, "Cross Fade");
        assert!(transition_spec("nope").is_none());
    }

    #[test]
    fn new_clamps_duration_to_at_least_one() {
        let t = Transition::new(ClipId::from_raw(1), ClipId::from_raw(2), "crossfade", 0);
        assert_eq!(t.duration, 1);
    }
}
