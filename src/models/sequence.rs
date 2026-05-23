use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::rational::Rational;

use super::track::Track;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Sequence {
    pub id: String,
    pub name: String,
    pub fps: Rational,
    pub drop_frame: bool,
    /// Stable iteration order of lanes, top → bottom in the timeline.
    /// Invariant: every `Track` whose `kind == Video` precedes every
    /// `Track` whose `kind == Audio`. New lanes inserted by commands
    /// must preserve this — see [`crate::command`].
    pub track_order: Vec<String>,
    pub tracks: HashMap<String, Track>,
    /// Monotonic counter used to mint unique ids for lanes that are
    /// created at edit time (e.g. when a drop collision forces a new
    /// lane to spawn). Persisted with the sequence so resume across
    /// sessions doesn't reuse ids.
    #[serde(default)]
    pub next_track_id: u32,
    pub width: f32,
    pub height: f32,
}
