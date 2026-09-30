//! Live preview overrides for individual [`ClipParam`] values.
//!
//! Unlike the whole-struct transform/look/styles/generator lanes, this map can
//! hold many (clip, param) entries at once — e.g. crop x+y during a rect drag.
//! Empty maps are free on the resolve hot path (early-out before any clone).

use cutlass_models::{Clip, ClipId, ClipParam, Map, ParamValue};

/// Session-only param substitutions keyed by `(clip, param)`.
///
/// Preview frames sample these in place of the stored/keyframed value; the
/// project, history, and export never see them.
#[derive(Debug, Default, Clone)]
pub struct ParamOverrides {
    map: Map<(ClipId, ClipParam), ParamValue>,
}

impl ParamOverrides {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Insert or replace the live value for `(clip, param)`.
    pub fn set(&mut self, clip: ClipId, param: ClipParam, value: ParamValue) {
        self.map.insert((clip, param), value);
    }

    /// Drop every override for `clip` (release / abandon one inspector drag).
    pub fn clear_clip(&mut self, clip: ClipId) {
        self.map.retain(|&(id, _), _| id != clip);
    }

    /// Drop every override (session reset / project reload).
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Remove one `(clip, param)` entry after that param is committed.
    pub fn clear_param(&mut self, clip: ClipId, param: ClipParam) {
        self.map.remove(&(clip, param));
    }

    pub fn get(&self, clip: ClipId, param: ClipParam) -> Option<ParamValue> {
        self.map.get(&(clip, param)).copied()
    }

    /// Clone `base` and apply every override for its id. Returns `None` when
    /// this map holds nothing for `base.id` (no allocation).
    pub fn overlay_clip(&self, base: &Clip) -> Option<Clip> {
        if self.map.is_empty() {
            return None;
        }
        let mut out = None;
        for (&(id, param), &value) in &self.map {
            if id != base.id {
                continue;
            }
            let clip = out.get_or_insert_with(|| base.clone());
            // Invalid values (missing chroma, wrong type) are dropped — the
            // committed sample stays in effect for that param.
            let _ = clip.set_param_constant(param, value);
        }
        out
    }
}
