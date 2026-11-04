//! The Animations panel: a clip's in and out animation and the presets.

use cutlass_model::{ANIMATION_PRESETS, AnimType};
use cutlass_session::Micros;

/// One side of a clip's animations: the preset's resource id and length.
pub use cutlass_editor::ClipAnimation as PhoneAnimation;
/// A clip's in and out animations and its length, which they share.
pub use cutlass_editor::ClipAnimations as PhoneAnimations;

/// One preset in the panel's grid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhoneAnimationPreset {
    pub name: String,
    /// `true` for the In tab, `false` for Out.
    pub enter: bool,
    pub resource_id: String,
    /// The length a new one starts with.
    pub duration: Micros,
}

/// The presets, in the panel's order after None.
#[must_use]
pub fn animation_presets() -> Vec<PhoneAnimationPreset> {
    ANIMATION_PRESETS
        .iter()
        .map(|preset| PhoneAnimationPreset {
            name: preset.name.to_owned(),
            enter: preset.kind == AnimType::In,
            resource_id: preset.resource_id.to_owned(),
            duration: preset.duration,
        })
        .collect()
}
