//! The Animations panel: a clip's in and out animation and the presets.

use cutlass_model::{ANIMATION_PRESETS, AnimType, Draft, animation_preset, segment_animations};
use cutlass_session::{Micros, SegmentId};

/// One side of a clip's animations: the preset's resource id and length.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhoneAnimation {
    pub resource_id: String,
    pub duration: Micros,
}

/// A clip's in and out animations and its length, which they share.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhoneAnimations {
    pub enter: Option<PhoneAnimation>,
    pub leave: Option<PhoneAnimation>,
    pub length: Micros,
}

impl PhoneAnimations {
    /// The animations the panel can show for `segment_id`. One it does not
    /// offer, such as a package from the reference editor, reads as none.
    pub(crate) fn of(draft: &Draft, segment_id: &SegmentId) -> Option<Self> {
        let segment = draft
            .tracks
            .iter()
            .flat_map(|track| track.segments.iter())
            .find(|segment| &segment.id == segment_id)?;
        let side = |kind: AnimType| {
            segment_animations(draft, segment)?
                .animations
                .iter()
                .filter(|animation| animation_preset(animation).is_some_and(|p| p.kind == kind))
                .find_map(|animation| {
                    Some(PhoneAnimation {
                        resource_id: animation.resource_id.clone(),
                        duration: animation.duration?,
                    })
                })
        };
        Some(Self {
            enter: side(AnimType::In),
            leave: side(AnimType::Out),
            length: segment.target.duration,
        })
    }
}

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
