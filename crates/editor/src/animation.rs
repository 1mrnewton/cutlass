//! A clip's in and out animations, as the Animation panels show them.

use cutlass_model::{AnimType, Draft, animation_preset, segment_animations};
use cutlass_session::{Micros, SegmentId};

use crate::stage::{TextSizes, stage_box};

/// One side of a clip's animations: the preset's resource id and length.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClipAnimation {
    pub resource_id: String,
    pub duration: Micros,
}

/// A clip's in and out animations and its length, which they share.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClipAnimations {
    pub enter: Option<ClipAnimation>,
    pub leave: Option<ClipAnimation>,
    pub length: Micros,
}

/// The animations the panel can show for `segment_id`, or `None` when the
/// clip is not a picture or title on the canvas. One the panel does not
/// offer, such as a package from the reference editor, reads as none.
#[must_use]
pub fn clip_animations(
    draft: &Draft,
    segment_id: &SegmentId,
    sizes: &dyn TextSizes,
) -> Option<ClipAnimations> {
    let segment = draft
        .tracks
        .iter()
        .flat_map(|track| track.segments.iter())
        .find(|segment| &segment.id == segment_id)?;
    stage_box(draft, segment_id, segment.target.start, sizes)?;
    let side = |kind: AnimType| {
        segment_animations(draft, segment)?
            .animations
            .iter()
            .filter(|animation| animation_preset(animation).is_some_and(|p| p.kind == kind))
            .find_map(|animation| {
                Some(ClipAnimation {
                    resource_id: animation.resource_id.clone(),
                    duration: animation.duration?,
                })
            })
    };
    Some(ClipAnimations {
        enter: side(AnimType::In),
        leave: side(AnimType::Out),
        length: segment.target.duration,
    })
}
