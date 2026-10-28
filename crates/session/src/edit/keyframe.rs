use cutlass_model::{CommonKeyframeGroup, Draft, SegmentId};

use super::speed::locate;
use super::{EditError, ensure_unlocked};

/// Replace the segment's keyframe groups and leave every other field.
pub fn set_keyframes(
    draft: &mut Draft,
    segment_id: &SegmentId,
    keyframes: &[CommonKeyframeGroup],
) -> Result<(), EditError> {
    let (track_index, segment_index) = locate(draft, segment_id)?;
    ensure_unlocked(draft.tracks[track_index].attribute)?;
    draft.tracks[track_index].segments[segment_index].common_keyframes = keyframes.to_vec();
    Ok(())
}
