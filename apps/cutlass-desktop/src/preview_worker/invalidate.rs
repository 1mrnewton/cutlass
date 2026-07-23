//! Preview-repaint classification for worker mutations.

use super::*;

/// Whether an executed mutation changes the visible composite at the current
/// playhead and should therefore trigger a preview re-render. The only frame
/// trigger used to be playhead movement, so edits (delete, generator/font
/// change, …) looked stale until the user scrubbed. `SetTransform` and
/// `ClearTransformOverride` render themselves with their own tick, so they're
/// excluded here to avoid a redundant second composite; pure session ops
/// (import, copy, auto-save, export, linkage, rename) don't alter the canvas.
pub(super) fn message_invalidates_preview(msg: &WorkerMsg) -> bool {
    matches!(
        msg,
        WorkerMsg::AddClip { .. }
            | WorkerMsg::AddGenerated { .. }
            | WorkerMsg::MoveClip { .. }
            | WorkerMsg::MoveGroup { .. }
            | WorkerMsg::TrimClip { .. }
            | WorkerMsg::RemoveClips { .. }
            | WorkerMsg::SetGenerator { .. }
            | WorkerMsg::SetGeneratorFill { .. }
            | WorkerMsg::SetClipSpeed { .. }
            | WorkerMsg::SetClipPitch { .. }
            | WorkerMsg::SetSpeedCurve { .. }
            | WorkerMsg::SetSpeedCurvePoint { .. }
            | WorkerMsg::SetClipCrop { .. }
            | WorkerMsg::SetBlendMode { .. }
            | WorkerMsg::SetMotionBlur { .. }
            | WorkerMsg::SetLayerStyles { .. }
            | WorkerMsg::ToggleLayerStyle { .. }
            | WorkerMsg::SetMask { .. }
            | WorkerMsg::SetMaskKind { .. }
            | WorkerMsg::SetMaskInvert { .. }
            | WorkerMsg::SetChroma { .. }
            | WorkerMsg::SetChromaColor { .. }
            | WorkerMsg::SetClipFilter { .. }
            | WorkerMsg::SetClipAdjust { .. }
            | WorkerMsg::SetClipLut { .. }
            | WorkerMsg::SetClipAnimation { .. }
            // Effects and transitions repaint the canvas at the playhead.
            | WorkerMsg::AddEffect { .. }
            | WorkerMsg::RemoveEffect { .. }
            | WorkerMsg::SetEffectParam { .. }
            | WorkerMsg::AddTransition { .. }
            | WorkerMsg::RemoveTransition { .. }
            | WorkerMsg::SetTransition { .. }
            // Aspect reshapes the composite, background recolors it.
            | WorkerMsg::SetCanvas { .. }
            | WorkerMsg::SetParamKeyframe { .. }
            | WorkerMsg::SetParamKeyframeTangents { .. }
            | WorkerMsg::SetParamConstant { .. }
            | WorkerMsg::RemoveParamKeyframe { .. }
            | WorkerMsg::MoveParamKeyframe { .. }
            | WorkerMsg::ApplyEasingPreset { .. }
            | WorkerMsg::RetimeKeyframes { .. }
            | WorkerMsg::RemoveKeyframesAt { .. }
            | WorkerMsg::SplitClip { .. }
            | WorkerMsg::RippleDeleteClips { .. }
            | WorkerMsg::ReverseClip { .. }
            | WorkerMsg::PasteAt { .. }
            | WorkerMsg::DuplicateClips { .. }
            | WorkerMsg::Undo
            | WorkerMsg::Redo
            // A replayed agent plan can create/move/restyle any clip; repaint
            // the canvas so the result is visible without a scrub.
            | WorkerMsg::AgentApplyPlan { .. }
            | WorkerMsg::SetMainMagnet(_)
            | WorkerMsg::SetTrackFlag { .. }
            | WorkerMsg::OpenProject { .. }
            | WorkerMsg::OpenProjectRpc { .. }
            | WorkerMsg::NewProject
            | WorkerMsg::NewProjectRpc { .. }
            // A filled template is a whole new composite.
            | WorkerMsg::ApplyTemplate { .. }
            | WorkerMsg::ApplyTemplateRpc { .. }
            // Relinked media decodes again — refresh the stale composite.
            | WorkerMsg::RelinkMedia { .. }
            | WorkerMsg::RelinkFolder { .. }
            | WorkerMsg::RelinkMediaRpc { .. }
            | WorkerMsg::RelinkFolderRpc { .. }
            // A bound proxy swaps the decode source; repaint through it so
            // the (cleared) frame cache refills at the cheap decode cost.
            | WorkerMsg::ProxyReady { .. }
            // A forced library delete removes the source's clips too; an
            // unreferenced delete touches nothing on the canvas.
            | WorkerMsg::RemoveMedia { force: true, .. }
    )
}
