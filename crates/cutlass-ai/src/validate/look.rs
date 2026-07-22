//! Validation + lowering for the clip-look command family: mask,
//! chroma key, stabilization, filter, adjustments, animations, audio role.

use super::*;

use crate::wire::{
    SetAudioRole, SetClipAdjustments, SetClipAnimation, SetClipBlendMode, SetClipChroma,
    SetClipFilter, SetClipMask, SetClipStabilize,
};

pub(super) fn set_clip_mask(
    project: &Project,
    args: &SetClipMask,
) -> Result<EditCommand, Rejection> {
    let clip = clip_ref(project, args.clip)?;
    reject_audio_lane(project, clip, "masks need a visual frame", args.clip)?;
    if clip.is_generated() {
        return Err(Rejection::new(format!(
            "clip {} is a generated clip; set_clip_mask only works on media \
                     clips (footage with a source file)",
            args.clip
        )));
    }
    let mask = match &args.mask {
        None => None,
        Some(wire) => Some(lower_mask(wire)?),
    };
    Ok(EditCommand::SetClipMask {
        clip: clip.id,
        mask,
    })
}

pub(super) fn set_clip_chroma(
    project: &Project,
    args: &SetClipChroma,
) -> Result<EditCommand, Rejection> {
    let clip = clip_ref(project, args.clip)?;
    reject_audio_lane(project, clip, "chroma key needs a visual frame", args.clip)?;
    if clip.is_generated() {
        return Err(Rejection::new(format!(
            "clip {} is a generated clip; set_clip_chroma only works on media \
                     clips (footage with a source file)",
            args.clip
        )));
    }
    let chroma = match &args.chroma {
        None => None,
        Some(wire) => Some(lower_chroma(wire)?),
    };
    Ok(EditCommand::SetClipChroma {
        clip: clip.id,
        chroma,
    })
}

pub(super) fn set_clip_stabilize(
    project: &Project,
    args: &SetClipStabilize,
) -> Result<EditCommand, Rejection> {
    let clip = clip_ref(project, args.clip)?;
    reject_audio_lane(
        project,
        clip,
        "stabilization needs a visual frame",
        args.clip,
    )?;
    if clip.is_generated() {
        return Err(Rejection::new(format!(
            "clip {} is a generated clip; set_clip_stabilize only works on media \
                     clips (footage with a source file)",
            args.clip
        )));
    }
    if let cutlass_models::ClipSource::Media { media, .. } = &clip.content
        && project.media(*media).is_some_and(|m| m.is_image)
    {
        return Err(Rejection::new(format!(
            "clip {} is a still image; stabilization requires video",
            args.clip
        )));
    }
    let stabilize = args.level.map(lower_stabilize);
    Ok(EditCommand::SetClipStabilize {
        clip: clip.id,
        stabilize,
    })
}

pub(super) fn set_clip_filter(
    project: &Project,
    args: &SetClipFilter,
) -> Result<EditCommand, Rejection> {
    let clip = clip_ref(project, args.clip)?;
    reject_audio_lane(project, clip, "filters need a visual frame", args.clip)?;
    let filter = match &args.filter {
        None => None,
        Some(wire) => Some(lower_filter(wire)?),
    };
    Ok(EditCommand::SetClipFilter {
        clip: clip.id,
        filter,
    })
}

pub(super) fn set_clip_blend_mode(
    project: &Project,
    args: &SetClipBlendMode,
) -> Result<EditCommand, Rejection> {
    let clip = clip_ref(project, args.clip)?;
    reject_audio_lane(project, clip, "blend modes need a visual frame", args.clip)?;
    Ok(EditCommand::SetClipBlendMode {
        clip: clip.id,
        mode: lower_blend_mode(args.mode),
    })
}

pub(super) fn set_clip_adjustments(
    project: &Project,
    args: &SetClipAdjustments,
) -> Result<EditCommand, Rejection> {
    let clip = clip_ref(project, args.clip)?;
    reject_audio_lane(project, clip, "adjustments need a visual frame", args.clip)?;
    let mut adjust = clip.adjust.clone();
    if let Some(v) = args.brightness {
        adjust.brightness = unit_slider(v, "brightness")?.into();
    }
    if let Some(v) = args.contrast {
        adjust.contrast = unit_slider(v, "contrast")?.into();
    }
    if let Some(v) = args.saturation {
        adjust.saturation = unit_slider(v, "saturation")?.into();
    }
    if let Some(v) = args.exposure {
        adjust.exposure = unit_slider(v, "exposure")?.into();
    }
    if let Some(v) = args.temperature {
        adjust.temperature = unit_slider(v, "temperature")?.into();
    }
    adjust
        .validate()
        .map_err(|e| Rejection::new(e.to_string()))?;
    Ok(EditCommand::SetClipAdjustments {
        clip: clip.id,
        adjust,
    })
}

pub(super) fn set_clip_animation(
    project: &Project,
    args: &SetClipAnimation,
) -> Result<EditCommand, Rejection> {
    let clip = clip_ref(project, args.clip)?;
    reject_audio_lane(project, clip, "animations need a visual frame", args.clip)?;
    let slot = lower_animation_slot(args.slot);
    let animation = match &args.animation {
        None => None,
        Some(id) => {
            let spec = animation_spec(id).ok_or_else(|| {
                Rejection::new(format!(
                    "unknown animation '{id}'; available animations include fade_in, \
                             fade_out, pulse, slide_up, zoom_in, and others from the catalog"
                ))
            })?;
            if spec.slot != slot {
                return Err(Rejection::new(format!(
                    "animation '{id}' does not fit the {} slot",
                    match args.slot {
                        WireAnimationSlot::In => "in",
                        WireAnimationSlot::Out => "out",
                        WireAnimationSlot::Combo => "combo",
                    }
                )));
            }
            if spec.text_only
                && !matches!(
                    clip.content,
                    cutlass_models::ClipSource::Generated(Generator::Text { .. })
                )
            {
                return Err(Rejection::new(format!(
                    "animation '{id}' is a text-only preset"
                )));
            }
            let anim = AnimationRef {
                id: id.clone(),
                speed: args
                    .speed
                    .unwrap_or(cutlass_models::ANIMATION_PARAM_DEFAULT),
                intensity: args
                    .intensity
                    .unwrap_or(cutlass_models::ANIMATION_PARAM_DEFAULT),
                stagger: args
                    .stagger
                    .unwrap_or(cutlass_models::ANIMATION_PARAM_DEFAULT),
            };
            Some(
                anim.normalized_for(spec)
                    .map_err(|e| Rejection::new(format!("invalid animation params: {e}")))?,
            )
        }
    };
    Ok(EditCommand::SetClipAnimation {
        clip: clip.id,
        slot,
        animation,
    })
}

pub(super) fn set_audio_role(
    project: &Project,
    args: &SetAudioRole,
) -> Result<EditCommand, Rejection> {
    let clip = clip_ref(project, args.clip)?;
    let timeline = project.timeline();
    let on_audio = timeline
        .track_of(clip.id)
        .and_then(|id| timeline.track(id))
        .is_some_and(|t| t.kind == TrackKind::Audio);
    if !on_audio {
        return Err(Rejection::new(format!(
            "clip {} is not on an audio lane; set_audio_role only works on audio clips",
            args.clip
        )));
    }
    Ok(EditCommand::SetAudioRole {
        clip: clip.id,
        role: args.role.map(lower_audio_role),
    })
}
