//! In and out animations the app draws itself: fades, slides and zooms.
//!
//! The reference editor stores a clip's animations in one
//! `sticker_animation` row of `material_animations`, named in the
//! segment's `extra_material_refs`. Each animation has a `type` of `in` or
//! `out`, a `start` from the segment's target start, a `duration`, and the
//! `resource_id` of its effect package. An in animation starts at 0; an out
//! animation ends where the segment ends. The presets here carry a motion
//! instead of a package, so a row whose `resource_id` is one of theirs is
//! drawn here and not looked up as a package.

use crate::clip::{Clip, UniformScale};
use crate::draft::Draft;
use crate::enums::AnimType;
use crate::keyframe::clip_at;
use crate::materials::{MaterialAnimations, MaterialRef, StickerAnimation};
use crate::segment::Segment;
use crate::time::Micros;

/// How a preset moves the clip on its way in or out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Motion {
    /// Opacity from nothing.
    Fade,
    /// Travel in this direction, in canvas half-widths, y up. In arrives
    /// travelling this way; out leaves travelling this way.
    Slide(f64, f64),
    /// Size away from rest, faded at the far end.
    Zoom(f64),
}

/// One animation the panel offers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnimationPreset {
    pub name: &'static str,
    pub kind: AnimType,
    pub resource_id: &'static str,
    pub motion: Motion,
    /// The length a new animation starts with.
    pub duration: Micros,
}

const HALF: Micros = Micros(500_000);

/// The presets, in the order the panel shows them. Fade In and Fade Out
/// keep the reference editor's resource ids, so its drafts play here too.
pub const ANIMATION_PRESETS: &[AnimationPreset] = &[
    AnimationPreset {
        name: "Fade In",
        kind: AnimType::In,
        resource_id: "7649736866270350613",
        motion: Motion::Fade,
        duration: Micros(1_200_000),
    },
    AnimationPreset {
        name: "Slide Right",
        kind: AnimType::In,
        resource_id: "cutlass-anim-in-slide-right",
        motion: Motion::Slide(1.0, 0.0),
        duration: HALF,
    },
    AnimationPreset {
        name: "Slide Left",
        kind: AnimType::In,
        resource_id: "cutlass-anim-in-slide-left",
        motion: Motion::Slide(-1.0, 0.0),
        duration: HALF,
    },
    AnimationPreset {
        name: "Slide Up",
        kind: AnimType::In,
        resource_id: "cutlass-anim-in-slide-up",
        motion: Motion::Slide(0.0, 1.0),
        duration: HALF,
    },
    AnimationPreset {
        name: "Slide Down",
        kind: AnimType::In,
        resource_id: "cutlass-anim-in-slide-down",
        motion: Motion::Slide(0.0, -1.0),
        duration: HALF,
    },
    AnimationPreset {
        name: "Zoom In",
        kind: AnimType::In,
        resource_id: "cutlass-anim-in-zoom-in",
        motion: Motion::Zoom(0.5),
        duration: HALF,
    },
    AnimationPreset {
        name: "Zoom Out",
        kind: AnimType::In,
        resource_id: "cutlass-anim-in-zoom-out",
        motion: Motion::Zoom(1.6),
        duration: HALF,
    },
    AnimationPreset {
        name: "Fade Out",
        kind: AnimType::Out,
        resource_id: "6798320902548230669",
        motion: Motion::Fade,
        duration: HALF,
    },
    AnimationPreset {
        name: "Slide Right",
        kind: AnimType::Out,
        resource_id: "cutlass-anim-out-slide-right",
        motion: Motion::Slide(1.0, 0.0),
        duration: HALF,
    },
    AnimationPreset {
        name: "Slide Left",
        kind: AnimType::Out,
        resource_id: "cutlass-anim-out-slide-left",
        motion: Motion::Slide(-1.0, 0.0),
        duration: HALF,
    },
    AnimationPreset {
        name: "Slide Up",
        kind: AnimType::Out,
        resource_id: "cutlass-anim-out-slide-up",
        motion: Motion::Slide(0.0, 1.0),
        duration: HALF,
    },
    AnimationPreset {
        name: "Slide Down",
        kind: AnimType::Out,
        resource_id: "cutlass-anim-out-slide-down",
        motion: Motion::Slide(0.0, -1.0),
        duration: HALF,
    },
    AnimationPreset {
        name: "Zoom In",
        kind: AnimType::Out,
        resource_id: "cutlass-anim-out-zoom-in",
        motion: Motion::Zoom(1.6),
        duration: HALF,
    },
    AnimationPreset {
        name: "Zoom Out",
        kind: AnimType::Out,
        resource_id: "cutlass-anim-out-zoom-out",
        motion: Motion::Zoom(0.5),
        duration: HALF,
    },
];

/// The preset a stored animation names, if it is one of ours.
#[must_use]
pub fn animation_preset(animation: &StickerAnimation) -> Option<&'static AnimationPreset> {
    let kind = animation.anim_type()?;
    ANIMATION_PRESETS
        .iter()
        .find(|preset| preset.resource_id == animation.resource_id && preset.kind == kind)
}

/// The clip's animation row: the first `material_animations` row its
/// `extra_material_refs` name.
#[must_use]
pub fn segment_animations<'a>(
    draft: &'a Draft,
    segment: &Segment,
) -> Option<&'a MaterialAnimations> {
    segment
        .extra_material_refs
        .iter()
        .find_map(|id| match draft.materials.get(id.as_str()) {
            Some(MaterialRef::Animations(group)) => Some(group),
            _ => None,
        })
}

/// The clip at `time` with its keyframes and then its animations applied.
#[must_use]
pub fn animated_at(draft: &Draft, segment: &Segment, time: Micros) -> Option<(Clip, UniformScale)> {
    let (mut clip, mut uniform) = clip_at(segment, &draft.keyframe_graph_list, time)?;
    if let Some(group) = segment_animations(draft, segment) {
        let offset = time.0.saturating_sub(segment.target.start.0);
        let length = segment.target.duration.0;
        for animation in &group.animations {
            animate(&mut clip, &mut uniform, animation, offset, length);
        }
    }
    Some((clip, uniform))
}

/// Move `clip` by one animation at `offset` from the segment's start. An
/// out animation ends with the segment, `length` long, whatever its stored
/// `start` says, so a trim keeps it on the end.
fn animate(
    clip: &mut Clip,
    uniform: &mut UniformScale,
    animation: &StickerAnimation,
    offset: i64,
    length: i64,
) {
    let Some(preset) = animation_preset(animation) else {
        return;
    };
    let Some(duration) = animation.duration.map(|d| d.0).filter(|d| *d > 0) else {
        return;
    };
    let start = if preset.kind == AnimType::Out {
        length - duration
    } else {
        animation.start.0
    };
    let along = ((offset - start) as f64 / duration as f64).clamp(0.0, 1.0);
    // How far from rest, 1 being all the way away: an in eases out of its
    // far end, an out eases into it.
    let away = match preset.kind {
        AnimType::In => (1.0 - along).powi(3),
        AnimType::Out => along.powi(3),
        _ => return,
    };
    if away == 0.0 {
        return;
    }
    match preset.motion {
        Motion::Fade => clip.alpha *= 1.0 - away,
        Motion::Slide(dx, dy) => {
            // In arrives travelling (dx, dy), so it starts behind; out
            // leaves travelling (dx, dy), so it ends ahead. Far enough is
            // the canvas half plus the clip's own half, grown by its size
            // and its turn, so a large or turned clip leaves the frame too.
            let size = if uniform.on {
                uniform.value.abs()
            } else {
                clip.scale.x.abs().max(clip.scale.y.abs())
            };
            let (sin, cos) = clip.rotation.to_radians().sin_cos();
            let far = 1.0 + size * (cos.abs() + sin.abs());
            let sign = if preset.kind == AnimType::In {
                -far
            } else {
                far
            };
            clip.transform.x += dx * sign * away;
            clip.transform.y += dy * sign * away;
        }
        Motion::Zoom(far) => {
            let factor = 1.0 + (far - 1.0) * away;
            if uniform.on {
                uniform.value *= factor;
            } else {
                clip.scale.x *= factor;
                clip.scale.y *= factor;
            }
            clip.alpha *= 1.0 - away;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stored(preset: &AnimationPreset, start: i64, duration: i64) -> StickerAnimation {
        StickerAnimation {
            kind: if preset.kind == AnimType::In {
                "in"
            } else {
                "out"
            }
            .into(),
            resource_id: preset.resource_id.into(),
            start: Micros(start),
            duration: Some(Micros(duration)),
            ..StickerAnimation::default()
        }
    }

    fn by(name: &str, kind: AnimType) -> &'static AnimationPreset {
        ANIMATION_PRESETS
            .iter()
            .find(|p| p.name == name && p.kind == kind)
            .expect("preset")
    }

    #[test]
    fn a_fade_in_rises_from_nothing_and_a_fade_out_falls_to_it() {
        let mut uniform = UniformScale::default();
        let fade_in = stored(by("Fade In", AnimType::In), 0, 1_000);
        let mut clip = Clip::default();
        animate(&mut clip, &mut uniform, &fade_in, 0, 10_000);
        assert_eq!(clip.alpha, 0.0);
        let mut clip = Clip::default();
        animate(&mut clip, &mut uniform, &fade_in, 500, 10_000);
        assert!((clip.alpha - 0.875).abs() < 1e-9);
        let mut clip = Clip::default();
        animate(&mut clip, &mut uniform, &fade_in, 2_000, 10_000);
        assert_eq!(clip, Clip::default());

        let fade_out = stored(by("Fade Out", AnimType::Out), 9_000, 1_000);
        let mut clip = Clip::default();
        animate(&mut clip, &mut uniform, &fade_out, 8_000, 10_000);
        assert_eq!(clip.alpha, 1.0);
        let mut clip = Clip::default();
        animate(&mut clip, &mut uniform, &fade_out, 10_000, 10_000);
        assert_eq!(clip.alpha, 0.0);
    }

    #[test]
    fn a_slide_comes_from_behind_and_leaves_ahead_at_the_clip_end() {
        let mut uniform = UniformScale::default();
        let mut clip = Clip::default();
        animate(
            &mut clip,
            &mut uniform,
            &stored(by("Slide Right", AnimType::In), 0, 100),
            0,
            10_000,
        );
        assert_eq!(clip.transform.x, -2.0);
        let mut clip = Clip::default();
        animate(
            &mut clip,
            &mut uniform,
            &stored(by("Slide Up", AnimType::Out), 0, 100),
            10_000,
            10_000,
        );
        assert_eq!(clip.transform.y, 2.0);
    }

    #[test]
    fn a_zoom_scales_whichever_size_drives_the_clip() {
        let zoom = stored(by("Zoom Out", AnimType::Out), 0, 100);
        let mut clip = Clip::default();
        let mut uniform = UniformScale {
            on: true,
            value: 2.0,
            ..UniformScale::default()
        };
        animate(&mut clip, &mut uniform, &zoom, 10_000, 10_000);
        assert_eq!(uniform.value, 1.0);
        assert_eq!(clip.scale.x, 1.0);
        assert_eq!(clip.alpha, 0.0);
    }

    #[test]
    fn a_package_or_a_mismatched_type_is_not_ours() {
        let mut other = stored(by("Fade In", AnimType::In), 0, 100);
        other.resource_id = "123".into();
        assert!(animation_preset(&other).is_none());
        let mut flipped = stored(by("Fade In", AnimType::In), 0, 100);
        flipped.kind = "out".into();
        assert!(animation_preset(&flipped).is_none());
    }
}
