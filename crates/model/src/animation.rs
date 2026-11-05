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
    /// A title's letters or words move one after another; the clip stays.
    Letters(LetterMotion),
    /// Over and over for as long as the clip shows, once each `duration`.
    Loop(LoopMotion),
}

/// How a loop moves the clip through one round.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoopMotion {
    /// Grows a little and back.
    Pulse,
    /// Dims now and then, like a failing light.
    Flicker,
    /// Drifts up and down.
    Float,
    /// Jitters side to side.
    Shake,
    /// Rocks about its middle.
    Swing,
    /// Slowly swells and fades a little, and back.
    Breathe,
}

/// How a title's letters arrive one after another, or leave.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LetterMotion {
    /// Each letter appears whole, in turn.
    Typewriter,
    /// Each letter fades up, overlapping the next.
    Fade,
    /// Each letter grows from nothing past its size and settles.
    Pop,
    /// Each word rises into place as it fades up.
    Rise,
}

/// How one letter is drawn at a moment of a letter animation: its opacity,
/// its shift in ems (y down), and its size around its own middle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LetterPose {
    pub alpha: f64,
    pub dx: f64,
    pub dy: f64,
    pub scale: f64,
}

impl LetterPose {
    /// Where every letter rests.
    pub const REST: Self = Self {
        alpha: 1.0,
        dx: 0.0,
        dy: 0.0,
        scale: 1.0,
    };

    /// This pose and then `other` on top of it.
    #[must_use]
    pub fn then(self, other: Self) -> Self {
        Self {
            alpha: self.alpha * other.alpha,
            dx: self.dx + other.dx,
            dy: self.dy + other.dy,
            scale: self.scale * other.scale,
        }
    }
}

/// Which letter of a title one is: its place among the drawn letters and
/// the word it is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Letter {
    pub index: usize,
    pub count: usize,
    pub word: usize,
    pub words: usize,
}

/// One letter animation of a title at a moment: how it moves and how far
/// along it is, from 0 to 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LetterStep {
    pub motion: LetterMotion,
    pub kind: AnimType,
    pub along: f64,
}

impl LetterStep {
    /// How `letter` is drawn at this step. An in animation brings the
    /// letters in first to last; an out one takes them away last to first.
    #[must_use]
    pub fn pose(&self, letter: Letter) -> LetterPose {
        let at = match self.kind {
            AnimType::Out => 1.0 - self.along,
            _ => self.along,
        };
        let (unit, units) = if self.motion == LetterMotion::Rise {
            (letter.word, letter.words)
        } else {
            (letter.index, letter.count)
        };
        let units = units.max(1);
        if self.motion == LetterMotion::Typewriter {
            let shown = (at * units as f64).floor() as usize > unit || at >= 1.0;
            let alpha = if shown { 1.0 } else { 0.0 };
            return LetterPose {
                alpha,
                ..LetterPose::REST
            };
        }
        // Each unit takes a window of the animation; the windows overlap so
        // the next starts before this one settles.
        let window = (3.0 / (units as f64 + 2.0)).min(1.0);
        let start = if units > 1 {
            unit.min(units - 1) as f64 * (1.0 - window) / (units - 1) as f64
        } else {
            0.0
        };
        let t = ((at - start) / window).clamp(0.0, 1.0);
        let eased = 1.0 - (1.0 - t).powi(3);
        match self.motion {
            LetterMotion::Typewriter | LetterMotion::Fade => LetterPose {
                alpha: eased,
                ..LetterPose::REST
            },
            LetterMotion::Pop => {
                // Grows past its size and settles back.
                let back = 1.701_58;
                let u = t - 1.0;
                LetterPose {
                    alpha: (t * 3.0).min(1.0),
                    scale: (1.0 + (back + 1.0) * u.powi(3) + back * u.powi(2)).max(0.0),
                    ..LetterPose::REST
                }
            }
            LetterMotion::Rise => LetterPose {
                alpha: eased,
                dy: 0.6 * (1.0 - eased),
                ..LetterPose::REST
            },
        }
    }
}

/// The letter animations of a title at `time`, in the order stored.
#[must_use]
pub fn letters_at(draft: &Draft, segment: &Segment, time: Micros) -> Vec<LetterStep> {
    let Some(group) = segment_animations(draft, segment) else {
        return Vec::new();
    };
    let offset = time.0.saturating_sub(segment.target.start.0);
    let length = segment.target.duration.0;
    group
        .animations
        .iter()
        .filter_map(|animation| {
            let preset = animation_preset(animation)?;
            let Motion::Letters(motion) = preset.motion else {
                return None;
            };
            let along = progress(preset, animation, offset, length)?;
            Some(LetterStep {
                motion,
                kind: preset.kind,
                along,
            })
        })
        .collect()
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
        name: "Typewriter",
        kind: AnimType::In,
        resource_id: "cutlass-anim-in-typewriter",
        motion: Motion::Letters(LetterMotion::Typewriter),
        duration: Micros(1_000_000),
    },
    AnimationPreset {
        name: "Letter Fade",
        kind: AnimType::In,
        resource_id: "cutlass-anim-in-letter-fade",
        motion: Motion::Letters(LetterMotion::Fade),
        duration: Micros(1_000_000),
    },
    AnimationPreset {
        name: "Pop",
        kind: AnimType::In,
        resource_id: "cutlass-anim-in-pop",
        motion: Motion::Letters(LetterMotion::Pop),
        duration: Micros(1_000_000),
    },
    AnimationPreset {
        name: "Word Rise",
        kind: AnimType::In,
        resource_id: "cutlass-anim-in-word-rise",
        motion: Motion::Letters(LetterMotion::Rise),
        duration: Micros(1_000_000),
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
    AnimationPreset {
        name: "Erase",
        kind: AnimType::Out,
        resource_id: "cutlass-anim-out-erase",
        motion: Motion::Letters(LetterMotion::Typewriter),
        duration: Micros(1_000_000),
    },
    AnimationPreset {
        name: "Letter Fade",
        kind: AnimType::Out,
        resource_id: "cutlass-anim-out-letter-fade",
        motion: Motion::Letters(LetterMotion::Fade),
        duration: Micros(1_000_000),
    },
    AnimationPreset {
        name: "Pop",
        kind: AnimType::Out,
        resource_id: "cutlass-anim-out-pop",
        motion: Motion::Letters(LetterMotion::Pop),
        duration: Micros(1_000_000),
    },
    AnimationPreset {
        name: "Word Sink",
        kind: AnimType::Out,
        resource_id: "cutlass-anim-out-word-sink",
        motion: Motion::Letters(LetterMotion::Rise),
        duration: Micros(1_000_000),
    },
    AnimationPreset {
        name: "Pulse",
        kind: AnimType::Loop,
        resource_id: "cutlass-anim-loop-pulse",
        motion: Motion::Loop(LoopMotion::Pulse),
        duration: Micros(1_000_000),
    },
    AnimationPreset {
        name: "Flicker",
        kind: AnimType::Loop,
        resource_id: "cutlass-anim-loop-flicker",
        motion: Motion::Loop(LoopMotion::Flicker),
        duration: Micros(1_000_000),
    },
    AnimationPreset {
        name: "Float",
        kind: AnimType::Loop,
        resource_id: "cutlass-anim-loop-float",
        motion: Motion::Loop(LoopMotion::Float),
        duration: Micros(2_000_000),
    },
    AnimationPreset {
        name: "Shake",
        kind: AnimType::Loop,
        resource_id: "cutlass-anim-loop-shake",
        motion: Motion::Loop(LoopMotion::Shake),
        duration: Micros(1_000_000),
    },
    AnimationPreset {
        name: "Swing",
        kind: AnimType::Loop,
        resource_id: "cutlass-anim-loop-swing",
        motion: Motion::Loop(LoopMotion::Swing),
        duration: Micros(1_500_000),
    },
    AnimationPreset {
        name: "Breathe",
        kind: AnimType::Loop,
        resource_id: "cutlass-anim-loop-breathe",
        motion: Motion::Loop(LoopMotion::Breathe),
        duration: Micros(3_000_000),
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
    let Some(along) = progress(preset, animation, offset, length) else {
        return;
    };
    // How far from rest, 1 being all the way away: an in eases out of its
    // far end, an out eases into it.
    if let Motion::Loop(motion) = preset.motion {
        swing(clip, uniform, motion, along);
        return;
    }
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
        Motion::Letters(_) | Motion::Loop(_) => {}
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

/// Move `clip` by a loop `round` of the way through one of its rounds.
fn swing(clip: &mut Clip, uniform: &mut UniformScale, motion: LoopMotion, round: f64) {
    let wave = (round * std::f64::consts::TAU).sin();
    let rise = 0.5 - 0.5 * (round * std::f64::consts::TAU).cos();
    let grow = |clip: &mut Clip, uniform: &mut UniformScale, factor: f64| {
        if uniform.on {
            uniform.value *= factor;
        } else {
            clip.scale.x *= factor;
            clip.scale.y *= factor;
        }
    };
    match motion {
        LoopMotion::Pulse => grow(clip, uniform, 1.0 + 0.1 * rise),
        LoopMotion::Flicker => {
            // Two short dips each round.
            if (0.10..0.16).contains(&round) || (0.52..0.60).contains(&round) {
                clip.alpha *= 0.25;
            }
        }
        LoopMotion::Float => clip.transform.y += 0.04 * wave,
        LoopMotion::Shake => clip.transform.x += 0.02 * (round * std::f64::consts::TAU * 4.0).sin(),
        LoopMotion::Swing => clip.rotation += 8.0 * wave,
        LoopMotion::Breathe => {
            grow(clip, uniform, 1.0 + 0.05 * rise);
            clip.alpha *= 1.0 - 0.2 * rise;
        }
    }
}

/// How far along one animation is at `offset` from the segment's start,
/// from 0 to 1. An out animation ends with the segment, `length` long,
/// whatever its stored `start` says, so a trim keeps it on the end. A loop
/// starts over each `duration`, so this is how far through its round.
fn progress(
    preset: &AnimationPreset,
    animation: &StickerAnimation,
    offset: i64,
    length: i64,
) -> Option<f64> {
    let duration = animation.duration.map(|d| d.0).filter(|d| *d > 0)?;
    if preset.kind == AnimType::Loop {
        let into = (offset - animation.start.0).max(0);
        return Some((into % duration) as f64 / duration as f64);
    }
    let start = if preset.kind == AnimType::Out {
        length - duration
    } else {
        animation.start.0
    };
    Some(((offset - start) as f64 / duration as f64).clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stored(preset: &AnimationPreset, start: i64, duration: i64) -> StickerAnimation {
        StickerAnimation {
            kind: match preset.kind {
                AnimType::In => "in",
                AnimType::Out => "out",
                _ => "loop",
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

    fn letter(index: usize, count: usize) -> Letter {
        Letter {
            index,
            count,
            word: index / 2,
            words: count.div_ceil(2),
        }
    }

    #[test]
    fn a_typewriter_shows_letters_in_turn_and_an_erase_takes_them_from_the_end() {
        let step = |kind, along| LetterStep {
            motion: LetterMotion::Typewriter,
            kind,
            along,
        };
        let shown = |s: LetterStep| {
            (0..4)
                .map(|i| s.pose(letter(i, 4)).alpha > 0.0)
                .collect::<Vec<_>>()
        };
        assert_eq!(shown(step(AnimType::In, 0.0)), [false; 4]);
        assert_eq!(shown(step(AnimType::In, 0.5)), [true, true, false, false]);
        assert_eq!(shown(step(AnimType::In, 1.0)), [true; 4]);
        assert_eq!(shown(step(AnimType::Out, 0.25)), [true, true, true, false]);
        assert_eq!(shown(step(AnimType::Out, 1.0)), [false; 4]);
    }

    #[test]
    fn letters_fade_pop_and_rise_one_after_another_to_rest() {
        for motion in [LetterMotion::Fade, LetterMotion::Pop, LetterMotion::Rise] {
            let at = |along| LetterStep {
                motion,
                kind: AnimType::In,
                along,
            };
            assert_eq!(at(1.0).pose(letter(5, 6)), LetterPose::REST, "{motion:?}");
            assert_eq!(at(0.0).pose(letter(0, 6)).alpha, 0.0, "{motion:?}");
            // Partway, the first is further in than the last.
            let first = at(0.3).pose(letter(0, 6));
            let last = at(0.3).pose(letter(5, 6));
            assert!(first.alpha > last.alpha, "{motion:?}");
        }
        let pop = LetterStep {
            motion: LetterMotion::Pop,
            kind: AnimType::In,
            along: 0.7,
        };
        assert!(pop.pose(letter(0, 1)).scale > 1.0);
        let rise = LetterStep {
            motion: LetterMotion::Rise,
            kind: AnimType::In,
            along: 0.1,
        };
        assert!(rise.pose(letter(0, 4)).dy > 0.0);
    }

    #[test]
    fn a_letter_animation_moves_letters_not_the_clip() {
        let mut uniform = UniformScale::default();
        let mut clip = Clip::default();
        let pop = stored(by("Pop", AnimType::In), 0, 1_000);
        animate(&mut clip, &mut uniform, &pop, 0, 10_000);
        assert_eq!(clip, Clip::default());
    }

    #[test]
    fn a_loop_comes_round_again_each_duration() {
        let swing = stored(by("Swing", AnimType::Loop), 0, 1_000);
        let at = |offset| {
            let mut clip = Clip::default();
            let mut uniform = UniformScale::default();
            animate(&mut clip, &mut uniform, &swing, offset, 10_000);
            clip.rotation
        };
        assert!((at(250) - 8.0).abs() < 1e-9);
        assert!((at(750) + 8.0).abs() < 1e-9);
        assert!((at(5_250) - at(250)).abs() < 1e-9);
        assert!(at(1_000).abs() < 1e-9);
        let pulse = stored(by("Pulse", AnimType::Loop), 0, 1_000);
        let mut clip = Clip::default();
        let mut uniform = UniformScale::default();
        animate(&mut clip, &mut uniform, &pulse, 500, 10_000);
        let size = if uniform.on {
            uniform.value
        } else {
            clip.scale.x
        };
        assert!((size - 1.1).abs() < 1e-9);
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
