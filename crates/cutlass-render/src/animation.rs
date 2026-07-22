//! Resolve persisted look-animation presets into transform/opacity deltas.

use cutlass_core::Rational;
use cutlass_models::{
    Clip, ClipTransform, Easing, animation_spec, look_animation_combo_period_ticks,
    look_animation_window_ticks,
};

/// Normalized slide distance as a fraction of canvas height (+y down).
const SLIDE_OFFSET: f32 = 0.18;

/// Multiplicative transform delta sampled from one animation preset.
#[derive(Debug, Clone, Copy, PartialEq)]
struct AnimationDelta {
    position: [f32; 2],
    scale: f32,
    rotation: f32,
    opacity: f32,
}

impl AnimationDelta {
    const IDENTITY: Self = Self {
        position: [0.0, 0.0],
        scale: 1.0,
        rotation: 0.0,
        opacity: 1.0,
    };
}

/// Fold look-animation presets onto a clip's sampled transform at resolve time.
pub(crate) fn apply_look_animations(
    clip: &Clip,
    base: ClipTransform,
    local_tick: i64,
    local_tick_f: f64,
    rate: Rational,
) -> ClipTransform {
    let duration = clip.timeline.duration.value.max(1);
    let window = look_animation_window_ticks(duration, rate);
    let mut deltas = Vec::with_capacity(2);

    if let Some(combo) = &clip.animation_combo {
        // Per-character (text_only) presets are sampled into TextAnimation at
        // resolve time and applied per glyph — skip the whole-layer path.
        if !is_per_character(&combo.id) {
            let period = look_animation_combo_period_ticks(rate);
            let phase = (local_tick_f % period as f64) / period as f64;
            deltas.push(sample_combo(&combo.id, phase));
        }
    } else {
        if let Some(anim) = &clip.animation_in
            && local_tick < window
            && !is_per_character(&anim.id)
        {
            let raw = (local_tick_f / window as f64).clamp(0.0, 1.0);
            let eased = f64::from(Easing::EaseOut.apply(raw as f32));
            deltas.push(sample_entrance(&anim.id, eased));
        }
        if let Some(anim) = &clip.animation_out {
            let out_start = duration - window;
            if local_tick >= out_start && !is_per_character(&anim.id) {
                let raw = ((local_tick_f - out_start as f64) / (window - 1).max(1) as f64)
                    .clamp(0.0, 1.0);
                let eased = f64::from(Easing::EaseIn.apply(raw as f32));
                deltas.push(sample_exit(&anim.id, eased));
            }
        }
    }

    if deltas.is_empty() {
        return base;
    }
    compose_transform(base, &deltas)
}

fn compose_transform(base: ClipTransform, deltas: &[AnimationDelta]) -> ClipTransform {
    let mut xf = base;
    for delta in deltas {
        xf.position[0] += delta.position[0];
        xf.position[1] += delta.position[1];
        xf.scale *= delta.scale;
        xf.rotation += delta.rotation;
        xf.opacity = (xf.opacity * delta.opacity).clamp(0.0, 1.0);
    }
    xf
}

/// Text-only catalog presets animate per character on the glyph path.
pub(crate) fn is_per_character(id: &str) -> bool {
    animation_spec(id).is_some_and(|s| s.text_only)
}

fn sample_entrance(id: &str, t: f64) -> AnimationDelta {
    let t = t.clamp(0.0, 1.0);
    let inv = 1.0 - t;
    match id {
        "fade_in" => AnimationDelta {
            opacity: t as f32,
            ..AnimationDelta::IDENTITY
        },
        "slide_up" => AnimationDelta {
            position: [0.0, inv as f32 * SLIDE_OFFSET],
            opacity: t as f32,
            ..AnimationDelta::IDENTITY
        },
        "zoom_in" => AnimationDelta {
            scale: (0.25 + 0.75 * t) as f32,
            opacity: t as f32,
            ..AnimationDelta::IDENTITY
        },
        "spin_in" => AnimationDelta {
            rotation: (inv * -360.0) as f32,
            opacity: t as f32,
            scale: (0.5 + 0.5 * t) as f32,
            ..AnimationDelta::IDENTITY
        },
        "bounce" => AnimationDelta {
            scale: bounce_scale(t) as f32,
            opacity: t as f32,
            position: [0.0, inv as f32 * SLIDE_OFFSET * 0.35],
            ..AnimationDelta::IDENTITY
        },
        _ => AnimationDelta::IDENTITY,
    }
}

fn sample_exit(id: &str, t: f64) -> AnimationDelta {
    let t = t.clamp(0.0, 1.0);
    let inv = 1.0 - t;
    match id {
        "fade_out" => AnimationDelta {
            opacity: inv as f32,
            ..AnimationDelta::IDENTITY
        },
        "slide_down" => AnimationDelta {
            position: [0.0, t as f32 * SLIDE_OFFSET],
            opacity: inv as f32,
            ..AnimationDelta::IDENTITY
        },
        "zoom_out" => AnimationDelta {
            scale: (1.0 - 0.75 * t) as f32,
            opacity: inv as f32,
            ..AnimationDelta::IDENTITY
        },
        "spin_out" => AnimationDelta {
            rotation: (t * 360.0) as f32,
            opacity: inv as f32,
            scale: (1.0 - 0.5 * t) as f32,
            ..AnimationDelta::IDENTITY
        },
        "drop" => AnimationDelta {
            position: [0.0, t as f32 * SLIDE_OFFSET * 1.4],
            opacity: inv as f32,
            scale: (1.0 - 0.35 * t) as f32,
            ..AnimationDelta::IDENTITY
        },
        _ => AnimationDelta::IDENTITY,
    }
}

fn sample_combo(id: &str, phase: f64) -> AnimationDelta {
    let phase = phase.fract();
    let wave = (phase * std::f64::consts::TAU).sin();
    match id {
        "pulse" => AnimationDelta {
            scale: (1.0 + 0.08 * wave) as f32,
            ..AnimationDelta::IDENTITY
        },
        "rock" => AnimationDelta {
            rotation: (6.0 * wave) as f32,
            ..AnimationDelta::IDENTITY
        },
        "swing" => AnimationDelta {
            rotation: (12.0 * (phase * std::f64::consts::PI).sin()) as f32,
            ..AnimationDelta::IDENTITY
        },
        "flicker" => AnimationDelta {
            opacity: if (phase * 8.0).fract() < 0.5 {
                1.0
            } else {
                0.35
            },
            ..AnimationDelta::IDENTITY
        },
        "breathe" => AnimationDelta {
            scale: (1.0 + 0.05 * wave) as f32,
            opacity: (0.85 + 0.15 * ((phase * std::f64::consts::PI).sin() + 1.0) * 0.5) as f32,
            ..AnimationDelta::IDENTITY
        },
        // text_only presets (typewriter, text_fade, …) are handled by
        // `render::text_anim` — they must not reach this whole-layer sampler.
        _ => AnimationDelta::IDENTITY,
    }
}

/// Penner-style ease-out bounce for the entrance preset.
fn bounce_scale(t: f64) -> f64 {
    if t < 1.0 / 2.75 {
        7.5625 * t * t
    } else if t < 2.0 / 2.75 {
        let t = t - 1.5 / 2.75;
        7.5625 * t * t + 0.75
    } else if t < 2.5 / 2.75 {
        let t = t - 2.25 / 2.75;
        7.5625 * t * t + 0.9375
    } else {
        let t = t - 2.625 / 2.75;
        7.5625 * t * t + 0.984375
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_core::Rational;
    use cutlass_models::{AnimationRef, Generator, TimeRange, animation_catalog, animation_spec};

    const R24: Rational = Rational::new(24, 1);

    fn solid_clip(duration: i64) -> Clip {
        Clip::generated(
            Generator::SolidColor {
                rgba: [255, 0, 0, 255],
            },
            TimeRange::at_rate(0, duration, R24),
        )
    }

    #[test]
    fn catalog_ids_all_have_handlers() {
        for spec in animation_catalog() {
            // Per-character (text_only) presets are sampled in text_anim.rs.
            if spec.text_only {
                assert!(animation_spec(spec.id).is_some());
                continue;
            }
            let delta = match spec.slot {
                cutlass_models::AnimationSlot::In => sample_entrance(spec.id, 0.5),
                cutlass_models::AnimationSlot::Out => sample_exit(spec.id, 0.92),
                cutlass_models::AnimationSlot::Combo => sample_combo(spec.id, 0.07),
            };
            assert!(
                delta != AnimationDelta::IDENTITY,
                "animation '{}' produced identity at sample",
                spec.id
            );
            assert!(animation_spec(spec.id).is_some());
        }
    }

    #[test]
    fn fade_in_ramps_opacity_from_zero_at_start() {
        let mut clip = solid_clip(48);
        clip.animation_in = Some(AnimationRef::new("fade_in"));
        let start = apply_look_animations(&clip, ClipTransform::IDENTITY, 0, 0.0, R24);
        assert!(start.opacity < 0.01, "opacity at start = {}", start.opacity);

        let mid = apply_look_animations(&clip, ClipTransform::IDENTITY, 24, 24.0, R24);
        approx(mid.opacity, 1.0);
    }

    #[test]
    fn slide_up_offsets_center_at_start() {
        let mut clip = solid_clip(48);
        clip.animation_in = Some(AnimationRef::new("slide_up"));
        let start = apply_look_animations(&clip, ClipTransform::IDENTITY, 0, 0.0, R24);
        assert!(start.position[1] > 0.05);
        let mid = apply_look_animations(&clip, ClipTransform::IDENTITY, 24, 24.0, R24);
        approx(mid.position[1], 0.0);
    }

    #[test]
    fn zoom_in_scales_down_at_start() {
        let mut clip = solid_clip(48);
        clip.animation_in = Some(AnimationRef::new("zoom_in"));
        let start = apply_look_animations(&clip, ClipTransform::IDENTITY, 0, 0.0, R24);
        assert!(start.scale < 0.5);
    }

    #[test]
    fn fade_out_ramps_at_tail() {
        let mut clip = solid_clip(48);
        clip.animation_out = Some(AnimationRef::new("fade_out"));
        let tail = apply_look_animations(&clip, ClipTransform::IDENTITY, 47, 47.0, R24);
        assert!(tail.opacity < 0.05);
        let mid = apply_look_animations(&clip, ClipTransform::IDENTITY, 20, 20.0, R24);
        approx(mid.opacity, 1.0);
    }

    #[test]
    fn combo_loops_and_supersedes_in_out() {
        let mut clip = solid_clip(48);
        clip.animation_in = Some(AnimationRef::new("fade_in"));
        clip.animation_out = Some(AnimationRef::new("fade_out"));
        clip.animation_combo = Some(AnimationRef::new("pulse"));
        let a = apply_look_animations(&clip, ClipTransform::IDENTITY, 0, 0.0, R24);
        let b = apply_look_animations(&clip, ClipTransform::IDENTITY, 6, 6.0, R24);
        assert!((a.scale - b.scale).abs() > 0.001);
        approx(a.opacity, 1.0);
    }

    #[test]
    fn combo_repeats_after_one_period() {
        let mut clip = solid_clip(120);
        clip.animation_combo = Some(AnimationRef::new("pulse"));
        let period = look_animation_combo_period_ticks(R24) as f64;
        let a = apply_look_animations(&clip, ClipTransform::IDENTITY, 0, 0.0, R24);
        let b = apply_look_animations(&clip, ClipTransform::IDENTITY, 0, period, R24);
        approx(a.scale, b.scale);
    }

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-4, "{a} != {b}");
    }
}
