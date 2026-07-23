//! Transform-only motion blur via temporal supersampling.
//!
//! When a clip has [`MotionBlur`](cutlass_models::MotionBlur) enabled and an
//! animated transform, export re-samples the clip's placement across the
//! shutter interval and the compositor averages those placements into one
//! premultiplied layer before the normal stack composite.
//!
//! ## Approximation vs true shutter integration
//!
//! Averaging happens **pre-composite** in the layer's own offscreen: edges that
//! reveal the background blend against the layer's accumulated alpha, not
//! against whatever sits below in the stack. Acceptable for CapCut-style
//! transform motion blur; whole-frame shutter integration is out of scope.
//!
//! ## Transform-only (not source-motion)
//!
//! The layer's texture stays the frame-at-T content — we do **not** re-decode
//! video at sub-ticks. This matches After Effects' transform motion blur;
//! source-motion / optical-flow blur is out of scope.
//!
//! ## Cost
//!
//! Per blurred layer: **N** extra placement draws + N weighted additive blits
//! (N clamped to `2..=16`), then one normal composite of the average. Layers
//! without motion blur pay nothing.

use cutlass_compositor::LayerPlacement;
use cutlass_core::RationalTime;
use cutlass_models::{Clip, ClipId, MotionBlur, Project};

use crate::animation::apply_look_animations;
use crate::scene::{LayerSource, Scene, SceneBlurPass, SizeSpec};

/// Populate [`SceneLayer::blur_passes`](crate::scene::SceneLayer::blur_passes)
/// for every clip that opts into motion blur with an animated transform.
///
/// Call **before** [`Scene::fit_within`] / [`Scene::fit_into`] so scale and
/// letterbox adjust the sample placements with the rest of the scene.
/// [`SizeSpec::BitmapScaled`] layers are skipped here (bitmap size is unknown
/// until realize) and filled later via [`blur_passes_for_placement`].
///
/// `motion_blur_override` (inspector shutter/samples drag) wins for its clip
/// over the committed [`Clip::motion_blur`] — session-only, like styles/look.
pub fn attach_motion_blur_passes(
    project: &Project,
    scene: &mut Scene,
    motion_blur_override: Option<(ClipId, MotionBlur)>,
) {
    let tick = scene.tick;
    let cw = scene.width as f32;
    let ch = scene.height as f32;
    for layer in &mut scene.layers {
        if matches!(
            layer.source,
            LayerSource::CanvasPass | LayerSource::Transition { .. }
        ) {
            continue;
        }
        let SizeSpec::Fixed(base_size) = layer.size else {
            continue;
        };
        let Some(clip_id) = layer.clip else {
            continue;
        };
        let Some(clip) = project.clip(clip_id) else {
            continue;
        };
        let mb = match motion_blur_override {
            Some((id, blur)) if id == clip_id => blur,
            _ => clip.motion_blur,
        };
        layer.blur_passes = sample_blur_passes(clip, mb, tick, cw, ch, base_size);
    }
}

/// Compute blur sample placements once the realized pixel size is known
/// (text / path bitmaps). Returns an empty vec when blur does not apply.
pub fn blur_passes_for_placement(
    project: &Project,
    scene: &Scene,
    clip_id: cutlass_models::ClipId,
    base: LayerPlacement,
    anchor_point: [f32; 2],
) -> Vec<LayerPlacement> {
    let Some(clip) = project.clip(clip_id) else {
        return Vec::new();
    };
    let cw = scene.width as f32;
    let ch = scene.height as f32;
    let passes = sample_blur_passes(clip, clip.motion_blur, scene.tick, cw, ch, base.size);
    passes
        .into_iter()
        .map(|p| placement_from_blur_pass(&p, anchor_point))
        .collect()
}

/// Convert scene-space blur passes (anchor `center`) into compositor
/// placements (quad center), using the clip's anchor point.
pub fn layer_placements_from_blur_passes(
    passes: &[SceneBlurPass],
    anchor_point: [f32; 2],
) -> Vec<LayerPlacement> {
    passes
        .iter()
        .map(|p| placement_from_blur_pass(p, anchor_point))
        .collect()
}

fn sample_blur_passes(
    clip: &Clip,
    mb: MotionBlur,
    t: RationalTime,
    cw: f32,
    ch: f32,
    base_size: [f32; 2],
) -> Vec<SceneBlurPass> {
    if !mb.enabled || mb.shutter_deg <= 0.0 || !clip.transform.is_animated() {
        return Vec::new();
    }
    let n = mb.samples.clamp(2, 16);
    // Export steps one timeline tick per output frame at the sampling rate.
    let shutter_span = f64::from(mb.shutter_deg) / 360.0;

    let local0 = clip.animation_tick_f(t.value as f64);
    let xf0 = apply_look_animations(
        clip,
        clip.transform.sample_at(local0),
        local0.round() as i64,
        local0,
        t.rate,
    );
    let sx = xf0.scale.x.abs().max(1e-6);
    let sy = xf0.scale.y.abs().max(1e-6);
    let natural = [base_size[0] / sx, base_size[1] / sy];

    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n {
        let u = i as f64 / f64::from(n - 1);
        let sample_t = t.value as f64 + shutter_span * (u - 0.5);
        let local = clip.animation_tick_f(sample_t);
        let xf = apply_look_animations(
            clip,
            clip.transform.sample_at(local),
            local.round() as i64,
            local,
            t.rate,
        );
        let size = [natural[0] * xf.scale.x, natural[1] * xf.scale.y];
        out.push(SceneBlurPass {
            center: [cw * (0.5 + xf.position[0]), ch * (0.5 + xf.position[1])],
            size,
            rotation: xf.rotation.to_radians(),
            opacity: xf.opacity.clamp(0.0, 1.0),
        });
    }
    out
}

fn placement_from_blur_pass(pass: &SceneBlurPass, anchor_point: [f32; 2]) -> LayerPlacement {
    let to_center = [
        (0.5 - anchor_point[0]) * pass.size[0],
        (0.5 - anchor_point[1]) * pass.size[1],
    ];
    let center = if to_center == [0.0, 0.0] {
        pass.center
    } else {
        let (sin, cos) = pass.rotation.sin_cos();
        [
            pass.center[0] + to_center[0] * cos - to_center[1] * sin,
            pass.center[1] + to_center[0] * sin + to_center[1] * cos,
        ]
    };
    LayerPlacement {
        center,
        size: pass.size,
        rotation: pass.rotation,
        opacity: pass.opacity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_models::{
        ClipParam, Easing, Generator, MotionBlur, ParamValue, Project, Rational, RationalTime,
        TimeRange, TrackKind,
    };

    const FPS: Rational = Rational::FPS_30;

    #[test]
    fn animated_clip_with_blur_gets_subframe_passes() {
        let mut project = Project::new("mb", FPS);
        let track = project.add_track(TrackKind::Sticker, "S1");
        let clip = project
            .add_generated(
                track,
                Generator::SolidColor {
                    rgba: [255, 255, 255, 255],
                },
                TimeRange::at_rate(0, 30, FPS),
            )
            .unwrap();
        project
            .set_param_keyframe(
                clip,
                ClipParam::Position,
                RationalTime::new(0, FPS),
                ParamValue::Vec2([-0.4, 0.0]),
                Easing::Linear,
                None,
            )
            .unwrap();
        project
            .set_param_keyframe(
                clip,
                ClipParam::Position,
                RationalTime::new(29, FPS),
                ParamValue::Vec2([0.4, 0.0]),
                Easing::Linear,
                None,
            )
            .unwrap();
        project
            .set_motion_blur(
                clip,
                MotionBlur {
                    enabled: true,
                    shutter_deg: 180.0,
                    samples: 4,
                },
            )
            .unwrap();

        let mut scene = crate::resolve::resolve(&project, RationalTime::new(15, FPS)).unwrap();
        attach_motion_blur_passes(&project, &mut scene, None);
        let layer = scene.layers.iter().find(|l| l.clip == Some(clip)).unwrap();
        assert_eq!(layer.blur_passes.len(), 4);
        // Horizontal sweep: first sample left of last.
        assert!(layer.blur_passes[0].center[0] < layer.blur_passes[3].center[0]);
    }

    #[test]
    fn static_transform_yields_no_passes() {
        let mut project = Project::new("mb", FPS);
        let track = project.add_track(TrackKind::Sticker, "S1");
        let clip = project
            .add_generated(
                track,
                Generator::SolidColor {
                    rgba: [255, 255, 255, 255],
                },
                TimeRange::at_rate(0, 30, FPS),
            )
            .unwrap();
        project
            .set_motion_blur(
                clip,
                MotionBlur {
                    enabled: true,
                    shutter_deg: 180.0,
                    samples: 8,
                },
            )
            .unwrap();
        let mut scene = crate::resolve::resolve(&project, RationalTime::new(10, FPS)).unwrap();
        attach_motion_blur_passes(&project, &mut scene, None);
        let layer = scene.layers.iter().find(|l| l.clip == Some(clip)).unwrap();
        assert!(layer.blur_passes.is_empty());
    }
}
