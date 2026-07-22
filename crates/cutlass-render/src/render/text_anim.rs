//! Per-character text animation: sample a look preset into one transform /
//! opacity delta per shaping cluster.
//!
//! Clusters come from [`cutlass_text::TextRenderer::shape`] in logical order
//! (line, then byte position) — the same order a typewriter should reveal,
//! including for RTL. Stagger timing is driven by that index.

use cutlass_compositor::GlyphInstance;
use cutlass_models::AnimationSlot;
use cutlass_text::{ClusterBox, ShapedText, TextStyle};

use crate::scene::TextAnimation;

/// Multiplicative / additive delta applied to one cluster's rest placement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ClusterDelta {
    /// Offset from rest center, in *run* pixels (scaled later).
    pub position: [f32; 2],
    pub scale: f32,
    /// Extra clockwise rotation in radians.
    pub rotation: f32,
    pub opacity: f32,
}

impl ClusterDelta {
    pub const IDENTITY: Self = Self {
        position: [0.0, 0.0],
        scale: 1.0,
        rotation: 0.0,
        opacity: 1.0,
    };

    fn with_intensity(self, intensity: f32) -> Self {
        let i = intensity.clamp(0.0, 2.0);
        Self {
            position: [self.position[0] * i, self.position[1] * i],
            scale: 1.0 + (self.scale - 1.0) * i,
            rotation: self.rotation * i,
            opacity: 1.0 + (self.opacity - 1.0) * i,
        }
    }
}

/// Compute per-cluster deltas for `anim` over `shaped`.
pub(super) fn cluster_deltas(shaped: &ShapedText, anim: &TextAnimation) -> Vec<ClusterDelta> {
    let n = shaped.clusters.len().max(1) as f32;
    shaped
        .clusters
        .iter()
        .enumerate()
        .map(|(i, cluster)| {
            let stagger = i as f32 / n;
            sample_cluster(anim, stagger, cluster.line, i)
        })
        .collect()
}

fn sample_cluster(anim: &TextAnimation, stagger: f32, _line: usize, index: usize) -> ClusterDelta {
    let t = anim.t.clamp(0.0, 1.0);
    // Catalog stagger window 0.55, stretched by the clip's stagger knob.
    let window = (0.55 * anim.stagger.clamp(0.05, 2.0)).clamp(0.05, 0.95);
    let delta = match anim.slot {
        AnimationSlot::In => sample_entrance(&anim.id, t, stagger, window),
        AnimationSlot::Out => sample_exit(&anim.id, t, stagger, window),
        AnimationSlot::Combo => sample_combo(&anim.id, t, stagger, index, window),
    };
    delta.with_intensity(anim.intensity)
}

/// Map global progress `t` and per-cluster `stagger` ∈ [0,1) into a local
/// 0…1 progress that rises earlier for lower-index clusters.
fn stagger_progress(t: f32, stagger: f32, window: f32) -> f32 {
    let window = window.clamp(0.05, 0.95);
    let start = stagger * (1.0 - window);
    ((t - start) / window).clamp(0.0, 1.0)
}

fn sample_entrance(id: &str, t: f32, stagger: f32, window: f32) -> ClusterDelta {
    let local = stagger_progress(t, stagger, window);
    let inv = 1.0 - local;
    match id {
        "typewriter" | "char_typewriter" => ClusterDelta {
            opacity: if local > 0.0 { 1.0 } else { 0.0 },
            ..ClusterDelta::IDENTITY
        },
        "char_fade_in" => ClusterDelta {
            opacity: local,
            ..ClusterDelta::IDENTITY
        },
        "char_bounce_in" => ClusterDelta {
            position: [0.0, inv * 18.0],
            scale: 0.4 + 0.6 * bounce_out(local),
            opacity: local,
            ..ClusterDelta::IDENTITY
        },
        "char_slide_in" => ClusterDelta {
            position: [inv * 24.0, 0.0],
            opacity: local,
            ..ClusterDelta::IDENTITY
        },
        "char_pop_in" => ClusterDelta {
            scale: 0.2 + 0.8 * local,
            opacity: local,
            ..ClusterDelta::IDENTITY
        },
        _ => ClusterDelta::IDENTITY,
    }
}

fn sample_exit(id: &str, t: f32, stagger: f32, window: f32) -> ClusterDelta {
    // Exit staggers in reverse so the last character leaves first.
    let local = stagger_progress(t, 1.0 - stagger, window);
    let inv = 1.0 - local;
    match id {
        "char_fade_out" => ClusterDelta {
            opacity: inv,
            ..ClusterDelta::IDENTITY
        },
        "char_fall_away" => ClusterDelta {
            position: [0.0, local * 28.0],
            scale: 1.0 - 0.35 * local,
            opacity: inv,
            ..ClusterDelta::IDENTITY
        },
        "char_typewriter_out" => ClusterDelta {
            opacity: if local < 1.0 { 1.0 } else { 0.0 },
            ..ClusterDelta::IDENTITY
        },
        _ => ClusterDelta::IDENTITY,
    }
}

fn sample_combo(id: &str, phase: f32, stagger: f32, index: usize, window: f32) -> ClusterDelta {
    let phase = phase.fract();
    let wave = ((phase + stagger) * std::f32::consts::TAU).sin();
    match id {
        "typewriter" => {
            // Looping reveal: phase 0..0.85 types in, then holds / resets.
            let reveal = (phase / 0.85).clamp(0.0, 1.0);
            let local = stagger_progress(reveal, stagger, (window * 0.4 / 0.55).clamp(0.05, 0.95));
            ClusterDelta {
                opacity: if local > 0.0 { 1.0 } else { 0.0 },
                ..ClusterDelta::IDENTITY
            }
        }
        "text_fade" => ClusterDelta {
            opacity: 0.55
                + 0.45 * ((phase * std::f32::consts::PI + stagger * 2.0).sin() * 0.5 + 0.5),
            ..ClusterDelta::IDENTITY
        },
        "text_bounce" => ClusterDelta {
            position: [0.0, wave.abs() * 6.0],
            scale: 1.0 + 0.08 * wave.abs(),
            ..ClusterDelta::IDENTITY
        },
        "text_slide" => ClusterDelta {
            position: [wave * 5.0, 0.0],
            ..ClusterDelta::IDENTITY
        },
        "pop" => {
            let pulse = (phase * std::f32::consts::PI + stagger * std::f32::consts::PI)
                .sin()
                .max(0.0);
            ClusterDelta {
                scale: 1.0 + 0.18 * pulse,
                ..ClusterDelta::IDENTITY
            }
        }
        "wave" => ClusterDelta {
            position: [0.0, wave * 8.0],
            rotation: wave * 0.12,
            ..ClusterDelta::IDENTITY
        },
        "char_jitter" => {
            // Deterministic tiny noise from index + phase buckets.
            let bucket = (phase * 12.0).floor();
            let seed = (index as f32 * 12.9898 + bucket * 78.233).sin() * 43_758.547;
            let nx = seed.fract() * 2.0 - 1.0;
            let ny = ((seed * 1.37).fract()) * 2.0 - 1.0;
            ClusterDelta {
                position: [nx * 2.5, ny * 2.5],
                rotation: nx * 0.05,
                ..ClusterDelta::IDENTITY
            }
        }
        "char_pulse" => ClusterDelta {
            scale: 1.0 + 0.12 * ((phase * std::f32::consts::TAU + stagger * 3.0).sin()),
            ..ClusterDelta::IDENTITY
        },
        _ => ClusterDelta::IDENTITY,
    }
}

fn bounce_out(t: f32) -> f32 {
    let t = t as f64;
    if t < 1.0 / 2.75 {
        (7.5625 * t * t) as f32
    } else if t < 2.0 / 2.75 {
        let t = t - 1.5 / 2.75;
        (7.5625 * t * t + 0.75) as f32
    } else if t < 2.5 / 2.75 {
        let t = t - 2.25 / 2.75;
        (7.5625 * t * t + 0.9375) as f32
    } else {
        let t = t - 2.625 / 2.75;
        (7.5625 * t * t + 0.984375) as f32
    }
}

/// Place shaped clusters on the canvas with per-cluster deltas applied.
///
/// `origin` is the top-left of the shaped extent on the canvas (after text
/// alignment). `scale` is the bitmap scale factor. `layer_rotation` /
/// `layer_opacity` are the clip's whole-layer transform (already excluding
/// per-character look presets).
pub(super) fn place_clusters(
    shaped: &ShapedText,
    deltas: &[ClusterDelta],
    origin: [f32; 2],
    scale: f32,
    layer_rotation: f32,
    layer_opacity: f32,
) -> Vec<GlyphInstance> {
    shaped
        .clusters
        .iter()
        .zip(deltas.iter())
        .enumerate()
        .filter(|(_, (c, _))| c.image.width > 0 && c.image.height > 0)
        .map(|(i, (cluster, delta))| {
            instance_for_cluster(
                i as u32,
                cluster,
                delta,
                origin,
                scale,
                layer_rotation,
                layer_opacity,
            )
        })
        .collect()
}

fn instance_for_cluster(
    glyph: u32,
    cluster: &ClusterBox,
    delta: &ClusterDelta,
    origin: [f32; 2],
    scale: f32,
    layer_rotation: f32,
    layer_opacity: f32,
) -> GlyphInstance {
    let rest_size = [
        cluster.image.width as f32 * scale,
        cluster.image.height as f32 * scale,
    ];
    let size = [rest_size[0] * delta.scale, rest_size[1] * delta.scale];
    // Rest center of the glyph quad.
    let rest_center = [
        origin[0] + cluster.offset[0] * scale + rest_size[0] * 0.5,
        origin[1] + cluster.offset[1] * scale + rest_size[1] * 0.5,
    ];
    // Anchor rise/drop about the baseline for natural motion.
    let baseline_y = origin[1] + cluster.baseline * scale;
    let anchor = [rest_center[0], baseline_y];
    let from_anchor = [rest_center[0] - anchor[0], rest_center[1] - anchor[1]];
    let scaled_from = [from_anchor[0] * delta.scale, from_anchor[1] * delta.scale];
    let center = [
        anchor[0] + scaled_from[0] + delta.position[0] * scale,
        anchor[1] + scaled_from[1] + delta.position[1] * scale,
    ];
    GlyphInstance {
        glyph,
        center,
        size,
        rotation: layer_rotation + delta.rotation,
        opacity: (layer_opacity * delta.opacity).clamp(0.0, 1.0),
    }
}

/// Top-left canvas origin for an ink-tight run given the layer's aligned
/// quad center (from [`crate::scene::SceneLayer::text_quad_center`]).
pub(super) fn extent_origin(aligned_center: [f32; 2], extent: (u32, u32), scale: f32) -> [f32; 2] {
    [
        aligned_center[0] - extent.0 as f32 * scale * 0.5,
        aligned_center[1] - extent.1 as f32 * scale * 0.5,
    ]
}

/// Stable atlas cache key for a text run + style (shape-affecting fields).
pub(super) fn atlas_key(content: &str, style: &TextStyle) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    content.hash(&mut h);
    style.font_size.to_bits().hash(&mut h);
    style.line_height.to_bits().hash(&mut h);
    style.color.hash(&mut h);
    style.family.hash(&mut h);
    style.bold.hash(&mut h);
    style.italic.hash(&mut h);
    style.letter_spacing.to_bits().hash(&mut h);
    style.align.hash(&mut h);
    style.max_width.map(f32::to_bits).unwrap_or(0).hash(&mut h);
    // Effects folded into cluster images change atlas contents.
    style.underline.hash(&mut h);
    style
        .stroke
        .map(|s| (s.rgba, s.width.to_bits()))
        .hash(&mut h);
    style
        .shadow
        .map(|s| (s.rgba, s.blur.to_bits(), s.distance.to_bits()))
        .hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutlass_models::AnimationSlot;

    fn anim(id: &str, slot: AnimationSlot, t: f32) -> TextAnimation {
        TextAnimation {
            id: id.into(),
            slot,
            t,
            intensity: 1.0,
            stagger: 1.0,
        }
    }

    #[test]
    fn typewriter_reveals_in_order() {
        let shaped = ShapedText {
            extent: (40, 20),
            clusters: (0..4)
                .map(|i| ClusterBox {
                    text_range: i..i + 1,
                    line: 0,
                    offset: [i as f32 * 10.0, 0.0],
                    baseline: 16.0,
                    image: cutlass_core::RgbaImage::new(8, 12, vec![255; 8 * 12 * 4]),
                })
                .collect(),
        };
        // Mid-reveal: earlier clusters visible, later ones not.
        let deltas = cluster_deltas(&shaped, &anim("typewriter", AnimationSlot::Combo, 0.3));
        assert!(deltas[0].opacity > 0.5);
        assert!(deltas[3].opacity < 0.5);
    }

    #[test]
    fn wave_offsets_vary_by_index() {
        let shaped = ShapedText {
            extent: (30, 10),
            clusters: (0..3)
                .map(|i| ClusterBox {
                    text_range: i..i + 1,
                    line: 0,
                    offset: [i as f32 * 10.0, 0.0],
                    baseline: 8.0,
                    image: cutlass_core::RgbaImage::new(6, 8, vec![255; 6 * 8 * 4]),
                })
                .collect(),
        };
        let deltas = cluster_deltas(&shaped, &anim("wave", AnimationSlot::Combo, 0.1));
        assert!(
            (deltas[0].position[1] - deltas[1].position[1]).abs() > 0.01
                || (deltas[0].rotation - deltas[1].rotation).abs() > 0.001,
            "wave should differentiate neighboring clusters"
        );
    }
}
