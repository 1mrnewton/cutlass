//! Preview-viewport geometry: the canvas → viewport letterbox mapping, click
//! hit-testing, and the selection outline (preview roadmap Phase 2).
//!
//! The preview shows the composited frame aspect-fitted (`ImageFit.contain`)
//! inside the viewport element. Hit-testing inverts that mapping into canvas
//! pixels, then asks the engine's [`layer_placement`] — the same function the
//! compositor renders with — whether a layer's rotated quad contains the
//! point, walking lanes top-first (CapCut: the topmost layer under the cursor
//! wins). The selection box runs the mapping forward to outline the selected
//! clip's placement in viewport coordinates.

use cutlass_compositor::{CompositorConfig, LayerPlacement};
use cutlass_engine::layer_placement;
use cutlass_models::ClipTransform;
use slint::Model;

use crate::{Clip, PreviewDragResolution, PreviewHit, PreviewSelectionBox, Sequence, TrackKind};

/// Aspect-fit (`ImageFit.contain`) mapping of the canvas into the viewport:
/// `(scale, offset_x, offset_y)` such that `view = canvas · scale + offset`.
pub(crate) fn contain_mapping(
    canvas_w: f32,
    canvas_h: f32,
    view_w: f32,
    view_h: f32,
) -> (f32, f32, f32) {
    if canvas_w <= 0.0 || canvas_h <= 0.0 || view_w <= 0.0 || view_h <= 0.0 {
        return (1.0, 0.0, 0.0);
    }
    let scale = (view_w / canvas_w).min(view_h / canvas_h);
    (
        scale,
        (view_w - canvas_w * scale) / 2.0,
        (view_h - canvas_h * scale) / 2.0,
    )
}

pub(crate) fn canvas_config(sequence: &Sequence) -> CompositorConfig {
    CompositorConfig::new(
        sequence.width.max(1.0).round() as u32,
        sequence.height.max(1.0).round() as u32,
    )
}

/// Whether the composite path draws this clip at all: media, or a generator
/// the raster step supports. Sticker/effect/filter/adjustment clips aren't
/// composited yet, so they can't be picked (mirrors `resolve_layers`).
pub(crate) fn is_composited(clip: &Clip) -> bool {
    !clip.media_id.is_empty()
        || matches!(
            clip.generator_kind.as_str(),
            "text" | "solid" | "rect" | "ellipse"
        )
}

/// The clip's canvas placement, sized to its visible content.
///
/// Media clips use the shared engine helper (native size aspect-fit into the
/// canvas) — identical to what the compositor draws. Generators raster at
/// canvas size (fit 1:1) with their content centered inside, so the canvas
/// placement is computed the same way the compositor does and then the size
/// is shrunk to the drawn-content bounds the projection measured — the
/// selection box and hit-test hug the shape/text, not the transparent raster
/// (CapCut). Unknown bounds (0×0, e.g. empty text) keep the canvas-sized box.
pub(crate) fn clip_placement(clip: &Clip, canvas: &CompositorConfig) -> LayerPlacement {
    let transform = ClipTransform {
        position: [clip.transform_position_x, clip.transform_position_y],
        scale: clip.transform_scale,
        rotation: clip.transform_rotation,
        opacity: clip.transform_opacity,
    };
    let has_size = clip.media_width > 0 && clip.media_height > 0;
    if !clip.media_id.is_empty() {
        let (w, h) = if has_size {
            (clip.media_width as u32, clip.media_height as u32)
        } else {
            // Media that vanished from the pool: degrade to canvas size.
            (canvas.width, canvas.height)
        };
        return layer_placement(&transform, w, h, canvas);
    }
    let mut placement = layer_placement(&transform, canvas.width, canvas.height, canvas);
    if has_size {
        placement.size = [
            clip.media_width as f32 * transform.scale,
            clip.media_height as f32 * transform.scale,
        ];
    }
    placement
}

fn covers_tick(clip: &Clip, tick: i32) -> bool {
    let start = clip.timeline_start.value;
    let end = start.saturating_add(clip.source_range.duration.value);
    start <= tick && tick < end
}

/// Point-in-rotated-rect, both in canvas pixels. Inverts the compositor's
/// clockwise rotation `R = [cos, -sin; sin, cos]` (+y down) about the center.
fn placement_contains(p: &LayerPlacement, x: f32, y: f32) -> bool {
    let dx = x - p.center[0];
    let dy = y - p.center[1];
    let (sin, cos) = p.rotation.sin_cos();
    let local_x = dx * cos + dy * sin;
    let local_y = -dx * sin + dy * cos;
    local_x.abs() <= p.size[0] / 2.0 && local_y.abs() <= p.size[1] / 2.0
}

/// Topmost visible, unlocked clip under `(x, y)` (viewport-element logical
/// px) at `tick`. Lanes walk top-first; hidden lanes aren't composited and
/// locked lanes don't hit-test (same rule as timeline selection), both fall
/// through to the layer below. Empty `clip_id` ⇔ miss.
pub fn hit_test(
    sequence: &Sequence,
    tick: i32,
    x: f32,
    y: f32,
    view_w: f32,
    view_h: f32,
) -> PreviewHit {
    let canvas = canvas_config(sequence);
    let (cw, ch) = (canvas.width as f32, canvas.height as f32);
    let (scale, ox, oy) = contain_mapping(cw, ch, view_w, view_h);
    if scale <= 0.0 {
        return PreviewHit::default();
    }
    let px = (x - ox) / scale;
    let py = (y - oy) / scale;
    if px < 0.0 || py < 0.0 || px > cw || py > ch {
        return PreviewHit::default(); // letterbox bar
    }

    for row in 0..sequence.tracks.row_count() {
        let Some(track) = sequence.tracks.row_data(row) else {
            continue;
        };
        if track.kind == TrackKind::Audio || !track.enabled || track.locked {
            continue;
        }
        for idx in 0..track.clips.row_count() {
            let Some(mut clip) = track.clips.row_data(idx) else {
                continue;
            };
            if !covers_tick(&clip, tick) || !is_composited(&clip) {
                continue;
            }
            // Animated clips are picked where the playhead renders them.
            crate::params::apply_sampled_transform(&mut clip, tick);
            if placement_contains(&clip_placement(&clip, &canvas), px, py) {
                return PreviewHit {
                    track_id: track.id.clone(),
                    clip_id: clip.id.clone(),
                };
            }
        }
    }
    PreviewHit::default()
}

/// How far below the box's bottom edge the rotate affordance floats, in
/// viewport px (constant UI size regardless of zoom/letterbox — CapCut).
const ROTATE_HANDLE_OFFSET_PX: f32 = 26.0;

/// The placement's quad corners mapped into viewport coordinates, clockwise
/// from the content's top-left (rotation applied about the center).
fn placement_corners(p: &LayerPlacement, scale: f32, ox: f32, oy: f32) -> [[f32; 2]; 4] {
    let (sin, cos) = p.rotation.sin_cos();
    let (hw, hh) = (p.size[0] / 2.0, p.size[1] / 2.0);
    [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)].map(|(lx, ly)| {
        // Clockwise rotation in +y-down screen coords (same matrix as the
        // compositor's placement uniforms), then canvas → viewport.
        let x = p.center[0] + lx * cos - ly * sin;
        let y = p.center[1] + lx * sin + ly * cos;
        [ox + x * scale, oy + y * scale]
    })
}

/// Selection outline for `clip_id` in viewport-element coordinates.
/// Invisible when the id is empty/unknown, the clip isn't under the
/// playhead, or its lane is hidden — the layer has no pixels on screen.
///
/// During a transform gesture the projection still holds the press-time
/// transform (the live value is a worker-side override, by design), so the
/// panel passes the gesture's resolution to keep the box glued to the
/// content — position for moves, scale for corner drags, rotation for the
/// rotate affordance.
pub fn selection_box(
    sequence: &Sequence,
    clip_id: &str,
    tick: i32,
    view_w: f32,
    view_h: f32,
    gesture: Option<&PreviewDragResolution>,
) -> PreviewSelectionBox {
    if clip_id.is_empty() {
        return PreviewSelectionBox::default();
    }
    let canvas = canvas_config(sequence);
    let (cw, ch) = (canvas.width as f32, canvas.height as f32);
    let (scale, ox, oy) = contain_mapping(cw, ch, view_w, view_h);

    for row in 0..sequence.tracks.row_count() {
        let Some(track) = sequence.tracks.row_data(row) else {
            continue;
        };
        if track.kind == TrackKind::Audio || !track.enabled {
            continue;
        }
        for idx in 0..track.clips.row_count() {
            let Some(mut clip) = track.clips.row_data(idx) else {
                continue;
            };
            if clip.id != clip_id {
                continue;
            }
            if !covers_tick(&clip, tick) || !is_composited(&clip) {
                return PreviewSelectionBox::default();
            }
            // Box follows the rendered frame on animated clips; a live
            // gesture's resolution then wins (it previews via override).
            crate::params::apply_sampled_transform(&mut clip, tick);
            if let Some(res) = gesture {
                clip.transform_position_x = res.position_x;
                clip.transform_position_y = res.position_y;
                clip.transform_scale = res.scale;
                clip.transform_rotation = res.rotation;
            }
            let p = clip_placement(&clip, &canvas);
            let [c0, c1, c2, c3] = placement_corners(&p, scale, ox, oy);
            // Rotate affordance: floats a constant viewport distance below
            // the content's bottom edge (between c3 and c2), riding the
            // box's rotation. Outward = the edge direction rotated +90°
            // (y-down), which points away from the content for any angle.
            let mid = [(c2[0] + c3[0]) / 2.0, (c2[1] + c3[1]) / 2.0];
            let edge = [c2[0] - c3[0], c2[1] - c3[1]];
            let len = edge[0].hypot(edge[1]).max(f32::EPSILON);
            let out = [-edge[1] / len, edge[0] / len];
            return PreviewSelectionBox {
                visible: true,
                x0: c0[0],
                y0: c0[1],
                x1: c1[0],
                y1: c1[1],
                x2: c2[0],
                y2: c2[1],
                x3: c3[0],
                y3: c3[1],
                hx: mid[0] + out[0] * ROTATE_HANDLE_OFFSET_PX,
                hy: mid[1] + out[1] * ROTATE_HANDLE_OFFSET_PX,
            };
        }
    }
    PreviewSelectionBox::default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Rational, RationalTime, TimeRange, Track};
    use slint::{ModelRc, SharedString, VecModel};
    use std::rc::Rc;

    fn rt(value: i32) -> RationalTime {
        RationalTime {
            value,
            rate: Rational { num: 24, den: 1 },
        }
    }

    /// Media clip [start, start+dur) with native size `w×h` and an identity
    /// transform, overridable by the caller.
    fn media_clip(id: &str, start: i32, dur: i32, w: i32, h: i32) -> Clip {
        Clip {
            id: SharedString::from(id),
            name: SharedString::from(id),
            timeline_start: rt(start),
            source_range: TimeRange {
                start: rt(0),
                duration: rt(dur),
            },
            media_id: SharedString::from("m1"),
            media_width: w,
            media_height: h,
            transform_scale: 1.0,
            transform_opacity: 1.0,
            ..Default::default()
        }
    }

    fn track(id: &str, kind: TrackKind, clips: Vec<Clip>) -> Track {
        Track {
            id: SharedString::from(id),
            name: SharedString::from(id),
            kind,
            color: slint::Color::from_rgb_u8(0x4A, 0x6F, 0xA5),
            clips: ModelRc::from(Rc::new(VecModel::from(clips))),
            enabled: true,
            muted: false,
            locked: false,
        }
    }

    /// 1920×1080 canvas; tracks top-first like the projection publishes.
    fn sequence(tracks: Vec<Track>) -> Sequence {
        Sequence {
            id: SharedString::from("1"),
            name: SharedString::from("Sequence 1"),
            fps: Rational { num: 24, den: 1 },
            drop_frame: false,
            tracks: ModelRc::from(Rc::new(VecModel::from(tracks))),
            width: 1920.0,
            height: 1080.0,
        }
    }

    // Viewport at exactly half the canvas: scale 0.5, no letterbox.
    const VW: f32 = 960.0;
    const VH: f32 = 540.0;

    #[test]
    fn hit_picks_topmost_layer() {
        let seq = sequence(vec![
            track("2", TrackKind::Video, vec![media_clip("top", 0, 100, 1920, 1080)]),
            track("1", TrackKind::Video, vec![media_clip("bottom", 0, 100, 1920, 1080)]),
        ]);
        let hit = hit_test(&seq, 10, 480.0, 270.0, VW, VH);
        assert_eq!((hit.track_id.as_str(), hit.clip_id.as_str()), ("2", "top"));
    }

    #[test]
    fn hit_skips_locked_hidden_and_audio_lanes() {
        let mut locked = track("3", TrackKind::Video, vec![media_clip("locked", 0, 100, 1920, 1080)]);
        locked.locked = true;
        let mut hidden = track("2", TrackKind::Video, vec![media_clip("hidden", 0, 100, 1920, 1080)]);
        hidden.enabled = false;
        let seq = sequence(vec![
            locked,
            hidden,
            track("9", TrackKind::Audio, vec![media_clip("audio", 0, 100, 0, 0)]),
            track("1", TrackKind::Video, vec![media_clip("base", 0, 100, 1920, 1080)]),
        ]);
        let hit = hit_test(&seq, 10, 480.0, 270.0, VW, VH);
        assert_eq!(hit.clip_id.as_str(), "base");
    }

    #[test]
    fn hit_respects_playhead_coverage() {
        let seq = sequence(vec![track(
            "1",
            TrackKind::Video,
            vec![media_clip("A", 50, 50, 1920, 1080)],
        )]);
        assert_eq!(hit_test(&seq, 49, 480.0, 270.0, VW, VH).clip_id.as_str(), "");
        assert_eq!(hit_test(&seq, 50, 480.0, 270.0, VW, VH).clip_id.as_str(), "A");
        assert_eq!(hit_test(&seq, 99, 480.0, 270.0, VW, VH).clip_id.as_str(), "A");
        assert_eq!(hit_test(&seq, 100, 480.0, 270.0, VW, VH).clip_id.as_str(), "");
    }

    #[test]
    fn hit_misses_letterbox_bars() {
        let seq = sequence(vec![track(
            "1",
            TrackKind::Video,
            vec![media_clip("A", 0, 100, 1920, 1080)],
        )]);
        // Viewport wider than 16:9: content spans x ∈ [20, 980).
        let (vw, vh) = (1000.0, 540.0);
        assert_eq!(hit_test(&seq, 10, 10.0, 270.0, vw, vh).clip_id.as_str(), "");
        assert_eq!(hit_test(&seq, 10, 500.0, 270.0, vw, vh).clip_id.as_str(), "A");
        assert_eq!(hit_test(&seq, 10, 990.0, 270.0, vw, vh).clip_id.as_str(), "");
    }

    #[test]
    fn hit_honors_clip_transform() {
        // Half size, centered in the top-left quadrant: center (480, 270),
        // size 960×540 ⇒ canvas rect [0,960]×[0,540] ⇒ the viewport's
        // top-left quadrant at scale 0.5.
        let mut clip = media_clip("A", 0, 100, 1920, 1080);
        clip.transform_scale = 0.5;
        clip.transform_position_x = -0.25;
        clip.transform_position_y = -0.25;
        let seq = sequence(vec![track("1", TrackKind::Video, vec![clip])]);

        assert_eq!(hit_test(&seq, 10, 120.0, 67.0, VW, VH).clip_id.as_str(), "A");
        // Bottom-right quadrant of the viewport: empty canvas.
        assert_eq!(hit_test(&seq, 10, 720.0, 405.0, VW, VH).clip_id.as_str(), "");
    }

    #[test]
    fn hit_honors_rotation() {
        // Half-size centered quad (960×540 in canvas px), rotated 90°: its
        // long axis is now vertical, so a point 300 canvas px right of center
        // falls outside (270 half-height) while 300 px below falls inside.
        let mut clip = media_clip("A", 0, 100, 1920, 1080);
        clip.transform_scale = 0.5;
        clip.transform_rotation = 90.0;
        let seq = sequence(vec![track("1", TrackKind::Video, vec![clip])]);

        // canvas (1260, 540) → viewport (630, 270)
        assert_eq!(hit_test(&seq, 10, 630.0, 270.0, VW, VH).clip_id.as_str(), "");
        // canvas (960, 840) → viewport (480, 420)
        assert_eq!(hit_test(&seq, 10, 480.0, 420.0, VW, VH).clip_id.as_str(), "A");
    }

    /// Generated clip with measured content bounds `w×h` in canvas px
    /// (0×0 ⇔ unmeasured, falls back to a canvas-sized box).
    fn generated_clip(id: &str, kind: &str, w: i32, h: i32) -> Clip {
        let mut clip = media_clip(id, 0, 100, w, h);
        clip.media_id = SharedString::default();
        clip.generator_kind = SharedString::from(kind);
        clip
    }

    #[test]
    fn generators_without_bounds_hit_at_canvas_size() {
        let clip = generated_clip("T", "text", 0, 0);
        let mut sticker = generated_clip("S", "", 0, 0); // not composited yet
        sticker.generator_kind = SharedString::default();
        let seq = sequence(vec![
            track("2", TrackKind::Video, vec![sticker]),
            track("1", TrackKind::Video, vec![clip]),
        ]);
        let hit = hit_test(&seq, 10, 480.0, 270.0, VW, VH);
        assert_eq!(hit.clip_id.as_str(), "T", "sticker lane falls through");
    }

    #[test]
    fn generator_hit_uses_content_bounds() {
        // Content 960×540 centered on the 1920×1080 canvas spans
        // [480,1440]×[270,810] in canvas px — half that in the viewport.
        let seq = sequence(vec![track(
            "1",
            TrackKind::Video,
            vec![generated_clip("E", "ellipse", 960, 540)],
        )]);
        assert_eq!(hit_test(&seq, 10, 480.0, 270.0, VW, VH).clip_id.as_str(), "E");
        // Inside the canvas but outside the drawn content: falls through.
        assert_eq!(hit_test(&seq, 10, 100.0, 50.0, VW, VH).clip_id.as_str(), "");
    }

    #[test]
    fn generator_selection_box_hugs_content_bounds() {
        // Same 960×540 content: the box is content-sized about the canvas
        // center — at viewport scale 0.5, 480×270 centered at (480, 270).
        let seq = sequence(vec![track(
            "1",
            TrackKind::Video,
            vec![generated_clip("E", "ellipse", 960, 540)],
        )]);
        let b = selection_box(&seq, "E", 10, VW, VH, None);
        assert!(b.visible);
        assert_eq!(
            corners(&b),
            [(240.0, 135.0), (720.0, 135.0), (720.0, 405.0), (240.0, 405.0)]
        );
        assert_eq!((b.hx, b.hy), (480.0, 405.0 + 26.0));
    }

    #[test]
    fn generator_content_box_rides_the_transform_scale() {
        let mut clip = generated_clip("E", "ellipse", 960, 540);
        clip.transform_scale = 0.5;
        let seq = sequence(vec![track("1", TrackKind::Video, vec![clip])]);
        let b = selection_box(&seq, "E", 10, VW, VH, None);
        assert!(b.visible);
        assert_eq!(
            corners(&b),
            [(360.0, 202.5), (600.0, 202.5), (600.0, 337.5), (360.0, 337.5)]
        );
    }

    fn corners(b: &PreviewSelectionBox) -> [(f32, f32); 4] {
        [(b.x0, b.y0), (b.x1, b.y1), (b.x2, b.y2), (b.x3, b.y3)]
    }

    #[test]
    fn selection_box_maps_placement_to_viewport() {
        // 960×1080 media on a 1920×1080 canvas: aspect-fit 1.0 ⇒ a centered
        // 960×1080 pillarboxed rect; at viewport scale 0.5 the box is
        // 480×540 centered at (480, 270).
        let seq = sequence(vec![track(
            "1",
            TrackKind::Video,
            vec![media_clip("A", 0, 100, 960, 1080)],
        )]);
        let b = selection_box(&seq, "A", 10, VW, VH, None);
        assert!(b.visible);
        assert_eq!(
            corners(&b),
            [(240.0, 0.0), (720.0, 0.0), (720.0, 540.0), (240.0, 540.0)]
        );
        // Rotate affordance: a constant offset below the bottom edge's
        // midpoint (480, 540).
        assert_eq!((b.hx, b.hy), (480.0, 540.0 + 26.0));
    }

    #[test]
    fn selection_box_rotates_corners() {
        // Half-size centered quad rotated 90° cw: the content's top-left
        // corner (canvas Δ(-480, -270)) lands at Δ(270, -480) from center —
        // canvas (1230, 60), viewport (615, 30).
        let mut clip = media_clip("A", 0, 100, 1920, 1080);
        clip.transform_scale = 0.5;
        clip.transform_rotation = 90.0;
        let seq = sequence(vec![track("1", TrackKind::Video, vec![clip])]);
        let b = selection_box(&seq, "A", 10, VW, VH, None);
        assert!(b.visible);
        let [c0, c1, c2, c3] = corners(&b);
        let expect = [(615.0, 30.0), (615.0, 510.0), (345.0, 510.0), (345.0, 30.0)];
        for ((x, y), (ex, ey)) in [c0, c1, c2, c3].into_iter().zip(expect) {
            assert!((x - ex).abs() < 1e-3 && (y - ey).abs() < 1e-3, "({x},{y}) vs ({ex},{ey})");
        }
        // The rotate affordance rides the rotation: 90° cw points the
        // content's bottom edge left, so the handle sits left of the box.
        assert!((b.hx - (345.0 - 26.0)).abs() < 1e-3 && (b.hy - 270.0).abs() < 1e-3);
    }

    #[test]
    fn selection_geometry_follows_the_playhead_sample() {
        use crate::ParamKeyframe;
        let mut clip = media_clip("A", 0, 100, 1920, 1080);
        // Scale animates 1.0 → 0.5 over ticks 0..40.
        clip.kf_scale = ModelRc::from(Rc::new(VecModel::from(vec![
            ParamKeyframe { tick: 0, value_x: 1.0, ..Default::default() },
            ParamKeyframe { tick: 40, value_x: 0.5, ..Default::default() },
        ])));
        let seq = sequence(vec![track("1", TrackKind::Video, vec![clip])]);

        // At the last keyframe the content is half-size: the box shrinks to
        // the centered 480×270 quad, and a click near the viewport corner
        // (inside at tick 0) now misses.
        let b = selection_box(&seq, "A", 40, VW, VH, None);
        assert_eq!(
            corners(&b),
            [(240.0, 135.0), (720.0, 135.0), (720.0, 405.0), (240.0, 405.0)]
        );
        assert_eq!(hit_test(&seq, 0, 100.0, 60.0, VW, VH).clip_id.as_str(), "A");
        assert_eq!(hit_test(&seq, 40, 100.0, 60.0, VW, VH).clip_id.as_str(), "");
    }

    #[test]
    fn selection_box_hides_when_off_playhead_or_hidden() {
        let mut hidden = track("2", TrackKind::Video, vec![media_clip("H", 0, 100, 1920, 1080)]);
        hidden.enabled = false;
        let seq = sequence(vec![
            hidden,
            track("1", TrackKind::Video, vec![media_clip("A", 50, 50, 1920, 1080)]),
        ]);
        assert!(!selection_box(&seq, "", 60, VW, VH, None).visible);
        assert!(!selection_box(&seq, "A", 10, VW, VH, None).visible, "off playhead");
        assert!(selection_box(&seq, "A", 60, VW, VH, None).visible);
        assert!(!selection_box(&seq, "H", 60, VW, VH, None).visible, "hidden lane");
        assert!(!selection_box(&seq, "404", 60, VW, VH, None).visible);
    }
}
