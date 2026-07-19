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
        transform_anchor_x: 0.5,
        transform_anchor_y: 0.5,
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
        duck_source: false,
        pinned: false,
        is_main: false,
        transitions: ModelRc::default(),
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
        markers: Default::default(),
        width: 1920.0,
        height: 1080.0,
        aspect_index: 0,
        background: Default::default(),
    }
}

// Viewport at exactly half the canvas: scale 0.5, no letterbox.
const VW: f32 = 960.0;
const VH: f32 = 540.0;

#[test]
fn hit_picks_topmost_layer() {
    let seq = sequence(vec![
        track(
            "2",
            TrackKind::Video,
            vec![media_clip("top", 0, 100, 1920, 1080)],
        ),
        track(
            "1",
            TrackKind::Video,
            vec![media_clip("bottom", 0, 100, 1920, 1080)],
        ),
    ]);
    let hit = hit_test(&seq, 10, 480.0, 270.0, VW, VH);
    assert_eq!((hit.track_id.as_str(), hit.clip_id.as_str()), ("2", "top"));
}

#[test]
fn hit_skips_locked_hidden_and_audio_lanes() {
    let mut locked = track(
        "3",
        TrackKind::Video,
        vec![media_clip("locked", 0, 100, 1920, 1080)],
    );
    locked.locked = true;
    let mut hidden = track(
        "2",
        TrackKind::Video,
        vec![media_clip("hidden", 0, 100, 1920, 1080)],
    );
    hidden.enabled = false;
    let seq = sequence(vec![
        locked,
        hidden,
        track(
            "9",
            TrackKind::Audio,
            vec![media_clip("audio", 0, 100, 0, 0)],
        ),
        track(
            "1",
            TrackKind::Video,
            vec![media_clip("base", 0, 100, 1920, 1080)],
        ),
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
    assert_eq!(
        hit_test(&seq, 49, 480.0, 270.0, VW, VH).clip_id.as_str(),
        ""
    );
    assert_eq!(
        hit_test(&seq, 50, 480.0, 270.0, VW, VH).clip_id.as_str(),
        "A"
    );
    assert_eq!(
        hit_test(&seq, 99, 480.0, 270.0, VW, VH).clip_id.as_str(),
        "A"
    );
    assert_eq!(
        hit_test(&seq, 100, 480.0, 270.0, VW, VH).clip_id.as_str(),
        ""
    );
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
    assert_eq!(
        hit_test(&seq, 10, 500.0, 270.0, vw, vh).clip_id.as_str(),
        "A"
    );
    assert_eq!(
        hit_test(&seq, 10, 990.0, 270.0, vw, vh).clip_id.as_str(),
        ""
    );
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

    assert_eq!(
        hit_test(&seq, 10, 120.0, 67.0, VW, VH).clip_id.as_str(),
        "A"
    );
    // Bottom-right quadrant of the viewport: empty canvas.
    assert_eq!(
        hit_test(&seq, 10, 720.0, 405.0, VW, VH).clip_id.as_str(),
        ""
    );
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
    assert_eq!(
        hit_test(&seq, 10, 630.0, 270.0, VW, VH).clip_id.as_str(),
        ""
    );
    // canvas (960, 840) → viewport (480, 420)
    assert_eq!(
        hit_test(&seq, 10, 480.0, 420.0, VW, VH).clip_id.as_str(),
        "A"
    );
}

/// Generated clip with measured content bounds `w×h` in canvas px
/// (0×0 ⇔ unmeasured, falls back to a canvas-sized box).
fn generated_clip(id: &str, kind: &str, w: i32, h: i32) -> Clip {
    let mut clip = media_clip(id, 0, 100, w, h);
    clip.media_id = SharedString::default();
    clip.generator_kind = SharedString::from(kind);
    if kind == "text" {
        // Engine TextStyle defaults: centered horizontally and vertically.
        clip.text_style.align_h = 1;
        clip.text_style.align_v = 1;
    }
    clip
}

#[test]
fn generators_without_bounds_hit_at_canvas_size() {
    let clip = generated_clip("T", "text", 0, 0);
    // A legacy empty-asset sticker: draws nothing, so no composited tag.
    let mut sticker = generated_clip("S", "", 0, 0);
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
    assert_eq!(
        hit_test(&seq, 10, 480.0, 270.0, VW, VH).clip_id.as_str(),
        "E"
    );
    // Inside the canvas but outside the drawn content: falls through.
    assert_eq!(hit_test(&seq, 10, 100.0, 50.0, VW, VH).clip_id.as_str(), "");
}

#[test]
fn selected_overflow_clip_is_grabbable_in_the_letterbox() {
    // A title wider than the canvas (4000 px content on a 1920 canvas):
    // its raster overflows the frame into the letterbox bars. In a
    // viewport wider than 16:9 (scale 0.5, left bar x ∈ [0, 20)), a press
    // at x = 8 lands in the bar — the normal hit-test deselects there, but
    // the selected clip's box still covers it, so it stays grabbable.
    let (vw, vh) = (1000.0, 540.0);
    let seq = sequence(vec![track(
        "1",
        TrackKind::Video,
        vec![generated_clip("T", "text", 4000, 1080)],
    )]);
    let contains = |id: &str, tick: i32, x: f32, y: f32| {
        selected_clip_contains_in_viewport(&seq, id, tick, x, y, vw, vh, 1.0, 0.0, 0.0)
    };

    // Normal hit-test misses out in the letterbox bar...
    assert_eq!(hit_test(&seq, 10, 8.0, 270.0, vw, vh).clip_id.as_str(), "");
    // ...but the selected overflowing clip is still grabbable there.
    assert!(contains("T", 10, 8.0, 270.0));
    // Off the playhead it isn't draggable, so it isn't grabbable.
    assert!(!contains("T", 200, 8.0, 270.0));
    // No selection ⇒ nothing to grab.
    assert!(!contains("", 10, 8.0, 270.0));
}

#[test]
fn canvas_sized_clip_stays_deselectable_in_the_letterbox() {
    // A clip that exactly fills the canvas doesn't reach the letterbox, so
    // a press out there is genuine empty space: not grabbable (the caller
    // deselects, CapCut-style).
    let (vw, vh) = (1000.0, 540.0);
    let seq = sequence(vec![track(
        "1",
        TrackKind::Video,
        vec![generated_clip("C", "text", 1920, 1080)],
    )]);
    assert!(!selected_clip_contains_in_viewport(
        &seq, "C", 10, 8.0, 270.0, vw, vh, 1.0, 0.0, 0.0
    ));
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
        [
            (240.0, 135.0),
            (720.0, 135.0),
            (720.0, 405.0),
            (240.0, 405.0)
        ]
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
        [
            (360.0, 202.5),
            (600.0, 202.5),
            (600.0, 337.5),
            (360.0, 337.5)
        ]
    );
}

#[test]
fn text_selection_box_follows_canvas_alignment() {
    let mut clip = generated_clip("T", "text", 400, 200);
    clip.text_style.align_h = 0;
    clip.text_style.align_v = 2;
    let seq = sequence(vec![track("1", TrackKind::Text, vec![clip])]);
    let b = selection_box(&seq, "T", 10, VW, VH, None);
    assert!(b.visible);
    // Canvas 1920x1080 is shown at 0.5x: the 400x200 text box hugs the
    // bottom-left corner and remains ink-tight (200x100 in the viewport).
    assert_eq!(
        corners(&b),
        [(0.0, 440.0), (200.0, 440.0), (200.0, 540.0), (0.0, 540.0)]
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
fn selection_box_follows_preview_zoom_and_pan() {
    let seq = sequence(vec![track(
        "1",
        TrackKind::Video,
        vec![media_clip("A", 0, 100, 1920, 1080)],
    )]);
    let b = selection_box_in_viewport(&seq, "A", 10, VW, VH, 2.0, 20.0, -10.0, None);
    assert!(b.visible);
    assert_eq!(
        corners(&b),
        [
            (-460.0, -280.0),
            (1460.0, -280.0),
            (1460.0, 800.0),
            (-460.0, 800.0)
        ]
    );
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
        assert!(
            (x - ex).abs() < 1e-3 && (y - ey).abs() < 1e-3,
            "({x},{y}) vs ({ex},{ey})"
        );
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
        ParamKeyframe {
            tick: 0,
            value_x: 1.0,
            ..Default::default()
        },
        ParamKeyframe {
            tick: 40,
            value_x: 0.5,
            ..Default::default()
        },
    ])));
    let seq = sequence(vec![track("1", TrackKind::Video, vec![clip])]);

    // At the last keyframe the content is half-size: the box shrinks to
    // the centered 480×270 quad, and a click near the viewport corner
    // (inside at tick 0) now misses.
    let b = selection_box(&seq, "A", 40, VW, VH, None);
    assert_eq!(
        corners(&b),
        [
            (240.0, 135.0),
            (720.0, 135.0),
            (720.0, 405.0),
            (240.0, 405.0)
        ]
    );
    assert_eq!(hit_test(&seq, 0, 100.0, 60.0, VW, VH).clip_id.as_str(), "A");
    assert_eq!(hit_test(&seq, 40, 100.0, 60.0, VW, VH).clip_id.as_str(), "");
}

#[test]
fn cropped_clip_keeps_the_full_frame_quad() {
    // Keep the right half of full-frame media. This branch's compositor
    // stretches the kept region across the full-frame quad (UV crop), so
    // the hit box and outline stay full-frame — the box must match the
    // rendered pixels. PORT (Phase 6): CapCut's kept-region re-fit
    // arrives with the engine-side placement work.
    let mut clip = media_clip("A", 0, 100, 1920, 1080);
    clip.crop_x = 0.5;
    clip.crop_y = 0.0;
    clip.crop_w = 0.5;
    clip.crop_h = 1.0;
    let seq = sequence(vec![track("1", TrackKind::Video, vec![clip])]);

    // Full-frame geometry: the left edge still hits.
    assert_eq!(
        hit_test(&seq, 10, 100.0, 270.0, VW, VH).clip_id.as_str(),
        "A"
    );
    assert_eq!(
        hit_test(&seq, 10, 480.0, 270.0, VW, VH).clip_id.as_str(),
        "A"
    );

    let b = selection_box(&seq, "A", 10, VW, VH, None);
    assert!(b.visible);
    assert_eq!(
        corners(&b),
        [(0.0, 0.0), (960.0, 0.0), (960.0, 540.0), (0.0, 540.0)]
    );
}

#[test]
fn zeroed_crop_fields_mean_uncropped() {
    // Default-constructed rows (and projections written before crop
    // existed) carry an all-zero rect: full-frame geometry, not a
    // degenerate window.
    let seq = sequence(vec![track(
        "1",
        TrackKind::Video,
        vec![media_clip("A", 0, 100, 1920, 1080)],
    )]);
    assert_eq!(
        hit_test(&seq, 10, 480.0, 270.0, VW, VH).clip_id.as_str(),
        "A"
    );
    let b = selection_box(&seq, "A", 10, VW, VH, None);
    assert_eq!(
        corners(&b),
        [(0.0, 0.0), (960.0, 0.0), (960.0, 540.0), (0.0, 540.0)]
    );
}

#[test]
fn selection_box_hides_when_off_playhead_or_hidden() {
    let mut hidden = track(
        "2",
        TrackKind::Video,
        vec![media_clip("H", 0, 100, 1920, 1080)],
    );
    hidden.enabled = false;
    let seq = sequence(vec![
        hidden,
        track(
            "1",
            TrackKind::Video,
            vec![media_clip("A", 50, 50, 1920, 1080)],
        ),
    ]);
    assert!(!selection_box(&seq, "", 60, VW, VH, None).visible);
    assert!(
        !selection_box(&seq, "A", 10, VW, VH, None).visible,
        "off playhead"
    );
    assert!(selection_box(&seq, "A", 60, VW, VH, None).visible);
    assert!(
        !selection_box(&seq, "H", 60, VW, VH, None).visible,
        "hidden lane"
    );
    assert!(!selection_box(&seq, "404", 60, VW, VH, None).visible);
}

#[test]
fn sprite_placement_is_frame_aligned_at_identity_gesture() {
    let seq = sequence(vec![track(
        "1",
        TrackKind::Video,
        vec![media_clip("A", 0, 100, 1920, 1080)],
    )]);
    let gesture = PreviewDragResolution {
        valid: true,
        moved: false,
        position_x: 0.0,
        position_y: 0.0,
        anchor_x: 0.5,
        anchor_y: 0.5,
        scale: 1.0,
        rotation: 0.0,
        opacity: 1.0,
        ..PreviewDragResolution::default()
    };
    let sprite = sprite_placement_in_viewport(&seq, "A", 10, VW, VH, 1.0, 0.0, 0.0, Some(&gesture));
    assert!(sprite.visible);
    assert!((sprite.x - 0.0).abs() < 1e-3);
    assert!((sprite.y - 0.0).abs() < 1e-3);
    assert!((sprite.width - VW).abs() < 1e-3);
    assert!((sprite.height - VH).abs() < 1e-3);
    assert!((sprite.scale - 1.0).abs() < 1e-3);
    assert!((sprite.rotation_degrees - 0.0).abs() < 1e-3);
}

#[test]
fn sprite_placement_scales_with_gesture() {
    let seq = sequence(vec![track(
        "1",
        TrackKind::Video,
        vec![media_clip("A", 0, 100, 1920, 1080)],
    )]);
    let gesture = PreviewDragResolution {
        valid: true,
        moved: true,
        position_x: 0.0,
        position_y: 0.0,
        anchor_x: 0.5,
        anchor_y: 0.5,
        scale: 0.5,
        rotation: 0.0,
        opacity: 1.0,
        ..PreviewDragResolution::default()
    };
    let sprite = sprite_placement_in_viewport(&seq, "A", 10, VW, VH, 1.0, 0.0, 0.0, Some(&gesture));
    assert!(sprite.visible);
    assert!((sprite.scale - 0.5).abs() < 1e-3);
    let sel_box = selection_box_in_viewport(&seq, "A", 10, VW, VH, 1.0, 0.0, 0.0, Some(&gesture));
    assert!(sel_box.visible);
    // Scaled content sits in the centered quarter of the viewport.
    assert!((sel_box.x0 - 240.0).abs() < 1e-3);
    assert!((sel_box.x2 - 720.0).abs() < 1e-3);
}

#[test]
fn sprite_placement_falls_back_to_committed_transform() {
    let mut clip = media_clip("A", 0, 100, 1920, 1080);
    clip.transform_position_x = 0.1;
    clip.transform_position_y = -0.2;
    clip.transform_scale = 0.6;
    clip.transform_rotation = 30.0;
    clip.transform_opacity = 0.7;
    let seq = sequence(vec![track("1", TrackKind::Video, vec![clip])]);
    let gesture = PreviewDragResolution {
        valid: true,
        moved: true,
        position_x: 0.1,
        position_y: -0.2,
        anchor_x: 0.5,
        anchor_y: 0.5,
        scale: 0.6,
        rotation: 30.0,
        opacity: 0.7,
        ..PreviewDragResolution::default()
    };

    let committed = sprite_placement_in_viewport(&seq, "A", 10, VW, VH, 1.0, 0.0, 0.0, None);
    let live = sprite_placement_in_viewport(&seq, "A", 10, VW, VH, 1.0, 0.0, 0.0, Some(&gesture));

    assert!(committed.visible);
    assert!((committed.x - live.x).abs() < 1e-3);
    assert!((committed.y - live.y).abs() < 1e-3);
    assert!((committed.scale - live.scale).abs() < 1e-3);
    assert!((committed.rotation_degrees - live.rotation_degrees).abs() < 1e-3);
    assert!((committed.opacity - live.opacity).abs() < 1e-3);
}
