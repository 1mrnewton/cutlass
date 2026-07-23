use super::*;
use cutlass_models::{
    ClipTransform, Easing, MediaSource, Param, RationalTime, Scale2, TimeRange, TrackKind,
};

const R24: Rational = Rational::FPS_24;

fn media_clip_project() -> (Project, cutlass_models::ClipId) {
    let mut project = Project::new("xform", R24);
    let media = project.add_media(MediaSource::new(
        "/footage/clip.mp4",
        1920,
        1080,
        R24,
        24 * 10,
        true,
    ));
    let video = project.add_track(TrackKind::Video, "V1");
    let clip = project
        .add_clip(
            video,
            media,
            TimeRange::at_rate(0, 48, R24),
            RationalTime::new(24, R24),
        )
        .unwrap();
    (project, clip)
}

#[test]
fn summary_is_deterministic_and_complete() {
    let mut project = Project::new("demo", R24);
    let media = project.add_media(MediaSource::new(
        "/footage/interview.mp4",
        1920,
        1080,
        R24,
        24 * 60,
        true,
    ));
    let video = project.add_track(TrackKind::Video, "V1");
    let text = project.add_track(TrackKind::Text, "Titles");

    // Insert out of timeline order to prove ordering is by start time.
    let late = project
        .add_clip(
            video,
            media,
            TimeRange::at_rate(0, 48, R24),
            RationalTime::new(96, R24),
        )
        .unwrap();
    let early = project
        .add_clip(
            video,
            media,
            TimeRange::at_rate(48, 48, R24),
            RationalTime::new(0, R24),
        )
        .unwrap();
    project
        .add_generated(
            text,
            Generator::text("INTRO"),
            TimeRange::at_rate(24, 48, R24),
        )
        .unwrap();

    let summary = summarize(&project);
    assert_eq!(summary.name, "demo");
    assert_eq!(summary.frame_rate_fps, 24.0);
    assert_eq!(summary.duration_seconds, 6.0);
    assert_eq!(summary.tracks.len(), 2);
    assert_eq!(summary.media.len(), 1);

    let v1 = &summary.tracks[0];
    assert_eq!(v1.kind, "video");
    let clip_ids: Vec<u64> = v1.clips.iter().map(|c| c.id).collect();
    assert_eq!(clip_ids, vec![early.raw(), late.raw()]);
    assert_eq!(v1.clips[0].start_seconds, 0.0);
    assert_eq!(v1.clips[0].duration_seconds, 2.0);
    assert_eq!(v1.clips[0].start_frames, 0);
    assert_eq!(v1.clips[0].duration_frames, 48);
    match &v1.clips[0].content {
        ClipContent::Media {
            file,
            source_start_seconds,
            ..
        } => {
            assert_eq!(file, "interview.mp4");
            assert_eq!(*source_start_seconds, 2.0);
        }
        other => panic!("expected media content, got {other:?}"),
    }

    let titles = &summary.tracks[1];
    assert_eq!(titles.kind, "text");
    assert_eq!(
        titles.clips[0].content,
        ClipContent::Text {
            text: "INTRO".to_string()
        }
    );

    assert_eq!(summary.media[0].file, "interview.mp4");
    assert_eq!(summary.media[0].duration_seconds, 60.0);
    assert!(summary.media[0].has_audio);
}

#[test]
fn phantom_generators_surface_as_other() {
    let mut project = Project::new("phantoms", R24);
    let adj = project.add_track(TrackKind::Adjustment, "FX");
    project
        .add_generated(adj, Generator::Adjustment, TimeRange::at_rate(0, 24, R24))
        .unwrap();

    let summary = summarize(&project);
    assert_eq!(summary.tracks[0].kind, "adjustment");
    assert_eq!(
        summary.tracks[0].clips[0].content,
        ClipContent::Other {
            kind: "adjustment".to_string()
        }
    );
}

#[test]
fn canvas_always_includes_pixel_size() {
    let mut project = Project::new("canvas", R24);
    // Empty Auto project → default 1920×1080.
    assert_eq!(
        summarize(&project).canvas,
        CanvasSummary {
            width: 1920,
            height: 1080,
            aspect: "auto".to_string(),
            background: [0, 0, 0],
        }
    );

    project
        .timeline_mut()
        .set_canvas(cutlass_models::CanvasSettings {
            aspect: cutlass_models::CanvasAspect::Tall9x16,
            background: [20, 20, 28],
        });
    assert_eq!(
        summarize(&project).canvas,
        CanvasSummary {
            width: 1080,
            height: 1920,
            aspect: "9:16".to_string(),
            background: [20, 20, 28],
        }
    );
}

#[test]
fn summary_and_context_serialize_to_stable_json() {
    let mut project = Project::new("json", R24);
    let track = project.add_track(TrackKind::Text, "T");
    project
        .add_generated(track, Generator::text("hi"), TimeRange::at_rate(0, 24, R24))
        .unwrap();

    let summary_json = serde_json::to_value(summarize(&project)).unwrap();
    let clip = &summary_json["tracks"][0]["clips"][0];
    assert_eq!(clip["content"], "text");
    assert_eq!(clip["text"], "hi");

    let ctx = EditorContext {
        selected_clips: vec![12],
        playhead_seconds: 3.5,
        in_point_seconds: None,
        out_point_seconds: None,
    };
    let ctx_json = serde_json::to_value(&ctx).unwrap();
    assert_eq!(
        ctx_json,
        serde_json::json!({ "selected_clips": [12], "playhead_seconds": 3.5 })
    );
}

#[test]
fn identity_transform_omits_placement_fields() {
    let (project, _) = media_clip_project();
    let clip = &summarize(&project).tracks[0].clips[0];
    assert_eq!(clip.position, None);
    assert_eq!(clip.anchor, None);
    assert_eq!(clip.scale, None);
    assert_eq!(clip.rotation, None);
    assert_eq!(clip.opacity, None);
}

#[test]
fn static_transform_surfaces_exact_values() {
    let (mut project, clip_id) = media_clip_project();
    project
        .set_transform(
            clip_id,
            ClipTransform {
                position: [0.25, -0.125],
                anchor_point: [0.0, 1.0],
                scale: Scale2::uniform(1.5),
                rotation: 45.0,
                opacity: 0.75,
            },
            None,
        )
        .unwrap();

    let clip = &summarize(&project).tracks[0].clips[0];
    assert_eq!(clip.position, Some([0.25, -0.125]));
    assert_eq!(clip.anchor, Some([0.0, 1.0]));
    assert_eq!(clip.scale, Some(WireScale::Uniform(1.5)));
    assert_eq!(clip.rotation, Some(45.0));
    assert_eq!(clip.opacity, Some(0.75));
}

#[test]
fn animated_position_omits_static_field() {
    let (mut project, clip_id) = media_clip_project();
    {
        let clip = project.timeline_mut().clip_mut(clip_id).unwrap();
        clip.transform.position = Param::Constant([0.5, 0.0]);
        clip.transform
            .position
            .set_keyframe(0, [-1.0, 0.0], Easing::Linear);
        clip.transform
            .position
            .set_keyframe(24, [0.0, 0.0], Easing::EaseOut);
    }

    let clip = &summarize(&project).tracks[0].clips[0];
    assert_eq!(clip.position, None);
    // Other static transform fields still surface at identity / defaults.
    assert_eq!(clip.anchor, None);
    assert_eq!(clip.rotation, None);
    assert_eq!(clip.opacity, None);
}

#[test]
fn keyframes_absent_when_nothing_animated() {
    let (project, _) = media_clip_project();
    assert_eq!(summarize(&project).tracks[0].clips[0].keyframes, None);
}

#[test]
fn position_keyframes_use_absolute_timeline_seconds() {
    // Clip starts at 1.0s (24 ticks @ 24fps); keyframes at clip-relative 0 and 24.
    let (mut project, clip_id) = media_clip_project();
    {
        let clip = project.timeline_mut().clip_mut(clip_id).unwrap();
        clip.transform
            .position
            .set_keyframe(0, [-1.0, 0.0], Easing::Linear);
        clip.transform
            .position
            .set_keyframe(24, [0.0, 0.0], Easing::EaseOut);
    }

    let clip = &summarize(&project).tracks[0].clips[0];
    let kfs = clip.keyframes.as_ref().expect("keyframes present");
    let position = kfs.get("position").expect("position keyframes");
    assert_eq!(position.len(), 2);
    assert_eq!(position[0].at, 1.0);
    assert_eq!(position[0].value, serde_json::json!([-1.0, 0.0]));
    assert_eq!(position[0].easing, None);
    assert_eq!(position[1].at, 2.0);
    assert_eq!(position[1].value, serde_json::json!([0.0, 0.0]));
    assert_eq!(position[1].easing, Some(serde_json::json!("ease_out")));
    assert_eq!(clip.position, None);
}

#[test]
fn scale_keyframes_use_wire_scale_shape() {
    let (mut project, clip_id) = media_clip_project();
    {
        let clip = project.timeline_mut().clip_mut(clip_id).unwrap();
        clip.transform
            .scale
            .set_keyframe(0, Scale2::uniform(1.0), Easing::Linear);
        clip.transform
            .scale
            .set_keyframe(24, Scale2 { x: 1.5, y: 0.5 }, Easing::EaseIn);
    }

    let summary = summarize(&project);
    let kfs = summary.tracks[0].clips[0]
        .keyframes
        .as_ref()
        .expect("keyframes present");
    let scale = kfs.get("scale").expect("scale keyframes");
    assert_eq!(scale[0].value, serde_json::json!(1.0));
    assert_eq!(scale[1].value, serde_json::json!([1.5, 0.5]));
    assert_eq!(scale[1].easing, Some(serde_json::json!("ease_in")));
}

#[test]
fn custom_bezier_easing_describes_wire_object_shape() {
    let (mut project, clip_id) = media_clip_project();
    let points = [0.2, 0.0, 0.8, 1.0];
    {
        let clip = project.timeline_mut().clip_mut(clip_id).unwrap();
        clip.transform
            .position
            .set_keyframe(0, [-1.0, 0.0], Easing::Bezier { points });
        clip.transform
            .position
            .set_keyframe(24, [0.0, 0.0], Easing::Linear);
    }
    let summary = summarize(&project);
    let kfs = summary.tracks[0].clips[0]
        .keyframes
        .as_ref()
        .expect("keyframes");
    let position = kfs.get("position").expect("position");
    assert_eq!(
        position[0].easing,
        Some(serde_json::json!({"bezier":{"points":[0.2, 0.0, 0.8, 1.0]}})),
        "custom bezier must emit the wire object, not the bare string \"bezier\""
    );
    assert_eq!(position[1].easing, None);
}

#[test]
fn describe_round_trips_non_dyadic_scale_through_f32() {
    // 1.3 is not exact in f32; raw f64::from would expand to 1.2999… and
    // teach the model the wrong wire value. Describe must keep 1.3.
    let (mut project, clip_id) = media_clip_project();
    {
        let clip = project.timeline_mut().clip_mut(clip_id).unwrap();
        clip.transform
            .scale
            .set_keyframe(0, Scale2::uniform(1.3), Easing::Linear);
    }
    let summary = summarize(&project);
    let scale = &summary.tracks[0].clips[0]
        .keyframes
        .as_ref()
        .expect("keyframes")["scale"];
    assert_eq!(scale[0].value, serde_json::json!(1.3));
}

/// Motion fields only — pins presence/absence plus `t`/`v`/`e` shape.
fn motion_json_subtree(clip: &serde_json::Value) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for key in [
        "position",
        "anchor",
        "scale",
        "rotation",
        "opacity",
        "keyframes",
    ] {
        if let Some(v) = clip.get(key) {
            map.insert(key.to_string(), v.clone());
        }
    }
    serde_json::Value::Object(map)
}

#[test]
fn motion_state_json_subtree_is_pinned() {
    // One keyframed clip + one static transformed clip on a 1920×1080 canvas.
    // Exact JSON pin so future describe edits cannot silently drop motion.
    let mut project = Project::new("motion-pin", R24);
    project
        .timeline_mut()
        .set_canvas(cutlass_models::CanvasSettings {
            aspect: cutlass_models::CanvasAspect::Wide16x9,
            background: [0, 0, 0],
        });
    let media = project.add_media(MediaSource::new(
        "/footage/pin.mp4",
        1920,
        1080,
        R24,
        24 * 30,
        true,
    ));
    let video = project.add_track(TrackKind::Video, "V1");
    let animated = project
        .add_clip(
            video,
            media,
            TimeRange::at_rate(0, 72, R24),
            RationalTime::new(0, R24),
        )
        .unwrap();
    let placed = project
        .add_clip(
            video,
            media,
            TimeRange::at_rate(0, 48, R24),
            RationalTime::new(96, R24),
        )
        .unwrap();

    {
        let clip = project.timeline_mut().clip_mut(animated).unwrap();
        clip.transform
            .position
            .set_keyframe(0, [-1.0, 0.0], Easing::Linear);
        clip.transform
            .position
            .set_keyframe(24, [0.0, 0.0], Easing::EaseOut);
        clip.transform
            .scale
            .set_keyframe(0, Scale2::uniform(1.0), Easing::Linear);
        clip.transform
            .scale
            .set_keyframe(24, Scale2::uniform(1.3), Easing::EaseIn);
    }
    project
        .set_transform(
            placed,
            ClipTransform {
                position: [0.25, -0.125],
                anchor_point: [0.0, 1.0],
                scale: Scale2::uniform(1.5),
                rotation: 45.0,
                opacity: 0.75,
            },
            None,
        )
        .unwrap();

    let summary = serde_json::to_value(summarize(&project)).unwrap();
    assert_eq!(
        summary["canvas"],
        serde_json::json!({
            "width": 1920,
            "height": 1080,
            "aspect": "16:9",
            "background": [0, 0, 0],
        })
    );

    let clips = &summary["tracks"][0]["clips"];
    assert_eq!(clips[0]["id"], animated.raw());
    assert_eq!(clips[1]["id"], placed.raw());

    // Animated: static position/scale omitted; keyframes carry absolute `t`.
    assert_eq!(
        motion_json_subtree(&clips[0]),
        serde_json::json!({
            "keyframes": {
                "position": [
                    { "t": 0.0, "v": [-1.0, 0.0] },
                    { "t": 1.0, "v": [0.0, 0.0], "e": "ease_out" },
                ],
                "scale": [
                    { "t": 0.0, "v": 1.0 },
                    { "t": 1.0, "v": 1.3, "e": "ease_in" },
                ],
            }
        })
    );

    // Static: full placement fields, no keyframes map.
    assert_eq!(
        motion_json_subtree(&clips[1]),
        serde_json::json!({
            "position": [0.25, -0.125],
            "anchor": [0.0, 1.0],
            "scale": 1.5,
            "rotation": 45.0,
            "opacity": 0.75,
        })
    );
}
