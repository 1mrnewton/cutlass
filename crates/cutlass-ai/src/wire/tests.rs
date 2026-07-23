use super::*;

#[test]
fn tagged_json_round_trips() {
    let cmd = WireCommand::TrimClip(TrimClip {
        clip: 12,
        start: 14.0,
        duration: 4.0,
    });
    let json = serde_json::to_value(&cmd).unwrap();
    assert_eq!(
        json,
        serde_json::json!({
            "command": "trim_clip",
            "clip": 12,
            "start": 14.0,
            "duration": 4.0,
        })
    );
    let back: WireCommand = serde_json::from_value(json).unwrap();
    assert_eq!(back, cmd);
}

#[test]
fn from_tool_call_decodes_arguments() {
    let cmd =
        WireCommand::from_tool_call("split_clip", serde_json::json!({ "clip": 7, "at": 12.4 }))
            .unwrap();
    assert_eq!(cmd, WireCommand::SplitClip(SplitClip { clip: 7, at: 12.4 }));
    assert_eq!(cmd.tool_name(), "split_clip");
}

#[test]
fn duplicate_clip_tagged_json_and_tool_arguments_are_strict() {
    let json = serde_json::json!({
        "command": "duplicate_clip",
        "clip": 7,
        "to_track": 3,
        "start": 12.5,
    });
    let command: WireCommand = serde_json::from_value(json).unwrap();
    assert_eq!(
        command,
        WireCommand::DuplicateClip(DuplicateClip {
            clip: 7,
            to_track: 3,
            start: 12.5,
        })
    );
    assert_eq!(command.tool_name(), "duplicate_clip");

    let missing = WireCommand::from_tool_call(
        "duplicate_clip",
        serde_json::json!({ "clip": 7, "to_track": 3 }),
    )
    .unwrap_err();
    assert!(missing.contains("missing field `start`"), "{missing}");

    let extra = WireCommand::from_tool_call(
        "duplicate_clip",
        serde_json::json!({
            "clip": 7,
            "to_track": 3,
            "start": 12.5,
            "ripple": true,
        }),
    )
    .unwrap_err();
    assert!(extra.contains("unknown field `ripple`"), "{extra}");
}

#[test]
fn from_tool_call_rejects_unknown_tool_and_bad_args() {
    let err = WireCommand::from_tool_call("save_project", serde_json::json!({})).unwrap_err();
    assert!(err.contains("unknown tool 'save_project'"));
    assert!(err.contains("add_clip"));

    let err =
        WireCommand::from_tool_call("trim_clip", serde_json::json!({ "clip": "not-a-number" }))
            .unwrap_err();
    assert!(err.contains("invalid arguments for trim_clip"));
}

#[test]
fn generator_decode_rejection_carries_a_corrective_example() {
    // The historical failure mode: a model passes the title text as a
    // bare string instead of the tagged object.
    let err = WireCommand::from_tool_call(
        "add_generated",
        serde_json::json!({
            "track": 2,
            "generator": "Hello world",
            "start": 0.0,
            "duration": 3.0,
        }),
    )
    .unwrap_err();
    assert!(err.contains("invalid arguments for add_generated"), "{err}");
    assert!(
        err.contains("{\"type\": \"text\", \"content\": \"Hello\"}"),
        "{err}"
    );
}

#[test]
fn tool_schemas_are_fully_inlined() {
    // No `$ref` indirection anywhere: weak local models read schemas
    // literally and guess when the shape is behind a reference.
    for spec in tool_specs() {
        let rendered = spec.parameters.to_string();
        assert!(
            !rendered.contains("$ref") && !rendered.contains("$defs"),
            "{} schema is not self-contained: {rendered}",
            spec.name
        );
    }
}

#[test]
fn set_param_keyframe_schema_uses_compact_wire_clip_param() {
    let spec = tool_specs()
        .into_iter()
        .find(|s| s.name == "set_param_keyframe")
        .expect("set_param_keyframe tool");
    let compact = serde_json::to_string(&spec.parameters).expect("serialize");
    assert!(
        compact.len() < 3_000,
        "set_param_keyframe schema is {} bytes; WireClipParam should be a \
         compact string enum + tagged branches, not per-variant oneOf prose",
        compact.len()
    );
    let param = &spec.parameters["properties"]["param"];
    let one_of = param["oneOf"].as_array().expect("param oneOf");
    let string_branch = one_of
        .iter()
        .find(|b| b.get("type") == Some(&serde_json::json!("string")))
        .expect("string enum branch");
    let variants = string_branch["enum"].as_array().expect("enum");
    assert!(
        variants.iter().any(|v| v == "position")
            && variants.iter().any(|v| v == "anchor_point")
            && variants.iter().any(|v| v == "crop"),
        "compact enum missing unit variants: {string_branch}"
    );
    assert!(
        !one_of.iter().any(|b| b.get("const").is_some()),
        "param schema still uses per-variant const oneOf branches"
    );
}

#[test]
fn generator_wire_format_is_tagged_lowercase() {
    let shape = WireGenerator::Shape {
        shape: WireShape::Ellipse,
        rgba: [255, 0, 0, 255],
        width: None,
        height: None,
    };
    assert_eq!(
        serde_json::to_value(&shape).unwrap(),
        serde_json::json!({ "type": "shape", "shape": "ellipse", "rgba": [255, 0, 0, 255] })
    );
}

#[test]
fn remap_ids_rewrites_only_mapped_references() {
    let clip_map = std::collections::HashMap::from([(10u64, 99u64)]);
    let track_map = std::collections::HashMap::from([(2u64, 7u64)]);
    let marker_map = std::collections::HashMap::from([(4u64, 40u64)]);

    let mut mv = WireCommand::MoveClip(MoveClip {
        clip: 10,
        to_track: 2,
        start: 1.0,
    });
    mv.remap_ids(&clip_map, &track_map, &marker_map);
    assert_eq!(
        mv,
        WireCommand::MoveClip(MoveClip {
            clip: 99,
            to_track: 7,
            start: 1.0,
        })
    );

    let mut move_effect = WireCommand::MoveEffect(MoveEffect {
        clip: 10,
        from_index: 0,
        to_index: 2,
    });
    move_effect.remap_ids(&clip_map, &track_map, &marker_map);
    assert_eq!(
        move_effect,
        WireCommand::MoveEffect(MoveEffect {
            clip: 99,
            from_index: 0,
            to_index: 2,
        })
    );

    let mut extract = WireCommand::ExtractAudio(ExtractAudio { clip: 10, track: 2 });
    extract.remap_ids(&clip_map, &track_map, &marker_map);
    assert_eq!(
        extract,
        WireCommand::ExtractAudio(ExtractAudio { clip: 99, track: 7 })
    );

    let mut duplicate = WireCommand::DuplicateClip(DuplicateClip {
        clip: 10,
        to_track: 2,
        start: 8.0,
    });
    duplicate.remap_ids(&clip_map, &track_map, &marker_map);
    assert_eq!(
        duplicate,
        WireCommand::DuplicateClip(DuplicateClip {
            clip: 99,
            to_track: 7,
            start: 8.0,
        })
    );

    // Unmapped ids pass through; link lists remap element-wise.
    let mut link = WireCommand::LinkClips(LinkClips {
        clips: vec![10, 11],
    });
    link.remap_ids(&clip_map, &track_map, &marker_map);
    assert_eq!(
        link,
        WireCommand::LinkClips(LinkClips {
            clips: vec![99, 11],
        })
    );

    let mut unlink = WireCommand::UnlinkClips(UnlinkClips {
        clips: vec![11, 10],
    });
    unlink.remap_ids(&clip_map, &track_map, &marker_map);
    assert_eq!(
        unlink,
        WireCommand::UnlinkClips(UnlinkClips {
            clips: vec![11, 99],
        })
    );

    // Marker references follow the marker map (sandbox add_marker ids
    // land on the live engine's ids during plan replay).
    let mut blend = WireCommand::SetClipBlendMode(SetClipBlendMode {
        clip: 10,
        mode: WireBlendMode::Multiply,
    });
    blend.remap_ids(&clip_map, &track_map, &marker_map);
    assert_eq!(
        blend,
        WireCommand::SetClipBlendMode(SetClipBlendMode {
            clip: 99,
            mode: WireBlendMode::Multiply,
        })
    );

    let mut motion_blur = WireCommand::SetMotionBlur(SetMotionBlur {
        clip: 10,
        enabled: true,
        shutter_deg: Some(180.0),
        samples: Some(8),
    });
    motion_blur.remap_ids(&clip_map, &track_map, &marker_map);
    assert_eq!(
        motion_blur,
        WireCommand::SetMotionBlur(SetMotionBlur {
            clip: 99,
            enabled: true,
            shutter_deg: Some(180.0),
            samples: Some(8),
        })
    );

    let mut styles = WireCommand::SetClipLayerStyles(SetClipLayerStyles {
        clip: 10,
        styles: WireLayerStyles {
            shadow: Some(WireLayerShadow {
                rgba: [0, 0, 0, 128],
                offset: [4.0, 4.0],
                blur: 8.0,
            }),
            ..Default::default()
        },
    });
    styles.remap_ids(&clip_map, &track_map, &marker_map);
    assert_eq!(
        styles,
        WireCommand::SetClipLayerStyles(SetClipLayerStyles {
            clip: 99,
            styles: WireLayerStyles {
                shadow: Some(WireLayerShadow {
                    rgba: [0, 0, 0, 128],
                    offset: [4.0, 4.0],
                    blur: 8.0,
                }),
                ..Default::default()
            },
        })
    );

    let mut set = WireCommand::SetMarker(SetMarker {
        marker: 4,
        at: Some(2.0),
        name: None,
        color: None,
    });
    set.remap_ids(&clip_map, &track_map, &marker_map);
    assert_eq!(
        set,
        WireCommand::SetMarker(SetMarker {
            marker: 40,
            at: Some(2.0),
            name: None,
            color: None,
        })
    );
}

#[test]
fn tool_specs_cover_every_command_with_object_schemas() {
    let specs = tool_specs();
    assert_eq!(specs.len(), 51);
    for spec in &specs {
        assert!(
            !spec.description.is_empty(),
            "{} missing description",
            spec.name
        );
        assert_eq!(
            spec.parameters.get("type").and_then(|t| t.as_str()),
            Some("object"),
            "{} schema is not an object",
            spec.name
        );
    }
}

#[test]
fn extract_audio_schema_requires_explicit_clip_and_track() {
    let specs = tool_specs();
    let extract = specs
        .iter()
        .find(|spec| spec.name == "extract_audio")
        .expect("extract_audio tool");
    assert_eq!(
        extract.parameters["required"],
        serde_json::json!(["clip", "track"])
    );
    assert!(extract.description.contains("Track id required"));
    assert!(extract.description.contains("add_track"));
    assert!(
        WireCommand::from_tool_call("extract_audio", serde_json::json!({"clip": 7}))
            .unwrap_err()
            .contains("missing field `track`")
    );
}

#[test]
fn duplicate_clip_schema_requires_only_explicit_placement_fields() {
    let specs = tool_specs();
    let duplicate = specs
        .iter()
        .find(|spec| spec.name == "duplicate_clip")
        .expect("duplicate_clip tool");
    assert_eq!(
        duplicate.parameters["required"],
        serde_json::json!(["clip", "to_track", "start"])
    );
    assert_eq!(duplicate.parameters["additionalProperties"], false);
    assert!(duplicate.description.contains("Deep-copy"));
    assert!(duplicate.description.contains("Fresh unlinked id"));
    assert!(duplicate.description.contains("explicit track and start"));
    assert!(duplicate.description.contains("does not ripple"));
}

#[test]
fn move_effect_schema_uses_u32_indices() {
    let specs = tool_specs();
    let move_effect = specs
        .iter()
        .find(|spec| spec.name == "move_effect")
        .expect("move_effect tool");
    assert_eq!(
        move_effect.parameters["required"],
        serde_json::json!(["clip", "from_index", "to_index"])
    );
    for field in ["from_index", "to_index"] {
        let index = &move_effect.parameters["properties"][field];
        // Schema cleaner strips schemars `format` / default `minimum: 0`
        // noise; the wire type remains a non-negative integer index.
        assert_eq!(index["type"], "integer");
        assert!(index.get("format").is_none());
    }
}

#[test]
fn unlink_schema_is_a_bounded_nonempty_clip_list() {
    let specs = tool_specs();
    let unlink = specs
        .iter()
        .find(|spec| spec.name == "unlink_clips")
        .expect("unlink_clips tool");
    let clips = &unlink.parameters["properties"]["clips"];
    assert_eq!(clips["type"], "array");
    assert_eq!(clips["minItems"], 1);
    assert_eq!(clips["maxItems"], MAX_MULTI_CLIP_REFS);
    assert_eq!(clips["items"]["type"], "integer");
    assert_eq!(unlink.parameters["required"], serde_json::json!(["clips"]));
}
