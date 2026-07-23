use super::*;
use cutlass_models::ClipId;

#[test]
fn action_lines_read_like_an_edit_log() {
    let split = WireCommand::SplitClip(wire::SplitClip { clip: 7, at: 12.4 });
    assert_eq!(
        describe_action(&split, Some(&EditOutcome::Created(ClipId::from_raw(21)))),
        "split clip 7 at 12.40s (new clip 21)"
    );

    let move_effect = WireCommand::MoveEffect(wire::MoveEffect {
        clip: 7,
        from_index: 0,
        to_index: 2,
    });
    assert_eq!(
        describe_action(&move_effect, None),
        "moved effect 0 to 2 on clip 7"
    );

    let extract = WireCommand::ExtractAudio(wire::ExtractAudio { clip: 7, track: 3 });
    assert_eq!(
        describe_action(&extract, Some(&EditOutcome::Created(ClipId::from_raw(22)))),
        "extracted audio from clip 7 onto track 3 (new clip 22)"
    );

    let duplicate = WireCommand::DuplicateClip(wire::DuplicateClip {
        clip: 7,
        to_track: 3,
        start: 12.5,
    });
    assert_eq!(
        describe_action(
            &duplicate,
            Some(&EditOutcome::Created(ClipId::from_raw(23)))
        ),
        "duplicated clip 7 onto track 3 at 12.50s (new clip 23)"
    );

    let trim = WireCommand::TrimClip(wire::TrimClip {
        clip: 12,
        start: 3.0,
        duration: 7.0,
    });
    assert_eq!(
        describe_action(&trim, Some(&EditOutcome::Updated(ClipId::from_raw(12)))),
        "trimmed clip 12 to 3.00s–10.00s"
    );

    let title = WireCommand::AddGenerated(wire::AddGenerated {
        track: 3,
        generator: wire::WireGenerator::Text {
            content: "INTRO".into(),
        },
        start: 0.0,
        duration: 3.0,
    });
    assert_eq!(
        describe_action(&title, None),
        "added text 'INTRO' at 0.00s for 3.00s on track 3"
    );

    let canvas = WireCommand::SetCanvas(wire::SetCanvas {
        aspect: Some(wire::WireCanvasAspect::Tall9x16),
        background: Some([20, 20, 28]),
    });
    assert_eq!(
        describe_action(&canvas, Some(&EditOutcome::UpdatedCanvas)),
        "set canvas aspect 9:16, background rgb(20, 20, 28)"
    );
}

#[test]
fn system_prompt_carries_state_and_trim_rule() {
    let summary = ProjectSummary {
        name: "demo".into(),
        frame_rate_fps: 24.0,
        duration_seconds: 10.0,
        tracks: vec![],
        markers: vec![],
        canvas: None,
        media: vec![],
    };
    let ctx = EditorContext {
        selected_clips: vec![12],
        playhead_seconds: 3.5,
        ..Default::default()
    };
    let prompt = system_prompt(&summary, &ctx, &AgentExtensions::default());
    assert!(prompt.contains("\"selected_clips\":[12]"));
    assert!(prompt.contains("INCREASE start"));
    assert!(prompt.contains("\"name\":\"demo\""));
    // The Q&A rule: answer from the pushed state, no tool calls.
    assert!(prompt.contains("answer directly from"));
    // The re-inspect rule: after edits, read the new state, don't give up.
    assert!(prompt.contains("call describe_project to read the new"));
    // Unknown footage content is fetchable, not grounds to refuse.
    assert!(prompt.contains("Unknown source-footage content is not missing project state"));
    assert!(prompt.contains("instead of declining the task"));
    // An empty timeline can be constructed directly from media-pool items.
    assert!(prompt.contains("add_clip is the operation that places media-pool footage"));
    assert!(prompt.contains("An empty timeline is a starting point"));
    assert!(prompt.contains("Never ask the user to pre-place footage"));
    // The overlap rule: make room before growing into a packed track.
    assert!(prompt.contains("Clips on one track can never overlap"));
    // No extensions ⇒ no rules or skills sections.
    assert!(!prompt.contains("User rules"));
    assert!(!prompt.contains("read_skill"));
}

#[test]
fn system_prompt_injects_rules_and_skill_index_only() {
    let summary = ProjectSummary {
        name: "demo".into(),
        frame_rate_fps: 24.0,
        duration_seconds: 10.0,
        tracks: vec![],
        markers: vec![],
        canvas: None,
        media: vec![],
    };
    let extensions = AgentExtensions {
        rules: "[user]\nalways vertical 9:16".into(),
        skills: vec![crate::extend::Skill {
            id: "podcast-cleanup".into(),
            name: "Podcast cleanup".into(),
            description: "Clean up a talk recording.".into(),
            body: "SECRET BODY".into(),
        }],
    };
    let prompt = system_prompt(&summary, &EditorContext::default(), &extensions);
    assert!(prompt.contains("always vertical 9:16"));
    assert!(prompt.contains("podcast-cleanup (Podcast cleanup): Clean up a talk recording."));
    // Only the index enters the prompt — bodies load through read_skill.
    assert!(!prompt.contains("SECRET BODY"));
}

#[test]
fn engine_sense_rules_make_open_ended_creative_work_actionable() {
    let rules = engine_sense_rules(&[
        HostToolSpec {
            name: "media_pool_sheet".into(),
            description: "Survey imported visual media.".into(),
            parameters: serde_json::json!({"type": "object"}),
            tier: crate::tools::ToolTier::ReadOnly,
        },
        HostToolSpec {
            name: "media_asset_strip".into(),
            description: "Inspect one source over time.".into(),
            parameters: serde_json::json!({"type": "object"}),
            tier: crate::tools::ToolTier::ReadOnly,
        },
    ]);

    assert!(rules.contains("complete current project snapshot"));
    assert!(rules.contains("freestyle edits"));
    assert!(rules.contains("media_pool_sheet"));
    assert!(rules.contains("media_asset_strip"));
    assert!(rules.contains("rather than declining"));
    assert!(rules.contains("create the required tracks with add_track"));
    assert!(rules.contains("build the sequence with add_clip"));
}

#[test]
fn image_budget_drops_oldest_and_leaves_labeled_placeholders() {
    let mut messages = vec![
        Message::system("s"),
        Message::User {
            content: "look at these".into(),
            images: vec![
                ImagePart::png(vec![1], "timeline at 2.00s"),
                ImagePart::png(vec![2], "timeline at 5.00s"),
            ],
        },
        Message::ToolResult {
            call_id: "call_1".into(),
            content: "screenshot taken".into(),
            images: vec![ImagePart::jpeg(vec![3], "preview at 8.00s")],
        },
    ];

    enforce_image_budget(&mut messages, 1, usize::MAX);

    match &messages[1] {
        Message::User { content, images } => {
            assert!(images.is_empty(), "both older images dropped");
            assert!(content.contains("no longer attached: timeline at 2.00s"));
            assert!(content.contains("no longer attached: timeline at 5.00s"));
        }
        other => panic!("unexpected {other:?}"),
    }
    match &messages[2] {
        Message::ToolResult {
            content, images, ..
        } => {
            assert_eq!(images.len(), 1, "the newest image survives");
            assert_eq!(images[0].label, "preview at 8.00s");
            assert!(!content.contains("no longer attached"));
        }
        other => panic!("unexpected {other:?}"),
    }

    // Under budget: untouched.
    let mut under = vec![Message::User {
        content: "one".into(),
        images: vec![ImagePart::png(vec![1], "a")],
    }];
    enforce_image_budget(&mut under, 8, usize::MAX);
    assert_eq!(image_count(&under[0]), 1);
}

#[test]
fn image_byte_budget_keeps_the_newest_payload_that_fits() {
    let mut messages = vec![
        Message::User {
            content: "old".into(),
            images: vec![ImagePart::png(vec![1; 6], "old six bytes")],
        },
        Message::ToolResult {
            call_id: "call_1".into(),
            content: "new".into(),
            images: vec![
                ImagePart::png(vec![2; 4], "new four bytes"),
                ImagePart::png(vec![3; 5], "newest five bytes"),
            ],
        },
    ];

    enforce_image_budget(&mut messages, 8, 9);

    let Message::User { content, images } = &messages[0] else {
        panic!("user message");
    };
    assert!(images.is_empty());
    assert!(content.contains("old six bytes"));
    let Message::ToolResult {
        content, images, ..
    } = &messages[1]
    else {
        panic!("tool result");
    };
    assert_eq!(
        images
            .iter()
            .map(|image| image.label.as_str())
            .collect::<Vec<_>>(),
        vec!["new four bytes", "newest five bytes"]
    );
    assert!(!content.contains("no longer attached"));

    enforce_image_budget(&mut messages, 8, 3);
    let Message::ToolResult {
        content, images, ..
    } = &messages[1]
    else {
        panic!("tool result");
    };
    assert!(
        images.is_empty(),
        "an individually oversized newest image drops"
    );
    assert!(content.contains("new four bytes"));
    assert!(content.contains("newest five bytes"));
}

#[test]
fn tool_output_budget_drops_before_transcript_delivery() {
    let mut content = "frames ready".to_string();
    let mut images = vec![
        ImagePart::png(vec![1; 5], "old"),
        ImagePart::png(vec![2; 4], "middle"),
        ImagePart::png(vec![3; 3], "new"),
    ];

    enforce_tool_output_image_budget(&mut content, &mut images, 2, 7);

    assert_eq!(
        images
            .iter()
            .map(|image| image.label.as_str())
            .collect::<Vec<_>>(),
        vec!["middle", "new"]
    );
    assert!(content.contains("request budget: old"), "{content}");
}

#[test]
fn turn_messages_strip_images_to_labels() {
    let messages = vec![
        Message::system("s"),
        Message::User {
            content: "what's here?".into(),
            images: vec![ImagePart::png(vec![1], "frame at 0.00s")],
        },
        Message::Assistant {
            content: String::new(),
            tool_calls: Vec::new(),
        },
        Message::ToolResult {
            call_id: "call_1".into(),
            content: "took the shot".into(),
            images: vec![ImagePart::jpeg(vec![2], "preview at 3.00s")],
        },
    ];

    let turn = collect_turn_messages(messages, 1, &[], "done");

    for message in &turn {
        assert_eq!(image_count(message), 0, "history is text-only: {message:?}");
    }
    match &turn[0] {
        Message::User { content, .. } => {
            assert!(content.contains("[image: frame at 0.00s]"), "{content}");
        }
        other => panic!("unexpected {other:?}"),
    }
    match &turn[2] {
        Message::ToolResult { content, .. } => {
            assert!(content.contains("[image: preview at 3.00s]"), "{content}");
        }
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(
        turn.last(),
        Some(&Message::assistant_text("done")),
        "the final answer is appended"
    );
}

#[test]
fn host_action_summary_keeps_the_first_line_capped() {
    assert_eq!(host_action_summary("saved\ndetails follow"), "saved");
    let long = "x".repeat(200);
    let summary = host_action_summary(&long);
    assert_eq!(summary.chars().count(), 121, "120 chars + ellipsis");
    assert!(summary.ends_with('…'));
}

#[test]
fn read_skill_returns_body_or_lists_available() {
    let skills = vec![crate::extend::Skill {
        id: "podcast-cleanup".into(),
        name: "Podcast cleanup".into(),
        description: "d".into(),
        body: "Step 1: denoise.".into(),
    }];
    let ok = read_skill_result(&skills, &serde_json::json!({ "id": "podcast-cleanup" }));
    assert!(ok.contains("Step 1: denoise."));
    let missing = read_skill_result(&skills, &serde_json::json!({ "id": "nope" }));
    assert!(missing.starts_with("rejected: unknown skill 'nope'"));
    assert!(missing.contains("podcast-cleanup"));
}
