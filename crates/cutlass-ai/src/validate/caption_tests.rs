//! Caption validation: what the agent may say about captions, and what it
//! gets told when it says it wrong.

use super::*;
use crate::wire;

const R24: Rational = Rational::FPS_24;

/// A 24 fps project with a text lane and, for the group cases, one caption
/// group of two cues (0–2 s, 2–4 s) already on it.
fn fixture() -> (Project, u64) {
    let mut project = Project::new("captions", R24);
    project.add_track(TrackKind::Video, "V1");
    let text = project.add_track(TrackKind::Text, "Captions");
    (project, text.raw())
}

fn with_group(project: &mut Project, track: u64) -> (u64, Vec<u64>) {
    let (group, cues) = project
        .add_caption_group(
            &CaptionGroupSpec::manual(TrackId::from_raw(track), "Captions"),
            &[
                CaptionCueSpec::new("first line", TimeRange::at_rate(0, 48, R24)),
                CaptionCueSpec::new("second line", TimeRange::at_rate(48, 48, R24)),
            ],
        )
        .expect("caption group");
    (group.raw(), cues.into_iter().map(|id| id.raw()).collect())
}

fn lower(project: &Project, cmd: WireCommand) -> EditCommand {
    match validate(&cmd, project).expect("command should validate") {
        Command::Edit(edit) => edit,
        other => panic!("expected edit, got {other:?}"),
    }
}

fn reject(project: &Project, cmd: WireCommand) -> String {
    validate(&cmd, project)
        .expect_err("command should be rejected")
        .message
}

fn cue(text: &str, start: f64, duration: f64) -> wire::WireCaptionCue {
    wire::WireCaptionCue {
        text: text.into(),
        start,
        duration,
    }
}

#[test]
fn add_captions_lowers_cues_to_frame_snapped_ranges() {
    let (project, text) = fixture();
    let lowered = lower(
        &project,
        WireCommand::AddCaptions(wire::AddCaptions {
            track: text,
            cues: vec![
                cue("hello there", 0.0, 1.5),
                cue("general kenobi", 1.5, 2.0),
            ],
            label: Some("Intro".into()),
            template: Some("karaoke_pop".into()),
        }),
    );
    let EditCommand::AddCaptionGroup { group, cues } = lowered else {
        panic!("expected AddCaptionGroup, got {lowered:?}");
    };
    assert_eq!(group.track, TrackId::from_raw(text));
    assert_eq!(group.label, "Intro");
    assert_eq!(group.template.as_deref(), Some("karaoke_pop"));
    assert_eq!(group.source, CaptionSource::Manual);
    assert_eq!(cues.len(), 2);
    assert_eq!(cues[0].text, "hello there");
    assert_eq!(cues[0].timeline, TimeRange::at_rate(0, 36, R24));
    assert_eq!(cues[1].timeline, TimeRange::at_rate(36, 48, R24));
    // Word timings are not part of the agent's vocabulary.
    assert!(cues.iter().all(|cue| cue.words.is_empty()));
}

#[test]
fn add_captions_names_the_missing_text_lane_and_the_template_catalog() {
    let (mut project, text) = fixture();
    let video = project
        .timeline()
        .tracks_ordered()
        .find(|t| t.kind == TrackKind::Video)
        .expect("video lane")
        .id
        .raw();

    let message = reject(
        &project,
        WireCommand::AddCaptions(wire::AddCaptions {
            track: video,
            cues: vec![cue("nope", 0.0, 1.0)],
            label: None,
            template: None,
        }),
    );
    assert!(message.contains("video lane"), "{message}");
    assert!(message.contains("add_track"), "{message}");

    let message = reject(
        &project,
        WireCommand::AddCaptions(wire::AddCaptions {
            track: text,
            cues: vec![cue("nope", 0.0, 1.0)],
            label: None,
            template: Some("neon_wobble".into()),
        }),
    );
    assert!(message.contains("neon_wobble"), "{message}");
    assert!(message.contains("karaoke_pop"), "{message}");

    // Blank lines, empty lists, and overlaps are named before anything lands.
    assert!(
        reject(
            &project,
            WireCommand::AddCaptions(wire::AddCaptions {
                track: text,
                cues: vec![],
                label: None,
                template: None,
            }),
        )
        .contains("at least one cue")
    );
    let message = reject(
        &project,
        WireCommand::AddCaptions(wire::AddCaptions {
            track: text,
            cues: vec![cue("   ", 0.0, 1.0)],
            label: None,
            template: None,
        }),
    );
    assert!(message.contains("no text"), "{message}");
    let message = reject(
        &project,
        WireCommand::AddCaptions(wire::AddCaptions {
            track: text,
            cues: vec![cue("one", 0.0, 2.0), cue("two", 1.0, 2.0)],
            label: None,
            template: None,
        }),
    );
    assert!(message.contains("must not overlap"), "{message}");

    // A group id is not a track id: unknown ids list what exists.
    let (group, _) = with_group(&mut project, text);
    let message = reject(
        &project,
        WireCommand::RemoveCaptions(wire::RemoveCaptions { group: group + 100 }),
    );
    assert!(message.contains("caption groups: "), "{message}");
    assert!(message.contains(&group.to_string()), "{message}");
}

#[test]
fn add_captions_caps_the_cue_list() {
    let (project, text) = fixture();
    let cues = (0..=MAX_WIRE_CAPTION_CUES)
        .map(|i| cue("line", i as f64, 0.5))
        .collect();
    let message = reject(
        &project,
        WireCommand::AddCaptions(wire::AddCaptions {
            track: text,
            cues,
            label: None,
            template: None,
        }),
    );
    assert!(message.contains("several calls"), "{message}");
}

#[test]
fn set_caption_style_patches_and_keeps_the_rest() {
    let (mut project, text) = fixture();
    let (group, _) = with_group(&mut project, text);
    let before = project
        .timeline()
        .caption_group(CaptionGroupId::from_raw(group))
        .expect("group")
        .style
        .clone();

    let lowered = lower(
        &project,
        WireCommand::SetCaptionStyle(wire::SetCaptionStyle {
            group,
            font: None,
            size: Some(96.0),
            fill: None,
            bold: None,
            italic: None,
            uppercase: Some(true),
            position_y: Some(0.1),
            scale: None,
            keep_overrides: None,
        }),
    );
    let EditCommand::SetCaptionGroupStyle { style, scope, .. } = lowered else {
        panic!("expected SetCaptionGroupStyle, got {lowered:?}");
    };
    assert_eq!(style.text.size.constant(), Some(96.0));
    assert_eq!(style.text.case, cutlass_models::TextCase::Upper);
    assert_eq!(style.position[1], 0.1);
    assert_eq!(
        style.text.fill, before.text.fill,
        "an omitted field keeps its current value"
    );
    assert_eq!(
        scope,
        CaptionStyleScope::All,
        "restyling is CapCut's apply-to-all by default"
    );

    let lowered = lower(
        &project,
        WireCommand::SetCaptionStyle(wire::SetCaptionStyle {
            group,
            font: None,
            size: None,
            fill: Some([1, 2, 3, 4]),
            bold: None,
            italic: None,
            uppercase: None,
            position_y: None,
            scale: None,
            keep_overrides: Some(true),
        }),
    );
    let EditCommand::SetCaptionGroupStyle { scope, .. } = lowered else {
        panic!("expected SetCaptionGroupStyle, got {lowered:?}");
    };
    assert_eq!(scope, CaptionStyleScope::KeepOverrides);

    // Out-of-range values are rejected with the range, not clamped silently.
    let message = reject(
        &project,
        WireCommand::SetCaptionStyle(wire::SetCaptionStyle {
            group,
            font: None,
            size: None,
            fill: None,
            bold: None,
            italic: None,
            uppercase: None,
            position_y: Some(4.0),
            scale: None,
            keep_overrides: None,
        }),
    );
    assert!(message.contains("between -0.5"), "{message}");
    let message = reject(
        &project,
        WireCommand::SetCaptionStyle(wire::SetCaptionStyle {
            group,
            font: None,
            size: None,
            fill: None,
            bold: None,
            italic: None,
            uppercase: None,
            position_y: None,
            scale: Some(50.0),
            keep_overrides: None,
        }),
    );
    assert!(message.contains("caption scale"), "{message}");
}

#[test]
fn set_caption_layout_converts_seconds_to_milliseconds() {
    let (mut project, text) = fixture();
    let (group, _) = with_group(&mut project, text);
    let lowered = lower(
        &project,
        WireCommand::SetCaptionLayout(wire::SetCaptionLayout {
            group,
            max_chars_per_line: Some(20),
            max_lines: Some(1),
            min_duration: Some(0.8),
            max_duration: Some(3.5),
            min_gap: None,
            safe_area_bottom: Some(0.25),
        }),
    );
    let EditCommand::SetCaptionGroupLayout { layout, .. } = lowered else {
        panic!("expected SetCaptionGroupLayout, got {lowered:?}");
    };
    assert_eq!(layout.max_chars_per_line, 20);
    assert_eq!(layout.max_lines, 1);
    assert_eq!(layout.min_duration_ms, 800);
    assert_eq!(layout.max_duration_ms, 3_500);
    assert_eq!(layout.safe_area_bottom, 0.25);
    assert_eq!(
        layout.min_gap_ms,
        CaptionLayoutDefaults::default().min_gap_ms,
        "an omitted rule keeps its current value"
    );

    let message = reject(
        &project,
        WireCommand::SetCaptionLayout(wire::SetCaptionLayout {
            group,
            max_chars_per_line: Some(2),
            max_lines: None,
            min_duration: None,
            max_duration: None,
            min_gap: None,
            safe_area_bottom: None,
        }),
    );
    assert!(message.contains("max_chars_per_line"), "{message}");
}

/// The group's default layout, for asserting "omitted keeps".
type CaptionLayoutDefaults = cutlass_models::CaptionLayout;

#[test]
fn set_caption_highlight_turns_on_off_and_patches_colors() {
    let (mut project, text) = fixture();
    let (group, _) = with_group(&mut project, text);

    let lowered = lower(
        &project,
        WireCommand::SetCaptionHighlight(wire::SetCaptionHighlight {
            group,
            mode: wire::WireCaptionHighlightMode::Word,
            fill: Some([255, 0, 0, 255]),
            plate: Some([0, 0, 255, 200]),
            plate_radius: Some(1.0),
            scale: Some(1.2),
        }),
    );
    let EditCommand::SetCaptionHighlight { highlight, .. } = lowered else {
        panic!("expected SetCaptionHighlight, got {lowered:?}");
    };
    let highlight = highlight.expect("word mode sets a highlight");
    assert_eq!(highlight.mode, CaptionHighlightMode::Word);
    assert_eq!(highlight.fill, [255, 0, 0, 255]);
    assert_eq!(highlight.plate, Some([0, 0, 255, 200]));
    assert_eq!(highlight.plate_radius, 1.0);
    assert_eq!(highlight.scale, 1.2);

    // A fully transparent plate means "no plate", not an invisible card.
    let lowered = lower(
        &project,
        WireCommand::SetCaptionHighlight(wire::SetCaptionHighlight {
            group,
            mode: wire::WireCaptionHighlightMode::Line,
            fill: None,
            plate: Some([0, 0, 0, 0]),
            plate_radius: None,
            scale: None,
        }),
    );
    let EditCommand::SetCaptionHighlight { highlight, .. } = lowered else {
        panic!("expected SetCaptionHighlight, got {lowered:?}");
    };
    let highlight = highlight.expect("line mode sets a highlight");
    assert_eq!(highlight.mode, CaptionHighlightMode::Line);
    assert_eq!(highlight.plate, None);

    let lowered = lower(
        &project,
        WireCommand::SetCaptionHighlight(wire::SetCaptionHighlight {
            group,
            mode: wire::WireCaptionHighlightMode::Off,
            fill: Some([1, 2, 3, 4]),
            plate: None,
            plate_radius: None,
            scale: None,
        }),
    );
    assert!(
        matches!(
            lowered,
            EditCommand::SetCaptionHighlight {
                highlight: None,
                ..
            }
        ),
        "off clears the highlight, ignoring the colors: {lowered:?}"
    );
}

#[test]
fn set_caption_highlight_patches_onto_the_current_colors() {
    let (mut project, text) = fixture();
    let (group, _) = with_group(&mut project, text);
    project
        .set_caption_highlight(
            CaptionGroupId::from_raw(group),
            Some(cutlass_models::CaptionHighlight::word([9, 9, 9, 255])),
        )
        .unwrap();
    let lowered = lower(
        &project,
        WireCommand::SetCaptionHighlight(wire::SetCaptionHighlight {
            group,
            mode: wire::WireCaptionHighlightMode::Line,
            fill: None,
            plate: None,
            plate_radius: None,
            scale: None,
        }),
    );
    let EditCommand::SetCaptionHighlight { highlight, .. } = lowered else {
        panic!("expected SetCaptionHighlight, got {lowered:?}");
    };
    assert_eq!(
        highlight.expect("highlight").fill,
        [9, 9, 9, 255],
        "switching word→line keeps the color the user picked"
    );
}

#[test]
fn cue_edits_reject_plain_titles_and_cross_group_merges() {
    let (mut project, text) = fixture();
    let (_, cues) = with_group(&mut project, text);
    let title = project
        .add_generated(
            TrackId::from_raw(text),
            Generator::text("INTRO"),
            TimeRange::at_rate(240, 48, R24),
        )
        .unwrap()
        .raw();

    let message = reject(
        &project,
        WireCommand::SetCaptionText(wire::SetCaptionText {
            clip: title,
            text: "nope".into(),
            speaker: None,
        }),
    );
    assert!(message.contains("not a caption cue"), "{message}");
    assert!(message.contains("set_generator"), "{message}");

    let lowered = lower(
        &project,
        WireCommand::SetCaptionText(wire::SetCaptionText {
            clip: cues[0],
            text: "corrected line".into(),
            speaker: Some("Ana".into()),
        }),
    );
    let EditCommand::SetCaptionCue {
        text: corrected,
        words,
        speaker,
        ..
    } = lowered
    else {
        panic!("expected SetCaptionCue, got {lowered:?}");
    };
    assert_eq!(corrected, "corrected line");
    assert_eq!(speaker.as_deref(), Some("Ana"));
    assert!(
        words.is_none(),
        "existing timings are remapped, not dropped"
    );

    // A second group makes the cross-group merge case reachable.
    let (other, other_cues) = {
        let (group, cues) = project
            .add_caption_group(
                &CaptionGroupSpec::manual(TrackId::from_raw(text), "Outro"),
                &[CaptionCueSpec::new(
                    "outro",
                    TimeRange::at_rate(480, 48, R24),
                )],
            )
            .expect("second group");
        (group.raw(), cues)
    };
    let _ = other;
    let message = reject(
        &project,
        WireCommand::MergeCaptions(wire::MergeCaptions {
            clips: vec![cues[0], other_cues[0].raw()],
        }),
    );
    assert!(message.contains("across groups"), "{message}");

    assert!(
        reject(
            &project,
            WireCommand::MergeCaptions(wire::MergeCaptions {
                clips: vec![cues[0]],
            }),
        )
        .contains("at least two")
    );
    let message = reject(
        &project,
        WireCommand::MergeCaptions(wire::MergeCaptions {
            clips: vec![cues[0], cues[0]],
        }),
    );
    assert!(message.contains("twice"), "{message}");

    let lowered = lower(
        &project,
        WireCommand::MergeCaptions(wire::MergeCaptions {
            clips: vec![cues[0], cues[1]],
        }),
    );
    assert!(
        matches!(lowered, EditCommand::MergeCaptionCues { ref clips } if clips.len() == 2),
        "{lowered:?}"
    );
}

#[test]
fn splitting_a_cue_routes_to_the_caption_aware_split() {
    let (mut project, text) = fixture();
    let (_, cues) = with_group(&mut project, text);
    let title = project
        .add_generated(
            TrackId::from_raw(text),
            Generator::text("INTRO"),
            TimeRange::at_rate(240, 48, R24),
        )
        .unwrap()
        .raw();

    let lowered = lower(
        &project,
        WireCommand::SplitClip(wire::SplitClip {
            clip: cues[0],
            at: 1.0,
        }),
    );
    assert!(
        matches!(lowered, EditCommand::SplitCaptionCue { .. }),
        "a caption cue splits its text and timings: {lowered:?}"
    );

    let lowered = lower(
        &project,
        WireCommand::SplitClip(wire::SplitClip {
            clip: title,
            at: 11.0,
        }),
    );
    assert!(
        matches!(lowered, EditCommand::SplitClip { .. }),
        "an ordinary title still splits generically: {lowered:?}"
    );
}
