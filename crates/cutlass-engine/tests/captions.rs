//! Caption command coverage: one command per group, and exact undo/redo
//! oscillation for every caption edit.

use cutlass_commands::{Command, EditCommand, EditOutcome};
use cutlass_engine::{ApplyOutcome, Engine, EngineConfig};
use cutlass_models::{
    CaptionCueSpec, CaptionGroupId, CaptionGroupSpec, CaptionHighlight, CaptionHighlightMode,
    CaptionLayout, CaptionStyle, CaptionStyleScope, CaptionWord, ClipId, Param, Project, Rational,
    RationalTime, TimeRange, TrackId, TrackKind,
};

fn rt(value: i64) -> RationalTime {
    RationalTime::new(value, Rational::FPS_30)
}

fn tr(start: i64, duration: i64) -> TimeRange {
    TimeRange::at_rate(start, duration, Rational::FPS_30)
}

fn engine_with_text_lane() -> (Engine, TrackId) {
    let mut project = Project::new("captions", Rational::FPS_30);
    let track = project.add_track(TrackKind::Text, "Captions");
    let engine = Engine::with_project(EngineConfig { undo_limit: 32 }, project).expect("engine");
    (engine, track)
}

fn cues() -> Vec<CaptionCueSpec> {
    vec![
        CaptionCueSpec::new("hello world", tr(0, 30)).with_words(vec![
            CaptionWord::new(0, 500, 0..5),
            CaptionWord::new(500, 1000, 6..11),
        ]),
        CaptionCueSpec::new("second line", tr(30, 30)),
    ]
}

fn apply(engine: &mut Engine, command: EditCommand) -> EditOutcome {
    match engine.apply(Command::Edit(command)).expect("apply") {
        ApplyOutcome::Edited(outcome) => outcome,
        other => panic!("expected an edit outcome, got {other:?}"),
    }
}

fn add_captions(engine: &mut Engine, track: TrackId) -> (CaptionGroupId, Vec<ClipId>) {
    let outcome = apply(
        engine,
        EditCommand::AddCaptionGroup {
            group: Box::new(CaptionGroupSpec::manual(track, "Captions")),
            cues: cues(),
        },
    );
    let EditOutcome::CreatedCaptionGroup(group) = outcome else {
        panic!("expected a caption group, got {outcome:?}");
    };
    let clips = engine.project().timeline().caption_cue_ids(group);
    (group, clips)
}

/// Every caption clip and the group itself, as a comparable snapshot.
fn snapshot(engine: &Engine, group: CaptionGroupId) -> String {
    let timeline = engine.project().timeline();
    let cues: Vec<_> = timeline.caption_cues(group).into_iter().cloned().collect();
    format!("{:?}{:?}", timeline.caption_group(group), cues)
}

/// Apply `command`, then assert undo restores the prior state exactly and redo
/// reproduces the post-command state exactly.
fn assert_oscillates(engine: &mut Engine, group: CaptionGroupId, command: EditCommand) {
    let before = snapshot(engine, group);
    apply(engine, command);
    let after = snapshot(engine, group);
    assert_ne!(before, after, "the command changed nothing to oscillate");

    assert!(engine.undo(), "undo");
    assert_eq!(snapshot(engine, group), before, "undo must restore exactly");
    assert!(engine.redo(), "redo");
    assert_eq!(
        snapshot(engine, group),
        after,
        "redo must reproduce exactly"
    );
}

// --- creation -------------------------------------------------------------

#[test]
fn one_command_creates_the_group_and_every_cue() {
    let (mut engine, track) = engine_with_text_lane();
    let (group, clips) = add_captions(&mut engine, track);

    assert_eq!(clips.len(), 2);
    assert_eq!(engine.project().timeline().caption_group_count(), 1);
    assert_eq!(
        engine.project().timeline().clip_count(),
        2,
        "cues are ordinary text clips"
    );
    assert!(engine.project().clip(clips[0]).unwrap().is_caption());
    assert_eq!(
        engine
            .project()
            .clip(clips[0])
            .unwrap()
            .caption
            .as_ref()
            .unwrap()
            .group,
        group
    );
}

#[test]
fn creating_a_group_is_a_single_undo_entry() {
    let (mut engine, track) = engine_with_text_lane();
    let (group, _) = add_captions(&mut engine, track);

    assert!(engine.undo(), "undo");
    assert_eq!(engine.project().timeline().caption_group_count(), 0);
    assert_eq!(engine.project().timeline().clip_count(), 0);

    assert!(engine.redo(), "redo");
    assert_eq!(engine.project().timeline().caption_group_count(), 1);
    assert_eq!(engine.project().timeline().caption_cue_ids(group).len(), 2);
}

#[test]
fn redo_of_a_group_keeps_every_cue_id() {
    let (mut engine, track) = engine_with_text_lane();
    let (group, clips) = add_captions(&mut engine, track);
    assert!(engine.undo(), "undo");
    assert!(engine.redo(), "redo");
    assert_eq!(
        engine.project().timeline().caption_cue_ids(group),
        clips,
        "ids must survive so selection and deeper history keep resolving"
    );
}

#[test]
fn a_rejected_batch_places_nothing() {
    let (mut engine, track) = engine_with_text_lane();
    let overlapping = vec![
        CaptionCueSpec::new("one", tr(0, 30)),
        CaptionCueSpec::new("two", tr(10, 30)),
    ];
    assert!(
        engine
            .apply(Command::Edit(EditCommand::AddCaptionGroup {
                group: Box::new(CaptionGroupSpec::manual(track, "Captions")),
                cues: overlapping,
            }))
            .is_err()
    );
    assert_eq!(engine.project().timeline().clip_count(), 0);
    assert_eq!(engine.project().timeline().caption_group_count(), 0);
    assert!(!engine.can_undo(), "a rejected command leaves no history");
}

// --- group edits ----------------------------------------------------------

#[test]
fn restyling_a_group_oscillates() {
    let (mut engine, track) = engine_with_text_lane();
    let (group, _) = add_captions(&mut engine, track);
    let style = CaptionStyle {
        text: cutlass_models::TextStyle {
            size: Param::Constant(120.0),
            ..Default::default()
        },
        ..CaptionStyle::default()
    };
    assert_oscillates(
        &mut engine,
        group,
        EditCommand::SetCaptionGroupStyle {
            group,
            style: Box::new(style),
            scope: CaptionStyleScope::All,
        },
    );
}

#[test]
fn applying_a_template_oscillates() {
    let (mut engine, track) = engine_with_text_lane();
    let (group, _) = add_captions(&mut engine, track);
    assert_oscillates(
        &mut engine,
        group,
        EditCommand::SetCaptionGroupTemplate {
            group,
            template: "karaoke_pop".into(),
        },
    );
    assert_eq!(
        engine
            .project()
            .timeline()
            .caption_group(group)
            .unwrap()
            .highlight
            .as_ref()
            .map(|h| h.mode),
        Some(CaptionHighlightMode::Word)
    );
}

#[test]
fn layout_highlight_and_label_edits_oscillate() {
    let (mut engine, track) = engine_with_text_lane();
    let (group, _) = add_captions(&mut engine, track);

    assert_oscillates(
        &mut engine,
        group,
        EditCommand::SetCaptionGroupLayout {
            group,
            layout: CaptionLayout {
                safe_area_bottom: 0.35,
                ..CaptionLayout::default()
            },
        },
    );
    assert_oscillates(
        &mut engine,
        group,
        EditCommand::SetCaptionHighlight {
            group,
            highlight: Some(CaptionHighlight::word([255, 0, 0, 255])),
        },
    );
    assert_oscillates(
        &mut engine,
        group,
        EditCommand::SetCaptionGroupLabel {
            group,
            label: "English".into(),
        },
    );
}

#[test]
fn removing_a_group_takes_its_cues_and_undo_brings_them_back() {
    let (mut engine, track) = engine_with_text_lane();
    let (group, clips) = add_captions(&mut engine, track);
    let before = snapshot(&engine, group);

    let outcome = apply(&mut engine, EditCommand::RemoveCaptionGroup { group });
    assert_eq!(outcome, EditOutcome::RemovedCaptionGroup(group));
    assert_eq!(engine.project().timeline().clip_count(), 0);

    assert!(engine.undo(), "undo");
    assert_eq!(snapshot(&engine, group), before);
    assert_eq!(engine.project().timeline().caption_cue_ids(group), clips);

    assert!(engine.redo(), "redo");
    assert_eq!(engine.project().timeline().caption_group_count(), 0);
}

// --- cue edits ------------------------------------------------------------

#[test]
fn editing_a_cue_oscillates() {
    let (mut engine, track) = engine_with_text_lane();
    let (group, clips) = add_captions(&mut engine, track);
    assert_oscillates(
        &mut engine,
        group,
        EditCommand::SetCaptionCue {
            clip: clips[0],
            text: "hello there world".into(),
            words: None,
            speaker: Some("Host".into()),
        },
    );
    assert_eq!(
        engine.project().clip(clips[0]).unwrap().text_content(),
        Some("hello there world")
    );
}

#[test]
fn splitting_a_cue_oscillates_and_renumbers() {
    let (mut engine, track) = engine_with_text_lane();
    let (group, clips) = add_captions(&mut engine, track);
    assert_oscillates(
        &mut engine,
        group,
        EditCommand::SplitCaptionCue {
            clip: clips[0],
            at: rt(15),
        },
    );
    let indices: Vec<u32> = engine
        .project()
        .timeline()
        .caption_cues(group)
        .iter()
        .map(|clip| clip.caption.as_ref().unwrap().index)
        .collect();
    assert_eq!(indices, vec![0, 1, 2]);
}

#[test]
fn merging_cues_oscillates() {
    let (mut engine, track) = engine_with_text_lane();
    let (group, clips) = add_captions(&mut engine, track);
    assert_oscillates(
        &mut engine,
        group,
        EditCommand::MergeCaptionCues {
            clips: clips.clone(),
        },
    );
    assert_eq!(
        engine.project().clip(clips[0]).unwrap().text_content(),
        Some("hello world second line")
    );
}

#[test]
fn ungrouping_leaves_text_clips_and_undo_reattaches_them() {
    let (mut engine, track) = engine_with_text_lane();
    let (group, clips) = add_captions(&mut engine, track);
    let before = snapshot(&engine, group);

    apply(
        &mut engine,
        EditCommand::UngroupCaptions {
            clips: clips.clone(),
        },
    );
    assert_eq!(engine.project().timeline().caption_group_count(), 0);
    assert!(engine.project().clip(clips[0]).unwrap().caption.is_none());
    assert_eq!(
        engine.project().clip(clips[0]).unwrap().text_content(),
        Some("hello world"),
        "the text survives ungrouping"
    );

    assert!(engine.undo(), "undo");
    assert_eq!(snapshot(&engine, group), before);
    assert!(engine.redo(), "redo");
    assert_eq!(engine.project().timeline().caption_group_count(), 0);
}

// --- rejections -----------------------------------------------------------

#[test]
fn caption_commands_reject_unknown_groups_and_plain_clips() {
    let (mut engine, track) = engine_with_text_lane();
    let (_, clips) = add_captions(&mut engine, track);
    let missing = CaptionGroupId::from_raw(u64::MAX - 5);

    assert!(
        engine
            .apply(Command::Edit(EditCommand::SetCaptionGroupTemplate {
                group: missing,
                template: "clean".into(),
            }))
            .is_err()
    );

    let plain = match engine
        .apply(Command::Edit(EditCommand::AddGenerated {
            track,
            generator: cutlass_models::Generator::Text {
                content: "title".into(),
                style: Default::default(),
            },
            timeline: tr(120, 30),
        }))
        .expect("plain title")
    {
        ApplyOutcome::Edited(EditOutcome::Created(id)) => id,
        other => panic!("expected a created clip, got {other:?}"),
    };
    assert!(
        engine
            .apply(Command::Edit(EditCommand::SetCaptionCue {
                clip: plain,
                text: "hi".into(),
                words: None,
                speaker: None,
            }))
            .is_err(),
        "a plain title is not a cue"
    );
    assert!(
        engine
            .apply(Command::Edit(EditCommand::MergeCaptionCues {
                clips: vec![clips[0], plain],
            }))
            .is_err()
    );
}

#[test]
fn an_unknown_caption_template_is_rejected() {
    let (mut engine, track) = engine_with_text_lane();
    let (group, _) = add_captions(&mut engine, track);
    assert!(
        engine
            .apply(Command::Edit(EditCommand::SetCaptionGroupTemplate {
                group,
                template: "does_not_exist".into(),
            }))
            .is_err()
    );
    assert_eq!(
        engine
            .project()
            .timeline()
            .caption_group(group)
            .unwrap()
            .template,
        None
    );
}
