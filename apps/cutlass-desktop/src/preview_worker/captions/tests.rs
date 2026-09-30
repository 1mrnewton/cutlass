use super::*;
use cutlass_models::CaptionWord;

/// A text lane holding one caption cue over `[start, start + duration)`.
fn lane_with_cue(engine: &mut Engine, start: i64, duration: i64) -> TrackId {
    let track = create_track(engine, TrackKind::Text, 0).expect("text lane");
    let rate = engine.project().timeline().frame_rate;
    engine
        .apply(Command::Edit(EditCommand::AddCaptionGroup {
            group: Box::new(CaptionGroupSpec::manual(track, "Captions")),
            cues: vec![CaptionCueSpec::new(
                "hello",
                TimeRange::at_rate(start, duration, rate),
            )],
        }))
        .expect("caption group");
    track
}

fn cues(engine: &Engine, spans: &[(i64, i64)]) -> Vec<CaptionCueSpec> {
    let rate = engine.project().timeline().frame_rate;
    spans
        .iter()
        .map(|&(start, duration)| {
            CaptionCueSpec::new("line", TimeRange::at_rate(start, duration, rate))
        })
        .collect()
}

/// One four-second cue, four words of one second each, on a fresh text lane,
/// optionally in a catalog template's look.
fn group_with_one_long_cue(engine: &mut Engine, template: Option<&str>) -> CaptionGroupId {
    let track = create_track(engine, TrackKind::Text, 0).expect("text lane");
    let rate = engine.project().timeline().frame_rate;
    let text = "alpha bravo charlie delta";
    let words = vec![
        CaptionWord::new(0, 1_000, 0..5),
        CaptionWord::new(1_000, 2_000, 6..11),
        CaptionWord::new(2_000, 3_000, 12..19),
        CaptionWord::new(3_000, 4_000, 20..25),
    ];
    let mut spec = CaptionGroupSpec::manual(track, "Captions");
    if let Some(template) = template {
        spec = spec.with_template(template);
    }
    let outcome = engine
        .apply(Command::Edit(EditCommand::AddCaptionGroup {
            group: Box::new(spec),
            cues: vec![
                CaptionCueSpec::new(text, TimeRange::at_rate(0, 120, rate)).with_words(words),
            ],
        }))
        .expect("caption group");
    match outcome {
        ApplyOutcome::Edited(EditOutcome::CreatedCaptionGroup(id)) => id,
        other => panic!("unexpected outcome: {other:?}"),
    }
}

fn cue_texts(engine: &Engine, group: CaptionGroupId) -> Vec<String> {
    engine
        .project()
        .timeline()
        .caption_cues(group)
        .into_iter()
        .filter_map(|clip| clip.text_content().map(str::to_owned))
        .collect()
}

fn layout(max_chars_per_line: u16, max_lines: u8) -> CaptionLayout {
    CaptionLayout {
        max_chars_per_line,
        max_lines,
        ..CaptionLayout::default()
    }
}

#[test]
fn a_one_line_budget_cuts_a_long_cue_into_one_cue_per_line() {
    let mut engine = Engine::new(EngineConfig::default()).expect("engine");
    let group = group_with_one_long_cue(&mut engine, None);

    reflow_cues(&mut engine, group, layout(7, 1));

    assert_eq!(
        cue_texts(&engine, group),
        vec!["alpha", "bravo", "charlie", "delta"]
    );
}

#[test]
fn a_two_line_budget_halves_the_cue_and_keeps_two_lines_each() {
    let mut engine = Engine::new(EngineConfig::default()).expect("engine");
    let group = group_with_one_long_cue(&mut engine, None);

    reflow_cues(&mut engine, group, layout(7, 2));

    assert_eq!(
        cue_texts(&engine, group),
        vec!["alpha\nbravo", "charlie\ndelta"]
    );
}

#[test]
fn a_budget_the_cue_already_fits_only_rewraps_it() {
    let mut engine = Engine::new(EngineConfig::default()).expect("engine");
    let group = group_with_one_long_cue(&mut engine, None);

    reflow_cues(&mut engine, group, layout(64, 2));

    assert_eq!(cue_texts(&engine, group), vec!["alpha bravo charlie delta"]);
}

#[test]
fn cut_cues_keep_the_word_timings_that_belong_to_them() {
    let mut engine = Engine::new(EngineConfig::default()).expect("engine");
    let group = group_with_one_long_cue(&mut engine, None);

    reflow_cues(&mut engine, group, layout(7, 1));

    let cues = engine.project().timeline().caption_cues(group);
    for (clip, expected) in cues.iter().zip(["alpha", "bravo", "charlie", "delta"]) {
        let cue = clip.caption.as_ref().expect("cue metadata");
        let text = clip.text_content().expect("cue text");
        assert_eq!(cue.words.len(), 1, "{expected} kept one word");
        assert_eq!(cue.words[0].text(text), expected, "word range follows text");
        assert_eq!(cue.words[0].start_ms, 0, "timings rebase onto their cue");
    }
}

#[test]
fn an_animated_template_still_gets_cut_to_the_line_budget() {
    let mut engine = Engine::new(EngineConfig::default()).expect("engine");
    // Karaoke pop puts a per-character entrance on every cue, and each cut
    // piece is shorter than that entrance's window — the case a general split
    // refuses, which used to leave half the group over its line budget.
    let group = group_with_one_long_cue(&mut engine, Some("karaoke_pop"));

    reflow_cues(&mut engine, group, layout(7, 1));

    assert_eq!(
        cue_texts(&engine, group),
        vec!["alpha", "bravo", "charlie", "delta"]
    );
    for clip in engine.project().timeline().caption_cues(group) {
        assert!(
            clip.animation_in.is_some(),
            "every line still pops in on its own"
        );
    }
}

#[test]
fn a_busy_text_lane_is_not_offered_to_an_overlapping_batch() {
    let mut engine = Engine::new(EngineConfig::default()).expect("engine");
    lane_with_cue(&mut engine, 0, 60);

    let batch = cues(&engine, &[(30, 30), (90, 30)]);
    assert_eq!(free_text_lane(&engine, &batch), None);
}

#[test]
fn a_batch_that_clears_the_existing_cues_reuses_the_lane() {
    let mut engine = Engine::new(EngineConfig::default()).expect("engine");
    let lane = lane_with_cue(&mut engine, 0, 60);

    let batch = cues(&engine, &[(60, 30), (120, 30)]);
    assert_eq!(free_text_lane(&engine, &batch), Some(lane));
}

#[test]
fn a_second_batch_falls_through_to_the_free_lane() {
    let mut engine = Engine::new(EngineConfig::default()).expect("engine");
    let busy = lane_with_cue(&mut engine, 0, 60);
    let free = create_track(&mut engine, TrackKind::Text, 0).expect("second text lane");

    let batch = cues(&engine, &[(0, 60)]);
    let picked = free_text_lane(&engine, &batch);
    assert_eq!(picked, Some(free));
    assert_ne!(picked, Some(busy));
}
