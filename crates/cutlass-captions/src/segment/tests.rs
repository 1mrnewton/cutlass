use super::*;
use cutlass_models::Rational;

fn words(spec: &[(u32, u32, &str)]) -> Vec<TimedWord> {
    spec.iter()
        .map(|&(start, end, text)| TimedWord::new(start, end, text))
        .collect()
}

fn options(layout: CaptionLayout) -> SegmentOptions {
    SegmentOptions::new(Placement::at_rate(Rational::FPS_30)).with_layout(layout)
}

/// One short line per cue, so character packing is the only rule in play.
fn one_line(max_chars_per_line: u16) -> CaptionLayout {
    CaptionLayout {
        max_chars_per_line,
        max_lines: 1,
        ..CaptionLayout::default()
    }
}

fn texts(specs: &[CaptionCueSpec]) -> Vec<&str> {
    specs.iter().map(|spec| spec.text.as_str()).collect()
}

#[test]
fn no_words_yields_no_cues() {
    let specs = segment(&[], &options(CaptionLayout::default())).unwrap();
    assert!(specs.is_empty());
}

#[test]
fn words_pack_up_to_the_character_limit() {
    let specs = segment(
        &words(&[
            (0, 200, "the"),
            (200, 400, "quick"),
            (400, 600, "brown"),
            (600, 800, "fox"),
        ]),
        &options(one_line(10)),
    )
    .unwrap();
    assert_eq!(texts(&specs), ["the quick", "brown fox"]);
}

#[test]
fn a_full_cue_wraps_onto_its_second_line() {
    let layout = CaptionLayout {
        max_chars_per_line: 10,
        max_lines: 2,
        ..CaptionLayout::default()
    };
    let specs = segment(
        &words(&[
            (0, 200, "the"),
            (200, 400, "quick"),
            (400, 600, "brown"),
            (600, 800, "fox"),
            (800, 1_000, "jumped"),
        ]),
        &options(layout),
    )
    .unwrap();
    assert_eq!(texts(&specs), ["the quick\nbrown fox", "jumped"]);
}

#[test]
fn a_sentence_end_starts_a_new_cue() {
    let specs = segment(
        &words(&[(0, 200, "Stop."), (200, 400, "Go")]),
        &options(CaptionLayout::default()),
    )
    .unwrap();
    assert_eq!(texts(&specs), ["Stop.", "Go"]);
}

#[test]
fn a_sentence_end_behind_a_quote_still_breaks() {
    let specs = segment(
        &words(&[(0, 200, "\"Stop!\""), (200, 400, "Go")]),
        &options(CaptionLayout::default()),
    )
    .unwrap();
    assert_eq!(specs.len(), 2);
}

#[test]
fn sentence_breaking_can_be_turned_off() {
    let mut opts = options(CaptionLayout::default());
    opts.break_on_sentence = false;
    let specs = segment(&words(&[(0, 200, "Stop."), (200, 400, "Go")]), &opts).unwrap();
    assert_eq!(texts(&specs), ["Stop. Go"]);
}

#[test]
fn a_decimal_point_mid_word_does_not_break() {
    let specs = segment(
        &words(&[(0, 200, "3.5"), (200, 400, "seconds")]),
        &options(CaptionLayout::default()),
    )
    .unwrap();
    assert_eq!(texts(&specs), ["3.5 seconds"]);
}

#[test]
fn a_pause_starts_a_new_cue() {
    let specs = segment(
        &words(&[(0, 200, "one"), (1_000, 1_200, "two")]),
        &options(CaptionLayout::default()),
    )
    .unwrap();
    assert_eq!(texts(&specs), ["one", "two"]);
}

#[test]
fn pause_breaking_can_be_turned_off() {
    let mut opts = options(CaptionLayout::default());
    opts.pause_break_ms = 0;
    let specs = segment(&words(&[(0, 200, "one"), (9_000, 9_200, "two")]), &opts).unwrap();
    assert_eq!(
        texts(&specs),
        ["one", "two"],
        "the max duration still splits it"
    );
}

#[test]
fn the_max_duration_splits_a_long_run_of_speech() {
    let layout = CaptionLayout {
        max_chars_per_line: 200,
        max_duration_ms: 2_000,
        ..CaptionLayout::default()
    };
    // Words 100 ms apart with no pauses and no punctuation: only duration can
    // break this.
    let spoken: Vec<TimedWord> = (0..40)
        .map(|i| TimedWord::new(i * 100, i * 100 + 100, "word"))
        .collect();
    let specs = segment(&spoken, &options(layout)).unwrap();
    assert!(specs.len() >= 2, "expected a duration split, got {specs:?}");
    for spec in &specs {
        let ms = (spec.timeline.duration.value as f64 * 1000.0 / 30.0).round() as u32;
        assert!(ms <= 2_000, "cue ran {ms} ms");
    }
}

#[test]
fn a_quick_word_is_held_for_the_minimum_duration() {
    let specs = segment(
        &words(&[(0, 80, "hi")]),
        &options(CaptionLayout {
            min_duration_ms: 600,
            ..CaptionLayout::default()
        }),
    )
    .unwrap();
    assert_eq!(specs[0].timeline.duration.value, 18, "600 ms at 30 fps");
}

#[test]
fn a_held_cue_yields_to_the_next_one() {
    // Two sentences 200 ms apart: the first cannot take its full 600 ms hold.
    let specs = segment(
        &words(&[(0, 100, "Hi."), (200, 300, "Bye.")]),
        &options(CaptionLayout::default()),
    )
    .unwrap();
    assert_eq!(specs.len(), 2);
    assert!(
        specs[0].timeline.end_tick() <= specs[1].timeline.start.value,
        "{specs:?}"
    );
}

#[test]
fn cues_are_ascending_and_never_overlap() {
    let spoken: Vec<TimedWord> = (0..200)
        .map(|i| {
            let start = i * 137;
            TimedWord::new(start, start + 90, if i % 5 == 0 { "end." } else { "word" })
        })
        .collect();
    let specs = segment(&spoken, &options(CaptionLayout::default())).unwrap();
    assert!(specs.len() > 10);
    let mut previous_end = i64::MIN;
    for spec in &specs {
        assert!(spec.timeline.start.value >= previous_end, "{spec:?}");
        assert!(spec.timeline.duration.value >= 1);
        previous_end = spec.timeline.end_tick();
    }
}

#[test]
fn word_ranges_slice_the_cue_text_and_start_clip_relative() {
    let specs = segment(
        &words(&[(500, 700, "the"), (700, 900, "quick"), (900, 1_100, "fox")]),
        &options(one_line(32)),
    )
    .unwrap();
    let spec = &specs[0];
    assert_eq!(spec.text, "the quick fox");
    let sliced: Vec<&str> = spec.words.iter().map(|w| w.text(&spec.text)).collect();
    assert_eq!(sliced, ["the", "quick", "fox"]);
    assert_eq!(spec.words[0].start_ms, 0, "rebased onto the placed clip");
    assert!(spec.words[2].end_ms >= 550, "{:?}", spec.words);
}

#[test]
fn word_ranges_survive_a_multi_line_cue() {
    let layout = CaptionLayout {
        max_chars_per_line: 10,
        max_lines: 2,
        ..CaptionLayout::default()
    };
    let specs = segment(
        &words(&[
            (0, 200, "the"),
            (200, 400, "quick"),
            (400, 600, "brown"),
            (600, 800, "fox"),
        ]),
        &options(layout),
    )
    .unwrap();
    let spec = &specs[0];
    assert_eq!(spec.text, "the quick\nbrown fox");
    let sliced: Vec<&str> = spec.words.iter().map(|w| w.text(&spec.text)).collect();
    assert_eq!(sliced, ["the", "quick", "brown", "fox"]);
}

#[test]
fn confidence_averages_the_words_that_have_one() {
    let spoken = vec![
        TimedWord::new(0, 100, "sure").with_confidence(1.0),
        TimedWord::new(100, 200, "maybe").with_confidence(0.5),
    ];
    let specs = segment(&spoken, &options(one_line(32))).unwrap();
    assert!((specs[0].confidence.unwrap() - 0.75).abs() < 1e-6);
}

#[test]
fn confidence_is_absent_when_no_word_carries_one() {
    let specs = segment(&words(&[(0, 100, "hi")]), &options(one_line(32))).unwrap();
    assert_eq!(specs[0].confidence, None);
}

#[test]
fn blank_words_are_dropped() {
    let specs = segment(
        &words(&[(0, 100, "  "), (100, 200, " hi "), (200, 300, "")]),
        &options(one_line(32)),
    )
    .unwrap();
    assert_eq!(texts(&specs), ["hi"]);
}

#[test]
fn backwards_word_times_are_forced_forward() {
    let specs = segment(
        &words(&[(500, 700, "one"), (100, 200, "two")]),
        &options(one_line(32)),
    )
    .unwrap();
    assert_eq!(texts(&specs), ["one two"]);
    let table = &specs[0].words;
    assert!(table[1].start_ms >= table[0].end_ms, "{table:?}");
}

#[test]
fn an_overlong_word_gets_a_cue_to_itself() {
    let specs = segment(
        &words(&[
            (0, 200, "a"),
            (200, 400, "supercalifragilisticexpialidocious"),
        ]),
        &options(one_line(10)),
    )
    .unwrap();
    assert_eq!(texts(&specs), ["a", "supercalifragilisticexpialidocious"]);
}

#[test]
fn the_offset_places_cues_later_on_the_timeline() {
    let mut opts = options(one_line(32));
    opts.placement = Placement::new(Rational::FPS_30, 300);
    let specs = segment(&words(&[(0, 1_000, "hi")]), &opts).unwrap();
    assert_eq!(specs[0].timeline.start.value, 300);
    assert_eq!(specs[0].words[0].start_ms, 0);
}

#[test]
fn an_invalid_layout_is_rejected() {
    let layout = CaptionLayout {
        max_lines: 0,
        ..CaptionLayout::default()
    };
    assert!(matches!(
        segment(&words(&[(0, 100, "hi")]), &options(layout)),
        Err(CaptionError::Rules(_))
    ));
}

#[test]
fn more_cues_than_a_group_holds_is_rejected() {
    // A pause after every word makes one cue per word.
    let spoken: Vec<TimedWord> = (0..=MAX_CAPTION_CUES as u32)
        .map(|i| TimedWord::new(i * 1_000, i * 1_000 + 100, "word"))
        .collect();
    assert!(matches!(
        segment(&spoken, &options(one_line(32))),
        Err(CaptionError::TooManyCues { .. })
    ));
}
