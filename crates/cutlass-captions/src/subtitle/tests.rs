use super::*;
use cutlass_models::Rational;

fn options() -> ImportOptions {
    ImportOptions::new(Placement::at_rate(Rational::FPS_30))
}

#[test]
fn cues_land_on_frames_at_the_file_times() {
    let specs = place_subtitles(
        &[
            SubtitleCue::new(1_000, 2_000, "first"),
            SubtitleCue::new(3_000, 4_000, "second"),
        ],
        &options(),
    )
    .unwrap();
    assert_eq!(specs[0].timeline.start.value, 30);
    assert_eq!(specs[0].timeline.duration.value, 30);
    assert_eq!(specs[1].timeline.start.value, 90);
    assert_eq!(specs[0].text, "first");
}

#[test]
fn the_file_keeps_its_own_line_breaks_by_default() {
    let specs = place_subtitles(
        &[SubtitleCue::new(0, 2_000, "a deliberately\nbroken line")],
        &options(),
    )
    .unwrap();
    assert_eq!(specs[0].text, "a deliberately\nbroken line");
}

#[test]
fn rewrapping_reflows_to_the_layout() {
    let mut opts = options().with_layout(CaptionLayout {
        max_chars_per_line: 10,
        ..CaptionLayout::default()
    });
    opts.rewrap = true;
    let specs =
        place_subtitles(&[SubtitleCue::new(0, 2_000, "the quick brown fox")], &opts).unwrap();
    assert_eq!(specs[0].text, "the quick\nbrown fox");
}

#[test]
fn overlapping_cues_are_pushed_apart() {
    let specs = place_subtitles(
        &[
            SubtitleCue::new(0, 5_000, "long one"),
            SubtitleCue::new(1_000, 2_000, "interrupting"),
        ],
        &options(),
    )
    .unwrap();
    assert!(
        specs[0].timeline.end_tick() <= specs[1].timeline.start.value,
        "{specs:?}"
    );
}

#[test]
fn a_too_short_cue_is_held_for_the_minimum() {
    let specs = place_subtitles(&[SubtitleCue::new(0, 50, "blink")], &options()).unwrap();
    assert_eq!(specs[0].timeline.duration.value, 18, "600 ms at 30 fps");
}

#[test]
fn a_long_cue_keeps_its_own_duration() {
    // Unlike segmentation, an imported file's durations are authoritative: a
    // 20-second cue was timed that way on purpose.
    let specs = place_subtitles(&[SubtitleCue::new(0, 20_000, "held")], &options()).unwrap();
    assert_eq!(specs[0].timeline.duration.value, 600);
}

#[test]
fn word_timings_are_estimated_only_when_asked() {
    let mut opts = options();
    let plain = place_subtitles(&[SubtitleCue::new(0, 2_000, "one two")], &opts).unwrap();
    assert!(plain[0].words.is_empty());

    opts.estimate_words = true;
    let karaoke = place_subtitles(&[SubtitleCue::new(0, 2_000, "one two")], &opts).unwrap();
    assert_eq!(karaoke[0].words.len(), 2);
    assert_eq!(karaoke[0].words[0].text(&karaoke[0].text), "one");
    assert_eq!(karaoke[0].words[0].start_ms, 0, "clip-relative");
}

#[test]
fn estimated_timings_are_clip_relative_after_the_offset() {
    let mut opts = ImportOptions::new(Placement::new(Rational::FPS_30, 300));
    opts.estimate_words = true;
    let specs = place_subtitles(&[SubtitleCue::new(5_000, 6_000, "one two")], &opts).unwrap();
    assert_eq!(specs[0].timeline.start.value, 450, "300 + 5 s at 30 fps");
    assert_eq!(specs[0].words[0].start_ms, 0);
    assert!(specs[0].words[1].end_ms <= 1_000, "{:?}", specs[0].words);
}

#[test]
fn blank_cues_are_dropped() {
    let specs = place_subtitles(
        &[
            SubtitleCue::new(0, 1_000, ""),
            SubtitleCue::new(2_000, 3_000, "real"),
        ],
        &options(),
    )
    .unwrap();
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].text, "real");
}

#[test]
fn nothing_in_nothing_out() {
    assert!(place_subtitles(&[], &options()).unwrap().is_empty());
}

#[test]
fn an_invalid_layout_is_rejected() {
    let opts = options().with_layout(CaptionLayout {
        min_duration_ms: 0,
        ..CaptionLayout::default()
    });
    assert!(place_subtitles(&[SubtitleCue::new(0, 1_000, "hi")], &opts).is_err());
}

#[test]
fn the_format_is_sniffed_from_the_header() {
    let (format, cues) = parse_subtitles("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nhi\n").unwrap();
    assert_eq!(format, CaptionFileFormat::Vtt);
    assert_eq!(cues.len(), 1);

    let (format, cues) = parse_subtitles("1\n00:00:01,000 --> 00:00:02,000\nhi\n").unwrap();
    assert_eq!(format, CaptionFileFormat::Srt);
    assert_eq!(cues.len(), 1);
}

#[test]
fn clips_convert_back_into_subtitle_cues_in_order() {
    let rate = Rational::FPS_30;
    let placement = Placement::new(rate, 30);
    let cues = subtitles_from_clips(
        [
            (TimeRange::at_rate(120, 30, rate), "second".to_owned()),
            (TimeRange::at_rate(30, 60, rate), "first".to_owned()),
        ],
        placement,
    );
    assert_eq!(
        cues,
        [
            SubtitleCue::new(0, 2_000, "first"),
            SubtitleCue::new(3_000, 4_000, "second"),
        ]
    );
}

#[test]
fn a_placed_group_survives_an_srt_round_trip() {
    let rate = Rational::FPS_30;
    let placement = Placement::at_rate(rate);
    let original = [
        SubtitleCue::new(1_000, 2_000, "first"),
        SubtitleCue::new(3_000, 4_000, "second line"),
    ];
    let specs = place_subtitles(&original, &options()).unwrap();
    let exported = subtitles_from_clips(
        specs
            .iter()
            .map(|spec| (spec.timeline, spec.text.clone()))
            .collect::<Vec<_>>(),
        placement,
    );
    assert_eq!(exported, original);
    assert_eq!(
        crate::parse_srt(&crate::write_srt(&exported)).unwrap(),
        original
    );
}
