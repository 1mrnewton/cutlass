use super::pipeline::{Recognized, RecognizedWord, centiseconds_to_ms};
use super::*;

fn word(start_ms: u32, end_ms: u32, text: &str) -> RecognizedWord {
    RecognizedWord {
        start_ms,
        end_ms,
        text: text.to_owned(),
        confidence: Some(0.9),
    }
}

/// A 10-second clip of a longer asset, starting 5 s into the file and landing
/// one second into the sequence at 30 fps.
fn request() -> AutoCaptionRequest {
    AutoCaptionRequest {
        path: PathBuf::from("/tmp/interview.mp4"),
        media: "7".to_owned(),
        name: "interview.mp4".to_owned(),
        start_tick: 30,
        duration_ticks: 300,
        source_in_seconds: 5.0,
        speed: 1.0,
        rate: Rational::new(30, 1),
        language: Some("en".to_owned()),
        template: "clean".to_owned(),
        layout: CaptionLayout::default(),
    }
}

fn cues(request: &AutoCaptionRequest, words: Vec<RecognizedWord>) -> Vec<CaptionCueSpec> {
    let recognized = Recognized {
        words,
        language: None,
        from_cache: false,
    };
    place(request, &recognized).expect("a valid layout places its words")
}

#[test]
fn only_speech_inside_the_clips_window_becomes_cues() {
    let placed = cues(
        &request(),
        vec![
            word(1_000, 1_500, "before"),
            word(5_200, 5_600, "Hello"),
            word(5_700, 6_100, "there."),
            word(40_000, 40_500, "after"),
        ],
    );
    assert_eq!(placed.len(), 1, "{placed:#?}");
    assert_eq!(placed[0].text, "Hello there.");
}

#[test]
fn cues_land_at_the_clips_start_tick_not_the_assets() {
    let placed = cues(&request(), vec![word(5_000, 5_400, "Now.")]);
    assert_eq!(placed[0].timeline.start.value, 30);
}

#[test]
fn asset_time_compresses_by_the_clips_speed() {
    let mut request = request();
    request.speed = 2.0;
    // At 2x the clip plays 20 s of audio in its 10 s, so a word 5 s past the
    // in-point plays 2.5 s (75 ticks) into the clip.
    let placed = cues(&request, vec![word(10_000, 10_400, "Halfway.")]);
    assert_eq!(placed[0].timeline.start.value, 30 + 75);
}

#[test]
fn speech_past_the_clips_out_point_is_dropped() {
    let placed = cues(
        &request(),
        vec![word(5_100, 5_400, "Kept."), word(16_000, 16_400, "Cut.")],
    );
    assert_eq!(placed.len(), 1);
    assert_eq!(placed[0].text, "Kept.");
}

#[test]
fn no_speech_in_the_window_produces_no_cues() {
    assert!(cues(&request(), vec![word(30_000, 30_400, "elsewhere")]).is_empty());
}

#[test]
fn a_zero_length_clip_has_an_empty_window() {
    let mut request = request();
    request.duration_ticks = 0;
    assert_eq!(request.timeline_seconds(), 0.0);
    assert_eq!(request.window_seconds(), (5.0, 5.0));
    assert!(cues(&request, vec![word(5_000, 5_400, "Nothing.")]).is_empty());
}

#[test]
fn an_invalid_rate_yields_no_window_rather_than_a_division_by_zero() {
    let mut request = request();
    request.rate = Rational::new(0, 0);
    assert_eq!(request.timeline_seconds(), 0.0);
    assert!(cues(&request, vec![word(5_000, 5_400, "Nothing.")]).is_empty());
}

#[test]
fn whisper_centiseconds_convert_without_overflowing() {
    assert_eq!(centiseconds_to_ms(0), 0);
    assert_eq!(centiseconds_to_ms(123), 1_230);
    assert_eq!(centiseconds_to_ms(u64::MAX), u32::MAX);
}
