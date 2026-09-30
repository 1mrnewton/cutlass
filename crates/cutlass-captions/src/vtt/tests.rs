use super::*;

const SAMPLE: &str = "WEBVTT\n\
\n\
NOTE this file was exported by hand\n\
\n\
intro\n\
00:00:01.000 --> 00:00:02.500 align:start position:50%\n\
Hello there\n\
\n\
00:00:03.000 --> 00:00:04.000\n\
Two lines\n\
here\n";

#[test]
fn parses_a_file_with_notes_identifiers_and_cue_settings() {
    let cues = parse_vtt(SAMPLE).unwrap();
    assert_eq!(cues.len(), 2);
    assert_eq!(cues[0], SubtitleCue::new(1_000, 2_500, "Hello there"));
    assert_eq!(cues[1].text, "Two lines\nhere");
}

#[test]
fn skips_style_and_region_blocks() {
    let styled = "WEBVTT\n\n\
                  STYLE\n::cue { color: yellow }\n\n\
                  REGION\nid:top width:40%\n\n\
                  00:00:01.000 --> 00:00:02.000\nreal\n";
    let cues = parse_vtt(styled).unwrap();
    assert_eq!(cues.len(), 1);
    assert_eq!(cues[0].text, "real");
}

#[test]
fn strips_voice_and_karaoke_tags() {
    let tagged = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n\
                  <v Ada><00:00:01.200>Hello <c.yellow>world</c>\n";
    assert_eq!(parse_vtt(tagged).unwrap()[0].text, "Hello world");
}

#[test]
fn accepts_hour_less_timestamps() {
    let short = "WEBVTT\n\n01:02.500 --> 01:04.000\ntext\n";
    assert_eq!(parse_vtt(short).unwrap()[0].start_ms, 62_500);
}

#[test]
fn accepts_a_header_with_a_trailing_description() {
    let described = "WEBVTT - Captions\n\n00:00:01.000 --> 00:00:02.000\ntext\n";
    assert_eq!(parse_vtt(described).unwrap().len(), 1);
}

#[test]
fn a_missing_header_is_rejected() {
    let srt_shaped = "1\n00:00:01,000 --> 00:00:02,000\ntext\n";
    let error = parse_vtt(srt_shaped).unwrap_err();
    assert!(
        matches!(error, CaptionError::Parse { line: 1, .. }),
        "{error:?}"
    );
}

#[test]
fn an_empty_file_is_rejected() {
    assert!(parse_vtt("").is_err());
}

#[test]
fn a_header_with_no_cues_parses_to_nothing() {
    assert!(parse_vtt("WEBVTT\n\n").unwrap().is_empty());
}

#[test]
fn writes_a_header_and_dot_separated_cues() {
    let written = write_vtt(&[SubtitleCue::new(1_000, 2_500, "Hello")]);
    assert_eq!(
        written,
        "WEBVTT\n\n1\n00:00:01.000 --> 00:00:02.500\nHello\n\n"
    );
}

#[test]
fn round_trips_through_write_and_parse() {
    let cues = parse_vtt(SAMPLE).unwrap();
    assert_eq!(parse_vtt(&write_vtt(&cues)).unwrap(), cues);
}

#[test]
fn a_comma_separated_timestamp_is_still_read() {
    // Files in the wild mix the separators; rejecting this helps nobody.
    let mixed = "WEBVTT\n\n00:00:01,000 --> 00:00:02,000\ntext\n";
    assert_eq!(parse_vtt(mixed).unwrap()[0].start_ms, 1_000);
}
