use super::*;

const SAMPLE: &str = "1\n\
00:00:01,000 --> 00:00:02,500\n\
Hello there\n\
\n\
2\n\
00:00:03,000 --> 00:00:04,000\n\
Two lines\n\
here\n";

#[test]
fn parses_a_plain_file() {
    let cues = parse_srt(SAMPLE).unwrap();
    assert_eq!(cues.len(), 2);
    assert_eq!(cues[0], SubtitleCue::new(1_000, 2_500, "Hello there"));
    assert_eq!(cues[1].text, "Two lines\nhere");
}

#[test]
fn tolerates_crlf_a_byte_order_mark_and_extra_blank_lines() {
    let messy = "\u{feff}1\r\n00:00:01,000 --> 00:00:02,000\r\nHi\r\n\r\n\r\n\
                 2\r\n00:00:03,000 --> 00:00:04,000\r\nBye\r\n";
    let cues = parse_srt(messy).unwrap();
    assert_eq!(cues.len(), 2);
    assert_eq!(cues[1].text, "Bye");
}

#[test]
fn tolerates_missing_and_wrong_cue_numbers() {
    let odd = "00:00:01,000 --> 00:00:02,000\nfirst\n\n\
               99\n00:00:03,000 --> 00:00:04,000\nsecond\n";
    let cues = parse_srt(odd).unwrap();
    assert_eq!(cues.len(), 2);
    assert_eq!(cues[0].text, "first");
}

#[test]
fn strips_markup_from_cue_text() {
    let styled = "1\n00:00:01,000 --> 00:00:02,000\n{\\an8}<i>Ital&amp;ic</i>\n";
    assert_eq!(parse_srt(styled).unwrap()[0].text, "Ital&ic");
}

#[test]
fn drops_a_cue_with_no_text() {
    let empty = "1\n00:00:01,000 --> 00:00:02,000\n\n2\n00:00:03,000 --> 00:00:04,000\nreal\n";
    let cues = parse_srt(empty).unwrap();
    assert_eq!(cues.len(), 1);
    assert_eq!(cues[0].text, "real");
}

#[test]
fn an_empty_file_has_no_cues() {
    assert!(parse_srt("").unwrap().is_empty());
    assert!(parse_srt("\n\n  \n").unwrap().is_empty());
}

#[test]
fn a_missing_timing_line_reports_its_line_number() {
    let broken = "1\n00:00:01,000 --> 00:00:02,000\nfine\n\n2\nno timing here\n";
    let error = parse_srt(broken).unwrap_err();
    assert!(
        matches!(error, CaptionError::Parse { line: 5, .. }),
        "{error:?}"
    );
}

#[test]
fn an_unreadable_timestamp_reports_its_line_number() {
    let broken = "1\n00:00:01,000 --> banana\ntext\n";
    let error = parse_srt(broken).unwrap_err();
    assert!(
        matches!(error, CaptionError::Parse { line: 2, .. }),
        "{error:?}"
    );
}

#[test]
fn a_backwards_cue_is_rejected() {
    let broken = "1\n00:00:05,000 --> 00:00:02,000\ntext\n";
    assert!(parse_srt(broken).is_err());
}

#[test]
fn writes_numbered_comma_separated_cues() {
    let written = write_srt(&[
        SubtitleCue::new(1_000, 2_500, "Hello there"),
        SubtitleCue::new(3_000, 4_000, "Two lines\nhere"),
    ]);
    assert_eq!(
        written,
        "1\n00:00:01,000 --> 00:00:02,500\nHello there\n\n\
         2\n00:00:03,000 --> 00:00:04,000\nTwo lines\nhere\n\n"
    );
}

#[test]
fn round_trips_through_write_and_parse() {
    let cues = parse_srt(SAMPLE).unwrap();
    assert_eq!(parse_srt(&write_srt(&cues)).unwrap(), cues);
}

#[test]
fn writing_never_emits_a_backwards_cue() {
    let written = write_srt(&[SubtitleCue::new(5_000, 1_000, "oops")]);
    assert!(
        written.contains("00:00:05,000 --> 00:00:05,000"),
        "{written}"
    );
    assert!(parse_srt(&written).is_ok());
}
