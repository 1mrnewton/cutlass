//! Word highlight (karaoke), resolve → realize → composite.
//!
//! The resolve half asserts the sampled span; the GPU half asserts the
//! highlight actually lands on the right letters, by checking that the
//! highlight-colored ink moves across the line as the playhead advances.

use cutlass_core::{Rational, RationalTime};
use cutlass_models::{
    CaptionCueSpec, CaptionGroupSpec, CaptionHighlight, CaptionHighlightMode, CaptionStyle,
    CaptionWord, ClipId, Project, TextCase, TextStyle as ModelTextStyle, TimeRange, TrackKind,
};
use cutlass_render::{LayerSource, Renderer, resolve};

const FPS_24: Rational = Rational::FPS_24;
const TEXT: &str = "one two three";
const HIGHLIGHT_FILL: [u8; 4] = [255, 0, 0, 255];
const PLATE: [u8; 4] = [0, 0, 255, 255];

fn rt(value: i64) -> RationalTime {
    RationalTime::new(value, FPS_24)
}

/// "one two three" as one cue spanning 2s, a word every 500ms.
fn words() -> Vec<CaptionWord> {
    vec![
        CaptionWord::new(0, 400, 0..3),
        CaptionWord::new(500, 900, 4..7),
        CaptionWord::new(1_000, 1_900, 8..13),
    ]
}

/// A project with one highlighted caption cue on a text lane, and that cue's
/// clip id.
fn captioned(highlight: CaptionHighlight, case: TextCase) -> (Project, ClipId) {
    let mut project = Project::new("captions", FPS_24);
    let track = project.add_track(TrackKind::Text, "T1");
    let spec = CaptionGroupSpec {
        style: Some(CaptionStyle {
            text: ModelTextStyle {
                size: 120.0.into(),
                fill: [255, 255, 255, 255].into(),
                case,
                ..ModelTextStyle::default()
            },
            ..CaptionStyle::default()
        }),
        highlight: Some(highlight),
        ..CaptionGroupSpec::manual(track, "Captions")
    };
    let cue = CaptionCueSpec::new(TEXT, TimeRange::at_rate(0, 48, FPS_24)).with_words(words());
    let (_, clips) = project.add_caption_group(&spec, &[cue]).unwrap();
    (project, clips[0])
}

/// The highlight sampled from the only text layer at `tick`.
fn sampled(project: &Project, tick: i64) -> Option<cutlass_render::TextHighlight> {
    let scene = resolve(project, rt(tick)).unwrap();
    match &scene.layers[0].source {
        LayerSource::Text { highlight, .. } => highlight.clone(),
        other => panic!("expected a text layer, got {other:?}"),
    }
}

#[test]
fn word_mode_tracks_the_spoken_word() {
    let (project, _) = captioned(CaptionHighlight::word(HIGHLIGHT_FILL), TextCase::Normal);
    // 24 fps: frame 0 is 0ms, 12 is 500ms, 24 is 1000ms.
    assert_eq!(sampled(&project, 0).expect("first word").range, 0..3);
    assert_eq!(sampled(&project, 12).expect("second word").range, 4..7);
    assert_eq!(sampled(&project, 24).expect("third word").range, 8..13);
    // A gap after a word holds it rather than flickering off.
    assert_eq!(sampled(&project, 11).expect("gap holds").range, 0..3);
}

#[test]
fn line_mode_fills_from_the_start_of_the_cue() {
    let highlight = CaptionHighlight {
        mode: CaptionHighlightMode::Line,
        ..CaptionHighlight::word(HIGHLIGHT_FILL)
    };
    let (project, _) = captioned(highlight, TextCase::Normal);
    assert_eq!(sampled(&project, 0).expect("first word").range, 0..3);
    assert_eq!(sampled(&project, 12).expect("through second").range, 0..7);
    assert_eq!(sampled(&project, 24).expect("whole line").range, 0..13);
}

#[test]
fn an_off_highlight_and_a_cue_without_timings_sample_nothing() {
    let (project, _) = captioned(CaptionHighlight::default(), TextCase::Normal);
    assert!(
        sampled(&project, 12).is_none(),
        "off mode highlights nothing"
    );

    let (mut project, clip) = captioned(CaptionHighlight::word(HIGHLIGHT_FILL), TextCase::Normal);
    project
        .set_caption_cue(clip, TEXT.into(), Some(Vec::new()), None)
        .unwrap();
    assert!(
        sampled(&project, 12).is_none(),
        "a cue with no word timings renders plainly"
    );
}

#[test]
fn a_cased_run_shifts_the_highlight_onto_the_shaped_text() {
    // The "ﬁ" ligature is three bytes and upper-cases to two ("FI"), so every
    // span after it sits one byte earlier in the text the rasterizer shapes.
    let (mut project, clip) = captioned(CaptionHighlight::word(HIGHLIGHT_FILL), TextCase::Upper);
    project
        .set_caption_cue(
            clip,
            "ﬁre two three".into(),
            Some(vec![
                CaptionWord::new(0, 400, 0..5),
                CaptionWord::new(500, 900, 6..9),
                CaptionWord::new(1_000, 1_900, 10..15),
            ]),
            None,
        )
        .unwrap();
    assert_eq!(sampled(&project, 12).expect("second word").range, 5..8);
}

/// Horizontal centroid of pixels dominated by `rgba`'s red-or-blue signature,
/// and how many there were.
fn ink_centroid(frame: &cutlass_render::RgbaImage, of: impl Fn([u8; 4]) -> bool) -> (f32, usize) {
    let mut sum = 0.0;
    let mut count = 0;
    for (i, px) in frame.pixels.chunks_exact(4).enumerate() {
        if of([px[0], px[1], px[2], px[3]]) {
            sum += (i as u32 % frame.width) as f32;
            count += 1;
        }
    }
    (if count == 0 { 0.0 } else { sum / count as f32 }, count)
}

fn is_red(px: [u8; 4]) -> bool {
    px[0] > 150 && px[1] < 90 && px[2] < 90
}

fn is_blue(px: [u8; 4]) -> bool {
    px[2] > 150 && px[0] < 90 && px[1] < 90
}

#[test]
fn the_highlight_fill_and_plate_travel_along_the_line() {
    let highlight = CaptionHighlight {
        plate: Some(PLATE),
        plate_radius: 0.4,
        scale: 1.15,
        ..CaptionHighlight::word(HIGHLIGHT_FILL)
    };
    let (project, _) = captioned(highlight, TextCase::Normal);

    let mut renderer = match Renderer::new_headless() {
        Ok(renderer) => renderer,
        Err(e) => {
            eprintln!("skipping caption highlight composite: no GPU ({e})");
            return;
        }
    };
    let first = renderer.render_frame(&project, rt(0)).expect("first word");
    let last = renderer.render_frame(&project, rt(24)).expect("third word");

    let (first_x, first_lit) = ink_centroid(&first, is_red);
    let (last_x, last_lit) = ink_centroid(&last, is_red);
    assert!(
        first_lit > 0,
        "the active word should be painted in the highlight fill"
    );
    assert!(
        last_lit > 0,
        "the highlight should survive to the last word"
    );
    assert!(
        last_x > first_x + 20.0,
        "highlight should move right across the line ({first_x} → {last_x})"
    );

    let (plate_x, plate_px) = ink_centroid(&first, is_blue);
    assert!(plate_px > 0, "the plate should be painted behind the word");
    assert!(
        (plate_x - first_x).abs() < 80.0,
        "the plate should sit under the highlighted word ({plate_x} vs {first_x})"
    );
}
