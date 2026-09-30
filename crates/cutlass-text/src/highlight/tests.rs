use super::*;
use crate::style::TextStroke;
use crate::{TextRenderer, TextStyle};

const TEST_FONT: &[u8] = include_bytes!("../../assets/Micro5-Regular.ttf");

fn renderer() -> TextRenderer {
    let mut renderer = TextRenderer::new();
    assert!(renderer.load_font(TEST_FONT.to_vec()) > 0);
    renderer
}

const TEXT: &str = "one two";
/// Byte range of "two".
const SECOND_WORD: Range<usize> = 4..7;

fn style() -> TextStyle {
    TextStyle::new(48.0).with_color([255, 255, 255, 255])
}

/// The clusters the highlight covers, as source text.
fn covered_text(renderer: &mut TextRenderer, painted: &HighlightedText) -> String {
    let shaped = renderer.shape(TEXT, &style());
    shaped
        .clusters
        .iter()
        .zip(&painted.covered)
        .filter(|(_, covered)| **covered)
        .map(|(cluster, _)| &TEXT[cluster.text_range.clone()])
        .collect()
}

/// Whether any pixel of `image` is dominated by red.
fn is_red(image: &RgbaImage) -> bool {
    image
        .pixels
        .chunks_exact(4)
        .filter(|pixel| pixel[3] > 200)
        .all(|pixel| pixel[0] > pixel[1] && pixel[0] > pixel[2])
}

#[test]
fn only_the_active_word_is_covered() {
    let mut renderer = renderer();
    let painted = renderer.paint_highlighted(
        TEXT,
        &style(),
        &Highlight::new(SECOND_WORD, [255, 0, 0, 255]),
    );
    assert!(painted.has_highlight());
    assert_eq!(covered_text(&mut renderer, &painted), "two");
}

#[test]
fn the_lit_run_carries_the_highlight_fill() {
    let mut renderer = renderer();
    let painted = renderer.paint_highlighted(
        TEXT,
        &style(),
        &Highlight::new(SECOND_WORD, [255, 0, 0, 255]),
    );
    // Every cluster is painted in both fills — the caller picks, so a glyph
    // atlas keyed on the run stays valid as the active word moves.
    assert_eq!(painted.lit.len(), painted.rest.clusters.len());
    let lit_with_ink = painted
        .lit
        .iter()
        .filter(|image| image.width > 0)
        .peekable()
        .count();
    assert!(lit_with_ink > 0, "no lit cluster had ink");
    for image in painted.lit.iter().filter(|image| image.width > 0) {
        assert!(is_red(image), "lit cluster is not painted in the fill");
    }
    for image in painted
        .rest
        .clusters
        .iter()
        .map(|cluster| &cluster.image)
        .filter(|image| image.width > 0)
    {
        assert!(!is_red(image), "resting cluster took the highlight fill");
    }
}

#[test]
fn an_empty_range_highlights_nothing() {
    let mut renderer = renderer();
    let painted =
        renderer.paint_highlighted(TEXT, &style(), &Highlight::new(4..4, [255, 0, 0, 255]));
    assert!(!painted.has_highlight());
    assert!(painted.plate.is_none());
}

#[test]
fn the_plate_spans_the_highlighted_word_only() {
    let mut renderer = renderer();
    let plate = HighlightPlate {
        rgba: [0, 0, 0, 220],
        radius: 0.5,
    };
    let painted = renderer.paint_highlighted(
        TEXT,
        &style(),
        &Highlight::new(SECOND_WORD, [255, 0, 0, 255]).with_plate(plate),
    );
    let (image, offset) = painted.plate.expect("plate painted");
    assert!(image.is_well_formed());
    assert!(image.pixels.chunks_exact(4).any(|pixel| pixel[3] > 0));

    // The plate starts at the second word, not at the run's origin, and ends
    // before the run does.
    let shaped = renderer.shape(TEXT, &style());
    let word_start = shaped
        .clusters
        .iter()
        .find(|cluster| cluster.text_range.start == SECOND_WORD.start)
        .expect("second word shaped")
        .offset[0];
    assert!(
        offset[0] < word_start && offset[0] > 0.0,
        "plate x {} should sit just left of the word at {word_start}",
        offset[0]
    );
    assert!(
        (offset[0] + image.width as f32) < shaped.extent.0 as f32 + 8.0,
        "plate runs past the text"
    );
    // Tall enough to cover a line regardless of which letters are lit.
    assert!(image.height >= (style().font_size * 0.9) as u32);
}

#[test]
fn a_transparent_plate_paints_nothing() {
    let mut renderer = renderer();
    let painted = renderer.paint_highlighted(
        TEXT,
        &style(),
        &Highlight::new(SECOND_WORD, [255, 0, 0, 255]).with_plate(HighlightPlate {
            rgba: [0, 0, 0, 0],
            radius: 0.0,
        }),
    );
    assert!(painted.has_highlight());
    assert!(painted.plate.is_none());
}

#[test]
fn a_stroked_highlight_keeps_the_outline_in_the_stroke_color() {
    let mut renderer = renderer();
    let style = style().with_stroke(TextStroke {
        rgba: [0, 0, 255, 255],
        width: 6.0,
    });
    let painted =
        renderer.paint_highlighted(TEXT, &style, &Highlight::new(SECOND_WORD, [255, 0, 0, 255]));
    // Painting the lit run separately (rather than recoloring finished pixels)
    // is what keeps the outline blue while the fill turns red.
    let lit = painted
        .lit
        .iter()
        .find(|image| image.width > 0)
        .expect("a lit cluster with ink");
    let blue = lit
        .pixels
        .chunks_exact(4)
        .any(|pixel| pixel[2] > 200 && pixel[0] < 80);
    let red = lit
        .pixels
        .chunks_exact(4)
        .any(|pixel| pixel[0] > 200 && pixel[2] < 80);
    assert!(blue && red, "expected a blue outline around a red fill");
}

#[test]
fn walking_the_words_costs_no_extra_paints() {
    let mut renderer = renderer();
    let style = style();
    for range in [0..3, 4..7, 0..7] {
        let _ = renderer.paint_highlighted(TEXT, &style, &Highlight::new(range, [255, 0, 0, 255]));
    }
    // Two shapes and two paints (resting + highlight fill) for the whole cue,
    // whatever the active word is.
    assert_eq!(renderer.memo_sizes(), (2, 0));
    assert_eq!(renderer.animate_memo.len(), 2);
}
