//! Active-word highlight (karaoke) for caption runs.
//!
//! A highlight recolors the bytes being spoken and optionally sets a plate
//! behind them. Rather than repainting one run per active word, the run is
//! painted **twice** — once in the resting fill, once in the highlight fill —
//! and the caller picks per cluster. Both paints are memoized by
//! [`TextRenderer`](crate::TextRenderer), so a whole cue costs two paints no
//! matter how many words it contains, and a compositor can key its glyph atlas
//! on the run instead of rebuilding it every time the word moves.

use std::ops::Range;

use cutlass_core::RgbaImage;

use crate::animated::AnimatedText;
use crate::effects::{CardRect, fill_rounded_rect};
use crate::style::TextStyle;
use crate::{ClusterBox, ShapedText};

/// Plate box relative to the line's baseline, as fractions of the font size.
/// Cap height and descender vary per face, so a fixed pair keeps every word on
/// a line the same height — a plate that grew for "gh" and shrank for "so"
/// would jitter its way through a sentence.
const PLATE_ABOVE_BASELINE: f32 = 0.82;
const PLATE_BELOW_BASELINE: f32 = 0.24;
/// Horizontal breathing room around the highlighted ink.
const PLATE_SIDE_PAD: f32 = 0.14;

/// Defensive ceiling on a plate's edge, mirroring the effect-extent cap: this
/// crate's API is public, and a corrupt font size must not turn into a
/// multi-gigabyte allocation.
const MAX_PLATE_EDGE: f32 = 8192.0;

/// Which bytes of a run are highlighted, and how they are painted.
#[derive(Debug, Clone, PartialEq)]
pub struct Highlight {
    /// Byte range into the run's text. Empty highlights nothing.
    pub range: Range<usize>,
    /// Fill for the highlighted clusters.
    pub fill: [u8; 4],
    /// Plate painted behind them, or `None` for a pure color swap.
    pub plate: Option<HighlightPlate>,
}

impl Highlight {
    /// A plain color swap over `range`.
    pub fn new(range: Range<usize>, fill: [u8; 4]) -> Self {
        Self {
            range,
            fill,
            plate: None,
        }
    }

    pub fn with_plate(mut self, plate: HighlightPlate) -> Self {
        self.plate = Some(plate);
        self
    }
}

/// The card drawn behind the highlighted word.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HighlightPlate {
    pub rgba: [u8; 4],
    /// Corner rounding, `0.0` (square) ..= `1.0` (pill), of the shorter side.
    pub radius: f32,
}

/// A run painted for word highlighting.
///
/// `lit` and `covered` are index-aligned with `rest.clusters`, so a caller can
/// pick a cluster's resting or highlighted bitmap without re-measuring
/// anything.
#[derive(Debug, Clone, PartialEq)]
pub struct HighlightedText {
    /// The run in its resting fill (stroke/shadow folded in, plus the
    /// whole-run background card).
    pub rest: AnimatedText,
    /// Every cluster painted in the highlight fill.
    pub lit: Vec<RgbaImage>,
    /// Whether each cluster falls inside the highlight.
    pub covered: Vec<bool>,
    /// Plate behind the covered clusters, with its top-left offset in the same
    /// space as the cluster offsets.
    pub plate: Option<(RgbaImage, [f32; 2])>,
}

impl HighlightedText {
    /// Whether any cluster is highlighted.
    pub fn has_highlight(&self) -> bool {
        self.covered.iter().any(|covered| *covered)
    }
}

/// Combine two paints of the same run into a highlightable one.
///
/// `rest` and `lit` must be paints of the same text and style differing only in
/// fill; a mismatch (which would mean the two runs shaped differently) yields a
/// result with nothing highlighted rather than mismatched glyphs.
pub(crate) fn paint_highlighted(
    shaped: &ShapedText,
    rest: AnimatedText,
    lit: AnimatedText,
    style: &TextStyle,
    highlight: &Highlight,
) -> HighlightedText {
    let aligned =
        rest.clusters.len() == lit.clusters.len() && shaped.clusters.len() == rest.clusters.len();
    let covered: Vec<bool> = if aligned && !highlight.range.is_empty() {
        shaped
            .clusters
            .iter()
            .map(|cluster| overlaps(&cluster.text_range, &highlight.range))
            .collect()
    } else {
        vec![false; rest.clusters.len()]
    };

    let plate = highlight
        .plate
        .filter(|_| covered.iter().any(|covered| *covered))
        .and_then(|plate| paint_plate(shaped, &covered, style, plate));

    HighlightedText {
        rest,
        lit: lit.clusters.into_iter().map(|c| c.image).collect(),
        covered,
        plate,
    }
}

/// Whether a cluster is part of the highlighted span. A ligature straddling
/// the boundary highlights whole — it is one indivisible bitmap.
fn overlaps(cluster: &Range<usize>, highlight: &Range<usize>) -> bool {
    if cluster.is_empty() {
        return highlight.contains(&cluster.start);
    }
    cluster.start < highlight.end && highlight.start < cluster.end
}

/// Paint the plate behind the covered clusters on their own line.
///
/// A wrapped word (rare, but hyphenation and CJK allow it) plates only its
/// first line: one box spanning two lines would cover the text between them.
fn paint_plate(
    shaped: &ShapedText,
    covered: &[bool],
    style: &TextStyle,
    plate: HighlightPlate,
) -> Option<(RgbaImage, [f32; 2])> {
    if plate.rgba[3] == 0 {
        return None;
    }
    let lit = || {
        shaped
            .clusters
            .iter()
            .zip(covered)
            .filter_map(|(cluster, covered)| covered.then_some(cluster))
    };
    let line = lit().next()?.line;
    let mut box_ = lit()
        .filter(|cluster| cluster.line == line)
        .filter(|cluster| cluster.image.width > 0)
        .map(ink_span)
        .reduce(|a, b| (a.0.min(b.0), a.1.max(b.1)))?;

    let font_size = style.font_size.max(1.0);
    let pad = font_size * PLATE_SIDE_PAD;
    box_ = (box_.0 - pad, box_.1 + pad);
    let baseline = lit().find(|cluster| cluster.line == line)?.baseline;
    let top = baseline - font_size * PLATE_ABOVE_BASELINE;
    let bottom = baseline + font_size * PLATE_BELOW_BASELINE;

    let width = (box_.1 - box_.0).clamp(0.0, MAX_PLATE_EDGE).ceil() as u32;
    let height = (bottom - top).clamp(0.0, MAX_PLATE_EDGE).ceil() as u32;
    if width == 0 || height == 0 {
        return None;
    }
    let mut pixels = vec![0u8; (width as usize) * (height as usize) * 4];
    fill_rounded_rect(
        &mut pixels,
        width,
        height,
        CardRect {
            x0: 0.0,
            y0: 0.0,
            x1: width as f32,
            y1: height as f32,
        },
        plate.radius,
        plate.rgba,
    );
    Some((RgbaImage::new(width, height, pixels), [box_.0, top]))
}

/// Horizontal ink span of one cluster in run space.
fn ink_span(cluster: &ClusterBox) -> (f32, f32) {
    (
        cluster.offset[0],
        cluster.offset[0] + cluster.image.width as f32,
    )
}

#[cfg(test)]
mod tests;
