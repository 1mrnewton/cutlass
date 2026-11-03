//! The characters layout shapes, and the color of each styled range.
//!
//! A rich title's `content` is JSON holding its words and styled ranges
//! (see [`cutlass_model::RichText`]). Its words are shaped, and a range with
//! a solid fill paints its letters in that color. A range's `size` scales
//! its letters against the title's `font_size`, and its strokes and shadows
//! replace the title's outline and shadow on its letters; only the first of
//! each is drawn. A range's `font` path sets its letters in that face, when
//! the shell has it. A rich title whose content does not read is refused
//! instead of being drawn as raw JSON.

use std::borrow::Cow;

use cutlass_model::MaterialText;

use crate::TextError;
use crate::style::{Outline, Shadow, unit_color};

/// How far a range's size may go from the title's, either way.
const MOST_SCALE: f32 = 10.0;

/// The words to shape and the ranges that paint their own color.
pub(crate) struct Words<'a> {
    pub(crate) text: Cow<'a, str>,
    pub(crate) runs: Vec<Run>,
}

/// Letters from byte `start` up to `end` and how they look. `fill` is
/// straight RGBA; the outline and shadow are in pixels at the title's size,
/// and `None` takes the title's own.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Run {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) fill: Option<[u8; 4]>,
    /// Times the title's size.
    pub(crate) size: f32,
    pub(crate) outline: Option<Option<Outline>>,
    pub(crate) shadow: Option<Option<Shadow>>,
    /// The face file its letters are set in, when not the title's.
    pub(crate) font: Option<String>,
}

/// The range holding the letter at byte `source`. A later range wins.
pub(crate) fn run_at(runs: &[Run], source: usize) -> Option<&Run> {
    runs.iter()
        .rev()
        .find(|run| (run.start..run.end).contains(&source))
}

/// How many times the title's size the letter at byte `source` is.
pub(crate) fn size_at(runs: &[Run], source: usize) -> f32 {
    run_at(runs, source).map_or(1.0, |run| run.size)
}

pub(crate) fn read(material: &MaterialText) -> Result<Words<'_>, TextError> {
    if !material.is_rich_text {
        return Ok(Words {
            text: Cow::Borrowed(material.content.as_str()),
            runs: Vec::new(),
        });
    }
    let rich = material.rich().ok_or(TextError::RichTextCommands)?;
    let global = unit(material.global_alpha);
    #[allow(clippy::cast_possible_truncation)]
    let font_size = material.font_size as f32;
    let bytes: Vec<usize> = rich
        .text
        .char_indices()
        .map(|(at, _)| at)
        .chain([rich.text.len()])
        .collect();
    let byte = |char_index: usize| bytes.get(char_index).copied().unwrap_or(rich.text.len());
    let runs = rich
        .styles
        .iter()
        .map(|style| {
            let channel = |value: f64| {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let byte = (unit(value) * 255.0).round() as u8;
                byte
            };
            let fill = style.solid().map(|[red, green, blue, alpha]| {
                [
                    channel(red),
                    channel(green),
                    channel(blue),
                    channel(alpha * global),
                ]
            });
            let outline = style.strokes.as_ref().map(|strokes| {
                let stroke = strokes.first()?;
                Outline::new(unit_color(stroke.rgba()?, global), stroke.width, font_size)
            });
            let shadow = style.shadows.as_ref().map(|shadows| {
                let shadow = shadows.first()?;
                Shadow::new(
                    unit_color(shadow.rgba()?, global),
                    [shadow.distance, shadow.angle, shadow.feather],
                    font_size,
                )
            });
            Run {
                start: byte(style.range[0]),
                end: byte(style.range[1]),
                fill,
                size: scale(style.size, material.font_size),
                outline,
                shadow,
                font: style
                    .font
                    .as_ref()
                    .map(|font| font.path.trim().to_owned())
                    .filter(|path| !path.is_empty()),
            }
        })
        .filter(|run| run.start < run.end)
        .collect();
    Ok(Words {
        text: Cow::Owned(rich.text),
        runs,
    })
}

/// `size` against the title's `font_size`, or 1 when either is not usable.
fn scale(size: Option<f64>, font_size: f64) -> f32 {
    let Some(size) = size.filter(|size| size.is_finite() && *size > 0.0) else {
        return 1.0;
    };
    if !(font_size.is_finite() && font_size > 0.0) {
        return 1.0;
    }
    #[allow(clippy::cast_possible_truncation)]
    let ratio = (size / font_size) as f32;
    ratio.clamp(1.0 / MOST_SCALE, MOST_SCALE)
}

fn unit(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}
