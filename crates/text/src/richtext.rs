//! The characters layout shapes, and the color of each styled range.
//!
//! A rich title's `content` is JSON holding its words and styled ranges
//! (see [`cutlass_model::RichText`]). Its words are shaped, and a range with
//! a solid fill paints its letters in that color. A rich title whose
//! content does not read is refused instead of being drawn as raw JSON.

use std::borrow::Cow;

use cutlass_model::MaterialText;

use crate::TextError;

/// The words to shape and the ranges that paint their own color.
pub(crate) struct Words<'a> {
    pub(crate) text: Cow<'a, str>,
    pub(crate) runs: Vec<Run>,
}

/// Letters from byte `start` up to `end` painted `fill`, straight RGBA.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Run {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) fill: [u8; 4],
}

/// The fill of the letter at byte `source`, when a range paints it. A
/// later range wins.
pub(crate) fn fill_at(runs: &[Run], source: usize) -> Option<[u8; 4]> {
    runs.iter()
        .rev()
        .find(|run| (run.start..run.end).contains(&source))
        .map(|run| run.fill)
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
        .filter_map(|style| {
            let [red, green, blue, alpha] = style.solid()?;
            let channel = |value: f64| {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let byte = (unit(value) * 255.0).round() as u8;
                byte
            };
            Some(Run {
                start: byte(style.range[0]),
                end: byte(style.range[1]),
                fill: [
                    channel(red),
                    channel(green),
                    channel(blue),
                    channel(alpha * global),
                ],
            })
        })
        .filter(|run| run.start < run.end)
        .collect();
    Ok(Words {
        text: Cow::Owned(rich.text),
        runs,
    })
}

fn unit(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}
