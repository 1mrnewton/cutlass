//! Rich titles: `content` holds JSON with the words and styled ranges.
//!
//! `{"text": "...", "styles": [{"range": [start, end], "fill": {...}, ...}]}`.
//! A range counts characters, start included and end left out. A fill's
//! solid color is three channels from 0 to 1 with an alpha from 0 to 1.
//! `size` is in the title's `font_size` units. A stroke's `width` is in ems,
//! like `border_width`; a shadow's `distance`, `angle`, and `feather` read
//! like `shadow_distance`, `shadow_angle`, and `shadow_smoothing`. A range
//! with no `strokes` or `shadows` key takes the title's own; an empty list
//! has none. Keys this model does not name stay as they were.

use std::borrow::Cow;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::text::MaterialText;

/// The words of a rich title and their styles.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct RichText {
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub styles: Vec<RichStyle>,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// One styled range.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct RichStyle {
    #[serde(default)]
    pub range: [usize; 2],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fill: Option<RichFill>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strokes: Option<Vec<RichStroke>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadows: Option<Vec<RichShadow>>,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// An outline around a range's letters.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct RichStroke {
    #[serde(default = "one")]
    pub alpha: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<RichPaint>,
    #[serde(default)]
    pub width: f64,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// A shadow under a range's letters.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct RichShadow {
    #[serde(default = "one")]
    pub alpha: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<RichPaint>,
    #[serde(default)]
    pub distance: f64,
    #[serde(default)]
    pub angle: f64,
    #[serde(default)]
    pub feather: f64,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct RichFill {
    #[serde(default = "one")]
    pub alpha: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<RichPaint>,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct RichPaint {
    #[serde(default)]
    pub render_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solid: Option<RichSolid>,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct RichSolid {
    #[serde(default)]
    pub color: Vec<f64>,
    #[serde(default = "one")]
    pub alpha: f64,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

const fn one() -> f64 {
    1.0
}

impl RichPaint {
    /// The solid color as red, green, blue, and alpha, each from 0 to 1,
    /// with its alpha times `alpha`.
    #[must_use]
    pub fn rgba(&self, alpha: f64) -> Option<[f64; 4]> {
        let solid = self.solid.as_ref()?;
        let [red, green, blue] = solid.color.get(..3)?.try_into().ok()?;
        Some([red, green, blue, solid.alpha * alpha])
    }
}

impl RichStroke {
    /// See [`RichPaint::rgba`].
    #[must_use]
    pub fn rgba(&self) -> Option<[f64; 4]> {
        self.content.as_ref()?.rgba(self.alpha)
    }
}

impl RichShadow {
    /// See [`RichPaint::rgba`].
    #[must_use]
    pub fn rgba(&self) -> Option<[f64; 4]> {
        self.content.as_ref()?.rgba(self.alpha)
    }
}

impl RichStyle {
    /// The solid fill as red, green, blue, and alpha, each from 0 to 1; the
    /// fill's alpha and the color's multiply.
    #[must_use]
    pub fn solid(&self) -> Option<[f64; 4]> {
        let fill = self.fill.as_ref()?;
        fill.content.as_ref()?.rgba(fill.alpha)
    }

    /// Paint the range solid `rgb`, each from 0 to 1, keeping its alpha.
    pub fn paint(&mut self, rgb: [f64; 3]) {
        let fill = self.fill.get_or_insert_with(|| RichFill {
            alpha: 1.0,
            ..RichFill::default()
        });
        let paint = fill.content.get_or_insert_with(RichPaint::default);
        paint.render_type = "solid".to_owned();
        let solid = paint.solid.get_or_insert_with(|| RichSolid {
            alpha: 1.0,
            ..RichSolid::default()
        });
        solid.color = rgb.to_vec();
    }
}

impl RichText {
    /// Replace the words. Each range is cut to the new length and the last
    /// one reaches its end, so typing after the words takes their style.
    pub fn set_words(&mut self, words: &str) {
        let length = words.chars().count();
        self.text = words.to_owned();
        for style in &mut self.styles {
            style.range = [style.range[0].min(length), style.range[1].min(length)];
        }
        let last = self
            .styles
            .iter()
            .enumerate()
            .max_by_key(|(_, style)| style.range[0])
            .map(|(index, _)| index);
        if let Some(last) = last {
            self.styles[last].range[1] = length;
        }
        let kept: Vec<_> = self
            .styles
            .iter()
            .filter(|style| style.range[0] < style.range[1])
            .cloned()
            .collect();
        if !kept.is_empty() || length == 0 {
            self.styles = kept;
        } else if let Some(first) = self.styles.first_mut() {
            first.range = [0, length];
            self.styles.truncate(1);
        }
    }
}

impl MaterialText {
    /// The words and styles, when the title is rich and `content` reads as
    /// rich text.
    #[must_use]
    pub fn rich(&self) -> Option<RichText> {
        if !self.is_rich_text {
            return None;
        }
        serde_json::from_str(&self.content).ok()
    }

    /// The words to show: a rich title's text, else `content`. A rich title
    /// that does not read as rich text has none.
    #[must_use]
    pub fn words(&self) -> Option<Cow<'_, str>> {
        if !self.is_rich_text {
            return Some(Cow::Borrowed(&self.content));
        }
        self.rich().map(|rich| Cow::Owned(rich.text))
    }

    /// Replace the words, keeping a rich title's styles. A rich title that
    /// does not read as rich text becomes plain.
    pub fn set_words(&mut self, words: &str) {
        match self.rich() {
            Some(mut rich) => {
                rich.set_words(words);
                self.set_rich(&rich);
            }
            None => {
                self.content = words.to_owned();
                self.is_rich_text = false;
            }
        }
    }

    /// Paint every range of a rich title solid `rgb`, each from 0 to 1.
    pub fn paint_rich(&mut self, rgb: [f64; 3]) {
        if let Some(mut rich) = self.rich() {
            for style in &mut rich.styles {
                style.paint(rgb);
            }
            self.set_rich(&rich);
        }
    }

    fn set_rich(&mut self, rich: &RichText) {
        if let Ok(content) = serde_json::to_string(rich) {
            self.content = content;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STORED: &str = r#"{"text":"Hi you","styles":[{"range":[0,2],"fill":{"alpha":1.0,"content":{"render_type":"solid","solid":{"alpha":1.0,"color":[1.0,0.0,0.0]}}},"size":5.0},{"range":[2,6],"bold":true}],"extra":1}"#;

    fn rich() -> MaterialText {
        let mut text = MaterialText::new("t");
        text.is_rich_text = true;
        text.content = STORED.to_owned();
        text
    }

    #[test]
    fn a_rich_title_reads_its_words_and_ranges_and_keeps_the_rest() {
        let text = rich();
        let read = text.rich().unwrap();
        assert_eq!(text.words().unwrap(), "Hi you");
        assert_eq!(read.styles[0].solid(), Some([1.0, 0.0, 0.0, 1.0]));
        assert_eq!(read.styles[1].solid(), None);
        let back: Value = serde_json::to_value(&read).unwrap();
        assert_eq!(back, serde_json::from_str::<Value>(STORED).unwrap());

        let mut plain = MaterialText::new("p");
        plain.content = "{\"text\":\"x\"}".to_owned();
        assert_eq!(plain.words().unwrap(), "{\"text\":\"x\"}");
        let mut broken = rich();
        broken.content = "Hi".to_owned();
        assert!(broken.words().is_none());
        broken.set_words("Hello");
        assert_eq!(
            (broken.content.as_str(), broken.is_rich_text),
            ("Hello", false)
        );
    }

    #[test]
    fn a_range_reads_its_size_strokes_and_shadows() {
        let style: RichStyle = serde_json::from_str(
            r#"{"range":[0,1],"size":8,"strokes":[{"content":{"solid":{"color":[0,0,0]}},"width":0.08}],"shadows":[{"alpha":0.5,"content":{"solid":{"alpha":1,"color":[0,0,1]},"render_type":"solid"},"distance":8,"angle":-45,"feather":0.33}]}"#,
        )
        .unwrap();
        assert_eq!(style.size, Some(8.0));
        let stroke = &style.strokes.as_ref().unwrap()[0];
        assert_eq!(
            (stroke.width, stroke.rgba()),
            (0.08, Some([0.0, 0.0, 0.0, 1.0]))
        );
        let shadow = &style.shadows.as_ref().unwrap()[0];
        assert_eq!(shadow.rgba(), Some([0.0, 0.0, 1.0, 0.5]));
        assert_eq!(
            (shadow.distance, shadow.angle, shadow.feather),
            (8.0, -45.0, 0.33)
        );
        let none: RichStyle = serde_json::from_str(r#"{"range":[0,1],"strokes":[]}"#).unwrap();
        assert_eq!(none.strokes, Some(Vec::new()));
        assert_eq!(none.shadows, None);
    }

    #[test]
    fn new_words_keep_the_ranges_and_a_color_paints_them_all() {
        let mut text = rich();
        text.set_words("Hi there friend");
        let read = text.rich().unwrap();
        assert_eq!(read.text, "Hi there friend");
        assert_eq!(read.styles[0].range, [0, 2]);
        assert_eq!(read.styles[1].range, [2, 15]);
        assert_eq!(read.rest.get("extra"), Some(&Value::from(1)));

        text.set_words("H");
        let read = text.rich().unwrap();
        assert_eq!(read.styles.len(), 1);
        assert_eq!(read.styles[0].range, [0, 1]);
        text.set_words("Hé!");
        assert_eq!(text.rich().unwrap().styles[0].range, [0, 3]);

        text.paint_rich([0.0, 0.0, 1.0]);
        let read = text.rich().unwrap();
        assert!(
            read.styles
                .iter()
                .all(|style| style.solid() == Some([0.0, 0.0, 1.0, 1.0]))
        );
        assert_eq!(read.styles[0].size, Some(5.0));
    }
}
