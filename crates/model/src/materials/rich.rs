//! Rich titles: `content` holds JSON with the words and styled ranges.
//!
//! `{"text": "...", "styles": [{"range": [start, end], "fill": {...}, ...}]}`.
//! A range counts characters, start included and end left out. A fill's
//! solid color is three channels from 0 to 1 with an alpha from 0 to 1.
//! `size` is in the title's `font_size` units. A stroke's `width` is in ems,
//! like `border_width`; a shadow's `distance`, `angle`, and `feather` read
//! like `shadow_distance`, `shadow_angle`, and `shadow_smoothing`. A range
//! with no `strokes` or `shadows` key takes the title's own; an empty list
//! has none. A `font`'s `path` names the face file its letters are set
//! in. Keys this model does not name stay as they were.

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
    pub font: Option<RichFont>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strokes: Option<Vec<RichStroke>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadows: Option<Vec<RichShadow>>,
    /// The range's letters are bold, or not, whatever the title is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bold: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub italic: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub underline: Option<bool>,
    #[serde(flatten, default)]
    pub rest: Map<String, Value>,
}

/// The face a range's letters are set in.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct RichFont {
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub id: String,
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

/// Which of bold, italic and underline a range of letters has.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RichPattern {
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
}

/// A change to the pattern of a range: each of the three is set, or left
/// as it is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PatternChange {
    pub bold: Option<bool>,
    pub italic: Option<bool>,
    pub underline: Option<bool>,
}

impl RichStyle {
    /// Whether two styles say the same of their letters, wherever the
    /// letters are.
    fn same_look(&self, other: &Self) -> bool {
        let mut other = other.clone();
        other.range = self.range;
        *self == other
    }
}

/// The most cells of the table that lines the old words up with the new.
/// Past it, everything between the shared start and end counts as typed
/// over.
const MAX_ALIGN: usize = 1 << 20;

/// For each letter of `new`, the letter of `old` whose style it takes, by
/// place: itself where the two share it, the first letter it was typed
/// over, or else the letter before it. The places never go down.
fn sources(old: &[char], new: &[char]) -> Vec<usize> {
    let head = old
        .iter()
        .zip(new)
        .take_while(|(before, after)| before == after)
        .count();
    let tail = old[head..]
        .iter()
        .rev()
        .zip(new[head..].iter().rev())
        .take_while(|(before, after)| before == after)
        .count();
    let (gone, come) = (&old[head..old.len() - tail], &new[head..new.len() - tail]);
    // The letters of the two middles that pair up, in order: the longest
    // run of letters they share.
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let (rows, columns) = (gone.len(), come.len());
    if rows > 0 && columns > 0 && rows.saturating_mul(columns) <= MAX_ALIGN {
        let width = columns + 1;
        let mut longest = vec![0_u32; (rows + 1) * width];
        for row in (0..rows).rev() {
            for column in (0..columns).rev() {
                longest[row * width + column] = if gone[row] == come[column] {
                    longest[(row + 1) * width + column + 1] + 1
                } else {
                    longest[(row + 1) * width + column].max(longest[row * width + column + 1])
                };
            }
        }
        let (mut row, mut column) = (0, 0);
        while row < rows && column < columns {
            if gone[row] == come[column] {
                pairs.push((row, column));
                row += 1;
                column += 1;
            } else if longest[(row + 1) * width + column] >= longest[row * width + column + 1] {
                row += 1;
            } else {
                column += 1;
            }
        }
    }
    let mut sources: Vec<usize> = (0..head).collect();
    // Between two paired letters: new letters typed over old ones take the
    // first of those, and new letters typed in take the letter before.
    let (mut row, mut column) = (0, 0);
    for (pair_row, pair_column) in pairs.into_iter().chain([(rows, columns)]) {
        let source = if pair_row > row {
            head + row
        } else {
            (head + row).saturating_sub(1)
        };
        sources.extend(std::iter::repeat_n(source, pair_column - column));
        if pair_row < rows {
            sources.push(head + pair_row);
        }
        (row, column) = (pair_row + 1, pair_column + 1);
    }
    sources.extend(old.len() - tail..old.len());
    sources
}

impl RichText {
    /// Replace the words. The ranges follow the words: the letters the old
    /// and the new words share keep their ranges, letters typed in take
    /// the style of the letter before them, and letters typed over others
    /// take the style of what they replaced.
    pub fn set_words(&mut self, words: &str) {
        let old: Vec<char> = self.text.chars().collect();
        let new: Vec<char> = words.chars().collect();
        let length = new.len();
        // Whether the ranges covered every letter, as a title made with
        // colors and sizes for all its letters does.
        let whole = {
            let mut ranges: Vec<[usize; 2]> = self.styles.iter().map(|style| style.range).collect();
            ranges.sort_unstable();
            let mut at = 0;
            for [from, to] in ranges {
                if from > at {
                    break;
                }
                at = at.max(to);
            }
            at >= old.len()
        };
        let sources = sources(&old, &new);
        // The first new letter that came from old letter `place` or later.
        let first_from = |place: usize| sources.partition_point(|source| *source < place);
        self.text = words.to_owned();
        for style in &mut self.styles {
            let [start, end] = style.range;
            style.range = [first_from(start), first_from(end.max(start))];
        }
        let kept: Vec<_> = self
            .styles
            .iter()
            .filter(|style| style.range[0] < style.range[1])
            .cloned()
            .collect();
        if !kept.is_empty() || length == 0 || !whole {
            self.styles = kept;
        } else if let Some(first) = self.styles.first_mut() {
            // Every letter had a style and none is left: the new letters
            // take the first.
            first.range = [0, length];
            self.styles.truncate(1);
        }
    }

    /// Change the pattern of the letters from `start` up to `end`, which
    /// count characters. A range that reaches past them is cut in two or
    /// three, and letters no range covers get one. Nothing changes for an
    /// empty stretch or one past the words.
    pub fn set_pattern(&mut self, start: usize, end: usize, change: PatternChange) {
        let length = self.text.chars().count();
        let (start, end) = (start.min(length), end.min(length));
        if start >= end {
            return;
        }
        let apply = |style: &mut RichStyle| {
            style.bold = change.bold.or(style.bold);
            style.italic = change.italic.or(style.italic);
            style.underline = change.underline.or(style.underline);
        };
        let mut pieces: Vec<RichStyle> = Vec::with_capacity(self.styles.len() + 2);
        // The parts of the stretch that a range covers, to find the rest.
        let mut covered: Vec<[usize; 2]> = Vec::new();
        for style in &self.styles {
            let [from, to] = style.range;
            let piece = |range: [usize; 2]| {
                let mut piece = style.clone();
                piece.range = range;
                piece
            };
            if from < to.min(start) {
                pieces.push(piece([from, to.min(start)]));
            }
            let inside = [from.max(start), to.min(end)];
            if inside[0] < inside[1] {
                let mut changed = piece(inside);
                apply(&mut changed);
                pieces.push(changed);
                covered.push(inside);
            }
            if from.max(end) < to {
                pieces.push(piece([from.max(end), to]));
            }
        }
        covered.sort_unstable();
        let mut at = start;
        for [from, to] in covered.into_iter().chain([[end, end]]) {
            if at < from {
                let mut bare = RichStyle {
                    range: [at, from],
                    ..RichStyle::default()
                };
                apply(&mut bare);
                pieces.push(bare);
            }
            at = at.max(to);
        }
        pieces.sort_by_key(|style| style.range);
        // Neighbours that now say the same are one range again.
        let mut joined: Vec<RichStyle> = Vec::with_capacity(pieces.len());
        for piece in pieces {
            match joined.last_mut() {
                Some(last) if last.range[1] == piece.range[0] && last.same_look(&piece) => {
                    last.range[1] = piece.range[1];
                }
                _ => joined.push(piece),
            }
        }
        self.styles = joined;
    }

    /// The pattern every letter from `start` up to `end` has, where a
    /// letter with nothing said of it has the title's `own`. An empty
    /// stretch has the title's own.
    #[must_use]
    pub fn pattern(&self, start: usize, end: usize, own: RichPattern) -> RichPattern {
        let length = self.text.chars().count();
        let (start, end) = (start.min(length), end.min(length));
        if start >= end {
            return own;
        }
        let mut all = RichPattern {
            bold: true,
            italic: true,
            underline: true,
        };
        let mut at = start;
        let mut ranges: Vec<&RichStyle> = self
            .styles
            .iter()
            .filter(|style| style.range[0] < end && style.range[1] > start)
            .collect();
        ranges.sort_by_key(|style| style.range);
        let mut take = |pattern: RichPattern| {
            all.bold &= pattern.bold;
            all.italic &= pattern.italic;
            all.underline &= pattern.underline;
        };
        for style in ranges {
            // Letters before this range that no range covers.
            if at < style.range[0] {
                take(own);
            }
            take(RichPattern {
                bold: style.bold.unwrap_or(own.bold),
                italic: style.italic.unwrap_or(own.italic),
                underline: style.underline.unwrap_or(own.underline),
            });
            at = at.max(style.range[1]);
        }
        if at < end {
            take(own);
        }
        all
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

    /// Change the pattern of the letters from `start` up to `end`, which
    /// count characters of the words. A plain title becomes a rich one
    /// that holds the range. `false` when the title has no such letters,
    /// or is a rich title that does not read.
    pub fn set_range_pattern(&mut self, start: usize, end: usize, change: PatternChange) -> bool {
        let mut rich = match self.rich() {
            Some(rich) => rich,
            None if self.is_rich_text => return false,
            None => RichText {
                text: self.content.clone(),
                ..RichText::default()
            },
        };
        if start >= end || end > rich.text.chars().count() {
            return false;
        }
        rich.set_pattern(start, end, change);
        self.is_rich_text = true;
        self.set_rich(&rich);
        true
    }

    /// The pattern every letter from `start` up to `end` has. `own` is
    /// the title's own pattern, which a plain title's letters all have.
    #[must_use]
    pub fn range_pattern(&self, start: usize, end: usize, own: RichPattern) -> RichPattern {
        self.rich()
            .map_or(own, |rich| rich.pattern(start, end, own))
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
            r#"{"range":[0,1],"size":8,"font":{"path":"/fonts/Bold.ttf","id":"7"},"strokes":[{"content":{"solid":{"color":[0,0,0]}},"width":0.08}],"shadows":[{"alpha":0.5,"content":{"solid":{"alpha":1,"color":[0,0,1]},"render_type":"solid"},"distance":8,"angle":-45,"feather":0.33}]}"#,
        )
        .unwrap();
        assert_eq!(style.size, Some(8.0));
        assert_eq!(style.font.as_ref().unwrap().path, "/fonts/Bold.ttf");
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

    /// The ranges of a rich title and what each says of bold.
    fn bolds(text: &MaterialText) -> Vec<([usize; 2], Option<bool>)> {
        text.rich()
            .unwrap()
            .styles
            .iter()
            .map(|style| (style.range, style.bold))
            .collect()
    }

    const BOLD: PatternChange = PatternChange {
        bold: Some(true),
        italic: None,
        underline: None,
    };

    #[test]
    fn a_stretch_of_letters_is_made_bold_and_its_range_is_cut_out() {
        // A plain title becomes a rich one of one range.
        let mut text = MaterialText::new("t");
        text.content = "Hello world".to_owned();
        assert!(text.set_range_pattern(6, 11, BOLD));
        assert!(text.is_rich_text);
        assert_eq!(text.words().unwrap(), "Hello world");
        assert_eq!(bolds(&text), [([6, 11], Some(true))]);
        // A stretch inside it is cut out of it, with what else it says.
        let plain = PatternChange {
            bold: Some(false),
            italic: Some(true),
            underline: None,
        };
        assert!(text.set_range_pattern(8, 9, plain));
        assert_eq!(
            bolds(&text),
            [
                ([6, 8], Some(true)),
                ([8, 9], Some(false)),
                ([9, 11], Some(true))
            ]
        );
        let read = text.rich().unwrap();
        assert_eq!(read.styles[1].italic, Some(true));
        assert_eq!(read.styles[0].italic, None);
        // A stretch over ranges and bare letters changes them all, and
        // neighbours that now say the same are one range.
        let all = PatternChange {
            bold: Some(true),
            italic: Some(true),
            underline: None,
        };
        assert!(text.set_range_pattern(0, 11, all));
        assert_eq!(bolds(&text), [([0, 11], Some(true))]);
        assert_eq!(text.rich().unwrap().styles[0].italic, Some(true));
        // A range keeps its color and size when a part of it changes.
        let mut colored = rich();
        assert!(colored.set_range_pattern(1, 4, BOLD));
        let read = colored.rich().unwrap();
        let ranges: Vec<[usize; 2]> = read.styles.iter().map(|style| style.range).collect();
        assert_eq!(ranges, [[0, 1], [1, 2], [2, 6]]);
        assert_eq!(read.styles[0].size, Some(5.0));
        assert_eq!(read.styles[1].size, Some(5.0));
        assert_eq!(read.styles[1].solid(), Some([1.0, 0.0, 0.0, 1.0]));
        assert_eq!(read.styles[1].bold, Some(true));
        assert_eq!(read.styles[0].bold, None);
        // The range that was bold already takes the rest in.
        assert_eq!(read.styles[2].bold, Some(true));
        assert_eq!(read.rest.get("extra"), Some(&Value::from(1)));
    }

    #[test]
    fn a_stretch_that_is_empty_or_past_the_words_changes_nothing() {
        let mut text = MaterialText::new("t");
        text.content = "Hé!".to_owned();
        for (start, end) in [(1, 1), (2, 1), (0, 4), (3, 9), (usize::MAX, usize::MAX)] {
            assert!(!text.set_range_pattern(start, end, BOLD), "{start} {end}");
            assert!(!text.is_rich_text);
            assert_eq!(text.content, "Hé!");
        }
        // Characters are counted, not bytes: the last is the third.
        assert!(text.set_range_pattern(2, 3, BOLD));
        assert_eq!(bolds(&text), [([2, 3], Some(true))]);
        // A rich title that does not read is left as it is.
        let mut broken = rich();
        broken.content = "not rich".to_owned();
        assert!(!broken.set_range_pattern(0, 2, BOLD));
        assert_eq!(broken.content, "not rich");
        // A range written past the words, or backwards, is no trouble.
        let mut odd = rich();
        odd.content =
            r#"{"text":"abc","styles":[{"range":[2,900]},{"range":[5,1],"bold":true}]}"#.to_owned();
        assert!(odd.set_range_pattern(0, 3, BOLD));
        assert!(
            odd.rich()
                .unwrap()
                .styles
                .iter()
                .all(|style| style.range[0] <= style.range[1])
        );
    }

    #[test]
    fn a_stretch_has_a_pattern_when_every_letter_of_it_has() {
        let own = RichPattern::default();
        let mut text = MaterialText::new("t");
        text.content = "Hello world".to_owned();
        // A plain title's letters all have the title's own.
        assert_eq!(text.range_pattern(0, 5, own), own);
        let underlined = RichPattern {
            underline: true,
            ..own
        };
        assert_eq!(text.range_pattern(0, 5, underlined), underlined);
        assert!(text.set_range_pattern(0, 5, BOLD));
        assert!(text.range_pattern(0, 5, own).bold);
        assert!(text.range_pattern(1, 3, own).bold);
        // One letter outside the bold range and the stretch is not bold.
        assert!(!text.range_pattern(0, 6, own).bold);
        assert!(!text.range_pattern(5, 11, own).bold);
        // A bold title is bold where no range says otherwise.
        let bold = RichPattern { bold: true, ..own };
        assert!(text.range_pattern(0, 11, bold).bold);
        let off = PatternChange {
            bold: Some(false),
            ..PatternChange::default()
        };
        assert!(text.set_range_pattern(7, 8, off));
        assert!(!text.range_pattern(0, 11, bold).bold);
        assert!(text.range_pattern(8, 11, bold).bold);
        // An empty stretch has the title's own.
        assert_eq!(text.range_pattern(4, 4, bold), bold);
    }

    #[test]
    fn ranges_follow_their_letters_when_words_are_typed_or_taken_out() {
        let mut text = MaterialText::new("t");
        text.content = "one two three".to_owned();
        assert!(text.set_range_pattern(4, 7, BOLD));
        // Typed before the range: it moves with its letters.
        text.set_words("zero one two three");
        assert_eq!(bolds(&text), [([9, 12], Some(true))]);
        // Typed inside it: it grows. Typed at its end: it takes them.
        text.set_words("zero one twooo three");
        assert_eq!(bolds(&text), [([9, 14], Some(true))]);
        // Typed after it, past a space: it stays.
        text.set_words("zero one twooo three four");
        assert_eq!(bolds(&text), [([9, 14], Some(true))]);
        // Letters taken out before it and inside it.
        text.set_words("one too three four");
        assert_eq!(bolds(&text), [([4, 7], Some(true))]);
        // A stretch over its start replaced: the new letters take the
        // style of what they replaced, and the range keeps its tail.
        text.set_words("on! oo three four");
        assert_eq!(text.words().unwrap(), "on! oo three four");
        let ranges = bolds(&text);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].0[1], 6);
        // The range taken out whole is gone, and the title has none.
        text.set_words("three four");
        assert!(text.rich().unwrap().styles.is_empty() || bolds(&text)[0].0 == [0, 10]);
        // Many edits of a long title are quick.
        let mut long = MaterialText::new("t");
        long.content = "word ".repeat(2_000);
        for start in (0..9_000).step_by(90) {
            assert!(long.set_range_pattern(start, start + 4, BOLD));
        }
        let started = std::time::Instant::now();
        for round in 0..50 {
            let words = format!("{}{}", "x".repeat(round), "word ".repeat(2_000));
            long.set_words(&words);
        }
        assert_eq!(long.rich().unwrap().styles.len(), 100);
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
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
